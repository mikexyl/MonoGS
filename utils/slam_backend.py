import random
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.general_utils import build_scaling_rotation
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping


class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        self.initialized_at_iteration = 0 if self.initialized else None
        self.last_structural_phase = None

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.use_structural_commitment = self.config["Training"].get(
            "use_structural_commitment", False
        )
        self.commitment_alpha = self.config["Training"].get("commitment_alpha", 0.05)
        self.commitment_knn = self.config["Training"].get("commitment_knn", 8)
        self.commitment_stable_quantile = self.config["Training"].get(
            "commitment_stable_quantile", 0.75
        )
        self.commitment_obs_weight = self.config["Training"].get(
            "commitment_obs_weight", 0.5
        )
        self.lambda_coh = self.config["Training"].get("lambda_coh", 0.0)
        self.lambda_anchor = self.config["Training"].get("lambda_anchor", 0.5)
        self.lambda_thin = self.config["Training"].get("lambda_thin", 0.0)
        self.commitment_prune_bias = self.config["Training"].get(
            "commitment_prune_bias", 0.0
        )
        self.commitment_protect_threshold = self.config["Training"].get(
            "commitment_protect_threshold", 0.85
        )
        self.commitment_chunk_size = self.config["Training"].get(
            "commitment_chunk_size", 256
        )
        self.commitment_max_points = self.config["Training"].get(
            "commitment_max_points", 2048
        )
        self.commitment_log_every = self.config["Training"].get(
            "commitment_log_every", 50
        )
        self.commitment_start_after_init_iters = self.config["Training"].get(
            "commitment_start_after_init_iters", 100
        )
        self.commitment_ramp_iters = self.config["Training"].get(
            "commitment_ramp_iters", 200
        )
        self.commitment_anchor_alpha = self.config["Training"].get(
            "commitment_anchor_alpha", 0.01
        )
        self.commitment_field_smoothing = self.config["Training"].get(
            "commitment_field_smoothing", 0.75
        )
        self.commitment_motion_damping = self.config["Training"].get(
            "commitment_motion_damping", 0.75
        )
        self.commitment_damping_threshold = self.config["Training"].get(
            "commitment_damping_threshold", 0.6
        )
        self.commitment_damping_quantile = self.config["Training"].get(
            "commitment_damping_quantile", 0.9
        )
        self.commitment_damping_power = self.config["Training"].get(
            "commitment_damping_power", 2.0
        )
        self.map_timing_log_every = self.config["Training"].get(
            "map_timing_log_every", 0
        )
        self.map_slow_iteration_ms = self.config["Training"].get(
            "map_slow_iteration_ms", 250.0
        )
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )

    def _get_xyz_lr(self):
        for param_group in self.gaussians.optimizer.param_groups:
            if param_group["name"] == "xyz":
                return param_group["lr"]
        return 0.0

    def _get_active_gaussian_mask(self, visibility_filters):
        num_points = self.gaussians.get_xyz.shape[0]
        if num_points == 0:
            return torch.zeros(0, dtype=torch.bool, device="cuda")

        active_mask = torch.zeros(num_points, dtype=torch.bool, device="cuda")
        for visibility_filter in visibility_filters:
            active_mask |= visibility_filter
        return active_mask

    def _get_structural_phase(self):
        if not self.use_structural_commitment or self.gaussians is None:
            return 0.0, "disabled", 0
        if self.gaussians.get_xyz.shape[0] == 0:
            return 0.0, "empty", 0
        if self.initialized_at_iteration is None:
            return 0.0, "warmup", 0

        live_iters = max(0, self.iteration_count - self.initialized_at_iteration)
        if live_iters < self.commitment_start_after_init_iters:
            return 0.0, "delay", live_iters

        if self.commitment_ramp_iters <= 0:
            return 1.0, "live", live_iters

        ramp_iters = live_iters - self.commitment_start_after_init_iters + 1
        structural_scale = min(1.0, ramp_iters / self.commitment_ramp_iters)
        structural_phase = "ramp" if structural_scale < 1.0 else "live"
        return structural_scale, structural_phase, live_iters

    def _apply_structural_motion_damping(self, structural_scale, debug_stats):
        if (
            structural_scale <= 0.0
            or self.commitment_motion_damping <= 0.0
            or self.gaussians is None
            or self.gaussians._xyz.grad is None
            or debug_stats is None
            or "regularizer_idx" not in debug_stats
            or "field_commitment" not in debug_stats
        ):
            return None

        damping_idx = debug_stats["regularizer_idx"]
        field_commitment = debug_stats["field_commitment"]
        if damping_idx.numel() == 0 or field_commitment.numel() == 0:
            return None

        commitment = field_commitment.detach().view(-1)
        threshold = torch.clamp(
            torch.tensor(
                self.commitment_damping_threshold,
                device=commitment.device,
                dtype=commitment.dtype,
            ),
            0.0,
            1.0 - 1e-6,
        )
        damping_quantile = torch.clamp(
            torch.tensor(
                self.commitment_damping_quantile,
                device=commitment.device,
                dtype=commitment.dtype,
            ),
            0.0,
            1.0,
        )
        if commitment.numel() > 1 and damping_quantile.item() < 1.0:
            adaptive_threshold = torch.quantile(commitment, damping_quantile)
            adaptive_threshold = adaptive_threshold.clamp_(0.1, 1.0 - 1e-6)
            threshold = torch.minimum(threshold, adaptive_threshold)
        anchor_strength = ((commitment - threshold) / (1.0 - threshold)).clamp_(0.0, 1.0)
        if not anchor_strength.any():
            return {
                "damping_mean": commitment.new_tensor(0.0),
                "damping_max": commitment.new_tensor(0.0),
                "damped_points": 0,
                "damping_threshold": threshold.detach(),
            }

        damping_strength = (
            structural_scale
            * self.commitment_motion_damping
            * anchor_strength.pow(self.commitment_damping_power)
        ).clamp_(0.0, 1.0)
        damped_grad = self.gaussians._xyz.grad.index_select(0, damping_idx) * (
            1.0 - damping_strength
        ).unsqueeze(-1)
        self.gaussians._xyz.grad.index_copy_(0, damping_idx, damped_grad)
        return {
            "damping_mean": damping_strength.mean().detach(),
            "damping_max": damping_strength.max().detach(),
            "damped_points": int((anchor_strength > 0).sum().item()),
            "damping_threshold": threshold.detach(),
        }

    def _maybe_log_structural_phase(self, structural_phase, structural_scale, live_iters):
        if structural_phase == self.last_structural_phase:
            return

        self.last_structural_phase = structural_phase
        Log(
            "Structural phase",
            f"iter={self.iteration_count}",
            f"phase={structural_phase}",
            f"scale={structural_scale:.3f}",
            f"live_iters={live_iters}",
            f"start_after={self.commitment_start_after_init_iters}",
            f"ramp_iters={self.commitment_ramp_iters}",
        )

    def _should_log_map_timing(
        self, iter_ms, prune, update_gaussian, opacity_reset, structural_phase
    ):
        if (
            self.map_timing_log_every > 0
            and self.iteration_count % self.map_timing_log_every == 0
        ):
            return True
        if prune or update_gaussian or opacity_reset:
            return True
        if iter_ms >= self.map_slow_iteration_ms:
            return True
        if structural_phase in {"delay", "ramp", "live"} and self.last_sent <= 1:
            return True
        return False

    def _log_map_timing(
        self,
        prune,
        structural_phase,
        structural_scale,
        live_iters,
        render_ms,
        structural_ms,
        densify_ms,
        opacity_reset_ms,
        optimizer_ms,
        iter_ms,
        update_gaussian,
        opacity_reset,
        point_count_before,
        point_count_after,
        active_count,
        regularizer_count,
    ):
        if not self._should_log_map_timing(
            iter_ms, prune, update_gaussian, opacity_reset, structural_phase
        ):
            return

        event_tokens = []
        if prune:
            event_tokens.append("prune")
        if update_gaussian:
            event_tokens.append("densify")
        if opacity_reset:
            event_tokens.append("opacity_reset")
        if not event_tokens:
            event_tokens.append("iteration")

        Log(
            "Map timing",
            f"iter={self.iteration_count}",
            f"events={'+'.join(event_tokens)}",
            f"phase={structural_phase}",
            f"scale={structural_scale:.3f}",
            f"live_iters={live_iters}",
            f"window={len(self.current_window)}",
            f"points={point_count_before}->{point_count_after}",
            f"active={active_count}",
            f"regularizer={regularizer_count}",
            f"render_ms={render_ms:.1f}",
            f"struct_ms={structural_ms:.1f}",
            f"densify_ms={densify_ms:.1f}",
            f"opacity_ms={opacity_reset_ms:.1f}",
            f"optim_ms={optimizer_ms:.1f}",
            f"iter_ms={iter_ms:.1f}",
        )

    def _build_local_knn(self, xyz):
        num_points = xyz.shape[0]
        if num_points <= 1:
            return None, None

        knn = min(self.commitment_knn, num_points - 1)
        if knn <= 0:
            return None, None

        point_ids = torch.arange(num_points, device=xyz.device)
        chunk_size = min(self.commitment_chunk_size, num_points)
        min_chunk_size = max(knn + 1, 64)

        while chunk_size >= min_chunk_size:
            neighbors = []
            neighbor_dist2 = []
            try:
                for start in range(0, num_points, chunk_size):
                    end = min(start + chunk_size, num_points)
                    dist = torch.cdist(xyz[start:end], xyz)
                    dist[
                        torch.arange(end - start, device=xyz.device), point_ids[start:end]
                    ] = float("inf")
                    knn_dist, knn_idx = torch.topk(dist, k=knn, dim=1, largest=False)
                    neighbors.append(knn_idx)
                    neighbor_dist2.append(knn_dist.pow(2))
                return torch.cat(neighbors, dim=0), torch.cat(neighbor_dist2, dim=0)
            except torch.OutOfMemoryError:
                neighbors.clear()
                neighbor_dist2.clear()
                torch.cuda.empty_cache()
                next_chunk_size = chunk_size // 2
                if next_chunk_size < min_chunk_size:
                    raise
                Log(
                    "Structural commitment",
                    f"Reducing kNN chunk size from {chunk_size} to {next_chunk_size}",
                )
                chunk_size = next_chunk_size

        return None, None

    def _get_commitment_proposals(self, grad_norm, observation_support=None):
        if grad_norm.numel() == 0:
            return grad_norm
        if grad_norm.numel() == 1:
            return torch.ones_like(grad_norm)

        sorted_idx = torch.argsort(grad_norm)
        ranks = torch.zeros_like(grad_norm)
        ranks[sorted_idx] = torch.linspace(
            0.0,
            1.0,
            steps=grad_norm.numel(),
            device=grad_norm.device,
            dtype=grad_norm.dtype,
        )
        proposals = 1.0 - ranks
        stable_quantile = torch.clamp(
            torch.tensor(
                self.commitment_stable_quantile,
                device=grad_norm.device,
                dtype=grad_norm.dtype,
            ),
            0.0,
            1.0 - 1e-6,
        )
        # Only the stable tail accumulates commitment; the rest decay toward zero.
        proposals = (proposals - stable_quantile) / (1.0 - stable_quantile)
        proposals = proposals.clamp_(0.0, 1.0)
        if observation_support is not None and self.commitment_obs_weight > 0.0:
            support = observation_support.clamp_(0.0, 1.0)
            support_weight = torch.clamp(
                torch.tensor(
                    self.commitment_obs_weight,
                    device=grad_norm.device,
                    dtype=grad_norm.dtype,
                ),
                0.0,
                1.0,
            )
            proposals = proposals * ((1.0 - support_weight) + support_weight * support)
        return proposals.clamp_(0.0, 1.0)

    def _get_commitment_field(self, commitment, neighbor_idx, kernel):
        if commitment.numel() == 0 or neighbor_idx is None:
            return commitment.detach()

        blend = torch.clamp(
            torch.tensor(
                self.commitment_field_smoothing,
                device=commitment.device,
                dtype=commitment.dtype,
            ),
            0.0,
            1.0,
        )
        if blend.item() <= 0.0:
            return commitment.detach()

        neighbor_commitment = commitment[neighbor_idx].squeeze(-1)
        neighbor_mean = (
            kernel * neighbor_commitment
        ).sum(dim=1, keepdim=True) / kernel.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (
            (1.0 - blend) * commitment.detach() + blend * neighbor_mean.detach()
        ).clamp_(0.0, 1.0)

    def _compute_structural_commitment_terms(self, photo_xyz_grad, visibility_filters):
        zero = self.gaussians.get_xyz.new_tensor(0.0)
        empty_idx = torch.empty((0,), dtype=torch.long, device=self.gaussians.get_xyz.device)
        active_mask = self._get_active_gaussian_mask(visibility_filters)
        if photo_xyz_grad is None or active_mask.numel() == 0 or not active_mask.any():
            return (
                zero,
                zero,
                zero,
                active_mask,
                self.gaussians.get_xyz.new_empty((0,)),
                0,
                {"regularizer_idx": empty_idx, "field_commitment": self.gaussians.get_xyz.new_empty((0,))},
            )

        active_idx = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
        active_grad = photo_xyz_grad[active_idx]
        grad_norm = torch.norm(active_grad.detach(), dim=1)
        observation_support = None
        if self.commitment_obs_weight > 0.0:
            observation_support = (
                self.gaussians.n_obs[active_idx.cpu()]
                .to(device=grad_norm.device, dtype=grad_norm.dtype)
                .clamp_(min=0)
                / max(float(self.window_size), 1.0)
            )
            observation_support = observation_support.clamp_(0.0, 1.0)
        commitment_proposals = self._get_commitment_proposals(
            grad_norm, observation_support
        )

        if active_idx.numel() <= 1 or (
            self.lambda_coh <= 0.0
            and self.lambda_anchor <= 0.0
            and self.lambda_thin <= 0.0
        ):
            return zero, zero, zero, active_mask, commitment_proposals, 0, None

        regularizer_idx = active_idx
        if (
            self.commitment_max_points > 0
            and active_idx.numel() > self.commitment_max_points
        ):
            sample_idx = torch.randperm(active_idx.numel(), device=active_idx.device)[
                : self.commitment_max_points
            ]
            regularizer_idx = active_idx[sample_idx]

        regularizer_count = int(regularizer_idx.numel())
        if regularizer_count <= 1:
            return (
                zero,
                zero,
                zero,
                active_mask,
                commitment_proposals,
                regularizer_count,
                None,
            )

        xyz = self.gaussians.get_xyz[regularizer_idx]
        commitment = self.gaussians.get_structural_commitment[regularizer_idx]
        anchor_xyz = self.gaussians.get_structural_anchor_xyz[regularizer_idx]
        neighbor_idx, neighbor_dist2 = self._build_local_knn(xyz)
        if neighbor_idx is None:
            return (
                zero,
                zero,
                zero,
                active_mask,
                commitment_proposals,
                regularizer_count,
                None,
            )

        disp = xyz[neighbor_idx] - xyz.unsqueeze(1)
        local_scale = neighbor_dist2[:, -1:].clamp_min(1e-6)
        kernel = torch.exp(-neighbor_dist2 / local_scale)
        field_commitment = self._get_commitment_field(commitment, neighbor_idx, kernel)
        field_commitment_detached = field_commitment.detach()

        sampled_grad = photo_xyz_grad[regularizer_idx]
        xyz_lr = max(self._get_xyz_lr(), 1e-6)
        photo_delta = (-xyz_lr * sampled_grad).detach()
        neighbor_commitment = field_commitment_detached[neighbor_idx].squeeze(-1)
        neighbor_delta = photo_delta[neighbor_idx]
        self_commitment = field_commitment_detached.squeeze(-1)
        consensus_weights = kernel * neighbor_commitment
        weights = torch.cat((self_commitment.unsqueeze(1), consensus_weights), dim=1)
        delta_bank = torch.cat((photo_delta.unsqueeze(1), neighbor_delta), dim=1)
        raw_denom = weights.sum(dim=1, keepdim=True)
        denom = raw_denom.clamp_min(1e-6)
        consensus_delta = (
            weights.unsqueeze(-1) * delta_bank
        ).sum(dim=1) / denom
        fallback = raw_denom.squeeze(-1) <= 1e-6
        if fallback.any():
            consensus_delta[fallback] = photo_delta[fallback]

        effective_delta = (1.0 - field_commitment_detached) * photo_delta + (
            field_commitment_detached * consensus_delta.detach()
        )
        coherence_target = xyz.detach() + effective_delta
        coherence_residual = xyz - coherence_target
        loss_coh = (
            field_commitment_detached
            * coherence_residual.pow(2).sum(dim=1, keepdim=True)
        ).mean() / xyz_lr

        scaling = self.gaussians.get_scaling[regularizer_idx]
        if scaling.shape[-1] == 1:
            scaling = scaling.repeat(1, 3)
        anchor_scale = local_scale.sqrt().detach().clamp_min(1e-3)
        anchor_residual = (xyz - anchor_xyz.detach()) / anchor_scale
        anchor_weight = field_commitment_detached.pow(2)
        loss_anchor = (
            anchor_weight * anchor_residual.pow(2).sum(dim=1, keepdim=True)
        ).mean()

        commitment_delta = neighbor_commitment - self_commitment.unsqueeze(1)
        gradient_field = (
            kernel.unsqueeze(-1)
            * commitment_delta.unsqueeze(-1)
            * disp
            / (neighbor_dist2.unsqueeze(-1) + 1e-6)
        ).sum(dim=1)
        gradient_field = gradient_field / kernel.sum(dim=1, keepdim=True).clamp_min(1e-6)
        interface_weight = gradient_field.norm(dim=1, keepdim=True).detach()
        interface_normal = -gradient_field / gradient_field.norm(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        interface_normal = interface_normal.detach()

        rotation = self.gaussians.get_rotation[regularizer_idx]
        covariance_root = build_scaling_rotation(scaling, rotation)
        covariance = covariance_root @ covariance_root.transpose(1, 2)
        extent_along_normal = (
            interface_normal
            * torch.bmm(covariance, interface_normal.unsqueeze(-1)).squeeze(-1)
        ).sum(dim=1, keepdim=True)
        trace = covariance.diagonal(dim1=1, dim2=2).sum(dim=1, keepdim=True)
        loss_thin = (
            interface_weight * extent_along_normal / trace.clamp_min(1e-6)
        ).mean()

        photo_step_mean = photo_delta.norm(dim=1).mean().detach()
        photo_grad_proxy_mean = (photo_step_mean / xyz_lr).detach()
        coh_grad_proxy_mean = (
            (
                field_commitment_detached.squeeze(-1) * coherence_residual.norm(dim=1)
            ).mean()
            / xyz_lr
        ).detach()
        anchor_disp_mean = (
            (anchor_weight.squeeze(-1) * (xyz.detach() - anchor_xyz.detach()).norm(dim=1))
            .mean()
            .detach()
        )
        anchor_norm_disp_mean = (
            (anchor_weight.squeeze(-1) * anchor_residual.detach().norm(dim=1)).mean()
        ).detach()
        field_std = (
            field_commitment_detached.squeeze(-1).std(unbiased=False).detach()
            if regularizer_count > 1
            else zero
        )
        debug_stats = {
            "photo_step_mean": photo_step_mean,
            "consensus_step_mean": consensus_delta.norm(dim=1).mean().detach(),
            "photo_grad_proxy_mean": photo_grad_proxy_mean,
            "coh_grad_proxy_mean": coh_grad_proxy_mean,
            "anchor_norm_disp_mean": anchor_norm_disp_mean,
            "weighted_coh_ratio": (
                self.lambda_coh
                * coh_grad_proxy_mean
                / photo_grad_proxy_mean.clamp_min(1e-12)
            ).detach(),
            "weighted_anchor_ratio": (
                self.lambda_anchor
                * anchor_norm_disp_mean
                / photo_step_mean.clamp_min(1e-12)
            ).detach(),
            "anchor_disp_mean": anchor_disp_mean,
            "interface_mean": interface_weight.mean().detach(),
            "field_mean": field_commitment_detached.mean().detach(),
            "field_std": field_std,
            "field_max": field_commitment_detached.max().detach(),
            "regularizer_idx": regularizer_idx.detach(),
            "field_commitment": field_commitment_detached.squeeze(-1),
        }

        return (
            loss_coh,
            loss_anchor,
            loss_thin,
            active_mask,
            commitment_proposals,
            regularizer_count,
            debug_stats,
        )

    def _log_structural_commitment_status(
        self,
        photo_loss,
        loss_coh,
        loss_anchor,
        loss_thin,
        active_mask,
        commitment_proposals,
        regularizer_count,
        debug_stats,
        structural_phase,
        structural_scale,
        live_iters,
    ):
        if (
            not self.use_structural_commitment
            or self.commitment_log_every <= 0
            or self.iteration_count % self.commitment_log_every != 0
            or self.gaussians.get_structural_commitment.numel() == 0
        ):
            return

        commitment = self.gaussians.get_structural_commitment.detach().squeeze(-1)
        active_count = int(active_mask.sum().item()) if active_mask is not None else 0
        total_count = int(commitment.numel())
        global_std = commitment.std(unbiased=False).item() if total_count > 1 else 0.0
        global_mean = commitment.mean().item()
        global_min = commitment.min().item()
        global_max = commitment.max().item()
        protected_count = int(
            (commitment >= self.commitment_protect_threshold).sum().item()
        )

        if active_mask is not None and active_mask.any():
            active_commitment = commitment[active_mask]
            active_mean = active_commitment.mean().item()
            active_std = (
                active_commitment.std(unbiased=False).item()
                if active_commitment.numel() > 1
                else 0.0
            )
        else:
            active_mean = 0.0
            active_std = 0.0

        proposal_mean = (
            commitment_proposals.mean().item()
            if commitment_proposals is not None and commitment_proposals.numel() > 0
            else 0.0
        )

        Log(
            "Structural commitment",
            f"iter={self.iteration_count}",
            f"phase={structural_phase}",
            f"scale={structural_scale:.3f}",
            f"live_iters={live_iters}",
            f"photo={photo_loss.detach().item():.6f}",
            f"coh={loss_coh.detach().item():.3e}",
            f"anchor={loss_anchor.detach().item():.3e}",
            f"thin={loss_thin.detach().item():.6f}",
            f"coh_w={(self.lambda_coh * loss_coh.detach()).item():.3e}",
            f"anchor_w={(self.lambda_anchor * loss_anchor.detach()).item():.3e}",
            f"thin_w={(self.lambda_thin * loss_thin.detach()).item():.6f}",
            f"active={active_count}/{total_count}",
            f"regularizer={regularizer_count}",
            f"protected={protected_count}",
            f"proposal_mean={proposal_mean:.4f}",
            f"mean={global_mean:.4f}",
            f"std={global_std:.4f}",
            f"active_mean={active_mean:.4f}",
            f"active_std={active_std:.4f}",
            f"min={global_min:.4f}",
            f"max={global_max:.4f}",
        )
        if debug_stats is not None:
            scale_tokens = [
                "Structural commitment scales",
                f"iter={self.iteration_count}",
            ]
            if "photo_step_mean" in debug_stats:
                scale_tokens.extend(
                    [
                        f"photo_step={debug_stats['photo_step_mean'].item():.3e}",
                        f"consensus_step={debug_stats['consensus_step_mean'].item():.3e}",
                        f"photo_grad={debug_stats['photo_grad_proxy_mean'].item():.3e}",
                        f"coh_grad_proxy={debug_stats['coh_grad_proxy_mean'].item():.3e}",
                        f"coh_ratio={debug_stats['weighted_coh_ratio'].item():.3e}",
                        f"anchor_norm_disp={debug_stats['anchor_norm_disp_mean'].item():.3e}",
                        f"anchor_ratio={debug_stats['weighted_anchor_ratio'].item():.3e}",
                        f"anchor_disp={debug_stats['anchor_disp_mean'].item():.3e}",
                        f"interface={debug_stats['interface_mean'].item():.3e}",
                    ]
                )
            if "field_mean" in debug_stats:
                scale_tokens.extend(
                    [
                        f"field_mean={debug_stats['field_mean'].item():.4f}",
                        f"field_std={debug_stats['field_std'].item():.4f}",
                        f"field_max={debug_stats['field_max'].item():.4f}",
                    ]
                )
            if "damping_mean" in debug_stats:
                scale_tokens.extend(
                    [
                        f"damping_threshold={debug_stats['damping_threshold'].item():.4f}",
                        f"damping_mean={debug_stats['damping_mean'].item():.3e}",
                        f"damping_max={debug_stats['damping_max'].item():.3e}",
                        f"damped={debug_stats['damped_points']}",
                    ]
                )
            Log(*scale_tokens)

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.initialized_at_iteration = 0 if self.initialized else None
        self.last_structural_phase = None
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return render_pkg

    def map(self, current_window, prune=False, iters=1):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent += 1
            iter_start = time.perf_counter()

            photo_loss = self.gaussians.get_xyz.new_tensor(0.0)
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []
            point_count_before = int(self.gaussians.get_xyz.shape[0])
            render_ms = 0.0
            structural_ms = 0.0
            densify_ms = 0.0
            opacity_reset_ms = 0.0
            optimizer_ms = 0.0
            opacity_reset = False

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                photo_loss += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                photo_loss += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            render_ms = (time.perf_counter() - iter_start) * 1000.0
            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping = photo_loss + 10 * isotropic_loss.mean()
            loss_mapping.backward()

            loss_coh = self.gaussians.get_xyz.new_tensor(0.0)
            loss_anchor = self.gaussians.get_xyz.new_tensor(0.0)
            loss_thin = self.gaussians.get_xyz.new_tensor(0.0)
            active_mask = None
            commitment_proposals = None
            regularizer_count = 0
            debug_stats = None
            structural_scale, structural_phase, live_iters = self._get_structural_phase()
            self._maybe_log_structural_phase(
                structural_phase, structural_scale, live_iters
            )
            structural_live = structural_scale > 0.0
            if structural_live:
                structural_start = time.perf_counter()
                photo_xyz_grad = (
                    None
                    if self.gaussians._xyz.grad is None
                    else self.gaussians._xyz.grad.detach()
                )
                (
                    loss_coh,
                    loss_anchor,
                    loss_thin,
                    active_mask,
                    commitment_proposals,
                    regularizer_count,
                    debug_stats,
                ) = self._compute_structural_commitment_terms(
                    photo_xyz_grad, visibility_filter_acm
                )
                structural_loss = structural_scale * (
                    self.lambda_coh * loss_coh
                    + self.lambda_anchor * loss_anchor
                    + self.lambda_thin * loss_thin
                )
                if structural_loss.requires_grad:
                    structural_loss.backward()
                damping_stats = self._apply_structural_motion_damping(
                    structural_scale, debug_stats
                )
                if debug_stats is not None and damping_stats is not None:
                    debug_stats.update(damping_stats)
                structural_ms = (time.perf_counter() - structural_start) * 1000.0
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                if (
                    self.use_structural_commitment
                    and structural_scale > 0.0
                    and active_mask is not None
                    and commitment_proposals is not None
                    and commitment_proposals.numel() > 0
                ):
                    self.gaussians.update_structural_commitment(
                        active_mask,
                        commitment_proposals,
                        self.commitment_alpha * structural_scale,
                    )
                    self._log_structural_commitment_status(
                        photo_loss,
                        loss_coh,
                        loss_anchor,
                        loss_thin,
                        active_mask,
                        commitment_proposals,
                        regularizer_count,
                        debug_stats,
                        structural_phase,
                        structural_scale,
                        live_iters,
                    )
                    active_commitment = (
                        self.gaussians.get_structural_commitment[active_mask]
                        .detach()
                        .squeeze(-1)
                    )
                    self.gaussians.update_structural_anchor_state(
                        active_mask,
                        active_commitment,
                        self.commitment_anchor_alpha * structural_scale,
                    )

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if (
                            to_prune is not None
                            and self.use_structural_commitment
                            and structural_scale >= 1.0
                            and self.gaussians.get_structural_commitment.numel() > 0
                        ):
                            protected = (
                                self.gaussians.get_structural_commitment.squeeze()
                                >= self.commitment_protect_threshold
                            ).cpu()
                            to_prune = torch.logical_and(to_prune, ~protected)
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            self.initialized_at_iteration = self.iteration_count
                            self.last_structural_phase = None
                            Log("Initialized SLAM")
                        point_count_after = int(self.gaussians.get_xyz.shape[0])
                        iter_ms = (time.perf_counter() - iter_start) * 1000.0
                        active_count = (
                            int(active_mask.sum().item())
                            if active_mask is not None
                            else 0
                        )
                        self._log_map_timing(
                            prune,
                            structural_phase,
                            structural_scale,
                            live_iters,
                            render_ms,
                            structural_ms,
                            densify_ms,
                            opacity_reset_ms,
                            optimizer_ms,
                            iter_ms,
                            False,
                            False,
                            point_count_before,
                            point_count_after,
                            active_count,
                            regularizer_count,
                        )
                        # # make sure we don't split the gaussians, break here.
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    densify_start = time.perf_counter()
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                        commitment_prune_bias=self.commitment_prune_bias
                        * structural_scale,
                        commitment_protect_threshold=self.commitment_protect_threshold,
                    )
                    densify_ms = (time.perf_counter() - densify_start) * 1000.0
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    opacity_reset_start = time.perf_counter()
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    opacity_reset_ms = (
                        time.perf_counter() - opacity_reset_start
                    ) * 1000.0
                    opacity_reset = True
                    gaussian_split = True

                optimizer_start = time.perf_counter()
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
                optimizer_ms = (time.perf_counter() - optimizer_start) * 1000.0
                point_count_after = int(self.gaussians.get_xyz.shape[0])
                iter_ms = (time.perf_counter() - iter_start) * 1000.0
                active_count = (
                    int(active_mask.sum().item()) if active_mask is not None else 0
                )
                self._log_map_timing(
                    prune,
                    structural_phase,
                    structural_scale,
                    live_iters,
                    render_ms,
                    structural_ms,
                    densify_ms,
                    opacity_reset_ms,
                    optimizer_ms,
                    iter_ms,
                    update_gaussian,
                    opacity_reset,
                    point_count_before,
                    point_count_after,
                    active_count,
                    regularizer_count,
                )
        return gaussian_split

    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
        Log("Map refinement done")

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"

        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)

    def run(self):
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                self.map(self.current_window)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
                                    "name": "trans_{}".format(viewpoint.uid),
                                }
                            )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_a],
                                "lr": 0.01,
                                "name": "exposure_a_{}".format(viewpoint.uid),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_b],
                                "lr": 0.01,
                                "name": "exposure_b_{}".format(viewpoint.uid),
                            }
                        )
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    self.map(self.current_window, iters=iter_per_kf)
                    self.map(self.current_window, prune=True)
                    self.push_to_frontend("keyframe")
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return
