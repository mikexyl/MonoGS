# MonoGS Structural Commitment: Implementation Progress

## Status

Date: 2026-03-27

The first implementation pass of the structural commitment method is now in the codebase.
This pass follows the "minimal core" plan:

- per-Gaussian structural commitment state is persistent,
- commitment is updated from relative photometric stability with a detached EMA,
- local mapping now supports commitment-weighted coherence and interface-thinness regularization,
- pruning and densification now preserve and use commitment,
- the feature is disabled by default in shared configs.

This pass does **not** implement explicit post-Adam mean-step blending.
The current implementation uses differentiable regularization plus commitment-aware lifecycle behavior.

Update: 2026-03-28

A continuation pass added runtime observability and experiment scaffolding:

- periodic structural commitment logging during mapping,
- config support for log cadence control,
- a ready-to-run opt-in example config for TUM FR1 Desk,
- an explicit note that no local dataset checkout was available for runtime validation in this workspace.

Update: 2026-03-28 (later)

Validation and runtime tooling continued:

- the TUM FR1 Desk dataset was downloaded locally for smoke testing,
- Pixi-based smoke runs were used for both headless and GUI validation,
- the Pixi activation environment now exports `DISPLAY=:0` by default so GUI runs do not require a per-command display override.

Update: 2026-03-28 (runtime diagnosis and stabilization)

The first real smoke tests exposed two practical issues in the structural path:

- the structural backend initially consumed far more GPU memory than the baseline smoke configuration,
- commitment-aware lifecycle effects were activating during MonoGS initialization, which is the least stable phase of the pipeline.

Both were addressed in follow-up patches:

- the structural losses now use the detached xyz gradient that already exists after the photometric backward pass, rather than keeping the render graph alive for a separate `autograd.grad` call,
- the expensive structural regularizers now run on a capped active subset (`commitment_max_points`) while commitment proposals and EMA updates still cover the full active set,
- local kNN field construction now halves its chunk size automatically on CUDA OOM instead of hard-failing immediately,
- commitment-aware pruning bias and SLAM-window protection are now gated to the post-initialization phase (`lifecycle=live`) instead of affecting the `warmup` phase.

## Implemented Components

### 1. Persistent Gaussian state

`gaussian_splatting/scene/gaussian_model.py`

Implemented:

- `structural_commitment` tensor stored per Gaussian on CUDA,
- low-commitment initialization for newly inserted Gaussians,
- commitment inheritance across clone/split,
- commitment preservation across prune,
- optional PLY round-trip support through a `structural_commitment` vertex field,
- detached EMA helper for commitment updates.

Current behavior:

- new Gaussians start at `commitment_init_value`,
- cloned and split Gaussians inherit the parent commitment,
- saved point clouds now include commitment,
- old point clouds without the field still load with the configured default value.

### 2. Mapping-time commitment update

`utils/slam_backend.py`

Implemented:

- extraction of per-Gaussian photometric xyz gradients from `photo_loss`,
- active-Gaussian percentile ranking,
- commitment proposal `q = 1 - percentile_rank`,
- detached EMA update of commitment for Gaussians active in the current mapping iteration.

Important implementation note:

- commitment is updated after the backward pass has exposed the photometric gradient signal, but before same-iteration prune/split bookkeeping runs.
- this ordering is intentional so structural maintenance can use the freshest commitment values without introducing optimizer-state index mismatch after pruning.

### 3. Local commitment field approximation

`utils/slam_backend.py`

Implemented approximation:

- field computations are restricted to Gaussians active in the current mapping iteration,
- local neighborhoods are built with chunked kNN search using `torch.cdist`,
- the expensive structural regularizers run on a capped active subset while commitment proposals still use the full active set,
- chunked kNN search now reduces its chunk size automatically if CUDA memory is tight,
- consensus and interface estimates are computed from that bounded local neighborhood graph.

This is the concrete approximation to the continuous field definitions in the design note.
It keeps the implementation bounded and avoids a global all-pairs field.

### 4. Coherence regularization

`utils/slam_backend.py`

Implemented:

- photometric xyz gradients are converted into detached local motion proposals,
- a commitment-weighted local consensus motion is computed from neighboring active Gaussians,
- a coherence target position is formed from the blended photometric/consensus proposal,
- `lambda_coh` scales a differentiable penalty that pulls committed Gaussians toward that target.

Interpretation:

- low-commitment Gaussians are effectively dominated by the original photometric objective,
- high-commitment Gaussians receive stronger pressure toward locally coherent motion.

### 5. Interface-localized thinness prior

`utils/slam_backend.py`

Implemented:

- local commitment gradients are estimated from neighbor commitment differences,
- interface normals are inferred from those local gradients,
- covariance extent along the inferred normal is penalized,
- the penalty is weighted by interface strength and scaled by `lambda_thin`.

Important implementation note:

- interface normals and interface weights are detached when used in the thinness term.
- this keeps the term focused on shaping covariance rather than trying to optimize the interface estimate itself.

### 6. Commitment-aware lifecycle behavior

`gaussian_splatting/scene/gaussian_model.py`
`utils/slam_backend.py`

Implemented:

- opacity pruning threshold is modulated by commitment,
- low-commitment Gaussians are pruned more aggressively,
- high-commitment Gaussians are pruned more conservatively on opacity,
- SLAM-window pruning protects highly committed Gaussians from the observation-count culling path.

Guardrail:

- size-based safeguards remain active even for high-commitment Gaussians.
- commitment-aware pruning/protection are now delayed until MonoGS reports `Initialized SLAM`, so the fragile initialization phase stays closer to baseline behavior.

## Config Surface

Shared base and live configs now expose:

- `use_structural_commitment`
- `commitment_init_value`
- `commitment_alpha`
- `commitment_knn`
- `commitment_chunk_size`
- `commitment_max_points`
- `commitment_log_every`
- `lambda_coh`
- `lambda_thin`
- `commitment_prune_bias`
- `commitment_protect_threshold`

Current default posture:

- the feature is off by default,
- the added values are conservative placeholders for opt-in experiments.

Example opt-in config:

- `configs/mono/tum/fr1_desk_structural_commitment.yaml`

## Validation Completed

Validated:

- `python3 -m py_compile slam.py utils/slam_backend.py gaussian_splatting/scene/gaussian_model.py utils/eval_utils.py utils/wandb_utils.py`
- baseline Pixi smoke test on TUM FR1 Desk with `configs/mono/tum/fr1_desk_baseline_smoke.yaml`
- structural Pixi smoke tests on TUM FR1 Desk with `configs/mono/tum/fr1_desk_structural_commitment_smoke.yaml`
- GUI startup validation through Pixi with the same structural smoke configuration and `DISPLAY=:0`

Runtime observations from the smoke tests:

- the baseline smoke configuration reset once during initialization, then reached `Initialized SLAM`, completed cleanly, and reported about `102.36s` total time at about `5.78 FPS`,
- the first structural implementation also ran, but it showed a large GPU-memory overhead and commitment-aware lifecycle effects during initialization,
- after the memory refactor and capped-regularizer change, the structural path dropped from multi-GB backend usage to baseline-class usage,
- a clean baseline sample measured about `494 MiB + 640 MiB` across the two main MonoGS Python processes,
- a clean capped structural sample measured about `488 MiB + 540 MiB` early in the run and about `628 MiB + 782 MiB` later in the live phase,
- the structural logs now report `regularizer=2048`, confirming that the expensive loss path is bounded,
- the structural logs now report `lifecycle=warmup` before initialization and `lifecycle=live` after `Initialized SLAM`, confirming that commitment-aware lifecycle behavior is deferred until the map is initialized.

The runtime observability now exposes:

- photometric loss,
- coherence loss,
- weighted coherence contribution,
- thinness loss,
- weighted thinness contribution,
- active/total Gaussian counts,
- regularizer subset size,
- lifecycle phase,
- protected-Gaussian count,
- commitment proposal mean,
- global and active commitment mean/std,
- commitment min/max.

## Known Risks / Follow-up Work

1. The current kNN field is local and active-set-only, not a full-map continuous splat field.
2. Coherence is realized as a target-position regularizer, not explicit optimizer-step blending.
3. The coherence term is now observable in logs but remains numerically small compared with the thinness term and the photometric objective, so further convergence-focused tuning is still required.
4. Commitment values still climb quickly, so the chosen EMA rate, protection threshold, and proposal definition may need refinement once memory is no longer the bottleneck.
5. The existing isotropic scaling prior is still active alongside the thinness prior; their interaction should be evaluated experimentally.

## Recommended Next Experiment

1. Use the capped structural smoke config as the new safe starting point.
2. Compare convergence against the baseline now that GPU memory is no longer the limiting issue.
3. Tune the coherence formulation and/or weight, because it is still much weaker than the thinness path in the current logs.
4. Re-evaluate commitment protection thresholds now that lifecycle behavior is deferred to the post-initialization phase.
