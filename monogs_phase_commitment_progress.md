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
- structural losses, commitment EMA updates, commitment-aware pruning bias, and SLAM-window protection are now all gated to the post-initialization phase (`lifecycle=live`) instead of affecting MonoGS `warmup`.

Update: 2026-03-28 (coherence retune validated)

The next validation pass focused on whether coherence was still too weak after the memory fixes.

- an extra live diagnostic (`coh_ratio`) was added to compare the weighted coherence proxy against the photometric xyz gradient scale,
- the shared structural default for `lambda_coh` was increased from `0.1` to `0.5`,
- a clean TUM FR1 Desk structural smoke run then completed successfully with the retuned setting.

Validated outcome:

- the structural smoke run completed cleanly in about `104.08s` at about `5.69 FPS`,
- live `coh_ratio` moved from the earlier `~0.09` regime into a much healthier `~0.4` to `~0.7` band,
- the earlier "6x baseline GPU memory" impression was traced to overlapping stale structural runs on the same GPU rather than a single active structural process pair,
- with only the current structural run active, total GPU memory sampled at about `1918 MiB`, which is still above the clean baseline sample but no longer pathological.

Update: 2026-03-28 (two-type anchor revision)

The first GUI run after the coherence retune still diverged visually. The logs pointed to a more specific issue:

- the raw percentile proposal always had `proposal_mean=0.5000` by construction,
- global commitment therefore drifted toward the middle of the range,
- the number of protected Gaussians climbed into the thousands, which was far too broad for a map that is supposed to have a relatively small structural-anchor subset.

That behavior was too soft and too democratic for the intended two-type Gaussian idea. The formulation was therefore revised in both code and design notes:

- commitment proposals are now sparse and only assigned to the stable tail of active Gaussians (`commitment_stable_quantile = 0.75` by default),
- Gaussians outside that stable tail decay back toward the exploratory regime rather than drifting toward `0.5`,
- the structural protection threshold was raised from `0.7` to `0.85`,
- the live `coh_ratio` diagnostic was corrected to include commitment weighting, because the old proxy became misleading once commitment turned sparse.

Validated outcome:

- the updated structural smoke run completed cleanly in about `100.68s` at about `5.88 FPS`,
- `proposal_mean` dropped from `0.5000` to about `0.1250`,
- global commitment mean stayed in the rough `0.08` to `0.15` range instead of drifting toward `0.5`,
- protected-Gaussian counts stayed in the low hundreds rather than the low thousands,
- the corrected commitment-weighted `coh_ratio` in early live mapping is now around `1e-3` to `1e-2`, which matches the intentionally sparse anchor set.

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
- structural commitment updates and commitment-aware pruning/protection are now delayed until MonoGS reports `Initialized SLAM`, so the fragile initialization phase stays closer to baseline behavior.

## Config Surface

Shared base and live configs now expose:

- `use_structural_commitment`
- `commitment_init_value`
- `commitment_alpha`
- `commitment_knn`
- `commitment_stable_quantile`
- `commitment_chunk_size`
- `commitment_max_points`
- `commitment_log_every`
- `lambda_coh`
- `lambda_thin`
- `commitment_prune_bias`
- `commitment_protect_threshold`

Current default posture:

- the feature is off by default,
- the added values are conservative placeholders for opt-in experiments,
- the shared opt-in default for `lambda_coh` is now `0.5` after live diagnostics showed that `0.1` left coherence too weak to matter,
- the shared opt-in default for `commitment_stable_quantile` is now `0.75`,
- the shared opt-in default for `commitment_protect_threshold` is now `0.85` so only a smaller anchor subset affects lifecycle decisions.

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
- after raising `lambda_coh` from `0.1` to `0.5`, the retuned structural smoke run completed cleanly and reported about `104.08s` total time at about `5.69 FPS`,
- the live diagnostics now report `coh_ratio` mostly in the `0.4` to `0.7` range, confirming that coherence is now material relative to the photometric xyz signal during live mapping,
- the structural diagnostics only begin after `Initialized SLAM`, confirming that the structural path is bypassed during warmup rather than merely tagged as a separate lifecycle state,
- the corrected clean-GPU reading for a single active structural smoke run was about `1918 MiB` total device memory at sample time, which is higher than baseline but far below the earlier contaminated multi-run readings.
- after the sparse-anchor revision, the structural smoke run completed cleanly in about `100.68s` total time at about `5.88 FPS`,
- after that revision, `proposal_mean` was about `0.1250`, commitment mean stayed near `0.1`, and protected-Gaussian counts stayed in the low hundreds instead of the low thousands,
- the corrected commitment-weighted `coh_ratio` now reports about `1e-3` to `1e-2` in early live mapping, which matches the intended anchor-sparse regime far better than the old unweighted proxy.

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
3. The coherence term is no longer large once commitment is made sparse; the next question is whether it is now too weak to improve geometry on longer runs.
4. The anchor subset is now much smaller, but some Gaussians still saturate near commitment `1.0`, so longer-run validation is still needed to see whether anchor identities remain sensible over time.
5. The structural path now appears to cost roughly baseline-plus-some-headroom rather than baseline-times-six, but it is still measurably above baseline and should be profiled again on longer sequences.
6. The existing isotropic scaling prior is still active alongside the thinness prior; their interaction should be evaluated experimentally.

## Recommended Next Experiment

1. Use the capped structural smoke config as the new safe starting point.
2. Run the GUI configuration again with the sparse-anchor formulation and check whether the earlier visible divergence is reduced.
3. Run the full FR1 Desk structural config and compare convergence against the baseline now that the short smoke run is stable.
4. Profile structural memory again on a longer run to verify that the clean smoke-test memory picture holds outside the short validation sequence.
