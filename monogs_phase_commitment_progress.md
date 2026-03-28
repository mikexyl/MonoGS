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

Update: 2026-03-28 (timing diagnosis, delayed ramp, EuRoC prep)

The next user-visible problem report was more specific: there was a noticeably slow iteration around the early `~10%` part of the run, and the map then appeared to start drifting afterward.

That required separating three different possibilities:

- a structural-regularizer cost spike,
- a standard MonoGS densify/prune or initialization event,
- structural commitment activating too early after initialization.

The following continuation work was completed:

- EuRoC experiment scaffolding was added with
  - `configs/stereo/euroc/mh02_structural_commitment.yaml`
  - `configs/stereo/euroc/mh02_baseline_smoke.yaml`
  - `configs/stereo/euroc/mh02_structural_commitment_smoke.yaml`
- an EuRoC cross-scene run was attempted, but the official ETH dataset host stalled from this machine on both `http` and `https`, so the EuRoC comparison is prepared but still blocked on data access,
- the backend map loop was instrumented to log:
  - structural phase transitions,
  - per-iteration timing breakdown (`render_ms`, `struct_ms`, `densify_ms`, `opacity_ms`, `optim_ms`, `iter_ms`),
  - lifecycle events (`densify`, `prune`, `opacity_reset`),
  - point-count changes across those events,
- the structural formulation was revised again so commitment no longer activates immediately after `Initialized SLAM`:
  - `commitment_start_after_init_iters = 100`
  - `commitment_ramp_iters = 200`
  - structural losses and detached EMA updates are scaled by the ramp,
  - commitment-aware prune bias is scaled by the same ramp,
  - high-commitment prune protection is only enabled after the ramp reaches full strength.

Validated outcome from the timing runs:

- in the updated structural smoke run, the visible early slow step was **not** caused by structural commitment; the logs showed structural `scale=0.000` throughout warmup,
- the main early spikes before initialization were ordinary MonoGS lifecycle events:
  - densify around `iter=200` and `iter=350`,
  - `Performing initial BA for initialization` once the window first reached 8 keyframes,
- in the updated structural run, MonoGS reported `Initialized SLAM` at about `iter=788`,
- immediately after that boundary, the structural phase switched to explicit `delay`, not `live`,
- later in deep live mapping, the structural path added only a bounded extra cost on top of rendering:
  - typical `struct_ms` was about `4-6 ms`,
  - typical `render_ms` remained the dominant cost at roughly `10-30 ms`,
  - dense-event iterations remained dominated by render/prune/densify rather than the structural term,
- a baseline timing run with structural commitment disabled showed the same pre-init densify and initialization-region slow events, confirming that the early stall is a MonoGS lifecycle feature rather than a structural-only artifact.

Interpretation:

- the user's reported slow step and the beginning of structural divergence were previously conflated because they happened near the same part of the run,
- the new logs separate them cleanly:
  - the slow early step comes from MonoGS initialization / local map maintenance,
  - the structural path used to begin too close to that boundary,
  - the new delay/ramp keeps the two effects separated.

This does **not** yet prove that long-horizon convergence is fixed, but it removes the strongest early confound and gives a much cleaner basis for the next GUI validation pass.

Update: 2026-03-28 (anchor-rest reformulation)

The next user report was more specific again: even after separating initialization from structural activation, the map still diverged when the trajectory moved to the other side of the table.

The live logs explained why the previous formulation was still insufficient:

- `coh_w` remained effectively negligible,
- `thin_w` was the only consistently material structural term,
- the method therefore had shape pressure and some lifecycle bias, but still no real long-horizon stabilizer for the committed subset.

That led to a second formulation change that remains faithful to the two-type idea:

- a persistent per-Gaussian anchor-rest position was added to the Gaussian state,
- committed Gaussians are now pulled toward that slow anchor memory through a new `lambda_anchor` term,
- the anchor-rest state is updated detached by a slow EMA weighted by commitment,
- commitment proposals are now additionally modulated by recent observation support from SLAM-window visibility counts, so anchor candidates are biased toward repeatedly re-observed structure rather than merely low-gradient points.

State plumbing implemented:

- `gaussian_splatting/scene/gaussian_model.py` now carries `structural_anchor_xyz`,
- new Gaussians initialize anchor-rest from their current xyz,
- clone/split inherit the parent anchor-rest state,
- prune preserves it,
- PLY save/load now round-trips the anchor-rest coordinates as `anchor_x`, `anchor_y`, `anchor_z`.

Validation outcome so far:

- the first anchor-rest smoke run proved the new term was finally material, but its initial normalization used the Gaussian's own splat scale and made the anchor loss too strong,
- that was corrected by normalizing anchor drift with local neighborhood scale instead,
- the clean recalibrated structural smoke then showed:
  - `anchor_w` around `5e-3` to `1.3e-2`,
  - `thin_w` around `2e-3`,
  - `coh_w` still negligible,
  - sparse commitment with only about `6-11` protected anchors in the sampled live region,
  - `max` commitment above `0.94`, which means the structural subset is once again genuinely anchor-like rather than diffuse.

Interpretation:

- the structural term is now doing something qualitatively closer to the intended solid/liquid split,
- the dominant stabilizer is no longer the interface-thinness prior,
- the far-side table traversal now needs to be judged visually under this anchor-rest formulation rather than under the earlier weak-coherence version.

Update: 2026-03-28 (field-solidness and adaptive damping)

The next user observation was that the revised method still did not get past the table cleanly, and they suggested an important reformulation: treat "solidness" as a property of the local field rather than of each Gaussian independently.

That is now reflected in the implementation.

Formulation change:

- persistent commitment is still stored per Gaussian for lifecycle, clone/split inheritance, and checkpointing,
- but the optimizer no longer uses that raw state directly as the immediate physical solidness,
- instead it builds a local kNN-smoothed field estimate and uses that field for coherence, anchor-rest weighting, interface estimation, and direct xyz damping,
- the direct damping gate is no longer tied to a fixed raw commitment level alone; it is derived from the upper tail of the current field, capped by the configured threshold.

Why this was necessary:

- the first field-smoothed smoke showed the right qualitative behavior, but it also revealed that a fixed damping threshold of `0.6` was no longer appropriate,
- after smoothing, `field_max` was typically only around `0.3-0.57` even when raw commitment `max` had become anchor-like again,
- that meant the newly added direct damping path existed in code but remained inactive in practice.

Implemented backend changes:

- `utils/slam_backend.py` now computes a field-smoothed solidness estimate from the local kNN graph,
- structural losses now use that field estimate instead of raw stored commitment,
- xyz gradients are now damped directly on the sampled structural subset before the optimizer step,
- the damping threshold is now logged explicitly and derived from the field's upper quantile (`commitment_damping_quantile`) rather than only from a fixed absolute level.

Validation outcome so far:

- the first smoke after the field refactor confirmed the problem clearly: `damping_mean=0`, `damped=0`, because the field never crossed the old absolute threshold,
- after switching to the adaptive field-relative threshold, the headless TUM FR1 Desk smoke showed sustained non-zero damping in live mode,
- representative live values from the clean smoke:
  - around `iter=1210`: `field_max=0.3761`, `damping_threshold=0.1722`, `damping_mean=4.514e-04`, `damping_max=4.547e-02`, `damped=205`,
  - around `iter=1240`: `field_max=0.4681`, `damping_threshold=0.1752`, `damping_mean=9.085e-04`, `damping_max=9.460e-02`, `damped=205`,
  - around `iter=1300`: `field_max=0.3146`, `damping_threshold=0.1313`, `damping_mean=3.326e-04`, `damping_max=3.340e-02`, `damped=205`.

Interpretation:

- the structural subset is now acting more like a field-defined anchor region than like a few isolated committed splats,
- the direct stabilizer is finally active during live mapping instead of being a dormant code path,
- this does not yet prove that the table-crossing divergence is solved visually, but it does remove an important implementation gap: the method now actually exerts optimizer-side resistance where the field says the map should be solid.

Update: 2026-03-28 (settled-solidness diagnosis and geometric-mean trust fix)

The next round of user feedback was more discriminating: the new field-driven anchor version still looked worse, and the user asked whether the system might be solidifying too much during initialization.

The logs answered that question clearly for the current build:

- structural commitment was completely inactive throughout MonoGS warmup,
- after `Initialized SLAM`, the system still stayed in the explicit `delay` phase for `100` post-init iterations,
- the first ramp activation happened only after that delay, with tiny commitment values,
- so initialization over-solidification is **not** the current failure mode.

Instead, the first "settled-solidness" implementation was too strict in a different way. The operative solidness was defined as a full product of:

- the field-smoothed commitment,
- the current sparse stable-tail proposal,
- observation support,
- anchor settledness.

That made the new gate mathematically clean but practically too suppressive. In the headless FR1 Desk smoke run:

- raw field values became significant in live mode, for example around `iter=1390` the field reached roughly `field_mean=0.0681` and `field_max=0.4073`,
- but the operative solid subset collapsed to roughly `solid_mean=0.0013` and `solid_max=0.1465`,
- damping therefore stayed weak or sparse even late in live mode,
- protected-anchor counts also stayed at or near zero, showing that the anchor subset was not really consolidating.

That diagnosis motivated a second formulation change:

- keep the same three trust factors,
- but combine them with a **geometric mean** instead of a full product,
- then multiply that softer trust score by the field commitment to obtain operative solidness.

The new definition is:

- field commitment still determines where structural material exists,
- stability, support, and settledness now modulate that field without annihilating it whenever one factor is merely modest rather than near one.

Validated outcome from the updated headless smoke:

- around `iter=1390` in live mode:
  - `field_mean=0.0814`
  - `field_max=0.5478`
  - `trust_mean=0.0537`
  - `solid_mean=0.0122`
  - `solid_max=0.3495`
  - `damping_threshold=0.0325`
  - `damped=205`
- around `iter=1410`:
  - `field_max=0.5083`
  - `solid_max=0.4205`
  - `damping_max=0.1289`
  - `damped=179`
  - `protected=5`
- around `iter=1450`:
  - `field_mean=0.0767`
  - `solid_mean=0.0101`
  - `solid_max=0.3632`
  - `damping_threshold=0.0290`
  - `damped=205`
  - `protected=9`

Interpretation:

- the operative solid subset no longer collapses toward zero while the field itself is strong,
- the damping path is once again meaningfully active on the sampled structural subset,
- the current build is now much closer to the intended "field exists, trust gates how rigid it becomes" behavior,
- but the next question is visual rather than purely scalar: whether this restored solid subset actually helps the far-side table traversal, or whether it has again become too rigid later in the sequence.

Update: 2026-03-28 (GUI verdict on checkpoint `ed72cc0`)

The next step after the trust-weighted field fix was a direct GUI inspection on the monocular TUM FR1 Desk run using:

- pushed checkpoint `ed72cc0`,
- `configs/mono/tum/fr1_desk_structural_commitment_gui_smoke.yaml`,
- Pixi with `DISPLAY=:0`.

User-visible outcome:

- the map looked **slightly better** than the previous field-driven version,
- but it still **diverged visually**,
- so the current checkpoint is an improvement, not a solution.

What this checkpoint now establishes with reasonable confidence:

- the current failure is **not** "structural commitment hardens the map during initialization",
- the current failure is also **not** the earlier "solid subset collapses to nearly zero" bug from the overly multiplicative trust gate,
- the current build does produce a real operative solid subset in live mode,
- but that restored solid subset is still not enough to carry the monocular trajectory robustly through the difficult far-side table traversal.

Representative live values from the current checkpoint during the validated headless and GUI-linked runs:

- `field_mean` is typically around `0.06-0.08`,
- `field_max` is typically around `0.42-0.55`,
- `trust_mean` is typically around `0.04-0.06`,
- `solid_mean` is typically around `0.008-0.012`,
- `solid_max` is typically around `0.33-0.42`,
- sampled damping is active on a large subset (`damped` often around `179-205`),
- the protection count is no longer zero-only, but still modest (single digits to low teens rather than thousands).

Interpretation:

- the current method is now in a more plausible regime: there is a real but selective structural subset, and it is not activating during initialization,
- the slight visual improvement reported by the user is consistent with that regime change,
- however, the far-side table failure remains, which means the present formulation still does not provide enough correct long-horizon geometric stabilization for the hard monocular viewpoint transition.

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
- structural commitment updates and commitment-aware pruning/protection are now delayed until MonoGS reports `Initialized SLAM`, and then further delayed/ramped for a fixed number of iterations so the fragile post-init settling phase stays closer to baseline behavior.

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
- `commitment_start_after_init_iters`
- `commitment_ramp_iters`
- `commitment_obs_weight`
- `commitment_anchor_alpha`
- `map_timing_log_every`
- `map_slow_iteration_ms`
- `lambda_coh`
- `lambda_anchor`
- `lambda_thin`
- `commitment_prune_bias`
- `commitment_protect_threshold`

Current default posture:

- the feature is off by default,
- the added values are conservative placeholders for opt-in experiments,
- the shared opt-in default for `lambda_coh` is now `0.5` after live diagnostics showed that `0.1` left coherence too weak to matter,
- the shared opt-in default for `commitment_stable_quantile` is now `0.75`,
- the shared opt-in defaults now delay structural activation by `100` iterations after MonoGS initialization and ramp it over the next `200` iterations,
- the shared opt-in defaults now also include observation-weighted commitment proposals and a slow anchor-rest EMA,
- the shared opt-in default for `commitment_protect_threshold` is now `0.85` so only a smaller anchor subset affects lifecycle decisions.

Example opt-in config:

- `configs/mono/tum/fr1_desk_structural_commitment.yaml`

## Validation Completed

Validated:

- `python3 -m py_compile slam.py utils/slam_backend.py gaussian_splatting/scene/gaussian_model.py utils/eval_utils.py utils/wandb_utils.py`
- baseline Pixi smoke test on TUM FR1 Desk with `configs/mono/tum/fr1_desk_baseline_smoke.yaml`
- structural Pixi smoke tests on TUM FR1 Desk with `configs/mono/tum/fr1_desk_structural_commitment_smoke.yaml`
- GUI startup validation through Pixi with the same structural smoke configuration and `DISPLAY=:0`
- timing-instrumented structural and baseline Pixi smoke runs on TUM FR1 Desk to isolate the early slow iteration from structural activation

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
- after the delayed-ramp change, the timing logs showed `Initialized SLAM` around `iter=788` in the structural smoke run, followed by explicit `phase=delay` rather than immediate structural activation,
- in that updated structural timing run, the pre-init slow points were still densify / initialization-region events while `scale=0.000`, which strongly suggests the user's early slow-step observation is not caused by the structural term itself,
- once the updated structural run reached full `phase=live`, `struct_ms` stayed bounded at roughly `4-6 ms` while total iteration time remained dominated by render and normal MonoGS lifecycle work,
- the baseline timing run reproduced the same pre-init densify and initialization-region slow events with structural commitment fully disabled,
- EuRoC smoke configs are now ready, but the actual EuRoC runtime comparison remains blocked because the official dataset host stalled during download from this machine.
- after the anchor-rest reformulation, the first smoke run confirmed that the new stabilizer is finally material in live mapping, but it initially had excessive strength due to a bad scale normalization,
- after recalibrating anchor normalization to local neighborhood scale, the clean smoke run showed `anchor_w` in the `5e-3` to `1.3e-2` band, which is stronger than `thin_w` but no longer pathological,
- in that recalibrated live region, commitment remained sparse, with only a handful of protected anchors while `max` commitment still exceeded `0.94`,
- an updated GUI run was launched on the recalibrated anchor-rest formulation for direct visual inspection of the far-side table traversal.

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
4. The current trust-weighted field gate gives a real solid subset again, but the user still observed visual divergence on the far-side table traversal, so the operative solidness is not yet translating into sufficient scene-level stability.
5. The anchor subset is now much smaller than in the earlier over-protected build, but some Gaussians still saturate near commitment `1.0`, so longer-run validation is still needed to see whether anchor identities remain sensible over time.
6. The current anchor-rest contribution can become large relative to the raw step scale in some live segments, which suggests the next revision should look at how anchor-rest influence is scheduled or normalized during hard viewpoint transitions rather than only at whether anchors exist.
7. The structural path now appears to cost roughly baseline-plus-some-headroom rather than baseline-times-six, but it is still measurably above baseline and should be profiled again on longer sequences.
8. The existing isotropic scaling prior is still active alongside the thinness prior; their interaction should be evaluated experimentally.
9. EuRoC cross-scene validation is currently blocked on dataset availability because the official host did not respond from this environment.
10. The anchor-rest ratio diagnostic is currently only a coarse scale-comparison heuristic; it is useful for relative tuning, but not yet a calibrated physical metric.

## Recommended Next Experiment

1. Resume from checkpoint `ed72cc0`, which is the current best-documented monocular structural-commitment build.
2. Keep the current post-init delay and trust-weighted field gate as the starting point; do not go back to the old full-product trust rule or the earlier aggressive scene-specific damping override.
3. Focus the next revision on how anchor-rest influence behaves during difficult viewpoint transitions:
   - first inspect whether `lambda_anchor` is still too strong once solidness becomes nontrivial,
   - then inspect whether `commitment_anchor_alpha` updates the anchor memory too slowly or too quickly through the table-crossing segment,
   - then inspect whether anchor-rest should be applied more selectively when support drops sharply even if field commitment remains high.
4. Only after that, retune `lambda_thin` or the sparse-anchor commitment proposal itself.
5. Resolve EuRoC dataset access, then repeat the same baseline-vs-structural timing comparison on `MH_02_easy` to check whether the divergence remains scene-sensitive.

## Resume Checkpoint

- Best current checkpoint: `ed72cc0`
- Visual verdict: slight improvement, still diverging
- Most likely current bottleneck: later-phase structural behavior during the hard far-side monocular transition, not initialization
- Most important validated negative result: initialization over-solidification is not the active bug in the current build
