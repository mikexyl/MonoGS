# Structural Commitment for Gaussian Splatting SLAM

## MonoGS-Based Formulation

## 1. Motivation

MonoGS is an especially natural baseline for this idea because its map is already **incremental**, **Gaussian-native**, and **continuously updated** by photometric tracking and local mapping. In such a setting, Gaussians do not all play the same role throughout training:

- some Gaussians become **persistent structural support** for the map,
- some remain **tentative, flexible compensators** for view inconsistency, transient appearance effects, and underconstrained regions.

We model this distinction explicitly with a soft per-Gaussian **structural commitment** variable. The practical goal is still a **two-type map**:

- a relatively small population of **structural anchor Gaussians**,
- a larger population of **exploratory / compensator Gaussians**.

The implementation should therefore not push the whole map toward middling commitment. Instead, it should keep most Gaussians liquid by default and only let a stable minority harden into anchors. We still use a soft state variable, but it is meant to realize an emergent two-regime behavior:

- **liquid / tentative**: free to move with local photometric evidence,
- **solid / committed**: trusted map support, encouraged to move coherently with neighboring committed Gaussians.

This is a natural fit for MonoGS because SLAM already provides an incremental lifecycle: birth, repeated observation, consolidation, correction, pruning.

---

## 2. Representation

Let the MonoGS map consist of Gaussians
\[
\mathcal{G} = \{g_i\}_{i=1}^N,
\]
with each Gaussian parameterized by
\[
g_i = (\mu_i, \Sigma_i, \alpha_i, c_i, s_i),
\]
where:

- \(\mu_i \in \mathbb{R}^3\): mean,
- \(\Sigma_i \in \mathbb{R}^{3\times 3}\): covariance,
- \(\alpha_i\): opacity,
- \(c_i\): appearance parameters,
- \(s_i \in [0,1]\): **structural commitment**.

Interpretation:

- \(s_i \approx 0\): tentative / liquid Gaussian,
- \(s_i \approx 1\): committed / solid Gaussian.

Unlike a hard label, \(s_i\) is a soft state variable that evolves online with the map.

---

## 3. Structural Commitment Update

### 3.1 Relative photometric stability

At a MonoGS mapping step, let
\[
v_i^{(t)} = \Delta \mu_i^{\text{photo},(t)}
\]
denote the raw photometric update for Gaussian \(i\), and let
\[
g_i^{(t)} = \left\| \nabla_{\mu_i} \mathcal{L}_{\text{photo}}^{(t)} \right\|
\]
be the magnitude of its mean-gradient under the photometric loss.

Rather than using an absolute threshold, we use a **relative ranking** across active Gaussians. Define the percentile rank
\[
r_i^{(t)} = \operatorname{percentile}(g_i^{(t)}) \in [0,1],
\]
where larger \(r_i\) means larger photometric instability.

Define the raw stability score as
\[
p_i^{(t)} = 1 - r_i^{(t)}.
\]
So Gaussians in the low-gradient tail get high stability score; Gaussians in the high-gradient tail get low score.

However, using \(q_i = p_i\) directly is too permissive: it drives the average commitment toward \(0.5\) by construction, which is exactly what we do **not** want if the map is supposed to separate into anchor-like and exploratory Gaussians.

Instead, we only allow the stable tail to accumulate commitment. Let \(\tau \in [0,1)\) be a stable-tail quantile, for example \(\tau = 0.75\). Define
\[
q_i^{(t)} = \operatorname{clip}\left(\frac{p_i^{(t)} - \tau}{1-\tau}, 0, 1\right).
\]

Interpretation:

- only the top \((1-\tau)\) fraction of relatively stable active Gaussians receive nonzero commitment proposals,
- Gaussians outside that stable tail receive \(q_i \approx 0\) and therefore decay back toward the exploratory state,
- commitment is therefore sparse by design, which is much closer to the intended two-type behavior.

### 3.2 Soft online update

We update structural commitment with an exponential moving average:
\[
s_i^{(t+1)} = (1-\alpha) s_i^{(t)} + \alpha q_i^{(t)},
\]
with \(\alpha \in (0,1]\) a single smoothing parameter.

This has three advantages:

- avoids brittle freeze/melt thresholds,
- normalizes away global gradient-scale drift,
- lets commitment be **earned over time** rather than decided from a single iteration,
- prevents the whole active set from drifting toward middling commitment.

### 3.3 Post-initialization activation schedule

MonoGS has a fragile transition between map initialization and normal SLAM operation. In practice, the first full-window initialization bundle-adjustment and the first post-init prune can both cause large transient map changes. Turning structural commitment on immediately at that boundary encourages the method to harden premature anchor identities.

So in the practical MonoGS implementation, structural commitment is not activated immediately when MonoGS first reports `Initialized SLAM`. Instead, let \(t_{\mathrm{init}}\) denote that iteration, let \(\Delta_{\mathrm{delay}}\) be a fixed delay, and let \(\Delta_{\mathrm{ramp}}\) be a ramp length. Define a structural activation scale
\[
\gamma^{(t)} =
\begin{cases}
0, & t - t_{\mathrm{init}} < \Delta_{\mathrm{delay}}, \\
\operatorname{clip}\left(\frac{t - t_{\mathrm{init}} - \Delta_{\mathrm{delay}} + 1}{\Delta_{\mathrm{ramp}}}, 0, 1\right), & \text{otherwise.}
\end{cases}
\]

Interpretation:

- during the delay period, the map is allowed to settle with no structural influence,
- during the ramp period, commitment EMA updates and structural losses are introduced gradually,
- pruning protection for high-commitment anchors is only enabled after the ramp reaches full strength.

This keeps the two-type idea intact while avoiding a known failure mode where unstable post-init geometry is hardened too early.

### 3.4 Separate state update

Importantly, \(s_i\) is **not** optimized jointly with the Gaussian parameters by gradient descent. Instead, it is updated **after** the Gaussian optimization step as a detached state update.

This keeps the semantics clean:

- Gaussian parameters explain images,
- \(s_i\) tracks how structurally committed each Gaussian has become.

---

## 4. Commitment Field and Consensus Motion

### 4.1 Commitment field

We splat commitment into a continuous field:
\[
\mathcal{S}(x) = \frac{\sum_i s_i \, \mathcal{G}_i(x)}{\sum_i \mathcal{G}_i(x)}.
\]
This field measures the local degree of structural commitment in the current Gaussian map.

### 4.2 Consensus motion field

Let \(v_i = \Delta \mu_i^{\text{photo}}\) denote the raw photometric motion proposal. We define a local consensus motion field using commitment-weighted Gaussian splatting:
\[
U(x) = \frac{\sum_i s_i \, \mathcal{G}_i(x) \, v_i}{\sum_i s_i \, \mathcal{G}_i(x) + \varepsilon}.
\]
This is not literal physical viscosity; it is better interpreted as a **coherence field** for committed Gaussians.

Committed regions are therefore encouraged to move like a patch rather than as independent particles.

---

## 5. Commitment-Modulated Gaussian Motion

In the idealized formulation, each Gaussian would follow a commitment-modulated mean update:
\[
\Delta \mu_i^{\text{eff}} = (1-s_i) \, \Delta \mu_i^{\text{photo}} + s_i \, U(\mu_i).
\]

Interpretation:

- low-\(s_i\) Gaussians remain agile and follow photometric evidence,
- high-\(s_i\) Gaussians move with local consensus and resist isolated drift.

This is especially natural in SLAM:

- newly inserted Gaussians should stay mobile,
- repeatedly re-observed Gaussians should become stable map support,
- inconsistent Gaussians should remain flexible or be pruned.

In the current MonoGS implementation, this is used as an **interpretive target**, not as a literal overwrite of Adam's xyz step. The implementation realizes the effect through:

- a coherence loss that encourages committed Gaussians to align with local consensus motion,
- a thinness prior localized by the commitment interface,
- commitment-aware pruning bias and anchor protection,
- the post-initialization delay/ramp schedule above.

That choice keeps the optimizer plumbing simpler and was empirically safer in the existing MonoGS codepath.

### Coherence regularization

We add a coherence penalty:
\[
\mathcal{L}_{\text{coh}} = \sum_i s_i \, \left\| v_i - U(\mu_i) \right\|^2.
\]
This penalizes committed Gaussians that try to move differently from the local consensus motion.

---

## 6. Interface-Localized Surface Prior

### 6.1 Commitment interface

The gradient of the commitment field is large near transitions between tentative and committed regions:
\[
\nabla \mathcal{S}(x).
\]
We interpret this as an **emergent interface of structural commitment**. In practice, this interface is expected to correlate with evolving surface support, but it should not be claimed to equal the geometric surface exactly.

### 6.2 Interface normal

A local interface normal is estimated as
\[
\hat{n}_i = - \frac{\nabla \mathcal{S}(\mu_i)}{\|\nabla \mathcal{S}(\mu_i)\| + \varepsilon}.
\]

### 6.3 Normal-aligned thinness

Let \(\Sigma_i\) be the covariance of Gaussian \(i\). Instead of merely forcing the smallest eigenvalue to shrink, we directly penalize covariance extent **along the inferred interface normal**:
\[
\mathcal{L}_{\text{thin}} = \sum_i w_i \, \frac{\hat{n}_i^\top \Sigma_i \hat{n}_i}{\operatorname{tr}(\Sigma_i) + \varepsilon},
\]
where
\[
w_i = \|\nabla \mathcal{S}(\mu_i)\|.
\]

This does two things:

- applies the surface-shape prior only near the commitment interface,
- aligns thinness with the inferred normal direction instead of encouraging arbitrary anisotropy.

So unlike global surfel-like regularization, thinness is imposed only where structural commitment is forming.

---

## 7. Total Mapping Objective

At a MonoGS local mapping step, we optimize:
\[
\mathcal{L} = \mathcal{L}_{\text{photo}} + \lambda_{\text{coh}} \mathcal{L}_{\text{coh}} + \lambda_{\text{thin}} \mathcal{L}_{\text{thin}}.
\]

Roles of each term:

- \(\mathcal{L}_{\text{photo}}\): image reconstruction and tracking consistency,
- \(\mathcal{L}_{\text{coh}}\): coherent motion for committed Gaussians,
- \(\mathcal{L}_{\text{thin}}\): interface-localized surface shaping.

The key simplification is that structural commitment itself is updated from **relative photometric stability**, but only the stable tail is allowed to harden into anchors. This keeps the method close to the two-type Gaussian idea without introducing a hard discrete label.

---

## 8. MonoGS-Based Incremental Algorithm

For each local mapping iteration:

1. **Tracking / rendering**  
   Use the current MonoGS map to render the active views and compute the photometric loss \(\mathcal{L}_{\text{photo}}\).

2. **Raw Gaussian update proposal**  
   Compute raw mean updates \(\Delta \mu_i^{\text{photo}}\) and mean-gradient magnitudes \(g_i\).

3. **Build commitment-aware fields**  
   Using current \(s_i\), build \(\mathcal{S}(x)\), \(U(x)\), and \(\nabla \mathcal{S}(\mu_i)\).

4. **Commitment-modulated Gaussian optimization**  
   Optimize the Gaussian parameters under
   \[
   \mathcal{L}_{\text{photo}} + \gamma^{(t)} \left(\lambda_{\text{coh}} \mathcal{L}_{\text{coh}} + \lambda_{\text{thin}} \mathcal{L}_{\text{thin}}\right).
   \]
   In the current implementation, this is how the idealized commitment-modulated motion is realized in practice.

5. **Detached commitment update**  
   Recompute or reuse the latest \(g_i\), convert them to stable-tail proposals
   \[
   q_i = \operatorname{clip}\left(\frac{(1-r_i) - \tau}{1-\tau}, 0, 1\right),
   \]
   and update
   \[
   s_i \leftarrow (1-\alpha\gamma^{(t)})s_i + \alpha\gamma^{(t)} q_i.
   \]
   During the initial delay period, this reduces to no commitment update at all.

6. **Map management**  
   Use \(s_i\) as a map-lifecycle signal:
   - low-\(s_i\) Gaussians remain exploratory and can be pruned or replaced more aggressively,
   - high-\(s_i\) Gaussians form the structural-anchor subset and are protected more conservatively once the structural ramp is fully active,
   - newly spawned Gaussians start with low \(s_i\).

---

## 9. Why This is a Better Fit for MonoGS than Offline 3DGS

The same idea is more natural in MonoGS than in offline batch 3DGS because SLAM already supplies a notion of:

- tentative vs. mature map elements,
- repeated observation over time,
- structural persistence under viewpoint change,
- ongoing correction after pose updates.

So the method is best understood not as “detecting two ontological Gaussian types,” but as **estimating the structural commitment of each Gaussian in an incremental map**.

This makes the interpretation of \(s_i\) much stronger:

- in offline 3DGS, \(s_i\) is a geometric-role hypothesis,
- in MonoGS, \(s_i\) is a live confidence variable for map structure.

---

## 10. Main Claims

This MonoGS-based formulation makes four claims:

1. **Not all Gaussians should be optimized identically.**  
   Some should remain flexible; others should gradually become structural support.

2. **Structural commitment should be soft and incremental.**  
   Commitment is an online state variable, not a hard class label.

3. **Committed Gaussians should move coherently.**  
   A commitment-weighted consensus field stabilizes the map better than independent damping.

4. **Surface shaping should be localized to the commitment interface.**  
   Thinness should emerge where structure is consolidating, not be imposed globally.

---

## 11. Minimal Hyperparameter Version

A minimal version of the method uses only:

- \(\alpha\): commitment EMA update rate,
- \(\lambda_{\text{coh}}\): coherence regularization weight,
- \(\lambda_{\text{thin}}\): thinness regularization weight.

This keeps the formulation significantly simpler than threshold-heavy phase-switching or evidence-mixture approaches.

---

## 12. Suggested Name

A clean way to present the method is:

**Structural Commitment for Gaussian Splatting SLAM**

or, if you want the phase metaphor to remain visible:

**Progressive Solidification for MonoGS**

The first is safer and more precise; the second is more stylistic.
