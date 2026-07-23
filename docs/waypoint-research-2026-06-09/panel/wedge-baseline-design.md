# Strongest-Linear-Baseline Design for the WITHIN-SPACE Wedge Test

**Status:** DESIGN DRAFT — not frozen, no data touched, zero GPU spent.
**Author:** baseline-design agent (panel task), 2026-07-19.
**Scope:** this document specifies **the opponent**, not the whole experiment. It defines (a) the
strongest linear readout the nonlinear operator must beat, (b) the tuning protocol for it, (c) an
explicit capacity/information **parity contract**, (d) the **power math that must come first**, and
(e) the pre-registered conditions under which the linear baseline **should** win.
**Provenance:** replacement cell recommended by the HI-1 design red-team
(`panel/out_HI1_design_redteam.txt`, verdict CUT, §C.8) and accepted in
`panel/prereg-harmonic-invariance-HI1-draft.md`.

---

## 0. The one-sentence governing rule

> **HI-1 died of three things: a bar set before the power math, a baseline weaker than the one we
> already ship, and a treatment with six knobs against a baseline with one. This document exists to
> make all three impossible in the wedge test.**

Consequence, stated up front so it cannot be quietly dropped: **the primary metric recommended here
is not an accuracy delta.** It is a *minimum-dimension ratio* (§4). If the wedge-test chair insists
on "accuracy difference at one fixed `d`," the baseline design below is still correct but the power
problem that killed HI-1 returns immediately, and this document's recommendation flips to **CUT**.

---

## 1. What the task actually is (baseline strength is task-dependent)

The within-space wedge is a statement about **readout of superposed features from one frozen
representation**. Fix the generative model that the theory is about:

```
f  ∈ R^m      ground-truth feature vector, k-sparse (support uniform over m choose k)
             nonzero entries drawn from a fixed, frozen amplitude distribution
D  ∈ R^{d×m}  frozen overcomplete dictionary, m > d, columns unit-norm
ε  ∈ R^d      isotropic observation noise, variance σ_ε²
x  = D f + ε  the frozen "embedding"
```

**Readout task (primary):** given `x`, recover a *designated* feature coordinate `f_j` — either its
magnitude (regression) or its activity `1[f_j ≠ 0]` (detection). Averaged over `j` and over frozen
dictionary seeds.

This is the exact object of the Garg–Kleinberg–Peng gap: nonlinear (ℓ1 / greedy) decoding succeeds at
`d = O(k log(m/k))`; linear accessibility requires `d = Õ(k² log m)`, with a matching lower bound.
*(Citations in this document are as supplied by the session's deep-research pass — arXiv:2602.11246,
arXiv:2405.14860, arXiv:2512.05534, arXiv:2510.02348. I did not independently re-verify them and this
design does not depend on their exact numbers, only on the qualitative claim of a gap.)*

Two definitional points that decide which baselines are even admissible:

1. **"Linear" means the map `x ↦ ŷ` is affine.** A LASSO *probe* (ℓ1 on the probe weights `w`) is
   still a linear readout and belongs in the linear class. ℓ1 *decoding* (ℓ1 on the reconstructed
   feature vector `f̂`, i.e. `argmin ‖f‖₁ s.t. Df ≈ x`) is **not** linear in `x` and belongs to the
   treatment. Confusing the two is the single easiest way to fake this experiment in either
   direction. Freeze this sentence.
2. **The theorem is about `d`, not about parameter count.** The nonlinear decoder can have *zero*
   fitted parameters while the linear probe has `d`. Parameter-count parity is therefore the *wrong*
   parity axis here (§5).

---

## 2. Enumerated linear readouts, ranked

| # | Method | What it is | Strength on this task | Verdict |
|---|---|---|---|---|
| 1 | **Oracle linear (GLS / whitened matched filter)** | `w_j ∝ (D Σ_f Dᵀ + σ_ε² I)^{-1} d_j`, computed in **closed form from the known generative parameters** | The **theoretical ceiling** of the entire linear class for this model. No estimation error at all. | **PRIMARY BAR** |
| 2 | **Whitened / shrinkage-ridge probe (fitted)** | `ŵ_j = (X̃ᵀX̃ + λI)^{-1} X̃ᵀ y_j`, centered, λ on DEV | Sample plug-in of #1; converges to it as `n_fit → ∞`, `λ → 0`. Best *fitted* linear estimator. | **PRIMARY FITTED ARM** |
| 3 | **Regularized (shrinkage) LDA / Fisher** | Difference-in-means whitened by pooled within-class covariance, Ledoit–Wolf or DEV-tuned shrinkage | For the *detection* variant this is the same estimator as #2 up to a monotone rescaling → identical AUC. Include as an equivalence check, not a separate hope. | Include (tie-check) |
| 4 | **Logistic probe with L2, `C` on DEV** | MLE, same feature space | Can beat #2/#3 in the tails / under label imbalance. Cheap. | Include |
| 5 | **Reduced-rank regression / PLS / CCA** | Rank-`r` constrained multi-output linear map, `r` on DEV | Strictly a *constrained* #2. Can only win by **variance reduction at small `n_fit`**; never at large `n_fit`. Its real job is to remove the excuse "your baseline overfit." | Include (small-n insurance) |
| 6 | **LASSO / elastic-net probe** | ℓ1 on probe weights | Helps only if the optimal `w_j` is sparse in the *ambient* basis. Under superposition with a dense whitening it generally is not. Low expected value, cheap to run. | Include (cheap) |
| 7 | **Plain OLS linear probe** | No regularization | Dominated by #2 at every `λ` grid point including `λ→0`; ill-conditioned as `d → n_fit`. | Include only as a floor |
| 8 | **Orthogonal Procrustes** | Rotation-only map | Not applicable (this is a probe task, not a map-alignment task). Was the weak arm that got HI-1 faulted. | **Excluded** |

### 2.1 Which is strongest, and why

**#1 (oracle linear) is the strongest, and it must be the bar.** Justification:

- For the model in §1, the minimum-MSE **linear** estimator of `f_j` from `x` is
  `w_j = Σ_x^{-1} Σ_{x,f_j}`. With uniform-support k-sparsity, `Σ_f = E[ffᵀ]` is diagonal
  (`(k/m)·σ_f²·I`), so `Σ_x = D Σ_f Dᵀ + σ_ε² I` and `Σ_{x,f_j} = (k/m)σ_f² d_j`. Hence
  **`w_j ∝ (D Σ_f Dᵀ + σ_ε² I)^{-1} d_j`** — a closed-form whitened matched filter. Scale is
  irrelevant for AUC/rank metrics, so the direction is all that matters.
- This is *exactly* the estimator the compressed-sensing lower bound is about. If the nonlinear
  operator cannot beat it, **there is no wedge in this setting** — full stop.
- It removes the "you beat a badly-fit probe" escape entirely: it has **no fitting error, no
  hyperparameter, and no split**. A reviewer can verify it with a one-line linear solve.

**#2 is the strongest *fitted* linear arm** and must be run alongside #1 for a different reason: the
gap `#1 − #2` measures how much of any observed "wedge" is really **finite-sample estimation error on
the linear side**. That gap is a *gate*, not a result (§6, G3).

**Everything else is insurance against a specific accusation** (overfitting → #5/#7; wrong loss →
#4; wrong basis → #6). They are cheap; run them; take the DEV-best as the reported fitted-linear arm.

> **Freeze:** the reported linear baseline is
> **`LIN = max over {ridge, shrinkage-LDA, L2-logistic, RRR/PLS, elastic-net, OLS} selected on DEV`**,
> and the primary bar is **`LIN_ORACLE` (#1)**. Both are reported. A treatment win requires beating
> **both**.

---

## 3. Tuning protocol (DEV only, TEST never)

1. **Three disjoint splits per seed:** `FIT` (fit probe weights), `DEV` (choose every
   hyperparameter, and locate `d*` — §4), `TEST` (single evaluation after freeze). Split by item
   index from the seed's RNG stream; frozen before any fitting.
2. **Standardization** (centering/scaling) is estimated on `FIT` only and applied unchanged to
   `DEV`/`TEST`. Same for both arms.
3. **Every hyperparameter is chosen on `DEV` independently per `(seed, d, method)` cell.** No global
   λ tuned on the winning cell and reused. This is the specific leak that would let a reviewer call
   the baseline hand-tuned.
4. **`TEST` is touched once per `(seed, d, method)`**, after §4's `d*` is already determined on `DEV`.
   `TEST` may not be used to select `d*`, `λ`, `r`, `τ`, or which linear arm is reported.
5. **Grids are declared in the frozen prereg**, not chosen after looking:
   `λ ∈ 10^{-6..+3}` (19 log-spaced), `r ∈ {1,2,4,8,…,min(d,m)}`, `C` likewise 19 log-spaced.
6. **Seeds:** dictionary seed, data seed, and split seed are independent streams (the pattern already
   used in `research/drift_recovery/regimes/sparse_local_nonaffine.py` — `SeedSequence` with distinct
   tags — so changing one knob cannot silently change the corpus).

---

## 4. Power math FIRST — and the metric it forces

### 4.1 Why the obvious metric is a repeat of HI-1

"Accuracy of nonlinear minus accuracy of linear at one fixed `d`" is a small paired difference. This
program's own artifacts show what that costs: D1 paired contrast half-width **0.055 at n=60**;
rung-16 REPORT deltas of **−0.003 / −0.007** with half-widths around **0.05**. A design whose whole
effect lives inside its own confidence interval is a guaranteed fail-closed. **Do not do this again.**

### 4.2 The metric that has power: minimum sufficient dimension

The theorem is a statement **about `d`**. So measure `d`.

- Sweep `d` on a **geometric grid** with ratio `2^{1/4} ≈ 1.19` (fine enough that grid quantization
  contributes SD ≈ `0.25/√12 ≈ 0.07` on the log2 scale — negligible).
- For each `(seed, method)`, define **`d*` = the smallest grid `d` at which the DEV objective reaches
  a pre-frozen threshold `τ`** (primary `τ`: mean per-feature detection AUC ≥ **0.95**; robustness
  band `τ ∈ {0.90, 0.95, 0.99}` reported, 0.95 primary and binding).
  `d*` is located on **DEV for both arms symmetrically** — no cross-arm bias. TEST accuracy at `d*`
  is reported as a confirmation, and the DEV/TEST disagreement rate is reported.
- **Primary statistic, paired within seed:**
  `Δ_s = log2 d*_LIN_ORACLE(s) − log2 d*_NONLIN(s)`.
- Theory's prediction: `Δ ≈ log2(k)` up to logs — with `k = 8`, `Δ ≈ 3`. That is an **order-of-
  magnitude effect**, not a 0.01 sliver. That is the whole reason this metric is the right one.

### 4.3 The actual numbers

Replication unit = **generative seed** (not query). Synthetic data means seeds are **free** —
this is the structural difference from HI-1, whose `n` was capped at ~9–27 real queries.

Paired two-sided test, `α = 0.05`, power 0.80:
`S ≥ ((z_{.975}+z_{.80}) · SD(Δ) / Δ_true)² = (2.802 · SD/Δ)²`.

| SD(Δ) across seeds | Seeds `S` needed to detect **Δ=1.0** (2× dimension saving) | Seeds `S` needed to detect **Δ=0.585** (1.5×) |
|---|---|---|
| 0.50 | **2** | **6** |
| 1.00 | **8** | **23** |
| 1.50 | **18** | **52** |
| 2.00 | **32** | **92** |

Even at a pessimistic `SD = 2.0`, `S = 32` seeds detects a 2× effect — and the theoretical prediction
is ~8×. **The bar is clearable at achievable `n`.** This is the test HI-1 could not pass.

**Cost:** each cell is a `d×d` ridge solve (`d ≤ 512`) plus an OMP/ℓ1 decode. `S=40 × ~20 d-values ×
~8 methods` is tens of thousands of small CPU solves — **minutes to low hours, zero GPU.**

### 4.4 The bar, derived from the above (not asserted before it)

> **WIN** requires *all* of:
> 1. paired-bootstrap 95% CI on mean `Δ` has **lower bound > 0**, **and**
> 2. point estimate `Δ ≥ 1.0` (**≥ 2× dimension saving** — the pre-declared minimum interesting
>    effect; a 1.05× saving is not a wedge), **and**
> 3. the same holds against `LIN` (fitted) as well as `LIN_ORACLE`, **and**
> 4. gates G1–G4 (§6) all pass.

**Mandatory pilot-then-freeze:** `SD(Δ)` is unknown today. Run a **pilot of `S₀ = 8` seeds**,
estimate `SD`, read `S` off the table above, freeze `S`, then run the confirmatory set on **fresh
seeds** (pilot seeds are discarded from the confirmatory analysis). **If the table demands `S > 200`,
the effect is too fragile to be worth claiming → CUT.** Writing the CUT trigger into the protocol
before the pilot is the entire point.

---

## 5. THE PARITY CONTRACT (a reviewer must be able to tick every line)

Parity here is on **information and tuning**, *not* on parameter count (§1.2 — the theorem is about
`d`; the nonlinear decoder may legitimately have zero fitted parameters).

| # | Parity item | Requirement | How a reviewer checks it |
|---|---|---|---|
| **P1** | **Identical data** | Both arms see byte-identical `FIT`/`DEV`/`TEST` matrices and identical index sets. | Hash `X_fit, X_dev, X_test` and the index arrays per arm; assert equality in the harness. |
| **P2** | **Identical `d`** | At every grid point both arms consume the *same* `d`-dimensional `x`. No arm gets an extra projection, extra dimensions, or a different preprocessing. | Assert shapes and preprocessing-config hash equality. |
| **P3** | **Identical tuning budget** | Both arms get **the same number `T` of DEV-objective evaluations** (recommend `T = 64`). The linear arm needs fewer — so it must **spend the surplus on a richer grid** (more λ, more `r`, more arms). Unspent budget on the linear side is a design defect, not a courtesy. | Harness logs an evaluation counter per arm per cell; assert `T_lin == T_nonlin`. |
| **P4** | **Identical prior-knowledge tier** | Declare the tier once and apply it to **both** arms: **T0** = nothing known; **T1** = sparsity `k` known; **T2** = dictionary `D` known. If the nonlinear decoder is OMP/ℓ1 **with `D`**, it is at T2 → the linear arm must also get `D`, i.e. `LIN_ORACLE` (§2.1). **This is where most claimed wedges evaporate and it is non-negotiable.** | Tier recorded in the manifest; grep the two implementations for use of `D`, `k`, `σ_ε`, `Σ_f`. |
| **P5** | **Identical objective and metric** | Same `τ`, same AUC/MSE definition, same per-feature averaging, evaluated on the same items. | One shared metric function called by both arms. |
| **P6** | **Identical seeds and pairing** | `Δ_s` is computed within a seed; both arms run on every seed; no seed dropped for one arm only. | Assert equal seed sets; no post-hoc exclusions. |
| **P7** | **No test-set access for either arm** | §3.4 applies symmetrically. | Single `evaluate_on_test()` call site, invoked after `d*` is frozen. |
| **P8** | **Compute is recorded, not equalized** | Wall-clock and FLOPs logged for both arms and reported. Compute parity is *not* required (it would handicap the cheap side) but hiding a 1000× compute asymmetry is a material omission. | Timing columns in the results artifact. |
| **P9** | **Baseline sufficiency** | `n_fit` must be large enough that `LIN` (fitted) is close to `LIN_ORACLE` — otherwise "the wedge" is really "the probe was undertrained." Gate G3 enforces it. | Report `d*_LIN − d*_LIN_ORACLE` per seed. |
| **P10** | **Symmetric knob disclosure** | Every free knob on **each** side is listed in the prereg with its grid. If the nonlinear side has 6 knobs and the linear side has 1, the design is rejected before running (this is literally what sank HI-1). | Knob table in the frozen prereg; count them. |

---

## 6. Gates (run order, all pre-frozen)

- **G1 — Negative control / leakage check.** At `d < k log(m/k)` (below the information-theoretic
  floor) **neither** arm may reach `τ`. If the nonlinear arm succeeds below the floor, the harness
  leaks → **INVALID**, fix before anything else.
- **G2 — Positive control / linear must win.** In a regime where linear is provably adequate
  (`k = 1`, or `m ≤ d` i.e. no superposition), the linear arm must reach `τ` at a `d*` no larger than
  the nonlinear arm's. If the nonlinear arm wins here, the harness is biased → **INVALID**.
- **G3 — Baseline-sufficiency gate.** If `d*_LIN` is materially worse than `d*_LIN_ORACLE`
  (pre-frozen: `> 0.5` on the log2 scale, i.e. `> 1.4×`), the fitted linear arm is
  **estimation-limited, not accessibility-limited**. Increase `n_fit` and re-run. If no achievable
  `n_fit` closes it, the experiment cannot separate "wedge" from "small-sample" → **CUT**.
- **G4 — Band existence.** The wedge is predicted only in the band
  `k log(m/k) ≲ d ≲ k² log m`. Confirm the swept `d`-grid spans that band for the chosen `(m, k)`. If
  the grid does not span it, the result is uninformative regardless of sign.

---

## 7. When the LINEAR BASELINE SHOULD WIN (pre-registered falsifiers)

A test that can only come out one way is worth nothing. These are declared **now**, before running,
and each has a stated interpretation so neither outcome can be re-narrated afterwards.

| ID | Condition | Why linear should win there | Interpretation if observed |
|---|---|---|---|
| **F1** | `d ≥ Õ(k² log m)` (top of the grid) | Above the linear accessibility threshold the theorem says linear *is* sufficient. `Δ → 0`. | **Expected.** Confirms the harness. Not a wedge failure. |
| **F2** | `k` large relative to `d` (dense features), or `m ≤ d` (no superposition) | With no sparsity there is nothing for ℓ1 to exploit; the whitened matched filter is optimal. | **Wedge does not apply here.** Scope statement, not a result. |
| **F3** | `LIN_ORACLE` matches the nonlinear arm's `d*` **inside** the predicted band (G4 satisfied), `Δ ≈ 0` with a tight CI | The band is exactly where theory predicts a gap. No gap there = no *practically reachable* gap. | **THE WEDGE IS FALSIFIED IN THIS SETTING.** A real, publishable negative. |
| **F4** | A **gradient-trained** nonlinear decoder shows the gap but the **gradient-free** one does not | Would mean the wedge is real but not reachable without gradients. | **The most likely honest outcome**, and it is the one that connects to mini-vec2vec (linear ≥ nonlinear, gradient-free) and to the SAE non-identifiability result. Pre-declared as a **distinct finding**, not as "we failed." |
| **F5** | `LIN` (fitted) ≈ `LIN_ORACLE` **and** both lose to nonlinear only when the nonlinear arm is at a **higher prior tier** (P4 violated) | Then the "wedge" is just extra information. | **NOT a wedge.** Rejected under P4. |

**Symmetrically, the treatment wins only under §4.4.** Both branches are now decision-relevant, which
is precisely what §C.9 of the HI-1 red-team said the prior design lacked.

---

## 8. Repo assets to reuse (and one honest caveat)

- `research/drift_recovery/stats/paired_bootstrap.py` — deterministic paired bootstrap with frozen
  draw indices; directly reusable for the seed-paired `Δ` CI (its `floor`/`oracle` contract is
  drift-specific, so use the `BootstrapDraws` + `_metric_row` machinery, not `paired_query_bootstrap`
  wholesale).
- `research/drift_recovery/stats/multiple_testing.py` — Holm, for the `τ` robustness band.
- `research/drift_recovery/regimes/sparse_local_nonaffine.py` — the *pattern* for deterministic,
  independently-seeded synthetic generation (`SeedSequence` with distinct tags per stream). The regime
  itself is a cluster-warp drift regime, **not** a superposition regime; a new generator is required.
- `research/drift_recovery/methods/affine.py` (`AffineRidgeAdapter`) — the repo's leakage-safe affine
  ridge. **Caveat:** it fits a `d→d` *map* with an unregularized intercept and does **not**
  standardize or scale `λ` by `n`. It is the right *reference implementation style* for the probe, but
  it is not a probe and should not be bent into one. Write a small dedicated probe module; mirror its
  finite-value guards and its refusal to accept labels it should not see.

---

## 9. Bottom line

- **Strongest linear baseline = the closed-form oracle linear (GLS / whitened matched filter)
  `w_j ∝ (D Σ_f Dᵀ + σ_ε² I)^{-1} d_j`, with a DEV-tuned shrinkage-ridge probe as the fitted
  companion.** Anything less and a "win" is a fitting artifact. Orthogonal Procrustes — the arm that
  got HI-1 faulted — is not even applicable here.
- **Parity is on information and tuning, not parameters** (P1–P10). The load-bearing line is **P4**:
  if the nonlinear decoder knows `D`, the linear arm gets `D` too.
- **The power math comes first and it forces the metric.** Accuracy-delta-at-one-`d` is a rerun of
  HI-1's guaranteed fail-closed. The **log2 minimum-dimension ratio** turns an order-of-magnitude
  theoretical prediction into a statistic that `S = 8–32` free synthetic seeds can resolve — on CPU.
  Pilot at `S₀=8`, read `S` off the table, freeze, then run. **`S > 200` ⇒ CUT.**
- **The test can come out either way** (F1–F5 vs §4.4), and the most likely honest outcome (**F4**:
  wedge real, gradient-free operator cannot reach it) is a genuine result rather than a seventh
  pre-narrated negative.
- **This design does not resurrect HI-1** and takes no position on whether the wedge test as a whole
  should be run — only that *if* it is run, this is the opponent it must beat. If the chair will not
  accept the `d*`-ratio metric or the T2-tier oracle baseline, **the honest call is CUT**, because
  under those two refusals the experiment is either underpowered or unfalsifiable, which is exactly
  the pair of defects that killed the previous cell.
