# D3 report — recoverability estimator (with C2 adversarial-review follow-up)

Built a pure-NumPy feature path over frozen `EmbeddingPack` data: oracle-space margin quantiles,
leakage-safe ridge correction-error norms, and the per-query margin-bound violation rate. A strongly
regularized linear map produces `R_hat`; training-only cross-fitted overprediction residuals produce
its lower band.

Regimes used (4, all real & offline-built, SHA-manifested): `scifact_minilm_to_mpnet`,
`nfcorpus_minilm_to_mpnet`, `scifact_minilm_to_bge_large`, `nfcorpus_minilm_to_bge_large`. Actual
ridge recovery R = {0.843, 0.647, 0.722, 0.664}.

## Verdict (revised after C2): PREREGISTERED CALIBRATOR — VALIDATION-NEGATIVE · one live post-hoc lead

The **preregistered** multi-feature calibrator (8 fixed `FEATURE_NAMES`, ridge α=10) **fails** the
locked bar under honest leave-one-regime-out: **Spearman −0.400, 3-way bin accuracy 75%**. It is not a
circular artifact (LOO trains only on other regimes; features are pack-local; ridge fit is
leakage-safe). But two things make the flat "idea is dead" reading wrong, both surfaced by the C2
adversarial review and independently re-derived here in `baseline_analysis.py`
(→ `out/estimator/baseline_analysis.json`):

1. **The shipped calibrator does not even beat trivial baselines.** So its −0.40 is a *bad calibrator*,
   not evidence that recovery is unpredictable.
2. **A continuous margin feature that was left out of the preregistered set ranks recovery perfectly
   on this n=4** — so the negative is about *this map*, not the *idea*.

### Leave-one-regime-out comparison (machine-recomputed, n=4)

| Predictor | Spearman(R̂,R) | MAE | bin acc |
|---|---:|---:|---:|
| **shipped multi-feature R̂ (α=10, preregistered)** | **−0.40** | 0.079 | 0.75 |
| multi-feature ablation (α=0.01) | 0.00 | 0.125 | 0.75 |
| trivial: mean train R | −1.00 | 0.085 | 0.50 |
| trivial: constant 0.7 | n/a | **0.064** | 0.50 |
| trivial: floor NDCG → R (OLS) | 0.00 | 0.124 | 0.75 |
| trivial: oracle gap → R (OLS) | +0.60 | 0.072 | 1.00 |
| **post-hoc: `oracle_margin_mean` alone (OLS)** | **+1.00** | **0.033** | **1.00** |
| post-hoc: `oracle_margin_q50` alone (OLS) | +0.60 | 0.075 | 0.75 |
| post-hoc: `oracle_margin_sign_rate` alone (OLS) | +0.80 | 0.080 | 0.75 |

The shipped R̂ is **worse on rank than oracle-gap alone and worse on MAE than a constant**. The binary
`predicted_inversion_rate` feature is **constant 1.0** in all 4 regimes (100% predicted violation —
expected under catastrophic drift once queries are unit-normalized and ‖e‖≈0.4–0.5 ≫ typical margins),
so it carries no signal; the calibrator's advertised core feature is dead.

### Why the preregistered map fails (not the idea)
α=10 with 8 features and only 3 training regimes shrinks coefficients toward ~0.01 and the intercept
toward the train mean, so R̂ ≈ a shrunk mean whose residual noise anti-correlates with R (hence −0.40).
The continuous oracle-margin structure it *should* have used — `oracle_margin_mean`, which is emitted
in the feature dump but is **not** in `FEATURE_NAMES` — achieves perfect LOO rank and the lowest MAE.

### The honest caveats on that lead (do not over-read the +1.0)
- **Post-hoc.** `oracle_margin_mean` was chosen *after* seeing the data / the C2 review; this is
  feature selection on the evaluation set. It is a hypothesis, not a validated estimator.
- **n=4 is underpowered in both directions.** With four points, Spearman flips on a single swap and a
  perfect monotone rank is cheap. n=4 **cannot** support a high-power kill *or* a validated positive.
- **Degenerate bins.** All four R ∈ [0.65, 0.84], so only medium/high bins are occupied; the 75% bin
  accuracy of the shipped map ≈ the floor-only baseline, not near-success.

**Net:** the preregistered recoverability estimator, as built, does not predict recovery (VALIDATION-
NEGATIVE, and it loses to trivial baselines). But a single continuous oracle-margin feature ranks R
across all 4 held-out regimes, so recoverability estimation is **not** ruled out — it is
**unvalidated**. The decisive next step is more regimes (≥8–10: more datasets, encoder families, drift
magnitudes, anchor fractions) with `oracle_margin_mean` (and the margin quantiles) **preregistered**,
then an honestly-powered LOO. Gram/margin distortion alone still does not determine NDCG@10 and the
per-query bound is only a loose sufficient condition, so even that test may not close it.

## Most important limitation
n=4 regimes. Everything above — the −0.40 kill of the shipped map and the +1.0 of the single-feature
lead — rests on four points and is a *scoping* result, not a powered one.
