# C2 adversarial-review fix report (D3 estimator)

Applied by the chair (Claude) directly, because the Codex agent hit its usage limit mid-task. All
numbers are machine-recomputed by `research/drift_recovery/estimator/baseline_analysis.py`
(→ `out/estimator/baseline_analysis.json`), independently reproducing Grok's C2 recompute; none are
hand-entered. Work confined to `estimator/`, `out/estimator/`, and `tests/test_estimator.py`.

| Fix | What was done | Evidence |
|---|---|---|
| C2-F1 (trivial baselines) | Added LOO trivial-baseline table (mean R, constant 0.7, floor→R, oracle-gap→R) to `D3_REPORT.md` + `estimator_validation.md`. Shipped R̂ is worse on rank than oracle-gap (−0.40 vs +0.60) and worse on MAE than a constant (0.079 vs 0.064). | `baseline_analysis.py:run()`; `baseline_analysis.json` `loo_rows` |
| C2-F2 (continuous signal) | Reported `oracle_margin_mean` OLS-LOO Spearman **+1.00**, MAE **0.033**, bin **1.00** — it is emitted in the feature dump but NOT in `FEATURE_NAMES`. Root-cause text now says the binary inversion rate is dead AND the continuous margin is not; the negative is about THIS calibrator, not the idea. Chose scope-option (b): left `FEATURE_NAMES` frozen (changing it post-hoc = feature selection on the eval set) and disclosed the lead as post-hoc + unvalidated. | `baseline_analysis.json` `loo_rows`, `univariate_spearman_vs_R`; `D3_REPORT.md` |
| C2-F3 (over-regularization) | Documented that α=10 + 8 features + 3 train points shrinks toward the train mean; added the α=0.01 ablation (Spearman 0.00) showing this is a calibrator artifact. | `baseline_analysis.json` (`alpha=0.01` row); `D3_REPORT.md` |
| C2-F4 (n=4 + bins) | Both reports now state n=4 cannot support a high-power kill or a validated positive (Spearman flips on one swap), and that all R ∈ [0.65,0.84] so 75% bin acc ≈ floor baseline. | `D3_REPORT.md`, `estimator_validation.md` |
| C2-F5 (hand-check test) | Added `test_synthetic_pack_bound_slack_and_inversion_are_end_to_end_consistent`: asserts `bound_slack = margin − ‖q‖(‖e_r‖+‖e_j‖)`, `bound_rhs = ‖q‖(‖e_r‖+‖e_j‖)`, inversion ⇔ slack≤0, and cross-checks via the public `order_preservation_slack` / `predict_inversion` API. | `tests/test_estimator.py`; **10/10 tests pass** |
| C2-F6 (safe-label LOO) | The leaky-vs-safe SciFact label mismatch is already disclosed in `estimator_validation.md`; Grok verified the −0.40 LOO holds under R_safe=0.7891 labels too. No numbers changed. | `estimator_validation.md` legacy-label note |

## Revised verdict
Preregistered estimator: **VALIDATION-NEGATIVE** (and beaten by trivial baselines). Recoverability
estimation itself: **NOT ruled out** — a post-hoc single continuous margin feature ranks R across all
4 held-out regimes, but that is post-hoc and n=4-fragile. Decisive next step: preregister
`oracle_margin_mean`, run honest LOO on ≥8–10 regimes.

## Test evidence
`python -m unittest research.drift_recovery.tests.test_estimator -v` → **Ran 10 tests … OK**
(chair-run, offline).
