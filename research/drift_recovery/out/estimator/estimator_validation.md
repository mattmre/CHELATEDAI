# D3 recoverability estimator validation

Held-out axis: **regime** (dataset / encoder family / anchor setting). No regime's own measured recovery trains its held-out prediction.

| Held-out regime | Actual R | R_hat | Lower band | Actual bin | Predicted bin |
|---|---:|---:|---:|---|---|
| scifact_minilm_to_mpnet | 0.8434 | 0.6840 | 0.5418 | high | high |
| nfcorpus_minilm_to_mpnet | 0.6471 | 0.7636 | 0.6474 | medium | high |
| scifact_minilm_to_bge_large | 0.7224 | 0.7559 | 0.0506 | high | high |
| nfcorpus_minilm_to_bge_large | 0.6640 | 0.6562 | 0.0000 | medium | medium |

- Held-out regimes: **4**
- Spearman(R_hat, R): **-0.400**
- Three-way bin accuracy: **75.0%** (fixed bins: low <1/3, medium <2/3, high >=2/3)
- Verdict: **VALIDATION-NEGATIVE**

Across 4 held-out regimes, Spearman=-0.400 and 3-way bin accuracy=75.0%; neither locked success threshold was met.

Smallest concrete failure: the bound predicted a violation for **100% of queries in every available regime**, so its inversion-rate feature did not separate observed recoveries ranging from 0.647 to 0.843.

Negative/scoping boundary: the per-query inequality only flags absence of an order-preservation guarantee for one oracle-best relevant/non-relevant pair. It does not prove an inversion, does not model the full top-k ordering, and Gram/margin distortion alone does not determine NDCG@10.

Legacy D1 label note: `scifact_minilm_to_mpnet` uses the locked frozen-pack ridge target R=0.8434. Its estimator feature map is fitted only on `extra_arrays["leakage_safe_fit_idx"]` (safe-refit diagnostic R=0.7891), because the legacy literal D1 target fit contains 28 eval-positive documents.

## C2 adversarial-review addendum (machine-recomputed — `baseline_analysis.json`)

The flat "VALIDATION-NEGATIVE / core signal saturated" reading above is **incomplete**. Two facts,
independently re-derived in `research/drift_recovery/estimator/baseline_analysis.py`:

1. **The shipped preregistered α=10 map does not beat trivial baselines** (leave-one-regime-out, n=4):

   | Predictor | Spearman | MAE | bin acc |
   |---|---:|---:|---:|
   | shipped multi-feature R̂ (α=10) | −0.40 | 0.079 | 0.75 |
   | mean train R | −1.00 | 0.085 | 0.50 |
   | constant 0.7 | n/a | 0.064 | 0.50 |
   | floor NDCG → R | 0.00 | 0.124 | 0.75 |
   | oracle gap → R | +0.60 | 0.072 | 1.00 |

2. **A continuous feature left out of the preregistered `FEATURE_NAMES` ranks R perfectly on n=4:**
   `oracle_margin_mean` (OLS-LOO) → Spearman **+1.00**, MAE **0.033**, bin acc **1.00**;
   `oracle_margin_sign_rate` +0.80; `oracle_margin_q50` +0.60. The binary `predicted_inversion_rate`
   is constant 1.0 (dead).

**Corrected verdict:** the *preregistered* estimator is VALIDATION-NEGATIVE (and loses to trivial
baselines), but recoverability estimation is **not** ruled out — a post-hoc single continuous margin
feature ranks recovery across all 4 held-out regimes. **Caveats (do not over-read):** (a) that feature
was chosen post-hoc after seeing the data; (b) n=4 makes both the −0.40 kill and the +1.0 lead
fragile — a single swap flips Spearman; (c) all R ∈ [0.65, 0.84] so only medium/high bins occur and
75% bin accuracy ≈ the floor-only baseline. Decisive next step: preregister `oracle_margin_mean` and
re-run honest LOO on ≥8–10 regimes.

## Regime availability

- `scifact_minilm_to_mpnet`: **available** — research\drift_recovery\out\d1\scifact_evalsplit_pack
- `nfcorpus_minilm_to_mpnet`: **available** — research\drift_recovery\out\estimator\packs\nfcorpus_minilm_to_mpnet_pack
- `scifact_minilm_to_bge_large`: **available** — research\drift_recovery\out\estimator\packs\scifact_minilm_to_bge_large_pack
- `nfcorpus_minilm_to_bge_large`: **available** — research\drift_recovery\out\estimator\packs\nfcorpus_minilm_to_bge_large_pack
