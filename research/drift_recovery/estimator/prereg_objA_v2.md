# Obj A v2 pre-registration — powered `oracle_margin_mean` validation

**Frozen at:** 2026-07-14T01:05:39Z  
**Protocol ID:** `objA-oracle-margin-block-loo-v2`  
**Ordering lock:** this document and `prereg_objA_v2.json` are frozen and hash-locked before any of the six v2 packs is built. The v1 preregistration, packs, and outputs remain untouched.

## Primary, target, metrics, and nulls

These are identical to v1. The primary predictor is an univariate ordinary least-squares model with intercept using only `oracle_margin_mean`. The target is ridge oracle-gap recovery

`R = (ridge_ndcg - floor_ndcg) / (oracle_ndcg - floor_ndcg)`.

Predictions are not clipped. The primary metric is block-leave-one-out Spearman(`R_hat`, `R`), computed as Pearson correlation of average-tie ranks over the concatenated out-of-fold cell predictions. Block-LOO MAE over those identical predictions is also required. A constant ranked vector produces an undefined Spearman and cannot win a strict comparison. No multi-feature predictor, `oracle_margin_q50`, or `oracle_margin_sign_rate` fallback is permitted.

The frozen nulls are fit on the exact same training cells in every fold:

1. `oracle_gap_ols`: univariate OLS with intercept from `oracle_gap` to `R`.
2. `mean_R`: the mean `R` of the fold's training cells.

The success bar is the v1 AND-bar: margin must have **strictly greater Spearman and strictly lower MAE** than each null under both whole-dataset-block LOO and whole-encoder-family-block LOO. Equality, an undefined metric, or winning only one metric is a failure. No held-out block member may enter the primary fit or either null fit.

The locked descriptive incremental test is also unchanged: average-rank margin, `R`, and gap; separately residualize ranked margin and ranked `R` on an intercept plus ranked gap; then Pearson-correlate those residuals. Report partial-Spearman(margin, `R` | gap), raw Spearman(margin, `R`), raw Spearman(gap, `R`), and their difference.

## Frozen 4×3 regime matrix

Old/store encoder everywhere: `sentence-transformers/all-MiniLM-L6-v2`. Anchor fraction everywhere: **0.40**. Seed everywhere: **42**. Dataset recipe everywhere: merged-harness deterministic slice with at most 100 queries and exactly the existing **1,200-document sampled slice** contract.

| Dataset | `all-mpnet-base-v2` | `BAAI/bge-large-en-v1.5` | `intfloat/e5-base-v2` |
|---|---|---|---|
| SciFact | reuse `scifact_minilm_to_mpnet` | reuse `scifact_minilm_to_bge_large` | build `scifact_minilm_to_e5_base_v2_af040` |
| NFCorpus | reuse `nfcorpus_minilm_to_mpnet` | reuse `nfcorpus_minilm_to_bge_large` | build `nfcorpus_minilm_to_e5_base_v2_af040` |
| FiQA2018 | reuse `fiqa2018_minilm_to_mpnet_af040` | reuse `fiqa2018_minilm_to_bge_large_af040` | build `fiqa2018_minilm_to_e5_base_v2_af040` |
| ArguAna | build `arguana_minilm_to_mpnet_af040` | build `arguana_minilm_to_bge_large_af040` | build `arguana_minilm_to_e5_base_v2_af040` |

Exactly those six registered packs may be newly built, sequentially, with `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`. Every new pack must use the existing plain merged-harness encode path, freeze per-query floor/oracle/ridge scores, carry a leakage-safe fit index with no eval-positive qrel documents, and pass `assert_harness_parity(atol=1e-12)`.

`intfloat/e5-base-v2` receives the same plain text inputs as mpnet and bge-large: no `query:` or `passage:` prefixes. This may understate e5's absolute retrieval quality, but it preserves the encoder-swap protocol across families; the estimator predicts recovery for the swap that actually occurred.

## Degenerate cells and frozen labels

A pack whose oracle gap is non-positive is excluded and logged, exactly as in the existing `run_estimator.py` gate. It is not imputed and does not contribute to cell, independent-pair, dataset-block, or encoder-family-block counts.

Only these labels are permitted:

- **POSITIVE** if the frozen AND-bar is met with at least 4 available dataset blocks and at least 3 available encoder-family blocks.
- **PROMISING-BUT-UNDERPOWERED** if the frozen AND-bar is met with fewer than 4 dataset blocks or fewer than 3 encoder-family blocks, including honest build/degeneracy losses.
- **NEGATIVE** if the frozen AND-bar fails.

No other verdict label is allowed. The report must separately state whether the aggregate dataset-block analysis still beats gap-only at four dataset blocks, whether the fully novel ArguAna holdout agrees with the aggregate, whether the fully novel e5-family holdout agrees with the aggregate, and the within-held-block Spearman/MAE tie check for both new blocks.
