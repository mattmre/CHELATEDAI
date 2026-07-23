# Obj A pre-registration — `oracle_margin_mean` scale validation

**Frozen scope date:** 2026-07-10  
**Protocol ID:** `objA-oracle-margin-block-loo-v1`  
**Ordering lock:** this document and `prereg_objA.json` are frozen before either new FiQA2018 pack is built. No result-dependent edits are permitted after pack construction starts.

## Primary hypothesis and metric

The primary hypothesis is that a **univariate** OLS with intercept using only `oracle_margin_mean` predicts ridge oracle-gap recovery

`R = (ridge_ndcg - floor_ndcg) / (oracle_ndcg - floor_ndcg)`

better than both frozen trivial nulls. The single primary metric is block-leave-one-out Spearman(`R_hat`, `R`), computed as Pearson correlation of average-tie ranks over the concatenated out-of-fold cell predictions. Block-LOO MAE over those identical predictions is also required by the must-beat bar. Predictions are not clipped. If a ranked vector is constant, Spearman is recorded as undefined and cannot win a strict comparison.

No multi-feature predictor and no `oracle_margin_q50` or `oracle_margin_sign_rate` fallback is permitted.

## Frozen nulls and strict beat rule

Both nulls are trained on the exact same training cells in every fold:

1. `oracle_gap_ols`: univariate OLS with intercept from `oracle_gap` to `R`.
2. `mean_R`: the mean `R` of that fold's training cells.

“Beat” means **strictly greater Spearman AND strictly lower MAE**. Equality, an undefined metric, or winning only one metric is a failure.

## Binding block-LOO

Two analyses are frozen:

- **Dataset-block LOO:** hold out every cell from one whole dataset, train on the other datasets, and predict every cell in the held-out dataset.
- **Encoder-family-block LOO:** hold out every cell from one whole encoder family, train on the other family, and predict every cell in the held-out family.

No held-out block member may enter the primary fit or either null fit. Out-of-fold predictions are concatenated once per independent dataset×encoder-family pair for metric calculation. Reporting must distinguish:

- independent-pair **block-n**: unique dataset×encoder-family pairs (target 6);
- **cell-n**: logged regimes (target 6 here because no pseudo-replicates are allowed);
- dataset holdout-unit count (cache cap 3); and
- encoder-family holdout-unit count (cache cap 2).

Power claims use independent-pair block-n, never drift-magnitude or anchor-fraction cell count. Anchor-fraction or drift-magnitude variants of the same dataset×encoder pair do not increment block-n and cannot pad the primary analysis.

## Locked partial Spearman

Across the available independent pairs, average-rank `oracle_margin_mean`, `R`, and `oracle_gap`. Separately OLS-residualize ranked margin and ranked `R` on an intercept plus ranked gap, then Pearson-correlate those residuals. This descriptive partial Spearman is the locked incremental-association test; it does not replace block-LOO. Also report raw Spearman(margin, `R`), raw Spearman(gap, `R`), and their difference.

## Frozen offline regimes

Old/store encoder for every regime: `sentence-transformers/all-MiniLM-L6-v2`. Anchor fraction: 0.40. Seed: 42.

| Regime | Dataset | Encoder family | Action |
|---|---|---|---|
| `scifact_minilm_to_mpnet` | SciFact | mpnet | reuse existing D1 pack |
| `scifact_minilm_to_bge_large` | SciFact | bge-large | reuse existing pack |
| `nfcorpus_minilm_to_mpnet` | NFCorpus | mpnet | reuse existing pack |
| `nfcorpus_minilm_to_bge_large` | NFCorpus | bge-large | reuse existing pack |
| `fiqa2018_minilm_to_mpnet_af040` | FiQA2018 | mpnet | build new, sequentially |
| `fiqa2018_minilm_to_bge_large_af040` | FiQA2018 | bge-large | build new, sequentially |

Exactly the two FiQA packs may be newly built. No anchor-fraction pseudo-replicates, uncached dataset, uncached encoder family, or trust-remote-code harness change is registered. Builds run with `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`, conservative batches, and explicit GPU cleanup between regimes. Each new pack must carry a leakage-safe fit index and reproduce its frozen ridge NDCG through `assert_harness_parity(atol=1e-12)`.

If a FiQA oracle gap is non-positive, log the exact failure and continue without changing the target or data. If the achieved independent-pair block-n is below 6, the overall verdict is **UNDERPOWERED-NEGATIVE**.

## Frozen success bar and verdicts

The bar is an AND across all eight pairwise metric comparisons: margin must beat each of `oracle_gap_ols` and `mean_R` on both Spearman and MAE under both dataset-block and encoder-family-block LOO.

- If margin fails to beat gap-only on either required metric under either scheme: **NEGATIVE**.
- If any registered pair is unavailable and independent-pair block-n < 6: **UNDERPOWERED-NEGATIVE**.
- If every strict comparison passes with all six pairs: **PROMISING-BUT-UNDERPOWERED**, never “validated” or unqualified positive, because the cache supplies only three dataset holdout units and two encoder-family holdout units.

FiQA2018 is the decisive held-out-dataset robustness check. If the n=4 margin ordering collapses, approaches gap-level performance, or fails the strict gap-only comparison when FiQA is held out, that negative result must be reported plainly.
