# Obj A v2 report — powered margin-predictor validation

## Build and verdict

Reused all six frozen v1 packs and built only the six registered additions: ArguAna×{mpnet, bge-large, e5} plus SciFact/NFCorpus/FiQA2018×e5, sequentially and fully offline. Every available new pack used the frozen 1,200-document slice recipe, safe fit index, frozen per-query scores, and passed harness parity at 1e-12. No registered cell failed or had a non-positive oracle gap.

**Frozen-label verdict: NEGATIVE.** The AND-bar failed at 4 dataset blocks, 3 encoder blocks, and 12 cells. Dataset-block: margin rho -0.7063, MAE 0.149; gap rho -0.4965, MAE 0.124. Encoder-block: margin rho -0.6853, MAE 0.106; gap rho -0.4476, MAE 0.094.

## Brutal holdout read

Margin **does not beat gap-only on both locked metrics at four dataset blocks**.

ArguAna **agrees** with its aggregate gap-only result: margin rho -1.0000 / MAE 0.293 versus gap rho -0.5000 / MAE 0.120; the within-block rank relation is **loses** and the strict two-metric check **fails**.

E5 **agrees** with its aggregate gap-only result: margin rho -0.8000 / MAE 0.136 versus gap rho -0.6000 / MAE 0.130; the within-block rank relation is **loses** and the strict two-metric check **fails**.

E5 was encoded through the same plain harness path as mpnet/bge, with no `query:` or `passage:` prefixes. That may understate its absolute quality, but it preserves the encoder-swap protocol. The estimator therefore predicts the recovery of the swap actually run, not an E5-optimized retrieval deployment.
