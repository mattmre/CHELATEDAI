# Rung 16 Preregistration — Quant-Aware Routing Plane

Status: **FROZEN PRE-REPORT on 2026-07-14**. The machine-readable authority is
`prereg_rung16.json`. REPORT data may be evaluated once, only after ANCHOR fitting and SELECT
promotion are complete and frozen.

## Primary claim and arenas

The only positive claim is that the full routed plane beats a same-class, same-budget
single-global adapter under a leakage-safe query-encoder swap and survives simulated INT8. Arena A
uses SciFact and K=3 clusters mined from ANCHOR relevant-document vectors. Arena B pools SciFact,
NFCorpus, and FiQA2018 with domain as the route cluster. The multi-domain split is stratified by
dataset. Each dataset contributes at most 100 relevant queries and 1,200 documents, including all
sampled-query relevant documents.

Both arenas use MiniLM document embeddings and `all-mpnet-base-v2` swapped query embeddings through
the same frozen projection (seed 1616). The real campaign requires CUDA.

## Leakage boundary

- ANCHOR: 40%, seed 1616. It alone fits per-route adapters, the single-global adapter, and route and
  global centroids.
- SELECT: 30%. It alone chooses the single-best-route ablation, runs every promotion threshold,
  computes the paired bootstrap CI, and evaluates quant survival. The decision is then locked.
- REPORT: 30%. It receives one frozen evaluation. It cannot fit, select, tune, or promote anything.

The three ID sets must be exhaustive and pairwise disjoint. REPORT access is one-shot and recorded.

## Fixed baselines and adapter budget

The baselines are no-route (floor and honest quant baseline), single-global adapter (primary
comparator), C2O full swapped-encoder document re-embedding (ceiling context only), and
single-best-route (report-only degeneracy ablation selected on SELECT).

Every routed adapter and the global adapter is the same bounded MLP (`min_correction=0.01`,
`max_correction=0.5`), trained on the applicable ANCHOR pairs with InfoNCE, Adam, 2,000 steps, and
learning rate 0.10. Each ANCHOR query contributes one relevance-weighted centroid of its positive
document vectors, avoiding diagonal-InfoNCE false negatives between multiple positives for the same
query. Training uses deterministic seeded minibatches of 64 repeated to exactly 2,000 optimizer
steps. Route adapters do not receive extra steps. A route requires at least five distinct ANCHOR
documents.

Every condition uses the same graded NDCG@10 implementation: gain is `2^relevance - 1`, and IDCG is
computed from the query's full positive qrels rather than only from retrieved relevance values.

## Frozen gates

- Routing uses K=3 and applies a routed adapter only when its centroid cosine exceeds the global
  centroid cosine by at least 0.02; otherwise it falls back to the single-global adapter.
- The paired query bootstrap uses 5,000 resamples, seed 1617, and a two-sided 95% percentile CI for
  `NDCG_plane_fp32 - NDCG_single_global_fp32`.
- The SELECT CI lower bound must be strictly greater than +0.005 NDCG.
- The served quantized plane may not trail no-route SELECT NDCG by more than 0.005.
- Every adapter selected on SELECT, including the global margin fallback when used, must pass
  `QuantizationPromotionGate` against no-route NDCG on that same query subset: FP32 gain must be
  strictly greater than +0.010 NDCG and quantized gain retention must be at least 0.80.
- REPORT must show at least two non-global routed adapters each serving at least 10% of REPORT
  queries. Otherwise the verdict is `DEGENERATE`, regardless of score.

`PROMOTED` requires every SELECT gate plus the REPORT multi-route binding. A non-degenerate SELECT
failure is `FAIL-CLOSED`. REPORT never supplies a rescue path: positive route lift observed only on
REPORT is not promotion evidence.
