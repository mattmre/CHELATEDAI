# Drift Recovery — Query-Encoder-Swap Arena (PR-A4) — June 2026

> [!WARNING]
> **LEGACY_METRIC_LINEAGE_BLOCKED (2026-07-23).** Exact nDCG values,
> comparator ordering, and derived gates below are historical diagnostics, not
> accepted evidence, until corrected regeneration. See
> `docs/research/metric-lineage-repair-protocol-2026-07.md`.

Arena: `query_encoder_swap` (encoder upgrade; the C2 re-embed oracle is defeated).
Manifest: `experiment_runs/drift-recovery/swap/swap-campaign-manifest-2026-06.json`.
Base model: `sentence-transformers/all-MiniLM-L6-v2`; swap model: `all-mpnet-base-v2`.
Task SciFact, anchor_fraction 0.4, cycles 12, seeds [42, 1337, 7].

Conditions: C0 frozen (lower bound) · C2 re-embed-with-original (proven no-op) · C2O oracle re-embed into new space (upper bound) · C3a supervised bounded adapter · C4a supervised unbounded adapter. All scored on the SAME held-out eval subset.

## Main Matrix (mean over seeds)

| Condition | Baseline NDCG | Final NDCG | Std | Recovery@N | Applied runs |
|---|---:|---:|---:|---:|---:|
| C0 | 0.818737 | 0.006394 | 0.005594 | 0/3 | 0/3 |
| C2 | 0.818737 | 0.006394 | 0.005594 | 0/3 | 0/3 |
| C2O | 0.818737 | 0.813164 | 0.003981 | 3/3 | 0/3 |
| C3a | 0.818737 | 0.160864 | 0.003003 | 0/3 | 3/3 |
| C4a | 0.818737 | 0.206279 | 0.026529 | 0/3 | 3/3 |

## C3a Training-Budget Sweep (seed 42)

| Steps | LR | Final NDCG | should_correct | applied | mean norm |
|---:|---:|---:|---:|---:|---:|
| 30 | 0.01 | 0.156964 | 12/12 | 12/12 | 0.515772 |
| 200 | 0.05 | 0.174191 | 12/12 | 12/12 | 0.511576 |
| 1000 | 0.10 | 0.180376 | 12/12 | 12/12 | 0.514221 |
| 2000 | 0.10 | 0.181472 | 12/12 | 12/12 | 0.515058 |

Best budget cell: steps=2000, lr=0.10 (seed-42 final 0.181472). Three-seed confirmation:

| Steps | LR | Mean final NDCG | Std | Recovery@N |
|---:|---:|---:|---:|---:|
| 2000 | 0.10 | 0.165853 | 0.015153 | 0/3 |
