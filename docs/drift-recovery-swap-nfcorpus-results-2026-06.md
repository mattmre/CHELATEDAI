# Drift Recovery — Query-Encoder-Swap Arena (PR-A4) — June 2026

Arena: `query_encoder_swap` (encoder upgrade; the C2 re-embed oracle is defeated).
Manifest: `experiment_runs\drift-recovery\swap-nfcorpus/swap-campaign-manifest-2026-06.json`.
Base model: `sentence-transformers/all-MiniLM-L6-v2`; swap model: `all-mpnet-base-v2`.
Task NFCorpus, anchor_fraction 0.4, cycles 12, seeds [42, 1337, 7].

Conditions: C0 frozen (lower bound) · C2 re-embed-with-original (proven no-op) · C2O oracle re-embed into new space (upper bound) · C3a supervised bounded adapter · C4a supervised unbounded adapter. All scored on the SAME held-out eval subset.

## Main Matrix (mean over seeds)

| Condition | Baseline NDCG | Final NDCG | Std | Recovery@N | Applied runs |
|---|---:|---:|---:|---:|---:|
| C0 | 0.597642 | 0.040655 | 0.014475 | 0/3 | 0/3 |
| C2 | 0.597642 | 0.040655 | 0.014475 | 0/3 | 0/3 |
| C2O | 0.597642 | 0.611933 | 0.018032 | 3/3 | 0/3 |
| C3a | 0.598189 | 0.053501 | 0.008779 | 0/3 | 3/3 |
| C4a | 0.597642 | 0.054347 | 0.004909 | 0/3 | 3/3 |

## C3a Training-Budget Sweep (seed 42)

| Steps | LR | Final NDCG | should_correct | applied | mean norm |
|---:|---:|---:|---:|---:|---:|
| 30 | 0.01 | 0.051264 | 12/12 | 12/12 | 0.515708 |
| 200 | 0.05 | 0.039031 | 12/12 | 12/12 | 0.506980 |
| 1000 | 0.10 | 0.040058 | 12/12 | 12/12 | 0.508960 |
| 2000 | 0.10 | 0.041228 | 12/12 | 12/12 | 0.509369 |

Best budget cell: steps=30, lr=0.01 (seed-42 final 0.051264). Three-seed confirmation:

| Steps | LR | Mean final NDCG | Std | Recovery@N |
|---:|---:|---:|---:|---:|
| 30 | 0.01 | 0.053501 | 0.008779 | 0/3 |
