# Drift Recovery — Post-Bank Head-to-Head (H5) — June 2026

Arena: `query_encoder_swap`. Task NFCorpus, cycles 12, seeds [42, 1337, 7].

Conditions: C0 frozen (floor) · C2O oracle (ceiling) · **C5 living bank** · **C5s frozen static bank** · **C5r one-shot router**. All scored on the SAME eval subset.

## Head-to-head (mean over seeds)

| Condition | Baseline NDCG | Final NDCG | Std | Applied runs |
|---|---:|---:|---:|---:|
| C0 | 0.597642 | 0.040655 | 0.014475 | 0/3 |
| C2O | 0.597642 | 0.611933 | 0.018032 | 0/3 |
| C3a | 0.597642 | 0.052435 | 0.012268 | 3/3 |
| C3b | 0.597642 | 0.126910 | 0.008241 | 3/3 |

## Verdict (the H5 gate)

- C5 living mean: None
- C5s static mean: None
- C5r one-shot mean: None
- Living beats static (C5 > C5s): **False**
- Living beats one-shot (C5 > C5r): **False**
- **LIVING BANK WINS (beats both): False**

Gate: the living annealed bank must beat BOTH the frozen static bank and the one-shot router with the lifecycle exercised. This verdict is from the real run; it is not asserted until the campaign executes on the GPU.
