# Drift Recovery — Post-Bank Head-to-Head (H5) — June 2026

Arena: `query_encoder_swap`. Task SciFact, cycles 12, seeds [42, 1337, 7].

Conditions: C0 frozen (floor) · C2O oracle (ceiling) · **C5 living bank** · **C5s frozen static bank** · **C5r one-shot router**. All scored on the SAME eval subset.

## Head-to-head (mean over seeds)

| Condition | Baseline NDCG | Final NDCG | Std | Applied runs |
|---|---:|---:|---:|---:|
| C0 | 0.818737 | 0.006394 | 0.005594 | 0/3 |
| C2O | 0.818737 | 0.813164 | 0.003981 | 0/3 |
| C5 | 0.818737 | 0.131135 | 0.009802 | 3/3 |
| C5r | 0.818737 | 0.180862 | 0.009034 | 3/3 |
| C5s | 0.818737 | 0.131135 | 0.009802 | 3/3 |

## Verdict (the H5 gate)

- C5 living mean: 0.13113546862429465
- C5s static mean: 0.13113546862429465
- C5r one-shot mean: 0.18086182834701145
- Living beats static (C5 > C5s): **False**
- Living beats one-shot (C5 > C5r): **False**
- **LIVING BANK WINS (beats both): False**

Gate: the living annealed bank must beat BOTH the frozen static bank and the one-shot router with the lifecycle exercised. This verdict is from the real run; it is not asserted until the campaign executes on the GPU.
