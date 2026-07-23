# D1 paired-query bootstrap results

SciFact eval split: 60 fixed queries; 10,000 deterministic paired draws.
Recovery is the ratio of resampled query means. All methods, floor, and oracle use the same draws.

Floor NDCG@10 = 0.0000; oracle = 0.8076; gap = 0.8076.

| Method | NDCG@10 (95% CI) | ΔNDCG vs floor (95% CI) | Recovery (95% CI) |
|---|---:|---:|---:|
| ridge | 0.6812 [0.5838, 0.7731] | 0.6812 [0.5838, 0.7731] | 84.3% [77.0%, 91.0%] |
| closed-form orthogonal Procrustes | 0.6647 [0.5658, 0.7569] | 0.6647 [0.5658, 0.7569] | 82.3% [74.4%, 89.7%] |
| residual MLP | 0.6593 [0.5619, 0.7512] | 0.6593 [0.5619, 0.7512] | 81.6% [74.2%, 88.6%] |
| C3a | 0.1570 [0.0749, 0.2488] | 0.1570 [0.0749, 0.2488] | 19.4% [9.4%, 30.6%] |

## Preregistered paired contrasts

| Contrast | ΔNDCG (95% CI) | raw p | Holm p | Reject at 0.05 |
|---|---:|---:|---:|:---:|
| ridge minus mlp | 0.0219 [-0.0333, 0.0773] | 0.430957 | 0.430957 | no |
| ridge minus c3a | 0.5242 [0.3968, 0.6411] | 0.00019998 | 0.00039996 | yes |

G2 power gate: **FAIL**. The preregistered primary ridge−MLP half-width is 0.0553 (threshold ≤ 0.015). The median across both preregistered contrasts is a secondary diagnostic at 0.0887.

A rough 1/√n extrapolation from 60 queries suggests approximately 817 queries (about 800+) to reach the 0.015 ridge−MLP half-width. This is a scale estimate, not a prospective power calculation; variance must be re-estimated in a pilot.

## Leakage sensitivity of the literal waypoint headline

The literal 600-document fit contains 28 of the 62 eval-positive documents (45.2%). The leakage-safe full-600 fit contains 0. The literal ridge headline (84.3%) is therefore a leaky-fit point estimate, not a clean held-out estimate.

| Method | Literal waypoint recovery | Leakage-safe full-600 recovery | Inflation (pp) |
|---|---:|---:|---:|
| ridge | 84.3% | 78.9% | +5.4 |
| closed-form orthogonal Procrustes | 82.3% | 73.9% | +8.4 |
| residual MLP | 81.6% | 67.5% | +14.1 |

The primary ladder/bootstrap reproduces the literal waypoint recipe. The full-600 sensitivity and all learning-curve document fits exclude eval-positive documents; the two protocols must not be conflated.

C3a scalar audit: the old 0.1609 scalar implied 19.9% recovery on this split, but the actual frozen per-query score is 0.1570 (19.4%).
This is a supersession, not a tolerance pass: the recovery difference from the old cross-seed 0.200 reference is 0.0056, which exceeds the nominal 0.001 threshold.
