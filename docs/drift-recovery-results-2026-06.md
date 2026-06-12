# Drift Recovery Results - June 2026

Source artifacts: `experiment_runs/drift-recovery/scifact_C*_{rotation,noise}_seed*.json`.
Matrix: SciFact, conditions C0-C4, drift fraction 0.5, rotation angle 25 degrees, noise sigma 0.05, seeds 42/1337/7, `--max-queries 100 --sample-docs 1200 --cycles 12`.
Device recorded in run configs: cuda.

## Results Table

| Condition | Recovery@12 (rot) | Final NDCG (rot) | Recovery@12 (noise) | Final NDCG (noise) | Mean correction norm |
|---|---:|---:|---:|---:|---:|
| C0 | 3/3 (mean cycle 1.00) | 0.814730 +/- 0.006692 | 0/3 | 0.684292 +/- 0.021042 | n/a |
| C1 | 3/3 (mean cycle 1.00) | 0.814730 +/- 0.006692 | 0/3 | 0.684292 +/- 0.021042 | n/a |
| C2 | 3/3 (mean cycle 1.00) | 0.829212 +/- 0.010032 | 3/3 (mean cycle 1.00) | 0.829212 +/- 0.010032 | n/a |
| C3 | 3/3 (mean cycle 1.00) | 0.814820 +/- 0.006771 | 0/3 | 0.685332 +/- 0.020038 | 0.010000 |
| C4 | 3/3 (mean cycle 1.00) | 0.814730 +/- 0.006692 | 0/3 | 0.684332 +/- 0.021029 | 0.000189 |

## Primary Comparison

- rotation: C3 final NDCG 0.814820; C1 0.814730; C2 0.829212. C3 recovery@12 3/3; C1 3/3; C2 3/3.
- noise: C3 final NDCG 0.685332; C1 0.684292; C2 0.829212. C3 recovery@12 0/3; C1 0/3; C2 3/3.

## Plots

- `experiment_runs/drift-recovery/scifact_rotation_ndcg_mean_std.png`
- `experiment_runs/drift-recovery/scifact_noise_ndcg_mean_std.png`

## Honesty Notes

- All table values above are computed from completed JSON artifacts in this branch.
- The initial online model-load sanity check hit the known Hugging Face SSL verification issue on this host. The matrix used `HF_HUB_OFFLINE=1` with the local model and dataset cache; SSL verification was not disabled.
- C3/C4 artifacts distinguish `sedimentation_attempted` from `correction_applied`; runs where the engine found no collapse candidates are not counted as applied corrections merely because a cycle was attempted.

