# Drift Recovery Results - June 2026

Source artifacts: `experiment_runs/drift-recovery/scifact_C*_{rotation,noise}_seed*.json`.
Matrix: SciFact, conditions C0-C4, drift fraction 0.5, rotation angle 25 degrees, noise sigma 0.05, seeds 42/1337/7, `--max-queries 100 --sample-docs 1200 --cycles 12`.
Device recorded in run configs: cuda.

## Results Table

| Condition | Baseline NDCG | Recovery@12 (rot) | Final NDCG (rot) | Drop (rot) | Recovery@12 (noise) | Final NDCG (noise) | Drop (noise) | Mean correction norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.829212 +/- 0.010032 | 3/3 (mean cycle 1.00) | 0.814730 +/- 0.006692 | 1.739% | 0/3 | 0.684292 +/- 0.021042 | 17.471% | n/a |
| C1 | 0.829212 +/- 0.010032 | 3/3 (mean cycle 1.00) | 0.814730 +/- 0.006692 | 1.739% | 0/3 | 0.684292 +/- 0.021042 | 17.471% | n/a |
| C2 (oracle re-embed upper bound) | 0.829212 +/- 0.010032 | 3/3 (mean cycle 1.00) | 0.829212 +/- 0.010032 | 0.000% | 3/3 (mean cycle 1.00) | 0.829212 +/- 0.010032 | 0.000% | n/a |
| C3 | 0.829212 +/- 0.010032 | 3/3 (mean cycle 1.00) | 0.814820 +/- 0.006771 | 1.728% | 0/3 | 0.685332 +/- 0.020038 | 17.344% | 0.010000 |
| C4 | 0.829212 +/- 0.010032 | 3/3 (mean cycle 1.00) | 0.814730 +/- 0.006692 | 1.739% | 0/3 | 0.684332 +/- 0.021029 | 17.466% | 0.000189 |

## Primary Comparison

- rotation: C3 final NDCG 0.814820; C1 0.814730; C2 0.829212. C3 recovery@12 3/3; C1 3/3; C2 3/3.
- noise: C3 final NDCG 0.685332; C1 0.684292; C2 0.829212. C3 recovery@12 0/3; C1 0/3; C2 3/3.

## Plots

- `experiment_runs/drift-recovery/scifact_rotation_ndcg_mean_std.png`
- `experiment_runs/drift-recovery/scifact_noise_ndcg_mean_std.png`
- `experiment_runs/drift-recovery/scifact_rotation_ndcg_diagnostics_mean.png`
- `experiment_runs/drift-recovery/scifact_noise_ndcg_diagnostics_mean.png`

## Diagnostics

Source: `experiment_runs/drift-recovery/diagnostics-2026-06.json` and
`docs/drift-recovery-diagnostics-2026-06.md`.

- H1 confirmed: rotation drift at fraction 0.5 and 25 degrees is too weak for
  condition discrimination. Frozen C0 rotation drops only 1.739% from baseline,
  and all C0 rotation runs are already above the 95% recovery threshold at cycle
  1. Noise is in the target severity band with a 17.471% C0 drop.
- H2 confirmed: detection fires on every C3/C4 cycle and sedimentation is
  attempted every time, but `correction_applied` is false for all C3/C4 cycles.
  C3 correction norms saturate at about 0.010000, while C4 norms stay near
  0.000189.
- H3 confirmed: C2 returns exactly to the per-run pre-drift baseline in all six
  C2 runs (`max baseline-final diff = 0`), so this condition is relabeled as an
  oracle re-embed upper bound. It re-embeds affected raw text with the original
  frozen model, which undoes synthetic vector drift by construction.

## Honesty Notes

- All table values above are computed from completed JSON artifacts in this branch.
- The initial online model-load sanity check hit the known Hugging Face SSL verification issue on this host. The matrix used `HF_HUB_OFFLINE=1` with the local model and dataset cache; SSL verification was not disabled.
- C3/C4 artifacts distinguish `sedimentation_attempted` from `correction_applied`; runs where the engine found no collapse candidates are not counted as applied corrections merely because a cycle was attempted.
