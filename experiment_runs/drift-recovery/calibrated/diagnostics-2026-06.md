# Drift Recovery Diagnostics - June 2026

Diagnostics JSON: `experiment_runs/drift-recovery/calibrated/diagnostics-2026-06.json`.
Source artifact count: 30.

## Hypothesis Verdicts

- H1_rotation_too_weak: **refuted** - Rotation C0 mean drop is 7.760% and not all C0 rotation runs recover at cycle 1; noise C0 mean drop is 7.963%.
- H2_correction_barely_moves: **confirmed** - C3 mean correction norms are [0.009999639, 0.009999638]; C4 mean correction norms are [0.000190418, 0.000188623]; detection fired every C3/C4 cycle=True; C3 applied vector updates=False.
- H3_c2_oracle_reembed: **confirmed** - C2 baseline-final absolute diffs across 6 runs: max=0, tolerance=1e-12.

## Baseline vs Final

| Drift | Condition | Baseline NDCG | Final NDCG | Drop % | Recovery@12 |
|---|---|---:|---:|---:|---:|
| rotation | C0 | 0.829212 +/- 0.010032 | 0.764666 +/- 0.023299 | 7.760% | 1/3 |
| noise | C0 | 0.829212 +/- 0.010032 | 0.763149 +/- 0.017778 | 7.963% | 0/3 |
| rotation | C1 | 0.829212 +/- 0.010032 | 0.764666 +/- 0.023299 | 7.760% | 1/3 |
| noise | C1 | 0.829212 +/- 0.010032 | 0.763149 +/- 0.017778 | 7.963% | 0/3 |
| rotation | C2 | 0.829212 +/- 0.010032 | 0.829212 +/- 0.010032 | 0.000% | 3/3 |
| noise | C2 | 0.829212 +/- 0.010032 | 0.829212 +/- 0.010032 | 0.000% | 3/3 |
| rotation | C3 | 0.829212 +/- 0.010032 | 0.764609 +/- 0.023323 | 7.767% | 1/3 |
| noise | C3 | 0.829212 +/- 0.010032 | 0.763342 +/- 0.017833 | 7.940% | 0/3 |
| rotation | C4 | 0.829212 +/- 0.010032 | 0.764666 +/- 0.023299 | 7.760% | 1/3 |
| noise | C4 | 0.829212 +/- 0.010032 | 0.763149 +/- 0.017778 | 7.963% | 0/3 |

## C3/C4 Correction Trace

| Drift | Condition | should_correct | attempted | applied | Mean drift signal | Mean correction norm | Max correction norm |
|---|---|---:|---:|---:|---:|---:|---:|
| rotation | C3 | 36/36 | 36/36 | 0/36 | 0.002034993 | 0.009999639 | 0.009999711 |
| noise | C3 | 36/36 | 36/36 | 0/36 | 0.002052080 | 0.009999638 | 0.009999714 |
| rotation | C4 | 36/36 | 36/36 | 0/36 | 0.002035173 | 0.000190418 | 0.000238469 |
| noise | C4 | 36/36 | 36/36 | 0/36 | 0.002052330 | 0.000188623 | 0.000240747 |

## Severity

| Drift | C0 mean baseline | C0 mean final | Mean drop % | All C0 runs above 95% threshold |
|---|---:|---:|---:|---|
| rotation | 0.829212 | 0.764666 | 7.760% | False |
| noise | 0.829212 | 0.763149 | 7.963% | False |

## Plots

- `experiment_runs/drift-recovery/calibrated/scifact_rotation_ndcg_diagnostics_mean.png`
- `experiment_runs/drift-recovery/calibrated/scifact_noise_ndcg_diagnostics_mean.png`
