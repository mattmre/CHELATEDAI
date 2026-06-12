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

## Calibrated Severity

Source: `experiment_runs/drift-recovery/calibrated/calibration-manifest-2026-06.json`
and `docs/drift-recovery-calibrated-results-2026-06.md`.

Pre-registered choice rule: among scout cells, choose a setting inside the
8-20% baseline-drop zone closest to 12%; if none lands in-zone, choose the
closest overall and disclose that.

### Scout Cells

| Drift | Fraction | Angle | Sigma | Baseline | Final C0 | Drop % | In 8-20% zone |
|---|---:|---:|---:|---:|---:|---:|---|
| noise | 0.50 | 25.0 | 0.010 | 0.817821 | 0.819284 | -0.179% | False |
| noise | 0.50 | 25.0 | 0.020 | 0.817821 | 0.805277 | 1.534% | False |
| noise | 0.50 | 25.0 | 0.035 | 0.817821 | 0.741620 | 9.317% | True |
| rotation | 0.50 | 35.0 | 0.050 | 0.817821 | 0.754585 | 7.732% | False |
| rotation | 0.50 | 50.0 | 0.050 | 0.817821 | 0.628182 | 23.188% | False |
| rotation | 0.50 | 65.0 | 0.050 | 0.817821 | 0.514974 | 37.031% | False |
| rotation | 0.75 | 35.0 | 0.050 | 0.817821 | 0.794601 | 2.839% | False |
| rotation | 0.75 | 50.0 | 0.050 | 0.817821 | 0.591663 | 27.654% | False |
| rotation | 0.75 | 65.0 | 0.050 | 0.817821 | 0.344693 | 57.852% | False |

Chosen settings: noise uses fraction 0.5 / sigma 0.035 because it is in-zone
and closest to 12%; rotation uses fraction 0.5 / 35 degrees because no rotation
scout cell landed in the 8-20% zone and 7.732% was closest overall.

### Calibrated Full Matrix

| Condition | Baseline NDCG | Recovery@12 (rot) | Final NDCG (rot) | Drop (rot) | Recovery@12 (noise) | Final NDCG (noise) | Drop (noise) |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.829212 +/- 0.010032 | 1/3 | 0.764666 +/- 0.023299 | 7.760% | 0/3 | 0.763149 +/- 0.017778 | 7.963% |
| C1 | 0.829212 +/- 0.010032 | 1/3 | 0.764666 +/- 0.023299 | 7.760% | 0/3 | 0.763149 +/- 0.017778 | 7.963% |
| C2 (oracle re-embed upper bound) | 0.829212 +/- 0.010032 | 3/3 | 0.829212 +/- 0.010032 | 0.000% | 3/3 | 0.829212 +/- 0.010032 | 0.000% |
| C3 | 0.829212 +/- 0.010032 | 1/3 | 0.764609 +/- 0.023323 | 7.767% | 0/3 | 0.763342 +/- 0.017833 | 7.940% |
| C4 | 0.829212 +/- 0.010032 | 1/3 | 0.764666 +/- 0.023299 | 7.760% | 0/3 | 0.763149 +/- 0.017778 | 7.963% |

Calibrated result: C3 still does not beat C0/C1 materially and remains below
the C2 oracle upper bound in both modes. The severity calibration therefore
supports the negative/mechanistic story unless PR-8's pre-registered knob
sweep changes the confirmed three-seed result.

## C3 Knob Sweep

Source: `experiment_runs/drift-recovery/knob-sweep/knob-sweep-manifest-2026-06.json`
and `docs/drift-recovery-knob-sweep-2026-06.md`.

Scope: calibrated rotation setting only, using the chosen rotation scout setting
fraction 0.5 / 35 degrees / sigma 0.05. The 12-cell grid is seed 42 exploratory
selection evidence; the two highest seed-42 cells were then confirmed on seeds
42, 1337, and 7.

Selection policy: top two cells by seed-42 final NDCG; ties are broken by lower
trigger threshold, then profile name, then lower bound epsilon. The bound 0.10 /
trigger 0.15 cells tied the selected cells on seed-42 final NDCG, but were not
confirmed because the pre-declared tie-break selected the lower trigger
threshold. Profile note: `hotter` means the pre-registered aggressive schedule
profile (lower max-temperature cap plus doubled epoch scale), not a higher
temperature cap.

### All Grid Cells

| Bound epsilon | Trigger threshold | Profile | Final NDCG | Recovery cycle | should_correct | attempted | applied |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0.01 | 0.05 | default | 0.754416 |  | 0/12 | 0/12 | 0/12 |
| 0.01 | 0.05 | hotter | 0.754416 |  | 0/12 | 0/12 | 0/12 |
| 0.01 | 0.15 | default | 0.754416 |  | 0/12 | 0/12 | 0/12 |
| 0.01 | 0.15 | hotter | 0.754416 |  | 0/12 | 0/12 | 0/12 |
| 0.05 | 0.05 | default | 0.753156 |  | 0/12 | 0/12 | 0/12 |
| 0.05 | 0.05 | hotter | 0.753156 |  | 0/12 | 0/12 | 0/12 |
| 0.05 | 0.15 | default | 0.753156 |  | 0/12 | 0/12 | 0/12 |
| 0.05 | 0.15 | hotter | 0.753156 |  | 0/12 | 0/12 | 0/12 |
| 0.10 | 0.05 | default | 0.754530 |  | 0/12 | 0/12 | 0/12 |
| 0.10 | 0.05 | hotter | 0.754530 |  | 0/12 | 0/12 | 0/12 |
| 0.10 | 0.15 | default | 0.754530 |  | 0/12 | 0/12 | 0/12 |
| 0.10 | 0.15 | hotter | 0.754530 |  | 0/12 | 0/12 | 0/12 |

### Three-Seed Confirmation

| Cell | Mean final NDCG | Std | Recovery@12 | Versus C0 | Versus C2 |
|---|---:|---:|---:|---:|---:|
| bound 0.10 / trigger 0.05 / default | 0.760189 | 0.024486 | 1/3 | -0.004476 | -0.069022 |
| bound 0.10 / trigger 0.05 / hotter | 0.760189 | 0.024486 | 1/3 | -0.004476 | -0.069022 |

Calibrated references: C0 rotation final NDCG is 0.764666 +/- 0.023299 with
recovery@12 1/3; C2 oracle re-embed upper bound is 0.829212 +/- 0.010032 with
recovery@12 3/3.

Knob-sweep result: no grid or confirmation cell triggered correction
(`should_correct`, `sedimentation_attempted`, and `correction_applied` are all
0/12). The confirmed top-2 cells lose to frozen C0 and to the C2 oracle. This is
a negative sensitivity result, not evidence that tuning recovered retrieval
quality.

## Honesty Notes

- All table values above are computed from completed JSON artifacts in this branch.
- The initial online model-load sanity check hit the known Hugging Face SSL verification issue on this host. The matrix used `HF_HUB_OFFLINE=1` with the local model and dataset cache; SSL verification was not disabled.
- C3/C4 artifacts distinguish `sedimentation_attempted` from `correction_applied`; runs where the engine found no collapse candidates are not counted as applied corrections merely because a cycle was attempted.
