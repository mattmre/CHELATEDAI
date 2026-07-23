# Drift Recovery Diagnostics - June 2026

> [!WARNING]
> **LEGACY_METRIC_LINEAGE_BLOCKED (2026-07-23).** Exact nDCG values and
> derived comparisons below are historical diagnostics, not accepted evidence,
> until corrected regeneration. See
> `docs/research/metric-lineage-repair-protocol-2026-07.md`.

Diagnostics JSON: `experiment_runs/drift-recovery/diagnostics-2026-06.json`.
Source artifact count: 30.

## Hypothesis Verdicts

- H1_rotation_too_weak: **confirmed** - Rotation C0 mean drop is 1.739% and all C0 rotation runs recover at cycle 1; noise C0 mean drop is 17.471%.
- H2_correction_barely_moves: **confirmed** - C3 mean correction norms are [0.009999637, 0.009999638]; C4 mean correction norms are [0.000189831, 0.00018883]; detection fired every C3/C4 cycle=True; C3 applied vector updates=False.
- H3_c2_oracle_reembed: **confirmed** - C2 baseline-final absolute diffs across 6 runs: max=0, tolerance=1e-12.

## Baseline vs Final

| Drift | Condition | Baseline NDCG | Final NDCG | Drop % | Recovery@12 |
|---|---|---:|---:|---:|---:|
| rotation | C0 | 0.829212 +/- 0.010032 | 0.814730 +/- 0.006692 | 1.739% | 3/3 |
| noise | C0 | 0.829212 +/- 0.010032 | 0.684292 +/- 0.021042 | 17.471% | 0/3 |
| rotation | C1 | 0.829212 +/- 0.010032 | 0.814730 +/- 0.006692 | 1.739% | 3/3 |
| noise | C1 | 0.829212 +/- 0.010032 | 0.684292 +/- 0.021042 | 17.471% | 0/3 |
| rotation | C2 | 0.829212 +/- 0.010032 | 0.829212 +/- 0.010032 | 0.000% | 3/3 |
| noise | C2 | 0.829212 +/- 0.010032 | 0.829212 +/- 0.010032 | 0.000% | 3/3 |
| rotation | C3 | 0.829212 +/- 0.010032 | 0.814820 +/- 0.006771 | 1.728% | 3/3 |
| noise | C3 | 0.829212 +/- 0.010032 | 0.685332 +/- 0.020038 | 17.344% | 0/3 |
| rotation | C4 | 0.829212 +/- 0.010032 | 0.814730 +/- 0.006692 | 1.739% | 3/3 |
| noise | C4 | 0.829212 +/- 0.010032 | 0.684332 +/- 0.021029 | 17.466% | 0/3 |

## C3/C4 Correction Trace

| Drift | Condition | should_correct | attempted | applied | Mean drift signal | Mean correction norm | Max correction norm |
|---|---|---:|---:|---:|---:|---:|---:|
| rotation | C3 | 36/36 | 36/36 | 0/36 | 0.002010709 | 0.009999637 | 0.009999711 |
| noise | C3 | 36/36 | 36/36 | 0/36 | 0.002084925 | 0.009999638 | 0.009999711 |
| rotation | C4 | 36/36 | 36/36 | 0/36 | 0.002009232 | 0.000189831 | 0.000233389 |
| noise | C4 | 36/36 | 36/36 | 0/36 | 0.002085138 | 0.000188830 | 0.000242040 |

## Severity

| Drift | C0 mean baseline | C0 mean final | Mean drop % | All C0 runs above 95% threshold |
|---|---:|---:|---:|---|
| rotation | 0.829212 | 0.814730 | 1.739% | True |
| noise | 0.829212 | 0.684292 | 17.471% | False |

## Plots

- `experiment_runs/drift-recovery/scifact_rotation_ndcg_diagnostics_mean.png`
- `experiment_runs/drift-recovery/scifact_noise_ndcg_diagnostics_mean.png`
