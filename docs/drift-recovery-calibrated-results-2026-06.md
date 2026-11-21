# Drift Recovery Calibrated Severity Results - June 2026

> [!WARNING]
> **LEGACY_METRIC_LINEAGE_BLOCKED (2026-07-27; v2 blast-radius
> reconditioning).** Every exact nDCG value, comparator ordering, severity
> selection, recovery verdict, and H1/H2/H3 conclusion below is retained
> historical diagnostic text, not accepted evidence. This surface was produced
> directly or transitively from `drift_recovery_metrics.ndcg_at_k`, whose IDCG
> omitted positive qrels outside the retrieved top-k. Do not use it for
> scientific, promotion/rejection, paper, release, or roadmap claims until
> qrels-complete regeneration and hash-linked supersession close `CD-MLR-01`.

Manifest: `experiment_runs/drift-recovery/calibrated/calibration-manifest-2026-06.json`.
Choice rule: Among scout cells, choose a setting inside the 8-20% baseline-drop zone closest to 12%; if none land in-zone, choose the closest overall and disclose that.

## Scout Cells

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

## Chosen Settings

- rotation: fraction 0.50, angle 35.0, sigma 0.050; scout drop 7.732% (closest outside zone).
- noise: fraction 0.50, angle 25.0, sigma 0.035; scout drop 9.317% (in zone).

## Full Matrix

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

## Plots

- `experiment_runs/drift-recovery/calibrated/scifact_rotation_ndcg_diagnostics_mean.png`
- `experiment_runs/drift-recovery/calibrated/scifact_noise_ndcg_diagnostics_mean.png`
