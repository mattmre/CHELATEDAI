# Drift Recovery C3 Knob Sweep - June 2026

> [!WARNING]
> **LEGACY_METRIC_LINEAGE_BLOCKED (2026-07-27; v2 blast-radius
> reconditioning).** Every exact nDCG value, ranking, top-cell selection,
> recovery verdict, and negative-result conclusion below is retained historical
> diagnostic text, not accepted evidence. This surface was produced directly
> or transitively from `drift_recovery_metrics.ndcg_at_k`, whose IDCG omitted
> positive qrels outside the retrieved top-k. Do not use it for scientific,
> promotion/rejection, paper, release, or roadmap claims until qrels-complete
> regeneration and hash-linked supersession close `CD-MLR-01`.

Manifest: `experiment_runs/drift-recovery/knob-sweep/knob-sweep-manifest-2026-06.json`.
Scope: calibrated rotation setting only; all 12 seed-42 grid cells are reported.
Selection policy: Top two cells by seed-42 final NDCG; ties are broken by lower trigger threshold, then profile name, then lower bound epsilon.
Profile note: `hotter` means the pre-registered aggressive schedule profile (lower max-temperature cap plus doubled epoch scale), not a higher temperature cap.

## Grid Cells

| Bound epsilon | Trigger threshold | Profile | Max temperature | Epochs scale | Final NDCG | Recovery cycle | should_correct | attempted | applied |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.05 | default | 1.00 | 1.0 | 0.754416 |  | 0/12 | 0/12 | 0/12 |
| 0.01 | 0.05 | hotter | 0.01 | 2.0 | 0.754416 |  | 0/12 | 0/12 | 0/12 |
| 0.01 | 0.15 | default | 1.00 | 1.0 | 0.754416 |  | 0/12 | 0/12 | 0/12 |
| 0.01 | 0.15 | hotter | 0.01 | 2.0 | 0.754416 |  | 0/12 | 0/12 | 0/12 |
| 0.05 | 0.05 | default | 1.00 | 1.0 | 0.753156 |  | 0/12 | 0/12 | 0/12 |
| 0.05 | 0.05 | hotter | 0.01 | 2.0 | 0.753156 |  | 0/12 | 0/12 | 0/12 |
| 0.05 | 0.15 | default | 1.00 | 1.0 | 0.753156 |  | 0/12 | 0/12 | 0/12 |
| 0.05 | 0.15 | hotter | 0.01 | 2.0 | 0.753156 |  | 0/12 | 0/12 | 0/12 |
| 0.10 | 0.05 | default | 1.00 | 1.0 | 0.754530 |  | 0/12 | 0/12 | 0/12 |
| 0.10 | 0.05 | hotter | 0.01 | 2.0 | 0.754530 |  | 0/12 | 0/12 | 0/12 |
| 0.10 | 0.15 | default | 1.00 | 1.0 | 0.754530 |  | 0/12 | 0/12 | 0/12 |
| 0.10 | 0.15 | hotter | 0.01 | 2.0 | 0.754530 |  | 0/12 | 0/12 | 0/12 |

## Three-Seed Confirmation

| Bound epsilon | Trigger threshold | Profile | Mean final NDCG | Std | Recovery@12 |
|---:|---:|---|---:|---:|---:|
| 0.10 | 0.05 | default | 0.760189 | 0.024486 | 1/3 |
| 0.10 | 0.05 | hotter | 0.760189 | 0.024486 | 1/3 |

## Calibrated References

| Condition | Final NDCG mean | Std | Recovery@12 |
|---|---:|---:|---:|
| C0 | 0.764666 | 0.023299 | 1/3 |
| C2 | 0.829212 | 0.010032 | 3/3 |
