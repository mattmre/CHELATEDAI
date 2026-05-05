# AttnRes Road-Course Benchmark (2026-05-04)

## Command

```bash
python run_road_course_campaign.py --task SciFact --max-queries 20 --sample-docs 1200 --seed 42 --profile-set attnres_comparison --output experiment_runs/roadcourse-small/attnres_comparison_scifact_seed42.json
```

## Scope

- Task: `SciFact`
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Slice: 20 queries, 1200 sampled documents
- Profile set: `attnres_comparison`
- Output artifact: `experiment_runs/roadcourse-small/attnres_comparison_scifact_seed42.json`

## Results

| Profile | Adapter | Quantized | NDCG@10 | MAP@10 | MRR | Recall@10 | Mean latency ms | Action mix |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` | `mlp` | false | 0.816574 | 0.771726 | 0.770833 | 0.950000 | 16.98 | FAST=20 |
| `adaptive_p85_t0.01` | `mlp` | true | 0.816574 | 0.771726 | 0.770833 | 0.950000 | 15.87 | FAST=20 |
| `attnres_baseline` | `attnres` | false | 0.816574 | 0.771726 | 0.770833 | 0.950000 | 16.22 | FAST=20 |
| `attnres_balanced_p85_t0.01` | `attnres` | true | 0.816574 | 0.771726 | 0.770833 | 0.950000 | 16.66 | FAST=20 |

## Promotion Decision

No default change is justified from this run.

- Recommended profile: `baseline`
- Baseline NDCG@10: 0.816574
- Best NDCG@10: 0.816574
- Delta vs baseline: 0.000000
- Default change allowed: false
- Reason: `baseline_remains_best`

The quantization survival check also stayed fail-closed:

- Quantized NDCG@10: 0.815590
- Retained gain ratio: 0.0
- Gate passed: false
- Reasons: `fp32_gain_below_minimum`, `retained_gain_below_threshold`, `quantized_fitness_below_baseline`

## Interpretation

The balanced AttnRes adapter path is live in the road-course harness and does not regress
the retrieval metrics on this deterministic SciFact slice, but it also does not improve
over the near-identity MLP baseline before sedimentation training. Treat this as a harness
and safety validation, not as evidence to promote AttnRes as the default adapter.

Next useful slice: add explicit `num_blocks` profile controls and compare shallow,
balanced, and deep AttnRes variants after a sedimentation training pass rather than only
near-identity adapter inference.
