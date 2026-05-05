# AttnRes Trained Road-Course Benchmark (2026-05-05)

## Command

```bash
python run_road_course_campaign.py --task SciFact --max-queries 20 --sample-docs 1200 --seed 42 --profile-set attnres_trained_num_blocks --output experiment_runs/roadcourse-small/attnres_trained_num_blocks_scifact_seed42.json
```

## Scope

- Task: `SciFact`
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Slice: 20 queries, 1200 sampled documents
- Profile set: `attnres_trained_num_blocks`
- Training: 2 sedimentation epochs after an opt-in centering warmup
- Output artifact: `experiment_runs/roadcourse-small/attnres_trained_num_blocks_scifact_seed42.json`

## Results

| Profile | Adapter | Blocks | Epochs | Candidates | Events | NDCG@10 | MAP@10 | MRR | Recall@10 | Mean latency ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | `mlp` | 4 | 0 |  |  | 0.816574 | 0.772500 | 0.772500 | 0.950000 | 15.07 |
| `mlp_trained` | `mlp` | 4 | 2 | 625 | 1000 | 0.798669 | 0.748393 | 0.747500 | 0.950000 | 15.34 |
| `attnres_shallow_trained` | `attnres` | 2 | 2 | 625 | 1000 | 0.798669 | 0.748393 | 0.747500 | 0.950000 | 14.51 |
| `attnres_balanced_trained` | `attnres` | 4 | 2 | 625 | 1000 | 0.798669 | 0.748393 | 0.747500 | 0.950000 | 14.53 |
| `attnres_deep_trained` | `attnres` | 8 | 2 | 625 | 1000 | 0.798669 | 0.748393 | 0.747500 | 0.950000 | 15.26 |

## Promotion Decision

No default change is justified.

- Recommended profile: `baseline`
- Baseline NDCG@10: 0.816574
- Best NDCG@10: 0.816574
- Delta vs baseline: 0.000000
- Default change allowed: false
- Reason: `baseline_remains_best`

The quantization survival check also failed closed:

- Quantized NDCG@10: 0.815590
- Retained gain ratio: 0.0
- Gate passed: false
- Reasons: `fp32_gain_below_minimum`, `retained_gain_below_threshold`, `quantized_fitness_below_baseline`

## Interpretation

The trained grid confirms that the sedimentation warmup path executes and that
shallow/balanced/deep AttnRes adapters can be trained through the same benchmark
harness as MLP. On this deterministic SciFact slice, however, the two-epoch
homeostatic sedimentation pass damages NDCG for both MLP and AttnRes by the same
amount. There is no evidence here for promoting AttnRes, and no evidence that
deeper AttnRes blocks help under the current homeostatic target.

Next useful slice: run the same trained grid on NFCorpus for transfer evidence, then
test whether a less aggressive warmup/threshold or contrastive sedimentation loss
avoids the uniform NDCG drop.
