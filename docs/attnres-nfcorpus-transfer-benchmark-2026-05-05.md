# AttnRes NFCorpus Transfer Benchmark (2026-05-05)

## Command

```bash
python run_road_course_campaign.py --task NFCorpus --max-queries 20 --sample-docs 1200 --seed 42 --profile-set attnres_trained_num_blocks --output experiment_runs/roadcourse-small/attnres_trained_num_blocks_nfcorpus_seed42.json
```

## Scope

- Task: `NFCorpus`
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Slice: 20 queries, 1200 sampled documents
- Profile set: `attnres_trained_num_blocks`
- Training: 2 sedimentation epochs after an opt-in centering warmup
- Output artifact: `experiment_runs/roadcourse-small/attnres_trained_num_blocks_nfcorpus_seed42.json`

## Results

| Profile | Adapter | Blocks | Epochs | Candidates | Events | NDCG@10 | MAP@10 | MRR | Recall@10 | Mean latency ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | `mlp` | 4 | 0 |  |  | 0.837911 | 0.401512 | 0.858333 | 0.113108 | 13.94 |
| `mlp_trained` | `mlp` | 4 | 2 | 670 | 1000 | 0.853883 | 0.413845 | 0.863333 | 0.116585 | 13.95 |
| `attnres_shallow_trained` | `attnres` | 2 | 2 | 670 | 1000 | 0.853883 | 0.413845 | 0.863333 | 0.116585 | 14.12 |
| `attnres_balanced_trained` | `attnres` | 4 | 2 | 670 | 1000 | 0.857989 | 0.413720 | 0.863333 | 0.117271 | 14.59 |
| `attnres_deep_trained` | `attnres` | 8 | 2 | 670 | 1000 | 0.851135 | 0.412081 | 0.838333 | 0.117271 | 14.58 |

## Promotion Decision

Do not promote defaults from this run alone.

The profile-level recommendation is positive on this NFCorpus slice:

- Recommended profile: `attnres_balanced_trained`
- Baseline NDCG@10: 0.837911
- Best NDCG@10: 0.857989
- Delta vs baseline: +0.020078
- Default change allowed by the local road-course rule: true

The quantization survival gate still fails closed:

- Quantized NDCG@10: 0.837457
- Quantized gain vs baseline: -0.000453
- Retained gain ratio: -0.022576
- Gate passed: false
- Reasons: `retained_gain_below_threshold`, `quantized_fitness_below_baseline`

## Interpretation

This is the first positive trained AttnRes signal in the current track. Balanced
AttnRes outperformed both the untrained baseline and the trained MLP profile on
NFCorpus, while deep AttnRes underperformed balanced. That suggests the 4-block
configuration is the right candidate for additional evidence, not the 8-block
paper-scale variant.

The result is not promotable yet because the previous SciFact trained run failed
closed and quantized-survival erased the NFCorpus gain. Treat balanced AttnRes as
a research candidate that needs repeat seeds, transfer confirmation, and either a
quantization-aware training path or a stricter promotion gate before any default
change.

Repeat-seed evidence is now documented in
`docs/attnres-repeat-seed-benchmark-2026-05-05.md`. NFCorpus retained a positive
balanced-AttnRes signal across seeds 42/43/44, but quantization survival only
passed on one of the three seeds. Next useful slice: test whether contrastive
sedimentation preserves the NFCorpus gain under quantization.
