# AttnRes Repeat-Seed Benchmark (2026-05-05)

## Commands

```bash
python run_road_course_campaign.py --task SciFact --max-queries 20 --sample-docs 1200 --seed 43 --profile-set attnres_balanced_candidate --output experiment_runs/roadcourse-small/attnres_balanced_candidate_scifact_seed43.json
python run_road_course_campaign.py --task SciFact --max-queries 20 --sample-docs 1200 --seed 44 --profile-set attnres_balanced_candidate --output experiment_runs/roadcourse-small/attnres_balanced_candidate_scifact_seed44.json
python run_road_course_campaign.py --task NFCorpus --max-queries 20 --sample-docs 1200 --seed 43 --profile-set attnres_balanced_candidate --output experiment_runs/roadcourse-small/attnres_balanced_candidate_nfcorpus_seed43.json
python run_road_course_campaign.py --task NFCorpus --max-queries 20 --sample-docs 1200 --seed 44 --profile-set attnres_balanced_candidate --output experiment_runs/roadcourse-small/attnres_balanced_candidate_nfcorpus_seed44.json
```

## Scope

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Slice: 20 queries, 1200 sampled documents
- Profile set: `attnres_balanced_candidate`
- Profiles: `baseline`, `mlp_trained`, `attnres_balanced_trained`
- Training: 2 sedimentation epochs after opt-in centering warmup for trained profiles

## Summary

| Task | Seed | Baseline NDCG@10 | MLP trained NDCG@10 | Balanced AttnRes NDCG@10 | Delta vs baseline | Recommended | Quantization gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SciFact | 42 | 0.816574 | 0.798669 | 0.798669 | -0.017905 | baseline | fail |
| SciFact | 43 | 0.842134 | 0.823680 | 0.817134 | -0.025000 | baseline | fail |
| SciFact | 44 | 0.840797 | 0.840797 | 0.840797 | 0.000000 | baseline | fail |
| NFCorpus | 42 | 0.837911 | 0.853883 | 0.857989 | +0.020078 | balanced AttnRes | fail |
| NFCorpus | 43 | 0.863763 | 0.863362 | 0.864882 | +0.001119 | balanced AttnRes | fail |
| NFCorpus | 44 | 0.874058 | 0.873912 | 0.875174 | +0.001117 | balanced AttnRes | pass |

## Details

### SciFact seed 43

| Profile | Adapter | NDCG@10 | MAP@10 | MRR | Recall@10 | Mean latency ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | `mlp` | 0.842134 | 0.797917 | 0.812500 | 0.925000 | 17.33 |
| `mlp_trained` | `mlp` | 0.823680 | 0.772917 | 0.787500 | 0.925000 | 17.63 |
| `attnres_balanced_trained` | `attnres` | 0.817134 | 0.764583 | 0.779167 | 0.925000 | 17.62 |

Quantization gate failed: `fp32_gain_below_minimum`,
`retained_gain_below_threshold`, `quantized_fitness_below_baseline`.

### SciFact seed 44

| Profile | Adapter | NDCG@10 | MAP@10 | MRR | Recall@10 | Mean latency ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | `mlp` | 0.840797 | 0.800794 | 0.812698 | 0.925000 | 15.59 |
| `mlp_trained` | `mlp` | 0.840797 | 0.800794 | 0.812698 | 0.925000 | 16.49 |
| `attnres_balanced_trained` | `attnres` | 0.840797 | 0.800794 | 0.812698 | 0.925000 | 16.17 |

Quantization gate failed: `fp32_gain_below_minimum`.

### NFCorpus seed 43

| Profile | Adapter | NDCG@10 | MAP@10 | MRR | Recall@10 | Mean latency ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | `mlp` | 0.863763 | 0.415540 | 0.892143 | 0.117139 | 15.20 |
| `mlp_trained` | `mlp` | 0.863362 | 0.420714 | 0.893333 | 0.119399 | 14.62 |
| `attnres_balanced_trained` | `attnres` | 0.864882 | 0.421577 | 0.893333 | 0.119399 | 16.01 |

Quantization gate failed: `retained_gain_below_threshold`,
`quantized_fitness_below_baseline`.

### NFCorpus seed 44

| Profile | Adapter | NDCG@10 | MAP@10 | MRR | Recall@10 | Mean latency ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | `mlp` | 0.874058 | 0.423238 | 0.888889 | 0.114748 | 15.09 |
| `mlp_trained` | `mlp` | 0.873912 | 0.425960 | 0.889583 | 0.116381 | 15.19 |
| `attnres_balanced_trained` | `attnres` | 0.875174 | 0.426933 | 0.889583 | 0.116381 | 15.40 |

Quantization gate passed for this seed.

## Interpretation

Balanced AttnRes is consistently not useful on the current SciFact setup:
one repeat seed regressed, one tied, and the earlier trained SciFact run
regressed. NFCorpus is more promising: all three observed seeds recommend
balanced AttnRes, but two of the three quantization survival checks still fail.

This keeps the candidate in research status only. The best next implementation
slice is not a default promotion; it is a quantization-aware or contrastive
sedimentation variant that attempts to preserve the NFCorpus gain after the
quantization survival check.
