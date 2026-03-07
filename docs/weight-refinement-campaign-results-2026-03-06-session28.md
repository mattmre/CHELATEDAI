# Weight Refinement Campaign Results

**Run:** `experiment_runs/weight-refinement-20260306-session28-clean`  
**Campaign date:** 2026-03-06  
**Documentation date:** 2026-03-07

## Status

This was a partial bounded campaign.

Completed:

- Phase 1 standard sweep
- Phase 2 distillation weight study
- Phase 3 multitask suites
- Phase 4 BEIR small

Not completed:

- Phase 4 BEIR medium
- Phase 5 online-loss ablation
- Phase 6 large sweep

The run was interrupted during `phase4_beir_medium`. A resume-capable runner now exists, but the live resumed process was stopped once the overnight priority shifted to documentation and cleanup rather than continued benchmark execution.

## Setup

- model: `sentence-transformers/all-MiniLM-L6-v2`
- bounded query budget: `50`
- distillation cycles: `3`
- distillation queries per cycle: `30`
- distillation epochs: `3`
- learning rate for distillation and multitask phases: `0.001`

## Phase 1: Standard Sweep

Best observed configuration:

- learning rate: `0.01`
- threshold: `1`
- noise scale: `0.2`
- epochs: `5`

Result:

- baseline `NDCG@10`: `0.6289`
- post-learning `NDCG@10`: `0.6766`
- gain: `+0.0477`

Interpretation:

- There is still a useful SciFact operating region for bounded sedimentation-style tuning.
- That improvement signal is local to this phase and should not be promoted on its own.

## Phase 2: Distillation Weight Study

Teacher weights evaluated:

- `0.3`
- `0.5`
- `0.7`

Observed outcome:

| teacher weight | baseline mean NDCG@10 | offline mean NDCG@10 | hybrid mean NDCG@10 | offline pretraining time |
| --- | ---: | ---: | ---: | ---: |
| 0.3 | 0.7553 | 0.7553 | 0.7553 | 171.7s |
| 0.5 | 0.7553 | 0.7553 | 0.7553 | 216.3s |
| 0.7 | 0.7553 | 0.7553 | 0.7553 | 194.2s |

Interpretation:

- In this bounded run, distillation weight changes did not move retrieval quality.
- Offline pretraining added time cost without observed retrieval benefit.
- The next iteration should not prioritize teacher-weight tuning until a stronger learning signal appears.

## Phase 3: Multitask Generalization

| suite | successful tasks | mean NDCG@10 | mean stability | mean learning gain |
| --- | ---: | ---: | ---: | ---: |
| small | 2 | 0.6782 | 1.0000 | 0.0000 |
| medium | 3 | 0.6265 | 1.0000 | 0.0000 |

Interpretation:

- The engine remained structurally stable across the bounded suites.
- Stability did not translate into measurable learning gain.
- Current training settings preserve behavior more than they improve it.

## Phase 4: BEIR Small

Aggregate cross-dataset summary:

| configuration | mean NDCG@10 | mean MAP@10 | mean MRR | mean Recall@10 | mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.6839 | 0.4765 | 0.6648 | 0.5079 | 23.67 ms |
| random_mask_50pct | 0.6839 | 0.4765 | 0.6648 | 0.5079 | 26.10 ms |
| chelation+tempscale_0.5 | 0.5748 | 0.3835 | 0.5527 | 0.4563 | 59.92 ms |
| procrustes | 0.5748 | 0.3835 | 0.5527 | 0.4563 | 34.93 ms |
| chelation+tempscale_2.0 | 0.5746 | 0.3832 | 0.5524 | 0.4563 | 28.95 ms |
| online_updates | 0.5746 | 0.3832 | 0.5524 | 0.4563 | 87.75 ms |
| chelation | 0.5745 | 0.3831 | 0.5524 | 0.4563 | 26.59 ms |
| low_rank_16 | 0.5745 | 0.3831 | 0.5524 | 0.4563 | 34.96 ms |

Interpretation:

- The current chelation-heavy configurations underperformed the baseline on the bounded BEIR-small comparison.
- `online_updates` was especially unattractive in this slice because it added latency without offsetting quality gains.
- No chelation-family configuration from this run is a promotion candidate.

## Working Conclusions

1. The current benchmark stack needed reproducibility hardening more than new retrieval features.
2. The strongest positive signal remains the bounded SciFact sweep region from Phase 1.
3. Distillation weight changes and online updates did not justify promotion in this run.
4. Stable behavior alone is not enough; the missing signal is transferable quality improvement.

## Recommended Next Research Step

Resume from the missing `phase4_beir_medium` stage or rerun the bounded campaign from a clean run directory once review feedback lands on the benchmark hardening PR.

Do not:

- promote any chelation-family default from this run
- spend more time on teacher-weight sweeps before recovering a measurable gain signal
- treat the interrupted campaign as a complete validation pass
