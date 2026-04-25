# Weight Refinement Campaign Results (Session 32)

**Run:** `experiment_runs/weight-refinement-20260424-210826-session32`  
**Campaign date:** 2026-04-24 to 2026-04-25  
**Documentation date:** 2026-04-25

## Status

This was a partial bounded campaign.

Completed:

- Phase 1 standard sweep
- Phase 2 distillation:
  - `mlp` at teacher weights `0.3`, `0.5`, and `0.7`
  - `procrustes` at teacher weights `0.3` and `0.5`

Not completed:

- `procrustes` at teacher weight `0.7`
- all `low_rank` Phase 2 slices
- Phase 3 multitask suites
- Phase 4 BEIR tiers
- Phase 5 online-loss ablation
- Phase 6 large sweep

The live run was intentionally stopped during `phase2_distillation_procrustes_tw_07` once the documented promotion gates were already failed and continued execution could no longer justify a preset update.

## Setup

- model: `sentence-transformers/all-MiniLM-L6-v2`
- teacher: `sentence-transformers/all-mpnet-base-v2`
- bounded query budget: `100`
- distillation cycles: `5`
- distillation queries per cycle: `50`
- distillation epochs: `5`
- learning rate for the campaign: `0.01`
- run-label hardening and resume-config preservation landed during this session as PRs `#107` and `#110`

## Phase 1: Standard Sweep

Observed signal:

- baseline `NDCG@10`: `0.6101`
- post-learning `NDCG@10`: `0.6119`
- gain: `+0.0017`

Interpretation:

- The current Session 32 operating region still has a slight bounded SciFact gain signal.
- That gain is much smaller than the earlier Session 28 bounded sweep and is not strong enough to justify preset movement on its own.

## Phase 2: Distillation Weight Study

Completed slices:

| adapter | teacher weight | baseline final NDCG@10 | offline final NDCG@10 | hybrid final NDCG@10 | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `mlp` | 0.3 | 0.6012 | 0.0130 | 0.6239 | only local positive SciFact slice |
| `mlp` | 0.5 | 0.6012 | 0.0130 | 0.2961 | catastrophic late-cycle collapse |
| `mlp` | 0.7 | 0.6012 | 0.0130 | 0.2722 | catastrophic late-cycle collapse |
| `procrustes` | 0.3 | 0.1221 | 0.1010 | 0.1313 | nominal edge over a slice-local baseline that itself collapsed from 0.5764 |
| `procrustes` | 0.5 | 0.1221 | 0.1010 | 0.1170 | underperformed the already-collapsed slice-local baseline |

Interpretation:

- `mlp` with `teacher_weight=0.3` was the only completed Phase 2 slice that beat its frozen baseline on SciFact.
- That single positive slice is surrounded by instability: adjacent MLP teacher weights collapsed badly, and the completed `procrustes` slices were not promotion-worthy.
- Offline pretraining remained unattractive in every completed slice.

## Explicit Outcome

**No preset promotion from Session 32.**

Why the promotion gates failed:

1. **SciFact signal was too narrow.** Only one completed Phase 2 slice (`mlp`, `teacher_weight=0.3`, `hybrid`) improved over its baseline.
2. **Repeatability was not satisfied.** The documented promotion rule requires gains to hold across 2 or more independent runs, and this candidate has only one positive run.
3. **Stability was not satisfied.** Adjacent MLP teacher weights collapsed badly, and the completed `procrustes` slices showed severe negative drift across cycles.
4. **Transfer was not satisfied.** The current run never established multi-task or BEIR preservation for the lone positive slice.

Because those gates were already failed, continuing the remaining `procrustes`, `low_rank`, multitask, BEIR, and online-ablation phases could not support a preset refresh in this session.

## Recommended Next Research Step

If additional evaluation work is still desired, do **not** restart the full broad campaign first.

Instead:

1. Run a focused second independent SciFact repeatability check for `mlp` + `teacher_weight=0.3` from a clean run directory.
2. Only if that repeatability check still beats the frozen baseline should the follow-up expand into multi-task and BEIR transfer checks for that single candidate.
3. Keep current presets and defaults unchanged until both repeatability and cross-dataset preservation are demonstrated.
