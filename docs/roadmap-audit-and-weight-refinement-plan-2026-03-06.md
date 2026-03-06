# Roadmap Audit And Weight Refinement Plan

**Date:** 2026-03-06

## Executive Conclusion

There is no separate active development phase outside the current computational-storage follow-through.

As of 2026-03-06, the active tracker in `docs/ARCH AGENTIC ENGINEERING AND PLANNING/tracker-index.md` only shows:

1. Real RP2040 hardware evidence capture
2. The dated retention review window on or after 2026-04-05

Those are both tied to the computational-storage track. Outside that area, the remaining work is evaluation and research iteration, not missing implementation phases.

## What Was Verified

The old "Phase 4 / future work" language in `REFACTORING_PLAN.md`, `COMPLETION_SUMMARY.md`, and `PR_DESCRIPTION.md` is stale relative to the live code.

The following items are implemented already:

- `antigravity_engine.py`
  - `ingest_streaming(...)`
  - `enable_adaptive_threshold(...)`
- `teacher_distillation.py`
  - configurable `batch_size`
  - chunked encoding
  - ensemble parallel encoding
- `cross_lingual_distillation.py`
  - language-aware teacher routing
- `online_updater.py`
  - `triplet_margin`
  - `infonce`
  - `cosine_similarity`
  - diagnostics and scheduling hooks
- `benchmark_beir.py`
  - multi-dataset BEIR evaluation by tier
- `benchmark_multitask.py`
  - stability + learning-gain benchmarking across task suites
- `dashboard_server.py`
  - dashboard APIs for logs, tests, sweep results, and BEIR results
- `run_sweep.py`
  - standard parameter sweep
- `run_large_sweep.py`
  - large parameter sweep driver

Historical confirmation:

- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/session-log-2026-02-27-session22.md`
  records the Session 21 top-15 implementation items as completed.
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/session-log-2026-02-27-session23.md`
  records the remaining Session 22 PRs as reviewed, fixed, and merged.

## Remaining Non-Hardware Work

### Active engineering phases

None.

### Optional cleanup

- Reconcile stale roadmap language in:
  - `REFACTORING_PLAN.md`
  - `COMPLETION_SUMMARY.md`
  - `PR_DESCRIPTION.md`

This is documentation cleanup, not product implementation.

### Research backlog

- Execute the large sweep defined in `run_large_sweep.py`
- Harden or redesign the dashboard if it needs broader use than local research workflows
- Continue theory validation and weight refinement using the current benchmark stack

## Important Gap

`run_large_sweep.py` exists, but no completed large-sweep artifacts were found:

- `large_sweep_results.json` not present
- `large_sweep_results.csv` not present

That means the missing work is not feature creation. It is experiment execution and analysis.

## Weight Refinement Test Plan

## Goal

Shift the repo from feature implementation to controlled evaluation:

1. Refine sedimentation and teacher-guidance weights
2. Measure whether gains generalize beyond SciFact
3. Test whether the current theory improves retrieval without structural instability

## Research Questions

1. Can sedimentation reliably improve over the frozen baseline instead of merely preserving it?
2. Which adapter family is most stable for post-training correction: `mlp`, `procrustes`, or `low_rank`?
3. Do teacher-guided and hybrid modes outperform baseline under repeated cycles?
4. Do online-update loss variants help or destabilize retrieval?
5. Do improvements transfer across multiple BEIR-style tasks instead of overfitting SciFact?

## Baseline Freeze

Before any new runs:

1. Record the code baseline:
   - commit: `910e6c4`
2. Freeze the current default adapter artifact:
   - `adapter_weights.pt`
3. Record the exact model, task tier, and config preset used for each run.
4. Keep one untouched baseline run for comparison on every dataset family.

## Phase 1: Cheap Validation Sweep

Use the existing standard sweep first to avoid jumping straight into the 7,350-run search space.

Suggested command:

```bash
python run_sweep.py --task SciFact --model sentence-transformers/all-MiniLM-L6-v2 --out sweep_results_stage2.json
```

Questions answered:

- Is the safe operating region from the previous sweep still valid on current `main`?
- Do the tuned presets in `ChelationConfig.SEDIMENTATION_TUNED_PRESETS` still match observed behavior?

Promotion gate:

- Do not launch the large sweep unless the standard sweep reproduces a sane operating region around learning rates `0.001` to `0.01`.

## Phase 2: Distillation Weight And Schedule Study

Compare baseline, offline, and hybrid training with explicit teacher-weight tracking.

Suggested starting commands:

```bash
python benchmark_distillation.py --task SciFact --model sentence-transformers/all-MiniLM-L6-v2 --teacher sentence-transformers/all-MiniLM-L6-v2 --cycles 5 --queries-per-cycle 50 --epochs 5 --lr 0.001 --teacher-weight 0.3 --output benchmark_distillation_tw03.json
python benchmark_distillation.py --task SciFact --model sentence-transformers/all-MiniLM-L6-v2 --teacher sentence-transformers/all-MiniLM-L6-v2 --cycles 5 --queries-per-cycle 50 --epochs 5 --lr 0.001 --teacher-weight 0.5 --output benchmark_distillation_tw05.json
python benchmark_distillation.py --task SciFact --model sentence-transformers/all-MiniLM-L6-v2 --teacher sentence-transformers/all-MiniLM-L6-v2 --cycles 5 --queries-per-cycle 50 --epochs 5 --lr 0.001 --teacher-weight 0.7 --output benchmark_distillation_tw07.json
```

Variables to compare:

- `teacher_weight`: `0.3`, `0.5`, `0.7`
- adapter type: `mlp`, `procrustes`, `low_rank`
- schedule family from `teacher_weight_scheduler.py` and `ChelationConfig.TEACHER_WEIGHT_SCHEDULE_PRESETS`

Primary metric:

- final-cycle `NDCG@10`

Secondary metrics:

- trend over cycles
- degradation versus baseline
- teacher-alignment stability

## Phase 3: Multi-Task Generalization

Run the current engine against multiple tasks to see whether gains are local or transferable.

Suggested commands:

```bash
python benchmark_multitask.py --tasks small --epochs 5 --lr 0.001 --max-queries 100 --num-queries-train 50 --output benchmark_multitask_small.json
python benchmark_multitask.py --tasks medium --epochs 5 --lr 0.001 --max-queries 100 --num-queries-train 50 --output benchmark_multitask_medium.json
```

Track:

- `ndcg_10`
- `avg_jaccard`
- pre/post sedimentation gain
- number of tasks with positive gain

Interpretation rule:

- A configuration that improves SciFact but loses on the medium suite is not a promotion candidate.

## Phase 4: BEIR Cross-Dataset Evaluation

Use the BEIR runner for a cleaner cross-dataset retrieval comparison.

Suggested commands:

```bash
python benchmark_beir.py --tier small --output benchmark_beir_small.json
python benchmark_beir.py --tier medium --output benchmark_beir_medium.json
```

If those are stable, then escalate to:

```bash
python benchmark_beir.py --tier research --output benchmark_beir_research.json
```

Track:

- mean `NDCG@10`
- mean `MAP@10`
- mean `MRR`
- mean `Recall@10`
- latency

Promotion gate:

- No candidate becomes the new default unless it improves or preserves aggregate multi-dataset quality, not just a single benchmark.

## Phase 5: Online-Correction Ablation

The current code supports multiple online loss types. Test whether online correction helps theory or just adds instability.

Study matrix:

- `triplet_margin`
- `infonce`
- `cosine_similarity`

Compare:

- retrieval quality
- stability
- structural health / gradient diagnostics
- sensitivity to update interval and learning rate

Recommended approach:

- keep this phase behind the offline and hybrid sweeps
- do not mix online tuning with large-sweep sedimentation runs until a stable baseline is chosen

## Phase 6: Large Sweep Execution

Only run this after the smaller phases identify a credible search region.

Suggested command:

```bash
python run_large_sweep.py --task SciFact --model sentence-transformers/all-MiniLM-L6-v2 --out large_sweep
```

Notes:

- This script is monolithic and long-running.
- Treat it as a batch research phase, not as the first experiment to run.
- Capture both JSON and CSV outputs for later ranking and preset refresh.

## Metrics To Preserve For Every Run

- code commit
- model name
- dataset or task suite
- adapter type
- teacher weight
- learning rate
- threshold
- noise scale
- epochs
- `NDCG@10`
- `MAP@10`
- `MRR`
- `Recall@10`
- `avg_jaccard`
- pre/post learning gain
- latency
- any collapse or instability signal

## Decision Rules

Promote a candidate only if all of the following hold:

1. It beats or matches the frozen baseline on SciFact.
2. It does not materially regress the `small` or `medium` multi-task suites.
3. It does not show obvious instability through negative gain drift, collapse behavior, or poor stability metrics.
4. Its gains are repeatable across at least two independent runs or equivalent result files.

## Suggested Execution Order

1. Standard sweep refresh on SciFact
2. Distillation teacher-weight study
3. Multi-task benchmark on `small`
4. Multi-task benchmark on `medium`
5. BEIR `small`
6. BEIR `medium`
7. Online-correction ablation
8. Large sweep
9. Preset refresh only after the data above agrees

## Deliverables

Each evaluation pass should produce:

- raw JSON output files from the benchmark scripts
- a summary table ranking candidate configurations
- one recommended preset update or an explicit "no promotion" outcome
- a short analysis note under `docs/` if the run changes the project’s working theory

## Supporting Documents

- `docs/phase4-experiment-protocol.md`
- `docs/distillation-experiment-protocol.md`
- `docs/hybrid-distillation-research.md`
- `docs/research-2026-02-27-sweep-analysis.md`
