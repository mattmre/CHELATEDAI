# Architecture — Session 28 Weight-Refinement Campaign Recovery

## Goal

Make bounded weight-refinement evaluation:

1. real
2. isolated
3. resumable
4. reviewable without depending on transient local run directories

## Scope

This architecture covers the evaluation path only. It does not add new retrieval features or change the computational-storage roadmap.

## Design

### 1. Real-engine benchmark entrypoints

- `benchmark_comparative.py`
  - add a reusable `build_real_engine_factory(...)`
  - route CLI runs through real `AntigravityEngine` instances
  - support `--model` and `--max-queries`
- `benchmark_beir.py`
  - accept a real engine factory per dataset
  - expose `--model` and `--max-queries`
  - keep dataset-level evaluation bounded for overnight campaign use

### 2. Centralized adapter isolation

- Add `isolated_adapter_state()` to `benchmark_utils.py`.
- Treat the root adapter checkpoint as shared mutable state that must be hidden during isolated benchmark execution.
- Use the helper in:
  - comparative configurations
  - distillation mode transitions
  - multitask task loops

This keeps benchmark runs from:

- consuming stale adapter state
- overwriting the root checkpoint
- leaking new checkpoint files after temporary evaluations

### 3. Query-bounded sweep execution

- Extend:
  - `run_sweep.py`
  - `run_large_sweep.py`
  - related benchmark flows
- Add bounded query execution and isolated DB-path support.

This allows:

- faster overnight sweeps
- private per-run Qdrant paths
- safer concurrent experimentation without reusing the shared SciFact database

### 4. Resume-aware orchestration

- `run_weight_refinement_campaign.py` becomes the canonical campaign wrapper.
- Resume mode must:
  - load an existing manifest
  - recover completed phases from existing output files
  - execute only missing phases
  - preserve the run directory as the source of truth for campaign state

### 5. Durable analysis outside transient artifacts

- Do not rely on:
  - live background processes
  - ephemeral logs
  - local Qdrant folders
- Preserve the important results in a tracked documentation artifact under `docs/`.

## Branching Strategy

This session intentionally splits work into isolated PRs:

1. benchmark hardening + campaign recovery + results memo
2. stale roadmap documentation cleanup
3. session wrap / tracker refresh

That keeps:

- code-bearing evaluation changes reviewable on their own
- documentation normalization separate from benchmark logic
- session bookkeeping refreshable after the first two PRs exist

## Verification Approach For This Session

The user explicitly asked to skip test execution tonight.

Because of that, this session records:

- prior successful validation already performed during the recovery work
- static source review
- artifact inspection from the completed phases

New test execution is deferred to PR review and the next active validation window.

## Expected Review Questions

1. Should the repo commit raw experiment outputs?
   No. Commit the reusable runner changes and the durable memo, not transient local logs/Qdrant state.
2. Why keep the resume test file if tests are not rerun tonight?
   Because the resume path is substantial enough to require locked-in coverage once validation resumes.
3. Why stop the resumed run?
   Because tonight’s scope shifted from execution to documentation and cleanup, and leaving the run active would consume resources without improving the reviewable PR set.
