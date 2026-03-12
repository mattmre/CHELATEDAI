# Research — Session 28 Stale Roadmap Cleanup

## Problem

Three high-visibility legacy documents still implied that ChelatedAI had an unfinished numbered "Phase 4" implementation backlog:

- `REFACTORING_PLAN.md`
- `COMPLETION_SUMMARY.md`
- `PR_DESCRIPTION.md`

That wording conflicted with the current-state audit already captured in `docs/roadmap-audit-and-weight-refinement-plan-2026-03-06.md`.

## What Was Verified

- The old deferred language is historical, not current planning.
- The live repo already contains the major surfaces that those documents would lead a reader to think were still missing, including:
  - adaptive threshold support
  - BEIR benchmarking
  - multitask benchmarking
  - dashboard server APIs
  - broader distillation and teacher-weight infrastructure
- The active non-hardware work is evaluation and research iteration, not a hidden implementation phase.

## Risk

Leaving the old wording untouched causes context rot:

- future sessions can reopen already-completed implementation work
- reviewers can misread the repo as mid-migration when it is actually post-implementation
- session handoffs become inconsistent across canonical and legacy docs

## Decision

Do not rewrite these legacy documents as if they were authored today.

Instead:

1. preserve them as historical records
2. add explicit dated notes that their old "Phase 4" language is superseded
3. rename the misleading sections so readers do not mistake them for the active roadmap
