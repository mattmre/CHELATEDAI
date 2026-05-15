# Next Session Checklist

Purpose: Continue autonomous execution after the Engine-Scope and Model-Scope implementation cycles closed on `main`.

## Non-Negotiable Session Rules

These rules apply to every session, every task, every PR — no exceptions:

1. **Full Implementation Only.** No scaffolding, stubs, `pass`-body placeholders, or `TODO: implement` in committed code. Each sub-slice must be independently complete and working.
2. **PR on Completion.** Every finished slice must be in an open PR before the next slice starts. No accumulating dirty branches or uncommitted work across sessions.
3. **No Placeholder Data or Fake Metrics.** Every field shown in a dashboard or frontend must trace to a real source artifact or live computation. Hard-coded demo values and mock responses are forbidden in committed code.

## Session Start
- Sync local `main` to `origin/main`.
- Review `docs/ARCH AGENTIC ENGINEERING AND PLANNING/roadmap-execution-queue-2026-05-05.md`.
- Review `docs/heavyskill-engine-adaptation-2026-05-05.md`.
- Review the latest phase summary under `docs/ARCH AGENTIC ENGINEERING AND PLANNING/phase-summaries/`.
- Check open PRs for comments and failing checks before starting new work.
- Keep each implementation slice PR-sized, with focused tests and a phase-summary update.

## Current State (Post 2026-05-15 Reconciliation)
- Desktop machine (23 commits behind) has been reconciled with laptop work on branch `reconciliation/2026-05-desktop-sync`.
- BHS v3.3 tooling is now present in `scripts/` (validate_pr_brutal_honesty.py, schema drift validator, smoke_pipeline.py, etc.), but **not yet wired** into the AEP orchestrator.
- All major laptop 10-phase planning artifacts (`FINAL_PLAN.md` family + panel-analysis) have been landed into `docs/ARCH AGENTIC ENGINEERING AND PLANNING/planning/2026-05-reconciliation/`.
- New computational storage POC code (packed/CPU/sparse/repo-graph/MoE/REAP) moved to feature branch `feat/post-merge-comp-storage-substrate`.
- Engine-Scope and Model-Scope cycles remain complete as scaffolds. Promotion is still evidence-gated and fail-closed.
- A dedicated reconciliation reimplementation backlog now exists (see planning/2026-05-reconciliation/reconciliation-2026-05-15-reimplementation-backlog.md).

## Priority Order (Post-Reconciliation 2026-05-15)
1. **Wire BHS v3.3 honesty gates into the AEP orchestrator** (highest integrity gap — BHS tooling exists in scripts/ but is not enforced).
2. **Port and integrate the new computational storage substrate** (packed/CPU/sparse/repo-graph/MoE) from `feat/post-merge-comp-storage-substrate` with honesty + Model-Scope scoping.
3. **Re-scope and integrate the 10-phase laptop planning docs** (now in planning/2026-05-reconciliation/) against post-merge reality.
4. **Create scripts/ smoke + BHS validator integration** so honesty claims are actually testable.
5. **Update golden suite + research validity tests** for BHS v3.3 + Model-Scope + new storage code.
6. **Real computational-storage hardware evidence** (still externally blocked).
7. **Default promotion governance review** (still gated until strong evidence).
8. **Model-Scope runtime + hook bus** (was previous next slice).

## Resume Pointer
- Active architecture doc: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/architecture-2026-05-01-model-scope-roadmap.md`
- Golden-default roadmap slices 1–10 merged on `main` as of 2026-05-08 (PRs #219–#222; 1,643 tests passing)
- Execution queue (roadmap-execution-queue-2026-05-05.md): all 80 items complete
- **Next slice: Model-Scope Phase 1** — `feat/slice13-model-scope-runtime` (`model_scope_runtime.py` + `model_hook_bus.py`)
- Sequence: Phase 1 → Phase 2 (features) → Phase 3 (steering) → Phase 4 (memory) → Phase 5 (training) → Phase 6 (engine integration)

## Handoff Notes
- Do not add `pytest` imports to `test_*.py`; CI does not install `pytest`.
- Python 3.9 CI: avoid runtime `X | None` annotations unless the module uses deferred annotations.
- `ruff check` does not validate GitHub Actions YAML.
- Prefer compact, versioned artifact schemas over large raw dumps.
- Persistent promotion should target overlays and memory artifacts before any discussion of base-weight mutation.
- Default-promotion preflight is fail-closed; a nonzero exit can be the expected result when evidence says `no_default_change`.

## Cycle ID
- Autonomous continuation after AEP-2026-04-30 and AEP-2026-05-01
