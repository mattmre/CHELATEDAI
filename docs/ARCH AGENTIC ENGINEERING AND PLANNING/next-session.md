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

## Carried Debt (BHS v3.3 §6.3 — TTL = 1 cycle, BLOCKED if unresolved next cycle)

These are honesty gaps disclosed at merge time of PR #244 (2026-05-16, BHS_OFFICIAL = 55, OPERATOR_OVERRIDE). The cycle after this one **forbids new feature work until these are cleared** per §6.3 quantitative cycle definition.

- **CD-244-01 (P0, L1 + L4)** — Replace the `scripts/bhs_validator.py` stub with a real implementation. Current `validate_pr_brutal_honesty()` returns hardcoded `score=0.0` and `run_smoke_pipeline()` always returns `True`. AEP orchestrator hooks call these functions, so the headline "BHS scoring in synthesis/tiered_remediation/closure" claim is currently load-bearing on a placeholder. Either wire `scripts/validate_pr_brutal_honesty.py` logic into the importable module, or strip the hooks. Violates Session Rule #1.
- **CD-244-02 (P0, L4)** — `aep_orchestrator.py` `synthesis()`/`tiered_remediation()`/`closure()` hooks consume `bhs_metadata` but the resulting `summary["avg_bhs_score"]` is currently always 0.0. Once CD-244-01 lands, verify the score actually varies with finding content and is surfaced in the closure summary that operators read.
- **CD-244-03 (P1, L4 + L8)** — New computational-storage modules (`computational_storage_poc/moe_reap.py`, `sparse_cpu_inference.py`, `packed_graph.py`, `packed_cpu_inference.py`, `repo_graph_memory.py`, `integrated_repo_runtime.py`, `phase7_system_evaluation.py`, `disk_llm_estimator.py`, `cpu_backends.py`, and their benchmarks) are unit-tested but not consumed by any production path. They must either be wired into an actual code path (engine, AEP, dashboard) with runtime evidence, or be moved behind an explicitly-experimental flag/README label and removed from any "ships" claims.
- **CD-244-04 (P2, L11 risk)** — `aep_orchestrator.py` line 33 catches `Exception` (not `ImportError`) around the BHS import, silently absorbing any failure mode. After CD-244-01 fixes the underlying stub, narrow the except clause so a real failure is not hidden.
- **CD-244-05 (P2, hygiene)** — `model.cspg` (binary artifact, tracked via PR #244) should be either ignored, LFS-tracked, or removed depending on its role. Decide and document.

## Cycle ID
- Autonomous continuation after AEP-2026-04-30 and AEP-2026-05-01; PR #244 admin-merged 2026-05-16 with OPERATOR_OVERRIDE (BHS_OFFICIAL=55).
