# Session 31 Task Plan: PR Fixes, Merges, and Feature Implementation

## Goal
Fix open PR review comments, merge PRs #96/#97/#98 smartly, then implement priority architectural improvements from Session 30 research findings.

## Current Phase
ALL PHASES COMPLETE — Session 31 finished

## Merge Order (minimizes drift)
1. PR #97 (docs only, zero conflict risk) → merge first
2. PR #96 (code + Session 30 bug fixes) → rebase on main, merge second
3. PR #98 (session wrap / handoff) → rebase on main, merge last

## Task Dependency Graph
```
T1 (Fix PR #96 comments) ──┐
                            ├── T3 (Commit S30 fixes) ── T4 (Test+Lint) ──┐
T2 (Fix PR #97 comments) ── T5 (Merge PR #97) ──────────────────────────┤
                                                                          ├── T6 (Merge PR #96)
                                                                          │        │
                                                                          ├────────┤
                                                                          │        ├── T7 (Merge PR #98)
                                                                          │
                                                          T8 (Research contrastive loss) ──┐
                                                                                            ├── T9 (Implement contrastive loss) ──┐
                                                          T10 (Research scaler-constrainer)─┤                                     │
                                                                                            ├── T11 (Implement scaler) ───────────┤
                                                          T12 (INT8 quant fix) ──────────────────────────────────────────────────┤
                                                          T13 (DimensionProjection fix) ─────────────────────────────────────────┤
                                                          T14 (Kalman gain) ─────────────────────────────────────────────────────┤
                                                                                                                                  │
                                                                                                              T15 (Session wrap) ─┘
```

## Phases

### Phase 1: PR Review Fixes (T1, T2)
- Fix 5 Copilot comments on PR #96
- Fix 2 Copilot comments on PR #97
- Status: COMPLETE

### Phase 2: Commit, Test, Lint (T3, T4)
- Commit Session 30 bug fixes to PR #96 branch
- Run full test suite + ruff check — 986 tests passing, lint clean
- Status: COMPLETE

### Phase 3: Smart Merge Sequence (T5, T6, T7)
- Merged #97 → main (docs, cleanest)
- Rebased #96 on main, merged (code + Session 30 fixes)
- Rebased #98 on main, merged (wrap)
- Post-merge: 986 tests on main, lint clean
- Status: COMPLETE

### Phase 4: Research & Architecture (T8, T10)
- Research contrastive loss (InfoNCE) architecture
- Research scaler-constrainer / BoundedAdapter design
- Research DimensionProjection + Kalman gain
- 3 research docs written to docs/
- Status: COMPLETE

### Phase 5: Feature Implementation (T9, T11, T12, T13, T14)
- T13: DimensionProjection fix → PR #99 MERGED (991 tests)
- T11+T12: BoundedAdapter + INT8 fix → PR #100 MERGED (1013 tests)
- T9: Contrastive loss → PR #101 MERGED (1052 tests)
- T14: Kalman gain → PR #102 MERGED (1082 tests)
- Status: COMPLETE

### Phase 5b: Smart Merge Sequence for Feature PRs
- PR #99 → MERGED ✓
- PR #100 → Rebased, conflict resolved, MERGED ✓
- PR #101 → Rebased after #100, conflict resolved, MERGED ✓
- PR #102 → Rebased after #101, conflict resolved, MERGED ✓
- Final: 1082 tests passing on main
- Status: COMPLETE

### Phase 6: Session Wrap (T15)
- Updated session log, task plan, memory
- Status: COMPLETE

## PR #96 Review Comments (5 items)
1. benchmark_beir.py:820 - Cache per-corpus engine factory
2. docs/weight-refinement-campaign-results.md:70 - Fix `||` table syntax
3. run_weight_refinement_campaign.py:89 - staged_not_launched shouldn't count as complete
4. run_weight_refinement_campaign.py:383 - Preserve manifest config on resume
5. benchmark_utils.py:97 - Preserve pred_ids order in map_predicted_ids()

## PR #97 Review Comments (2 items)
1. REFACTORING_PLAN.md:42 - Move historical items out of Phase 2 task list
2. REFACTORING_PLAN.md:135 - Fix unchecked checkbox in completion criteria

## Session 30 Bug Fixes (uncommitted, on PR #96 branch)
- antigravity_engine.py: hybrid dimension mismatch fix
- chelation_adapter.py: Procrustes regularization+DSM, low-rank asymmetric init
- benchmark_distillation.py: chelation path fix
- test_integration_rlm.py: new Procrustes tests
- test_unit_core.py: new adapter tests

## Error Log
| Timestamp | Error | Resolution |
|-----------|-------|------------|
