# Session 31 Task Plan: PR Fixes, Merges, and Feature Implementation

## Session 32 Addendum: Disk-Resident LLM Feasibility

### Goal
- Quantify whether ChelatedAI can evolve from the current computational-storage proof into a disk-resident CPU inference path for larger language models.

### Phase Status
- Repo architecture review: COMPLETE
- External method alignment (`LLM in a Flash`, CPU low-bit lookup): COMPLETE
- In-repo estimator implementation: COMPLETE
- Feasibility memo: COMPLETE
- Revised roadmap and ARCH-AEP loop updates: COMPLETE
- Phase 1 storage substrate initial slice: COMPLETE
- Phase 2 CPU inference substrate initial slice: COMPLETE
- Phase 2 ARCH-AEP hardening and validation: COMPLETE
- Phase 2b prequantized packed artifacts and lower-overhead CPU path: COMPLETE
- Phase 3A dense/sparse FFN selective-loading path: COMPLETE
- Phase 4 retrieval and graph memory substrate: COMPLETE
- Phase 5 runtime integration prototype: COMPLETE
- Phase 6 memory compression on the integrated baseline: COMPLETE
- Phase 7 end-to-end evaluation and promotion review: COMPLETE
- Phase 3B parallel MoE / REAP branch: COMPLETE
- Final synthesis for the initial storage+CPU baseline: COMPLETE

### Current Addendum State
- Phase 1, Phase 2, Phase 2b, Phase 3A, Phase 4, Phase 5, Phase 6, and Phase 7 are complete and promotable as the baseline substrate for follow-on work.
- The repo now has an initial promoted low-bit path:
  - INT8 packed artifacts
  - prequantized weight reuse in the CPU backend
  - a measured microbenchmark win over the float32 baseline
- The repo now also has an initial selective-loading runtime:
  - streamed-block row-chunk loading
  - activation-driven routing
  - chunk reuse cache
  - measured streamed-byte reduction against the dense packed baseline
- The repo now has an initial disk-backed code-memory layer:
  - on-disk node / edge / embedding bundle
  - memory-mapped embedding index
  - hybrid retrieval with graph-aware reranking
  - independent ingest and query benchmark
- The repo now has an initial end-to-end CPU-only prototype path:
  - repository query -> disk-backed memory retrieval
  - retrieval-result featureization
  - packed INT8 reranking through the sparse CPU path
  - end-to-end latency and byte metrics
- The repo now has an initial compression path on top of that baseline:
  - int8-compressed repo-memory embeddings
  - mapped-byte reduction benchmarks
  - integrated-runtime compression comparison
- The repo now has a closed promotion review:
  - research-baseline promotion approved
  - production-ready promotion deferred
  - next-branch recommendation recorded
- The repo now has an initial parallel MoE / REAP branch:
  - disk-backed MoE artifact
  - routed-expert CPU execution
  - REAP-like expert-bank pruning compatibility
- The repo now also has a bounded post-Phase-7 optimization pass:
  - chunked int8 repo-memory query scoring
  - a shared retrieval evaluation suite across standalone and integrated benchmarks
  - broader repo-local evaluation coverage without changing architecture scope
- Next bounded decision is either:
  - further optimization under the current research baseline
  - end-to-end integration of the MoE branch, if a concrete MoE target is selected

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
