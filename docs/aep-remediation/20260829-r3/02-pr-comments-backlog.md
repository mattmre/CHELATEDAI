# Round 3 Tier 2 PR comments and checks backlog

All review-thread pagination ended with `hasNextPage=false`. UI-unresolved is recorded independently from outdated or substantively fixed; no unresolved flag is silently treated as a current defect.

| PR | Issue / review / UI-unresolved threads | Current disposition |
|---:|---:|---|
| 308 | 0 / 0 / 0 | newest body validator passes; block flag correctly remains `BLOCKED` |
| 297 | 0 / 0 / 0 | body validator misses 16 fields; body already says not ready; net-new code packet `PR297-001` |
| 296 | 0 / 0 / 0 | conflicting and block-flagged; prior aggregate controls |
| 295 | 0 / 1 / 2 | router race substantively fixed; remaining cache note is performance-only; Git fallback suggestion conflicts with intentional fail-closed provenance |
| 294 | 0 / 1 / 4 | three outdated and one current UI thread, all substantively fixed; separate net-new `PR294-001` |
| 293 | 0 / 1 / 2 | both outdated/unresolved and fixed at head |
| 292 | 0 / 1 / 0 | public head conflicting/stale; no new thread cause |
| 278 | 0 / 1 / 7 | all outdated newline-only comments; §4 evidence body failure remains current |
| 257 | 1 / 1 / 4 | all current/actionable but already packetized; parked by owner |
| 256 | 1 / 1 / 8 | all current; two map to existing mutable-state packet, six belong to an explicitly parked research stub |

## Current PR #257 thread mapping

- `discussion_r3367630939`, `antigravity_engine.py:1243`: Round 1 `PR257-001` Critical fallback configuration failure.
- `discussion_r3367630942`, `benchmark_utils.py:165`: Round 1 `PR257-002` nested isolation leak.
- `discussion_r3367630944`, `block_graph.py:136`: Round 2 `AEP-20260829-r2-PR257-001` primary exception replaced by secondary hook failure; A2's provisional “new” label was rejected during normalization.
- `discussion_r3367630945`, `aep_orchestrator.py:201`: Round 1 `PR257-003` serialization envelope escape.

## Current PR #256 thread mapping

- `discussion_r3314775287` and `discussion_r3314775289`, `shim_node.py:467,512`: Round 1 `PR256-002` mutable owned-state leak.
- `discussion_r3314775255`, `3314775263`, `3314775266`, `3314775274`, `3314775279`, and `3314775283`: interpreter, locale encoding, and current-directory persistence notes in `docs/.../long_running_orchestrator_stub.py`. They remain owner-parked research-stub work, not current production FE/BE claims, so no active product packet is created.
- Hosted F821 `corrective` failure remains Round 1 `PR256-001` and is inherited by #257.

## Check boundary

- #257 Python 3.9 stopped after approximately six hours; that runtime gate is `SKIPPED_GATE`, not a pass.
- #257/#256 lint failures were opened; existing packet mechanisms remain controlling.
- #278/#297 body-validator failures and #308/#296 block failures were opened and retained.
- No required-check configuration exists on any open PR head/base reviewed. Passing matrices and mergeability cannot supply branch-policy enforcement.
