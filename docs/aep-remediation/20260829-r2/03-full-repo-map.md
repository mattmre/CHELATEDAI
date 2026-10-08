# Round 2 Tier 3 — full-repository FE/backend map

Tier 3 began only after every dirty worktree was opened in Tier 1 and all ten open PRs plus their comments, review threads, checks, and failed/cancelled logs were opened in Tier 2.

## Critical paths opened

| Path / boundary | Runtime contract | Round 2 result |
|---|---|---|
| `run_qscci.py` ↔ owned child ↔ GPU service | restore the displaced service only after the experiment child/group is terminal and reaped | Critical WT-001: final wait can fail with a live child and still enter service restoration |
| `aep_orchestrator.py` lifecycle | repair and independent evidence must precede `VERIFIED` | Critical REPO-003: absent callbacks close untouched work; High REPO-004: timeout does not revoke a worker |
| `checkpoint_manager.py` catalog ↔ bytes | unique immutable identity plus durable metadata commit | High REPO-001/002: same-second overwrite and acknowledged-but-undiscoverable checkpoints |
| `dashboard_server.py` history loaders ↔ operator status | newest selected evidence must remain newest even when unreadable | High REPO-005: malformed newest validation is silently replaced by older green state |
| `antigravity_engine.py` ↔ `adapter_router.py` | accepted centroids share the engine's vector space | High REPO-006: incompatible geometry is published as enabled and fails on selection |
| adaptive-threshold configuration ↔ inference | validated ordered bounds commit atomically | High REPO-007: failed validation leaves mode enabled; reversed bounds are accepted |
| `dashboard/index.html` ↔ inline dashboard ↔ server selector | one canonical page exposes every retained operator panel | High REPO-008: static-first selection hides Phase C, Model Scope, and TTS in the normal checkout |
| inline summary/events request state | concurrent resources cannot erase each other's failure state | High REPO-009: later sibling success clears the shared banner |
| inline panel response ↔ visible render state | malformed success produces a visible panel error | Medium REPO-010: catches write into hidden loading elements |
| PR #257 block processing | secondary research hooks must preserve the original block failure | Medium PR257-001: secondary failure replaces the primary exception |

## Stop-line source scan

The bounded tracked-Python scan covered production paths for `TODO`, `FIXME`, `NotImplementedError`, `except Exception`, `except BaseException`, and permissive status patterns. It was triage input, not an automatic defect generator. Abstract interface methods and catches that preserve a declared fallback were not relabeled merely because a marker existed.

Material scan-to-runtime links were proved separately:

- broad catches in checkpoint persistence convert failure into acknowledged success (REPO-002);
- validation-history parse/read catches remove the newest state rather than exposing it (REPO-005);
- callback timeout handling returns before worker authority is revoked (REPO-004);
- dashboard handler catches were not called defects without a response/client consequence.

No new finding was created from a marker alone.

## Documentation versus mechanism (L9/L13)

- `docs/evidence-dashboard-runbook-2026-05-06.md:18` describes the validation surface as the latest pass state, while `load_validation_history()` can silently promote an older passing record after the newest record fails to parse.
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/planning/2026-05-reconciliation/panel-analysis/07-architecture-planning.md:99` calls the workflow executor production-grade; the current no-callback path can report an untouched finding verified with blank evidence.
- `enable_adaptive_threshold()` documents safety bounds but commits enabled state before validation and does not enforce an ordered bound pair.
- The server implements Phase C, Model Scope, and TTS handlers/panels, but deterministic static-first page selection omits those panels from the normal served surface.

These comparisons support the executable findings; prose alone is not the defect evidence.

## Evidence boundary

Obtained: exact root and PR heads, all dirty worktrees, all open PR discussion/check surfaces, local failure probes, source maps, focused unit modules, and a fresh repository-wide suite at round close. Not obtained: product fixes, a checked-in served-page browser gate, deployment, official Spark execution, hardware/model opt-in runs, or a fresh-checkout commit containing this audit. Those remain explicit skipped/deferred scope and cannot be inferred from installed tooling or passing unit tests.
