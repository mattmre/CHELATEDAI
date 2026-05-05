# Next Session Checklist

Purpose: Continue the Model-Scope ARCH-AEP cycle and build a true model-hook steering layer without overstating what the current repo can safely persist.

## Session Start
- Review `docs/model-scope-steering-architecture-2026-05-01.md`.
- Review `docs/ARCH AGENTIC ENGINEERING AND PLANNING/architecture-2026-05-01-model-scope-roadmap.md`.
- Review `docs/ARCH AGENTIC ENGINEERING AND PLANNING/cycles/2026-05-01/backlog-2026-05-01.md`.
- Review `docs/ARCH AGENTIC ENGINEERING AND PLANNING/cycles/2026-05-01/tracker-2026-05-01.md`.
- Review `docs/qwen-scope-engine-mapping-2026-04-30.md`.
- Sync local `main` to `origin/main`.
- Confirm local model/runtime dependencies before starting implementation.

## Priority Order
1. **Build the hook runtime first.**
   - do not start with steering policy code before a deterministic local hook bus exists
   - keep the first slice observation-only
2. **Use `Qwen3.5-9B` as the primary serious pilot.**
   - use `Qwen3.5-2B` or `Qwen3-1.7B` only for smoke/debug loops
   - do not pivot the primary plan to `Qwen3.6` until the hook pipeline is stable
3. **Add sparse feature extraction second.**
   - load official Qwen-Scope artifacts where supported
   - provide fallback feature summaries for unsupported targets
4. **Keep all steering fail-closed and shadow-mode at first.**
   - no persistent base-weight edits
   - no always-on intervention path
5. **Treat segmented memory as typed infrastructure, not a generic dump.**
   - separate working, episode, expectation, and persistent stores
   - add retention and promotion boundaries before scale
6. **Only then add training and promotion.**
   - promote overlays, probes, or steering artifacts only after replay and rollback checks
   - treat base-weight mutation as a later research phase, not the initial path

## Current State
- The repo now has a canonical plan for `Model-Scope`, which is distinct from the earlier Engine-Scope cycle.
- Engine-Scope remains useful as the engine-side telemetry/control layer and should not be deleted.
- The current missing component is a local hook-and-feature runtime, not another retrieval-only search loop.
- **2026-05-04:** `BlockAttnResAdapter` ("attnres") and `LayerAttentionAggregator` were added to `chelation_adapter.py`, based on MoonshotAI's Attention Residuals paper. The `LayerAttentionAggregator` is the planned bridge between the hook bus and AttnRes-style cross-layer aggregation — it is implemented but not yet wired into `model_scope_runtime.py`. See `docs/attnres-adapter-implementation-2026-05-04.md`.

## Handoff Notes
- Do not add `pytest` imports to `test_*.py`; CI does not install `pytest`.
- Python 3.9 CI: avoid runtime `X | None` annotations unless module uses deferred annotations.
- `ruff check` does not validate GitHub Actions YAML.
- Prefer compact, versioned artifact schemas over large raw dumps.
- Persistent promotion should target overlays and memory artifacts before any discussion of base-weight mutation.
- The canonical tracker pointer now lives at `docs/ARCH AGENTIC ENGINEERING AND PLANNING/cycles/2026-05-01/tracker-2026-05-01.md`.

## Cycle ID
- AEP-2026-05-01
