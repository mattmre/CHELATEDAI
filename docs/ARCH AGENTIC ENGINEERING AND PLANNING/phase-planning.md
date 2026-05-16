# Phase Planning (Long-Running)

Purpose: Single long-running planning record for the current ARCH-AEP cycle.

## Cycle Metadata
- Cycle start date: 2026-05-01
- Orchestrator: Codex
- PR range: `PR000` planning cycle plus future Model-Scope implementation PR series
- Refinement report:
  - `docs/model-scope-steering-architecture-2026-05-01.md`
  - `docs/ARCH AGENTIC ENGINEERING AND PLANNING/architecture-2026-05-01-model-scope-roadmap.md`
  - `docs/qwen-scope-engine-mapping-2026-04-30.md`
- Scope lock date/time: 2026-05-01 America/New_York
- Scope lock file: `docs/ARCH AGENTIC ENGINEERING AND PLANNING/cycles/2026-05-01/scope-lock-2026-05-01.md`
- PR list hash: planning-only scope lock; implementation PR list pending branch creation
- Cycle ID: `AEP-2026-05-01`

## Phase Goals
- Phase 1: build a local model runtime and hook bus for supported pilot models
- Phase 2: add Qwen-Scope-backed sparse feature extraction and fallback feature summaries
- Phase 3: add fail-closed shadow-mode steering actuators with provenance
- Phase 4: add segmented memory and expectation-comparator infrastructure
- Phase 5: add bounded overlay training, replay, and promotion gates
- Phase 6: integrate Model-Scope into ChelatedAI engine and evaluation surfaces

## Current Phase Contract
- Phase name:
- Phase owner:
- In scope:
- Out of scope:
- Entry gates:
- Exit gates:
- Promotion target:

## Phase Loop Checklist
- Architecture scope lock complete:
- Implementation PRs scoped:
- ARCH-AEP review complete:
- Code analysis / hardening complete:
- Promote or defer decision logged:

## Backlog Summary
- Critical: 0
- High: 3
- Medium: 3
- Low: 1

## Dependencies And Risks
- The hook runtime must exist before sparse features, steering, or memory promotion can be trusted.
- Persistent promotion must target overlays and artifacts first, not base model weights.
- The initial pilot should track verified feature-support surfaces rather than chase the newest model family prematurely.
- Segmented memory must remain typed and bounded or the cycle will collapse into unauditable persistence.

## Remediation Strategy
- Use the new cycle folder under `cycles/2026-05-01/` as the canonical implementation record.
- Land the work in small PRs that match the first three branches in `architecture-2026-05-01-model-scope-roadmap.md`.
- Keep Engine-Scope intact as the engine-side precursor layer rather than replacing it.
- Treat Qwen3.5-9B as the primary pilot, with smaller Qwen targets reserved for smoke and debug work.

## Retrospective (Mid-Cycle Changes)
Format: `YYYY-MM-DD - change - rationale`

## Decision Log
Format: `YYYY-MM-DD - decision - rationale`
- `2026-05-01 - open a dedicated Model-Scope cycle - true model-hook steering is now a distinct implementation program, not just a continuation of Engine-Scope`
- `2026-05-01 - select Qwen3.5-9B as the primary hook pilot - it matches the desired local size class and has verified official Qwen-Scope support`
- `2026-05-01 - defer persistent base-weight mutation - the first safe promotion surface is overlays, probes, and typed memory artifacts`
