# Documentation Home

This is the canonical entrypoint for repository documentation.

If you need to understand the codebase quickly, read the docs in this order:

1. [../README.md](../README.md)
2. [SYSTEM_BLUEPRINT.md](SYSTEM_BLUEPRINT.md)
3. [MODULE_GUIDE.md](MODULE_GUIDE.md)
4. [RESEARCH_TRACKS.md](RESEARCH_TRACKS.md)
5. [COMPUTATIONAL_STORAGE_DRIVE_NODES.md](COMPUTATIONAL_STORAGE_DRIVE_NODES.md) for the storage-node track

## Canonical Documents

| Canonical doc | Purpose |
|---|---|
| [../README.md](../README.md) | Repo-level overview, quick start, validation commands, and docs entrypoints |
| [SYSTEM_BLUEPRINT.md](SYSTEM_BLUEPRINT.md) | System architecture, stack, CI, and information-flow diagrams |
| [MODULE_GUIDE.md](MODULE_GUIDE.md) | Module-by-module inventory of the Python runtime, evaluation tooling, and storage POC |
| [RESEARCH_TRACKS.md](RESEARCH_TRACKS.md) | Research themes, status, open questions, and artifact pointers |
| [live-fire-diagnostics-2026-04-27.md](live-fire-diagnostics-2026-04-27.md) | Deterministic live-fire diagnostics results, calibration guidance, and next campaign priorities |
| [llm-architecture-ai-engineering-adaptation-review-2026-04-27.md](llm-architecture-ai-engineering-adaptation-review-2026-04-27.md) | Modern LLM architecture and AI-engineering operations review mapped to ChelatedAI adaptation opportunities |
| [COMPUTATIONAL_STORAGE_DRIVE_NODES.md](COMPUTATIONAL_STORAGE_DRIVE_NODES.md) | Detailed summary of hard-drive / storage-node experiments and current scope limits |
| [computational-storage-transport-scope-decision.md](computational-storage-transport-scope-decision.md) | Formal claim boundary for the RP2040 transport path |
| [computational-storage-hardware-evidence-runbook.md](computational-storage-hardware-evidence-runbook.md) | Operator runbook for real hardware evidence capture |
| [roadmap-audit-and-weight-refinement-plan-2026-03-06.md](roadmap-audit-and-weight-refinement-plan-2026-03-06.md) | Current conclusion on completed implementation vs. remaining evaluation work |
| [INDEX.md](INDEX.md) | Broader index, including the AEP process archive and historical research docs |

## Legacy To Canonical Map

This repo accumulated several historical or session-scoped documents before the canonical docs set above existed. Use this table to route older filenames to the current source of truth.

| Older file or area | Canonical doc | Status / note |
|---|---|---|
| `TECHNICAL_ANALYSIS.md` | [SYSTEM_BLUEPRINT.md](SYSTEM_BLUEPRINT.md) | Historical architecture deep dive; useful background, but not the current high-level map |
| `COMPLETION_SUMMARY.md` | [RESEARCH_TRACKS.md](RESEARCH_TRACKS.md) and [../README.md](../README.md) | Historical milestone snapshot |
| `PR_DESCRIPTION.md` | [SYSTEM_BLUEPRINT.md](SYSTEM_BLUEPRINT.md) and [RESEARCH_TRACKS.md](RESEARCH_TRACKS.md) | Historical delivery summary, not the current product narrative |
| `REFACTORING_PLAN.md` | [roadmap-audit-and-weight-refinement-plan-2026-03-06.md](roadmap-audit-and-weight-refinement-plan-2026-03-06.md) | Contains stale "future phase" language relative to current `main` |
| `phase4-experiment-protocol.md` | [RESEARCH_TRACKS.md](RESEARCH_TRACKS.md) | Still useful for background, but no longer the best overview of the repo state |
| `computational_storage_poc/README.md` | [COMPUTATIONAL_STORAGE_DRIVE_NODES.md](COMPUTATIONAL_STORAGE_DRIVE_NODES.md) | Keep reading the POC README for subsystem quickstart details |
| `docs/ARCH AGENTIC ENGINEERING AND PLANNING/` | [INDEX.md](INDEX.md) | Process archive; authoritative for session-by-session evidence and tracker continuity |

## Documentation Refresh Workflow

After a live-fire or benchmark campaign:

1. Run the relevant validation command, starting with `python run_live_fire_diagnostics.py --output live_fire_results.json` for deterministic control-plane checks.
2. Capture the outcome in a dated campaign document under `docs/`.
3. Link the result from `docs/INDEX.md`, `docs/RESEARCH_TRACKS.md`, and this docs home.
4. Add the validation command and result to `docs/ARCH AGENTIC ENGINEERING AND PLANNING/verification-log.md`.
5. Keep generated raw JSON out of commits unless it is intentionally promoted as a durable campaign artifact.

## Recommended Reading Paths

### Understand the runtime

- [SYSTEM_BLUEPRINT.md](SYSTEM_BLUEPRINT.md)
- [MODULE_GUIDE.md](MODULE_GUIDE.md)

### Understand the research portfolio

- [RESEARCH_TRACKS.md](RESEARCH_TRACKS.md)
- [live-fire-diagnostics-2026-04-27.md](live-fire-diagnostics-2026-04-27.md)
- [roadmap-audit-and-weight-refinement-plan-2026-03-06.md](roadmap-audit-and-weight-refinement-plan-2026-03-06.md)

### Understand the storage-node work

- [COMPUTATIONAL_STORAGE_DRIVE_NODES.md](COMPUTATIONAL_STORAGE_DRIVE_NODES.md)
- [computational-storage-transport-scope-decision.md](computational-storage-transport-scope-decision.md)
- [computational-storage-hardware-evidence-runbook.md](computational-storage-hardware-evidence-runbook.md)
