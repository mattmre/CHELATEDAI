# Orchestrator Briefing (ARCH-AEP)

Purpose: Quick reference for the ARCH-AEP documentation set, plus the narrative to start a new session.

Full file index: `docs/INDEX.md`

## What Exists In This Folder
- `README.md`: narrative overview and intent for ARCH-AEP.
- `workflow.md`: end-to-end workflow specification with phases, guardrails, and enhancements.
- `templates.md`: ID and branch conventions, tracker table format, and status log.
- `backlog-template.md`: template for the master backlog file.
- `backlog-index.md`: index of backlog files.
- `next-session.md`: short handoff checklist for resuming work.
- `phase-planning.md`: long-running planning record for the current cycle.
- `schedule-and-tracking.md`: cadence, gates, and milestone tracking.
- `agent-learning.md`: cross-session learnings and reusable patterns.
- `change-log.md`: scope/defer decisions with audit context.
- `risk-memo-template.md`: Critical/High risk memo template.
- `risk-memos/README.md`: storage location for risk memos.
- `phase-summary-template.md`: template for per-PR or per-phase summaries.
- `phase-summaries/README.md`: storage location for phase summaries.
- `verification-log.md`: test/build evidence mirror for PRs.
- `tracker-pointer.md`: pointer to the active tracker file.
- `tracker-index.md`: index of tracker files.
- `scope-lock-template.md`: template for scope lock record.
- `glossary.md`: shared definitions for ARCH-AEP terms.
- `test-matrix.md`: baseline and conditional test guidance.
- `active-tracker-template.md`: tracker template with editor lock.
- `cycle-layout-template.md`: optional structure for parallel cycles.
- `tier-close-checklist.md`: checklist to complete before moving to next tier.
- `cycle-summary-template.md`: end-of-cycle summary template.
- `cycle-summaries/README.md`: storage location for cycle summaries.

## Orchestrator Narrative (Session Start)
CODEX ARCH AGENTIC ENGINEERING AND PLANNING
Begin an ARCH-AEP cycle. Scope-lock the PR range and dates, then ingest `docs/agentic-review-framework.md` and the latest refinement report. Normalize all findings into a single master backlog with unique IDs using `docs/ARCH AGENTIC ENGINEERING AND PLANNING/templates.md` (Option A). De-duplicate and assign provisional severity. Spawn specialist agents (Architecture, Security, Testing, Performance, Reliability/Debug, Documentation, UX) to re-validate findings on current `main`, attach exact file paths and line ranges, and propose the smallest safe fix with acceptance criteria and effort sizing (S/M/L). Orchestrator merges validated findings, re-sorts strictly by Severity (Critical > High > Medium > Low), breaking ties by impact × (1/effort) and dependency order. For each tier, generate small remediation PRs (one coherent theme per PR) and do not advance to the next tier until the current tier is empty or formally blocked with a written unblock plan in `docs/ARCH AGENTIC ENGINEERING AND PLANNING/change-log.md`. Maintain the auditable tracker at `docs/refinement-remediation/YYYY-MM-DD_tracker.md` and record commands/results for build and tests after each PR. Use `docs/ARCH AGENTIC ENGINEERING AND PLANNING/phase-planning.md` for long-running planning, `docs/ARCH AGENTIC ENGINEERING AND PLANNING/next-session.md` for session handoffs, `docs/ARCH AGENTIC ENGINEERING AND PLANNING/schedule-and-tracking.md` for cadence and gates, and log cross-session learnings in `docs/ARCH AGENTIC ENGINEERING AND PLANNING/agent-learning.md`. For Critical/High findings, create a risk memo from `docs/ARCH AGENTIC ENGINEERING AND PLANNING/risk-memo-template.md`. Post a concise phase summary after each PR: what closed, what remains, what’s next. Repeat until all severities are remediated or formally deferred with scheduled follow-up.

## Cycle Start Checklist
- Create scope lock record from `docs/ARCH AGENTIC ENGINEERING AND PLANNING/scope-lock-template.md`.
- Create backlog file from `docs/ARCH AGENTIC ENGINEERING AND PLANNING/backlog-template.md`.
- Create tracker file from `docs/ARCH AGENTIC ENGINEERING AND PLANNING/active-tracker-template.md`.
- Update `docs/ARCH AGENTIC ENGINEERING AND PLANNING/backlog-index.md`.
- Update `docs/ARCH AGENTIC ENGINEERING AND PLANNING/tracker-index.md`.
- Update `docs/ARCH AGENTIC ENGINEERING AND PLANNING/tracker-pointer.md`.
- Confirm pointer, backlog index, and tracker index are updated by the same owner.

## Cycle Close Checklist
- Ensure all findings are merged or deferred with target date + exit criteria.
- Ensure verification evidence is recorded in the tracker and `verification-log.md`.
- Record final phase summary in `docs/ARCH AGENTIC ENGINEERING AND PLANNING/phase-summaries/`.
- Record cycle summary in `docs/ARCH AGENTIC ENGINEERING AND PLANNING/cycle-summaries/`.
- Archive closed risk memos to `docs/ARCH AGENTIC ENGINEERING AND PLANNING/risk-memos/closed/`.
- Confirm defer entries include target date + exit criteria in `change-log.md`.
