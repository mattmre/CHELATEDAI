# Architecture — Session 28 Overnight Orchestration

## Goal

Preserve the overnight implementation and documentation work as an auditable sequence instead of a single mixed branch.

## Branch Layout

### PR `#96`

- branch: `feat/session28-weight-refinement-recovery`
- scope:
  - benchmark hardening
  - adapter isolation
  - campaign resume support
  - durable Session 28 result memo

### PR `#97`

- branch: `docs/session28-roadmap-cleanup`
- scope:
  - annotate legacy hardening docs as historical
  - remove misleading active-roadmap implications

### Wrap PR

- branch: `docs/session28-wrap`
- scope:
  - session log
  - phase summaries
  - verification log
  - next-session refresh
  - CLAUDE learnings

## Operational Notes

- The live resumed BEIR-medium process was stopped once the session scope changed.
- No test suite was rerun in this session per user direction.
- Only lightweight non-test verification was recorded for the overnight branches.
