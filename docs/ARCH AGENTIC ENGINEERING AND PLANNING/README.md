# ARCH AGENTIC ENGINEERING AND PLANNING (ARCH-AEP)

ARCH-AEP is a manual-checkpoint workflow that sends a coordinated swarm of agents back through historical PRs to surface deferred work, normalize it into a single master refinement backlog, and drive remediation in strict severity order. It extends the micro refinement loop used for the last 10 PRs into a phase-level engine that activates after full phases complete or at explicit human checkpoints.

This workflow is designed for breadth and accountability: it is intentionally slow, audit-heavy, and conservative. The output is a master listing of refinement work with verified file paths, actionable fixes, and a recorded decision trail (implemented, deferred, or blocked). The loop continues until all severities are resolved or formally deferred with a scheduled follow-up.

Core idea:
- Architecture and planning happen before remediation.
- Agents re-validate every inherited finding against current main.
- Work is cut into small, coherent PRs by severity tier.
- Remediation does not advance tiers until the current tier is empty or formally blocked.
- Every step leaves auditable traces (tracker, PR links, evidence).

Use ARCH-AEP when:
- A phase finishes and the team needs a deep cleanup pass.
- The backlog of deferred fixes is growing or unclear.
- You need an authoritative master list of refinement work to drive the next cycle.

What success looks like:
- A single, normalized master backlog with verified findings and acceptance criteria.
- A tracker mapping each finding to owning agent, PR, status, and evidence.
- Remediation PRs that are reviewable, cohesive, and tested.
- Clear defer/blocked rationales with scheduled follow-ups.

Key reference:
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/orchestrator-briefing.md`

Cycle checklists:
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/orchestrator-briefing.md`

Authoritative file list:
- `docs/INDEX.md`

Minimum artifacts per cycle:
- Scope lock record
- Backlog file
- Tracker file
- Tracker pointer
- Backlog index entry
- Tracker index entry
- Verification log entries

Cycle ID format:
- `AEP-YYYYMMDD-N` (N is a global increment)

Cycle ID allocation:
- Increment N sequentially in `docs/ARCH AGENTIC ENGINEERING AND PLANNING/tracker-index.md`.

Default per-cycle evidence log:
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/cycles/YYYY-MM-DD/verification-log.md`

Additional references:
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/workflow.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/templates.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/next-session.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/phase-planning.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/schedule-and-tracking.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/tier-close-checklist.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/cycle-summary-template.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/cycle-summaries/README.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/backlog-template.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/backlog-index.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/tracker-pointer.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/tracker-index.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/verification-log.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/phase-summary-template.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/phase-summaries/README.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/risk-memo-template.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/risk-memos/README.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/risk-memos/closed/README.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/agent-learning.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/change-log.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/scope-lock-template.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/glossary.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/test-matrix.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/active-tracker-template.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/cycle-layout-template.md`
