# Architecture — Session 32 PR Orchestration

## Goal
Keep the remaining post-Session-31 work moving without contaminating the live overnight campaign. The orchestration strategy for Session 32 is to merge small independent follow-through PRs first, keep the campaign worktree frozen while the long run is active, and push all later documentation and preset decisions from separate worktrees.

## Branch Layout

### PR #107
- branch: `feat/session32-overnight-campaign`
- scope:
  - harden `run_overnight_campaign.py` before deeper Session 32 execution
  - normalize and bound run labels
  - prevent same-second run-dir collisions
  - ignore `experiment_runs/`

### PR #108
- branch: `feat/session32-retention-review`
- scope:
  - execute the dated computational-storage retention review
  - delete the remaining February Tier C remote backup refs
  - record the review outcome in the retention-policy document

### Session 32 campaign execution
- worktree: `C:\GitHub\repos\CHELATEDAI`
- scope:
  - run the bounded overnight campaign without changing the underlying worktree state
  - keep all later branch work in separate worktrees until the campaign exits

### Session wrap branch
- branch: `docs/session32-wrap`
- scope:
  - keep the Session 32 log current while the campaign is still in progress
  - capture orchestration notes, merged PRs, active constraints, and later campaign conclusions
  - refresh `next-session.md` only after the campaign reaches a promotion/no-promotion decision

## Operational Notes
- The campaign was launched before PR #107 merged, but the merged hardening still protects all subsequent launches and removes path/context-rot issues uncovered by the live run.
- Separate worktrees were required after the campaign started, because the parent runner launches later subprocesses from the same file tree. Switching branches in the original worktree would invalidate the experiment.
- The retention review was safe to merge independently because it only touched policy documentation and remote backup refs; it did not overlap the live campaign outputs.
