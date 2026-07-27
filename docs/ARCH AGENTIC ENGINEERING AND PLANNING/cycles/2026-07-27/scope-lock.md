# Scope Lock — Lossless Preservation and Merge

## Cycle ID

- `AEP-20260727-7`

## Scope

- PR range: `PR000` planning/audit plus the preservation PR or dependency-aware
  PR stack required to make all validated local work GitHub-durable.
- Date window: local work and residuals present through 2026-07-27.
- Refinement report:
  `docs/research/prime-ring-preservation-merge-plan-2026-07.md` (created and
  completed during this cycle).
- Branch baseline: refreshed `origin/main` plus the exact heads of every branch
  listed by `git worktree list --porcelain`; initial primary head
  `41779be47442fc74fcd93e0c68bd7ca9dc115b99`.
- Included:
  - the primary dirty/untracked prime-ring METHOD_DEV tree;
  - every linked worktree, local branch, remote branch, and stash;
  - ignored recovery bundles, experiment outputs, and retired-session material
    that may be the only copy of valuable work;
  - artifact provenance, GitHub upload suitability, exact-head tests, PR
    review, merge, and post-merge reachability verification.
- Non-goals:
  - deleting, pruning, resetting, regenerating retained evidence, dropping
    stashes, removing worktrees, or otherwise cleaning residual state;
  - new scientific experiments or feature expansion;
  - claiming novelty, production readiness, or scientific promotion beyond the
    evidence already recorded.

## Approvals

- Orchestrator: Codex `/root`.
- Approver: repository owner, by explicit 2026-07-27 task instruction to
  preserve, publish, and merge without material loss.
- Date: 2026-07-27.
- Cleanup approval: explicitly not granted. A separate item-by-item owner
  sign-off is required after merged-state verification.
