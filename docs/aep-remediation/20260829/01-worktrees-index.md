# Tier 1 — worktrees and uncommitted state

Commands: `git worktree list --porcelain`; per-worktree `rev-parse HEAD`, `branch --show-current`, `status --porcelain=v1 --untracked-files=all`; open-PR branch matching through `gh pr list`.

| # | Worktree / branch | HEAD | State | Open PR / disposition |
|---:|---|---|---|---|
| 1 | root / `codex/prime-ring-onion-method-dev` | `65c9085` | dirty: 7 tracked, 74 untracked | #296; finding WT001-001 |
| 2 | `agent-build` / `codex/recover-drift-research-20260722` | `b831493` | clean; 1 unmatched commit | no PR |
| 3 | `h2-rerun` / `feat/h2-swap-rerun-clean` | `6c3e184` | dirty: 24 tracked deletions | no PR; WT003-001 |
| 4 | `pr292-recondition` / `codex/pr292-metric-lineage-recondition` | `f2e41d4` | clean; 7 unmatched commits | public #292 is a different head |
| 5 | `pr293-fix` / `lattice/rung13-disintegration-20260714` | `454e4a3` | clean | #293 |
| 6 | `pr294-fix` / `lattice/rung17-diskpool-20260714` | `6e78cf4` | clean | #294 |
| 7 | `pr295-fix` / `lattice/rung16-routing-20260714` | `730b305` | clean | #295 |
| 8 | `relaxed-wozniak-271e04` / detached | `bf23a47` | clean | no PR |
| 9 | `semantic-cache-h1` / `codex/crsv-onion-method-dev` | `5937881` | clean; 6 unmatched commits | no PR |
| 10 | `waypoint-recovery` / `codex/recover-waypoint-research-20260722` | `65ae99a` | clean; 1 unmatched commit | no PR |
| 11 | `CHELATEDAI-EGV-ARCH` | `ebf6ebd` | clean; patch-equivalent in main | no PR |
| 12 | `CHELATEDAI-EGV-AVO-NEMOTRON` | `8ce7feb` | dirty: `.gitleaks-egv-avo.json` | #308; owner disposition, no runtime finding |
| 13 | `CHELATEDAI-EGV-BD-CAMPAIGN` | `64e60c3` | clean | no PR |
| 14 | `CHELATEDAI-EGV-CAMPAIGN` / `main` | `0cb5c85` | clean; 19 behind origin/main | base worktree |
| 15 | `CHELATEDAI-EGV-EVAL` | `1d1371c` | clean | no PR |
| 16 | `CHELATEDAI-EGV-HELDOUT-PROD` | `d72edd6` | clean | no PR |
| 17 | `CHELATEDAI-EGV-INTEGRATION` | `841b314` | dirty: `source.bin`, `target.bin`; 18 unmatched commits | no PR; residue only, no demonstrated runtime defect |
| 18 | `CHELATEDAI-EGV-ORCHESTRATOR` | `6db80e8` | clean; 2 unmatched commits | no PR |
| 19 | `CHELATEDAI-EGV-TRAINING` | `97a8237` | dirty: 3,898 Git-visible `.venv-egv-prod` paths; remote gone | WT019-001 |
| 20 | `CHELATEDAI-RHPC1` | `3059864` | dirty: 4 tracked, 9 untracked; no committed delta from origin/main | WT020-001 through WT020-004 |

### Tier 1 conclusions

- Six worktrees are dirty; no cleanup, restoration, staging, or deletion was authorized.
- RHPC passes 15 focused tests only in a dirty tree. Its tracked packaging/docs plus untracked implementation is not durable evidence.
- H2 deletions remain recoverable from Git, but committing them would break live document pointers unless an explicit retirement/supersession map is published.
- No dirty worktree touched the dashboard/browser FE roots.

Findings: [WT001-001](findings/AEP-20260829-WT001-001.md), [WT003-001](findings/AEP-20260829-WT003-001.md), [WT019-001](findings/AEP-20260829-WT019-001.md), [WT020-001](findings/AEP-20260829-WT020-001.md), [WT020-002](findings/AEP-20260829-WT020-002.md), [WT020-003](findings/AEP-20260829-WT020-003.md), [WT020-004](findings/AEP-20260829-WT020-004.md).
