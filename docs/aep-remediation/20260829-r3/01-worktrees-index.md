# Round 3 Tier 1 worktree index

Tier 1 opened all eight dirty worktrees after the Tier 0 freeze and before any PR body/comment/diff/log review. Two AVO worktrees changed during read-only inspection; they are explicitly blocked moving targets rather than false stable findings.

| Worktree / baseline | Tier 1 state | Disposition |
|---|---|---|
| root `65c9085cd048e8a7351a53e87666fdd5639e612b` | 7 tracked product modifications plus research/artifact/audit untracked paths | net-new Medium `WT001-001`; Round 2 QSCCI Critical remains duplicate carry; paired-intervention 25-test module passed |
| `h2-rerun@6c3e1847830ec231157a421735a5a34d7d71a658` | 24 tracked result/manifest deletions | exact duplicate of Round 1 `WT003-001`; no new root cause |
| AVO corpus runner `c459d1599a83ec88d96981db07052c3ef4009a27` | changed twice during inspection | `MOVING_TARGET / BLOCKED`; no defect claim until writer pauses and a fresh snapshot is frozen |
| AVO Nemotron `dc295442af7623acfd69b1c487a59b80cd4dc78f` | one untracked redacted scanner result | operational residue only; no FE/BE product defect |
| AVO runtime `88b0f389fa56c9c879c0c6a04e7f3fceef9dc243` | deletion became two different modified versions during inspection | `MOVING_TARGET / BLOCKED`; no defect claim |
| EGV integration `841b314b3f1e1524fab035a2bf2cb14cd0865c3b` | two identical untracked 28-byte files, zero references | vacuous residue, not product wiring or test evidence; no finding |
| EGV training `97a8237e3f1b92425b6802895c1fcbdeafeb2681` | 3,898 untracked environment paths | exact duplicate of Round 1 `WT019-001`; no product source in cohort |
| RHPC1 `305986427850dcb393a6078900201a7aa84c5b33` | 4 tracked docs/package changes plus 9 untracked implementation/test paths | two net-new High findings `WT025-001/002`; four Round 1 RHPC packets reverified as duplicates |

## Moving-target evidence

- Corpus runner: `22:02:31-04:00` four tracked files, `319+/13-`; `22:08:55` five tracked plus a new untracked DSL document, patch ID `5d74cf352d7b82d1bc0482aa130af2bd9f5c23fd`.
- AVO runtime: `22:02:31` scheduler deleted (`0+/1349-`); `22:05:41` it reappeared at 23,726 bytes / SHA-256 `4CC897DDDEB949BD34834DE56D28BE23882937DB10784CF64976E0A2B5912A13`; `22:08:55` it was 71,317 bytes / SHA-256 `C566351D4007C38068036C60E6EABDF7D8F706C686433CD7F8EFA0715CC544B6`, patch ID `8e02f3f4aa35b022cc2d7ca43a9d1ee64a7c908f`.

No audit agent wrote either worktree. Dependency for both: pause the owning writer and take a new exact worktree/porcelain/patch snapshot. Repeated reading would create a blended artifact, not stronger evidence.

## Duplicate carry-forwards

- `h2-rerun`: docs still point to the 22 deleted result JSONs and two deleted manifests (`AEP-20260829-WT003-001`).
- EGV training: `.gitignore` ignores `venv/` but not `.venv-egv-prod/`; 3,898 visible paths remain (`AEP-20260829-WT019-001`).
- RHPC1: tracked-to-untracked wiring, forgeable source identity, predecessor reread, and check-then-replace publication remain `WT020-001` through `WT020-004`.
- Root: Round 2 `AEP-20260829-r2-WT-001` remains controlling for service restoration before terminal/reaped child proof.

## Frontend disposition

`NO_NEW_FE_FINDINGS`: known frontend paths, web assets/configs, and tracked route/UI diff tokens were zero across all eight dirty worktrees. Two untracked lexical `document.get(...)` hits were Python JSON-map access, not DOM code. No browser/server/network gate was run.
