# Master remediation backlog

This is the authoritative tracker for cycle `AEP-20260829-1`. This control ledger binds ownership, exact target state, evidence disposition, and per-finding verification ID; the ranked table below it controls execution order.

Repair-commit field for every row: `NOT_CREATED_PLAN_ONLY`. A lesser fix agent must replace that value with its immutable candidate commit in the required handoff; this audit does not invent product commits.

| Finding | Owner | Target branch / SHA | Evidence result | Verification ID |
|---|---|---|---|---|
| PR257-001 | operator assigns replacement owner | PR #257 `663be66e3292aae654a0f615e7c5e2b4dd7a98d4` → clean replacement | source defect confirmed; runtime repair blocked on replacement | `VER-AEP-20260829-PR257-001` |
| WT020-002 | operator defines authority | `codex/rhpc1-method-dev-20260825@305986427850dcb393a6078900201a7aa84c5b33` dirty | gate absence confirmed; blocked | `VER-AEP-20260829-WT020-002` |
| PR296-001 | operator PR authority | live #292–#295 heads recorded in finding | live contradiction reproduced; blocked | `VER-AEP-20260829-PR296-001` |
| PR256-001 | operator assigns replacement owner | PR #256 `4379b9cdeaa54b1b53f043f034ffc68bb60494d4` → clean replacement | source/lint defect confirmed; blocked on target | `VER-AEP-20260829-PR256-001` |
| PR256-002 | operator assigns replacement owner | PR #256 `4379b9cdeaa54b1b53f043f034ffc68bb60494d4` → clean replacement | ownership leak confirmed; blocked on target | `VER-AEP-20260829-PR256-002` |
| FE003 | dashboard browser-harness owner | root baseline `65c9085cd048e8a7351a53e87666fdd5639e612b` | unsafe sinks confirmed; blocked on real browser harness | `VER-AEP-20260829-FE003` |
| FE006 | auth-contract owner | root baseline `65c9085cd048e8a7351a53e87666fdd5639e612b` | bypass paths confirmed; blocked by FE001 contract | `VER-AEP-20260829-FE006` |
| WT020-003 | RHPC boundary owner | RHPC dirty baseline `305986427850dcb393a6078900201a7aa84c5b33` | TOCTOU confirmed; blocked by WT020-002 | `VER-AEP-20260829-WT020-003` |
| WT020-004 | RHPC boundary owner | RHPC dirty baseline `305986427850dcb393a6078900201a7aa84c5b33` | publication race confirmed; blocked by WT020-002 | `VER-AEP-20260829-WT020-004` |
| FE001 | operator/auth threat-model owner | root baseline `65c9085cd048e8a7351a53e87666fdd5639e612b` | real HTTP boundary reproduced; blocked on auth model | `VER-AEP-20260829-FE001` |
| FE002 | report-contract owner | root baseline `65c9085cd048e8a7351a53e87666fdd5639e612b` | producer absence confirmed; blocked on produce/remove decision | `VER-AEP-20260829-FE002` |
| PR257-002 | operator assigns replacement owner | PR #257 `663be66e3292aae654a0f615e7c5e2b4dd7a98d4` → clean replacement | nesting defect confirmed; blocked on target | `VER-AEP-20260829-PR257-002` |
| BE001-001 | operator accepts rulebook migration | root baseline `65c9085cd048e8a7351a53e87666fdd5639e612b` | v3.3 drift confirmed; blocked | `VER-AEP-20260829-BE001-001` |
| PR257-003 | operator assigns replacement owner | PR #257 `663be66e3292aae654a0f615e7c5e2b4dd7a98d4` → clean replacement | envelope gap confirmed; blocked on target | `VER-AEP-20260829-PR257-003` |
| BE002-001 | unassigned backend fix agent | root dirty baseline `65c9085cd048e8a7351a53e87666fdd5639e612b` | deterministic pre-stage expiry probe reproduced; ready | `VER-AEP-20260829-BE002-001` |
| FE004 | dashboard browser-harness owner | root baseline `65c9085cd048e8a7351a53e87666fdd5639e612b` | keyboard defect confirmed; blocked on browser/a11y harness | `VER-AEP-20260829-FE004` |
| FE005 | lifecycle-manifest owner | root baseline `65c9085cd048e8a7351a53e87666fdd5639e612b` | false lifecycle confirmed; blocked | `VER-AEP-20260829-FE005` |
| WT003-001 | campaign owner | `h2-rerun` baseline inventoried in worktree index | deletion/pointer conflict confirmed; blocked | `VER-AEP-20260829-WT003-001` |
| WT001-001 | operator path owner | root dirty baseline `65c9085cd048e8a7351a53e87666fdd5639e612b` | mixed lanes confirmed; blocked | `VER-AEP-20260829-WT001-001` |
| WT020-001 | RHPC branch owner | RHPC dirty baseline `305986427850dcb393a6078900201a7aa84c5b33` | undurable wiring confirmed; blocked | `VER-AEP-20260829-WT020-001` |
| WT019-001 | unassigned hygiene fix agent | `venv-gitignore-20260820` baseline inventoried in worktree index | environment leak confirmed; ready | `VER-AEP-20260829-WT019-001` |

Sort key: severity, then impact × inverse effort, then dependency readiness. Impact weights Critical=4, High=3, Medium=2, Low=1; effort divisors S=1, M=2, L=3. A dependency may block execution but does not falsify the numeric ordering.

| Rank | Finding | Layer | Sev | Effort | Score | Status | Dependencies / next owner |
|---:|---|---|---|---|---:|---|---|
| 1 | [PR257-001](findings/AEP-20260829-PR257-001.md) fallback config TypeError | backend | Critical | S | 4.00 | blocked | operator assigns clean replacement branch/owner |
| 2 | [WT020-002](findings/AEP-20260829-WT020-002.md) reviewed-source SHA gate absent | boundary | Critical | M | 2.00 | blocked | operator defines trusted authorization source |
| 3 | [PR296-001](findings/AEP-20260829-PR296-001.md) rejected PRs still claim BHS100 | boundary | Critical | M | 2.00 | blocked | operator withdrawal/replacement authority; fresh exact-head Tier B |
| 4 | [PR256-001](findings/AEP-20260829-PR256-001.md) use-before-assignment | backend | High | S | 3.00 | blocked | operator assigns retained replacement lineage |
| 5 | [PR256-002](findings/AEP-20260829-PR256-002.md) mutable registry leak | backend | High | S | 3.00 | blocked | operator assigns retained replacement lineage |
| 6 | [FE003](findings/AEP-20260829-FE003.md) artifact DOM injection | frontend | High | S | 3.00 | blocked | real browser harness; external + inline clients atomically |
| 7 | [FE006](findings/AEP-20260829-FE006.md) errors rendered as empty success | boundary | High | S | 3.00 | blocked | FE001 auth and FE002 report contract decisions |
| 8 | [WT020-003](findings/AEP-20260829-WT020-003.md) predecessor hash/verify TOCTOU | boundary | High | M | 1.50 | blocked | WT020-002 authority contract and RHPC owner |
| 9 | [WT020-004](findings/AEP-20260829-WT020-004.md) absent-target publication race | boundary | High | M | 1.50 | blocked | WT020-002 authority contract and RHPC owner |
| 10 | [FE001](findings/AEP-20260829-FE001.md) browser auth path absent | boundary | High | M | 1.50 | blocked | operator selects browser auth/session threat model |
| 11 | [FE002](findings/AEP-20260829-FE002.md) test report producer absent | boundary | High | M | 1.50 | blocked | report owner chooses producer or feature removal |
| 12 | [PR257-002](findings/AEP-20260829-PR257-002.md) nested state isolation leak | boundary | High | M | 1.50 | blocked | operator assigns clean replacement branch/owner |
| 13 | [BE001-001](findings/AEP-20260829-BE001-001.md) BHS automation remains v3.3 | boundary | High | M | 1.50 | blocked | operator accepts v3.7.1 migration source |
| 14 | [PR257-003](findings/AEP-20260829-PR257-003.md) serialization error escapes envelope | backend | Medium | S | 2.00 | blocked | operator assigns clean replacement branch/owner |
| 15 | [BE002-001](findings/AEP-20260829-BE002-001.md) deadline stage-selection assumption | backend | Medium | S | 2.00 | ready-for-fix-agent | none |
| 16 | [FE004](findings/AEP-20260829-FE004.md) keyboard-inaccessible tabs | frontend | Medium | S | 2.00 | blocked | dashboard browser/a11y harness |
| 17 | [FE005](findings/AEP-20260829-FE005.md) sweep always appears running | boundary | Medium | S | 2.00 | blocked | backend lifecycle-manifest owner |
| 18 | [WT003-001](findings/AEP-20260829-WT003-001.md) evidence deletions vs doc pointers | boundary | Medium | M | 1.00 | blocked | campaign owner; CD-MLR-01 |
| 19 | [WT001-001](findings/AEP-20260829-WT001-001.md) mixed root dirty lanes | boundary | Medium | M | 1.00 | blocked | operator path ownership/disposition |
| 20 | [WT020-001](findings/AEP-20260829-WT020-001.md) tracked wiring/untracked RHPC | boundary | Medium | M | 1.00 | blocked | WT020-002/003/004 + branch owner |
| 21 | [WT019-001](findings/AEP-20260829-WT019-001.md) unignored environment | boundary | Low | S | 1.00 | ready-for-fix-agent | none |

## Recommended atomic slices

1. Stop public truth drift: PR296-001 operator disposition.
2. New PR257 replacement: PR257-001 first, then PR257-002/003; do not revive the conflicting draft by self-attestation.
3. RHPC trust boundary: WT020-002/003/004, then WT020-001 durability/build.
4. Dashboard contract: FE001 + FE006; FE002 producer; FE003 safe rendering; FE004 browser keyboard gate; FE005 only after backend manifest ownership.
5. Shim replacement: PR256-001/002 on a current-base atomic branch.
6. Process enforcement: BE001-001 as a dedicated convention/tooling migration.
7. Owner-only worktree normalization: WT001/003/019.

No finding is `open` without a disposition: actionable items are `ready-for-fix-agent`; authority/contract dependencies are `blocked`. No duplicate row was retained.
