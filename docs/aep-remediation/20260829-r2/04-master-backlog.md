# Round 2 master remediation backlog

This is the authoritative execution ledger for cycle `AEP-20260829-r2`. It adds twelve active packets to the twenty-one retained Round 1 packets. One Round 2 decomposition is marked duplicate and excluded from active counts. Every repair-commit value is `NOT_CREATED_PLAN_ONLY`; no product edit or candidate commit exists from this audit.

## Round 2 owner and evidence ledger

| Finding | Owner | Exact target | Evidence / verification ID | Repair commit |
|---|---|---|---|---|
| WT-001 | service-supervisor owner after operator recovery decision | root dirty overlay at `65c9085cd048e8a7351a53e87666fdd5639e612b` | live-child restoration reproduced / `R2-V0102` | `NOT_CREATED_PLAN_ONLY` |
| PR257-001 | replacement-branch backend owner | PR #257 `663be66e3292aae654a0f615e7c5e2b4dd7a98d4` → clean replacement | primary failure masked / `R2-V0202` | `NOT_CREATED_PLAN_ONLY` |
| REPO-001 | checkpoint owner | root HEAD `65c9085cd048e8a7351a53e87666fdd5639e612b` | same-second overwrite reproduced / `R2-V0301` | `NOT_CREATED_PLAN_ONLY` |
| REPO-002 | checkpoint owner | root HEAD `65c9085cd048e8a7351a53e87666fdd5639e612b` | false persistence success reproduced / `R2-V0302` | `NOT_CREATED_PLAN_ONLY` |
| REPO-003 | AEP lifecycle owner after receipt-contract decision | root HEAD `65c9085cd048e8a7351a53e87666fdd5639e612b` | untouched false closure reproduced / `R2-V0304` | `NOT_CREATED_PLAN_ONLY` |
| REPO-004 | AEP worker-runtime owner after isolation decision | root HEAD `65c9085cd048e8a7351a53e87666fdd5639e612b` | post-timeout mutation reproduced / `R2-V0305` | `NOT_CREATED_PLAN_ONLY` |
| REPO-005 | dashboard-history backend owner | root HEAD `65c9085cd048e8a7351a53e87666fdd5639e612b` | older green substitution reproduced / `R2-V0306` | `NOT_CREATED_PLAN_ONLY` |
| REPO-006 | generic adapter-router owner | root HEAD `65c9085cd048e8a7351a53e87666fdd5639e612b` | dimension failure reproduced / `R2-V0307` | `NOT_CREATED_PLAN_ONLY` |
| REPO-007 | adaptive-threshold owner | root HEAD `65c9085cd048e8a7351a53e87666fdd5639e612b` | non-atomic enable/reversed bounds reproduced / `R2-V0308` | `NOT_CREATED_PLAN_ONLY` |
| REPO-008 | dashboard owner after canonical-client decision | root HEAD `65c9085cd048e8a7351a53e87666fdd5639e612b` | component/selector map / `R2-V0310` | `NOT_CREATED_PLAN_ONLY` |
| REPO-009 | frontend state owner if inline client retained | root HEAD `65c9085cd048e8a7351a53e87666fdd5639e612b` | concurrent state trace / `R2-V0311` | `NOT_CREATED_PLAN_ONLY` |
| REPO-010 | frontend response-state owner if inline panels retained | root HEAD `65c9085cd048e8a7351a53e87666fdd5639e612b` | hidden-error trace / `R2-V0312` | `NOT_CREATED_PLAN_ONLY` |

## Ranked Round 2 tracker

Sort key is severity, then impact × inverse effort, then dependency readiness. Weights: Critical=4, High=3, Medium=2, Low=1; divisors S=1, M=2, L=3. Ties place cold-ready packets before decision-blocked packets without changing severity.

| Rank | Finding | Layer | Severity | Effort | Score | Status | Dependencies |
|---:|---|---|---|---|---:|---|---|
| 1 | [WT-001](findings/AEP-20260829-r2-WT-001.md) child still alive when service restoration starts | boundary | Critical | M | 2.00 | blocked | operator-defined `CHILD_UNREAPED` recovery and exact replacement target |
| 2 | [REPO-003](findings/AEP-20260829-r2-REPO-003.md) no-op cycle marks untouched item verified | boundary | Critical | M | 2.00 | blocked | receipt schema and analysis-only terminal-state decision |
| 3 | [REPO-001](findings/AEP-20260829-r2-REPO-001.md) checkpoint identity collision overwrites bytes | backend | High | S | 3.00 | ready-for-fix-agent | implement atomically with REPO-002 |
| 4 | [REPO-006](findings/AEP-20260829-r2-REPO-006.md) incompatible router geometry published enabled | backend | High | S | 3.00 | ready-for-fix-agent | add missing tracked root router regression module |
| 5 | [REPO-007](findings/AEP-20260829-r2-REPO-007.md) adaptive enablement commits before validation | backend | High | S | 3.00 | ready-for-fix-agent | none |
| 6 | [REPO-009](findings/AEP-20260829-r2-REPO-009.md) sibling success erases failure state | frontend | High | S | 3.00 | blocked | REPO-008 canonical-client decision |
| 7 | [REPO-002](findings/AEP-20260829-r2-REPO-002.md) checkpoint success survives catalog failure | backend | High | M | 1.50 | ready-for-fix-agent | implement atomically with REPO-001 |
| 8 | [REPO-005](findings/AEP-20260829-r2-REPO-005.md) malformed newest validation becomes older green | boundary | High | M | 1.50 | ready-for-fix-agent | coordinate response discriminant with frontend owner |
| 9 | [REPO-008](findings/AEP-20260829-r2-REPO-008.md) static-first surface hides three panels | frontend | High | M | 1.50 | blocked | operator selects canonical client and retained panels |
| 10 | [REPO-004](findings/AEP-20260829-r2-REPO-004.md) timed-out worker keeps mutation authority | backend | High | L | 1.00 | blocked | worker isolation/cancellation/external-effects contract |
| 11 | [PR257-001](findings/AEP-20260829-r2-PR257-001.md) secondary hook masks primary block failure | backend | Medium | S | 2.00 | blocked | operator assigns clean replacement branch/owner |
| 12 | [REPO-010](findings/AEP-20260829-r2-REPO-010.md) render errors written into hidden nodes | frontend | Medium | S | 2.00 | blocked | REPO-008 canonical-client decision |

Duplicate, excluded from ranks and counts: [PR295-001](findings/AEP-20260829-r2-PR295-001.md), an executable decomposition of existing `CD-R16-01` / Round 1 PR296 aggregate evidence.

## Prior-cycle active ledger

Round 1 remains authoritative for its packets and exact dependencies: [Round 1 master backlog](../20260829/04-master-backlog.md). Its active set is three Critical, ten High, seven Medium, and one Low; two are ready and nineteen are blocked. Round 2 does not silently rename, close, or re-score them. Critical/High execution order ahead of Round 2 lower-severity work is:

1. Round 1 PR257-001, WT020-002, and PR296-001 (Critical).
2. Round 2 WT-001 and REPO-003 (Critical).
3. Round 1 High packets in their existing rank order.
4. Round 2 High packets in the table above.
5. Medium/Low packets by score and dependency readiness within their controlling ledger.

## Aggregate open-plan counts after Round 2

| Scope | Critical | High | Medium | Low | Total | Ready | Blocked |
|---|---:|---:|---:|---:|---:|---:|---:|
| Round 2 net-new | 2 | 8 | 2 | 0 | 12 | 5 | 7 |
| Round 1 carried | 3 | 10 | 7 | 1 | 21 | 2 | 19 |
| cumulative active | 5 | 18 | 9 | 1 | 33 | 7 | 26 |

No row is `open` without disposition. A `blocked` row may become ready only after its named decision/dependency and exact target are recorded in this ledger; prose inside a packet does not self-unblock it.
