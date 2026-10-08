# Round 2 open PR index

Tier 2 opened all 10 PR bodies, exact heads/bases, file lists/diffs, reviews, issue comments, GraphQL review threads, check rollups, and failed/cancelled logs. Every `gh pr checks --required` invocation reports no required checks; that is not acceptance.

| PR / exact head | Checks / unresolved threads | Adversarial disposition |
|---|---|---|
| #308 `8ce7feb987490e8d39479673cba18604e376568d` | 4 failure / 12 success; 0 threads | empty body; §4 and block flag fail; docs-only draft, no FE/BE finding |
| #297 `53a86f0fbdaa1ebb220023e3787d9231b332eaa8` | 1 failure / 11 success; 0 | explicitly not ready; research automation/docs only |
| #296 `65c9085cd048e8a7351a53e87666fdd5639e612b` | failure + neutral + 10 success; 0 | conflicting; duplicate carry `AEP-20260829-PR296-001`; local QSCCI bytes absent from public head |
| #295 `730b305e8352b1c7f41e8b77062c4c7cba543dc6` | GitGuardian only; 2 | dimension-mismatch activation reproduced, but duplicate of durable `CD-R16-01`; later 60/Critical rejection controls |
| #294 `6e78cf419c59ada7d4dd3ad2dace8f8721552451` | GitGuardian only; 4 | no net-new root cause; prior replacement/fresh-review requirement survives |
| #293 `454e4a32453b29040eda9c286ed120a158a265b4` | GitGuardian only; 2 | no net-new root cause; prior rejection aggregate survives |
| #292 `eb7509583e81b6a62d13b82587398562e4bba09a` | failure + neutral + 14 success; 0 | stale/conflicting public head; duplicate carry `AEP-20260829-PR296-001` |
| #278 `9473e9f7d192299e6333608d91f3a659c63eda8d` | 2 failure / 21 success; 7 | docs-only; stale dossiers reserved for Round 4 contradiction pass |
| #257 `663be66e3292aae654a0f615e7c5e2b4dd7a98d4` | cancelled + failure + 10 success; 4 | prior C/H carry plus net-new Medium `AEP-20260829-r2-PR257-001`; parked/conflicting |
| #256 `4379b9cdeaa54b1b53f043f034ffc68bb60494d4` | 2 failure / 10 success; 8 | prior PR256 C/H carry; owner says parked/do not merge |

No open PR touches `dashboard_server.py`, `dashboard/index.html`, or a served FE/client path. Round 2 Tier 2 therefore has `NO_NEW_FE_FINDINGS`.
