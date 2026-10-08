# Round 2 worktrees and uncommitted state

Tier 1 opened all six dirty worktrees before any PR body/comment/diff review.

| Worktree / exact HEAD | Dirty state | Round 2 disposition |
|---|---|---|
| root `65c9085cd048e8a7351a53e87666fdd5639e612b` | 7 tracked + 115 untracked; 41 are prior audit docs | **net-new Critical** `AEP-20260829-r2-WT-001`; prior WT001-001 remains duplicate |
| `h2-rerun@6c3e1847830ec231157a421735a5a34d7d71a658` | 24 tracked deletions, including 22 result JSONs and 2 manifests | no new root cause; duplicate of `AEP-20260829-WT003-001` |
| `EGV-AVO-NEMOTRON@8ce7feb987490e8d39479673cba18604e376568d` | one redacted scanner JSON | `NO_FINDINGS`; no product wiring |
| `EGV-INTEGRATION@841b314b3f1e1524fab035a2bf2cb14cd0865c3b` | two byte-identical 28-byte binary fixtures | `NO_FINDINGS`; no consumers located |
| `EGV-TRAINING@97a8237e3f1b92425b6802895c1fcbdeafeb2681` | 3,898 visible `.venv-egv-prod` paths | no new root cause; duplicate of `AEP-20260829-WT019-001` |
| `RHPC1@305986427850dcb393a6078900201a7aa84c5b33` | 4 tracked + 9 untracked implementation/docs/test paths | Critical WT020-002 and High WT020-003/004 reverified; all remain duplicate carry-forwards |

## Reverified Critical/High carry-forwards

- `AEP-20260829-WT020-002` still has no operator-owned expected reviewed source/protocol identity. The official test accepts fabricated `source_git_sha="2"*40`; 15 RHPC tests pass and therefore do not close the trust defect.
- `AEP-20260829-WT020-003` still hashes predecessor/key bytes and independently rereads them for verification.
- `AEP-20260829-WT020-004` still checks target absence and later calls overwrite-capable `replace` instead of an atomic no-replace primitive.

These are carried by pointer and are not relabeled as Round 2 wins. The RHPC implementation remains an untracked overlay, so HEAD alone is not its immutable target.

## Frontend worktree result

`NO_NEW_FE_FINDINGS`: none of the six dirty worktrees modifies `dashboard_server.py`, `dashboard/index.html`, `test_dashboard_server.py`, or another browser component/auth/client-state route. Prior FE findings are deferred to live full-repo re-verification at Tier 3.

## Tier 1 command correction

The initial targeted command named nonexistent `test_qscci_supervisor.TestSupervisor` and failed with `AttributeError`. The corrected command below passed one test. This is an operator invocation error, not a repository finding.

```powershell
python -m unittest test_qscci_supervisor.TestQSCCISupervisor.test_timeout_after_kill_still_writes_canonical_private_evidence -v
```
