# Cold-command verification — Tier A iteration 4

Cycle: `AEP-20260829-1`. These are executed receipts for the commands challenged by Tier B iteration 3. Commands were run from `D:\GITHUB\CHELATEDAI` unless a worktree push is explicit.

| Verification ID | Executed command boundary | Exact result |
|---|---|---|
| `VER-AEP-20260829-PR296-001` | quoted `gh pr view` JSON field list for #292–#295 | exit 0; #292 `CONFLICTING/DIRTY`; #293–#295 `MERGEABLE/CLEAN`; all four body-contains-100 booleans `true` |
| `VER-AEP-20260829-WT020-003` | `Push-Location D:\GITHUB\CHELATEDAI-RHPC1`; RHPC Stage A module | exit 0; `Ran 15 tests in 27.630s`; `OK` |
| `VER-AEP-20260829-WT020-004` | same corrected RHPC worktree-scoped module command | exit 0; same 15-test receipt; test inventory has no hostile mutation/concurrent-publisher case |
| `VER-AEP-20260829-FE001` | real token-enabled `HTTPServer` handler probe | exit 0; page no-header 401, page bearer 200, API no-header 401; raw receipt `http-auth-probe-receipt.txt` |
| `VER-AEP-20260829-BE002-001` | deterministic pre-promotion deadline probe | exit 0; failure at named checkpoint; `target_is_dir=False`; raw receipt `be002-stage-probe-receipt.txt` |
| `VER-AEP-20260829-BE002-001` | focused two-test loop, 20 processes | 40 tests passed; zero failures; confirms current tests do not cover deterministic earlier expiry |
| `VER-AEP-20260829-WT019-001` | porcelain count plus exact known package `check-ignore` | exit 0 wrapper; `venv_visible_lines=3898`; `check_ignore_exit=1` |

All other findings are either source/PR/worktree inspections already preserved in their packets or explicitly `blocked`; none is advertised as cold-fix ready without an owned target/contract/harness. The only `ready-for-fix-agent` packets after normalization are BE002-001 and WT019-001, and both have executable current probes, atomic steps, AC1–AC3, and exact post-fix commands.

`PLATEAU_ESCALATION`: command executability survived as a theme across the first two failed reviews. The exact failure sets were not identical, but Tier A should still have stopped and escalated before claiming iteration 3 closure. Iteration 4 records that control failure, executes the challenged commands directly, downgrades decision-dependent packets to `blocked`, and requests one final fresh Tier B. No fifth Tier A loop is authorized if this exact failure set recurs.
