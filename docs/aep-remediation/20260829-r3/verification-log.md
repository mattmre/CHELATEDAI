# Round 3 verification log

| ID | Tier | Command / inspection | Result | Interpretation |
|---|---:|---|---|---|
| `R3-V0001` | T0 | A4 remote/branch/HEAD/upstream/backend-root/BHS inventory | complete at `2026-08-30T01:51:39.898Z` | exact local baseline; no product diff review |
| `R3-V0002` | T0 | A1 full porcelain and `git worktree list --porcelain` | 150 paths; 25 worktrees; 8 dirty | raw command/count/hash and worktree stream preserved |
| `R3-V0003` | T0 | A1 branch-to-open-PR metadata join | 5 matching heads; 18 feature branches without open PR; 1 main; 1 detached | metadata only; no branch content inspected |
| `R3-V0004` | T0 | A2 quoted `gh pr list` metadata | 10 open PRs | raw JSON preserved; no body/diff/comment/log claim |
| `R3-V0005` | T0 | A3 tracked FE/blob/route/harness inventory | 4 clean roots; 19 APIs; no tracked served-page harness | route/root inventory only |
| `R3-V0101` | T1 | A1 all stable legacy/RHPC dirty worktrees | 2 net-new High; 6 prior duplicates | exact matched-control output and source boundary preserved |
| `R3-V0102` | T1 | A2 AVO/integration dirty worktrees | 2 moving targets blocked; stable residue excluded | full timestamps, sizes, hashes, and patch IDs preserved |
| `R3-V0103` | T1 | A3 FE delta classification across 8 dirty worktrees | no new FE finding | no browser/server/network gate claimed |
| `R3-V0104` | T1 | A4 QSCCI deterministic dual-failure probe | primary observation error absent; 2 focused tests pass | net-new Medium boundary finding |
| `R3-V0105` | T1 | A4 paired-intervention module | 25 tests passed in 17.478s | bounded counter-evidence only |
| `R3-V0201` | T2 | A1 all issue comments/reviews/paginated threads | 27 UI-unresolved; current/outdated/fixed normalized | no unresolved item omitted or auto-accepted |
| `R3-V0202` | T2 | A2 all bodies/exact diffs/checks/failed logs | 10 PRs; exact 1–351 path sets | no required checks on any head/base |
| `R3-V0203` | T2 | A3 served-FE base/head classification | 4/4 known FE blobs equal for every PR | no new FE finding; no browser claim |
| `R3-V0204` | T2 | A4 PR293/294/295 focused suites + Ruff | 60/15/49 tests pass; Ruff passes | bounded counter-evidence |
| `R3-V0205` | T2 | A4 PR294 interrupted rewrite probe | new vectors returned with old IDs | net-new High boundary finding |
| `R3-V0206` | T2 | A4 PR297 duplicate-ID fixture | exits 0; totals inflate 88/77 to 89/78 | net-new Medium backend finding |

`SKIPPED_GATES` is intentionally not finalized before Tiers 2–3. Later results are recorded as new rows, never backfilled into the frozen Tier 0 rows.
