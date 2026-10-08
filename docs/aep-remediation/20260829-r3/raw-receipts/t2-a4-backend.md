# T2 A4 backend/reliability PR receipt

Snapshot date: 2026-08-29 EDT. All calls were read-only. No PR, branch, worktree, service, or product file was changed.

## Exact PR metadata and accumulated check rollups

| PR | Exact head / base | Draft / merge state | Files | Accumulated checks |
|---:|---|---|---:|---|
| 308 | `8ce7feb987490e8d39479673cba18604e376568d` / `3d94141e38114e1824f91e1b7a858352d2cbbddf` | draft / BLOCKED | 1 | 7 failure, 17 success across historical reruns |
| 297 | `53a86f0fbdaa1ebb220023e3787d9231b332eaa8` / `34ce4b5632e0d9cd2a16c29e0e1acc42e645b9c2` | draft / BLOCKED | 6 | 1 failure, 11 success |
| 296 | `65c9085cd048e8a7351a53e87666fdd5639e612b` / `34ce4b5632e0d9cd2a16c29e0e1acc42e645b9c2` | draft / DIRTY | 100 API rows / 213 exact diff paths | 1 failure, 1 neutral, 10 success |
| 295 | `730b305e8352b1c7f41e8b77062c4c7cba543dc6` / `12aa1be608f30a60d0e7dff56ff10acdff911960` | ready / CLEAN | 19 | 1 success |
| 294 | `6e78cf419c59ada7d4dd3ad2dace8f8721552451` / `98e9dec42f2f0d0beea16620a2fe8a39cd0e7c61` | ready / CLEAN | 5 | 1 success |
| 293 | `454e4a32453b29040eda9c286ed120a158a265b4` / `eb7509583e81b6a62d13b82587398562e4bba09a` | ready / CLEAN | 6 | 1 success |
| 292 | `eb7509583e81b6a62d13b82587398562e4bba09a` / `34ce4b5632e0d9cd2a16c29e0e1acc42e645b9c2` | ready / DIRTY | 38 | 1 failure, 1 neutral, 14 success |
| 278 | `9473e9f7d192299e6333608d91f3a659c63eda8d` / `34ce4b5632e0d9cd2a16c29e0e1acc42e645b9c2` | ready / BLOCKED | 8 | 2 failure, 21 success |
| 257 | `663be66e3292aae654a0f615e7c5e2b4dd7a98d4` / `3595283dc6ba113b91e19750b0b44fac149ed45e` | draft / DIRTY | 100 API rows / 351 exact diff paths | 1 cancelled, 1 failure, 10 success |
| 256 | `4379b9cdeaa54b1b53f043f034ffc68bb60494d4` / `3595283dc6ba113b91e19750b0b44fac149ed45e` | draft / BLOCKED | 100 API rows / 168 exact diff paths | 2 failure, 10 success |

`gh pr checks <n> --required` reported no required checks for every open head. `NO_REQUIRED_CHECKS` is an absent enforcement gate, not acceptance. PR #308's seven accumulated failures are repeated historical §4/block entries at the same head: the newest rerun has §4 PASS and one expected block-flag failure, while the test/lint matrix at the exact head remains green.

## Exact-head backend checks

```text
PR293 evidence DAG: Ran 60 tests in 0.039s; OK; Ruff passed
PR294 pool shard: Ran 15 tests in 0.123s; OK; Ruff passed
PR295 quant-aware routing: Ran 49 tests in 1.292s; OK; Ruff passed
```

These are bounded local checks. They do not replace PR discussion, required-check enforcement, or the failure probes below.

## Net-new failure probes

- PR #294 interrupted two-file rewrite: manifest publication was forced to fail after new payload publication. `read_pool_shard` then returned IDs `['A0','A1']` from the old manifest with vectors `[[101,102],[103,104]]` from the new payload. Packet: `PR294-001`.
- PR #297 identity validation: exact-head baseline reported coverage `100.0`, totals `88/77/4`. A disposable copy with duplicate `c01` and `w01` rows exited zero and reported `89/78/4`; the duplicate claim silently replaced the first mapping. Packet: `PR297-001`.

Exact executable commands and outputs are in the two finding packets. The five extracted PR #297 fixture files were individually deleted after the probe; the empty named temporary directories remain because a recursive cleanup command was rejected before execution and was not retried.

## Carry and exclusions

- PR #295 incompatible serving dimension remains a duplicate of `CD-R16-01` and Round 2 `PR295-001`.
- PR #296 BHS/public-readiness contradiction remains Round 1 `PR296-001`; its current root dirty overlay is outside the exact public head.
- PR #257 and #256 retained current backend threads and prior packets; failed lint/cancelled runtime evidence is not converted into a new root cause without a distinct contract.
- #308, #292, and #278 are documentation/evidence surfaces; #293 source and local checks revealed no new demonstrated defect in this pass.
- No browser, live service, external system, deployment, or hardware/model execution was attempted.
