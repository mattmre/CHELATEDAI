# PV7 Final Judge #2: Post-v7.0 Complete 64/64 PRE-RELEASE seal (Waves 16-21, Bundles #47-#64)

Judge: Claude Fable 5.2 (independent Final Judge, panel-of-experts + Brutal-Honesty Rulebook v3.3, effort medium)
Tree: `D:\GITHUB\CHELATEDAI-PV7-SPINE`, branch `pv7/w16-core`, HEAD `a252ae97de789406fb9283d951f64065f22e3ef4`
Date: 2026-11-25

Scope rules honoured: the parent checkout `D:\GITHUB\CHELATEDAI` was not modified. Every Python call ran with `CUDA_VISIBLE_DEVICES=""`. No GPU job ran. No `*sealed_key*` or `*blind-followup*` path was opened and `pm4_human_audit_100_blind_sheet.csv` was not touched. No existing repository file was edited and nothing was committed. This memo is the only file written. Judge #1's memo (`pv7-final-judge-carbon4-med.md`) appeared in the tree during this review and was deliberately not read.

## Verdict

**BLOCK** (narrow: two disclosure-text defects on scorecard rows #48 and #54, no rerun needed).

Everything the brief asked me to verify holds on the committed tree, and all three CPU gates pass. The scorecard does not inflate any tier, does not vacate any bundle ID, and keeps PRE-RELEASE. I am blocking on two items from the grok-4.7 peer memo that Round-2 neither repaired nor disclosed, because both leave a human-facing scorecard row reading stronger than its evidence:

1. **#48 (`live-pass`) understates tuning.** The registry row says "engine index parameters were not swept per engine". The commit that added the live receipt (`25ae2067`) also raised the index and search knobs. That change is not disclosed anywhere in the row, the scorecard or the receipt.
2. **#54 (`local-drycheck-pass`) hides a failed preregistered endpoint from the Master Index.** `H6B_merge.passed` is false (6/12 positive domains, 11 required). The Markdown index never mentions it.

Both fixes are text in `registry.d` plus a scorecard recompile. Section 5 gives the exact wording to add. With those two applied and the scorecard recompiled, my verdict becomes `APPROVED_FOR_PV7_64_SEAL` without a further review of the other items.

## 1. Gates executed

All on Windows, Python 3.11.9, `CUDA_VISIBLE_DEVICES=""`.

| Gate | Result |
|---|---|
| `python scripts/verify_non_nda_ip_clean.py` | rc 0. `ZERO_IP_LEAK_AUDIT: PASS (0/56 forbidden method patterns matched across 2 file(s))` |
| `python scripts/build_whitepaper_assets.py --check --allow-gitignored-raw-missing` | rc 0. `PASS (bundles=46/46, side_hashes=61, unhashed_hist_raw=20, pristine=15/15, registry_sha=6ef7eba4f414)` |
| `python scripts/verify_all_offline.py --fast --skip-verifiers` | rc 0. `PASS mode=fast`, 13.7 s. `fragments.compile: registry=18 routes=17 verifiers=18 compiled_sha256=f1e0806b9ace`. Verifier replays recorded as `SKIPPED_BY_FLAG`; formal tier `SKIPPED_USE_FULL_FORMAL`. |
| `python -m unittest tests/test_pv7_w21_loaders.py tests/test_pv7_w21_live_receipts.py` | 34 tests, OK. Same result with `-W error::DeprecationWarning`. |
| `pytest tests/test_pv7_w16* ... tests/test_pv7_w21* -q -W always` | 288 passed in 342 s. Zero `DeprecationWarning` lines in the output. |
| Verifiers 47, 48, 49, 53, 55, 57, 58, 62, 63, 64 run directly | All rc 0, all with non-empty JSON output, zero `DeprecationWarning`. |

Not run by me: verifiers 50, 51, 52, 54, 56, 59, 60, 61, the full 18-verifier pass, and `--full-formal`. Those are Judge #1's. The Master Index statement "`18/18` offline verifiers `rc=0`" is confirmed by me for 10 of 18 only.

## 2. Items verified

### #58 Cold-OSD

- `chelated_cold_osd_scanner.py` has one shared record codec used by both arms. `drycheck()` builds the challenger with `journal_codec="shared"` and gates on the `shared_symmetric_codec` row plus the `no_durability_floor` row. The pre-remediation codec survives only as the supplementary `as_built_asymmetric` row.
- `58_verifier.py` output matches the registry row: clustered saves 19,899 B (shared), 18,855 B (floor), 22,503 B (as built). Scattered is 0 B under the shared codec, -720 B against the floor, and +1,200 B as built. `scattered_gated=false`, `live_measurement=false`, sector unit `512_byte_device_sectors`.
- Registry `evidence_tier` is `verified-mixed`. The row discloses the 512 B sector model and that a zero-variance LCB is not a confidence interval.
- The sonnet-5.5 memo's fix list for #58 (symmetric accounting, manifest disclosure, LCB relabel) is fully applied.

Two things a reader should know, neither blocking:

- The gate now covers the clustered layout only. That is a change made after the scattered layout was shown to tie or lose. It is disclosed ("recorded, not gated") and the outcome is labelled mixed rather than a win, so I accept it.
- The verifier reports `evidence_tier: local-drycheck-floor` while the registry says `verified-mixed`. Both are true (mixed outcome, drycheck floor), but the registry row does not say in plain words that no live measurement exists. Add that sentence when the row is next touched.

### Declarative fragment loaders

- `scripts/verify_all_offline.py` merges the 46 spec bundles with the `registry.d` fragments and checks each fragment statically (18 PASS lines, #47 to #64).
- `demo_workbench/server.py` compiles `routes.d` under the pinned spine digest and fails closed.
- `tests/test_pv7_w21_loaders.py` passes.
- Correction to the brief: there is **one** `routes.d` fragment file (`61-fabric.json`) declaring **17 routes**, not 17 fragment files. The loader's own output (`routes=17`) counts routes. No repository document I read claims 17 files, so this is a wording issue in the brief only.

The sonnet-5.5 finding that the loaders ignored the fragments is closed.

### Live-receipt and CLI verifiers

| Verifier | What I observed |
|---|---|
| 48 | Reads the committed live receipt offline: file sha `714556c2...`, fixture sha `5eb91e0b...`, three engine versions, `live-receipt-verified`. Top-level status stays `local-drycheck-pass`. |
| 57 | Verifies the committed cartridge receipt (hashes, gate set, zero-event Clopper-Pearson bound). Row discloses synthetic payloads. |
| 61 | Not run by me (about 10 minutes). Row discloses software emulation, both hardware flags false, 3 of 5 encoders live and 2 dry stand-ins. |
| 62 | Stage-1 replay gates pass, and the output carries `live_worker_outcome: "FAILED"` with `status=replay-fail`. I read the committed manifest directly: `mode=run-live-workers`, `overall_pass=false`, `live_model_calls=7842`, 70 cells, 1,680 jobs, and every `on_time_jobs` field is 0. The scorecard key is `stage1-replay-pass-live-worker-FAILED`. |
| 49 | Real `__main__`. Prints `local-drycheck-pass`, `committed_gate_report_mode=local-drycheck`. |
| 55 | Real `__main__`. Prints `protocol-ready`, `wild_evidence=false`, `llm_write_tokens=0`. |
| 63, 64 | Real `__main__`. Both print `protocol-ready` on the open DEV fixture. |

The grok-4.7 findings on empty CLI exits (#49, #55, #63) and on #62's green verifier beside a failed live run are closed.

### `allow_local_drycheck_smoke` and #60

- Every drycheck call site named in the two peer memos now passes the flag explicitly (loft, 12-domain, BEAM, PM-6 shadow, QSCCI, wild revisions, CPU memory appliance).
- 288 PV7 tests emitted zero `DeprecationWarning`.
- `SSDKVCache.reload_persisted` refuses a persisted block with no `contract_digest`, a blank or non-string digest, a config-only identity, a wrong literal `model_weights_sha256`, and patched base or adapter weights. The negative tests are in `tests/test_pv7_w19_2b_2d_2e_appliances.py` (`PersistedKVReloadTests`) and pass.

### 64/64 scorecard

- I rebuilt the scorecard in memory with `build_scorecard()` and `render_markdown()` without writing. Both are byte-identical to the committed `COMPLETE_64_BUNDLE_EVIDENCE_SCORECARD.json` (sha `6a2a254b49e6`) and `WAVES_1_TO_21_MASTER_EVIDENCE_AND_PR_INDEX.md`. The Markdown has 0 CR bytes.
- 64 rows, 64 unique IDs, `vacated_or_pruned_bundle_ids = 0`.
- `release_status` is `PRE-RELEASE / HUMAN AUDIT & CLEAN-RERUN GATES PENDING`. The Master Index names PM-4b, PM-1b and PM-6 as open and makes no ship-ready claim.
- Post-v7.0 tier counts: 3 `live-pass` (#48, #57, #61), 1 `live-shadow-pass` (#53), 1 `verified-mixed` (#58), 1 `stage1-replay-pass-live-worker-FAILED` (#62), 10 `local-drycheck-pass`, 2 `protocol-ready` (#63, #64). No `protocol-ready`, drycheck or rc 0 cell is counted as a live confirmatory completion.
- #53 is scoped as observational with `live_pm6_gateway_executed=false` stated in the row.

## 3. Blocking findings

### B1. #48 tuning disclosure omits a parameter lift made in the seal commit (L4, misleading by omission)

- At `44f112f2`, `chelated_pv7_spine/hnsw_prefilter_pushdown.py` had `BASE_EF = 100` and Qdrant `m=16, ef_construct=128, full_scan_threshold=1`.
- At HEAD it has `BASE_EF = 220`, `m=64`, `ef_construct=512`, `full_scan_threshold=10`, plus the OpenSearch exact-search lock and the pgvector `max_scan_tuples` and `scan_mem_multiplier` settings.
- The live receipt was first committed in `25ae2067`, the same commit that raised those values.
- The receipt records none of those knobs and no source hash. Its `spine_commit` is `d4454e2d`, and that commit does not contain `hnsw_prefilter_pushdown.py` at all.
- The registry row's `tuning` field says only that parameters "were not swept per engine".

I do not know whether the knobs were raised in response to a failing earlier live attempt. No committed artifact says either way. That is exactly why it has to be disclosed: #48 is one of only three live-pass cells, and a reader cannot rule out tuning on the evaluation fixture. The tier label `live-pass` can stay.

### B2. #54 `local-drycheck-pass` with a failed primary merge endpoint that the Master Index never shows (L4/L13)

- Committed manifest: `H6B_merge.passed=false`, 6 positive domains against 11 required, LCB95 -0.0090.
- The gate `h6b_merge_evaluated` passes because it only checks that the statistic was computed.
- The scorecard JSON governance note carries "[H6B_merge 6/12 false]". The registry row's `evidence_disclosure` does not, so the Markdown table (which renders only that field) has zero mentions of H6B or 6/12.
- The compiler's docstring promises that failures are surfaced. #62 gets that treatment. #54 does not.

## 4. Non-blocking residue from the peer memos (carry forward)

These are open against the grok-4.7 repair list. None changes a scorecard tier, so I am not blocking on them, but each needs a Carried Debt entry.

| Item | State at HEAD |
|---|---|
| #48 telemetry predicate | `telemetry_passed = route_passed and graph_telemetry_complete is not False` still lets `None` pass for OpenSearch and pgvector. |
| #61 gate naming | `live_encoders_fail_closed_on_dry_fallback` still passes with two dry stand-ins allowed. The scale gate still does not require `linear_chain`. The row does disclose the stand-ins. |
| #47 status | Registry `status` is still `verified-pass` beside `evidence_tier=local-drycheck-pass`. The scorecard shows both columns, so the stronger word is visible but not counted. |
| #50 one-shot flag | The `one_shot_consumed` logic was changed; I did not verify the regenerated manifest value. |
| Host attribution | The scorecard governance note says #57 ran in a "Spark #3 6GiB cgroup". No committed #57 or #62 receipt contains the string "spark"; the receipts record only an aarch64 NVIDIA Linux platform. The Spark attribution comes from commit messages. |
| #58 wording | Add "no live measurement" to the registry row (section 2). |

## 5. Required to lift the block

1. `docs/research/registry.d/48-t6a-engine-closure.json`, `evidence_disclosure.tuning`: state that the commit which added the live receipt also raised `BASE_EF` 100 to 220, HNSW `m` 16 to 64 and `ef_construct` 128 to 512, and set the OpenSearch exact-search lock and pgvector scan knobs; that the receipt records none of these and stamps a `spine_commit` (`d4454e2d`) which predates the module; and that tuning against the evaluation fixture is therefore not excluded.
2. `docs/research/registry.d/54-t6b-12domain-4hop.json`, `evidence_disclosure`: add a field stating `H6B_merge` endpoint NOT met on the drycheck fixture (6/12 positive, 11/12 required, LCB95 -0.0090), and that the passing gate `h6b_merge_evaluated` means computed, not met.
3. Rerun `python scripts/compile_complete_64_scorecard.py`, then `python scripts/verify_all_offline.py --fast --skip-verifiers`, and confirm the new scorecard sha in the Master Index.

If either registry edit changes a hash pinned by a verifier or the fragment compiler, that verifier must be rerun. I did not check which hashes pin those two files.

## 6. Scoring

- BHS_TIER_B (this judge) = 94.
- BHS_TIER_B_SEVERITY: two L4-class disclosure omissions (B1, B2), each on a single scorecard row, both text-only to fix. No fabricated evidence, no inflated tier, no vacated ID found.
- Rulebook v3.3 allows only 100 to seal. 94 does not seal.

## Brutal Honesty

- I verified 10 of 18 verifiers. The 18/18 claim rests on Judge #1 for the other eight, including #61.
- I did not re-execute any live engine, vLLM worker or cgroup run. Every live number above is a reading of committed JSON.
- I did not run `--full-formal` or any Lean check.
- I did not verify the #46 PM-6 figures (6.25% / 11.76%) quoted in the Master Index; they belong to the already-sealed v7.0 base.
- I did not check whether the base `verified_registry.json` classifications for #1 to #46 in `classify_base_bundle` match each bundle's own receipts. I checked only that the count is 46, IDs are unique, and the compiled output is deterministic.
- B1 establishes that the knobs changed in the seal commit. It does not establish why. "Tuned on eval" is not proven and I am not asserting it.
