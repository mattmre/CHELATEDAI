# PV7 Final Judge #2, Round 2 Verification (Claude Fable 5.2, Medium)

- Date: 2026-11-25
- Repo: `D:\GITHUB\CHELATEDAI-PV7-SPINE`, branch `pv7/w16-core`
- Reviewed HEAD: `b0170cbbac45bd3e68ba176137ddf0494299ccbc`
- Round 1 memo: `docs/research/council-logs/pv7-final-judge-fable52-med.md` (94/100, blocked on B1 and B2)
- Scope: Round 1 §5 only. Nothing else was re-reviewed.

## Verdict

**`APPROVED_FOR_PV7_64_SEAL`**

- BHS_TIER_B (this judge, Round 2) = **100**
- BHS_TIER_B_SEVERITY: none open. B1 and B2 are closed. One non-blocking wording nit (N1) is recorded below.

The seal is a seal of an honest scorecard. Release status stays `PRE-RELEASE / HUMAN AUDIT & CLEAN-RERUN GATES PENDING`; this verdict does not change it.

## 1. What I checked against Round 1 §5

| §5 item | Result | Evidence |
|---|---|---|
| 1. B1, `#48` `evidence_disclosure.tuning` | Closed | Fragment now names commit `25ae2067`, `BASE_EF` 100 to 220, Qdrant HNSW `m` 16 to 64, `ef_construct` 128 to 512, `full_scan_threshold` 1 to 10, the OpenSearch exact-search lock, pgvector `max_scan_tuples` / `scan_mem_multiplier`, the `d4454e2d` stamp, and states tuning on the synthetic evaluation fixture is NOT excluded. |
| 2. B2, `#54` `evidence_disclosure.failed_endpoint` | Closed | New field states `H6B_merge.passed=false`, 6/12 positive vs 11/12 in code and >=10/12 in registry prose, `LCB95=-0.0090`, and that `h6b_merge_evaluated` means computed, not met. |
| 3. Recompile and fast offline verify | Closed | See §2. |

I did not take the B1 text on trust. `git show 25ae2067 -- chelated_pv7_spine/hnsw_prefilter_pushdown.py` shows every listed change: `BASE_EF = 100` to `220`; `hnsw_config` `{m 16, ef_construct 128, full_scan_threshold 1}` to `{64, 512, 10}`; OpenSearch index created with `ef_construction 512, m 64`; pgvector index `m = 64, ef_construction = 512`; `SET LOCAL hnsw.max_scan_tuples = 200000` and `hnsw.scan_mem_multiplier = 8` added. The same commit adds the live receipt. `d4454e2d` is an ancestor of `dad69261`, the commit that first added the module, so the stamped `spine_commit` does predate the module.

For B2, the committed manifest `artifacts/pv7_w18_6b_12domain_4hop/pv7_w18_6b_drycheck_manifest.json` reads `passed: false`, `positive_domains: 6`, `required_positive_domains: 11`, `lcb_95: -0.008954`, and `overall_verdict: local-diagnostic-not-supported`. The `>=10/12` prose is at line 15 of the `#54` fragment. The disclosure matches all of it.

Carried-debt polish, also confirmed:

- `#58` scope now says, in plain words, that no live 9.47M hardware or live Qdrant measurement exists. `58_verifier` output agrees: `live_measurement: false`, `evidence_tier: local-drycheck-floor`.
- `#57` governance note now reads "Linux aarch64 6GiB cgroup host [host name not recorded in receipt]". The receipt records only `platform: Linux-6.17.0-1029-nvidia-aarch64-with-glibc2.39`; I found no host name field in `artifacts/pv7_w19_2b_cartridges/report_receipt.json`.

## 2. Commands run (all with `CUDA_VISIBLE_DEVICES=""`)

| Command | Result |
|---|---|
| `python scripts/verifiers.d/48_verifier.py` | rc=0, live receipt `status: live-receipt-verified`, compiled sha `7d1c821a...` |
| `python scripts/verifiers.d/54_verifier.py` | rc=0, `passed: true` (verifier pass, which is not the H6B_merge endpoint), compiled sha `1dfe7d55...` |
| `python scripts/verifiers.d/58_verifier.py` | rc=0, `status: verified-mixed`, `live_measurement: false` |
| `python scripts/verify_all_offline.py --fast --skip-verifiers` | rc=0, `verify_all_offline: PASS mode=fast`, 12.4 s wall. One WARN, `fragments.verifiers: SKIPPED_BY_FLAG`, which is the flag I passed. `fragments.compile` reports registry=18, routes=17, verifiers=18, compiled sha `482a0bcb45d0`. |

Scorecard determinism, checked by importing the compiler without calling `main()` (so nothing was written):

- In-memory `build_scorecard()["scorecard_sha256"]` = `192855cdfd2d1400ef8d4adea28a607d941e5281cd59f01e5a81bd01454abffe`, equal to the on-disk value.
- `json.dumps(sc, indent=2, sort_keys=True) + "\n"` is byte-identical to `COMPLETE_64_BUNDLE_EVIDENCE_SCORECARD.json`.
- `render_markdown(sc)` is byte-identical to `WAVES_1_TO_21_MASTER_EVIDENCE_AND_PR_INDEX.md`.
- 64 bundles, 64 unique IDs, 46 base plus 18 fragment. 0 CR bytes in both files.
- The Markdown header cites `192855cdfd2d`, and the `#48`, `#54` and `#58` table rows carry the new disclosure text.
- `git status --short` was empty before I wrote this memo.

The Round 1 caveat about pinned hashes is settled by the above: the three touched fragments' verifiers pass and the fragment compile check passes at this HEAD.

## 3. Non-blocking nit

**N1 (`#48`, wording, my own Round 1 error).** The disclosure says "the receipt records none of these knobs". That is not quite right. The live receipt records a per-cell effective `hnsw_ef` (220, 245, 440, 1000, three cells each), and 220 is the raised `BASE_EF`. It does not record `m`, `ef_construct`, `full_scan_threshold`, `max_scan_tuples` or `scan_mem_multiplier`. The sentence was copied from my Round 1 §5, so the inaccuracy is mine. It understates what the receipt holds rather than overstating it, and the recorded 220 confirms the raised value was the one used live, so the conclusion "tuning on the evaluation fixture is not excluded" stands. I am not blocking on it. Suggested text for a later pass: "the receipt records the per-cell effective `hnsw_ef` but none of the index-build or scan knobs".

## 4. Reliance on Judge #1

I did not run `--full-formal`, Lean, `nanoda`, the 7,722-checkpoint conformance, or the other 15 verifiers in this round. The 110/110 and 18/18 figures are Judge #1's (`docs/research/council-logs/pv7-final-judge-carbon4-med.md`). I read that the memo exists in this commit; I did not reproduce its run.

## Brutal Honesty

- This round verified three of 18 verifiers (`48`, `54`, `58`). Round 1 covered ten; the remainder rest on Judge #1.
- No live engine, vLLM worker or cgroup run was re-executed. Every live number is a reading of committed JSON.
- B1 establishes that the knobs changed in the commit that sealed the receipt. It does not establish that they were tuned on the evaluation fixture, and neither the disclosure nor I assert that.
- `#54` code requires 11/12 and the registry prose says >=10/12. The disclosure names both; the mismatch itself is not reconciled. At 6/12 the endpoint fails under either.
- N1 is a known inaccurate clause that ships with this seal. I scored 100 because it under-claims and changes no tier, count or conclusion. A stricter reader could hold the score at 99 until it is reworded.
- I did not open any `*sealed_key*` or `*blind-followup*` file or `pm4_human_audit_100_blind_sheet.csv`. GPU use was zero.
- The only file I wrote is this memo. No commit was made.
