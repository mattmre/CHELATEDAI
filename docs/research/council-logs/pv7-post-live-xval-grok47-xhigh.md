# Post-v7.0 live-receipt cross-check, Waves 16–21 bundles #47–#55 and #61–#64

Reviewer: grok-4.7 adversarial post-remediation pass (Brutal Honesty Rulebook v3.3).
Tree: `D:\GITHUB\CHELATEDAI-PV7-SPINE`, branch `pv7/w16-core`, HEAD `25ae2067b7c018c08725e16cd1d0156507b29a9f`.
Range inspected: `d68f2d40..25ae2067` (seal commits `fb20a797`, `5e176b3a`, `44f112f2`, `25ae2067`).
Spine digest frozen on every in-scope registry row: `32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9`.
Date of this reading: 2026-11-25. GPU was not used. `CUDA_VISIBLE_DEVICES=""` was set for every local Python call. No `*sealed_key*` or `*blind-followup*` path was opened. No code, test, receipt, or registry file was edited. The parent checkout `D:\GITHUB\CHELATEDAI` was not modified.

## Verdict

There is no 64/64 Complete Evidence Scorecard in this tree. Council logs only name a future Complete Evidence Ledger. `docs/research/registry.d/*.json` has no `evidence_tier` field. The `tier` key is a design tier (T0–T4). `status` is `protocol-ready` on every in-scope row except #47, which is `verified-pass`.

A scorecard that marks a cell complete because the registry says `protocol-ready`, or because `python scripts/verifiers.d/N_verifier.py` exited 0, would mark 13/13 of this slice complete. That reading is false (L4, partial-as-complete). The verifiers replay a local drycheck or a stage-1 fixture. They do not load the committed live JSON. Three of them (#49, #55, #63) have no `__main__`, so the CLI exit 0 is a no-op.

What is actually in the repo, at the ceiling this review could check without re-running engines or vLLM:

- #48 is a real 12/12 contact with Qdrant 1.20.2, OpenSearch 3.9.0 FAISS, and pgvector 0.8.7 on PostgreSQL 17, on a synthetic 128-d Gaussian route probe. The receipt does not name a Spark host, does not record the index-build knobs that were raised in the same seal commit, and stamps a `spine_commit` that does not contain the HNSW module.
- #61 is a real `run-live-encoders` CPU report with three sentence-transformer checkpoints and two dry stand-ins. Hardware flags are false. The scale gate passes while `linear_chain` is false.
- #62 is a real local vLLM worker run (`live_model_calls=7842`) and it failed. `overall_pass=false`, `status=replay-fail`. The verifier ignores that file and returned `stage1-replay-pass`.
- #47 has two host A/A calibration JSON files (`desktop-rtx3090`, `spark3-gb10`) that the verifier does not open. The verifier recomputes a different synthetic receipt and returns `local-drycheck-pass`.
- #49–#55 and #63–#64, as committed, are local-drycheck, schema drycheck, an observational replay of a committed fixture, or an open DEV fixture. Family names, domain names, `run-live-shadow`, `token_length_est` near 131072, and `preregistered_conjunction_met` are labels on those fixtures.

Bundles #56–#60 were not audited. This memo does not score them.

## Verifier CLI, this review

Command shape: `$env:CUDA_VISIBLE_DEVICES=""` then `python scripts/verifiers.d/<id>_verifier.py` for ids 47–55 and 61–64. Logs: `C:\Users\mattm\AppData\Local\Temp\pv7-verifier-run\`. Wall time of the batch was 839 s. Exit codes:

| Id | Exit | ms | CLI status | What the process actually did |
| --- | ---: | ---: | --- | --- |
| 47 | 0 | 3604 | `local-drycheck-pass` | `run_deterministic_drycheck()`. Host id `pv7-w16-local-drycheck-synthetic-identical-arms`. Receipt sha `b76ea52e62ab69969ca2ebcc5cfebb3ec854002b3faadb43e47ce43cffb3ce5b`. Does not open the RTX 3090 or Spark3 JSON. |
| 48 | 0 | 29577 | `local-drycheck-pass` | `execute("local-drycheck")` only. Fixture sha `7d571304e7e2f71e5430ebb91fab345f3deee4fbd505b951ece2e56c1cd0acd3`. Canonical trace sha matched the live EK5 hash `ef93286bd6f1424e7eadf1843bc97948e0d614d64da140c9c404bb45c18c05ef` because that audit is deterministic and independent of fixture size. |
| 49 | 0 | 129 | empty log | No `__main__`. `verify_bundle()` was not called. The CLI did not read `gate_report.json`. |
| 50 | 0 | 21884 | `local-drycheck-pass` | `execute("local-drycheck")`. |
| 51 | 0 | 3416 | `local-drycheck-pass` | `execute("local-drycheck")`. |
| 52 | 0 | 53225 | `local-drycheck-pass` | `execute("local-drycheck")`. |
| 53 | 0 | 28621 | `local-drycheck-pass` | `execute("local-drycheck")`. Does not load the `run-live-shadow` manifest. |
| 54 | 0 | 29823 | `local-drycheck-pass` | `execute("local-drycheck")`. |
| 55 | 0 | 236 | empty log | No `__main__`. Direct call this review of `verifier_status()` returned `status=protocol-ready`, `wild_evidence=false`, `llm_write_tokens=0`, `authored_prompts=0`. |
| 61 | 0 | 640367 | `local-drycheck-pass` | `execute("local-drycheck")`. Does not load `pv7_w20_d12_fabric_report.json`. |
| 62 | 0 | 22685 | `stage1-replay-pass` | `stage1-controlled-replay` at pool size 4. `sham_reopen_control_valid` passed, `control_valid=true`, sham on-time jobs 192, sham mean latency 271.007 ms, withdrawal mean latency 276.109 ms. Does not load the live manifest. |
| 63 | 0 | 121 | empty log | No `__main__`. Direct call this review of `verify_bundle()` returned `passed=true`, `status=protocol-ready`, `suite_status=protocol-ready`. That is the open DEV fixture, not a partner field run. |
| 64 | 0 | 263 | `protocol-ready` | Offline DEV fixtures. `suite_sha256=3a96a98a7776708c5f6ffaee5eb509864c77ea46d86c09febc9920a7eb00c10e`. Docstring says the scorer never self-promotes a DEV fixture past `protocol-ready`. |

## Scorecard the registry can honestly carry

| Bundle | Design `tier` | Registry `status` | Evidence this tree actually holds | Honest cell |
| --- | --- | --- | --- | --- |
| 47 | T0 | `verified-pass` | Verifier drycheck above, plus two unread host A/A JSON files that self-report `passed` | Spine drycheck. Host calibrations are separate receipts, not verifier output. |
| 48 | T2 (+ T0) | `protocol-ready` | Live 12/12 on synthetic Gaussian `pv7-b48-synthetic-gaussian-v1`. Verifier ignores it. | Engine-route probe. Not a wild corpus. Not a named Spark result. |
| 49 | T1 | `protocol-ready` | `gate_report.json`: `mode=local-drycheck`, `execution=deterministic-local-smoke`, `live_model_calls=0`, 36/36 sessions. | Local smoke. Family strings are not live models. |
| 50 | T1 | `protocol-ready` | Drycheck manifest `local-drycheck-pass`, `is_live_confirmatory=false`, `one_shot_consumed=true` on the power-sized fixture. | Synthetic NI. One-shot flag already consumed. |
| 51 | T3 (+ T0) | `protocol-ready` | Drycheck only. 20 clusters × 400 questions. `mean_token_length_est=129007.2`. | Schema replay. The 128k field is a stamp. |
| 52 | T3 | `protocol-ready` | Drycheck. `live_weights_loaded=false`, `qwen_report_split_executed=false`, `claim_level=local-drycheck`. | Planted-hook identity. Not a Qwen ceiling. |
| 53 | T3 | `protocol-ready` | Drycheck on 320 scratch families, plus `run-live-shadow` with `live_pm6_gateway_executed=false`. | Observational replay of a committed input. Gateway not executed. |
| 54 | T1 | `protocol-ready` | Drycheck. `dataset_origin=synthetic_schema_drycheck_fixture`. All five `hf_id` values are `""`. `H6B_merge.passed=false` while gate `h6b_merge_evaluated` passed. | Synthetic factorial. Merge endpoint missed. |
| 55 | (no `tier` key) | `protocol-ready` | `status=DRYCHECK`, 4 synthetic families, `superiority_is_wild_evidence=false`, `wild_inventory_verified=false`. | Schema drycheck. Conjunction on 4 toy families. |
| 61 | T2 | `protocol-ready` | One `run-live-encoders` report, software emulation, two stand-ins, hardware flags false. Verifier is drycheck-only. | CPU encoder emulation. Not DPU, ASIC, P4, or flight hardware. |
| 62 | T2 | `protocol-ready` | Live vLLM receipt `overall_pass=false`. Verifier `stage1-replay-pass`. | Live run failed its primary gates. Do not score Complete. |
| 63 | T4 | `protocol-ready` | `fixture_split=DEV_OPEN`, suite sha `051a00eb0eaf037a74f7a89d281615f78402bedc1221007a69cb691bb4dc5117`. | Open DEV scorer. |
| 64 | T4 | `protocol-ready` | `data_origin=open_dev_fixture`, `fixture_split=DEV_OPEN`, protocol digest `559a785594464fe8f76ef70c29804a4f688f84306cb9ba3ac504267a0e2ee3ec`. | Open DEV scorer. |

## Bundle #48 — live engines, synthetic vectors, undisclosed knobs

Receipt: `artifacts/pv7_w17_6a_engine_closure_live/48-t6a-engine-closure.json`.
`mode=run-live`, `passed=true`, `fixture.id=pv7-b48-synthetic-gaussian-v1`, `fixture.synthetic=true`, `fixture.dimension=128`, `fixture.population=12000`, `fixture.queries=10000`, `fixture.seed=480017`, `fixture.origin=synthetic_engine_route_probe`, top-level `is_wild_corpus=false`, fixture digest `5eb91e0bd3ce1f53d70de6e91d3c9bca671ec825f115ab10f5590328db94ef83`.
The strings `spark`, `m`, `ef_construction`, `full_scan`, `disable_exact`, `max_scan_tuples`, and `scan_mem` do not occur in the JSON.

`scripts/verifiers.d/48_verifier.py:25-33` always calls `execute("local-drycheck")` and raises if a supplied receipt has any other `mode`. It cannot validate this file. The drycheck fixture sha from this review (`7d571304…`) is a different digest from the live fixture (`5eb91e0b…`).

Recall is same-engine overlap. `scripts/run_pv7_w17_6a_engine_closure.py:178-233` times `backend.search(..., exact=False)` against `backend.search(..., exact=True)`. `max_recall_gap=0.01` and `paired_requests=10000` on every cell. The p95 bound in the registry live protocol is 35 ms. Every live cell is under that bound. The largest p95 is pgvector at 90% retraction, 9.713 ms. This is a 12_000-point probe, not a scale claim.

`adaptive_ef` is `min(1000, ceil(BASE_EF / max(admitted_frac, 0.02)))` at `chelated_pv7_spine/hnsw_prefilter_pushdown.py:39-42`. At 90% retraction, admitted points are 1200 and `hnsw_ef` is 1000, so that cell searches nearly the whole admitted set.

Per-cell `recall_at_10` / ann p95 ms / `hnsw_ef` / `admitted_points` / route on the first plan probe:

| Engine | 0% | 10% | 50% | 90% |
| --- | --- | --- | --- | --- |
| Qdrant 1.20.2 `filtered_hnsw` | 0.999250 / 3.272 / ef 220 / 12000 | 0.999370 / 3.541 / 245 / 10800 | 0.999860 / 3.708 / 440 / 6000 | 0.998420 / 3.532 / 1000 / 1200 |
| OpenSearch 3.9.0 FAISS `filtered_hnsw` | 0.999200 / 4.234 / 220 / 12000 | 0.999530 / 4.118 / 245 / 10800 | 0.999990 / 4.401 / 440 / 6000 | 1.000000 / 5.886 / 1000 / 1200 |
| pgvector 0.8.7, `postgres_version_num=170012`, `iterative_hnsw` | 0.999970 / 6.175 / 220 / 12000 | 0.999990 / 6.345 / 245 / 10800 | 1.000000 / 7.667 / 440 / 6000 | 1.000000 / 9.713 / 1000 / 1200 |

p95 values are `ann_latency.p95_ms` rounded to 3 decimals. All 12 cells have `passed=true`.

Telemetry is not uniform. Qdrant cells have `graph_telemetry_complete=true` and `graph_telemetry_basis=qdrant_route_and_fallback_counters`. OpenSearch and pgvector cells have `graph_telemetry_complete=null` and `graph_telemetry_basis=null`, and still `telemetry_passed=true`. The predicate is `route_passed and graph_telemetry_complete is not False` at `scripts/run_pv7_w17_6a_engine_closure.py:287`. `None` therefore passes. All 12 cells have `per_query_visit_counts_available=false` and `graph_candidate_sample_count=0`.

EK5 block on the same receipt: `canonical_trace_sha256=ef93286bd6f1424e7eadf1843bc97948e0d614d64da140c9c404bb45c18c05ef`, `sequences=1000`, `steps_per_sequence=12`, `semantic_diffs=0`, `reason_export_diffs=0`, `pos_cyc_export_diffs=0`, `parity_passed=true`. That parity is real and it is not a retrieval-corpus result.

### Parameter lift is in the code and missing from the receipt

HEAD `hnsw_prefilter_pushdown.py` has `BASE_EF = 220` (line 26), Qdrant `hnsw_config` `m=64`, `ef_construct=512`, `full_scan_threshold=10` (line 302), OpenSearch `ef_construction=512`, `m=64` (line 385), and before measurement the ingest forces `index.knn.advanced.filtered_exact_search_threshold=0` and `index.knn.faiss.efficient_filter.disable_exact_search=true` (lines 391–393). pgvector creates the index `WITH (m = 64, ef_construction = 512)` (line 483), sets `maintenance_work_mem=512MB` (line 480), and `SET LOCAL hnsw.max_scan_tuples = 200000` plus `hnsw.scan_mem_multiplier = 8` (lines 499–500).

`git grep` at parent `44f112f2` shows `BASE_EF = 100` on line 27 of the same file. `git log` of the file is `25ae2067` then `dad69261`. The live JSON already records `hnsw_ef` 220/245/440/1000, so it was produced with the raised constant. The seal commit both raised the constant and added the receipt.

`spine_commit` inside the receipt and inside `docs/research/registry.d/48-t6a-engine-closure.json` is `d4454e2d9fca146ccdc3d18db22ac6c0b5d3e802`. That commit is an ancestor of HEAD. `git cat-file` of `d4454e2d:chelated_pv7_spine/hnsw_prefilter_pushdown.py` fails: the path is not in that commit. There is no source-file sha256 on the #48 receipt. Checking out the stamped commit cannot reproduce the measurement. The registry `live_protocol` names the fixture and the engine version pins. It does not record `m`, `ef_construction`, `full_scan_threshold`, the OpenSearch exact-search lock, or the pgvector scan knobs. Status stayed `protocol-ready` after the seal.

The commit message of `25ae2067` says Spark #1. The receipt has no host string. This review does not upgrade the platform claim past the JSON.

## Bundle #49 — 36/36 local smoke, zero live model calls

`artifacts/pv7_w17_6d_sdk_coldstart/gate_report.json`: `mode=local-drycheck`, `execution=deterministic-local-smoke`, `gate_pass=true`, `live_model_calls=0`, `sessions=36`, `sessions_passed=36`, families `astra-6.2` / `fable-5.2` / `public-reference` at 12 each, `readme_canary=false`, `failure_rate_wilson_upper_95=0.09641862859446367`. That upper bound is the Wilson ceiling of a 0/36 failure sample. It is not a measured failure rate.

`wheel.coldstart_origin` is a Windows temp path under `C:\Users\mattm\.t3\userdata\antigravity-tmp\...\chelated_kernel\__init__.py`. `wheel-a` and `wheel-b` are the same file: `chelated_kernel-1.0.0-py3-none-any.whl`, 26853 bytes, sha256 `4384c6fba143837ab4a90f486a496ffbbb03cc907938b8134a958bfa06b3ff6e`, matching `wheel_sha256` in the report.

`scripts/verifiers.d/49_verifier.py` defines `verify_bundle()` and has no `__main__`. This review did not call `verify_bundle()`. The committed gate report is the evidence. Counting the 129 ms CLI exit as a cold-start proof is L4.

## Bundle #61 — live CPU encoders inside a software emulation

`artifacts/pv7_w20_d12_network_orbital_fabric/pv7_w20_d12_fabric_report.json`: `mode=run-live-encoders`, `passed=true`, `python=3.11.9`, `scoreboard_sha256=9ba7e9726ffc5e83a4c317f537b43420dba81bb2c39696610632ecc4f4fab8e4`.
`hardware_asic_p4_silicon_in_loop_claimed=false`. `orbital_flight_hardware_claimed=false`.
Banner: software emulation only; synthetic CESNET-TLS / CIC-IDS traffic; two-body Keplerian, not SGP4.
Encoders: `enc_a` MiniLM, `enc_b` bge-small, `enc_c` mpnet are `live:`. `enc_d` and `enc_e` are stand-ins (`dry_standin_fallbacks`). `allow_dry_standin=true`.
Disclosure text includes: disjointness of the synthetic corpus from historical fingerprint inventories was not established.

`scripts/run_pv7_w20_d12_network_orbital_fabric.py:174` sets `live_ok` true when `allow_dry_standin` is true even if dry fallbacks exist. The gate is still named `live_encoders_fail_closed_on_dry_fallback` (line 183). The receipt detail says the stand-ins were explicitly allowed. The name overclaims fail-closed behavior.

`scripts/run_pv7_w20_d12_network_orbital_fabric.py:165-167` passes `scale_factor_within_tolerance_on_bridge_paths` on `orthogonal_chain and fabric_chain` only. The receipt detail is `{"fabric_chain": true, "linear_chain": false, "orthogonal_chain": true}` and the gate passed. `demo_workbench/network_orbital_engine.py:1890` maps public family `ridge` to `linear_chain`. The passing gate does not require that family.

`61_verifier.py` calls `execute("local-drycheck")` and does not open this report. A green verifier is a different run from the live-encoder receipt.

## Bundle #62 — live vLLM failed; the verifier passed a different world

Live manifest `artifacts/pv7_w20_d13_pooled_agent_loops/pv7_w20_d13_manifest.json`:
`mode=run-live-workers`, `status=replay-fail`, `overall_pass=false`, `live_model_calls=7842`, `total_cells_executed=70`, `total_eligible_jobs_executed=1680`, model `nemotron-lightning` requested and served, `vllm_url=http://127.0.0.1:8801/v1`.
Failed gates: `beats_full_restart_rebuild_jpm` and `sham_reopen_control_valid`.
Full-restart comparator: both pooled JPMs 0.0, advantage 0.0, 48 late-or-timeout jobs on both sides, pool size 4.
Primary 2×2 at pool 4, event regime `both`: all cited LCBs 0.0, mean deltas 0.0, `strictly_positive_lcb=false`, `matched_blocks=2`, `matched_twin_jobs=48`, `n_clusters=24`. All four arms have `on_time_cited_correct_jobs=0`.
Sham control: `control_valid=false`, sham mean latency 7852.611 ms, no-change mean 3914.185 ms, withdrawal-only mean 5406.114 ms, `net_ek5_lineage_invalidation_delta_ms=-2446.497`, sham and withdrawal both reopen mean 1.0 subtask, sham on-time jobs 0.

`report_receipt.json`: `generated_utc=2026-11-25T13:27:25.677583+00:00`, which sits after `44f112f2` (07:32 -0500) and before `25ae2067` (09:47 -0500). Runtime platform `Linux-6.17.0-1029-nvidia-aarch64-with-glibc2.39`, Python 3.12.3, `cuda_visible_devices=""`. The manifest body does not contain `spark` or `enable_thinking`. The commit message says Spark #3. The aarch64 NVIDIA platform is consistent with a GB10-class host. The receipt does not name Spark #3.

Source hashes in the receipt match the current files: `chelated_pooled_agent_runtime.py` `5a6ceced76259adbcc0dd8b1d9e03ee9a98f1725b6bf02d8dc00b71abdd819c6` and `scripts/run_pv7_w20_d13_pooled_agent_loops.py` `9c2ebc3381e81f8c4ef1b0145733241ebd6fe28419e344a6ec4675a3bd67dbcf`. Line 1276 of the runtime sends `chat_template_kwargs: {"enable_thinking": false}` with `temperature=0.0` and `max_tokens=96`. The receipt does not echo the request JSON, so the flag is proven by source hash, not by a logged payload, and nothing in the receipt proves the server honored it.

`scripts/verifiers.d/62_verifier.py:76-90` runs only `stage1-controlled-replay` and treats `stage1-replay-pass` or `local-drycheck-pass` as success. This review's CLI returned `stage1-replay-pass` with sham `control_valid=true` and 192 on-time sham jobs at ~271 ms. That is a different experiment from the live receipt (0 on-time, sham ~7852 ms). A green verifier beside a failed live receipt is the scorecard bug. Registry status remains `protocol-ready`.

## Bundles #50–#55 — drycheck ceilings, and the burn-ledger default

`chelated_pv7_spine/burn_ledger.py:357-429`. `assert_corpus_unburned(..., allow_local_drycheck_smoke=None)` : a non-empty inventory that does not cover all `BURNED_CORPORA` emits `DeprecationWarning` and stamps `inventory_scope=local_drycheck_smoke`, `confirmatory_eligible` false, opt-in `legacy_implicit`. `True` is the explicit smoke opt-in. `False` fails closed. Full coverage of the burned list is the confirmatory path. `admitted_unburned=true` does not mean confirmatory-eligible.

The brief's line numbers are stale except qscci. Current call sites that omit the flag:

| Cited | Current |
| --- | --- |
| `loft.py:402` | `chelated_loft_and_zero_write_ni.py:482` (line 402 is inside `size_disjoint_ni_sample`; a second call at line 461 also omits the flag) |
| `12domain:209` | `chelated_12domain_4hop_gauntlet.py:337` (line 209 is weight-auth) |
| `beam:372` | `chelated_beam_128k_replay.py:493` (line 372 is a `BeamMemoryTurn`) |
| `pm6:178` | `chelated_pm6_shadow_probe.py:206` and `:607` (line 178 is `scratch_family`) |
| `qscci:141` | `chelated_qscci_singleton_probe.py:141`, exact |
| `wild_revisions:740` | `scripts/run_pv7_w18_6c_wild_revisions.py:744` (the historical marker dict starts at 740 and has no `corpora` key) |

This review captured the warning at runtime for two callers, with `CUDA_VISIBLE_DEVICES=""`:

- `admit_scratch_prompts(scratch_families("control", 2, 0))` → 1 `DeprecationWarning` at `chelated_qscci_singleton_probe.py:141`.
- `admit_scratch_prompts([scratch_family(0,"dev"), scratch_family(1,"dev")])` → 1 `DeprecationWarning` at `chelated_pm6_shadow_probe.py:206`.

Message prefix: `assert_corpus_unburned admitted a partial historical inventory without allow_local_drycheck_smoke; stamped local_drycheck_smoke (not confirmatory-eligible)`.

The loft, 12-domain, BEAM, and wild drycheck inventories pass document and query marker fingerprints and no `corpora` key (`scripts/run_pv7_w18_3a_zero_write_ni.py` historical block, `run_pv7_w18_6b_12domain_4hop.py`, `run_pv7_w18_3b_beam_memory.py`, and the inline dict at wild line 740). That is the same `None`-default path. This review did not capture a warning at those four sites at runtime. The code path is sufficient to say they stamp smoke. It is not sufficient to say a warning was observed there.

### #50

`artifacts/pv7_w18_3a_zero_write_ni/pv7_w18_3a_drycheck_manifest.json` top status `local-drycheck-pass`, `passed=true`.
Disjoint block: `is_live_confirmatory=false`, `one_shot_consumed=true`, `ni_decision=pass`, N=1024 pairs, 240 docs, corpus `pv7-b50-disjoint-power-sized-ni`, burn admission sha `770884c5af546ec1e8d7134869328d425be8839291eea15c5585860d3db2a09a`, burn ledger sha `4fd822dcdcfaa2fc8c74b4170103a3d868d519b5f26d097f3b0b734b5a59d65c` (same ledger sha the #47 verifier printed).
Tango: `tango_score_lcb95=-0.021463891175340293` against margin `-0.05`, `clears_ni_margin=true`, `n00=100`, `n01=34`, `n10=28`, `n11=862`, `delta=-0.005859375`, Clopper LCB `-0.030876961913246084`. Historical PM-2 LCB `-0.0572` is preserved. Power sizing `required_n=1000` from the floor, `asymptotic_required_n=180`, `assumed_p10=0.032`, `assumed_p01=0.03`. Weight provenance `reason=drycheck-content-bytes`, `authenticated=false`.
`chelated_loft_and_zero_write_ni.py:830-836` sets `one_shot_consumed=true` whenever the pool is not underpowered and the run is not a refused unauthenticated live confirmatory. A drycheck therefore consumes the one-shot flag while status stays `local-drycheck-pass`. The underpowered probe in the same file (250 pairs) correctly stays `underpowered` with `one_shot_consumed=false` and does not loosen the margin. The admission object copied into the manifest does not include `inventory_scope` or `confirmatory_eligible`.

### #51

Only the drycheck manifest is present. `conversation_clusters=20`, `total_questions=400`, `source_kind=deterministic_local_drycheck_fixture`, `mean_token_length_est=129007.2`. `BEAM_CONTEXT_TOKENS = 131_072` at `chelated_beam_128k_replay.py:58`. Line 433 sets `token_length_est = BEAM_CONTEXT_TOKENS - rng.integers(0, 4096)` on the generated turns. The field is a label near 128k. It is not a measured token count of the template text. No `run-live-gpu` artifact is in the bundle directory.

### #52

`artifacts/pv7_w18_3c_qscci_singleton/pv7_w18_3c_drycheck_manifest.json`: `claim_level=local-drycheck`, `burn_inventory_scope=local_drycheck_sentinel`, `live_weights_loaded=false`, `qwen_report_split_executed=false`. Gate `live_weights_not_claimed` is in the manifest. `report_receipt.json` mode `local-drycheck`, manifest sha `db2e6373e56ad4a995a46f1cd2dcd2bca91a62a5de1331f1a44d64f36ff9fcfd`. The qscci sentinel builds a one-document ledger and calls `assert_corpus_unburned` without the flag at line 141, which this review warned.

### #53

Drycheck: `claim_level=local-drycheck`, `families=320`, `live_pm6_gateway_executed=false`, Holm CP upper `0.013600499218248636`, Holm p `0.0031143987561766083`, `included_in_t1_confirmatory_holm_family=false`.
Separate file `pv7_w18_3d_live_shadow_manifest.json`: `mode=run-live-shadow`, `passed=true`, `claim_level=observational-manifest`, `reason=manifest-shadow-observed`, `live_pm6_gateway_executed=false`, `report_emitted=false`, `shadow_flags=0`, `families=100`, `executed=true`. Gate `live_shadow_executed` passes on the detail string `manifest-shadow-observed`. The local T3 Holm family disclosed there is `1unit` and `4unit` only.
`manifest_sha256=8073afb04d0606c2781416e15db1c1ecfb46df6366d716e0bdf9a778983d5597` equals the committed input `pm6_wave14b_public_domain_shadow_input.json` (40481 bytes, sha256 confirmed this review). That input's `source_manifest` points at `D:/GITHUB/CHELATEDAI-V7-FORMAL-PM6/artifacts/wave14_d11_pm6_takedown_carryforward/wave14_d11_pm6_manifest.json` sha `7248d05a02ba4901aace9fd960c2a4d04d046f83bcd1846426159e5cab9ffc8`. The documents are templated archival sentences. `report_receipt.json` mode `run-live-shadow` hashes output manifest `e791fe5d1730b97bfe76735fdae7ba907631c5ec9cda11633f7a9374278d9458`, which is not the input sha. `53_verifier.py` never loads either file.

### #54

`artifacts/pv7_w18_6b_12domain_4hop/pv7_w18_6b_drycheck_manifest.json`: `mode=local-drycheck`, `status=local-drycheck-pass`, `passed=true`, `dataset_origin=synthetic_schema_drycheck_fixture`, `is_live_confirmatory=false`, `retuned_on_eval=false`, `holm_family_adjudication=null`.
Encoder chain labels MiniLM-384, bge-small-384, mpnet-768, bge-m3-1024, Qwen3-4096 all have `hf_id=""`. Weight provenance `authenticated=false`, `reason=drycheck-content-bytes`, `weight_file_digests` empty, `base_weights_sha256=eecb408cfc07caf2a200a619d6e92bd1b3284f16af26e32fd06fb991d089c241`. `preregistered_config_sha256=024145afe0c1c1b52124d8d41cb91b6dfdaf85d04771a72498e1b5bc5a5b976e`. TRA-1 digest `28e9a14a953e45e3ce43ab7b845fb95402590ede4ad83fc0c59a364f980261a9`.
`H6B_merge`: `passed=false`, `positive_domains=6`, `required_positive_domains=11`, `sign_test_required=11/12`, `lcb_95=-0.00895402030255958`, `exact_binomial_p=0.61279296875`, `registry_status_stamped=false`, `mean_delta=-0.0010717154576982406`, `overall_verdict=local-diagnostic-not-supported`.
Registry primary endpoint (`docs/research/registry.d/54-t6b-12domain-4hop.json`) says `>=10/12` positive. The receipt requires `11/12`. Either threshold fails this cell.
Gate `h6b_merge_evaluated` in the manifest passed with detail `pos=6/12 lcb95=-0.0090 p=0.6128`. The predicate at `scripts/run_pv7_w18_6b_12domain_4hop.py:232-234` is `n_domains==12` and the presence of `exact_binomial_p` and `lcb_95`. It does not require `H6B_merge.passed`.
`H6B_bridge_static.passed=true` with uncentered LCB `-0.0009414508702782779` (retirement floor `-0.01`, tight floor `-0.005`) and centered contrast `mean_delta=-0.013228359347055149`. Shift LCB `0.026161626221738944` (12/12) and 4-hop LCB `0.0069092906343811095` (10/12) passed. Domain ids include TREC-COVID, BioASQ-TaskB, and the rest of the locked 12. Those names sit on a synthetic fixture. `bioasq-taskb` `admitted_unburned` against that fixture is not a BioASQ run. BioASQ is outside `BURNED_CORPORA`.

### #55

`artifacts/pv7_w18_6c_wild_revisions/pv7_w18_6c_drycheck_manifest.json`: `mode=local-drycheck`, `status=DRYCHECK`, `passed=true`, `cohort_source=deterministic_schema_drycheck`, `family_count=4`, `confirmatory_n=false`, `synthetic_labeled_wild=0`, `wild_adjudicated_count=0`, `llm_write_tokens=0`, `authored_prompts=0`.
Families: `dry-echo`, `dry-conflict`, `dry-402`, `dry-amend`, all `synthetic=true`, one each of the four strata. Wild gate still says 240 families and 60 per stratum.
Retirement: `preregistered_conjunction_met=true`, `retire_ek5_proprietary_superiority=false`, `superiority_is_wild_evidence=false`, `sql_accuracy=0.4375`, `ek5_accuracy=1.0`, mirror SQL accuracy `0.625` on 8 trials, bootstrap LCB `0.1875`, mean `0.5625`, UCB `0.75`, resamples 2000, cluster wins 3, ties 1, losses 0, `mcnemar_p=0.25`, `ek5_cost_at_least_1_25x_lower=false`, `family_cluster_count=4`.
Burn: `osv_cve_seeds_rejected=true`, `scored_cohort_texts_checked=16`, `drycheck_texts_admitted=true`, `burn_inventory_scope=local_drycheck_smoke`, `wild_inventory_verified=false`.
`scripts/run_pv7_w18_6c_wild_revisions.py:878-890`: `superiority_is_wild` requires a wild-evaluated cohort and then the conjunction. A drycheck does not set that flag, and the committed JSON keeps it false. The defect is the sibling `passed` bit. Line 884 sets `live_requirements_met = True` whenever `mode != "run-wild"`. Line 890 sets `passed` from that bit and an unchanged registry. A 4-family DRYCHECK therefore carries `passed=true` next to `preregistered_conjunction_met=true`. Registry `55-t6c-wild-revisions.json` already says `synthetic_labeled_wild=0` and `superiority_is_wild_evidence=false`, and it has no `tier` and no `evidence_tier`. Status is still `protocol-ready`. `55_verifier.py` has no `__main__` and its status function returns `wild_evidence=false`. The CLI exit does not read the manifest.

## Bundle #47 — registry `verified-pass` is stronger than the verifier string

`scripts/verifiers.d/47_verifier.py:49-56` calls `run_deterministic_drycheck()` and returns `local-drycheck-pass`. This review's receipt sha is `b76ea52e62ab69969ca2ebcc5cfebb3ec854002b3faadb43e47ce43cffb3ce5b`, host `pv7-w16-local-drycheck-synthetic-identical-arms`. Rejections on that synthetic receipt: disposition 3, latency p95 7, nDCG@10 12, storage p95 6.

The verifier does not open:

- `artifacts/pv7_w16_core_spine/aa_null_calibration_desktop_rtx3090.json`, host `desktop-rtx3090`, seed 3090, schema `pv7_aa_calibration_v2`, trials 200, pairs 2000, 999 permutations, `passed=true`, receipt sha `844c5edd814454e1410f2cd0394f1f09ef69e3a4d34cd4bf98c2ddb5f39a83f8`. Disposition rejections 0 with `identical_measured_pairs=1172`. Latency p95 rejections 5, identical pairs 0. nDCG@10 rejections 8, identical pairs 136. Storage p95 rejections 0, identical pairs 1.
- `artifacts/pv7_w16_core_spine/aa_null_calibration_spark3_gb10.json`, host `spark3-gb10`, seed 3024, same schema and trial counts, `passed=true`, receipt sha `676355e363cc3c0c632a3535696db46089a0dd9d05c39be113e3e8d85c9e4ef9`. Disposition rejections 2, identical pairs 1128. Latency p95 rejections 2, identical pairs 0. nDCG@10 rejections 8, identical pairs 145. Storage p95 rejections 2, identical pairs 8.

Those two files self-describe `passed`. This review did not re-execute them. Registry status `verified-pass` is a stronger word than the verifier's `local-drycheck-pass`, and it is a different receipt sha.

## Bundles #63 and #64

Both registry rows are design tier T4, status `protocol-ready`, and they do not carry an evidence tier.
#63 suite `artifacts/pv7_w21_trial1_partner_search/pv7_w21_trial1_dev_suite.json`: `fixture_split=DEV_OPEN`, `status=protocol-ready`, `suite_sha256=051a00eb0eaf037a74f7a89d281615f78402bedc1221007a69cb691bb4dc5117`. `63_verifier.verify_bundle()` checks outcomes `win_zeromu_csls` / `tie_retire_zeromu_credit_csls` / `loss` and this review's direct call passed. The CLI did not.
#64 suite `artifacts/pv7_w21_trial2_partner_records/pv7_w21_trial2_dev_suite.json`: `data_origin=open_dev_fixture`, `fixture_split=DEV_OPEN`, `status=protocol-ready`, `suite_sha256=3a96a98a7776708c5f6ffaee5eb509864c77ea46d86c09febc9920a7eb00c10e`, `protocol_digest=559a785594464fe8f76ef70c29804a4f688f84306cb9ba3ac504267a0e2ee3ec`. The CLI reproduced that suite sha and stayed at `protocol-ready`.

These are offline DEV scorers. They are not partner-field evidence.

## Concrete repairs before any complete-evidence total

1. Add `evidence_tier` and a receipt path to every `docs/research/registry.d/*.json` row. Keep design `tier` as design. Do not let `status=protocol-ready` or `verified-pass` stand in for the receipt.
2. Give `49_verifier.py`, `55_verifier.py`, and `63_verifier.py` a `__main__` that calls the verifier function and returns non-zero on failure. A 100 ms empty exit must not be a green cell.
3. Teach verifiers 48, 53, 61, and 62 an optional committed-receipt mode that loads the live JSON and fails when `mode`, hashes, or `overall_pass` disagree with the file on disk. Until that exists, publish the verifier status string (`local-drycheck-pass`, `stage1-replay-pass`) and stop.
4. On the #48 receipt, record `m`, `ef_construction`, `full_scan_threshold`, the OpenSearch `disable_exact_search` lock and `filtered_exact_search_threshold`, the pgvector `max_scan_tuples` and `scan_mem_multiplier`, a sha256 of `hnsw_prefilter_pushdown.py`, and the commit that contains `BASE_EF=220`. Stop stamping `d4454e2d`.
5. Change `telemetry_passed` so `graph_telemetry_complete is None` is not a pass. OpenSearch and pgvector need their own basis or an explicit `telemetry_not_collected`.
6. Do not count #62 live as a pass. Keep `replay-fail` on the scorecard until a new live receipt clears `beats_full_restart_rebuild_jpm` and `sham_reopen_control_valid`. Log the chat-template body, including `enable_thinking`, in that receipt.
7. Do not count #49 family names as live models. Stamp `live_model_calls` onto the registry row. A Wilson upper bound of 0/36 is an uncertainty interval, not a failure rate.
8. Pass `allow_local_drycheck_smoke` explicitly at loft:482, loft:461, 12-domain:337, beam:493, pm6:206, pm6:607, qscci:141, and wild:744. Copy `inventory_scope` and `confirmatory_eligible` onto every manifest admission block.
9. Stop setting `one_shot_consumed=true` on a non-live #50 run (`chelated_loft_and_zero_write_ni.py:831`).
10. Split #54 `h6b_merge_evaluated` from `h6b_merge_endpoint_met`. The endpoint is currently false (6/12, LCB −0.009). Align the registry's `>=10/12` with the receipt's `11/12` before either number is quoted as the rule.
11. On #55, do not let `passed` be true for `mode=local-drycheck` merely because the registry bytes were unchanged (`run_pv7_w18_6c_wild_revisions.py:884-890`). Keep `preregistered_conjunction_met` next to `family_count=4` and `superiority_is_wild_evidence=false` whenever it is displayed.
12. Rename #61's stand-in gate so it cannot read as fail-closed while `allow_dry_standin` forces `live_ok`. Include `linear_chain` in the scale predicate, or rename the gate so a false `linear_chain` is visible as a non-requirement.
13. Stop letting #47 registry `verified-pass` outrun `local-drycheck-pass`. If the desktop and Spark3 JSON files are evidence, the verifier must load those paths and check `receipt_sha256`.
14. Keep #63 and #64 at `protocol-ready` while `fixture_split=DEV_OPEN`.
15. On #51, stop reading `token_length_est` as a measured 128k context. Stamp `token_length_est_is_label=true` until a tokenizer count is stored.
16. On #53, stop reading `run-live-shadow` as a live gateway while `live_pm6_gateway_executed=false` and the manifest sha equals the committed input.

## What this review did not do

- Did not re-execute Qdrant, OpenSearch, pgvector, vLLM, or the two host A/A calibrations. Live numbers above are readings of committed JSON.
- Did not call `49_verifier.verify_bundle()`.
- Did not capture the burn-ledger `DeprecationWarning` at runtime for loft, 12-domain, BEAM, or wild. The `None` default and the missing `corpora` key are in the source. The warning was observed only at qscci:141 and pm6:206.
- Did not trace whether the OpenSearch `exact=True` arm bypasses `disable_exact_search`. Recall remains same-engine overlap as implemented in `run_pv7_w17_6a_engine_closure.py`.
- Did not audit bundles #56–#60, and did not invent a 64-cell total.
- Did not assign `BHS_OFFICIAL`. This file is a review memo. It is not a merge request.

## Brutal Honesty

L4 is the scorecard collapse: verifier exit 0 and `protocol-ready` presented as complete evidence. The live artifacts that exist are narrower than their commit messages, and #62's live artifact failed. Empty answers: no sealed-key material was read; no parent-repo file was written; no second scorecard file was produced. The only file this review added is this memo.
