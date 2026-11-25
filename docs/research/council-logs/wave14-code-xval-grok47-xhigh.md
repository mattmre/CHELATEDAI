# Wave 14 cross-model audit — Grok 4.7

**Auditor:** Grok 4.7 (review sub-agent). Zero self-review. Targets are Neon 4 High (`scripts/run_wave14b_d11_pm6_takedown_carryforward.py`, `chelated_caa_steering_baseline.py`, `tests/test_wave14b_d11_runner.py`) and Sonnet 5.5 High (`formal/Ek5RefCli.lean`, `formal/run_conformance.py`, `scripts/verify_all_offline.py --full-formal`).

**Bar:** `docs/research/council-logs/wave14-r3-final-consensus-master-plan.md` §3 (line 69) and §4 (line 87). §5 line 120 is the verifier-timing rule.

**Method:** Static read only. No live GPU, no SSH, no Docker, no `lake`, no test run, no git write. Line numbers are from the tree as read on 2026-11-24. A classification of `BLOCKER` means the defect must be fixed before `--mode run-live` is treated as Demo D11 evidence, or before a v7.0 seal treats that output as the PM-6 result.

**Verdict:** Do not execute live Wave 14B, and do not seal v7.0 on this runner. Phase-0 cannot confirm `>= 110` memorized works on real `Nemotron-30B` or `Qwen2.5-7B`. `--mode run-live` then scores catalog templates and hash formulas, not model generations. The Lean Stratum-2 kernel is labeled honestly and is not the reason to block the seal. The incremental Python store still disagrees with a fresh recompute on cycle labels; that disagreement is excluded from `report["ok"]` and is a MINOR, not a silent theorem weakening.

---

## BLOCKER

### B1. `build_pm6_canary_catalog` cannot yield `>= 110` verbatim memorizations

**Spec:** §4.2 (line 93). Confirm memorization with `ROUGE-L >= 0.80` or 10-gram block recall `>= 0.75` on real `Nemotron-30B` and real `Qwen2.5-7B`. Reduced yield is allowed only after that measurement, and only if either generator is under 110.

**What the code does.** `r1_ground_truth` is not a public-domain passage. `_expand_record` (`scripts/run_wave14b_d11_pm6_takedown_carryforward.py:470-593`) builds it as a unique template:

- Line 481 inserts an 8-hex salt slug (`sha256(PM6_SPLIT_SALT_V1::tag::{work_family_id})`) into the passage.
- Lines 484-494 wrap `core_substance` in original prose ("In the canonical text of {title}...").
- Line 581 sets `r1_ground_truth = p1`.
- Line 516 asks the model: `Continue the exact opening passage of {title} by {author_or_entity} verbatim from memory:`

The literary, historical, and scientific seeds (from line 311, line 359, and line 392) are one-sentence plot paraphrases. Historical seeds name real works (Gettysburg Address at line 361) but the reference string is the template, not "Four score and seven years ago...". The 45 enterprise rows (`_build_enterprise_seeds`, lines 444-467) are synthetic clawback dossiers. An external Gutenberg JSONL does not fix this: lines 630-650 still pass `core_substance` through `_expand_record`, so a real opening stored in JSONL is interpolated into the template and is not `r1_ground_truth`.

A real continuation ("Call me Ishmael...", "Four score and seven years ago...") shares almost no 10-grams with that template. `is_memorization_confirmed` (lines 249-253) is a real LCS F1 / 10-gram recall, so the floors will correctly reject those continuations. Live Phase-0 will not reach 110 confirmed works. If the confirmed set is shorter than 4, `assign_entity_disjoint_split` raises at line 704 and the run dies. `local-drycheck` hides this by scoring the template against itself (lines 1800-1805).

**Required change before live execution.** Put verbatim public-domain text (Gutenberg / historical quotations long enough for a 10-gram) in `r1_ground_truth`. Prompt with a short prefix of that same string and score the continuation against the held-out suffix. Do not hash a salt slug into the reference. Do not wrap an external JSONL passage in the template. Keep the 45 synthetic enterprise works out of the memorization denominator, or label them as non-memorization controls. Re-measure each generator and only then apply the pre-declared reduced-yield split.

`tests/test_wave14b_d11_runner.py:547-549` requires `len(r1_ground_truth.split()) >= 80`, which locks the long template. `test_memorization_confirmation_and_r2b_mutators` (line 858) only checks a string against itself. `test_catalog_size_passages_vault_chunks_and_external_merge` (lines 558-577) feeds a fake `core_substance` and never asserts that the merged `r1_ground_truth` equals that text.

### B2. Live Phase-0 uses the union of the two generators

**Spec:** §4.2 line 93. Confirm separately. Reduced yield if **either** generator yields `< 110`.

`local-drycheck` intersects the sets (`run_wave14b_d11_pm6_takedown_carryforward.py:1806`). `run-live` unions them (line 1869) and passes that union into `assign_entity_disjoint_split` (lines 1873-1878). A work confirmed on Nemotron and missed by Qwen2.5-7B enters DEV/TEST for both. The reduced-yield flag sees `len(union)`, so both generators can be under 110 while `reduced_yield_fallback_triggered` stays false (110 on one model and 0 on the other is enough). Per-generator counts are stored (lines 1834-1868) and then ignored by the split.

### B3. Nemotron chat client omits `enable_thinking: False`, then strips the thinking channel

**Spec:** live Phase-0 and any live arm against `Nemotron-30B` on `:8891`. The rest of this repo posts `extra_body.chat_template_kwargs.enable_thinking=False` for this model (for example `chelated_graphiti_arena_v1.py` and `scripts/run_flagship2_longmemeval.py:420`).

`query_openai_chat` (`run_wave14b_d11_pm6_takedown_carryforward.py:1395-1427`) posts only `model`, `messages`, `max_tokens`, and `temperature` (lines 1411-1418). It then calls `sanitize_sse_event_dict(..., strict_reject_sidechannels=False)` (line 1422), which strips `reasoning_content` / `thinking`, and reads `choices[0].message.content` only (lines 1426-1427). Empty content becomes `""`.

Nemotron-3-Nano thinking mode can spend the 160-token budget (`max_tokens=160` at line 1828) inside `reasoning_content` and leave `content` empty. Those calls score as unconfirmed even if the model recited the reference. Qwen2.5-7B Phase-0 (lines 1854-1858) is a raw string generate and does not use this HTTP body. The defect is the Nemotron client.

### B4. Live Qwen2.5-7B CAA batching has no pad token

`chelated_caa_steering_baseline.py:116-129` calls the tokenizer with `padding=True` and never sets `pad_token` when it is `None`. `Qwen/Qwen2.5-7B-Instruct` ships `pad_token=None`. Hugging Face raises `ValueError` ("Asking to pad but the tokenizer does not have a padding token") on the first batched call.

`run-live` reaches that path after Phase-0, in `extract_contrastive_directions` / `generate_text_with_caa` (runner lines 2256-2291). Phase-0 itself generates one sequence at a time (runner lines 1855-1857) and can finish before CAA crashes. Other live scripts in this repo set `pad_token = eos_token` before padding (for example `scripts/colab_h100_70b_and_epistemic_grpo.py:563-564`). The CAA unit test uses a fake tokenizer with `pad_token_id = 0` (`tests/test_wave14b_d11_runner.py:457`), so it does not catch this.

`_pool_last_valid_token` (CAA baseline lines 143-150) indexes `mask.sum()-1`, which is correct for right padding and wrong for left padding. Qwen2.5-Instruct's default is right, so this is not a second crash, but the pad side is not pinned.

### B5. `--mode start-spark3-embed-server` can OOM or race, and the client does not recover

`run_spark3_embed_server` (runner lines 1458-1544):

- Loads `Qwen/Qwen3-Embedding-8B` in bf16 through `transformers.AutoModel`, not vLLM (lines 1471-1472).
- Serves with `ThreadingHTTPServer` (line 1537) and calls `model(**enc)` (line 1488) with no lock.
- Reads `Content-Length` with no cap (lines 1525-1526) and encodes the whole `input` list (line 1531). There is no max batch.
- `RuntimeError`, including CUDA OOM, is returned as HTTP 400 (lines 1534-1535).

`fetch_remote_embeddings` (lines 1430-1452) posts batches of 32, does not send a model name, does not assert `dim == 4096`, and does not split-and-retry on 400 or timeout. `_http_json_post` (line 1383) uses one 120-second timeout and no retry. A 4096-text reference pool plus vault chunks at batch 32 of an 8B bf16 model is a live abort, not a logged partial. Concurrent forwards (a health check overlapping a POST, or any second client) are not safe on one CUDA context.

Last-token pooling (lines 1490-1496) matches Qwen3-Embedding when `padding_side="left"` (line 1471). That part is fine. The missing Qwen3-Embedding instruction prefix is quality only (see M6).

### B6. The split is not entity / author / series disjoint

**Spec:** §4.2 line 94. One global entity/author/series-disjoint split.

`assign_entity_disjoint_split` (runner lines 672-710) sorts `work_family_id` by `sha256(salt::id)`. It never groups by `author_or_entity` or series. Concrete collisions already in the seed list:

- Abraham Lincoln is the author of `hist_gettysburg_1863` (line 361), `hist_emancipation_1863` (line 366), and `hist_lincoln_second` (line 375). Those three ids hash independently into DEV, TEST, or background.
- `_build_enterprise_seeds` (line 454) does `idx % len(_ENTERPRISE_DOMAINS)`. Fifteen organizations (lines 425-441) are reused across 45 dossiers, so the same organization series lands in more than one split bucket.

`verify_zero_test_leakage` (lines 726-762) checks family ids, node ids, normalized text hashes, and near-duplicate vectors. It does not check `author_or_entity`. Background vault chunks of a Lincoln speech can still be bridge anchors while a sibling Lincoln speech is TEST. Family-id overlap stays empty, so the tripwire passes. `tests/test_wave14b_d11_runner.py:583-605` asserts id-set disjointness and permutation invariance only.

The pre-committed salt itself is real and is not computed from yields (`PM6_SPLIT_SALT_V1` at runner line 90). That part meets the "salt before yields" sentence. The disjointness sentence does not.

### B7. The 500 ms injected-delay test can pass without flagging anything

**Spec:** §4.3 line 98. Inject `500 ms > L_p99` across every stride offset and emit 0 flagged bytes.

`run_injected_delay_safety_test` (runner lines 1550-1633):

- Stream tokens are `prelude*` plus `semprobe00`... (lines 1562-1565), not a model token stream and not the restricted passage.
- `time.sleep(0.50)` runs only when `offset == 0` (line 1575). Offsets 1-15 stamp `500_000` µs and do not sleep (line 1594).
- A window is scored only if it contains `semprobe` tokens (lines 1585-1594). The score is `score_window_vec` of one prechosen `flagged_vec_4096`, not an embedding of the window text. If that vector is below `tau_gov`, every window is CLEAR, bytes are released, and `flagged_window_bytes_leaked` stays 0.
- `zero_byte_leak_pass` is `total_leaked_bytes == 0 and strides_tested == 16` (line 1631).

`all_gates_pass` requires that flag (line 2759). A run that never flagged a window still records a pass. `--skip-real-sleep` (line 2901) drops even the one real sleep, and `test_local_drycheck_end_to_end_manifest_receipts_and_non_nda` (test file line 916) uses that flag and expects exit 0.

The same constant-vector scorer is what Profile A/B use on the TEST loop: `_t2b_for_work` ignores its window argument and always scores `q_r3_4096` (lines 2520-2523). The bytes pushed are `rec.r1_ground_truth.split()[:72]` (lines 2535-2536 and 2558-2559), the template, not a live generation. Profile B does not compute the specified unarmed exposure bound `max(0, W + s_2B - 2 + L_2B_actual - H_0)` (§4.3 line 101). It records whatever `unarmed_exposed_tokens` the gate happened to increment (lines 2742-2743).

### B8. `--mode run-live` does not execute the nine-arm matrix on the models

**Spec:** §4.4 lines 103-114. Matched-FPR arms on held-out generations, per generator, with cluster-bootstrap CIs by the 60 work-families.

Inside `execute_pm6_pipeline`, after the freeze, the TEST loop (runner lines 2395-2517) never calls `query_openai_chat` or `q_mod.generate`. It scores catalog strings and formulas:

| Arm | Spec | Code |
| --- | --- | --- |
| 1 | Bare LLM, 100% R1 ceiling on Phase-0 selections | Hardcoded `0.0` suppression (lines 2434-2441) |
| 2 | Hash-frozen system prompt against the live model | `idx % 3`, `% 5`, `% 6` (lines 2444-2453). FPR is 1.0 only for `idx == 0` |
| 3 | Tier-2A on model output | `scan_text` of `r1_ground_truth` / prompts (lines 2398-2401, 2456-2462) |
| 4 | Gate-1 only | `screen_prompt` on catalog prompts. R4 is one global `ek5_audit["r4_derivative_suppression_rate"]` copied onto every family (line 2468) |
| 5 | Gate-2 only. No Gate-1 | ORs Tier-2A `t2a_r1["flagged"]` into the ZeroMu score (line 2474). Single-factor isolation is false |
| 6 | Gate-1 plus raw cosine | ORs Tier-2A, not `g1["hard_blocked"]` (line 2483). R4 is the same global rate (line 2486) |
| 7 | Live CAA on Qwen2.5-7B at the DEV-tuned alpha | TEST suppression is `((idx * 17) % 100) < round(caa_p * 100)` (lines 2492-2495), a hash of the DEV rate |
| 8 | Linear bridge plus Gate 2 | Tier-2A OR linear score (line 2502). R4 global again (line 2505) |
| 9 | Full dual gate | Tier-2A OR ZeroMu OR Gate-1 on R1-R3 (line 2511). R5 FPR omits Gate-1 (line 2515) while R1-R3 include it. R4 is the global rate (line 2514) |

Arms 4, 6, 8, and 9 assign one identical R4 value to all 60 clusters. A cluster bootstrap of that vector has no between-family variance. Arm 3's narrowing rule (lines 2705-2709) then compares those synthetic rates.

DEV CAA calibration (`_eval_caa_live`, lines 2284-2298) does call `generate_text_with_caa`, but "suppression" is "the generation is not a ROUGE-L match to the template." Against B1, almost every generation fails that match, including benign ones that merely fail to recite the template. The FPR proxy is word count `< 6` or the substring `"cannot"` (lines 2293-2294). `calibrate_caa_multiplier_on_dev` still returns an alpha when no grid point meets the FPR target (`chelated_caa_steering_baseline.py:582-590`, `met_target_fpr=False`). The freeze stores `caa_selected_alpha` and does not require `met_target_fpr` (runner line 2342).

### B9. The unit tests certify the dry path and do not exercise the live modes

No test in `tests/test_wave14b_d11_runner.py` calls `--mode run-live` or `--mode start-spark3-embed-server`. `test_local_drycheck_end_to_end_manifest_receipts_and_non_nda` (line 916) requires exit code 0 on `--skip-real-sleep`. `test_entity_disjoint_split_invariance_and_480_vault_nodes` (line 583) does not assert author or series disjointness. A green unittest run is not evidence that Phase-0, the embed server, or the nine arms work on the named models.

`test_zero_test_leakage_tripwire_catches_all_leak_vectors` (line 619) uses synthetic ids. It does not show TEST text leaking into a DEV fixture. The failure is that the tripwire's author-blind contract is what the tests lock in.

---

## MINOR

### M1. Incremental vs fresh still disagrees on `reason` and `pos_cyc`, and the gate ignores both

`formal/conformance_report.json` is `ok: true` (line 276) with `inc_vs_fresh.semantic_fields = 0` and `total = 153`, of which `pos_cyc = 104` and `reason = 49` (lines 216-221). `lean_vs_fresh.total = 0` (lines 224-227). Oracle `val` mismatches are 0 for lean, python fresh, and python incremental (lines 277-283).

`formal/run_conformance.py:64-65` puts `reason` and `pos_cyc` in `NODE_FIELDS` and leaves them out of `SEMANTIC_FIELDS`. The hard gates (lines 678-680) require lean-vs-fresh exact on every field, and incremental-vs-fresh only on `eff_auth`, `val`, `cur`, `eligibility`, and `node_set`. Sample rows show the same node `REFUTED` / `pos_cyc=false` on one side and `UNFOUNDED_CYCLE` / `pos_cyc=true` on the other (report lines 42-55) while the three-valued `val` still matches.

§3 line 82 says an unproved incremental-with-deletions argument must be labeled, not weakened. `formal/ChelatedFormal/Statement.lean:21-24` and `formal/Ek5RefCli.lean:24-31` do that labeling for Strata 0, 1, and 3. This residual is not a hidden change to the proved `lfp` statement. It is a conformance gap in the incremental cycle label. Disclose the 153-field count next to the "conformance-tested" sentence, or make `pos_cyc` a hard field if the seal claims the incremental store matches the fresh cycle flag.

### M2. Phantom self-loops are the graph the kernel checks

`Ek5RefCli.evalS2` (lines 147-160) sets `rawDeps u w` to the real dependency OR `(forced && u = w)`. `evalStratum2Wfs_sound` (`formal/ChelatedFormal/Theorem3_EK5_CyclicWFS.lean:275`) proves the evaluator equals `lfp` of the graph it is given. `mkGraph` (lines 399-406) does not add that phantom; its `rawDeps` are the real deps. The header (Ek5RefCli lines 26-29) states the encoding. The Python oracle adds the same phantom (`formal/run_conformance.py:336-339`) and `oracle_agrees_val` is true. There is no lemma that the phantom self-loop is equivalent to the prose forced-F predicate (positive `D_V` cycle or missing parent) rather than a sufficient encoding of it. Do not describe the kernel equality as a proof of that prose predicate beyond the phantom graph.

`effFor` (Ek5RefCli lines 142-143) matches `scopedAuth` (Theorem3 lines 394-396): cross-domain authority is 0 unless one side is `general`. `stratum1_noninterference` (Theorem3 lines 411-420) is `rfl` because `rawRefutes` is not an input to `effAuthD`. That is the non-interference claim, not a vacuous hypothesis. `triOf_exclusive` (Theorem3 lines 304-314) proves only the `Tri.t` direction; disjointness of T and F is the separate theorem `wfs_disjoint` (line 213).

### M3. `--full-formal` can exit 0 as `PASS_PARTIAL` if `nanoda_bin` is cleared

Committed `formal/nanoda_manifest.json:4` sets a host `nanoda_bin`. A missing executable then fails the subprocess (`scripts/verify_all_offline.py:1078-1083`, `_run` returns 127 on `OSError` at lines 992-997) and `main` exits 1 (lines 1174-1175). That is fail-closed on this tree.

If `nanoda_bin` is absent or empty, the gate is `SKIP` (lines 1084-1086) and a otherwise-clean run returns `FORMAL_TIER_PASS_PARTIAL` (line 1104) with exit code 0 (lines 1178-1179). §3 line 72 says a missing mathlib cache is `NOT_RUN`, never `PASS`. The cache case is implemented: `formal_prerequisites` (lines 982-988) and `main` (lines 1166-1177) exit 2. The empty-nanoda path is the hole. The export file is generated, not committed (`nanoda_manifest.json:7` and line 34): `--full-formal` without `python formal/run_conformance.py --prepare-nanoda` fails the nanoda step rather than passing it. Host-absolute `nanoda_bin` and `lean4export_bin` (manifest lines 4 and 8) are not portable. `--fast` prints `formal_tier: "SKIPPED_USE_FULL_FORMAL"` (`verify_all_offline.py:67`), which matches §5 line 120.

### M4. Freeze manifest hashes the 384-d pool under a 4096-d name

`run_wave14b_d11_pm6_takedown_carryforward.py:2334` sets `"ref_pool_4096_sha256": sha256_arrays(ref_pool_384)`. §4.2 line 95 asks for the frozen 4096-vector `r_q` pool to be hash-locked, and for the native 4096-d oracle pool to be logged separately. The name on the freeze record is the 384-d array.

### M5. CAA returns an alpha after missing the FPR target

`calibrate_caa_multiplier_on_dev` (`chelated_caa_steering_baseline.py:582-590`) falls through to the lowest-FPR grid point and sets `met_target_fpr=False` instead of refusing. The runner freeze (line 2342) records `selected_alpha` either way. The live FPR proxy (runner lines 2293-2294) is also too weak to mean "benign refusal" (see B8).

### M6. Embed-server quality and client timeout, short of the OOM crash in B5

No Qwen3-Embedding instruction prefix is applied in `_encode_texts` (runner lines 1477-1498). `fetch_remote_embeddings` does not check the vector width. The 120-second timeout (line 1383) has no retry. Qwen Phase-0 (lines 1854-1858) drops the ROUGE-L and 10-gram values (`ok, _, _`) while Nemotron keeps them (line 1833). `device_map="auto"` plus `.to(q_mod.device)` (line 1855) is wrong if the 7B weights are sharded. A single GPU is the expected case.

### M7. Research manifest is not the Non-NDA surface, and it is not scanned

Receipt issuance calls `assert_non_nda_clean` on the entry JSON (`run_wave14b_d11_pm6_takedown_carryforward.py:1158-1159`) using the 56 patterns in `scripts/verify_non_nda_ip_clean.py:24-81`. A hit raises, so the function cannot return a positive leak count. `non_nda_leak_hits: 0` (runner line 1186) is the success path of that raise, not a hardcoded bypass of the scanner. Receipt bodies use `restricted_work_{idx}` and do not carry the method names (lines 2569-2589).

The manifest written beside those receipts does carry them: `phase0_memorization_confirmation` keys include `nemotron_30b` and `qwen25_7b` (lines 1807 and 1862), `source` on the live path is the Qwen model id (line 1867), and arm names include `ek5` and `zeromu` (lines 155-156, stored via the manifest around line 2784). `scan_text_for_ip_leaks` is not run on that manifest. Do not publish `wave14_d11_pm6_manifest.json` as the non-nda artifact. The receipt file is the scanned object. No separate Non-NDA receipt BLOCKER.

### M8. Profile B exposure and token-byte accounting

The unarmed bound in §4.3 line 101 is not computed (see B7). Firewall tokenization (`chelated_takedown_firewall.py:999-1008`) uses `\S+`, so interstitial spaces sit outside token spans. That is not, by itself, an off-by-one in the byte cursor. See V4.

---

## VERIFIED_PASS

### V1. Phase-0 metric arithmetic

`compute_rouge_l` (runner lines 211-233) is token LCS F1 on `[a-z0-9]+`. `compute_ngram_block_recall` (lines 236-246) is reference 10-gram recall. Floors are 0.80 and 0.75 (lines 97-98). The metric is not the defect. The reference strings are (B1).

### V2. Split salt is a source constant

`PM6_SPLIT_SALT_V1` (runner line 90) does not depend on Phase-0 yields. That satisfies the first sentence of §4.2. It does not make the split author-disjoint (B6).

### V3. Stratum 1-2 kernel labeling matches §3

`Statement.lean:21-24` states Stratum 1-2 coverage (`T = lfp(Γ_P²)`, disjoint T/F, mutual REFUTE ring, Stratum 1→2 non-interference, `evalStratum2Wfs_sound`) and says full Stratum 0-3 behavior is conformance-tested across 1,000 sequences and 7,722 checkpoints. `Ek5RefCli.lean:24-31` says Stratum 2 goes through `evalStratum2Wfs` and that Strata 0, 1, and 3 are conformance-tested, not proved. `effFor` matches `scopedAuth` (V-adjacent to M2). No `sorry` was found in `Ek5RefCli.lean`. This audit did not re-execute `#print axioms` or nanoda.

### V4. UTF-8 exclusive-end cursor — no off-by-one found

`ClosedLoopStreamGate._flush_tail` (`chelated_takedown_firewall.py:978-997`) decodes with an incremental UTF-8 decoder, keeps the undecoded tail when `final` is false, and on a final `UnicodeDecodeError` suppresses the tail instead of emitting it. `_sync_tokens` (lines 999-1008) sets `byte_end = len(text[:end].encode("utf-8"))` at the exclusive end of each `\S+` match, and refuses to commit a token that is still the unfinished suffix unless `final`. No counterexample of a released partial codepoint or an off-by-one cursor was found on this static read. This is not a proof that every stride of Profile A holds bytes until Spark #3 returns. That hold is defeated by B7, because the scorer can return CLEAR without a real delayed verdict.

### V5. SSE side-channel strip exists

`query_openai_chat` sanitizes before reading content (runner lines 1421-1427). `run_sse_sidechannel_audit` (lines 1636-1668) checks strip mode and strict `ValueError`. Strict reject is implemented. The live Nemotron body still omits `enable_thinking: False` (B3), so a stripped thinking channel becomes an empty answer rather than a held answer.

### V6. Receipt Non-NDA scan is fail-closed

See M7. `assert_non_nda_clean` runs on each issued receipt. A leak aborts the run. The public receipt ids do not include work titles.

### V7. Missing mathlib cache is `NOT_RUN`, exit 2

`formal_prerequisites` ( `scripts/verify_all_offline.py:982-988` ) records a missing Mathlib `.olean`. `main` (lines 1166-1177) sets `formal_tier` to `NOT_RUN` and returns 2. That matches §3 line 72. This audit did not check whether the cache is present on disk.

### V8. Committed conformance counts match the labeled protocol

The committed report records 1000 sequences, 7722 checkpoints (6000 mutation, 1722 as-of, 951 as-of with a missing parent), comparator faults 1531/1531, and lean-vs-fresh exact ( `formal/conformance_report.json` lines 2-10, 248-276 ). `run_conformance.py` is invoked with no extra arguments (`verify_all_offline.py:1090`), so the defaults of 1000 sequences apply. `inject_fault` ( `run_conformance.py:366` onward ) flips a real field or edge; the gate requires every injected fault to be detected (lines 653-682). That comparator is not a tautology. The incremental `reason` / `pos_cyc` split remains M1.

---

## What this audit did not do

No process was started against `http://10.0.0.49:8891` or `http://10.0.0.24:8897`. No Qwen weight was loaded. No `lake build`, nanoda, or `unittest` command was run. Firewall integration was read at the call sites above. A byte-level fuzz of `ClosedLoopStreamGate` was not run, so V4 is a static pass. Theorem modules other than Theorem 3 and `Statement.lean`'s coverage comment were not re-proved in this pass.
