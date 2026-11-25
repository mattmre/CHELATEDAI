# Wave 12 Non-Author Cross-Model Validation (`v6.0` Part VII: Sealed Demo D10 and Re-Sealed Demo D8)

- **panel_type:** brutal-honesty-v3.3-non-author-xval
- **run_date:** 2026-11-25
- **run_id:** wave12-xval-grok47-xhigh
- **reviewer:** grok-4.7 (non-author)
- **repo_head_audited:** `9e4921a99897d3acfed2f11206572003a8d4cd92` in `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1`
- **hard_safety_compliance:** `D:\GITHUB\CHELATEDAI` was not modified. No code, test, receipt, or manifest in the night-1 tree was edited. No `*sealed_key*` or `*blind-followup*` path was opened. No GPU job was run (`CUDA_VISIBLE_DEVICES=""`).

**Verdict:** **PASS_WITH_MANDATORY_DISCLOSURES**

The sealed D10 receipt binds the manifest and both input summaries. Independent Holm step-down on the stored Part A p-values reproduces the sealed 24-contrast family, including the Quora loss. Part B is sealed `COMPLETED` on `unsloth/Llama-3.3-70B-Instruct-bnb-4bit`. Its pre-registered Holm family has two contrasts and rejects neither. Both Sol 6.2 D8 blockers are fixed in the tree that matches `4b1520cd` and are covered by CPU tests that passed in this audit. Whitepaper `v6.0` still has to carry the Part A loss table and the Part B scope limits below. This audit did not rebuild the 9.47M index and did not re-execute the 70B sweep.

---

## (a) Verified SHA-256 and statistical recomputations

### A.1 Demo D10 binding

| Artifact | SHA-256 recomputed from the file bytes | Binding |
|---|---|---|
| `artifacts/wave11_d10_5m_real_cutover_and_70b_scale/wave11_d10_scale_manifest.json` | `e50b8746038755f16dc17d09ce15e13fe25c43c71131308477757f36500d9032` | Equals `report_receipt.json` `manifest_sha256` |
| `part_a_cutover_summary.json` | `166a44a5ec5e6462ef788ec1237acf7cca6e52e33beba0419df71827d9c3a88b` | Equals receipt `artifact_hashes`, receipt `input_part_a_sha256`, and manifest `input_part_a_sha256` |
| `part_b_70b_sweep_summary.json` | `e8d9906f7359b1c0912c85a31b2ad2b174012107047213a6f903ecdcfb18fd6f` | Equals receipt `artifact_hashes`, receipt `input_part_b_sha256`, and manifest `input_part_b_sha256` |

The manifest embeds both summaries. `part_a_real_corpus_cutover` is byte-equal to `part_a_cutover_summary.json` after JSON parse, and `part_b_70b_frontier_scale_sweep` is byte-equal to `part_b_70b_sweep_summary.json`. Calling `compute_part_a_multiplicity_and_loss_disclosure` on the Part A file reproduces `part_a_multiplicity_and_loss_disclosure` exactly, including all 24 Holm rows and all 9 loss rows.

Provenance: the manifest records `git_sha` `2a922f29206739dd0664117262eb04af107a6f54` (`chore(d8): re-seal...`, 2026-11-25 15:35:53Z). The receipt has no `git_sha`. The commit that adds the four D10 artifact files is HEAD `9e4921a9`. `git diff 2a922f29 -- scripts/run_wave11_d10_5m_real_cutover_and_70b_scale.py tests/test_wave11_d10_scale_proof.py` is empty, so the runner and the D10 tests at HEAD are the bytes that assembled this seal. Receipt `generated_at_utc` is `2026-11-25T15:52:09.157096+00:00`.

### A.2 Part A arithmetic that does not require the corpus

These identities were recomputed from the sealed integers. They do not recount BeIR.

- `8,841,823 + 623,716 = 9,465,539` and `7,574,082 + 1,891,457 = 9,465,539`.
- `synthetic_vector_fraction = 0.0`. `msmarco_subsampled = false`.
- Sidecar `8 * n` bytes: merged `75,724,312` bytes = `72.2163 MiB`; MSMARCO `70,734,584` bytes = `67.4578 MiB`. Ratios versus float32 are `192.0` (384-d), `512.0` (1024-d), and `2048.0` (4096-d). Rounded GiB footprints match the seal on both corpora (`36.1082 / 144.4327` GiB merged; `33.7289 / 134.9155` GiB MSMARCO).
- Packed fence bitset `ceil(7574082/8) + ceil(1891457/8) = 1,183,194` bytes.
- Stiefel anchors `2,000 / 9,465,539 = 0.00021129`, sealed `0.0002113`. Sealed orthogonality max error `2.4424906541753444e-15`.
- `IndexHNSWSQ(QT_fp16, M=32, efConstruction=128, efSearch=512, METRIC_INNER_PRODUCT)`. Query timings `p50 = 3.638 ms`, `p95 = 4.464 ms`. `upgrade_model_1024 = BAAI/bge-m3`.
- SHA-256 of `json.dumps(sampled_oracle_qids)` over the 50 stored SciFact ids equals `4b84ae3fe38533893bc8943c97ef0ce0df44962f1bb3ba0ceea1fd664a0495e4`. Sealed ANN-vs-exact recall is `0.984` on those 50 queries (floor `0.85`).
- Fence proof: `200` retracted docs, `0` fenced leaks (`0.0%`) versus `200/200` unmasked (`1.0`), `fill_at_10 = 1.0`, fenced admitted Recall@10 `0.944` on 25 distinct MSMARCO query ids.

The 1024-d shard is the realized `sha256(id|upg20)` first-4-bytes `% 5 == 0` subset of all seven domains, `1,891,457 / 9,465,539 = 19.9826%`. It is not "the six BEIR corpora fully re-embedded."

### A.3 Part A Holm-Bonferroni, recomputed from stored p-values

Eight corpora times three contrasts is 24. An independent step-down (sort by `(raw_p, name)`, adjusted p = running max of `raw_p * tests_remaining`, alpha `0.05`) matched every sealed `holm_adjusted_p` and every `reject_null_at_alpha` with zero mismatches. The same rows match `holm_bonferroni_stepdown` in `wave6_common.py`.

Per-query score rows are not in this bundle. The bootstrap that produced each p-value was not replayed. The stored two-sided p-value uses `min(1, 2 * min(P(boot<=0), P(boot>=0)) + 1/20000)` at `wave6_common.py:155`, so `5e-05` is the resolution floor, not a smaller tail. Ten contrasts sit on that floor. All ten reject. Holm adjusted p for each of them is `0.00120` (`24/20000`).

Nine of those ten are Chelated nDCG@10 wins:

| Contrast | mean delta nDCG@10 |
|---|---|
| `merged_9_47m_fiqa` vs `DEV_ZScore` | `+0.01445` |
| `merged_9_47m_fiqa` vs `Uncalibrated_Raw` | `+0.04282` |
| `merged_9_47m_msmarco` vs `Uncalibrated_Raw` | `+0.06042` |
| `merged_9_47m_quora` vs `DEV_ZScore` | `+0.09665` |
| `merged_9_47m_quora` vs `Uncalibrated_Raw` | `+0.31169` |
| `merged_9_47m_scidocs` vs `RRF_k60` | `+0.01355` |
| `merged_9_47m_scifact` vs `RRF_k60` | `+0.02847` |
| `msmarco_standalone_8_84m` vs `RRF_k60` | `+0.02235` |
| `msmarco_standalone_8_84m` vs `Uncalibrated_Raw` | `+0.04538` |

One is a loss. `merged_9_47m_quora::chelated_minus_rrf_k60` mean delta `-0.06919467090155686` (reported `-0.06919`). Sealed means are Chelated `0.52647` versus `RRF_k60` `0.59567`. The 95% percentile interval is `[-0.09138, -0.04750]`, entirely below zero. Holm rejects. Both `statistically_significant_loss` (unadjusted interval) and `statistically_significant_holm_005_loss` are true, and they agree on every sealed nDCG loss row.

Five contrasts have raw `p < 0.05` and do not survive the 24-test Holm gate:

| Contrast | delta | raw p | Holm p |
|---|---|---|---|
| merged MSMARCO vs `RRF_k60` | `+0.01691` | `0.00395` | `0.05530` |
| merged MSMARCO vs `DEV_ZScore` | `+0.01164` | `0.00695` | `0.09035` |
| SciDocs vs `DEV_ZScore` | `+0.00405` | `0.00695` | `0.09035` |
| NFCorpus vs `RRF_k60` | `+0.00294` | `0.01895` | `0.20845` |
| SciDocs vs `Uncalibrated_Raw` | `+0.00508` | `0.03975` | `0.39750` |

Two nDCG@10 deltas are negative and not rejected: FiQA vs `RRF_k60` `-0.00123` (interval `[-0.00941, +0.00674]`, Holm p `1.0`) and NFCorpus vs `DEV_ZScore` `-0.00079` (interval `[-0.00269, +0.00111]`, Holm p `1.0`).

Reconstructing each of the 24 deltas as `round(means, 5 digits)` minus the challenger mean differs from the stored unrounded `mean_delta` by at most `8.08e-06` (merged MSMARCO vs `DEV_ZScore`). That is display rounding.

Six Recall@10 point losses are disclosed and are not in the Holm family. No recall contrast has a p-value in this seal:

| Contrast | Chelated | Challenger | delta |
|---|---|---|---|
| ArguAna vs `RRF_k60` | `0.198` | `0.228` | `-0.030` |
| Quora vs `RRF_k60` | `0.70716` | `0.73676` | `-0.0296` |
| standalone MSMARCO vs `RRF_k60` | `0.24133` | `0.24433` | `-0.003` |
| SciFact vs `RRF_k60` | `0.10917` | `0.11194` | `-0.00277` |
| NFCorpus vs `DEV_ZScore` | `0.01064` | `0.01126` | `-0.00062` |
| NFCorpus vs `Uncalibrated_Raw` | `0.01064` | `0.01095` | `-0.00031` |

Standalone MSMARCO vs `RRF_k60` is a Holm win on nDCG@10 (`+0.02235`) and a point loss on Recall@10 (`-0.003`).

### A.4 Part B, recomputed from the sealed summary

Status `COMPLETED`. Suite `prospective_realworld_drift_suite_n60`, `n_cases = 60`, suite SHA-256 `cb6bc0082b5850f4edfea668d1ecb2a73fe6f898b7476e34ed11036cb378765b`. `design_scope = reader_size_varied_under_fixed_nemotron30b_extractor_governance_pipeline`.

Locked reader: `unsloth/Llama-3.3-70B-Instruct-bnb-4bit`, quantization string `4-bit NF4 (bitsandbytes Linear4bit, bfloat16 compute)`, nominal `70.6B`, load `245.22 s`, peak CUDA allocated `38.084 GiB`. This audit did not reload that model.

Arms present: `Bare_Llama_3_3_70B_4bit_HybridRAG_k5`, `Bare_Llama_3_3_70B_4bit_FullContext_k17`, `Llama_3_3_70B_4bit_Chelated` (label `Llama-3.3-70B (4-bit NF4) Reader + Chelated (Nemotron-30B Extractor)`), `Small_Qwen_Plus_Chelated_k5` (`Qwen/Qwen2.5-1.5B-Instruct`), `Mid_Qwen2_5_7B_Plus_Chelated_k5` (`Qwen/Qwen2.5-7B-Instruct`), `Bare_Qwen2_5_7B_bf16_HybridRAG_k5`, `Bare_Nemotron_30B_NVFP4_HybridRAG_k5`.

Strict exact match on N=60, as fractions consistent with the 5-decimal seal:

| Arm | strict exact | Nemotron-30B judge | Qwen2.5-7B judge | stale leak |
|---|---|---|---|---|
| Bare 70B k5 | `58/60 = 0.96667` | `1.0` | `58/60 = 0.96667` | `0.0` |
| Bare 70B k17 | `59/60 = 0.98333` | `1.0` | `59/60 = 0.98333` | `0.0` |
| Chelated 70B | `59/60 = 0.98333` | `1.0` | `59/60 = 0.98333` | `0.0` |
| Chelated 7B | `59/60 = 0.98333` | `1.0` | not scored (same family) | `0.0` |
| Chelated 1.5B | `55/60 = 0.91667` | `0.95` (`57/60`) | not scored (same family) | `0.05` (`3/60`) |
| Bare Qwen2.5-7B | `0.90000` (`54/60`) | `0.90000` | not scored (same family) | `0.01667` (`1/60`) |
| Bare Nemotron-30B | `0.71667` (`43/60`) | not scored (same family) | `0.73333` | `0.26667` (`16/60`) |

Judge routing in the runner matches that pattern: Nemotron-30B grades Llama and Qwen arms; Qwen2.5-7B grades Llama and Nemotron arms.

Gate versus reader cohorts exist only on the three Chelated arms. `14 + 46 = 60`. Chelated 70B and Chelated 7B: gate exact `13/14 = 0.92857`, reader exact `46/46 = 1.0`, which weights back to `59/60`. Chelated 1.5B: gate `13/14`, reader `42/46 = 0.91304`, which weights back to `55/60` within `1e-5` of the sealed `0.91667` after the cohort rates are rounded to five decimals. Chelated 70B Qwen-judge: gate `13/14 = 0.92857`, reader `46/46`, overall `59/60`.

Cohen's kappa between Nemotron-30B and Qwen2.5-7B on the Llama-70B rows was recomputed from those rates. Nemotron's sealed accuracy is `1.0` on every 70B arm, so its label vector on those rows is constant 1. Qwen positives are `58+59+59 = 176` of `180`, and `58+59 = 117` of `120` on the two bare arms. Observed agreement then equals chance agreement, and kappa is `0`. The seal records `cohens_kappa = 0.0` with `raw_agreement = 0.97778` on `180` rows and `0.975` on `120` rows. That is the constant-rater identity, and it matches. Judge bit-vectors are not stored, so this is a consistency proof from the sealed marginals, not a second grading pass.

Holm family, recomputed from the two stored p-values, matches the seal and rejects nothing:

| Contrast (Nemotron-30B judge) | delta | raw p | Holm p | reject |
|---|---|---|---|---|
| `Small_Chelated_minus_Bare_Llama70B_4bit_k5` | `-0.05` | `0.09645` | `0.1929` | false |
| `Mid_7B_Chelated_minus_Bare_Llama70B_4bit_k5` | `0.0` | `1.0` | `1.0` | false |

The four `descriptive_contrasts` are outside that family. On the Nemotron judge, `Chelated_70B_minus_Bare_70B_k5`, `Chelated_70B_minus_Bare_70B_k17`, and `Mid_7B_Chelated_minus_Bare_Llama70B_4bit_k17` are delta `0.0`, p `1.0`, interval `[0, 0]`. `Small_Chelated_minus_Bare_Llama70B_4bit_k17` is delta `-0.05`, p `0.09645`, interval `[-0.11667, 0.0]`. The one-case strict-exact gap between Bare 70B k5 (`58/60`) and Chelated 70B (`59/60`) is not the confirmatory endpoint.

Shared extractor cost: `nemotron_extract_mean_ms_per_case = 24607.08`, and `1476424.79 / 60 = 24607.08`. Chelated 70B reader mean `1989.39 ms` plus that extractor is the sealed `26596.47 ms` per case. Bare 70B k5 reader mean is `3257.82 ms` and Bare k17 is `5361.28 ms`, with no separate extractor line. End to end, the Chelated 70B path in this seal is the slow one.

`cache/osv_cve_seeds.json` is absent here. The N=60 suite was not rebuilt, because the builder fetches OSV when the cache is missing. Gold strings are not in `per_case_predictions`, so strict exact match was not re-graded from text.

### A.5 Re-sealed Demo D8

| Artifact | SHA-256 | Binding |
|---|---|---|
| `artifacts/wave11_d8_compliance_copilot/wave11_d8_copilot_manifest.json` | `7395d687c54f3711a0ebfbbbc0467e1c254128598a064e83fca58d469a7a154c` | Equals receipt `manifest_sha256` and `artifact_hashes` |

Receipt `git_sha` is `4b1520cd362a04b07ff6ef627ccc25604fccad08` (`fix(wave12): close D8 seal_receipts + --public NDA guard...`, 2026-11-25 15:35:42Z). Receipt time is `2026-11-25T15:35:52.650571+00:00`. That commit is an ancestor of HEAD. `git diff 4b1520cd --` the D8 runner, `demo_workbench/server.py`, `demo_workbench/compliance_copilot_engine.py`, `tests/test_wave11_d8_compliance_copilot.py`, and `tests/test_workbench_server.py` is empty.

The manifest says `all_checks_pass: true`, `checks_total: 38`, `checks_failed: []`. All 38 `pass` flags are true. The ledger-tamper check that used to die on `svc.ledger` assignment is in the seal: `tamper case: edited ledger entry breaks the chain at that seq` with `intact: false`, `first_bad_seq: 2`, `length: 3`. The runner now copies `svc.ledger`, writes the broken chain through `svc._ledgers["non_nda"]`, and restores that same list (`scripts/run_wave11_d8_compliance_copilot.py:221-227`). `ledger` is a read-only property returning `_ledgers["non_nda"]` (`demo_workbench/compliance_copilot_engine.py:1241-1243`).

`--public` hard-block is in `dispatch_api_request`: when `public_only` is set it drops `X-Copilot-Nda-Token` and returns `403` `mode_unavailable` if query or POST JSON `mode` is `nda` (`demo_workbench/server.py:609-631`). `main` also refuses `--public` together with `--nda-token` (`server.py:727-728`) and, for `--public` alone, installs `CopilotService(nda_token=None)`.

### A.6 CPU tests

Command, from `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1`, with `CUDA_VISIBLE_DEVICES=""`:

`python -m pytest tests/test_wave11_d10_scale_proof.py tests/test_wave11_d8_compliance_copilot.py tests/test_workbench_server.py -q`

Result: **85 passed in 50.17s**, no failures and no skips in the summary line.

The D8 seal regression is `tests/test_wave11_d8_compliance_copilot.py::TestHttpSmoke::test_runner_seal_receipts_reproducible`. It calls `seal_receipts` and requires every runner check, including the ledger edit, to pass. The `--public` regression is `tests/test_workbench_server.py::test_copilot_forwarder_and_public_only_isolation`. It configures `CopilotService(nda_token="unit-test-secret-token")`, observes `200` for `mode=nda` with that token when `public_only` is false, and observes `403` for the same token on both GET and POST when `public_only` is true.

---

## (b) Defects

1. **Stale sentence inside the D8 seal's own honesty block.** `brutal_honesty.stubs_L1` still says the `server.py` forwarder is "intentionally not wired by this slice." The same manifest's disclosure `L1` says the forwarder enforces `403` on `mode=nda` under `--public` even when a token is supplied, and the code at `4b1520cd` does that. The stubs sentence is false. It understates a feature that is present. It does not restore the old `AttributeError`, and the 38/38 seal plus the CPU test above still show the ledger path runs.
2. **Stale encoder comment.** `scripts/run_wave11_d10_5m_real_cutover_and_70b_scale.py:664` names `BAAI/bge-large-en-v1.5`. The encoder that actually runs uses `--upgrade-model-1024`, default and sealed value `BAAI/bge-m3` (`part_a_cutover_summary.json` field `upgrade_model_1024`). The sealed counts follow the code, not that comment.
3. **Two loss flags in Part A.** `statistically_significant_loss` is `ci_hi < 0` on the unadjusted percentile interval (`run_wave11_d10_5m_real_cutover_and_70b_scale.py:1924`). `statistically_significant_holm_005_loss` is the Holm rejection. On this seal they agree for every nDCG loss row (only Quora vs `RRF_k60` is true). A later corpus can split them. The whitepaper should cite the Holm flag.
4. **The CLI mutual exclusion is untested.** No test calls `main` with `--public` and `--nda-token` together. The behavior the Sol blocker named, `403` while a token is configured, is what `test_copilot_forwarder_and_public_only_isolation` passed. The sealed 38 checks exercise the copilot engine's own loopback server. They do not include a `demo_workbench` `--public` request. Coverage of that guard is the unit test, not the historical 38.

No hash mismatch, no Holm mismatch, and no failed test turned up. Items 1-4 are disclosure and comment defects. They are not a reason to reject the sealed numbers or to reopen the two Sol blockers.

---

## (c) Mandatory Brutal-Honesty disclosures for Whitepaper `v6.0`

### Demo D10 Part A

State the corpus as `8,841,823` MSMARCO plus `623,716` documents from the other six BEIR sets, equaling `9,465,539` real passages, `synthetic_vector_fraction = 0.0`, MSMARCO not subsampled. The 1024-d shard is `1,891,457` documents (`19.9826%`), the `sha256(...|upg20) % 5 == 0` slice across all seven domains, embedded with `BAAI/bge-m3`. The other `7,574,082` stay on `all-MiniLM-L6-v2` at 384-d. Do not write that the six BEIR corpora were fully upgraded, and do not quote the comment that names `bge-large-en-v1.5`.

State the index as `IndexHNSWSQ(QT_fp16, M=32, efConstruction=128, efSearch=512, METRIC_INNER_PRODUCT)`, sidecar `8` bytes per document (`72.2163 MiB` at 9.47M versus `36.1082 GiB` for a 1024-d float32 rewrite, ratio `512`). Stiefel anchors are `2,000` (`0.02113%` of 9.47M) with orthogonality error `2.44e-15`.

State the multiplicity result in full. Family size `24`, alpha `0.05`, Holm-Bonferroni. Ten contrasts reject, all with stored `p = 5e-05` (the `1/20000` bootstrap floor) and Holm p `0.00120`: the nine nDCG@10 wins in section A.3, plus one loss. The loss is `merged_9_47m_quora` versus `RRF_k60`, `-0.06919` nDCG@10, means `0.52647` versus `0.59567`, 95% interval `[-0.09138, -0.04750]`. Also state the five raw-`p < 0.05` contrasts that do not survive Holm, especially merged MSMARCO versus `RRF_k60` at `+0.01691`, Holm p `0.05530`. State the two non-rejected negative nDCG cells (FiQA vs `RRF` `-0.00123`; NFCorpus vs `DEV_ZScore` `-0.00079`).

State the six Recall@10 point losses as point losses. They were not multiplicity-tested. Call out ArguAna versus `RRF_k60` at `-0.030` Recall@10, and standalone MSMARCO versus `RRF_k60` as an nDCG Holm win (`+0.02235`) with a Recall@10 point loss (`-0.003`). Quora also loses Recall@10 versus `RRF_k60` by `-0.0296` (`0.70716` versus `0.73676`).

State the measurement scopes. `sampled_ann_vs_exact_flat_recall_at_10 = 0.984` is 50 SciFact queries against the 1024-d shard. `fenced_admitted_recall_at_10 = 0.944` is 25 MSMARCO queries on the dual shard. The fence retracted `200` documents (`0` fenced leaks versus `200` unmasked). That is a sparse retraction demo. `msmarco_standalone_8_84m` is the MSMARCO-only admission mask on the shared 9.47M graph, `n_queries = 500`, the same query count as `merged_9_47m_msmarco`.

### Demo D10 Part B

State `design_scope` verbatim: `reader_size_varied_under_fixed_nemotron30b_extractor_governance_pipeline`. The extractor and the governance pipeline stay the Nemotron-30B path. Only the reader changes. This is not a 70B-extractor result and not a claim that Chelated retrieval changed between arms.

State the model string verbatim: `unsloth/Llama-3.3-70B-Instruct-bnb-4bit`, `4-bit NF4 (bitsandbytes Linear4bit, bfloat16 compute)`, peak allocated `38.084 GiB`, load `245.22 s`. Suite A `N=60` on `prospective_realworld_drift_suite_n60`.

State the confirmatory family as two Holm tests, neither rejected: Chelated 1.5B minus Bare 70B k5 is `-5.0` percentage points on the Nemotron-30B judge (`p = 0.09645`, Holm `0.1929`, interval `[-0.11667, 0.0]`); Chelated 7B minus Bare 70B k5 is `0`. Do not describe a Holm-significant 70B Chelated win. The 70B-versus-70B contrasts are descriptive. On the Nemotron-30B judge, Chelated 70B, Bare 70B k5, and Bare 70B k17 are all accuracy `1.0` (delta `0`, p `1`). Strict exact match is `59/60`, `58/60`, and `59/60` respectively. That one-case exact-match gap is not the Holm endpoint.

State kappa with the raw agreement beside it. Inter-judge Cohen's kappa on the Llama-70B rows is `0.0` because the Nemotron-30B judge is constant-correct on those rows (`180/180` and `120/120`) while Qwen2.5-7B agrees on `176/180 = 0.97778` and `117/120 = 0.975`. Kappa `0` here is the prevalence identity. It is not a finding that the judges conflict, and it is not a finding that they are interchangeable beyond those rows.

State the cohorts. Chelated gate-decided cases are `14/60` (`CONTESTED_ABSTAIN` before the reader) at exact `13/14`. Reader-decided cases are `46/60`. Chelated 70B and Chelated 7B are exact `46/46` on the reader-decided slice. Chelated 1.5B is `42/46` there, with Nemotron-judge accuracy `0.93478`.

State the bare Nemotron-30B arm. It is in the sweep and outside the Holm family: strict exact `43/60 = 0.71667`, stale leak `16/60 = 0.26667`, false-commit rate on the abstention-required slice `1.0`, Qwen2.5-7B judge `0.73333`. Hiding it would make the 70B ceiling look like a retrieval result. The 30B reader is the weak arm on this fixture.

State latency on the same line as any accuracy. Chelated 70B reader mean `1,989.39 ms`, plus Nemotron extraction `24,607.08 ms`, equals `26,596.47 ms` per case. Bare 70B k5 is `3,257.82 ms` and k17 is `5,361.28 ms` without that extractor. Chelated 1.5B reader mean `147.81 ms` still sits on the same `24,607.08 ms` extractor. A reader-only latency comparison is a different quantity from end-to-end latency in this design.

### Demo D8, because the re-seal is part of this sign-off

Keep the floor-tier wording already in the seal. The demo is synthetic fixtures, hashed signed character 3-gram vectors (dimension 512), RRF at k=60, an in-memory runner key, and a loopback smoke. `38/38` is that seal, at manifest `7395d687c54f...` and git `4b1520cd`. Do not copy `stubs_L1`'s "forwarder is intentionally not wired." The forwarder is wired, and `--public` returns `403` for `mode=nda` even when an NDA token is configured. Quote disclosure `L1` from the manifest instead.

---

## (d) Verdict

**PASS_WITH_MANDATORY_DISCLOSURES**

D10's receipt, manifest, and both summary hashes match the files on disk. The 24-contrast Holm table, the Quora `-0.06919` loss, the six Recall@10 point losses, the footprint and bitset arithmetic, and the SciFact oracle-id digest all recompute. Part B's sealed Holm family does not reject, and its kappa of `0.0` is the constant-rater result implied by the sealed judge rates. Both D8 blockers from Sol 6.2 are fixed at `4b1520cd`, which matches the audited source, and the CPU pytest run of the three requested modules passed `85/85` in `50.17s`.

`v6.0` may cite these seals only with section (c) in the text. The sentence to withhold is any Holm-significant Chelated-70B advantage over Bare 70B, any reading of kappa `0` as judge disagreement, any claim that Recall@10 losses were multiplicity-corrected, and the D8 stubs sentence that says the public forwarder is unwired.

What this audit did not do: it did not recount the 9,465,539 passages from BeIR, it did not replay the 20,000-draw bootstraps from per-query rows, it did not re-grade Suite A from gold text, and it did not load `Llama-3.3-70B`.
