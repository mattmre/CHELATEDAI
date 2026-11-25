# Wave 12 Non-Author Cross-Model Validation Audit (`v6.0` Part VII: `D9`, `D10` Code & Part A, `verify_all_offline.py`)

- **`panel_type`:** `brutal-honesty-v3.3-non-author-xval`
- **`run_date`:** `2026-11-25`
- **`run_id`:** `wave12-xval-neon4-high`
- **`reviewer`:** `gemini-4-neon (High, non-author)`
- **`repo_head_audited`:** `6d4b41c9` (`D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1`)
- **`hard_safety_compliance`:** `D:\GITHUB\CHELATEDAI` untouched; zero edits to code/tests/receipts/manifests; zero reads of `*sealed_key*` or `*blind-followup*`; CPU-only verification (`CUDA_VISIBLE_DEVICES=""`).

**Verdict:** **`PASS_WITH_MANDATORY_DISCLOSURES`** *(for sealed Demo D9, sealed Demo D10 Part A + D10 code/tests, and `verify_all_offline.py --fast`; Demo D10 Part B remains `PENDING_SEAL` on Spark #2 and must undergo post-seal audit before `v6.0` final sign-off).*

---

## 1. Verified SHA-256 Digests & Independent Statistical Checks

### 1.1 Demo D9 (`artifacts/wave11_d9_fast_extractor_and_u_grpo/`)

| Artifact / Source | Independently Verified SHA-256 | Binding Check |
|---|---|---|
| `wave11_d9_extractor_grpo_manifest.json` | `c41fae097f76b4cf4514886626e39a04adf5c5c6430b369fabb3e099ecfd3f7a` | Exact match to `report_receipt.json` |
| `part_a_fast_extractor_results.json` | `0af8333739c9b15ab965bc03360d70a65b843c6cab008ebf18f261456277ba82` | Exact match to receipt & `manifest.part_a_fast_edge_extractor` |
| `part_b_u_grpo_results.json` | `39d67413a161a8379d06f9408a0e5be413ea76071bfd731b68b86be2957e3690` | Exact match to receipt & `manifest.part_b_virgin_holdout_u_grpo` |
| `selection_frozen.json` | `967dac7f24f1e2addf12b0e84f622253483c662facde4315a1a12aa76784c88c` | Exact match to receipt & manifest |
| `wave11_d9_part_a_suite_a_n60_rows.csv` | `1d43b0cba79f7e1debf59167c441fcfe4950101528f93a7df25a0071295704ab` | Exact match to receipt (`60` rows) |
| `wave11_d9_part_b_eval_virgin_v2_n120_rows.csv` | `299a4392d2674a113f8b38f803759293a926bb337d44f8b719bdebd27e2894ff` | Exact match to receipt (`840 = 7 x 120` rows) |
| `wave11_d9_part_b_eval_wave6_look2_n120_rows.csv` | `7931778ce3e88db89ca4932030983bc19bf4c3af4e4b5ba3de9a4949b7190377` | Exact match to receipt (`840 = 7 x 120` rows) |
| `docs/research/d9b_amendment_v1.json` | `fbb6fd446dc8ce71a63b9a43af0472f89f00c0418ad757a59a197a03b7e08508` | Exact match across disk, `selection_frozen.json`, and manifest |
| `scripts/run_wave11_d9_fast_extractor_and_u_grpo.py` | `e9891c1823956ede3a2b3b8936eb125cedb70374fd4e879967b5199609a034a2` | Matches `selection_frozen.json`; `prior_frozen_training_script_sha256` (`5f29f55b...`) verified byte-exact at `git 885fbedd~1` |

- **5-Way Split Disjointness (re-executed offline on CPU via `build_all_five_disjoint_splits()` + `assert_five_way_disjoint_splits()`):** `10/10` pairwise checks **PASS** across `SFT_WARMUP` (`N=128`, `32f62b33...`), `GRPO_RL` (`N=240`, `c5d2b1c3...`), `DEV` (`N=80`, `6de48c68...`), `EVAL_VIRGIN_V2` (`N=120`, `b605b4db...`), and `EVAL_WAVE6_LOOK2` (`N=120`, `6b08233c...`). Zero scenario overlaps (`25` unique scenarios), zero retractor overlaps (`5` distinct `noise-retractor-*` entities), `d1_entity_overlap = 0`, `d1_value_overlap = 0`, `virgin_v2_vs_base_slot_pool_overlap = 0`, `wave6_eval_sha256_matches_byte_identical = true`. All 5 recomputed digests match `selection_frozen.json` and `d9b_amendment_v1.json` byte for byte.
- **Zero Post-Hoc Checkpoint Snooping & Freeze Integrity:** `in_loop_holdout_evaluations_performed = 0`; `post_freeze_digest_verified = true`; freeze timestamp (`2026-11-25T13:51:38Z`) strictly precedes Stage-3 completion (`14:57:52Z`) and receipt (`14:57:53Z`). Independent replay of the pre-committed `DEV` selection rule (`det_em >= 0.6296 - 0.03 = 0.5996`, maximize `dev_exact_match`, tie-break `dev_mean_total_reward`) reproduces `seed_137 -> step 30` (`0.5750`, `0.1669`), `seed_2026 -> step 10` (`0.5625`, `0.1175`), and `seed_31415 -> step 50` (`0.5375`, `0.1138` > step 10 `0.0900`). All `best_dev` and `final_step` adapter SHA-256s match between `selection_frozen.json` and manifest, with `best != final` for all 3 seeds and `0` step-0 fallbacks. Git diff inspection of `885fbedd` + `f7a515db` confirms post-training edits fixed only the `paired_bootstrap_wave6` summary key (`"diff"` -> `"mean_delta"`, line 3905) and frozen-checkpoint resume binding (`prior_frozen_training_script_sha256`), without altering training hyperparameters, rewards, splits, or adapter weights.
- **Reload & Merge Parity:**
  - Part A (`0.5B` & `1.5B`): `reload_max_abs_logit_diff = 0.0`, `unmasked_prompt_token_count = 0`, `vs_base_max_abs_logit_shift = 12.34375` (`0.5B`).
  - Part B (`SFT` & 3 GRPO seeds): `sft_reload_max_abs_logit_diff = 0.0`, `reload_max_abs_logit_diff = 0.0`, `bf16_reload_max_abs_logit_diff = 0.0`, `kl_ref_disabled_adapter_vs_frozen_sft_max_abs_logit_diff = 0.0`, `active_grpo_adapter_vs_frozen_sft_max_abs_logit_shift = 2.17–2.50`.
  - **Spec-vs-Code Variance Found (`fp32 merge_and_unload()`):** Weight-space merge is exact (`max_weight_merge_abs_diff = 0.0 < 1e-6`, `argmax_match_rate = 1.0`, `196` LoRA modules), but 28-layer `fp32` forward logit diffs are `0.00257` (`seed_137`, rel `7.29e-5`), `0.00233` (`seed_2026`, rel `6.58e-5`), and `0.00418` (`seed_31415`, rel `1.19e-4`) — all `>= 1e-4` absolute. `run_wave11_d9_fast_extractor_and_u_grpo.py:2707-2713` gated on `max_weight_merge_abs_diff < 1e-6`, `max_rel_logit_diff < 5e-4` (docstring `:2619` says `< 2e-4`), `max_abs_logit_diff < 1e-2`, and `argmax_match_rate >= 0.98`, whereas `d9b_amendment_v1.json:100` states `"math.isfinite(diff) and diff < 1e-4"`. This reflects benign IEEE-754 `(W+BA)x` vs `Wx+B(Ax)` non-associativity across 28 layers, not an adapter bug, but must be disclosed explicitly.
- **Honest Statistical & Gate Audit (independently recomputed from raw CSVs):**
  - **Part A (`Suite A N=60`):** `semantic_accuracy = 0.9667` (`58/60`), `strict_exact_match = 0.9667`, `stale_leak_rate = 0.0333`, `json_valid_rate = 1.0` (`1,020/1,020`), beats `Mem0 Live infer=True` (`81.67%`) by `+15.00 pp` (`Holm p = 0.0003`, `lcb_975 = +0.0667`, `Tango lcb95 = +0.0808`, exact Clopper-Pearson `lcb95 = -0.0074`), ties Wave-6 30B (`96.67%`, `p = 1.0`). Mean latency = `2,104.01 ms/case` (`p50 = 2,095.48 ms`, `p95 = 2,369.73 ms`, `4.6x` faster than 30B's `9,688.87 ms`) — **honestly flagged** `ms_per_case_below_1500: false` (`latency_gate_not_met`) across manifest, `part_a_gate_diagnostics`, and `report_receipt.json`.
  - **Part B (`EVAL_VIRGIN_V2 N=120`, primary):** `SFT_7B = 0.5000` (`det=0.4625`, `u=0.5750`) vs `GRPO 3-seed final@60 = 0.4861 ± 0.0104` (`seed_137=0.4833`, `seed_2026=0.4750`, `seed_31415=0.5000`; `2-seed (137,2026) = 0.4792`). Recomputed `GRPO3 - SFT = -0.013889`, `CI95 = [-0.0750, +0.0472]` (manifest `p = 0.61825`, independent resample `p = 0.62605`, Holm `reject_null = False`). `GRPO3` beats `Bare_Nemotron_30B` (`0.2167`, `+26.94 pp`, Holm `p = 0.00015`) and `Base_7B` (`0.3500`, `+13.61 pp`, Holm `p = 0.0289`), both SFT-carried. Secondary `best_dev` mean = `0.4917` (`< SFT`). Secondary `EVAL_WAVE6_LOOK2 (N=120)`: `SFT = 0.6167`, `GRPO3 = 0.6056` (`-1.11 pp`, `p = 0.54945`), and **`determinate_non_inferiority_gate.passed = false`** (`0.8167` vs `0.8500`, `-3.33 pp` exceeding `-3.00 pp` margin). Verdict **`"K4 stands under Protocol v2"`** is verified and honest.

### 1.2 Demo D10 (`part_a_cutover_summary.json` + Code/Tests; Part B Pending on Spark #2)

- **Verified SHA-256s:**
  - `artifacts/wave11_d10_5m_real_cutover_and_70b_scale/part_a_cutover_summary.json`: `166a44a5ec5e6462ef788ec1237acf7cca6e52e33beba0419df71827d9c3a88b`
  - `scripts/run_wave11_d10_5m_real_cutover_and_70b_scale.py`: `37a7d871a56fc8becf1263db6b79ce967de9c2e554959ca0630840040413275a`
- **Scale, Storage, & Oracle Checks (100% exact):**
  - Corpus totals: `8,841,823` standalone MSMARCO + `623,716` 6-BEIR = `9,465,539` merged (`exact`); `legacy_384d = 7,574,082` + `upgraded_1024d (BAAI/bge-m3) = 1,891,457` (`19.9826%`) = `9,465,539` (`exact`); `synthetic_vector_fraction = 0.0`, `msmarco_subsampled = false`.
  - Exact binary footprints (`compute_exact_binary_footprint`): `72.2163 MiB` (`9.47M`) and `67.4578 MiB` (`8.84M`) sidecar (`8 B/doc`) vs `36.1082 GiB` / `33.7289 GiB` (`1024-d f32`, exact `512.0x`) and `144.4327 GiB` / `134.9155 GiB` (`4096-d f32`, exact `2,048.0x`).
  - Stiefel bridge: `2,000` anchors (`0.02113%` of `9.47M`), `stiefel_orthogonality_max_err = 2.4425e-15`.
  - FAISS & Memory: `IndexHNSWSQ(QT_fp16, M=32, efConstruction=128, efSearch=512, METRIC_INNER_PRODUCT)`, `pre_run_mem_available_gib = 55.32 >= 48.0`, `peak_rss_gib = 25.106 <= 44.0`, `dual_shard_query_p50/p95/p99 = 3.638/4.464/4.761 ms`.
  - Oracle & Retraction Fence: `sampled_oracle_qids_sha256` (`4b84ae3fe3853389...`) recomputes byte-exact from the 50 SciFact QIDs (`sampled_ann_vs_exact_flat_recall_at_10 = 0.984 >= 0.85`). Packed bitset = `ceil(7574082/8) + ceil(1891457/8) = 1,183,194` bytes (`exact`, `1.128 MiB`), `ek5_bitset_fence_compile_ms = 333.21 ms`, `0/200` (`0.0%`) fenced stale resurrection vs `200/200` (`100.0%`) unmasked on `200` retracted docs, `fill_at_10 = 1.0`, `fenced_admitted_recall_at_10 = 0.944 >= 0.85` on `n=25` MSMARCO queries.
- **24-Contrast Holm-Bonferroni & Loss Disclosure (`compute_part_a_multiplicity_and_loss_disclosure`):**
  - All `8` evaluation blocks' `means_ndcg_at_10` differences match `contrasts.*.mean_delta` within `< 2e-5`.
  - Running `compute_part_a_multiplicity_and_loss_disclosure(part_a)` yields **`24` contrasts** (`alpha = 0.05` Holm-Bonferroni):
    - **`9` significant wins (`holm_p = 0.00120`):** `merged_9_47m_fiqa` vs `DEV_ZScore` (`+0.01445`) & vs `Raw` (`+0.04282`); `merged_9_47m_msmarco` vs `Raw` (`+0.06042`); `merged_9_47m_quora` vs `DEV_ZScore` (`+0.09665`) & vs `Raw` (`+0.31169`); `merged_9_47m_scidocs` vs `RRF` (`+0.01355`); `merged_9_47m_scifact` vs `RRF` (`+0.02847`); `msmarco_standalone_8_84m` vs `RRF` (`+0.02235`) & vs `Raw` (`+0.04538`).
    - **`1` statistically significant loss (`holm_p = 0.00120`):** `merged_9_47m_quora::chelated_minus_rrf_k60 = -0.06919` (`CI95 = [-0.09138, -0.04750]`, `0.52647` vs `0.59567`).
    - **`12` non-significant positive:** includes 5 raw-`p < 0.05` contrasts that fail 24-contrast Holm (`merged_9_47m_msmarco` vs `RRF` `+0.01691` `raw_p=0.00395`/`holm_p=0.05530`, vs `DEV_ZScore` `+0.01164` `holm_p=0.09035`; `scidocs` vs `DEV_ZScore` `+0.00405` `holm_p=0.09035`, vs `Raw` `+0.00508` `holm_p=0.39750`; `nfcorpus` vs `RRF` `+0.00294` `holm_p=0.20845`), plus `arguana` (all 3), `scifact` (vs `DEV_ZScore`/`Raw`), `nfcorpus` (vs `Raw`), and `msmarco_standalone_8_84m` (vs `DEV_ZScore`).
    - **`2` non-significant negative nDCG@10 sub-cells:** `merged_9_47m_fiqa` vs `RRF` (`-0.00123`, `CI95=[-0.00941, +0.00674]`, `holm_p=1.0`) and `merged_9_47m_nfcorpus` vs `DEV_ZScore` (`-0.00079`, `CI95=[-0.00269, +0.00111]`, `holm_p=1.0`).
    - **`6` Recall@10 point losses surfaced (`9` total disclosed loss entries):** ArguAna vs `RRF` (`-0.0300`), Quora vs `RRF` (`-0.0296`), Standalone MSMARCO vs `RRF` (`-0.0030`), SciFact vs `RRF` (`-0.00277`), NFCorpus vs `DEV_ZScore` (`-0.00062`) & vs `Raw` (`-0.00031`).

### 1.3 Offline Verifier (`scripts/verify_all_offline.py`) & Pytest Execution

- `scripts/verify_all_offline.py` SHA-256: `65916c27aff1bdcf81ea2aafbc6793844a7d09032d90bc9f866bdbf06573dd05`.
- `CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_wave11_d10_scale_proof.py tests/test_verify_all_offline.py -q`: **`62 passed in 26.39s`** (`0` failures, `0` skips).
- `CUDA_VISIBLE_DEVICES="" python scripts/verify_all_offline.py --fast`: **`PASS` (`exit_code = 0`) in `11.9s`** (`cpu = 0.25s`, `< 45s` budget), reporting `formal_tier=SKIPPED_USE_FULL_FORMAL`, `40/40` spec bundles + `4` standalone Part VII receipts (`D6`, `D8`, `D7`, `D9`), `D8` Ed25519 + hash chain + live replay (`3` receipts, head `820705ef47fc`) PASS, `0/56` Non-NDA hits on all 3 targets, `metrics.d6` & `metrics.d8` PASS, `registry.content_sha256` (`6e6e9a156f38`) PASS, `release_status.honesty` PASS.

---

## 2. Code & Verifier Gaps to Close in Wave 13

1. **Missing Headline-Metric Gates for `D7`, `D9`, and `D10` in `scripts/verify_all_offline.py:718-740`:**
   `verify_headline_metrics()` currently checks only `D6` and `D8`. `D7`, `D9`, and `D10` get only generic SHA-256 receipt checks via `verify_part7()`. Add deterministic headline checkers in Wave 13:
   - `metrics.d7`: verify `primary_score == "settled_accuracy"`, `suite_b_pooled is False`, and schema-validation/truncation fields present.
   - `metrics.d9`: verify `part_a_gate_diagnostics.part_a_ms_per_case_below_1500 is False`, `part_b_hypothesis_summary.verdict == "K4 stands under Protocol v2"`, `in_loop_holdout_evaluations_performed == 0`, and cross-check `amendment_sha256` against `docs/research/d9b_amendment_v1.json`.
   - `metrics.d10`: verify `synthetic_vector_fraction == 0.0`, `msmarco_subsampled is False`, `chelated_fenced_stale_resurrection_rate == 0.0`, `fenced_admitted_recall_at_10 >= 0.85`, `sampled_ann_vs_exact_flat_recall_at_10 >= 0.85`, `n_contrasts_in_family == 24`, `part_b.status in ("COMPLETED", "NOT_RUN")`, and `input_part_a_sha256`/`input_part_b_sha256` binding.
2. **Part VII Bundles Still Optional / Unregistered in `bundle_registry_spec.json` (`40` vs `45`):**
   `verify_all_offline.py:98` keeps `wave11_d7_`, `wave11_d9_`, and `wave11_d10_` in `PART7_OPTIONAL_PREFIXES` (silently skipped if `report_receipt.json` is absent), and `bundle_registry_spec.json` / `verified_registry.json` still list only `40` bundles. At Wave 13 `v6.0` seal, register `D6`–`D10` (`45/45` bundles) with `part_vii_code_provenance`, rebuild `verified_registry.json`, and promote `D7`/`D9`/`D10` to required so missing receipts fail closed.
3. **Zero Unit Test Coverage & Raw-vs-Holm Flag Mismatch in `compute_part_a_multiplicity_and_loss_disclosure` (`run_wave11_d10_5m_real_cutover_and_70b_scale.py:1903-1945`):**
   `tests/test_wave11_d10_scale_proof.py` never exercises `compute_part_a_multiplicity_and_loss_disclosure` (its `assemble-manifest` test passes `{"wall_seconds": 1.25}`, leaving `n_contrasts_in_family == 0`). Also, line `1924` sets `statistically_significant_loss = bool(ci_hi < 0.0)` from the unadjusted 95% CI rather than Holm-adjusted rejection (`holm[key]["reject_null_at_alpha"] and delta < 0`); both agree on current Part A data (only Quora vs RRF), so no number is distorted, but Wave 13 should log both flags and add a unit test covering wins, losses, Recall@10 losses, and Holm step-down.
4. **Stale Docstring / Inline Comments in `run_wave11_d10_5m_real_cutover_and_70b_scale.py`:**
   Line `17` states `M=16` whereas actual execution (`:1006`, `:1009`) and sealed Part A use `M=32, efConstruction=128, efSearch=512`; line `664` comment says `BAAI/bge-large-en-v1.5` whereas `--upgrade-model-1024` (`:1871`) and sealed Part A (`:592`) use `BAAI/bge-m3`. Clean up after D10 seals (recording provenance SHA) to avoid reader confusion.
5. **D9 Parity Docstring/Amendment Alignment:**
   Do **not** mutate frozen `d9b_amendment_v1.json`, but reconcile the internal docstring discrepancy in `run_wave11_d9_fast_extractor_and_u_grpo.py` (`:2619` says `max_rel_logit_diff < 2e-4` vs `:2710` enforcing `< 5e-4`) in Wave 13 provenance documentation and note the `d9b_amendment_v1.json:100` (`diff < 1e-4` absolute) variance explicitly.
6. **Minor Observation (`verify_all_offline.py:559-563`):**
   `d8.sealed_file_hashes` checks `ek5_v3_stratified.py` under the `demo_workbench/compliance_copilot_engine.py` key (documented legacy key binding); `compliance_copilot_engine.py` integrity relies on `d8.live_replay`, so Wave 13 CI/release runs must never pass `--no-d8-replay`.

---

## 3. Mandatory Brutal-Honesty Scope Disclosures Required in Whitepaper `v6.0`

1. **D9 Part A (`FastEdge 0.5B`) Latency Miss & Small-N CIs:**
   - State verbatim: *"Latency gate not met (`2,104.0 ms/case` mean, `p50=2,095.5 ms`, `p95=2,369.7 ms` vs `1,500 ms` target; `4.6x` faster than Wave-6 `30B`'s `9,688.9 ms/case`)."*
   - Report `Suite A (N=60)` `96.67%` accuracy (`100.0%` JSON validity on `1,020/1,020` docs) beside `Mem0 Live infer=True` (`81.67%`, `+15.00 pp`, Holm `p=0.0003`, bootstrap `LCB97.5=+6.67 pp`, Tango `LCB95=+8.08 pp`, conservative exact Clopper-Pearson `LCB95=-0.74 pp`).
2. **D9 Part B (`Protocol v2` U-GRPO) Honest Null & Provenance:**
   - State primary verdict verbatim: **`"K4 stands under Protocol v2"`** (`EVAL_VIRGIN_V2 N=120`: `final_step@60` 3-seed mean `48.61% ± 1.04%`, 2-seed `47.92%` vs `SFT_7B = 50.00%`, `delta = -1.39 pp`, `CI95 = [-7.50 pp, +4.72 pp]`, Holm `p = 0.61825`; secondary `best_dev = 49.17%`). Explicitly attribute gains over `Base_7B` (`+13.61 pp`) and `Bare_Nemotron_30B` (`+26.94 pp`) to **SFT**, not GRPO.
   - Disclose `EVAL_WAVE6_LOOK2` secondary determinate non-inferiority gate failure (`81.67%` vs `85.00%`, `-3.33 pp` vs `-3.00 pp` margin).
   - Disclose `eval_virgin_v2_scope = "virgin_for_grpo_arm_only"` (Base/SFT scored in R0 Stage 1), `seed_42` retired as DEV-diagnostic only, `protocol_v2_single_knob_attribution_claimed = false` (7 bundled fixes; any retune requires `EVAL_VIRGIN_V3`), the post-eval summary-key resume patch (`5f29f55b...` -> `e9891c18...`, zero weight/hyperparameter changes), and the `fp32 merge_and_unload()` variance (`0.0` weight merge error, `1.0` argmax match, `2.33e-3..4.18e-3` abs / `6.58e-5..1.19e-4` rel 28-layer GEMM logit diff vs `< 1e-4` abs in `d9b_amendment_v1.json`).
3. **D10 Part A (`8.84M`/`9.47M` Cutover) Multiplicity, Losses, & Methodological Boundaries:**
   - Report the full 24-contrast Holm-Bonferroni (`alpha=0.05`) split: `9` significant wins, **`1` significant loss (`merged_9_47m_quora` vs `RRF_k60`: `-0.06919` nDCG@10, `0.52647` vs `0.59567`, `CI95=[-0.09138, -0.04750]`, `holm_p=0.00120`)**, `12` non-significant positive (explicitly noting `merged_9_47m_msmarco` vs `RRF_k60` `+0.01691` `holm_p=0.05530` and vs `DEV_ZScore` `+0.01164` `holm_p=0.09035` do not clear 24-contrast Holm), and `2` non-significant negative nDCG@10 cells (`FiQA` vs `RRF` `-0.00123`; `NFCorpus` vs `DEV_ZScore` `-0.00079`), plus all `6` Recall@10 point losses (including ArguAna `-0.0300` and Quora `-0.0296` vs `RRF_k60`).
   - Disclose three exact measurement scopes: (i) `msmarco_standalone_8_84m` is bitset-isolated on the shared `9.47M` `IndexHNSWSQ(QT_fp16, M=32, efSearch=512)` graph (and shares its 500 MSMARCO queries with `merged_9_47m_msmarco`); (ii) `sampled_ann_vs_exact_flat_recall_at_10 = 0.984` (`n=50` SciFact) audits the `1.89M` `1024-d` native shard, while `fenced_admitted_recall_at_10 = 0.944` (`n=25` MSMARCO) audits the full `9.47M` dual-shard ZeroMu+CSLS oracle; (iii) `retraction_fence_9_47m_proof` retracts `200` docs (`0.0021%` density; high-density `10%/50%/90%` retraction is evidenced in `D6`, not `D10`).
   - Do **not** cite `D10 Part B` (`70B`) results in Whitepaper `v6.0` until Spark #2 seals `part_b_70b_sweep_summary.json` and `report_receipt.json`; once sealed, retain reader-scale-only framing under the fixed `Nemotron-30B` extractor + `4-bit NF4` disclosure (or omit `70B` claims if `NOT_RUN`).
4. **Release Gate & Fast-Tier Scope:**
   - Retain `PRE-RELEASE / HUMAN AUDIT & CLEAN-RERUN GATES PENDING` until independent human grading of `PM-4b` completes, and state explicitly that `v6.0` offline verification runs in `--fast` mode (`formal_tier = "SKIPPED_USE_FULL_FORMAL"`, Lean 4 / `nanoda_bin` / EK5 conformance deferred to `v7.0`).

---

## 4. Final Verdict

**`PASS_WITH_MANDATORY_DISCLOSURES`**

All audited SHA-256 bindings (`D9` manifest `c41fae097f76...`, 7 receipt sidecars, `d9b_amendment_v1.json`, `selection_frozen.json`, 5-way split digests, `D10` Part A `166a44a5ec5e...` + binary footprints + bitset + oracle QID digest), all `62/62` offline unit tests, and `verify_all_offline.py --fast` (`11.9s`, `rc=0`) pass cleanly. Both `D9` Part A's latency gate miss (`2,104 ms > 1,500 ms`) and `D9` Part B's null result (`K4 stands under Protocol v2`) are reported honestly without snooping, and `D10` Part A's 24-contrast Holm & loss disclosure surfaces the Quora vs RRF loss (`-0.06919`) and non-significant cells faithfully. Promotion to `v6.0` seal requires (1) carrying Section 3's mandatory disclosures verbatim into Whitepaper `v6.0`, (2) auditing `D10` Part B once Spark #2 seals `wave11_d10_scale_manifest.json`, and (3) closing Section 2's verifier/registry/test gaps in Wave 13.
