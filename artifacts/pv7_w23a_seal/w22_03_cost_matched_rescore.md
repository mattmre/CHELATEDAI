# W22-03 — Cost-matched re-score of four operational claims (#41 D6, #45 D10-B, #46/#53 D11, #62 D13)

- Mode: read-only re-score of stored artifacts. No models were run, no git state was changed, and `scripts/verify_all_offline.py` was not invoked.
- Date: 2026-11-26.
- Kill criterion applied: any deployment-advantage claim is SUSPENDED where raw cosine, linear routing, bare 70B, full restart, or another conventional arm ties or beats Chelated at matched cost, latency and coverage.
- Artifact paths below are relative to `D:\GITHUB\CHELATEDAI-PV7-SPINE\artifacts\`.

| # | Item | Verdict |
|---|---|---|
| 41 | D6 exact-fallback serving | **SUSPEND** (deployment and latency claim) |
| 45 | D10-B chelated vs bare 70B | **SUSPEND** |
| 46 / 53 | D11 PM-6 arms, plus RCPS closure | **SUSPEND** |
| 62 | D13 pooled agent loops | **SUSPEND** |

---

## #41 D6 — chelated certified path vs "~940 ms exact fallback"

**Claim.** The chelated served path (exact-fallback certificate) is a deployable advantage over a conventional backend, framed against a "~940 ms exact fallback".

**Conventional baselines.** Two:
- The same backend's initial ANN shortlist, with no certificate.
- The uncalibrated raw-cosine merge, DEV z-score merge and RRF merge on the same shortlist.

**Artifact.** `wave11_d6_opensearch_pgvector_qdrant/wave11_d6_multibackend_manifest.json` (sha256 prefix 0578d56b).
Key pattern: `benchmarks.{FiQA2018|SciFact}.{1024|4096}.cases.{engine}.directions.{dir}.retraction_percent.{0|10|50|90}`.

**Matched numbers.**

1. **The 940 ms figure is the chelated served path itself, not a baseline.**
   - Cell: `FiQA2018.4096.opensearch.d384_to_d4096.0`.
   - `service_rtt.p95_ms` = 940.02 (p50 125).
   - `served_routes` = opensearch_knn_score_exact.
   - `exact_fallback_fraction_of_shard_queries` = 1.0.
   - The conventional ANN shortlist for the same cell (`initial_backend_requests_rtt_sum`) has p50/p95 of 25/31 ms. That is about 30× faster at p95.
   - The only other source of "940 ms" is `docs/research/council-logs/post_v7_panel_r1_fable52_medium.md:21`, which marks it unverified.
2. **Coverage.**
   - Fallback fraction is 1.0 in 62 of 70 cells.
   - The other 8 are `pgvector_exact_vector4096` cells, at 0.845 and 0.934. They are already exact.
   - So every served path is an exact scan, which means the certificate never uses the fast path.
3. **Latency across the 56 cells that have both RTTs.**
   - Median served p50/p95: 32.8/52.2 ms. Median initial shortlist: 14.5/20.5 ms.
   - Ratio of served p95 to shortlist p95: median 2.7×, minimum 1.7×, maximum 50.9×.
   - The 14 cells at 90% retraction have no initial-RTT key, so I don't know their latency.
4. **Quality at matched shortlist.**
   - Served `quality.ndcg_at_10` equals `initial_shortlist_quality.ndcg_at_10` in 22 of 28 cells at 0% retraction.
   - The largest served-minus-shortlist gain is +0.0022 nDCG (recall +0.0053, qdrant FiQA 4096 d4096→d384).
   - So the certificate buys essentially no nDCG for 1.7–50.9× the p95 latency.
5. **Merge quality vs conventional merges on the same shortlist** (`...paired_bootstrap_wave6`, Chelated minus baseline).
   - Vs raw cosine (`Uncalibrated_Raw_Merge`), nDCG wins in all 7 direction groups. Examples:
     - FiQA 4096 d384→d4096: +0.0382 (LCB95 0.0217).
     - SciFact 4096 d4096→d384: +0.2765.
   - Raw cosine ties on recall in one group: FiQA 1024 d384→d1024, +0.023, LCB −0.0004, p = 0.105.
   - DEV z-score beats chelated on nDCG in SciFact 4096 d384→d4096: −0.0258, p = 0.053.
   - RRF and DEV tie on recall in several groups, e.g. FiQA 4096 d384→d4096 recall: vs RRF −0.0103 (p = 0.535), vs DEV −0.0011.
6. **Stale resurrection.**
   - `stale_resurrection_rate` = 0.0 for chelated.
   - `raw_engine_stale_hit_rate` = 0.0, so the raw allowlist baseline ties chelated.
   - The only baseline that leaks is `lagged_replica_only_stale_hit_rate`, at 5.2–8.2%.

**Verdict: SUSPEND (deployment/latency advantage).**
- At matched coverage, the conventional ANN shortlist serves the same nDCG (22 of 28 cells identical, maximum delta +0.0022) at 1.7–50.9× lower p95.
- The raw allowlist ties on stale hits.
- Only a narrow claim survives: merge-quality over uncalibrated raw cosine, at shortlist-matched cost. Two limits apply to that:
  - Baseline-merge latency was never timed separately, because all merges share one shortlist.
  - There are no per-query rows and no replicate runs.
- I don't know the latency for the 90%-retraction cells.

---

## #45 D10-B — Bare 70B vs Chelated 70B

**Claim.** The chelated pipeline (Nemotron-30B extractor plus reader) is worth deploying over a bare 70B reader.

**Conventional baseline.** Bare-70B at k = 5 and at k = 17.

**Artifact.** `wave11_d10_5m_real_cutover_and_70b_scale/part_b_70b_sweep_summary.json`.
Keys used: `arm_metrics`, `small_and_mid_reader_telemetry`, `per_case_predictions` (60 cases × 7 arms), `paired_contrasts_nemotron_blind_judge`, `holm_bonferroni_family`.

**Matched numbers (N = 60).**

| Arm | Strict EM | Judge | Stale leak | False commit | Mean e2e latency |
|---|---|---|---|---|---|
| Bare-70B k5 | 0.96667 | 1.000 | 0 | 0 | 3,257.82 ms |
| Bare-70B k17 | 0.98333 | 1.000 | 0 | 0 | 5,361.28 ms |
| Chelated-70B | 0.98333 | 1.000 | 0 | 0 | 26,596.47 ms (reader 1,989.39 + `nemotron_extract_mean_ms_per_case` 24,607.08) |
| Mid-7B + Chelated | 0.98333 | 1.000 | 0 | 0 | 25,145.41 ms (reader 538.33, p95 1,021.17) |
| Small-1.5B + Chelated | 0.91667 | 0.95 | 0.05 | 0.1333 | 24,754.89 ms |

- **Contrasts.**
  - `Chelated_70B_minus_Bare_70B_k5` and `Chelated_70B_minus_Bare_70B_k17`: `mean_delta` = 0.0, p = 1.0, CI ±0.0602.
  - Mid-7B vs Bare-70B k5: Holm-adjusted p = 1.0.
- **Per-case agreement.**
  - Chelated-70B agrees with Bare k17 on 58 of 60 cases and with Bare k5 on 57 of 60.
  - The disagreements:
    - iam_token_forgery_505: bare abstains.
    - cobalt_aero_dod: chelated abstains.
    - One formatting difference on k5 ('REDOSONLY').
- **Memory.**
  - Bare-70B: 38.084 GiB. Mid-7B reader: 14.279 GiB.
  - The Nemotron-30B extractor's memory is not stored, so I don't know the end-to-end VRAM. VRAM cannot be matched.
- **Faster extractor.** The 0.5B extractor in `wave11_d9_fast_extractor_and_u_grpo/part_a_fast_extractor_results.json` (key `suite_a_evaluation_n60.extraction_benchmark.ms_per_case_mean` = 2,104 ms, det acc 0.9667) is a different run. It is not composable with D10-B, and no stored run pairs it with a reader.

**Verdict: SUSPEND.**
- Bare-70B k17 ties Chelated-70B exactly on EM, judge score, stale leak and false commit. It does so at 5.36 s vs 26.60 s, about 5× faster.
- Bare k5 ties on the judge score at 3.26 s, about 8× faster.
- Only latency means are stored (no per-case latency), so I don't know the tail latencies.

---

## #46 D11 (PM-6 nine-arm) + #53 PM-6 RCPS closure

**Claim.** The chelated full dual gate (Arm 9) is the right deployable suppressor of withdrawn content (R3) at a controlled benign false-refusal rate (R5).

**Conventional baselines.**
- Arm 6: Gate-1 + Gate-2 raw cosine.
- Arm 8: linear bridge (CAR) + Gate-2.
- Arm 4: Gate-1-only EK5 prefix.

### Operating points (stored)

Artifact: `wave14_d11_pm6_takedown_carryforward/pm6_posthoc_statistical_reanalysis.json`.
Key: `full_9arm_comparison.by_generator.{nemotron_30b|qwen25_7b}.arms`.

Arm-level metrics are cross-checked against `wave14_d11_pm6_takedown_carryforward/wave14_d11_pm6_manifest.json` → `nine_arm_matrix_by_generator.{gen}.{arm}.{r3_en_suppression_rate, r5_benign_fpr, r3_vs_r5_roc}`.

| Generator | Arm | R3 TPR | AUROC | R5 FPR (false refusals / N) | CP_UB95 |
|---|---|---|---|---|---|
| Nemotron | 4 EK5 prefix | 0.3125 | 0.500 | 0 / 32 | 0.0894 |
| Nemotron | 6 raw cosine | **1.000** | **1.000** | 1 / 32 = 0.03125 | 0.1398 |
| Nemotron | 8 linear bridge | 0.8125 | 0.9932 | **0 / 32** | 0.0894 |
| Nemotron | 9 chelated | 0.9375 | 0.9893 | 2 / 32 = 0.0625 | 0.1839 |
| Qwen | 4 | 0.1765 | 0.500 | 0 / 17 | 0.1616 |
| Qwen | 6 raw cosine | **1.000** | **0.9965** | 1 / 17 = 0.0588 | 0.2501 |
| Qwen | 8 linear bridge | 0.8824 | 0.9896 | **0 / 17** | 0.1616 |
| Qwen | 9 chelated | 0.8824 | 0.9758 | 2 / 17 = 0.1176 | 0.3262 |

### Matched-FPR TPR

I computed these from the stored ROC arrays at `nine_arm_matrix_by_generator.{gen}.{arm}.r3_vs_r5_roc.{fpr,tpr}`. The value is the maximum TPR at FPR ≤ x.

| Generator | FPR ≤ | Arm 4 | Arm 6 raw cos | Arm 8 linear | Arm 9 chelated |
|---|---|---|---|---|---|
| Nemotron (N = 32/32) | 0 | 0.000 | **1.000** | 0.9375 | 0.9062 |
| Nemotron | 1/32 | 0.000 | **1.000** | 0.9375 | 0.9375 |
| Nemotron | 2/32 | 0.000 | **1.000** | 0.9688 | 0.9375 |
| Qwen (N = 17/17) | 0 | 0.000 | **0.9412** | **0.9412** | 0.8235 |
| Qwen | ≤ 1/17 | 0.000 | **1.000** | 0.9412 | 0.8824 |

**Holm tests** (`pm6_posthoc_statistical_reanalysis.json`):
- Arm 9 minus Arm 6 AUROC: −0.0107 (Nemotron), −0.0208 (Qwen); adjusted p = 1.0.
- Arm 9 minus Arm 8: −0.0039 and −0.0138; adjusted p = 1.0.
- The only comparisons that survive Holm are vs Arm 3: +0.4736 and +0.4758.
- `min_n_to_certify` at 0 errors = 149 benign families. Both generators are far below this (32 and 17).

### #53 RCPS closure

Artifact: `pv7_colab_l4_closure_20261125/colab_l4_closure_manifest.json`.
Key: `track_a_pm6_benign_fpr_calibration_closure.arms.{arm}`.
- Evaluation set: N_neg = 599, N_pos = 236.
- Calibration set: 594 negatives; target FPR 0.02; delta 0.05.

| Arm | eval FPR (fp / 599) | `eval_r5_benign_fpr_cp_ub95` | `eval_r3_suppression_tpr` | AUROC |
|---|---|---|---|---|
| arm_c_rcps_conformal_zeromu_raw_cosine | **0.005008 (3)** | **0.012893** | 0.182203 (43/236) | 0.823279 |
| arm_b_rcps_conformal_zeromu_csls | 0.006678 (4) | 0.015216 | 0.127119 (30/236) | 0.815335 |
| arm_d_rcps_conformal_procrustes_csls | 0.008347 (5) | 0.01747 | 0.118644 (28/236) | 0.752257 |
| arm_e_rcps_conformal_native_4096_oracle | 0.010017 (6) | 0.019674 | 0.317797 (75/236) | 0.8842 |
| arm_a_d11_naive_n50_zeromu_csls (fails) | 0.035058 (21) | 0.050094 | 0.334746 | 0.815335 |

- Under identical RCPS calibration, raw cosine (arm_c) is at least as good as the CSLS arm (arm_b) on every metric: lower FPR, lower UB, higher TPR and higher AUROC.
- The headline "FPR 0.50% / CP_UB95 1.29%" belongs to the raw-cosine arm.
- At the matched RCPS operating point, suppression is only 18.2% TPR. Coverage is therefore low for every arm.
- Scope: offline calibration on Colab L4. The live gateway was not executed, so I don't know live FPR or latency.

**Verdict: SUSPEND.**
- At matched FPR, raw cosine (Arm 6) has equal or higher TPR than Arm 9 at every FPR level in both generators.
- Linear routing (Arm 8) matches or beats Arm 9 at FPR = 0.
- Arm 9 carries the highest benign FPR and CP_UB95 of the four arms.
- In #53, RawCos+RCPS dominates CSLS+RCPS.
- None of these differences reaches significance at N = 32/17. A tie at that sample size is still a kill under the criterion.

---

## #62 D13 — pooled agent loops: Spark #3 replay-fail vs Wave 22 Colab L4

**Claim.** ZeroMu + EK5_v3 (chelated incremental re-ingest) delivers more on-time, cited, correct jobs per minute (JPM) than conventional or full-restart handling under encoder change plus withdrawal.

**Conventional baselines.** All within the same run:
- Conv+SQLite.
- Conv+EK5_v3.
- ZeroMu+SQLite.
- full_restart_rebuild.

### Spark #3

Artifacts: `pv7_w20_d13_pooled_agent_loops/pv7_w20_d13_manifest.json` and `report_receipt.json` (status replay-fail).
- Model `nemotron-lightning`; `live_model_calls` 7,842; `total_eligible_jobs_executed` 1,680.
- `interaction_by_pool_size.4.arm_summary.*.on_time_cited_correct_jobs` = 0 of 48 for every arm, including ZeroMu+EK5_v3.
- Every arm's `pooled_on_time_cited_correct_jpm` = 0.0.
- `full_restart_comparator.jpm_advantage_over_full_restart` = 0.0, and `gates.beats_full_restart_rebuild_jpm` = false.
- `sham_reopen_isolation` mean latency (ms):
  - `no_change_mean_latency_ms` 3,914.2
  - `withdrawal_only_mean_latency_ms` 5,406.1
  - `sham_reopen_mean_latency_ms` 7,852.6
- 6 of 8 gates pass; overall_pass is false.

### Wave 22 Colab L4

Artifacts: `pv7_w22b_b62_d13_colab_l4/colab_l4_provenance.json`, `pv7_w20_d13_manifest.json` and `report_receipt.json` (status replay-fail, `runner_exit_code` 1).
- Model `Qwen/Qwen2.5-0.5B-Instruct`; `live_model_calls` 7,866; NVIDIA L4.
- Arm summary at `primary_2x2_interaction.arm_summary` (pool 4, regime "both", 48 eligible jobs each):

| Arm | on-time cited-correct jobs | rate | pooled JPM | stale commits |
|---|---|---|---|---|
| Conv+EK5_v3 | 7 | 0.1458 | **62.969** | 0 |
| Conv+SQLite (fully conventional) | 7 | 0.1458 | **62.866** | 0 |
| ZeroMu+EK5_v3 (chelated) | 8 | 0.1667 | 59.382 | 0 |
| ZeroMu+SQLite | 8 | 0.1667 | 59.037 | 0 |
| full_restart_rebuild (`full_restart_comparator`) | 0 (48 late or timed out) | 0 | 0.0 | 0 |

Effects (`primary_2x2_interaction`):
- `simple_main_effects_pooled_jpm.zeromu_gain_with_ek5` = **−3.5875 JPM**.
- `zeromu_gain_with_sqlite` = −3.8297.
- `ek5_gain_with_zeromu` = +0.3448.
- JPM interaction mean +0.6779:
  - per-job bootstrap LCB95 +0.2586.
  - **block bootstrap LCB95 0.0**: only 2 matched blocks, so it is not strictly positive.
- Rate interaction: `rate_interaction_matter_cluster_bootstrap.mean_delta_rate` = 0.0, LCB 0.0, UCB 0.0.

Sham control (`sham_reopen_isolation`, `control_valid` = false):
- `sham_reopen_mean_latency_ms` = 2,141.713.
- `sham_on_time_cited_correct_jobs` = 0 of 48.
- `no_change_mean_latency_ms` = 697.3.
- `isolated_checkpoint_reopen_overhead_ms` = 1,444.4.
- `net_ek5_lineage_invalidation_delta_ms` = −698.9.

### Reconciliation

- The two runs are not cross-matched. They used different hardware (DGX Spark aarch64 vs Colab L4 x86) and different served models (nemotron-lightning vs Qwen2.5-0.5B), at mean latencies 3.9–7.9 s vs 0.7–2.1 s.
- The Spark run fails because every arm, conventional and chelated alike, is slower than the SLA. It is uninformative about chelation, not evidence against it.
- The Colab run is the only informative one:
  - The chelated arm beats full restart (59.38 vs 0.00 JPM). Full restart is the weakest conventional baseline.
  - The chelated arm loses to the fully conventional Conv+SQLite arm (59.38 vs 62.87 JPM, −3.48) and to Conv+EK5_v3 (62.97).
  - Its on-time rate advantage is 1 job in 48 (8 vs 7), and the rate interaction is exactly 0.0.
- The "59.38 vs 0.00" headline compares against the strawman baseline only.
- The sham control is invalid: a sham reopen alone costs about 1.44 s and pushes every sham job past the SLA. So the run also cannot attribute latency to EK5 lineage invalidation vs checkpoint reopen.
- The 560 ms SLA value was not found as a JSON key in these three artifacts. I used it as given in the task brief; its exact definition is in the runner script, which I did not inspect.

**Verdict: SUSPEND.**
- A conventional arm (Conv+SQLite) beats Chelated on JPM in the same run, at matched hardware, model, jobs and SLA.
- The full-restart win is real but is against the weakest baseline.
- Neither run passes overall, and both are status replay-fail.

---

## What is missing (I don't know)

- **D6:** latency for the 90%-retraction cells, per-query rows, separately timed baseline merges, replicate runs.
- **D10-B:** extractor VRAM, per-case latency and tails, and any composed run of the fast 0.5B extractor with a reader.
- **D11:** any result at N ≥ 149 benign families, and live-gateway FPR and latency for #53.
- **D13:** the UncCSLS+SQLite and UncCSLS+EK5_v3 cells are listed in `system_cells`, but no arm_summary entry for them was found under `primary_2x2_interaction`. There is also no same-hardware, same-model replicate of the Colab run.
