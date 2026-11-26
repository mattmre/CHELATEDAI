# W22-03 — Cost-matched re-score of four operational claims (#41 D6, #45 D10-B, #46/#53 D11, #62 D13)

- Mode: read-only re-score of stored artifacts. No models were run, no git state was changed, and `scripts/verify_all_offline.py` was not invoked.
- Date: 2026-11-26.
- Kill criterion applied: any deployment-advantage claim is SUSPENDED where raw cosine, linear routing, bare 70B, full restart, or another conventional arm ties or beats Chelated at matched cost, latency and coverage.
- Artifact paths below are relative to `D:\GITHUB\CHELATEDAI-PV7-SPINE\artifacts\`.

| # | Item | Verdict | Already disclosed in the scorecard? |
|---|---|---|---|
| 41 | D6 exact-fallback serving | **PARTIAL.** The merge-quality claim survives. A latency or deployment advantage is not shown. | Mostly. The scorecard already says "70/70 exact scans ... no pure ANN latency claim". |
| 45 | D10-B chelated vs bare 70B | **SUSPEND** any deployment-advantage reading | Yes. The 0.00 pp tie and 3.26 s / 5.36 s vs 26.60 s are in the scorecard row, so this suspend adds nothing new. |
| 46 / 53 | D11 PM-6 arms, plus RCPS closure | **SUSPEND** any "chelated arm is the better suppressor" reading | #46: yes ("parity/loss vs Arm 6 raw cosine ... with higher R5 FPR"). #53: partly. The anti-L4 note gives "0.50%/0.67% FPR" without saying that the better figure belongs to the raw-cosine arm. |
| 62 | D13 pooled agent loops | **No advantage over conventional arms shown** (a tie, not a loss) | No. The anti-L4 note reports "beating full_restart 59.38 vs 0.00 JPM" and omits the conventional Conv+SQLite and Conv+EK5_v3 arms in the same run. |

**Correction history.** A fresh Tier B reviewer checked the first draft of this report. Its corrections are folded in below:
- #41 was moved from SUSPEND to PARTIAL;
- the #41 counts were fixed (24/28, 8 direction groups);
- the Qwen ≤1/17 row was fixed;
- #62 was moved from "beats" to "tie";
- every **Claim.** line now cites where the claim is made.

The scorecard citations are line numbers in `docs/research/COMPLETE_64_BUNDLE_EVIDENCE_SCORECARD.json` (embedded digest `7954f9fcf0c7`).

---

## #41 D6 — chelated certified path vs "~940 ms exact fallback"

**Claim.** In the scorecard (bundle at line 489), the claim is cross-engine merge quality plus 0% stale resurrection. Examples: "FiQA 4096->384 CSLS=0.3068 vs DEV_ZScore=0.2563 [+5.05 pp] & Raw=0.1962".
- The scorecard **already disclaims latency**: "70/70 exact scans, single-host single-run with no replicates -> no pure ANN latency claim".
- The deployment framing against a "~940 ms exact fallback" appears only in `docs/research/council-logs/post_v7_panel_r1_fable52_medium.md:21`, which marks it unverified.
- So the latency part of this re-score tests a reading that the scorecard does not make.

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
   - For the same cell, `initial_backend_requests_rtt_sum` has p50/p95 of 25/31 ms.
   - **These two timers are not matched.**
     - `service_rtt` times the whole served request: shortlist, certificate, exact fallback and merge.
     - `initial_backend_requests_rtt_sum` times only the initial backend calls.
     - The ratio is an upper bound on the overhead of the certificate plus fallback, not a like-for-like ANN-vs-chelated latency.
   - The "Chelated shortlist" baseline is still Chelated CSLS. It is only scored on the shortlist, before the certificate runs, so it is not a conventional arm.
2. **Coverage.**
   - Fallback fraction is 1.0 in 62 of 70 cells.
   - The other 8 are all `pgvector_exact_vector4096` cells:
     - 6 at 0.5 (SciFact 4096, both directions, retraction 0/10/50);
     - 2 at 0.845 and 0.934 (FiQA 4096, 0% retraction).
   - That backend is exact by construction, so every served path is an exact scan. The scorecard already discloses this ("70/70 exact scans").
3. **Latency across the 56 cells that have both RTTs** (unmatched timers, see item 1).
   - Median served p50/p95: 32.8/52.2 ms. Median initial shortlist: 14.5/20.5 ms.
   - Ratio of served p95 to shortlist p95: median 2.7×, minimum 1.7×, maximum 50.9×.
   - The 14 cells at 90% retraction have no initial-RTT key, so I don't know their latency.
4. **Quality at matched shortlist.**
   - The served Chelated nDCG@10 mean equals the shortlist Chelated mean in **24 of 28** cells at 0% retraction.
   - An earlier draft said 22/28. That count compared whole `ndcg_at_10` dicts, and in 2 cells only a baseline mean differed.
   - The largest served-minus-shortlist gain is +0.0022 nDCG (qdrant FiQA 4096 d4096→d384).
   - So the certificate buys essentially no nDCG. Its stated purpose is the stale-resurrection guarantee, not nDCG.
5. **Merge quality vs conventional merges on the same shortlist** (`...paired_bootstrap_wave6`, Chelated minus baseline, 0% retraction, n = 190 queries).
   - There are **8** direction groups (2 datasets × 2 dims × 2 directions), not 7.
   - **Vs raw cosine** (`Uncalibrated_Raw_Merge`): Chelated wins nDCG in **all 8** groups with LCB95 > 0. The smallest is FiQA 1024 d384→d1024, +0.0227 (LCB95 0.0088). The largest is SciFact 1024 d1024→d384, +0.3687.
   - **Vs RRF:** Chelated wins nDCG in all 8, with LCB95 > 0.
   - **Vs DEV z-score:** Chelated wins in 7 of 8 at one-sided LCB95 > 0, but one of the 7 is marginal. In FiQA 4096
    d384→d4096 the delta is +0.0141 with LCB95 **+0.0004**, LCB97.5 −0.0022, two-sided p = 0.091. Under the
    artifact's own Bonferroni-2 band (`p0125_bonf2` −0.0045) it is a tie. So the robust count is 6 of 8. In SciFact 4096 d384→d4096 DEV z-score beats Chelated: −0.0258, one-sided UCB95 **−0.0036**, so the loss is significant one-sided (two-sided p = 0.053).
   - Raw cosine ties on recall in one group: FiQA 1024 d384→d1024, +0.023, LCB −0.0004, p = 0.105.
   - RRF and DEV tie on recall in several groups, e.g. FiQA 4096 d384→d4096 recall: vs RRF −0.0103 (p = 0.535), vs DEV −0.0011.
6. **Stale resurrection.**
   - `stale_resurrection_rate` = 0.0 for chelated.
   - `raw_engine_stale_hit_rate` = 0.0, so the raw allowlist baseline ties chelated.
   - The only baseline that leaks is `lagged_replica_only_stale_hit_rate`. On SciFact it is 5.2–8.2% at 10% and 50%
    retraction, and **31–36% at 90% retraction** (for example 0.3634 for opensearch 4096 d4096→d384). On FiQA it is
    `null` in every cell, so FiQA has no stale-hit baseline.

**Verdict: PARTIAL.**
- **Survives:** the scorecard's actual claim. Chelated CSLS merge quality beats raw cosine and RRF in all 8 direction groups at shortlist-matched cost. Against DEV z-score it wins 7 of 8 one-sided, but one of those (FiQA 4096 d384→d4096, p = 0.091) is marginal, so 6 of 8 is robust.
- **Not shown:** any latency or deployment advantage. That is consistent with the scorecard's own disclaimer, so suspending it changes nothing in the scorecard.
- **Stale hits:** the raw allowlist ties on stale hits, which is also already disclosed ("raw-engine 0% stale is zero by allowlist construction").
- **New:** one disclosure to add. DEV z-score significantly beats Chelated in SciFact 4096 d384→d4096.
- **Limits:**
  - Baseline-merge latency was never timed separately.
  - There are no per-query rows and no replicate runs.
  - I don't know the latency for the 90%-retraction cells.

---

## #45 D10-B — Bare 70B vs Chelated 70B

**Claim.** This is the deployment reading tested here: that the chelated pipeline (Nemotron-30B extractor plus reader) is worth deploying over a bare 70B reader.
- **The scorecard does not make this claim.**
  - The scorecard row (bundle at line 537) already reports the tie: "0.00 pp delta (Tango CI95=[-6.02, +6.02 pp], Holm p=1.0)".
  - It also reports the latency gap: "24.75-26.60s/case e2e vs 3.26s Bare-70B k=5".
- So this suspend is of an already-disclosed item.
- The only scorecard-level argument for the chelated pipeline is the small-reader path (Mid-7B at 14.28 GiB vs 38.08 GiB). That would have to be argued on amortized extractor cost, and no stored artifact argues that.

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
  - Chelated-70B agrees with Bare k17 on 58 of 60 cases and with Bare k5 on 57 of 60, by exact string match.
  - Vs k17, the 2 disagreements are:
    - `sec_financial_rumor_cobalt_aero_dod`: chelated abstains, bare answers CONTRACT_ACTIVE_ON_SCHEDULE.
    - `cybersec_cve_rumor_cve_2024_27351_orm_sqli`: a formatting difference only (chelated REDOS_ONLY_…, bare
      REDOSONLY_…).
  - Vs k5, the same 2, plus `cloud_sre_rumor_iam_token_forgery_505`, where bare abstains.
  - An earlier draft put the REDOSONLY difference on k5 only. It is in both.
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

**Claim.** This is the reading tested here: that the chelated full dual gate (Arm 9) is the right deployable suppressor of withdrawn content (R3) at a controlled benign false-refusal rate (R5).
- **#46 already discloses the counter-evidence.** Its scorecard row (bundle at line 549) says: "On English R3, Arm 9 shows parity/loss vs Arm 6 raw cosine ... & Arm 8 linear bridge ... with higher R5 FPR". It also says the R5 gate fails.
- **#53** (bundle at line 684, and the anti-L4 governance note at line 2) reports "0.50%/0.67% FPR, CP_UB95=1.29%/1.52%" without naming the arms. The new information below is that the better pair belongs to the raw-cosine RCPS arm.

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
| Qwen | ≤ 1/17 | 0.000 | **0.9412** | **0.9412** | 0.8235 |
| Qwen | ≤ 2/17 | 0.000 | **1.000** | 0.9412 | 0.8824 |

An earlier draft gave the ≤2/17 values under the ≤1/17 label. A Tier B reviewer caught this, and it was recomputed from the ROC arrays.

**Holm tests** (`pm6_posthoc_statistical_reanalysis.json`):
- Arm 9 minus Arm 6 AUROC: −0.0107 (Nemotron), −0.0208 (Qwen); adjusted p = 1.0.
- Arm 9 minus Arm 8: −0.0039 and −0.0138; adjusted p = 1.0.
- In the full 12-test family, three comparisons survive Holm:
  - Arm 9 vs Arm 3: +0.4736 (Nemotron) and +0.4758 (Qwen).
  - The cross-lingual ablation `phase3:de:paired_bootstrap_c_vs_a`: +0.1522, Holm-adjusted p = 0.029. It is about
    German cross-lingual configuration C vs A, not about Arm 9 vs the conventional arms.
  - An earlier draft said only the Arm 3 comparisons survive. That missed this one.
- **Arm 5 ≡ Arm 9 on R3/R5.** In `wave14_d11_pm6_manifest.json`, Arm 5 (gate 2 only, ZeroMu CSLS) and Arm 9 (the full
  dual gate) have identical R3 suppression and R5 FPR in both generators:
  - Nemotron: 0.9375 and 2/32.
  - Qwen: 0.8824 and 2/17.

  So on these metrics gate 1 adds nothing on top of gate 2.
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

**Claim.** What the scorecard says:
- The anti-L4 governance note (scorecard line 2) and the #62 row (bundle at line 860, `live_worker_outcome_colab_l4` at line 851) report that the chelated arm is "beating full_restart 59.38 vs 0.00 JPM".
- Neither mentions the conventional Conv+SQLite and Conv+EK5_v3 arms from the same run.
- The reading tested here is whether ZeroMu + EK5_v3 beats *conventional* handling, not only full restart.

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
| UncCSLS+EK5_v3 (recomputed from `cells[]`) | 6 | 0.125 | 45.58 | — |
| UncCSLS+SQLite (recomputed from `cells[]`) | 6 | 0.125 | 45.96 | — |

The two UncCSLS rows are not in `arm_summary`. I recomputed them from `cells[]` (regime "both", pool 4, both
branches) as Σ on-time jobs / Σ `window_duration_ms`. The same method reproduces all five `arm_summary` rows above
exactly.
- **Per branch:**
  - Branch A: ZeroMu leads at about 164 JPM, vs about 144 for Conv and about 127 for UncCSLS.
  - Branch B: every arm scores 0.
  - ZeroMu loses on the pooled measure because its windows are longer.
- So the chelated arm wins the per-cell comparison on one branch and loses the pooled one. That is a further reason
  the comparison is a tie, not a win.

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
  - Against the fully conventional Conv+SQLite arm it is a **tie**:
    - JPM: 59.38 vs 62.87 (−3.48), with no per-arm CI stored;
    - on-time jobs: 8 vs 7 of 48, also with no CI.
  - Conv+EK5_v3 is at 62.97 JPM.
  - The rate interaction is exactly 0.0. Neither direction of the difference is supported.
- The "59.38 vs 0.00" headline compares against the weakest baseline only.
- The sham control is invalid: a sham reopen alone costs about 1.44 s and pushes every sham job past the SLA. So the run also cannot attribute latency to EK5 lineage invalidation vs checkpoint reopen.
- The 560 ms SLA is recorded in the scorecard's #62 `tuning` field (line 855): "The preregistered 560.0 ms deadline and stall constants were NEVER retuned".

**Verdict: no advantage over conventional arms shown.**
- The chelated arm ties the conventional arms.
- The full-restart win is real, but it is against the weakest baseline. The scorecard's anti-L4 note should add the conventional arms beside it.
- Neither run passes overall, and both are status replay-fail.

---

## What is missing (I don't know)

- **D6:** latency for the 90%-retraction cells, per-query rows, separately timed baseline merges, replicate runs.
- **D10-B:** extractor VRAM, per-case latency and tails, and any composed run of the fast 0.5B extractor with a reader.
- **D11:** any result at N ≥ 149 benign families, and live-gateway FPR and latency for #53.
- **D13:** the UncCSLS cells have no `arm_summary` entry, but they are in `cells[]`, and their numbers are recomputed in the table above. An earlier draft said no data existed for them, which was wrong. There is still no same-hardware, same-model replicate of the Colab run, and no per-arm CI.
