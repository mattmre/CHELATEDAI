# Wave 25–28 Master Research & Engineering Queue — Gate-0 Falsifiers, Theory/Formal Expansions & Product Moats

```yaml
queue_id: wave25-plus-master-queue-2026-11-26
created_date: 2026-11-26
updated_utc: 2026-11-27T02:00:00Z
status: WAVE_25_TO_27_AUTOMATED_COMPLETE__WAVE_28_HUMAN_ONLY
release_status_preserved: PRE-RELEASE / HUMAN AUDIT & CLEAN-RERUN GATES PENDING
baseline_heads:
  pv7_spine_wave24_complete: "D:/GITHUB/CHELATEDAI-PV7-SPINE @ ecc9f97a (13/13 Wave 24 automated lanes + W24-10b/13b)"
  method_dev_repo: "D:/GITHUB/CHELATEDAI @ 65c9085c"
panel_consensus:
  - Gemini 4 Neon (Medium) [coordinator & synthesis]
  - Claude Fable 5.2 (High) [adversarial script/manifest audit & Wave 26+ expansion]
  - GPT-6.2 Astra (High) [mathematical/statistical rigor, systems architecture & Wave 26+ expansion]
policies:
  github_paid_billing: "$0.00 FOREVER (self-hosted Spark 2 ARM64 runner only)"
  fleet: "3x DGX Spark GB10 (128 GB UMA) + Desktop RTX 3090 (24 GB) + Colab Ultra (3,030.75 CU)"
  human_time: "STRICTLY LAST (Wave 28)"
  brutal_honesty_rulebook: "v3.3"
```

---

## 1. Unanimous Three-Model Wave 24 Audit & Mandatory Relabelings

Before spending any Wave 26+ compute, both independent panelists audited the Wave 24 scripts and manifests (`ecc9f97a`) and surfaced concrete claim tightenings that govern Wave 25:

1. **`W24-13b` (`T-Rn` Gated-DeltaNet + KV Retraction) → Relabel as `exploratory positive, n=8, mechanism unconfirmed`:**
   - At `2,053` tokens, **untouched KV-only has lower recurrent-state relative L2 error (`0.1149`) than damped subtraction (`0.1344`) and exact-decay (`0.1366`)**, even while next-token KL improves (`0.0250 → 0.0084`). The logit gain is therefore not yet shown to be mediated by restoring counterfactual state.
   - `exact_decay` uses only scalar $\exp(\sum g)$, omitting the directional erase-projector product $\prod (I - \beta k k^\top)$ and upper-layer input dependence.
   - All `8` cases repeat one unrelated filler sentence (zero dirty-cone interference), $\gamma$ was selected post hoc on the evaluation cases, the KV copy scales with surviving length (not $O(1)$ end-to-end), and no bf16/fp32 noise floor, correction-notice, or segment-tree baseline was run.
2. **`W24-12` (`QSCCI` Multi-Layer Restoration) → Relabel as `exploratory, quantization-limited`:**
   - `8` REPORT families with `1/64` quantization steps; `k=1` and `k=16` are identical to 4 decimals on both models while `k=4/8` dip non-monotonically; rank-1 uses the mean output gradient of the same Yes/No margin later scored; CAA loses by construction because REPORT inverts SELECT's tag nuisance.
3. **`W24-09` (Flagship #1 Retrieval):**
   - Individual Tango NI passed on `7/12` adaptive and `8/12` uniform domains; the `-0.0120` LCB95 is **pooled** over `1,865` queries on binary **`hit@10`** using a qrel-preferential corpus loader. Adaptive backfill is null vs uniform (`-0.003 pp`). The `+31–39 pp` `CSLS` lift needs a 2-parameter per-source affine calibration control to separate hubness from score-scale mismatch.
4. **`W24-10b` (`PM-6` Gateway):**
   - The NFKC-normalized 5-token span guard drove both `100%` TPR and all `3/250` sibling FPs (`ZeroMu` alone had `0/250` FP and `2.4–17.6%` TPR). At `1.2%` true FPR, `N_eval=500` has only ~28% power to clear `CP_UB95 <= 2.0%` (`<=4/500`); **`N_eval >= 2,000–3,000`** fresh source-disjoint benigns is required.
5. **`W24-07` & `W24-11` (`T-Rs` / `EK5`):**
   - `Theorem5_IncrementalWFS_CW1` assumes `AffectedClosed` and calls full-graph `gammaFast` intersected with the cone; runtime locality additionally needs a verified cone finder + sparse evaluator refinement. Folding `Theorem5` into `Spec.lean` requires versioning the SHA lock. SQLite cannot express WFS—true comparators are **XSB incremental tabling** and DuckDB/Postgres + DBSP Lean positioning.
6. **`W24-08` (`T-D` $\lambda^*$):**
   - `0.10` is the first passing grid point, where the primary harm-difference CI spans `[-0.0312, +0.0151]` (`strict_ci_crossover = null`). Treat `0.10` as a simulation-grid boundary, not a statistically located universal threshold.

**Operating Rule:** Execute **Wave 25** (`W25-01`..`W25-06`) first. Do **not** launch 8K–32K `T-Rn` scale-ups (`W26-01`), `QSCCI × T-Rn` stacking (`W27-01`), or 3-node distributed revocation (`W27-02`) until their respective Wave 25 Gate-0 / Step-0 / `#62` prerequisites pass.

---

## 2. Wave 25 — Gate-0 Falsifiers, Claim Hygiene & Operational Closure (`5/5` Automated Runs COMPLETE)

| Wave ID | Prio | Title | Status & Verdict | Verified Empirical Summary (Rulebook v3.3) |
|---|---|---|---|---|
| **`W25-01`** | P0 | **Wave 24 Operational PR Closure (`W24-04`)** | **READY FOR OWNER PUSH** | Local branches tested & clean (`f245d923`, `5b0b604b`, `692b44aa`); awaiting owner approval for outward `git push` / PR updates. |
| **`W25-02`** | P0 | **`T-Rn` Gate 0: Noise Floor, State Alignment & `n=200` Dirty-Cone Falsifier** | **DONE — REGIME SPLIT (`KILL UNROUTED` / `ADVANCE CLEAN-CONE + EK5 ROUTER`)** | Noise floor `0.001015 KL`; cache has `18` `recurrent_states` + `18` `conv_states` + `6` KV layers; SELECT `cos=0.1609`. **Pooled unrouted FAILS** (`+0.0108 KL` worse than `kv_only`) because `dependent_dirty_cone` suffix copies defeat all zero-replay edits (`0.146–0.173 KL`). **Clean-cone (`natural_unrelated`, `n=100`, 4 templates, diverse real prose, SELECT-frozen `gamma*_lh`) PASSES ALL 3 CRITERIA:** **`-0.005132 KL`** vs `kv_only` (`CI [-0.00754, -0.00272]`, **`5.06×` noise floor**), lowers state L2 (`0.0861→0.0746`) & collateral KL (`0.0227→0.0200`), crushes correction notice (`-0.3079 KL`), `100%` margin-old, `18.2×` speedup. Scalar `exact_decay` fails (`+0.0142 KL`). Artifact: `artifacts/pv7_w25_02_trn_gate0_spark2/w25_02_trn_gate0_manifest.json`. |
| **`W25-03`** | P0 | **`QSCCI` Step 0: Powered `fp32` Replication & Cross-Encoding Invariance** | **DONE — KILLED (`KILL_QSCCI_STEP0`, `1/3` models)** | `k=1==k=16` diagnosed as `bf16` `1/64` quantization. `Qwen2.5-1.5B` passes Yes/No (`3/3` seeds `+0.391..0.393`), but `SmolLM2-1.7B` (`-0.189/-0.062/+0.022`) and `microsoft/phi-2` (ties `balCAA` `2/3` seeds) FAIL. **Cross-encoding falsifier kills QSCCI:** on `unseen_codes_alpha_omega`, QSCCI collapses (`+0.011` Qwen, `-0.180` Phi-2) while `balCAA` generalizes (`+0.171` Qwen, `+1.833` Phi-2); on `reversed_labels`, Phi-2 QSCCI flips negative (`-0.082`, tracks `"Yes"` token not truth) while `balCAA` tracks truth (`+0.597`). Retired from `W27-01`/`W27-03`. Artifact: `artifacts/pv7_w25_03_qscci_step0_fp32/w25_03_qscci_step0_manifest.json`. |
| **`W25-04`** | P1 | **`#62` Pooled Agent Loops Live `vLLM` `Qwen2.5-0.5B-Instruct`** | **DONE — HOLD (`7/8` gates; `W27-02` STAYS GATED)** | Full canonical `1,120`-cell / `26,880`-job live sweep (`129,153` live `vLLM` calls, `0` errors, `0` stale commits, `0` EK5 parity failures). Crushes `full_restart_rebuild` (`+173.88` vs `0.00` JPM at `pool=4`) and beats `Conv` at `pool=8` (`+129.6..+147.2` JPM), **but FAILS `sham_reopen_control_valid`** (`1,618 ms > 560 ms` SLA, `0/192` on-time) and trails `Conv+EK5` at `pool=4` (`173.88` vs `183.66` JPM; `2×2` interaction `-8.05`, CI `[-32.58, +12.78]`) because raw `178 ms` vLLM calls were not routed through `W24-06`'s `29 ms` pre-filter. Artifact: `artifacts/pv7_w25_04_b62_vllm_spark2/pv7_w20_d13_manifest.json`. |
| **`W25-05`** | P1 | **`#48` HNSW Pushdown on Qrel-Independent Full Real Corpora** | **DONE — SELECTIVITY-GATED WIN** | `92,811` full real docs (0 qrel filtering; `cqadupstack-{android,programmers,gis}`), DEV-frozen FAISS HNSW params, `0` fallbacks. Post-filter is faster at `adm >= 0.20` (`0.61×–0.92×`); at **`adm = 0.05`**, `#48` pre-filter pushdown wins across `3/3` full corpora (**`1.49×` p50 `5.83 vs 8.67 ms`**, **`1.42×` p95 `7.38 vs 10.51 ms`**) at matched `nDCG@10` (`0.38391`). Exposes `-20.9 pp` full-corpus `20%`-backfill gap vs `100%` `bge-m3` oracle (`0.2344` vs `0.4435`) for `W26-04`. Artifact: `artifacts/pv7_w25_05_hnsw_full_real_spark3/w25_05_hnsw_full_real_manifest.json`. |
| **`W25-06`** | P1 | **`PM-6` `W24-10b` `G6` Powered Calibration (`N=3,000`) & Span-Guard Ablation** | **DONE — G6 PASS (SPAN-DOMINATED)** | `N_eval=3,000` fresh source-disjoint siblings across 8 forums (`N_cal=1,200` DEV-frozen). Old raw 5-tok span guard fails G6 (`2.47%`, `CP_UB95=2.99%`). DEV-fitted **high-IDF 5-tok span guard alone** crushes G6 (**`1/3,000 = 0.033%`**, `CP_UB95=0.158%`, `cluster_UB95=0.100%`, `99.5%` verbatim TPR, `0.5%` R2b bypass, `0.0%` paraphrase). Combined live gateway clears G6 (**`5/3,000 = 0.167%`**, `CP_UB95=0.350%`, `cluster_UB95=0.500%`, `+1.5 pp` paraphrase TPR at `4×` FPs). Artifact: `artifacts/pv7_w25_06_pm6_g6_spark3/w25_06_pm6_g6_manifest.json`. |

---

## 3. Wave 26 — High-Upside Theory, Formal Verification & Core Expansions (`4/4` Automated Runs COMPLETE)

| Wave ID | Prio | Title | Status & Verdict | Verified Empirical & Formal Summary (Rulebook v3.3) |
|---|---|---|---|---|
| **`W26-01`** | P1 | **Retraction Half-Life Law, Operator-Exact DeltaNet Surgery & Certified-or-Replay Router** | **DONE — `ADVANCE_OPERATOR_EXACT_CLEAN_CONE` (WITH WALL-CLOCK CAVEAT)** | Exact $\prod_t \alpha_t (I - \beta_t k_t k_t^\top)$ verified on all `18/18` `Qwen3_5GatedDeltaNet` layers (`Qwen/Qwen3.5-2B-Base`, algebraic error `2.7e-6`). Clean-cone REPORT (`n=40`): cuts KL **`2.8×`** vs `kv_only` (`0.02572 -> 0.00921`, CI `[-0.0237, -0.0104]`) and state L2 (`0.0860 -> 0.0617`). Scales monotonically to `8,141` tokens: Layer-0 residual L2 = **`0.00114`**, KL `0.01015 -> 0.00682`, retained norm `20.6% -> 14.6% -> 11.4% -> 9.8%`. `EK5`+operator router beats lexical heuristic by **`+0.0545` AUROC** (`0.8935` vs `0.8390`). **Honest caveat:** fails on dirty cones (must replay) and naive sequential Python propagation at `8K` took `25.97s` vs `1.64s` GPU replay (chunked-WY tested in `W27-01`). Artifact: `artifacts/pv7_w26_01_operator_exact_spark2/w26_01_operator_exact_manifest.json`. |
| **`W26-02`** | P1 | **Proof-Carrying `EK5` Updates & Verified Sparse Cone Refinement (`Theorem6`)** | **DONE — `ADVANCE_W26_02_PCC` (STRONGEST PAPER RESULT)** | Machine-checked Lean 4 module `formal/ChelatedFormal/Theorem6_VerifiedConeAndPCC.lean` proves verified cone closure (`computeAffectedCone_affectedClosed`), strictly sparse cone evaluator soundness (`sparseConeWfs_sound`), and zero-false-accept certificate soundness (`verifyWfsCert_sound`) across `{T, F, U}` with `0 sorry` / `0 native_decide`; `verify_all_offline.py --full-formal` PASSED (`93` Lean theorems, **`25,604` `nanoda_bin` external-kernel declarations checked**, `Spec.lean` SHA-256 lock intact, `18/18` verifiers). Empirical Spark 2 stress: **`1,270,000` node checkpoints** (`N=1k/5k/10k`), **`0` mismatches**, **`0/492` adversarial forgeries accepted**, `0.024 ms` verify vs `19.12 ms` full rebuild (`p95_ratio = 0.20% << 10%`), **`89 B` certs** (`3,151×` smaller than full justifications). Artifact: `artifacts/pv7_w26_02_pcc_ek5_spark2/w26_02_pcc_ek5_manifest.json`. |
| **`W26-03`** | P2 | **Value of the Next Provenance Check (VoI) & Live OpenAlex Retracted Cascades** | **DONE — `ADVANCE_W26_03` (WITH LOW-COST PHASE BOUNDARY)** | Exact Bayes VoI `M_eff * max(0, min(p,1-p) - eps) * H > c` satisfies `dp <= greedy` invariant in `6/6` cells. Under budget `B=1` check selection, copy-aware VoI strictly beats greedy independent VoI at **all 3 costs** (`c=0.08/0.25/0.40`) on both held-out topologies (`H1: -0.365/-0.406/-0.600`, `H2: -0.277/-0.291/-0.382`); unconstrained threshold strictly wins at `c=0.25/0.40` and narrows to null (`-0.018/-0.012`) at ultra-cheap `c=0.08`. Live `api.openalex.org` retracted works (`heldout year >= 2023`): `118` hop-1 + `528` hop-2 = `646` real affected papers; `EK5` covers `100%` vs one-hop `18.27%` (**`+81.73 pp`**, CI `[+78.79, +84.68]`). Live SQLite recursive CTE gets `0.0%` recall on citogenesis loops vs `100%` (`0` FP) for `EK5` WFS. Artifact: `artifacts/pv7_w26_03_voi_cascades_spark2/w26_03_voi_cascades_manifest.json`. |
| **`W26-04`** | P2 | **48-Cell Quantized/Truncated Hubness Law & Dynamic CSLS Churn (`CenteredProcrustes` ONLY)** | **DONE — KILL TRIGGERED (`AFFINE_EXPLAINS_91_TO_96PCT`)** | `48` cells (`4` qrel-independent corpora × `4` quantization regimes `float32/int8/binary_1bit/truncated_512d` × `3` backfill fractions). CSLS beats QB-Norm (`+9.88 pp`, CI `[+9.32, +10.42]`) and selective churn cuts refresh traffic **`9.33×`** across `4/4` domains. **HOWEVER, kill gate triggers:** a 2-param per-source affine calibration alone recovers **`95.8%`** of CSLS gain on `float32`, **`94.1%`** on `int8`, **`90.8%`** on `truncated_512d` (`+1.55–3.11 pp` residual; `80.9%` / `+3.88 pp` on `1-bit` binary), and hubness-law correlation fails (`Spearman(k-skew, gain) = 0.275 < 0.60`). Parked as commodity calibration finding. Artifact: `artifacts/pv7_w26_04_hubness_law_spark2/w26_04_hubness_law_manifest.json`. |

---

## 4. Wave 27 — End-to-End Benchmarks, Adversarial Suites & Product Moats (Automated Runs COMPLETE; `W27-02` Properly Gated)

| Wave ID | Prio | Title | Status & Verdict | Verified Empirical Summary (Rulebook v3.3) |
|---|---|---|---|---|
| **`W27-01`** | P2 | **`RetractBench`: 5-Probe Leakage Suite (`Qwen3.5-2B`, Chunked-WY, `QSCCI` Excluded)** | **DONE — `REGIME_QUALIFIED_OR_HOLD_W27_01` (CRUSHES LEACE/CORRECTION/KV-ONLY ON `P4`; VOLATILE-TAIL WINS WHEN REORDERABLE)** | Chunked-WY exactness verified to `6.52e-9` max abs error vs sequential propagation (`QSCCI` excluded per `W25-03`). Across `n=60` REPORT (`30` clean + `30` dirty): unedited hidden-state probe `P4` AUROC = `1.000`, `appended_correction_notice` = `1.000` (`86.7%` extraction leak, `0.1882` collat KL), `leace_linear_erasure` = `0.915` (`0.0219` collat KL; `leace_matches_probe_kill = false`), `unrouted_kv_excise_only` = `0.775` (`0.0182` collat KL). **`ek5_routed_operator_exact_wy` drops `P4` probe AUROC to `0.566`** (near `0.50` gold chance floor), matches gold exactly (`0.00 pp` gap) on `P1` (`16.7%`), `P2` (`25.0%`), and `P5` (`50.0%`), within `1.67 pp` on `P3` (`53.3% vs 55.0%`), cuts collateral KL **`3.2×`** vs KV-only (`0.00561`), and cuts clean-cone p95 latency to **`0.727×`** of gold replay. **Honest qualification:** pooled p95 = `1.018×` gold (dirty-cone replay tail) and `volatile_tail_prefix_cache` achieves exact gold parity at `74.1 ms` p50 (`<= 2×` latency) whenever prompt order can be controlled. Artifact: `artifacts/pv7_w27_01_retractbench_spark2/w27_01_retractbench_manifest.json`. |
| **`W27-02`** | P2 | **Epoch-Consistent Revocation Across the Distributed Serving Stack** | **PROPERLY GATED (`HOLD` PER `W25-04`)** | Prerequisite `W25-04` (`#62` live `vLLM`) held at `7/8` gates (`sham_reopen_control_valid = false`, `1,618 ms > 560 ms` SLA). Respecting Wave 25 gate discipline, `W27-02` remains gated until `W24-06` `29 ms` pre-filter is wired into `LiveVLLMWorkerBackend`. |
| **`W27-03`** | P2 | **`PM-6` Adaptive Exfiltration Benchmark vs. Conventional Floor (`QSCCI` Excluded)** | **DONE — `CONVENTIONAL_FLOOR_DOMINATES_OR_CHUNKING_GAP` (RETIRES BOTH `ZeroMu` & RESIDUAL PROBE)** | `N_eval = 2,400` fresh source-disjoint siblings across 8 unburned forums × `6` adversarial exfiltration families (`1,200` attacks; Desktop RTX 3090). **Zero-ML conventional floor** (`NFKC + UTS #39 + Base64/ROT13/leet decoder + pruned 3-tok spans + char 5-gram Jaccard`) hits **`0.167%` FPR** (`4/2,400`, `CP_UB95=0.381%`), **`0.0%` bypass** on verbatim, R2b homoglyphs, and Base64/ROT13/leet, **`7.0%`** bypass on `<5`-token chunking, and `95.5–100%` bypass on cross-lingual & heavy paraphrase (`40.5%` mean adaptive bypass). **Kill gate retires BOTH learned/white-box arms (`< 10 pp` reduction over conventional floor at `FPR <= 1%`):** `ZeroMu` cuts bypass by only **`+0.5 pp`** (`40.5% -> 40.0%`, `RETIRE_ZEROMU`) and `Qwen2.5-1.5B` residual-stream linear probe cuts bypass by only **`+0.2 pp`** (`40.5% -> 40.3%`, `RETIRE_PROBE`). Combined `<5`-token chunking bypass = **`6.5% > 5.0%`** (`reopens_pm6_gates = true`). Artifact: `artifacts/pv7_w27_03_pm6_adaptive_exfil_rtx3090/w27_03_pm6_adaptive_exfil_manifest.json`. |

---

## 5. Wave 28 — Human & External Partner Closure (Strictly Last)

| Wave ID | Prio | Title | Owner | Acceptance Criterion |
|---|---|---|---|---|
| **`W28-01`** | P3 | **`PM-4b` Independent Human Audit (`W24-15`)** | Independent human grader (~6 hrs) | Complete `pm4_human_audit_100_blind_sheet.csv` without touching sealed key; lifts `PRE-RELEASE` on v6.0/v7.0. |
| **`W28-02`** | P3 | **Outside Workload-Owner Trial (`W24-16`: `#63` Search or `#64` Records)** | External partner | Partner sets workload, natural prevalence, error prices, and ordinary comparator; advance only if quality/harm **and** full-cost bounds clear frozen margins. |

---

## 6. Permanently Closed / Parked Lanes (Do Not Re-Open Without a New Mechanism)

`T-K` (cross-model KV transfer), `W22-08` / `T-A` (capture-recapture audit residual, K2 confirmed by blind coder `W24-14`), `D4`/`D9` GRPO (replicated null vs SFT), `IQ-01` (beaten by Frobenius norm), `IQ-03`/`IQ-05`/`IQ-06`/`IQ-07` (exact algebraic reductions / invalid battery), `ZeroMu` static/EMA bridge superiority over `CenteredProcrustes` (`W24-09`), PR #361 `#356/#359/#360`, and chelation correctors / H5 / Rung-15 GNN.

---

## 7. Delegate Audit Provenance

- **Claude Fable 5.2 (High):** `thread:delegated-task:command%3Amcp%3Ad5533b7c-6ffd-42fc-9d49-a78f01204ac2%3Adelegate-task%3Awave25-plus-expansion-fable52-high-20261126`
- **GPT-6.2 Astra (High):** `thread:delegated-task:command%3Amcp%3Ad5533b7c-6ffd-42fc-9d49-a78f01204ac2%3Adelegate-task%3Awave25-plus-expansion-astra62-high-20261126`
