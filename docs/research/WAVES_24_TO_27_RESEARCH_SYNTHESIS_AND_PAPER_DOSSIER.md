# Waves 24–27 Research Synthesis, Paper Dossiers, & Honest Falsification Ledger

> **Date:** `2026-11-28`  
> **Release Governance Status:** **`PRE-RELEASE / HUMAN AUDIT & CLEAN-RERUN GATES PENDING`** (`PM-4b` independent human grading of `pm4_human_audit_100_blind_sheet.csv` and `W28-02` outside workload-owner trial remain strictly human-gated).  
> **Formal Verification Tier:** **`PASS`** — `python scripts/verify_all_offline.py --full-formal` (`18/18` offline verifiers `rc=0`, `93` Lean 4 theorems across `11` `.lean` modules, `0 sorry`, `0 native_decide`, **`25,604` declarations checked by external Rust kernel `nanoda_bin 0.4.32`**, `Spec.lean` SHA-256 lock intact).  
> **Compute Provenance:** `3× NVIDIA DGX Spark GB10 (128 GB UMA)` + `Desktop NVIDIA RTX 3090 (24 GB)` + `Colab NVIDIA L4` (`$0.00` GitHub Actions paid billing).

---

## 1. Executive Summary: What Survived vs. What Died Under Adversarial Baselines

Across Waves 24–27 (`24` automated experimental & formal lanes layered on the `64/64` `v7.0` / `Post-v7.0` archive), every surviving mechanism from Waves 1–23 was tested head-on against strong, zero-handicap conventional baselines under **Brutal Honesty Rulebook v3.3**:

1. **Two Genuine, Publishable Computer-Science Contributions Survived:**
   - **Paper 1 (Formal Methods & Provenance Systems):** *Proof-Carrying Incremental Well-Founded Semantics & Priced Provenance Cascades (`EK5`, `Theorem5`, `Theorem6`, `W26-02`, `W26-03`).*
   - **Paper 2 (Hybrid Sequence-Model State Mechanics):** *Operator-Exact Chunked-WY State Retraction & `RetractBench` on Hybrid Gated-DeltaNet Transformers (`W25-02`, `W26-01`, `W27-01`).*
2. **Every Standalone Commercial "Magic Moat" Hypothesis Was Falsified or Scope-Bounded:**
   - **`QSCCI` Activation Steering (`W25-03`):** **KILLED** (`1/3` models in `fp32`; failed cross-encoding invariance on unseen codes and flipped sign on reversed labels while plain nuisance-balanced CAA generalized).
   - **`ZeroMu` Embedding Bridge & 48-Cell Hubness Law (`W24-09`, `W26-04`):** **RETIRED / COMMODITY** (`CenteredProcrustes` tied `ZeroMu`; a 2-parameter per-source affine calibration recovered **`95.8%`** on `float32`, **`94.1%`** on `int8`, and **`90.8%`** on `512d` truncated embeddings of `CSLS`'s gain, and `Spearman(k-skew, gain) = 0.275 < 0.60`).
   - **`PM-6` Learned / White-Box Exfiltration Detection (`W25-06`, `W27-03`):** **CONVENTIONAL ZERO-ML FLOOR DOMINATES** (deterministic `NFKC + UTS #39 + Base64/ROT13/leet decoder + pruned 3-tok + char 5-gram` floor achieved `0.167%` FPR on `2,400` fresh sibling benigns with `0%` verbatim/homoglyph/encoding bypass; `ZeroMu` added `+0.5 pp` and a `Qwen2.5-1.5B` residual-stream linear probe added `+0.2 pp`, retiring both under the `< 10 pp` rule).
   - **`#48` HNSW Pre-Filter Pushdown (`W25-05`):** **SELECTIVITY-GATED** (on `92,811` qrel-independent full real docs, post-filter oversampling is faster at `adm >= 0.20`; pre-filter pushdown wins `1.49×` p50 / `1.42×` p95 at `adm = 0.05` across `3/3` corpora).
   - **`#62` Pooled Live `vLLM` Agent Loops (`W25-04`):** **HELD (`7/8` gates)** (`129,153` live `Qwen2.5-0.5B-Instruct` calls, `0` errors, `0` stale commits, `+173.88` vs `0.00` JPM over `full_restart_rebuild`, but raw `178 ms` vLLM calls missed the `560 ms` sham-reopen SLA at `1,618 ms`, keeping `W27-02` gated).

---

## 2. Paper Dossier #1: Proof-Carrying Incremental Well-Founded Semantics & Provenance Cascades

### 2.1 Machine-Checked Lean 4 + External Rust Kernel (`nanoda_bin`) Proofs
- **Modules:**
  - `formal/ChelatedFormal/Theorem3_EK5_CyclicWFS.lean` — Stratum 1–2 alternating-fixpoint Well-Founded Semantics (`Γ_P` antimonotone, `Γ_P²` monotone, `T = lfp(Γ_P²)`, `T ∩ F = ∅`, `evalStratum2Wfs_sound`).
  - `formal/ChelatedFormal/Theorem5_IncrementalWFS_CW1.lean` (`W24-11`) — Affected-cone frame isolation (`wfs_frame_isolation`), incremental splice soundness under arbitrary deletions/mutations inside $A$ (`incrementalWfsSplice_sound`), unfounded positive-cycle rejection (`unfounded_pos_cycle_false`), and edge-deletion monotonicity.
  - `formal/ChelatedFormal/Theorem6_VerifiedConeAndPCC.lean` (`W26-02`) — Verified affected-cone closure (`computeAffectedCone_affectedClosed`), strictly sparse cone-only evaluator soundness (`sparseConeWfs_sound`), zero-false-accept proof-carrying certificate verifier (`verifyWfsCert_sound`) across `{T, F, U}`, and non-vacuity witness (`theorem6_witness_nonvacuous`).
- **Verification Receipt:** `93` Lean 4 theorems (`0 sorry`, `0 native_decide`, standard axioms `['Classical.choice', 'Quot.sound', 'propext']` only), **`25,604` declarations re-verified by `nanoda_bin 0.4.32`**, `Spec.lean` SHA-256 lock (`formal/spec_sha256.json`) untouched.

### 2.2 `1.27M`-Checkpoint Proof-Carrying Update Stress (`W26-02`)
- **Script & Manifest:** `scripts/run_w26_02_pcc_ek5_million_stress.py` · `artifacts/pv7_w26_02_pcc_ek5_spark2/w26_02_pcc_ek5_manifest.json`

| Graph Size $N$ | Node Checkpoints | Mean Cone Fraction | Full Rebuild (`ms`) | PCC Verify Mean / p95 (`ms`) | p95 Verify / Full Work Ratio | Mean Cert Size (`B`) | Compression vs. Full Justifications |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `1,000` | `120,000` | `0.492%` | `1.482` | `0.0170 / 0.0246` | **`1.71%`** | `82.8 B` | **`338.2×`** |
| `5,000` | `500,000` | `0.099%` | `8.354` | `0.0198 / 0.0275` | **`0.34%`** | `94.1 B` | **`1,487.6×`** |
| `10,000` | `650,000` | `0.042%` | `19.124` | `0.0237 / 0.0379` | **`0.20%`** | `88.9 B` | **`3,151.0×`** |

- **Soundness & Forgery Audit:** **`0` semantic mismatches** across **`1,270,000`** node checkpoints; **`0 / 492` adversarial certificate forgeries accepted** (bit-flips, truncated cones, mislabeled unfounded cycles, stale boundary splices).

### 2.3 Exact Bayes VoI & Live OpenAlex Retracted Cascades (`W26-03`)
- **Script & Manifest:** `scripts/run_w26_03_voi_and_real_cascades.py` · `artifacts/pv7_w26_03_voi_cascades_spark2/w26_03_voi_cascades_manifest.json`
- **Analytical Provenance Buy/Stop Rule:** Under noisy binary check error $\varepsilon$, prior $p$, per-claim harm $H$, and shared-source downstream mass $M_{\mathrm{eff}}$, exact Bayes Value of Information is $\mathrm{VoI} = H M_{\mathrm{eff}} \max(0, \min(p, 1-p) - \varepsilon) > c$ (`dp <= greedy` invariant verified in `6/6` cells):
  - **Budget `B = 1` check selection:** Copy-aware VoI strictly beats greedy independent VoI at all 3 costs (`c = 0.08 / 0.25 / 0.40`) on both frozen held-out topologies (`H1: -0.365 / -0.406 / -0.600`, `H2: -0.277 / -0.291 / -0.382`; all 95% CIs strictly `< 0`).
  - **Unconstrained threshold mode:** Strictly wins at `c = 0.25` and `c = 0.40` (`H1: -0.224 / -0.615`, `H2: -0.157 / -0.400`), narrowing to null (`-0.018` / `-0.012`) at ultra-cheap `c = 0.08` where greedy checks `>90%` of roots anyway.
- **Live `api.openalex.org` Retracted Citation Cascades (`heldout year >= 2023`):** Across `646` real affected citing papers (`118` hop-1 + `528` hop-2), 1-hop retraction covers only **`18.27%`** while `EK5` multi-hop covers **`100.0%`** (**`+81.73 pp`**, 95% CI `[+78.79, +84.68]`).
- **Circular-Reporting Citogenesis Sweep (`2% / 5% / 10%` cluster incidence):** `EK5` WFS achieves **`100.0%` recall (`0` FP)** in `1.79–2.14 ms` vs. **`0.0%` recall** for downward 1-hop and live SQLite recursive CTE queries.

---

## 3. Paper Dossier #2: Operator-Exact Gated-DeltaNet State Retraction & `RetractBench`

### 3.1 Clean- vs. Dirty-Cone Regime Split (`W25-02`, `n = 200` REPORT)
- **Manifest:** `artifacts/pv7_w25_02_trn_gate0_spark2/w25_02_trn_gate0_manifest.json` (`Qwen/Qwen3.5-2B-Base`: `18` `Qwen3_5GatedDeltaNet` layers + `6` full-attention KV layers; `bf16/fp32` KL noise floor = `0.001015`).
- **Dirty-Cone Falsification (`dependent_dirty_cone`, `n = 100`):** When downstream suffix tokens explicitly copy or reason over the retracted fact, *all* zero-replay edits fail (`0.146–0.173 KL`) — suffix replay is mandatory.
- **Clean-Cone Discovery (`natural_unrelated`, `n = 100`, 4 templates, SELECT-frozen `gamma*_lh`):** Zero-replay damped subtraction + KV excision beats `kv_only` by **`-0.005132 KL`** (95% CI `[-0.00754, -0.00272]`, **`5.06×` noise floor**), lowers state L2 (`0.0861 -> 0.0746`) and collateral KL (`0.0227 -> 0.0200`), and beats appended correction notices by **`-0.3079 KL`** at `18.2×` speedup.

### 3.2 Operator-Exact Homogeneous Transition Surgery up to `8,141` Tokens (`W26-01`)
- **Script & Manifest:** `scripts/run_w26_01_operator_exact_deltanet_and_router.py` · `artifacts/pv7_w26_01_operator_exact_spark2/w26_01_operator_exact_manifest.json`
- **Exact Recurrence Operator:** Propagating initial perturbation $\Delta S_0 = S(A+X) - S(A)$ through $\Delta S_t = \alpha_t (I - \beta_t k_t k_t^\top) \Delta S_{t-1}$ across all `18/18` linear-attention layers (`algebraic self-test err = 2.72e-6`):
  - Cuts clean-cone REPORT (`n = 40`) KL **`2.8×`** vs. `kv_excise_only` (`0.02572 -> 0.00921`, 95% CI `[-0.0237, -0.0104]`) and state L2 (`0.0860 -> 0.0617`).
  - Scales monotonically from `408` to **`8,141` suffix tokens**: Layer-0 relative state L2 drops `0.00478 -> 0.00330 -> 0.00182 -> 0.00114` (`0.11%` error at `8K`!), operator retention decays `20.6% -> 14.6% -> 11.4% -> 9.8%`, and KL stays below `kv_only` at every length (`0.01015 -> 0.00682` at `8,141` tok).
  - **`EK5` + Operator-Norm Router:** Predicts `KL > 0.05` unsafe cones at **`0.8935` AUROC** vs. **`0.8390`** for lexical overlap (**`+0.0545` AUROC**).

### 3.3 `RetractBench`: Chunked-WY Exact Propagation Across 5 Adversarial Leakage Probes (`W27-01`)
- **Script & Manifest:** `scripts/run_w27_01_retractbench_5probes.py` · `artifacts/pv7_w27_01_retractbench_spark2/w27_01_retractbench_manifest.json`
- **Chunked-WY Reformulation:** Exploiting scalar decay commutativity $\prod_{i=0}^{c-1} \alpha_i (I - \beta_i k_i k_i^\top) = e^{\sum g_i} \prod_{i=0}^{c-1}(I - \beta_i k_i k_i^\top)$ via unit-lower-triangular `solve_triangular` (`chunk_size = 32`, verified to **`6.52e-9`** max abs error vs. sequential scan) eliminates `W26-01`'s Python token-loop bottleneck (`clean-cone p95 = 0.727×` gold replay).

| Arm (`n = 60` REPORT: `30` Clean + `30` Dirty) | `P1` Direct Leak | `P2` Paraphrase Leak | `P3` Multi-Hop Leak | `P4` Hidden-State Probe AUROC vs. Gold (`0.50` = Ideal) | `P5` Extraction Leak | Collateral KL | p50 / p95 Latency (`ms`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gold_delete_and_rebuild` *(ceiling)* | `16.67%` | `25.00%` | `55.00%` | **`0.5000`** | `50.00%` | `0.00000` | `148.7 / 163.0` |
| `volatile_tail_prefix_cache` *(when reorderable)* | `16.67%` | `25.00%` | `55.00%` | **`0.5000`** | `50.00%` | `0.00000` | `74.1 / 164.4` |
| `appended_correction_notice` | `25.00%` | `43.33%` | `46.67%` | `1.0000` | `86.67%` | `0.18824` | `114.9 / 127.0` |
| `leace_linear_erasure` | `0.00%` | `13.33%` | `51.67%` | `0.9153` | `45.00%` | `0.02186` | `6.8 / 7.6` |
| `unrouted_kv_excise_only` | `0.00%` | `13.33%` | `51.67%` | `0.7747` | `45.00%` | `0.01819` | `6.7 / 7.6` |
| **`ek5_routed_operator_exact_wy` (Ours)** | **`16.67%`** | **`25.00%`** | **`53.33%`** | **`0.5661`** | **`50.00%`** | **`0.00561`** | `139.4 / 165.9` *(clean p95 `0.727×`)* |

- **Honest Scope Boundary:** Whenever prompt layout can place volatile spans at the tail (`volatile_tail_prefix_cache`), prefix truncation achieves exact gold fidelity at `74.1 ms` p50 (`0.50×` gold). `ek5_routed_operator_exact_wy` is specifically valuable for **fixed-order historical streams** where prompt reordering is impossible — where it is the **only** non-replay method to collapse residual-state linear-probe detectability (`1.000 -> 0.566`) while cutting collateral KL `3.2×` below `kv_only`.

---

## 4. Complete Wave 24–27 Negative, Killed, & Bounded Ledger (Anti-Rediscovery Map)

| Wave / ID | Hypothesis Tested | Decisive Conventional Comparator | Empirical Outcome & Permanent Disposition | Artifact |
|---|---|---|---|---|
| **`W25-03`** | `QSCCI` multi-layer causal steering (`fp32`, `n=260` families, 3 backbones) | Nuisance-balanced CAA (`balCAA`) | **KILLED (`KILL_QSCCI_STEP0`):** `k=1==k=16` traced to `bf16` `1/64` quantization. Passed Yes/No on `Qwen2.5-1.5B` only (`1/3` models; failed `SmolLM2-1.7B` & `phi-2`). On `unseen_codes_alpha_omega`, `QSCCI` collapsed (`+0.011` / `-0.180`) while `balCAA` generalized (`+0.171` / `+1.833`); on `reversed_labels`, `phi-2` `QSCCI` tracked `" Yes"` surface (`-0.082`) instead of truth (`balCAA` `+0.597`). | `artifacts/pv7_w25_03_qscci_step0_fp32/w25_03_qscci_step0_manifest.json` |
| **`W26-04`** | 48-Cell Hubness Law (`float32`, `int8`, `binary_1bit`, `truncated_512d` × 4 corpora × 3 backfills; `CenteredProcrustes` only) | 2-param per-source affine calibration + QB-Norm + RRF | **KILLED (`KILL_HUBNESS_LAW`):** While `CSLS` beat `QB-Norm` (`+9.88 pp`) and selective refresh cut churn traffic `9.33×`, a **2-parameter per-source affine fit recovered `95.8%` (`float32`), `94.1%` (`int8`), and `90.8%` (`512d`)** of `CSLS`'s gain (`80.9%` on `1-bit` binary), and `Spearman(k-skew, gain) = 0.275 < 0.60`. | `artifacts/pv7_w26_04_hubness_law_spark2/w26_04_hubness_law_manifest.json` |
| **`W25-06` & `W27-03`** | `PM-6` semantic (`ZeroMu`) & white-box (`Qwen2.5-1.5B` residual linear probe) exfiltration guards (`N=3,000` & `N=2,400` fresh benigns × 6 attack suites) | Zero-ML `NFKC + UTS #39 + Base64/ROT13/leet + pruned 3/5-tok + char 5-gram` floor | **LEARNED ARMS RETIRED:** Zero-ML floor achieved `0.167%` FPR (`4/2,400`, `CP_UB95=0.381%`) and `0.0%` bypass on verbatim, homoglyph, and Base64/ROT13/leet (`40.5%` mean adaptive bypass; `7.0%` `<5`-tok chunking). `ZeroMu` reduced adaptive bypass by **`+0.5 pp`** and residual probe by **`+0.2 pp`**, retiring both (`< 10 pp` rule); `<5`-tok chunking (`6.5% > 5%`) keeps `PM-6` adaptive gate open. | `artifacts/pv7_w25_06_pm6_g6_spark3/w25_06_pm6_g6_manifest.json`, `artifacts/pv7_w27_03_pm6_adaptive_exfil_rtx3090/w27_03_pm6_adaptive_exfil_manifest.json` |
| **`W25-05`** | `#48` FAISS HNSW pre-filter pushdown on `92,811` full real docs (0 qrel filtering) | Filterable HNSW + post-filter oversampling | **SELECTIVITY-GATED:** Post-filter faster at `adm >= 0.20` (`0.61×–0.92×`); pre-filter pushdown wins at **`adm = 0.05`** across `3/3` full corpora (**`1.49×` p50 `5.83 vs 8.67 ms`**, **`1.42×` p95 `7.38 vs 10.51 ms`**) at matched `nDCG@10` (`0.38391`). | `artifacts/pv7_w25_05_hnsw_full_real_spark3/w25_05_hnsw_full_real_manifest.json` |
| **`W25-04`** | `#62` pooled agent loops (`1,120` cells, `26,880` jobs, `129,153` live `vLLM` calls) | `full_restart_rebuild`, `Conv+EK5`, sham-reopen SLA (`560 ms`) | **HOLD (`7/8` gates; `W27-02` gated):** Crushed `full_restart` (`+173.88` vs `0.00` JPM at `pool=4`, `0` errors, `0` stale commits), beat `Conv` at `pool=8`, but raw `178 ms` vLLM calls missed sham-reopen `560 ms` SLA (`1,618 ms`) without `W24-06`'s `29 ms` pre-filter. | `artifacts/pv7_w25_04_b62_vllm_spark2/pv7_w20_d13_manifest.json` |
| **`W24-14`** | `W22-08` capture-recapture audit defect estimator (independent blind coder) | Seeded/held-out miss validation & overlap stability | **KILLED (`K2` confirmed independently):** Blind second coder reproduced instability / overlap violation; permanently closed. | `docs/research/wave24_master_queue_2026-11-26.json` |

---

## 5. Master Index of Key Wave 24–27 Files

- **Master Queues & Scorecards:**
  - `docs/research/wave24_master_queue_2026-11-26.{md,json}` (`13/13` Wave 24 automated lanes)
  - `docs/research/wave25_plus_master_queue_2026-11-26.{md,json}` (`Waves 25–28` master ledger)
  - `docs/research/COMPLETE_64_BUNDLE_EVIDENCE_SCORECARD.json` (`Bundles #1–#64` machine scorecard)
  - `docs/research/WAVES_1_TO_21_MASTER_EVIDENCE_AND_PR_INDEX.md` (`Waves 1–22` bundle & PR index)
- **Lean 4 + `nanoda_bin` Formal Suite:**
  - `formal/ChelatedFormal/Spec.lean` & `formal/spec_sha256.json` (SHA-locked Wave 14A umbrella)
  - `formal/ChelatedFormal/Theorem5_IncrementalWFS_CW1.lean` (`W24-11` `CW-1` splice soundness)
  - `formal/ChelatedFormal/Theorem6_VerifiedConeAndPCC.lean` (`W26-02` verified cone & zero-FP PCC certs)
  - `formal/ChelatedFormal/Statement.lean` & `formal/nanoda_manifest.json` (`37` exported roots, `25,604` kernel-checked declarations)
- **Wave 25–27 Experiment Runners:**
  - `scripts/run_w25_02_trn_gate0_falsifier.py`
  - `scripts/run_w25_03_qscci_step0_fp32.py`
  - `scripts/run_w25_05_hnsw_full_real_corpora.py`
  - `scripts/run_w25_06_pm6_g6_powered_ablation.py`
  - `scripts/run_w26_01_operator_exact_deltanet_and_router.py`
  - `scripts/run_w26_02_pcc_ek5_million_stress.py`
  - `scripts/run_w26_03_voi_and_real_cascades.py`
  - `scripts/run_w26_04_hubness_law_and_churn.py`
  - `scripts/run_w27_01_retractbench_5probes.py`
  - `scripts/run_w27_03_pm6_adaptive_exfil.py`
