# Dual-Spark Research & Validation Portfolio Summary (2026-11-21)

**Branch:** `fix/stage0-metric-lineage-and-m1-ladder` (base `e40c0883`)  
**Hardware:** NVIDIA DGX Spark `gx10-37c3` (`NVIDIA GB10`) & `gx10-6724` (`NVIDIA GB10`, `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`)  
**Metric Contract:** Canonical complete-qrels `mlr-v1` (`ndcg_at_k` graded gain), Quarantine v2 enforced (`113/113` PASS).

## 1. Flagship #1 — Closed-Form Spherical Migration Bridge (`SHIP / READY`)

- **Packaged module & tests:** `spherical_migration_bridge.py`, `tests/test_spherical_migration_bridge.py` (6/6 tests passing).
- **Winning closed-form stack:** `P6` Spherical k-Means++ anchors + `A2c` Conformal Radius-Matched Centered Orthogonal Procrustes (`y = scale * ((x - mu_x) @ Q) + mu_y`, `scale = ||Y_c||_F / ||X_c||_F`, exact `cond = 1.00`) + `A3` Shrinkage-Whitened (CORAL) Spherical Procrustes + `#355` CSLS + `B1/B3` Quantile-CDF Backfill Calibration & Exposure×Spherical-Distortion prioritization + hardened `TRA-1` ABI.
- **Strictly additive ablation kills:** Explicit 3rd-order cumulants (`A4`/`A6`) hurt 6/6 standalone and add `0.000` over Spherical KL (**KILL**); Nyström Kernel `P7` hurts `-0.08` to `-0.14` (**KILL**); synthetic off-manifold random probes `P1` hurt `-0.10` to `-0.14` (**KILL**); M4 QPP adds `0.000` net value (**KILL**).
- **Verified Generalization across 6 BEIR Domains (`SciFact`, `NFCorpus`, `FiQA2018`, unseen `ArguAna`, `SCIDOCS`, `Quora`) × 5 Encoders (`E0`–`E4`):**
  - Single-hop `A3` vs Ridge: **17/18** cells (`A3+CSLS` vs `Ridge+CSLS`: **17/18**; `A3` rescues narrow-cone collapse on **18/18**).
  - 20% Selective Backfill (`A3 + Quantile-CDF + Exposure×Distortion`) vs Night-1 `BF20`: **16/18** cells (`+0.030` to `+0.072`).
  - **4-Hop Lineage Chain (`E4→E3→E2→E1→E0`) with Conformal `A2c`:** **24/24** wins vs. Composed Tuned Ridge across all 6 BEIR domains × all 4 hops at exact **`cond = 1.00`** (vs. Ridge's `1.18e+12`–`5.49e+12`), curing the Hop-2 narrow-cone dip by `+0.029` to `+0.182` nDCG@10.
- **Receipts:**
  - `20261121-flagship1-r1/flagship1_verdict_2026-11-21.md`
  - `20261121-spherical-r1/spherical/spherical_ablation_summary.md`
  - `20261121-precipitated-r1/precipitated/precipitated_summary.md`
  - `20261121-tandem-r1/f1_expanded/f1_expanded_summary.md` (`manifest_sha256: 281ce6eab4557716...`)
  - `20261121-tandem-r2/f1_conformal_4hop/f1_conformal_4hop_summary.md` (`manifest_sha256: 78da59866ad65532...`)

## 2. Flagship #2 — Temporal Evidence Correction & Bounded Retraction (`IQ-04 / EK7 / EK5`, `BREAKTHROUGH CANDIDATE`)

- **Packaged module & tests:** `iq04_temporal_evidence.py`, `tests/test_iq04_temporal_evidence.py` (4/4 tests passing).
- **Stage 2A (Deterministic `EK5` + 800-Lifecycle `EK7` REPORT):**
  - `EK5` Sparse Invalidation: **PASS** (`exact_parity=True`, `max_work_ratio=0.0497 <= 0.25`).
  - `EK7` REPORT stale error: `C0/C1/C2 = 0.8571`, `C3_ordinary_bitemporal_tms = 0.1429` (100% failure on unlinked paraphrase copies), **`C4_iq04_candidate = 0.0000`** (`LCB95 = +0.1171` vs `C3`, `0.00 pp` clean recall loss).
- **Stage 2C (Hard Adversarial Live `nemotron-lightning`, 600 calls, 0 API errors, zero role-tag hints, 10–12 copy floods > `top_k=5` + hard lexical-overlap clean trap):**
  - `C0_bm25_flat`: `65.0%` stale error (`48.0%` overall acc)
  - `C1_scalemas_unsigned` (`SCALEMAS_ADAPTED_SURROGATE`): `66.7%` stale error (`46.7%` overall acc)
  - `C3_ordinary_bitemporal_tms`: `55.0%` stale error (`56.0%` overall acc)
  - **`C4_iq04_candidate`: `0.0%` stale error (`100.0%` overall acc, `100.0%` hard clean-trap acc, `-40.2%` prompt tokens vs `C3`, `LCB95 = +0.4583` vs `C3`)**
- **External Real-World Benchmark — ICLR 2025 `LongMemEval-S` (`xiaowu0162/longmemeval-cleaned`, all 78 `knowledge-update` + 50 `single-session-user` clean controls, 1,536 live `nemotron-lightning` calls, 0 API errors, strictly observable retrieval):**
  - **Regime A (Unmodified LongMemEval-S):** `M4b_iq04_echo_collapse_ranked` achieves `70.51%` KU accuracy (`98.72%` Latest-Session Recall@5), within `1.28 pp` (1 question, statistical tie) of unmodified baselines (`71.79%`), fixing the over-deletion observed in `M4` (`66.67%`).
  - **Regime B (Multi-Session Echo-Persistence Stress, 6 unlinked earlier echoes):** `M4b_iq04_echo_collapse_ranked` achieves **`74.36%` KU accuracy (`98.72%` Latest-Session Recall@5, `78.12%` overall)**, crushing every comparator with `LCB95 >= +14.1 pp`:
    - vs `M2_scalemas_unsigned` *(surrogate, `33.33%`)*: **`+41.03 pp` (`LCB95 = +28.21 pp`)**
    - vs `M1_hybrid_flat` (`46.15%`): **`+28.21 pp` (`LCB95 = +15.38 pp`)**
    - vs `M3_recency_hybrid` (`47.44%`): **`+26.92 pp` (`LCB95 = +15.38 pp`)**
    - vs `M0_bm25_flat` (`48.72%`): **`+25.64 pp` (`LCB95 = +14.10 pp`)**
- **Receipts:**
  - `20261121-tandem-r1/f2_iq04/f2_iq04_verdict.md` (`manifest_sha256: 148d80bf3faadd77...`)
  - `20261121-tandem-r1/f2_hard/f2_hard_verdict.md` (`manifest_sha256: f7960dd626f7f17a...`)
  - `20261121-tandem-r2/f2_longmemeval_v2/f2_longmemeval_verdict.md` (`manifest_sha256: c14845a78c6f1caf...`)

## 3. Science Runner-Up `IQ-01` — Commutator & Non-Normal Geometry Falsifier (`DEMOTE TO DIAGNOSTIC`)

- **Result (`168` real ordered adapter compositions across `SciFact`, `NFCorpus`, `FiQA2018`):**
  - Pre-outcome commutator norm tracks order asymmetry (`ρ = +0.5988` to `+0.7542`, modest edge over trivial norm on 2/3 datasets), **but** Henrici non-normality (`ρ = +0.224`–`+0.421`) and spectral condition (`ρ = +0.090`–`+0.553`) are beaten by trivial Frobenius deviation norm (`ρ = +0.365`–`+0.715`) for predicting composition harm.
  - **Decision:** Demote `IQ-01` to a cheap optional order-sensitivity diagnostic; do not promote to flagship.
- **Receipt:** `20261121-tandem-r1/iq01/iq01_falsifier_summary.md` (`manifest_sha256: 7467ab0681259733...`)
