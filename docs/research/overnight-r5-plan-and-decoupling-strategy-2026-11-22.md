# Post-R4 Strategic Assessment, Paper Decoupling Architecture & Overnight R5 Execution Plan (2026-11-22)

**Base Immutable Checkpoint:** `ceb67cfad9a69d6252150ff73fbee43127150488` (tagged `checkpoint/r4-complete-pre-overnight-r5-2026-11-22` on GitHub, mirrored byte-for-byte with SHA-256 verification in `archive/r4-complete-ceb67cfa-backup/` and `/home/mattmre/chelated-night1/backups_r4_ceb67cfa/` on both `gx10-6724` and `gx10-37c3`).  
**Non-Destructive Guarantee:** Zero R3/R4 artifacts, scripts, or whitepaper files are modified or deleted; all R5 code (`em4_soft_telemetry_routing.py`, `scripts/run_overnight_r5_*.py`) and outputs (`artifacts/flagship1-night1/20261122-overnight-r5/`) are **100% additive** with per-stage atomic SHA-256 receipts.

---

## 1. Impact of Carbon's Strategic Advice on Next Steps

### A. Decoupling the Papers / Releases (Immediate Strategic Win)
Carbon's recommendation to separate the **Systems Monograph (Track A)** from **Unlinked Conversational Extraction & Benchmark Gating (Track B)** directly protects our strongest contributions:
- **Track A — Self-Contained Systems Monograph (Empirically Complete Today):**
  1. **Flagship #1 (`spherical_migration_bridge.py`):** `EM-2` (`A2-ZeroMu` zero-write & 1-float zero-reindex bridges winning `16/18` cross-domain OOD cells up to `51.2×` CAR and `18/18` multi-hop chains), `A2c-Polar` (Theorem 1 interior-peak spectral repair `368.6 -> 1.0000`), `A2g` (`GeodesicPolarBridge` $C^1$ soft-collar), `A3-EffRank`, `B4 + CSLS + B2-OldSpace`, and `TRA-1` (`8/8` wrong-space blocks).
  2. **Flagship #2 Formal Core (`ek5_v3_stratified.py`):** Stratified $K=4$ two-axis Well-Founded Semantics engine with **`0 / 381,497` oracle mismatches**, **`0.904 ms` `100k` persistent service p95** (`PM-5` CLEARED, `27.6×` margin), `7.94 ms` `1M` power-law cyclic p95, `10k` hubs `|D| = (0, 0, 0)`, and passive `EM-3` telemetry.
- **Track B — Conversational Memory Extraction & Coupled End-to-End Benchmarks (`PM-1a/1b`, `PM-2`, `PM-3`, `PM-4`):**
  Executed additively tonight in **Overnight Campaign R5** across both Dual Sparks without risking or delaying Track A.

### B. Per-Item Forensic Proof of Carbon's `-1.33 pp` `PM-1b` Hypothesis (`EM-4`: `M4f` / `M4g`)
Inspecting `f2_r4_locomo_300_rows.csv` row-by-row confirmed Carbon's exact diagnosis:
- On `LoCoMo-300` Regime A Clean, `M4e` had `6` wins and `10` losses vs `M1_hybrid_flat` (`-4 / 300 = -1.33 pp`), whereas `M1c_hybrid_jaccard` (`Jaccard >= 0.58`, no loose dense cutoff) had **`0.00 pp` loss (`0.3833` vs `0.3833`, `evidence_recall_at_5 = 0.630`)**.
- **Root Cause:** `M4e` hard-pruned cross-session turns whenever `MiniLM cos >= 0.72` even with **zero explicit `EK5 v3` `REFUTE` edge** and low lexical overlap (`Jaccard < 0.35`). This caused **3 direct gold-evidence recall drops** (`loc_1_39`, `loc_3_2`, `loc_9_0`: `0.630 -> 0.610`) and **3 abstention hallucinations** (`loc_2_152`, `loc_7_226`, `loc_9_163`) from backfilling rank-7 noise into `top_k=5`.
- **Additive Solution (`em4_soft_telemetry_routing.py` — `M4f_soft_telemetry_fallback` & `M4g_ek5_soft_telemetry_recency`):**
  1. **Hard-drop ONLY** on explicit `EK5 v3` `AUDIT_ONLY` edges (`REFUTE`/`SUPERSEDE`) OR verified duplicate copy-echoes (`stripped_jaccard >= 0.58` or `cos >= 0.86` confirmed `ECHO_COLLAPSE`).
  2. **De-spoof recap timestamps:** Anchor every echo cluster to `origin_ts = min(ts)` so repeated intervening digests cannot impersonate recent updates.
  3. **Route via `EM-3` Soft Telemetry Tags:** Preserve conflicting cross-session updates side-by-side with `[EM-3 TELEMETRY: EARLIER RECORD (echoed Nx)...]` vs `[EM-3 TELEMETRY: LATEST DATED UPDATE]` tags and gate backfill at `hyb >= 0.58 * top_hyb`.

### C. Exact Gate-by-Gate Bottleneck Breakdown (`PM-1a` .. `PM-4`)

| Gate | What Specifically Was Gating It After R4? | Overnight R5 Action Tonight (Dual Sparks) |
|---|---|---|
| **`PM-1a`** | 6/8 `Zero-Shortcut v2` families had `<= 5` events (`<= top_k`), letting flat `C0..C2` fit the whole graph into prompt (`249` vs `150` tokens). | **`gx10-6724` Stage 1:** Run **`Zero-Shortcut v3` (`N=240 = 8×30`, `16` records/entity > `top_k=5`)** alongside `ZS v2` to test `LCB95 > +15 pp` over all non-WFS arms. |
| **`PM-1b`** | (1) `-1.33 pp` clean gap on `LoCoMo-300` (vs `-1.00 pp` bar) & tie with `M1c` on blind echoes; (2) full unmodified `LongMemEval-S N=500` clean (`70` `single-session-user`) not yet run end-to-end. | **`gx10-6724` Stages 2 & 3:** Run `LoCoMo-300` (3 regimes incl. paraphrased update+echo) + **full `LongMemEval-S N=500` clean & `78`-item KU echo stress** with `M4f` & `M4g` alongside `M0..M4e`. |
| **`PM-2`** | Needed a **joint coupled run** combining live `384↔4096` / `3-hop` / `E0→E3→E1` bridges + `12` `TRA-1` hostile probes + simultaneous `EK5 v3` supersession & `6×` echo stress. | **`gx10-37c3` Stage 1:** Run coupled persistent service (`7` arms × `3` seeds + `N=240` live QA + `12/12` `TRA-1` probes) over cached `SciFact` + `Legal_Contracts` (`D384..D4096`) + `BEIR`. |
| **`PM-3`** | Live pinned upstream `mem0ai` / `graphiti-core` / `ScaleMAS` execution (`mem0ai` requires per-turn LLM extraction calls; `ScaleMAS` has no public PyPI/GitHub release). | **`gx10-37c3` Stage 2:** Isolated `.venv-pm3` running **unmodified `mem0ai`** (`nemotron-37c3` + `MiniLM` + local Qdrant) on `ZS K=4` (`N=120`) + `LongMemEval` (`N=80`), probe `graphiti-core`, and record formal `UNAVAILABLE_UPSTREAM_ARCHIVE` attestation for `ScaleMAS`. |
| **`PM-4`** | (1) Single model family (`nemotron-lightning`) for generation & judging; (2) `100`-item human audit requires a real human grader. | **`gx10-6724` Stage 4:** Run **`Qwen/Qwen2.5-7B-Instruct`** in `.venv` (51 GiB free GB10 RAM, zero Docker changes!) for a **true `2×2` `{Nemotron, Qwen2.5} × {Nemotron Judge, Qwen2.5 Judge}`** $\kappa$ matrix (`N=390`), plus generate a **sealed blind `100`-item human audit packet + scorer**. |
