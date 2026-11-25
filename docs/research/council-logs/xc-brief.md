# Tri-Model Council — Cross-Check Phase Brief (2026-11-21)

**Council Roster:** `gemini-4-carbon` (`xhigh`), `gemini-4-neon` (`xhigh`), `gpt-6.2-astra` (`xhigh`)  
**Current Stage:** **Cross-Check Phase** (Each model independently verifies, challenges, and cross-checks the **other two models'** Round 0 reports before entering Refinement Rounds R1 → R2 → R3).  
**Read-Only Rule:** Do **NOT** edit any repository files during Cross-Check. Run only read-only in-memory Python/inspection probes. All code, whitepaper, and HTML edits occur in the consolidated Implementation Pass after R3.

---

## 1. Round 0 Reports on Disk to Cross-Check
- **Carbon XHigh R0:** `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1\docs\research\council-logs\r0-carbon.md`
- **Neon XHigh R0:** `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1\docs\research\council-logs\r0-neon.md`
- **Astra 6.2 XHigh R0:** `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1\docs\research\council-logs\r0-astra.md`

## 2. Updated Ground-Truth State Since Round 0 Launch
1. **Pristine Byte-Exact Archive Remediated Non-Destructively (Astra R0 §1):**
   - Original damaged copy `archive/pristine-b37e6a55/` (UTF-8 BOM + stripped LFs from PowerShell pipe) left untouched as forensic evidence.
   - New byte-exact read-only export created at `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1\archive\pristine-b37e6a55-exact\` with `PRISTINE_MANIFEST.json` (`15/15` files SHA-256 matched to annotated tag `baseline/pre-fix-pristine-2026-11-21` @ `b37e6a5523f8647c0d682178628db1f1545dc1ed` and `ast.parse`-verified).
2. **Rigor-R3 Flagship #2 Receipts Landed on Disk (`artifacts/flagship1-night1/20261121-rigor-r3/`):**
   - `f2_ek5_v2_1m/` (`ALL_SHA256_OK` `7f306adb7a558626`): `1000/1000` parity (325 `CONTESTED`), `1M` nodes `0.0275ms` vs `3325.25ms` (`120,918×`), dense hub `0.01%..60%` (`4,609×..1.43×`).
   - `f2_unified_zs_locomo/` (`ALL_SHA256_OK` `caf58b0f40f6d913`): Part A (`C0/C1/C4_v2 = 150/150`, `C2b = 120/150`, `C4_v2` `-44%` prompt tokens); Part B `LoCoMo` (`M4c/M4d = 22/80` answerable vs `M1b_bm25_jaccard = 19/80`, `LCB = -5.0 pp`, `M0 = 10/80`).
3. **Still Running in Background (`gx10-37c3`, `task-821`):**
   - `f1_4096_ood` (`MiniLM 384 → bge-large 1024 → Qwen3-4B 2560 → Qwen3-8B 4096` + `CoSQA` / `LegalBench` / `LegalQuAD` + Centered Affine Ridge + linear & $2^r-1$ nDCG@10) finished `D2560` and is encoding `D4096`; its verified receipts will enter Refinement Round 1/2.

---

## 3. High-Priority Cross-Check Adjudication Matrix (Every Model Must Audit & Verdict)

For every item below surfaced by the other two models, independently verify via code/math/artifact inspection and issue an explicit verdict (**`CONFIRMED`**, **`CONFIRMED_WITH_NUANCE`**, or **`DISPUTED`**) with exact evidence:

1. **Carbon `F2-NEW-1` (Zombie Reinstatement on 2-Step `SUPERSEDE` $v_1 \to v_2 \to v_3$):**
   - Does `HEAD` `iq04_temporal_evidence.py` resurrect stale $v_1$ to `ADMIT` when $e_3 \xrightarrow{\text{SUP}} e_2 \xrightarrow{\text{SUP}} e_1$?
   - Does Carbon's **Two-Axis Epistemic-Temporal Semantics** (compute grounded validity via `REFUTE` + positive deps first; then let any epistemically `VALID` node exert transitive `SUPERSEDE` force even if itself superseded) hold under *all* edge combinations—including **cycles of `SUPERSEDE`**, **`SUPERSEDE` + `REFUTE` mixed cycles**, **`CONTESTED` superseder** ($e_2$ is `CONTESTED` and supersedes $e_1$: what should $e_1$'s state be?), and **derived copies (`DERIVES_FROM`/`COPY_OF`) of a superseded node**? Find any edge case that breaks Carbon's 2-stage proposal!
2. **All Three Models (`EK5 v2` Kripke–Kleene vs. True Van Gelder–Ross–Schlipf WFS):**
   - Neon + Carbon recommend upgrading to **true WFS (Greatest Unfounded Set elimination)** so circular `DERIVES_FROM`/`COPY_OF` loops collapse to `AUDIT_ONLY` and cannot poison grounded claims `C`. Astra notes `200/256` (2-node) and `2,352/4,096` (3-node) disagreements between Kripke-Kleene and WFS and asks the council to adjudicate the exact semantics + incremental complexity of maintaining GUS incrementally!
   - **Adjudicate:** Can incremental WFS with GUS elimination (plus two-axis `REFUTE` vs `SUPERSEDE` + DRed-style support counters + bitemporal cutoffs) still run in bounded incremental time with `100%` parity against an **independent** textbook alternating-fixpoint WFS oracle? Give the exact data structures, edge cases (including deletions/rewinds and initial-insert deltas), and complexity!
3. **Flagship #1 Mathematical Adjudication:**
   - **Down-projection conformal scale (`Neon §2.2.2`, `Astra §3`):** Verify `rad_x = ||X_c Q||_F` vs `||X_c||_F` vs Carbon's **Proposition 1b** (moment-matching $s_{\text{rad}}$ vs Umeyama $s_{\text{LS}} = \operatorname{tr}(\Sigma)/\|X_c\|_F^2$): does Carbon's $s_{\text{LS}} = \bar{\rho}_{\text{canon}} s_{\text{rad}}$ identity hold exactly (check units/singular values carefully!)? What happens on constant-source / rank-deficient $n < d$ (`A2c-Subspace`)?
   - **Non-monotone chains (`Neon` `384→768→384` `cond=147.21` via Jordan principal angles + `A2c-Polar` vs `Carbon` Theorem 1 condition `min d_k >= min(d_m, d_0)`):** Wait — check Carbon's claim in `WP-2` (`r0-carbon.md:175`): *"prove full-rank $\kappa = 1.00$ iff dimensions are monotone OR $\min_k d_k \ge \min(d_m, d_0)$"*!! Is Carbon's "OR" clause **FALSE** on `384→768→384` (where $\min_k d_k = 384 = \min(384, 384)$, yet Neon proved `cond = 147.21`)?! Audit that rigorously!
   - **Backfill (`Neon`: `BF20 vs A3+CSLS = 9/18` + `csls_k` ignored; `Carbon`: CDF upper-tail clamping inverting rank 1 vs 2 + `B2-OldSpace` kNN interpolation):** Verify both and adjudicate whether to upgrade backfill (`B4`) or narrow Flagship #1's flagship claims to zero-write `A2c-Subspace`/`A3-Subspace` + `TRA-1`.
   - **Modality Offset (`Carbon F1-NEW-4` `A2c-Q` on `E2` e5 prefixes) & Voronoi-Weighted `P6` (`Neon Crumb B` `A2c-IW`) & Shared-Anchor Telescoping (`Astra §3`):** Rank which successor crumbs are highest ROI to test cleanly in our Implementation Pass.
4. **Flagship #2 Live Benchmark & Forensic Adjudication:**
   - Verify **Astra's** `150/150` observable-field solver on Part A Zero-Shortcut, `loc_0_28` judge-cache race (25 rows / 3 keys), `conv-26` (117) + `conv-30` (3) sampling bug, compound ID `D8:6; D9:17`, and `15/56` distractor-evidence overlap.
   - Verify **Neon's** `Crumb A` (`LongMemEval-S` Regime B `"from earlier chat"` cue leak explaining Regime B > Regime A on 3 KU items) + Part A `valid_time <= retracted_tx` axis mix + `val_sup_cnt` popularity re-introduction + `M1c_hybrid_jaccard_dedup` omission.
   - Verify **Carbon's** `F2-NEW-6` (`2.56 pp` NVFP4 dynamic-batching jitter between LongMemEval v1 and v2) and `M4d` overwrite bug.
5. **Reconcile `PM-1..PM-5`, Whitepaper v2 Structure & Presentation HTML v2 Architecture:**
   - Identify every point of tension across Carbon, Neon, and Astra's proposed `PM-1..PM-5` gates, whitepaper structure, and HTML redesign (e.g., `PM-1` clean margin `-1.0 pp` vs `-1.5 pp` vs `2.56 pp` NVFP4 jitter floor; single-source JSON/MD build script for HTML + MD tables; exact ledger inclusion rule across all 14 bundles).

## 4. Required Output Structure
1. **Cross-Check Verification Table** (one row per major finding from the other two models: `Source`, `Claim`, `Verdict`, `Independent Evidence / Counterexample`).
2. **Errors Caught in Peer Round 0 Reports** (be ruthlessly honest — catch any flawed proof, overstatement, or subtle edge case in your peers' R0 proposals!).
3. **Unified Resolution Proposals for Refinement Round 1** (concrete, mathematically airtight positions on: Two-Axis WFS + GUS + DRed, Flagship #1 exact Theorem 1 + Subspace + Backfill, Flagship #2 clean zero-shortcut v2 + blind LoCoMo/LongMemEval protocol, `PM-1..PM-5` harmonized gates, Whitepaper v2 & HTML v2).
