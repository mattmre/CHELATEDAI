# Tri-Model Research Council Dossier — Round 0 Brief (2026-11-21)

**Council Roster:** `gemini-4-carbon` (`xhigh`), `gemini-4-neon` (`xhigh`), `gpt-6.2-astra` (`xhigh`)  
**Protocol:** Round 0 (Independent Review & Inputs) → Cross-Check (Each Model Audits the Other Two) → 3 Iterative Refinement Rounds (R1, R2, R3) → Implementation Pass.  
**Pristine Preservation Guarantee:** All original files and artifacts used in the original findings are immutably preserved at git tag `baseline/pre-fix-pristine-2026-11-21` (`b37e6a55`) AND copied read-only into `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1\archive\pristine-b37e6a55\`.

---

## 1. Current Artifacts on Disk to Inspect
- **Pristine Code Archive (`b37e6a55`):** `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1\archive\pristine-b37e6a55\`
- **Fixed Code (HEAD, post-audit):** `spherical_migration_bridge.py`, `m1_baseline_ladder.py`, `iq04_temporal_evidence.py` (with `EK5 v2` 3-valued Well-Founded Semantics + persistent `IncrementalEvidenceStore`), `tests/test_spherical_migration_bridge.py`, `tests/test_iq04_temporal_evidence.py` (12/12 passing in `0.092s`)
- **Doctoral PhD Whitepaper (Draft v1):** `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1\docs\research\doctoral-whitepaper-chelated-memory-and-spherical-migration-2026-11-21.md`
- **Public-Site Presentation HTML (Draft v1):** `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1\docs\research\chelated-memory-spherical-migration-whitepaper-presentation.html`
- **Flagship #1 & #2 Full Reports (Draft v1):** `docs/research/flagship1-conformal-spherical-migration-bridge-full-report-2026-11-21.md`, `docs/research/flagship2-iq04-temporal-evidence-correction-full-report-2026-11-21.md`
- **All 11 Verified Pristine Spark Artifact Bundles:** `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1\artifacts\flagship1-night1\`

---

## 2. Verified Summary of Prior Independent Audits (Astra 6.2 XHigh Audits 1–3)

### Audit 1 — Flagship #1 (`CONFIRMED_WITH_REVISIONS`)
- **Verified:** All 8/8 manifest SHA-256 hashes, all 426 numeric entries (0 rounding diffs), 6/6 bridge tests, 113/113 quarantine validator, and **24/24** uniform `A2c` sweep over Ridge across 6 BEIR domains × 4 hops.
- **Critical Defects Caught:**
  1. `benchmark_utils.py` computes **linear graded gain (`rel`)**, NOT exponential `2^rel - 1` as written in Draft v1.
  2. Night-1 Gate 3 overall verdict was `TRA1_OR_M4_NEEDS_REVIEW` (worst-family `LCB = -0.045658` vs `-0.02`), NOT `TRA1_PASS / M4_KILL` (though `0/8` hostile wrong-space resolver authorizations held).
- **Major Caveats Caught:**
  1. Single-hop Ridge & `A3` shrinkage (`11/18` chose `0.50`, `7/18` chose `0.20`) were tuned on DEV qrels; FiQA/Quora 40k doc cap retained test positives before split; Gate 1 chose best ordinary arm on REPORT (oracle frontier).
  2. Theorem 1 (`cond = 1.00`) holds on monotone-dimension chains (`768→768→384→384→384`), NOT arbitrary non-monotone chains (`2→3→2` gives `cond=2`, `3→2→3→2→3` gives singular values `1, 0.5, 0`). Constant target gave `scale=0` → `cond=0` (now fixed to `inf`).
  3. Ridge comparator was uncentered without intercept and used censored `1e-12` SVD floor (already `~1e12` at Hop 1). Needs **Centered Affine Ridge** comparator + fresh label-independent holdout.
  4. 20% backfill is an index-write budget, NOT a 20% encoding budget when `B2/B3` transforms all new docs.
  5. Strict all-3-seed `LCB95 > 0` sensitivity counts: `10/18` (`A3` vs Ridge), `8/18` (backfill), `12/24` (uniform `A2c` vs Ridge; `14/72` seed LCBs nonpositive).
  6. Cutoff tie-break partial-sort bug on 160 equal scores (`096` vs `000` — now fixed in code & tested); `run_night1_flagship1_expanded.py` needed explicit `match_radius=False` pin (now fixed).

### Audit 2 — Flagship #2 (`DISPUTED` on Broader Claims; Echo-Stress Win Confirmed)
- **Verified:** All 4/4 manifest SHA-256 hashes, all numeric tables & bootstrap bounds, 4/4 unit tests, and genuine LongMemEval-S Regime B echo-stress win (`M4b` **58/78** vs BM25 **38/78**, Hybrid **36/78**, ScaleMAS surrogate **26/78**, Recency **37/78**).
- **Critical Defects Caught:**
  1. Pristine `EK5` `sparse_invalidate_incremental` failed three executable counterexamples: (a) dropping external independent refuter `B` readmitted `X`, (b) depth-`H` `COPY_OF` horizon mismatch, (c) mutual refutation `A ↔ B` parity oscillation (`H=7` vs `H=8`). Pristine 1,000-graph fuzzer confirmed **39.3%** divergence and **5.6%** oscillation!
  2. Reported `4.97%` work ratio was a traversal counter while `sparse_invalidate_incremental` rebuilt adjacency over all visible edges per call (`O(|E|)`), and `C4`/`M4b` never invoked `EK5`.
- **Major Caveats Caught:**
  1. Stage 2C had a **150/150 newest-timestamp shortcut**, and `C4` still used synthetic `role` tags internally (only the prompt omitted them).
  2. LongMemEval-S Regime B used gold-directed fixture construction (6 fixed recap prefixes on `70/78` KU items; `8/78` got 0 echoes), tested open-text dedup (`M4b`, tuned in-sample after v1!) rather than the signed graph engine, and a simple **BM25 + Jaccard-0.58** baseline already recovered **75/78** latest sessions vs **77/78** for `M4b`.
  3. LongMemEval-S grader had nondeterminism on identical answers (`2698e78f_abs`), rejected abstention on `6 KU + 4 clean` `_abs` items, and substring fast-path accepted contradictory `"7:00 pm (or 6:00 pm)"`. Exact total live calls across 2B/2C/LME-v2 = `2,616` answer calls + `898` judge calls = `3,514` requests.
  4. `C1`/`M2` comparator disclosure needed exact detail (`M2` has no citation edges/decay; `C2` applies both cutoffs; `C3` doesn't dedup valid copies).

### Audit 3 — `$panel-of-experts` on Doctoral Whitepaper & Presentation HTML (15 Findings)
- **Verdicts:** Whitepaper Draft v1 = *Major Revision Required*; Presentation HTML Draft v1 = *NO-GO for Public Release*.
- **Additional Presentation & Protocol Defects Caught:**
  1. Print CSS strips pre-release warning from page 1, hides dataset selector, leaves full table collapsed, shows only 1 dataset despite "Across All 6" title.
  2. Mobile (`375px`) had `629px` horizontal overflow and ~`3.1px` SVG labels; orange `KILLED` badge contrast `3.76:1` (< WCAG 4.5:1); tabs lack `aria-selected`; literal `\!\bigl` / `\bigr` in HTML formula; 9 last-decimal JS rounding deltas; Figure 2 mixes overall acc vs KU acc and omits Regime A & error bars.
  3. Ledger had 11 rows labeled "All 10", truncated `f2_iq04` SHA-256 (full: `148d80bf3faadd77f9ad0f9eeb1f1a089d8e87b89025c9f645d4a685ad7e619b`), missing `s1` wall time (`1226.40s`), and manifests didn't hash raw CSVs.
  4. Bootstrap helper uses 2.5th percentile (two-sided 95% CI lower bound, not one-sided 5th percentile); Stage 2C uses 10,000 draws; `PM-1..PM-5` lacked formal denominators, blind multi-generator/multi-judge criteria, and consistent HTML/MD specs.

---

## 3. Pristine Forensic Successor Crumbs & Active Extreme Dual-Spark Campaigns (In Flight)
1. **Crumb #1 (`EK5 v2` Exact 3-Valued Well-Founded Semantics + Persistent `IncrementalEvidenceStore`):** Implemented & verified **`1000/1000` (100.0%)** exact bitwise parity (325 `CONTESTED`-cycle graphs, vs `60.7%` pre-fix) and **`0.0159 ms` (723× speedup at 10k)** / **`0.0231 ms` (9,455× speedup at 100k)** real persistent wall-clock speedup on `gx10-6724` (1M-node + dense hub phase transition + Zero-Shortcut live suite + fresh held-out ACL 2024 `LoCoMo` currently running!).
2. **Crumb #2 (Non-Destructive Cross-Session Supersession Tagging `M4d`):** Forensics showed 19/20 Regime B `M4b` misses ALREADY had the latest session in top-5 (5 abstention bugs + 14 cases where flat bullets caused the LLM to pick the earlier stale value). `M4d` collapses strictly cross-session echoes (`session_id(p) != session_id(q)`) AND tags superseded vs newer cross-session turns non-destructively (`[EARLIER RECORD]` / `[NEWER UPDATE]`), being evaluated cold on fresh held-out `LoCoMo` against `BM25 + Jaccard-0.58`.
3. **Crumb #3 (`384→4096` Extreme Dimensionality & Code↔Legal OOD):** `gx10-37c3` is running `MiniLM (384) → bge-large (1024) → Qwen3-4B (2560) → Qwen3-8B (4096)` across `SciFact`, full `20,604`-doc `CoSQA` (Python code), `LegalBench Consumer Contracts`, and `LegalQuAD` (zero label-conditioned capping), comparing Uncentered Ridge, Centered Affine Ridge, Centered Orthogonal Procrustes (`A2`), `A2c`, `A3`, and `A3-Subspace` under BOTH linear and `2^rel−1` complete-qrels nDCG@10.

---

## 4. Round 0 Mandate for Each Council Model
Independently inspect the dossier, pristine vs. fixed code, whitepaper v1, and presentation HTML v1, and provide:
1. **Rigorous Critique of Findings & Positioning:** Where is the real, publishable Tier-1 scientific core for Flagship #1 and Flagship #2? Should Flagship #2 pivot toward structured multi-agent/tool-trace truth maintenance (`EK5 v2` WFS) vs. conversational RAG, or unify both cleanly?
2. **Hidden Successor Crumbs & Algorithmic Opportunities:** What deeper mathematical or architectural successor opportunities hide inside the pristine failure modes (e.g., `ArguAna` up-projection, `384→4096` subspace geometry, 3-valued WFS + retrieval ranking integration, dense-cascade mitigation, cross-session contradiction detection)?
3. **Advised Next Steps, Experimental Design & Proof-Model Architecture (`PM-1..PM-5`):** Concrete, falsifiable improvements to the Proof Model, benchmarks, statistical testing, and release gates.
4. **Whitepaper & Presentation HTML Refinement Blueprint:** Specific structural, mathematical, visual, and editorial upgrades needed to reach doctoral dissertation + public website release quality.
