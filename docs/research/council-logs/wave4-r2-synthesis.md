# Wave-4 Two-Round Frontier Advisory Council & Cross-Judgment Synthesis

**Date:** 2026-11-22  
**Advisory Roster (Read-Only Peer Audit + Cross-Judgment):**
1. **OpenAI Codex — `gpt-6.2-sol` (`reasoningEffort: "xhigh"`):** [R1 Audit](wave4-r1-sol62-xhigh.md) | [R2 Judgment](wave4-r2-judgment-sol62-xhigh.md)
2. **Anthropic Claude — `claude-opus-5-5` (`effort: "high"`):** [R1 Audit](wave4-r1-opus55-high.md) | [R2 Judgment](wave4-r2-judgment-opus55-high.md)
3. **xAI Grok — `grok-4.7` (`reasoningEffort: "xhigh"`):** [R1 Audit](wave4-r1-grok47-xhigh.md) | [R2 Judgment](wave4-r2-judgment-grok47-xhigh.md)
4. **Google Gemini — `gemini-4-neon` (`thought_level: "high"`):** [R1 Audit](wave4-r1-neon-high.md) | [R2 Judgment](wave4-r2-judgment-neon-high.md)

---

## 4/4 Unanimous Cross-Judgment Resolutions (`J1–J7`)

| ID | Unanimous Verdict | Binding Anti-Leakage & Methodological Guardrails |
|---|---|---|
| **`J1` Colab GRPO** | **`MODIFY` (Code Fix)** | Wire `colab_h100_70b_and_epistemic_grpo.py` directly to `ek5_wfs_rlvr_env.py`; enforce entity-, value-, template-family-, and seed-disjoint `TRAIN / DEV / EVAL` splits (`train ∩ eval = ∅`); exact canonical span / closed-lexicon `ABSTAIN` matching (rejecting negations & mixed commitments); format bonus `<= 0.05` paid **only** when valid parse **and** core reward `== 1.0`. Scope Whitepaper v4.0 strictly to offline verifier separation (`0/10,827` node mismatches) until H100 LoRA weights are trained. |
| **`J2` PM-2 Zero-Oracle** | **`MODIFY` (Code Fix + Rerun)** | Redact `entity`, `slot`, `family`, and `tenant` from evaluation query objects (`KeyError` on oracle access); route **every** arm (`Flat_No_EK5` included) through one unified retrieval-inferred slot policy; derive provenance flags from runtime trace assertions + 5-seed query-order shuffle EMA spread; rename `Query0W_CenteredTarget` $\to$ `DualFit_1Float_FixedMuX`; disclose `Native = 1.000` ceiling and `Tango LCB95`. |
| **`J3` D2-4096d & 1M** | **`MODIFY` (Code Fix + Rerun)** | Verify `ood_SciFact_D384.npz` row-for-row float32 cosine `>= 1 - 1e-5` against sorted BEIR `SciFact` IDs + SHA-256 digest; add DEV-only calibrated `z-score` and `RRF (k=60)` mixed-shard baselines alongside raw merge and `CSLS`; report exact-DOT vs HNSW recall@10 (`99.77%` mean / `90%` min @ `ef=32`, `100%` @ `ef=64`); standardize `D2-1M` to two-sided `2.5%–97.5%` CIs and inspect `QueryAffine = 0.0`. |
| **`J4` PM-1b Conv v3** | **`MODIFY` (Code Fix + Holdout Run)** | Credit clean `LME-500` `+7.8 pp` to `DualGran BGE-M3` (`EK5` clean $\Delta = 0$); freeze `R1` (closed-speaker-set attribution header, never forced abstention) + `R2/R3` (RRF gap adaptive session budget + rank-ordered session rendering) + `R4` (token-matched flat control) **strictly on `LoCoMo` convs `0–4` and hash-split `LME` DEV**, reporting **held-out `LoCoMo` convs `5–9` and `LME` TEST** as primary confirmatory metrics (with pooled `N=300/500` labeled secondary). |
| **`J5` Workbench & PM-4** | **`ACCEPT_CODE_FIX` + `MODIFY`** | Fix `demo_workbench/index.html` (`4096-d` raw Qdrant target vectors + bridged `384-d` query, `99.77%` mean / `90%` min @ `ef=32`, `100%` @ `ef=64` across all tables); bind `pm4_human_grader.html` `localStorage` to `item_id + content_hash`, blind `dataset`/`qid`, fail closed on missing questions, and include preference rubric parity; reject dual `YES+NO` parses and add `LoCoMo` cluster-bootstrap + stratum CIs to `N=200` conversational $\kappa = 0.8400$ (`CI95` crosses `0.80`, isolated from old-rubric structured rows). |
| **`J6` Demo D3** | **`MODIFY` (Ablation + Scope)** | Pre-declare complete-linkage vs single-linkage copy clustering comparison, allow cross-value copy checks, report `{attack, correct, abstain, other}` + two-sided Clopper–Pearson 95% bounds per family (`0/20 <= 16.84%`, `0/80 <= 4.51%`), and foreground template co-design + trusted bus tiers. |
| **`J7` Whitepaper v4.0** | **`ACCEPT_DISCLOSURE` (Post-Freeze)** | Commit updated scripts before final reruns so `git_sha` binds immutably; integrate all 8 2025–2026 comparators + `TRA-1` vs `vec2vec` boundary; state `ZeroMu` trails uncentered Procrustes by `~1 pp` in-domain under tight `std=0.038` cones while winning `16/18` OOD + `12/12` multi-hop + `4`-byte HNSW-DOT + `CSLS` backfill; retain `PRE-RELEASE` banner until human `PM-4b` completes. |
