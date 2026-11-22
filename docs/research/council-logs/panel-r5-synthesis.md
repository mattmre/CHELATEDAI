# Post-R5b 4-Model Highest-Reasoning Frontier Panel Synthesis

**Date:** 2026-11-22  
**Panel Roster (Maximum Reasoning Tiers):**
1. **OpenAI Codex — `gpt-6.2-astra` (`reasoningEffort: "ultra"`):** `thread://v2/c5bb8594-9c79-4948-9251-bd83e8527cf6/thread:delegated-task:command%3Amcp%3A3b84a70c-9824-441c-8750-b9902399a932%3Adelegate-task%3Apanel-r5-astra-ultra-20261122` ([panel-r5-astra-ultra.md](panel-r5-astra-ultra.md))
2. **Anthropic Claude — `claude-opus-5-5` (`effort: "ultrathink"`):** `thread://v2/c5bb8594-9c79-4948-9251-bd83e8527cf6/thread:delegated-task:command%3Amcp%3A3b84a70c-9824-441c-8750-b9902399a932%3Adelegate-task%3Apanel-r5-claude-ultrathink-20261122` ([panel-r5-claude-opus-ultrathink.md](panel-r5-claude-opus-ultrathink.md))
3. **xAI Grok — `grok-4.7` (`reasoningEffort: "xhigh"`):** `thread://v2/c5bb8594-9c79-4948-9251-bd83e8527cf6/thread:delegated-task:command%3Amcp%3A3b84a70c-9824-441c-8750-b9902399a932%3Adelegate-task%3Apanel-r5-grok-xhigh-20261122` ([panel-r5-grok-xhigh.md](panel-r5-grok-xhigh.md))
4. **Google Gemini — `gemini-4-neon` (`thought_level: "xhigh"`):** `thread://v2/c5bb8594-9c79-4948-9251-bd83e8527cf6/thread:delegated-task:command%3Amcp%3A3b84a70c-9824-441c-8750-b9902399a932%3Adelegate-task%3Apanel-r5-neon-xhigh-20261122` ([panel-r5-neon-xhigh.md](panel-r5-neon-xhigh.md))

---

## 1. 4/4 Unanimous Verdicts & Verified Ground Truth

1. **Receipt & SHA-256 Parity (`4/4` Verified):** All `27/27` bundle manifest digests, `25/25` receipt-bound side-artifact digests (`2` Part I + `6` Part II + `4` Part III + `13` Part IV), `20/20` publication-pinned historical raw files, and `15/15` byte-exact pristine mirror files (`b37e6a55`) verify cleanly, with **zero gold-answer leakage** inside `route_soft_telemetry_top_k` or `C4_v4` retrieval.
2. **Release Banner (`4/4` Unanimous):** Retain **`PRE-RELEASE / HUMAN AUDIT & CLEAN-RERUN GATES PENDING`**. Do **not** declare full gate closure on `PM-1b` stress, `PM-2` integrated QA, `PM-3` raw-text `N=30` KU, or `PM-4b` human audit.

---

## 2. High-Severity Corrections & Disclosures Applied to Whitepaper v3, HTML v3, & Code

| # | Finding (Discovering Models) | Resolution Applied |
|---|---|---|
| **1** | **WFS Oracle Count (`353,105` vs `381,497`)** *(Neon XHigh, Astra Ultra)* | Replaced `381,497` everywhere with **`353,105` independent WFS oracle checks** (`256 + 65,536 + 262,144 + 169 materialized bitemporal snapshots + 25,000 random ops`), clarifying that `28,561` (`169²`) checked cached dictionary non-nullity without re-running `as_of`. |
| **2** | **`PM-1a` Jaccard Denominator, Structured Edges, & `0.95–1.00` Range** *(All 4)* | Corrected `C2b` attribution: `C2b` scored **`29/30` (`0.9667`)** on `flood_12_unlinked_paraphrase_copies` and `0.3375` across all 8 families due to `0.0000` on the 5 structured authority/lineage/refutation families. Disclosed that `C4_v4` (`run_r5b_spark6724_completion.py`) consumes fixture-supplied structured edges post-hoc after `r5_s1` and scored `1.000` (`N=240` & `4×80` `PM-4a`) and `0.950–0.975` on `r5b_s2`/`s2b` (`'ABSTRACT'` decoding). |
| **3** | **`PM-1b` Stress Sub-Gate Unmet, Unstripped `M1c`, & `EM-4` Scope** *(All 4)* | Relabeled `PM-1b` to **`CLEAN AGGREGATE PASS / CONVERSATIONAL STRESS GATE UNMET (FAILED)`** (`+24.36 < +25 pp`, `LCB95 +12.82 < +15 pp`, unstripped `M1c = 0.7692 > M4g = 0.7564`). Added `M4g` zero-spanning clean CIs, `297/300` & `487/500` identical-prompt statistics, R4 error-inspection tuning, 6-prefix regex match, and NVFP4 `2.33 pp` batch variance. |
| **4** | **`PM-2` Full Arm Table, `n=86` Scorer Timer, & Family-Label Guard** *(All 4)* | Reported `Query0W = 0.6950` beside transductive `DualAdap = 0.9650` (`Tango LCB95 = -5.29 pp`, bootstrap `-4.50 pp`), scoped `0.7181 ms` p95 strictly to the `n=86` pre-encoded SciFact numpy scorer with stride-7 mask (vs `36.55 ms` uncached), disclosed the `c["family"]` QA guard (`run_r5b_spark37c3_part2_and_hybrid_pm2.py:234`), and marked `PM-2` **`PARTIAL / REOPENED FOR LABEL-FREE RERUN`**. |
| **5** | **`PM-3` Structured-vs-Raw Timing & `N=30` Non-Significance** *(All 4)* | Qualified `0.48 ms` vs `2,987.95 ms` as structured-graph lookup vs raw-text `Memory.add` LLM extraction (`80/80 nonempty searches`), noted default `0/80` `'ABSTRACT'` outputs, annotated HTML Fig 7 (`N=240` vs `N=80` `0.975`), and marked Suite B (`N=30`, `LCB95 = -3.33 pp`, `top20_chron`) **inconclusive / non-significant**. |
| **6** | **`PM-4` `N=80` Reuse, Off-Git Sealed Key, & Hardened Human Scorer** *(Claude, Grok, Astra)* | Clarified `80` cases reused across `4` cells (`800` rows), foregrounded conversational `κ = 0.7414` (`87/100`), excluded `pm4_human_audit_100_sealed_key*.json` via `.gitignore` (preserving SHA commitment only), and hardened `scripts/score_pm4_human_audit.py` (`100` unique expected IDs + zero-variance `κ` guard). |
| **7** | **`PM-5` Python Implementation Disclosures & `EM-4` Code Fixes** *(All 4)* | Replaced `"0 spurious scans"` with `"0 recorded dense fallbacks"`, disclosed `dict(self.eligibility)` copy (`~7.8 ms` floor) + Stratum-1 dual-Tarjan `2.62s` tail + `100k` dict-index read, and patched `em4_soft_telemetry_routing.py` (fixed inverted representative sort key, complete-linkage echo clustering, removed `"update"` from recap regex, removed dead loop, measured actual fallback witness length, validated `ts`, and added regression tests). |
| **8** | **Part IV Code Provenance Binding** *(Claude Opus 5.5)* | Disclosed `git_sha: 18973789...` in Part IV manifests as the remote Spark base commit prior to `scp` runner sync and added `module_source_sha256` verification in `scripts/build_whitepaper_assets.py`. |
