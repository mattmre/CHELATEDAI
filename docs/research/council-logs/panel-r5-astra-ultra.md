# Final 4-Model Panel Audit — OpenAI GPT-6.2 Astra (`reasoningEffort: ultra`)

- **Thread:** `thread://v2/c5bb8594-9c79-4948-9251-bd83e8527cf6/thread:delegated-task:command%3Amcp%3A3b84a70c-9824-441c-8750-b9902399a932%3Adelegate-task%3Apanel-r5-astra-ultra-20261122`
- **Model:** `codex` / `gpt-6.2-astra` (`reasoningEffort="ultra"`)
- **Date:** 2026-11-22

---

**Hold Draft v3 for correction. Keep `PRE-RELEASE / HUMAN AUDIT PENDING`.** The 27 bundles verify. The human audit is only one of the remaining gates. Three findings change the release decision: the claimed paraphrase advantage uses the wrong denominator, PM-2 consults a benchmark-family label and times a different workload from QA, and the reported oracle total counts cached snapshot pairs as independent checks.

### Claim-to-receipt audit and required edits

I independently matched **27/27 manifest hashes**, all 27 ledger entries in both publications, **25/25 receipt-bound side hashes**, **20/20 publication-pinned historical raw files**, and **15/15 files against the pristine mirror manifest**. All nine R5/R5b observations matched their source manifests. `b57b05660edf…` is the canonical **content** digest. The serialized registry file hashes to `3a8aac940b14…`. Paper line 358 should attribute the 25 side hashes across Parts I–IV: **2 + 6 + 4 + 13**.

| Gate | Verified result and edit before commit |
|---|---|
| **PM-1a — structured ZS v3** | C4_v4 returned the exact expected answer on **240/240**, averaging **162.1 prompt tokens**. Jaccard scored **0.3375 across all eight families**, giving **+66.25 pp**, bootstrap lower bound **+60.41 pp**. On the named **12-unlinked-paraphrase** family, Jaccard scored **29/30** (`0.9667`), C4_v4 **30/30**, and latest-timestamp **30/30**. Replace “lexical Jaccard collapses on paraphrased echoes” with those denominators (`16.0` mean, `14–18` events/case; vs C4_v3 bootstrap LCB `+8.33 pp`, Tango `+8.90 pp`). |
| **PM-1b — conversational memory** | M4f matched M1 on every clean scored item (`0.3867` LoCoMo-300, `0.5480` LME-500; prompts identical on `297/300` and `487/500`). Replace “100% clean Recall@5” with **“matched M1’s 0.630 any-evidence hit rate at five on 200 answerable questions.”** M4f Tango lower bounds: `-1.26 / -0.76 pp` overall and `-5.20 pp` on SSU-70. On KU-78, M4g (`0.7564`) vs M1 (`0.5128`) and Jaccard (`0.7692`) missed all three stress requirements (`+24.36 pp`, LCB `+12.82 pp`, `-1.28 pp` vs Jaccard). Label dashboard **“clean aggregate pass / conversational stress gate unmet.”** |
| **PM-2 — coupled service** | **12/12** isolated TRA-1 probes blocked. **0.7181 ms p95** covered **86 pre-embedded SciFact queries**, cached projection, simulated stride-7 mask, and top-10. Hybrid QA (`200` cases): `193/200` DualAdap (`0.965`), `197/200` Native (`0.985`), `139/200` Query0W (`0.695`), `139/200` CAR (`0.695`), `34/200` flat (`0.170`). Bootstrap LCB `-4.50 pp`, Tango `-5.29 pp` (`< -5 pp` on 1 win / 5 losses), and QA harness empties context conditional on `case["family"]` for EK5 arms -> reopen integrated service claim for label-free rerun. |
| **PM-3 — live mem0ai 2.2.1** | Default scored `0/80` (all 80 answers decoded as `ABSTRACT`). Steelmanned ZS scored `38/80` (`0.475`) vs C4_v4 `78/80` (`0.975`), `+50.00 pp`, LCB `+38.75 pp` (`80/80 nonempty searches`). Qualify `2,987.95 ms` text-to-memory add vs `0.480 ms` structured-graph routing. On first 30 KU items, M4g `21/30` vs mem0 `18/30` has LCB `-3.33 pp` (query-selected top-20 user turns). |
| **PM-4 — judge agreement** | `80/80` C4_v4 per cell across 4 cells reusing 80 cases (`800` synthetic rows = 80 cases × 5 arms × 2 generators). Synthetic κ `0.9847`, pooled κ `0.9570` (`[0.9376, 0.9751]`), conversational κ `0.7414` (`87/100`). Harden `score_pm4_human_audit.py` to require 100 unique expected IDs and handle undefined κ. |
| **PM-5 and oracle coverage** | Replace `"381,497 independent oracle checks"` with **`353,105`** (`169` materialized bitemporal snapshots; `28,561` loop checks cached non-nullity without rewind). Replace `"zero spurious scans"` with **`zero recorded dense fallbacks`** and disclose per-mutation `dict(self.eligibility)` copy and harness-local 100k read index. |

### Algorithmic disclosures & Next demonstrations
- Disclose conversational heuristic routing (`ek5_store=None`), R4 LoCoMo error inspection, synthetic 6-prefix match, unstripped `M1c`, finite conflict rules, transitive Union-Find risk, `DualAdap` evaluation-batch query centering, and qrel-selected `COPY_OF` echoes in dense PM-2.
- **Highest-value next demos:** (1) Prospective post-cutoff SEC filings & amendments benchmark with frozen thresholds and streaming centroids; (2) Cross-Spark live encoder handoff + multi-family agent recomposition with H100 70B+ reference; (3) Audited independent epistemic RLVR verifier + 7–8B SFT vs SFT+RLVR adapter ablation on held-out organizations and schemas.
