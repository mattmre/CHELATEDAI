# Final 4-Model Panel Audit — Claude Opus 5.5 (`effort: ultrathink`)

- **Thread:** `thread://v2/c5bb8594-9c79-4948-9251-bd83e8527cf6/thread:delegated-task:command%3Amcp%3A3b84a70c-9824-441c-8750-b9902399a932%3Adelegate-task%3Apanel-r5-claude-ultrathink-20261122`
- **Model:** `claudeAgent` / `claude-opus-5-5` (`effort="ultrathink"`, `fastMode=false`)
- **Date:** 2026-11-22

---

# Panel audit: Post-R5b Draft v3 (27/27 bundles)

**Verdict:** keep **PRE-RELEASE / HUMAN AUDIT PENDING**. Most of the numbers do trace to receipts. The problems are elsewhere: one gate is labelled cleared when it failed, one headline number is not stable across runs, the PM-1a fixture favours the arm being tested, and the provenance record is broken. None of that supports a release.

The audit was read-only and stayed inside `D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1`. The qscci report was not opened and the blind sheet was not graded.

## 1. Claim-to-receipt parity

| Claim | Registry | Status |
|---|---|---|
| **PM-1a** C4_v4 = 1.000, N=240, 162.1 tokens; LCB95 vs C2b +60.4 pp | Matches the receipts | PASS number. Note §2: fixture supplies structured edges to C4 |
| **PM-1a** C4_v4 "= 1.000" stated as stable | Same arm on ZS v3 N=80 scored **0.9500** (r5b_s2), **0.9750** (r5b_s2b) and **1.0000** (r5b_s4, all 4 cells) | Needs range disclosure `0.95–1.00` across N=80 reruns |
| **PM-1b clean** M4f: LoCoMo +0.0000 (Tango −1.26 pp), LME +0.0000 (Tango −0.76 pp); M4g +1.33 / +1.00 pp | Matches | Parity on 10-conv 300-item LoCoMo subsample & full LME-500 |
| **PM-1b stress** "Clean PASSED on all 800 queries" plus a scoped stress row (L334) | M4g +24.36 pp (< +25 pp); LCB95 +12.82 pp (< +15 pp); 0.7564 (< M1c 0.7692) | Stress sub-gate missed all 3 numeric bars -> label stress sub-gate FAILED / NOT MET |
| **PM-1b** M4e is "#1 (0.4133)" on blind echoes (L245, B-01) | M4e 0.4133 vs M1c 0.4000 (LCB95 = -1.33 pp) | Include CI or soften "#1" |
| **PM-2** p95 = 0.7181 ms; hybrid 0.9650 vs native 0.9850; 12/12 TRA-1 blocked | Matches | Lead with -2.0 pp vs native (LCB -4.5 pp), specify resident-cached (vs 36.55 ms uncached), note author-designed probes |
| **PM-3** default mem0 0.000 vs C4_v4 0.95; steelmanned 0.475 vs 0.975, LCB +38.75 pp | Matches | Suite B LCB95 = -3.33 pp at N=30 (not significant); qualify 6,225x as structured-edge lookup vs ingest LLM extraction |
| **PM-4** κ 0.9570 [0.9376, 0.9751], 0.9847 on ZS, 0.7414 on real conversations | Matches | Foreground real-conv κ = 0.7414 (87% raw agreement); keep sealed key out of public git commit |
| **PM-5** 0.904 ms p95 at 100k, 7.94 ms at 1M | Consistent with the ledger | PASS |

## 2. EM-4 and C4_v4: methodology
- **A. Structured-edge asymmetry in ZS v3:** `C4_v4` consumes `case["attestations"]` (`REFUTE`, `SUPERSEDE`, `DERIVES_CURRENT`, `authority_tier`), whereas text-only baselines (`C0`, `C2b`) only see text, and `authority_laundering` includes an explicit `DERIVES_CURRENT` edge. Must state explicitly that `1.000` proves exact resolution *given structured provenance edges*, and schedule edge-inferred / corrupted-edge `ZS v3` in Phase 2.
- **B. Post-hoc adaptation disclosure:** `C4_v4` was added after `r5_s1` (`C4_v3 = 0.875`) and evaluated on the same deterministic 240 cases (`seed=42`). Disclose explicitly.
- **C. Scoring rule disclosure:** Substring value match (`gold in pred and no stale in pred`).
- **D. EM-4 threshold tuning disclosure:** `em4_soft_telemetry_routing.py` thresholds were tuned against R4 LoCoMo error cases (`loc_1_39`, `loc_3_2`, `loc_9_0`), with `LongMemEval-S N=500` serving as the cross-benchmark check; no gold/answer leakage found.
- **E. Code hygiene items in `em4_soft_telemetry_routing.py`:** Remove dead loop (`:296-298`), validate missing `ts`, prevent `"Update: ..."` false-positive recap match, and guard transitive union-find diameter.

## 3. Provenance & Calibration Fixes
- Disclose `git_sha: 18973789...` in Part IV manifests as the remote Spark base commit prior to `scp` overlay of R5/R5b scripts, and record exact SHA-256 digests of `ek5_v3_stratified.py`, `em4_soft_telemetry_routing.py`, and runner scripts in `verified_registry.json`.
- Replace "cryptographic evidence bundles" over-framing with "SHA-256 hash-verified receipt bundles (integrity-checked)" where appropriate, remove premature completion wording on the 4-model panel until all 4 logs land, and keep `pm4_human_audit_100_sealed_key*` excluded from git commits (hash commitment only).

## 4. Next-Phase Strategy
- Do NOT pretrain from scratch.
- Priority order: (1) provenance & disclosure tightening -> (2) LLM-extracted / corrupted-edge `ZS v3` + full `LoCoMo` on Spark -> (3) unblinded `PM-4b` human audit with off-repo key -> (4) heterogeneous multi-agent `TRA-1 + EK5` shared-memory attack benchmark (Nemotron-3.5 + Qwen + Llama-70B) -> (5) `EK5/TRA-1` RLVR reward-verifier `GRPO/LoRA` pilot on 7B–30B evaluated strictly on held-out real conversations.
