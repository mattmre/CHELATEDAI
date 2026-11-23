# Wave-4 R2 Judgment — Opus 5.5 (high)

Read-only; sources: four R1 memos.

| # | Vote | Peer basis | Guardrails |
|---|---|---|---|
| J1 GRPO | **MODIFY** (code) | Grok C1/C2 | Score one exact-match canonical span. Reject negation and closed-lexicon abstain on non-U items. Pay the format bonus (≤0.05) only on a valid parse **and** core reward >0, so a formatted wrong answer (≤0.05) can never beat a bare correct one (1.0). Splits disjoint on entity, value **and** template; hash-assert train∩eval=∅. Report reward and exact-match separately; disclose the +0.25 verdict+ABSTAIN payment. |
| J2 PM-2 | **MODIFY** (code) | Opus C1, Neon #3 | Pass a redacted case object (no `entity`/`slot`/`family`/tenant keys) so any oracle access raises a KeyError. Route **every** arm, Flat included, through the same `select_context_for_query`. Derive the flags from that guard and a runtime trace, not literals. Report the EMA spread over ≥5 query-order shuffles. Accept the rename. Disclose Native=1.000 as a ceiling task. |
| J3 D2 | **MODIFY** (code) | Sol, Opus M4/M5 | Pin the row-ID list and its sha256. Require cosine ≥1−1e-5 per row, not ==1.0. z-score/min-max stats come from DEV only. Fix RRF k=60 a priori, untuned. Restate the CSLS gain against the **best** calibrated baseline. Replace REPORT-positive anchor exclusion with a DEV-only or hash-random split. Use two-sided 2.5% CIs everywhere. Debug QueryAffine recall 0.0 before citing it. |
| J4 PM-1b | **MODIFY** (code) | Opus C2/M1–M3 | Fix the LME DEV IDs by hash **before** any run. Commit and hash the frozen config (τ, δ, budgets, thresholds), then make one held-out shot. **Primary = LoCoMo convs 5–9 + LME held-out only.** Full N=300/500 contains tuning items: secondary, labelled contaminated. R1 never forces abstention. Token-match R4 within ±5% and add a retrieve-nothing control. Report Cat-1/2/4 cost. |
| J5 Workbench/PM-4 | **ACCEPT_CODE_FIX** + additions | Sol, Grok C3/M3 | Workbench: 4096-d; ef/k_os=32 → 99.77% mean / 90% min recall, 100% at 64; label nDCG as exact-search until J3 supplies per-k_os values. Grader: hide dataset/qid, fail closed on a missing question, rubric parity including the preference rule. Reject a parse containing both YES and NO. State κ LCB 0.7587 < 0.80; +0.0986 is cross-cohort. Remove the pooled gate unless structured rows are regraded under v8. |
| J6 D3 | **MODIFY** (code) | Sol MAJOR, Grok C4 | Pre-declare complete-linkage as primary and report single-linkage as an ablation. **Never pick the linkage post hoc on eval.** Re-fit τ on templates disjoint from eval, not seed-only. Clopper–Pearson is two-sided 95% per family; disclose that 0/20 gives an upper bound ≈16.8%. |
| J7 v4.0 | **MODIFY** (disclosure) | Neon, Opus | Write "trails by ~1 pp, LCB −3.4 pp", not "parity". 1M is 99% synthetic. The 4-byte sidecar needs a shared D-float centroid, and the raw index stays resident. Every number cites manifest, n, two-sided CI. Run J1–J6 **before** freezing v4.0. A rerun that worsens a number replaces it; no selective updates. Print `assignment_gates_pass=false` and clean EK5 Δ=0; keep the banner. |

## Global anti-leakage rules

1. No hyperparameter is set from REPORT or held-out data. Split IDs are hash-committed before the code runs.
2. One held-out shot per frozen config. A second shot is a new pre-registration, and both are reported.
3. All arms are reported, including failing and negative ones. Intervals are two-sided and cluster-aware where data are grouped.
4. The answer, echo-writer and judge models must not share a family in confirmatory runs.
