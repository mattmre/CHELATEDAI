# Wave-8 R2 Cross-Judgment — Opus 5.5 (high)

**Do not seal `v5.0` yet.** Votes: M1 MODIFY, M2 MODIFY, M3 APPROVE_WITH_GUARDRAILS, M4 MODIFY (blocking), M5 MODIFY. `PRE-RELEASE` stays.

## Disputes settled from the receipts

1. **D5 recall at 50%.** Sol and Opus are right; Neon is wrong.
   - At 50% retraction the fenced admitted recall is `0.994186`, below the 0.995 gate.
   - The `rescored_recall…_ge_0_995=True` flag covers only the `k_os≥32` grid. `k_os=10` scores `0.887`.
2. **Suite A significance.** Both camps are right.
   - Bootstrap `lcb_975>0` against all four arms.
   - Clopper–Pearson `lcb95≤0` against three: Hindsight −0.38, Mem0-True −2.74, OpenSearch −1.57 pp.
   - Thesis L43 ("significantly beating **every** competitor") must name its test.
3. **K2 flags.** Neon's "all 8 pass" is rejected.
   - `assignment_gates_pass` is false.
   - The distinct-quantile flag is contradicted by 34 ties.
   - LME-500 is labelled `…DO_NOT_CITE_AS_HELD_OUT`.
4. **Brief L20.** The v4−MiniLM delta is +29.49 pp (Neon), not +28.21. The Holm `0.00025` belongs to that contrast. "+5.13 pp over DualGran" (p 0.3025) must not be shown as significant.
5. **Provenance.** The Wave-6 producing code is untracked, so `git_sha` cannot reproduce the runs.

## M1 — Gate table: MODIFY

- **PM-1a, PM-2, PM-5:** carried unchanged. They were not re-audited, so they are not upgraded.
- **PM-1b:** `SCOPED / OPEN`.
  - Virgin-120 Holm p=0.0766 does not reject. v4 trails NoEK5 by 0.83 pp.
  - Report token compaction only:
    - Virgin: 1,028 vs 2,488 tokens.
    - LME-500: 2,191 tokens, with the label.
    - J4 denominator: 6,088 tokens at N=251, cited from the J4 receipt.
- **PM-3 Suite A:** `LCB97.5>0` against all four arms by bootstrap, with the Clopper–Pearson column beside it.
- **PM-3 Suite B:** 70–207× token savings against the LLM-write arms, 1.56× against infer=False. Accuracy is parity only (`LCB97.5≤0`); equivalence was not tested.
- **PM-4:** `PM-4b` human grading is the sole remaining human gate.

## M2 — D5 and Track B: MODIFY

**Backend disclosure.** Binding text: "project-defined `SearchService.KnnQuery` Protobuf over live Qdrant gRPC `:16334`; no OpenSearch Java cluster." Deck L276 ("in front of OpenSearch 3.9") must change.

**Latency.**
- The 21.19 ms p95 at 250 Hz is service RTT. Print open-loop e2e p95/p99 (153.75 / 251.86 ms) beside it.
- The direct baseline was saturated (218.79 Hz achieved), so cite the same-dimension +18.0 ms p95.
- 90% density is 100% exact fallback.

**Mode B.**
- Cite "n=500 anchors (0.87% FiQA / 9.65% SciFact / 13.76% NFCorpus)".
- The pooled win is carried by FiQA; the per-domain NFCorpus and SciFact results are null.
- Uncalibrated wins hold only at p=20%.
- Every bf16 minimum is below 0.9999.

**Mode A.** SciFact is null. NFCorpus fails Look #2. FiQA is −2.08 pp versus Procrustes on the positive-excluded set.

## M3 — D4: APPROVE_WITH_GUARDRAILS

Binding text: "The lift over Base and 30B is attributable to SFT. GRPO adds nothing measurable (EVAL +2.22 pp, Holm 0.4495; D1 −0.56 pp, 0.733)."

- D1 is an SFT result (SFT scores 100%). Replace "zero-shot" with "zero entity overlap".
- Disclose `false_commit_rate_on_u=0.875` (no better than 30B), 30B `valid_format_rate=0`, and maximum zero-std fraction 1.0.
- Change the budget from 256 to 272 (thesis L75).

## M4 — Thesis, deck, workbench: MODIFY (blocking)

- **Wrong label.** `server.py:305/309` labels the 1,900 Mem0 read tokens "Measured"; the measured range is 371–507. Tag 620/740/580/1,900 as `ASSUMPTION`.
- **Defaults.** Default OpenSearch to 2,178.6 (13,500 is at L313) and keep 13,500 as a toggle.
- **Missing cost lines.** Add embedding, ANN and storage lines, or mark them as equal offsets.
- **Margin.** Split L120 "85%+" into BYO-LLM software versus bundled 70.7% / 95.8%.
- **Vendor prose.** Strip prose that assigns our emulations' behaviour to real vendors.
- **`index.html` fixes:** L632 "46.7%" → 33.33%; replace stale L750 `0.6082`; scope L660's "no superiority vs Mem0" to D1 v2, R6 and Suite B (L652 prints LCB +5.00 pp).
- **Test.** Assert the arithmetic identity, not the ranking.

## M5 — Sign-off checklist: MODIFY

1. Commit the Wave-6 code and re-stamp `git_sha`.
2. Fix brief hashes (Track B/E) and +28.21 → +29.49.
3. Bind hashes for the five new Part VI entries (35→40).
4. Print the 50%-retraction recall miss.
5. Rename `stale_resurrection_*` as a substring provenance scan.
6. Apply all M1–M4 edits.
7. Receipt-ID or delete out-of-bundle claims (16/18, 51.2×, TRA-1, D3).
8. Keep `PRE-RELEASE` until `PM-4b` lands.
