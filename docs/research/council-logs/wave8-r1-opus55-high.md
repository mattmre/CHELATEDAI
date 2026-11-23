# Wave-8 Round-1 Audit Memo — Opus 5.5 (high)

**Verdict: do NOT seal v5.0 as drafted.** All 5 manifest SHA-256 values match their receipts. But several headline claims mis-cite their own manifests.

**Cross-cutting.**
- Receipts pin `git_sha 5a9bd46f`, but the producing code (`wave6_common.py`, `chelated_grpc_gateway.py`, `scripts/run_wave6_*.py`, `demo_workbench/server.py`) is **untracked**, so the SHA does not reproduce the run.
- In the brief, the Track B hash has 65 hex characters and the Track E short hash is garbled.

## W8-Q1 — K1 / K3

**D5.**
- *Latency.* "Outperforms direct 4096-d Qdrant (43.55/53.20 ms)" is a load artifact. The baseline was saturated: it achieved 218.79 Hz against a 250 Hz target, with e2e p50 3,309 ms. So `incremental_service_p95 = −32.01` is meaningless. The honest grid maximum incremental p95 is 18.0 ms. The gateway's own open-loop e2e p95/p99 (153.75 / 251.86 ms) is omitted.
- *Recall gate.* At 50% density, admitted Recall@10 = **0.994186 < the K1 0.995 gate**.
- *90% density.* Its recall of 1.0 is 100% exact fallback, with latency unreported.
- *Naive baseline.* It is not budget-matched (k32/ef32 against adaptive ef ≤ 256).
- *Image.* `qdrant:latest` is unpinned.
- *SciFact.* ZeroMu (0.5252) < Procrustes (0.5318), and it is n.s. against QueryAffine (p 0.135).

**D2.**
- Pooled CSLS > z-score reproduces (+2.69 pp, lcb₉₇.₅ +1.32). The win is FiQA-carried: NFCorpus +0.43 pp (p 0.46), SciFact lcb₉₇.₅ −0.4 pp (p 0.081).
- **"0.87%" is FiQA-only.** The same 500 anchors are 9.65% of SciFact and 13.76% of NFCorpus.
- Mode A against QueryAffine is null on SciFact (p 0.89), which the thesis omits.
- "Matching Procrustes" hides the FiQA sensitivity result ZeroMu − Procrustes = −2.08 pp (lcb₉₇.₅ −3.67, p 0.0075).
- The CSLS − uncalibrated deltas are correct at p=20%. At p=50%, SciFact is −0.79 pp (p 0.78).

## W8-Q2 — K2: PM-1b stays **SCOPED / OPEN**

- **Primary virgin test fails Holm.** v4 vs Legacy has Holm p 0.0766. v4 < DualGran_NoEK5 (−0.83 pp).
- **KU gain is the embedder.** The +41.67 pp is matched exactly by `Flat_BGEM3_TokenMatched` (95.83%). Its p=5e-5 lies outside the Holm family.
- **"+5.13 pp over DualGran, Holm 0.00025" is misattributed.** That Holm p is v4 vs **MiniLM**. v4 vs DualGran is p 0.3025 (lcb₉₇.₅ −3.85 pp). On TEST-only r5, v4 = DualGran (76.92%): the delta is entirely DEV items.
- **LME-500 carries a do-not-cite label.** It is `SECONDARY_POOLED_CONTAINS_DEV_DO_NOT_CITE_AS_HELD_OUT`, yet it is headlined. On LME_TEST n=251, Flat (60.96%, 1,996 tok) ≥ v4.
- **2.78× tokens** is a cross-run ratio.
- **K2 flag false.** `distinct_bootstrap_quantiles…= true` is contradicted by ties, e.g. virgin-KU lcb₉₇.₅ = lcb₉₅ = −0.125.

**Defensible claim:** token-efficiency parity only.

## W8-Q3 — K4 / D4

- **Guardrails.** Split disjointness, completion-only SFT, KL-ref 0.0, and reload parity 0.0 are verified.
- **frac_zero_std.** `max_frac_reward_zero_std = 1.0`, but the abort flag tests the mean (0.1667). Disclose both. Mean KL 0.0016 means GRPO barely moved the policy.
- **Critical — false commits on U.** On EVAL, GRPO `false_commit_rate_on_u = 0.875`. That equals SFT and is **worse than Bare 30B (0.80)**.
  - The lift over 30B is entirely determinate items.
  - Moat claim #3 ("verifiable 3-valued reward") is unsupported.
- **Format confound.** Bare 30B `valid_format_rate = 0.0`, so the comparison partly scores the SFT'd schema.
- **D1.** SFT is 100% on every D1 regime, so D1 is an SFT result. "Zero-shot" should read "zero entity overlap, shared scenario grammar."

## W8-Q4 — K5 / K6

**PM-3 Suite A.**
- Bootstrap and Tango LCBs are > 0. But the manifest's Clopper-Pearson lcb₉₅ is **≤ 0 against 3 of 4 arms**: Hindsight −0.38, Mem0-True −2.74, OpenSearch −1.57 pp.
- OpenSearch-emulation determinate accuracy is 100% vs Chelated 97.78%. The margin is 15 U items in a self-authored corpus.
- Chelated `stale_resurrection = 0.7833` contradicts Regime II's "$0 ops."

**PM-3 Suite B.**
- LCBs are correctly withheld.
- The "70×–207×" slogan omits Mem0 infer=False at **1.56×**.
- Chelated read p50 is **521 ms vs 5–14 ms**, which is omitted.

**K5.** The labels are correct, but the prose ("silently overwrite history", "boosts echoed rumors") assigns our emulations' behaviour to real vendors.

**K6.** All TCO cells recompute exactly ($526,920; $75,457; $794,880; −133.9%). The **coefficients are not measured**:
- Mem0 620/1,900, Hindsight 740, OpenSearch 580, and the 13,500-token setup appear in no manifest.
- Measured Mem0-True read is 371–507 tokens. With that, Mem0 read COGS (≈$152k) falls *below* Chelated's $526,920.
- `server.py:305` labels them "Measured".
- `test_workbench_server.py:85-90` asserts the ranking.
- K6's embed/ANN terms are missing.
- Migration is ≤ $180 per arm, contradicting the thesis's "multi-hundred-thousand-dollar wall."
- Vendor-risk disclosure is present.

## W8-Q5 — Edits before sealing

1. Commit the Wave-6 code, re-stamp `git_sha`, and fix the brief hashes.
2. **D5:** saturated-baseline wording, max incremental p95 18.0 ms, e2e p95/p99, 50% density **gate miss**, fallback latency, pinned digest.
3. **D2:** "n=500 anchors (0.87–13.76%)", per-domain nulls, SciFact Mode A null, FiQA −2.08 pp.
4. **PM-1b:** strike +5.13 pp/Holm 0.00025 and the LME-500 headline; credit BGE-M3; K2 flag false; keep SCOPED/OPEN.
5. **D4:** add the U false-commit row, 30B format 0.0, and max zero-std 1.0; reword "zero-shot"; scope moat #3 to determinate items.
6. **PM-3:** add Clopper-Pearson columns, "1.56×–207×", Suite B read latency; strip vendor-behaviour prose.
7. **TCO:**
   - Re-source competitor coefficients from PM-3 or tag them `ASSUMPTION`.
   - Fix the `server.py` comment and make the test assert the arithmetic identity.
   - Add the embed/ANN terms and drop Regime II's $0 ops.
8. Give out-of-bundle citations (353,105 WFS checks, 16/18 OOD, TRA-1 8/8 + 12/12, D3 20/20) receipt IDs, or remove them.
