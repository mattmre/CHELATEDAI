# Wave-5 R2 Cross-Judgment — claude-opus-5-5 (high)

**Self-correction.** Line 597 reads `max_abs_dot_vs_numpy: 2.682209014892578e-07`. My R1 headline 4 ("2.68") misread that value, and I retract it. Float32 parity holds.

## K1 — D5 gateway: **MODIFY**
- **(a) Request-only scoring.** Definitional: request-only = `q̂=(q−μ_x)W`, no `+μ_y`, DOT.
  - It differs from ZeroMu only by the `1/‖d_j−μ_d‖` divisor, so it ranks like Procrustes (1M: 0.54 vs 0.55, CI ∋ 0).
  - QueryAffine (0.38) is a strawman, never default.
  - Egress payload: 4 B `n_j` + 4 B `r_d` (float32) = 8 B/doc.
  - Claim egress rescoring only if its lift over request-only Procrustes-DOT has LCB95 > 0.
  - The certified stop must also bound per-shard `r_d,min`.
- **(b) Filtering.** Pre-filter with `ef_eff=min(ef_max,⌈k_os/p_admit⌉)` and fall back to exact search below `p_admit` 0.1. Sweep `ef` independently of `k_os`.
  - The egress EK5 fence is mandatory at 0/10/50/90% retraction, measured against exact admitted top-k.
  - `CONTESTED` is a tri-valued payload.
  - "0.00% resurrection" requires `wait=true` and epoch-pinned reads.
- **(c) Latency.** 2,290.8 ms is labeled 30B decode (Δ native 8.9 ms). Keep the reader off the timer.
  - Grid: REST vs gRPC `:6334` vs OpenSearch `KnnQuery`, ×384/1024/4096-d.
  - ≥10k open-loop queries (N=200 cannot support p99).
  - Gate: recall@10 ≥ 0.995 vs exact ZeroMu, and p95 overhead ≤ 25 ms.
  - Unsupported OpenSearch gRPC → UNSUPPORTED, not emulated.

## K2 — PM-1b v4: **ADOPT + guardrails**
- Run EK5 echo collapse before budgeting, and keep update chains chronological.
- Tune the reranker, window and token cap on LoCoMo 0–4 and LME DEV only, with the config SHA-frozen.
- The comparator is the token-matched flat BGE-M3 control (0.610). The clean EK5 delta is 0.00, so claim EK5 lift on stress only.
- Emit bootstrap percentiles explicitly: `p5.0`, `p2.5` and `p1.25`. Add a test asserting `p2.5 < p5.0` whenever variance > 0.
- LoCoMo 5–9 and LME TEST are labeled Post-J4 Look #2, α=0.025, **secondary**.
- **Primary** is a virgin, lineage-disjoint holdout with paired cluster intervals.
- Use a cross-family judge plus a blind human audit.

## K3 — D2 three domains: **ADOPT, both modes**
- Generate the missing NFCorpus and FiQA 384-d caches.
- Run identity checks at both widths: all queries plus ≥1,000 stratified docs per domain.
  - NN-top1 = 1.0, with a text SHA stored in each `.npz`.
  - cos ≥ 0.99999 at 384-d and ≥ 0.9999 at 4096-d (bf16, batch composition logged).
- **Mode B (4096→384) is primary.**
  - Unmigrated docs live in a 384-d collection and backfilled docs in a 4096-d one. This removes the `docs_high` contamination.
  - Use n=500 anchors, which must be ≥ TRA-1 `min_fit_count`.
  - Primary contrast: CSLS − DEV z-score at 20% backfill, domain-stratified. Holm covers the rest, including RRF(k=60).
- Mode A secondary; within-corpus anchors labeled transductive.

## K4 — D4 GRPO: **ADOPT all 7, MODIFY 2**
- Apply the −0.15 penalty to determinate-gold items only, which puts break-even at p≈0.425. Genuine-U items keep abstention as the top reward.
- Close the +0.25 "verdict + ABSTAIN" hole.
- Hash `noise-retractor` into the disjointness check.
- Pin the keyword reward signature; called positionally, it scores the prompt.
- The primary checkpoint is the pre-registered step 300, with best-DEV secondary. Run ≥3 seeds.
- Raise the Part-1 extraction cap to ≥256, and run AWQ at fp16.
- "Live trained" requires a hashed reload with logit parity.

## K5 — PM-3 v2 arena: **ADOPT**
- GPU encoding, no lock inside the timer, no N60+N40 pooling.
- Keep separate clocks for:
  - ack
  - visibility (Δ∈{0,1,10,60}s plus post-consolidation)
  - read
  - LLM
- Every arm gets blind cross-family grading, and the substring markers are deleted.
- Reproduce each vendor's published number first.
- Anything not run on live AWS is labeled "emulation (ours)".
- Current state: LCB −6 pp, **no win**.

## K6 — Valuation: **MODIFY**
- **Rejected as unsupported:** ">85% GM", "71×" (it mixes ingest and reader tokens), "1023×", "900+ GPU-hour" and "irreproducible".
- Parameterize 100M-event TCO from measured inputs only:
  - writes × (tokens + embed)
  - reads × (tokens + ANN)
  - storage
  - bridge re-encode: 49% for Mode A; 0.87% for Mode B if validated
  - ops and compliance
- No accuracy claim wherever LCB ≤ 0.
- Measured moat:
  - TRA-1 wrong-space refusal
  - masked resurrection 0.00 vs 0.60–0.765
  - deterministic `as_of` and WFS replay
- Disclose that AWS/Elastic could ship `CONTESTED` within a quarter; integration option until D5 and a prospective split land.
