# Wave 8 Round 2 Cross-Judgment

## M1 — APPROVE_WITH_GUARDRAILS

Use identical labels in the whitepaper and HTML.

- **PM-1a — STRUCTURED PASS / D1 v2 RAW-PROSE FLEET VERIFIED.** Scope the claims to 240/240 structured cases and 56/60 D1 v2 cases.
- **PM-1b — SCOPED / OPEN.** Historical R5/J4 clean EK5 delta remains open (`assignment_gates_pass=false`). Virgin-120 uses 1,028 vs 2,488 tokens (2.42×), with 95.00% vs NoEK5’s 95.83% and Holm p=.0766 vs MiniLM. LME-500 uses 2,191 tokens vs historical J4 TEST’s 6,088 (2.78× cross-run, N=500 vs 251). Label the pooled DEV-inclusive results secondary and TEST exploratory Look #2. Disclose percentile ties and report matching 95.83% Virgin KU scores for v4 and Flat BGE-M3.
- **PM-2 — ZERO-ORACLE BOOTSTRAP PASS / EXACT TANGO MISS.** −5.72 pp misses −5 pp.
- **PM-3 — SUITE A SUPERIORITY / SUITE B TOKEN SAVINGS, ACCURACY INCONCLUSIVE.** A’s 58/60 has bootstrap LCB97.5 > 0 against all four arms, with three conservative Clopper–Pearson intervals including zero. B’s 25/40 uses 70×–207× fewer tokens versus LLM-write arms and 1.56× versus infer=False. All LCB97.5 ≤ 0. Treat accuracy parity descriptively and equivalence as untested.
- **PM-4 — PM-4a CONVERSATIONAL κ=.8400 / PM-4b HUMAN GRADING PENDING.** Cluster CI [.7535,.9122]. The blind 100-item audit is the sole remaining human gate.
- **PM-5 — CLEARED (PASS), BENCHMARK-SCOPED.** Report the harness workload, 1M tail and zero recorded dense fallbacks.

[Existing gate table](../doctoral-whitepaper-chelated-memory-and-spherical-migration-2026-11-21.md#L353) · [Arena receipt](../../../artifacts/wave6_pm3_v2_competitor_arena/wave6_pm3_v2_arena_manifest.json)

## M2 — MODIFY

D5 uses project-defined OpenSearch-style `SearchService.KnnQuery` Protobuf over live Qdrant gRPC. Native Java OpenSearch interoperability is unverified. At target 250 Hz LAN, gateway service p50/p95 is 7.90/21.19 ms with stopping disabled, versus scheduled end-to-end p95/p99 of 153.75/251.86 ms. Direct Qdrant saturated at 218.79 Hz. Use the matched grid’s +18.00 ms maximum incremental service p95. Scope zero stale returns to tested epoch pins and simulated lag. Report 0 vector bytes/doc, 8 bytes/doc sidecar, exact-prefix certificate conditions, 100% recall at k=ef=64, 99.42% recall below the 99.5% floor at 50% retraction, and exact fallback at 90%.

Mode B’s 500 anchors are 0.87% FiQA, 9.65% SciFact and 13.76% NFCorpus. Live backfill tests cover 0% and 20%. The pooled 20% CSLS gains are +2.69 pp over z-score and +5.42 pp over RRF, with per-domain uncertainty. Secondary Mode A beats QueryAffine by +12.36 pp on FiQA and +3.07 pp on NFCorpus before adjustment. Disclose SciFact’s null, NFCorpus’s adjusted Look #2 miss, FiQA’s −2.08 pp positive-excluded Procrustes loss and all three 4096-d cosine-floor misses.

[Gateway receipt](../../../artifacts/wave6_d5_grpc_interceptor_gateway/wave6_d5_grpc_gateway_manifest.json) · [Migration receipt](../../../artifacts/wave6_d2_4096d_3beir_complete/wave6_d2_3beir_manifest.json)

## M3 — MODIFY

Most observed lift is already present after SFT. SFT/SFT+GRPO score 42.50%/44.72% on EVAL (N=120) and 100%/99.44% on zero-shot D1 (N=60, k=17, disjoint entities, shared scenario grammar). Paired GRPO increments are +2.22 pp [−1.11,+5.83] and −0.56 pp [−1.67,0], both statistically inconclusive. Attribute significant baseline contrasts to the combined arm. Disclose 87.5% EVAL false commits on U, Bare 30B’s 0% valid format, 272-token K4 deviation and zero-reward-variance mean/maximum .1667/1.0.

[D4 receipt](../../../artifacts/wave6_d4_trained_epistemic_grpo/wave6_d4_epistemic_grpo_manifest.json)

## M4 — MODIFY

Align thesis, deck, `server.py` and `index.html`. Default to 506.65 Mem0 Suite B read tokens ($376,875 under batched assumptions, below Chelated Regime I) and 2,178.6 compact OpenSearch tokens. Keep 13,500 as sensitivity ($4,209,980, −133.9% margin). Include raw prose, batched/unbatched writes, infrastructure and output costs, separate anchor fitting from backfill, and label prices, batching and incident shares as assumptions. Remove unsupported 85%+ margin and zero audit cost claims. Refresh stale workbench results, label the NumPy simulator and rehash before displaying verification.

[Thesis](../investor-thesis-and-competitive-moat-2026-11-23.md) · [Deck](../investor-solution-architecture-deck.html)

## M5 — BLOCK

Retain **PRE-RELEASE** through PM-4b. Before sealing, implement M1–M4, pin producer sources and Qdrant image, correct Track B/E digests, add five Part VI entries and verify 40/40 bundles plus registered side hashes, and check Markdown/HTML parity. Disclose Suite A’s pre-ack extraction, provenance substring metric, live Mem0 configuration, emulations and missing intermediate visibility clocks. Resolve K1–K5 deviations with compliant receipts or explicit council variances. Qualify or remove unsupported vendor, production and commercial guarantees.

[Binding contract](wave5-r2-synthesis.md)
