# Wave-5 R2 Cross-Judgment — Grok 4.7 (xhigh)

Four R1 memos: Sol 62 xhigh, Opus 5.5 high, Grok 4.7 xhigh, Neon high. [d2_4096d_qdrant_manifest.json](D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1\artifacts\phase2_d2_qdrant_hnsw_csls_backfill\d2_4096d_qdrant_manifest.json) lines 597/606/615: `max_abs_dot_vs_numpy = 2.682209014892578e-07` (SciFact, `k_os` 10/32/64, 5,183 DOT). Float32 Qdrant-vs-numpy parity, `2.68e-7`. Opus’s bare `2.68` misreads it. Not row identity.

## K1 MODIFY

Request-only `q_hat` on raw DOT is QueryAffine. Procrustes is the comparator; its SciFact LCB includes 0. 1M recall@10 is `0.380` vs ZeroMu `0.550`. The −26.9 pp cell is SciFact 20% nDCG (`0.339` vs CSLS `0.608`). Egress: 4-byte `||d_j−μ_d||_2`, 8-byte `r_d`, shortlist `r_q`; `n_min/n_max` stop, else exact fallback. Sweep `ef` apart from `k_os`. TRA-1 fails closed first (fit ≥ 32, validation ≥ 16, dim and index match).

Fence 0/10/50/90%: epoch pin, `ADMITTED` bitset, `CONTESTED` as its own payload, `AUDIT_ONLY` dropped before CSLS. `ef_eff = min(ef_max, ceil(k_os/max(p_admit,ε)))`; exact scan if `p_admit<0.3` or the stop fails. `0.00%` resurrection needs `wait=true`. Report recall vs exact admitted top-k.

`2,281.9 ms` and `2,290.8 ms` (gap 8.9) are 30B decode; `95.6 ms` is the 3B reader. Transport only, reader off this clock, no SSH in the timer: REST/JSON vs Qdrant gRPC `:6334` vs pinned OpenSearch `SearchService.KnnQuery`; 384/1024/4096; pool on/off; batch 1 and 32; one NIC; local and 3090→Spark; N≥10,000; p50/p95/p99, bytes, errors, parity. Separate gate: rescored recall@10 ≥ 0.995 vs exact ZeroMu.

## K2 MODIFY

EK5 collapse before the session budget. Cap-first J4: echo-stress `0.756→0.667`, evidence-all `0.972→0.708`. `bge-reranker-v2-m3` is DEV-only (LoCoMo 0–4, frozen LME DEV) vs the token-matched control, then frozen. Distinct `p5.0`, `p2.5`, `p1.25`: `lcb_95=p5.0`, `lcb_975=p2.5`, `p1.25` reported only; a copy fails closed. LoCoMo 5–9 and LME TEST are Post-J4 Look #2 at α=0.025, beside the original J4 number. No fork for the +0.9 pp miss. Primary: one virgin lineage-disjoint holdout, blind cross-family judge.

## K3 MODIFY

`2.68e-7` is SciFact HNSW-vs-numpy DOT parity only (`reencoded_8b=false`; published check is MiniLM 384-d). Re-encode SciFact 5,183, NFCorpus 3,633, FiQA2018 57,638 at D384 and D4096: cosine ≥ 0.99999, NN-top1 = 1, text SHA-256. Pin Qwen3-Embedding-8B, left pad, 8192, final non-pad pool, L2 fp32, query instruction only. Partial samples fail.

Mode A: 384→4096 up-bridge. Mode B: unmigrated docs stay in 384-d HNSW; backfilled `p%` alone sit in 4096-d HNSW. n=500 is 0.87% of FiQA, not SciFact `2,538/5,183`. No paste onto `docs_high`. Arms: causal DEV-only CSLS, DEV z-score, RRF(`k=60`). Primary: CSLS−z at backfill 0.2, Holm elsewhere. 0% and 100% are single-space controls. Disclose qrels-blind vs positive-excluded. Freeze the salt first.

## K4 ADOPT

(1) `merge_and_unload()` SFT, then a fresh GRPO LoRA, so KL anchors to SFT; no second 7B. (2) Disjoint `SFT_WARMUP` and `GRPO_RL`; include `noise-retractor`; log `frac_zero_std`; abort above 0.8. (3) Left padding; batch 8 = `num_generations`; pin TRL/PEFT/Transformers. (4) Completion-only SFT; no `store_slot_status` in the target. (5) Train, eval, D1, and the Part-1 extractor use `max_new_tokens=256`. (6) False abstention on labeled T/F is −0.15 before the advantage, not 0 or +0.25. (7) Zero-shot Base, SFT, SFT+GRPO, Bare 30B on unseen D1; ≥3 seeds; matched decode; DEV does not pick the checkpoint. Reward sees completions only.

## K5 ADOPT

GPU encode, no `enc_lock`, CPU not forced. Split clocks: write-ack, consolidation lag, read, LLM. Primary freshness: Δ=0s vs first post-consolidation read. Keep N=60 out of the N=40 mean. Pooled +4 pp (LCB −6 pp) is not a headline. Blind cross-family semantic grades replace `ABSTAIN_MARKERS`; schema compliance is separate; re-grade every arm. Label `mem0ai 2.2.1` `infer=True` (LCB misses 0), Hindsight only with Retain/Recall/Reflect, and OpenSearch as Memory-parity async SEMANTIC+SUMMARY unless the vendor service is timed.

## K6 MODIFY

100M-event identity: writes×(tokens+embed)+reads×(tokens+ANN)+storage+migration re-encode+retraction+operations+compliance. Measured or symbolic coefficients only. Reject “>85%”, “71×”, “900+” GPU-hours, “1023×”, and “irreproducible.” Position: compliance and migration control plane (3-valued WFS, `as_of`, zero write-path LLM tokens, TRA-1 refusal). Accuracy only if a pre-registered LCB excludes 0. PM-3, Mem0 `infer=True`, ZeroMu vs Procrustes, and clean-LME EK5 are undistinguished. Disclose that a vendor can ship `CONTESTED`, `as_of`, and a CSLS sidecar. No premium.
