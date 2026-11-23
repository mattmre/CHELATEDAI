# Wave-5 R1 — architecture and investor rigor (Grok 4.7)

Read-only. Published manifests only. No GPU run, no sealed keys.

## Q1. Request-only gRPC is QueryAffine; 2,290.8 ms is the 30B reader

Writing `q_hat` into `KnnQuery.vector` or Qdrant `query` and returning raw DOT/cosine is `score_query_only_affine`: cosine of `(q−μ_x)W+μ_y` against `d_raw`, no centered-norm division (`scripts/run_r6_pm3_and_d2_qdrant_csls_backfill.py:172-181`). ZeroMu is `(d_j·q_hat − μ_d·q_hat) / ||d_j−μ_d||` on a 4-byte sidecar (`:235-247`). That divisor requires centered unit rows: a write. At 1M, masked QueryAffine recall@10 is `0.38` and ZeroMu is `0.55` (Δ `+0.17`, CI `[0.10, 0.24]`). Procrustes is `0.54`; that gap includes zero. The `−26.9 pp` cell is SciFact at 20% backfill, not the 1M cell: uncalibrated nDCG@10 `0.339` versus in-shard CSLS `0.608` (LCB95 `+0.183`). `mixed_shard_scores` (`:471-501`) pastes native scores onto backfilled columns. CSLS is `2s − r_d − r_q` per shard, `r_d` from DEV. At 0% and 100%, CSLS is unidentified.

`resolve_with_tra1` (`spherical_migration_bridge.py:1387-1408`) checks `subspace_rank`, `csls_k`, `scoring_mode`, and the hop chain (`:1355-1366`). Fail closed below fit_count 32 or validation_count 16, or on a dim or index-id mismatch. Source-side `W` only (SciFact residual `2.8e-15`). `qdrant_dot_shortlist` sets `hnsw_ef = k_os` (`:351-363`): exact-top-10 recall `0.864` (min `0.60`) at 10, `0.9988` at 32, `1.0` at 64. Egress is `rescore_shortlist` (`:317-333`), then k=10.

EK5 is `ADMIT` / `AUDIT_ONLY` / `CONTESTED` (`ek5_v3_stratified.py:920-928`). A binary payload filter cannot hold `CONTESTED`. Pre-filter drops points and leaves HNSW edges; under 5,183 twins (`0.60–0.765` unmasked resurrection) raise `ef` as `1/(1−ρ)`. Post-filter oversamples, drops `AUDIT_ONLY`, keeps `CONTESTED` as its own payload, then masks before CSLS. Mixed-shard is two tagged collections plus in-shard CSLS.

Same 30B reader: hetero `2,290.8 ms`, native 1024-d `2,281.9 ms` (Δ `8.9 ms`). Demo D5 times transport only: REST/JSON, Qdrant gRPC `:6334`, OpenSearch `KnnQuery` protobuf; dims 384/1024/4096; k=10 and k_os=64; pooled HTTP/2; N=200; 3090→Spark; no SSH on the timer. Claim the gateway only if rescored recall@10 ≥ `0.995` versus exact ZeroMu and p95 overhead ≤ 25 ms. Label `2,290.8 ms` as 30B decode.

## Q2. Clean lift is the retriever; the session cap drops KU evidence

LongMemEval clean (N=500): DualGran `0.596`, `6,098.9` tokens, `gate_active=10`. Hybrid matches accuracy and KU (`0.782`). SSU is `60/70` versus Legacy `63/70`. Flat BGE-M3 token-matched is `0.610` at `5,823` tokens. `context_chars=26000`, 12 anchors, ±2 (`scripts/run_r6_pm1b_conversational_sota_closure.py:67`).

J4 freezes `min_sessions=3`, `max_sessions=6`, gap `0.14` (`:48-56`). Tokens stay `6,079`; multi-session `0.489→0.429`; KU `0.782→0.718`. Echo stress: DualGran `0.7564` (59/78), evidence-all `0.972`; J4 `0.6667` (52/78), evidence-all `0.708`, LCB `−18.0 pp`. DualGran−Legacy is `+25.64 pp`, LCB `+14.10 pp`, short of `+15`. `lcb_975` copies `lcb_95`. `assignment_gates_pass` is false.

Tune reranker, window, and token cap on LoCoMo 0–4 and LME DEV only, against the token-matched control. Entity chain only on a cross-session `SUPERSEDE`. J4 spent LoCoMo 5–9 and LME TEST; a repeat is a post-J4 iteration. One unseen slice, family-wise error, judge off the answer model. Keep the original J4 TEST number beside it.

## Q3. SciFact only; 4096-d rows are not re-bound to Qwen3

`reencoded_8b=false`. FiQA and NFCorpus are `MISSING_4096`. Row identity used MiniLM-L6 at `max_seq_length=384` (min doc cosine `1.0`), not Qwen3-Embedding-8B on the 4096 rows. One 8B hash: model-card query instruction, raw documents, logged truncation. A second process re-encodes every query and 64 docs; cosine ≥ `1−1e-5` and text sha256. Fit excludes report-positive docs (`2,538/5,183`). Blind fit `0.586` exceeds primary ZeroMu `0.525`; freeze the salt first. DEV sets CSLS references and `(μ, σ)` only. Report domains separately.

Raw DOT, `k_os∈{32,64}`, then rescore. SciFact `μ/radius=0.733`. ZeroMu misses both drift LCBs (Procrustes `−1.0 pp`, LCB `−3.4`; QueryAffine `+3.7 pp`, LCB `−1.4`). The 1M win was recall@10 on a `99.48%` synthetic cone, not BEIR nDCG. FiQA (57,638 docs) is the anisotropy test.

## Q4. GRPO can grade the prompt and pull toward the base

`trained_grpo_checkpoint_present=false`. Families and slots are split (`scripts/colab_h100_70b_and_epistemic_grpo.py:96-126`). `noise-retractor` sits outside the disjointness hash (`:184-185`); include it. `ek5_epistemic_verifiability_reward(completions, episode, **_)` (`:316-322`) holds only when TRL passes those keywords. A positional `(prompts, completions)` call scores the prompt. Pin that signature in `verify-only`. Batch 4 does not divide `num_generations=8` (`:644-650`).

SFT is LoRA `r=32`, dropout `0.05`, lr `1.5e-4`, one epoch, gold label and value inside `<think>` (`:224-232`). GRPO reuses `sft_trainer.model` (`:658`). If the KL reference disables adapters, `beta=0.04` targets the base 7B. Freeze the SFT adapter as the reference; dropout off on rollout. Right verdict plus `ABSTAIN` on a T/F item is `+0.25` (`ek5_wfs_rlvr_env.py:56-59`, `:841-851`); wrong or an unclosed `<think>` is `−1` (`:273-274`).

Score Base, SFT, SFT+GRPO, and Bare-30B on the frozen D1 grader and, separately, on WFS exact-match. Pre-register 300 steps. DEV does not pick the checkpoint.

## Q5. Abstention is a substring; the write clock re-encodes the haystack

`grade_prediction` (`scripts/run_r6_pm3_and_d2_qdrant_csls_backfill.py:1106-1109`) accepts any marker substring (`:67-83`) and skips the judge (`:1292-1294`). `I don't have that information` and `I found 1 matching records` miss. `ABSTRACT` fails; `ABSTAIN` passes. `c8090214_abs` passes while naming the Holiday Market because it contains `not mentioned`. Closed lexicon, no extra entity, judge from another family, all four systems.

`chelated_write_ms` stops before the reader (`:1360-1369`) and re-encodes every turn: N=40 mean `500,198 ms`, tagged `zero_llm_write`. Mem0 `infer=true` is `430,209 ms` and `307,574` tokens; `infer=false` is `62,789 ms`. Pooled N=100 (`0.73` vs `0.69`, LCB `−6 pp`) mixes those clocks with the reused N=60 write of `4,582 ms`.

Same raw text, no gold-session filter, no family oracle, one shared cross-encoder or none. D1 N=60 accuracy, resurrection, and CONTESTED precision at t=0 and after a published delay; full LongMemEval and LoCoMo; one retraction and one laundered summary, scored by absence of the retracted value. Sidecar `3.82 MiB` beside a `3,906 MiB` index (re-embed `9.08 h`). Small-3B beats hetero-30B on D1 (`0.933` vs `0.900`) at `95.6 ms`. Measured moat: TRA-1 wrong-space refusal, masked resurrection `0.00` versus `0.60–0.765`, and mixed-shard CSLS. Until D5 and a prospective v4 split, this is an integration option.
