# Wave-5 R1 — claude-opus-5-5 (high): Architecture & Investor-Rigor Memo

## Headline corrections
1. **The 2,290.8 ms is reader time, not transport.** Native 1024-d (no bridge, no cross-node vectors) takes **2,281.9 ms**: gap **8.9 ms**. The 95.6 ms is a 3B reader on the 3090.
2. **ZeroMu ≈ a plain rotation.** ZeroMu − Procrustes: **−0.99 pp (LCB −3.4)** on SciFact-4096; +1 pp (LCB −3) at "1M".
3. **The migration is not zero-GPU.** The bridge fit needs paired encodings: `n_fit_docs=2,538/5,183`, about 49% of the corpus. "1M" is 99.48% synthetic.
4. **D2-4096d row identity is unverified.** `row_identity_verification` ran on `all-MiniLM-L6-v2`, so only the 384-d cache was checked, and `reencoded_8b:false`. `max_abs_dot_vs_numpy = 2.68` is not rounding; explain it before claiming HNSW parity.
5. **The mixed-shard backfill never stores old-space docs.** In `run_r6_pm3_and_d2_qdrant_csls_backfill.py` lines 703–707, both `zeromu_scores` and `report_native` score against `docs_high`. A real migration has un-backfilled docs in 384-d only.

## Q1 — Gateway contract
- **Request-only rewriting.** Within one query, ZeroMu differs from raw `d·q̂` only by the `1/n_j` divisor; `μ_d·q̂` is a per-query constant. The divisor's CV is about 4.7% (0.038/0.807). Egress rescoring is worth about +0.7 pp (HNSW rescore 0.516 at `k_os=10` vs 0.525 at `k_os=32`). Request-only degenerates to *Procrustes*, not QueryAffine.
- **Minimal bidirectional contract.**
  - Ingress sends `q̂`, `k_os` and a filter `{admitted, epoch≤E}`.
  - Egress returns ids, DOT scores and 8 B/doc of payload (`n_j`, CSLS `r_d`) plus a `shard_tag`. Never ship 16 KB vectors.
  - **Certified stop:** fetch more until the last hit has `s−c < (s₁₀−c)·n_min/n_max`.
  - Today `hnsw_ef ≡ k_os`, which confounds them. Sweep the two independently.
- **Pre- vs post-filter.** Retracted twins sit next to their originals, so a post-filter needs `k_os/(1−r_local)`. Measure HNSW recall against exact under filterable HNSW at 0/10/50/90% retraction.
  - "0.00% resurrection" needs `wait=true` acks and epoch-pinned reads: a consistency contract.
- **Mixed shards.** Run a tagged dual query, with the bridged `q̂` on old shards and the native query on new ones. CSLS `r_d` must be causal, not DEV-derived.
- **D5.** Fixed or no reader.
  - Grid: REST vs gRPC × pooled/unpooled × {384, 1024, 4096}-d × batch {1, 32}, same NIC, ≥10k open-loop queries, p50/p95/p99 with CIs.
  - Expect a few ms. The OpenSearch gRPC `KnnQuery` and "Uber" claims are unverified; pin the version.

## Q2 — PM-1b v4
- **Diagnosis.** The clean LME gain is **entirely DualGran**. The EK5 clean delta is 0.00 pp, and on DEV `Chelated_v3_DualGran == CleanBaseline` exactly.
- **Circularity.** One Nemotron endpoint paraphrases the stress fixture, answers *and* judges, and the echoes are self-injected. Add a cross-family judge and a blind human audit.
- **Accounting.**
  - LoCoMo 5–9 and LME TEST are spent. Label v4 "post-J4 exploratory, burned holdout".
  - Freeze the config on DEV; confirm on a newly sealed set.
  - One primary endpoint, Holm for the rest. Chasing a 0.9 pp miss is a forking path.
- **`lcb_975 == lcb_95` in every manifest.** Fix it before any gate check.

## Q3 — D2-4096d, three domains
- **Encoding.** Qwen3-Embedding-8B with the `Instruct:` prompt on queries only, last-token pooling with left padding, L2 normalization. Log seq length, dtype and batch order. FiQA and NFCorpus also lack 384-d caches.
- **Identity.** Re-encode ≥1,000 stratified docs plus all queries. Require cos ≥0.999, NN-top1 = 1.0 and stored text SHAs.
- **Splits.** Keep `QUERY_SPLIT_SALT`. The qrels-blind fit (0.586) includes REPORT positives, which is transductive; disclose it or exclude them.
- **Statistics.** 54 comparisons. Primary: CSLS − DEV z-score at fraction 0.2, domain-stratified, with Holm for the rest. Drop the uncalibrated strawman.

## Q4 — GRPO script bugs
1. **The KL reference is the base model, not SFT.** GRPO on the SFT PeftModel uses the adapter-disabled base as reference. Fix: `merge_and_unload()`, then a fresh LoRA.
2. **SFT and GRPO share `train_eps`.** Memorized rollouts give zero reward variance, so the advantage is zero. Use fresh seeds, log `frac_reward_zero_std`, and abort above 0.8.
3. **Versions are unpinned.** Batch 4 vs `num_generations=8` breaks older TRL; default `max_prompt_length` can truncate the system prompt.
4. **Eval/train length mismatch.** Eval uses `max_new_tokens=192` against 224 in training, so truncated `<think>` blocks score −1.
5. **SFT loss includes the prompt.** Use completion-only loss. The CoT leaks oracle `store_slot_status`.
6. **Part 1 vLLM.**
   - AWQ with `bfloat16` should be fp16.
   - **`max_tokens=48` also truncates JSON claim extraction**, which handicaps Chelated.
7. **No D1 prose transfer eval exists.** Pre-register it; ≥3 seeds.

## Q5 — PM-3, arena, valuation
- **Abstention grading.** Substring `ABSTAIN_MARKERS` miss "I don't have that information" and "The records do not contain…". At least 4 Chelated refusals were scored false (3/20 → ~7/20). "I found N matching records" appears in *both* arms but not in `READER_PM3_SYSTEM`, so audit the endpoint for template or state leakage. Re-grade all arms blind.
- **`chelated_write_ms ≈ 500 s` is an artifact.** It is CPU BGE-M3 (`CUDA_VISIBLE_DEVICES=""`, L1454) behind an `enc_lock` shared by 6 workers, and it includes retrieval.
  - Equalize encoder and device; time write and read separately; never pool reused N60 with N40.
  - Against Mem0 infer=True, LCB is −0.06: **no win**.
- **Arena.**
  - First reproduce each vendor's published number in its own harness, with vendor configs and a shared LLM and judge.
  - Use real OpenSearch Memory or label ours as ours.
  - Metrics: per-category accuracy, abstention P/R, stale resurrection, write/read p95, $/1k ops, staleness at Δ ∈ {0, 1, 10, 60} s.
  - Regulated-drift data must be external and sealed (OSV/NVD, SEC 10-K/A).
- **Valuation.** Diligence will find items 1–5, so claim no accuracy moat.
  - The asset is a **deterministic, replayable 3-valued store**: Tarjan-SCC WFS, `as_of` reads and zero write-path LLM tokens.
  - TCO per 100M events = writes×(LLM tokens + embed) + reads×(tokens + ANN) + storage + bridge-fit re-encode (~50%). Parameterize it; invent no prices.
  - Wedge: compliance auditability via an open-core gateway; AWS could ship `CONTESTED` within a quarter.
