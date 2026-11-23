# Wave-5 Round-1 Architecture & Investor-Rigor Audit (`gemini-4-neon`)

**Scope:** Read-only code & manifest audit (`D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1`, `v4.0`).

## Q1. OpenSearch/Qdrant gRPC Interceptor & `D1 v2` Latency Gap

**Request-Only Failure:** Exact `ZeroMu` scoring is $s(q,d_j)=(\hat{q}^\top d_j - \hat{q}^\top \mu_d)/\|d_j-\mu_d\|_2$. Omitting denominator division—or adding $+\mu_y$ for cosine indexes—collapses into `DriftAdapter_QueryOnly_Affine`: `0.380` vs. `0.550` Recall@10 at `1M` vectors (`phase2_wave2_d1v2_and_d2_1m_manifest.json:227,263`) and `-26.9 pp` nDCG@10 on mixed-shard SciFact 20% backfill (`0.3392` uncalibrated vs. `0.6082` CSLS, `d2_4096d_qdrant_manifest.json:178,204`). Even under raw `DOT`, `k_os=10` without egress rescoring achieves only `0.864` Recall@10 of exact `ZeroMu` vs. `0.9988` (`k_os=32`) and `1.0000` (`k_os=64`).

**Bidirectional gRPC Contract:**
- **Ingress:** Enforce fail-closed `TRA-1` `BridgeContract` validation (`weights_sha256`, `hyperparameters_sha256`, `min_fit_count>=32`, `confidence>=0.70`, `spherical_migration_bridge.py:1368-1408`). Compute strictly causal pre-query EMA $\hat{q}=\text{norm}((q-\mu_q)Q)$ (updating $\mu_q$ post-score), rewrite `KnnQuery`/`SearchPoints` to $\hat{q}$ under `DOT/innerproduct`, and set `k_os >= max(64, 6k)`, `ef_search >= k_os`.
- **`EK5 v3` Retraction Trap:** Post-filtering collapses under echo floods; naive pre-filtering fractures HNSW connectivity when admission rate $p_{\text{admit}} < 0.3$. Use Lucene/Qdrant pre-filter bitsets with adaptive `ef_eff = min(ef_max, ceil(k_os / p_admit))` + exact-scan fallback, plus mandatory egress bitset filtering (`eligibility==ADMITTED`) so `0.00%` stale resurrection holds across lagged replicas.
- **Mixed-Shard Egress:** Run parallel HTTP/2 multiplexed queries to `bridged` ($\hat{q}$) and `native` ($q_{\text{new}}$) shards; rescore bridged hits via 4-byte `float32` $\|d_j-\mu_d\|_2$ sidecar, apply per-shard `CSLS` ($2s - r_d - r_q$, offline `DEV` $r_d$ + online shortlist $r_q$), and merge with canonical `(-score, doc_id)` tie-breaking.

**`D1 v2` `2,290.8 ms` Audit & `Demo D5`:** `Native_1024d` on Spark takes `2,281.9 ms` vs. `2,290.8 ms` for `Hetero_3090toSpark_ZeroMu` (`+8.9 ms` delta; local 3B is `95.6 ms`, `phase2_wave2...:108,144,168`). Over `99%` is **Nemotron-30B decode**, not JSON overhead. Claiming gRPC closes `2,290.8 -> 95.6 ms` fails DD. `Demo D5` must isolate transport+gateway from LLM decode: benchmark warmed HTTP/1.1 REST/JSON vs. Qdrant gRPC `:6334` + OpenSearch `KnnQuery` Protobuf across `RTX 3090 -> Spark` (`384/1024/4096-d`, `k_os={32,64}`), reporting wire bytes, serialization CPU, RTT, and `<0.1 ms` egress rescore separately from TTFT/decode.

## Q2. `PM-1b v4` Reranking & Post-`J4` Holdout Discipline

**Diagnosis (`r6_pm1b_manifest.json`, `chelated_conversational_v3.py`):**
1. **Context bloat:** `1,666` tokens on `LoCoMo` (`3,694` flat control) vs. `413` (`MiniLM`) collapses Type-5 abstention (`0.13` vs. `0.39`; `0.05` flat) despite higher evidence recall (`0.855` vs. `0.630`). On `LME`, `6,099` tokens depresses `SSU` (`0.857` vs. `0.900`) and preferences (`0.10` vs. `0.20`).
2. **`J4` cap bug:** `adaptive_rrf_gap=0.14` + `max_sessions=6` (`chelated_conversational_v3.py:368-416`) ran *before* `EK5` echo collapse, letting 6 echo sessions evict updated turns (`evidence_all_recall` `0.972 -> 0.708`, `KU` `0.756 -> 0.667`); `session_order="relevance"` broke chronology.
3. **`KU` LCB95 (`+14.10 pp` vs. `+15.0 pp`):** Almost all of `+25.64 pp` came from `BGE-M3` (`+24.36 pp` clean baseline); `EK5` added `+1.28 pp` because `6.1k`-token prompts diluted supersession tags.

**`v4` Fix & Holdout Accounting:** Run `EK5` echo collapse *before* session budgeting; rerank top-36 with `BAAI/bge-reranker-v2-m3`; compact to `~1.2k–2.0k` tokens (`±1` default window, chronological order on updates, no hard session cap on `KU`). Freeze all `v4` params strictly on `DEV` (SHA-256 pre-registered), label `LoCoMo 5–9`/`LME TEST` as **Post-`J4` Look #2 (Bonferroni $\alpha=0.025$)**, and add a virgin sealed prospective conversational drift holdout as primary confirmatory gate.

## Q3. `D2-4096d` Complete 3-Domain BEIR (`SciFact + NFCorpus + FiQA2018`)

Pin `Qwen3-Embedding-8B` commit SHA, `max_seq_length`, `bf16->fp32` cast, and **identical asymmetric query-instruction prompts** across all 3 domains. `verify_d384_cache_row_identity` (`run_r6_pm3...:555-618`) checks only `MiniLM`; embed `doc_ids`/`query_ids` SHA-256 digests inside each `.npz` and run sampled `K=256` cosine checks on **both `D384` and `D4096`**. On `SciFact`, `disjoint_fit_indices` (`0.5252`) suffered positive-doc deletion bias vs. `qrels`-blind hash anchors (`0.5864`, `d2_4096d_qdrant_manifest.json:303,638`): report **`qrels`-blind** primary and positive-excluded sensitivity, keeping `DEV`-only `CSLS`/`FastFill`. Benchmark live Qdrant gRPC (`:6334`), multi-point vector identity (`0.0` delta), `k_os in {10,32,64,128}` on `FiQA2018` (`57.6k` docs), and anisotropy against both `QueryOnly_Affine` and `Uncentered_Procrustes` (`0.535` vs. `0.525` on `SciFact`).

## Q4. Audit of `colab_h100_70b_and_epistemic_grpo.py` (`Demo D4`)

1. **SFT-to-GRPO KL Bug (L588–665):** Passing `sft_trainer.model` (already a `PeftModel`) into `GRPOTrainer(..., beta=0.04)` causes `disable_adapter()` to regularize against the **un-SFT'd base 7B** (or deep-copy 7B, OOMing Spark #2's `51 GB`). **Fix:** `merged = sft_trainer.model.merge_and_unload()`, then attach a fresh `peft_config` so $\pi_{\text{ref}}=\pi_{\text{SFT}}$ at zero extra VRAM.
2. **Padding & Batch Bugs (L562, L644–657):** Unset `tok.padding_side="left"` corrupts batched Qwen2.5 rollouts; `per_device_train_batch_size=4 < num_generations=8` breaks TRL divisibility. Set `per_device_train_batch_size=8` (or explicit `generation_batch_size=8`), `gradient_checkpointing=True`, `use_cache=False`.
3. **Abstention & Variance Collapse (L224–233, L684):** Determinate `VERDICT:U / ABSTAIN` yields `0.0` vs. `-1.0` wrong commit—for $p_{\text{correct}}<0.5$, unconditional abstention strictly dominates guessing! Single-template SFT also risks zero-variance rollouts. Cap SFT (`lr=5e-5`, `<=1` epoch), sample `T=0.8–1.0`, add a `-0.15` false-abstention penalty, log `abstain_rate`/`frac_zero_std`/KL, and checkpoint best-`DEV`.
4. **Zero-Shot `D1` Transfer:** Evaluate `Base 7B` vs. `SFT 7B` vs. `SFT+GRPO 7B` vs. `Bare 30B` on unseen `prospective_realworld_drift_corpus.py`.

## Q5. 4-Way Arena & Institutional Valuation Thesis

**`r6_pm3_manifest.json` Anomalies:** (1) `ABSTAIN_MARKERS` (`run_r6_pm3...:67-83`) omits `"do not contain"`, `"don't have"`, `"do not specify"`, falsely failing 5+ valid Chelated abstentions (`2311e44b_abs`, `60bf93ed_abs`, `6aeb4375_abs`, `982b5123_abs`, `bc8a6e93_abs`), while `4.3k`-token bloat induced `"I found N matching records"` hallucinations (`mem0_infer_true` emitted `"ABSTRACT"` 3x). (2) `chelated_write_ms = 500,198 ms` is a harness bug: `CUDA_VISIBLE_DEVICES=""` (L1454) forced `BGE-M3` onto CPU while 6 threads queued on one `enc_lock` (L1353) inside the `t0` timer (`88s` uncontended -> `585s` queued).

**Zero-Strawman 4-Way Arena:** Benchmark `Chelated v4`, `Mem0 2.2.1`, OSS `Vectorize Hindsight` (`Retain/Recall/Reflect`), and clean-room `OpenSearch Memory`-parity async `SEMANTIC/SUMMARY` consolidation on identical SHA-256 timelines, shared reader LLM, GPU embeddings, lock-free timers, and calibrated abstention judging across **Enterprise Regulated Drift/Security** and **Conversational Memory**.

**DD-Proof Valuation Thesis:** Anchor on three auditable moats: (1) **Gross margins**—zero-LLM write path (`4.3k` vs. `307.6k` tok = `71x` lower burn) decouples COGS from `100M+` ingestion (`>85%` software GM vs. `40–60%` LLM-wrapper margins); (2) **Drop-in wedge**—`ZeroMu+CSLS` gRPC Gateway eliminates `900+` GPU-hour re-embeds and `1023x` sidecar RAM on live OpenSearch/Qdrant clusters; (3) **Defensible IP**—3-valued `WFS`, `CITES_AUDIT` anti-laundering, and bitemporal `as_of` auditability create an irreproducible compliance control plane for strategic acquirers (AWS, Elastic, Qdrant, Snowflake).
