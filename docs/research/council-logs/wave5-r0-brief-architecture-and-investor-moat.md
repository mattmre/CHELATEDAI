# Wave-5 Advisory Council Brief: OpenSearch/Qdrant gRPC Gateway, Remaining Gate Closure (`PM-1b`, `PM-3`, `D2-4096d`, `D4-GRPO`), & Shareholder Value vs. Mem0 / Hindsight / OpenSearch Memory

**Date:** 2026-11-22 (Overnight Wave 5)  
**Baseline:** `v4.0` (`HEAD` `5a9bd46f`, `35/35` verified receipt bundles, `41/41` side hashes, `registry_sha: 5603e923c5e3`)  
**Role of Frontier Council (`gpt-6.2-sol`, `claude-opus-5-5`, `grok-4.7`, `gemini-4-neon`):**  
Strictly read-only 2-round **Peer Analysis (Round 1)** + **Cross-Judgment Adjudication (Round 2)** to lock methodology, prevent leakage/strawmen, identify subtle systems pitfalls, and sharpen our investor/shareholder value architecture before Wave-6 GPU execution.

---

## 1. Context & Open Validation / Strategic Targets

### A. Carbon's OpenSearch gRPC + Amazon OpenSearch Memory Hypothesis (Needs Rigorous Stress-Testing)
A separate Gemini Carbon session reviewed our `D1 v2` & `D2` results alongside two recent industry developments:
1. **OpenSearch 3.5–3.9 Native gRPC `SearchService` / `KnnQuery` Protobuf Transport** (co-engineered with Uber Engineering) and **Qdrant gRPC (`:6334`)**: eliminates REST/JSON float-array serialization overhead (`~53–65%` payload reduction, HTTP/2 connection multiplexing).
2. **AWS Amazon OpenSearch Memory (launched Nov 2026)**: serverless agentic memory using asynchronous background Bedrock LLM passes (`SEMANTIC`, `SUMMARY`, `CUSTOM` strategies) over hybrid BM25 + k-NN (`~93.0%` reported on standard `LongMemEval` at `~13,500` input tokens/query vs `~167,000` full-context).

**Carbon's Proposed Product & Systems Hook:**
- Position Chelated AI as a **Drop-In gRPC Proxy / Interceptor (`Chelated Gateway`)** in front of OpenSearch `SearchService.KnnQuery` or Qdrant `Points.Search`, simultaneously:
  1. Closing our `D1 v2` cross-node latency gap (`2,290.8 ms` on `RTX 3090 -> Spark` vs `95.6 ms` local) by moving cross-node `384/1024/4096-d` float vectors and `EK5 v3` admission bitsets over gRPC Protobuf; and
  2. Giving Fortune 500 OpenSearch/Qdrant customers **zero-GPU encoder migration (`ZeroMu` 4-byte sidecar + `CSLS` progressive backfill)** and **`0.00%` synchronous policy-resurrection / 3-valued `CONTESTED` guarantees** without ripping out their existing vector DB.

**Critical Engineering Questions We Need the Council to Audit (`Q1`):**
- **Pitfall 1 (Request-Only vs. Bidirectional Interceptor):** If a naive gRPC proxy only rewrites `KnnQuery.vector -> q_hat` on the request path without rescoring `SearchResponse` hits by `1 / ||d_j - mu_d||_2` + `CSLS` hubness, doesn't it degenerate into `DriftAdapter_QueryOnly_Affine` (`0.380` vs `0.550` at `1M`; `-26.9 pp` during mixed-shard backfill)? What is the exact minimum-latency **bidirectional (Ingress + Egress)** gRPC interceptor contract (`k_os` oversampling, `EK5 v3` pre-filter vs post-filter HNSW graph connectivity under heavy retractions, mixed-shard dual-query or tagged-shard routing)?
- **Pitfall 2 (`D1 v2` `2,290.8 ms` Root-Cause Decomposition):** How much of `2,290.8 ms` in `D1 v2` was JSON float serialization vs. sequential unpooled HTTP/SSH calls + LLM generation, and what exact controlled `REST/JSON vs gRPC Protobuf` + `End-to-End Cross-Node Gateway` benchmark (`Demo D5`) must we run so our latency claims are unassailable?

---

### B. Remaining Open Technical Gates & Stretch Proofs (`Q2–Q4`)
1. **`Q2` — `PM-1b v4` Conversational SOTA Closure (`artifacts/r6_pm1b_conversational_sota/r6_pm1b_manifest.json`):**
   - In Wave 4 (`J4`), `Chelated_v3_DualGran` hit `+25.64 pp` mean delta on `LME` `knowledge-update` echo stress (`0.7564` vs `0.5000` `Legacy_MiniLM`), missing `LCB95 >= +15.0 pp` by just `0.9 pp` (`+14.10 pp`), while `single-session-user` (`60/70` vs `63/70`) and `LoCoMo` suffered from stuffing ~`6,100` tokens of un-reranked session turns into the reader, and `J4`'s hard `3–6` session cap hurt multi-session `KU` (`0.6667`).
   - **Proposed `v4` Fix:** Add **`BAAI/bge-reranker-v2-m3`** cross-encoder reranking over top Dual-Granularity candidates + focused turn-window extraction (`±2` turns around reranked hits + chronological entity/topic state chain when `EK5` detects cross-session updates, without hard session-count truncation on `KU`), tuned **strictly on `DEV` (`LoCoMo` convs `0–4` + `LME` SHA-256 DEV half)** and evaluated in a **single locked shot on held-out `LoCoMo` convs `5–9` + `LME` TEST** (plus secondary full `N=300/500`). Note: Since `J4` already ran once on `LoCoMo 5–9` + `LME TEST`, how must we label `v4` (`Wave-6 post-J4 architectural iteration` with prospective unseen conversational drift or strict holdout accounting) so we never commit silent multiple-testing?
2. **`Q3` — `D2-4096d` Complete 3-Domain BEIR (`SciFact`, `NFCorpus`, `FiQA2018`) (`artifacts/phase2_d2_qdrant_hnsw_csls_backfill/`):**
   - `SciFact` passed row-identity (`min_doc_cos = 1.000000` at `max_seq_length=384`), Qdrant HNSW (`100%` @ `ef=64`), and `CSLS` (`LCB95 > 0` vs raw, `DEV z-score`, and `RRF`), while `FiQA2018` and `NFCorpus` were `MISSING_4096`. Both datasets and `Qwen3-Embedding-8B` are cached on Spark #1 (`10.0.0.49`). Encoding `NFCorpus` (`3,633` docs) and `FiQA2018` (`57,638` docs) completes all 3 requested BEIR domains (`0` missing domains) and tests higher-anisotropy / larger-corpus `4096-d` behavior!
3. **`Q4` — `Demo D4` Live Trained `Epistemic-WFS-GRPO` LoRA (`scripts/colab_h100_70b_and_epistemic_grpo.py`):**
   - Currently `trained_grpo_checkpoint_present = false`. Spark #2 (`10.0.0.194`) has `51 GB` free unified Blackwell GPU RAM and cached `Qwen/Qwen2.5-7B-Instruct`. We can execute real `SFT` vs `SFT + TRL GRPOTrainer` LoRA training on the `J1`-hardened disjoint `TRAIN / DEV / EVAL` WFS splits (`0` entity/slot/value/scenario overlap, gated `<= 0.05` format bonus) AND test zero-shot transfer onto real-world unseen `D1` prose (`Base 7B` vs `SFT 7B` vs `SFT+GRPO 7B` vs `Bare 30B`)!

---

### C. Competitive Arena & Shareholder Value Thesis vs. `Mem0`, `Vectorize Hindsight`, & `AWS OpenSearch Memory` (`Q5`)
- **Competitor Landscape:**
  1. **`Mem0` (`mem0ai 2.2.1`, $24M Series A):** Usage-metered Adds/Retrievals; `infer=True` runs expensive synchronous LLM extraction/reconciliation (`307,574` prompt tokens & `430s` write latency per `LME` timeline in `r6_pm3`; `46.7%` on peer-conflict `ABSTAIN`).
  2. **`Vectorize Hindsight` (`vectorize-io/hindsight`):** Open-source 4-network agent memory (`World Facts`, `Agent Experiences`, `Synthesized Entity Summaries`, `Evolving Beliefs` with scalar confidence scores) + 3 operations (`Retain`, `Recall`, `Reflect` over `pgvector`/BM25/graph). Strong on casual chat (`~90%+` reported with frontier backbones), but relies on LLM `Retain`/`Reflect` loops and scalar belief confidence rather than deterministic 3-valued `WFS` stratification, `CITES_AUDIT` vs `D_V` anti-laundering, or zero-GPU `ZeroMu` vector migration.
  3. **`AWS Amazon OpenSearch Memory`:** Managed serverless `SEMANTIC`/`SUMMARY` async Bedrock consolidation (`~13,500` read tokens/query on `LongMemEval`, eventual consistency staleness window, zero cross-encoder geometric bridging, zero formal `CONTESTED` truth state).
- **Questions for `Q5`:**
  1. Looking at `r6_pm3_manifest.json` (`N=40` `LME` slice where `Chelated` output `"I don't have that information"` or `"I found 1 matching records"` and got marked `false` on `abstention` items while `mem0_infer_true` output `"ABSTAIN"` or `"ABSTRACT"`), what caused that artifact and how do we run a **100% fair, steelmanned 4-way Arena (`Chelated v4` vs `Mem0` vs `Hindsight` vs `OpenSearch Memory`-style Consolidation)** across both **Enterprise Regulated Drift/Security** and **Multi-Session Conversational Memory**?
  2. What makes a **credible, institutional-grade Investor & Shareholder Valuation Thesis** (unit economics, gross-margin expansion, TCO per `100M` events, defensible IP moat, and build-vs-buy wedge against AWS/Mem0/Hindsight) vs. naive pitch hype that sophisticated VCs or acquirers would reject?
