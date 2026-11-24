# Wave 10 Round-0 Master Engineering Specification (`Demos D6–D10`: Tier 1 & Tier 2 Investor Flagships)

- **`panel_type`:** `architecture-planning` (`$panel-of-experts` Mode A + Mode B)
- **`artifact`:** `docs/research/council-logs/wave10-r0-tier1-tier2-master-spec.md`
- **`run_date`:** `2026-11-23`
- **`run_id`:** `wave10-r0-d6-d10-spec`
- **`challenge_depth`:** `1` (Advisor + 4-Model Factory Challenge -> Guardrail Synthesis `L1–L5`)
- **`active_experts`:** `["gemini-4-carbon (Advisor/Judge)", "claude-fable-5-2 (Advisor/Judge)", "claude-sonnet-5-5 (Factory)", "gpt-6.2-sol (Factory)", "gemini-4-neon (Factory)", "grok-4.7 (Factory)"]`
- **`persist_version`:** `1.1.0`

---

## 1. Objective & Non-Negotiable Zero-Stub Contract

We are closing **all five remaining Tier-1 (commercial diligence) and Tier-2 (scale & model-efficiency moat) demonstrations (`Demos D6–D10`)** as cryptographically sealed **`Part VII`** bundles (`45/45` total across `verified_registry.json` `v6.0`) plus a live buyer-ready **Regulated Enterprise Compliance Copilot** application.

**Zero-Stub Mandate:**
1. Every external database/framework claim MUST execute against a **real containerized engine** with pinned image digest (`docker inspect --format '{{index .RepoDigests 0}}'`), real SQL/REST/gRPC/Bolt queries, and zero mock/emulated substitutes masquerading as live services.
2. Every existing `v5.0` receipt bundle (`40/40` in `Parts I–VI`) and running fleet service (`nemotron-37c3` `:8891`, `nemotron-6724` `:8890`, `glm53-remote-nfs`, `nemotron-eba1` `:8801`, user services on Spark #3) MUST remain 100% untouched.
3. Parallel factory delegates (`Sonnet 5.5 High`, `Sol 6.2 High`, `Neon 4 High`, `Grok 4.7 XHigh`) operate under **strictly disjoint file ownership** during Wave 11 implementation, followed by **cross-model peer validation** and **`Carbon 4 Medium` + `Fable 5.2 Medium` final adjudication** in Wave 12.

---

## 2. Concrete Specifications & Disjoint File Ownership (`Demos D6–D10`)

### `Demo D6` (Tier 1 — Live Multi-Backend Drop-In Appliance: Real `OpenSearch 3.0.0` Java Cluster + Real `pgvector:pg17` + Real `Qdrant`)
- **Closes `v5.0` Gap:** `live_opensearch_java_cluster_claimed = false` in `D5` + proves Chelated is a **universal multi-backend control plane** across the top 3 enterprise vector engines (`AWS/OpenSearch`, `PostgreSQL pgvector`, `Qdrant`).
- **Factory Implementation Owner:** **`gpt-6.2-sol` (High)** (`chelated_multibackend_appliance.py`, `scripts/run_wave11_d6_opensearch_pgvector_qdrant.py`, `tests/test_wave11_d6_multibackend.py`).
- **Output Bundle:** `artifacts/wave11_d6_opensearch_pgvector_qdrant/` (`wave11_d6_multibackend_manifest.json` + `report_receipt.json`).
- **Live Hardware & Container Topology (`Spark #1` `10.0.0.49`, `--memory=8g` each, cleaned up cleanly after receipt verification):**
  1. **Live `opensearchproject/opensearch:3.0.0` (`linux/arm64` Java 21 cluster):** container `wave11-d6-opensearch` on `127.0.0.1:19200` (REST/HTTP) & `19600`, `discovery.type=single-node`, `DISABLE_SECURITY_PLUGIN=true`, `OPENSEARCH_JAVA_OPTS=-Xms4g -Xmx4g`, `knn.algo_param.index_thread_qty=4`, Lucene/Faiss HNSW `knn_vector` (`space_type="innerproduct"`). Also probe whether OpenSearch 3.0 `transport-grpc` auxiliary port is enabled/available and report exact wire protocol per backend honestly!
  2. **Live `pgvector/pgvector:pg17` (`linux/arm64` PostgreSQL 17 + `CREATE EXTENSION vector`):** container `wave11-d6-pgvector` on `127.0.0.1:15432` (`CREATE INDEX ... USING hnsw (embedding vector_ip_ops) WITH (m = 16, ef_construction = 128)`), queried with `SET LOCAL hnsw.ef_search = ...` and `ORDER BY embedding <#> $1 LIMIT $2`, returning ONLY `(doc_id, -1 * (embedding <#> $1) AS raw_dot)` (`0 B/doc` vector egress!).
  3. **Live `qdrant/qdrant` (`linux/arm64` gRPC):** container `wave11-d6-qdrant` on `127.0.0.1:16333/16334` (`with_vectors=False`, `with_payload=False`).
- **Unified Benchmark Matrix (Real BEIR `SciFact` `5,183` docs & `FiQA2018` `57,638` docs, `D384 <-> D4096` Qwen3-8B):**
  - Dual-shard `20%` backfill (`80%` `384-d` legacy shard + `20%` `4096-d` native shard) across **all 3 live engines** (`OpenSearch 3.0`, `pgvector pg17`, `Qdrant`).
  - Arms per engine: `Uncalibrated_Raw_Merge`, `DEV_ZScore_Merge`, `RRF_k60`, and `Chelated_ZeroMu_CSLS_8B_Sidecar` (`TRA-1` ingress guard + `8 B/doc` sidecar + `EK5 v3` epoch-pinned read fence).
  - Retraction & Lag Matrix (`0%`, `10%`, `50%`, `90%` retraction densities, tuning adaptive `k_os` / pre-filter / exact fallback so **even 50% retraction hits `Recall@10 >= 0.995`**, closing the `0.9942` `D5` miss!): `0.00%` stale resurrection across all 3 engines, plus `p50/p95/p99` service RTT & vector-egress bytes (`0 B/doc` vs full `4096-d` egress).

### `Demo D7` (Tier 1 — Live Upstream `Zep / Graphiti` (`graphiti-core 0.31.2`) + Live `Neo4j 5.26` vs. `Chelated v4` & Live `Mem0 2.2.1`)
- **Closes `v5.0` Gap:** `REQUIRES_EXTERNAL_GRAPH_DAEMON` for `Zep / Graphiti` (`graphiti-core`) — upgrading our competitor arena so **both** top VC-backed open-source agent memory incumbents (`Mem0` AND `Zep/Graphiti`) are tested as **100% live upstream libraries** against real backing databases (not architectural emulations!).
- **Factory Implementation Owner:** **`grok-4.7` (XHigh)** (`scripts/run_wave11_d7_live_zep_graphiti_arena.py`, `tests/test_wave11_d7_zep_graphiti.py`).
- **Output Bundle:** `artifacts/wave11_d7_live_zep_graphiti_arena/` (`wave11_d7_zep_graphiti_manifest.json` + `report_receipt.json`).
- **Live Stack (`Spark #1` `10.0.0.49` `.venv-wave5`):**
  - Ephemeral `neo4j:5.26-community` container `wave11-d7-neo4j` on `127.0.0.1:17474` (HTTP) & `127.0.0.1:17687` (Bolt), `--memory=8g`, `NEO4J_AUTH=neo4j/chelated_bench_2026`.
  - Upstream `graphiti_core==0.31.2` (`Graphiti(uri="bolt://127.0.0.1:17687", ...)`), wired to our local OpenAI-compatible vLLM (`Nemotron-3.5-30B` on `:8891` with `enable_thinking=False` + structured JSON compatibility adapter if `Nemotron` formatting needs steelmanning, logged transparently!) and local GPU `BAAI/bge-m3` embedder + reranker.
- **Head-to-Head Evaluation:**
  - **Suite A (`N=60` Regulated Enterprise Drift: CVE/SEC/SRE supersession, rumor laundering, and symmetric peer-authority conflict `U`):** Compare `Chelated_v4`, `Live_Graphiti_Zep_v0_31_2`, and `Live_Mem0_v2_2_1_InferTrue` on:
    1) Post-ingestion blind-judged accuracy (`BlindCrossFamilySemanticJudge` using `Qwen2.5-7B-Instruct` on `RTX 3090` / `Spark #1`),
    2) Peer-authority conflict abstention accuracy (`U` subset),
    3) Immediate `Delta=0s` retraction visibility (querying synchronously at `t_retract + 0 ms` before Graphiti's multi-turn LLM episode edge extraction finishes vs Chelated's `<1 ms` synchronous `EK5 v3` epoch fence!),
    4) Exact LLM `write_tokens`, `read_tokens`, `write_latency_ms`, and `read_latency_ms`,
    5) Paired bootstrap `LCB97.5`, Tango score CI, AND exact Clopper–Pearson discordant CIs (satisfying Wave-8 `M1`!).
  - **Suite B (`N=20` Temporal Multi-Session `LongMemEval-S` Knowledge-Update Subset):** Measure temporal conversational accuracy + write/read token burn (`Graphiti` heavy entity/edge extraction LLM calls vs `Chelated` `0` write LLM tokens).

### `Demo D8` (Tier 1 — Buyer & Investor Reference App: "Regulated Enterprise Compliance Copilot" + Non-NDA Black-Box Mode)
- **Closes Commercial Diligence Gap:** Turns our engine + workbench into a self-explanatory, executive-ready **Enterprise Vertical Application** that a bank CIO, healthcare CISO, or VC partner can click through in 90 seconds.
- **Factory Implementation Owner:** **`claude-sonnet-5-5` (High)** (`demo_workbench/compliance_copilot_engine.py`, `demo_workbench/compliance_copilot.html`, `tests/test_wave11_d8_compliance_copilot.py`). *(Additive route mount `/copilot` and `/api/v2/copilot/*` into `demo_workbench/server.py` + nav link in `demo_workbench/index.html` wired cleanly once `compliance_copilot_engine.py` passes unit tests.)*
- **Output Bundle:** `artifacts/wave11_d8_compliance_copilot/` (`wave11_d8_copilot_manifest.json` + `report_receipt.json` capturing deterministic end-to-end execution, latency, SHA-256 audit receipt verification, and `0/55` Non-NDA IP-leak verification in Black-Box mode).
- **Mandatory Capabilities (Zero Stubs):**
  1. **Three Real End-to-End Regulated Workflows (plus Custom Live Document/Retraction Injection):**
     - **Workflow 1 — Financial Services (`SEC Form 8-K Item 4.02` Non-Reliance Restatement & Peer Analyst Deadlock):** Initial `10-Q` revenue guidance $\to$ viral sell-side chat echoes $\to$ authoritative `SEC 8-K` restatement $\to$ optional conflicting Co-Lead Auditor dispute (`CONTESTED` human-escalation trigger).
     - **Workflow 2 — Privacy & Legal (`GDPR Article 17` Right-to-Erasure & Derived-Summary Cascade):** Raw customer PII ticket + 3 downstream AI-generated CRM summaries (`DERIVES_VALUE` / `COPY_OF`). Shows how naive vector delete removes ONLY the raw chunk while leaving derived summaries leaking PII, whereas Chelated cascades synchronous revocation across the entire value-derivation lineage in `<2 ms`!
     - **Workflow 3 — Life Sciences & Cyber (`FDA Clinical Protocol Dose Withdrawal / Critical CVE False-Positive Retraction` + `CITES_AUDIT` Investigator Survival):** Retracted dosing/advisory blocked from clinical/SRE copilot (`0.00%` resurrection at `Delta=0s`), while internal Safety Investigator memo citing the retracted record for root-cause postmortem (`CITES_AUDIT`) survives intact!
  2. **Live Side-by-Side Execution:** Real `FastBM25Index` + real dense/hybrid vector similarity + real `StratifiedEvidenceStore` (`EK5 v3` + alternating-fixpoint oracle cross-check) + optional live local LLM reader (`Qwen2.5-3B/7B` or `Nemotron-30B` when reachable, with deterministic extractive grounded answer synthesis always available offline in `<10 ms`) comparing:
     - **Pane A:** `Standard Hybrid RAG` & `Async Overwrite Memory (Mem0/Summary Style)` (shows exact leaked retracted claims / PII / silent conflict guessing)
     - **Pane B:** `Chelated Governance Control Plane` (shows `Delta=0s` block, `CONTESTED` escalation banner, authority lineage graph, and latency meter).
  3. **Live Presentation Mode Toggle (`NDA Technical Diligence Mode` $\leftrightarrow$ `External Investor Non-NDA Black-Box Mode`):**
     - In `Non-NDA Black-Box Mode`, ALL internal method names/acronyms (`EK5`, `WFS`, `Well-Founded`, `Procrustes`, `ZeroMu`, `CSLS`, `TRA-1`, `Stratum`, `D_V`, `CITES_AUDIT`, etc.) are scrubbed from both API responses (`?mode=non_nda`) and DOM text, verified programmatically against `verify_non_nda_ip_clean.py`'s `55` forbidden regex patterns!
  4. **Cryptographically Signed Compliance Audit Receipt Export:** Generates downloadable JSON + printable executive HTML/PDF Compliance Certificate with SHA-256 event-ledger hash, timestamp, revoked lineage IDs, and independent oracle verification status.

### `Demo D9` (Tier 2 — Compact `0.5B/1.5B` LoRA Fast Epistemic Extractor `<1.5s/case` + Curriculum `Epistemic-WFS-GRPO` `U`-Abstention Closure)
- **Closes Two Major Wave-8 Technical Gaps:**
  1. Cuts Suite A unstructured prose ingestion latency from **`9.69 s/case` (`6,883` 30B write tokens)** down to **`<1.5 s/case` (`>6.5×` speedup, `0` 30B write tokens)** via a fine-tuned compact **`Qwen/Qwen2.5-0.5B-Instruct` (and/or `1.5B-Instruct`) LoRA Epistemic Claim Extractor** while maintaining `>= 93.3%` Suite A end-to-end accuracy!
  2. Closes the Wave-6 `D4` **`EVAL` `false_commit_rate_on_u = 0.875` & `GRPO - SFT` non-significance (`p=0.45`)** gap by diagnosing *why* 16-step `D4` failed on `EVAL` `U` (`odd_refutation_cycle` and `contested_successor` topologies had structural features absent from `SFT_WARMUP`'s 2 simple `U` templates, and 16 optimizer steps was `12×` too short for RL exploration!) and training a **longer-horizon (`160` steps) multi-seed `Epistemic-WFS-GRPO` LoRA** on `Spark #2` (`10.0.0.194`) while keeping `EVAL` (`N=120`) and `D1` (`N=60`) **100% untouched, byte-identical, and entity/slot/value/template/seed-disjoint**!
- **Factory Implementation Owner:** **`gemini-4-neon` (High)** (`chelated_fast_edge_extractor.py`, `scripts/run_wave11_d9_fast_extractor_and_u_grpo.py`, `tests/test_wave11_d9_extractor_and_grpo.py`).
- **Output Bundle:** `artifacts/wave11_d9_fast_extractor_and_u_grpo/` (`wave11_d9_extractor_grpo_manifest.json` + `report_receipt.json`).
- **Hardware Placement:**
  - **Part A (`Qwen2.5-0.5B-Instruct` & `Qwen2.5-1.5B-Instruct` LoRA Extractor):** Train on **Desktop `RTX 3090` (`24 GB` VRAM)** (where `Qwen2.5-0.5B-Instruct` is already cached!) on `N=600` synthetically generated domain-disjoint raw operational documents (zero overlap on entities, slots, values, or templates with `prospective_realworld_drift_corpus.py` `N=60` — verified by automated disjointness gate!), then evaluate batched local extraction across all `1,020` raw documents (`N=60` cases) of `Suite A` (`D1`), measuring exact wall-clock `ms/case` (`target < 1,500 ms/case` vs `9,690 ms/case`), JSON validity (`target >= 99%`), and blind-judged end-to-end accuracy (`target >= 93.3%`, beating live `Mem0 81.67%` with `LCB97.5 > 0`!).
  - **Part B (`160`-Step Curriculum `Epistemic-WFS-GRPO` on `Qwen2.5-7B-Instruct`):** Execute on **`Spark #2` (`10.0.0.194` `.venv-rlvr`)** across all 3 pre-registered seeds `(42, 137, 2026)`:
    - Preserve exact `EVAL` (`N=120`) and `D1` (`N=60`) test sets from Wave-6 `D4` (`0` overlap verified!).
    - Diagnose and fix the `U`-exploration bottleneck cleanly without touching `EVAL` templates: calibrate `U`-vs-`T/F` group sampling balance, asymmetric false-commit-on-`U` penalty (`-1.25` for hallucinating a value when `gold == U` vs `-0.20` cautious abstention on `T/F`), structural graph-trace `<think>` SFT format scaffolding (teaching explicit cycle/successor-contest verification rules on abstract graph motifs without leaking `EVAL` scenario generators), and `120–160` GRPO steps (`8–10×` Wave-6's 16 steps).
    - Report `Base 7B`, `SFT 7B`, and `SFT + Curriculum-GRPO 7B` (3 seeds) on `EVAL` (`N=120`), `EVAL U-only` (`n=40`), and zero-shot `D1` (`N=60`) with `paired_bootstrap_wave6` and `holm_bonferroni_stepdown`.

### `Demo D10` (Tier 2 — `5,000,000+` Real-Document Corpus Zero-Downtime Cutover + `72B` Frontier Model Scale Sweep)
- **Closes Two Major Investor Scale Objections:**
  1. Upgrades `D2-1M` (`99.5%` synthetic Gaussian perturbations) to a **`5,000,000+` (`5.02M`) 100% REAL natural-language document corpus** (`0.0%` synthetic vectors!) with official human-judged BEIR `qrels` across **7 real domains** (`MSMARCO` `~4.40M` real web passages + `Quora` `522.9k` + `FiQA2018` `57.6k` + `SciDocs` `25.7k` + `ArguAna` `8.7k` + `SciFact` `5.2k` + `NFCorpus` `3.6k` = **`5,023,716` real documents**), executing live zero-downtime heterogeneous search & `20%` (`1,004,743`-vector) rolling migration inside `Spark #1/#2` `128 GB` RAM!
  2. Executes a real **Local `72B` Frontier Reader Scale Sweep (`unsloth/Qwen2.5-72B-Instruct-bnb-4bit` `72.7B` parameters on `Spark #2` `GB10` unified memory)** vs **`Bare Nemotron-30B`**, **`Bare Qwen2.5-7B`**, **`Small Qwen2.5-3B + Chelated`**, and **`Mid Qwen2.5-7B + Chelated`** on regulated enterprise drift (`N=60`, blind cross-family judged), proving empirically that **compact `3B/7B + Chelated` beats bare `72B` frontier-scale models** while using **`10×–24×` fewer active parameters** and a fraction of inference latency/VRAM!
- **Factory Implementation Owner:** **Orchestrator + `Sol 6.2 High` / `Neon 4 High` modular helpers** (`scripts/run_wave11_d10_5m_real_cutover_and_72b_scale.py`, `tests/test_wave11_d10_scale_proof.py`).
- **Output Bundle:** `artifacts/wave11_d10_5m_real_cutover_and_72b_scale/` (`wave11_d10_scale_manifest.json` + `report_receipt.json`).
- **4-GPU Distributed Encoding & Scale Architecture:**
  - Shard the `4.40M` `BeIR/msmarco` passages across our 4 GPUs (`RTX 3090` + `Spark #1` + `Spark #2` + `Spark #3`) to encode real `384-d` (`all-MiniLM-L6-v2` fp16) embeddings in parallel (`~4–6 min` wall time!) + encode `1,004,743` (`20%` backfill shard including multi-domain BEIR evaluation slices + MSMARCO shard) in `1024-d` (`BAAI/bge-large-en-v1.5` / `BAAI/bge-m3`).
  - Build live `FAISS` HNSW / IVF-HNSW (`IndexHNSWFlat` / `IndexIVFPQ+HNSW` or partitioned HNSW shards in `Spark #1` `52 GiB` / `Spark #2` `75 GiB` RAM) + **`38.33 MiB` (`8 B/doc` × `5,023,716`) Chelated `ZeroMu + CSLS` sidecar** (vs **`20.09 GiB`** full `1024-d` float32 rewrite or **`80.38 GiB`** `4096-d` rewrite — **`524×` to `2,097×` smaller**!).
  - Measure `nDCG@10` & `Recall@10` across all 7 real BEIR query sets (`MSMARCO dev`, `SciFact`, `NFCorpus`, `FiQA2018`, `ArguAna`, `SciDocs`, `Quora`) comparing `Uncalibrated_DualShard`, `DriftAdapter_QueryAffine`, `DEV_ZScore`, `RRF_k60`, and `Chelated_ZeroMu_CSLS_8B`, plus `0.00%` stale resurrection under `EK5 v3` admission fencing at `N = 5,023,716` real documents!

---

## 3. Panel Review Questions for Wave-10 Advisors (`Carbon 4 Medium`, `Fable 5.2 Medium`) & Factory Specialists (`L1–L5`)

Follow `PANEL_OPERATING_INSTRUCTIONS.md`: report only evidence-backed defects, hidden failure modes, statistical traps, and concrete engineering corrections across:
- **`L1` (`Demo D6` Multi-Backend `OpenSearch 3.0` + `pgvector` + `Qdrant`):** Exact OpenSearch 3.0 k-NN mapping/scoring subtleties (`lucene` vs `faiss` inner-product offset `1 + dot` or `1 / (1 - dot)` in OpenSearch k-NN!), `pgvector` `<#>` negative inner-product operator sign convention, and ensuring `Recall@10 >= 0.995` at `50%` retraction density without silent metric mismatches.
- **`L2` (`Demo D7` Live `Graphiti / Zep` + `Neo4j 5.26`):** Known `graphiti-core 0.31.2` + local OpenAI-compatible vLLM (`Nemotron-30B`) pitfalls (structured output `response_format` / tool-calling / embedding dim hardcodings in `graphiti_core`, rate/concurrency locks on Neo4j Bolt, fair steelmanning so Graphiti does not fail on trivial JSON/schema plumbing).
- **`L3` (`Demo D8` Compliance Copilot + Non-NDA Black-Box Mode):** Architectural traps in GDPR `D_V` lineage cascade vs `CITES_AUDIT`, offline/online determinism, cryptographic receipt canonicalization, and guaranteeing `0/55` IP-leak patterns when `?mode=non_nda` is active.
- **`L4` (`Demo D9` `0.5B/1.5B` LoRA Extractor + `160`-Step `U`-Abstention GRPO):** Ensuring strict zero-leak split disjointness on both the synthetic extractor training set (`vs Suite A N=60`) and the curriculum GRPO environment (`vs Wave-6 EVAL N=120`), reward-shaping stability so `GRPO` does not collapse into `100%` trivial `ABSTAIN`, and exact paired statistical gates (`GRPO > SFT` on `EVAL`).
- **`L5` (`Demo D10` `5M+` Real-Corpus Cutover + `72B` Scale Sweep):** Exact RAM math on `128 GB` unified-memory Sparks (`5.02M` HNSW graph links `M=16/32` memory overhead + float16/float32 vector storage so we never OOM Spark #1/#2!), `qrels` ID alignment across `MSMARCO` + 6 BEIR corpora, and fair prompt/judge parity for `Qwen2.5-72B-Instruct-bnb-4bit` (`Qwen` family!) when `Qwen2.5-7B` is one of the judges (use **dual cross-family blind judging** with `Nemotron-30B` + `Qwen2.5-7B` or deterministic exact-value + `Nemotron-30B` blind judge so `Qwen-72B` vs `Qwen-7B` has zero same-family judge bias!).
