# Wave 10 Panel-of-Experts Synthesis & Binding Guardrails (`L1–L5` for Wave 11 Execution)

- **`panel_type`:** `architecture-planning`
- **`artifact`:** `docs/research/council-logs/wave10-r1-synthesis-guardrails.md`
- **`run_date`:** `2026-11-23`
- **`run_id`:** `wave10-r1-synthesis-d6-d10`
- **`challenge_depth`:** `1` (`6/6` Independent Multi-Model Reviews Synthesized)
- **`active_experts`:**
  - `gemini-4-carbon` (`Medium`, Advisor/Judge — `wave10-advisor-carbon4-med.md`)
  - `claude-fable-5-2` (`Medium`, Co-Advisor/Skeptic Judge — `wave10-advisor-fable52-med.md`)
  - `gpt-6.2-sol` (`High`, Factory Systems — `wave10-factory-sol62-high.md`)
  - `grok-4.7` (`XHigh`, Factory Graphiti/Neo4j — `wave10-factory-grok47-xhigh.md`)
  - `claude-sonnet-5-5` (`High`, Factory Compliance Copilot — `wave10-factory-sonnet55-high.md`)
  - `gemini-4-neon` (`High`, Factory Extractor/GRPO/70B — `wave10-factory-neon4-high.md`)
- **`persist_version`:** `1.1.0`

---

## Overall Verdict: `APPROVED WITH MANDATORY PRE-FLIGHT & EXECUTION GUARDRAILS (L1–L5)`

All 6 independent model reviews converged with zero unresolved architectural deadlocks and surfaced **6 high-consequence pre-implementation catches** that are now binding across Wave 11 implementation and Wave 12 final adjudication:

---

## `L1` — `Demo D6` (`chelated_multibackend_appliance.py` + `OpenSearch 3.9.0` + `pgvector:pg17` + `Qdrant`)
1. **Version & Container Provenance Reconciliation (`Fable 5.2`, `Carbon 4`, `Sol 6.2`):**
   - Run **official `opensearchproject/opensearch:3.9.0`** (`sha256:3bf4fe26aeb9f7b03c0ebcb1302bb24e855ee6b89fd926d26a71a83ef6f7e076`), reconciling `D6` directly with the `OpenSearch 3.9` version cited in the investor materials, alongside `pgvector/pgvector:pg17` and `qdrant/qdrant:latest`.
   - Persist exact `RepoDigests`, live engine-reported server versions (`GET /`, `SELECT version()` + `pg_available_extensions`, Qdrant telemetry), exact wire transport per arm, and post-ingest engine counts (`_count`, `SELECT count(*)`, `points_count`) matching expected shard sizes.
2. **Exact `raw_dot` Recovery (`0 B/doc` Vector Egress):**
   - **OpenSearch `3.9.0` (`engine: "faiss"`, `space_type: "innerproduct"`, `_source: {"excludes": ["embedding"]}`):** Invert `_score` in `float64` piecewise: `raw_dot = score - 1.0` if `score >= 1.0` else `1.0 - (1.0 / score)`.
   - **PostgreSQL `pgvector:pg17`:** Project `-(embedding <#> $1) AS raw_dot` (`ORDER BY embedding <#> $1 ASC`) with zero vector column in `SELECT`.
   - **Mandatory Canary Gate:** On `>= 1,000` returned hits per engine spanning positive, zero, negative, and near-tie cosines, assert `max_abs_err(recovered_dot, exact_f64_dot) < 1e-5` (fail-closed if violated).
3. **`pgvector` `4096-d` HNSW Ceiling Resolution (`2,000` `vector` / `4,000` `halfvec`):**
   - On the `80%` `384-d` legacy shard, build native `USING hnsw (embedding vector_ip_ops)`.
   - On the `20%` `4096-d` native shard, `pgvector` caps `vector` HNSW at `2,000` dims and `halfvec` HNSW at `4,000` dims on 8 KiB pages. Build **both** (a) genuine expression HNSW `USING hnsw ((subvector(embedding, 1, 4000)::halfvec(4000)) halfvec_ip_ops)` with a single-query SQL CTE re-scoring candidates via exact `4096-d` `-(embedding <#> $1::halfvec(4096)) AS raw_dot` (`0 B/doc` egress) **and** (b) exact admitted `vector(4096)` scan, recording `EXPLAIN` plans for every route and never mislabeling sequential scan as HNSW. Also test full-HNSW `384-d <-> 1024-d` (`vector(1024)` native HNSW on both shards) for apples-to-apples pure-HNSW parity!
4. **Closing `D5`'s `50%` Retraction `Recall@10` (`>= 0.995` Frozen on `DEV`):**
   - Freeze shortlist/ef scaling rules on `DEV` queries (zero test-query tuning loops): pass native admission pre-filters (`OpenSearch` `knn.filter`, `pgvector 0.8+` `SET LOCAL hnsw.iterative_scan = relaxed_order`, `Qdrant` `Filter`) paired with `k_os_eff = min(1024, ceil(4 * k_os / max(p_admit, 0.05)))`, `ef_max = 512`, and verified-exhaustion exact fallback at `p_admit <= 0.15`. Gate `Recall@10 >= 0.995` across `0%, 10%, 50%, 90%` retraction densities and `0` stale leaks from raw engine responses.

---

## `L2` — `Demo D7` (Live Upstream `graphiti-core 0.31.2` + `Neo4j 5.26` vs `Chelated v4` & Live `Mem0 2.2.1`)
1. **Verified `graphiti-core 0.31.2` + Local vLLM (`:8891`) Steelmanning (`Grok 4.7`, `Carbon 4`, `Fable 5.2`):**
   - Use `OpenAIGenericClient` (not `OpenAIClient`, which calls `responses.parse`) with `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`, `temperature=0.0`, automatic `json_schema` $\to$ `json_object` fallback (publishing exact schema-repair/fallback counts), and wrapped `AsyncOpenAI` usage accounting (`prompt_tokens` + `completion_tokens` tracked separately for write vs read; missing `usage` fails closed).
   - Implement `EmbedderClient` (`create` + `create_batch` over GPU `BAAI/bge-m3` `1024-d`, asserting `len == 1024`) and GPU `CrossEncoderClient` (`BAAI/bge-reranker-v2-m3`), call `await driver.build_indices_and_constraints()`, and cap `max_coroutines=6`.
2. **Single-DB Neo4j Community `group_id` Isolation & Non-Empty Graph Gate:**
   - Keep Neo4j Community database pinned to `"neo4j"` (shimming `_resolve_request_scope` so `group_id = f"d7_case_{i:02d}"` is stored/filtered on node & `RELATES_TO` properties via `group_ids=[gid]` rather than attempting Enterprise `CREATE DATABASE`).
   - After every settled case write, execute Cypher `MATCH (n {group_id: $gid}) RETURN count(n)` and `MATCH ()-[r:RELATES_TO {group_id: $gid}]->() RETURN count(r)`: require both `> 0` (any `0`-edge case is tagged `ARM_FAILURE`, never silently swallowed via broad `except`).
3. **Fair `Delta=0s` vs `Settled` Evaluation & Statistical Reporting:**
   - Use `EDGE_HYBRID_SEARCH_CROSS_ENCODER` with `invalid_at IS NULL` so Graphiti's reader receives current active edges (not raw historical episode blobs).
   - Measure `Delta=0s` (search issued immediately while retraction `add_episode` task is in flight) separately from `Settled` (post-`await`), reporting context vs answer stale leakage separately, `paired_bootstrap_wave6`, Tango, Clopper–Pearson 95%, and Holm–Bonferroni on `Suite A (N=60)`. Keep `Suite B (N=20)` unpooled and descriptive only.

---

## `L3` — `Demo D8` (Regulated Enterprise Compliance Copilot + Non-NDA Black-Box Mode)
1. **Exact `EK5 v3` Semantics & GDPR Tombstone Hygiene (`Sonnet 5.5`, `Fable 5.2`, `Carbon 4`):**
   - Zero calls to `SYNTHETIC_NUMPY_MODE_B_SIMULATOR`. All 3 arms (`Unprotected Hybrid RAG` with real delete-by-id, `Overwrite Policy (emulation, not live Mem0)`, and `Chelated Control Plane`) share one real corpus + `FastBM25Index` + dense embeddings + RRF `k=60`.
   - Use valid `V3_VALID_EDGE_TYPES` (`REFUTE` cascading across `DERIVES_CURRENT`, `DERIVES_SNAPSHOT`, `COPY_OF`, `AMENDS`; `SUPERSEDE` cascading across `DERIVES_CURRENT` & `COPY_OF`; `CITES_AUDIT` outside `D_V` paired with an explicit Copilot-layer `clinician` vs `safety_investigator` role filter).
   - Enforce slot-scoped `resolve_slot` for `CONTESTED_ABSTAIN`, physical GDPR PII tombstoning (`[REDACTED_GDPR_ART17:sha256:...]`) + `erased_content_sha256` re-ingestion blocklist, compensating-event undo (preserving `as_of` history), and warm-store `work_stats["wall_ms"]` `p50/p95`.
2. **Server-Side Allowlist Non-NDA Mode & Ed25519 Chained Receipts:**
   - Default to `non_nda` mode; build responses from a server-side typed public projection (never shipping forbidden terms as keys in client HTML/JS!) backed by `scripts/verify_non_nda_ip_clean.py` (`56` patterns, tested across HTML/JS source, every scenario × step × role JSON response, error payloads, and exported receipts, plus a seeded-leak positive control).
   - Export canonical JSON **Retraction Audit Receipts** (integer microseconds only, excluding wall-clock latency from hashed payload, `SHA-256` hash-chained ledger) signed with real **Ed25519** (`cryptography` library) + `/api/v2/copilot/verify-receipt` public-key verification endpoint.

---

## `L4` — `Demo D9` (Fast `0.5B/1.5B` LoRA Extractor + Virgin-Holdout Curriculum `Epistemic-WFS-GRPO`)
1. **Part A — Disjoint Synthetic Corpus & Batched Merged LoRA Extractor (`Neon 4`, `Fable 5.2`, `Carbon 4`):**
   - Generate `N=600` (`150`/stance) across 4 non-Suite-A domains (`pharma_clinical`, `supply_chain`, `telecom_bgp`, `energy_scada`) with two-value counter-filing & rumor-debunk contrasts. Enforce fail-closed `0` entity/slot/value overlap AND `< 0.25` 5-gram Jaccard vs all `1,020` `Suite A` docs (`_ECHO_VARIATION_TEMPLATES` included).
   - Benchmark merged `bf16` batched (`batch_size=17`, `max_new_tokens=48`) inference on RTX 3090 (`0.5B` and `1.5B`, promoting on synthetic `DEV` before touching `Suite A`), reporting `< 1,500 ms/case`, `>= 99%` JSON validity, and paired bootstrap/Tango/CP95 CIs vs both live `Mem0 (81.67%)` and the `30B` extractor (`96.67%`).
2. **Part B — Pre-Sealed Virgin `EVAL_VIRGIN_V2 (N=120)` + Anti-Collapse `160`-Step `GRPO`:**
   - **Primary Holdout Fix (`Fable 5.2` Defect #2):** Before any training runs, generate and SHA-256-seal a brand-new **`EVAL_VIRGIN_V2` (`N=120`)** on a fresh disjoint slot/entity/value/seed subspace (`SLOT_POOL_V2`) as **PRIMARY**, while also evaluating byte-identical Wave-6 `EVAL` (`N=120`, labeled `Look #2 / development-diagnosed`) and zero-shot `D1` (`N=60`).
   - Fix Wave-6's 16-prompt bottleneck and abstention-collapse trap: untruncated 4-step WFS SFT trace, `160` steps × `grad_accum=2` (`320` stratified prompt-groups across seeds `42, 137, 2026`; fixed final step 3-seed mean PRIMARY), `FALSE_ABSTAIN_REWARD = -0.55` (`E[R_always_abstain] = -0.017 < 0`!), `FALSE_COMMIT_ON_U = -1.15`, explicit `Always_Abstain` baseline row, and `T/F` determinate non-inferiority gate (`>= SFT - 0.03`).

---

## `L5` — `Demo D10` (Full Unsubsampled `8.84M` / `9.47M` Real-Document Cutover + `70B` Scale Sweep)
1. **Zero-Subsampling Official `BeIR/msmarco` (`8,841,823`) + 6 BEIR (`623,716`) = `9,465,539` Real Passages (`Fable 5.2`, `Sol 6.2`, `Carbon 4`):**
   - Encode **100% of `BeIR/msmarco` (`8,841,823` passages — zero subsampling!)** plus all 6 cached BEIR corpora (`623,716` passages) across our 4 GPUs (`RTX 3090` + `Sparks #1–#3`), prefixing every ID `{dataset}:{orig_id}` (`assert len(unique_ids) == 9_465_539`, excluding ArguAna self-matches).
   - Report `MSMARCO dev` on the exact unsubsampled `8,841,823` corpus AND all 7 query sets against the merged `9,465,539`-document (`80%` `384-d` + `20%` `1024-d`) FAISS index (recording exact FAISS factory string, `M`, `efSearch`, sampled Flat-oracle recall, and peak RSS `<= 44 GiB`).
   - Use strict binary units everywhere: **`8 B/doc` sidecar = `72.22 MiB` (`75,724,312 B`)** vs `1024-d` float32 **`36.11 GiB` (`512×` exact)** and `4096-d` float32 **`144.43 GiB` (`2,048×` exact)**, streaming sidecar construction in `65,536`-row `np.memmap` chunks and calling `del` + `malloc_trim(0)` before `70B` load.
2. **Single-Locked `unsloth/Llama-3.3-70B-Instruct-bnb-4bit` (`4-bit NF4` Explicitly Disclosed) & Cross-Family Judging (`Neon 4`, `Carbon 4`, `Fable 5.2`):**
   - Lock `unsloth/Llama-3.3-70B-Instruct-bnb-4bit` (`4-bit NF4` disclosed in every table/headline) on `Spark #2` (`GB10`), evaluated bare at both `k=5` and `k=17` (`100%` recall control) alongside `Bare Nemotron-30B`, `Bare Qwen2.5-7B`, `Small Qwen2.5-3B + Chelated (k=5)`, and `Mid Qwen2.5-7B + Chelated (k=5)` on `Suite A (N=60)`.
   - Report deterministic `strict_exact_match` across all arms, plus zero-violation cross-family blind semantic judging (`Nemotron-30B` judging `Llama-70B` & `Qwen-3B/7B`; `Qwen2.5-7B` judging `Llama-70B` & `Nemotron-30B`).
