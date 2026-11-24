# Wave 10 Advisor & Architecture Panel Report (`Carbon 4 Medium`)

- **`panel_type`:** `architecture-planning`
- **`artifact`:** `docs/research/council-logs/wave10-r0-tier1-tier2-master-spec.md`
- **`run_date`:** `2026-11-23`
- **`run_id`:** `wave10-advisor-carbon4-med`
- **`challenge_depth`:** `1`
- **`active_experts`:** `["gemini-4-carbon (Medium — Independent Advisor; Systems, DB, RL/Stats & Governance)"]`
- **`persist_version`:** `1.1.0`

## Scope & Verdict

**Verdict: `APPROVED WITH MANDATORY GUARDRAILS`.** Reviewed `wave10-r0-tier1-tier2-master-spec.md` against `chelated_grpc_gateway.py`, `ek5_v3_stratified.py`, `chelated_epistemic_proposer_v2.py`, `prospective_realworld_drift_corpus.py`, `wave6_common.py`, `demo_workbench/server.py`, `scripts/run_wave6_pm3_v2_4way_arena.py`, `scripts/run_wave6_d4_trained_epistemic_grpo.py`, and `docs/research/external-investor-teaser-non-nda.md`. Five caller-verified defects (`L1–L5`) will cause DDL crashes, RL collapse, judge exceptions, or Spark OOMs unless gated pre-flight.

## Actionable Findings (`L1–L5`, Ordered by Consequence)

1. **`[L1 — Blocker]` `pgvector` `4096-d` HNSW Ceiling, OpenSearch Nonlinear `_score`, & `50%` Retraction Recall (`master-spec.md:31-37`, `chelated_grpc_gateway.py:290-308,368-383`)**
   - **Evidence & Impact:** Spec line 32's `USING hnsw (embedding vector_ip_ops)` on `4096-d` crashes DDL: 8 KiB Postgres pages cap `pgvector` HNSW at **`2,000` dims (`vector`)** and **`4,000` dims (`halfvec`)**. OpenSearch 3.0 `space_type="innerproduct"` (`faiss`) maps raw dot `s` to `1 + s` (`s >= 0`) or `1 / (1 - s)` (`s < 0`) (`lucene`: `(1 + s)/2`), corrupting `ZeroMu`, `CSLS`, and `compute_unseen_score_upper_bound`. At `p_admit=0.50`, `exact_fallback_threshold=0.10` and `ef_max=256` (`gateway.py:297,1237`) starve shortlists (`0.9942 < 0.995`).
   - **Correction:** Store `halfvec(4096)` and build expression HNSW `((subvector(embedding,1,4000)::halfvec(4000)) halfvec_ip_ops)` (or `bit(4096)` BQ HNSW) with a SQL CTE re-ranking candidates via `-1.0 * (embedding <#> $1::halfvec) AS raw_dot` (`ORDER BY embedding <#> $1 ASC`, `0 B/doc` egress). In OpenSearch, pin `engine="faiss"`, invert `_score` piecewise (`score - 1` if `score >= 1` else `1 - 1/score`), and pair native pre-filters (`knn.filter`, `hnsw.iterative_scan=relaxed_order`) with `k_os_eff = min(1024, ceil(8 * k_os / p_admit))`, `ef_max >= 512`.
   - **Verification:** Assert `max_abs_err(recovered_dot, np.dot(q,d)) < 1e-5` on signed canary dots across all 3 engines, `Recall@10 >= 0.995` at `50%` retraction, and `0.00%` stale resurrection.

2. **`[L4 — Blocker]` GRPO Unconditional-`ABSTAIN` Collapse, `SFT` Topology Gap, & Split Hygiene (`master-spec.md:78-82`, `run_wave6_d4_trained_epistemic_grpo.py:486-513,1058-1210`)**
   - **Evidence & Impact:** On `1/3 T, 1/3 F, 1/3 U` batches, Spec line 81's `-1.25` (`U` false-commit) vs `-0.20` (`T/F` false-abstain) rewards give unconditional `VERDICT: U / ABSTAIN` positive expected return `E[R] = +0.217`, beating `65%`-accurate determinate policies (`-0.195`) and collapsing 160-step GRPO into `100%` `ABSTAIN`. `SFT_WARMUP` (`d4.py:487-492`) omits odd cycles and contested successors (`0.875` `EVAL` `U` false-commit). Part A (`0.5B/1.5B` LoRA) risks memorizing `prospective_realworld_drift_corpus.py:44-55` echo templates.
   - **Correction:** Set `FALSE_ABSTAIN_REWARD = -0.55` (`E[R_always_abstain] = -0.017 < 0`) and `FALSE_COMMIT_ON_U = -1.15`; add abstract `5-cycle` (`k != 3`) and `2-hop` contested-branch motifs to `GRPO_RL` while keeping `EVAL` (`N=120`) and `D1` (`N=60`) byte-frozen. Gate Part A's `N=600` set for `0` entity/slot/value overlap and 5-gram Jaccard `< 0.25` vs `_ECHO_VARIATION_TEMPLATES`.
   - **Verification:** Assert `abstain_rate <= 0.50`, `false_abstain_rate <= 0.12`, `EVAL` `GRPO > SFT` (`lcb_975 > 0`), and Part A `< 1,500 ms/case` (`>= 99%` valid JSON).

3. **`[L5 — High]` Same-Family Judge Crash, `Spark` RAM/VRAM OOM, & BEIR ID Collisions (`master-spec.md:86-93,104`, `wave6_common.py:254-257`, `chelated_grpc_gateway.py:414-436`)**
   - **Evidence & Impact:** `BlindCrossFamilySemanticJudge` (`wave6_common.py:254`) raises `ValueError` when `Qwen2.5-7B` judges `Qwen2.5-72B` or `Qwen2.5-3B/7B + Chelated`. `build_shard_sidecar` (`gateway.py:416,425`) casts full matrices to `float64` (`+19.2 GiB` at `5.02M` docs), and co-residing `FAISS` (`~10.8 GiB` `384d+1024d` HNSW, `M=32`) with `70B/72B-4bit` (`~44 GiB`) OOMs `Spark #2` (`75 GiB` free alongside untouched `:8890`). `MSMARCO`/`FiQA`/`Quora`/`SciFact` collide on bare integer IDs.
   - **Correction:** Prefix all 7 corpora, queries, and `qrels` (`msmarco:`, `scifact:`, etc.). Stream sidecar fits in `65,536`-row `np.memmap` chunks (`38.33 MiB` sidecar), then `del` FAISS + `malloc_trim(0)` before loading `Llama-3.3-70B-bnb-4bit` (preferred cross-family reader; grade `Qwen`/`Llama` arms via `Nemotron-30B` blind judge + `score_d1_prose_prediction`, and `Bare Nemotron-30B` via `Qwen2.5-7B`).
   - **Verification:** Assert `len(unique_doc_ids) == 5_023_716`, `peak_rss_gib <= 44` (`Spark #1`) / `<= 64` (`Spark #2`), and `0` judge family violations.

4. **`[L2 — High]` `graphiti-core 0.31.2` vLLM Schema, Dim, & Bolt Deadlock Traps (`master-spec.md:44-53`, `run_wave6_pm3_v2_4way_arena.py:188-195`)**
   - **Evidence & Impact:** `graphiti_core`'s `OpenAIClient` (`beta.chat.completions.parse`) fails on vLLM, concurrent `add_episode` calls deadlock Neo4j 5.26 Bolt locks, and `arena.py:188` conflates audit-context substrings with reader leakage.
   - **Correction & Verification:** Wire `OpenAIGenericClient` (`enable_thinking=False`) + `1024-d` `bge-m3` `EmbedderClient`, serialize `add_episode` per `group_id=case_id` (`SEMAPHORE_LIMIT=4`), query `Delta=0s` at `t_retract + 0ms` before `add_episode(retract)` settles vs post-await, and assert `<2%` schema errors, `0` deadlocks, separated context/answer leak metrics, and `Clopper-Pearson`/`Tango` CIs.

5. **`[L3 — Medium]` Invalid `DERIVES_VALUE` Edge, Query-String Drop, & Receipt Scrubbing (`master-spec.md:62,68-69`, `ek5_v3_stratified.py:39-55`, `demo_workbench/server.py:150,589`)**
   - **Evidence & Impact:** `"DERIVES_VALUE"` fails `V3_VALID_EDGE_TYPES`; `server.py:589` drops `?mode=non_nda`; deleting GDPR nodes breaks `CITES_AUDIT` endpoints; latency fields break deterministic receipt hashes.
   - **Correction & Verification:** Use `DERIVES_CURRENT`/`COPY_OF` for `D_V` cascade while tombstoning `text="[REDACTED_GDPR_ART17]"` so `CITES_AUDIT` survives; parse `parse_qs`, exclude `latency_ms` from signed digests, and assert `0/55` Non-NDA regex hits across raw HTTP bytes and exported receipts (`wfs_oracle_mismatches == 0`).

## Unknowns & Independent Acceptance

Live container/GPU behavior is unverified in this read-only review (`inference` from static callers; separate feasibility panel skipped). Run DDL/score-inversion (`D6/D7`) and 5-step GRPO (`D9`) pre-flight gates first, then Wave-12 cross-model adjudication (`gemini-4-carbon` + `claude-fable-5-2`) over sealed `Part VII` receipts.
