# L2 Demo D7 — Competitor Arena Engineering Design & Risk Memo

- **`panel_type`:** `architecture-planning`
- **`artifact`:** `docs/research/council-logs/wave10-factory-grok47-xhigh.md`
- **`run_date`:** `2026-11-23`
- **`run_id`:** `wave10-factory-grok47-xhigh-l2`
- **`challenge_depth`:** `1`
- **`active_experts`:** `["grok-4.7 (Factory, L2 Demo D7)"]`
- **`persist_version`:** `1.1.0`

**Scope.** Live `graphiti-core==0.31.2` on `neo4j:5.26-community` (`bolt://127.0.0.1:17687`) versus `Chelated v4` and live `mem0ai==2.2.1` `infer=True`. Tag sources: `openai_client.py`, `openai_generic_client.py`, `graphiti.py`, `embedder/client.py`, `search_ops.py`, `search_filters.py`, `bge_reranker_client.py`, plus `scripts/run_wave6_pm3_v2_4way_arena.py` and `wave6_common.py`. No Neo4j, vLLM, or install was run.

## 1. vLLM client, 1024-d embedder, GPU reranker

`OpenAIClient._create_structured_completion` calls `client.responses.parse` with `text_format=<Pydantic model>`. Unstructured calls use `chat.completions.create` and `response_format={"type":"json_object"}`. This tag has no `chat.completions.parse`. `:8891` is a chat-completions server, so `Graphiti()`'s default `OpenAIClient` is the wrong client. Pass `llm_client=OpenAIGenericClient`.

`OpenAIGenericClient` posts `chat.completions.create`. Default `structured_output_mode='json_schema'` sends `response_format={type:json_schema, json_schema:{name, schema: model_json_schema()}}` and omits `strict: true`. `'json_object'` sends `{type:json_object}` and appends the schema to the last message. Probe `json_schema` once. On HTTP 400 or a non-object body, switch to `json_object`, strip `<think>` and fences, and log the mode. The stock client strips fences only and sends no `extra_body`. The AsyncOpenAI wrapper sets `extra_body.chat_template_kwargs.enable_thinking=false` on every `create`, matching `call_vllm_timed`. Temperature stays 0. Tenacity retries four times (5–120 s); log attempts. A retry timeout is an endpoint failure.

`OpenAIGenericClient` discards `response.usage`, so `Graphiti.token_tracker` stays 0. The wrapper adds `usage.prompt_tokens` and `usage.completion_tokens`. Reset around `add_episode` (write) and around the reader call (read). Absent `usage` fails the receipt.

`EmbedderClient.create` returns 1024 floats from GPU `BAAI/bge-m3`. Implement `create_batch`; the ABC raises `NotImplementedError`. `EMBEDDING_DIM` is read at import (default 1024). `OpenAIEmbedder` slices to that width, so a 1536-d endpoint silently truncates. Tag `0.31.2` does not `CREATE VECTOR INDEX`. `node_similarity_search` / `edge_similarity_search` call `vector.similarity.cosine` on `n.name_embedding` and `e.fact_embedding`. Unequal length throws. Assert length 1024 before write.

The default cross-encoder is `OpenAIRerankerClient` and would spend Nemotron tokens on rerank. Use a `CrossEncoderClient` with the same `rank(query, passages)` contract as `BGERerankerClient`, loading `BAAI/bge-reranker-v2-m3` on CUDA. The stock class sets no device.

`await driver.build_indices_and_constraints()` on a running loop. `Neo4jDriver.__init__` skips that task with no running loop, and fulltext then returns empty. Set `max_coroutines=8` on one Bolt server. The default semaphore is 20 per nested `semaphore_gather`.

## 2. `group_id` on one Community database

`_resolve_request_scope` clones `Neo4jDriver(database=group_id)` when `group_id != driver._database` (issue #1676). Community 5.26 has one user database. Sixty database names miss the indexes built on `neo4j` or fail outright. Logged shim: after `validate_group_id`, keep database `neo4j` and store `d7_case_00` … `d7_case_59` on the property (`[A-Za-z0-9_-]+`). `group_id=None` becomes `''` and merges every case.

Every `search` passes `group_ids=[gid]`. `None` drops `n.group_id IN $group_ids` and the Lucene `group_id:` clause, so cosine and BM25 scan all cases. After each awaited add, on database `neo4j`: `MATCH (n {group_id: $gid}) RETURN count(n)` must be `> 0`. Also count `RELATES_TO` with that `group_id`. An episode node alone is `episode_only` and fails the settled gate. Every hit from `search(..., group_ids=[gid])` must carry that `group_id`. Cases stay co-resident. The `Delta=0s` probe records the count and allows 0.

## 3. `Delta=0s`, tokens, intervals

`add_episode` returns only after extraction. Its docstring says to queue it and to await episodes in order. Await every pre-retraction episode. `create_task` only the retraction, search immediately (`Delta=0s`, log `task.done()`), then `await` that same task and search again (settled). `write_ack_ms` is schedule time; `consolidation_bg_ms` is task wall time. Chelated stays the synchronous EK5 fence.

Reader context is current facts. Default `SearchFilters.invalid_at` is unset, and `COMBINED_HYBRID_SEARCH_CROSS_ENCODER` also returns raw episode text (`store_raw_episode_content` defaults true), which still holds the retracted string. Use `EDGE_HYBRID_SEARCH_CROSS_ENCODER` plus `invalid_at IS NULL`. Run `check_stale_resurrection` on that context only. Settled accuracy is the primary score; `Delta=0s` is a separate horizon.

Suite A `N=60` only. `paired_bootstrap_wave6(seed=42, n_bootstrap=20000)` on case-id keys, scores in `{0,1}`, once per horizon: `lcb_975`, Clopper-Pearson 95%, Tango at `z=1.959964`, Look #2 `z=2.241403`. Holm-Bonferroni over Chelated−Graphiti and Chelated−Mem0 inside that horizon. Keep `U` separate (`n<40` uses weak monotonicity). Mem0 is live `2.2.1` `infer=True`, same bytes, same reader, `Qwen2.5-7B` judge, Nemotron generator. Suite B stays unpooled.

**Unknown.** Bolt behavior under the shim, vLLM `json_schema` acceptance, and `usage` presence are unverified until the arena runs.
