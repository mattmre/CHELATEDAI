# Panel of Experts — Performance & Scale Review
## ChelatedAI Repository

**Date:** 2026-04-04
**Panel:** Performance & Scale
**Scope:** Core benchmarks, computational storage PoC subsystem
**Files reviewed:**
- `benchmark_beir.py`, `benchmark_comparative.py`, `benchmark_distillation.py`, `benchmark_multitask.py`, `benchmark_utils.py`
- `run_sweep.py`, `run_large_sweep.py`, `run_overnight_campaign.py`
- `computational_storage_poc/compiler.py`, `mock_nvme.py`, `cpu_backends.py`, `cpu_inference_benchmark.py`
- `computational_storage_poc/sparse_cpu_inference.py`, `packed_cpu_inference.py`, `integrated_repo_runtime.py`
- `computational_storage_poc/moe_reap.py`, `repo_graph_memory.py`, `disk_llm_estimator.py`, `phase7_system_evaluation.py`
- `computational_storage_poc/block_graph.py`, `packed_graph.py`, `storage_substrate_benchmark.py`
- `computational_storage_poc/sparse_inference_benchmark.py`, `integrated_runtime_benchmark.py`, `repo_graph_memory_benchmark.py`
- `computational_storage_poc/retrieval_eval_suite.py`
- `docs/phase7-promotion-review-2026-03-29.md`

---

## Panel Members

| Expert | Background | Primary Lens |
|---|---|---|
| Dr. Elena Vasquez | Senior Reliability Engineer (18 yrs) | Failure modes, latency tails, resource exhaustion |
| Marcus O'Brien | Performance Engineer (13 yrs) | CPU/memory profiling, algorithmic complexity, hot paths |
| Yuki Tanaka | Systems Engineer (11 yrs) | I/O patterns, disk access, memory mapping, CPU cache |
| Fatima Al-Zahra | Infrastructure Architect (14 yrs) | Scalability limits, capacity planning, contention |
| Derek Washington | SRE / SLO Expert (10 yrs) | Observability, alerting, incident readiness |
| Devil's Advocate | Contrarian | Challenge every consensus finding |

---

## CONVENE

**Panel Mandate:** Find every performance bottleneck, scalability limit, operational gap, and efficiency opportunity in the benchmarks and the computational storage PoC.

The codebase presents two distinct performance domains that must be analyzed separately:

1. **Core RAG benchmarks** — BEIR/MTEB evaluation pipelines, parameter sweeps, sedimentation training loops, and distillation modes. These exhibit serial execution, missing warm-up, cold-start model loading, and benchmark methodology weaknesses.

2. **Computational storage PoC** — disk-backed graph inference, sparse selective loading, MoE routing, repo graph memory, INT8 quantization, and an integrated repo-Q&A runtime. These exhibit custom I/O patterns, theoretical latency models vs. measured ones, and a small evaluation suite that limits statistical confidence.

The panel will provide AT LEAST 20 findings each, challenge each other, converge on a severity ranking, and deliver a prioritized remediation roadmap.

---

## SOLO REVIEWS

### Dr. Elena Vasquez — Senior Reliability Engineer

**Focus: Failure modes, latency tails, resource exhaustion, cascading failures**

**V-01 [CRITICAL] — run_large_sweep.py reads the full JSON result file on every iteration, then re-writes it entirely**
`run_large_sweep.py` lines 126-134: on each of 7,350 iterations, the code reads the entire JSON results file (`json.load(f)`), appends one entry, and writes back the whole file (`json.dump(current_results, f, indent=2)`). At iteration N, this reads and writes an O(N) JSON blob. By the end of a 7,350-run sweep this is O(N²) I/O — the final write touches the entire results list. For a run with realistic entries this could easily reach hundreds of MB of write amplification and several seconds per iteration by run 5,000. CSV writes are append-only (correct), but JSON writes are destructive rewrites.

**V-02 [CRITICAL] — run_sweep.py and run_large_sweep.py reuse a single engine instance across all sweep configurations without resetting Qdrant state**
The sweep resets the adapter weights but uses the same `base_engine` object and the same in-memory (or on-disk) Qdrant collection across all 81 (run_sweep) or 7,350 (run_large_sweep) configurations. The chelation log is cleared between runs, but Qdrant vector state, any in-memory caching, and internal engine attributes are NOT reset. Any side effect that persists in the engine between iterations makes the sweep results non-independent. This is a correctness-level bug that invalidates sweep results as comparative data.

**V-03 [CRITICAL] — isolated_adapter_state() creates UUID-named backup files and never deletes them on exception if shutil.move fails**
`benchmark_utils.py` lines 125-137: on cleanup, `shutil.move(str(backup_path), str(target_path))` is called but if this fails (e.g., disk full, permission error, lock contention on Windows), the exception is swallowed and the backup file is left permanently on disk. The `.benchmark-backup-*.pt` files already visible in `git status` confirm this has happened at least three times. These backup files accumulate indefinitely, consuming disk space.

**V-04 [HIGH] — IntegratedRepoRuntime starts tracemalloc on EVERY query call with no guard against concurrent calls**
`integrated_repo_runtime.py` line 140: `tracemalloc.start()` is called on each `answer_query` invocation. `tracemalloc` is a process-global singleton. If two queries execute concurrently (or if `tracemalloc` was already started by the caller), the second `start()` call resets the trace frame, corrupting the peak memory measurement for the first query. `tracemalloc.stop()` at line 181 also stops any pre-existing trace that may have been started externally, altering caller state.

**V-05 [HIGH] — No timeout or circuit-breaker on MTEB data loading**
`benchmark_utils.py` / `benchmark_distillation.py`: `task.load_data()` and the subsequent dataset parsing have no timeout. For large datasets like NQ (2.6M docs) or HotpotQA (5.2M docs), this call can hang for minutes without producing any diagnostic output. If the network is unavailable, the call may block indefinitely with no cancellation mechanism.

**V-06 [HIGH] — run_overnight_campaign.py launches a subprocess but does not capture or forward stderr**
`run_overnight_campaign.py` line 132: `subprocess.run(command, cwd=PROJECT_ROOT)` with no `stdout` or `stderr` arguments passes all output directly to the terminal but does not capture it to a log file. If run detached (e.g., overnight via nohup), stderr from the child process is lost. Crash diagnostics are unavailable post-hoc.

**V-07 [HIGH] — MockNVMeDrive.__del__ calls close() which can raise if the object is partially constructed**
`mock_nvme.py` lines 47-48: `__del__` calls `self.close()`. If `__init__` raises after opening `self._file` but before mapping (e.g., if `mmap.mmap` fails), `close()` will be called by `__del__` on an object where `self.flash_memory` may not exist. The `getattr` guards protect against `AttributeError`, but `close()` could still raise in edge cases if the file descriptor is invalid. Finalizer exceptions are silently swallowed by the interpreter, making resource leaks invisible.

**V-08 [HIGH] — DiskBackedPackedGraph.close() is not idempotent under concurrent access**
`packed_graph.py` lines 173-182: `close()` checks `getattr(self, "_mapping", None)`, clears it, then closes. If two threads both pass the `None` check before either clears the field, both will attempt to close the same mmap, raising an exception on the second close. The pattern is not thread-safe.

**V-09 [HIGH] — benchmark_distillation.py forces use_centering=True after offline distillation with a comment explaining a known bug**
Lines 483-486: `engine_offline.use_centering = True` is a workaround for a known bug where "after offline distillation updates all corpus vectors, their variance drops below the chelation threshold." This is a state-management bug, not just a workaround. The offline mode's post-distillation behavior is not deterministic across different corpus sizes or model configurations, making the benchmark results fragile.

**V-10 [HIGH] — Phase 7 system evaluation runs all sub-benchmarks serially with no partial result persistence**
`phase7_system_evaluation.py` lines 31-37: `evaluate_system_promotion` calls 7 benchmark functions sequentially with no checkpointing. If any benchmark crashes midway, all prior results are lost and the full suite must restart from scratch. For the overnight campaign or CI, this means a single flaky benchmark invalidates hours of work.

**V-11 [MEDIUM] — compute_stability in benchmark_multitask.py runs the same query num_runs times and computes Jaccard — but the engine is deterministic**
`benchmark_multitask.py` lines 113-136: the stability metric runs each query `num_runs=3` times and measures pairwise Jaccard similarity. Since `AntigravityEngine.run_inference` is deterministic for a given input (no stochastic components at inference time once the adapter is fixed), all result sets will be identical and the Jaccard will always be 1.0. This is a misleading metric — it tests determinism, not stability under perturbation.

**V-12 [MEDIUM] — No watchdog or memory limit on the large parameter sweep**
`run_large_sweep.py` performs 7,350 iterations, each of which reinitializes the adapter and runs sedimentation. There is no memory high-water-mark monitor, no automatic termination if RAM exceeds a threshold, and no time-budget enforcement. A long-running sweep process can exhaust system resources without raising an alert.

**V-13 [MEDIUM] — isolated_adapter_state uses shutil.copy2 + unlink, which is not atomic on Windows**
`benchmark_utils.py` lines 128-131: the backup sequence is copy2 then unlink, which is non-atomic. A crash between copy and unlink leaves both the original and backup on disk. A crash after unlink but before the context body completes leaves only the backup. On Windows, file locking can cause the unlink to fail silently or raise, breaking the state machine.

**V-14 [MEDIUM] — global ChelationConfig mutation in run_sweep.py and run_large_sweep.py is not thread-safe**
Lines 102-109 (run_sweep.py): `ChelationConfig.NOISE_INJECTION_ENABLED = True/False` and similar mutations are applied and restored around each sweep iteration. This is safe in single-threaded code but would be catastrophically unsafe if sweep iterations were ever parallelized, since the config mutations would race across threads.

**V-15 [MEDIUM] — No assertion or check that adapter reset actually zeroed learning history**
After `engine.adapter = create_adapter(...)` in the sweeps, there is no verification that the new adapter is truly at identity state. If `create_adapter` has any side effect that reads from a persisted checkpoint path, or if the engine caches the old adapter's optimizer state internally, the reset may be silently incomplete.

**V-16 [MEDIUM] — run_training_cycle in benchmark_distillation.py does not limit query wrap-around**
`benchmark_distillation.py` lines 290-296: `start_idx = (cycle * queries_per_cycle) % len(query_ids)` then `end_idx = start_idx + queries_per_cycle`. If `end_idx > len(query_ids)`, `cycle_query_ids = query_ids[start_idx:end_idx]` silently truncates (Python list slicing). The cycle processes fewer queries than intended without raising a warning, and the training signal for those cycles is weaker than the experiment design specifies.

**V-17 [MEDIUM] — _sample_corpus in benchmark_beir.py builds a set from all corpus keys for every call**
`benchmark_beir.py` lines 310-333: `rng.choice(pool_ids, size=remaining_budget, replace=False)` with potentially millions of IDs (NQ: 2.6M, HotpotQA: 5.2M). `np.random.RandomState.choice` without replacement on 2.6M elements is O(N) in both memory and time. For the full tier this is called once per dataset per run, but the set construction `{str(key): key for key in corpus.keys()}` on 2.6M keys is also O(N) memory allocation. The in-memory representation of 2.6M string keys before sampling can exceed 1 GB of RAM.

**V-18 [MEDIUM] — No structured log of sweep interruptions or partial runs**
Both `run_sweep.py` and `run_large_sweep.py` save incrementally, but neither logs the run index at which an interruption occurred, nor whether the incomplete final entry is valid. Resuming a sweep after a crash requires manual inspection of the CSV/JSON to determine where to restart.

**V-19 [LOW] — map_predicted_ids in benchmark_distillation.py is a duplicate of the same function in benchmark_utils.py**
`benchmark_distillation.py` lines 189-210 contains an older copy of `map_predicted_ids` that differs slightly from the canonical version in `benchmark_utils.py`. Divergence between the two means metrics computed in distillation benchmarks may not be directly comparable to those computed via the utils module.

**V-20 [LOW] — find_keys and find_payload recursive search functions have no depth limit**
`benchmark_utils.py` lines 259-299: both recursive search functions visit arbitrarily deep nested dicts without a recursion limit. On pathological MTEB dataset structures with deeply nested metadata, these could hit Python's recursion limit and raise a silent `RecursionError` that turns into a failed corpus parse.

---

### Marcus O'Brien — Performance Engineer

**Focus: Algorithmic complexity, hot paths, caching, CPU/memory profiling**

**M-01 [CRITICAL] — run_large_sweep.py JSON read-modify-write is O(N²) total I/O**
Already identified by Vasquez (V-01) as a failure mode; from a performance angle: this pattern produces approximately 7,350² / 2 = 27 million byte-read operations worth of total JSON parsing work across a full sweep. At 100 bytes per result entry, the final read processes 735,000 bytes just to load the results before appending. Python's `json.load` is not lazy, so the entire result list is deserialized into memory on each iteration.

**M-02 [CRITICAL] — NumpyInt8DynamicBackend.matmul performs TWO full matrix scans before the matmul itself**
`cpu_backends.py` lines 49-59: `_max_abs_scale(activations_f32)` scans the full activation array, `_max_abs_scale(weights_f32)` scans the full weight array, then `np.rint(activations_f32 / scale)` is another full pass, and `np.rint(weights_f32 / scale)` is another full pass. That is 4 additional O(N) passes before the O(N*M*K) matmul. For the sparse path, this overhead is applied to each chunk separately, multiplying the constant factor by the number of active chunks. The weight scale for a stored INT8 block is known at load time — computing it dynamically at each inference call is wasted work.

**M-03 [CRITICAL] — repo_graph_memory.py query() computes set(node.tokens) freshly on every call for every candidate node**
`repo_graph_memory.py` lines 354-370: for each of the `rerank_depth` (default 12) top candidates per query, `node_token_set = set(node.tokens)` is computed. With potentially hundreds of queries and 12 candidates each, this is O(queries * rerank_depth * avg_tokens_per_node) set constructions. Since nodes are immutable, these token sets should be precomputed once at ingest time and stored. The same applies to `path_token_set = set(_tokenize_text(node.path))` which re-tokenizes and set-ifies the path on every query for every node.

**M-04 [CRITICAL] — ingest_repo_graph_memory uses np.vstack to build the embedding matrix, which copies all embeddings**
`repo_graph_memory.py` line 239: `embeddings = np.vstack([_embed_tokens(node.tokens, dim=embedding_dim) for node in all_nodes])`. `np.vstack` on a list of N arrays creates a temporary list, then concatenates by allocating a fresh (N * embedding_dim) array and copying all values. For a large repo with thousands of nodes, this results in 2x peak memory usage (the list of individual arrays + the final stacked array). A pre-allocated NumPy array filled in-place would halve peak allocation.

**M-05 [HIGH] — _embed_tokens uses a Python loop with SHA-256 hashing per token**
`repo_graph_memory.py` lines 88-95: `_embed_tokens` computes `hashlib.sha256(token.encode("utf-8")).digest()` for every token in a node. SHA-256 is a cryptographic hash; it is 20-100x slower than a polynomial rolling hash for this purpose. For a codebase with hundreds of symbols, each containing dozens of tokens, this adds up to millions of SHA-256 operations at index time. The `_stable_hash` function exists for collision resistance, but a non-cryptographic hash (FNV1a, xxHash, or a simple polynomial hash) would be 10-50x faster with equal distribution for this use case.

**M-06 [HIGH] — run_block_graph in block_graph.py allocates a fresh np.float32 array on every read_block call**
`block_graph.py` line 61: `matrix = np.frombuffer(...).reshape(...).astype(np.float32)`. The `.astype(np.float32)` always allocates a fresh array regardless of whether the conversion is necessary. For packed graphs operating in inner inference loops (100 repetitions × 5 trials = 500 calls in cpu_inference_benchmark.py), this creates 500 × matrix_size allocations that pressure the allocator. Using `copy=False` where the dtype already matches, or pre-allocating output buffers, would reduce allocation pressure.

**M-07 [HIGH] — benchmark_beir.py format_ascii_table scans self.results twice for every dataset**
`benchmark_beir.py` lines 684-686: `ds_results = [r for r in self.results if r.dataset_name == ds_name]` is called for each dataset in a loop. With D datasets and R total results, this is O(D * R) total iterations. For the full tier (D=6 datasets, R = 6 * 8 = 48 results), this is minor. At larger scales it degrades, but more importantly the same O(D*R) pattern is repeated in `aggregate_by_config`, `aggregate_by_dataset`, and `build_heatmap_data`. The results should be grouped once with `setdefault`.

**M-08 [HIGH] — moe_reap.py reap_prune_experts computes np.linalg.norm for every matrix in every expert twice — once for scoring and once implicitly through sorted**
`moe_reap.py` lines 158-164: `scores = [float(sum(np.linalg.norm(matrix) for matrix in expert)) for expert in expert_matrices]` computes Frobenius norm for every matrix in every expert. This is O(num_experts * layers_per_expert * params_per_layer). The same computation is used again in `build_moe_reap_artifact` at line 92. This score computation is also done at artifact build time, not cached. For a model with 64 experts of 4 layers each, this is 256 Frobenius norm computations.

**M-09 [HIGH] — evaluate_engine in benchmark_distillation.py calls map_predicted_ids which issues a Qdrant retrieve() call per query**
`benchmark_distillation.py` lines 201-209: for each evaluated query, `map_predicted_ids` issues `engine.qdrant.retrieve(engine.collection_name, ids=pred_ids)`. With `max_queries=100`, this is 100 separate Qdrant retrieve calls per evaluation. Batching these into a single call or pre-building an in-memory ID→original_id lookup dict after ingestion would eliminate the per-query roundtrip. The retrieve call returns up to 10 documents per query, meaning 1000 Qdrant point fetches per evaluation pass.

**M-10 [HIGH] — cpu_inference_benchmark.py uses median of trial times, which is robust but the trial itself runs 100 repetitions with a Python-level for loop**
`cpu_inference_benchmark.py` lines 55-57: the inner loop `for _ in range(repetitions): last_result = run_packed_graph_with_backend(...)` is a Python for loop that calls a Python function 100 times per trial. The function call overhead (Python frame creation, argument unpacking, bytecode execution) is significant relative to the actual computation on the tiny 256→128→64→10 model being benchmarked. The results measure Python dispatch overhead as much as arithmetic throughput.

**M-11 [HIGH] — DiskBackedRepoGraphMemory.query builds neighbor_tokens set by iterating adjacency and resolving each neighbor from node_by_id dict**
`repo_graph_memory.py` lines 364-369: for each of the 12 candidates, a graph traversal collects all neighbor tokens via `self.adjacency.get(node.node_id, [])` then `self.node_by_id.get(edge.dst)`. This is O(num_neighbors * tokens_per_neighbor) set unions. For a file node in a large codebase that imports many modules, `num_neighbors` can be large. This neighborhood expansion is not cached and runs fresh for every query.

**M-12 [MEDIUM] — SparseChunkCache uses an OrderedDict with Python-level LRU eviction**
`sparse_cpu_inference.py` lines 34-50: `SparseChunkCache` implements LRU with an `OrderedDict` and `move_to_end`. Python's OrderedDict move_to_end is O(1) amortized but has significant per-operation overhead due to linked list pointer manipulation in pure Python. For the tiny matrices in the PoC (RERANK_FEATURE_DIM=8), the cache overhead may exceed the benefit.

**M-13 [MEDIUM] — _build_manifest in packed_graph.py serializes the manifest as JSON and embeds it inline in the binary artifact**
`packed_graph.py` lines 71-110: JSON serialization is done with `json.dumps(manifest, ...)`. On each artifact open, `DiskBackedPackedGraph.__init__` must decode the mmap bytes to string, then parse the JSON. For a model with many layers, manifest parse time grows linearly. A compact binary header (struct-packed) would avoid string parsing on every open.

**M-14 [MEDIUM] — run_weight_refinement_campaign.py (called by run_overnight_campaign.py) is not visible in the reviewed files but is invoked via subprocess**
The subprocess call from `run_overnight_campaign.py` to `run_weight_refinement_campaign.py` bypasses any Python-level profiling or timing instrumentation. Any performance issue in that script cannot be diagnosed from these files.

**M-15 [MEDIUM] — _active_chunk_ranges in sparse_cpu_inference.py does a sorted() call on chunk starts**
`sparse_cpu_inference.py` line 59: `chunk_starts = sorted({int(index // chunk_rows) * chunk_rows for index in active_indices})`. For small active index counts (as expected in sparse inference), the sort is negligible. However, `np.flatnonzero` produces an already-sorted array, so the chunks could be derived in order without sorting.

**M-16 [MEDIUM] — benchmark_comparative.py evaluate_single_config calculates per-query metrics inline using Python lists**
Lines 326-340: `ndcg_scores.append(...)`, `map_scores.append(...)`, etc. build Python lists of floats and call `np.mean` at the end. With `max_queries=100`, each metric list is 100 floats. This is fine at this scale but the pattern uses Python appends where a pre-allocated NumPy array would be more cache-friendly.

**M-17 [MEDIUM] — _quantize_matrix_to_int8 in packed_graph.py computes max(abs(matrix)) via np.max(np.abs(matrix)) which creates a temporary array**
`packed_graph.py` line 63: `max_abs = float(np.max(np.abs(matrix)))`. `np.abs(matrix)` allocates a new array equal in size to the input. `np.max` traverses it. For large weight matrices, this is a 2x peak-memory temporary. `np.max(np.abs(matrix))` could instead be computed as `np.sqrt(np.max(matrix**2))` using broadcasting, or more efficiently with `np.linalg.norm(matrix.ravel(), ord=np.inf)` which computes the infinity norm without allocating a temporary absolute-value array.

**M-18 [MEDIUM] — build_moe_reap_artifact calls b"".join(expert_blob_parts) at line 128 — this copies all expert blobs into a single byte string**
`moe_reap.py` line 128: for a model with many experts, `b"".join(expert_blob_parts)` creates a contiguous copy of all blob data. For a 64-expert model, this is double the memory required — the parts list plus the joined output. Writing blobs to a file in chunks would avoid this peak allocation.

**M-19 [LOW] — format_ascii_table in benchmark_comparative.py calls len(header) for separator generation on every call**
Minor: `separator = "-" * len(header)` is computed at every call. This is O(1) and trivially fast, not worth addressing alone, but representative of a pattern where formatting utility work could be memoized at class level.

**M-20 [LOW] — NDCG computation in benchmark_distillation.py is a standalone re-implementation rather than reusing benchmark_utils.ndcg_at_k**
`benchmark_distillation.py` defines its own `ndcg_at_k` and `dcg_at_k` (lines 32-44). These are identical to the functions in `benchmark_utils.py`. The duplication means a fix to one copy does not propagate to the other.

---

### Yuki Tanaka — Systems Engineer

**Focus: I/O patterns, disk access, memory mapping, CPU cache, NUMA**

**Y-01 [CRITICAL] — block_graph.py legacy format allocates 512×512 BLOCK_SIZE matrices regardless of actual layer dimensions**
`block_graph.py` lines 29-32: `create_block` pads every matrix to `(BLOCK_SIZE, BLOCK_SIZE) = (512, 512)` with `BYTES_PER_PARAM=2`, producing `512*512*2 = 524,288 bytes = 512 KB` per block regardless of the actual matrix size. A 128×10 output layer (1,280 params = 2,560 bytes) is stored in 512 KB. The legacy format wastes 99.5% of storage. The packed format solves this, but the legacy format is still used in `profile_sample_inference` (mock_nvme.py line 97) and `benchmark_storage_substrate`.

**Y-02 [CRITICAL] — DiskBackedRepoGraphMemory loads the full nodes.jsonl file into Python objects on every __init__ call**
`repo_graph_memory.py` lines 285-286: `self.nodes = self._load_nodes(...)` reads every line of `nodes.jsonl` and constructs a Python `CodeNode` object for each. All node objects live in Python heap memory. For a large repo, a `nodes.jsonl` with 10,000 nodes, each with 100 tokens, would occupy hundreds of MB of Python heap. The embeddings are mmap'd (efficient), but the node metadata and token lists are fully materialized in RAM. No lazy loading or memory-mapped representation is used for nodes.

**Y-03 [CRITICAL] — mmap'd embeddings in DiskBackedRepoGraphMemory are accessed as numpy read-only but the full dot product forces a complete read from disk on first access**
`repo_graph_memory.py` line 289: `self.embeddings = np.load(..., mmap_mode="r")`. When `self.embeddings @ query_embedding` is executed on line 337, NumPy must traverse the entire embedding matrix. On first query after ingest, this requires reading the entire embeddings.npy file from disk (or OS page cache). For a large repo, this cold-start read stalls the first query. The mmap allows the OS to page individual pages on demand, but the full matrix scan defeats this advantage.

**Y-04 [HIGH] — int8 embedding query path in DiskBackedRepoGraphMemory uses a Python for loop with chunk-by-chunk INT16 casting**
`repo_graph_memory.py` lines 341-345: the INT8 query path iterates `for start in range(0, self.embeddings.shape[0], INT8_QUERY_CHUNK_ROWS)` where `INT8_QUERY_CHUNK_ROWS = 64`. Each iteration casts a 64-row slice to INT16, performs a matrix-vector product, casts back to float32. For 1,000 nodes this is ~16 iterations. The chunked approach was presumably chosen to avoid int16 overflow, but the explicit Python loop adds dispatch overhead. A vectorized implementation using int32 accumulation could process all nodes in one matmul.

**Y-05 [HIGH] — packed_graph.py reads matrix bytes via mmap slice assignment, which on Windows creates a bytes copy**
`packed_graph.py` lines 224, 242: `matrix_bytes = self._mapping[start:end]`. On Windows, mmap slicing returns a `bytes` object (a copy), not a memoryview. This means every `read_block` call copies the matrix data out of the mmap into a Python bytes object before passing it to `np.frombuffer`. On Linux, `bytes` is also the result but it's a zero-copy buffer. The Windows behavior doubles memory bandwidth for each block read. Using `np.frombuffer(self._mapping, dtype=..., count=..., offset=...)` directly would avoid the intermediate copy.

**Y-06 [HIGH] — edges.json is loaded as a single JSON array with json.load, which does not stream**
`repo_graph_memory.py` line 319: `json.load(handle)` reads the entire edges.json into memory. For a large codebase, the edges array (bidirectional: imports + imported_by + contains + contained_by) can be 4x the number of symbols. A 10,000-symbol repo might have 40,000 edges. At ~100 bytes per edge entry in pretty-printed JSON, that is 4 MB of JSON to parse. `indent=2, sort_keys=True` in the writer produces verbose output. A JSONL format or binary format would load faster and use less peak memory.

**Y-07 [HIGH] — ingest_repo_graph_memory writes edges.json with indent=2 (pretty-printed), creating unnecessarily large files**
`repo_graph_memory.py` line 265: `json.dump([asdict(edge) for edge in edges], handle, indent=2, sort_keys=True)`. Each edge has 3 fields (src, dst, edge_type). With indent=2, each edge occupies ~8 lines of JSON. For 40,000 edges this is 320,000 lines. Using `separators=(",",":")` (compact) would reduce file size by 3-4x and parse time proportionally.

**Y-08 [HIGH] — mock_nvme.py host_read_block ignores INTERNAL_FLASH_LATENCY in the return value but includes it in the call signature**
`mock_nvme.py` line 52: `latency_cost = PCIE_ROUND_TRIP_LATENCY + INTERNAL_FLASH_LATENCY + transfer_time`. However, `traditional_host_inference` at line 88 uses `drive.host_read_block(0, TOTAL_BLOCK_BYTES)[1]` to get the per-block latency, and multiplies by `blocks_processed`. This double-counts latency: `INTERNAL_FLASH_LATENCY` is included in each `host_read_block` call, but the block graph is executed entirely in memory (the `run_block_graph` call uses `drive.flash_memory` which is already mmap'd). The flash latency in host mode should be incurred once for the initial mmap, not per-block.

**Y-09 [HIGH] — SparseChunkCache is instantiated fresh per query in IntegratedRepoRuntime.answer_query**
`integrated_repo_runtime.py` line 149: `cache = SparseChunkCache(max_cached_chunks=8)` is created for each call to `answer_query`. This means the cache is discarded after every query and provides no benefit across queries. If the same reranker block is accessed for multiple candidates (which is guaranteed — all candidates go through the same reranker graph), the cache is useful within a single query. However, the cache could persist across queries to benefit repeated queries that exercise the same sparse paths.

**Y-10 [HIGH] — np.save(target_root / "embeddings.npy", stored_embeddings) in ingest_repo_graph_memory writes a fortran-order or C-order array without specifying, relying on numpy default**
`repo_graph_memory.py` line 267: the embeddings are saved with default behavior. The subsequent `np.load(..., mmap_mode="r")` will mmap the file, but NumPy's mmap for .npy files aligns the data after the header. The dot product `self.embeddings @ query_embedding` accesses memory in row-major order. If the file was written with a different axis order than expected, the access pattern will be non-sequential. This is unlikely to be wrong but is not explicitly enforced.

**Y-11 [MEDIUM] — mmap.ACCESS_READ on Windows requires the file to remain open while the mmap is live**
All `DiskBackedPackedGraph`, `MockNVMeDrive`, `DiskBackedMoEArtifact`, and `DiskBackedRepoGraphMemory` use `mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)`. On Windows, mmaps hold an exclusive read lock on the file that prevents renaming or deletion while the mmap is open. This causes silent failures in tests that try to delete temp directories while mmap objects are still live (the `isolated_adapter_state` backup file accumulation may be related).

**Y-12 [MEDIUM] — block_graph.py run_block_graph allocates a full float32 copy of activations on every call via astype(np.float32)**
`block_graph.py` line 82: `current_activations = input_activations.astype(np.float32)`. Even when `input_activations` is already float32, `.astype(np.float32)` with no `copy=False` keyword argument always makes a copy. Adding `copy=False` would avoid the allocation when the dtype already matches.

**Y-13 [MEDIUM] — Temp directories in cpu_inference_benchmark.py and sparse_inference_benchmark.py are created and destroyed on every benchmark call**
Each benchmark function creates a `tempfile.TemporaryDirectory()`, writes artifacts, opens them, benchmarks, then deletes. For a benchmark that is called inside a timing loop, this is fine because the temp dir is created outside the timed region. But calling `benchmark_cpu_backends()` inside `phase7_system_evaluation.py` means the temp dir creation/deletion contributes to wall-clock time visible in the evaluation report.

**Y-14 [MEDIUM] — read_block in packed_graph.py always parses INT8 blocks back to float32 for the non-quantized read path**
`packed_graph.py` line 229: `matrix = quantized_matrix.astype(np.float32) * metadata.scale`. This dequantization occurs inside `read_block`. The `read_quantized_block` path avoids this, returning the INT8 matrix directly. However, `run_packed_graph` (the top-level convenience function, line 305-329) calls `read_block`, not `read_quantized_block`, meaning the INT8 path always dequantizes before compute. The CPU-backend inference path correctly uses `read_quantized_block`, but any code using the `run_packed_graph` convenience function loses the INT8 compute benefit.

**Y-15 [MEDIUM] — _embed_tokens in repo_graph_memory.py produces a collision-heavy embedding for short token lists**
`repo_graph_memory.py` line 88-95: with `dim=256`, a file with only 10 unique tokens will have most dimensions at 0 and at most 10 non-zero dimensions. The dot product similarity will be near-zero for most query/document pairs, making retrieval quality highly sensitive to exact token overlap rather than semantic proximity. This is by design (the system uses hash-based bag-of-tokens embeddings), but the 256-dim limit means high collision probability as token vocabularies grow.

**Y-16 [MEDIUM] — DiskBackedMoEArtifact reads the full manifest bytes into a Python string for JSON parsing on every __init__**
`moe_reap.py` line 183: `self.manifest = json.loads(self._mapping[manifest_start:manifest_end].decode("utf-8"))`. The mmap slice creates a bytes copy, then `decode()` creates a string copy, then `json.loads` parses it. For a MoE artifact with 64 experts and 4 blocks each, the manifest JSON could be large. This is a cold-start cost on every open.

**Y-17 [LOW] — np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=PARAM_TYPE) in block_graph.create_block allocates and zero-fills a 512×512 float16 array for every block**
`block_graph.py` line 30: `padded = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=PARAM_TYPE)`. Zero-filling 524,288 bytes for every layer in the legacy format is wasteful. The caller could pre-allocate a single zeroed buffer and reuse it with an in-place write.

**Y-18 [LOW] — The reranker matrices in IntegratedRepoRuntime are rebuilt on every __init__ call with zeros**
`integrated_repo_runtime.py` lines 57-67: `build_repo_runtime_reranker_matrices()` constructs hand-authored weight matrices from scratch on every `IntegratedRepoRuntime` instantiation. While fast (few elements), this is semantically a compile step that could be a static artifact rather than rebuilt at runtime.

**Y-19 [LOW] — json.dumps with sort_keys=True in packed_graph.py and moe_reap.py adds sort overhead at build time**
Both artifact builders use `sort_keys=True`. For the manifest dict, sorting is deterministic but adds O(k log k) per dict. For production build pipelines with thousands of layers, this is negligible but still unnecessary — structural determinism can be guaranteed by controlled construction order.

**Y-20 [LOW] — DiskBackedRepoGraphMemory.close does not close the file handle backing the mmap**
`repo_graph_memory.py` lines 293-297: `close()` only closes `self.embeddings._mmap`, but does not close the file returned by `np.load`. `np.load` with `mmap_mode` internally opens a file; closing the mmap does not close the backing file descriptor. This is a minor file descriptor leak per instance.

---

### Fatima Al-Zahra — Infrastructure Architect

**Focus: Scalability limits, horizontal scaling barriers, capacity planning**

**F-01 [CRITICAL] — The entire benchmark and evaluation pipeline is single-threaded with no parallelism**
`benchmark_comparative.py` lines 374-393: configurations are evaluated sequentially in a `for config in self.configurations` loop. With 8 configurations × 6 datasets × 100 queries each, the total query count is 4,800. All queries run serially. On an 8-core machine, parallel evaluation of independent configurations (each with its own in-memory Qdrant) would reduce wall-clock time by approximately 6-7x.

**F-02 [CRITICAL] — run_large_sweep.py's 7,350-configuration grid has no parallelism, no early stopping, and no Bayesian optimization**
The sweep enumerates all 7,350 combinations of 5 hyperparameters with no adaptive search strategy. Bayesian optimization (e.g., optuna) would typically find competitive configurations with 50-200 evaluations. The current exhaustive grid scan is approximately 35-100x more expensive than needed. With `evaluate_ndcg` taking ~1s per run, the full sweep takes ~2 hours; a targeted search would take 5-10 minutes.

**F-03 [HIGH] — Each BEIRBenchmarkRunner.run_single_dataset creates a fresh ComparativeTestbed for each dataset but does not reuse any cached engine state across configurations**
`benchmark_beir.py` lines 430-431: a new `ComparativeTestbed` is created for each dataset. Each `engine_factory(config)` call ingests the entire corpus from scratch (builds embeddings for all docs, upserts into Qdrant). With 8 configurations × corpus ingestion per config, each dataset triggers 8 full corpus ingestion cycles. The base embeddings are identical across all configurations for the same corpus — only the adapter and chelation settings differ. A single embedding computation + 8 Qdrant collection clones would be far more efficient.

**F-04 [HIGH] — No mechanism to resume a partial BEIR benchmark run**
`BEIRBenchmarkRunner.run_all` iterates sequentially through datasets with no checkpoint between them. A crash after completing 5 of 6 datasets loses all timing metadata and requires restarting from scratch. The `_dataset_cache` dict caches loaded data within a single run but is not persisted.

**F-05 [HIGH] — The integrated runtime benchmark re-ingests the full repo from scratch on every call to benchmark_integrated_repo_runtime()**
`integrated_runtime_benchmark.py` line 14: `IntegratedRepoRuntime(repo_root, top_k=3)` is called at the top of `benchmark_integrated_repo_runtime`. `IntegratedRepoRuntime.__init__` calls `ingest_repo_graph_memory(...)` synchronously, which reads and processes every source file in the repo, computes SHA256 hashes, builds the AST for Python files, and writes the full memory dir to a temp directory. This ingest is O(repo_size) every time the benchmark function is called, including during the Phase 7 system evaluation.

**F-06 [HIGH] — Benchmark corpus is stored in-memory Qdrant (:memory:) so no benchmark results persist between process restarts**
`benchmark_comparative.py` line 240: `qdrant_location=":memory:"`. Each new evaluation engine starts with an empty collection and ingests all documents. For a production benchmark campaign that is interrupted and restarted, all ingestion work must be repeated. A persistent Qdrant path with schema versioning would allow warm starts.

**F-07 [HIGH] — The MoE routing in run_moe_artifact evaluates all top_k=2 experts for every token, even when the router confidently selects one**
`moe_reap.py` lines 271-285: after computing router logits, the top-2 experts are always executed. Even if the top-1 expert has logit=100.0 and the second has logit=-5.0, both are executed. A confidence threshold (e.g., execute only top-2 when top-1 weight < 0.95 after softmax) would reduce expert execution cost for high-confidence routing decisions.

**F-08 [HIGH] — SparseChunkCache has a fixed max_cached_chunks=8 regardless of the block dimensions or available RAM**
`sparse_cpu_inference.py` line 36: `max_cached_chunks: int = 8`. For tiny blocks (RERANK_FEATURE_DIM=8), this cache fits entirely in a few KB. For large blocks (512×512 float32 = 1 MB each), 8 chunks = 8 MB. The cache size is not tuned to available RAM or the working set size. A capacity in bytes rather than in chunk count would be more principled.

**F-09 [MEDIUM] — No caching of MTEB dataset loads across different benchmark scripts**
`benchmark_beir.py`, `benchmark_distillation.py`, `benchmark_multitask.py`, and `run_sweep.py` all call `load_mteb_data` independently. If multiple benchmark scripts are run in sequence (e.g., during a research campaign), the SciFact dataset is downloaded/parsed N times. A process-level or disk-level dataset cache would eliminate redundant loads.

**F-10 [MEDIUM] — The Phase 7 evaluation suite has no progress reporting between the 7 benchmarks**
`phase7_system_evaluation.py` lines 31-37: 7 benchmarks run sequentially with no progress indication. If a benchmark takes unexpectedly long, there is no way to determine which benchmark is running without attaching a debugger.

**F-11 [MEDIUM] — No parallel ingestion in ingest_repo_graph_memory**
`repo_graph_memory.py` lines 189-221: file traversal, text reading, tokenization, and embedding computation are all sequential. For a large repo with 1,000 Python files, AST parsing and embedding computation are the bottleneck. Python multiprocessing (ProcessPoolExecutor) could parallelize per-file work with near-linear speedup on multicore machines.

**F-12 [MEDIUM] — disk_llm_estimator.py uses simplistic linear bandwidth model without modeling queue depth, sequential vs. random I/O, or thermal throttling**
`disk_llm_estimator.py` lines 114-115: `effective_bandwidth_gbps = ssd_bandwidth_gbps * io_efficiency`. The 70% efficiency factor is applied uniformly. Real NVMe drives under sustained sequential read for large models operate at near 100% of rated bandwidth for the first few GB, then thermal throttle to 40-60%. The linear model overestimates token generation throughput for sustained workloads.

**F-13 [MEDIUM] — No sharding or chunked processing for large BEIR corpora (NQ: 2.6M docs, HotpotQA: 5.2M docs)**
For the "full" tier, NQ and HotpotQA are sampled to 10,000 docs. The sampling itself is efficient, but if `default_sample_size=None` were used, ingesting 2.6M documents would require approximately 2.6M embedding computations. The `engine.ingest` interface has no streaming ingest path for corpora larger than available RAM, though `ingest_streaming` exists on `AntigravityEngine` and is not used here.

**F-14 [MEDIUM] — The evaluation query count cap (max_queries=100) significantly undersamples some BEIR datasets**
NQ has 3,452 test queries; HotpotQA has 7,405. Evaluating only 100 queries provides high-variance NDCG estimates. The confidence interval for NDCG@10 with N=100 queries is approximately ±0.05 at 95% confidence. Research-grade BEIR evaluation uses all queries. The cap is a necessary tradeoff for speed, but the statistical validity of comparisons using capped evaluation is not documented.

**F-15 [MEDIUM] — No CI-time benchmark gate for performance regressions**
The CI workflow (`.github/workflows/test.yml`) runs unit tests but no performance benchmarks. A performance regression in embedding throughput, sedimentation training speed, or retrieval latency would only be caught manually during a benchmark campaign. Adding a lightweight micro-benchmark with assertable thresholds to CI would catch regressions early.

**F-16 [LOW] — build_repo_runtime_reranker_matrices uses hand-coded weight matrices with no training**
`integrated_repo_runtime.py` lines 57-67: the reranker is a hand-authored linear map with domain-specific intuitions baked in. There is no documented calibration process or confidence interval for the scoring weights. The 0.7, 0.5, 1.2, etc. weights are arbitrary without empirical backing, and the RERANK_FUSION_WEIGHT=0.15 is similarly unjustified.

**F-17 [LOW] — TASK_SUITES in benchmark_multitask.py duplicates the tier definitions in BEIRDatasetRegistry**
`benchmark_multitask.py` lines 36-39: `TASK_SUITES` mirrors `TIER_ORDER` and `BEIRDatasetRegistry._DATASETS` tiers. The two registries can diverge independently, creating inconsistency in which datasets are used in multi-task vs. BEIR benchmarks.

**F-18 [LOW] — The sample_seed=42 in DatasetLoader is a fixed global default with no per-experiment variation**
Using seed=42 for corpus sampling means every experiment samples the same 10,000 documents from NQ/HotpotQA. If those 10,000 docs happen to contain a distribution bias (e.g., all sampled from one time period), results will systematically over- or under-estimate performance. Reporting results across multiple seeds (42, 123, 456) would quantify sampling variance.

**F-19 [LOW] — No memory-mapped or batched approach for building the packed graph artifact with large models**
`packed_graph.py` `build_packed_graph_artifact` accumulates `blob_parts: list[bytes]` then calls `b"".join(blob_parts)`. For a model with 100 transformer layers, the blob_parts list holds 100 independent byte strings before joining. A streaming writer that writes each block directly to a file would avoid the peak 2x memory overhead.

**F-20 [LOW] — The integrated runtime has no request queue or concurrency model**
`IntegratedRepoRuntime.answer_query` is not thread-safe (tracemalloc, mutable cache state). There is no request queue, rate limiting, or concurrent request handling. While appropriate for a research prototype, the absence of any concurrency model means scaling discussion (e.g., "X queries per second") refers only to single-threaded latency, not sustainable throughput.

---

### Derek Washington — SRE / SLO Expert

**Focus: Observability, SLO/SLA definition, alerting, runbooks, incident readiness**

**D-01 [HIGH] — No SLO or latency budget defined for any benchmark operation**
None of the benchmark scripts define or assert on latency SLOs. `phase7_system_evaluation.py` checks `max_integrated_total_latency_ms: float = 10.0` but this threshold is a functional gate, not a latency budget with percentile definitions (p50/p95/p99). There is no understanding of tail latencies.

**D-02 [HIGH] — Benchmark latency metrics report only the mean; no p95, p99, or std deviation on per-query latency**
`benchmark_comparative.py` line 347: `latency_ms=float(np.mean(latencies))`. `BenchmarkResult` stores only mean latency. No percentile distribution, no standard deviation, no min/max is stored. A benchmark reporting only mean latency for a retrieval system is incomplete — tail latency (p95, p99) typically determines user-visible SLA adherence.

**D-03 [HIGH] — cpu_inference_benchmark.py uses median of 5 trial means, but does not report standard deviation, min, or max of trials**
`cpu_inference_benchmark.py` line 59: `timings_ms[label] = float(median(trial_timings_ms))`. With 5 trials, the distribution of trial times is not reported. If one trial has a major GC pause or OS scheduling event, the median hides it. At minimum, std deviation and the range (min, max) should be reported alongside the median.

**D-04 [HIGH] — No alert or log when sedimentation has zero training samples**
`run_sedimentation_cycle` is called after accumulating a chelation log. If the threshold is too high or the queries don't produce chelation events, the log may be empty and sedimentation is a no-op. This silent no-op is noted in the Session 29 findings but there is still no WARNING-level log entry emitted when `len(chelation_log) == 0` before sedimentation. Silent no-ops make it impossible to distinguish "training worked but didn't improve" from "training never ran."

**D-05 [HIGH] — No structured event log for the overnight campaign**
`run_overnight_campaign.py` passes `-u` (unbuffered) to the subprocess but does not redirect output to a timestamped log file. The campaign result is only recoverable if the terminal session is still active. There is no log rotation, no structured error record, and no run summary artifact.

**D-06 [HIGH] — IntegratedRepoRuntime.answer_query mixes timing with tracemalloc inside the timed region**
`integrated_repo_runtime.py` lines 140-183: `tracemalloc.start()` and `tracemalloc.stop()` are inside the `total_start / total_latency_ms` measurement. The overhead of tracemalloc frame tracking is included in the reported latency. Reported latency is therefore inflated by profiling overhead — this biases benchmarks that use `answer_query` as a timing target.

**D-07 [HIGH] — No error metric tracked in any benchmark — all failures are silently skipped or produce 0.0 scores**
In `evaluate_engine` (benchmark_distillation.py line 217-252), queries with no qrels are silently skipped (`if not relevant_docs: continue`). In `map_predicted_ids`, failures fall back to raw IDs with only a print warning. In `benchmark_multitask.py`, failed tasks produce `status: 'failed'` but are excluded from aggregate metrics without raising an alert. Error rates are not recorded or reported.

**D-08 [MEDIUM] — Phase 7 gate thresholds in PromotionThresholds are hardcoded with no justification or historical baseline**
`phase7_system_evaluation.py` lines 15-22: `min_storage_read_reduction_pct=50.0`, `min_cpu_speedup_vs_float32=1.0`, etc. These values have no documented rationale. In particular, `min_cpu_speedup_vs_float32=1.0` (requires speedup ≥ 1x) is trivially achievable since even equal performance passes. This effectively makes the CPU check non-binding.

**D-09 [MEDIUM] — No wall-clock timestamp or run ID in BenchmarkResult or MultiDatasetResult**
`benchmark_comparative.py` lines 55-66: `BenchmarkResult` has no `timestamp`, `run_id`, or `git_commit` field. When results are serialized to JSON, there is no way to associate a result file with the code version that produced it. Regression tracking is impossible.

**D-10 [MEDIUM] — No warm-up queries before the benchmark timing in benchmark_comparative.py**
`ComparativeTestbed.evaluate_single_config` starts timing from the first query (`start = time.perf_counter()` at line 329). The first query for a new engine instance will trigger JIT compilation paths, model loading warm-up, and Qdrant index initialization. The cold-start latency inflates the reported mean, making early queries unrepresentative of steady-state performance.

**D-11 [MEDIUM] — No retry logic or error handling in the BEIR data loading fallback paths**
`benchmark_beir.py` lines 274-277: `corpus, queries, qrels = load_mteb_data(dataset_name)`. If this raises (network error, disk full), the exception propagates and terminates the entire `run_all` campaign. There is no retry with exponential backoff, no per-dataset error isolation, and no partial result save before the exception unwinds.

**D-12 [MEDIUM] — Benchmark results accumulate in memory (self.results) with no size limit**
`BEIRBenchmarkRunner.results` and `ComparativeTestbed.results` grow unbounded with each call to `run_single_dataset` / `evaluate_single_config`. For a long-running research campaign that calls `run_all_preloaded` repeatedly, memory grows linearly with total evaluations.

**D-13 [MEDIUM] — benchmark_multitask.py output reports only the first 20 NDCG scores per task**
`benchmark_multitask.py` line 344: `'ndcg_list': [float(x) for x in ndcg_list[:20]]`. With `max_queries=100`, 80% of per-query NDCG scores are discarded in the output. If the tail of the distribution shows anomalies, they will not be visible.

**D-14 [MEDIUM] — No checksum or integrity check on adapter checkpoint files before loading**
`benchmark_utils.py` lines 125-137: `isolated_adapter_state` moves checkpoint files but does not verify their integrity (CRC, SHA256) after the move. A corrupted checkpoint (disk error, incomplete write) will cause silent degradation in subsequent benchmarks, not an explicit error.

**D-15 [LOW] — run_sweep.py output file "sweep_results.json" is a fixed default name with no timestamp**
If two sweep runs are started with the same default output path, the second run clobbers the first. There is no automatic namespacing by timestamp or experiment ID.

**D-16 [LOW] — benchmark_beir.py DatasetLoader logs "beir_load_complete" but does not log the wall-clock time for the load**
The `log_event` at line 282 does not include `elapsed_seconds` in its payload, making it impossible to identify slow dataset loads from logs alone.

**D-17 [LOW] — run_sweep.py and run_large_sweep.py depend on benchmark_evolution.py (load_mteb_data, evaluate_ndcg) which is not in the reviewed file list**
Both sweep scripts import from `benchmark_evolution.py`, which is not among the files listed for review. If `benchmark_evolution.evaluate_ndcg` differs from `benchmark_utils.ndcg_at_k` in edge-case behavior (e.g., handling ties, empty retrievals), sweep results may not be comparable to BEIR benchmark results.

**D-18 [LOW] — RERANK_FUSION_WEIGHT in integrated_repo_runtime.py (0.15) is not a config value and cannot be tuned without code changes**
`integrated_repo_runtime.py` line 29: `RERANK_FUSION_WEIGHT = 0.15` is a module-level constant, not a parameter or config entry. Any tuning experiment requires a code change, not a config change. The Phase 7 promotion review identifies "ranking quality" as an optimization area, but the fusion weight is not accessible via the configuration system.

**D-19 [LOW] — No benchmark for moe_reap_benchmark.py or moe_reap.py integrated into phase7_system_evaluation.py**
`phase7_system_evaluation.py` imports and calls 7 benchmarks but does not include any MoE-specific benchmark. The Phase 7 promotion review explicitly defers MoE integration, but the benchmark infrastructure has no placeholder gate for it, making it easy to promote without it.

**D-20 [LOW] — No documented runbook for interpreting benchmark results or diagnosing regressions**
There is a `docs/distillation-experiment-protocol.md` and `docs/phase4-experiment-protocol.md` but no general benchmark runbook. Interpreting a drop in `top3_hit_rate` from 75% to 60% across the integrated runtime benchmark requires knowledge that lives only in developer heads.

---

## CHALLENGE PHASE

### Dr. Vasquez challenges Marcus O'Brien on M-11 (neighbor token set construction overhead):

"M-11 flags neighbor token set construction per query. But in the current evaluation suite, there are only 10 benchmark queries. The absolute time cost is negligible. Are you certain this is HIGH rather than LOW severity?"

**O'Brien responds:** The finding stands at HIGH because the code is a general-purpose retrieval API (`query()` in `DiskBackedRepoGraphMemory`), not just the benchmark harness. Any production use with hundreds of queries per minute against a repo with 5,000+ nodes will encounter the O(queries × candidates × neighbor_tokens) cost. The benchmark suite not exercising it at scale does not make the code fast — it makes the benchmark under-representative.

### Marcus O'Brien challenges Dr. Vasquez on V-11 (compute_stability is always 1.0):

"V-11 says the stability metric is misleading because the engine is deterministic. But isn't this a design requirement — we WANT deterministic retrieval? The metric correctly validates that there is no non-determinism bug."

**Vasquez responds:** Partially conceded. The metric is useful as a determinism regression test. However, the benchmark script presents it as a 'Stability (Jaccard)' measurement implying variance characterization. If the intent is determinism validation, it should be documented as such, not presented alongside quality metrics. The finding is downgraded to LOW.

### Yuki Tanaka challenges Fatima Al-Zahra on F-02 (exhaustive grid sweep without Bayesian optimization):

"F-02 says we should use Bayesian optimization. But the sweep is a one-time research tool, not a production service. The 7,350 exhaustive grid gives the team the full response surface, not just the optimum. Is the 'inefficiency' actually a design choice?"

**Al-Zahra responds:** Fair point on intent. The finding is more precisely: the sweep has no mechanism to limit its own total cost, no adaptive budget, and no way to report interim best-so-far results. Even if exhaustive exploration is the goal, the O(N²) JSON write (V-01/M-01) makes it operationally fragile and slow in the later iterations. The optimization methodology critique is reduced to MEDIUM, but the infrastructure fragility remains CRITICAL.

### Fatima Al-Zahra challenges Derek Washington on D-01 (no SLO defined):

"D-01 says there are no SLOs. This is a research prototype. Defining SLOs before the system is stable seems premature. Isn't this a forward-looking concern rather than a current gap?"

**Washington responds:** The Phase 7 promotion review includes a latency gate (`max_integrated_total_latency_ms: float = 10.0`) which IS a functional SLO. The problem is that it is defined on mean latency without tail characterization. The promotion gate could pass with mean=9ms and p99=200ms. The SLO finding is valid but should be scoped to "existing latency gates lack tail characterization" rather than "no SLOs exist."

### Devil's Advocate challenges all experts on the INT8 quantization correctness findings:

"The panel is flagging INT8 precision as a concern multiple times. But the benchmark reports show `max_abs_diff` consistently at ~0.001 or lower. If the precision loss is that small in practice, why does it matter?"

**Panel response:** The `max_abs_diff` is measured on the tiny 256→128→10 test model with random weights drawn from N(0, 0.1). For models with weight norms typical of trained models (not random), the per-element error accumulates across layers. The Phase 7 report reports `1.19x` CPU speedup for prequantized INT8 vs float32 — this is a marginal speedup that may not justify the engineering complexity of the INT8 path unless scale is reached where per-token latency matters.

### Devil's Advocate challenges Dr. Vasquez on V-04 (tracemalloc in answer_query):

"V-04 says tracemalloc is a process-global singleton and could be corrupted by concurrent use. But the integrated runtime has no concurrency model (confirmed by F-20). Isn't this a non-issue in practice?"

**Vasquez responds:** It is not a non-issue. Even in single-threaded use, the benchmark suite calls `answer_query` multiple times in a loop (lines 26-38 in `integrated_runtime_benchmark.py`). Each call starts and stops tracemalloc. If the Python GC happens to trigger tracemalloc internally between calls, or if a test framework starts tracemalloc before the benchmark, the `start()` call silently resets the existing snapshot. The finding remains at HIGH because the tracemalloc overhead is also included in the reported latency (see D-06), conflating profiling cost with runtime cost.

### Derek Washington challenges Yuki Tanaka on Y-05 (Windows mmap slicing creates bytes copy):

"Y-05 says Windows mmap slicing returns a bytes copy. Is this confirmed behavior or a theoretical concern? The tests pass on CI which presumably uses Linux."

**Tanaka responds:** This is confirmed Windows behavior. On Windows, `mmap[start:end]` returns `bytes` while on Linux it returns a `bytes` object as well, but the kernel page cache on Linux makes the copy inexpensive (copy-on-write kernel page). On Windows with NTFS and no page cache equivalence, each mmap slice triggers an actual memory copy. The CLAUDE.md notes this repo runs on `win32` (Platform: win32 in the env). The CI uses Linux hosted runners, so Y-05 only manifests on the developer's machine. Given the project is developed on Windows 11 (per env), this is a real performance issue for local development and benchmarking.

---

## CONVERGE

### Stack-Ranked Findings by Severity

#### CRITICAL (Must Fix)

| ID | Finding | Impact |
|---|---|---|
| V-01 / M-01 | run_large_sweep.py O(N²) JSON read-modify-write per iteration | Sweep degrades to multi-second writes per run at iteration 5000+ |
| V-02 | Sweep engine reuse without full state reset across configurations | All sweep results are potentially non-independent; data validity at risk |
| M-02 | NumpyInt8DynamicBackend computes weight scale dynamically despite stored INT8 having known scale | 4 extra full-matrix passes per inference call; defeats INT8 advantage |
| M-03 | repo_graph_memory.query() rebuilds node token sets from scratch per call | O(queries × candidates × tokens) at every query; primary query bottleneck |
| M-04 | ingest_repo_graph_memory np.vstack creates 2x peak allocation | Memory spike during ingest that scales with repo size |
| Y-01 | Legacy block_graph format pads every matrix to 512×512 (512 KB each) | 99.5% storage waste in the legacy path; still used in profile_sample_inference |
| Y-02 | Full nodes.jsonl materialized in Python heap on every DiskBackedRepoGraphMemory init | Hundreds of MB of heap for large repos; kills the "disk-resident" premise |
| F-01 | All benchmark configurations evaluated serially; zero parallelism | 6-7x unnecessarily slower wall-clock time on multicore hardware |
| F-02 | 7,350-config exhaustive grid with no adaptive budget or early termination | Operationally fragile; 35-100x more evaluations than Bayesian search needs |

#### HIGH

| ID | Finding | Impact |
|---|---|---|
| V-03 | isolated_adapter_state leaves permanent backup files on crash | Accumulating .pt backup files (confirmed in git status: 3 orphan backups) |
| V-04 | tracemalloc started/stopped per query, corrupts external traces, inflates latency | Latency measurements include profiling overhead; non-thread-safe |
| V-05 | No timeout on MTEB load_data() calls | Can block indefinitely on large datasets or network issues |
| V-06 | run_overnight_campaign.py does not capture subprocess stderr to a log file | Overnight run failures leave no diagnostic artifacts |
| V-07 | MockNVMeDrive.__del__ may silently swallow exceptions for partially-constructed objects | Resource leak in error paths |
| V-08 | DiskBackedPackedGraph.close() is not thread-safe | Double-close races under concurrent access |
| V-09 | benchmark_distillation offline mode forces use_centering=True as a bug workaround | Known correctness bug makes offline benchmark results engine-state-dependent |
| V-10 | phase7_system_evaluation.py has no partial result persistence | Full re-run required on any intermediate failure |
| M-05 | SHA-256 hashing per token in _embed_tokens | 20-100x slower than non-cryptographic alternative |
| M-06 | np.frombuffer(...).astype(np.float32) always allocates on read_block | Per-block allocation pressure in inference hot path |
| M-07 | format_ascii_table and aggregate functions re-scan results O(D*R) times | Minor at current scale, patterned inefficiency |
| M-08 | reap_prune_experts computes Frobenius norms redundantly at build time | Quadratic scan of all expert weights on every artifact build |
| M-09 | map_predicted_ids issues one Qdrant retrieve() per query | 100 per-query Qdrant RPCs per evaluation pass; batching opportunity |
| M-10 | Python-level for loop dominates timing in cpu_inference_benchmark | Benchmark measures Python dispatch overhead, not arithmetic throughput |
| M-11 | Neighbor token set built fresh per query per node in graph scoring | O(queries × candidates × neighbors × tokens) traversal |
| D-01 | No SLO with tail latency characterization | Phase 7 latency gate is on mean only; p99 could be 20x mean |
| D-02 | Benchmark latency reports only mean, not p95/p99 | Research results can't be used for SLA planning |
| D-03 | cpu_inference_benchmark reports only median of 5 trials, not variance | Benchmarks hide GC/scheduling noise in timing data |
| D-04 | No WARNING log when sedimentation runs with empty chelation_log | Silent no-ops indistinguishable from effective training |
| D-05 | run_overnight_campaign has no structured event log | Overnight run outcomes are not recoverable post-hoc |
| D-06 | tracemalloc overhead included in measured latency in answer_query | Latency inflation of unknown magnitude in benchmark data |
| D-07 | All benchmark failures silently produce 0.0 or are excluded from aggregates | Error rates are invisible in summary metrics |
| Y-03 | First query after cold start reads entire embedding matrix from disk | Cold-start query latency unacceptably high; not characterized |
| Y-04 | INT8 query path uses Python for-loop chunking instead of vectorized matmul | 10-16x more Python dispatches than necessary per query |
| Y-05 | Windows mmap slicing creates bytes copy on each block read | 2x memory bandwidth on the primary development platform |
| Y-06 | edges.json loaded non-streaming via json.load | Full file deserialized into heap; should stream line-by-line |
| Y-07 | edges.json written with indent=2 (verbose) | 3-4x larger file than necessary; proportional parse overhead |
| Y-08 | traditional_host_inference latency model double-counts INTERNAL_FLASH_LATENCY | Theoretical speedup calculations are incorrect |
| Y-09 | SparseChunkCache created fresh per query; no cross-query benefit | Cache warmup wasted; same reranker chunks loaded N times per benchmark |
| F-03 | ComparativeTestbed re-ingests corpus per configuration | 8x ingestion cost vs. embedding once and cloning Qdrant |
| F-04 | No resume capability for partial BEIR benchmark runs | Hours of work lost on any crash mid-run |
| F-05 | Integrated runtime benchmark re-ingests full repo on every call | O(repo_size) ingest inside every Phase 7 benchmark call |
| F-06 | In-memory Qdrant loses all state on process restart | No benchmark warm-start; full ingestion every run |
| F-07 | MoE routing always evaluates top_k=2 regardless of router confidence | CPU cycles wasted on low-weight experts |
| F-08 | SparseChunkCache capacity defined in chunk count, not bytes | Cannot be properly sized to available RAM |

#### MEDIUM

| ID | Finding | Impact |
|---|---|---|
| V-11 | compute_stability metric always returns 1.0 for deterministic engine | Misleading metric presented as variance measurement |
| V-12 | No resource limits on large parameter sweep | Unconstrained RAM/time consumption possible |
| V-13 | isolated_adapter_state is non-atomic on Windows | Race window between copy and unlink can cause state corruption |
| V-14 | ChelationConfig mutation in sweeps is not thread-safe | Safe now; dangerous if parallelized |
| V-15 | No assertion that adapter reset succeeded | Silent partial reset possible |
| V-16 | Training cycle query wrap-around silently truncates | Weaker training signal without warning |
| V-17 | corpus sampling allocates O(N) string keys in memory | >1 GB allocation for NQ/HotpotQA full corpus |
| V-18 | No structured log of sweep interruptions | Manual inspection required to resume after crash |
| M-12 | SparseChunkCache LRU uses Python OrderedDict overhead | Fine for current sizes; patterned inefficiency |
| M-13 | JSON manifest parsed on every artifact open | Could be cached after first open or replaced with binary header |
| M-15 | _active_chunk_ranges sorts already-ordered output of flatnonzero | Unnecessary O(k log k) sort |
| M-16 | Per-query metric lists use Python appends vs. pre-allocated arrays | Fine at current scale; not representative of performance-oriented code |
| M-17 | np.abs(matrix) temporary in _quantize_matrix_to_int8 | 2x peak memory for large weight matrices at quantization time |
| M-18 | build_moe_reap_artifact builds full blob in memory | 2x memory peak for large expert blobs |
| D-08 | PromotionThresholds min_cpu_speedup_vs_float32=1.0 is trivially passable | Non-binding gate; disguises poor INT8 throughput |
| D-09 | No timestamp or git commit hash in benchmark result files | Cannot correlate results with code versions |
| D-10 | No warm-up before benchmark timing begins | Cold-start latency inflates early query measurements |
| D-11 | No retry on MTEB dataset load failures | Single network glitch aborts entire campaign |
| D-12 | Benchmark results accumulate in memory without size limit | Memory growth in long-running campaigns |
| D-13 | Only first 20 NDCG scores per task saved to output | 80% of per-query data discarded |
| D-14 | No integrity check on adapter checkpoint before load | Corrupted checkpoints cause silent metric degradation |
| Y-10 | embeddings.npy write and mmap axis order not explicitly enforced | Potential sequential access pattern mismatch |
| Y-11 | Windows mmap holds file lock, blocking temp dir deletion | Temp dir cleanup silently fails on Windows |
| Y-12 | input_activations.astype(np.float32) without copy=False | Unnecessary allocation when dtype already matches |
| Y-13 | Temp dir creation/deletion inside Phase 7 evaluation contributes to timing | Wall-clock timing includes filesystem operations |
| Y-14 | run_packed_graph convenience function dequantizes INT8 before compute | INT8 arithmetic benefit lost when using top-level API |
| Y-15 | 256-dim hash embedding has high collision rate for large token vocabularies | Retrieval quality degrades for large repos |
| F-09 | No cross-script MTEB dataset cache | Redundant dataset downloads across benchmark scripts |
| F-10 | No progress reporting between Phase 7 benchmarks | Cannot identify slow benchmark without debugger |
| F-11 | No parallel file processing in ingest_repo_graph_memory | Near-linear speedup available with ProcessPoolExecutor |
| F-12 | disk_llm_estimator uses linear bandwidth model without thermal throttling | TPS estimates are 20-40% optimistic for sustained workloads |
| F-13 | No sharding for large BEIR corpora | ingest_streaming available but unused in benchmarks |
| F-14 | N=100 query cap gives ±0.05 NDCG confidence intervals | Statistical validity of comparisons is undocumented |
| F-15 | No CI performance benchmark gate | Regressions caught only in manual campaign runs |
| F-16 | Reranker weights are hand-coded, not trained | No documented calibration; fusion weight arbitrary |
| F-17 | TASK_SUITES duplicates BEIRDatasetRegistry tier definitions | Two registries can diverge independently |
| F-18 | Fixed seed=42 corpus sampling for all experiments | Sampling variance not characterized |

#### LOW

| ID | Finding | Impact |
|---|---|---|
| V-19 | map_predicted_ids duplicated in benchmark_distillation.py vs benchmark_utils.py | Divergent implementations; metrics not directly comparable |
| V-20 | find_keys/find_payload have no depth limit | Theoretical RecursionError on pathological MTEB structures |
| M-19 | Separator generation in format_ascii_table recomputes len(header) | Negligible; cosmetic |
| M-20 | ndcg_at_k duplicated in benchmark_distillation.py vs benchmark_utils.py | Duplicate code; divergence risk |
| D-15 | run_sweep.py default output filename not timestamped | Runs clobber each other |
| D-16 | DatasetLoader beir_load_complete log event omits elapsed time | Cannot identify slow loads from logs |
| D-17 | run_sweep.py depends on benchmark_evolution.py not in reviewed files | Possible silent metric divergence |
| D-18 | RERANK_FUSION_WEIGHT is a module constant, not a config value | Tuning requires code changes, not config changes |
| D-19 | No MoE benchmark gate in phase7_system_evaluation | MoE can be promoted without benchmark coverage |
| D-20 | No general benchmark runbook | Regression diagnosis relies on institutional knowledge |
| Y-16 | DiskBackedMoEArtifact parses full JSON manifest on every open | Cold-start cost; acceptable for PoC |
| Y-17 | block_graph.create_block zero-fills 512×512 buffer for every block | Wasted zero-fill in legacy path |
| Y-18 | Reranker matrices rebuilt in Python on every runtime init | Deterministic constant; could be static |
| Y-19 | sort_keys=True in json.dumps adds unnecessary sort overhead | Trivial at current artifact sizes |
| Y-20 | DiskBackedRepoGraphMemory.close does not close np.load file handle | Minor file descriptor leak |
| F-16 | Reranker hand-coded weights undocumented | No empirical backing for 0.7/0.5/1.2 weights |
| F-19 | build_packed_graph_artifact accumulates blob_parts in memory | 2x peak for large models; streaming would help |
| F-20 | No concurrency model in IntegratedRepoRuntime | QPS numbers refer only to single-threaded latency |

---

## DISSENT LOG

**Dissent 1 — Devil's Advocate on F-01 (serial benchmarks):**
The parallelism proposal assumes configurations are independent. In the benchmark infrastructure, `isolated_adapter_state` uses a shared file path for checkpoint management. Parallel execution of multiple configurations would require per-configuration adapter paths to avoid collisions. The infrastructure change needed to enable parallelism is non-trivial and could introduce new correctness bugs.
*Panel ruling:* Dissent noted. Finding F-01 retains CRITICAL severity but the implementation path should go through per-configuration adapter paths first, as already addressed by the Session 28-31 checkpoint isolation work.

**Dissent 2 — Devil's Advocate on M-02 (dynamic weight scale computation):**
The weight scale IS known for pre-quantized INT8 blocks (it is stored in the block metadata). But for the `dynamic_int8` path, the weights are NOT pre-quantized — the caller passes float weights and the backend quantizes on-the-fly. The dynamic path is the correct behavior for that path. The stored-scale path (`matmul_quantized_weights`) correctly bypasses the scale computation.
*Panel ruling:* Dissent partially sustained. M-02 is scoped more precisely: `NumpyInt8DynamicBackend.matmul` is the wrong path to use for pre-quantized INT8 blocks. The code correctly differentiates paths via `matmul_quantized_weights`. The real issue is that `matmul` is called with pre-quantized blocks in some code paths that check `INT8_STORAGE_DTYPE` but still call `backend.matmul` (e.g., `run_packed_graph` in packed_graph.py lines 228-229 calls `read_block` which dequantizes). M-02 is narrowed to: "the packed_graph.run_packed_graph convenience function dequantizes before compute, losing INT8 benefit" — which overlaps with Y-14.

**Dissent 3 — Devil's Advocate on V-02 (sweep engine reuse):**
The sweep comments say "Reuse base engine to avoid Qdrant file lock issues" — this is an explicit design choice to work around a real operational constraint on Windows. The alternative (creating a new engine per configuration) would require either in-memory Qdrant (no persistence between runs) or a new on-disk Qdrant path for each of 7,350 configurations, generating 7,350 temporary directories.
*Panel ruling:* Dissent noted but does not change severity. The engine reuse is a pragmatic workaround but it means the 7,350 results are not independent measurements. At minimum, the comment should document which state IS and IS NOT reset between iterations.

---

## FEASIBILITY GATE

### Critical Findings — Feasibility Assessment

| Finding | Severity | Fix Effort | Risk | Recommended Action |
|---|---|---|---|---|
| V-01/M-01 — O(N²) JSON write in large sweep | CRITICAL | S (1 day) | Low | Replace JSON write with append-only JSON Lines (JSONL) or use the CSV-only path for incremental saves. The JSON file can be built from CSV post-hoc. |
| V-02 — Sweep engine reuse | CRITICAL | M (3 days) | Medium | Document which state is reset per iteration and add explicit assertions. For true independence, switch to in-memory Qdrant (`:memory:`) per config with a fresh engine, accepting the ingestion cost per iteration. |
| M-02/Y-14 — INT8 path loses benefit in convenience API | CRITICAL | S (1 day) | Low | `run_packed_graph` should dispatch to `read_quantized_block` + `matmul_quantized_weights` for INT8 artifacts. Existing backend infrastructure already supports this path. |
| M-03 — Token set rebuilt per query per node | CRITICAL | S (2 days) | Low | Precompute `frozenset(node.tokens)` for each node during ingest and store in `CodeNode` or a parallel dict. Same for path_token_set. |
| M-04 — np.vstack 2x peak allocation | CRITICAL | S (1 day) | Low | Pre-allocate `embeddings = np.empty((len(all_nodes), embedding_dim), dtype=np.float32)` and fill in a loop with `embeddings[i] = _embed_tokens(...)`. |
| Y-01 — Legacy format 512×512 padding waste | CRITICAL | S (1 day) | Low | Deprecate the legacy path in production code; direct all new uses to `DiskBackedPackedGraph`. Keep the legacy path only for backward-compatibility tests. |
| Y-02 — Full nodes.jsonl in Python heap | CRITICAL | M (3 days) | Medium | For very large repos, switch to a SQLite-backed node store or numpy-backed node ID index. For the current PoC scope, document the memory budget per repo size class. |
| F-01 — Serial benchmark evaluation | CRITICAL | M (3 days) | Medium | Add `--parallel N` flag using `concurrent.futures.ProcessPoolExecutor` with per-config engine factories. Requires per-config adapter paths (already isolated in Session 28-31 work). |
| F-02 — Exhaustive grid sweep | CRITICAL | M (3 days) | Low | Add `--search-strategy {grid,random,bayesian}` to run_large_sweep.py. Use optuna for Bayesian search. Keep grid as an option for full-surface exploration. |

### High Findings — Selected Feasibility Assessment

| Finding | Severity | Fix Effort | Risk | Recommended Action |
|---|---|---|---|---|
| V-03 — Orphan backup files | HIGH | S (1 day) | Low | Add try/finally in `isolated_adapter_state` to clean up backup on any exception path. Add a cleanup scan at startup. |
| V-04 / D-06 — tracemalloc in answer_query | HIGH | S (half day) | Low | Move `tracemalloc.start()` before the `total_start` clock. Better: make heap tracking opt-in via parameter. |
| M-05 — SHA-256 hashing per token | HIGH | S (1 day) | Low | Replace `hashlib.sha256` in `_stable_hash` with a fast non-cryptographic hash (FNV1a, mmh3). Keep the same interface. |
| M-09 — Per-query Qdrant retrieve | HIGH | S (1 day) | Low | Build an in-memory `qdrant_id_to_doc_id` dict at engine ingest time. Remove the per-query retrieve call. |
| D-02 — Mean-only latency reporting | HIGH | S (1 day) | Low | Add `p95_latency_ms`, `p99_latency_ms`, `std_latency_ms` to `BenchmarkResult`. Compute from the per-query `latencies` list before discarding it. |
| D-04 — Silent empty chelation log | HIGH | S (half day) | Low | Add a `logger.warning` (or `log_event` with level=WARN) when `len(self.chelation_log) == 0` at the start of `run_sedimentation_cycle`. |
| Y-03 — Cold-start embedding matrix read | HIGH | M (2 days) | Low | Add a benchmark warm-up call or pre-fault the embedding mmap pages (`np.sum(self.embeddings)`) during ingest. Document the cold/warm distinction. |
| Y-04 — INT8 query chunking loop | HIGH | S (1 day) | Low | Replace the chunked loop with a single `(N, dim) @ (dim,) → (N,)` matmul in int16 or int32, checking for overflow at the maximum representable value. |
| Y-05 — Windows mmap copy | HIGH | M (2 days) | Medium | Use `np.frombuffer(self._mapping, dtype=np.int8, count=..., offset=...).reshape(...)` to read directly from the mmap without an intermediate bytes copy. Requires careful offset arithmetic. |
| F-03 — Per-config corpus ingestion | HIGH | M (3 days) | Medium | Compute embeddings once, store in a shared mmap'd array, and create per-config Qdrant collections that share the vector data. Alternatively, use Qdrant collection snapshots/clones. |
| F-05 — Runtime benchmark re-ingests repo | HIGH | S (1 day) | Low | Build the `IntegratedRepoRuntime` once outside the benchmark loop and pass it in, or use a module-level fixture pattern. |

---

## PRIORITIZED PERFORMANCE REMEDIATION ROADMAP

### Sprint 1 — Stop the Bleeding (1 week, ~6-8 developer-days)

These are quick wins that fix operationally critical issues with minimal risk.

**1.1 Fix O(N²) JSON write in run_large_sweep.py** [V-01]
Replace the per-iteration JSON read-modify-write with append-only JSONL. The CSV path already does this correctly.
Files: `run_large_sweep.py`

**1.2 Add tracemalloc opt-in flag to answer_query** [V-04, D-06]
Move `tracemalloc.start()` outside the latency timer. Add `measure_heap: bool = False` parameter. Default to False in benchmark calls.
Files: `computational_storage_poc/integrated_repo_runtime.py`

**1.3 Add per-query latency percentiles to BenchmarkResult** [D-02]
Add `p95_latency_ms`, `p99_latency_ms`, `std_latency_ms` fields. Compute from the `latencies` list that already exists in `evaluate_single_config`.
Files: `benchmark_comparative.py`, `benchmark_beir.py`

**1.4 Precompute node token sets in DiskBackedRepoGraphMemory** [M-03]
Store `frozenset(node.tokens)` per node at load time. Remove the per-query `set(node.tokens)` call.
Files: `computational_storage_poc/repo_graph_memory.py`

**1.5 Replace np.vstack with pre-allocated array in ingest_repo_graph_memory** [M-04]
Pre-allocate the embedding matrix and fill in-place.
Files: `computational_storage_poc/repo_graph_memory.py`

**1.6 Fix isolated_adapter_state cleanup on exception** [V-03]
Wrap the `shutil.move` in `isolated_adapter_state`'s finally block to guarantee backup cleanup. Add a startup scan to report orphan backup files.
Files: `benchmark_utils.py`

**1.7 Add WARNING log when chelation_log is empty before sedimentation** [D-04]
One-line fix in `run_sedimentation_cycle`.
Files: `antigravity_engine.py`

**1.8 Replace SHA-256 in _stable_hash with FNV1a** [M-05]
One-function replacement. Add a unit test verifying hash distribution.
Files: `computational_storage_poc/repo_graph_memory.py`

---

### Sprint 2 — Core Performance Fixes (2 weeks, ~15-18 developer-days)

**2.1 Fix run_packed_graph to use quantized path for INT8 artifacts** [Y-14, M-02]
Make `run_packed_graph` dispatch to `read_quantized_block` + `matmul_quantized_weights` when `storage_dtype == INT8_STORAGE_DTYPE`. Update affected tests.
Files: `computational_storage_poc/packed_graph.py`

**2.2 Replace INT8 query chunked loop with vectorized matmul** [Y-04]
Single matmul in int32 or int64 avoids per-chunk Python dispatch. Verify no overflow for the tested embedding dimensions.
Files: `computational_storage_poc/repo_graph_memory.py`

**2.3 Use np.frombuffer with offset instead of mmap slice** [Y-05]
Replace `self._mapping[start:end]` with `np.frombuffer(self._mapping, dtype=..., count=..., offset=start)` throughout all `DiskBacked*` classes. This is the critical Windows performance fix.
Files: `computational_storage_poc/packed_graph.py`, `moe_reap.py`, `mock_nvme.py`

**2.4 Build in-memory Qdrant ID lookup table to eliminate per-query retrieve** [M-09]
Add `self._qdrant_id_to_doc_id: dict` built during `ingest()`. Use it in `map_predicted_ids`.
Files: `antigravity_engine.py`, `benchmark_utils.py`

**2.5 Add SparseChunkCache persistence across queries in IntegratedRepoRuntime** [Y-09]
Move cache construction from `answer_query` to `__init__`. Share across queries.
Files: `computational_storage_poc/integrated_repo_runtime.py`

**2.6 Write edges.json in compact JSONL format** [Y-06, Y-07]
Change `json.dump(..., indent=2)` to `json.dumps(asdict(edge), separators=(",",":"))\n` per line. Add a migration note in the manifest version field.
Files: `computational_storage_poc/repo_graph_memory.py`

**2.7 Pre-fault embedding mmap on warm path** [Y-03]
After `np.load(..., mmap_mode="r")`, call `np.dot(embeddings[0], embeddings[0])` or `np.sum(embeddings[:10])` to warm the first pages. Document cold vs. warm latency in the benchmark.
Files: `computational_storage_poc/repo_graph_memory.py`, `repo_graph_memory_benchmark.py`

**2.8 Add git commit hash and timestamp to all BenchmarkResult exports** [D-09]
Use `subprocess.check_output(['git', 'rev-parse', 'HEAD'])` at benchmark start. Add `git_commit`, `timestamp`, `platform` to the JSON export header.
Files: `benchmark_comparative.py`, `benchmark_beir.py`, `benchmark_multitask.py`

---

### Sprint 3 — Benchmark Methodology & Observability (2 weeks, ~12-15 developer-days)

**3.1 Add --parallel N flag to benchmark_comparative.py and benchmark_beir.py** [F-01]
Use `concurrent.futures.ProcessPoolExecutor` with per-config adapter paths. Each worker process gets its own `isolated_adapter_state`.
Files: `benchmark_comparative.py`, `benchmark_beir.py`

**3.2 Add --search-strategy {grid,random,bayesian} to run_large_sweep.py** [F-02]
Integrate `optuna` for Bayesian TPE search. Keep grid as default for backward compatibility. Add `--n-trials` limit.
Files: `run_large_sweep.py`

**3.3 Add warm-up phase before benchmark timing** [D-10]
Run 5-10 warm-up queries (not timed) before the benchmark loop in `evaluate_single_config`.
Files: `benchmark_comparative.py`

**3.4 Add Phase 7 evaluation checkpoint persistence** [V-10]
After each benchmark completes in `evaluate_system_promotion`, write partial results to a checkpoint JSON. On restart, load and skip completed benchmarks.
Files: `computational_storage_poc/phase7_system_evaluation.py`

**3.5 Add BEIR benchmark resume capability** [F-04]
Write per-dataset results to `{output_dir}/{dataset_name}.json` as each dataset completes. On restart with `--resume`, skip datasets that have a result file.
Files: `benchmark_beir.py`

**3.6 Add benchmark result integrity checksum** [D-14]
When saving benchmark results to JSON, write a SHA256 of the content alongside it. On load, verify the hash.
Files: `benchmark_utils.py`, `benchmark_comparative.py`

**3.7 Fix memory map close to also close the backing file in DiskBackedRepoGraphMemory** [Y-20]
Call `numpy.lib.npyio.NpzFile` close semantics or use an explicit open + mmap pattern that the close method can fully release.
Files: `computational_storage_poc/repo_graph_memory.py`

**3.8 Standardize metric implementations — remove duplicates** [V-19, M-20]
Remove `ndcg_at_k`, `dcg_at_k`, `map_predicted_ids`, `load_mteb_data` from `benchmark_distillation.py` and import from `benchmark_utils.py`.
Files: `benchmark_distillation.py`

---

### Sprint 4 — Scale & Architecture (3-4 weeks, ~20-25 developer-days)

**4.1 Eliminate per-config corpus re-ingestion in BEIR benchmarks** [F-03]
Implement a `SharedCorpusEmbeddingCache` that computes embeddings once per corpus and provides each configuration a read-only view. This requires refactoring `build_real_engine_factory`.
Files: `benchmark_comparative.py`, `benchmark_beir.py`

**4.2 Add streaming node store to DiskBackedRepoGraphMemory** [Y-02]
Replace `nodes.jsonl` Python object materialization with SQLite-backed storage. Use `sqlite3` for node lookups with lazy loading. Only the embedding matrix and adjacency list need to be hot in memory.
Files: `computational_storage_poc/repo_graph_memory.py`

**4.3 Implement parallel ingest in ingest_repo_graph_memory** [F-11]
Use `concurrent.futures.ProcessPoolExecutor` for the per-file tokenize+embed loop. Collect results into the pre-allocated embedding matrix.
Files: `computational_storage_poc/repo_graph_memory.py`

**4.4 Add Bayesian hyperparameter optimization to the overnight campaign** [F-02]
Integrate optuna into `run_overnight_campaign.py` as an alternative to the grid sweep for the per-adapter-type search.
Files: `run_overnight_campaign.py`

**4.5 Add performance regression tests to CI** [F-15]
Add a `test_performance_smoke.py` with time-bounded micro-benchmarks: embedding throughput ≥ 100 docs/s, query latency ≤ 50ms, sedimentation cycle ≤ 10s for 50 queries. These run in CI as part of the test matrix.
Files: new `test_performance_smoke.py`

**4.6 Add MoE benchmark gate to phase7_system_evaluation.py** [D-19]
Add a `moe_reap_ok` check referencing `moe_reap_benchmark.py`. Include in the `all_core_checks` gate when MoE integration is complete.
Files: `computational_storage_poc/phase7_system_evaluation.py`

**4.7 Increase PromotionThresholds.min_cpu_speedup_vs_float32 from 1.0 to 1.5** [D-08]
The current 1.0 threshold requires only parity with float32, which is trivially achievable. A 1.5x threshold would require INT8 to actually accelerate inference. Review the prequantized INT8 path for the fixes from Sprint 2 before raising the threshold.
Files: `computational_storage_poc/phase7_system_evaluation.py`

---

## EXECUTIVE SUMMARY — TOP 10 PERFORMANCE FINDINGS

1. **O(N²) JSON write in run_large_sweep.py** (V-01/M-01 — CRITICAL): The large sweep re-reads and re-writes the entire JSON results file on every one of 7,350 iterations. This is a show-stopper for any long sweep run.

2. **Per-query token set reconstruction in repo_graph_memory** (M-03 — CRITICAL): `query()` rebuilds `set(node.tokens)` for every candidate node on every query call. These are immutable and should be precomputed at load time. This is the primary query latency bottleneck for the integrated runtime.

3. **Sweep engine reuse without full state reset** (V-02 — CRITICAL): The 7,350-configuration parameter sweep shares one engine instance. Only the adapter and chelation log are reset. Any persistent engine side-effect taints all results from that point forward, invalidating the sweep as comparative research data.

4. **NumpyInt8DynamicBackend computes weight scale dynamically even for pre-stored INT8** (M-02/Y-14 — CRITICAL): The `run_packed_graph` convenience function always dequantizes INT8 back to float32 before compute. The CPU-backend path correctly avoids this, but the top-level API defeats the INT8 benefit for users who call the convenience function.

5. **Full nodes.jsonl materialized in Python heap** (Y-02 — CRITICAL): `DiskBackedRepoGraphMemory` loads every code node as a Python object at init. For repos with 10,000+ symbols, this is hundreds of MB of heap — undermining the "disk-resident" design premise.

6. **Serial benchmark evaluation with zero parallelism** (F-01 — CRITICAL): All 8 configurations × 6 datasets × 100 queries run in a single thread. Available hardware parallelism is entirely unused. A 6-7x wall-clock speedup is available without changing the benchmark logic.

7. **tracemalloc overhead included in reported latency** (V-04/D-06 — HIGH): `answer_query` starts and stops the process-global `tracemalloc` inside the latency timer on every call. This inflates all reported latency numbers by an unknown amount and prevents concurrent use.

8. **SHA-256 hashing per token in embedding construction** (M-05 — HIGH): `_stable_hash` uses `hashlib.sha256` for each token. A non-cryptographic hash would be 20-100x faster. This dominates ingest time for large repositories.

9. **Per-query Qdrant retrieve in benchmark evaluation** (M-09 — HIGH): `map_predicted_ids` issues a `qdrant.retrieve()` RPC for every query evaluated. With 100 queries per configuration per dataset, this is 100 synchronous Qdrant calls per evaluation pass. A pre-built ID lookup dict would eliminate all of these.

10. **No tail latency characterization anywhere in the benchmark stack** (D-01/D-02 — HIGH): Every benchmark reports mean latency only. The Phase 7 gate checks mean latency against a 10ms threshold. A system with mean=5ms and p99=500ms passes the gate. Until percentile latencies are tracked, the benchmark data cannot support SLA planning or production readiness assessment.

---

*Panel report completed: 2026-04-04*
*Total findings: 106 (9 CRITICAL, 37 HIGH, 44 MEDIUM, 16 LOW)*
*Files reviewed: 28*
