# Computational Storage AI Proof of Concept

This project implements a testbed for the **Computational Storage (SSD Array) AI Inference** architecture.
The premise replaces matrix multiplication entirely with direct memory routing using memory tables, mapping it directly onto NAND flash memory.

Related repository docs:

- [docs/COMPUTATIONAL_STORAGE_DRIVE_NODES.md](../docs/COMPUTATIONAL_STORAGE_DRIVE_NODES.md) - canonical repo-wide summary of the hard-drive / storage-node research track
- [docs/computational-storage-transport-scope-decision.md](../docs/computational-storage-transport-scope-decision.md) - current claim boundary for the RP2040 path
- [docs/computational-storage-hardware-evidence-runbook.md](../docs/computational-storage-hardware-evidence-runbook.md) - operator workflow for real hardware evidence capture

## What Is Verified

- Exact block-graph traversal from offset `0x0`, including the first block.
- Hidden-layer `ReLU` application on non-terminal blocks so the storage path matches the trained MLP.
- Deterministic host-vs-storage output parity and theoretical latency comparisons.
- Real-data round-trip validation on the handwritten digits dataset with explicit accuracy floors.
- Deterministic trigger-sector payload generation for the USB/emulation transport path.
- Host-reader decoding and virtual-disk interception tests for sector `100`.

## Binary Format

The `compiler.py` packs neural network matrices into continuous graph nodes with the following rigid format:

| Offset | Length (Bytes) | Description |
|--------|----------------|-------------|
| 0x0 | 524,288 | **Matrix Payload** (512x512 matrix stored in FP16 format). Unused dimensions are zero-padded. |
| 0x80000 | 8 | **64-bit Guide Node**: Little-endian unsigned integer representing the byte offset of the next node/block in the neural network graph. A value of 0 indicates the end of inference. |

**Total Block Size**: 524,296 bytes (0x80008).

### Packed Manifest Artifact (Phase 1 Substrate)

The repository now also includes an experimental packed artifact path:

- `packed_graph.py` stores exact matrix shapes in a manifest
- matrix payloads are stored without `512 x 512` zero padding
- `DiskBackedPackedGraph` reads the artifact through `mmap` instead of preloading the whole file
- the packed artifact now supports both:
  - FP16 packed weights
  - INT8 packed weights with per-block scales

This path is intended as a storage-substrate upgrade for future transformer work.
It is not yet the promoted default runtime.

Run the storage-substrate benchmark with:

```bash
python storage_substrate_benchmark.py
```

Generate a packed sample artifact with:

```bash
python compiler.py --format packed
```

Generate a prequantized INT8 packed artifact with:

```bash
python compiler.py --format packed_int8
```

## CPU Inference Substrate

The repository now also includes an initial CPU execution substrate for packed artifacts:

- `cpu_backends.py` defines the backend abstraction
- `packed_cpu_inference.py` executes packed graphs through a selected backend
- `cpu_inference_benchmark.py` compares:
  - float32 on FP16 packed artifacts
  - dynamic int8 on FP16 packed artifacts
  - prequantized int8 on INT8 packed artifacts

Phase 2b adds the first low-overhead CPU path:

- weights can now be stored prequantized in the packed artifact
- the int8 backend can reuse those stored weights directly
- this avoids per-call weight requantization in the CPU runtime

Run the CPU benchmark with:

```bash
python cpu_inference_benchmark.py
```

## Sparse Selective-Loading Runtime

The repository now also includes an initial `Phase 3A` sparse FFN path:

- `sparse_cpu_inference.py` adds selective row-chunk loading for streamed blocks
- activation sparsity acts as the routing heuristic
- `SparseChunkCache` provides a small chunk-reuse cache for repeated lookups
- `sparse_inference_benchmark.py` measures bytes-per-token reduction against the dense packed baseline

Run the sparse benchmark with:

```bash
python sparse_inference_benchmark.py
```

## Repo Graph Memory

The repository now also includes an initial `Phase 4` disk-backed code-memory substrate:

- `repo_graph_memory.py` builds a local on-disk memory bundle with:
  - code-file and symbol nodes
  - hashed embedding vectors
  - graph edges for containment and local imports
- `DiskBackedRepoGraphMemory` memory-maps the embedding matrix and exposes a query API
- retrieval blends:
  - vector similarity
  - lexical coverage
  - graph-aware reranking
- `repo_graph_memory_benchmark.py` measures ingest latency, query latency, and simple hit-rate quality on repo-local code queries
- `retrieval_eval_suite.py` defines the shared retrieval benchmark cases used across the standalone memory and integrated runtime evaluations
- the int8 query path scores compressed embeddings in row chunks so it does not widen the full memory-mapped matrix on every query

Run the repo-graph memory benchmark with:

```bash
python repo_graph_memory_benchmark.py
```

## Integrated Repo Runtime

The repository now also includes an initial `Phase 5` CPU-only integration prototype for repository Q&A style retrieval:

- `integrated_repo_runtime.py` combines:
  - disk-backed repo memory retrieval
  - the packed INT8 reranker artifact
  - the sparse CPU execution path
- the runtime converts retrieval results into compact reranker features and executes a packed INT8 graph to rerank candidate files and symbols
- final candidate ordering is retrieval-dominant, with the packed reranker acting as a refinement term instead of overriding stronger retrieval evidence
- `integrated_runtime_benchmark.py` measures:
  - retrieval latency
  - inference latency
  - total end-to-end latency
  - bytes streamed from the packed artifact
  - mapped bytes and peak Python heap
  - top-1 / top-3 repo-local hit rate
- the integrated benchmark uses the same shared retrieval-eval suite as the standalone repo-memory benchmark so Phase 4, Phase 5, and Phase 6 comparisons stay aligned

Run the integrated runtime benchmark with:

```bash
python integrated_runtime_benchmark.py
```

## Memory Compression

The repository now also includes an initial `Phase 6` memory-compression path for the repo-memory layer:

- `repo_graph_memory.py` supports both:
  - float32 embedding storage
  - int8-compressed embedding storage with a stored scale
- `repo_graph_memory_compression_benchmark.py` compares mapped bytes, latency, and hit-rate quality for float32 vs int8 repo memory
- `integrated_runtime_compression_benchmark.py` compares the same compression choice inside the end-to-end integrated runtime
- the current compressed-memory path keeps Python heap usage close to the float32 path because query scoring no longer widens the full embedding matrix

Run the compression benchmarks with:

```bash
python repo_graph_memory_compression_benchmark.py
python integrated_runtime_compression_benchmark.py
```

## MoE / REAP Branch

The repository now also includes an initial parallel `Phase 3B` branch for MoE-style expert-bank storage and pruning:

- `moe_reap.py` provides:
  - a disk-backed MoE artifact format
  - routed-expert CPU execution
  - a REAP-like norm-based expert pruning compatibility path
- `moe_reap_benchmark.py` measures expert-bank pruning impact on:
  - artifact bytes
  - bytes read
  - experts evaluated

Run the MoE benchmark with:

```bash
python moe_reap_benchmark.py
```

### Usage
Generate a sample binary payload:
```bash
python compiler.py
```
This will produce a `model.bin` file containing the connected graph.

### Regenerating the computational-storage artifact

`*.cspg` packed-graph artifacts (e.g. `model.cspg`, `real_model.cspg`) are **not** tracked in the repo. They are regenerable on demand from `train_and_compile.py` / `compiler.py` and are listed in the top-level `.gitignore`.

To produce a packed FP16 artifact:

```bash
python compiler.py --format packed         # writes model.cspg (header CSPG018)
```

To produce a packed INT8 artifact:

```bash
python compiler.py --format packed_int8    # writes model.cspg (header CSPG01Q)
```

To produce a fully-trained INT8 artifact from the digits MLP (the path used by `test_real_model.py`):

```bash
python train_and_compile.py
# writes real_model.bin by default; pass artifact_format="packed_int8"
# and an output_path ending in .cspg for the packed INT8 form.
```

Notes:

- The artifact is **not** bit-deterministic across `scikit-learn` versions because training uses a small MLP whose Adam trajectory depends on numeric library versions. Tests therefore validate accuracy floors / output parity, not byte equality.
- No source code or test loads a tracked `computational_storage_poc/model.cspg` by name; tests generate artifacts in `tempfile.TemporaryDirectory()` paths (see `test_packed_graph.py`, `test_cpu_inference.py`, `test_sparse_cpu_inference.py`).

Run the full proof-of-concept validation:

```bash
python run_all_tests.py
```

The validation suite now fails if:

- host and storage outputs diverge,
- storage accuracy falls below the configured floor, or
- the speculative execution demo regresses against its sequential baseline.

## USB / Emulation Payload Contract

The USB firmware and FUSE emulator now share a deterministic transport contract:

- sector `100` returns a JSON payload rather than a hard-coded demo string,
- the payload includes the deterministic toy input vector plus the computed `4 -> 3 -> 2` block-graph result, and
- the host-side reader is tested against both the raw sector bytes and a virtual-disk file path.

Run the transport-layer tests with:

```bash
python -m unittest test_computational_storage_payload.py -v
```

## Current Scope Lock

The RP2040 path is still an experimental payload track. The full digits model is validated in software today; the firmware currently uses the deterministic toy graph above to prove the USB interception path and descriptor plumbing.

The wider repository also contains a more speculative storage-node thread:

- `mock_array.py` models multi-drive speculative node racing
- `CHELATEDAI_integration_demo.py` shows how recursive node requests could dispatch into that array

Treat that as research into drive-resident node orchestration, not as proof that the repo already runs a full LLM from a physical hard drive.

The current claim boundary is intentionally narrow:

- this is a deterministic transport proof today,
- it is not yet a validated on-device digits-model workload, and
- promotion beyond that boundary requires the gates in [docs/computational-storage-transport-scope-decision.md](../docs/computational-storage-transport-scope-decision.md).
