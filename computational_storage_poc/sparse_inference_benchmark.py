from __future__ import annotations

import os
import tempfile
import time

import numpy as np

from block_graph import BLOCK_SIZE
from cpu_backends import NumpyInt8DynamicBackend
from packed_cpu_inference import run_packed_graph_with_backend
from packed_graph import DiskBackedPackedGraph, INT8_STORAGE_DTYPE, write_packed_graph_artifact
from sparse_cpu_inference import SparseChunkCache, SparseInferenceConfig, run_sparse_packed_graph_with_backend


def _build_sparse_benchmark_matrices() -> list[np.ndarray]:
    rng = np.random.default_rng(2026)
    w1 = np.zeros((256, 128), dtype=np.float32)
    w1[:128, :128] = np.eye(128, dtype=np.float32)
    # Make the streamed block materially larger than the resident block so
    # selective loading produces a meaningful byte-reduction signal.
    w2 = rng.normal(size=(128, 512)).astype(np.float32) * 0.1
    return [w1, w2]


def _build_sparse_input_sequence(token_count: int = 32) -> list[np.ndarray]:
    inputs: list[np.ndarray] = []
    for token_index in range(token_count):
        input_activations = np.zeros((1, BLOCK_SIZE), dtype=np.float16)
        base = (token_index % 8) * 8
        active_indices = [base, base + 1, base + 8, base + 9]
        input_activations[0, active_indices] = [0.9, 0.7, 0.8, 0.6]
        inputs.append(input_activations)
    return inputs


def benchmark_sparse_inference(token_count: int = 32) -> dict[str, float]:
    matrices = _build_sparse_benchmark_matrices()
    input_sequence = _build_sparse_input_sequence(token_count=token_count)
    backend = NumpyInt8DynamicBackend()
    config = SparseInferenceConfig(chunk_rows=8, activation_epsilon=1e-6, stream_from_block=1)

    dense_outputs: list[np.ndarray] = []
    sparse_outputs: list[np.ndarray] = []
    dense_bytes = 0
    sparse_bytes = 0
    dense_latency = 0.0
    sparse_latency = 0.0
    cache_hits = 0
    cache_misses = 0
    chunks_loaded = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_path = os.path.join(temp_dir, "sparse_graph_int8.cspg")
        write_packed_graph_artifact(artifact_path, matrices, storage_dtype=INT8_STORAGE_DTYPE)
        with DiskBackedPackedGraph(artifact_path) as graph:
            cache = SparseChunkCache(max_cached_chunks=8)
            for input_activations in input_sequence:
                dense_start = time.perf_counter()
                dense_result = run_packed_graph_with_backend(graph, input_activations, backend)
                dense_latency += time.perf_counter() - dense_start
                dense_outputs.append(dense_result.output)
                dense_bytes += dense_result.bytes_read

                sparse_start = time.perf_counter()
                sparse_result = run_sparse_packed_graph_with_backend(
                    graph,
                    input_activations,
                    backend,
                    config=config,
                    cache=cache,
                )
                sparse_latency += time.perf_counter() - sparse_start
                sparse_outputs.append(sparse_result.output)
                sparse_bytes += sparse_result.bytes_read
                cache_hits += sparse_result.cache_hits
                cache_misses += sparse_result.cache_misses
                chunks_loaded += sparse_result.chunks_loaded

    max_abs_diff = float(
        np.max(
            [
                np.max(np.abs(dense_output[:, :10] - sparse_output[:, :10]))
                for dense_output, sparse_output in zip(dense_outputs, sparse_outputs)
            ]
        )
    )

    avg_dense_latency_ms = (dense_latency / token_count) * 1000.0
    avg_sparse_latency_ms = (sparse_latency / token_count) * 1000.0

    return {
        "token_count": float(token_count),
        "avg_dense_latency_ms": avg_dense_latency_ms,
        "avg_sparse_latency_ms": avg_sparse_latency_ms,
        "dense_bytes_per_token": dense_bytes / token_count,
        "sparse_bytes_per_token": sparse_bytes / token_count,
        "streamed_byte_reduction_pct": 100.0 * (1.0 - (sparse_bytes / dense_bytes)),
        "cache_hits": float(cache_hits),
        "cache_misses": float(cache_misses),
        "chunks_loaded": float(chunks_loaded),
        "max_abs_diff": max_abs_diff,
    }


if __name__ == "__main__":
    metrics = benchmark_sparse_inference()
    print("Sparse Inference Benchmark")
    for key, value in metrics.items():
        if key.endswith("_pct"):
            print(f"  {key}: {value:.2f}%")
        elif "latency" in key:
            print(f"  {key}: {value:.4f}")
        elif key == "max_abs_diff":
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value:.0f}")
