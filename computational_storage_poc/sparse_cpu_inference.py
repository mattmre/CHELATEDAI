from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from block_graph import apply_hidden_activation
from cpu_backends import CPUInferenceBackend, NumpyInt8DynamicBackend
from packed_graph import DiskBackedPackedGraph, INT8_STORAGE_DTYPE
from _experimental import mark_experimental

# Research-stage / POC module. No production code path in this repo consumes it.
# EXPERIMENTAL is read by _experimental.mark_experimental — flipping it to
# False suppresses the import-time warning; a non-bool raises TypeError.
EXPERIMENTAL = True
mark_experimental(__name__, EXPERIMENTAL)


@dataclass(frozen=True)
class SparseInferenceConfig:
    chunk_rows: int = 16
    activation_epsilon: float = 1e-6
    stream_from_block: int = 1


@dataclass(frozen=True)
class SparsePackedCPUInferenceResult:
    output: np.ndarray
    backend_name: str
    blocks_processed: int
    bytes_read: int
    dense_equivalent_bytes: int
    chunks_loaded: int
    cache_hits: int
    cache_misses: int
    active_rows: int


class SparseChunkCache:
    def __init__(self, max_cached_chunks: int = 8):
        self.max_cached_chunks = max_cached_chunks
        self._entries: OrderedDict[tuple[int, int, int], Any] = OrderedDict()

    def get(self, key: tuple[int, int, int]) -> Any | None:
        value = self._entries.get(key)
        if value is None:
            return None
        self._entries.move_to_end(key)
        return value

    def put(self, key: tuple[int, int, int], value: Any) -> None:
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_cached_chunks:
            self._entries.popitem(last=False)


def _active_chunk_ranges(activations: np.ndarray, rows: int, chunk_rows: int, activation_epsilon: float) -> list[tuple[int, int]]:
    active_mask = np.any(np.abs(activations[:, :rows]) > activation_epsilon, axis=0)
    active_indices = np.flatnonzero(active_mask)
    if active_indices.size == 0:
        return []

    chunk_starts = sorted({int(index // chunk_rows) * chunk_rows for index in active_indices})
    return [(chunk_start, min(chunk_start + chunk_rows, rows)) for chunk_start in chunk_starts]


def _run_sparse_block(
    graph: DiskBackedPackedGraph,
    block_index: int,
    activations: np.ndarray,
    backend: CPUInferenceBackend,
    config: SparseInferenceConfig,
    cache: SparseChunkCache | None,
) -> tuple[np.ndarray, int, int, int, int, int]:
    metadata = graph.blocks[block_index]
    chunk_ranges = _active_chunk_ranges(activations, metadata.rows, config.chunk_rows, config.activation_epsilon)
    dense_equivalent_bytes = metadata.matrix_nbytes

    if not chunk_ranges:
        return (
            np.zeros((activations.shape[0], metadata.cols), dtype=np.float32),
            0,
            dense_equivalent_bytes,
            0,
            0,
            0,
        )

    if len(chunk_ranges) * config.chunk_rows >= metadata.rows:
        if metadata.storage_dtype == INT8_STORAGE_DTYPE:
            block = graph.read_quantized_block(block_index)
            backend_result = backend.matmul_quantized_weights(activations[:, : metadata.rows], block.quantized_matrix, block.scale)
        else:
            block = graph.read_block(block_index)
            backend_result = backend.matmul(activations[:, : metadata.rows], block.matrix)
        return backend_result.output, metadata.matrix_nbytes, dense_equivalent_bytes, 1, 0, len(chunk_ranges)

    output = np.zeros((activations.shape[0], metadata.cols), dtype=np.float32)
    bytes_read = 0
    chunks_loaded = 0
    cache_hits = 0
    cache_misses = 0

    for row_start, row_end in chunk_ranges:
        cache_key = (block_index, row_start, row_end)
        cached_chunk = cache.get(cache_key) if cache is not None else None

        if cached_chunk is None:
            if metadata.storage_dtype == INT8_STORAGE_DTYPE:
                chunk = graph.read_quantized_block_chunk(block_index, row_start, row_end)
            else:
                chunk = graph.read_block_chunk(block_index, row_start, row_end)
            if cache is not None:
                cache.put(cache_key, chunk)
            bytes_read += chunk.bytes_read
            chunks_loaded += 1
            cache_misses += 1
        else:
            chunk = cached_chunk
            cache_hits += 1

        activation_chunk = activations[:, row_start:row_end]
        if metadata.storage_dtype == INT8_STORAGE_DTYPE:
            backend_result = backend.matmul_quantized_weights(activation_chunk, chunk.quantized_matrix, chunk.scale)
        else:
            backend_result = backend.matmul(activation_chunk, chunk.matrix)
        output += backend_result.output

    return output, bytes_read, dense_equivalent_bytes, chunks_loaded, cache_hits, cache_misses


def run_sparse_packed_graph_with_backend(
    graph: DiskBackedPackedGraph,
    input_activations: np.ndarray,
    backend: CPUInferenceBackend | None = None,
    *,
    config: SparseInferenceConfig | None = None,
    cache: SparseChunkCache | None = None,
    trigger_block: int = 0,
    hidden_activation: str | None = "relu",
) -> SparsePackedCPUInferenceResult:
    inference_backend = backend or NumpyInt8DynamicBackend()
    sparse_config = config or SparseInferenceConfig()
    current_block = int(trigger_block)
    current_activations = input_activations.astype(np.float32)
    blocks_processed = 0
    bytes_read = 0
    dense_equivalent_bytes = 0
    chunks_loaded = 0
    cache_hits = 0
    cache_misses = 0
    active_rows = 0

    while True:
        metadata = graph.blocks[current_block]
        should_stream = current_block >= sparse_config.stream_from_block

        if should_stream:
            current_activations, block_bytes, dense_block_bytes, block_chunks_loaded, block_cache_hits, block_cache_misses = (
                _run_sparse_block(
                    graph,
                    current_block,
                    current_activations,
                    inference_backend,
                    sparse_config,
                    cache,
                )
            )
            active_rows += int(np.count_nonzero(np.any(np.abs(current_activations) > sparse_config.activation_epsilon, axis=0)))
        else:
            dense_equivalent_bytes += metadata.matrix_nbytes
            if metadata.storage_dtype == INT8_STORAGE_DTYPE:
                block = graph.read_quantized_block(current_block)
                backend_result = inference_backend.matmul_quantized_weights(
                    current_activations[:, : metadata.rows],
                    block.quantized_matrix,
                    block.scale,
                )
            else:
                block = graph.read_block(current_block)
                backend_result = inference_backend.matmul(current_activations[:, : metadata.rows], block.matrix)
            current_activations = backend_result.output
            block_bytes = metadata.matrix_nbytes
            dense_block_bytes = metadata.matrix_nbytes
            block_chunks_loaded = 1
            block_cache_hits = 0
            block_cache_misses = 0

        bytes_read += block_bytes
        dense_equivalent_bytes += dense_block_bytes if should_stream else 0
        chunks_loaded += block_chunks_loaded
        cache_hits += block_cache_hits
        cache_misses += block_cache_misses
        blocks_processed += 1

        if metadata.next_block is None:
            break

        current_activations = apply_hidden_activation(current_activations, hidden_activation)
        current_block = metadata.next_block

    return SparsePackedCPUInferenceResult(
        output=current_activations,
        backend_name=inference_backend.name,
        blocks_processed=blocks_processed,
        bytes_read=bytes_read,
        dense_equivalent_bytes=dense_equivalent_bytes,
        chunks_loaded=chunks_loaded,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        active_rows=active_rows,
    )
