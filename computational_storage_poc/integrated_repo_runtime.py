from __future__ import annotations

import tempfile
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from block_graph import BLOCK_SIZE
from cpu_backends import NumpyInt8DynamicBackend
from packed_graph import DiskBackedPackedGraph, INT8_STORAGE_DTYPE, write_packed_graph_artifact
from repo_graph_memory import (
    DiskBackedRepoGraphMemory,
    FLOAT32_EMBEDDING_DTYPE,
    QueryResult,
    ingest_repo_graph_memory,
)
from sparse_cpu_inference import SparseChunkCache, SparseInferenceConfig, run_sparse_packed_graph_with_backend
from _experimental import mark_experimental

# Research-stage / POC module. No production code path in this repo consumes it.
# EXPERIMENTAL is read by _experimental.mark_experimental — flipping it to
# False suppresses the import-time warning; a non-bool raises TypeError.
EXPERIMENTAL = True
mark_experimental(__name__, EXPERIMENTAL)

RERANK_FEATURE_DIM = 8
RERANK_FUSION_WEIGHT = 0.15


@dataclass(frozen=True)
class RuntimeCandidate:
    path: str
    kind: str
    final_score: float
    retrieval_score: float
    rerank_score: float
    embedding_score: float
    lexical_score: float
    graph_score: float
    path_score: float


@dataclass(frozen=True)
class RuntimeMetrics:
    retrieval_latency_ms: float
    inference_latency_ms: float
    total_latency_ms: float
    bytes_read: int
    mapped_bytes: int
    python_heap_peak_kb: float
    queries_per_second: float


@dataclass(frozen=True)
class RuntimeResponse:
    query: str
    candidates: list[RuntimeCandidate]
    metrics: RuntimeMetrics


def build_repo_runtime_reranker_matrices() -> list[np.ndarray]:
    w1 = np.zeros((RERANK_FEATURE_DIM, 16), dtype=np.float32)
    for feature_index in range(RERANK_FEATURE_DIM):
        w1[feature_index, feature_index] = 1.0
        w1[feature_index, feature_index + 8] = 0.5

    w2 = np.zeros((16, 1), dtype=np.float32)
    base_weights = np.array([0.7, 0.5, 1.2, 0.8, 1.1, 0.4, 0.1, 0.3], dtype=np.float32)
    w2[:8, 0] = base_weights
    w2[8:, 0] = base_weights * 0.25
    return [w1, w2]


def _candidate_to_input(candidate: QueryResult) -> np.ndarray:
    activations = np.zeros((1, BLOCK_SIZE), dtype=np.float16)
    activations[0, :RERANK_FEATURE_DIM] = [
        float(candidate.score),
        float(candidate.embedding_score),
        float(candidate.lexical_score),
        float(candidate.graph_score),
        float(candidate.path_score),
        1.0 if candidate.kind == "file" else 0.0,
        1.0 if candidate.kind == "symbol" else 0.0,
        max(float(candidate.lexical_score), float(candidate.path_score)),
    ]
    return activations


class IntegratedRepoRuntime:
    def __init__(
        self,
        repo_root: str | Path,
        *,
        top_k: int = 5,
        embedding_storage_dtype: str = FLOAT32_EMBEDDING_DTYPE,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.top_k = top_k
        self.embedding_storage_dtype = embedding_storage_dtype
        self._temp_dir = tempfile.TemporaryDirectory()
        self.memory_dir = Path(self._temp_dir.name) / "memory"
        self.artifact_path = Path(self._temp_dir.name) / "repo_runtime_reranker.cspg"

        ingest_repo_graph_memory(
            self.repo_root,
            self.memory_dir,
            exclude_substrings=("benchmark", "test_", "README", "__pycache__"),
            embedding_storage_dtype=self.embedding_storage_dtype,
        )
        write_packed_graph_artifact(
            self.artifact_path,
            build_repo_runtime_reranker_matrices(),
            storage_dtype=INT8_STORAGE_DTYPE,
        )
        self.memory = DiskBackedRepoGraphMemory(self.memory_dir)
        self.graph = DiskBackedPackedGraph(self.artifact_path)
        self.backend = NumpyInt8DynamicBackend()
        self.sparse_config = SparseInferenceConfig(chunk_rows=4, stream_from_block=1)

    def close(self) -> None:
        memory = getattr(self, "memory", None)
        if memory is not None:
            memory.close()
            self.memory = None
        graph = getattr(self, "graph", None)
        if graph is not None:
            graph.close()
            self.graph = None
        temp_dir = getattr(self, "_temp_dir", None)
        if temp_dir is not None:
            temp_dir.cleanup()
            self._temp_dir = None

    def __enter__(self) -> "IntegratedRepoRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def answer_query(self, query_text: str, *, retrieval_depth: int = 8) -> RuntimeResponse:
        tracemalloc.start()
        total_start = time.perf_counter()

        retrieval_start = time.perf_counter()
        retrieved = self.memory.query(query_text, top_k=retrieval_depth)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

        inference_start = time.perf_counter()
        cache = SparseChunkCache(max_cached_chunks=8)
        candidates: list[RuntimeCandidate] = []
        total_bytes_read = 0

        for result in retrieved:
            activations = _candidate_to_input(result)
            inference = run_sparse_packed_graph_with_backend(
                self.graph,
                activations,
                self.backend,
                config=self.sparse_config,
                cache=cache,
            )
            rerank_score = float(inference.output[0, 0])
            final_score = float(result.score) + (RERANK_FUSION_WEIGHT * rerank_score)
            total_bytes_read += inference.bytes_read
            candidates.append(
                RuntimeCandidate(
                    path=result.path,
                    kind=result.kind,
                    final_score=final_score,
                    retrieval_score=result.score,
                    rerank_score=rerank_score,
                    embedding_score=result.embedding_score,
                    lexical_score=result.lexical_score,
                    graph_score=result.graph_score,
                    path_score=result.path_score,
                )
            )

        inference_latency_ms = (time.perf_counter() - inference_start) * 1000.0
        candidates.sort(key=lambda candidate: candidate.final_score, reverse=True)
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        _ = current_bytes
        tracemalloc.stop()
        total_latency_ms = (time.perf_counter() - total_start) * 1000.0

        mapped_bytes = int(self.memory.mapped_bytes) + int(self.graph.artifact_size_bytes)
        metrics = RuntimeMetrics(
            retrieval_latency_ms=retrieval_latency_ms,
            inference_latency_ms=inference_latency_ms,
            total_latency_ms=total_latency_ms,
            bytes_read=total_bytes_read,
            mapped_bytes=mapped_bytes,
            python_heap_peak_kb=peak_bytes / 1024.0,
            queries_per_second=1000.0 / max(total_latency_ms, 1e-9),
        )

        return RuntimeResponse(
            query=query_text,
            candidates=candidates[: self.top_k],
            metrics=metrics,
        )
