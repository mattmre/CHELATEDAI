from __future__ import annotations

import tempfile
import time
from pathlib import Path

from repo_graph_memory import DiskBackedRepoGraphMemory, FLOAT32_EMBEDDING_DTYPE, ingest_repo_graph_memory
from retrieval_eval_suite import get_retrieval_eval_cases


def benchmark_repo_graph_memory(embedding_storage_dtype: str = FLOAT32_EMBEDDING_DTYPE) -> dict[str, float]:
    repo_root = Path(__file__).resolve().parent
    benchmark_queries = get_retrieval_eval_cases()

    with tempfile.TemporaryDirectory() as temp_dir:
        ingest_start = time.perf_counter()
        ingest_metrics = ingest_repo_graph_memory(
            repo_root,
            temp_dir,
            exclude_substrings=("benchmark", "test_", "README", "__pycache__"),
            embedding_storage_dtype=embedding_storage_dtype,
        )
        ingest_elapsed_ms = (time.perf_counter() - ingest_start) * 1000.0

        total_query_latency_ms = 0.0
        top1_hits = 0
        top3_hits = 0
        mapped_bytes = 0

        with DiskBackedRepoGraphMemory(temp_dir) as memory:
            mapped_bytes = memory.mapped_bytes
            for benchmark_case in benchmark_queries:
                query_start = time.perf_counter()
                results = memory.query(benchmark_case.query, top_k=3)
                total_query_latency_ms += (time.perf_counter() - query_start) * 1000.0
                returned_paths = [result.path for result in results]
                if returned_paths and returned_paths[0].endswith(benchmark_case.expected_path_suffix):
                    top1_hits += 1
                if any(path.endswith(benchmark_case.expected_path_suffix) for path in returned_paths):
                    top3_hits += 1

    return {
        "node_count": float(ingest_metrics["node_count"]),
        "edge_count": float(ingest_metrics["edge_count"]),
        "mapped_bytes": float(mapped_bytes),
        "ingest_latency_ms": ingest_elapsed_ms,
        "query_count": float(len(benchmark_queries)),
        "avg_query_latency_ms": total_query_latency_ms / len(benchmark_queries),
        "top1_hit_rate": top1_hits / len(benchmark_queries),
        "top3_hit_rate": top3_hits / len(benchmark_queries),
    }


if __name__ == "__main__":
    metrics = benchmark_repo_graph_memory()
    print("Repo Graph Memory Benchmark")
    for key, value in metrics.items():
        if key.endswith("_rate"):
            print(f"  {key}: {value:.2%}")
        elif "latency" in key:
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value:.0f}")
