from __future__ import annotations

import time
from pathlib import Path

from integrated_repo_runtime import IntegratedRepoRuntime
from retrieval_eval_suite import get_retrieval_eval_cases


def benchmark_integrated_repo_runtime() -> dict[str, float]:
    repo_root = Path(__file__).resolve().parent
    benchmark_queries = get_retrieval_eval_cases()

    top1_hits = 0
    top3_hits = 0
    total_retrieval_latency_ms = 0.0
    total_inference_latency_ms = 0.0
    total_latency_ms = 0.0
    total_bytes_read = 0
    total_mapped_bytes = 0
    total_peak_heap_kb = 0.0

    benchmark_start = time.perf_counter()
    with IntegratedRepoRuntime(repo_root, top_k=3) as runtime:
        for benchmark_case in benchmark_queries:
            response = runtime.answer_query(benchmark_case.query)
            returned_paths = [candidate.path for candidate in response.candidates]
            if returned_paths and returned_paths[0].endswith(benchmark_case.expected_path_suffix):
                top1_hits += 1
            if any(path.endswith(benchmark_case.expected_path_suffix) for path in returned_paths):
                top3_hits += 1

            total_retrieval_latency_ms += response.metrics.retrieval_latency_ms
            total_inference_latency_ms += response.metrics.inference_latency_ms
            total_latency_ms += response.metrics.total_latency_ms
            total_bytes_read += response.metrics.bytes_read
            total_mapped_bytes = response.metrics.mapped_bytes
            total_peak_heap_kb = max(total_peak_heap_kb, response.metrics.python_heap_peak_kb)
    benchmark_elapsed_ms = (time.perf_counter() - benchmark_start) * 1000.0

    query_count = len(benchmark_queries)
    return {
        "query_count": float(query_count),
        "avg_retrieval_latency_ms": total_retrieval_latency_ms / query_count,
        "avg_inference_latency_ms": total_inference_latency_ms / query_count,
        "avg_total_latency_ms": total_latency_ms / query_count,
        "queries_per_second": (1000.0 * query_count) / max(benchmark_elapsed_ms, 1e-9),
        "avg_bytes_read_per_query": total_bytes_read / query_count,
        "mapped_bytes": float(total_mapped_bytes),
        "peak_python_heap_kb": total_peak_heap_kb,
        "top1_hit_rate": top1_hits / query_count,
        "top3_hit_rate": top3_hits / query_count,
    }


if __name__ == "__main__":
    metrics = benchmark_integrated_repo_runtime()
    print("Integrated Runtime Benchmark")
    for key, value in metrics.items():
        if key.endswith("_rate"):
            print(f"  {key}: {value:.2%}")
        elif "latency" in key:
            print(f"  {key}: {value:.4f}")
        elif key.endswith("_kb"):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value:.0f}")
