from __future__ import annotations

from pathlib import Path

from integrated_repo_runtime import IntegratedRepoRuntime
from repo_graph_memory import FLOAT32_EMBEDDING_DTYPE, INT8_EMBEDDING_DTYPE
from retrieval_eval_suite import get_retrieval_eval_cases


def _run_integrated_suite(embedding_storage_dtype: str) -> dict[str, float]:
    repo_root = Path(__file__).resolve().parent
    benchmark_queries = get_retrieval_eval_cases()

    top1_hits = 0
    top3_hits = 0
    total_retrieval_latency_ms = 0.0
    total_inference_latency_ms = 0.0
    total_latency_ms = 0.0
    total_bytes_read = 0
    mapped_bytes = 0
    peak_python_heap_kb = 0.0

    with IntegratedRepoRuntime(repo_root, top_k=3, embedding_storage_dtype=embedding_storage_dtype) as runtime:
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
            mapped_bytes = response.metrics.mapped_bytes
            peak_python_heap_kb = max(peak_python_heap_kb, response.metrics.python_heap_peak_kb)

    query_count = len(benchmark_queries)
    return {
        "avg_retrieval_latency_ms": total_retrieval_latency_ms / query_count,
        "avg_inference_latency_ms": total_inference_latency_ms / query_count,
        "avg_total_latency_ms": total_latency_ms / query_count,
        "avg_bytes_read_per_query": total_bytes_read / query_count,
        "mapped_bytes": float(mapped_bytes),
        "peak_python_heap_kb": peak_python_heap_kb,
        "top1_hit_rate": top1_hits / query_count,
        "top3_hit_rate": top3_hits / query_count,
    }


def benchmark_integrated_runtime_compression() -> dict[str, float]:
    float_metrics = _run_integrated_suite(FLOAT32_EMBEDDING_DTYPE)
    int8_metrics = _run_integrated_suite(INT8_EMBEDDING_DTYPE)

    return {
        "float32_mapped_bytes": float_metrics["mapped_bytes"],
        "int8_mapped_bytes": int8_metrics["mapped_bytes"],
        "mapped_byte_reduction_pct": 1.0 - (int8_metrics["mapped_bytes"] / max(float_metrics["mapped_bytes"], 1e-9)),
        "float32_avg_total_latency_ms": float_metrics["avg_total_latency_ms"],
        "int8_avg_total_latency_ms": int8_metrics["avg_total_latency_ms"],
        "float32_top1_hit_rate": float_metrics["top1_hit_rate"],
        "int8_top1_hit_rate": int8_metrics["top1_hit_rate"],
        "float32_top3_hit_rate": float_metrics["top3_hit_rate"],
        "int8_top3_hit_rate": int8_metrics["top3_hit_rate"],
        "float32_peak_python_heap_kb": float_metrics["peak_python_heap_kb"],
        "int8_peak_python_heap_kb": int8_metrics["peak_python_heap_kb"],
    }


if __name__ == "__main__":
    metrics = benchmark_integrated_runtime_compression()
    print("Integrated Runtime Compression Benchmark")
    for key, value in metrics.items():
        if key.endswith("_pct") or key.endswith("_rate"):
            print(f"  {key}: {value:.2%}")
        elif "latency" in key:
            print(f"  {key}: {value:.4f}")
        elif key.endswith("_kb"):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value:.0f}")
