from __future__ import annotations

from repo_graph_memory import FLOAT32_EMBEDDING_DTYPE, INT8_EMBEDDING_DTYPE
from repo_graph_memory_benchmark import benchmark_repo_graph_memory


def benchmark_repo_graph_memory_compression() -> dict[str, float]:
    float_metrics = benchmark_repo_graph_memory(FLOAT32_EMBEDDING_DTYPE)
    int8_metrics = benchmark_repo_graph_memory(INT8_EMBEDDING_DTYPE)

    return {
        "float32_mapped_bytes": float_metrics["mapped_bytes"],
        "int8_mapped_bytes": int8_metrics["mapped_bytes"],
        "mapped_byte_reduction_pct": 1.0 - (int8_metrics["mapped_bytes"] / max(float_metrics["mapped_bytes"], 1e-9)),
        "float32_avg_query_latency_ms": float_metrics["avg_query_latency_ms"],
        "int8_avg_query_latency_ms": int8_metrics["avg_query_latency_ms"],
        "float32_top1_hit_rate": float_metrics["top1_hit_rate"],
        "int8_top1_hit_rate": int8_metrics["top1_hit_rate"],
        "float32_top3_hit_rate": float_metrics["top3_hit_rate"],
        "int8_top3_hit_rate": int8_metrics["top3_hit_rate"],
    }


if __name__ == "__main__":
    metrics = benchmark_repo_graph_memory_compression()
    print("Repo Graph Memory Compression Benchmark")
    for key, value in metrics.items():
        if key.endswith("_pct") or key.endswith("_rate"):
            print(f"  {key}: {value:.2%}")
        elif "latency" in key:
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value:.0f}")
