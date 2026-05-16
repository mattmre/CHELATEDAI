from __future__ import annotations

from dataclasses import dataclass

from cpu_inference_benchmark import benchmark_cpu_backends
from integrated_runtime_benchmark import benchmark_integrated_repo_runtime
from integrated_runtime_compression_benchmark import benchmark_integrated_runtime_compression
from repo_graph_memory_benchmark import benchmark_repo_graph_memory
from repo_graph_memory_compression_benchmark import benchmark_repo_graph_memory_compression
from sparse_inference_benchmark import benchmark_sparse_inference
from storage_substrate_benchmark import benchmark_storage_substrate
from _experimental import mark_experimental

# Research-stage / POC module. No production code path in this repo consumes it.
# EXPERIMENTAL is read by _experimental.mark_experimental — flipping it to
# False suppresses the import-time warning; a non-bool raises TypeError.
EXPERIMENTAL = True
mark_experimental(__name__, EXPERIMENTAL)


@dataclass(frozen=True)
class PromotionThresholds:
    min_storage_read_reduction_pct: float = 50.0
    min_cpu_speedup_vs_float32: float = 1.0
    min_sparse_stream_reduction_pct: float = 25.0
    min_repo_top3_hit_rate: float = 0.5
    min_integrated_top3_hit_rate: float = 0.5
    max_integrated_total_latency_ms: float = 10.0
    min_compression_mapped_reduction_ratio: float = 0.5
    max_compression_top3_drop: float = 0.0


def evaluate_system_promotion(
    thresholds: PromotionThresholds | None = None,
) -> dict[str, object]:
    active_thresholds = thresholds or PromotionThresholds()

    storage = benchmark_storage_substrate()
    cpu = benchmark_cpu_backends()
    sparse = benchmark_sparse_inference()
    memory = benchmark_repo_graph_memory()
    integrated = benchmark_integrated_repo_runtime()
    memory_compression = benchmark_repo_graph_memory_compression()
    integrated_compression = benchmark_integrated_runtime_compression()

    checks = {
        "storage_reduction_ok": storage["read_reduction_pct"] >= active_thresholds.min_storage_read_reduction_pct,
        "cpu_baseline_ok": cpu["prequantized_int8_vs_float32_speedup"] >= active_thresholds.min_cpu_speedup_vs_float32,
        "sparse_runtime_ok": sparse["streamed_byte_reduction_pct"] >= active_thresholds.min_sparse_stream_reduction_pct,
        "repo_memory_ok": memory["top3_hit_rate"] >= active_thresholds.min_repo_top3_hit_rate,
        "integrated_runtime_ok": (
            integrated["top3_hit_rate"] >= active_thresholds.min_integrated_top3_hit_rate
            and integrated["avg_total_latency_ms"] <= active_thresholds.max_integrated_total_latency_ms
        ),
        "memory_compression_ok": (
            memory_compression["mapped_byte_reduction_pct"] >= active_thresholds.min_compression_mapped_reduction_ratio
            and (memory_compression["float32_top3_hit_rate"] - memory_compression["int8_top3_hit_rate"])
            <= active_thresholds.max_compression_top3_drop
        ),
        "integrated_compression_ok": (
            integrated_compression["mapped_byte_reduction_pct"] >= active_thresholds.min_compression_mapped_reduction_ratio
            and (integrated_compression["float32_top3_hit_rate"] - integrated_compression["int8_top3_hit_rate"])
            <= active_thresholds.max_compression_top3_drop
        ),
    }

    promotion_questions = {
        "cpu_only_practical": bool(checks["cpu_baseline_ok"] and checks["integrated_runtime_ok"]),
        "disk_assistance_helpful": bool(checks["storage_reduction_ok"] and checks["integrated_compression_ok"]),
        "retrieval_memory_reduces_resident_burden": bool(checks["repo_memory_ok"] and checks["integrated_runtime_ok"]),
        "simpler_than_gpu_stack": False,
    }

    all_core_checks = all(
        checks[key]
        for key in (
            "storage_reduction_ok",
            "cpu_baseline_ok",
            "sparse_runtime_ok",
            "repo_memory_ok",
            "integrated_runtime_ok",
            "memory_compression_ok",
            "integrated_compression_ok",
        )
    )

    if all_core_checks:
        overall_recommendation = "promote_research_baseline"
    else:
        overall_recommendation = "defer_and_iterate"

    production_promotion = (
        overall_recommendation == "promote_research_baseline" and promotion_questions["simpler_than_gpu_stack"]
    )

    return {
        "thresholds": active_thresholds.__dict__,
        "metrics": {
            "storage": storage,
            "cpu": cpu,
            "sparse": sparse,
            "memory": memory,
            "integrated": integrated,
            "memory_compression": memory_compression,
            "integrated_compression": integrated_compression,
        },
        "checks": checks,
        "promotion_questions": promotion_questions,
        "overall_recommendation": overall_recommendation,
        "production_promotion": production_promotion,
    }


if __name__ == "__main__":
    evaluation = evaluate_system_promotion()
    print("Phase 7 System Evaluation")
    print(f"  overall_recommendation: {evaluation['overall_recommendation']}")
    print(f"  production_promotion: {evaluation['production_promotion']}")
    print("  checks:")
    for key, value in evaluation["checks"].items():
        print(f"    {key}: {value}")
