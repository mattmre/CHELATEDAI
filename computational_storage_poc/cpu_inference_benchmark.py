from __future__ import annotations

import os
import tempfile
import time
from statistics import median

import numpy as np

from block_graph import BLOCK_SIZE
from cpu_backends import NumpyFloat32Backend, NumpyInt8DynamicBackend
from packed_cpu_inference import run_packed_graph_with_backend
from packed_graph import DiskBackedPackedGraph, INT8_STORAGE_DTYPE, write_packed_graph_artifact


def benchmark_cpu_backends(repetitions: int = 100, trials: int = 5, warmup_runs: int = 5) -> dict[str, float]:
    rng = np.random.default_rng(123)
    matrices = [
        rng.normal(size=(256, 128)).astype(np.float32) * 0.1,
        rng.normal(size=(128, 64)).astype(np.float32) * 0.1,
        rng.normal(size=(64, 10)).astype(np.float32) * 0.1,
    ]

    input_activations = np.zeros((1, BLOCK_SIZE), dtype=np.float16)
    input_activations[0, :256] = rng.normal(size=256).astype(np.float32) * 0.1

    float_backend = NumpyFloat32Backend()
    int8_backend = NumpyInt8DynamicBackend()
    timings_ms: dict[str, float] = {}
    outputs: dict[str, np.ndarray] = {}
    bytes_read_by_path: dict[str, float] = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        float_artifact_path = os.path.join(temp_dir, "graph_fp16.cspg")
        int8_artifact_path = os.path.join(temp_dir, "graph_int8.cspg")
        write_packed_graph_artifact(float_artifact_path, matrices)
        write_packed_graph_artifact(int8_artifact_path, matrices, storage_dtype=INT8_STORAGE_DTYPE)

        benchmark_paths = [
            ("float32", float_artifact_path, float_backend),
            ("dynamic_int8", float_artifact_path, int8_backend),
            ("prequantized_int8", int8_artifact_path, int8_backend),
        ]

        for label, artifact_path, backend in benchmark_paths:
            with DiskBackedPackedGraph(artifact_path) as graph:
                last_result = None
                for _ in range(warmup_runs):
                    last_result = run_packed_graph_with_backend(graph, input_activations, backend)

                trial_timings_ms: list[float] = []
                for _ in range(trials):
                    start = time.perf_counter()
                    for _ in range(repetitions):
                        last_result = run_packed_graph_with_backend(graph, input_activations, backend)
                    elapsed = time.perf_counter() - start
                    trial_timings_ms.append((elapsed / repetitions) * 1000.0)
                assert last_result is not None
                timings_ms[label] = float(median(trial_timings_ms))
                outputs[label] = last_result.output
                bytes_read_by_path[label] = float(last_result.bytes_read)

    max_abs_diff = float(
        np.max(np.abs(outputs["float32"][:, :10] - outputs["prequantized_int8"][:, :10]))
    )
    dynamic_int8_max_abs_diff = float(
        np.max(np.abs(outputs["float32"][:, :10] - outputs["dynamic_int8"][:, :10]))
    )

    return {
        "float32_latency_ms": timings_ms["float32"],
        "dynamic_int8_latency_ms": timings_ms["dynamic_int8"],
        "prequantized_int8_latency_ms": timings_ms["prequantized_int8"],
        "dynamic_int8_vs_float32_speedup": timings_ms["float32"] / timings_ms["dynamic_int8"],
        "prequantized_int8_vs_dynamic_int8_speedup": timings_ms["dynamic_int8"] / timings_ms["prequantized_int8"],
        "prequantized_int8_vs_float32_speedup": timings_ms["float32"] / timings_ms["prequantized_int8"],
        "dynamic_int8_max_abs_diff": dynamic_int8_max_abs_diff,
        "prequantized_int8_max_abs_diff": max_abs_diff,
        "float32_bytes_read": bytes_read_by_path["float32"],
        "dynamic_int8_bytes_read": bytes_read_by_path["dynamic_int8"],
        "prequantized_int8_bytes_read": bytes_read_by_path["prequantized_int8"],
    }


if __name__ == "__main__":
    metrics = benchmark_cpu_backends()
    print("CPU Inference Benchmark")
    print(f"  float32_latency_ms: {metrics['float32_latency_ms']:.4f}")
    print(f"  dynamic_int8_latency_ms: {metrics['dynamic_int8_latency_ms']:.4f}")
    print(f"  prequantized_int8_latency_ms: {metrics['prequantized_int8_latency_ms']:.4f}")
    print(f"  dynamic_int8_vs_float32_speedup: {metrics['dynamic_int8_vs_float32_speedup']:.4f}")
    print(
        "  prequantized_int8_vs_dynamic_int8_speedup: "
        f"{metrics['prequantized_int8_vs_dynamic_int8_speedup']:.4f}"
    )
    print(f"  prequantized_int8_vs_float32_speedup: {metrics['prequantized_int8_vs_float32_speedup']:.4f}")
    print(f"  dynamic_int8_max_abs_diff: {metrics['dynamic_int8_max_abs_diff']:.6f}")
    print(f"  prequantized_int8_max_abs_diff: {metrics['prequantized_int8_max_abs_diff']:.6f}")
    print(f"  float32_bytes_read: {metrics['float32_bytes_read']:.0f}")
    print(f"  prequantized_int8_bytes_read: {metrics['prequantized_int8_bytes_read']:.0f}")
