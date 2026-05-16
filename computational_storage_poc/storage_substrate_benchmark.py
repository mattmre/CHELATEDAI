from __future__ import annotations

import os
import tempfile

import numpy as np

from block_graph import BLOCK_SIZE, TOTAL_BLOCK_BYTES, build_graph_payload, run_block_graph
from packed_graph import DiskBackedPackedGraph, write_packed_graph_artifact, run_packed_graph


def benchmark_storage_substrate() -> dict[str, float]:
    rng = np.random.default_rng(42)
    matrices = [
        rng.normal(size=(256, 128)).astype(np.float32) * 0.1,
        rng.normal(size=(128, 64)).astype(np.float32) * 0.1,
        rng.normal(size=(64, 10)).astype(np.float32) * 0.1,
    ]
    legacy_payload = build_graph_payload(matrices)

    input_activations = np.zeros((1, BLOCK_SIZE), dtype=np.float16)
    input_activations[0, :256] = rng.normal(size=256).astype(np.float32) * 0.1
    legacy_output, legacy_blocks = run_block_graph(legacy_payload, input_activations, hidden_activation="relu")

    with tempfile.TemporaryDirectory() as temp_dir:
        packed_path = os.path.join(temp_dir, "graph.cspg")
        write_packed_graph_artifact(packed_path, matrices)
        with DiskBackedPackedGraph(packed_path) as graph:
            packed_output, packed_blocks, packed_bytes_read = run_packed_graph(
                graph,
                input_activations,
                hidden_activation="relu",
            )
            packed_artifact_bytes = graph.artifact_size_bytes

    legacy_bytes_read = legacy_blocks * TOTAL_BLOCK_BYTES
    max_abs_diff = float(np.max(np.abs(legacy_output[:, :10] - packed_output[:, :10])))

    return {
        "legacy_artifact_bytes": float(len(legacy_payload)),
        "packed_artifact_bytes": float(packed_artifact_bytes),
        "legacy_bytes_read": float(legacy_bytes_read),
        "packed_bytes_read": float(packed_bytes_read),
        "artifact_size_reduction_pct": 100.0 * (1.0 - (packed_artifact_bytes / len(legacy_payload))),
        "read_reduction_pct": 100.0 * (1.0 - (packed_bytes_read / legacy_bytes_read)),
        "max_abs_diff": max_abs_diff,
        "block_count": float(packed_blocks),
    }


if __name__ == "__main__":
    metrics = benchmark_storage_substrate()
    print("Storage Substrate Benchmark")
    for key, value in metrics.items():
        if key.endswith("_pct"):
            print(f"  {key}: {value:.2f}%")
        elif key == "max_abs_diff":
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value:.0f}")
