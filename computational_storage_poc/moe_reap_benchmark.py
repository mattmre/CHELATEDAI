from __future__ import annotations

import os
import tempfile

import numpy as np

from block_graph import BLOCK_SIZE
from moe_reap import DiskBackedMoEArtifact, reap_prune_experts, run_moe_artifact, write_moe_reap_artifact


def benchmark_moe_reap() -> dict[str, float]:
    rng = np.random.default_rng(314)
    router_weights = rng.normal(size=(16, 4)).astype(np.float32) * 0.1
    expert_matrices = [
        [
            rng.normal(size=(16, 8)).astype(np.float32) * (0.2 + (0.05 * expert_id)),
            rng.normal(size=(8, 4)).astype(np.float32) * (0.2 + (0.05 * expert_id)),
        ]
        for expert_id in range(4)
    ]

    kept_experts, scores = reap_prune_experts(expert_matrices, keep_fraction=0.5)
    input_activations = np.zeros((1, BLOCK_SIZE), dtype=np.float16)
    input_activations[0, :16] = rng.normal(size=16).astype(np.float32) * 0.1
    input_activations[0, 10] = 2.5

    with tempfile.TemporaryDirectory() as temp_dir:
        full_path = os.path.join(temp_dir, "moe_full.csm")
        pruned_path = os.path.join(temp_dir, "moe_pruned.csm")
        full_metrics = write_moe_reap_artifact(full_path, router_weights, expert_matrices, expert_scores=scores)
        pruned_metrics = write_moe_reap_artifact(
            pruned_path,
            router_weights,
            expert_matrices,
            expert_scores=scores,
            active_expert_ids=kept_experts,
        )
        with DiskBackedMoEArtifact(full_path) as full_artifact:
            full_result = run_moe_artifact(full_artifact, input_activations, top_k=4)
        with DiskBackedMoEArtifact(pruned_path) as pruned_artifact:
            pruned_result = run_moe_artifact(pruned_artifact, input_activations, top_k=4)

    return {
        "full_artifact_bytes": float(full_metrics["artifact_bytes"]),
        "pruned_artifact_bytes": float(pruned_metrics["artifact_bytes"]),
        "artifact_reduction_pct": 100.0
        * (1.0 - (pruned_metrics["artifact_bytes"] / max(full_metrics["artifact_bytes"], 1e-9))),
        "full_bytes_read": float(full_result.bytes_read),
        "pruned_bytes_read": float(pruned_result.bytes_read),
        "read_reduction_pct": 100.0
        * (1.0 - (pruned_result.bytes_read / max(full_result.bytes_read, 1e-9))),
        "full_experts_evaluated": float(full_result.experts_evaluated),
        "pruned_experts_evaluated": float(pruned_result.experts_evaluated),
        "active_experts_after_prune": float(len(kept_experts)),
        "output_shift_l2": float(np.linalg.norm(full_result.output - pruned_result.output)),
    }


if __name__ == "__main__":
    metrics = benchmark_moe_reap()
    print("MoE REAP Benchmark")
    for key, value in metrics.items():
        if key.endswith("_pct"):
            print(f"  {key}: {value:.2f}%")
        elif key == "output_shift_l2":
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value:.0f}")
