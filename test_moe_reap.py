import os
import sys
import tempfile
import unittest

import numpy as np

POC_DIR = os.path.join(os.path.dirname(__file__), "computational_storage_poc")
if POC_DIR not in sys.path:
    sys.path.insert(0, POC_DIR)

from block_graph import BLOCK_SIZE  # noqa: E402
from moe_reap import (  # noqa: E402
    DiskBackedMoEArtifact,
    reap_prune_experts,
    run_moe_artifact,
    write_moe_reap_artifact,
)
from moe_reap_benchmark import benchmark_moe_reap  # noqa: E402


class MoEReapTests(unittest.TestCase):
    def test_reap_pruning_keeps_high_scoring_experts(self):
        experts = [
            [np.ones((4, 4), dtype=np.float32) * scale, np.ones((4, 2), dtype=np.float32) * scale]
            for scale in (0.1, 0.2, 0.3, 0.4)
        ]
        kept_experts, scores = reap_prune_experts(experts, keep_fraction=0.5)

        self.assertEqual(kept_experts, [2, 3])
        self.assertEqual(len(scores), 4)

    def test_moe_artifact_preserves_expert_metadata_and_routes_active_experts(self):
        rng = np.random.default_rng(55)
        router_weights = np.zeros((16, 4), dtype=np.float32)
        router_weights[0, 3] = 4.0
        router_weights[0, 2] = 3.0
        experts = [
            [
                rng.normal(size=(16, 8)).astype(np.float32) * 0.1,
                rng.normal(size=(8, 4)).astype(np.float32) * 0.1,
            ]
            for _ in range(4)
        ]

        input_activations = np.zeros((1, BLOCK_SIZE), dtype=np.float16)
        input_activations[0, 0] = 1.0

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = os.path.join(temp_dir, "moe.csm")
            write_moe_reap_artifact(artifact_path, router_weights, experts, active_expert_ids=[2, 3])
            with DiskBackedMoEArtifact(artifact_path) as artifact:
                result = run_moe_artifact(artifact, input_activations, top_k=2)
                self.assertEqual([expert.expert_id for expert in artifact.experts if expert.active], [2, 3])

        self.assertEqual(result.active_expert_ids, [3, 2])
        self.assertEqual(result.experts_evaluated, 2)

    def test_moe_reap_benchmark_reports_pruning_reduction(self):
        metrics = benchmark_moe_reap()
        self.assertLess(metrics["pruned_artifact_bytes"], metrics["full_artifact_bytes"])
        self.assertLess(metrics["pruned_bytes_read"], metrics["full_bytes_read"])
        self.assertGreater(metrics["artifact_reduction_pct"], 0.0)
        self.assertGreater(metrics["read_reduction_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
