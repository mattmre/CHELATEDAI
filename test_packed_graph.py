import os
import sys
import tempfile
import unittest

import numpy as np

POC_DIR = os.path.join(os.path.dirname(__file__), "computational_storage_poc")
if POC_DIR not in sys.path:
    sys.path.insert(0, POC_DIR)

from block_graph import BLOCK_SIZE, TOTAL_BLOCK_BYTES, build_graph_payload, run_block_graph  # noqa: E402
from packed_graph import (  # noqa: E402
    DiskBackedPackedGraph,
    INT8_STORAGE_DTYPE,
    build_packed_graph_artifact,
    run_packed_graph,
    write_packed_graph_artifact,
)
from storage_substrate_benchmark import benchmark_storage_substrate  # noqa: E402
from train_and_compile import TinyDigitClassifier, compile_model  # noqa: E402


class PackedGraphTests(unittest.TestCase):
    def test_packed_graph_round_trip_matches_legacy_block_graph(self):
        rng = np.random.default_rng(7)
        matrices = [
            rng.normal(size=(16, 8)).astype(np.float32),
            rng.normal(size=(8, 4)).astype(np.float32),
        ]
        legacy_payload = build_graph_payload(matrices)

        input_activations = np.zeros((1, BLOCK_SIZE), dtype=np.float16)
        input_activations[0, :16] = rng.normal(size=16).astype(np.float32)
        legacy_output, legacy_blocks = run_block_graph(legacy_payload, input_activations, hidden_activation="relu")

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = os.path.join(temp_dir, "graph.cspg")
            write_packed_graph_artifact(artifact_path, matrices)
            with DiskBackedPackedGraph(artifact_path) as graph:
                packed_output, packed_blocks, packed_bytes_read = run_packed_graph(
                    graph,
                    input_activations,
                    hidden_activation="relu",
                )

        np.testing.assert_allclose(packed_output[:, :4], legacy_output[:, :4], rtol=1e-4, atol=1e-4)
        self.assertEqual(packed_blocks, legacy_blocks)
        self.assertLess(packed_bytes_read, legacy_blocks * TOTAL_BLOCK_BYTES)

    def test_packed_artifact_is_smaller_than_legacy_dense_payload(self):
        rng = np.random.default_rng(11)
        matrices = [
            rng.normal(size=(32, 12)).astype(np.float32),
            rng.normal(size=(12, 6)).astype(np.float32),
        ]
        packed_payload = build_packed_graph_artifact(matrices)
        legacy_payload = build_graph_payload(matrices)
        self.assertLess(len(packed_payload), len(legacy_payload))

    def test_int8_packed_artifact_is_smaller_than_float16_packed_artifact(self):
        rng = np.random.default_rng(12)
        matrices = [
            rng.normal(size=(32, 12)).astype(np.float32),
            rng.normal(size=(12, 6)).astype(np.float32),
        ]
        float16_payload = build_packed_graph_artifact(matrices)
        int8_payload = build_packed_graph_artifact(matrices, storage_dtype=INT8_STORAGE_DTYPE)
        self.assertLess(len(int8_payload), len(float16_payload))

    def test_storage_substrate_benchmark_reports_read_reduction_and_parity(self):
        metrics = benchmark_storage_substrate()
        self.assertLess(metrics["packed_artifact_bytes"], metrics["legacy_artifact_bytes"])
        self.assertLess(metrics["packed_bytes_read"], metrics["legacy_bytes_read"])
        self.assertLess(metrics["max_abs_diff"], 1e-4)

    def test_train_and_compile_supports_packed_artifacts(self):
        model = TinyDigitClassifier()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "real_model.cspg")
            metrics = compile_model(model, output_path, artifact_format="packed")
            with DiskBackedPackedGraph(output_path) as graph:
                self.assertEqual(graph.block_count, 2)
                self.assertEqual(metrics["artifact_format"], "packed")

    def test_train_and_compile_supports_packed_int8_artifacts(self):
        model = TinyDigitClassifier()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "real_model_int8.cspg")
            metrics = compile_model(model, output_path, artifact_format="packed_int8")
            with DiskBackedPackedGraph(output_path) as graph:
                self.assertEqual(graph.block_count, 2)
                self.assertEqual(metrics["artifact_format"], "packed_int8")
                self.assertTrue(all(block.storage_dtype == INT8_STORAGE_DTYPE for block in graph.blocks))


if __name__ == "__main__":
    unittest.main()
