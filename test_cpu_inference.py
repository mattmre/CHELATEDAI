import os
import sys
import tempfile
import unittest

import numpy as np

POC_DIR = os.path.join(os.path.dirname(__file__), "computational_storage_poc")
if POC_DIR not in sys.path:
    sys.path.insert(0, POC_DIR)

from block_graph import BLOCK_SIZE  # noqa: E402
from cpu_backends import NumpyFloat32Backend, NumpyInt8DynamicBackend  # noqa: E402
from cpu_inference_benchmark import benchmark_cpu_backends  # noqa: E402
from packed_cpu_inference import run_packed_graph_with_backend  # noqa: E402
from packed_graph import (  # noqa: E402
    DiskBackedPackedGraph,
    INT8_STORAGE_DTYPE,
    write_packed_graph_artifact,
)


class CPUInferenceBackendTests(unittest.TestCase):
    def test_float32_and_dynamic_int8_backends_remain_close(self):
        rng = np.random.default_rng(21)
        matrices = [
            rng.normal(size=(16, 8)).astype(np.float32) * 0.1,
            rng.normal(size=(8, 4)).astype(np.float32) * 0.1,
        ]
        input_activations = np.zeros((1, BLOCK_SIZE), dtype=np.float16)
        input_activations[0, :16] = rng.normal(size=16).astype(np.float32) * 0.1

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = os.path.join(temp_dir, "graph.cspg")
            write_packed_graph_artifact(artifact_path, matrices)
            with DiskBackedPackedGraph(artifact_path) as graph:
                float_result = run_packed_graph_with_backend(graph, input_activations, NumpyFloat32Backend())
                int8_result = run_packed_graph_with_backend(graph, input_activations, NumpyInt8DynamicBackend())

        np.testing.assert_allclose(int8_result.output[:, :4], float_result.output[:, :4], rtol=0.15, atol=0.02)
        self.assertEqual(float_result.blocks_processed, int8_result.blocks_processed)
        self.assertEqual(float_result.bytes_read, int8_result.bytes_read)

    def test_float32_and_prequantized_int8_backends_remain_close(self):
        rng = np.random.default_rng(31)
        matrices = [
            rng.normal(size=(16, 8)).astype(np.float32) * 0.1,
            rng.normal(size=(8, 4)).astype(np.float32) * 0.1,
        ]
        input_activations = np.zeros((1, BLOCK_SIZE), dtype=np.float16)
        input_activations[0, :16] = rng.normal(size=16).astype(np.float32) * 0.1

        with tempfile.TemporaryDirectory() as temp_dir:
            float_artifact_path = os.path.join(temp_dir, "graph_fp16.cspg")
            int8_artifact_path = os.path.join(temp_dir, "graph_int8.cspg")
            write_packed_graph_artifact(float_artifact_path, matrices)
            write_packed_graph_artifact(int8_artifact_path, matrices, storage_dtype=INT8_STORAGE_DTYPE)
            with DiskBackedPackedGraph(float_artifact_path) as float_graph:
                float_result = run_packed_graph_with_backend(float_graph, input_activations, NumpyFloat32Backend())
            with DiskBackedPackedGraph(int8_artifact_path) as int8_graph:
                int8_result = run_packed_graph_with_backend(int8_graph, input_activations, NumpyInt8DynamicBackend())

        np.testing.assert_allclose(int8_result.output[:, :4], float_result.output[:, :4], rtol=0.15, atol=0.02)
        self.assertEqual(float_result.blocks_processed, int8_result.blocks_processed)
        self.assertLess(int8_result.bytes_read, float_result.bytes_read)

    def test_cpu_benchmark_returns_expected_metrics(self):
        metrics = benchmark_cpu_backends(repetitions=10)
        self.assertGreater(metrics["float32_latency_ms"], 0.0)
        self.assertGreater(metrics["dynamic_int8_latency_ms"], 0.0)
        self.assertGreater(metrics["prequantized_int8_latency_ms"], 0.0)
        self.assertGreater(metrics["float32_bytes_read"], 0.0)
        self.assertGreater(metrics["prequantized_int8_bytes_read"], 0.0)
        self.assertLess(metrics["prequantized_int8_max_abs_diff"], 0.1)
        self.assertLess(metrics["dynamic_int8_max_abs_diff"], 0.1)
        self.assertLess(metrics["prequantized_int8_bytes_read"], metrics["float32_bytes_read"])


if __name__ == "__main__":
    unittest.main()
