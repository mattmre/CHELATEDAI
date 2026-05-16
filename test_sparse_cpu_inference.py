import os
import sys
import tempfile
import unittest

import numpy as np

POC_DIR = os.path.join(os.path.dirname(__file__), "computational_storage_poc")
if POC_DIR not in sys.path:
    sys.path.insert(0, POC_DIR)

from block_graph import BLOCK_SIZE  # noqa: E402
from cpu_backends import NumpyInt8DynamicBackend  # noqa: E402
from packed_cpu_inference import run_packed_graph_with_backend  # noqa: E402
from packed_graph import DiskBackedPackedGraph, INT8_STORAGE_DTYPE, write_packed_graph_artifact  # noqa: E402
from sparse_cpu_inference import SparseChunkCache, SparseInferenceConfig, run_sparse_packed_graph_with_backend  # noqa: E402
from sparse_inference_benchmark import benchmark_sparse_inference  # noqa: E402


class SparseCPUInferenceTests(unittest.TestCase):
    def test_sparse_runtime_matches_dense_runtime_on_sparse_input(self):
        rng = np.random.default_rng(41)
        matrices = [
            np.eye(16, dtype=np.float32),
            rng.normal(size=(16, 4)).astype(np.float32) * 0.1,
        ]
        input_activations = np.zeros((1, BLOCK_SIZE), dtype=np.float16)
        input_activations[0, [0, 1, 8, 9]] = [0.9, 0.7, 0.8, 0.6]

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = os.path.join(temp_dir, "graph_int8.cspg")
            write_packed_graph_artifact(artifact_path, matrices, storage_dtype=INT8_STORAGE_DTYPE)
            with DiskBackedPackedGraph(artifact_path) as graph:
                dense_result = run_packed_graph_with_backend(graph, input_activations, NumpyInt8DynamicBackend())
                sparse_result = run_sparse_packed_graph_with_backend(
                    graph,
                    input_activations,
                    NumpyInt8DynamicBackend(),
                    config=SparseInferenceConfig(chunk_rows=4, stream_from_block=1),
                    cache=SparseChunkCache(max_cached_chunks=4),
                )

        np.testing.assert_allclose(sparse_result.output[:, :4], dense_result.output[:, :4], rtol=0.15, atol=0.02)
        self.assertLess(sparse_result.bytes_read, dense_result.bytes_read)

    def test_sparse_runtime_cache_reduces_repeated_read_bytes(self):
        rng = np.random.default_rng(42)
        matrices = [
            np.eye(16, dtype=np.float32),
            rng.normal(size=(16, 4)).astype(np.float32) * 0.1,
        ]
        input_activations = np.zeros((1, BLOCK_SIZE), dtype=np.float16)
        input_activations[0, [0, 1, 8, 9]] = [0.9, 0.7, 0.8, 0.6]

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = os.path.join(temp_dir, "graph_int8.cspg")
            write_packed_graph_artifact(artifact_path, matrices, storage_dtype=INT8_STORAGE_DTYPE)
            cache = SparseChunkCache(max_cached_chunks=4)
            config = SparseInferenceConfig(chunk_rows=4, stream_from_block=1)
            with DiskBackedPackedGraph(artifact_path) as graph:
                first_result = run_sparse_packed_graph_with_backend(
                    graph,
                    input_activations,
                    NumpyInt8DynamicBackend(),
                    config=config,
                    cache=cache,
                )
                second_result = run_sparse_packed_graph_with_backend(
                    graph,
                    input_activations,
                    NumpyInt8DynamicBackend(),
                    config=config,
                    cache=cache,
                )

        self.assertGreater(first_result.bytes_read, second_result.bytes_read)
        self.assertGreater(second_result.cache_hits, 0)

    def test_sparse_benchmark_reports_streaming_reduction_and_parity(self):
        metrics = benchmark_sparse_inference(token_count=16)
        self.assertGreater(metrics["dense_bytes_per_token"], metrics["sparse_bytes_per_token"])
        self.assertGreater(metrics["streamed_byte_reduction_pct"], 0.0)
        self.assertLess(metrics["max_abs_diff"], 0.1)


if __name__ == "__main__":
    unittest.main()
