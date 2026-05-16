import os
import sys
import unittest

POC_DIR = os.path.join(os.path.dirname(__file__), "computational_storage_poc")
if POC_DIR not in sys.path:
    sys.path.insert(0, POC_DIR)

from integrated_runtime_compression_benchmark import benchmark_integrated_runtime_compression  # noqa: E402
from repo_graph_memory import FLOAT32_EMBEDDING_DTYPE, INT8_EMBEDDING_DTYPE  # noqa: E402
from repo_graph_memory_benchmark import benchmark_repo_graph_memory  # noqa: E402
from repo_graph_memory_compression_benchmark import benchmark_repo_graph_memory_compression  # noqa: E402


class MemoryCompressionTests(unittest.TestCase):
    def test_repo_graph_memory_supports_float32_and_int8_storage(self):
        float_metrics = benchmark_repo_graph_memory(FLOAT32_EMBEDDING_DTYPE)
        int8_metrics = benchmark_repo_graph_memory(INT8_EMBEDDING_DTYPE)

        self.assertGreater(float_metrics["mapped_bytes"], int8_metrics["mapped_bytes"])
        self.assertGreater(float_metrics["top3_hit_rate"], 0.0)
        self.assertGreater(int8_metrics["top3_hit_rate"], 0.0)

    def test_repo_graph_memory_compression_reports_memory_savings(self):
        metrics = benchmark_repo_graph_memory_compression()
        self.assertGreater(metrics["float32_mapped_bytes"], metrics["int8_mapped_bytes"])
        self.assertGreater(metrics["mapped_byte_reduction_pct"], 0.0)
        self.assertGreater(metrics["float32_top3_hit_rate"], 0.0)
        self.assertGreater(metrics["int8_top3_hit_rate"], 0.0)

    def test_integrated_runtime_compression_reports_memory_savings(self):
        metrics = benchmark_integrated_runtime_compression()
        self.assertGreater(metrics["float32_mapped_bytes"], metrics["int8_mapped_bytes"])
        self.assertGreater(metrics["mapped_byte_reduction_pct"], 0.0)
        self.assertGreater(metrics["float32_top3_hit_rate"], 0.0)
        self.assertGreater(metrics["int8_top3_hit_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
