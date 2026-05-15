import os
import sys
import tempfile
import unittest

from pathlib import Path

POC_DIR = os.path.join(os.path.dirname(__file__), "computational_storage_poc")
if POC_DIR not in sys.path:
    sys.path.insert(0, POC_DIR)

from integrated_repo_runtime import IntegratedRepoRuntime  # noqa: E402
from integrated_runtime_benchmark import benchmark_integrated_repo_runtime  # noqa: E402


class IntegratedRepoRuntimeTests(unittest.TestCase):
    def test_runtime_returns_ranked_candidates_and_metrics(self):
        with tempfile.TemporaryDirectory() as repo_dir:
            root = Path(repo_dir)
            (root / "packed_graph.py").write_text(
                "def build_packed_graph_artifact():\n    return b'packed graph int8 artifact'\n",
                encoding="utf-8",
            )
            (root / "sparse_cpu_inference.py").write_text(
                "class SparseChunkCache:\n    pass\n\ndef selective_loading():\n    return 'sparse chunk cache'\n",
                encoding="utf-8",
            )
            (root / "cpu_inference_benchmark.py").write_text(
                "def prequantized_int8_cpu_benchmark():\n    return 'prequantized int8 cpu benchmark'\n",
                encoding="utf-8",
            )

            with IntegratedRepoRuntime(root, top_k=3) as runtime:
                response = runtime.answer_query("packed graph int8 artifact")

            self.assertTrue(response.candidates)
            self.assertEqual(response.candidates[0].path, "packed_graph.py")
            self.assertGreater(response.metrics.retrieval_latency_ms, 0.0)
            self.assertGreater(response.metrics.inference_latency_ms, 0.0)
            self.assertGreater(response.metrics.total_latency_ms, 0.0)
            self.assertGreater(response.metrics.bytes_read, 0)
            self.assertGreater(response.metrics.mapped_bytes, 0)

    def test_integrated_runtime_benchmark_reports_hit_rate_and_latency(self):
        metrics = benchmark_integrated_repo_runtime()
        self.assertGreater(metrics["query_count"], 0)
        self.assertGreater(metrics["avg_retrieval_latency_ms"], 0.0)
        self.assertGreater(metrics["avg_inference_latency_ms"], 0.0)
        self.assertGreater(metrics["avg_total_latency_ms"], 0.0)
        self.assertGreater(metrics["queries_per_second"], 0.0)
        self.assertGreater(metrics["avg_bytes_read_per_query"], 0.0)
        self.assertGreater(metrics["mapped_bytes"], 0.0)
        self.assertGreaterEqual(metrics["top3_hit_rate"], metrics["top1_hit_rate"])
        self.assertGreater(metrics["top1_hit_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
