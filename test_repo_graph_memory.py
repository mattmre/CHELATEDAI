import os
import sys
import tempfile
import unittest

from pathlib import Path

POC_DIR = os.path.join(os.path.dirname(__file__), "computational_storage_poc")
if POC_DIR not in sys.path:
    sys.path.insert(0, POC_DIR)

from repo_graph_memory import DiskBackedRepoGraphMemory, ingest_repo_graph_memory  # noqa: E402
from repo_graph_memory_benchmark import benchmark_repo_graph_memory  # noqa: E402


class RepoGraphMemoryTests(unittest.TestCase):
    def test_ingest_writes_disk_backed_memory_files(self):
        with tempfile.TemporaryDirectory() as repo_dir, tempfile.TemporaryDirectory() as memory_dir:
            root = Path(repo_dir)
            (root / "alpha.py").write_text(
                "from beta import helper\n\n\ndef build_cache():\n    return helper()\n",
                encoding="utf-8",
            )
            (root / "beta.py").write_text(
                "def helper():\n    return 'ok'\n",
                encoding="utf-8",
            )

            metrics = ingest_repo_graph_memory(root, memory_dir)

            self.assertGreaterEqual(metrics["node_count"], 4)
            self.assertGreater(metrics["edge_count"], 0)
            self.assertTrue((Path(memory_dir) / "manifest.json").exists())
            self.assertTrue((Path(memory_dir) / "nodes.jsonl").exists())
            self.assertTrue((Path(memory_dir) / "edges.json").exists())
            self.assertTrue((Path(memory_dir) / "embeddings.npy").exists())

    def test_query_returns_relevant_file_for_code_task(self):
        with tempfile.TemporaryDirectory() as repo_dir, tempfile.TemporaryDirectory() as memory_dir:
            root = Path(repo_dir)
            (root / "packed_graph.py").write_text(
                "def build_packed_graph_artifact():\n    return b'packed int8 artifact'\n",
                encoding="utf-8",
            )
            (root / "sparse_cpu_inference.py").write_text(
                "class SparseChunkCache:\n    pass\n",
                encoding="utf-8",
            )

            ingest_repo_graph_memory(root, memory_dir)
            with DiskBackedRepoGraphMemory(memory_dir) as memory:
                results = memory.query("packed int8 artifact", top_k=3)

            self.assertTrue(results)
            self.assertEqual(results[0].path, "packed_graph.py")

    def test_query_prefers_core_module_for_direct_module_query(self):
        with tempfile.TemporaryDirectory() as repo_dir, tempfile.TemporaryDirectory() as memory_dir:
            root = Path(repo_dir)
            emulation_dir = root / "emulation"
            emulation_dir.mkdir()
            (emulation_dir / "virtual_controller.py").write_text(
                "class VirtualController:\n"
                "    def read_sector(self):\n"
                "        return b''\n",
                encoding="utf-8",
            )
            (emulation_dir / "validate_emulation_path.py").write_text(
                "from emulation.virtual_controller import VirtualController\n\n"
                "def validate_path_reads():\n"
                "    controller = VirtualController()\n"
                "    return controller.read_sector()\n",
                encoding="utf-8",
            )

            ingest_repo_graph_memory(root, memory_dir)
            with DiskBackedRepoGraphMemory(memory_dir) as memory:
                results = memory.query("virtual controller reads sector", top_k=3)

            self.assertTrue(results)
            self.assertEqual(results[0].path, "emulation/virtual_controller.py")
            self.assertEqual(len({result.path for result in results}), len(results))

    def test_repo_graph_memory_benchmark_reports_quality_and_latency(self):
        metrics = benchmark_repo_graph_memory()
        self.assertGreater(metrics["node_count"], 0)
        self.assertGreater(metrics["edge_count"], 0)
        self.assertGreater(metrics["query_count"], 0)
        self.assertGreater(metrics["ingest_latency_ms"], 0.0)
        self.assertGreater(metrics["avg_query_latency_ms"], 0.0)
        self.assertGreaterEqual(metrics["top3_hit_rate"], metrics["top1_hit_rate"])
        self.assertGreater(metrics["top1_hit_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
