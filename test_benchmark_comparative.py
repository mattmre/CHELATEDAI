"""
Tests for ComparativeTestbed and extended metrics (Phase 6)

Run: python -m pytest test_benchmark_comparative.py -v
"""

import unittest
from unittest.mock import patch
import json
import tempfile
import os

from benchmark_comparative import (
    BenchmarkConfiguration,
    BenchmarkResult,
    ComparativeTestbed,
    mean_average_precision_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    get_default_configurations,
)


# =============================================================================
# Extended Metric Tests
# =============================================================================

class TestMeanAveragePrecisionAtK(unittest.TestCase):
    """Tests for MAP@k metric."""

    def test_perfect_retrieval(self):
        """All retrieved are relevant -> AP = 1.0."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        result = mean_average_precision_at_k(retrieved, relevant, k=3)
        self.assertAlmostEqual(result, 1.0)

    def test_no_relevant(self):
        """No relevant docs -> AP = 0.0."""
        retrieved = ["a", "b", "c"]
        result = mean_average_precision_at_k(retrieved, set(), k=3)
        self.assertAlmostEqual(result, 0.0)

    def test_partial_retrieval(self):
        """One relevant at rank 1 out of 3 -> AP = 1/1 / min(1,3) = 1.0."""
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        result = mean_average_precision_at_k(retrieved, relevant, k=3)
        # Precision at rank 1 = 1/1 = 1.0; sum_precision = 1.0; AP = 1.0 / min(1, 3) = 1.0
        self.assertAlmostEqual(result, 1.0)

    def test_relevant_at_end(self):
        """Relevant doc at last position."""
        retrieved = ["x", "y", "a"]
        relevant = {"a"}
        result = mean_average_precision_at_k(retrieved, relevant, k=3)
        # Precision at rank 3 = 1/3; AP = (1/3) / min(1, 3) = 1/3
        self.assertAlmostEqual(result, 1.0 / 3.0, places=5)

    def test_k_cutoff(self):
        """Only first k results considered."""
        retrieved = ["x", "y", "z", "a"]
        relevant = {"a"}
        result = mean_average_precision_at_k(retrieved, relevant, k=3)
        self.assertAlmostEqual(result, 0.0)  # "a" is at rank 4, beyond k=3


class TestMeanReciprocalRank(unittest.TestCase):
    """Tests for MRR metric."""

    def test_relevant_at_rank_1(self):
        """First result is relevant -> RR = 1.0."""
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        self.assertAlmostEqual(mean_reciprocal_rank(retrieved, relevant), 1.0)

    def test_relevant_at_rank_3(self):
        """Relevant at rank 3 -> RR = 1/3."""
        retrieved = ["x", "y", "a"]
        relevant = {"a"}
        self.assertAlmostEqual(mean_reciprocal_rank(retrieved, relevant), 1.0 / 3.0)

    def test_no_relevant(self):
        """No relevant -> RR = 0.0."""
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        self.assertAlmostEqual(mean_reciprocal_rank(retrieved, relevant), 0.0)


class TestRecallAtK(unittest.TestCase):
    """Tests for Recall@k metric."""

    def test_perfect_recall(self):
        """All relevant found in top-k -> Recall = 1.0."""
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "b"}
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=4), 1.0)

    def test_no_recall(self):
        """No relevant found -> Recall = 0.0."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=3), 0.0)

    def test_partial_recall(self):
        """One of two relevant found -> Recall = 0.5."""
        retrieved = ["a", "x", "y"]
        relevant = {"a", "b"}
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, k=3), 0.5)

    def test_empty_relevant(self):
        """No relevant docs exist -> Recall = 0.0."""
        retrieved = ["a", "b"]
        self.assertAlmostEqual(recall_at_k(retrieved, set(), k=2), 0.0)


# =============================================================================
# BenchmarkConfiguration Tests
# =============================================================================

class TestBenchmarkConfiguration(unittest.TestCase):
    """Tests for BenchmarkConfiguration dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        config = BenchmarkConfiguration(name="test")
        self.assertEqual(config.name, "test")
        self.assertEqual(config.chelation_p, 85)
        self.assertFalse(config.use_centering)
        self.assertEqual(config.temperature, 1.0)
        self.assertEqual(config.adapter_type, "mlp")

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BenchmarkConfiguration(
            name="custom", chelation_p=90, use_centering=True,
            temperature=0.5, adapter_type="procrustes"
        )
        self.assertEqual(config.chelation_p, 90)
        self.assertTrue(config.use_centering)
        self.assertEqual(config.temperature, 0.5)


# =============================================================================
# ComparativeTestbed Tests
# =============================================================================

@patch('benchmark_comparative.get_logger')
class TestComparativeTestbed(unittest.TestCase):
    """Tests for the ComparativeTestbed orchestrator."""

    def setUp(self):
        """Create minimal test dataset."""
        self.corpus = {
            "doc1": "Neural networks are powerful models",
            "doc2": "Transformers use attention mechanisms",
            "doc3": "Embeddings represent text numerically",
        }
        self.queries = {
            "q1": "What are neural networks?",
            "q2": "How do transformers work?",
        }
        self.qrels = {
            "q1": {"doc1": 1, "doc2": 0, "doc3": 0},
            "q2": {"doc1": 0, "doc2": 1, "doc3": 0},
        }

    def test_initialization_default_configs(self, mock_logger):
        """Test testbed initializes with default configurations."""
        testbed = ComparativeTestbed()
        self.assertEqual(len(testbed.configurations), 8)

    def test_initialization_custom_configs(self, mock_logger):
        """Test testbed with custom configuration list."""
        configs = [BenchmarkConfiguration(name="only_one")]
        testbed = ComparativeTestbed(configurations=configs)
        self.assertEqual(len(testbed.configurations), 1)

    def test_evaluate_single_config_no_engine(self, mock_logger):
        """Test evaluation with no engine factory (dummy mode)."""
        testbed = ComparativeTestbed()
        config = BenchmarkConfiguration(name="test")
        result = testbed.evaluate_single_config(
            config, self.corpus, self.queries, self.qrels
        )
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.config_name, "test")
        self.assertEqual(result.num_queries, 2)
        self.assertEqual(result.num_docs, 3)

    def test_run_all_no_engine(self, mock_logger):
        """Test running all default configs without engine."""
        configs = [
            BenchmarkConfiguration(name="config_a"),
            BenchmarkConfiguration(name="config_b"),
        ]
        testbed = ComparativeTestbed(configurations=configs)
        results = testbed.run_all(self.corpus, self.queries, self.qrels)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].config_name, "config_a")
        self.assertEqual(results[1].config_name, "config_b")

    def test_format_ascii_table(self, mock_logger):
        """Test ASCII table formatting."""
        testbed = ComparativeTestbed()
        testbed.results = [
            BenchmarkResult("test", 0.5, 0.4, 0.6, 0.3, 10.0, 100, 1000)
        ]
        table = testbed.format_ascii_table()
        self.assertIn("test", table)
        self.assertIn("NDCG@10", table)
        self.assertIn("MAP@10", table)

    def test_format_ascii_table_empty(self, mock_logger):
        """Test ASCII table with no results."""
        testbed = ComparativeTestbed()
        table = testbed.format_ascii_table()
        self.assertIn("No results", table)

    def test_export_json_returns_dict(self, mock_logger):
        """Test JSON export returns correct structure."""
        testbed = ComparativeTestbed(configurations=[
            BenchmarkConfiguration(name="t1")
        ])
        testbed.results = [
            BenchmarkResult("t1", 0.5, 0.4, 0.6, 0.3, 10.0, 100, 1000)
        ]
        data = testbed.export_json()
        self.assertIn("configurations", data)
        self.assertIn("results", data)
        self.assertIn("summary", data)
        self.assertEqual(data["summary"]["num_results"], 1)

    def test_export_json_to_file(self, mock_logger):
        """Test JSON export writes valid file."""
        testbed = ComparativeTestbed(configurations=[
            BenchmarkConfiguration(name="t1")
        ])
        testbed.results = [
            BenchmarkResult("t1", 0.5, 0.4, 0.6, 0.3, 10.0, 100, 1000)
        ]

        tmpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmpfile.close()
        try:
            testbed.export_json(tmpfile.name)
            with open(tmpfile.name, 'r') as f:
                data = json.load(f)
            self.assertIn("results", data)
        finally:
            os.unlink(tmpfile.name)

    def test_default_configurations_count(self, mock_logger):
        """Test that default configurations returns expected count."""
        configs = get_default_configurations()
        self.assertEqual(len(configs), 8)
        names = [c.name for c in configs]
        self.assertIn("baseline", names)
        self.assertIn("chelation", names)
        self.assertIn("procrustes", names)
        self.assertIn("low_rank_16", names)
        self.assertIn("online_updates", names)
        self.assertIn("random_mask_50pct", names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
