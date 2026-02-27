"""
Tests for BEIR Multi-Dataset Evaluation Framework

Run: python test_benchmark_beir.py
"""

import unittest
import json
import tempfile
import os
from unittest.mock import patch
from dataclasses import asdict

from benchmark_beir import (
    BEIRDatasetRegistry,
    BEIRBenchmarkRunner,
    BEIRBenchmarkReport,
    DatasetInfo,
    DatasetLoader,
    MultiDatasetResult,
    TIER_ORDER,
)
from benchmark_comparative import (
    BenchmarkConfiguration,
    BenchmarkResult,
)


# =============================================================================
# Helper: create minimal test data
# =============================================================================

def make_test_dataset(num_docs=10, num_queries=3):
    """Create a minimal synthetic dataset for testing."""
    corpus = {f"doc{i}": f"Document content number {i}" for i in range(num_docs)}
    queries = {f"q{i}": f"Test query number {i}" for i in range(num_queries)}
    qrels = {}
    for i in range(num_queries):
        qrels[f"q{i}"] = {f"doc{i % num_docs}": 1}
    return corpus, queries, qrels


# =============================================================================
# TestBEIRDatasetRegistry
# =============================================================================

class TestBEIRDatasetRegistry(unittest.TestCase):
    """Tests for the BEIRDatasetRegistry."""

    def setUp(self):
        BEIRDatasetRegistry._reset_registry()

    def tearDown(self):
        BEIRDatasetRegistry._reset_registry()

    def test_get_dataset_scifact(self):
        """Get SciFact dataset info."""
        ds = BEIRDatasetRegistry.get_dataset("SciFact")
        self.assertEqual(ds.name, "SciFact")
        self.assertEqual(ds.tier, "quick")
        self.assertEqual(ds.domain, "scientific")

    def test_get_dataset_not_found(self):
        """Getting unknown dataset raises KeyError."""
        with self.assertRaises(KeyError):
            BEIRDatasetRegistry.get_dataset("NonExistentDataset")

    def test_get_tier_quick(self):
        """Quick tier contains only SciFact."""
        datasets = BEIRDatasetRegistry.get_tier_datasets("quick")
        names = [d.name for d in datasets]
        self.assertEqual(names, ["SciFact"])

    def test_get_tier_small(self):
        """Small tier contains SciFact + NFCorpus."""
        datasets = BEIRDatasetRegistry.get_tier_datasets("small")
        names = sorted([d.name for d in datasets])
        self.assertIn("SciFact", names)
        self.assertIn("NFCorpus", names)
        self.assertEqual(len(names), 2)

    def test_get_tier_medium(self):
        """Medium tier contains 3 datasets."""
        datasets = BEIRDatasetRegistry.get_tier_datasets("medium")
        names = [d.name for d in datasets]
        self.assertEqual(len(names), 3)
        self.assertIn("FiQA2018", names)

    def test_get_tier_research(self):
        """Research tier contains 4 datasets."""
        datasets = BEIRDatasetRegistry.get_tier_datasets("research")
        names = [d.name for d in datasets]
        self.assertEqual(len(names), 4)
        self.assertIn("TRECCOVID", names)

    def test_get_tier_full(self):
        """Full tier contains all 6 datasets."""
        datasets = BEIRDatasetRegistry.get_tier_datasets("full")
        self.assertEqual(len(datasets), 6)
        names = [d.name for d in datasets]
        self.assertIn("NQ", names)
        self.assertIn("HotpotQA", names)

    def test_invalid_tier(self):
        """Invalid tier raises ValueError."""
        with self.assertRaises(ValueError):
            BEIRDatasetRegistry.get_tier_datasets("nonexistent")

    def test_list_all(self):
        """List all returns all 6 datasets."""
        all_ds = BEIRDatasetRegistry.list_all()
        self.assertEqual(len(all_ds), 6)

    def test_list_tiers(self):
        """List tiers returns the correct order."""
        tiers = BEIRDatasetRegistry.list_tiers()
        self.assertEqual(tiers, ["quick", "small", "medium", "research", "full"])

    def test_register_dataset(self):
        """Can register a new dataset."""
        new_ds = DatasetInfo(
            name="CustomDS",
            description="Custom test dataset",
            corpus_size=100,
            query_count=10,
            domain="test",
            tier="quick",
        )
        BEIRDatasetRegistry.register_dataset(new_ds)
        retrieved = BEIRDatasetRegistry.get_dataset("CustomDS")
        self.assertEqual(retrieved.name, "CustomDS")

    def test_register_duplicate_raises(self):
        """Registering an existing name raises ValueError."""
        dup = DatasetInfo(
            name="SciFact",
            description="duplicate",
            corpus_size=0,
            query_count=0,
            domain="test",
            tier="quick",
        )
        with self.assertRaises(ValueError):
            BEIRDatasetRegistry.register_dataset(dup)

    def test_default_sample_sizes(self):
        """Large datasets have default_sample_size set."""
        fiqa = BEIRDatasetRegistry.get_dataset("FiQA2018")
        self.assertEqual(fiqa.default_sample_size, 10000)
        scifact = BEIRDatasetRegistry.get_dataset("SciFact")
        self.assertIsNone(scifact.default_sample_size)


# =============================================================================
# TestDatasetLoader
# =============================================================================

@patch('benchmark_beir.get_logger')
class TestDatasetLoader(unittest.TestCase):
    """Tests for DatasetLoader with corpus sampling."""

    def test_sample_corpus_preserves_qrels(self, mock_logger):
        """Sampling preserves all documents referenced in qrels."""
        loader = DatasetLoader(sample_seed=42)
        corpus = {f"doc{i}": f"Text {i}" for i in range(100)}
        qrels = {"q1": {"doc0": 1, "doc50": 1, "doc99": 1}}

        sampled = loader._sample_corpus(corpus, qrels, sample_size=10)
        self.assertLessEqual(len(sampled), 10 + 3)  # budget + required
        # All qrels docs must be present
        for doc_id in ["doc0", "doc50", "doc99"]:
            self.assertIn(doc_id, sampled)

    def test_sample_corpus_respects_size(self, mock_logger):
        """Sampled corpus size roughly matches sample_size."""
        loader = DatasetLoader(sample_seed=42)
        corpus = {f"doc{i}": f"Text {i}" for i in range(1000)}
        qrels = {"q1": {"doc0": 1}}

        sampled = loader._sample_corpus(corpus, qrels, sample_size=100)
        self.assertLessEqual(len(sampled), 100)

    def test_sample_corpus_no_sampling_needed(self, mock_logger):
        """When corpus <= sample_size, no sampling occurs."""
        loader = DatasetLoader(sample_seed=42)
        corpus = {f"doc{i}": f"Text {i}" for i in range(50)}
        qrels = {"q1": {"doc0": 1}}

        sampled = loader._sample_corpus(corpus, qrels, sample_size=100)
        self.assertEqual(len(sampled), 50)

    def test_sample_reproducibility(self, mock_logger):
        """Same seed produces same sample."""
        corpus = {f"doc{i}": f"Text {i}" for i in range(200)}
        qrels = {"q1": {"doc0": 1}}

        loader1 = DatasetLoader(sample_seed=123)
        s1 = loader1._sample_corpus(corpus, qrels, sample_size=50)

        loader2 = DatasetLoader(sample_seed=123)
        s2 = loader2._sample_corpus(corpus, qrels, sample_size=50)

        self.assertEqual(set(s1.keys()), set(s2.keys()))

    def test_sample_different_seeds(self, mock_logger):
        """Different seeds produce different samples."""
        corpus = {f"doc{i}": f"Text {i}" for i in range(200)}
        qrels = {"q1": {"doc0": 1}}

        loader1 = DatasetLoader(sample_seed=1)
        s1 = loader1._sample_corpus(corpus, qrels, sample_size=50)

        loader2 = DatasetLoader(sample_seed=999)
        s2 = loader2._sample_corpus(corpus, qrels, sample_size=50)

        # Very unlikely to be identical with different seeds
        self.assertNotEqual(set(s1.keys()), set(s2.keys()))

    def test_sample_corpus_empty_qrels(self, mock_logger):
        """Empty qrels still allows sampling."""
        loader = DatasetLoader(sample_seed=42)
        corpus = {f"doc{i}": f"Text {i}" for i in range(100)}
        qrels = {}

        sampled = loader._sample_corpus(corpus, qrels, sample_size=10)
        self.assertEqual(len(sampled), 10)


# =============================================================================
# TestMultiDatasetResult
# =============================================================================

class TestMultiDatasetResult(unittest.TestCase):
    """Tests for MultiDatasetResult dataclass."""

    def test_creation(self):
        """Can create MultiDatasetResult."""
        br = BenchmarkResult("cfg1", 0.5, 0.4, 0.6, 0.3, 10.0, 100, 1000)
        mdr = MultiDatasetResult(
            dataset_name="SciFact",
            config_name="cfg1",
            result=br,
            elapsed_seconds=1.5,
        )
        self.assertEqual(mdr.dataset_name, "SciFact")
        self.assertEqual(mdr.config_name, "cfg1")
        self.assertEqual(mdr.result.ndcg_at_10, 0.5)
        self.assertEqual(mdr.elapsed_seconds, 1.5)

    def test_default_elapsed(self):
        """Default elapsed_seconds is 0.0."""
        br = BenchmarkResult("cfg1", 0.5, 0.4, 0.6, 0.3, 10.0, 50, 500)
        mdr = MultiDatasetResult(
            dataset_name="Test",
            config_name="cfg1",
            result=br,
        )
        self.assertEqual(mdr.elapsed_seconds, 0.0)

    def test_asdict(self):
        """MultiDatasetResult is serializable via asdict."""
        br = BenchmarkResult("cfg1", 0.5, 0.4, 0.6, 0.3, 10.0, 50, 500)
        mdr = MultiDatasetResult(
            dataset_name="Test",
            config_name="cfg1",
            result=br,
        )
        d = asdict(mdr)
        self.assertIn("dataset_name", d)
        self.assertIn("result", d)


# =============================================================================
# TestBEIRBenchmarkRunner
# =============================================================================

@patch('benchmark_beir.get_logger')
class TestBEIRBenchmarkRunner(unittest.TestCase):
    """Tests for BEIRBenchmarkRunner."""

    def setUp(self):
        BEIRDatasetRegistry._reset_registry()

    def tearDown(self):
        BEIRDatasetRegistry._reset_registry()

    def test_initialization_defaults(self, mock_logger):
        """Runner initializes with default configurations."""
        runner = BEIRBenchmarkRunner(tier="quick")
        self.assertEqual(len(runner.configurations), 8)
        self.assertEqual(runner.tier, "quick")
        self.assertEqual(len(runner.datasets), 1)

    def test_initialization_custom_configs(self, mock_logger):
        """Runner accepts custom configurations."""
        configs = [BenchmarkConfiguration(name="single")]
        runner = BEIRBenchmarkRunner(configurations=configs, tier="quick")
        self.assertEqual(len(runner.configurations), 1)

    def test_run_single_dataset(self, mock_logger):
        """Run a single dataset evaluation."""
        configs = [
            BenchmarkConfiguration(name="a"),
            BenchmarkConfiguration(name="b"),
        ]
        runner = BEIRBenchmarkRunner(configurations=configs, tier="quick")
        corpus, queries, qrels = make_test_dataset()

        results = runner.run_single_dataset("TestDS", corpus, queries, qrels)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].dataset_name, "TestDS")
        self.assertEqual(results[1].dataset_name, "TestDS")

    def test_run_all_preloaded(self, mock_logger):
        """Run all configurations on pre-loaded datasets."""
        configs = [BenchmarkConfiguration(name="cfg1")]
        runner = BEIRBenchmarkRunner(configurations=configs, tier="quick")

        ds1_data = make_test_dataset(10, 2)
        ds2_data = make_test_dataset(20, 3)

        results = runner.run_all_preloaded({
            "DS1": ds1_data,
            "DS2": ds2_data,
        })
        self.assertEqual(len(results), 2)
        ds_names = {r.dataset_name for r in results}
        self.assertEqual(ds_names, {"DS1", "DS2"})

    def test_get_results_for_dataset(self, mock_logger):
        """Filter results by dataset name."""
        configs = [BenchmarkConfiguration(name="cfg1")]
        runner = BEIRBenchmarkRunner(configurations=configs, tier="quick")

        runner.run_all_preloaded({
            "DS1": make_test_dataset(5, 2),
            "DS2": make_test_dataset(5, 2),
        })

        ds1_results = runner.get_results_for_dataset("DS1")
        self.assertEqual(len(ds1_results), 1)
        self.assertEqual(ds1_results[0].dataset_name, "DS1")

    def test_get_results_for_config(self, mock_logger):
        """Filter results by configuration name."""
        configs = [
            BenchmarkConfiguration(name="fast"),
            BenchmarkConfiguration(name="slow"),
        ]
        runner = BEIRBenchmarkRunner(configurations=configs, tier="quick")

        runner.run_all_preloaded({"DS1": make_test_dataset(5, 2)})

        fast_results = runner.get_results_for_config("fast")
        self.assertEqual(len(fast_results), 1)
        self.assertEqual(fast_results[0].config_name, "fast")

    def test_results_have_elapsed_time(self, mock_logger):
        """Results include non-negative elapsed time."""
        configs = [BenchmarkConfiguration(name="t")]
        runner = BEIRBenchmarkRunner(configurations=configs, tier="quick")

        runner.run_all_preloaded({"DS1": make_test_dataset()})

        for r in runner.results:
            self.assertGreaterEqual(r.elapsed_seconds, 0.0)

    def test_multiple_configs_multiple_datasets(self, mock_logger):
        """Cross-product: 3 configs x 2 datasets = 6 results."""
        configs = [
            BenchmarkConfiguration(name="a"),
            BenchmarkConfiguration(name="b"),
            BenchmarkConfiguration(name="c"),
        ]
        runner = BEIRBenchmarkRunner(configurations=configs, tier="quick")

        runner.run_all_preloaded({
            "DS1": make_test_dataset(),
            "DS2": make_test_dataset(),
        })

        self.assertEqual(len(runner.results), 6)

    def test_invalid_tier_raises(self, mock_logger):
        """Invalid tier raises ValueError."""
        with self.assertRaises(ValueError):
            BEIRBenchmarkRunner(tier="invalid_tier")

    def test_result_contains_benchmark_result(self, mock_logger):
        """Each MultiDatasetResult contains a proper BenchmarkResult."""
        configs = [BenchmarkConfiguration(name="t")]
        runner = BEIRBenchmarkRunner(configurations=configs, tier="quick")

        runner.run_all_preloaded({"DS1": make_test_dataset()})

        mdr = runner.results[0]
        self.assertIsInstance(mdr.result, BenchmarkResult)
        self.assertEqual(mdr.result.config_name, "t")


# =============================================================================
# TestBEIRBenchmarkReport
# =============================================================================

@patch('benchmark_beir.get_logger')
class TestBEIRBenchmarkReport(unittest.TestCase):
    """Tests for BEIRBenchmarkReport."""

    def _make_results(self):
        """Create sample results for 2 configs x 2 datasets."""
        results = []
        for ds in ["SciFact", "NFCorpus"]:
            for cfg, ndcg in [("baseline", 0.4), ("chelation", 0.6)]:
                br = BenchmarkResult(cfg, ndcg, 0.3, 0.5, 0.2, 5.0, 100, 500)
                mdr = MultiDatasetResult(
                    dataset_name=ds,
                    config_name=cfg,
                    result=br,
                    elapsed_seconds=1.0,
                )
                results.append(mdr)
        return results

    def test_get_dataset_names(self, mock_logger):
        """Report extracts unique dataset names."""
        results = self._make_results()
        report = BEIRBenchmarkReport(results)
        names = report.get_dataset_names()
        self.assertEqual(set(names), {"SciFact", "NFCorpus"})

    def test_get_config_names(self, mock_logger):
        """Report extracts unique config names."""
        results = self._make_results()
        report = BEIRBenchmarkReport(results)
        names = report.get_config_names()
        self.assertEqual(set(names), {"baseline", "chelation"})

    def test_aggregate_by_config(self, mock_logger):
        """Aggregation by config computes mean across datasets."""
        results = self._make_results()
        report = BEIRBenchmarkReport(results)
        agg = report.aggregate_by_config()

        self.assertIn("baseline", agg)
        self.assertIn("chelation", agg)
        # baseline has 0.4 on both datasets
        self.assertAlmostEqual(agg["baseline"]["mean_ndcg_at_10"], 0.4)
        self.assertAlmostEqual(agg["chelation"]["mean_ndcg_at_10"], 0.6)
        self.assertEqual(agg["baseline"]["num_datasets"], 2)

    def test_aggregate_by_dataset(self, mock_logger):
        """Aggregation by dataset computes mean across configs."""
        results = self._make_results()
        report = BEIRBenchmarkReport(results)
        agg = report.aggregate_by_dataset()

        self.assertIn("SciFact", agg)
        # SciFact has baseline=0.4 and chelation=0.6, mean=0.5
        self.assertAlmostEqual(agg["SciFact"]["mean_ndcg_at_10"], 0.5)

    def test_build_heatmap_data(self, mock_logger):
        """Heatmap data has correct structure."""
        results = self._make_results()
        report = BEIRBenchmarkReport(results)
        heatmap = report.build_heatmap_data()

        self.assertIn("configs", heatmap)
        self.assertIn("datasets", heatmap)
        self.assertIn("ndcg_matrix", heatmap)
        self.assertEqual(len(heatmap["configs"]), 2)
        self.assertEqual(len(heatmap["datasets"]), 2)
        self.assertEqual(len(heatmap["ndcg_matrix"]), 2)
        self.assertEqual(len(heatmap["ndcg_matrix"][0]), 2)

    def test_format_ascii_table(self, mock_logger):
        """ASCII table contains expected content."""
        results = self._make_results()
        report = BEIRBenchmarkReport(results)
        table = report.format_ascii_table()

        self.assertIn("SciFact", table)
        self.assertIn("NFCorpus", table)
        self.assertIn("baseline", table)
        self.assertIn("chelation", table)
        self.assertIn("Aggregated", table)

    def test_format_ascii_table_empty(self, mock_logger):
        """ASCII table handles empty results."""
        report = BEIRBenchmarkReport([])
        table = report.format_ascii_table()
        self.assertIn("No BEIR results", table)

    def test_export_json_returns_dict(self, mock_logger):
        """JSON export returns correct structure."""
        results = self._make_results()
        report = BEIRBenchmarkReport(results)
        data = report.export_json()

        self.assertIn("results", data)
        self.assertIn("aggregated_by_config", data)
        self.assertIn("aggregated_by_dataset", data)
        self.assertIn("heatmap", data)
        self.assertIn("summary", data)
        self.assertEqual(data["summary"]["total_evaluations"], 4)

    def test_export_json_to_file(self, mock_logger):
        """JSON export writes valid file."""
        results = self._make_results()
        report = BEIRBenchmarkReport(results)

        tmpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmpfile.close()
        try:
            report.export_json(tmpfile.name)
            with open(tmpfile.name, 'r') as f:
                data = json.load(f)
            self.assertIn("results", data)
            self.assertEqual(len(data["results"]), 4)
        finally:
            os.unlink(tmpfile.name)


# =============================================================================
# TestCorpusSampling
# =============================================================================

@patch('benchmark_beir.get_logger')
class TestCorpusSampling(unittest.TestCase):
    """Edge-case tests for corpus sampling logic."""

    def test_all_docs_in_qrels(self, mock_logger):
        """When all docs are in qrels, sample_size may be exceeded."""
        loader = DatasetLoader(sample_seed=42)
        corpus = {f"doc{i}": f"Text {i}" for i in range(10)}
        qrels = {f"q{i}": {f"doc{i}": 1} for i in range(10)}

        sampled = loader._sample_corpus(corpus, qrels, sample_size=5)
        # All 10 docs are referenced in qrels, so all are preserved
        self.assertEqual(len(sampled), 10)

    def test_sample_size_equals_corpus_size(self, mock_logger):
        """When sample_size equals corpus size, all docs kept."""
        loader = DatasetLoader(sample_seed=42)
        corpus = {f"doc{i}": f"Text {i}" for i in range(50)}
        qrels = {"q0": {"doc0": 1}}

        sampled = loader._sample_corpus(corpus, qrels, sample_size=50)
        self.assertEqual(len(sampled), 50)

    def test_sample_single_doc_corpus(self, mock_logger):
        """Single-doc corpus sampling works."""
        loader = DatasetLoader(sample_seed=42)
        corpus = {"doc0": "Only doc"}
        qrels = {"q0": {"doc0": 1}}

        sampled = loader._sample_corpus(corpus, qrels, sample_size=1)
        self.assertEqual(len(sampled), 1)
        self.assertIn("doc0", sampled)

    def test_qrels_reference_nonexistent_docs(self, mock_logger):
        """Qrels referencing docs not in corpus are handled gracefully."""
        loader = DatasetLoader(sample_seed=42)
        corpus = {f"doc{i}": f"Text {i}" for i in range(20)}
        qrels = {"q0": {"doc0": 1, "ghost_doc": 1}}

        sampled = loader._sample_corpus(corpus, qrels, sample_size=10)
        # ghost_doc not in corpus, so only doc0 is required
        self.assertIn("doc0", sampled)
        self.assertNotIn("ghost_doc", sampled)
        self.assertLessEqual(len(sampled), 10)

    def test_large_sample_budget(self, mock_logger):
        """When budget > pool, all docs are included."""
        loader = DatasetLoader(sample_seed=42)
        corpus = {f"doc{i}": f"Text {i}" for i in range(30)}
        qrels = {"q0": {"doc0": 1}}

        sampled = loader._sample_corpus(corpus, qrels, sample_size=100)
        self.assertEqual(len(sampled), 30)


# =============================================================================
# TestConfigPresetIntegration
# =============================================================================

class TestConfigPresetIntegration(unittest.TestCase):
    """Tests for BEIR presets in ChelationConfig."""

    def test_beir_preset_quick(self):
        """Quick BEIR preset loads correctly."""
        from config import ChelationConfig
        preset = ChelationConfig.get_preset("quick", "beir")
        self.assertEqual(preset["tier"], "quick")
        self.assertIn("description", preset)

    def test_beir_preset_full(self):
        """Full BEIR preset loads correctly."""
        from config import ChelationConfig
        preset = ChelationConfig.get_preset("full", "beir")
        self.assertEqual(preset["tier"], "full")

    def test_beir_preset_invalid(self):
        """Invalid BEIR preset raises ValueError."""
        from config import ChelationConfig
        with self.assertRaises(ValueError):
            ChelationConfig.get_preset("nonexistent", "beir")

    def test_all_beir_presets_match_tiers(self):
        """Each BEIR preset tier matches a valid TIER_ORDER entry."""
        from config import ChelationConfig
        for tier_name in TIER_ORDER:
            preset = ChelationConfig.get_preset(tier_name, "beir")
            self.assertEqual(preset["tier"], tier_name)


# =============================================================================
# TestTierOrder
# =============================================================================

class TestTierOrder(unittest.TestCase):
    """Tests for tier ordering and cumulative behavior."""

    def test_tier_order_values(self):
        """TIER_ORDER has expected values."""
        self.assertEqual(TIER_ORDER, ["quick", "small", "medium", "research", "full"])

    def test_tiers_are_cumulative(self):
        """Each higher tier includes all datasets from lower tiers."""
        for i in range(1, len(TIER_ORDER)):
            lower_tier = TIER_ORDER[i - 1]
            higher_tier = TIER_ORDER[i]
            lower_ds = set(d.name for d in BEIRDatasetRegistry.get_tier_datasets(lower_tier))
            higher_ds = set(d.name for d in BEIRDatasetRegistry.get_tier_datasets(higher_tier))
            self.assertTrue(
                lower_ds.issubset(higher_ds),
                f"Tier '{lower_tier}' datasets should be subset of '{higher_tier}'"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
