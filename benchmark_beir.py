"""
BEIR Multi-Dataset Evaluation Framework

Extends ChelatedAI benchmarking to support multiple BEIR datasets with
tier-based grouping, corpus sampling for large datasets, and cross-product
result aggregation.

Composes existing ComparativeTestbed (no modifications to it).

Usage:
    python benchmark_beir.py --tier medium --output benchmark_beir_results.json

Dataset tiers:
    quick    -> SciFact
    small    -> SciFact, NFCorpus
    medium   -> SciFact, NFCorpus, FiQA2018
    research -> SciFact, NFCorpus, FiQA2018, TRECCOVID
    full     -> SciFact, NFCorpus, FiQA2018, TRECCOVID, NQ, HotpotQA
"""

import json
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Set, Callable

from benchmark_comparative import (
    BenchmarkConfiguration,
    BenchmarkResult,
    ComparativeTestbed,
    get_default_configurations,
)
from chelation_logger import get_logger


# =============================================================================
# Dataset Metadata
# =============================================================================

@dataclass
class DatasetInfo:
    """Metadata for a single BEIR dataset.

    Attributes:
        name: MTEB/BEIR task name (e.g. "SciFact")
        description: Short human-readable description
        corpus_size: Approximate number of documents
        query_count: Approximate number of test queries
        domain: Subject domain (e.g. "scientific", "biomedical")
        tier: Minimum tier that includes this dataset
        default_sample_size: Default corpus sample size (None = no sampling)
    """
    name: str
    description: str
    corpus_size: int
    query_count: int
    domain: str
    tier: str
    default_sample_size: Optional[int] = None


# =============================================================================
# BEIRDatasetRegistry
# =============================================================================

TIER_ORDER = ["quick", "small", "medium", "research", "full"]


class BEIRDatasetRegistry:
    """Central registry of BEIR datasets with tier-based grouping.

    Tier hierarchy (cumulative):
        quick    -> {SciFact}
        small    -> quick + {NFCorpus}
        medium   -> small + {FiQA2018}
        research -> medium + {TRECCOVID}
        full     -> research + {NQ, HotpotQA}
    """

    _DATASETS: Dict[str, DatasetInfo] = {
        "SciFact": DatasetInfo(
            name="SciFact",
            description="Scientific claim verification",
            corpus_size=5183,
            query_count=300,
            domain="scientific",
            tier="quick",
            default_sample_size=None,
        ),
        "NFCorpus": DatasetInfo(
            name="NFCorpus",
            description="Nutrition and fitness information retrieval",
            corpus_size=3633,
            query_count=323,
            domain="biomedical",
            tier="small",
            default_sample_size=None,
        ),
        "FiQA2018": DatasetInfo(
            name="FiQA2018",
            description="Financial question answering",
            corpus_size=57638,
            query_count=648,
            domain="financial",
            tier="medium",
            default_sample_size=10000,
        ),
        "TRECCOVID": DatasetInfo(
            name="TRECCOVID",
            description="COVID-19 information retrieval",
            corpus_size=171332,
            query_count=50,
            domain="biomedical",
            tier="research",
            default_sample_size=10000,
        ),
        "NQ": DatasetInfo(
            name="NQ",
            description="Natural Questions (Google)",
            corpus_size=2681468,
            query_count=3452,
            domain="general",
            tier="full",
            default_sample_size=10000,
        ),
        "HotpotQA": DatasetInfo(
            name="HotpotQA",
            description="Multi-hop question answering",
            corpus_size=5233329,
            query_count=7405,
            domain="general",
            tier="full",
            default_sample_size=10000,
        ),
    }

    @classmethod
    def get_dataset(cls, name: str) -> DatasetInfo:
        """Get dataset metadata by name.

        Args:
            name: Dataset name (e.g. "SciFact")

        Returns:
            DatasetInfo with metadata

        Raises:
            KeyError: If dataset not in registry
        """
        if name not in cls._DATASETS:
            available = ", ".join(cls._DATASETS.keys())
            raise KeyError(
                f"Dataset '{name}' not found in registry. "
                f"Available: {available}"
            )
        return cls._DATASETS[name]

    @classmethod
    def get_tier_datasets(cls, tier: str) -> List[DatasetInfo]:
        """Get all datasets up to and including the specified tier.

        Args:
            tier: Tier name ("quick", "small", "medium", "research", "full")

        Returns:
            List of DatasetInfo objects in this tier

        Raises:
            ValueError: If tier is not recognized
        """
        if tier not in TIER_ORDER:
            valid = ", ".join(TIER_ORDER)
            raise ValueError(f"Unknown tier '{tier}'. Valid tiers: {valid}")

        target_idx = TIER_ORDER.index(tier)
        included_tiers = set(TIER_ORDER[:target_idx + 1])

        return [
            ds for ds in cls._DATASETS.values()
            if ds.tier in included_tiers
        ]

    @classmethod
    def list_all(cls) -> List[DatasetInfo]:
        """List all registered datasets.

        Returns:
            List of all DatasetInfo objects
        """
        return list(cls._DATASETS.values())

    @classmethod
    def list_tiers(cls) -> List[str]:
        """List all available tier names in order.

        Returns:
            List of tier name strings
        """
        return list(TIER_ORDER)

    @classmethod
    def register_dataset(cls, info: DatasetInfo):
        """Register a new dataset (for extensibility).

        Args:
            info: DatasetInfo to register

        Raises:
            ValueError: If dataset name already registered
        """
        if info.name in cls._DATASETS:
            raise ValueError(f"Dataset '{info.name}' already registered")
        cls._DATASETS[info.name] = info

    @classmethod
    def _reset_registry(cls):
        """Reset registry to defaults (for testing only)."""
        # Re-initialize with the default datasets
        cls._DATASETS = {
            "SciFact": DatasetInfo(
                name="SciFact",
                description="Scientific claim verification",
                corpus_size=5183, query_count=300,
                domain="scientific", tier="quick",
                default_sample_size=None,
            ),
            "NFCorpus": DatasetInfo(
                name="NFCorpus",
                description="Nutrition and fitness information retrieval",
                corpus_size=3633, query_count=323,
                domain="biomedical", tier="small",
                default_sample_size=None,
            ),
            "FiQA2018": DatasetInfo(
                name="FiQA2018",
                description="Financial question answering",
                corpus_size=57638, query_count=648,
                domain="financial", tier="medium",
                default_sample_size=10000,
            ),
            "TRECCOVID": DatasetInfo(
                name="TRECCOVID",
                description="COVID-19 information retrieval",
                corpus_size=171332, query_count=50,
                domain="biomedical", tier="research",
                default_sample_size=10000,
            ),
            "NQ": DatasetInfo(
                name="NQ",
                description="Natural Questions (Google)",
                corpus_size=2681468, query_count=3452,
                domain="general", tier="full",
                default_sample_size=10000,
            ),
            "HotpotQA": DatasetInfo(
                name="HotpotQA",
                description="Multi-hop question answering",
                corpus_size=5233329, query_count=7405,
                domain="general", tier="full",
                default_sample_size=10000,
            ),
        }


# =============================================================================
# DatasetLoader
# =============================================================================

class DatasetLoader:
    """Wraps MTEB data loading with optional corpus sampling.

    For large datasets, samples the corpus while preserving qrels integrity:
    all documents referenced by qrels are always included.
    """

    def __init__(self, sample_seed: int = 42):
        """Initialize the loader.

        Args:
            sample_seed: Random seed for reproducible sampling
        """
        self.sample_seed = sample_seed
        self.logger = get_logger()

    def load(
        self,
        dataset_name: str,
        sample_size: Optional[int] = None,
    ) -> Tuple[Dict, Dict, Dict]:
        """Load a dataset with optional corpus sampling.

        Args:
            dataset_name: BEIR/MTEB task name
            sample_size: Max corpus docs to keep (None = all).
                         When sampling, all docs referenced in qrels are preserved.

        Returns:
            Tuple of (corpus, queries, qrels) dicts

        Raises:
            RuntimeError: If dataset loading fails
        """
        from benchmark_utils import load_mteb_data

        self.logger.log_event(
            "beir_load_start",
            f"Loading BEIR dataset: {dataset_name}",
            dataset=dataset_name,
            sample_size=sample_size,
        )

        corpus, queries, qrels = load_mteb_data(dataset_name)

        if corpus is None:
            raise RuntimeError(f"Failed to load dataset '{dataset_name}'")

        if sample_size is not None and len(corpus) > sample_size:
            corpus = self._sample_corpus(corpus, qrels, sample_size)

        self.logger.log_event(
            "beir_load_complete",
            f"Loaded {dataset_name}: {len(corpus)} docs, {len(queries)} queries",
            dataset=dataset_name,
            corpus_size=len(corpus),
            query_count=len(queries),
        )

        return corpus, queries, qrels

    def _sample_corpus(
        self,
        corpus: Dict,
        qrels: Dict,
        sample_size: int,
    ) -> Dict:
        """Sample corpus while preserving all qrels-referenced documents.

        Args:
            corpus: Full corpus dict {doc_id: text}
            qrels: Relevance judgments {qid: {doc_id: score}}
            sample_size: Target number of documents

        Returns:
            Sampled corpus dict
        """
        # Collect all doc IDs referenced in qrels
        qrel_doc_ids: Set[str] = set()
        for qid_rels in qrels.values():
            for doc_id in qid_rels:
                qrel_doc_ids.add(str(doc_id))

        # Ensure qrels docs are present in corpus
        required_ids = qrel_doc_ids & set(str(k) for k in corpus.keys())

        # Remaining pool (not in qrels)
        all_ids = set(str(k) for k in corpus.keys())
        pool_ids = list(all_ids - required_ids)

        # How many more do we need?
        remaining_budget = max(0, sample_size - len(required_ids))

        # Sample from pool
        rng = np.random.RandomState(self.sample_seed)
        if remaining_budget < len(pool_ids):
            sampled_extra = set(rng.choice(pool_ids, size=remaining_budget, replace=False))
        else:
            sampled_extra = set(pool_ids)

        # Combine
        keep_ids = required_ids | sampled_extra

        # Build sampled corpus (match original key types)
        sampled = {}
        for k, v in corpus.items():
            if str(k) in keep_ids:
                sampled[k] = v

        self.logger.log_event(
            "beir_corpus_sampled",
            f"Sampled corpus: {len(sampled)} from {len(corpus)} "
            f"(preserved {len(required_ids)} qrels docs)",
            original_size=len(corpus),
            sampled_size=len(sampled),
            qrels_preserved=len(required_ids),
        )

        return sampled


# =============================================================================
# Multi-Dataset Result
# =============================================================================

@dataclass
class MultiDatasetResult:
    """Results from running one configuration across one dataset.

    Attributes:
        dataset_name: Name of the BEIR dataset
        config_name: Configuration that produced this result
        result: The BenchmarkResult from ComparativeTestbed
        elapsed_seconds: Wall-clock time for this evaluation
    """
    dataset_name: str
    config_name: str
    result: BenchmarkResult
    elapsed_seconds: float = 0.0


# =============================================================================
# BEIRBenchmarkRunner
# =============================================================================

class BEIRBenchmarkRunner:
    """Orchestrates ComparativeTestbed across multiple BEIR datasets.

    Composes (does not inherit from) ComparativeTestbed.
    """

    def __init__(
        self,
        configurations: Optional[List[BenchmarkConfiguration]] = None,
        tier: str = "quick",
        sample_seed: int = 42,
        engine_factory: Optional[Callable] = None,
    ):
        """Initialize the BEIR benchmark runner.

        Args:
            configurations: Benchmark configurations to evaluate.
                If None, uses get_default_configurations().
            tier: Dataset tier to evaluate ("quick"/"small"/"medium"/"research"/"full")
            sample_seed: Seed for corpus sampling reproducibility
            engine_factory: Optional engine factory for real evaluation
        """
        self.configurations = configurations or get_default_configurations()
        self.tier = tier
        self.sample_seed = sample_seed
        self.engine_factory = engine_factory
        self.logger = get_logger()

        self.datasets = BEIRDatasetRegistry.get_tier_datasets(tier)
        self.loader = DatasetLoader(sample_seed=sample_seed)

        # Results storage
        self.results: List[MultiDatasetResult] = []
        self._dataset_cache: Dict[str, Tuple[Dict, Dict, Dict]] = {}

    def run_single_dataset(
        self,
        dataset_name: str,
        corpus: Dict,
        queries: Dict,
        qrels: Dict,
    ) -> List[MultiDatasetResult]:
        """Run all configurations against a single pre-loaded dataset.

        Args:
            dataset_name: Name of the dataset
            corpus: Corpus dict
            queries: Queries dict
            qrels: Qrels dict

        Returns:
            List of MultiDatasetResult for each configuration
        """
        testbed = ComparativeTestbed(configurations=self.configurations)

        dataset_results = []

        for config in self.configurations:
            self.logger.log_event(
                "beir_eval_start",
                f"Evaluating {config.name} on {dataset_name}",
                config=config.name,
                dataset=dataset_name,
            )

            start = time.perf_counter()
            result = testbed.evaluate_single_config(
                config, corpus, queries, qrels, self.engine_factory
            )
            elapsed = time.perf_counter() - start

            mdr = MultiDatasetResult(
                dataset_name=dataset_name,
                config_name=config.name,
                result=result,
                elapsed_seconds=elapsed,
            )
            dataset_results.append(mdr)
            self.results.append(mdr)

            self.logger.log_event(
                "beir_eval_complete",
                f"{config.name} on {dataset_name}: NDCG@10={result.ndcg_at_10:.4f}",
                config=config.name,
                dataset=dataset_name,
                ndcg=result.ndcg_at_10,
                elapsed=elapsed,
            )

        return dataset_results

    def run_all(self) -> List[MultiDatasetResult]:
        """Run all configurations across all datasets in the selected tier.

        Returns:
            List of all MultiDatasetResult objects
        """
        self.results = []

        self.logger.log_event(
            "beir_run_start",
            f"Starting BEIR evaluation: tier={self.tier}, "
            f"{len(self.datasets)} datasets, {len(self.configurations)} configs",
            tier=self.tier,
            num_datasets=len(self.datasets),
            num_configs=len(self.configurations),
        )

        for ds_info in self.datasets:
            # Load (or use cache)
            if ds_info.name in self._dataset_cache:
                corpus, queries, qrels = self._dataset_cache[ds_info.name]
            else:
                corpus, queries, qrels = self.loader.load(
                    ds_info.name,
                    sample_size=ds_info.default_sample_size,
                )
                self._dataset_cache[ds_info.name] = (corpus, queries, qrels)

            self.run_single_dataset(ds_info.name, corpus, queries, qrels)

        self.logger.log_event(
            "beir_run_complete",
            f"BEIR evaluation complete: {len(self.results)} total results",
            total_results=len(self.results),
        )

        return self.results

    def run_all_preloaded(
        self,
        datasets: Dict[str, Tuple[Dict, Dict, Dict]],
    ) -> List[MultiDatasetResult]:
        """Run all configurations across pre-loaded datasets.

        Useful for testing without actual MTEB data loading.

        Args:
            datasets: Dict mapping dataset_name -> (corpus, queries, qrels)

        Returns:
            List of all MultiDatasetResult objects
        """
        self.results = []

        for ds_name, (corpus, queries, qrels) in datasets.items():
            self.run_single_dataset(ds_name, corpus, queries, qrels)

        return self.results

    def get_results_for_dataset(self, dataset_name: str) -> List[MultiDatasetResult]:
        """Get all results for a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Filtered list of MultiDatasetResult
        """
        return [r for r in self.results if r.dataset_name == dataset_name]

    def get_results_for_config(self, config_name: str) -> List[MultiDatasetResult]:
        """Get all results for a specific configuration.

        Args:
            config_name: Name of the configuration

        Returns:
            Filtered list of MultiDatasetResult
        """
        return [r for r in self.results if r.config_name == config_name]


# =============================================================================
# BEIRBenchmarkReport
# =============================================================================

class BEIRBenchmarkReport:
    """Generates cross-product reports from BEIR benchmark results.

    Provides:
    - Per-dataset tables
    - Cross-dataset aggregation (mean across datasets per config)
    - ASCII table formatting
    - JSON export (dashboard-compatible)
    """

    def __init__(self, results: List[MultiDatasetResult]):
        """Initialize the report.

        Args:
            results: List of MultiDatasetResult from a BEIRBenchmarkRunner
        """
        self.results = results

    def get_dataset_names(self) -> List[str]:
        """Get unique dataset names from results.

        Returns:
            Sorted list of dataset names
        """
        return sorted(set(r.dataset_name for r in self.results))

    def get_config_names(self) -> List[str]:
        """Get unique configuration names from results.

        Returns:
            Sorted list of config names
        """
        return sorted(set(r.config_name for r in self.results))

    def aggregate_by_config(self) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics across datasets for each configuration.

        Returns:
            Dict mapping config_name -> {metric_name: mean_value}
        """
        config_results: Dict[str, List[MultiDatasetResult]] = {}
        for r in self.results:
            config_results.setdefault(r.config_name, []).append(r)

        aggregated = {}
        for config_name, mdr_list in config_results.items():
            ndcg_vals = [m.result.ndcg_at_10 for m in mdr_list]
            map_vals = [m.result.map_at_10 for m in mdr_list]
            mrr_vals = [m.result.mrr for m in mdr_list]
            recall_vals = [m.result.recall_at_10 for m in mdr_list]
            latency_vals = [m.result.latency_ms for m in mdr_list]

            aggregated[config_name] = {
                "mean_ndcg_at_10": float(np.mean(ndcg_vals)) if ndcg_vals else 0.0,
                "mean_map_at_10": float(np.mean(map_vals)) if map_vals else 0.0,
                "mean_mrr": float(np.mean(mrr_vals)) if mrr_vals else 0.0,
                "mean_recall_at_10": float(np.mean(recall_vals)) if recall_vals else 0.0,
                "mean_latency_ms": float(np.mean(latency_vals)) if latency_vals else 0.0,
                "num_datasets": len(mdr_list),
            }

        return aggregated

    def aggregate_by_dataset(self) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics across configurations for each dataset.

        Returns:
            Dict mapping dataset_name -> {metric_name: mean_value}
        """
        ds_results: Dict[str, List[MultiDatasetResult]] = {}
        for r in self.results:
            ds_results.setdefault(r.dataset_name, []).append(r)

        aggregated = {}
        for ds_name, mdr_list in ds_results.items():
            ndcg_vals = [m.result.ndcg_at_10 for m in mdr_list]
            aggregated[ds_name] = {
                "mean_ndcg_at_10": float(np.mean(ndcg_vals)) if ndcg_vals else 0.0,
                "num_configs": len(mdr_list),
            }

        return aggregated

    def build_heatmap_data(self) -> Dict[str, Any]:
        """Build heatmap-ready data structure for the dashboard.

        Returns:
            Dict with "configs", "datasets", and "ndcg_matrix" keys.
            ndcg_matrix[i][j] = NDCG@10 for config i, dataset j.
        """
        configs = self.get_config_names()
        datasets = self.get_dataset_names()

        # Build lookup
        lookup: Dict[Tuple[str, str], float] = {}
        for r in self.results:
            lookup[(r.config_name, r.dataset_name)] = r.result.ndcg_at_10

        matrix = []
        for cfg in configs:
            row = [lookup.get((cfg, ds), 0.0) for ds in datasets]
            matrix.append(row)

        return {
            "configs": configs,
            "datasets": datasets,
            "ndcg_matrix": matrix,
        }

    def format_ascii_table(self) -> str:
        """Format results as ASCII table grouped by dataset.

        Returns:
            Formatted string
        """
        if not self.results:
            return "No BEIR results to display."

        lines = []
        datasets = self.get_dataset_names()

        for ds_name in datasets:
            ds_results = [r for r in self.results if r.dataset_name == ds_name]
            if not ds_results:
                continue

            lines.append(f"\n=== {ds_name} ===")
            header = (
                f"{'Configuration':<25} | {'NDCG@10':>7} | {'MAP@10':>7} | "
                f"{'MRR':>7} | {'Recall@10':>9} | {'Latency(ms)':>11}"
            )
            lines.append(header)
            lines.append("-" * len(header))

            for mdr in ds_results:
                r = mdr.result
                line = (
                    f"{r.config_name:<25} | {r.ndcg_at_10:>7.4f} | "
                    f"{r.map_at_10:>7.4f} | {r.mrr:>7.4f} | "
                    f"{r.recall_at_10:>9.4f} | {r.latency_ms:>11.1f}"
                )
                lines.append(line)

        # Aggregated summary
        agg = self.aggregate_by_config()
        if agg:
            lines.append("\n=== Aggregated (mean across datasets) ===")
            header = f"{'Configuration':<25} | {'Mean NDCG@10':>12} | {'Mean MAP@10':>11} | {'Datasets':>8}"
            lines.append(header)
            lines.append("-" * len(header))
            for cfg, metrics in sorted(agg.items()):
                line = (
                    f"{cfg:<25} | {metrics['mean_ndcg_at_10']:>12.4f} | "
                    f"{metrics['mean_map_at_10']:>11.4f} | "
                    f"{metrics['num_datasets']:>8}"
                )
                lines.append(line)

        return "\n".join(lines)

    def export_json(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """Export results as JSON (dashboard-compatible).

        Args:
            filepath: Optional path to write JSON file

        Returns:
            Dict with all results, aggregations, and heatmap data
        """
        data = {
            "results": [
                {
                    "dataset_name": r.dataset_name,
                    "config_name": r.config_name,
                    "ndcg_at_10": r.result.ndcg_at_10,
                    "map_at_10": r.result.map_at_10,
                    "mrr": r.result.mrr,
                    "recall_at_10": r.result.recall_at_10,
                    "latency_ms": r.result.latency_ms,
                    "num_queries": r.result.num_queries,
                    "num_docs": r.result.num_docs,
                    "elapsed_seconds": r.elapsed_seconds,
                }
                for r in self.results
            ],
            "aggregated_by_config": self.aggregate_by_config(),
            "aggregated_by_dataset": self.aggregate_by_dataset(),
            "heatmap": self.build_heatmap_data(),
            "summary": {
                "num_datasets": len(self.get_dataset_names()),
                "num_configs": len(self.get_config_names()),
                "total_evaluations": len(self.results),
            },
        }

        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

        return data


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for BEIR multi-dataset evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ChelatedAI BEIR Multi-Dataset Evaluation"
    )
    parser.add_argument(
        "--tier",
        default="quick",
        choices=TIER_ORDER,
        help="Dataset tier to evaluate (default: quick)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output file path (default: benchmark_beir_results.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for corpus sampling (default: 42)",
    )
    args = parser.parse_args()

    output_path = args.output or "benchmark_beir_results.json"

    datasets = BEIRDatasetRegistry.get_tier_datasets(args.tier)
    print(f"BEIR Evaluation: tier={args.tier}")
    print(f"Datasets ({len(datasets)}):")
    for ds in datasets:
        sample_note = f" (sampled to {ds.default_sample_size})" if ds.default_sample_size else ""
        print(f"  - {ds.name}: {ds.description}{sample_note}")

    runner = BEIRBenchmarkRunner(tier=args.tier, sample_seed=args.seed)

    print(f"\nRunning {len(runner.configurations)} configurations across {len(datasets)} datasets...")
    results = runner.run_all()

    report = BEIRBenchmarkReport(results)
    print(report.format_ascii_table())

    report.export_json(output_path)
    print(f"\nResults exported to {output_path}")


if __name__ == "__main__":
    main()
