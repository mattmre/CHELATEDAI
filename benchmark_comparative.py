"""
Comparative Testbed for ChelatedAI (Phase 6)

Benchmarks multiple ChelatedAI configurations side-by-side with
extended metrics (NDCG@10, MAP@10, MRR, Recall@10, Latency).

Usage:
    python benchmark_comparative.py [--task SciFact]
"""

import time
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Any, Callable
from benchmark_utils import ndcg_at_k, canonicalize_id
from chelation_logger import get_logger


@dataclass
class BenchmarkConfiguration:
    """
    A single benchmark configuration to evaluate.

    Attributes:
        name: Human-readable configuration name
        chelation_p: Chelation percentile parameter
        use_centering: Whether to enable spectral chelation
        use_quantization: Whether to enable quantization
        temperature: Temperature scaling for spectral chelation
        adapter_type: Adapter variant ("mlp", "procrustes", "low_rank")
        adapter_kwargs: Additional kwargs for adapter factory
        online_updates: Whether to enable online gradient updates
        random_mask_pct: If set, percentage of dimensions to randomly mask
        extra_setup: Optional callable for additional engine configuration
    """
    name: str
    chelation_p: int = 85
    use_centering: bool = False
    use_quantization: bool = False
    temperature: float = 1.0
    adapter_type: str = "mlp"
    adapter_kwargs: Dict[str, Any] = field(default_factory=dict)
    online_updates: bool = False
    random_mask_pct: Optional[float] = None
    extra_setup: Optional[Callable] = field(default=None, repr=False)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark configuration run."""
    config_name: str
    ndcg_at_10: float
    map_at_10: float
    mrr: float
    recall_at_10: float
    latency_ms: float
    num_queries: int
    num_docs: int


def mean_average_precision_at_k(retrieved_ids, relevant_ids, k=10):
    """
    Mean Average Precision at rank k for a single query.

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevant_ids: Set of relevant document IDs
        k: Rank cutoff

    Returns:
        float: AP@k score
    """
    retrieved = retrieved_ids[:k]
    relevant_set = set(relevant_ids)

    if not relevant_set:
        return 0.0

    num_relevant = 0
    sum_precision = 0.0

    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant_set:
            num_relevant += 1
            sum_precision += num_relevant / (i + 1)

    if num_relevant == 0:
        return 0.0

    return sum_precision / min(len(relevant_set), k)


def mean_reciprocal_rank(retrieved_ids, relevant_ids):
    """
    Reciprocal Rank for a single query.

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevant_ids: Set of relevant document IDs

    Returns:
        float: RR score (1/rank of first relevant result, or 0)
    """
    relevant_set = set(relevant_ids)

    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)

    return 0.0


def recall_at_k(retrieved_ids, relevant_ids, k=10):
    """
    Recall at rank k for a single query.

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevant_ids: Set of relevant document IDs
        k: Rank cutoff

    Returns:
        float: Recall@k score
    """
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    if not relevant_set:
        return 0.0

    return len(retrieved_set & relevant_set) / len(relevant_set)


# ===== Default Configurations =====

def get_default_configurations():
    """
    Get the standard set of benchmark configurations.

    Returns:
        list of BenchmarkConfiguration
    """
    return [
        BenchmarkConfiguration(
            name="baseline",
            use_centering=False,
            use_quantization=False
        ),
        BenchmarkConfiguration(
            name="chelation",
            use_centering=True,
            use_quantization=False
        ),
        BenchmarkConfiguration(
            name="chelation+tempscale_0.5",
            use_centering=True,
            temperature=0.5
        ),
        BenchmarkConfiguration(
            name="chelation+tempscale_2.0",
            use_centering=True,
            temperature=2.0
        ),
        BenchmarkConfiguration(
            name="procrustes",
            use_centering=True,
            adapter_type="procrustes"
        ),
        BenchmarkConfiguration(
            name="low_rank_16",
            use_centering=True,
            adapter_type="low_rank",
            adapter_kwargs={"rank": 16}
        ),
        BenchmarkConfiguration(
            name="online_updates",
            use_centering=True,
            online_updates=True
        ),
        BenchmarkConfiguration(
            name="random_mask_50pct",
            use_centering=False,
            random_mask_pct=50.0
        ),
    ]


class ComparativeTestbed:
    """
    Orchestrates comparative benchmarking across configurations.

    Args:
        configurations: List of BenchmarkConfiguration to evaluate
    """

    def __init__(self, configurations=None):
        self.configurations = configurations or get_default_configurations()
        self.results = []
        self.logger = get_logger()

    def evaluate_single_config(self, config, corpus, queries, qrels,
                               engine_factory=None):
        """
        Evaluate a single configuration against a dataset.

        Args:
            config: BenchmarkConfiguration
            corpus: dict {doc_id: text}
            queries: dict {query_id: text}
            qrels: dict {query_id: {doc_id: relevance}}
            engine_factory: Optional callable(config) -> engine instance

        Returns:
            BenchmarkResult
        """
        ndcg_scores = []
        map_scores = []
        mrr_scores = []
        recall_scores = []
        latencies = []

        # If no engine factory, simulate with dummy scores
        if engine_factory is None:
            for qid, qtext in queries.items():
                rel_docs = qrels.get(str(qid), qrels.get(qid, {}))
                relevant_set = set(str(d) for d, s in rel_docs.items() if s > 0)

                # Dummy retrieval (for testing framework)
                retrieved = list(corpus.keys())[:10]
                retrieved = [str(d) for d in retrieved]

                # Compute metrics
                r = [1 if d in relevant_set else 0 for d in retrieved]
                ndcg_scores.append(ndcg_at_k(r, 10))
                map_scores.append(mean_average_precision_at_k(retrieved, relevant_set, 10))
                mrr_scores.append(mean_reciprocal_rank(retrieved, relevant_set))
                recall_scores.append(recall_at_k(retrieved, relevant_set, 10))
                latencies.append(0.0)
        else:
            engine = engine_factory(config)
            try:
                for qid, qtext in queries.items():
                    rel_docs = qrels.get(str(qid), qrels.get(qid, {}))
                    relevant_set = set(str(d) for d, s in rel_docs.items() if s > 0)

                    start = time.perf_counter()
                    std_top, chel_top, mask, jaccard = engine.run_inference(qtext)
                    elapsed = (time.perf_counter() - start) * 1000

                    retrieved = [canonicalize_id(d) for d in chel_top[:10]]
                    r = [1 if d in relevant_set else 0 for d in retrieved]

                    ndcg_scores.append(ndcg_at_k(r, 10))
                    map_scores.append(mean_average_precision_at_k(retrieved, relevant_set, 10))
                    mrr_scores.append(mean_reciprocal_rank(retrieved, relevant_set))
                    recall_scores.append(recall_at_k(retrieved, relevant_set, 10))
                    latencies.append(elapsed)
            finally:
                if hasattr(engine, 'close'):
                    engine.close()

        result = BenchmarkResult(
            config_name=config.name,
            ndcg_at_10=float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
            map_at_10=float(np.mean(map_scores)) if map_scores else 0.0,
            mrr=float(np.mean(mrr_scores)) if mrr_scores else 0.0,
            recall_at_10=float(np.mean(recall_scores)) if recall_scores else 0.0,
            latency_ms=float(np.mean(latencies)) if latencies else 0.0,
            num_queries=len(queries),
            num_docs=len(corpus)
        )

        return result

    def run_all(self, corpus, queries, qrels, engine_factory=None):
        """
        Run all configurations and collect results.

        Args:
            corpus: dict {doc_id: text}
            queries: dict {query_id: text}
            qrels: dict {query_id: {doc_id: relevance}}
            engine_factory: Optional callable(config) -> engine instance

        Returns:
            list of BenchmarkResult
        """
        self.results = []

        for config in self.configurations:
            self.logger.log_event(
                "benchmark_start",
                f"Evaluating configuration: {config.name}",
                config_name=config.name
            )

            result = self.evaluate_single_config(
                config, corpus, queries, qrels, engine_factory
            )
            self.results.append(result)

            self.logger.log_event(
                "benchmark_complete",
                f"{config.name}: NDCG@10={result.ndcg_at_10:.4f}",
                config_name=config.name,
                ndcg=result.ndcg_at_10
            )

        return self.results

    def format_ascii_table(self, results=None):
        """
        Format results as ASCII table.

        Args:
            results: Optional list of BenchmarkResult (uses self.results if None)

        Returns:
            str: Formatted ASCII table
        """
        results = results or self.results
        if not results:
            return "No results to display."

        # Header
        header = f"{'Configuration':<25} | {'NDCG@10':>7} | {'MAP@10':>7} | {'MRR':>7} | {'Recall@10':>9} | {'Latency(ms)':>11}"
        separator = "-" * len(header)

        lines = [header, separator]

        for r in results:
            line = (
                f"{r.config_name:<25} | {r.ndcg_at_10:>7.4f} | {r.map_at_10:>7.4f} | "
                f"{r.mrr:>7.4f} | {r.recall_at_10:>9.4f} | {r.latency_ms:>11.1f}"
            )
            lines.append(line)

        return "\n".join(lines)

    def export_json(self, filepath=None):
        """
        Export results as JSON.

        Args:
            filepath: Optional file path to write. If None, returns dict.

        Returns:
            dict: Results as JSON-serializable dict
        """
        data = {
            "configurations": [asdict(c) for c in self.configurations
                               if not callable(getattr(c, 'extra_setup', None))
                               or True],
            "results": [asdict(r) for r in self.results],
            "summary": {
                "num_configurations": len(self.configurations),
                "num_results": len(self.results)
            }
        }

        # Clean up non-serializable fields
        for c in data["configurations"]:
            c.pop("extra_setup", None)

        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

        return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ChelatedAI Comparative Benchmark")
    parser.add_argument("--task", default="SciFact", help="MTEB task name")
    parser.add_argument("--output", default=None, help="JSON output file")
    args = parser.parse_args()

    from benchmark_utils import load_mteb_data

    print(f"Loading {args.task} dataset...")
    corpus, queries, qrels = load_mteb_data(args.task)

    if corpus is None:
        print("Failed to load dataset. Exiting.")
        exit(1)

    print(f"Loaded {len(corpus)} documents, {len(queries)} queries")

    testbed = ComparativeTestbed()
    results = testbed.run_all(corpus, queries, qrels)

    print("\n" + testbed.format_ascii_table())

    if args.output:
        testbed.export_json(args.output)
        print(f"\nResults exported to {args.output}")
