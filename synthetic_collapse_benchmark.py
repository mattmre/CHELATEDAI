"""Deterministic synthetic semantic-collapse benchmark."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Mapping

import numpy as np

from benchmark_utils import mean_reciprocal_rank, ndcg_at_k, recall_at_k


def _cosine_scores(query: np.ndarray, documents: Mapping[str, np.ndarray]) -> Dict[str, float]:
    query_norm = float(np.linalg.norm(query))
    scores = {}
    for doc_id, vector in documents.items():
        denom = query_norm * float(np.linalg.norm(vector))
        scores[doc_id] = float(np.dot(query, vector) / denom) if denom else 0.0
    return scores


def _rank(scores: Mapping[str, float]) -> List[str]:
    return [
        doc_id
        for doc_id, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


def _metric_row(rankings: Mapping[str, List[str]], qrels: Mapping[str, str], k: int = 3) -> Dict[str, float]:
    ndcg_scores = []
    mrr_scores = []
    recall_scores = []
    for query_id, relevant_doc in qrels.items():
        retrieved = rankings[query_id][:k]
        relevant = {relevant_doc}
        relevance_by_rank = [1 if doc_id in relevant else 0 for doc_id in retrieved]
        ndcg_scores.append(float(ndcg_at_k(relevance_by_rank, k)))
        mrr_scores.append(float(mean_reciprocal_rank(retrieved, relevant)))
        recall_scores.append(float(recall_at_k(retrieved, relevant, k)))
    return {
        "ndcg_at_3": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "recall_at_3": float(np.mean(recall_scores)) if recall_scores else 0.0,
    }


def build_synthetic_collapse_fixture(topic_count: int = 4, collapse_strength: float = 4.0) -> Dict[str, Any]:
    """Create queries where one noisy dimension overwhelms semantic dimensions."""

    if topic_count < 2:
        raise ValueError("topic_count must be at least 2")
    semantic_dims = topic_count
    collapse_dim = topic_count
    documents: Dict[str, np.ndarray] = {}
    queries: Dict[str, np.ndarray] = {}
    qrels: Dict[str, str] = {}
    for topic in range(topic_count):
        relevant_id = f"d{topic}_relevant"
        distractor_id = f"d{topic}_collapse_distractor"
        query_id = f"q{topic}"
        relevant = np.zeros(semantic_dims + 1, dtype=float)
        relevant[topic] = 1.0
        distractor = np.zeros(semantic_dims + 1, dtype=float)
        distractor[collapse_dim] = collapse_strength
        query = np.zeros(semantic_dims + 1, dtype=float)
        query[topic] = 1.0
        query[collapse_dim] = collapse_strength
        documents[relevant_id] = relevant
        documents[distractor_id] = distractor
        queries[query_id] = query
        qrels[query_id] = relevant_id
    return {
        "queries": queries,
        "documents": documents,
        "qrels": qrels,
        "collapse_dim": collapse_dim,
    }


def evaluate_synthetic_collapse(
    fixture: Dict[str, Any],
    *,
    masked_dims: List[int] | None = None,
) -> Dict[str, Any]:
    """Evaluate synthetic retrieval before or after masking collapsed dimensions."""

    documents = fixture["documents"]
    queries = fixture["queries"]
    mask = None
    if masked_dims is not None:
        vector_dim = len(next(iter(documents.values())))
        mask = np.ones(vector_dim, dtype=float)
        for dim in masked_dims:
            mask[dim] = 0.0
    rankings: Dict[str, List[str]] = {}
    for query_id, query in queries.items():
        masked_query = query * mask if mask is not None else query
        masked_documents = {
            doc_id: vector * mask if mask is not None else vector
            for doc_id, vector in documents.items()
        }
        rankings[query_id] = _rank(_cosine_scores(masked_query, masked_documents))
    metrics = _metric_row(rankings, fixture["qrels"])
    return {
        "metrics": metrics,
        "rankings": rankings,
        "masked_dims": masked_dims or [],
    }


def run_synthetic_collapse_benchmark(
    topic_count: int = 4,
    collapse_strength: float = 4.0,
) -> Dict[str, Any]:
    """Run baseline and known-good collapse-mask evaluations."""

    fixture = build_synthetic_collapse_fixture(topic_count=topic_count, collapse_strength=collapse_strength)
    baseline = evaluate_synthetic_collapse(fixture)
    masked = evaluate_synthetic_collapse(fixture, masked_dims=[fixture["collapse_dim"]])
    return {
        "topic_count": topic_count,
        "collapse_strength": collapse_strength,
        "collapse_dim": fixture["collapse_dim"],
        "baseline": baseline,
        "masked": masked,
        "delta_ndcg_at_3": masked["metrics"]["ndcg_at_3"] - baseline["metrics"]["ndcg_at_3"],
        "recovered": masked["metrics"]["ndcg_at_3"] > baseline["metrics"]["ndcg_at_3"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the synthetic semantic-collapse benchmark")
    parser.add_argument("--topic-count", type=int, default=4)
    parser.add_argument("--collapse-strength", type=float, default=4.0)
    args = parser.parse_args()
    print(json.dumps(
        run_synthetic_collapse_benchmark(
            topic_count=args.topic_count,
            collapse_strength=args.collapse_strength,
        ),
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
