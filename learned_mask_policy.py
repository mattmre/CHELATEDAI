"""Lightweight learned dimension-mask policies for collapse diagnostics."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Mapping

import numpy as np

from synthetic_collapse_benchmark import (
    build_synthetic_collapse_fixture,
    evaluate_synthetic_collapse,
)


def learn_pairwise_collapse_mask(
    queries: Mapping[str, np.ndarray],
    documents: Mapping[str, np.ndarray],
    qrels: Mapping[str, str],
    *,
    mask_budget: int = 1,
) -> Dict[str, Any]:
    """Learn dimensions where distractors align with queries more than relevant docs."""

    if mask_budget < 1:
        raise ValueError("mask_budget must be at least 1")
    vector_dim = len(next(iter(documents.values())))
    harmful_scores = np.zeros(vector_dim, dtype=float)
    for query_id, query_vector in queries.items():
        relevant_id = qrels[query_id]
        relevant_vector = documents[relevant_id]
        for doc_id, doc_vector in documents.items():
            if doc_id == relevant_id:
                continue
            harmful_scores += np.maximum(0.0, query_vector * doc_vector - query_vector * relevant_vector)
    ranked_dims = [
        int(index)
        for index in np.argsort(-harmful_scores)
        if harmful_scores[index] > 0
    ]
    masked_dims = ranked_dims[:mask_budget]
    return {
        "policy": "pairwise_harmful_alignment",
        "masked_dims": masked_dims,
        "dimension_scores": [float(score) for score in harmful_scores],
    }


def run_learned_mask_smoke(
    topic_count: int = 4,
    collapse_strength: float = 4.0,
) -> Dict[str, Any]:
    fixture = build_synthetic_collapse_fixture(topic_count=topic_count, collapse_strength=collapse_strength)
    learned = learn_pairwise_collapse_mask(
        fixture["queries"],
        fixture["documents"],
        fixture["qrels"],
        mask_budget=1,
    )
    baseline = evaluate_synthetic_collapse(fixture)
    learned_result = evaluate_synthetic_collapse(fixture, masked_dims=learned["masked_dims"])
    return {
        "learned_mask": learned,
        "expected_collapse_dim": fixture["collapse_dim"],
        "baseline": baseline,
        "learned": learned_result,
        "delta_ndcg_at_3": (
            learned_result["metrics"]["ndcg_at_3"] - baseline["metrics"]["ndcg_at_3"]
        ),
        "recovered": learned_result["metrics"]["ndcg_at_3"] > baseline["metrics"]["ndcg_at_3"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run learned-mask collapse smoke analysis")
    parser.add_argument("--topic-count", type=int, default=4)
    parser.add_argument("--collapse-strength", type=float, default=4.0)
    args = parser.parse_args()
    print(json.dumps(
        run_learned_mask_smoke(
            topic_count=args.topic_count,
            collapse_strength=args.collapse_strength,
        ),
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
