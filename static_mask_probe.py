"""Train/holdout static dimension-mask probe for real retrieval slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

from benchmark_utils import load_mteb_data
from embedding_backend import create_embedding_backend
from query_reformulator import query_lexical_features
from run_road_course_campaign import evaluate_rankings
from run_thousand_query_tuning import query_metric_row
from run_thousand_query_tuning import select_query_window


CONDITIONAL_MASK_FEATURES = [
    "baseline_score_margin",
    "baseline_top_score",
    "query_norm",
    "query_token_count",
    "query_stopword_ratio",
    "query_negation_count",
    "query_claim_cue_count",
]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def rank_by_cosine(
    query_embeddings: Mapping[str, np.ndarray],
    doc_embeddings: Mapping[str, np.ndarray],
    *,
    mask: np.ndarray | None = None,
) -> Dict[str, List[str]]:
    doc_ids = list(doc_embeddings)
    doc_matrix = np.vstack([doc_embeddings[doc_id] for doc_id in doc_ids])
    if mask is not None:
        doc_matrix = doc_matrix * mask
    doc_matrix = _normalize(doc_matrix)
    rankings: Dict[str, List[str]] = {}
    for query_id, query_vector in query_embeddings.items():
        query = query_vector * mask if mask is not None else query_vector
        query = _normalize(query.reshape(1, -1))[0]
        scores = doc_matrix @ query
        order = np.argsort(-scores)
        rankings[query_id] = [doc_ids[index] for index in order[:10]]
    return rankings


def cosine_score_details(
    query_embeddings: Mapping[str, np.ndarray],
    doc_embeddings: Mapping[str, np.ndarray],
    *,
    mask: np.ndarray | None = None,
) -> Dict[str, Dict[str, Any]]:
    doc_ids = list(doc_embeddings)
    doc_matrix = np.vstack([doc_embeddings[doc_id] for doc_id in doc_ids])
    if mask is not None:
        doc_matrix = doc_matrix * mask
    doc_matrix = _normalize(doc_matrix)
    details: Dict[str, Dict[str, Any]] = {}
    for query_id, query_vector in query_embeddings.items():
        query = query_vector * mask if mask is not None else query_vector
        query = _normalize(query.reshape(1, -1))[0]
        scores = doc_matrix @ query
        order = np.argsort(-scores)
        top_scores = [float(scores[index]) for index in order[:2]]
        details[query_id] = {
            "ranking": [doc_ids[index] for index in order[:10]],
            "top_score": top_scores[0] if top_scores else 0.0,
            "second_score": top_scores[1] if len(top_scores) > 1 else 0.0,
            "score_margin": (top_scores[0] - top_scores[1]) if len(top_scores) > 1 else 0.0,
        }
    return details


def learn_harmful_dimension_mask(
    query_embeddings: Mapping[str, np.ndarray],
    doc_embeddings: Mapping[str, np.ndarray],
    qrels: Mapping[str, Mapping[str, float]],
    *,
    mask_fraction: float = 0.02,
    distractor_k: int = 5,
) -> Dict[str, Any]:
    """Learn dimensions where top distractors out-contribute relevant docs."""

    if not 0.0 < mask_fraction < 1.0:
        raise ValueError("mask_fraction must be between 0 and 1")
    vector_size = len(next(iter(doc_embeddings.values())))
    harmful = np.zeros(vector_size, dtype=float)
    baseline_rankings = rank_by_cosine(query_embeddings, doc_embeddings)
    examples = 0
    for query_id, query_vector in query_embeddings.items():
        relevant_ids = [
            doc_id for doc_id, score in qrels.get(query_id, {}).items()
            if score > 0 and doc_id in doc_embeddings
        ]
        if not relevant_ids:
            continue
        relevant_vector = np.mean([doc_embeddings[doc_id] for doc_id in relevant_ids], axis=0)
        distractor_ids = [
            doc_id for doc_id in baseline_rankings[query_id]
            if doc_id not in relevant_ids
        ][:distractor_k]
        if not distractor_ids:
            continue
        distractor_vector = np.mean([doc_embeddings[doc_id] for doc_id in distractor_ids], axis=0)
        harmful += np.maximum(0.0, query_vector * distractor_vector - query_vector * relevant_vector)
        examples += 1
    mask_count = max(1, int(round(vector_size * mask_fraction)))
    masked_dims = [int(index) for index in np.argsort(-harmful)[:mask_count] if harmful[index] > 0]
    mask = np.ones(vector_size, dtype=float)
    for dim in masked_dims:
        mask[dim] = 0.0
    return {
        "policy": "supervised_harmful_dimension_mask",
        "mask_fraction": mask_fraction,
        "mask_count": len(masked_dims),
        "masked_dims": masked_dims,
        "training_examples": examples,
        "dimension_scores": [float(score) for score in harmful],
        "mask": mask,
    }


def build_conditional_mask_examples(
    query_embeddings: Mapping[str, np.ndarray],
    doc_embeddings: Mapping[str, np.ndarray],
    qrels: Mapping[str, Mapping[str, float]],
    mask: np.ndarray,
    *,
    query_texts: Mapping[str, str] | None = None,
) -> List[Dict[str, Any]]:
    baseline_details = cosine_score_details(query_embeddings, doc_embeddings)
    masked_details = cosine_score_details(query_embeddings, doc_embeddings, mask=mask)
    examples = []
    for query_id, relevance in qrels.items():
        baseline_metrics = query_metric_row(query_id, baseline_details[query_id]["ranking"], relevance)
        masked_metrics = query_metric_row(query_id, masked_details[query_id]["ranking"], relevance)
        lexical = query_lexical_features((query_texts or {}).get(query_id, ""))
        examples.append({
            "query_id": query_id,
            "delta_ndcg_at_10": masked_metrics["ndcg_at_10"] - baseline_metrics["ndcg_at_10"],
            "delta_mrr": masked_metrics["mrr"] - baseline_metrics["mrr"],
            "baseline_rank": baseline_metrics["first_relevant_rank"],
            "masked_rank": masked_metrics["first_relevant_rank"],
            "baseline_score_margin": baseline_details[query_id]["score_margin"],
            "baseline_top_score": baseline_details[query_id]["top_score"],
            "query_norm": float(np.linalg.norm(query_embeddings[query_id])),
            "query_token_count": lexical["token_count"],
            "query_stopword_ratio": lexical["stopword_ratio"],
            "query_negation_count": lexical["negation_count"],
            "query_claim_cue_count": lexical["claim_cue_count"],
        })
    return examples


def _gate_matches(example: Dict[str, Any], gate: Dict[str, Any]) -> bool:
    if gate.get("type") == "linear_classifier":
        score = _classifier_score(example, gate)
        return score >= float(gate["threshold"])
    value = example.get(gate["feature"])
    if value is None:
        return False
    if gate["operator"] == "<=":
        return float(value) <= float(gate["threshold"])
    if gate["operator"] == ">=":
        return float(value) >= float(gate["threshold"])
    raise ValueError(f"unsupported operator {gate['operator']}")


def _classifier_score(example: Dict[str, Any], gate: Dict[str, Any]) -> float:
    values = np.array([float(example.get(feature, 0.0)) for feature in gate["features"]], dtype=float)
    means = np.array(gate["means"], dtype=float)
    scales = np.array(gate["scales"], dtype=float)
    weights = np.array(gate["weights"], dtype=float)
    normalized = (values - means) / np.maximum(scales, 1e-12)
    logit = float(normalized @ weights + float(gate["intercept"]))
    return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))


def _summarize_gate_examples(gate: Dict[str, Any], examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    matched = [example for example in examples if _gate_matches(example, gate)]
    deltas = [float(example["delta_ndcg_at_10"]) for example in matched]
    return {
        "support": len(matched),
        "mean_delta_ndcg_at_10": float(np.mean(deltas)) if deltas else 0.0,
        "positive_examples": sum(delta > 0.001 for delta in deltas),
        "negative_examples": sum(delta < -0.001 for delta in deltas),
    }


def train_conditional_mask_gate(
    examples: List[Dict[str, Any]],
    *,
    min_support: int = 3,
    min_mean_delta: float = 0.001,
    max_negative_examples: int = 0,
) -> Dict[str, Any]:
    """Train a fail-closed one-feature gate for applying a learned mask."""

    candidates = []
    for feature in CONDITIONAL_MASK_FEATURES:
        values = [float(example[feature]) for example in examples if example.get(feature) is not None]
        if not values:
            continue
        for threshold in sorted(set(np.quantile(values, [0.25, 0.5, 0.75]).tolist())):
            for operator in ("<=", ">="):
                gate = {"feature": feature, "operator": operator, "threshold": float(threshold)}
                record = {
                    "gate": gate,
                    **_summarize_gate_examples(gate, examples),
                }
                record["accepted"] = (
                    record["support"] >= min_support
                    and record["mean_delta_ndcg_at_10"] >= min_mean_delta
                    and record["negative_examples"] <= max_negative_examples
                )
                candidates.append(record)
    accepted = [candidate for candidate in candidates if candidate["accepted"]]
    accepted.sort(key=lambda item: (-item["mean_delta_ndcg_at_10"], -item["support"], item["gate"]["feature"]))
    return {
        "policy": "conditional_static_mask_gate",
        "gate": accepted[0]["gate"] if accepted else None,
        "accepted": accepted[:5],
        "rejected_count": len(candidates) - len(accepted),
        "criteria": {
            "min_support": min_support,
            "min_mean_delta": min_mean_delta,
            "max_negative_examples": max_negative_examples,
        },
    }


def train_regularized_conditional_mask_gate(
    examples: List[Dict[str, Any]],
    *,
    validation_fraction: float = 0.4,
    min_support: int = 3,
    min_mean_delta: float = 0.001,
    max_negative_examples: int = 0,
) -> Dict[str, Any]:
    """Train a conditional mask gate that must pass an internal validation split."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(examples) < 4:
        return {
            "policy": "regularized_conditional_static_mask_gate",
            "gate": None,
            "accepted": [],
            "rejected_count": 0,
            "criteria": {
                "validation_fraction": validation_fraction,
                "min_support": min_support,
                "min_mean_delta": min_mean_delta,
                "max_negative_examples": max_negative_examples,
            },
        }
    split_index = max(1, int(round(len(examples) * (1.0 - validation_fraction))))
    train_examples = examples[:split_index]
    validation_examples = examples[split_index:]
    train_gate = train_conditional_mask_gate(
        train_examples,
        min_support=min_support,
        min_mean_delta=min_mean_delta,
        max_negative_examples=max_negative_examples,
    )
    accepted = []
    for candidate in train_gate["accepted"]:
        validation_summary = _summarize_gate_examples(candidate["gate"], validation_examples)
        passed_validation = (
            validation_summary["support"] >= max(1, min_support // 2)
            and validation_summary["mean_delta_ndcg_at_10"] >= 0.0
            and validation_summary["negative_examples"] <= max_negative_examples
        )
        if passed_validation:
            accepted.append({
                "gate": candidate["gate"],
                "train": {
                    key: candidate[key]
                    for key in ["support", "mean_delta_ndcg_at_10", "positive_examples", "negative_examples"]
                },
                "validation": validation_summary,
                "accepted": True,
            })
    accepted.sort(
        key=lambda item: (
            -item["validation"]["mean_delta_ndcg_at_10"],
            -item["train"]["mean_delta_ndcg_at_10"],
            -item["validation"]["support"],
        )
    )
    return {
        "policy": "regularized_conditional_static_mask_gate",
        "gate": accepted[0]["gate"] if accepted else None,
        "accepted": accepted[:5],
        "rejected_count": len(train_gate["accepted"]) - len(accepted) + train_gate["rejected_count"],
        "criteria": {
            "validation_fraction": validation_fraction,
            "min_support": min_support,
            "min_mean_delta": min_mean_delta,
            "max_negative_examples": max_negative_examples,
        },
    }


def _classifier_matrix(examples: List[Dict[str, Any]], features: List[str]) -> np.ndarray:
    return np.array(
        [[float(example.get(feature, 0.0)) for feature in features] for example in examples],
        dtype=float,
    )


def _train_logistic_weights(
    train_examples: List[Dict[str, Any]],
    features: List[str],
    *,
    positive_delta: float,
    l2_penalty: float,
    learning_rate: float = 0.08,
    epochs: int = 300,
) -> Dict[str, Any] | None:
    matrix = _classifier_matrix(train_examples, features)
    labels = np.array(
        [1.0 if float(example["delta_ndcg_at_10"]) > positive_delta else 0.0 for example in train_examples],
        dtype=float,
    )
    positive_count = int(np.sum(labels))
    negative_count = int(len(labels) - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None
    means = np.mean(matrix, axis=0)
    scales = np.std(matrix, axis=0)
    normalized = (matrix - means) / np.maximum(scales, 1e-12)
    weights = np.zeros(normalized.shape[1], dtype=float)
    intercept = 0.0
    sample_weights = np.where(
        labels > 0.0,
        len(labels) / (2.0 * positive_count),
        len(labels) / (2.0 * negative_count),
    )
    for _epoch in range(epochs):
        logits = np.clip(normalized @ weights + intercept, -40.0, 40.0)
        predictions = 1.0 / (1.0 + np.exp(-logits))
        errors = (predictions - labels) * sample_weights
        weights -= learning_rate * ((normalized.T @ errors) / len(labels) + l2_penalty * weights)
        intercept -= learning_rate * float(np.mean(errors))
    return {
        "features": features,
        "means": [float(value) for value in means],
        "scales": [float(value) for value in np.maximum(scales, 1e-12)],
        "weights": [float(value) for value in weights],
        "intercept": float(intercept),
        "positive_count": positive_count,
        "negative_count": negative_count,
    }


def train_classifier_conditional_mask_gate(
    examples: List[Dict[str, Any]],
    *,
    validation_fraction: float = 0.4,
    min_support: int = 3,
    min_mean_delta: float = 0.001,
    max_negative_examples: int = 0,
    positive_delta: float = 0.001,
    min_positive_examples: int = 4,
    l2_penalty: float = 0.08,
) -> Dict[str, Any]:
    """Train a small regularized classifier gate with an internal validation split."""

    criteria = {
        "validation_fraction": validation_fraction,
        "min_support": min_support,
        "min_mean_delta": min_mean_delta,
        "max_negative_examples": max_negative_examples,
        "positive_delta": positive_delta,
        "min_positive_examples": min_positive_examples,
        "l2_penalty": l2_penalty,
    }
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(examples) < 8:
        return {
            "policy": "classifier_conditional_static_mask_gate",
            "gate": None,
            "accepted": [],
            "rejected_count": 0,
            "criteria": criteria,
        }
    split_index = max(1, int(round(len(examples) * (1.0 - validation_fraction))))
    train_examples = examples[:split_index]
    validation_examples = examples[split_index:]
    model = _train_logistic_weights(
        train_examples,
        CONDITIONAL_MASK_FEATURES,
        positive_delta=positive_delta,
        l2_penalty=l2_penalty,
    )
    if model is None:
        return {
            "policy": "classifier_conditional_static_mask_gate",
            "gate": None,
            "accepted": [],
            "rejected_count": 0,
            "criteria": criteria,
            "training_summary": {
                "positive_examples": sum(float(example["delta_ndcg_at_10"]) > positive_delta for example in train_examples),
                "train_examples": len(train_examples),
            },
        }
    if model["positive_count"] < min_positive_examples:
        return {
            "policy": "classifier_conditional_static_mask_gate",
            "gate": None,
            "accepted": [],
            "rejected_count": 0,
            "criteria": criteria,
            "training_summary": {
                "positive_examples": model["positive_count"],
                "negative_examples": model["negative_count"],
                "train_examples": len(train_examples),
                "validation_examples": len(validation_examples),
            },
        }
    train_scores = []
    for example in train_examples:
        gate = {"type": "linear_classifier", "threshold": 0.0, **model}
        train_scores.append(_classifier_score(example, gate))
    thresholds = sorted(set(np.quantile(train_scores, [0.5, 0.65, 0.8, 0.9]).tolist()), reverse=True)
    accepted = []
    rejected_count = 0
    for threshold in thresholds:
        gate = {"type": "linear_classifier", "threshold": float(threshold), **model}
        train_summary = _summarize_gate_examples(gate, train_examples)
        validation_summary = _summarize_gate_examples(gate, validation_examples)
        train_passed = (
            train_summary["support"] >= min_support
            and train_summary["mean_delta_ndcg_at_10"] >= min_mean_delta
            and train_summary["negative_examples"] <= max_negative_examples
        )
        validation_passed = (
            validation_summary["support"] >= max(1, min_support // 2)
            and validation_summary["mean_delta_ndcg_at_10"] >= min_mean_delta
            and validation_summary["negative_examples"] <= max_negative_examples
        )
        record = {
            "gate": gate,
            "train": train_summary,
            "validation": validation_summary,
            "accepted": train_passed and validation_passed,
        }
        if record["accepted"]:
            accepted.append(record)
        else:
            rejected_count += 1
    accepted.sort(
        key=lambda item: (
            -item["validation"]["mean_delta_ndcg_at_10"],
            -item["train"]["mean_delta_ndcg_at_10"],
            -item["validation"]["support"],
            -item["train"]["support"],
        )
    )
    return {
        "policy": "classifier_conditional_static_mask_gate",
        "gate": accepted[0]["gate"] if accepted else None,
        "accepted": accepted[:5],
        "rejected_count": rejected_count,
        "criteria": criteria,
        "training_summary": {
            "positive_examples": model["positive_count"],
            "negative_examples": model["negative_count"],
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
        },
    }


def evaluate_conditional_mask(
    query_embeddings: Mapping[str, np.ndarray],
    doc_embeddings: Mapping[str, np.ndarray],
    qrels: Mapping[str, Mapping[str, float]],
    mask: np.ndarray,
    gate: Dict[str, Any] | None,
    *,
    query_texts: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    baseline_details = cosine_score_details(query_embeddings, doc_embeddings)
    masked_details = cosine_score_details(query_embeddings, doc_embeddings, mask=mask)
    examples = build_conditional_mask_examples(
        query_embeddings,
        doc_embeddings,
        qrels,
        mask,
        query_texts=query_texts,
    )
    rankings = {}
    applied = 0
    for example in examples:
        query_id = example["query_id"]
        if gate is not None and _gate_matches(example, gate):
            rankings[query_id] = masked_details[query_id]["ranking"]
            applied += 1
        else:
            rankings[query_id] = baseline_details[query_id]["ranking"]
    return {
        "metrics": evaluate_rankings(rankings, qrels),
        "applied_queries": applied,
        "total_queries": len(examples),
    }


def _split_mapping(mapping: Mapping[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    key_set = set(keys)
    return {key: value for key, value in mapping.items() if key in key_set}


def _embed_mapping(backend, mapping: Mapping[str, str]) -> Dict[str, np.ndarray]:
    ids = list(mapping)
    vectors = backend.embed_raw([mapping[item_id] for item_id in ids])
    return {item_id: vectors[index] for index, item_id in enumerate(ids)}


def run_static_mask_probe(
    *,
    task: str = "SciFact",
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    query_offset: int = 0,
    max_queries: int = 100,
    train_queries: int = 50,
    sample_docs: int = 200,
    seed: int = 140,
    mask_fraction: float = 0.02,
    conditional: bool = False,
    regularized_gate: bool = False,
    classifier_gate: bool = False,
    gate_validation_fraction: float = 0.4,
) -> Dict[str, Any]:
    corpus, queries, qrels = load_mteb_data(task)
    selected_corpus, selected_queries, selected_qrels = select_query_window(
        corpus,
        queries,
        qrels,
        query_offset=query_offset,
        max_queries=max_queries,
        sample_docs=sample_docs,
        seed=seed,
    )
    query_ids = list(selected_queries)
    if not 0 < train_queries < len(query_ids):
        raise ValueError("train_queries must leave at least one holdout query")
    train_ids = query_ids[:train_queries]
    holdout_ids = query_ids[train_queries:]
    backend = create_embedding_backend(model)
    doc_embeddings = _embed_mapping(backend, selected_corpus)
    query_embeddings = _embed_mapping(backend, selected_queries)
    train_query_embeddings = _split_mapping(query_embeddings, train_ids)
    holdout_query_embeddings = _split_mapping(query_embeddings, holdout_ids)
    train_qrels = _split_mapping(selected_qrels, train_ids)
    holdout_qrels = _split_mapping(selected_qrels, holdout_ids)

    learned = learn_harmful_dimension_mask(
        train_query_embeddings,
        doc_embeddings,
        train_qrels,
        mask_fraction=mask_fraction,
    )
    train_baseline = evaluate_rankings(rank_by_cosine(train_query_embeddings, doc_embeddings), train_qrels)
    train_masked = evaluate_rankings(
        rank_by_cosine(train_query_embeddings, doc_embeddings, mask=learned["mask"]),
        train_qrels,
    )
    holdout_baseline = evaluate_rankings(rank_by_cosine(holdout_query_embeddings, doc_embeddings), holdout_qrels)
    holdout_masked = evaluate_rankings(
        rank_by_cosine(holdout_query_embeddings, doc_embeddings, mask=learned["mask"]),
        holdout_qrels,
    )
    result = {
        "task": task,
        "model": model,
        "query_offset": query_offset,
        "max_queries": max_queries,
        "train_queries": len(train_ids),
        "holdout_queries": len(holdout_ids),
        "sample_docs": len(selected_corpus),
        "seed": seed,
        "mask": {
            key: value
            for key, value in learned.items()
            if key not in {"mask", "dimension_scores"}
        },
        "train": {
            "baseline": train_baseline,
            "masked": train_masked,
            "delta_ndcg_at_10": train_masked["ndcg_at_10"] - train_baseline["ndcg_at_10"],
        },
        "holdout": {
            "baseline": holdout_baseline,
            "masked": holdout_masked,
            "delta_ndcg_at_10": holdout_masked["ndcg_at_10"] - holdout_baseline["ndcg_at_10"],
        },
        "promotion_candidate": (
            holdout_masked["ndcg_at_10"] - holdout_baseline["ndcg_at_10"] > 0.001
            and train_masked["ndcg_at_10"] >= train_baseline["ndcg_at_10"]
        ),
    }
    if conditional or regularized_gate or classifier_gate:
        train_examples = build_conditional_mask_examples(
            train_query_embeddings,
            doc_embeddings,
            train_qrels,
            learned["mask"],
            query_texts=_split_mapping(selected_queries, train_ids),
        )
        if classifier_gate:
            gate = train_classifier_conditional_mask_gate(
                train_examples,
                validation_fraction=gate_validation_fraction,
            )
        elif regularized_gate:
            gate = train_regularized_conditional_mask_gate(
                train_examples,
                validation_fraction=gate_validation_fraction,
            )
        else:
            gate = train_conditional_mask_gate(train_examples)
        train_conditional = evaluate_conditional_mask(
            train_query_embeddings,
            doc_embeddings,
            train_qrels,
            learned["mask"],
            gate["gate"],
            query_texts=_split_mapping(selected_queries, train_ids),
        )
        holdout_conditional = evaluate_conditional_mask(
            holdout_query_embeddings,
            doc_embeddings,
            holdout_qrels,
            learned["mask"],
            gate["gate"],
            query_texts=_split_mapping(selected_queries, holdout_ids),
        )
        result["conditional"] = {
            "gate": gate,
            "train": {
                **train_conditional,
                "delta_ndcg_at_10": (
                    train_conditional["metrics"]["ndcg_at_10"] - train_baseline["ndcg_at_10"]
                ),
            },
            "holdout": {
                **holdout_conditional,
                "delta_ndcg_at_10": (
                    holdout_conditional["metrics"]["ndcg_at_10"] - holdout_baseline["ndcg_at_10"]
                ),
            },
        }
        result["conditional_promotion_candidate"] = (
            gate["gate"] is not None
            and result["conditional"]["holdout"]["delta_ndcg_at_10"] > 0.001
            and result["conditional"]["train"]["delta_ndcg_at_10"] >= 0.0
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a train/holdout static dimension-mask probe")
    parser.add_argument("--task", default="SciFact")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--train-queries", type=int, default=50)
    parser.add_argument("--sample-docs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=140)
    parser.add_argument("--mask-fraction", type=float, default=0.02)
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--regularized-gate", action="store_true")
    parser.add_argument("--classifier-gate", action="store_true")
    parser.add_argument("--gate-validation-fraction", type=float, default=0.4)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = run_static_mask_probe(
        task=args.task,
        model=args.model,
        query_offset=args.query_offset,
        max_queries=args.max_queries,
        train_queries=args.train_queries,
        sample_docs=args.sample_docs,
        seed=args.seed,
        mask_fraction=args.mask_fraction,
        conditional=args.conditional or args.regularized_gate or args.classifier_gate,
        regularized_gate=args.regularized_gate,
        classifier_gate=args.classifier_gate,
        gate_validation_fraction=args.gate_validation_fraction,
    )
    text = json.dumps(result, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
