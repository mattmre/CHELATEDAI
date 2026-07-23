"""Cluster-level retrieval-harm detector for Regime-C.

The scorer is trained on anchor-validation retrieval loss.  Its features are
computed from corrupted document geometry only; held-out eval qrels are used
only after prediction to measure AUPRC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from research.drift_recovery.harness_bridge import (
    aggregate_ndcg,
    per_query_ndcg_from_rankings,
)


def assert_query_partitions_disjoint(
    validation_query_ids: Sequence[str], eval_query_ids: Sequence[str]
) -> None:
    overlap = set(map(str, validation_query_ids)).intersection(map(str, eval_query_ids))
    if overlap:
        raise ValueError(f"anchor-validation/eval query leakage: {sorted(overlap)[:5]}")


def cluster_geometry_features(
    corrupted_documents: np.ndarray,
    assignments: np.ndarray,
    cluster_count: Optional[int] = None,
) -> np.ndarray:
    """Extract label-free cluster features from corrupted geometry only."""

    documents = np.asarray(corrupted_documents, dtype=np.float64)
    labels = np.asarray(assignments, dtype=np.int64)
    if documents.ndim != 2 or labels.shape != (len(documents),):
        raise ValueError("document/assignment shape mismatch")
    if not np.all(np.isfinite(documents)):
        raise ValueError("documents contain non-finite values")
    count = int(cluster_count if cluster_count is not None else labels.max() + 1)
    if count < 2 or np.any(labels < 0) or np.any(labels >= count):
        raise ValueError("assignments are outside the cluster range")
    centroids = np.vstack(
        [documents[labels == cluster].mean(axis=0) for cluster in range(count)]
    )
    centroid_distances = np.sqrt(
        np.maximum(
            np.sum(centroids * centroids, axis=1, keepdims=True)
            + np.sum(centroids * centroids, axis=1)[None, :]
            - 2.0 * centroids @ centroids.T,
            0.0,
        )
    )
    np.fill_diagonal(centroid_distances, np.inf)
    nearest_centroid = np.min(centroid_distances, axis=1)
    rows = []
    for cluster in range(count):
        members = documents[labels == cluster]
        if len(members) == 0:
            raise ValueError("empty clusters are not supported")
        residual = members - centroids[cluster]
        radius = np.linalg.norm(residual, axis=1)
        singular = np.linalg.svd(residual, compute_uv=False, full_matrices=False)
        energy = singular * singular
        top_energy = float(energy[0] / energy.sum()) if energy.sum() > 0.0 else 0.0
        member_norms = np.linalg.norm(members, axis=1)
        centroid_norm = float(np.linalg.norm(centroids[cluster]))
        cosine_to_centroid = (
            members @ centroids[cluster]
            / np.maximum(member_norms * max(centroid_norm, 1e-12), 1e-12)
        )
        rows.append(
            [
                len(members) / len(documents),
                float(radius.mean()),
                float(radius.std()),
                float(np.quantile(radius, 0.9)),
                centroid_norm,
                float(cosine_to_centroid.mean()),
                top_energy,
                float(nearest_centroid[cluster]),
            ]
        )
    features = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(features)):
        raise ValueError("cluster features contain non-finite values")
    return features


def cluster_counterfactual_harm(
    clean_documents: np.ndarray,
    corrupted_documents: np.ndarray,
    assignments: np.ndarray,
    query_vectors: np.ndarray,
    query_ids: Sequence[str],
    doc_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, float]],
    k: int = 10,
    cluster_count: Optional[int] = None,
) -> np.ndarray:
    """Measure each cluster's actual clean-to-corrupted retrieval loss.

    Each row is a counterfactual in which exactly that cluster is replaced by
    its corrupted vectors.  This attributes loss to document clusters rather
    than to the injection knob or result-list overlap.
    """

    clean = np.asarray(clean_documents, dtype=np.float64)
    corrupted = np.asarray(corrupted_documents, dtype=np.float64)
    labels = np.asarray(assignments, dtype=np.int64)
    if clean.shape != corrupted.shape or labels.shape != (len(clean),):
        raise ValueError("clean/corrupted/assignment shape mismatch")
    queries = np.asarray(query_vectors, dtype=np.float64)
    if queries.ndim != 2 or queries.shape[1] != clean.shape[1] or len(queries) != len(query_ids):
        raise ValueError("query vectors do not match IDs or document dimension")
    clean_norm = np.linalg.norm(clean, axis=1, keepdims=True)
    corrupted_norm = np.linalg.norm(corrupted, axis=1, keepdims=True)
    query_norm = np.linalg.norm(queries, axis=1, keepdims=True)
    normalized_queries = queries / np.where(query_norm > 0.0, query_norm, 1.0)
    clean_scores = normalized_queries @ (
        clean / np.where(clean_norm > 0.0, clean_norm, 1.0)
    ).T
    corrupted_scores = normalized_queries @ (
        corrupted / np.where(corrupted_norm > 0.0, corrupted_norm, 1.0)
    ).T
    clean_rankings = np.argsort(-clean_scores, axis=1, kind="mergesort")[:, : min(k, len(clean))]
    clean_score = aggregate_ndcg(
        per_query_ndcg_from_rankings(clean_rankings, query_ids, doc_ids, qrels, k=k)
    )
    count = int(cluster_count if cluster_count is not None else labels.max() + 1)
    losses = np.zeros(count, dtype=np.float64)
    for cluster in range(count):
        mask = labels == cluster
        counterfactual_scores = clean_scores.copy()
        counterfactual_scores[:, mask] = corrupted_scores[:, mask]
        rankings = np.argsort(
            -counterfactual_scores, axis=1, kind="mergesort"
        )[:, : min(k, len(clean))]
        damaged_score = aggregate_ndcg(
            per_query_ndcg_from_rankings(rankings, query_ids, doc_ids, qrels, k=k)
        )
        losses[cluster] = max(0.0, clean_score - damaged_score)
    return losses


@dataclass
class ClusterHarmScorer:
    regularization: float = 1.0
    learning_rate: float = 0.1
    steps: int = 500
    mean_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None
    weights_: Optional[np.ndarray] = None
    constant_probability_: Optional[float] = None

    def fit(
        self,
        features: np.ndarray,
        harm_losses: np.ndarray,
        harm_threshold: float = 1e-8,
    ) -> "ClusterHarmScorer":
        matrix = np.asarray(features, dtype=np.float64)
        losses = np.asarray(harm_losses, dtype=np.float64)
        if matrix.ndim != 2 or losses.shape != (len(matrix),):
            raise ValueError("feature/harm shape mismatch")
        labels = (losses > float(harm_threshold)).astype(np.float64)
        if len(np.unique(labels)) == 1:
            self.constant_probability_ = float(labels[0])
            self.mean_ = np.zeros(matrix.shape[1], dtype=np.float64)
            self.scale_ = np.ones(matrix.shape[1], dtype=np.float64)
            self.weights_ = np.zeros(matrix.shape[1] + 1, dtype=np.float64)
            return self
        self.mean_ = matrix.mean(axis=0)
        self.scale_ = matrix.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        standardized = (matrix - self.mean_) / self.scale_
        design = np.column_stack([np.ones(len(matrix)), standardized])
        weights = np.zeros(design.shape[1], dtype=np.float64)
        penalty = np.r_[0.0, np.ones(design.shape[1] - 1)]
        for _ in range(int(self.steps)):
            logits = np.clip(design @ weights, -40.0, 40.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            gradient = design.T @ (probabilities - labels) / len(labels)
            gradient += float(self.regularization) * penalty * weights / len(labels)
            weights -= float(self.learning_rate) * gradient
        self.weights_ = weights
        self.constant_probability_ = None
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or self.weights_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("harm scorer is not fitted or feature shape is invalid")
        if self.constant_probability_ is not None:
            return np.full(len(matrix), self.constant_probability_, dtype=np.float64)
        design = np.column_stack([np.ones(len(matrix)), (matrix - self.mean_) / self.scale_])
        logits = np.clip(design @ self.weights_, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))


def average_precision(
    harm_losses: np.ndarray,
    probabilities: np.ndarray,
    harm_threshold: float = 1e-8,
) -> Tuple[Optional[float], str]:
    labels = np.asarray(harm_losses, dtype=np.float64) > float(harm_threshold)
    scores = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != scores.shape:
        raise ValueError("harm/probability shape mismatch")
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        return None, "undefined_single_class"
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int64)
    true_positive = 0
    retrieved = 0
    previous_recall = 0.0
    area = 0.0
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        true_positive += int(sorted_labels[start:end].sum())
        retrieved = end
        recall = true_positive / positives
        precision = true_positive / retrieved
        area += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return float(area), "ok"


def select_gate_threshold(
    harm_losses: np.ndarray,
    probabilities: np.ndarray,
    candidates: Sequence[float],
    harm_threshold: float = 1e-8,
) -> float:
    """Select a gate on validation labels only using F1 with stable ties."""

    labels = np.asarray(harm_losses) > float(harm_threshold)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not np.any(labels):
        return 1.0
    calibrated_candidates = set(map(float, candidates))
    calibrated_candidates.update(float(value) for value in np.unique(probabilities))
    best = None
    for threshold in sorted(calibrated_candidates):
        predicted = probabilities >= threshold
        true_positive = int(np.count_nonzero(predicted & labels))
        false_positive = int(np.count_nonzero(predicted & ~labels))
        false_negative = int(np.count_nonzero(~predicted & labels))
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        # Stable ties prefer the more conservative (higher) gate.
        candidate = (f1, threshold)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise ValueError("at least one gate threshold is required")
    return float(best[1])


def evaluate_detector(
    *,
    features: np.ndarray,
    validation_harm_losses: np.ndarray,
    eval_harm_losses: np.ndarray,
    validation_query_ids: Sequence[str],
    eval_query_ids: Sequence[str],
    gate_candidates: Sequence[float],
    harm_threshold: float = 1e-8,
) -> Mapping[str, Any]:
    """Fit on validation harm, then evaluate already-computed held-out harm."""

    assert_query_partitions_disjoint(validation_query_ids, eval_query_ids)
    scorer = ClusterHarmScorer().fit(features, validation_harm_losses, harm_threshold=harm_threshold)
    probabilities = scorer.predict_proba(features)
    selected_gate = select_gate_threshold(
        validation_harm_losses, probabilities, gate_candidates, harm_threshold=harm_threshold
    )
    auprc, status = average_precision(eval_harm_losses, probabilities, harm_threshold=harm_threshold)
    return {
        "record_type": "cluster_harm_detector_evaluation",
        "label_definition": f"counterfactual cluster NDCG drop > {float(harm_threshold):.12g}",
        "auprc": auprc,
        "auprc_status": status,
        "selected_gate_threshold": selected_gate,
        "probabilities": probabilities,
        "validation_harm_losses": np.asarray(validation_harm_losses, dtype=np.float64),
        "eval_harm_losses": np.asarray(eval_harm_losses, dtype=np.float64),
        "validation_positive_clusters": int(
            np.count_nonzero(np.asarray(validation_harm_losses) > float(harm_threshold))
        ),
        "eval_positive_clusters": int(
            np.count_nonzero(np.asarray(eval_harm_losses) > float(harm_threshold))
        ),
    }
