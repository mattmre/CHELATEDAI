"""Pure-NumPy margin-bound features over frozen embedding packs.

The calculations intentionally operate in the cosine-normalized geometry used
by the merged retrieval harness.  Fitting uses document pairs selected by the
pack's leakage-safe fit index; eval qrels are used only after fitting to choose
the relevant/competitor pair for each diagnostic inequality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np

from research.drift_recovery.artifacts import EmbeddingPack
from research.drift_recovery.estimator.margin_bound import (
    order_preservation_slack,
    predict_inversion,
)


RIDGE_REGULARIZATION = 1.0
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("expected a two-dimensional matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.where(norms > 0.0, norms, 1.0)


def _positive_eval_doc_ids(pack: EmbeddingPack) -> set:
    positives = set()
    for query_id in np.asarray(pack.query_ids, dtype=str):
        relevance = pack.qrels.get(str(query_id), {})
        positives.update(
            str(doc_id) for doc_id, score in relevance.items() if float(score) > 0.0
        )
    return positives


def leakage_safe_fit_indices(pack: EmbeddingPack) -> np.ndarray:
    """Return the audited safe index, rejecting eval-positive contamination."""

    if "leakage_safe_fit_idx" in pack.extra_arrays:
        indices = np.asarray(pack.extra_arrays["leakage_safe_fit_idx"], dtype=np.int64)
        source = "extra_arrays.leakage_safe_fit_idx"
    else:
        indices = np.asarray(pack.fit_idx, dtype=np.int64)
        source = "fit_idx"
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError(f"{source} must be a non-empty one-dimensional index")
    if np.any(indices < 0) or np.any(indices >= len(pack.doc_ids)):
        raise ValueError(f"{source} contains an out-of-bounds document index")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{source} contains duplicate document indices")

    positive_ids = _positive_eval_doc_ids(pack)
    fit_ids = set(np.asarray(pack.doc_ids, dtype=str)[indices].tolist())
    overlap = sorted(fit_ids.intersection(positive_ids))
    if overlap:
        raise ValueError(
            f"{source} is not leakage-safe: {len(overlap)} eval-positive documents in fit"
        )
    return indices.copy()


def fit_ridge_map(
    source: np.ndarray,
    target: np.ndarray,
    fit_idx: np.ndarray,
    regularization: float = RIDGE_REGULARIZATION,
) -> np.ndarray:
    """Fit the same no-intercept ridge map used by D1 on selected document pairs."""

    source_values = np.asarray(source, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    indices = np.asarray(fit_idx, dtype=np.int64)
    if source_values.shape != target_values.shape or source_values.ndim != 2:
        raise ValueError("ridge source and target must be same-shape matrices")
    if not np.isfinite(regularization) or float(regularization) <= 0.0:
        raise ValueError("regularization must be finite and positive")
    selected_source = source_values[indices]
    selected_target = target_values[indices]
    dimension = source_values.shape[1]
    return np.linalg.solve(
        selected_source.T @ selected_source
        + float(regularization) * np.eye(dimension, dtype=np.float64),
        selected_source.T @ selected_target,
    )


@dataclass(frozen=True)
class RidgeFit:
    fit_idx: np.ndarray
    ridge_map: np.ndarray
    corrected_documents: np.ndarray
    normalized_corrected_documents: np.ndarray
    normalized_oracle_documents: np.ndarray
    error_norms: np.ndarray


def fit_ridge_for_pack(
    pack: EmbeddingPack,
    regularization: float = RIDGE_REGULARIZATION,
) -> RidgeFit:
    """Fit from safe document pairs without passing qrels into the optimizer."""

    pack.validate()
    fit_idx = leakage_safe_fit_indices(pack)
    ridge_map = fit_ridge_map(pack.Do, pack.Dor, fit_idx, regularization=regularization)
    corrected = np.asarray(pack.Do, dtype=np.float64) @ ridge_map
    corrected_normalized = _normalize_rows(corrected)
    oracle_normalized = _normalize_rows(pack.Dor)
    errors = np.linalg.norm(corrected_normalized - oracle_normalized, axis=1)
    return RidgeFit(
        fit_idx=fit_idx,
        ridge_map=ridge_map,
        corrected_documents=corrected,
        normalized_corrected_documents=corrected_normalized,
        normalized_oracle_documents=oracle_normalized,
        error_norms=errors,
    )


def _quantile_summary(values: np.ndarray, prefix: str) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{prefix} values must be a finite non-empty vector")
    result = {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_std": float(np.std(array, ddof=0)),
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_max": float(np.max(array)),
    }
    for quantile in QUANTILES:
        label = int(round(100 * quantile))
        result[f"{prefix}_q{label:02d}"] = float(np.quantile(array, quantile))
    return result


def _relevance(pack: EmbeddingPack, query_id: str) -> Mapping[str, float]:
    relevance = pack.qrels.get(str(query_id))
    if relevance is None:
        raise ValueError(f"qrels missing eval query {query_id}")
    return relevance


def extract_regime_features(
    pack: EmbeddingPack,
    regularization: float = RIDGE_REGULARIZATION,
) -> Dict[str, Any]:
    """Extract deterministic per-query and aggregate margin-bound features."""

    fit = fit_ridge_for_pack(pack, regularization=regularization)
    queries = _normalize_rows(pack.Qd)
    query_norms = np.linalg.norm(queries, axis=1)
    oracle_scores = queries @ fit.normalized_oracle_documents.T
    document_ids = np.asarray(pack.doc_ids, dtype=str)
    doc_lookup = {doc_id: index for index, doc_id in enumerate(document_ids)}

    margins = []
    relevant_errors = []
    competitor_errors = []
    relevant_indices = []
    competitor_indices = []
    for row, query_id in enumerate(np.asarray(pack.query_ids, dtype=str)):
        relevant = [
            doc_lookup[str(doc_id)]
            for doc_id, score in _relevance(pack, query_id).items()
            if float(score) > 0.0 and str(doc_id) in doc_lookup
        ]
        if not relevant:
            raise ValueError(f"eval query {query_id} has no in-pack positive document")
        relevant_array = np.asarray(sorted(set(relevant)), dtype=np.int64)
        relevant_score_values = oracle_scores[row, relevant_array]
        relevant_index = int(relevant_array[int(np.argmax(relevant_score_values))])

        non_relevant_mask = np.ones(len(document_ids), dtype=bool)
        non_relevant_mask[relevant_array] = False
        if not np.any(non_relevant_mask):
            raise ValueError(f"eval query {query_id} has no non-relevant competitor")
        non_relevant_indices = np.flatnonzero(non_relevant_mask)
        competitor_index = int(
            non_relevant_indices[int(np.argmax(oracle_scores[row, non_relevant_indices]))]
        )

        margin = oracle_scores[row, relevant_index] - oracle_scores[row, competitor_index]
        margins.append(float(margin))
        relevant_errors.append(float(fit.error_norms[relevant_index]))
        competitor_errors.append(float(fit.error_norms[competitor_index]))
        relevant_indices.append(relevant_index)
        competitor_indices.append(competitor_index)

    margin_array = np.asarray(margins, dtype=np.float64)
    relevant_error_array = np.asarray(relevant_errors, dtype=np.float64)
    competitor_error_array = np.asarray(competitor_errors, dtype=np.float64)
    rhs = query_norms * (relevant_error_array + competitor_error_array)
    slack = order_preservation_slack(
        margin_array,
        query_norms,
        relevant_error_array,
        competitor_error_array,
    )
    inversions = predict_inversion(
        margin_array,
        query_norms,
        relevant_error_array,
        competitor_error_array,
    )

    summary: Dict[str, Any] = {
        "query_count": int(len(pack.query_ids)),
        "document_count": int(len(pack.doc_ids)),
        "embedding_dimension": int(pack.Do.shape[1]),
        "fit_count": int(len(fit.fit_idx)),
        "fit_index_source": (
            "extra_arrays.leakage_safe_fit_idx"
            if "leakage_safe_fit_idx" in pack.extra_arrays
            else "fit_idx"
        ),
        "eval_positive_docs_in_fit": 0,
        "ridge_regularization": float(regularization),
        "oracle_margin_sign_rate": float(np.mean(margin_array > 0.0)),
        "predicted_inversion_rate": float(np.mean(inversions)),
        "predicted_preservation_rate": float(np.mean(~inversions)),
    }
    summary.update(_quantile_summary(margin_array, "oracle_margin"))
    summary.update(_quantile_summary(fit.error_norms[fit.fit_idx], "fit_error_norm"))
    summary.update(_quantile_summary(fit.error_norms, "all_doc_error_norm"))
    summary.update(_quantile_summary(relevant_error_array, "relevant_error_norm"))
    summary.update(_quantile_summary(competitor_error_array, "competitor_error_norm"))
    summary.update(_quantile_summary(rhs, "bound_rhs"))
    summary.update(_quantile_summary(np.asarray(slack), "bound_slack"))

    return {
        "record_type": "recoverability_margin_features",
        "feature_version": 1,
        "regime": {
            "dataset": str(pack.metadata.get("dataset", "unknown")),
            "old_model": str(pack.metadata.get("models", {}).get("old", "unknown")),
            "swap_model": str(pack.metadata.get("models", {}).get("swap", "unknown")),
            "anchor_fraction": float(pack.metadata.get("anchor_fraction", float("nan"))),
            "seed": int(pack.metadata.get("seed", 0)),
        },
        "summary": summary,
        "per_query": {
            "query_ids": np.asarray(pack.query_ids, dtype=str).tolist(),
            "relevant_doc_ids": document_ids[np.asarray(relevant_indices)].tolist(),
            "competitor_doc_ids": document_ids[np.asarray(competitor_indices)].tolist(),
            "oracle_margin": margin_array.tolist(),
            "query_norm": query_norms.tolist(),
            "relevant_error_norm": relevant_error_array.tolist(),
            "competitor_error_norm": competitor_error_array.tolist(),
            "bound_rhs": rhs.tolist(),
            "bound_slack": np.asarray(slack, dtype=np.float64).tolist(),
            "predicted_inversion": np.asarray(inversions, dtype=bool).tolist(),
        },
    }
