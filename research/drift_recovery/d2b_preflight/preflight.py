"""CPU preflight that tries to falsify an Obj B nonlinear residual.

The experiment is deliberately generous to the linear alternative.  All
routing is inferred from corrupted document vectors, a single anchor-derived
detector gate is shared by every correction method, and ridge regularization
(plus low-rank rank) is selected on an anchor-only development split.  Eval
queries are used exactly once, after the fits and method choices are frozen.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from research.drift_recovery.methods._common import deterministic_kmeans
from research.drift_recovery.regimes.sparse_local_nonaffine import (
    generate_sparse_local_regime,
)


CLOSE_ONE_LINER = (
    "Obj B fails constructively — under the residual>=0.05 gate no cell admits. "
    "The residuals are small mainly because the absolute recoverable NDCG gaps are "
    "small, and the dev-selected gated local ridge frequently does not even beat the "
    "no-op floor. This does NOT show local ridge dominates chelation; it shows the "
    "synthetic preflight could not construct a discriminating home-turf residual under "
    "small-magnitude non-affine warps, so no GPU chelation run is justified."
)
DEFAULT_LAMBDAS = (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)
DEFAULT_RANKS = (1, 2, 4, 8)


@dataclass(frozen=True)
class RidgeMap:
    """Affine ridge map, optionally restricted to a residual-map rank."""

    weights: np.ndarray
    rank: Optional[int]
    regularization: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = _matrix(values, "values")
        augmented = np.column_stack((matrix, np.ones(len(matrix))))
        return augmented @ self.weights


@dataclass(frozen=True)
class ProcrustesMap:
    rotation: np.ndarray
    source_mean: np.ndarray
    target_mean: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = _matrix(values, "values")
        return (matrix - self.source_mean) @ self.rotation + self.target_mean


@dataclass
class LocalRidgeModel:
    maps: Dict[int, RidgeMap]
    selected_lambdas: Dict[int, float]
    selected_rank: Optional[int]
    dev_mse: float
    training_audit: Dict[str, object]


@dataclass(frozen=True)
class SelectionAudit:
    fit_indices: Tuple[int, ...]
    dev_indices: Tuple[int, ...]
    eval_indices: Tuple[int, ...]
    selected_lambda: float
    selected_rank: Optional[int]
    dev_mse: float


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not len(matrix) or not matrix.shape[1]:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def infer_corrupted_clusters(
    corrupted_docs: np.ndarray, k: int, seed: int = 0
) -> np.ndarray:
    """Infer routing from corrupted vectors only; clean labels are not accepted."""

    del seed  # deterministic farthest-first k-means has no randomized state.
    _centroids, assignments = deterministic_kmeans(
        _matrix(corrupted_docs, "corrupted_docs"), int(k), iterations=50
    )
    return assignments


def fit_global_ridge(
    source: np.ndarray,
    target: np.ndarray,
    lam: float,
    rank: Optional[int] = None,
) -> RidgeMap:
    """Fit an affine ridge map; rank limits the learned residual linear part."""

    x = _matrix(source, "source")
    y = _matrix(target, "target")
    if x.shape != y.shape:
        raise ValueError("source and target shapes differ")
    if not np.isfinite(lam) or float(lam) < 0.0:
        raise ValueError("lam must be finite and nonnegative")
    d = x.shape[1]
    augmented = np.column_stack((x, np.ones(len(x))))
    penalty = np.eye(d + 1, dtype=np.float64) * float(lam)
    penalty[-1, -1] = 0.0
    if rank is None:
        weights = np.linalg.pinv(augmented.T @ augmented + penalty) @ augmented.T @ y
        return RidgeMap(weights=weights, rank=None, regularization=float(lam))

    requested_rank = int(rank)
    if not 1 <= requested_rank <= d:
        raise ValueError("rank must be in [1, dimension]")
    residual = y - x
    raw = np.linalg.pinv(augmented.T @ augmented + penalty) @ augmented.T @ residual
    linear = raw[:-1]
    left, singular, right_t = np.linalg.svd(linear, full_matrices=False)
    effective = min(requested_rank, len(singular))
    truncated = (left[:, :effective] * singular[:effective]) @ right_t[:effective]
    weights = np.vstack((np.eye(d) + truncated, raw[-1]))
    return RidgeMap(weights=weights, rank=requested_rank, regularization=float(lam))


def fit_procrustes(source: np.ndarray, target: np.ndarray) -> ProcrustesMap:
    x = _matrix(source, "source")
    y = _matrix(target, "target")
    if x.shape != y.shape:
        raise ValueError("source and target shapes differ")
    source_mean = x.mean(axis=0)
    target_mean = y.mean(axis=0)
    left, _singular, right_t = np.linalg.svd(
        (x - source_mean).T @ (y - target_mean), full_matrices=False
    )
    return ProcrustesMap(left @ right_t, source_mean, target_mean)


def select_ridge_hyperparameters(
    source: np.ndarray,
    target: np.ndarray,
    fit_indices: Sequence[int],
    dev_indices: Sequence[int],
    eval_indices: Optional[Sequence[int]] = None,
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
    ranks: Sequence[Optional[int]] = (None,),
) -> SelectionAudit:
    """Choose lambda/rank solely by anchor-dev MSE.

    ``eval_indices`` is recorded for leakage auditing but is never indexed or
    dereferenced.  This makes accidental eval-based tuning testable.
    """

    x = _matrix(source, "source")
    y = _matrix(target, "target")
    fit = np.asarray(tuple(fit_indices), dtype=np.int64)
    dev = np.asarray(tuple(dev_indices), dtype=np.int64)
    if not len(fit) or not len(dev):
        raise ValueError("fit_indices and dev_indices must both be non-empty")
    if set(fit.tolist()).intersection(dev.tolist()):
        raise ValueError("fit/dev overlap")
    best: Optional[Tuple[float, int, float, Optional[int]]] = None
    for rank in ranks:
        rank_key = x.shape[1] + 1 if rank is None else int(rank)
        for lam in lambdas:
            model = fit_global_ridge(x[fit], y[fit], float(lam), rank=rank)
            mse = float(np.mean((model.transform(x[dev]) - y[dev]) ** 2))
            candidate = (mse, rank_key, float(lam), rank)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    if best is None:
        raise ValueError("no hyperparameter candidates")
    eval_values = () if eval_indices is None else eval_indices
    eval_tuple = tuple(int(value) for value in eval_values)
    return SelectionAudit(
        fit_indices=tuple(int(value) for value in fit),
        dev_indices=tuple(int(value) for value in dev),
        eval_indices=eval_tuple,
        selected_lambda=float(best[2]),
        selected_rank=best[3],
        dev_mse=float(best[0]),
    )


def fit_local_ridges(
    source: np.ndarray,
    target: np.ndarray,
    routing_labels: np.ndarray,
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
    dev_indices: Optional[Sequence[int]] = None,
    rank: Optional[int] = None,
    train_indices: Optional[Sequence[int]] = None,
    candidate_ranks: Optional[Sequence[int]] = None,
    eval_indices: Optional[Sequence[int]] = None,
) -> LocalRidgeModel:
    """Fit per-inferred-cluster maps with shared anchor-dev hyperparameters."""

    x = _matrix(source, "source")
    y = _matrix(target, "target")
    labels = np.asarray(routing_labels, dtype=np.int64)
    if x.shape != y.shape or labels.shape != (len(x),):
        raise ValueError("local fit arrays are misaligned")
    if dev_indices is None:
        raise ValueError("dev_indices are required")
    dev = np.asarray(tuple(dev_indices), dtype=np.int64)
    if train_indices is None:
        train = np.asarray(
            [index for index in range(len(x)) if index not in set(dev.tolist())],
            dtype=np.int64,
        )
    else:
        train = np.asarray(tuple(train_indices), dtype=np.int64)
    ranks: Sequence[Optional[int]]
    if candidate_ranks is not None:
        ranks = tuple(int(value) for value in candidate_ranks)
    else:
        ranks = (rank,)

    # Select one shared lambda/rank by aggregate dev error across inferred
    # clusters.  Sparse clusters fall back to the pooled map for selection.
    best: Optional[Tuple[float, int, float, Optional[int]]] = None
    for candidate_rank in ranks:
        rank_key = x.shape[1] + 1 if candidate_rank is None else int(candidate_rank)
        for lam in lambdas:
            predictions = np.zeros_like(y[dev])
            for position, row in enumerate(dev):
                cluster = int(labels[row])
                local_train = train[labels[train] == cluster]
                fit_rows = local_train if len(local_train) >= 2 else train
                model = fit_global_ridge(
                    x[fit_rows], y[fit_rows], float(lam), rank=candidate_rank
                )
                predictions[position] = model.transform(x[row : row + 1])[0]
            mse = float(np.mean((predictions - y[dev]) ** 2))
            candidate = (mse, rank_key, float(lam), candidate_rank)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    if best is None:
        raise ValueError("no local hyperparameter candidates")

    selected_rank = best[3]
    selected_lambda = float(best[2])
    all_anchor_rows = np.unique(np.concatenate((train, dev)))
    maps: Dict[int, RidgeMap] = {}
    selected_lambdas: Dict[int, float] = {}
    for cluster in np.unique(labels[all_anchor_rows]):
        local_rows = all_anchor_rows[labels[all_anchor_rows] == cluster]
        if len(local_rows) < 2:
            continue
        maps[int(cluster)] = fit_global_ridge(
            x[local_rows], y[local_rows], selected_lambda, rank=selected_rank
        )
        selected_lambdas[int(cluster)] = selected_lambda
    audit = {
        "fit_indices": [int(value) for value in train],
        "dev_indices": [int(value) for value in dev],
        "eval_indices": [
            int(value) for value in (() if eval_indices is None else eval_indices)
        ],
        "selection_source": "anchor_dev_mse_only",
        "clean_cluster_ids_accepted": False,
        # C3-review SHOULD-FIX: freeze the dev-selected hyperparameters so the
        # "fair CV-lambda" claim is auditable offline without re-running selection.
        "selected_lambda": selected_lambda,
        "selected_rank": None if selected_rank is None else int(selected_rank),
        "selected_lambdas_per_cluster": {
            str(cluster): float(value) for cluster, value in selected_lambdas.items()
        },
    }
    return LocalRidgeModel(
        maps=maps,
        selected_lambdas=selected_lambdas,
        selected_rank=selected_rank,
        dev_mse=float(best[0]),
        training_audit=audit,
    )


def apply_local_ridges(
    model: LocalRidgeModel,
    values: np.ndarray,
    routing_labels: np.ndarray,
    gated_clusters: Optional[Iterable[int]] = None,
) -> np.ndarray:
    matrix = _matrix(values, "values")
    labels = np.asarray(routing_labels, dtype=np.int64)
    if labels.shape != (len(matrix),):
        raise ValueError("routing_labels length mismatch")
    allowed = set(model.maps) if gated_clusters is None else set(map(int, gated_clusters))
    output = matrix.copy()
    for cluster, ridge in model.maps.items():
        if cluster not in allowed:
            continue
        rows = labels == cluster
        if np.any(rows):
            output[rows] = ridge.transform(matrix[rows])
    return output


def _anchor_split(labels: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed) ^ 0xA11CE)
    train: List[int] = []
    dev: List[int] = []
    for cluster in np.unique(labels):
        rows = np.flatnonzero(labels == cluster).copy()
        rng.shuffle(rows)
        dev_count = max(1, int(round(0.25 * len(rows)))) if len(rows) >= 3 else 1
        if len(rows) == 1:
            train.extend(rows.tolist())
            dev.extend(rows.tolist())
        else:
            dev.extend(rows[:dev_count].tolist())
            train.extend(rows[dev_count:].tolist())
    # A one-row cluster can only occur under extreme routing error; move its
    # duplicate dev row to a pooled disjoint split rather than leak it.
    overlap = set(train).intersection(dev)
    if overlap:
        dev = [value for value in dev if value not in overlap]
    if not dev:
        order = np.arange(len(labels))
        rng.shuffle(order)
        dev = [int(order[0])]
        train = [int(value) for value in order[1:]]
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(sorted(dev), dtype=np.int64)


def _detector_gate(
    corrupted: np.ndarray,
    clean: np.ndarray,
    anchor_indices: np.ndarray,
    inferred_labels: np.ndarray,
    cluster_count: int,
) -> Tuple[np.ndarray, float, np.ndarray]:
    displacement = np.linalg.norm(clean[anchor_indices] - corrupted[anchor_indices], axis=1)
    scale = np.maximum(np.linalg.norm(corrupted[anchor_indices], axis=1), 1e-12)
    relative = displacement / scale
    scores = np.zeros(int(cluster_count), dtype=np.float64)
    for cluster in range(int(cluster_count)):
        values = relative[inferred_labels[anchor_indices] == cluster]
        scores[cluster] = float(values.mean()) if len(values) else 0.0
    positives = scores[scores > 1e-10]
    if not len(positives):
        tau = math.inf
    else:
        candidates = np.unique(
            np.concatenate(([1e-10], positives, (positives[:-1] + positives[1:]) / 2.0))
        )
        # Labels are paired-anchor evidence, not clean cluster IDs.  Select the
        # lowest threshold attaining maximum balanced accuracy.
        labels = scores > 1e-10
        best = None
        for candidate in candidates:
            predicted = scores >= candidate
            tpr = float(predicted[labels].mean()) if np.any(labels) else 1.0
            tnr = float((~predicted[~labels]).mean()) if np.any(~labels) else 1.0
            item = (-(tpr + tnr) / 2.0, float(candidate))
            if best is None or item < best:
                best = item
        tau = float(best[1])
    detected = np.flatnonzero(scores >= tau).astype(np.int64)
    return scores, tau, detected


def _cosine_scores(queries: np.ndarray, documents: np.ndarray) -> np.ndarray:
    q = _matrix(queries, "queries")
    d = _matrix(documents, "documents")
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    dn = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    return qn @ dn.T


def score_ndcg(
    queries: np.ndarray,
    documents: np.ndarray,
    relevance: Sequence[object],
    k: int = 10,
) -> float:
    """Mean binary NDCG@k for planted positives or relevance sets."""

    scores = _cosine_scores(queries, documents)
    if len(relevance) != len(scores):
        raise ValueError("relevance length must equal query count")
    values: List[float] = []
    for row, relevant in zip(scores, relevance):
        dense_gains: Optional[np.ndarray] = None
        if isinstance(relevant, (int, np.integer)):
            relevant_set = {int(relevant)}
        elif isinstance(relevant, Mapping):
            relevant_set = {int(key) for key, value in relevant.items() if float(value) > 0.0}
        elif (
            isinstance(relevant, np.ndarray)
            and relevant.ndim == 1
            and len(relevant) == len(row)
        ):
            dense_gains = np.asarray(relevant, dtype=np.float64)
            relevant_set = set(np.flatnonzero(dense_gains > 0.0).tolist())
        else:
            relevant_set = {int(value) for value in relevant}
        order = np.argsort(-row, kind="stable")[: int(k)]
        gains = (
            dense_gains[order]
            if dense_gains is not None
            else np.asarray([1.0 if int(index) in relevant_set else 0.0 for index in order])
        )
        discounts = 1.0 / np.log2(np.arange(2, len(order) + 2, dtype=np.float64))
        dcg = float(np.sum(gains * discounts))
        if dense_gains is not None:
            ideal = np.sort(dense_gains)[::-1][: int(k)]
            idcg = float(np.sum(ideal * discounts[: len(ideal)]))
        else:
            ideal_count = min(len(relevant_set), int(k))
            idcg = float(np.sum(discounts[:ideal_count])) if ideal_count else 0.0
        values.append(dcg / idcg if idcg else 0.0)
    return float(np.mean(values))


def _apply_gated_global(
    model: object,
    documents: np.ndarray,
    assignments: np.ndarray,
    detected: np.ndarray,
) -> np.ndarray:
    output = np.asarray(documents, dtype=np.float64).copy()
    mask = np.isin(assignments, detected)
    if np.any(mask):
        output[mask] = model.transform(output[mask])
    return output


def _parameter_count(dimension: int, clusters: int, rank: Optional[int]) -> int:
    if rank is None:
        return int(clusters * (dimension * dimension + dimension))
    return int(clusters * (2 * dimension * int(rank) + dimension))


def _cluster_purity(inferred: np.ndarray, diagnostic_truth: np.ndarray) -> float:
    """Diagnostic-only partition purity; its result is never fed to a fit."""

    inferred_labels = np.asarray(inferred, dtype=np.int64)
    truth = np.asarray(diagnostic_truth, dtype=np.int64)
    if inferred_labels.shape != truth.shape:
        raise ValueError("partition purity arrays must align")
    matched = 0
    for cluster in np.unique(inferred_labels):
        counts = np.bincount(truth[inferred_labels == cluster])
        matched += int(counts.max())
    return float(matched / len(truth))


def run_preflight_cell(
    family: str,
    gamma: float,
    anchors_per_harmed_cluster: int,
    sparse_fraction: float,
    seed: int,
    n_docs: int = 800,
    dim: int = 16,
    n_clusters: int = 20,
    n_queries_per_cluster: int = 6,
) -> Dict[str, object]:
    if int(n_docs) % int(n_clusters) != 0:
        raise ValueError("n_docs must be divisible by n_clusters")
    regime = generate_sparse_local_regime(
        seed=int(seed),
        docs_per_cluster=int(n_docs) // int(n_clusters),
        dimension=int(dim),
        n_clusters=int(n_clusters),
        sparsity=float(sparse_fraction),
        gamma=float(gamma),
        family=str(family),
        queries_per_cluster=int(n_queries_per_cluster),
        anchors_per_harmed_cluster=int(anchors_per_harmed_cluster),
    )
    clean = np.asarray(regime.clean_docs, dtype=np.float64)
    corrupted = np.asarray(regime.corrupted_docs, dtype=np.float64)
    queries = np.asarray(regime.queries, dtype=np.float64)
    relevance = regime.relevance
    assignments = infer_corrupted_clusters(corrupted, int(n_clusters), seed=int(seed))
    corrupted_cluster_purity = _cluster_purity(
        assignments, np.asarray(regime.clean_cluster_ids, dtype=np.int64)
    )
    anchor_indices = np.asarray(regime.anchor_indices, dtype=np.int64)
    anchor_labels = assignments[anchor_indices]
    train, dev = _anchor_split(anchor_labels, seed=int(seed))
    source_anchor = corrupted[anchor_indices]
    target_anchor = clean[anchor_indices]

    detector_scores, tau, detected = _detector_gate(
        corrupted, clean, anchor_indices, assignments, int(n_clusters)
    )
    global_selection = select_ridge_hyperparameters(
        source_anchor,
        target_anchor,
        train,
        dev,
        eval_indices=(),
        lambdas=DEFAULT_LAMBDAS,
        ranks=(None,),
    )
    global_ridge = fit_global_ridge(
        source_anchor,
        target_anchor,
        global_selection.selected_lambda,
        rank=None,
    )
    procrustes = fit_procrustes(source_anchor[train], target_anchor[train])
    proc_dev_mse = float(
        np.mean((procrustes.transform(source_anchor[dev]) - target_anchor[dev]) ** 2)
    )

    local_dense = fit_local_ridges(
        source_anchor,
        target_anchor,
        anchor_labels,
        lambdas=DEFAULT_LAMBDAS,
        train_indices=train,
        dev_indices=dev,
        rank=None,
        eval_indices=(),
    )
    global_parameter_budget = _parameter_count(dim, 1, None)
    rank_candidates = tuple(
        rank
        for rank in DEFAULT_RANKS
        if rank <= dim
        and _parameter_count(dim, max(1, len(detected)), rank)
        <= global_parameter_budget
    )
    if not rank_candidates:
        rank_candidates = (1,)
    local_low_rank = fit_local_ridges(
        source_anchor,
        target_anchor,
        anchor_labels,
        lambdas=DEFAULT_LAMBDAS,
        train_indices=train,
        dev_indices=dev,
        candidate_ranks=rank_candidates,
        eval_indices=(),
    )

    oracle = score_ndcg(queries, clean, relevance, k=10)
    floor = score_ndcg(queries, corrupted, relevance, k=10)
    global_docs = _apply_gated_global(global_ridge, corrupted, assignments, detected)
    proc_docs = _apply_gated_global(procrustes, corrupted, assignments, detected)
    dense_docs = apply_local_ridges(
        local_dense, corrupted, assignments, gated_clusters=detected
    )
    low_rank_docs = apply_local_ridges(
        local_low_rank, corrupted, assignments, gated_clusters=detected
    )
    method_ndcg = {
        "global_ridge": score_ndcg(queries, global_docs, relevance, k=10),
        "global_procrustes": score_ndcg(queries, proc_docs, relevance, k=10),
        "gated_local_dense_ridge": score_ndcg(queries, dense_docs, relevance, k=10),
        "gated_local_low_rank_ridge": score_ndcg(
            queries, low_rank_docs, relevance, k=10
        ),
    }
    best_global = (
        "global_ridge"
        if global_selection.dev_mse <= proc_dev_mse
        else "global_procrustes"
    )
    best_local = (
        "gated_local_dense_ridge"
        if local_dense.dev_mse <= local_low_rank.dev_mse
        else "gated_local_low_rank_ridge"
    )
    clean_norm = np.maximum(np.linalg.norm(clean, axis=1), 1e-12)
    displacement = np.linalg.norm(corrupted - clean, axis=1)
    nonzero = displacement > 1e-12
    return {
        "seed": int(seed),
        "family": str(family),
        "gamma": float(gamma),
        "anchors_per_harmed_cluster": int(anchors_per_harmed_cluster),
        "sparse_fraction": float(sparse_fraction),
        "oracle_ndcg": oracle,
        "floor_ndcg": floor,
        "method_ndcg": method_ndcg,
        "selected_best_global": best_global,
        "selected_best_local": best_local,
        "ladders": {
            "oracle_minus_floor": oracle - floor,
            "oracle_minus_best_global": oracle - method_ndcg[best_global],
            "oracle_minus_best_gated_local_ridge": oracle - method_ndcg[best_local],
        },
        "residual_band": oracle - method_ndcg[best_local],
        "displacement": {
            "mean_all_docs": float(displacement.mean()),
            "max_all_docs": float(displacement.max()),
            "mean_harmed_docs": float(displacement[nonzero].mean()) if np.any(nonzero) else 0.0,
            "max_relative_to_doc_norm": float(np.max(displacement / clean_norm)),
            "mean_relative_to_doc_norm": float(np.mean(displacement / clean_norm)),
        },
        "detector": {
            "tau": tau,
            "scores": detector_scores.tolist(),
            "detected_cluster_count": int(len(detected)),
            "corrupted_space_cluster_count": int(len(np.unique(assignments))),
            "corrupted_space_cluster_purity_diagnostic": corrupted_cluster_purity,
            # C3-review SHOULD-FIX: derive from the actual fit audits instead of
            # hardcoding, so a future clean-ID leak flips this True and is caught.
            "clean_ids_used_by_fit": bool(
                local_dense.training_audit.get("clean_cluster_ids_accepted", True)
                or local_low_rank.training_audit.get("clean_cluster_ids_accepted", True)
            ),
        },
        "selection": {
            "global_ridge": asdict(global_selection),
            "procrustes_dev_mse": proc_dev_mse,
            "local_dense": local_dense.training_audit
            | {"dev_mse": local_dense.dev_mse, "rank": None},
            "local_low_rank": local_low_rank.training_audit
            | {
                "dev_mse": local_low_rank.dev_mse,
                "rank": local_low_rank.selected_rank,
            },
            "method_family_choice_source": "anchor_dev_mse_only",
        },
        "parameter_counts": {
            "global_dense": global_parameter_budget,
            "local_dense_allocated": _parameter_count(dim, len(detected), None),
            "local_low_rank_allocated": _parameter_count(
                dim, len(detected), local_low_rank.selected_rank
            ),
            "low_rank_within_global_dense_budget": _parameter_count(
                dim, len(detected), local_low_rank.selected_rank
            )
            <= global_parameter_budget,
        },
        "diagnostic": {
            "true_harmed_cluster_count": int(len(regime.harmed_clusters)),
            "true_harmed_fraction": float(len(regime.harmed_clusters) / n_clusters),
            "clean_ids_available_only_in_regime_diagnostic": True,
        },
    }


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return float(np.mean(rows)) if rows else math.nan


def _aggregate_cell(runs: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    first = runs[0]
    method_names = tuple(first["method_ndcg"])
    ladders = {
        name: _mean(run["ladders"][name] for run in runs)
        for name in first["ladders"]
    }
    displacement_names = tuple(first["displacement"])
    return {
        "family": first["family"],
        "gamma": first["gamma"],
        "anchors_per_harmed_cluster": first["anchors_per_harmed_cluster"],
        "sparse_fraction": first["sparse_fraction"],
        "seed_count": len(runs),
        "seeds": [run["seed"] for run in runs],
        "mean_oracle_ndcg": _mean(run["oracle_ndcg"] for run in runs),
        "mean_floor_ndcg": _mean(run["floor_ndcg"] for run in runs),
        "mean_method_ndcg": {
            name: _mean(run["method_ndcg"][name] for run in runs)
            for name in method_names
        },
        "three_ladders": ladders,
        "residual_band": ladders["oracle_minus_best_gated_local_ridge"],
        "mean_displacement": {
            name: _mean(run["displacement"][name] for run in runs)
            for name in displacement_names
        },
        "selected_best_global_by_seed": [run["selected_best_global"] for run in runs],
        "selected_best_local_by_seed": [run["selected_best_local"] for run in runs],
        "detector_tau_by_seed": [run["detector"]["tau"] for run in runs],
        "detected_cluster_count_by_seed": [
            run["detector"]["detected_cluster_count"] for run in runs
        ],
        "mean_corrupted_space_cluster_purity": _mean(
            run["detector"]["corrupted_space_cluster_purity_diagnostic"]
            for run in runs
        ),
        # OR over per-run detector audits (same derivation as run_preflight_cell):
        # a future clean-ID leak on any seed must flip the aggregate flag.
        "clean_ids_used_by_fit": bool(
            any(bool(run["detector"]["clean_ids_used_by_fit"]) for run in runs)
        ),
        "runs": list(runs),
    }


def run_sweep(
    output_dir: Path,
    gammas: Sequence[float] = (0.10, 0.30, 0.60, 1.00),
    anchor_counts: Sequence[int] = (4, 8, 16),
    sparse_fractions: Sequence[float] = (0.15, 0.25),
    families: Sequence[str] = ("quadratic", "soft_fold"),
    seeds: Sequence[int] = (7, 42, 101),
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells: List[Dict[str, object]] = []
    for family in families:
        for sparse_fraction in sparse_fractions:
            for gamma in gammas:
                for anchors in anchor_counts:
                    runs = [
                        run_preflight_cell(
                            family=family,
                            gamma=float(gamma),
                            anchors_per_harmed_cluster=int(anchors),
                            sparse_fraction=float(sparse_fraction),
                            seed=int(seed),
                        )
                        for seed in seeds
                    ]
                    cells.append(_aggregate_cell(runs))

    criteria = {
        "minimum_residual_band": 0.05,
        "small_mean_relative_displacement_max": 0.12,
        "small_max_relative_displacement_max": 0.35,
        "sane_oracle_minus_floor_min": 0.015,
        "sane_oracle_minus_floor_max": 0.35,
        "minimum_corrupted_space_cluster_purity": 0.90,
        "corrupted_space_clustering_required": True,
    }
    family_results: Dict[str, object] = {}
    admitted_any = False
    for family in families:
        family_cells = [cell for cell in cells if cell["family"] == family]
        eligible = []
        for cell in family_cells:
            displacement = cell["mean_displacement"]
            ladder = cell["three_ladders"]["oracle_minus_floor"]
            checks = {
                "residual": cell["residual_band"] >= criteria["minimum_residual_band"],
                "small_mean": displacement["mean_relative_to_doc_norm"]
                <= criteria["small_mean_relative_displacement_max"],
                "small_max": displacement["max_relative_to_doc_norm"]
                <= criteria["small_max_relative_displacement_max"],
                "sane_floor_gap": criteria["sane_oracle_minus_floor_min"]
                <= ladder
                <= criteria["sane_oracle_minus_floor_max"],
                "corrupted_space_clustering": (
                    not cell["clean_ids_used_by_fit"]
                    and cell["mean_corrupted_space_cluster_purity"]
                    >= criteria["minimum_corrupted_space_cluster_purity"]
                ),
            }
            cell["admission_checks"] = checks
            cell["admitted"] = bool(all(checks.values()))
            if cell["admitted"]:
                eligible.append(cell)
        operating = (
            sorted(
                eligible,
                key=lambda item: (
                    -item["residual_band"],
                    item["gamma"],
                    item["anchors_per_harmed_cluster"],
                    item["sparse_fraction"],
                ),
            )[0]
            if eligible
            else None
        )
        admitted = operating is not None
        admitted_any = admitted_any or admitted
        family_results[family] = {
            "verdict": "PROCEED-TO-GPU" if admitted else "CLOSE",
            "operating_point": (
                {
                    "gamma": operating["gamma"],
                    "anchors_per_harmed_cluster": operating[
                        "anchors_per_harmed_cluster"
                    ],
                    "sparse_fraction": operating["sparse_fraction"],
                    "residual_band": operating["residual_band"],
                }
                if operating
                else None
            ),
            "smallest_reason": None
            if admitted
            else _family_close_reason(family_cells, criteria),
        }
    overall_verdict = "PROCEED-TO-GPU" if admitted_any else "CLOSE"
    all_runs = [run for cell in cells for run in cell["runs"]]
    local_choice_counts = {
        name: sum(run["selected_best_local"] == name for run in all_runs)
        for name in (
            "gated_local_dense_ridge",
            "gated_local_low_rank_ridge",
        )
    }
    protocol_audit = {
        "cell_count": len(cells),
        "seed_run_count": len(all_runs),
        "clean_id_fit_violations": sum(
            bool(run["detector"]["clean_ids_used_by_fit"]) for run in all_runs
        ),
        "low_rank_budget_violations": sum(
            not bool(run["parameter_counts"]["low_rank_within_global_dense_budget"])
            for run in all_runs
        ),
        "local_choice_counts": local_choice_counts,
        "minimum_corrupted_space_cluster_purity": min(
            cell["mean_corrupted_space_cluster_purity"] for cell in cells
        ),
        "shared_detector_gate_for_all_methods": True,
        "detector_tau_is_anchor_displacement_presence_gate": True,
        "eval_selected_hyperparameters_or_methods": False,
    }
    result = {
        "schema_version": 1,
        "objective": "Obj B CPU preflight falsifier",
        "compute": "CPU NumPy only; no GPU; no embedding model",
        "config": {
            "gammas": list(map(float, gammas)),
            "anchor_counts": list(map(int, anchor_counts)),
            "sparse_fractions": list(map(float, sparse_fractions)),
            "families": list(families),
            "seeds": list(map(int, seeds)),
            "lambda_candidates": list(DEFAULT_LAMBDAS),
            "rank_candidates": list(DEFAULT_RANKS),
            "method_selection": "anchor_dev_mse_only",
            "detector_tau_policy": (
                "paired-anchor relative-displacement presence gate (>1e-10); shared; "
                "deliberately oracle-generous"
            ),
            "admission_criteria": criteria,
        },
        "family_results": family_results,
        "protocol_audit": protocol_audit,
        "verdict": overall_verdict,
        "close_one_liner": CLOSE_ONE_LINER if overall_verdict == "CLOSE" else None,
        "cells": cells,
    }
    (output_dir / "preflight_grid.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "preflight_report.md").write_text(
        _render_report(result), encoding="utf-8"
    )
    return result


def _family_close_reason(
    cells: Sequence[Mapping[str, object]], criteria: Mapping[str, object]
) -> str:
    small_sane = [
        cell
        for cell in cells
        if cell["mean_displacement"]["mean_relative_to_doc_norm"]
        <= criteria["small_mean_relative_displacement_max"]
        and cell["mean_displacement"]["max_relative_to_doc_norm"]
        <= criteria["small_max_relative_displacement_max"]
        and criteria["sane_oracle_minus_floor_min"]
        <= cell["three_ladders"]["oracle_minus_floor"]
        <= criteria["sane_oracle_minus_floor_max"]
    ]
    if not small_sane:
        return "No swept cell was both small-magnitude and retrieval-discriminating."
    maximum = max(cell["residual_band"] for cell in small_sane)
    return (
        "The strongest small, sane cell left only "
        f"{maximum:.4f} NDCG after the dev-selected gated local ridge "
        f"(< {criteria['minimum_residual_band']:.2f})."
    )


def _render_report(result: Mapping[str, object]) -> str:
    lines = [
        "# Obj B CPU preflight falsifier",
        "",
        f"**Verdict: {result['verdict']}**",
        "",
    ]
    if result["close_one_liner"]:
        lines.extend((f"> {result['close_one_liner']}", ""))
    lines.extend(
        (
            "This run used synthetic Gaussian-cluster vectors and NumPy closed-form fits only. "
            "It used no GPU and no embedding model. Routing was inferred from corrupted vectors; "
            "clean cluster IDs were retained only for diagnostics and corpus construction.",
            "",
            "## Per-family admission",
            "",
            "| Family | Verdict | Operating point / smallest reason |",
            "|---|---:|---|",
        )
    )
    for family, summary in result["family_results"].items():
        if summary["operating_point"]:
            point = summary["operating_point"]
            reason = (
                f"gamma={point['gamma']}, n={point['anchors_per_harmed_cluster']}, "
                f"s={point['sparse_fraction']}, residual={point['residual_band']:.4f}"
            )
        else:
            reason = summary["smallest_reason"]
        lines.append(f"| {family} | {summary['verdict']} | {reason} |")
    lines.extend(
        (
            "",
            "## Grid",
            "",
            "The three ladders are oracle minus floor, oracle minus the anchor-dev-selected global "
            "map, and oracle minus the anchor-dev-selected detector-gated local ridge.",
            "",
            "| Family | s | gamma | n | oracle-floor | oracle-global | oracle-local | residual | mean/max disp | mean/max rel disp | purity | admitted |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for cell in result["cells"]:
        ladders = cell["three_ladders"]
        displacement = cell["mean_displacement"]
        lines.append(
            f"| {cell['family']} | {cell['sparse_fraction']:.2f} | {cell['gamma']:.2f} | "
            f"{cell['anchors_per_harmed_cluster']} | {ladders['oracle_minus_floor']:.4f} | "
            f"{ladders['oracle_minus_best_global']:.4f} | "
            f"{ladders['oracle_minus_best_gated_local_ridge']:.4f} | "
            f"{cell['residual_band']:.4f} | {displacement['mean_all_docs']:.4f}/"
            f"{displacement['max_all_docs']:.4f} | "
            f"{displacement['mean_relative_to_doc_norm']:.4f}/"
            f"{displacement['max_relative_to_doc_norm']:.4f} | "
            f"{cell['mean_corrupted_space_cluster_purity']:.3f} | "
            f"{'yes' if cell['admitted'] else 'no'} |"
        )
    criteria = result["config"]["admission_criteria"]
    audit = result["protocol_audit"]
    lines.extend(
        (
            "",
            "## Frozen admission and parity rules",
            "",
            f"A cell requires residual >= {criteria['minimum_residual_band']:.2f}, mean relative "
            f"displacement <= {criteria['small_mean_relative_displacement_max']:.2f}, max relative "
            f"displacement <= {criteria['small_max_relative_displacement_max']:.2f}, and "
            f"oracle-floor in [{criteria['sane_oracle_minus_floor_min']:.3f}, "
            f"{criteria['sane_oracle_minus_floor_max']:.3f}], with corrupted-space partition "
            f"purity >= {criteria['minimum_corrupted_space_cluster_purity']:.2f}. Lambda and rank use the same "
            "anchor train/dev selection boundary; eval queries select no hyperparameter or method.",
            "",
            "Dense local ridge is intentionally not handicapped. The low-rank local alternative "
            "uses the same lambda candidates and chooses rank on the same anchor dev rows. The "
            "reported best-local ladder uses whichever local family had lower anchor-dev MSE.",
            "",
            f"Protocol audit: {audit['seed_run_count']} seed-runs; "
            f"{audit['clean_id_fit_violations']} clean-ID fit violations; "
            f"{audit['low_rank_budget_violations']} low-rank budget violations; minimum "
            f"corrupted-space purity {audit['minimum_corrupted_space_cluster_purity']:.3f}. "
            f"The dev-selected local method was low-rank in "
            f"{audit['local_choice_counts']['gated_local_low_rank_ridge']} seed-runs and dense "
            f"in {audit['local_choice_counts']['gated_local_dense_ridge']}; method selection "
            "never used eval NDCG.",
            "",
            "Detector limitation: anchors are generated only for truly harmed clusters, so the "
            "anchor-derived detector labels are equivalent to nonzero paired displacement and "
            "tau is 1e-10 in every run. This is an oracle-generous presence gate shared by all "
            "methods, not an independently learned or stress-tested detector. It biases the "
            "preflight toward finding that local ridge can close the gap.",
            "",
        )
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/drift_recovery/out/d2b_preflight"),
    )
    args = parser.parse_args(argv)
    result = run_sweep(args.output_dir)
    print(result["verdict"])
    if result["close_one_liner"]:
        print(result["close_one_liner"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
