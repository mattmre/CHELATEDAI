"""Deterministic local semantic-collapse corruption for Regime-C.

The corruption is intentionally modest: selected document clusters are pulled
toward their own clean centroid by ``beta``.  It does not rotate the space,
change the encoder, or expose retrieval labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def _as_matrix(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("vectors must be a non-trivial 2D matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("vectors contain non-finite values")
    return matrix


def _squared_distances(matrix: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    matrix_norm = np.sum(matrix * matrix, axis=1, keepdims=True)
    centroid_norm = np.sum(centroids * centroids, axis=1)[None, :]
    return np.maximum(matrix_norm + centroid_norm - 2.0 * matrix @ centroids.T, 0.0)


def _kmeans_plus_plus(matrix: np.ndarray, cluster_count: int, rng: np.random.Generator) -> np.ndarray:
    indices = [int(rng.integers(0, len(matrix)))]
    closest = _squared_distances(matrix, matrix[indices]).reshape(-1)
    for _ in range(1, cluster_count):
        total = float(closest.sum())
        if total <= 0.0:
            remaining = np.setdiff1d(np.arange(len(matrix)), np.asarray(indices), assume_unique=False)
            indices.append(int(remaining[0]))
        else:
            indices.append(int(rng.choice(len(matrix), p=closest / total)))
        candidate = _squared_distances(matrix, matrix[[indices[-1]]]).reshape(-1)
        closest = np.minimum(closest, candidate)
    return matrix[np.asarray(indices, dtype=np.int64)].copy()


def deterministic_kmeans(
    vectors: np.ndarray,
    cluster_count: int = 32,
    seed: int = 42,
    iterations: int = 25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return stable assignments and centroids without an external estimator."""

    matrix = _as_matrix(vectors)
    if not 2 <= int(cluster_count) <= len(matrix):
        raise ValueError("cluster_count must be between 2 and the document count")
    if int(iterations) < 1:
        raise ValueError("iterations must be positive")
    rng = np.random.default_rng(int(seed))
    centroids = _kmeans_plus_plus(matrix, int(cluster_count), rng)
    assignments = np.zeros(len(matrix), dtype=np.int64)
    for _ in range(int(iterations)):
        distances = _squared_distances(matrix, centroids)
        next_assignments = np.argmin(distances, axis=1).astype(np.int64)
        counts = np.bincount(next_assignments, minlength=int(cluster_count))
        if np.any(counts == 0):
            nearest = distances[np.arange(len(matrix)), next_assignments]
            available = np.argsort(-nearest, kind="mergesort").tolist()
            used = set()
            for empty in np.flatnonzero(counts == 0):
                replacement = next(index for index in available if index not in used)
                used.add(replacement)
                next_assignments[replacement] = int(empty)
        next_centroids = np.vstack(
            [matrix[next_assignments == cluster].mean(axis=0) for cluster in range(int(cluster_count))]
        )
        if np.array_equal(assignments, next_assignments) and np.allclose(
            centroids, next_centroids, rtol=0.0, atol=1e-14
        ):
            assignments = next_assignments
            centroids = next_centroids
            break
        assignments = next_assignments
        centroids = next_centroids
    return assignments, centroids


@dataclass(frozen=True)
class CollapseResult:
    clean: np.ndarray
    corrupted: np.ndarray
    assignments: np.ndarray
    centroids: np.ndarray
    collapsed_clusters: np.ndarray
    beta: float
    seed: int

    @property
    def changed_mask(self) -> np.ndarray:
        return np.isin(self.assignments, self.collapsed_clusters)

    @property
    def mean_displacement(self) -> float:
        return float(np.linalg.norm(self.corrupted - self.clean, axis=1).mean())


def inject_semantic_collapse(
    vectors: np.ndarray,
    collapse_cluster_count: int,
    beta: float = 0.10,
    seed: int = 42,
    total_clusters: int = 32,
    kmeans_iterations: int = 25,
) -> CollapseResult:
    """Pull exactly ``collapse_cluster_count`` clusters toward clean centroids."""

    clean = _as_matrix(vectors).copy()
    if not 0.0 < float(beta) < 1.0:
        raise ValueError("beta must be in (0, 1)")
    if not 1 <= int(collapse_cluster_count) <= int(total_clusters):
        raise ValueError("collapse_cluster_count must be in [1, total_clusters]")
    assignments, centroids = deterministic_kmeans(
        clean,
        cluster_count=int(total_clusters),
        seed=int(seed),
        iterations=int(kmeans_iterations),
    )
    selection_rng = np.random.default_rng(int(seed) ^ 0xD2C0)
    collapsed = np.sort(
        selection_rng.choice(
            int(total_clusters), size=int(collapse_cluster_count), replace=False
        ).astype(np.int64)
    )
    corrupted = clean.copy()
    mask = np.isin(assignments, collapsed)
    local_centroids = centroids[assignments[mask]]
    corrupted[mask] = clean[mask] + float(beta) * (local_centroids - clean[mask])
    return CollapseResult(
        clean=clean,
        corrupted=corrupted,
        assignments=assignments,
        centroids=centroids,
        collapsed_clusters=collapsed,
        beta=float(beta),
        seed=int(seed),
    )
