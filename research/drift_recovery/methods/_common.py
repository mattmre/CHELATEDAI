"""Shared numerical validation for the D2 NumPy method implementations."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def finite_matrix(values: np.ndarray, name: str, columns: Optional[int] = None) -> np.ndarray:
    """Return a finite float64 matrix with an optional fixed column count."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if columns is not None and matrix.shape[1] != int(columns):
        raise ValueError(f"{name} has dimension {matrix.shape[1]}; expected {int(columns)}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def same_shape_pair(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Validate a finite, same-shape source/target matrix pair."""

    source_matrix = finite_matrix(source, "source")
    target_matrix = finite_matrix(target, "target")
    if source_matrix.shape != target_matrix.shape:
        raise ValueError(
            "source and target must have the same shape; "
            f"got {source_matrix.shape} and {target_matrix.shape}"
        )
    return source_matrix, target_matrix


def validate_architecture(
    dimension: int,
    global_rank: int,
    local_rank: int,
    cluster_count: int,
) -> None:
    """Validate the shared parameter-matched low-rank architecture."""

    if int(dimension) <= 0:
        raise ValueError("dimension must be positive")
    if not 0 <= int(global_rank) <= int(dimension):
        raise ValueError("global_rank must be between zero and dimension")
    if not 0 <= int(local_rank) <= int(dimension):
        raise ValueError("local_rank must be between zero and dimension")
    if int(cluster_count) <= 0:
        raise ValueError("cluster_count must be positive")
    if int(global_rank) == 0 and int(local_rank) == 0:
        raise ValueError("at least one of global_rank and local_rank must be positive")


def parameter_budget(
    dimension: int,
    global_rank: int,
    local_rank: int,
    cluster_count: int,
) -> int:
    """Return the preregistered allocated low-rank parameter count."""

    validate_architecture(dimension, global_rank, local_rank, cluster_count)
    return int(2 * dimension * (global_rank + cluster_count * local_rank))


def truncated_map_factors(
    source: np.ndarray,
    residual: np.ndarray,
    rank: int,
    regularization: float,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit deterministic regularized low-rank residual factors.

    The factor coefficients are scalar weighted-ridge fits.  Block power
    iteration finds the paired source/residual directions without constructing
    and decomposing a dense ``dimension x dimension`` map for every local
    expert, which keeps the 32-cluster D2 diagnostic computationally modest.
    """

    source_matrix = finite_matrix(source, "source")
    residual_matrix = finite_matrix(residual, "residual", columns=source_matrix.shape[1])
    if source_matrix.shape != residual_matrix.shape:
        raise ValueError("source and residual must have the same shape")
    if not np.isfinite(regularization) or float(regularization) <= 0.0:
        raise ValueError("regularization must be finite and positive")
    requested_rank = int(rank)
    if not 0 <= requested_rank <= source_matrix.shape[1]:
        raise ValueError("rank must be between zero and the embedding dimension")
    dimension = source_matrix.shape[1]
    if requested_rank == 0:
        empty = np.zeros((dimension, 0), dtype=np.float64)
        return empty.copy(), empty.copy()

    if weights is None:
        weight_vector = np.ones(source_matrix.shape[0], dtype=np.float64)
    else:
        weight_vector = np.asarray(weights, dtype=np.float64)
        if weight_vector.shape != (source_matrix.shape[0],):
            raise ValueError("weights must have one value per source row")
        if not np.all(np.isfinite(weight_vector)) or np.any(weight_vector < 0.0):
            raise ValueError("weights must be finite and nonnegative")
        if float(np.sum(weight_vector)) <= 0.0:
            empty = np.zeros((dimension, requested_rank), dtype=np.float64)
            return empty.copy(), empty.copy()

    # A fixed trigonometric basis avoids randomized state while ensuring the
    # initialization is not aligned to a single coordinate.
    coordinates = np.arange(1, dimension + 1, dtype=np.float64)[:, None]
    frequencies = np.arange(1, requested_rank + 1, dtype=np.float64)[None, :]
    right_directions, _ = np.linalg.qr(
        np.sin(coordinates * frequencies) + np.cos(coordinates * (frequencies + 0.5))
    )

    def cross_times_right(right: np.ndarray) -> np.ndarray:
        return source_matrix.T @ (weight_vector[:, None] * (residual_matrix @ right))

    def cross_transpose_times_left(left: np.ndarray) -> np.ndarray:
        return residual_matrix.T @ (weight_vector[:, None] * (source_matrix @ left))

    left_directions = np.zeros((dimension, requested_rank), dtype=np.float64)
    for _ in range(8):
        left_raw = cross_times_right(right_directions)
        if float(np.linalg.norm(left_raw)) <= np.finfo(np.float64).eps:
            empty = np.zeros((dimension, requested_rank), dtype=np.float64)
            return empty.copy(), empty.copy()
        left_directions, _ = np.linalg.qr(left_raw)
        right_raw = cross_transpose_times_left(left_directions)
        if float(np.linalg.norm(right_raw)) <= np.finfo(np.float64).eps:
            empty = np.zeros((dimension, requested_rank), dtype=np.float64)
            return empty.copy(), empty.copy()
        right_directions, _ = np.linalg.qr(right_raw)

    # Resolve the small block rotation, then perform one ridge coefficient fit
    # per orthogonal factor.
    small_cross = left_directions.T @ cross_times_right(right_directions)
    small_left, _small_singular, small_right_t = np.linalg.svd(
        small_cross, full_matrices=False
    )
    left_directions = left_directions @ small_left
    right_directions = right_directions @ small_right_t.T
    left = np.zeros((dimension, requested_rank), dtype=np.float64)
    right = np.zeros((dimension, requested_rank), dtype=np.float64)
    for component in range(requested_rank):
        source_coordinate = source_matrix @ left_directions[:, component]
        residual_coordinate = residual_matrix @ right_directions[:, component]
        numerator = float(np.sum(weight_vector * source_coordinate * residual_coordinate))
        denominator = float(
            np.sum(weight_vector * source_coordinate * source_coordinate)
            + float(regularization)
        )
        coefficient = numerator / denominator if denominator > 0.0 else 0.0
        magnitude = math.sqrt(abs(coefficient))
        left[:, component] = left_directions[:, component] * magnitude
        right[:, component] = (
            right_directions[:, component] * math.copysign(magnitude, coefficient)
            if magnitude > 0.0
            else 0.0
        )
    return left, right


def deterministic_kmeans(
    values: np.ndarray,
    cluster_count: int,
    iterations: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster rows with deterministic farthest-first initialization and Lloyd updates."""

    matrix = finite_matrix(values, "values")
    count = int(cluster_count)
    if count <= 0 or count > matrix.shape[0]:
        raise ValueError("cluster_count must be between one and the number of rows")
    if int(iterations) <= 0:
        raise ValueError("iterations must be positive")

    center = np.mean(matrix, axis=0)
    first = int(np.argmax(np.sum((matrix - center) ** 2, axis=1)))
    chosen = [first]
    minimum_distance = np.sum((matrix - matrix[first]) ** 2, axis=1)
    for _ in range(1, count):
        candidate = int(np.argmax(minimum_distance))
        chosen.append(candidate)
        candidate_distance = np.sum((matrix - matrix[candidate]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, candidate_distance)
    centroids = matrix[np.asarray(chosen, dtype=np.int64)].copy()
    assignments = np.full(matrix.shape[0], -1, dtype=np.int64)

    for _ in range(int(iterations)):
        distances = squared_distances(matrix, centroids)
        updated_assignments = np.argmin(distances, axis=1).astype(np.int64)
        if np.array_equal(assignments, updated_assignments):
            break
        assignments = updated_assignments
        updated_centroids = centroids.copy()
        closest_distance = distances[np.arange(matrix.shape[0]), assignments]
        for cluster in range(count):
            members = matrix[assignments == cluster]
            if len(members):
                updated_centroids[cluster] = np.mean(members, axis=0)
            else:
                replacement = int(np.argmax(closest_distance))
                updated_centroids[cluster] = matrix[replacement]
                assignments[replacement] = cluster
                closest_distance[replacement] = -np.inf
        centroids = updated_centroids

    final_distances = squared_distances(matrix, centroids)
    final_assignments = np.argmin(final_distances, axis=1).astype(np.int64)
    # Exact duplicate rows can leave centroids tied and clusters empty. Preserve
    # a deterministic non-empty partition for downstream per-cluster fits.
    for cluster in range(count):
        if np.any(final_assignments == cluster):
            continue
        cluster_sizes = np.bincount(final_assignments, minlength=count)
        movable = np.flatnonzero(cluster_sizes[final_assignments] > 1)
        if len(movable) == 0:
            raise RuntimeError("deterministic k-means could not form non-empty clusters")
        assigned_distance = final_distances[movable, final_assignments[movable]]
        replacement = int(movable[int(np.argmax(assigned_distance))])
        final_assignments[replacement] = cluster
    for cluster in range(count):
        centroids[cluster] = np.mean(matrix[final_assignments == cluster], axis=0)
    return centroids, final_assignments


def squared_distances(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Return stable pairwise squared Euclidean distances."""

    matrix = finite_matrix(values, "values")
    centers = finite_matrix(centroids, "centroids", columns=matrix.shape[1])
    distances = (
        np.sum(matrix * matrix, axis=1)[:, None]
        + np.sum(centers * centers, axis=1)[None, :]
        - 2.0 * matrix @ centers.T
    )
    return np.maximum(distances, 0.0)


def soft_routing(values: np.ndarray, centroids: np.ndarray, temperature: float) -> np.ndarray:
    """Return deterministic row-normalized soft cluster assignments."""

    if not np.isfinite(temperature) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    logits = -squared_distances(values, centroids) / float(temperature)
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    denominators = np.sum(weights, axis=1, keepdims=True)
    return weights / denominators


def principal_projection_factors(centered: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return padded factors for an orthogonal PCA projection."""

    matrix = finite_matrix(centered, "centered")
    requested_rank = int(rank)
    dimension = matrix.shape[1]
    if not 0 <= requested_rank <= dimension:
        raise ValueError("rank must be between zero and the embedding dimension")
    if requested_rank == 0:
        empty = np.zeros((dimension, 0), dtype=np.float64)
        return empty.copy(), empty.copy()
    _left, _singular, right_t = np.linalg.svd(matrix, full_matrices=False)
    effective_rank = min(requested_rank, right_t.shape[0])
    factors = right_t[:effective_rank, :].T
    if effective_rank < requested_rank:
        factors = np.pad(factors, ((0, 0), (0, requested_rank - effective_rank)))
    return factors.copy(), factors.copy()
