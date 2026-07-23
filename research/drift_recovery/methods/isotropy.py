"""Unsupervised isotropy baselines for D2 document embeddings."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from research.drift_recovery.contracts import QueryAdapter
from research.drift_recovery.methods._common import (
    deterministic_kmeans,
    finite_matrix,
    soft_routing,
)


def _validate_optional_target(source: np.ndarray, target: Optional[np.ndarray]) -> None:
    """Validate interface compatibility without using paired targets for fitting."""

    if target is None:
        return
    target_matrix = finite_matrix(target, "target", columns=source.shape[1])
    if target_matrix.shape != source.shape:
        raise ValueError("target must match source shape when supplied")


def _isotropy_matrix(centered: np.ndarray, epsilon: float) -> np.ndarray:
    """Build a scale-preserving ZCA matrix from centered observations."""

    covariance = centered.T @ centered / float(max(1, centered.shape[0]))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    reference_scale = float(np.mean(eigenvalues))
    scales = np.sqrt(reference_scale + epsilon) / np.sqrt(eigenvalues + epsilon)
    return (eigenvectors * scales[None, :]) @ eigenvectors.T


class ZCAAdapter(QueryAdapter):
    """Apply global scale-preserving ZCA whitening learned from source vectors."""

    def __init__(self, epsilon: float = 1e-5) -> None:
        if not np.isfinite(epsilon) or float(epsilon) <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        self.epsilon = float(epsilon)
        self.dimension_: Optional[int] = None
        self.mean_: Optional[np.ndarray] = None
        self.whitener_: Optional[np.ndarray] = None

    def fit(self, source: np.ndarray, target: Optional[np.ndarray] = None) -> "ZCAAdapter":
        """Fit from corrupted source geometry only; a supplied target is ignored."""

        source_matrix = finite_matrix(source, "source")
        _validate_optional_target(source_matrix, target)
        mean = np.mean(source_matrix, axis=0)
        self.dimension_ = source_matrix.shape[1]
        self.mean_ = mean
        self.whitener_ = _isotropy_matrix(source_matrix - mean, self.epsilon)
        return self

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """Transform every row with the fitted global ZCA map."""

        if self.dimension_ is None or self.mean_ is None or self.whitener_ is None:
            raise RuntimeError("ZCAAdapter must be fitted before transform")
        matrix = finite_matrix(vectors, "vectors", columns=self.dimension_)
        output = self.mean_ + (matrix - self.mean_) @ self.whitener_
        if not np.all(np.isfinite(output)):
            raise RuntimeError("ZCA transform produced non-finite values")
        return np.asarray(output, dtype=np.float64)


class AllButTopAdapter(QueryAdapter):
    """Center embeddings and remove their leading principal components."""

    def __init__(self, components: int = 1) -> None:
        if int(components) <= 0:
            raise ValueError("components must be positive")
        self.components = int(components)
        self.dimension_: Optional[int] = None
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None

    def fit(self, source: np.ndarray, target: Optional[np.ndarray] = None) -> "AllButTopAdapter":
        """Fit principal components from corrupted source geometry only."""

        source_matrix = finite_matrix(source, "source")
        _validate_optional_target(source_matrix, target)
        if self.components > source_matrix.shape[1]:
            raise ValueError("components cannot exceed the embedding dimension")
        mean = np.mean(source_matrix, axis=0)
        _left, _singular, right_t = np.linalg.svd(source_matrix - mean, full_matrices=False)
        if self.components > right_t.shape[0]:
            raise ValueError("components cannot exceed the available source rank")
        self.dimension_ = source_matrix.shape[1]
        self.mean_ = mean
        self.components_ = right_t[: self.components].copy()
        return self

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """Center all rows and remove their fitted top-component projections."""

        if self.dimension_ is None or self.mean_ is None or self.components_ is None:
            raise RuntimeError("AllButTopAdapter must be fitted before transform")
        matrix = finite_matrix(vectors, "vectors", columns=self.dimension_)
        centered = matrix - self.mean_
        output = centered - (centered @ self.components_.T) @ self.components_
        if not np.all(np.isfinite(output)):
            raise RuntimeError("all-but-top transform produced non-finite values")
        return np.asarray(output, dtype=np.float64)


class CBIEAdapter(QueryAdapter):
    """Cluster-based isotropy enhancement with deterministic soft routing."""

    def __init__(
        self,
        cluster_count: int = 16,
        epsilon: float = 1e-5,
        routing_temperature: float = 1.0,
        kmeans_iterations: int = 50,
    ) -> None:
        if int(cluster_count) <= 0:
            raise ValueError("cluster_count must be positive")
        if not np.isfinite(epsilon) or float(epsilon) <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if not np.isfinite(routing_temperature) or float(routing_temperature) <= 0.0:
            raise ValueError("routing_temperature must be finite and positive")
        if int(kmeans_iterations) <= 0:
            raise ValueError("kmeans_iterations must be positive")
        self.cluster_count = int(cluster_count)
        self.epsilon = float(epsilon)
        self.routing_temperature = float(routing_temperature)
        self.kmeans_iterations = int(kmeans_iterations)
        self.dimension_: Optional[int] = None
        self.centroids_: Optional[np.ndarray] = None
        self.whiteners_: Optional[List[np.ndarray]] = None

    def fit(self, source: np.ndarray, target: Optional[np.ndarray] = None) -> "CBIEAdapter":
        """Fit cluster geometry from source only; paired targets do not affect it."""

        source_matrix = finite_matrix(source, "source")
        _validate_optional_target(source_matrix, target)
        if source_matrix.shape[0] < self.cluster_count:
            raise ValueError("source must have at least cluster_count rows")
        centroids, assignments = deterministic_kmeans(
            source_matrix,
            self.cluster_count,
            iterations=self.kmeans_iterations,
        )
        whiteners = []
        for cluster in range(self.cluster_count):
            members = source_matrix[assignments == cluster]
            centered = members - centroids[cluster]
            whiteners.append(_isotropy_matrix(centered, self.epsilon))
        self.dimension_ = source_matrix.shape[1]
        self.centroids_ = centroids
        self.whiteners_ = whiteners
        return self

    def routing_weights(self, vectors: np.ndarray) -> np.ndarray:
        """Return soft cluster routing probabilities."""

        if self.dimension_ is None or self.centroids_ is None:
            raise RuntimeError("CBIEAdapter must be fitted before routing")
        matrix = finite_matrix(vectors, "vectors", columns=self.dimension_)
        return soft_routing(matrix, self.centroids_, self.routing_temperature)

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """Apply all local isotropy maps to every vector via soft routing."""

        if self.dimension_ is None or self.centroids_ is None or self.whiteners_ is None:
            raise RuntimeError("CBIEAdapter must be fitted before transform")
        matrix = finite_matrix(vectors, "vectors", columns=self.dimension_)
        routing = soft_routing(matrix, self.centroids_, self.routing_temperature)
        output = np.zeros_like(matrix, dtype=np.float64)
        for cluster, whitener in enumerate(self.whiteners_):
            local_output = self.centroids_[cluster] + (
                matrix - self.centroids_[cluster]
            ) @ whitener
            output += routing[:, cluster, None] * local_output
        if not np.all(np.isfinite(output)):
            raise RuntimeError("CBIE transform produced non-finite values")
        return output


CBIE = CBIEAdapter
ZCA = ZCAAdapter
AllButTop = AllButTopAdapter
