"""Parameter-matched paired global-plus-local low-rank diagnostic."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from research.drift_recovery.contracts import QueryAdapter
from research.drift_recovery.methods._common import (
    deterministic_kmeans,
    finite_matrix,
    parameter_budget,
    same_shape_pair,
    soft_routing,
    truncated_map_factors,
    validate_architecture,
)


def allocated_parameter_count(
    dimension: int,
    global_rank: int,
    local_rank: int,
    cluster_count: int,
) -> int:
    """Return the preregistered low-rank allocation for this architecture."""

    return parameter_budget(dimension, global_rank, local_rank, cluster_count)


class SoftRoutedLocalAdapter(QueryAdapter):
    """Fit a global low-rank residual plus soft-routed local residual maps.

    This is deliberately a paired diagnostic: it learns from corrupted/clean
    vector pairs. Routing is learned from corrupted vectors only and remains
    deterministic under repeated fits with identical row order.
    """

    def __init__(
        self,
        dimension: int,
        global_rank: int,
        local_rank: int,
        cluster_count: int,
        regularization: float = 1.0,
        routing_temperature: float = 1.0,
        kmeans_iterations: int = 50,
    ) -> None:
        validate_architecture(dimension, global_rank, local_rank, cluster_count)
        if not np.isfinite(regularization) or float(regularization) <= 0.0:
            raise ValueError("regularization must be finite and positive")
        if not np.isfinite(routing_temperature) or float(routing_temperature) <= 0.0:
            raise ValueError("routing_temperature must be finite and positive")
        if int(kmeans_iterations) <= 0:
            raise ValueError("kmeans_iterations must be positive")
        self.dimension = int(dimension)
        self.global_rank = int(global_rank)
        self.local_rank = int(local_rank)
        self.cluster_count = int(cluster_count)
        self.regularization = float(regularization)
        self.routing_temperature = float(routing_temperature)
        self.kmeans_iterations = int(kmeans_iterations)
        self.centroids_: Optional[np.ndarray] = None
        self.global_factors_: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.local_factors_: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None

    @property
    def allocated_parameter_count(self) -> int:
        """Report the allocation shared with bounded chelation."""

        return allocated_parameter_count(
            self.dimension,
            self.global_rank,
            self.local_rank,
            self.cluster_count,
        )

    def fit(self, source: np.ndarray, target: np.ndarray) -> "SoftRoutedLocalAdapter":
        """Fit paired residual maps without accepting qrels or eval labels."""

        source_matrix, target_matrix = same_shape_pair(source, target)
        if source_matrix.shape[1] != self.dimension:
            raise ValueError(
                f"source has dimension {source_matrix.shape[1]}; expected {self.dimension}"
            )
        if source_matrix.shape[0] < self.cluster_count:
            raise ValueError("source must have at least cluster_count rows")

        residual = target_matrix - source_matrix
        global_factors = truncated_map_factors(
            source_matrix,
            residual,
            rank=self.global_rank,
            regularization=self.regularization,
        )
        global_prediction = source_matrix @ global_factors[0] @ global_factors[1].T
        remaining_residual = residual - global_prediction
        centroids, _assignments = deterministic_kmeans(
            source_matrix,
            self.cluster_count,
            iterations=self.kmeans_iterations,
        )
        routing = soft_routing(source_matrix, centroids, self.routing_temperature)
        local_factors = []
        for cluster in range(self.cluster_count):
            local_factors.append(
                truncated_map_factors(
                    source_matrix,
                    remaining_residual,
                    rank=self.local_rank,
                    regularization=self.regularization,
                    weights=routing[:, cluster],
                )
            )
        self.centroids_ = centroids
        self.global_factors_ = global_factors
        self.local_factors_ = local_factors
        return self

    def routing_weights(self, vectors: np.ndarray) -> np.ndarray:
        """Return soft routing probabilities for fitted cluster experts."""

        if self.centroids_ is None:
            raise RuntimeError("SoftRoutedLocalAdapter must be fitted before routing")
        matrix = finite_matrix(vectors, "vectors", columns=self.dimension)
        return soft_routing(matrix, self.centroids_, self.routing_temperature)

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """Apply global and soft-routed local corrections to all vectors."""

        if self.global_factors_ is None or self.local_factors_ is None or self.centroids_ is None:
            raise RuntimeError("SoftRoutedLocalAdapter must be fitted before transform")
        matrix = finite_matrix(vectors, "vectors", columns=self.dimension)
        correction = matrix @ self.global_factors_[0] @ self.global_factors_[1].T
        routing = soft_routing(matrix, self.centroids_, self.routing_temperature)
        for cluster, (left, right) in enumerate(self.local_factors_):
            local_correction = matrix @ left @ right.T
            correction += routing[:, cluster, None] * local_correction
        output = matrix + correction
        if not np.all(np.isfinite(output)):
            raise RuntimeError("local adapter transform produced non-finite values")
        return np.asarray(output, dtype=np.float64)


LowRankLocalAdapter = SoftRoutedLocalAdapter
LocalLowRankAdapter = SoftRoutedLocalAdapter
LocalAdapter = SoftRoutedLocalAdapter
