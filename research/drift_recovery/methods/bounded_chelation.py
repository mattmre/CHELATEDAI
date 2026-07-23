"""Unpaired detector-gated radial de-collapse for the D2 kill-screen."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from research.drift_recovery.contracts import QueryAdapter
from research.drift_recovery.methods._common import (
    deterministic_kmeans,
    finite_matrix,
    parameter_budget,
    principal_projection_factors,
    squared_distances,
    validate_architecture,
)


def allocated_parameter_count(
    dimension: int,
    global_rank: int,
    local_rank: int,
    cluster_count: int,
) -> int:
    """Return the exact allocation shared with the paired local adapter."""

    return parameter_budget(dimension, global_rank, local_rank, cluster_count)


class DetectorGatedChelationAdapter(QueryAdapter):
    """Expand detector-flagged cluster geometry along fitted PCA directions.

    The adapter is intentionally unpaired. Its factors, cluster centers, and
    gates are functions only of corrupted vectors, cluster assignments, and
    precomputed detector probabilities. A target argument is accepted solely
    for compatibility with ``QueryAdapter`` and is never used to fit state.

    ``alpha=None`` is the primary unbounded arm. A finite alpha caps each row's
    displacement at ``alpha * ||x||``, making ``alpha=0.05`` the separate
    near-identity stress arm.
    """

    def __init__(
        self,
        dimension: int,
        global_rank: int,
        local_rank: int,
        cluster_count: int,
        alpha: Optional[float] = None,
        radial_gain: float = 0.1,
        detector_threshold: float = 0.5,
        kmeans_iterations: int = 50,
    ) -> None:
        validate_architecture(dimension, global_rank, local_rank, cluster_count)
        if alpha is not None and (not np.isfinite(alpha) or float(alpha) <= 0.0):
            raise ValueError("alpha must be None or finite and positive")
        if not np.isfinite(radial_gain) or float(radial_gain) < 0.0:
            raise ValueError("radial_gain must be finite and nonnegative")
        if not np.isfinite(detector_threshold) or not 0.0 <= float(detector_threshold) <= 1.0:
            raise ValueError("detector_threshold must be in [0, 1]")
        if int(kmeans_iterations) <= 0:
            raise ValueError("kmeans_iterations must be positive")
        self.dimension = int(dimension)
        self.global_rank = int(global_rank)
        self.local_rank = int(local_rank)
        self.cluster_count = int(cluster_count)
        self.alpha = None if alpha is None else float(alpha)
        self.radial_gain = float(radial_gain)
        self.detector_threshold = float(detector_threshold)
        self.kmeans_iterations = int(kmeans_iterations)

        self.global_mean_: Optional[np.ndarray] = None
        self.centroids_: Optional[np.ndarray] = None
        self.global_factors_: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.local_factors_: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        self.cluster_probabilities_: Optional[np.ndarray] = None
        self.fit_source_: Optional[np.ndarray] = None
        self.fit_assignments_: Optional[np.ndarray] = None
        self.fit_diagnostics_: Optional[Dict[str, float]] = None
        self.last_diagnostics_: Optional[Dict[str, float]] = None
        self.mean_displacement_: Optional[float] = None
        self.max_displacement_: Optional[float] = None
        self.mean_relative_displacement_: Optional[float] = None
        self.max_relative_displacement_: Optional[float] = None

    @property
    def allocated_parameter_count(self) -> int:
        """Report the preregistered parameter allocation."""

        return allocated_parameter_count(
            self.dimension,
            self.global_rank,
            self.local_rank,
            self.cluster_count,
        )

    def fit(
        self,
        source: np.ndarray,
        target: Optional[np.ndarray] = None,
        cluster_assignments: Optional[np.ndarray] = None,
        detector_probabilities: Optional[np.ndarray] = None,
    ) -> "DetectorGatedChelationAdapter":
        """Fit unpaired radial factors and cluster gates from corrupted geometry."""

        source_matrix = finite_matrix(source, "source", columns=self.dimension)
        if target is not None:
            target_matrix = finite_matrix(target, "target", columns=self.dimension)
            if target_matrix.shape != source_matrix.shape:
                raise ValueError("target must match source shape when supplied")
            # Target values are intentionally not referenced below this point.
        if source_matrix.shape[0] < self.cluster_count:
            raise ValueError("source must have at least cluster_count rows")

        if cluster_assignments is None:
            _initial_centroids, assignments = deterministic_kmeans(
                source_matrix,
                self.cluster_count,
                iterations=self.kmeans_iterations,
            )
        else:
            raw_assignments = np.asarray(cluster_assignments)
            if raw_assignments.shape != (source_matrix.shape[0],):
                raise ValueError("cluster_assignments must have one value per source row")
            if not np.all(np.isfinite(raw_assignments)):
                raise ValueError("cluster_assignments must be finite")
            assignments = raw_assignments.astype(np.int64)
            if not np.array_equal(raw_assignments, assignments):
                raise ValueError("cluster_assignments must contain integer labels")
            if np.any(assignments < 0) or np.any(assignments >= self.cluster_count):
                raise ValueError("cluster_assignments labels are out of range")
            present = np.unique(assignments)
            if len(present) != self.cluster_count:
                raise ValueError("cluster_assignments must represent every configured cluster")

        if detector_probabilities is None:
            raise ValueError("detector_probabilities are required for detector-gated fitting")
        probabilities = np.asarray(detector_probabilities, dtype=np.float64)
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0) or np.any(
            probabilities > 1.0
        ):
            raise ValueError("detector_probabilities must be finite values in [0, 1]")
        if probabilities.shape == (self.cluster_count,):
            cluster_probabilities = probabilities.copy()
        elif probabilities.shape == (source_matrix.shape[0],):
            cluster_probabilities = np.asarray(
                [np.mean(probabilities[assignments == cluster]) for cluster in range(self.cluster_count)],
                dtype=np.float64,
            )
        else:
            raise ValueError(
                "detector_probabilities must have one value per cluster or source row"
            )

        global_mean = np.mean(source_matrix, axis=0)
        global_factors = principal_projection_factors(
            source_matrix - global_mean,
            self.global_rank,
        )
        centroids = np.zeros((self.cluster_count, self.dimension), dtype=np.float64)
        local_factors = []
        for cluster in range(self.cluster_count):
            members = source_matrix[assignments == cluster]
            centroid = np.mean(members, axis=0)
            centroids[cluster] = centroid
            local_factors.append(
                principal_projection_factors(members - centroid, self.local_rank)
            )

        self.global_mean_ = global_mean
        self.centroids_ = centroids
        self.global_factors_ = global_factors
        self.local_factors_ = local_factors
        self.cluster_probabilities_ = cluster_probabilities
        self.fit_source_ = source_matrix.copy()
        self.fit_assignments_ = assignments.copy()
        fitted_output = self._transform_with_assignments(source_matrix, assignments)
        self.fit_diagnostics_ = self._displacement_diagnostics(source_matrix, fitted_output)
        self._record_diagnostics(self.fit_diagnostics_)
        return self

    def _require_fitted(self) -> None:
        if (
            self.global_mean_ is None
            or self.centroids_ is None
            or self.global_factors_ is None
            or self.local_factors_ is None
            or self.cluster_probabilities_ is None
        ):
            raise RuntimeError("DetectorGatedChelationAdapter must be fitted before transform")

    def _route(self, matrix: np.ndarray) -> np.ndarray:
        self._require_fitted()
        assert self.centroids_ is not None
        if (
            self.fit_source_ is not None
            and self.fit_assignments_ is not None
            and np.array_equal(matrix, self.fit_source_)
        ):
            return self.fit_assignments_.copy()
        return np.argmin(squared_distances(matrix, self.centroids_), axis=1).astype(np.int64)

    def _transform_with_assignments(self, matrix: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        self._require_fitted()
        assert self.global_mean_ is not None
        assert self.centroids_ is not None
        assert self.global_factors_ is not None
        assert self.local_factors_ is not None
        assert self.cluster_probabilities_ is not None

        global_left, global_right = self.global_factors_
        global_projection = (matrix - self.global_mean_) @ global_left @ global_right.T
        correction = np.zeros_like(matrix, dtype=np.float64)
        for cluster, (local_left, local_right) in enumerate(self.local_factors_):
            member_mask = assignments == cluster
            if not np.any(member_mask):
                continue
            probability = float(self.cluster_probabilities_[cluster])
            gate = probability if probability >= self.detector_threshold else 0.0
            if gate == 0.0 or self.radial_gain == 0.0:
                continue
            local_values = matrix[member_mask] - self.centroids_[cluster]
            local_projection = local_values @ local_left @ local_right.T
            correction[member_mask] = (
                self.radial_gain
                * gate
                * (global_projection[member_mask] + local_projection)
            )

        if self.alpha is not None:
            displacement_norm = np.linalg.norm(correction, axis=1)
            vector_norm = np.linalg.norm(matrix, axis=1)
            cap = self.alpha * vector_norm
            scale = np.ones_like(displacement_norm)
            nonzero = displacement_norm > 0.0
            scale[nonzero] = np.minimum(1.0, cap[nonzero] / displacement_norm[nonzero])
            correction *= scale[:, None]
        output = matrix + correction
        if not np.all(np.isfinite(output)):
            raise RuntimeError("chelation transform produced non-finite values")
        return output

    @staticmethod
    def _displacement_diagnostics(source: np.ndarray, output: np.ndarray) -> Dict[str, float]:
        displacement = np.linalg.norm(output - source, axis=1)
        source_norm = np.linalg.norm(source, axis=1)
        relative = displacement / np.where(source_norm > 0.0, source_norm, 1.0)
        return {
            "mean_displacement": float(np.mean(displacement)),
            "max_displacement": float(np.max(displacement)),
            "mean_relative_displacement": float(np.mean(relative)),
            "max_relative_displacement": float(np.max(relative)),
            "moved_fraction": float(np.mean(displacement > 0.0)),
        }

    @property
    def mean_displacement(self) -> float:
        """Return mean displacement from the most recent transform."""

        if self.mean_displacement_ is None:
            raise RuntimeError("no displacement diagnostics are available before fit")
        return float(self.mean_displacement_)

    @property
    def displacement_diagnostics(self) -> Dict[str, float]:
        """Return a copy of the most recent displacement diagnostics."""

        if self.last_diagnostics_ is None:
            raise RuntimeError("no displacement diagnostics are available before fit")
        return dict(self.last_diagnostics_)

    def _record_diagnostics(self, diagnostics: Dict[str, float]) -> None:
        self.last_diagnostics_ = dict(diagnostics)
        self.mean_displacement_ = float(diagnostics["mean_displacement"])
        self.max_displacement_ = float(diagnostics["max_displacement"])
        self.mean_relative_displacement_ = float(diagnostics["mean_relative_displacement"])
        self.max_relative_displacement_ = float(diagnostics["max_relative_displacement"])

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """Apply detector-gated radial corrections to every provided row."""

        self._require_fitted()
        matrix = finite_matrix(vectors, "vectors", columns=self.dimension)
        assignments = self._route(matrix)
        output = self._transform_with_assignments(matrix, assignments)
        self._record_diagnostics(self._displacement_diagnostics(matrix, output))
        return np.asarray(output, dtype=np.float64)


BoundedChelationAdapter = DetectorGatedChelationAdapter
BoundedChelation = DetectorGatedChelationAdapter
