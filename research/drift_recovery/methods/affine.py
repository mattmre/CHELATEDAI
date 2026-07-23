"""Leakage-agnostic affine ridge baseline for D2."""

from __future__ import annotations

from typing import Optional

import numpy as np

from research.drift_recovery.contracts import QueryAdapter
from research.drift_recovery.methods._common import finite_matrix, same_shape_pair


class AffineRidgeAdapter(QueryAdapter):
    """Fit a global ridge map with an unregularized affine intercept."""

    def __init__(self, regularization: float = 1.0) -> None:
        if not np.isfinite(regularization) or float(regularization) <= 0.0:
            raise ValueError("regularization must be finite and positive")
        self.regularization = float(regularization)
        self.dimension_: Optional[int] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None

    def fit(self, source: np.ndarray, target: np.ndarray) -> "AffineRidgeAdapter":
        """Fit from paired vectors without accepting retrieval labels or qrels."""

        source_matrix, target_matrix = same_shape_pair(source, target)
        source_mean = np.mean(source_matrix, axis=0)
        target_mean = np.mean(target_matrix, axis=0)
        centered_source = source_matrix - source_mean
        centered_target = target_matrix - target_mean
        dimension = source_matrix.shape[1]
        gram = centered_source.T @ centered_source
        cross = centered_source.T @ centered_target
        coefficient = np.linalg.solve(
            gram + self.regularization * np.eye(dimension, dtype=np.float64),
            cross,
        )
        intercept = target_mean - source_mean @ coefficient
        if not np.all(np.isfinite(coefficient)) or not np.all(np.isfinite(intercept)):
            raise RuntimeError("ridge fit produced non-finite parameters")
        self.dimension_ = dimension
        self.coef_ = coefficient
        self.intercept_ = intercept
        return self

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """Apply the learned affine map to every provided vector."""

        if self.dimension_ is None or self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("AffineRidgeAdapter must be fitted before transform")
        matrix = finite_matrix(vectors, "vectors", columns=self.dimension_)
        output = matrix @ self.coef_ + self.intercept_
        if not np.all(np.isfinite(output)):
            raise RuntimeError("ridge transform produced non-finite values")
        return np.asarray(output, dtype=np.float64)

    @property
    def parameter_count(self) -> int:
        """Report the number of fitted coefficient and intercept scalars."""

        if self.dimension_ is None:
            raise RuntimeError("AffineRidgeAdapter must be fitted before parameter_count")
        return int(self.dimension_ * self.dimension_ + self.dimension_)


GlobalRidgeAdapter = AffineRidgeAdapter
GlobalRidge = AffineRidgeAdapter
AffineRidge = AffineRidgeAdapter
