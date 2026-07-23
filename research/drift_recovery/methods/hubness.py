"""Deterministic document-hub score scaling for D2 retrieval matrices."""

from __future__ import annotations

from typing import Optional

import numpy as np

from research.drift_recovery.contracts import ScoreTransform
from research.drift_recovery.methods._common import finite_matrix


class HubnessScoreScaling(ScoreTransform):
    """Down-scale scores for documents that are unusually strong across queries.

    Fitting consumes an anchor-query by document score matrix only. For each
    document, the mean of its strongest anchor-query scores estimates hubness.
    Only positive robust-standardized excess is penalized, preserving ordinary
    and below-median documents.
    """

    def __init__(self, reference_k: int = 10, strength: float = 1.0) -> None:
        if int(reference_k) <= 0:
            raise ValueError("reference_k must be positive")
        if not np.isfinite(strength) or float(strength) < 0.0:
            raise ValueError("strength must be finite and nonnegative")
        self.reference_k = int(reference_k)
        self.strength = float(strength)
        self.document_count_: Optional[int] = None
        self.hubness_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, scores: np.ndarray) -> "HubnessScoreScaling":
        """Fit document scales from anchor scores without qrels or eval labels."""

        matrix = finite_matrix(scores, "scores")
        reference_count = min(self.reference_k, matrix.shape[0])
        boundary = matrix.shape[0] - reference_count
        strongest = np.partition(matrix, boundary, axis=0)[boundary:, :]
        hubness = np.mean(strongest, axis=0)
        center = float(np.median(hubness))
        absolute_deviation = np.abs(hubness - center)
        robust_scale = float(1.4826 * np.median(absolute_deviation))
        if robust_scale <= np.finfo(np.float64).eps:
            robust_scale = float(np.std(hubness, ddof=0))
        if robust_scale <= np.finfo(np.float64).eps:
            excess = np.zeros_like(hubness)
        else:
            excess = np.maximum((hubness - center) / robust_scale, 0.0)
        scale = 1.0 / (1.0 + self.strength * excess)
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise RuntimeError("hubness fit produced invalid score scales")
        self.document_count_ = matrix.shape[1]
        self.hubness_ = np.asarray(hubness, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)
        return self

    def transform_scores(self, scores: np.ndarray) -> np.ndarray:
        """Apply fitted document-wise scaling while preserving matrix shape."""

        if self.document_count_ is None or self.scale_ is None:
            raise RuntimeError("HubnessScoreScaling must be fitted before score transformation")
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim not in (1, 2) or values.size == 0:
            raise ValueError("scores must be a non-empty one- or two-dimensional array")
        if values.shape[-1] != self.document_count_:
            raise ValueError(
                f"scores have {values.shape[-1]} documents; expected {self.document_count_}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("scores must contain only finite values")
        # Scale each score's nonnegative distance above that query's minimum.
        # This penalizes hubs even when raw cosine scores happen to be negative.
        baseline = np.min(values, axis=-1, keepdims=True)
        adjusted = baseline + (values - baseline) * self.scale_
        if not np.all(np.isfinite(adjusted)):
            raise RuntimeError("hubness score transformation produced non-finite values")
        return np.asarray(adjusted, dtype=np.float64)

    def rank(self, scores: np.ndarray, k: int) -> np.ndarray:
        """Return stable descending top-k document indices for each score row."""

        if int(k) <= 0:
            raise ValueError("k must be positive")
        adjusted = self.transform_scores(scores)
        document_count = adjusted.shape[-1]
        top_k = min(int(k), document_count)
        if adjusted.ndim == 1:
            return np.argsort(-adjusted, kind="mergesort")[:top_k].astype(np.int64)
        order = np.argsort(-adjusted, axis=1, kind="mergesort")
        return np.asarray(order[:, :top_k], dtype=np.int64)


HubnessScoreScaler = HubnessScoreScaling
