"""Small cross-fittable linear calibration map for regime-level recovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np


# Fixed before validation.  The strong ridge penalty is intentional because
# outer-fold training sets contain only a handful of regimes.
FEATURE_NAMES = (
    "predicted_preservation_rate",
    "oracle_margin_sign_rate",
    "oracle_margin_q10",
    "oracle_margin_q50",
    "fit_error_norm_q50",
    "fit_error_norm_q90",
    "bound_slack_q10",
    "bound_slack_q50",
)
DEFAULT_REGULARIZATION = 10.0
DEFAULT_LOWER_COVERAGE = 0.80


def feature_vector(features: Mapping[str, Any]) -> np.ndarray:
    """Return the predeclared calibration vector from a feature record."""

    summary = features.get("summary", features)
    missing = [name for name in FEATURE_NAMES if name not in summary]
    if missing:
        raise ValueError(f"feature record is missing calibration fields: {missing}")
    vector = np.asarray([summary[name] for name in FEATURE_NAMES], dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError("calibration features must be finite")
    return vector


@dataclass(frozen=True)
class LinearCalibration:
    feature_names: Sequence[str]
    feature_mean: Sequence[float]
    feature_scale: Sequence[float]
    coefficients: Sequence[float]
    intercept: float
    regularization: float
    lower_coverage: float
    lower_offset: float
    training_count: int
    training_regimes: Sequence[str]
    prediction_clip_low: float = 0.0
    prediction_clip_high: float = 1.0
    record_type: str = "recoverability_linear_calibration"
    format_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LinearCalibration":
        expected = payload.get("record_type", "recoverability_linear_calibration")
        if expected != "recoverability_linear_calibration":
            raise ValueError(f"unexpected calibration record type: {expected}")
        fields = {
            key: payload[key]
            for key in cls.__dataclass_fields__
            if key in payload
        }
        return cls(**fields)

    def predict_vector(self, vector: np.ndarray) -> float:
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (len(self.feature_names),):
            raise ValueError("calibration vector has the wrong shape")
        mean = np.asarray(self.feature_mean, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        raw = float(self.intercept + ((values - mean) / scale) @ coefficients)
        return float(np.clip(raw, self.prediction_clip_low, self.prediction_clip_high))

    def predict(self, features: Mapping[str, Any]) -> float:
        return self.predict_vector(feature_vector(features))

    def lower_band(self, features: Mapping[str, Any]) -> float:
        prediction = self.predict(features)
        return float(max(self.prediction_clip_low, prediction - self.lower_offset))


def _fit_core(
    matrix: np.ndarray,
    targets: np.ndarray,
    regularization: float,
) -> tuple:
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (matrix - mean) / scale
    target_mean = float(np.mean(targets))
    centered_targets = targets - target_mean
    penalty = float(regularization) * np.eye(matrix.shape[1], dtype=np.float64)
    coefficients = np.linalg.solve(
        standardized.T @ standardized + penalty,
        standardized.T @ centered_targets,
    )
    return mean, scale, coefficients, target_mean


def _predict_core(
    vector: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
) -> float:
    return float(intercept + ((vector - mean) / scale) @ coefficients)


def _higher_quantile(values: np.ndarray, coverage: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    index = min(len(ordered) - 1, max(0, int(np.ceil(float(coverage) * len(ordered))) - 1))
    return float(ordered[index])


def fit_calibration(
    feature_records: Sequence[Mapping[str, Any]],
    recoveries: Sequence[float],
    regime_names: Sequence[str] = (),
    regularization: float = DEFAULT_REGULARIZATION,
    lower_coverage: float = DEFAULT_LOWER_COVERAGE,
) -> LinearCalibration:
    """Fit a regularized linear map and a training-only one-sided lower offset."""

    if len(feature_records) != len(recoveries) or len(feature_records) == 0:
        raise ValueError("features and recoveries must have the same non-zero length")
    if regime_names and len(regime_names) != len(feature_records):
        raise ValueError("regime_names length mismatch")
    if not np.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("regularization must be finite and positive")
    if not 0.0 < lower_coverage < 1.0:
        raise ValueError("lower_coverage must lie strictly between zero and one")

    matrix = np.vstack([feature_vector(record) for record in feature_records])
    targets = np.asarray(recoveries, dtype=np.float64)
    if not np.all(np.isfinite(targets)):
        raise ValueError("recoveries must be finite")
    mean, scale, coefficients, intercept = _fit_core(matrix, targets, regularization)

    if len(targets) < 2:
        # A single regime cannot calibrate residual uncertainty.  A full recovery
        # point is deliberately subtracted so the lower output is non-assertive.
        lower_offset = 1.0
    else:
        cross_fitted = []
        for held_out in range(len(targets)):
            train = np.arange(len(targets)) != held_out
            fold = _fit_core(matrix[train], targets[train], regularization)
            cross_fitted.append(_predict_core(matrix[held_out], *fold))
        overprediction = np.maximum(np.asarray(cross_fitted) - targets, 0.0)
        lower_offset = _higher_quantile(overprediction, lower_coverage)

    names = tuple(str(name) for name in regime_names) if regime_names else tuple()
    return LinearCalibration(
        feature_names=FEATURE_NAMES,
        feature_mean=mean.tolist(),
        feature_scale=scale.tolist(),
        coefficients=coefficients.tolist(),
        intercept=float(intercept),
        regularization=float(regularization),
        lower_coverage=float(lower_coverage),
        lower_offset=float(lower_offset),
        training_count=int(len(targets)),
        training_regimes=names,
    )


__all__ = [
    "DEFAULT_LOWER_COVERAGE",
    "DEFAULT_REGULARIZATION",
    "FEATURE_NAMES",
    "LinearCalibration",
    "feature_vector",
    "fit_calibration",
]
