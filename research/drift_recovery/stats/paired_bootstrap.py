"""Deterministic paired-query bootstrap for frozen per-query retrieval scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class BootstrapDraws:
    indices: np.ndarray
    seed: int

    @classmethod
    def create(cls, query_count: int, draws: int = 10_000, seed: int = 20260630) -> "BootstrapDraws":
        if query_count < 2:
            raise ValueError("paired bootstrap requires at least two queries")
        if draws < 100:
            raise ValueError("draws must be >= 100")
        rng = np.random.default_rng(seed)
        return cls(indices=rng.integers(0, query_count, size=(draws, query_count)), seed=seed)


def _ci(values: np.ndarray, confidence: float) -> Tuple[float, float]:
    alpha = 1.0 - confidence
    lower, upper = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


def _metric_row(point: float, samples: np.ndarray, confidence: float) -> dict:
    lower, upper = _ci(samples, confidence)
    return {
        "estimate": float(point),
        "ci_low": lower,
        "ci_high": upper,
        "half_width": float((upper - lower) / 2.0),
    }


def paired_query_bootstrap(
    per_query_scores: Mapping[str, Sequence[float]],
    methods: Sequence[str],
    draws: int = 10_000,
    seed: int = 20260630,
    confidence: float = 0.95,
    epsilon: float = 1e-12,
    bootstrap_draws: Optional[BootstrapDraws] = None,
) -> dict:
    """Bootstrap NDCG, delta NDCG, and recovery from the same query draws.

    Recovery is the ratio of resampled query means, never a mean of per-query
    ratios. Draws with a non-positive oracle gap are invalid; more than 1%
    invalid draws blocks the result.
    """

    if "floor" not in per_query_scores or "oracle" not in per_query_scores:
        raise ValueError("floor and oracle scores are required")
    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in per_query_scores.items()}
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("all per-query score arrays must have equal length")
    query_count = lengths.pop()
    paired = bootstrap_draws or BootstrapDraws.create(query_count, draws=draws, seed=seed)
    if paired.indices.shape[1] != query_count:
        raise ValueError("bootstrap draw width does not match query count")
    indices = paired.indices
    floor_draw = arrays["floor"][indices].mean(axis=1)
    oracle_draw = arrays["oracle"][indices].mean(axis=1)
    gap_draw = oracle_draw - floor_draw
    valid = gap_draw > float(epsilon)
    invalid_count = int(np.count_nonzero(~valid))
    invalid_fraction = invalid_count / len(valid)
    if invalid_count * 100 > len(valid):
        raise RuntimeError(
            f"bootstrap invalid fraction {invalid_fraction:.2%} exceeds the 1% protocol limit"
        )
    floor_point = float(arrays["floor"].mean())
    oracle_point = float(arrays["oracle"].mean())
    point_gap = oracle_point - floor_point
    if point_gap <= epsilon:
        raise RuntimeError("point-estimate oracle gap is non-positive")

    result: Dict[str, object] = {
        "record_type": "paired_query_bootstrap",
        "draws": int(indices.shape[0]),
        "query_count": query_count,
        "seed": paired.seed,
        "confidence": confidence,
        "invalid_draws": invalid_count,
        "invalid_fraction": invalid_fraction,
        "floor_ndcg": floor_point,
        "oracle_ndcg": oracle_point,
        "oracle_gap": point_gap,
        "methods": {},
    }
    method_draws: Dict[str, np.ndarray] = {}
    resampled_means: Dict[str, np.ndarray] = {
        "floor": floor_draw,
        "oracle": oracle_draw,
    }
    for method in methods:
        values = arrays[method]
        means = values[indices].mean(axis=1)
        delta = means - floor_draw
        recovery = delta[valid] / gap_draw[valid]
        point_ndcg = float(values.mean())
        point_delta = point_ndcg - floor_point
        method_draws[method] = means
        resampled_means[method] = means
        result["methods"][method] = {
            "ndcg": _metric_row(point_ndcg, means, confidence),
            "delta_ndcg": _metric_row(point_delta, delta, confidence),
            "recovery": _metric_row(point_delta / point_gap, recovery, confidence),
        }
    result["_bootstrap_indices"] = indices
    result["_resampled_means"] = resampled_means
    result["_method_draws"] = method_draws
    return result


def paired_contrast(
    bootstrap_result: Mapping[str, object],
    left: str,
    right: str,
    confidence: float = 0.95,
) -> dict:
    """Compute a paired method contrast and a two-sided bootstrap sign p-value."""

    method_draws = bootstrap_result.get("_method_draws")
    if not isinstance(method_draws, dict) or left not in method_draws or right not in method_draws:
        raise ValueError("bootstrap result does not contain requested method draws")
    samples = np.asarray(method_draws[left]) - np.asarray(method_draws[right])
    methods = bootstrap_result["methods"]
    point = float(methods[left]["ndcg"]["estimate"] - methods[right]["ndcg"]["estimate"])
    # Add one to numerator/denominator for a finite Monte-Carlo p-value.
    non_positive = int(np.count_nonzero(samples <= 0.0))
    non_negative = int(np.count_nonzero(samples >= 0.0))
    p_value = min(1.0, 2.0 * (min(non_positive, non_negative) + 1.0) / (len(samples) + 1.0))
    row = _metric_row(point, samples, confidence)
    row.update({"left": left, "right": right, "p_value": float(p_value)})
    return row


def strip_draws(result: Mapping[str, object]) -> dict:
    """Remove in-memory arrays before JSON serialization."""

    return {key: value for key, value in result.items() if not key.startswith("_")}
