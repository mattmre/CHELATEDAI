"""Order-preservation bound for the drift-recovery estimator.

For a query ``q``, relevant document ``r``, and strongest non-relevant
competitor ``j``, the clean oracle margin is sufficient to preserve their
order when

``clean_margin > ||q|| * (||e_r|| + ||e_j||)``.

A violation of this sufficient condition is an *inversion prediction*, not
proof that the corrected ranking is inverted.  The functions here implement
that distinction directly and operate only on NumPy values.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import ArrayLike


NumericResult = Union[np.ndarray, np.generic]


def _finite_array(value: ArrayLike, name: str) -> np.ndarray:
    """Convert ``value`` to a finite floating-point NumPy array."""

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _norm_array(value: ArrayLike, name: str) -> np.ndarray:
    """Convert ``value`` to a finite array of nonnegative norms."""

    array = _finite_array(value, name)
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return array


def order_preservation_slack(
    clean_margin: ArrayLike,
    query_norm: ArrayLike,
    relevant_error_norm: ArrayLike,
    competitor_error_norm: ArrayLike,
) -> NumericResult:
    """Return the amount by which the sufficient bound is satisfied.

    The returned value is

    ``clean_margin - query_norm * (relevant_error_norm + competitor_error_norm)``.

    Positive slack guarantees order preservation under the stated bound.
    Zero or negative slack means only that the bound is violated.  Inputs use
    normal NumPy broadcasting, making the function suitable for either one
    hand-checkable query or a vector of per-query quantities.

    Raises:
        ValueError: If an input is non-numeric/non-finite, if a norm is
            negative, if shapes cannot broadcast, or if arithmetic overflows.
    """

    margin = _finite_array(clean_margin, "clean_margin")
    q_norm = _norm_array(query_norm, "query_norm")
    relevant_norm = _norm_array(relevant_error_norm, "relevant_error_norm")
    competitor_norm = _norm_array(competitor_error_norm, "competitor_error_norm")

    try:
        with np.errstate(over="raise", invalid="raise"):
            error_sum = np.add(relevant_norm, competitor_norm)
            penalty = np.multiply(q_norm, error_sum)
            slack = np.subtract(margin, penalty)
    except (FloatingPointError, ValueError) as exc:
        raise ValueError("margin-bound inputs must broadcast to finite results") from exc

    if not np.all(np.isfinite(slack)):
        raise ValueError("margin-bound arithmetic produced a non-finite result")
    return slack


def order_is_preserved(
    clean_margin: ArrayLike,
    query_norm: ArrayLike,
    relevant_error_norm: ArrayLike,
    competitor_error_norm: ArrayLike,
) -> NumericResult:
    """Return where the strict sufficient order-preservation bound holds."""

    return order_preservation_slack(
        clean_margin,
        query_norm,
        relevant_error_norm,
        competitor_error_norm,
    ) > 0.0


def predict_inversion(
    clean_margin: ArrayLike,
    query_norm: ArrayLike,
    relevant_error_norm: ArrayLike,
    competitor_error_norm: ArrayLike,
) -> NumericResult:
    """Predict bound violations, including equality, query by query.

    ``True`` means

    ``query_norm * (relevant_error_norm + competitor_error_norm) >= clean_margin``.

    This is deliberately the logical complement of :func:`order_is_preserved`.
    It identifies queries without a preservation guarantee; it does not assert
    that their actual corrected retrieval order is inverted.
    """

    return order_preservation_slack(
        clean_margin,
        query_norm,
        relevant_error_norm,
        competitor_error_norm,
    ) <= 0.0


__all__ = [
    "order_is_preserved",
    "order_preservation_slack",
    "predict_inversion",
]
