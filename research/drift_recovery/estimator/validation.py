"""Leave-one-regime-out validation for the recoverability estimator."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from research.drift_recovery.estimator.calibration import fit_calibration


LOW_MEDIUM_THRESHOLD = 1.0 / 3.0
MEDIUM_HIGH_THRESHOLD = 2.0 / 3.0
MIN_VALIDATED_REGIMES = 3


def recovery_bin(value: float) -> str:
    """Map recovery to fixed, predeclared low/medium/high thirds."""

    if float(value) < LOW_MEDIUM_THRESHOLD:
        return "low"
    if float(value) < MEDIUM_HIGH_THRESHOLD:
        return "medium"
    return "high"


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]):
    """Compute Spearman rho with average tie ranks, or None if undefined."""

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        return None
    x_rank = _average_ranks(x)
    y_rank = _average_ranks(y)
    if np.std(x_rank) == 0.0 or np.std(y_rank) == 0.0:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def leave_one_regime_out(regimes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Predict each regime using calibrator targets from every other regime only."""

    names = [str(item["name"]) for item in regimes]
    if len(set(names)) != len(names):
        raise ValueError("regime names must be unique")
    rows = []
    actual = []
    predicted = []
    for held_out, regime in enumerate(regimes):
        train = [item for index, item in enumerate(regimes) if index != held_out]
        if not train:
            rows.append(
                {
                    "regime": names[held_out],
                    "actual_recovery": float(regime["recovery"]),
                    "R_hat": None,
                    "lower_band": None,
                    "actual_bin": recovery_bin(float(regime["recovery"])),
                    "predicted_bin": None,
                    "bin_correct": False,
                    "training_regimes": [],
                }
            )
            continue
        calibrator = fit_calibration(
            [item["features"] for item in train],
            [float(item["recovery"]) for item in train],
            regime_names=[str(item["name"]) for item in train],
        )
        estimate = calibrator.predict(regime["features"])
        lower = calibrator.lower_band(regime["features"])
        actual_value = float(regime["recovery"])
        actual_bin = recovery_bin(actual_value)
        predicted_bin = recovery_bin(estimate)
        rows.append(
            {
                "regime": names[held_out],
                "actual_recovery": actual_value,
                "R_hat": estimate,
                "lower_band": lower,
                "actual_bin": actual_bin,
                "predicted_bin": predicted_bin,
                "bin_correct": predicted_bin == actual_bin,
                "lower_band_covers": lower <= actual_value,
                "training_regimes": list(calibrator.training_regimes),
            }
        )
        actual.append(actual_value)
        predicted.append(estimate)

    held_out_count = len(predicted)
    rho = spearman_correlation(predicted, actual)
    accuracy = (
        float(np.mean([row["bin_correct"] for row in rows if row["R_hat"] is not None]))
        if held_out_count
        else None
    )
    coverage = (
        float(np.mean([row["lower_band_covers"] for row in rows if row["R_hat"] is not None]))
        if held_out_count
        else None
    )
    enough_regimes = held_out_count >= MIN_VALIDATED_REGIMES
    success = bool(
        enough_regimes
        and ((rho is not None and rho >= 0.7) or (accuracy is not None and accuracy >= 0.80))
    )
    if success:
        verdict = "SHIP-POSITIVE"
        note = (
            "Leave-one-regime-out validation meets the locked success bar. "
            "This remains a calibrated estimator, not a certificate that the bound controls NDCG."
        )
    elif not enough_regimes:
        verdict = "SCOPING-NEGATIVE"
        note = (
            f"Only {held_out_count} held-out regimes were available; at least "
            f"{MIN_VALIDATED_REGIMES} are required for the locked cross-regime verdict."
        )
    else:
        verdict = "VALIDATION-NEGATIVE"
        rho_text = "undefined" if rho is None else f"{rho:.3f}"
        accuracy_text = "undefined" if accuracy is None else f"{100.0 * accuracy:.1f}%"
        note = (
            f"Across {held_out_count} held-out regimes, Spearman={rho_text} and "
            f"3-way bin accuracy={accuracy_text}; neither locked success threshold was met."
        )

    return {
        "record_type": "recoverability_leave_one_regime_out_validation",
        "held_out_axis": "regime (dataset / encoder family / anchor setting)",
        "regime_count": int(len(regimes)),
        "held_out_count": int(held_out_count),
        "spearman": rho,
        "bin_accuracy": accuracy,
        "lower_band_coverage": coverage,
        "bin_thresholds": {
            "low": f"R < {LOW_MEDIUM_THRESHOLD:.6f}",
            "medium": (
                f"{LOW_MEDIUM_THRESHOLD:.6f} <= R < {MEDIUM_HIGH_THRESHOLD:.6f}"
            ),
            "high": f"R >= {MEDIUM_HIGH_THRESHOLD:.6f}",
        },
        "success_bar": "Spearman >= 0.7 OR bin accuracy >= 0.80, across >=3 regimes",
        "verdict": verdict,
        "scoping_note": note,
        "rows": rows,
    }


__all__ = [
    "leave_one_regime_out",
    "recovery_bin",
    "spearman_correlation",
]
