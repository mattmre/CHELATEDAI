"""Public estimator entry point backed by a serialized regime calibrator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from research.drift_recovery.artifacts import EmbeddingPack
from research.drift_recovery.estimator.calibration import LinearCalibration
from research.drift_recovery.estimator.features import extract_regime_features


DEFAULT_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[1] / "out" / "estimator" / "calibration.json"
)


def load_calibration(path: Path = DEFAULT_CALIBRATION_PATH) -> LinearCalibration:
    selected = Path(path)
    if not selected.exists():
        raise FileNotFoundError(
            f"calibration artifact not found at {selected}; run the D3 estimator build first"
        )
    payload = json.loads(selected.read_text(encoding="utf-8"))
    return LinearCalibration.from_dict(payload)


def estimate_recovery(
    pack: EmbeddingPack,
    calibration: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Estimate ridge oracle-gap recovery without calculating corrected NDCG.

    Validation of packs used to train the default calibration must use the
    separate leave-one-regime-out path; applying the all-regime artifact back to
    a training pack is an in-sample prediction and is not validation evidence.
    """

    features = extract_regime_features(pack)
    calibrator = (
        LinearCalibration.from_dict(calibration)
        if calibration is not None
        else load_calibration()
    )
    estimate = calibrator.predict(features)
    lower = calibrator.lower_band(features)
    return {
        "record_type": "recoverability_estimate",
        "R_hat": estimate,
        "lower_band": lower,
        "features": features,
        "calibration_training_count": calibrator.training_count,
        "calibration_training_regimes": list(calibrator.training_regimes),
        "scope_note": (
            "Margin-bound violations are predictors, not proven inversions; "
            "Gram/margin distortion alone does not determine NDCG@k."
        ),
    }


__all__ = ["DEFAULT_CALIBRATION_PATH", "estimate_recovery", "load_calibration"]
