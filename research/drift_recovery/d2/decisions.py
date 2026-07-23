"""Binding G2-before-G3 kill-screen decisions for D2."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


FROZEN_CONTRASTS = (
    "chelation_vs_cbie",
    "chelation_vs_hubness",
    "chelation_vs_global_ridge",
)


def evaluate_g2(
    contrasts: Mapping[str, Mapping[str, float]],
    query_count: int,
    threshold: float = 0.015,
    family: Sequence[str] = FROZEN_CONTRASTS,
) -> Mapping[str, Any]:
    missing = [name for name in family if name not in contrasts]
    if missing:
        raise ValueError(f"missing preregistered G2 contrasts: {missing}")
    half_widths = [float(contrasts[name]["half_width"]) for name in family]
    if any(not np.isfinite(value) or value < 0.0 for value in half_widths):
        raise ValueError("G2 half-widths must be finite and nonnegative")
    median = float(np.median(half_widths))
    rough_count = int(math.ceil(int(query_count) * (median / float(threshold)) ** 2))
    return {
        "gate": "G2",
        "family": list(family),
        "contrast_half_widths": dict(zip(family, half_widths)),
        "median_half_width": median,
        "threshold": float(threshold),
        "pass": bool(median <= float(threshold)),
        "query_count": int(query_count),
        "rough_query_count_for_threshold": max(int(query_count), rough_count),
    }


def _contrast_passes_g3(row: Mapping[str, Any], seed_count: int) -> Mapping[str, Any]:
    ndcg = row["delta_ndcg"]
    recovery = row["recovery_point_advantage"]
    seed_deltas = [float(value) for value in row["seed_delta_ndcg"]]
    holm = row.get("multiple_testing", {})
    recovery_estimate = recovery.get("estimate")
    recovery_ci_low = recovery.get("ci_low")
    checks = {
        "absolute_ndcg_at_least_0_02": float(ndcg["estimate"]) >= 0.02,
        "ndcg_ci_excludes_zero_positive": float(ndcg["ci_low"]) > 0.0,
        "recovery_fraction_at_least_0_05": (
            recovery_estimate is not None and float(recovery_estimate) >= 0.05
        ),
        "recovery_ci_excludes_zero_positive": (
            recovery_ci_low is not None and float(recovery_ci_low) > 0.0
        ),
        "same_positive_sign_all_seeds": (
            len(seed_deltas) == int(seed_count) and all(value > 0.0 for value in seed_deltas)
        ),
        "holm_rejects": bool(holm.get("reject", False)),
    }
    return {"pass": all(checks.values()), "checks": checks}


def apply_kill_screen(
    *,
    g2: Mapping[str, Any],
    contrasts: Mapping[str, Mapping[str, Any]],
    detector_auprc: Any,
    min_oracle_gap: float,
    seed_count: int = 5,
    auprc_threshold: float = 0.80,
    oracle_gap_threshold: float = 0.05,
    invalid_oracle_gap_fraction: float = 0.0,
    invalid_oracle_gap_limit: float = 0.01,
    primary_movement_valid: bool = True,
    family: Sequence[str] = FROZEN_CONTRASTS,
) -> Mapping[str, Any]:
    """Apply G2 first; only a powered cell receives a G3 interpretation."""

    if not bool(g2.get("pass", False)):
        return {
            "status": "UNDERPOWERED_NEGATIVE",
            "g2_pass": False,
            "g3_interpreted": False,
            "win": False,
            "verdict": "NO_G3_VERDICT",
            "reason": "G2 median bootstrap half-width exceeds the preregistered threshold",
        }
    if float(invalid_oracle_gap_fraction) > float(invalid_oracle_gap_limit):
        return {
            "status": "INVALID_ORACLE_GAP_BOOTSTRAP",
            "g2_pass": True,
            "g3_interpreted": False,
            "win": False,
            "verdict": "NO_G3_VERDICT",
            "invalid_oracle_gap_fraction": float(invalid_oracle_gap_fraction),
            "invalid_oracle_gap_limit": float(invalid_oracle_gap_limit),
            "reason": "non-positive oracle-gap draws exceed the protocol limit",
        }
    if not bool(primary_movement_valid):
        return {
            "status": "INVALID_IDENTITY_PRIMARY_ARM",
            "g2_pass": True,
            "g3_interpreted": False,
            "win": False,
            "verdict": "NO_G3_VERDICT",
            "reason": "judged primary arm was identity: moved no documents in any seed",
        }
    if detector_auprc is None or not np.isfinite(float(detector_auprc)):
        detection_pass = False
    else:
        detection_pass = float(detector_auprc) >= float(auprc_threshold)
    contrast_checks = {
        name: _contrast_passes_g3(contrasts[name], seed_count=int(seed_count))
        for name in family
    }
    oracle_gap_pass = float(min_oracle_gap) >= float(oracle_gap_threshold)
    correction_pass = oracle_gap_pass and all(row["pass"] for row in contrast_checks.values())
    win = detection_pass and correction_pass
    if win:
        verdict = "SURPRISING_CHELATION_WIN"
    elif detection_pass and not correction_pass:
        verdict = "KILL_CORRECTOR_PARK_AS_REEMBED_ROUTER"
    else:
        verdict = "KILL_CORRECTOR"
    return {
        "status": "POWERED",
        "g2_pass": True,
        "g3_interpreted": True,
        "win": bool(win),
        "verdict": verdict,
        "detection": {
            "auprc": detector_auprc,
            "threshold": float(auprc_threshold),
            "pass": bool(detection_pass),
        },
        "correction": {
            "pass": bool(correction_pass),
            "min_oracle_gap": float(min_oracle_gap),
            "oracle_gap_threshold": float(oracle_gap_threshold),
            "oracle_gap_pass": bool(oracle_gap_pass),
            "contrasts": contrast_checks,
        },
    }
