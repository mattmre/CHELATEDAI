"""Gate persistence — train reform and mask gates from an attribution pool.

Loads an attribution pool produced by build_attribution_pool, trains two
fail-closed logistic-regression gates, and writes them as versioned JSON
artifacts alongside a summary file.

Usage::

    python run_gate_train.py [--pool attribution_pool.json] [--out-dir .] [--min-samples 10]

Outputs
-------
<out-dir>/reform_gate_v1.json
    Trained reform gate parameters (or fail-closed stub when insufficient data).
<out-dir>/mask_gate_v1.json
    Trained mask gate parameters (or fail-closed stub when insufficient data).
<out-dir>/gate_train_summary.json
    Training summary with recommendation field.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from chelation_logger import get_logger
from learned_reformulation_gate import load_attribution_pool, train_gate_from_pool
from learned_mask_gate import enrich_mask_rows, train_mask_gate

_SCHEMA_VERSION = "1.0"


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _count_pool_samples(pool: Dict[str, Any]) -> int:
    """Return combined row count used for gate training."""
    query_rows = pool.get("query_attribution_rows") or []
    mask_rows = pool.get("mask_probe_rows") or []
    return len(query_rows) + len(mask_rows)


def _train_reform_gate(
    pool: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    """Train the reform gate; return (artifact_dict, outcome)."""
    query_rows = pool.get("query_attribution_rows") or []
    n_samples = len(query_rows)
    try:
        config = train_gate_from_pool(pool)
        outcome = "trained"
    except (ValueError, KeyError) as exc:
        config = {"gate": None, "error": str(exc)}
        outcome = "failed"
    artifact = {
        "trained_at": _iso_now(),
        "gate_type": "reform_gate",
        "n_training_samples": n_samples,
        "model_params": config,
        "schema_version": _SCHEMA_VERSION,
    }
    return artifact, outcome


def _train_mask_gate(
    pool: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    """Train the mask gate; return (artifact_dict, outcome)."""
    raw_rows = list(pool.get("mask_probe_rows") or [])
    n_samples = len(raw_rows)
    try:
        enriched = enrich_mask_rows(raw_rows)
        config = train_mask_gate(enriched)
        outcome = "trained"
    except (ValueError, KeyError) as exc:
        config = {"gate": None, "error": str(exc)}
        outcome = "failed"
    artifact = {
        "trained_at": _iso_now(),
        "gate_type": "mask_gate",
        "n_training_samples": n_samples,
        "model_params": config,
        "schema_version": _SCHEMA_VERSION,
    }
    return artifact, outcome


def main(argv: Optional[list] = None) -> int:  # noqa: UP006 — kept for 3.9 compat
    parser = argparse.ArgumentParser(
        description="Train reform and mask gates from an attribution pool."
    )
    parser.add_argument(
        "--pool",
        default="attribution_pool.json",
        help="Path to attribution pool JSON (default: attribution_pool.json)",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Output directory for gate artifacts (default: current directory)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum combined pool rows required to train (default: 10)",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Override path for gate_train_summary.json",
    )
    args = parser.parse_args(argv)

    logger = get_logger()
    pool_path = Path(args.pool)
    out_dir = Path(args.out_dir)
    summary_path = (
        Path(args.summary_out) if args.summary_out else out_dir / "gate_train_summary.json"
    )

    if not pool_path.is_file():
        print(
            f"ERROR: attribution pool file not found: {pool_path}",
            file=sys.stderr,
        )
        return 1

    try:
        pool = load_attribution_pool(pool_path)
    except Exception as exc:
        print(f"ERROR: failed to load attribution pool: {exc}", file=sys.stderr)
        return 1

    n_samples = _count_pool_samples(pool)

    if n_samples < args.min_samples:
        logger.log_event(
            "gate_train_skipped",
            f"Pool has only {n_samples} samples (< {args.min_samples}); skipping gate training.",
            level="WARNING",
            n_samples=n_samples,
            min_samples=args.min_samples,
        )
        print(
            f"WARNING: pool has {n_samples} samples, below minimum {args.min_samples}."
            " Skipping training.",
            file=sys.stderr,
        )
        summary: Dict[str, Any] = {
            "trained_at": _iso_now(),
            "schema_version": _SCHEMA_VERSION,
            "n_total_samples": n_samples,
            "reform_gate": {"outcome": "skipped", "reason": "insufficient_data"},
            "mask_gate": {"outcome": "skipped", "reason": "insufficient_data"},
            "recommendation": "insufficient_data",
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    reform_artifact, reform_outcome = _train_reform_gate(pool)
    mask_artifact, mask_outcome = _train_mask_gate(pool)

    reform_path = out_dir / "reform_gate_v1.json"
    mask_path = out_dir / "mask_gate_v1.json"

    reform_path.write_text(json.dumps(reform_artifact, indent=2), encoding="utf-8")
    mask_path.write_text(json.dumps(mask_artifact, indent=2), encoding="utf-8")

    summary = {
        "trained_at": _iso_now(),
        "schema_version": _SCHEMA_VERSION,
        "n_total_samples": n_samples,
        "reform_gate": {
            "outcome": reform_outcome,
            "path": str(reform_path),
            "n_training_samples": reform_artifact["n_training_samples"],
            "gate_present": reform_artifact["model_params"].get("gate") is not None,
        },
        "mask_gate": {
            "outcome": mask_outcome,
            "path": str(mask_path),
            "n_training_samples": mask_artifact["n_training_samples"],
            "gate_present": mask_artifact["model_params"].get("gate") is not None,
        },
        "recommendation": "gates_saved",
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "pool": str(pool_path),
                "n_total_samples": n_samples,
                "reform_gate_path": str(reform_path),
                "mask_gate_path": str(mask_path),
                "reform_gate_present": summary["reform_gate"]["gate_present"],
                "mask_gate_present": summary["mask_gate"]["gate_present"],
                "summary_path": str(summary_path),
            },
            indent=2,
        )
    )

    logger.log_event(
        "gate_train_complete",
        "Gate training complete.",
        n_total_samples=n_samples,
        reform_outcome=reform_outcome,
        mask_outcome=mask_outcome,
        reform_gate_present=summary["reform_gate"]["gate_present"],
        mask_gate_present=summary["mask_gate"]["gate_present"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
