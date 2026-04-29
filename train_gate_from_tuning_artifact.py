"""Train conservative learned-gate rules from tuning artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


NUMERIC_GATES = [
    ("jaccard_mean", 0.9, ">="),
    ("jaccard_mean", 0.9, "<"),
    ("mask_density_mean", 0.98, ">="),
    ("mask_density_mean", 0.98, "<"),
    ("variance_mean", 0.002, ">="),
    ("variance_mean", 0.002, "<"),
]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and np.isfinite(float(value))


def load_gate_feature_rows(path: str | Path) -> List[Dict[str, Any]]:
    """Load gate feature rows from a tuning artifact."""

    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = artifact.get("gate_feature_rows")
    if not isinstance(rows, list):
        raise ValueError("artifact does not contain gate_feature_rows")
    return [dict(row) for row in rows]


def _row_matches_gate(row: Dict[str, Any], gate: Dict[str, Any]) -> bool:
    if row.get("profile") != gate["profile"]:
        return False
    task = gate.get("task")
    if task is not None and row.get("task") != task:
        return False
    feature = gate.get("feature")
    if feature is None:
        return True
    value = row.get(feature)
    if not _is_number(value):
        return False
    threshold = float(gate["threshold"])
    if gate["operator"] == ">=":
        return float(value) >= threshold
    if gate["operator"] == "<":
        return float(value) < threshold
    raise ValueError(f"unsupported operator {gate['operator']}")


def _summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    deltas = [float(row["delta_vs_baseline"]) for row in rows]
    fault_counts: Dict[str, int] = {}
    tasks = sorted({str(row.get("task")) for row in rows})
    for row in rows:
        fault = str(row.get("fault_class", "unknown"))
        fault_counts[fault] = fault_counts.get(fault, 0) + 1
    return {
        "windows": len(rows),
        "tasks": tasks,
        "mean_delta_vs_baseline": float(np.mean(deltas)) if deltas else 0.0,
        "best_delta_vs_baseline": float(np.max(deltas)) if deltas else 0.0,
        "worst_delta_vs_baseline": float(np.min(deltas)) if deltas else 0.0,
        "improved": sum(delta > 0.001 for delta in deltas),
        "tied": sum(abs(delta) <= 0.001 for delta in deltas),
        "regressed": sum(delta < -0.001 for delta in deltas),
        "fault_counts": fault_counts,
    }


def _passes_gate_summary(
    summary: Dict[str, Any],
    min_windows: int,
    min_mean_delta: float,
    max_regressions: int,
) -> bool:
    faults = summary.get("fault_counts", {})
    return (
        int(summary["windows"]) >= min_windows
        and float(summary["mean_delta_vs_baseline"]) >= min_mean_delta
        and int(summary["regressed"]) <= max_regressions
        and int(faults.get("actuator_active_negative", 0)) <= max_regressions
        and int(faults.get("metric_changed_without_actuator", 0)) == 0
    )


def _candidate_gates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    profiles = sorted({
        str(row["profile"])
        for row in rows
        if row.get("profile") not in {"baseline", "guard_p85_t0.01"}
    })
    gates: List[Dict[str, Any]] = []
    for profile in profiles:
        gates.append({"profile": profile, "task": None, "feature": None})
        tasks = sorted({str(row["task"]) for row in rows if row.get("profile") == profile})
        for task in tasks:
            gates.append({"profile": profile, "task": task, "feature": None})
        for feature, threshold, operator in NUMERIC_GATES:
            gates.append({
                "profile": profile,
                "task": None,
                "feature": feature,
                "operator": operator,
                "threshold": threshold,
            })
            for task in tasks:
                gates.append({
                    "profile": profile,
                    "task": task,
                    "feature": feature,
                    "operator": operator,
                    "threshold": threshold,
                })
    return gates


def train_gate_rules(
    rows: Iterable[Dict[str, Any]],
    *,
    holdout_modulo: int = 5,
    holdout_remainder: int = 0,
    min_train_windows: int = 3,
    min_holdout_windows: int = 2,
    min_mean_delta: float = 0.001,
    max_regressions: int = 0,
) -> Dict[str, Any]:
    """Train conservative rules that pass both train and holdout summaries."""

    all_rows = [dict(row) for row in rows]
    if holdout_modulo <= 1:
        raise ValueError("holdout_modulo must be > 1")
    accepted = []
    rejected = []
    for gate in _candidate_gates(all_rows):
        matched = [row for row in all_rows if _row_matches_gate(row, gate)]
        train_rows = [
            row for row in matched
            if int(row.get("global_window", 0)) % holdout_modulo != holdout_remainder
        ]
        holdout_rows = [
            row for row in matched
            if int(row.get("global_window", 0)) % holdout_modulo == holdout_remainder
        ]
        train_summary = _summarize_rows(train_rows)
        holdout_summary = _summarize_rows(holdout_rows)
        passed = (
            _passes_gate_summary(train_summary, min_train_windows, min_mean_delta, max_regressions)
            and _passes_gate_summary(holdout_summary, min_holdout_windows, min_mean_delta, max_regressions)
        )
        record = {
            "gate": gate,
            "train": train_summary,
            "holdout": holdout_summary,
            "passed": passed,
        }
        if passed:
            accepted.append(record)
        else:
            rejected.append(record)

    accepted.sort(
        key=lambda item: (
            -item["holdout"]["mean_delta_vs_baseline"],
            -item["holdout"]["windows"],
            item["gate"]["profile"],
        )
    )
    return {
        "version": 1,
        "policy": "default_fast_allow_list",
        "holdout": {
            "modulo": holdout_modulo,
            "remainder": holdout_remainder,
        },
        "criteria": {
            "min_train_windows": min_train_windows,
            "min_holdout_windows": min_holdout_windows,
            "min_mean_delta": min_mean_delta,
            "max_regressions": max_regressions,
        },
        "rules": [item["gate"] for item in accepted],
        "accepted": accepted,
        "rejected_count": len(rejected),
    }


def write_gate_config(config: Dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train conservative gate rules from a tuning artifact")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--holdout-modulo", type=int, default=5)
    parser.add_argument("--holdout-remainder", type=int, default=0)
    parser.add_argument("--min-train-windows", type=int, default=3)
    parser.add_argument("--min-holdout-windows", type=int, default=2)
    parser.add_argument("--min-mean-delta", type=float, default=0.001)
    parser.add_argument("--max-regressions", type=int, default=0)
    args = parser.parse_args()

    rows = load_gate_feature_rows(args.input)
    config = train_gate_rules(
        rows,
        holdout_modulo=args.holdout_modulo,
        holdout_remainder=args.holdout_remainder,
        min_train_windows=args.min_train_windows,
        min_holdout_windows=args.min_holdout_windows,
        min_mean_delta=args.min_mean_delta,
        max_regressions=args.max_regressions,
    )
    write_gate_config(config, args.output)
    print(json.dumps({
        "output": args.output,
        "rules": len(config["rules"]),
        "accepted": len(config["accepted"]),
        "rejected_count": config["rejected_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
