"""Summarize repeat-seed AttnRes evidence into a promotion decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_ARTIFACTS = (
    "experiment_runs/roadcourse-small/attnres_trained_num_blocks_scifact_seed42.json",
    "experiment_runs/roadcourse-small/attnres_balanced_candidate_scifact_seed43.json",
    "experiment_runs/roadcourse-small/attnres_balanced_candidate_scifact_seed44.json",
    "experiment_runs/roadcourse-small/attnres_trained_num_blocks_nfcorpus_seed42.json",
    "experiment_runs/roadcourse-small/attnres_balanced_candidate_nfcorpus_seed43.json",
    "experiment_runs/roadcourse-small/attnres_balanced_candidate_nfcorpus_seed44.json",
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_attnres_repeat_seed_decision(paths: Iterable[str | Path] = DEFAULT_ARTIFACTS) -> dict[str, Any]:
    rows = []
    quantization_failures = 0
    positive_deltas = 0
    non_positive_deltas = 0
    for path in paths:
        payload = _load(path)
        recommendation = dict(payload.get("default_recommendation", {}))
        quantization_gate = dict(payload.get("quantization_survival", {}).get("gate", {}))
        delta = float(recommendation.get("delta_vs_baseline", 0.0) or 0.0)
        quantization_passed = bool(quantization_gate.get("passed", False))
        rows.append(
            {
                "path": str(path),
                "task": payload.get("task"),
                "seed": payload.get("seed"),
                "recommended_profile": recommendation.get("recommended_profile"),
                "baseline_ndcg_at_10": recommendation.get("baseline_ndcg_at_10"),
                "best_ndcg_at_10": recommendation.get("best_ndcg_at_10"),
                "delta_vs_baseline": delta,
                "local_default_change_allowed": bool(recommendation.get("default_change_allowed", False)),
                "quantization_gate_passed": quantization_passed,
                "quantization_reasons": list(quantization_gate.get("reasons") or []),
            }
        )
        if delta > 0.0:
            positive_deltas += 1
        else:
            non_positive_deltas += 1
        if not quantization_passed:
            quantization_failures += 1

    task_summary: dict[str, dict[str, Any]] = {}
    for row in rows:
        task = str(row.get("task") or "unknown")
        summary = task_summary.setdefault(task, {"seed_count": 0, "positive_delta_count": 0, "quantization_pass_count": 0})
        summary["seed_count"] += 1
        summary["positive_delta_count"] += int(float(row["delta_vs_baseline"]) > 0.0)
        summary["quantization_pass_count"] += int(bool(row["quantization_gate_passed"]))

    promote = positive_deltas == len(rows) and quantization_failures == 0
    reasons = []
    if non_positive_deltas:
        reasons.append("repeat_seed_non_positive_delta_present")
    if quantization_failures:
        reasons.append("quantization_survival_failures_present")
    if any(item["positive_delta_count"] < item["seed_count"] for item in task_summary.values()):
        reasons.append("task_seed_consistency_failed")

    return {
        "record_type": "attnres_repeat_seed_decision",
        "artifact_count": len(rows),
        "rows": rows,
        "task_summary": task_summary,
        "positive_delta_count": positive_deltas,
        "non_positive_delta_count": non_positive_deltas,
        "quantization_failure_count": quantization_failures,
        "promote_default": bool(promote),
        "decision": "no_default_change" if not promote else "promote_candidate",
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize repeat-seed AttnRes decision evidence")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    parser.add_argument("artifacts", nargs="*", default=list(DEFAULT_ARTIFACTS), help="Road-course artifacts to summarize")
    args = parser.parse_args()
    summary = summarize_attnres_repeat_seed_decision(args.artifacts)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    print(json.dumps(_json_safe(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
