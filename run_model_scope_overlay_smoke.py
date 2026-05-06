"""Run a deterministic Model-Scope campaign with supplied overlay reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from adaptive_overlay import build_overlay_report
from model_scope_artifacts import build_model_scope_artifact, write_model_scope_artifact
from run_model_scope_campaign import run_model_scope_campaign


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _artifact(query_id: str, prompt_hash: str, *, feature_id: str, value: float) -> dict[str, Any]:
    return build_model_scope_artifact(
        runtime={"model_name": "smoke-model", "device": "cpu"},
        capture={
            "prompt_hash": prompt_hash,
            "token_count": 3,
            "captured_layer_count": 1,
            "layer_indices": [0],
            "metadata": {"query_id": query_id, "query_text": f"smoke query {query_id}"},
            "observations": [
                {
                    "layer_index": 0,
                    "feature_summary": {
                        "feature_space": "smoke_scope",
                        "active_features": [{"feature_id": feature_id, "value": value}],
                    },
                }
            ],
        },
    )


def _overlay_rows(seed: int) -> list[dict[str, Any]]:
    rows = []
    for query_id in ("q1", "q2", "q3"):
        rows.append(
            {
                "task": "SmokeTask",
                "seed": seed,
                "query_id": query_id,
                "profile": "baseline",
                "delta_ndcg_at_10": 0.0,
                "fault_class": "reference",
            }
        )
        rows.append(
            {
                "task": "SmokeTask",
                "seed": seed,
                "query_id": query_id,
                "profile": "guard_learned_reform_gate_v1",
                "delta_ndcg_at_10": 0.02,
                "fault_class": "actuator_active_positive",
                "promotion_blocker": False,
                "budget_units": 2,
            }
        )
    return rows


def run_smoke(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    input_dir = root / "input_artifacts"
    campaign_dir = root / "campaign"
    reports_dir = root / "overlay_reports"
    input_dir.mkdir(parents=True, exist_ok=True)
    campaign_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    write_model_scope_artifact(input_dir / "q1_ref.json", _artifact("q1", "hash-q1", feature_id="101", value=1.2))
    write_model_scope_artifact(input_dir / "q1_repeat.json", _artifact("q1", "hash-q1", feature_id="101", value=1.1))
    write_model_scope_artifact(input_dir / "q2_ref.json", _artifact("q2", "hash-q2", feature_id="202", value=1.2))
    write_model_scope_artifact(input_dir / "q2_repeat.json", _artifact("q2", "hash-q2", feature_id="202", value=1.1))

    overlay_path = reports_dir / "adaptive_overlay_report.json"
    holdout_path = reports_dir / "adaptive_overlay_holdout_report.json"
    overlay_path.write_text(json.dumps(build_overlay_report(_overlay_rows(seed=1)), indent=2), encoding="utf-8")
    holdout_path.write_text(json.dumps(build_overlay_report(_overlay_rows(seed=2)), indent=2), encoding="utf-8")

    report = run_model_scope_campaign(
        input_dir,
        output_dir=campaign_dir,
        max_rules=4,
        min_alignment_score=0.5,
        holdout_report={"passed": True, "score": 0.8},
        safety_report={"passed": True},
        adaptive_overlay_report=overlay_path,
        adaptive_overlay_holdout_report=holdout_path,
        require_adaptive_overlay_readiness=True,
    )
    smoke_summary = {
        "record_type": "model_scope_overlay_smoke_summary",
        "output_dir": str(root),
        "campaign_report": report["outputs"]["campaign_report"],
        "required_outputs": {
            key: report["outputs"][key]
            for key in (
                "adaptive_overlay_report",
                "adaptive_overlay_holdout_report",
                "adaptive_overlay_validation_report",
                "adaptive_overlay_collection_policy",
                "adaptive_overlay_artifact_card",
                "verifier_evidence_cards",
                "promotion_decision",
            )
        },
        "promotion_ready": bool(report["promotion_decision"]["promotion_ready"]),
        "adaptive_overlay_ready": bool(report["promotion_decision"]["adaptive_overlay_ready"]),
        "validation_ready": bool(report["adaptive_overlay_validation_report"]["validation_ready"]),
    }
    summary_path = root / "smoke_summary.json"
    summary_path.write_text(json.dumps(_json_safe(smoke_summary), indent=2), encoding="utf-8")
    smoke_summary["smoke_summary"] = str(summary_path)
    return smoke_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic Model-Scope overlay smoke campaign")
    parser.add_argument(
        "--output-dir",
        default="experiment_runs/model-scope-overlay-smoke/latest",
        help="Directory for generated smoke inputs, overlay reports, and campaign outputs",
    )
    args = parser.parse_args()
    print(json.dumps(_json_safe(run_smoke(args.output_dir)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
