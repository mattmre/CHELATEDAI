"""Run the default-promotion evidence chain and preserve fail-closed status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from audit_promotion_linkage import audit_promotion_linkage
from attnres_repeat_seed_decision import summarize_attnres_repeat_seed_decision
from default_promotion_preflight import evaluate_default_promotion_preflight
from run_overlay_model_scope_validation import run_validation_bundle


DEFAULT_OUTPUT_DIR = Path("experiment_runs") / "default-promotion-evidence-chain" / "latest"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def run_default_promotion_evidence_chain(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    experiment_root: str | Path = "experiment_runs",
    timeout_seconds: int = 300,
    attnres_artifacts: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Run validation, audit, repeat decision, and preflight into one linked summary."""

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    validation_dir = resolved_output_dir / "overlay-model-scope-validation"
    validation_summary = run_validation_bundle(output_dir=validation_dir, timeout_seconds=timeout_seconds)
    validation_path = Path(validation_summary["summary_path"])

    audit_path = resolved_output_dir / "promotion-linkage-audit.json"
    audit_summary = audit_promotion_linkage(experiment_root, output=audit_path)

    decision_path = resolved_output_dir / "attnres-repeat-seed-decision.json"
    decision_summary = summarize_attnres_repeat_seed_decision(attnres_artifacts) if attnres_artifacts else summarize_attnres_repeat_seed_decision()
    decision_path.write_text(json.dumps(_json_safe(decision_summary), indent=2), encoding="utf-8")

    preflight_path = resolved_output_dir / "default-promotion-preflight.json"
    preflight_summary = evaluate_default_promotion_preflight(
        validation_summary=validation_path,
        linkage_audit=audit_path,
        attnres_decision=decision_path,
        output=preflight_path,
    )

    command_failures = []
    if not validation_summary.get("passed", False):
        command_failures.append("overlay_model_scope_validation")
    if not audit_summary.get("passed", False):
        command_failures.append("promotion_linkage_audit")

    summary = {
        "record_type": "default_promotion_evidence_chain",
        "output_dir": str(resolved_output_dir),
        "chain_passed": len(command_failures) == 0,
        "review_allowed": bool(preflight_summary.get("review_allowed", False)),
        "default_change_allowed": bool(preflight_summary.get("default_change_allowed", False)),
        "command_failures": command_failures,
        "preflight_blockers": preflight_summary.get("blockers", []),
        "artifacts": {
            "validation_summary": str(validation_path),
            "promotion_linkage_audit": str(audit_path),
            "attnres_repeat_seed_decision": str(decision_path),
            "default_promotion_preflight": str(preflight_path),
        },
    }
    summary_path = resolved_output_dir / "evidence_chain_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run default-promotion evidence chain")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for linked evidence outputs")
    parser.add_argument("--experiment-root", default="experiment_runs", help="Experiment root to audit")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="Timeout per validation command")
    parser.add_argument(
        "--attnres-artifact",
        action="append",
        dest="attnres_artifacts",
        help="Optional AttnRes repeat-seed artifact path; may be provided more than once",
    )
    parser.add_argument(
        "--fail-on-blocked-review",
        action="store_true",
        help="Return nonzero when the evidence chain passes but preflight blocks review",
    )
    args = parser.parse_args()
    summary = run_default_promotion_evidence_chain(
        output_dir=args.output_dir,
        experiment_root=args.experiment_root,
        timeout_seconds=args.timeout_seconds,
        attnres_artifacts=args.attnres_artifacts,
    )
    print(json.dumps(_json_safe(summary), indent=2))
    if not summary["chain_passed"]:
        return 1
    if args.fail_on_blocked_review and not summary["review_allowed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
