"""Fail-closed preflight for default-promotion review readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_VALIDATION_SUMMARY = Path("experiment_runs/overlay-model-scope-validation/latest/validation_summary.json")
DEFAULT_LINKAGE_AUDIT = Path("experiment_runs/promotion-linkage-audit/latest/audit.json")
DEFAULT_ATTNRES_DECISION = Path("experiment_runs/attnres-repeat-seed-decision/latest/decision.json")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _load_optional_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "unreadable"
    if not isinstance(payload, dict):
        return None, "not_object"
    return payload, None


def _artifact_status(path: Path, payload: dict[str, Any] | None, load_error: str | None, passed: bool) -> dict[str, Any]:
    return {
        "path": str(path),
        "present": payload is not None,
        "load_error": load_error,
        "passed": bool(passed),
    }


def evaluate_default_promotion_preflight(
    *,
    validation_summary: str | Path = DEFAULT_VALIDATION_SUMMARY,
    linkage_audit: str | Path = DEFAULT_LINKAGE_AUDIT,
    attnres_decision: str | Path = DEFAULT_ATTNRES_DECISION,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Return a fail-closed readiness decision for starting default-promotion review."""

    validation_path = Path(validation_summary)
    audit_path = Path(linkage_audit)
    decision_path = Path(attnres_decision)

    validation, validation_error = _load_optional_json(validation_path)
    audit, audit_error = _load_optional_json(audit_path)
    decision, decision_error = _load_optional_json(decision_path)

    blockers = []

    validation_passed = bool(validation and validation.get("passed") is True)
    if validation is None:
        blockers.append(f"validation_summary_{validation_error}")
    elif not validation_passed:
        blockers.append("validation_bundle_failed")

    audit_passed = bool(audit and audit.get("passed") is True)
    if audit is None:
        blockers.append(f"promotion_linkage_audit_{audit_error}")
    elif not audit_passed:
        blockers.append("promotion_linkage_audit_failed")

    repeat_evidence_passed = bool(decision and decision.get("promote_default") is True)
    if decision is None:
        blockers.append(f"repeat_seed_decision_{decision_error}")
    elif not repeat_evidence_passed:
        blockers.append("repeat_seed_evidence_does_not_support_default_promotion")

    summary = {
        "record_type": "default_promotion_preflight",
        "review_allowed": len(blockers) == 0,
        "default_change_allowed": False,
        "blockers": blockers,
        "artifacts": {
            "validation_summary": _artifact_status(validation_path, validation, validation_error, validation_passed),
            "promotion_linkage_audit": _artifact_status(audit_path, audit, audit_error, audit_passed),
            "repeat_seed_decision": _artifact_status(decision_path, decision, decision_error, repeat_evidence_passed),
        },
    }
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
        summary["output"] = str(output_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight default-promotion review readiness")
    parser.add_argument("--validation-summary", default=str(DEFAULT_VALIDATION_SUMMARY))
    parser.add_argument("--linkage-audit", default=str(DEFAULT_LINKAGE_AUDIT))
    parser.add_argument("--attnres-decision", default=str(DEFAULT_ATTNRES_DECISION))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    summary = evaluate_default_promotion_preflight(
        validation_summary=args.validation_summary,
        linkage_audit=args.linkage_audit,
        attnres_decision=args.attnres_decision,
        output=args.output,
    )
    print(json.dumps(_json_safe(summary), indent=2))
    return 0 if summary["review_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
