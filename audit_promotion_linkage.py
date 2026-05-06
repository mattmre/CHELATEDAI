"""Audit campaign reports for promotion artifact-card and rollback linkage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_ROOT = Path("experiment_runs")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _report_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        {
            *root.rglob("campaign_report.json"),
            *root.rglob("*campaign*report*.json"),
        }
    )


def _audit_report(path: Path, root: Path) -> dict[str, Any]:
    payload = _load_json(path)
    promotion = payload.get("promotion_decision")
    if not isinstance(promotion, Mapping):
        promotion = {}
    has_overlay_card = isinstance(payload.get("adaptive_overlay_artifact_card"), Mapping) or "adaptive_overlay_artifact_card" in (
        payload.get("outputs") or {}
    )
    has_overlay_report = isinstance(payload.get("adaptive_overlay"), Mapping) or "adaptive_overlay_report" in (
        payload.get("outputs") or {}
    )
    artifact_ref = promotion.get("artifact_card_reference")
    if not isinstance(artifact_ref, Mapping):
        artifact_ref = {}
    rollback_path = promotion.get("rollback_path")
    promotion_ready = bool(promotion.get("promotion_ready", False))
    requires_linkage = promotion_ready or has_overlay_card or has_overlay_report
    blockers = []
    if requires_linkage and not (artifact_ref.get("path") or artifact_ref.get("card_id")):
        blockers.append("missing_artifact_card_reference")
    if requires_linkage and not rollback_path:
        blockers.append("missing_rollback_path")

    return {
        "path": str(path.relative_to(root.parent)) if path.is_relative_to(root.parent) else str(path),
        "promotion_ready": promotion_ready,
        "has_overlay_report": bool(has_overlay_report),
        "has_overlay_artifact_card": bool(has_overlay_card),
        "requires_linkage": bool(requires_linkage),
        "artifact_card_reference": _json_safe(artifact_ref),
        "rollback_path": rollback_path,
        "blockers": blockers,
        "passed": not blockers,
    }


def audit_promotion_linkage(root: str | Path = DEFAULT_ROOT, *, output: str | Path | None = None) -> dict[str, Any]:
    """Audit campaign reports and optionally write a JSON report."""
    root_path = Path(root)
    reports = []
    skipped = 0
    for path in _report_paths(root_path):
        try:
            reports.append(_audit_report(path, root_path))
        except (OSError, json.JSONDecodeError):
            skipped += 1
    blocked = [report for report in reports if not report["passed"]]
    summary = {
        "record_type": "promotion_linkage_audit",
        "root": str(root_path),
        "passed": len(blocked) == 0,
        "report_count": len(reports),
        "skipped_count": skipped,
        "blocked_count": len(blocked),
        "blocked_reports": [report["path"] for report in blocked],
        "reports": reports,
    }
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
        summary["output"] = str(output_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit campaign promotion linkage")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Experiment root to scan")
    parser.add_argument("--output", default=None, help="Optional JSON audit output path")
    args = parser.parse_args()
    summary = audit_promotion_linkage(args.root, output=args.output)
    print(json.dumps(_json_safe(summary), indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
