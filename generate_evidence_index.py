"""Generate a compact cross-artifact evidence index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_ROOT = Path("experiment_runs")
DEFAULT_OUTPUT = Path("evidence_index.json")

# Patterns scanned under experiment_runs/
ARTIFACT_PATTERNS = {
    "validation_summaries": "validation_summary.json",
    "promotion_linkage_audits": "*promotion-linkage-audit*.json",
    "attnres_repeat_seed_decisions": "*attnres-repeat-seed-decision*.json",
    "default_promotion_preflights": "*preflight*.json",
    "evidence_chain_summaries": "evidence_chain_summary.json",
    "campaign_reports": "campaign_report.json",
    "adaptive_overlay_artifact_cards": "adaptive_overlay_artifact_card.json",
}

# Tracked artifacts at repo root, always included regardless of experiment_runs/ state
TRACKED_ARTIFACT_PATHS: dict[str, str] = {
    "beir_results": "benchmark_beir_results.json",
}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root.parent)) if path.is_relative_to(root.parent) else str(path)


def _record(path: Path, root: Path) -> dict[str, Any]:
    payload = _load_json(path) or {}
    stat = path.stat()
    promotion = payload.get("promotion_decision")
    if not isinstance(promotion, dict):
        promotion = {}
    return {
        "path": _relative(path, root),
        "record_type": payload.get("record_type"),
        "modified_at": stat.st_mtime,
        "passed": payload.get("passed", payload.get("chain_passed")),
        "review_allowed": payload.get("review_allowed"),
        "promotion_ready": promotion.get("promotion_ready", payload.get("promotion_ready")),
        "decision": payload.get("decision") or promotion.get("decision"),
        "blockers": payload.get("blockers", []),
    }


def generate_evidence_index(
    *,
    root: str | Path = DEFAULT_ROOT,
    output: str | Path = DEFAULT_OUTPUT,
    limit_per_type: int = 25,
    tracked_root: str | Path | None = None,
) -> dict[str, Any]:
    """Generate and write a compact index of evidence artifacts under root.

    Scans both `root` (typically experiment_runs/) for pattern-matched artifacts
    and `tracked_root` (defaults to cwd) for tracked repository artifacts listed in
    TRACKED_ARTIFACT_PATHS (e.g. benchmark_beir_results.json).
    """

    root_path = Path(root)
    artifacts: dict[str, list[dict[str, Any]]] = {}
    for name, pattern in ARTIFACT_PATTERNS.items():
        paths = []
        if root_path.exists():
            paths = sorted(root_path.rglob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
        artifacts[name] = [_record(path, root_path) for path in paths[: max(0, limit_per_type)]]

    # Include tracked repo-root artifacts regardless of experiment_runs/ state
    resolved_tracked_root = Path(tracked_root) if tracked_root is not None else Path(".")
    for name, filename in TRACKED_ARTIFACT_PATHS.items():
        tracked_path = resolved_tracked_root / filename
        if tracked_path.exists():
            artifacts[name] = [_record(tracked_path, resolved_tracked_root)]
        else:
            artifacts[name] = []

    latest_preflight = artifacts["default_promotion_preflights"][0] if artifacts["default_promotion_preflights"] else {}
    latest_chain = artifacts["evidence_chain_summaries"][0] if artifacts["evidence_chain_summaries"] else {}
    summary = {
        "artifact_counts": {name: len(records) for name, records in artifacts.items()},
        "latest_review_allowed": latest_preflight.get("review_allowed") if latest_preflight else None,
        "latest_preflight_blockers": latest_preflight.get("blockers", []) if latest_preflight else [],
        "latest_chain_passed": latest_chain.get("passed") if latest_chain else None,
    }
    index = {
        "record_type": "cross_artifact_evidence_index",
        "root": str(root_path),
        "limit_per_type": int(limit_per_type),
        "summary": summary,
        "artifacts": artifacts,
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index["output"] = str(output_path)
    output_path.write_text(json.dumps(_json_safe(index), indent=2), encoding="utf-8")
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate cross-artifact evidence index")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Experiment root to scan")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON path")
    parser.add_argument("--limit-per-type", type=int, default=25, help="Maximum records to keep per artifact type")
    args = parser.parse_args()
    index = generate_evidence_index(root=args.root, output=args.output, limit_per_type=args.limit_per_type)
    print(json.dumps(_json_safe(index), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
