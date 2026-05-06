"""Plan safe dry-run cleanup candidates for generated evidence artifacts."""

from __future__ import annotations

import argparse
import json
import stat
from pathlib import Path
from typing import Any, Mapping


DEFAULT_ROOT = Path("experiment_runs")
DEFAULT_KEEP_LATEST = 1
DEFAULT_EVIDENCE_INDEX = DEFAULT_ROOT / "evidence-index" / "latest" / "evidence_index.json"
DEFAULT_FRESHNESS_AUDIT = DEFAULT_ROOT / "evidence-index" / "latest" / "freshness_audit.json"

ARTIFACT_PATTERNS = {
    "validation_summaries": "validation_summary.json",
    "promotion_linkage_audits": "*promotion-linkage-audit*.json",
    "attnres_repeat_seed_decisions": "*attnres-repeat-seed-decision*.json",
    "default_promotion_preflights": "*preflight*.json",
    "evidence_chain_summaries": "evidence_chain_summary.json",
    "evidence_indexes": "evidence_index.json",
    "freshness_audits": "freshness_audit.json",
    "campaign_reports": "campaign_report.json",
    "adaptive_overlay_artifact_cards": "adaptive_overlay_artifact_card.json",
}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _relative(path: Path, root: Path) -> str:
    relative = path.relative_to(root.parent) if path.is_relative_to(root.parent) else path
    return relative.as_posix()


def _display_path(path: str | Path | None) -> str | None:
    return Path(path).as_posix() if path is not None else None


def _source_status(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "present": False}
    source_path = Path(path)
    return {"path": source_path.as_posix(), "present": source_path.exists()}


def _record(path: Path, root: Path, artifact_type: str, disposition: str, stat_result: Any) -> dict[str, Any]:
    return {
        "artifact_type": artifact_type,
        "path": _relative(path, root),
        "modified_at": stat_result.st_mtime,
        "size_bytes": stat_result.st_size,
        "disposition": disposition,
    }


def plan_evidence_artifact_cleanup(
    *,
    root: str | Path = DEFAULT_ROOT,
    keep_latest: int = DEFAULT_KEEP_LATEST,
    evidence_index: str | Path | None = DEFAULT_EVIDENCE_INDEX,
    freshness_audit: str | Path | None = DEFAULT_FRESHNESS_AUDIT,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Return a dry-run deletion plan without deleting any files."""

    root_path = Path(root)
    keep_count = max(0, int(keep_latest))
    retained: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    seen: set[Path] = set()

    for artifact_type, pattern in ARTIFACT_PATTERNS.items():
        paths_with_stat = []
        if root_path.exists():
            for path in root_path.rglob(pattern):
                if path in seen:
                    continue
                try:
                    stat_result = path.stat()
                except OSError:
                    continue
                if not stat.S_ISREG(stat_result.st_mode):
                    continue
                paths_with_stat.append((path, stat_result))
        paths_with_stat.sort(key=lambda item: item[1].st_mtime, reverse=True)
        seen.update(path for path, _stat_result in paths_with_stat)
        for index, (path, stat_result) in enumerate(paths_with_stat):
            disposition = "retain_latest" if index < keep_count else "candidate"
            record = _record(path, root_path, artifact_type, disposition, stat_result)
            if disposition == "retain_latest":
                retained.append(record)
            else:
                candidates.append(record)

    source_status = {
        "evidence_index": _source_status(evidence_index),
        "freshness_audit": _source_status(freshness_audit),
    }
    missing_sources = [
        name for name, status_record in source_status.items() if not bool(status_record.get("present", False))
    ]

    plan = {
        "record_type": "evidence_artifact_cleanup_plan",
        "dry_run": True,
        "root": str(root_path),
        "keep_latest": keep_count,
        "source_artifacts": {
            "evidence_index": _display_path(evidence_index),
            "freshness_audit": _display_path(freshness_audit),
        },
        "source_status": source_status,
        "summary": {
            "cleanup_review_allowed": len(missing_sources) == 0,
            "missing_source_artifacts": missing_sources,
            "candidate_count": len(candidates),
            "retained_count": len(retained),
            "candidate_bytes": sum(int(record["size_bytes"]) for record in candidates),
        },
        "candidates": candidates,
        "retained": retained,
    }
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plan["output"] = str(output_path)
        output_path.write_text(json.dumps(_json_safe(plan), indent=2), encoding="utf-8")
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan generated evidence artifact cleanup without deleting files")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Experiment root to scan")
    parser.add_argument("--keep-latest", type=int, default=DEFAULT_KEEP_LATEST, help="Artifacts to retain per type")
    parser.add_argument("--evidence-index", default=str(DEFAULT_EVIDENCE_INDEX), help="Evidence index path linked by the plan")
    parser.add_argument("--freshness-audit", default=str(DEFAULT_FRESHNESS_AUDIT), help="Freshness audit path linked by the plan")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()
    plan = plan_evidence_artifact_cleanup(
        root=args.root,
        keep_latest=args.keep_latest,
        evidence_index=args.evidence_index,
        freshness_audit=args.freshness_audit,
        output=args.output,
    )
    print(json.dumps(_json_safe(plan), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
