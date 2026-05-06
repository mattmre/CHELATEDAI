"""Audit whether the cross-artifact evidence index is present, linked, and fresh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_INDEX = Path("experiment_runs/evidence-index/latest/evidence_index.json")


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


def _resolve_indexed_path(path_text: str, index_path: Path, repo_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidates = [
        repo_root / path,
        index_path.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def audit_evidence_index_freshness(
    *,
    index: str | Path = DEFAULT_INDEX,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Return a fail-closed freshness/linkage audit for an evidence index."""

    index_path = Path(index)
    payload = _load_json(index_path)
    blockers: list[str] = []
    warnings: list[str] = []
    missing_paths: list[str] = []
    stale_paths: list[str] = []
    checked_paths = 0

    if payload is None:
        blockers.append("evidence_index_missing_or_unreadable")
        payload = {}

    artifacts = payload.get("artifacts", {})
    if payload and not isinstance(artifacts, dict):
        blockers.append("evidence_index_artifacts_not_object")
        artifacts = {}

    index_mtime = index_path.stat().st_mtime if index_path.exists() else 0.0
    repo_root = Path.cwd()
    for records in artifacts.values():
        if not isinstance(records, list):
            warnings.append("artifact_records_not_list")
            continue
        for record in records:
            if not isinstance(record, dict):
                warnings.append("artifact_record_not_object")
                continue
            path_text = record.get("path")
            if not path_text:
                warnings.append("artifact_record_missing_path")
                continue
            checked_paths += 1
            source_path = _resolve_indexed_path(str(path_text), index_path, repo_root)
            if not source_path.exists():
                missing_paths.append(str(path_text))
                continue
            if source_path.stat().st_mtime > index_mtime:
                stale_paths.append(str(path_text))

    if missing_paths:
        blockers.append("indexed_artifact_paths_missing")
    if stale_paths:
        blockers.append("evidence_index_stale")

    summary = {
        "record_type": "evidence_index_freshness_audit",
        "index": str(index_path),
        "passed": len(blockers) == 0,
        "blockers": blockers,
        "warnings": warnings,
        "checked_path_count": checked_paths,
        "missing_paths": missing_paths,
        "stale_paths": stale_paths,
    }
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary["output"] = str(output_path)
        output_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit evidence-index freshness and linked paths")
    parser.add_argument("--index", default=str(DEFAULT_INDEX), help="Evidence index JSON path")
    parser.add_argument("--output", default=None, help="Optional JSON audit output path")
    args = parser.parse_args()
    summary = audit_evidence_index_freshness(index=args.index, output=args.output)
    print(json.dumps(_json_safe(summary), indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
