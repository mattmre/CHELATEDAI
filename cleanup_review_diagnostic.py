"""Emit compact diagnostics for evidence cleanup review plans."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from json import JSONDecodeError
from typing import Any, Mapping


DEFAULT_PLAN = Path("experiment_runs/evidence-cleanup/ci/cleanup_plan.json")


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def render_cleanup_review_diagnostic(plan_path: str | Path = DEFAULT_PLAN, *, mode: str = "blocked") -> str:
    """Return a markdown cleanup-review diagnostic for logs and GitHub summaries."""

    path = Path(plan_path)
    title = "Cleanup Review Warning Mode" if mode == "warn" else "Cleanup Review Blocked"
    lines = [f"## {title}", ""]
    if not path.exists():
        lines.append(f"- Cleanup plan missing: `{path.as_posix()}`")
        return "\n".join(lines)

    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, JSONDecodeError) as exc:
        lines.append(f"- Cleanup plan unreadable: `{path.as_posix()}`")
        lines.append(f"- Error: {exc}")
        return "\n".join(lines)
    summary = _as_mapping(plan.get("summary", {}))
    source_status = _as_mapping(plan.get("source_status", {}))
    missing = [str(item) for item in _as_list(summary.get("missing_source_artifacts", []))]
    allowed = bool(summary.get("cleanup_review_allowed", False))

    if mode == "warn":
        lines.append(f"- Cleanup review: {'allowed' if allowed else 'blocked'}")
    lines.extend(
        [
            f"- Plan: `{path.as_posix()}`",
            f"- Missing source artifacts: {', '.join(missing) if missing else '-'}",
            f"- Candidate count: {summary.get('candidate_count', '-')}",
            f"- Retained count: {summary.get('retained_count', '-')}",
            f"- Candidate bytes: {summary.get('candidate_bytes', '-')}",
        ]
    )
    for name, status in sorted(source_status.items()):
        status_record = _as_mapping(status)
        present = bool(status_record.get("present", False))
        source_path = status_record.get("path", "-")
        lines.append(f"- Source {name}: {'present' if present else 'missing'} `{source_path}`")
    if mode == "warn" and not allowed:
        lines.append("- Warning: blocked cleanup candidates must not be used for deletion decisions.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit cleanup-review diagnostics")
    parser.add_argument("--plan", default=str(DEFAULT_PLAN), help="Cleanup plan JSON path")
    parser.add_argument("--mode", choices=["blocked", "warn"], default="blocked", help="Diagnostic mode")
    parser.add_argument("--github-summary", action="store_true", help="Append output to GITHUB_STEP_SUMMARY")
    args = parser.parse_args()

    text = render_cleanup_review_diagnostic(args.plan, mode=args.mode)
    print(text)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if args.github_summary and summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
