"""
One-time migration: tag all experiment_runs JSON artifacts that originate
from smoke tests with  "data_source": "smoke_test".

An artifact is considered a smoke-test artifact if it already contains
either of these telltale strings produced by run_model_scope_overlay_smoke.py:
  - "smoke-model"
  - "smoke query"

Artifacts that already carry a "data_source" key are NOT modified (idempotent).

Run with:
    python label_smoke_artifacts.py [--dry-run] [--root experiment_runs]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_SMOKE_MARKERS = ("smoke-model", "smoke query")


def _raw_contains_smoke(path: Path) -> bool:
    """Return True if the file's raw bytes contain any smoke marker."""
    try:
        raw = path.read_bytes()
        return any(m.encode() in raw for m in _SMOKE_MARKERS)
    except OSError:
        return False


def _label_file(path: Path, dry_run: bool) -> str:
    """
    Attempt to add "data_source": "smoke_test" to a JSON object file.

    Returns one of: "skipped_non_object", "skipped_has_source", "skipped_no_marker",
                    "labeled" (dry-run or real), "error".
    """
    # Fast path: skip files with no smoke markers (raw bytes, no JSON parse)
    if not _raw_contains_smoke(path):
        return "skipped_no_marker"

    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except Exception:
        return "error"

    if not isinstance(data, dict):
        return "skipped_non_object"

    if "data_source" in data:
        return "skipped_has_source"

    if dry_run:
        return "labeled"

    data["data_source"] = "smoke_test"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return "labeled"


def main() -> None:
    parser = argparse.ArgumentParser(description="Label smoke-test experiment_runs artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--root", default="experiment_runs", help="Root directory to scan")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: root directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    counts: dict[str, int] = {
        "labeled": 0,
        "skipped_has_source": 0,
        "skipped_no_marker": 0,
        "skipped_non_object": 0,
        "error": 0,
    }

    labeled_paths: list[str] = []

    for path in sorted(root.rglob("*.json")):
        result = _label_file(path, dry_run=args.dry_run)
        counts[result] = counts.get(result, 0) + 1
        if result == "labeled":
            labeled_paths.append(str(path))

    mode = "DRY RUN — " if args.dry_run else ""
    print(f"{mode}Scan complete.")
    print(f"  Files that would be / were labeled:      {counts['labeled']}")
    print(f"  Already have data_source (skipped):      {counts['skipped_has_source']}")
    print(f"  No smoke marker (skipped):               {counts['skipped_no_marker']}")
    print(f"  Non-object JSON (skipped):               {counts['skipped_non_object']}")
    print(f"  Errors (skipped):                        {counts['error']}")

    if labeled_paths:
        print("\nAffected files:")
        for p in labeled_paths[:50]:
            print(f"  {p}")
        if len(labeled_paths) > 50:
            print(f"  ... and {len(labeled_paths) - 50} more")


if __name__ == "__main__":
    main()
