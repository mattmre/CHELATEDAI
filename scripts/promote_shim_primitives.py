#!/usr/bin/env python3
"""Promote shim research artifacts to guarded root runtime copies.

SHIM promotion path:
- shim_node.py -> shim_node_promoted.py
- shim_collapse_benchmark_extension.py -> shim_collapse_benchmark_extension_promoted.py

Production imports should use the promoted root copies when CHELATED_SHIM_PROMOTED=1.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROMOTION_TARGETS = [
    {
        "source": ROOT / "docs" / "steering_chelation_rag_dag_research" / "artifacts" / "shim_node.py",
        "dest": ROOT / "shim_node_promoted.py",
    },
    {
        "source": ROOT
        / "docs"
        / "steering_chelation_rag_dag_research"
        / "artifacts"
        / "shim_collapse_benchmark_extension.py",
        "dest": ROOT / "shim_collapse_benchmark_extension_promoted.py",
    },
]
HEADER_MARK = "PROMOTED_FROM_RESEARCH_ARTIFACTS"


def _rewrite_header(text: str) -> str:
    text = text.replace(
        "Placement: research/artifacts/ ONLY. Do not import from any core runtime file",
        f"{HEADER_MARK}: copied to repo root for guarded runtime import (CHELATED_SHIM_PROMOTED=1). "
        "Do not import from docs/steering_chelation_rag_dag_research/artifacts/ at runtime",
    )
    return text


def main() -> int:
    for spec in PROMOTION_TARGETS:
        source = spec["source"]
        dest = spec["dest"]
        if not source.exists():
            print(f"ERROR: missing source {source}", file=sys.stderr)
            return 2

        src_text = source.read_text(encoding="utf-8")
        dest_text = _rewrite_header(src_text)
        marker = f"# {HEADER_MARK}\n# source: {source.relative_to(ROOT)}\n"
        destination_text = marker + dest_text

        if dest.exists():
            existing = dest.read_text(encoding="utf-8")
            if existing == destination_text:
                print(f"Up to date: {dest}")
                continue

        dest.write_text(destination_text, encoding="utf-8")
        print(f"Wrote {dest} ({dest.stat().st_size} bytes)")
    return _smoke()


def _smoke() -> int:
    for spec in PROMOTION_TARGETS:
        if _smoke_one(spec) != 0:
            return 2
    return 0


def _smoke_one(spec) -> int:
    dest = spec["dest"]

    try:
        text = dest.read_text(encoding="utf-8")
        compile(text, str(dest), "exec")
        print(f"{dest.name} smoke compile OK")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Smoke import failure for {dest}: {exc!r}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
