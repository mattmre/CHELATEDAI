"""CLI for Bundle 52 / Item 3C QSCCI planted control and norm-matched arms."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_qscci_singleton_probe import execute as probe_execute  # noqa: E402


def execute(mode: str, model_name: str = "Qwen/Qwen3.5-2B-Base") -> dict:
    return probe_execute(mode, model_name)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("local-drycheck", "run-live-gpu"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = execute(args.mode, args.model)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed", report.get("executed") is False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
