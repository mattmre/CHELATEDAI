"""CLI for Bundle 53 / Item 3D one-way PM-6 shadow probe."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_pm6_shadow_probe import execute as probe_execute  # noqa: E402


def execute(
    mode: str,
    pm6_manifest: Optional[str] = None,
    *,
    historical_fingerprints: Optional[Mapping[str, Sequence[str]]] = None,
) -> dict:
    return probe_execute(mode, pm6_manifest, historical_fingerprints=historical_fingerprints)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("local-drycheck", "run-live-shadow"))
    parser.add_argument("--pm6-manifest", default=None)
    parser.add_argument("--historical-fingerprints", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    hist_fp = None
    if args.historical_fingerprints:
        hist_fp = json.loads(Path(args.historical_fingerprints).read_text(encoding="utf-8"))
    report = execute(args.mode, args.pm6_manifest, historical_fingerprints=hist_fp)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report.get("executed", False):
        return 2
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
