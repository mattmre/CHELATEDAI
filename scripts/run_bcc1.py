#!/usr/bin/env python3
"""Analyze METHOD_DEV; fail closed on combined CONFIRMATORY packs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bcc1_experiment import BCC1ValidationError, run_bcc1_files  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze an existing SHA-bound BCC-1 METHOD_DEV JSON pack. "
            "Combined CONFIRMATORY packs are blocked before pack access until "
            "separate decisions/outcomes and operational evidence schemas exist."
        )
    )
    parser.add_argument(
        "--pack",
        required=True,
        type=Path,
        help="Path to an existing frozen chelatedai.bcc1.pack.v1 JSON artifact.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to its public chelatedai.bcc1.manifest.v1 JSON envelope.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New path for the deterministic report; existing files are refused.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_bcc1_files(
            pack_path=args.pack,
            manifest_path=args.manifest,
            output_path=args.output,
        )
    except BCC1ValidationError as exc:
        print("BCC-1 validation failed: {}".format(exc), file=sys.stderr)
        return 2
    outcome = "exploratory METHOD_DEV diagnostic written; no REPORT evaluated"
    print(
        "BCC-1 {}: {} (pack_id={}, report_queries={})".format(
            outcome,
            args.output,
            report["pack_id"],
            report["data_contract"]["report_query_count"],
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
