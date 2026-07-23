"""CLI for the offline sequential D2 crossover kill-screen."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.drift_recovery.d2.crossover import run_d2  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the lean D2 Regime-C kill-screen")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("research/drift_recovery/out/d2")
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Archive a non-empty output directory before a clean rerun",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("research/drift_recovery/protocols/d2_crossover.yaml"),
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = run_d2(
        output_dir=args.output_dir,
        protocol_path=args.protocol,
        device=args.device,
        command="python " + " ".join(sys.argv),
        force=args.force,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
