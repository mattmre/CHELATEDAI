"""Single command for the complete Phase B1 D1 build."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.drift_recovery.d1.paper_stats import run_d1  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Phase B1 D1 statistics and paper assets")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/drift_recovery/out/d1"),
    )
    parser.add_argument("--rebuild-pack", action="store_true")
    parser.add_argument("--rerun-learning-curve", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--paper", type=Path, default=None, help="Read-only paper source for edit locations")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = run_d1(
        output_dir=args.output_dir,
        rebuild_pack=args.rebuild_pack,
        rerun_learning_curve=args.rerun_learning_curve,
        device=args.device,
        paper_path=args.paper,
        command="python " + " ".join(sys.argv),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
