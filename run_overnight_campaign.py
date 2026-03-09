"""
Overnight weight-refinement campaign with Session 29 research fixes.

This wraps run_weight_refinement_campaign.py with the corrected parameters
identified in Session 29 Tier 3 research:

1. use_quantization=True in benchmark scripts (already patched in source)
2. LR=0.01 (Phase 1 sweet spot, not the old 0.001 default)
3. threshold=1 (already patched in source)
4. Adapter init fixes (Procrustes: randn not zeros; Low-rank: 0.01 not 0.001)

Run this script to launch a full campaign overnight:

    python run_overnight_campaign.py

Or with the large sweep background launch:

    python run_overnight_campaign.py --launch-large-sweep
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch overnight weight-refinement campaign (Session 29 fixes)"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=100,
        help="Query budget (default: 100, up from Session 28's 50)",
    )
    parser.add_argument(
        "--distill-cycles",
        type=int,
        default=5,
        help="Distillation cycles (default: 5, up from Session 28's 3)",
    )
    parser.add_argument(
        "--distill-queries-per-cycle",
        type=int,
        default=50,
        help="Queries per distillation cycle (default: 50, up from Session 28's 30)",
    )
    parser.add_argument(
        "--distill-epochs",
        type=int,
        default=5,
        help="Epochs per distillation cycle (default: 5, up from Session 28's 3)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01, Phase 1 validated sweet spot)",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Teacher model for distillation (default: all-mpnet-base-v2, 768-dim)",
    )
    parser.add_argument(
        "--launch-large-sweep",
        action="store_true",
        help="Launch large sweep in background after bounded phases",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = PROJECT_ROOT / "experiment_runs" / f"weight-refinement-{timestamp}-session29"

    print("=" * 60)
    print("Session 29 Overnight Weight Refinement Campaign")
    print("=" * 60)
    print(f"Run directory: {run_dir}")
    print(f"Query budget: {args.max_queries}")
    print(f"Distillation: {args.distill_cycles} cycles x {args.distill_queries_per_cycle} queries x {args.distill_epochs} epochs")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Teacher model: {args.teacher}")
    print(f"Large sweep: {'yes' if args.launch_large_sweep else 'no'}")
    print()
    print("Fixes applied in this campaign:")
    print("  - use_quantization=True (chelation path enabled)")
    print("  - threshold=1 (not 3)")
    print("  - LR=0.01 (not 0.001)")
    print("  - Procrustes init: randn (not zeros)")
    print("  - Low-rank init: std=0.01 (not 0.001)")
    print("=" * 60)

    command = [
        sys.executable,
        "-u",
        "run_weight_refinement_campaign.py",
        "--run-dir",
        str(run_dir),
        "--model",
        "sentence-transformers/all-MiniLM-L6-v2",
        "--max-queries",
        str(args.max_queries),
        "--distill-cycles",
        str(args.distill_cycles),
        "--distill-queries-per-cycle",
        str(args.distill_queries_per_cycle),
        "--distill-epochs",
        str(args.distill_epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--teacher",
        args.teacher,
    ]

    if args.launch_large_sweep:
        command.append("--launch-large-sweep")

    print(f"\nLaunching: {' '.join(command)}\n")
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
