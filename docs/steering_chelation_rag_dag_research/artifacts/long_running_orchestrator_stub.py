#!/usr/bin/env python3
"""
Long-running sustained phase orchestrator (v0.3).

Delegates turn execution to scripts/phase_development_loop.py for:
  - next-slice recommendations (phase plan + SHIM-CD register)
  - bounded handler execution (verify, evidence scripts, tests)
  - turn reports with turn_status=completed and trigger_next_turn

Usage:
  nohup python docs/steering_chelation_rag_dag_research/artifacts/long_running_orchestrator_stub.py \
    --auto-continue --max-rounds 100 > artifacts/phase_loop/orchestrator.log 2>&1 &
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-plan", type=Path, default=None)
    parser.add_argument("--driver", type=Path, default=None)
    parser.add_argument("--max-rounds", type=int, default=100)
    parser.add_argument("--auto-continue", action="store_true", default=True)
    parser.add_argument("--max-wall-min", type=int, default=10, help="Unused; kept for CLI compat")
    parser.add_argument("--round-timebox-min", type=int, default=45, help="Unused; kept for CLI compat")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent.parent.parent
    loop_script = root / "scripts" / "phase_development_loop.py"
    if not loop_script.is_file():
        log(f"ERROR: missing {loop_script}")
        sys.exit(2)

    cmd = [
        sys.executable,
        str(loop_script),
        "--max-turns",
        str(args.max_rounds),
    ]
    if args.auto_continue:
        cmd.append("--auto-continue")
    else:
        cmd.append("--once")

    log(f"Delegating to phase_development_loop: {' '.join(cmd)}")
    if args.phase_plan:
        log(f"Phase plan (reference): {args.phase_plan}")
    if args.driver:
        log(f"Driver doc (reference): {args.driver}")

    proc = subprocess.run(cmd, cwd=root)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()