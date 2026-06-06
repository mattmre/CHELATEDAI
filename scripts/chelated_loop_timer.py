#!/usr/bin/env python3
"""Wall-clock loop timer for ChelatedAI priority work.

Runs until duration expires, optionally invoking --on-expire command each tick
or once at end. Used by run_10min_priority_bhs_loop.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description="ChelatedAI wall-clock loop timer")
    parser.add_argument("--minutes", type=float, default=10.0, help="Duration in minutes")
    parser.add_argument("--goal", default="", help="Goal text (logged only)")
    parser.add_argument("--workers", default="", help="Worker instructions (logged only)")
    parser.add_argument("--bhs-target", type=int, default=100, help="Target BHS score")
    parser.add_argument(
        "--on-expire",
        default="",
        help="Shell command to run once when timer fires (optional)",
    )
    parser.add_argument(
        "--tick-sec",
        type=float,
        default=30.0,
        help="Progress log interval",
    )
    args = parser.parse_args()

    deadline = time.monotonic() + args.minutes * 60.0
    print(f"[{_utc()}] loop timer start: {args.minutes} min, BHS target {args.bhs_target}/100")
    if args.goal:
        print(f"GOAL: {args.goal[:500]}")
    if args.workers:
        print(f"WORKERS: {args.workers[:800]}")

    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        print(f"[{_utc()}] remaining {remaining:.0f}s")
        sleep_for = min(args.tick_sec, max(0.0, remaining))
        if sleep_for <= 0:
            break
        time.sleep(sleep_for)

    print(f"[{_utc()}] timer expired ({args.minutes} min)")
    if args.on_expire:
        print(f"on-expire: {args.on_expire}")
        proc = subprocess.run(args.on_expire, shell=True)
        return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())