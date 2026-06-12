#!/usr/bin/env bash
# 10-minute core track loop (SHIM deferred last per ROADMAP_EXECUTION.md).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MINUTES="${1:-10}"
export CHELATED_EXEC_TRACK=core
exec python3 scripts/run_10min_priority_bhs_loop.py \
  --minutes "$MINUTES" \
  --bhs-target 100 \
  --goal-file docs/loop_workers/GOAL_CORE_TRACK.md \
  --skip-shim-improvement-cycles