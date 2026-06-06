#!/usr/bin/env bash
# /loop 10m — 10-minute BHS priority loop (goal + workers → BHS 100/100).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MINUTES="${1:-10}"
exec python3 scripts/run_10min_priority_bhs_loop.py --minutes "$MINUTES" --bhs-target 100