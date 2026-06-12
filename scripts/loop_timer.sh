#!/usr/bin/env bash
# 10-minute priority loop with BHS 100/100 worker charter.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec python3 scripts/run_10min_priority_bhs_loop.py --minutes "${1:-10}" --bhs-target 100