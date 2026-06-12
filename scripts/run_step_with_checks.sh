#!/usr/bin/env bash
# Run a long-running step with companion checks in parallel after an elapsed-threshold.
#
# Primary goal:
# - Avoid idle operator time when long tests run.
# - Keep the queue moving by running an independent low-cost validation while
#   the long-running primary command continues.
# - Require both primary and companion checks to pass.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_THRESHOLD_SECONDS=30
THRESHOLD_SECONDS=$DEFAULT_THRESHOLD_SECONDS
FORCE_COMPANION=0
PRIMARY_CMD=()
COMPANION_CMDS=(
  "python scripts/check_block_flag.py"
)

print_usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_step_with_checks.sh [options] -- <primary command...>

Options:
  --companion "<command>"  Add one companion check command (can be repeated).
  --threshold <seconds>     Delay before launching companion checks (default 30).
  --always                  Always launch companion checks, regardless of runtime.
  --help, -h                Show this help text.

Companion checks run in parallel if the primary command still executes after the
threshold. Step exits non-zero unless both primary and all launched checks pass.
If the primary command exits before the threshold, no companion checks are launched
unless --always is set.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --companion)
      shift
      if [[ $# -eq 0 ]]; then
        echo "run_step_with_checks.sh: --companion requires a command" >&2
        exit 2
      fi
      COMPANION_CMDS+=("$1")
      ;;
    --threshold)
      shift
      if [[ $# -eq 0 || ! "$1" =~ ^[0-9]+$ ]]; then
        echo "run_step_with_checks.sh: --threshold requires a non-negative integer" >&2
        exit 2
      fi
      THRESHOLD_SECONDS="$1"
      ;;
    --always)
      FORCE_COMPANION=1
      ;;
    --help|-h)
      print_usage
      exit 0
      ;;
    --)
      shift
      PRIMARY_CMD=("$@")
      break
      ;;
    *)
      echo "run_step_with_checks.sh: unknown argument: $1" >&2
      echo "Use --help for usage." >&2
      exit 2
      ;;
  esac
  shift
done

if [[ ${#PRIMARY_CMD[@]} -eq 0 ]]; then
  echo "run_step_with_checks.sh: no primary command provided" >&2
  print_usage >&2
  exit 2
fi

WORK_DIR="$(mktemp -d)"

PRIMARY_LOG="${WORK_DIR}/primary.log"
PRIMARY_PID=
PRIMARY_RC=0
run_primary() {
  "$@" >"$PRIMARY_LOG" 2>&1
  return $?
}

echo "Primary command:"
printf '  %q ' "${PRIMARY_CMD[@]}"
printf '\n'
echo "Companion threshold: ${THRESHOLD_SECONDS}s"
echo "Root: $ROOT_DIR"

run_primary "${PRIMARY_CMD[@]}" &
PRIMARY_PID=$!

sleep 1
companion_started=0
companion_pids=()
companion_logs=()
companion_cmds=()

run_companion_check() {
  local cmd="$1"
  local idx="$2"
  local log_file="${WORK_DIR}/companion_${idx}.log"
  set +e
  bash -c "$cmd" >"$log_file" 2>&1
  local rc=$?
  set -e
  echo "$rc" >"${WORK_DIR}/companion_${idx}.rc"
}

if (( FORCE_COMPANION == 1 )); then
  if (( ${#COMPANION_CMDS[@]} > 0 )); then
    echo "Companion checks requested via --always"
    companion_started=1
    for i in "${!COMPANION_CMDS[@]}"; do
      cmd="${COMPANION_CMDS[$i]}"
        run_companion_check "$cmd" "$i" &
        companion_pids+=("$!")
        companion_logs+=("${WORK_DIR}/companion_${i}.log")
        companion_cmds+=("$cmd")
        echo "  [$i] $cmd"
      done
  fi
else
  elapsed=0
  while true; do
    if ! kill -0 "$PRIMARY_PID" 2>/dev/null; then
      break
    fi
    if (( elapsed >= THRESHOLD_SECONDS )); then
      companion_started=1
      if (( ${#COMPANION_CMDS[@]} > 0 )); then
        echo "Primary still running after ${THRESHOLD_SECONDS}s; launching companion checks"
        for i in "${!COMPANION_CMDS[@]}"; do
          cmd="${COMPANION_CMDS[$i]}"
          run_companion_check "$cmd" "$i" &
          companion_pids+=("$!")
          companion_logs+=("${WORK_DIR}/companion_${i}.log")
          companion_cmds+=("$cmd")
          echo "  [$i] $cmd"
        done
      else
        echo "Primary still running after threshold, but no companion commands configured."
      fi
      break
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
fi

if ! wait "$PRIMARY_PID"; then
  PRIMARY_RC=1
fi

if (( companion_started == 0 && FORCE_COMPANION == 0 )); then
  echo
  echo "Primary command completed before companion threshold."
fi

overall_rc=0
if (( ${#companion_pids[@]} > 0 )); then
  for idx in "${!companion_pids[@]}"; do
    pid="${companion_pids[$idx]}"
    log_file="${companion_logs[$idx]}"
    cmd="${companion_cmds[$idx]}"
    if ! wait "$pid"; then
      overall_rc=1
    fi
    rc_file="${log_file%.log}.rc"
    rc_val="$(cat "$rc_file" 2>/dev/null || echo 1)"
    if [[ "$rc_val" != "0" ]]; then
      overall_rc=1
    fi
  done
fi

if (( PRIMARY_RC != 0 )); then
  overall_rc=1
fi

if (( overall_rc != 0 )); then
  echo
  echo "=== Step failed ==="
  echo
  echo "Primary log: $PRIMARY_LOG"
  if (( ${#companion_logs[@]} > 0 )); then
    for idx in "${!companion_logs[@]}"; do
      log_file="${companion_logs[$idx]}"
      cmd="${companion_cmds[$idx]}"
      rc_file="${log_file%.log}.rc"
      echo "Companion log (${cmd}): $log_file (rc=$(cat "$rc_file" 2>/dev/null || echo unknown))"
    done
  fi
else
  echo
  echo "=== Step passed ==="
  echo "Primary and companion checks passed."
fi

if (( overall_rc == 0 )); then
  rm -rf "$WORK_DIR"
fi

exit "$overall_rc"
