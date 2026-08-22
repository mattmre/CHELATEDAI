#!/usr/bin/env bash
# scripts/smoke.sh — Brutal Honesty Rulebook v3.3 Rule 5 smoke gate.
#
# Single deterministic smoke command. Exercises the production code path end-to-end.
# Returns 0 only if every check it ran passed.
#
# Honest disclosure (built into the script's output):
#   - Stage 1 (API / surface boot): always runs. Validates the application stack
#     can construct (whatever "the application" is in your repo — FastAPI app,
#     Django settings, CLI argument parser, etc.). The default invocation runs
#     `python -m unittest tests.test_e2e_smoke` — replace with your own surface
#     check if you do not have that file. ChelatedAI standardises on the stdlib
#     `unittest` runner per CLAUDE.md Test Conventions (CI does not install
#     pytest), so pytest invocations would break Stage 1 even when the test
#     file is present.
#   - Stage 2 (production-pipeline import + minimal exercise): runs ONLY if your
#     production deps are importable. If they are not, Stage 2 is reported as
#     SKIPPED with a clear reason — and the script EXITS NON-ZERO because Rule 5
#     demands an actual end-to-end production-path verification.
#
# Two-tier framing per v3.3 Rule 5 (§1 + §10):
#   - Floor: import + surface-check through production code paths (what the
#     reference smoke_pipeline.py.template emits). Acceptable for a v3.2 minimum.
#   - Ceiling: true end-to-end against a real fixture (your real OCR/web/CLI
#     entry point producing a real output that downstream consumers depend on).
#     This is the target. Until you reach ceiling, your PR body's SMOKE: line
#     MUST name the tier you ran AND any ceiling gap MUST appear as a Carried
#     Debt entry in your next-session.md.
#
# Use:
#   bash scripts/smoke.sh                 # full local smoke
#   bash scripts/smoke.sh --api-only      # only Stage 1 (CI use, docs-only PRs)
#   bash scripts/smoke.sh --skip-stage2   # alias for --api-only

# BHS v3.3 integration note (added 2026-05-15 during reconciliation):
# This script is the canonical smoke gate. It should eventually call
# validate_pr_brutal_honesty + run_smoke_pipeline for any changed planning docs,
# AEP cycles, or computational storage artifacts.
# Current state: Real BHS v3.3 scripts exist in this directory. Full wiring into
# the AEP orchestrator and CI is in progress (see reconciliation backlog).
#
# The --api-only / --skip-stage2 flags exist to let docs-only PRs (and pre-deploy
# check-the-API CI jobs) run a partial smoke. v3.2 §4 PR template requires that
# any "complete" PR's SMOKE: line includes the FULL output (not the partial one)
# OR explicitly calls out that this PR is docs-only and Stage 2 was intentionally
# skipped — and even then, the script EXITS NON-ZERO so the skip is visible.

set -u  # treat unset vars as errors; do NOT set -e (we handle errors explicitly)
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Keep the smoke path usable on systems that expose only `python3` (the CI
# contract uses `python`, but local evidence capture should not fail before it
# reaches the production code path).
if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python}"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python3}"
else
    echo "smoke.sh: neither python nor python3 is available" >&2
    exit 2
fi

API_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --api-only|--skip-stage2)
            API_ONLY=1
            ;;
        --help|-h)
            sed -n '2,40p' "$0"
            exit 0
            ;;
        *)
            echo "smoke.sh: unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

# Status accumulator
SMOKE_STATUS=0
EGV_RESULT="NOT RUN"
STAGE1_RESULT="NOT RUN"
STAGE2_RESULT="NOT RUN"
STAGE2_REASON=""
EVALUATION_RESULT="NOT RUN"
EVALUATION_JSON_FILE="$(mktemp "${TMPDIR:-/tmp}/egv-evaluation-smoke.XXXXXX.json")"
trap 'rm -f "$EVALUATION_JSON_FILE"' EXIT

echo "========================================================================="
echo "Brutal Honesty Rulebook v3.3 Rule 5 smoke"
echo "Repo root: $REPO_ROOT"
echo "Started:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "========================================================================="

if [ "$API_ONLY" -eq 1 ]; then
    # API-only is decided before either Slice 2 Docker path starts.  It is a
    # partial Rule 5 check and remains non-zero so callers cannot mistake it
    # for the full production smoke.
    EGV_RESULT="SKIPPED (--api-only before Slice 2 execution)"
    EVALUATION_RESULT="SKIPPED (--api-only before Docker evaluation)"
    echo
    echo "--- Slice 2 execution: SKIPPED (--api-only selected before Docker) ---"
else
    # ----------------------------------------------------------------------------
    # Slice 2 EGV evidence-core smoke — explicit production path
    # ----------------------------------------------------------------------------
    echo
    echo "--- EGV evidence-core smoke (ledger + receipts + public replay) ---"
    if "$PYTHON_BIN" -m egv smoke --json --two-process; then
        EGV_RESULT="PASS (runtime output above; bounded fixture and declared gaps remain)"
        echo "EGV evidence-core PASS"
    else
        EGV_RESULT="FAIL"
        SMOKE_STATUS=1
        echo "EGV evidence-core FAIL" >&2
    fi

    # ----------------------------------------------------------------------------
    # Evaluation slice smoke — deterministic corpus, sandbox, receipts, and OS IPC
    # ----------------------------------------------------------------------------
    echo
    echo "--- Evaluation slice smoke (corpus + hidden evaluator + two processes) ---"
    if "$PYTHON_BIN" -m egv evaluation smoke --json | tee "$EVALUATION_JSON_FILE"; then
        if EVALUATION_SUMMARY="$("$PYTHON_BIN" - "$EVALUATION_JSON_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
if report.get("smoke") != "PASS":
    raise SystemExit("evaluation smoke did not report PASS")
runtime_tier = report.get("runtime_tier")
receipts = report.get("receipts") or {}
journal = receipts.get("journal") or {}
ledger = receipts.get("ledger") or {}
evaluation = report.get("evaluation") or {}
fields = (
    runtime_tier,
    evaluation.get("receipt_count"),
    journal.get("count"),
    journal.get("chain_valid"),
    ledger.get("receipt_count"),
    ledger.get("chain_valid"),
)
if not fields[0] or any(value is None for value in fields[1:]):
    raise SystemExit("evaluation smoke omitted derived runtime or receipt-chain fields")
print("tier={}; evaluation_receipts={}; journal_count={}; journal_chain_valid={}; ledger_receipt_count={}; ledger_chain_valid={}".format(*fields))
PY
        )"; then
            EVALUATION_RESULT="PASS (${EVALUATION_SUMMARY}; model/GPU/network/OpenShell/Qdrant-campaign/real-campaign/dual-Spark phases are not claimed)"
            echo "Evaluation slice PASS (${EVALUATION_SUMMARY})"
        else
            EVALUATION_RESULT="FAIL (runtime report omitted verifiable fields)"
            SMOKE_STATUS=1
            echo "Evaluation slice FAIL — runtime report could not be verified" >&2
        fi
    else
        EVALUATION_RESULT="FAIL"
        SMOKE_STATUS=1
        echo "Evaluation slice FAIL" >&2
    fi
fi

# -----------------------------------------------------------------------------
# Stage 1 — application/surface boot smoke (always runs)
# -----------------------------------------------------------------------------
echo
echo "--- Stage 1: application/surface boot smoke ---"
if [ -f "$REPO_ROOT/tests/test_e2e_smoke.py" ]; then
    # ChelatedAI uses stdlib unittest per CLAUDE.md Test Conventions; pytest is
    # not installed in CI. Invoke via -m unittest with the module path so
    # discovery does not depend on the caller's working directory.
    if "$PYTHON_BIN" -m unittest -v tests.test_e2e_smoke 2>&1; then
        STAGE1_RESULT="PASS"
        echo "Stage 1 PASS"
    else
        STAGE1_RESULT="FAIL"
        SMOKE_STATUS=1
        echo "Stage 1 FAIL — application surface does not boot. Halting before Stage 2." >&2
    fi
else
    STAGE1_RESULT="SKIPPED"
    STAGE2_REASON="tests/test_e2e_smoke.py not present"
    SMOKE_STATUS=1
    echo "Stage 1 SKIPPED — no tests/test_e2e_smoke.py found." >&2
    echo "Add a surface-boot test (FastAPI client, Django setup, CLI parser, etc.)" >&2
    echo "and re-run. Rule 5 requires a deterministic Stage 1 surface check." >&2
fi

# -----------------------------------------------------------------------------
# Stage 2 — production-pipeline smoke (the load-bearing tier)
# -----------------------------------------------------------------------------
if [ "$API_ONLY" -eq 1 ]; then
    STAGE2_RESULT="SKIPPED"
    STAGE2_REASON="--api-only flag set by caller"
    SMOKE_STATUS=1
    echo
    echo "--- Stage 2: SKIPPED (--api-only) ---"
    echo "NOTE: Rule 5 (v3.2) requires end-to-end production-path verification."
    echo "      --api-only is acceptable for docs-only PRs but smoke EXITS NON-ZERO"
    echo "      so the caller must explicitly disclose the skip in PR body SMOKE: line."
elif [ "$STAGE1_RESULT" != "PASS" ]; then
    STAGE2_RESULT="SKIPPED"
    STAGE2_REASON="Stage 1 failed — Stage 2 cannot run on a broken stack"
    echo
    echo "--- Stage 2: SKIPPED (Stage 1 failed) ---"
elif [ ! -f "$REPO_ROOT/scripts/smoke_pipeline.py" ]; then
    if [[ "$EVALUATION_RESULT" == PASS* ]]; then
        STAGE2_RESULT="PASS"
        STAGE2_REASON="Evaluation slice is the current bounded CPU production path"
        echo
        echo "--- Stage 2: PASS (Evaluation slice production path) ---"
    else
        STAGE2_RESULT="FAIL"
        STAGE2_REASON="Evaluation slice smoke failed and no scripts/smoke_pipeline.py is present"
        SMOKE_STATUS=1
        echo
        echo "--- Stage 2: FAIL (no runnable production path) ---" >&2
    fi
else
    echo
    echo "--- Stage 2: production-pipeline smoke ---"
    if "$PYTHON_BIN" "$REPO_ROOT/scripts/smoke_pipeline.py"; then
        STAGE2_RESULT="PASS"
        echo "Stage 2 PASS"
    else
        STAGE2_RESULT="FAIL"
        STAGE2_REASON="see scripts/smoke_pipeline.py output above"
        SMOKE_STATUS=1
        echo "Stage 2 FAIL" >&2
    fi
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo
echo "========================================================================="
echo "SMOKE SUMMARY"
echo "  EGV evidence core:              $EGV_RESULT"
echo "  Evaluation slice:              $EVALUATION_RESULT"
echo "  Stage 1 (surface boot):       $STAGE1_RESULT"
echo "  Stage 2 (production pipeline): $STAGE2_RESULT${STAGE2_REASON:+ ($STAGE2_REASON)}"
echo "  Overall exit code:            $SMOKE_STATUS"
echo "  Finished:                     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "========================================================================="

exit $SMOKE_STATUS
