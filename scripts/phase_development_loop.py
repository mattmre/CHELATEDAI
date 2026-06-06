#!/usr/bin/env python3
"""Sustained phase/slice development loop with recommendations and auto-continue.

Runs bounded work items (verification, evidence scripts, gates) in priority order
derived from FULL_SHIM_LOOP_PHASE_PLAN.md and docs/next-session.md SHIM-CD rows.
Each turn emits a report with ``turn_status: completed``; ``--auto-continue`` chains
the next recommended slice without idle wall time (see SUSTAINED_PHASE_ROUND_DRIVER.md).

Usage:
  # One turn: recommend + execute + report
  python scripts/phase_development_loop.py --once

  # Loop until max turns or all executable slices done
  python scripts/phase_development_loop.py --auto-continue --max-turns 50

  # Recommendation only (no execution)
  python scripts/phase_development_loop.py --recommend-only

  # Background (user pattern)
  nohup python scripts/phase_development_loop.py --auto-continue \\
    > artifacts/phase_loop/orchestrator.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
LOOP_DIR = ROOT / "artifacts" / "phase_loop"
STATE_PATH = LOOP_DIR / "state.json"
REPORT_DIR = LOOP_DIR / "reports"
ARTIFACT_DIR = ROOT / "artifacts"

SCHEDULER_EVIDENCE_PREFIX = "bhs_shim_scheduler_evidence_"

PHASE_PLAN = ROOT / "docs/steering_chelation_rag_dag_research/FULL_SHIM_LOOP_PHASE_PLAN.md"
DRIVER_DOC = (
    ROOT / "docs/steering_chelation_rag_dag_research/artifacts/SUSTAINED_PHASE_ROUND_DRIVER.md"
)
NEXT_SESSION = ROOT / "docs/next-session.md"
OVERRIDE_FILE = ROOT / "docs/steering_chelation_rag_dag_research/artifacts/OPERATOR_OVERRIDE.md"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str) -> None:
    print(f"[{_utc_now()}] {msg}", flush=True)


@dataclass
class WorkSlice:
    slice_id: str
    title: str
    phase: int
    shim_cd: Optional[str]
    blocking: bool
    priority: int  # lower = sooner
    handler: str  # key into HANDLERS
    rationale: str
    completed_key: str  # state["completed"][key]


HANDLERS: Dict[str, Callable[[], Tuple[bool, str]]] = {}


def _register_handler(name: str):
    def deco(fn: Callable[[], Tuple[bool, str]]):
        HANDLERS[name] = fn
        return fn

    return deco


@_register_handler("verify_shim_development")
def _run_verify_shim() -> Tuple[bool, str]:
    proc = subprocess.run(
        ["bash", "scripts/verify_shim_development.sh"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    ok = proc.returncode == 0
    tail = (proc.stdout or "")[-2000:] + (proc.stderr or "")[-1000:]
    return ok, f"exit={proc.returncode}\n{tail}"


@_register_handler("record_inference_evidence")
def _run_inference_evidence() -> Tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "scripts/record_shim_inference_evidence.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")


@_register_handler("record_prod_evidence")
def _run_prod_evidence() -> Tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "scripts/record_shim_prod_evidence.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")


@_register_handler("record_tts_intercept_evidence")
def _run_tts_intercept() -> Tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "scripts/record_shim_tts_intercept_evidence.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")


@_register_handler("five_worker_shim_gate")
def _run_five_worker_gate() -> Tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "scripts/run_five_worker_shim_gate.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")


@_register_handler("record_shim_scheduler_evidence")
def _run_shim_scheduler_evidence() -> Tuple[bool, str]:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/record_shim_scheduler_evidence.py",
            "--strict",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")


@_register_handler("record_engine_embed_evidence")
def _run_engine_embed_evidence() -> Tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "scripts/record_shim_engine_embed_evidence.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")


@_register_handler("shim_insert_once_tests")
def _run_insert_once_tests() -> Tuple[bool, str]:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_shim_promoted_insert_once",
            "-q",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")


@_register_handler("record_promoted_sip_evidence")
def _run_promoted_sip_evidence() -> Tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "scripts/record_shim_promoted_sip_evidence.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")


@_register_handler("promote_shim_primitives")
def _run_promote_shim() -> Tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "scripts/promote_shim_primitives.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")


@_register_handler("core_smoke_gate")
def _run_core_smoke() -> Tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "scripts/smoke_pipeline.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc.returncode == 0, (proc.stdout or "")[-2000:] + (proc.stderr or "")[-800:]


@_register_handler("core_sedimentation_tests")
def _run_core_sedimentation_tests() -> Tuple[bool, str]:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "test_sedimentation_loss",
            "test_sedimentation_trainer",
            "-q",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")


@_register_handler("core_ceiling_unittest")
def _run_core_ceiling_unittest() -> Tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "test_smoke_pipeline_ceiling", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")


@_register_handler("shim_unit_tests")
def _run_shim_tests() -> Tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_shim*.py", "-v"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc.returncode == 0, (proc.stdout or "")[-1500:] + (proc.stderr or "")[-500:]


def build_slice_registry() -> List[WorkSlice]:
    """Ordered backlog of executable slices (code/tests/evidence), not doc-only."""
    return [
        WorkSlice(
            slice_id="CORE-SLICE-SMOKE",
            title="Production smoke_pipeline floor+ceiling",
            phase=0,
            shim_cd=None,
            blocking=False,
            priority=1,
            handler="core_smoke_gate",
            rationale="ROADMAP_EXECUTION step 0 gate before ML fixes.",
            completed_key="core_smoke_gate",
        ),
        WorkSlice(
            slice_id="CORE-SLICE-CEILING-UT",
            title="unittest test_smoke_pipeline_ceiling",
            phase=0,
            shim_cd=None,
            blocking=False,
            priority=2,
            handler="core_ceiling_unittest",
            rationale="ROADMAP_EXECUTION — AntigravityEngine ceiling regression.",
            completed_key="core_ceiling_unittest",
        ),
        WorkSlice(
            slice_id="CORE-SLICE-SEDIMENTATION",
            title="Sedimentation loss + trainer unit tests",
            phase=0,
            shim_cd=None,
            blocking=False,
            priority=3,
            handler="core_sedimentation_tests",
            rationale="ROADMAP_EXECUTION steps 1–2 regression surface.",
            completed_key="core_sedimentation_tests",
        ),
        WorkSlice(
            slice_id="SHIM-SLICE-CD01-ENGINE-EMBED",
            title="Promoted SIP at AntigravityEngine.get_chelated_vector",
            phase=3,
            shim_cd="SHIM-CD-01",
            blocking=True,
            priority=201,
            handler="record_engine_embed_evidence",
            rationale="Second production SIP seam + JSON evidence.",
            completed_key="cd01_engine_embed_sip",
        ),
        WorkSlice(
            slice_id="SHIM-SLICE-CD01-INSERT-ONCE",
            title="Insert-once unittest for promoted cascade",
            phase=3,
            shim_cd="SHIM-CD-01",
            blocking=True,
            priority=202,
            handler="shim_insert_once_tests",
            rationale="BHS regression: unique cascade_ids per apply.",
            completed_key="cd01_insert_once_tests",
        ),
        WorkSlice(
            slice_id="SHIM-SLICE-CD01-SIP",
            title="Promote minimal ShimNode / insert-once at VectorSteerer seam",
            phase=3,
            shim_cd="SHIM-CD-01",
            blocking=True,
            priority=203,
            handler="record_promoted_sip_evidence",
            rationale="First real promoted SIP probe + JSON evidence.",
            completed_key="cd01_promoted_sip_probe",
        ),
        WorkSlice(
            slice_id="SHIM-SLICE-FIVE-WORKER",
            title="Five parallel worker unittest gate (A–E)",
            phase=0,
            shim_cd="SHIM-CD-06",
            blocking=True,
            priority=204,
            handler="five_worker_shim_gate",
            rationale="Measurable 5-worker execution for process fidelity.",
            completed_key="five_worker_shim_gate",
        ),
        WorkSlice(
            slice_id="SHIM-SLICE-SCHEDULER-06",
            title="Verify SHIM scheduler task wiring (019e669bf1bb)",
            phase=0,
            shim_cd="SHIM-CD-06",
            blocking=True,
            priority=206,
            handler="record_shim_scheduler_evidence",
            rationale="Close SHIM-CD-06 by proving scheduler dispatch intent and capacity.",
            completed_key="cd06_scheduler_evidence",
        ),
        WorkSlice(
            slice_id="SHIM-SLICE-PROMOTE",
            title="Promote shim_node to shim_node_promoted.py",
            phase=0,
            shim_cd="SHIM-CD-02",
            blocking=True,
            priority=205,
            handler="promote_shim_primitives",
            rationale="BHS promotion copy to repo root; smoke via __main__.",
            completed_key="promote_shim_primitives",
        ),
        WorkSlice(
            slice_id="SHIM-SLICE-CD05-INFERENCE",
            title="AntigravityEngine.run_inference + enable_tts evidence JSON",
            phase=3,
            shim_cd="SHIM-CD-05",
            blocking=True,
            priority=210,
            handler="record_inference_evidence",
            rationale="Closes partial SHIM-CD-05: engine inference path with TTS + research meta.",
            completed_key="cd05_inference_evidence",
        ),
        WorkSlice(
            slice_id="SHIM-SLICE-VERIFY",
            title="Full shim development verification gate",
            phase=0,
            shim_cd=None,
            blocking=False,
            priority=220,
            handler="verify_shim_development",
            rationale="Regression gate after evidence or seam changes.",
            completed_key="verify_shim_development",
        ),
        WorkSlice(
            slice_id="SHIM-SLICE-CD05-PROD",
            title="VectorSteerer.steer prod probe evidence",
            phase=3,
            shim_cd="SHIM-CD-05",
            blocking=True,
            priority=230,
            handler="record_prod_evidence",
            rationale="Refresh prod steer evidence artifact.",
            completed_key="cd05_prod_evidence",
        ),
        WorkSlice(
            slice_id="SHIM-SLICE-CD05-TTS",
            title="TTSPipeline.apply intercept evidence",
            phase=3,
            shim_cd="SHIM-CD-05",
            blocking=True,
            priority=240,
            handler="record_tts_intercept_evidence",
            rationale="Refresh TTS intercept evidence artifact.",
            completed_key="cd05_tts_intercept",
        ),
        WorkSlice(
            slice_id="SHIM-SLICE-TESTS",
            title="Shim-focused unit test suite",
            phase=0,
            shim_cd="SHIM-CD-04",
            blocking=False,
            priority=250,
            handler="shim_unit_tests",
            rationale="Fast test pass before longer verification.",
            completed_key="shim_unit_tests",
        ),
    ]


# Non-executable recommendations (require agent/human implementation)
ADVISORY_SLICES: List[Dict[str, Any]] = [
    {
        "slice_id": "SHIM-SLICE-CD02-PROMOTE",
        "phase": 0,
        "shim_cd": "SHIM-CD-02",
        "blocking": True,
        "priority": 15,
        "title": "BHS promotion path for research primitives (or honest scope reduction)",
        "rationale": "Primitives isolated under artifacts/; promotion or documented termination.",
        "executable": False,
        "agent_roles": ["D", "J", "E"],
    },
    {
        "slice_id": "SHIM-SLICE-PHASE2-USAGE",
        "phase": 2,
        "shim_cd": None,
        "blocking": False,
        "priority": 25,
        "title": "Phase 2 pivot/resilience on real harness usage (not L9 doc-only)",
        "rationale": "FULL_SHIM_LOOP Phase 2 unblocked; needs runtime pivot evidence.",
        "executable": False,
        "agent_roles": ["B", "C", "I", "J"],
    },
]


def load_state() -> Dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {
        "turn": 0,
        "completed": {},
        "last_report_path": None,
        "started_at": _utc_now(),
    }


def save_state(state: Dict[str, Any]) -> None:
    LOOP_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def parse_block_flag() -> Dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "scripts/check_block_flag.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    blocked = proc.returncode == 1
    debt_match = re.search(
        r"Carried Debt row count \(OPEN\):\s*(\d+)", proc.stdout or ""
    )
    debt_rows = int(debt_match.group(1)) if debt_match else None
    return {
        "blocked": blocked,
        "exit_code": proc.returncode,
        "debt_rows": debt_rows,
        "stdout_tail": (proc.stdout or "")[-800:],
    }


def parse_open_shim_cds() -> List[Dict[str, Any]]:
    if not NEXT_SESSION.exists():
        return []
    text = NEXT_SESSION.read_text(encoding="utf-8")
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith("| SHIM-CD-"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 7:
            continue
        row_id = parts[1]
        status_cell = parts[6]
        blocking_cell = parts[5]
        if "OPEN" not in status_cell.upper():
            continue
        rows.append(
            {
                "id": row_id,
                "blocking": blocking_cell.upper().startswith("YES"),
                "summary": parts[2][:200],
            }
        )
    return rows


def _latest_json_artifact(prefix: str) -> Optional[Dict[str, Any]]:
    """Read the latest JSON artifact matching `prefix` in the artifacts directory."""
    if not ARTIFACT_DIR.exists():
        return None
    matches = sorted(
        ARTIFACT_DIR.glob(f"{prefix}*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not matches:
        return None
    try:
        return json.loads(matches[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _is_cd06_scheduler_verified() -> bool:
    """Return True only when the latest scheduler evidence proves proof-positive CD-06."""
    payload = _latest_json_artifact(SCHEDULER_EVIDENCE_PREFIX)
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("found")) and bool(payload.get("verified"))


def _is_completed_slice(completed: Dict[str, Any], slice_obj: WorkSlice) -> bool:
    """Treat completion entries as valid only when critical evidence gates are still healthy."""
    if not completed.get(slice_obj.completed_key):
        return False
    if slice_obj.completed_key == "cd06_scheduler_evidence":
        return _is_cd06_scheduler_verified()
    return True


def prune_invalid_completions(state: Dict[str, Any]) -> Dict[str, Any]:
    """Drop stale completion markers whose evidence gate has regressed."""
    completed = state.get("completed")
    if not isinstance(completed, dict):
        return state

    for ws in build_slice_registry():
        if ws.completed_key in completed and not _is_completed_slice(completed, ws):
            completed.pop(ws.completed_key, None)

    state["completed"] = completed
    return state


def check_override() -> Tuple[bool, str]:
    try:
        content = OVERRIDE_FILE.read_text(encoding="utf-8")
        if "OVERRIDE: ACTIVE" in content:
            return True, "OVERRIDE ACTIVE"
    except OSError:
        pass
    return False, "OVERRIDE: NONE"


def recommend_next(
    state: Dict[str, Any], *, include_advisory: bool = True
) -> Dict[str, Any]:
    registry = build_slice_registry()
    completed = state.get("completed", {})
    pending_exec = [s for s in registry if not _is_completed_slice(completed, s)]
    pending_exec.sort(key=lambda s: s.priority)

    advisory = sorted(ADVISORY_SLICES, key=lambda x: x["priority"])
    open_cds = parse_open_shim_cds()
    block = parse_block_flag()
    override_on, override_reason = check_override()

    next_exec = pending_exec[0] if pending_exec else None
    next_advisory = advisory[0] if advisory and include_advisory else None

    # Prefer open *blocking* shim CDs only when any exist (deferred shim rows use Blocking=NO)
    if next_exec and open_cds:
        blocking_ids = {r["id"] for r in open_cds if r["blocking"]}
        if blocking_ids:
            for s in pending_exec:
                if s.shim_cd and s.shim_cd in blocking_ids:
                    next_exec = s
                    break

    recommendation = {
        "timestamp": _utc_now(),
        "block_flag": block,
        "override_active": override_on,
        "override_reason": override_reason,
        "open_shim_cds": open_cds,
        "next_executable": asdict(next_exec) if next_exec else None,
        "next_advisory": next_advisory,
        "pending_executable_count": len(pending_exec),
        "phase_plan": str(PHASE_PLAN),
        "driver": str(DRIVER_DOC),
        "substrate_note": (
            "Active track: docs/ROADMAP_EXECUTION.md (core first, SHIM deferred last). "
            "SHIM slices run only after CORE-SLICE-* queue drains unless OVERRIDE ACTIVE."
        ),
    }
    if not next_exec and next_advisory:
        recommendation["primary_action"] = "agent_implementation"
        recommendation["primary_slice_id"] = next_advisory["slice_id"]
    elif next_exec:
        recommendation["primary_action"] = "execute_handler"
        recommendation["primary_slice_id"] = next_exec.slice_id
    else:
        recommendation["primary_action"] = "pause_or_human"
        recommendation["primary_slice_id"] = None

    return recommendation


def execute_slice(slice_obj: WorkSlice) -> Dict[str, Any]:
    handler = HANDLERS.get(slice_obj.handler)
    if handler is None:
        return {
            "success": False,
            "error": f"unknown handler {slice_obj.handler}",
            "output": "",
        }
    start = time.time()
    try:
        ok, output = handler()
    except subprocess.TimeoutExpired as exc:
        ok, output = False, f"timeout: {exc}"
    except Exception as exc:
        ok, output = False, f"exception: {exc!r}"
    return {
        "success": ok,
        "wall_seconds": round(time.time() - start, 2),
        "output_tail": output[-4000:] if output else "",
    }


def write_turn_report(
    turn: int,
    recommendation: Dict[str, Any],
    execution: Optional[Dict[str, Any]],
    *,
    turn_status: str = "completed",
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = REPORT_DIR / f"turn_{turn:04d}_{stamp}"

    report = {
        "turn": turn,
        "turn_status": turn_status,
        "trigger_next_turn": turn_status == "completed",
        "timestamp": _utc_now(),
        "recommendation": recommendation,
        "execution": execution,
        "process_improvement_note": {
            "auto_continue": (
                "When trigger_next_turn is true, run phase_development_loop.py "
                "--auto-continue or read this file from the stub."
            ),
            "measured_wall_sec": execution.get("wall_seconds") if execution else None,
        },
    }

    json_path = base.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        f"# Phase loop turn {turn}",
        "",
        f"- **turn_status**: `{turn_status}`",
        f"- **trigger_next_turn**: `{report['trigger_next_turn']}`",
        f"- **timestamp**: {report['timestamp']}",
        "",
        "## Recommendation",
        "",
    ]
    primary = recommendation.get("primary_slice_id")
    action = recommendation.get("primary_action")
    md_lines.append(f"- **primary_action**: {action}")
    md_lines.append(f"- **primary_slice_id**: {primary}")
    if recommendation.get("next_executable"):
        ex = recommendation["next_executable"]
        md_lines.append(f"- **next_executable**: {ex.get('slice_id')} — {ex.get('title')}")
    if recommendation.get("next_advisory"):
        adv = recommendation["next_advisory"]
        md_lines.append(f"- **next_advisory**: {adv.get('slice_id')} — {adv.get('title')}")
    md_lines.append(f"- **substrate_note**: {recommendation.get('substrate_note')}")
    md_lines.append("")
    if execution:
        md_lines.extend(
            [
                "## Execution",
                "",
                f"- **success**: {execution.get('success')}",
                f"- **wall_seconds**: {execution.get('wall_seconds')}",
                "",
                "```",
                (execution.get("output_tail") or "")[:3000],
                "```",
                "",
            ]
        )
    md_lines.append("## Next step")
    md_lines.append("")
    if turn_status == "completed" and recommendation.get("next_executable"):
        nxt = recommendation["next_executable"]
        md_lines.append(
            f"Auto-continue should run handler `{nxt.get('handler')}` "
            f"for slice `{nxt.get('slice_id')}`."
        )
    elif recommendation.get("next_advisory"):
        adv = recommendation["next_advisory"]
        md_lines.append(
            f"Dispatch agent roles {adv.get('agent_roles')} for `{adv.get('slice_id')}`."
        )
    else:
        md_lines.append("No pending executable slices; human review or OVERRIDE required.")

    md_path = base.with_suffix(".md")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    latest = LOOP_DIR / "latest_turn_report.json"
    latest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return json_path


def run_turn(
    state: Dict[str, Any],
    *,
    execute: bool,
    respect_pause_gate: bool,
) -> Tuple[Dict[str, Any], str]:
    state = prune_invalid_completions(state)
    state["turn"] = int(state.get("turn", 0)) + 1
    turn = state["turn"]
    rec = recommend_next(state)

    override_on = rec.get("override_active")
    block = rec.get("block_flag", {})
    if (
        respect_pause_gate
        and not override_on
        and block.get("blocked")
        and rec.get("primary_action") == "agent_implementation"
    ):
        _log(f"Turn {turn}: PAUSE gate (BLOCKED, no executable slice).")
        report_path = write_turn_report(turn, rec, None, turn_status="paused_gate")
        state["last_report_path"] = str(report_path)
        save_state(state)
        return state, "paused_gate"

    execution: Optional[Dict[str, Any]] = None
    if execute and rec.get("primary_action") == "execute_handler":
        ex_data = rec.get("next_executable")
        if ex_data:
            slice_obj = next(
                s for s in build_slice_registry() if s.slice_id == ex_data["slice_id"]
            )
            _log(f"Turn {turn}: executing {slice_obj.slice_id} ({slice_obj.handler})")
            execution = execute_slice(slice_obj)
            execution["slice_id"] = slice_obj.slice_id
            if execution.get("success"):
                state.setdefault("completed", {})[slice_obj.completed_key] = _utc_now()
            # Re-recommend for report "next" section after this completion
            rec = recommend_next(state)

    report_path = write_turn_report(turn, rec, execution, turn_status="completed")
    state["last_report_path"] = str(report_path)
    _write_agent_handoff(rec)
    save_state(state)
    _log(f"Turn {turn} completed → {report_path}")
    if rec.get("pending_executable_count", 0) == 0 and rec.get("primary_action") == "agent_implementation":
        return state, "awaiting_agent_slice"
    return state, "completed"


def _write_agent_handoff(recommendation: Dict[str, Any]) -> None:
    """Single file agents/cron can read for the next non-script slice."""
    handoff = {
        "timestamp": recommendation.get("timestamp"),
        "primary_action": recommendation.get("primary_action"),
        "primary_slice_id": recommendation.get("primary_slice_id"),
        "next_executable": recommendation.get("next_executable"),
        "next_advisory": recommendation.get("next_advisory"),
        "open_shim_cds": recommendation.get("open_shim_cds"),
        "substrate_note": recommendation.get("substrate_note"),
    }
    path = LOOP_DIR / "NEXT_AGENT_SLICE.json"
    path.write_text(json.dumps(handoff, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase/slice development loop")
    parser.add_argument("--once", action="store_true", help="Single turn then exit")
    parser.add_argument(
        "--auto-continue",
        action="store_true",
        help="Chain turns when report has turn_status=completed",
    )
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--recommend-only", action="store_true")
    parser.add_argument(
        "--sleep-on-pause-min",
        type=int,
        default=10,
        help="Sleep when PAUSE gate blocks agent-only work",
    )
    parser.add_argument(
        "--inter-turn-sec",
        type=float,
        default=0.5,
        help="Delay between auto-continued turns",
    )
    parser.add_argument(
        "--idle-sleep-min",
        type=int,
        default=10,
        help="When only agent/advisory slices remain, sleep this many minutes then re-check",
    )
    parser.add_argument(
        "--exit-when-idle",
        action="store_true",
        help="Stop auto-continue when no executable slices remain (default: sleep and re-check)",
    )
    parser.add_argument("--no-pause-gate", action="store_true")
    args = parser.parse_args()

    LOOP_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()

    if args.recommend_only:
        rec = recommend_next(state)
        # stdout must be pure JSON (loop drivers parse it).
        sys.stdout.write(json.dumps(rec, indent=2) + "\n")
        return 0

    _log(f"Phase development loop starting (turn={state.get('turn', 0)})")

    max_turns = 1 if args.once else args.max_turns
    turns_run = 0

    while turns_run < max_turns:
        state, status = run_turn(
            state,
            execute=not args.recommend_only,
            respect_pause_gate=not args.no_pause_gate,
        )
        turns_run += 1

        if args.once or not args.auto_continue:
            break
        if status == "paused_gate":
            _log(f"Sleeping {args.sleep_on_pause_min}min (PAUSE gate)")
            time.sleep(args.sleep_on_pause_min * 60)
            continue
        if status == "awaiting_agent_slice":
            handoff = LOOP_DIR / "NEXT_AGENT_SLICE.json"
            _log(
                f"Executable backlog drained; agent slice required (see {handoff}). "
                f"Advisory: {load_state().get('last_report_path')}"
            )
            if args.exit_when_idle:
                _log("exit-when-idle: stopping auto-continue until next launch.")
                break
            _log(f"Sleeping {args.idle_sleep_min}min before re-checking recommendations.")
            time.sleep(args.idle_sleep_min * 60)
            continue
        if status != "completed":
            break
        time.sleep(args.inter_turn_sec)

    _log("Phase development loop exiting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
