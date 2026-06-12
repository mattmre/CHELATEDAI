#!/usr/bin/env python3
"""Run a 10-minute priority loop with BHS 100/100 worker instructions.

Chains phase_development_loop turns until wall clock expires, then runs full
shim verification and BHS structural audit on touched modules.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

from scripts.bhs_validator import HonestyTier, run_smoke_pipeline

GOAL_DOC = ROOT / "docs" / "loop_workers" / "GOAL_10MIN_BHS100.md"
ARTIFACT_DIR = ROOT / "artifacts" / "bhs_10min_loop"
STATE_PATH = ARTIFACT_DIR / "loop_run_state.json"

BHS_MODULES = [
    "chelated_shim_research.py",
    "tts_pipeline.py",
    "scripts/record_shim_inference_evidence.py",
    "scripts/phase_development_loop.py",
]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str) -> None:
    print(f"[{_utc()}] {msg}", flush=True)


def _run(
    cmd: list[str],
    *,
    timeout: int = 600,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    run_env.setdefault(
        "CHELATED_SHIM_EVIDENCE_DIR",
        str(ARTIFACT_DIR),
    )
    try:
        return subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=run_env,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            cmd,
            returncode=124,
            stdout=exc.stdout or "",
            stderr=(exc.stderr or "")
            + f"\nCommand timed out after {timeout}s: {' '.join(cmd)}",
        )
    except OSError as exc:
        return subprocess.CompletedProcess(
            cmd,
            returncode=127,
            stdout="",
            stderr=f"Command execution failed: {exc!r}",
        )


def _bhs_audit() -> dict:
    """Run shim verification unit tests required before BHS smoke review."""
    tests = _run(
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_chelated_shim_research",
            "tests.test_shim_inference_evidence",
            "tests.test_shim_promoted_probe",
            "tests.test_shim_promoted_sip_apply",
            "-v",
        ],
        timeout=300,
    )
    return {
        "exit_code": 0 if tests.returncode == 0 else 1,
        "unittest_exit": tests.returncode,
        "unittest_tail": (tests.stdout or "")[-2000:] + (tests.stderr or "")[-500:],
    }


def _bhs_smoke_gate(
    bhs_target: int,
    *,
    include_ceiling: bool = True,
) -> dict:
    """Run floor smoke and optional ceiling smoke if requested."""
    summary = {
        "include_ceiling": include_ceiling,
        "floor": {"passed": False, "exit_code": 1},
        "ceiling": {"passed": False, "exit_code": 1},
    }

    floor_pass = run_smoke_pipeline(HonestyTier.FLOOR)
    summary["floor"]["passed"] = bool(floor_pass)
    summary["floor"]["exit_code"] = 0 if floor_pass else 1

    if bhs_target >= 100 and include_ceiling:
        # Full production smoke is only required when target is the primary goal.
        ceiling_pass = run_smoke_pipeline(HonestyTier.CEILING)
        summary["ceiling"]["passed"] = bool(ceiling_pass)
        summary["ceiling"]["exit_code"] = 0 if ceiling_pass else 1
    else:
        summary["ceiling"]["passed"] = True
        summary["ceiling"]["exit_code"] = 0

    return summary


def _promote_shim_if_needed() -> dict:
    proc = _run([sys.executable, "scripts/promote_shim_primitives.py"], timeout=180)
    return {
        "exit_code": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-1500:],
        "stderr_tail": (proc.stderr or "")[-500:],
    }


def _run_bhs_improvement_cycle(*, bhs_target: int, cycle_index: int) -> dict:
    """When phase slices are done, keep iterating: evidence refresh + tests + smoke."""
    _log(f"BHS improvement cycle {cycle_index} (target {bhs_target}/100)")
    steps: list[dict] = []

    for label, cmd in [
        ("promote", [sys.executable, "scripts/promote_shim_primitives.py"]),
        ("prod_evidence", [sys.executable, "scripts/record_shim_prod_evidence.py"]),
        ("tts_evidence", [sys.executable, "scripts/record_shim_tts_intercept_evidence.py"]),
        ("inference_evidence", [sys.executable, "scripts/record_shim_inference_evidence.py"]),
        ("promoted_sip_evidence", [sys.executable, "scripts/record_shim_promoted_sip_evidence.py"]),
        ("engine_embed_evidence", [sys.executable, "scripts/record_shim_engine_embed_evidence.py"]),
        ("insert_once_tests", [sys.executable, "-m", "unittest", "tests.test_shim_promoted_insert_once", "-q"]),
        ("five_worker", [sys.executable, "scripts/run_five_worker_shim_gate.py"]),
        ("shim_scheduler", [sys.executable, "scripts/record_shim_scheduler_evidence.py", "--strict"]),
        (
            "shim_unittests",
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_shim*.py", "-q"],
        ),
    ]:
        proc = _run(cmd, timeout=300)
        steps.append(
            {
                "step": label,
                "exit_code": proc.returncode,
                "ok": proc.returncode == 0,
            }
        )

    smoke = _bhs_smoke_gate(bhs_target, include_ceiling=bhs_target >= 100)
    audit = _bhs_audit()
    result = {
        "timestamp": _utc(),
        "cycle_index": cycle_index,
        "bhs_target": bhs_target,
        "steps": steps,
        "all_steps_ok": all(s["ok"] for s in steps),
        "bhs_smoke": smoke,
        "bhs_audit": audit,
        "cycle_ok": (
            all(s["ok"] for s in steps)
            and audit.get("exit_code") == 0
            and smoke["floor"]["exit_code"] == 0
            and smoke["ceiling"]["exit_code"] == 0
        ),
    }
    cycle_dir = ARTIFACT_DIR / "cycles"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    path = cycle_dir / f"improvement_{cycle_index:03d}_{datetime.now(timezone.utc).strftime('%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="10-minute BHS priority loop")
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--bhs-target", type=int, default=100)
    parser.add_argument("--inter-turn-sec", type=float, default=0.25)
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    goal_text = GOAL_DOC.read_text(encoding="utf-8") if GOAL_DOC.exists() else ""
    worker_brief = (
        "Iteratively improve implemented code toward BHS "
        f"{args.bhs_target}/100; re-test after each slice; no doc-only work."
    )

    manifest = {
        "started_at": _utc(),
        "minutes": args.minutes,
        "bhs_target": args.bhs_target,
        "goal_doc": str(GOAL_DOC),
        "worker_brief": worker_brief,
    }
    worker_instructions = {
        "bhs_target": args.bhs_target,
        "mandate": worker_brief,
        "roles": {
            "Goal": "Execute phase slices only; reject doc-only backlog while BLOCKED.",
            "A": "chelated_shim_research + registry probes — iterate to BHS 100.",
            "B": "VectorSteerer / TTS seams — tests must fail on regression.",
            "C": "Inference + enable_tts evidence scripts.",
            "D": "Promotion + apply_shim_cascade SIP apply.",
            "E": "Integrator: verify_shim_development + five_worker gate + block flag.",
        },
        "iteration_rule": "After each slice: unittest + record JSON artifact before next slice.",
    }
    (ARTIFACT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (ARTIFACT_DIR / "WORKER_INSTRUCTIONS.json").write_text(
        json.dumps(worker_instructions, indent=2) + "\n", encoding="utf-8"
    )
    _log(f"Goal doc: {GOAL_DOC}")
    _log(worker_brief)

    deadline = time.monotonic() + args.minutes * 60.0
    turns = 0

    idle_improvement_sec = 90.0
    improvement_cycles: list[dict] = []
    improvement_index = 0
    last_improvement_at = 0.0

    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        rec_proc = _run(
            [sys.executable, "scripts/phase_development_loop.py", "--recommend-only"],
            timeout=60,
        )
        pending = 0
        if rec_proc.returncode == 0 and rec_proc.stdout:
            try:
                raw = rec_proc.stdout.strip()
                if not raw.startswith("{"):
                    start = raw.find("{")
                    raw = raw[start:] if start >= 0 else raw
                rec = json.loads(raw)
                pending = int(rec.get("pending_executable_count") or 0)
            except json.JSONDecodeError:
                pending = 0

        if pending == 0:
            now = time.monotonic()
            if now - last_improvement_at >= idle_improvement_sec and remaining > 120:
                improvement_index += 1
                improvement_cycles.append(
                    _run_bhs_improvement_cycle(
                        bhs_target=args.bhs_target,
                        cycle_index=improvement_index,
                    )
                )
                last_improvement_at = now
                if remaining < 45:
                    break
                continue

            if remaining <= 45.0:
                _log("executable backlog drained and timer near expiration; exiting loop")
                break

            sleep_for = min(15.0, max(1.0, remaining - 40.0))
            _log(
                f"executable backlog drained; wait {sleep_for:.0f}s "
                f"(next BHS cycle in {max(0.0, idle_improvement_sec - (now - last_improvement_at)):.0f}s)"
            )
            time.sleep(sleep_for)
            continue

        _log(f"phase loop turn (remaining {remaining:.0f}s, pending={pending})")
        proc = _run(
            [
                sys.executable,
                "scripts/phase_development_loop.py",
                "--once",
                "--no-pause-gate",
            ],
            timeout=min(300, int(remaining) + 5),
        )
        turns += 1
        turn_record = {
            "turn": turns,
            "timestamp": _utc(),
            "exit_code": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-2500:],
            "stderr_tail": (proc.stderr or "")[-800:],
        }
        STATE_PATH.write_text(json.dumps(turn_record, indent=2) + "\n", encoding="utf-8")
        if proc.returncode != 0:
            _log(f"turn {turns} non-zero exit {proc.returncode}")
        if remaining < 45:
            break
        time.sleep(args.inter_turn_sec)

    _log("promote shim primitives (if stale)")
    promote_result = _promote_shim_if_needed()

    _log("full shim verification")
    verify = _run(["bash", "scripts/verify_shim_development.sh"], timeout=600)

    _log("BHS structural audit")
    bhs_smoke = _bhs_smoke_gate(args.bhs_target, include_ceiling=args.bhs_target >= 100)
    bhs = _bhs_audit()

    block = _run([sys.executable, "scripts/check_block_flag.py"], timeout=60)

    summary = {
        "finished_at": _utc(),
        "turns": turns,
        "bhs_target": args.bhs_target,
        "promote": promote_result,
        "verify_exit_code": verify.returncode,
        "verify_stdout_tail": (verify.stdout or "")[-3000:],
        "verify_stderr_tail": (verify.stderr or "")[-800:],
        "bhs_smoke": bhs_smoke,
        "bhs_audit": bhs,
        "block_flag_exit": block.returncode,
        "block_flag_stdout": (block.stdout or "")[-1500:],
        "goal_excerpt": goal_text[:1200],
        "worker_brief": worker_brief,
        "improvement_cycles": improvement_cycles,
        "improvement_cycle_count": len(improvement_cycles),
        "improvement_cycles_all_ok": (
            all(c.get("cycle_ok") for c in improvement_cycles) if improvement_cycles else None
        ),
    }
    out_path = ARTIFACT_DIR / f"summary_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _log(f"Wrote {out_path}")

    work_ok = (
        verify.returncode == 0
        and bhs.get("exit_code") == 0
        and bhs_smoke["floor"]["exit_code"] == 0
        and bhs_smoke["ceiling"]["exit_code"] == 0
    )
    if not work_ok:
        return 1
    if block.returncode != 0:
        _log("block flag still BLOCKED (expected until SHIM debts close)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
