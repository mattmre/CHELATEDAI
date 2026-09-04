#!/usr/bin/env python3
"""
Long-Running Sustained Phase Orchestrator Stub (v0.2 — 2026-05-27, zero-wall auto-chain update)
Purpose: Provides a true "run for hours like /goal" launcher for the new sustained round model with
**zero (or minimal) wall time** between rounds. After one round completes (synthesis + gates + report),
it immediately begins the next (auto-continue) unless the PAUSE gate is active.

This is the implementation of the user request: "no wall time and you continue working. After you're
done with whatever phase or your turn is complete, automatically start the next loop and begin again.
Iterate and improve the process." + 10min wall as safety/recovery (the 1h scheduler is replaced by 10min).

Usage (zero-wall persistent run):
  nohup python docs/steering_chelation_rag_dag_research/artifacts/long_running_orchestrator_stub.py \
    --phase-plan docs/steering_chelation_rag_dag_research/FULL_SHIM_LOOP_PHASE_PLAN.md \
    --driver docs/steering_chelation_rag_dag_research/artifacts/SUSTAINED_PHASE_ROUND_DRIVER.md \
    --auto-continue --max-wall-min 10 > sustained_orchestrator.log 2>&1 &

The stub will:
- Re-verify BLOCKED / 0-prod / OVERRIDE (OPERATOR_OVERRIDE.md) at the start of every round and before any auto-chain.
- If OVERRIDE: NONE and §128 conditions (11+ cycles 0 substrate + BLOCKED:2 + SHIM-CD-01 OPEN): emit a fresh gate report artifact and sleep the safety interval (10min) instead of dispatching a new round. No silent iteration.
- Drive full 10-agent waves when the gate is clear (wiring for spawn_subagent etc. still required in run_one...).
- Log wall time (productive vs idle) and append a "Process Improvement Note" after each round for self-iteration.
- The 10min scheduler (recovery backstop) will re-awaken if this process dies.
- Respect the research guard and never touch prod paths.

BHS note: This stub itself is L4 (skeleton + gate logic). All round output must still carry full
"0 substrate / does not satisfy goal success def #1 while BLOCKED + SHIM-CD-01" language.
See updated SUSTAINED_PHASE_ROUND_DRIVER.md "Zero-Wall Auto-Chain Mode" section.

Run at your own risk. Monitor the log. Kill with pkill when done.
"""

import argparse
import time
import json
from datetime import datetime, timezone
from pathlib import Path

def log(msg):
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] {msg}", flush=True)

def check_block_and_prod():
    """Live gate check. In real deployment this calls the actual scripts and greps."""
    import subprocess
    try:
        result = subprocess.run(
            ["python", "scripts/check_block_flag.py"],
            cwd=Path(__file__).parent.parent.parent.parent,  # adjust to CHELATEDAI root
            capture_output=True, text=True, timeout=10
        )
        blocked = "RESULT: FAIL" in result.stdout or "BLOCKED" in result.stdout
        debt = 2 if blocked else 0
    except Exception:
        blocked, debt = True, 2
    prod_ok = True  # placeholder; real impl does the exact 0-prod grep for "exactly 2 research files"
    return {"blocked": blocked, "debt_rows": debt, "prod_ok": prod_ok}

def check_override_and_pause():
    """Returns (override_active: bool, reason: str). Must be called before every auto-chain."""
    override_file = Path(__file__).parent / "OPERATOR_OVERRIDE.md"
    try:
        content = override_file.read_text()
        if "OVERRIDE: ACTIVE" in content:
            return True, "OVERRIDE ACTIVE per operator file"
    except Exception:
        pass
    return False, "OVERRIDE: NONE (or file unreadable) + §128 conditions likely active"

def run_one_sustained_round(round_num: int, phase_plan_path: Path, driver_path: Path, timebox_min: int,
                               auto_continue: bool, max_wall_min: int) -> bool:
    round_start = time.time()
    log(f"=== STARTING SUSTAINED ROUND {round_num} (timebox ~{timebox_min} min, auto_continue={auto_continue}) ===")

    # === PAUSE / OVERRIDE GATE (mandatory before any work or auto-chain) ===
    override_active, override_reason = check_override_and_pause()
    block_state = check_block_and_prod()
    log(f"Gate at round start: override_active={override_active} ({override_reason}), block={block_state}")

    if not override_active and (block_state.get("blocked") or block_state.get("debt_rows", 0) > 0):
        # Emit gate report and sleep safety interval instead of dispatching
        gate_report = {
            "type": "pause_gate",
            "round_attempt": round_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "override": "NONE",
            "block_state": block_state,
            "note": "0 substrate / does not satisfy goal success def #1 while BLOCKED + SHIM-CD-01. PAUSE gate enforced per DRIVER:24 + PROTOCOL §8. No agents dispatched.",
            "recommendation": "Edit OPERATOR_OVERRIDE.md to OVERRIDE: ACTIVE with reason + sign-off, or kill scheduler and scope-reduce."
        }
        gate_path = Path("loop_02") / f"auto_gate_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}_round{round_num:02d}.md"
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        gate_path.write_text(json.dumps(gate_report, indent=2))
        log(f"PAUSE GATE EMITTED: {gate_path}. Sleeping safety interval ({max_wall_min}min) — no new round.")
        time.sleep(max_wall_min * 60)
        return True  # "success" from gate perspective; loop continues to re-check later

    # === Real round would go here once gate is clear ===
    log("GATE CLEAR (OVERRIDE ACTIVE or debts resolved). Proceeding with round body.")
    log("ROUND BODY: (stub) — wire real 10-agent dispatch + protocol logic (spawn_subagent etc.) here.")
    log("Simulating productive work for demo purposes... (replace with actual agent spawns + full BHS artifacts)")
    time.sleep(5)  # placeholder

    wall_seconds = time.time() - round_start
    artifact = {
        "round": round_num,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "block_state": block_state,
        "override_active": override_active,
        "wall_seconds": round(wall_seconds, 1),
        "note": "0 substrate on goal #1 (SHIM-CD-01 + BLOCKED active). Full 10-agent fidelity target for this model.",
        "driver": str(driver_path),
        "phase_plan": str(phase_plan_path),
        "process_improvement_note": {
            "measured_wall_sec": round(wall_seconds, 1),
            "suggestion": "When OVERRIDE ACTIVE, reduce any remaining safety sleep to <5s for true zero-wall chaining. Current fidelity still limited by research guard + 0 real SIPs. Next iteration should wire actual spawn_subagent for A-J.",
            "l9_risk": "Doc volume while #1 0% remains a carried L9 per plan:83/85."
        }
    }
    out_path = Path("loop_02") / f"sustained_round_{round_num:02d}_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    log(f"Round {round_num} artifact + improvement note written: {out_path}")
    log(f"=== ROUND {round_num} COMPLETE in {round(wall_seconds,1)}s (stub) ===")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-plan", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--max-rounds", type=int, default=20, help="Safety cap; set high for hours-long runs")
    parser.add_argument("--round-timebox-min", type=int, default=45)
    parser.add_argument("--auto-continue", action="store_true", default=True,
                        help="Zero-wall mode: immediately start next round after this one completes (the requested behavior)")
    parser.add_argument("--max-wall-min", type=int, default=10,
                        help="Safety interval (minutes) to sleep when PAUSE gate is active or as recovery heartbeat. 10min per user request.")
    parser.add_argument("--sleep-between-rounds-sec", type=int, default=5,
                        help="Only used when --no-auto-continue. Kept tiny for the new model.")
    args = parser.parse_args()

    log("Long-Running Sustained Phase Orchestrator starting (stub v0.2 zero-wall auto-chain)")
    log(f"Phase Plan: {args.phase_plan}")
    log(f"Driver:     {args.driver}")
    log(f"auto-continue={args.auto_continue}, max_wall_min={args.max_wall_min}")
    log("10min scheduler (or this process) is recovery backstop. Kill externally when desired.")

    r = 1
    while r <= args.max_rounds:
        success = run_one_sustained_round(
            r, args.phase_plan, args.driver, args.round_timebox_min,
            auto_continue=args.auto_continue, max_wall_min=args.max_wall_min
        )
        if not success:
            log("Round failed or operator intervention requested — stopping.")
            break

        if not args.auto_continue:
            log(f"Sleeping {args.sleep_between_rounds_sec}s before next (non-auto mode)...")
            time.sleep(args.sleep_between_rounds_sec)
        else:
            # Zero-wall: immediately continue to next iteration (the requested "automatically start the next loop")
            log("auto-continue enabled — immediately planning next round (near-zero wall).")
            # Tiny sleep only to allow logs to flush / prevent tight CPU spin if something is wrong
            time.sleep(0.5)

        r += 1

    log("Orchestrator exiting (max-rounds reached or stopped).")

    log("Orchestrator exiting. All rounds completed or stopped.")

if __name__ == "__main__":
    main()