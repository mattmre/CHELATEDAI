"""Run the focused overlay and Model-Scope validation bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("experiment_runs") / "overlay-model-scope-validation" / "latest"

VALIDATION_COMMANDS = (
    (
        "overlay_model_scope_tests",
        [
            sys.executable,
            "-m",
            "unittest",
            "test_adaptive_overlay.py",
            "test_run_model_scope_campaign.py",
            "test_model_scope_overlay_smoke.py",
            "test_promotion_contract.py",
            "test_attnres_repeat_seed_decision.py",
            "-v",
        ],
    ),
    (
        "model_scope_overlay_smoke",
        [
            sys.executable,
            "run_model_scope_overlay_smoke.py",
            "--output-dir",
            str(DEFAULT_OUTPUT_DIR / "smoke"),
        ],
    ),
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _run_command(name: str, command: list[str], *, cwd: Path, timeout_seconds: int) -> dict[str, Any]:
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        return {
            "name": name,
            "command": command,
            "returncode": int(completed.returncode),
            "duration_seconds": round(time.time() - started, 3),
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "command": command,
            "returncode": 124,
            "duration_seconds": round(time.time() - started, 3),
            "stdout_tail": (exc.stdout or "")[-4000:],
            "stderr_tail": (exc.stderr or "")[-4000:],
            "timed_out": True,
        }


def run_validation_bundle(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    timeout_seconds: int = 300,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    """Run focused tests and smoke validation, then write a replayable summary."""

    root = Path(cwd or Path.cwd()).resolve()
    resolved_output_dir = Path(output_dir)
    if not resolved_output_dir.is_absolute():
        resolved_output_dir = root / resolved_output_dir
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    commands = []
    for name, command in VALIDATION_COMMANDS:
        resolved_command = list(command)
        if name == "model_scope_overlay_smoke":
            resolved_command[-1] = str(resolved_output_dir / "smoke")
        commands.append((name, resolved_command))

    results = [_run_command(name, command, cwd=root, timeout_seconds=timeout_seconds) for name, command in commands]
    passed = all(result["returncode"] == 0 for result in results)
    summary = {
        "record_type": "overlay_model_scope_validation_bundle",
        "output_dir": str(resolved_output_dir),
        "passed": bool(passed),
        "command_count": len(results),
        "failed_commands": [result["name"] for result in results if result["returncode"] != 0],
        "results": results,
    }
    summary_path = resolved_output_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run focused overlay and Model-Scope validation bundle")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for validation outputs")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="Timeout per validation command")
    args = parser.parse_args()
    summary = run_validation_bundle(output_dir=args.output_dir, timeout_seconds=args.timeout_seconds)
    print(json.dumps(_json_safe(summary), indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
