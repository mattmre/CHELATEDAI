#!/usr/bin/env python3
"""Record external scheduler verification evidence for SHIM-CD-06.

SHIM-CD-06 requires the external recurring 5-minute scheduler task to be
explicitly verified against the expected worker profile.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_SCHEDULER_ID = "019e669bf1bb"
DEFAULT_EXPECTED_AGENTS = 5
DEFAULT_CMD = os.environ.get("CHELATED_SHIM_SCHEDULER_CMD", "scheduler_list")
DEFAULT_EVIDENCE_DIR = os.environ.get(
    "CHELATED_SHIM_EVIDENCE_DIR", str((ROOT / "artifacts").resolve())
)
FALLBACK_CMDS = (
    DEFAULT_CMD,
    "systemctl list-timers --all",
    "crontab -l",
    "atq",
    "systemctl --user list-timers --all",
)
DEFAULT_TIMEOUT = 10


AGENT_COUNT_RE = re.compile(
    r"(?i)(?:agents?|workers?|parallelism|concurrency|slots?)\s*(?:count)?\s*[:=]\s*(\d+)"
)
EXACTLY_RE = re.compile(r"(?i)exactly\s+(\d+)\b")
ID_RE = re.compile(r"\b([0-9a-f]{10,})\b")


@dataclass
class ProbeResult:
    found: bool
    verified: bool
    details: Dict[str, Any]
    notes: list[str]


FALSE_VALUES = {"", "0", "false", "False", "FALSE", "off", "OFF", "no", "No", "NO"}


def _read_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        return f"file-read-failure: {exc!r}"


def _is_truthy(value: Optional[str]) -> bool:
    if not value:
        return False
    return value.strip() not in FALSE_VALUES


def _collect_numeric_fields(payload: Any) -> list[int]:
    values: list[int] = []
    if isinstance(payload, dict):
        for key in ("agents", "agent_count", "parallelism", "worker_count", "size"):
            raw = payload.get(key)
            if isinstance(raw, int):
                values.append(raw)
            elif isinstance(raw, str) and raw.isdigit():
                values.append(int(raw))
        # nested objects like {"config": {"agents": 5}}
        for raw in payload.values():
            values.extend(_collect_numeric_fields(raw))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(_collect_numeric_fields(item))
    return values


def _extract_from_text(text: str, scheduler_id: str) -> tuple[bool, Optional[int], str]:
    if scheduler_id not in text:
        return False, None, ""

    target_lines: list[str] = []
    for line in text.splitlines():
        if scheduler_id in line:
            target_lines.append(line)
    if not target_lines:
        return False, None, ""

    for line in target_lines:
        match = AGENT_COUNT_RE.search(line)
        if match:
            return True, int(match.group(1)), line.strip()
        match = EXACTLY_RE.search(line)
        if match:
            return True, int(match.group(1)), line.strip()

    # Sometimes the scheduler table format puts the id and count apart.
    for line in target_lines:
        for m in ID_RE.finditer(line):
            if m.group(1) == scheduler_id:
                return True, None, line.strip()

    return True, None, target_lines[0].strip()


def _split_command(cmd: str) -> list[str]:
    command = shlex.split(cmd)
    if not command:
        return []
    executable = command[0]
    if shutil.which(executable):
        return command
    if os.path.isabs(executable) and os.access(executable, os.X_OK):
        return command
    return []


def _probe_scheduler_cli(cmd: str) -> dict[str, Any]:
    command = _split_command(cmd)
    if not command:
        return {
            "source": "cli",
            "command": cmd,
            "executed": False,
            "exit_code": 127,
            "stderr": f"command not found: {cmd}",
            "stdout": "",
        }

    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=DEFAULT_TIMEOUT,
    )
    return {
        "source": "cli",
        "command": cmd,
        "executed": True,
        "exit_code": proc.returncode,
        "stderr": proc.stderr or "",
        "stdout": proc.stdout or "",
    }


def _probe_from_file(path: str) -> dict[str, Any]:
    content = _read_file(path)
    return {
        "source": "file",
        "command": path,
        "executed": True,
        "exit_code": 0 if content is not None else 1,
        "stdout": "" if content is None else content,
        "stderr": "" if isinstance(content, str) else str(content),
    }


def verify_scheduler(
    scheduler_id: str,
    expected_agent_count: int,
) -> ProbeResult:
    probe: dict[str, Any] = {
        "source": "cli|file",
        "commands": [],
        "executed": False,
        "exit_code": 1,
        "stdout": "",
        "stderr": "",
        "attempts": [],
    }
    source = "cli"
    if os.environ.get("CHELATED_SHIM_SCHEDULER_FIXTURE"):
        source = "file"
        probe = _probe_from_file(os.environ["CHELATED_SHIM_SCHEDULER_FIXTURE"])
    else:
        cmds = []
        for candidate in FALLBACK_CMDS:
            if not candidate:
                continue
            if candidate in cmds:
                continue
            cmds.append(candidate)
        for cmd in cmds:
            attempt = _probe_scheduler_cli(cmd)
            probe["commands"].append(cmd)
            probe["attempts"].append(attempt)
            probe.update(attempt)
            if attempt["executed"] and attempt["exit_code"] == 0:
                break
        if not probe["executed"]:
            # none of the candidates are available in this runtime.
            probe["commands"] = cmds
            probe["source"] = "cli"

    # Track whether this run can be considered deterministic.
    deterministic = bool(
        source == "file"
        or bool(probe.get("executed"))
    )

    stdout = probe.get("stdout", "") or ""
    parsed_json = None
    try:
        parsed_json = json.loads(stdout)
    except json.JSONDecodeError:
        parsed_json = None

    found = False
    discovered_agents: Optional[int] = None
    matching_line = ""
    evidence_chunks: list[str] = []

    if parsed_json is not None:
        for scheduler in _walk_objects(parsed_json):
            sid = str(
                scheduler.get("id")
                or scheduler.get("scheduler_id")
                or scheduler.get("task_id")
                or ""
            )
            if sid != scheduler_id:
                continue
            found = True
            nums = _collect_numeric_fields(scheduler)
            if nums:
                discovered_agents = max(nums)
            evidence_chunks.append(str(scheduler))
    else:
        found, discovered_agents, matching_line = _extract_from_text(stdout, scheduler_id)
        if matching_line:
            evidence_chunks.append(matching_line)

    notes: list[str] = []
    if not deterministic and not os.environ.get("CHELATED_SHIM_SCHEDULER_FIXTURE"):
        notes.append(
            "non-deterministic: scheduler verification commands unavailable in this runtime"
        )
        return ProbeResult(
            found=False,
            verified=False,
            details={"probe": probe, "deterministic": False},
            notes=notes,
        )
    if not found:
        notes.append("scheduler id not present in probe output")
        return ProbeResult(
            found=False,
            verified=False,
            details={
                "probe": probe,
                "deterministic": deterministic,
                "parsed_as_json": parsed_json is not None,
                "scheduler_id": scheduler_id,
            },
            notes=notes,
        )
    if discovered_agents is None:
        notes.append("scheduler id found but agent count not detectable in probe")
        return ProbeResult(
            found=True,
            verified=False,
            details={
                "probe": probe,
                "deterministic": deterministic,
                "scheduler_id": scheduler_id,
                "evidence_chunks": evidence_chunks,
            },
            notes=notes,
        )
    if discovered_agents < expected_agent_count:
        notes.append(
            f"scheduler {scheduler_id} shows {discovered_agents} agents, "
            f"below expected {expected_agent_count}"
        )
        return ProbeResult(
            found=True,
            verified=False,
            details={
                "probe": probe,
                "deterministic": deterministic,
                "scheduler_id": scheduler_id,
                "discovered_agent_count": discovered_agents,
                "expected_agent_count": expected_agent_count,
                "evidence_chunks": evidence_chunks,
            },
            notes=notes,
        )

    notes.append(
        f"scheduler {scheduler_id} found with at least {discovered_agents} agents"
    )
    return ProbeResult(
        found=True,
        verified=True,
        details={
            "probe": probe,
            "deterministic": deterministic,
            "scheduler_id": scheduler_id,
            "discovered_agent_count": discovered_agents,
            "expected_agent_count": expected_agent_count,
            "evidence_chunks": evidence_chunks,
        },
        notes=notes,
    )


def _walk_objects(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        out: list[Any] = [payload]
        for item in payload.values():
            out.extend(_walk_objects(item))
        return out
    if isinstance(payload, list):
        out: list[Any] = []
        for item in payload:
            out.extend(_walk_objects(item))
        return out
    if isinstance(payload, tuple):
        out: list[Any] = []
        for item in payload:
            out.extend(_walk_objects(item))
        return out
    return []


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect scheduler evidence for SHIM-CD-06",
    )
    parser.add_argument(
        "--scheduler-id",
        default=DEFAULT_SCHEDULER_ID,
        help="scheduler id to verify (default: env/SHIM constant)",
    )
    parser.add_argument(
        "--expected-agents",
        type=int,
        default=DEFAULT_EXPECTED_AGENTS,
        help="expected minimum worker count for this scheduler",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat missing / unverified scheduler profile as hard failure.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    result = verify_scheduler(args.scheduler_id, args.expected_agents)
    strict = args.strict or _is_truthy(
        os.environ.get("CHELATED_SHIM_REQUIRE_SCHEDULER_VERIFICATION")
    ) or _is_truthy(os.environ.get("CHELATED_SHIM_SCHEDULER_REQUIRE"))

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scheduler_id": args.scheduler_id,
        "expected_agent_count": args.expected_agents,
        "found": result.found,
        "verified": result.verified,
        "strict": strict,
        "notes": result.notes,
        "details": result.details,
        "env": {
            "CHELATED_SHIM_SCHEDULER_CMD": os.environ.get("CHELATED_SHIM_SCHEDULER_CMD"),
            "CHELATED_SHIM_SCHEDULER_FIXTURE": os.environ.get(
                "CHELATED_SHIM_SCHEDULER_FIXTURE"
            ),
            "CHELATED_SHIM_EVIDENCE_DIR": os.environ.get(
                "CHELATED_SHIM_EVIDENCE_DIR"
            ),
        },
        "block_flag_note": (
            "Run scripts/check_block_flag.py separately; this script does not flip BLOCKED."
        ),
    }

    out_dir = Path(
        os.environ.get(
            "CHELATED_SHIM_EVIDENCE_DIR",
            str(Path(DEFAULT_EVIDENCE_DIR)),
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        "bhs_shim_scheduler_evidence_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    )
    payload = json.dumps(out, indent=2)
    out_path.write_text(payload + "\n", encoding="utf-8")

    # Keep output compact but still human-readable.
    print(f"Wrote {out_path}")
    print(payload)

    if strict:
        return 0 if result.verified else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
