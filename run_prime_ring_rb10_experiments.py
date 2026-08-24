"""Sequential, process-isolated runner for the bounded RB-10 experiments.

There is intentionally no implicit "run everything" mode.  Callers must name
each stage.  Every stage runs in a fresh child process after a physical-memory
headroom check.  The parent reactively samples the operating-system peak
working set, terminates an observed overage, and records it separately from the
experiment's modeled allocation estimate.  This monitor is not a hard kernel
allocation cap.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from prime_ring_rb10_contract import (
    ArtifactEnvelope,
    Deadline,
    ExperimentBudget,
    RB10ResourceError,
    RB10ValidationError,
    atomic_write_json,
    validate_artifact_envelope,
)


PROTOCOL_ID = "CHELATEDAI-PRW-RB10-SEQUENTIAL-RUNNER-v1"
MAX_RUNNER_RSS_BYTES = 256 * 1024 * 1024
MAX_RUNNER_SECONDS = 30.0
MINIMUM_AVAILABLE_MEMORY_BYTES = 1024 * 1024 * 1024
HEADROOM_MULTIPLIER = 4
POLL_SECONDS = 0.01
RSS_GUARD_MODE = "WINDOWS_PEAK_WORKING_SET_REACTIVE_POLL"
STAGE_FILENAMES = {
    "jo1": "rb10-prw-jo1-exact-p11-k2.json",
    "a1": "rb10-prw-a1-nondegeneracy-p11.json",
    "g2-p7": "rb10-prw-g2-algebraic-p7-n3.json",
}
STAGE_ENVELOPE_IDS = {
    "jo1": "PRW-JO1-P11-K2",
    "a1": "PRW-A1-NONDEGENERACY-P11",
    "g2-p7": "PRW-G2-ALGEBRA",
}
MANIFEST_FILENAME = "rb10-bounded-experiment-manifest.json"
STAGE_RECORD_FIELDS = {
    "stage",
    "status",
    "artifact_file",
    "artifact_digest",
    "artifact_bytes",
    "modeled_peak_bytes",
    "modeled_work_units",
    "modeled_peak_is_process_rss",
    "process_peak_rss_bytes",
    "process_peak_measurement_available",
    "available_memory_before_stage_bytes",
    "minimum_required_available_memory_bytes",
    "rss_cap_bytes",
    "rss_guard_mode",
    "timeout_seconds",
    "elapsed_seconds",
    "exit_code",
}


class BoundedRunnerError(RuntimeError):
    """Raised when the sequential runner refuses or aborts a stage."""


def _plain_stages(stages: object) -> Tuple[str, ...]:
    if type(stages) not in (tuple, list) or not stages:
        raise BoundedRunnerError("at least one explicit stage is required")
    checked = []
    for index, stage in enumerate(stages):
        if type(stage) is not str or stage not in STAGE_FILENAMES:
            raise BoundedRunnerError(
                "stage {} must be one of {}".format(
                    index,
                    tuple(STAGE_FILENAMES),
                )
            )
        checked.append(stage)
    if len(set(checked)) != len(checked):
        raise BoundedRunnerError("stages cannot be repeated")
    return tuple(checked)


def _runner_limits(
    max_rss_bytes: object,
    timeout_seconds: object,
) -> Tuple[int, float]:
    if type(max_rss_bytes) is not int or type(max_rss_bytes) is bool:
        raise BoundedRunnerError("max_rss_bytes must be a plain integer")
    if not 16 * 1024 * 1024 <= max_rss_bytes <= MAX_RUNNER_RSS_BYTES:
        raise BoundedRunnerError("max_rss_bytes must be between 16 MiB and 256 MiB")
    if type(timeout_seconds) not in (int, float):
        raise BoundedRunnerError("timeout_seconds must be a plain number")
    checked_timeout = float(timeout_seconds)
    if not 0.1 <= checked_timeout <= MAX_RUNNER_SECONDS:
        raise BoundedRunnerError("timeout_seconds must be between 0.1 and 30.0")
    return max_rss_bytes, checked_timeout


def _available_physical_memory() -> Optional[int]:
    if os.name != "nt":
        return None

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullAvailPhys)


def _windows_working_set(
    process: subprocess.Popen,
) -> Tuple[Optional[int], Optional[int]]:
    if os.name != "nt":
        return None, None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    handle = getattr(process, "_handle", None)
    if handle is None:
        return None, None
    succeeded = ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.c_void_p(int(handle)),
        ctypes.byref(counters),
        counters.cb,
    )
    if not succeeded:
        return None, None
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)


def _child_budget(
    max_rss_bytes: int,
    timeout_seconds: float,
) -> ExperimentBudget:
    return ExperimentBudget(
        max_estimated_bytes=max_rss_bytes,
        max_work_units=25_000_000,
        max_seconds=timeout_seconds,
        max_assignments=200_000,
        max_nodes=6,
        max_factors=8,
        max_factor_arity=3,
        max_output_bytes=1_048_576,
    )


def _execute_stage(
    stage: str,
    budget: ExperimentBudget,
) -> ArtifactEnvelope:
    if stage == "jo1":
        from prime_ring_joint_orbit_spectrum import (
            analyze_joint_orbit_spectrum,
            build_joint_orbit_artifact,
        )

        result = analyze_joint_orbit_spectrum(budget=budget)
        return build_joint_orbit_artifact(result, budget=budget)
    if stage == "a1":
        from prime_ring_action_nondegeneracy import (
            analyze_action_nondegeneracy,
            build_action_nondegeneracy_artifact,
        )

        result = analyze_action_nondegeneracy(budget=budget)
        return build_action_nondegeneracy_artifact(result, budget=budget)
    if stage == "g2-p7":
        from prime_ring_irreducible_factors import (
            run_g2_algebraic_screen,
        )

        result = run_g2_algebraic_screen(7, budget=budget)
        return result.artifact_envelope(budget)
    raise BoundedRunnerError("unknown child stage {!r}".format(stage))


def _child_main(
    stage: str,
    output_path: Path,
    max_rss_bytes: int,
    timeout_seconds: float,
) -> int:
    budget = _child_budget(max_rss_bytes, timeout_seconds)
    envelope = _execute_stage(stage, budget)
    atomic_write_json(
        output_path,
        envelope,
        max_encoded_bytes=budget.max_output_bytes,
        deadline=Deadline.start(budget),
    )
    return 0


def _read_validated_artifact(
    output_path: Path,
    maximum_bytes: int,
) -> Dict[str, object]:
    size = output_path.stat().st_size
    if size > maximum_bytes:
        raise BoundedRunnerError("artifact exceeds its output cap after child completion")
    with output_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not validate_artifact_envelope(payload):
        raise BoundedRunnerError("child artifact failed envelope validation")
    return payload


def _run_child_stage(
    stage: str,
    output_path: Path,
    max_rss_bytes: int,
    timeout_seconds: float,
) -> Dict[str, object]:
    available_before_stage = _available_physical_memory()
    minimum_available = max(
        MINIMUM_AVAILABLE_MEMORY_BYTES,
        HEADROOM_MULTIPLIER * max_rss_bytes,
    )
    if available_before_stage is None:
        raise BoundedRunnerError("OS working-set and available-memory measurements are required")
    if available_before_stage < minimum_available:
        raise BoundedRunnerError(
            "available physical memory is below the prelaunch floor: {} < {}".format(
                available_before_stage, minimum_available
            )
        )
    command = (
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-stage",
        stage,
        "--child-output",
        str(output_path),
        "--max-rss-bytes",
        str(max_rss_bytes),
        "--timeout-seconds",
        str(timeout_seconds),
    )
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=str(Path(__file__).resolve().parent),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
    )
    sampled_peak = 0
    measurement_available = False
    violation = None
    while process.poll() is None:
        current_rss, os_peak = _windows_working_set(process)
        if current_rss is not None and os_peak is not None:
            measurement_available = True
            sampled_peak = max(sampled_peak, current_rss, os_peak)
            if sampled_peak > max_rss_bytes:
                violation = "RSS_CAP_EXCEEDED"
                process.kill()
                break
        if time.monotonic() - started > timeout_seconds:
            violation = "TIMEOUT_EXCEEDED"
            process.kill()
            break
        time.sleep(POLL_SECONDS)

    current_rss, os_peak = _windows_working_set(process)
    if current_rss is not None and os_peak is not None:
        measurement_available = True
        sampled_peak = max(sampled_peak, current_rss, os_peak)
    stdout, stderr = process.communicate(timeout=5.0)
    elapsed = time.monotonic() - started
    if violation is not None:
        raise BoundedRunnerError(
            "{} aborted {} at {} bytes after {:.6f} seconds".format(
                stage,
                violation,
                sampled_peak,
                elapsed,
            )
        )
    if process.returncode != 0:
        error_text = stderr.decode("utf-8", errors="replace")[-4000:]
        raise BoundedRunnerError(
            "{} child failed with exit {}: {}".format(
                stage,
                process.returncode,
                error_text,
            )
        )
    if stdout:
        raise BoundedRunnerError("{} child emitted unexpected stdout".format(stage))
    if not measurement_available:
        raise BoundedRunnerError("{} completed without an OS peak-working-set measurement".format(stage))
    artifact = _read_validated_artifact(output_path, 1_048_576)
    estimate = artifact["resource_estimate"]
    return {
        "stage": stage,
        "status": "COMPLETE",
        "artifact_file": output_path.name,
        "artifact_digest": artifact["artifact_digest"],
        "artifact_bytes": output_path.stat().st_size,
        "modeled_peak_bytes": estimate["estimated_peak_bytes"],
        "modeled_work_units": estimate["estimated_work_units"],
        "modeled_peak_is_process_rss": estimate["estimate_is_process_rss"],
        "process_peak_rss_bytes": (sampled_peak if measurement_available else None),
        "process_peak_measurement_available": measurement_available,
        "available_memory_before_stage_bytes": available_before_stage,
        "minimum_required_available_memory_bytes": minimum_available,
        "rss_cap_bytes": max_rss_bytes,
        "rss_guard_mode": RSS_GUARD_MODE,
        "timeout_seconds": timeout_seconds,
        "elapsed_seconds": elapsed,
        "exit_code": process.returncode,
    }


def validate_run_manifest(
    payload: object,
    artifact_dir: object,
) -> bool:
    """Validate the manifest and rebind every record to its artifact file."""

    if type(payload) is not dict:
        raise RB10ValidationError("runner manifest must be a plain object")
    if type(artifact_dir) is not str and not isinstance(artifact_dir, Path):
        raise RB10ValidationError("artifact_dir must be a string or Path")
    directory = Path(artifact_dir).resolve()
    required = {
        "protocol_id",
        "status",
        "execution_order",
        "sequential_process_isolation",
        "available_memory_before_bytes",
        "available_memory_after_bytes",
        "minimum_required_available_memory_bytes",
        "rss_cap_bytes",
        "rss_guard_mode",
        "timeout_seconds_per_stage",
        "stages",
        "promotion_eligible",
        "novelty_claim",
        "limitations",
    }
    if set(payload) != required:
        raise RB10ValidationError("runner manifest fields do not match")
    if payload["protocol_id"] != PROTOCOL_ID:
        raise RB10ValidationError("runner manifest protocol mismatch")
    if payload["status"] != "COMPLETE":
        raise RB10ValidationError("runner manifest must be complete")
    if payload["sequential_process_isolation"] is not True:
        raise RB10ValidationError("runner manifest must record isolation")
    if payload["rss_guard_mode"] != RSS_GUARD_MODE:
        raise RB10ValidationError("runner RSS guard mode drifted")
    if payload["promotion_eligible"] is not False:
        raise RB10ValidationError("runner cannot establish promotion")
    if payload["novelty_claim"] is not False:
        raise RB10ValidationError("runner cannot establish novelty")
    order = payload["execution_order"]
    stages = payload["stages"]
    if type(order) is not list or type(stages) is not list:
        raise RB10ValidationError("runner stage fields must be lists")
    checked_order = _plain_stages(order)
    if len(stages) != len(checked_order):
        raise RB10ValidationError("runner stage count does not match order")
    if type(payload["rss_cap_bytes"]) is not int:
        raise RB10ValidationError("runner RSS cap must be a plain integer")
    if type(payload["timeout_seconds_per_stage"]) is not float:
        raise RB10ValidationError("runner timeout must be a plain float")
    _runner_limits(
        payload["rss_cap_bytes"],
        payload["timeout_seconds_per_stage"],
    )
    expected_minimum_available = max(
        MINIMUM_AVAILABLE_MEMORY_BYTES,
        HEADROOM_MULTIPLIER * payload["rss_cap_bytes"],
    )
    if payload["minimum_required_available_memory_bytes"] != expected_minimum_available:
        raise RB10ValidationError("runner memory headroom floor drifted")
    for field_name in (
        "available_memory_before_bytes",
        "available_memory_after_bytes",
    ):
        if type(payload[field_name]) is not int or payload[field_name] < 0:
            raise RB10ValidationError("{} must be a nonnegative plain integer".format(field_name))
    if payload["available_memory_before_bytes"] < expected_minimum_available:
        raise RB10ResourceError("runner did not satisfy the initial memory floor")
    if type(payload["limitations"]) is not list or not all(
        type(item) is str and item for item in payload["limitations"]
    ):
        raise RB10ValidationError("runner limitations must be strings")
    if any(type(stage) is not dict or set(stage) != STAGE_RECORD_FIELDS for stage in stages):
        raise RB10ValidationError("runner stage record schema drifted")
    if [stage["stage"] for stage in stages] != list(checked_order):
        raise RB10ValidationError("runner stage records do not match order")
    for stage in stages:
        if stage["status"] != "COMPLETE" or stage["exit_code"] != 0:
            raise RB10ValidationError("runner stage is not complete")
        expected_file = STAGE_FILENAMES[stage["stage"]]
        if stage["artifact_file"] != expected_file:
            raise RB10ValidationError("runner artifact filename drifted")
        if stage["rss_cap_bytes"] != payload["rss_cap_bytes"]:
            raise RB10ValidationError("runner stage RSS cap drifted")
        if stage["rss_guard_mode"] != RSS_GUARD_MODE:
            raise RB10ValidationError("runner stage RSS guard mode drifted")
        if stage["process_peak_measurement_available"] is not True:
            raise RB10ValidationError("runner stage lacks an OS peak measurement")
        if type(stage["process_peak_rss_bytes"]) is not int or stage["process_peak_rss_bytes"] <= 0:
            raise RB10ValidationError("runner stage peak RSS is invalid")
        if stage["process_peak_rss_bytes"] > stage["rss_cap_bytes"]:
            raise RB10ResourceError("runner stage exceeded RSS cap")
        if (
            type(stage["available_memory_before_stage_bytes"]) is not int
            or stage["available_memory_before_stage_bytes"] < expected_minimum_available
        ):
            raise RB10ResourceError("runner stage did not satisfy the prelaunch memory floor")
        if stage["minimum_required_available_memory_bytes"] != expected_minimum_available:
            raise RB10ValidationError("runner stage memory floor drifted")
        if (
            type(stage["elapsed_seconds"]) is not float
            or not math.isfinite(stage["elapsed_seconds"])
            or stage["elapsed_seconds"] < 0.0
        ):
            raise RB10ValidationError("runner elapsed time is invalid")
        if stage["timeout_seconds"] != payload["timeout_seconds_per_stage"]:
            raise RB10ValidationError("runner stage timeout drifted")

        artifact_path = directory / expected_file
        try:
            artifact = _read_validated_artifact(
                artifact_path,
                1_048_576,
            )
        except (BoundedRunnerError, OSError, ValueError) as exc:
            raise RB10ValidationError("runner stage artifact cannot be validated") from exc
        if artifact["status"] != "COMPLETE":
            raise RB10ValidationError("runner stage artifact is not complete")
        if artifact["stage_id"] != STAGE_ENVELOPE_IDS[stage["stage"]]:
            raise RB10ValidationError("runner envelope stage ID drifted")
        if artifact["artifact_digest"] != stage["artifact_digest"]:
            raise RB10ValidationError("runner artifact digest mismatch")
        if artifact_path.stat().st_size != stage["artifact_bytes"]:
            raise RB10ValidationError("runner artifact byte count mismatch")
        estimate = artifact["resource_estimate"]
        if estimate["estimated_peak_bytes"] != stage["modeled_peak_bytes"]:
            raise RB10ValidationError("runner modeled byte count mismatch")
        if estimate["estimated_work_units"] != stage["modeled_work_units"]:
            raise RB10ValidationError("runner modeled work count mismatch")
        if (
            estimate["estimate_is_process_rss"] != stage["modeled_peak_is_process_rss"]
            or stage["modeled_peak_is_process_rss"] is not False
        ):
            raise RB10ValidationError("runner modeled/RSS boundary drifted")
    return True


def run_stages(
    stages: object,
    output_dir: object,
    *,
    max_rss_bytes: object = MAX_RUNNER_RSS_BYTES,
    timeout_seconds: object = MAX_RUNNER_SECONDS,
) -> Dict[str, object]:
    checked_stages = _plain_stages(stages)
    checked_rss, checked_timeout = _runner_limits(
        max_rss_bytes,
        timeout_seconds,
    )
    if type(output_dir) is not str and not isinstance(output_dir, Path):
        raise BoundedRunnerError("output_dir must be a string or Path")
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / MANIFEST_FILENAME
    target_paths = [directory / STAGE_FILENAMES[stage] for stage in checked_stages]
    existing = [str(path) for path in [manifest_path] + target_paths if path.exists()]
    if existing:
        raise BoundedRunnerError("refusing to overwrite existing evidence: {}".format(existing))

    available_before = _available_physical_memory()
    minimum_available = max(
        MINIMUM_AVAILABLE_MEMORY_BYTES,
        HEADROOM_MULTIPLIER * checked_rss,
    )
    if available_before is None:
        raise BoundedRunnerError("this runner requires Windows available-memory measurement")
    if available_before < minimum_available:
        raise BoundedRunnerError(
            "available physical memory is below the run floor: {} < {}".format(available_before, minimum_available)
        )
    records = []
    active_stage = None
    try:
        for stage, output_path in zip(checked_stages, target_paths):
            active_stage = stage
            records.append(
                _run_child_stage(
                    stage,
                    output_path,
                    checked_rss,
                    checked_timeout,
                )
            )
    except Exception as exc:
        failure_manifest = {
            "protocol_id": PROTOCOL_ID,
            "status": "FAILED",
            "execution_order": list(checked_stages),
            "completed_stages": records,
            "failed_stage": active_stage,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc)[:1000],
            "promotion_eligible": False,
            "novelty_claim": False,
            "recovery": (
                "Completed artifacts are retained with this failure manifest. Use a fresh output directory for a retry."
            ),
        }
        atomic_write_json(
            manifest_path,
            failure_manifest,
            max_encoded_bytes=1_048_576,
        )
        raise
    available_after = _available_physical_memory()
    if available_after is None:
        raise BoundedRunnerError("available-memory measurement disappeared after execution")
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "execution_order": list(checked_stages),
        "sequential_process_isolation": True,
        "available_memory_before_bytes": available_before,
        "available_memory_after_bytes": available_after,
        "minimum_required_available_memory_bytes": minimum_available,
        "rss_cap_bytes": checked_rss,
        "rss_guard_mode": RSS_GUARD_MODE,
        "timeout_seconds_per_stage": checked_timeout,
        "stages": records,
        "promotion_eligible": False,
        "novelty_claim": False,
        "limitations": [
            "Process peak working set is an OS measurement, not the modeled allocation estimate.",
            "The RSS guard is reactive sampling, not a hard OS allocation cap.",
            "A prelaunch physical-memory floor reduces risk but cannot guarantee that an arbitrary child will never allocate abruptly.",
            "The stages ran sequentially, so their peak working sets are not additive.",
            "These exact synthetic method-development cells do not establish utility, scale, promotion, or novelty.",
        ],
    }
    validate_run_manifest(manifest, directory)
    atomic_write_json(
        manifest_path,
        manifest,
        max_encoded_bytes=1_048_576,
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=tuple(STAGE_FILENAMES))
    parser.add_argument("--output-dir")
    parser.add_argument("--max-rss-bytes", type=int, default=MAX_RUNNER_RSS_BYTES)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=MAX_RUNNER_SECONDS,
    )
    parser.add_argument(
        "--child-stage",
        choices=tuple(STAGE_FILENAMES),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--child-output", help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.child_stage is not None:
        if arguments.child_output is None or arguments.stages is not None:
            raise BoundedRunnerError("invalid child invocation")
        return _child_main(
            arguments.child_stage,
            Path(arguments.child_output),
            *_runner_limits(
                arguments.max_rss_bytes,
                arguments.timeout_seconds,
            ),
        )
    if arguments.stages is None or arguments.output_dir is None:
        raise BoundedRunnerError("parent invocation requires --stages and --output-dir")
    run_stages(
        arguments.stages,
        arguments.output_dir,
        max_rss_bytes=arguments.max_rss_bytes,
        timeout_seconds=arguments.timeout_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
