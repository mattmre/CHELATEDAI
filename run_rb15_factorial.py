"""Atomic artifact runner for the preregistered RB-15 2 x 2 factorial."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import rb15_factorial_experiments as factorial

from rb15_factorial_experiments import (
    NOVELTY_CLAIM_STATUS,
    PROTOCOL_ID,
    RUN_IDS,
    SCIENTIFIC_CLAIM_STATUS,
    run_factorial,
)


FACTORIAL_FILENAME = "factorial.json"
MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA = "rb15-factorial-manifest-v1"
ALLOWED_STATUSES = (
    "FACTORIAL_SURVIVES_FROZEN_SYNTHETIC_GATES",
    "FACTORIAL_KILLED_ON_FROZEN_SYNTHETIC_GATES",
)
EXPECTED_ARTIFACT_KEYS = {
    "protocol_id",
    "run_id",
    "status",
    "evidence_mode",
    "scientific_claim_status",
    "novelty_claim_status",
    "production_path_changed",
    "model_or_corpus_loaded",
    "phase_and_continuation_policy",
    "fixture",
    "design_preflight",
    "branches",
    "endpoints",
    "gates",
    "failure_count",
    "failures",
    "resource_usage",
}


class FactorialArtifactError(ValueError):
    """Raised when an RB-15 factorial artifact violates its contract."""


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _checked_output_directory(output_directory: Path, *, create: bool) -> Path:
    directory = Path(output_directory)
    if directory.exists() and directory.is_symlink():
        raise FactorialArtifactError("output directory must not be a symlink")
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    if not directory.is_dir():
        raise FactorialArtifactError("output directory is not a directory")
    return directory


def _checked_member(directory: Path, member: object) -> Path:
    if not isinstance(member, str) or not member:
        raise FactorialArtifactError("artifact member must be a nonempty string")
    relative = Path(member)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name != member:
        raise FactorialArtifactError("artifact member must be one relative filename")
    path = directory / relative
    if path.is_symlink() or path.parent.resolve() != directory.resolve():
        raise FactorialArtifactError("artifact member is unsafe")
    return path


def _load_canonical_mapping(path: Path, label: str) -> Tuple[Dict[str, object], bytes]:
    if not path.is_file():
        raise FactorialArtifactError(f"missing {label}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FactorialArtifactError(f"invalid {label} JSON") from exc
    if not isinstance(payload, dict) or raw != _canonical_json_bytes(payload):
        raise FactorialArtifactError(f"{label} must be a canonical JSON object")
    return payload, raw


def _resource_guards(payload: Mapping[str, object]) -> Dict[str, object]:
    usage = payload.get("resource_usage")
    if not isinstance(usage, dict):
        raise FactorialArtifactError("factorial payload is missing resource_usage")
    fields = {
        "deadline_enforced_during_integration": usage.get(
            "deadline_enforced_during_integration"
        ),
        "deadline_guard_passed": usage.get("deadline_guard_passed"),
        "rss_enforced_during_integration": usage.get("rss_enforced_during_integration"),
        "rss_guard_passed": usage.get("rss_guard_passed"),
    }
    if any(type(value) is not bool for value in fields.values()):
        raise FactorialArtifactError("factorial payload has invalid resource guards")
    return fields


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _validate_harmonic(value: object, label: str) -> None:
    expected = {
        "amplitude",
        "offset",
        "sin_coefficient",
        "cos_coefficient",
        "residual_rms",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise FactorialArtifactError(f"{label} harmonic schema is invalid")
    if any(not _finite_number(item) for item in value.values()):
        raise FactorialArtifactError(f"{label} harmonic contains nonfinite data")
    if value["amplitude"] < 0 or value["residual_rms"] < 0:
        raise FactorialArtifactError(f"{label} harmonic has a negative invariant")
    expected_amplitude = math.hypot(
        value["sin_coefficient"], value["cos_coefficient"]
    )
    if not math.isclose(
        value["amplitude"], expected_amplitude, rel_tol=1.0e-12, abs_tol=1.0e-15
    ):
        raise FactorialArtifactError(f"{label} harmonic amplitude is inconsistent")


def _validate_response(value: object, label: str) -> None:
    expected = {
        "fundamental",
        "third_harmonic",
        "previous_half_fundamental_amplitude",
        "final_half_fundamental_amplitude",
        "relative_settling_delta",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise FactorialArtifactError(f"{label} response schema is invalid")
    _validate_harmonic(value["fundamental"], f"{label} fundamental")
    _validate_harmonic(value["third_harmonic"], f"{label} third")
    for field in (
        "previous_half_fundamental_amplitude",
        "final_half_fundamental_amplitude",
        "relative_settling_delta",
    ):
        if not _finite_number(value[field]) or value[field] < 0:
            raise FactorialArtifactError(f"{label} response has invalid {field}")
    expected_settling = abs(
        value["final_half_fundamental_amplitude"]
        - value["previous_half_fundamental_amplitude"]
    ) / max(
        value["final_half_fundamental_amplitude"], factorial.DENOMINATOR_FLOOR
    )
    if not math.isclose(
        value["relative_settling_delta"],
        expected_settling,
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        raise FactorialArtifactError(f"{label} settling delta is inconsistent")


def _validate_fixture(artifact: Mapping[str, object]) -> None:
    fixture = artifact.get("fixture")
    expected = {
        "randomness": "none",
        "fixed_run_ids": list(factorial.RUN_IDS),
        "frequencies": [float(value) for value in factorial.FREQUENCIES],
        "forcing_amplitude": factorial.FORCING_AMPLITUDE,
        "forcing_node": factorial.FORCING_NODE,
        "measurement_node": factorial.MEASUREMENT_NODE,
        "periods_per_cell": factorial.TOTAL_PERIODS,
        "measured_final_periods": factorial.MEASURE_PERIODS,
        "steps_per_period": factorial.STEPS_PER_PERIOD,
        "resource_check_interval_steps": factorial.RESOURCE_CHECK_INTERVAL_STEPS,
        "factor_configurations": list(factorial.FACTOR_CONFIGURATIONS),
        "report_only_control_configuration": factorial.CONTROL_CONFIGURATION,
    }
    if fixture != expected:
        raise FactorialArtifactError("artifact fixture does not match frozen protocol")
    design = artifact.get("design_preflight")
    if design != factorial.resource_preflight():
        raise FactorialArtifactError("artifact design preflight is not the frozen design")


def _validate_branches(artifact: Mapping[str, object]) -> bool:
    branches = artifact.get("branches")
    if not isinstance(branches, dict) or set(branches) != set(
        factorial.ALL_CONFIGURATIONS
    ):
        raise FactorialArtifactError("artifact branches have wrong configurations")
    complete = True
    expected_steps = factorial.TOTAL_PERIODS * factorial.STEPS_PER_PERIOD
    systems = factorial._configurations()
    finite_keys = {
        "frequency",
        "forcing_amplitude",
        "forcing_node",
        "measurement_node",
        "finite_trajectory",
        "failure",
        "steps_completed",
        "rhs_evaluations",
        "response",
        "endpoint_fundamental_transfer_gain",
        "endpoint_third_harmonic_transfer_gain",
        "attachment_local_mismatch_responses",
        "final_state",
        "configuration",
        "direction",
    }
    nonfinite_keys = {
        "frequency",
        "finite_trajectory",
        "failure",
        "steps_completed",
        "rhs_evaluations",
        "configuration",
        "direction",
    }
    for configuration in factorial.ALL_CONFIGURATIONS:
        directions = branches[configuration]
        if not isinstance(directions, dict) or not set(directions).issubset(
            {"forward", "reverse"}
        ):
            raise FactorialArtifactError("artifact branch directions are invalid")
        for direction in ("forward", "reverse"):
            cells = directions.get(direction, [])
            if not isinstance(cells, list) or len(cells) > len(factorial.FREQUENCIES):
                raise FactorialArtifactError("artifact branch cell count is invalid")
            sweep_has_nonfinite_state = False
            expected_frequencies = (
                factorial.FREQUENCIES
                if direction == "forward"
                else factorial.FREQUENCIES[::-1]
            )
            for index, cell in enumerate(cells):
                if not isinstance(cell, dict):
                    raise FactorialArtifactError("artifact cell must be an object")
                if cell.get("configuration") != configuration:
                    raise FactorialArtifactError("artifact cell configuration mismatch")
                if cell.get("direction") != direction:
                    raise FactorialArtifactError("artifact cell direction mismatch")
                if cell.get("frequency") != float(expected_frequencies[index]):
                    raise FactorialArtifactError("artifact cell frequency/order mismatch")
                steps = cell.get("steps_completed")
                evaluations = cell.get("rhs_evaluations")
                if type(steps) is not int or not 1 <= steps <= expected_steps:
                    raise FactorialArtifactError("artifact cell step count is invalid")
                if type(evaluations) is not int or evaluations != 4 * steps:
                    raise FactorialArtifactError("artifact cell RHS count is invalid")
                if type(cell.get("finite_trajectory")) is not bool:
                    raise FactorialArtifactError("artifact cell finite flag is invalid")
                if cell["finite_trajectory"]:
                    if sweep_has_nonfinite_state:
                        raise FactorialArtifactError(
                            "finite cell follows nonfinite continuation state"
                        )
                    if set(cell) != finite_keys:
                        raise FactorialArtifactError("finite cell schema is invalid")
                    if steps != expected_steps or cell.get("failure") is not None:
                        raise FactorialArtifactError("finite cell is incomplete or failed")
                    if (
                        cell["forcing_amplitude"] != factorial.FORCING_AMPLITUDE
                        or cell["forcing_node"] != factorial.FORCING_NODE
                        or cell["measurement_node"] != factorial.MEASUREMENT_NODE
                    ):
                        raise FactorialArtifactError("finite cell forcing fields changed")
                    response = cell["response"]
                    _validate_response(response, "cell")
                    for gain_name, harmonic_name in (
                        ("endpoint_fundamental_transfer_gain", "fundamental"),
                        ("endpoint_third_harmonic_transfer_gain", "third_harmonic"),
                    ):
                        gain = cell[gain_name]
                        if not _finite_number(gain) or gain < 0:
                            raise FactorialArtifactError("finite cell gain is invalid")
                        expected_gain = (
                            response[harmonic_name]["amplitude"]
                            / factorial.FORCING_AMPLITUDE
                        )
                        if not math.isclose(
                            gain, expected_gain, rel_tol=1.0e-12, abs_tol=1.0e-15
                        ):
                            raise FactorialArtifactError("finite cell gain is inconsistent")
                    final_state = cell["final_state"]
                    expected_state_size = systems[configuration].state_size
                    if (
                        not isinstance(final_state, list)
                        or len(final_state) != expected_state_size
                        or any(not _finite_number(value) for value in final_state)
                    ):
                        raise FactorialArtifactError("finite cell final state is invalid")
                    local = cell["attachment_local_mismatch_responses"]
                    attachments = systems[configuration].attachments
                    if not isinstance(local, list) or len(local) != len(attachments):
                        raise FactorialArtifactError("finite cell local-response count is invalid")
                    local_keys = {
                        "attachment_index",
                        "attachment_node",
                        "fundamental",
                        "third_harmonic",
                        "fundamental_phase_relative_to_forcing_radians",
                        "descriptive_first_harmonic_effective_stiffness",
                    }
                    for local_index, (local_response, attachment) in enumerate(
                        zip(local, attachments)
                    ):
                        if not isinstance(local_response, dict) or set(local_response) != local_keys:
                            raise FactorialArtifactError("local-response schema is invalid")
                        if (
                            local_response["attachment_index"] != local_index
                            or local_response["attachment_node"] != attachment.node
                        ):
                            raise FactorialArtifactError("local-response attachment mismatch")
                        _validate_harmonic(local_response["fundamental"], "local fundamental")
                        _validate_harmonic(local_response["third_harmonic"], "local third")
                        phase = local_response[
                            "fundamental_phase_relative_to_forcing_radians"
                        ]
                        stiffness = local_response[
                            "descriptive_first_harmonic_effective_stiffness"
                        ]
                        if not _finite_number(phase) or not _finite_number(stiffness):
                            raise FactorialArtifactError("local-response scalar is nonfinite")
                        if not -math.pi <= phase <= math.pi:
                            raise FactorialArtifactError("local-response phase is out of range")
                        amplitude = local_response["fundamental"]["amplitude"]
                        expected_phase = math.atan2(
                            local_response["fundamental"]["cos_coefficient"],
                            local_response["fundamental"]["sin_coefficient"],
                        )
                        if not math.isclose(
                            phase,
                            expected_phase,
                            rel_tol=1.0e-12,
                            abs_tol=1.0e-15,
                        ):
                            raise FactorialArtifactError("local-response phase is inconsistent")
                        expected_stiffness = (
                            attachment.linear_stiffness
                            + 0.75 * attachment.cubic_stiffness * amplitude**2
                        )
                        if stiffness < 0 or not math.isclose(
                            stiffness,
                            expected_stiffness,
                            rel_tol=1.0e-12,
                            abs_tol=1.0e-15,
                        ):
                            raise FactorialArtifactError("local effective stiffness is inconsistent")
                elif (
                    set(cell) != nonfinite_keys
                    or cell.get("failure") != "NONFINITE_TRAJECTORY"
                ):
                    raise FactorialArtifactError("nonfinite cell schema/failure is invalid")
                else:
                    complete = False
                    sweep_has_nonfinite_state = True
            if len(cells) != len(factorial.FREQUENCIES):
                complete = False
    return complete


def _validate_endpoints(endpoints: object) -> None:
    if not isinstance(endpoints, dict):
        raise FactorialArtifactError("complete artifact must contain endpoints")
    required = {
        "mean_fundamental_transfer_gain",
        "aggregate_interaction_ratio",
        "distributed_duffing_mean_gain_ratio",
        "worst_case_distributed_duffing_excess_ratio",
        "maximum_relative_hysteresis",
        "maximum_relative_settling_delta",
        "cell_interaction_ratios",
    }
    if set(endpoints) != required:
        raise FactorialArtifactError("artifact endpoint schema is invalid")
    means = endpoints["mean_fundamental_transfer_gain"]
    if not isinstance(means, dict) or set(means) != set(
        factorial.FACTOR_CONFIGURATIONS
    ):
        raise FactorialArtifactError("artifact mean-gain schema is invalid")
    numeric_fields = required - {
        "mean_fundamental_transfer_gain",
        "cell_interaction_ratios",
    }
    if any(not _finite_number(endpoints[field]) for field in numeric_fields):
        raise FactorialArtifactError("artifact has a nonfinite endpoint")
    if any(not _finite_number(value) for value in means.values()):
        raise FactorialArtifactError("artifact has a nonfinite mean gain")
    if any(value < 0 for value in means.values()):
        raise FactorialArtifactError("artifact has a negative mean gain")
    for field in (
        "distributed_duffing_mean_gain_ratio",
        "maximum_relative_hysteresis",
        "maximum_relative_settling_delta",
    ):
        if endpoints[field] < 0:
            raise FactorialArtifactError("artifact has a negative physical endpoint")
    ratios = endpoints["cell_interaction_ratios"]
    if not isinstance(ratios, list) or len(ratios) != 34:
        raise FactorialArtifactError("artifact interaction-cell endpoint count is invalid")
    if any(not _finite_number(value) for value in ratios):
        raise FactorialArtifactError("artifact has a nonfinite interaction endpoint")


def _validate_gates(artifact: Mapping[str, object], complete: bool) -> bool:
    gates = artifact.get("gates")
    if not isinstance(gates, list) or not gates:
        raise FactorialArtifactError("artifact gates are missing")
    expected = {
        "complete_finite_grid": ("==", True),
        "aggregate_interaction_ratio": (">=", factorial.AGGREGATE_INTERACTION_MINIMUM),
        "distributed_duffing_mean_gain_ratio": (
            "<=",
            factorial.DISTRIBUTED_DUFFING_MEAN_RATIO_MAXIMUM,
        ),
        "worst_case_distributed_duffing_excess_ratio": (
            "<=",
            factorial.WORST_CASE_EXCESS_RATIO_MAXIMUM,
        ),
        "maximum_relative_hysteresis": (
            "<=",
            factorial.MAXIMUM_RELATIVE_HYSTERESIS,
        ),
        "maximum_relative_settling_delta": (
            "<=",
            factorial.MAXIMUM_RELATIVE_SETTLING_DELTA,
        ),
    }
    names = [gate.get("name") if isinstance(gate, dict) else None for gate in gates]
    if len(names) != len(set(names)) or any(name not in expected for name in names):
        raise FactorialArtifactError("artifact gate set is invalid")
    frozen_names = (
        list(expected) if complete else ["complete_finite_grid"]
    )
    if names != frozen_names:
        raise FactorialArtifactError("artifact gates are not in frozen production order")
    endpoints = artifact.get("endpoints")
    for gate in gates:
        name = gate["name"]
        operator, threshold = expected[name]
        if set(gate) != {"name", "operator", "threshold", "observed", "passed"}:
            raise FactorialArtifactError("artifact gate schema is invalid")
        if gate["operator"] != operator or gate["threshold"] != threshold:
            raise FactorialArtifactError("artifact gate contract changed")
        observed = gate["observed"]
        if name == "complete_finite_grid":
            if type(observed) is not bool or observed != complete:
                raise FactorialArtifactError("complete-grid gate contradicts branches")
            recomputed = observed is True
        else:
            if not isinstance(endpoints, dict) or observed != endpoints.get(name):
                raise FactorialArtifactError("artifact gate contradicts endpoints")
            if not _finite_number(observed):
                raise FactorialArtifactError("artifact gate observation is nonfinite")
            recomputed = observed >= threshold if operator == ">=" else observed <= threshold
        if type(gate["passed"]) is not bool or gate["passed"] != recomputed:
            raise FactorialArtifactError("artifact gate pass flag is contradictory")
    return any(not gate["passed"] for gate in gates)


def _valid_resource_checkpoint(value: object) -> bool:
    if value in ("runtime_admission", "finalize"):
        return True
    if not isinstance(value, str):
        return False
    parts = value.split(":")
    if len(parts) != 4:
        return False
    configuration, direction, frequency_text, suffix = parts
    if configuration not in factorial.ALL_CONFIGURATIONS:
        return False
    if direction not in ("forward", "reverse"):
        return False
    valid_frequency_text = {
        f"{float(frequency):.17g}" for frequency in factorial.FREQUENCIES
    }
    if frequency_text not in valid_frequency_text:
        return False
    if suffix in ("start", "complete"):
        return True
    if not suffix.startswith("step-"):
        return False
    try:
        step = int(suffix[5:])
    except ValueError:
        return False
    maximum = factorial.TOTAL_PERIODS * factorial.STEPS_PER_PERIOD
    return (
        0 <= step < maximum
        and step % factorial.RESOURCE_CHECK_INTERVAL_STEPS == 0
    )


def _ordered_cell_locations():
    for configuration in factorial.ALL_CONFIGURATIONS:
        for direction, frequencies in (
            ("forward", factorial.FREQUENCIES),
            ("reverse", factorial.FREQUENCIES[::-1]),
        ):
            for index, frequency in enumerate(frequencies):
                yield configuration, direction, index, float(frequency)


def _completed_guard_trace(branches: Mapping[str, object]):
    count = 0
    terminal = "runtime_admission"
    first_missing = None
    missing_seen = False
    all_finite = True
    completed_cells = 0
    for configuration, direction, index, frequency in _ordered_cell_locations():
        cells = branches[configuration].get(direction, [])
        if index >= len(cells):
            if first_missing is None:
                first_missing = (configuration, direction, frequency)
            missing_seen = True
            continue
        if missing_seen:
            raise FactorialArtifactError("branch data is not a frozen-order prefix")
        cell = cells[index]
        completed_cells += 1
        prefix = f"{configuration}:{direction}:{frequency:.17g}"
        if cell["finite_trajectory"]:
            count += 49
            terminal = f"{prefix}:complete"
        else:
            all_finite = False
            steps = cell["steps_completed"]
            step_checkpoint = ((steps - 1) // factorial.RESOURCE_CHECK_INTERVAL_STEPS) * (
                factorial.RESOURCE_CHECK_INTERVAL_STEPS
            )
            count += 1 + step_checkpoint // factorial.RESOURCE_CHECK_INTERVAL_STEPS + 1
            terminal = f"{prefix}:step-{step_checkpoint}"
    return count, terminal, first_missing, all_finite, completed_cells


def _missing_cell_failure_checks(checkpoint: str, reason: str) -> int:
    suffix = checkpoint.rsplit(":", 1)[1]
    failed_check_counted = reason != "COOPERATIVE_DEADLINE_EXCEEDED"
    if suffix == "start":
        return int(failed_check_counted)
    if suffix == "complete":
        return 48 + int(failed_check_counted)
    step = int(suffix[5:])
    return 1 + step // factorial.RESOURCE_CHECK_INTERVAL_STEPS + int(
        failed_check_counted
    )


def _validate_guard_trace(
    artifact: Mapping[str, object],
    usage: Mapping[str, object],
    resource_failures: Sequence[Mapping[str, object]],
) -> None:
    if len(resource_failures) > 1:
        raise FactorialArtifactError("artifact has more than one terminal resource failure")
    cell_checks, terminal, first_missing, all_finite, completed_cells = (
        _completed_guard_trace(artifact["branches"])
    )
    failure = resource_failures[0] if resource_failures else None
    expected_count = 0
    expected_terminal = terminal
    if failure is not None and failure["checkpoint"] == "runtime_admission":
        if completed_cells != 0:
            raise FactorialArtifactError("admission failure has completed cells")
        if failure["reason"] == "RSS_MEASUREMENT_METHOD_CHANGED":
            raise FactorialArtifactError("method change is impossible at admission")
        expected_count = int(
            failure["reason"] != "COOPERATIVE_DEADLINE_EXCEEDED"
        )
        expected_terminal = "runtime_admission"
    else:
        if not usage["runtime_admission"]["passed"]:
            raise FactorialArtifactError("post-admission trace lacks passed admission")
        expected_count = 1 + cell_checks
        if failure is not None:
            checkpoint = failure["checkpoint"]
            if checkpoint == "finalize":
                if (
                    first_missing is not None
                    or completed_cells != 170
                    or not all_finite
                    or artifact.get("endpoints") is None
                ):
                    raise FactorialArtifactError("finalize failure precedes complete endpoints")
                expected_count += int(
                    failure["reason"] != "COOPERATIVE_DEADLINE_EXCEEDED"
                )
            else:
                if first_missing is None:
                    raise FactorialArtifactError("cell checkpoint failure has no missing cell")
                configuration, direction, frequency = first_missing
                expected_prefix = f"{configuration}:{direction}:{frequency:.17g}:"
                if not checkpoint.startswith(expected_prefix):
                    raise FactorialArtifactError("resource failure is not at first missing cell")
                expected_count += _missing_cell_failure_checks(
                    checkpoint, failure["reason"]
                )
            expected_terminal = checkpoint
        elif first_missing is not None:
            raise FactorialArtifactError("missing cell lacks a resource interruption")
        elif all_finite:
            if artifact.get("endpoints") is None:
                raise FactorialArtifactError("complete finite trace lacks endpoints")
            expected_count += 1
            expected_terminal = "finalize"
    if usage["resource_check_count"] != expected_count:
        raise FactorialArtifactError("resource check count contradicts branch trace")
    if usage["last_checked_checkpoint"] != expected_terminal:
        raise FactorialArtifactError("last checkpoint contradicts branch trace")
    if not 0 <= expected_count <= 8332:
        raise FactorialArtifactError("resource check count is outside frozen bounds")
    if completed_cells == 170 and all_finite and failure is None:
        if expected_count != 8332 or expected_terminal != "finalize":
            raise FactorialArtifactError("complete survivor trace is not exactly 8332 checks")
    if usage["computation_wall_seconds"] < usage["last_guard_elapsed_seconds"]:
        raise FactorialArtifactError("wall duration is below last guard elapsed time")
    if failure is None:
        return
    reason = failure["reason"]
    if reason == "COOPERATIVE_DEADLINE_EXCEEDED":
        if usage["last_guard_elapsed_seconds"] <= factorial.MAX_COMPUTATION_SECONDS:
            raise FactorialArtifactError("deadline failure did not exceed deadline")
    elif reason == "SELF_PROCESS_RSS_CEILING_EXCEEDED":
        if max(
            usage["maximum_observed_current_rss_bytes"],
            usage["maximum_observed_peak_rss_bytes"],
        ) <= factorial.MAX_RSS_BYTES:
            raise FactorialArtifactError("RSS failure did not exceed ceiling")
    else:
        if expected_count < 2:
            raise FactorialArtifactError("method change lacks two sampled checks")
        if failure["previous_rss_measurement_method"] != usage[
            "rss_measurement_method"
        ]:
            raise FactorialArtifactError("method-change previous method is inconsistent")
    guards = _resource_guards(artifact)
    if not guards["deadline_guard_passed"] and not guards["rss_guard_passed"]:
        raise FactorialArtifactError("both terminal resource guards cannot fail")


def _validate_artifact(artifact: Mapping[str, object]) -> None:
    if set(artifact) != EXPECTED_ARTIFACT_KEYS:
        raise FactorialArtifactError("factorial artifact top-level schema is invalid")
    if artifact.get("protocol_id") != PROTOCOL_ID:
        raise FactorialArtifactError("unexpected protocol")
    if type(artifact.get("run_id")) is not int or artifact["run_id"] not in RUN_IDS:
        raise FactorialArtifactError("unexpected run_id")
    if artifact.get("evidence_mode") != factorial.EVIDENCE_MODE:
        raise FactorialArtifactError("official artifact has unexpected evidence mode")
    if artifact.get("scientific_claim_status") != SCIENTIFIC_CLAIM_STATUS:
        raise FactorialArtifactError("unexpected scientific claim status")
    if artifact.get("novelty_claim_status") != NOVELTY_CLAIM_STATUS:
        raise FactorialArtifactError("unexpected novelty claim status")
    if artifact.get("production_path_changed") is not False:
        raise FactorialArtifactError("artifact changes the production-path claim")
    if artifact.get("model_or_corpus_loaded") is not False:
        raise FactorialArtifactError("artifact changes the model/corpus claim")
    if artifact.get("phase_and_continuation_policy") != factorial.FORCING_PHASE_POLICY:
        raise FactorialArtifactError("artifact phase policy changed")
    _validate_fixture(artifact)
    complete = _validate_branches(artifact)
    endpoints = artifact.get("endpoints")
    if complete:
        _validate_endpoints(endpoints)
        recomputed_endpoints = factorial.compute_endpoints(artifact["branches"])
        if endpoints != recomputed_endpoints:
            raise FactorialArtifactError("artifact endpoints contradict branch cells")
    elif endpoints is not None:
        raise FactorialArtifactError("incomplete artifact must not report endpoints")
    failed_gate = _validate_gates(artifact, complete)
    failures = artifact.get("failures")
    count = artifact.get("failure_count")
    if not isinstance(failures, list) or type(count) is not int or count != len(failures):
        raise FactorialArtifactError("artifact failure count is inconsistent")
    for failure in failures:
        if not isinstance(failure, dict) or not isinstance(failure.get("reason"), str):
            raise FactorialArtifactError("artifact failure record is invalid")
        check = failure.get("check")
        if check == "resource_guard":
            if failure["reason"] not in (
                "COOPERATIVE_DEADLINE_EXCEEDED",
                "SELF_PROCESS_RSS_CEILING_EXCEEDED",
                "RSS_MEASUREMENT_METHOD_CHANGED",
            ):
                raise FactorialArtifactError("resource failure reason is invalid")
            expected_resource_keys = {"check", "reason", "checkpoint"}
            if failure["reason"] == "RSS_MEASUREMENT_METHOD_CHANGED":
                expected_resource_keys |= {
                    "previous_rss_measurement_method",
                    "observed_rss_measurement_method",
                }
                previous = failure["previous_rss_measurement_method"]
                observed = failure["observed_rss_measurement_method"]
                if (
                    not isinstance(previous, str)
                    or not previous
                    or not isinstance(observed, str)
                    or not observed
                    or previous == observed
                ):
                    raise FactorialArtifactError("resource method-change evidence is invalid")
            if set(failure) != expected_resource_keys:
                raise FactorialArtifactError("resource failure schema is invalid")
            if not _valid_resource_checkpoint(failure["checkpoint"]):
                raise FactorialArtifactError("resource failure checkpoint is invalid")
        elif check == "frozen_gate":
            if set(failure) != {"check", "reason", "gate"}:
                raise FactorialArtifactError("gate failure schema is invalid")
            if failure["reason"] != "FROZEN_THRESHOLD_FAILED":
                raise FactorialArtifactError("gate failure reason is invalid")
        elif check == "finite_trajectory":
            expected_keys = {
                "check",
                "configuration",
                "direction",
                "frequency",
                "reason",
            }
            if set(failure) != expected_keys:
                raise FactorialArtifactError("trajectory failure schema is invalid")
            if failure["reason"] != "NONFINITE_TRAJECTORY":
                raise FactorialArtifactError("trajectory failure reason is invalid")
        else:
            raise FactorialArtifactError("artifact has an unknown failure check")
    usage = artifact.get("resource_usage")
    guards = _resource_guards(artifact)
    if not isinstance(usage, dict):
        raise FactorialArtifactError("artifact resource usage is invalid")
    expected_usage_keys = {
        "computation_wall_seconds",
        "cooperative_deadline_seconds",
        "deadline_enforced_during_integration",
        "deadline_guard_passed",
        "self_process_rss_ceiling_bytes",
        "rss_enforced_during_integration",
        "rss_guard_passed",
        "maximum_observed_current_rss_bytes",
        "maximum_observed_peak_rss_bytes",
        "initial_current_rss_bytes",
        "initial_peak_rss_bytes",
        "rss_measurement_method",
        "resource_check_count",
        "last_guard_elapsed_seconds",
        "last_checked_checkpoint",
        "runtime_admission",
        "guard_scope",
    }
    if set(usage) != expected_usage_keys:
        raise FactorialArtifactError("artifact resource-usage schema is invalid")
    if usage["cooperative_deadline_seconds"] != factorial.MAX_COMPUTATION_SECONDS:
        raise FactorialArtifactError("artifact deadline changed")
    if usage["self_process_rss_ceiling_bytes"] != factorial.MAX_RSS_BYTES:
        raise FactorialArtifactError("artifact RSS ceiling changed")
    for field in ("computation_wall_seconds", "last_guard_elapsed_seconds"):
        if not _finite_number(usage[field]) or usage[field] < 0:
            raise FactorialArtifactError("artifact resource duration is invalid")
    for field in (
        "maximum_observed_current_rss_bytes",
        "maximum_observed_peak_rss_bytes",
        "resource_check_count",
    ):
        if type(usage[field]) is not int or usage[field] < 0:
            raise FactorialArtifactError("artifact resource counter is invalid")
    if usage["maximum_observed_peak_rss_bytes"] < usage[
        "maximum_observed_current_rss_bytes"
    ]:
        raise FactorialArtifactError("artifact peak RSS is below current RSS")
    if usage["resource_check_count"] > 0:
        if not isinstance(usage["rss_measurement_method"], str) or not usage[
            "rss_measurement_method"
        ]:
            raise FactorialArtifactError("artifact RSS method is missing")
        for field in ("initial_current_rss_bytes", "initial_peak_rss_bytes"):
            if type(usage[field]) is not int or usage[field] <= 0:
                raise FactorialArtifactError("artifact initial RSS sample is invalid")
        if usage["initial_peak_rss_bytes"] < usage["initial_current_rss_bytes"]:
            raise FactorialArtifactError("artifact initial peak is below current RSS")
        if usage["maximum_observed_current_rss_bytes"] < usage[
            "initial_current_rss_bytes"
        ] or usage["maximum_observed_peak_rss_bytes"] < usage[
            "initial_peak_rss_bytes"
        ]:
            raise FactorialArtifactError("artifact maxima omit initial RSS sample")
    elif usage["rss_measurement_method"] is not None:
        raise FactorialArtifactError("artifact RSS method contradicts check count")
    elif usage["initial_current_rss_bytes"] is not None or usage[
        "initial_peak_rss_bytes"
    ] is not None:
        raise FactorialArtifactError("artifact initial RSS contradicts zero checks")
    if guards["rss_guard_passed"] and usage[
        "maximum_observed_peak_rss_bytes"
    ] > factorial.MAX_RSS_BYTES:
        raise FactorialArtifactError("passing RSS guard contradicts observed peak")
    if guards["deadline_guard_passed"] and usage[
        "last_guard_elapsed_seconds"
    ] > factorial.MAX_COMPUTATION_SECONDS:
        raise FactorialArtifactError("passing deadline guard contradicts last check")
    if not _valid_resource_checkpoint(usage["last_checked_checkpoint"]):
        raise FactorialArtifactError("artifact last resource checkpoint is invalid")
    admission = usage.get("runtime_admission")
    if not isinstance(admission, dict) or set(admission) != {
        "checkpoint",
        "current_rss_bytes",
        "peak_rss_bytes",
        "rss_measurement_method",
        "passed",
    }:
        raise FactorialArtifactError("artifact runtime admission is invalid")
    if admission["checkpoint"] != "runtime_admission" or type(
        admission["passed"]
    ) is not bool:
        raise FactorialArtifactError("artifact runtime admission fields are invalid")
    if (
        admission["current_rss_bytes"] != usage["initial_current_rss_bytes"]
        or admission["peak_rss_bytes"] != usage["initial_peak_rss_bytes"]
        or admission["rss_measurement_method"] != usage["rss_measurement_method"]
    ):
        raise FactorialArtifactError("runtime admission contradicts initial RSS sample")
    if guards["deadline_enforced_during_integration"] is not True:
        raise FactorialArtifactError("deadline enforcement claim is false")
    if guards["rss_enforced_during_integration"] is not True:
        raise FactorialArtifactError("RSS enforcement claim is false")
    guard_failed = (
        not guards["deadline_guard_passed"]
        or not guards["rss_guard_passed"]
        or not admission["passed"]
    )
    resource_failures = [
        failure
        for failure in failures
        if isinstance(failure, dict) and failure.get("check") == "resource_guard"
    ]
    if resource_failures and resource_failures[-1]["checkpoint"] != usage[
        "last_checked_checkpoint"
    ]:
        raise FactorialArtifactError("resource failure is not the last checked checkpoint")
    expected_admission = bool(
        usage["resource_check_count"] >= 1
        and isinstance(usage["rss_measurement_method"], str)
        and bool(usage["rss_measurement_method"])
        and type(usage["initial_current_rss_bytes"]) is int
        and type(usage["initial_peak_rss_bytes"]) is int
        and usage["initial_current_rss_bytes"] <= factorial.MAX_RSS_BYTES
        and usage["initial_peak_rss_bytes"] <= factorial.MAX_RSS_BYTES
        and not any(
            failure.get("checkpoint") == "runtime_admission"
            for failure in resource_failures
        )
    )
    if admission["passed"] != expected_admission:
        raise FactorialArtifactError("runtime admission pass flag is contradictory")
    _validate_guard_trace(artifact, usage, resource_failures)
    expected_failures = []
    for configuration in factorial.ALL_CONFIGURATIONS:
        for direction in ("forward", "reverse"):
            for cell in artifact["branches"][configuration].get(direction, []):
                if not cell["finite_trajectory"]:
                    expected_failures.append(
                        {
                            "check": "finite_trajectory",
                            "configuration": configuration,
                            "direction": direction,
                            "frequency": cell["frequency"],
                            "reason": "NONFINITE_TRAJECTORY",
                        }
                    )
    if resource_failures:
        expected_failures.append(resource_failures[0])
    elif not expected_failures and complete:
        expected_failures.extend(
            {
                "check": "frozen_gate",
                "reason": "FROZEN_THRESHOLD_FAILED",
                "gate": gate["name"],
            }
            for gate in artifact["gates"]
            if not gate["passed"]
        )
    if failures != expected_failures:
        raise FactorialArtifactError(
            "failure history is not the exact production-order sequence"
        )
    for failure in resource_failures:
        reason = failure["reason"]
        if reason == "COOPERATIVE_DEADLINE_EXCEEDED" and guards[
            "deadline_guard_passed"
        ]:
            raise FactorialArtifactError("deadline failure contradicts passing guard")
        if reason in (
            "SELF_PROCESS_RSS_CEILING_EXCEEDED",
            "RSS_MEASUREMENT_METHOD_CHANGED",
        ) and guards["rss_guard_passed"]:
            raise FactorialArtifactError("RSS failure contradicts passing guard")
    if not admission["passed"] and not any(
        failure.get("checkpoint") == "runtime_admission"
        for failure in resource_failures
    ):
        raise FactorialArtifactError("failed runtime admission lacks matching failure")
    if not guards["deadline_guard_passed"] and not any(
        failure.get("reason") == "COOPERATIVE_DEADLINE_EXCEEDED"
        for failure in resource_failures
    ):
        raise FactorialArtifactError("failed deadline guard lacks matching failure")
    if not guards["rss_guard_passed"] and not any(
        failure.get("reason")
        in ("SELF_PROCESS_RSS_CEILING_EXCEEDED", "RSS_MEASUREMENT_METHOD_CHANGED")
        for failure in resource_failures
    ):
        raise FactorialArtifactError("failed RSS guard lacks matching failure")
    if not complete and not (
        resource_failures
        or any(
            isinstance(failure, dict)
            and failure.get("check") == "finite_trajectory"
            for failure in failures
        )
    ):
        raise FactorialArtifactError("incomplete grid lacks a numerical/resource failure")
    status = artifact.get("status")
    if status not in ALLOWED_STATUSES:
        raise FactorialArtifactError("unexpected status")
    if status == ALLOWED_STATUSES[0]:
        if count != 0 or failed_gate or guard_failed or not complete:
            raise FactorialArtifactError("survivor contradicts failures, gates, or guards")
        if not all(gate["passed"] for gate in artifact["gates"]):
            raise FactorialArtifactError("survivor contains a failed gate")
    elif count < 1 or not (failed_gate or guard_failed or not complete):
        raise FactorialArtifactError("killed status lacks a consistent failing predicate")


def _build_manifest(payload: Mapping[str, object], artifact_bytes: bytes) -> Dict[str, object]:
    _validate_artifact(payload)
    required = (
        "protocol_id",
        "run_id",
        "status",
        "scientific_claim_status",
        "novelty_claim_status",
        "failure_count",
    )
    if any(field not in payload for field in required):
        raise FactorialArtifactError("factorial payload is missing manifest fields")
    return {
        "manifest_schema": MANIFEST_SCHEMA,
        **{field: payload[field] for field in required},
        "resource_guards": _resource_guards(payload),
        "factorial_artifact": {
            "path": FACTORIAL_FILENAME,
            "sha256": _sha256(artifact_bytes),
            "bytes": len(artifact_bytes),
        },
    }


def write_run(output_directory: Path, run_id: int) -> Dict[str, object]:
    """Transactionally publish one verified two-file factorial result set."""

    if type(run_id) is not int or run_id not in RUN_IDS:
        raise FactorialArtifactError("run_id must be exactly 7 or 11")
    directory = Path(output_directory)
    if directory.is_symlink():
        raise FactorialArtifactError("output directory must not be a symlink")
    parent = directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        checked = _checked_output_directory(directory, create=False)
        if any(checked.iterdir()):
            try:
                existing_manifest, existing_artifact = verify_manifest(checked)
            except FactorialArtifactError as exc:
                raise FactorialArtifactError(
                    "nonempty output is not a valid recoverable result set"
                ) from exc
            if existing_artifact["run_id"] != run_id:
                raise FactorialArtifactError("existing result belongs to another run_id")
            return existing_manifest
    payload = run_factorial(run_id)
    artifact_bytes = _canonical_json_bytes(payload)
    manifest = _build_manifest(payload, artifact_bytes)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{directory.name}.stage-", dir=str(parent))
    )
    try:
        _atomic_write(_checked_member(staging, FACTORIAL_FILENAME), artifact_bytes)
        _atomic_write(
            _checked_member(staging, MANIFEST_FILENAME),
            _canonical_json_bytes(manifest),
        )
        verified, _ = verify_manifest(staging)
        _fsync_directory(staging)
        if directory.exists():
            directory.rmdir()
        os.replace(staging, directory)
        staging = None
        try:
            _fsync_directory(parent)
        except OSError:
            # The atomic rename is the publication commit point.  If the final
            # set verifies, report committed success instead of raising and
            # leaving a valid result that would block an operator retry.
            committed, _ = verify_manifest(directory)
            return committed
        return verified
    finally:
        if staging is not None and staging.exists():
            if staging.parent.resolve() != parent.resolve():
                raise FactorialArtifactError("unsafe staging cleanup path")
            shutil.rmtree(staging)


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory metadata fsync where stdlib supports it."""

    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(str(directory), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_manifest(output_directory: Path) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Verify canonical bytes, safe paths, digest, size, and duplicated fields."""

    directory = _checked_output_directory(Path(output_directory), create=False)
    manifest, _ = _load_canonical_mapping(
        _checked_member(directory, MANIFEST_FILENAME), "manifest"
    )
    expected_manifest_keys = {
        "manifest_schema",
        "protocol_id",
        "run_id",
        "status",
        "scientific_claim_status",
        "novelty_claim_status",
        "failure_count",
        "resource_guards",
        "factorial_artifact",
    }
    if set(manifest) != expected_manifest_keys:
        raise FactorialArtifactError("manifest schema is incomplete or extended")
    if manifest.get("manifest_schema") != MANIFEST_SCHEMA:
        raise FactorialArtifactError("unexpected manifest schema")
    entry = manifest.get("factorial_artifact")
    if not isinstance(entry, dict) or entry.get("path") != FACTORIAL_FILENAME:
        raise FactorialArtifactError("manifest names an unexpected artifact")
    if set(entry) != {"path", "sha256", "bytes"}:
        raise FactorialArtifactError("manifest artifact-entry schema is invalid")
    guards = manifest.get("resource_guards")
    if not isinstance(guards, dict) or set(guards) != {
        "deadline_enforced_during_integration",
        "deadline_guard_passed",
        "rss_enforced_during_integration",
        "rss_guard_passed",
    }:
        raise FactorialArtifactError("manifest resource-guard schema is invalid")
    artifact, raw = _load_canonical_mapping(
        _checked_member(directory, entry["path"]), "factorial artifact"
    )
    if type(entry.get("bytes")) is not int or entry["bytes"] != len(raw):
        raise FactorialArtifactError("factorial artifact byte count mismatch")
    digest = entry.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise FactorialArtifactError("factorial artifact digest is invalid")
    if not hmac.compare_digest(_sha256(raw), digest.lower()):
        raise FactorialArtifactError("factorial artifact digest mismatch")
    duplicated = (
        "protocol_id",
        "run_id",
        "status",
        "scientific_claim_status",
        "novelty_claim_status",
        "failure_count",
    )
    for field in duplicated:
        if field not in manifest or manifest[field] != artifact.get(field):
            raise FactorialArtifactError(f"manifest/artifact mismatch for {field}")
    if artifact["protocol_id"] != PROTOCOL_ID:
        raise FactorialArtifactError("unexpected protocol")
    if type(artifact["run_id"]) is not int or artifact["run_id"] not in RUN_IDS:
        raise FactorialArtifactError("unexpected run_id")
    if artifact["status"] not in ALLOWED_STATUSES:
        raise FactorialArtifactError("unexpected status")
    if artifact["scientific_claim_status"] != SCIENTIFIC_CLAIM_STATUS:
        raise FactorialArtifactError("unexpected scientific claim status")
    if artifact["novelty_claim_status"] != NOVELTY_CLAIM_STATUS:
        raise FactorialArtifactError("unexpected novelty claim status")
    if artifact.get("endpoints") is None and artifact["status"] != ALLOWED_STATUSES[1]:
        raise FactorialArtifactError("surviving artifact must contain endpoints")
    if manifest.get("resource_guards") != _resource_guards(artifact):
        raise FactorialArtifactError("manifest/artifact resource-guard mismatch")
    _validate_artifact(artifact)
    return manifest, artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, type=int, choices=RUN_IDS)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    manifest = write_run(args.output_directory, args.run_id)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
