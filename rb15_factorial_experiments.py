"""Preregistered RB-15 distribution-by-nonlinearity factorial.

The immutable protocol is documented in
``docs/research/rb15-distribution-nonlinearity-factorial-protocol-2026-08-15.md``.
This dependency-light module is CPU-only and does not load a model or corpus.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from nonlinear_neutralizer_experiments import (
    Attachment,
    GraphSystem,
    HOST_DAMPING,
    _self_rss_bytes,
    fit_harmonic,
    graph_derivative,
    grounded_path_stiffness,
    rk4_step,
)


PROTOCOL_ID = "PRW-RCM1-NLN-FACTORIAL-001"
EVIDENCE_MODE = "PREREGISTERED_DEPENDENCY_LIGHT_SYNTHETIC_FACTORIAL"
SCIENTIFIC_CLAIM_STATUS = "UNCONFIRMED"
NOVELTY_CLAIM_STATUS = "UNCONFIRMED"
RUN_IDS = (7, 11)

FREQUENCIES = np.linspace(0.70, 1.50, 17, dtype=np.float64)
FORCING_AMPLITUDE = 0.20
FORCING_NODE = 0
MEASUREMENT_NODE = 4
STEPS_PER_PERIOD = 200
TOTAL_PERIODS = 60
MEASURE_PERIODS = 20

MAX_RSS_BYTES = 512 * 1024 * 1024
MAX_COMPUTATION_SECONDS = 300.0
RESOURCE_CHECK_INTERVAL_STEPS = 256
MATCH_TOLERANCE = 1.0e-15
DENOMINATOR_FLOOR = 1.0e-12

AGGREGATE_INTERACTION_MINIMUM = 0.05
DISTRIBUTED_DUFFING_MEAN_RATIO_MAXIMUM = 0.95
WORST_CASE_EXCESS_RATIO_MAXIMUM = 0.10
MAXIMUM_RELATIVE_HYSTERESIS = 0.15
MAXIMUM_RELATIVE_SETTLING_DELTA = 0.05

FACTOR_CONFIGURATIONS = (
    "single_linear",
    "distributed_linear",
    "single_duffing",
    "distributed_duffing",
)
CONTROL_CONFIGURATION = "no_sidecar_control"
ALL_CONFIGURATIONS = FACTOR_CONFIGURATIONS + (CONTROL_CONFIGURATION,)
FORCING_PHASE_POLICY = (
    "local integration time and forcing phase reset to zero at every frequency "
    "cell; state continuation occurs only within one configuration and direction"
)


class FactorialValidationError(ValueError):
    """Raised when the frozen factorial contract is violated."""


class ResourceLimitExceeded(RuntimeError):
    """Raised cooperatively when the frozen deadline or RSS ceiling is crossed."""

    def __init__(
        self,
        code: str,
        checkpoint: str,
        *,
        previous_rss_method: Optional[str] = None,
        observed_rss_method: Optional[str] = None,
    ) -> None:
        super().__init__(f"{code} at {checkpoint}")
        self.code = code
        self.checkpoint = checkpoint
        self.previous_rss_method = previous_rss_method
        self.observed_rss_method = observed_rss_method


@dataclass
class ResourceGuard:
    """Cooperative self-process deadline and RSS guard."""

    started: float
    deadline_seconds: float = MAX_COMPUTATION_SECONDS
    rss_ceiling_bytes: int = MAX_RSS_BYTES
    maximum_observed_current_rss_bytes: int = 0
    maximum_observed_peak_rss_bytes: int = 0
    rss_measurement_method: Optional[str] = None
    check_count: int = 0
    last_check_elapsed_seconds: float = 0.0
    initial_current_rss_bytes: Optional[int] = None
    initial_peak_rss_bytes: Optional[int] = None
    last_checkpoint: Optional[str] = None

    def check(self, checkpoint: str) -> None:
        self.last_checkpoint = checkpoint
        elapsed = time.perf_counter() - self.started
        self.last_check_elapsed_seconds = elapsed
        if elapsed > self.deadline_seconds:
            raise ResourceLimitExceeded("COOPERATIVE_DEADLINE_EXCEEDED", checkpoint)
        current, peak, method = _self_rss_bytes()
        if self.check_count == 0:
            self.initial_current_rss_bytes = current
            self.initial_peak_rss_bytes = peak
        self.check_count += 1
        self.maximum_observed_current_rss_bytes = max(
            self.maximum_observed_current_rss_bytes, current
        )
        self.maximum_observed_peak_rss_bytes = max(
            self.maximum_observed_peak_rss_bytes, peak
        )
        if self.rss_measurement_method is None:
            self.rss_measurement_method = method
        elif self.rss_measurement_method != method:
            raise ResourceLimitExceeded(
                "RSS_MEASUREMENT_METHOD_CHANGED",
                checkpoint,
                previous_rss_method=self.rss_measurement_method,
                observed_rss_method=method,
            )
        if current > self.rss_ceiling_bytes or peak > self.rss_ceiling_bytes:
            raise ResourceLimitExceeded("SELF_PROCESS_RSS_CEILING_EXCEEDED", checkpoint)


def _configurations() -> Dict[str, GraphSystem]:
    stiffness = grounded_path_stiffness(5)
    return {
        "single_linear": GraphSystem(
            stiffness,
            (Attachment(2, 0.40, 0.40, 0.064, 0.0),),
        ),
        "distributed_linear": GraphSystem(
            stiffness,
            (
                Attachment(1, 0.20, 0.20, 0.032, 0.0),
                Attachment(3, 0.20, 0.20, 0.032, 0.0),
            ),
        ),
        "single_duffing": GraphSystem(
            stiffness,
            (Attachment(2, 0.40, 0.40, 0.064, 1.0),),
        ),
        "distributed_duffing": GraphSystem(
            stiffness,
            (
                Attachment(1, 0.20, 0.20, 0.032, 0.50),
                Attachment(3, 0.20, 0.20, 0.032, 0.50),
            ),
        ),
        CONTROL_CONFIGURATION: GraphSystem(stiffness, ()),
    }


def _configuration_totals(system: GraphSystem) -> Dict[str, object]:
    return {
        "auxiliary_mass_total": float(sum(item.mass for item in system.attachments)),
        "attachment_linear_stiffness_total": float(
            sum(item.linear_stiffness for item in system.attachments)
        ),
        "attachment_relative_damping_total": float(
            sum(item.relative_damping for item in system.attachments)
        ),
        "attachment_cubic_stiffness_total": float(
            sum(item.cubic_stiffness for item in system.attachments)
        ),
        "attachment_nodes": [item.node for item in system.attachments],
        "sidecar_count": len(system.attachments),
        "total_dynamic_state_count": system.state_size,
        "state_bytes_float64": 8 * system.state_size,
    }


def _expected_host_stiffness() -> np.ndarray:
    return np.asarray(
        (
            (1.20, -1.00, 0.00, 0.00, 0.00),
            (-1.00, 2.20, -1.00, 0.00, 0.00),
            (0.00, -1.00, 2.20, -1.00, 0.00),
            (0.00, 0.00, -1.00, 2.20, -1.00),
            (0.00, 0.00, 0.00, -1.00, 1.20),
        ),
        dtype=np.float64,
    )


def _attachment_signature(system: GraphSystem) -> Tuple[Tuple[object, ...], ...]:
    return tuple(
        (
            item.node,
            item.mass,
            item.linear_stiffness,
            item.relative_damping,
            item.cubic_stiffness,
        )
        for item in system.attachments
    )


def resource_preflight(*, reduced_fixture: bool = False) -> Dict[str, object]:
    """Validate the immutable design and return its exact modeled work."""

    if np.dtype(np.float64).itemsize != 8:
        raise FactorialValidationError("numpy float64 must be eight bytes")
    expected_frequencies = np.linspace(0.70, 1.50, 17, dtype=np.float64)
    if not reduced_fixture:
        if not np.array_equal(FREQUENCIES, expected_frequencies):
            raise FactorialValidationError("official frequency grid changed")
        frozen_scalars = (
            (HOST_DAMPING, 0.04),
            (FORCING_AMPLITUDE, 0.20),
            (FORCING_NODE, 0),
            (MEASUREMENT_NODE, 4),
            (STEPS_PER_PERIOD, 200),
            (TOTAL_PERIODS, 60),
            (MEASURE_PERIODS, 20),
            (MAX_RSS_BYTES, 512 * 1024 * 1024),
            (MAX_COMPUTATION_SECONDS, 300.0),
            (RESOURCE_CHECK_INTERVAL_STEPS, 256),
            (MATCH_TOLERANCE, 1.0e-15),
            (DENOMINATOR_FLOOR, 1.0e-12),
            (AGGREGATE_INTERACTION_MINIMUM, 0.05),
            (DISTRIBUTED_DUFFING_MEAN_RATIO_MAXIMUM, 0.95),
            (WORST_CASE_EXCESS_RATIO_MAXIMUM, 0.10),
            (MAXIMUM_RELATIVE_HYSTERESIS, 0.15),
            (MAXIMUM_RELATIVE_SETTLING_DELTA, 0.05),
        )
        if any(observed != expected for observed, expected in frozen_scalars):
            raise FactorialValidationError("official scalar protocol constant changed")
        if RUN_IDS != (7, 11):
            raise FactorialValidationError("official run IDs changed")
        expected_phase_policy = (
            "local integration time and forcing phase reset to zero at every "
            "frequency cell; state continuation occurs only within one "
            "configuration and direction"
        )
        if FORCING_PHASE_POLICY != expected_phase_policy:
            raise FactorialValidationError("official phase policy changed")
    configurations = _configurations()
    if tuple(configurations) != ALL_CONFIGURATIONS:
        raise FactorialValidationError("factorial configuration order changed")
    expected_host = _expected_host_stiffness()
    expected_attachments = {
        "single_linear": ((2, 0.40, 0.40, 0.064, 0.0),),
        "distributed_linear": (
            (1, 0.20, 0.20, 0.032, 0.0),
            (3, 0.20, 0.20, 0.032, 0.0),
        ),
        "single_duffing": ((2, 0.40, 0.40, 0.064, 1.0),),
        "distributed_duffing": (
            (1, 0.20, 0.20, 0.032, 0.50),
            (3, 0.20, 0.20, 0.032, 0.50),
        ),
        CONTROL_CONFIGURATION: (),
    }
    for name, system in configurations.items():
        if not np.array_equal(system.stiffness, expected_host):
            raise FactorialValidationError(f"{name} host stiffness changed")
        if _attachment_signature(system) != expected_attachments[name]:
            raise FactorialValidationError(f"{name} ordered attachments changed")
        for attachment in system.attachments:
            values = (
                attachment.mass,
                attachment.linear_stiffness,
                attachment.relative_damping,
                attachment.cubic_stiffness,
            )
            if not all(math.isfinite(value) for value in values):
                raise FactorialValidationError(f"{name} has a nonfinite parameter")
            if attachment.mass <= 0.0 or min(values[1:]) < 0.0:
                raise FactorialValidationError(f"{name} has a nonpassive parameter")
    totals = {
        name: _configuration_totals(system)
        for name, system in configurations.items()
    }
    expected_linear = {
        "auxiliary_mass_total": 0.40,
        "attachment_linear_stiffness_total": 0.40,
        "attachment_relative_damping_total": 0.064,
        "attachment_cubic_stiffness_total": 0.0,
    }
    expected_duffing = dict(expected_linear)
    expected_duffing["attachment_cubic_stiffness_total"] = 1.0
    for name in ("single_linear", "distributed_linear"):
        for field, expected in expected_linear.items():
            if not math.isclose(
                float(totals[name][field]),
                expected,
                rel_tol=0.0,
                abs_tol=MATCH_TOLERANCE,
            ):
                raise FactorialValidationError(f"{name} violates matched {field}")
    for name in ("single_duffing", "distributed_duffing"):
        for field, expected in expected_duffing.items():
            if not math.isclose(
                float(totals[name][field]),
                expected,
                rel_tol=0.0,
                abs_tol=MATCH_TOLERANCE,
            ):
                raise FactorialValidationError(f"{name} violates matched {field}")
    for run_id, scale in ((7, 1.0), (11, -0.8)):
        expected_host_state = np.asarray(
            (0.010, -0.005, 0.002, 0.0, 0.0), dtype=np.float64
        ) * np.float64(scale)
        for name, system in configurations.items():
            state = _initial_state(system, run_id)
            if not np.array_equal(state[:5], expected_host_state):
                raise FactorialValidationError(f"{name} initial host state changed")
            if not np.array_equal(state[5:10], np.zeros(5, dtype=np.float64)):
                raise FactorialValidationError(f"{name} initial host velocity changed")
            offset = 2 * system.node_count
            for index, attachment in enumerate(system.attachments):
                if state[offset + index] != expected_host_state[attachment.node]:
                    raise FactorialValidationError(
                        f"{name} initial attachment mismatch changed"
                    )
            if np.any(state[offset + len(system.attachments) :] != 0.0):
                raise FactorialValidationError(
                    f"{name} initial attachment velocity changed"
                )
    frequency_count = len(FREQUENCIES)
    factorial_cells = len(FACTOR_CONFIGURATIONS) * 2 * frequency_count
    control_cells = 2 * frequency_count
    total_cells = factorial_cells + control_cells
    steps = total_cells * TOTAL_PERIODS * STEPS_PER_PERIOD
    return {
        "design_validation_passed": True,
        "profile": "REDUCED_TEST_FIXTURE" if reduced_fixture else "OFFICIAL_FROZEN",
        "arithmetic": "numpy_float64",
        "cpu_only_dependency_path": True,
        "model_or_corpus_loaded": False,
        "child_processes_created": 0,
        "factorial_cell_count": factorial_cells,
        "report_only_control_cell_count": control_cells,
        "total_cell_count": total_cells,
        "modeled_rk4_step_count": steps,
        "modeled_rhs_evaluation_count": 4 * steps,
        "configuration_totals": totals,
    }


def _initial_state(system: GraphSystem, run_id: int) -> np.ndarray:
    if type(run_id) is not int or run_id not in RUN_IDS:
        raise FactorialValidationError("run_id must be exactly 7 or 11")
    host = np.asarray((0.010, -0.005, 0.002, 0.0, 0.0), dtype=np.float64)
    if run_id == 11:
        host *= np.float64(-0.8)
    state = np.zeros(system.state_size, dtype=np.float64)
    state[: system.node_count] = host
    sidecar_offset = 2 * system.node_count
    for index, attachment in enumerate(system.attachments):
        state[sidecar_offset + index] = host[attachment.node]
    return state


def _response(times: np.ndarray, values: np.ndarray, frequency: float) -> Dict[str, object]:
    fundamental = fit_harmonic(times, values, frequency, 1)
    third = fit_harmonic(times, values, frequency, 3)
    midpoint = values.shape[0] // 2
    previous = fit_harmonic(times[:midpoint], values[:midpoint], frequency, 1)
    final = fit_harmonic(times[midpoint:], values[midpoint:], frequency, 1)
    settling_delta = abs(final["amplitude"] - previous["amplitude"]) / max(
        final["amplitude"], DENOMINATOR_FLOOR
    )
    return {
        "fundamental": fundamental,
        "third_harmonic": third,
        "previous_half_fundamental_amplitude": previous["amplitude"],
        "final_half_fundamental_amplitude": final["amplitude"],
        "relative_settling_delta": float(settling_delta),
    }


def _run_cell(
    system: GraphSystem,
    frequency: float,
    initial_state: np.ndarray,
    guard: ResourceGuard,
    checkpoint_prefix: str,
) -> Tuple[Dict[str, object], np.ndarray]:
    step = float((2.0 * math.pi / frequency) / STEPS_PER_PERIOD)
    total_steps = TOTAL_PERIODS * STEPS_PER_PERIOD
    measurement_steps = MEASURE_PERIODS * STEPS_PER_PERIOD
    state = np.asarray(initial_state, dtype=np.float64).copy()
    times = np.empty(measurement_steps, dtype=np.float64)
    values = np.empty(measurement_steps, dtype=np.float64)
    mismatches = np.empty(
        (measurement_steps, len(system.attachments)), dtype=np.float64
    )

    def derivative(current_time: float, current_state: np.ndarray) -> np.ndarray:
        return graph_derivative(
            current_time,
            current_state,
            system,
            FORCING_AMPLITUDE,
            frequency,
            FORCING_NODE,
        )

    current_time = 0.0
    sample_index = 0
    guard.check(f"{checkpoint_prefix}:start")
    for index in range(total_steps):
        if index % RESOURCE_CHECK_INTERVAL_STEPS == 0:
            guard.check(f"{checkpoint_prefix}:step-{index}")
        state = rk4_step(derivative, current_time, state, step)
        current_time += step
        if not np.all(np.isfinite(state)):
            return (
                {
                    "frequency": frequency,
                    "finite_trajectory": False,
                    "failure": "NONFINITE_TRAJECTORY",
                    "steps_completed": index + 1,
                    "rhs_evaluations": 4 * (index + 1),
                },
                state,
            )
        if index >= total_steps - measurement_steps:
            times[sample_index] = current_time
            values[sample_index] = state[MEASUREMENT_NODE]
            sidecar_offset = 2 * system.node_count
            for attachment_index, attachment in enumerate(system.attachments):
                mismatches[sample_index, attachment_index] = (
                    state[sidecar_offset + attachment_index]
                    - state[attachment.node]
                )
            sample_index += 1
    guard.check(f"{checkpoint_prefix}:complete")
    response = _response(times, values, frequency)
    local_responses = []
    for index, attachment in enumerate(system.attachments):
        local = _response(times, mismatches[:, index], frequency)
        fundamental = local["fundamental"]
        amplitude = float(fundamental["amplitude"])
        local_responses.append(
            {
                "attachment_index": index,
                "attachment_node": attachment.node,
                "fundamental": fundamental,
                "third_harmonic": local["third_harmonic"],
                "fundamental_phase_relative_to_forcing_radians": math.atan2(
                    float(fundamental["cos_coefficient"]),
                    float(fundamental["sin_coefficient"]),
                ),
                "descriptive_first_harmonic_effective_stiffness": float(
                    attachment.linear_stiffness
                    + 0.75 * attachment.cubic_stiffness * amplitude**2
                ),
            }
        )
    return (
        {
            "frequency": frequency,
            "forcing_amplitude": FORCING_AMPLITUDE,
            "forcing_node": FORCING_NODE,
            "measurement_node": MEASUREMENT_NODE,
            "finite_trajectory": True,
            "failure": None,
            "steps_completed": total_steps,
            "rhs_evaluations": 4 * total_steps,
            "response": response,
            "endpoint_fundamental_transfer_gain": float(
                response["fundamental"]["amplitude"] / FORCING_AMPLITUDE
            ),
            "endpoint_third_harmonic_transfer_gain": float(
                response["third_harmonic"]["amplitude"] / FORCING_AMPLITUDE
            ),
            "attachment_local_mismatch_responses": local_responses,
            "final_state": state.tolist(),
        },
        state,
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise FactorialValidationError("cannot aggregate an empty sequence")
    return float(math.fsum(float(value) for value in values) / len(values))


def _index_cells(
    branches: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
) -> Dict[Tuple[str, str, float], Mapping[str, object]]:
    return {
        (configuration, direction, float(cell["frequency"])): cell
        for configuration, directions in branches.items()
        for direction, cells in directions.items()
        for cell in cells
    }


def compute_endpoints(
    branches: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
) -> Dict[str, object]:
    """Compute all preregistered endpoints from a complete finite factorial."""

    index = _index_cells(branches)
    frequencies = tuple(float(value) for value in FREQUENCIES)
    expected = len(ALL_CONFIGURATIONS) * 2 * len(frequencies)
    if len(index) != expected:
        raise FactorialValidationError("branches do not contain the complete grid")
    gains: Dict[str, List[float]] = {name: [] for name in FACTOR_CONFIGURATIONS}
    interactions: List[float] = []
    duffing_excesses: List[float] = []
    settling: List[float] = []
    for direction in ("forward", "reverse"):
        for frequency in frequencies:
            cell_gains = {}
            for configuration in FACTOR_CONFIGURATIONS:
                cell = index[(configuration, direction, frequency)]
                if not cell.get("finite_trajectory"):
                    raise FactorialValidationError("endpoint input contains a failed cell")
                gain = float(cell["endpoint_fundamental_transfer_gain"])
                if not math.isfinite(gain):
                    raise FactorialValidationError("endpoint input contains a nonfinite gain")
                gains[configuration].append(gain)
                cell_gains[configuration] = gain
                settling.append(float(cell["response"]["relative_settling_delta"]))
            interactions.append(
                (cell_gains["single_duffing"] - cell_gains["distributed_duffing"])
                - (cell_gains["single_linear"] - cell_gains["distributed_linear"])
            )
            duffing_excesses.append(
                cell_gains["distributed_duffing"] - cell_gains["single_duffing"]
            )
    mean_gains = {name: _mean(values) for name, values in gains.items()}
    baseline = max(mean_gains["single_linear"], DENOMINATOR_FLOOR)
    hysteresis_values = []
    for configuration in FACTOR_CONFIGURATIONS:
        for frequency in frequencies:
            forward = float(
                index[(configuration, "forward", frequency)][
                    "endpoint_fundamental_transfer_gain"
                ]
            )
            reverse = float(
                index[(configuration, "reverse", frequency)][
                    "endpoint_fundamental_transfer_gain"
                ]
            )
            hysteresis_values.append(
                abs(forward - reverse)
                / max(0.5 * (forward + reverse), DENOMINATOR_FLOOR)
            )
    return {
        "mean_fundamental_transfer_gain": mean_gains,
        "aggregate_interaction_ratio": float(_mean(interactions) / baseline),
        "distributed_duffing_mean_gain_ratio": float(
            mean_gains["distributed_duffing"]
            / max(mean_gains["single_duffing"], DENOMINATOR_FLOOR)
        ),
        "worst_case_distributed_duffing_excess_ratio": float(
            max(duffing_excesses) / baseline
        ),
        "maximum_relative_hysteresis": float(max(hysteresis_values)),
        "maximum_relative_settling_delta": float(max(settling)),
        "cell_interaction_ratios": [float(value / baseline) for value in interactions],
    }


def evaluate_gates(endpoints: Mapping[str, object]) -> List[Dict[str, object]]:
    """Evaluate every frozen survival gate without changing thresholds."""

    definitions = (
        (
            "aggregate_interaction_ratio",
            ">=",
            AGGREGATE_INTERACTION_MINIMUM,
            float(endpoints["aggregate_interaction_ratio"]),
        ),
        (
            "distributed_duffing_mean_gain_ratio",
            "<=",
            DISTRIBUTED_DUFFING_MEAN_RATIO_MAXIMUM,
            float(endpoints["distributed_duffing_mean_gain_ratio"]),
        ),
        (
            "worst_case_distributed_duffing_excess_ratio",
            "<=",
            WORST_CASE_EXCESS_RATIO_MAXIMUM,
            float(endpoints["worst_case_distributed_duffing_excess_ratio"]),
        ),
        (
            "maximum_relative_hysteresis",
            "<=",
            MAXIMUM_RELATIVE_HYSTERESIS,
            float(endpoints["maximum_relative_hysteresis"]),
        ),
        (
            "maximum_relative_settling_delta",
            "<=",
            MAXIMUM_RELATIVE_SETTLING_DELTA,
            float(endpoints["maximum_relative_settling_delta"]),
        ),
    )
    gates = []
    for name, operator, threshold, observed in definitions:
        passed = observed >= threshold if operator == ">=" else observed <= threshold
        gates.append(
            {
                "name": name,
                "operator": operator,
                "threshold": threshold,
                "observed": observed,
                "passed": bool(passed),
            }
        )
    return gates


def run_factorial(run_id: int, *, reduced_fixture: bool = False) -> Dict[str, object]:
    """Execute one complete preregistered factorial run."""

    if type(run_id) is not int or run_id not in RUN_IDS:
        raise FactorialValidationError("run_id must be exactly 7 or 11")
    preflight = resource_preflight(reduced_fixture=reduced_fixture)
    started = time.perf_counter()
    guard = ResourceGuard(started=started)
    branches: Dict[str, Dict[str, List[Dict[str, object]]]] = {
        name: {} for name in ALL_CONFIGURATIONS
    }
    failures: List[Dict[str, object]] = []
    resource_breach: Optional[ResourceLimitExceeded] = None
    try:
        guard.check("runtime_admission")
        for configuration, system in _configurations().items():
            for direction, frequency_values in (
                ("forward", FREQUENCIES),
                ("reverse", FREQUENCIES[::-1]),
            ):
                state = _initial_state(system, run_id)
                cells: List[Dict[str, object]] = []
                branches[configuration][direction] = cells
                for frequency_value in frequency_values:
                    frequency = float(frequency_value)
                    prefix = f"{configuration}:{direction}:{frequency:.17g}"
                    cell, state = _run_cell(system, frequency, state, guard, prefix)
                    cell["configuration"] = configuration
                    cell["direction"] = direction
                    cells.append(cell)
                    if cell["failure"] is not None:
                        failures.append(
                            {
                                "check": "finite_trajectory",
                                "configuration": configuration,
                                "direction": direction,
                                "frequency": frequency,
                                "reason": cell["failure"],
                            }
                        )
    except ResourceLimitExceeded as exc:
        resource_breach = exc
        failure = {
            "check": "resource_guard",
            "reason": exc.code,
            "checkpoint": exc.checkpoint,
        }
        if exc.code == "RSS_MEASUREMENT_METHOD_CHANGED":
            failure["previous_rss_measurement_method"] = exc.previous_rss_method
            failure["observed_rss_measurement_method"] = exc.observed_rss_method
        failures.append(failure)

    complete = all(
        len(branches[name].get(direction, ())) == len(FREQUENCIES)
        for name in ALL_CONFIGURATIONS
        for direction in ("forward", "reverse")
    )
    endpoints: Optional[Dict[str, object]] = None
    gates: List[Dict[str, object]] = [
        {
            "name": "complete_finite_grid",
            "operator": "==",
            "threshold": True,
            "observed": bool(complete and not failures),
            "passed": bool(complete and not failures),
        }
    ]
    if complete and not failures:
        try:
            endpoints = compute_endpoints(branches)
            gates.extend(evaluate_gates(endpoints))
            guard.check("finalize")
        except ResourceLimitExceeded as exc:
            resource_breach = exc
            failure = {
                "check": "resource_guard",
                "reason": exc.code,
                "checkpoint": exc.checkpoint,
            }
            if exc.code == "RSS_MEASUREMENT_METHOD_CHANGED":
                failure[
                    "previous_rss_measurement_method"
                ] = exc.previous_rss_method
                failure[
                    "observed_rss_measurement_method"
                ] = exc.observed_rss_method
            failures.append(failure)
        if resource_breach is None:
            for gate in gates:
                if not gate["passed"]:
                    failures.append(
                        {
                            "check": "frozen_gate",
                            "reason": "FROZEN_THRESHOLD_FAILED",
                            "gate": gate["name"],
                        }
                    )
    wall_seconds = time.perf_counter() - started
    survived = bool(not failures and gates and all(gate["passed"] for gate in gates))
    runtime_admission_passed = bool(
        guard.check_count >= 1
        and isinstance(guard.rss_measurement_method, str)
        and bool(guard.rss_measurement_method)
        and guard.initial_current_rss_bytes is not None
        and guard.initial_peak_rss_bytes is not None
        and guard.initial_current_rss_bytes <= MAX_RSS_BYTES
        and guard.initial_peak_rss_bytes <= MAX_RSS_BYTES
        and not (
            resource_breach is not None
            and resource_breach.checkpoint == "runtime_admission"
        )
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "status": (
            "FACTORIAL_SURVIVES_FROZEN_SYNTHETIC_GATES"
            if survived
            else "FACTORIAL_KILLED_ON_FROZEN_SYNTHETIC_GATES"
        ),
        "evidence_mode": (
            "REDUCED_TEST_FIXTURE_NOT_OFFICIAL_PROTOCOL"
            if reduced_fixture
            else EVIDENCE_MODE
        ),
        "scientific_claim_status": SCIENTIFIC_CLAIM_STATUS,
        "novelty_claim_status": NOVELTY_CLAIM_STATUS,
        "production_path_changed": False,
        "model_or_corpus_loaded": False,
        "phase_and_continuation_policy": FORCING_PHASE_POLICY,
        "fixture": {
            "randomness": "none",
            "fixed_run_ids": list(RUN_IDS),
            "frequencies": [float(value) for value in FREQUENCIES],
            "forcing_amplitude": FORCING_AMPLITUDE,
            "forcing_node": FORCING_NODE,
            "measurement_node": MEASUREMENT_NODE,
            "periods_per_cell": TOTAL_PERIODS,
            "measured_final_periods": MEASURE_PERIODS,
            "steps_per_period": STEPS_PER_PERIOD,
            "resource_check_interval_steps": RESOURCE_CHECK_INTERVAL_STEPS,
            "factor_configurations": list(FACTOR_CONFIGURATIONS),
            "report_only_control_configuration": CONTROL_CONFIGURATION,
        },
        "design_preflight": preflight,
        "branches": branches,
        "endpoints": endpoints,
        "gates": gates,
        "failure_count": len(failures),
        "failures": failures,
        "resource_usage": {
            "computation_wall_seconds": wall_seconds,
            "cooperative_deadline_seconds": MAX_COMPUTATION_SECONDS,
            "deadline_enforced_during_integration": True,
            "deadline_guard_passed": resource_breach is None
            or resource_breach.code != "COOPERATIVE_DEADLINE_EXCEEDED",
            "self_process_rss_ceiling_bytes": MAX_RSS_BYTES,
            "rss_enforced_during_integration": True,
            "rss_guard_passed": resource_breach is None
            or resource_breach.code
            not in (
                "SELF_PROCESS_RSS_CEILING_EXCEEDED",
                "RSS_MEASUREMENT_METHOD_CHANGED",
            ),
            "maximum_observed_current_rss_bytes": (
                guard.maximum_observed_current_rss_bytes
            ),
            "maximum_observed_peak_rss_bytes": guard.maximum_observed_peak_rss_bytes,
            "initial_current_rss_bytes": guard.initial_current_rss_bytes,
            "initial_peak_rss_bytes": guard.initial_peak_rss_bytes,
            "rss_measurement_method": guard.rss_measurement_method,
            "resource_check_count": guard.check_count,
            "last_guard_elapsed_seconds": guard.last_check_elapsed_seconds,
            "last_checked_checkpoint": guard.last_checkpoint,
            "runtime_admission": {
                "checkpoint": "runtime_admission",
                "current_rss_bytes": guard.initial_current_rss_bytes,
                "peak_rss_bytes": guard.initial_peak_rss_bytes,
                "rss_measurement_method": guard.rss_measurement_method,
                "passed": runtime_admission_passed,
            },
            "guard_scope": "cooperative deadline and self-process RSS; no children",
        },
    }
