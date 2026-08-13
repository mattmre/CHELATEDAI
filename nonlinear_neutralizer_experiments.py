"""Frozen RB-15 Stage-A nonlinear-neutraliser numerical sanity harness.

This module implements only the dependency-light mathematical fixture frozen in
``docs/research/nonlinear-neutraliser-subspace-transfer-2026-08.md``.  It uses
NumPy float64 state, fixed-step classical RK4, and stdlib process accounting.
It does not load a model or corpus, alter a retrieval path, or support a
scientific, product, RAG, or novelty claim.
"""

from __future__ import annotations

import ctypes
import math
import os
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


PROTOCOL_ID = "PRW-RCM1-NLN-STAGE-A-001"
EVIDENCE_MODE = "FROZEN_DEPENDENCY_LIGHT_MATHEMATICAL_SANITY"
SCIENTIFIC_CLAIM_STATUS = "UNCONFIRMED"
NOVELTY_CLAIM_STATUS = "UNCONFIRMED"
RUN_IDS = (7, 11)

HOST_DAMPING = 0.04
GROUNDING_STIFFNESS = 0.20
FIXED_STEP = 0.001
LINEAR_END_TIME = 1.0
ENERGY_END_TIME = 4.0
LINEAR_TOLERANCE = 1.0e-6
ENERGY_STEP_TOLERANCE = 1.0e-9
PROTECTED_LEAKAGE_TOLERANCE = 1.0e-12

STEPS_PER_PERIOD = 200
TOTAL_PERIODS = 60
MEASURE_PERIODS = 20
SCALAR_FREQUENCIES = np.linspace(0.70, 1.60, 37, dtype=np.float64)
SCALAR_AMPLITUDES = (0.05, 0.30)
GRAPH_FREQUENCIES = np.linspace(0.70, 1.50, 17, dtype=np.float64)
GRAPH_FORCING_AMPLITUDE = 0.20
PEAK_TIE_TOLERANCE = 1.0e-12

MAX_RSS_BYTES = 256 * 1024 * 1024
MAX_WALL_SECONDS = 120.0
FORCING_PHASE_CONVENTION = (
    "each frequency cell resets local integration time to t=0 and therefore "
    "starts A*sin(omega*t) at phase 0; continuation warm-starts state only"
)


class StageAValidationError(ValueError):
    """Raised when a caller violates the immutable Stage-A contract."""


@dataclass(frozen=True)
class Attachment:
    """One point attachment with a passive hardening Duffing potential."""

    node: int
    mass: float
    linear_stiffness: float
    relative_damping: float
    cubic_stiffness: float

    def __post_init__(self) -> None:
        if isinstance(self.node, bool) or not isinstance(self.node, int):
            raise StageAValidationError("attachment node must be an integer")
        values = (
            self.mass,
            self.linear_stiffness,
            self.relative_damping,
            self.cubic_stiffness,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise StageAValidationError("attachment parameters must be finite")
        if self.mass <= 0.0:
            raise StageAValidationError("attachment mass must be positive")
        if min(values[1:]) < 0.0:
            raise StageAValidationError(
                "Stage A permits only passive nonnegative attachment coefficients"
            )


@dataclass(frozen=True)
class GraphSystem:
    """Episode-fixed symmetric host operator and point attachments."""

    stiffness: np.ndarray
    attachments: Tuple[Attachment, ...] = ()

    def __post_init__(self) -> None:
        matrix = np.asarray(self.stiffness, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
            raise StageAValidationError("stiffness must be a non-empty square matrix")
        if not np.array_equal(matrix, matrix.T):
            raise StageAValidationError("stiffness must be exactly symmetric")
        if not np.all(np.isfinite(matrix)):
            raise StageAValidationError("stiffness must be finite")
        if float(np.min(np.linalg.eigvalsh(matrix))) < -1.0e-12:
            raise StageAValidationError("stiffness must be positive semidefinite")
        for attachment in self.attachments:
            if attachment.node < 0 or attachment.node >= matrix.shape[0]:
                raise StageAValidationError("attachment node is outside the host")
        object.__setattr__(self, "stiffness", matrix.copy())

    @property
    def node_count(self) -> int:
        return int(self.stiffness.shape[0])

    @property
    def state_size(self) -> int:
        return 2 * self.node_count + 2 * len(self.attachments)


def grounded_path_stiffness(node_count: int) -> np.ndarray:
    """Return the frozen path Laplacian plus ``0.20 I`` in float64."""

    if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count < 2:
        raise StageAValidationError("node_count must be an integer >= 2")
    matrix = np.zeros((node_count, node_count), dtype=np.float64)
    for index in range(node_count - 1):
        matrix[index, index] += 1.0
        matrix[index + 1, index + 1] += 1.0
        matrix[index, index + 1] -= 1.0
        matrix[index + 1, index] -= 1.0
    matrix += GROUNDING_STIFFNESS * np.eye(node_count, dtype=np.float64)
    return matrix


def _state_slices(system: GraphSystem) -> Tuple[slice, slice, slice, slice]:
    nodes = system.node_count
    count = len(system.attachments)
    return (
        slice(0, nodes),
        slice(nodes, 2 * nodes),
        slice(2 * nodes, 2 * nodes + count),
        slice(2 * nodes + count, 2 * nodes + 2 * count),
    )


def graph_derivative(
    time_value: float,
    state: np.ndarray,
    system: GraphSystem,
    forcing_amplitude: float = 0.0,
    forcing_frequency: float = 0.0,
    forcing_node: int = 0,
) -> np.ndarray:
    """Evaluate the frozen equal-and-opposite host/sidecar equations."""

    vector = np.asarray(state, dtype=np.float64)
    if vector.shape != (system.state_size,):
        raise StageAValidationError("state has the wrong size for the graph system")
    if forcing_node < 0 or forcing_node >= system.node_count:
        raise StageAValidationError("forcing_node is outside the host")
    x_slice, velocity_slice, z_slice, z_velocity_slice = _state_slices(system)
    x = vector[x_slice]
    velocity = vector[velocity_slice]
    z = vector[z_slice]
    z_velocity = vector[z_velocity_slice]
    acceleration = -system.stiffness @ x - HOST_DAMPING * velocity
    if forcing_amplitude != 0.0:
        acceleration[forcing_node] += forcing_amplitude * math.sin(
            forcing_frequency * time_value
        )
    z_acceleration = np.empty(len(system.attachments), dtype=np.float64)
    for index, attachment in enumerate(system.attachments):
        mismatch = z[index] - x[attachment.node]
        mismatch_velocity = z_velocity[index] - velocity[attachment.node]
        force = (
            attachment.relative_damping * mismatch_velocity
            + attachment.linear_stiffness * mismatch
            + attachment.cubic_stiffness * mismatch**3
        )
        acceleration[attachment.node] += force
        z_acceleration[index] = -force / attachment.mass
    derivative = np.empty_like(vector)
    derivative[x_slice] = velocity
    derivative[velocity_slice] = acceleration
    derivative[z_slice] = z_velocity
    derivative[z_velocity_slice] = z_acceleration
    return derivative


Derivative = Callable[[float, np.ndarray], np.ndarray]


def rk4_step(derivative: Derivative, time_value: float, state: np.ndarray, step: float) -> np.ndarray:
    """Take one fixed classical fourth-order Runge-Kutta step."""

    dt = np.float64(step)
    half = np.float64(0.5) * dt
    k1 = derivative(time_value, state)
    k2 = derivative(time_value + float(half), state + half * k1)
    k3 = derivative(time_value + float(half), state + half * k2)
    k4 = derivative(time_value + float(dt), state + dt * k3)
    return np.asarray(
        state + (dt / np.float64(6.0)) * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
        dtype=np.float64,
    )


def total_energy(state: np.ndarray, system: GraphSystem) -> float:
    """Evaluate the exact passive energy frozen in the RB-15 protocol."""

    vector = np.asarray(state, dtype=np.float64)
    x_slice, velocity_slice, z_slice, z_velocity_slice = _state_slices(system)
    x = vector[x_slice]
    velocity = vector[velocity_slice]
    z = vector[z_slice]
    z_velocity = vector[z_velocity_slice]
    energy = 0.5 * float(velocity @ velocity)
    energy += 0.5 * float(x @ system.stiffness @ x)
    for index, attachment in enumerate(system.attachments):
        mismatch = float(z[index] - x[attachment.node])
        energy += 0.5 * attachment.mass * float(z_velocity[index] ** 2)
        energy += 0.5 * attachment.linear_stiffness * mismatch**2
        energy += 0.25 * attachment.cubic_stiffness * mismatch**4
    return float(energy)


def _initial_displacements(run_id: int) -> Tuple[np.ndarray, float]:
    if type(run_id) is not int or run_id not in RUN_IDS:
        raise StageAValidationError("run_id must be exactly 7 or 11")
    host = np.asarray((0.10, -0.05, 0.02, 0.00), dtype=np.float64)
    sidecar = 0.03
    if run_id == 11:
        host = np.asarray(-0.8 * host, dtype=np.float64)
        sidecar *= -0.8
    return host, float(sidecar)


def _graph_initial_state(
    system: GraphSystem,
    host_displacement: Optional[np.ndarray] = None,
    sidecar_displacement: Optional[Sequence[float]] = None,
) -> np.ndarray:
    state = np.zeros(system.state_size, dtype=np.float64)
    nodes = system.node_count
    if host_displacement is not None:
        host = np.asarray(host_displacement, dtype=np.float64)
        if host.shape != (nodes,):
            raise StageAValidationError("host initial displacement has wrong shape")
        state[:nodes] = host
    if sidecar_displacement is not None:
        sidecars = np.asarray(sidecar_displacement, dtype=np.float64)
        if sidecars.shape != (len(system.attachments),):
            raise StageAValidationError("sidecar initial displacement has wrong shape")
        state[2 * nodes : 2 * nodes + len(system.attachments)] = sidecars
    return state


def _integrate_fixed(
    system: GraphSystem,
    initial_state: np.ndarray,
    end_time: float,
    step: float,
) -> Tuple[np.ndarray, int]:
    steps = int(round(end_time / step))
    if not math.isclose(steps * step, end_time, rel_tol=0.0, abs_tol=1.0e-14):
        raise StageAValidationError("end_time must be an integer multiple of step")
    state = np.asarray(initial_state, dtype=np.float64).copy()

    def derivative(current_time: float, current_state: np.ndarray) -> np.ndarray:
        return graph_derivative(current_time, current_state, system)

    current_time = 0.0
    for _ in range(steps):
        state = rk4_step(derivative, current_time, state, step)
        current_time += step
    return state, 4 * steps


def _linear_state_matrix(system: GraphSystem) -> np.ndarray:
    """Build the independent first-order linear operator for kappa == 0."""

    if any(attachment.cubic_stiffness != 0.0 for attachment in system.attachments):
        raise StageAValidationError("linear oracle requires cubic_stiffness == 0")
    nodes = system.node_count
    count = len(system.attachments)
    size = system.state_size
    matrix = np.zeros((size, size), dtype=np.float64)
    x_slice, velocity_slice, z_slice, z_velocity_slice = _state_slices(system)
    matrix[x_slice, velocity_slice] = np.eye(nodes, dtype=np.float64)
    matrix[z_slice, z_velocity_slice] = np.eye(count, dtype=np.float64)
    coupling = np.zeros((count, nodes), dtype=np.float64)
    masses = np.empty(count, dtype=np.float64)
    stiffnesses = np.empty(count, dtype=np.float64)
    dampings = np.empty(count, dtype=np.float64)
    for index, attachment in enumerate(system.attachments):
        coupling[index, attachment.node] = 1.0
        masses[index] = attachment.mass
        stiffnesses[index] = attachment.linear_stiffness
        dampings[index] = attachment.relative_damping
    linear = np.diag(stiffnesses)
    damping = np.diag(dampings)
    inverse_mass = np.diag(1.0 / masses)
    matrix[velocity_slice, x_slice] = -system.stiffness - coupling.T @ linear @ coupling
    matrix[velocity_slice, velocity_slice] = (
        -HOST_DAMPING * np.eye(nodes, dtype=np.float64)
        - coupling.T @ damping @ coupling
    )
    matrix[velocity_slice, z_slice] = coupling.T @ linear
    matrix[velocity_slice, z_velocity_slice] = coupling.T @ damping
    matrix[z_velocity_slice, x_slice] = inverse_mass @ linear @ coupling
    matrix[z_velocity_slice, velocity_slice] = inverse_mass @ damping @ coupling
    matrix[z_velocity_slice, z_slice] = -inverse_mass @ linear
    matrix[z_velocity_slice, z_velocity_slice] = -inverse_mass @ damping
    return matrix


def _eigendecomposed_transition(
    system: GraphSystem, initial_state: np.ndarray, end_time: float
) -> Tuple[np.ndarray, float]:
    operator = _linear_state_matrix(system)
    eigenvalues, eigenvectors = np.linalg.eig(operator)
    coefficients = np.linalg.solve(eigenvectors, np.asarray(initial_state, dtype=np.complex128))
    evolved = eigenvectors @ (np.exp(eigenvalues * end_time) * coefficients)
    imaginary_residual = float(np.max(np.abs(np.imag(evolved))))
    return np.asarray(np.real(evolved), dtype=np.float64), imaginary_residual


def run_linear_limit_check(run_id: int) -> Dict[str, object]:
    """Compare RK4 with an independently eigendecomposed linear transition."""

    host, sidecar = _initial_displacements(run_id)
    system = GraphSystem(
        grounded_path_stiffness(4),
        (Attachment(1, 0.20, 0.20, 0.032, 0.0),),
    )
    initial = _graph_initial_state(system, host, (sidecar,))
    rk4_final, evaluations = _integrate_fixed(
        system, initial, LINEAR_END_TIME, FIXED_STEP
    )
    oracle_final, imaginary_residual = _eigendecomposed_transition(
        system, initial, LINEAR_END_TIME
    )
    error = float(np.linalg.norm(rk4_final - oracle_final))
    return {
        "final_state_l2_error": error,
        "tolerance": LINEAR_TOLERANCE,
        "passed": error <= LINEAR_TOLERANCE,
        "oracle": "independent_numpy_eigendecomposition_of_first_order_operator",
        "oracle_imaginary_residual": imaginary_residual,
        "rk4_final_state": rk4_final.tolist(),
        "oracle_final_state": oracle_final.tolist(),
        "rhs_evaluations": evaluations,
    }


def run_energy_check(run_id: int) -> Dict[str, object]:
    """Trace the exact unforced energy of the frozen hardening cell."""

    host, sidecar = _initial_displacements(run_id)
    system = GraphSystem(
        grounded_path_stiffness(4),
        (Attachment(1, 0.20, 0.20, 0.032, 0.50),),
    )
    state = _graph_initial_state(system, host, (sidecar,))

    def derivative(current_time: float, current_state: np.ndarray) -> np.ndarray:
        return graph_derivative(current_time, current_state, system)

    steps = int(round(ENERGY_END_TIME / FIXED_STEP))
    prior_energy = total_energy(state, system)
    initial_energy = prior_energy
    minimum_energy = prior_energy
    maximum_energy = prior_energy
    maximum_positive_increment = 0.0
    maximum_state_norm = float(np.linalg.norm(state))
    current_time = 0.0
    finite = True
    for _ in range(steps):
        state = rk4_step(derivative, current_time, state, FIXED_STEP)
        current_time += FIXED_STEP
        if not np.all(np.isfinite(state)):
            finite = False
            break
        energy = total_energy(state, system)
        maximum_positive_increment = max(
            maximum_positive_increment, energy - prior_energy
        )
        minimum_energy = min(minimum_energy, energy)
        maximum_energy = max(maximum_energy, energy)
        maximum_state_norm = max(maximum_state_norm, float(np.linalg.norm(state)))
        prior_energy = energy
    return {
        "initial_energy": initial_energy,
        "final_energy": prior_energy if finite else None,
        "minimum_energy": minimum_energy,
        "maximum_energy": maximum_energy,
        "maximum_positive_energy_step": maximum_positive_increment,
        "tolerance": ENERGY_STEP_TOLERANCE,
        "finite_trajectory": finite,
        "maximum_state_l2_norm": maximum_state_norm,
        "passed": finite and maximum_positive_increment <= ENERGY_STEP_TOLERANCE,
        "steps_completed": steps if finite else int(round(current_time / FIXED_STEP)),
        "rhs_evaluations": 4 * int(round(current_time / FIXED_STEP)),
        "boundedness_scope": (
            "fixture-specific grounded positive-definite stiffness; energy "
            "monotonicity alone would not prove bounded state for a semidefinite operator"
        ),
    }


def run_protected_channel_check(run_id: int) -> Dict[str, object]:
    """Exercise the exact block-diagonal protected-channel invariant cell."""

    host, sidecar = _initial_displacements(run_id)
    base = grounded_path_stiffness(4)
    duplicated = np.zeros((8, 8), dtype=np.float64)
    duplicated[:4, :4] = base
    duplicated[4:, 4:] = base
    candidate = GraphSystem(
        duplicated,
        (Attachment(1, 0.20, 0.20, 0.032, 0.50),),
    )
    reference = GraphSystem(base, ())
    candidate_state = _graph_initial_state(
        candidate, np.concatenate((host, host)), (sidecar,)
    )
    reference_state = _graph_initial_state(reference, host)

    def candidate_derivative(
        current_time: float, current_state: np.ndarray
    ) -> np.ndarray:
        return graph_derivative(current_time, current_state, candidate)

    def reference_derivative(
        current_time: float, current_state: np.ndarray
    ) -> np.ndarray:
        return graph_derivative(current_time, current_state, reference)

    steps = int(round(ENERGY_END_TIME / FIXED_STEP))
    maximum_difference = 0.0
    current_time = 0.0
    for _ in range(steps):
        candidate_state = rk4_step(
            candidate_derivative, current_time, candidate_state, FIXED_STEP
        )
        reference_state = rk4_step(
            reference_derivative, current_time, reference_state, FIXED_STEP
        )
        current_time += FIXED_STEP
        difference = candidate_state[4:8] - reference_state[:4]
        maximum_difference = max(maximum_difference, float(np.linalg.norm(difference)))
    final_difference = candidate_state[4:8] - reference_state[:4]
    denominator = float(
        float(np.linalg.norm(reference_state[:4])) + np.finfo(np.float64).eps
    )
    relative_leakage = float(float(np.linalg.norm(final_difference)) / denominator)
    coupling_to_protected_norm = 0.0
    off_block_operator_norm = float(
        np.linalg.norm(duplicated[:4, 4:]) + np.linalg.norm(duplicated[4:, :4])
    )
    return {
        "final_relative_leakage": relative_leakage,
        "maximum_protected_state_l2_difference": maximum_difference,
        "tolerance": PROTECTED_LEAKAGE_TOLERANCE,
        "bitwise_equal_final": bool(
            np.array_equal(candidate_state[4:8], reference_state[:4])
        ),
        "attachment_map_times_protected_norm": coupling_to_protected_norm,
        "off_block_operator_norm": off_block_operator_norm,
        "passed": bool(relative_leakage <= PROTECTED_LEAKAGE_TOLERANCE),
        "rhs_evaluations": 8 * steps,
        "scope": (
            "exact synthetic block-diagonal invariant only; does not establish "
            "protected invariance for learned or coupled representations"
        ),
    }


def fit_harmonic(
    times: np.ndarray, values: np.ndarray, frequency: float, harmonic: int
) -> Dict[str, float]:
    """Fit ``[1, sin(h*w*t), cos(h*w*t)]`` by float64 least squares."""

    if harmonic not in (1, 3):
        raise StageAValidationError("only the frozen fundamental and third harmonic are used")
    sample_times = np.asarray(times, dtype=np.float64)
    samples = np.asarray(values, dtype=np.float64)
    phase = np.float64(harmonic * frequency) * sample_times
    design = np.column_stack(
        (
            np.ones(sample_times.shape[0], dtype=np.float64),
            np.sin(phase),
            np.cos(phase),
        )
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, samples, rcond=None)
    fitted = design @ coefficients
    residual_rms = float(np.sqrt(np.mean((samples - fitted) ** 2)))
    amplitude = float(math.hypot(float(coefficients[1]), float(coefficients[2])))
    return {
        "amplitude": amplitude,
        "offset": float(coefficients[0]),
        "sin_coefficient": float(coefficients[1]),
        "cos_coefficient": float(coefficients[2]),
        "residual_rms": residual_rms,
    }


def _measured_response(
    times: np.ndarray, values: np.ndarray, frequency: float
) -> Dict[str, object]:
    fundamental = fit_harmonic(times, values, frequency, 1)
    third = fit_harmonic(times, values, frequency, 3)
    midpoint = values.shape[0] // 2
    prior_half = fit_harmonic(times[:midpoint], values[:midpoint], frequency, 1)
    last_half = fit_harmonic(times[midpoint:], values[midpoint:], frequency, 1)
    convergence_delta = abs(last_half["amplitude"] - prior_half["amplitude"])
    return {
        "fundamental": fundamental,
        "third_harmonic": third,
        "last_half_fundamental_amplitude": last_half["amplitude"],
        "previous_half_fundamental_amplitude": prior_half["amplitude"],
        "last_half_vs_previous_half_amplitude_delta": convergence_delta,
        "convergence_assessment": "UNASSESSED_NO_FROZEN_GATE",
    }


def _scalar_cell(
    frequency: float,
    forcing_amplitude: float,
    initial_state: np.ndarray,
) -> Tuple[Dict[str, object], np.ndarray]:
    step = float((2.0 * math.pi / frequency) / STEPS_PER_PERIOD)
    total_steps = TOTAL_PERIODS * STEPS_PER_PERIOD
    measurement_steps = MEASURE_PERIODS * STEPS_PER_PERIOD
    state = np.asarray(initial_state, dtype=np.float64).copy()
    times = np.empty(measurement_steps, dtype=np.float64)
    values = np.empty(measurement_steps, dtype=np.float64)

    def derivative(current_time: float, current_state: np.ndarray) -> np.ndarray:
        displacement = float(current_state[0])
        velocity = float(current_state[1])
        return np.asarray(
            (
                velocity,
                forcing_amplitude * math.sin(frequency * current_time)
                - 0.12 * velocity
                - displacement
                - displacement**3,
            ),
            dtype=np.float64,
        )

    current_time = 0.0
    sample_index = 0
    finite = True
    for index in range(total_steps):
        state = rk4_step(derivative, current_time, state, step)
        current_time += step
        if not np.all(np.isfinite(state)):
            finite = False
            break
        if index >= total_steps - measurement_steps:
            times[sample_index] = current_time
            values[sample_index] = state[0]
            sample_index += 1
    if not finite or sample_index != measurement_steps:
        result: Dict[str, object] = {
            "frequency": frequency,
            "forcing_amplitude": forcing_amplitude,
            "finite_trajectory": False,
            "nonconverged": True,
            "failure": "NONFINITE_TRAJECTORY",
            "steps_completed": index + 1,
            "rhs_evaluations": 4 * (index + 1),
        }
        return result, state
    response = _measured_response(times, values, frequency)
    result = {
        "frequency": frequency,
        "forcing_amplitude": forcing_amplitude,
        "finite_trajectory": True,
        "nonconverged": None,
        "convergence_assessment": "UNASSESSED_NO_FROZEN_GATE",
        "failure": None,
        "steps_completed": total_steps,
        "rhs_evaluations": 4 * total_steps,
        "response": response,
        "final_state": state.tolist(),
    }
    return result, state


def _peak_summary(
    cells: Sequence[Dict[str, object]],
    declared_frequencies: Sequence[float],
) -> Dict[str, object]:
    declared_grid = sorted(float(value) for value in declared_frequencies)
    if not declared_grid:
        raise StageAValidationError("declared peak grid must not be empty")
    finite_cells = [
        cell
        for cell in cells
        if cell["finite_trajectory"]
        and math.isfinite(float(cell["response"]["fundamental"]["amplitude"]))
    ]
    if not finite_cells:
        return {
            "resolved": False,
            "failure": "NO_FINITE_FREQUENCY_CELL",
            "frequency": None,
            "amplitude": None,
            "boundary_censored": True,
        }
    maximum = max(
        float(cell["response"]["fundamental"]["amplitude"])
        for cell in finite_cells
    )
    tied = [
        cell
        for cell in finite_cells
        if abs(float(cell["response"]["fundamental"]["amplitude"]) - maximum)
        <= PEAK_TIE_TOLERANCE
    ]
    selected = min(tied, key=lambda cell: float(cell["frequency"]))
    frequency = float(selected["frequency"])
    boundary_censored = frequency in (declared_grid[0], declared_grid[-1])
    return {
        "resolved": not boundary_censored,
        "failure": "BOUNDARY_CENSORED_ARGMAX" if boundary_censored else None,
        "frequency": frequency,
        "amplitude": maximum,
        "boundary_censored": boundary_censored,
        "tie_count": len(tied),
        "tie_policy": "lowest_frequency_within_absolute_amplitude_1e-12",
    }


def run_scalar_sweeps() -> Dict[str, object]:
    """Run every frozen scalar Duffing cell in both continuation directions."""

    branches: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    discrepancies: List[Dict[str, object]] = []
    peaks: Dict[str, Dict[str, Dict[str, object]]] = {}
    failures: List[Dict[str, object]] = []
    rhs_evaluations = 0
    for amplitude in SCALAR_AMPLITUDES:
        amplitude_key = format(amplitude, ".2f")
        branches[amplitude_key] = {}
        peaks[amplitude_key] = {}
        for direction, frequencies in (
            ("forward", SCALAR_FREQUENCIES),
            ("reverse", SCALAR_FREQUENCIES[::-1]),
        ):
            state = np.zeros(2, dtype=np.float64)
            cells: List[Dict[str, object]] = []
            for frequency_value in frequencies:
                cell, state = _scalar_cell(
                    float(frequency_value), amplitude, state
                )
                cell["direction"] = direction
                cells.append(cell)
                rhs_evaluations += int(cell["rhs_evaluations"])
                if cell["failure"] is not None:
                    failures.append(
                        {
                            "cell": "scalar",
                            "amplitude": amplitude,
                            "direction": direction,
                            "frequency": float(frequency_value),
                            "reason": cell["failure"],
                        }
                    )
            branches[amplitude_key][direction] = cells
            peaks[amplitude_key][direction] = _peak_summary(
                cells, SCALAR_FREQUENCIES
            )
        forward_by_frequency = {
            float(cell["frequency"]): cell
            for cell in branches[amplitude_key]["forward"]
        }
        reverse_by_frequency = {
            float(cell["frequency"]): cell
            for cell in branches[amplitude_key]["reverse"]
        }
        for frequency in SCALAR_FREQUENCIES:
            value = float(frequency)
            forward = forward_by_frequency[value]
            reverse = reverse_by_frequency[value]
            discrepancy = None
            if forward["finite_trajectory"] and reverse["finite_trajectory"]:
                discrepancy = abs(
                    float(forward["response"]["fundamental"]["amplitude"])
                    - float(reverse["response"]["fundamental"]["amplitude"])
                )
            discrepancies.append(
                {
                    "forcing_amplitude": amplitude,
                    "frequency": value,
                    "forward_reverse_fundamental_amplitude_delta": discrepancy,
                }
            )
    shift_checks: Dict[str, Dict[str, object]] = {}
    for direction in ("forward", "reverse"):
        low = peaks[format(SCALAR_AMPLITUDES[0], ".2f")][direction]
        high = peaks[format(SCALAR_AMPLITUDES[1], ".2f")][direction]
        validated = bool(low["resolved"] and high["resolved"])
        shifted_up = bool(
            validated and float(high["frequency"]) > float(low["frequency"])
        )
        shift_checks[direction] = {
            "low_amplitude_peak": low,
            "high_amplitude_peak": high,
            "validated": validated,
            "shifted_upward": shifted_up,
        }
        if not shifted_up:
            failures.append(
                {
                    "cell": "scalar_peak_shift",
                    "direction": direction,
                    "reason": (
                        "BOUNDARY_CENSORED_PEAK"
                        if not validated
                        else "HARDENING_PEAK_DID_NOT_MOVE_UPWARD"
                    ),
                }
            )
    return {
        "fixture": {
            "equation": "r_ddot + 0.12*r_dot + r + r^3 = A*sin(omega*t)",
            "forcing_amplitudes": list(SCALAR_AMPLITUDES),
            "frequencies": SCALAR_FREQUENCIES.tolist(),
            "periods_per_cell": TOTAL_PERIODS,
            "steps_per_period": STEPS_PER_PERIOD,
            "measured_final_periods": MEASURE_PERIODS,
            "branch_initialization": (
                "all-zero at each direction endpoint; direction-local warm starts only"
            ),
            "forcing_phase_at_each_frequency_cell_radians": 0.0,
            "forcing_phase_convention": FORCING_PHASE_CONVENTION,
        },
        "branches": branches,
        "peak_summaries": peaks,
        "shift_checks": shift_checks,
        "all_directions_validate_upward_shift": all(
            bool(value["shifted_upward"]) for value in shift_checks.values()
        ),
        "forward_reverse_discrepancies": discrepancies,
        "failures": failures,
        "rhs_evaluations": rhs_evaluations,
    }


def _graph_configurations() -> Dict[str, GraphSystem]:
    stiffness = grounded_path_stiffness(5)
    return {
        "no_sidecar": GraphSystem(stiffness, ()),
        "single_full_physical_coefficients": GraphSystem(
            stiffness,
            (Attachment(2, 0.40, 0.40, 0.064, 1.00),),
        ),
        "two_distributed_half_parameters": GraphSystem(
            stiffness,
            (
                Attachment(1, 0.20, 0.20, 0.032, 0.50),
                Attachment(3, 0.20, 0.20, 0.032, 0.50),
            ),
        ),
        "two_colocated_half_parameters": GraphSystem(
            stiffness,
            (
                Attachment(2, 0.20, 0.20, 0.032, 0.50),
                Attachment(2, 0.20, 0.20, 0.032, 0.50),
            ),
        ),
    }


def _configuration_totals(system: GraphSystem) -> Dict[str, object]:
    return {
        "host_mass_total": float(system.node_count),
        "auxiliary_mass_total": float(sum(item.mass for item in system.attachments)),
        "combined_physical_mass_total": float(
            system.node_count + sum(item.mass for item in system.attachments)
        ),
        "attachment_linear_stiffness_total": float(
            sum(item.linear_stiffness for item in system.attachments)
        ),
        "attachment_relative_damping_total": float(
            sum(item.relative_damping for item in system.attachments)
        ),
        "attachment_cubic_stiffness_total": float(
            sum(item.cubic_stiffness for item in system.attachments)
        ),
        "sidecar_count": len(system.attachments),
        "auxiliary_dynamic_state_count": 2 * len(system.attachments),
        "host_dynamic_state_count": 2 * system.node_count,
        "total_dynamic_state_count": system.state_size,
        "state_bytes_float64": 8 * system.state_size,
        "attachment_nodes": [item.node for item in system.attachments],
    }


def _graph_cell(
    system: GraphSystem, frequency: float, initial_state: np.ndarray
) -> Tuple[Dict[str, object], np.ndarray]:
    step = float((2.0 * math.pi / frequency) / STEPS_PER_PERIOD)
    total_steps = TOTAL_PERIODS * STEPS_PER_PERIOD
    measurement_steps = MEASURE_PERIODS * STEPS_PER_PERIOD
    state = np.asarray(initial_state, dtype=np.float64).copy()
    times = np.empty(measurement_steps, dtype=np.float64)
    values = np.empty(measurement_steps, dtype=np.float64)
    local_mismatches = np.empty(
        (measurement_steps, len(system.attachments)), dtype=np.float64
    )

    def derivative(current_time: float, current_state: np.ndarray) -> np.ndarray:
        return graph_derivative(
            current_time,
            current_state,
            system,
            GRAPH_FORCING_AMPLITUDE,
            frequency,
            0,
        )

    current_time = 0.0
    sample_index = 0
    finite = True
    for index in range(total_steps):
        state = rk4_step(derivative, current_time, state, step)
        current_time += step
        if not np.all(np.isfinite(state)):
            finite = False
            break
        if index >= total_steps - measurement_steps:
            times[sample_index] = current_time
            values[sample_index] = state[4]
            sidecar_offset = 2 * system.node_count
            for attachment_index, attachment in enumerate(system.attachments):
                local_mismatches[sample_index, attachment_index] = (
                    state[sidecar_offset + attachment_index]
                    - state[attachment.node]
                )
            sample_index += 1
    if not finite or sample_index != measurement_steps:
        return (
            {
                "frequency": frequency,
                "finite_trajectory": False,
                "nonconverged": True,
                "failure": "NONFINITE_TRAJECTORY",
                "steps_completed": index + 1,
                "rhs_evaluations": 4 * (index + 1),
            },
            state,
        )
    response = _measured_response(times, values, frequency)
    attachment_local_responses: List[Dict[str, object]] = []
    for attachment_index, attachment in enumerate(system.attachments):
        local_response = _measured_response(
            times, local_mismatches[:, attachment_index], frequency
        )
        fundamental = local_response["fundamental"]
        third_harmonic = local_response["third_harmonic"]
        mismatch_amplitude = float(fundamental["amplitude"])
        attachment_local_responses.append(
            {
                "attachment_index": attachment_index,
                "attachment_node": attachment.node,
                "fundamental": fundamental,
                "third_harmonic": third_harmonic,
                "fundamental_phase_relative_to_forcing_radians": math.atan2(
                    float(fundamental["cos_coefficient"]),
                    float(fundamental["sin_coefficient"]),
                ),
                "third_harmonic_phase_relative_to_forcing_radians": math.atan2(
                    float(third_harmonic["cos_coefficient"]),
                    float(third_harmonic["sin_coefficient"]),
                ),
                "descriptive_first_harmonic_effective_stiffness": (
                    attachment.linear_stiffness
                    + 0.75
                    * attachment.cubic_stiffness
                    * mismatch_amplitude**2
                ),
                "last_half_fundamental_amplitude": local_response[
                    "last_half_fundamental_amplitude"
                ],
                "previous_half_fundamental_amplitude": local_response[
                    "previous_half_fundamental_amplitude"
                ],
                "last_half_vs_previous_half_amplitude_delta": local_response[
                    "last_half_vs_previous_half_amplitude_delta"
                ],
                "convergence_assessment": "UNASSESSED_NO_FROZEN_GATE",
            }
        )
    return (
        {
            "frequency": frequency,
            "forcing_amplitude": GRAPH_FORCING_AMPLITUDE,
            "measurement_node": 4,
            "finite_trajectory": True,
            "nonconverged": None,
            "convergence_assessment": "UNASSESSED_NO_FROZEN_GATE",
            "failure": None,
            "steps_completed": total_steps,
            "rhs_evaluations": 4 * total_steps,
            "response": response,
            "attachment_local_mismatch_responses": attachment_local_responses,
            "endpoint_fundamental_transfer_gain": float(
                response["fundamental"]["amplitude"] / GRAPH_FORCING_AMPLITUDE
            ),
            "endpoint_third_harmonic_transfer_gain": float(
                response["third_harmonic"]["amplitude"]
                / GRAPH_FORCING_AMPLITUDE
            ),
            "final_state": state.tolist(),
        },
        state,
    )


def run_graph_sweeps() -> Dict[str, object]:
    """Run all four matched graph configurations in both directions."""

    configurations = _graph_configurations()
    branches: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    discrepancies: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    rhs_evaluations = 0
    for name, system in configurations.items():
        branches[name] = {}
        for direction, frequencies in (
            ("forward", GRAPH_FREQUENCIES),
            ("reverse", GRAPH_FREQUENCIES[::-1]),
        ):
            state = np.zeros(system.state_size, dtype=np.float64)
            cells: List[Dict[str, object]] = []
            for frequency_value in frequencies:
                cell, state = _graph_cell(system, float(frequency_value), state)
                cell["direction"] = direction
                cells.append(cell)
                rhs_evaluations += int(cell["rhs_evaluations"])
                if cell["failure"] is not None:
                    failures.append(
                        {
                            "cell": "graph",
                            "configuration": name,
                            "direction": direction,
                            "frequency": float(frequency_value),
                            "reason": cell["failure"],
                        }
                    )
            branches[name][direction] = cells
        forward_by_frequency = {
            float(cell["frequency"]): cell for cell in branches[name]["forward"]
        }
        reverse_by_frequency = {
            float(cell["frequency"]): cell for cell in branches[name]["reverse"]
        }
        for frequency in GRAPH_FREQUENCIES:
            value = float(frequency)
            forward = forward_by_frequency[value]
            reverse = reverse_by_frequency[value]
            delta = None
            if forward["finite_trajectory"] and reverse["finite_trajectory"]:
                delta = abs(
                    float(forward["response"]["fundamental"]["amplitude"])
                    - float(reverse["response"]["fundamental"]["amplitude"])
                )
            discrepancies.append(
                {
                    "configuration": name,
                    "frequency": value,
                    "forward_reverse_fundamental_amplitude_delta": delta,
                }
            )
    comparison_cells: List[Dict[str, object]] = []
    colocated_single_deltas: List[float] = []
    colocated_single_local_deltas: List[float] = []
    colocated_pair_local_deltas: List[float] = []
    for direction in ("forward", "reverse"):
        indexed = {
            name: {
                float(cell["frequency"]): cell
                for cell in directions[direction]
            }
            for name, directions in branches.items()
        }
        for frequency in GRAPH_FREQUENCIES:
            value = float(frequency)
            amplitudes = {
                name: (
                    float(cells[value]["response"]["fundamental"]["amplitude"])
                    if cells[value]["finite_trajectory"]
                    else None
                )
                for name, cells in indexed.items()
            }
            distributed = amplitudes["two_distributed_half_parameters"]
            single = amplitudes["single_full_physical_coefficients"]
            colocated = amplitudes["two_colocated_half_parameters"]
            single_cell = indexed["single_full_physical_coefficients"][value]
            colocated_cell = indexed["two_colocated_half_parameters"][value]
            local_single = None
            local_colocated: Optional[List[float]] = None
            if single_cell["finite_trajectory"] and colocated_cell["finite_trajectory"]:
                local_single = float(
                    single_cell["attachment_local_mismatch_responses"][0][
                        "fundamental"
                    ]["amplitude"]
                )
                local_colocated = [
                    float(local["fundamental"]["amplitude"])
                    for local in colocated_cell[
                        "attachment_local_mismatch_responses"
                    ]
                ]
            comparison_cells.append(
                {
                    "direction": direction,
                    "frequency": value,
                    "fundamental_amplitudes": amplitudes,
                    "distributed_minus_single": (
                        distributed - single
                        if distributed is not None and single is not None
                        else None
                    ),
                    "distributed_minus_colocated": (
                        distributed - colocated
                        if distributed is not None and colocated is not None
                        else None
                    ),
                    "colocated_two_half_minus_single_full": (
                        colocated - single
                        if colocated is not None and single is not None
                        else None
                    ),
                    "single_full_local_mismatch_fundamental_amplitude": local_single,
                    "colocated_half_local_mismatch_fundamental_amplitudes": (
                        local_colocated
                    ),
                }
            )
            if colocated is not None and single is not None:
                colocated_single_deltas.append(abs(colocated - single))
            if local_single is not None and local_colocated is not None:
                colocated_single_local_deltas.extend(
                    abs(local_value - local_single)
                    for local_value in local_colocated
                )
                colocated_pair_local_deltas.append(
                    abs(local_colocated[0] - local_colocated[1])
                )
    colocated_equivalence_max_delta = (
        max(colocated_single_deltas) if colocated_single_deltas else None
    )
    colocated_equivalence_tolerance = 1.0e-8
    colocated_single_local_max_delta = (
        max(colocated_single_local_deltas)
        if colocated_single_local_deltas
        else None
    )
    colocated_pair_local_max_delta = (
        max(colocated_pair_local_deltas) if colocated_pair_local_deltas else None
    )
    colocated_equivalence_passed = bool(
        colocated_equivalence_max_delta is not None
        and colocated_equivalence_max_delta <= colocated_equivalence_tolerance
        and colocated_single_local_max_delta is not None
        and colocated_single_local_max_delta <= colocated_equivalence_tolerance
        and colocated_pair_local_max_delta is not None
        and colocated_pair_local_max_delta <= colocated_equivalence_tolerance
    )
    if not colocated_equivalence_passed:
        failures.append(
            {
                "cell": "graph_colocated_factorization_control",
                "reason": "COLOCATED_TWO_HALF_NOT_EQUIVALENT_TO_SINGLE_FULL",
                "maximum_entire_grid_fundamental_amplitude_delta": (
                    colocated_equivalence_max_delta
                ),
            }
        )
    return {
        "fixture": {
            "frequencies": GRAPH_FREQUENCIES.tolist(),
            "forcing_amplitude": GRAPH_FORCING_AMPLITUDE,
            "forcing_node": 0,
            "measurement_node": 4,
            "periods_per_cell": TOTAL_PERIODS,
            "steps_per_period": STEPS_PER_PERIOD,
            "measured_final_periods": MEASURE_PERIODS,
            "branch_initialization": (
                "all-zero at each direction endpoint; direction-local warm starts only"
            ),
            "forcing_phase_at_each_frequency_cell_radians": 0.0,
            "forcing_phase_convention": FORCING_PHASE_CONVENTION,
        },
        "configuration_totals": {
            name: _configuration_totals(system)
            for name, system in configurations.items()
        },
        "matching_boundary": {
            "single_full_physical_coefficients": (
                "matches two-sidecar total auxiliary mass, attachment stiffness, "
                "relative damping, and cubic coefficient; has one auxiliary mode"
            ),
            "two_colocated_half_parameters": (
                "matches distributed pair auxiliary-mode/state count and all "
                "attachment coefficient totals"
            ),
            "attachment_maps": "all Stage-A C rows are unit point selectors",
            "sidecar_matrices": "diagonal and local with no sidecar-sidecar coupling",
        },
        "branches": branches,
        "forward_reverse_discrepancies": discrepancies,
        "matched_comparison_cells": comparison_cells,
        "colocated_two_half_vs_single_full_equivalence": {
            "entire_grid_and_both_directions_checked": True,
            "maximum_fundamental_amplitude_delta": colocated_equivalence_max_delta,
            "maximum_local_mismatch_amplitude_delta_vs_single_full": (
                colocated_single_local_max_delta
            ),
            "maximum_local_mismatch_amplitude_delta_between_colocated_halves": (
                colocated_pair_local_max_delta
            ),
            "tolerance": colocated_equivalence_tolerance,
            "passed": colocated_equivalence_passed,
            "interpretation": (
                "a pass exposes ordinary co-located factorization equivalence; "
                "it is not evidence for a nonlinear-sidecar advantage"
            ),
        },
        "failures": failures,
        "rhs_evaluations": rhs_evaluations,
    }


def _self_rss_bytes() -> Tuple[int, int, str]:
    """Return current and OS-reported peak RSS for this child-free process."""

    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
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
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        success = psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        )
        if not success:
            error_code = ctypes.get_last_error()
            raise OSError(error_code, "GetProcessMemoryInfo failed")
        return (
            int(counters.WorkingSetSize),
            int(counters.PeakWorkingSetSize),
            "windows_GetProcessMemoryInfo_working_set",
        )
    import resource

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if os.uname().sysname == "Darwin":
        peak_bytes = peak
    else:
        peak_bytes = peak * 1024
    current_bytes = peak_bytes
    statm = Path("/proc/self/statm")
    if statm.is_file():
        fields = statm.read_text(encoding="ascii").split()
        current_bytes = int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    return current_bytes, peak_bytes, "stdlib_resource_ru_maxrss"


def run_stage_a(run_id: int) -> Dict[str, object]:
    """Execute the complete immutable RB-15 Stage-A fixture for run 7 or 11."""

    if type(run_id) is not int or run_id not in RUN_IDS:
        raise StageAValidationError("run_id must be exactly 7 or 11")
    started = time.perf_counter()
    initial_current_rss, initial_peak_rss, rss_method = _self_rss_bytes()
    linear = run_linear_limit_check(run_id)
    energy = run_energy_check(run_id)
    protected = run_protected_channel_check(run_id)
    scalar = run_scalar_sweeps()
    graph = run_graph_sweeps()
    wall_seconds = time.perf_counter() - started
    final_current_rss, final_peak_rss, final_rss_method = _self_rss_bytes()
    peak_rss = max(initial_peak_rss, final_peak_rss)
    failures: List[Dict[str, object]] = []
    for name, check in (
        ("linear_limit", linear),
        ("unforced_energy", energy),
        ("protected_channel", protected),
    ):
        if not check["passed"]:
            failures.append({"check": name, "reason": "FROZEN_TOLERANCE_FAILED"})
    failures.extend(scalar["failures"])
    failures.extend(graph["failures"])
    if peak_rss > MAX_RSS_BYTES:
        failures.append(
            {"check": "resource_rss", "reason": "SELF_PEAK_RSS_SCREEN_EXCEEDED"}
        )
    if wall_seconds > MAX_WALL_SECONDS:
        failures.append(
            {
                "check": "resource_duration",
                "reason": "COMPUTATION_DURATION_SCREEN_EXCEEDED",
            }
        )
    resource = {
        "computation_wall_seconds": wall_seconds,
        "computation_duration_screen_seconds": MAX_WALL_SECONDS,
        "computation_duration_screen_passed": wall_seconds <= MAX_WALL_SECONDS,
        "computation_duration_measurement_scope": (
            "post-hoc run_stage_a numerical-computation duration; excludes "
            "canonical serialization, atomic writes, and manifest verification"
        ),
        "deadline_enforced_during_integration": False,
        "initial_current_rss_bytes": initial_current_rss,
        "final_current_rss_bytes": final_current_rss,
        "self_peak_rss_bytes": peak_rss,
        "self_peak_rss_screen_bytes": MAX_RSS_BYTES,
        "self_peak_rss_screen_passed": peak_rss <= MAX_RSS_BYTES,
        "rss_measurement_method": final_rss_method,
        "rss_method_consistent": rss_method == final_rss_method,
        "child_processes_created": 0,
        "rss_measurement_scope": "self only; this harness creates no child process",
        "rhs_evaluations": int(
            linear["rhs_evaluations"]
            + energy["rhs_evaluations"]
            + protected["rhs_evaluations"]
            + scalar["rhs_evaluations"]
            + graph["rhs_evaluations"]
        ),
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "stage": "A",
        "run_id": run_id,
        "status": (
            "FROZEN_FAILURES_RETAINED"
            if failures
            else "EXECUTION_CONSISTENT_ON_FROZEN_FIXTURES"
        ),
        "evidence_mode": EVIDENCE_MODE,
        "scientific_claim_status": SCIENTIFIC_CLAIM_STATUS,
        "novelty_claim_status": NOVELTY_CLAIM_STATUS,
        "production_path_changed": False,
        "model_or_corpus_loaded": False,
        "arithmetic": "numpy_float64_and_cpython_binary64",
        "integrator": "fixed_step_classical_rk4",
        "episode_fixed_attachment_map": True,
        "host_operator_symmetric": True,
        "forced_grid_identity_across_run_ids": True,
        "protocol_disclosures": [
            (
                "The implementation reset forcing time/phase at every frequency "
                "cell while warm-starting state. This convention existed in source "
                "before official execution but was not separately recorded in the "
                "frozen prose, so forced-grid results are execution evidence rather "
                "than clean confirmatory preregistration."
            ),
            (
                "The 120-second and 256-MiB values are post-hoc numerical-computation "
                "and self-peak-RSS screens. They do not interrupt integration and do "
                "not cover artifact serialization, writing, or verification."
            ),
        ],
        "run_id_boundary": (
            "run 11 changes only nonzero four-node initial displacements by "
            "-0.8; forced scalar/graph grids are intentionally identical and "
            "are not an independent robustness replication"
        ),
        "metric_definitions": {
            "linear_error": "L2 norm of complete final RK4 state minus eigensolved state",
            "energy_increment": "max(0, E[n+1]-E[n]) sampled after every 0.001 RK4 step",
            "protected_leakage": (
                "L2 final protected displacement difference divided by matched "
                "reference L2 norm plus float64 epsilon"
            ),
            "harmonic_amplitude": (
                "least-squares columns [1,sin(h*omega*t),cos(h*omega*t)] "
                "over every final-20-period sample; amplitude=hypot(sin,cos)"
            ),
            "harmonic_residual": "root-mean-square residual of the same least-squares fit",
            "convergence_delta": (
                "absolute fundamental-amplitude difference between previous and "
                "last 10-period halves; reported without an invented threshold"
            ),
            "peak_policy": (
                "lowest frequency among amplitudes equal within absolute 1e-12; "
                "endpoint argmax is boundary-censored and is not a validated peak"
            ),
        },
        "linear_limit": linear,
        "unforced_energy": energy,
        "protected_channel": protected,
        "scalar_sweeps": scalar,
        "graph_sweeps": graph,
        "resource_usage": resource,
        "failures": failures,
        "failure_count": len(failures),
        "scope_limits": [
            "Stage A records only execution behavior on the exact frozen fixtures.",
            "A consistent execution does not prove the abstraction, general passivity, or source fidelity.",
            "The protected cell is an exact block-diagonal synthetic invariant.",
            "Downstream response amplitude/gain is not a physical wave transmission coefficient.",
            "No AI, RAG, scientific, product, production, or novelty claim is supported.",
            "The forcing-phase convention was not separately preregistered before the official outputs.",
            "Resource values are post-hoc screens, not cooperative hard-stop guards.",
        ],
    }


__all__ = [
    "Attachment",
    "GraphSystem",
    "StageAValidationError",
    "fit_harmonic",
    "graph_derivative",
    "grounded_path_stiffness",
    "rk4_step",
    "run_energy_check",
    "run_graph_sweeps",
    "run_linear_limit_check",
    "run_protected_channel_check",
    "run_scalar_sweeps",
    "run_stage_a",
    "total_energy",
]
