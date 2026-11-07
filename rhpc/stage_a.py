"""Deterministic constructed-mechanism experiment for RHPC Stage A.

The fixture deliberately separates three mechanical questions:

* can shared bases, coherent per-slice rotations, and sparse residuals exactly
  reconstruct a constructed expert bank;
* can position-permuted Fourier-holographic codes preserve an ordered route;
* can recurrent factorization lock a perturbed route while rejecting cycles
  and dead apparent stability.

Nothing in this module trains a model or establishes learned-model efficacy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from egv.canonical import canonical_bytes, digest_for

from .errors import RHPCConfigurationError, RHPCResultError


PROTOCOL_ID = "CHELATEDAI-RHPC-STAGE-A-v1"
RESULT_SCHEMA = "chelatedai-rhpc-stage-a-result-v1"
SCIENTIFIC_STATUS = "CONSTRUCTED_MECHANISM_ONLY"
NOVELTY_STATUS = "UNCONFIRMED"
PASS_STATUS = "SURVIVES_CONSTRUCTED_MECHANISM_GATES"
FAIL_STATUS = "DOES_NOT_SURVIVE_CONSTRUCTED_MECHANISM_GATES"

_CONFIG_FIELDS = frozenset(
    {
        "run_id",
        "stations",
        "experts",
        "matrix_dim",
        "rank",
        "slice_width",
        "code_dim",
        "trials",
        "max_iterations",
        "stable_iterations",
        "alignment_threshold",
        "coherence_threshold",
        "corruption_stride",
    }
)

_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "run_id",
        "claim_boundary",
        "scientific_status",
        "novelty_status",
        "promotion_eligible",
        "official_config",
        "config",
        "matrix_summary",
        "route_endpoints",
        "controls",
        "trials",
        "gates",
        "status",
    }
)

_PERMUTATION_A = (1, 3, 7, 9)
_PERMUTATION_B = (0, 127, 257, 509)
_ROOTS = np.asarray((1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j), dtype=np.complex128)


def _exact_int(value: Any, field: str, *, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RHPCConfigurationError(f"{field} must be an integer >= {minimum}")
    return value


def _finite_float(value: Any, field: str, *, lower: float = 0.0, upper: float = 1.0) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RHPCConfigurationError(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < lower or normalized > upper:
        raise RHPCConfigurationError(f"{field} must be finite in [{lower}, {upper}]")
    return normalized


@dataclass(frozen=True)
class StageAConfig:
    """Closed deterministic configuration for one Stage-A run."""

    run_id: int
    stations: int = 4
    experts: int = 6
    matrix_dim: int = 32
    rank: int = 16
    slice_width: int = 4
    code_dim: int = 1000
    trials: int = 256
    max_iterations: int = 8
    stable_iterations: int = 2
    alignment_threshold: float = 0.85
    coherence_threshold: float = 0.85
    corruption_stride: int = 20

    def __post_init__(self) -> None:
        _exact_int(self.run_id, "run_id", minimum=0)
        _exact_int(self.stations, "stations")
        _exact_int(self.experts, "experts", minimum=2)
        _exact_int(self.matrix_dim, "matrix_dim", minimum=4)
        _exact_int(self.rank, "rank", minimum=2)
        _exact_int(self.slice_width, "slice_width", minimum=2)
        _exact_int(self.code_dim, "code_dim", minimum=64)
        _exact_int(self.trials, "trials", minimum=4)
        _exact_int(self.max_iterations, "max_iterations", minimum=2)
        _exact_int(self.stable_iterations, "stable_iterations", minimum=2)
        _exact_int(self.corruption_stride, "corruption_stride", minimum=2)
        _finite_float(self.alignment_threshold, "alignment_threshold")
        _finite_float(self.coherence_threshold, "coherence_threshold")
        frozen = {
            "stations": (self.stations, 4),
            "experts": (self.experts, 6),
            "matrix_dim": (self.matrix_dim, 32),
            "rank": (self.rank, 16),
            "slice_width": (self.slice_width, 4),
            "code_dim": (self.code_dim, 1000),
            "max_iterations": (self.max_iterations, 8),
            "stable_iterations": (self.stable_iterations, 2),
            "alignment_threshold": (self.alignment_threshold, 0.85),
            "coherence_threshold": (self.coherence_threshold, 0.85),
            "corruption_stride": (self.corruption_stride, 20),
        }
        changed = [name for name, (observed, expected) in frozen.items() if observed != expected]
        if changed:
            raise RHPCConfigurationError("Stage-A frozen parameters changed: " + ", ".join(changed))
        for station, a_value in enumerate(_PERMUTATION_A):
            if math.gcd(a_value, self.code_dim) != 1:
                raise RHPCConfigurationError(f"station {station} affine multiplier is not invertible")

    @property
    def official(self) -> bool:
        return self.trials == 256 and self.run_id in (7, 11)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StageAConfig":
        if not isinstance(value, Mapping) or set(value) != _CONFIG_FIELDS:
            raise RHPCConfigurationError("Stage-A configuration is not a closed schema")
        return cls(**dict(value))


def _rounded(value: float) -> float:
    if not math.isfinite(float(value)):
        raise RHPCResultError("non-finite metric produced")
    rounded = round(float(value), 15)
    return 0.0 if rounded == 0.0 else rounded


def _walsh_hadamard4() -> np.ndarray:
    return 0.5 * np.asarray(
        (
            (1.0, 1.0, 1.0, 1.0),
            (1.0, -1.0, 1.0, -1.0),
            (1.0, 1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0, 1.0),
        ),
        dtype=np.float64,
    )


def _coordinate_basis(matrix_dim: int, rank: int, station: int, side: int) -> np.ndarray:
    step = 5 if side == 0 else 7
    offset = station * (3 if side == 0 else 5) + side
    columns = [((offset + step * index) % matrix_dim) for index in range(rank)]
    if len(set(columns)) != rank:
        raise RHPCResultError("constructed coordinate basis contains duplicate columns")
    return np.eye(matrix_dim, dtype=np.float64)[:, columns]


def _slice_rotation(config: StageAConfig, station: int, expert: int) -> np.ndarray:
    base = _walsh_hadamard4()
    blocks: List[np.ndarray] = []
    for slice_id in range(config.rank // config.slice_width):
        shift = (station + expert + slice_id) % config.slice_width
        block = np.roll(base, shift=shift, axis=1)
        signs = np.asarray(
            [1.0 if ((station + expert + slice_id + index) % 2 == 0) else -1.0 for index in range(4)],
            dtype=np.float64,
        )
        block = signs[:, None] * block
        blocks.append(block)
    rotation = np.zeros((config.rank, config.rank), dtype=np.float64)
    for block_id, block in enumerate(blocks):
        start = block_id * config.slice_width
        rotation[start : start + config.slice_width, start : start + config.slice_width] = block
    identity = rotation.T @ rotation
    if not np.array_equal(identity, np.eye(config.rank, dtype=np.float64)):
        raise RHPCResultError("constructed slice rotation is not exactly orthogonal")
    return rotation


def _spectrum(config: StageAConfig, station: int, expert: int) -> np.ndarray:
    return np.asarray(
        [0.55 + 0.025 * ((index * 3 + expert * 5 + station * 7) % 17) for index in range(config.rank)],
        dtype=np.float64,
    )


def _sparse_residual(config: StageAConfig, station: int, expert: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    residual = np.zeros((config.matrix_dim, config.matrix_dim), dtype=np.float64)
    records: List[Dict[str, Any]] = []
    occupied = set()
    cursor = 0
    while len(records) < 6:
        row = (station * 11 + expert * 7 + cursor * 5 + 3) % config.matrix_dim
        column = (station * 13 + expert * 3 + cursor * 9 + 1) % config.matrix_dim
        cursor += 1
        if (row, column) in occupied:
            continue
        occupied.add((row, column))
        sign = 1.0 if ((station + expert + len(records)) % 2 == 0) else -1.0
        value = sign * (0.0125 + 0.0025 * ((station + expert + len(records)) % 3))
        residual[row, column] = value
        records.append({"row": row, "column": column, "value": _rounded(value)})
    return residual, records


def _relative_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise RHPCResultError("constructed matrix has invalid norm")
    return _rounded(float(np.linalg.norm(reference - candidate)) / denominator)


def _expert_fixture(config: StageAConfig, station: int, expert: int) -> Dict[str, Any]:
    left = _coordinate_basis(config.matrix_dim, config.rank, station, 0)
    right = _coordinate_basis(config.matrix_dim, config.rank, station, 1)
    rotation = _slice_rotation(config, station, expert)
    spectrum = _spectrum(config, station, expert)
    residual, residual_records = _sparse_residual(config, station, expert)
    core = left @ rotation @ np.diag(spectrum) @ rotation.T @ right.T
    matrix = core + residual
    reconstructed = left @ rotation @ np.diag(spectrum) @ rotation.T @ right.T + residual
    core_coordinates = left.T @ (matrix - residual) @ right
    no_rotation = left @ np.diag(np.diag(core_coordinates)) @ right.T + residual
    no_residual = core
    rotation_payload = {
        "station": station,
        "expert": expert,
        "rotation": rotation.tolist(),
        "spectrum": spectrum.tolist(),
    }
    residual_payload = {"station": station, "expert": expert, "entries": residual_records}
    return {
        "station": station,
        "expert": expert,
        "basis_id": digest_for({"station": station, "left": left.tolist(), "right": right.tolist()}),
        "rotation_id": digest_for(rotation_payload),
        "residual_code_id": digest_for(residual_payload),
        "full_relative_error": _relative_error(matrix, reconstructed),
        "no_rotation_relative_error": _relative_error(matrix, no_rotation),
        "no_residual_relative_error": _relative_error(matrix, no_residual),
    }


def _matrix_summary(config: StageAConfig) -> Tuple[Dict[str, Any], Dict[Tuple[int, int], Dict[str, Any]]]:
    experts = [
        _expert_fixture(config, station, expert)
        for station in range(config.stations)
        for expert in range(config.experts)
    ]
    lookup = {(row["station"], row["expert"]): row for row in experts}
    full = [row["full_relative_error"] for row in experts]
    no_rotation = [row["no_rotation_relative_error"] for row in experts]
    no_residual = [row["no_residual_relative_error"] for row in experts]
    return (
        {
            "expert_count": len(experts),
            "experts": experts,
            "maximum_full_relative_error": _rounded(max(full)),
            "mean_no_rotation_relative_error": _rounded(float(np.mean(no_rotation))),
            "mean_no_residual_relative_error": _rounded(float(np.mean(no_residual))),
        },
        lookup,
    )


def _root_code(symbol: str, dimension: int) -> np.ndarray:
    values: List[int] = []
    block = 0
    while len(values) < dimension:
        payload = hashlib.sha256(f"{symbol}:{block}".encode("utf-8")).digest()
        for byte in payload:
            for shift in (0, 2, 4, 6):
                values.append((byte >> shift) & 0x03)
                if len(values) == dimension:
                    break
            if len(values) == dimension:
                break
        block += 1
    return _ROOTS[np.asarray(values, dtype=np.int8)]


def _codebook(config: StageAConfig) -> np.ndarray:
    return np.stack([_root_code(f"rhpc-expert-{expert}", config.code_dim) for expert in range(config.experts)])


def _permutation_parameters(config: StageAConfig, *, ordered: bool) -> Tuple[Tuple[int, int], ...]:
    if not ordered:
        return tuple((1, 0) for _ in range(config.stations))
    return tuple(
        (_PERMUTATION_A[station], _PERMUTATION_B[station] % config.code_dim)
        for station in range(config.stations)
    )


def _permute(code: np.ndarray, parameters: Tuple[int, int]) -> np.ndarray:
    a_value, b_value = parameters
    indices = (a_value * np.arange(code.size, dtype=np.int64) + b_value) % code.size
    return code[indices]


def _inverse_permute(code: np.ndarray, parameters: Tuple[int, int]) -> np.ndarray:
    a_value, b_value = parameters
    indices = (a_value * np.arange(code.size, dtype=np.int64) + b_value) % code.size
    restored = np.empty_like(code)
    restored[indices] = code
    return restored


def _encode_path(
    path: Sequence[int],
    codebook: np.ndarray,
    parameters: Sequence[Tuple[int, int]],
) -> np.ndarray:
    if len(path) != len(parameters):
        raise RHPCResultError("path length differs from station count")
    encoded = np.ones(codebook.shape[1], dtype=np.complex128)
    for station, expert in enumerate(path):
        encoded *= _permute(codebook[int(expert)], parameters[station])
    return encoded


def _code_digest(code: np.ndarray) -> str:
    distances = np.abs(code[:, None] - _ROOTS[None, :])
    indices = np.argmin(distances, axis=1).astype(np.uint8)
    if not np.all(np.min(distances, axis=1) <= 1e-12):
        raise RHPCResultError("holographic code left the root-of-unity alphabet")
    return hashlib.sha256(indices.tobytes()).hexdigest()


def _corrupt_code(code: np.ndarray, config: StageAConfig, trial_index: int) -> np.ndarray:
    corrupted = np.array(code, copy=True)
    residue = (config.run_id * 7 + trial_index * 11) % config.corruption_stride
    indices = residue + config.corruption_stride * np.arange(config.code_dim // config.corruption_stride)
    corrupted[indices] *= -1.0
    return corrupted


def _route_from_trial(config: StageAConfig, trial_index: int) -> Tuple[int, ...]:
    universe = config.experts ** config.stations
    encoded = (config.run_id * 97 + trial_index * 313) % universe
    route: List[int] = []
    for _station in range(config.stations):
        route.append(encoded % config.experts)
        encoded //= config.experts
    return tuple(route)


def _initial_hint(config: StageAConfig, trial_index: int, route: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    candidates = list(route)
    margins = [0.75] * config.stations
    if trial_index % 4:
        station = (trial_index // 4) % config.stations
        offset = 1 + ((trial_index // 16) % (config.experts - 1))
        candidates[station] = (route[station] + offset) % config.experts
        margins[station] = 0.02
    return tuple(candidates), tuple(margins)


def _alignment(target: np.ndarray, reconstruction: np.ndarray) -> Tuple[float, float]:
    cross_phase = target * np.conjugate(reconstruction)
    mean_phase = np.mean(cross_phase)
    return _rounded(float(np.real(mean_phase))), _rounded(float(np.abs(mean_phase)))


def _trace_rows(
    paths: Sequence[Sequence[int]],
    target: np.ndarray,
    codebook: np.ndarray,
    parameters: Sequence[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for iteration, path in enumerate(paths):
        reconstruction = _encode_path(path, codebook, parameters)
        alignment, coherence = _alignment(target, reconstruction)
        rows.append(
            {
                "iteration": iteration,
                "path": [int(item) for item in path],
                "alignment": alignment,
                "coherence": coherence,
                "phase_residual": _rounded(1.0 - coherence),
            }
        )
    return rows


def classify_lock_trace(
    trace: Sequence[Sequence[int]],
    target: np.ndarray,
    codebook: np.ndarray,
    parameters: Sequence[Tuple[int, int]],
    *,
    stable_iterations: int,
    alignment_threshold: float,
    coherence_threshold: float,
) -> Dict[str, Any]:
    """Classify a candidate trace without letting standing cycles masquerade as locks."""

    if not trace:
        raise RHPCResultError("lock trace cannot be empty")
    normalized = [tuple(int(item) for item in path) for path in trace]
    expected_length = len(parameters)
    if any(len(path) != expected_length for path in normalized):
        raise RHPCResultError("lock trace contains a path with the wrong station count")
    seen: Dict[Tuple[int, ...], int] = {}
    cycle_period: Optional[int] = None
    for index, path in enumerate(normalized):
        previous = seen.get(path)
        if previous is not None and index - previous > 1:
            cycle_period = index - previous
            break
        seen[path] = index
    final_path = normalized[-1]
    reconstruction = _encode_path(final_path, codebook, parameters)
    alignment, coherence = _alignment(target, reconstruction)
    if cycle_period is not None:
        state = "CYCLE"
    else:
        stable_count = 1
        for index in range(len(normalized) - 2, -1, -1):
            if normalized[index] != final_path:
                break
            stable_count += 1
        if (
            len(normalized) - 1 >= stable_iterations
            and stable_count >= stable_iterations
            and alignment >= alignment_threshold
            and coherence >= coherence_threshold
        ):
            state = "LOCKED"
        else:
            state = "UNRESOLVED"
    return {
        "state": state,
        "final_path": list(final_path),
        "iterations": max(0, len(normalized) - 1),
        "alignment": alignment,
        "coherence": coherence,
        "cycle_period": cycle_period,
        "trace": _trace_rows(normalized, target, codebook, parameters),
    }


def _resolve_route(
    target: np.ndarray,
    codebook: np.ndarray,
    parameters: Sequence[Tuple[int, int]],
    initial: Tuple[int, ...],
    margins: Tuple[float, ...],
    config: StageAConfig,
) -> Dict[str, Any]:
    candidates = list(initial)
    trace: List[Tuple[int, ...]] = [tuple(candidates)]
    update_order = sorted(range(config.stations), key=lambda station: (margins[station], station))
    for _iteration in range(config.max_iterations):
        for station in update_order:
            other_product = np.ones(config.code_dim, dtype=np.complex128)
            for other_station, other_expert in enumerate(candidates):
                if other_station == station:
                    continue
                other_product *= _permute(codebook[other_expert], parameters[other_station])
            probe_positioned = target * np.conjugate(other_product)
            probe = _inverse_permute(probe_positioned, parameters[station])
            similarities = np.real(codebook.conjugate() @ probe) / config.code_dim
            candidates[station] = int(np.argmax(similarities))
        trace.append(tuple(candidates))
        classified = classify_lock_trace(
            trace,
            target,
            codebook,
            parameters,
            stable_iterations=config.stable_iterations,
            alignment_threshold=config.alignment_threshold,
            coherence_threshold=config.coherence_threshold,
        )
        if classified["state"] in {"LOCKED", "CYCLE"}:
            return classified
    return classify_lock_trace(
        trace,
        target,
        codebook,
        parameters,
        stable_iterations=config.stable_iterations,
        alignment_threshold=config.alignment_threshold,
        coherence_threshold=config.coherence_threshold,
    )


def _method_metrics(records: Sequence[Tuple[Tuple[int, ...], Mapping[str, Any]]]) -> Dict[str, Any]:
    total = len(records)
    if total == 0:
        raise RHPCResultError("method metric set cannot be empty")
    correct = 0
    locked = 0
    correct_locked = 0
    false_locked = 0
    cycles = 0
    iterations: List[int] = []
    phase_residual_attenuations: List[float] = []
    for truth, result in records:
        final_path = tuple(result["final_path"])
        is_correct = final_path == truth
        correct += int(is_correct)
        if result["state"] == "LOCKED":
            locked += 1
            correct_locked += int(is_correct)
            false_locked += int(not is_correct)
            iterations.append(int(result["iterations"]))
        cycles += int(result["state"] == "CYCLE")
        trace = result.get("trace")
        if not isinstance(trace, list) or not trace:
            raise RHPCResultError("method result lacks a phase trace")
        phase_residual_attenuations.append(
            float(trace[0]["phase_residual"]) - float(trace[-1]["phase_residual"])
        )
    return {
        "trial_count": total,
        "path_accuracy": _rounded(correct / total),
        "lock_coverage": _rounded(locked / total),
        "lock_precision": _rounded(correct_locked / locked) if locked else 0.0,
        "false_lock_rate": _rounded(false_locked / total),
        "cycle_rate": _rounded(cycles / total),
        "mean_lock_iterations": _rounded(float(np.mean(iterations))) if iterations else None,
        "mean_phase_residual_attenuation": _rounded(float(np.mean(phase_residual_attenuations))),
    }


def _one_shot_metrics(records: Sequence[Tuple[Tuple[int, ...], Tuple[int, ...]]]) -> Dict[str, Any]:
    total = len(records)
    return {
        "trial_count": total,
        "path_accuracy": _rounded(sum(int(truth == candidate) for truth, candidate in records) / total),
    }


def _gate(name: str, observed: float, comparator: str, threshold: float, passed: bool) -> Dict[str, Any]:
    return {
        "name": name,
        "observed": _rounded(observed),
        "comparator": comparator,
        "threshold": _rounded(threshold),
        "passed": bool(passed),
    }


def _build_stage_a(config: StageAConfig) -> Dict[str, Any]:
    matrix_summary, matrix_lookup = _matrix_summary(config)
    codes = _codebook(config)
    ordered_parameters = _permutation_parameters(config, ordered=True)
    orderless_parameters = _permutation_parameters(config, ordered=False)

    full_records: List[Tuple[Tuple[int, ...], Mapping[str, Any]]] = []
    orderless_records: List[Tuple[Tuple[int, ...], Mapping[str, Any]]] = []
    smoothing_records: List[Tuple[Tuple[int, ...], Mapping[str, Any]]] = []
    dead_records: List[Tuple[Tuple[int, ...], Mapping[str, Any]]] = []
    one_shot_records: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    trials: List[Dict[str, Any]] = []

    for trial_index in range(config.trials):
        truth = _route_from_trial(config, trial_index)
        initial, margins = _initial_hint(config, trial_index, truth)
        clean = _encode_path(truth, codes, ordered_parameters)
        observed = _corrupt_code(clean, config, trial_index)
        full = _resolve_route(observed, codes, ordered_parameters, initial, margins, config)
        orderless = _resolve_route(observed, codes, orderless_parameters, initial, margins, config)
        smoothing = classify_lock_trace(
            [initial] * (config.stable_iterations + 1),
            observed,
            codes,
            ordered_parameters,
            stable_iterations=config.stable_iterations,
            alignment_threshold=config.alignment_threshold,
            coherence_threshold=config.coherence_threshold,
        )
        dead_path = tuple(0 for _ in range(config.stations))
        dead = classify_lock_trace(
            [dead_path] * (config.stable_iterations + 1),
            observed,
            codes,
            ordered_parameters,
            stable_iterations=config.stable_iterations,
            alignment_threshold=config.alignment_threshold,
            coherence_threshold=config.coherence_threshold,
        )
        full_records.append((truth, full))
        orderless_records.append((truth, orderless))
        smoothing_records.append((truth, smoothing))
        dead_records.append((truth, dead))
        one_shot_records.append((truth, initial))
        specifier_chain = []
        for station, expert in enumerate(truth):
            expert_record = matrix_lookup[(station, expert)]
            specifier_chain.append(
                {
                    "station": station,
                    "expert": expert,
                    "basis_id": expert_record["basis_id"],
                    "rotation_id": expert_record["rotation_id"],
                    "slice_ids": list(range(config.rank // config.slice_width)),
                    "residual_code_id": expert_record["residual_code_id"],
                }
            )
        trials.append(
            {
                "trial_id": f"RHPC-A-{config.run_id:03d}-{trial_index:04d}",
                "true_path": list(truth),
                "initial_path": list(initial),
                "hint_margins": [_rounded(value) for value in margins],
                "observed_code_digest": _code_digest(observed),
                "specifier_chain": specifier_chain,
                "full": full,
                "orderless": orderless,
                "smoothing": smoothing,
                "dead": dead,
            }
        )

    full_metrics = _method_metrics(full_records)
    orderless_metrics = _method_metrics(orderless_records)
    smoothing_metrics = _method_metrics(smoothing_records)
    dead_metrics = _method_metrics(dead_records)
    one_shot_metrics = _one_shot_metrics(one_shot_records)

    cycle_truth = (0, 0, 0, 0)
    cycle_target = _encode_path(cycle_truth, codes, ordered_parameters)
    injected_cycle = classify_lock_trace(
        (cycle_truth, (1, 0, 0, 0), cycle_truth),
        cycle_target,
        codes,
        ordered_parameters,
        stable_iterations=config.stable_iterations,
        alignment_threshold=config.alignment_threshold,
        coherence_threshold=config.coherence_threshold,
    )

    gates = [
        _gate(
            "exact_matrix_reconstruction",
            matrix_summary["maximum_full_relative_error"],
            "<=",
            1e-12,
            matrix_summary["maximum_full_relative_error"] <= 1e-12,
        ),
        _gate(
            "rotation_is_load_bearing",
            matrix_summary["mean_no_rotation_relative_error"],
            ">=",
            0.01,
            matrix_summary["mean_no_rotation_relative_error"] >= 0.01,
        ),
        _gate(
            "residual_is_load_bearing",
            matrix_summary["mean_no_residual_relative_error"],
            ">=",
            0.001,
            matrix_summary["mean_no_residual_relative_error"] >= 0.001,
        ),
        _gate("recurrent_path_accuracy", full_metrics["path_accuracy"], ">=", 0.95, full_metrics["path_accuracy"] >= 0.95),
        _gate(
            "recurrent_advantage_over_one_shot",
            full_metrics["path_accuracy"] - one_shot_metrics["path_accuracy"],
            ">=",
            0.25,
            full_metrics["path_accuracy"] - one_shot_metrics["path_accuracy"] >= 0.25,
        ),
        _gate(
            "ordered_binding_advantage",
            full_metrics["path_accuracy"] - orderless_metrics["path_accuracy"],
            ">=",
            0.25,
            full_metrics["path_accuracy"] - orderless_metrics["path_accuracy"] >= 0.25,
        ),
        _gate("recurrent_lock_coverage", full_metrics["lock_coverage"], ">=", 0.90, full_metrics["lock_coverage"] >= 0.90),
        _gate("recurrent_lock_precision", full_metrics["lock_precision"], ">=", 0.99, full_metrics["lock_precision"] >= 0.99),
        _gate("recurrent_false_lock_rate", full_metrics["false_lock_rate"], "<=", 0.01, full_metrics["false_lock_rate"] <= 0.01),
        _gate("dead_lock_coverage", dead_metrics["lock_coverage"], "<=", 0.01, dead_metrics["lock_coverage"] <= 0.01),
        _gate(
            "recurrent_phase_residual_attenuation",
            full_metrics["mean_phase_residual_attenuation"],
            ">=",
            0.50,
            full_metrics["mean_phase_residual_attenuation"] >= 0.50,
        ),
        {
            "name": "standing_cycle_rejected",
            "observed": injected_cycle["state"],
            "comparator": "==",
            "threshold": "CYCLE",
            "passed": injected_cycle["state"] == "CYCLE",
        },
    ]
    status = PASS_STATUS if all(gate["passed"] for gate in gates) else FAIL_STATUS
    return {
        "schema_version": RESULT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "run_id": config.run_id,
        "claim_boundary": "constructed mechanism only; no learned-model efficacy or promotion claim",
        "scientific_status": SCIENTIFIC_STATUS,
        "novelty_status": NOVELTY_STATUS,
        "promotion_eligible": False,
        "official_config": config.official,
        "config": config.to_dict(),
        "matrix_summary": matrix_summary,
        "route_endpoints": {
            "full_recurrent": full_metrics,
            "one_shot": one_shot_metrics,
            "orderless_recurrent": orderless_metrics,
        },
        "controls": {
            "smoothing": smoothing_metrics,
            "dead_lock": dead_metrics,
            "injected_cycle": injected_cycle,
        },
        "trials": trials,
        "gates": gates,
        "status": status,
    }


def run_stage_a(config: StageAConfig) -> Dict[str, Any]:
    """Run one deterministic constructed-mechanism experiment."""

    if not isinstance(config, StageAConfig):
        raise RHPCConfigurationError("run_stage_a requires StageAConfig")
    return _build_stage_a(config)


def validate_stage_a_result(value: Mapping[str, Any], *, regenerate: bool = True) -> Dict[str, Any]:
    """Validate the closed result and optionally reproduce every canonical byte."""

    if not isinstance(value, Mapping) or set(value) != _RESULT_FIELDS:
        raise RHPCResultError("Stage-A result is not a closed schema")
    if value.get("schema_version") != RESULT_SCHEMA or value.get("protocol_id") != PROTOCOL_ID:
        raise RHPCResultError("Stage-A result identity is unsupported")
    if value.get("scientific_status") != SCIENTIFIC_STATUS or value.get("novelty_status") != NOVELTY_STATUS:
        raise RHPCResultError("Stage-A claim status changed")
    if value.get("promotion_eligible") is not False:
        raise RHPCResultError("Stage-A result cannot be promotion eligible")
    config = StageAConfig.from_mapping(value.get("config"))
    if value.get("run_id") != config.run_id or value.get("official_config") is not config.official:
        raise RHPCResultError("Stage-A run/config identity mismatch")
    gates = value.get("gates")
    if not isinstance(gates, list) or len(gates) != 12 or any(not isinstance(item, Mapping) for item in gates):
        raise RHPCResultError("Stage-A gate vector is malformed")
    expected_status = PASS_STATUS if all(item.get("passed") is True for item in gates) else FAIL_STATUS
    if value.get("status") != expected_status:
        raise RHPCResultError("Stage-A status contradicts its gate vector")
    if regenerate:
        expected = _build_stage_a(config)
        if canonical_bytes(value) != canonical_bytes(expected):
            raise RHPCResultError("Stage-A result does not exactly regenerate from its frozen configuration")
    return dict(value)


__all__ = [
    "FAIL_STATUS",
    "NOVELTY_STATUS",
    "PASS_STATUS",
    "PROTOCOL_ID",
    "RESULT_SCHEMA",
    "SCIENTIFIC_STATUS",
    "StageAConfig",
    "classify_lock_trace",
    "run_stage_a",
    "validate_stage_a_result",
]
