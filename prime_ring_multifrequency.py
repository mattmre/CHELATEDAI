"""Finite, resource-guarded diagnostics for PRW-H1.

This module compares selected Fourier-phase bins with matched controls for
recovering one shared cyclic shift.  Multi-frequency phase synchronization is
known mathematics; this code is a bounded falsification harness, not a novelty
or production claim.

All phase conditions search exactly ``type_count * prime`` states.  A decoder
must never choose an independent shift per frequency bin, because that would
inflate the state space to ``type_count * prime**bin_count``.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from prime_ring_waypoint import is_prime_exact, legendre_carrier


MIB = 1024 * 1024

HARD_MAX_ESTIMATED_BYTES = 512 * MIB
HARD_MAX_SECONDS = 120.0
HARD_MAX_WORK_UNITS = 50_000_000
HARD_MAX_HYPOTHESES = 1024
HARD_MAX_EXACT_PATTERNS = 1 << 20
HARD_MAX_PRIME = 31
HARD_MAX_TYPES = 32
HARD_MAX_NODES = 12
HARD_MAX_CONDITIONS = 8
FROZEN_CONTROL_CONDITION_COUNT = 8

DEFAULT_MAX_ESTIMATED_BYTES = 64 * MIB
DEFAULT_MAX_SECONDS = 10.0
DEFAULT_MAX_WORK_UNITS = 2_000_000
DEFAULT_MAX_HYPOTHESES = 256
DEFAULT_MAX_EXACT_PATTERNS = 4096

EVIDENCE_STATUS = "FINITE_SYNTHETIC_FOURIER_PHASE_DIAGNOSTIC_ONLY"


class PRWH1ValidationError(ValueError):
    """Raised when a PRW-H1 contract is malformed."""


class PRWH1DegenerateFeatureError(PRWH1ValidationError):
    """Raised when a valid decoder condition has no usable feature energy."""


class PRWH1ResourceError(RuntimeError):
    """Raised before an analysis exceeds a frozen resource budget."""


def _plain_int(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise PRWH1ValidationError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise PRWH1ValidationError(f"{name} must be >= {minimum}")
    return result


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise PRWH1ValidationError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise PRWH1ValidationError(f"{name} must be a finite positive number")
    return result


def _crossover(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise PRWH1ValidationError("crossover must be a finite probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 0.5:
        raise PRWH1ValidationError("crossover must be in [0, 0.5)")
    return result


@dataclass(frozen=True)
class MultifrequencyBudget:
    """Caller-lowerable limits beneath immutable PRW-H1 hard ceilings."""

    max_estimated_bytes: int = DEFAULT_MAX_ESTIMATED_BYTES
    max_seconds: float = DEFAULT_MAX_SECONDS
    max_work_units: int = DEFAULT_MAX_WORK_UNITS
    max_hypotheses: int = DEFAULT_MAX_HYPOTHESES
    max_exact_patterns: int = DEFAULT_MAX_EXACT_PATTERNS
    max_prime: int = HARD_MAX_PRIME
    max_types: int = HARD_MAX_TYPES
    max_nodes: int = HARD_MAX_NODES
    max_conditions: int = HARD_MAX_CONDITIONS

    def __post_init__(self) -> None:
        integer_caps = {
            "max_estimated_bytes": HARD_MAX_ESTIMATED_BYTES,
            "max_work_units": HARD_MAX_WORK_UNITS,
            "max_hypotheses": HARD_MAX_HYPOTHESES,
            "max_exact_patterns": HARD_MAX_EXACT_PATTERNS,
            "max_prime": HARD_MAX_PRIME,
            "max_types": HARD_MAX_TYPES,
            "max_nodes": HARD_MAX_NODES,
            "max_conditions": HARD_MAX_CONDITIONS,
        }
        for name, hard_cap in integer_caps.items():
            value = _plain_int(getattr(self, name), name, 1)
            if value > hard_cap:
                raise PRWH1ValidationError(
                    f"{name} cannot exceed the hard ceiling {hard_cap}"
                )
        seconds = _positive_float(self.max_seconds, "max_seconds")
        if seconds > HARD_MAX_SECONDS:
            raise PRWH1ValidationError(
                f"max_seconds cannot exceed the hard ceiling {HARD_MAX_SECONDS}"
            )


@dataclass(frozen=True)
class FrequencyCondition:
    """One decoder condition with an explicit coefficient/search contract."""

    name: str
    mode: str
    bins: Tuple[int, ...] = ()
    seed: Optional[int] = None
    phase_bits: Optional[int] = None
    quantizer_origin_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise PRWH1ValidationError("condition name must be non-empty")
        if self.mode not in {
            "phase",
            "magnitude",
            "randomized_phase",
            "time_domain",
        }:
            raise PRWH1ValidationError(
                "condition mode must be phase, magnitude, randomized_phase, "
                "or time_domain"
            )
        if not isinstance(self.bins, tuple):
            raise PRWH1ValidationError(
                "condition bins must be a bounded tuple"
            )
        if len(self.bins) > (HARD_MAX_PRIME - 1) // 2:
            raise PRWH1ResourceError(
                "condition bin count exceeds the bounded PRW-H1 limit"
            )
        normalized_bins = tuple(
            _plain_int(value, "frequency bin", 1) for value in self.bins
        )
        if self.mode == "time_domain":
            if normalized_bins:
                raise PRWH1ValidationError(
                    "time_domain conditions must not declare frequency bins"
                )
        elif not normalized_bins:
            raise PRWH1ValidationError(
                "Fourier conditions must declare at least one frequency bin"
            )
        if self.mode == "randomized_phase" and self.seed is None:
            raise PRWH1ValidationError(
                "randomized_phase conditions require a deterministic seed"
            )
        if self.mode != "randomized_phase" and self.seed is not None:
            raise PRWH1ValidationError(
                "only randomized_phase conditions may declare a seed"
            )
        normalized_seed = (
            _plain_int(self.seed, "seed", 0)
            if self.seed is not None
            else None
        )
        normalized_bits = None
        if self.phase_bits is not None:
            normalized_bits = _plain_int(
                self.phase_bits, "phase_bits", 1
            )
            if normalized_bits > 16:
                raise PRWH1ValidationError("phase_bits must be <= 16")
        origin = self.quantizer_origin_fraction
        if isinstance(origin, (bool, np.bool_)) or not isinstance(
            origin, (int, float, np.integer, np.floating)
        ):
            raise PRWH1ValidationError(
                "quantizer_origin_fraction must be finite in [0, 1)"
            )
        origin = float(origin)
        if not math.isfinite(origin) or not 0.0 <= origin < 1.0:
            raise PRWH1ValidationError(
                "quantizer_origin_fraction must be finite in [0, 1)"
            )
        if self.phase_bits is None and origin != 0.0:
            raise PRWH1ValidationError(
                "quantizer origin is only defined when phase_bits is set"
            )
        if (
            self.mode in {"magnitude", "time_domain"}
            and normalized_bits is not None
        ):
            raise PRWH1ValidationError(
                f"{self.mode} conditions cannot apply phase quantization"
            )
        object.__setattr__(self, "bins", normalized_bins)
        object.__setattr__(self, "seed", normalized_seed)
        object.__setattr__(self, "phase_bits", normalized_bits)
        object.__setattr__(self, "quantizer_origin_fraction", origin)


@dataclass(frozen=True)
class MultifrequencyBank:
    """A small real-valued type/node/ring template bank."""

    prime: int
    templates: np.ndarray
    phase_signatures: np.ndarray
    overlap_one_verified: bool
    legendre_equal_nonzero_magnitudes_verified: bool
    build_estimated_peak_bytes: int
    build_estimated_work_units: int

    @property
    def type_count(self) -> int:
        return int(self.templates.shape[0])

    @property
    def node_count(self) -> int:
        return int(self.templates.shape[1])

    @property
    def hypothesis_count(self) -> int:
        return self.type_count * self.prime


@dataclass(frozen=True)
class DecodeResult:
    """One bounded decoder result."""

    condition: FrequencyCondition
    scores: np.ndarray
    top_states: Tuple[Tuple[int, int], ...]
    unique_winner: bool
    abstained: bool
    abstention_reason: Optional[str]
    best_score: Optional[float]
    runner_up_score: Optional[float]
    margin: Optional[float]
    candidate_count: int
    accounting: Mapping[str, Any]


def _start_deadline(budget: MultifrequencyBudget) -> float:
    return time.monotonic() + budget.max_seconds


def _check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise PRWH1ResourceError(
            "finite PRW-H1 calculation exceeded its wall-clock deadline"
        )


def _array_shape(
    value: object,
    name: str,
    dimensions: int,
) -> Tuple[int, ...]:
    """Read only declared ndarray shape before any normalization/allocation."""

    if not isinstance(value, np.ndarray):
        raise PRWH1ValidationError(
            f"{name} must be a numpy array so shape can be preflighted"
        )
    if value.ndim != dimensions or any(int(size) <= 0 for size in value.shape):
        raise PRWH1ValidationError(
            f"{name} must be a non-empty {dimensions}-dimensional array"
        )
    return tuple(int(size) for size in value.shape)


def _enforce_dimensions(
    *,
    prime: int,
    types: int,
    nodes: int,
    budget: MultifrequencyBudget,
) -> None:
    if prime > budget.max_prime:
        raise PRWH1ResourceError(
            f"prime exceeds bounded PRW-H1 limit: {prime} > {budget.max_prime}"
        )
    if types > budget.max_types:
        raise PRWH1ResourceError(
            f"type count exceeds bounded PRW-H1 limit: {types} > {budget.max_types}"
        )
    if nodes > budget.max_nodes:
        raise PRWH1ResourceError(
            f"node count exceeds bounded PRW-H1 limit: {nodes} > {budget.max_nodes}"
        )
    hypotheses = types * prime
    if hypotheses > budget.max_hypotheses:
        raise PRWH1ResourceError(
            "hypothesis count exceeds bounded PRW-H1 limit: "
            f"{hypotheses} > {budget.max_hypotheses}"
        )


def _enforce_resource_estimate(
    estimated_bytes: int,
    estimated_work: int,
    budget: MultifrequencyBudget,
) -> None:
    if estimated_bytes > budget.max_estimated_bytes:
        raise PRWH1ResourceError(
            "estimated peak bytes exceed bounded PRW-H1 limit: "
            f"{estimated_bytes} > {budget.max_estimated_bytes}"
        )
    if estimated_work > budget.max_work_units:
        raise PRWH1ResourceError(
            "estimated work exceeds bounded PRW-H1 limit: "
            f"{estimated_work} > {budget.max_work_units}"
        )


def _phase_matrix_shape(
    phase_signatures: object,
) -> Tuple[int, int]:
    return _array_shape(phase_signatures, "phase_signatures", 2)


def estimate_bank_peak_bytes(
    *,
    prime: int,
    type_count: int,
    node_count: int,
) -> int:
    """Conservative co-resident bytes for bank construction."""

    p = _plain_int(prime, "prime", 3)
    types = _plain_int(type_count, "type_count", 1)
    nodes = _plain_int(node_count, "node_count", 1)
    signatures = types * nodes * 8
    templates = types * nodes * p * 8
    carrier = p
    fft_probe = types * nodes * p * 16
    # NumPy's FFT may hold a real/complex conversion workspace alongside the
    # returned complex spectrum.
    fft_workspace = templates
    magnitude_probe = types * nodes * max(1, p - 1) * 8
    magnitude_reference = magnitude_probe
    comparison_scratch = types * nodes * max(1, p - 1)
    return (
        signatures
        + templates
        + carrier
        + fft_probe
        + fft_workspace
        + magnitude_probe
        + magnitude_reference
        + comparison_scratch
    )


def estimate_bank_work_units(
    *,
    prime: int,
    type_count: int,
    node_count: int,
) -> int:
    p = _plain_int(prime, "prime", 3)
    types = _plain_int(type_count, "type_count", 1)
    nodes = _plain_int(node_count, "node_count", 1)
    return types * nodes * p * max(4, int(math.ceil(math.log2(p))))


def _normalized_signatures(
    phase_signatures: np.ndarray,
    prime: int,
) -> np.ndarray:
    if phase_signatures.dtype.kind not in "iu" or phase_signatures.dtype.kind == "b":
        raise PRWH1ValidationError("phase_signatures must contain exact integers")
    signatures = np.ascontiguousarray(phase_signatures, dtype=np.int64)
    if np.any(signatures < 0) or np.any(signatures >= prime):
        raise PRWH1ValidationError(
            "phase_signatures must contain residues in [0, prime)"
        )
    if np.any(signatures[:, 0] != 0):
        raise PRWH1ValidationError(
            "phase_signatures must be gauge-fixed with first phase zero"
        )
    if np.unique(signatures, axis=0).shape[0] != signatures.shape[0]:
        raise PRWH1ValidationError("phase_signatures must be unique")
    return signatures


def _overlap_one(signatures: np.ndarray) -> bool:
    for left in range(signatures.shape[0]):
        for right in range(left + 1, signatures.shape[0]):
            if int(np.count_nonzero(signatures[left] == signatures[right])) > 1:
                return False
    return True


def _equal_nonzero_magnitudes(templates: np.ndarray) -> bool:
    spectra = np.fft.fft(templates, axis=2)
    magnitudes = np.abs(spectra[:, :, 1:])
    if magnitudes.size == 0:
        return False
    expected = math.sqrt(templates.shape[2] + 1.0)
    return all(
        math.isclose(
            float(value),
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for value in magnitudes.reshape(-1)
    )


def build_legendre_phase_bank(
    prime: int,
    phase_signatures: np.ndarray,
    *,
    budget: MultifrequencyBudget = MultifrequencyBudget(),
) -> MultifrequencyBank:
    """Build a tiny Legendre bank after shape/resource preflight."""

    p = _plain_int(prime, "prime", 3)
    types, nodes = _phase_matrix_shape(phase_signatures)
    _enforce_dimensions(prime=p, types=types, nodes=nodes, budget=budget)
    if p % 4 != 3 or not is_prime_exact(p):
        raise PRWH1ValidationError(
            "prime must be an odd prime congruent to 3 modulo 4"
        )
    estimated_bytes = estimate_bank_peak_bytes(
        prime=p,
        type_count=types,
        node_count=nodes,
    )
    if (
        phase_signatures.dtype != np.dtype(np.int64)
        or not phase_signatures.flags.c_contiguous
    ):
        estimated_bytes += int(phase_signatures.nbytes)
    estimated_work = estimate_bank_work_units(
        prime=p,
        type_count=types,
        node_count=nodes,
    )
    _enforce_resource_estimate(estimated_bytes, estimated_work, budget)
    deadline = _start_deadline(budget)
    _check_deadline(deadline)

    signatures = _normalized_signatures(phase_signatures, p)
    carrier = legendre_carrier(p)
    templates = np.empty((types, nodes, p), dtype=np.float64)
    for type_id in range(types):
        for node in range(nodes):
            _check_deadline(deadline)
            templates[type_id, node] = np.roll(
                carrier, int(signatures[type_id, node])
            )
    equal_magnitudes = _equal_nonzero_magnitudes(templates)
    signatures.setflags(write=False)
    templates.setflags(write=False)
    return MultifrequencyBank(
        prime=p,
        templates=templates,
        phase_signatures=signatures,
        overlap_one_verified=_overlap_one(signatures),
        legendre_equal_nonzero_magnitudes_verified=equal_magnitudes,
        build_estimated_peak_bytes=estimated_bytes,
        build_estimated_work_units=estimated_work,
    )


def deterministic_control_bins(
    prime: int,
    count: int,
    seed: int,
) -> Tuple[int, ...]:
    """Choose reproducible distinct positive real-frequency bins."""

    p = _plain_int(prime, "prime", 3)
    number = _plain_int(count, "count", 1)
    random_seed = _plain_int(seed, "seed", 0)
    if p > HARD_MAX_PRIME:
        raise PRWH1ResourceError(
            f"prime exceeds bounded PRW-H1 limit: {p} > {HARD_MAX_PRIME}"
        )
    if not is_prime_exact(p):
        raise PRWH1ValidationError("prime must be prime")
    available = (p - 1) // 2
    if number > available:
        raise PRWH1ValidationError(
            f"count exceeds the {available} nonconjugate positive bins"
        )
    digest = hashlib.sha256(
        f"PRW-H1|CONTROL-BINS|{p}|{number}|{random_seed}".encode("ascii")
    ).digest()
    rng = np.random.Generator(
        np.random.PCG64(int.from_bytes(digest[:16], "big"))
    )
    chosen = rng.choice(
        np.arange(1, available + 1, dtype=np.int64),
        size=number,
        replace=False,
    )
    return tuple(sorted(int(value) for value in chosen.tolist()))


def _validate_bins(
    condition: FrequencyCondition,
    prime: int,
) -> Tuple[int, ...]:
    if condition.mode == "time_domain":
        return ()
    maximum = (prime - 1) // 2
    if any(value > maximum for value in condition.bins):
        raise PRWH1ValidationError(
            f"frequency bins must be in [1, {maximum}] to avoid conjugate duplicates"
        )
    return condition.bins


def _bounded_bin_tuple(
    values: object,
    prime: int,
    name: str,
) -> Tuple[int, ...]:
    """Normalize bins only after a prime-derived length check."""

    if isinstance(values, (str, bytes)) or not isinstance(
        values, SequenceABC
    ):
        raise PRWH1ValidationError(
            f"{name} must be a finite sized sequence"
        )
    maximum = (prime - 1) // 2
    count = len(values)
    if count == 0:
        raise PRWH1ValidationError(f"{name} must be non-empty")
    if count > maximum:
        raise PRWH1ResourceError(
            f"{name} has {count} entries; bounded maximum is {maximum}"
        )
    normalized = tuple(
        _plain_int(values[index], "frequency bin", 1)
        for index in range(count)
    )
    if len(set(normalized)) != len(normalized):
        raise PRWH1ValidationError(f"{name} must contain distinct bins")
    if any(value > maximum for value in normalized):
        raise PRWH1ValidationError(
            f"frequency bins must be in [1, {maximum}]"
        )
    return normalized


def _quantized_unit_phase(
    values: np.ndarray,
    phase_bits: Optional[int],
    quantizer_origin_fraction: float = 0.0,
) -> np.ndarray:
    magnitudes = np.abs(values)
    tolerance = np.finfo(np.float64).eps * max(
        1.0, float(np.max(magnitudes, initial=0.0))
    )
    if np.any(magnitudes <= tolerance):
        raise PRWH1DegenerateFeatureError(
            "selected Fourier coefficient has zero magnitude"
        )
    unit = values / magnitudes
    if phase_bits is None:
        return unit
    levels = 1 << phase_bits
    angles = np.mod(np.angle(unit), 2.0 * math.pi)
    indices = np.floor(
        angles * levels / (2.0 * math.pi)
        - quantizer_origin_fraction
        + 0.5
    ).astype(np.int64) % levels
    return np.exp(
        2.0j
        * math.pi
        * (indices + quantizer_origin_fraction)
        / levels
    )


def estimate_decode_peak_bytes(
    bank: MultifrequencyBank,
    condition: FrequencyCondition,
    query: Optional[np.ndarray] = None,
) -> int:
    """Conservative bytes for one sequential decode."""

    types, nodes, p = bank.templates.shape
    slots = len(condition.bins)
    resident = int(bank.templates.nbytes + bank.phase_signatures.nbytes)
    normalized_query_bytes = nodes * p * 8
    if query is None:
        query_bytes = normalized_query_bytes
    else:
        query_bytes = int(query.nbytes)
        if query.dtype != np.dtype(np.float64) or not query.flags.c_contiguous:
            query_bytes += normalized_query_bytes
    scores = types * p * 8
    if condition.mode == "time_domain":
        temporaries = nodes * p * 8 * 4
    else:
        spectra = (types * nodes * p + nodes * p) * 16
        selected = (types * nodes * slots + nodes * slots) * 16 * 3
        # ``query_unit * conj(template_unit)`` can hold the conjugated
        # template and the resulting cross-spectrum at the same time.
        conjugate_and_cross = types * nodes * slots * 16 * 2
        phase_grid = slots * p * 16
        temporaries = (
            spectra
            + selected
            + conjugate_and_cross
            + phase_grid
            + scores
        )
    return resident + query_bytes + scores + temporaries


def estimate_decode_work_units(
    bank: MultifrequencyBank,
    condition: FrequencyCondition,
) -> int:
    types, nodes, p = bank.templates.shape
    hypotheses = types * p
    if condition.mode == "time_domain":
        return hypotheses * nodes * p
    slots = len(condition.bins)
    fft_work = (types + 1) * nodes * p * max(
        4, int(math.ceil(math.log2(p)))
    )
    if condition.mode == "magnitude":
        score_work = types * nodes * slots + hypotheses
    else:
        score_work = hypotheses * nodes * slots * 4
    return fft_work + score_work


def _score_summary(scores: np.ndarray) -> Tuple[
    Tuple[Tuple[int, int], ...],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    flat = scores.reshape(-1)
    if flat.size == 0 or not np.all(np.isfinite(flat)):
        return (), None, None, None
    best = float(np.max(flat))
    tolerance = 1e-12 * max(1.0, abs(best))
    winner_indices = np.flatnonzero(np.abs(flat - best) <= tolerance)
    prime = scores.shape[1]
    winners = tuple(
        (int(index // prime), int(index % prime))
        for index in winner_indices.tolist()
    )
    if len(winners) > 1:
        runner_up = best
        margin = 0.0
    else:
        below = flat[flat < best - tolerance]
        runner_up = float(np.max(below)) if below.size else best
        margin = best - runner_up if below.size else 0.0
    return winners, best, runner_up, margin


def _abstention_result(
    condition: FrequencyCondition,
    candidate_count: int,
    reason: str,
    accounting: Mapping[str, Any],
) -> DecodeResult:
    scores = np.empty((0, 0), dtype=np.float64)
    scores.setflags(write=False)
    return DecodeResult(
        condition=condition,
        scores=scores,
        top_states=(),
        unique_winner=False,
        abstained=True,
        abstention_reason=reason,
        best_score=None,
        runner_up_score=None,
        margin=None,
        candidate_count=candidate_count,
        accounting=accounting,
    )


def _randomized_query_phase(
    query_spectrum: np.ndarray,
    bins: Tuple[int, ...],
    condition: FrequencyCondition,
    query: np.ndarray,
) -> np.ndarray:
    digest = hashlib.sha256()
    digest.update(b"PRW-H1|RANDOMIZED-PHASE|")
    digest.update(str(condition.seed).encode("ascii"))
    digest.update(query.tobytes(order="C"))
    seed = int.from_bytes(digest.digest()[:16], "big")
    rng = np.random.Generator(np.random.PCG64(seed))
    shape = (query_spectrum.shape[0], len(bins))
    angles = rng.uniform(0.0, 2.0 * math.pi, size=shape)
    randomized = np.exp(1.0j * angles)
    if condition.phase_bits is not None:
        randomized = _quantized_unit_phase(
            randomized,
            condition.phase_bits,
            condition.quantizer_origin_fraction,
        )
    return randomized


def _decode_time_domain(
    bank: MultifrequencyBank,
    query: np.ndarray,
    deadline: float,
) -> np.ndarray:
    types, nodes, p = bank.templates.shape
    scores = np.empty((types, p), dtype=np.float64)
    query_norms = np.linalg.norm(query, axis=1)
    if np.any(query_norms == 0.0):
        raise PRWH1DegenerateFeatureError(
            "query node has zero time-domain norm"
        )
    for type_id in range(types):
        template_norms = np.linalg.norm(bank.templates[type_id], axis=1)
        if np.any(template_norms == 0.0):
            raise PRWH1DegenerateFeatureError(
                "template node has zero time-domain norm"
            )
        for shift in range(p):
            _check_deadline(deadline)
            rolled = np.roll(bank.templates[type_id], shift, axis=1)
            per_node = np.sum(query * rolled, axis=1) / (
                query_norms * template_norms
            )
            scores[type_id, shift] = float(np.mean(per_node))
    return scores


def _decode_fourier(
    bank: MultifrequencyBank,
    query: np.ndarray,
    condition: FrequencyCondition,
    bins: Tuple[int, ...],
    deadline: float,
) -> np.ndarray:
    types, nodes, p = bank.templates.shape
    template_spectra = np.fft.fft(bank.templates, axis=2)
    query_spectrum = np.fft.fft(query, axis=1)
    selected_template = template_spectra[:, :, bins]
    selected_query = query_spectrum[:, bins]

    if condition.mode == "magnitude":
        template_magnitudes = np.abs(selected_template)
        query_magnitudes = np.abs(selected_query)
        template_norms = np.linalg.norm(template_magnitudes, axis=(1, 2))
        query_norm = float(np.linalg.norm(query_magnitudes))
        if query_norm == 0.0 or np.any(template_norms == 0.0):
            raise PRWH1DegenerateFeatureError(
                "selected magnitude feature has zero norm"
            )
        type_scores = np.sum(
            template_magnitudes * query_magnitudes[np.newaxis, :, :],
            axis=(1, 2),
        ) / (template_norms * query_norm)
        return np.repeat(type_scores[:, np.newaxis], p, axis=1)

    template_unit = _quantized_unit_phase(
        selected_template,
        condition.phase_bits,
        condition.quantizer_origin_fraction,
    )
    if condition.mode == "randomized_phase":
        query_unit = _randomized_query_phase(
            query_spectrum, bins, condition, query
        )
    else:
        query_unit = _quantized_unit_phase(
            selected_query,
            condition.phase_bits,
            condition.quantizer_origin_fraction,
        )
    frequency = np.asarray(bins, dtype=np.float64)
    shifts = np.arange(p, dtype=np.float64)
    correction = np.exp(
        2.0j
        * math.pi
        * frequency[:, np.newaxis]
        * shifts[np.newaxis, :]
        / p
    )
    cross = query_unit[np.newaxis, :, :] * np.conj(template_unit)
    _check_deadline(deadline)
    scores = np.real(
        np.einsum("tvk,ks->ts", cross, correction, optimize=True)
    ) / (nodes * len(bins))
    return scores


def _condition_accounting(
    bank: MultifrequencyBank,
    condition: FrequencyCondition,
    estimated_bytes: int,
    estimated_work: int,
) -> Dict[str, Any]:
    types, nodes, p = bank.templates.shape
    slots = len(condition.bins) if condition.mode != "time_domain" else p
    unique_bins = (
        len(set(condition.bins))
        if condition.mode != "time_domain"
        else 0
    )
    feature_norm = (
        1.0
        if condition.mode
        in {"phase", "randomized_phase", "magnitude", "time_domain"}
        else None
    )
    return {
        "logical_coefficient_slots_per_node": slots,
        "unique_real_frequency_bins": unique_bins,
        "logical_stored_real_scalars_per_type": (
            2 * nodes * len(condition.bins)
            if condition.mode in {"phase", "randomized_phase"}
            else nodes * len(condition.bins)
            if condition.mode == "magnitude"
            else nodes * p
        ),
        "candidate_count": types * p,
        "nominal_score_work_units": estimated_work,
        "feature_norm": feature_norm,
        "unit_energy_normalization": condition.mode
        in {"phase", "randomized_phase", "magnitude", "time_domain"},
        "estimated_peak_bytes": estimated_bytes,
        "hard_max_estimated_bytes": HARD_MAX_ESTIMATED_BYTES,
        "hard_max_seconds": HARD_MAX_SECONDS,
        "hard_max_work_units": HARD_MAX_WORK_UNITS,
        "independent_shift_per_bin_permitted": False,
    }


def _decode_multifrequency_with_deadline(
    bank: MultifrequencyBank,
    query: np.ndarray,
    condition: FrequencyCondition,
    *,
    budget: MultifrequencyBudget,
    deadline: float,
) -> DecodeResult:
    """Decode one condition after shape-derived resource preflight."""

    query_shape = _array_shape(query, "query", 2)
    types, nodes, p = _array_shape(bank.templates, "bank.templates", 3)
    if p != bank.prime:
        raise PRWH1ValidationError("bank prime does not match template length")
    if query_shape != (nodes, p):
        raise PRWH1ValidationError(
            f"query must have shape {(nodes, p)}, got {query_shape}"
        )
    _enforce_dimensions(prime=p, types=types, nodes=nodes, budget=budget)
    bins = _validate_bins(condition, p)
    estimated_bytes = estimate_decode_peak_bytes(bank, condition, query)
    estimated_work = estimate_decode_work_units(bank, condition)
    _enforce_resource_estimate(estimated_bytes, estimated_work, budget)
    _check_deadline(deadline)
    accounting = _condition_accounting(
        bank, condition, estimated_bytes, estimated_work
    )

    if query.dtype.kind not in "iuf" or query.dtype.kind == "b":
        raise PRWH1ValidationError("query must contain real numeric values")
    normalized_query = np.ascontiguousarray(query, dtype=np.float64)
    if not np.all(np.isfinite(normalized_query)):
        raise PRWH1ValidationError("query must contain finite values")
    try:
        if condition.mode == "time_domain":
            scores = _decode_time_domain(bank, normalized_query, deadline)
        else:
            scores = _decode_fourier(
                bank, normalized_query, condition, bins, deadline
            )
    except PRWH1DegenerateFeatureError as exc:
        return _abstention_result(
            condition,
            bank.hypothesis_count,
            str(exc),
            accounting,
        )
    if not np.all(np.isfinite(scores)):
        raise PRWH1ValidationError("decoder produced non-finite scores")
    winners, best, runner_up, margin = _score_summary(scores)
    scores = np.ascontiguousarray(scores, dtype=np.float64)
    scores.setflags(write=False)
    return DecodeResult(
        condition=condition,
        scores=scores,
        top_states=winners,
        unique_winner=len(winners) == 1,
        abstained=False,
        abstention_reason=None,
        best_score=best,
        runner_up_score=runner_up,
        margin=margin,
        candidate_count=bank.hypothesis_count,
        accounting=accounting,
    )


def decode_multifrequency(
    bank: MultifrequencyBank,
    query: np.ndarray,
    condition: FrequencyCondition,
    *,
    budget: MultifrequencyBudget = MultifrequencyBudget(),
) -> DecodeResult:
    """Decode one condition under one bounded wall-clock deadline."""

    return _decode_multifrequency_with_deadline(
        bank,
        query,
        condition,
        budget=budget,
        deadline=_start_deadline(budget),
    )


def _conditions(
    prime: int,
    bins: Tuple[int, ...],
    control_seed: int,
    phase_bits: Optional[int],
    quantizer_origin_fraction: float,
) -> Tuple[FrequencyCondition, ...]:
    random_bins = deterministic_control_bins(
        prime, len(bins), control_seed
    )
    return (
        FrequencyCondition(
            "proposed_bins",
            "phase",
            bins,
            phase_bits=phase_bits,
            quantizer_origin_fraction=quantizer_origin_fraction,
        ),
        FrequencyCondition(
            "random_same_count_bins",
            "phase",
            random_bins,
            phase_bits=phase_bits,
            quantizer_origin_fraction=quantizer_origin_fraction,
        ),
        FrequencyCondition(
            "repeated_single_bin",
            "phase",
            tuple(bins[0] for _ in bins),
            phase_bits=phase_bits,
            quantizer_origin_fraction=quantizer_origin_fraction,
        ),
        FrequencyCondition(
            "single_bin",
            "phase",
            (bins[0],),
            phase_bits=phase_bits,
            quantizer_origin_fraction=quantizer_origin_fraction,
        ),
        FrequencyCondition(
            "magnitude_only",
            "magnitude",
            bins,
            phase_bits=None,
            quantizer_origin_fraction=0.0,
        ),
        FrequencyCondition(
            "randomized_phase",
            "randomized_phase",
            bins,
            seed=control_seed,
            phase_bits=phase_bits,
            quantizer_origin_fraction=quantizer_origin_fraction,
        ),
        FrequencyCondition(
            "all_nonconjugate_bins",
            "phase",
            tuple(range(1, (prime - 1) // 2 + 1)),
            phase_bits=phase_bits,
            quantizer_origin_fraction=quantizer_origin_fraction,
        ),
        FrequencyCondition("time_domain", "time_domain"),
    )


def _preflight_bank_query(
    bank: MultifrequencyBank,
    query: np.ndarray,
    budget: MultifrequencyBudget,
) -> Tuple[int, int, int]:
    """Refuse malformed/oversized shapes before controls or content scans."""

    types, nodes, p = _array_shape(bank.templates, "bank.templates", 3)
    signatures_shape = _array_shape(
        bank.phase_signatures, "bank.phase_signatures", 2
    )
    query_shape = _array_shape(query, "query", 2)
    if p != bank.prime:
        raise PRWH1ValidationError(
            "bank prime does not match template length"
        )
    if signatures_shape != (types, nodes):
        raise PRWH1ValidationError(
            "bank phase signatures do not match template type/node shape"
        )
    if query_shape != (nodes, p):
        raise PRWH1ValidationError(
            f"query must have shape {(nodes, p)}, got {query_shape}"
        )
    _enforce_dimensions(
        prime=p, types=types, nodes=nodes, budget=budget
    )
    resident_bytes = int(
        bank.templates.nbytes
        + bank.phase_signatures.nbytes
        + query.nbytes
    )
    _enforce_resource_estimate(resident_bytes, 1, budget)
    return types, nodes, p


def compare_multifrequency_controls(
    bank: MultifrequencyBank,
    query: np.ndarray,
    bins: Sequence[int],
    *,
    planted_state: Optional[Tuple[int, int]] = None,
    control_seed: int = 0,
    phase_bits: Optional[int] = None,
    quantizer_origin_fraction: float = 0.0,
    budget: MultifrequencyBudget = MultifrequencyBudget(),
) -> Dict[str, Any]:
    """Run the frozen controls sequentially for one tiny observation."""

    _preflight_bank_query(bank, query, budget)
    normalized_bins = _bounded_bin_tuple(
        bins, bank.prime, "proposed bins"
    )
    if FROZEN_CONTROL_CONDITION_COUNT > budget.max_conditions:
        raise PRWH1ResourceError(
            "condition count exceeds bounded PRW-H1 limit: "
            f"{FROZEN_CONTROL_CONDITION_COUNT} > "
            f"{budget.max_conditions}"
        )
    conditions = _conditions(
        bank.prime,
        normalized_bins,
        _plain_int(control_seed, "control_seed", 0),
        phase_bits,
        quantizer_origin_fraction,
    )
    if len(conditions) != FROZEN_CONTROL_CONDITION_COUNT:
        raise PRWH1ValidationError(
            "frozen PRW-H1 control condition count changed"
        )
    if len(conditions) > budget.max_conditions:
        raise PRWH1ResourceError(
            "condition count exceeds bounded PRW-H1 limit"
        )
    per_condition_peak_bytes = tuple(
        estimate_decode_peak_bytes(bank, condition, query)
        for condition in conditions
    )
    retained_score_bytes = (
        (len(conditions) - 1)
        * bank.type_count
        * bank.prime
        * 8
    )
    aggregate_peak_bytes = (
        max(per_condition_peak_bytes) + retained_score_bytes
    )
    aggregate_work_units = sum(
        estimate_decode_work_units(bank, condition)
        for condition in conditions
    )
    _enforce_resource_estimate(
        aggregate_peak_bytes, aggregate_work_units, budget
    )
    aggregate_deadline = _start_deadline(budget)
    if planted_state is not None:
        planted_type = _plain_int(planted_state[0], "planted type", 0)
        planted_shift = _plain_int(planted_state[1], "planted shift", 0)
        if planted_type >= bank.type_count or planted_shift >= bank.prime:
            raise PRWH1ValidationError("planted_state is outside the bank")
        truth = (planted_type, planted_shift)
    else:
        truth = None

    results: Dict[str, Any] = {}
    proposed = None
    for condition in conditions:
        _check_deadline(aggregate_deadline)
        decoded = _decode_multifrequency_with_deadline(
            bank,
            query,
            condition,
            budget=budget,
            deadline=aggregate_deadline,
        )
        _check_deadline(aggregate_deadline)
        correct = (
            bool(decoded.unique_winner and decoded.top_states[0] == truth)
            if truth is not None and not decoded.abstained
            else None
        )
        result = {
            "decode": decoded,
            "unique_correct": correct,
            "coefficient_budget_matches_proposed": (
                len(condition.bins) == len(normalized_bins)
                and condition.mode != "time_domain"
            ),
            "energy_budget_matches_proposed": condition.mode
            in {
                "phase",
                "randomized_phase",
                "magnitude",
                "time_domain",
            },
            "stored_scalar_budget_matches_proposed": (
                condition.mode in {"phase", "randomized_phase"}
                and len(condition.bins) == len(normalized_bins)
            ),
            "hypothesis_budget_matches_proposed": (
                decoded.candidate_count == bank.hypothesis_count
            ),
            "arithmetic_slot_budget_matches_proposed": (
                condition.mode in {"phase", "randomized_phase"}
                and len(condition.bins) == len(normalized_bins)
            ),
        }
        results[condition.name] = result
        if condition.name == "proposed_bins":
            proposed = result

    assert proposed is not None
    proposed_decode = proposed["decode"]
    contrasts = {}
    for name, result in results.items():
        if name == "proposed_bins":
            continue
        decoded = result["decode"]
        contrasts[name] = {
            "margin_difference": (
                proposed_decode.margin - decoded.margin
                if proposed_decode.margin is not None
                and decoded.margin is not None
                else None
            ),
            "correctness_difference": (
                int(bool(proposed["unique_correct"]))
                - int(bool(result["unique_correct"]))
                if truth is not None
                else None
            ),
        }
    return {
        "status": EVIDENCE_STATUS,
        "prime": bank.prime,
        "type_count": bank.type_count,
        "node_count": bank.node_count,
        "proposed_bins": list(normalized_bins),
        "phase_bits": phase_bits,
        "shared_state_count": bank.hypothesis_count,
        "independent_shift_per_bin_permitted": False,
        "results": results,
        "random_control_collides_with_proposed": (
            set(
                results["random_same_count_bins"]["decode"].condition.bins
            )
            == set(normalized_bins)
        ),
        "selected_frequency_comparison_eligible": (
            set(
                results["random_same_count_bins"]["decode"].condition.bins
            )
            != set(normalized_bins)
        ),
        "all_bins_control_collides_with_proposed": (
            set(
                results["all_nonconjugate_bins"]["decode"].condition.bins
            )
            == set(normalized_bins)
        ),
        "aggregate_resource_guard": {
            "estimated_peak_bytes": aggregate_peak_bytes,
            "estimated_work_units": aggregate_work_units,
            "max_seconds": budget.max_seconds,
            "deadline_scope": "all_conditions",
            "byte_scope": (
                "model_arrays_and_declared_temporaries_not_process_rss"
            ),
            "retained_prior_score_arrays_included": True,
        },
        "quantization_contract": {
            "phase_bits": conditions[0].phase_bits,
            "quantizer_origin_fraction": (
                conditions[0].quantizer_origin_fraction
            ),
            "unit_phase_angles_quantized": (
                conditions[0].phase_bits is not None
            ),
            "applies_to_modes": ["phase", "randomized_phase"],
            "magnitude_only_quantization_applied": False,
            "time_domain_quantization_applied": False,
            "fft_and_score_arithmetic": "float64_and_complex128",
            "packed_representation_implemented": False,
            "packed_storage_or_speed_claim_eligible": False,
        },
        "paired_single_observation_contrasts": contrasts,
        "novelty_claim": False,
        "promotion_evidence": False,
        "false_unlock_rate_computed": False,
        "frequency_diversity_established": False,
        "interpretation": (
            "selected bins can test bounded redundancy under corruption; "
            "they do not add noiseless address capacity"
        ),
    }


def _exact_comparison_estimate(
    bank: MultifrequencyBank,
    conditions: Sequence[FrequencyCondition],
    patterns: int,
) -> Tuple[int, int]:
    max_decode_bytes = max(
        estimate_decode_peak_bytes(bank, condition)
        for condition in conditions
    )
    query_bytes = bank.node_count * bank.prime * 8
    retained_prior_score_bytes = bank.type_count * bank.prime * 8
    estimated_bytes = (
        max_decode_bytes + query_bytes + retained_prior_score_bytes
    )
    per_pattern = sum(
        estimate_decode_work_units(bank, condition)
        for condition in conditions
    )
    flip_work = bank.node_count * bank.prime
    return estimated_bytes, patterns * (per_pattern + flip_work)


def _bipolar_templates(
    templates: np.ndarray,
    deadline: float,
) -> bool:
    for index, value in enumerate(templates.reshape(-1)):
        if index % 1024 == 0:
            _check_deadline(deadline)
        if float(value) not in (-1.0, 1.0):
            return False
    return True


def exact_bsc_comparison(
    bank: MultifrequencyBank,
    *,
    planted_type: int,
    planted_shift: int,
    crossover: float,
    bins: Sequence[int],
    control_seed: int = 0,
    phase_bits: Optional[int] = None,
    quantizer_origin_fraction: float = 0.0,
    budget: MultifrequencyBudget = MultifrequencyBudget(),
) -> Dict[str, Any]:
    """Exactly stream every bipolar flip pattern for a tiny PRW-H1 bank."""

    types, nodes, p = _array_shape(bank.templates, "bank.templates", 3)
    signatures_shape = _array_shape(
        bank.phase_signatures, "bank.phase_signatures", 2
    )
    if p != bank.prime:
        raise PRWH1ValidationError(
            "bank prime does not match template length"
        )
    if signatures_shape != (types, nodes):
        raise PRWH1ValidationError(
            "bank phase signatures do not match template type/node shape"
        )
    _enforce_dimensions(
        prime=p, types=types, nodes=nodes, budget=budget
    )
    truth_type = _plain_int(planted_type, "planted_type", 0)
    truth_shift = _plain_int(planted_shift, "planted_shift", 0)
    if truth_type >= bank.type_count or truth_shift >= bank.prime:
        raise PRWH1ValidationError("planted state is outside the bank")
    probability = _crossover(crossover)
    normalized_bins = _bounded_bin_tuple(
        bins, p, "exact comparison bins"
    )
    bit_count = bank.node_count * bank.prime
    if bit_count >= 63:
        raise PRWH1ResourceError(
            "exact corruption enumeration is restricted to fewer than 63 bits"
        )
    pattern_count = 1 << bit_count
    if pattern_count > budget.max_exact_patterns:
        raise PRWH1ResourceError(
            "exact pattern count exceeds bounded PRW-H1 limit: "
            f"{pattern_count} > {budget.max_exact_patterns}"
        )
    resident_bytes = int(
        bank.templates.nbytes + bank.phase_signatures.nbytes
    )
    _enforce_resource_estimate(resident_bytes, 1, budget)
    if FROZEN_CONTROL_CONDITION_COUNT > budget.max_conditions:
        raise PRWH1ResourceError(
            "condition count exceeds bounded PRW-H1 limit: "
            f"{FROZEN_CONTROL_CONDITION_COUNT} > "
            f"{budget.max_conditions}"
        )
    conditions = _conditions(
        bank.prime,
        normalized_bins,
        _plain_int(control_seed, "control_seed", 0),
        phase_bits,
        quantizer_origin_fraction,
    )
    if len(conditions) != FROZEN_CONTROL_CONDITION_COUNT:
        raise PRWH1ValidationError(
            "frozen PRW-H1 control condition count changed"
        )
    if len(conditions) > budget.max_conditions:
        raise PRWH1ResourceError(
            "condition count exceeds bounded PRW-H1 limit"
        )
    estimated_bytes, estimated_work = _exact_comparison_estimate(
        bank, conditions, pattern_count
    )
    _enforce_resource_estimate(estimated_bytes, estimated_work, budget)
    deadline = _start_deadline(budget)
    if not _bipolar_templates(bank.templates, deadline):
        raise PRWH1ValidationError(
            "exact BSC comparison requires bipolar templates"
        )

    names = tuple(condition.name for condition in conditions)
    correct_probability = {name: 0.0 for name in names}
    tie_probability = {name: 0.0 for name in names}
    abstain_probability = {name: 0.0 for name in names}
    proposed_only = {name: 0.0 for name in names if name != "proposed_bins"}
    control_only = {name: 0.0 for name in names if name != "proposed_bins"}
    probability_mass = 0.0
    base_query = np.roll(
        bank.templates[truth_type], truth_shift, axis=1
    ).copy()
    truth = (truth_type, truth_shift)

    for pattern in range(pattern_count):
        _check_deadline(deadline)
        # ``int.bit_count`` is Python 3.10+; the repository supports 3.9.
        weight = bin(pattern).count("1")
        mass = probability**weight * (1.0 - probability) ** (
            bit_count - weight
        )
        probability_mass += mass
        query = base_query.copy()
        for coordinate in range(bit_count):
            if pattern & (1 << coordinate):
                query.reshape(-1)[coordinate] *= -1.0
        outcomes: Dict[str, bool] = {}
        for condition in conditions:
            _check_deadline(deadline)
            decoded = _decode_multifrequency_with_deadline(
                bank,
                query,
                condition,
                budget=budget,
                deadline=deadline,
            )
            correct = bool(
                not decoded.abstained
                and decoded.unique_winner
                and decoded.top_states[0] == truth
            )
            outcomes[condition.name] = correct
            if correct:
                correct_probability[condition.name] += mass
            if decoded.abstained:
                abstain_probability[condition.name] += mass
            elif not decoded.unique_winner:
                tie_probability[condition.name] += mass
        proposed_correct = outcomes["proposed_bins"]
        for name in proposed_only:
            if proposed_correct and not outcomes[name]:
                proposed_only[name] += mass
            elif outcomes[name] and not proposed_correct:
                control_only[name] += mass

    paired = {
        name: {
            "proposed_correct_control_not_uniquely_correct_probability": (
                proposed_only[name]
            ),
            "control_correct_proposed_not_uniquely_correct_probability": (
                control_only[name]
            ),
            "paired_accuracy_difference": (
                proposed_only[name] - control_only[name]
            ),
        }
        for name in proposed_only
    }
    wrong_unique_probability = {}
    for name in names:
        residual = (
            probability_mass
            - correct_probability[name]
            - tie_probability[name]
            - abstain_probability[name]
        )
        tolerance = 1e-12 * max(1.0, probability_mass)
        if residual < -tolerance:
            raise PRWH1ValidationError(
                f"outcome probabilities over-count mass for {name}"
            )
        wrong_unique_probability[name] = max(0.0, residual)
    condition_contracts = {
        condition.name: {
            "mode": condition.mode,
            "bins": list(condition.bins),
            "phase_bits": condition.phase_bits,
            "quantizer_origin_fraction": (
                condition.quantizer_origin_fraction
            ),
            "quantization_applied": (
                condition.phase_bits is not None
                and condition.mode in {"phase", "randomized_phase"}
            ),
            "logical_coefficient_slots_per_node": (
                len(condition.bins)
                if condition.mode != "time_domain"
                else bank.prime
            ),
            "candidate_count": bank.hypothesis_count,
        }
        for condition in conditions
    }
    random_collision = (
        set(condition_contracts["random_same_count_bins"]["bins"])
        == set(normalized_bins)
    )
    return {
        "status": EVIDENCE_STATUS,
        "prime": bank.prime,
        "type_count": bank.type_count,
        "node_count": bank.node_count,
        "bit_count": bit_count,
        "pattern_count": pattern_count,
        "crossover": probability,
        "probability_mass": probability_mass,
        "correct_probability": correct_probability,
        "tie_probability": tie_probability,
        "abstain_probability": abstain_probability,
        "wrong_unique_probability": wrong_unique_probability,
        "paired_effects_vs_proposed": paired,
        "condition_contracts": condition_contracts,
        "random_control_collides_with_proposed": random_collision,
        "selected_frequency_comparison_eligible": not random_collision,
        "quantization_contract": {
            "phase_bits": conditions[0].phase_bits,
            "quantizer_origin_fraction": (
                conditions[0].quantizer_origin_fraction
            ),
            "unit_phase_angles_quantized": (
                conditions[0].phase_bits is not None
            ),
            "applies_to_modes": ["phase", "randomized_phase"],
            "magnitude_only_quantization_applied": False,
            "time_domain_quantization_applied": False,
            "fft_and_score_arithmetic": "float64_and_complex128",
            "packed_representation_implemented": False,
            "packed_storage_or_speed_claim_eligible": False,
        },
        "resource_guard": {
            "estimated_peak_bytes": estimated_bytes,
            "estimated_work_units": estimated_work,
            "max_seconds": budget.max_seconds,
            "byte_scope": (
                "model_arrays_and_declared_temporaries_not_process_rss"
            ),
            "retained_prior_score_array_included": True,
            "hard_max_estimated_bytes": HARD_MAX_ESTIMATED_BYTES,
            "hard_max_seconds": HARD_MAX_SECONDS,
            "hard_max_work_units": HARD_MAX_WORK_UNITS,
            "hard_max_exact_patterns": HARD_MAX_EXACT_PATTERNS,
        },
        "novelty_claim": False,
        "promotion_evidence": False,
        "false_unlock_rate_computed": False,
        "frequency_diversity_established": False,
        "tie_policy": "ties_count_as_incorrect",
        "enumeration_storage": "streamed_without_pattern_matrix",
    }


__all__ = [
    "DEFAULT_MAX_ESTIMATED_BYTES",
    "DEFAULT_MAX_EXACT_PATTERNS",
    "DEFAULT_MAX_HYPOTHESES",
    "DEFAULT_MAX_SECONDS",
    "DEFAULT_MAX_WORK_UNITS",
    "DecodeResult",
    "EVIDENCE_STATUS",
    "FrequencyCondition",
    "HARD_MAX_ESTIMATED_BYTES",
    "HARD_MAX_EXACT_PATTERNS",
    "HARD_MAX_HYPOTHESES",
    "HARD_MAX_PRIME",
    "HARD_MAX_SECONDS",
    "HARD_MAX_WORK_UNITS",
    "MultifrequencyBank",
    "MultifrequencyBudget",
    "PRWH1DegenerateFeatureError",
    "PRWH1ResourceError",
    "PRWH1ValidationError",
    "build_legendre_phase_bank",
    "compare_multifrequency_controls",
    "decode_multifrequency",
    "deterministic_control_bins",
    "estimate_bank_peak_bytes",
    "estimate_bank_work_units",
    "estimate_decode_peak_bytes",
    "estimate_decode_work_units",
    "exact_bsc_comparison",
]
