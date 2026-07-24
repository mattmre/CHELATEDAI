"""Finite, resource-guarded diagnostics for the PRW-T1 conjecture.

This module studies maximum-likelihood error events for a frozen bank of
Legendre-shift carrier hypotheses with optional cross-layer polarity masks.
It is deliberately limited to small banks.  It does not run importance
sampling, establish an asymptotic theorem, or provide scientific evidence.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from prime_ring_waypoint import (
    PRWValidationError,
    legendre_carrier,
    policy_masks,
)


MIB = 1024 * 1024
HARD_MAX_ESTIMATED_BYTES = 512 * MIB
HARD_MAX_SECONDS = 120.0
HARD_MAX_HYPOTHESES = 1024
HARD_MAX_COMPETITOR_PAIRS = 250_000
HARD_MAX_COORDINATES = 100_000
HARD_MAX_BRUTEFORCE_PATTERNS = 1 << 20
HARD_MAX_WORK_UNITS = 50_000_000


class PRWT1ValidationError(ValueError):
    """Raised when an intersection-analysis contract is malformed."""


class PRWT1ResourceError(RuntimeError):
    """Raised before allocation when a bounded analysis exceeds its budget."""


def _plain_int(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise PRWT1ValidationError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise PRWT1ValidationError(f"{name} must be >= {minimum}")
    return result


def _probability(value: object, name: str = "crossover") -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise PRWT1ValidationError(f"{name} must be a finite probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 0.5:
        raise PRWT1ValidationError(
            f"{name} must be in [0, 0.5) for nearest-Hamming ML"
        )
    return result


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise PRWT1ValidationError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise PRWT1ValidationError(f"{name} must be a finite positive number")
    return result


@dataclass(frozen=True)
class EnumerationBudget:
    """Fail-closed limits for one finite PRW-T1 analysis."""

    max_estimated_bytes: int = HARD_MAX_ESTIMATED_BYTES
    max_seconds: float = HARD_MAX_SECONDS
    max_work_units: int = HARD_MAX_WORK_UNITS
    max_hypotheses: int = HARD_MAX_HYPOTHESES
    max_competitor_pairs: int = HARD_MAX_COMPETITOR_PAIRS
    max_coordinates: int = HARD_MAX_COORDINATES
    max_bruteforce_patterns: int = HARD_MAX_BRUTEFORCE_PATTERNS

    def __post_init__(self) -> None:
        for name in (
            "max_estimated_bytes",
            "max_work_units",
            "max_hypotheses",
            "max_competitor_pairs",
            "max_coordinates",
            "max_bruteforce_patterns",
        ):
            _plain_int(getattr(self, name), name, 1)
        _positive_float(self.max_seconds, "max_seconds")
        hard_caps = {
            "max_estimated_bytes": HARD_MAX_ESTIMATED_BYTES,
            "max_seconds": HARD_MAX_SECONDS,
            "max_work_units": HARD_MAX_WORK_UNITS,
            "max_hypotheses": HARD_MAX_HYPOTHESES,
            "max_competitor_pairs": HARD_MAX_COMPETITOR_PAIRS,
            "max_coordinates": HARD_MAX_COORDINATES,
            "max_bruteforce_patterns": HARD_MAX_BRUTEFORCE_PATTERNS,
        }
        for name, hard_cap in hard_caps.items():
            if getattr(self, name) > hard_cap:
                raise PRWT1ValidationError(
                    f"{name} cannot exceed the hard ceiling {hard_cap}"
                )


def _start_deadline(budget: EnumerationBudget) -> float:
    return time.monotonic() + budget.max_seconds


def _check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise PRWT1ResourceError(
            "finite PRW-T1 calculation exceeded its wall-clock deadline"
        )


@dataclass(frozen=True)
class HypothesisState:
    """Canonical identifier for one type/shift/mask hypothesis."""

    type_id: int
    shift: int
    mask_id: int


@dataclass(frozen=True)
class LegendreMaskBank:
    """A small, materialized bipolar hypothesis bank."""

    prime: int
    layers: int
    phase_signatures: np.ndarray
    masks: np.ndarray
    templates: np.ndarray
    states: Tuple[HypothesisState, ...]
    build_estimated_peak_bytes: int
    build_estimated_work_units: int
    overlap_one_verified: bool
    rm_1_3_verified: bool
    prime_3_mod_4_verified: bool
    prw_t1_structure_verified: bool

    @property
    def hypothesis_count(self) -> int:
        return int(self.templates.shape[0])

    @property
    def coordinate_count(self) -> int:
        return int(self.templates.shape[1])


def estimate_bank_peak_bytes(
    *,
    prime: int,
    type_count: int,
    layers: int,
    mask_count: int,
) -> int:
    """Conservatively estimate simultaneous bank-construction buffers."""

    p = _plain_int(prime, "prime", 3)
    types = _plain_int(type_count, "type_count", 1)
    layer_count = _plain_int(layers, "layers", 1)
    masks = _plain_int(mask_count, "mask_count", 1)
    hypotheses = types * p * masks
    coordinates = layer_count * p
    template_bytes = hypotheses * coordinates
    one_state_layers = layer_count * p
    normalized_inputs = (
        types * layer_count * 8
        + masks * layer_count * 8
        + masks * layer_count
    )
    uniqueness_scratch = max(
        types * layer_count * 8 + types * 128,
        masks * layer_count * 8 + masks * 128,
    )
    state_metadata = hypotheses * 256
    return int(
        template_bytes
        + 4 * one_state_layers
        + p
        + normalized_inputs
        + uniqueness_scratch
        + state_metadata
    )


def estimate_bank_work_units(
    *,
    prime: int,
    type_count: int,
    layers: int,
    mask_count: int,
) -> int:
    """Bound carrier construction, overlap checks, and template writes."""

    p = _plain_int(prime, "prime", 3)
    types = _plain_int(type_count, "type_count", 1)
    layer_count = _plain_int(layers, "layers", 1)
    masks = _plain_int(mask_count, "mask_count", 1)
    hypotheses = types * p * masks
    coordinates = layer_count * p
    overlap_checks = types * max(types - 1, 0) * layer_count // 2
    return int(2 * hypotheses * coordinates + overlap_checks)


def estimate_analysis_peak_bytes(
    *,
    hypothesis_count: int,
    competitor_count: int,
    coordinate_count: int,
) -> int:
    """Estimate analysis buffers, including Python pair-cache overhead."""

    hypotheses = _plain_int(
        hypothesis_count, "hypothesis_count", 1
    )
    competitors = _plain_int(
        competitor_count, "competitor_count", 1
    )
    coordinates = _plain_int(
        coordinate_count, "coordinate_count", 1
    )
    pairs = competitors * (competitors - 1) // 2
    templates = hypotheses * coordinates
    disagreements = competitors * coordinates
    int32_disagreements = 4 * competitors * coordinates
    intersection_matrix = 4 * competitors * competitors
    probability_cache_and_spectrum = 640 * pairs
    vector_metadata = 256 * competitors
    binomial_scratch = 40 * (coordinates + 2)
    return int(
        templates
        + disagreements
        + int32_disagreements
        + intersection_matrix
        + probability_cache_and_spectrum
        + vector_metadata
        + binomial_scratch
    )


def estimate_analysis_work_units(
    *,
    hypothesis_count: int,
    competitor_count: int,
    coordinate_count: int,
) -> int:
    """Conservatively bound vector, intersection, and pairwise work."""

    hypotheses = _plain_int(
        hypothesis_count, "hypothesis_count", 1
    )
    competitors = _plain_int(
        competitor_count, "competitor_count", 1
    )
    coordinates = _plain_int(
        coordinate_count, "coordinate_count", 1
    )
    pairs = competitors * (competitors - 1) // 2
    return int(
        hypotheses * coordinates
        + competitors * competitors * coordinates
        + 3 * pairs * coordinates
    )


def _enforce_build_budget(
    *,
    hypotheses: int,
    coordinates: int,
    estimated_bytes: int,
    estimated_work_units: int,
    budget: EnumerationBudget,
) -> None:
    if hypotheses > budget.max_hypotheses:
        raise PRWT1ResourceError(
            "hypothesis count exceeds bounded enumeration budget: "
            f"{hypotheses} > {budget.max_hypotheses}"
        )
    if coordinates > budget.max_coordinates:
        raise PRWT1ResourceError(
            "coordinate count exceeds bounded enumeration budget: "
            f"{coordinates} > {budget.max_coordinates}"
        )
    if estimated_bytes > budget.max_estimated_bytes:
        raise PRWT1ResourceError(
            "estimated bank peak exceeds byte budget: "
            f"{estimated_bytes} > {budget.max_estimated_bytes}"
        )
    if estimated_work_units > budget.max_work_units:
        raise PRWT1ResourceError(
            "estimated work exceeds bounded enumeration budget: "
            f"{estimated_work_units} > {budget.max_work_units}"
        )


def _enforce_analysis_budget(
    *,
    hypotheses: int,
    competitors: int,
    coordinates: int,
    estimated_bytes: int,
    estimated_work_units: int,
    budget: EnumerationBudget,
) -> None:
    _enforce_build_budget(
        hypotheses=hypotheses,
        coordinates=coordinates,
        estimated_bytes=estimated_bytes,
        estimated_work_units=estimated_work_units,
        budget=budget,
    )
    pairs = competitors * (competitors - 1) // 2
    if pairs > budget.max_competitor_pairs:
        raise PRWT1ResourceError(
            "competitor-pair count exceeds bounded enumeration budget: "
            f"{pairs} > {budget.max_competitor_pairs}"
        )


def _exact_integer_matrix(
    values: object,
    name: str,
    *,
    columns: Optional[int] = None,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise PRWT1ValidationError(
            f"{name} must be a non-empty two-dimensional array"
        )
    if columns is not None and int(array.shape[1]) != columns:
        raise PRWT1ValidationError(
            f"{name} must have exactly {columns} columns"
        )
    if array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise PRWT1ValidationError(f"{name} must contain exact integers")
    return np.ascontiguousarray(array, dtype=np.int64)


def _matrix_shape(
    values: object,
    name: str,
    *,
    columns: Optional[int] = None,
    deadline: Optional[float] = None,
) -> Tuple[int, int]:
    if isinstance(values, np.ndarray):
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise PRWT1ValidationError(
                f"{name} must be a non-empty two-dimensional array"
            )
        rows, width = int(values.shape[0]), int(values.shape[1])
    else:
        if isinstance(values, (str, bytes)) or not isinstance(
            values, Sequence
        ):
            raise PRWT1ValidationError(
                f"{name} must be a rectangular sequence of rows"
            )
        rows = len(values)
        if rows == 0:
            raise PRWT1ValidationError(
                f"{name} must be a non-empty two-dimensional array"
            )
        first = values[0]
        if isinstance(first, (str, bytes)) or not isinstance(first, Sequence):
            raise PRWT1ValidationError(
                f"{name} must be a rectangular sequence of rows"
            )
        width = len(first)
        if width == 0:
            raise PRWT1ValidationError(
                f"{name} must be a non-empty two-dimensional array"
            )
        for index, row in enumerate(values):
            if deadline is not None and index % 1024 == 0:
                _check_deadline(deadline)
            if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
                raise PRWT1ValidationError(
                    f"{name} must be a rectangular sequence of rows"
                )
            if len(row) != width:
                raise PRWT1ValidationError(
                    f"{name} must be rectangular"
                )
    if columns is not None and width != columns:
        raise PRWT1ValidationError(
            f"{name} must have exactly {columns} columns"
        )
    return rows, width


def _rows_unique(
    values: np.ndarray,
    name: str,
    *,
    deadline: float,
) -> bool:
    seen = set()
    for index, row in enumerate(values):
        if index % 1024 == 0:
            _check_deadline(deadline)
        key = row.tobytes()
        if key in seen:
            return False
        seen.add(key)
    return True


def _overlap_one_verified(
    signatures: np.ndarray,
    prime: int,
    *,
    deadline: float,
) -> bool:
    type_count, layers = signatures.shape
    if type_count < 2:
        return False
    for left in range(type_count):
        for right in range(left + 1, type_count):
            _check_deadline(deadline)
            counts: Dict[int, int] = {}
            for layer in range(layers):
                difference = int(
                    (signatures[right, layer] - signatures[left, layer])
                    % prime
                )
                counts[difference] = counts.get(difference, 0) + 1
                if counts[difference] > 1:
                    return False
    return True


def _rm_1_3_verified(masks: np.ndarray) -> bool:
    if masks.shape != (16, 8):
        return False
    expected = policy_masks("typed16", 8)
    observed_rows = {
        tuple(int(value) for value in row) for row in masks.tolist()
    }
    expected_rows = {
        tuple(int(value) for value in row) for row in expected.tolist()
    }
    return observed_rows == expected_rows


def build_legendre_mask_bank(
    prime: int,
    phase_signatures: Sequence[Sequence[int]],
    masks: Sequence[Sequence[int]],
    *,
    budget: EnumerationBudget = EnumerationBudget(),
) -> LegendreMaskBank:
    """Materialize a small gauge-fixed Legendre×mask hypothesis bank."""

    if not isinstance(budget, EnumerationBudget):
        raise PRWT1ValidationError("budget must be EnumerationBudget")
    deadline = _start_deadline(budget)
    p = _plain_int(prime, "prime", 3)
    type_count, layer_count = _matrix_shape(
        phase_signatures,
        "phase_signatures",
        deadline=deadline,
    )
    mask_count, _ = _matrix_shape(
        masks,
        "masks",
        columns=layer_count,
        deadline=deadline,
    )
    hypotheses = type_count * p * mask_count
    coordinates = layer_count * p
    estimated = estimate_bank_peak_bytes(
        prime=p,
        type_count=type_count,
        layers=layer_count,
        mask_count=mask_count,
    )
    estimated_work = estimate_bank_work_units(
        prime=p,
        type_count=type_count,
        layers=layer_count,
        mask_count=mask_count,
    )
    _enforce_build_budget(
        hypotheses=hypotheses,
        coordinates=coordinates,
        estimated_bytes=estimated,
        estimated_work_units=estimated_work,
        budget=budget,
    )
    _check_deadline(deadline)
    signatures = _exact_integer_matrix(
        phase_signatures,
        "phase_signatures",
    )
    mask_array = _exact_integer_matrix(
        masks,
        "masks",
        columns=layer_count,
    )
    if np.any(signatures < 0) or np.any(signatures >= p):
        raise PRWT1ValidationError(
            "phase_signatures must contain residues in [0, prime)"
        )
    if np.any(signatures[:, 0] != 0):
        raise PRWT1ValidationError(
            "phase_signatures must be gauge-fixed with first phase zero"
        )
    if not _rows_unique(
        signatures,
        "phase_signatures",
        deadline=deadline,
    ):
        raise PRWT1ValidationError("phase_signatures must be unique")
    if not np.all(np.isin(mask_array, (-1, 1))):
        raise PRWT1ValidationError("masks must contain only -1 and +1")
    if not _rows_unique(mask_array, "masks", deadline=deadline):
        raise PRWT1ValidationError("masks must be unique")
    overlap_one = _overlap_one_verified(
        signatures,
        p,
        deadline=deadline,
    )
    mask_values = np.ascontiguousarray(mask_array, dtype=np.int8)
    del mask_array
    rm_1_3 = _rm_1_3_verified(mask_values)
    prime_3_mod_4 = p % 4 == 3

    try:
        carrier = legendre_carrier(p)
    except PRWValidationError as exc:
        raise PRWT1ValidationError(str(exc)) from exc
    templates = np.empty((hypotheses, coordinates), dtype=np.int8)
    states = []
    index = 0
    for type_id in range(type_count):
        for shift in range(p):
            _check_deadline(deadline)
            phase_layers = np.vstack(
                [
                    np.roll(
                        carrier,
                        int((signatures[type_id, layer] + shift) % p),
                    )
                    for layer in range(layer_count)
                ]
            )
            for mask_id in range(mask_count):
                templates[index] = (
                    phase_layers * mask_values[mask_id, :, np.newaxis]
                ).reshape(-1)
                states.append(
                    HypothesisState(
                        type_id=type_id,
                        shift=shift,
                        mask_id=mask_id,
                    )
                )
                index += 1

    signatures = np.ascontiguousarray(signatures, dtype=np.int64)
    for array in (signatures, mask_values, templates):
        array.setflags(write=False)
    return LegendreMaskBank(
        prime=p,
        layers=layer_count,
        phase_signatures=signatures,
        masks=mask_values,
        templates=templates,
        states=tuple(states),
        build_estimated_peak_bytes=estimated,
        build_estimated_work_units=estimated_work,
        overlap_one_verified=overlap_one,
        rm_1_3_verified=rm_1_3,
        prime_3_mod_4_verified=prime_3_mod_4,
        prw_t1_structure_verified=(
            overlap_one and rm_1_3 and prime_3_mod_4
        ),
    )


def _binomial_table_storage_bytes(trials: int) -> int:
    return int(16 * (trials + 2) + 256)


def _compute_binomial_tables(
    trials: int,
    probability: float,
    *,
    deadline: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n = _plain_int(trials, "trials", 0)
    if n > HARD_MAX_COORDINATES:
        raise PRWT1ResourceError(
            "binomial trial count exceeds the hard coordinate ceiling"
        )
    q = _probability(probability)
    if deadline is not None:
        _check_deadline(deadline)
    if q == 0.0:
        pmf = np.zeros(n + 1, dtype=np.float64)
        pmf[0] = 1.0
    else:
        log_q = math.log(q)
        log_one_minus_q = math.log1p(-q)
        log_values = np.fromiter(
            (
                math.lgamma(n + 1)
                - math.lgamma(k + 1)
                - math.lgamma(n - k + 1)
                + k * log_q
                + (n - k) * log_one_minus_q
                for k in range(n + 1)
            ),
            dtype=np.float64,
            count=n + 1,
        )
        log_values -= float(np.max(log_values))
        np.exp(log_values, out=log_values)
        log_values /= float(np.sum(log_values))
        pmf = log_values
    tails = np.zeros(n + 2, dtype=np.float64)
    running = 0.0
    for k in range(n, -1, -1):
        if deadline is not None and k % 4096 == 0:
            _check_deadline(deadline)
        running += float(pmf[k])
        tails[k] = min(1.0, max(0.0, running))
    pmf.setflags(write=False)
    tails.setflags(write=False)
    return pmf, tails


class _BinomialCache:
    """One-analysis cache with explicit byte accounting and no persistence."""

    def __init__(self, max_bytes: int, deadline: float) -> None:
        self.max_bytes = max_bytes
        self.deadline = deadline
        self.used_bytes = 0
        self.tables: OrderedDict[
            Tuple[int, float], Tuple[np.ndarray, np.ndarray]
        ] = OrderedDict()

    def get(
        self, trials: int, probability: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        key = (trials, probability)
        existing = self.tables.get(key)
        if existing is not None:
            self.tables.move_to_end(key)
            return existing
        required = _binomial_table_storage_bytes(trials)
        if self.used_bytes + required > self.max_bytes:
            raise PRWT1ResourceError(
                "analysis-local binomial cache would exceed its byte budget"
            )
        _check_deadline(self.deadline)
        tables = _compute_binomial_tables(
            trials,
            probability,
            deadline=self.deadline,
        )
        self.tables[key] = tables
        self.used_bytes += required
        return tables


def _binomial_tables(
    trials: int,
    probability: float,
    *,
    cache: Optional[_BinomialCache] = None,
    deadline: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if cache is not None:
        return cache.get(trials, probability)
    return _compute_binomial_tables(
        trials,
        probability,
        deadline=deadline,
    )


def _binomial_upper_tail(
    trials: int,
    crossover: float,
    threshold: int,
    *,
    cache: Optional[_BinomialCache] = None,
    deadline: Optional[float] = None,
) -> float:
    n = _plain_int(trials, "trials", 0)
    if n > HARD_MAX_COORDINATES:
        raise PRWT1ResourceError(
            "binomial trial count exceeds the hard coordinate ceiling"
        )
    q = _probability(crossover)
    cutoff = _plain_int(threshold, "threshold", 0)
    if cutoff <= 0:
        return 1.0
    if cutoff > n:
        return 0.0
    return float(
        _binomial_tables(
            n,
            q,
            cache=cache,
            deadline=deadline,
        )[1][cutoff]
    )


def binomial_upper_tail(
    trials: int, crossover: float, threshold: int
) -> float:
    """Return ``P[Bin(trials, crossover) >= threshold]``."""

    return _binomial_upper_tail(trials, crossover, threshold)


def _bsc_pairwise_error(
    distance: int,
    crossover: float,
    *,
    cache: Optional[_BinomialCache] = None,
    deadline: Optional[float] = None,
) -> float:
    d = _plain_int(distance, "distance", 1)
    if d > HARD_MAX_COORDINATES:
        raise PRWT1ResourceError(
            "pairwise distance exceeds the hard coordinate ceiling"
        )
    return _binomial_upper_tail(
        d,
        crossover,
        (d + 1) // 2,
        cache=cache,
        deadline=deadline,
    )


def bsc_pairwise_error(distance: int, crossover: float) -> float:
    """Tie-as-error pairwise ML probability at Hamming distance ``distance``."""

    return _bsc_pairwise_error(distance, crossover)


def _pair_event_intersection_probability(
    distance_left: int,
    distance_right: int,
    disagreement_intersection: int,
    crossover: float,
    *,
    cache: Optional[_BinomialCache] = None,
    deadline: Optional[float] = None,
) -> float:
    left = _plain_int(distance_left, "distance_left", 1)
    right = _plain_int(distance_right, "distance_right", 1)
    if max(left, right) > HARD_MAX_COORDINATES:
        raise PRWT1ResourceError(
            "pairwise distance exceeds the hard coordinate ceiling"
        )
    overlap = _plain_int(
        disagreement_intersection,
        "disagreement_intersection",
        0,
    )
    if overlap > min(left, right):
        raise PRWT1ValidationError(
            "disagreement_intersection cannot exceed either distance"
        )
    q = _probability(crossover)
    left_only = left - overlap
    right_only = right - overlap
    overlap_pmf = _binomial_tables(
        overlap,
        q,
        cache=cache,
        deadline=deadline,
    )[0]
    left_tails = _binomial_tables(
        left_only,
        q,
        cache=cache,
        deadline=deadline,
    )[1]
    right_tails = _binomial_tables(
        right_only,
        q,
        cache=cache,
        deadline=deadline,
    )[1]
    left_threshold = (left + 1) // 2
    right_threshold = (right + 1) // 2
    probability = 0.0
    for shared_flips, mass in enumerate(overlap_pmf):
        if deadline is not None and shared_flips % 4096 == 0:
            _check_deadline(deadline)
        left_cutoff = max(0, left_threshold - shared_flips)
        right_cutoff = max(0, right_threshold - shared_flips)
        left_mass = (
            0.0
            if left_cutoff > left_only
            else float(left_tails[left_cutoff])
        )
        right_mass = (
            0.0
            if right_cutoff > right_only
            else float(right_tails[right_cutoff])
        )
        probability += float(mass) * left_mass * right_mass
    upper = min(
        _bsc_pairwise_error(
            left,
            q,
            cache=cache,
            deadline=deadline,
        ),
        _bsc_pairwise_error(
            right,
            q,
            cache=cache,
            deadline=deadline,
        ),
    )
    return float(min(upper, max(0.0, probability)))


def pair_event_intersection_probability(
    distance_left: int,
    distance_right: int,
    disagreement_intersection: int,
    crossover: float,
) -> float:
    """Return the exact finite probability of two tie-as-error ML events."""

    return _pair_event_intersection_probability(
        distance_left,
        distance_right,
        disagreement_intersection,
        crossover,
    )


def _competitor_indices(
    bank: LegendreMaskBank,
    planted_index: int,
    wrong_type_only: bool,
) -> np.ndarray:
    planted = _plain_int(planted_index, "planted_index", 0)
    if planted >= bank.hypothesis_count:
        raise PRWT1ValidationError("planted_index is outside the bank")
    planted_type = bank.states[planted].type_id
    retained = [
        index
        for index, state in enumerate(bank.states)
        if index != planted
        and (not wrong_type_only or state.type_id != planted_type)
    ]
    if not retained:
        raise PRWT1ValidationError("analysis requires at least one competitor")
    return np.asarray(retained, dtype=np.int64)


def _maximum_spanning_tree_weight(
    distances: np.ndarray,
    intersections: np.ndarray,
    crossover: float,
    *,
    probability_by_signature: Optional[
        Dict[Tuple[int, int, int], float]
    ] = None,
    binomial_cache: Optional[_BinomialCache] = None,
    deadline: Optional[float] = None,
) -> Tuple[float, int]:
    count = int(distances.size)
    if count <= 1:
        return 0.0, 0
    visited = np.zeros(count, dtype=bool)
    best = np.full(count, -1.0, dtype=np.float64)
    visited[0] = True
    probability_cache: Dict[Tuple[int, int, int], float] = {}

    def weight(left: int, right: int) -> float:
        d_left = int(distances[left])
        d_right = int(distances[right])
        overlap = int(intersections[left, right])
        key = (
            min(d_left, d_right),
            max(d_left, d_right),
            overlap,
        )
        if probability_by_signature is not None:
            return probability_by_signature[key]
        if key not in probability_cache:
            probability_cache[key] = _pair_event_intersection_probability(
                d_left,
                d_right,
                overlap,
                crossover,
                cache=binomial_cache,
                deadline=deadline,
            )
        return probability_cache[key]

    for node in range(1, count):
        best[node] = weight(0, node)
    total = 0.0
    edge_count = 0
    for _ in range(count - 1):
        if deadline is not None:
            _check_deadline(deadline)
        candidates = np.flatnonzero(~visited)
        if candidates.size == 0:
            break
        selected = int(
            max(
                (int(value) for value in candidates),
                key=lambda value: (best[value], -value),
            )
        )
        if best[selected] < 0.0:
            raise PRWT1ValidationError(
                "complete event graph unexpectedly disconnected"
            )
        total += float(best[selected])
        edge_count += 1
        visited[selected] = True
        for node in np.flatnonzero(~visited):
            candidate = weight(selected, int(node))
            if candidate > best[int(node)]:
                best[int(node)] = candidate
    return float(total), edge_count


def brute_force_union_probability(
    disagreements: np.ndarray,
    crossover: float,
    *,
    budget: EnumerationBudget = EnumerationBudget(),
    _deadline: Optional[float] = None,
) -> float:
    """Enumerate the exact union probability for a tiny coordinate space."""

    if not isinstance(budget, EnumerationBudget):
        raise PRWT1ValidationError("budget must be EnumerationBudget")
    deadline = (
        _start_deadline(budget)
        if _deadline is None
        else min(_deadline, _start_deadline(budget))
    )
    matrix = np.asarray(disagreements)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise PRWT1ValidationError(
            "disagreements must be a non-empty two-dimensional array"
        )
    if matrix.dtype.kind != "b":
        raise PRWT1ValidationError("disagreements must be boolean")
    q = _probability(crossover)
    competitor_count = int(matrix.shape[0])
    coordinates = int(matrix.shape[1])
    if coordinates > budget.max_coordinates:
        raise PRWT1ResourceError(
            "brute-force coordinates exceed bounded enumeration budget"
        )
    patterns = 1 << coordinates
    if patterns > budget.max_bruteforce_patterns:
        raise PRWT1ResourceError(
            "brute-force pattern count exceeds budget: "
            f"{patterns} > {budget.max_bruteforce_patterns}"
        )
    estimated_bytes = int(
        matrix.nbytes
        + competitor_count * (((coordinates + 7) // 8) + 192)
    )
    if estimated_bytes > budget.max_estimated_bytes:
        raise PRWT1ResourceError(
            "brute-force estimated peak exceeds byte budget: "
            f"{estimated_bytes} > {budget.max_estimated_bytes}"
        )
    estimated_work = int(
        competitor_count * coordinates
        + patterns * (competitor_count + 2)
    )
    if estimated_work > budget.max_work_units:
        raise PRWT1ResourceError(
            "brute-force estimated work exceeds bounded budget: "
            f"{estimated_work} > {budget.max_work_units}"
        )
    event_masks = []
    thresholds = []
    for row in matrix:
        bitmask = 0
        for coordinate in np.flatnonzero(row):
            bitmask |= 1 << int(coordinate)
        event_masks.append(bitmask)
        distance = int(np.count_nonzero(row))
        if distance <= 0:
            raise PRWT1ValidationError(
                "competitor disagreements must have positive distance"
            )
        thresholds.append((distance + 1) // 2)
    probability = 0.0
    for pattern in range(patterns):
        if pattern % 1024 == 0:
            _check_deadline(deadline)
        if any(
            (pattern & event_mask).bit_count() >= threshold
            for event_mask, threshold in zip(event_masks, thresholds)
        ):
            flips = pattern.bit_count()
            probability += (
                q**flips * (1.0 - q) ** (coordinates - flips)
            )
    return float(min(1.0, max(0.0, probability)))


def analyze_legendre_mask_bank(
    bank: LegendreMaskBank,
    *,
    planted_index: int,
    crossover: float,
    wrong_type_only: bool = True,
    compute_bruteforce_union: bool = False,
    require_prw_t1_structure: bool = True,
    budget: EnumerationBudget = EnumerationBudget(),
) -> Dict[str, Any]:
    """Compute finite distance/intersection bounds for one planted state."""

    if not isinstance(bank, LegendreMaskBank):
        raise PRWT1ValidationError("bank must be LegendreMaskBank")
    if not isinstance(budget, EnumerationBudget):
        raise PRWT1ValidationError("budget must be EnumerationBudget")
    if not isinstance(wrong_type_only, bool):
        raise PRWT1ValidationError("wrong_type_only must be boolean")
    if not isinstance(compute_bruteforce_union, bool):
        raise PRWT1ValidationError(
            "compute_bruteforce_union must be boolean"
        )
    if not isinstance(require_prw_t1_structure, bool):
        raise PRWT1ValidationError(
            "require_prw_t1_structure must be boolean"
        )
    if require_prw_t1_structure and not bank.prw_t1_structure_verified:
        raise PRWT1ValidationError(
            "bank is not a verified overlap-one Legendre-by-RM(1,3) "
            "bank at a prime congruent to 3 mod 4"
        )
    started = time.monotonic()
    deadline = started + budget.max_seconds
    q = _probability(crossover)
    competitors = _competitor_indices(
        bank, planted_index, wrong_type_only
    )
    competitor_count = int(competitors.size)
    competitor_pairs = competitor_count * (competitor_count - 1) // 2
    estimated = estimate_analysis_peak_bytes(
        hypothesis_count=bank.hypothesis_count,
        competitor_count=competitor_count,
        coordinate_count=bank.coordinate_count,
    )
    estimated_work = estimate_analysis_work_units(
        hypothesis_count=bank.hypothesis_count,
        competitor_count=competitor_count,
        coordinate_count=bank.coordinate_count,
    )
    if compute_bruteforce_union:
        patterns = 1 << bank.coordinate_count
        estimated += int(
            competitor_count
            * (((bank.coordinate_count + 7) // 8) + 192)
        )
        estimated_work += int(
            competitor_count * bank.coordinate_count
            + patterns * (competitor_count + 2)
        )
    _enforce_analysis_budget(
        hypotheses=bank.hypothesis_count,
        competitors=competitor_count,
        coordinates=bank.coordinate_count,
        estimated_bytes=estimated,
        estimated_work_units=estimated_work,
        budget=budget,
    )
    _check_deadline(deadline)

    planted = bank.templates[int(planted_index)]
    disagreements = np.not_equal(bank.templates[competitors], planted)
    distances = np.count_nonzero(disagreements, axis=1).astype(np.int64)
    if np.any(distances <= 0):
        raise PRWT1ValidationError(
            "distinct competitor hypotheses produced zero distance"
        )
    minimum = int(np.min(distances))
    nearest_positions = np.flatnonzero(distances == minimum)
    nearest_multiplicity = int(nearest_positions.size)
    distance_spectrum: Dict[str, int] = {}
    for distance in distances:
        key = str(int(distance))
        distance_spectrum[key] = distance_spectrum.get(key, 0) + 1

    int32_disagreements = disagreements.astype(np.int32)
    intersections = int32_disagreements @ int32_disagreements.T
    signature_counts: Dict[Tuple[int, int, int], int] = {}
    nearest_signature_counts: Dict[Tuple[int, int, int], int] = {}
    required_binomial_trials = {
        int(distance) for distance in distances.tolist()
    }
    pair_counter = 0
    for left in range(competitor_count):
        d_left = int(distances[left])
        for right in range(left + 1, competitor_count):
            if pair_counter % 1024 == 0:
                _check_deadline(deadline)
            d_right = int(distances[right])
            overlap = int(intersections[left, right])
            signature = (
                min(d_left, d_right),
                max(d_left, d_right),
                overlap,
            )
            signature_counts[signature] = (
                signature_counts.get(signature, 0) + 1
            )
            if d_left == minimum and d_right == minimum:
                nearest_signature_counts[signature] = (
                    nearest_signature_counts.get(signature, 0) + 1
                )
            required_binomial_trials.update(
                (overlap, d_left - overlap, d_right - overlap)
            )
            pair_counter += 1
    if pair_counter != competitor_pairs:
        raise PRWT1ValidationError(
            "competitor-pair enumeration count is inconsistent"
        )
    binomial_cache_bytes = int(
        sum(
            _binomial_table_storage_bytes(trials)
            for trials in required_binomial_trials
        )
    )
    total_estimated = estimated + binomial_cache_bytes
    _enforce_analysis_budget(
        hypotheses=bank.hypothesis_count,
        competitors=competitor_count,
        coordinates=bank.coordinate_count,
        estimated_bytes=total_estimated,
        estimated_work_units=estimated_work,
        budget=budget,
    )
    binomial_cache = _BinomialCache(
        max_bytes=binomial_cache_bytes,
        deadline=deadline,
    )
    pairwise = np.asarray(
        [
            _bsc_pairwise_error(
                int(distance),
                q,
                cache=binomial_cache,
                deadline=deadline,
            )
            for distance in distances
        ],
        dtype=np.float64,
    )
    raw_union = float(math.fsum(float(value) for value in pairwise))
    nearest_pairwise_probability = _bsc_pairwise_error(
        minimum,
        q,
        cache=binomial_cache,
        deadline=deadline,
    )
    nearest_term = float(
        nearest_multiplicity * nearest_pairwise_probability
    )
    farther_term = float(max(0.0, raw_union - nearest_term))
    probability_by_signature: Dict[
        Tuple[int, int, int], float
    ] = {}
    intersection_spectrum = []
    nearest_intersection_spectrum = []
    nearest_intersection_sum = 0.0
    for signature in sorted(signature_counts):
        _check_deadline(deadline)
        d_left, d_right, overlap = signature
        probability = _pair_event_intersection_probability(
            d_left,
            d_right,
            overlap,
            q,
            cache=binomial_cache,
            deadline=deadline,
        )
        probability_by_signature[signature] = probability
        count = signature_counts[signature]
        row = {
            "distance_left": d_left,
            "distance_right": d_right,
            "disagreement_intersection": overlap,
            "pair_count": count,
            "intersection_probability": probability,
            "summed_intersection_probability": float(
                count * probability
            ),
        }
        intersection_spectrum.append(row)
        nearest_count = nearest_signature_counts.get(signature, 0)
        if nearest_count:
            nearest_row = dict(row)
            nearest_row["pair_count"] = nearest_count
            nearest_row["summed_intersection_probability"] = float(
                nearest_count * probability
            )
            nearest_intersection_spectrum.append(nearest_row)
            nearest_intersection_sum += nearest_count * probability
    mst_weight, mst_edges = _maximum_spanning_tree_weight(
        distances,
        intersections,
        q,
        probability_by_signature=probability_by_signature,
        binomial_cache=binomial_cache,
        deadline=deadline,
    )
    hunter_raw = float(raw_union - mst_weight)
    brute_force = None
    if compute_bruteforce_union:
        brute_force = brute_force_union_probability(
            disagreements,
            q,
            budget=budget,
            _deadline=deadline,
        )

    nearest_states = [
        {
            "bank_index": int(competitors[int(position)]),
            "type_id": bank.states[int(competitors[int(position)])].type_id,
            "shift": bank.states[int(competitors[int(position)])].shift,
            "mask_id": bank.states[int(competitors[int(position)])].mask_id,
        }
        for position in nearest_positions
    ]
    elapsed = time.monotonic() - started
    _check_deadline(deadline)
    direct_prw_t1_scope = (
        bank.prw_t1_structure_verified and wrong_type_only
    )
    return {
        "status": (
            "FINITE_PRW_T1_STRUCTURE_DIAGNOSTIC_ONLY"
            if direct_prw_t1_scope
            else "FINITE_GENERIC_COMPETITOR_EVENT_DIAGNOSTIC_ONLY"
        ),
        "tie_semantics": "competitor_tie_counts_as_error",
        "competitor_event_scope": (
            "existence_of_wrong_type_candidate_tying_or_beating_truth"
            if wrong_type_only
            else "existence_of_any_nontruth_candidate_tying_or_beating_truth"
        ),
        "final_decoder_error_probability_computed": False,
        "wrong_type_only": wrong_type_only,
        "require_prw_t1_structure": require_prw_t1_structure,
        "prime": bank.prime,
        "layers": bank.layers,
        "structure_verification": {
            "overlap_one_verified": bank.overlap_one_verified,
            "rm_1_3_verified": bank.rm_1_3_verified,
            "prime_3_mod_4_verified": bank.prime_3_mod_4_verified,
            "prw_t1_structure_verified": (
                bank.prw_t1_structure_verified
            ),
            "direct_prw_t1_event_scope": direct_prw_t1_scope,
        },
        "hypothesis_count": bank.hypothesis_count,
        "competitor_count": competitor_count,
        "competitor_pair_count": competitor_pairs,
        "coordinate_count": bank.coordinate_count,
        "crossover": q,
        "distance_spectrum": dict(
            sorted(
                distance_spectrum.items(),
                key=lambda item: int(item[0]),
            )
        ),
        "minimum_distance": minimum,
        "nearest_state_multiplicity": nearest_multiplicity,
        "nearest_states": nearest_states,
        "nearest_pairwise_error_probability": (
            nearest_pairwise_probability
        ),
        "nearest_state_union_term": nearest_term,
        "farther_state_union_term": farther_term,
        "farther_to_nearest_union_term_ratio": (
            None if nearest_term == 0.0 else farther_term / nearest_term
        ),
        "intersection_signature_spectrum": intersection_spectrum,
        "nearest_intersection_signature_spectrum": (
            nearest_intersection_spectrum
        ),
        "nearest_pair_intersection_sum": float(
            nearest_intersection_sum
        ),
        "nearest_intersection_to_union_term_ratio": (
            None
            if nearest_term == 0.0
            else float(nearest_intersection_sum / nearest_term)
        ),
        "ordinary_union_bound_raw": raw_union,
        "ordinary_union_bound": min(1.0, raw_union),
        "hunter_spanning_tree_intersection_weight": mst_weight,
        "hunter_spanning_tree_edge_count": mst_edges,
        "hunter_upper_bound_raw": hunter_raw,
        "hunter_upper_bound": min(1.0, max(0.0, hunter_raw)),
        "bruteforce_competitor_tie_or_better_union_probability": (
            brute_force
        ),
        "resource_guard": {
            "estimated_peak_bytes": total_estimated,
            "max_estimated_bytes": budget.max_estimated_bytes,
            "estimated_work_units": estimated_work,
            "max_work_units": budget.max_work_units,
            "elapsed_seconds": float(elapsed),
            "max_seconds": budget.max_seconds,
            "binomial_cache_estimated_bytes": binomial_cache_bytes,
            "binomial_cache_actual_accounted_bytes": (
                binomial_cache.used_bytes
            ),
            "max_hypotheses": budget.max_hypotheses,
            "max_competitor_pairs": budget.max_competitor_pairs,
            "max_coordinates": budget.max_coordinates,
            "max_bruteforce_patterns": budget.max_bruteforce_patterns,
            "full_size_campaign_permitted": False,
        },
        "novelty_claim": False,
        "transmitted_state_averaging_complete": False,
        "asymptotic_claim_established": False,
    }
