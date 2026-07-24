"""Bounded p=11 quantizer-grid null for PRW-H1Q.

This module exhaustively compares every unordered pair of the five
nonconjugate positive Fourier bins at prime 11.  It streams the 2**11 bipolar
corruption patterns once and evaluates all matched conditions on the same
query, so the comparison is paired without materializing a pattern matrix.

The four shared quantizer origins are a finite hardware-grid nuisance audit.
They are not stochastic dither, continuous origin integration, or a
multiplier-invariant group average.  In particular, a multiplier can conjugate
only one selected coefficient, while one common origin constrains both
coefficients to the diagonal of the per-bin origin space.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from prime_ring_multifrequency import (
    HARD_MAX_ESTIMATED_BYTES as H1_HARD_MAX_ESTIMATED_BYTES,
    HARD_MAX_SECONDS as H1_HARD_MAX_SECONDS,
    HARD_MAX_WORK_UNITS as H1_HARD_MAX_WORK_UNITS,
    DecodeResult,
    FrequencyCondition,
    MultifrequencyBudget,
    PRWH1DegenerateFeatureError,
    PRWH1ValidationError,
    _abstention_result,
    _quantized_unit_phase,
    _score_summary,
    build_legendre_phase_bank,
    estimate_bank_peak_bytes,
    estimate_bank_work_units,
)


MIB = 1024 * 1024

PRIME = 11
NONCONJUGATE_BINS = (1, 2, 3, 4, 5)
TWO_BIN_SUBSETS = tuple(itertools.combinations(NONCONJUGATE_BINS, 2))
DECLARED_ORIGINS = (0.0, 0.25, 0.5, 0.75)
DEFAULT_PHASE_BITS = 8
DEFAULT_CROSSOVER = 0.45
DEFAULT_PLANTED_SHIFT = 3
EQUALITY_TOLERANCE = 1e-12

ORBIT_A = ((1, 2), (1, 5), (2, 4), (3, 4), (3, 5))
ORBIT_B = ((1, 3), (1, 4), (2, 3), (2, 5), (4, 5))
MULTIPLIER_ORBITS = {
    "A": ORBIT_A,
    "B": ORBIT_B,
}
ORBIT_BY_SUBSET = {
    subset: orbit
    for orbit, members in MULTIPLIER_ORBITS.items()
    for subset in members
}

HARD_MAX_ESTIMATED_BYTES = 8 * MIB
HARD_MAX_SECONDS = 30.0
HARD_MAX_WORK_UNITS = 25_000_000
HARD_MAX_EXACT_PATTERNS = 1 << PRIME
HARD_MAX_EVALUATIONS = 64

DEFAULT_MAX_ESTIMATED_BYTES = 1 * MIB
DEFAULT_MAX_SECONDS = 25.0
DEFAULT_MAX_WORK_UNITS = 20_000_000

EVIDENCE_STATUS = "FINITE_SYNTHETIC_P11_QUANTIZER_GRID_NULL_ONLY"
BOUNDARY_TOLERANCE_LEVEL_UNITS = (
    64.0 * np.finfo(np.float64).eps * (1 << DEFAULT_PHASE_BITS)
)


if set(ORBIT_BY_SUBSET) != set(TWO_BIN_SUBSETS):
    raise RuntimeError("p=11 multiplier orbits must partition all two-bin subsets")
if any(len(members) != 5 for members in MULTIPLIER_ORBITS.values()):
    raise RuntimeError("each p=11 multiplier orbit must contain exactly five pairs")


class PRWH1QValidationError(ValueError):
    """Raised when the frozen PRW-H1Q contract is malformed."""


class PRWH1QResourceError(RuntimeError):
    """Raised before PRW-H1Q exceeds a caller-lowered resource budget."""


def _plain_int(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise PRWH1QValidationError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise PRWH1QValidationError(f"{name} must be >= {minimum}")
    return result


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise PRWH1QValidationError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise PRWH1QValidationError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class QuantizerNullSpec:
    """Frozen scientific parameters for one p=11 exact comparison."""

    crossover: float = DEFAULT_CROSSOVER
    planted_shift: int = DEFAULT_PLANTED_SHIFT
    phase_bits: int = DEFAULT_PHASE_BITS
    origins: Tuple[float, ...] = DECLARED_ORIGINS

    def __post_init__(self) -> None:
        crossover = _finite_float(self.crossover, "crossover")
        if not 0.0 <= crossover < 0.5:
            raise PRWH1QValidationError("crossover must be in [0, 0.5)")
        shift = _plain_int(self.planted_shift, "planted_shift", 0)
        if shift >= PRIME:
            raise PRWH1QValidationError(
                f"planted_shift must be in [0, {PRIME - 1}]"
            )
        bits = _plain_int(self.phase_bits, "phase_bits", 1)
        if bits != DEFAULT_PHASE_BITS:
            raise PRWH1QValidationError(
                f"PRW-H1Q freezes phase_bits at {DEFAULT_PHASE_BITS}"
            )
        if not isinstance(self.origins, tuple):
            raise PRWH1QValidationError("origins must be a predeclared tuple")
        origins = tuple(
            _finite_float(value, "quantizer origin") for value in self.origins
        )
        if origins != DECLARED_ORIGINS:
            raise PRWH1QValidationError(
                f"PRW-H1Q freezes origins at {DECLARED_ORIGINS}"
            )
        object.__setattr__(self, "crossover", crossover)
        object.__setattr__(self, "planted_shift", shift)
        object.__setattr__(self, "phase_bits", bits)
        object.__setattr__(self, "origins", origins)


@dataclass(frozen=True)
class QuantizerNullBudget:
    """Caller-lowerable limits beneath immutable local and PRW-H1 ceilings."""

    max_estimated_bytes: int = DEFAULT_MAX_ESTIMATED_BYTES
    max_seconds: float = DEFAULT_MAX_SECONDS
    max_work_units: int = DEFAULT_MAX_WORK_UNITS
    max_exact_patterns: int = HARD_MAX_EXACT_PATTERNS
    max_evaluations: int = HARD_MAX_EVALUATIONS

    def __post_init__(self) -> None:
        integer_caps = {
            "max_estimated_bytes": min(
                HARD_MAX_ESTIMATED_BYTES,
                H1_HARD_MAX_ESTIMATED_BYTES,
            ),
            "max_work_units": min(
                HARD_MAX_WORK_UNITS,
                H1_HARD_MAX_WORK_UNITS,
            ),
            "max_exact_patterns": HARD_MAX_EXACT_PATTERNS,
            "max_evaluations": HARD_MAX_EVALUATIONS,
        }
        for name, hard_cap in integer_caps.items():
            value = _plain_int(getattr(self, name), name, 1)
            if value > hard_cap:
                raise PRWH1QValidationError(
                    f"{name} cannot exceed the hard ceiling {hard_cap}"
                )
            object.__setattr__(self, name, value)
        seconds = _finite_float(self.max_seconds, "max_seconds")
        seconds_cap = min(HARD_MAX_SECONDS, H1_HARD_MAX_SECONDS)
        if seconds <= 0.0 or seconds > seconds_cap:
            raise PRWH1QValidationError(
                f"max_seconds must be in (0, {seconds_cap}]"
            )
        object.__setattr__(self, "max_seconds", seconds)


@dataclass(frozen=True)
class _CachedPhaseCondition:
    """One phase condition with reusable template and correction state."""

    pair: Tuple[int, int]
    origin: Optional[float]
    condition: FrequencyCondition
    template_unit: np.ndarray
    correction: np.ndarray
    accounting: Mapping[str, Any]


def _check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise PRWH1QResourceError(
            "p=11 quantizer null exceeded its single aggregate deadline"
        )


def _contract_digest(spec: QuantizerNullSpec) -> str:
    payload = {
        "schema": "PRW-H1Q-v1",
        "prime": PRIME,
        "two_bin_subsets": [list(pair) for pair in TWO_BIN_SUBSETS],
        "origins": list(spec.origins),
        "phase_bits": spec.phase_bits,
        "crossover": spec.crossover,
        "planted_shift": spec.planted_shift,
        "origin_selection": "none",
        "subset_selection": "none",
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _conditions(
    spec: QuantizerNullSpec,
) -> Tuple[Tuple[Tuple[int, int], Optional[float], FrequencyCondition], ...]:
    conditions = []
    for pair in TWO_BIN_SUBSETS:
        conditions.append(
            (
                pair,
                None,
                FrequencyCondition(
                    f"p11_unquantized_{pair[0]}_{pair[1]}",
                    "phase",
                    pair,
                ),
            )
        )
        for origin_index, origin in enumerate(spec.origins):
            conditions.append(
                (
                    pair,
                    origin,
                    FrequencyCondition(
                        (
                            f"p11_q{spec.phase_bits}_{pair[0]}_{pair[1]}"
                            f"_origin_{origin_index}"
                        ),
                        "phase",
                        pair,
                        phase_bits=spec.phase_bits,
                        quantizer_origin_fraction=origin,
                    ),
                )
            )
    return tuple(conditions)


def _upstream_budget(budget: QuantizerNullBudget) -> MultifrequencyBudget:
    return MultifrequencyBudget(
        max_estimated_bytes=budget.max_estimated_bytes,
        max_seconds=budget.max_seconds,
        max_work_units=budget.max_work_units,
        max_hypotheses=PRIME,
        max_exact_patterns=budget.max_exact_patterns,
        max_prime=PRIME,
        max_types=1,
        max_nodes=1,
        max_conditions=1,
    )


def _build_phase_cache(
    bank: Any,
    conditions: Tuple[Any, ...],
    *,
    deadline: float,
) -> Tuple[Tuple[_CachedPhaseCondition, ...], np.ndarray]:
    """Cache template phases and shift corrections for the frozen grid."""

    _check_deadline(deadline)
    template_spectrum = np.fft.fft(bank.templates, axis=2)
    correction_by_pair: Dict[Tuple[int, int], np.ndarray] = {}
    shifts = np.arange(bank.prime, dtype=np.float64)
    for pair in TWO_BIN_SUBSETS:
        frequencies = np.asarray(pair, dtype=np.float64)
        correction = np.exp(
            2.0j
            * math.pi
            * frequencies[:, np.newaxis]
            * shifts[np.newaxis, :]
            / bank.prime
        )
        correction = np.ascontiguousarray(
            correction,
            dtype=np.complex128,
        )
        correction.setflags(write=False)
        correction_by_pair[pair] = correction

    cached = []
    for pair, origin, condition in conditions:
        _check_deadline(deadline)
        selected_template = template_spectrum[:, :, condition.bins]
        template_unit = _quantized_unit_phase(
            selected_template,
            condition.phase_bits,
            condition.quantizer_origin_fraction,
        )
        template_unit = np.ascontiguousarray(
            template_unit,
            dtype=np.complex128,
        )
        template_unit.setflags(write=False)
        cached.append(
            _CachedPhaseCondition(
                pair=pair,
                origin=origin,
                condition=condition,
                template_unit=template_unit,
                correction=correction_by_pair[pair],
                accounting={
                    "candidate_count": bank.hypothesis_count,
                    "feature_norm": 1.0,
                    "unit_energy_normalization": True,
                    "cached_template_fft": True,
                    "cached_template_unit_phase": True,
                    "cached_shift_correction": True,
                    "query_fft_shared_per_pattern": True,
                },
            )
        )
    template_spectrum = np.ascontiguousarray(
        template_spectrum,
        dtype=np.complex128,
    )
    template_spectrum.setflags(write=False)
    _check_deadline(deadline)
    return tuple(cached), template_spectrum


def _decode_cached_phase(
    bank: Any,
    query_spectrum: np.ndarray,
    cached: _CachedPhaseCondition,
    *,
    deadline: float,
) -> DecodeResult:
    """Apply the reference phase-score semantics to cached Fourier state."""

    _check_deadline(deadline)
    selected_query = query_spectrum[:, cached.condition.bins]
    try:
        query_unit = _quantized_unit_phase(
            selected_query,
            cached.condition.phase_bits,
            cached.condition.quantizer_origin_fraction,
        )
    except PRWH1DegenerateFeatureError as exc:
        return _abstention_result(
            cached.condition,
            bank.hypothesis_count,
            str(exc),
            cached.accounting,
        )
    cross = query_unit[np.newaxis, :, :] * np.conj(
        cached.template_unit
    )
    _check_deadline(deadline)
    scores = np.real(
        np.einsum(
            "tvk,ks->ts",
            cross,
            cached.correction,
            optimize=True,
        )
    ) / (bank.node_count * len(cached.condition.bins))
    if not np.all(np.isfinite(scores)):
        raise PRWH1ValidationError("decoder produced non-finite scores")
    winners, best, runner_up, margin = _score_summary(scores)
    scores = np.ascontiguousarray(scores, dtype=np.float64)
    scores.setflags(write=False)
    return DecodeResult(
        condition=cached.condition,
        scores=scores,
        top_states=winners,
        unique_winner=len(winners) == 1,
        abstained=False,
        abstention_reason=None,
        best_score=best,
        runner_up_score=runner_up,
        margin=margin,
        candidate_count=bank.hypothesis_count,
        accounting=cached.accounting,
    )


def _resource_plan(
    spec: QuantizerNullSpec,
    budget: QuantizerNullBudget,
    *,
    deadline: float,
) -> Tuple[Any, Tuple[Any, ...], Dict[str, Any]]:
    pattern_count = 1 << PRIME
    evaluation_count = len(TWO_BIN_SUBSETS) * (1 + len(spec.origins))
    if pattern_count > budget.max_exact_patterns:
        raise PRWH1QResourceError(
            "exact pattern count exceeds the caller-lowered PRW-H1Q limit: "
            f"{pattern_count} > {budget.max_exact_patterns}"
        )
    if evaluation_count > budget.max_evaluations:
        raise PRWH1QResourceError(
            "evaluation count exceeds the caller-lowered PRW-H1Q limit: "
            f"{evaluation_count} > {budget.max_evaluations}"
        )

    bank_estimated_bytes = estimate_bank_peak_bytes(
        prime=PRIME,
        type_count=1,
        node_count=1,
    )
    bank_estimated_work = estimate_bank_work_units(
        prime=PRIME,
        type_count=1,
        node_count=1,
    )
    if bank_estimated_bytes > budget.max_estimated_bytes:
        raise PRWH1QResourceError(
            "bank estimate exceeds the caller-lowered PRW-H1Q byte limit"
        )
    if bank_estimated_work > budget.max_work_units:
        raise PRWH1QResourceError(
            "bank estimate exceeds the caller-lowered PRW-H1Q work limit"
        )
    _check_deadline(deadline)
    bank = build_legendre_phase_bank(
        PRIME,
        np.asarray(((0,),), dtype=np.int64),
        budget=_upstream_budget(budget),
    )
    _check_deadline(deadline)
    conditions = _conditions(spec)
    if len(conditions) != evaluation_count:
        raise PRWH1QValidationError("frozen p=11 evaluation grid changed")

    slots = 2
    fft_work = PRIME * max(4, int(math.ceil(math.log2(PRIME))))
    score_work = PRIME * slots * 4
    quantization_work = slots * 16
    score_summary_work = PRIME * 4
    cached_condition_work = (
        score_work + quantization_work + score_summary_work
    )
    cache_build_work = (
        fft_work
        + len(TWO_BIN_SUBSETS) * slots * PRIME * 4
        + evaluation_count * slots * 16
    )
    shared_query_fft_work = pattern_count * fft_work
    cached_score_work = (
        pattern_count * evaluation_count * cached_condition_work
    )
    query_flip_work = pattern_count * (4 * PRIME + 16)
    outcome_partition_work = pattern_count * evaluation_count * 6
    boundary_audit_work = (
        pattern_count
        * len(spec.origins)
        * len(NONCONJUGATE_BINS)
        * 8
    )
    estimated_work = (
        bank_estimated_work
        + cache_build_work
        + shared_query_fft_work
        + cached_score_work
        + query_flip_work
        + outcome_partition_work
        + boundary_audit_work
    )

    accumulator_bytes = evaluation_count * 4 * 8
    cache_metadata_bytes = evaluation_count * 768
    template_spectrum_bytes = PRIME * 16
    cached_template_unit_bytes = evaluation_count * slots * 16
    cached_correction_bytes = (
        len(TWO_BIN_SUBSETS) * slots * PRIME * 16
    )
    pair_diagnostic_bytes = (
        math.comb(len(TWO_BIN_SUBSETS), 2)
        * len(spec.origins)
        * 4
        * 8
    )
    boundary_state_bytes = (
        len(TWO_BIN_SUBSETS) * len(spec.origins) * 8 * 8
    )
    query_state_bytes = (
        2 * PRIME * 8
        + PRIME * 16
        + 3 * slots * 16
        + PRIME * 8
    )
    cached_condition_temporary_peak_bytes = (
        3 * slots * 16 + PRIME * 8
    )
    enumeration_peak_bytes = (
        int(bank.templates.nbytes + bank.phase_signatures.nbytes)
        + template_spectrum_bytes
        + cached_template_unit_bytes
        + cached_correction_bytes
        + cache_metadata_bytes
        + accumulator_bytes
        + pair_diagnostic_bytes
        + boundary_state_bytes
        + query_state_bytes
        + cached_condition_temporary_peak_bytes
    )
    estimated_peak_bytes = max(
        bank_estimated_bytes,
        enumeration_peak_bytes,
    )
    if estimated_peak_bytes > budget.max_estimated_bytes:
        raise PRWH1QResourceError(
            "aggregate byte estimate exceeds the caller-lowered PRW-H1Q "
            f"limit: {estimated_peak_bytes} > {budget.max_estimated_bytes}"
        )
    if estimated_work > budget.max_work_units:
        raise PRWH1QResourceError(
            "aggregate work estimate exceeds the caller-lowered PRW-H1Q "
            f"limit: {estimated_work} > {budget.max_work_units}"
        )

    stronger_evaluations = len(TWO_BIN_SUBSETS) * (
        1 + len(spec.origins) ** 2
    )
    stronger_cache_build_work = (
        fft_work
        + len(TWO_BIN_SUBSETS) * slots * PRIME * 4
        + stronger_evaluations * slots * 16
    )
    stronger_estimated_work = (
        bank_estimated_work
        + stronger_cache_build_work
        + shared_query_fft_work
        + pattern_count * stronger_evaluations * cached_condition_work
        + pattern_count * stronger_evaluations * 6
        + query_flip_work
        + boundary_audit_work
    )
    plan = {
        "estimated_peak_bytes": estimated_peak_bytes,
        "estimated_work_units": estimated_work,
        "work_components": {
            "bank_build": bank_estimated_work,
            "phase_cache_build": cache_build_work,
            "shared_query_fft": shared_query_fft_work,
            "cached_condition_scoring": cached_score_work,
            "query_and_flip_stream": query_flip_work,
            "outcome_partition": outcome_partition_work,
            "boundary_audit": boundary_audit_work,
        },
        "pattern_count": pattern_count,
        "bit_count": PRIME,
        "evaluation_count": evaluation_count,
        "cached_condition_evaluation_count": (
            pattern_count * evaluation_count
        ),
        "public_reference_decoder_invocation_count": 0,
        "query_fft_count": pattern_count,
        "template_fft_count": 1,
        "per_cached_condition_estimated_work_units": (
            cached_condition_work
        ),
        "cached_condition_temporary_peak_bytes": (
            cached_condition_temporary_peak_bytes
        ),
        "max_seconds": budget.max_seconds,
        "deadline_scope": (
            "bank_build_and_all_patterns_subsets_origins_and_baselines"
        ),
        "single_aggregate_deadline": True,
        "preflight_before_first_cached_condition_evaluation": True,
        "patterns_streamed_once": True,
        "pattern_matrix_allocated": False,
        "same_query_shared_across_all_evaluations_per_pattern": True,
        "same_query_fft_shared_across_all_evaluations_per_pattern": True,
        "template_fft_cached_across_entire_batch": True,
        "shift_corrections_cached_across_entire_batch": True,
        "reference_score_and_tie_primitives_reused": True,
        "byte_scope": (
            "model_arrays_declared_temporaries_and_conservative_python_"
            "metadata_allowance_not_process_rss"
        ),
        "immutable_hard_ceilings": {
            "estimated_bytes": HARD_MAX_ESTIMATED_BYTES,
            "seconds": HARD_MAX_SECONDS,
            "work_units": HARD_MAX_WORK_UNITS,
            "exact_patterns": HARD_MAX_EXACT_PATTERNS,
            "evaluations": HARD_MAX_EVALUATIONS,
        },
        "symmetry_closed_per_bin_extension": {
            "implemented": False,
            "origin_vectors_per_pair": len(spec.origins) ** 2,
            "evaluation_count": stronger_evaluations,
            "cached_condition_evaluation_count": (
                pattern_count * stronger_evaluations
            ),
            "estimated_work_units_on_current_cached_path": (
                stronger_estimated_work
            ),
            "exceeds_local_hard_work_ceiling": (
                stronger_estimated_work > HARD_MAX_WORK_UNITS
            ),
            "exceeds_upstream_hard_work_ceiling": (
                stronger_estimated_work > H1_HARD_MAX_WORK_UNITS
            ),
            "required_condition_model": (
                "one_origin_per_selected_bin_shared_between_template_and_query"
            ),
            "current_condition_model_supports_per_bin_origins": False,
            "refusal_reason": (
                "the cached scorer preserves the current one-common-origin "
                "condition contract and the straightforward exact A4^2 path "
                "still exceeds immutable work ceilings"
            ),
        },
    }
    return bank, conditions, plan


def preflight_p11_quantizer_null(
    spec: QuantizerNullSpec = QuantizerNullSpec(),
    *,
    budget: QuantizerNullBudget = QuantizerNullBudget(),
) -> Dict[str, Any]:
    """Return the complete cost contract without running any decoder."""

    deadline = time.monotonic() + budget.max_seconds
    _, _, plan = _resource_plan(spec, budget, deadline=deadline)
    return plan


def _near_half_step(
    values: np.ndarray,
    *,
    phase_bits: int,
    origin: float,
) -> Tuple[np.ndarray, np.ndarray]:
    magnitudes = np.abs(values)
    tolerance = np.finfo(np.float64).eps * max(
        1.0, float(np.max(magnitudes, initial=0.0))
    )
    zero_magnitude = magnitudes <= tolerance
    near_boundary = np.zeros(values.shape, dtype=np.bool_)
    active = ~zero_magnitude
    if not np.any(active):
        return near_boundary, zero_magnitude
    levels = 1 << phase_bits
    scaled = (
        np.mod(
            np.angle(values[active] / magnitudes[active]),
            2.0 * math.pi,
        )
        * levels
        / (2.0 * math.pi)
        - origin
    )
    fractional = scaled - np.floor(scaled)
    near_boundary[active] = (
        np.abs(fractional - 0.5)
        <= BOUNDARY_TOLERANCE_LEVEL_UNITS
    )
    return near_boundary, zero_magnitude


def _outcome_dict() -> Dict[str, float]:
    return {
        "correct": 0.0,
        "wrong_unique": 0.0,
        "tie": 0.0,
        "abstain": 0.0,
    }


def _outcome_for_report(values: Mapping[str, float]) -> Dict[str, float]:
    return {
        "unique_correct_probability": float(values["correct"]),
        "wrong_unique_probability": float(values["wrong_unique"]),
        "tie_probability": float(values["tie"]),
        "abstain_probability": float(values["abstain"]),
    }


def _sign(value: float, tolerance: float) -> int:
    if abs(value) <= tolerance:
        return 0
    return 1 if value > 0.0 else -1


def _rank_groups(
    values: Mapping[Tuple[int, int], float],
    tolerance: float,
) -> list[list[list[int]]]:
    ordered = sorted(values, key=lambda pair: (-values[pair], pair))
    groups: list[list[Tuple[int, int]]] = []
    anchors: list[float] = []
    for pair in ordered:
        value = values[pair]
        if not groups or abs(value - anchors[-1]) > tolerance:
            groups.append([pair])
            anchors.append(value)
        else:
            groups[-1].append(pair)
    return [
        [list(pair) for pair in sorted(group)]
        for group in groups
    ]


def _orbit_summary(
    values: Mapping[Tuple[int, int], float],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for orbit, members in MULTIPLIER_ORBITS.items():
        orbit_values = [values[pair] for pair in members]
        result[orbit] = {
            "mean": math.fsum(orbit_values) / len(orbit_values),
            "spread": max(orbit_values) - min(orbit_values),
        }
    result["between_orbit_mean_difference_A_minus_B"] = (
        result["A"]["mean"] - result["B"]["mean"]
    )
    return result


def _summarize(
    spec: QuantizerNullSpec,
    outcomes: Mapping[
        Tuple[Tuple[int, int], Optional[float]],
        Mapping[str, float],
    ],
    boundary: Mapping[
        Tuple[Tuple[int, int], float],
        Mapping[str, float],
    ],
    probability_mass: float,
) -> Dict[str, Any]:
    unquantized = {
        pair: outcomes[(pair, None)]["correct"] for pair in TWO_BIN_SUBSETS
    }
    origin_values = {
        origin: {
            pair: outcomes[(pair, origin)]["correct"]
            for pair in TWO_BIN_SUBSETS
        }
        for origin in spec.origins
    }
    grid_average = {
        pair: (
            math.fsum(origin_values[origin][pair] for origin in spec.origins)
            / len(spec.origins)
        )
        for pair in TWO_BIN_SUBSETS
    }

    cells = []
    for pair in TWO_BIN_SUBSETS:
        per_origin = []
        for origin in spec.origins:
            boundary_cell = boundary[(pair, origin)]
            per_origin.append(
                {
                    "origin": origin,
                    **_outcome_for_report(outcomes[(pair, origin)]),
                    "boundary_audit": {
                        "template_near_half_step_coefficient_count": int(
                            boundary_cell["template_count"]
                        ),
                        "template_zero_magnitude_coefficient_count": int(
                            boundary_cell["template_zero_count"]
                        ),
                        "query_near_half_step_pattern_count": int(
                            boundary_cell["query_pattern_count"]
                        ),
                        "query_near_half_step_probability_mass": float(
                            boundary_cell["query_probability_mass"]
                        ),
                        "template_or_query_near_half_step_probability_mass": (
                            float(boundary_cell["any_probability_mass"])
                        ),
                        "query_zero_magnitude_pattern_count": int(
                            boundary_cell["query_zero_pattern_count"]
                        ),
                        "query_zero_magnitude_probability_mass": float(
                            boundary_cell["query_zero_probability_mass"]
                        ),
                        "template_or_query_zero_magnitude_probability_mass": (
                            float(boundary_cell["any_zero_probability_mass"])
                        ),
                    },
                }
            )
        accuracies = [item["unique_correct_probability"] for item in per_origin]
        cells.append(
            {
                "bins": list(pair),
                "multiplier_orbit": ORBIT_BY_SUBSET[pair],
                "unquantized": _outcome_for_report(outcomes[(pair, None)]),
                "quantized_by_declared_origin": per_origin,
                "declared_origin_grid_average_accuracy": grid_average[pair],
                "grid_average_effect_vs_unquantized": (
                    grid_average[pair] - unquantized[pair]
                ),
                "origin_grid_spread": max(accuracies) - min(accuracies),
            }
        )

    reversals = []
    near_tie_pairs = []
    for left_index, left in enumerate(TWO_BIN_SUBSETS):
        for right in TWO_BIN_SUBSETS[left_index + 1 :]:
            signs = tuple(
                _sign(
                    origin_values[origin][left]
                    - origin_values[origin][right],
                    EQUALITY_TOLERANCE,
                )
                for origin in spec.origins
            )
            if 1 in signs and -1 in signs:
                reversals.append(
                    {
                        "left": list(left),
                        "right": list(right),
                        "origin_order_signs": list(signs),
                    }
                )
            if 0 in signs:
                near_tie_pairs.append(
                    {
                        "left": list(left),
                        "right": list(right),
                        "origin_order_signs": list(signs),
                    }
                )

    strict_dominance = []
    for candidate in TWO_BIN_SUBSETS:
        if all(
            all(
                origin_values[origin][candidate]
                > origin_values[origin][other] + EQUALITY_TOLERANCE
                for origin in spec.origins
            )
            for other in TWO_BIN_SUBSETS
            if other != candidate
        ):
            strict_dominance.append(list(candidate))

    unquantized_spread = max(unquantized.values()) - min(
        unquantized.values()
    )
    grid_average_spread = max(grid_average.values()) - min(
        grid_average.values()
    )
    unquantized_orbits = _orbit_summary(unquantized)
    unquantized_within_orbit_exchangeable = all(
        unquantized_orbits[orbit]["spread"] <= EQUALITY_TOLERANCE
        for orbit in MULTIPLIER_ORBITS
    )
    per_origin_diagnostics = []
    for origin in spec.origins:
        values = origin_values[origin]
        per_origin_diagnostics.append(
            {
                "origin": origin,
                "rank_groups_with_tolerance": _rank_groups(
                    values, EQUALITY_TOLERANCE
                ),
                "subset_accuracy_spread": max(values.values())
                - min(values.values()),
                "orbit_diagnostic": _orbit_summary(values),
            }
        )

    return {
        "probability_mass": probability_mass,
        "cells": cells,
        "unquantized_diagnostic": {
            "rank_groups_with_tolerance": _rank_groups(
                unquantized, EQUALITY_TOLERANCE
            ),
            "subset_accuracy_spread": unquantized_spread,
            "all_subsets_exchangeable_within_tolerance": (
                unquantized_spread <= EQUALITY_TOLERANCE
            ),
            "within_each_multiplier_orbit_exchangeable_within_tolerance": (
                unquantized_within_orbit_exchangeable
            ),
            "between_orbit_equality_required": False,
            "orbit_diagnostic": unquantized_orbits,
        },
        "declared_origin_grid_average": {
            "origin_weights": [
                {"origin": origin, "weight": 1.0 / len(spec.origins)}
                for origin in spec.origins
            ],
            "rank_groups_with_tolerance": _rank_groups(
                grid_average, EQUALITY_TOLERANCE
            ),
            "subset_accuracy_spread": grid_average_spread,
            "all_subsets_exchangeable_within_tolerance": (
                grid_average_spread <= EQUALITY_TOLERANCE
            ),
            "orbit_diagnostic": _orbit_summary(grid_average),
            "stochastic_dither_implemented": False,
            "continuous_uniform_origin_integration_implemented": False,
            "post_hoc_origin_selection_permitted": False,
        },
        "per_origin_diagnostics": per_origin_diagnostics,
        "rank_stability": {
            "comparison_tolerance": EQUALITY_TOLERANCE,
            "strict_ranks_near_ties_reported": False,
            "rank_reversal_pair_count": len(reversals),
            "rank_reversal_pairs": reversals,
            "pairs_with_a_near_tie_count": len(near_tie_pairs),
            "pairs_with_a_near_tie": near_tie_pairs,
            "origin_invariant_strictly_dominant_subsets": strict_dominance,
        },
        "maximum_origin_grid_spread": max(
            cell["origin_grid_spread"] for cell in cells
        ),
        "maximum_absolute_grid_average_effect_vs_unquantized": max(
            abs(cell["grid_average_effect_vs_unquantized"])
            for cell in cells
        ),
        "formal_decisions": {
            "H1Q_0a_unquantized_within_orbit_equivariance_supported": (
                unquantized_within_orbit_exchangeable
            ),
            "H1Q_0b_no_origin_invariant_strict_dominance_supported": (
                not strict_dominance
            ),
            "origin_sensitive_pair_ordering_observed": bool(reversals),
            "between_orbit_unquantized_difference_is_pair_specific_novelty": (
                False
            ),
        },
        "boundary_audit_contract": {
            "definition": (
                "distance to a quantizer half-step in quantization-level units"
            ),
            "numerical_tolerance_level_units": (
                BOUNDARY_TOLERANCE_LEVEL_UNITS
            ),
            "upper_half_step_tie_rule": "floor(value + 0.5)",
            "exact_boundary_can_break_conjugation_symmetry": True,
            "zero_magnitudes_classified_as_boundaries": False,
            "zero_magnitudes_audited_separately": True,
        },
    }


def run_p11_quantizer_null(
    spec: QuantizerNullSpec = QuantizerNullSpec(),
    *,
    budget: QuantizerNullBudget = QuantizerNullBudget(),
) -> Dict[str, Any]:
    """Run the complete p=11 exact grid under one preflighted deadline."""

    started = time.monotonic()
    deadline = started + budget.max_seconds
    bank, conditions, plan = _resource_plan(
        spec,
        budget,
        deadline=deadline,
    )
    cached_conditions, cached_template_spectrum = _build_phase_cache(
        bank,
        conditions,
        deadline=deadline,
    )
    _check_deadline(deadline)
    truth = (0, spec.planted_shift)
    base_query = np.roll(
        bank.templates[truth[0]], truth[1], axis=1
    ).copy()
    outcome_keys = tuple((pair, origin) for pair, origin, _ in conditions)
    outcomes = {key: _outcome_dict() for key in outcome_keys}
    boundary = {
        (pair, origin): {
            "template_count": 0.0,
            "template_zero_count": 0.0,
            "query_pattern_count": 0.0,
            "query_probability_mass": 0.0,
            "any_probability_mass": 0.0,
            "query_zero_pattern_count": 0.0,
            "query_zero_probability_mass": 0.0,
            "any_zero_probability_mass": 0.0,
        }
        for pair in TWO_BIN_SUBSETS
        for origin in spec.origins
    }

    template_spectrum = cached_template_spectrum[0, 0]
    template_hits = {}
    for origin in spec.origins:
        hits, zero_magnitudes = _near_half_step(
            template_spectrum[list(NONCONJUGATE_BINS)],
            phase_bits=spec.phase_bits,
            origin=origin,
        )
        template_hits[origin] = {
            frequency: bool(hits[index])
            for index, frequency in enumerate(NONCONJUGATE_BINS)
        }
        template_zero = {
            frequency: bool(zero_magnitudes[index])
            for index, frequency in enumerate(NONCONJUGATE_BINS)
        }
        for pair in TWO_BIN_SUBSETS:
            boundary[(pair, origin)]["template_count"] = float(
                sum(template_hits[origin][frequency] for frequency in pair)
            )
            boundary[(pair, origin)]["template_zero_count"] = float(
                sum(template_zero[frequency] for frequency in pair)
            )

    probability_mass = 0.0
    for pattern in range(plan["pattern_count"]):
        _check_deadline(deadline)
        weight = bin(pattern).count("1")
        mass = spec.crossover**weight * (
            1.0 - spec.crossover
        ) ** (PRIME - weight)
        probability_mass += mass
        query = base_query.copy()
        flat_query = query.reshape(-1)
        for coordinate in range(PRIME):
            if pattern & (1 << coordinate):
                flat_query[coordinate] *= -1.0

        query_spectrum = np.fft.fft(query, axis=1)
        query_hits = {}
        for origin in spec.origins:
            hits, zero_magnitudes = _near_half_step(
                query_spectrum[0, list(NONCONJUGATE_BINS)],
                phase_bits=spec.phase_bits,
                origin=origin,
            )
            query_hits[origin] = {
                frequency: bool(hits[index])
                for index, frequency in enumerate(NONCONJUGATE_BINS)
            }
            query_zero = {
                frequency: bool(zero_magnitudes[index])
                for index, frequency in enumerate(NONCONJUGATE_BINS)
            }
            for pair in TWO_BIN_SUBSETS:
                cell = boundary[(pair, origin)]
                query_hit = any(
                    query_hits[origin][frequency] for frequency in pair
                )
                query_zero_hit = any(
                    query_zero[frequency] for frequency in pair
                )
                template_hit = cell["template_count"] > 0.0
                template_zero_hit = cell["template_zero_count"] > 0.0
                if query_hit:
                    cell["query_pattern_count"] += 1.0
                    cell["query_probability_mass"] += mass
                if query_hit or template_hit:
                    cell["any_probability_mass"] += mass
                if query_zero_hit:
                    cell["query_zero_pattern_count"] += 1.0
                    cell["query_zero_probability_mass"] += mass
                if query_zero_hit or template_zero_hit:
                    cell["any_zero_probability_mass"] += mass

        for cached in cached_conditions:
            _check_deadline(deadline)
            decoded = _decode_cached_phase(
                bank,
                query_spectrum,
                cached,
                deadline=deadline,
            )
            if decoded.candidate_count != PRIME:
                raise PRWH1QValidationError(
                    "all p=11 conditions must search exactly 11 shared shifts"
                )
            cell = outcomes[(cached.pair, cached.origin)]
            if decoded.abstained:
                cell["abstain"] += mass
            elif not decoded.unique_winner:
                cell["tie"] += mass
            elif decoded.top_states[0] == truth:
                cell["correct"] += mass
            else:
                cell["wrong_unique"] += mass
        _check_deadline(deadline)

    for key, cell in outcomes.items():
        categorized = math.fsum(cell.values())
        if not math.isclose(
            categorized,
            probability_mass,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise PRWH1QValidationError(
                f"outcome partition does not conserve mass for {key}"
            )
    summary = _summarize(
        spec,
        outcomes,
        boundary,
        probability_mass,
    )
    elapsed = time.monotonic() - started
    return {
        "status": EVIDENCE_STATUS,
        "hypothesis_id": "PRW-H1Q",
        "formal_hypotheses": {
            "H1Q_0a_unquantized_within_orbit_equivariance": (
                "exact unquantized unique-correct probability is constant "
                "within each of the two multiplier/relabeling orbits; a "
                "between-orbit difference is allowed ratio-class geometry"
            ),
            "H1Q_0b_no_origin_invariant_strict_dominance": (
                "under the four predeclared common-origin nuisance grids, no "
                "two-bin subset strictly dominates every other subset at "
                "every origin"
            ),
            "orbit_audit": (
                "two multiplier/relabeling orbits are reported diagnostically; "
                "the common-origin quantizer grid is not orbit invariant"
            ),
        },
        "contract_digest": _contract_digest(spec),
        "prime": PRIME,
        "type_count": 1,
        "node_count": 1,
        "shared_state_count": PRIME,
        "independent_shift_per_bin_permitted": False,
        "two_bin_subsets": [list(pair) for pair in TWO_BIN_SUBSETS],
        "multiplier_orbits": {
            orbit: [list(pair) for pair in members]
            for orbit, members in MULTIPLIER_ORBITS.items()
        },
        "phase_bits": spec.phase_bits,
        "declared_origins": list(spec.origins),
        "crossover": spec.crossover,
        "planted_state": list(truth),
        "selection_contract": {
            "all_subsets_retained": True,
            "all_declared_origins_retained": True,
            "evaluation_time_origin_selection_permitted": False,
            "evaluation_time_subset_selection_permitted": False,
            "primary_statistic": (
                "equal_weight_arithmetic_mean_over_declared_origin_grid"
            ),
            "same_origin_shared_by_both_bins_template_and_query": True,
            "same_patterns_and_queries_paired_across_all_conditions": True,
        },
        "resource_guard": plan,
        "elapsed_seconds": elapsed,
        "results": summary,
        "interpretation_limits": {
            "novelty_claim": False,
            "promotion_evidence": False,
            "real_retrieval_evidence": False,
            "training_cost_or_speed_evidence": False,
            "stochastic_dither_implemented": False,
            "continuous_origin_average_implemented": False,
            "common_origin_grid_multiplier_invariant": False,
            "unquantized_within_orbit_equality_required_for_null": True,
            "quantized_common_grid_orbit_equality_required_for_null": False,
            "quantized_storage_or_integer_arithmetic_implemented": False,
            "finite_synthetic_exact_probability_only": True,
        },
    }


__all__ = [
    "BOUNDARY_TOLERANCE_LEVEL_UNITS",
    "DECLARED_ORIGINS",
    "DEFAULT_CROSSOVER",
    "DEFAULT_PHASE_BITS",
    "DEFAULT_PLANTED_SHIFT",
    "EQUALITY_TOLERANCE",
    "EVIDENCE_STATUS",
    "HARD_MAX_ESTIMATED_BYTES",
    "HARD_MAX_EVALUATIONS",
    "HARD_MAX_EXACT_PATTERNS",
    "HARD_MAX_SECONDS",
    "HARD_MAX_WORK_UNITS",
    "MULTIPLIER_ORBITS",
    "NONCONJUGATE_BINS",
    "ORBIT_A",
    "ORBIT_B",
    "PRIME",
    "PRWH1QResourceError",
    "PRWH1QValidationError",
    "QuantizerNullBudget",
    "QuantizerNullSpec",
    "TWO_BIN_SUBSETS",
    "preflight_p11_quantizer_null",
    "run_p11_quantizer_null",
]
