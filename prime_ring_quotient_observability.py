"""Exact quotient-observability screen for a four-coordinate ``Z_7`` cascade.

The frozen screen treats a state ``x = (x0, x1, x2, x3)`` as equivalent to
every global translation ``x + g * (1, 1, 1, 1)``.  Its canonical observable
coordinates are therefore

``(x1 - x0, x2 - x0, x3 - x0) mod 7``.

All ``7**4 == 2,401`` states are streamed and grouped into the ``7**3 == 343``
exact quotient classes.  A declared observer exposes stage scores, margins,
ties, routes, and payload reads that depend only on those three relative
coordinates.  An otherwise identical observer tied to an external coordinate
origin is retained as a gauge-sensitive negative control.

This is a bounded algebraic falsifier.  It does not establish that every
high-dimensional representation is a projection artifact, nor does it test
learned representations, noisy measurements, approximate equality, retrieval
quality, latency, compression, or scientific novelty.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from itertools import product
from numbers import Integral, Real
from typing import Dict, Iterable, Optional, Sequence, Tuple


PROTOCOL_ID = "PRW-QUOTIENT-OBSERVABILITY-P7-L4-001"
MODULUS = 7
CASCADE_LENGTH = 4
QUOTIENT_DIMENSION = CASCADE_LENGTH - 1
STATE_COUNT = MODULUS**CASCADE_LENGTH
QUOTIENT_CLASS_COUNT = MODULUS**QUOTIENT_DIMENSION
GAUGE_ORBIT_SIZE = MODULUS

HARD_MAX_ESTIMATED_BYTES = 2_097_152
HARD_MAX_WORK_UNITS = 1_000_000
HARD_MAX_SECONDS = 30.0
DEFAULT_MAX_ESTIMATED_BYTES = 1_572_864
DEFAULT_MAX_WORK_UNITS = 600_000
DEFAULT_MAX_SECONDS = 5.0

EVIDENCE_MODE = "EXACT_BOUNDED_QUOTIENT_FALSIFICATION"
HYPOTHESIS_STATUS = "NON_PROMOTIONAL_COORDINATE_REDUNDANCY_SCREEN"
LIMITATIONS = (
    "The exact result is restricted to the additive global-shift action on "
    "four coordinates over Z_7.",
    "The declared observer is intentionally constructed from quotient "
    "coordinates; the screen verifies the implementation and its behavioral "
    "consequences rather than proving that an unknown observer must be invariant.",
    "The anchored observer is a sensitivity control tied to an external "
    "coordinate origin, not a matched learned baseline or useful model.",
    "Exact equality is tested; approximate symmetries, noise, quantization, "
    "continuous manifolds, and broken gauge actions are outside the screen.",
    "The finite-difference rank is the rank of the declared linear quotient "
    "map over F_7, not an estimate of neural intrinsic dimension.",
    "Payload reads are fixed symbolic identifiers and do not measure memory "
    "capacity, information content, recall, training cost, or runtime speed.",
    "Byte and work figures are conservative modeled estimates, not measured "
    "process RSS, allocator peaks, wall-clock benchmarks, or hardware use.",
    "The result makes no physical, perceptual, universal-dimensionality, "
    "novelty, ML-utility, or production-readiness claim.",
)


class QuotientObservabilityValidationError(ValueError):
    """Raised when an input violates the frozen exact-screen contract."""


class QuotientObservabilityResourceError(RuntimeError):
    """Raised before or during enumeration when a resource limit is crossed."""


def _plain_int(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise QuotientObservabilityValidationError(
            "{} must be an integer".format(name)
        )
    result = int(value)
    if result < minimum:
        raise QuotientObservabilityValidationError(
            "{} must be >= {}".format(name, minimum)
        )
    return result


def _canonical_residue(value: object, name: str) -> int:
    result = _plain_int(value, name)
    if result >= MODULUS:
        raise QuotientObservabilityValidationError(
            "{} must be a canonical residue in [0, {})".format(name, MODULUS)
        )
    return result


def _positive_seconds(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise QuotientObservabilityValidationError(
            "{} must be a finite positive number".format(name)
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise QuotientObservabilityValidationError(
            "{} must be a finite positive number".format(name)
        )
    return result


def _state(value: object, name: str = "state") -> Tuple[int, int, int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise QuotientObservabilityValidationError(
            "{} must be a four-value sequence".format(name)
        )
    if len(value) != CASCADE_LENGTH:
        raise QuotientObservabilityValidationError(
            "{} must contain exactly {} coordinates".format(
                name,
                CASCADE_LENGTH,
            )
        )
    normalized = tuple(
        _canonical_residue(value[index], "{}[{}]".format(name, index))
        for index in range(CASCADE_LENGTH)
    )
    return normalized  # type: ignore[return-value]


def _quotient_key(value: object, name: str = "quotient_key") -> Tuple[int, int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise QuotientObservabilityValidationError(
            "{} must be a three-value sequence".format(name)
        )
    if len(value) != QUOTIENT_DIMENSION:
        raise QuotientObservabilityValidationError(
            "{} must contain exactly {} coordinates".format(
                name,
                QUOTIENT_DIMENSION,
            )
        )
    normalized = tuple(
        _canonical_residue(value[index], "{}[{}]".format(name, index))
        for index in range(QUOTIENT_DIMENSION)
    )
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True)
class QuotientObservabilityBudget:
    """Immutable enumeration budget bounded by hard caps."""

    max_estimated_bytes: int = DEFAULT_MAX_ESTIMATED_BYTES
    max_work_units: int = DEFAULT_MAX_WORK_UNITS
    max_seconds: float = DEFAULT_MAX_SECONDS

    def __post_init__(self) -> None:
        integer_limits = (
            (
                "max_estimated_bytes",
                self.max_estimated_bytes,
                HARD_MAX_ESTIMATED_BYTES,
            ),
            ("max_work_units", self.max_work_units, HARD_MAX_WORK_UNITS),
        )
        for name, value, hard_maximum in integer_limits:
            checked = _plain_int(value, name, minimum=1)
            if checked > hard_maximum:
                raise QuotientObservabilityValidationError(
                    "{} cannot exceed immutable hard cap {}".format(
                        name,
                        hard_maximum,
                    )
                )
            object.__setattr__(self, name, checked)
        seconds = _positive_seconds(self.max_seconds, "max_seconds")
        if seconds > HARD_MAX_SECONDS:
            raise QuotientObservabilityValidationError(
                "max_seconds cannot exceed immutable hard cap {:g}".format(
                    HARD_MAX_SECONDS
                )
            )
        object.__setattr__(self, "max_seconds", seconds)


@dataclass(frozen=True)
class QuotientObservation:
    """All behavior declared visible to the bounded observer."""

    scores: Tuple[Tuple[int, ...], ...]
    margins: Tuple[int, ...]
    tie_counts: Tuple[int, ...]
    tied_routes: Tuple[Tuple[int, ...], ...]
    routes: Tuple[int, ...]
    payload_reads: Tuple[int, ...]

    def __post_init__(self) -> None:
        routes = _quotient_key(self.routes, "routes")
        if isinstance(self.scores, (str, bytes)) or not isinstance(
            self.scores,
            Sequence,
        ):
            raise QuotientObservabilityValidationError(
                "scores must be a sequence"
            )
        if len(self.scores) != QUOTIENT_DIMENSION:
            raise QuotientObservabilityValidationError(
                "scores must contain one row per quotient coordinate"
            )
        for stage, row in enumerate(self.scores):
            if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
                raise QuotientObservabilityValidationError(
                    "scores[{}] must be a sequence".format(stage)
                )
            if len(row) != MODULUS:
                raise QuotientObservabilityValidationError(
                    "scores[{}] must contain {} route scores".format(
                        stage,
                        MODULUS,
                    )
                )
        # _plain_int is nonnegative by default; score values need a separate
        # strict conversion because negative circular losses are intentional.
        score_rows = []
        for stage, row in enumerate(self.scores):
            converted = []
            for route, value in enumerate(row):
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise QuotientObservabilityValidationError(
                        "scores[{}][{}] must be an integer".format(
                            stage,
                            route,
                        )
                    )
                converted.append(int(value))
            score_rows.append(tuple(converted))
        margins = _positive_int_tuple(
            self.margins,
            "margins",
            QUOTIENT_DIMENSION,
        )
        tie_counts = _positive_int_tuple(
            self.tie_counts,
            "tie_counts",
            QUOTIENT_DIMENSION,
        )
        if isinstance(self.tied_routes, (str, bytes)) or not isinstance(
            self.tied_routes,
            Sequence,
        ):
            raise QuotientObservabilityValidationError(
                "tied_routes must be a sequence"
            )
        if len(self.tied_routes) != QUOTIENT_DIMENSION:
            raise QuotientObservabilityValidationError(
                "tied_routes must contain one tuple per stage"
            )
        normalized_ties = []
        for stage, tied in enumerate(self.tied_routes):
            if isinstance(tied, (str, bytes)) or not isinstance(tied, Sequence):
                raise QuotientObservabilityValidationError(
                    "tied_routes[{}] must be a sequence".format(stage)
                )
            normalized = tuple(
                _canonical_residue(
                    tied[index],
                    "tied_routes[{}][{}]".format(stage, index),
                )
                for index in range(len(tied))
            )
            if len(normalized) != tie_counts[stage]:
                raise QuotientObservabilityValidationError(
                    "tie_counts[{}] does not match tied_routes".format(stage)
                )
            normalized_ties.append(normalized)
        payload_reads = _nonnegative_int_tuple(
            self.payload_reads,
            "payload_reads",
            QUOTIENT_DIMENSION,
        )
        for stage, route in enumerate(routes):
            row = score_rows[stage]
            maximum = max(row)
            ties = tuple(index for index, score in enumerate(row) if score == maximum)
            ordered = sorted(row, reverse=True)
            if len(ties) == MODULUS:
                raise QuotientObservabilityValidationError(
                    "scores[{}] must retain a non-tied comparator".format(
                        stage
                    )
                )
            margin = maximum - ordered[len(ties)]
            if route not in ties:
                raise QuotientObservabilityValidationError(
                    "routes[{}] is not score-maximizing".format(stage)
                )
            if ties != normalized_ties[stage]:
                raise QuotientObservabilityValidationError(
                    "tied_routes[{}] does not match scores".format(stage)
                )
            if margins[stage] != margin:
                raise QuotientObservabilityValidationError(
                    "margins[{}] does not match scores".format(stage)
                )
            expected_payload = 1_000 + stage * MODULUS + route
            if payload_reads[stage] != expected_payload:
                raise QuotientObservabilityValidationError(
                    "payload_reads[{}] does not match the declared route table".format(
                        stage
                    )
                )
        object.__setattr__(self, "scores", tuple(score_rows))
        object.__setattr__(self, "margins", margins)
        object.__setattr__(self, "tie_counts", tie_counts)
        object.__setattr__(self, "tied_routes", tuple(normalized_ties))
        object.__setattr__(self, "routes", routes)
        object.__setattr__(self, "payload_reads", payload_reads)


def _nonnegative_int_tuple(
    values: object,
    name: str,
    expected_length: int,
) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise QuotientObservabilityValidationError(
            "{} must be a sequence".format(name)
        )
    if len(values) != expected_length:
        raise QuotientObservabilityValidationError(
            "{} must contain {} values".format(name, expected_length)
        )
    return tuple(
        _plain_int(values[index], "{}[{}]".format(name, index))
        for index in range(expected_length)
    )


def _positive_int_tuple(
    values: object,
    name: str,
    expected_length: int,
) -> Tuple[int, ...]:
    normalized = _nonnegative_int_tuple(values, name, expected_length)
    if any(value < 1 for value in normalized):
        raise QuotientObservabilityValidationError(
            "{} values must be positive".format(name)
        )
    return normalized


@dataclass(frozen=True)
class FiniteDifferenceAudit:
    """Exact modular Jacobian and kernel audit of the quotient map."""

    jacobian_rows: Tuple[Tuple[int, ...], ...]
    rank: int
    kernel_dimension: int
    null_direction_count: int
    gauge_direction: Tuple[int, ...]
    gauge_difference: Tuple[int, ...]
    gauge_direction_is_null: bool
    kernel_equals_global_gauge_span: bool
    matrix_constant_over_all_states: bool
    states_checked: int


@dataclass(frozen=True)
class QuotientResourceEstimate:
    """Conservative static preflight estimate for the exact screen."""

    states: int
    quotient_classes: int
    expected_orbit_size: int
    estimated_peak_bytes: int
    estimated_work_units: int
    component_bytes: Tuple[Tuple[str, int], ...]
    max_estimated_bytes: int
    max_work_units: int
    max_seconds: float
    states_streamed_not_materialized: bool
    class_representatives_materialized: bool
    caller_owned_inputs_included: bool
    measured_process_peak: bool


@dataclass(frozen=True)
class QuotientObservabilityAnalysis:
    """Complete exact result for the frozen p=7, L=4 quotient screen."""

    protocol_id: str
    evidence_mode: str
    modulus: int
    cascade_length: int
    total_states: int
    quotient_class_count: int
    expected_quotient_class_count: int
    class_size_min: int
    class_size_max: int
    expected_class_size: int
    all_classes_have_exact_gauge_orbit_size: bool
    invariant_observer_constant_within_every_class: bool
    distinct_invariant_observations: int
    invariant_observer_separates_all_quotient_classes: bool
    quotient_unit_perturbations_checked: int
    quotient_unit_perturbations_all_observable: bool
    anchored_control_classes_checked: int
    anchored_control_gauge_changes_checked: int
    anchored_control_gauge_changes_detected: int
    anchored_control_sensitive_in_every_class: bool
    finite_difference: FiniteDifferenceAudit
    resource_guard: QuotientResourceEstimate
    hypothesis_status: str
    promotion_eligible: bool
    universal_dimension_claim: bool
    physical_claim: bool
    novelty_claim: bool
    ml_utility_claim: bool
    limitations: Tuple[str, ...]


def apply_global_gauge(
    state: Sequence[int],
    shift: int,
) -> Tuple[int, int, int, int]:
    """Translate every coordinate by one canonical ``Z_7`` shift."""

    normalized = _state(state)
    gauge = _canonical_residue(shift, "shift")
    return tuple((coordinate + gauge) % MODULUS for coordinate in normalized)  # type: ignore[return-value]


def quotient_key(state: Sequence[int]) -> Tuple[int, int, int]:
    """Return the exact ordered global-shift quotient coordinates."""

    normalized = _state(state)
    anchor = normalized[0]
    return tuple(
        (normalized[index] - anchor) % MODULUS
        for index in range(1, CASCADE_LENGTH)
    )  # type: ignore[return-value]


def canonicalize_state(
    state: Sequence[int],
) -> Tuple[int, int, int, int]:
    """Return the unique representative whose first coordinate is zero."""

    key = quotient_key(state)
    return (0,) + key


def _circular_distance(left: int, right: int) -> int:
    forward = (left - right) % MODULUS
    backward = (right - left) % MODULUS
    return min(forward, backward)


def _observe_routes(routes: Sequence[int]) -> QuotientObservation:
    normalized = _quotient_key(routes, "routes")
    score_rows = []
    margins = []
    tie_counts = []
    tied_routes = []
    payload_reads = []
    for stage, target in enumerate(normalized):
        row = tuple(
            -(_circular_distance(candidate, target) ** 2)
            for candidate in range(MODULUS)
        )
        maximum = max(row)
        tied = tuple(
            candidate
            for candidate, score in enumerate(row)
            if score == maximum
        )
        ordered = sorted(row, reverse=True)
        score_rows.append(row)
        margins.append(maximum - ordered[len(tied)])
        tie_counts.append(len(tied))
        tied_routes.append(tied)
        payload_reads.append(1_000 + stage * MODULUS + target)
    return QuotientObservation(
        scores=tuple(score_rows),
        margins=tuple(margins),
        tie_counts=tuple(tie_counts),
        tied_routes=tuple(tied_routes),
        routes=normalized,
        payload_reads=tuple(payload_reads),
    )


def observe_quotient(state: Sequence[int]) -> QuotientObservation:
    """Observe only the canonical quotient coordinates."""

    return _observe_routes(quotient_key(state))


def observe_anchored(state: Sequence[int]) -> QuotientObservation:
    """Negative control that incorrectly treats the ambient origin as absolute.

    It agrees with ``observe_quotient`` on canonical representatives but uses
    raw coordinates ``x1, x2, x3`` after a global gauge shift.
    """

    normalized = _state(state)
    return _observe_routes(normalized[1:])


def quotient_difference(
    state: Sequence[int],
    direction: Sequence[int],
) -> Tuple[int, int, int]:
    """Return the exact finite difference of the quotient map over ``F_7``."""

    origin = _state(state)
    displacement = _state(direction, "direction")
    shifted = tuple(
        (origin[index] + displacement[index]) % MODULUS
        for index in range(CASCADE_LENGTH)
    )
    before = quotient_key(origin)
    after = quotient_key(shifted)
    return tuple(
        (after[index] - before[index]) % MODULUS
        for index in range(QUOTIENT_DIMENSION)
    )  # type: ignore[return-value]


def finite_difference_matrix(
    state: Sequence[int],
) -> Tuple[Tuple[int, ...], ...]:
    """Return the 3-by-4 modular finite-difference Jacobian at ``state``."""

    origin = _state(state)
    columns = []
    for coordinate in range(CASCADE_LENGTH):
        direction = tuple(
            1 if index == coordinate else 0
            for index in range(CASCADE_LENGTH)
        )
        columns.append(quotient_difference(origin, direction))
    return tuple(
        tuple(columns[column][row] for column in range(CASCADE_LENGTH))
        for row in range(QUOTIENT_DIMENSION)
    )


def _rank_modulus(matrix: Sequence[Sequence[int]]) -> int:
    rows = [list(row) for row in matrix]
    if not rows:
        return 0
    column_count = len(rows[0])
    rank = 0
    for column in range(column_count):
        pivot = next(
            (
                row
                for row in range(rank, len(rows))
                if rows[row][column] % MODULUS
            ),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inverse = pow(rows[rank][column] % MODULUS, -1, MODULUS)
        rows[rank] = [
            (value * inverse) % MODULUS for value in rows[rank]
        ]
        for row in range(len(rows)):
            if row == rank:
                continue
            factor = rows[row][column] % MODULUS
            if factor:
                rows[row] = [
                    (left - factor * right) % MODULUS
                    for left, right in zip(rows[row], rows[rank])
                ]
        rank += 1
        if rank == len(rows):
            break
    return rank


def _iter_states() -> Iterable[Tuple[int, int, int, int]]:
    return product(range(MODULUS), repeat=CASCADE_LENGTH)


def _check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise QuotientObservabilityResourceError(
            "exact quotient screen exceeded its declared deadline"
        )


def estimate_quotient_resources(
    budget: Optional[QuotientObservabilityBudget] = None,
) -> QuotientResourceEstimate:
    """Return and enforce the frozen conservative preflight estimate."""

    if budget is None:
        budget = QuotientObservabilityBudget()
    if not isinstance(budget, QuotientObservabilityBudget):
        raise QuotientObservabilityValidationError(
            "budget must be a QuotientObservabilityBudget"
        )
    components = (
        ("class_count_dictionary_and_keys", 196_608),
        ("class_observation_dictionary_and_values", 786_432),
        ("finite_difference_and_enumeration_scratch", 131_072),
        ("result_objects_and_interpreter_headroom", 196_608),
    )
    estimated_peak_bytes = sum(value for _name, value in components)
    # A deliberately conservative abstract operation count.  It covers both
    # observers, score/tie/payload derivation, class checks, every-state
    # finite differences, all kernel directions, and quotient perturbations.
    estimated_work_units = (
        STATE_COUNT * 180
        + QUOTIENT_CLASS_COUNT * 120
        + STATE_COUNT * (CASCADE_LENGTH + 1) * QUOTIENT_DIMENSION
    )
    if estimated_peak_bytes > budget.max_estimated_bytes:
        raise QuotientObservabilityResourceError(
            "estimated bytes exceed budget: {} > {}".format(
                estimated_peak_bytes,
                budget.max_estimated_bytes,
            )
        )
    if estimated_work_units > budget.max_work_units:
        raise QuotientObservabilityResourceError(
            "estimated work exceeds budget: {} > {}".format(
                estimated_work_units,
                budget.max_work_units,
            )
        )
    return QuotientResourceEstimate(
        states=STATE_COUNT,
        quotient_classes=QUOTIENT_CLASS_COUNT,
        expected_orbit_size=GAUGE_ORBIT_SIZE,
        estimated_peak_bytes=estimated_peak_bytes,
        estimated_work_units=estimated_work_units,
        component_bytes=components,
        max_estimated_bytes=budget.max_estimated_bytes,
        max_work_units=budget.max_work_units,
        max_seconds=budget.max_seconds,
        states_streamed_not_materialized=True,
        class_representatives_materialized=True,
        caller_owned_inputs_included=False,
        measured_process_peak=False,
    )


def _finite_difference_audit(deadline: float) -> FiniteDifferenceAudit:
    expected = (
        (MODULUS - 1, 1, 0, 0),
        (MODULUS - 1, 0, 1, 0),
        (MODULUS - 1, 0, 0, 1),
    )
    constant = True
    checked = 0
    for state in _iter_states():
        _check_deadline(deadline)
        if finite_difference_matrix(state) != expected:
            constant = False
        checked += 1
    gauge_direction = (1,) * CASCADE_LENGTH
    gauge_difference = quotient_difference(
        (0,) * CASCADE_LENGTH,
        gauge_direction,
    )
    null_directions = []
    for direction in _iter_states():
        _check_deadline(deadline)
        if quotient_difference((0,) * CASCADE_LENGTH, direction) == (
            0,
        ) * QUOTIENT_DIMENSION:
            null_directions.append(direction)
    expected_kernel = {
        (shift,) * CASCADE_LENGTH for shift in range(MODULUS)
    }
    return FiniteDifferenceAudit(
        jacobian_rows=expected,
        rank=_rank_modulus(expected),
        kernel_dimension=CASCADE_LENGTH - _rank_modulus(expected),
        null_direction_count=len(null_directions),
        gauge_direction=gauge_direction,
        gauge_difference=gauge_difference,
        gauge_direction_is_null=gauge_difference
        == (0,) * QUOTIENT_DIMENSION,
        kernel_equals_global_gauge_span=set(null_directions)
        == expected_kernel,
        matrix_constant_over_all_states=constant,
        states_checked=checked,
    )


def analyze_quotient_observability(
    *,
    budget: Optional[QuotientObservabilityBudget] = None,
) -> QuotientObservabilityAnalysis:
    """Run the complete exact p=7, L=4 quotient-observability screen."""

    if budget is None:
        budget = QuotientObservabilityBudget()
    if not isinstance(budget, QuotientObservabilityBudget):
        raise QuotientObservabilityValidationError(
            "budget must be a QuotientObservabilityBudget"
        )
    started = time.monotonic()
    deadline = started + budget.max_seconds
    resources = estimate_quotient_resources(budget)
    _check_deadline(deadline)

    class_counts: Dict[Tuple[int, int, int], int] = {}
    class_observations: Dict[
        Tuple[int, int, int],
        QuotientObservation,
    ] = {}
    invariant_constant = True
    total_states = 0
    for state in _iter_states():
        _check_deadline(deadline)
        key = quotient_key(state)
        observation = observe_quotient(state)
        if key in class_observations:
            if observation != class_observations[key]:
                invariant_constant = False
        else:
            class_observations[key] = observation
        class_counts[key] = class_counts.get(key, 0) + 1
        total_states += 1

    quotient_perturbations_checked = 0
    quotient_perturbations_observable = True
    anchored_changes_checked = 0
    anchored_changes_detected = 0
    anchored_sensitive_classes = 0
    for key in product(range(MODULUS), repeat=QUOTIENT_DIMENSION):
        _check_deadline(deadline)
        canonical = (0,) + key
        base = observe_quotient(canonical)
        for coordinate in range(QUOTIENT_DIMENSION):
            perturbed_key = list(key)
            perturbed_key[coordinate] = (
                perturbed_key[coordinate] + 1
            ) % MODULUS
            perturbed = observe_quotient((0,) + tuple(perturbed_key))
            quotient_perturbations_checked += 1
            if perturbed == base:
                quotient_perturbations_observable = False

        anchored_base = observe_anchored(canonical)
        if anchored_base != base:
            raise AssertionError(
                "anchored control must agree at the canonical origin"
            )
        class_is_sensitive = False
        for shift in range(1, MODULUS):
            shifted = apply_global_gauge(canonical, shift)
            anchored_changes_checked += 1
            if observe_anchored(shifted) != anchored_base:
                anchored_changes_detected += 1
                class_is_sensitive = True
        if class_is_sensitive:
            anchored_sensitive_classes += 1

    finite_difference = _finite_difference_audit(deadline)
    class_sizes = tuple(class_counts.values())
    distinct_observations = len(set(class_observations.values()))
    return QuotientObservabilityAnalysis(
        protocol_id=PROTOCOL_ID,
        evidence_mode=EVIDENCE_MODE,
        modulus=MODULUS,
        cascade_length=CASCADE_LENGTH,
        total_states=total_states,
        quotient_class_count=len(class_counts),
        expected_quotient_class_count=QUOTIENT_CLASS_COUNT,
        class_size_min=min(class_sizes),
        class_size_max=max(class_sizes),
        expected_class_size=GAUGE_ORBIT_SIZE,
        all_classes_have_exact_gauge_orbit_size=all(
            size == GAUGE_ORBIT_SIZE for size in class_sizes
        ),
        invariant_observer_constant_within_every_class=invariant_constant,
        distinct_invariant_observations=distinct_observations,
        invariant_observer_separates_all_quotient_classes=(
            distinct_observations == QUOTIENT_CLASS_COUNT
        ),
        quotient_unit_perturbations_checked=quotient_perturbations_checked,
        quotient_unit_perturbations_all_observable=(
            quotient_perturbations_observable
        ),
        anchored_control_classes_checked=QUOTIENT_CLASS_COUNT,
        anchored_control_gauge_changes_checked=anchored_changes_checked,
        anchored_control_gauge_changes_detected=anchored_changes_detected,
        anchored_control_sensitive_in_every_class=(
            anchored_sensitive_classes == QUOTIENT_CLASS_COUNT
        ),
        finite_difference=finite_difference,
        resource_guard=resources,
        hypothesis_status=HYPOTHESIS_STATUS,
        promotion_eligible=False,
        universal_dimension_claim=False,
        physical_claim=False,
        novelty_claim=False,
        ml_utility_claim=False,
        limitations=LIMITATIONS,
    )


__all__ = (
    "CASCADE_LENGTH",
    "DEFAULT_MAX_ESTIMATED_BYTES",
    "DEFAULT_MAX_SECONDS",
    "DEFAULT_MAX_WORK_UNITS",
    "EVIDENCE_MODE",
    "GAUGE_ORBIT_SIZE",
    "HARD_MAX_ESTIMATED_BYTES",
    "HARD_MAX_SECONDS",
    "HARD_MAX_WORK_UNITS",
    "HYPOTHESIS_STATUS",
    "LIMITATIONS",
    "MODULUS",
    "PROTOCOL_ID",
    "QUOTIENT_CLASS_COUNT",
    "QUOTIENT_DIMENSION",
    "STATE_COUNT",
    "FiniteDifferenceAudit",
    "QuotientObservation",
    "QuotientObservabilityAnalysis",
    "QuotientObservabilityBudget",
    "QuotientObservabilityResourceError",
    "QuotientObservabilityValidationError",
    "QuotientResourceEstimate",
    "analyze_quotient_observability",
    "apply_global_gauge",
    "canonicalize_state",
    "estimate_quotient_resources",
    "finite_difference_matrix",
    "observe_anchored",
    "observe_quotient",
    "quotient_difference",
    "quotient_key",
)
