"""Exact restricted joint-plank spectrum screen for PRW-JO1.

The frozen cell uses the p=11, eight-layer, two-type, typed16 bank.  Ten
orbit-restricted slope actions are available and every two-action schedule
with repetition is enumerated.  Fixed planks are also concatenated directly:
that flattened word is mathematically identical to the stack and is retained
as an invariant control.

The result can establish shell shaping inside this restricted action library.
It cannot establish a new stacking mechanism, novelty, exact decoder error,
or systems utility.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import combinations_with_replacement
from math import comb
import random
from typing import Dict, Iterable, Tuple

from prime_ring_rb10_contract import (
    ArtifactEnvelope,
    Deadline,
    ExperimentBudget,
    ResourceEstimate,
    RunContract,
    preflight_experiment,
)


PROTOCOL_ID = "CHELATEDAI-PRW-JO1-P11-K2-v1"
HYPOTHESIS_ID = "PRW-JO1"
FLAT_HYPOTHESIS_ID = "PRW-JO1-FLAT-EQUIVALENCE"
CONSTRUCTION_HYPOTHESIS_ID = "PRW-JO1-CONSTRUCTION"
MODULUS = 11
LAYERS = 8
TYPE_COUNT = 2
MASK_COUNT = 16
STATE_COUNT = TYPE_COUNT * MODULUS * MASK_COUNT
COORDINATES_PER_PLANK = LAYERS * MODULUS
ACTION_IDS = tuple(range(1, MODULUS))
SCHEDULE_LENGTH = 2
SCHEDULE_COUNT = 55
CROSSOVER_NUMERATOR = 1
CROSSOVER_DENOMINATOR = 5
COMMON_CHANNEL_ID = "BSC-Q-1-5-IID-BIPOLAR-CHIP-FLIP"
TIE_POLICY_ID = "PAIRWISE-TIE-OR-BETTER"
DECISION_RULE_ID = "COMPLETE-LOW-DISTANCE-LEXIMIN-THEN-PAIRWISE-TAIL"
RANDOM_CONTROL_SEED = 4691
EVIDENCE_MODE = "EXACT_BOUNDED_RESTRICTED_CONSTRUCTION_METHOD_DEV"

LIMITATIONS = (
    "Fixed planks and their byte-identical flattened concatenation are the same code.",
    "The exact rational tail sum is a pairwise union bound, not exact event-union probability.",
    "Only ten slope actions and all 55 two-action schedules at p=11 are searched.",
    "Known complementary-sequence, interleaver, and unrestricted matched-cost code controls remain required before any positive construction claim.",
    "Two canonical representatives are checked through the frozen type/shift/mask symmetry; no corpus or learned model is used.",
    "Modeled bytes are not process RSS and the cooperative deadline is not preemptive.",
    "A shell-shaping lead is not evidence of retrieval, training, latency, memory, or cost utility.",
    "No novelty, production, physical-dimensionality, gravity, or resonance claim is made.",
)


class JointOrbitValidationError(ValueError):
    """Raised when the frozen JO1 contract is violated."""


@dataclass(frozen=True, order=True)
class OrbitState:
    type_id: int
    shift: int
    mask_id: int


@dataclass(frozen=True)
class SpectrumSummary:
    shells: Tuple[Tuple[int, int], ...]
    competitor_count: int
    minimum_distance: int
    minimum_multiplicity: int
    maximum_distance: int
    exact_pairwise_tail_union_bound_numerator: str
    exact_pairwise_tail_union_bound_denominator: str

    @property
    def expanded_distances(self) -> Tuple[int, ...]:
        return tuple(distance for distance, multiplicity in self.shells for _ in range(multiplicity))


@dataclass(frozen=True)
class ScheduleSummary:
    actions: Tuple[int, int]
    type_zero_spectrum: SpectrumSummary
    type_one_spectrum: SpectrumSummary
    representative_spectra_equal: bool


@dataclass(frozen=True)
class JointOrbitSpectrumResult:
    protocol_id: str
    hypothesis_id: str
    modulus: int
    layers: int
    state_count: int
    coordinates_per_plank: int
    schedule_length: int
    searched_schedule_count: int
    common_channel_id: str
    crossover_numerator: int
    crossover_denominator: int
    aligned_repetition: ScheduleSummary
    candidate_schedule: ScheduleSummary
    complementary_control: ScheduleSummary
    seeded_random_control: ScheduleSummary
    seeded_random_control_seed: int
    best_restricted_actions: Tuple[Tuple[int, int], ...]
    best_restricted_tie_count: int
    candidate_is_restricted_optimum: bool
    candidate_improves_minimum_distance: bool
    candidate_reduces_minimum_multiplicity: bool
    candidate_tail_bound_below_repetition: bool
    flattened_distance_equality_all_competitors: bool
    flattened_spectrum_equality: bool
    flat_equivalence_status: str
    construction_status: str
    known_design_controls_complete: bool
    unrestricted_matched_cost_control_complete: bool
    promotion_eligible: bool
    novelty_claim: bool
    evidence_mode: str
    resource_guard: ResourceEstimate
    limitations: Tuple[str, ...]

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _legendre_carrier() -> Tuple[int, ...]:
    return (1,) + tuple(
        1 if pow(coordinate, (MODULUS - 1) // 2, MODULUS) == 1 else -1 for coordinate in range(1, MODULUS)
    )


def _typed16_masks() -> Tuple[Tuple[int, ...], ...]:
    """Return bipolar RM(1,3) affine truth tables in canonical coefficient order."""

    masks = []
    for coefficients in range(MASK_COUNT):
        values = []
        for layer in range(LAYERS):
            parity = (coefficients >> 0) & 1
            parity ^= ((coefficients >> 1) & 1) & ((layer >> 0) & 1)
            parity ^= ((coefficients >> 2) & 1) & ((layer >> 1) & 1)
            parity ^= ((coefficients >> 3) & 1) & ((layer >> 2) & 1)
            values.append(-1 if parity else 1)
        masks.append(tuple(values))
    if len(set(masks)) != MASK_COUNT:
        raise AssertionError("typed16 construction must contain 16 unique masks")
    return tuple(masks)


LEGENDRE_CARRIER = _legendre_carrier()
TYPED16_MASKS = _typed16_masks()
STATES = tuple(
    OrbitState(type_id, shift, mask_id)
    for type_id in range(TYPE_COUNT)
    for shift in range(MODULUS)
    for mask_id in range(MASK_COUNT)
)
TYPE_ZERO_REPRESENTATIVE = OrbitState(0, 0, 0)
TYPE_ONE_REPRESENTATIVE = OrbitState(1, 0, 0)


def _validated_action(action: object, name: str) -> int:
    if type(action) is not int or action not in ACTION_IDS:
        raise JointOrbitValidationError("{} must be a plain integer in {}".format(name, ACTION_IDS))
    return action


def _validated_schedule(actions: object) -> Tuple[int, int]:
    if type(actions) is not tuple or len(actions) != SCHEDULE_LENGTH:
        raise JointOrbitValidationError("actions must be a plain two-item tuple")
    checked = tuple(_validated_action(action, "actions[{}]".format(index)) for index, action in enumerate(actions))
    return checked


def orbit_word(action: object, state: object) -> Tuple[int, ...]:
    """Build one exact bipolar word for a frozen state/action pair."""

    checked_action = _validated_action(action, "action")
    if type(state) is not OrbitState or state not in STATES:
        raise JointOrbitValidationError("state must be one frozen OrbitState")
    mask = TYPED16_MASKS[state.mask_id]
    word = []
    for layer in range(LAYERS):
        type_phase = checked_action * layer if state.type_id else 0
        phase = (state.shift + type_phase) % MODULUS
        word.extend(mask[layer] * LEGENDRE_CARRIER[(coordinate - phase) % MODULUS] for coordinate in range(MODULUS))
    return tuple(word)


def _hamming(left: Tuple[int, ...], right: Tuple[int, ...]) -> int:
    if len(left) != len(right):
        raise AssertionError("internal words must have equal length")
    return sum(left_value != right_value for left_value, right_value in zip(left, right))


def _pairwise_tie_or_better_tail(distance: int) -> Fraction:
    threshold = (distance + 1) // 2
    numerator = sum(
        comb(distance, flips)
        * CROSSOVER_NUMERATOR**flips
        * (CROSSOVER_DENOMINATOR - CROSSOVER_NUMERATOR) ** (distance - flips)
        for flips in range(threshold, distance + 1)
    )
    denominator = CROSSOVER_DENOMINATOR**distance
    return Fraction(numerator, denominator)


def _spectrum_summary(distances: Iterable[int]) -> SpectrumSummary:
    counts = Counter(distances)
    if not counts:
        raise AssertionError("a representative must have competitors")
    shells = tuple(sorted((int(distance), count) for distance, count in counts.items()))
    tail = sum(
        (multiplicity * _pairwise_tie_or_better_tail(distance) for distance, multiplicity in shells),
        Fraction(0, 1),
    )
    competitor_count = sum(multiplicity for _distance, multiplicity in shells)
    return SpectrumSummary(
        shells=shells,
        competitor_count=competitor_count,
        minimum_distance=shells[0][0],
        minimum_multiplicity=shells[0][1],
        maximum_distance=shells[-1][0],
        exact_pairwise_tail_union_bound_numerator=str(tail.numerator),
        exact_pairwise_tail_union_bound_denominator=str(tail.denominator),
    )


def _action_fingerprint(
    action: int,
    truth: OrbitState,
    deadline: Deadline,
) -> Tuple[int, ...]:
    truth_word = orbit_word(action, truth)
    distances = []
    for index, state in enumerate(STATES):
        if index % 64 == 0:
            deadline.check("jo1_action_fingerprint")
        if state == truth:
            continue
        distances.append(_hamming(truth_word, orbit_word(action, state)))
    return tuple(distances)


def _schedule_summary(
    actions: Tuple[int, int],
    fingerprints: Dict[Tuple[int, OrbitState], Tuple[int, ...]],
) -> ScheduleSummary:
    checked = _validated_schedule(actions)
    spectra = []
    for truth in (TYPE_ZERO_REPRESENTATIVE, TYPE_ONE_REPRESENTATIVE):
        left = fingerprints[(checked[0], truth)]
        right = fingerprints[(checked[1], truth)]
        spectra.append(_spectrum_summary(left_value + right_value for left_value, right_value in zip(left, right)))
    return ScheduleSummary(
        actions=checked,
        type_zero_spectrum=spectra[0],
        type_one_spectrum=spectra[1],
        representative_spectra_equal=(spectra[0].shells == spectra[1].shells),
    )


def _schedule_key(summary: ScheduleSummary) -> Tuple[int, ...]:
    """Leximax of sorted low distances implements the frozen shell ordering."""

    if not summary.representative_spectra_equal:
        raise AssertionError("representative symmetry must hold")
    return summary.type_zero_spectrum.expanded_distances


def _tail_fraction(summary: SpectrumSummary) -> Fraction:
    return Fraction(
        int(summary.exact_pairwise_tail_union_bound_numerator),
        int(summary.exact_pairwise_tail_union_bound_denominator),
    )


def _flattened_equivalence(
    actions: Tuple[int, int],
    fingerprints: Dict[Tuple[int, OrbitState], Tuple[int, ...]],
    deadline: Deadline,
) -> Tuple[bool, bool]:
    all_equal = True
    flattened_spectra = []
    summed_spectra = []
    for truth in (TYPE_ZERO_REPRESENTATIVE, TYPE_ONE_REPRESENTATIVE):
        truth_flat = tuple(value for action in actions for value in orbit_word(action, truth))
        flattened_distances = []
        summed_distances = []
        competitor_index = 0
        for index, state in enumerate(STATES):
            if index % 64 == 0:
                deadline.check("jo1_flat_equivalence")
            if state == truth:
                continue
            candidate_flat = tuple(value for action in actions for value in orbit_word(action, state))
            flat_distance = _hamming(truth_flat, candidate_flat)
            summed_distance = sum(fingerprints[(action, truth)][competitor_index] for action in actions)
            all_equal = all_equal and flat_distance == summed_distance
            flattened_distances.append(flat_distance)
            summed_distances.append(summed_distance)
            competitor_index += 1
        flattened_spectra.append(_spectrum_summary(flattened_distances).shells)
        summed_spectra.append(_spectrum_summary(summed_distances).shells)
    return all_equal, flattened_spectra == summed_spectra


def preflight_joint_orbit_screen(
    budget: ExperimentBudget = ExperimentBudget(),
) -> ResourceEstimate:
    """Admit the entire 55-schedule cell before word construction."""

    fingerprint_count = len(ACTION_IDS) * 2 * (STATE_COUNT - 1)
    word_comparisons = len(ACTION_IDS) * 2 * (STATE_COUNT - 1) * COORDINATES_PER_PLANK
    schedule_additions = SCHEDULE_COUNT * 2 * (STATE_COUNT - 1)
    flattened_checks = 2 * (STATE_COUNT - 1) * 2 * COORDINATES_PER_PLANK
    component_bytes = (
        ("states_and_masks", STATE_COUNT * 64 + MASK_COUNT * LAYERS * 8),
        ("distance_fingerprints", fingerprint_count * 40),
        ("transient_words", 4 * COORDINATES_PER_PLANK * 32),
        ("schedule_summaries", SCHEDULE_COUNT * 4096),
        ("result_and_exact_tail_allowance", 524_288),
    )
    component_work = (
        ("fingerprint_chip_comparisons", word_comparisons),
        ("schedule_distance_additions", schedule_additions),
        ("flattened_equivalence_comparisons", flattened_checks),
        ("exact_tail_arithmetic_allowance", 500_000),
    )
    return preflight_experiment(
        assignment_count=STATE_COUNT,
        node_count=1,
        factor_arities=(),
        component_bytes=component_bytes,
        component_work=component_work,
        budget=budget,
    )


def make_run_contract(
    budget: ExperimentBudget = ExperimentBudget(),
) -> RunContract:
    return RunContract.create(
        stage_id="PRW-JO1-P11-K2",
        hypothesis_id=HYPOTHESIS_ID,
        decision_rule_id=DECISION_RULE_ID,
        channel_id=COMMON_CHANNEL_ID,
        tie_policy_id=TIE_POLICY_ID,
        control_ids=(
            "BYTE_IDENTICAL_FLATTENED_CONCATENATION",
            "ALIGNED_REPETITION_1_1",
            "DUPLICATE_ACTION",
            "COMPLEMENTARY_1_10",
            "SEEDED_RANDOM_PAIR",
            "EXHAUSTIVE_RESTRICTED_LIBRARY_OPTIMUM",
        ),
        seeds=(RANDOM_CONTROL_SEED,),
        parameters={
            "modulus": MODULUS,
            "layers": LAYERS,
            "mask_family": "RM(1,3)_typed16",
            "state_count": STATE_COUNT,
            "actions": list(ACTION_IDS),
            "schedule_length": SCHEDULE_LENGTH,
            "crossover": [
                CROSSOVER_NUMERATOR,
                CROSSOVER_DENOMINATOR,
            ],
        },
        budget=budget,
    )


def analyze_joint_orbit_spectrum(
    *,
    budget: ExperimentBudget = ExperimentBudget(),
) -> JointOrbitSpectrumResult:
    """Enumerate the complete frozen restricted JO1 schedule library."""

    resource = preflight_joint_orbit_screen(budget)
    deadline = Deadline.start(budget)
    fingerprints: Dict[Tuple[int, OrbitState], Tuple[int, ...]] = {}
    for action in ACTION_IDS:
        for truth in (TYPE_ZERO_REPRESENTATIVE, TYPE_ONE_REPRESENTATIVE):
            fingerprints[(action, truth)] = _action_fingerprint(
                action,
                truth,
                deadline,
            )

    schedules = []
    for actions in combinations_with_replacement(ACTION_IDS, SCHEDULE_LENGTH):
        deadline.check("jo1_schedule_enumeration")
        schedules.append(_schedule_summary(tuple(actions), fingerprints))
    if len(schedules) != SCHEDULE_COUNT:
        raise AssertionError("frozen action library must contain 55 schedules")
    if not all(item.representative_spectra_equal for item in schedules):
        raise AssertionError("type representative spectra must agree")

    best_key = max(_schedule_key(item) for item in schedules)
    best = tuple(item for item in schedules if _schedule_key(item) == best_key)
    best_actions = tuple(item.actions for item in best)
    by_actions = {item.actions: item for item in schedules}
    aligned = by_actions[(1, 1)]
    candidate = by_actions[(1, 5)]
    complementary = by_actions[(1, 10)]
    random_generator = random.Random(RANDOM_CONTROL_SEED)
    random_actions = tuple(
        sorted(
            (
                random_generator.choice(ACTION_IDS),
                random_generator.choice(ACTION_IDS),
            )
        )
    )
    random_control = by_actions[random_actions]
    flat_distances_equal, flat_spectra_equal = _flattened_equivalence(
        candidate.actions,
        fingerprints,
        deadline,
    )
    candidate_spectrum = candidate.type_zero_spectrum
    aligned_spectrum = aligned.type_zero_spectrum
    candidate_tail_lower = _tail_fraction(candidate_spectrum) < _tail_fraction(aligned_spectrum)
    deadline.check("jo1_finalize")

    return JointOrbitSpectrumResult(
        protocol_id=PROTOCOL_ID,
        hypothesis_id=HYPOTHESIS_ID,
        modulus=MODULUS,
        layers=LAYERS,
        state_count=STATE_COUNT,
        coordinates_per_plank=COORDINATES_PER_PLANK,
        schedule_length=SCHEDULE_LENGTH,
        searched_schedule_count=len(schedules),
        common_channel_id=COMMON_CHANNEL_ID,
        crossover_numerator=CROSSOVER_NUMERATOR,
        crossover_denominator=CROSSOVER_DENOMINATOR,
        aligned_repetition=aligned,
        candidate_schedule=candidate,
        complementary_control=complementary,
        seeded_random_control=random_control,
        seeded_random_control_seed=RANDOM_CONTROL_SEED,
        best_restricted_actions=best_actions,
        best_restricted_tie_count=len(best),
        candidate_is_restricted_optimum=(candidate.actions in best_actions),
        candidate_improves_minimum_distance=(candidate_spectrum.minimum_distance > aligned_spectrum.minimum_distance),
        candidate_reduces_minimum_multiplicity=(
            candidate_spectrum.minimum_distance == aligned_spectrum.minimum_distance
            and candidate_spectrum.minimum_multiplicity < aligned_spectrum.minimum_multiplicity
        ),
        candidate_tail_bound_below_repetition=candidate_tail_lower,
        flattened_distance_equality_all_competitors=flat_distances_equal,
        flattened_spectrum_equality=flat_spectra_equal,
        flat_equivalence_status=(
            "EXACT_EQUALITY_CONFIRMED_STACK_IS_ONE_LONGER_CODE"
            if flat_distances_equal and flat_spectra_equal
            else "INVARIANT_FAILURE"
        ),
        construction_status=(
            "RESTRICTED_SHELL_SHAPING_LEAD_CONTROLS_INCOMPLETE"
            if candidate.actions in best_actions
            and candidate_spectrum.minimum_multiplicity < aligned_spectrum.minimum_multiplicity
            else "NO_RESTRICTED_SHELL_SHAPING_LEAD"
        ),
        known_design_controls_complete=False,
        unrestricted_matched_cost_control_complete=False,
        promotion_eligible=False,
        novelty_claim=False,
        evidence_mode=EVIDENCE_MODE,
        resource_guard=resource,
        limitations=LIMITATIONS,
    )


def build_joint_orbit_artifact(
    result: object,
    *,
    budget: ExperimentBudget = ExperimentBudget(),
) -> ArtifactEnvelope:
    if type(result) is not JointOrbitSpectrumResult:
        raise JointOrbitValidationError("result must be exactly JointOrbitSpectrumResult")
    contract = make_run_contract(budget)
    return ArtifactEnvelope.create(
        stage_id="PRW-JO1-P11-K2",
        status="COMPLETE",
        run_contract=contract,
        resource_estimate=result.resource_guard,
        result=result.as_dict(),
        limitations=result.limitations,
    )


__all__ = [
    "ACTION_IDS",
    "COMMON_CHANNEL_ID",
    "CONSTRUCTION_HYPOTHESIS_ID",
    "FLAT_HYPOTHESIS_ID",
    "JointOrbitSpectrumResult",
    "JointOrbitValidationError",
    "LIMITATIONS",
    "MODULUS",
    "OrbitState",
    "PROTOCOL_ID",
    "ScheduleSummary",
    "SpectrumSummary",
    "STATES",
    "TYPE_ONE_REPRESENTATIVE",
    "TYPE_ZERO_REPRESENTATIVE",
    "analyze_joint_orbit_spectrum",
    "build_joint_orbit_artifact",
    "make_run_contract",
    "orbit_word",
    "preflight_joint_orbit_screen",
]
