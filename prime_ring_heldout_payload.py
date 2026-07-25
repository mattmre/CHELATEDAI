"""Deterministic held-out falsifier for conditional prime-ring routing.

This is a deliberately small systems screen, not a production benchmark.  For
each ``p`` in ``{7, 11}`` and each of three frozen seeds, it builds:

* 64 training examples: 32 true-unlock examples (one per type/waypoint) and
  32 unrelated false-unlock controls;
* 128 strictly ID-disjoint held-out examples: 96 true-unlock examples (three
  per type/waypoint) and 32 unrelated controls; and
* one common bank of 32 candidates (four types by eight waypoints).

The teacher's two type bits are XOR interactions over four raw metadata bits,
so neither bit is linearly separable from its raw pair.  A fitted conditional
router learns two four-entry Boolean tables.  Its exact paid transcript is
then supplied to a flat four-leaf decision-table router.  Randomized,
main-effects-only direct-metadata, and payload-only controls use the same
candidate bank and a deliberately equalized resource envelope.

The flat comparison answers only whether the conditional representation adds
held-out output capacity beyond its two paid branch bits.  It does not test
latency, trainability at scale, learned representations, corpus utility, or
distribution-shift generalization.  "Disjoint" below means example-identifier
disjointness only; train and held-out rows intentionally share the frozen
synthetic generator and class structure.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Dict, Optional, Sequence, Tuple


PROTOCOL_ID = "PRW-HELDOUT-NONSEPARABLE-P7-P11-001"
MODULI = (7, 11)
FROZEN_SEEDS = (67, 4691, 65537)
TYPE_COUNT = 4
WAYPOINT_COUNT = 8
CANDIDATE_COUNT = TYPE_COUNT * WAYPOINT_COUNT
PREDICATE_BITS = 2
LEAF_COUNT = 1 << PREDICATE_BITS
TRAIN_COUNT = 64
TRAIN_TRUE_UNLOCK_COUNT = 32
TRAIN_FALSE_UNLOCK_COUNT = 32
HELDOUT_COUNT = 128
HELDOUT_TRUE_UNLOCK_COUNT = 96
HELDOUT_FALSE_UNLOCK_COUNT = 32
ROUTER_NAMES = (
    "conditional",
    "paid_transcript_flat",
    "randomized",
    "direct_metadata",
    "payload_only",
)

HARD_MAX_ESTIMATED_BYTES = 64 * 1024 * 1024
HARD_MAX_WORK_UNITS = 5_000_000
HARD_MAX_SECONDS = 30.0
HARD_MAX_PREDICATE_BITS = 2
DEFAULT_MAX_ESTIMATED_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_WORK_UNITS = HARD_MAX_WORK_UNITS
DEFAULT_MAX_SECONDS = 10.0

EVIDENCE_MODE = "DETERMINISTIC_HELDOUT_SYNTHETIC_FALSIFICATION"
KILLED_STATUS = "NO_EXTRA_HELDOUT_CAPACITY_BEYOND_PAID_TRANSCRIPT"
SURVIVES_STATUS = "PAID_TRANSCRIPT_FLAT_MISMATCH_ON_HELDOUT"
LIMITATIONS = (
    "This is a tiny deterministic synthetic screen at p in {7,11}; it is not "
    "a statistical sample of models, corpora, tasks, or hardware.",
    "Here p controls only the synthetic payload-vector length; no prime-field, "
    "ring-transform, rotation, or number-theoretic advantage is tested.",
    "The two-bit XOR teacher is intentionally favorable to a four-leaf "
    "conditional router and intentionally hostile to an additive linear "
    "metadata control.",
    "The paid-transcript flat router is granted the conditional router's exact "
    "two branch bits; the test isolates output capacity, not transcript cost.",
    "All routers receive padded equal charges for candidates, branch bits, "
    "leaves, storage, operations, and false-unlock opportunities; those charges "
    "are modeled rather than measured.",
    "Payload prototypes are frozen synthetic codewords with small deterministic "
    "noise, and unrelated controls have deliberately low payload amplitude.",
    "Correction means recovery of a held-out true target missed by the "
    "payload-only control; it is not online training or model-weight repair.",
    "No result establishes novelty, production readiness, training savings, "
    "latency improvement, compression, or real-world recall.",
)


class HeldoutPayloadValidationError(ValueError):
    """Raised when an input violates the frozen held-out contract."""


class HeldoutPayloadResourceError(RuntimeError):
    """Raised before or during work when a bounded resource cap is crossed."""


def _plain_int(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise HeldoutPayloadValidationError(
            "{} must be an integer".format(name)
        )
    result = int(value)
    if result < minimum:
        raise HeldoutPayloadValidationError(
            "{} must be >= {}".format(name, minimum)
        )
    return result


def _positive_seconds(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise HeldoutPayloadValidationError(
            "{} must be a finite positive number".format(name)
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise HeldoutPayloadValidationError(
            "{} must be a finite positive number".format(name)
        )
    return result


def _frozen_modulus(value: object) -> int:
    modulus = _plain_int(value, "modulus", minimum=2)
    if modulus not in MODULI:
        raise HeldoutPayloadValidationError(
            "modulus must be one of {}".format(MODULI)
        )
    return modulus


def _frozen_seed(value: object) -> int:
    seed = _plain_int(value, "seed")
    if seed not in FROZEN_SEEDS:
        raise HeldoutPayloadValidationError(
            "seed must be one of the three frozen seeds"
        )
    return seed


@dataclass(frozen=True)
class HeldoutPayloadBudget:
    """Immutable bounded budget with hostile-subclass normalization."""

    max_estimated_bytes: int = DEFAULT_MAX_ESTIMATED_BYTES
    max_work_units: int = DEFAULT_MAX_WORK_UNITS
    max_seconds: float = DEFAULT_MAX_SECONDS
    max_predicate_bits: int = HARD_MAX_PREDICATE_BITS

    def __post_init__(self) -> None:
        integer_limits = (
            (
                "max_estimated_bytes",
                self.max_estimated_bytes,
                HARD_MAX_ESTIMATED_BYTES,
            ),
            ("max_work_units", self.max_work_units, HARD_MAX_WORK_UNITS),
            (
                "max_predicate_bits",
                self.max_predicate_bits,
                HARD_MAX_PREDICATE_BITS,
            ),
        )
        for name, value, hard_maximum in integer_limits:
            checked = _plain_int(value, name, minimum=1)
            if checked > hard_maximum:
                raise HeldoutPayloadValidationError(
                    "{} cannot exceed immutable hard cap {}".format(
                        name,
                        hard_maximum,
                    )
                )
            object.__setattr__(self, name, checked)
        seconds = _positive_seconds(self.max_seconds, "max_seconds")
        if seconds > HARD_MAX_SECONDS:
            raise HeldoutPayloadValidationError(
                "max_seconds cannot exceed immutable hard cap {:g}".format(
                    HARD_MAX_SECONDS
                )
            )
        object.__setattr__(self, "max_seconds", seconds)


@dataclass(frozen=True)
class SyntheticExample:
    """One immutable true- or false-unlock example."""

    example_id: str
    modulus: int
    metadata: Tuple[int, int, int, int]
    payload: Tuple[float, ...]
    target_candidate: Optional[int]
    should_unlock: bool


@dataclass(frozen=True)
class DatasetSplit:
    """Frozen deterministic training and held-out partitions."""

    modulus: int
    seed: int
    train: Tuple[SyntheticExample, ...]
    heldout: Tuple[SyntheticExample, ...]


@dataclass(frozen=True)
class LinearRule:
    """One additive threshold rule with no interaction term."""

    bias: int
    first_weight: int
    second_weight: int

    def predict(self, first: int, second: int) -> int:
        score = (
            self.bias
            + self.first_weight * first
            + self.second_weight * second
        )
        return int(score >= 0)


@dataclass(frozen=True)
class Candidate:
    """One type-waypoint candidate and its fitted payload prototype."""

    candidate_id: int
    type_index: int
    waypoint: int
    prototype: Tuple[float, ...]


@dataclass(frozen=True)
class FittedRouter:
    """Training-only fitted state shared by all matched routers."""

    modulus: int
    seed: int
    gate_tables: Tuple[Tuple[int, int, int, int], ...]
    linear_rules: Tuple[LinearRule, LinearRule]
    candidates: Tuple[Candidate, ...]
    flat_leaves: Tuple[Tuple[int, ...], ...]
    payload_only_leaves: Tuple[Tuple[int, ...], ...]
    random_leaf_permutation: Tuple[int, ...]
    unlock_threshold: float
    fit_example_ids: Tuple[str, ...]
    gate_training_errors: int
    linear_training_errors: int
    threshold_training_false_unlocks: int
    threshold_training_false_locks: int


@dataclass(frozen=True)
class Prediction:
    """One router prediction at the shared unlock threshold."""

    candidate_id: Optional[int]
    branch_transcript: int
    score: float
    unlocked: bool


@dataclass(frozen=True)
class RouterCharge:
    """One side of the padded matched-resource comparison."""

    name: str
    candidate_count: int
    candidate_scores_per_query: int
    branch_bits: int
    leaf_count: int
    charged_storage_bytes: int
    charged_operations_per_query: int
    false_unlock_opportunities: int
    unlocks_allowed_per_query: int
    measured_resources: bool
    padding_to_common_envelope: bool

    @property
    def matching_signature(self) -> Tuple[object, ...]:
        return (
            self.candidate_count,
            self.candidate_scores_per_query,
            self.branch_bits,
            self.leaf_count,
            self.charged_storage_bytes,
            self.charged_operations_per_query,
            self.false_unlock_opportunities,
            self.unlocks_allowed_per_query,
            self.measured_resources,
            self.padding_to_common_envelope,
        )


@dataclass(frozen=True)
class RouterMetrics:
    """Held-out retrieval, correction, and false-unlock metrics."""

    router_name: str
    true_unlock_opportunities: int
    correct_true_unlocks: int
    recall: float
    payload_only_misses: int
    corrections_of_payload_only: int
    correction_rate: float
    false_unlock_opportunities: int
    false_unlock_events: int
    false_unlock_rate: float
    false_locks: int


@dataclass(frozen=True)
class RunResult:
    """One modulus/seed held-out result."""

    modulus: int
    seed: int
    train_count: int
    heldout_count: int
    train_true_unlock_count: int
    train_false_unlock_count: int
    heldout_true_unlock_count: int
    heldout_false_unlock_count: int
    identifiers_disjoint: bool
    xor_teacher_non_linearly_separable: bool
    gate_training_errors: int
    linear_training_errors: int
    threshold_training_false_unlocks: int
    threshold_training_false_locks: int
    conditional_flat_exact_cases: int
    conditional_flat_exact_on_heldout: bool
    controls_matched: bool
    control_charges: Tuple[RouterCharge, ...]
    metrics: Tuple[RouterMetrics, ...]


@dataclass(frozen=True)
class ResourceEstimate:
    """Conservative modeled preflight for the entire six-run screen."""

    run_count: int
    train_examples_per_run: int
    heldout_examples_per_run: int
    total_train_examples: int
    total_heldout_examples: int
    candidate_count: int
    predicate_bits: int
    leaf_count: int
    estimated_peak_bytes: int
    estimated_work_units: int
    component_bytes: Tuple[Tuple[str, int], ...]
    component_work: Tuple[Tuple[str, int], ...]
    max_estimated_bytes: int
    max_work_units: int
    max_seconds: float
    max_predicate_bits: int
    one_run_materialized_at_a_time: bool
    measured_process_peak: bool


@dataclass(frozen=True)
class HeldoutPayloadAnalysis:
    """Complete non-promotional result from all frozen runs."""

    protocol_id: str
    evidence_mode: str
    moduli: Tuple[int, ...]
    seeds: Tuple[int, ...]
    runs: Tuple[RunResult, ...]
    aggregate_metrics: Tuple[RouterMetrics, ...]
    total_conditional_flat_exact_cases: int
    conditional_flat_exact_on_all_heldout: bool
    extra_capacity_kill_criterion_met: bool
    hypothesis_status: str
    all_split_identifiers_disjoint: bool
    distribution_shift_claim: bool
    class_separation_claim: bool
    all_controls_matched: bool
    resource_guard: ResourceEstimate
    promotion_eligible: bool
    novelty_claim: bool
    production_claim: bool
    training_savings_claim: bool
    runtime_benefit_claim: bool
    limitations: Tuple[str, ...]


def _teacher_type(metadata: Tuple[int, int, int, int]) -> int:
    first_left, first_right, second_left, second_right = metadata
    return (first_left ^ first_right) | (
        (second_left ^ second_right) << 1
    )


def xor_teacher_non_linearly_separable() -> bool:
    """Return the exact XOR half-space contradiction.

    For any additive score ``b + w0*x0 + w1*x1``, the sum of scores at XOR's
    two positive points equals the sum at its two negative points.  Strict
    separation would require the former sum to be larger, a contradiction.
    """

    positive_points = ((0, 1), (1, 0))
    negative_points = ((0, 0), (1, 1))
    positive_coefficients = (
        len(positive_points),
        sum(first for first, _second in positive_points),
        sum(second for _first, second in positive_points),
    )
    negative_coefficients = (
        len(negative_points),
        sum(first for first, _second in negative_points),
        sum(second for _first, second in negative_points),
    )
    return positive_coefficients == negative_coefficients == (2, 1, 1)


def _waypoint_code(modulus: int, waypoint: int) -> Tuple[float, ...]:
    bits = (
        waypoint & 1,
        (waypoint >> 1) & 1,
        (waypoint >> 2) & 1,
    )
    first, second, third = bits
    code_bits = (
        first,
        second,
        third,
        first ^ second,
        second ^ third,
        first ^ third,
        first ^ second ^ third,
        1 ^ first,
        1 ^ second,
        1 ^ third,
        1 ^ first ^ second ^ third,
    )
    return tuple(
        1.0 if code_bits[index] else -1.0
        for index in range(modulus)
    )


def _mixed_seed(
    modulus: int,
    seed: int,
    split_code: int,
    example_kind: int,
    ordinal: int,
) -> int:
    return (
        seed * 1_000_003
        + modulus * 97_409
        + split_code * 65_537
        + example_kind * 32_771
        + ordinal * 8_191
    ) & 0xFFFFFFFFFFFFFFFF


def _metadata_for_type(
    type_index: int,
    waypoint: int,
    repeat: int,
    phase: int,
) -> Tuple[int, int, int, int]:
    values = []
    for bit_index in range(PREDICATE_BITS):
        target_bit = (type_index >> bit_index) & 1
        options = (
            ((0, 0), (1, 1))
            if target_bit == 0
            else ((0, 1), (1, 0))
        )
        option = (waypoint + repeat + phase + bit_index) & 1
        values.extend(options[option])
    metadata = tuple(values)
    return (metadata[0], metadata[1], metadata[2], metadata[3])


def _positive_payload(
    modulus: int,
    waypoint: int,
    rng: random.Random,
) -> Tuple[float, ...]:
    code = _waypoint_code(modulus, waypoint)
    noise_values = (-0.10, -0.05, 0.0, 0.05, 0.10)
    return tuple(
        value + noise_values[rng.randrange(len(noise_values))]
        for value in code
    )


def _unrelated_payload(
    modulus: int,
    rng: random.Random,
) -> Tuple[float, ...]:
    return tuple(rng.uniform(-0.15, 0.15) for _index in range(modulus))


def _build_examples(
    modulus: int,
    seed: int,
    *,
    split_name: str,
) -> Tuple[SyntheticExample, ...]:
    split_code = 1 if split_name == "train" else 2
    positive_repeats = 1 if split_name == "train" else 3
    examples = []
    ordinal = 0
    phase = (seed + modulus + split_code) & 1
    for type_index in range(TYPE_COUNT):
        for waypoint in range(WAYPOINT_COUNT):
            for repeat in range(positive_repeats):
                rng = random.Random(
                    _mixed_seed(
                        modulus,
                        seed,
                        split_code,
                        1,
                        ordinal,
                    )
                )
                examples.append(
                    SyntheticExample(
                        example_id=(
                            "p{}-s{}-{}-true-{:03d}".format(
                                modulus,
                                seed,
                                split_name,
                                ordinal,
                            )
                        ),
                        modulus=modulus,
                        metadata=_metadata_for_type(
                            type_index,
                            waypoint,
                            repeat,
                            phase,
                        ),
                        payload=_positive_payload(
                            modulus,
                            waypoint,
                            rng,
                        ),
                        target_candidate=(
                            type_index * WAYPOINT_COUNT + waypoint
                        ),
                        should_unlock=True,
                    )
                )
                ordinal += 1

    for negative_index in range(32):
        rng = random.Random(
            _mixed_seed(
                modulus,
                seed,
                split_code,
                0,
                negative_index,
            )
        )
        metadata = tuple(rng.randrange(2) for _index in range(4))
        examples.append(
            SyntheticExample(
                example_id=(
                    "p{}-s{}-{}-false-{:03d}".format(
                        modulus,
                        seed,
                        split_name,
                        negative_index,
                    )
                ),
                modulus=modulus,
                metadata=(
                    metadata[0],
                    metadata[1],
                    metadata[2],
                    metadata[3],
                ),
                payload=_unrelated_payload(modulus, rng),
                target_candidate=None,
                should_unlock=False,
            )
        )

    shuffler = random.Random(
        _mixed_seed(modulus, seed, split_code, 3, 0)
    )
    shuffler.shuffle(examples)
    return tuple(examples)


def generate_dataset(modulus: int, seed: int) -> DatasetSplit:
    """Generate one frozen, strictly ID-disjoint deterministic split."""

    modulus_value = _frozen_modulus(modulus)
    seed_value = _frozen_seed(seed)
    train = _build_examples(
        modulus_value,
        seed_value,
        split_name="train",
    )
    heldout = _build_examples(
        modulus_value,
        seed_value,
        split_name="heldout",
    )
    train_ids = {example.example_id for example in train}
    heldout_ids = {example.example_id for example in heldout}
    if len(train) != TRAIN_COUNT or len(heldout) != HELDOUT_COUNT:
        raise AssertionError("frozen split size invariant failed")
    if train_ids & heldout_ids:
        raise AssertionError("training and held-out IDs overlap")
    return DatasetSplit(modulus_value, seed_value, train, heldout)


def _pair_index(first: int, second: int) -> int:
    return first | (second << 1)


def _fit_gate_tables(
    positives: Sequence[SyntheticExample],
) -> Tuple[Tuple[Tuple[int, int, int, int], ...], int]:
    tables = []
    total_errors = 0
    for bit_index in range(PREDICATE_BITS):
        counts = [[0, 0] for _pair in range(4)]
        for example in positives:
            offset = bit_index * 2
            pair = _pair_index(
                example.metadata[offset],
                example.metadata[offset + 1],
            )
            target_type = int(example.target_candidate) // WAYPOINT_COUNT
            label = (target_type >> bit_index) & 1
            counts[pair][label] += 1
        table = []
        for pair_counts in counts:
            if sum(pair_counts) == 0:
                raise HeldoutPayloadValidationError(
                    "training split does not cover every predicate input"
                )
            table.append(int(pair_counts[1] >= pair_counts[0]))
        typed_table = (table[0], table[1], table[2], table[3])
        tables.append(typed_table)
        for example in positives:
            offset = bit_index * 2
            pair = _pair_index(
                example.metadata[offset],
                example.metadata[offset + 1],
            )
            target_type = int(example.target_candidate) // WAYPOINT_COUNT
            label = (target_type >> bit_index) & 1
            total_errors += int(typed_table[pair] != label)
    return (tables[0], tables[1]), total_errors


def _fit_linear_rules(
    positives: Sequence[SyntheticExample],
    deadline: float,
) -> Tuple[Tuple[LinearRule, LinearRule], int]:
    rules = []
    total_errors = 0
    values = range(-2, 3)
    for bit_index in range(PREDICATE_BITS):
        best_rule = None
        best_errors = None
        for bias in values:
            for first_weight in values:
                for second_weight in values:
                    _check_deadline(deadline)
                    rule = LinearRule(bias, first_weight, second_weight)
                    errors = 0
                    for example in positives:
                        offset = bit_index * 2
                        target_type = (
                            int(example.target_candidate)
                            // WAYPOINT_COUNT
                        )
                        label = (target_type >> bit_index) & 1
                        errors += int(
                            rule.predict(
                                example.metadata[offset],
                                example.metadata[offset + 1],
                            )
                            != label
                        )
                    if best_errors is None or errors < best_errors:
                        best_rule = rule
                        best_errors = errors
        if best_rule is None or best_errors is None:
            raise AssertionError("linear rule search produced no candidate")
        rules.append(best_rule)
        total_errors += best_errors
    return (rules[0], rules[1]), total_errors


def _fit_candidates(
    positives: Sequence[SyntheticExample],
    modulus: int,
) -> Tuple[Candidate, ...]:
    waypoint_vectors = []
    for waypoint in range(WAYPOINT_COUNT):
        members = tuple(
            example.payload
            for example in positives
            if int(example.target_candidate) % WAYPOINT_COUNT == waypoint
        )
        if not members:
            raise HeldoutPayloadValidationError(
                "training split omits a waypoint"
            )
        prototype = tuple(
            sum(vector[index] for vector in members) / len(members)
            for index in range(modulus)
        )
        waypoint_vectors.append(prototype)
    candidates = []
    for type_index in range(TYPE_COUNT):
        for waypoint in range(WAYPOINT_COUNT):
            candidates.append(
                Candidate(
                    candidate_id=(
                        type_index * WAYPOINT_COUNT + waypoint
                    ),
                    type_index=type_index,
                    waypoint=waypoint,
                    prototype=waypoint_vectors[waypoint],
                )
            )
    return tuple(candidates)


def _payload_score(
    payload: Tuple[float, ...],
    candidate: Candidate,
    modulus: int,
) -> float:
    return sum(
        payload[index] * candidate.prototype[index]
        for index in range(modulus)
    ) / modulus


def _best_candidate(
    payload: Tuple[float, ...],
    candidates: Tuple[Candidate, ...],
    modulus: int,
    allowed_ids: Tuple[int, ...],
) -> Tuple[int, float]:
    allowed = frozenset(allowed_ids)
    best_id = None
    best_score = -math.inf
    for candidate in candidates:
        score = _payload_score(payload, candidate, modulus)
        if candidate.candidate_id not in allowed:
            continue
        if (
            score > best_score
            or (
                score == best_score
                and (
                    best_id is None
                    or candidate.candidate_id < best_id
                )
            )
        ):
            best_id = candidate.candidate_id
            best_score = score
    if best_id is None:
        raise AssertionError("candidate leaf must not be empty")
    return best_id, best_score


def _fit_unlock_threshold(
    train: Sequence[SyntheticExample],
    candidates: Tuple[Candidate, ...],
    modulus: int,
) -> Tuple[float, int, int]:
    all_ids = tuple(range(CANDIDATE_COUNT))
    positive_scores = []
    negative_scores = []
    for example in train:
        _candidate, score = _best_candidate(
            example.payload,
            candidates,
            modulus,
            all_ids,
        )
        if example.should_unlock:
            positive_scores.append(score)
        else:
            negative_scores.append(score)
    minimum_positive = min(positive_scores)
    maximum_negative = max(negative_scores)
    if not maximum_negative < minimum_positive:
        raise HeldoutPayloadValidationError(
            "training payload controls are not threshold-separable"
        )
    threshold = (minimum_positive + maximum_negative) / 2.0
    false_unlocks = sum(score >= threshold for score in negative_scores)
    false_locks = sum(score < threshold for score in positive_scores)
    return threshold, false_unlocks, false_locks


def fit_router(
    train: Sequence[SyntheticExample],
    modulus: int,
    seed: int,
    *,
    deadline: Optional[float] = None,
) -> FittedRouter:
    """Fit all router state using training examples only."""

    modulus_value = _frozen_modulus(modulus)
    seed_value = _frozen_seed(seed)
    if isinstance(train, (str, bytes)) or not isinstance(train, Sequence):
        raise HeldoutPayloadValidationError(
            "train must be a sequence of SyntheticExample values"
        )
    normalized = tuple(train[index] for index in range(len(train)))
    if len(normalized) != TRAIN_COUNT:
        raise HeldoutPayloadValidationError(
            "train must contain exactly {} examples".format(TRAIN_COUNT)
        )
    for index, example in enumerate(normalized):
        if not isinstance(example, SyntheticExample):
            raise HeldoutPayloadValidationError(
                "train[{}] must be a SyntheticExample".format(index)
            )
        if example.modulus != modulus_value:
            raise HeldoutPayloadValidationError(
                "training example modulus mismatch"
            )
    now = time.monotonic()
    if deadline is None:
        effective_deadline = now + HARD_MAX_SECONDS
    else:
        if isinstance(deadline, bool) or not isinstance(deadline, Real):
            raise HeldoutPayloadValidationError(
                "deadline must be a finite monotonic timestamp"
            )
        effective_deadline = float(deadline)
        if (
            not math.isfinite(effective_deadline)
            or effective_deadline <= now
            or effective_deadline - now > HARD_MAX_SECONDS
        ):
            raise HeldoutPayloadValidationError(
                "deadline must be within the immutable 30-second window"
            )
    _check_deadline(effective_deadline)
    positives = tuple(
        example for example in normalized if example.should_unlock
    )
    negatives = tuple(
        example for example in normalized if not example.should_unlock
    )
    if (
        len(positives) != TRAIN_TRUE_UNLOCK_COUNT
        or len(negatives) != TRAIN_FALSE_UNLOCK_COUNT
    ):
        raise HeldoutPayloadValidationError(
            "training split must contain the frozen true/false counts"
        )
    gate_tables, gate_errors = _fit_gate_tables(positives)
    linear_rules, linear_errors = _fit_linear_rules(
        positives,
        effective_deadline,
    )
    candidates = _fit_candidates(positives, modulus_value)
    threshold, threshold_false_unlocks, threshold_false_locks = (
        _fit_unlock_threshold(normalized, candidates, modulus_value)
    )
    flat_leaves = tuple(
        tuple(
            type_index * WAYPOINT_COUNT + waypoint
            for waypoint in range(WAYPOINT_COUNT)
        )
        for type_index in range(TYPE_COUNT)
    )
    payload_leaf = tuple(range(CANDIDATE_COUNT))
    payload_only_leaves = tuple(
        payload_leaf for _leaf in range(LEAF_COUNT)
    )
    random_permutation = list(range(LEAF_COUNT))
    random.Random(
        _mixed_seed(modulus_value, seed_value, 4, 4, 0)
    ).shuffle(random_permutation)
    _check_deadline(effective_deadline)
    return FittedRouter(
        modulus=modulus_value,
        seed=seed_value,
        gate_tables=gate_tables,
        linear_rules=linear_rules,
        candidates=candidates,
        flat_leaves=flat_leaves,
        payload_only_leaves=payload_only_leaves,
        random_leaf_permutation=tuple(random_permutation),
        unlock_threshold=threshold,
        fit_example_ids=tuple(
            example.example_id for example in normalized
        ),
        gate_training_errors=gate_errors,
        linear_training_errors=linear_errors,
        threshold_training_false_unlocks=threshold_false_unlocks,
        threshold_training_false_locks=threshold_false_locks,
    )


def _conditional_transcript(
    example: SyntheticExample,
    model: FittedRouter,
) -> int:
    transcript = 0
    for bit_index, table in enumerate(model.gate_tables):
        offset = bit_index * 2
        pair = _pair_index(
            example.metadata[offset],
            example.metadata[offset + 1],
        )
        transcript |= table[pair] << bit_index
    return transcript


def _direct_metadata_transcript(
    example: SyntheticExample,
    model: FittedRouter,
) -> int:
    transcript = 0
    for bit_index, rule in enumerate(model.linear_rules):
        offset = bit_index * 2
        transcript |= rule.predict(
            example.metadata[offset],
            example.metadata[offset + 1],
        ) << bit_index
    return transcript


def _payload_only_transcript(example: SyntheticExample) -> int:
    return int(example.payload[0] >= 0.0) | (
        int(example.payload[1] >= 0.0) << 1
    )


def _predict_from_leaf(
    example: SyntheticExample,
    model: FittedRouter,
    transcript: int,
    leaves: Tuple[Tuple[int, ...], ...],
) -> Prediction:
    candidate_id, score = _best_candidate(
        example.payload,
        model.candidates,
        model.modulus,
        leaves[transcript],
    )
    unlocked = score >= model.unlock_threshold
    return Prediction(
        candidate_id=candidate_id if unlocked else None,
        branch_transcript=transcript,
        score=score,
        unlocked=unlocked,
    )


def predict_conditional(
    example: SyntheticExample,
    model: FittedRouter,
) -> Prediction:
    """Predict with the fitted two-predicate conditional router."""

    transcript = _conditional_transcript(example, model)
    return _predict_from_leaf(
        example,
        model,
        transcript,
        model.flat_leaves,
    )


def predict_paid_transcript_flat(
    example: SyntheticExample,
    model: FittedRouter,
    transcript: int,
) -> Prediction:
    """Predict from a transcript supplied by the matched conditional router."""

    transcript_value = _plain_int(transcript, "transcript")
    if transcript_value >= LEAF_COUNT:
        raise HeldoutPayloadValidationError(
            "transcript must fit in two predicate bits"
        )
    return _predict_from_leaf(
        example,
        model,
        transcript_value,
        model.flat_leaves,
    )


def predict_randomized(
    example: SyntheticExample,
    model: FittedRouter,
    conditional_transcript: int,
) -> Prediction:
    """Apply the frozen randomized leaf permutation to the paid transcript."""

    transcript = model.random_leaf_permutation[conditional_transcript]
    return _predict_from_leaf(
        example,
        model,
        transcript,
        model.flat_leaves,
    )


def predict_direct_metadata(
    example: SyntheticExample,
    model: FittedRouter,
) -> Prediction:
    """Predict with fitted additive main effects and no XOR interaction."""

    transcript = _direct_metadata_transcript(example, model)
    return _predict_from_leaf(
        example,
        model,
        transcript,
        model.flat_leaves,
    )


def predict_payload_only(
    example: SyntheticExample,
    model: FittedRouter,
) -> Prediction:
    """Search the shared candidate bank without metadata-based restriction."""

    transcript = _payload_only_transcript(example)
    return _predict_from_leaf(
        example,
        model,
        transcript,
        model.payload_only_leaves,
    )


def _control_charge(
    name: str,
    false_unlock_opportunities: int,
    modulus: int,
) -> RouterCharge:
    common_storage = (
        CANDIDATE_COUNT * modulus * 16
        + LEAF_COUNT * CANDIDATE_COUNT * 8
        + PREDICATE_BITS * 4 * 8
        + 4_096
    )
    common_operations = (
        CANDIDATE_COUNT * modulus * 2
        + CANDIDATE_COUNT
        + PREDICATE_BITS * 8
        + LEAF_COUNT
    )
    return RouterCharge(
        name=name,
        candidate_count=CANDIDATE_COUNT,
        candidate_scores_per_query=CANDIDATE_COUNT,
        branch_bits=PREDICATE_BITS,
        leaf_count=LEAF_COUNT,
        charged_storage_bytes=common_storage,
        charged_operations_per_query=common_operations,
        false_unlock_opportunities=false_unlock_opportunities,
        unlocks_allowed_per_query=1,
        measured_resources=False,
        padding_to_common_envelope=True,
    )


def _metric_from_predictions(
    name: str,
    heldout: Sequence[SyntheticExample],
    predictions: Sequence[Prediction],
    payload_predictions: Sequence[Prediction],
) -> RouterMetrics:
    true_examples = tuple(
        index
        for index, example in enumerate(heldout)
        if example.should_unlock
    )
    false_examples = tuple(
        index
        for index, example in enumerate(heldout)
        if not example.should_unlock
    )
    correct = sum(
        predictions[index].candidate_id
        == heldout[index].target_candidate
        for index in true_examples
    )
    payload_misses = sum(
        payload_predictions[index].candidate_id
        != heldout[index].target_candidate
        for index in true_examples
    )
    corrections = sum(
        payload_predictions[index].candidate_id
        != heldout[index].target_candidate
        and predictions[index].candidate_id
        == heldout[index].target_candidate
        for index in true_examples
    )
    false_unlocks = sum(
        predictions[index].unlocked for index in false_examples
    )
    false_locks = sum(
        not predictions[index].unlocked for index in true_examples
    )
    return RouterMetrics(
        router_name=name,
        true_unlock_opportunities=len(true_examples),
        correct_true_unlocks=correct,
        recall=correct / len(true_examples),
        payload_only_misses=payload_misses,
        corrections_of_payload_only=corrections,
        correction_rate=(
            corrections / payload_misses if payload_misses else 0.0
        ),
        false_unlock_opportunities=len(false_examples),
        false_unlock_events=false_unlocks,
        false_unlock_rate=false_unlocks / len(false_examples),
        false_locks=false_locks,
    )


def _run_one(
    modulus: int,
    seed: int,
    deadline: float,
) -> RunResult:
    split = generate_dataset(modulus, seed)
    _check_deadline(deadline)
    model = fit_router(
        split.train,
        modulus,
        seed,
        deadline=deadline,
    )
    train_ids = set(model.fit_example_ids)
    heldout_ids = {example.example_id for example in split.heldout}
    identifiers_disjoint = not bool(train_ids & heldout_ids)

    prediction_lists: Dict[str, list] = {
        name: [] for name in ROUTER_NAMES
    }
    exact_cases = 0
    exact = True
    for example in split.heldout:
        _check_deadline(deadline)
        conditional = predict_conditional(example, model)
        flat = predict_paid_transcript_flat(
            example,
            model,
            conditional.branch_transcript,
        )
        randomized = predict_randomized(
            example,
            model,
            conditional.branch_transcript,
        )
        direct = predict_direct_metadata(example, model)
        payload = predict_payload_only(example, model)
        prediction_lists["conditional"].append(conditional)
        prediction_lists["paid_transcript_flat"].append(flat)
        prediction_lists["randomized"].append(randomized)
        prediction_lists["direct_metadata"].append(direct)
        prediction_lists["payload_only"].append(payload)
        exact_cases += 1
        if (
            conditional.candidate_id != flat.candidate_id
            or conditional.branch_transcript != flat.branch_transcript
            or conditional.score != flat.score
            or conditional.unlocked != flat.unlocked
        ):
            exact = False

    payload_predictions = tuple(prediction_lists["payload_only"])
    metrics = tuple(
        _metric_from_predictions(
            name,
            split.heldout,
            tuple(prediction_lists[name]),
            payload_predictions,
        )
        for name in ROUTER_NAMES
    )
    charges = tuple(
        _control_charge(
            name,
            HELDOUT_FALSE_UNLOCK_COUNT,
            modulus,
        )
        for name in ROUTER_NAMES
    )
    controls_matched = len(
        {charge.matching_signature for charge in charges}
    ) == 1
    return RunResult(
        modulus=modulus,
        seed=seed,
        train_count=len(split.train),
        heldout_count=len(split.heldout),
        train_true_unlock_count=sum(
            example.should_unlock for example in split.train
        ),
        train_false_unlock_count=sum(
            not example.should_unlock for example in split.train
        ),
        heldout_true_unlock_count=sum(
            example.should_unlock for example in split.heldout
        ),
        heldout_false_unlock_count=sum(
            not example.should_unlock for example in split.heldout
        ),
        identifiers_disjoint=identifiers_disjoint,
        xor_teacher_non_linearly_separable=(
            xor_teacher_non_linearly_separable()
        ),
        gate_training_errors=model.gate_training_errors,
        linear_training_errors=model.linear_training_errors,
        threshold_training_false_unlocks=(
            model.threshold_training_false_unlocks
        ),
        threshold_training_false_locks=(
            model.threshold_training_false_locks
        ),
        conditional_flat_exact_cases=exact_cases,
        conditional_flat_exact_on_heldout=exact,
        controls_matched=controls_matched,
        control_charges=charges,
        metrics=metrics,
    )


def _aggregate_metrics(
    runs: Sequence[RunResult],
) -> Tuple[RouterMetrics, ...]:
    aggregates = []
    for name in ROUTER_NAMES:
        selected = tuple(
            next(
                metric
                for metric in run.metrics
                if metric.router_name == name
            )
            for run in runs
        )
        true_opportunities = sum(
            metric.true_unlock_opportunities for metric in selected
        )
        correct = sum(
            metric.correct_true_unlocks for metric in selected
        )
        payload_misses = sum(
            metric.payload_only_misses for metric in selected
        )
        corrections = sum(
            metric.corrections_of_payload_only for metric in selected
        )
        false_opportunities = sum(
            metric.false_unlock_opportunities for metric in selected
        )
        false_unlocks = sum(
            metric.false_unlock_events for metric in selected
        )
        false_locks = sum(metric.false_locks for metric in selected)
        aggregates.append(
            RouterMetrics(
                router_name=name,
                true_unlock_opportunities=true_opportunities,
                correct_true_unlocks=correct,
                recall=correct / true_opportunities,
                payload_only_misses=payload_misses,
                corrections_of_payload_only=corrections,
                correction_rate=(
                    corrections / payload_misses
                    if payload_misses
                    else 0.0
                ),
                false_unlock_opportunities=false_opportunities,
                false_unlock_events=false_unlocks,
                false_unlock_rate=(
                    false_unlocks / false_opportunities
                ),
                false_locks=false_locks,
            )
        )
    return tuple(aggregates)


def _resource_estimate(
    budget: HeldoutPayloadBudget,
) -> ResourceEstimate:
    if PREDICATE_BITS > budget.max_predicate_bits:
        raise HeldoutPayloadResourceError(
            "predicate-bit count exceeds budget: {} > {}".format(
                PREDICATE_BITS,
                budget.max_predicate_bits,
            )
        )
    run_count = len(MODULI) * len(FROZEN_SEEDS)
    maximum_modulus = max(MODULI)
    component_bytes = (
        (
            "one_run_example_vectors",
            (TRAIN_COUNT + HELDOUT_COUNT) * maximum_modulus * 16,
        ),
        (
            "one_run_example_records",
            (TRAIN_COUNT + HELDOUT_COUNT) * 256,
        ),
        (
            "five_matched_router_envelopes",
            len(ROUTER_NAMES)
            * CANDIDATE_COUNT
            * maximum_modulus
            * 16,
        ),
        ("truth_tables_and_linear_search", 65_536),
        ("streaming_evaluation_scratch", 1_048_576),
        ("all_run_summaries", 262_144),
    )
    linear_rule_count = 5**3
    component_work = (
        (
            "dataset_generation",
            run_count
            * (TRAIN_COUNT + HELDOUT_COUNT)
            * maximum_modulus
            * 4,
        ),
        (
            "conditional_table_fit",
            run_count * TRAIN_COUNT * PREDICATE_BITS * 8,
        ),
        (
            "main_effects_linear_fit",
            run_count
            * PREDICATE_BITS
            * linear_rule_count
            * TRAIN_TRUE_UNLOCK_COUNT
            * 4,
        ),
        (
            "payload_prototype_and_threshold_fit",
            run_count
            * (
                TRAIN_COUNT * maximum_modulus * 4
                + TRAIN_COUNT
                * CANDIDATE_COUNT
                * maximum_modulus
                * 2
            ),
        ),
        (
            "five_router_heldout_candidate_scoring",
            run_count
            * HELDOUT_COUNT
            * len(ROUTER_NAMES)
            * CANDIDATE_COUNT
            * maximum_modulus
            * 2,
        ),
        (
            "metric_and_match_accounting",
            run_count
            * HELDOUT_COUNT
            * len(ROUTER_NAMES)
            * 8,
        ),
    )
    estimated_bytes = sum(value for _name, value in component_bytes)
    estimated_work = sum(value for _name, value in component_work)
    if estimated_bytes > budget.max_estimated_bytes:
        raise HeldoutPayloadResourceError(
            "estimated bytes exceed budget: {} > {}".format(
                estimated_bytes,
                budget.max_estimated_bytes,
            )
        )
    if estimated_work > budget.max_work_units:
        raise HeldoutPayloadResourceError(
            "estimated work exceeds budget: {} > {}".format(
                estimated_work,
                budget.max_work_units,
            )
        )
    return ResourceEstimate(
        run_count=run_count,
        train_examples_per_run=TRAIN_COUNT,
        heldout_examples_per_run=HELDOUT_COUNT,
        total_train_examples=run_count * TRAIN_COUNT,
        total_heldout_examples=run_count * HELDOUT_COUNT,
        candidate_count=CANDIDATE_COUNT,
        predicate_bits=PREDICATE_BITS,
        leaf_count=LEAF_COUNT,
        estimated_peak_bytes=estimated_bytes,
        estimated_work_units=estimated_work,
        component_bytes=component_bytes,
        component_work=component_work,
        max_estimated_bytes=budget.max_estimated_bytes,
        max_work_units=budget.max_work_units,
        max_seconds=budget.max_seconds,
        max_predicate_bits=budget.max_predicate_bits,
        one_run_materialized_at_a_time=True,
        measured_process_peak=False,
    )


def _check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise HeldoutPayloadResourceError(
            "held-out payload screen exceeded its wall-clock deadline"
        )


def analyze_heldout_payload(
    *,
    budget: HeldoutPayloadBudget = HeldoutPayloadBudget(),
) -> HeldoutPayloadAnalysis:
    """Run every preregistered modulus/seed held-out comparison."""

    if not isinstance(budget, HeldoutPayloadBudget):
        raise HeldoutPayloadValidationError(
            "budget must be a HeldoutPayloadBudget"
        )
    started = time.monotonic()
    deadline = started + budget.max_seconds
    resource = _resource_estimate(budget)
    _check_deadline(deadline)
    runs = []
    for modulus in MODULI:
        for seed in FROZEN_SEEDS:
            _check_deadline(deadline)
            runs.append(_run_one(modulus, seed, deadline))
    typed_runs = tuple(runs)
    aggregates = _aggregate_metrics(typed_runs)
    exact_cases = sum(
        run.conditional_flat_exact_cases for run in typed_runs
    )
    exact = all(
        run.conditional_flat_exact_on_heldout for run in typed_runs
    )
    all_identifiers_disjoint = all(
        run.identifiers_disjoint for run in typed_runs
    )
    all_matched = all(run.controls_matched for run in typed_runs)
    kill = (
        exact
        and exact_cases == resource.total_heldout_examples
        and all_identifiers_disjoint
        and all_matched
    )
    _check_deadline(deadline)
    return HeldoutPayloadAnalysis(
        protocol_id=PROTOCOL_ID,
        evidence_mode=EVIDENCE_MODE,
        moduli=MODULI,
        seeds=FROZEN_SEEDS,
        runs=typed_runs,
        aggregate_metrics=aggregates,
        total_conditional_flat_exact_cases=exact_cases,
        conditional_flat_exact_on_all_heldout=exact,
        extra_capacity_kill_criterion_met=kill,
        hypothesis_status=KILLED_STATUS if kill else SURVIVES_STATUS,
        all_split_identifiers_disjoint=all_identifiers_disjoint,
        distribution_shift_claim=False,
        class_separation_claim=False,
        all_controls_matched=all_matched,
        resource_guard=resource,
        promotion_eligible=False,
        novelty_claim=False,
        production_claim=False,
        training_savings_claim=False,
        runtime_benefit_claim=False,
        limitations=LIMITATIONS,
    )


def metric_by_name(
    metrics: Sequence[RouterMetrics],
    name: str,
) -> RouterMetrics:
    """Return one uniquely named metric record."""

    matches = tuple(
        metric for metric in metrics if metric.router_name == name
    )
    if len(matches) != 1:
        raise HeldoutPayloadValidationError(
            "router metric name must have exactly one match"
        )
    return matches[0]


__all__ = [
    "CANDIDATE_COUNT",
    "Candidate",
    "DEFAULT_MAX_ESTIMATED_BYTES",
    "DEFAULT_MAX_SECONDS",
    "DEFAULT_MAX_WORK_UNITS",
    "DatasetSplit",
    "EVIDENCE_MODE",
    "FROZEN_SEEDS",
    "FittedRouter",
    "HARD_MAX_ESTIMATED_BYTES",
    "HARD_MAX_PREDICATE_BITS",
    "HARD_MAX_SECONDS",
    "HARD_MAX_WORK_UNITS",
    "HELDOUT_COUNT",
    "HELDOUT_FALSE_UNLOCK_COUNT",
    "HELDOUT_TRUE_UNLOCK_COUNT",
    "HeldoutPayloadAnalysis",
    "HeldoutPayloadBudget",
    "HeldoutPayloadResourceError",
    "HeldoutPayloadValidationError",
    "KILLED_STATUS",
    "LEAF_COUNT",
    "LIMITATIONS",
    "LinearRule",
    "MODULI",
    "PREDICATE_BITS",
    "PROTOCOL_ID",
    "Prediction",
    "ROUTER_NAMES",
    "ResourceEstimate",
    "RouterCharge",
    "RouterMetrics",
    "RunResult",
    "SURVIVES_STATUS",
    "SyntheticExample",
    "TRAIN_COUNT",
    "TRAIN_FALSE_UNLOCK_COUNT",
    "TRAIN_TRUE_UNLOCK_COUNT",
    "TYPE_COUNT",
    "WAYPOINT_COUNT",
    "analyze_heldout_payload",
    "fit_router",
    "generate_dataset",
    "metric_by_name",
    "predict_conditional",
    "predict_direct_metadata",
    "predict_paid_transcript_flat",
    "predict_payload_only",
    "predict_randomized",
    "xor_teacher_non_linearly_separable",
]
