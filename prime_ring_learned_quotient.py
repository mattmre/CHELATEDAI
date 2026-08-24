"""Resource-bounded exact-affine-solver screen for global gauge invariance.

The existing exact quotient audit deliberately constructs an observer from
relative coordinates.  This module asks a narrower, independent question:
does an exact affine solver operating on raw ``Z_7^4`` coordinates recover a
planted invariant affine task without being given quotient coordinates?

For each frozen seed, quotient classes are split before examples are built.
Training classes contribute one raw representative to the unconstrained
learner and three representatives to the gauge-augmented learner.  Four other
representatives from the same classes form an unseen-gauge evaluation split.
Entirely withheld quotient classes form a class-disjoint evaluation split.

The hypothesis class is deliberately small: exact modular-affine maps.  The
first five frozen raw rows already form a full-rank design and identify the
planted rule.  This makes the result deterministic, dependency-free, and
exactly auditable, but also makes it non-confirmatory.  Success establishes
solver recovery of the planted affine gauge-null direction; it does not
establish empirical or neural emergence, arbitrary-observer invariance,
retrieval utility, compression, novelty, or any claim about physical
dimensionality.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from itertools import product
from numbers import Integral, Real
from typing import Callable, Dict, List, Optional, Sequence, Tuple


PROTOCOL_ID = "PRW-LEARNED-QUOTIENT-P7-L4-001"
EVIDENCE_MODE = "DETERMINISTIC_SYNTHETIC_EXACT_AFFINE_SOLVER_CONTROL"
MODULUS = 7
CASCADE_LENGTH = 4
QUOTIENT_DIMENSION = CASCADE_LENGTH - 1
MINIMAL_FULL_RANK_ROWS = CASCADE_LENGTH + 1
QUOTIENT_CLASS_COUNT = MODULUS**QUOTIENT_DIMENSION
TRAIN_CLASS_COUNT = 274
HELDOUT_CLASS_COUNT = QUOTIENT_CLASS_COUNT - TRAIN_CLASS_COUNT
FROZEN_SEEDS = (7, 42, 1337)

# For each training quotient class, these seven offsets partition the orbit.
# The augmented learner sees the first set; every evaluated gauge is therefore
# absent from both raw training sets.
AUGMENTATION_OFFSETS = (0, 1, 3)
UNSEEN_GAUGE_OFFSETS = (2, 4, 5, 6)

# The planted raw affine rule has coefficient sum zero modulo seven:
# 6*x0 + x1 + 2*x2 + 5*x3 + 3.  In quotient coordinates this is
# d1 + 2*d2 + 5*d3 + 3.
INVARIANT_RAW_COEFFICIENTS = (6, 1, 2, 5)
INVARIANT_QUOTIENT_COEFFICIENTS = (1, 2, 5)
INVARIANT_BIAS = 3
ANCHORED_RAW_COEFFICIENTS = (1, 0, 0, 0)
ANCHORED_BIAS = 0

MODEL_UNCONSTRAINED_RAW = "unconstrained_raw_affine"
MODEL_AUGMENTED_RAW = "gauge_augmented_raw_affine"
MODEL_QUOTIENT_ORACLE = "explicit_quotient_oracle_affine"
MODEL_ANCHORED_CONTROL = "anchored_gauge_sensitive_raw_control"
MODEL_MINIMAL_RANK_WITNESS = "five_row_exact_raw_affine_solver_witness"
SPLIT_UNSEEN_GAUGE = "unseen_gauge"
SPLIT_UNSEEN_CLASS = "unseen_quotient_class"

HARD_MAX_SEED_COUNT = len(FROZEN_SEEDS)
HARD_MAX_ESTIMATED_BYTES = 16 * 1024 * 1024
HARD_MAX_WORK_UNITS = 5_000_000
HARD_MAX_SECONDS = 30.0
DEFAULT_MAX_ESTIMATED_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_WORK_UNITS = 1_000_000
DEFAULT_MAX_SECONDS = 10.0

SUPPORTED_STATUS = "NARROW_EXACT_AFFINE_SOLVER_RECOVERY_NON_CONFIRMATORY"
UNSUPPORTED_STATUS = "EXACT_AFFINE_SOLVER_RECOVERY_OR_CONTROL_GATE_FAILED"
LIMITATIONS = (
    "The task is a planted noiseless modular-affine rule over Z_7^4.",
    "The exact solver hypothesis class contains the planted teacher.",
    "Five full-rank raw rows identify the affine rule in every frozen split; "
    "the larger training set does not provide evidence of empirical emergence.",
    "Gauge augmentation is generated from the known group action; it is an "
    "engineering control, not evidence of spontaneous symmetry discovery.",
    "The explicit quotient learner is an oracle-feature control and cannot establish independently learned invariance.",
    "The class-disjoint split tests affine generalization only; it is not a "
    "new corpus, semantic subdomain, or real distribution shift.",
    "Byte and work figures are conservative modeled bounds, not measured "
    "process RSS, allocator peaks, hardware timings, or energy use.",
    "No result establishes empirical or neural emergence, arbitrary-observer "
    "invariance, intrinsic dimension, retrieval quality, training savings at "
    "scale, novelty, or production utility.",
)


class LearnedQuotientValidationError(ValueError):
    """Raised when an input violates the frozen experiment contract."""


class LearnedQuotientResourceError(RuntimeError):
    """Raised before or during work when a resource ceiling is crossed."""


class LearnedQuotientFitError(RuntimeError):
    """Raised when the exact affine learner cannot fit its declared task."""


def _plain_int(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise LearnedQuotientValidationError("{} must be an integer".format(name))
    result = int(value)
    if result < minimum:
        raise LearnedQuotientValidationError("{} must be >= {}".format(name, minimum))
    return result


def _positive_seconds(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise LearnedQuotientValidationError("{} must be a finite positive number".format(name))
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise LearnedQuotientValidationError("{} must be a finite positive number".format(name))
    return result


def _validated_seeds(values: object) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise LearnedQuotientValidationError("seeds must be a sequence")
    # Shape-first refusal prevents a hostile sequence from being normalized.
    if len(values) == 0:
        raise LearnedQuotientValidationError("seeds must not be empty")
    if len(values) > HARD_MAX_SEED_COUNT:
        raise LearnedQuotientValidationError("seed count exceeds immutable hard cap {}".format(HARD_MAX_SEED_COUNT))
    normalized = tuple(_plain_int(values[index], "seeds[{}]".format(index)) for index in range(len(values)))
    if len(set(normalized)) != len(normalized):
        raise LearnedQuotientValidationError("seeds must be unique")
    if any(seed not in FROZEN_SEEDS for seed in normalized):
        raise LearnedQuotientValidationError("seeds must be selected from {}".format(FROZEN_SEEDS))
    return normalized


@dataclass(frozen=True)
class LearnedQuotientBudget:
    """Immutable resource budget bounded by hard experiment ceilings."""

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
            (
                "max_work_units",
                self.max_work_units,
                HARD_MAX_WORK_UNITS,
            ),
        )
        for name, value, hard_maximum in integer_limits:
            checked = _plain_int(value, name, minimum=1)
            if checked > hard_maximum:
                raise LearnedQuotientValidationError(
                    "{} cannot exceed immutable hard cap {}".format(
                        name,
                        hard_maximum,
                    )
                )
            object.__setattr__(self, name, checked)
        seconds = _positive_seconds(self.max_seconds, "max_seconds")
        if seconds > HARD_MAX_SECONDS:
            raise LearnedQuotientValidationError(
                "max_seconds cannot exceed immutable hard cap {:g}".format(HARD_MAX_SECONDS)
            )
        object.__setattr__(self, "max_seconds", seconds)


@dataclass(frozen=True)
class ResourceEstimate:
    """Shape-only preflight for the complete requested seed set."""

    seed_count: int
    quotient_class_count: int
    train_class_count: int
    heldout_class_count: int
    unique_states_per_seed: int
    raw_training_rows_per_seed: int
    augmented_training_rows_per_seed: int
    unseen_gauge_rows_per_seed: int
    unseen_class_rows_per_seed: int
    total_logical_rows: int
    estimated_peak_bytes: int
    estimated_work_units: int
    max_estimated_bytes: int
    max_work_units: int
    max_seconds: float
    one_seed_materialized_at_a_time: bool
    measured_process_rss: bool


@dataclass(frozen=True)
class LearnedExample:
    """One immutable train or evaluation example."""

    example_id: str
    quotient_class: Tuple[int, int, int]
    state: Tuple[int, int, int, int]
    invariant_target: int
    anchored_target: int


@dataclass(frozen=True)
class AffineRule:
    """One exact modular-affine rule fitted from examples."""

    feature_space: str
    coefficients: Tuple[int, ...]
    bias: int
    training_rows: int
    design_rank: int

    @property
    def gauge_coefficient_sum(self) -> Optional[int]:
        if self.feature_space != "raw":
            return None
        return int(sum(self.coefficients) % MODULUS)

    def predict_features(self, features: Sequence[int]) -> int:
        if len(features) != len(self.coefficients):
            raise LearnedQuotientValidationError("feature length does not match fitted rule")
        total = self.bias
        for coefficient, value in zip(self.coefficients, features):
            total += coefficient * int(value)
        return int(total % MODULUS)


@dataclass(frozen=True)
class ModelSplitMetrics:
    """Accuracy and orbit consistency for one model on one split."""

    model_name: str
    split: str
    expected_gauge_invariant: bool
    example_count: int
    correct_count: int
    task_accuracy: float
    orbit_count: int
    exactly_consistent_orbits: int
    orbit_consistency_rate: float
    orbit_pair_count: int
    disagreeing_orbit_pairs: int
    orbit_disagreement_rate: float


@dataclass(frozen=True)
class SeedRunResult:
    """Complete deterministic result for one frozen seed."""

    seed: int
    train_class_count: int
    heldout_class_count: int
    raw_training_rows: int
    augmented_training_rows: int
    unseen_gauge_rows: int
    unseen_class_rows: int
    train_evaluation_ids_disjoint: bool
    train_evaluation_states_disjoint: bool
    train_heldout_classes_disjoint: bool
    unseen_gauge_classes_match_train_classes: bool
    unconstrained_raw_rule: AffineRule
    augmented_raw_rule: AffineRule
    quotient_oracle_rule: AffineRule
    anchored_control_rule: AffineRule
    pre_witness_design_rank: int
    minimal_rank_witness_rule: AffineRule
    metrics: Tuple[ModelSplitMetrics, ...]

    def metric(self, model_name: str, split: str) -> ModelSplitMetrics:
        for value in self.metrics:
            if value.model_name == model_name and value.split == split:
                return value
        raise KeyError((model_name, split))


@dataclass(frozen=True)
class LearnedQuotientAnalysis:
    """Non-promotional aggregate over every requested frozen seed."""

    protocol_id: str
    evidence_mode: str
    seeds: Tuple[int, ...]
    runs: Tuple[SeedRunResult, ...]
    resource_guard: ResourceEstimate
    unconstrained_raw_invariant_all_seeds: bool
    augmented_raw_invariant_all_seeds: bool
    quotient_oracle_invariant_all_seeds: bool
    anchored_control_sensitive_all_seeds: bool
    minimal_rank_solver_witness_all_seeds: bool
    all_identifier_and_class_separation_checks_pass: bool
    all_publication_conditions_pass: bool
    hypothesis_status: str
    non_confirmatory: bool
    promotion_eligible: bool
    novelty_claim: bool
    empirical_emergence_claim: bool
    arbitrary_observer_claim: bool
    intrinsic_dimension_claim: bool
    real_world_utility_claim: bool
    limitations: Tuple[str, ...]


def estimate_learned_quotient_resources(
    seeds: object = FROZEN_SEEDS,
    budget: LearnedQuotientBudget = LearnedQuotientBudget(),
) -> ResourceEstimate:
    """Return and enforce a shape-only resource preflight."""

    normalized_seeds = _validated_seeds(seeds)
    if not isinstance(budget, LearnedQuotientBudget):
        raise LearnedQuotientValidationError("budget must be a LearnedQuotientBudget")

    raw_training = TRAIN_CLASS_COUNT
    augmented_training = TRAIN_CLASS_COUNT * len(AUGMENTATION_OFFSETS)
    unseen_gauge = TRAIN_CLASS_COUNT * len(UNSEEN_GAUGE_OFFSETS)
    unseen_class = HELDOUT_CLASS_COUNT * MODULUS
    unique_states = augmented_training + unseen_gauge + unseen_class
    if unique_states != MODULUS**CASCADE_LENGTH:
        raise LearnedQuotientFitError("frozen train/evaluation partition does not cover Z_7^4 exactly")

    # Each materialized example is conservatively charged for the dataclass,
    # strings, two tuples, and integer objects.  Only one seed is live at once.
    example_bytes = unique_states * 320
    split_index_bytes = unique_states * 96
    elimination_bytes = augmented_training * (CASCADE_LENGTH + 2) * 32
    retained_result_bytes = len(normalized_seeds) * 96 * 1024
    estimated_bytes = example_bytes + split_index_bytes + elimination_bytes + retained_result_bytes

    raw_fit_work = raw_training * (CASCADE_LENGTH + 1) ** 2
    augmented_fit_work = augmented_training * (CASCADE_LENGTH + 1) ** 2
    quotient_fit_work = raw_training * (QUOTIENT_DIMENSION + 1) ** 2
    anchored_fit_work = raw_fit_work
    minimal_witness_fit_work = (MINIMAL_FULL_RANK_ROWS + MINIMAL_FULL_RANK_ROWS - 1) * (CASCADE_LENGTH + 1) ** 2
    evaluation_rows = unseen_gauge + unseen_class
    prediction_work = evaluation_rows * (4 * (CASCADE_LENGTH + 1) + (QUOTIENT_DIMENSION + 1) + (CASCADE_LENGTH + 1))
    orbit_pair_work = (
        TRAIN_CLASS_COUNT * (len(UNSEEN_GAUGE_OFFSETS) * (len(UNSEEN_GAUGE_OFFSETS) - 1) // 2)
        + HELDOUT_CLASS_COUNT * (MODULUS * (MODULUS - 1) // 2)
    ) * 5
    per_seed_work = (
        unique_states
        + raw_fit_work
        + augmented_fit_work
        + quotient_fit_work
        + anchored_fit_work
        + minimal_witness_fit_work
        + prediction_work
        + orbit_pair_work
    )
    estimated_work = per_seed_work * len(normalized_seeds)

    if estimated_bytes > budget.max_estimated_bytes:
        raise LearnedQuotientResourceError(
            "estimated peak bytes exceed budget: {} > {}".format(
                estimated_bytes,
                budget.max_estimated_bytes,
            )
        )
    if estimated_work > budget.max_work_units:
        raise LearnedQuotientResourceError(
            "estimated work units exceed budget: {} > {}".format(
                estimated_work,
                budget.max_work_units,
            )
        )

    return ResourceEstimate(
        seed_count=len(normalized_seeds),
        quotient_class_count=QUOTIENT_CLASS_COUNT,
        train_class_count=TRAIN_CLASS_COUNT,
        heldout_class_count=HELDOUT_CLASS_COUNT,
        unique_states_per_seed=unique_states,
        raw_training_rows_per_seed=raw_training,
        augmented_training_rows_per_seed=augmented_training,
        unseen_gauge_rows_per_seed=unseen_gauge,
        unseen_class_rows_per_seed=unseen_class,
        total_logical_rows=unique_states * len(normalized_seeds),
        estimated_peak_bytes=estimated_bytes,
        estimated_work_units=estimated_work,
        max_estimated_bytes=budget.max_estimated_bytes,
        max_work_units=budget.max_work_units,
        max_seconds=budget.max_seconds,
        one_seed_materialized_at_a_time=True,
        measured_process_rss=False,
    )


def quotient_key(state: Sequence[int]) -> Tuple[int, int, int]:
    """Return canonical relative coordinates for one raw state."""

    if isinstance(state, (str, bytes)) or not isinstance(state, Sequence):
        raise LearnedQuotientValidationError("state must be a sequence")
    if len(state) != CASCADE_LENGTH:
        raise LearnedQuotientValidationError("state must contain exactly four coordinates")
    values = tuple(_plain_int(state[index], "state[{}]".format(index)) for index in range(CASCADE_LENGTH))
    if any(value >= MODULUS for value in values):
        raise LearnedQuotientValidationError("state coordinates must be canonical residues")
    anchor = values[0]
    return (
        (values[1] - anchor) % MODULUS,
        (values[2] - anchor) % MODULUS,
        (values[3] - anchor) % MODULUS,
    )


def apply_global_gauge(
    state: Sequence[int],
    shift: object,
) -> Tuple[int, int, int, int]:
    """Apply one canonical global gauge shift."""

    key = quotient_key(state)
    normalized_shift = _plain_int(shift, "shift")
    if normalized_shift >= MODULUS:
        raise LearnedQuotientValidationError("shift must be a canonical residue")
    anchor = (int(state[0]) + normalized_shift) % MODULUS
    return (
        anchor,
        (anchor + key[0]) % MODULUS,
        (anchor + key[1]) % MODULUS,
        (anchor + key[2]) % MODULUS,
    )


def invariant_teacher(state: Sequence[int]) -> int:
    """Return the planted invariant modular-affine target."""

    quotient_key(state)
    values = tuple(int(state[index]) for index in range(CASCADE_LENGTH))
    return int(
        (
            sum(
                coefficient * value
                for coefficient, value in zip(
                    INVARIANT_RAW_COEFFICIENTS,
                    values,
                )
            )
            + INVARIANT_BIAS
        )
        % MODULUS
    )


def anchored_teacher(state: Sequence[int]) -> int:
    """Return the intentionally gauge-sensitive anchored target."""

    quotient_key(state)
    return int(state[0]) % MODULUS


def _state_from_key(
    key: Tuple[int, int, int],
    gauge: int,
) -> Tuple[int, int, int, int]:
    return (
        gauge,
        (gauge + key[0]) % MODULUS,
        (gauge + key[1]) % MODULUS,
        (gauge + key[2]) % MODULUS,
    )


def _example(
    seed: int,
    split: str,
    key: Tuple[int, int, int],
    gauge: int,
) -> LearnedExample:
    state = _state_from_key(key, gauge)
    return LearnedExample(
        example_id=(
            "{}:{}:{}-{}-{}:g{}".format(
                seed,
                split,
                key[0],
                key[1],
                key[2],
                gauge,
            )
        ),
        quotient_class=key,
        state=state,
        invariant_target=invariant_teacher(state),
        anchored_target=anchored_teacher(state),
    )


@dataclass(frozen=True)
class _SeedDataset:
    train_classes: Tuple[Tuple[int, int, int], ...]
    heldout_classes: Tuple[Tuple[int, int, int], ...]
    raw_training: Tuple[LearnedExample, ...]
    augmented_training: Tuple[LearnedExample, ...]
    unseen_gauge: Tuple[LearnedExample, ...]
    unseen_class: Tuple[LearnedExample, ...]


def _check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise LearnedQuotientResourceError("learned quotient experiment exceeded its wall-clock deadline")


def _build_seed_dataset(seed: int, deadline: float) -> _SeedDataset:
    rng = random.Random(seed)
    keys = list(product(range(MODULUS), repeat=QUOTIENT_DIMENSION))
    rng.shuffle(keys)
    train_keys = tuple(keys[:TRAIN_CLASS_COUNT])
    heldout_keys = tuple(keys[TRAIN_CLASS_COUNT:])

    raw_training: List[LearnedExample] = []
    augmented_training: List[LearnedExample] = []
    unseen_gauge: List[LearnedExample] = []
    unseen_class: List[LearnedExample] = []

    for index, key in enumerate(train_keys):
        if index % 64 == 0:
            _check_deadline(deadline)
        base_gauge = rng.randrange(MODULUS)
        for offset in AUGMENTATION_OFFSETS:
            gauge = (base_gauge + offset) % MODULUS
            value = _example(seed, "train", key, gauge)
            augmented_training.append(value)
            if offset == 0:
                raw_training.append(value)
        for offset in UNSEEN_GAUGE_OFFSETS:
            gauge = (base_gauge + offset) % MODULUS
            unseen_gauge.append(_example(seed, SPLIT_UNSEEN_GAUGE, key, gauge))

    for index, key in enumerate(heldout_keys):
        if index % 32 == 0:
            _check_deadline(deadline)
        for gauge in range(MODULUS):
            unseen_class.append(_example(seed, SPLIT_UNSEEN_CLASS, key, gauge))

    return _SeedDataset(
        train_classes=train_keys,
        heldout_classes=heldout_keys,
        raw_training=tuple(raw_training),
        augmented_training=tuple(augmented_training),
        unseen_gauge=tuple(unseen_gauge),
        unseen_class=tuple(unseen_class),
    )


FeatureFunction = Callable[[LearnedExample], Tuple[int, ...]]
TargetFunction = Callable[[LearnedExample], int]


def _raw_features(example: LearnedExample) -> Tuple[int, ...]:
    return example.state


def _quotient_features(example: LearnedExample) -> Tuple[int, ...]:
    return example.quotient_class


def _invariant_target(example: LearnedExample) -> int:
    return example.invariant_target


def _anchored_target(example: LearnedExample) -> int:
    return example.anchored_target


def _fit_affine_rule(
    examples: Sequence[LearnedExample],
    feature_space: str,
    feature_function: FeatureFunction,
    target_function: TargetFunction,
    feature_dimension: int,
    deadline: float,
) -> AffineRule:
    """Fit one exact affine rule by deterministic RREF over ``F_7``."""

    if len(examples) == 0:
        raise LearnedQuotientFitError("cannot fit an empty training set")
    variable_count = feature_dimension + 1
    matrix: List[List[int]] = []
    for index, example in enumerate(examples):
        if index % 128 == 0:
            _check_deadline(deadline)
        features = feature_function(example)
        if len(features) != feature_dimension:
            raise LearnedQuotientFitError("training feature dimension does not match declaration")
        matrix.append([int(value) % MODULUS for value in features] + [1, target_function(example) % MODULUS])

    pivot_columns: List[int] = []
    pivot_row = 0
    for column in range(variable_count):
        _check_deadline(deadline)
        selected = None
        for row_index in range(pivot_row, len(matrix)):
            if matrix[row_index][column] % MODULUS != 0:
                selected = row_index
                break
        if selected is None:
            continue
        matrix[pivot_row], matrix[selected] = matrix[selected], matrix[pivot_row]
        inverse = pow(matrix[pivot_row][column], MODULUS - 2, MODULUS)
        matrix[pivot_row] = [(value * inverse) % MODULUS for value in matrix[pivot_row]]
        for row_index, row in enumerate(matrix):
            if row_index == pivot_row:
                continue
            factor = row[column] % MODULUS
            if factor == 0:
                continue
            matrix[row_index] = [(left - factor * right) % MODULUS for left, right in zip(row, matrix[pivot_row])]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(matrix):
            break

    for row in matrix:
        if all(value % MODULUS == 0 for value in row[:variable_count]):
            if row[-1] % MODULUS != 0:
                raise LearnedQuotientFitError("declared affine task is inconsistent")

    solution = [0] * variable_count
    for row_index, column in enumerate(pivot_columns):
        solution[column] = matrix[row_index][-1] % MODULUS
    rule = AffineRule(
        feature_space=feature_space,
        coefficients=tuple(solution[:-1]),
        bias=solution[-1],
        training_rows=len(examples),
        design_rank=len(pivot_columns),
    )
    for index, example in enumerate(examples):
        if index % 128 == 0:
            _check_deadline(deadline)
        if rule.predict_features(feature_function(example)) != target_function(example):
            raise LearnedQuotientFitError("fitted affine rule does not reproduce its training rows")
    return rule


def _evaluate_rule(
    model_name: str,
    split: str,
    rule: AffineRule,
    examples: Sequence[LearnedExample],
    feature_function: FeatureFunction,
    target_function: TargetFunction,
    expected_gauge_invariant: bool,
    deadline: float,
) -> ModelSplitMetrics:
    predictions_by_class: Dict[Tuple[int, int, int], List[int]] = {}
    correct = 0
    for index, example in enumerate(examples):
        if index % 256 == 0:
            _check_deadline(deadline)
        predicted = rule.predict_features(feature_function(example))
        if predicted == target_function(example):
            correct += 1
        predictions_by_class.setdefault(example.quotient_class, []).append(predicted)

    consistent = 0
    pair_count = 0
    disagreements = 0
    for predictions in predictions_by_class.values():
        if len(set(predictions)) == 1:
            consistent += 1
        for left in range(len(predictions)):
            for right in range(left + 1, len(predictions)):
                pair_count += 1
                if predictions[left] != predictions[right]:
                    disagreements += 1

    example_count = len(examples)
    orbit_count = len(predictions_by_class)
    return ModelSplitMetrics(
        model_name=model_name,
        split=split,
        expected_gauge_invariant=expected_gauge_invariant,
        example_count=example_count,
        correct_count=correct,
        task_accuracy=(correct / example_count if example_count else 0.0),
        orbit_count=orbit_count,
        exactly_consistent_orbits=consistent,
        orbit_consistency_rate=(consistent / orbit_count if orbit_count else 0.0),
        orbit_pair_count=pair_count,
        disagreeing_orbit_pairs=disagreements,
        orbit_disagreement_rate=(disagreements / pair_count if pair_count else 0.0),
    )


def _run_seed(seed: int, deadline: float) -> SeedRunResult:
    dataset = _build_seed_dataset(seed, deadline)
    unconstrained_rule = _fit_affine_rule(
        dataset.raw_training,
        "raw",
        _raw_features,
        _invariant_target,
        CASCADE_LENGTH,
        deadline,
    )
    augmented_rule = _fit_affine_rule(
        dataset.augmented_training,
        "raw",
        _raw_features,
        _invariant_target,
        CASCADE_LENGTH,
        deadline,
    )
    quotient_rule = _fit_affine_rule(
        dataset.raw_training,
        "quotient",
        _quotient_features,
        _invariant_target,
        QUOTIENT_DIMENSION,
        deadline,
    )
    anchored_rule = _fit_affine_rule(
        dataset.raw_training,
        "raw",
        _raw_features,
        _anchored_target,
        CASCADE_LENGTH,
        deadline,
    )
    pre_witness_rule = _fit_affine_rule(
        dataset.raw_training[: MINIMAL_FULL_RANK_ROWS - 1],
        "raw",
        _raw_features,
        _invariant_target,
        CASCADE_LENGTH,
        deadline,
    )
    minimal_witness_rule = _fit_affine_rule(
        dataset.raw_training[:MINIMAL_FULL_RANK_ROWS],
        "raw",
        _raw_features,
        _invariant_target,
        CASCADE_LENGTH,
        deadline,
    )

    model_specs = (
        (
            MODEL_UNCONSTRAINED_RAW,
            unconstrained_rule,
            _raw_features,
            _invariant_target,
            True,
        ),
        (
            MODEL_AUGMENTED_RAW,
            augmented_rule,
            _raw_features,
            _invariant_target,
            True,
        ),
        (
            MODEL_QUOTIENT_ORACLE,
            quotient_rule,
            _quotient_features,
            _invariant_target,
            True,
        ),
        (
            MODEL_ANCHORED_CONTROL,
            anchored_rule,
            _raw_features,
            _anchored_target,
            False,
        ),
        (
            MODEL_MINIMAL_RANK_WITNESS,
            minimal_witness_rule,
            _raw_features,
            _invariant_target,
            True,
        ),
    )
    metrics: List[ModelSplitMetrics] = []
    for model_name, rule, features, target, expected_invariant in model_specs:
        metrics.append(
            _evaluate_rule(
                model_name,
                SPLIT_UNSEEN_GAUGE,
                rule,
                dataset.unseen_gauge,
                features,
                target,
                expected_invariant,
                deadline,
            )
        )
        metrics.append(
            _evaluate_rule(
                model_name,
                SPLIT_UNSEEN_CLASS,
                rule,
                dataset.unseen_class,
                features,
                target,
                expected_invariant,
                deadline,
            )
        )

    train_ids = {example.example_id for example in dataset.augmented_training}
    evaluation_ids = {example.example_id for example in dataset.unseen_gauge + dataset.unseen_class}
    train_states = {example.state for example in dataset.augmented_training}
    evaluation_states = {example.state for example in dataset.unseen_gauge + dataset.unseen_class}
    train_classes = set(dataset.train_classes)
    heldout_classes = set(dataset.heldout_classes)
    unseen_gauge_classes = {example.quotient_class for example in dataset.unseen_gauge}

    return SeedRunResult(
        seed=seed,
        train_class_count=len(dataset.train_classes),
        heldout_class_count=len(dataset.heldout_classes),
        raw_training_rows=len(dataset.raw_training),
        augmented_training_rows=len(dataset.augmented_training),
        unseen_gauge_rows=len(dataset.unseen_gauge),
        unseen_class_rows=len(dataset.unseen_class),
        train_evaluation_ids_disjoint=train_ids.isdisjoint(evaluation_ids),
        train_evaluation_states_disjoint=train_states.isdisjoint(evaluation_states),
        train_heldout_classes_disjoint=train_classes.isdisjoint(heldout_classes),
        unseen_gauge_classes_match_train_classes=(unseen_gauge_classes == train_classes),
        unconstrained_raw_rule=unconstrained_rule,
        augmented_raw_rule=augmented_rule,
        quotient_oracle_rule=quotient_rule,
        anchored_control_rule=anchored_rule,
        pre_witness_design_rank=pre_witness_rule.design_rank,
        minimal_rank_witness_rule=minimal_witness_rule,
        metrics=tuple(metrics),
    )


def _invariant_model_passes(run: SeedRunResult, model_name: str) -> bool:
    for split in (SPLIT_UNSEEN_GAUGE, SPLIT_UNSEEN_CLASS):
        metric = run.metric(model_name, split)
        if metric.task_accuracy != 1.0:
            return False
        if metric.orbit_disagreement_rate != 0.0:
            return False
        if metric.orbit_consistency_rate != 1.0:
            return False
    return True


def _anchored_control_passes(run: SeedRunResult) -> bool:
    for split in (SPLIT_UNSEEN_GAUGE, SPLIT_UNSEEN_CLASS):
        metric = run.metric(MODEL_ANCHORED_CONTROL, split)
        if metric.task_accuracy != 1.0:
            return False
        if metric.orbit_disagreement_rate != 1.0:
            return False
        if metric.orbit_consistency_rate != 0.0:
            return False
    return True


def _rule_matches_frozen_solution(
    rule: AffineRule,
    feature_space: str,
    coefficients: Tuple[int, ...],
    bias: int,
    training_rows: int,
    design_rank: int,
) -> bool:
    return (
        rule.feature_space == feature_space
        and rule.coefficients == coefficients
        and rule.bias == bias
        and rule.training_rows == training_rows
        and rule.design_rank == design_rank
    )


def _minimal_rank_witness_passes(run: SeedRunResult) -> bool:
    return (
        run.pre_witness_design_rank < MINIMAL_FULL_RANK_ROWS
        and _rule_matches_frozen_solution(
            run.minimal_rank_witness_rule,
            "raw",
            INVARIANT_RAW_COEFFICIENTS,
            INVARIANT_BIAS,
            MINIMAL_FULL_RANK_ROWS,
            MINIMAL_FULL_RANK_ROWS,
        )
        and _invariant_model_passes(
            run,
            MODEL_MINIMAL_RANK_WITNESS,
        )
    )


def analyze_learned_quotient(
    seeds: object = FROZEN_SEEDS,
    budget: LearnedQuotientBudget = LearnedQuotientBudget(),
) -> LearnedQuotientAnalysis:
    """Run the complete bounded learned-observer experiment."""

    normalized_seeds = _validated_seeds(seeds)
    resource = estimate_learned_quotient_resources(
        normalized_seeds,
        budget,
    )
    deadline = time.monotonic() + budget.max_seconds
    _check_deadline(deadline)

    runs: List[SeedRunResult] = []
    for seed in normalized_seeds:
        _check_deadline(deadline)
        runs.append(_run_seed(seed, deadline))

    unconstrained = all(
        _rule_matches_frozen_solution(
            run.unconstrained_raw_rule,
            "raw",
            INVARIANT_RAW_COEFFICIENTS,
            INVARIANT_BIAS,
            TRAIN_CLASS_COUNT,
            MINIMAL_FULL_RANK_ROWS,
        )
        and run.unconstrained_raw_rule.gauge_coefficient_sum == 0
        and _invariant_model_passes(run, MODEL_UNCONSTRAINED_RAW)
        for run in runs
    )
    augmented = all(
        _rule_matches_frozen_solution(
            run.augmented_raw_rule,
            "raw",
            INVARIANT_RAW_COEFFICIENTS,
            INVARIANT_BIAS,
            TRAIN_CLASS_COUNT * len(AUGMENTATION_OFFSETS),
            MINIMAL_FULL_RANK_ROWS,
        )
        and run.augmented_raw_rule.gauge_coefficient_sum == 0
        and _invariant_model_passes(run, MODEL_AUGMENTED_RAW)
        for run in runs
    )
    quotient = all(
        _rule_matches_frozen_solution(
            run.quotient_oracle_rule,
            "quotient",
            INVARIANT_QUOTIENT_COEFFICIENTS,
            INVARIANT_BIAS,
            TRAIN_CLASS_COUNT,
            QUOTIENT_DIMENSION + 1,
        )
        and _invariant_model_passes(run, MODEL_QUOTIENT_ORACLE)
        for run in runs
    )
    anchored = all(
        _rule_matches_frozen_solution(
            run.anchored_control_rule,
            "raw",
            ANCHORED_RAW_COEFFICIENTS,
            ANCHORED_BIAS,
            TRAIN_CLASS_COUNT,
            MINIMAL_FULL_RANK_ROWS,
        )
        and run.anchored_control_rule.gauge_coefficient_sum != 0
        and _anchored_control_passes(run)
        for run in runs
    )
    minimal_witness = all(_minimal_rank_witness_passes(run) for run in runs)
    separated = all(
        run.train_evaluation_ids_disjoint
        and run.train_evaluation_states_disjoint
        and run.train_heldout_classes_disjoint
        and run.unseen_gauge_classes_match_train_classes
        for run in runs
    )
    publication_conditions = unconstrained and augmented and quotient and anchored and minimal_witness and separated
    status = SUPPORTED_STATUS if publication_conditions else UNSUPPORTED_STATUS

    return LearnedQuotientAnalysis(
        protocol_id=PROTOCOL_ID,
        evidence_mode=EVIDENCE_MODE,
        seeds=normalized_seeds,
        runs=tuple(runs),
        resource_guard=resource,
        unconstrained_raw_invariant_all_seeds=unconstrained,
        augmented_raw_invariant_all_seeds=augmented,
        quotient_oracle_invariant_all_seeds=quotient,
        anchored_control_sensitive_all_seeds=anchored,
        minimal_rank_solver_witness_all_seeds=minimal_witness,
        all_identifier_and_class_separation_checks_pass=separated,
        all_publication_conditions_pass=publication_conditions,
        hypothesis_status=status,
        non_confirmatory=True,
        promotion_eligible=False,
        novelty_claim=False,
        empirical_emergence_claim=False,
        arbitrary_observer_claim=False,
        intrinsic_dimension_claim=False,
        real_world_utility_claim=False,
        limitations=LIMITATIONS,
    )


__all__ = [
    "ANCHORED_BIAS",
    "ANCHORED_RAW_COEFFICIENTS",
    "AUGMENTATION_OFFSETS",
    "AffineRule",
    "CASCADE_LENGTH",
    "DEFAULT_MAX_ESTIMATED_BYTES",
    "DEFAULT_MAX_SECONDS",
    "DEFAULT_MAX_WORK_UNITS",
    "EVIDENCE_MODE",
    "FROZEN_SEEDS",
    "HARD_MAX_ESTIMATED_BYTES",
    "HARD_MAX_SECONDS",
    "HARD_MAX_SEED_COUNT",
    "HARD_MAX_WORK_UNITS",
    "HELDOUT_CLASS_COUNT",
    "INVARIANT_BIAS",
    "INVARIANT_QUOTIENT_COEFFICIENTS",
    "INVARIANT_RAW_COEFFICIENTS",
    "LIMITATIONS",
    "LearnedExample",
    "LearnedQuotientAnalysis",
    "LearnedQuotientBudget",
    "LearnedQuotientFitError",
    "LearnedQuotientResourceError",
    "LearnedQuotientValidationError",
    "MODEL_ANCHORED_CONTROL",
    "MODEL_AUGMENTED_RAW",
    "MODEL_MINIMAL_RANK_WITNESS",
    "MODEL_QUOTIENT_ORACLE",
    "MODEL_UNCONSTRAINED_RAW",
    "MINIMAL_FULL_RANK_ROWS",
    "MODULUS",
    "ModelSplitMetrics",
    "PROTOCOL_ID",
    "QUOTIENT_CLASS_COUNT",
    "QUOTIENT_DIMENSION",
    "ResourceEstimate",
    "SPLIT_UNSEEN_CLASS",
    "SPLIT_UNSEEN_GAUGE",
    "SUPPORTED_STATUS",
    "SeedRunResult",
    "TRAIN_CLASS_COUNT",
    "UNSEEN_GAUGE_OFFSETS",
    "UNSUPPORTED_STATUS",
    "analyze_learned_quotient",
    "anchored_teacher",
    "apply_global_gauge",
    "estimate_learned_quotient_resources",
    "invariant_teacher",
    "quotient_key",
]
