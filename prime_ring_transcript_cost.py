"""Bounded synthetic screen for structural-prior training-search effects.

The earlier runtime-gate result showed that a flat decision table has the same
output capacity as a conditional program *after* it is given the same branch
transcript.  This module asks the narrower follow-up question: can a declared
factorization make those bits cheaper to learn from raw inputs?

The screen is deliberately finite and non-confirmatory.  It compares:

* a factorized learner given the declared causal feature groups;
* a matched flat feature-search learner that must discover those groups;
* a causal-only flat-search counterfactual that must discover the groups but
  cannot select the deliberately spurious payload shortcut;
* an equal-complexity factorized control given deterministically wrong groups;
* a paid-transcript control that receives the answer as side information;
* a payload-only learner; and
* a deterministic random control.

Factorized and feature-search learners have the same padded storage and
inference envelope.  Their actual parameter use and training-search work are
retained separately because learning/search cost is the object under test.  The
causal-only counterfactual recovers the factorized rule at full coverage; this
shows that the unrestricted flat learner's shifted failure is a
shortcut-selection/structural-prior effect, not a representational or generic
inference advantage.  Frozen ``spurious_reversal`` cells change
payload/transcript alignment from one in the training environment to zero in
the shifted environment.  ``no_shift`` cells are true equal-environment nulls;
``nuisance_remapping`` cells retain the earlier split-hashed nuisance control
under an explicit, non-null name.

Nothing here establishes representational superiority, real retrieval utility,
training savings for neural models, novelty, or production readiness.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Dict, List, Optional, Sequence, Tuple


PROTOCOL_ID = "PRW-TRANSCRIPT-COST-SYNTHETIC-001"
SCHEMA_VERSION = "1.1"
EVIDENCE_MODE = "NON_CONFIRMATORY_DETERMINISTIC_SYNTHETIC_SCREEN"
DECISION_RULE_ID = "PRW-TRANSCRIPT-COST-CANDIDATE-V2"

LEARNER_NAMES = (
    "factorized",
    "matched_flat",
    "causal_only_flat",
    "wrong_group_factorized",
    "paid_transcript",
    "payload_only",
    "deterministic_random",
)
TEACHER_FAMILIES = ("separable", "xor", "parity")
SHIFT_MODES = ("no_shift", "nuisance_remapping", "spurious_reversal")

MIN_BRANCH_BITS = 2
MAX_BRANCH_BITS = 3
MAX_GROUP_WIDTH = 3
MAX_TEACHER_CELLS = 8
MAX_TRAIN_SIZES = 8
MAX_RAW_FEATURES = 12

HARD_MAX_ESTIMATED_BYTES = 64 * 1024 * 1024
HARD_MAX_WORK_UNITS = 50_000_000
HARD_MAX_SECONDS = 30.0
DEFAULT_MAX_ESTIMATED_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_WORK_UNITS = 20_000_000
DEFAULT_MAX_SECONDS = 10.0

FROZEN_TRAIN_SIZES = (4, 8, 16, 32, 64, 128, 256, 512)
TARGET_SHIFTED_ROUTE_ACCURACY = 0.95
CANDIDATE_MIN_ROUTE_ACCURACY_DELTA = 0.10
NULL_MAX_ROUTE_ACCURACY_DELTA = 0.01

CANDIDATE_STATUS = "CANDIDATE_STRUCTURAL_PRIOR_TRAINING_SEARCH_SIGNAL_NON_CONFIRMATORY"
NO_SIGNAL_STATUS = "NO_CANDIDATE_SIGNAL_IN_FROZEN_SCREEN"
NULL_PASS_STATUS = "SEPARABLE_NULL_SANITY_PASS"
NULL_FAIL_STATUS = "SEPARABLE_NULL_SANITY_FAIL"
GLOBAL_HYPOTHESIS_STATUS = "UNRESOLVED_NON_CONFIRMATORY_SYNTHETIC"

LIMITATIONS = (
    "The factorized learner is given the correct causal grouping; the flat "
    "learner must select raw features from observational training rows.",
    "The causal-only flat counterfactual recovers the factorized shifted "
    "accuracy at full coverage.  The unrestricted flat accuracy gap is "
    "therefore a deliberately induced shortcut-selection effect, not a generic "
    "inference or representational advantage.",
    "The equal-complexity wrong-group control underperforms only because its "
    "structural grouping is deliberately incorrect; it does not show that the "
    "declared grouping can be discovered without prior knowledge.",
    "Spurious-reversal cells are intentionally adversarial identification "
    "tests, not samples from a natural corpus or deployment distribution.",
    "No-shift cells repeat exactly the same raw-feature/target environment under "
    "disjoint identifiers.  Nuisance-remapping cells instead alter only the "
    "split-hashed nuisance payload and are not described as no-shift controls.",
    "The learners are finite Boolean lookup rules, not neural networks, "
    "transformers, mixture-of-experts routers, or production retrieval code.",
    "Storage and operation figures are deterministic modeled charges, not "
    "measured RSS, latency, energy, allocator peaks, or hardware utilization.",
    "Common-envelope padding matches storage and inference opportunity; actual "
    "training-search work remains different because that is the tested cost.",
    "The paid-transcript control receives branch_bits of side information per "
    "example and therefore tests capacity after payment, not inference cost.",
    "The candidate is only a structural-prior training-search signal.  It can "
    "justify a larger preregistered test but does not establish utility, "
    "novelty, an asymptotic separation, or promotion.",
)


class TranscriptCostValidationError(ValueError):
    """Raised when a transcript-cost contract is malformed."""


class TranscriptCostResourceError(RuntimeError):
    """Raised before or during work that exceeds a frozen resource budget."""


def _plain_int(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TranscriptCostValidationError("{} must be an integer".format(name))
    result = int(value)
    if result < minimum:
        raise TranscriptCostValidationError("{} must be >= {}".format(name, minimum))
    return result


def _finite_probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TranscriptCostValidationError("{} must be a finite probability".format(name))
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise TranscriptCostValidationError("{} must be in [0, 1]".format(name))
    return result


def _positive_seconds(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TranscriptCostValidationError("{} must be a finite positive number".format(name))
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise TranscriptCostValidationError("{} must be a finite positive number".format(name))
    return result


def _stable_int(*parts: object) -> int:
    encoded = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _bits_from_int(value: int, width: int) -> Tuple[int, ...]:
    return tuple((value >> index) & 1 for index in range(width))


def _route_from_bits(bits: Sequence[int]) -> int:
    route = 0
    for index, bit in enumerate(bits):
        route |= int(bit) << index
    return route


def _check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise TranscriptCostResourceError("transcript-cost analysis exceeded its wall-clock deadline")


@dataclass(frozen=True)
class TranscriptTeacher:
    """One frozen Boolean transcript teacher and environment split."""

    family: str
    branch_bits: int
    shift_mode: str
    seed: int

    def __post_init__(self) -> None:
        if self.family not in TEACHER_FAMILIES:
            raise TranscriptCostValidationError("family must be one of {}".format(TEACHER_FAMILIES))
        branch_bits = _plain_int(self.branch_bits, "branch_bits", MIN_BRANCH_BITS)
        if branch_bits > MAX_BRANCH_BITS:
            raise TranscriptCostValidationError("branch_bits must be <= {}".format(MAX_BRANCH_BITS))
        if self.shift_mode not in SHIFT_MODES:
            raise TranscriptCostValidationError("shift_mode must be one of {}".format(SHIFT_MODES))
        seed = _plain_int(self.seed, "seed", 0)
        object.__setattr__(self, "branch_bits", branch_bits)
        object.__setattr__(self, "seed", seed)

    @property
    def group_width(self) -> int:
        if self.family == "parity":
            return 3
        return 2

    @property
    def nonlinear(self) -> bool:
        return self.family in ("xor", "parity")

    @property
    def cell_id(self) -> str:
        return "b{}-{}-{}-s{}".format(
            self.branch_bits,
            self.family,
            self.shift_mode,
            self.seed,
        )


DEFAULT_TEACHERS = (
    TranscriptTeacher("separable", 2, "no_shift", 67),
    TranscriptTeacher("separable", 2, "nuisance_remapping", 67),
    TranscriptTeacher("xor", 2, "spurious_reversal", 4691),
    TranscriptTeacher("separable", 3, "no_shift", 65537),
    TranscriptTeacher("separable", 3, "nuisance_remapping", 65537),
    TranscriptTeacher("parity", 3, "spurious_reversal", 104729),
)


@dataclass(frozen=True)
class TranscriptCostConfig:
    """Frozen campaign configuration for the bounded screen."""

    teachers: Tuple[TranscriptTeacher, ...] = DEFAULT_TEACHERS
    train_sizes: Tuple[int, ...] = FROZEN_TRAIN_SIZES
    target_shifted_route_accuracy: float = TARGET_SHIFTED_ROUTE_ACCURACY
    candidate_min_route_accuracy_delta: float = CANDIDATE_MIN_ROUTE_ACCURACY_DELTA
    null_max_route_accuracy_delta: float = NULL_MAX_ROUTE_ACCURACY_DELTA

    def __post_init__(self) -> None:
        if isinstance(self.teachers, (str, bytes)) or not isinstance(self.teachers, Sequence):
            raise TranscriptCostValidationError("teachers must be a sequence of TranscriptTeacher values")
        teachers = tuple(self.teachers)
        if not teachers or len(teachers) > MAX_TEACHER_CELLS:
            raise TranscriptCostValidationError(
                "teachers must contain between 1 and {} cells".format(MAX_TEACHER_CELLS)
            )
        if any(not isinstance(teacher, TranscriptTeacher) for teacher in teachers):
            raise TranscriptCostValidationError("every teacher must be TranscriptTeacher")
        if len({teacher.cell_id for teacher in teachers}) != len(teachers):
            raise TranscriptCostValidationError("teacher cells must be unique")

        if isinstance(self.train_sizes, (str, bytes)) or not isinstance(self.train_sizes, Sequence):
            raise TranscriptCostValidationError("train_sizes must be an increasing integer sequence")
        train_sizes = tuple(_plain_int(value, "train_size", 1) for value in self.train_sizes)
        if not train_sizes or len(train_sizes) > MAX_TRAIN_SIZES:
            raise TranscriptCostValidationError(
                "train_sizes must contain between 1 and {} values".format(MAX_TRAIN_SIZES)
            )
        if tuple(sorted(set(train_sizes))) != train_sizes:
            raise TranscriptCostValidationError("train_sizes must be unique and strictly increasing")

        target = _finite_probability(
            self.target_shifted_route_accuracy,
            "target_shifted_route_accuracy",
        )
        candidate_delta = _finite_probability(
            self.candidate_min_route_accuracy_delta,
            "candidate_min_route_accuracy_delta",
        )
        null_delta = _finite_probability(
            self.null_max_route_accuracy_delta,
            "null_max_route_accuracy_delta",
        )
        object.__setattr__(self, "teachers", teachers)
        object.__setattr__(self, "train_sizes", train_sizes)
        object.__setattr__(self, "target_shifted_route_accuracy", target)
        object.__setattr__(
            self,
            "candidate_min_route_accuracy_delta",
            candidate_delta,
        )
        object.__setattr__(
            self,
            "null_max_route_accuracy_delta",
            null_delta,
        )


@dataclass(frozen=True)
class TranscriptCostBudget:
    """Caller-lowerable limits bounded by immutable hard ceilings."""

    max_estimated_bytes: int = DEFAULT_MAX_ESTIMATED_BYTES
    max_work_units: int = DEFAULT_MAX_WORK_UNITS
    max_seconds: float = DEFAULT_MAX_SECONDS

    def __post_init__(self) -> None:
        estimated_bytes = _plain_int(
            self.max_estimated_bytes,
            "max_estimated_bytes",
            1,
        )
        work_units = _plain_int(
            self.max_work_units,
            "max_work_units",
            1,
        )
        seconds = _positive_seconds(self.max_seconds, "max_seconds")
        if estimated_bytes > HARD_MAX_ESTIMATED_BYTES:
            raise TranscriptCostValidationError("max_estimated_bytes exceeds its hard ceiling")
        if work_units > HARD_MAX_WORK_UNITS:
            raise TranscriptCostValidationError("max_work_units exceeds its hard ceiling")
        if seconds > HARD_MAX_SECONDS:
            raise TranscriptCostValidationError("max_seconds exceeds its hard ceiling")
        object.__setattr__(self, "max_estimated_bytes", estimated_bytes)
        object.__setattr__(self, "max_work_units", work_units)
        object.__setattr__(self, "max_seconds", seconds)


@dataclass(frozen=True)
class TranscriptCostRunContract:
    """Canonical standalone representation of one analyzed run."""

    decision_rule_id: str
    teacher_specs: Tuple[Tuple[str, int, str, int], ...]
    train_sizes: Tuple[int, ...]
    target_shifted_route_accuracy: float
    candidate_min_route_accuracy_delta: float
    null_max_route_accuracy_delta: float
    learner_names: Tuple[str, ...]
    max_estimated_bytes: int
    max_work_units: int
    max_seconds: float


@dataclass(frozen=True)
class TranscriptExample:
    """One raw-input row with a hidden teacher transcript."""

    example_id: str
    split: str
    raw_features: Tuple[int, ...]
    payload_features: Tuple[int, ...]
    causal_features: Tuple[int, ...]
    transcript: Tuple[int, ...]
    route: int


@dataclass(frozen=True)
class TranscriptDataset:
    """ID-disjoint training and shifted environments for one teacher."""

    teacher: TranscriptTeacher
    train: Tuple[TranscriptExample, ...]
    shifted: Tuple[TranscriptExample, ...]
    train_digest: str
    shifted_digest: str
    train_content_digest: str
    shifted_content_digest: str
    identifiers_disjoint: bool
    raw_rows_disjoint: bool
    train_payload_alignment: float
    shifted_payload_alignment: float
    payload_remapped_fraction: float
    equal_environment_witness: bool
    nuisance_remapping_witness: bool
    distribution_shift_witness: bool


@dataclass(frozen=True)
class BitRule:
    """One deterministic Boolean lookup rule."""

    feature_indices: Tuple[int, ...]
    table: Tuple[Tuple[Tuple[int, ...], int], ...]
    fallback: int

    def predict(self, raw_features: Sequence[int]) -> int:
        key = tuple(raw_features[index] for index in self.feature_indices)
        for table_key, value in self.table:
            if table_key == key:
                return value
        return self.fallback


@dataclass(frozen=True)
class TranscriptModel:
    """A fitted multi-bit transcript predictor."""

    name: str
    rules: Tuple[BitRule, ...]
    branch_bits: int
    raw_feature_count: int
    training_examples: int
    candidate_subsets_evaluated: int
    training_search_work_units: int
    actual_parameter_bits: int
    actual_serialized_bytes: int
    actual_inference_operations: int

    def predict(self, raw_features: Sequence[int]) -> Tuple[int, ...]:
        return tuple(rule.predict(raw_features) for rule in self.rules)


@dataclass(frozen=True)
class ControlCharge:
    """Actual and common-envelope modeled resource charges."""

    learner_name: str
    actual_parameter_bits: int
    actual_serialized_bytes: int
    actual_inference_operations: int
    training_search_work_units: int
    matched_parameter_bits: int
    matched_serialized_bytes: int
    matched_inference_operations: int
    side_information_bits_per_example: int
    padding_to_common_envelope: bool
    measured_resources: bool = False

    @property
    def matching_signature(self) -> Tuple[int, int, int]:
        return (
            self.matched_parameter_bits,
            self.matched_serialized_bytes,
            self.matched_inference_operations,
        )


@dataclass(frozen=True)
class LearnerMetric:
    """Shifted-environment performance and cost for one learner."""

    learner_name: str
    shifted_examples: int
    correct_routes: int
    route_accuracy: float
    correct_bits: int
    bit_accuracy: float
    selected_feature_subsets: Tuple[Tuple[int, ...], ...]
    charge: ControlCharge


@dataclass(frozen=True)
class LearningPoint:
    """All learner results at one common training sample count."""

    train_examples: int
    metrics: Tuple[LearnerMetric, ...]
    all_common_envelopes_matched: bool


@dataclass(frozen=True)
class TranscriptCellResult:
    """Frozen learning curve and within-cell candidate/kill decision."""

    cell_id: str
    family: str
    branch_bits: int
    group_width: int
    shift_mode: str
    seed: int
    train_pool_count: int
    shifted_count: int
    train_sizes: Tuple[int, ...]
    learning_curve: Tuple[LearningPoint, ...]
    examples_to_target: Tuple[Tuple[str, Optional[int]], ...]
    final_factorized_minus_flat_route_accuracy: float
    final_factorized_minus_causal_only_route_accuracy: float
    final_factorized_minus_wrong_group_route_accuracy: float
    final_factorized_minus_payload_route_accuracy: float
    final_factorized_minus_random_route_accuracy: float
    final_factorized_minus_flat_actual_parameter_bits: int
    final_factorized_minus_flat_actual_inference_operations: int
    final_flat_to_factorized_training_work_ratio: float
    final_causal_only_to_factorized_training_work_ratio: float
    actual_compression_advantage: bool
    actual_inference_operation_advantage: bool
    factorized_reaches_target_before_unrestricted_flat: bool
    causal_only_counterfactual_matches_factorized: bool
    wrong_group_equal_complexity: bool
    wrong_group_control_underperforms: bool
    unrestricted_flat_shortcut_selection_witness: bool
    equal_environment_witness: bool
    nuisance_remapping_witness: bool
    distribution_shift_witness: bool
    identifiers_disjoint: bool
    matched_inference_envelopes: bool
    candidate_signal: bool
    frozen_screen_kill: bool
    null_sanity_pass: bool
    status: str


@dataclass(frozen=True)
class TranscriptCostResourceGuard:
    """Allocation-free resource estimate retained with the result."""

    teacher_cell_count: int
    maximum_state_count: int
    total_learning_points: int
    maximum_raw_features: int
    maximum_flat_candidate_subsets_per_bit: int
    component_bytes: Tuple[Tuple[str, int], ...]
    component_work: Tuple[Tuple[str, int], ...]
    estimated_peak_bytes: int
    estimated_work_units: int
    max_estimated_bytes: int
    max_work_units: int
    max_seconds: float
    measured_process_peak: bool


@dataclass(frozen=True)
class TranscriptCostResult:
    """Top-level non-confirmatory result."""

    protocol_id: str
    schema_version: str
    evidence_mode: str
    run_contract: TranscriptCostRunContract
    run_contract_digest: str
    resource_guard: TranscriptCostResourceGuard
    cells: Tuple[TranscriptCellResult, ...]
    learner_names: Tuple[str, ...]
    all_identifiers_disjoint: bool
    all_shift_cells_have_distribution_shift_witness: bool
    all_no_shift_cells_have_equal_environment_witness: bool
    all_nuisance_remapping_cells_have_remapping_witness: bool
    all_inference_envelopes_matched: bool
    all_candidate_cells_have_wrong_group_control: bool
    all_nonlinear_candidate_cells_pass: bool
    all_separable_null_cells_pass: bool
    candidate_signal: bool
    frozen_screen_status: str
    hypothesis_status: str
    representational_capacity_claim: bool
    compression_advantage_claim: bool
    real_utility_claim: bool
    novelty_claim: bool
    promotion_eligible: bool
    limitations: Tuple[str, ...]


def _make_run_contract(
    config: TranscriptCostConfig,
    budget: TranscriptCostBudget,
) -> TranscriptCostRunContract:
    return TranscriptCostRunContract(
        decision_rule_id=DECISION_RULE_ID,
        teacher_specs=tuple(
            (
                teacher.family,
                teacher.branch_bits,
                teacher.shift_mode,
                teacher.seed,
            )
            for teacher in config.teachers
        ),
        train_sizes=config.train_sizes,
        target_shifted_route_accuracy=config.target_shifted_route_accuracy,
        candidate_min_route_accuracy_delta=config.candidate_min_route_accuracy_delta,
        null_max_route_accuracy_delta=config.null_max_route_accuracy_delta,
        learner_names=LEARNER_NAMES,
        max_estimated_bytes=budget.max_estimated_bytes,
        max_work_units=budget.max_work_units,
        max_seconds=budget.max_seconds,
    )


def _run_contract_digest(contract: TranscriptCostRunContract) -> str:
    payload = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "evidence_mode": EVIDENCE_MODE,
        "run_contract": asdict(contract),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _teacher_transcript(
    teacher: TranscriptTeacher,
    causal_features: Sequence[int],
) -> Tuple[int, ...]:
    expected = teacher.branch_bits * teacher.group_width
    if len(causal_features) != expected:
        raise TranscriptCostValidationError("causal_features must contain exactly {} bits".format(expected))
    output = []
    for bit_index in range(teacher.branch_bits):
        start = bit_index * teacher.group_width
        group = causal_features[start : start + teacher.group_width]
        if teacher.family == "separable":
            output.append(int(group[0]))
        elif teacher.family == "xor":
            output.append(int(group[0]) ^ int(group[1]))
        else:
            output.append(sum(int(value) for value in group) & 1)
    return tuple(output)


def _payload_features(
    teacher: TranscriptTeacher,
    causal_state: int,
    transcript: Sequence[int],
    split: str,
) -> Tuple[int, ...]:
    if teacher.shift_mode == "spurious_reversal":
        if split == "train":
            return tuple(int(bit) for bit in transcript)
        return tuple(1 - int(bit) for bit in transcript)
    environment_key = "equal_environment" if teacher.shift_mode == "no_shift" else split
    return tuple(
        _stable_int(
            PROTOCOL_ID,
            teacher.cell_id,
            environment_key,
            causal_state,
            bit_index,
        )
        & 1
        for bit_index in range(teacher.branch_bits)
    )


def _dataset_digest(examples: Sequence[TranscriptExample]) -> str:
    hasher = hashlib.sha256()
    for example in examples:
        hasher.update(example.example_id.encode("ascii"))
        hasher.update(bytes(example.raw_features))
        hasher.update(bytes(example.transcript))
    return hasher.hexdigest()


def _dataset_content_digest(examples: Sequence[TranscriptExample]) -> str:
    """Hash modeled content while intentionally excluding split identifiers."""

    hasher = hashlib.sha256()
    for example in examples:
        hasher.update(bytes(example.raw_features))
        hasher.update(bytes(example.causal_features))
        hasher.update(bytes(example.transcript))
        hasher.update(example.route.to_bytes(2, "big"))
    return hasher.hexdigest()


def _alignment(examples: Sequence[TranscriptExample]) -> float:
    matched = sum(
        int(payload == transcript)
        for example in examples
        for payload, transcript in zip(
            example.payload_features,
            example.transcript,
        )
    )
    opportunities = sum(len(example.transcript) for example in examples)
    return matched / opportunities if opportunities else 0.0


def generate_transcript_dataset(
    teacher: TranscriptTeacher,
) -> TranscriptDataset:
    """Generate deterministic train and shifted environments for one teacher."""

    if not isinstance(teacher, TranscriptTeacher):
        raise TranscriptCostValidationError("teacher must be TranscriptTeacher")
    causal_width = teacher.branch_bits * teacher.group_width
    state_count = 1 << causal_width
    if teacher.branch_bits + causal_width > MAX_RAW_FEATURES:
        raise TranscriptCostValidationError("teacher exceeds the raw-feature ceiling")

    split_rows = {}
    for split in ("train", "shifted"):
        rows = []
        for causal_state in range(state_count):
            causal = _bits_from_int(causal_state, causal_width)
            transcript = _teacher_transcript(teacher, causal)
            payload = _payload_features(
                teacher,
                causal_state,
                transcript,
                "train" if split == "train" else "shifted",
            )
            raw = payload + causal
            example_id = hashlib.sha256(
                "{}|{}|{}|{}".format(
                    PROTOCOL_ID,
                    teacher.cell_id,
                    split,
                    causal_state,
                ).encode("ascii")
            ).hexdigest()[:24]
            rows.append(
                TranscriptExample(
                    example_id=example_id,
                    split=split,
                    raw_features=raw,
                    payload_features=payload,
                    causal_features=causal,
                    transcript=transcript,
                    route=_route_from_bits(transcript),
                )
            )
        split_rows[split] = tuple(rows)

    train = split_rows["train"]
    shifted = split_rows["shifted"]
    train_ids = {example.example_id for example in train}
    shifted_ids = {example.example_id for example in shifted}
    train_raw = {example.raw_features for example in train}
    shifted_raw = {example.raw_features for example in shifted}
    train_alignment = _alignment(train)
    shifted_alignment = _alignment(shifted)
    train_content_digest = _dataset_content_digest(train)
    shifted_content_digest = _dataset_content_digest(shifted)
    causal_targets_equal = all(
        train_row.causal_features == shifted_row.causal_features
        and train_row.transcript == shifted_row.transcript
        and train_row.route == shifted_row.route
        for train_row, shifted_row in zip(train, shifted)
    )
    remapped_bits = sum(
        int(train_payload != shifted_payload)
        for train_row, shifted_row in zip(train, shifted)
        for train_payload, shifted_payload in zip(
            train_row.payload_features,
            shifted_row.payload_features,
        )
    )
    payload_opportunities = len(train) * teacher.branch_bits
    payload_remapped_fraction = remapped_bits / payload_opportunities
    equal_environment_witness = (
        teacher.shift_mode == "no_shift"
        and causal_targets_equal
        and train_content_digest == shifted_content_digest
        and all(train_row.raw_features == shifted_row.raw_features for train_row, shifted_row in zip(train, shifted))
    )
    nuisance_remapping_witness = (
        teacher.shift_mode == "nuisance_remapping"
        and causal_targets_equal
        and payload_remapped_fraction > 0.0
        and train_content_digest != shifted_content_digest
    )
    distribution_shift_witness = (
        teacher.shift_mode == "spurious_reversal"
        and train_alignment == 1.0
        and shifted_alignment == 0.0
        and payload_remapped_fraction == 1.0
        and not (train_raw & shifted_raw)
    )
    return TranscriptDataset(
        teacher=teacher,
        train=train,
        shifted=shifted,
        train_digest=_dataset_digest(train),
        shifted_digest=_dataset_digest(shifted),
        train_content_digest=train_content_digest,
        shifted_content_digest=shifted_content_digest,
        identifiers_disjoint=not bool(train_ids & shifted_ids),
        raw_rows_disjoint=not bool(train_raw & shifted_raw),
        train_payload_alignment=train_alignment,
        shifted_payload_alignment=shifted_alignment,
        payload_remapped_fraction=payload_remapped_fraction,
        equal_environment_witness=equal_environment_witness,
        nuisance_remapping_witness=nuisance_remapping_witness,
        distribution_shift_witness=distribution_shift_witness,
    )


def _candidate_subsets(
    feature_indices: Sequence[int],
    maximum_width: int,
) -> Tuple[Tuple[int, ...], ...]:
    subsets = []
    for width in range(1, min(maximum_width, len(feature_indices)) + 1):
        subsets.extend(itertools.combinations(feature_indices, width))
    return tuple(subsets)


def _majority(zero_count: int, one_count: int) -> int:
    return 1 if one_count > zero_count else 0


def _fit_bit_rule(
    examples: Sequence[TranscriptExample],
    feature_indices: Tuple[int, ...],
    transcript_index: int,
) -> Tuple[BitRule, int, int]:
    counts: Dict[Tuple[int, ...], List[int]] = {}
    total = [0, 0]
    for example in examples:
        key = tuple(example.raw_features[index] for index in feature_indices)
        bucket = counts.setdefault(key, [0, 0])
        target = example.transcript[transcript_index]
        bucket[target] += 1
        total[target] += 1
    fallback = _majority(total[0], total[1])
    table = tuple((key, _majority(values[0], values[1])) for key, values in sorted(counts.items()))
    rule = BitRule(feature_indices, table, fallback)
    errors = sum(rule.predict(example.raw_features) != example.transcript[transcript_index] for example in examples)
    work = len(examples) * (2 * len(feature_indices) + 4)
    return rule, int(errors), work


def _actual_parameter_bits(
    rules: Sequence[BitRule],
    raw_feature_count: int,
) -> int:
    index_bits = max(1, math.ceil(math.log2(max(2, raw_feature_count))))
    return sum(len(rule.feature_indices) * index_bits + len(rule.table) + 1 for rule in rules)


def _model_from_rules(
    name: str,
    rules: Sequence[BitRule],
    raw_feature_count: int,
    training_examples: int,
    candidate_subsets_evaluated: int,
    training_search_work_units: int,
) -> TranscriptModel:
    rules_tuple = tuple(rules)
    parameter_bits = _actual_parameter_bits(rules_tuple, raw_feature_count)
    inference_operations = sum(len(rule.feature_indices) + 2 for rule in rules_tuple)
    return TranscriptModel(
        name=name,
        rules=rules_tuple,
        branch_bits=len(rules_tuple),
        raw_feature_count=raw_feature_count,
        training_examples=training_examples,
        candidate_subsets_evaluated=candidate_subsets_evaluated,
        training_search_work_units=training_search_work_units,
        actual_parameter_bits=parameter_bits,
        actual_serialized_bytes=(parameter_bits + 7) // 8,
        actual_inference_operations=inference_operations,
    )


def _fit_factorized(
    examples: Sequence[TranscriptExample],
    teacher: TranscriptTeacher,
) -> TranscriptModel:
    raw_feature_count = len(examples[0].raw_features)
    rules = []
    work = 0
    for transcript_index in range(teacher.branch_bits):
        causal_start = teacher.branch_bits + transcript_index * teacher.group_width
        if teacher.family == "separable":
            subset = (causal_start,)
        else:
            subset = tuple(range(causal_start, causal_start + teacher.group_width))
        rule, _errors, rule_work = _fit_bit_rule(
            examples,
            subset,
            transcript_index,
        )
        rules.append(rule)
        work += rule_work
    return _model_from_rules(
        "factorized",
        rules,
        raw_feature_count,
        len(examples),
        teacher.branch_bits,
        work,
    )


def _fit_wrong_group_factorized(
    examples: Sequence[TranscriptExample],
    teacher: TranscriptTeacher,
) -> TranscriptModel:
    """Fit the same-width lookup rules on deterministic non-causal groupings."""

    raw_feature_count = len(examples[0].raw_features)
    causal_offset = teacher.branch_bits
    rules = []
    work = 0
    for transcript_index in range(teacher.branch_bits):
        if teacher.family == "separable":
            wrong_group = (transcript_index + 1) % teacher.branch_bits
            subset = (causal_offset + wrong_group * teacher.group_width,)
        else:
            subset = tuple(
                sorted(
                    causal_offset
                    + ((transcript_index + within_group) % teacher.branch_bits) * teacher.group_width
                    + within_group
                    for within_group in range(teacher.group_width)
                )
            )
        rule, _errors, rule_work = _fit_bit_rule(
            examples,
            subset,
            transcript_index,
        )
        rules.append(rule)
        work += rule_work
    return _model_from_rules(
        "wrong_group_factorized",
        rules,
        raw_feature_count,
        len(examples),
        teacher.branch_bits,
        work,
    )


def _fit_feature_search(
    examples: Sequence[TranscriptExample],
    teacher: TranscriptTeacher,
    name: str,
    allowed_features: Sequence[int],
    deadline: float,
) -> TranscriptModel:
    raw_feature_count = len(examples[0].raw_features)
    candidates = _candidate_subsets(
        tuple(allowed_features),
        teacher.group_width,
    )
    if not candidates:
        raise TranscriptCostValidationError("{} has no candidate feature subsets".format(name))
    rules = []
    total_work = 0
    for transcript_index in range(teacher.branch_bits):
        best = None
        for candidate_index, subset in enumerate(candidates):
            if candidate_index % 32 == 0:
                _check_deadline(deadline)
            rule, errors, work = _fit_bit_rule(
                examples,
                subset,
                transcript_index,
            )
            total_work += work
            # Fewer errors wins.  Minimum description length then prefers fewer
            # raw features and fewer observed table rows before canonical order.
            score = (
                errors,
                len(rule.feature_indices),
                len(rule.table),
                rule.feature_indices,
            )
            if best is None or score < best[0]:
                best = (score, rule)
        if best is None:
            raise TranscriptCostValidationError("{} failed to select a feature subset".format(name))
        rules.append(best[1])
    return _model_from_rules(
        name,
        rules,
        raw_feature_count,
        len(examples),
        len(candidates) * teacher.branch_bits,
        total_work,
    )


def _common_envelope(
    teacher: TranscriptTeacher,
    raw_feature_count: int,
) -> Tuple[int, int, int]:
    index_bits = max(1, math.ceil(math.log2(max(2, raw_feature_count))))
    parameter_bits = teacher.branch_bits * (teacher.group_width * index_bits + (1 << teacher.group_width) + 1)
    serialized_bytes = (parameter_bits + 7) // 8
    inference_operations = teacher.branch_bits * (teacher.group_width + 2)
    return parameter_bits, serialized_bytes, inference_operations


def _charge(
    learner_name: str,
    teacher: TranscriptTeacher,
    raw_feature_count: int,
    model: Optional[TranscriptModel],
) -> ControlCharge:
    matched_bits, matched_bytes, matched_operations = _common_envelope(
        teacher,
        raw_feature_count,
    )
    actual_bits = model.actual_parameter_bits if model is not None else 0
    actual_bytes = model.actual_serialized_bytes if model is not None else 0
    actual_operations = model.actual_inference_operations if model is not None else 0
    training_work = model.training_search_work_units if model is not None else 0
    return ControlCharge(
        learner_name=learner_name,
        actual_parameter_bits=actual_bits,
        actual_serialized_bytes=actual_bytes,
        actual_inference_operations=actual_operations,
        training_search_work_units=training_work,
        matched_parameter_bits=matched_bits,
        matched_serialized_bytes=matched_bytes,
        matched_inference_operations=matched_operations,
        side_information_bits_per_example=(teacher.branch_bits if learner_name == "paid_transcript" else 0),
        padding_to_common_envelope=(
            actual_bits < matched_bits or actual_bytes < matched_bytes or actual_operations < matched_operations
        ),
    )


def _evaluate(
    learner_name: str,
    examples: Sequence[TranscriptExample],
    teacher: TranscriptTeacher,
    model: Optional[TranscriptModel],
) -> LearnerMetric:
    correct_routes = 0
    correct_bits = 0
    selected_subsets: Tuple[Tuple[int, ...], ...] = ()
    if model is not None:
        selected_subsets = tuple(rule.feature_indices for rule in model.rules)
    for example in examples:
        if learner_name == "paid_transcript":
            prediction = example.transcript
        elif learner_name == "deterministic_random":
            prediction = _bits_from_int(
                _stable_int(
                    PROTOCOL_ID,
                    teacher.cell_id,
                    learner_name,
                    example.example_id,
                )
                % (1 << teacher.branch_bits),
                teacher.branch_bits,
            )
        else:
            if model is None:
                raise TranscriptCostValidationError("{} requires a fitted model".format(learner_name))
            prediction = model.predict(example.raw_features)
        correct_routes += int(tuple(prediction) == example.transcript)
        correct_bits += sum(int(predicted == target) for predicted, target in zip(prediction, example.transcript))
    shifted_count = len(examples)
    bit_count = shifted_count * teacher.branch_bits
    raw_feature_count = len(examples[0].raw_features)
    return LearnerMetric(
        learner_name=learner_name,
        shifted_examples=shifted_count,
        correct_routes=correct_routes,
        route_accuracy=correct_routes / shifted_count,
        correct_bits=correct_bits,
        bit_accuracy=correct_bits / bit_count,
        selected_feature_subsets=selected_subsets,
        charge=_charge(
            learner_name,
            teacher,
            raw_feature_count,
            model,
        ),
    )


def metric_by_name(
    metrics: Sequence[LearnerMetric],
    learner_name: str,
) -> LearnerMetric:
    """Return one named metric or fail rather than silently selecting."""

    matches = [metric for metric in metrics if metric.learner_name == learner_name]
    if len(matches) != 1:
        raise TranscriptCostValidationError("expected exactly one metric named {!r}".format(learner_name))
    return matches[0]


def _ordered_training_rows(
    dataset: TranscriptDataset,
) -> Tuple[TranscriptExample, ...]:
    rows = list(dataset.train)
    random.Random(dataset.teacher.seed).shuffle(rows)
    return tuple(rows)


def _effective_train_sizes(
    config: TranscriptCostConfig,
    state_count: int,
) -> Tuple[int, ...]:
    sizes = [size for size in config.train_sizes if size <= state_count]
    if state_count not in sizes:
        sizes.append(state_count)
    return tuple(sorted(set(sizes)))


def _combination_count(feature_count: int, width: int) -> int:
    return sum(math.comb(feature_count, candidate_width) for candidate_width in range(1, min(feature_count, width) + 1))


def preflight_transcript_cost(
    config: TranscriptCostConfig = TranscriptCostConfig(),
    budget: TranscriptCostBudget = TranscriptCostBudget(),
) -> TranscriptCostResourceGuard:
    """Return a conservative allocation/work model or refuse before generation."""

    if not isinstance(config, TranscriptCostConfig):
        raise TranscriptCostValidationError("config must be TranscriptCostConfig")
    if not isinstance(budget, TranscriptCostBudget):
        raise TranscriptCostValidationError("budget must be TranscriptCostBudget")

    maximum_states = 0
    total_learning_points = 0
    maximum_raw_features = 0
    maximum_candidates = 0
    dataset_bytes = 0
    retained_result_bytes = 0
    learner_work = 0
    evaluation_work = 0

    for teacher in config.teachers:
        causal_width = teacher.branch_bits * teacher.group_width
        state_count = 1 << causal_width
        raw_features = teacher.branch_bits + causal_width
        sizes = _effective_train_sizes(config, state_count)
        flat_candidates = _combination_count(
            raw_features,
            teacher.group_width,
        )
        maximum_states = max(maximum_states, state_count)
        maximum_raw_features = max(maximum_raw_features, raw_features)
        maximum_candidates = max(maximum_candidates, flat_candidates)
        total_learning_points += len(sizes)

        # Conservative Python-object model.  Both environments coexist while a
        # cell is evaluated; only compact dataclass summaries are retained.
        row_bytes = 512 + 64 * (raw_features + teacher.branch_bits)
        dataset_bytes = max(dataset_bytes, 2 * state_count * row_bytes)
        retained_result_bytes += len(sizes) * len(LEARNER_NAMES) * 2_048 + 8_192

        weighted_flat_candidates = sum(
            math.comb(raw_features, width) * (2 * width + 4) for width in range(1, teacher.group_width + 1)
        )
        weighted_payload_candidates = sum(
            math.comb(teacher.branch_bits, width) * (2 * width + 4)
            for width in range(
                1,
                min(teacher.branch_bits, teacher.group_width) + 1,
            )
        )
        weighted_causal_candidates = sum(
            math.comb(causal_width, width) * (2 * width + 4)
            for width in range(
                1,
                min(causal_width, teacher.group_width) + 1,
            )
        )
        factorized_width = 1 if teacher.family == "separable" else teacher.group_width
        for train_size in sizes:
            learner_work += (
                teacher.branch_bits
                * train_size
                * (
                    (2 * factorized_width + 4)
                    + (2 * factorized_width + 4)
                    + weighted_flat_candidates
                    + weighted_causal_candidates
                    + weighted_payload_candidates
                )
            )
            common_operations = teacher.branch_bits * (teacher.group_width + 2)
            evaluation_work += state_count * (5 * common_operations + teacher.branch_bits + teacher.branch_bits)

    component_bytes = (
        ("largest_two_environment_dataset", dataset_bytes),
        ("retained_learning_curve_summaries", retained_result_bytes),
        ("allocator_and_interpreter_reserve", 1_048_576),
    )
    component_work = (
        ("learner_fit_and_feature_search", learner_work),
        ("shifted_environment_evaluation", evaluation_work),
        ("digest_and_split_checks", maximum_states * len(config.teachers) * 16),
    )
    estimated_peak_bytes = sum(value for _name, value in component_bytes)
    estimated_work_units = sum(value for _name, value in component_work)
    guard = TranscriptCostResourceGuard(
        teacher_cell_count=len(config.teachers),
        maximum_state_count=maximum_states,
        total_learning_points=total_learning_points,
        maximum_raw_features=maximum_raw_features,
        maximum_flat_candidate_subsets_per_bit=maximum_candidates,
        component_bytes=component_bytes,
        component_work=component_work,
        estimated_peak_bytes=estimated_peak_bytes,
        estimated_work_units=estimated_work_units,
        max_estimated_bytes=budget.max_estimated_bytes,
        max_work_units=budget.max_work_units,
        max_seconds=budget.max_seconds,
        measured_process_peak=False,
    )
    if estimated_peak_bytes > budget.max_estimated_bytes:
        raise TranscriptCostResourceError(
            "estimated peak {} bytes exceeds budget {}".format(
                estimated_peak_bytes,
                budget.max_estimated_bytes,
            )
        )
    if estimated_work_units > budget.max_work_units:
        raise TranscriptCostResourceError(
            "estimated work {} exceeds budget {}".format(
                estimated_work_units,
                budget.max_work_units,
            )
        )
    return guard


def _examples_to_target(
    curve: Sequence[LearningPoint],
    learner_name: str,
    target: float,
) -> Optional[int]:
    for point in curve:
        metric = metric_by_name(point.metrics, learner_name)
        if metric.route_accuracy >= target:
            return point.train_examples
    return None


def _analyze_cell(
    teacher: TranscriptTeacher,
    config: TranscriptCostConfig,
    deadline: float,
) -> TranscriptCellResult:
    dataset = generate_transcript_dataset(teacher)
    ordered_train = _ordered_training_rows(dataset)
    train_sizes = _effective_train_sizes(config, len(ordered_train))
    curve = []

    for train_size in train_sizes:
        _check_deadline(deadline)
        train = ordered_train[:train_size]
        factorized = _fit_factorized(train, teacher)
        wrong_group_factorized = _fit_wrong_group_factorized(train, teacher)
        flat = _fit_feature_search(
            train,
            teacher,
            "matched_flat",
            tuple(range(len(train[0].raw_features))),
            deadline,
        )
        causal_only_flat = _fit_feature_search(
            train,
            teacher,
            "causal_only_flat",
            tuple(range(teacher.branch_bits, len(train[0].raw_features))),
            deadline,
        )
        payload_only = _fit_feature_search(
            train,
            teacher,
            "payload_only",
            tuple(range(teacher.branch_bits)),
            deadline,
        )
        metrics = (
            _evaluate(
                "factorized",
                dataset.shifted,
                teacher,
                factorized,
            ),
            _evaluate(
                "matched_flat",
                dataset.shifted,
                teacher,
                flat,
            ),
            _evaluate(
                "causal_only_flat",
                dataset.shifted,
                teacher,
                causal_only_flat,
            ),
            _evaluate(
                "wrong_group_factorized",
                dataset.shifted,
                teacher,
                wrong_group_factorized,
            ),
            _evaluate(
                "paid_transcript",
                dataset.shifted,
                teacher,
                None,
            ),
            _evaluate(
                "payload_only",
                dataset.shifted,
                teacher,
                payload_only,
            ),
            _evaluate(
                "deterministic_random",
                dataset.shifted,
                teacher,
                None,
            ),
        )
        signatures = {metric.charge.matching_signature for metric in metrics}
        curve.append(
            LearningPoint(
                train_examples=train_size,
                metrics=metrics,
                all_common_envelopes_matched=len(signatures) == 1,
            )
        )

    final_metrics = curve[-1].metrics
    factorized_metric = metric_by_name(final_metrics, "factorized")
    flat_metric = metric_by_name(final_metrics, "matched_flat")
    causal_only_metric = metric_by_name(final_metrics, "causal_only_flat")
    wrong_group_metric = metric_by_name(
        final_metrics,
        "wrong_group_factorized",
    )
    paid_metric = metric_by_name(final_metrics, "paid_transcript")
    payload_metric = metric_by_name(final_metrics, "payload_only")
    random_metric = metric_by_name(final_metrics, "deterministic_random")

    examples_to_target = tuple(
        (
            learner_name,
            (
                0
                if learner_name == "paid_transcript"
                else _examples_to_target(
                    curve,
                    learner_name,
                    config.target_shifted_route_accuracy,
                )
            ),
        )
        for learner_name in LEARNER_NAMES
    )
    sample_map = dict(examples_to_target)
    factorized_samples = sample_map["factorized"]
    flat_samples = sample_map["matched_flat"]
    factorized_precedes_unrestricted_flat = factorized_samples is not None and (
        flat_samples is None or factorized_samples < flat_samples
    )

    delta_flat = factorized_metric.route_accuracy - flat_metric.route_accuracy
    delta_causal_only = factorized_metric.route_accuracy - causal_only_metric.route_accuracy
    delta_wrong_group = factorized_metric.route_accuracy - wrong_group_metric.route_accuracy
    delta_payload = factorized_metric.route_accuracy - payload_metric.route_accuracy
    delta_random = factorized_metric.route_accuracy - random_metric.route_accuracy
    parameter_bit_delta = factorized_metric.charge.actual_parameter_bits - flat_metric.charge.actual_parameter_bits
    inference_operation_delta = (
        factorized_metric.charge.actual_inference_operations - flat_metric.charge.actual_inference_operations
    )
    factorized_training_work = factorized_metric.charge.training_search_work_units
    flat_to_factorized_work_ratio = flat_metric.charge.training_search_work_units / factorized_training_work
    causal_only_to_factorized_work_ratio = (
        causal_only_metric.charge.training_search_work_units / factorized_training_work
    )
    matched = all(point.all_common_envelopes_matched for point in curve)
    causal_only_matches = (
        causal_only_metric.route_accuracy >= config.target_shifted_route_accuracy
        and abs(delta_causal_only) <= config.null_max_route_accuracy_delta
        and causal_only_metric.selected_feature_subsets == factorized_metric.selected_feature_subsets
    )
    wrong_group_equal_complexity = (
        wrong_group_metric.charge.actual_parameter_bits == factorized_metric.charge.actual_parameter_bits
        and wrong_group_metric.charge.actual_serialized_bytes == factorized_metric.charge.actual_serialized_bytes
        and wrong_group_metric.charge.actual_inference_operations
        == factorized_metric.charge.actual_inference_operations
        and wrong_group_metric.charge.training_search_work_units == factorized_metric.charge.training_search_work_units
        and wrong_group_metric.selected_feature_subsets != factorized_metric.selected_feature_subsets
    )
    wrong_group_underperforms = delta_wrong_group >= config.candidate_min_route_accuracy_delta
    flat_selects_payload_shortcut = bool(flat_metric.selected_feature_subsets) and all(
        subset and all(index < teacher.branch_bits for index in subset)
        for subset in flat_metric.selected_feature_subsets
    )
    shortcut_selection_witness = (
        teacher.nonlinear
        and teacher.shift_mode == "spurious_reversal"
        and causal_only_matches
        and flat_selects_payload_shortcut
        and delta_flat >= config.candidate_min_route_accuracy_delta
    )
    structural_prior_training_search_witness = causal_only_matches and causal_only_to_factorized_work_ratio > 1.0
    candidate_signal = (
        teacher.nonlinear
        and teacher.shift_mode == "spurious_reversal"
        and dataset.distribution_shift_witness
        and dataset.identifiers_disjoint
        and matched
        and paid_metric.route_accuracy == 1.0
        and factorized_metric.route_accuracy >= config.target_shifted_route_accuracy
        and delta_flat >= config.candidate_min_route_accuracy_delta
        and delta_payload >= config.candidate_min_route_accuracy_delta
        and delta_random >= config.candidate_min_route_accuracy_delta
        and factorized_precedes_unrestricted_flat
        and shortcut_selection_witness
        and structural_prior_training_search_witness
        and wrong_group_equal_complexity
        and wrong_group_underperforms
    )
    null_sanity_pass = (
        teacher.family == "separable"
        and factorized_metric.route_accuracy >= config.target_shifted_route_accuracy
        and flat_metric.route_accuracy >= config.target_shifted_route_accuracy
        and causal_only_metric.route_accuracy >= config.target_shifted_route_accuracy
        and abs(delta_flat) <= config.null_max_route_accuracy_delta
        and abs(delta_causal_only) <= config.null_max_route_accuracy_delta
    )
    frozen_screen_kill = teacher.nonlinear and not candidate_signal

    if teacher.family == "separable":
        status = NULL_PASS_STATUS if null_sanity_pass else NULL_FAIL_STATUS
    else:
        status = CANDIDATE_STATUS if candidate_signal else NO_SIGNAL_STATUS
    return TranscriptCellResult(
        cell_id=teacher.cell_id,
        family=teacher.family,
        branch_bits=teacher.branch_bits,
        group_width=teacher.group_width,
        shift_mode=teacher.shift_mode,
        seed=teacher.seed,
        train_pool_count=len(dataset.train),
        shifted_count=len(dataset.shifted),
        train_sizes=train_sizes,
        learning_curve=tuple(curve),
        examples_to_target=examples_to_target,
        final_factorized_minus_flat_route_accuracy=delta_flat,
        final_factorized_minus_causal_only_route_accuracy=delta_causal_only,
        final_factorized_minus_wrong_group_route_accuracy=delta_wrong_group,
        final_factorized_minus_payload_route_accuracy=delta_payload,
        final_factorized_minus_random_route_accuracy=delta_random,
        final_factorized_minus_flat_actual_parameter_bits=parameter_bit_delta,
        final_factorized_minus_flat_actual_inference_operations=(inference_operation_delta),
        final_flat_to_factorized_training_work_ratio=(flat_to_factorized_work_ratio),
        final_causal_only_to_factorized_training_work_ratio=(causal_only_to_factorized_work_ratio),
        actual_compression_advantage=parameter_bit_delta < 0,
        actual_inference_operation_advantage=inference_operation_delta < 0,
        factorized_reaches_target_before_unrestricted_flat=(factorized_precedes_unrestricted_flat),
        causal_only_counterfactual_matches_factorized=causal_only_matches,
        wrong_group_equal_complexity=wrong_group_equal_complexity,
        wrong_group_control_underperforms=wrong_group_underperforms,
        unrestricted_flat_shortcut_selection_witness=(shortcut_selection_witness),
        equal_environment_witness=dataset.equal_environment_witness,
        nuisance_remapping_witness=dataset.nuisance_remapping_witness,
        distribution_shift_witness=dataset.distribution_shift_witness,
        identifiers_disjoint=dataset.identifiers_disjoint,
        matched_inference_envelopes=matched,
        candidate_signal=candidate_signal,
        frozen_screen_kill=frozen_screen_kill,
        null_sanity_pass=null_sanity_pass,
        status=status,
    )


def analyze_transcript_cost(
    config: TranscriptCostConfig = TranscriptCostConfig(),
    budget: TranscriptCostBudget = TranscriptCostBudget(),
) -> TranscriptCostResult:
    """Run the frozen resource-bounded structural-prior training-search screen."""

    resource_guard = preflight_transcript_cost(config, budget)
    run_contract = _make_run_contract(config, budget)
    deadline = time.monotonic() + budget.max_seconds
    cells = []
    for teacher in config.teachers:
        _check_deadline(deadline)
        cells.append(_analyze_cell(teacher, config, deadline))
    cells_tuple = tuple(cells)

    nonlinear_cells = tuple(cell for cell in cells_tuple if cell.family in ("xor", "parity"))
    separable_cells = tuple(cell for cell in cells_tuple if cell.family == "separable")
    all_nonlinear_pass = bool(nonlinear_cells) and all(cell.candidate_signal for cell in nonlinear_cells)
    all_separable_pass = bool(separable_cells) and all(cell.null_sanity_pass for cell in separable_cells)
    all_shift_witnesses = all(
        cell.distribution_shift_witness for cell in cells_tuple if cell.shift_mode == "spurious_reversal"
    ) and any(cell.shift_mode == "spurious_reversal" for cell in cells_tuple)
    all_no_shift_witnesses = all(
        cell.equal_environment_witness for cell in cells_tuple if cell.shift_mode == "no_shift"
    ) and any(cell.shift_mode == "no_shift" for cell in cells_tuple)
    all_nuisance_remapping_witnesses = all(
        cell.nuisance_remapping_witness for cell in cells_tuple if cell.shift_mode == "nuisance_remapping"
    ) and any(cell.shift_mode == "nuisance_remapping" for cell in cells_tuple)
    all_wrong_group_controls = bool(nonlinear_cells) and all(
        cell.wrong_group_equal_complexity and cell.wrong_group_control_underperforms for cell in nonlinear_cells
    )
    all_identifiers_disjoint = all(cell.identifiers_disjoint for cell in cells_tuple)
    all_inference_envelopes_matched = all(cell.matched_inference_envelopes for cell in cells_tuple)
    candidate_signal = (
        all_nonlinear_pass
        and all_separable_pass
        and all_shift_witnesses
        and all_no_shift_witnesses
        and all_nuisance_remapping_witnesses
        and all_wrong_group_controls
        and all_identifiers_disjoint
        and all_inference_envelopes_matched
    )
    return TranscriptCostResult(
        protocol_id=PROTOCOL_ID,
        schema_version=SCHEMA_VERSION,
        evidence_mode=EVIDENCE_MODE,
        run_contract=run_contract,
        run_contract_digest=_run_contract_digest(run_contract),
        resource_guard=resource_guard,
        cells=cells_tuple,
        learner_names=LEARNER_NAMES,
        all_identifiers_disjoint=all_identifiers_disjoint,
        all_shift_cells_have_distribution_shift_witness=all_shift_witnesses,
        all_no_shift_cells_have_equal_environment_witness=(all_no_shift_witnesses),
        all_nuisance_remapping_cells_have_remapping_witness=(all_nuisance_remapping_witnesses),
        all_inference_envelopes_matched=all_inference_envelopes_matched,
        all_candidate_cells_have_wrong_group_control=(all_wrong_group_controls),
        all_nonlinear_candidate_cells_pass=all_nonlinear_pass,
        all_separable_null_cells_pass=all_separable_pass,
        candidate_signal=candidate_signal,
        frozen_screen_status=(CANDIDATE_STATUS if candidate_signal else NO_SIGNAL_STATUS),
        hypothesis_status=GLOBAL_HYPOTHESIS_STATUS,
        representational_capacity_claim=False,
        compression_advantage_claim=False,
        real_utility_claim=False,
        novelty_claim=False,
        promotion_eligible=False,
        limitations=LIMITATIONS,
    )


def validate_result_integrity(result: TranscriptCostResult) -> bool:
    """Fail closed if a standalone result no longer matches its run contract."""

    if not isinstance(result, TranscriptCostResult):
        raise TranscriptCostValidationError("result must be TranscriptCostResult")
    if (
        result.protocol_id != PROTOCOL_ID
        or result.schema_version != SCHEMA_VERSION
        or result.evidence_mode != EVIDENCE_MODE
    ):
        raise TranscriptCostValidationError("result protocol metadata does not match this implementation")
    if result.run_contract.decision_rule_id != DECISION_RULE_ID:
        raise TranscriptCostValidationError("result decision rule is unknown")
    if result.run_contract_digest != _run_contract_digest(result.run_contract):
        raise TranscriptCostValidationError("result run-contract digest mismatch")
    if result.learner_names != LEARNER_NAMES or result.run_contract.learner_names != LEARNER_NAMES:
        raise TranscriptCostValidationError("result learner contract mismatch")
    observed_specs = tuple(
        (
            cell.family,
            cell.branch_bits,
            cell.shift_mode,
            cell.seed,
        )
        for cell in result.cells
    )
    if observed_specs != result.run_contract.teacher_specs:
        raise TranscriptCostValidationError("result cells do not match the run contract")
    for cell in result.cells:
        expected_cell_id = TranscriptTeacher(
            cell.family,
            cell.branch_bits,
            cell.shift_mode,
            cell.seed,
        ).cell_id
        if cell.cell_id != expected_cell_id:
            raise TranscriptCostValidationError("result cell identifier mismatch")
    guard_budget = (
        result.resource_guard.max_estimated_bytes,
        result.resource_guard.max_work_units,
        result.resource_guard.max_seconds,
    )
    contract_budget = (
        result.run_contract.max_estimated_bytes,
        result.run_contract.max_work_units,
        result.run_contract.max_seconds,
    )
    if guard_budget != contract_budget:
        raise TranscriptCostValidationError("resource guard does not match the run contract")
    recomputed_candidate = (
        result.all_nonlinear_candidate_cells_pass
        and result.all_separable_null_cells_pass
        and result.all_shift_cells_have_distribution_shift_witness
        and result.all_no_shift_cells_have_equal_environment_witness
        and result.all_nuisance_remapping_cells_have_remapping_witness
        and result.all_candidate_cells_have_wrong_group_control
        and result.all_identifiers_disjoint
        and result.all_inference_envelopes_matched
    )
    if result.candidate_signal != recomputed_candidate:
        raise TranscriptCostValidationError("result candidate decision is internally inconsistent")
    expected_status = CANDIDATE_STATUS if recomputed_candidate else NO_SIGNAL_STATUS
    if result.frozen_screen_status != expected_status:
        raise TranscriptCostValidationError("result status is internally inconsistent")
    return True


def result_as_dict(result: TranscriptCostResult) -> Dict[str, object]:
    """Return an artifact-friendly representation without changing evidence tier."""

    validate_result_integrity(result)
    return asdict(result)


__all__ = [
    "CANDIDATE_MIN_ROUTE_ACCURACY_DELTA",
    "CANDIDATE_STATUS",
    "DEFAULT_MAX_ESTIMATED_BYTES",
    "DEFAULT_MAX_SECONDS",
    "DEFAULT_MAX_WORK_UNITS",
    "DEFAULT_TEACHERS",
    "DECISION_RULE_ID",
    "EVIDENCE_MODE",
    "FROZEN_TRAIN_SIZES",
    "GLOBAL_HYPOTHESIS_STATUS",
    "HARD_MAX_ESTIMATED_BYTES",
    "HARD_MAX_SECONDS",
    "HARD_MAX_WORK_UNITS",
    "LEARNER_NAMES",
    "LIMITATIONS",
    "NO_SIGNAL_STATUS",
    "NULL_MAX_ROUTE_ACCURACY_DELTA",
    "NULL_PASS_STATUS",
    "PROTOCOL_ID",
    "TARGET_SHIFTED_ROUTE_ACCURACY",
    "ControlCharge",
    "LearnerMetric",
    "LearningPoint",
    "TranscriptCellResult",
    "TranscriptCostBudget",
    "TranscriptCostConfig",
    "TranscriptCostResourceError",
    "TranscriptCostResourceGuard",
    "TranscriptCostResult",
    "TranscriptCostRunContract",
    "TranscriptCostValidationError",
    "TranscriptDataset",
    "TranscriptExample",
    "TranscriptTeacher",
    "analyze_transcript_cost",
    "generate_transcript_dataset",
    "metric_by_name",
    "preflight_transcript_cost",
    "result_as_dict",
    "validate_result_integrity",
]
