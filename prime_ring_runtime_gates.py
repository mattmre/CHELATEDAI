"""Exact runtime-gate decision-table screen over the binary ring ``Z_7``.

The bounded question is deliberately narrower than a learned-memory claim:
does a program with at most two query/state-dependent predicate bits and four
operations expose output states that cannot be represented by a flat table of
at most ``2**b`` static affine-plus-last-write-wins leaves?

The flat control is charged the same predicate transcript, leaf capacity,
replacement templates, mask overlap, search count, storage envelope, and work
envelope as the runtime program.  It is explicitly *given* the paid branch
transcript.  The screen therefore tests behavioral state count, not whether a
flat implementation can infer the transcript more cheaply.

All 128 binary payloads and all four two-bit queries are streamed exactly.  No
model, corpus, training loop, retrieval benchmark, or large allocation is
involved.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Iterable, Sequence, Tuple, Union

from prime_ring_conditional_replacements import (
    AddressOperation,
    AffineMap,
    OverwriteRule,
    ReplacementNormalForm,
    apply_normal_form,
    apply_overwrite,
    compose_affine,
    compose_overwrites,
    relabel_overwrite,
    relabel_payload,
)


PROTOCOL_ID = "PRW-RUNTIME-GATE-P7-001"
MODULUS = 7
PAYLOAD_COUNT = 1 << MODULUS
QUERY_BITS = 2
QUERY_COUNT = 1 << QUERY_BITS
PREDICATE_INPUT_COUNT = PAYLOAD_COUNT * QUERY_COUNT
MAX_GATE_BITS = 2
MAX_OPERATIONS = 4

HARD_MAX_ESTIMATED_BYTES = 1_048_576
HARD_MAX_WORK_UNITS = 1_000_000
HARD_MAX_SECONDS = 5.0
DEFAULT_MAX_ESTIMATED_BYTES = 262_144
DEFAULT_MAX_WORK_UNITS = 250_000
DEFAULT_MAX_SECONDS = 2.0

EVIDENCE_MODE = "EXACT_BOUNDED_BEHAVIORAL_EQUIVALENCE"
KILLED_STATUS = "NO_EXTRA_STATE_BEYOND_PAID_BRANCH_TRANSCRIPT"
SURVIVES_STATUS = "FLAT_EQUIVALENCE_FAILED_WITHIN_FROZEN_CLASS"
LIMITATIONS = (
    "The screen is fixed to p=7, 128 binary payloads, four two-bit queries, "
    "at most two predicate bits, and at most four operations.",
    "Predicates may be arbitrary explicit functions of the current seven-bit "
    "state and two-bit query, but overwrite masks and replacement values are "
    "fixed; state-dependent replacement values are outside this screen.",
    "The flat control is granted the exact runtime branch transcript and is "
    "not required to infer or predict it independently.",
    "Equal storage and work are declared padded control charges, not measured "
    "process memory, latency, energy, or hardware utilization.",
    "The result says nothing about learned predicates, generalization, noisy "
    "routing, differentiability, training cost, recall, or corpus behavior.",
    "Finite decision-tree compilation is an algebraic equivalence result, not "
    "evidence of novelty, production utility, or universal dimensionality.",
)
KILL_CRITERIA = (
    "Every one of the 512 payload-query executions equals its transcript-indexed "
    "static affine-plus-last-write-wins leaf.",
    "Every compiled static leaf equals its unreduced forced-branch program on "
    "all 128 payloads.",
    "The table contains no more than 2**b leaves for b paid predicate bits.",
    "Structured and flat controls have identical declared charges for branch "
    "bits, leaves, replacements, overlap, search, storage, and work.",
)


class RuntimeGateValidationError(ValueError):
    """Raised when an input violates the frozen p=7 runtime-gate contract."""


class RuntimeGateResourceError(RuntimeError):
    """Raised before or during enumeration when a resource limit is crossed."""


def _plain_int(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise RuntimeGateValidationError("{} must be an integer".format(name))
    result = int(value)
    if result < minimum:
        raise RuntimeGateValidationError(
            "{} must be >= {}".format(name, minimum)
        )
    return result


def _positive_seconds(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeGateValidationError(
            "{} must be a finite positive number".format(name)
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeGateValidationError(
            "{} must be a finite positive number".format(name)
        )
    return result


def _payload(value: object, name: str = "payload") -> int:
    result = _plain_int(value, name)
    if result >= PAYLOAD_COUNT:
        raise RuntimeGateValidationError(
            "{} must be a canonical {}-bit payload".format(name, MODULUS)
        )
    return result


def _query(value: object, name: str = "query") -> int:
    result = _plain_int(value, name)
    if result >= QUERY_COUNT:
        raise RuntimeGateValidationError(
            "{} must be a canonical {}-bit query".format(name, QUERY_BITS)
        )
    return result


@dataclass(frozen=True)
class RuntimeGateBudget:
    """Immutable resource budget bounded by non-overridable hard caps."""

    max_estimated_bytes: int = DEFAULT_MAX_ESTIMATED_BYTES
    max_work_units: int = DEFAULT_MAX_WORK_UNITS
    max_seconds: float = DEFAULT_MAX_SECONDS
    max_operations: int = MAX_OPERATIONS
    max_gate_bits: int = MAX_GATE_BITS

    def __post_init__(self) -> None:
        integer_limits = (
            (
                "max_estimated_bytes",
                self.max_estimated_bytes,
                HARD_MAX_ESTIMATED_BYTES,
            ),
            ("max_work_units", self.max_work_units, HARD_MAX_WORK_UNITS),
            ("max_operations", self.max_operations, MAX_OPERATIONS),
            ("max_gate_bits", self.max_gate_bits, MAX_GATE_BITS),
        )
        for name, value, hard_maximum in integer_limits:
            checked = _plain_int(value, name, minimum=1)
            if checked > hard_maximum:
                raise RuntimeGateValidationError(
                    "{} cannot exceed immutable hard cap {}".format(
                        name,
                        hard_maximum,
                    )
                )
            object.__setattr__(self, name, checked)
        seconds = _positive_seconds(self.max_seconds, "max_seconds")
        if seconds > HARD_MAX_SECONDS:
            raise RuntimeGateValidationError(
                "max_seconds cannot exceed immutable hard cap {:g}".format(
                    HARD_MAX_SECONDS
                )
            )
        object.__setattr__(self, "max_seconds", seconds)


@dataclass(frozen=True)
class RuntimePredicate:
    """An arbitrary canonical Boolean function of query and current state.

    Bit ``query * 128 + state`` in ``truth_bits`` is the predicate result.
    Thus the complete predicate is explicit, deterministic, callback-free, and
    bounded to 512 truth values.
    """

    truth_bits: int

    def __post_init__(self) -> None:
        bits = _plain_int(self.truth_bits, "truth_bits")
        if bits.bit_length() > PREDICATE_INPUT_COUNT:
            raise RuntimeGateValidationError(
                "truth_bits must encode at most {} predicate values".format(
                    PREDICATE_INPUT_COUNT
                )
            )
        object.__setattr__(self, "truth_bits", bits)

    def evaluate(self, query: int, state: int) -> bool:
        query_value = _query(query)
        state_value = _payload(state, "state")
        index = query_value * PAYLOAD_COUNT + state_value
        return bool((self.truth_bits >> index) & 1)


@dataclass(frozen=True)
class RuntimeOverwrite:
    """Apply one fixed overwrite when its runtime predicate returns true."""

    predicate: RuntimePredicate
    overwrite: OverwriteRule

    def __post_init__(self) -> None:
        if not isinstance(self.predicate, RuntimePredicate):
            raise RuntimeGateValidationError(
                "predicate must be a RuntimePredicate"
            )
        if not isinstance(self.overwrite, OverwriteRule):
            raise RuntimeGateValidationError(
                "overwrite must be an OverwriteRule"
            )


RuntimeOperation = Union[AddressOperation, RuntimeOverwrite]


@dataclass(frozen=True)
class RuntimeExecution:
    """One direct runtime execution and its paid branch transcript."""

    input_payload: int
    query: int
    output_payload: int
    branch_transcript: int
    decisions: Tuple[bool, ...]


@dataclass(frozen=True)
class StaticLeaf:
    """One transcript-selected static affine-plus-LWW normal form."""

    transcript: int
    active_gate_indices: Tuple[int, ...]
    normal_form: ReplacementNormalForm

    def __post_init__(self) -> None:
        transcript = _plain_int(self.transcript, "transcript")
        if transcript >= 1 << MAX_GATE_BITS:
            raise RuntimeGateValidationError(
                "leaf transcript exceeds the immutable predicate-bit cap"
            )
        if isinstance(self.active_gate_indices, (str, bytes)):
            raise RuntimeGateValidationError(
                "active_gate_indices must be a sequence of gate indices"
            )
        try:
            active = tuple(
                _plain_int(value, "active_gate_indices", minimum=0)
                for value in self.active_gate_indices
            )
        except TypeError as error:
            raise RuntimeGateValidationError(
                "active_gate_indices must be a sequence of gate indices"
            ) from error
        if (
            tuple(sorted(set(active))) != active
            or any(index >= MAX_GATE_BITS for index in active)
        ):
            raise RuntimeGateValidationError(
                "active_gate_indices must be sorted unique bounded indices"
            )
        expected = tuple(
            index
            for index in range(MAX_GATE_BITS)
            if (transcript >> index) & 1
        )
        if active != expected:
            raise RuntimeGateValidationError(
                "active_gate_indices must match the leaf transcript"
            )
        if not isinstance(self.normal_form, ReplacementNormalForm):
            raise RuntimeGateValidationError(
                "normal_form must be a ReplacementNormalForm"
            )
        object.__setattr__(self, "transcript", transcript)
        object.__setattr__(self, "active_gate_indices", active)


@dataclass(frozen=True)
class FlatDecisionTable:
    """Complete static leaf table indexed by a paid branch transcript."""

    gate_bits: int
    leaves: Tuple[StaticLeaf, ...]

    def __post_init__(self) -> None:
        gate_bits = _plain_int(self.gate_bits, "gate_bits", minimum=1)
        if gate_bits > MAX_GATE_BITS:
            raise RuntimeGateValidationError(
                "gate_bits cannot exceed {}".format(MAX_GATE_BITS)
            )
        expected = 1 << gate_bits
        if isinstance(self.leaves, (str, bytes)):
            raise RuntimeGateValidationError(
                "leaves must be a sequence of StaticLeaf values"
            )
        try:
            leaves = tuple(self.leaves)
        except TypeError as error:
            raise RuntimeGateValidationError(
                "leaves must be a sequence of StaticLeaf values"
            ) from error
        if len(leaves) != expected:
            raise RuntimeGateValidationError(
                "decision table must contain exactly {} leaves".format(
                    expected
                )
            )
        for transcript, leaf in enumerate(leaves):
            if not isinstance(leaf, StaticLeaf):
                raise RuntimeGateValidationError(
                    "leaves[{}] must be a StaticLeaf".format(transcript)
                )
            if leaf.transcript != transcript:
                raise RuntimeGateValidationError(
                    "leaf transcripts must be canonical and ordered"
                )
        object.__setattr__(self, "gate_bits", gate_bits)
        object.__setattr__(self, "leaves", leaves)


@dataclass(frozen=True)
class ControlCharge:
    """One side of the deliberately equalized structured/flat control."""

    name: str
    branch_bits: int
    leaf_capacity: int
    compiled_leaves: int
    replacement_templates: int
    maximum_replacement_applications: int
    replacement_mask_sizes: Tuple[int, ...]
    replacement_pair_overlap: int
    leaf_searches_per_case: int
    charged_storage_bytes: int
    charged_work_units_per_case: int
    measured_storage_or_work: bool
    common_envelope_padding_allowed: bool

    @property
    def matching_signature(self) -> Tuple[object, ...]:
        return (
            self.branch_bits,
            self.leaf_capacity,
            self.compiled_leaves,
            self.replacement_templates,
            self.maximum_replacement_applications,
            self.replacement_mask_sizes,
            self.replacement_pair_overlap,
            self.leaf_searches_per_case,
            self.charged_storage_bytes,
            self.charged_work_units_per_case,
            self.measured_storage_or_work,
            self.common_envelope_padding_allowed,
        )


@dataclass(frozen=True)
class ResourceEstimate:
    """Pre-enumeration modeled upper bounds for the complete exact screen."""

    operation_count: int
    gate_bits: int
    leaf_capacity: int
    payload_count: int
    query_count: int
    runtime_case_count: int
    static_leaf_case_count: int
    estimated_peak_bytes: int
    estimated_work_units: int
    component_bytes: Tuple[Tuple[str, int], ...]
    component_work: Tuple[Tuple[str, int], ...]
    max_estimated_bytes: int
    max_work_units: int
    max_seconds: float
    max_operations: int
    max_gate_bits: int
    cases_streamed_not_materialized: bool
    measured_process_peak: bool


@dataclass(frozen=True)
class RuntimeGateAnalysis:
    """Complete exact bounded result for one accepted runtime program."""

    protocol_id: str
    evidence_mode: str
    modulus: int
    payload_count: int
    query_count: int
    operation_count: int
    gate_bits: int
    decision_table: FlatDecisionTable
    reachable_transcripts: Tuple[int, ...]
    exhaustive_runtime_cases: int
    exhaustive_static_leaf_cases: int
    all_four_queries_tested: bool
    flat_equivalence_holds: bool
    static_leaf_normal_forms_hold: bool
    leaf_bound_holds: bool
    structured_control: ControlCharge
    flat_control: ControlCharge
    controls_matched: bool
    branch_transcript_granted_to_flat_control: bool
    kill_criteria: Tuple[str, ...]
    kill_criteria_met: bool
    behavioral_state_gain_supported: bool
    hypothesis_status: str
    resource_guard: ResourceEstimate
    promotion_eligible: bool
    novelty_claim: bool
    ml_utility_claim: bool
    runtime_benefit_claim: bool
    limitations: Tuple[str, ...]


def affine_parity_predicate(
    *,
    query_mask: int = 0,
    state_mask: int = 0,
    constant: int = 0,
) -> RuntimePredicate:
    """Build a bounded linear-polarity predicate truth table.

    This helper is convenient for experiments; :class:`RuntimePredicate`
    itself accepts any explicit Boolean truth table over the frozen domain.
    """

    query_bits = _plain_int(query_mask, "query_mask")
    if query_bits >= QUERY_COUNT:
        raise RuntimeGateValidationError(
            "query_mask must fit in {} bits".format(QUERY_BITS)
        )
    state_bits = _payload(state_mask, "state_mask")
    constant_bit = _plain_int(constant, "constant")
    if constant_bit not in (0, 1):
        raise RuntimeGateValidationError("constant must be 0 or 1")
    if query_bits == 0 and state_bits == 0:
        raise RuntimeGateValidationError(
            "at least one query or state mask bit must be selected"
        )

    truth_bits = 0
    for query in range(QUERY_COUNT):
        for state in range(PAYLOAD_COUNT):
            value = (
                constant_bit
                ^ ((query & query_bits).bit_count() & 1)
                ^ ((state & state_bits).bit_count() & 1)
            )
            if value:
                index = query * PAYLOAD_COUNT + state
                truth_bits |= 1 << index
    return RuntimePredicate(truth_bits)


def runtime_overwrite(
    predicate: RuntimePredicate,
    overwrite: OverwriteRule,
) -> RuntimeOverwrite:
    """Construct one validated runtime-gated overwrite."""

    return RuntimeOverwrite(predicate, overwrite)


def _normalize_program(
    operations: Sequence[RuntimeOperation],
    budget: RuntimeGateBudget,
) -> Tuple[RuntimeOperation, ...]:
    if isinstance(operations, (str, bytes)) or not isinstance(
        operations,
        Sequence,
    ):
        raise RuntimeGateValidationError("operations must be a sequence")
    operation_count = len(operations)
    if operation_count < 1:
        raise RuntimeGateValidationError("operations must not be empty")
    if operation_count > budget.max_operations:
        raise RuntimeGateResourceError(
            "operation count exceeds budget: {} > {}".format(
                operation_count,
                budget.max_operations,
            )
        )
    normalized = tuple(
        operations[index] for index in range(operation_count)
    )
    gate_count = 0
    for index, operation in enumerate(normalized):
        if isinstance(operation, RuntimeOverwrite):
            gate_count += 1
        elif not isinstance(operation, AddressOperation):
            raise RuntimeGateValidationError(
                "operations[{}] has an unsupported type".format(index)
            )
    if gate_count < 1:
        raise RuntimeGateValidationError(
            "the runtime screen requires at least one predicate bit"
        )
    if gate_count > budget.max_gate_bits:
        raise RuntimeGateResourceError(
            "predicate-bit count exceeds budget: {} > {}".format(
                gate_count,
                budget.max_gate_bits,
            )
        )
    return normalized


def _static_normal_form(
    operations: Sequence[Union[AddressOperation, OverwriteRule]],
) -> ReplacementNormalForm:
    affine = AffineMap(1, 0)
    overwrite = None
    for operation in operations:
        if isinstance(operation, AddressOperation):
            mapping = operation.affine
            affine = compose_affine(mapping, affine)
            if overwrite is not None:
                overwrite = relabel_overwrite(overwrite, mapping)
        elif isinstance(operation, OverwriteRule):
            previous = () if overwrite is None else (overwrite,)
            overwrite = compose_overwrites(previous + (operation,))
        else:
            raise RuntimeGateValidationError(
                "static leaf contains an unsupported operation"
            )
    if overwrite is None:
        return ReplacementNormalForm(affine, 0, 0)
    return ReplacementNormalForm(
        affine,
        overwrite.mask,
        overwrite.values,
    )


def _static_program_for_transcript(
    operations: Tuple[RuntimeOperation, ...],
    transcript: int,
) -> Tuple[Union[AddressOperation, OverwriteRule], ...]:
    static_operations = []
    gate_index = 0
    for operation in operations:
        if isinstance(operation, AddressOperation):
            static_operations.append(operation)
        else:
            if (transcript >> gate_index) & 1:
                static_operations.append(operation.overwrite)
            gate_index += 1
    return tuple(static_operations)


def _effective_overwrite_templates(
    operations: Tuple[RuntimeOperation, ...],
) -> Tuple[OverwriteRule, ...]:
    """Move every gated overwrite through later address maps.

    This expresses mask sizes and overlap in the common final coordinate frame
    where last-write-wins interactions actually occur.
    """

    suffix_affine = AffineMap(1, 0)
    reversed_overwrites = []
    for operation in reversed(operations):
        if isinstance(operation, AddressOperation):
            suffix_affine = compose_affine(
                suffix_affine,
                operation.affine,
            )
        else:
            reversed_overwrites.append(
                relabel_overwrite(operation.overwrite, suffix_affine)
            )
    return tuple(reversed(reversed_overwrites))


def compile_flat_decision_table(
    operations: Sequence[RuntimeOperation],
    *,
    budget: RuntimeGateBudget = RuntimeGateBudget(),
) -> FlatDecisionTable:
    """Compile every possible branch transcript to one static normal form."""

    if not isinstance(budget, RuntimeGateBudget):
        raise RuntimeGateValidationError(
            "budget must be a RuntimeGateBudget"
        )
    normalized = _normalize_program(operations, budget)
    gate_count = sum(
        isinstance(operation, RuntimeOverwrite)
        for operation in normalized
    )
    leaves = []
    for transcript in range(1 << gate_count):
        static_program = _static_program_for_transcript(
            normalized,
            transcript,
        )
        active = tuple(
            index
            for index in range(gate_count)
            if (transcript >> index) & 1
        )
        leaves.append(
            StaticLeaf(
                transcript=transcript,
                active_gate_indices=active,
                normal_form=_static_normal_form(static_program),
            )
        )
    return FlatDecisionTable(gate_count, tuple(leaves))


def _execute_normalized(
    payload: int,
    query: int,
    operations: Tuple[RuntimeOperation, ...],
) -> RuntimeExecution:
    input_payload = _payload(payload)
    query_value = _query(query)
    state = input_payload
    transcript = 0
    decisions = []
    gate_index = 0
    for operation in operations:
        if isinstance(operation, AddressOperation):
            state = relabel_payload(state, operation.affine)
        else:
            decision = operation.predicate.evaluate(query_value, state)
            decisions.append(decision)
            if decision:
                transcript |= 1 << gate_index
                state = apply_overwrite(state, operation.overwrite)
            gate_index += 1
    return RuntimeExecution(
        input_payload=input_payload,
        query=query_value,
        output_payload=state,
        branch_transcript=transcript,
        decisions=tuple(decisions),
    )


def execute_runtime_program(
    payload: int,
    query: int,
    operations: Sequence[RuntimeOperation],
    *,
    budget: RuntimeGateBudget = RuntimeGateBudget(),
) -> RuntimeExecution:
    """Execute one validated runtime program and expose its paid branch bits."""

    if not isinstance(budget, RuntimeGateBudget):
        raise RuntimeGateValidationError(
            "budget must be a RuntimeGateBudget"
        )
    normalized = _normalize_program(operations, budget)
    return _execute_normalized(payload, query, normalized)


def execute_static_branch(
    payload: int,
    operations: Sequence[RuntimeOperation],
    transcript: int,
    *,
    budget: RuntimeGateBudget = RuntimeGateBudget(),
) -> int:
    """Execute the unreduced static program selected by one transcript."""

    if not isinstance(budget, RuntimeGateBudget):
        raise RuntimeGateValidationError(
            "budget must be a RuntimeGateBudget"
        )
    normalized = _normalize_program(operations, budget)
    gate_count = sum(
        isinstance(operation, RuntimeOverwrite)
        for operation in normalized
    )
    transcript_value = _plain_int(transcript, "transcript")
    if transcript_value >= 1 << gate_count:
        raise RuntimeGateValidationError(
            "transcript does not fit the program predicate-bit count"
        )
    state = _payload(payload)
    for operation in _static_program_for_transcript(
        normalized,
        transcript_value,
    ):
        if isinstance(operation, AddressOperation):
            state = relabel_payload(state, operation.affine)
        else:
            state = apply_overwrite(state, operation)
    return state


def apply_flat_decision_table(
    payload: int,
    transcript: int,
    table: FlatDecisionTable,
) -> int:
    """Apply the static leaf selected by an already-paid branch transcript."""

    if not isinstance(table, FlatDecisionTable):
        raise RuntimeGateValidationError(
            "table must be a FlatDecisionTable"
        )
    transcript_value = _plain_int(transcript, "transcript")
    if transcript_value >= len(table.leaves):
        raise RuntimeGateValidationError(
            "transcript does not select a decision-table leaf"
        )
    return apply_normal_form(
        _payload(payload),
        table.leaves[transcript_value].normal_form,
    )


def _iter_runtime_cases() -> Iterable[Tuple[int, int]]:
    for payload in range(PAYLOAD_COUNT):
        for query in range(QUERY_COUNT):
            yield payload, query


def _check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise RuntimeGateResourceError(
            "runtime-gate screen exceeded its wall-clock deadline"
        )


def _resource_estimate(
    operation_count: int,
    gate_count: int,
    budget: RuntimeGateBudget,
) -> ResourceEstimate:
    leaf_capacity = 1 << gate_count
    runtime_cases = PAYLOAD_COUNT * QUERY_COUNT
    static_leaf_cases = PAYLOAD_COUNT * leaf_capacity
    component_bytes = (
        ("normalized_program", operation_count * 4_096),
        ("explicit_predicate_tables", gate_count * 4_096),
        ("flat_static_leaves", leaf_capacity * 8_192),
        ("streaming_execution_scratch", 16_384),
        ("result_and_control_records", 32_768),
    )
    component_work = (
        (
            "compile_all_static_leaves",
            leaf_capacity * operation_count * 32,
        ),
        (
            "validate_static_leaf_normal_forms",
            static_leaf_cases * (operation_count * 16 + 16),
        ),
        (
            "runtime_flat_equivalence",
            runtime_cases
            * (operation_count * 32 + gate_count * 16 + 32),
        ),
        ("matched_control_accounting", runtime_cases * 8),
    )
    estimated_bytes = sum(value for _name, value in component_bytes)
    estimated_work = sum(value for _name, value in component_work)
    if estimated_bytes > budget.max_estimated_bytes:
        raise RuntimeGateResourceError(
            "estimated bytes exceed budget: {} > {}".format(
                estimated_bytes,
                budget.max_estimated_bytes,
            )
        )
    if estimated_work > budget.max_work_units:
        raise RuntimeGateResourceError(
            "estimated work exceeds budget: {} > {}".format(
                estimated_work,
                budget.max_work_units,
            )
        )
    return ResourceEstimate(
        operation_count=operation_count,
        gate_bits=gate_count,
        leaf_capacity=leaf_capacity,
        payload_count=PAYLOAD_COUNT,
        query_count=QUERY_COUNT,
        runtime_case_count=runtime_cases,
        static_leaf_case_count=static_leaf_cases,
        estimated_peak_bytes=estimated_bytes,
        estimated_work_units=estimated_work,
        component_bytes=component_bytes,
        component_work=component_work,
        max_estimated_bytes=budget.max_estimated_bytes,
        max_work_units=budget.max_work_units,
        max_seconds=budget.max_seconds,
        max_operations=budget.max_operations,
        max_gate_bits=budget.max_gate_bits,
        cases_streamed_not_materialized=True,
        measured_process_peak=False,
    )


def _control_charge(
    *,
    name: str,
    operations: Tuple[RuntimeOperation, ...],
    table: FlatDecisionTable,
    resource: ResourceEstimate,
) -> ControlCharge:
    overwrites = _effective_overwrite_templates(operations)
    overlap = 0
    if len(overwrites) == 2:
        overlap = (overwrites[0].mask & overwrites[1].mask).bit_count()
    common_storage = (
        dict(resource.component_bytes)["normalized_program"]
        + dict(resource.component_bytes)["explicit_predicate_tables"]
        + dict(resource.component_bytes)["flat_static_leaves"]
    )
    common_work = (
        len(operations) * 32
        + table.gate_bits * 16
        + table.gate_bits
        + 32
    )
    return ControlCharge(
        name=name,
        branch_bits=table.gate_bits,
        leaf_capacity=1 << table.gate_bits,
        compiled_leaves=len(table.leaves),
        replacement_templates=len(overwrites),
        maximum_replacement_applications=len(overwrites),
        replacement_mask_sizes=tuple(
            overwrite.mask.bit_count() for overwrite in overwrites
        ),
        replacement_pair_overlap=overlap,
        leaf_searches_per_case=1,
        charged_storage_bytes=common_storage,
        charged_work_units_per_case=common_work,
        measured_storage_or_work=False,
        common_envelope_padding_allowed=True,
    )


def analyze_runtime_gates(
    operations: Sequence[RuntimeOperation],
    *,
    budget: RuntimeGateBudget = RuntimeGateBudget(),
) -> RuntimeGateAnalysis:
    """Run the exact bounded runtime-gate versus flat-table screen."""

    if not isinstance(budget, RuntimeGateBudget):
        raise RuntimeGateValidationError(
            "budget must be a RuntimeGateBudget"
        )
    started = time.monotonic()
    deadline = started + budget.max_seconds
    normalized = _normalize_program(operations, budget)
    gate_count = sum(
        isinstance(operation, RuntimeOverwrite)
        for operation in normalized
    )
    resource = _resource_estimate(
        len(normalized),
        gate_count,
        budget,
    )
    _check_deadline(deadline)

    table = compile_flat_decision_table(normalized, budget=budget)
    _check_deadline(deadline)

    static_leaf_holds = True
    static_leaf_cases = 0
    for leaf in table.leaves:
        for payload in range(PAYLOAD_COUNT):
            _check_deadline(deadline)
            direct_static = execute_static_branch(
                payload,
                normalized,
                leaf.transcript,
                budget=budget,
            )
            reduced_static = apply_flat_decision_table(
                payload,
                leaf.transcript,
                table,
            )
            static_leaf_cases += 1
            if direct_static != reduced_static:
                static_leaf_holds = False

    reachable = set()
    seen_queries = set()
    flat_holds = True
    runtime_cases = 0
    for payload, query in _iter_runtime_cases():
        _check_deadline(deadline)
        direct = _execute_normalized(payload, query, normalized)
        flat = apply_flat_decision_table(
            payload,
            direct.branch_transcript,
            table,
        )
        runtime_cases += 1
        reachable.add(direct.branch_transcript)
        seen_queries.add(query)
        if direct.output_payload != flat:
            flat_holds = False

    structured_control = _control_charge(
        name="runtime_structured_program",
        operations=normalized,
        table=table,
        resource=resource,
    )
    flat_control = _control_charge(
        name="paid_transcript_flat_table",
        operations=normalized,
        table=table,
        resource=resource,
    )
    controls_matched = (
        structured_control.matching_signature
        == flat_control.matching_signature
    )
    leaf_bound_holds = len(table.leaves) <= 1 << gate_count
    all_queries = seen_queries == set(range(QUERY_COUNT))
    kill_criteria_met = (
        runtime_cases == resource.runtime_case_count
        and static_leaf_cases == resource.static_leaf_case_count
        and all_queries
        and flat_holds
        and static_leaf_holds
        and leaf_bound_holds
        and controls_matched
    )
    _check_deadline(deadline)

    return RuntimeGateAnalysis(
        protocol_id=PROTOCOL_ID,
        evidence_mode=EVIDENCE_MODE,
        modulus=MODULUS,
        payload_count=PAYLOAD_COUNT,
        query_count=QUERY_COUNT,
        operation_count=len(normalized),
        gate_bits=gate_count,
        decision_table=table,
        reachable_transcripts=tuple(sorted(reachable)),
        exhaustive_runtime_cases=runtime_cases,
        exhaustive_static_leaf_cases=static_leaf_cases,
        all_four_queries_tested=all_queries,
        flat_equivalence_holds=flat_holds,
        static_leaf_normal_forms_hold=static_leaf_holds,
        leaf_bound_holds=leaf_bound_holds,
        structured_control=structured_control,
        flat_control=flat_control,
        controls_matched=controls_matched,
        branch_transcript_granted_to_flat_control=True,
        kill_criteria=KILL_CRITERIA,
        kill_criteria_met=kill_criteria_met,
        behavioral_state_gain_supported=not kill_criteria_met,
        hypothesis_status=(
            KILLED_STATUS if kill_criteria_met else SURVIVES_STATUS
        ),
        resource_guard=resource,
        promotion_eligible=False,
        novelty_claim=False,
        ml_utility_claim=False,
        runtime_benefit_claim=False,
        limitations=LIMITATIONS,
    )


__all__ = [
    "ControlCharge",
    "DEFAULT_MAX_ESTIMATED_BYTES",
    "DEFAULT_MAX_SECONDS",
    "DEFAULT_MAX_WORK_UNITS",
    "EVIDENCE_MODE",
    "FlatDecisionTable",
    "HARD_MAX_ESTIMATED_BYTES",
    "HARD_MAX_SECONDS",
    "HARD_MAX_WORK_UNITS",
    "KILL_CRITERIA",
    "KILLED_STATUS",
    "LIMITATIONS",
    "MAX_GATE_BITS",
    "MAX_OPERATIONS",
    "MODULUS",
    "PAYLOAD_COUNT",
    "PREDICATE_INPUT_COUNT",
    "PROTOCOL_ID",
    "QUERY_BITS",
    "QUERY_COUNT",
    "ResourceEstimate",
    "RuntimeExecution",
    "RuntimeGateAnalysis",
    "RuntimeGateBudget",
    "RuntimeGateResourceError",
    "RuntimeGateValidationError",
    "RuntimeOverwrite",
    "RuntimePredicate",
    "SURVIVES_STATUS",
    "StaticLeaf",
    "affine_parity_predicate",
    "analyze_runtime_gates",
    "apply_flat_decision_table",
    "compile_flat_decision_table",
    "execute_runtime_program",
    "execute_static_branch",
    "runtime_overwrite",
]
