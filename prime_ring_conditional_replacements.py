"""Bounded conditional-replacement normal-form screen over ``Z_7``.

This module tests exact finite algebra, not retrieval quality.  A program of at
most four operations may interleave:

* rotations ``x -> x + b``;
* nonzero multiplicative pivots ``x -> a*x``; and
* conditional overwrites on declared address masks.

Every accepted program is reduced to one affine address permutation followed
by one last-write-wins overwrite.  The implementation streams all 128 binary
payloads to falsify that normal form and matched conjugacy controls.  It makes
no novelty, ML-utility, latency, compression, or production claim.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union


PROTOCOL_ID = "PRW-CR-NORMAL-FORM-P7-001"
MODULUS = 7
PAYLOAD_COUNT = 1 << MODULUS
MAX_OPERATIONS = 4
PRIMITIVE_ROOTS = (3, 5)

HARD_MAX_ESTIMATED_BYTES = 1_048_576
HARD_MAX_WORK_UNITS = 500_000
HARD_MAX_SECONDS = 5.0
DEFAULT_MAX_ESTIMATED_BYTES = 262_144
DEFAULT_MAX_WORK_UNITS = 100_000
DEFAULT_MAX_SECONDS = 1.0

EVIDENCE_MODE = "EXACT_BOUNDED_ALGEBRAIC_FALSIFICATION"
HYPOTHESIS_STATUS = "NON_PROMOTIONAL_NORMAL_FORM_SCREEN"
LIMITATIONS = (
    "The screen is fixed to p=7, binary payloads, two overwrite rules, and at "
    "most four total operations.",
    "Each condition is a fixed address mask with constant replacement bits; "
    "payload-, query-, or layer-state-dependent predicates and replacement "
    "functions are outside this screen.",
    "The random matched control is a seeded address permutation; it is a "
    "structural conjugacy control, not an independent statistical sample.",
    "Alternative primitive-root checks are address-label invariance checks, "
    "not evidence that either labeling improves search.",
    "The byte and work figures are declared conservative estimates; they are "
    "not measured process memory, latency, or hardware utilization.",
    "Exact finite identities do not establish novelty, retrieval utility, "
    "training savings, recall gains, compression, or production readiness.",
)


class ConditionalReplacementValidationError(ValueError):
    """Raised when an input violates the frozen p=7 screen contract."""


class ConditionalReplacementResourceError(RuntimeError):
    """Raised before or during enumeration when a resource limit is crossed."""


def _plain_int(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ConditionalReplacementValidationError(
            "{} must be an integer".format(name)
        )
    result = int(value)
    if result < minimum:
        raise ConditionalReplacementValidationError(
            "{} must be >= {}".format(name, minimum)
        )
    return result


def _population_count(value: object) -> int:
    """Count set bits in a nonnegative integer using Python 3.9 stdlib."""

    return bin(_plain_int(value, "population_count_value")).count("1")


def _canonical_residue(value: object, name: str, *, nonzero: bool = False) -> int:
    result = _plain_int(value, name)
    if result >= MODULUS:
        raise ConditionalReplacementValidationError(
            "{} must be a canonical residue in [0, {})".format(name, MODULUS)
        )
    if nonzero and result == 0:
        raise ConditionalReplacementValidationError(
            "{} must be nonzero modulo {}".format(name, MODULUS)
        )
    return result


def _positive_seconds(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConditionalReplacementValidationError(
            "{} must be a finite positive number".format(name)
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ConditionalReplacementValidationError(
            "{} must be a finite positive number".format(name)
        )
    return result


def _payload(value: object, name: str = "payload") -> int:
    result = _plain_int(value, name)
    if result >= PAYLOAD_COUNT:
        raise ConditionalReplacementValidationError(
            "{} must be a canonical {}-bit payload".format(name, MODULUS)
        )
    return result


@dataclass(frozen=True)
class ConditionalReplacementBudget:
    """Immutable small-state budget bounded by hard caps."""

    max_estimated_bytes: int = DEFAULT_MAX_ESTIMATED_BYTES
    max_work_units: int = DEFAULT_MAX_WORK_UNITS
    max_seconds: float = DEFAULT_MAX_SECONDS
    max_operations: int = MAX_OPERATIONS

    def __post_init__(self) -> None:
        integer_limits = (
            (
                "max_estimated_bytes",
                self.max_estimated_bytes,
                HARD_MAX_ESTIMATED_BYTES,
            ),
            ("max_work_units", self.max_work_units, HARD_MAX_WORK_UNITS),
            ("max_operations", self.max_operations, MAX_OPERATIONS),
        )
        for name, value, hard_maximum in integer_limits:
            checked = _plain_int(value, name, minimum=1)
            if checked > hard_maximum:
                raise ConditionalReplacementValidationError(
                    "{} cannot exceed immutable hard cap {}".format(
                        name, hard_maximum
                    )
                )
            object.__setattr__(self, name, checked)
        seconds = _positive_seconds(self.max_seconds, "max_seconds")
        if seconds > HARD_MAX_SECONDS:
            raise ConditionalReplacementValidationError(
                "max_seconds cannot exceed immutable hard cap {:g}".format(
                    HARD_MAX_SECONDS
                )
            )
        object.__setattr__(self, "max_seconds", seconds)


@dataclass(frozen=True)
class AffineMap:
    """One invertible affine address map ``x -> multiplier*x + offset``."""

    multiplier: int
    offset: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "multiplier",
            _canonical_residue(
                self.multiplier,
                "multiplier",
                nonzero=True,
            ),
        )
        object.__setattr__(
            self,
            "offset",
            _canonical_residue(self.offset, "offset"),
        )


IDENTITY_AFFINE = AffineMap(1, 0)


@dataclass(frozen=True)
class AddressOperation:
    """A rotation or nonzero multiplicative pivot on ring addresses."""

    kind: str
    value: int

    def __post_init__(self) -> None:
        if self.kind not in ("rotation", "pivot"):
            raise ConditionalReplacementValidationError(
                "address operation kind must be rotation or pivot"
            )
        canonical = _canonical_residue(
            self.value,
            "value",
            nonzero=self.kind == "pivot",
        )
        object.__setattr__(self, "value", canonical)

    @property
    def affine(self) -> AffineMap:
        if self.kind == "rotation":
            return AffineMap(1, self.value)
        return AffineMap(self.value, 0)


@dataclass(frozen=True)
class OverwriteRule:
    """Overwrite ``values`` at every address selected by ``mask``."""

    mask: int
    values: int

    def __post_init__(self) -> None:
        mask = _payload(self.mask, "mask")
        values = _payload(self.values, "values")
        if mask == 0:
            raise ConditionalReplacementValidationError(
                "overwrite mask must select at least one address"
            )
        if values & ~mask:
            raise ConditionalReplacementValidationError(
                "overwrite values must be zero outside the mask"
            )
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "values", values)


ProgramOperation = Union[AddressOperation, OverwriteRule]


@dataclass(frozen=True)
class ReplacementNormalForm:
    """One affine relabeling followed by one optional overwrite."""

    affine: AffineMap
    overwrite_mask: int
    overwrite_values: int

    def __post_init__(self) -> None:
        mask = _payload(self.overwrite_mask, "overwrite_mask")
        values = _payload(self.overwrite_values, "overwrite_values")
        if values & ~mask:
            raise ConditionalReplacementValidationError(
                "normal-form values must be zero outside the overwrite mask"
            )
        object.__setattr__(self, "overwrite_mask", mask)
        object.__setattr__(self, "overwrite_values", values)

    @property
    def overwrite(self) -> Optional[OverwriteRule]:
        if self.overwrite_mask == 0:
            return None
        return OverwriteRule(self.overwrite_mask, self.overwrite_values)


@dataclass(frozen=True)
class SectorProfile:
    """Permutation-invariant size and overlap profile for two masks."""

    mask_sizes: Tuple[int, int]
    overlap_size: int
    union_size: int


@dataclass(frozen=True)
class ControlResult:
    """One size/overlap-matched overwrite-pair conjugacy control."""

    name: str
    permutation: Tuple[int, ...]
    masks: Tuple[int, int]
    profile: SectorProfile
    profile_matches_original: bool
    overwrite_sequence_conjugacy_holds: bool
    last_write_wins_holds: bool
    full_program_invariance_checked: bool
    full_program_invariance_holds: Optional[bool]


@dataclass(frozen=True)
class PrimitiveRootResult:
    """Full-program address-label invariance under one p=7 primitive root."""

    root: int
    is_primitive_root: bool
    profile_matches_original: bool
    overwrite_sequence_conjugacy_holds: bool
    last_write_wins_holds: bool
    full_program_invariance_holds: bool
    conjugated_affine: AffineMap


@dataclass(frozen=True)
class ResourceEstimate:
    """Declared conservative estimate for streamed exact enumeration."""

    operation_count: int
    address_operation_count: int
    overwrite_operation_count: int
    exhaustive_payloads_per_pass: int
    exhaustive_payload_passes_upper_bound: int
    exhaustive_payload_cases_upper_bound: int
    primitive_root_controls: int
    estimated_peak_bytes: int
    estimated_work_units: int
    component_bytes: Tuple[Tuple[str, int], ...]
    max_estimated_bytes: int
    max_work_units: int
    max_seconds: float
    max_operations: int
    payloads_streamed_not_materialized: bool
    caller_owned_inputs_included: bool
    measured_process_peak: bool


@dataclass(frozen=True)
class ConditionalReplacementAnalysis:
    """Complete non-promotional result from the frozen p=7 screen."""

    protocol_id: str
    evidence_mode: str
    modulus: int
    operation_count: int
    address_operation_count: int
    overwrite_operation_count: int
    affine_normal_form: AffineMap
    replacement_normal_form: ReplacementNormalForm
    address_normal_form_holds: bool
    program_normal_form_holds: bool
    pair_profile: SectorProfile
    pair_is_disjoint: bool
    pair_commutes_on_all_payloads: bool
    disjoint_commutation_applicable: bool
    disjoint_commutation_holds: Optional[bool]
    overlapping_last_write_wins_applicable: bool
    last_write_wins_holds: bool
    shift_conjugacy_holds: bool
    shift_control: ControlResult
    random_matched_control: ControlResult
    primitive_root_controls: Tuple[PrimitiveRootResult, ...]
    resource_guard: ResourceEstimate
    hypothesis_status: str
    promotion_eligible: bool
    novelty_claim: bool
    ml_utility_claim: bool
    runtime_benefit_claim: bool
    limitations: Tuple[str, ...]


def rotation(shift: int) -> AddressOperation:
    """Construct a canonical additive rotation."""

    return AddressOperation("rotation", shift)


def pivot(multiplier: int) -> AddressOperation:
    """Construct a canonical nonzero multiplicative pivot."""

    return AddressOperation("pivot", multiplier)


def sector_overwrite(
    start: int,
    bits: Sequence[int],
) -> OverwriteRule:
    """Construct one cyclic contiguous-sector overwrite.

    ``bits[i]`` is written to address ``start + i (mod 7)``.  Sector lengths
    from one through seven are accepted.
    """

    origin = _canonical_residue(start, "start")
    if isinstance(bits, (str, bytes)) or not isinstance(bits, Sequence):
        raise ConditionalReplacementValidationError(
            "bits must be a sequence of binary integers"
        )
    if not 1 <= len(bits) <= MODULUS:
        raise ConditionalReplacementValidationError(
            "bits must contain between 1 and {} values".format(MODULUS)
        )
    mask = 0
    values = 0
    bit_count = len(bits)
    for index in range(bit_count):
        bit = bits[index]
        checked = _plain_int(bit, "bits[{}]".format(index))
        if checked not in (0, 1):
            raise ConditionalReplacementValidationError(
                "bits[{}] must be 0 or 1".format(index)
            )
        address = (origin + index) % MODULUS
        mask |= 1 << address
        values |= checked << address
    return OverwriteRule(mask, values)


def compose_affine(outer: AffineMap, inner: AffineMap) -> AffineMap:
    """Return ``outer(inner(x))``."""

    if not isinstance(outer, AffineMap) or not isinstance(inner, AffineMap):
        raise ConditionalReplacementValidationError(
            "compose_affine requires two AffineMap values"
        )
    return AffineMap(
        (outer.multiplier * inner.multiplier) % MODULUS,
        (outer.multiplier * inner.offset + outer.offset) % MODULUS,
    )


def inverse_affine(mapping: AffineMap) -> AffineMap:
    """Return the exact inverse of an invertible p=7 affine map."""

    if not isinstance(mapping, AffineMap):
        raise ConditionalReplacementValidationError(
            "inverse_affine requires an AffineMap"
        )
    inverse_multiplier = pow(mapping.multiplier, -1, MODULUS)
    return AffineMap(
        inverse_multiplier,
        (-inverse_multiplier * mapping.offset) % MODULUS,
    )


def conjugate_affine(mapping: AffineMap, labels: AffineMap) -> AffineMap:
    """Return ``labels o mapping o labels**-1``."""

    return compose_affine(
        labels,
        compose_affine(mapping, inverse_affine(labels)),
    )


def apply_affine(address: int, mapping: AffineMap) -> int:
    """Apply one affine map to a canonical ring address."""

    coordinate = _canonical_residue(address, "address")
    if not isinstance(mapping, AffineMap):
        raise ConditionalReplacementValidationError(
            "mapping must be an AffineMap"
        )
    return (
        mapping.multiplier * coordinate + mapping.offset
    ) % MODULUS


def affine_normal_form(
    operations: Sequence[AddressOperation],
) -> AffineMap:
    """Collapse rotations and pivots, in execution order, to one affine map."""

    if isinstance(operations, (str, bytes)) or not isinstance(
        operations, Sequence
    ):
        raise ConditionalReplacementValidationError(
            "operations must be a sequence"
        )
    if len(operations) > MAX_OPERATIONS:
        raise ConditionalReplacementResourceError(
            "address operation count exceeds immutable cap {}".format(
                MAX_OPERATIONS
            )
        )
    operation_count = len(operations)
    result = IDENTITY_AFFINE
    for index in range(operation_count):
        operation = operations[index]
        if not isinstance(operation, AddressOperation):
            raise ConditionalReplacementValidationError(
                "operations[{}] must be an AddressOperation".format(index)
            )
        result = compose_affine(operation.affine, result)
    return result


def _relabel_by_permutation(
    value: int,
    permutation: Tuple[int, ...],
) -> int:
    result = 0
    for old_address, new_address in enumerate(permutation):
        if value & (1 << old_address):
            result |= 1 << new_address
    return result


def _affine_permutation(mapping: AffineMap) -> Tuple[int, ...]:
    return tuple(apply_affine(address, mapping) for address in range(MODULUS))


def _inverse_permutation(permutation: Tuple[int, ...]) -> Tuple[int, ...]:
    inverse = [0] * MODULUS
    for old_address, new_address in enumerate(permutation):
        inverse[new_address] = old_address
    return tuple(inverse)


def relabel_payload(payload: int, mapping: AffineMap) -> int:
    """Move every payload bit along the declared affine address map."""

    normalized = _payload(payload)
    return _relabel_by_permutation(
        normalized,
        _affine_permutation(mapping),
    )


def relabel_overwrite(
    rule: OverwriteRule,
    mapping: AffineMap,
) -> OverwriteRule:
    """Conjugate one overwrite rule by an affine address relabeling."""

    if not isinstance(rule, OverwriteRule):
        raise ConditionalReplacementValidationError(
            "rule must be an OverwriteRule"
        )
    permutation = _affine_permutation(mapping)
    return OverwriteRule(
        _relabel_by_permutation(rule.mask, permutation),
        _relabel_by_permutation(rule.values, permutation),
    )


def apply_overwrite(payload: int, rule: OverwriteRule) -> int:
    """Apply one exact last-write-wins overwrite."""

    normalized = _payload(payload)
    if not isinstance(rule, OverwriteRule):
        raise ConditionalReplacementValidationError(
            "rule must be an OverwriteRule"
        )
    return (normalized & ~rule.mask) | rule.values


def compose_overwrites(rules: Sequence[OverwriteRule]) -> Optional[OverwriteRule]:
    """Collapse an ordered overwrite sequence using declared last-write-wins."""

    if isinstance(rules, (str, bytes)) or not isinstance(rules, Sequence):
        raise ConditionalReplacementValidationError(
            "rules must be a sequence"
        )
    if len(rules) > MAX_OPERATIONS:
        raise ConditionalReplacementResourceError(
            "overwrite count exceeds immutable cap {}".format(MAX_OPERATIONS)
        )
    mask = 0
    values = 0
    rule_count = len(rules)
    for index in range(rule_count):
        rule = rules[index]
        if not isinstance(rule, OverwriteRule):
            raise ConditionalReplacementValidationError(
                "rules[{}] must be an OverwriteRule".format(index)
            )
        values = (values & ~rule.mask) | rule.values
        mask |= rule.mask
    if mask == 0:
        return None
    return OverwriteRule(mask, values & mask)


def canonicalize_program(
    operations: Sequence[ProgramOperation],
) -> ReplacementNormalForm:
    """Reduce a bounded mixed program to affine-then-overwrite normal form."""

    normalized = _normalize_program(operations, MAX_OPERATIONS)
    affine = IDENTITY_AFFINE
    overwrite: Optional[OverwriteRule] = None
    for operation in normalized:
        if isinstance(operation, AddressOperation):
            mapping = operation.affine
            affine = compose_affine(mapping, affine)
            if overwrite is not None:
                overwrite = relabel_overwrite(overwrite, mapping)
        else:
            current = () if overwrite is None else (overwrite,)
            overwrite = compose_overwrites(current + (operation,))
    if overwrite is None:
        return ReplacementNormalForm(affine, 0, 0)
    return ReplacementNormalForm(
        affine,
        overwrite.mask,
        overwrite.values,
    )


def _effective_overwrite_rules(
    operations: Tuple[ProgramOperation, ...],
) -> Tuple[OverwriteRule, ...]:
    """Move each overwrite through later address maps into final coordinates."""

    suffix_affine = IDENTITY_AFFINE
    reversed_rules = []
    for operation in reversed(operations):
        if isinstance(operation, AddressOperation):
            suffix_affine = compose_affine(suffix_affine, operation.affine)
        else:
            reversed_rules.append(
                relabel_overwrite(operation, suffix_affine)
            )
    return tuple(reversed(reversed_rules))


def apply_program(
    payload: int,
    operations: Sequence[ProgramOperation],
) -> int:
    """Execute a validated bounded program directly."""

    state = _payload(payload)
    normalized = _normalize_program(operations, MAX_OPERATIONS)
    for operation in normalized:
        if isinstance(operation, AddressOperation):
            state = relabel_payload(state, operation.affine)
        else:
            state = apply_overwrite(state, operation)
    return state


def apply_normal_form(
    payload: int,
    normal_form: ReplacementNormalForm,
) -> int:
    """Execute one affine-then-overwrite normal form."""

    if not isinstance(normal_form, ReplacementNormalForm):
        raise ConditionalReplacementValidationError(
            "normal_form must be a ReplacementNormalForm"
        )
    state = relabel_payload(_payload(payload), normal_form.affine)
    overwrite = normal_form.overwrite
    if overwrite is not None:
        state = apply_overwrite(state, overwrite)
    return state


def _normalize_program(
    operations: Sequence[ProgramOperation],
    max_operations: int,
) -> Tuple[ProgramOperation, ...]:
    if isinstance(operations, (str, bytes)) or not isinstance(
        operations, Sequence
    ):
        raise ConditionalReplacementValidationError(
            "operations must be a sequence"
        )
    operation_count = len(operations)
    if operation_count < 1:
        raise ConditionalReplacementValidationError(
            "operations must not be empty"
        )
    if operation_count > max_operations:
        raise ConditionalReplacementResourceError(
            "operation count exceeds budget: {} > {}".format(
                operation_count,
                max_operations,
            )
        )
    normalized = tuple(
        operations[index] for index in range(operation_count)
    )
    for index, operation in enumerate(normalized):
        if not isinstance(operation, (AddressOperation, OverwriteRule)):
            raise ConditionalReplacementValidationError(
                "operations[{}] has an unsupported type".format(index)
            )
    return normalized


def _apply_overwrite_sequence(
    payload: int,
    rules: Sequence[OverwriteRule],
) -> int:
    state = payload
    for rule in rules:
        state = apply_overwrite(state, rule)
    return state


def _sector_profile(rules: Tuple[OverwriteRule, OverwriteRule]) -> SectorProfile:
    left, right = rules
    return SectorProfile(
        mask_sizes=(
            _population_count(left.mask),
            _population_count(right.mask),
        ),
        overlap_size=_population_count(left.mask & right.mask),
        union_size=_population_count(left.mask | right.mask),
    )


def _relabel_rule_by_permutation(
    rule: OverwriteRule,
    permutation: Tuple[int, ...],
) -> OverwriteRule:
    return OverwriteRule(
        _relabel_by_permutation(rule.mask, permutation),
        _relabel_by_permutation(rule.values, permutation),
    )


def _iter_payloads() -> Iterable[int]:
    return range(PAYLOAD_COUNT)


def _all_payloads_equal(
    left: Callable[[int], int],
    right: Callable[[int], int],
    deadline: float,
) -> bool:
    for payload in _iter_payloads():
        _check_deadline(deadline)
        if left(payload) != right(payload):
            return False
    return True


def _last_write_wins_holds(
    rules: Tuple[OverwriteRule, OverwriteRule],
    deadline: float,
) -> bool:
    composed = compose_overwrites(rules)
    if composed is None:
        raise ConditionalReplacementValidationError(
            "internal overwrite composition unexpectedly empty"
        )
    return _all_payloads_equal(
        lambda payload: _apply_overwrite_sequence(payload, rules),
        lambda payload: apply_overwrite(payload, composed),
        deadline,
    )


def _pair_commutes(
    rules: Tuple[OverwriteRule, OverwriteRule],
    deadline: float,
) -> bool:
    reverse = (rules[1], rules[0])
    return _all_payloads_equal(
        lambda payload: _apply_overwrite_sequence(payload, rules),
        lambda payload: _apply_overwrite_sequence(payload, reverse),
        deadline,
    )


def _overwrite_sequence_conjugacy_holds(
    rules: Tuple[OverwriteRule, OverwriteRule],
    permutation: Tuple[int, ...],
    deadline: float,
) -> bool:
    inverse = _inverse_permutation(permutation)
    relabeled = tuple(
        _relabel_rule_by_permutation(rule, permutation) for rule in rules
    )
    return _all_payloads_equal(
        lambda payload: _relabel_by_permutation(
            _apply_overwrite_sequence(
                _relabel_by_permutation(payload, inverse),
                rules,
            ),
            permutation,
        ),
        lambda payload: _apply_overwrite_sequence(payload, relabeled),
        deadline,
    )


def _full_normal_form_conjugacy_holds(
    normal_form: ReplacementNormalForm,
    labels: AffineMap,
    deadline: float,
) -> bool:
    inverse_labels = inverse_affine(labels)
    overwrite = normal_form.overwrite
    conjugated_overwrite = (
        None if overwrite is None else relabel_overwrite(overwrite, labels)
    )
    conjugated = ReplacementNormalForm(
        conjugate_affine(normal_form.affine, labels),
        0 if conjugated_overwrite is None else conjugated_overwrite.mask,
        0 if conjugated_overwrite is None else conjugated_overwrite.values,
    )
    return _all_payloads_equal(
        lambda payload: relabel_payload(
            apply_normal_form(
                relabel_payload(payload, inverse_labels),
                normal_form,
            ),
            labels,
        ),
        lambda payload: apply_normal_form(payload, conjugated),
        deadline,
    )


def _rule_shift_conjugacy_holds(
    rule: OverwriteRule,
    shift: AffineMap,
    deadline: float,
) -> bool:
    inverse_shift = inverse_affine(shift)
    shifted_rule = relabel_overwrite(rule, shift)
    return _all_payloads_equal(
        lambda payload: relabel_payload(
            apply_overwrite(
                relabel_payload(payload, inverse_shift),
                rule,
            ),
            shift,
        ),
        lambda payload: apply_overwrite(payload, shifted_rule),
        deadline,
    )


def _random_permutation(seed: int) -> Tuple[int, ...]:
    addresses = list(range(MODULUS))
    random.Random(seed).shuffle(addresses)
    return tuple(addresses)


def _primitive_roots() -> Tuple[int, ...]:
    return PRIMITIVE_ROOTS


def _resource_estimate(
    *,
    operation_count: int,
    address_count: int,
    overwrite_count: int,
    budget: ConditionalReplacementBudget,
) -> ResourceEstimate:
    primitive_root_count = len(_primitive_roots())
    payload_passes = (
        1  # direct program versus normal form
        + 1  # original last-write-wins reduction
        + 1  # original pair commutation
        + overwrite_count  # one shift-conjugacy pass per rule
        + 1  # full-program shift-label conjugacy
        + 1  # shifted overwrite-sequence conjugacy
        + 1  # shifted-control last-write-wins reduction
        + 1  # random-permutation overwrite-sequence conjugacy
        + 1  # random-control last-write-wins reduction
        + primitive_root_count * 3  # sequence, LWW, and full program per root
    )
    payload_cases = PAYLOAD_COUNT * payload_passes
    components = (
        (
            "normalized_program_and_visible_caller_value_allowance",
            operation_count * 2_048,
        ),
        (
            "streamed_payload_and_normal_form_workspace",
            16 * 1_024,
        ),
        (
            "shift_random_and_primitive_root_control_workspace",
            (3 + primitive_root_count) * MODULUS * 256,
        ),
        ("seeded_rng_and_permutation_workspace", 16 * 1_024),
        ("immutable_result_and_diagnostics", 48 * 1_024),
    )
    estimated_bytes = sum(value for _name, value in components)
    estimated_work = (
        MODULUS * max(1, address_count) * 8
        + PAYLOAD_COUNT * (2 * operation_count + 12)
        + PAYLOAD_COUNT * (overwrite_count + 4)
        + PAYLOAD_COUNT * 6
        + PAYLOAD_COUNT * overwrite_count * 12
        + PAYLOAD_COUNT * (2 * operation_count + 16)
        + PAYLOAD_COUNT * (2 * overwrite_count + 16)
        + PAYLOAD_COUNT * (overwrite_count + 4)
        + PAYLOAD_COUNT * (2 * overwrite_count + 16)
        + PAYLOAD_COUNT * (overwrite_count + 4)
        + primitive_root_count
        * PAYLOAD_COUNT
        * (
            (2 * overwrite_count + 16)
            + (overwrite_count + 4)
            + (2 * operation_count + 16)
        )
    )
    if estimated_bytes > budget.max_estimated_bytes:
        raise ConditionalReplacementResourceError(
            "estimated peak exceeds byte budget: {} > {}".format(
                estimated_bytes,
                budget.max_estimated_bytes,
            )
        )
    if estimated_work > budget.max_work_units:
        raise ConditionalReplacementResourceError(
            "estimated work exceeds budget: {} > {}".format(
                estimated_work,
                budget.max_work_units,
            )
        )
    return ResourceEstimate(
        operation_count=operation_count,
        address_operation_count=address_count,
        overwrite_operation_count=overwrite_count,
        exhaustive_payloads_per_pass=PAYLOAD_COUNT,
        exhaustive_payload_passes_upper_bound=payload_passes,
        exhaustive_payload_cases_upper_bound=payload_cases,
        primitive_root_controls=primitive_root_count,
        estimated_peak_bytes=estimated_bytes,
        estimated_work_units=estimated_work,
        component_bytes=components,
        max_estimated_bytes=budget.max_estimated_bytes,
        max_work_units=budget.max_work_units,
        max_seconds=budget.max_seconds,
        max_operations=budget.max_operations,
        payloads_streamed_not_materialized=True,
        caller_owned_inputs_included=False,
        measured_process_peak=False,
    )


def _check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise ConditionalReplacementResourceError(
            "conditional-replacement screen exceeded its wall-clock deadline"
        )


def _control_result(
    *,
    name: str,
    rules: Tuple[OverwriteRule, OverwriteRule],
    permutation: Tuple[int, ...],
    original_profile: SectorProfile,
    deadline: float,
    full_program_invariance: Optional[bool],
) -> ControlResult:
    relabeled = tuple(
        _relabel_rule_by_permutation(rule, permutation) for rule in rules
    )
    typed_relabeled = (relabeled[0], relabeled[1])
    profile = _sector_profile(typed_relabeled)
    return ControlResult(
        name=name,
        permutation=permutation,
        masks=(typed_relabeled[0].mask, typed_relabeled[1].mask),
        profile=profile,
        profile_matches_original=profile == original_profile,
        overwrite_sequence_conjugacy_holds=(
            _overwrite_sequence_conjugacy_holds(
                rules,
                permutation,
                deadline,
            )
        ),
        last_write_wins_holds=_last_write_wins_holds(
            typed_relabeled,
            deadline,
        ),
        full_program_invariance_checked=full_program_invariance is not None,
        full_program_invariance_holds=full_program_invariance,
    )


def analyze_conditional_replacements(
    operations: Sequence[ProgramOperation],
    *,
    control_shift: int = 1,
    random_seed: int = 4691,
    budget: ConditionalReplacementBudget = ConditionalReplacementBudget(),
) -> ConditionalReplacementAnalysis:
    """Run the exact p=7 normal-form and matched-control screen.

    Exactly two overwrite rules are required because this lane tests the
    disjoint/overlap pair laws.  Up to two address operations may be interleaved
    with them, for an immutable maximum of four total operations.
    """

    if not isinstance(budget, ConditionalReplacementBudget):
        raise ConditionalReplacementValidationError(
            "budget must be a ConditionalReplacementBudget"
        )
    started = time.monotonic()
    deadline = started + budget.max_seconds
    normalized = _normalize_program(operations, budget.max_operations)
    shift_value = _canonical_residue(
        control_shift,
        "control_shift",
        nonzero=True,
    )
    seed = _plain_int(random_seed, "random_seed")
    if seed > 0xFFFFFFFF:
        raise ConditionalReplacementValidationError(
            "random_seed must fit in an unsigned 32-bit integer"
        )
    address_operations = tuple(
        operation
        for operation in normalized
        if isinstance(operation, AddressOperation)
    )
    overwrite_rules = _effective_overwrite_rules(normalized)
    if len(overwrite_rules) != 2:
        raise ConditionalReplacementValidationError(
            "the paired screen requires exactly two overwrite rules"
        )
    typed_rules = (overwrite_rules[0], overwrite_rules[1])
    resource = _resource_estimate(
        operation_count=len(normalized),
        address_count=len(address_operations),
        overwrite_count=len(typed_rules),
        budget=budget,
    )
    _check_deadline(deadline)

    normal_form = canonicalize_program(normalized)
    affine = affine_normal_form(address_operations)
    address_normal_form_holds = all(
        _apply_address_operations(address, address_operations)
        == apply_affine(address, affine)
        for address in range(MODULUS)
    )
    program_normal_form_holds = _all_payloads_equal(
        lambda payload: apply_program(payload, normalized),
        lambda payload: apply_normal_form(payload, normal_form),
        deadline,
    )
    profile = _sector_profile(typed_rules)
    pair_is_disjoint = profile.overlap_size == 0
    pair_commutes = _pair_commutes(typed_rules, deadline)
    lww_holds = _last_write_wins_holds(typed_rules, deadline)

    shift_labels = AffineMap(1, shift_value)
    shift_permutation = _affine_permutation(shift_labels)
    shift_rule_conjugacy = all(
        _rule_shift_conjugacy_holds(rule, shift_labels, deadline)
        for rule in typed_rules
    )
    shift_full_invariance = _full_normal_form_conjugacy_holds(
        normal_form,
        shift_labels,
        deadline,
    )
    shift_control = _control_result(
        name="cyclic_shift_conjugate",
        rules=typed_rules,
        permutation=shift_permutation,
        original_profile=profile,
        deadline=deadline,
        full_program_invariance=shift_full_invariance,
    )

    random_permutation = _random_permutation(seed)
    random_control = _control_result(
        name="seeded_random_address_permutation",
        rules=typed_rules,
        permutation=random_permutation,
        original_profile=profile,
        deadline=deadline,
        full_program_invariance=None,
    )

    primitive_results = []
    for root in _primitive_roots():
        _check_deadline(deadline)
        root_labels = AffineMap(root, 0)
        root_permutation = _affine_permutation(root_labels)
        relabeled = tuple(
            _relabel_rule_by_permutation(rule, root_permutation)
            for rule in typed_rules
        )
        typed_relabeled = (relabeled[0], relabeled[1])
        primitive_results.append(
            PrimitiveRootResult(
                root=root,
                is_primitive_root=True,
                profile_matches_original=(
                    _sector_profile(typed_relabeled) == profile
                ),
                overwrite_sequence_conjugacy_holds=(
                    _overwrite_sequence_conjugacy_holds(
                        typed_rules,
                        root_permutation,
                        deadline,
                    )
                ),
                last_write_wins_holds=_last_write_wins_holds(
                    typed_relabeled,
                    deadline,
                ),
                full_program_invariance_holds=(
                    _full_normal_form_conjugacy_holds(
                        normal_form,
                        root_labels,
                        deadline,
                    )
                ),
                conjugated_affine=conjugate_affine(
                    normal_form.affine,
                    root_labels,
                ),
            )
        )
    _check_deadline(deadline)

    return ConditionalReplacementAnalysis(
        protocol_id=PROTOCOL_ID,
        evidence_mode=EVIDENCE_MODE,
        modulus=MODULUS,
        operation_count=len(normalized),
        address_operation_count=len(address_operations),
        overwrite_operation_count=len(typed_rules),
        affine_normal_form=affine,
        replacement_normal_form=normal_form,
        address_normal_form_holds=address_normal_form_holds,
        program_normal_form_holds=program_normal_form_holds,
        pair_profile=profile,
        pair_is_disjoint=pair_is_disjoint,
        pair_commutes_on_all_payloads=pair_commutes,
        disjoint_commutation_applicable=pair_is_disjoint,
        disjoint_commutation_holds=pair_commutes if pair_is_disjoint else None,
        overlapping_last_write_wins_applicable=not pair_is_disjoint,
        last_write_wins_holds=lww_holds,
        shift_conjugacy_holds=shift_rule_conjugacy,
        shift_control=shift_control,
        random_matched_control=random_control,
        primitive_root_controls=tuple(primitive_results),
        resource_guard=resource,
        hypothesis_status=HYPOTHESIS_STATUS,
        promotion_eligible=False,
        novelty_claim=False,
        ml_utility_claim=False,
        runtime_benefit_claim=False,
        limitations=LIMITATIONS,
    )


def _apply_address_operations(
    address: int,
    operations: Sequence[AddressOperation],
) -> int:
    state = address
    for operation in operations:
        state = apply_affine(state, operation.affine)
    return state


__all__ = [
    "AddressOperation",
    "AffineMap",
    "ConditionalReplacementAnalysis",
    "ConditionalReplacementBudget",
    "ConditionalReplacementResourceError",
    "ConditionalReplacementValidationError",
    "ControlResult",
    "DEFAULT_MAX_ESTIMATED_BYTES",
    "DEFAULT_MAX_SECONDS",
    "DEFAULT_MAX_WORK_UNITS",
    "EVIDENCE_MODE",
    "HARD_MAX_ESTIMATED_BYTES",
    "HARD_MAX_SECONDS",
    "HARD_MAX_WORK_UNITS",
    "HYPOTHESIS_STATUS",
    "LIMITATIONS",
    "MAX_OPERATIONS",
    "MODULUS",
    "OverwriteRule",
    "PAYLOAD_COUNT",
    "PRIMITIVE_ROOTS",
    "PROTOCOL_ID",
    "PrimitiveRootResult",
    "ReplacementNormalForm",
    "ResourceEstimate",
    "SectorProfile",
    "affine_normal_form",
    "analyze_conditional_replacements",
    "apply_affine",
    "apply_normal_form",
    "apply_overwrite",
    "apply_program",
    "canonicalize_program",
    "compose_affine",
    "compose_overwrites",
    "conjugate_affine",
    "inverse_affine",
    "pivot",
    "relabel_overwrite",
    "relabel_payload",
    "rotation",
    "sector_overwrite",
]
