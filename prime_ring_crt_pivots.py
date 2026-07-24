"""Proof-first, resource-bounded algebraic screen for PRW-C1.

The executable domain is the punctured field ``F_p^*``.  If ``g`` is a
primitive root and ``x = g**n``, the function action

``M_t f(x) = f(g**(-t) * x)``

is exactly the ordinary cyclic gather ``f_tilde(n - t)`` in exponent
coordinates.  A CRT representation of the exponent only reshapes that one
shift into componentwise additions; it does not create additional states.

For the standard quadratic character, ``M_t chi = (-1)**t chi`` on
``F_p^*``.  The repository's bipolar carrier convention stores ``L(0)=+1``
instead of ``chi(0)=0``.  Consequently an odd pivot is ``-L + 2*e_0`` on the
full ring: one nonzero-coordinate polarity bit plus one fixed-coordinate
exception, not a literal global sign flip.

This module makes no retrieval, learned-payload, implementation-cost,
performance, or novelty claim.  It uses only the Python standard library and
refuses an analysis before any order-sized allocation when its conservative
byte/work estimate exceeds a caller-lowerable budget.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple


TARGET_PRIME = 4691
TARGET_ORDER = TARGET_PRIME - 1
TARGET_PRIMITIVE_ROOT = 2
TARGET_CRT_FACTORS = (2, 5, 7, 67)
DEFAULT_PIVOT_EXPONENTS = (0, 1, 2, 5, 7, 67, 2345, 4689)

MIB = 1024 * 1024
HARD_MAX_PRIME = TARGET_PRIME
HARD_MAX_FACTORS = 8
HARD_MAX_PIVOTS = 32
HARD_MAX_LAYERS = 64
HARD_MAX_AFFINE_WORD_LENGTH = 64
HARD_MAX_INPUT_BITS = 64
HARD_MAX_ESTIMATED_BYTES = 64 * MIB
HARD_MAX_WORK_UNITS = 20_000_000
HARD_MAX_SECONDS = 30.0

DEFAULT_MAX_ESTIMATED_BYTES = 8 * MIB
DEFAULT_MAX_WORK_UNITS = 4_000_000
DEFAULT_MAX_SECONDS = 5.0

EVIDENCE_STATUS = "EXACT_FINITE_GROUP_REDUNDANCY_SCREEN_ONLY"
THEOREM_STATEMENT = (
    "On F_p^*, phi(n)=g^n identifies multiplication by g^(-t) with "
    "the additive exponent shift n->n-t. CRT coordinates are a bijective "
    "reshape of that same shift. The quadratic character changes by "
    "(-1)^t on F_p^*; with the repository convention L(0)=+1, zero remains "
    "fixed and an odd pivot is -L+2e_0 rather than a global sign flip. A "
    "carrier centered at b moves to g^t*b with the same fixed-center "
    "exception."
)


class PRWC1ValidationError(ValueError):
    """Raised when a PRW-C1 algebraic contract is malformed."""


class PRWC1ResourceError(RuntimeError):
    """Raised before a PRW-C1 calculation exceeds a frozen resource budget."""


def _plain_int(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PRWC1ValidationError(f"{name} must be an integer")
    result = int(value)
    if result.bit_length() > HARD_MAX_INPUT_BITS:
        raise PRWC1ResourceError(
            f"{name} exceeds the {HARD_MAX_INPUT_BITS}-bit input ceiling"
        )
    if result < minimum:
        raise PRWC1ValidationError(f"{name} must be >= {minimum}")
    return result


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PRWC1ValidationError(f"{name} must be a finite positive number")
    if isinstance(value, int):
        result = float(_plain_int(value, name, 0))
    else:
        result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise PRWC1ValidationError(f"{name} must be a finite positive number")
    return result


def _is_prime_exact(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def _prime_divisors(value: int) -> Tuple[int, ...]:
    remaining = value
    factors = []
    divisor = 2
    while divisor * divisor <= remaining:
        if remaining % divisor == 0:
            factors.append(divisor)
            while remaining % divisor == 0:
                remaining //= divisor
        divisor = 3 if divisor == 2 else divisor + 2
    if remaining > 1:
        factors.append(remaining)
    return tuple(factors)


def _is_primitive_root_exact(root: int, prime: int) -> bool:
    if not _is_prime_exact(prime) or not 1 < root < prime:
        return False
    order = prime - 1
    return all(
        pow(root, order // factor, prime) != 1
        for factor in _prime_divisors(order)
    )


def _extended_gcd(left: int, right: int) -> Tuple[int, int, int]:
    old_r, r = left, right
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    return old_r, old_s, old_t


def _mod_inverse(value: int, modulus: int) -> int:
    divisor, coefficient, _unused = _extended_gcd(value, modulus)
    if divisor != 1:
        raise PRWC1ValidationError("CRT moduli must be pairwise coprime")
    return coefficient % modulus


def _validated_factors(
    factors: object,
    order: int,
) -> Tuple[int, ...]:
    if not isinstance(factors, tuple):
        raise PRWC1ValidationError("crt_factors must be a bounded tuple")
    if not factors or len(factors) > HARD_MAX_FACTORS:
        raise PRWC1ValidationError(
            f"crt_factors must contain 1..{HARD_MAX_FACTORS} moduli"
        )
    normalized = tuple(
        _plain_int(value, "CRT factor", 2) for value in factors
    )
    product = 1
    for index, modulus in enumerate(normalized):
        for prior in normalized[:index]:
            if math.gcd(modulus, prior) != 1:
                raise PRWC1ValidationError(
                    "CRT factors must be pairwise coprime"
                )
        product *= modulus
    if product != order:
        raise PRWC1ValidationError(
            f"CRT factor product must equal p-1 ({order})"
        )
    return normalized


def _validated_pivots(
    pivot_exponents: object,
    order: int,
    maximum: int,
) -> Tuple[int, ...]:
    if not isinstance(pivot_exponents, tuple):
        raise PRWC1ValidationError(
            "pivot_exponents must be a bounded tuple"
        )
    if not pivot_exponents or len(pivot_exponents) > maximum:
        raise PRWC1ValidationError(
            f"pivot_exponents must contain 1..{maximum} values"
        )
    normalized = tuple(
        _plain_int(value, "pivot exponent", 0) % order
        for value in pivot_exponents
    )
    if len(set(normalized)) != len(normalized):
        raise PRWC1ValidationError(
            "pivot_exponents must be distinct modulo p-1"
        )
    return normalized


@dataclass(frozen=True)
class CRTBudget:
    """Caller-lowerable caps beneath immutable PRW-C1 ceilings."""

    max_estimated_bytes: int = DEFAULT_MAX_ESTIMATED_BYTES
    max_work_units: int = DEFAULT_MAX_WORK_UNITS
    max_seconds: float = DEFAULT_MAX_SECONDS
    max_prime: int = HARD_MAX_PRIME
    max_pivots: int = HARD_MAX_PIVOTS
    max_layers: int = HARD_MAX_LAYERS

    def __post_init__(self) -> None:
        integer_limits = (
            ("max_estimated_bytes", HARD_MAX_ESTIMATED_BYTES),
            ("max_work_units", HARD_MAX_WORK_UNITS),
            ("max_prime", HARD_MAX_PRIME),
            ("max_pivots", HARD_MAX_PIVOTS),
            ("max_layers", HARD_MAX_LAYERS),
        )
        for name, hard_limit in integer_limits:
            value = _plain_int(getattr(self, name), name, 1)
            if value > hard_limit:
                raise PRWC1ValidationError(
                    f"{name} cannot exceed hard ceiling {hard_limit}"
                )
            object.__setattr__(self, name, value)
        seconds = _positive_float(self.max_seconds, "max_seconds")
        if seconds > HARD_MAX_SECONDS:
            raise PRWC1ValidationError(
                f"max_seconds cannot exceed hard ceiling {HARD_MAX_SECONDS:g}"
            )
        object.__setattr__(self, "max_seconds", seconds)


@dataclass(frozen=True)
class CRTPreflight:
    """Conservative accounting performed before order-sized allocation."""

    prime: int
    order: int
    primitive_root: int
    crt_factors: Tuple[int, ...]
    pivot_count: int
    layer_count: int
    estimated_peak_bytes: int
    estimated_work_units: int
    max_estimated_bytes: int
    max_work_units: int
    allowed: bool
    component_bytes: Tuple[Tuple[str, int], ...]
    exact_prime_verified: bool = False
    primitive_root_verified: bool = False
    measured_process_peak: bool = False


@dataclass(frozen=True)
class AffineMap:
    """Canonical map ``x -> multiplier*x + offset (mod prime)``."""

    prime: int
    multiplier: int
    offset: int

    def __post_init__(self) -> None:
        p = _plain_int(self.prime, "prime", 3)
        if p > HARD_MAX_PRIME or not _is_prime_exact(p):
            raise PRWC1ValidationError(
                f"prime must be an odd prime <= {HARD_MAX_PRIME}"
            )
        multiplier = _plain_int(
            self.multiplier,
            "multiplier",
            1,
        ) % p
        if multiplier == 0:
            raise PRWC1ValidationError(
                "affine multiplier must be nonzero modulo prime"
            )
        offset = _plain_int(self.offset, "offset", 0) % p
        object.__setattr__(self, "prime", p)
        object.__setattr__(self, "multiplier", multiplier)
        object.__setattr__(self, "offset", offset)

    def apply(self, value: int) -> int:
        coordinate = _plain_int(value, "value", 0)
        return (self.multiplier * coordinate + self.offset) % self.prime


@dataclass(frozen=True)
class PivotWitness:
    """One bounded executable witness for the theorem-level all-t identity."""

    exponent: int
    inverse_multiplier: int
    exponent_delta: int
    crt_delta: Tuple[int, ...]
    rows_checked: int
    multiplication_equals_exponent_shift: bool
    crt_component_shift_equivalent: bool
    flat_payload_gather_equivalent: bool
    alternate_root_control_verified: bool
    zero_fixed: bool
    nonzero_legendre_polarity_sign: int
    nonzero_legendre_polarity_equivalent: bool
    full_bipolar_global_polarity_equivalent: bool
    full_bipolar_fixed_zero_exception: bool
    shifted_legendre_source_center: int
    shifted_legendre_target_center: int
    shifted_legendre_center_transport_verified: bool
    flat_permutation_digest: str
    transported_payload_digest: str


@dataclass(frozen=True)
class RandomConjugacyControl:
    """A deterministic random relabeling of the same cyclic action."""

    seed: int
    coordinate_count: int
    family_state_count: int
    relabeling_digest: str
    generator_digest: str
    relabeling_is_bijection: bool
    conjugated_generator_is_bijection: bool
    conjugated_generator_has_full_order: bool
    cyclic_composition_verified: bool
    native_exponent_shift: bool
    matched_as_permutation_only: bool
    implementation_cost_matched: bool
    utility_tested: bool


@dataclass(frozen=True)
class KillCriterion:
    """A falsifier and whether this algebraic screen triggers it."""

    name: str
    triggered: bool
    evidence: str


@dataclass(frozen=True)
class PRWC1Report:
    """Exact finite-group result, bounded witnesses, and explicit limitations."""

    evidence_status: str
    theorem_statement: str
    theorem_domain: str
    theorem_applies_to_all_pivots: bool
    executable_witnesses_are_exhaustive_over_all_pivots: bool
    root_choice_invariance_theorem_applied: bool
    alternate_root_controls_exhaustive: bool
    preflight: CRTPreflight
    power_coordinate_bijection_verified: bool
    crt_round_trip_verified: bool
    crt_tuple_count: int
    component_factorization_creates_extra_states: bool
    shared_pivot_state_count: int
    layer_count: int
    conditional_independent_layer_control_tuple_count: int
    conditional_independent_layer_control_information_bits: float
    conditional_independent_layer_min_fixed_code_bits: int
    independent_layer_addressability_assumed_not_verified: bool
    distinguishable_cross_layer_states_measured: bool
    independent_layers_enumerated: bool
    zero_excluded_from_exponent_coordinates: bool
    zero_fixed_point_verified: bool
    legendre_stabilizer_size: int
    legendre_effective_orbit_size: int
    factor_two_coordinate_is_nonzero_polarity_bit: bool
    shifted_legendre_center_transport_verified: bool
    pivot_witnesses: Tuple[PivotWitness, ...]
    random_control: RandomConjugacyControl
    affine_state_upper_bound: int
    affine_conjugation_identity_verified: bool
    affine_closure_theorem_applied: bool
    mixed_permutation_word_runtime_examples_exhaustive: bool
    kill_criteria: Tuple[KillCriterion, ...]
    arbitrary_learned_payload_polarity_degeneracy_claimed: bool
    learned_payload_utility_tested: bool
    implementation_cost_tested: bool
    performance_benchmarked: bool
    novelty_claimed: bool
    limitations: Tuple[str, ...]


def _memory_components(
    order: int,
    pivot_count: int,
) -> Tuple[Tuple[str, int], ...]:
    # These are deliberately coarse Python-object allowances rather than
    # process-RSS measurements.
    return (
        ("power_table", order * 64),
        ("inverse_power_dictionary", order * 192),
        ("quadratic_character_table", order * 64),
        ("current_flat_permutation_allowance", order * 64),
        ("random_relabeling_and_inverse", order * 128),
        ("random_bijection_seen_bitmaps", 3 * order),
        ("payload_and_digest_workspace", order * 128),
        ("fixed_runtime_overhead", 256 * 1024),
        ("witness_report_overhead", pivot_count * 4096),
    )


def estimate_crt_pivot_resources(
    *,
    prime: int = TARGET_PRIME,
    primitive_root: int = TARGET_PRIMITIVE_ROOT,
    crt_factors: Tuple[int, ...] = TARGET_CRT_FACTORS,
    pivot_exponents: Tuple[int, ...] = DEFAULT_PIVOT_EXPONENTS,
    layer_count: int = 8,
    budget: CRTBudget = CRTBudget(),
) -> CRTPreflight:
    """Estimate the exact screen without primality work or large allocation."""

    if not isinstance(budget, CRTBudget):
        raise PRWC1ValidationError("budget must be a CRTBudget")
    p = _plain_int(prime, "prime", 3)
    if p % 2 == 0:
        raise PRWC1ValidationError("prime must be odd")
    root = _plain_int(primitive_root, "primitive_root", 2)
    if p > budget.max_prime:
        raise PRWC1ResourceError(
            f"prime exceeds bounded PRW-C1 limit: {p} > {budget.max_prime}"
        )
    order = p - 1
    factors = _validated_factors(crt_factors, order)
    pivots = _validated_pivots(
        pivot_exponents,
        order,
        budget.max_pivots,
    )
    layers = _plain_int(layer_count, "layer_count", 1)
    if layers > budget.max_layers:
        raise PRWC1ResourceError(
            "layer_count exceeds bounded PRW-C1 limit: "
            f"{layers} > {budget.max_layers}"
        )
    components = _memory_components(order, len(pivots))
    estimated_bytes = sum(size for _name, size in components)
    estimated_work = order * (
        80 + 24 * len(factors) + 36 * len(pivots)
    )
    return CRTPreflight(
        prime=p,
        order=order,
        primitive_root=root,
        crt_factors=factors,
        pivot_count=len(pivots),
        layer_count=layers,
        estimated_peak_bytes=estimated_bytes,
        estimated_work_units=estimated_work,
        max_estimated_bytes=budget.max_estimated_bytes,
        max_work_units=budget.max_work_units,
        allowed=(
            estimated_bytes <= budget.max_estimated_bytes
            and estimated_work <= budget.max_work_units
        ),
        component_bytes=components,
    )


def preflight_crt_pivot_screen(
    *,
    prime: int = TARGET_PRIME,
    primitive_root: int = TARGET_PRIMITIVE_ROOT,
    crt_factors: Tuple[int, ...] = TARGET_CRT_FACTORS,
    pivot_exponents: Tuple[int, ...] = DEFAULT_PIVOT_EXPONENTS,
    layer_count: int = 8,
    budget: CRTBudget = CRTBudget(),
) -> CRTPreflight:
    """Return an allowed and algebraically validated preflight."""

    estimate = estimate_crt_pivot_resources(
        prime=prime,
        primitive_root=primitive_root,
        crt_factors=crt_factors,
        pivot_exponents=pivot_exponents,
        layer_count=layer_count,
        budget=budget,
    )
    if not estimate.allowed:
        raise PRWC1ResourceError(
            "PRW-C1 estimate exceeds caller budget: "
            f"{estimate.estimated_peak_bytes} bytes/"
            f"{estimate.estimated_work_units} work units"
        )
    if not _is_prime_exact(estimate.prime):
        raise PRWC1ValidationError("prime must be exactly prime")
    if not _is_primitive_root_exact(
        estimate.primitive_root,
        estimate.prime,
    ):
        raise PRWC1ValidationError(
            "primitive_root must generate every nonzero residue"
        )
    return replace(
        estimate,
        exact_prime_verified=True,
        primitive_root_verified=True,
    )


def crt_coordinates(
    exponent: int,
    *,
    order: int = TARGET_ORDER,
    factors: Tuple[int, ...] = TARGET_CRT_FACTORS,
) -> Tuple[int, ...]:
    """Return the declared CRT coordinates of one exponent."""

    modulus = _plain_int(order, "order", 2)
    validated = _validated_factors(factors, modulus)
    value = _plain_int(exponent, "exponent", 0) % modulus
    return tuple(value % factor for factor in validated)


def crt_reconstruct(
    residues: object,
    *,
    order: int = TARGET_ORDER,
    factors: Tuple[int, ...] = TARGET_CRT_FACTORS,
) -> int:
    """Reconstruct the unique exponent from a bounded CRT tuple."""

    modulus = _plain_int(order, "order", 2)
    validated = _validated_factors(factors, modulus)
    if not isinstance(residues, tuple) or len(residues) != len(validated):
        raise PRWC1ValidationError(
            "residues must be a tuple matching crt_factors"
        )
    total = 0
    for residue, factor in zip(residues, validated):
        normalized = _plain_int(residue, "CRT residue", 0)
        if normalized >= factor:
            raise PRWC1ValidationError(
                "CRT residues must be canonical for their factors"
            )
        partial = modulus // factor
        total += normalized * partial * _mod_inverse(partial, factor)
    return total % modulus


def reduce_affine_word(
    prime: int,
    operations: object,
) -> AffineMap:
    """Collapse rotations and multiplications into one affine map.

    Operations are applied in listed order.  Each item is ``("T", b)`` for
    ``x -> x+b`` or ``("M", a)`` for ``x -> a*x``.
    """

    p = _plain_int(prime, "prime", 3)
    if p > HARD_MAX_PRIME or not _is_prime_exact(p):
        raise PRWC1ValidationError(
            f"prime must be an odd prime <= {HARD_MAX_PRIME}"
        )
    if not isinstance(operations, tuple):
        raise PRWC1ValidationError("operations must be a bounded tuple")
    if len(operations) > HARD_MAX_AFFINE_WORD_LENGTH:
        raise PRWC1ResourceError(
            "affine word exceeds the frozen operation limit"
        )
    multiplier = 1
    offset = 0
    for operation in operations:
        if (
            not isinstance(operation, tuple)
            or len(operation) != 2
            or operation[0] not in ("T", "M")
        ):
            raise PRWC1ValidationError(
                "each operation must be ('T', b) or ('M', a)"
            )
        value = _plain_int(operation[1], "operation value", 0) % p
        if operation[0] == "T":
            offset = (offset + value) % p
        else:
            if value == 0:
                raise PRWC1ValidationError(
                    "multiplicative affine operations must be nonzero"
                )
            multiplier = (value * multiplier) % p
            offset = (value * offset) % p
    return AffineMap(
        prime=p,
        multiplier=multiplier,
        offset=offset,
    )


def _deadline_check(deadline: float, index: int) -> None:
    if index % 256 == 0 and time.monotonic() > deadline:
        raise PRWC1ResourceError(
            "PRW-C1 exact screen exceeded its wall-clock deadline"
        )


def _crt_constants(
    order: int,
    factors: Tuple[int, ...],
) -> Tuple[Tuple[int, int, int], ...]:
    return tuple(
        (
            factor,
            order // factor,
            _mod_inverse(order // factor, factor),
        )
        for factor in factors
    )


def _crt_reconstruct_unchecked(
    residues: Sequence[int],
    order: int,
    constants: Tuple[Tuple[int, int, int], ...],
) -> int:
    return (
        sum(
            residue * partial * inverse
            for residue, (_factor, partial, inverse) in zip(
                residues,
                constants,
            )
        )
        % order
    )


def _digest_update_int(digest: "hashlib._Hash", value: int) -> None:
    digest.update(int(value).to_bytes(8, "little", signed=False))


def _payload_value(residue: int) -> int:
    # An injective, non-character payload witness over the bounded residue
    # domain.  Equality cannot be hidden by payload collisions.
    return residue


def _splitmix64_step(state: int) -> Tuple[int, int]:
    mask = (1 << 64) - 1
    state = (state + 0x9E3779B97F4A7C15) & mask
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return state, value & mask


def _random_conjugacy_control(
    order: int,
    seed: int,
    deadline: float,
) -> RandomConjugacyControl:
    permutation = list(range(order))
    state = seed & ((1 << 64) - 1)
    for index in range(order - 1, 0, -1):
        _deadline_check(deadline, order - index)
        state, random_value = _splitmix64_step(state)
        selected = random_value % (index + 1)
        permutation[index], permutation[selected] = (
            permutation[selected],
            permutation[index],
        )
    inverse = [0] * order
    seen = bytearray(order)
    relabeling_digest = hashlib.sha256()
    bijection = True
    for index, target in enumerate(permutation):
        _deadline_check(deadline, index)
        if target < 0 or target >= order or seen[target]:
            bijection = False
        else:
            seen[target] = 1
            inverse[target] = index
        _digest_update_int(relabeling_digest, target)

    def conjugated_shift(coordinate: int, shift: int) -> int:
        return permutation[(inverse[coordinate] - shift) % order]

    generator_digest = hashlib.sha256()
    conjugated_seen = bytearray(order)
    generator_bijection = True
    native_shift = True
    native_delta: Optional[int] = None
    composition = True
    first_shift = 1
    second_shift = 2 if order > 2 else 1
    for coordinate in range(order):
        _deadline_check(deadline, coordinate)
        target = conjugated_shift(coordinate, first_shift)
        if conjugated_seen[target]:
            generator_bijection = False
        conjugated_seen[target] = 1
        _digest_update_int(generator_digest, target)
        delta = (target - coordinate) % order
        if native_delta is None:
            native_delta = delta
        elif delta != native_delta:
            native_shift = False
        composed = conjugated_shift(
            conjugated_shift(coordinate, second_shift),
            first_shift,
        )
        expected = conjugated_shift(
            coordinate,
            (first_shift + second_shift) % order,
        )
        if composed != expected:
            composition = False
    cycle_seen = bytearray(order)
    cycle_coordinate = 0
    full_order = True
    for step in range(order):
        _deadline_check(deadline, step)
        if cycle_seen[cycle_coordinate]:
            full_order = False
            break
        cycle_seen[cycle_coordinate] = 1
        cycle_coordinate = conjugated_shift(
            cycle_coordinate,
            first_shift,
        )
    if cycle_coordinate != 0 or not all(cycle_seen):
        full_order = False
    permutation_match = (
        bijection
        and generator_bijection
        and full_order
        and composition
    )
    return RandomConjugacyControl(
        seed=seed,
        coordinate_count=order,
        family_state_count=order,
        relabeling_digest=relabeling_digest.hexdigest(),
        generator_digest=generator_digest.hexdigest(),
        relabeling_is_bijection=bijection,
        conjugated_generator_is_bijection=generator_bijection,
        conjugated_generator_has_full_order=full_order,
        cyclic_composition_verified=composition,
        native_exponent_shift=native_shift,
        matched_as_permutation_only=permutation_match,
        implementation_cost_matched=False,
        utility_tested=False,
    )


def run_prw_c1_redundancy_screen(
    *,
    prime: int = TARGET_PRIME,
    primitive_root: int = TARGET_PRIMITIVE_ROOT,
    crt_factors: Tuple[int, ...] = TARGET_CRT_FACTORS,
    pivot_exponents: Tuple[int, ...] = DEFAULT_PIVOT_EXPONENTS,
    layer_count: int = 8,
    random_seed: int = 20260724,
    budget: CRTBudget = CRTBudget(),
) -> PRWC1Report:
    """Run the bounded exact PRW-C1 redundancy screen."""

    seed = _plain_int(random_seed, "random_seed", 0)
    preflight = preflight_crt_pivot_screen(
        prime=prime,
        primitive_root=primitive_root,
        crt_factors=crt_factors,
        pivot_exponents=pivot_exponents,
        layer_count=layer_count,
        budget=budget,
    )
    pivots = _validated_pivots(
        pivot_exponents,
        preflight.order,
        budget.max_pivots,
    )
    deadline = time.monotonic() + budget.max_seconds
    p = preflight.prime
    order = preflight.order
    root = preflight.primitive_root
    factors = preflight.crt_factors

    powers_list = [0] * order
    inverse_powers = {}
    residue = 1
    power_bijection = True
    for exponent in range(order):
        _deadline_check(deadline, exponent)
        if residue == 0 or residue in inverse_powers:
            power_bijection = False
        powers_list[exponent] = residue
        inverse_powers[residue] = exponent
        residue = (residue * root) % p
    if (
        not power_bijection
        or residue != 1
        or len(inverse_powers) != order
    ):
        raise RuntimeError("primitive-root power table failed exact bijection")
    powers = tuple(powers_list)

    constants = _crt_constants(order, factors)
    crt_round_trip = True
    for exponent in range(order):
        _deadline_check(deadline, exponent)
        coordinates = tuple(exponent % factor for factor in factors)
        if (
            _crt_reconstruct_unchecked(coordinates, order, constants)
            != exponent
        ):
            crt_round_trip = False
            break

    character_list = [0] * order
    for exponent, value in enumerate(powers):
        _deadline_check(deadline, exponent)
        euler = pow(value, order // 2, p)
        character = 1 if euler == 1 else -1 if euler == p - 1 else 0
        expected = 1 if exponent % 2 == 0 else -1
        if character != expected:
            raise RuntimeError(
                "Euler character disagrees with primitive-root parity"
            )
        character_list[exponent] = character
    character = tuple(character_list)

    root_change_unit = 3
    while math.gcd(root_change_unit, order) != 1:
        root_change_unit += 2
    root_change_inverse = _mod_inverse(root_change_unit, order)
    alternate_root = pow(root, root_change_unit, p)
    if not _is_primitive_root_exact(alternate_root, p):
        raise RuntimeError("root-choice control failed to produce a generator")

    witnesses = []
    zero_fixed_all = True
    shifted_center_all = True
    shifted_source_center = 17 % p
    for pivot_index, exponent in enumerate(pivots):
        _deadline_check(deadline, pivot_index)
        source_delta = (-exponent) % order
        inverse_multiplier = powers[source_delta]
        crt_delta = tuple(source_delta % factor for factor in factors)
        sign = -1 if exponent % 2 else 1
        permutation_digest = hashlib.sha256()
        payload_digest = hashlib.sha256()
        multiplication_equal = True
        crt_equal = True
        payload_equal = True
        polarity_equal = True
        root_invariant = True
        shifted_center_equal = True
        alternate_shift = (
            exponent * root_change_inverse
        ) % order
        for coordinate in range(order):
            _deadline_check(deadline, coordinate)
            target = (coordinate - exponent) % order
            product = (inverse_multiplier * powers[coordinate]) % p
            if product != powers[target]:
                multiplication_equal = False
            coordinates = tuple(
                coordinate % factor for factor in factors
            )
            shifted_coordinates = tuple(
                (value + delta) % factor
                for value, delta, factor in zip(
                    coordinates,
                    crt_delta,
                    factors,
                )
            )
            if (
                _crt_reconstruct_unchecked(
                    shifted_coordinates,
                    order,
                    constants,
                )
                != target
            ):
                crt_equal = False
            generic_value = _payload_value(powers[target])
            multiplicative_value = _payload_value(product)
            if generic_value != multiplicative_value:
                payload_equal = False
            if character[target] != sign * character[coordinate]:
                polarity_equal = False
            alternate_coordinate = (
                root_change_unit * coordinate
            ) % order
            alternate_target = (
                root_change_unit
                * ((coordinate - alternate_shift) % order)
            ) % order
            if (
                inverse_multiplier * powers[alternate_coordinate] % p
                != powers[alternate_target]
            ):
                root_invariant = False
            _digest_update_int(permutation_digest, target)
            _digest_update_int(payload_digest, generic_value)
        shifted_target_center = (
            powers[exponent] * shifted_source_center
        ) % p
        for coordinate in range(p):
            _deadline_check(deadline, coordinate)
            source_coordinate = inverse_multiplier * coordinate % p
            source_difference = (
                source_coordinate - shifted_source_center
            ) % p
            observed = (
                1
                if source_difference == 0
                else character[inverse_powers[source_difference]]
            )
            target_difference = (
                coordinate - shifted_target_center
            ) % p
            target_carrier = (
                1
                if target_difference == 0
                else character[inverse_powers[target_difference]]
            )
            expected = sign * target_carrier
            if coordinate == shifted_target_center:
                expected += 1 - sign
            if observed != expected:
                shifted_center_equal = False
        zero_fixed = inverse_multiplier * 0 % p == 0
        zero_fixed_all = zero_fixed_all and zero_fixed
        shifted_center_all = shifted_center_all and shifted_center_equal
        witnesses.append(
            PivotWitness(
                exponent=exponent,
                inverse_multiplier=inverse_multiplier,
                exponent_delta=source_delta,
                crt_delta=crt_delta,
                rows_checked=order,
                multiplication_equals_exponent_shift=multiplication_equal,
                crt_component_shift_equivalent=crt_equal,
                flat_payload_gather_equivalent=payload_equal,
                alternate_root_control_verified=root_invariant,
                zero_fixed=zero_fixed,
                nonzero_legendre_polarity_sign=sign,
                nonzero_legendre_polarity_equivalent=polarity_equal,
                full_bipolar_global_polarity_equivalent=(sign == 1),
                full_bipolar_fixed_zero_exception=(sign == -1),
                shifted_legendre_source_center=shifted_source_center,
                shifted_legendre_target_center=shifted_target_center,
                shifted_legendre_center_transport_verified=(
                    shifted_center_equal
                ),
                flat_permutation_digest=permutation_digest.hexdigest(),
                transported_payload_digest=payload_digest.hexdigest(),
            )
        )

    random_control = _random_conjugacy_control(order, seed, deadline)

    affine_multiplier = powers[1]
    affine_inverse = _mod_inverse(affine_multiplier, p)
    affine_offset = 17 % p
    conjugation = reduce_affine_word(
        p,
        (
            ("M", affine_inverse),
            ("T", affine_offset),
            ("M", affine_multiplier),
        ),
    )
    affine_conjugation = (
        conjugation.multiplier == 1
        and conjugation.offset
        == affine_multiplier * affine_offset % p
    )

    all_flat = all(
        witness.multiplication_equals_exponent_shift
        and witness.crt_component_shift_equivalent
        and witness.flat_payload_gather_equivalent
        and witness.alternate_root_control_verified
        for witness in witnesses
    )
    all_polarity = all(
        witness.nonzero_legendre_polarity_equivalent
        and (
            witness.full_bipolar_global_polarity_equivalent
            != witness.full_bipolar_fixed_zero_exception
        )
        for witness in witnesses
    )
    factor_two_polarity = 2 in factors and all(
        witness.crt_delta[factors.index(2)] == witness.exponent % 2
        and witness.nonzero_legendre_polarity_sign
        == (-1 if witness.crt_delta[factors.index(2)] else 1)
        for witness in witnesses
    )
    component_count = math.prod(factors)
    layer_states = pow(order, preflight.layer_count)
    layer_information = preflight.layer_count * math.log2(order)
    return PRWC1Report(
        evidence_status=EVIDENCE_STATUS,
        theorem_statement=THEOREM_STATEMENT,
        theorem_domain=(
            f"F_{p}^* with primitive root {root}; zero is a separate "
            "fixed point outside exponent/CRT coordinates"
        ),
        theorem_applies_to_all_pivots=True,
        executable_witnesses_are_exhaustive_over_all_pivots=(
            len(pivots) == order
        ),
        root_choice_invariance_theorem_applied=True,
        alternate_root_controls_exhaustive=False,
        preflight=preflight,
        power_coordinate_bijection_verified=power_bijection,
        crt_round_trip_verified=crt_round_trip,
        crt_tuple_count=component_count,
        component_factorization_creates_extra_states=(
            component_count != order
        ),
        shared_pivot_state_count=order,
        layer_count=preflight.layer_count,
        conditional_independent_layer_control_tuple_count=layer_states,
        conditional_independent_layer_control_information_bits=(
            layer_information
        ),
        conditional_independent_layer_min_fixed_code_bits=math.ceil(
            layer_information
        ),
        independent_layer_addressability_assumed_not_verified=True,
        distinguishable_cross_layer_states_measured=False,
        independent_layers_enumerated=False,
        zero_excluded_from_exponent_coordinates=True,
        zero_fixed_point_verified=zero_fixed_all,
        legendre_stabilizer_size=order // 2,
        legendre_effective_orbit_size=2,
        factor_two_coordinate_is_nonzero_polarity_bit=(
            factor_two_polarity
        ),
        shifted_legendre_center_transport_verified=shifted_center_all,
        pivot_witnesses=tuple(witnesses),
        random_control=random_control,
        affine_state_upper_bound=p * order,
        affine_conjugation_identity_verified=affine_conjugation,
        affine_closure_theorem_applied=True,
        mixed_permutation_word_runtime_examples_exhaustive=False,
        kill_criteria=(
            KillCriterion(
                name="flat_permutation_relabeling",
                triggered=all_flat,
                evidence=(
                    "Every bounded witness is the ordinary exponent "
                    "permutation n->n-t; the theorem covers every t and "
                    "all primitive-root coordinate choices."
                ),
            ),
            KillCriterion(
                name="crt_components_add_no_states",
                triggered=(component_count == order and crt_round_trip),
                evidence=(
                    "The CRT tuple count equals p-1 and exhaustive round "
                    "trip maps each tuple to one flat exponent."
                ),
            ),
            KillCriterion(
                name="legendre_polarity_plus_fixed_zero",
                triggered=(
                    all_polarity
                    and zero_fixed_all
                    and factor_two_polarity
                    and shifted_center_all
                ),
                evidence=(
                    "On F_p^* the effect is one parity bit; repo L(0)=+1 "
                    "remains fixed, so odd pivots are -L+2e_0. Shifted "
                    "centers are transported rather than creating a new "
                    "carrier family."
                ),
            ),
            KillCriterion(
                name="mixed_rotation_pivot_layering",
                triggered=affine_conjugation,
                evidence=(
                    "Rotation/multiplication words reduce to one affine "
                    "map x->a*x+b with at most p(p-1) states. This is an "
                    "algebraic closure result; runtime examples are not "
                    "exhaustive."
                ),
            ),
            KillCriterion(
                name="random_or_additive_control_matches_utility",
                triggered=False,
                evidence=(
                    "The random conjugacy control is matched only as a "
                    "permutation family; utility and implementation cost "
                    "were not tested."
                ),
            ),
        ),
        arbitrary_learned_payload_polarity_degeneracy_claimed=False,
        learned_payload_utility_tested=False,
        implementation_cost_tested=False,
        performance_benchmarked=False,
        novelty_claimed=False,
        limitations=(
            "Executable pivot witnesses are bounded samples; the all-t "
            "result is theorem-level algebra, not exhaustive pair testing.",
            "One alternate primitive root is checked executably; invariance "
            "across every primitive-root choice is theorem-level.",
            "No arbitrary or learned payload inherits the Legendre "
            "polarity degeneracy merely from this carrier result.",
            "The random conjugacy is permutation-matched only, not "
            "compute-, storage-, latency-, or utility-matched.",
            "The 4690^L value is only the conditional count of control "
            "tuples if layers are independently addressable; independence "
            "and distinguishability are neither verified nor enumerated.",
            "The affine collapse covers sequential pure coordinate "
            "permutations on one ring; independent per-layer affine maps "
            "still multiply the control state count, and runtime word "
            "examples are not exhaustive.",
            "Replacement operators are outside this screen; future tests "
            "must stratify disjoint versus overlapping support.",
        ),
    )


__all__ = [
    "AffineMap",
    "CRTBudget",
    "CRTPreflight",
    "DEFAULT_PIVOT_EXPONENTS",
    "EVIDENCE_STATUS",
    "KillCriterion",
    "PRWC1Report",
    "PRWC1ResourceError",
    "PRWC1ValidationError",
    "PivotWitness",
    "RandomConjugacyControl",
    "TARGET_CRT_FACTORS",
    "TARGET_ORDER",
    "TARGET_PRIME",
    "TARGET_PRIMITIVE_ROOT",
    "THEOREM_STATEMENT",
    "crt_coordinates",
    "crt_reconstruct",
    "estimate_crt_pivot_resources",
    "preflight_crt_pivot_screen",
    "reduce_affine_word",
    "run_prw_c1_redundancy_screen",
]
