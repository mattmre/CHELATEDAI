"""Exact leading-shell audit for the frozen PRW-T1 bank.

The PRW-T1 protocol normalizes the wrong-type tie-or-better event union by the
56 minimum-distance competitors in the two-type, eight-layer, ``typed16`` bank
and separately requires every farther-shell pairwise term to be little-o of
that quantity.  The auxiliary farther-shell condition is false: eight
additional competitors are always only four Hamming units farther away.

The repaired PRW-T1R leading term therefore contains 64 states.  The exact
shell and overlap certificates in this module support the analytic proof
recorded in the method-development protocol: the other shells have a linear
distance gap, and every one of the 2,016 leading-event pairs has a strictly
larger joint large-deviation rate than a single leading event.  Bonferroni then
gives the reconditioned tie-as-error event-union asymptotic.  The accompanying
proof supplies an explicit Hamming-isometry/regular-action lemma transferring
the shell and overlap fields from the canonical state to all 32p transmitted
states in this frozen bank. ``analyze_canonical_first_decoder`` preserves the
original type-zero diagnostic. ``analyze_type_one_canonical_first_decoder``
binds the adverse inclusive-tie rule for type-one truth, and
``analyze_all_states_canonical_first_decoder`` freezes the transmitted-type
mixture at 50/50. Nothing here makes a retrieval, production, or novelty claim.
"""

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Tuple


HARD_MAX_PRIME = 100_003
HARD_MAX_TAIL_TERMS = 1_000_000
HARD_MAX_INTERSECTION_PRIME = 31
HARD_MAX_CLOSED_FORM_INTERSECTION_PRIME = 127


class LeadingShellValidationError(ValueError):
    """Raised when a requested leading-shell certificate is malformed."""


class LeadingShellResourceError(RuntimeError):
    """Raised before a requested numerical tail calculation is too large."""


@dataclass(frozen=True)
class DistanceShell:
    """One exact wrong-type distance shell."""

    distance: int
    multiplicity: int
    sources: Tuple[str, ...]


@dataclass(frozen=True)
class LeadingOverlapClass:
    """One exact pair class among the 64 leading wrong-type events."""

    left_shell: str
    right_shell: str
    left_distance: int
    right_distance: int
    disagreement_intersection: int
    pair_multiplicity: int


@dataclass(frozen=True)
class LeadingShellAnalysis:
    """Exact shell certificate plus stable finite-probability diagnostics."""

    prime: int
    crossover: float
    distance_shells: Tuple[DistanceShell, ...]
    minimum_distance: int
    nearest_multiplicity: int
    adjacent_distance: int
    adjacent_multiplicity: int
    adjacent_to_nearest_ratio: float
    adjacent_to_nearest_asymptotic_limit: float
    old_nearest_normalized_event_union_limit: float
    original_farther_to_nearest_ratio: float
    reconditioned_remainder_to_leading_ratio: float
    log_nearest_term: float
    log_reconditioned_leading_term: float
    original_prw_t1_farther_little_o_condition: bool
    original_prw_t1_status: str
    reconditioned_hypothesis_id: str
    reconditioned_hypothesis_status: str
    event_union_scope: str
    nonleading_remainder_status: str
    leading_pair_intersection_status: str
    final_decoder_status: str
    evidence_mode: str
    promotion_eligible: bool


@dataclass(frozen=True)
class LeadingIntersectionDiagnostic:
    """Finite exact pair-intersection screen for the 64 leading events."""

    prime: int
    crossover: float
    leading_state_count: int
    nearest_nearest_to_leading_ratio: float
    nearest_adjacent_to_leading_ratio: float
    adjacent_adjacent_to_leading_ratio: float
    all_pair_intersections_to_leading_ratio: float
    asymptotic_claim: bool
    evidence_mode: str


@dataclass(frozen=True)
class CanonicalFirstDecoderAnalysis:
    """Finite diagnostics for the abstract canonical-first exact-Hamming rule."""

    prime: int
    crossover: float
    transmitted_type: int
    tie_policy: str
    correct_type_distance_shells: Tuple[DistanceShell, ...]
    finite_strict_to_inclusive_leading_ratio: float
    strict_to_inclusive_asymptotic_limit: float
    correct_type_interference_to_strict_leading_upper_ratio: float
    hypothesis_id: str
    hypothesis_status: str
    all_transmitted_states_status: str
    evidence_mode: str
    promotion_eligible: bool


@dataclass(frozen=True)
class TypeOneDecoderAnalysis:
    """Finite diagnostics for type-one truth under canonical-first ties."""

    prime: int
    crossover: float
    transmitted_type: int
    tie_policy: str
    correct_type_distance_shells: Tuple[DistanceShell, ...]
    finite_inclusive_to_inclusive_leading_ratio: float
    inclusive_to_inclusive_asymptotic_limit: float
    correct_type_interference_to_inclusive_leading_upper_ratio: float
    hypothesis_id: str
    hypothesis_status: str
    evidence_mode: str
    promotion_eligible: bool


@dataclass(frozen=True)
class AllStatesDecoderAnalysis:
    """Frozen 50/50 transmitted-type analysis for the production tie rule."""

    prime: int
    crossover: float
    transmitted_type_weights: Tuple[float, float]
    tie_policy: str
    type_zero: CanonicalFirstDecoderAnalysis
    type_one: TypeOneDecoderAnalysis
    finite_balanced_to_inclusive_leading_ratio: float
    balanced_to_inclusive_asymptotic_limit: float
    correct_type_interference_to_balanced_leading_upper_ratio: float
    hypothesis_id: str
    hypothesis_status: str
    transmitted_state_scope: str
    evidence_mode: str
    promotion_eligible: bool


def _plain_int(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LeadingShellValidationError("{} must be an integer".format(name))
    if value < minimum:
        raise LeadingShellValidationError("{} must be at least {}".format(name, minimum))
    return value


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


def _validated_prime(value: object) -> int:
    prime = _plain_int(value, "prime", 11)
    if prime > HARD_MAX_PRIME:
        raise LeadingShellResourceError(
            "prime exceeds the immutable exact-shell ceiling: {} > {}".format(prime, HARD_MAX_PRIME)
        )
    if not _is_prime_exact(prime):
        raise LeadingShellValidationError("prime must be prime")
    if prime % 4 != 3:
        raise LeadingShellValidationError("prime must be congruent to 3 modulo 4")
    return prime


def _validated_crossover(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LeadingShellValidationError("crossover must be numeric")
    crossover = float(value)
    if not math.isfinite(crossover) or not 0.0 < crossover < 0.5:
        raise LeadingShellValidationError("crossover must be finite and strictly between 0 and 0.5")
    return crossover


def exact_distance_shells(prime: int) -> Tuple[DistanceShell, ...]:
    """Return the complete wrong-type distance spectrum without a codebook.

    For the frozen signatures ``(0,...,0)`` and ``(0,1,...,7)``, exactly eight
    global shifts align one layer.  At each aligned shift, seven balanced
    ``RM(1,3)`` masks with a positive aligned coordinate give the 56 nearest
    states, while the all-positive mask gives one state at distance
    ``d_min + 4``.  The remaining aligned and unaligned masks give the other
    five formula shells below.
    """

    p = _validated_prime(prime)
    minimum = (7 * p - 1) // 2
    raw_shells = (
        (minimum, 56, "aligned_balanced_positive"),
        (minimum + 4, 8, "aligned_all_positive"),
        (4 * p - 4, p - 8, "unaligned_all_negative"),
        (4 * p, 14 * (p - 8), "unaligned_balanced"),
        (4 * p + 4, p - 8, "unaligned_all_positive"),
        ((9 * p - 7) // 2, 8, "aligned_all_negative"),
        ((9 * p + 1) // 2, 56, "aligned_balanced_negative"),
    )
    aggregated: Dict[int, Tuple[int, Tuple[str, ...]]] = {}
    for distance, multiplicity, source in raw_shells:
        prior_multiplicity, prior_sources = aggregated.get(distance, (0, ()))
        aggregated[distance] = (
            prior_multiplicity + multiplicity,
            prior_sources + (source,),
        )
    shells = tuple(
        DistanceShell(distance, multiplicity, sources)
        for distance, (multiplicity, sources) in sorted(aggregated.items())
    )
    if sum(shell.multiplicity for shell in shells) != 16 * p:
        raise AssertionError("exact shell multiplicities must cover 16p states")
    return shells


def distance_spectrum(prime: int) -> Dict[int, int]:
    """Return the exact distance-to-multiplicity map."""

    return {shell.distance: shell.multiplicity for shell in exact_distance_shells(prime)}


def exact_correct_type_distance_shells(prime: int) -> Tuple[DistanceShell, ...]:
    """Return all nontransmitted, correct-type distances for the canonical state."""

    p = _validated_prime(prime)
    shells = (
        DistanceShell(4 * p - 4, p - 1, ("same_type_shifted_all_negative",)),
        DistanceShell(4 * p, 14 * p, ("same_type_balanced",)),
        DistanceShell(4 * p + 4, p - 1, ("same_type_shifted_all_positive",)),
        DistanceShell(8 * p, 1, ("same_type_unshifted_all_negative",)),
    )
    if sum(shell.multiplicity for shell in shells) != 16 * p - 1:
        raise AssertionError("correct-type shell multiplicities must cover 16p-1 states")
    return shells


def leading_overlap_classes(prime: int) -> Tuple[LeadingOverlapClass, ...]:
    """Return every exact pair orbit among the 56+8 leading events."""

    p = _validated_prime(prime)
    nearest = (7 * p - 1) // 2
    adjacent = nearest + 4
    common_high = (3 * p + 3) // 2
    classes = (
        LeadingOverlapClass(
            "nearest",
            "nearest",
            nearest,
            nearest,
            (3 * p - 5) // 2,
            84,
        ),
        LeadingOverlapClass(
            "nearest",
            "nearest",
            nearest,
            nearest,
            (3 * p - 1) // 2,
            1_344,
        ),
        LeadingOverlapClass(
            "nearest",
            "nearest",
            nearest,
            nearest,
            common_high,
            112,
        ),
        LeadingOverlapClass(
            "nearest",
            "adjacent",
            nearest,
            adjacent,
            common_high,
            448,
        ),
        LeadingOverlapClass(
            "adjacent",
            "adjacent",
            adjacent,
            adjacent,
            common_high,
            28,
        ),
    )
    if sum(value.pair_multiplicity for value in classes) != 64 * 63 // 2:
        raise AssertionError("leading overlap classes must cover C(64, 2)")
    return classes


def _log_pairwise_tail(distance: int, crossover: float) -> float:
    """Stable log of the tie-as-error BSC pairwise tail."""

    trials = _plain_int(distance, "distance", 1)
    q = _validated_crossover(crossover)
    threshold = (trials + 1) // 2
    tail_terms = trials - threshold + 1
    if tail_terms > HARD_MAX_TAIL_TERMS:
        raise LeadingShellResourceError(
            "binomial tail exceeds the immutable term ceiling: {} > {}".format(tail_terms, HARD_MAX_TAIL_TERMS)
        )
    log_probability = (
        math.lgamma(trials + 1)
        - math.lgamma(threshold + 1)
        - math.lgamma(trials - threshold + 1)
        + threshold * math.log(q)
        + (trials - threshold) * math.log1p(-q)
    )
    relative_term = 1.0
    relative_sum = 1.0
    odds = q / (1.0 - q)
    for flips in range(threshold, trials):
        relative_term *= (float(trials - flips) / float(flips + 1)) * odds
        relative_sum += relative_term
    return log_probability + math.log(relative_sum)


def _log_pairwise_strict_tail(distance: int, crossover: float) -> float:
    """Stable log probability that a competitor strictly beats at distance d."""

    trials = _plain_int(distance, "distance", 1)
    q = _validated_crossover(crossover)
    threshold = trials // 2 + 1
    tail_terms = trials - threshold + 1
    if tail_terms > HARD_MAX_TAIL_TERMS:
        raise LeadingShellResourceError(
            "binomial tail exceeds the immutable term ceiling: {} > {}".format(
                tail_terms,
                HARD_MAX_TAIL_TERMS,
            )
        )
    log_probability = (
        math.lgamma(trials + 1)
        - math.lgamma(threshold + 1)
        - math.lgamma(trials - threshold + 1)
        + threshold * math.log(q)
        + (trials - threshold) * math.log1p(-q)
    )
    relative_term = 1.0
    relative_sum = 1.0
    odds = q / (1.0 - q)
    for flips in range(threshold, trials):
        relative_term *= (float(trials - flips) / float(flips + 1)) * odds
        relative_sum += relative_term
    return log_probability + math.log(relative_sum)


def _logsumexp(values: Iterable[float]) -> float:
    materialized = tuple(values)
    if not materialized:
        return float("-inf")
    maximum = max(materialized)
    return maximum + math.log(sum(math.exp(value - maximum) for value in materialized))


def analyze_leading_shell(
    prime: int,
    crossover: float = 0.20,
) -> LeadingShellAnalysis:
    """Falsify PRW-T1's farther-shell clause and measure repaired terms."""

    p = _validated_prime(prime)
    q = _validated_crossover(crossover)
    shells = exact_distance_shells(p)
    minimum = (7 * p - 1) // 2
    adjacent = minimum + 4
    spectrum = {shell.distance: shell for shell in shells}
    if spectrum[minimum].multiplicity != 56:
        raise AssertionError("nearest multiplicity must be 56")
    if spectrum[adjacent].multiplicity != 8:
        raise AssertionError("adjacent leading-shell multiplicity must be 8")

    log_tails = {shell.distance: _log_pairwise_tail(shell.distance, q) for shell in shells}
    log_nearest = math.log(56.0) + log_tails[minimum]
    log_adjacent = math.log(8.0) + log_tails[adjacent]
    log_leading = _logsumexp((log_nearest, log_adjacent))
    log_original_farther = _logsumexp(
        math.log(float(shell.multiplicity)) + log_tails[shell.distance] for shell in shells if shell.distance != minimum
    )
    log_reconditioned_remainder = _logsumexp(
        math.log(float(shell.multiplicity)) + log_tails[shell.distance]
        for shell in shells
        if shell.distance not in (minimum, adjacent)
    )
    adjacent_ratio = math.exp(log_adjacent - log_nearest)
    asymptotic_ratio = ((4.0 * q * (1.0 - q)) ** 2) / 7.0

    return LeadingShellAnalysis(
        prime=p,
        crossover=q,
        distance_shells=shells,
        minimum_distance=minimum,
        nearest_multiplicity=56,
        adjacent_distance=adjacent,
        adjacent_multiplicity=8,
        adjacent_to_nearest_ratio=adjacent_ratio,
        adjacent_to_nearest_asymptotic_limit=asymptotic_ratio,
        old_nearest_normalized_event_union_limit=1.0 + asymptotic_ratio,
        original_farther_to_nearest_ratio=math.exp(log_original_farther - log_nearest),
        reconditioned_remainder_to_leading_ratio=math.exp(log_reconditioned_remainder - log_leading),
        log_nearest_term=log_nearest,
        log_reconditioned_leading_term=log_leading,
        original_prw_t1_farther_little_o_condition=False,
        original_prw_t1_status=("NEAREST_ONLY_TIE_AS_ERROR_EVENT_UNION_FALSIFIED"),
        reconditioned_hypothesis_id="PRW-T1R-TE-EVENT-UNION",
        reconditioned_hypothesis_status=("CLOSED_INTERNAL_ANALYTIC_SCOPE"),
        event_union_scope=("ALL_FROZEN_BANK_STATES_WRONG_TYPE_COMPETITOR_TIE_OR_BETTER_UNION"),
        nonleading_remainder_status=("PROVED_LITTLE_O_BY_LINEAR_DISTANCE_GAP"),
        leading_pair_intersection_status=("PROVED_LITTLE_O_BY_STRICT_JOINT_LARGE_DEVIATION_RATE"),
        final_decoder_status=("EXACT_HAMMING_TIE_COROLLARIES_COMPLETE_" "CURRENT_FLOAT_FFT_LINKAGE_FAILED"),
        evidence_mode="NON_CONFIRMATORY_METHOD_DEV",
        promotion_eligible=False,
    )


def analyze_leading_intersections(
    prime: int,
    crossover: float = 0.20,
) -> LeadingIntersectionDiagnostic:
    """Exactly screen all pair intersections among the 64 leading events.

    This deliberately bounded diagnostic cross-checks the reconditioned
    hypothesis at ``p <= 31``.  It is finite evidence, not an asymptotic proof.
    Imports are local so the closed-form shell certificate itself stays free of
    array allocation.
    """

    p = _validated_prime(prime)
    q = _validated_crossover(crossover)
    if p > HARD_MAX_INTERSECTION_PRIME:
        raise LeadingShellResourceError(
            "intersection diagnostic exceeds the immutable prime ceiling: {} > {}".format(
                p, HARD_MAX_INTERSECTION_PRIME
            )
        )

    from collections import Counter
    import itertools

    import numpy as np

    from prime_ring_intersection import (
        bsc_pairwise_error,
        build_legendre_mask_bank,
        pair_event_intersection_probability,
    )
    from prime_ring_waypoint import policy_masks

    signatures = np.mod(
        np.arange(2, dtype=np.int64)[:, np.newaxis] * np.arange(8, dtype=np.int64)[np.newaxis, :],
        p,
    )
    bank = build_legendre_mask_bank(
        p,
        signatures,
        policy_masks("typed16", 8),
    )
    planted = bank.templates[0]
    competitors = tuple(index for index, state in enumerate(bank.states) if state.type_id != 0)
    distances = {index: int(np.count_nonzero(planted != bank.templates[index])) for index in competitors}
    if {
        distance: sum(1 for observed in distances.values() if observed == distance)
        for distance in set(distances.values())
    } != distance_spectrum(p):
        raise AssertionError("materialized bank must match the exact spectrum")

    minimum = (7 * p - 1) // 2
    adjacent = minimum + 4
    leading = tuple(index for index in competitors if distances[index] in (minimum, adjacent))
    if len(leading) != 64:
        raise AssertionError("reconditioned leading shell must have 64 states")
    leading_term = sum(bsc_pairwise_error(distances[index], q) for index in leading)
    grouped = {
        "nearest_nearest": 0.0,
        "nearest_adjacent": 0.0,
        "adjacent_adjacent": 0.0,
    }
    observed_classes = Counter()
    for left, right in itertools.combinations(leading, 2):
        left_distance = distances[left]
        right_distance = distances[right]
        overlap = int(np.count_nonzero((planted != bank.templates[left]) & (planted != bank.templates[right])))
        probability = pair_event_intersection_probability(
            left_distance,
            right_distance,
            overlap,
            q,
        )
        if left_distance == minimum and right_distance == minimum:
            group = "nearest_nearest"
            left_shell = "nearest"
            right_shell = "nearest"
        elif left_distance == adjacent and right_distance == adjacent:
            group = "adjacent_adjacent"
            left_shell = "adjacent"
            right_shell = "adjacent"
        else:
            group = "nearest_adjacent"
            left_shell = "nearest"
            right_shell = "adjacent"
        grouped[group] += probability
        observed_classes[
            (
                left_shell,
                right_shell,
                minimum if left_shell == "nearest" else adjacent,
                adjacent if right_shell == "adjacent" else minimum,
                overlap,
            )
        ] += 1

    expected_classes = Counter(
        {
            (
                value.left_shell,
                value.right_shell,
                value.left_distance,
                value.right_distance,
                value.disagreement_intersection,
            ): value.pair_multiplicity
            for value in leading_overlap_classes(p)
        }
    )
    if observed_classes != expected_classes:
        raise AssertionError("materialized leading-event pairs must match exact orbit classes")

    return LeadingIntersectionDiagnostic(
        prime=p,
        crossover=q,
        leading_state_count=len(leading),
        nearest_nearest_to_leading_ratio=(grouped["nearest_nearest"] / leading_term),
        nearest_adjacent_to_leading_ratio=(grouped["nearest_adjacent"] / leading_term),
        adjacent_adjacent_to_leading_ratio=(grouped["adjacent_adjacent"] / leading_term),
        all_pair_intersections_to_leading_ratio=(sum(grouped.values()) / leading_term),
        asymptotic_claim=False,
        evidence_mode="FINITE_PRW_T1R_DIAGNOSTIC_ONLY",
    )


def analyze_leading_intersections_closed_form(
    prime: int,
    crossover: float = 0.20,
) -> LeadingIntersectionDiagnostic:
    """Screen the five certified pair classes without materializing a bank.

    This remains a finite diagnostic rather than an asymptotic proof.  Its
    separate immutable ceiling keeps the exact binomial tables bounded while
    making the protocol's ``p=43`` cross-check directly reproducible.
    """

    p = _validated_prime(prime)
    q = _validated_crossover(crossover)
    if p > HARD_MAX_CLOSED_FORM_INTERSECTION_PRIME:
        raise LeadingShellResourceError(
            "closed-form intersection diagnostic exceeds the immutable prime ceiling: {} > {}".format(
                p,
                HARD_MAX_CLOSED_FORM_INTERSECTION_PRIME,
            )
        )

    from prime_ring_intersection import (
        bsc_pairwise_error,
        pair_event_intersection_probability,
    )

    minimum = (7 * p - 1) // 2
    adjacent = minimum + 4
    leading_term = 56.0 * bsc_pairwise_error(minimum, q) + 8.0 * bsc_pairwise_error(adjacent, q)
    if not math.isfinite(leading_term) or leading_term <= 0.0:
        raise LeadingShellResourceError(
            "closed-form intersection leading term underflowed; use a smaller prime or less extreme crossover"
        )
    grouped = {
        "nearest_nearest": 0.0,
        "nearest_adjacent": 0.0,
        "adjacent_adjacent": 0.0,
    }
    for value in leading_overlap_classes(p):
        if value.left_shell == "nearest" and value.right_shell == "nearest":
            group = "nearest_nearest"
        elif value.left_shell == "adjacent" and value.right_shell == "adjacent":
            group = "adjacent_adjacent"
        else:
            group = "nearest_adjacent"
        grouped[group] += value.pair_multiplicity * pair_event_intersection_probability(
            value.left_distance,
            value.right_distance,
            value.disagreement_intersection,
            q,
        )

    return LeadingIntersectionDiagnostic(
        prime=p,
        crossover=q,
        leading_state_count=64,
        nearest_nearest_to_leading_ratio=(grouped["nearest_nearest"] / leading_term),
        nearest_adjacent_to_leading_ratio=(grouped["nearest_adjacent"] / leading_term),
        adjacent_adjacent_to_leading_ratio=(grouped["adjacent_adjacent"] / leading_term),
        all_pair_intersections_to_leading_ratio=(sum(grouped.values()) / leading_term),
        asymptotic_claim=False,
        evidence_mode="FINITE_PRW_T1R_CLOSED_FORM_DIAGNOSTIC_ONLY",
    )


def _canonical_first_decoder_terms(
    prime: int,
    crossover: float,
) -> Tuple[
    int,
    float,
    float,
    float,
    float,
    Tuple[DistanceShell, ...],
]:
    """Return validated shared log terms for the canonical-first decoder."""

    p = _validated_prime(prime)
    q = _validated_crossover(crossover)
    minimum = (7 * p - 1) // 2
    adjacent = minimum + 4
    log_inclusive_leading = _logsumexp(
        (
            math.log(56.0) + _log_pairwise_tail(minimum, q),
            math.log(8.0) + _log_pairwise_tail(adjacent, q),
        )
    )
    log_strict_leading = _logsumexp(
        (
            math.log(56.0) + _log_pairwise_strict_tail(minimum, q),
            math.log(8.0) + _log_pairwise_strict_tail(adjacent, q),
        )
    )
    correct_type_shells = exact_correct_type_distance_shells(p)
    log_correct_type_union = _logsumexp(
        math.log(float(shell.multiplicity)) + _log_pairwise_tail(shell.distance, q) for shell in correct_type_shells
    )
    if not log_strict_leading < log_inclusive_leading:
        raise AssertionError("strict leading probability must be below inclusive leading probability")
    return (
        p,
        q,
        log_inclusive_leading,
        log_strict_leading,
        log_correct_type_union,
        correct_type_shells,
    )


def analyze_canonical_first_decoder(
    prime: int,
    crossover: float = 0.20,
) -> CanonicalFirstDecoderAnalysis:
    """Derive the exact-Hamming canonical-first corollary for type-zero truth.

    Under exact Hamming scoring, a declared lowest-type-ID rule resolves an
    exact cross-type tie in favor of type zero. The current float-FFT scoring
    path does not preserve every mathematical tie and is intentionally outside
    this helper's claim. Other correct-type states have a linear distance gap,
    so their union bound is little-o of the strict 64-state leading term.
    """

    (
        p,
        q,
        log_inclusive_leading,
        log_strict_leading,
        log_correct_type_union,
        correct_type_shells,
    ) = _canonical_first_decoder_terms(
        prime,
        crossover,
    )

    return CanonicalFirstDecoderAnalysis(
        prime=p,
        crossover=q,
        transmitted_type=0,
        tie_policy="LOWEST_CANONICAL_TYPE_ID_WINS_EXACT_TIE",
        correct_type_distance_shells=correct_type_shells,
        finite_strict_to_inclusive_leading_ratio=math.exp(log_strict_leading - log_inclusive_leading),
        strict_to_inclusive_asymptotic_limit=(q / (1.0 - q)),
        correct_type_interference_to_strict_leading_upper_ratio=math.exp(log_correct_type_union - log_strict_leading),
        hypothesis_id="PRW-T1D-CANONICAL-FIRST-TYPE0",
        hypothesis_status="ANALYTIC_TIE_COROLLARY_COMPLETE_EXACT_HAMMING_ONLY",
        all_transmitted_states_status=("ANALYTIC_ALL_STATES_COROLLARY_CURRENT_FLOAT_FFT_LINKAGE_FAILED"),
        evidence_mode="NON_CONFIRMATORY_METHOD_DEV",
        promotion_eligible=False,
    )


def analyze_type_one_canonical_first_decoder(
    prime: int,
    crossover: float = 0.20,
) -> TypeOneDecoderAnalysis:
    """Derive the exact-Hamming canonical-first corollary for type-one truth.

    Under exact Hamming scoring, the lowest canonical type ID wins an exact
    score tie. Therefore a type-zero competitor that ties type-one truth already
    causes class error. The current float-FFT scoring path is outside this
    helper's claim because it does not preserve every mathematical tie.
    """

    (
        p,
        q,
        log_inclusive_leading,
        _log_strict_leading,
        log_correct_type_union,
        correct_type_shells,
    ) = _canonical_first_decoder_terms(
        prime,
        crossover,
    )
    return TypeOneDecoderAnalysis(
        prime=p,
        crossover=q,
        transmitted_type=1,
        tie_policy="LOWEST_CANONICAL_TYPE_ID_WINS_EXACT_TIE",
        correct_type_distance_shells=correct_type_shells,
        finite_inclusive_to_inclusive_leading_ratio=1.0,
        inclusive_to_inclusive_asymptotic_limit=1.0,
        correct_type_interference_to_inclusive_leading_upper_ratio=math.exp(
            log_correct_type_union - log_inclusive_leading
        ),
        hypothesis_id="PRW-T1D-CANONICAL-FIRST-TYPE1",
        hypothesis_status=("ANALYTIC_TIE_COROLLARY_COMPLETE_EXACT_HAMMING_ONLY"),
        evidence_mode="NON_CONFIRMATORY_METHOD_DEV",
        promotion_eligible=False,
    )


def analyze_all_states_canonical_first_decoder(
    prime: int,
    crossover: float = 0.20,
) -> AllStatesDecoderAnalysis:
    """Return the frozen 50/50 type-mixture decoder asymptotic.

    Conditional on type zero, the leading term is strict because exact ties
    preserve the correct canonical type. Conditional on type one, it is
    inclusive because those same ties select wrong type zero. With immutable
    transmitted-type weights ``(1/2, 1/2)``, the balanced leading term is
    ``(S_p + B_p) / 2`` and its ratio to ``B_p`` tends to
    ``1 / (2 * (1 - q))``.
    """

    (
        p,
        q,
        log_inclusive_leading,
        log_strict_leading,
        log_correct_type_union,
        _correct_type_shells,
    ) = _canonical_first_decoder_terms(
        prime,
        crossover,
    )
    type_zero = analyze_canonical_first_decoder(p, q)
    type_one = analyze_type_one_canonical_first_decoder(p, q)
    strict_to_inclusive = math.exp(log_strict_leading - log_inclusive_leading)
    finite_balanced_ratio = 0.5 * (strict_to_inclusive + 1.0)
    balanced_limit = 1.0 / (2.0 * (1.0 - q))
    log_balanced_leading = _logsumexp((log_strict_leading, log_inclusive_leading)) - math.log(2.0)
    if not strict_to_inclusive < finite_balanced_ratio < 1.0:
        raise AssertionError("balanced leading ratio must lie between type-zero and type-one ratios")
    return AllStatesDecoderAnalysis(
        prime=p,
        crossover=q,
        transmitted_type_weights=(0.5, 0.5),
        tie_policy="LOWEST_CANONICAL_TYPE_ID_WINS_EXACT_TIE",
        type_zero=type_zero,
        type_one=type_one,
        finite_balanced_to_inclusive_leading_ratio=finite_balanced_ratio,
        balanced_to_inclusive_asymptotic_limit=balanced_limit,
        correct_type_interference_to_balanced_leading_upper_ratio=math.exp(
            log_correct_type_union - log_balanced_leading
        ),
        hypothesis_id="PRW-T1D-ALL-STATES",
        hypothesis_status=("ANALYTIC_TIE_COROLLARY_COMPLETE_EXACT_HAMMING_ONLY_" "CURRENT_FLOAT_FFT_LINKAGE_FAILED"),
        transmitted_state_scope=("EVERY_FROZEN_BANK_STATE_WITH_HALF_TOTAL_MASS_PER_TYPE"),
        evidence_mode="NON_CONFIRMATORY_METHOD_DEV",
        promotion_eligible=False,
    )


__all__ = [
    "AllStatesDecoderAnalysis",
    "CanonicalFirstDecoderAnalysis",
    "DistanceShell",
    "HARD_MAX_CLOSED_FORM_INTERSECTION_PRIME",
    "HARD_MAX_INTERSECTION_PRIME",
    "HARD_MAX_PRIME",
    "HARD_MAX_TAIL_TERMS",
    "LeadingIntersectionDiagnostic",
    "LeadingOverlapClass",
    "LeadingShellAnalysis",
    "LeadingShellResourceError",
    "LeadingShellValidationError",
    "TypeOneDecoderAnalysis",
    "analyze_all_states_canonical_first_decoder",
    "analyze_canonical_first_decoder",
    "analyze_leading_intersections",
    "analyze_leading_intersections_closed_form",
    "analyze_leading_shell",
    "analyze_type_one_canonical_first_decoder",
    "distance_spectrum",
    "exact_correct_type_distance_shells",
    "exact_distance_shells",
    "leading_overlap_classes",
]
