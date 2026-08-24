"""Exact transmitted-state orbit certificate for the narrow PRW-T1 bank.

This module certifies only the two-type, eight-layer bank whose phase
signatures are ``(0, ..., 0)`` and ``(0, ..., 7)`` and whose masks are
``RM(1, 3)``.  It does not certify arbitrary distinct-signature controls,
asymptotics, decoder error, utility, or novelty.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from prime_ring_intersection import HypothesisState, LegendreMaskBank
from prime_ring_waypoint import legendre_carrier, policy_masks


HARD_MAX_ESTIMATED_BYTES = 512 * 1024 * 1024
HARD_MAX_SECONDS = 120.0
HARD_MAX_SYMBOL_CHECKS = 50_000_000
HARD_MAX_WORK_UNITS = 50_000_000
HARD_MAX_HYPOTHESES = 1_024
ALLOCATOR_RESERVE_BYTES = 1024 * 1024


class PRWOrbitValidationError(ValueError):
    """Raised when the claimed exact PRW-T1 structure is absent."""


class PRWOrbitResourceError(RuntimeError):
    """Raised before or during work that exceeds a declared hard limit."""


def _plain_int(value: object, name: str, minimum: int) -> int:
    if type(value) is not int:
        raise PRWOrbitValidationError(f"{name} must be a plain integer")
    if value < minimum:
        raise PRWOrbitValidationError(f"{name} must be at least {minimum}")
    if value.bit_length() > 63:
        raise PRWOrbitValidationError(f"{name} exceeds the 63-bit input limit")
    return value


def _positive_float(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise PRWOrbitValidationError(f"{name} must be a plain finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise PRWOrbitValidationError(f"{name} must be a finite positive number")
    return result


@dataclass(frozen=True)
class OrbitCertificateBudget:
    """Fail-closed limits for one orbit certificate."""

    max_estimated_bytes: int = HARD_MAX_ESTIMATED_BYTES
    max_seconds: float = HARD_MAX_SECONDS
    max_symbol_checks: int = HARD_MAX_SYMBOL_CHECKS
    max_work_units: int = HARD_MAX_WORK_UNITS
    max_hypotheses: int = HARD_MAX_HYPOTHESES

    def __post_init__(self) -> None:
        byte_limit = _plain_int(
            self.max_estimated_bytes,
            "max_estimated_bytes",
            1,
        )
        seconds = _positive_float(self.max_seconds, "max_seconds")
        checks = _plain_int(self.max_symbol_checks, "max_symbol_checks", 1)
        work_units = _plain_int(self.max_work_units, "max_work_units", 1)
        hypotheses = _plain_int(
            self.max_hypotheses,
            "max_hypotheses",
            1,
        )
        if byte_limit > HARD_MAX_ESTIMATED_BYTES:
            raise PRWOrbitValidationError(
                "max_estimated_bytes exceeds its hard ceiling"
            )
        if seconds > HARD_MAX_SECONDS:
            raise PRWOrbitValidationError("max_seconds exceeds its hard ceiling")
        if checks > HARD_MAX_SYMBOL_CHECKS:
            raise PRWOrbitValidationError(
                "max_symbol_checks exceeds its hard ceiling"
            )
        if work_units > HARD_MAX_WORK_UNITS:
            raise PRWOrbitValidationError(
                "max_work_units exceeds its hard ceiling"
            )
        if hypotheses > HARD_MAX_HYPOTHESES:
            raise PRWOrbitValidationError(
                "max_hypotheses exceeds its hard ceiling"
            )
        object.__setattr__(self, "max_estimated_bytes", byte_limit)
        object.__setattr__(self, "max_seconds", seconds)
        object.__setattr__(self, "max_symbol_checks", checks)
        object.__setattr__(self, "max_work_units", work_units)
        object.__setattr__(self, "max_hypotheses", hypotheses)


def _check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise PRWOrbitResourceError(
            "orbit certificate exceeded its wall-clock deadline"
        )


def preflight_prw_t1_transmitted_orbit(
    bank: LegendreMaskBank,
    budget: OrbitCertificateBudget = OrbitCertificateBudget(),
) -> Dict[str, Any]:
    """Return an allocation-free resource model or fail before certification."""

    if not isinstance(bank, LegendreMaskBank):
        raise PRWOrbitValidationError("bank must be LegendreMaskBank")
    if not isinstance(budget, OrbitCertificateBudget):
        raise PRWOrbitValidationError("budget must be OrbitCertificateBudget")
    if not isinstance(bank.templates, np.ndarray) or bank.templates.ndim != 2:
        raise PRWOrbitValidationError("bank templates must be a rank-two array")
    hypothesis_count = int(bank.templates.shape[0])
    coordinate_count = int(bank.templates.shape[1])
    mask_count = (
        int(bank.masks.shape[0])
        if isinstance(bank.masks, np.ndarray) and bank.masks.ndim == 2
        else 0
    )
    generator_symbol_checks = (
        hypothesis_count
        * coordinate_count
        * (1 + mask_count + 1 + 1)
    )
    template_alphabet_symbol_checks = hypothesis_count * coordinate_count
    canonical_symbol_checks = coordinate_count
    modeled_symbol_checks = (
        generator_symbol_checks
        + template_alphabet_symbol_checks
        + canonical_symbol_checks
    )
    # A work unit is a conservative scalar-equivalent operation, not a
    # throughput or timing claim.  Symbol comparisons stay separately exact.
    generator_execution_work_units = 3 * generator_symbol_checks
    template_alphabet_scan_work_units = (
        4 * template_alphabet_symbol_checks
    )
    canonical_scan_work_units = 2 * canonical_symbol_checks
    state_and_mapping_work_units = hypothesis_count * (
        32 + 16 * mask_count
    )
    normal_form_work_units = hypothesis_count * 12
    coordinate_permutation_work_units = coordinate_count * 16
    mask_structure_work_units = max(1, mask_count) * (
        max(1, int(bank.masks.shape[1]))
        if isinstance(bank.masks, np.ndarray) and bank.masks.ndim == 2
        else 1
    ) * 6
    estimated_work_units = (
        generator_execution_work_units
        + template_alphabet_scan_work_units
        + canonical_scan_work_units
        + state_and_mapping_work_units
        + normal_form_work_units
        + coordinate_permutation_work_units
        + mask_structure_work_units
    )
    input_bank_template_bytes = int(bank.templates.nbytes)
    input_bank_other_array_bytes = (
        int(bank.masks.nbytes)
        if isinstance(bank.masks, np.ndarray)
        else 0
    ) + (
        int(bank.phase_signatures.nbytes)
        if isinstance(bank.phase_signatures, np.ndarray)
        else 0
    )
    incremental_memory_components = (
        (
            "template_alphabet_scan_and_boolean_temporaries",
            4 * hypothesis_count * coordinate_count,
        ),
        (
            "state_lookup_and_expected_key_sets",
            hypothesis_count * 1024,
        ),
        (
            "normal_form_images_and_tuple_allowance",
            hypothesis_count * 512,
        ),
        (
            "generator_coordinate_and_mask_temporaries",
            coordinate_count * (64 + max(1, mask_count)),
        ),
        (
            "python_numpy_allocator_reserve",
            ALLOCATOR_RESERVE_BYTES,
        ),
    )
    estimated_incremental_peak_bytes = sum(
        size for _name, size in incremental_memory_components
    )
    estimated_co_resident_peak_bytes = (
        input_bank_template_bytes
        + input_bank_other_array_bytes
        + estimated_incremental_peak_bytes
    )
    reasons = []
    if hypothesis_count > budget.max_hypotheses:
        reasons.append("hypothesis count exceeds budget")
    if estimated_co_resident_peak_bytes > budget.max_estimated_bytes:
        reasons.append("estimated peak bytes exceed budget")
    if modeled_symbol_checks > budget.max_symbol_checks:
        reasons.append("modeled symbol checks exceed budget")
    if estimated_work_units > budget.max_work_units:
        reasons.append("estimated total work units exceed budget")
    if reasons:
        raise PRWOrbitResourceError("; ".join(reasons))
    return {
        "status": "ALLOWED",
        "arrays_allocated": False,
        "hypothesis_count": hypothesis_count,
        "coordinate_count": coordinate_count,
        "mask_count": mask_count,
        "generator_symbol_checks": generator_symbol_checks,
        "template_alphabet_symbol_checks": (
            template_alphabet_symbol_checks
        ),
        "canonical_symbol_checks": canonical_symbol_checks,
        "modeled_symbol_checks": modeled_symbol_checks,
        "generator_execution_work_units": (
            generator_execution_work_units
        ),
        "template_alphabet_scan_work_units": (
            template_alphabet_scan_work_units
        ),
        "canonical_scan_work_units": canonical_scan_work_units,
        "state_and_mapping_work_units": state_and_mapping_work_units,
        "normal_form_work_units": normal_form_work_units,
        "coordinate_permutation_work_units": (
            coordinate_permutation_work_units
        ),
        "mask_structure_work_units": mask_structure_work_units,
        "estimated_work_units": estimated_work_units,
        "input_bank_template_bytes": input_bank_template_bytes,
        "input_bank_other_array_bytes": input_bank_other_array_bytes,
        "incremental_memory_components": incremental_memory_components,
        "estimated_incremental_peak_bytes": (
            estimated_incremental_peak_bytes
        ),
        "estimated_co_resident_peak_bytes": (
            estimated_co_resident_peak_bytes
        ),
        "estimated_peak_bytes": estimated_co_resident_peak_bytes,
        "bank_construction_included": False,
        "live_input_bank_storage_included": True,
        "memory_model_is_process_rss_measurement": False,
        "process_rss_measured": False,
        "work_model_is_time_calibration": False,
        "budget": {
            "max_estimated_bytes": budget.max_estimated_bytes,
            "max_seconds": budget.max_seconds,
            "max_symbol_checks": budget.max_symbol_checks,
            "max_work_units": budget.max_work_units,
            "max_hypotheses": budget.max_hypotheses,
        },
    }


def _state_key(state: HypothesisState) -> Tuple[int, int, int]:
    if not isinstance(state, HypothesisState):
        raise PRWOrbitValidationError(
            "every bank state must be HypothesisState"
        )
    if (
        type(state.type_id) is not int
        or type(state.shift) is not int
        or type(state.mask_id) is not int
    ):
        raise PRWOrbitValidationError(
            "state identifiers must be plain integers"
        )
    return (state.type_id, state.shift, state.mask_id)


def _exact_structure(
    bank: LegendreMaskBank,
) -> Tuple[Dict[Tuple[int, int, int], int], Dict[Tuple[int, ...], int], int]:
    p = int(bank.prime)
    if type(bank.prime) is not int or p < 3 or p % 4 != 3:
        raise PRWOrbitValidationError(
            "prime must be a plain prime-length value congruent to 3 mod 4"
        )
    if type(bank.layers) is not int or bank.layers != 8:
        raise PRWOrbitValidationError("certificate requires exactly eight layers")
    expected_signatures = np.asarray(
        (
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 1, 2, 3, 4, 5, 6, 7),
        ),
        dtype=np.int64,
    )
    if (
        not isinstance(bank.phase_signatures, np.ndarray)
        or bank.phase_signatures.shape != (2, 8)
        or not np.array_equal(bank.phase_signatures, expected_signatures % p)
    ):
        raise PRWOrbitValidationError(
            "certificate requires the exact ordered two-type signatures"
        )
    expected_masks = policy_masks("typed16", 8).astype(np.int8)
    if (
        not isinstance(bank.masks, np.ndarray)
        or bank.masks.shape != (16, 8)
        or not np.all(np.isin(bank.masks, (-1, 1)))
    ):
        raise PRWOrbitValidationError(
            "certificate requires sixteen bipolar RM(1,3) masks"
        )
    mask_lookup = {
        tuple(int(value) for value in row): index
        for index, row in enumerate(bank.masks)
    }
    expected_mask_rows = {
        tuple(int(value) for value in row) for row in expected_masks
    }
    if set(mask_lookup) != expected_mask_rows or len(mask_lookup) != 16:
        raise PRWOrbitValidationError(
            "mask rows are not exactly the RM(1,3) group"
        )
    expected_shape = (32 * p, 8 * p)
    if (
        not isinstance(bank.templates, np.ndarray)
        or bank.templates.shape != expected_shape
        or bank.templates.dtype != np.dtype(np.int8)
        or not np.all(np.isin(bank.templates, (-1, 1)))
    ):
        raise PRWOrbitValidationError(
            "template bank has the wrong shape, dtype, or alphabet"
        )
    if len(bank.states) != 32 * p:
        raise PRWOrbitValidationError("state count is not the full 32p product")
    state_lookup: Dict[Tuple[int, int, int], int] = {}
    for index, state in enumerate(bank.states):
        key = _state_key(state)
        if (
            key[0] not in (0, 1)
            or not 0 <= key[1] < p
            or not 0 <= key[2] < 16
            or key in state_lookup
        ):
            raise PRWOrbitValidationError(
                "states are not a unique two-type/shift/mask product"
            )
        state_lookup[key] = index
    expected_keys = {
        (type_id, shift, mask_id)
        for type_id in range(2)
        for shift in range(p)
        for mask_id in range(16)
    }
    if set(state_lookup) != expected_keys:
        raise PRWOrbitValidationError("state product is incomplete")
    identity_mask_id = mask_lookup[(1, 1, 1, 1, 1, 1, 1, 1)]
    return state_lookup, mask_lookup, identity_mask_id


def certify_prw_t1_transmitted_orbit(
    bank: LegendreMaskBank,
    budget: OrbitCertificateBudget = OrbitCertificateBudget(),
) -> Dict[str, Any]:
    """Certify one regular 32p orbit and its disagreement equivariance."""

    started = time.monotonic()
    preflight = preflight_prw_t1_transmitted_orbit(bank, budget)
    deadline = started + budget.max_seconds
    _check_deadline(deadline)
    state_lookup, mask_lookup, identity_mask_id = _exact_structure(bank)
    p = int(bank.prime)
    layers = int(bank.layers)
    coordinates = int(bank.templates.shape[1])
    h = (7 * pow(2, -1, p)) % p

    carrier = legendre_carrier(p).astype(np.int8, copy=False)
    canonical_index = state_lookup[(0, 0, identity_mask_id)]
    canonical_expected = np.tile(carrier, layers)
    if not np.array_equal(
        bank.templates[canonical_index],
        canonical_expected,
    ):
        raise PRWOrbitValidationError(
            "canonical template does not equal the declared Legendre bank"
        )

    shift_permutation = np.fromiter(
        (
            layer * p + ((x + 1) % p)
            for layer in range(layers)
            for x in range(p)
        ),
        dtype=np.int64,
        count=coordinates,
    )
    c_permutation = np.fromiter(
        (
            (7 - layer) * p + ((x + (7 - layer) - h) % p)
            for layer in range(layers)
            for x in range(p)
        ),
        dtype=np.int64,
        count=coordinates,
    )
    if (
        np.unique(shift_permutation).size != coordinates
        or np.unique(c_permutation).size != coordinates
    ):
        raise PRWOrbitValidationError(
            "declared actions are not coordinate permutations"
        )
    if not np.array_equal(
        c_permutation[c_permutation],
        np.arange(coordinates, dtype=np.int64),
    ):
        raise PRWOrbitValidationError("declared C coordinate action is not involutive")

    mask_signs = [
        np.repeat(bank.masks[mask_id], p).astype(np.int8, copy=False)
        for mask_id in range(16)
    ]
    generator_checks = 0
    for source_index, state in enumerate(bank.states):
        _check_deadline(deadline)
        source = bank.templates[source_index]

        target_a = state_lookup[
            (state.type_id, (state.shift - 1) % p, state.mask_id)
        ]
        if not np.array_equal(source[shift_permutation], bank.templates[target_a]):
            raise PRWOrbitValidationError("A_1 template action failed")
        generator_checks += coordinates

        source_mask = bank.masks[state.mask_id]
        for action_mask_id, signs in enumerate(mask_signs):
            if action_mask_id % 4 == 0:
                _check_deadline(deadline)
            product_key = tuple(
                int(value)
                for value in (
                    source_mask * bank.masks[action_mask_id]
                )
            )
            target_mask_id = mask_lookup.get(product_key)
            if target_mask_id is None:
                raise PRWOrbitValidationError(
                    "RM(1,3) is not closed under componentwise multiplication"
                )
            target_b = state_lookup[
                (state.type_id, state.shift, target_mask_id)
            ]
            if not np.array_equal(source * signs, bank.templates[target_b]):
                raise PRWOrbitValidationError("B_n template action failed")
            generator_checks += coordinates

        reversed_key = tuple(int(value) for value in source_mask[::-1])
        target_reversed_mask_id = mask_lookup.get(reversed_key)
        if target_reversed_mask_id is None:
            raise PRWOrbitValidationError(
                "RM(1,3) is not closed under layer reversal"
            )
        target_c_key = (
            1 - state.type_id,
            (state.shift + (2 * state.type_id - 1) * h) % p,
            target_reversed_mask_id,
        )
        target_c = state_lookup[target_c_key]
        transformed_c = source[c_permutation]
        if not np.array_equal(transformed_c, bank.templates[target_c]):
            raise PRWOrbitValidationError("C template action failed")
        generator_checks += coordinates

        c_state = bank.states[target_c]
        c2_reversed_key = tuple(
            int(value) for value in bank.masks[c_state.mask_id][::-1]
        )
        c2_key = (
            1 - c_state.type_id,
            (c_state.shift + (2 * c_state.type_id - 1) * h) % p,
            mask_lookup[c2_reversed_key],
        )
        if c2_key != _state_key(state):
            raise PRWOrbitValidationError("C squared failed on state labels")
        if not np.array_equal(transformed_c[c_permutation], source):
            raise PRWOrbitValidationError("C squared failed on templates")
        generator_checks += coordinates

    if generator_checks != preflight["generator_symbol_checks"]:
        raise PRWOrbitValidationError("symbol-check accounting drifted")

    normal_form_images = set()
    identity_mask = bank.masks[identity_mask_id]
    for c_power in (0, 1):
        for shift_action in range(p):
            for mask_id in range(16):
                if c_power == 0:
                    type_id = 0
                    shift = (-shift_action) % p
                    mask_key = tuple(
                        int(value)
                        for value in (identity_mask * bank.masks[mask_id])
                    )
                else:
                    type_id = 1
                    shift = (-h - shift_action) % p
                    mask_key = tuple(
                        int(value)
                        for value in (
                            identity_mask[::-1] * bank.masks[mask_id]
                        )
                    )
                normal_form_images.add(
                    (type_id, shift, mask_lookup[mask_key])
                )
    if (
        len(normal_form_images) != 32 * p
        or normal_form_images != set(state_lookup)
    ):
        raise PRWOrbitValidationError(
            "A/B/C normal forms do not act regularly on all 32p states"
        )

    elapsed = time.monotonic() - started
    _check_deadline(deadline)
    return {
        "protocol_id": "PRW-T1-TRANSMITTED-ORBIT-v1",
        "status": "EXACT_FINITE_ORBIT_CERTIFIED",
        "prime": p,
        "layers": layers,
        "type_count": 2,
        "mask_count": 16,
        "transmitted_state_count": 32 * p,
        "orbit_size": 32 * p,
        "group_parameter_count": 32 * p,
        "unique_normal_form_count": len(normal_form_images),
        "canonical_state": {
            "type_id": 0,
            "shift": 0,
            "mask_id": identity_mask_id,
        },
        "half_layer_span": h,
        "actions": {
            "A_a": "(t,s,m) -> (t,s-a,m)",
            "B_n": "(t,s,m) -> (t,s,n*m)",
            "C": "(t,s,m) -> (1-t,s+(2*t-1)*h,reverse(m))",
            "C_squared_is_identity": True,
            "two_h_equals_seven_mod_p": (2 * h) % p == 7 % p,
        },
        "finite_transmitted_state_orbit_complete": True,
        "action_is_transitive": True,
        "action_is_regular": True,
        "wrong_type_scope_preserved": True,
        "disagreement_family_equivariance": (
            "coordinate permutation plus common coordinatewise sign relabeling"
        ),
        "invariant_fields": (
            "distances",
            "distance_spectrum",
            "disagreement_intersections",
            "intersection_spectrum",
            "pairwise_bsc_probabilities",
            "ordinary_union_bound",
            "exact_competitor_event_union",
            "hunter_maximum_spanning_tree_weight",
            "hunter_upper_bound",
        ),
        "equivariant_not_literal_fields": ("nearest_states",),
        "nondeterministic_fields": ("resource_guard.elapsed_seconds",),
        "fixed_zero_convention": "L(0)=+1",
        "uses_ring_reflection": False,
        "fixed_zero_exception_handled": True,
        "reflection_sign_argument_rejected": True,
        "generator_symbol_checks_completed": generator_checks,
        "resource_guard": {
            **preflight,
            "elapsed_seconds": elapsed,
            "process_rss_measured": False,
        },
        "final_decoder_error_probability_computed": False,
        "asymptotic_claim_established": False,
        "arbitrary_distinct_signature_controls_certified": False,
        "utility_claim": False,
        "novelty_claim": False,
    }


__all__ = [
    "HARD_MAX_ESTIMATED_BYTES",
    "HARD_MAX_HYPOTHESES",
    "HARD_MAX_SECONDS",
    "HARD_MAX_SYMBOL_CHECKS",
    "HARD_MAX_WORK_UNITS",
    "OrbitCertificateBudget",
    "PRWOrbitResourceError",
    "PRWOrbitValidationError",
    "certify_prw_t1_transmitted_orbit",
    "preflight_prw_t1_transmitted_orbit",
]
