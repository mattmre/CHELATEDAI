from __future__ import annotations

import dataclasses
import unittest
from unittest.mock import patch

import numpy as np

from prime_ring_intersection import build_legendre_mask_bank
from prime_ring_intersection_orbits import (
    OrbitCertificateBudget,
    PRWOrbitResourceError,
    PRWOrbitValidationError,
    certify_prw_t1_transmitted_orbit,
    preflight_prw_t1_transmitted_orbit,
)
from prime_ring_waypoint import legendre_carrier, policy_masks


def _bank(prime: int):
    return build_legendre_mask_bank(
        prime,
        (
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 1, 2, 3, 4, 5, 6, 7),
        ),
        policy_masks("typed16", 8),
    )


class OrbitCertificateTests(unittest.TestCase):
    def test_exact_certificates_at_all_preregistered_primes(self):
        expected_checks = {
            11: 588_544,
            19: 1_755_904,
            31: 4_674_304,
        }
        for prime, expected in expected_checks.items():
            with self.subTest(prime=prime):
                result = certify_prw_t1_transmitted_orbit(_bank(prime))
                self.assertEqual(
                    result["status"],
                    "EXACT_FINITE_ORBIT_CERTIFIED",
                )
                self.assertEqual(result["orbit_size"], 32 * prime)
                self.assertEqual(
                    result["generator_symbol_checks_completed"],
                    expected,
                )
                self.assertTrue(result["action_is_regular"])
                self.assertTrue(result["wrong_type_scope_preserved"])
                self.assertTrue(result["fixed_zero_exception_handled"])
                self.assertFalse(result["uses_ring_reflection"])
                self.assertFalse(result["novelty_claim"])

    def test_preflight_models_all_scans_and_conservative_peak_memory(self):
        bank = _bank(31)
        result = preflight_prw_t1_transmitted_orbit(bank)
        self.assertEqual(result["hypothesis_count"], 992)
        self.assertEqual(result["coordinate_count"], 248)
        self.assertEqual(result["generator_symbol_checks"], 4_674_304)
        self.assertEqual(
            result["template_alphabet_symbol_checks"],
            246_016,
        )
        self.assertEqual(result["canonical_symbol_checks"], 248)
        self.assertEqual(
            result["modeled_symbol_checks"],
            4_920_568,
        )
        self.assertGreater(
            result["estimated_work_units"],
            result["modeled_symbol_checks"],
        )
        self.assertGreaterEqual(
            result["estimated_incremental_peak_bytes"],
            2_959_696,
        )
        self.assertGreater(
            result["estimated_co_resident_peak_bytes"],
            result["estimated_incremental_peak_bytes"],
        )
        self.assertFalse(result["bank_construction_included"])
        self.assertTrue(result["live_input_bank_storage_included"])
        self.assertFalse(result["arrays_allocated"])
        self.assertFalse(result["process_rss_measured"])

    def test_declared_actions_and_claim_boundaries_are_explicit(self):
        result = certify_prw_t1_transmitted_orbit(_bank(11))
        self.assertTrue(result["actions"]["C_squared_is_identity"])
        self.assertTrue(result["actions"]["two_h_equals_seven_mod_p"])
        self.assertTrue(result["finite_transmitted_state_orbit_complete"])
        self.assertIn("ordinary_union_bound", result["invariant_fields"])
        self.assertEqual(
            result["equivariant_not_literal_fields"],
            ("nearest_states",),
        )
        self.assertFalse(result["final_decoder_error_probability_computed"])
        self.assertFalse(result["asymptotic_claim_established"])
        self.assertFalse(
            result["arbitrary_distinct_signature_controls_certified"]
        )

    def test_flags_are_recomputed_not_trusted(self):
        bank = dataclasses.replace(
            _bank(11),
            overlap_one_verified=False,
            rm_1_3_verified=False,
            prime_3_mod_4_verified=False,
            prw_t1_structure_verified=False,
        )
        result = certify_prw_t1_transmitted_orbit(bank)
        self.assertTrue(result["finite_transmitted_state_orbit_complete"])

    def test_mutated_template_fails(self):
        bank = _bank(11)
        templates = bank.templates.copy()
        templates[0, 0] *= -1
        forged = dataclasses.replace(bank, templates=templates)
        with self.assertRaises(PRWOrbitValidationError):
            certify_prw_t1_transmitted_orbit(forged)

    def test_duplicate_or_missing_state_fails(self):
        bank = _bank(11)
        forged = dataclasses.replace(
            bank,
            states=(bank.states[0],) + bank.states[1:-1] + (bank.states[0],),
        )
        with self.assertRaises(PRWOrbitValidationError):
            certify_prw_t1_transmitted_orbit(forged)

    def test_arbitrary_distinct_signatures_are_rejected(self):
        bank = build_legendre_mask_bank(
            11,
            (
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 1, 2, 3, 4, 5, 6, 8),
            ),
            policy_masks("typed16", 8),
        )
        with self.assertRaises(PRWOrbitValidationError):
            certify_prw_t1_transmitted_orbit(bank)

    def test_non_rm_mask_bank_is_rejected(self):
        masks = policy_masks("typed16", 8).copy()
        masks[0] = masks[1]
        with self.assertRaises(Exception):
            build_legendre_mask_bank(
                11,
                (
                    (0, 0, 0, 0, 0, 0, 0, 0),
                    (0, 1, 2, 3, 4, 5, 6, 7),
                ),
                masks,
            )

    def test_preflight_refuses_symbol_budget_before_exact_work(self):
        bank = _bank(11)
        budget = OrbitCertificateBudget(max_symbol_checks=500_000)
        with self.assertRaises(PRWOrbitResourceError):
            preflight_prw_t1_transmitted_orbit(bank, budget)
        with patch(
            "prime_ring_intersection_orbits._exact_structure"
        ) as exact_structure:
            with self.assertRaises(PRWOrbitResourceError):
                certify_prw_t1_transmitted_orbit(bank, budget)
            exact_structure.assert_not_called()

    def test_preflight_refuses_total_work_before_exact_work(self):
        bank = _bank(11)
        preflight = preflight_prw_t1_transmitted_orbit(bank)
        budget = OrbitCertificateBudget(
            max_work_units=preflight["estimated_work_units"] - 1,
        )
        with patch(
            "prime_ring_intersection_orbits._exact_structure"
        ) as exact_structure:
            with self.assertRaises(PRWOrbitResourceError):
                certify_prw_t1_transmitted_orbit(bank, budget)
            exact_structure.assert_not_called()

    def test_preflight_refuses_memory_before_exact_work(self):
        bank = _bank(11)
        preflight = preflight_prw_t1_transmitted_orbit(bank)
        budget = OrbitCertificateBudget(
            max_estimated_bytes=(
                preflight["estimated_co_resident_peak_bytes"] - 1
            ),
        )
        with patch(
            "prime_ring_intersection_orbits._exact_structure"
        ) as exact_structure:
            with self.assertRaises(PRWOrbitResourceError):
                certify_prw_t1_transmitted_orbit(bank, budget)
            exact_structure.assert_not_called()

    def test_deadline_fails_closed(self):
        bank = _bank(11)
        budget = OrbitCertificateBudget(max_seconds=1e-12)
        with self.assertRaises(PRWOrbitResourceError):
            certify_prw_t1_transmitted_orbit(bank, budget)

    def test_hostile_budget_subclasses_are_rejected(self):
        class HostileInt(int):
            pass

        with self.assertRaises(PRWOrbitValidationError):
            OrbitCertificateBudget(max_symbol_checks=HostileInt(1_000_000))

    def test_fixed_zero_reflection_sign_shortcut_is_invalid(self):
        carrier = legendre_carrier(11)
        self.assertEqual(int(carrier[0]), 1)
        reflected = carrier[(-np.arange(11)) % 11]
        self.assertNotEqual(int(reflected[0]), int(-carrier[0]))


if __name__ == "__main__":
    unittest.main()
