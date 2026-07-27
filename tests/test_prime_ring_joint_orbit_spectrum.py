import unittest
from unittest.mock import patch

from prime_ring_joint_orbit_spectrum import (
    JointOrbitValidationError,
    OrbitState,
    analyze_joint_orbit_spectrum,
    build_joint_orbit_artifact,
    orbit_word,
    preflight_joint_orbit_screen,
)
from prime_ring_rb10_contract import (
    ExperimentBudget,
    RB10ResourceError,
    validate_artifact_envelope,
)


ALIGNED_SPECTRUM = (
    (76, 56),
    (80, 13),
    (84, 8),
    (88, 196),
    (92, 8),
    (96, 13),
    (100, 56),
    (176, 1),
)

CANDIDATE_SPECTRUM = (
    (76, 19),
    (80, 10),
    (82, 42),
    (84, 5),
    (86, 6),
    (88, 186),
    (90, 6),
    (92, 5),
    (94, 42),
    (96, 10),
    (100, 19),
    (176, 1),
)


class TestPrimeRingJointOrbitSpectrum(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = analyze_joint_orbit_spectrum()

    def test_frozen_bank_and_complete_schedule_count(self):
        self.assertEqual(self.result.modulus, 11)
        self.assertEqual(self.result.layers, 8)
        self.assertEqual(self.result.state_count, 352)
        self.assertEqual(self.result.coordinates_per_plank, 88)
        self.assertEqual(self.result.searched_schedule_count, 55)

    def test_aligned_and_candidate_exact_spectra_match_independent_fixture(self):
        self.assertEqual(
            self.result.aligned_repetition.type_zero_spectrum.shells,
            ALIGNED_SPECTRUM,
        )
        self.assertEqual(
            self.result.aligned_repetition.type_one_spectrum.shells,
            ALIGNED_SPECTRUM,
        )
        self.assertEqual(
            self.result.candidate_schedule.type_zero_spectrum.shells,
            CANDIDATE_SPECTRUM,
        )
        self.assertEqual(
            self.result.candidate_schedule.type_one_spectrum.shells,
            CANDIDATE_SPECTRUM,
        )

    def test_candidate_reduces_leading_multiplicity_but_not_distance(self):
        self.assertFalse(self.result.candidate_improves_minimum_distance)
        self.assertTrue(self.result.candidate_reduces_minimum_multiplicity)
        self.assertTrue(self.result.candidate_tail_bound_below_repetition)
        self.assertTrue(self.result.candidate_is_restricted_optimum)
        self.assertEqual(self.result.best_restricted_tie_count, 15)
        self.assertIn((1, 5), self.result.best_restricted_actions)
        self.assertEqual(
            self.result.construction_status,
            "RESTRICTED_SHELL_SHAPING_LEAD_CONTROLS_INCOMPLETE",
        )

    def test_flattened_stack_is_exactly_the_same_longer_code(self):
        self.assertTrue(self.result.flattened_distance_equality_all_competitors)
        self.assertTrue(self.result.flattened_spectrum_equality)
        self.assertEqual(
            self.result.flat_equivalence_status,
            "EXACT_EQUALITY_CONFIRMED_STACK_IS_ONE_LONGER_CODE",
        )

    def test_exact_tail_bound_is_rational_and_nonpromotional(self):
        candidate = self.result.candidate_schedule.type_zero_spectrum
        aligned = self.result.aligned_repetition.type_zero_spectrum
        self.assertTrue(candidate.exact_pairwise_tail_union_bound_numerator.isdigit())
        self.assertTrue(candidate.exact_pairwise_tail_union_bound_denominator.isdigit())
        self.assertNotEqual(
            candidate.exact_pairwise_tail_union_bound_numerator,
            aligned.exact_pairwise_tail_union_bound_numerator,
        )
        self.assertFalse(self.result.known_design_controls_complete)
        self.assertFalse(self.result.unrestricted_matched_cost_control_complete)
        self.assertFalse(self.result.promotion_eligible)
        self.assertFalse(self.result.novelty_claim)

    def test_preflight_is_small_and_refuses_before_word_construction(self):
        estimate = preflight_joint_orbit_screen()
        self.assertLess(estimate.estimated_peak_bytes, 2 * 1024 * 1024)
        self.assertLess(estimate.estimated_work_units, 2_000_000)
        budget = ExperimentBudget(max_assignments=100)
        with patch("prime_ring_joint_orbit_spectrum.orbit_word") as word:
            with self.assertRaises(RB10ResourceError):
                analyze_joint_orbit_spectrum(budget=budget)
        word.assert_not_called()

    def test_artifact_is_deterministic_and_valid(self):
        first = build_joint_orbit_artifact(self.result)
        second = build_joint_orbit_artifact(self.result)
        self.assertEqual(first.artifact_digest, second.artifact_digest)
        self.assertTrue(validate_artifact_envelope(first))

    def test_inputs_fail_closed(self):
        with self.assertRaises(JointOrbitValidationError):
            orbit_word(True, OrbitState(0, 0, 0))
        with self.assertRaises(JointOrbitValidationError):
            orbit_word(1, (0, 0, 0))
        with self.assertRaises(JointOrbitValidationError):
            build_joint_orbit_artifact({})


if __name__ == "__main__":
    unittest.main()
