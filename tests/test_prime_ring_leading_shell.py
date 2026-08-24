import unittest

import numpy as np

from prime_ring_intersection import (
    analyze_legendre_mask_bank,
    build_legendre_mask_bank,
)
from prime_ring_leading_shell import (
    LeadingShellResourceError,
    LeadingShellValidationError,
    analyze_all_states_canonical_first_decoder,
    analyze_canonical_first_decoder,
    analyze_leading_intersections,
    analyze_leading_intersections_closed_form,
    analyze_leading_shell,
    analyze_type_one_canonical_first_decoder,
    distance_spectrum,
    exact_correct_type_distance_shells,
    leading_overlap_classes,
)
from prime_ring_waypoint import policy_masks
from run_prime_ring_waypoint_experiment import _top_two_semantics


class TestExactLeadingShell(unittest.TestCase):
    def _enumerated_spectrum(self, prime):
        signatures = np.mod(
            np.arange(2, dtype=np.int64)[:, np.newaxis] * np.arange(8, dtype=np.int64)[np.newaxis, :],
            prime,
        )
        bank = build_legendre_mask_bank(
            prime,
            signatures,
            policy_masks("typed16", 8),
        )
        result = analyze_legendre_mask_bank(
            bank,
            planted_index=0,
            crossover=0.20,
        )
        return {int(distance): multiplicity for distance, multiplicity in result["distance_spectrum"].items()}

    def test_closed_form_matches_current_analyzer_at_frozen_primes(self):
        for prime in (11, 19, 31):
            with self.subTest(prime=prime):
                self.assertEqual(
                    distance_spectrum(prime),
                    self._enumerated_spectrum(prime),
                )

    def test_complete_formula_covers_every_wrong_type_state(self):
        expected_19 = {
            66: 56,
            70: 8,
            72: 11,
            76: 154,
            80: 11,
            82: 8,
            86: 56,
        }
        self.assertEqual(distance_spectrum(19), expected_19)
        spectrum_4691 = distance_spectrum(4691)
        self.assertEqual(sum(spectrum_4691.values()), 16 * 4691)
        self.assertEqual(spectrum_4691[(7 * 4691 - 1) // 2], 56)
        self.assertEqual(spectrum_4691[(7 * 4691 + 7) // 2], 8)

    def test_original_nearest_only_condition_is_algebraically_false(self):
        observed = []
        for prime in (11, 19, 31):
            result = analyze_leading_shell(prime, 0.20)
            observed.append(result.adjacent_to_nearest_ratio)
            self.assertFalse(result.original_prw_t1_farther_little_o_condition)
            self.assertEqual(
                result.original_prw_t1_status,
                "NEAREST_ONLY_TIE_AS_ERROR_EVENT_UNION_FALSIFIED",
            )
            self.assertEqual(
                result.reconditioned_hypothesis_id,
                "PRW-T1R-TE-EVENT-UNION",
            )
            self.assertEqual(
                result.reconditioned_hypothesis_status,
                "CLOSED_INTERNAL_ANALYTIC_SCOPE",
            )
            self.assertEqual(
                result.final_decoder_status,
                "EXACT_HAMMING_TIE_COROLLARIES_COMPLETE_" "CURRENT_FLOAT_FFT_LINKAGE_FAILED",
            )
            self.assertFalse(result.promotion_eligible)
        self.assertAlmostEqual(observed[0], 0.05581037834178485, places=13)
        self.assertAlmostEqual(observed[1], 0.05687601023715105, places=13)
        self.assertAlmostEqual(observed[2], 0.05748348689588855, places=13)

    def test_adjacent_shell_ratio_converges_to_positive_limit(self):
        result = analyze_leading_shell(4691, 0.20)
        self.assertGreater(
            result.adjacent_to_nearest_asymptotic_limit,
            0.0,
        )
        self.assertAlmostEqual(
            result.adjacent_to_nearest_asymptotic_limit,
            ((4.0 * 0.2 * 0.8) ** 2) / 7.0,
            places=15,
        )
        self.assertLess(
            abs(result.adjacent_to_nearest_ratio - result.adjacent_to_nearest_asymptotic_limit),
            1e-4,
        )

    def test_reconditioned_nonleading_shells_decay_in_finite_screen(self):
        ratios = [
            analyze_leading_shell(prime, 0.20).reconditioned_remainder_to_leading_ratio for prime in (43, 59, 83, 127)
        ]
        self.assertTrue(all(left > right for left, right in zip(ratios, ratios[1:])))
        self.assertLess(ratios[-1], 0.0001)

    def test_fixed_q_theorem_is_not_a_uniform_near_half_finite_certificate(self):
        moderate_noise = analyze_leading_shell(4691, 0.45)
        near_half_noise = analyze_leading_shell(4691, 0.49)
        self.assertGreater(moderate_noise.reconditioned_remainder_to_leading_ratio, 0.005)
        self.assertLess(moderate_noise.reconditioned_remainder_to_leading_ratio, 0.02)
        self.assertGreater(near_half_noise.reconditioned_remainder_to_leading_ratio, 100.0)

        moderate_intersections = analyze_leading_intersections(31, 0.45)
        near_half_intersections = analyze_leading_intersections(31, 0.49)
        self.assertGreater(moderate_intersections.all_pair_intersections_to_leading_ratio, 10.0)
        self.assertGreater(near_half_intersections.all_pair_intersections_to_leading_ratio, 15.0)

    def test_reconditioned_leading_intersections_drop_sharply(self):
        ratios = []
        for prime in (11, 19, 31):
            diagnostic = analyze_leading_intersections(prime, 0.20)
            self.assertEqual(diagnostic.leading_state_count, 64)
            self.assertFalse(diagnostic.asymptotic_claim)
            ratios.append(diagnostic.all_pair_intersections_to_leading_ratio)
        self.assertAlmostEqual(ratios[0], 0.2906143992587361, places=13)
        self.assertAlmostEqual(ratios[1], 0.020266849743127275, places=13)
        self.assertAlmostEqual(ratios[2], 0.0004256262543957568, places=13)
        self.assertTrue(all(left > right for left, right in zip(ratios, ratios[1:])))

    def test_closed_form_pair_classes_match_materialized_and_reproduce_p43(self):
        for prime in (11, 19, 31):
            with self.subTest(prime=prime):
                materialized = analyze_leading_intersections(prime, 0.20)
                closed_form = analyze_leading_intersections_closed_form(prime, 0.20)
                self.assertAlmostEqual(
                    closed_form.all_pair_intersections_to_leading_ratio,
                    materialized.all_pair_intersections_to_leading_ratio,
                    places=13,
                )
        p43 = analyze_leading_intersections_closed_form(43, 0.20)
        self.assertAlmostEqual(
            p43.all_pair_intersections_to_leading_ratio,
            9.576888918946738e-6,
            places=17,
        )
        with self.assertRaises(LeadingShellResourceError):
            analyze_leading_intersections_closed_form(131, 0.20)

    def test_exact_hamming_canonical_first_rule_binds_type_zero_strict_events(self):
        winner, score, margin, tie_count = _top_two_semantics(np.asarray([1.0, 1.0]))
        self.assertEqual((winner, score, margin, tie_count), (0, 1.0, 0.0, 2))

        analysis = analyze_canonical_first_decoder(4691, 0.20)
        self.assertEqual(analysis.transmitted_type, 0)
        self.assertEqual(
            analysis.tie_policy,
            "LOWEST_CANONICAL_TYPE_ID_WINS_EXACT_TIE",
        )
        self.assertEqual(
            analysis.hypothesis_status,
            "ANALYTIC_TIE_COROLLARY_COMPLETE_EXACT_HAMMING_ONLY",
        )
        self.assertAlmostEqual(
            analysis.strict_to_inclusive_asymptotic_limit,
            0.25,
            places=15,
        )
        self.assertLess(
            abs(analysis.finite_strict_to_inclusive_leading_ratio - analysis.strict_to_inclusive_asymptotic_limit),
            1e-4,
        )
        self.assertFalse(analysis.promotion_eligible)

    def test_canonical_first_type_one_ties_are_inclusive_errors(self):
        winner, score, margin, tie_count = _top_two_semantics(np.asarray([1.0, 1.0]))
        self.assertEqual((winner, score, margin, tie_count), (0, 1.0, 0.0, 2))
        self.assertNotEqual(winner, 1)

        analysis = analyze_type_one_canonical_first_decoder(4691, 0.20)
        self.assertEqual(analysis.transmitted_type, 1)
        self.assertEqual(
            analysis.tie_policy,
            "LOWEST_CANONICAL_TYPE_ID_WINS_EXACT_TIE",
        )
        self.assertEqual(
            analysis.hypothesis_id,
            "PRW-T1D-CANONICAL-FIRST-TYPE1",
        )
        self.assertEqual(
            analysis.hypothesis_status,
            "ANALYTIC_TIE_COROLLARY_COMPLETE_EXACT_HAMMING_ONLY",
        )
        self.assertEqual(
            analysis.finite_inclusive_to_inclusive_leading_ratio,
            1.0,
        )
        self.assertEqual(
            analysis.inclusive_to_inclusive_asymptotic_limit,
            1.0,
        )
        self.assertFalse(analysis.promotion_eligible)

    def test_frozen_mixture_closes_only_the_analytic_tie_corollary(self):
        analysis = analyze_all_states_canonical_first_decoder(4691, 0.20)
        self.assertEqual(analysis.transmitted_type_weights, (0.5, 0.5))
        self.assertEqual(analysis.type_zero.transmitted_type, 0)
        self.assertEqual(analysis.type_one.transmitted_type, 1)
        self.assertEqual(analysis.hypothesis_id, "PRW-T1D-ALL-STATES")
        self.assertEqual(
            analysis.hypothesis_status,
            "ANALYTIC_TIE_COROLLARY_COMPLETE_EXACT_HAMMING_ONLY_" "CURRENT_FLOAT_FFT_LINKAGE_FAILED",
        )
        self.assertEqual(
            analysis.transmitted_state_scope,
            "EVERY_FROZEN_BANK_STATE_WITH_HALF_TOTAL_MASS_PER_TYPE",
        )
        self.assertEqual(
            analysis.type_zero.all_transmitted_states_status,
            "ANALYTIC_ALL_STATES_COROLLARY_CURRENT_FLOAT_FFT_LINKAGE_FAILED",
        )
        self.assertEqual(
            analysis.finite_balanced_to_inclusive_leading_ratio,
            0.5 * (1.0 + analysis.type_zero.finite_strict_to_inclusive_leading_ratio),
        )
        self.assertAlmostEqual(
            analysis.balanced_to_inclusive_asymptotic_limit,
            0.625,
            places=15,
        )
        self.assertLess(
            abs(analysis.finite_balanced_to_inclusive_leading_ratio - analysis.balanced_to_inclusive_asymptotic_limit),
            1e-4,
        )
        self.assertFalse(analysis.promotion_eligible)

    def test_correct_type_competition_has_a_linear_distance_gap(self):
        expected_19 = {
            72: 18,
            76: 266,
            80: 18,
            152: 1,
        }
        observed_19 = {shell.distance: shell.multiplicity for shell in exact_correct_type_distance_shells(19)}
        self.assertEqual(observed_19, expected_19)

        ratios = [
            analyze_canonical_first_decoder(prime, 0.20).correct_type_interference_to_strict_leading_upper_ratio
            for prime in (43, 59, 83, 127)
        ]
        self.assertTrue(all(left > right for left, right in zip(ratios, ratios[1:])))
        self.assertLess(ratios[-1], 0.0001)

        type_one_ratios = [
            analyze_type_one_canonical_first_decoder(
                prime, 0.20
            ).correct_type_interference_to_inclusive_leading_upper_ratio
            for prime in (43, 59, 83, 127)
        ]
        balanced_ratios = [
            analyze_all_states_canonical_first_decoder(
                prime, 0.20
            ).correct_type_interference_to_balanced_leading_upper_ratio
            for prime in (43, 59, 83, 127)
        ]
        self.assertTrue(all(left > right for left, right in zip(type_one_ratios, type_one_ratios[1:])))
        self.assertTrue(all(left > right for left, right in zip(balanced_ratios, balanced_ratios[1:])))
        self.assertLess(type_one_ratios[-1], 0.0001)
        self.assertLess(balanced_ratios[-1], 0.0001)

    def test_exact_leading_pair_orbits_cover_every_pair(self):
        for prime in (11, 19, 31, 4691):
            with self.subTest(prime=prime):
                classes = leading_overlap_classes(prime)
                self.assertEqual(
                    sum(value.pair_multiplicity for value in classes),
                    64 * 63 // 2,
                )
                self.assertEqual(
                    [value.pair_multiplicity for value in classes],
                    [84, 1344, 112, 448, 28],
                )
                self.assertEqual(
                    [value.disagreement_intersection for value in classes],
                    [
                        (3 * prime - 5) // 2,
                        (3 * prime - 1) // 2,
                        (3 * prime + 3) // 2,
                        (3 * prime + 3) // 2,
                        (3 * prime + 3) // 2,
                    ],
                )

    def test_invalid_domains_fail_closed(self):
        for prime in (7, 13, 15):
            with self.subTest(prime=prime):
                with self.assertRaises(LeadingShellValidationError):
                    distance_spectrum(prime)
        with self.assertRaises(LeadingShellValidationError):
            analyze_leading_shell(19, 0.5)
        with self.assertRaises(LeadingShellResourceError):
            distance_spectrum(100_019)
        with self.assertRaises(LeadingShellResourceError):
            analyze_leading_intersections(43)
        with self.assertRaises(LeadingShellValidationError):
            analyze_type_one_canonical_first_decoder(19, 0.5)
        with self.assertRaises(LeadingShellValidationError):
            analyze_all_states_canonical_first_decoder(13, 0.20)


if __name__ == "__main__":
    unittest.main()
