import unittest
from unittest.mock import patch

from prime_ring_irreducible_factors import (
    G2ValidationError,
    analyze_hyper_interaction,
    analyze_pair_interaction,
    estimate_g2_resources,
    exact_primal_treewidth,
    run_g2_algebraic_screen,
)
from prime_ring_rb10_contract import (
    ExperimentBudget,
    RB10ResourceError,
    validate_artifact_envelope,
)


class TestPrimeRingIrreducibleFactors(unittest.TestCase):
    def test_pair_residual_separates_unary_and_orbit_factors(self):
        modulus = 7
        separable = tuple(tuple(2 * left - 3 * right for right in range(modulus)) for left in range(modulus))
        orbit = tuple(
            tuple(int((right - left) % modulus in {1, modulus - 1}) for right in range(modulus))
            for left in range(modulus)
        )

        self.assertTrue(analyze_pair_interaction(separable).separable_into_unaries)
        orbit_analysis = analyze_pair_interaction(orbit)
        self.assertFalse(orbit_analysis.separable_into_unaries)
        self.assertGreater(orbit_analysis.nonzero_count, 0)
        self.assertIsNotNone(orbit_analysis.witness)

    def test_hyper_residual_separates_lower_order_and_mod_sum(self):
        modulus = 7
        allowed = {1, modulus - 1}
        lower_order = tuple(
            tuple(
                tuple(
                    left - 2 * middle + 3 * right + int((middle - left) % modulus in allowed)
                    for right in range(modulus)
                )
                for middle in range(modulus)
            )
            for left in range(modulus)
        )
        mod_sum = tuple(
            tuple(
                tuple(int((left + middle + right) % modulus == 0) for right in range(modulus))
                for middle in range(modulus)
            )
            for left in range(modulus)
        )

        self.assertTrue(analyze_hyper_interaction(lower_order).strict_subset_reducible)
        interaction = analyze_hyper_interaction(mod_sum)
        self.assertFalse(interaction.strict_subset_reducible)
        self.assertGreater(interaction.nonzero_count, 0)

    def test_exact_treewidth_for_path_triangle_and_hyperedge(self):
        self.assertEqual(
            exact_primal_treewidth(3, ((0, 1), (1, 2))).exact_treewidth,
            1,
        )
        self.assertEqual(
            exact_primal_treewidth(
                3,
                ((0, 1), (1, 2), (0, 2)),
            ).exact_treewidth,
            2,
        )
        self.assertEqual(
            exact_primal_treewidth(3, ((0, 1, 2),)).exact_treewidth,
            2,
        )

    def test_complete_p7_algebraic_screen_matches_frozen_counts(self):
        result = run_g2_algebraic_screen(7)

        self.assertTrue(result.pair_separable_control.separable_into_unaries)
        self.assertFalse(result.pair_orbit_factor.separable_into_unaries)
        self.assertEqual(result.g1_consistent_triangle.maximizer_count, 7)
        self.assertEqual(result.g1_consistent_triangle.diagonal_orbit_count, 1)
        self.assertTrue(result.g1_consistent_triangle.fixed_offset_reducible)
        self.assertEqual(result.g2p_ambiguous_path.maximizer_count, 28)
        self.assertEqual(result.g2p_ambiguous_path.diagonal_orbit_count, 4)
        self.assertFalse(result.g2p_ambiguous_path.fixed_offset_reducible)
        self.assertEqual(result.frustrated_triangle.maximizer_count, 21)
        self.assertEqual(result.frustrated_triangle.diagonal_orbit_count, 3)
        self.assertEqual(result.independent_nodes_control.maximizer_count, 343)
        self.assertEqual(
            result.independent_nodes_control.diagonal_orbit_count,
            49,
        )
        self.assertEqual(result.shuffled_label_control.maximizer_count, 28)
        self.assertEqual(result.wrong_grouping_control.maximizer_count, 28)
        self.assertEqual(
            result.matched_random_factor_control.maximizer_count,
            28,
        )
        self.assertTrue(result.lower_order_hyper_control.strict_subset_reducible)
        self.assertFalse(result.mod_sum3_hyperfactor.strict_subset_reducible)
        self.assertTrue(result.query_unary_only_change.separable_into_unaries)
        self.assertFalse(result.query_interaction_change.separable_into_unaries)
        self.assertTrue(result.flat_factorized_scores_equal)
        self.assertTrue(result.flat_factorized_selected_assignment_equal)
        self.assertEqual(
            result.overall_status,
            "SURVIVES_ALGEBRAIC_SCREEN_ONLY_CONTROLS_MATCH",
        )
        self.assertTrue(result.execution_completed)
        self.assertTrue(result.control_screen_passed)
        self.assertFalse(result.static_control_advantage_established)
        self.assertFalse(result.promotion_eligible)
        self.assertFalse(result.novelty_claim)
        self.assertFalse(result.resource_guard.measured_process_peak)
        self.assertFalse(result.resource_guard.estimate_is_process_rss)

    def test_p11_remains_inside_the_same_resource_envelope(self):
        estimate = estimate_g2_resources(11)
        self.assertGreater(estimate.estimated_peak_bytes, 1024 * 1024)
        self.assertLess(estimate.estimated_peak_bytes, 256 * 1024 * 1024)
        self.assertLess(estimate.estimated_work_units, 25_000_000)
        result = run_g2_algebraic_screen(11)
        self.assertEqual(
            result.overall_status,
            "SURVIVES_ALGEBRAIC_SCREEN_ONLY_CONTROLS_MATCH",
        )
        self.assertEqual(
            result.generic_pairwise_auxiliary_upper_bound_bytes,
            3 * 11**3 * 8,
        )

    def test_oversized_assignment_budget_refuses_before_fixture_build(self):
        budget = ExperimentBudget(max_assignments=100)
        with patch("prime_ring_irreducible_factors._pair_table") as pair_table:
            with self.assertRaises(RB10ResourceError):
                run_g2_algebraic_screen(7, budget=budget)
        pair_table.assert_not_called()

    def test_artifact_envelope_is_deterministic_and_validated(self):
        budget = ExperimentBudget()
        result = run_g2_algebraic_screen(7, budget=budget)
        first = result.artifact_envelope(budget)
        second = result.artifact_envelope(budget)

        self.assertEqual(first.artifact_digest, second.artifact_digest)
        self.assertTrue(validate_artifact_envelope(first.as_dict()))

    def test_strict_validation_rejects_bool_and_ragged_tables(self):
        with self.assertRaises(G2ValidationError):
            run_g2_algebraic_screen(True)
        with self.assertRaises(G2ValidationError):
            analyze_pair_interaction(((0, 1), (1,)))
        with self.assertRaises(G2ValidationError):
            analyze_hyper_interaction((((0,),),))


if __name__ == "__main__":
    unittest.main()
