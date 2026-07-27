import unittest
from unittest.mock import patch

from prime_ring_action_nondegeneracy import (
    ActionNondegeneracyValidationError,
    analyze_action_nondegeneracy,
    build_action_nondegeneracy_artifact,
    preflight_action_nondegeneracy,
)
from prime_ring_rb10_contract import (
    ExperimentBudget,
    RB10ResourceError,
    validate_artifact_envelope,
)


class TestPrimeRingActionNondegeneracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = analyze_action_nondegeneracy()

    def test_fixed_semantic_labels_are_nondegenerate(self):
        result = self.result

        self.assertEqual(result.action_count, 10)
        self.assertEqual(result.competitor_count, 351)
        self.assertEqual(result.unique_semantic_fingerprint_count, 10)
        self.assertGreater(result.posterior_relevant_variable_pair_count, 0)
        self.assertGreater(result.action_pair_crossover_count, 0)
        self.assertIsNone(result.universal_weakly_dominating_action)
        self.assertTrue(result.fixed_semantic_labels_nondegenerate)

    def test_action_distance_multisets_are_relabeling_equivalent(self):
        result = self.result

        self.assertEqual(result.unique_distance_multiset_count, 1)
        self.assertTrue(result.all_actions_distance_multiset_equivalent)
        self.assertEqual(
            result.action_status,
            ("SURVIVES_FIXED_LABEL_CROSSOVER_SEMANTIC_RELABELING_COST_PENDING"),
        )

    def test_frozen_crossover_witness_has_both_rankings(self):
        witness = self.result.crossover_witness

        self.assertEqual(witness.first_action, 1)
        self.assertEqual(witness.second_action, 5)
        self.assertEqual(witness.first_competitor_distances, (48, 42))
        self.assertEqual(witness.second_competitor_distances, (44, 50))

    def test_isometry_controls_and_claim_boundaries_are_explicit(self):
        result = self.result

        self.assertTrue(result.copied_action_fingerprint_equal)
        self.assertTrue(result.common_coordinate_roll_fingerprint_equal)
        self.assertFalse(result.posterior_policy_executed)
        self.assertFalse(result.observation_saving_established)
        self.assertFalse(result.policy_compute_saving_established)
        self.assertFalse(result.promotion_eligible)
        self.assertFalse(result.novelty_claim)

    def test_resource_estimate_is_small_but_not_process_rss(self):
        estimate = preflight_action_nondegeneracy()

        self.assertLess(estimate.estimated_peak_bytes, 2 * 1024 * 1024)
        self.assertLess(estimate.estimated_work_units, 1_000_000)
        self.assertFalse(estimate.measured_process_peak)
        self.assertFalse(estimate.estimate_is_process_rss)

    def test_preflight_refuses_before_fingerprint_construction(self):
        budget = ExperimentBudget(max_work_units=1)
        with patch("prime_ring_action_nondegeneracy._fingerprint") as fingerprint:
            with self.assertRaises(RB10ResourceError):
                analyze_action_nondegeneracy(budget=budget)
        fingerprint.assert_not_called()

    def test_artifact_is_deterministic_and_validated(self):
        budget = ExperimentBudget()
        first = build_action_nondegeneracy_artifact(
            self.result,
            budget=budget,
        )
        second = build_action_nondegeneracy_artifact(
            self.result,
            budget=budget,
        )

        self.assertEqual(first.artifact_digest, second.artifact_digest)
        self.assertTrue(validate_artifact_envelope(first.as_dict()))

    def test_artifact_rejects_wrong_result_type(self):
        with self.assertRaises(ActionNondegeneracyValidationError):
            build_action_nondegeneracy_artifact({})


if __name__ == "__main__":
    unittest.main()
