import unittest

from compute_budget_policy import (
    ComputeBudgetPolicyConfig,
    decide_compute_budget,
    summarize_compute_budget_decisions,
)


class TestComputeBudgetPolicy(unittest.TestCase):
    def test_baseline_budget_does_not_escalate_for_clean_signals(self):
        decision = decide_compute_budget(
            {
                "uncertainty_score": 0.1,
                "retrieval_support_score": 0.95,
                "evaluator_disagreement": 0.0,
                "safety_risk": 0.0,
                "hard_negative_risk": 0.0,
                "query_token_count": 3,
            }
        )

        self.assertFalse(decision["escalated"])
        self.assertEqual(decision["budget_units"], 1)
        self.assertEqual(decision["actions"], ["answer_direct"])

    def test_escalates_but_caps_budget_for_risky_signals(self):
        decision = decide_compute_budget(
            {
                "uncertainty_score": 0.9,
                "retrieval_support_score": 0.2,
                "evaluator_disagreement": 0.8,
                "safety_risk": 0.9,
                "hard_negative_risk": 0.9,
                "query_token_count": 30,
            },
            config=ComputeBudgetPolicyConfig(max_budget_units=5),
        )

        self.assertTrue(decision["escalated"])
        self.assertEqual(decision["budget_units"], 5)
        self.assertIn("corrective_retrieve", decision["actions"])
        self.assertIn("safety_verifier", decision["actions"])
        self.assertIn("abstain_if_unsupported", decision["actions"])

    def test_summary_counts_actions_and_reasons(self):
        decisions = [
            decide_compute_budget({"retrieval_support_score": 1.0}),
            decide_compute_budget({"retrieval_support_score": 0.1, "uncertainty_score": 0.9}),
        ]

        summary = summarize_compute_budget_decisions(decisions)

        self.assertEqual(summary["decision_count"], 2)
        self.assertEqual(summary["escalated_count"], 1)
        self.assertEqual(summary["reason_counts"]["low_retrieval_support"], 1)


if __name__ == "__main__":
    unittest.main()
