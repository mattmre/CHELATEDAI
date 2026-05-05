import unittest

from evaluator_fabric import detect_reward_overoptimization, grade_trace_events, summarize_evaluator_results


class TestEvaluatorFabric(unittest.TestCase):
    def test_summary_reports_majority_agreement_and_mean_score(self):
        summary = summarize_evaluator_results(
            [
                {"evaluator_id": "replay", "passed": True, "score": 0.9},
                {"evaluator_id": "holdout", "passed": True, "score": 0.7},
                {"evaluator_id": "negative", "passed": False, "score": 0.2},
            ]
        )

        self.assertTrue(summary["majority_passed"])
        self.assertEqual(summary["pass_count"], 2)
        self.assertEqual(summary["fail_count"], 1)
        self.assertAlmostEqual(summary["agreement_score"], 2 / 3)
        self.assertAlmostEqual(summary["mean_score"], 0.6)

    def test_trace_grade_surfaces_event_failures_and_missing_required_surfaces(self):
        grade = grade_trace_events(
            [{"event_id": "evt_1", "surface": "model_scope", "decision": "passed"}],
            required_surfaces={"model_scope", "evaluator"},
        )

        self.assertFalse(grade["passed"])
        self.assertEqual(grade["failure_count"], 1)
        self.assertEqual(grade["failures"][0]["failure_type"], "missing_required_surface")

    def test_reward_overoptimization_detects_heldout_divergence(self):
        report = detect_reward_overoptimization(
            {"score": 0.9},
            {"score": 0.4},
            max_divergence=0.2,
            min_training_gain=0.5,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(report["heldout_divergence_detected"])


if __name__ == "__main__":
    unittest.main()
