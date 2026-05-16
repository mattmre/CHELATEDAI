import os
import sys
import unittest

POC_DIR = os.path.join(os.path.dirname(__file__), "computational_storage_poc")
if POC_DIR not in sys.path:
    sys.path.insert(0, POC_DIR)

from phase7_system_evaluation import evaluate_system_promotion  # noqa: E402


class Phase7SystemEvaluationTests(unittest.TestCase):
    def test_phase7_evaluation_returns_expected_sections(self):
        evaluation = evaluate_system_promotion()

        self.assertIn("metrics", evaluation)
        self.assertIn("checks", evaluation)
        self.assertIn("promotion_questions", evaluation)
        self.assertIn("overall_recommendation", evaluation)
        self.assertIn("production_promotion", evaluation)
        self.assertIn(evaluation["overall_recommendation"], {"promote_research_baseline", "defer_and_iterate"})
        self.assertIn("storage_reduction_ok", evaluation["checks"])
        self.assertIn("integrated_runtime_ok", evaluation["checks"])
        self.assertIn("cpu_only_practical", evaluation["promotion_questions"])


if __name__ == "__main__":
    unittest.main()
