import unittest

from attnres_repeat_seed_decision import summarize_attnres_repeat_seed_decision


class TestAttnResRepeatSeedDecision(unittest.TestCase):
    def test_repeat_seed_decision_fails_closed(self):
        summary = summarize_attnres_repeat_seed_decision()

        self.assertEqual(summary["record_type"], "attnres_repeat_seed_decision")
        self.assertEqual(summary["artifact_count"], 6)
        self.assertEqual(summary["decision"], "no_default_change")
        self.assertFalse(summary["promote_default"])
        self.assertGreater(summary["quantization_failure_count"], 0)
        self.assertIn("quantization_survival_failures_present", summary["reasons"])
        self.assertIn("SciFact", summary["task_summary"])
        self.assertIn("NFCorpus", summary["task_summary"])


if __name__ == "__main__":
    unittest.main()
