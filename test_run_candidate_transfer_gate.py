import unittest
from pathlib import Path

import run_candidate_transfer_gate as transfer_gate


class TestRunCandidateTransferGate(unittest.TestCase):
    def test_parse_reuse_result(self):
        task, path = transfer_gate.parse_reuse_result(r"SciFact=C:\runs\scifact\results.json")
        self.assertEqual(task, "SciFact")
        self.assertEqual(path, Path(r"C:\runs\scifact\results.json"))

    def test_resolve_multitask_medium_tasks(self):
        self.assertEqual(
            transfer_gate.resolve_tasks("multitask", "medium"),
            ["SciFact", "NFCorpus", "FiQA2018"],
        )

    def test_build_transfer_summary_passes_when_all_tasks_gain(self):
        summary = transfer_gate.build_transfer_summary(
            "multitask",
            "small",
            [
                {
                    "task": "SciFact",
                    "baseline_final_ndcg": 0.60,
                    "hybrid_final_ndcg": 0.62,
                    "hybrid_gain_absolute": 0.02,
                    "passes_task_gate": True,
                },
                {
                    "task": "NFCorpus",
                    "baseline_final_ndcg": 0.40,
                    "hybrid_final_ndcg": 0.41,
                    "hybrid_gain_absolute": 0.01,
                    "passes_task_gate": True,
                },
            ],
            min_task_gain=0.0,
        )

        self.assertTrue(summary["passes_transfer_gate"])
        self.assertEqual(summary["recommended_next_step"], "run-medium-transfer")
        self.assertEqual(summary["aggregate"]["positive_gains"], 2)

    def test_build_transfer_summary_fails_when_any_task_regresses(self):
        summary = transfer_gate.build_transfer_summary(
            "beir",
            "medium",
            [
                {
                    "task": "SciFact",
                    "baseline_final_ndcg": 0.60,
                    "hybrid_final_ndcg": 0.58,
                    "hybrid_gain_absolute": -0.02,
                    "passes_task_gate": False,
                },
                {
                    "task": "NFCorpus",
                    "baseline_final_ndcg": 0.40,
                    "hybrid_final_ndcg": 0.41,
                    "hybrid_gain_absolute": 0.01,
                    "passes_task_gate": True,
                },
            ],
            min_task_gain=0.0,
        )

        self.assertFalse(summary["passes_transfer_gate"])
        self.assertEqual(summary["failed_tasks"], ["SciFact"])
        self.assertEqual(summary["recommended_next_step"], "stop-and-review")


if __name__ == "__main__":
    unittest.main(verbosity=2)
