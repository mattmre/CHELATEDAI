import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

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

    def test_build_distillation_command_forwards_es_options(self):
        args = SimpleNamespace(
            model="model",
            teacher="teacher",
            cycles=1,
            queries_per_cycle=2,
            epochs=3,
            learning_rate=0.01,
            max_eval_queries=4,
            teacher_weight=0.3,
            threshold=1,
            adapter_type="low_rank",
            seed=7,
            sedimentation_optimizer="eggroll_es",
            es_retrieval_fitness=True,
            quantization_gate=True,
            es_antithetic_sampling=True,
            es_rollback_to_elite=True,
            es_quantization_aware=True,
            es_kalman_sigma=True,
            es_population_size=6,
            es_rank=2,
            es_sigma=0.02,
            es_generations=5,
            es_elite_pool_size=2,
            es_fitness_shaping="linear_rank",
            es_storage_profile="consumer_nvme",
            quantization_gate_threshold=0.9,
            structural_health_weight=0.25,
        )

        command = transfer_gate.build_distillation_command("SciFact", Path("results.json"), args)

        self.assertEqual(command[0], sys.executable)
        self.assertIn("--es-retrieval-fitness", command)
        self.assertIn("--es-antithetic-sampling", command)
        self.assertIn("--es-rollback-to-elite", command)
        self.assertIn("--es-quantization-aware", command)
        self.assertIn("--es-kalman-sigma", command)
        self.assertIn("--es-storage-profile", command)
        self.assertIn("consumer_nvme", command)

    def test_extract_quantization_gate_status_requires_observed_passing_gates(self):
        status = transfer_gate.extract_quantization_gate_status({
            "hybrid": [
                {"cycle": 1, "es_result": {"quantization_gate": {"passed": True}}},
                {"cycle": 2, "es_result": {"quantization_gate": {"passed": False}}},
            ]
        })

        self.assertFalse(status["passes_quantization_gate"])
        self.assertEqual(len(status["failed"]), 1)

    def test_summarize_task_result_can_require_quantization_gate(self):
        path = Path("results.json")
        results = {
            "baseline": [{"cycle": 1, "ndcg": 0.5}],
            "offline": {"cycles": [{"cycle": 1, "ndcg": 0.5}]},
            "hybrid": [{"cycle": 1, "ndcg": 0.6, "es_result": {"quantization_gate": {"passed": True}}}],
        }
        original_load = transfer_gate.load_results
        try:
            transfer_gate.load_results = lambda _path: results
            summary = transfer_gate.summarize_task_result(
                "SciFact",
                path,
                reused=False,
                min_task_gain=0.0,
                require_quantization_gate=True,
            )
        finally:
            transfer_gate.load_results = original_load

        self.assertTrue(summary["passes_task_gate"])
        self.assertTrue(summary["quantization_gate"]["passes_quantization_gate"])

    def test_summarize_task_result_returns_failed_summary_for_missing_results(self):
        summary = transfer_gate.summarize_task_result(
            "SciFact",
            Path("missing-results.json"),
            reused=False,
            min_task_gain=0.0,
            require_quantization_gate=True,
        )

        self.assertFalse(summary["passes_task_gate"])
        self.assertIn("error", summary)
        self.assertEqual(summary["baseline_final_ndcg"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
