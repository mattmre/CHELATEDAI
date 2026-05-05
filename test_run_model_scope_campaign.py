import tempfile
import unittest
from pathlib import Path

from model_scope_artifacts import build_model_scope_artifact, write_model_scope_artifact
from run_model_scope_campaign import run_model_scope_campaign


def _artifact(query_id: str, prompt_hash: str, *, feature_id: str, value: float = 1.0):
    return build_model_scope_artifact(
        runtime={"model_name": "Qwen/Qwen3.5-2B", "device": "cpu"},
        capture={
            "prompt_hash": prompt_hash,
            "token_count": 3,
            "captured_layer_count": 1,
            "layer_indices": [0],
            "metadata": {"query_id": query_id},
            "observations": [
                {
                    "layer_index": 0,
                    "feature_summary": {
                        "feature_space": "qwen_scope_sae",
                        "active_features": [{"feature_id": feature_id, "value": value}],
                    },
                }
            ],
        },
    )


class TestRunModelScopeCampaign(unittest.TestCase):
    def test_campaign_builds_memory_replay_and_candidate_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "artifacts"
            output_dir = root / "outputs"
            input_dir.mkdir()

            write_model_scope_artifact(input_dir / "q1_ref.json", _artifact("q1", "hash-q1", feature_id="101", value=1.2))
            write_model_scope_artifact(input_dir / "q1_drift.json", _artifact("q1", "hash-q1", feature_id="202", value=1.4))
            write_model_scope_artifact(input_dir / "q2_ref.json", _artifact("q2", "hash-q2", feature_id="101", value=1.1))
            write_model_scope_artifact(input_dir / "q2_repeat.json", _artifact("q2", "hash-q2", feature_id="101", value=1.0))

            report = run_model_scope_campaign(input_dir, output_dir=output_dir, max_rules=4, min_alignment_score=0.5)

            self.assertEqual(report["artifact_count"], 4)
            self.assertEqual(report["comparison_failure_count"], 1)
            self.assertIn("campaign_report", report["outputs"])
            self.assertTrue((output_dir / "memory_snapshot.json").exists())
            self.assertTrue((output_dir / "replay_bundle.json").exists())
            self.assertTrue((output_dir / "comparison_report.json").exists())
            self.assertTrue((output_dir / "shadow_policy_candidate.json").exists())
            self.assertTrue((output_dir / "evidence_bundle.json").exists())
            self.assertTrue((output_dir / "evaluator_summary.json").exists())
            self.assertTrue((output_dir / "promotion_decision.json").exists())
            self.assertTrue((output_dir / "trace_grade.json").exists())
            self.assertTrue((output_dir / "compute_budget_summary.json").exists())
            self.assertTrue((output_dir / "reward_overoptimization_report.json").exists())
            self.assertTrue((output_dir / "feature_scorecard.json").exists())
            self.assertTrue((output_dir / "holdout_report.json").exists())
            self.assertTrue((output_dir / "safety_report.json").exists())
            self.assertEqual(report["evidence_summary"]["surfaces"]["model_scope"], 4)
            self.assertEqual(report["evidence_summary"]["surfaces"]["evaluator"], 4)
            self.assertEqual(report["compute_budget_summary"]["decision_count"], 4)
            self.assertFalse(report["trace_grade"]["passed"])
            self.assertEqual(report["trace_grade"]["failure_count"], 1)
            self.assertFalse(report["holdout_report"]["passed"])
            self.assertFalse(report["safety_report"]["passed"])
            self.assertGreaterEqual(report["feature_scorecard"]["feature_count"], 2)
            self.assertFalse(report["promotion_decision"]["promotion_ready"])
            self.assertIn("holdout_failed", report["promotion_decision"]["reasons"])
            self.assertIn("safety_failed", report["promotion_decision"]["reasons"])
            self.assertIn("hard_negative_blockers_present", report["promotion_decision"]["reasons"])

    def test_campaign_accepts_explicit_holdout_and_safety_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "artifacts"
            output_dir = root / "outputs"
            input_dir.mkdir()

            write_model_scope_artifact(input_dir / "q1_ref.json", _artifact("q1", "hash-q1", feature_id="101", value=1.2))
            write_model_scope_artifact(input_dir / "q1_repeat.json", _artifact("q1", "hash-q1", feature_id="101", value=1.1))
            write_model_scope_artifact(input_dir / "q2_ref.json", _artifact("q2", "hash-q2", feature_id="202", value=1.2))
            write_model_scope_artifact(input_dir / "q2_repeat.json", _artifact("q2", "hash-q2", feature_id="202", value=1.1))

            report = run_model_scope_campaign(
                input_dir,
                output_dir=output_dir,
                max_rules=4,
                min_alignment_score=0.5,
                holdout_report={"passed": True, "score": 0.8},
                safety_report={"passed": True},
            )

            self.assertNotIn("missing_holdout_report", report["promotion_decision"]["reasons"])
            self.assertNotIn("missing_safety_report", report["promotion_decision"]["reasons"])


if __name__ == "__main__":
    unittest.main()
