import tempfile
import unittest
from pathlib import Path

from checkpoint_manager import CheckpointManager
from model_scope_trainer import ModelScopeShadowPolicyTrainer, ModelScopeTrainerConfig


def _artifact(feature_id: str, value: float):
    return {
        "capture": {
            "observations": [
                {
                    "layer_index": 0,
                    "feature_summary": {
                        "feature_space": "qwen_scope_sae",
                        "active_features": [{"feature_id": feature_id, "value": value}],
                    },
                }
            ]
        }
    }


class TestModelScopeShadowPolicyTrainer(unittest.TestCase):
    def test_train_shadow_policy_learns_bounded_rules(self):
        replay_bundle = {
            "entries": [
                {"entry_id": "p1", "metadata": {"label": "positive"}, "artifact": _artifact("101", 1.2)},
                {"entry_id": "p2", "metadata": {"label": "positive"}, "artifact": _artifact("101", 1.1)},
                {"entry_id": "n1", "metadata": {"label": "negative"}, "artifact": _artifact("202", 1.4)},
                {"entry_id": "n2", "metadata": {"label": "negative"}, "artifact": _artifact("202", 1.3)},
            ]
        }
        trainer = ModelScopeShadowPolicyTrainer(
            ModelScopeTrainerConfig(
                max_rules=4,
                min_examples=4,
                min_feature_gap=0.2,
                min_alignment_score=0.5,
            )
        )

        candidate = trainer.train_shadow_policy(replay_bundle, candidate_id="shadow_v1")

        self.assertEqual(candidate["candidate_id"], "shadow_v1")
        self.assertTrue(candidate["promotion_gate"]["promotion_ready"])
        self.assertFalse(candidate["artifact_manifest"]["base_weights_mutated"])
        self.assertEqual(candidate["artifact_manifest"]["artifact_class"], "shadow_policy_overlay")
        action_types = {rule["action_type"] for rule in candidate["policy"]["rules"]}
        self.assertIn("amplify", action_types)
        self.assertIn("suppress", action_types)

    def test_train_shadow_policy_fails_closed_when_labels_are_sparse(self):
        replay_bundle = {
            "entries": [
                {"entry_id": "p1", "metadata": {"label": "positive"}, "artifact": _artifact("101", 1.2)},
                {"entry_id": "p2", "metadata": {"label": "positive"}, "artifact": _artifact("101", 1.1)},
            ]
        }
        trainer = ModelScopeShadowPolicyTrainer(ModelScopeTrainerConfig(min_examples=3))

        candidate = trainer.train_shadow_policy(replay_bundle, candidate_id="shadow_v2")

        self.assertFalse(candidate["promotion_gate"]["promotion_ready"])
        self.assertEqual(candidate["policy"]["rules"], [])
        self.assertFalse(candidate["artifact_manifest"]["base_weights_mutated"])

    def test_promote_candidate_writes_policy_file(self):
        replay_bundle = {
            "entries": [
                {"entry_id": "p1", "metadata": {"label": "positive"}, "artifact": _artifact("101", 1.2)},
                {"entry_id": "p2", "metadata": {"label": "positive"}, "artifact": _artifact("101", 1.1)},
                {"entry_id": "n1", "metadata": {"label": "negative"}, "artifact": _artifact("202", 1.4)},
                {"entry_id": "n2", "metadata": {"label": "negative"}, "artifact": _artifact("202", 1.3)},
            ]
        }
        trainer = ModelScopeShadowPolicyTrainer(ModelScopeTrainerConfig(min_alignment_score=0.5))
        candidate = trainer.train_shadow_policy(replay_bundle, candidate_id="shadow_v3")

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "shadow_policy.json"
            target_path.write_text('{"name": "previous"}', encoding="utf-8")
            checkpoint_manager = CheckpointManager(Path(tmpdir) / "checkpoints")

            result = trainer.promote_candidate(candidate, target_path, checkpoint_manager=checkpoint_manager)

            self.assertTrue(result["promoted"])
            self.assertTrue(target_path.read_text(encoding="utf-8").startswith("{"))
            self.assertIsNotNone(result["checkpoint_id"])


if __name__ == "__main__":
    unittest.main()
