import unittest

from model_scope_steering import ModelScopeShadowSteerer
from steering_policy import ModelScopeSteeringPolicy


class TestModelScopeShadowSteerer(unittest.TestCase):
    def test_shadow_policy_matches_features_without_applying_runtime_edits(self):
        steerer = ModelScopeShadowSteerer(
            ModelScopeSteeringPolicy.from_mapping(
                {
                    "name": "shadow_qwen_scope_test",
                    "deployment_mode": "shadow_mode",
                    "feature_space": "qwen_scope_sae",
                    "rules": [
                        {
                            "feature_id": "42",
                            "min_value": 0.8,
                            "layer_index": 3,
                            "action_type": "suppress",
                            "strength": 0.4,
                        }
                    ],
                }
            )
        )
        artifact = {
            "capture": {
                "observations": [
                    {
                        "layer_index": 3,
                        "feature_summary": {
                            "feature_space": "qwen_scope_sae",
                            "active_features": [
                                {"feature_id": 42, "value": 1.2},
                                {"feature_id": 7, "value": 0.3},
                            ],
                        },
                    }
                ]
            }
        }

        result = steerer.evaluate_capture(artifact)

        self.assertEqual(result["policy_name"], "shadow_qwen_scope_test")
        self.assertEqual(result["matched_rule_count"], 1)
        self.assertFalse(result["runtime_applied"])
        self.assertEqual(result["recommended_actions"][0]["feature_id"], "42")


if __name__ == "__main__":
    unittest.main()
