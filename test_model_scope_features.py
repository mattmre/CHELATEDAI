import unittest

import torch

from model_scope_features import FallbackActivationFeatureExtractor, QwenScopeFeatureExtractor, build_feature_scorecard
from qwen_scope_adapter import QwenScopeLayerSAE


class TestModelScopeFeatures(unittest.TestCase):
    def test_fallback_feature_extractor_uses_last_token_dimensions(self):
        extractor = FallbackActivationFeatureExtractor(top_dimensions=2)
        activation = torch.tensor([[[0.1, -0.9, 0.5], [1.5, -2.0, 0.2]]], dtype=torch.float32)

        summary = extractor.summarize(layer_index=1, activation=activation)

        self.assertEqual(summary["feature_space"], "activation_dimension_fallback")
        self.assertEqual(summary["layer_index"], 1)
        self.assertEqual(len(summary["active_features"]), 2)
        self.assertEqual(summary["active_features"][0]["feature_id"], "dim_1")

    def test_qwen_scope_feature_extractor_prefers_sae_and_falls_back_when_missing(self):
        sae = QwenScopeLayerSAE.from_state_dict(
            {
                "W_enc": torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                    ],
                    dtype=torch.float32,
                ),
                "b_enc": torch.tensor([0.0, 0.0, -0.5], dtype=torch.float32),
            },
            layer_index=0,
            top_k=2,
        )
        fallback = FallbackActivationFeatureExtractor(top_dimensions=1)
        extractor = QwenScopeFeatureExtractor({0: sae}, top_features=2, fallback=fallback)
        activation = torch.tensor([[[0.4, 1.2]]], dtype=torch.float32)

        sae_summary = extractor.summarize(layer_index=0, activation=activation)
        fallback_summary = extractor.summarize(layer_index=4, activation=activation)

        self.assertEqual(sae_summary["feature_space"], "qwen_scope_sae")
        self.assertEqual(fallback_summary["feature_space"], "activation_dimension_fallback")

    def test_feature_scorecard_tracks_support_polarity_and_risk(self):
        entries = [
            {
                "label": "positive",
                "artifact": {
                    "capture": {
                        "observations": [
                            {
                                "layer_index": 0,
                                "feature_summary": {
                                    "feature_space": "qwen_scope_sae",
                                    "active_features": [{"feature_id": "101", "value": 1.0}],
                                },
                            }
                        ]
                    }
                },
            },
            {
                "label": "negative",
                "artifact": {
                    "capture": {
                        "observations": [
                            {
                                "layer_index": 0,
                                "feature_summary": {
                                    "feature_space": "qwen_scope_sae",
                                    "active_features": [{"feature_id": "202", "value": 2.0}],
                                },
                            }
                        ]
                    }
                },
            },
        ]

        scorecard = build_feature_scorecard(entries)

        self.assertEqual(scorecard["entry_count"], 2)
        self.assertEqual(scorecard["feature_count"], 2)
        postures = {feature["feature_id"]: feature["recommended_posture"] for feature in scorecard["features"]}
        self.assertEqual(postures["101"], "candidate_amplify")
        self.assertEqual(postures["202"], "candidate_suppress")


if __name__ == "__main__":
    unittest.main()
