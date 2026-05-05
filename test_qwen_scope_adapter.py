import unittest

import torch

from qwen_scope_adapter import QwenScopeLayerSAE


class TestQwenScopeLayerSAE(unittest.TestCase):
    def test_encode_and_summarize_follow_official_shape_contract(self):
        sae = QwenScopeLayerSAE.from_state_dict(
            {
                "W_enc": torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ],
                    dtype=torch.float32,
                ),
                "b_enc": torch.tensor([0.0, 0.0, 0.0, -1.0], dtype=torch.float32),
            },
            layer_index=3,
            top_k=2,
        )
        residual = torch.tensor([[[0.5, 1.5, 2.5]]], dtype=torch.float32)

        acts = sae.encode(residual)

        self.assertEqual(tuple(acts.shape), (1, 1, 4))
        self.assertEqual(int((acts != 0).sum().item()), 2)
        summary = sae.summarize_last_token(residual, top_features=2)
        self.assertEqual(summary["feature_space"], "qwen_scope_sae")
        self.assertEqual(summary["layer_index"], 3)
        self.assertEqual(summary["active_feature_count"], 2)
        self.assertEqual(len(summary["active_features"]), 2)


if __name__ == "__main__":
    unittest.main()
