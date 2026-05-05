import unittest

import torch
import torch.nn as nn

from model_hook_bus import HookObservationConfig, ModelHookBus


class _FakeDecoderLayer(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = float(offset)

    def forward(self, hidden_states):
        return hidden_states + self.offset


class _FakeInnerModel(nn.Module):
    def __init__(self, layer_count):
        super().__init__()
        self.layers = nn.ModuleList([
            _FakeDecoderLayer(index + 1.0)
            for index in range(layer_count)
        ])


class _FakeCausalModel(nn.Module):
    def __init__(self, layer_count=3, hidden_size=6):
        super().__init__()
        self.hidden_size = hidden_size
        self.model = _FakeInnerModel(layer_count)

    def forward(self, input_ids=None, attention_mask=None, **_kwargs):
        del attention_mask
        hidden_states = input_ids.unsqueeze(-1).float().repeat(1, 1, self.hidden_size)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return {"last_hidden_state": hidden_states}


class TestModelHookBus(unittest.TestCase):
    def test_capture_selected_layers_and_emits_summaries(self):
        model = _FakeCausalModel(layer_count=3, hidden_size=6)
        bus = ModelHookBus(HookObservationConfig(layer_indices=[0, 2], summary_top_dimensions=3))

        artifact = bus.capture(
            model,
            {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)},
            model_name="Qwen/Qwen3.5-2B",
            prompt_text="alpha beta gamma",
            metadata={"source": "unit_test"},
        )

        self.assertEqual(artifact["schema_version"], 1)
        self.assertEqual(artifact["captured_layer_count"], 2)
        self.assertEqual(artifact["layer_indices"], [0, 2])
        self.assertEqual(artifact["token_count"], 3)
        self.assertEqual(artifact["metadata"]["source"], "unit_test")
        self.assertEqual(len(artifact["observations"]), 2)
        self.assertEqual(artifact["observations"][0]["layer_index"], 0)
        self.assertEqual(artifact["observations"][1]["layer_index"], 2)
        self.assertEqual(len(artifact["observations"][0]["top_dimensions"]), 3)

    def test_capture_rejects_invalid_layer_index(self):
        model = _FakeCausalModel(layer_count=2, hidden_size=4)
        bus = ModelHookBus(HookObservationConfig(layer_indices=[3]))

        with self.assertRaises(ValueError):
            bus.capture(
                model,
                {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)},
                model_name="Qwen/Qwen3.5-2B",
            )

    def test_capture_can_emit_mean_pooled_embeddings(self):
        model = _FakeCausalModel(layer_count=2, hidden_size=4)
        bus = ModelHookBus(
            HookObservationConfig(
                layer_indices=[0, 1],
                capture_raw_embeddings=True,
            )
        )

        artifact = bus.capture(
            model,
            {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)},
            model_name="Qwen/Qwen3.5-2B",
        )

        self.assertEqual(artifact["captured_layer_count"], 2)
        pooled = artifact["observations"][0]["mean_pooled_embedding"]
        self.assertEqual(pooled["shape"], [1, 4])
        self.assertEqual(len(pooled["values"][0]), 4)


if __name__ == "__main__":
    unittest.main()
