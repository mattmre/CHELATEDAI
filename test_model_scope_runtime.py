import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from expectation_comparator import ModelScopeExpectationComparator
from model_scope_artifacts import summarize_model_scope_artifact
from model_scope_features import FallbackActivationFeatureExtractor
from model_scope_memory import ModelScopeMemoryStore
from model_scope_steering import ModelScopeShadowSteerer
from model_scope_runtime import ModelScopeRuntime, ModelScopeRuntimeConfig, load_model_scope_artifact
from steering_policy import ModelScopeSteeringPolicy


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
    def __init__(self, layer_count=2, hidden_size=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.model = _FakeInnerModel(layer_count)

    def forward(self, input_ids=None, attention_mask=None, **_kwargs):
        del attention_mask
        hidden_states = input_ids.unsqueeze(-1).float().repeat(1, 1, self.hidden_size)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return {"last_hidden_state": hidden_states}


class _FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=None):
        del truncation
        tokens = [min(index + 1, 9) for index, _token in enumerate(text.split())]
        if not tokens:
            tokens = [1]
        if max_length is not None:
            tokens = tokens[:max_length]
        input_ids = torch.tensor([tokens], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class TestModelScopeRuntime(unittest.TestCase):
    def test_observe_text_writes_versioned_artifact(self):
        runtime = ModelScopeRuntime(
            ModelScopeRuntimeConfig(
                model_name="Qwen/Qwen3.5-2B",
                layer_indices=[0, 1],
                max_input_tokens=8,
                summary_top_dimensions=2,
            ),
            model=_FakeCausalModel(layer_count=2, hidden_size=5),
            tokenizer=_FakeTokenizer(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "runtime_artifact.json"
            artifact = runtime.observe_text("alpha beta gamma", output_path=output_path)
            loaded = load_model_scope_artifact(output_path)

        self.assertEqual(artifact["schema_version"], 1)
        self.assertEqual(artifact["capture"]["captured_layer_count"], 2)
        self.assertEqual(loaded["runtime"]["model_name"], "Qwen/Qwen3.5-2B")
        summary = summarize_model_scope_artifact(loaded)
        self.assertEqual(summary["captured_layer_count"], 2)
        self.assertEqual(summary["token_count"], 3)
        self.assertEqual(
            loaded["capture"]["observations"][0]["feature_summary"]["feature_space"],
            "activation_dimension_fallback",
        )

    def test_close_releases_model_and_tokenizer(self):
        runtime = ModelScopeRuntime(
            ModelScopeRuntimeConfig(model_name="Qwen/Qwen3.5-2B"),
            model=_FakeCausalModel(layer_count=1, hidden_size=4),
            tokenizer=_FakeTokenizer(),
        )

        runtime.close()

        self.assertIsNone(runtime.model)
        self.assertIsNone(runtime.tokenizer)

    def test_runtime_uses_custom_feature_extractor(self):
        runtime = ModelScopeRuntime(
            ModelScopeRuntimeConfig(model_name="Qwen/Qwen3.5-2B", layer_indices=[0]),
            model=_FakeCausalModel(layer_count=1, hidden_size=4),
            tokenizer=_FakeTokenizer(),
            feature_extractor=FallbackActivationFeatureExtractor(top_dimensions=1),
        )

        artifact = runtime.observe_text("alpha beta")

        feature_summary = artifact["capture"]["observations"][0]["feature_summary"]
        self.assertEqual(feature_summary["feature_space"], "activation_dimension_fallback")
        self.assertEqual(len(feature_summary["active_features"]), 1)

    def test_runtime_aggregates_captured_layer_embeddings(self):
        runtime = ModelScopeRuntime(
            ModelScopeRuntimeConfig(
                model_name="Qwen/Qwen3.5-2B",
                layer_indices=[0, 1],
                capture_raw_embeddings=True,
                enable_layer_attention_aggregation=True,
                layer_attention_proj_dim=2,
            ),
            model=_FakeCausalModel(layer_count=2, hidden_size=4),
            tokenizer=_FakeTokenizer(),
        )

        artifact = runtime.observe_text("alpha beta")

        aggregation = artifact["layer_attention_aggregation"]
        self.assertEqual(aggregation["method"], "layer_attention_aggregator")
        self.assertEqual(aggregation["layer_indices"], [0, 1])
        self.assertEqual(aggregation["input_shape"], [1, 2, 4])
        self.assertEqual(aggregation["output_shape"], [1, 4])
        self.assertEqual(len(aggregation["embedding"][0]), 4)

    def test_runtime_emits_shadow_steering_summary(self):
        class _ConstantFeatureExtractor:
            def summarize(self, *, layer_index, activation):
                del activation
                return {
                    "feature_space": "qwen_scope_sae",
                    "layer_index": int(layer_index),
                    "active_feature_count": 1,
                    "active_features": [{"feature_id": 7, "value": 1.5}],
                }

        steerer = ModelScopeShadowSteerer(
            ModelScopeSteeringPolicy.from_mapping(
                {
                    "feature_space": "qwen_scope_sae",
                    "rules": [{"feature_id": "7", "min_value": 1.0, "layer_index": 0}],
                }
            )
        )
        runtime = ModelScopeRuntime(
            ModelScopeRuntimeConfig(model_name="Qwen/Qwen3.5-2B", layer_indices=[0]),
            model=_FakeCausalModel(layer_count=1, hidden_size=4),
            tokenizer=_FakeTokenizer(),
            feature_extractor=_ConstantFeatureExtractor(),
            steerer=steerer,
        )

        artifact = runtime.observe_text("alpha beta")

        self.assertEqual(artifact["steering"]["matched_rule_count"], 1)
        self.assertFalse(artifact["steering"]["runtime_applied"])

    def test_runtime_records_memory_and_expectation_comparison(self):
        reference_runtime = ModelScopeRuntime(
            ModelScopeRuntimeConfig(model_name="Qwen/Qwen3.5-2B", layer_indices=[0]),
            model=_FakeCausalModel(layer_count=1, hidden_size=4),
            tokenizer=_FakeTokenizer(),
        )
        reference_artifact = reference_runtime.observe_text("alpha beta")
        memory = ModelScopeMemoryStore()
        comparator = ModelScopeExpectationComparator()
        profile = comparator.build_expectation_profile(reference_artifact, profile_id="expected_q1")
        memory.store_expectation_profile(profile)

        runtime = ModelScopeRuntime(
            ModelScopeRuntimeConfig(model_name="Qwen/Qwen3.5-2B", layer_indices=[0]),
            model=_FakeCausalModel(layer_count=1, hidden_size=4),
            tokenizer=_FakeTokenizer(),
            memory_store=memory,
            expectation_comparator=comparator,
        )

        artifact = runtime.observe_text("alpha beta", metadata={"expectation_profile_id": "expected_q1"})

        self.assertIn("memory", artifact)
        self.assertIn("episode_entry_id", artifact["memory"])
        self.assertTrue(artifact["expectation_comparison"]["passed"])


if __name__ == "__main__":
    unittest.main()
