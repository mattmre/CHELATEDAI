from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from model_scope_features import (
    FeatureExtractor,
    FallbackActivationFeatureExtractor,
    QwenScopeFeatureExtractor,
    SparseFeatureEvent,
    build_feature_scorecard,
)
from model_scope_runtime import ActivationEvent
from qwen_scope_adapter import QwenScopeAdapter, QwenScopeLayerSAE


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


def _make_activation(
    model_id: str = "Qwen3.5-7B",
    layer_id: str = "model.layers.0",
    token_count: int = 10,
    shape: tuple = (1, 10, 4096),
    mean_activation: float = 0.5,
    norm_activation: float = 1.2,
    run_id: str = "run-001",
) -> ActivationEvent:
    return ActivationEvent(
        schema_version="1.0",
        model_id=model_id,
        layer_id=layer_id,
        token_count=token_count,
        shape=shape,
        mean_activation=mean_activation,
        norm_activation=norm_activation,
        captured_at=datetime.now(timezone.utc).isoformat(),
        run_id=run_id,
    )


def _loaded_adapter(rows: int = 8, cols: int = 4, model_family: str = "Qwen3.5") -> QwenScopeAdapter:
    rng = np.random.default_rng(42)
    weights = rng.random((rows, cols)).astype(np.float64)
    adapter = QwenScopeAdapter(model_family=model_family)
    adapter.load_checkpoint(
        Path("fake.npy"),
        checkpoint_loader=lambda _p: weights,
    )
    return adapter


class TestSparseFeatureEvent(unittest.TestCase):
    def _make_event(self, features: dict | None = None) -> SparseFeatureEvent:
        if features is None:
            features = {"feature_0": 1.5, "feature_2": 0.3}
        activation = _make_activation()
        return SparseFeatureEvent(
            source_activation=activation,
            feature_source="qwen_scope_sae",
            features=features,
            feature_count=8,
            nonzero_count=len(features),
            extracted_at=datetime.now(timezone.utc).isoformat(),
        )

    def test_construction_fields_accessible(self):
        event = self._make_event()
        self.assertEqual(event.feature_source, "qwen_scope_sae")
        self.assertEqual(event.feature_count, 8)

    def test_schema_version_default(self):
        event = self._make_event()
        self.assertEqual(event.schema_version, "1.0")

    def test_nonzero_count_matches_features_len(self):
        features = {"feature_0": 1.5, "feature_2": 0.3, "feature_7": 0.9}
        event = self._make_event(features=features)
        self.assertEqual(event.nonzero_count, len(event.features))

    def test_extracted_at_is_iso(self):
        event = self._make_event()
        parsed = datetime.fromisoformat(event.extracted_at)
        self.assertIsNotNone(parsed)

    def test_features_is_dict(self):
        event = self._make_event()
        self.assertIsInstance(event.features, dict)

    def test_source_activation_type(self):
        event = self._make_event()
        self.assertIsInstance(event.source_activation, ActivationEvent)


class TestFeatureExtractorNoAdapter(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor()
        self.activation = _make_activation()

    def test_no_adapter_uses_raw_stats(self):
        result = self.extractor.extract(self.activation)
        self.assertEqual(result.feature_source, "raw_stats")

    def test_no_adapter_returns_sparse_feature_event(self):
        result = self.extractor.extract(self.activation)
        self.assertIsInstance(result, SparseFeatureEvent)

    def test_raw_stats_includes_mean_activation(self):
        result = self.extractor._raw_stats_fallback(self.activation)
        self.assertIn("mean_activation", result.features)

    def test_raw_stats_includes_norm_activation(self):
        result = self.extractor._raw_stats_fallback(self.activation)
        self.assertIn("norm_activation", result.features)

    def test_raw_stats_nonzero_count_correct(self):
        result = self.extractor._raw_stats_fallback(self.activation)
        self.assertEqual(result.nonzero_count, len(result.features))

    def test_raw_stats_feature_count_positive(self):
        result = self.extractor._raw_stats_fallback(self.activation)
        self.assertGreater(result.feature_count, 0)

    def test_raw_stats_feature_source_string(self):
        result = self.extractor._raw_stats_fallback(self.activation)
        self.assertEqual(result.feature_source, "raw_stats")

    def test_raw_stats_includes_raw_activation_dim_count(self):
        result = self.extractor._raw_stats_fallback(self.activation)
        self.assertIn("raw_activation_dim_count", result.features)
        self.assertEqual(
            result.features["raw_activation_dim_count"],
            float(np.prod(self.activation.shape)),
        )

    def test_extract_batch_empty(self):
        result = self.extractor.extract_batch([])
        self.assertEqual(result, [])

    def test_extract_batch_correct_length(self):
        activations = [_make_activation(run_id=f"run-{i}") for i in range(5)]
        results = self.extractor.extract_batch(activations)
        self.assertEqual(len(results), 5)

    def test_extract_batch_returns_list(self):
        activations = [_make_activation()]
        results = self.extractor.extract_batch(activations)
        self.assertIsInstance(results, list)


class TestFeatureExtractorWithAdapter(unittest.TestCase):
    def test_loaded_matching_adapter_uses_sae(self):
        adapter = _loaded_adapter()
        extractor = FeatureExtractor(adapter=adapter)
        activation = _make_activation(model_id="Qwen3.5-7B")
        result = extractor.extract(activation)
        self.assertEqual(result.feature_source, "qwen_scope_sae")

    def test_loaded_adapter_nonmatching_model_fallback(self):
        adapter = _loaded_adapter(model_family="Qwen3.5")
        extractor = FeatureExtractor(adapter=adapter)
        activation = _make_activation(model_id="Llama-3-8B")
        result = extractor.extract(activation)
        self.assertEqual(result.feature_source, "raw_stats")

    def test_unloaded_adapter_fallback(self):
        adapter = QwenScopeAdapter(model_family="Qwen3.5")
        extractor = FeatureExtractor(adapter=adapter)
        activation = _make_activation(model_id="Qwen3.5-7B")
        result = extractor.extract(activation)
        self.assertEqual(result.feature_source, "raw_stats")

    def test_sae_result_feature_count_matches_adapter(self):
        adapter = _loaded_adapter(rows=8, cols=4)
        extractor = FeatureExtractor(adapter=adapter)
        activation = _make_activation(model_id="Qwen3.5-7B")
        result = extractor.extract(activation)
        self.assertEqual(result.feature_count, 8)

    def test_sae_result_source_activation_preserved(self):
        adapter = _loaded_adapter()
        extractor = FeatureExtractor(adapter=adapter)
        activation = _make_activation(model_id="Qwen3.5-7B")
        result = extractor.extract(activation)
        self.assertIs(result.source_activation, activation)

    def test_batch_with_mixed_models(self):
        adapter = _loaded_adapter(model_family="Qwen3.5")
        extractor = FeatureExtractor(adapter=adapter)
        activations = [
            _make_activation(model_id="Qwen3.5-7B", run_id="run-1"),
            _make_activation(model_id="Llama-3-8B", run_id="run-2"),
        ]
        results = extractor.extract_batch(activations)
        self.assertEqual(results[0].feature_source, "qwen_scope_sae")
        self.assertEqual(results[1].feature_source, "raw_stats")


if __name__ == "__main__":
    unittest.main()
