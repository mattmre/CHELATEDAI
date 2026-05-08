import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from model_scope_runtime import ActivationEvent
from qwen_scope_adapter import QwenScopeAdapter, QwenScopeLayerSAE, SAECheckpointMetadata


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


def _small_weights(rows: int = 8, cols: int = 4) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((rows, cols)).astype(np.float64)


class TestSAECheckpointMetadata(unittest.TestCase):
    def test_construction_all_fields(self):
        meta = SAECheckpointMetadata(
            model_family="Qwen3.5",
            layer_id="layer0",
            feature_count=256,
            checkpoint_path="/checkpoints/sae.npy",
            loaded_at="2024-01-01T00:00:00+00:00",
        )
        self.assertEqual(meta.model_family, "Qwen3.5")
        self.assertEqual(meta.layer_id, "layer0")
        self.assertEqual(meta.feature_count, 256)

    def test_schema_version_default(self):
        meta = SAECheckpointMetadata(
            model_family="Qwen3.5",
            layer_id="layer0",
            feature_count=8,
            checkpoint_path="/path",
            loaded_at="2024-01-01T00:00:00+00:00",
        )
        self.assertEqual(meta.schema_version, "1.0")

    def test_loaded_at_is_valid_iso(self):
        ts = datetime.now(timezone.utc).isoformat()
        meta = SAECheckpointMetadata(
            model_family="Qwen3.5",
            layer_id="layer0",
            feature_count=8,
            checkpoint_path="/path",
            loaded_at=ts,
        )
        parsed = datetime.fromisoformat(meta.loaded_at)
        self.assertIsNotNone(parsed)

    def test_feature_count_is_int(self):
        meta = SAECheckpointMetadata(
            model_family="Qwen3.5",
            layer_id="layer0",
            feature_count=128,
            checkpoint_path="/path",
            loaded_at="2024-01-01T00:00:00+00:00",
        )
        self.assertIsInstance(meta.feature_count, int)


class TestQwenScopeAdapterInit(unittest.TestCase):
    def test_stores_model_family(self):
        adapter = QwenScopeAdapter(model_family="Qwen3.5")
        self.assertEqual(adapter.model_family, "Qwen3.5")

    def test_default_model_family(self):
        adapter = QwenScopeAdapter()
        self.assertEqual(adapter.model_family, "Qwen3.5")

    def test_not_loaded_initially(self):
        adapter = QwenScopeAdapter()
        self.assertFalse(adapter.is_loaded())

    def test_feature_count_raises_when_not_loaded(self):
        adapter = QwenScopeAdapter()
        with self.assertRaises(RuntimeError):
            adapter.feature_count()

    def test_extract_features_raises_when_not_loaded(self):
        adapter = QwenScopeAdapter()
        with self.assertRaises(RuntimeError):
            adapter.extract_features(_make_activation())


class TestQwenScopeAdapterLoad(unittest.TestCase):
    def _load_adapter(self, rows: int = 8, cols: int = 4) -> QwenScopeAdapter:
        adapter = QwenScopeAdapter(model_family="Qwen3.5")
        weights = _small_weights(rows, cols)
        adapter.load_checkpoint(
            Path("fake_checkpoint.npy"),
            checkpoint_loader=lambda _p: weights,
        )
        return adapter

    def test_load_returns_metadata(self):
        adapter = QwenScopeAdapter(model_family="Qwen3.5")
        weights = _small_weights()
        meta = adapter.load_checkpoint(
            Path("fake.npy"),
            checkpoint_loader=lambda _p: weights,
        )
        self.assertIsInstance(meta, SAECheckpointMetadata)

    def test_is_loaded_after_load(self):
        adapter = self._load_adapter()
        self.assertTrue(adapter.is_loaded())

    def test_feature_count_after_load(self):
        adapter = self._load_adapter(rows=8, cols=4)
        self.assertEqual(adapter.feature_count(), 8)

    def test_feature_count_matches_weights_rows(self):
        adapter = self._load_adapter(rows=16, cols=4)
        self.assertEqual(adapter.feature_count(), 16)

    def test_metadata_feature_count(self):
        adapter = QwenScopeAdapter(model_family="Qwen3.5")
        weights = _small_weights(rows=8, cols=4)
        meta = adapter.load_checkpoint(
            Path("fake.npy"),
            checkpoint_loader=lambda _p: weights,
        )
        self.assertEqual(meta.feature_count, 8)

    def test_metadata_model_family(self):
        adapter = QwenScopeAdapter(model_family="Qwen3.5")
        weights = _small_weights()
        meta = adapter.load_checkpoint(
            Path("fake.npy"),
            checkpoint_loader=lambda _p: weights,
        )
        self.assertEqual(meta.model_family, "Qwen3.5")

    def test_metadata_loaded_at_is_iso(self):
        adapter = QwenScopeAdapter(model_family="Qwen3.5")
        weights = _small_weights()
        meta = adapter.load_checkpoint(
            Path("fake.npy"),
            checkpoint_loader=lambda _p: weights,
        )
        parsed = datetime.fromisoformat(meta.loaded_at)
        self.assertIsNotNone(parsed)

    def test_load_with_dict_checkpoint(self):
        adapter = QwenScopeAdapter(model_family="Qwen3.5")
        weights = _small_weights(rows=4, cols=3)
        ckpt_dict = {"weights": weights, "layer_id": "layer_0"}
        meta = adapter.load_checkpoint(
            Path("fake.npy"),
            checkpoint_loader=lambda _p: ckpt_dict,
        )
        self.assertEqual(meta.feature_count, 4)

    def test_load_dict_layer_id_used(self):
        adapter = QwenScopeAdapter(model_family="Qwen3.5")
        weights = _small_weights(rows=4, cols=3)
        ckpt_dict = {"weights": weights, "layer_id": "my_custom_layer"}
        meta = adapter.load_checkpoint(
            Path("fake.npy"),
            checkpoint_loader=lambda _p: ckpt_dict,
        )
        self.assertEqual(meta.layer_id, "my_custom_layer")


class TestQwenScopeAdapterExtract(unittest.TestCase):
    def setUp(self):
        self.adapter = QwenScopeAdapter(model_family="Qwen3.5")
        weights = _small_weights(rows=8, cols=4)
        self.adapter.load_checkpoint(
            Path("fake.npy"),
            checkpoint_loader=lambda _p: weights,
        )
        self.activation = _make_activation(mean_activation=0.5, norm_activation=1.2)

    def test_extract_returns_dict(self):
        result = self.adapter.extract_features(self.activation)
        self.assertIsInstance(result, dict)

    def test_extract_values_are_floats(self):
        result = self.adapter.extract_features(self.activation)
        for v in result.values():
            self.assertIsInstance(v, float)

    def test_extract_keys_are_feature_strings(self):
        result = self.adapter.extract_features(self.activation)
        for k in result.keys():
            self.assertTrue(k.startswith("feature_"))

    def test_extract_only_positive_values(self):
        result = self.adapter.extract_features(self.activation)
        for v in result.values():
            self.assertGreater(v, 0.0)

    def test_extract_nonempty_with_positive_input(self):
        result = self.adapter.extract_features(self.activation)
        self.assertGreater(len(result), 0)


class TestQwenScopeAdapterSupports(unittest.TestCase):
    def test_supports_matching_model(self):
        adapter = QwenScopeAdapter(model_family="Qwen3.5")
        self.assertTrue(adapter.supports_model("Qwen3.5-7B"))

    def test_does_not_support_other_model(self):
        adapter = QwenScopeAdapter(model_family="Qwen3.5")
        self.assertFalse(adapter.supports_model("Llama-3-8B"))

    def test_supports_exact_family_name(self):
        adapter = QwenScopeAdapter(model_family="Qwen3.5")
        self.assertTrue(adapter.supports_model("Qwen3.5"))

    def test_partial_prefix_no_match(self):
        adapter = QwenScopeAdapter(model_family="Qwen3.5")
        self.assertFalse(adapter.supports_model("Qwen2-7B"))


if __name__ == "__main__":
    unittest.main()
