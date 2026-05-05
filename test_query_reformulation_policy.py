"""Focused tests for learned query reformulation gate support."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from query_reformulator import QueryReformulation, should_apply_reformulation

try:
    import torch  # noqa: F401
    from antigravity_engine import AntigravityEngine

    HAS_ENGINE_DEPS = True
except ImportError:
    HAS_ENGINE_DEPS = False


class TestStructuredReformulationPolicy(unittest.TestCase):
    def test_string_policies_still_work(self):
        self.assertTrue(should_apply_reformulation("plain lookup", "always"))
        self.assertFalse(should_apply_reformulation("plain lookup", "never"))

    def test_linear_policy_uses_lexical_features(self):
        policy = {
            "type": "linear",
            "weights": {
                "claim_cue_count": 2.0,
                "token_count": 0.1,
            },
            "bias": -0.5,
            "threshold": 1.0,
        }

        self.assertTrue(should_apply_reformulation("magnesium increase absorption", policy))

    def test_linear_policy_fails_closed_for_unknown_features(self):
        policy = {
            "type": "linear",
            "weights": {"baseline_score_margin": 1.0},
            "threshold": 0.0,
        }

        self.assertFalse(should_apply_reformulation("magnesium increase absorption", policy))

    def test_linear_classifier_supports_query_prefixed_feature_names(self):
        policy = {
            "type": "linear_classifier",
            "features": ["query_claim_cue_count", "query_token_count"],
            "means": [0.0, 0.0],
            "scales": [1.0, 10.0],
            "weights": [4.0, 1.0],
            "intercept": -1.0,
            "threshold": 0.5,
        }

        self.assertTrue(should_apply_reformulation("magnesium increase absorption", policy))

    def test_linear_classifier_fails_closed_for_shape_mismatch(self):
        policy = {
            "type": "linear_classifier",
            "features": ["query_claim_cue_count"],
            "means": [0.0],
            "scales": [1.0, 1.0],
            "weights": [3.0],
            "threshold": 0.5,
        }

        self.assertFalse(should_apply_reformulation("magnesium increase absorption", policy))

    def test_structured_policy_respects_advisory_only_mode(self):
        policy = {
            "type": "linear_classifier",
            "deployment_mode": "advisory_only",
            "runtime_compatible": False,
            "features": ["query_claim_cue_count", "query_token_count"],
            "means": [0.0, 0.0],
            "scales": [1.0, 10.0],
            "weights": [4.0, 1.0],
            "intercept": -1.0,
            "threshold": 0.5,
        }

        self.assertFalse(should_apply_reformulation("magnesium increase absorption", policy))


@unittest.skipUnless(HAS_ENGINE_DEPS, "Requires AntigravityEngine dependencies")
class TestAntigravityEngineReformulationPolicy(unittest.TestCase):
    def setUp(self):
        self.logger_patcher = patch("antigravity_engine.get_logger")
        self.adapter_patcher = patch("antigravity_engine.create_adapter")
        self.backend_patcher = patch("antigravity_engine.create_embedding_backend")
        self.vector_store_patcher = patch("antigravity_engine.create_vector_store")

        self.mock_get_logger = self.logger_patcher.start()
        self.mock_adapter_cls = self.adapter_patcher.start()
        self.mock_backend_factory = self.backend_patcher.start()
        self.mock_vector_store_factory = self.vector_store_patcher.start()

        self.mock_logger = MagicMock()
        self.mock_get_logger.return_value = self.mock_logger

        self.mock_adapter = MagicMock()
        self.mock_adapter.load.return_value = False
        self.mock_adapter.side_effect = lambda tensor: tensor
        self.mock_adapter_cls.return_value = self.mock_adapter

        self.mock_backend = MagicMock()
        self.mock_backend.vector_size = 768
        self.mock_backend.embed_raw.side_effect = (
            lambda texts: np.random.randn(len(texts), 768).astype(np.float32)
        )
        self.mock_backend_factory.return_value = self.mock_backend

        self.mock_vector_store = MagicMock()
        self.mock_vector_store.collection_exists.return_value = False
        self.mock_vector_store_factory.return_value = self.mock_vector_store

    def tearDown(self):
        self.vector_store_patcher.stop()
        self.backend_patcher.stop()
        self.adapter_patcher.stop()
        self.logger_patcher.stop()

    def _make_engine(self, **kwargs):
        return AntigravityEngine(
            qdrant_location=":memory:",
            model_name="all-MiniLM-L6-v2",
            use_quantization=True,
            use_centering=False,
            **kwargs,
        )

    def test_enable_query_reformulation_snapshots_policy_config(self):
        engine = self._make_engine()
        policy = {
            "type": "linear",
            "weights": {"claim_cue_count": 3.0},
            "bias": -1.0,
            "threshold": 0.0,
        }

        engine.enable_query_reformulation(policy=policy)
        policy["weights"]["claim_cue_count"] = -10.0

        self.assertEqual(engine._query_reformulator_policy["weights"]["claim_cue_count"], 3.0)
        self.assertTrue(
            should_apply_reformulation(
                "magnesium increase absorption",
                engine._query_reformulator_policy,
            )
        )

    def test_run_inference_uses_structured_policy_gate(self):
        engine = self._make_engine()
        policy = {
            "type": "linear_classifier",
            "features": ["query_claim_cue_count", "query_token_count"],
            "means": [0.0, 0.0],
            "scales": [1.0, 10.0],
            "weights": [4.0, 1.0],
            "intercept": -1.0,
            "threshold": 0.5,
        }

        engine.enable_query_reformulation(max_variants=2, policy=policy)
        engine._query_reformulator = MagicMock()
        engine._query_reformulator.reformulate.return_value = [
            QueryReformulation(
                text="magnesium increase absorption",
                strategy="original",
            )
        ]

        qvec = np.random.randn(768)
        engine.embed = MagicMock(return_value=np.array([qvec]))
        points = [SimpleNamespace(id=i, vector=np.ones(768).tolist(), score=0.9) for i in range(10)]
        self.mock_vector_store.query_points.return_value = SimpleNamespace(points=points)

        _, chel_top, _, _ = engine.run_inference("magnesium increase absorption")

        engine._query_reformulator.reformulate.assert_called_once_with(
            "magnesium increase absorption",
            max_variants=2,
        )
        self.assertEqual(chel_top, list(range(10)))
        diagnostics = engine.get_last_runtime_diagnostics()
        self.assertEqual(diagnostics["runtime"]["action"], "REFORMULATE")
        self.assertEqual(diagnostics["query_reformulation"]["variant_count"], 1)
