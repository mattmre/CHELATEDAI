import unittest
from unittest.mock import MagicMock, patch
import numpy as np
try:
    import torch
    from antigravity_engine import AntigravityEngine
    HAS_NOISE_DEPS = True
except ImportError:
    HAS_NOISE_DEPS = False

@unittest.skipUnless(HAS_NOISE_DEPS, "Requires torch and engine dependencies")
class TestNoiseInjection(unittest.TestCase):
    def _make_engine(self, chelation_log):
        engine = AntigravityEngine(qdrant_location=":memory:")
        self.addCleanup(engine.qdrant.close)
        engine.chelation_log = chelation_log

        def mock_retrieve(collection_name, ids, with_vectors):
            points = []
            for id_val in ids:
                point = MagicMock()
                point.id = id_val
                point.vector = np.random.rand(768).tolist()
                point.payload = {"text": "test"}
                points.append(point)
            return points

        engine.qdrant.retrieve = mock_retrieve
        engine.qdrant.upsert = MagicMock()
        return engine

    def test_noise_injection_enabled(self):
        """Test that noise injection runs without errors when enabled."""
        engine = self._make_engine(
            {
                "id1": [np.random.rand(768) for _ in range(5)],
                "id2": [np.random.rand(768) for _ in range(1)],
            }
        )

        with patch("torch.randn_like", wraps=torch.randn_like) as mock_randn:
            engine.run_sedimentation_cycle(threshold=1, epochs=2, noise_injection=0.1)
            self.assertGreater(mock_randn.call_count, 0)

    def test_noise_injection_disabled(self):
        """Test that noise injection is skipped when not enabled."""
        engine = self._make_engine(
            {
                "id1": [np.random.rand(768) for _ in range(5)],
            }
        )

        with patch("torch.randn_like", wraps=torch.randn_like) as mock_randn:
            engine.run_sedimentation_cycle(threshold=1, epochs=2)
            self.assertEqual(mock_randn.call_count, 0)

if __name__ == "__main__":
    unittest.main()
