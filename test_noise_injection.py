import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from antigravity_engine import AntigravityEngine

def test_noise_injection_enabled():
    """Test that noise injection runs without errors when enabled."""
    engine = AntigravityEngine(qdrant_location=":memory:")
    
    # Mock some basic state
    engine.chelation_log = {
        "id1": [np.random.rand(768) for _ in range(5)],  # complex structure
        "id2": [np.random.rand(768) for _ in range(1)]   # less complex
    }
    
    # Mock qdrant retrieve
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
    
    # Mock qdrant upsert and scroll
    engine.qdrant.upsert = MagicMock()
    
    # Spy on torch.randn_like to see if noise is being injected
    with patch("torch.randn_like", wraps=torch.randn_like) as mock_randn:
        engine.run_sedimentation_cycle(threshold=1, epochs=2, noise_injection=0.1)
        
        # Check that randn_like was called (which means noise was injected)
        assert mock_randn.call_count > 0, "Noise injection should call torch.randn_like"
        
def test_noise_injection_disabled():
    """Test that noise injection is skipped when not enabled."""
    engine = AntigravityEngine(qdrant_location=":memory:")
    
    engine.chelation_log = {
        "id1": [np.random.rand(768) for _ in range(5)],
    }
    
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
    
    with patch("torch.randn_like", wraps=torch.randn_like) as mock_randn:
        # Defaults to None, which uses ChelationConfig.NOISE_INJECTION_ENABLED (False by default)
        engine.run_sedimentation_cycle(threshold=1, epochs=2)
        
        assert mock_randn.call_count == 0, "Noise injection should NOT call torch.randn_like if disabled"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
