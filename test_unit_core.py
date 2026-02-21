"""
Unit Tests for ChelatedAI Core Algorithms

Tests the fundamental algorithms without requiring external services (Qdrant, Ollama).
"""

import unittest
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path

# Import components to test
from chelation_adapter import ChelationAdapter
from config import ChelationConfig, get_config


class TestChelationAdapter(unittest.TestCase):
    """Test the ChelationAdapter neural module."""

    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 768
        self.adapter = ChelationAdapter(input_dim=self.input_dim)
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temp files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test that adapter initializes correctly."""
        self.assertIsNotNone(self.adapter)
        self.assertEqual(self.adapter.input_dim, self.input_dim)

    def test_forward_pass_shape(self):
        """Test that forward pass maintains input shape."""
        batch_size = 10
        input_tensor = torch.randn(batch_size, self.input_dim)

        output = self.adapter(input_tensor)

        self.assertEqual(output.shape, input_tensor.shape)
        self.assertEqual(output.dtype, torch.float32)

    def test_identity_initialization(self):
        """Test that adapter starts near identity function."""
        input_tensor = torch.randn(5, self.input_dim)

        output = self.adapter(input_tensor)

        # Output should be very close to normalized input
        input_normalized = torch.nn.functional.normalize(input_tensor, p=2, dim=1)
        cosine_sim = torch.nn.functional.cosine_similarity(output, input_normalized, dim=1)

        # Should be >0.99 similar to identity
        self.assertTrue(torch.all(cosine_sim > 0.99).item(),
                       f"Adapter not identity at init: min cosine sim = {cosine_sim.min().item()}")

    def test_output_normalized(self):
        """Test that output vectors are L2 normalized."""
        input_tensor = torch.randn(10, self.input_dim)

        output = self.adapter(input_tensor)

        norms = torch.norm(output, p=2, dim=1)

        # All norms should be ~1.0
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
                       f"Output not normalized: norms range {norms.min():.6f} to {norms.max():.6f}")

    def test_save_and_load(self):
        """Test saving and loading adapter weights."""
        save_path = self.temp_dir / "test_adapter.pt"

        # Modify adapter slightly
        input_tensor = torch.randn(5, self.input_dim)
        target = torch.randn(5, self.input_dim)
        target = torch.nn.functional.normalize(target, p=2, dim=1)

        optimizer = torch.optim.Adam(self.adapter.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()

        for _ in range(10):
            optimizer.zero_grad()
            output = self.adapter(input_tensor)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Save
        self.adapter.save(str(save_path))
        self.assertTrue(save_path.exists())

        # Load into new adapter
        new_adapter = ChelationAdapter(input_dim=self.input_dim)
        success = new_adapter.load(str(save_path))
        self.assertTrue(success)

        # Outputs should match
        output1 = self.adapter(input_tensor)
        output2 = new_adapter(input_tensor)

        self.assertTrue(torch.allclose(output1, output2, atol=1e-5))

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file returns False."""
        fake_path = self.temp_dir / "nonexistent.pt"
        success = self.adapter.load(str(fake_path))
        self.assertFalse(success)

    def test_dimension_mismatch_handling(self):
        """Test that loading wrong dimension weights fails gracefully."""
        # Create and save adapter with different dimension
        wrong_adapter = ChelationAdapter(input_dim=512)
        save_path = self.temp_dir / "wrong_dim.pt"
        wrong_adapter.save(str(save_path))

        # Try to load into different dimension adapter
        success = self.adapter.load(str(save_path))
        self.assertFalse(success)

    def test_save_path_traversal_blocked(self):
        """Test that path traversal attempts are blocked in save()."""
        traversal_path = self.temp_dir / ".." / "escaping_adapter.pt"
        with self.assertRaises(ValueError) as cm:
            self.adapter.save(str(traversal_path))
        self.assertIn("traversal", str(cm.exception).lower())

    def test_load_path_traversal_blocked(self):
        """Test that path traversal attempts are blocked in load()."""
        traversal_path = self.temp_dir / ".." / "malicious.pt"
        with self.assertRaises(ValueError) as cm:
            self.adapter.load(str(traversal_path))
        self.assertIn("traversal", str(cm.exception).lower())

    def test_1d_input_returns_1d_output(self):
        """Test that 1D input returns 1D output of same shape (F-024)."""
        input_1d = torch.randn(self.input_dim)
        
        output = self.adapter(input_1d)
        
        # Output should be 1D with same shape
        self.assertEqual(output.dim(), 1)
        self.assertEqual(output.shape, input_1d.shape)
        self.assertEqual(output.dtype, torch.float32)

    def test_1d_output_normalized(self):
        """Test that 1D output is properly normalized (F-024)."""
        input_1d = torch.randn(self.input_dim)
        
        output = self.adapter(input_1d)
        
        # Output should be L2 normalized
        norm = torch.norm(output, p=2)
        self.assertTrue(torch.allclose(norm, torch.tensor(1.0), atol=1e-5),
                       f"1D output not normalized: norm = {norm:.6f}")

    def test_1d_matches_2d_batch_behavior(self):
        """Test that 1D input behavior matches corresponding 2D single-batch (F-024)."""
        input_1d = torch.randn(self.input_dim)
        
        # Run as 1D
        output_1d = self.adapter(input_1d)
        
        # Run same input as 2D batch of 1
        input_2d = input_1d.unsqueeze(0)
        output_2d = self.adapter(input_2d)
        output_2d_squeezed = output_2d.squeeze(0)
        
        # Results should match
        self.assertTrue(torch.allclose(output_1d, output_2d_squeezed, atol=1e-6),
                       f"1D and 2D behavior mismatch: max diff = {(output_1d - output_2d_squeezed).abs().max():.6e}")

    def test_invalid_rank_raises_error(self):
        """Test that invalid tensor ranks raise ValueError (F-024)."""
        # 0D tensor (scalar)
        input_0d = torch.tensor(3.14)
        with self.assertRaises(ValueError) as cm:
            self.adapter(input_0d)
        self.assertIn("1D or 2D", str(cm.exception))
        
        # 3D tensor
        input_3d = torch.randn(2, 3, self.input_dim)
        with self.assertRaises(ValueError) as cm:
            self.adapter(input_3d)
        self.assertIn("1D or 2D", str(cm.exception))


class TestChelationConfig(unittest.TestCase):
    """Test the configuration management system."""

    def test_path_portability(self):
        """Test that paths are platform-independent."""
        db_path = ChelationConfig.get_db_path("SciFact")

        # Should be a Path object
        self.assertIsInstance(db_path, Path)

        # Should contain proper components
        self.assertIn("scifact", str(db_path).lower())
        self.assertIn("evolution", str(db_path).lower())

    def test_validate_chelation_p(self):
        """Test chelation_p validation and clamping."""
        # Valid value
        self.assertEqual(ChelationConfig.validate_chelation_p(85), 85)

        # Clamp high
        self.assertEqual(ChelationConfig.validate_chelation_p(150), 100)

        # Clamp low
        self.assertEqual(ChelationConfig.validate_chelation_p(-10), 0)

    def test_validate_learning_rate(self):
        """Test learning_rate validation and clamping."""
        # Valid value
        self.assertEqual(ChelationConfig.validate_learning_rate(0.01), 0.01)

        # Clamp high
        self.assertEqual(ChelationConfig.validate_learning_rate(10.0), 1.0)

        # Clamp low
        self.assertEqual(ChelationConfig.validate_learning_rate(0.00001), 0.0001)

    def test_validate_epochs(self):
        """Test epochs validation and clamping."""
        # Valid value
        self.assertEqual(ChelationConfig.validate_epochs(10), 10)

        # Clamp high
        self.assertEqual(ChelationConfig.validate_epochs(200), 100)

        # Clamp low
        self.assertEqual(ChelationConfig.validate_epochs(0), 1)

    def test_get_preset_chelation(self):
        """Test retrieving chelation presets."""
        preset = ChelationConfig.get_preset("balanced", "chelation")

        self.assertIn("chelation_p", preset)
        self.assertIn("chelation_threshold", preset)
        self.assertIn("description", preset)

    def test_get_preset_adapter(self):
        """Test retrieving adapter presets."""
        preset = ChelationConfig.get_preset("medium", "adapter")

        self.assertIn("learning_rate", preset)
        self.assertIn("epochs", preset)
        self.assertIn("threshold", preset)

    def test_get_preset_invalid(self):
        """Test that invalid preset raises ValueError."""
        with self.assertRaises(ValueError):
            ChelationConfig.get_preset("nonexistent", "chelation")

    def test_save_and_load_config(self):
        """Test saving and loading configuration files."""
        temp_dir = Path(tempfile.mkdtemp())
        config_path = temp_dir / "test_config.json"

        try:
            config = {
                "chelation_p": 90,
                "learning_rate": 0.01,
                "epochs": 20
            }

            # Save
            ChelationConfig.save_to_file(config, config_path)
            self.assertTrue(config_path.exists())

            # Load
            loaded_config = ChelationConfig.load_from_file(config_path)
            self.assertEqual(loaded_config, config)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_config_load_path_traversal_blocked(self):
        """Test that path traversal is blocked in load_from_file."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Attempt path traversal
            traversal_path = temp_dir / ".." / "evil_config.json"
            with self.assertRaises(ValueError) as cm:
                ChelationConfig.load_from_file(traversal_path)
            self.assertIn("traversal", str(cm.exception).lower())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_config_save_path_traversal_blocked(self):
        """Test that path traversal is blocked in save_to_file."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            config = {"test": "data"}
            traversal_path = temp_dir / ".." / "evil_config.json"
            with self.assertRaises(ValueError) as cm:
                ChelationConfig.save_to_file(config, traversal_path)
            self.assertIn("traversal", str(cm.exception).lower())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_validate_max_depth(self):
        """Test max_depth validation and clamping (F-053)."""
        # Valid value
        self.assertEqual(ChelationConfig.validate_max_depth(5), 5)
        
        # Clamp high
        self.assertEqual(ChelationConfig.validate_max_depth(20), 10)
        
        # Clamp low
        self.assertEqual(ChelationConfig.validate_max_depth(0), 1)
    
    def test_get_preset_rlm(self):
        """Test retrieving RLM presets (F-053)."""
        preset = ChelationConfig.get_preset("balanced", "rlm")
        
        self.assertIn("max_depth", preset)
        self.assertIn("min_support", preset)
        self.assertIn("description", preset)
    
    def test_get_preset_sedimentation(self):
        """Test retrieving sedimentation presets (F-053)."""
        preset = ChelationConfig.get_preset("balanced", "sedimentation")
        
        self.assertIn("collapse_threshold", preset)
        self.assertIn("push_magnitude", preset)
        self.assertIn("description", preset)
    
    def test_get_preset_invalid_type(self):
        """Test that invalid preset_type raises ValueError (F-053)."""
        with self.assertRaises(ValueError) as cm:
            ChelationConfig.get_preset("balanced", "invalid_type")
        self.assertIn("Invalid preset_type", str(cm.exception))
        self.assertIn("chelation", str(cm.exception))
        self.assertIn("adapter", str(cm.exception))
        self.assertIn("rlm", str(cm.exception))
        self.assertIn("sedimentation", str(cm.exception))
    
    def test_get_config_default(self):
        """Test get_config() returns expected default keys (F-053)."""
        config = get_config()
        
        self.assertIn("chelation_p", config)
        self.assertIn("chelation_threshold", config)
        self.assertIn("learning_rate", config)
        self.assertIn("epochs", config)
        self.assertIn("scout_k", config)
    
    def test_get_config_preset(self):
        """Test get_config('balanced') returns chelation preset keys (F-053)."""
        config = get_config("balanced")
        
        self.assertIn("chelation_p", config)
        self.assertIn("chelation_threshold", config)
        self.assertIn("description", config)


class TestChelationAlgorithms(unittest.TestCase):
    """Test core chelation algorithms (without Qdrant dependency)."""

    def test_variance_calculation(self):
        """Test that variance-based masking works correctly."""
        # Create synthetic cluster with varying dimensions
        n_samples = 100
        n_dims = 10

        # Dims 0-4: low variance (stable)
        # Dims 5-9: high variance (toxic)
        cluster = np.random.randn(n_samples, n_dims)
        cluster[:, 0:5] *= 0.1  # Low variance
        cluster[:, 5:10] *= 2.0  # High variance

        # Calculate variance
        dim_variance = np.var(cluster, axis=0)

        # Verify low variance dims have lower variance
        self.assertTrue(np.all(dim_variance[0:5] < dim_variance[5:10]))

        # Test percentile-based masking (keep bottom 50%)
        threshold = np.percentile(dim_variance, 50)
        mask = (dim_variance < threshold).astype(float)

        # Should mask out mostly high-variance dims
        num_low_kept = np.sum(mask[0:5])
        num_high_kept = np.sum(mask[5:10])

        self.assertGreater(num_low_kept, num_high_kept)

    def test_spectral_centering(self):
        """Test center-of-mass centering algorithm."""
        # Create cluster
        n_samples = 50
        n_dims = 128

        # Cluster centered at some offset
        center = np.random.randn(n_dims)
        cluster = center + np.random.randn(n_samples, n_dims) * 0.1

        # Calculate center of mass
        computed_center = np.mean(cluster, axis=0)

        # Should be very close to original center
        distance = np.linalg.norm(computed_center - center)
        self.assertLess(distance, 0.5)

        # Center the cluster
        centered_cluster = cluster - computed_center

        # New center should be at origin
        new_center = np.mean(centered_cluster, axis=0)
        new_center_norm = np.linalg.norm(new_center)

        self.assertLess(new_center_norm, 0.01)

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        # Test vectors
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        v3 = np.array([0.0, 1.0, 0.0])
        v4 = np.array([-1.0, 0.0, 0.0])

        def cosine_sim(a, b):
            """Manual cosine similarity."""
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return np.dot(a, b) / (norm_a * norm_b)

        # Same vector: similarity = 1
        self.assertAlmostEqual(cosine_sim(v1, v2), 1.0)

        # Orthogonal: similarity = 0
        self.assertAlmostEqual(cosine_sim(v1, v3), 0.0)

        # Opposite: similarity = -1
        self.assertAlmostEqual(cosine_sim(v1, v4), -1.0)

    def test_homeostatic_update_direction(self):
        """Test that homeostatic updates push vectors away from noise."""
        # Original vector
        v_orig = np.array([1.0, 0.0, 0.0])

        # Noise center (where vector keeps collapsing to)
        noise_center = np.array([0.0, 1.0, 0.0])

        # Homeostatic update: push away from noise
        diff = v_orig - noise_center
        diff_norm = diff / (np.linalg.norm(diff) + 1e-9)
        v_new = v_orig + (diff_norm * 0.1)

        # Distance from noise should increase
        dist_before = np.linalg.norm(v_orig - noise_center)
        dist_after = np.linalg.norm(v_new - noise_center)

        self.assertGreater(dist_after, dist_before)


class TestIDManagement(unittest.TestCase):
    """Test ID type handling and conversions."""

    def test_string_to_int_conversion(self):
        """Test converting string IDs to integers."""
        # Numeric string
        self.assertEqual(int("123"), 123)

        # Non-numeric should raise
        with self.assertRaises(ValueError):
            int("abc123")

    def test_uuid5_deterministic(self):
        """Test that UUID5 hashing is deterministic."""
        import uuid

        doc_id = "some_document_id"

        # Same input should give same UUID
        uuid1 = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))
        uuid2 = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))

        self.assertEqual(uuid1, uuid2)

        # Different input should give different UUID
        uuid3 = str(uuid.uuid5(uuid.NAMESPACE_DNS, "different_id"))
        self.assertNotEqual(uuid1, uuid3)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
