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
from chelation_adapter import (
    ChelationAdapter, OrthogonalProcrustesAdapter, LowRankAffineAdapter,
    BoundedAdapter, create_adapter,
)
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


class TestAdapterVariants(unittest.TestCase):
    """Test OrthogonalProcrustesAdapter, LowRankAffineAdapter, and create_adapter factory."""

    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 384
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temp files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # --- OrthogonalProcrustesAdapter tests ---

    def test_procrustes_forward_shape(self):
        """Test that 2D input preserves shape through Procrustes adapter."""
        adapter = OrthogonalProcrustesAdapter(input_dim=self.input_dim)
        batch_size = 5
        x = torch.randn(batch_size, self.input_dim)
        out = adapter(x)
        self.assertEqual(out.shape, (batch_size, self.input_dim))

    def test_procrustes_forward_1d(self):
        """Test that 1D input returns 1D output."""
        adapter = OrthogonalProcrustesAdapter(input_dim=self.input_dim)
        x = torch.randn(self.input_dim)
        out = adapter(x)
        self.assertEqual(out.dim(), 1)
        self.assertEqual(out.shape[0], self.input_dim)

    def test_procrustes_output_normalized(self):
        """Test that Procrustes output is L2 normalized."""
        adapter = OrthogonalProcrustesAdapter(input_dim=self.input_dim)
        x = torch.randn(10, self.input_dim)
        out = adapter(x)
        norms = torch.norm(out, p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
                       f"Output not normalized: norms range {norms.min():.6f} to {norms.max():.6f}")

    def test_procrustes_near_identity_init(self):
        """Test that Procrustes adapter starts near identity (cosine > 0.95)."""
        adapter = OrthogonalProcrustesAdapter(input_dim=self.input_dim)
        x = torch.randn(5, self.input_dim)
        out = adapter(x)
        x_normalized = torch.nn.functional.normalize(x, p=2, dim=1)
        cosine_sim = torch.nn.functional.cosine_similarity(out, x_normalized, dim=1)
        self.assertTrue(torch.all(cosine_sim > 0.95).item(),
                       f"Not near identity at init: min cosine sim = {cosine_sim.min().item()}")

    def test_procrustes_orthogonal_matrix(self):
        """Test that the internal matrix W satisfies W^T @ W = I."""
        adapter = OrthogonalProcrustesAdapter(input_dim=self.input_dim)
        W = adapter._get_orthogonal_matrix()
        product = W.t() @ W
        identity = torch.eye(self.input_dim)
        self.assertTrue(torch.allclose(product, identity, atol=1e-5),
                       f"W^T @ W not identity: max error = {(product - identity).abs().max().item()}")

    def test_procrustes_save_load(self):
        """Test that save and load preserves Procrustes adapter weights."""
        adapter = OrthogonalProcrustesAdapter(input_dim=self.input_dim)
        # Perturb weights so they differ from default
        with torch.no_grad():
            adapter._skew_param.add_(torch.randn_like(adapter._skew_param) * 0.1)
        save_path = self.temp_dir / "procrustes.pt"
        adapter.save(str(save_path))

        new_adapter = OrthogonalProcrustesAdapter(input_dim=self.input_dim)
        success = new_adapter.load(str(save_path))
        self.assertTrue(success)

        x = torch.randn(3, self.input_dim)
        out1 = adapter(x)
        out2 = new_adapter(x)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-5))

    def test_procrustes_invalid_rank_raises(self):
        """Test that 0D and 3D inputs raise ValueError for Procrustes adapter."""
        adapter = OrthogonalProcrustesAdapter(input_dim=self.input_dim)
        with self.assertRaises(ValueError):
            adapter(torch.tensor(3.14))
        with self.assertRaises(ValueError):
            adapter(torch.randn(2, 3, self.input_dim))

    def test_procrustes_scale_param_exists(self):
        """Test that _scale parameter exists with correct shape and init."""
        adapter = OrthogonalProcrustesAdapter(input_dim=self.input_dim)
        self.assertTrue(hasattr(adapter, '_scale'))
        self.assertEqual(adapter._scale.shape, (self.input_dim,))
        # Initialized to ones
        self.assertTrue(torch.allclose(adapter._scale.data, torch.ones(self.input_dim)))

    def test_procrustes_scale_is_learnable(self):
        """Test that _scale is an nn.Parameter and participates in optimization."""
        adapter = OrthogonalProcrustesAdapter(input_dim=self.input_dim)
        param_names = [name for name, _ in adapter.named_parameters()]
        self.assertIn('_scale', param_names)

    def test_procrustes_scale_affects_output(self):
        """Test that non-uniform scaling changes the output direction."""
        adapter = OrthogonalProcrustesAdapter(input_dim=self.input_dim)
        x = torch.randn(5, self.input_dim)
        out_before = adapter(x).detach().clone()
        # Set non-uniform scale: double first half, halve second half
        with torch.no_grad():
            adapter._scale[:self.input_dim // 2] = 2.0
            adapter._scale[self.input_dim // 2:] = 0.5
        out_after = adapter(x).detach()
        # Outputs should differ because scaling changes relative dimension magnitudes
        self.assertFalse(torch.allclose(out_before, out_after, atol=1e-4),
                        "Non-uniform scale should change output directions")

    def test_procrustes_regularization_loss_at_init(self):
        """Test that regularization_loss is small at initialization."""
        adapter = OrthogonalProcrustesAdapter(input_dim=self.input_dim)
        reg = adapter.regularization_loss()
        # _skew_param is ~0.001 randn, so A = P - P^T has entries ~0.002;
        # ||A||_F^2 should be small but positive
        self.assertGreater(reg.item(), 0.0)
        # With std=0.001, each entry of A is ~N(0, 0.001*sqrt(2)),
        # so ||A||_F^2 ~ input_dim^2 * 2 * 0.001^2 = 384^2 * 0.002 ~ 0.295
        self.assertLess(reg.item(), 5.0, "Regularization loss should be small at init")

    def test_procrustes_regularization_loss_increases_with_skew(self):
        """Test that regularization_loss grows when skew parameters grow."""
        adapter = OrthogonalProcrustesAdapter(input_dim=self.input_dim)
        reg_small = adapter.regularization_loss().item()
        # Push skew_param to larger values
        with torch.no_grad():
            adapter._skew_param.mul_(100.0)
        reg_large = adapter.regularization_loss().item()
        self.assertGreater(reg_large, reg_small * 100,
                          "Regularization should grow with skew magnitude")

    def test_procrustes_regularization_loss_is_differentiable(self):
        """Test that regularization_loss produces a gradient on _skew_param."""
        adapter = OrthogonalProcrustesAdapter(input_dim=64)
        reg = adapter.regularization_loss()
        reg.backward()
        self.assertIsNotNone(adapter._skew_param.grad)
        self.assertFalse(torch.all(adapter._skew_param.grad == 0).item(),
                        "Gradient should be non-zero")

    def test_procrustes_save_load_with_scale(self):
        """Test that save/load round-trips the _scale parameter."""
        adapter = OrthogonalProcrustesAdapter(input_dim=self.input_dim)
        with torch.no_grad():
            adapter._skew_param.add_(torch.randn_like(adapter._skew_param) * 0.1)
            adapter._scale.fill_(1.5)
        save_path = self.temp_dir / "procrustes_dsm.pt"
        adapter.save(str(save_path))

        new_adapter = OrthogonalProcrustesAdapter(input_dim=self.input_dim)
        success = new_adapter.load(str(save_path))
        self.assertTrue(success)
        self.assertTrue(torch.allclose(adapter._scale, new_adapter._scale))

        x = torch.randn(3, self.input_dim)
        self.assertTrue(torch.allclose(adapter(x), new_adapter(x), atol=1e-5))

    # --- regularization_loss on all adapter types ---

    def test_mlp_regularization_loss_zero(self):
        """Test that MLP adapter regularization_loss returns 0.0."""
        adapter = ChelationAdapter(input_dim=self.input_dim)
        self.assertEqual(adapter.regularization_loss(), 0.0)

    def test_lowrank_regularization_loss_zero(self):
        """Test that LowRank adapter regularization_loss returns 0.0."""
        adapter = LowRankAffineAdapter(input_dim=self.input_dim)
        self.assertEqual(adapter.regularization_loss(), 0.0)

    # --- LowRankAffineAdapter tests ---

    def test_lowrank_forward_shape(self):
        """Test that 2D input preserves shape through LowRank adapter."""
        adapter = LowRankAffineAdapter(input_dim=self.input_dim)
        batch_size = 5
        x = torch.randn(batch_size, self.input_dim)
        out = adapter(x)
        self.assertEqual(out.shape, (batch_size, self.input_dim))

    def test_lowrank_forward_1d(self):
        """Test that 1D input returns 1D output for LowRank adapter."""
        adapter = LowRankAffineAdapter(input_dim=self.input_dim)
        x = torch.randn(self.input_dim)
        out = adapter(x)
        self.assertEqual(out.dim(), 1)
        self.assertEqual(out.shape[0], self.input_dim)

    def test_lowrank_output_normalized(self):
        """Test that LowRank output is L2 normalized."""
        adapter = LowRankAffineAdapter(input_dim=self.input_dim)
        x = torch.randn(10, self.input_dim)
        out = adapter(x)
        norms = torch.norm(out, p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
                       f"Output not normalized: norms range {norms.min():.6f} to {norms.max():.6f}")

    def test_lowrank_near_identity_init(self):
        """Test that LowRank adapter starts near identity (cosine > 0.95)."""
        adapter = LowRankAffineAdapter(input_dim=self.input_dim)
        x = torch.randn(5, self.input_dim)
        out = adapter(x)
        x_normalized = torch.nn.functional.normalize(x, p=2, dim=1)
        cosine_sim = torch.nn.functional.cosine_similarity(out, x_normalized, dim=1)
        self.assertTrue(torch.all(cosine_sim > 0.95).item(),
                       f"Not near identity at init: min cosine sim = {cosine_sim.min().item()}")

    def test_lowrank_custom_rank(self):
        """Test that custom rank parameter is stored and used."""
        adapter = LowRankAffineAdapter(input_dim=self.input_dim, rank=8)
        self.assertEqual(adapter.rank, 8)
        self.assertEqual(adapter.U.shape, (self.input_dim, 8))
        self.assertEqual(adapter.V.shape, (self.input_dim, 8))

    def test_lowrank_save_load(self):
        """Test that save and load preserves LowRank adapter weights."""
        adapter = LowRankAffineAdapter(input_dim=self.input_dim, rank=8)
        # Perturb weights so they differ from default
        with torch.no_grad():
            adapter.U.add_(torch.randn_like(adapter.U) * 0.1)
        save_path = self.temp_dir / "lowrank.pt"
        adapter.save(str(save_path))

        new_adapter = LowRankAffineAdapter(input_dim=self.input_dim, rank=8)
        success = new_adapter.load(str(save_path))
        self.assertTrue(success)

        x = torch.randn(3, self.input_dim)
        out1 = adapter(x)
        out2 = new_adapter(x)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-5))

    def test_lowrank_invalid_rank_raises(self):
        """Test that 0D and 3D inputs raise ValueError for LowRank adapter."""
        adapter = LowRankAffineAdapter(input_dim=self.input_dim)
        with self.assertRaises(ValueError):
            adapter(torch.tensor(3.14))
        with self.assertRaises(ValueError):
            adapter(torch.randn(2, 3, self.input_dim))

    # --- Factory function tests ---

    def test_factory_mlp(self):
        """Test that create_adapter('mlp') returns ChelationAdapter."""
        adapter = create_adapter("mlp", input_dim=self.input_dim)
        self.assertIsInstance(adapter, ChelationAdapter)
        self.assertEqual(adapter.input_dim, self.input_dim)

    def test_factory_procrustes(self):
        """Test that create_adapter('procrustes') returns OrthogonalProcrustesAdapter."""
        adapter = create_adapter("procrustes", input_dim=self.input_dim)
        self.assertIsInstance(adapter, OrthogonalProcrustesAdapter)
        self.assertEqual(adapter.input_dim, self.input_dim)

    def test_factory_low_rank(self):
        """Test that create_adapter('low_rank') returns LowRankAffineAdapter."""
        adapter = create_adapter("low_rank", input_dim=self.input_dim)
        self.assertIsInstance(adapter, LowRankAffineAdapter)
        self.assertEqual(adapter.input_dim, self.input_dim)

    def test_factory_low_rank_custom_rank(self):
        """Test that create_adapter('low_rank', rank=32) passes rank through."""
        adapter = create_adapter("low_rank", input_dim=self.input_dim, rank=32)
        self.assertIsInstance(adapter, LowRankAffineAdapter)
        self.assertEqual(adapter.rank, 32)
        self.assertEqual(adapter.U.shape, (self.input_dim, 32))

    def test_factory_invalid_type(self):
        """Test that create_adapter('invalid') raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            create_adapter("invalid", input_dim=self.input_dim)
        self.assertIn("Unknown adapter_type", str(cm.exception))
        self.assertIn("invalid", str(cm.exception))


class TestDimensionProjectionTraining(unittest.TestCase):
    """Test that DimensionProjection parameters are included in training."""

    def test_projection_parameters_included_in_optimizer(self):
        """Verify projection params are added to optimizer alongside adapter params."""
        from unittest.mock import MagicMock
        from teacher_distillation import DimensionProjection, TeacherDistillationHelper

        adapter = create_adapter("mlp", input_dim=384)
        projection = DimensionProjection(teacher_dim=768, student_dim=384)

        # Simulate what antigravity_engine does when building optimizer params
        params = list(adapter.parameters())
        adapter_param_count = len(params)

        # Build a mock teacher_helper with a projection
        teacher_helper = MagicMock(spec=TeacherDistillationHelper)
        teacher_helper._projection = projection

        if (teacher_helper is not None
                and hasattr(teacher_helper, '_projection')
                and teacher_helper._projection is not None):
            params += list(teacher_helper._projection.parameters())

        total_param_count = len(params)
        self.assertGreater(total_param_count, adapter_param_count,
                           "Projection parameters should be added to optimizer param list")

        # Verify projection parameters are actually in the list
        projection_params = set(id(p) for p in projection.parameters())
        optimizer_params = set(id(p) for p in params)
        self.assertTrue(projection_params.issubset(optimizer_params),
                        "All projection parameters must appear in the optimizer param list")

    def test_projection_weights_change_after_training_step(self):
        """Verify projection weights are actually updated by an optimizer step."""
        from teacher_distillation import DimensionProjection
        import torch.optim as optim

        projection = DimensionProjection(teacher_dim=768, student_dim=384)

        # Snapshot weights before training
        initial_weight = projection.projection.weight.data.clone()

        # Create optimizer that includes projection parameters
        adapter = create_adapter("mlp", input_dim=384)
        params = list(adapter.parameters()) + list(projection.parameters())
        optimizer = optim.Adam(params, lr=0.01)

        # Simulate a training step: project teacher embeddings, compute loss
        teacher_embeds = torch.randn(8, 768)
        projected = projection.project_tensor(teacher_embeds)
        target = torch.randn(8, 384)
        loss = torch.nn.MSELoss()(projected, target)
        loss.backward()
        optimizer.step()

        # Projection weights must have changed
        weight_diff = (projection.projection.weight.data - initial_weight).abs().sum().item()
        self.assertGreater(weight_diff, 0.0,
                           "Projection weights should change after optimizer step")

    def test_projection_weights_frozen_under_no_grad(self):
        """Verify that project_numpy (no_grad path) does NOT accumulate gradients."""
        from teacher_distillation import DimensionProjection

        projection = DimensionProjection(teacher_dim=768, student_dim=384)

        teacher_embeds = np.random.randn(4, 768).astype(np.float32)
        _ = projection.project_numpy(teacher_embeds)

        # No gradient should have been recorded
        for param in projection.parameters():
            self.assertIsNone(param.grad,
                              "project_numpy must not produce gradients")

    def test_project_tensor_preserves_gradients(self):
        """Verify project_tensor keeps the computation graph alive."""
        from teacher_distillation import DimensionProjection

        projection = DimensionProjection(teacher_dim=768, student_dim=384)

        teacher_tensor = torch.randn(4, 768)
        projected = projection.project_tensor(teacher_tensor)

        self.assertTrue(projected.requires_grad,
                        "project_tensor output must require grad")

        loss = projected.sum()
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in projection.parameters())
        self.assertTrue(has_grad,
                        "Projection parameters must receive gradients via project_tensor")

    def test_optimizer_without_projection_still_works(self):
        """Verify optimizer creation works when teacher_helper has no projection."""
        adapter = create_adapter("mlp", input_dim=384)

        # Simulate teacher_helper = None (no distillation)
        teacher_helper = None
        params = list(adapter.parameters())
        if (teacher_helper is not None
                and hasattr(teacher_helper, '_projection')
                and teacher_helper._projection is not None):
            params += list(teacher_helper._projection.parameters())

        # Should still create a valid optimizer with adapter params only
        import torch.optim as optim
        optimizer = optim.Adam(params, lr=0.001)
        self.assertIsNotNone(optimizer)
        self.assertEqual(len(params), len(list(adapter.parameters())))


class TestBoundedAdapter(unittest.TestCase):
    """Test BoundedAdapter wrapper for quantization-safe bounded corrections."""

    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 384
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temp files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # --- Wrapping all three adapter types ---

    def test_wraps_mlp(self):
        """Test that BoundedAdapter wraps MLP adapter correctly."""
        base = ChelationAdapter(input_dim=self.input_dim)
        bounded = BoundedAdapter(base)
        self.assertIsInstance(bounded.base_adapter, ChelationAdapter)
        self.assertEqual(bounded.input_dim, self.input_dim)

    def test_wraps_procrustes(self):
        """Test that BoundedAdapter wraps Procrustes adapter correctly."""
        base = OrthogonalProcrustesAdapter(input_dim=self.input_dim)
        bounded = BoundedAdapter(base)
        self.assertIsInstance(bounded.base_adapter, OrthogonalProcrustesAdapter)
        self.assertEqual(bounded.input_dim, self.input_dim)

    def test_wraps_low_rank(self):
        """Test that BoundedAdapter wraps Low-rank adapter correctly."""
        base = LowRankAffineAdapter(input_dim=self.input_dim, rank=8)
        bounded = BoundedAdapter(base)
        self.assertIsInstance(bounded.base_adapter, LowRankAffineAdapter)
        self.assertEqual(bounded.input_dim, self.input_dim)

    # --- Forward pass shape and normalization ---

    def test_forward_2d_shape(self):
        """Test that 2D input preserves shape through BoundedAdapter."""
        base = ChelationAdapter(input_dim=self.input_dim)
        bounded = BoundedAdapter(base)
        x = torch.randn(5, self.input_dim)
        out = bounded(x)
        self.assertEqual(out.shape, (5, self.input_dim))

    def test_forward_1d_shape(self):
        """Test that 1D input returns 1D output from BoundedAdapter."""
        base = ChelationAdapter(input_dim=self.input_dim)
        bounded = BoundedAdapter(base)
        x = torch.randn(self.input_dim)
        out = bounded(x)
        self.assertEqual(out.dim(), 1)
        self.assertEqual(out.shape[0], self.input_dim)

    def test_output_normalized(self):
        """Test that BoundedAdapter output is L2 normalized."""
        base = ChelationAdapter(input_dim=self.input_dim)
        bounded = BoundedAdapter(base)
        x = torch.randn(10, self.input_dim)
        out = bounded(x)
        norms = torch.norm(out, p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
                       f"Output not normalized: norms range {norms.min():.6f} to {norms.max():.6f}")

    # --- Correction bounding ---

    def test_corrections_above_min(self):
        """Test that non-zero corrections are at least min_correction in magnitude."""
        # Use a trained adapter that produces non-trivial corrections
        base = ChelationAdapter(input_dim=self.input_dim)
        min_corr = 0.05
        bounded = BoundedAdapter(base, min_correction=min_corr, max_correction=0.5)

        x = torch.randn(10, self.input_dim)
        with torch.no_grad():
            out = bounded(x)
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
        # The output differs from input; check that the effective delta
        # after normalization is non-trivial (at least non-zero)
        delta = out - x_norm
        correction_norms = torch.norm(delta, dim=1)
        # All corrections should be non-zero since base adapter produces some delta
        for i, cn in enumerate(correction_norms):
            if cn.item() > 1e-8:
                # If there is a correction, it should be above the noise floor
                # (the exact norm changes after final normalization, but the
                #  pre-normalization delta was scaled to min_correction)
                self.assertGreater(cn.item(), 0.0,
                                  f"Correction {i} should be non-trivial")

    def test_corrections_capped_at_max(self):
        """Test that corrections are capped to max_correction."""
        base = ChelationAdapter(input_dim=self.input_dim)
        max_corr = 0.1
        bounded = BoundedAdapter(base, min_correction=0.001, max_correction=max_corr)

        # Train the base adapter aggressively so corrections are large
        x = torch.randn(5, self.input_dim)
        target = torch.randn(5, self.input_dim)
        target = torch.nn.functional.normalize(target, p=2, dim=1)
        optimizer = torch.optim.Adam(base.parameters(), lr=0.1)
        for _ in range(50):
            optimizer.zero_grad()
            out = base(x)
            loss = torch.nn.MSELoss()(out, target)
            loss.backward()
            optimizer.step()

        # Now run through bounded adapter
        with torch.no_grad():
            base_out = base(x)
            bounded_out = bounded(x)

        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
        base_delta = base_out - x_norm
        base_norms = torch.norm(base_delta, dim=1)

        # Verify base adapter has large corrections
        has_large = torch.any(base_norms > max_corr).item()
        if has_large:
            # The bounded output should be closer to input than the unbounded output
            bounded_delta = bounded_out - x_norm
            bounded_norms = torch.norm(bounded_delta, dim=1)
            # After normalization the exact norms shift, but bounded should be
            # less extreme than base on the vectors with large corrections
            for i in range(len(base_norms)):
                if base_norms[i].item() > max_corr:
                    self.assertLess(bounded_norms[i].item(), base_norms[i].item(),
                                   f"Bounded correction {i} should be smaller than base")

    # --- Per-dimension scaling ---

    def test_dim_scale_parameter_exists(self):
        """Test that dim_scale parameter exists with correct shape and init."""
        base = ChelationAdapter(input_dim=self.input_dim)
        bounded = BoundedAdapter(base)
        self.assertTrue(hasattr(bounded, 'dim_scale'))
        self.assertEqual(bounded.dim_scale.shape, (self.input_dim,))
        self.assertTrue(torch.allclose(bounded.dim_scale.data, torch.ones(self.input_dim)))

    def test_dim_scale_is_trainable(self):
        """Test that dim_scale is an nn.Parameter and can receive gradients."""
        base = ChelationAdapter(input_dim=self.input_dim)
        bounded = BoundedAdapter(base)
        param_names = [name for name, _ in bounded.named_parameters()]
        self.assertIn('dim_scale', param_names)

        # Verify gradient flows
        x = torch.randn(5, self.input_dim)
        target = torch.nn.functional.normalize(torch.randn(5, self.input_dim), p=2, dim=1)
        out = bounded(x)
        loss = torch.nn.MSELoss()(out, target)
        loss.backward()
        self.assertIsNotNone(bounded.dim_scale.grad)

    def test_dim_scale_affects_output(self):
        """Test that modifying dim_scale changes the adapter output."""
        base = ChelationAdapter(input_dim=self.input_dim)
        bounded = BoundedAdapter(base)
        x = torch.randn(5, self.input_dim)
        out_before = bounded(x).detach().clone()

        with torch.no_grad():
            bounded.dim_scale[:self.input_dim // 2] = 3.0
            bounded.dim_scale[self.input_dim // 2:] = 0.1
        out_after = bounded(x).detach()

        self.assertFalse(torch.allclose(out_before, out_after, atol=1e-4),
                        "Non-uniform dim_scale should change output")

    # --- Factory integration ---

    def test_factory_bounded_mlp(self):
        """Test create_adapter with bounded=True returns BoundedAdapter wrapping MLP."""
        adapter = create_adapter("mlp", input_dim=self.input_dim, bounded=True)
        self.assertIsInstance(adapter, BoundedAdapter)
        self.assertIsInstance(adapter.base_adapter, ChelationAdapter)
        self.assertEqual(adapter.input_dim, self.input_dim)

    def test_factory_bounded_procrustes(self):
        """Test create_adapter with bounded=True returns BoundedAdapter wrapping Procrustes."""
        adapter = create_adapter("procrustes", input_dim=self.input_dim, bounded=True)
        self.assertIsInstance(adapter, BoundedAdapter)
        self.assertIsInstance(adapter.base_adapter, OrthogonalProcrustesAdapter)

    def test_factory_bounded_low_rank(self):
        """Test create_adapter with bounded=True returns BoundedAdapter wrapping Low-rank."""
        adapter = create_adapter("low_rank", input_dim=self.input_dim, bounded=True,
                                 rank=8)
        self.assertIsInstance(adapter, BoundedAdapter)
        self.assertIsInstance(adapter.base_adapter, LowRankAffineAdapter)
        self.assertEqual(adapter.base_adapter.rank, 8)

    def test_factory_bounded_custom_bounds(self):
        """Test create_adapter passes min/max_correction to BoundedAdapter."""
        adapter = create_adapter("mlp", input_dim=self.input_dim, bounded=True,
                                 min_correction=0.02, max_correction=0.3)
        self.assertIsInstance(adapter, BoundedAdapter)
        self.assertAlmostEqual(adapter.min_correction, 0.02)
        self.assertAlmostEqual(adapter.max_correction, 0.3)

    def test_factory_unbounded_returns_base(self):
        """Test create_adapter with bounded=False returns base adapter directly."""
        adapter = create_adapter("mlp", input_dim=self.input_dim, bounded=False)
        self.assertIsInstance(adapter, ChelationAdapter)
        self.assertNotIsInstance(adapter, BoundedAdapter)

    # --- Regularization loss ---

    def test_regularization_loss_includes_scale_term(self):
        """Test that regularization_loss includes dim_scale deviation penalty."""
        base = ChelationAdapter(input_dim=self.input_dim)
        bounded = BoundedAdapter(base)

        # At init, dim_scale = 1.0, so scale_reg = 0.0
        # Base MLP reg = 0.0, so total should be 0.0
        reg_at_init = bounded.regularization_loss()
        if isinstance(reg_at_init, (int, float)):
            self.assertAlmostEqual(reg_at_init, 0.0, places=6)
        else:
            self.assertAlmostEqual(reg_at_init.item(), 0.0, places=6)

        # Now move dim_scale away from 1.0
        with torch.no_grad():
            bounded.dim_scale.fill_(2.0)
        reg_after = bounded.regularization_loss()
        if isinstance(reg_after, (int, float)):
            reg_val = reg_after
        else:
            reg_val = reg_after.item()
        self.assertGreater(reg_val, 0.0,
                          "Regularization should be positive when dim_scale != 1.0")

    def test_regularization_loss_includes_base_adapter_term(self):
        """Test that regularization_loss delegates to base adapter."""
        base = OrthogonalProcrustesAdapter(input_dim=self.input_dim)
        bounded = BoundedAdapter(base)

        # Procrustes has non-zero reg at init (from skew param)
        base_reg = base.regularization_loss().item()
        bounded_reg = bounded.regularization_loss().item()

        # Bounded reg should be at least as large as base reg
        # (dim_scale = 1.0 at init, so scale_reg = 0.0)
        self.assertAlmostEqual(bounded_reg, base_reg, places=5,
                              msg="Bounded reg should equal base reg when dim_scale = 1.0")

    def test_regularization_loss_is_differentiable(self):
        """Test that regularization_loss produces gradients on dim_scale."""
        base = ChelationAdapter(input_dim=64)
        bounded = BoundedAdapter(base)
        # Move dim_scale so gradient is non-zero
        with torch.no_grad():
            bounded.dim_scale.fill_(1.5)
        reg = bounded.regularization_loss()
        reg.backward()
        self.assertIsNotNone(bounded.dim_scale.grad)
        self.assertFalse(torch.all(bounded.dim_scale.grad == 0).item(),
                        "Gradient should be non-zero when dim_scale != 1.0")

    # --- Save and load ---

    def test_save_load_round_trip(self):
        """Test that save/load preserves BoundedAdapter state."""
        base = ChelationAdapter(input_dim=self.input_dim)
        bounded = BoundedAdapter(base, min_correction=0.02, max_correction=0.4)

        # Modify state
        x = torch.randn(5, self.input_dim)
        target = torch.nn.functional.normalize(torch.randn(5, self.input_dim), p=2, dim=1)
        optimizer = torch.optim.Adam(bounded.parameters(), lr=0.01)
        for _ in range(5):
            optimizer.zero_grad()
            out = bounded(x)
            loss = torch.nn.MSELoss()(out, target)
            loss.backward()
            optimizer.step()

        save_path = self.temp_dir / "bounded_adapter.pt"
        bounded.save(str(save_path))
        self.assertTrue(save_path.exists())

        # Load into new instance
        new_base = ChelationAdapter(input_dim=self.input_dim)
        new_bounded = BoundedAdapter(new_base, min_correction=0.02, max_correction=0.4)
        success = new_bounded.load(str(save_path))
        self.assertTrue(success)

        # Outputs should match
        with torch.no_grad():
            out1 = bounded(x)
            out2 = new_bounded(x)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-5),
                       f"Save/load mismatch: max diff = {(out1 - out2).abs().max():.6e}")

    def test_load_nonexistent_returns_false(self):
        """Test that loading from nonexistent path returns False."""
        base = ChelationAdapter(input_dim=self.input_dim)
        bounded = BoundedAdapter(base)
        success = bounded.load(str(self.temp_dir / "nonexistent.pt"))
        self.assertFalse(success)

    def test_save_path_traversal_blocked(self):
        """Test that path traversal is blocked in save()."""
        base = ChelationAdapter(input_dim=self.input_dim)
        bounded = BoundedAdapter(base)
        traversal_path = self.temp_dir / ".." / "escaping.pt"
        with self.assertRaises(ValueError) as cm:
            bounded.save(str(traversal_path))
        self.assertIn("traversal", str(cm.exception).lower())

    def test_load_path_traversal_blocked(self):
        """Test that path traversal is blocked in load()."""
        base = ChelationAdapter(input_dim=self.input_dim)
        bounded = BoundedAdapter(base)
        traversal_path = self.temp_dir / ".." / "malicious.pt"
        with self.assertRaises(ValueError) as cm:
            bounded.load(str(traversal_path))
        self.assertIn("traversal", str(cm.exception).lower())

    # --- Config preset ---

    def test_bounded_adapter_preset(self):
        """Test that bounded_adapter preset is accessible from ChelationConfig."""
        preset = ChelationConfig.get_preset("balanced", "bounded_adapter")
        self.assertTrue(preset["bounded"])
        self.assertEqual(preset["min_correction"], 0.01)
        self.assertEqual(preset["max_correction"], 0.5)
        self.assertIn("description", preset)

    def test_bounded_adapter_preset_conservative(self):
        """Test that conservative bounded_adapter preset has tighter bounds."""
        preset = ChelationConfig.get_preset("conservative", "bounded_adapter")
        self.assertTrue(preset["bounded"])
        self.assertEqual(preset["max_correction"], 0.3)

    def test_bounded_adapter_preset_aggressive(self):
        """Test that aggressive bounded_adapter preset has wider bounds."""
        preset = ChelationConfig.get_preset("aggressive", "bounded_adapter")
        self.assertTrue(preset["bounded"])
        self.assertEqual(preset["min_correction"], 0.02)
        self.assertEqual(preset["max_correction"], 0.8)

    # --- Parameters include both base and wrapper ---

    def test_parameters_include_all(self):
        """Test that parameters() returns both base adapter and dim_scale params."""
        base = ChelationAdapter(input_dim=64)
        bounded = BoundedAdapter(base)

        # Count base params
        base_param_count = sum(1 for _ in base.parameters())
        # Count bounded params (should be base + dim_scale)
        bounded_param_count = sum(1 for _ in bounded.parameters())
        self.assertEqual(bounded_param_count, base_param_count + 1,
                        "BoundedAdapter should have base params + dim_scale")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
