"""Tests for sedimentation loss functions.

Validates InfoNCE, hybrid, and factory for sedimentation training losses.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock
from collections import defaultdict

import torch
import torch.nn as nn

from sedimentation_loss import (
    SedimentationInfoNCELoss,
    SedimentationHybridLoss,
    HardNegativeMiner,
    create_sedimentation_loss,
)


class TestSedimentationInfoNCELoss(unittest.TestCase):
    """Tests for SedimentationInfoNCELoss."""

    def test_forward_returns_scalar(self):
        """Forward pass returns a scalar tensor."""
        loss_fn = SedimentationInfoNCELoss(temperature=0.07)
        outputs = torch.randn(8, 64)
        targets = torch.randn(8, 64)
        loss = loss_fn(outputs, targets)
        self.assertEqual(loss.dim(), 0, "Loss should be scalar (0-dim tensor)")

    def test_forward_loss_is_finite(self):
        """Loss value should be finite and non-negative."""
        loss_fn = SedimentationInfoNCELoss(temperature=0.07)
        outputs = torch.randn(16, 128)
        targets = torch.randn(16, 128)
        loss = loss_fn(outputs, targets)
        self.assertTrue(torch.isfinite(loss).item(), "Loss should be finite")
        self.assertGreaterEqual(loss.item(), 0.0, "Loss should be non-negative")

    def test_matched_pairs_lower_loss_than_random(self):
        """When outputs match targets on diagonal, loss should be lower."""
        loss_fn = SedimentationInfoNCELoss(temperature=0.07)
        batch_size = 16
        dim = 64

        # Create targets
        targets = torch.randn(batch_size, dim)

        # Matched: outputs are very close to their targets (small noise)
        matched_outputs = targets + torch.randn_like(targets) * 0.01
        loss_matched = loss_fn(matched_outputs, targets).item()

        # Random: outputs are unrelated to targets
        random_outputs = torch.randn(batch_size, dim)
        loss_random = loss_fn(random_outputs, targets).item()

        self.assertLess(
            loss_matched,
            loss_random,
            "Matched outputs should produce lower loss than random outputs",
        )

    def test_perfect_alignment_gives_near_zero_loss(self):
        """When outputs exactly equal targets, loss should approach zero."""
        loss_fn = SedimentationInfoNCELoss(temperature=0.07)
        targets = torch.randn(8, 32)
        # Identical outputs
        loss = loss_fn(targets.clone(), targets).item()
        # With temperature=0.07, perfect alignment on 8 samples gives very low loss
        self.assertLess(loss, 0.5, "Perfect alignment should give very low loss")

    def test_gradient_flows(self):
        """Loss should produce gradients for backprop."""
        loss_fn = SedimentationInfoNCELoss(temperature=0.07)
        outputs = torch.randn(8, 64, requires_grad=True)
        targets = torch.randn(8, 64)
        loss = loss_fn(outputs, targets)
        loss.backward()
        self.assertIsNotNone(outputs.grad, "Gradients should flow to outputs")
        self.assertFalse(
            torch.all(outputs.grad == 0).item(),
            "Gradients should be non-zero",
        )

    def test_temperature_affects_loss(self):
        """Different temperatures should produce different loss values."""
        outputs = torch.randn(8, 64)
        targets = torch.randn(8, 64)

        loss_cold = SedimentationInfoNCELoss(temperature=0.01)(outputs, targets).item()
        loss_warm = SedimentationInfoNCELoss(temperature=1.0)(outputs, targets).item()

        self.assertNotAlmostEqual(
            loss_cold,
            loss_warm,
            places=2,
            msg="Different temperatures should produce different losses",
        )

    def test_invalid_temperature_raises(self):
        """Non-positive temperature should raise ValueError."""
        with self.assertRaises(ValueError):
            SedimentationInfoNCELoss(temperature=0.0)
        with self.assertRaises(ValueError):
            SedimentationInfoNCELoss(temperature=-0.1)

    def test_batch_size_one_works(self):
        """Loss should work with batch_size=1 (degenerate case)."""
        loss_fn = SedimentationInfoNCELoss(temperature=0.07)
        outputs = torch.randn(1, 64)
        targets = torch.randn(1, 64)
        loss = loss_fn(outputs, targets)
        self.assertTrue(torch.isfinite(loss).item())


class TestSedimentationHybridLoss(unittest.TestCase):
    """Tests for SedimentationHybridLoss."""

    def test_forward_returns_scalar(self):
        """Hybrid loss returns a scalar."""
        loss_fn = SedimentationHybridLoss()
        outputs = torch.randn(8, 64)
        targets = torch.randn(8, 64)
        loss = loss_fn(outputs, targets)
        self.assertEqual(loss.dim(), 0)

    def test_combines_both_terms(self):
        """Hybrid loss is between pure MSE and pure InfoNCE."""
        outputs = torch.randn(16, 64)
        targets = torch.randn(16, 64)

        mse_only = nn.MSELoss()(outputs, targets).item()
        infonce_only = SedimentationInfoNCELoss(temperature=0.07)(
            outputs, targets
        ).item()
        hybrid = SedimentationHybridLoss(
            temperature=0.07, contrastive_weight=0.5, mse_weight=0.5
        )(outputs, targets).item()

        expected = 0.5 * mse_only + 0.5 * infonce_only
        self.assertAlmostEqual(
            hybrid,
            expected,
            places=4,
            msg="Hybrid should be weighted sum of MSE and InfoNCE",
        )

    def test_weights_affect_output(self):
        """Changing weights changes the loss value."""
        outputs = torch.randn(8, 64)
        targets = torch.randn(8, 64)

        loss_mse_heavy = SedimentationHybridLoss(
            contrastive_weight=0.1, mse_weight=0.9
        )(outputs, targets).item()
        loss_infonce_heavy = SedimentationHybridLoss(
            contrastive_weight=0.9, mse_weight=0.1
        )(outputs, targets).item()

        self.assertNotAlmostEqual(
            loss_mse_heavy,
            loss_infonce_heavy,
            places=2,
            msg="Different weights should produce different losses",
        )

    def test_gradient_flows(self):
        """Hybrid loss gradients flow correctly."""
        loss_fn = SedimentationHybridLoss()
        outputs = torch.randn(8, 64, requires_grad=True)
        targets = torch.randn(8, 64)
        loss = loss_fn(outputs, targets)
        loss.backward()
        self.assertIsNotNone(outputs.grad)


class TestCreateSedimentationLoss(unittest.TestCase):
    """Tests for create_sedimentation_loss factory."""

    def test_mse_returns_mse_loss(self):
        """Factory returns MSELoss for 'mse' type."""
        loss = create_sedimentation_loss("mse")
        self.assertIsInstance(loss, nn.MSELoss)

    def test_infonce_returns_infonce_loss(self):
        """Factory returns SedimentationInfoNCELoss for 'infonce' type."""
        loss = create_sedimentation_loss("infonce")
        self.assertIsInstance(loss, SedimentationInfoNCELoss)

    def test_infonce_with_temperature(self):
        """Factory passes temperature kwarg to InfoNCE."""
        loss = create_sedimentation_loss("infonce", temperature=0.1)
        self.assertIsInstance(loss, SedimentationInfoNCELoss)
        self.assertAlmostEqual(loss.temperature, 0.1)

    def test_hybrid_returns_hybrid_loss(self):
        """Factory returns SedimentationHybridLoss for 'hybrid' type."""
        loss = create_sedimentation_loss("hybrid")
        self.assertIsInstance(loss, SedimentationHybridLoss)

    def test_hybrid_with_params(self):
        """Factory passes kwargs to HybridLoss."""
        loss = create_sedimentation_loss(
            "hybrid",
            temperature=0.1,
            contrastive_weight=0.3,
            mse_weight=0.7,
        )
        self.assertIsInstance(loss, SedimentationHybridLoss)
        self.assertAlmostEqual(loss.contrastive_weight, 0.3)
        self.assertAlmostEqual(loss.mse_weight, 0.7)

    def test_unknown_type_raises(self):
        """Unknown loss type raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            create_sedimentation_loss("unknown")
        self.assertIn("unknown", str(ctx.exception))

    def test_default_is_mse(self):
        """Default (no args) returns MSE."""
        loss = create_sedimentation_loss()
        self.assertIsInstance(loss, nn.MSELoss)


class TestHardNegativeMiner(unittest.TestCase):
    """Tests for HardNegativeMiner."""

    def test_empty_log_returns_empty_lists(self):
        """Empty chelation_log returns empty negative lists."""
        miner = HardNegativeMiner(chelation_log={}, max_negatives=16)
        result = miner.get_hard_negative_indices([0, 1, 2], total_size=10)
        self.assertEqual(len(result), 3)
        for neg_list in result:
            self.assertEqual(len(neg_list), 0)

    def test_collision_map_extraction(self):
        """Miner extracts collision partners from dict entries."""
        log = defaultdict(list)
        log["doc_0"] = [
            {"doc_id": 0, "collisions": [3, 5, 7]},
        ]
        log["doc_1"] = [
            {"doc_id": 1, "collisions": [4, 6]},
        ]
        miner = HardNegativeMiner(chelation_log=log, max_negatives=16)
        result = miner.get_hard_negative_indices([0, 1, 2], total_size=10)
        self.assertEqual(result[0], [3, 5, 7])
        self.assertEqual(result[1], [4, 6])
        self.assertEqual(result[2], [])

    def test_max_negatives_limit(self):
        """Miner respects max_negatives limit."""
        log = defaultdict(list)
        log["doc_0"] = [
            {"doc_id": 0, "collisions": list(range(1, 100))},
        ]
        miner = HardNegativeMiner(chelation_log=log, max_negatives=5)
        result = miner.get_hard_negative_indices([0], total_size=100)
        self.assertLessEqual(len(result[0]), 5)

    def test_filters_out_of_range_indices(self):
        """Miner filters indices >= total_size."""
        log = defaultdict(list)
        log["doc_0"] = [
            {"doc_id": 0, "collisions": [1, 2, 99]},
        ]
        miner = HardNegativeMiner(chelation_log=log, max_negatives=16)
        result = miner.get_hard_negative_indices([0], total_size=10)
        self.assertNotIn(99, result[0])
        self.assertIn(1, result[0])
        self.assertIn(2, result[0])

    def test_filters_self_reference(self):
        """Miner excludes self-index from negatives."""
        log = defaultdict(list)
        log["doc_0"] = [
            {"doc_id": 0, "collisions": [0, 1, 2]},
        ]
        miner = HardNegativeMiner(chelation_log=log, max_negatives=16)
        result = miner.get_hard_negative_indices([0], total_size=10)
        self.assertNotIn(0, result[0])

    def test_non_dict_entries_ignored(self):
        """Non-dict entries in chelation_log are safely ignored."""
        log = defaultdict(list)
        # The actual chelation_log stores numpy arrays (center_of_mass), not dicts
        import numpy as np
        log[0] = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        miner = HardNegativeMiner(chelation_log=log, max_negatives=16)
        result = miner.get_hard_negative_indices([0, 1], total_size=10)
        # Should gracefully return empty lists
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 0)


class TestEngineIntegration(unittest.TestCase):
    """Tests for set_sedimentation_loss integration with AntigravityEngine."""

    @patch("antigravity_engine.create_embedding_backend")
    @patch("antigravity_engine.create_vector_store")
    @patch("antigravity_engine.get_logger")
    def test_set_sedimentation_loss_stores_config(
        self, mock_logger_fn, mock_store, mock_backend
    ):
        """set_sedimentation_loss stores type and kwargs on engine."""
        mock_logger_fn.return_value = MagicMock()
        mock_store.return_value = MagicMock()
        mock_backend.return_value = MagicMock()

        from antigravity_engine import AntigravityEngine

        engine = AntigravityEngine.__new__(AntigravityEngine)
        engine.logger = MagicMock()

        engine.set_sedimentation_loss("infonce", temperature=0.1)
        self.assertEqual(engine._sedimentation_loss_type, "infonce")
        self.assertEqual(engine._sedimentation_loss_kwargs["temperature"], 0.1)

    @patch("antigravity_engine.create_embedding_backend")
    @patch("antigravity_engine.create_vector_store")
    @patch("antigravity_engine.get_logger")
    def test_set_sedimentation_loss_invalid_type(
        self, mock_logger_fn, mock_store, mock_backend
    ):
        """set_sedimentation_loss rejects invalid loss types."""
        mock_logger_fn.return_value = MagicMock()

        from antigravity_engine import AntigravityEngine

        engine = AntigravityEngine.__new__(AntigravityEngine)
        engine.logger = MagicMock()

        with self.assertRaises(ValueError):
            engine.set_sedimentation_loss("bogus")

    @patch("antigravity_engine.create_embedding_backend")
    @patch("antigravity_engine.create_vector_store")
    @patch("antigravity_engine.get_logger")
    def test_set_sedimentation_loss_hybrid(
        self, mock_logger_fn, mock_store, mock_backend
    ):
        """set_sedimentation_loss stores hybrid config."""
        mock_logger_fn.return_value = MagicMock()

        from antigravity_engine import AntigravityEngine

        engine = AntigravityEngine.__new__(AntigravityEngine)
        engine.logger = MagicMock()

        engine.set_sedimentation_loss(
            "hybrid", contrastive_weight=0.3, mse_weight=0.7, temperature=0.1
        )
        self.assertEqual(engine._sedimentation_loss_type, "hybrid")
        self.assertAlmostEqual(
            engine._sedimentation_loss_kwargs["contrastive_weight"], 0.3
        )

    @patch("antigravity_engine.create_embedding_backend")
    @patch("antigravity_engine.create_vector_store")
    @patch("antigravity_engine.get_logger")
    def test_default_loss_is_mse_when_not_configured(
        self, mock_logger_fn, mock_store, mock_backend
    ):
        """Without set_sedimentation_loss, getattr falls back to 'mse'."""
        mock_logger_fn.return_value = MagicMock()

        from antigravity_engine import AntigravityEngine

        engine = AntigravityEngine.__new__(AntigravityEngine)
        engine.logger = MagicMock()

        # No call to set_sedimentation_loss
        loss_type = getattr(engine, "_sedimentation_loss_type", "mse")
        self.assertEqual(loss_type, "mse")

    def test_set_sedimentation_loss_logs_event(self):
        """set_sedimentation_loss logs the configuration change."""
        from antigravity_engine import AntigravityEngine

        engine = AntigravityEngine.__new__(AntigravityEngine)
        engine.logger = MagicMock()

        engine.set_sedimentation_loss("infonce", temperature=0.05)
        engine.logger.log_event.assert_called_once()
        call_args = engine.logger.log_event.call_args
        self.assertEqual(call_args[0][0], "sedimentation_loss_set")


class TestConfigPresets(unittest.TestCase):
    """Tests for sedimentation_loss config presets."""

    def test_mse_preset(self):
        """MSE preset returns correct config."""
        from config import ChelationConfig

        preset = ChelationConfig.get_preset("mse", "sedimentation_loss")
        self.assertEqual(preset["loss_type"], "mse")

    def test_contrastive_preset(self):
        """Contrastive preset returns InfoNCE config."""
        from config import ChelationConfig

        preset = ChelationConfig.get_preset("contrastive", "sedimentation_loss")
        self.assertEqual(preset["loss_type"], "infonce")
        self.assertAlmostEqual(preset["temperature"], 0.07)

    def test_hybrid_preset(self):
        """Hybrid preset returns hybrid config with weights."""
        from config import ChelationConfig

        preset = ChelationConfig.get_preset("hybrid", "sedimentation_loss")
        self.assertEqual(preset["loss_type"], "hybrid")
        self.assertAlmostEqual(preset["contrastive_weight"], 0.5)
        self.assertAlmostEqual(preset["mse_weight"], 0.5)
        self.assertAlmostEqual(preset["temperature"], 0.07)

    def test_invalid_preset_name_raises(self):
        """Invalid preset name raises ValueError."""
        from config import ChelationConfig

        with self.assertRaises(ValueError):
            ChelationConfig.get_preset("nonexistent", "sedimentation_loss")


if __name__ == "__main__":
    unittest.main()
