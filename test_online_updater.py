"""
Tests for OnlineUpdater (Phase 3: Online Gradient Updates)

Run: python -m pytest test_online_updater.py -v
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
import torch.nn as nn

from online_updater import OnlineUpdater
from chelation_adapter import ChelationAdapter


@patch('online_updater.get_logger')
class TestOnlineUpdater(unittest.TestCase):
    """Tests for the OnlineUpdater class."""

    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 128
        self.adapter = ChelationAdapter(input_dim=self.input_dim)
        self.query_vec = np.random.randn(self.input_dim).astype(np.float32)
        self.top_k = np.random.randn(5, self.input_dim).astype(np.float32)
        self.bottom_k = np.random.randn(5, self.input_dim).astype(np.float32)

    def test_initialization_defaults(self, mock_logger):
        """Test default initialization parameters."""
        updater = OnlineUpdater(self.adapter)
        self.assertEqual(updater.learning_rate, 0.0001)
        self.assertEqual(updater.micro_steps, 1)
        self.assertEqual(updater.momentum, 0.9)
        self.assertEqual(updater.max_grad_norm, 1.0)
        self.assertEqual(updater.update_interval, 1)
        self.assertEqual(updater.margin, 0.1)

    def test_initialization_custom(self, mock_logger):
        """Test custom initialization parameters."""
        updater = OnlineUpdater(
            self.adapter, learning_rate=0.001, micro_steps=3,
            momentum=0.5, max_grad_norm=2.0, update_interval=5, margin=0.2
        )
        self.assertEqual(updater.learning_rate, 0.001)
        self.assertEqual(updater.micro_steps, 3)
        self.assertEqual(updater.update_interval, 5)

    def test_initialization_invalid_adapter(self, mock_logger):
        """Test that non-nn.Module adapter raises TypeError."""
        with self.assertRaises(TypeError):
            OnlineUpdater("not_a_module")

    def test_initialization_invalid_learning_rate(self, mock_logger):
        """Test that non-positive learning rate raises ValueError."""
        with self.assertRaises(ValueError):
            OnlineUpdater(self.adapter, learning_rate=0)
        with self.assertRaises(ValueError):
            OnlineUpdater(self.adapter, learning_rate=-0.001)

    def test_initialization_invalid_micro_steps(self, mock_logger):
        """Test that micro_steps < 1 raises ValueError."""
        with self.assertRaises(ValueError):
            OnlineUpdater(self.adapter, micro_steps=0)

    def test_initialization_invalid_max_grad_norm(self, mock_logger):
        """Test that non-positive max_grad_norm raises ValueError."""
        with self.assertRaises(ValueError):
            OnlineUpdater(self.adapter, max_grad_norm=0)

    def test_initialization_invalid_update_interval(self, mock_logger):
        """Test that update_interval < 1 raises ValueError."""
        with self.assertRaises(ValueError):
            OnlineUpdater(self.adapter, update_interval=0)

    def test_update_performs_gradient_step(self, mock_logger):
        """Test that update modifies adapter parameters."""
        updater = OnlineUpdater(self.adapter, learning_rate=0.01)

        # Snapshot parameters before
        params_before = [p.clone() for p in self.adapter.parameters()]

        result = updater.update(self.query_vec, self.top_k, self.bottom_k)

        self.assertTrue(result["updated"])
        self.assertIsNotNone(result["loss"])

        # At least some parameters should have changed
        any_changed = False
        for p_before, p_after in zip(params_before, self.adapter.parameters()):
            if not torch.allclose(p_before, p_after.data):
                any_changed = True
                break
        self.assertTrue(any_changed, "Parameters should change after update")

    def test_update_returns_loss(self, mock_logger):
        """Test that update returns a finite loss value."""
        updater = OnlineUpdater(self.adapter)
        result = updater.update(self.query_vec, self.top_k, self.bottom_k)
        self.assertTrue(result["updated"])
        self.assertIsInstance(result["loss"], float)
        self.assertTrue(np.isfinite(result["loss"]))

    def test_update_empty_positives_skips(self, mock_logger):
        """Test that empty top_k skips update."""
        updater = OnlineUpdater(self.adapter)
        result = updater.update(self.query_vec, np.array([]).reshape(0, self.input_dim), self.bottom_k)
        self.assertFalse(result["updated"])
        self.assertIsNone(result["loss"])

    def test_update_empty_negatives_skips(self, mock_logger):
        """Test that empty bottom_k skips update."""
        updater = OnlineUpdater(self.adapter)
        result = updater.update(self.query_vec, self.top_k, np.array([]).reshape(0, self.input_dim))
        self.assertFalse(result["updated"])
        self.assertIsNone(result["loss"])

    def test_update_interval_respected(self, mock_logger):
        """Test that updates only happen at the specified interval."""
        updater = OnlineUpdater(self.adapter, update_interval=3)

        r1 = updater.update(self.query_vec, self.top_k, self.bottom_k)
        self.assertFalse(r1["updated"])  # query 1

        r2 = updater.update(self.query_vec, self.top_k, self.bottom_k)
        self.assertFalse(r2["updated"])  # query 2

        r3 = updater.update(self.query_vec, self.top_k, self.bottom_k)
        self.assertTrue(r3["updated"])  # query 3 (interval hit)

    def test_micro_steps_multiple(self, mock_logger):
        """Test that multiple micro-steps are performed."""
        updater = OnlineUpdater(self.adapter, micro_steps=3, learning_rate=0.01)
        result = updater.update(self.query_vec, self.top_k, self.bottom_k)
        self.assertTrue(result["updated"])
        # Loss should be the average over micro_steps
        self.assertIsInstance(result["loss"], float)

    def test_query_count_tracking(self, mock_logger):
        """Test that query count increments correctly."""
        updater = OnlineUpdater(self.adapter)
        self.assertEqual(updater.query_count, 0)
        updater.update(self.query_vec, self.top_k, self.bottom_k)
        self.assertEqual(updater.query_count, 1)
        updater.update(self.query_vec, self.top_k, self.bottom_k)
        self.assertEqual(updater.query_count, 2)

    def test_update_count_tracking(self, mock_logger):
        """Test that update count increments correctly."""
        updater = OnlineUpdater(self.adapter, update_interval=2)
        self.assertEqual(updater.update_count, 0)
        updater.update(self.query_vec, self.top_k, self.bottom_k)
        self.assertEqual(updater.update_count, 0)  # Not at interval
        updater.update(self.query_vec, self.top_k, self.bottom_k)
        self.assertEqual(updater.update_count, 1)  # At interval

    def test_average_loss(self, mock_logger):
        """Test average loss computation."""
        updater = OnlineUpdater(self.adapter)
        self.assertEqual(updater.average_loss, 0.0)
        updater.update(self.query_vec, self.top_k, self.bottom_k)
        self.assertGreater(updater.average_loss, 0.0)

    def test_get_stats(self, mock_logger):
        """Test get_stats returns expected keys."""
        updater = OnlineUpdater(self.adapter)
        stats = updater.get_stats()
        self.assertIn("query_count", stats)
        self.assertIn("update_count", stats)
        self.assertIn("average_loss", stats)
        self.assertIn("learning_rate", stats)
        self.assertIn("micro_steps", stats)
        self.assertIn("update_interval", stats)

    def test_reset_stats(self, mock_logger):
        """Test that reset_stats clears counters."""
        updater = OnlineUpdater(self.adapter)
        updater.update(self.query_vec, self.top_k, self.bottom_k)
        self.assertGreater(updater.query_count, 0)
        updater.reset_stats()
        self.assertEqual(updater.query_count, 0)
        self.assertEqual(updater.update_count, 0)
        self.assertEqual(updater.average_loss, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
