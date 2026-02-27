"""
Tests for Online Correction Refinements (Session 22)

Pluggable loss functions, adaptive margins, loss scheduling, and diagnostics
for the OnlineUpdater.

Run: python test_online_correction.py
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
import torch.nn as nn

from online_updater import (
    TripletMarginOnlineLoss,
    InfoNCEOnlineLoss,
    CosineSimilarityOnlineLoss,
    create_online_loss,
    AdaptiveMargin,
    OnlineLossScheduler,
    OnlineUpdateDiagnostics,
    OnlineUpdater,
)
from chelation_adapter import ChelationAdapter
from config import ChelationConfig


# ============================================================
# Loss Function Tests (15 tests)
# ============================================================

class TestTripletMarginOnlineLoss(unittest.TestCase):
    """Tests for TripletMarginOnlineLoss."""

    def setUp(self):
        self.dim = 64
        self.query = torch.randn(1, self.dim)
        self.pos = torch.randn(3, self.dim)
        self.neg = torch.randn(3, self.dim)

    def test_compute_mean_aggregation(self):
        """Test triplet loss with mean aggregation (default)."""
        loss_fn = TripletMarginOnlineLoss(margin=0.1, aggregation="mean")
        loss = loss_fn.compute(self.query, self.pos, self.neg)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # scalar
        self.assertTrue(torch.isfinite(loss))

    def test_compute_per_vector_aggregation(self):
        """Test triplet loss with per_vector aggregation."""
        loss_fn = TripletMarginOnlineLoss(margin=0.1, aggregation="per_vector")
        loss = loss_fn.compute(self.query, self.pos, self.neg)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.isfinite(loss))

    def test_mean_aggregation_matches_original(self):
        """Verify mean aggregation reproduces original OnlineUpdater behavior."""
        loss_fn = TripletMarginOnlineLoss(margin=0.1, aggregation="mean")
        ref_loss = nn.TripletMarginLoss(margin=0.1)

        pos_mean = self.pos.mean(dim=0, keepdim=True)
        neg_mean = self.neg.mean(dim=0, keepdim=True)

        expected = ref_loss(self.query, pos_mean, neg_mean)
        actual = loss_fn.compute(self.query, self.pos, self.neg)

        self.assertAlmostEqual(expected.item(), actual.item(), places=5)

    def test_invalid_margin(self):
        """Test that negative margin raises ValueError."""
        with self.assertRaises(ValueError):
            TripletMarginOnlineLoss(margin=-0.1)

    def test_invalid_aggregation(self):
        """Test that invalid aggregation raises ValueError."""
        with self.assertRaises(ValueError):
            TripletMarginOnlineLoss(aggregation="invalid")

    def test_get_state(self):
        """Test get_state returns expected keys."""
        loss_fn = TripletMarginOnlineLoss(margin=0.2, aggregation="per_vector")
        loss_fn.compute(self.query, self.pos, self.neg)
        state = loss_fn.get_state()
        self.assertEqual(state["loss_type"], "triplet_margin")
        self.assertEqual(state["margin"], 0.2)
        self.assertEqual(state["aggregation"], "per_vector")
        self.assertEqual(state["call_count"], 1)

    def test_call_count_increments(self):
        """Test that call_count tracks compute calls."""
        loss_fn = TripletMarginOnlineLoss()
        self.assertEqual(loss_fn.get_state()["call_count"], 0)
        loss_fn.compute(self.query, self.pos, self.neg)
        loss_fn.compute(self.query, self.pos, self.neg)
        self.assertEqual(loss_fn.get_state()["call_count"], 2)


class TestInfoNCEOnlineLoss(unittest.TestCase):
    """Tests for InfoNCEOnlineLoss."""

    def setUp(self):
        self.dim = 64
        self.query = torch.randn(1, self.dim)
        self.pos = torch.randn(3, self.dim)
        self.neg = torch.randn(5, self.dim)

    def test_compute_returns_scalar(self):
        """Test InfoNCE loss computes a finite scalar."""
        loss_fn = InfoNCEOnlineLoss(temperature=0.07)
        loss = loss_fn.compute(self.query, self.pos, self.neg)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_temperature_affects_loss(self):
        """Test that different temperatures produce different losses."""
        loss_low_t = InfoNCEOnlineLoss(temperature=0.01)
        loss_high_t = InfoNCEOnlineLoss(temperature=1.0)

        val_low = loss_low_t.compute(self.query, self.pos, self.neg).item()
        val_high = loss_high_t.compute(self.query, self.pos, self.neg).item()

        # Lower temperature should produce higher loss (sharper distribution)
        self.assertNotAlmostEqual(val_low, val_high, places=2)

    def test_invalid_temperature(self):
        """Test that non-positive temperature raises ValueError."""
        with self.assertRaises(ValueError):
            InfoNCEOnlineLoss(temperature=0)
        with self.assertRaises(ValueError):
            InfoNCEOnlineLoss(temperature=-0.1)

    def test_get_state(self):
        """Test get_state returns expected structure."""
        loss_fn = InfoNCEOnlineLoss(temperature=0.05)
        state = loss_fn.get_state()
        self.assertEqual(state["loss_type"], "infonce")
        self.assertEqual(state["temperature"], 0.05)
        self.assertEqual(state["call_count"], 0)

    def test_gradient_flows(self):
        """Test that gradients flow through InfoNCE loss."""
        query = self.query.clone().requires_grad_(True)
        loss_fn = InfoNCEOnlineLoss(temperature=0.07)
        loss = loss_fn.compute(query, self.pos, self.neg)
        loss.backward()
        self.assertIsNotNone(query.grad)
        self.assertTrue(torch.any(query.grad != 0))


class TestCosineSimilarityOnlineLoss(unittest.TestCase):
    """Tests for CosineSimilarityOnlineLoss."""

    def setUp(self):
        self.dim = 64
        self.query = torch.randn(1, self.dim)
        self.pos = torch.randn(3, self.dim)
        self.neg = torch.randn(3, self.dim)

    def test_compute_returns_scalar(self):
        """Test cosine similarity loss computes a finite scalar."""
        loss_fn = CosineSimilarityOnlineLoss()
        loss = loss_fn.compute(self.query, self.pos, self.neg)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_weights_affect_loss(self):
        """Test that pos_weight and neg_weight change the loss value."""
        loss_equal = CosineSimilarityOnlineLoss(pos_weight=1.0, neg_weight=1.0)
        loss_pos_heavy = CosineSimilarityOnlineLoss(pos_weight=2.0, neg_weight=0.5)

        val_eq = loss_equal.compute(self.query, self.pos, self.neg).item()
        val_ph = loss_pos_heavy.compute(self.query, self.pos, self.neg).item()

        self.assertNotAlmostEqual(val_eq, val_ph, places=3)

    def test_invalid_weights(self):
        """Test that negative weights raise ValueError."""
        with self.assertRaises(ValueError):
            CosineSimilarityOnlineLoss(pos_weight=-1.0)
        with self.assertRaises(ValueError):
            CosineSimilarityOnlineLoss(neg_weight=-0.5)

    def test_get_state(self):
        """Test get_state returns expected keys."""
        loss_fn = CosineSimilarityOnlineLoss(pos_weight=1.5, neg_weight=0.8)
        state = loss_fn.get_state()
        self.assertEqual(state["loss_type"], "cosine_similarity")
        self.assertEqual(state["pos_weight"], 1.5)
        self.assertEqual(state["neg_weight"], 0.8)


class TestCreateOnlineLoss(unittest.TestCase):
    """Tests for the create_online_loss factory."""

    def test_create_triplet_margin(self):
        """Test factory creates TripletMarginOnlineLoss."""
        loss = create_online_loss("triplet_margin", margin=0.2)
        self.assertIsInstance(loss, TripletMarginOnlineLoss)
        self.assertEqual(loss.margin, 0.2)

    def test_create_infonce(self):
        """Test factory creates InfoNCEOnlineLoss."""
        loss = create_online_loss("infonce", temperature=0.1)
        self.assertIsInstance(loss, InfoNCEOnlineLoss)
        self.assertEqual(loss.temperature, 0.1)

    def test_create_cosine_similarity(self):
        """Test factory creates CosineSimilarityOnlineLoss."""
        loss = create_online_loss("cosine_similarity", pos_weight=2.0)
        self.assertIsInstance(loss, CosineSimilarityOnlineLoss)
        self.assertEqual(loss.pos_weight, 2.0)

    def test_unknown_loss_type_raises(self):
        """Test that unknown loss type raises ValueError."""
        with self.assertRaises(ValueError):
            create_online_loss("unknown_loss")

    def test_default_is_triplet_margin(self):
        """Test that default loss type is triplet_margin."""
        loss = create_online_loss()
        self.assertIsInstance(loss, TripletMarginOnlineLoss)


# ============================================================
# Adaptive Margin Tests (8 tests)
# ============================================================

class TestAdaptiveMargin(unittest.TestCase):
    """Tests for AdaptiveMargin."""

    def test_initial_margin(self):
        """Test initial margin equals base_margin."""
        am = AdaptiveMargin(base_margin=0.15)
        self.assertEqual(am.current_margin, 0.15)

    def test_update_with_large_gap(self):
        """Test that large quality gap shrinks margin."""
        am = AdaptiveMargin(base_margin=0.2, adaptation_rate=0.5)
        # High pos scores, low neg scores = large gap
        margin = am.update([0.9, 0.85, 0.8], [0.1, 0.15, 0.2])
        # With large gap, target is base/(1+gap) < base
        self.assertLess(margin, 0.2)

    def test_update_with_small_gap(self):
        """Test that small quality gap keeps margin closer to base."""
        am = AdaptiveMargin(base_margin=0.2, adaptation_rate=0.5)
        # Similar scores = small gap
        margin = am.update([0.5, 0.51], [0.49, 0.48])
        # Small gap, target close to base_margin / (1 + small_gap)
        self.assertIsInstance(margin, float)
        self.assertTrue(am.min_margin <= margin <= am.max_margin)

    def test_update_with_negative_gap(self):
        """Test that negative gap pushes toward max_margin."""
        am = AdaptiveMargin(base_margin=0.1, max_margin=0.5, adaptation_rate=1.0)
        # Negatives closer than positives
        margin = am.update([0.1, 0.2], [0.8, 0.9])
        self.assertEqual(margin, 0.5)  # Should hit max with rate=1.0

    def test_margin_clamped_to_range(self):
        """Test that margin stays within [min_margin, max_margin]."""
        am = AdaptiveMargin(base_margin=0.1, min_margin=0.05, max_margin=0.3)
        for _ in range(100):
            am.update([0.9], [0.1])
        self.assertGreaterEqual(am.current_margin, 0.05)
        self.assertLessEqual(am.current_margin, 0.3)

    def test_window_size_respected(self):
        """Test that quality history is bounded by window_size."""
        am = AdaptiveMargin(window_size=5)
        for _ in range(20):
            am.update([0.8], [0.2])
        self.assertLessEqual(len(am._quality_history), 5)

    def test_empty_scores_no_change(self):
        """Test that empty score arrays don't change margin."""
        am = AdaptiveMargin(base_margin=0.1)
        margin = am.update([], [0.5])
        self.assertEqual(margin, 0.1)
        margin = am.update([0.5], [])
        self.assertEqual(margin, 0.1)

    def test_reset(self):
        """Test that reset restores base margin."""
        am = AdaptiveMargin(base_margin=0.1)
        am.update([0.9], [0.1])
        am.reset()
        self.assertEqual(am.current_margin, 0.1)
        self.assertEqual(len(am._quality_history), 0)

    def test_get_state(self):
        """Test get_state returns expected keys."""
        am = AdaptiveMargin(base_margin=0.1, min_margin=0.01, max_margin=0.5)
        am.update([0.8], [0.3])
        state = am.get_state()
        self.assertIn("current_margin", state)
        self.assertIn("base_margin", state)
        self.assertIn("avg_quality_gap", state)
        self.assertEqual(state["history_length"], 1)

    def test_invalid_params(self):
        """Test that invalid parameters raise errors."""
        with self.assertRaises(ValueError):
            AdaptiveMargin(base_margin=-0.1)
        with self.assertRaises(ValueError):
            AdaptiveMargin(min_margin=-1)
        with self.assertRaises(ValueError):
            AdaptiveMargin(min_margin=0.5, max_margin=0.1)
        with self.assertRaises(ValueError):
            AdaptiveMargin(adaptation_rate=0)
        with self.assertRaises(ValueError):
            AdaptiveMargin(window_size=0)


# ============================================================
# Online Loss Scheduler Tests (7 tests)
# ============================================================

@patch('online_updater.get_logger')
class TestOnlineLossScheduler(unittest.TestCase):
    """Tests for OnlineLossScheduler."""

    def test_constant_schedule(self, mock_logger):
        """Test constant schedule maintains initial weight."""
        sched = OnlineLossScheduler(schedule="constant", initial_weight=0.8)
        for _ in range(10):
            w = sched.step()
        self.assertAlmostEqual(w, 0.8, places=5)

    def test_linear_decay_schedule(self, mock_logger):
        """Test linear decay reduces weight over time."""
        sched = OnlineLossScheduler(
            schedule="linear_decay", initial_weight=1.0,
            final_weight=0.1, total_steps=10
        )
        weights = [sched.step() for _ in range(10)]
        self.assertGreater(weights[0], weights[-1])
        self.assertAlmostEqual(weights[-1], 0.1, places=1)

    def test_cosine_annealing_schedule(self, mock_logger):
        """Test cosine annealing decays smoothly."""
        sched = OnlineLossScheduler(
            schedule="cosine_annealing", initial_weight=1.0,
            final_weight=0.0, total_steps=20
        )
        weights = [sched.step() for _ in range(20)]
        # Should decrease overall
        self.assertGreater(weights[0], weights[-1])

    def test_current_weight_property(self, mock_logger):
        """Test current_weight reflects scheduler state."""
        sched = OnlineLossScheduler(schedule="constant", initial_weight=0.5)
        self.assertEqual(sched.current_weight, 0.5)
        sched.step()
        self.assertEqual(sched.current_weight, 0.5)

    def test_reset(self, mock_logger):
        """Test reset restores initial state."""
        sched = OnlineLossScheduler(
            schedule="linear_decay", initial_weight=1.0,
            final_weight=0.1, total_steps=10
        )
        for _ in range(5):
            sched.step()
        sched.reset()
        self.assertEqual(sched.current_weight, 1.0)

    def test_get_state(self, mock_logger):
        """Test get_state returns expected keys."""
        sched = OnlineLossScheduler(schedule="constant", initial_weight=0.7)
        sched.step()
        state = sched.get_state()
        self.assertEqual(state["schedule"], "constant")
        self.assertEqual(state["initial_weight"], 0.7)
        self.assertEqual(state["step_count"], 1)

    def test_adaptive_schedule(self, mock_logger):
        """Test adaptive schedule responds to loss values."""
        sched = OnlineLossScheduler(
            schedule="adaptive", initial_weight=0.5, patience=2
        )
        # Improving losses should decrease weight
        sched.step(loss=1.0)
        w1 = sched.current_weight
        sched.step(loss=0.5)
        w2 = sched.current_weight
        self.assertLess(w2, w1)


# ============================================================
# Online Update Diagnostics Tests (8 tests)
# ============================================================

class TestOnlineUpdateDiagnostics(unittest.TestCase):
    """Tests for OnlineUpdateDiagnostics."""

    def setUp(self):
        self.dim = 64
        self.adapter = ChelationAdapter(input_dim=self.dim)

    def test_initialization(self):
        """Test diagnostics initializes with correct dimensions."""
        diag = OnlineUpdateDiagnostics(input_dim=self.dim)
        stats = diag.get_per_dimension_stats()
        self.assertEqual(len(stats["mean"]), self.dim)
        self.assertEqual(stats["count"], 0)

    def test_record_gradients(self):
        """Test recording gradients from adapter."""
        diag = OnlineUpdateDiagnostics(input_dim=self.dim)
        # Create a fake backward pass to populate gradients
        x = torch.randn(1, self.dim)
        out = self.adapter(x)
        loss = out.sum()
        loss.backward()

        diag.record_gradients(self.adapter)
        stats = diag.get_per_dimension_stats()
        self.assertEqual(stats["count"], 1)

    def test_record_loss(self):
        """Test recording loss values."""
        diag = OnlineUpdateDiagnostics(input_dim=self.dim)
        diag.record_loss(0.5)
        diag.record_loss(0.3)
        trend = diag.get_loss_trend()
        self.assertEqual(trend["count"], 2)
        self.assertTrue(trend["improving"])  # 0.5 -> 0.3 is improvement

    def test_gradient_health_initial(self):
        """Test gradient health when no gradients recorded."""
        diag = OnlineUpdateDiagnostics(input_dim=self.dim)
        health = diag.get_gradient_health()
        self.assertEqual(health["count"], 0)
        self.assertFalse(health["vanishing"])
        self.assertFalse(health["exploding"])

    def test_gradient_health_detection(self):
        """Test vanishing gradient detection."""
        diag = OnlineUpdateDiagnostics(input_dim=self.dim)
        # Simulate tiny gradients
        diag._grad_norm_history = [1e-9, 1e-10, 1e-8]
        health = diag.get_gradient_health()
        self.assertTrue(health["vanishing"])
        self.assertFalse(health["exploding"])

    def test_loss_trend_slope(self):
        """Test loss trend computation with improving losses."""
        diag = OnlineUpdateDiagnostics(input_dim=self.dim)
        for i in range(10):
            diag.record_loss(1.0 - 0.1 * i)
        trend = diag.get_loss_trend()
        self.assertLess(trend["slope"], 0)
        self.assertTrue(trend["improving"])

    def test_history_size_bounded(self):
        """Test that history is bounded by history_size."""
        diag = OnlineUpdateDiagnostics(input_dim=self.dim, history_size=10)
        for i in range(50):
            diag.record_loss(float(i))
        self.assertLessEqual(len(diag._loss_history), 10)

    def test_stability_tracker_bridge(self):
        """Test bridging to StabilityTracker."""
        mock_tracker = MagicMock()
        diag = OnlineUpdateDiagnostics(
            input_dim=self.dim, stability_tracker=mock_tracker
        )

        # Record loss should bridge
        diag.record_loss(0.5)
        mock_tracker.record_loss.assert_called_once_with(0.5)

        # Record gradients should bridge
        x = torch.randn(1, self.dim)
        out = self.adapter(x)
        loss = out.sum()
        loss.backward()

        diag.record_gradients(self.adapter)
        mock_tracker.record_adapter_snapshot.assert_called_once()

    def test_reset(self):
        """Test reset clears all state."""
        diag = OnlineUpdateDiagnostics(input_dim=self.dim)
        diag.record_loss(0.5)
        diag._grad_norm_history = [1.0, 2.0]
        diag._grad_count = 5
        diag.reset()
        self.assertEqual(diag._grad_count, 0)
        self.assertEqual(len(diag._loss_history), 0)
        self.assertEqual(len(diag._grad_norm_history), 0)

    def test_get_report(self):
        """Test comprehensive report structure."""
        diag = OnlineUpdateDiagnostics(input_dim=self.dim)
        report = diag.get_report()
        self.assertIn("per_dimension", report)
        self.assertIn("gradient_health", report)
        self.assertIn("loss_trend", report)

    def test_invalid_params(self):
        """Test that invalid params raise errors."""
        with self.assertRaises(ValueError):
            OnlineUpdateDiagnostics(input_dim=0)
        with self.assertRaises(ValueError):
            OnlineUpdateDiagnostics(input_dim=64, history_size=0)


# ============================================================
# Extended OnlineUpdater Tests (10 tests)
# ============================================================

@patch('online_updater.get_logger')
class TestOnlineUpdaterExtended(unittest.TestCase):
    """Tests for OnlineUpdater with pluggable loss functions."""

    def setUp(self):
        self.input_dim = 128
        self.adapter = ChelationAdapter(input_dim=self.input_dim)
        self.query_vec = np.random.randn(self.input_dim).astype(np.float32)
        self.top_k = np.random.randn(5, self.input_dim).astype(np.float32)
        self.bottom_k = np.random.randn(5, self.input_dim).astype(np.float32)

    def test_default_loss_type_is_triplet(self, mock_logger):
        """Test default loss type is triplet_margin for backward compat."""
        updater = OnlineUpdater(self.adapter)
        self.assertEqual(updater.loss_type, "triplet_margin")
        self.assertIsInstance(updater.loss_function, TripletMarginOnlineLoss)

    def test_infonce_loss_type(self, mock_logger):
        """Test creating updater with InfoNCE loss."""
        updater = OnlineUpdater(
            self.adapter, loss_type="infonce",
            loss_kwargs={"temperature": 0.1}
        )
        self.assertIsInstance(updater.loss_function, InfoNCEOnlineLoss)
        result = updater.update(self.query_vec, self.top_k, self.bottom_k)
        self.assertTrue(result["updated"])
        self.assertTrue(np.isfinite(result["loss"]))

    def test_cosine_similarity_loss_type(self, mock_logger):
        """Test creating updater with cosine similarity loss."""
        updater = OnlineUpdater(
            self.adapter, loss_type="cosine_similarity",
            loss_kwargs={"pos_weight": 1.5, "neg_weight": 0.5}
        )
        self.assertIsInstance(updater.loss_function, CosineSimilarityOnlineLoss)
        result = updater.update(self.query_vec, self.top_k, self.bottom_k)
        self.assertTrue(result["updated"])
        self.assertTrue(np.isfinite(result["loss"]))

    def test_with_adaptive_margin(self, mock_logger):
        """Test updater with adaptive margin."""
        am = AdaptiveMargin(base_margin=0.1)
        updater = OnlineUpdater(self.adapter, adaptive_margin=am)
        result = updater.update(
            self.query_vec, self.top_k, self.bottom_k,
            pos_scores=[0.9, 0.85], neg_scores=[0.2, 0.15]
        )
        self.assertTrue(result["updated"])
        # Margin should have been updated
        self.assertNotEqual(am.current_margin, 0.1)

    def test_with_scheduler(self, mock_logger):
        """Test updater with loss scheduler."""
        sched = OnlineLossScheduler(
            schedule="linear_decay", initial_weight=1.0,
            final_weight=0.1, total_steps=10
        )
        updater = OnlineUpdater(self.adapter, scheduler=sched)
        # First update
        updater.update(self.query_vec, self.top_k, self.bottom_k)
        self.assertGreater(sched.get_state()["step_count"], 0)

    def test_with_diagnostics(self, mock_logger):
        """Test updater with diagnostics enabled."""
        diag = OnlineUpdateDiagnostics(input_dim=self.input_dim)
        updater = OnlineUpdater(self.adapter, diagnostics=diag,
                                learning_rate=0.01)
        updater.update(self.query_vec, self.top_k, self.bottom_k)
        report = diag.get_report()
        self.assertGreater(report["loss_trend"]["count"], 0)

    def test_get_stats_includes_loss_state(self, mock_logger):
        """Test that get_stats includes loss function state."""
        updater = OnlineUpdater(self.adapter, loss_type="infonce",
                                loss_kwargs={"temperature": 0.05})
        stats = updater.get_stats()
        self.assertIn("loss_type", stats)
        self.assertIn("loss_state", stats)
        self.assertEqual(stats["loss_state"]["loss_type"], "infonce")

    def test_get_stats_includes_all_components(self, mock_logger):
        """Test that get_stats includes all optional component states."""
        am = AdaptiveMargin()
        sched = OnlineLossScheduler()
        diag = OnlineUpdateDiagnostics(input_dim=self.input_dim)
        updater = OnlineUpdater(
            self.adapter, adaptive_margin=am,
            scheduler=sched, diagnostics=diag
        )
        stats = updater.get_stats()
        self.assertIn("adaptive_margin", stats)
        self.assertIn("scheduler", stats)
        self.assertIn("diagnostics", stats)

    def test_unknown_loss_type_raises(self, mock_logger):
        """Test that unknown loss type raises ValueError."""
        with self.assertRaises(ValueError):
            OnlineUpdater(self.adapter, loss_type="nonexistent")

    def test_backward_compat_margin_passed_to_triplet(self, mock_logger):
        """Test that margin param is passed to triplet loss for backward compat."""
        updater = OnlineUpdater(self.adapter, margin=0.25)
        self.assertEqual(updater.loss_function.margin, 0.25)


# ============================================================
# Config Preset Tests (3 tests)
# ============================================================

class TestOnlineUpdatePresets(unittest.TestCase):
    """Tests for online_update config presets."""

    def test_conservative_preset(self):
        """Test conservative preset exists and has expected fields."""
        preset = ChelationConfig.get_preset("conservative", "online_update")
        self.assertEqual(preset["loss_type"], "triplet_margin")
        self.assertIn("learning_rate", preset)
        self.assertIn("description", preset)

    def test_balanced_preset(self):
        """Test balanced preset exists and has expected fields."""
        preset = ChelationConfig.get_preset("balanced", "online_update")
        self.assertEqual(preset["loss_type"], "triplet_margin")
        self.assertEqual(preset["margin"], 0.1)

    def test_aggressive_preset(self):
        """Test aggressive preset uses InfoNCE and diagnostics."""
        preset = ChelationConfig.get_preset("aggressive", "online_update")
        self.assertEqual(preset["loss_type"], "infonce")
        self.assertTrue(preset["diagnostics"])
        self.assertTrue(preset["adaptive_margin"])

    def test_invalid_preset_raises(self):
        """Test that invalid preset name raises ValueError."""
        with self.assertRaises(ValueError):
            ChelationConfig.get_preset("nonexistent", "online_update")


if __name__ == "__main__":
    unittest.main(verbosity=2)
