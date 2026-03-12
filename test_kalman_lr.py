"""
Unit Tests for Kalman-Gain Adaptive Learning Rate Scheduler

Tests the KalmanLRScheduler module and its integration with
AntigravityEngine's sedimentation training loops.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock


class TestKalmanLRSchedulerBasic(unittest.TestCase):
    """Test KalmanLRScheduler core functionality."""

    def setUp(self):
        patcher = patch('kalman_lr_scheduler.get_logger', return_value=MagicMock())
        self.mock_logger = patcher.start()
        self.addCleanup(patcher.stop)

        from kalman_lr_scheduler import KalmanLRScheduler
        self.KalmanLRScheduler = KalmanLRScheduler

    def test_initial_lr_is_base_lr(self):
        """Scheduler starts at base_lr before any steps."""
        sched = self.KalmanLRScheduler(base_lr=0.01)
        self.assertAlmostEqual(sched.current_lr, 0.01)

    def test_initial_kalman_gain_is_one(self):
        """Kalman gain starts at 1.0."""
        sched = self.KalmanLRScheduler(base_lr=0.01)
        self.assertAlmostEqual(sched.kalman_gain, 1.0)

    def test_first_step_keeps_base_lr(self):
        """With only one loss sample, variance cannot be computed; LR stays."""
        sched = self.KalmanLRScheduler(base_lr=0.01)
        lr = sched.step(0.5)
        self.assertAlmostEqual(lr, 0.01)

    def test_step_returns_updated_lr(self):
        """After 2+ steps, an updated LR is returned."""
        sched = self.KalmanLRScheduler(base_lr=0.01, process_noise=0.1)
        sched.step(0.5)
        lr = sched.step(0.4)
        self.assertIsInstance(lr, float)
        self.assertGreater(lr, 0)

    def test_current_lr_property_matches_step_return(self):
        """current_lr property equals the last step() return."""
        sched = self.KalmanLRScheduler(base_lr=0.01)
        sched.step(0.5)
        lr = sched.step(0.4)
        self.assertAlmostEqual(sched.current_lr, lr)


class TestKalmanLRVarianceBehavior(unittest.TestCase):
    """Test that variance-based Kalman gain behaves correctly."""

    def setUp(self):
        patcher = patch('kalman_lr_scheduler.get_logger', return_value=MagicMock())
        self.mock_logger = patcher.start()
        self.addCleanup(patcher.stop)

        from kalman_lr_scheduler import KalmanLRScheduler
        self.KalmanLRScheduler = KalmanLRScheduler

    def test_high_variance_lowers_lr(self):
        """Wildly oscillating losses should drive LR below base_lr."""
        sched = self.KalmanLRScheduler(base_lr=0.01, process_noise=0.1,
                                       min_lr_ratio=0.01, max_lr_ratio=5.0)
        # Feed wildly varying losses
        losses = [0.1, 10.0, 0.1, 10.0, 0.1, 10.0]
        for loss in losses:
            sched.step(loss)
        self.assertLess(sched.current_lr, 0.01,
                        "High variance should lower LR below base_lr")

    def test_low_variance_raises_lr(self):
        """Very stable losses should drive the Kalman gain close to 1, giving ~base_lr."""
        sched = self.KalmanLRScheduler(base_lr=0.01, process_noise=0.1,
                                       min_lr_ratio=0.01, max_lr_ratio=5.0)
        # Feed identical losses -> R ~ 0 -> K ~ 1 -> lr ~ base_lr
        for _ in range(10):
            sched.step(0.5)
        # With near-zero variance, gain approaches 1.0 and LR approaches base_lr
        self.assertAlmostEqual(sched.current_lr, 0.01, places=3,
                               msg="Low variance should keep LR near base_lr")
        self.assertGreater(sched.kalman_gain, 0.9)

    def test_stable_vs_noisy_lr_ordering(self):
        """Stable-loss scheduler should have higher LR than noisy-loss scheduler."""
        stable = self.KalmanLRScheduler(base_lr=0.01, process_noise=0.1,
                                        min_lr_ratio=0.01, max_lr_ratio=5.0)
        noisy = self.KalmanLRScheduler(base_lr=0.01, process_noise=0.1,
                                       min_lr_ratio=0.01, max_lr_ratio=5.0)
        for i in range(10):
            stable.step(0.5)
            noisy.step(0.5 + (i % 2) * 2.0)  # alternate between 0.5 and 2.5

        self.assertGreater(stable.current_lr, noisy.current_lr)


class TestKalmanLRClamping(unittest.TestCase):
    """Test that LR stays within min/max bounds."""

    def setUp(self):
        patcher = patch('kalman_lr_scheduler.get_logger', return_value=MagicMock())
        self.mock_logger = patcher.start()
        self.addCleanup(patcher.stop)

        from kalman_lr_scheduler import KalmanLRScheduler
        self.KalmanLRScheduler = KalmanLRScheduler

    def test_lr_never_below_min(self):
        """LR never falls below base_lr * min_lr_ratio."""
        sched = self.KalmanLRScheduler(base_lr=0.01, process_noise=0.001,
                                       min_lr_ratio=0.5, max_lr_ratio=2.0)
        min_lr = 0.01 * 0.5
        # High variance losses
        for loss in [0.01, 100.0, 0.01, 100.0, 0.01, 100.0]:
            sched.step(loss)
        self.assertGreaterEqual(sched.current_lr, min_lr - 1e-12)

    def test_lr_never_above_max(self):
        """LR never exceeds base_lr * max_lr_ratio."""
        sched = self.KalmanLRScheduler(base_lr=0.01, process_noise=100.0,
                                       min_lr_ratio=0.1, max_lr_ratio=1.5)
        max_lr = 0.01 * 1.5
        # Very stable losses with huge process noise should try to push K -> 1
        for _ in range(20):
            sched.step(0.5)
        self.assertLessEqual(sched.current_lr, max_lr + 1e-12)


class TestKalmanLRReset(unittest.TestCase):
    """Test reset() restores initial state."""

    def setUp(self):
        patcher = patch('kalman_lr_scheduler.get_logger', return_value=MagicMock())
        self.mock_logger = patcher.start()
        self.addCleanup(patcher.stop)

        from kalman_lr_scheduler import KalmanLRScheduler
        self.KalmanLRScheduler = KalmanLRScheduler

    def test_reset_restores_initial_state(self):
        """After reset, scheduler should behave as freshly constructed."""
        sched = self.KalmanLRScheduler(base_lr=0.01, process_noise=0.1)
        # Step a few times
        for loss in [0.5, 0.4, 0.3, 0.35]:
            sched.step(loss)
        self.assertNotEqual(sched._step_count, 0)

        sched.reset()

        self.assertEqual(sched._step_count, 0)
        self.assertAlmostEqual(sched.current_lr, 0.01)
        self.assertAlmostEqual(sched.kalman_gain, 1.0)
        self.assertEqual(len(sched._loss_history), 0)

    def test_reset_allows_fresh_training(self):
        """After reset, first step should return base_lr again."""
        sched = self.KalmanLRScheduler(base_lr=0.05)
        sched.step(1.0)
        sched.step(0.5)
        sched.reset()
        lr = sched.step(0.7)
        self.assertAlmostEqual(lr, 0.05)


class TestKalmanLRGetState(unittest.TestCase):
    """Test get_state() returns accurate snapshot."""

    def setUp(self):
        patcher = patch('kalman_lr_scheduler.get_logger', return_value=MagicMock())
        self.mock_logger = patcher.start()
        self.addCleanup(patcher.stop)

        from kalman_lr_scheduler import KalmanLRScheduler
        self.KalmanLRScheduler = KalmanLRScheduler

    def test_state_keys_present(self):
        """get_state() returns all expected keys."""
        sched = self.KalmanLRScheduler(base_lr=0.01)
        state = sched.get_state()
        expected_keys = {"current_lr", "kalman_gain", "base_lr", "process_noise",
                         "step_count", "loss_variance", "window_size"}
        self.assertEqual(set(state.keys()), expected_keys)

    def test_state_reflects_steps(self):
        """State reflects the number of steps taken."""
        sched = self.KalmanLRScheduler(base_lr=0.01)
        sched.step(0.5)
        sched.step(0.4)
        sched.step(0.3)
        state = sched.get_state()
        self.assertEqual(state["step_count"], 3)
        self.assertGreater(state["loss_variance"], 0)

    def test_state_variance_zero_before_two_steps(self):
        """Variance is 0 when fewer than 2 losses are recorded."""
        sched = self.KalmanLRScheduler(base_lr=0.01)
        state = sched.get_state()
        self.assertEqual(state["loss_variance"], 0.0)

        sched.step(0.5)
        state = sched.get_state()
        self.assertEqual(state["loss_variance"], 0.0)


class TestKalmanLRWindowBehavior(unittest.TestCase):
    """Test rolling window mechanics."""

    def setUp(self):
        patcher = patch('kalman_lr_scheduler.get_logger', return_value=MagicMock())
        self.mock_logger = patcher.start()
        self.addCleanup(patcher.stop)

        from kalman_lr_scheduler import KalmanLRScheduler
        self.KalmanLRScheduler = KalmanLRScheduler

    def test_window_bounded(self):
        """Loss history never exceeds window_size."""
        sched = self.KalmanLRScheduler(base_lr=0.01, window_size=5)
        for i in range(20):
            sched.step(float(i))
        self.assertLessEqual(len(sched._loss_history), 5)

    def test_old_losses_forgotten(self):
        """After window slides, old high-variance losses are forgotten."""
        sched = self.KalmanLRScheduler(base_lr=0.01, process_noise=0.1,
                                       window_size=5,
                                       min_lr_ratio=0.01, max_lr_ratio=5.0)
        # Phase 1: high variance
        for loss in [0.1, 10.0, 0.1, 10.0, 0.1]:
            sched.step(loss)
        lr_after_noisy = sched.current_lr

        # Phase 2: stable losses push old noisy ones out of window
        for _ in range(10):
            sched.step(0.5)
        lr_after_stable = sched.current_lr

        self.assertGreater(lr_after_stable, lr_after_noisy,
                           "LR should recover as noisy losses leave the window")


class TestKalmanLRValidation(unittest.TestCase):
    """Test constructor validation."""

    def setUp(self):
        patcher = patch('kalman_lr_scheduler.get_logger', return_value=MagicMock())
        self.mock_logger = patcher.start()
        self.addCleanup(patcher.stop)

        from kalman_lr_scheduler import KalmanLRScheduler
        self.KalmanLRScheduler = KalmanLRScheduler

    def test_negative_base_lr_raises(self):
        with self.assertRaises(ValueError):
            self.KalmanLRScheduler(base_lr=-0.01)

    def test_zero_base_lr_raises(self):
        with self.assertRaises(ValueError):
            self.KalmanLRScheduler(base_lr=0.0)

    def test_negative_process_noise_raises(self):
        with self.assertRaises(ValueError):
            self.KalmanLRScheduler(process_noise=-0.1)

    def test_window_size_too_small_raises(self):
        with self.assertRaises(ValueError):
            self.KalmanLRScheduler(window_size=1)

    def test_max_ratio_less_than_min_raises(self):
        with self.assertRaises(ValueError):
            self.KalmanLRScheduler(min_lr_ratio=2.0, max_lr_ratio=1.0)


class TestKalmanLRConfigPresets(unittest.TestCase):
    """Test config presets for Kalman LR."""

    def test_conservative_preset(self):
        from config import ChelationConfig
        preset = ChelationConfig.get_preset("conservative", "kalman_lr")
        self.assertEqual(preset["process_noise"], 0.05)
        self.assertEqual(preset["min_lr_ratio"], 0.1)
        self.assertEqual(preset["max_lr_ratio"], 1.5)

    def test_balanced_preset(self):
        from config import ChelationConfig
        preset = ChelationConfig.get_preset("balanced", "kalman_lr")
        self.assertEqual(preset["process_noise"], 0.1)
        self.assertEqual(preset["min_lr_ratio"], 0.1)
        self.assertEqual(preset["max_lr_ratio"], 2.0)

    def test_aggressive_preset(self):
        from config import ChelationConfig
        preset = ChelationConfig.get_preset("aggressive", "kalman_lr")
        self.assertEqual(preset["process_noise"], 0.2)
        self.assertEqual(preset["min_lr_ratio"], 0.2)
        self.assertEqual(preset["max_lr_ratio"], 3.0)

    def test_invalid_preset_raises(self):
        from config import ChelationConfig
        with self.assertRaises(ValueError):
            ChelationConfig.get_preset("nonexistent", "kalman_lr")


class TestKalmanLREngineIntegration(unittest.TestCase):
    """Test that AntigravityEngine.enable_kalman_lr activates the scheduler."""

    def setUp(self):
        patcher = patch('antigravity_engine.get_logger', return_value=MagicMock())
        self.mock_logger = patcher.start()
        self.addCleanup(patcher.stop)

    def test_enable_kalman_lr_sets_flags(self):
        """enable_kalman_lr stores configuration on the engine."""
        from antigravity_engine import AntigravityEngine
        engine = AntigravityEngine(
            qdrant_location=":memory:",
            model_name="all-MiniLM-L6-v2",
            use_quantization=True,
        )
        try:
            engine.enable_kalman_lr(
                process_noise=0.05,
                min_lr_ratio=0.2,
                max_lr_ratio=3.0,
                window_size=15,
            )
            self.assertTrue(engine._kalman_lr_enabled)
            self.assertAlmostEqual(engine._kalman_process_noise, 0.05)
            self.assertAlmostEqual(engine._kalman_min_lr_ratio, 0.2)
            self.assertAlmostEqual(engine._kalman_max_lr_ratio, 3.0)
            self.assertEqual(engine._kalman_window_size, 15)
        finally:
            engine.close()

    def test_enable_kalman_lr_default_values(self):
        """enable_kalman_lr uses defaults when called with no arguments."""
        from antigravity_engine import AntigravityEngine
        engine = AntigravityEngine(
            qdrant_location=":memory:",
            model_name="all-MiniLM-L6-v2",
            use_quantization=True,
        )
        try:
            engine.enable_kalman_lr()
            self.assertTrue(engine._kalman_lr_enabled)
            self.assertAlmostEqual(engine._kalman_process_noise, 0.1)
            self.assertAlmostEqual(engine._kalman_min_lr_ratio, 0.1)
            self.assertAlmostEqual(engine._kalman_max_lr_ratio, 2.0)
            self.assertEqual(engine._kalman_window_size, 10)
        finally:
            engine.close()

    def test_kalman_not_enabled_by_default(self):
        """Kalman LR should not be active unless explicitly enabled."""
        from antigravity_engine import AntigravityEngine
        engine = AntigravityEngine(
            qdrant_location=":memory:",
            model_name="all-MiniLM-L6-v2",
            use_quantization=True,
        )
        try:
            self.assertFalse(getattr(engine, '_kalman_lr_enabled', False))
        finally:
            engine.close()


class TestKalmanLRProcessNoiseSensitivity(unittest.TestCase):
    """Test sensitivity to process_noise parameter Q."""

    def setUp(self):
        patcher = patch('kalman_lr_scheduler.get_logger', return_value=MagicMock())
        self.mock_logger = patcher.start()
        self.addCleanup(patcher.stop)

        from kalman_lr_scheduler import KalmanLRScheduler
        self.KalmanLRScheduler = KalmanLRScheduler

    def test_higher_Q_means_higher_gain(self):
        """Larger process_noise Q trusts corrections more (higher gain)."""
        low_q = self.KalmanLRScheduler(base_lr=0.01, process_noise=0.01,
                                       min_lr_ratio=0.01, max_lr_ratio=5.0)
        high_q = self.KalmanLRScheduler(base_lr=0.01, process_noise=1.0,
                                        min_lr_ratio=0.01, max_lr_ratio=5.0)
        # Same moderately noisy losses
        losses = [0.5, 0.3, 0.7, 0.4, 0.6]
        for loss in losses:
            low_q.step(loss)
            high_q.step(loss)

        self.assertGreater(high_q.kalman_gain, low_q.kalman_gain,
                           "Higher Q should yield higher Kalman gain")
        self.assertGreater(high_q.current_lr, low_q.current_lr,
                           "Higher Q should yield higher LR")


if __name__ == "__main__":
    unittest.main()
