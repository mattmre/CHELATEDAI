"""Unit Tests for TeacherWeightScheduler module."""

import unittest
from unittest.mock import MagicMock, patch
from teacher_weight_scheduler import TeacherWeightScheduler, create_weight_scheduler


@patch("teacher_weight_scheduler.get_logger")
class TestTeacherWeightScheduler(unittest.TestCase):
    """Test all schedule types and edge cases."""

    def test_constant_schedule(self, mock_logger):
        """Test constant schedule returns initial weight."""
        mock_logger.return_value = MagicMock()
        s = TeacherWeightScheduler(schedule="constant", initial_weight=0.5)
        for _ in range(10):
            w = s.step()
        self.assertAlmostEqual(w, 0.5)

    def test_linear_decay_progress(self, mock_logger):
        """Test linear decay changes over steps."""
        mock_logger.return_value = MagicMock()
        s = TeacherWeightScheduler(
            schedule="linear_decay", initial_weight=1.0,
            final_weight=0.0, total_steps=10,
        )
        weights = [s.step() for _ in range(10)]
        # Should be monotonically decreasing
        for i in range(1, len(weights)):
            self.assertLessEqual(weights[i], weights[i - 1])

    def test_linear_decay_final_value(self, mock_logger):
        """Test linear decay reaches final value."""
        mock_logger.return_value = MagicMock()
        s = TeacherWeightScheduler(
            schedule="linear_decay", initial_weight=0.8,
            final_weight=0.1, total_steps=10,
        )
        for _ in range(10):
            w = s.step()
        self.assertAlmostEqual(w, 0.1, places=2)

    def test_cosine_annealing_start(self, mock_logger):
        """Test cosine annealing starts near initial weight."""
        mock_logger.return_value = MagicMock()
        s = TeacherWeightScheduler(
            schedule="cosine_annealing", initial_weight=0.8,
            final_weight=0.1, total_steps=100,
        )
        w = s.step()
        # First step should be close to initial
        self.assertGreater(w, 0.7)

    def test_cosine_annealing_midpoint(self, mock_logger):
        """Test cosine annealing at midpoint."""
        mock_logger.return_value = MagicMock()
        s = TeacherWeightScheduler(
            schedule="cosine_annealing", initial_weight=1.0,
            final_weight=0.0, total_steps=100,
        )
        for _ in range(50):
            w = s.step()
        # At midpoint, cosine(pi/2) = 0, so weight ~ 0.5*(1.0)(1+0) = 0.5
        self.assertAlmostEqual(w, 0.5, places=1)

    def test_cosine_annealing_end(self, mock_logger):
        """Test cosine annealing reaches final value."""
        mock_logger.return_value = MagicMock()
        s = TeacherWeightScheduler(
            schedule="cosine_annealing", initial_weight=0.8,
            final_weight=0.1, total_steps=100,
        )
        for _ in range(100):
            w = s.step()
        self.assertAlmostEqual(w, 0.1, places=2)

    def test_step_decay_values(self, mock_logger):
        """Test step decay decreases at intervals."""
        mock_logger.return_value = MagicMock()
        s = TeacherWeightScheduler(
            schedule="step_decay", initial_weight=1.0,
            gamma=0.5, step_size=5,
        )
        # After 5 steps: 1.0 * 0.5^1 = 0.5
        for _ in range(5):
            w = s.step()
        self.assertAlmostEqual(w, 0.5)
        # After 10 steps: 1.0 * 0.5^2 = 0.25
        for _ in range(5):
            w = s.step()
        self.assertAlmostEqual(w, 0.25)

    def test_step_decay_gamma(self, mock_logger):
        """Test step decay with different gamma."""
        mock_logger.return_value = MagicMock()
        s = TeacherWeightScheduler(
            schedule="step_decay", initial_weight=1.0,
            gamma=0.1, step_size=3,
        )
        for _ in range(3):
            w = s.step()
        self.assertAlmostEqual(w, 0.1)

    def test_adaptive_loss_improving(self, mock_logger):
        """Test adaptive decreases weight when loss improves."""
        mock_logger.return_value = MagicMock()
        s = TeacherWeightScheduler(
            schedule="adaptive", initial_weight=0.5,
            decrease_factor=0.9,
        )
        # Decreasing losses
        w = s.step(loss=1.0)
        w = s.step(loss=0.5)
        # Weight should have decreased
        self.assertLess(w, 0.5)

    def test_adaptive_loss_stagnant(self, mock_logger):
        """Test adaptive increases weight when loss stagnates."""
        mock_logger.return_value = MagicMock()
        s = TeacherWeightScheduler(
            schedule="adaptive", initial_weight=0.5,
            patience=2, increase_factor=1.5,
        )
        # First step sets best loss
        s.step(loss=1.0)
        initial_w = s.current_weight
        # Two stagnant steps
        s.step(loss=1.5)
        s.step(loss=1.5)
        # Weight should have increased after patience exceeded
        self.assertGreater(s.current_weight, initial_w)

    def test_adaptive_patience(self, mock_logger):
        """Test adaptive respects patience before increasing."""
        mock_logger.return_value = MagicMock()
        s = TeacherWeightScheduler(
            schedule="adaptive", initial_weight=0.5,
            patience=5, increase_factor=2.0,
        )
        s.step(loss=1.0)
        w_after_first = s.current_weight
        # 4 stagnant steps (patience=5 not yet exceeded)
        for _ in range(4):
            s.step(loss=2.0)
        # Weight should not have increased via increase_factor yet
        # (only decreased from the first improving step, then stayed)
        self.assertLessEqual(s.current_weight, w_after_first * 2.0)

    def test_warmup_phase(self, mock_logger):
        """Test warmup phase linearly increases weight."""
        mock_logger.return_value = MagicMock()
        s = TeacherWeightScheduler(
            schedule="constant", initial_weight=1.0, warmup_steps=4,
        )
        w1 = s.step()
        self.assertAlmostEqual(w1, 0.25)
        w2 = s.step()
        self.assertAlmostEqual(w2, 0.5)
        w3 = s.step()
        self.assertAlmostEqual(w3, 0.75)
        w4 = s.step()
        self.assertAlmostEqual(w4, 1.0)

    def test_warmup_then_decay(self, mock_logger):
        """Test warmup followed by linear decay."""
        mock_logger.return_value = MagicMock()
        s = TeacherWeightScheduler(
            schedule="linear_decay", initial_weight=1.0,
            final_weight=0.0, total_steps=12, warmup_steps=2,
        )
        # Warmup
        w1 = s.step()
        self.assertAlmostEqual(w1, 0.5)
        w2 = s.step()
        self.assertAlmostEqual(w2, 1.0)
        # Decay phase (10 effective steps)
        for _ in range(10):
            w = s.step()
        # Should reach final weight
        self.assertAlmostEqual(w, 0.0, places=1)

    def test_min_weight_floor(self, mock_logger):
        """Test weight never goes below min_weight."""
        mock_logger.return_value = MagicMock()
        s = TeacherWeightScheduler(
            schedule="step_decay", initial_weight=1.0,
            gamma=0.01, step_size=1, min_weight=0.05,
        )
        for _ in range(100):
            w = s.step()
        self.assertGreaterEqual(w, 0.05)

    def test_reset(self, mock_logger):
        """Test reset restores initial state."""
        mock_logger.return_value = MagicMock()
        s = TeacherWeightScheduler(
            schedule="linear_decay", initial_weight=0.8,
            final_weight=0.1, total_steps=10,
        )
        for _ in range(10):
            s.step()
        s.reset()
        self.assertAlmostEqual(s.current_weight, 0.8)
        self.assertEqual(s._step_count, 0)
        self.assertEqual(len(s._loss_history), 0)
        self.assertEqual(s._best_loss, float('inf'))

    def test_get_summary(self, mock_logger):
        """Test get_summary returns expected keys."""
        mock_logger.return_value = MagicMock()
        s = TeacherWeightScheduler(schedule="constant", initial_weight=0.5)
        s.step()
        summary = s.get_summary()
        self.assertEqual(summary["schedule"], "constant")
        self.assertAlmostEqual(summary["initial_weight"], 0.5)
        self.assertEqual(summary["step_count"], 1)
        self.assertIn("current_weight", summary)
        self.assertIn("total_steps", summary)

    def test_factory_constant(self, mock_logger):
        """Test factory creates constant scheduler."""
        mock_logger.return_value = MagicMock()
        s = create_weight_scheduler(schedule="constant", initial_weight=0.3)
        self.assertEqual(s.schedule, "constant")
        self.assertAlmostEqual(s.initial_weight, 0.3)

    def test_factory_linear(self, mock_logger):
        """Test factory creates linear scheduler."""
        mock_logger.return_value = MagicMock()
        s = create_weight_scheduler(
            schedule="linear_decay", initial_weight=0.9, final_weight=0.1,
        )
        self.assertEqual(s.schedule, "linear_decay")

    def test_factory_cosine(self, mock_logger):
        """Test factory creates cosine scheduler."""
        mock_logger.return_value = MagicMock()
        s = create_weight_scheduler(schedule="cosine_annealing")
        self.assertEqual(s.schedule, "cosine_annealing")

    def test_factory_step(self, mock_logger):
        """Test factory creates step decay scheduler."""
        mock_logger.return_value = MagicMock()
        s = create_weight_scheduler(schedule="step_decay", gamma=0.3)
        self.assertEqual(s.schedule, "step_decay")
        self.assertAlmostEqual(s.gamma, 0.3)

    def test_factory_adaptive(self, mock_logger):
        """Test factory creates adaptive scheduler."""
        mock_logger.return_value = MagicMock()
        s = create_weight_scheduler(schedule="adaptive", patience=10)
        self.assertEqual(s.schedule, "adaptive")
        self.assertEqual(s.patience, 10)

    def test_factory_kwargs_passthrough(self, mock_logger):
        """Test factory passes kwargs correctly."""
        mock_logger.return_value = MagicMock()
        s = create_weight_scheduler(
            schedule="linear_decay", initial_weight=0.7,
            total_steps=50, final_weight=0.05, min_weight=0.02,
        )
        self.assertEqual(s.total_steps, 50)
        self.assertAlmostEqual(s.final_weight, 0.05)
        self.assertAlmostEqual(s.min_weight, 0.02)

    def test_zero_total_steps_no_crash(self, mock_logger):
        """Test scheduler does not crash with zero total steps."""
        mock_logger.return_value = MagicMock()
        s = TeacherWeightScheduler(
            schedule="linear_decay", initial_weight=0.5,
            total_steps=0,
        )
        w = s.step()
        self.assertIsInstance(w, float)

    def test_negative_loss_handling(self, mock_logger):
        """Test adaptive handles negative loss values."""
        mock_logger.return_value = MagicMock()
        s = TeacherWeightScheduler(schedule="adaptive", initial_weight=0.5)
        w = s.step(loss=-1.0)
        self.assertIsInstance(w, float)
        w = s.step(loss=-2.0)
        self.assertIsInstance(w, float)

    def test_multiple_resets(self, mock_logger):
        """Test multiple reset cycles work correctly."""
        mock_logger.return_value = MagicMock()
        s = TeacherWeightScheduler(
            schedule="linear_decay", initial_weight=1.0,
            final_weight=0.0, total_steps=5,
        )
        for _ in range(5):
            s.step()
        s.reset()
        self.assertAlmostEqual(s.current_weight, 1.0)
        for _ in range(5):
            s.step()
        s.reset()
        self.assertAlmostEqual(s.current_weight, 1.0)
        self.assertEqual(s._step_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
