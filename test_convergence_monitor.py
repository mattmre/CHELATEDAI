"""
Unit Tests for Convergence Monitor

Tests patience-based early stopping logic for sedimentation and distillation
training loops without requiring external services.
"""

import math
import unittest
from unittest.mock import patch, MagicMock

# Patch the logger before importing convergence_monitor so that no log files
# are created during tests.
_mock_logger = MagicMock()


def _fake_get_logger(*args, **kwargs):
    return _mock_logger


with patch("chelation_logger.get_logger", _fake_get_logger):
    import convergence_monitor
    convergence_monitor.get_logger = _fake_get_logger

from convergence_monitor import ConvergenceMonitor


class TestConvergenceMonitorInit(unittest.TestCase):
    """Test ConvergenceMonitor initialization and validation."""

    def test_initialization_defaults(self):
        """Default patience=5, rel_threshold=0.001, min_epochs=3."""
        mon = ConvergenceMonitor()
        self.assertEqual(mon.patience, 5)
        self.assertAlmostEqual(mon.rel_threshold, 0.001)
        self.assertEqual(mon.min_epochs, 3)
        self.assertFalse(mon.converged)
        self.assertEqual(mon.total_epochs, 0)

    def test_initialization_custom(self):
        """Custom values are stored correctly."""
        mon = ConvergenceMonitor(patience=10, rel_threshold=0.05, min_epochs=7)
        self.assertEqual(mon.patience, 10)
        self.assertAlmostEqual(mon.rel_threshold, 0.05)
        self.assertEqual(mon.min_epochs, 7)

    def test_initialization_invalid_patience(self):
        """patience < 1 raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            ConvergenceMonitor(patience=0)
        self.assertIn("patience", str(cm.exception))

        with self.assertRaises(ValueError):
            ConvergenceMonitor(patience=-3)

    def test_initialization_invalid_rel_threshold(self):
        """Negative rel_threshold raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            ConvergenceMonitor(rel_threshold=-0.01)
        self.assertIn("rel_threshold", str(cm.exception))

    def test_initialization_invalid_min_epochs(self):
        """min_epochs < 1 raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            ConvergenceMonitor(min_epochs=0)
        self.assertIn("min_epochs", str(cm.exception))

        with self.assertRaises(ValueError):
            ConvergenceMonitor(min_epochs=-1)


class TestConvergenceMonitorRecordLoss(unittest.TestCase):
    """Test record_loss convergence detection logic."""

    def test_record_loss_decreasing(self):
        """Steadily decreasing loss never triggers convergence."""
        mon = ConvergenceMonitor(patience=3, min_epochs=2)
        for i in range(20):
            # Large decreases that always exceed rel_threshold
            result = mon.record_loss(1.0 - i * 0.04)
            self.assertFalse(result, f"Should not converge at epoch {i + 1}")
        self.assertFalse(mon.converged)

    def test_record_loss_converged(self):
        """Flat loss triggers convergence after patience epochs."""
        mon = ConvergenceMonitor(patience=3, rel_threshold=0.001, min_epochs=2)
        # Epoch 1: set baseline
        self.assertFalse(mon.record_loss(0.5))
        # Epochs 2-4: same loss (3 epochs without improvement)
        self.assertFalse(mon.record_loss(0.5))
        self.assertFalse(mon.record_loss(0.5))
        # Epoch 4: patience=3 reached, min_epochs=2 satisfied
        self.assertTrue(mon.record_loss(0.5))
        self.assertTrue(mon.converged)

    def test_record_loss_min_epochs_respected(self):
        """Cannot converge before min_epochs even if patience is exhausted."""
        mon = ConvergenceMonitor(patience=1, rel_threshold=0.001, min_epochs=5)
        # Epochs 1-4: flat loss, patience exhausted but min_epochs not met
        for i in range(4):
            result = mon.record_loss(1.0)
            self.assertFalse(result, f"Should not converge before min_epochs at epoch {i + 1}")
        # Epoch 5: now min_epochs is met
        self.assertTrue(mon.record_loss(1.0))

    def test_record_loss_first_epoch_sets_best(self):
        """First epoch always updates best_loss regardless of value."""
        mon = ConvergenceMonitor()
        mon.record_loss(42.0)
        self.assertAlmostEqual(mon.best_loss, 42.0)
        self.assertEqual(mon.epochs_without_improvement, 0)

    def test_record_loss_rel_threshold(self):
        """Small improvements below rel_threshold count as no improvement."""
        # rel_threshold=0.1 means 10% improvement needed
        mon = ConvergenceMonitor(patience=2, rel_threshold=0.1, min_epochs=1)
        mon.record_loss(1.0)  # baseline
        # Improvement of 0.001 / 1.0 = 0.1% -- well below 10% threshold
        mon.record_loss(0.999)
        self.assertEqual(mon.epochs_without_improvement, 1)
        # Large improvement: 0.999 -> 0.5 is > 10% relative to best (1.0)
        mon.record_loss(0.5)
        self.assertEqual(mon.epochs_without_improvement, 0)
        self.assertAlmostEqual(mon.best_loss, 0.5)

    def test_record_loss_nan_handling(self):
        """NaN loss does not trigger stop and is not recorded in history."""
        mon = ConvergenceMonitor()
        mon.record_loss(1.0)
        result = mon.record_loss(float('nan'))
        self.assertFalse(result)
        # NaN should not appear in loss history
        self.assertEqual(len(mon.loss_history), 1)

    def test_record_loss_inf_handling(self):
        """Inf loss does not trigger stop and is not recorded in history."""
        mon = ConvergenceMonitor()
        mon.record_loss(1.0)
        result = mon.record_loss(float('inf'))
        self.assertFalse(result)
        self.assertEqual(len(mon.loss_history), 1)

        result_neg = mon.record_loss(float('-inf'))
        self.assertFalse(result_neg)
        self.assertEqual(len(mon.loss_history), 1)


class TestConvergenceMonitorProperties(unittest.TestCase):
    """Test property accessors and state tracking."""

    def test_converged_property_false_initially(self):
        """converged starts False."""
        mon = ConvergenceMonitor()
        self.assertFalse(mon.converged)

    def test_converged_property_true_after_convergence(self):
        """converged is True after detection."""
        mon = ConvergenceMonitor(patience=1, min_epochs=2)
        mon.record_loss(1.0)
        mon.record_loss(1.0)  # min_epochs=2 met, patience=1 exhausted
        self.assertTrue(mon.converged)

    def test_loss_history_tracking(self):
        """loss_history records all finite values in order."""
        mon = ConvergenceMonitor()
        values = [0.9, 0.8, 0.7, 0.6, 0.5]
        for v in values:
            mon.record_loss(v)
        self.assertEqual(mon.loss_history, values)
        # Returned list should be a copy
        mon.loss_history.append(999)
        self.assertEqual(len(mon.loss_history), 5)

    def test_best_loss_tracking(self):
        """best_loss tracks the minimum loss seen."""
        mon = ConvergenceMonitor()
        mon.record_loss(0.9)
        self.assertAlmostEqual(mon.best_loss, 0.9)
        mon.record_loss(0.5)
        self.assertAlmostEqual(mon.best_loss, 0.5)
        mon.record_loss(0.7)  # worse
        self.assertAlmostEqual(mon.best_loss, 0.5)

    def test_epochs_without_improvement_tracking(self):
        """Counter increments on non-improving epochs, resets on improvement."""
        mon = ConvergenceMonitor(rel_threshold=0.01)
        mon.record_loss(1.0)  # epoch 1: baseline
        self.assertEqual(mon.epochs_without_improvement, 0)
        mon.record_loss(1.0)  # epoch 2: no improvement
        self.assertEqual(mon.epochs_without_improvement, 1)
        mon.record_loss(1.0)  # epoch 3: no improvement
        self.assertEqual(mon.epochs_without_improvement, 2)
        mon.record_loss(0.5)  # epoch 4: big improvement
        self.assertEqual(mon.epochs_without_improvement, 0)

    def test_total_epochs_property(self):
        """total_epochs matches the number of recorded losses."""
        mon = ConvergenceMonitor()
        self.assertEqual(mon.total_epochs, 0)
        mon.record_loss(1.0)
        self.assertEqual(mon.total_epochs, 1)
        mon.record_loss(0.9)
        mon.record_loss(0.8)
        self.assertEqual(mon.total_epochs, 3)


class TestConvergenceMonitorResetAndSummary(unittest.TestCase):
    """Test reset and get_summary methods."""

    def test_reset_clears_state(self):
        """reset() clears all state back to initial values."""
        mon = ConvergenceMonitor(patience=2, min_epochs=1)
        # Build up state
        mon.record_loss(1.0)
        mon.record_loss(1.0)
        mon.record_loss(1.0)  # triggers convergence
        self.assertTrue(mon.converged)
        self.assertGreater(mon.total_epochs, 0)

        mon.reset()

        self.assertFalse(mon.converged)
        self.assertEqual(mon.total_epochs, 0)
        self.assertEqual(mon.loss_history, [])
        self.assertEqual(mon.best_loss, float('inf'))
        self.assertEqual(mon.epochs_without_improvement, 0)

    def test_get_summary_initial(self):
        """Summary for a fresh monitor has expected keys and defaults."""
        mon = ConvergenceMonitor(patience=5, rel_threshold=0.001, min_epochs=3)
        summary = mon.get_summary()

        self.assertFalse(summary["converged"])
        self.assertEqual(summary["total_epochs"], 0)
        self.assertIsNone(summary["best_loss"])
        self.assertEqual(summary["epochs_without_improvement"], 0)
        self.assertEqual(summary["patience"], 5)
        self.assertAlmostEqual(summary["rel_threshold"], 0.001)
        self.assertEqual(summary["min_epochs"], 3)
        self.assertEqual(summary["loss_history"], [])

    def test_get_summary_after_training(self):
        """Summary reflects training state correctly."""
        mon = ConvergenceMonitor(patience=2, min_epochs=2)
        mon.record_loss(1.0)
        mon.record_loss(0.5)
        mon.record_loss(0.5)
        mon.record_loss(0.5)  # converges here

        summary = mon.get_summary()
        self.assertTrue(summary["converged"])
        self.assertEqual(summary["total_epochs"], 4)
        self.assertAlmostEqual(summary["best_loss"], 0.5)
        self.assertEqual(summary["loss_history"], [1.0, 0.5, 0.5, 0.5])

    def test_patience_1_stops_immediately(self):
        """patience=1 stops after the first epoch without improvement."""
        mon = ConvergenceMonitor(patience=1, rel_threshold=0.001, min_epochs=1)
        mon.record_loss(1.0)  # epoch 1: baseline, min_epochs met
        # epoch 2: same loss => 1 epoch without improvement => patience hit
        result = mon.record_loss(1.0)
        self.assertTrue(result)
        self.assertTrue(mon.converged)
        self.assertEqual(mon.total_epochs, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
