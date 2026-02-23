"""
Tests for StabilityTracker (Phase 5: Structural Stability Metrics)

Run: python -m pytest test_stability_tracker.py -v
"""

import unittest
from unittest.mock import patch
import numpy as np
import torch

from stability_tracker import StabilityTracker
from chelation_adapter import ChelationAdapter


@patch('stability_tracker.get_logger')
class TestStabilityTracker(unittest.TestCase):
    """Tests for the StabilityTracker class."""

    def test_initialization(self, mock_logger):
        """Test fresh tracker state."""
        tracker = StabilityTracker()
        report = tracker.get_stability_report()
        self.assertEqual(report["total_inferences_tracked"], 0)
        self.assertEqual(report["total_training_cycles_tracked"], 0)

    def test_record_mask(self, mock_logger):
        """Test recording masks."""
        tracker = StabilityTracker()
        mask = np.array([1, 0, 1, 1, 0])
        tracker.record_mask(mask)
        self.assertEqual(len(tracker._mask_history), 1)

    def test_mask_stability_identical(self, mock_logger):
        """Test Jaccard=1.0 for identical consecutive masks."""
        tracker = StabilityTracker()
        mask = np.array([1, 0, 1, 1, 0])
        tracker.record_mask(mask)
        tracker.record_mask(mask)
        stab = tracker.compute_mask_stability()
        self.assertEqual(len(stab), 1)
        self.assertAlmostEqual(stab[0], 1.0)

    def test_mask_stability_disjoint(self, mock_logger):
        """Test Jaccard=0.0 for fully disjoint masks."""
        tracker = StabilityTracker()
        tracker.record_mask(np.array([1, 1, 0, 0]))
        tracker.record_mask(np.array([0, 0, 1, 1]))
        stab = tracker.compute_mask_stability()
        self.assertAlmostEqual(stab[0], 0.0)

    def test_mask_stability_insufficient_data(self, mock_logger):
        """Test empty result with fewer than 2 masks."""
        tracker = StabilityTracker()
        tracker.record_mask(np.array([1, 0]))
        self.assertEqual(tracker.compute_mask_stability(), [])

    def test_record_variance_distribution(self, mock_logger):
        """Test recording variance distributions."""
        tracker = StabilityTracker()
        tracker.record_variance_distribution(np.array([0.1, 0.2, 0.3]))
        self.assertEqual(len(tracker._variance_history), 1)

    def test_variance_convergence_identical(self, mock_logger):
        """Test correlation=1.0 for identical distributions."""
        tracker = StabilityTracker()
        v = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
        tracker.record_variance_distribution(v)
        tracker.record_variance_distribution(v)
        corr = tracker.compute_variance_convergence()
        self.assertEqual(len(corr), 1)
        self.assertAlmostEqual(corr[0], 1.0, places=5)

    def test_variance_convergence_insufficient_data(self, mock_logger):
        """Test empty result with fewer than 2 distributions."""
        tracker = StabilityTracker()
        tracker.record_variance_distribution(np.array([0.1]))
        self.assertEqual(tracker.compute_variance_convergence(), [])

    def test_record_collapse_set(self, mock_logger):
        """Test recording collapse sets."""
        tracker = StabilityTracker()
        tracker.record_collapse_set([1, 2, 3])
        self.assertEqual(len(tracker._collapse_history), 1)
        self.assertEqual(tracker._collapse_history[0], {1, 2, 3})

    def test_persistent_collapse_ratio(self, mock_logger):
        """Test persistent collapse ratio computation."""
        tracker = StabilityTracker()
        # Doc 1 appears in all 4 cycles (persistent), doc 2 in only 1 (not persistent)
        tracker.record_collapse_set([1, 2])
        tracker.record_collapse_set([1])
        tracker.record_collapse_set([1])
        tracker.record_collapse_set([1])
        ratio = tracker.compute_persistent_collapse_ratio()
        # Doc 1: 4 appearances > 4/2=2.0 threshold -> persistent
        # Doc 2: 1 appearance <= 2.0 -> not persistent
        # persistent/total = 1/2 = 0.5
        self.assertAlmostEqual(ratio, 0.5)

    def test_persistent_collapse_empty(self, mock_logger):
        """Test persistent collapse ratio with no data."""
        tracker = StabilityTracker()
        self.assertEqual(tracker.compute_persistent_collapse_ratio(), 0.0)

    def test_record_threshold(self, mock_logger):
        """Test recording thresholds."""
        tracker = StabilityTracker()
        tracker.record_threshold(0.001)
        tracker.record_threshold(0.002)
        self.assertEqual(len(tracker._threshold_history), 2)

    def test_threshold_oscillation(self, mock_logger):
        """Test threshold oscillation (std dev)."""
        tracker = StabilityTracker()
        tracker.record_threshold(0.001)
        tracker.record_threshold(0.003)
        osc = tracker.compute_threshold_oscillation()
        expected = float(np.std([0.001, 0.003]))
        self.assertAlmostEqual(osc, expected, places=8)

    def test_threshold_oscillation_insufficient(self, mock_logger):
        """Test oscillation with fewer than 2 thresholds."""
        tracker = StabilityTracker()
        tracker.record_threshold(0.001)
        self.assertEqual(tracker.compute_threshold_oscillation(), 0.0)

    def test_adapter_drift(self, mock_logger):
        """Test adapter weight drift computation."""
        tracker = StabilityTracker()
        adapter = ChelationAdapter(input_dim=32)
        tracker.record_adapter_snapshot(adapter)

        # Modify adapter weights
        with torch.no_grad():
            for p in adapter.parameters():
                p.add_(0.1)

        tracker.record_adapter_snapshot(adapter)
        drifts = tracker.compute_adapter_drift()
        self.assertEqual(len(drifts), 1)
        self.assertGreater(drifts[0], 0.0)

    def test_adapter_drift_insufficient(self, mock_logger):
        """Test drift with fewer than 2 snapshots."""
        tracker = StabilityTracker()
        self.assertEqual(tracker.compute_adapter_drift(), [])

    def test_record_loss(self, mock_logger):
        """Test recording loss values."""
        tracker = StabilityTracker()
        tracker.record_loss(0.5)
        tracker.record_loss(0.3)
        self.assertEqual(tracker._loss_history, [0.5, 0.3])

    def test_get_stability_report_structure(self, mock_logger):
        """Test report has all expected keys."""
        tracker = StabilityTracker()
        report = tracker.get_stability_report()
        self.assertIn("mask_stability", report)
        self.assertIn("variance_convergence", report)
        self.assertIn("persistent_collapse_ratio", report)
        self.assertIn("threshold_oscillation", report)
        self.assertIn("adapter_drift", report)
        self.assertIn("loss_history", report)
        self.assertIn("total_inferences_tracked", report)
        self.assertIn("total_training_cycles_tracked", report)

    def test_reset(self, mock_logger):
        """Test that reset clears all state."""
        tracker = StabilityTracker()
        tracker.record_mask(np.array([1, 0]))
        tracker.record_variance_distribution(np.array([0.1]))
        tracker.record_collapse_set([1])
        tracker.record_threshold(0.001)
        tracker.record_loss(0.5)
        tracker.reset()
        report = tracker.get_stability_report()
        self.assertEqual(report["total_inferences_tracked"], 0)
        self.assertEqual(report["total_training_cycles_tracked"], 0)
        self.assertEqual(len(report["loss_history"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
