"""
Unit Tests for Sweep-Validated Sedimentation Presets (Session 22)

Tests the sedimentation_tuned preset category and collapse-risk validation
derived from 81-config parameter sweep on SciFact.
"""

import unittest
import io
from contextlib import redirect_stdout

from config import ChelationConfig


class TestSedimentationTunedPresets(unittest.TestCase):
    """Test sweep-validated sedimentation_tuned presets."""

    def test_get_preset_conservative(self):
        """Test conservative sedimentation_tuned preset values."""
        preset = ChelationConfig.get_preset("conservative", "sedimentation_tuned")

        self.assertEqual(preset["learning_rate"], 0.001)
        self.assertEqual(preset["threshold"], 1)
        self.assertEqual(preset["noise_scale"], 0.05)
        self.assertEqual(preset["epochs"], 5)
        self.assertIn("description", preset)

    def test_get_preset_balanced(self):
        """Test balanced sedimentation_tuned preset values."""
        preset = ChelationConfig.get_preset("balanced", "sedimentation_tuned")

        self.assertEqual(preset["learning_rate"], 0.01)
        self.assertEqual(preset["threshold"], 1)
        self.assertEqual(preset["noise_scale"], 0.1)
        self.assertEqual(preset["epochs"], 5)
        self.assertIn("description", preset)

    def test_get_preset_aggressive(self):
        """Test aggressive sedimentation_tuned preset values."""
        preset = ChelationConfig.get_preset("aggressive", "sedimentation_tuned")

        self.assertEqual(preset["learning_rate"], 0.01)
        self.assertEqual(preset["threshold"], 1)
        self.assertEqual(preset["noise_scale"], 0.2)
        self.assertEqual(preset["epochs"], 10)
        self.assertIn("description", preset)

    def test_preset_returns_copy(self):
        """Test that get_preset returns a copy, not a reference."""
        preset1 = ChelationConfig.get_preset("balanced", "sedimentation_tuned")
        preset2 = ChelationConfig.get_preset("balanced", "sedimentation_tuned")

        preset1["learning_rate"] = 999
        self.assertEqual(preset2["learning_rate"], 0.01)

    def test_invalid_preset_name_raises(self):
        """Test that invalid preset name raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            ChelationConfig.get_preset("nonexistent", "sedimentation_tuned")
        self.assertIn("nonexistent", str(cm.exception))

    def test_all_presets_have_required_keys(self):
        """Test that all sedimentation_tuned presets have required keys."""
        required_keys = {"learning_rate", "threshold", "noise_scale", "epochs", "description"}

        for name in ["conservative", "balanced", "aggressive"]:
            preset = ChelationConfig.get_preset(name, "sedimentation_tuned")
            for key in required_keys:
                self.assertIn(key, preset, f"Preset '{name}' missing key '{key}'")

    def test_all_learning_rates_below_collapse_threshold(self):
        """Test that no tuned preset uses LR >= 0.1 (collapse risk)."""
        for name in ["conservative", "balanced", "aggressive"]:
            preset = ChelationConfig.get_preset(name, "sedimentation_tuned")
            self.assertLess(
                preset["learning_rate"],
                ChelationConfig.SWEEP_LR_COLLAPSE_THRESHOLD,
                f"Preset '{name}' learning_rate {preset['learning_rate']} >= collapse threshold"
            )

    def test_validate_sedimentation_lr_warns_on_collapse_risk(self):
        """Test that validate_sedimentation_learning_rate warns for LR >= 0.1."""
        captured = io.StringIO()
        with redirect_stdout(captured):
            result = ChelationConfig.validate_sedimentation_learning_rate(0.5)

        self.assertEqual(result, 0.5)
        output = captured.getvalue()
        self.assertIn("WARNING", output)
        self.assertIn("catastrophic collapse", output)
        self.assertIn("61.5%", output)

    def test_validate_sedimentation_lr_no_warn_safe_value(self):
        """Test that validate_sedimentation_learning_rate does not warn for safe LR."""
        captured = io.StringIO()
        with redirect_stdout(captured):
            result = ChelationConfig.validate_sedimentation_learning_rate(0.01)

        self.assertEqual(result, 0.01)
        output = captured.getvalue()
        self.assertEqual(output, "")

    def test_validate_sedimentation_lr_clamps_out_of_range(self):
        """Test that validate_sedimentation_learning_rate still clamps extreme values."""
        captured = io.StringIO()
        with redirect_stdout(captured):
            result = ChelationConfig.validate_sedimentation_learning_rate(10.0)

        self.assertEqual(result, 1.0)
        output = captured.getvalue()
        self.assertIn("WARNING", output)

    def test_sedimentation_tuned_in_preset_type_list(self):
        """Test that sedimentation_tuned is listed in invalid preset_type error."""
        with self.assertRaises(ValueError) as cm:
            ChelationConfig.get_preset("balanced", "invalid_type")
        self.assertIn("sedimentation_tuned", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
