import tempfile
import unittest
from pathlib import Path

from run_model_scope_overlay_smoke import run_smoke


class TestModelScopeOverlaySmoke(unittest.TestCase):
    def test_smoke_campaign_generates_full_overlay_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_smoke(Path(tmpdir))

            self.assertEqual(summary["record_type"], "model_scope_overlay_smoke_summary")
            self.assertTrue(summary["adaptive_overlay_ready"])
            self.assertTrue(summary["validation_ready"])
            self.assertFalse(summary["promotion_ready"])
            self.assertTrue(Path(summary["smoke_summary"]).exists())
            for output_path in summary["required_outputs"].values():
                self.assertTrue(Path(output_path).exists())


if __name__ == "__main__":
    unittest.main()
