import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_overlay_model_scope_validation import run_validation_bundle


class TestOverlayModelScopeValidation(unittest.TestCase):
    def test_validation_bundle_writes_summary_and_reports_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "validation"

            def fake_run(name, command, *, cwd, timeout_seconds):
                return {
                    "name": name,
                    "command": command,
                    "returncode": 1 if name == "model_scope_overlay_smoke" else 0,
                    "duration_seconds": 0.01,
                    "stdout_tail": "",
                    "stderr_tail": "boom" if name == "model_scope_overlay_smoke" else "",
                }

            with patch("run_overlay_model_scope_validation._run_command", side_effect=fake_run):
                summary = run_validation_bundle(output_dir=output_dir, timeout_seconds=10, cwd=Path.cwd())

            summary_path = Path(summary["summary_path"])
            self.assertTrue(summary_path.exists())
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertFalse(payload["passed"])
            self.assertEqual(payload["failed_commands"], ["model_scope_overlay_smoke"])
            self.assertEqual(payload["command_count"], 2)

    def test_validation_bundle_passes_when_all_commands_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "validation"

            def fake_run(name, command, *, cwd, timeout_seconds):
                return {
                    "name": name,
                    "command": command,
                    "returncode": 0,
                    "duration_seconds": 0.01,
                    "stdout_tail": "",
                    "stderr_tail": "",
                }

            with patch("run_overlay_model_scope_validation._run_command", side_effect=fake_run):
                summary = run_validation_bundle(output_dir=output_dir, timeout_seconds=10, cwd=Path.cwd())

            self.assertTrue(summary["passed"])
            self.assertEqual(summary["failed_commands"], [])


if __name__ == "__main__":
    unittest.main()
