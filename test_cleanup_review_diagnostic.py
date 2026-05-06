import json
import os
import subprocess
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from cleanup_review_diagnostic import main, render_cleanup_review_diagnostic


class CleanupReviewDiagnosticTests(unittest.TestCase):
    def test_render_warn_mode_reports_blocked_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "cleanup_plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "summary": {
                            "cleanup_review_allowed": False,
                            "missing_source_artifacts": ["freshness_audit"],
                            "candidate_count": 2,
                            "retained_count": 3,
                            "candidate_bytes": 123,
                        },
                        "source_status": {
                            "evidence_index": {"present": True, "path": "index.json"},
                            "freshness_audit": {"present": False, "path": "freshness.json"},
                        },
                    }
                ),
                encoding="utf-8",
            )

            text = render_cleanup_review_diagnostic(plan_path, mode="warn")

            self.assertIn("## Cleanup Review Warning Mode", text)
            self.assertIn("- Cleanup review: blocked", text)
            self.assertIn("- Missing source artifacts: freshness_audit", text)
            self.assertIn("- Candidate count: 2", text)
            self.assertIn("- Candidate bytes: 123", text)
            self.assertIn("- Source freshness_audit: missing `freshness.json`", text)
            self.assertIn("blocked cleanup candidates must not be used", text)

    def test_render_blocked_mode_handles_missing_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "missing.json"

            text = render_cleanup_review_diagnostic(plan_path, mode="blocked")

            self.assertIn("## Cleanup Review Blocked", text)
            self.assertIn("Cleanup plan missing", text)

    def test_render_handles_invalid_json_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "cleanup_plan.json"
            plan_path.write_text("{not-json", encoding="utf-8")

            text = render_cleanup_review_diagnostic(plan_path, mode="blocked")

            self.assertIn("Cleanup plan unreadable", text)
            self.assertIn("Error:", text)
            self.assertNotIn("Traceback", text)

    def test_main_writes_github_summary_when_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "cleanup_plan.json"
            summary_path = Path(tmpdir) / "summary.md"
            plan_path.write_text(
                json.dumps(
                    {
                        "summary": {
                            "cleanup_review_allowed": True,
                            "candidate_count": 0,
                            "retained_count": 1,
                            "candidate_bytes": 0,
                        },
                        "source_status": {},
                    }
                ),
                encoding="utf-8",
            )
            argv = [
                "cleanup_review_diagnostic.py",
                "--plan",
                plan_path.as_posix(),
                "--mode",
                "warn",
                "--github-summary",
            ]

            with (
                patch.object(sys, "argv", argv),
                patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": summary_path.as_posix()}),
                patch("sys.stdout", new_callable=StringIO) as stdout,
            ):
                exit_code = main()

            self.assertEqual(exit_code, 0)
            self.assertIn("Cleanup review: allowed", stdout.getvalue())
            self.assertIn("Cleanup review: allowed", summary_path.read_text(encoding="utf-8"))

    def test_module_cli_renders_warn_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "cleanup_plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "summary": {
                            "cleanup_review_allowed": False,
                            "missing_source_artifacts": ["evidence_index"],
                            "candidate_count": 1,
                            "retained_count": 1,
                            "candidate_bytes": 50,
                        },
                        "source_status": {"evidence_index": {"present": False, "path": "index.json"}},
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cleanup_review_diagnostic",
                    "--plan",
                    plan_path.as_posix(),
                    "--mode",
                    "warn",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("Cleanup review: blocked", result.stdout)
            self.assertIn("Missing source artifacts: evidence_index", result.stdout)

    def test_module_cli_handles_invalid_json_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "cleanup_plan.json"
            plan_path.write_text("{not-json", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cleanup_review_diagnostic",
                    "--plan",
                    plan_path.as_posix(),
                    "--mode",
                    "blocked",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("Cleanup plan unreadable", result.stdout)
            self.assertNotIn("Traceback", result.stdout)
            self.assertEqual(result.stderr, "")

    def test_module_cli_handles_missing_plan_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "missing_cleanup_plan.json"

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cleanup_review_diagnostic",
                    "--plan",
                    plan_path.as_posix(),
                    "--mode",
                    "blocked",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("Cleanup plan missing", result.stdout)
            self.assertIn(plan_path.as_posix(), result.stdout)
            self.assertNotIn("Traceback", result.stdout)
            self.assertEqual(result.stderr, "")

    def test_main_appends_missing_plan_to_github_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "missing_cleanup_plan.json"
            summary_path = Path(tmpdir) / "summary.md"
            argv = [
                "cleanup_review_diagnostic.py",
                "--plan",
                plan_path.as_posix(),
                "--mode",
                "blocked",
                "--github-summary",
            ]

            with (
                patch.object(sys, "argv", argv),
                patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": summary_path.as_posix()}),
                patch("sys.stdout", new_callable=StringIO) as stdout,
            ):
                exit_code = main()

            summary_text = summary_path.read_text(encoding="utf-8")
            self.assertEqual(exit_code, 0)
            self.assertIn("Cleanup plan missing", stdout.getvalue())
            self.assertIn("Cleanup plan missing", summary_text)
            self.assertIn(plan_path.as_posix(), summary_text)
            self.assertNotIn("Traceback", summary_text)

    def test_main_appends_unreadable_plan_to_github_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "cleanup_plan.json"
            summary_path = Path(tmpdir) / "summary.md"
            plan_path.write_text("{not-json", encoding="utf-8")
            argv = [
                "cleanup_review_diagnostic.py",
                "--plan",
                plan_path.as_posix(),
                "--mode",
                "blocked",
                "--github-summary",
            ]

            with (
                patch.object(sys, "argv", argv),
                patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": summary_path.as_posix()}),
                patch("sys.stdout", new_callable=StringIO) as stdout,
            ):
                exit_code = main()

            summary_text = summary_path.read_text(encoding="utf-8")
            self.assertEqual(exit_code, 0)
            self.assertIn("Cleanup plan unreadable", stdout.getvalue())
            self.assertIn("Cleanup plan unreadable", summary_text)
            self.assertIn("Error:", summary_text)
            self.assertNotIn("Traceback", summary_text)


if __name__ == "__main__":
    unittest.main()
