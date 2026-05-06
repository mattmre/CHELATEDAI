import json
import tempfile
import unittest
from pathlib import Path

from cleanup_review_diagnostic import render_cleanup_review_diagnostic


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


if __name__ == "__main__":
    unittest.main()
