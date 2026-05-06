import json
import tempfile
import unittest
from pathlib import Path

from audit_evidence_index_freshness import audit_evidence_index_freshness


class EvidenceIndexFreshnessAuditTests(unittest.TestCase):
    def test_audit_passes_when_indexed_paths_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "experiment_runs" / "validation" / "validation_summary.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text(json.dumps({"passed": True}), encoding="utf-8")
            index = root / "experiment_runs" / "evidence-index" / "latest" / "evidence_index.json"
            index.parent.mkdir(parents=True)
            index.write_text(
                json.dumps(
                    {
                        "artifacts": {
                            "validation_summaries": [
                                {"path": str(artifact)},
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            summary = audit_evidence_index_freshness(index=index)

            self.assertTrue(summary["passed"])
            self.assertEqual(summary["checked_path_count"], 1)
            self.assertEqual(summary["missing_paths"], [])

    def test_audit_blocks_missing_indexed_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            index = root / "evidence_index.json"
            index.write_text(
                json.dumps({"artifacts": {"campaign_reports": [{"path": "missing/campaign_report.json"}]}}),
                encoding="utf-8",
            )

            summary = audit_evidence_index_freshness(index=index)

            self.assertFalse(summary["passed"])
            self.assertIn("indexed_artifact_paths_missing", summary["blockers"])
            self.assertEqual(summary["missing_paths"], ["missing/campaign_report.json"])

    def test_audit_blocks_missing_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = audit_evidence_index_freshness(index=Path(tmpdir) / "missing.json")

        self.assertFalse(summary["passed"])
        self.assertIn("evidence_index_missing_or_unreadable", summary["blockers"])

    def test_audit_resolves_relative_paths_from_index_parent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            index_dir = root / "nested" / "index"
            index_dir.mkdir(parents=True)
            artifact = index_dir / "relative" / "artifact.json"
            artifact.parent.mkdir()
            artifact.write_text(json.dumps({"passed": True}), encoding="utf-8")
            index = index_dir / "evidence_index.json"
            index.write_text(
                json.dumps({"artifacts": {"validation_summaries": [{"path": "relative/artifact.json"}]}}),
                encoding="utf-8",
            )

            summary = audit_evidence_index_freshness(index=index)

            self.assertTrue(summary["passed"])
            self.assertEqual(summary["checked_path_count"], 1)


if __name__ == "__main__":
    unittest.main()
