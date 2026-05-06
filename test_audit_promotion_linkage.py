import json
import tempfile
import unittest
from pathlib import Path

from audit_promotion_linkage import audit_promotion_linkage


class TestPromotionLinkageAudit(unittest.TestCase):
    def test_audit_passes_empty_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = audit_promotion_linkage(Path(tmpdir) / "missing")

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["report_count"], 0)

    def test_audit_blocks_overlay_report_missing_linkage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "experiment_runs"
            report_dir = root / "run" / "campaign"
            report_dir.mkdir(parents=True)
            (report_dir / "campaign_report.json").write_text(
                json.dumps(
                    {
                        "adaptive_overlay": {"record_type": "adaptive_overlay_report"},
                        "promotion_decision": {"promotion_ready": False},
                    }
                ),
                encoding="utf-8",
            )

            summary = audit_promotion_linkage(root)

        self.assertFalse(summary["passed"])
        self.assertEqual(summary["blocked_count"], 1)
        self.assertIn("missing_artifact_card_reference", summary["reports"][0]["blockers"])
        self.assertIn("missing_rollback_path", summary["reports"][0]["blockers"])

    def test_audit_accepts_linked_overlay_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "experiment_runs"
            report_dir = root / "run" / "campaign"
            report_dir.mkdir(parents=True)
            (report_dir / "campaign_report.json").write_text(
                json.dumps(
                    {
                        "adaptive_overlay_artifact_card": {"record_type": "adaptive_overlay_artifact_card"},
                        "promotion_decision": {
                            "promotion_ready": False,
                            "artifact_card_reference": {"path": "run/campaign/adaptive_overlay_artifact_card.json"},
                            "rollback_path": "run/campaign/shadow_policy_candidate.json",
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = audit_promotion_linkage(root)

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["blocked_count"], 0)


if __name__ == "__main__":
    unittest.main()
