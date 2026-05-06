import json
import tempfile
import unittest
from pathlib import Path

from generate_evidence_index import generate_evidence_index


class GenerateEvidenceIndexTests(unittest.TestCase):
    def _write_json(self, root: Path, relative: str, payload: dict) -> Path:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_index_links_expected_artifact_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "experiment_runs"
            self._write_json(root, "validation/latest/validation_summary.json", {"record_type": "validation", "passed": True})
            self._write_json(root, "audit/latest/promotion-linkage-audit.json", {"record_type": "audit", "passed": True})
            self._write_json(
                root,
                "decision/latest/attnres-repeat-seed-decision.json",
                {"record_type": "attnres_repeat_seed_decision", "decision": "no_default_change"},
            )
            self._write_json(
                root,
                "preflight/latest/preflight.json",
                {
                    "record_type": "default_promotion_preflight",
                    "review_allowed": False,
                    "blockers": ["repeat_seed_evidence_does_not_support_default_promotion"],
                },
            )
            self._write_json(
                root,
                "chain/latest/evidence_chain_summary.json",
                {"record_type": "default_promotion_evidence_chain", "chain_passed": True},
            )
            self._write_json(
                root,
                "campaign/latest/campaign_report.json",
                {"record_type": "model_scope_campaign_report", "promotion_decision": {"promotion_ready": False}},
            )
            self._write_json(
                root,
                "campaign/latest/adaptive_overlay_artifact_card.json",
                {"record_type": "adaptive_overlay_artifact_card"},
            )
            output = root / "evidence-index" / "latest" / "evidence_index.json"

            index = generate_evidence_index(root=root, output=output)

            self.assertTrue(output.exists())
            counts = index["summary"]["artifact_counts"]
            self.assertEqual(counts["validation_summaries"], 1)
            self.assertEqual(counts["promotion_linkage_audits"], 1)
            self.assertEqual(counts["attnres_repeat_seed_decisions"], 1)
            self.assertEqual(counts["default_promotion_preflights"], 1)
            self.assertEqual(counts["evidence_chain_summaries"], 1)
            self.assertEqual(counts["campaign_reports"], 1)
            self.assertEqual(counts["adaptive_overlay_artifact_cards"], 1)
            self.assertFalse(index["summary"]["latest_review_allowed"])
            self.assertEqual(
                index["summary"]["latest_preflight_blockers"],
                ["repeat_seed_evidence_does_not_support_default_promotion"],
            )
            self.assertTrue(index["summary"]["latest_chain_passed"])

    def test_index_handles_missing_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "missing"
            output = Path(tmpdir) / "index.json"

            index = generate_evidence_index(root=root, output=output)

            self.assertTrue(output.exists())
            self.assertEqual(sum(index["summary"]["artifact_counts"].values()), 0)
            self.assertIsNone(index["summary"]["latest_review_allowed"])


if __name__ == "__main__":
    unittest.main()
