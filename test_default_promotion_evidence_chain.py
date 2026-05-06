import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_default_promotion_evidence_chain import run_default_promotion_evidence_chain


class DefaultPromotionEvidenceChainTests(unittest.TestCase):
    def test_chain_links_artifacts_and_preserves_blocked_review(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def fake_validation(output_dir, timeout_seconds):
                path = Path(output_dir) / "validation_summary.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"passed": True}), encoding="utf-8")
                return {"passed": True, "summary_path": str(path)}

            def fake_audit(root, output):
                Path(output).write_text(json.dumps({"passed": True}), encoding="utf-8")
                return {"passed": True}

            with (
                patch("run_default_promotion_evidence_chain.run_validation_bundle", side_effect=fake_validation),
                patch(
                    "run_default_promotion_evidence_chain.audit_promotion_linkage",
                    side_effect=fake_audit,
                ),
                patch(
                    "run_default_promotion_evidence_chain.summarize_attnres_repeat_seed_decision",
                    return_value={"record_type": "attnres_repeat_seed_decision", "promote_default": False},
                ),
            ):
                summary = run_default_promotion_evidence_chain(output_dir=root / "chain")

            self.assertTrue(summary["chain_passed"])
            self.assertFalse(summary["review_allowed"])
            self.assertEqual(
                summary["preflight_blockers"],
                ["repeat_seed_evidence_does_not_support_default_promotion"],
            )
            for path in summary["artifacts"].values():
                self.assertTrue(Path(path).exists())

    def test_chain_fails_when_validation_or_audit_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            def fake_validation(output_dir, timeout_seconds):
                path = Path(output_dir) / "validation_summary.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"passed": False}), encoding="utf-8")
                return {"passed": False, "summary_path": str(path)}

            def fake_audit(root, output):
                Path(output).write_text(json.dumps({"passed": False}), encoding="utf-8")
                return {"passed": False}

            with (
                patch("run_default_promotion_evidence_chain.run_validation_bundle", side_effect=fake_validation),
                patch(
                    "run_default_promotion_evidence_chain.audit_promotion_linkage",
                    side_effect=fake_audit,
                ),
                patch(
                    "run_default_promotion_evidence_chain.summarize_attnres_repeat_seed_decision",
                    return_value={"record_type": "attnres_repeat_seed_decision", "promote_default": True},
                ),
            ):
                summary = run_default_promotion_evidence_chain(output_dir=root / "chain")

            self.assertFalse(summary["chain_passed"])
            self.assertEqual(
                summary["command_failures"],
                ["overlay_model_scope_validation", "promotion_linkage_audit"],
            )


if __name__ == "__main__":
    unittest.main()
