import json
import tempfile
import unittest
from pathlib import Path

from default_promotion_preflight import evaluate_default_promotion_preflight


class DefaultPromotionPreflightTests(unittest.TestCase):
    def _write_json(self, root: Path, name: str, payload: dict) -> Path:
        path = root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_preflight_passes_only_when_all_evidence_supports_review(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            validation = self._write_json(root, "validation.json", {"passed": True})
            audit = self._write_json(root, "audit.json", {"passed": True})
            decision = self._write_json(root, "decision.json", {"promote_default": True})

            summary = evaluate_default_promotion_preflight(
                validation_summary=validation,
                linkage_audit=audit,
                attnres_decision=decision,
            )

            self.assertTrue(summary["review_allowed"])
            self.assertFalse(summary["default_change_allowed"])
            self.assertEqual(summary["blockers"], [])

    def test_preflight_blocks_missing_and_non_promoting_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            validation = self._write_json(root, "validation.json", {"passed": True})
            audit = self._write_json(root, "audit.json", {"passed": False})
            missing_decision = root / "missing-decision.json"

            summary = evaluate_default_promotion_preflight(
                validation_summary=validation,
                linkage_audit=audit,
                attnres_decision=missing_decision,
            )

            self.assertFalse(summary["review_allowed"])
            self.assertIn("promotion_linkage_audit_failed", summary["blockers"])
            self.assertIn("repeat_seed_decision_missing", summary["blockers"])
            self.assertFalse(summary["artifacts"]["repeat_seed_decision"]["present"])

    def test_preflight_writes_output_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            validation = self._write_json(root, "validation.json", {"passed": False})
            audit = self._write_json(root, "audit.json", {"passed": True})
            decision = self._write_json(root, "decision.json", {"promote_default": False})
            output = root / "preflight.json"

            summary = evaluate_default_promotion_preflight(
                validation_summary=validation,
                linkage_audit=audit,
                attnres_decision=decision,
                output=output,
            )

            self.assertTrue(output.exists())
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(written["record_type"], "default_promotion_preflight")
            self.assertEqual(summary["output"], str(output))


if __name__ == "__main__":
    unittest.main()
