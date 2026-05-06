import json
import tempfile
import time
import unittest
from pathlib import Path

from plan_evidence_artifact_cleanup import plan_evidence_artifact_cleanup


class EvidenceArtifactCleanupPlanTests(unittest.TestCase):
    def _write_json(self, root: Path, relative: str, payload: dict) -> Path:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_plan_lists_candidates_without_deleting_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "experiment_runs"
            old_path = self._write_json(root, "evidence-index/old/evidence_index.json", {"record_type": "old"})
            time.sleep(0.01)
            latest_path = self._write_json(root, "evidence-index/latest/evidence_index.json", {"record_type": "latest"})

            plan = plan_evidence_artifact_cleanup(root=root, keep_latest=1)

            self.assertTrue(old_path.exists())
            self.assertTrue(latest_path.exists())
            self.assertEqual(plan["summary"]["candidate_count"], 1)
            self.assertEqual(plan["summary"]["retained_count"], 1)
            self.assertEqual(plan["candidates"][0]["path"], "experiment_runs/evidence-index/old/evidence_index.json")
            self.assertEqual(plan["retained"][0]["path"], "experiment_runs/evidence-index/latest/evidence_index.json")
            self.assertTrue(plan["dry_run"])

    def test_plan_writes_optional_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "experiment_runs"
            self._write_json(root, "chain/latest/evidence_chain_summary.json", {"chain_passed": True})
            output = Path(tmpdir) / "cleanup_plan.json"

            plan = plan_evidence_artifact_cleanup(root=root, output=output)

            self.assertTrue(output.exists())
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(written["record_type"], "evidence_artifact_cleanup_plan")
            self.assertEqual(written["summary"], plan["summary"])

    def test_plan_records_source_artifact_references(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "experiment_runs"
            evidence_index = root / "evidence-index" / "latest" / "evidence_index.json"
            freshness_audit = root / "evidence-index" / "latest" / "freshness_audit.json"
            self._write_json(root, "evidence-index/latest/evidence_index.json", {"record_type": "index"})

            plan = plan_evidence_artifact_cleanup(
                root=root,
                evidence_index=evidence_index,
                freshness_audit=freshness_audit,
            )

            self.assertEqual(plan["source_artifacts"]["evidence_index"], evidence_index.as_posix())
            self.assertEqual(plan["source_artifacts"]["freshness_audit"], freshness_audit.as_posix())
            self.assertTrue(plan["source_status"]["evidence_index"]["present"])
            self.assertFalse(plan["source_status"]["freshness_audit"]["present"])
            self.assertFalse(plan["summary"]["cleanup_review_allowed"])
            self.assertEqual(plan["summary"]["missing_source_artifacts"], ["freshness_audit"])

    def test_plan_allows_cleanup_review_when_sources_are_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "experiment_runs"
            evidence_index = self._write_json(root, "evidence-index/latest/evidence_index.json", {"record_type": "index"})
            freshness_audit = self._write_json(
                root,
                "evidence-index/latest/freshness_audit.json",
                {"record_type": "freshness"},
            )

            plan = plan_evidence_artifact_cleanup(
                root=root,
                evidence_index=evidence_index,
                freshness_audit=freshness_audit,
            )

            self.assertTrue(plan["summary"]["cleanup_review_allowed"])
            self.assertEqual(plan["summary"]["missing_source_artifacts"], [])


if __name__ == "__main__":
    unittest.main()
