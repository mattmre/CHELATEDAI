import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prime_ring_rb10_contract import (
    RB10ValidationError,
    validate_artifact_envelope,
)
from run_prime_ring_rb10_experiments import (
    BoundedRunnerError,
    MANIFEST_FILENAME,
    MAX_RUNNER_RSS_BYTES,
    STAGE_FILENAMES,
    run_stages,
    validate_run_manifest,
)


class TestRunPrimeRingRB10Experiments(unittest.TestCase):
    def test_explicit_stage_and_hard_limit_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(BoundedRunnerError):
                run_stages((), temporary)
            with self.assertRaises(BoundedRunnerError):
                run_stages(
                    ("a1",),
                    temporary,
                    max_rss_bytes=MAX_RUNNER_RSS_BYTES + 1,
                )

    @unittest.skipUnless(os.name == "nt", "Windows RSS measurement required")
    def test_one_stage_runs_in_a_fresh_bounded_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = run_stages(
                ("a1",),
                temporary,
                timeout_seconds=10.0,
            )

            self.assertTrue(validate_run_manifest(manifest, temporary))
            self.assertTrue(manifest["sequential_process_isolation"])
            self.assertEqual(manifest["execution_order"], ["a1"])
            stage = manifest["stages"][0]
            self.assertEqual(stage["status"], "COMPLETE")
            self.assertEqual(stage["exit_code"], 0)
            self.assertLessEqual(
                stage["process_peak_rss_bytes"],
                stage["rss_cap_bytes"],
            )
            if os.name == "nt":
                self.assertTrue(stage["process_peak_measurement_available"])
                self.assertGreater(stage["process_peak_rss_bytes"], 0)

            artifact_path = Path(temporary) / STAGE_FILENAMES["a1"]
            with artifact_path.open("r", encoding="utf-8") as handle:
                artifact = json.load(handle)
            self.assertTrue(validate_artifact_envelope(artifact))

            manifest_path = Path(temporary) / MANIFEST_FILENAME
            with manifest_path.open("r", encoding="utf-8") as handle:
                persisted_manifest = json.load(handle)
            self.assertTrue(validate_run_manifest(persisted_manifest, temporary))

            tampered = json.loads(json.dumps(persisted_manifest))
            tampered["stages"][0]["artifact_digest"] = "0" * 64
            with self.assertRaises(RB10ValidationError):
                validate_run_manifest(tampered, temporary)

            tampered = json.loads(json.dumps(persisted_manifest))
            tampered["stages"][0]["artifact_file"] = "wrong.json"
            with self.assertRaises(RB10ValidationError):
                validate_run_manifest(tampered, temporary)

    @unittest.skipUnless(os.name == "nt", "Windows RSS measurement required")
    def test_child_failure_writes_a_recovery_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "run_prime_ring_rb10_experiments._run_child_stage",
                side_effect=BoundedRunnerError("forced failure"),
            ):
                with self.assertRaises(BoundedRunnerError):
                    run_stages(("a1",), temporary)

            manifest_path = Path(temporary) / MANIFEST_FILENAME
            with manifest_path.open("r", encoding="utf-8") as handle:
                failure = json.load(handle)
            self.assertEqual(failure["status"], "FAILED")
            self.assertEqual(failure["failed_stage"], "a1")
            self.assertEqual(failure["completed_stages"], [])

    def test_existing_evidence_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / STAGE_FILENAMES["a1"]).write_text(
                "sentinel",
                encoding="utf-8",
            )
            with self.assertRaises(BoundedRunnerError):
                run_stages(("a1",), temporary)


if __name__ == "__main__":
    unittest.main()
