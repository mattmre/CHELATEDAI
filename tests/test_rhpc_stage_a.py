from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from egv.canonical import canonical_bytes, digest_for
from rhpc.admission import admit_official_spark_run, verify_predecessor_completion
from rhpc.artifacts import MANIFEST_NAME, RESULT_NAME, publish_stage_a_artifact, verify_stage_a_artifact
from rhpc.errors import RHPCAdmissionError, RHPCConfigurationError, RHPCResultError
from rhpc.stage_a import PASS_STATUS, StageAConfig, run_stage_a, validate_stage_a_result


ZERO_SHA = "0" * 64
FROZEN_PROTOCOL_SHA256 = "fa556285107fd74a202bfbb7140b24c4d29c0bc171a02615068e4ef8834b88d7"
FROZEN_DEVELOPMENT_RESULTS = {
    7: "ccd77924af2b53fb2b1a53bb95b45c7bd78e3958ab44f40d2f87dcab93efb9a6",
    11: "b99e0e1bdf4c34926b26393ab5945f24f12a4464b15c1e735d848cbc3a07abac",
}


def _closed_predecessor_completion():
    verification = {
        "valid": True,
        "replay_type": "public cryptographic decision replay",
        "decisions": {},
        "recorded_decisions": {},
        "receipt_count": 1,
        "event_count": 1,
        "candidate_count": 0,
        "dependency_count": 0,
        "public_receipt_head_digest": "a" * 64,
        "public_event_head_digest": "b" * 64,
        "terminal_seal_verified": True,
        "limitations": ["constructed test report"],
    }
    return {
        "schema_version": "rhpc-predecessor-completion-v1",
        "projection_tree_sha256": "c" * 64,
        "public_key_sha256": "d" * 64,
        "protocol_digest": "e" * 64,
        "evaluator_digest": None,
        "verification": verification,
        "verification_digest": digest_for(verification),
    }


class StageAMechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = StageAConfig(run_id=7, trials=16)
        cls.result = run_stage_a(cls.config)

    def test_full_mechanism_survives_and_controls_are_load_bearing(self) -> None:
        result = self.result
        self.assertEqual(PASS_STATUS, result["status"])
        self.assertEqual(1.0, result["route_endpoints"]["full_recurrent"]["path_accuracy"])
        self.assertEqual(2.0, result["route_endpoints"]["full_recurrent"]["mean_lock_iterations"])
        self.assertGreaterEqual(
            result["route_endpoints"]["full_recurrent"]["mean_phase_residual_attenuation"],
            0.50,
        )
        self.assertEqual(0.0, result["controls"]["smoothing"]["mean_phase_residual_attenuation"])
        self.assertGreaterEqual(result["matrix_summary"]["mean_no_rotation_relative_error"], 0.01)
        self.assertGreaterEqual(result["matrix_summary"]["mean_no_residual_relative_error"], 0.001)
        self.assertGreaterEqual(
            result["route_endpoints"]["full_recurrent"]["path_accuracy"]
            - result["route_endpoints"]["one_shot"]["path_accuracy"],
            0.25,
        )
        self.assertGreaterEqual(
            result["route_endpoints"]["full_recurrent"]["path_accuracy"]
            - result["route_endpoints"]["orderless_recurrent"]["path_accuracy"],
            0.25,
        )

    def test_standing_cycle_is_rejected_and_dead_stability_does_not_false_lock(self) -> None:
        self.assertEqual("CYCLE", self.result["controls"]["injected_cycle"]["state"])
        self.assertLessEqual(self.result["controls"]["dead_lock"]["lock_coverage"], 0.01)
        for trial in self.result["trials"]:
            dead = trial["dead"]
            if dead["final_path"] != trial["true_path"]:
                self.assertNotEqual("LOCKED", dead["state"])

    def test_exact_regeneration_is_deterministic(self) -> None:
        second = run_stage_a(self.config)
        self.assertEqual(canonical_bytes(self.result), canonical_bytes(second))
        validate_stage_a_result(copy.deepcopy(self.result), regenerate=True)

    def test_result_unknown_field_and_contradictory_status_are_rejected(self) -> None:
        unknown = copy.deepcopy(self.result)
        unknown["unexpected"] = True
        with self.assertRaises(RHPCResultError):
            validate_stage_a_result(unknown)
        contradictory = copy.deepcopy(self.result)
        contradictory["status"] = "DOES_NOT_SURVIVE_CONSTRUCTED_MECHANISM_GATES"
        with self.assertRaises(RHPCResultError):
            validate_stage_a_result(contradictory)

    def test_configuration_is_closed_and_official_matrix_is_exact(self) -> None:
        self.assertTrue(StageAConfig(7).official)
        self.assertTrue(StageAConfig(11).official)
        self.assertFalse(StageAConfig(7, trials=16).official)
        value = self.config.to_dict()
        value["unknown"] = 1
        with self.assertRaises(RHPCConfigurationError):
            StageAConfig.from_mapping(value)
        with self.assertRaises(RHPCConfigurationError):
            StageAConfig(7, code_dim=1024)
        with self.assertRaises(RHPCConfigurationError):
            StageAConfig(7, alignment_threshold=0.80)


class StageAFrozenVectorTests(unittest.TestCase):
    def test_frozen_protocol_and_both_development_vectors(self) -> None:
        root = Path(__file__).resolve().parents[1]
        protocol = root / "docs" / "research" / "rhpc-stage-a-method-dev-protocol-2026-08-25.md"
        self.assertEqual(FROZEN_PROTOCOL_SHA256, hashlib.sha256(protocol.read_bytes()).hexdigest())
        for run_id, expected_digest in FROZEN_DEVELOPMENT_RESULTS.items():
            result = run_stage_a(StageAConfig(run_id))
            validate_stage_a_result(result, regenerate=True)
            self.assertEqual(expected_digest, digest_for(result))


class StageAArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.result = run_stage_a(StageAConfig(run_id=5, trials=8))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _publish(self, name: str = "artifact") -> Path:
        destination = self.root / name
        publish_stage_a_artifact(
            destination,
            self.result,
            execution_mode="DEVELOPMENT",
            protocol_sha256="1" * 64,
            source_git_sha="2" * 40,
            source_tree_clean=False,
        )
        return destination

    def test_atomic_artifact_round_trip(self) -> None:
        destination = self._publish()
        verified = verify_stage_a_artifact(destination)
        self.assertEqual(5, verified["result"]["run_id"])
        self.assertEqual("DEVELOPMENT", verified["manifest"]["execution_mode"])

    def test_extra_file_and_result_tamper_are_rejected(self) -> None:
        destination = self._publish()
        (destination / "extra.txt").write_text("extra", encoding="utf-8")
        with self.assertRaises(RHPCResultError):
            verify_stage_a_artifact(destination)
        (destination / "extra.txt").unlink()
        tampered = json.loads((destination / RESULT_NAME).read_text(encoding="utf-8"))
        tampered["status"] = "DOES_NOT_SURVIVE_CONSTRUCTED_MECHANISM_GATES"
        (destination / RESULT_NAME).write_bytes(canonical_bytes(tampered))
        with self.assertRaises(RHPCResultError):
            verify_stage_a_artifact(destination)

    def test_noncanonical_and_unknown_manifest_are_rejected(self) -> None:
        destination = self._publish()
        manifest_path = destination / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["unexpected"] = True
        manifest_path.write_bytes(canonical_bytes(manifest))
        with self.assertRaises(RHPCResultError):
            verify_stage_a_artifact(destination)

        second = self._publish("noncanonical")
        manifest = json.loads((second / MANIFEST_NAME).read_text(encoding="utf-8"))
        (second / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        with self.assertRaises(RHPCResultError):
            verify_stage_a_artifact(second)

    def test_hard_linked_member_is_rejected(self) -> None:
        destination = self._publish()
        result_path = destination / RESULT_NAME
        outside = self.root / "outside-result.json"
        shutil.copyfile(result_path, outside)
        result_path.unlink()
        os.link(outside, result_path)
        with self.assertRaises(RHPCResultError):
            verify_stage_a_artifact(destination)

    def test_official_artifact_requires_official_configuration_and_predecessor(self) -> None:
        with self.assertRaises(RHPCResultError):
            publish_stage_a_artifact(
                self.root / "official",
                self.result,
                execution_mode="SPARK_OFFICIAL",
                protocol_sha256="1" * 64,
                source_git_sha="2" * 40,
                source_tree_clean=True,
                predecessor_completion={"verification": {"valid": True, "terminal_seal_verified": True}},
            )

    def test_official_artifact_round_trip_enforces_closed_predecessor(self) -> None:
        result = run_stage_a(StageAConfig(7))
        destination = self.root / "official-round-trip"
        verified = publish_stage_a_artifact(
            destination,
            result,
            execution_mode="SPARK_OFFICIAL",
            protocol_sha256=FROZEN_PROTOCOL_SHA256,
            source_git_sha="2" * 40,
            source_tree_clean=True,
            predecessor_completion=_closed_predecessor_completion(),
        )
        self.assertEqual("SPARK_OFFICIAL", verified["manifest"]["execution_mode"])
        self.assertTrue(verified["manifest"]["predecessor_completion"]["verification"]["terminal_seal_verified"])


class StageAAdmissionTests(unittest.TestCase):
    def test_dirty_source_fails_before_predecessor_verification(self) -> None:
        with mock.patch(
            "rhpc.admission.source_checkout_state",
            return_value={"source_git_sha": "1" * 40, "source_tree_clean": False},
        ), mock.patch("rhpc.admission.verify_predecessor_completion") as predecessor:
            with self.assertRaises(RHPCAdmissionError):
                admit_official_spark_run(
                    Path("source"),
                    Path("projection"),
                    Path("public-key"),
                    protocol_digest="2" * 64,
                )
        predecessor.assert_not_called()

    def test_predecessor_verification_binds_terminal_report(self) -> None:
        class Report:
            terminal_seal_verified = True

            @staticmethod
            def to_dict():
                return {"valid": True, "terminal_seal_verified": True, "receipt_count": 2}

        with tempfile.TemporaryDirectory() as temporary:
            projection = Path(temporary) / "projection"
            projection.mkdir()
            (projection / "projection-manifest.json").write_text("{}", encoding="utf-8")
            public_key = Path(temporary) / "key.pem"
            public_key.write_bytes(b"public-key")
            with mock.patch("rhpc.admission.verify_public_projection", return_value=Report()) as verifier:
                completion = verify_predecessor_completion(
                    projection,
                    public_key,
                    protocol_digest="3" * 64,
                )
        self.assertEqual(digest_for(completion["verification"]), completion["verification_digest"])
        self.assertTrue(verifier.call_args.kwargs["require_terminal_seal"])

    def test_missing_terminal_seal_fails_closed(self) -> None:
        class Report:
            terminal_seal_verified = False

        with tempfile.TemporaryDirectory() as temporary:
            projection = Path(temporary) / "projection"
            projection.mkdir()
            (projection / "projection-manifest.json").write_text("{}", encoding="utf-8")
            public_key = Path(temporary) / "key.pem"
            public_key.write_bytes(b"public-key")
            with mock.patch("rhpc.admission.verify_public_projection", return_value=Report()):
                with self.assertRaises(RHPCAdmissionError):
                    verify_predecessor_completion(
                        projection,
                        public_key,
                        protocol_digest=ZERO_SHA,
                    )


if __name__ == "__main__":
    unittest.main()
