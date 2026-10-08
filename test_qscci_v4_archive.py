import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import qscci_experiment
from qscci_experiment import QSCCIError, artifact_digest, canonical_json
from run_qscci import verify_v4_archive


ARCHIVE_FIXTURE = Path(__file__).resolve().parent / "artifacts" / "method-dev" / "qscci-v4" / "run-001"
ARTIFACT_SHA256 = "043f5b395549f383b21f39c5fdbd16572ea78b763b423a0314d2d48d378dbad6"
MANIFEST_SHA256 = "dd0cc08b5f3e3833b28a2ada70f0f78fe4d86ac099cc03fbdc38afb90538e991"
COMMIT_SHA256 = "c16438aeaeceee03d638f2eea6a3f5c14727937590160cdf6ebf63e976189c20"
MEMBERS = ("qscci.json", "manifest.json", "COMMIT.json")


class TestQSCCIV4PortableArchive(unittest.TestCase):
    def setUp(self):
        if not ARCHIVE_FIXTURE.is_dir():
            self.skipTest("the frozen published v4 archive fixture is unavailable")
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.archive = self.root / "published-result"
        shutil.copytree(ARCHIVE_FIXTURE, self.archive)

    def tearDown(self):
        self.temporary.cleanup()

    def _rewrite_json(self, member, value):
        (self.archive / member).write_bytes(canonical_json(value))

    def _coherently_reseal(self, mutate):
        artifact = json.loads((self.archive / "qscci.json").read_bytes())
        mutate(artifact)
        artifact["artifact_digest"] = artifact_digest(artifact)
        self._rewrite_json("qscci.json", artifact)
        artifact_sha = hashlib.sha256((self.archive / "qscci.json").read_bytes()).hexdigest()

        manifest = json.loads((self.archive / "manifest.json").read_bytes())
        manifest["artifact_sha256"] = artifact_sha
        manifest["artifact_digest"] = artifact["artifact_digest"]
        self._rewrite_json("manifest.json", manifest)
        manifest_sha = hashlib.sha256((self.archive / "manifest.json").read_bytes()).hexdigest()

        receipt = json.loads((self.archive / "COMMIT.json").read_bytes())
        receipt["artifact_sha256"] = artifact_sha
        receipt["manifest_sha256"] = manifest_sha
        self._rewrite_json("COMMIT.json", receipt)

    def test_exact_frozen_archive_returns_custody_only_status_and_roots(self):
        result = verify_v4_archive(self.archive)
        self.assertEqual(
            result,
            {
                "status": "ARCHIVED_QSCCI_V4_VERIFIED",
                "protocol_id": "CHELATEDAI-QSCCI-v4",
                "disposition": "DOES_NOT_SURVIVE_SMALL_LABEL_ORACLE_FIXTURE",
                "scientific_claim_status": "UNCONFIRMED",
                "novelty_claim_status": "UNCONFIRMED",
                "artifact_sha256": ARTIFACT_SHA256,
                "manifest_sha256": MANIFEST_SHA256,
                "commit_sha256": COMMIT_SHA256,
                "artifact_digest": "fe75a63bb2e2ef14d911c8be457f01044d329377c8982ccf598a7a19ac71d1e2",
                "custody_only": True,
                "reexecution_verified": False,
                "current_target_verified": False,
                "live_service_lifecycle_verified": False,
                "pinned_model_sae_cache_verified": False,
            },
        )

    def test_archive_is_location_independent_after_ordinary_directory_move(self):
        destination = self.root / "nested" / "renamed-custody-copy"
        destination.parent.mkdir()
        shutil.move(self.archive, destination)
        result = verify_v4_archive(destination)
        self.assertEqual(result["artifact_sha256"], ARTIFACT_SHA256)

    def test_hidden_stage_and_quarantine_named_roots_are_rejected(self):
        for name in (".hidden", ".result.stage-123", ".result.quarantine-123", "result.quarantine-123"):
            with self.subTest(name=name):
                target = self.root / name
                shutil.copytree(self.archive, target)
                with self.assertRaises(QSCCIError):
                    verify_v4_archive(target)

    def test_missing_extra_and_renamed_members_are_rejected(self):
        mutations = {
            "missing": lambda path: (path / "manifest.json").unlink(),
            "extra": lambda path: (path / "notes.txt").write_text("not admitted", encoding="utf-8"),
            "renamed": lambda path: (path / "COMMIT.json").rename(path / "commit.json"),
            "directory_member": lambda path: ((path / "COMMIT.json").unlink(), (path / "COMMIT.json").mkdir()),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                target = self.root / f"case-{name}"
                shutil.copytree(self.archive, target)
                mutate(target)
                with self.assertRaises(QSCCIError):
                    verify_v4_archive(target)

    @unittest.skipIf(os.name == "nt", "POSIX symlink semantics")
    def test_symlink_root_and_each_symlink_member_are_rejected(self):
        root_link = self.root / "linked-result"
        root_link.symlink_to(self.archive, target_is_directory=True)
        with self.assertRaises(QSCCIError):
            verify_v4_archive(root_link)
        for member in MEMBERS:
            with self.subTest(member=member):
                target = self.root / f"linked-{member.replace('.', '-')}"
                shutil.copytree(self.archive, target)
                external = self.root / f"external-{member}"
                shutil.copy2(target / member, external)
                (target / member).unlink()
                (target / member).symlink_to(external)
                with self.assertRaises(QSCCIError):
                    verify_v4_archive(target)

    @unittest.skipUnless(os.name == "nt", "Windows junction/reparse semantics")
    def test_windows_junction_root_and_member_are_rejected(self):
        root_junction = self.root / "junction-result"
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(root_junction), str(self.archive)],
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            self.skipTest("junction creation unavailable")
        with self.assertRaises(QSCCIError):
            verify_v4_archive(root_junction)

        target = self.root / "junction-member-case"
        shutil.copytree(self.archive, target)
        (target / "COMMIT.json").unlink()
        external = self.root / "external-directory"
        external.mkdir()
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(target / "COMMIT.json"), str(external)],
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            self.skipTest("member junction creation unavailable")
        with self.assertRaises(QSCCIError):
            verify_v4_archive(target)

    def test_noncanonical_duplicate_key_bom_and_truncation_are_rejected(self):
        mutations = {
            "trailing-whitespace": lambda raw: raw + b" ",
            "bom": lambda raw: b"\xef\xbb\xbf" + raw,
            "duplicate-key": lambda raw: raw[:-2] + b',"status":"COMPLETE"}\n',
            "truncated": lambda raw: raw[: len(raw) // 2],
        }
        original = (self.archive / "qscci.json").read_bytes()
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                target = self.root / f"json-{name}"
                shutil.copytree(self.archive, target)
                (target / "qscci.json").write_bytes(mutate(original))
                with self.assertRaises(QSCCIError):
                    verify_v4_archive(target)

    def test_byte_tamper_and_nonfinite_json_are_rejected(self):
        tampered = bytearray((self.archive / "qscci.json").read_bytes())
        offset = tampered.index(b'"COMPLETE"') + 1
        tampered[offset] = ord("X")
        (self.archive / "qscci.json").write_bytes(tampered)
        with self.assertRaises(QSCCIError):
            verify_v4_archive(self.archive)

        shutil.rmtree(self.archive)
        shutil.copytree(ARCHIVE_FIXTURE, self.archive)
        artifact = json.loads((self.archive / "qscci.json").read_bytes())
        artifact["resources"]["wall_seconds"] = float("nan")
        (self.archive / "qscci.json").write_text(
            json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(QSCCIError):
            verify_v4_archive(self.archive)

    def test_coherent_reseal_cannot_replace_pinned_archive_roots(self):
        self._coherently_reseal(
            lambda artifact: artifact["resources"].__setitem__(
                "wall_seconds", artifact["resources"]["wall_seconds"] + 1.0
            )
        )
        with self.assertRaises(QSCCIError):
            verify_v4_archive(self.archive)

    def test_retrusted_reseal_still_rejects_cache_independent_gate_tamper(self):
        self._coherently_reseal(
            lambda artifact: artifact["gates"].__setitem__(
                "correct_positive_wrong_negative", False
            )
        )
        artifact = json.loads((self.archive / "qscci.json").read_bytes())
        roots = {
            "V4_ARCHIVE_ARTIFACT_SHA256": hashlib.sha256(
                (self.archive / "qscci.json").read_bytes()
            ).hexdigest(),
            "V4_ARCHIVE_MANIFEST_SHA256": hashlib.sha256(
                (self.archive / "manifest.json").read_bytes()
            ).hexdigest(),
            "V4_ARCHIVE_COMMIT_SHA256": hashlib.sha256(
                (self.archive / "COMMIT.json").read_bytes()
            ).hexdigest(),
            "V4_ARCHIVE_ARTIFACT_DIGEST": artifact["artifact_digest"],
        }
        with mock.patch.multiple("run_qscci", **roots), self.assertRaises(QSCCIError):
            verify_v4_archive(self.archive)

    def test_archive_verification_does_not_call_live_or_cache_dependent_verifier(self):
        with mock.patch("run_qscci.verify_artifact", side_effect=AssertionError("live/cache verifier called")):
            result = verify_v4_archive(self.archive)
        self.assertIs(result["custody_only"], True)
        self.assertIs(result["pinned_model_sae_cache_verified"], False)

    def test_live_and_archive_wrappers_hardcode_opposite_sae_provenance_modes(self):
        sentinel = {
            "verified": True,
            "protocol_id": "CHELATEDAI-QSCCI-v4",
            "disposition": "DOES_NOT_SURVIVE_SMALL_LABEL_ORACLE_FIXTURE",
            "attested_model_leaves_reexecuted": False,
        }
        repo_root = Path(__file__).resolve().parent
        with mock.patch(
            "qscci_experiment._verify_artifact_impl", return_value=sentinel
        ) as implementation:
            self.assertIs(qscci_experiment.verify_artifact({}, root=repo_root), sentinel)
            implementation.assert_called_once_with(
                {}, root=repo_root, require_pinned_sae_provenance=True
            )
        with mock.patch(
            "qscci_experiment._verify_artifact_impl", return_value=sentinel
        ) as implementation:
            portable = qscci_experiment.verify_artifact_cache_independent({}, root=repo_root)
            implementation.assert_called_once_with(
                {}, root=repo_root, require_pinned_sae_provenance=False
            )
        self.assertNotIn("verified", portable)
        self.assertEqual(
            portable,
            {
                "status": "CACHE_INDEPENDENT_RETAINED_LEAF_SEMANTICS_VERIFIED",
                "protocol_id": "CHELATEDAI-QSCCI-v4",
                "disposition": "DOES_NOT_SURVIVE_SMALL_LABEL_ORACLE_FIXTURE",
                "retained_leaf_semantics_verified": True,
                "pinned_model_sae_cache_verified": False,
                "decoder_floating_replay_verified": False,
                "reexecution_verified": False,
                "attested_model_leaves_reexecuted": False,
            },
        )

    def test_cache_independent_public_api_never_emits_generic_verified_claim(self):
        artifact = json.loads((self.archive / "qscci.json").read_bytes())
        result = qscci_experiment.verify_artifact_cache_independent(
            artifact, root=Path(__file__).resolve().parent
        )
        self.assertEqual(
            result,
            {
                "status": "CACHE_INDEPENDENT_RETAINED_LEAF_SEMANTICS_VERIFIED",
                "protocol_id": "CHELATEDAI-QSCCI-v4",
                "disposition": "DOES_NOT_SURVIVE_SMALL_LABEL_ORACLE_FIXTURE",
                "retained_leaf_semantics_verified": True,
                "pinned_model_sae_cache_verified": False,
                "decoder_floating_replay_verified": False,
                "reexecution_verified": False,
                "attested_model_leaves_reexecuted": False,
            },
        )

    def test_blocked_torch_import_fails_live_but_archive_remains_portable(self):
        artifact = json.loads((self.archive / "qscci.json").read_bytes())
        repo_root = Path(__file__).resolve().parent
        with mock.patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(QSCCIError, "Torch is required"):
                qscci_experiment.verify_artifact(artifact, root=repo_root)
            result = verify_v4_archive(self.archive)
        self.assertEqual(result["status"], "ARCHIVED_QSCCI_V4_VERIFIED")
        self.assertIs(result["pinned_model_sae_cache_verified"], False)

    def test_missing_sae_cache_fails_live_verifier_but_not_archive_verifier(self):
        artifact = json.loads((self.archive / "qscci.json").read_bytes())
        repo_root = Path(__file__).resolve().parent
        with mock.patch(
            "huggingface_hub.hf_hub_download",
            side_effect=FileNotFoundError("synthetic absent cache"),
        ):
            with self.assertRaisesRegex(QSCCIError, "pinned SAE cache"):
                qscci_experiment.verify_artifact(artifact, root=repo_root)
            result = verify_v4_archive(self.archive)
        self.assertEqual(result["status"], "ARCHIVED_QSCCI_V4_VERIFIED")
        self.assertIs(result["pinned_model_sae_cache_verified"], False)

    def test_archive_cli_is_standalone_and_rejects_ambiguous_live_subcommand(self):
        completed = subprocess.run(
            [sys.executable, "run_qscci.py", "--verify-v4-archive", str(self.archive)],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "ARCHIVED_QSCCI_V4_VERIFIED")

        ambiguous = subprocess.run(
            [
                sys.executable,
                "run_qscci.py",
                "--verify-v4-archive",
                str(self.archive),
                "verify",
                "--result-dir",
                str(self.archive),
            ],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(ambiguous.returncode, 0)


if __name__ == "__main__":
    unittest.main()
