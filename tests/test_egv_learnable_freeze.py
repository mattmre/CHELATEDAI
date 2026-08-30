from __future__ import annotations

import base64
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from egv.canonical import canonical_bytes, canonical_json, digest_for
from egv.evaluation.blind_solver import BLIND_SOLVER_VERSION
import egv.evaluation.blind_solver as blind_solver_module
from egv.evaluation.learnable import LearnableCorpusV2
from egv.evaluation.learnable_freeze import (
    CORPUS_VALIDATION_SCOPE,
    build_blind_solver_manifest,
    build_corpus_freeze_bundle,
    validate_blind_solver_manifest,
    validate_corpus_freeze_bundle,
)
from egv.receipts import ReceiptSigner, key_id_for_public_key


def _runtime_manifest() -> dict:
    manifest = {
        "schema_version": "egv-learnable-production-runtime-v2",
        "backend": "docker-enforced-v1",
        "image_ref": "example.invalid/evaluator@sha256:" + "1" * 64,
        "pinned_image_id": "sha256:" + "2" * 64,
        "candidate_contract": "pure-return-v1",
        "network": "none",
        "read_only_root": True,
        "cap_drop": "ALL",
        "no_new_privileges": True,
        "user": "65534:65534",
        "pids_limit": 64,
        "memory_limit": "128m",
        "tmpfs_size": "16m",
        "timeout_seconds": 2.0,
        "output_limit": 65536,
    }
    manifest["manifest_digest"] = digest_for(manifest)
    return manifest


def _resign(bundle: dict, signer: ReceiptSigner, **changes) -> dict:
    changed = deepcopy(bundle)
    changed.update(changes)
    core = {key: value for key, value in changed.items() if key not in {"bundle_digest", "signature"}}
    changed["bundle_digest"] = digest_for(core)
    changed["signature"] = signer.sign_bytes(
        canonical_bytes({**core, "bundle_digest": changed["bundle_digest"], "signature": None})
    )
    return changed


class CorpusFreezeBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.solver_source = Path(blind_solver_module.__file__).read_bytes()

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.seed = bytes(range(32))
        seed_path = self.root / "seed.bin"
        seed_path.write_bytes(self.seed)
        self.corpus = LearnableCorpusV2.generate(secret_seed_file=seed_path)
        self.public_manifest = self.corpus.public_manifest()
        self.private_manifest = self.corpus.private_manifest()
        self.runtime_manifest = _runtime_manifest()
        self.blind_manifest = build_blind_solver_manifest(
            self.solver_source,
            solver_version=BLIND_SOLVER_VERSION,
        )
        self.freeze_signer = ReceiptSigner.generate()
        self.controller_signer = ReceiptSigner.generate()
        self.evaluator_revision = digest_for("gate-a-evaluator-exact-revision")
        self.bundle = self._build()

    def _build(self, **overrides) -> dict:
        arguments = {
            "public_manifest": self.public_manifest,
            "private_manifest": self.private_manifest,
            "runtime_manifest": self.runtime_manifest,
            "blind_solver_manifest": self.blind_manifest,
            "blind_solver_source": self.solver_source,
            "blind_solver_version": BLIND_SOLVER_VERSION,
            "evaluator_revision": self.evaluator_revision,
            "controller_public_key": self.controller_signer.public_key,
            "freeze_signer": self.freeze_signer,
        }
        arguments.update(overrides)
        return build_corpus_freeze_bundle(**arguments)

    def _validate(self, record=None, **overrides) -> dict:
        arguments = {
            "public_manifest": self.public_manifest,
            "private_manifest": self.private_manifest,
            "runtime_manifest": self.runtime_manifest,
            "blind_solver_manifest": self.blind_manifest,
            "blind_solver_source": self.solver_source,
            "blind_solver_version": BLIND_SOLVER_VERSION,
            "evaluator_revision": self.evaluator_revision,
            "controller_public_key": self.controller_signer.public_key,
            "freeze_public_key": self.freeze_signer.public_key,
        }
        arguments.update(overrides)
        return validate_corpus_freeze_bundle(record or self.bundle, **arguments)

    def test_round_trip_pins_every_gate_a_binding(self):
        validated = self._validate()
        self.assertEqual(validated, self.bundle)
        self.assertIsNot(validated, self.bundle)
        self.assertEqual(validated["scope"], CORPUS_VALIDATION_SCOPE)
        self.assertFalse(validated["task_bearing_model_requests_permitted"])
        self.assertEqual(validated["public_manifest_digest"], self.public_manifest["manifest_digest"])
        self.assertEqual(validated["private_manifest_digest"], self.private_manifest["manifest_digest"])
        self.assertEqual(validated["runtime_manifest_digest"], self.runtime_manifest["manifest_digest"])
        self.assertEqual(validated["blind_solver_manifest_digest"], self.blind_manifest["manifest_digest"])
        self.assertEqual(validated["blind_solver_source_digest"], self.blind_manifest["source_digest"])
        self.assertEqual(validated["blind_solver_version"], BLIND_SOLVER_VERSION)
        self.assertEqual(validated["evaluator_revision"], self.evaluator_revision)
        self.assertEqual(validated["controller_key_id"], self.controller_signer.key_id)
        self.assertEqual(validated["freeze_signer_key_id"], self.freeze_signer.key_id)

    def test_evaluator_revision_must_be_an_exact_sha256_source_manifest_digest(self):
        with self.assertRaises(ValueError):
            self._build(evaluator_revision="reviewed-but-not-a-source-digest")
        with self.assertRaises(ValueError):
            self._validate(evaluator_revision="reviewed-but-not-a-source-digest")

    def test_bundle_never_serializes_private_key_material(self):
        serialized = canonical_json(self.bundle)
        for signer in (self.freeze_signer, self.controller_signer):
            private = signer.private_key_raw
            encoded = base64.urlsafe_b64encode(private).decode("ascii").rstrip("=")
            self.assertNotIn(private.hex(), serialized)
            self.assertNotIn(encoded, serialized)
        self.assertNotIn("private_key", serialized)
        self.assertNotIn("seed_digest", serialized)

    def test_blind_solver_manifest_is_closed_and_source_exact(self):
        validated = validate_blind_solver_manifest(
            self.blind_manifest,
            solver_source=self.solver_source,
            solver_version=BLIND_SOLVER_VERSION,
        )
        self.assertEqual(validated, self.blind_manifest)
        for mutation in (
            {**self.blind_manifest, "extra": True},
            {key: value for key, value in self.blind_manifest.items() if key != "source_digest"},
        ):
            with self.assertRaisesRegex(ValueError, "closed record"):
                validate_blind_solver_manifest(
                    mutation,
                    solver_source=self.solver_source,
                    solver_version=BLIND_SOLVER_VERSION,
                )
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_blind_solver_manifest(
                self.blind_manifest,
                solver_source=self.solver_source + b"\n",
                solver_version=BLIND_SOLVER_VERSION,
            )

    def test_rejects_extra_missing_and_tampered_bundle_fields(self):
        extra = {**self.bundle, "unexpected": "field"}
        missing = {key: value for key, value in self.bundle.items() if key != "scope"}
        for label, mutation in (("extra", extra), ("missing", missing)):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "closed record"):
                self._validate(mutation)

        mutations = {
            "public_manifest_digest": "0" * 64,
            "private_manifest_digest": "0" * 64,
            "runtime_manifest_digest": "0" * 64,
            "blind_solver_manifest_digest": "0" * 64,
            "blind_solver_source_digest": "0" * 64,
            "blind_solver_version": "forged-solver",
            "evaluator_revision": "forged-revision",
            "controller_key_id": "key_forged",
            "freeze_signer_key_id": "key_forged",
            "bundle_digest": "0" * 64,
        }
        for field, value in mutations.items():
            with self.subTest(field=field), self.assertRaises(ValueError):
                self._validate({**self.bundle, field: value})

    def test_rejects_forged_signature_and_wrong_freeze_authority(self):
        attacker = ReceiptSigner.generate()
        forged = _resign(self.bundle, attacker)
        with self.assertRaisesRegex(ValueError, "invalid Ed25519 signature"):
            self._validate(forged)
        with self.assertRaisesRegex(ValueError, "exact expected bindings"):
            self._validate(freeze_public_key=attacker.public_key)

    def test_rejects_resigned_task_bearing_scope(self):
        for changes in (
            {"scope": "GO_TASK_BEARING"},
            {"task_bearing_model_requests_permitted": True},
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "task-bearing scope"):
                self._validate(_resign(self.bundle, self.freeze_signer, **changes))

    def test_rejects_controller_key_and_key_id_rebinding(self):
        attacker_controller = ReceiptSigner.generate()
        rebound_key = _resign(
            self.bundle,
            self.freeze_signer,
            controller_public_key=base64.urlsafe_b64encode(attacker_controller.public_key_raw)
            .decode("ascii")
            .rstrip("="),
            controller_key_id=attacker_controller.key_id,
        )
        with self.assertRaisesRegex(ValueError, "exact expected bindings"):
            self._validate(rebound_key)
        with self.assertRaisesRegex(ValueError, "exact expected bindings"):
            self._validate(controller_public_key=attacker_controller.public_key)

    def test_rejects_valid_but_different_expected_manifests_and_revision(self):
        alternate_seed_path = self.root / "alternate-seed.bin"
        alternate_seed_path.write_bytes(bytes(reversed(range(32))))
        alternate_private = LearnableCorpusV2.generate(secret_seed_file=alternate_seed_path).private_manifest()

        alternate_runtime = deepcopy(self.runtime_manifest)
        alternate_runtime["pids_limit"] = 32
        alternate_runtime["manifest_digest"] = digest_for(
            {key: value for key, value in alternate_runtime.items() if key != "manifest_digest"}
        )
        alternate_source = self.solver_source + b"\n# alternate reviewed source\n"
        alternate_blind = build_blind_solver_manifest(
            alternate_source,
            solver_version=BLIND_SOLVER_VERSION,
        )
        alternatives = (
            {"private_manifest": alternate_private},
            {"runtime_manifest": alternate_runtime},
            {"blind_solver_manifest": alternate_blind, "blind_solver_source": alternate_source},
            {"evaluator_revision": digest_for("another-evaluator-revision")},
        )
        for arguments in alternatives:
            with (
                self.subTest(arguments=tuple(arguments)),
                self.assertRaisesRegex(
                    ValueError,
                    "exact expected bindings",
                ),
            ):
                self._validate(**arguments)

    def test_rejects_cross_bound_private_manifest_even_when_rehashed(self):
        rebound = deepcopy(self.private_manifest)
        rebound["public_manifest_digest"] = "0" * 64
        rebound["manifest_digest"] = digest_for(
            {key: value for key, value in rebound.items() if key != "manifest_digest"}
        )
        with self.assertRaisesRegex(ValueError, "not bound to the public"):
            self._build(private_manifest=rebound)

    def test_rejects_private_key_objects_for_controller_and_freeze_verification(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        private_controller = Ed25519PrivateKey.from_private_bytes(self.controller_signer.private_key_raw)
        with self.assertRaisesRegex(ValueError, "must not contain a private key"):
            self._build(controller_public_key=private_controller)
        private_freeze = Ed25519PrivateKey.from_private_bytes(self.freeze_signer.private_key_raw)
        with self.assertRaisesRegex(ValueError, "must not contain a private key"):
            self._validate(freeze_public_key=private_freeze)

    def test_controller_public_key_id_is_derived_not_caller_selected(self):
        raw = self.controller_signer.public_key_raw
        self.assertEqual(self.bundle["controller_key_id"], key_id_for_public_key(raw))


if __name__ == "__main__":
    unittest.main()
