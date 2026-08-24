from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import egv.cli as cli_module
from egv.canonical import canonical_json, digest_for
from egv.cli import build_parser, main
from egv.evaluation.dataset import EvaluationCorpus
from egv.experiment import FrozenHeldoutProtocol, HeldoutJournal, HeldoutVerifierServiceManifest
from egv.experiment import ordered_public_heldout_task_records
from egv.receipts import ReceiptSigner
from tests.test_egv_heldout_campaign import (
    _complete_envelopes,
    _protocol as complete_protocol,
    _restore_receipt,
)


class HeldoutCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.signer = ReceiptSigner(b"C" * 32)
        self.bindings = {
            name: digest_for({"binding": name})
            for name in FrozenHeldoutProtocol.REQUIRED_BINDINGS
        }
        self.bindings_path = self.root / "bindings.json"
        self.seed_path = self.root / "evaluator-seed.bin"
        self.public_key_path = self.root / "evaluator-public-key.bin"
        self.private_key_path = self.root / "evaluator-private-key.bin"
        self.protocol_path = self.root / "protocol.json"
        self.inputs_path = self.root / "trainer-inputs.json"
        self.sources_path = self.root / "trainer-sources.json"
        self.service_path = self.root / "service.json"
        self.bindings_path.write_bytes((canonical_json(self.bindings) + "\n").encode("utf-8"))
        self.seed_path.write_bytes(b"S" * 32)
        self.public_key_path.write_bytes(self.signer.public_key_raw)
        self.private_key_path.write_bytes(self.signer.private_key_raw)
        if os.name == "posix":
            self.seed_path.chmod(0o600)
            self.private_key_path.chmod(0o600)

    def tearDown(self):
        self.temporary.cleanup()

    def invoke(self, arguments, *, stdin_text=None):
        stdout = io.StringIO()
        stderr = io.StringIO()
        stdin = io.StringIO(stdin_text) if stdin_text is not None else None
        patches = [patch("sys.stdout", stdout), patch("sys.stderr", stderr)]
        if stdin is not None:
            patches.append(patch("sys.stdin", stdin))
        for context in patches:
            context.start()
        try:
            status = main(arguments)
        finally:
            for context in reversed(patches):
                context.stop()
        return status, stdout.getvalue(), stderr.getvalue()

    def prepare(self):
        status, output, error = self.invoke([
            "heldout", "prepare",
            "--campaign-id", "egv-campaign-1234567890abcdef",
            "--bindings", str(self.bindings_path),
            "--evaluator-seed", str(self.seed_path),
            "--evaluator-public-key", str(self.public_key_path),
            "--generation-profile-digest", digest_for("generation-profile"),
            "--schedule-seed", "91",
            "--bootstrap-seed", "92",
            "--protocol-output", str(self.protocol_path),
            "--trainer-inputs-output", str(self.inputs_path),
            "--trainer-sources-output", str(self.sources_path),
            "--json",
        ])
        self.assertEqual((status, error), (0, ""))
        return json.loads(output)

    def task_records(self):
        corpus = EvaluationCorpus.generate(secret_seed_file=self.seed_path)
        return ordered_public_heldout_task_records(corpus)

    def test_parser_exposes_only_bounded_heldout_workflow(self):
        parser = build_parser()
        for command in (
            "prepare", "freeze-evaluator-service", "evaluator-once", "run", "finalize"
        ):
            with self.subTest(command=command):
                with self.assertRaises(SystemExit):
                    parser.parse_args(["heldout", command])

    def test_prepare_and_freeze_service_emit_closed_content_bound_artifacts(self):
        report = self.prepare()
        self.assertEqual(report["coordinates"], 228)
        protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        self.assertEqual(protocol["protocol_digest"], report["protocol_digest"])
        trainer_inputs = json.loads(self.inputs_path.read_text(encoding="utf-8"))
        self.assertEqual(protocol["heldout_task_records"], trainer_inputs["tasks"])
        self.assertEqual(
            protocol["heldout_task_records_digest"],
            digest_for(protocol["heldout_task_records"]),
        )
        self.assertNotIn("hidden_spec", self.sources_path.read_text(encoding="utf-8"))

        status, output, error = self.invoke([
            "heldout", "freeze-evaluator-service",
            "--protocol", str(self.protocol_path),
            "--output", str(self.service_path),
            "--json",
        ])
        self.assertEqual((status, error), (0, ""))
        manifest = json.loads(self.service_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["manifest_digest"], json.loads(output)["service_manifest_digest"]
        )
        self.assertNotIn("endpoint", manifest)
        self.assertNotIn("command", manifest)

    def test_prepare_refuses_to_replace_any_frozen_output(self):
        self.protocol_path.write_text("reserved", encoding="utf-8")
        status, _output, error = self.invoke([
            "heldout", "prepare",
            "--campaign-id", "egv-campaign-1234567890abcdef",
            "--bindings", str(self.bindings_path),
            "--evaluator-seed", str(self.seed_path),
            "--evaluator-public-key", str(self.public_key_path),
            "--generation-profile-digest", digest_for("generation-profile"),
            "--schedule-seed", "91", "--bootstrap-seed", "92",
            "--protocol-output", str(self.protocol_path),
            "--trainer-inputs-output", str(self.inputs_path),
            "--trainer-sources-output", str(self.sources_path),
        ])
        self.assertEqual(status, 1)
        self.assertIn("refusing to overwrite", error)
        self.assertFalse(self.inputs_path.exists())

    def test_evaluator_once_fails_closed_without_real_verifier_callable(self):
        self.prepare()
        manifest = HeldoutVerifierServiceManifest.from_protocol(
            FrozenHeldoutProtocol.build(
                campaign_id="egv-campaign-1234567890abcdef",
                bindings=self.bindings,
                evaluator_public_key=self.signer.public_key,
                schedule_seed=91,
                bootstrap_seed=92,
                heldout_task_records=self.task_records(),
            )
        )
        self.service_path.write_bytes((canonical_json(manifest.to_dict()) + "\n").encode("utf-8"))
        status, _output, error = self.invoke([
            "heldout", "evaluator-once",
            "--protocol", str(self.protocol_path),
            "--service-manifest", str(self.service_path),
            "--private-key", str(self.private_key_path),
            "--state-root", str(self.root / "state"),
            "--observation-verifier", "missing_module:verify",
        ])
        self.assertEqual(status, 1)
        self.assertIn("not an approved integration", error)

    def test_run_and_finalize_fail_closed_without_integrations_or_complete_evidence(self):
        self.prepare()
        status, _output, error = self.invoke([
            "heldout", "run",
            "--protocol", str(self.protocol_path),
            "--journal", str(self.root / "journal"),
            "--operations", str(self.root / "operations"),
            "--runner", "missing_module:run",
            "--result-verifier", "missing_module:verify",
            "--reconciler", "missing_module:reconcile",
        ])
        self.assertEqual(status, 1)
        self.assertIn("not an approved integration", error)

        restoration = self.root / "restoration.json"
        restoration.write_bytes((canonical_json({"not": "a receipt"}) + "\n").encode("utf-8"))
        private_output = self.root / "private-final.json"
        status, _output, error = self.invoke([
            "heldout", "finalize",
            "--protocol", str(self.protocol_path),
            "--journal", str(self.root / "journal"),
            "--restoration-receipt", str(restoration),
            "--private-output", str(private_output),
        ])
        self.assertEqual(status, 1)
        self.assertIn("requires all 228", error)
        self.assertFalse(private_output.exists())

    def test_prepare_is_transactional_and_rejects_output_aliases(self):
        original = cli_module._atomic_publish_new_json
        calls = 0

        def fail_second(path, value):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected publication failure")
            return original(path, value)

        arguments = [
            "heldout", "prepare",
            "--campaign-id", "egv-campaign-1234567890abcdef",
            "--bindings", str(self.bindings_path),
            "--evaluator-seed", str(self.seed_path),
            "--evaluator-public-key", str(self.public_key_path),
            "--generation-profile-digest", digest_for("generation-profile"),
            "--schedule-seed", "91", "--bootstrap-seed", "92",
            "--protocol-output", str(self.protocol_path),
            "--trainer-inputs-output", str(self.inputs_path),
            "--trainer-sources-output", str(self.sources_path),
        ]
        with patch.object(cli_module, "_atomic_publish_new_json", fail_second):
            status, _output, error = self.invoke(arguments)
        self.assertEqual(status, 1)
        self.assertIn("publication failure", error)
        self.assertFalse(self.protocol_path.exists())
        self.assertFalse(self.inputs_path.exists())
        self.assertFalse(self.sources_path.exists())

        alias_arguments = list(arguments)
        alias_arguments[alias_arguments.index(str(self.sources_path))] = str(self.inputs_path)
        status, _output, error = self.invoke(alias_arguments)
        self.assertEqual(status, 1)
        self.assertIn("distinct", error)

    def test_prepare_rollback_preserves_same_bytes_replacement_inode(self):
        original = cli_module._atomic_publish_new_json
        calls = 0
        replacement = self.root / "replacement.json"
        replacement_identity = None

        def replace_first_then_fail_second(path, value):
            nonlocal calls, replacement_identity
            calls += 1
            if calls == 2:
                replacement.write_bytes(self.protocol_path.read_bytes())
                replacement_identity = (replacement.stat().st_dev, replacement.stat().st_ino)
                os.replace(str(replacement), str(self.protocol_path))
                raise OSError("injected second-output failure")
            return original(path, value)

        arguments = [
            "heldout", "prepare",
            "--campaign-id", "egv-campaign-1234567890abcdef",
            "--bindings", str(self.bindings_path),
            "--evaluator-seed", str(self.seed_path),
            "--evaluator-public-key", str(self.public_key_path),
            "--generation-profile-digest", digest_for("generation-profile"),
            "--schedule-seed", "91", "--bootstrap-seed", "92",
            "--protocol-output", str(self.protocol_path),
            "--trainer-inputs-output", str(self.inputs_path),
            "--trainer-sources-output", str(self.sources_path),
        ]
        with patch.object(
            cli_module, "_atomic_publish_new_json", replace_first_then_fail_second
        ):
            status, _output, error = self.invoke(arguments)
        self.assertEqual(status, 1)
        self.assertIn("second-output failure", error)
        self.assertTrue(self.protocol_path.is_file())
        self.assertEqual(
            (self.protocol_path.stat().st_dev, self.protocol_path.stat().st_ino),
            replacement_identity,
        )
        self.assertFalse(self.inputs_path.exists())
        self.assertFalse(self.sources_path.exists())
        self.assertEqual(
            json.loads(self.protocol_path.read_text(encoding="utf-8"))["campaign_id"],
            "egv-campaign-1234567890abcdef",
        )

    def test_prepare_rollback_does_not_move_foreign_hardlink(self):
        original = cli_module._atomic_publish_new_json
        calls = 0
        foreign = self.root / "foreign.json"
        replacement_link = self.root / "foreign-link.json"
        foreign_identity = None

        def replace_first_with_hardlink_then_fail(path, value):
            nonlocal calls, foreign_identity
            calls += 1
            if calls == 2:
                foreign.write_bytes(self.protocol_path.read_bytes())
                foreign_identity = (foreign.stat().st_dev, foreign.stat().st_ino)
                os.link(str(foreign), str(replacement_link))
                os.replace(str(replacement_link), str(self.protocol_path))
                raise OSError("injected hardlink replacement failure")
            return original(path, value)

        arguments = [
            "heldout", "prepare",
            "--campaign-id", "egv-campaign-1234567890abcdef",
            "--bindings", str(self.bindings_path),
            "--evaluator-seed", str(self.seed_path),
            "--evaluator-public-key", str(self.public_key_path),
            "--generation-profile-digest", digest_for("generation-profile"),
            "--schedule-seed", "91", "--bootstrap-seed", "92",
            "--protocol-output", str(self.protocol_path),
            "--trainer-inputs-output", str(self.inputs_path),
            "--trainer-sources-output", str(self.sources_path),
        ]
        with patch.object(
            cli_module, "_atomic_publish_new_json", replace_first_with_hardlink_then_fail
        ):
            status, _output, error = self.invoke(arguments)
        self.assertEqual(status, 1)
        self.assertIn("hardlink replacement failure", error)
        self.assertTrue(self.protocol_path.is_file())
        self.assertEqual(
            (self.protocol_path.stat().st_dev, self.protocol_path.stat().st_ino),
            foreign_identity,
        )
        self.assertEqual(
            (foreign.stat().st_dev, foreign.stat().st_ino),
            foreign_identity,
        )

    def test_prepare_rollback_restores_raced_in_directory_without_replacement(self):
        original_publish = cli_module._atomic_publish_new_json
        original_replace = os.replace
        calls = 0
        injected = False
        directory_identity = None
        displaced_created = self.root / "displaced-created.json"

        def fail_second(path, value):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected directory race failure")
            return original_publish(path, value)

        def race_before_quarantine(source, destination):
            nonlocal injected, directory_identity
            source_path = Path(source)
            destination_path = Path(destination)
            if (
                not injected
                and source_path == self.protocol_path
                and destination_path.parent.name.startswith(".egv-rollback-quarantine-")
            ):
                injected = True
                original_replace(str(source_path), str(displaced_created))
                source_path.mkdir()
                directory_identity = (source_path.stat().st_dev, source_path.stat().st_ino)
            return original_replace(str(source_path), str(destination_path))

        arguments = [
            "heldout", "prepare",
            "--campaign-id", "egv-campaign-1234567890abcdef",
            "--bindings", str(self.bindings_path),
            "--evaluator-seed", str(self.seed_path),
            "--evaluator-public-key", str(self.public_key_path),
            "--generation-profile-digest", digest_for("generation-profile"),
            "--schedule-seed", "91", "--bootstrap-seed", "92",
            "--protocol-output", str(self.protocol_path),
            "--trainer-inputs-output", str(self.inputs_path),
            "--trainer-sources-output", str(self.sources_path),
        ]
        with patch.object(cli_module, "_atomic_publish_new_json", fail_second), patch.object(
            cli_module.os, "replace", race_before_quarantine
        ):
            status, _output, error = self.invoke(arguments)
        self.assertEqual(status, 1)
        self.assertIn("directory race failure", error)
        self.assertTrue(self.protocol_path.is_dir())
        self.assertEqual(
            (self.protocol_path.stat().st_dev, self.protocol_path.stat().st_ino),
            directory_identity,
        )
        self.assertTrue(displaced_created.is_file())

    def test_key_admission_rejects_hardlinks_symlinks_special_files_and_wrong_size(self):
        hardlink = self.root / "hardlinked-public-key.bin"
        try:
            os.link(self.public_key_path, hardlink)
        except OSError:
            hardlink = None
        base = [
            "heldout", "prepare", "--campaign-id", "egv-campaign-1234567890abcdef",
            "--bindings", str(self.bindings_path), "--evaluator-seed", str(self.seed_path),
            "--evaluator-public-key", str(hardlink or self.public_key_path),
            "--generation-profile-digest", digest_for("generation-profile"),
            "--schedule-seed", "91", "--bootstrap-seed", "92",
            "--protocol-output", str(self.protocol_path),
            "--trainer-inputs-output", str(self.inputs_path),
            "--trainer-sources-output", str(self.sources_path),
        ]
        if hardlink is not None:
            status, _output, error = self.invoke(base)
            self.assertEqual(status, 1)
            self.assertIn("regular non-symlink", error)
            hardlink.unlink()

        wrong_size = self.root / "wrong-size.bin"
        wrong_size.write_bytes(b"K" * 33)
        public_key_argument = str(hardlink or self.public_key_path)
        base[base.index(public_key_argument)] = str(wrong_size)
        status, _output, error = self.invoke(base)
        self.assertEqual(status, 1)
        self.assertIn("invalid bounded size", error)

        base[base.index(str(wrong_size))] = str(self.root)
        status, _output, error = self.invoke(base)
        self.assertEqual(status, 1)
        self.assertIn("regular non-symlink", error)

        symlink = self.root / "public-key-link.bin"
        try:
            symlink.symlink_to(self.public_key_path)
        except OSError:
            symlink = None
        current_key_argument = str(self.root)
        if symlink is not None:
            base[base.index(current_key_argument)] = str(symlink)
            current_key_argument = str(symlink)
            status, _output, error = self.invoke(base)
            self.assertEqual(status, 1)
            self.assertIn("regular non-symlink", error)

        real_parent = self.root / "real-key-parent"
        real_parent.mkdir()
        nested_key = real_parent / "key.bin"
        nested_key.write_bytes(self.signer.public_key_raw)
        alias_parent = self.root / "key-parent-alias"
        try:
            alias_parent.symlink_to(real_parent, target_is_directory=True)
        except OSError:
            alias_parent = None
        if alias_parent is not None:
            base[base.index(current_key_argument)] = str(alias_parent / "key.bin")
            status, _output, error = self.invoke(base)
            self.assertEqual(status, 1)
            self.assertIn("ancestor", error)

    def test_service_mismatch_and_bounded_stdin_fail_closed(self):
        self.prepare()
        protocol = complete_protocol(campaign_id="egv-campaign-fedcba0987654321")
        mismatch = HeldoutVerifierServiceManifest.from_protocol(protocol)
        self.service_path.write_bytes((canonical_json(mismatch.to_dict()) + "\n").encode("utf-8"))
        arguments = [
            "heldout", "evaluator-once", "--protocol", str(self.protocol_path),
            "--service-manifest", str(self.service_path),
            "--private-key", str(self.private_key_path), "--state-root", str(self.root / "state"),
            "--observation-verifier", "fail-closed-v1",
        ]
        status, _output, error = self.invoke(arguments, stdin_text="{}")
        self.assertEqual(status, 1)
        self.assertIn("differs", error)

        own_protocol = FrozenHeldoutProtocol.build(
            campaign_id="egv-campaign-1234567890abcdef", bindings=self.bindings,
            evaluator_public_key=self.signer.public_key, schedule_seed=91, bootstrap_seed=92,
            heldout_task_records=self.task_records(),
        )
        service = HeldoutVerifierServiceManifest.from_protocol(own_protocol)
        self.service_path.write_bytes((canonical_json(service.to_dict()) + "\n").encode("utf-8"))
        status, _output, error = self.invoke(
            arguments, stdin_text="x" * (cli_module.HELDOUT_CLI_REQUEST_LIMIT + 1)
        )
        self.assertEqual(status, 1)
        self.assertIn("bounded input limit", error)

    def test_runtime_integration_error_is_sanitized(self):
        self.prepare()
        for index, failure in enumerate((
            RuntimeError(r"C:\private\operator\secret"),
            ImportError(r"C:\private\plugin\secret"),
        )):
            with self.subTest(failure=type(failure).__name__):
                def private_failure(_coordinate, error=failure):
                    raise error

                arguments = [
                    "heldout", "run", "--protocol", str(self.protocol_path),
                    "--journal", str(self.root / ("journal-{}".format(index))),
                    "--operations", str(self.root / ("operations-{}".format(index))),
                    "--runner", "reviewed-test-runner", "--result-verifier", "fail-closed-v1",
                    "--reconciler", "fail-closed-v1",
                ]
                with patch.dict(
                    cli_module.HELDOUT_COORDINATE_RUNNERS,
                    {"reviewed-test-runner": private_failure},
                ):
                    status, _output, error = self.invoke(arguments)
                self.assertEqual(status, 1)
                self.assertIn("integration failed safely", error)
                self.assertNotIn("private", error)
                self.assertNotIn("secret", error)

    def test_exact_228_finalize_rejects_invalid_then_accepts_valid_restoration(self):
        protocol = complete_protocol()
        self.protocol_path.write_bytes((canonical_json(protocol.to_private_dict()) + "\n").encode("utf-8"))
        journal_path = self.root / "complete-journal"
        journal = HeldoutJournal(journal_path, protocol)
        for envelope in _complete_envelopes(protocol):
            journal.append(envelope)
        restoration = _restore_receipt(protocol)
        invalid = dict(restoration, signature="invalid")
        invalid_path = self.root / "invalid-restoration.json"
        valid_path = self.root / "valid-restoration.json"
        invalid_path.write_bytes((canonical_json(invalid) + "\n").encode("utf-8"))
        valid_path.write_bytes((canonical_json(restoration) + "\n").encode("utf-8"))
        private_output = self.root / "private-final.json"
        arguments = [
            "heldout", "finalize", "--protocol", str(self.protocol_path),
            "--journal", str(journal_path), "--restoration-receipt", str(invalid_path),
            "--private-output", str(private_output), "--json",
        ]
        status, _output, error = self.invoke(arguments)
        self.assertEqual(status, 1)
        self.assertIn("signature", error)
        self.assertFalse(private_output.exists())
        arguments[arguments.index(str(invalid_path))] = str(valid_path)
        status, output, error = self.invoke(arguments)
        self.assertEqual((status, error), (0, ""))
        self.assertEqual(json.loads(output)["signed_outcomes"], 228)
        final = json.loads(private_output.read_text(encoding="utf-8"))
        self.assertEqual(len(final["signed_result_envelopes"]), 228)

    @unittest.skipUnless(os.name == "posix", "POSIX permission semantics")
    def test_private_key_requires_private_posix_permissions(self):
        self.prepare()
        self.private_key_path.chmod(0o644)
        service = HeldoutVerifierServiceManifest.from_protocol(
            FrozenHeldoutProtocol.build(
                campaign_id="egv-campaign-1234567890abcdef", bindings=self.bindings,
                evaluator_public_key=self.signer.public_key, schedule_seed=91, bootstrap_seed=92,
                heldout_task_records=self.task_records(),
            )
        )
        self.service_path.write_bytes((canonical_json(service.to_dict()) + "\n").encode("utf-8"))
        status, _output, error = self.invoke([
            "heldout", "evaluator-once", "--protocol", str(self.protocol_path),
            "--service-manifest", str(self.service_path), "--private-key", str(self.private_key_path),
            "--state-root", str(self.root / "state"),
            "--observation-verifier", "fail-closed-v1",
        ], stdin_text="{}")
        self.assertEqual(status, 1)
        self.assertIn("permissions", error)


if __name__ == "__main__":
    unittest.main()
