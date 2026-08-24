from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import time
import unittest

from egv.canonical import digest_for
from egv.experiment.heldout import CoordinateOperationStore, HeldoutProtocolError
from egv.experiment.remote import (
    EvaluatorVerification,
    HeldoutVerifierServiceManifest,
    RemoteHeldoutReconciler,
    RemoteHeldoutResultVerifier,
    build_heldout_verifier_command,
    run_heldout_verifier_once,
)
from tests.test_egv_heldout_campaign import SIGNER, _base_result, _protocol


class HeldoutRemoteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.protocol = _protocol()
        self.manifest = HeldoutVerifierServiceManifest.from_protocol(self.protocol)
        self.coordinate = self.protocol.coordinates[0]
        self.operations = CoordinateOperationStore(self.root / "trainer-operations", self.protocol)
        self.operation = self.operations.begin(self.coordinate.coordinate_id)
        self.receipt_root = digest_for({"receipts": self.coordinate.coordinate_id})
        self.ledger_head = digest_for({"ledger": self.coordinate.coordinate_id})

    def tearDown(self):
        self.temporary.cleanup()

    def command(self, action, observation=None):
        return build_heldout_verifier_command(
            self.protocol,
            self.manifest,
            action=action,
            coordinate_id=self.coordinate.coordinate_id,
            operation_state=self.operation,
            receipt_collection_root=self.receipt_root,
            ledger_head_digest=self.ledger_head,
            observation=observation,
        )

    def verifier(self, _coordinate, observation):
        # Deliberately derive the result from evaluator-owned fixtures, not
        # from any booleans supplied by the runner observation.
        self.assertEqual(observation, {"opaque_output_digest": digest_for("runner-output")})
        return EvaluatorVerification(
            result=_base_result(self.protocol, self.coordinate),
            receipt_collection_root=self.receipt_root,
            ledger_head_digest=self.ledger_head,
        )

    def executor(self, command, probe=None):
        return run_heldout_verifier_once(
            self.protocol,
            self.manifest,
            command,
            signer=SIGNER,
            state_root=self.root / "evaluator-state",
            observation_verifier=self.verifier,
            reconciliation_probe=probe,
        )

    def test_manifest_is_closed_and_exactly_protocol_bound(self):
        value = self.manifest.to_dict()
        self.assertEqual(HeldoutVerifierServiceManifest.load(value, self.protocol), self.manifest)
        with self.assertRaisesRegex(HeldoutProtocolError, "non-closed"):
            HeldoutVerifierServiceManifest.load(dict(value, endpoint="private-host"), self.protocol)
        changed = dict(value, adapter_digest=digest_for("substitution"))
        with self.assertRaisesRegex(HeldoutProtocolError, "differs"):
            HeldoutVerifierServiceManifest.load(changed, self.protocol)

    def test_every_remote_admission_rejects_in_memory_protocol_substitution(self):
        command = self.command("BEGIN")
        original = self.protocol.coordinates
        other_task = next(item.task_id for item in original if item.task_id != original[0].task_id)
        substituted = replace(original[0], task_id=other_task)
        object.__setattr__(self.protocol, "coordinates", (substituted,) + original[1:])

        with self.assertRaisesRegex(HeldoutProtocolError, "immutable admission"):
            self.operations.begin(self.coordinate.coordinate_id)
        with self.assertRaisesRegex(HeldoutProtocolError, "immutable admission"):
            build_heldout_verifier_command(
                self.protocol,
                self.manifest,
                action="BEGIN",
                coordinate_id=self.coordinate.coordinate_id,
                operation_state=self.operation,
                receipt_collection_root=self.receipt_root,
                ledger_head_digest=self.ledger_head,
            )
        with self.assertRaisesRegex(HeldoutProtocolError, "immutable admission"):
            run_heldout_verifier_once(
                self.protocol,
                self.manifest,
                command,
                signer=SIGNER,
                state_root=self.root / "evaluator-state",
                observation_verifier=self.verifier,
            )

    def test_command_rejects_impossible_operation_state_and_action_pair(self):
        impossible = dict(self.operation)
        impossible["state"] = "COMPLETED"
        impossible["state_digest"] = digest_for(
            {key: value for key, value in impossible.items() if key != "state_digest"}
        )
        with self.assertRaisesRegex(HeldoutProtocolError, "requires a signed envelope"):
            build_heldout_verifier_command(
                self.protocol,
                self.manifest,
                action="BEGIN",
                coordinate_id=self.coordinate.coordinate_id,
                operation_state=impossible,
                receipt_collection_root=self.receipt_root,
                ledger_head_digest=self.ledger_head,
            )

        completed = dict(self.operation)
        completed.update(
            state="COMPLETED",
            revision=2,
            previous_state_digest=self.operation["state_digest"],
            envelope_digest=digest_for("signed-envelope"),
        )
        completed["state_digest"] = digest_for(
            {key: value for key, value in completed.items() if key != "state_digest"}
        )
        with self.assertRaisesRegex(HeldoutProtocolError, "action and operation state disagree"):
            build_heldout_verifier_command(
                self.protocol,
                self.manifest,
                action="VERIFY",
                coordinate_id=self.coordinate.coordinate_id,
                operation_state=completed,
                receipt_collection_root=self.receipt_root,
                ledger_head_digest=self.ledger_head,
                observation={"opaque_output_digest": digest_for("runner-output")},
            )
        reconcile = build_heldout_verifier_command(
            self.protocol,
            self.manifest,
            action="RECONCILE",
            coordinate_id=self.coordinate.coordinate_id,
            operation_state=completed,
            receipt_collection_root=self.receipt_root,
            ledger_head_digest=self.ledger_head,
        )
        self.assertEqual(reconcile["operation_state"]["state"], "COMPLETED")

    def test_base_arm_command_projects_no_adapter_but_binds_base_model(self):
        command = self.command("BEGIN")
        self.assertIsNone(command["adapter_digest"])
        self.assertEqual(command["base_model_digest"], self.protocol.bindings["base_model_digest"])
        tampered = dict(command, adapter_digest=self.protocol.bindings["adapter_digest"])
        with self.assertRaisesRegex(HeldoutProtocolError, "adapter_digest binding mismatch"):
            RemoteHeldoutResultVerifier(self.protocol, self.manifest.to_dict(), lambda _: {}).execute(tampered)

    def test_begin_is_signed_and_exactly_idempotent(self):
        command = self.command("BEGIN")
        remote = RemoteHeldoutResultVerifier(
            self.protocol, self.manifest.to_dict(), lambda request: self.executor(request)
        )
        first = remote.execute(command)
        second = remote.execute(command)
        self.assertEqual(first, second)
        self.assertIsNone(first["result_envelope"])
        self.assertIsNone(first["reconciliation_envelope"])
        self.assertEqual(first["acknowledgement"]["request_id"], command["request_id"])

    def test_verify_signs_only_evaluator_recomputed_result_and_caches_response(self):
        self.executor(self.command("BEGIN"))
        observation = {"opaque_output_digest": digest_for("runner-output")}
        command = self.command("VERIFY", observation)
        remote = RemoteHeldoutResultVerifier(
            self.protocol, self.manifest.to_dict(), lambda request: self.executor(request)
        )
        first = remote.execute(command)
        second = remote.execute(command)
        self.assertEqual(first, second)
        self.assertEqual(first["result_envelope"]["result"], _base_result(self.protocol, self.coordinate))
        self.assertEqual(first["result_envelope"]["receipt_collection_root"], self.receipt_root)
        self.assertEqual(first["result_envelope"]["ledger_head_digest"], self.ledger_head)

    def test_concurrent_distinct_verify_cannot_fork_signed_outcomes(self):
        self.executor(self.command("BEGIN"))
        commands = [
            self.command(
                "VERIFY",
                {"opaque_output_digest": digest_for("runner-{}".format(tokens)), "tokens": tokens},
            )
            for tokens in (101, 202)
        ]
        start = threading.Barrier(2)

        def evaluator(_coordinate, observation):
            time.sleep(0.05)
            result = _base_result(self.protocol, self.coordinate)
            result["costs"] = dict(result["costs"], tokens=observation["tokens"])
            return EvaluatorVerification(
                result=result,
                receipt_collection_root=self.receipt_root,
                ledger_head_digest=self.ledger_head,
            )

        def execute(command):
            start.wait()
            try:
                return run_heldout_verifier_once(
                    self.protocol,
                    self.manifest,
                    command,
                    signer=SIGNER,
                    state_root=self.root / "evaluator-state",
                    observation_verifier=evaluator,
                )
            except HeldoutProtocolError as exc:
                return exc

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(execute, commands))
        accepted = [item for item in outcomes if isinstance(item, dict)]
        rejected = [item for item in outcomes if isinstance(item, HeldoutProtocolError)]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 1)
        self.assertRegex(str(rejected[0]), "already has an evaluator-signed result")
        winning_tokens = accepted[0]["result_envelope"]["result"]["costs"]["tokens"]
        winning_command = commands[(101, 202).index(winning_tokens)]
        cached = run_heldout_verifier_once(
            self.protocol,
            self.manifest,
            winning_command,
            signer=SIGNER,
            state_root=self.root / "evaluator-state",
            observation_verifier=evaluator,
        )
        self.assertEqual(cached, accepted[0])

    def test_verify_rejects_evaluator_roots_that_differ_from_frozen_command(self):
        self.executor(self.command("BEGIN"))

        def bad_verifier(_coordinate, _observation):
            return EvaluatorVerification(
                result=_base_result(self.protocol, self.coordinate),
                receipt_collection_root=digest_for("other-receipts"),
                ledger_head_digest=self.ledger_head,
            )

        command = self.command("VERIFY", {"opaque_output_digest": digest_for("runner-output")})
        with self.assertRaisesRegex(HeldoutProtocolError, "roots differ"):
            run_heldout_verifier_once(
                self.protocol,
                self.manifest,
                command,
                signer=SIGNER,
                state_root=self.root / "evaluator-state",
                observation_verifier=bad_verifier,
            )

    def test_reconcile_without_proof_is_signed_unknown(self):
        self.executor(self.command("BEGIN"))
        command = self.command("RECONCILE")
        remote = RemoteHeldoutReconciler(
            self.protocol,
            self.manifest.to_dict(),
            lambda request: self.executor(request, probe=lambda _coordinate, _state: "COMPLETED"),
        )
        response = remote.execute(command)
        self.assertEqual(response["reconciliation_envelope"]["decision"], "UNKNOWN")
        self.assertIsNone(response["result_envelope"])

    def test_reconcile_not_executed_requires_positive_evaluator_probe(self):
        self.executor(self.command("BEGIN"))
        command = self.command("RECONCILE")
        remote = RemoteHeldoutReconciler(
            self.protocol,
            self.manifest.to_dict(),
            lambda request: self.executor(request, probe=lambda _coordinate, _state: "NOT_EXECUTED"),
        )
        response = remote.execute(command)
        self.assertEqual(response["reconciliation_envelope"]["decision"], "NOT_EXECUTED")

    def test_reconcile_completed_returns_exact_cached_signed_result(self):
        self.executor(self.command("BEGIN"))
        self.executor(self.command("VERIFY", {"opaque_output_digest": digest_for("runner-output")}))
        command = self.command("RECONCILE")
        response = RemoteHeldoutReconciler(
            self.protocol, self.manifest.to_dict(), lambda request: self.executor(request)
        ).execute(command)
        self.assertEqual(response["reconciliation_envelope"]["decision"], "COMPLETED")
        self.assertEqual(
            response["reconciliation_envelope"]["result_envelope_digest"],
            digest_for(response["result_envelope"]),
        )

    def test_response_signature_and_exact_command_bindings_are_enforced(self):
        command = self.command("BEGIN")
        response = self.executor(command)
        response["acknowledgement"]["ledger_head_digest"] = digest_for("tampered")
        remote = RemoteHeldoutResultVerifier(self.protocol, self.manifest.to_dict(), lambda _: response)
        with self.assertRaisesRegex(HeldoutProtocolError, "ledger_head_digest mismatch"):
            remote.execute(command)


if __name__ == "__main__":
    unittest.main()
