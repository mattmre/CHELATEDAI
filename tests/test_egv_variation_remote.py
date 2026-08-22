from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from egv.canonical import GENESIS_HASH, canonical_bytes, canonical_json, digest_bytes, digest_for
from egv.campaign.trajectories import GenerationRequest, GenerationResponse, validate_accepted_response
from egv.evaluation.authority import AuthorityPolicy
from egv.evaluation.dataset import EvaluationCorpus
from egv.evaluation.sandbox import DockerSandboxConfig
from egv.ledger import EvidenceLedger
from egv.receipts import ReceiptSigner, receipt_hash
from egv.variation.errors import VariationConfigurationError, VariationDependencyError
from egv.variation.arms import arm_policy
from egv.variation.loop import VARIATION_PROTOCOL_DIGEST
from egv.variation.remote import (
    REMOTE_VARIATION_SERVICE_SCHEMA,
    RemoteControllerEvaluationGateway,
    RemoteEvaluatorServiceManifest,
    _durable_remote_response,
    build_remote_evaluator_service_manifest,
)


RESPONDER = r'''from __future__ import annotations
import json, os, sys
from egv.canonical import GENESIS_HASH, canonical_bytes, canonical_json, content_id, digest_bytes, digest_for
from egv.receipts import ReceiptSigner, receipt_hash

request = json.loads(sys.stdin.read())
capture = os.environ.get("EGV_REMOTE_TEST_CAPTURE")
if capture:
    open(capture, "w", encoding="utf-8").write(canonical_json(request))
signer = ReceiptSigner(bytes.fromhex("__PRIVATE_KEY__"))
common = {
    "campaign_id": request["campaign_id"], "run_id": request["run_id"],
    "task_id": request["task_id"], "candidate_id": request["candidate_id"],
    "candidate_artifact_digest": request["candidate_artifact_digest"],
    "protocol_digest": request["protocol_digest"], "policy_digest": request["policy_digest"],
    "arm_policy_digest": request["arm_policy_digest"],
    "evaluator_digest": request["service_manifest_digest"],
    "task_family": request["public_task_binding"]["family_id"],
    "normalized_public_locus": request["public_task_binding"]["public_locus"],
    "public_rule_id": request["public_task_binding"]["public_rule_id"],
}
sequence = request["receipt_sequence_start"]
previous = request["previous_receipt_hash"]
receipts = []
for kind, fields in (
    ("AUTHORITY", {"request_id": "request-authority-" + request["candidate_id"], "decision": "ALLOW"}),
    ("VERDICT", {"request_id": "request-verdict-" + request["candidate_id"], "decision": "PASS",
        "diagnostic_enum": "PASS", "resource_bucket": "UNDER_25", "exit_status_class": "SUCCESS",
        "input_digest": digest_for("evaluator-private"), "output_digest": digest_bytes(b"ok")}),
    ("EFFECT", {"request_id": "request-effect-" + request["candidate_id"], "decision": "ALLOW",
        "diagnostic_enum": "PASS", "normalized_action_hash": digest_for({"action":"execute_candidate","locus":request["declared_locus"]}),
        "sandbox_id": "sealed-sandbox", "started_at": "2026-08-22T00:00:00Z",
        "finished_at": "2026-08-22T00:00:01Z", "exit_status_class": "SUCCESS",
        "output_digest": digest_bytes(b"ok"), "environment_diff_digest": digest_for({})}),
):
    receipt = signer.sign_receipt({**common, "receipt_type": kind, **fields}, sequence=sequence,
        previous_receipt_hash=previous, idempotency_key=content_id("remote-test", {"request":request["request_digest"],"kind":kind}))
    receipts.append(receipt); previous = receipt_hash(receipt); sequence += 1
result = {
    "candidate_id": request["candidate_id"], "task_id": request["task_id"],
    "candidate_artifact_digest": request["candidate_artifact_digest"], "diagnostic_enum": "PASS",
    "resource_bucket": "UNDER_25", "disposition": "PROMOTED", "infrastructure_loss": False,
    "receipt_ids": [item["receipt_id"] for item in receipts], "output_digest": digest_bytes(b"ok"),
}
unsigned = {"schema_version":"egv-remote-variation-response-v1", "operation_digest":request["operation_digest"],
    "request_digest":request["request_digest"],
    "service_manifest_digest":request["service_manifest_digest"], "result":result, "receipts":receipts,
    "signing_key_id":signer.key_id}
mode = os.environ.get("EGV_REMOTE_TEST_MODE")
if mode == "stale-request": unsigned["request_digest"] = digest_for("stale")
response = {**unsigned, "signature":signer.sign_bytes(canonical_bytes(unsigned))}
if mode == "tamper-result": response["result"]["candidate_artifact_digest"] = digest_for("substitute")
print(canonical_json(response))
'''


class RemoteVariationGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="egv-remote-variation-")
        self.root = Path(self.temporary.name)
        self._old_pythonpath = os.environ.get("PYTHONPATH")
        repository_root = str(Path(__file__).resolve().parents[1])
        os.environ["PYTHONPATH"] = repository_root + (os.pathsep + self._old_pythonpath if self._old_pythonpath else "")
        self.signer = ReceiptSigner(b"V" * 32)
        self.key_path = self.root / "evaluator.pub"
        self.key_path.write_bytes(self.signer.public_key_raw)
        self.command = self.root / "remote-endpoint.py"
        self.command.write_text(RESPONDER.replace("__PRIVATE_KEY__", self.signer.private_key_raw.hex()), encoding="utf-8")
        self.task = {
            "template_id": "task-heldout-001",
            "family_id": "family-test",
            "split": "train",
            "ordinal": 1,
            "source_digest": digest_for("source"),
            "public_rule_id": "rule-test",
            "public_locus": "module:solve",
        }
        policy_digest = AuthorityPolicy.candidate_execution().digest
        unsigned = {
            "schema_version": REMOTE_VARIATION_SERVICE_SCHEMA,
            "campaign_id": "campaign-remote-test",
            "model_digest": digest_for("model"),
            "protocol_digest": VARIATION_PROTOCOL_DIGEST,
            "policy_digest": policy_digest,
            "data_manifest_digest": digest_for("data"),
            "task_manifest_digest": digest_for([self.task]),
            "task_bindings": [self.task],
            "evaluator_revision": "remote-evaluator-test-v1",
            "evaluator_digest": digest_for("remote-evaluator-test-v1"),
            "docker_image_digest": DockerSandboxConfig().pinned_image_id,
            "docker_config_digest": digest_for(dict(DockerSandboxConfig().__dict__)),
            "authority_policy_digest": policy_digest,
            "command_digest": hashlib.sha256(self.command.read_bytes()).hexdigest(),
            "evaluator_key_id": self.signer.key_id,
            "evaluator_public_key_digest": hashlib.sha256(self.key_path.read_bytes()).hexdigest(),
        }
        self.manifest_value = {**unsigned, "service_manifest_digest": digest_for(unsigned)}
        self.manifest_path = self.root / "service-manifest.json"
        self.manifest_path.write_text(canonical_json(self.manifest_value) + "\n", encoding="utf-8")
        self.ledger = EvidenceLedger(self.root / "ledger.sqlite", clock=lambda: "2026-08-22T00:00:00Z")

    def tearDown(self) -> None:
        self.ledger.close()
        os.environ.pop("EGV_REMOTE_TEST_CAPTURE", None)
        os.environ.pop("EGV_REMOTE_TEST_MODE", None)
        if self._old_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = self._old_pythonpath
        self.temporary.cleanup()

    def gateway(self) -> RemoteControllerEvaluationGateway:
        return RemoteControllerEvaluationGateway(
            ledger=self.ledger,
            manifest_path=self.manifest_path,
            public_key_path=self.key_path,
            command=self.command,
        )

    def register_commissioning_candidate(
        self, request: GenerationRequest, candidate_id: str, source: bytes
    ) -> None:
        self.ledger.create_campaign(
            request.campaign_id,
            protocol_hash=request.variation_protocol_digest,
            source_commit="commissioning-test",
            model_revision="test-model",
            data_manifest_hash=request.corpus_manifest_digest,
            evaluator_hash=self.manifest_value["service_manifest_digest"],
            policy_hash=AuthorityPolicy.candidate_execution().digest,
            seed_set=(0, 1),
        )
        self.ledger.create_run(
            request.run_id,
            campaign_id=request.campaign_id,
            arm=request.arm_id,
            task_id=request.task_id,
            seed=request.seed,
            parent_checkpoint=None,
            start_state="READY",
            host_role="spark_trainer",
            software_manifest_hash=digest_for("test-software"),
        )
        self.ledger.append_candidate(
            candidate_id,
            campaign_id=request.campaign_id,
            run_id=request.run_id,
            task_id=request.task_id,
            parent_candidate_id=None,
            mutation_family=request.task_family,
            patch_hash=digest_for(source.decode("utf-8")),
            requested_authority="EXECUTE_CANDIDATE",
            prompt_hash=digest_for("prompt"),
            model_hash=request.model_manifest_digest,
            metadata={"arm_id": request.arm_id, "candidate_artifact_digest": digest_bytes(source)},
        )

    def durable_request(self, label: str) -> dict:
        body = {
            "operation_digest": digest_for({"operation": label}),
            "receipt_sequence_start": 1,
            "previous_receipt_hash": GENESIS_HASH,
        }
        return {**body, "request_digest": digest_for(body)}

    def durable_builder(self, request: dict):
        def build() -> dict:
            receipt = self.signer.sign_receipt(
                {
                    "receipt_type": "AUTHORITY",
                    "campaign_id": "campaign-remote-test",
                    "run_id": "run-evaluation",
                    "task_id": self.task["template_id"],
                    "request_id": "request-authority-durable",
                    "candidate_id": "candidate-durable",
                    "decision": "DENY",
                },
                sequence=request["receipt_sequence_start"],
                previous_receipt_hash=request["previous_receipt_hash"],
                idempotency_key="durable:" + request["operation_digest"],
            )
            unsigned = {
                "schema_version": "egv-remote-variation-response-v1",
                "operation_digest": request["operation_digest"],
                "request_digest": request["request_digest"],
                "service_manifest_digest": self.manifest_value["service_manifest_digest"],
                "result": {},
                "receipts": [receipt],
                "signing_key_id": self.signer.key_id,
            }
            return {**unsigned, "signature": self.signer.sign_bytes(canonical_bytes(unsigned))}

        return build

    def test_remote_gateway_ingests_verified_chain_without_private_input_or_path(self) -> None:
        capture = self.root / "request.json"
        os.environ["EGV_REMOTE_TEST_CAPTURE"] = str(capture)
        result = self.gateway().evaluate(
            candidate_id="candidate-001",
            task_id=self.task["template_id"],
            source=b"def solve(value):\n    return value\n",
            opaque_input=None,
            requested_authority="EXECUTE_CANDIDATE",
            declared_locus=self.task["public_locus"],
            candidate_source_path=str(self.root / "private" / "candidate.py"),
        )
        self.assertEqual(result.disposition, "PROMOTED")
        self.assertEqual(len(self.ledger.receipts()), 3)
        replay = self.ledger.receipts()
        with self.assertRaises(Exception):
            self.ledger.ingest_receipts_atomic(replay, self.signer.public_key)
        self.assertEqual(self.ledger.receipts(), replay)
        request = json.loads(capture.read_text(encoding="utf-8"))
        serialized = canonical_json(request)
        self.assertNotIn("never", serialized)
        self.assertNotIn("send-this-private-value", serialized)
        self.assertNotIn("candidate.py", serialized)
        self.assertEqual(set(request), {
            "schema_version", "operation_digest", "request_digest", "service_manifest_digest", "campaign_id", "model_digest",
            "protocol_digest", "policy_digest", "data_manifest_digest", "task_manifest_digest",
            "run_id", "arm_policy_digest",
            "evaluator_digest", "docker_image_digest", "candidate_id", "task_id", "public_task_binding",
            "candidate_artifact_digest", "candidate_source_b64", "requested_authority", "declared_locus",
            "receipt_sequence_start", "previous_receipt_hash",
        })

    def test_remote_promotions_bind_commissioning_run_and_distinct_arm_policy_for_b_and_d(self) -> None:
        for index, arm_id in enumerate(("B", "D")):
            with self.subTest(arm_id=arm_id):
                request = GenerationRequest.build(
                    campaign_id=self.manifest_value["campaign_id"],
                    task_record=self.task,
                    corpus_manifest_digest=self.manifest_value["data_manifest_digest"],
                    arm_id=arm_id,
                    seed=index,
                    model_manifest_digest=self.manifest_value["model_digest"],
                    variation_protocol_digest=self.manifest_value["protocol_digest"],
                )
                source = "def solve(value):\n    return value + {}\n".format(index).encode("utf-8")
                candidate_id = "candidate-commissioning-" + arm_id.lower()
                self.register_commissioning_candidate(request, candidate_id, source)
                result = self.gateway().evaluate(
                    candidate_id=candidate_id,
                    task_id=request.task_id,
                    source=source,
                    opaque_input=None,
                    requested_authority="EXECUTE_CANDIDATE",
                    declared_locus=self.task["public_locus"],
                )
                receipts = [self.ledger.receipt_by_id(item)["receipt"] for item in result.receipt_ids]
                payload = {
                    "schema_version": "egv-commissioning-generation-response-v1",
                    "request_id": request.request_id,
                    "candidate_id": candidate_id,
                    "candidate_artifact_digest": result.candidate_artifact_digest,
                    "model_output_digest": result.candidate_artifact_digest,
                    "output_byte_count": len(source),
                    "disposition": "PROMOTED",
                    "receipts": receipts,
                }
                from egv.canonical import content_id

                payload["response_id"] = content_id("genresp", payload)
                accepted = validate_accepted_response(
                    request,
                    GenerationResponse.from_mapping(payload),
                    evaluator_public_key=self.signer.public_key,
                    evaluator_digest=self.manifest_value["service_manifest_digest"],
                    expected_first_sequence=receipts[0]["sequence"],
                    expected_previous_receipt_hash=receipts[0]["previous_receipt_hash"],
                )
                self.assertEqual(accepted["arm_id"], arm_id)
                self.assertTrue(all(item["run_id"] == request.run_id for item in receipts))
                self.assertTrue(all(item["arm_policy_digest"] == arm_policy(arm_id).digest for item in receipts))
                self.assertTrue(all(item["policy_digest"] == AuthorityPolicy.candidate_execution().digest for item in receipts))

    def test_response_tamper_and_stale_binding_fail_before_ingestion(self) -> None:
        for mode in ("stale-request", "tamper-result"):
            with self.subTest(mode=mode):
                os.environ["EGV_REMOTE_TEST_MODE"] = mode
                with self.assertRaises(VariationConfigurationError):
                    self.gateway().evaluate(
                        candidate_id="candidate-" + mode,
                        task_id=self.task["template_id"],
                        source=b"def solve(value):\n    return value\n",
                        opaque_input=None,
                        requested_authority="EXECUTE_CANDIDATE",
                        declared_locus=self.task["public_locus"],
                    )
                self.assertEqual(self.ledger.receipts(), [])

    def test_command_tamper_and_manifest_mutation_fail_closed(self) -> None:
        gateway = self.gateway()
        self.command.write_text(self.command.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaises(VariationDependencyError):
            gateway.validate_runtime()
        manifest = RemoteEvaluatorServiceManifest(self.manifest_value)
        manifest._value["campaign_id"] = "stale"  # adversarial mutation of an internal container
        with self.assertRaises(VariationDependencyError):
            manifest.validate_integrity()

    def test_wrong_key_closed_manifest_and_timeout_fail_closed(self) -> None:
        wrong_key = self.root / "wrong.pub"
        wrong_key.write_bytes(ReceiptSigner(b"W" * 32).public_key_raw)
        with self.assertRaises(VariationConfigurationError):
            RemoteControllerEvaluationGateway(
                ledger=self.ledger,
                manifest_path=self.manifest_path,
                public_key_path=wrong_key,
                command=self.command,
            )
        malformed = dict(self.manifest_value)
        malformed["unexpected"] = "field"
        with self.assertRaises(VariationConfigurationError):
            RemoteEvaluatorServiceManifest(malformed)
        gateway = self.gateway()
        with patch("egv.variation.remote._run_bounded_command", side_effect=VariationDependencyError("timeout")):
            with self.assertRaises(VariationDependencyError):
                gateway.evaluate(
                    candidate_id="candidate-timeout",
                    task_id=self.task["template_id"],
                    source=b"def solve(value):\n    return value\n",
                    opaque_input=None,
                    requested_authority="EXECUTE_CANDIDATE",
                    declared_locus=self.task["public_locus"],
                )
        self.assertEqual(self.ledger.receipts(), [])

    def test_public_task_registry_accepts_train_and_heldout_but_rejects_dev(self) -> None:
        for split in ("train", "heldout"):
            with self.subTest(split=split):
                task = {**self.task, "split": split}
                unsigned = {
                    key: value
                    for key, value in self.manifest_value.items()
                    if key != "service_manifest_digest"
                }
                unsigned["task_bindings"] = [task]
                unsigned["task_manifest_digest"] = digest_for([task])
                manifest = RemoteEvaluatorServiceManifest(
                    {**unsigned, "service_manifest_digest": digest_for(unsigned)}
                )
                self.assertEqual(manifest.public_record(task["template_id"]), task)
        task = {**self.task, "split": "dev"}
        unsigned = {
            key: value
            for key, value in self.manifest_value.items()
            if key != "service_manifest_digest"
        }
        unsigned["task_bindings"] = [task]
        unsigned["task_manifest_digest"] = digest_for([task])
        with self.assertRaises(VariationConfigurationError):
            RemoteEvaluatorServiceManifest({**unsigned, "service_manifest_digest": digest_for(unsigned)})

    def test_service_freeze_omits_private_seed_path_and_dev_records(self) -> None:
        seed = self.root / "evaluator-private-seed.bin"
        seed.write_bytes(b"S" * 32)
        corpus = EvaluationCorpus.generate(secret_seed_file=seed)
        config = DockerSandboxConfig()
        with patch.object(DockerSandboxConfig, "verify_image", return_value=config.pinned_image_id):
            manifest = build_remote_evaluator_service_manifest(
                campaign_id="campaign-remote-test",
                model_digest=digest_for("model"),
                protocol_digest=VARIATION_PROTOCOL_DIGEST,
                policy_digest=AuthorityPolicy.candidate_execution().digest,
                corpus=corpus,
                evaluator_revision="remote-evaluator-test-v1",
                public_key_path=self.key_path,
                command=self.command,
                docker_config=config,
            )
        serialized = canonical_json(manifest)
        self.assertNotIn(str(seed), serialized)
        self.assertNotIn("evaluator-private-seed", serialized)
        self.assertEqual({task["split"] for task in manifest["task_bindings"]}, {"train", "heldout"})
        self.assertNotIn("dev", {task["split"] for task in manifest["task_bindings"]})

    def test_atomic_receipt_chain_rolls_back_if_later_signature_is_invalid(self) -> None:
        first = self.signer.sign_receipt(
            {
                "receipt_type": "AUTHORITY", "campaign_id": "campaign", "run_id": "run", "task_id": "task",
                "request_id": "request-a", "candidate_id": "candidate", "decision": "ALLOW",
            },
            sequence=1,
            previous_receipt_hash=GENESIS_HASH,
            idempotency_key="atomic-a",
        )
        second = self.signer.sign_receipt(
            {
                "receipt_type": "VERDICT", "campaign_id": "campaign", "run_id": "run", "task_id": "task",
                "request_id": "request-v", "candidate_id": "candidate", "decision": "PASS",
                "diagnostic_enum": "PASS",
            },
            sequence=2,
            previous_receipt_hash=receipt_hash(first),
            idempotency_key="atomic-v",
        )
        second["signature"] = ("A" if second["signature"][0] != "A" else "B") + second["signature"][1:]
        with self.assertRaises(Exception):
            self.ledger.ingest_receipts_atomic([first, second], self.signer.public_key)
        self.assertEqual(self.ledger.receipts(), [])

    def test_durable_evaluator_returns_byte_identical_cached_response_after_restart(self) -> None:
        manifest = RemoteEvaluatorServiceManifest(self.manifest_value)
        state_root = self.root / "durable-state"
        request = self.durable_request("retry")
        first = _durable_remote_response(
            request,
            manifest=manifest,
            signer=self.signer,
            state_root=state_root,
            build_response=self.durable_builder(request),
        )
        called = []
        resumed = {**request, "receipt_sequence_start": 2, "previous_receipt_hash": receipt_hash(first["receipts"][-1])}
        resumed["request_digest"] = digest_for({key: value for key, value in resumed.items() if key != "request_digest"})
        second = _durable_remote_response(
            resumed,
            manifest=manifest,
            signer=self.signer,
            state_root=state_root,
            build_response=lambda: called.append(True),
        )
        self.assertEqual(canonical_bytes(first), canonical_bytes(second))
        self.assertEqual(called, [])

    def test_crash_after_execution_intent_quarantines_retry_without_reexecution(self) -> None:
        manifest = RemoteEvaluatorServiceManifest(self.manifest_value)
        state_root = self.root / "crash-state"
        request = self.durable_request("crash")
        executions = []

        def crash_after_effect() -> dict:
            executions.append("executed")
            raise RuntimeError("simulated process loss after sandbox effect")

        with self.assertRaises(RuntimeError):
            _durable_remote_response(
                request,
                manifest=manifest,
                signer=self.signer,
                state_root=state_root,
                build_response=crash_after_effect,
            )
        with self.assertRaises(VariationDependencyError):
            _durable_remote_response(
                request,
                manifest=manifest,
                signer=self.signer,
                state_root=state_root,
                build_response=crash_after_effect,
            )
        self.assertEqual(executions, ["executed"])

    def test_concurrent_requests_cannot_sign_receipt_forks(self) -> None:
        manifest = RemoteEvaluatorServiceManifest(self.manifest_value)
        state_root = self.root / "fork-state"
        requests = [self.durable_request("fork-a"), self.durable_request("fork-b")]
        outcomes = []

        def worker(request: dict) -> None:
            try:
                _durable_remote_response(
                    request,
                    manifest=manifest,
                    signer=self.signer,
                    state_root=state_root,
                    build_response=self.durable_builder(request),
                )
                outcomes.append("complete")
            except VariationConfigurationError:
                outcomes.append("stale")

        threads = [threading.Thread(target=worker, args=(request,)) for request in requests]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertCountEqual(outcomes, ["complete", "stale"])


if __name__ == "__main__":
    unittest.main()
