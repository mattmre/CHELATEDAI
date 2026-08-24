from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from egv.canonical import GENESIS_HASH, canonical_json, digest_bytes, digest_for
from egv.cli import main
from egv.evaluation.authority import AuthorityPolicy
from egv.evaluation.dataset import EvaluationCorpus
from egv.evaluation.sandbox import DockerSandboxConfig
from egv.experiment.heldout import (
    CoordinateOperationStore,
    FrozenHeldoutProtocol,
    HeldoutJournal,
    HeldoutProtocolError,
    SHOCK_PHASE,
    run_pending_coordinates,
    verify_signed_reconciliation,
)
from egv.experiment.production import (
    ARTIFACT_NAMES,
    DEPLOYMENT_MANIFEST_SCHEMA,
    DigestPinnedJsonExecutor,
    PRODUCTION_INTEGRATION_NAME,
    PRODUCTION_REQUEST_LIMIT,
    PRODUCTION_RESPONSE_LIMIT,
    ProductionCoordinateDispatcher,
    ProductionDeploymentPaths,
    ProductionHeldoutSchedulerAdapters,
    ProductionObservationVerifier,
    ProductionQwenLoopBuilders,
    ProductionVariationRouterManifest,
    SOURCE_ISOLATION_SCOPE,
    SealedHeldoutDeploymentManifest,
    _observation,
    _provision_router_operation,
    _shock_evidence_bundle,
    build_production_source_manifest,
    build_production_variation_router_command,
    run_routed_remote_variation_once,
)
from egv.experiment.remote import (
    EvaluatorVerification,
    HeldoutVerifierServiceManifest,
    run_heldout_verifier_once,
)
from egv.experiment.runtime import (
    HeldoutTrainerInputs,
    HeldoutTrainerSources,
    build_trainer_evidence_package,
    ordered_public_heldout_task_records,
)
from egv.experiment.shock_engine import ProductionShockEngineFactory
from egv.experiment.shock_runtime import CorrectionShockCoordinateRunner, ShockRuntimeContext
from egv.receipts import ReceiptSigner
from egv.variation.generator import SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST
from egv.variation.loop import MAX_CANDIDATE_ATTEMPTS, VARIATION_PROTOCOL_DIGEST
from egv.variation.model import MODEL_REVISION
from egv.variation.remote import REMOTE_VARIATION_SERVICE_SCHEMA, REMOTE_VARIATION_TIMEOUT_SECONDS
from tests import test_egv_shock_engine as _shock_engine_tests
from tests.test_egv_heldout_campaign import _base_result
from tests.test_egv_heldout_runtime import _protocol


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((canonical_json(value) + "\n").encode("utf-8"))


class _BundleReader:
    def __init__(self, receipt_root: str, ledger_head: str) -> None:
        self.receipt_root = receipt_root
        self.ledger_head = ledger_head

    def bundle_for(self, _coordinate, _runtime_root):
        return {
            "schema_version": "test-private-evidence-v1",
            "receipt_collection_root": self.receipt_root,
            "ledger_head_digest": self.ledger_head,
        }


class HeldoutProductionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        seed = self.root / "seed.bin"
        seed.write_bytes(b"S" * 32)
        self.corpus = EvaluationCorpus.generate(secret_seed_file=seed)
        self.protocol = _protocol(self.corpus)
        raw_inputs, raw_sources = build_trainer_evidence_package(
            self.corpus,
            self.protocol,
            generation_profile_digest=digest_for("generation-profile"),
        )
        self.inputs = HeldoutTrainerInputs(raw_inputs, protocol=self.protocol)
        self.sources = HeldoutTrainerSources(raw_sources, trainer_inputs=self.inputs)
        self.signer = ReceiptSigner(b"P" * 32)

        model_root = self.root / "model"
        adapter_root = self.root / "adapter"
        model_root.mkdir()
        adapter_root.mkdir()
        self.protocol_path = self.root / "protocol.json"
        self.inputs_path = self.root / "trainer-inputs.json"
        self.sources_path = self.root / "trainer-sources.json"
        self.service_path = self.root / "heldout-service.json"
        self.variation_manifest = self.root / "variation-service.json"
        self.variation_key = self.root / "variation-public-key.bin"
        self.variation_command = self.root / "variation-endpoint.py"
        self.variation_router_manifest = self.root / "variation-router.json"
        self.heldout_command = self.root / "heldout-endpoint.py"
        _write_json(self.protocol_path, self.protocol.to_private_dict())
        _write_json(self.inputs_path, self.inputs.canonical_dict())
        _write_json(self.sources_path, self.sources.canonical_dict())
        _write_json(
            self.service_path,
            HeldoutVerifierServiceManifest.from_protocol(self.protocol).to_dict(),
        )
        _write_json(model_root / "model-manifest.json", {"fixture": "model-manifest"})
        _write_json(adapter_root / "adapter-manifest.json", {"fixture": "adapter-manifest"})
        _write_json(self.variation_manifest, {"fixture": "variation-service"})
        self.variation_key.write_bytes(b"V" * 32)
        self.variation_command.write_text("print('{}')\n", encoding="utf-8")
        _write_json(self.variation_router_manifest, {"fixture": "variation-router"})
        self.heldout_command.write_text(
            "import json,sys\n"
            "value=json.loads(sys.stdin.read())\n"
            "raw=json.dumps(value,sort_keys=True,separators=(',',':')).encode('utf-8')+b'\\n'\n"
            "sys.stdout.buffer.write(raw)\n",
            encoding="utf-8",
        )
        self.paths = ProductionDeploymentPaths(
            protocol=self.protocol_path,
            trainer_inputs=self.inputs_path,
            trainer_sources=self.sources_path,
            heldout_service_manifest=self.service_path,
            model_root=model_root,
            adapter_root=adapter_root,
            variation_evaluator_manifest=self.variation_manifest,
            variation_evaluator_public_key=self.variation_key,
            variation_evaluator_command=self.variation_command,
            variation_receipt_router_manifest=self.variation_router_manifest,
            heldout_evaluator_command=self.heldout_command,
            python_executable=Path(sys.executable),
        )
        self.deployment = SealedHeldoutDeploymentManifest.freeze(
            protocol=self.protocol,
            trainer_inputs=self.inputs,
            trainer_sources=self.sources,
            paths=self.paths,
            source_commit="a" * 40,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_manifest_binds_exact_command_bytes_and_contains_no_private_paths(self):
        value = self.deployment.to_dict()
        encoded = canonical_json(value)
        self.assertEqual(value["source_isolation_scope"], "variation-router-only-v1")
        self.assertNotIn(str(self.root), encoded)
        self.assertNotIn("10." + "0.0.", encoded)
        self.assertNotIn("password", encoded.lower())
        self.assertNotIn("authentication", encoded.lower())
        self.assertEqual(
            value["artifacts"]["heldout_evaluator_command"]["sha256"],
            hashlib.sha256(self.heldout_command.read_bytes()).hexdigest(),
        )
        changed = json.loads(json.dumps(value))
        changed["source_commit"] = "b" * 40
        with self.assertRaisesRegex(HeldoutProtocolError, "manifest digest"):
            SealedHeldoutDeploymentManifest(changed)
        overstated = json.loads(json.dumps(value))
        overstated["source_isolation_scope"] = "whole-integration-v1"
        unsigned = dict(overstated)
        unsigned.pop("deployment_manifest_digest")
        overstated["deployment_manifest_digest"] = digest_for(unsigned)
        with self.assertRaisesRegex(HeldoutProtocolError, "manifest digest"):
            SealedHeldoutDeploymentManifest(overstated)

    def _receipt_router_fixture(self):
        signer = ReceiptSigner(b"Q" * 32)
        private_key = self.root / "router-private.key"
        public_key = self.root / "router-public.key"
        private_key.write_bytes(signer.private_key_raw)
        public_key.write_bytes(signer.public_key_raw)
        service = self.root / "router-service.json"
        command = self.root / "router-command"
        state_root = self.root / "router-state"
        workspace = self.root / "router-workspace"
        egv_package_root = Path(__file__).resolve().parents[1] / "egv"
        config = {
            "schema_version": "egv-variation-receipt-router-config-v1",
            "service_manifest": str(service.resolve()),
            "evaluator_seed": str((self.root / "seed.bin").resolve()),
            "evaluator_private_key": str(private_key.resolve()),
            "workspace": str(workspace.resolve()),
            "state_root": str(state_root.resolve()),
            "egv_package_root": str(egv_package_root.resolve()),
            "egv_source_manifest": build_production_source_manifest(
                egv_package_root.resolve(),
                "a" * 40,
            ),
            "python_executable": str(Path(sys.executable).resolve()),
            "bootstrap_executable": str(Path(sys.executable).resolve()),
            "runtime_import_roots": [str(egv_package_root.resolve().parent)],
        }
        command.write_bytes(build_production_variation_router_command(config))
        tasks = sorted((dict(item) for item in self.inputs.tasks.values()), key=lambda item: item["template_id"])
        policy = AuthorityPolicy.candidate_execution().digest
        docker = DockerSandboxConfig()
        revision = "production-router-test-evaluator-v1"
        unsigned = {
            "schema_version": REMOTE_VARIATION_SERVICE_SCHEMA,
            "campaign_id": self.protocol.campaign_id,
            "model_digest": self.protocol.bindings["base_model_digest"],
            "protocol_digest": VARIATION_PROTOCOL_DIGEST,
            "policy_digest": policy,
            "data_manifest_digest": self.corpus.manifest_digest(),
            "task_manifest_digest": digest_for(tasks),
            "task_bindings": tasks,
            "evaluator_revision": revision,
            "evaluator_digest": digest_for(revision),
            "docker_image_digest": docker.pinned_image_id,
            "docker_config_digest": digest_for(dict(docker.__dict__)),
            "authority_policy_digest": policy,
            "command_digest": hashlib.sha256(command.read_bytes()).hexdigest(),
            "evaluator_key_id": signer.key_id,
            "evaluator_public_key_digest": hashlib.sha256(public_key.read_bytes()).hexdigest(),
        }
        _write_json(service, {**unsigned, "service_manifest_digest": digest_for(unsigned)})
        manifest = ProductionVariationRouterManifest.freeze(
            service_manifest=service,
            command=command,
        )
        return signer, private_key, service, command, state_root, workspace, manifest

    def _routed_request(self, label, service, *, sequence=1, previous=GENESIS_HASH):
        manifest = json.loads(Path(service).read_text(encoding="utf-8"))
        task = manifest["task_bindings"][0]
        source = ("def routed_{}():\n    return True\n".format(label.replace("-", "_"))).encode("utf-8")
        stable = {
            "schema_version": "egv-remote-variation-request-v1",
            "service_manifest_digest": manifest["service_manifest_digest"],
            "campaign_id": manifest["campaign_id"],
            "model_digest": manifest["model_digest"],
            "protocol_digest": manifest["protocol_digest"],
            "policy_digest": manifest["policy_digest"],
            "run_id": "router-test-run-" + label,
            "arm_policy_digest": digest_for("router-test-arm"),
            "data_manifest_digest": manifest["data_manifest_digest"],
            "task_manifest_digest": manifest["task_manifest_digest"],
            "evaluator_digest": manifest["evaluator_digest"],
            "docker_image_digest": manifest["docker_image_digest"],
            "candidate_id": "router-test-candidate-" + label,
            "task_id": task["template_id"],
            "public_task_binding": task,
            "candidate_artifact_digest": digest_bytes(source),
            "candidate_source_b64": base64.urlsafe_b64encode(source).decode("ascii").rstrip("="),
            "requested_authority": "execute_candidate",
            "declared_locus": task["public_locus"],
        }
        operation = digest_for(stable)
        body = {
            **stable,
            "operation_digest": operation,
            "receipt_sequence_start": sequence,
            "previous_receipt_hash": previous,
        }
        return {**body, "request_digest": digest_for(body)}

    def test_receipt_router_manifest_rejects_command_substitution_and_hides_paths(self):
        _signer, _private, service, command, _state, _workspace, manifest = self._receipt_router_fixture()
        encoded = canonical_json(manifest.to_dict())
        self.assertNotIn(str(self.root), encoded)
        compile(command.read_bytes(), str(command), "exec")
        command_bytes = command.read_bytes()
        self.assertIn(b" -I -S", command_bytes)
        self.assertIn(b"sys.flags.isolated", command_bytes)
        self.assertIn(b"EGV was imported before source admission", command_bytes)
        self.assertIn(b'return compile(raw_source, str(path), "exec", dont_inherit=True)', command_bytes)
        self.assertEqual(
            manifest["python_executable_digest"],
            hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
        )
        manifest.admit(service_manifest=service, command=command)
        python_bypass = self.root / "router-command.py"
        python_bypass.write_bytes(command.read_bytes())
        with self.assertRaisesRegex(HeldoutProtocolError, "isolated executable launcher"):
            ProductionVariationRouterManifest.freeze(
                service_manifest=service,
                command=python_bypass,
            )
        command.write_bytes(command.read_bytes() + b"\n")
        with self.assertRaisesRegex(HeldoutProtocolError, "reviewed router"):
            manifest.admit(service_manifest=service, command=command)

    def test_receipt_router_rejects_source_substitution_before_import(self):
        _signer, _private, service, command, _state, _workspace, _manifest = self._receipt_router_fixture()
        copied = self.root / "copied-egv"
        shutil.copytree(Path(__file__).resolve().parents[1] / "egv", copied)
        config_match = re.search(
            rb'^CONFIG_B64 = "([A-Za-z0-9_-]+)"$',
            command.read_bytes(),
            flags=re.MULTILINE,
        )
        self.assertIsNotNone(config_match)
        token = config_match.group(1)
        config = json.loads(base64.urlsafe_b64decode(token + b"=" * (-len(token) % 4)))
        config["egv_package_root"] = str(copied.resolve())
        config["egv_source_manifest"] = build_production_source_manifest(copied, "a" * 40)
        command.write_bytes(build_production_variation_router_command(config))
        service_value = json.loads(service.read_text(encoding="utf-8"))
        service_unsigned = {key: value for key, value in service_value.items() if key != "service_manifest_digest"}
        service_unsigned["command_digest"] = hashlib.sha256(command.read_bytes()).hexdigest()
        _write_json(service, {**service_unsigned, "service_manifest_digest": digest_for(service_unsigned)})
        (copied / "canonical.py").write_bytes((copied / "canonical.py").read_bytes() + b"\n")
        with self.assertRaisesRegex(HeldoutProtocolError, "source bytes differ"):
            ProductionVariationRouterManifest.freeze(service_manifest=service, command=command)

    def test_receipt_router_operation_provisioning_is_atomic_across_crash(self):
        operations = self.root / "atomic-router-operations"
        operations.mkdir()
        operation_root = operations / ("f" * 64)
        request_raw = b'{"request":"exact"}\n'
        state_raw = b'{"state":"anchor"}\n'
        real_replace = os.replace

        def crash_before_publish(source, destination):
            if Path(source).name.startswith(".receipt-route-"):
                raise RuntimeError("injected crash before atomic publish")
            return real_replace(source, destination)

        with patch("egv.experiment.production.os.replace", side_effect=crash_before_publish):
            with self.assertRaisesRegex(RuntimeError, "before atomic publish"):
                _provision_router_operation(operations, operation_root, request_raw, state_raw)
        self.assertFalse(operation_root.exists())
        _provision_router_operation(operations, operation_root, request_raw, state_raw)
        self.assertEqual((operation_root / "route-request.json").read_bytes(), request_raw)
        self.assertEqual((operation_root / "remote-evaluator-state.json").read_bytes(), state_raw)

    def test_receipt_router_shares_cached_pre_and_forks_distinct_chains(self):
        signer, private_key, service, _command, state_root, workspace, _manifest = self._receipt_router_fixture()
        effects = []

        def remote(request, *, state_root, **_kwargs):
            state_path = Path(state_root) / "remote-evaluator-state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
            else:
                body = {
                    "schema_version": "egv-remote-variation-state-v1",
                    "service_manifest_digest": json.loads(service.read_text(encoding="utf-8"))[
                        "service_manifest_digest"
                    ],
                    "evaluator_key_id": signer.key_id,
                    "next_sequence": 1,
                    "receipt_head": GENESIS_HASH,
                    "operation_order": [],
                    "responses": {},
                    "pending_operation": None,
                }
                state = {**body, "state_digest": digest_for(body)}
            cached = state["responses"].get(request["operation_digest"])
            if cached is not None:
                return cached
            if state["pending_operation"] is not None:
                raise HeldoutProtocolError("fake remote is quarantined")
            if (
                state["next_sequence"] != request["receipt_sequence_start"]
                or state["receipt_head"] != request["previous_receipt_hash"]
            ):
                raise HeldoutProtocolError("fake remote rejected a stale or forked receipt anchor")
            receipt = signer.sign_receipt(
                {
                    "receipt_type": "AUTHORITY",
                    "request_id": "router-test-" + request["operation_digest"],
                    "decision": "DENY",
                    "campaign_id": "router-test-campaign",
                    "run_id": "router-test-run",
                    "task_id": "router-test-task",
                    "candidate_id": "router-test-candidate-" + request["operation_digest"],
                    "policy_digest": digest_for("router-test-policy"),
                },
                sequence=request["receipt_sequence_start"],
                previous_receipt_hash=request["previous_receipt_hash"],
                idempotency_key="router-test:" + request["operation_digest"],
            )
            response = {
                "operation_digest": request["operation_digest"],
                "request_digest": request["request_digest"],
                "service_manifest_digest": state["service_manifest_digest"],
                "receipts": [receipt],
            }
            body = {key: value for key, value in state.items() if key != "state_digest"}
            body["next_sequence"] = receipt["sequence"] + 1
            body["receipt_head"] = digest_for(receipt)
            body["operation_order"] = list(body["operation_order"]) + [request["operation_digest"]]
            body["responses"] = {**body["responses"], request["operation_digest"]: response}
            state_path.parent.mkdir(parents=True, exist_ok=True)
            _write_json(state_path, {**body, "state_digest": digest_for(body)})
            effects.append(request["operation_digest"])
            return response

        first = self._routed_request("shared-pre", service)
        with patch("egv.experiment.production.run_remote_evaluator_once", side_effect=remote):
            first_response = run_routed_remote_variation_once(
                first,
                service_manifest=service,
                evaluator_seed=self.root / "seed.bin",
                evaluator_private_key=private_key,
                workspace=workspace,
                state_root=state_root,
            )
            replay = run_routed_remote_variation_once(
                first,
                service_manifest=service,
                evaluator_seed=self.root / "seed.bin",
                evaluator_private_key=private_key,
                workspace=workspace,
                state_root=state_root,
            )
            branch = self._routed_request("other-genesis", service)
            run_routed_remote_variation_once(
                branch,
                service_manifest=service,
                evaluator_seed=self.root / "seed.bin",
                evaluator_private_key=private_key,
                workspace=workspace,
                state_root=state_root,
            )
            first_receipt = first_response["receipts"][-1]
            continuation = self._routed_request(
                "shared-pre-next",
                service,
                sequence=first_receipt["sequence"] + 1,
                previous=digest_for(first_receipt),
            )
            run_routed_remote_variation_once(
                continuation,
                service_manifest=service,
                evaluator_seed=self.root / "seed.bin",
                evaluator_private_key=private_key,
                workspace=workspace,
                state_root=state_root,
            )
        self.assertEqual(replay, first_response)
        self.assertEqual(len(effects), 3)

    def test_receipt_router_preserves_ambiguous_pending_operation_for_quarantine(self):
        signer, private_key, service, _command, state_root, workspace, _manifest = self._receipt_router_fixture()
        attempts = []

        def crashing_remote(request, *, state_root, **_kwargs):
            state_path = Path(state_root) / "remote-evaluator-state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state["pending_operation"] is not None:
                    raise HeldoutProtocolError("ambiguous interrupted execution is quarantined")
            attempts.append(request["operation_digest"])
            service_digest = json.loads(service.read_text(encoding="utf-8"))["service_manifest_digest"]
            body = {
                "schema_version": "egv-remote-variation-state-v1",
                "service_manifest_digest": service_digest,
                "evaluator_key_id": signer.key_id,
                "next_sequence": 1,
                "receipt_head": GENESIS_HASH,
                "operation_order": [],
                "responses": {},
                "pending_operation": {
                    "operation_digest": request["operation_digest"],
                    "request_digest": request["request_digest"],
                    "receipt_sequence_start": request["receipt_sequence_start"],
                    "previous_receipt_hash": request["previous_receipt_hash"],
                },
            }
            state_path.parent.mkdir(parents=True, exist_ok=True)
            _write_json(state_path, {**body, "state_digest": digest_for(body)})
            raise RuntimeError("crash after durable evaluator intent")

        request = self._routed_request("ambiguous-effect", service)
        arguments = {
            "service_manifest": service,
            "evaluator_seed": self.root / "seed.bin",
            "evaluator_private_key": private_key,
            "workspace": workspace,
            "state_root": state_root,
        }
        with patch(
            "egv.experiment.production.run_remote_evaluator_once",
            side_effect=crashing_remote,
        ):
            with self.assertRaisesRegex(RuntimeError, "durable evaluator intent"):
                run_routed_remote_variation_once(request, **arguments)
            with self.assertRaisesRegex(HeldoutProtocolError, "quarantined"):
                run_routed_remote_variation_once(request, **arguments)
        self.assertEqual(attempts, [request["operation_digest"]])

    def test_digest_pinned_executor_rejects_command_substitution(self):
        executor = DigestPinnedJsonExecutor(
            self.heldout_command,
            command_digest=self.deployment["artifacts"]["heldout_evaluator_command"]["sha256"],
            python_executable=Path(sys.executable),
            python_digest=self.deployment["artifacts"]["python_executable"]["sha256"],
        )
        self.assertEqual(executor({"hello": "world"}), {"hello": "world"})
        self.heldout_command.write_text("print('{}')\n", encoding="utf-8")
        with self.assertRaisesRegex(HeldoutProtocolError, "changed after admission"):
            executor({"hello": "world"})

    def _dispatcher(self, coordinate, receipt_root, ledger_head, *, crash=False):
        result = _base_result(self.protocol, coordinate)

        def run(_coordinate):
            if crash:
                raise RuntimeError("injected execution crash")
            return result

        return ProductionCoordinateDispatcher(
            protocol=self.protocol,
            deployment=self.deployment,
            runtime_root=self.root / "runtime",
            main_runner=run,
            shock_runner=run,
            main_evidence_reader=_BundleReader(receipt_root, ledger_head),
        )

    def test_dispatcher_requires_exact_scheduler_mapping(self):
        coordinate = self.protocol.coordinates[0]
        operation_store = CoordinateOperationStore(self.root / "ops-map", self.protocol)
        mapping = coordinate.to_dict(self.protocol.digest, self.protocol.campaign_id)
        mapping["idempotency_key"] = operation_store.idempotency_key(coordinate.coordinate_id)
        dispatcher = self._dispatcher(
            coordinate,
            digest_for("receipts"),
            digest_for("ledger"),
        )
        observation = dispatcher(mapping)
        self.assertEqual(observation["result"]["coordinate_id"], coordinate.coordinate_id)
        substituted = dict(mapping, seed=mapping["seed"] + 1)
        with self.assertRaisesRegex(HeldoutProtocolError, "exact frozen coordinate"):
            dispatcher(substituted)

    def _scheduler(self, *, crash=False):
        coordinate = self.protocol.coordinates[0]
        operations = CoordinateOperationStore(self.root / ("ops-crash" if crash else "ops"), self.protocol)
        receipt_root = digest_for({"receipts": coordinate.coordinate_id})
        ledger_head = digest_for({"ledger": coordinate.coordinate_id})
        manifest = HeldoutVerifierServiceManifest.from_protocol(self.protocol)
        state_root = self.root / ("remote-crash" if crash else "remote")

        def observation_verifier(_coordinate, observation):
            bundle = observation["evidence_bundle"]
            return EvaluatorVerification(
                result=observation["result"],
                receipt_collection_root=bundle["receipt_collection_root"],
                ledger_head_digest=bundle["ledger_head_digest"],
            )

        def execute(command):
            return run_heldout_verifier_once(
                self.protocol,
                manifest,
                command,
                signer=self.signer,
                state_root=state_root,
                observation_verifier=observation_verifier,
            )

        # This protocol was built with another signer.  Rebuild only the test
        # signer to match the frozen key while preserving all other bindings.
        self.signer = ReceiptSigner(b"R" * 32)
        self.protocol = type(self.protocol).build(
            campaign_id=self.protocol.campaign_id,
            bindings=dict(self.protocol.bindings),
            evaluator_public_key=self.signer.public_key,
            schedule_seed=self.protocol.schedule_seed,
            bootstrap_seed=self.protocol.bootstrap_seed,
            heldout_task_records=[dict(item) for item in self.protocol.heldout_task_records],
        )
        manifest = HeldoutVerifierServiceManifest.from_protocol(self.protocol)
        operations = CoordinateOperationStore(self.root / ("ops-crash" if crash else "ops"), self.protocol)
        coordinate = self.protocol.coordinates[0]
        dispatcher = self._dispatcher(coordinate, receipt_root, ledger_head, crash=crash)
        adapters = ProductionHeldoutSchedulerAdapters(
            protocol=self.protocol,
            service_manifest=manifest.to_dict(),
            operation_store=operations,
            dispatcher=dispatcher,
            executor=execute,
            dispatch_root=self.root / ("dispatch-crash" if crash else "dispatch"),
            deployment=self.deployment,
        )
        return coordinate, operations, adapters

    def test_begin_execution_verify_and_reconcile_signed_result_roots(self):
        coordinate, operations, adapters = self._scheduler()
        operation = operations.begin(coordinate.coordinate_id)
        mapping = coordinate.to_dict(self.protocol.digest, self.protocol.campaign_id)
        mapping["idempotency_key"] = operations.idempotency_key(coordinate.coordinate_id)
        observation = adapters.runner(mapping)
        self.assertEqual(adapters.dispatch.load(coordinate.coordinate_id)["stage"], "EXECUTED")
        response = adapters.reconciler(mapping, operation)
        reconciliation = verify_signed_reconciliation(
            self.protocol,
            operation,
            response["reconciliation_envelope"],
            response["result_envelope"],
        )
        self.assertEqual(reconciliation["decision"], "COMPLETED")
        record = adapters.dispatch.load(coordinate.coordinate_id)
        self.assertEqual(record["stage"], "VERIFIED")
        self.assertEqual(
            response["result_envelope"]["receipt_collection_root"],
            observation["evidence_bundle"]["receipt_collection_root"],
        )

    def test_crash_after_begin_reconciles_unknown_and_scheduler_quarantines(self):
        coordinate, operations, adapters = self._scheduler(crash=True)
        journal = HeldoutJournal(self.root / "journal-crash", self.protocol)
        with self.assertRaisesRegex(RuntimeError, "injected execution crash"):
            run_pending_coordinates(
                self.protocol,
                journal,
                operations,
                adapters.runner,
                adapters.result_verifier,
                adapters.reconciler,
            )
        self.assertEqual(adapters.dispatch.load(coordinate.coordinate_id)["stage"], "BEGUN")
        with self.assertRaisesRegex(HeldoutProtocolError, "quarantined without rerun"):
            run_pending_coordinates(
                self.protocol,
                journal,
                operations,
                adapters.runner,
                adapters.result_verifier,
                adapters.reconciler,
            )
        self.assertEqual(operations.load(coordinate.coordinate_id)["state"], "QUARANTINED")

    def test_base_and_adapter_builder_guards_cannot_swap_model_identity(self):
        builder = object.__new__(ProductionQwenLoopBuilders)
        base = next(item for item in self.protocol.coordinates if item.treatment == "A")
        adapted = next(item for item in self.protocol.coordinates if item.treatment == "E")
        with patch.object(builder, "_build", return_value="base") as build:
            self.assertEqual(builder.base(coordinate=base), "base")
            self.assertFalse(build.call_args.kwargs["adapter"])
        with self.assertRaisesRegex(HeldoutProtocolError, "adapter arm"):
            builder.base(coordinate=adapted)
        with patch.object(builder, "_build", return_value="adapted") as build:
            self.assertEqual(builder.adapted(coordinate=adapted), "adapted")
            self.assertTrue(build.call_args.kwargs["adapter"])
        with self.assertRaisesRegex(HeldoutProtocolError, "base arm"):
            builder.adapted(coordinate=base)

    def test_genuine_shock_bundle_is_reconstructed_and_journal_claim_is_not_trusted(self):
        short_parent = Path(__file__).resolve().parents[2] if os.name == "nt" else None
        with tempfile.TemporaryDirectory(prefix="egv-e2e-", dir=short_parent) as scratch:
            temporary_root = Path(Path.cwd().anchor) if os.name == "nt" else Path(scratch)
            with patch.object(tempfile, "tempdir", os.fspath(temporary_root)):
                case = _shock_engine_tests.ShockEngineTests(
                    methodName="test_exact_six_pre_attempts_continue_after_early_promotion"
                )
                case.setUp()
                try:
                    campaign_id = "egv-campaign-feedfacecafebeef"
                    records = ordered_public_heldout_task_records(case.corpus)
                    authority = AuthorityPolicy.candidate_execution().digest
                    revision = "shock-hidden-evaluator-v1"
                    docker = DockerSandboxConfig()
                    service_unsigned = {
                        "schema_version": REMOTE_VARIATION_SERVICE_SCHEMA,
                        "campaign_id": campaign_id,
                        "model_digest": case.generator.model_digest,
                        "protocol_digest": VARIATION_PROTOCOL_DIGEST,
                        "policy_digest": authority,
                        "data_manifest_digest": case.corpus.manifest_digest(),
                        "task_manifest_digest": digest_for(records),
                        "task_bindings": records,
                        "evaluator_revision": revision,
                        "evaluator_digest": digest_for(revision),
                        "docker_image_digest": docker.pinned_image_id,
                        "docker_config_digest": digest_for(dict(docker.__dict__)),
                        "authority_policy_digest": authority,
                        "command_digest": hashlib.sha256(case.command.read_bytes()).hexdigest(),
                        "evaluator_key_id": case.signer.key_id,
                        "evaluator_public_key_digest": hashlib.sha256(case.public_key.read_bytes()).hexdigest(),
                    }
                    service = case.root / "production-aligned-service.json"
                    service_value = {
                        **service_unsigned,
                        "service_manifest_digest": digest_for(service_unsigned),
                    }
                    _write_json(service, service_value)
                    bindings = {
                        name: digest_for({"binding": name})
                        for name in FrozenHeldoutProtocol.REQUIRED_BINDINGS
                    }
                    bindings.update(
                        {
                            "base_model_digest": case.generator.model_digest,
                            "trained_model_digest": (
                                case.generator.adapter_attestation.applied_model_state_digest
                            ),
                            "adapter_digest": case.generator.adapter_digest,
                            "prompt_manifest_digest": case.generator.prompt_manifest_digest,
                            "evaluator_digest": service_value["service_manifest_digest"],
                            "policy_manifest_digest": authority,
                            "data_manifest_digest": case.corpus.manifest_digest(),
                        }
                    )
                    protocol = FrozenHeldoutProtocol.build(
                        campaign_id=campaign_id,
                        bindings=bindings,
                        evaluator_public_key=case.signer.public_key,
                        schedule_seed=81,
                        bootstrap_seed=82,
                        heldout_task_records=records,
                    )
                    raw_inputs, raw_sources = build_trainer_evidence_package(
                        case.corpus,
                        protocol,
                        generation_profile_digest=case.generator.generation_profile_digest,
                    )
                    inputs = HeldoutTrainerInputs(raw_inputs, protocol=protocol)
                    sources = HeldoutTrainerSources(raw_sources, trainer_inputs=inputs)
                    artifacts = {
                        name: {"sha256": digest_for({"artifact": name}), "bytes": 1}
                        for name in ARTIFACT_NAMES
                    }
                    deployment_unsigned = {
                        "schema_version": DEPLOYMENT_MANIFEST_SCHEMA,
                        "integration_name": PRODUCTION_INTEGRATION_NAME,
                        "campaign_id": protocol.campaign_id,
                        "protocol_digest": protocol.digest,
                        "trainer_inputs_digest": inputs.digest,
                        "trainer_sources_digest": sources.digest,
                        "base_model_digest": protocol.bindings["base_model_digest"],
                        "trained_model_digest": protocol.bindings["trained_model_digest"],
                        "adapter_digest": protocol.bindings["adapter_digest"],
                        "model_revision": MODEL_REVISION,
                        "generation_profile_digest": case.generator.generation_profile_digest,
                        "response_contract_digest": SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST,
                        "source_commit": "a" * 40,
                        "source_isolation_scope": SOURCE_ISOLATION_SCOPE,
                        "device": "cpu",
                        "torch_dtype": "float32",
                        "max_attempts": MAX_CANDIDATE_ATTEMPTS,
                        "max_new_tokens": 512,
                        "command_timeout_seconds": REMOTE_VARIATION_TIMEOUT_SECONDS,
                        "request_limit_bytes": PRODUCTION_REQUEST_LIMIT,
                        "response_limit_bytes": PRODUCTION_RESPONSE_LIMIT,
                        "artifacts": artifacts,
                    }
                    deployment = SealedHeldoutDeploymentManifest(
                        {
                            **deployment_unsigned,
                            "deployment_manifest_digest": digest_for(deployment_unsigned),
                        }
                    )
                    coordinate = next(
                        item
                        for item in protocol.coordinates
                        if item.phase == SHOCK_PHASE and item.treatment == "dependency-aware"
                    )
                    case.generator.tokenizer.function_name = inputs.tasks[coordinate.task_id][
                        "public_locus"
                    ].rsplit(":", 1)[-1]
                    factory = ProductionShockEngineFactory(
                        generator=case.generator,
                        evaluator_manifest=service,
                        evaluator_public_key=case.public_key,
                        evaluator_command=case.command,
                        source_commit="a" * 40,
                        model_revision=MODEL_REVISION,
                        data_manifest_digest=case.corpus.manifest_digest(),
                        response_contract_digest=case.generator.response_contract_digest,
                        generation_profile_digest=case.generator.generation_profile_digest,
                    )
                    runtime_root = case.root / "production-runtime"
                    runner = CorrectionShockCoordinateRunner(
                        ShockRuntimeContext(
                            protocol=protocol,
                            trainer_inputs=inputs,
                            trainer_sources=sources,
                            root=runtime_root,
                            engine_factory=factory,
                        )
                    )
                    result = runner(coordinate)
                    bundle = _shock_evidence_bundle(
                        protocol=protocol,
                        deployment=deployment,
                        coordinate=coordinate,
                        runtime_root=runtime_root,
                    )
                    observation = _observation(
                        protocol=protocol,
                        deployment=deployment,
                        coordinate=coordinate,
                        result=result,
                        evidence_bundle=bundle,
                    )
                    verifier = ProductionObservationVerifier(
                        protocol=protocol,
                        deployment=deployment,
                        trainer_sources=sources,
                        evaluator_public_key=case.signer.public_key_raw,
                        tokenizer=case.generator.tokenizer,
                    )
                    coordinate_value = coordinate.to_dict(protocol.digest, protocol.campaign_id)
                    verified = verifier(coordinate_value, observation)
                    self.assertEqual(verified.result, result)
                    self.assertEqual(result["eligible_attempts"], 7)
                    self.assertEqual(result["recovery_attempt"], 1)

                    substituted_bundle = json.loads(json.dumps(bundle))
                    substituted_bundle["journal"]["recovery_attempt"] = None
                    unsigned_bundle = dict(substituted_bundle)
                    unsigned_bundle.pop("evidence_bundle_digest")
                    substituted_bundle["evidence_bundle_digest"] = digest_for(unsigned_bundle)
                    substituted_observation = _observation(
                        protocol=protocol,
                        deployment=deployment,
                        coordinate=coordinate,
                        result=result,
                        evidence_bundle=substituted_bundle,
                    )
                    with self.assertRaisesRegex(HeldoutProtocolError, "recovery marker"):
                        verifier(coordinate_value, substituted_observation)
                finally:
                    case.tearDown()

    def test_reviewed_cli_name_requires_all_explicit_sealed_configuration(self):
        status = main(
            [
                "heldout",
                "run",
                "--protocol",
                str(self.protocol_path),
                "--journal",
                str(self.root / "cli-journal"),
                "--operations",
                str(self.root / "cli-operations"),
                "--runner",
                "qwen-heldout-production-v1",
                "--result-verifier",
                "qwen-heldout-production-v1",
                "--reconciler",
                "qwen-heldout-production-v1",
            ]
        )
        self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()
