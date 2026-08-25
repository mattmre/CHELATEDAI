from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from egv.canonical import (
    GENESIS_HASH,
    canonical_bytes,
    canonical_json,
    chain_digest,
    content_id,
    digest_bytes,
    digest_for,
)
from egv.cli import main
from egv.experiment import production as production_module
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
    AuthoritativeMainEvidenceReader,
    DEPLOYMENT_MANIFEST_SCHEMA,
    DigestPinnedJsonExecutor,
    HELDOUT_EVALUATOR_EXECUTION_MODE,
    PRODUCTION_INTEGRATION_NAME,
    PRODUCTION_REQUEST_LIMIT,
    PRODUCTION_RESPONSE_LIMIT,
    ProductionCoordinateDispatcher,
    ProductionDeploymentPaths,
    ProductionHeldoutSchedulerAdapters,
    ProductionMainCoordinateRunner,
    ProductionObservationVerifier,
    ProductionQwenLoopBuilders,
    ProductionVariationRouterManifest,
    SOURCE_ISOLATION_SCOPE,
    SealedHeldoutDeploymentManifest,
    _main_candidate_id,
    _main_run_id,
    _observation,
    _exact_event_envelope,
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
    HeldoutCoordinateRunner,
    HeldoutRuntimeContext,
    HeldoutTrainerInputs,
    HeldoutTrainerSources,
    build_trainer_evidence_package,
    ordered_public_heldout_task_records,
)
from egv.experiment.shock_engine import ProductionShockEngineFactory
from egv.experiment.shock_runtime import CorrectionShockCoordinateRunner, ShockRuntimeContext
from egv.ledger import EvidenceLedger
from egv.receipts import ReceiptSigner
from egv.variation.generator import (
    CandidateContext,
    CandidateGenerationFailureEvidence,
    SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST,
    model_response_contract_digest,
    render_candidate_prompt,
)
from egv.variation.loop import (
    BoundedCandidateLoop,
    MAX_CANDIDATE_ATTEMPTS,
    SourceContractBudgetExhausted,
    VARIATION_PROTOCOL_DIGEST,
)
from egv.variation.errors import VariationCheckpointError, VariationDependencyError
from egv.variation.model import MODEL_REVISION
from egv.variation.checkpoint import VariationCheckpoint
from egv.variation.private import PrivateTrajectoryStore
from egv.variation.retrieval import retrieval_policy
from egv.variation.remote import REMOTE_VARIATION_SERVICE_SCHEMA, REMOTE_VARIATION_TIMEOUT_SECONDS
from egv.variation.remote import RemoteControllerEvaluationGateway
from tests import test_egv_shock_engine as _shock_engine_tests
from tests.test_egv_heldout_campaign import _base_result
from tests.test_egv_heldout_runtime import _protocol


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((canonical_json(value) + "\n").encode("utf-8"))


def _tree_snapshot(root: Path):
    return tuple(
        (
            path.relative_to(root).as_posix(),
            "directory" if path.is_dir() else "file",
            None if path.is_dir() else path.read_bytes(),
        )
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
    )


def _unlink_immutable_fixture(path: Path) -> None:
    path.chmod(0o600)
    path.unlink()


def _reseal_ledger_records(records):
    previous = GENESIS_HASH
    event_hashes = {}
    immutable_fields = (
        "event_type",
        "campaign_id",
        "run_id",
        "task_id",
        "valid_time",
        "subject_id",
        "payload_hash",
        "payload_json",
        "blob_digest",
        "source_class",
        "disposition",
        "evaluator_identity",
        "idempotency_key",
    )
    hash_fields = (
        "sequence",
        "event_id",
        "campaign_id",
        "run_id",
        "task_id",
        "event_type",
        "transaction_time",
        "valid_time",
        "subject_id",
        "payload_hash",
        "payload_json",
        "blob_digest",
        "source_class",
        "disposition",
        "evaluator_identity",
        "idempotency_key",
        "previous_hash",
    )
    for record in records:
        if record.get("record_type") != "EVENT":
            continue
        record["event_id"] = content_id(
            "evt", {key: record.get(key) for key in immutable_fields}
        )
        record["previous_hash"] = previous
        record["event_hash"] = chain_digest(
            previous,
            {
                "schema_version": record["ledger_schema_version"],
                **{key: record.get(key) for key in hash_fields},
            },
        )
        event_hashes[record["event_id"]] = record["event_hash"]
        previous = record["event_hash"]
    for record in records:
        if record.get("record_type") == "CHECKPOINT":
            checkpoint = record["checkpoint"]
            checkpoint["ledger_hash"] = event_hashes[checkpoint["last_durable_event_id"]]
    return "".join(canonical_json(record) + "\n" for record in records), previous


def _substitute_event_envelope(export: str, event_type: str, field: str, value):
    records = [json.loads(line) for line in export.splitlines()]
    target = next(
        record
        for record in reversed(records)
        if record.get("record_type") == "EVENT" and record.get("event_type") == event_type
    )
    target[field] = value
    return _reseal_ledger_records(records)


def _replace_event_payload(export: str, event_type: str, subject_id: str, payload):
    records = [json.loads(line) for line in export.splitlines()]
    target = next(
        record
        for record in reversed(records)
        if record.get("record_type") == "EVENT"
        and record.get("event_type") == event_type
        and record.get("subject_id") == subject_id
    )
    payload_bytes = canonical_bytes(payload)
    target["payload"] = payload
    target["payload_hash"] = digest_bytes(payload_bytes)
    target["payload_json"] = payload_bytes.decode("utf-8")
    target["blob_digest"] = None
    return _reseal_ledger_records(records)


def _raw_ledger_export_tamper(export: str, mode: str) -> str:
    records = [json.loads(line) for line in export.splitlines()]
    last_event_index = max(
        index
        for index, record in enumerate(records)
        if record.get("record_type") == "EVENT"
    )
    if mode == "unknown-field":
        records[last_event_index]["forged_unaccounted_metadata"] = {
            "claim": "must-not-be-ignored"
        }
    elif mode == "duplicate-event":
        records.insert(last_event_index + 1, dict(records[last_event_index]))
    else:
        raise AssertionError("unsupported ledger export tamper")
    return "".join(canonical_json(record) + "\n" for record in records)


def _swap_first_receipt_events(export: str, *, subject_id=None):
    records = [json.loads(line) for line in export.splitlines()]
    indices = [
        index
        for index, record in enumerate(records)
        if record.get("record_type") == "EVENT"
        and record.get("event_type") == "RECEIPT"
        and (subject_id is None or record.get("subject_id") == subject_id)
    ]
    if len(indices) < 2:
        raise AssertionError("receipt swap fixture lacks a signed suffix")
    records[indices[0]], records[indices[1]] = records[indices[1]], records[indices[0]]
    sequence = 0
    for record in records:
        if record.get("record_type") == "EVENT":
            sequence += 1
            record["sequence"] = sequence
    return _reseal_ledger_records(records)


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
        self.variation_key.write_bytes(b"V" * 32)
        self.variation_command.write_text("print('{}')\n", encoding="utf-8")
        self.fixture_evaluator_revision = "production-fixture-evaluator-v1"
        fixture_tasks = [dict(item) for item in self.protocol.heldout_task_records]
        fixture_docker = DockerSandboxConfig()
        fixture_service_unsigned = {
            "schema_version": REMOTE_VARIATION_SERVICE_SCHEMA,
            "campaign_id": self.protocol.campaign_id,
            "model_digest": self.protocol.bindings["base_model_digest"],
            "protocol_digest": VARIATION_PROTOCOL_DIGEST,
            "policy_digest": self.protocol.bindings["policy_manifest_digest"],
            "data_manifest_digest": self.protocol.bindings["data_manifest_digest"],
            "task_manifest_digest": digest_for(fixture_tasks),
            "task_bindings": fixture_tasks,
            "evaluator_revision": self.fixture_evaluator_revision,
            "evaluator_digest": digest_for(self.fixture_evaluator_revision),
            "docker_image_digest": fixture_docker.pinned_image_id,
            "docker_config_digest": digest_for(dict(fixture_docker.__dict__)),
            "authority_policy_digest": AuthorityPolicy.candidate_execution().digest,
            "command_digest": hashlib.sha256(
                self.variation_command.read_bytes()
            ).hexdigest(),
            "evaluator_key_id": "fixture-evaluator-key-v1",
            "evaluator_public_key_digest": hashlib.sha256(
                self.variation_key.read_bytes()
            ).hexdigest(),
        }
        _write_json(
            self.variation_manifest,
            {
                **fixture_service_unsigned,
                "service_manifest_digest": digest_for(fixture_service_unsigned),
            },
        )
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
            python_executable=Path(sys.executable).resolve(),
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
        self.assertEqual(value["evaluator_revision"], self.fixture_evaluator_revision)
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

    def _receipt_router_fixture(
        self,
        *,
        python_executable: Path = Path(sys.executable).resolve(),
        bootstrap_executable: Path = Path(sys.executable).resolve(),
    ):
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
            "python_executable": str(Path(python_executable).resolve()),
            "bootstrap_executable": str(Path(bootstrap_executable).resolve()),
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
            hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
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

    def test_receipt_router_rejects_unmanifested_sourceless_egv_fallback(self):
        _signer, _private, service, command, _state, _workspace, _manifest = (
            self._receipt_router_fixture()
        )
        copied = self.root / "fallback-egv"
        shutil.copytree(Path(__file__).resolve().parents[1] / "egv", copied)
        remote_source = copied / "variation" / "remote.py"
        remote_source.write_bytes(
            remote_source.read_bytes() + b"\nfrom egv import unmanifested_fallback\n"
        )
        fallback_source = self.root / "unmanifested_fallback.py"
        fallback_source.write_text("VALUE = 'ambient-bytecode'\n", encoding="utf-8")
        py_compile.compile(
            str(fallback_source),
            cfile=str(copied / "unmanifested_fallback.pyc"),
            doraise=True,
        )
        fallback_source.unlink()
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
        config["runtime_import_roots"] = list(
            dict.fromkeys(
                str(Path(value).resolve())
                for value in sys.path
                if value and Path(value).is_absolute()
            )
        )
        command.write_bytes(build_production_variation_router_command(config))
        service_value = json.loads(service.read_text(encoding="utf-8"))
        service_unsigned = {
            key: value for key, value in service_value.items() if key != "service_manifest_digest"
        }
        service_unsigned["command_digest"] = hashlib.sha256(command.read_bytes()).hexdigest()
        _write_json(
            service,
            {**service_unsigned, "service_manifest_digest": digest_for(service_unsigned)},
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-S", str(command)],
            input=b"{}\n",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=15,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            b"absent from the admitted source manifest",
            completed.stderr,
        )

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
            python_executable=Path(sys.executable).resolve(),
            python_digest=self.deployment["artifacts"]["python_executable"]["sha256"],
        )
        self.assertEqual(executor({"hello": "world"}), {"hello": "world"})
        self.heldout_command.write_text("print('{}')\n", encoding="utf-8")
        with self.assertRaisesRegex(HeldoutProtocolError, "changed after admission"):
            executor({"hello": "world"})

    def test_digest_pinned_executor_holds_verified_python_identity_through_launch(self):
        suffix = ".exe" if os.name == "nt" else ""
        pinned_python = self.root / ("pinned-python" + suffix)
        shutil.copy2(Path(sys.executable).resolve(), pinned_python)
        original_digest = hashlib.sha256(pinned_python.read_bytes()).hexdigest()
        replacement = self.root / ("replacement-python" + suffix)
        replacement.write_bytes(b"substituted-interpreter-bytes")
        executor = DigestPinnedJsonExecutor(
            self.heldout_command,
            command_digest=self.deployment["artifacts"]["heldout_evaluator_command"]["sha256"],
            python_executable=pinned_python,
            python_digest=original_digest,
        )
        observed = {}

        def replace_during_launch(invocation, **kwargs):
            observed["invocation"] = list(invocation)
            observed["popen_kwargs"] = dict(kwargs)
            try:
                os.replace(replacement, pinned_python)
                observed["pathname_replaced"] = True
            except OSError:
                observed["pathname_replaced"] = False
            pass_fds = tuple(kwargs.get("pass_fds", ()))
            if pass_fds:
                descriptor = pass_fds[0]
                os.lseek(descriptor, 0, os.SEEK_SET)
                digest = hashlib.sha256()
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                observed["open_identity_digest"] = digest.hexdigest()
            raise OSError("deterministic launch-race sentinel")

        with patch("egv.experiment.production.subprocess.Popen", side_effect=replace_during_launch):
            with self.assertRaisesRegex(HeldoutProtocolError, "could not start"):
                executor({"hello": "world"})
        if os.name == "nt":
            self.assertFalse(observed["pathname_replaced"])
            self.assertEqual(observed["invocation"][0], str(pinned_python.resolve()))
            self.assertEqual(hashlib.sha256(pinned_python.read_bytes()).hexdigest(), original_digest)
        else:
            self.assertTrue(observed["pathname_replaced"])
            self.assertRegex(observed["invocation"][0], r"^/(?:proc/self|dev)/fd/\d+$")
            self.assertEqual(observed["open_identity_digest"], original_digest)

    def test_digest_pinned_executor_context_exit_failure_cleans_every_started_resource(self):
        process = unittest.mock.Mock(pid=12345)
        process.stdin = unittest.mock.Mock()
        process.stdout = unittest.mock.Mock()
        process.stderr = unittest.mock.Mock()
        process.kill.side_effect = OSError("fault-injected pre-containment kill failure")
        process.wait.side_effect = subprocess.TimeoutExpired(["pinned"], 5)
        process.stdin.close.side_effect = OSError("fault-injected stdin close failure")
        process.stdout.close.side_effect = OSError("fault-injected stdout close failure")
        process.stderr.close.side_effect = OSError("fault-injected stderr close failure")

        @contextmanager
        def identity_then_fail(*_args, **_kwargs):
            yield str(Path(sys.executable).resolve()), {}
            raise OSError("fault-injected pinned identity release failure")

        def terminate_then_fail(started, _job):
            started.kill()

        executor = DigestPinnedJsonExecutor(
            self.heldout_command,
            command_digest=self.deployment["artifacts"]["heldout_evaluator_command"]["sha256"],
            python_executable=Path(sys.executable).resolve(),
            python_digest=self.deployment["artifacts"]["python_executable"]["sha256"],
        )
        with patch(
            "egv.experiment.production.pinned_python_invocation",
            side_effect=identity_then_fail,
        ), patch(
            "egv.experiment.production.subprocess.Popen",
            return_value=process,
        ), patch(
            "egv.experiment.production._terminate_process_tree",
            side_effect=terminate_then_fail,
        ):
            with self.assertRaisesRegex(
                HeldoutProtocolError,
                "could not complete verified startup",
            ) as raised:
                executor({"hello": "world"})
        self.assertIsInstance(raised.exception.__cause__, OSError)
        self.assertIn(
            "pinned identity release failure",
            str(raised.exception.__cause__),
        )
        self.assertEqual(
            getattr(raised.exception, "cleanup_context", ()),
            (
                "fault-injected pre-containment kill failure",
                "Command '['pinned']' timed out after 5 seconds",
                "fault-injected stdin close failure",
                "fault-injected stdout close failure",
                "fault-injected stderr close failure",
            ),
        )
        process.kill.assert_called_once()
        process.wait.assert_called_once()
        process.stdin.close.assert_called_once()
        process.stdout.close.assert_called_once()
        process.stderr.close.assert_called_once()

    def test_digest_pinned_executor_cleanup_only_failure_preserves_every_error(self):
        process = unittest.mock.Mock(pid=12345)
        process.returncode = 0
        process.stdin = unittest.mock.Mock()
        process.stdout = unittest.mock.Mock()
        process.stderr = unittest.mock.Mock()
        process.stdin.write.side_effect = lambda value: len(value)
        process.stdout.read.return_value = b""
        process.stderr.read.return_value = b""
        process.wait.side_effect = OSError("fault-injected cleanup wait failure")
        process.stdout.close.side_effect = OSError(
            "fault-injected stdout close failure"
        )
        process.stderr.close.side_effect = OSError(
            "fault-injected stderr close failure"
        )
        tree_error = OSError("fault-injected process-tree cleanup wrapper")
        setattr(
            tree_error,
            "cleanup_context",
            (
                "fault-injected tree termination failure",
                "fault-injected tree reap failure",
            ),
        )

        @contextmanager
        def pinned_identity(*_args, **_kwargs):
            yield str(Path(sys.executable).resolve()), {}

        executor = DigestPinnedJsonExecutor(
            self.heldout_command,
            command_digest=self.deployment["artifacts"]["heldout_evaluator_command"][
                "sha256"
            ],
            python_executable=Path(sys.executable).resolve(),
            python_digest=self.deployment["artifacts"]["python_executable"][
                "sha256"
            ],
        )
        expected_context = [
            "fault-injected process-tree cleanup wrapper",
            "fault-injected tree termination failure",
            "fault-injected tree reap failure",
            "fault-injected cleanup wait failure",
            "fault-injected stdout close failure",
            "fault-injected stderr close failure",
        ]
        patches = [
            patch(
                "egv.experiment.production.pinned_python_invocation",
                side_effect=pinned_identity,
            ),
            patch("egv.experiment.production.subprocess.Popen", return_value=process),
            patch(
                "egv.experiment.production._bounded_process_exited_without_reap",
                return_value=True,
            ),
            patch(
                "egv.experiment.production._terminate_process_tree",
                side_effect=tree_error,
            ),
        ]
        if os.name == "nt":
            patches.extend(
                (
                    patch(
                        "egv.experiment.production._assign_kill_on_close_job",
                        return_value=9876,
                    ),
                    patch("egv.experiment.production._resume_windows_process"),
                    patch(
                        "egv.experiment.production._close_windows_job",
                        side_effect=OSError(
                            "fault-injected Job Object close failure"
                        ),
                    ),
                )
            )
            expected_context.append("fault-injected Job Object close failure")

        with patches[0], patches[1], patches[2], patches[3]:
            if os.name == "nt":
                with patches[4], patches[5], patches[6]:
                    with self.assertRaisesRegex(
                        HeldoutProtocolError,
                        "process-tree cleanup failed",
                    ) as raised:
                        executor({"hello": "world"})
            else:
                with self.assertRaisesRegex(
                    HeldoutProtocolError,
                    "process-tree cleanup failed",
                ) as raised:
                    executor({"hello": "world"})

        self.assertIs(raised.exception.__cause__, tree_error)
        self.assertEqual(
            getattr(raised.exception, "cleanup_context", ()),
            tuple(expected_context),
        )
        process.stdin.close.assert_called_once()
        process.stdout.close.assert_called_once()
        process.stderr.close.assert_called_once()

    @unittest.skipUnless(os.name == "nt", "Windows CloseHandle fault canary")
    def test_native_handle_close_failure_is_never_silent(self):
        close_handle = unittest.mock.Mock(return_value=False)
        with self.assertRaisesRegex(OSError, "primary thread close failed"):
            production_module._close_windows_native_handle(
                close_handle,
                12345,
                "primary thread",
            )
        close_handle.assert_called_once_with(12345)

    @unittest.skipUnless(os.name == "nt", "Windows multi-handle cleanup canary")
    def test_native_handle_cleanup_attempts_and_retains_every_close_failure(self):
        close_handle = unittest.mock.Mock(return_value=False)
        errors = production_module._close_windows_native_handles(
            close_handle,
            (
                (12345, "primary thread"),
                (67890, "thread snapshot"),
            ),
        )
        self.assertEqual(len(errors), 2)
        self.assertIn("primary thread close failed", str(errors[0]))
        self.assertIn("thread snapshot close failed", str(errors[1]))
        self.assertEqual(
            close_handle.call_args_list,
            [unittest.mock.call(12345), unittest.mock.call(67890)],
        )

    @unittest.skipUnless(os.name == "nt", "Windows Job Object close fault matrix")
    def test_job_setup_failure_preserves_its_job_close_failure(self):
        for failed_operation in ("configure", "assign"):
            with self.subTest(failed_operation=failed_operation):
                kernel32 = unittest.mock.Mock()
                kernel32.CreateJobObjectW.return_value = 12345
                kernel32.SetInformationJobObject.return_value = (
                    failed_operation != "configure"
                )
                kernel32.AssignProcessToJobObject.return_value = False
                kernel32.CloseHandle.return_value = False
                process = unittest.mock.Mock()
                process._handle = 67890
                with patch("ctypes.WinDLL", return_value=kernel32):
                    with self.assertRaises(OSError) as raised:
                        production_module._assign_kill_on_close_job(process)
                self.assertTrue(
                    any(
                        "Job Object close failed" in item
                        for item in getattr(raised.exception, "cleanup_context", ())
                    )
                )
                kernel32.CloseHandle.assert_called_once_with(12345)

    def test_production_router_ignores_substituted_shebang_bootstrap_and_uses_pinned_python(self):
        suffix = ".exe" if os.name == "nt" else ""
        pinned_python = self.root / ("router-python" + suffix)
        bootstrap = self.root / ("isolated-bootstrap" + suffix)
        shutil.copy2(Path(sys.executable).resolve(), pinned_python)
        shutil.copy2(Path(sys.executable).resolve(), bootstrap)
        _signer, _private, service, command, _state, _workspace, router = self._receipt_router_fixture(
            python_executable=pinned_python,
            bootstrap_executable=bootstrap,
        )
        router.admit(service_manifest=service, command=command)
        bootstrap.write_bytes(b"post-admission-substituted-bootstrap")
        replacement = self.root / ("replacement-router-python" + suffix)
        replacement.write_bytes(b"post-admission-substituted-python")
        ledger = EvidenceLedger(self.root / "router-launch-ledger.sqlite3")
        observed = {}

        def capture(invocation, request_text, *, popen_kwargs=None):
            observed["invocation"] = list(invocation)
            observed["request_text"] = request_text
            observed["popen_kwargs"] = dict(popen_kwargs or {})
            try:
                os.replace(replacement, pinned_python)
                observed["pathname_replaced"] = True
            except OSError:
                observed["pathname_replaced"] = False
            pass_fds = tuple(observed["popen_kwargs"].get("pass_fds", ()))
            if pass_fds:
                descriptor = pass_fds[0]
                os.lseek(descriptor, 0, os.SEEK_SET)
                digest = hashlib.sha256()
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                observed["open_identity_digest"] = digest.hexdigest()
            return 1, b"", b""

        try:
            gateway = RemoteControllerEvaluationGateway(
                ledger=ledger,
                manifest_path=service,
                public_key_path=self.root / "router-public.key",
                command=command,
                python_executable=pinned_python,
                python_digest=router["python_executable_digest"],
            )
            with patch("egv.variation.remote._run_bounded_command", side_effect=capture):
                with self.assertRaisesRegex(VariationDependencyError, "nonzero"):
                    gateway._invoke({})
        finally:
            ledger.close()
        self.assertEqual(observed["invocation"][1:3], ["-I", "-S"])
        self.assertEqual(Path(observed["invocation"][-1]).suffix, "")
        self.assertNotIn(str(bootstrap), observed["invocation"])
        self.assertEqual(observed["request_text"], "{}")
        if os.name == "nt":
            self.assertFalse(observed["pathname_replaced"])
            self.assertEqual(observed["invocation"][0], str(pinned_python.resolve()))
        else:
            self.assertTrue(observed["pathname_replaced"])
            self.assertEqual(observed["open_identity_digest"], router["python_executable_digest"])

    def test_digest_pinned_executor_execution_mode_does_not_depend_on_command_suffix(self):
        extensionless = self.root / "heldout-evaluator-copy"
        extensionless.write_bytes(self.heldout_command.read_bytes())
        command_digest = hashlib.sha256(extensionless.read_bytes()).hexdigest()
        observed = []

        def capture(invocation, **kwargs):
            observed.append((list(invocation), dict(kwargs)))
            raise OSError("execution-mode capture")

        for command in (self.heldout_command, extensionless):
            executor = DigestPinnedJsonExecutor(
                command,
                command_digest=command_digest,
                python_executable=Path(sys.executable).resolve(),
                python_digest=self.deployment["artifacts"]["python_executable"]["sha256"],
                execution_mode=self.deployment["heldout_evaluator_execution_mode"],
            )
            with patch("egv.experiment.production.subprocess.Popen", side_effect=capture):
                with self.assertRaisesRegex(HeldoutProtocolError, "could not start"):
                    executor({"hello": "world"})
            self.assertEqual(executor.execution_mode, "python-json-v1")
        self.assertEqual(len(observed), 2)
        self.assertEqual(len(observed[0][0]), 2)
        self.assertEqual(len(observed[1][0]), 2)
        self.assertEqual(Path(observed[0][0][1]).name, command_digest)
        self.assertEqual(Path(observed[1][0][1]).name, command_digest)

    @unittest.skipUnless(os.name == "nt", "Windows unassigned suspended-child cleanup canary")
    def test_digest_pinned_executor_job_assignment_failure_kills_and_reaps_suspended_child(self):
        started = []
        creationflags = []
        real_popen = subprocess.Popen

        def capture_start(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            started.append(process)
            creationflags.append(int(kwargs.get("creationflags", 0)))
            return process

        executor = DigestPinnedJsonExecutor(
            self.heldout_command,
            command_digest=self.deployment["artifacts"]["heldout_evaluator_command"]["sha256"],
            python_executable=Path(sys.executable).resolve(),
            python_digest=self.deployment["artifacts"]["python_executable"]["sha256"],
        )
        with patch(
            "egv.experiment.production.subprocess.Popen",
            side_effect=capture_start,
        ), patch(
            "egv.experiment.production._assign_kill_on_close_job",
            side_effect=OSError("fault-injected assignment failure"),
        ):
            with self.assertRaisesRegex(HeldoutProtocolError, "could not enter"):
                executor({"hello": "world"})
        self.assertEqual(len(started), 1)
        self.assertTrue(creationflags[0] & production_module._WINDOWS_CREATE_SUSPENDED)
        self.assertIsNotNone(started[0].poll())

    @unittest.skipUnless(os.name == "nt", "Windows failed-start cleanup canary")
    def test_digest_pinned_executor_job_assignment_cleanup_failure_is_not_suppressed(self):
        process = unittest.mock.Mock(pid=12345)
        process.stdin = unittest.mock.Mock()
        process.stdout = unittest.mock.Mock()
        process.stderr = unittest.mock.Mock()
        process.kill.side_effect = OSError("fault-injected direct kill failure")
        process.wait.side_effect = subprocess.TimeoutExpired(["pinned"], 1)
        executor = DigestPinnedJsonExecutor(
            self.heldout_command,
            command_digest=self.deployment["artifacts"]["heldout_evaluator_command"]["sha256"],
            python_executable=Path(sys.executable).resolve(),
            python_digest=self.deployment["artifacts"]["python_executable"]["sha256"],
        )
        with patch(
            "egv.experiment.production.subprocess.Popen",
            return_value=process,
        ), patch(
            "egv.experiment.production._assign_kill_on_close_job",
            side_effect=OSError("fault-injected assignment failure"),
        ):
            with self.assertRaisesRegex(HeldoutProtocolError, "cleanup did not complete") as raised:
                executor({"hello": "world"})
        self.assertIn("fault-injected direct kill failure", raised.exception.cleanup_context)
        process.kill.assert_called_once()
        process.wait.assert_called_once()

    def test_exact_event_envelope_closes_every_main_and_shock_projection_family(self):
        families = (
            "CAMPAIGN",
            "RUN",
            "CANDIDATE",
            "RECEIPT",
            "VERDICT",
            "EFFECT_RECEIPT",
        )
        for projection_path in ("main", "shock"):
            for family in families:
                expected = {
                    "event_type": family,
                    "payload": {"schema_version": "test-envelope-v1", "family": family},
                    "campaign_id": "egv-campaign-cafebabedeadbeef",
                    "run_id": "egv-run-envelope",
                    "task_id": "task-envelope",
                    "subject_id": "subject-envelope",
                    "source_class": "FROZEN_EVALUATOR",
                    "disposition": "VERIFIED",
                    "evaluator_identity": "evaluator-envelope",
                    "idempotency_key": "envelope:{}".format(family),
                }
                payload_bytes = canonical_bytes(expected["payload"])
                immutable = {
                    key: value for key, value in expected.items() if key != "payload"
                }
                immutable.update(
                    {
                        "valid_time": None,
                        "payload_hash": digest_bytes(payload_bytes),
                        "payload_json": payload_bytes.decode("utf-8"),
                        "blob_digest": None,
                    }
                )
                event = {
                    **immutable,
                    "event_id": content_id("evt", immutable),
                    "payload": expected["payload"],
                }
                self.assertTrue(_exact_event_envelope(event, **expected))
                for field in (
                    "event_id",
                    "event_type",
                    "payload",
                    "campaign_id",
                    "run_id",
                    "task_id",
                    "valid_time",
                    "subject_id",
                    "payload_hash",
                    "payload_json",
                    "blob_digest",
                    "source_class",
                    "disposition",
                    "evaluator_identity",
                    "idempotency_key",
                ):
                    with self.subTest(
                        projection_path=projection_path,
                        event_family=family,
                        substituted_field=field,
                    ):
                        tampered = json.loads(json.dumps(event))
                        tampered[field] = (
                            {"schema_version": "tampered-envelope-v1"}
                            if field == "payload"
                            else "tampered-envelope-field"
                        )
                        self.assertFalse(_exact_event_envelope(tampered, **expected))
        large_expected = {
            "event_type": "CANDIDATE",
            "payload": {"schema_version": "test-envelope-v1", "body": "x" * 20000},
            "campaign_id": "egv-campaign-cafebabedeadbeef",
            "run_id": "egv-run-large-envelope",
            "task_id": "task-large-envelope",
            "subject_id": "subject-large-envelope",
            "source_class": None,
            "disposition": "OBSERVED",
            "evaluator_identity": None,
            "idempotency_key": None,
        }
        large_bytes = canonical_bytes(large_expected["payload"])
        large_hash = digest_bytes(large_bytes)
        large_immutable = {
            key: value for key, value in large_expected.items() if key != "payload"
        }
        large_immutable.update(
            {
                "valid_time": None,
                "payload_hash": large_hash,
                "payload_json": None,
                "blob_digest": large_hash,
            }
        )
        large_event = {
            **large_immutable,
            "event_id": content_id("evt", large_immutable),
            "payload": large_expected["payload"],
        }
        self.assertTrue(_exact_event_envelope(large_event, **large_expected))
        for field in ("payload_hash", "payload_json", "blob_digest"):
            with self.subTest(blob_backed_substituted_field=field):
                tampered = dict(large_event)
                tampered[field] = "tampered-blob-storage"
                self.assertFalse(_exact_event_envelope(tampered, **large_expected))

    def test_digest_pinned_executor_times_out_child_that_never_reads_large_stdin(self):
        command = self.root / "never-read.py"
        command.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
        executor = DigestPinnedJsonExecutor(
            command,
            command_digest=hashlib.sha256(command.read_bytes()).hexdigest(),
            python_executable=Path(sys.executable).resolve(),
            python_digest=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
            timeout_seconds=1,
            request_limit=3 * 1024 * 1024,
        )
        started = time.monotonic()
        with self.assertRaisesRegex(HeldoutProtocolError, "timed out"):
            executor({"payload": "x" * (2 * 1024 * 1024)})
        self.assertLess(time.monotonic() - started, 5.0)

    def test_digest_pinned_executor_timeout_terminates_descendant_process_tree(self):
        ready = self.root / "descendant-ready.txt"
        survivor = self.root / "descendant-survived.txt"
        command = self.root / "spawn-descendant.py"
        child = (
            "import pathlib,time;pathlib.Path({!r}).write_text('ready',encoding='utf-8');"
            "time.sleep(1.5);"
            "pathlib.Path({!r}).write_text('survived',encoding='utf-8')"
        ).format(str(ready), str(survivor))
        command.write_text(
            "import subprocess,sys,time\n"
            "subprocess.Popen([" + repr(str(Path(sys.executable).resolve())) + ",'-c'," + repr(child) + "])\n"
            "time.sleep(60)\n",
            encoding="utf-8",
        )
        executor = DigestPinnedJsonExecutor(
            command,
            command_digest=hashlib.sha256(command.read_bytes()).hexdigest(),
            python_executable=Path(sys.executable).resolve(),
            python_digest=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
            timeout_seconds=1,
        )
        started = time.monotonic()
        with self.assertRaisesRegex(HeldoutProtocolError, "timed out"):
            executor({"bounded": True})
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertTrue(ready.exists())
        time.sleep(1.0)
        self.assertFalse(survivor.exists())

    def test_digest_pinned_executor_bounds_descendant_inherited_output_pipe(self):
        ready = self.root / "inherited-pipe-descendant-ready.txt"
        survivor = self.root / "inherited-pipe-descendant-survived.txt"
        command = self.root / "inherited-pipe.py"
        child = (
            "import pathlib,time;pathlib.Path({!r}).write_text('ready',encoding='utf-8');"
            "time.sleep(1.5);"
            "pathlib.Path({!r}).write_text('survived',encoding='utf-8')"
        ).format(str(ready), str(survivor))
        command.write_text(
            "import pathlib,subprocess,sys,time\n"
            "sys.stdin.buffer.read()\n"
            "subprocess.Popen([" + repr(str(Path(sys.executable).resolve())) + ",'-c'," + repr(child) + "])\n"
            "deadline=time.monotonic()+1\n"
            "while not pathlib.Path(" + repr(str(ready)) + ").exists() and time.monotonic()<deadline: time.sleep(0.01)\n"
            "sys.stdout.buffer.write(b'{\"ok\":true}\\n')\n"
            "sys.stdout.buffer.flush()\n",
            encoding="utf-8",
        )
        executor = DigestPinnedJsonExecutor(
            command,
            command_digest=hashlib.sha256(command.read_bytes()).hexdigest(),
            python_executable=Path(sys.executable).resolve(),
            python_digest=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
            timeout_seconds=1,
        )
        started = time.monotonic()
        self.assertEqual(executor({"bounded": True}), {"ok": True})
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertTrue(ready.exists())
        time.sleep(1.0)
        self.assertFalse(survivor.exists())

    def test_digest_pinned_executor_output_overflow_terminates_ready_descendant(self):
        ready = self.root / "overflow-descendant-ready.txt"
        survivor = self.root / "overflow-descendant-survived.txt"
        command = self.root / "overflow-tree.py"
        child = (
            "import pathlib,time;pathlib.Path({!r}).write_text('ready',encoding='utf-8');"
            "time.sleep(1.5);"
            "pathlib.Path({!r}).write_text('survived',encoding='utf-8')"
        ).format(str(ready), str(survivor))
        command.write_text(
            "import pathlib,subprocess,sys,time\n"
            "ready=pathlib.Path(" + repr(str(ready)) + ")\n"
            "subprocess.Popen([" + repr(str(Path(sys.executable).resolve())) + ",'-c'," + repr(child) + "])\n"
            "deadline=time.monotonic()+1\n"
            "while not ready.exists() and time.monotonic()<deadline: time.sleep(0.01)\n"
            "sys.stdout.write('x'*4096)\n"
            "sys.stdout.flush()\n",
            encoding="utf-8",
        )
        executor = DigestPinnedJsonExecutor(
            command,
            command_digest=hashlib.sha256(command.read_bytes()).hexdigest(),
            python_executable=Path(sys.executable).resolve(),
            python_digest=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
            timeout_seconds=2,
            response_limit=128,
        )
        started = time.monotonic()
        with self.assertRaisesRegex(HeldoutProtocolError, "stdout exceeded its byte limit"):
            executor({"bounded": True})
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertTrue(ready.exists())
        time.sleep(1.0)
        self.assertFalse(survivor.exists())

    def test_digest_pinned_executor_reader_failure_is_captured_and_fails_closed(self):
        real_popen = subprocess.Popen

        class FailingReader:
            def __init__(self, stream):
                self.stream = stream

            def read(self, _size):
                raise OSError("fault-injected response reader failure")

            def close(self):
                self.stream.close()

        def start_with_failing_stdout(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            process.stdout = FailingReader(process.stdout)
            return process

        executor = DigestPinnedJsonExecutor(
            self.heldout_command,
            command_digest=self.deployment["artifacts"]["heldout_evaluator_command"]["sha256"],
            python_executable=Path(sys.executable).resolve(),
            python_digest=self.deployment["artifacts"]["python_executable"]["sha256"],
        )
        with patch("egv.experiment.production.subprocess.Popen", side_effect=start_with_failing_stdout):
            with self.assertRaisesRegex(HeldoutProtocolError, "response read did not complete"):
                executor({"hello": "world"})

    def test_digest_pinned_executor_overflow_remains_primary_when_cleanup_also_fails(self):
        from egv.experiment import production as production_module

        command = self.root / "overflow-primary.py"
        command.write_text(
            "import sys\nsys.stdin.buffer.read()\nsys.stdout.write('x'*4096)\nsys.stdout.flush()\n",
            encoding="utf-8",
        )
        executor = DigestPinnedJsonExecutor(
            command,
            command_digest=hashlib.sha256(command.read_bytes()).hexdigest(),
            python_executable=Path(sys.executable).resolve(),
            python_digest=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
            response_limit=128,
        )
        real_cleanup = production_module._terminate_process_tree

        def cleanup_then_report(*args, **kwargs):
            real_cleanup(*args, **kwargs)
            raise OSError("fault-injected secondary cleanup failure")

        with patch("egv.experiment.production._terminate_process_tree", side_effect=cleanup_then_report):
            with self.assertRaisesRegex(HeldoutProtocolError, "stdout exceeded") as raised:
                executor({"bounded": True})
        self.assertEqual(
            getattr(raised.exception, "cleanup_context", ()),
            ("fault-injected secondary cleanup failure",),
        )

    @unittest.skipIf(os.name == "nt", "POSIX waitid ordering canary")
    def test_digest_pinned_executor_cleanup_anchors_leader_immediately_before_killpg(self):
        process = unittest.mock.Mock(pid=12345)
        order = []
        with patch(
            "egv.experiment.production.os.waitid",
            side_effect=lambda *args: order.append(("waitid", args[1])) or object(),
        ), patch(
            "egv.experiment.production.os.killpg",
            side_effect=lambda pid, sig: order.append(("killpg", pid)),
        ):
            production_module._terminate_process_tree(process, None)
        self.assertEqual(order, [("waitid", process.pid), ("killpg", process.pid)])

    @unittest.skipIf(os.name == "nt", "POSIX lost-anchor cleanup canary")
    def test_digest_pinned_executor_cleanup_never_signals_after_lost_leader_anchor(self):
        process = unittest.mock.Mock(pid=12345)
        with patch(
            "egv.experiment.production.os.waitid",
            side_effect=ChildProcessError("fault-injected lost leader"),
        ), patch("egv.experiment.production.os.killpg") as killpg:
            with self.assertRaisesRegex(OSError, "leader anchor was lost"):
                production_module._terminate_process_tree(process, None)
        killpg.assert_not_called()

    def test_digest_pinned_executor_late_native_startup_fails_as_timeout(self):
        real_popen = subprocess.Popen

        def delayed_start(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            time.sleep(1.1)
            return process

        executor = DigestPinnedJsonExecutor(
            self.heldout_command,
            command_digest=self.deployment["artifacts"]["heldout_evaluator_command"]["sha256"],
            python_executable=Path(sys.executable).resolve(),
            python_digest=self.deployment["artifacts"]["python_executable"]["sha256"],
            timeout_seconds=1,
        )
        with patch("egv.experiment.production.subprocess.Popen", side_effect=delayed_start):
            with self.assertRaisesRegex(HeldoutProtocolError, "timed out"):
                executor({"hello": "world"})

    def test_private_raw_generation_replays_contract_failure_and_marks_runtime_trust_boundary(self):
        rendered = b"exact private prompt"
        context = CandidateContext(
            campaign_id="egv-campaign-raw-replay",
            run_id="egv-run-raw-replay",
            seed=17,
            arm_id="E",
            task_id="task-raw-replay",
            family_id="family-raw-replay",
            public_locus="module.py:solve",
            public_rule_id="rule-raw-replay",
            attempt_index=1,
            parent_candidate_id=None,
            retrieval_records=(),
            retrieval_digest=digest_for([]),
            model_digest=digest_for("model-raw-replay"),
            adapter_digest=digest_for("adapter-raw-replay"),
            prompt_digest=digest_bytes(rendered),
            task_statement="Repair the bounded raw replay task.",
            initial_source="def solve(value):\n    return value\n",
            initial_source_digest=digest_bytes(b"def solve(value):\n    return value\n"),
            response_contract="source-only-v1",
            response_contract_digest=SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST,
            generation_profile_digest=digest_for("generation-profile-raw-replay"),
        )
        invalid = b"not a complete solve function"
        contract_failure = CandidateGenerationFailureEvidence(
            stage="RESPONSE_CONTRACT",
            response_contract="source-only-v1",
            rendered_prompt=rendered,
            rendered_prompt_digest=digest_bytes(rendered),
            decoded_model_response=invalid,
            decoded_model_response_digest=digest_bytes(invalid),
            contract_response=invalid,
            contract_response_digest=digest_bytes(invalid),
            error_code="VariationDependencyError",
        )
        contract_failure.validate(context)
        with patch.object(
            sys,
            "exception",
            create=True,
            side_effect=AssertionError("Python 3.11-only sys.exception was called"),
        ):
            record = production_module._failed_generation_record(
                "candidate-raw-replay", context, contract_failure
            )
            material = production_module._private_raw_generation_material(
                contract_failure, status="FAILED"
            )
            self.assertIsNone(
                production_module._verify_private_raw_generation(material, record, context)
            )
        self.assertEqual(material["verification_mode"], "INDEPENDENT_FAILURE_REPLAY")
        self.assertEqual(
            record["replay_error_chain"],
            ["VariationDependencyError", "SyntaxError"],
        )
        self.assertIsNone(production_module._verify_private_raw_generation(material, record, context))
        reclassified = dict(record)
        reclassified["error_code"] = "SyntaxError"
        with self.assertRaisesRegex(HeldoutProtocolError, "exception chain is not closed"):
            production_module._verify_private_raw_generation(material, reclassified, context)

        missing_locus = b"value = 1"
        missing_locus_failure = CandidateGenerationFailureEvidence(
            stage="RESPONSE_CONTRACT",
            response_contract="source-only-v1",
            rendered_prompt=rendered,
            rendered_prompt_digest=digest_bytes(rendered),
            decoded_model_response=missing_locus,
            decoded_model_response_digest=digest_bytes(missing_locus),
            contract_response=missing_locus,
            contract_response_digest=digest_bytes(missing_locus),
            error_code="VariationDependencyError",
        )
        missing_locus_record = production_module._failed_generation_record(
            "candidate-missing-locus", context, missing_locus_failure
        )
        missing_locus_material = production_module._private_raw_generation_material(
            missing_locus_failure, status="FAILED"
        )
        self.assertEqual(
            missing_locus_record["replay_error_chain"],
            ["VariationDependencyError"],
        )
        syntax_as_missing = dict(record)
        syntax_as_missing["replay_error_chain"] = missing_locus_record["replay_error_chain"]
        with self.assertRaisesRegex(HeldoutProtocolError, "failure chain changed"):
            production_module._verify_private_raw_generation(material, syntax_as_missing, context)
        missing_as_syntax = dict(missing_locus_record)
        missing_as_syntax["replay_error_chain"] = record["replay_error_chain"]
        with self.assertRaisesRegex(HeldoutProtocolError, "failure chain changed"):
            production_module._verify_private_raw_generation(
                missing_locus_material,
                missing_as_syntax,
                context,
            )

        runtime_failure = CandidateGenerationFailureEvidence(
            stage="MODEL_GENERATION",
            response_contract="source-only-v1",
            rendered_prompt=rendered,
            rendered_prompt_digest=digest_bytes(rendered),
            decoded_model_response=None,
            decoded_model_response_digest=None,
            contract_response=None,
            contract_response_digest=None,
            error_code="RuntimeError",
        )
        runtime_failure.validate(context)
        runtime_record = production_module._failed_generation_record(
            "candidate-runtime-boundary", context, runtime_failure
        )
        runtime_material = production_module._private_raw_generation_material(
            runtime_failure, status="FAILED"
        )
        self.assertEqual(
            runtime_material["verification_mode"], "TRAINER_ATTESTED_RUNTIME_FAILURE"
        )
        self.assertIsNone(
            production_module._verify_private_raw_generation(
                runtime_material, runtime_record, context
            )
        )

    def test_private_reconciliation_rejects_every_unexpected_inventory_entry(self):
        cases = (
            ("generation_intents", "foreign.bin", "file"),
            ("generation_starts", "foreign", "directory"),
            ("generation_records", "foreign.json", "hardlink"),
            ("records", "foreign.json", "symlink"),
        )
        for index, (root_name, name, kind) in enumerate(cases):
            with self.subTest(root=root_name, kind=kind):
                store = PrivateTrajectoryStore(
                    self.root / "closed-private-inventory-{}".format(index)
                )
                root = getattr(store, root_name)
                path = root / name
                if kind == "file":
                    path.write_bytes(b"unaccounted")
                elif kind == "directory":
                    path.mkdir()
                else:
                    backing = self.root / "closed-private-backing-{}".format(index)
                    backing.write_bytes(b"unaccounted")
                    try:
                        if kind == "hardlink":
                            os.link(backing, path)
                        else:
                            path.symlink_to(backing)
                    except OSError:
                        if kind == "symlink":
                            continue
                        raise
                with self.assertRaisesRegex(
                    VariationCheckpointError,
                    "inventory (?:contains an unexpected entry|entry is linked or not regular)",
                ):
                    store._reconcile_generation_intents(repair=False)

    def test_private_prompt_integrity_failure_round_trips_and_rejects_intent_tamper(self):
        expected_rendered = b"expected private prompt"
        differing_rendered = b"differing private prompt"
        source = "def solve(value):\n    return value\n"
        context = CandidateContext(
            campaign_id="egv-campaign-prompt-integrity",
            run_id="egv-run-prompt-integrity",
            seed=29,
            arm_id="E",
            task_id="task-prompt-integrity",
            family_id="family-prompt-integrity",
            public_locus="module.py:solve",
            public_rule_id="rule-prompt-integrity",
            attempt_index=1,
            parent_candidate_id=None,
            retrieval_records=(),
            retrieval_digest=digest_for([]),
            model_digest=digest_for("model-prompt-integrity"),
            adapter_digest=digest_for("adapter-prompt-integrity"),
            prompt_digest=digest_bytes(expected_rendered),
            task_statement="Repair the bounded prompt-integrity task.",
            initial_source=source,
            initial_source_digest=digest_bytes(source.encode("utf-8")),
            response_contract="source-only-v1",
            response_contract_digest=SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST,
            generation_profile_digest=digest_for("generation-profile-prompt-integrity"),
        )
        failure = CandidateGenerationFailureEvidence(
            stage="PROMPT_INTEGRITY",
            response_contract=context.response_contract,
            rendered_prompt=differing_rendered,
            rendered_prompt_digest=digest_bytes(differing_rendered),
            decoded_model_response=None,
            decoded_model_response_digest=None,
            contract_response=None,
            contract_response_digest=None,
            error_code="PromptDigestMismatch",
        )
        candidate_id = "candidate-prompt-integrity"
        store = PrivateTrajectoryStore(self.root / "prompt-integrity-private")
        store.record_generation_start(candidate_id=candidate_id, context=context)
        record_digest = store.record_generation_failure(
            candidate_id=candidate_id,
            context=context,
            evidence=failure,
        )
        loaded_context, loaded_failure, loaded_digest = store.load_generation_failure_read_only(
            candidate_id
        )
        self.assertEqual(loaded_context, context)
        self.assertEqual(loaded_failure, failure)
        self.assertEqual(loaded_digest, record_digest)
        record = json.loads(
            (store.generation_records / (candidate_id + ".json")).read_text(encoding="utf-8")
        )
        self.assertEqual(record["schema_version"], "egv-private-generation-evidence-v2")
        self.assertIsNone(record["replay_error_chain"])

        intent_path = store.generation_intents / (candidate_id + ".json")
        intent = json.loads(intent_path.read_text(encoding="utf-8"))
        self.assertEqual(intent["schema_version"], "egv-private-generation-intent-v2")
        legacy_schema_intent = dict(intent)
        legacy_schema_intent["schema_version"] = "egv-private-generation-intent-v1"
        legacy_schema_intent.pop("replay_error_chain")
        intent_path.chmod(0o600)
        intent_path.write_bytes(canonical_bytes(legacy_schema_intent))
        with self.assertRaisesRegex(VariationCheckpointError, "not canonical and closed"):
            store.load_generation_failure_read_only(candidate_id)
        intent_path.write_bytes(canonical_bytes(intent))

        record_path = store.generation_records / (candidate_id + ".json")
        legacy_schema_record = dict(record)
        legacy_schema_record["schema_version"] = "egv-private-generation-evidence-v1"
        legacy_schema_record.pop("replay_error_chain")
        record_path.chmod(0o600)
        record_path.write_bytes(canonical_bytes(legacy_schema_record))
        with self.assertRaisesRegex(VariationCheckpointError, "not canonical and closed"):
            store.load_generation_failure_read_only(candidate_id)
        record_path.write_bytes(canonical_bytes(record))

        intent["rendered_prompt_b64"] = base64.b64encode(expected_rendered).decode(
            "ascii"
        )
        intent_path.chmod(0o600)
        intent_path.write_bytes(canonical_bytes(intent))
        with self.assertRaisesRegex(
            VariationCheckpointError,
            "does not preserve differing prompt bytes",
        ):
            store.load_generation_failure_read_only(candidate_id)

    def test_private_raw_generation_rejects_self_consistent_stage_shape_forgeries(self):
        exact_rendered = b"exact private prompt"
        differing_rendered = b"different private prompt"
        context = CandidateContext(
            campaign_id="egv-campaign-raw-shape",
            run_id="egv-run-raw-shape",
            seed=19,
            arm_id="E",
            task_id="task-raw-shape",
            family_id="family-raw-shape",
            public_locus="module.py:solve",
            public_rule_id="rule-raw-shape",
            attempt_index=1,
            parent_candidate_id=None,
            retrieval_records=(),
            retrieval_digest=digest_for([]),
            model_digest=digest_for("model-raw-shape"),
            adapter_digest=digest_for("adapter-raw-shape"),
            prompt_digest=digest_bytes(exact_rendered),
            task_statement="Repair the bounded raw shape task.",
            initial_source="def solve(value):\n    return value\n",
            initial_source_digest=digest_bytes(b"def solve(value):\n    return value\n"),
            response_contract="source-only-v1",
            response_contract_digest=SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST,
            generation_profile_digest=digest_for("generation-profile-raw-shape"),
        )

        def assert_shape_rejected(stage, rendered, decoded, contract):
            failure = CandidateGenerationFailureEvidence(
                stage=stage,
                response_contract=context.response_contract,
                rendered_prompt=rendered,
                rendered_prompt_digest=(digest_bytes(rendered) if rendered is not None else None),
                decoded_model_response=decoded,
                decoded_model_response_digest=(digest_bytes(decoded) if decoded is not None else None),
                contract_response=contract,
                contract_response_digest=(digest_bytes(contract) if contract is not None else None),
                error_code="VariationDependencyError",
            )
            record = production_module._failed_generation_record(
                "candidate-raw-shape-{}".format(stage.lower()), context, failure
            )
            material = production_module._private_raw_generation_material(
                failure, status="FAILED"
            )
            with self.assertRaisesRegex(HeldoutProtocolError, "artifact shape"):
                production_module._verify_private_raw_generation(material, record, context)

        invalid = b"not a complete solve function"
        cases = (
            ("PROMPT_RENDER", exact_rendered, None, None),
            ("PROMPT_INTEGRITY", None, None, None),
            ("PROMPT_INTEGRITY", differing_rendered, invalid, None),
            ("MODEL_GENERATION", None, None, None),
            ("MODEL_GENERATION", exact_rendered, invalid, None),
            ("RESPONSE_CONTRACT", exact_rendered, None, invalid),
        )
        for case in cases:
            with self.subTest(stage=case[0], presence=tuple(item is not None for item in case[1:])):
                assert_shape_rejected(*case)

    def test_private_raw_generation_binds_source_only_decoded_and_contract_bytes(self):
        rendered = b"exact private prompt"
        initial_source = "def solve(value):\n    return value\n"
        base_context = CandidateContext(
            campaign_id="egv-campaign-raw-binding",
            run_id="egv-run-raw-binding",
            seed=23,
            arm_id="E",
            task_id="task-raw-binding",
            family_id="family-raw-binding",
            public_locus="module.py:solve",
            public_rule_id="rule-raw-binding",
            attempt_index=1,
            parent_candidate_id=None,
            retrieval_records=(),
            retrieval_digest=digest_for([]),
            model_digest=digest_for("model-raw-binding"),
            adapter_digest=digest_for("adapter-raw-binding"),
            prompt_digest=digest_bytes(rendered),
            task_statement="Repair the bounded raw binding task.",
            initial_source=initial_source,
            initial_source_digest=digest_bytes(initial_source.encode("utf-8")),
            response_contract="source-only-v1",
            response_contract_digest=SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST,
            generation_profile_digest=digest_for("generation-profile-raw-binding"),
        )

        def verify_rejected(context, decoded, contract):
            failure = CandidateGenerationFailureEvidence(
                stage="RESPONSE_CONTRACT",
                response_contract=context.response_contract,
                rendered_prompt=rendered,
                rendered_prompt_digest=digest_bytes(rendered),
                decoded_model_response=decoded,
                decoded_model_response_digest=digest_bytes(decoded),
                contract_response=contract,
                contract_response_digest=digest_bytes(contract),
                error_code="VariationDependencyError",
            )
            record = production_module._failed_generation_record(
                "candidate-raw-binding", context, failure
            )
            material = production_module._private_raw_generation_material(
                failure, status="FAILED"
            )
            with self.assertRaisesRegex(HeldoutProtocolError, "not bound"):
                production_module._verify_private_raw_generation(material, record, context)

        verify_rejected(
            base_context,
            b"not a complete solve function",
            b"different incomplete solve function",
        )

        prefill_context = replace(
            base_context,
            response_contract="source-only-prefill-v1",
            response_contract_digest=model_response_contract_digest("source-only-prefill-v1"),
        )
        decoded = b"    return value ???"
        verify_rejected(
            prefill_context,
            decoded,
            b"def solve(value):\n    return other ???",
        )
        valid_prefill_contract = b"def solve(value):\n" + decoded
        valid_prefill_failure = CandidateGenerationFailureEvidence(
            stage="RESPONSE_CONTRACT",
            response_contract=prefill_context.response_contract,
            rendered_prompt=rendered,
            rendered_prompt_digest=digest_bytes(rendered),
            decoded_model_response=decoded,
            decoded_model_response_digest=digest_bytes(decoded),
            contract_response=valid_prefill_contract,
            contract_response_digest=digest_bytes(valid_prefill_contract),
            error_code="VariationDependencyError",
        )
        valid_prefill_record = production_module._failed_generation_record(
            "candidate-valid-prefill-binding",
            prefill_context,
            valid_prefill_failure,
        )
        valid_prefill_material = production_module._private_raw_generation_material(
            valid_prefill_failure,
            status="FAILED",
        )
        self.assertEqual(
            valid_prefill_record["replay_error_chain"],
            ["VariationDependencyError", "SyntaxError"],
        )
        self.assertIsNone(
            production_module._verify_private_raw_generation(
                valid_prefill_material,
                valid_prefill_record,
                prefill_context,
            )
        )

    def test_all_invalid_main_generation_is_durable_verified_budget_exhaustion(self):
        signer = ReceiptSigner(b"H" * 32)
        bindings = dict(self.protocol.bindings)
        bindings["policy_manifest_digest"] = AuthorityPolicy.candidate_execution().digest
        self.protocol = FrozenHeldoutProtocol.build(
            campaign_id=self.protocol.campaign_id,
            bindings=bindings,
            evaluator_public_key=signer.public_key,
            schedule_seed=self.protocol.schedule_seed,
            bootstrap_seed=self.protocol.bootstrap_seed,
            heldout_task_records=[dict(item) for item in self.protocol.heldout_task_records],
        )
        raw_inputs, raw_sources = build_trainer_evidence_package(
            self.corpus,
            self.protocol,
            generation_profile_digest=digest_for("generation-profile"),
        )
        self.inputs = HeldoutTrainerInputs(raw_inputs, protocol=self.protocol)
        self.sources = HeldoutTrainerSources(raw_sources, trainer_inputs=self.inputs)
        _write_json(self.protocol_path, self.protocol.to_private_dict())
        _write_json(self.inputs_path, self.inputs.canonical_dict())
        _write_json(self.sources_path, self.sources.canonical_dict())
        _write_json(
            self.service_path,
            HeldoutVerifierServiceManifest.from_protocol(self.protocol).to_dict(),
        )
        self.deployment = SealedHeldoutDeploymentManifest.freeze(
            protocol=self.protocol,
            trainer_inputs=self.inputs,
            trainer_sources=self.sources,
            paths=self.paths,
            source_commit="a" * 40,
        )
        coordinate = next(
            item
            for item in self.protocol.coordinates
            if item.phase != SHOCK_PHASE and item.treatment == "A"
        )
        if os.name == "nt":
            runtime_root = Path(tempfile.mkdtemp(prefix="egv-m-", dir=Path.cwd().anchor))
            self.addCleanup(shutil.rmtree, runtime_root, ignore_errors=True)
        else:
            runtime_root = self.root / "all-invalid-runtime"
        coordinate_root = runtime_root / coordinate.coordinate_id
        private_store = PrivateTrajectoryStore(coordinate_root / "private")
        run_id = _main_run_id(self.protocol, coordinate)
        task = dict(self.inputs.tasks[coordinate.task_id])
        source = self.sources.source_for(coordinate.task_id)
        tokenizer = _shock_engine_tests._Tokenizer()
        tokenizer.function_name = task["public_locus"].rsplit(":", 1)[-1]
        with EvidenceLedger(coordinate_root / "ledger.sqlite3") as ledger:
            ledger.create_campaign(
                self.protocol.campaign_id,
                protocol_hash=VARIATION_PROTOCOL_DIGEST,
                source_commit=self.deployment["source_commit"],
                model_revision=MODEL_REVISION,
                data_manifest_hash=self.protocol.bindings["data_manifest_digest"],
                evaluator_hash=self.protocol.bindings["evaluator_digest"],
                policy_hash=self.protocol.bindings["policy_manifest_digest"],
                seed_set=self.protocol.seeds,
            )
            ledger.create_run(
                run_id,
                campaign_id=self.protocol.campaign_id,
                arm=coordinate.treatment,
                task_id=coordinate.task_id,
                seed=coordinate.seed,
                parent_checkpoint=None,
                start_state="READY",
                host_role="spark_trainer",
                software_manifest_hash=digest_for(
                    {
                        "model": self.protocol.bindings["base_model_digest"],
                        "protocol": VARIATION_PROTOCOL_DIGEST,
                    }
                ),
            )
        for attempt in range(1, MAX_CANDIDATE_ATTEMPTS + 1):
            candidate_id = _main_candidate_id(
                self.protocol,
                coordinate,
                run_id=run_id,
                attempt=attempt,
                parent=None,
            )
            provisional = CandidateContext(
                campaign_id=self.protocol.campaign_id,
                run_id=run_id,
                seed=coordinate.seed,
                arm_id=coordinate.treatment,
                task_id=coordinate.task_id,
                family_id=task["family_id"],
                public_locus=task["public_locus"],
                public_rule_id=task["public_rule_id"],
                attempt_index=attempt,
                parent_candidate_id=None,
                retrieval_records=tuple(),
                retrieval_digest=digest_for([]),
                model_digest=self.protocol.bindings["base_model_digest"],
                adapter_digest=None,
                prompt_digest=GENESIS_HASH,
                task_statement="Repair the bounded {} task at {}.".format(
                    task["family_id"], task["public_locus"]
                ),
                initial_source=source.decode("utf-8"),
                initial_source_digest=digest_bytes(source),
                response_contract="source-only-v1",
                response_contract_digest=SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST,
                generation_profile_digest=self.inputs.generation_profile_digest,
            )
            rendered = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": render_candidate_prompt(
                            provisional,
                            response_contract="source-only-v1",
                        ),
                    }
                ],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            ).encode("utf-8")
            context = replace(provisional, prompt_digest=digest_bytes(rendered))
            invalid = b"this is not valid Python source !!!"
            failure = CandidateGenerationFailureEvidence(
                stage="RESPONSE_CONTRACT",
                response_contract="source-only-v1",
                rendered_prompt=rendered,
                rendered_prompt_digest=digest_bytes(rendered),
                decoded_model_response=invalid,
                decoded_model_response_digest=digest_bytes(invalid),
                contract_response=invalid,
                contract_response_digest=digest_bytes(invalid),
                error_code="VariationDependencyError",
            )
            private_store.record_generation_start(candidate_id=candidate_id, context=context)
            private_store.record_generation_failure(
                candidate_id=candidate_id,
                context=context,
                evidence=failure,
            )
        summaries = private_store.source_contract_failures(
            run_id=run_id,
            task_id=coordinate.task_id,
            arm_id=coordinate.treatment,
        )
        exhausted = SourceContractBudgetExhausted(
            run_id=run_id,
            failure_count=len(summaries),
            last_failure_digest=summaries[-1]["record_digest"],
        )
        reader = AuthoritativeMainEvidenceReader(
            protocol=self.protocol,
            deployment=self.deployment,
            evaluator_public_key=signer.public_key_raw,
            tokenizer=tokenizer,
        )

        class ExhaustingRunner:
            def __init__(self):
                self.context = type(
                    "Context",
                    (),
                    {"root": runtime_root, "protocol": self_protocol},
                )()

            def __call__(self, _coordinate):
                raise exhausted

        self_protocol = self.protocol
        runner = ProductionMainCoordinateRunner(ExhaustingRunner(), reader)
        dispatcher = ProductionCoordinateDispatcher(
            protocol=self.protocol,
            deployment=self.deployment,
            runtime_root=runtime_root,
            main_runner=runner,
            shock_runner=lambda _coordinate: None,
            main_evidence_reader=reader,
        )
        operation_store = CoordinateOperationStore(self.root / "all-invalid-ops", self.protocol)
        mapping = coordinate.to_dict(self.protocol.digest, self.protocol.campaign_id)
        mapping["idempotency_key"] = operation_store.idempotency_key(coordinate.coordinate_id)
        observation = dispatcher(mapping)
        self.assertEqual(observation["result"]["status"], "BUDGET_EXHAUSTED")
        self.assertEqual(
            observation["result"]["costs"]["candidate_attempts"],
            MAX_CANDIDATE_ATTEMPTS,
        )
        self.assertEqual(observation["result"]["verdict_receipts_required"], 0)
        self.assertTrue(observation["result"]["signature_valid"])
        verifier = ProductionObservationVerifier(
            protocol=self.protocol,
            deployment=self.deployment,
            trainer_sources=self.sources,
            evaluator_public_key=signer.public_key_raw,
            tokenizer=tokenizer,
        )
        coordinate_value = coordinate.to_dict(self.protocol.digest, self.protocol.campaign_id)
        verified = verifier(coordinate_value, observation)
        self.assertEqual(verified.result, observation["result"])

        first_failure = observation["evidence_bundle"]["generation_failures"][0]
        self.assertEqual(
            first_failure["generation_record"]["replay_error_chain"],
            ["VariationDependencyError", "SyntaxError"],
        )
        persisted_candidate = first_failure["candidate_id"]
        persisted_intent = json.loads(
            (private_store.generation_intents / (persisted_candidate + ".json")).read_text(
                encoding="utf-8"
            )
        )
        persisted_record = json.loads(
            (private_store.generation_records / (persisted_candidate + ".json")).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            persisted_intent["replay_error_chain"],
            ["VariationDependencyError", "SyntaxError"],
        )
        self.assertEqual(
            persisted_record["replay_error_chain"],
            ["VariationDependencyError", "SyntaxError"],
        )

        chain_tampered_bundle = json.loads(json.dumps(observation["evidence_bundle"]))
        chain_tampered_failure = chain_tampered_bundle["generation_failures"][0]
        chain_tampered_failure["generation_record"]["replay_error_chain"] = [
            "VariationDependencyError"
        ]
        chain_tampered_failure["generation_record_digest"] = digest_bytes(
            canonical_bytes(chain_tampered_failure["generation_record"])
        )
        unsigned_chain_tampered = dict(chain_tampered_bundle)
        unsigned_chain_tampered.pop("evidence_bundle_digest")
        chain_tampered_bundle["evidence_bundle_digest"] = digest_for(
            unsigned_chain_tampered
        )
        chain_tampered_observation = _observation(
            protocol=self.protocol,
            deployment=self.deployment,
            coordinate=coordinate,
            result=observation["result"],
            evidence_bundle=chain_tampered_bundle,
        )
        with self.assertRaisesRegex(HeldoutProtocolError, "failure chain changed"):
            verifier(coordinate_value, chain_tampered_observation)

        substituted_bundle = json.loads(json.dumps(observation["evidence_bundle"]))
        substituted_bundle["generation_failures"].pop()
        unsigned = dict(substituted_bundle)
        unsigned.pop("evidence_bundle_digest")
        substituted_bundle["evidence_bundle_digest"] = digest_for(unsigned)
        substituted = _observation(
            protocol=self.protocol,
            deployment=self.deployment,
            coordinate=coordinate,
            result=observation["result"],
            evidence_bundle=substituted_bundle,
        )
        with self.assertRaisesRegex(HeldoutProtocolError, "bounded trajectory"):
            verifier(coordinate_value, substituted)

        terminal_candidate = str(summaries[-1]["candidate_id"])
        terminal_record = json.loads(
            (
                private_store.generation_records / (terminal_candidate + ".json")
            ).read_text(encoding="utf-8")
        )
        for missing_kind in ("record", "artifact"):
            with self.subTest(read_only_terminal_missing=missing_kind):
                copied_root = self.root / ("all-invalid-missing-" + missing_kind)
                shutil.copytree(coordinate_root, copied_root)
                copied_store = PrivateTrajectoryStore(copied_root / "private")
                if missing_kind == "record":
                    _unlink_immutable_fixture(
                        copied_store.generation_records / (terminal_candidate + ".json")
                    )
                else:
                    artifact_digest = terminal_record["contract_response_digest"]
                    _unlink_immutable_fixture(
                        copied_store.artifacts.root
                        / "blobs"
                        / "sha256"
                        / artifact_digest[:2]
                        / artifact_digest[2:4]
                        / artifact_digest
                    )
                before = _tree_snapshot(copied_root)
                with self.assertRaises((HeldoutProtocolError, VariationCheckpointError)):
                    reader.record_source_exhaustion(
                        coordinate,
                        exhausted,
                        copied_store,
                    )
                self.assertEqual(_tree_snapshot(copied_root), before)

    def test_genuine_main_bundle_reconstructs_exact_trajectory_and_rejects_tampering(self):
        short_parent = Path(__file__).resolve().parents[2] if os.name == "nt" else None
        with tempfile.TemporaryDirectory(prefix="egv-main-e2e-", dir=short_parent) as scratch:
            case = _shock_engine_tests.ShockEngineTests(
                methodName="test_exact_six_pre_attempts_continue_after_early_promotion"
            )
            case.setUp()
            try:
                campaign_id = "egv-campaign-cafebabedeadbeef"
                records = ordered_public_heldout_task_records(case.corpus)
                mixed_task_id = records[-1]["template_id"]
                responder = case.command.read_text(encoding="utf-8")
                mixed_condition = "request['task_id'] == {!r}".format(mixed_task_id)
                responder = responder.replace(
                    '"decision": "PASS"',
                    '"decision": ("FAIL" if {} else "PASS")'.format(mixed_condition),
                ).replace(
                    '"diagnostic_enum": "PASS"',
                    '"diagnostic_enum": ("WRONG_OUTPUT" if {} else "PASS")'.format(
                        mixed_condition
                    ),
                ).replace(
                    '"disposition": "PROMOTED"',
                    '"disposition": ("REJECTED" if {} else "PROMOTED")'.format(
                        mixed_condition
                    ),
                )
                case.command.write_text(responder, encoding="utf-8")
                authority = AuthorityPolicy.candidate_execution().digest
                revision = "main-hidden-evaluator-v1"
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
                    "evaluator_public_key_digest": hashlib.sha256(
                        case.public_key.read_bytes()
                    ).hexdigest(),
                }
                service = case.root / "main-production-service.json"
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
                    "evaluator_revision": revision,
                    "generation_profile_digest": case.generator.generation_profile_digest,
                    "response_contract_digest": SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST,
                    "heldout_evaluator_execution_mode": HELDOUT_EVALUATOR_EXECUTION_MODE,
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
                    if item.phase != SHOCK_PHASE
                    and item.treatment == "E"
                    and item.task_id != mixed_task_id
                )
                case.generator.tokenizer.function_name = inputs.tasks[coordinate.task_id][
                    "public_locus"
                ].rsplit(":", 1)[-1]

                def builder(**kwargs):
                    evaluator = RemoteControllerEvaluationGateway(
                        ledger=kwargs["ledger"],
                        manifest_path=service,
                        public_key_path=case.public_key,
                        command=case.command,
                    )
                    return BoundedCandidateLoop(
                        ledger=kwargs["ledger"],
                        evaluator=evaluator,
                        generator=case.generator,
                        isolation=kwargs["isolation"],
                        workspace_root=kwargs["isolation"].root,
                        campaign_id=protocol.campaign_id,
                        source_commit="a" * 40,
                        model_revision=MODEL_REVISION,
                        model_digest=protocol.bindings["base_model_digest"],
                        data_manifest_digest=protocol.bindings["data_manifest_digest"],
                        policy_digest=protocol.bindings["policy_manifest_digest"],
                        arm_id=coordinate.treatment,
                        max_attempts=MAX_CANDIDATE_ATTEMPTS,
                        adapter_digest=protocol.bindings["adapter_digest"],
                        adapter_artifact=case.generator.adapter_artifact,
                        seed_set=protocol.seeds,
                        private_store=kwargs["private_store"],
                        initial_source=kwargs["initial_source"],
                        response_contract_digest=SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST,
                        generation_profile_digest=inputs.generation_profile_digest,
                    )

                if os.name == "nt":
                    runtime_root = Path(
                        tempfile.mkdtemp(prefix="egv-e-", dir=Path.cwd().anchor)
                    )
                    self.addCleanup(shutil.rmtree, runtime_root, ignore_errors=True)
                else:
                    runtime_root = Path(scratch) / "runtime"
                reader = AuthoritativeMainEvidenceReader(
                    protocol=protocol,
                    deployment=deployment,
                    evaluator_public_key=case.signer.public_key_raw,
                    tokenizer=case.generator.tokenizer,
                )
                heldout_runner = HeldoutCoordinateRunner(
                    HeldoutRuntimeContext(
                        protocol=protocol,
                        trainer_inputs=inputs,
                        trainer_sources=sources,
                        root=runtime_root,
                        base_loop_builder=builder,
                        adapter_loop_builder=builder,
                        evidence_reader=reader,
                    )
                )
                runner = ProductionMainCoordinateRunner(heldout_runner, reader)
                dispatcher = ProductionCoordinateDispatcher(
                    protocol=protocol,
                    deployment=deployment,
                    runtime_root=runtime_root,
                    main_runner=runner,
                    shock_runner=lambda _coordinate: None,
                    main_evidence_reader=reader,
                )
                operations = CoordinateOperationStore(Path(scratch) / "operations", protocol)
                mapping = coordinate.to_dict(protocol.digest, protocol.campaign_id)
                mapping["idempotency_key"] = operations.idempotency_key(coordinate.coordinate_id)
                observation = dispatcher(mapping)
                verifier = ProductionObservationVerifier(
                    protocol=protocol,
                    deployment=deployment,
                    trainer_sources=sources,
                    evaluator_public_key=case.signer.public_key_raw,
                    tokenizer=case.generator.tokenizer,
                )
                coordinate_value = coordinate.to_dict(protocol.digest, protocol.campaign_id)
                verified = verifier(coordinate_value, observation)
                self.assertEqual(verified.result, observation["result"])
                self.assertEqual(observation["result"]["status"], "COMPLETED")
                self.assertEqual(observation["result"]["costs"]["candidate_attempts"], 1)

                def reseal(bundle):
                    unsigned_bundle = dict(bundle)
                    unsigned_bundle.pop("evidence_bundle_digest", None)
                    return {
                        **unsigned_bundle,
                        "evidence_bundle_digest": digest_for(unsigned_bundle),
                    }

                def finalize_main_ledger_bundle(tampered, ledger_export, head):
                    records = [json.loads(line) for line in ledger_export.splitlines()]
                    attempt_events = {
                        record["subject_id"]: record
                        for record in records
                        if record.get("record_type") == "EVENT"
                        and record.get("event_type") == "VARIATION_ATTEMPT"
                    }
                    for attempt in tampered["report"]["attempts"]:
                        attempt["ledger_head_hash"] = attempt_events[
                            attempt["candidate_id"]
                        ]["event_hash"]
                    tampered["report"]["ledger_head_hash"] = head
                    tampered["report"]["ledger_integrity"]["ledger_head_hash"] = head
                    report_value = production_module._report_from_mapping(
                        tampered["report"]
                    )
                    completed = []
                    checkpoint_records = {
                        record["checkpoint"]["last_durable_event_id"]: record[
                            "checkpoint"
                        ]
                        for record in records
                        if record.get("record_type") == "CHECKPOINT"
                    }
                    for attempt in report_value.attempts:
                        completed.append(attempt)
                        event = attempt_events[attempt.candidate_id]
                        status = (
                            report_value.terminal_status
                            if attempt is report_value.attempts[-1]
                            else "RUNNING"
                        )
                        state = {
                            "attempts": [item.to_dict() for item in completed],
                            "run_id": report_value.run_id,
                            "arm_id": coordinate.treatment,
                            "task_id": coordinate.task_id,
                            "status": status,
                        }
                        artifacts_value = {
                            "artifacts": [
                                {
                                    "attempt_index": item.attempt_index,
                                    "candidate_id": item.candidate_id,
                                    "candidate_artifact_digest": (
                                        item.candidate_artifact_digest
                                    ),
                                }
                                for item in completed
                            ]
                        }
                        projection = digest_for(
                            {
                                "ledger_head_event_id": event["event_id"],
                                "ledger_head_hash": event["event_hash"],
                                "arm_id": coordinate.treatment,
                                "run_id": report_value.run_id,
                            }
                        )
                        expected_checkpoint = VariationCheckpoint(
                            campaign_id=protocol.campaign_id,
                            run_id=report_value.run_id,
                            arm_id=coordinate.treatment,
                            task_id=coordinate.task_id,
                            seed=coordinate.seed,
                            attempt_index=attempt.attempt_index,
                            last_candidate_id=attempt.candidate_id,
                            ledger_head_event_id=event["event_id"],
                            ledger_head_hash=event["event_hash"],
                            projection_generation=projection,
                            artifact_manifest_hash=digest_for(artifacts_value),
                            protocol_digest=VARIATION_PROTOCOL_DIGEST,
                            model_digest=protocol.bindings["base_model_digest"],
                            adapter_digest=report_value.adapter_digest,
                            retrieval_policy_digest=retrieval_policy(
                                report_value.retrieval_policy
                            ).digest,
                            state_digest=digest_for(state),
                            status=status,
                        )
                        checkpoint = checkpoint_records[event["event_id"]]
                        checkpoint["checkpoint_id"] = expected_checkpoint.digest
                        checkpoint["ledger_hash"] = event["event_hash"]
                        checkpoint["projection_generation"] = projection
                        checkpoint["artifact_manifest_hash"] = (
                            expected_checkpoint.artifact_manifest_hash
                        )
                    tampered["ledger_export"] = "".join(
                        canonical_json(record) + "\n" for record in records
                    )
                    tampered["ledger_export_digest"] = digest_bytes(
                        tampered["ledger_export"].encode("utf-8")
                    )
                    tampered["ledger_head_digest"] = head
                    return reseal(tampered)

                for raw_export_mode in ("unknown-field", "duplicate-event"):
                    with self.subTest(main_raw_ledger_export=raw_export_mode):
                        raw_tampered = json.loads(
                            json.dumps(observation["evidence_bundle"])
                        )
                        raw_tampered["ledger_export"] = _raw_ledger_export_tamper(
                            raw_tampered["ledger_export"], raw_export_mode
                        )
                        raw_tampered["ledger_export_digest"] = digest_bytes(
                            raw_tampered["ledger_export"].encode("utf-8")
                        )
                        with self.assertRaisesRegex(
                            HeldoutProtocolError,
                            "export schema|replay export",
                        ):
                            verifier(
                                coordinate_value,
                                _observation(
                                    protocol=protocol,
                                    deployment=deployment,
                                    coordinate=coordinate,
                                    result=observation["result"],
                                    evidence_bundle=reseal(raw_tampered),
                                ),
                            )

                receipt_order = json.loads(json.dumps(observation["evidence_bundle"]))
                swapped_export, swapped_head = _swap_first_receipt_events(
                    receipt_order["ledger_export"]
                )
                receipt_order = finalize_main_ledger_bundle(
                    receipt_order, swapped_export, swapped_head
                )
                receipt_order_observation = _observation(
                    protocol=protocol,
                    deployment=deployment,
                    coordinate=coordinate,
                    result=observation["result"],
                    evidence_bundle=receipt_order,
                )
                with patch(
                    "egv.experiment.production._require_signed_receipt_event_order",
                    return_value=None,
                ):
                    old_order_verified = verifier(
                        coordinate_value, receipt_order_observation
                    )
                self.assertEqual(old_order_verified.result, observation["result"])
                with self.assertRaisesRegex(HeldoutProtocolError, "receipt event order"):
                    verifier(coordinate_value, receipt_order_observation)

                forged_revision = "forged-independent-evaluator-revision-v999"
                revision_tampered = json.loads(
                    json.dumps(observation["evidence_bundle"])
                )
                revision_records = [
                    json.loads(line)
                    for line in revision_tampered["ledger_export"].splitlines()
                ]
                for record in revision_records:
                    if (
                        record.get("record_type") == "EVENT"
                        and record.get("event_type") == "VERDICT"
                    ):
                        payload = dict(record["payload"])
                        payload["evaluator_revision"] = forged_revision
                        payload_bytes = canonical_bytes(payload)
                        record["payload"] = payload
                        record["payload_hash"] = digest_bytes(payload_bytes)
                        record["payload_json"] = payload_bytes.decode("utf-8")
                revision_export, revision_head = _reseal_ledger_records(
                    revision_records
                )
                revision_tampered = finalize_main_ledger_bundle(
                    revision_tampered, revision_export, revision_head
                )
                revision_observation = _observation(
                    protocol=protocol,
                    deployment=deployment,
                    coordinate=coordinate,
                    result=observation["result"],
                    evidence_bundle=revision_tampered,
                )
                original_getitem = SealedHeldoutDeploymentManifest.__getitem__

                def old_self_trusting_revision(manifest, key):
                    if manifest is deployment and key == "evaluator_revision":
                        return forged_revision
                    return original_getitem(manifest, key)

                with patch.object(
                    SealedHeldoutDeploymentManifest,
                    "__getitem__",
                    new=old_self_trusting_revision,
                ):
                    old_trusting_verified = verifier(
                        coordinate_value, revision_observation
                    )
                self.assertEqual(old_trusting_verified.result, observation["result"])
                with self.assertRaisesRegex(HeldoutProtocolError, "verdict"):
                    verifier(coordinate_value, revision_observation)

                duplicate = json.loads(json.dumps(observation["evidence_bundle"]))
                duplicate["report"]["attempts"].append(
                    dict(duplicate["report"]["attempts"][0])
                )
                duplicate["token_materials"].append(dict(duplicate["token_materials"][0]))
                with self.assertRaisesRegex(HeldoutProtocolError, "duplicated|order"):
                    verifier(
                        coordinate_value,
                        _observation(
                            protocol=protocol,
                            deployment=deployment,
                            coordinate=coordinate,
                            result=observation["result"],
                            evidence_bundle=reseal(duplicate),
                        ),
                    )

                source_tamper = json.loads(json.dumps(observation["evidence_bundle"]))
                source_tamper["token_materials"][0]["candidate_source_b64"] = (
                    base64.urlsafe_b64encode(b"def substituted():\n    return 0")
                    .decode("ascii")
                    .rstrip("=")
                )
                with self.assertRaisesRegex(HeldoutProtocolError, "source|proposal"):
                    verifier(
                        coordinate_value,
                        _observation(
                            protocol=protocol,
                            deployment=deployment,
                            coordinate=coordinate,
                            result=observation["result"],
                            evidence_bundle=reseal(source_tamper),
                        ),
                    )

                context_tamper = json.loads(json.dumps(observation["evidence_bundle"]))
                record = context_tamper["token_materials"][0]["generation_record"]
                record["context"]["run_id"] = "run-cross-coordinate"
                context_tamper["token_materials"][0]["generation_record_digest"] = digest_bytes(
                    canonical_json(record).encode("utf-8")
                )
                with self.assertRaisesRegex(HeldoutProtocolError, "context|prompt"):
                    verifier(
                        coordinate_value,
                        _observation(
                            protocol=protocol,
                            deployment=deployment,
                            coordinate=coordinate,
                            result=observation["result"],
                            evidence_bundle=reseal(context_tamper),
                        ),
                    )

                raw_omission = json.loads(json.dumps(observation["evidence_bundle"]))
                raw_omission["token_materials"][0].pop("raw_generation")
                with self.assertRaisesRegex(HeldoutProtocolError, "closed object"):
                    verifier(
                        coordinate_value,
                        _observation(
                            protocol=protocol,
                            deployment=deployment,
                            coordinate=coordinate,
                            result=observation["result"],
                            evidence_bundle=reseal(raw_omission),
                        ),
                    )

                raw_tamper = json.loads(json.dumps(observation["evidence_bundle"]))
                raw_material = raw_tamper["token_materials"][0]["raw_generation"]
                raw_material["contract_response_b64"] = base64.urlsafe_b64encode(
                    b"forged private contract response"
                ).decode("ascii").rstrip("=")
                raw_body = dict(raw_material)
                raw_body.pop("material_digest")
                raw_material["material_digest"] = digest_for(raw_body)
                with self.assertRaisesRegex(HeldoutProtocolError, "raw generation bytes"):
                    verifier(
                        coordinate_value,
                        _observation(
                            protocol=protocol,
                            deployment=deployment,
                            coordinate=coordinate,
                            result=observation["result"],
                            evidence_bundle=reseal(raw_tamper),
                        ),
                    )

                retrieval_tamper = json.loads(json.dumps(observation["evidence_bundle"]))
                retrieval_record = retrieval_tamper["token_materials"][0][
                    "generation_record"
                ]
                retrieval_record["context"]["retrieval_records"] = [
                    {
                        "event_id": "forged-retrieval",
                        "event_type": "CANDIDATE",
                        "subject_id": "forged-subject",
                        "task_id": coordinate.task_id,
                        "recorded_disposition": "PROMOTED",
                    }
                ]
                retrieval_record["context"]["retrieval_digest"] = digest_for(
                    retrieval_record["context"]["retrieval_records"]
                )
                retrieval_tamper["token_materials"][0][
                    "generation_record_digest"
                ] = digest_bytes(canonical_json(retrieval_record).encode("utf-8"))
                with self.assertRaisesRegex(HeldoutProtocolError, "context|retrieval"):
                    verifier(
                        coordinate_value,
                        _observation(
                            protocol=protocol,
                            deployment=deployment,
                            coordinate=coordinate,
                            result=observation["result"],
                            evidence_bundle=reseal(retrieval_tamper),
                        ),
                    )

                generation_tamper = json.loads(json.dumps(observation["evidence_bundle"]))
                generation_record = generation_tamper["token_materials"][0][
                    "generation_record"
                ]
                generation_record["rendered_prompt_digest"] = digest_for(
                    "forged-rendered-prompt"
                )
                generation_tamper["token_materials"][0][
                    "generation_record_digest"
                ] = digest_bytes(canonical_json(generation_record).encode("utf-8"))
                with self.assertRaisesRegex(HeldoutProtocolError, "context|prompt"):
                    verifier(
                        coordinate_value,
                        _observation(
                            protocol=protocol,
                            deployment=deployment,
                            coordinate=coordinate,
                            result=observation["result"],
                            evidence_bundle=reseal(generation_tamper),
                        ),
                    )

                cross_coordinate = json.loads(json.dumps(observation["evidence_bundle"]))
                cross_coordinate["report"]["run_id"] = "egv-run-cross-coordinate"
                with self.assertRaisesRegex(HeldoutProtocolError, "coordinate"):
                    verifier(
                        coordinate_value,
                        _observation(
                            protocol=protocol,
                            deployment=deployment,
                            coordinate=coordinate,
                            result=observation["result"],
                            evidence_bundle=reseal(cross_coordinate),
                        ),
                    )

                ledger_tamper = json.loads(json.dumps(observation["evidence_bundle"]))
                with tempfile.TemporaryDirectory(prefix="egv-main-ledger-") as replay_root:
                    replay = EvidenceLedger.replay_jsonl(
                        ledger_tamper["ledger_export"],
                        Path(replay_root) / "ledger.sqlite3",
                    )
                    replay.append_event(
                        "EXTRA_DESCENDANT",
                        {"schema_version": "forged-extra-v1"},
                        campaign_id=protocol.campaign_id,
                        run_id=_main_run_id(protocol, coordinate),
                        task_id=coordinate.task_id,
                        subject_id="forged-extra-subject",
                        source_class="GENERATOR",
                        disposition="OBSERVED",
                        idempotency_key="forged-extra-event",
                    )
                    ledger_tamper["ledger_export"] = replay.export_jsonl()
                    ledger_tamper["ledger_export_digest"] = digest_bytes(
                        ledger_tamper["ledger_export"].encode("utf-8")
                    )
                    ledger_tamper["ledger_head_digest"] = replay.ledger_head_hash()
                    ledger_tamper["report"]["ledger_head_hash"] = replay.ledger_head_hash()
                    ledger_tamper["report"]["ledger_integrity"] = replay.verify_integrity()
                    replay.close()
                with self.assertRaisesRegex(HeldoutProtocolError, "unaccounted"):
                    verifier(
                        coordinate_value,
                        _observation(
                            protocol=protocol,
                            deployment=deployment,
                            coordinate=coordinate,
                            result=observation["result"],
                            evidence_bundle=reseal(ledger_tamper),
                        ),
                    )

                def assert_ledger_mutation_rejected(mutate, pattern):
                    tampered = json.loads(json.dumps(observation["evidence_bundle"]))
                    with tempfile.TemporaryDirectory(prefix="egv-main-ledger-") as replay_root:
                        replay = EvidenceLedger.replay_jsonl(
                            tampered["ledger_export"],
                            Path(replay_root) / "ledger.sqlite3",
                        )
                        mutate(replay)
                        tampered["ledger_export"] = replay.export_jsonl()
                        tampered["ledger_export_digest"] = digest_bytes(
                            tampered["ledger_export"].encode("utf-8")
                        )
                        tampered["ledger_head_digest"] = replay.ledger_head_hash()
                        tampered["report"]["ledger_head_hash"] = replay.ledger_head_hash()
                        tampered["report"]["ledger_integrity"] = replay.verify_integrity()
                        replay.close()
                    with self.assertRaisesRegex(HeldoutProtocolError, pattern):
                        verifier(
                            coordinate_value,
                            _observation(
                                protocol=protocol,
                                deployment=deployment,
                                coordinate=coordinate,
                                result=observation["result"],
                                evidence_bundle=reseal(tampered),
                            ),
                        )

                def add_extra_run(replay):
                    replay.create_run(
                        "egv-run-" + digest_for("forged-run")[:40],
                        campaign_id=protocol.campaign_id,
                        arm=coordinate.treatment,
                        task_id=coordinate.task_id,
                        seed=coordinate.seed,
                        parent_checkpoint=None,
                        start_state="READY",
                        host_role="spark_trainer",
                        software_manifest_hash=digest_for("forged-software"),
                    )

                assert_ledger_mutation_rejected(add_extra_run, "campaign or run inventory")

                def add_extra_candidate(replay):
                    replay.append_candidate(
                        "egv-candidate-" + digest_for("forged-candidate"),
                        campaign_id=protocol.campaign_id,
                        run_id=_main_run_id(protocol, coordinate),
                        task_id=coordinate.task_id,
                        parent_candidate_id=None,
                        mutation_family="forged",
                        patch_hash=digest_for("forged-patch"),
                        requested_authority="EXECUTE_CANDIDATE",
                        prompt_hash=digest_for("forged-prompt"),
                        model_hash=protocol.bindings["base_model_digest"],
                        adapter_hash=protocol.bindings["adapter_digest"],
                        metadata={"schema_version": "forged-candidate-v1"},
                    )

                assert_ledger_mutation_rejected(add_extra_candidate, "candidate projection inventory")

                def add_extra_edge(replay):
                    replay.append_dependency(
                        protocol.campaign_id,
                        observation["evidence_bundle"]["report"]["attempts"][0][
                            "candidate_id"
                        ],
                        edge_type="FORGED_EDGE",
                        campaign_id=protocol.campaign_id,
                        run_id=_main_run_id(protocol, coordinate),
                        task_id=coordinate.task_id,
                        idempotency_key="forged-main-edge",
                    )

                assert_ledger_mutation_rejected(add_extra_edge, "dependency inventory")

                def assert_event_envelope_substitution_rejected(
                    event_type,
                    field,
                    value,
                    pattern,
                ):
                    tampered = json.loads(json.dumps(observation["evidence_bundle"]))
                    tampered["ledger_export"], head = _substitute_event_envelope(
                        tampered["ledger_export"],
                        event_type,
                        field,
                        value,
                    )
                    tampered["ledger_export_digest"] = digest_bytes(
                        tampered["ledger_export"].encode("utf-8")
                    )
                    tampered["ledger_head_digest"] = head
                    tampered["report"]["ledger_head_hash"] = head
                    tampered["report"]["ledger_integrity"]["ledger_head_hash"] = head
                    with self.assertRaisesRegex(HeldoutProtocolError, pattern):
                        verifier(
                            coordinate_value,
                            _observation(
                                protocol=protocol,
                                deployment=deployment,
                                coordinate=coordinate,
                                result=observation["result"],
                                evidence_bundle=reseal(tampered),
                            ),
                        )

                for event_type, field, value, pattern in (
                    ("CAMPAIGN", "source_class", "GENERATOR", "campaign/run event"),
                    ("RUN", "idempotency_key", "forged-run-envelope", "campaign/run event"),
                    (
                        "CANDIDATE",
                        "evaluator_identity",
                        "forged-evaluator",
                        "candidate ledger projection",
                    ),
                    (
                        "RECEIPT",
                        "disposition",
                        "OBSERVED",
                        "main attempt ledger event|receipt event projection",
                    ),
                    (
                        "VERDICT",
                        "idempotency_key",
                        "forged-verdict-envelope",
                        "main attempt ledger event|verdict event projection",
                    ),
                    (
                        "EFFECT_RECEIPT",
                        "source_class",
                        "GENERATOR",
                        "main attempt ledger event|effect event projection",
                    ),
                ):
                    with self.subTest(main_event_envelope=event_type):
                        assert_event_envelope_substitution_rejected(
                            event_type,
                            field,
                            value,
                            pattern,
                        )

                invalid_coordinate = next(
                    item
                    for item in protocol.coordinates
                    if item.phase != SHOCK_PHASE
                    and item.treatment == "E"
                    and item.coordinate_id != coordinate.coordinate_id
                    and item.task_id != mixed_task_id
                )
                case.generator.tokenizer.function_name = inputs.tasks[
                    invalid_coordinate.task_id
                ]["public_locus"].rsplit(":", 1)[-1]
                first_invalid_call = case.generator.tokenizer.decode_calls + 1
                case.generator.tokenizer.invalid_decode_calls.update(
                    range(first_invalid_call, first_invalid_call + MAX_CANDIDATE_ATTEMPTS)
                )
                invalid_mapping = invalid_coordinate.to_dict(
                    protocol.digest, protocol.campaign_id
                )
                invalid_mapping["idempotency_key"] = operations.idempotency_key(
                    invalid_coordinate.coordinate_id
                )
                invalid_observation = dispatcher(invalid_mapping)
                invalid_coordinate_value = invalid_coordinate.to_dict(
                    protocol.digest, protocol.campaign_id
                )
                invalid_verified = verifier(
                    invalid_coordinate_value,
                    invalid_observation,
                )
                self.assertEqual(invalid_verified.result, invalid_observation["result"])
                self.assertEqual(invalid_observation["result"]["status"], "BUDGET_EXHAUSTED")
                self.assertEqual(
                    invalid_observation["result"]["costs"]["candidate_attempts"],
                    MAX_CANDIDATE_ATTEMPTS,
                )
                self.assertEqual(invalid_observation["result"]["verdict_receipts_required"], 0)
                self.assertTrue(invalid_observation["result"]["signature_valid"])

                mixed_coordinate = next(
                    item
                    for item in protocol.coordinates
                    if item.phase != SHOCK_PHASE
                    and item.treatment == "E"
                    and item.task_id == mixed_task_id
                )
                case.generator.tokenizer.function_name = inputs.tasks[
                    mixed_coordinate.task_id
                ]["public_locus"].rsplit(":", 1)[-1]
                mixed_pattern = (
                    "SUCCESS",
                    "FAILED",
                    "SUCCESS",
                    "FAILED",
                    "FAILED",
                    "SUCCESS",
                    "FAILED",
                    "FAILED",
                    "SUCCESS",
                    "FAILED",
                    "SUCCESS",
                    "FAILED",
                )
                first_mixed_call = case.generator.tokenizer.decode_calls + 1
                case.generator.tokenizer.invalid_decode_calls.update(
                    first_mixed_call + index
                    for index, status in enumerate(mixed_pattern)
                    if status == "FAILED"
                )
                mixed_mapping = mixed_coordinate.to_dict(
                    protocol.digest, protocol.campaign_id
                )
                mixed_mapping["idempotency_key"] = operations.idempotency_key(
                    mixed_coordinate.coordinate_id
                )
                original_record_source_exhaustion = reader.record_source_exhaustion
                materialization_calls = []

                def crash_before_first_mixed_materialization(*args, **kwargs):
                    materialization_calls.append(True)
                    if len(materialization_calls) == 1:
                        raise RuntimeError("crash before mixed terminal bundle materialization")
                    return original_record_source_exhaustion(*args, **kwargs)

                with patch.object(
                    reader,
                    "record_source_exhaustion",
                    side_effect=crash_before_first_mixed_materialization,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "crash before mixed terminal bundle materialization",
                    ):
                        dispatcher(mixed_mapping)
                    crashed_model_calls = case.model.calls
                    crashed_effect_count = int(
                        (case.remote_state / "effect-count.txt").read_text(encoding="ascii")
                    )
                    self.assertFalse(
                        (
                            runtime_root
                            / mixed_coordinate.coordinate_id
                            / "main-evidence.json"
                        ).exists()
                    )
                    mixed_observation = dispatcher(mixed_mapping)
                self.assertEqual(materialization_calls, [True, True])
                self.assertEqual(case.model.calls, crashed_model_calls)
                self.assertEqual(
                    int((case.remote_state / "effect-count.txt").read_text(encoding="ascii")),
                    crashed_effect_count,
                )
                mixed_coordinate_value = mixed_coordinate.to_dict(
                    protocol.digest, protocol.campaign_id
                )
                mixed_verified = verifier(mixed_coordinate_value, mixed_observation)
                self.assertEqual(mixed_verified.result, mixed_observation["result"])
                self.assertEqual(mixed_observation["result"]["status"], "BUDGET_EXHAUSTED")
                self.assertIsNone(mixed_observation["result"]["success"])
                self.assertEqual(
                    mixed_observation["result"]["costs"]["candidate_attempts"],
                    MAX_CANDIDATE_ATTEMPTS,
                )
                self.assertEqual(mixed_observation["result"]["verdict_receipts_required"], 5)
                self.assertEqual(mixed_observation["result"]["public_replay_decisions"], 5)
                self.assertEqual(mixed_observation["result"]["evidence_opportunities"], 11)
                merged = sorted(
                    [
                        (
                            item["generation_record"]["context"]["attempt_index"],
                            "SUCCESS",
                        )
                        for item in mixed_observation["evidence_bundle"]["token_materials"]
                    ]
                    + [
                        (
                            item["generation_record"]["context"]["attempt_index"],
                            "FAILED",
                        )
                        for item in mixed_observation["evidence_bundle"]["generation_failures"]
                    ]
                )
                self.assertEqual(tuple(status for _index, status in merged), mixed_pattern)
                mixed_calls = case.model.calls
                effect_count = int(
                    (case.remote_state / "effect-count.txt").read_text(encoding="ascii")
                )
                repeated_observation = dispatcher(mixed_mapping)
                self.assertEqual(
                    repeated_observation["evidence_bundle_digest"],
                    mixed_observation["evidence_bundle_digest"],
                )
                self.assertEqual(case.model.calls, mixed_calls)
                self.assertEqual(
                    int((case.remote_state / "effect-count.txt").read_text(encoding="ascii")),
                    effect_count,
                )

                mixed_promotion = json.loads(json.dumps(mixed_observation["evidence_bundle"]))
                mixed_promotion["report"]["attempts"][0]["disposition"] = "PROMOTED"
                with self.assertRaisesRegex(HeldoutProtocolError, "attempt|rejection|receipt"):
                    verifier(
                        mixed_coordinate_value,
                        _observation(
                            protocol=protocol,
                            deployment=deployment,
                            coordinate=mixed_coordinate,
                            result=mixed_observation["result"],
                            evidence_bundle=reseal(mixed_promotion),
                        ),
                    )

                mixed_failure_id = json.loads(json.dumps(mixed_observation["evidence_bundle"]))
                mixed_failure_id["generation_failures"][-1]["candidate_id"] = (
                    "egv-candidate-forged-terminal-failure"
                )
                with self.assertRaisesRegex(HeldoutProtocolError, "record|trajectory|identity"):
                    verifier(
                        mixed_coordinate_value,
                        _observation(
                            protocol=protocol,
                            deployment=deployment,
                            coordinate=mixed_coordinate,
                            result=mixed_observation["result"],
                            evidence_bundle=reseal(mixed_failure_id),
                        ),
                    )

                mixed_private = PrivateTrajectoryStore(
                    runtime_root / mixed_coordinate.coordinate_id / "private"
                )
                mixed_exhausted = SourceContractBudgetExhausted(
                    run_id=_main_run_id(protocol, mixed_coordinate),
                    failure_count=7,
                    last_failure_digest=mixed_observation["evidence_bundle"][
                        "generation_failures"
                    ][-1]["generation_record_digest"],
                )
                successful_material = mixed_observation["evidence_bundle"][
                    "token_materials"
                ][0]
                successful_candidate = successful_material["candidate_id"]
                successful_record = successful_material["generation_record"]
                for missing_kind in ("record", "artifact"):
                    with self.subTest(mixed_read_only_terminal_missing=missing_kind):
                        copied_root = Path(scratch) / (
                            "mr" if missing_kind == "record" else "ma"
                        )
                        copied_root.mkdir()
                        shutil.copytree(
                            runtime_root / mixed_coordinate.coordinate_id / "private",
                            copied_root / "private",
                        )
                        shutil.copy2(
                            runtime_root
                            / mixed_coordinate.coordinate_id
                            / "ledger.sqlite3",
                            copied_root / "ledger.sqlite3",
                        )
                        copied_store = PrivateTrajectoryStore(copied_root / "private")
                        if missing_kind == "record":
                            _unlink_immutable_fixture(
                                copied_store.generation_records
                                / (successful_candidate + ".json")
                            )
                        else:
                            artifact_digest = successful_record["contract_response_digest"]
                            _unlink_immutable_fixture(
                                copied_store.artifacts.root
                                / "blobs"
                                / "sha256"
                                / artifact_digest[:2]
                                / artifact_digest[2:4]
                                / artifact_digest
                            )
                        before = _tree_snapshot(copied_root)
                        with self.assertRaises(
                            (HeldoutProtocolError, VariationCheckpointError)
                        ):
                            reader.record_source_exhaustion(
                                mixed_coordinate,
                                mixed_exhausted,
                                copied_store,
                            )
                        self.assertEqual(_tree_snapshot(copied_root), before)
                with self.assertRaisesRegex(HeldoutProtocolError, "bounded mixed trajectory"):
                    reader.record_source_exhaustion(
                        mixed_coordinate,
                        SourceContractBudgetExhausted(
                            run_id=_main_run_id(protocol, mixed_coordinate),
                            failure_count=7,
                            last_failure_digest=digest_for("forged-final-failure"),
                        ),
                        mixed_private,
                    )
            finally:
                case.tearDown()

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
                        "evaluator_revision": revision,
                        "generation_profile_digest": case.generator.generation_profile_digest,
                        "response_contract_digest": SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST,
                        "heldout_evaluator_execution_mode": HELDOUT_EVALUATOR_EXECUTION_MODE,
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
                    case.generator.tokenizer.invalid_decode_calls.add(7)
                    factory = ProductionShockEngineFactory(
                        generator=case.generator,
                        evaluator_manifest=service,
                        evaluator_public_key=case.public_key,
                        evaluator_command=case.command,
                        evaluator_python_executable=Path(sys.executable).resolve(),
                        evaluator_python_digest=hashlib.sha256(
                            Path(sys.executable).resolve().read_bytes()
                        ).hexdigest(),
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
                    self.assertEqual(result["eligible_attempts"], 8)
                    self.assertEqual(result["verdict_receipts_required"], 7)
                    self.assertEqual(result["verdict_receipts_valid"], 7)
                    self.assertTrue(result["signature_valid"])
                    self.assertEqual(result["recovery_attempt"], 2)

                    event_records = [
                        json.loads(line)
                        for line in bundle["ledger_export"].splitlines()
                        if json.loads(line).get("record_type") == "EVENT"
                    ]
                    actual_event_counts = {}
                    for event in event_records:
                        event_type = event["event_type"]
                        actual_event_counts[event_type] = (
                            actual_event_counts.get(event_type, 0) + 1
                        )
                    self.assertEqual(
                        actual_event_counts,
                        {
                            "CAMPAIGN": 1,
                            "RUN": 2,
                            "CANDIDATE": 7,
                            "RECEIPT": 21,
                            "VERDICT": 7,
                            "EFFECT_RECEIPT": 7,
                            "CORRECTION": 1,
                            "DEPENDENCY": 28,
                            "SHOCK_ATTEMPT": 7,
                            "SHOCK_GENERATION_FAILURE": 1,
                            "SHOCK_PREMISE": 1,
                            "SHOCK_UNRELATED_ROOT": 1,
                            "SHOCK_UNRELATED_EVIDENCE": 1,
                            "SHOCK_CORRECTED_PREMISE": 1,
                            "SHOCK_CORRECTION_COMMIT": 1,
                            "SHOCK_POLICY_ACTIVATION": 1,
                        },
                    )

                    def reseal_shock(bundle_value):
                        unsigned_bundle = dict(bundle_value)
                        unsigned_bundle.pop("evidence_bundle_digest", None)
                        return {
                            **unsigned_bundle,
                            "evidence_bundle_digest": digest_for(unsigned_bundle),
                        }

                    for raw_export_mode in ("unknown-field", "duplicate-event"):
                        with self.subTest(shock_raw_ledger_export=raw_export_mode):
                            raw_tampered = json.loads(json.dumps(bundle))
                            raw_tampered["ledger_export"] = _raw_ledger_export_tamper(
                                raw_tampered["ledger_export"], raw_export_mode
                            )
                            raw_tampered["ledger_export_digest"] = digest_bytes(
                                raw_tampered["ledger_export"].encode("utf-8")
                            )
                            with self.assertRaisesRegex(
                                HeldoutProtocolError,
                                "export schema|replay export",
                            ):
                                verifier(
                                    coordinate_value,
                                    _observation(
                                        protocol=protocol,
                                        deployment=deployment,
                                        coordinate=coordinate,
                                        result=result,
                                        evidence_bundle=reseal_shock(raw_tampered),
                                    ),
                                )

                    duplicated_operation = json.loads(json.dumps(bundle))
                    duplicated_operation["operations"].append(
                        dict(duplicated_operation["operations"][-1])
                    )
                    with self.assertRaisesRegex(HeldoutProtocolError, "duplicate"):
                        verifier(
                            coordinate_value,
                            _observation(
                                protocol=protocol,
                                deployment=deployment,
                                coordinate=coordinate,
                                result=result,
                                evidence_bundle=reseal_shock(duplicated_operation),
                            ),
                        )

                    complete_operation = next(
                        item
                        for item in reversed(bundle["operations"])
                        if item["phase"] == "POST" and item["status"] == "COMPLETE"
                    )
                    receipt_order = json.loads(json.dumps(bundle))
                    receipt_order["ledger_export"], receipt_order_head = (
                        _swap_first_receipt_events(
                            receipt_order["ledger_export"],
                            subject_id=complete_operation["candidate_id"],
                        )
                    )
                    receipt_order["ledger_export_digest"] = digest_bytes(
                        receipt_order["ledger_export"].encode("utf-8")
                    )
                    receipt_order["ledger_head_digest"] = receipt_order_head
                    receipt_order = reseal_shock(receipt_order)
                    receipt_order_observation = _observation(
                        protocol=protocol,
                        deployment=deployment,
                        coordinate=coordinate,
                        result=result,
                        evidence_bundle=receipt_order,
                    )
                    with patch(
                        "egv.experiment.production._require_signed_receipt_event_order",
                        return_value=None,
                    ), patch(
                        "egv.experiment.production._require_exact_shock_event_order",
                        return_value=None,
                    ):
                        old_order_verified = verifier(
                            coordinate_value, receipt_order_observation
                        )
                    self.assertEqual(old_order_verified.result, result)
                    with self.assertRaisesRegex(
                        HeldoutProtocolError, "receipt event order"
                    ):
                        verifier(coordinate_value, receipt_order_observation)

                    forged_revision = "forged-independent-evaluator-revision-v999"
                    revision_tampered = json.loads(json.dumps(bundle))
                    revision_records = [
                        json.loads(line)
                        for line in revision_tampered["ledger_export"].splitlines()
                    ]
                    for record in revision_records:
                        if (
                            record.get("record_type") == "EVENT"
                            and record.get("event_type") == "VERDICT"
                        ):
                            payload = dict(record["payload"])
                            payload["evaluator_revision"] = forged_revision
                            payload_bytes = canonical_bytes(payload)
                            record["payload"] = payload
                            record["payload_hash"] = digest_bytes(payload_bytes)
                            record["payload_json"] = payload_bytes.decode("utf-8")
                    _reseal_ledger_records(revision_records)
                    revision_commit = next(
                        record
                        for record in revision_records
                        if record.get("event_type") == "SHOCK_CORRECTION_COMMIT"
                    )
                    revision_policy = next(
                        record
                        for record in revision_records
                        if record.get("event_type") == "SHOCK_POLICY_ACTIVATION"
                    )
                    revision_policy_payload = dict(revision_policy["payload"])
                    revision_policy_payload["correction_receipt_digest"] = (
                        revision_commit["event_hash"]
                    )
                    revision_policy_bytes = canonical_bytes(revision_policy_payload)
                    revision_policy["payload"] = revision_policy_payload
                    revision_policy["payload_hash"] = digest_bytes(
                        revision_policy_bytes
                    )
                    revision_policy["payload_json"] = revision_policy_bytes.decode(
                        "utf-8"
                    )
                    revision_tampered["ledger_export"], revision_head = (
                        _reseal_ledger_records(revision_records)
                    )
                    revision_tampered["ledger_export_digest"] = digest_bytes(
                        revision_tampered["ledger_export"].encode("utf-8")
                    )
                    revision_tampered["ledger_head_digest"] = revision_head
                    revision_tampered = reseal_shock(revision_tampered)
                    revision_observation = _observation(
                        protocol=protocol,
                        deployment=deployment,
                        coordinate=coordinate,
                        result=result,
                        evidence_bundle=revision_tampered,
                    )
                    original_getitem = SealedHeldoutDeploymentManifest.__getitem__

                    def old_self_trusting_revision(manifest, key):
                        if manifest is deployment and key == "evaluator_revision":
                            return forged_revision
                        return original_getitem(manifest, key)

                    with patch.object(
                        SealedHeldoutDeploymentManifest,
                        "__getitem__",
                        new=old_self_trusting_revision,
                    ):
                        old_revision_verified = verifier(
                            coordinate_value, revision_observation
                        )
                    self.assertEqual(old_revision_verified.result, result)
                    with self.assertRaisesRegex(HeldoutProtocolError, "verdict"):
                        verifier(coordinate_value, revision_observation)

                    lifecycle_tampered = json.loads(json.dumps(bundle))
                    lifecycle_records = [
                        json.loads(line)
                        for line in lifecycle_tampered["ledger_export"].splitlines()
                    ]
                    commit_record = next(
                        record
                        for record in lifecycle_records
                        if record.get("event_type") == "SHOCK_CORRECTION_COMMIT"
                    )
                    policy_record = next(
                        record
                        for record in lifecycle_records
                        if record.get("event_type") == "SHOCK_POLICY_ACTIVATION"
                    )
                    lifecycle_records.remove(commit_record)
                    lifecycle_records.remove(policy_record)
                    unrelated_index = next(
                        index
                        for index, record in enumerate(lifecycle_records)
                        if record.get("event_type") == "SHOCK_UNRELATED_EVIDENCE"
                    )
                    lifecycle_records[unrelated_index + 1 : unrelated_index + 1] = [
                        commit_record,
                        policy_record,
                    ]
                    sequence = 0
                    for record in lifecycle_records:
                        if record.get("record_type") == "EVENT":
                            sequence += 1
                            record["sequence"] = sequence
                    _reseal_ledger_records(lifecycle_records)
                    policy_payload = dict(policy_record["payload"])
                    policy_payload["correction_receipt_digest"] = commit_record[
                        "event_hash"
                    ]
                    policy_payload_bytes = canonical_bytes(policy_payload)
                    policy_record["payload"] = policy_payload
                    policy_record["payload_hash"] = digest_bytes(policy_payload_bytes)
                    policy_record["payload_json"] = policy_payload_bytes.decode("utf-8")
                    lifecycle_tampered["ledger_export"], lifecycle_head = (
                        _reseal_ledger_records(lifecycle_records)
                    )
                    lifecycle_tampered["ledger_export_digest"] = digest_bytes(
                        lifecycle_tampered["ledger_export"].encode("utf-8")
                    )
                    lifecycle_tampered["ledger_head_digest"] = lifecycle_head
                    lifecycle_tampered = reseal_shock(lifecycle_tampered)
                    lifecycle_observation = _observation(
                        protocol=protocol,
                        deployment=deployment,
                        coordinate=coordinate,
                        result=result,
                        evidence_bundle=lifecycle_tampered,
                    )
                    with patch(
                        "egv.experiment.production._require_shock_lifecycle_event_order",
                        return_value=None,
                    ), patch(
                        "egv.experiment.production._require_exact_shock_event_order",
                        return_value=None,
                    ):
                        old_lifecycle_verified = verifier(
                            coordinate_value, lifecycle_observation
                        )
                    self.assertEqual(old_lifecycle_verified.result, result)
                    with self.assertRaisesRegex(
                        HeldoutProtocolError, "lifecycle order"
                    ):
                        verifier(coordinate_value, lifecycle_observation)

                    dependency_order = json.loads(json.dumps(bundle))
                    dependency_records = [
                        json.loads(line)
                        for line in dependency_order["ledger_export"].splitlines()
                    ]
                    failed_post = next(
                        item
                        for item in dependency_order["operations"]
                        if item["phase"] == "POST"
                        and item["status"] == "GENERATION_FAILED"
                    )
                    moved_dependency = next(
                        record
                        for record in dependency_records
                        if record.get("event_type") == "DEPENDENCY"
                        and record.get("subject_id") == failed_post["candidate_id"]
                    )
                    dependency_records.remove(moved_dependency)
                    replacement_index = next(
                        index
                        for index, record in enumerate(dependency_records)
                        if record.get("event_type") == "SHOCK_CORRECTED_PREMISE"
                    )
                    dependency_records.insert(replacement_index, moved_dependency)
                    sequence = 0
                    for record in dependency_records:
                        if record.get("record_type") == "EVENT":
                            sequence += 1
                            record["sequence"] = sequence
                    _reseal_ledger_records(dependency_records)
                    dependency_commit = next(
                        record
                        for record in dependency_records
                        if record.get("event_type") == "SHOCK_CORRECTION_COMMIT"
                    )
                    dependency_policy = next(
                        record
                        for record in dependency_records
                        if record.get("event_type") == "SHOCK_POLICY_ACTIVATION"
                    )
                    dependency_policy_payload = dict(dependency_policy["payload"])
                    dependency_policy_payload["correction_receipt_digest"] = (
                        dependency_commit["event_hash"]
                    )
                    dependency_policy_bytes = canonical_bytes(
                        dependency_policy_payload
                    )
                    dependency_policy["payload"] = dependency_policy_payload
                    dependency_policy["payload_hash"] = digest_bytes(
                        dependency_policy_bytes
                    )
                    dependency_policy["payload_json"] = (
                        dependency_policy_bytes.decode("utf-8")
                    )
                    dependency_order["ledger_export"], dependency_order_head = (
                        _reseal_ledger_records(dependency_records)
                    )
                    dependency_order["ledger_export_digest"] = digest_bytes(
                        dependency_order["ledger_export"].encode("utf-8")
                    )
                    dependency_order["ledger_head_digest"] = dependency_order_head
                    dependency_order = reseal_shock(dependency_order)
                    dependency_order_observation = _observation(
                        protocol=protocol,
                        deployment=deployment,
                        coordinate=coordinate,
                        result=result,
                        evidence_bundle=dependency_order,
                    )
                    with patch(
                        "egv.experiment.production._require_shock_lifecycle_event_order",
                        return_value=None,
                    ), patch(
                        "egv.experiment.production._require_exact_shock_event_order",
                        return_value=None,
                    ):
                        old_dependency_order_verified = verifier(
                            coordinate_value, dependency_order_observation
                        )
                    self.assertEqual(old_dependency_order_verified.result, result)
                    with self.assertRaisesRegex(
                        HeldoutProtocolError, "lifecycle order"
                    ):
                        verifier(coordinate_value, dependency_order_observation)

                    for event_type in actual_event_counts:
                        target = next(
                            event
                            for event in reversed(event_records)
                            if event["event_type"] == event_type
                        )
                        for field, value in (
                            ("payload_json", " " + target["payload_json"]),
                            ("evaluator_identity", "forged-evaluator"),
                        ):
                            with self.subTest(
                                shock_event_envelope=event_type,
                                substituted_field=field,
                            ):
                                tampered = json.loads(json.dumps(bundle))
                                tampered["ledger_export"], head = _substitute_event_envelope(
                                    tampered["ledger_export"],
                                    event_type,
                                    field,
                                    value,
                                )
                                tampered["ledger_export_digest"] = digest_bytes(
                                    tampered["ledger_export"].encode("utf-8")
                                )
                                tampered["ledger_head_digest"] = head
                                unsigned_tampered = dict(tampered)
                                unsigned_tampered.pop("evidence_bundle_digest")
                                tampered["evidence_bundle_digest"] = digest_for(
                                    unsigned_tampered
                                )
                                with self.assertRaises(HeldoutProtocolError):
                                    verifier(
                                        coordinate_value,
                                        _observation(
                                            protocol=protocol,
                                            deployment=deployment,
                                            coordinate=coordinate,
                                            result=result,
                                            evidence_bundle=tampered,
                                        ),
                                    )

                    def prompt_substituted_bundle(operation_status):
                        tampered = json.loads(json.dumps(bundle))
                        operation = next(
                            item
                            for item in tampered["operations"]
                            if item["phase"] == "POST"
                            and item["status"] == operation_status
                        )
                        material = next(
                            item
                            for item in tampered["source_materials"]
                            if item["candidate_id"] == operation["candidate_id"]
                        )
                        forged_prompt_digest = digest_for(
                            {
                                "forged_prompt_for": operation["operation_id"],
                                "status": operation_status,
                            }
                        )
                        generation_record = material["generation_record"]
                        generation_record["context"][
                            "prompt_digest"
                        ] = forged_prompt_digest
                        generation_record[
                            "rendered_prompt_digest"
                        ] = forged_prompt_digest
                        generation_evidence_digest = digest_bytes(
                            canonical_bytes(generation_record)
                        )
                        material[
                            "generation_record_digest"
                        ] = generation_evidence_digest
                        operation[
                            "generation_evidence_digest"
                        ] = generation_evidence_digest
                        operation["context_digest"] = digest_for(
                            generation_record["context"]
                        )
                        event_records = [
                            json.loads(line)
                            for line in tampered["ledger_export"].splitlines()
                        ]
                        if operation_status == "COMPLETE":
                            event_type = "CANDIDATE"
                            target = next(
                                event
                                for event in event_records
                                if event.get("record_type") == "EVENT"
                                and event.get("event_type") == event_type
                                and event.get("subject_id")
                                == operation["candidate_id"]
                            )
                            payload = json.loads(target["payload_json"])
                            payload["prompt_hash"] = forged_prompt_digest
                            payload["metadata"][
                                "generation_evidence_digest"
                            ] = generation_evidence_digest
                        else:
                            self.assertEqual(
                                generation_record["failure_stage"],
                                "RESPONSE_CONTRACT",
                            )
                            event_type = "SHOCK_GENERATION_FAILURE"
                            target = next(
                                event
                                for event in event_records
                                if event.get("record_type") == "EVENT"
                                and event.get("event_type") == event_type
                                and event.get("subject_id")
                                == operation["candidate_id"]
                            )
                            payload = json.loads(target["payload_json"])
                            payload["context_digest"] = operation[
                                "context_digest"
                            ]
                            payload[
                                "generation_evidence_digest"
                            ] = generation_evidence_digest
                            raw_digests = [
                                value
                                for value in (
                                    generation_record["rendered_prompt_digest"],
                                    generation_record[
                                        "decoded_model_response_digest"
                                    ],
                                    generation_record["contract_response_digest"],
                                )
                                if value is not None
                            ]
                            payload["raw_artifact_digest_root"] = digest_for(
                                raw_digests
                            )
                        tampered["ledger_export"], head = _replace_event_payload(
                            tampered["ledger_export"],
                            event_type,
                            operation["candidate_id"],
                            payload,
                        )
                        tampered["ledger_export_digest"] = digest_bytes(
                            tampered["ledger_export"].encode("utf-8")
                        )
                        tampered["ledger_head_digest"] = head
                        unsigned_tampered = dict(tampered)
                        unsigned_tampered.pop("evidence_bundle_digest")
                        tampered["evidence_bundle_digest"] = digest_for(
                            unsigned_tampered
                        )
                        return tampered, forged_prompt_digest, operation

                    for operation_status in ("COMPLETE", "GENERATION_FAILED"):
                        with self.subTest(
                            fully_resealed_shock_prompt_status=operation_status
                        ):
                            prompt_tampered, forged_prompt_digest, operation = (
                                prompt_substituted_bundle(operation_status)
                            )
                            prompt_tampered_observation = _observation(
                                protocol=protocol,
                                deployment=deployment,
                                coordinate=coordinate,
                                result=result,
                                evidence_bundle=prompt_tampered,
                            )
                            original_prompt_digest = (
                                production_module._candidate_prompt_digest
                            )

                            def old_trusting_prompt_digest(tokenizer, context):
                                if (
                                    context.run_id == operation["run_id"]
                                    and context.attempt_index
                                    == int(operation["attempt"]) + 6
                                ):
                                    return forged_prompt_digest
                                return original_prompt_digest(tokenizer, context)

                            with patch(
                                "egv.experiment.production._candidate_prompt_digest",
                                side_effect=old_trusting_prompt_digest,
                            ):
                                with self.assertRaisesRegex(
                                    HeldoutProtocolError, "raw generation bytes"
                                ):
                                    verifier(
                                        coordinate_value,
                                        prompt_tampered_observation,
                                    )
                            with self.assertRaisesRegex(
                                HeldoutProtocolError,
                                "context",
                            ):
                                verifier(
                                    coordinate_value,
                                    prompt_tampered_observation,
                                )

                    materials_by_candidate = {
                        item["candidate_id"]: item for item in bundle["source_materials"]
                    }
                    for operation_status in ("COMPLETE", "GENERATION_FAILED"):
                        operation = next(
                            item
                            for item in bundle["operations"]
                            if item["status"] == operation_status
                        )
                        material = materials_by_candidate[operation["candidate_id"]]
                        omitted_raw = json.loads(json.dumps(bundle))
                        omitted_material = next(
                            item
                            for item in omitted_raw["source_materials"]
                            if item["candidate_id"] == operation["candidate_id"]
                        )
                        omitted_material.pop("raw_generation")
                        with self.assertRaisesRegex(HeldoutProtocolError, "closed object"):
                            verifier(
                                coordinate_value,
                                _observation(
                                    protocol=protocol,
                                    deployment=deployment,
                                    coordinate=coordinate,
                                    result=result,
                                    evidence_bundle=reseal_shock(omitted_raw),
                                ),
                            )

                        tampered_raw = json.loads(json.dumps(bundle))
                        tampered_material = next(
                            item
                            for item in tampered_raw["source_materials"]
                            if item["candidate_id"] == operation["candidate_id"]
                        )["raw_generation"]
                        tampered_material["rendered_prompt_b64"] = base64.urlsafe_b64encode(
                            b"forged private shock prompt"
                        ).decode("ascii").rstrip("=")
                        raw_body = dict(tampered_material)
                        raw_body.pop("material_digest")
                        tampered_material["material_digest"] = digest_for(raw_body)
                        with self.assertRaisesRegex(HeldoutProtocolError, "raw generation bytes"):
                            verifier(
                                coordinate_value,
                                _observation(
                                    protocol=protocol,
                                    deployment=deployment,
                                    coordinate=coordinate,
                                    result=result,
                                    evidence_bundle=reseal_shock(tampered_raw),
                                ),
                            )
                        for missing_kind in ("record", "artifact"):
                            with self.subTest(
                                shock_terminal_status=operation_status,
                                shock_terminal_missing=missing_kind,
                            ):
                                copied_runtime = Path(
                                    tempfile.mkdtemp(prefix="es", dir=temporary_root)
                                )
                                self.addCleanup(
                                    shutil.rmtree,
                                    copied_runtime,
                                    ignore_errors=True,
                                )
                                shutil.copytree(
                                    runtime_root,
                                    copied_runtime,
                                    dirs_exist_ok=True,
                                )
                                copied_store = PrivateTrajectoryStore(
                                    copied_runtime
                                    / coordinate.coordinate_id
                                    / "private"
                                )
                                if missing_kind == "record":
                                    _unlink_immutable_fixture(
                                        copied_store.generation_records
                                        / (operation["candidate_id"] + ".json")
                                    )
                                else:
                                    artifact_digest = material["generation_record"][
                                        "contract_response_digest"
                                    ]
                                    _unlink_immutable_fixture(
                                        copied_store.artifacts.root
                                        / "blobs"
                                        / "sha256"
                                        / artifact_digest[:2]
                                        / artifact_digest[2:4]
                                        / artifact_digest
                                    )
                                before = _tree_snapshot(copied_runtime)
                                with self.assertRaises(
                                    (HeldoutProtocolError, VariationCheckpointError)
                                ):
                                    _shock_evidence_bundle(
                                        protocol=protocol,
                                        deployment=deployment,
                                        coordinate=coordinate,
                                        runtime_root=copied_runtime,
                                    )
                                self.assertEqual(
                                    _tree_snapshot(copied_runtime),
                                    before,
                                )

                    substituted_bundle = json.loads(json.dumps(bundle))
                    substituted_bundle["journal"]["recovery_attempt"] = 1
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

                    with tempfile.TemporaryDirectory(
                        prefix="egv-shock-inspect-", dir=temporary_root
                    ) as inspection_root:
                        inspection = EvidenceLedger.replay_jsonl(
                            bundle["ledger_export"],
                            Path(inspection_root) / "ledger.sqlite3",
                        )
                        try:
                            source_events = inspection.events()
                            post_failure_candidates = {
                                operation["candidate_id"]
                                for operation in bundle["operations"]
                                if operation["phase"] == "POST"
                                and operation["status"] == "GENERATION_FAILED"
                            }
                            failure_event = next(
                                event
                                for event in source_events
                                if event["event_type"] == "SHOCK_GENERATION_FAILURE"
                                and event["subject_id"] in post_failure_candidates
                            )
                            failed_dependency_event = next(
                                event
                                for event in source_events
                                if event["event_type"] == "DEPENDENCY"
                                and event["payload"]["child_id"]
                                == failure_event["subject_id"]
                            )
                        finally:
                            inspection.close()

                    def assert_failure_event_tamper_rejected(mutate, pattern):
                        tampered = json.loads(json.dumps(bundle))
                        with tempfile.TemporaryDirectory(
                            prefix="egv-shock-tamper-", dir=temporary_root
                        ) as replay_root:
                            replay = EvidenceLedger.replay_jsonl(
                                tampered["ledger_export"],
                                Path(replay_root) / "ledger.sqlite3",
                            )
                            mutate(replay)
                            tampered["ledger_export"] = replay.export_jsonl()
                            tampered["ledger_export_digest"] = digest_bytes(
                                tampered["ledger_export"].encode("utf-8")
                            )
                            tampered["receipt_collection_root"] = digest_for(
                                replay.receipts()
                            )
                            tampered["ledger_head_digest"] = replay.ledger_head_hash()
                            replay.close()
                        unsigned = dict(tampered)
                        unsigned.pop("evidence_bundle_digest")
                        tampered["evidence_bundle_digest"] = digest_for(unsigned)
                        with self.assertRaisesRegex(HeldoutProtocolError, pattern):
                            verifier(
                                coordinate_value,
                                _observation(
                                    protocol=protocol,
                                    deployment=deployment,
                                    coordinate=coordinate,
                                    result=result,
                                    evidence_bundle=tampered,
                                ),
                            )

                    def add_duplicate_failure_event(replay):
                        replay.append_event(
                            "SHOCK_GENERATION_FAILURE",
                            failure_event["payload"],
                            campaign_id=failure_event["campaign_id"],
                            run_id=failure_event["run_id"],
                            task_id=failure_event["task_id"],
                            subject_id=failure_event["subject_id"],
                            source_class="PINNED_MODEL",
                            disposition="REJECTED",
                            idempotency_key="forged-duplicate-shock-generation-failure",
                        )

                    assert_failure_event_tamper_rejected(
                        add_duplicate_failure_event,
                        "generation failure ledger evidence|terminal operation event inventory",
                    )

                    def add_orphan_failure_event(replay):
                        orphan_candidate = "egv-candidate-" + digest_for(
                            "orphan-shock-generation-failure"
                        )
                        payload = dict(failure_event["payload"])
                        payload["candidate_id"] = orphan_candidate
                        payload["operation_id"] = "egv-shock-operation-" + digest_for(
                            "orphan-shock-generation-operation"
                        )
                        replay.append_event(
                            "SHOCK_GENERATION_FAILURE",
                            payload,
                            campaign_id=failure_event["campaign_id"],
                            run_id=failure_event["run_id"],
                            task_id=failure_event["task_id"],
                            subject_id=orphan_candidate,
                            source_class="PINNED_MODEL",
                            disposition="REJECTED",
                            idempotency_key="shock-generation-failure:" + payload["operation_id"],
                        )

                    assert_failure_event_tamper_rejected(
                        add_orphan_failure_event,
                        "unaccounted evidence or projections",
                    )

                    def add_failure_event_for_complete_operation(replay):
                        complete = next(
                            operation
                            for operation in bundle["operations"]
                            if operation["status"] == "COMPLETE"
                        )
                        payload = dict(failure_event["payload"])
                        payload.update(
                            {
                                "block_id": (
                                    coordinate.block_id
                                    if complete["phase"] == "PRE"
                                    else None
                                ),
                                "coordinate_id": (
                                    coordinate.coordinate_id
                                    if complete["phase"] == "POST"
                                    else None
                                ),
                                "treatment": (
                                    coordinate.treatment
                                    if complete["phase"] == "POST"
                                    else None
                                ),
                                "operation_id": complete["operation_id"],
                                "phase": complete["phase"],
                                "attempt": complete["attempt"],
                                "candidate_id": complete["candidate_id"],
                                "run_id": complete["run_id"],
                                "context_digest": complete["context_digest"],
                                "generation_evidence_digest": complete[
                                    "generation_evidence_digest"
                                ],
                            }
                        )
                        replay.append_event(
                            "SHOCK_GENERATION_FAILURE",
                            payload,
                            campaign_id=protocol.campaign_id,
                            run_id=complete["run_id"],
                            task_id=coordinate.task_id,
                            subject_id=complete["candidate_id"],
                            source_class="PINNED_MODEL",
                            disposition="REJECTED",
                            idempotency_key=(
                                "shock-generation-failure:" + complete["operation_id"]
                            ),
                        )

                    assert_failure_event_tamper_rejected(
                        add_failure_event_for_complete_operation,
                        "unaccounted evidence or projections|lifecycle order",
                    )

                    def assert_dependency_metadata_tamper_rejected(field, value):
                        tampered = json.loads(json.dumps(bundle))
                        with tempfile.TemporaryDirectory(
                            prefix="egv-shock-edge-", dir=temporary_root
                        ) as rebuilt_root:
                            rebuilt = EvidenceLedger(Path(rebuilt_root) / "ledger.sqlite3")
                            try:
                                for event in source_events:
                                    metadata = {
                                        "campaign_id": event.get("campaign_id"),
                                        "run_id": event.get("run_id"),
                                        "task_id": event.get("task_id"),
                                        "valid_time": event.get("valid_time"),
                                        "subject_id": event.get("subject_id"),
                                        "source_class": event.get("source_class"),
                                        "disposition": event.get("disposition"),
                                        "evaluator_identity": event.get(
                                            "evaluator_identity"
                                        ),
                                        "idempotency_key": event.get("idempotency_key"),
                                        "transaction_time": event["transaction_time"],
                                    }
                                    if event["event_id"] == failed_dependency_event["event_id"]:
                                        metadata[field] = value
                                    rebuilt.append_event(
                                        event["event_type"],
                                        event["payload"],
                                        **metadata,
                                    )
                                tampered["ledger_export"] = rebuilt.export_jsonl()
                                tampered["ledger_export_digest"] = digest_bytes(
                                    tampered["ledger_export"].encode("utf-8")
                                )
                                tampered["ledger_head_digest"] = rebuilt.ledger_head_hash()
                            finally:
                                rebuilt.close()
                        unsigned = dict(tampered)
                        unsigned.pop("evidence_bundle_digest")
                        tampered["evidence_bundle_digest"] = digest_for(unsigned)
                        with self.assertRaisesRegex(
                            HeldoutProtocolError,
                            "dependency event projection",
                        ):
                            verifier(
                                coordinate_value,
                                _observation(
                                    protocol=protocol,
                                    deployment=deployment,
                                    coordinate=coordinate,
                                    result=result,
                                    evidence_bundle=tampered,
                                ),
                            )

                    for field, value in (
                        ("campaign_id", "egv-campaign-forged-dependency"),
                        ("run_id", "egv-run-forged-dependency"),
                        ("task_id", "forged-dependency-task"),
                        ("subject_id", "forged-dependency-subject"),
                        ("source_class", "UNTRUSTED_SOURCE"),
                        ("disposition", "VERIFIED"),
                        ("idempotency_key", "forged-dependency-idempotency"),
                    ):
                        with self.subTest(dependency_metadata_field=field):
                            assert_dependency_metadata_tamper_rejected(field, value)
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
