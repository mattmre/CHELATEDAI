from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from egv.canonical import canonical_bytes, canonical_json, content_id, digest_for
from egv.evaluation.authority import AuthorityPolicy
from egv.evaluation.dataset import EvaluationCorpus
from egv.evaluation.sandbox import DockerSandboxConfig
from egv.experiment.heldout import FrozenHeldoutProtocol, HeldoutProtocolError, SHOCK_PHASE
from egv.experiment.runtime import (
    HeldoutTrainerInputs,
    HeldoutTrainerSources,
    build_trainer_evidence_package,
    ordered_public_heldout_task_records,
)
from egv.experiment.shock_engine import (
    ProductionShockAttemptEngine,
    ProductionShockEngineFactory,
)
from egv.experiment.shock_runtime import (
    CorrectionShockCoordinateRunner,
    ShockRuntimeContext,
    ShockRuntimeJournal,
)
from egv.ledger import EvidenceLedger
from egv.receipts import ReceiptSigner
from egv.variation.arms import ArmIsolation
from egv.variation.adapter import (
    ADAPTER_MANIFEST_NAME,
    SealedAdapterArtifact,
    build_local_adapter_manifest,
)
from egv.variation.generator import ModelCandidateGenerator
from egv.variation.loop import VARIATION_PROTOCOL_DIGEST
from egv.variation.model import (
    MODEL_ARCHITECTURE,
    MODEL_CONFIG_CLASS,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    ADAPTER_ATTESTATION_SCHEMA,
    AdapterApplicationAttestation,
    LoadedPinnedModel,
    PinnedModelManifest,
    _ADAPTER_ATTESTATION_TOKEN,
    model_state_digest,
)
from egv.variation.private import PrivateTrajectoryStore
from egv.variation.remote import (
    REMOTE_VARIATION_SERVICE_SCHEMA,
)


RESPONDER = r'''from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path[:0] = __RUNTIME_IMPORT_ROOTS__
from egv.canonical import canonical_bytes, canonical_json, content_id, digest_bytes, digest_for
from egv.receipts import ReceiptSigner, receipt_hash

request = json.loads(sys.stdin.read())
state = Path(__STATE_ROOT__)
state.mkdir(parents=True, exist_ok=True)
cache = state / (request["operation_digest"] + ".json")
counter = state / "effect-count.txt"
if cache.exists():
    sys.stdout.write(cache.read_text(encoding="utf-8"))
    raise SystemExit(0)
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
output = digest_bytes(b"hidden-fixture-output")
receipts = []
for kind, fields in (
    ("AUTHORITY", {"request_id": "request-authority-" + request["candidate_id"], "decision": "ALLOW"}),
    ("VERDICT", {"request_id": "request-verdict-" + request["candidate_id"], "decision": "PASS",
        "diagnostic_enum": "PASS", "resource_bucket": "UNDER_25", "exit_status_class": "SUCCESS",
        "input_digest": digest_for("independent-hidden-fixture"), "output_digest": output}),
    ("EFFECT", {"request_id": "request-effect-" + request["candidate_id"], "decision": "ALLOW",
        "diagnostic_enum": "PASS",
        "normalized_action_hash": digest_for({"action": "execute_candidate", "locus": request["declared_locus"]}),
        "sandbox_id": "sealed-remote-sandbox", "started_at": "2026-08-24T00:00:00Z",
        "finished_at": "2026-08-24T00:00:01Z", "exit_status_class": "SUCCESS",
        "output_digest": output, "environment_diff_digest": digest_for({})}),
):
    receipt = signer.sign_receipt(
        {**common, "receipt_type": kind, **fields}, sequence=sequence,
        previous_receipt_hash=previous,
        idempotency_key=content_id("shock-test-receipt", {"operation": request["operation_digest"], "kind": kind}),
    )
    receipts.append(receipt)
    previous = receipt_hash(receipt)
    sequence += 1
result = {
    "candidate_id": request["candidate_id"], "task_id": request["task_id"],
    "candidate_artifact_digest": request["candidate_artifact_digest"], "diagnostic_enum": "PASS",
    "resource_bucket": "UNDER_25", "disposition": "PROMOTED", "infrastructure_loss": False,
    "receipt_ids": [item["receipt_id"] for item in receipts], "output_digest": output,
}
unsigned = {
    "schema_version": "egv-remote-variation-response-v1",
    "operation_digest": request["operation_digest"], "request_digest": request["request_digest"],
    "service_manifest_digest": request["service_manifest_digest"], "result": result,
    "receipts": receipts, "signing_key_id": signer.key_id,
}
response = {**unsigned, "signature": signer.sign_bytes(canonical_bytes(unsigned))}
cache.write_text(canonical_json(response), encoding="utf-8")
count = int(counter.read_text(encoding="ascii")) if counter.exists() else 0
counter.write_text(str(count + 1), encoding="ascii")
sys.stdout.write(canonical_json(response))
'''


class _Device:
    type = "cuda"


class _Parameter:
    device = _Device()


class _Tensor:
    def __init__(self) -> None:
        self.shape = (1, 1)
        self.device = None

    def to(self, *, device):
        self.device = device
        return self


class _PeftModel:
    pass


class _Model(_PeftModel):
    def __init__(self) -> None:
        self.calls = 0
        self.active_adapter = "default"
        self.peft_config = {
            "default": {"lora_alpha": 8, "peft_type": "LORA", "r": 4}
        }

    def parameters(self):
        return iter((_Parameter(),))

    def generate(self, **_kwargs):
        self.calls += 1
        return [[0, self.calls]]

    @staticmethod
    def state_dict():
        return {"adapter.default.weight": b"sealed-adapter-state"}


class _Tokenizer:
    chat_template = "frozen-test-chat-template-v1"

    def __init__(self) -> None:
        self.function_name = "solve"
        self.decode_calls = 0
        self.invalid_decode_calls: set[int] = set()

    def apply_chat_template(self, messages, **kwargs):
        if kwargs.get("tokenize") is False:
            return "<chat>" + "\n".join(item["content"] for item in messages) + "</chat>"
        return {"input_ids": _Tensor()}

    def __call__(self, text, **kwargs):
        if kwargs.get("return_tensors") == "pt":
            return {"input_ids": _Tensor()}
        return {"input_ids": [list(range(max(1, len(text.split()))))]}

    def decode(self, _tokens, *, skip_special_tokens):
        assert skip_special_tokens is True
        self.decode_calls += 1
        if self.decode_calls in self.invalid_decode_calls:
            return "this is not valid Python source !!!"
        return f"def {self.function_name}(value):\n    return value"


def _generator(adapter_root: Path | None = None) -> tuple[ModelCandidateGenerator, _Model]:
    manifest = PinnedModelManifest(
        repository=MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        architecture=MODEL_ARCHITECTURE,
        config_class=MODEL_CONFIG_CLASS,
        transformers_version="5.5.0",
        files={"model.safetensors": digest_for("test-weights")},
        license={"name": "Apache-2.0", "source": "test-fixture"},
    )
    model = _Model()
    adapter = None
    adapter_digest = None
    attestation = None
    base_state_digest = digest_for("test-base-state")
    if adapter_root is not None:
        adapter_root.mkdir(parents=True)
        (adapter_root / "adapter_config.json").write_text(
            '{"lora_alpha":8,"peft_type":"LORA","r":4}\n',
            encoding="utf-8",
        )
        (adapter_root / "adapter_model.safetensors").write_bytes(
            b"sealed-test-adapter"
        )
        adapter_manifest = build_local_adapter_manifest(adapter_root)
        (adapter_root / ADAPTER_MANIFEST_NAME).write_text(
            canonical_json(adapter_manifest.to_dict()) + "\n",
            encoding="utf-8",
        )
        adapter = SealedAdapterArtifact(adapter_root)
        adapter_digest = adapter.digest
        applied_state_digest = model_state_digest(model)
        assert applied_state_digest is not None
        attestation = AdapterApplicationAttestation(
            schema_version=ADAPTER_ATTESTATION_SCHEMA,
            adapter_digest=adapter_digest,
            base_model_manifest_digest=manifest.digest(),
            base_state_digest=base_state_digest,
            applied_model_state_digest=applied_state_digest,
            issuer_token=_ADAPTER_ATTESTATION_TOKEN,
        )
    loaded = LoadedPinnedModel(
        model=model,
        tokenizer=_Tokenizer(),
        manifest=manifest,
        manifest_digest=manifest.digest(),
        file_hashes=dict(manifest.files),
        load_report={"device": "cuda"},
        base_state_digest=base_state_digest,
        adapter_digest=adapter_digest,
        adapter_attestation=attestation,
    )
    return (
        ModelCandidateGenerator(
            loaded,
            model_digest=manifest.digest(),
            adapter_digest=adapter_digest,
            adapter_artifact=adapter,
            response_contract="source-only-v1",
        ),
        model,
    )


def _protocol(
    corpus: EvaluationCorpus,
    *,
    binding_overrides: dict[str, str] | None = None,
) -> FrozenHeldoutProtocol:
    bindings = {
        name: digest_for({"binding": name})
        for name in FrozenHeldoutProtocol.REQUIRED_BINDINGS
    }
    bindings.update(binding_overrides or {})
    signer = ReceiptSigner(b"H" * 32)
    return FrozenHeldoutProtocol.build(
        campaign_id="egv-campaign-1234567890abcdef",
        bindings=bindings,
        evaluator_public_key=signer.public_key,
        schedule_seed=81,
        bootstrap_seed=82,
        heldout_task_records=ordered_public_heldout_task_records(corpus),
    )


class ShockEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="egv-shock-engine-")
        self.root = Path(self.temporary.name)
        self.old_peft = sys.modules.get("peft")
        sys.modules["peft"] = SimpleNamespace(
            PeftModel=_PeftModel,
            PeftModelForCausalLM=None,
        )
        self.old_pythonpath = os.environ.get("PYTHONPATH")
        repository_root = str(Path(__file__).resolve().parents[1])
        os.environ["PYTHONPATH"] = repository_root + (
            os.pathsep + self.old_pythonpath if self.old_pythonpath else ""
        )
        seed = self.root / "heldout-seed.bin"
        seed.write_bytes(b"Z" * 32)
        self.corpus = EvaluationCorpus.generate(secret_seed_file=seed)
        self.generator, self.model = _generator(self.root / "sealed-adapter")
        self.protocol = _protocol(
            self.corpus,
            binding_overrides={
                "base_model_digest": self.generator.model_digest,
                "adapter_digest": self.generator.adapter_digest,
            },
        )
        raw_inputs, raw_sources = build_trainer_evidence_package(
            self.corpus,
            self.protocol,
            generation_profile_digest=digest_for("placeholder-generation"),
        )
        self.inputs = HeldoutTrainerInputs(raw_inputs, protocol=self.protocol)
        self.sources = HeldoutTrainerSources(raw_sources, trainer_inputs=self.inputs)
        self.signer = ReceiptSigner(b"R" * 32)
        self.public_key = self.root / "evaluator.pub"
        self.public_key.write_bytes(self.signer.public_key_raw)
        self.remote_state = self.root / "remote-state"
        self.command = self.root / "remote-evaluator.py"
        self.command.write_text(
            RESPONDER.replace("__PRIVATE_KEY__", self.signer.private_key_raw.hex())
            .replace("__STATE_ROOT__", repr(str(self.remote_state)))
            .replace(
                "__RUNTIME_IMPORT_ROOTS__",
                repr([repository_root, *(entry for entry in sys.path if entry and entry != repository_root)]),
            ),
            encoding="utf-8",
        )
        self.data_manifest_digest = self.corpus.manifest_digest()
        self.manifests = {}
        self.ledgers: list[EvidenceLedger] = []
        self.short_temporaries: list[tempfile.TemporaryDirectory] = []

    def tearDown(self) -> None:
        for ledger in self.ledgers:
            try:
                ledger.close()
            except Exception:
                pass
        if self.old_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = self.old_pythonpath
        if self.old_peft is None:
            sys.modules.pop("peft", None)
        else:
            sys.modules["peft"] = self.old_peft
        for temporary in self.short_temporaries:
            temporary.cleanup()
        self.temporary.cleanup()

    def coordinate(self, treatment: str, *, block_id: str | None = None):
        return next(
            item
            for item in self.protocol.coordinates
            if item.phase == SHOCK_PHASE
            and item.treatment == treatment
            and (block_id is None or item.block_id == block_id)
        )

    def manifest(self, coordinate) -> Path:
        task = dict(self.inputs.tasks[coordinate.task_id])
        key = coordinate.task_id
        if key in self.manifests:
            return self.manifests[key]
        authority = AuthorityPolicy.candidate_execution().digest
        unsigned = {
            "schema_version": REMOTE_VARIATION_SERVICE_SCHEMA,
            "campaign_id": self.protocol.campaign_id,
            "model_digest": self.generator.model_digest,
            "protocol_digest": VARIATION_PROTOCOL_DIGEST,
            "policy_digest": authority,
            "data_manifest_digest": self.data_manifest_digest,
            "task_manifest_digest": digest_for([task]),
            "task_bindings": [task],
            "evaluator_revision": "shock-hidden-evaluator-v1",
            "evaluator_digest": digest_for("shock-hidden-evaluator-v1"),
            "docker_image_digest": DockerSandboxConfig().pinned_image_id,
            "docker_config_digest": digest_for(dict(DockerSandboxConfig().__dict__)),
            "authority_policy_digest": authority,
            "command_digest": hashlib.sha256(self.command.read_bytes()).hexdigest(),
            "evaluator_key_id": self.signer.key_id,
            "evaluator_public_key_digest": hashlib.sha256(self.public_key.read_bytes()).hexdigest(),
        }
        value = {**unsigned, "service_manifest_digest": digest_for(unsigned)}
        path = self.root / (coordinate.task_id + "-service.json")
        path.write_text(canonical_json(value) + "\n", encoding="utf-8")
        self.manifests[key] = path
        return path

    def production_factory(self, coordinate) -> ProductionShockEngineFactory:
        self.generator.tokenizer.function_name = self.inputs.tasks[coordinate.task_id][
            "public_locus"
        ].rsplit(":", 1)[-1]
        return ProductionShockEngineFactory(
            generator=self.generator,
            evaluator_manifest=self.manifest(coordinate),
            evaluator_public_key=self.public_key,
            evaluator_command=self.command,
            evaluator_python_executable=Path(sys.executable).resolve(),
            evaluator_python_digest=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
            source_commit="test-shock-source-commit",
            model_revision=MODEL_REVISION,
            data_manifest_digest=self.data_manifest_digest,
            response_contract_digest=self.generator.response_contract_digest,
            generation_profile_digest=self.generator.generation_profile_digest,
        )

    def engine(self, coordinate, root: Path) -> tuple[ProductionShockAttemptEngine, EvidenceLedger]:
        root.mkdir(parents=True, exist_ok=True)
        ledger = EvidenceLedger(root / "ledger.sqlite3")
        self.ledgers.append(ledger)
        factory = self.production_factory(coordinate)
        engine = factory(
            coordinate=coordinate,
            task_record=dict(self.inputs.tasks[coordinate.task_id]),
            initial_source=self.sources.source_for(coordinate.task_id),
            ledger=ledger,
            isolation=ArmIsolation(root / "isolation", campaign_id=self.protocol.campaign_id),
            private_store=PrivateTrajectoryStore(root / "private"),
            base_model_digest=self.generator.model_digest,
            adapter_digest=self.generator.adapter_digest,
        )
        return engine, ledger

    @staticmethod
    def attempt_key(coordinate, phase: str, attempt: int) -> str:
        return digest_for(
            {
                "block_id": coordinate.block_id if phase == "PRE" else None,
                "coordinate_id": coordinate.coordinate_id if phase == "POST" else None,
                "phase": phase,
                "attempt": attempt,
            }
        )

    @staticmethod
    def correction_key(coordinate) -> str:
        return digest_for(
            {
                "coordinate_id": coordinate.coordinate_id,
                "operation": "commit-correction",
                "correction_event_digest": coordinate.correction_event_digest,
            }
        )

    @staticmethod
    def policy_key(coordinate) -> str:
        return digest_for(
            {
                "coordinate_id": coordinate.coordinate_id,
                "operation": "activate-policy",
                "policy": coordinate.treatment,
            }
        )

    def run_pre(self, engine, coordinate):
        observations = []
        for attempt in range(1, 7):
            observations.append(
                engine.pre_correction_attempt(
                    attempt,
                    idempotency_key=self.attempt_key(coordinate, "PRE", attempt),
                )
            )
        return observations

    def commit_and_activate(self, engine, coordinate):
        engine.commit_correction(
            coordinate.correction_event_digest,
            idempotency_key=self.correction_key(coordinate),
        )
        state = engine.freeze_pre_shock_state()
        affected = tuple(sorted(state.dependency_graph.descendants(state.accepted_premise_id)))
        if coordinate.treatment == "dependency-aware":
            engine.invalidate_dependencies(affected, idempotency_key=self.policy_key(coordinate))
        elif coordinate.treatment == "naive-reuse":
            engine.reuse_without_invalidation(idempotency_key=self.policy_key(coordinate))
        else:
            engine.full_restart(idempotency_key=self.policy_key(coordinate))
        return state, affected

    def effect_count(self) -> int:
        path = self.remote_state / "effect-count.txt"
        return int(path.read_text(encoding="ascii")) if path.exists() else 0

    def short_runtime_root(self, prefix: str) -> Path:
        temporary = tempfile.TemporaryDirectory(
            prefix=prefix,
            dir=Path(__file__).resolve().parents[2],
        )
        self.short_temporaries.append(temporary)
        return Path(temporary.name)

    def ready_generation_failure_campaign(self, name: str):
        coordinate = self.coordinate("dependency-aware")
        self.generator.tokenizer.invalid_decode_calls.add(1)
        engine, ledger = self.engine(coordinate, self.root / name)
        self.run_pre(engine, coordinate)
        self.commit_and_activate(engine, coordinate)
        engine.verification_evidence()
        failure = next(
            event
            for event in ledger.events()
            if event["event_type"] == "SHOCK_GENERATION_FAILURE"
        )
        records = {
            int(record["attempt"]): record
            for record in engine.operation_store.records()
            if record["phase"] == "PRE"
        }
        return engine, ledger, failure, records

    @staticmethod
    def private_tree_snapshot(store: PrivateTrajectoryStore):
        return tuple(
            (
                path.relative_to(store.root).as_posix(),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in sorted(store.root.rglob("*"))
            if path.is_file()
        )

    def assert_terminal_private_gap_is_not_healed(
        self,
        *,
        engine: ProductionShockAttemptEngine,
        ledger: EvidenceLedger,
        operation_id: str,
        missing_path: Path,
        error: str,
    ) -> None:
        missing_path.chmod(0o600)
        missing_path.unlink()
        before_private = self.private_tree_snapshot(engine.private_store)
        before_export = ledger.export_jsonl()
        before_head = ledger.ledger_head_hash()
        before_operation = engine.operation_store.load(operation_id)
        before_model_calls = self.model.calls
        before_effects = self.effect_count()

        with self.assertRaisesRegex(HeldoutProtocolError, error):
            engine.reconcile_attempt(operation_id)

        self.assertFalse(missing_path.exists())
        self.assertEqual(self.private_tree_snapshot(engine.private_store), before_private)
        self.assertEqual(ledger.export_jsonl(), before_export)
        self.assertEqual(ledger.ledger_head_hash(), before_head)
        self.assertEqual(engine.operation_store.load(operation_id), before_operation)
        self.assertEqual(self.model.calls, before_model_calls)
        self.assertEqual(self.effect_count(), before_effects)

    def generated_operation(self, name: str):
        coordinate = self.coordinate("dependency-aware")
        engine, ledger = self.engine(coordinate, self.root / name)
        key = self.attempt_key(coordinate, "PRE", 1)
        with patch.object(
            engine.operation_store,
            "complete",
            side_effect=RuntimeError("crash after generated checkpoint"),
        ):
            with self.assertRaisesRegex(RuntimeError, "generated checkpoint"):
                engine.pre_correction_attempt(1, idempotency_key=key)
        state = engine.operation_store.load(key)
        assert state is not None
        self.assertEqual(state["status"], "GENERATED")
        return engine, ledger, key, state

    @staticmethod
    def append_spurious_failure_event(
        ledger: EvidenceLedger,
        template,
        *,
        payload,
        subject_id: str,
        idempotency_key: str,
    ) -> None:
        ledger.append_event(
            "SHOCK_GENERATION_FAILURE",
            payload,
            campaign_id=template["campaign_id"],
            run_id=template["run_id"],
            task_id=template["task_id"],
            subject_id=subject_id,
            source_class="PINNED_MODEL",
            disposition="REJECTED",
            idempotency_key=idempotency_key,
        )

    def assert_corrupt_failure_dependency_rejected(self, field: str) -> None:
        coordinate = self.coordinate("dependency-aware")
        self.generator.tokenizer.invalid_decode_calls.add(1)
        engine, ledger = self.engine(
            coordinate, self.root / ("failure-dependency-" + field)
        )
        original_append = ledger.append_dependency
        injected = False

        def append_corrupt_dependency(parent_id, child_id, *, edge_type, **kwargs):
            nonlocal injected
            if injected:
                return original_append(
                    parent_id,
                    child_id,
                    edge_type=edge_type,
                    **kwargs,
                )
            injected = True
            dependency_payload = {
                "parent_id": parent_id,
                "child_id": child_id,
                "edge_type": edge_type,
            }
            event_payload = dict(dependency_payload)
            campaign_id = kwargs["campaign_id"]
            run_id = kwargs["run_id"]
            task_id = kwargs["task_id"]
            subject_id = child_id
            idempotency_key = kwargs["idempotency_key"]
            source_class = "FROZEN_PROTOCOL"
            disposition = "OBSERVED"
            if field == "campaign":
                campaign_id = None
            elif field == "run":
                run_id = None
            elif field == "task":
                task_id = None
            elif field == "subject":
                subject_id = parent_id
            elif field == "payload":
                event_payload["parent_id"] = child_id
            elif field == "idempotency":
                idempotency_key = "wrong-failure-dependency:" + child_id
            elif field == "source":
                source_class = "PINNED_MODEL"
            elif field == "disposition":
                disposition = "REJECTED"
            else:
                raise AssertionError("unknown dependency corruption field")
            event = ledger.append_event(
                "DEPENDENCY",
                event_payload,
                campaign_id=campaign_id,
                run_id=run_id,
                task_id=task_id,
                subject_id=subject_id,
                source_class=source_class,
                disposition=disposition,
                idempotency_key=idempotency_key,
            )
            dependency_id = content_id("dep", dependency_payload)
            ledger.connection.execute(
                "INSERT INTO dependencies"
                "(dependency_id,parent_id,child_id,edge_type,insertion_event_id) "
                "VALUES(?,?,?,?,?)",
                (
                    dependency_id,
                    parent_id,
                    child_id,
                    edge_type,
                    event["event_id"],
                ),
            )
            ledger.connection.commit()
            return {
                "dependency_id": dependency_id,
                **dependency_payload,
                "insertion_event_id": event["event_id"],
            }

        key = self.attempt_key(coordinate, "PRE", 1)
        with patch.object(
            ledger,
            "append_dependency",
            side_effect=append_corrupt_dependency,
        ):
            with self.assertRaisesRegex(
                HeldoutProtocolError, "dependency insertion event binding"
            ):
                engine.pre_correction_attempt(1, idempotency_key=key)
        self.assertEqual(self.model.calls, 1)
        self.assertEqual(self.effect_count(), 0)
        ledger.close()

    def assert_terminal_failure_missing_evidence_rejected_without_healing(
        self,
        *,
        event_only: bool,
    ) -> None:
        coordinate = self.coordinate("dependency-aware")
        self.generator.tokenizer.invalid_decode_calls.add(1)
        engine, ledger = self.engine(
            coordinate,
            self.root
            / ("terminal-failure-event-only" if event_only else "terminal-failure-empty"),
        )
        key = self.attempt_key(coordinate, "PRE", 1)

        def persist_event_only(state, context, evidence, evidence_digest):
            if not event_only:
                return None
            payload = engine._generation_failure_payload(
                state, context, evidence, evidence_digest
            )
            return ledger.append_event(
                "SHOCK_GENERATION_FAILURE",
                payload,
                campaign_id=self.protocol.campaign_id,
                run_id=state["run_id"],
                task_id=coordinate.task_id,
                subject_id=state["candidate_id"],
                source_class="PINNED_MODEL",
                disposition="REJECTED",
                idempotency_key="shock-generation-failure:" + state["operation_id"],
            )

        with patch.object(
            engine,
            "_persist_generation_failure",
            side_effect=persist_event_only,
        ), patch.object(
            engine,
            "_validate_generation_failure_inventory",
            return_value=None,
        ):
            engine.pre_correction_attempt(1, idempotency_key=key)
        terminal = engine.operation_store.load(key)
        assert terminal is not None
        self.assertEqual(terminal["status"], "GENERATION_FAILED")
        before_export = ledger.export_jsonl()
        before_head = ledger.ledger_head_hash()
        before_operation = dict(terminal)
        before_counts = (
            ledger.connection.execute(
                "SELECT COUNT(*) FROM events WHERE event_type='SHOCK_GENERATION_FAILURE'"
            ).fetchone()[0],
            ledger.connection.execute(
                "SELECT COUNT(*) FROM dependencies WHERE child_id=?",
                (terminal["candidate_id"],),
            ).fetchone()[0],
        )
        expected_error = "dependency set is incomplete" if event_only else "bound event"
        with self.assertRaisesRegex(HeldoutProtocolError, expected_error):
            engine.reconcile_attempt(key)
        self.assertEqual(ledger.export_jsonl(), before_export)
        self.assertEqual(ledger.ledger_head_hash(), before_head)
        self.assertEqual(engine.operation_store.load(key), before_operation)
        self.assertEqual(
            (
                ledger.connection.execute(
                    "SELECT COUNT(*) FROM events WHERE event_type='SHOCK_GENERATION_FAILURE'"
                ).fetchone()[0],
                ledger.connection.execute(
                    "SELECT COUNT(*) FROM dependencies WHERE child_id=?",
                    (terminal["candidate_id"],),
                ).fetchone()[0],
            ),
            before_counts,
        )
        self.assertEqual(self.model.calls, 1)
        self.assertEqual(self.effect_count(), 0)
        ledger.close()

    def test_exact_six_pre_attempts_continue_after_early_promotion(self) -> None:
        coordinate = self.coordinate("dependency-aware")
        engine, ledger = self.engine(coordinate, self.root / "exact")
        with self.assertRaisesRegex(HeldoutProtocolError, "order"):
            engine.pre_correction_attempt(
                2, idempotency_key=self.attempt_key(coordinate, "PRE", 2)
            )
        observations = self.run_pre(engine, coordinate)
        self.assertTrue(all(item.promoted for item in observations))
        self.assertEqual(self.model.calls, 6)
        frozen = engine.freeze_pre_shock_state()
        self.assertEqual(len(frozen.candidate_state["candidates"]), 6)
        self.assertEqual(len(frozen.candidate_state["unrelated_evidence_ids"]), 2)
        self.assertEqual(len(frozen.dependency_graph.edges), 17)
        self.assertEqual(len(ledger.receipts()), 18)
        self.assertEqual(self.effect_count(), 6)
        ledger.close()

    def test_terminal_attempt_replay_cannot_bypass_missing_earlier_attempt_range(self) -> None:
        coordinate = self.coordinate("dependency-aware")
        engine, ledger = self.engine(coordinate, self.root / "terminal-gap")
        with patch.object(engine, "_terminal_records", return_value=({"attempt": 3},)):
            with self.assertRaisesRegex(HeldoutProtocolError, "order"):
                engine._assert_attempt_order("PRE", 3)
        ledger.close()

    def test_evaluator_crash_reconciles_without_second_model_or_remote_effect(self) -> None:
        coordinate = self.coordinate("dependency-aware")
        root = self.root / "resume"
        engine, ledger = self.engine(coordinate, root)
        key = self.attempt_key(coordinate, "PRE", 1)
        with patch.object(engine.operation_store, "complete", side_effect=RuntimeError("crash after effect")):
            with self.assertRaisesRegex(RuntimeError, "crash after effect"):
                engine.pre_correction_attempt(1, idempotency_key=key)
        self.assertEqual(self.model.calls, 1)
        self.assertEqual(self.effect_count(), 1)
        self.assertEqual(len(ledger.receipts()), 3)
        generated = engine.operation_store.load(key)
        assert generated is not None
        self.assertEqual(generated["status"], "GENERATED")
        ledger.close()

        resumed, resumed_ledger = self.engine(coordinate, root)
        observation = resumed.reconcile_attempt(key)
        self.assertIsNotNone(observation)
        self.assertEqual(self.model.calls, 1)
        self.assertEqual(self.effect_count(), 1)
        self.assertEqual(len(resumed_ledger.receipts()), 3)
        resumed_ledger.close()

    def test_generated_operation_missing_private_record_is_rejected_without_healing(
        self,
    ) -> None:
        engine, ledger, key, state = self.generated_operation(
            "generated-missing-private-record"
        )
        record = engine.private_store.generation_records / (
            str(state["candidate_id"]) + ".json"
        )

        self.assert_terminal_private_gap_is_not_healed(
            engine=engine,
            ledger=ledger,
            operation_id=key,
            missing_path=record,
            error="shock model call has no recoverable generation result",
        )
        ledger.close()

    def test_generated_operation_missing_private_artifact_is_rejected_without_healing(
        self,
    ) -> None:
        engine, ledger, key, state = self.generated_operation(
            "generated-missing-private-artifact"
        )
        record = json.loads(
            (
                engine.private_store.generation_records
                / (str(state["candidate_id"]) + ".json")
            ).read_text(encoding="utf-8")
        )
        artifact_digest = str(record["contract_response_digest"])
        artifact = (
            engine.private_store.artifacts.root
            / "blobs"
            / "sha256"
            / artifact_digest[:2]
            / artifact_digest[2:4]
            / artifact_digest
        )

        self.assert_terminal_private_gap_is_not_healed(
            engine=engine,
            ledger=ledger,
            operation_id=key,
            missing_path=artifact,
            error="shock model call has no recoverable generation result",
        )
        ledger.close()

    def test_unrecorded_model_effect_fails_closed_instead_of_regenerating(self) -> None:
        coordinate = self.coordinate("dependency-aware")
        root = self.root / "unknown-model-effect"
        engine, ledger = self.engine(coordinate, root)
        key = self.attempt_key(coordinate, "PRE", 1)
        with patch.object(
            engine.private_store,
            "record_generation_success",
            side_effect=RuntimeError("crash before generation intent"),
        ):
            with self.assertRaisesRegex(RuntimeError, "generation intent"):
                engine.pre_correction_attempt(1, idempotency_key=key)
        self.assertEqual(self.model.calls, 1)
        ledger.close()
        resumed, resumed_ledger = self.engine(coordinate, root)
        with self.assertRaisesRegex(HeldoutProtocolError, "duplicate generation is forbidden"):
            resumed.reconcile_attempt(key)
        self.assertEqual(self.model.calls, 1)
        resumed_ledger.close()

    def test_existing_started_operation_never_issues_a_second_model_call(self) -> None:
        coordinate = self.coordinate("dependency-aware")
        engine, ledger = self.engine(coordinate, self.root / "concurrent-start")
        key = self.attempt_key(coordinate, "PRE", 1)
        run_id = engine._pre_run_id()
        context = engine._context("PRE", 1, run_id)
        _state, created = engine.operation_store.begin(
            operation_id=key,
            phase="PRE",
            attempt=1,
            candidate_id=engine._candidate_id("PRE", 1),
            run_id=run_id,
            context_digest=digest_for(asdict(context)),
        )
        self.assertTrue(created)
        with self.assertRaisesRegex(HeldoutProtocolError, "duplicate generation is forbidden"):
            engine.pre_correction_attempt(1, idempotency_key=key)
        self.assertEqual(self.model.calls, 0)
        self.assertEqual(self.effect_count(), 0)
        ledger.close()

    def test_private_generation_checkpoint_recovers_without_second_model_call(self) -> None:
        coordinate = self.coordinate("dependency-aware")
        root = self.root / "private-checkpoint"
        engine, ledger = self.engine(coordinate, root)
        key = self.attempt_key(coordinate, "PRE", 1)
        with patch.object(
            engine.operation_store,
            "generated",
            side_effect=RuntimeError("crash after private generation checkpoint"),
        ):
            with self.assertRaisesRegex(RuntimeError, "private generation checkpoint"):
                engine.pre_correction_attempt(1, idempotency_key=key)
        self.assertEqual(self.model.calls, 1)
        self.assertEqual(self.effect_count(), 0)
        ledger.close()

        resumed, resumed_ledger = self.engine(coordinate, root)
        observation = resumed.reconcile_attempt(key)
        self.assertIsNotNone(observation)
        self.assertEqual(self.model.calls, 1)
        self.assertEqual(self.effect_count(), 1)
        resumed_ledger.close()

    def test_generated_checkpoint_completes_missing_downstream_materialization(self) -> None:
        coordinate = self.coordinate("dependency-aware")
        root = self.root / "generated-before-downstream"
        engine, ledger = self.engine(coordinate, root)
        key = self.attempt_key(coordinate, "PRE", 1)
        with patch.object(
            engine,
            "_finish_operation",
            side_effect=RuntimeError("crash after durable generated checkpoint"),
        ):
            with self.assertRaisesRegex(RuntimeError, "durable generated checkpoint"):
                engine.pre_correction_attempt(1, idempotency_key=key)
        state = engine.operation_store.load(key)
        assert state is not None
        self.assertEqual(state["status"], "GENERATED")
        self.assertIsNone(
            ledger.connection.execute(
                "SELECT candidate_id FROM candidates WHERE candidate_id=?",
                (state["candidate_id"],),
            ).fetchone()
        )
        self.assertEqual(self.model.calls, 1)
        self.assertEqual(self.effect_count(), 0)
        ledger.close()

        resumed, resumed_ledger = self.engine(coordinate, root)
        observation = resumed.reconcile_attempt(key)
        self.assertIsNotNone(observation)
        completed = resumed.operation_store.load(key)
        assert completed is not None
        self.assertEqual(completed["status"], "COMPLETE")
        self.assertEqual(self.model.calls, 1)
        self.assertEqual(self.effect_count(), 1)
        self.assertEqual(len(resumed_ledger.receipts()), 3)
        resumed_ledger.close()

    def test_dependency_aware_marks_exact_closure_and_excludes_stale_pre_candidates(self) -> None:
        coordinate = self.coordinate("dependency-aware")
        engine, ledger = self.engine(coordinate, self.root / "stale")
        self.run_pre(engine, coordinate)
        _state, affected = self.commit_and_activate(engine, coordinate)
        self.assertTrue(affected)
        self.assertTrue(all(ledger.candidate_disposition(node) == "STALE_DEPENDENT" for node in affected))
        post = engine.post_correction_attempt(
            1, idempotency_key=self.attempt_key(coordinate, "POST", 1)
        )
        self.assertTrue(post.promoted)
        self.assertTrue(post.independent_hidden_fixture_passed)
        with self.assertRaisesRegex(HeldoutProtocolError, "continued after verified recovery"):
            engine.post_correction_attempt(
                2, idempotency_key=self.attempt_key(coordinate, "POST", 2)
            )
        post_record = next(item for item in engine.operation_store.records() if item["phase"] == "POST")
        context, _evidence, _digest = engine.private_store.load_generation_success(
            post_record["candidate_id"]
        )
        retrieved_subjects = {item.get("subject_id") for item in context.retrieval_records}
        self.assertTrue(set(affected).isdisjoint(retrieved_subjects))
        self.assertIn("EVIDENCE", {item["event_type"] for item in context.retrieval_records})
        verification = engine.verification_evidence()
        self.assertEqual(verification.private_replay_decisions, 7)
        self.assertEqual(verification.private_replay_agreements, 7)
        self.assertEqual(verification.public_replay_decisions, 7)
        self.assertEqual(verification.public_replay_agreements, 7)
        self.assertEqual(verification.unauthorized_successful_effects, 0)
        ledger.close()

    def test_policy_clones_freeze_identical_actual_pre_state_and_naive_reuses_stale(self) -> None:
        dependency = self.coordinate("dependency-aware")
        naive = self.coordinate("naive-reuse", block_id=dependency.block_id)
        first, first_ledger = self.engine(dependency, self.root / "clone-dependency")
        self.run_pre(first, dependency)
        first_state = first.freeze_pre_shock_state()
        self.commit_and_activate(first, dependency)
        first_replacement = next(
            event
            for event in first_ledger.events()
            if event["event_type"] == "SHOCK_CORRECTED_PREMISE"
        )
        first_ledger.close()

        second, second_ledger = self.engine(naive, self.root / "clone-naive")
        self.run_pre(second, naive)
        second_state, affected = self.commit_and_activate(second, naive)
        self.assertEqual(first_state, second_state)
        second_replacement = next(
            event
            for event in second_ledger.events()
            if event["event_type"] == "SHOCK_CORRECTED_PREMISE"
        )
        self.assertEqual(first_replacement["event_id"], second_replacement["event_id"])
        self.assertEqual(first_replacement["payload"], second_replacement["payload"])
        observation = second.post_correction_attempt(
            1, idempotency_key=self.attempt_key(naive, "POST", 1)
        )
        post_observations = [observation]
        for attempt in range(2, 7):
            post_observations.append(
                second.post_correction_attempt(
                    attempt,
                    idempotency_key=self.attempt_key(naive, "POST", attempt),
                )
            )
        with self.assertRaisesRegex(HeldoutProtocolError, "outside 1..6"):
            second.post_correction_attempt(
                7, idempotency_key=digest_for({"out-of-range": naive.coordinate_id})
            )
        post_record = next(item for item in second.operation_store.records() if item["phase"] == "POST")
        context, _evidence, _digest = second.private_store.load_generation_success(
            post_record["candidate_id"]
        )
        retrieved_subjects = {item.get("subject_id") for item in context.retrieval_records}
        self.assertTrue(set(affected).issubset(retrieved_subjects))
        self.assertTrue(all(not item.promoted for item in post_observations))
        self.assertTrue(
            all(item.independent_hidden_fixture_passed for item in post_observations)
        )
        self.assertEqual(
            second_ledger.candidate_disposition(post_record["candidate_id"]),
            "STALE_DEPENDENT",
        )
        second_ledger.close()

    def test_full_restart_opens_fresh_post_namespace_with_only_correction(self) -> None:
        coordinate = self.coordinate("full-restart")
        engine, ledger = self.engine(coordinate, self.root / "full-restart")
        self.run_pre(engine, coordinate)
        _state, affected = self.commit_and_activate(engine, coordinate)
        observation = engine.post_correction_attempt(
            1, idempotency_key=self.attempt_key(coordinate, "POST", 1)
        )
        post_record = next(item for item in engine.operation_store.records() if item["phase"] == "POST")
        context, _evidence, _digest = engine.private_store.load_generation_success(
            post_record["candidate_id"]
        )
        self.assertEqual(
            [item["event_type"] for item in context.retrieval_records],
            ["CORRECTION"],
        )
        self.assertTrue(
            set(affected).isdisjoint(
                item.get("subject_id") for item in context.retrieval_records
            )
        )
        self.assertIsNone(context.parent_candidate_id)
        self.assertTrue(observation.promoted)
        ledger.close()

    def test_operation_and_receipt_substitution_are_rejected_on_replay(self) -> None:
        coordinate = self.coordinate("dependency-aware")
        engine, ledger = self.engine(coordinate, self.root / "substitution")
        key = self.attempt_key(coordinate, "PRE", 1)
        engine.pre_correction_attempt(1, idempotency_key=key)
        path = engine.operation_store.root / (key + ".json")
        original = json.loads(path.read_text(encoding="utf-8"))
        value = dict(original)
        value["candidate_source_digest"] = digest_for("substituted-source")
        path.write_bytes(canonical_bytes(value))
        with self.assertRaisesRegex(HeldoutProtocolError, "durable operation state"):
            engine.reconcile_attempt(key)
        path.write_bytes(canonical_bytes(original))

        receipt = dict(ledger.receipts()[1])
        receipt["resource_bucket"] = "OVER_100"
        with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
            ledger.connection.execute(
                "UPDATE receipts SET payload_json=? WHERE receipt_id=?",
                (canonical_json(receipt), receipt["receipt_id"]),
            )
        ledger.connection.rollback()

        second_key = self.attempt_key(coordinate, "PRE", 2)
        engine.pre_correction_attempt(2, idempotency_key=second_key)
        second_state = engine.operation_store.load(second_key)
        assert second_state is not None
        substituted = json.loads(path.read_text(encoding="utf-8"))
        substituted["evaluation_result"]["receipt_ids"][1] = second_state[
            "evaluation_result"
        ]["receipt_ids"][1]
        path.write_bytes(canonical_bytes(substituted))
        with self.assertRaisesRegex(HeldoutProtocolError, "receipt suffix crossed"):
            engine.reconcile_attempt(key)
        ledger.close()

    def test_pre_generation_failure_is_terminal_and_six_attempts_continue(self) -> None:
        coordinate = self.coordinate("dependency-aware")
        root = self.root / "pre-generation-failure"
        self.generator.tokenizer.invalid_decode_calls.add(1)
        engine, ledger = self.engine(coordinate, root)
        key = self.attempt_key(coordinate, "PRE", 1)

        failure = engine.pre_correction_attempt(1, idempotency_key=key)
        self.assertFalse(failure.promoted)
        self.assertFalse(failure.receipt_valid)
        self.assertEqual(failure.diagnostic_enum, "PROTOCOL_VIOLATION")
        self.assertEqual(failure.tokens, 0)
        self.assertEqual(failure.evaluator_seconds, 0.0)
        self.assertIsNone(failure.verdict_receipt_digest)
        self.assertIsNone(failure.effect_receipt_digest)
        state = engine.operation_store.load(key)
        assert state is not None
        self.assertEqual(state["status"], "GENERATION_FAILED")
        self.assertIsNone(state["candidate_source_digest"])
        self.assertIsNone(state["evaluation_result"])
        context, private_failure, private_digest = engine.private_store.load_generation_failure(
            state["candidate_id"]
        )
        self.assertEqual(private_failure.stage, "RESPONSE_CONTRACT")
        self.assertEqual(private_digest, state["generation_evidence_digest"])
        self.assertEqual(digest_for(asdict(context)), state["context_digest"])
        self.assertIsNone(
            ledger.connection.execute(
                "SELECT candidate_id FROM candidates WHERE candidate_id=?",
                (state["candidate_id"],),
            ).fetchone()
        )
        self.assertEqual(len(ledger.receipts()), 0)
        self.assertEqual(self.effect_count(), 0)

        replay = engine.pre_correction_attempt(1, idempotency_key=key)
        self.assertEqual(replay, failure)
        self.assertEqual(self.model.calls, 1)
        observations = [failure]
        for attempt in range(2, 7):
            observations.append(
                engine.pre_correction_attempt(
                    attempt,
                    idempotency_key=self.attempt_key(coordinate, "PRE", attempt),
                )
            )
        self.assertEqual(len(observations), 6)
        self.assertEqual(self.model.calls, 6)
        self.assertEqual(self.effect_count(), 5)
        self.assertEqual(len(ledger.receipts()), 15)
        second = engine.operation_store.load(self.attempt_key(coordinate, "PRE", 2))
        assert second is not None
        second_context, _second_generation, _second_digest = (
            engine.private_store.load_generation_success(second["candidate_id"])
        )
        self.assertNotIn(
            state["candidate_id"],
            {str(item["subject_id"]) for item in second_context.retrieval_records},
        )
        frozen = engine.freeze_pre_shock_state()
        self.assertEqual(
            [item["status"] for item in frozen.candidate_state["candidates"]],
            ["GENERATION_FAILED"] + ["COMPLETE"] * 5,
        )
        _state, affected = self.commit_and_activate(engine, coordinate)
        self.assertIn(state["candidate_id"], affected)
        self.assertEqual(ledger.event_disposition(state["candidate_id"]), "STALE_DEPENDENT")
        verification = engine.verification_evidence()
        self.assertEqual(verification.private_replay_decisions, 6)
        self.assertEqual(verification.private_replay_agreements, 6)
        self.assertEqual(verification.public_replay_decisions, 5)
        self.assertEqual(verification.public_replay_agreements, 5)
        ledger.close()

    def test_generation_failure_checkpoint_crash_replays_without_duplicate_effect(self) -> None:
        coordinate = self.coordinate("dependency-aware")
        root = self.root / "failure-crash-replay"
        self.generator.tokenizer.invalid_decode_calls.add(1)
        engine, ledger = self.engine(coordinate, root)
        key = self.attempt_key(coordinate, "PRE", 1)
        with patch.object(
            engine.operation_store,
            "generation_failed",
            side_effect=RuntimeError("crash after private failure checkpoint"),
        ):
            with self.assertRaisesRegex(RuntimeError, "private failure checkpoint"):
                engine.pre_correction_attempt(1, idempotency_key=key)
        self.assertEqual(self.model.calls, 1)
        self.assertEqual(self.effect_count(), 0)
        self.assertEqual(len(ledger.receipts()), 0)
        started = engine.operation_store.load(key)
        assert started is not None
        self.assertEqual(started["status"], "STARTED")
        engine.private_store.load_generation_failure(started["candidate_id"])
        ledger.close()

        resumed, resumed_ledger = self.engine(coordinate, root)
        observation = resumed.reconcile_attempt(key)
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertFalse(observation.receipt_valid)
        self.assertEqual(resumed.reconcile_attempt(key), observation)
        self.assertEqual(self.model.calls, 1)
        self.assertEqual(self.effect_count(), 0)
        self.assertEqual(len(resumed_ledger.receipts()), 0)
        terminal = resumed.operation_store.load(key)
        assert terminal is not None
        self.assertEqual(terminal["status"], "GENERATION_FAILED")
        failure_events = [
            event
            for event in resumed_ledger.events()
            if event["event_type"] == "SHOCK_GENERATION_FAILURE"
        ]
        self.assertEqual(len(failure_events), 1)
        self.assertEqual(failure_events[0]["subject_id"], terminal["candidate_id"])
        self.assertNotIn("error_code", failure_events[0]["payload"])
        self.assertNotIn("decoded_model_response", failure_events[0]["payload"])
        resumed_ledger.close()

    def test_terminal_failure_missing_event_is_rejected_without_healing(self) -> None:
        self.assert_terminal_failure_missing_evidence_rejected_without_healing(
            event_only=False
        )

    def test_terminal_failure_missing_dependency_is_rejected_without_healing(self) -> None:
        self.assert_terminal_failure_missing_evidence_rejected_without_healing(
            event_only=True
        )

    def test_terminal_failure_missing_private_record_is_rejected_without_healing(
        self,
    ) -> None:
        coordinate = self.coordinate("dependency-aware")
        self.generator.tokenizer.invalid_decode_calls.add(1)
        engine, ledger = self.engine(
            coordinate, self.root / "terminal-failure-missing-private-record"
        )
        key = self.attempt_key(coordinate, "PRE", 1)
        engine.pre_correction_attempt(1, idempotency_key=key)
        state = engine.operation_store.load(key)
        assert state is not None
        record = engine.private_store.generation_records / (
            str(state["candidate_id"]) + ".json"
        )

        self.assert_terminal_private_gap_is_not_healed(
            engine=engine,
            ledger=ledger,
            operation_id=key,
            missing_path=record,
            error="terminal shock generation failure lacks exact private replay evidence",
        )
        ledger.close()

    def test_terminal_failure_missing_private_artifact_is_rejected_without_healing(
        self,
    ) -> None:
        coordinate = self.coordinate("dependency-aware")
        self.generator.tokenizer.invalid_decode_calls.add(1)
        engine, ledger = self.engine(
            coordinate, self.root / "terminal-failure-missing-private-artifact"
        )
        key = self.attempt_key(coordinate, "PRE", 1)
        engine.pre_correction_attempt(1, idempotency_key=key)
        state = engine.operation_store.load(key)
        assert state is not None
        record = json.loads(
            (
                engine.private_store.generation_records
                / (str(state["candidate_id"]) + ".json")
            ).read_text(encoding="utf-8")
        )
        artifact_digest = str(record["contract_response_digest"])
        artifact = (
            engine.private_store.artifacts.root
            / "blobs"
            / "sha256"
            / artifact_digest[:2]
            / artifact_digest[2:4]
            / artifact_digest
        )

        self.assert_terminal_private_gap_is_not_healed(
            engine=engine,
            ledger=ledger,
            operation_id=key,
            missing_path=artifact,
            error="terminal shock generation failure lacks exact private replay evidence",
        )
        ledger.close()

    def test_terminal_success_missing_generation_record_is_rejected_without_healing(
        self,
    ) -> None:
        coordinate = self.coordinate("dependency-aware")
        engine, ledger = self.engine(
            coordinate, self.root / "terminal-success-missing-generation-record"
        )
        key = self.attempt_key(coordinate, "PRE", 1)
        engine.pre_correction_attempt(1, idempotency_key=key)
        state = engine.operation_store.load(key)
        assert state is not None
        record = engine.private_store.generation_records / (
            str(state["candidate_id"]) + ".json"
        )

        self.assert_terminal_private_gap_is_not_healed(
            engine=engine,
            ledger=ledger,
            operation_id=key,
            missing_path=record,
            error="shock model call has no recoverable generation result",
        )
        ledger.close()

    def test_terminal_success_missing_private_artifact_is_rejected_without_healing(
        self,
    ) -> None:
        coordinate = self.coordinate("dependency-aware")
        engine, ledger = self.engine(
            coordinate, self.root / "terminal-success-missing-private-artifact"
        )
        key = self.attempt_key(coordinate, "PRE", 1)
        engine.pre_correction_attempt(1, idempotency_key=key)
        state = engine.operation_store.load(key)
        assert state is not None
        record = json.loads(
            (
                engine.private_store.generation_records
                / (str(state["candidate_id"]) + ".json")
            ).read_text(encoding="utf-8")
        )
        artifact_digest = str(record["contract_response_digest"])
        artifact = (
            engine.private_store.artifacts.root
            / "blobs"
            / "sha256"
            / artifact_digest[:2]
            / artifact_digest[2:4]
            / artifact_digest
        )

        self.assert_terminal_private_gap_is_not_healed(
            engine=engine,
            ledger=ledger,
            operation_id=key,
            missing_path=artifact,
            error="shock model call has no recoverable generation result",
        )
        ledger.close()

    def test_terminal_success_missing_trajectory_record_is_rejected_without_healing(
        self,
    ) -> None:
        coordinate = self.coordinate("dependency-aware")
        engine, ledger = self.engine(
            coordinate, self.root / "terminal-success-missing-trajectory-record"
        )
        key = self.attempt_key(coordinate, "PRE", 1)
        engine.pre_correction_attempt(1, idempotency_key=key)
        state = engine.operation_store.load(key)
        assert state is not None
        record = engine.private_store.records / (
            str(state["candidate_id"]) + ".json"
        )

        self.assert_terminal_private_gap_is_not_healed(
            engine=engine,
            ledger=ledger,
            operation_id=key,
            missing_path=record,
            error="terminal shock candidate lacks its private trajectory",
        )
        ledger.close()

    def test_duplicate_generation_failure_event_is_rejected(self) -> None:
        engine, ledger, failure, records = self.ready_generation_failure_campaign(
            "duplicate-failure-event"
        )
        failed = records[1]
        self.append_spurious_failure_event(
            ledger,
            failure,
            payload=dict(failure["payload"]),
            subject_id=failed["candidate_id"],
            idempotency_key="duplicate-shock-generation-failure:" + failed["operation_id"],
        )
        with self.assertRaisesRegex(HeldoutProtocolError, "event inventory"):
            engine.verification_evidence()
        ledger.close()

    def test_failure_dependency_wrong_campaign_is_rejected(self) -> None:
        self.assert_corrupt_failure_dependency_rejected("campaign")

    def test_failure_dependency_wrong_run_is_rejected(self) -> None:
        self.assert_corrupt_failure_dependency_rejected("run")

    def test_failure_dependency_wrong_task_is_rejected(self) -> None:
        self.assert_corrupt_failure_dependency_rejected("task")

    def test_failure_dependency_wrong_subject_is_rejected(self) -> None:
        self.assert_corrupt_failure_dependency_rejected("subject")

    def test_failure_dependency_wrong_payload_is_rejected(self) -> None:
        self.assert_corrupt_failure_dependency_rejected("payload")

    def test_failure_dependency_wrong_idempotency_is_rejected(self) -> None:
        self.assert_corrupt_failure_dependency_rejected("idempotency")

    def test_failure_dependency_wrong_source_is_rejected(self) -> None:
        self.assert_corrupt_failure_dependency_rejected("source")

    def test_failure_dependency_wrong_disposition_is_rejected(self) -> None:
        self.assert_corrupt_failure_dependency_rejected("disposition")

    def test_orphan_generation_failure_event_is_rejected(self) -> None:
        engine, ledger, failure, _records = self.ready_generation_failure_campaign(
            "orphan-failure-event"
        )
        orphan_operation = digest_for("orphan-shock-operation")
        orphan_candidate = "egv-candidate-orphan-generation-failure"
        payload = dict(failure["payload"])
        payload["operation_id"] = orphan_operation
        payload["candidate_id"] = orphan_candidate
        self.append_spurious_failure_event(
            ledger,
            failure,
            payload=payload,
            subject_id=orphan_candidate,
            idempotency_key="shock-generation-failure:" + orphan_operation,
        )
        with self.assertRaisesRegex(HeldoutProtocolError, "event inventory"):
            engine.verification_evidence()
        ledger.close()

    def test_cross_operation_generation_failure_event_is_rejected(self) -> None:
        engine, ledger, failure, records = self.ready_generation_failure_campaign(
            "cross-operation-failure-event"
        )
        failed = records[1]
        complete = records[2]
        payload = dict(failure["payload"])
        payload["operation_id"] = complete["operation_id"]
        self.append_spurious_failure_event(
            ledger,
            failure,
            payload=payload,
            subject_id=failed["candidate_id"],
            idempotency_key="shock-generation-failure:" + complete["operation_id"],
        )
        with self.assertRaisesRegex(HeldoutProtocolError, "event inventory"):
            engine.verification_evidence()
        ledger.close()

    def test_complete_operation_generation_failure_event_is_rejected(self) -> None:
        engine, ledger, failure, records = self.ready_generation_failure_campaign(
            "complete-operation-failure-event"
        )
        complete = records[2]
        payload = dict(failure["payload"])
        payload["operation_id"] = complete["operation_id"]
        payload["candidate_id"] = complete["candidate_id"]
        self.append_spurious_failure_event(
            ledger,
            failure,
            payload=payload,
            subject_id=complete["candidate_id"],
            idempotency_key="complete-shock-generation-failure:" + complete["operation_id"],
        )
        with self.assertRaisesRegex(HeldoutProtocolError, "event inventory"):
            engine.verification_evidence()
        ledger.close()

    def test_reserved_failure_event_namespace_wrong_type_for_complete_is_rejected(
        self,
    ) -> None:
        engine, ledger, failure, records = self.ready_generation_failure_campaign(
            "wrong-type-complete-failure-event"
        )
        complete = records[2]
        payload = dict(failure["payload"])
        payload.update(
            {
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
        ledger.append_event(
            "NOT_A_GENERATION_FAILURE",
            payload,
            campaign_id=self.protocol.campaign_id,
            run_id=complete["run_id"],
            task_id=self.coordinate("dependency-aware").task_id,
            subject_id=complete["candidate_id"],
            source_class="PINNED_MODEL",
            disposition="REJECTED",
            idempotency_key=(
                "shock-generation-failure:"
                + complete["operation_id"]
                + ":wrong-type"
            ),
        )
        with self.assertRaisesRegex(HeldoutProtocolError, "event inventory"):
            engine.verification_evidence()
        ledger.close()

    def test_spurious_failure_dependency_event_inventory_is_rejected(self) -> None:
        engine, ledger, _failure, records = self.ready_generation_failure_campaign(
            "spurious-failure-dependency-event"
        )
        failed = records[1]
        dependency = next(
            event
            for event in ledger.events()
            if event["event_type"] == "DEPENDENCY"
            and event["subject_id"] == failed["candidate_id"]
        )
        ledger.append_event(
            "DEPENDENCY",
            dict(dependency["payload"]),
            campaign_id=dependency["campaign_id"],
            run_id=dependency["run_id"],
            task_id=dependency["task_id"],
            subject_id=dependency["subject_id"],
            source_class=dependency["source_class"],
            disposition=dependency["disposition"],
            idempotency_key=dependency["idempotency_key"] + ":duplicate",
        )
        with self.assertRaisesRegex(
            HeldoutProtocolError, "dependency event inventory"
        ):
            engine.verification_evidence()
        ledger.close()

    def test_outgoing_dependency_from_failed_subject_is_rejected(self) -> None:
        engine, ledger, _failure, records = self.ready_generation_failure_campaign(
            "outgoing-failure-dependency"
        )
        failed = records[1]
        policy = next(
            event
            for event in ledger.events()
            if event["event_type"] == "SHOCK_POLICY_ACTIVATION"
        )
        ledger.append_dependency(
            failed["candidate_id"],
            policy["event_id"],
            edge_type="FORGED_OUTGOING_FAILURE_EDGE",
            campaign_id=self.protocol.campaign_id,
            run_id=failed["run_id"],
            task_id=self.coordinate("dependency-aware").task_id,
            idempotency_key="forged-outgoing-failure-dependency",
        )
        with self.assertRaisesRegex(
            HeldoutProtocolError, "dependency event inventory"
        ):
            engine.verification_evidence()
        ledger.close()

    def test_reserved_failure_dependency_namespace_wrong_event_type_is_rejected(self) -> None:
        engine, ledger, _failure, records = self.ready_generation_failure_campaign(
            "wrong-type-failure-dependency-event"
        )
        failed = records[1]
        dependency = next(
            event
            for event in ledger.events()
            if event["event_type"] == "DEPENDENCY"
            and event["subject_id"] == failed["candidate_id"]
        )
        ledger.append_event(
            "FORGED_FAILURE_DEPENDENCY",
            dict(dependency["payload"]),
            campaign_id=dependency["campaign_id"],
            run_id=dependency["run_id"],
            task_id=dependency["task_id"],
            subject_id=dependency["subject_id"],
            source_class="FROZEN_PROTOCOL",
            disposition="OBSERVED",
            idempotency_key=(
                "shock-failure-dependency:wrong-event-type:" + failed["candidate_id"]
            ),
        )
        with self.assertRaisesRegex(
            HeldoutProtocolError, "dependency event inventory"
        ):
            engine.verification_evidence()
        ledger.close()

    def test_runner_restart_accepts_exact_pre_generation_failure_journal(self) -> None:
        coordinate = self.coordinate("dependency-aware")
        runtime_root = self.short_runtime_root(".egv-rpf-")
        self.generator.tokenizer.invalid_decode_calls.add(1)
        runner = CorrectionShockCoordinateRunner(
            ShockRuntimeContext(
                self.protocol,
                self.inputs,
                self.sources,
                runtime_root,
                self.production_factory(coordinate),
            )
        )
        original_advance = ShockRuntimeJournal.advance
        crashed = False

        def crash_after_failure_checkpoint(journal, **updates):
            nonlocal crashed
            cursor = original_advance(journal, **updates)
            if (
                not crashed
                and updates.get("pre_attempts") == 1
                and updates.get("pending_attempt") is None
            ):
                crashed = True
                raise RuntimeError("crash after PRE failure journal checkpoint")
            return cursor

        with patch.object(
            ShockRuntimeJournal,
            "advance",
            new=crash_after_failure_checkpoint,
        ):
            with self.assertRaisesRegex(RuntimeError, "PRE failure journal checkpoint"):
                runner(coordinate)
        self.assertEqual(self.model.calls, 1)
        self.assertEqual(self.effect_count(), 0)

        result = runner(coordinate)
        self.assertEqual(result["costs"]["candidate_attempts"], 7)
        self.assertEqual(result["eligible_attempts"], 7)
        self.assertEqual(result["verdict_receipts_required"], 6)
        self.assertEqual(result["verdict_receipts_valid"], 6)
        self.assertIs(result["signature_valid"], True)
        self.assertEqual(result["recovery_attempt"], 1)
        self.assertEqual(self.model.calls, 7)
        self.assertEqual(self.effect_count(), 6)
        journal = ShockRuntimeJournal(
            runtime_root / coordinate.coordinate_id / "shock-journal.json",
            coordinate,
        ).load()
        self.assertEqual(journal["pre_attempts"], 6)
        self.assertFalse(journal["pre_observations"][0]["receipt_valid"])

    def test_runner_restart_accepts_exact_post_generation_failure_journal(self) -> None:
        coordinate = self.coordinate("dependency-aware")
        runtime_root = self.short_runtime_root(".egv-rof-")
        self.generator.tokenizer.invalid_decode_calls.update(range(7, 13))
        runner = CorrectionShockCoordinateRunner(
            ShockRuntimeContext(
                self.protocol,
                self.inputs,
                self.sources,
                runtime_root,
                self.production_factory(coordinate),
            )
        )
        original_advance = ShockRuntimeJournal.advance
        crashed = False

        def crash_after_failure_checkpoint(journal, **updates):
            nonlocal crashed
            cursor = original_advance(journal, **updates)
            if (
                not crashed
                and updates.get("post_attempts") == 1
                and updates.get("pending_attempt") is None
            ):
                crashed = True
                raise RuntimeError("crash after POST failure journal checkpoint")
            return cursor

        with patch.object(
            ShockRuntimeJournal,
            "advance",
            new=crash_after_failure_checkpoint,
        ):
            with self.assertRaisesRegex(RuntimeError, "POST failure journal checkpoint"):
                runner(coordinate)
        self.assertEqual(self.model.calls, 7)
        self.assertEqual(self.effect_count(), 6)

        result = runner(coordinate)
        self.assertEqual(result["costs"]["candidate_attempts"], 12)
        self.assertEqual(result["eligible_attempts"], 12)
        self.assertEqual(result["verdict_receipts_required"], 6)
        self.assertEqual(result["verdict_receipts_valid"], 6)
        self.assertIs(result["signature_valid"], True)
        self.assertIs(result["recovered_within_six"], False)
        self.assertEqual(self.model.calls, 12)
        self.assertEqual(self.effect_count(), 6)
        journal = ShockRuntimeJournal(
            runtime_root / coordinate.coordinate_id / "shock-journal.json",
            coordinate,
        ).load()
        self.assertEqual(journal["post_attempts"], 6)
        self.assertTrue(
            all(not observation["receipt_valid"] for observation in journal["post_observations"])
        )

    def test_six_post_generation_failures_are_counted_and_never_retrieved(self) -> None:
        coordinate = self.coordinate("dependency-aware")
        engine, ledger = self.engine(coordinate, self.root / "post-generation-failures")
        self.run_pre(engine, coordinate)
        self.commit_and_activate(engine, coordinate)
        self.generator.tokenizer.invalid_decode_calls.update(range(7, 13))

        observations = []
        for attempt in range(1, 7):
            observations.append(
                engine.post_correction_attempt(
                    attempt,
                    idempotency_key=self.attempt_key(coordinate, "POST", attempt),
                )
            )
        self.assertTrue(all(not item.promoted for item in observations))
        self.assertTrue(all(not item.receipt_valid for item in observations))
        self.assertTrue(all(not item.independent_hidden_fixture_passed for item in observations))
        self.assertEqual(self.model.calls, 12)
        self.assertEqual(self.effect_count(), 6)
        self.assertEqual(len(ledger.receipts()), 18)
        post_records = [
            item for item in engine.operation_store.records() if item["phase"] == "POST"
        ]
        self.assertEqual(len(post_records), 6)
        self.assertTrue(all(item["status"] == "GENERATION_FAILED" for item in post_records))
        failure_ids = {str(item["candidate_id"]) for item in post_records}
        for record in post_records:
            context, _failure, _digest = engine.private_store.load_generation_failure(
                record["candidate_id"]
            )
            retrieved = {str(item["subject_id"]) for item in context.retrieval_records}
            self.assertTrue(failure_ids.isdisjoint(retrieved))
            self.assertIsNone(
                ledger.connection.execute(
                    "SELECT candidate_id FROM candidates WHERE candidate_id=?",
                    (record["candidate_id"],),
                ).fetchone()
            )
        last_key = self.attempt_key(coordinate, "POST", 6)
        self.assertEqual(engine.reconcile_attempt(last_key), observations[-1])
        self.assertEqual(self.model.calls, 12)
        self.assertEqual(self.effect_count(), 6)
        verification = engine.verification_evidence()
        self.assertEqual(verification.private_replay_decisions, 12)
        self.assertEqual(verification.private_replay_agreements, 12)
        self.assertEqual(verification.public_replay_decisions, 6)
        self.assertEqual(verification.public_replay_agreements, 6)
        ledger.close()

    def test_production_factory_rejects_base_generator_without_trained_adapter(self) -> None:
        coordinate = self.coordinate("dependency-aware")
        base_generator, _model = _generator()
        with self.assertRaisesRegex(HeldoutProtocolError, "trained adapter"):
            ProductionShockEngineFactory(
                generator=base_generator,
                evaluator_manifest=self.manifest(coordinate),
                evaluator_public_key=self.public_key,
                evaluator_command=self.command,
                evaluator_python_executable=Path(sys.executable).resolve(),
                evaluator_python_digest=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
                source_commit="test-source",
                model_revision=MODEL_REVISION,
                data_manifest_digest=self.data_manifest_digest,
                response_contract_digest=base_generator.response_contract_digest,
                generation_profile_digest=base_generator.generation_profile_digest,
            )


if __name__ == "__main__":
    unittest.main()
