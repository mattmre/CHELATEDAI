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

from egv.canonical import canonical_bytes, canonical_json, digest_for
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


def _protocol(corpus: EvaluationCorpus) -> FrozenHeldoutProtocol:
    bindings = {
        name: digest_for({"binding": name})
        for name in FrozenHeldoutProtocol.REQUIRED_BINDINGS
    }
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
        self.protocol = _protocol(self.corpus)
        raw_inputs, raw_sources = build_trainer_evidence_package(
            self.corpus,
            self.protocol,
            generation_profile_digest=digest_for("placeholder-generation"),
        )
        self.inputs = HeldoutTrainerInputs(raw_inputs, protocol=self.protocol)
        self.sources = HeldoutTrainerSources(raw_sources, trainer_inputs=self.inputs)
        self.generator, self.model = _generator(self.root / "sealed-adapter")
        self.signer = ReceiptSigner(b"R" * 32)
        self.public_key = self.root / "evaluator.pub"
        self.public_key.write_bytes(self.signer.public_key_raw)
        self.remote_state = self.root / "remote-state"
        self.command = self.root / "remote-evaluator.py"
        self.command.write_text(
            RESPONDER.replace("__PRIVATE_KEY__", self.signer.private_key_raw.hex()).replace(
                "__STATE_ROOT__", repr(str(self.remote_state))
            ),
            encoding="utf-8",
        )
        self.data_manifest_digest = self.corpus.manifest_digest()
        self.manifests = {}
        self.ledgers: list[EvidenceLedger] = []

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

    def engine(self, coordinate, root: Path) -> tuple[ProductionShockAttemptEngine, EvidenceLedger]:
        root.mkdir(parents=True, exist_ok=True)
        ledger = EvidenceLedger(root / "ledger.sqlite3")
        self.ledgers.append(ledger)
        self.generator.tokenizer.function_name = self.inputs.tasks[coordinate.task_id][
            "public_locus"
        ].rsplit(":", 1)[-1]
        factory = ProductionShockEngineFactory(
            generator=self.generator,
            evaluator_manifest=self.manifest(coordinate),
            evaluator_public_key=self.public_key,
            evaluator_command=self.command,
            source_commit="test-shock-source-commit",
            model_revision=MODEL_REVISION,
            data_manifest_digest=self.data_manifest_digest,
            response_contract_digest=self.generator.response_contract_digest,
            generation_profile_digest=self.generator.generation_profile_digest,
        )
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
        ledger.close()

        resumed, resumed_ledger = self.engine(coordinate, root)
        observation = resumed.reconcile_attempt(key)
        self.assertIsNotNone(observation)
        self.assertEqual(self.model.calls, 1)
        self.assertEqual(self.effect_count(), 1)
        self.assertEqual(len(resumed_ledger.receipts()), 3)
        resumed_ledger.close()

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
        with self.assertRaisesRegex(HeldoutProtocolError, "evaluator replay changed"):
            engine.reconcile_attempt(key)
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
                source_commit="test-source",
                model_revision=MODEL_REVISION,
                data_manifest_digest=self.data_manifest_digest,
                response_contract_digest=base_generator.response_contract_digest,
                generation_profile_digest=base_generator.generation_profile_digest,
            )


if __name__ == "__main__":
    unittest.main()
