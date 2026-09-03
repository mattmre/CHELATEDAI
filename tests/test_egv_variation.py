"""Focused tests for the bounded EGV Variation slice."""

from __future__ import annotations

import json
import hashlib
import os
from dataclasses import replace
import inspect
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from weakref import WeakKeyDictionary

from egv.canonical import (
    canonical_bytes,
    canonical_json,
    chain_digest,
    content_id,
    digest_bytes,
    digest_for,
    failure_family_root,
)
from egv.evaluation.authority import AuthorityBroker, AuthorityPolicy
from egv.evaluation.controller import EvaluationResult, EvaluatorController, HiddenEvaluatorRunner
from egv.evaluation.dataset import EvaluationCorpus, EVALUATOR_SEED_BYTES, FAMILY_SPECS
from egv.evaluation.prompts import PromptRegistry
from egv.evaluation.sandbox import DockerCandidateSandbox, DockerSandboxConfig
from egv.evaluation.errors import DockerConfigurationError, LeakageError
from egv.ledger import EvidenceLedger
from egv.receipts import ReceiptJournal, ReceiptSigner
from egv.training.contracts import LedgerCutoff
from egv.training.dataset import TrajectoryDatasetBuilder
from egv.variation import (
    ADAPTER_MANIFEST_NAME,
    ARM_IDS,
    ArmIsolation,
    AdapterApplicationAttestation,
    BoundedCandidateLoop,
    CheckpointStore,
    ControllerEvaluationGateway,
    DeterministicFixtureGenerator,
    LoadedPinnedModel,
    MODEL_REVISION,
    ModelCandidateGenerator,
    PinnedModelManifest,
    PinnedModelLoader,
    PrivateTrajectoryStore,
    SealedAdapterArtifact,
    SourceContractBudgetExhausted,
    VariationBudgetError,
    VariationCheckpointError,
    VariationConfigurationError,
    VariationDependencyError,
    VariationIsolationError,
    VariationTask,
    build_local_adapter_manifest,
    arm_policy,
    build_local_manifest,
    retrieval_policy,
    run_variation_smoke,
    scan_public_variation_report,
)
from egv.variation.fixture import FixtureEvaluationGateway
from egv.variation.generator import (
    CandidateContext,
    CandidateGenerationEvidence,
    CandidateProposal,
    render_candidate_prompt,
)
import egv.variation.loop as variation_loop
from egv.variation.loop import CANDIDATE_SOURCE_LIMIT
from egv.variation.model import MODEL_ARCHITECTURE, MODEL_CONFIG_CLASS, MODEL_MANIFEST_SCHEMA, MODEL_REPOSITORY
from egv.variation.remote import REMOTE_VARIATION_SERVICE_SCHEMA, RemoteControllerEvaluationGateway


class VariationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="egv-variation-test-")
        self.root = Path(self.tempdir.name)
        self.seed_path = self.root / "private" / "corpus-seed.bin"
        self.seed_path.parent.mkdir(parents=True, exist_ok=True)
        self.seed_path.write_bytes(b"V" * EVALUATOR_SEED_BYTES)
        self.seed_path.chmod(0o600)
        self.corpus = EvaluationCorpus.generate(secret_seed_file=self.seed_path)
        self.ledger = None

    def tearDown(self) -> None:
        if self.ledger is not None:
            self.ledger.close()
        self.tempdir.cleanup()

    def make_runner(
        self,
        *,
        arm_id: str = "C",
        candidates=None,
        max_attempts: int = 2,
        fixture_mode: bool = True,
        campaign_id: str = "variation-test-campaign",
        generator_factory=None,
        evaluator_factory=None,
    ):
        repo = self.corpus.hidden_repositories()[0]
        task = VariationTask.from_microrepo(repo)
        public_source = dict(repo.source_files)["src/task.py"]
        sources = candidates or (public_source, repo.corrected_source)
        model_digest = digest_for({"fixture_model": MODEL_REVISION})
        policy_digest = digest_for({"policy": "variation-test-v1"})
        self.ledger = EvidenceLedger(
            self.root / "ledger.sqlite",
            blob_root=self.root / "ledger-blobs",
            clock=lambda: "2026-08-22T00:00:00Z",
        )
        evaluator = (
            evaluator_factory(self.corpus, self.ledger, policy_digest, campaign_id)
            if evaluator_factory is not None
            else FixtureEvaluationGateway(
                self.corpus,
                self.ledger,
                self.root / "fixture-evaluator",
                policy_digest=policy_digest,
                campaign_id=campaign_id,
            )
        )
        isolation = ArmIsolation(self.root / "arm-state", campaign_id=campaign_id)
        generator = (
            generator_factory(task, repo, model_digest)
            if generator_factory is not None
            else DeterministicFixtureGenerator(
                {task.task_id: tuple(sources)},
                public_locus={task.task_id: task.public_locus},
                model_digest=model_digest,
            )
        )
        try:
            runner = BoundedCandidateLoop(
                ledger=self.ledger,
                evaluator=evaluator,
                generator=generator,
                isolation=isolation,
                workspace_root=self.root / "variation-state",
                campaign_id=campaign_id,
                source_commit="variation-test-source",
                model_revision=MODEL_REVISION,
                model_digest=model_digest,
                data_manifest_digest=self.corpus.manifest_digest(),
                policy_digest=policy_digest,
                arm_id=arm_id,
                max_attempts=max_attempts,
                seed_set=(0, 1, 2),
                fixture_mode=fixture_mode,
            )
        except Exception:
            self.ledger.close()
            self.ledger = None
            raise
        return runner, task, repo, isolation

    def make_adapter_artifact(self) -> SealedAdapterArtifact:
        root = self.root / "sealed-adapter"
        root.mkdir(parents=True, exist_ok=True)
        (root / "adapter_config.json").write_text(
            '{"lora_alpha":8,"peft_type":"LORA","r":4}\n', encoding="utf-8"
        )
        (root / "adapter_model.safetensors").write_bytes(b"sealed-training-output")
        manifest = build_local_adapter_manifest(root)
        (root / ADAPTER_MANIFEST_NAME).write_text(canonical_json(manifest.to_dict()) + "\n", encoding="utf-8")
        return SealedAdapterArtifact(root)

    def make_production_generator(self):
        manifest = PinnedModelManifest(
            repository="Qwen/Qwen3.5-2B-Base",
            revision="b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
            architecture="Qwen3_5ForCausalLM",
            config_class="Qwen3_5TextConfig",
            transformers_version="5.5.0",
            files={"weights.safetensors": "a" * 64},
            license={"name": "test", "source": "test"},
        )
        manifest.validate_contract()
        loaded = LoadedPinnedModel(
            model=SimpleNamespace(),
            tokenizer=SimpleNamespace(),
            manifest=manifest,
            manifest_digest=manifest.digest(),
            file_hashes=dict(manifest.files),
            load_report={},
            base_state_digest=digest_for("base-state"),
        )
        return ModelCandidateGenerator(loaded, model_digest=manifest.digest()), manifest.digest()

    def make_source_contract_runner(self, responses, *, max_attempts: int = 12):
        class Device:
            type = "cuda"

        device = Device()

        class Tensor:
            def __init__(self, values, tensor_device=None):
                self.values = list(values)
                self.device = tensor_device
                self.shape = (1, len(self.values))

            def to(self, *, device):
                return Tensor(self.values, device)

        class Model:
            def __init__(self):
                self.generate_calls = 0

            def parameters(self):
                return iter((SimpleNamespace(device=device),))

            def generate(self, **kwargs):
                self.generate_calls += 1
                return [kwargs["input_ids"].values + [100 + self.generate_calls]]

        class Tokenizer:
            chat_template = "source-contract-retry-template-v1"

            def __init__(self, values):
                self.responses = list(values)
                self.decode_calls = 0

            def apply_chat_template(self, messages, **kwargs):
                return "<sealed-chat>\n" + messages[0]["content"]

            def __call__(self, text, **kwargs):
                return {"input_ids": Tensor([1, 2]), "attention_mask": Tensor([1, 1])}

            def decode(self, tokens, *, skip_special_tokens):
                self.decode_calls += 1
                if not self.responses:
                    raise AssertionError("source-contract test generated beyond its frozen response sequence")
                return self.responses.pop(0)

        base, model_digest = self.make_production_generator()
        model = Model()
        tokenizer = Tokenizer(responses)
        loaded = LoadedPinnedModel(
            model=model,
            tokenizer=tokenizer,
            manifest=base.loaded_model.manifest,
            manifest_digest=model_digest,
            file_hashes=base.loaded_model.file_hashes,
            load_report={},
            base_state_digest=base.loaded_model.base_state_digest,
        )
        generator = ModelCandidateGenerator(
            loaded,
            model_digest=model_digest,
            response_contract="source-only-v1",
        )
        repo = self.corpus.split("train")[0]
        task = VariationTask.from_microrepo(repo)
        initial_source = dict(repo.source_files)["src/task.py"]
        campaign_id = "source-contract-retry-campaign"
        policy_digest = digest_for({"policy": "source-contract-retry-v1"})
        self.ledger = EvidenceLedger(
            self.root / "source-contract-ledger.sqlite",
            blob_root=self.root / "source-contract-ledger-blobs",
            clock=lambda: "2026-08-24T00:00:00Z",
        )
        class TrainFixtureGateway(FixtureEvaluationGateway):
            def _common(self, *, task_id, candidate_id, artifact_digest):
                value = super()._common(
                    task_id=task_id,
                    candidate_id=candidate_id,
                    artifact_digest=artifact_digest,
                )
                row = self.ledger.connection.execute(
                    "SELECT run_id FROM candidates WHERE candidate_id=?",
                    (candidate_id,),
                ).fetchone()
                value["run_id"] = str(row["run_id"])
                return value

        evaluator = TrainFixtureGateway(
            self.corpus,
            self.ledger,
            self.root / "source-contract-evaluator",
            policy_digest=policy_digest,
            campaign_id=campaign_id,
        )
        evaluator.hidden_runner = HiddenEvaluatorRunner(
            {
                item.template_id: (
                    item.evaluator_input,
                    canonical_bytes(item.expected_output) + b"\n",
                    item.hidden_spec.get("resource_limit"),
                )
                for item in self.corpus.repositories
            },
            evaluator_revision=evaluator.evaluator_revision,
            public_loci={item.template_id: item.public_locus for item in self.corpus.repositories},
            public_records={
                item.template_id: item.public_manifest_record()
                for item in self.corpus.repositories
            },
        )
        isolation = ArmIsolation(self.root / "source-contract-arm-state", campaign_id=campaign_id)
        private_store = PrivateTrajectoryStore(self.root / "source-contract-private")
        runner = BoundedCandidateLoop(
            ledger=self.ledger,
            evaluator=evaluator,
            generator=generator,
            isolation=isolation,
            workspace_root=self.root / "source-contract-variation-state",
            campaign_id=campaign_id,
            source_commit="source-contract-test-source",
            model_revision=MODEL_REVISION,
            model_digest=model_digest,
            data_manifest_digest=self.corpus.manifest_digest(),
            policy_digest=policy_digest,
            arm_id="B",
            max_attempts=max_attempts,
            seed_set=(0,),
            fixture_mode=True,
            private_store=private_store,
            initial_source=initial_source,
            response_contract_digest=generator.response_contract_digest,
            generation_profile_digest=generator.generation_profile_digest,
        )
        return runner, task, repo, generator, tokenizer, model, private_store

    def test_controller_gateway_binds_real_controller_hidden_runner_and_docker(self) -> None:
        hidden_runner = HiddenEvaluatorRunner.from_corpus(self.corpus, evaluator_revision="gateway-test-evaluator")
        forged_signer = ReceiptSigner(b"F" * 32)
        with self.assertRaises(VariationDependencyError):
            ControllerEvaluationGateway(
                EvaluatorController(
                    sandbox=object.__new__(DockerCandidateSandbox),
                    hidden_runner=hidden_runner,
                    broker=AuthorityBroker(None),
                    signer=forged_signer,
                    journal=ReceiptJournal(self.root / "forged-gateway-receipts.jsonl", forged_signer.public_key),
                    ingest=lambda _receipt: None,
                    campaign_id="gateway-campaign",
                    protocol_digest=digest_for("gateway-protocol"),
                    policy_digest=digest_for("gateway-policy"),
                )
            )
        try:
            docker_sandbox = DockerCandidateSandbox(self.root / "gateway-docker")
        except DockerConfigurationError as exc:
            self.skipTest("cached pinned Docker image unavailable: {}".format(exc))
        signer = ReceiptSigner(b"G" * 32)
        controller = EvaluatorController(
            sandbox=docker_sandbox,
            hidden_runner=hidden_runner,
            broker=AuthorityBroker(None),
            signer=signer,
            journal=ReceiptJournal(self.root / "gateway-receipts.jsonl", signer.public_key),
            ingest=lambda _receipt: None,
            campaign_id="gateway-campaign",
            protocol_digest=digest_for("gateway-protocol"),
            policy_digest=digest_for("gateway-policy"),
        )
        gateway = ControllerEvaluationGateway(controller)
        self.assertIs(gateway.hidden_runner, hidden_runner)
        gateway.validate_runtime()
        controller.sandbox = SimpleNamespace(enforceable=True)
        with self.assertRaises(VariationDependencyError):
            gateway.validate_runtime()

    def test_controller_gateway_rejects_patched_instance_execute_and_image_binding(self) -> None:
        hidden_runner = HiddenEvaluatorRunner.from_corpus(self.corpus, evaluator_revision="patched-gateway-evaluator")
        try:
            docker_sandbox = DockerCandidateSandbox(self.root / "patched-gateway-docker")
        except DockerConfigurationError as exc:
            self.skipTest("cached pinned Docker image unavailable: {}".format(exc))
        patched_signer = ReceiptSigner(b"P" * 32)
        controller = EvaluatorController(
            sandbox=docker_sandbox,
            hidden_runner=hidden_runner,
            broker=AuthorityBroker(None),
            signer=patched_signer,
            journal=ReceiptJournal(
                self.root / "patched-gateway-receipts.jsonl", patched_signer.public_key
            ),
            ingest=lambda _receipt: None,
            campaign_id="patched-gateway-campaign",
            protocol_digest=digest_for("patched-gateway-protocol"),
            policy_digest=digest_for("patched-gateway-policy"),
        )
        gateway = ControllerEvaluationGateway(controller)
        docker_sandbox.execute = lambda **_kwargs: None  # type: ignore[attr-defined]
        with self.assertRaises(VariationDependencyError):
            gateway.validate_runtime()
        del docker_sandbox.__dict__["execute"]
        docker_sandbox.image_id = "sha256:" + ("b" * 64)
        with self.assertRaises(VariationDependencyError):
            gateway.validate_runtime()
        controller.sandbox = docker_sandbox
        controller.hidden_runner = SimpleNamespace(evaluator_revision="forged")
        with self.assertRaises(VariationDependencyError):
            gateway.validate_runtime()

    def test_production_run_revalidates_controller_and_docker_boundary(self) -> None:
        hidden_runner = HiddenEvaluatorRunner.from_corpus(self.corpus, evaluator_revision="run-boundary-evaluator")
        try:
            docker_sandbox = DockerCandidateSandbox(self.root / "run-boundary-docker")
        except DockerConfigurationError as exc:
            self.skipTest("cached pinned Docker image unavailable: {}".format(exc))
        signer = ReceiptSigner(b"R" * 32)
        controller = EvaluatorController(
            sandbox=docker_sandbox,
            hidden_runner=hidden_runner,
            broker=AuthorityBroker(None),
            signer=signer,
            journal=ReceiptJournal(self.root / "run-boundary-receipts.jsonl", signer.public_key),
            ingest=lambda _receipt: None,
            campaign_id="run-boundary-campaign",
            protocol_digest=digest_for("run-boundary-protocol"),
            policy_digest=digest_for("run-boundary-policy"),
        )
        gateway = ControllerEvaluationGateway(controller)
        self.ledger = EvidenceLedger(
            self.root / "run-boundary.sqlite",
            blob_root=self.root / "run-boundary-blobs",
            clock=lambda: "2026-08-22T00:00:00Z",
        )
        repo = self.corpus.hidden_repositories()[0]
        task = VariationTask.from_microrepo(repo)
        generator, model_digest = self.make_production_generator()
        runner = BoundedCandidateLoop(
            ledger=self.ledger,
            evaluator=gateway,
            generator=generator,
            isolation=ArmIsolation(self.root / "run-boundary-arms", campaign_id="run-boundary-campaign"),
            workspace_root=self.root / "run-boundary-state",
            campaign_id="run-boundary-campaign",
            source_commit="run-boundary-source",
            model_revision=MODEL_REVISION,
            model_digest=model_digest,
            data_manifest_digest=self.corpus.manifest_digest(),
            policy_digest=digest_for("run-boundary-policy"),
            arm_id="D",
            max_attempts=1,
            seed_set=(0,),
            fixture_mode=False,
        )
        controller.sandbox = SimpleNamespace(enforceable=True)
        with self.assertRaises(VariationDependencyError):
            runner.run(task, seed=0)

    def test_forged_promoted_checkpoint_requires_promoted_ledger_candidate(self) -> None:
        class StopBeforeSecond(DeterministicFixtureGenerator):
            def propose(self, context):
                if context.attempt_index == 2:
                    raise RuntimeError("intentional interruption")
                return super().propose(context)

        def generator_factory(task, repo, model_digest):
            public_source = dict(repo.source_files)["src/task.py"]
            return StopBeforeSecond(
                {task.task_id: (public_source, repo.corrected_source)},
                public_locus={task.task_id: task.public_locus},
                model_digest=model_digest,
            )

        runner, task, repo, isolation = self.make_runner(max_attempts=3, generator_factory=generator_factory)
        with self.assertRaises(RuntimeError):
            runner.run(task, seed=0)
        checkpoint_path, checkpoint = CheckpointStore(isolation.workspace("C", runner._run_id(task.task_id, 0)).checkpoints).latest(
            run_id=runner._run_id(task.task_id, 0)
        )  # type: ignore[misc]
        durable = runner._durable_attempts(run_id=runner._run_id(task.task_id, 0), task_id=task.task_id)
        forged_state = digest_for(
            {
                "attempts": [item.to_dict() for item in durable],
                "run_id": runner._run_id(task.task_id, 0),
                "arm_id": "C",
                "task_id": task.task_id,
                "status": "PROMOTED",
            }
        )
        forged = replace(checkpoint, status="PROMOTED", state_digest=forged_state)
        forged_path = CheckpointStore(checkpoint_path.parent).save(forged)
        with self.assertRaises(VariationCheckpointError):
            runner.run(task, seed=0, resume_from=forged_path)

    def test_checkpoint_rejects_unsupported_terminal_status(self) -> None:
        runner, task, _repo, isolation = self.make_runner()
        report = runner.run(task, seed=0)
        path, checkpoint = CheckpointStore(isolation.workspace("C", report.run_id).checkpoints).latest(run_id=report.run_id)  # type: ignore[misc]
        value = checkpoint.to_dict()
        value["status"] = "REJECTED"
        forged_path = path.parent / "checkpoint-{}.json".format(digest_for(value))
        forged_path.write_text(canonical_json(value) + "\n", encoding="utf-8")
        with self.assertRaises(VariationCheckpointError):
            CheckpointStore(path.parent).load(forged_path)

    def test_attempt_budget_is_read_only_and_revalidated_at_run_boundary(self) -> None:
        runner, _task, _repo, _isolation = self.make_runner(max_attempts=12)
        self.assertEqual(runner.max_attempts, 12)
        with self.assertRaises(AttributeError):
            runner.max_attempts = 13  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            runner._max_attempts = 13  # type: ignore[misc]
        object.__setattr__(runner, "_max_attempts", 13)  # adversarial bypass
        with self.assertRaises(VariationBudgetError):
            runner.run(_task, seed=0)

    def test_runtime_identity_seal_rejects_fixture_evaluator_and_generator_swaps(self) -> None:
        runner, task, _repo, _isolation = self.make_runner()
        with self.assertRaises(AttributeError):
            runner.fixture_mode = False  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            runner.evaluator = object()  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            runner.generator = object()  # type: ignore[misc]

        forged_mode = runner
        object.__setattr__(forged_mode, "fixture_mode", False)
        with self.assertRaises(VariationDependencyError):
            forged_mode.run(task, seed=0)
        object.__setattr__(forged_mode, "fixture_mode", True)

        forged_evaluator = runner
        object.__setattr__(forged_evaluator, "evaluator", object())
        with self.assertRaises(VariationDependencyError):
            forged_evaluator.run(task, seed=0)
        object.__setattr__(forged_evaluator, "evaluator", runner.evaluator)

        forged_generator = runner
        object.__setattr__(forged_generator, "generator", object())
        with self.assertRaises(VariationDependencyError):
            forged_generator.run(task, seed=0)

    def test_runtime_identity_authority_rejects_wholesale_instance_dict_rewrite(self) -> None:
        hidden_runner = HiddenEvaluatorRunner.from_corpus(self.corpus, evaluator_revision="dict-attack-evaluator")
        try:
            docker_sandbox = DockerCandidateSandbox(self.root / "dict-attack-docker")
        except DockerConfigurationError as exc:
            self.skipTest("cached pinned Docker image unavailable: {}".format(exc))
        signer = ReceiptSigner(b"W" * 32)
        controller = EvaluatorController(
            sandbox=docker_sandbox,
            hidden_runner=hidden_runner,
            broker=AuthorityBroker(None),
            signer=signer,
            journal=ReceiptJournal(self.root / "dict-attack-receipts.jsonl", signer.public_key),
            ingest=lambda _receipt: None,
            campaign_id="dict-attack-campaign",
            protocol_digest=digest_for("dict-attack-protocol"),
            policy_digest=digest_for("dict-attack-policy"),
        )
        self.ledger = EvidenceLedger(
            self.root / "dict-attack.sqlite",
            blob_root=self.root / "dict-attack-blobs",
            clock=lambda: "2026-08-22T00:00:00Z",
        )
        gateway = ControllerEvaluationGateway(controller)
        generator, model_digest = self.make_production_generator()
        repo = self.corpus.hidden_repositories()[0]
        task = VariationTask.from_microrepo(repo)
        isolation = ArmIsolation(self.root / "dict-attack-arms", campaign_id="dict-attack-campaign")
        runner = BoundedCandidateLoop(
            ledger=self.ledger,
            evaluator=gateway,
            generator=generator,
            isolation=isolation,
            workspace_root=self.root / "dict-attack-state",
            campaign_id="dict-attack-campaign",
            source_commit="dict-attack-source",
            model_revision=MODEL_REVISION,
            model_digest=model_digest,
            data_manifest_digest=self.corpus.manifest_digest(),
            policy_digest=digest_for("dict-attack-policy"),
            arm_id="D",
            max_attempts=1,
            seed_set=(0,),
            fixture_mode=False,
        )
        fixture_probe = BoundedCandidateLoop.__new__(BoundedCandidateLoop, fixture_mode=True)
        self.assertIsNot(type(runner), type(fixture_probe))
        self.assertIsNot(type(runner).run, type(fixture_probe).run)
        self.assertNotIn("_validate_fixture_boundary", inspect.getsource(type(runner).run))
        self.assertNotIn("fixture_mode", inspect.getsource(type(runner).run))
        with self.assertRaises(TypeError):
            runner.__class__ = type(fixture_probe)
        fixture_evaluator = FixtureEvaluationGateway(
            self.corpus,
            self.ledger,
            self.root / "dict-attack-fixture",
            policy_digest=digest_for("dict-attack-policy"),
            campaign_id="dict-attack-campaign",
        )
        fixture_generator = object()
        fixture_isolation = object()
        registry_cell = variation_loop._register_runtime_identity.__closure__[1]
        self.assertIsInstance(registry_cell.cell_contents, WeakKeyDictionary)
        registry_cell.cell_contents[runner] = variation_loop._RuntimeIdentityRecord(
            fixture_mode=True,
            evaluator=fixture_evaluator,
            generator=fixture_generator,
            isolation=fixture_isolation,
            private_store=runner.private_store,
        )
        forged = dict(runner.__dict__)
        forged.update(
            {
                "fixture_mode": True,
                "evaluator": fixture_evaluator,
                "generator": fixture_generator,
                "isolation": fixture_isolation,
                "_frozen_fixture_mode": True,
                "_frozen_evaluator": object(),
                "_frozen_generator": object(),
                "_frozen_isolation": object(),
                "_identity_contract": digest_for("forged-wholesale-contract"),
            }
        )
        runner.__dict__.clear()
        runner.__dict__.update(forged)
        with self.assertRaises(VariationDependencyError):
            runner.run(task, seed=0)
        self.assertEqual(
            [event for event in self.ledger.current_valid_events() if event.get("event_type") == "VARIATION_ATTEMPT"],
            [],
        )

    def test_lora_requires_verified_exhaustive_sealed_adapter_not_hex(self) -> None:
        with self.assertRaises(VariationDependencyError):
            self.make_runner(arm_id="E", max_attempts=1)
        with self.assertRaises(VariationDependencyError):
            DeterministicFixtureGenerator(
                {"task": (b"def main(value):\n    return value\n",)},
                public_locus={"task": "src/task.py:main"},
                model_digest=digest_for("model"),
                adapter_digest=digest_for("arbitrary-hex"),
            )
        artifact = self.make_adapter_artifact()
        self.assertEqual(artifact.digest, artifact.manifest.digest)
        self.assertEqual(set(artifact.verify()), set(artifact.manifest.files))
        (artifact.root / "unexpected.bin").write_bytes(b"extra")
        with self.assertRaises(VariationConfigurationError):
            artifact.verify()

    def test_lora_arm_rejects_duck_loaded_model_and_non_applied_adapter(self) -> None:
        artifact = self.make_adapter_artifact()
        generator, model_digest = self.make_production_generator()
        with self.assertRaises(VariationDependencyError):
            ModelCandidateGenerator(
                generator.loaded_model,
                model_digest=model_digest,
                adapter_digest=artifact.digest,
                adapter_artifact=artifact,
            )
        with self.assertRaises(VariationConfigurationError):
            ModelCandidateGenerator(
                SimpleNamespace(model=SimpleNamespace(), tokenizer=SimpleNamespace(), manifest_digest=model_digest),
                model_digest=model_digest,
            )

    def test_importable_attestation_sentinel_cannot_authorize_dummy_model(self) -> None:
        from egv.variation.model import ADAPTER_ATTESTATION_SCHEMA, _ADAPTER_ATTESTATION_TOKEN, model_state_digest

        artifact = self.make_adapter_artifact()
        base_generator, model_digest = self.make_production_generator()
        dummy_model = SimpleNamespace(state_dict=lambda: {})
        attestation = AdapterApplicationAttestation(
            schema_version=ADAPTER_ATTESTATION_SCHEMA,
            adapter_digest=artifact.digest,
            base_model_manifest_digest=model_digest,
            base_state_digest=digest_for("forged-base-state"),
            applied_model_state_digest=model_state_digest(dummy_model),
            issuer_token=_ADAPTER_ATTESTATION_TOKEN,
        )
        loaded = LoadedPinnedModel(
            model=dummy_model,
            tokenizer=base_generator.tokenizer,
            manifest=base_generator.loaded_model.manifest,
            manifest_digest=model_digest,
            file_hashes=base_generator.loaded_model.file_hashes,
            load_report={},
            base_state_digest=digest_for("forged-base-state"),
            adapter_digest=artifact.digest,
            adapter_attestation=attestation,
        )

        class PeftModel:
            pass

        with patch.dict(sys.modules, {"peft": SimpleNamespace(PeftModel=PeftModel)}):
            with self.assertRaises(VariationDependencyError):
                ModelCandidateGenerator(
                    loaded,
                    model_digest=model_digest,
                    adapter_digest=artifact.digest,
                    adapter_artifact=artifact,
                )

    def test_model_and_adapter_manifests_reject_directory_symlinks(self) -> None:
        model_root = self.root / "symlink-model"
        model_root.mkdir()
        (model_root / "weights.safetensors").write_bytes(b"model")
        model_manifest = build_local_manifest(model_root, license_name="test", license_source="test")
        (model_root / "model-manifest.json").write_text(
            canonical_json(model_manifest.to_dict()) + "\n", encoding="utf-8"
        )
        outside = self.root / "outside-model"
        outside.mkdir()
        (outside / "secret.safetensors").write_bytes(b"outside")
        (model_root / "linked-directory").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(VariationConfigurationError):
            PinnedModelLoader(model_root).preflight()

        artifact = self.make_adapter_artifact()
        adapter_outside = self.root / "outside-adapter"
        adapter_outside.mkdir()
        (adapter_outside / "unexpected.bin").write_bytes(b"outside")
        (artifact.root / "linked-directory").symlink_to(adapter_outside, target_is_directory=True)
        with self.assertRaises(VariationConfigurationError):
            artifact.verify()

    def test_nested_manifest_name_is_not_silently_excluded_from_exhaustive_adapter_tree(self) -> None:
        artifact = self.make_adapter_artifact()
        nested = artifact.root / "nested" / ADAPTER_MANIFEST_NAME
        nested.parent.mkdir()
        nested.write_text("unexpected nested manifest", encoding="utf-8")
        with self.assertRaises(VariationConfigurationError):
            artifact.verify()

    def test_sealed_adapter_application_is_explicit_and_local_only(self) -> None:
        artifact = self.make_adapter_artifact()
        calls = {}

        class AdapterConfig:
            def to_dict(self):
                return {"lora_alpha": 8, "peft_type": "LORA", "r": 4}

        class PeftModel:
            def __init__(self, model):
                self.model = model
                self.peft_config = {"default": AdapterConfig()}
                self.active_adapters = ["default"]

            def state_dict(self):
                return {"lora": b"applied"}

            @classmethod
            def from_pretrained(cls, model, root, **kwargs):
                calls["args"] = (model, root, kwargs)
                return cls(model)

        with patch.dict(sys.modules, {"peft": SimpleNamespace(PeftModel=PeftModel)}):
            applied = artifact.apply_to("base-model")
        self.assertIsInstance(applied, PeftModel)
        self.assertEqual(calls["args"][0], "base-model")
        self.assertTrue(calls["args"][2]["local_files_only"])
        self.assertFalse(calls["args"][2]["is_trainable"])

    def test_ordinary_failure_retrieval_excludes_promoted_success(self) -> None:
        runner, task, _repo, isolation = self.make_runner()
        report = runner.run(task, seed=0)
        retrieved = retrieval_policy("ORDINARY_FAILURE_SUMMARY").retrieve(
            self.ledger,
            campaign_id=report.campaign_id,
            arm_id="C",
            task_id=task.task_id,
            isolation=isolation,
        )
        self.assertEqual([record.recorded_disposition for record in retrieved.records], ["REJECTED"])

    def test_frozen_arm_policies_and_disjoint_namespaces(self) -> None:
        self.assertEqual(ARM_IDS, ("A", "B", "C", "D", "E", "F", "G", "H"))
        self.assertEqual(
            [(arm_policy(arm).model_mode, arm_policy(arm).retrieval_policy, arm_policy(arm).authority_enforced)
             for arm in ARM_IDS],
            [
                ("BASE", "SUCCESS_ONLY", False),
                ("BASE", "ORDINARY_FAILURE_SUMMARY", False),
                ("BASE", "CORRECTION_AWARE", False),
                ("BASE", "CORRECTION_AWARE", True),
                ("LORA", "CORRECTION_AWARE", True),
                ("LORA", "SUCCESS_ONLY", False),
                ("LORA", "ORDINARY_FAILURE_SUMMARY", False),
                ("LORA", "CORRECTION_AWARE", False),
            ],
        )
        isolation = ArmIsolation(self.root / "isolation", campaign_id="campaign")
        first = isolation.workspace("A", "run-a")
        second = isolation.workspace("C", "run-c")
        self.assertNotEqual(first.root, second.root)
        self.assertTrue(first.root.is_dir())
        self.assertTrue(second.root.is_dir())
        with self.assertRaises(VariationIsolationError):
            isolation.assert_candidate_id("A", "egv-candidate-campaign-C-not-a")
        with self.assertRaises(VariationIsolationError):
            isolation.assert_retrieval_arm("A", "C")

    def test_six_family_fixture_trajectories_are_bounded_and_promotable(self) -> None:
        model_digest = digest_for({"fixture_model": MODEL_REVISION, "families": True})
        policy_digest = digest_for("variation-six-family-policy")
        self.ledger = EvidenceLedger(
            self.root / "six-ledger.sqlite",
            blob_root=self.root / "six-blobs",
            clock=lambda: "2026-08-22T00:00:00Z",
        )
        campaign_id = "six-family-campaign"
        evaluator = FixtureEvaluationGateway(
            self.corpus,
            self.ledger,
            self.root / "six-evaluator",
            policy_digest=policy_digest,
            campaign_id=campaign_id,
        )
        isolation = ArmIsolation(self.root / "six-arms", campaign_id=campaign_id)
        candidates = {}
        loci = {}
        tasks = []
        for repo in self.corpus.hidden_repositories():
            task = VariationTask.from_microrepo(repo)
            tasks.append(task)
            candidates[task.task_id] = (dict(repo.source_files)["src/task.py"], repo.corrected_source)
            loci[task.task_id] = task.public_locus
        generator = DeterministicFixtureGenerator(candidates, public_locus=loci, model_digest=model_digest)
        runner = BoundedCandidateLoop(
            ledger=self.ledger,
            evaluator=evaluator,
            generator=generator,
            isolation=isolation,
            workspace_root=self.root / "six-state",
            campaign_id=campaign_id,
            source_commit="variation-six-family-test",
            model_revision=MODEL_REVISION,
            model_digest=model_digest,
            data_manifest_digest=self.corpus.manifest_digest(),
            policy_digest=policy_digest,
            arm_id="C",
            max_attempts=2,
            seed_set=(0,),
            fixture_mode=True,
        )
        reports = [runner.run(task, seed=0) for task in tasks]
        self.assertEqual({task.family_id for task in tasks}, {spec.family_id for spec in FAMILY_SPECS})
        self.assertTrue(all(report.promoted for report in reports))
        self.assertTrue(all(len(report.attempts) == 2 for report in reports))
        self.assertTrue(all(report.attempts[0].diagnostic_enum != "PASS" for report in reports))
        self.assertTrue(all(report.attempts[1].disposition == "PROMOTED" for report in reports))
        self.assertTrue(all(report.authority_enforced is False for report in reports))
        self.assertTrue(self.ledger.verify_integrity()["event_count"] > 0)

    def test_failure_retrieval_promotion_receipts_and_private_artifact_binding(self) -> None:
        runner, task, _repo, isolation = self.make_runner()
        report = runner.run(task, seed=0)
        self.assertTrue(report.promoted)
        self.assertEqual(len(report.attempts), 2)
        self.assertEqual(report.attempts[0].diagnostic_enum, "WRONG_OUTPUT")
        self.assertEqual(report.attempts[0].disposition, "REJECTED")
        self.assertEqual(report.attempts[1].disposition, "PROMOTED")
        self.assertTrue(report.attempts[1].evidence_ids)
        self.assertEqual(self.ledger.candidate_disposition(report.attempts[0].candidate_id), "REJECTED")
        self.assertEqual(self.ledger.candidate_disposition(report.attempts[1].candidate_id), "PROMOTED")
        self.assertEqual(len(self.ledger.receipts()), 6)
        workspace = isolation.workspace("C", report.run_id)
        candidate_files = tuple((workspace.candidates / "artifacts").rglob("*"))
        self.assertTrue(any(path.is_file() for path in candidate_files))
        self.assertTrue(all(path.stat().st_mode & 0o222 == 0 for path in candidate_files if path.is_file()))
        self.assertTrue(all("corpus-seed" not in path.name for path in candidate_files))
        policy_records = {
            policy: retrieval_policy(policy).retrieve(
                self.ledger,
                campaign_id=report.campaign_id,
                arm_id="C",
                task_id=task.task_id,
                isolation=isolation,
            )
            for policy in ("SUCCESS_ONLY", "ORDINARY_FAILURE_SUMMARY", "CORRECTION_AWARE")
        }
        self.assertEqual(len(policy_records["SUCCESS_ONLY"].records), 1)
        self.assertEqual(len(policy_records["ORDINARY_FAILURE_SUMMARY"].records), 1)
        self.assertEqual(policy_records["ORDINARY_FAILURE_SUMMARY"].records[0].recorded_disposition, "REJECTED")
        self.assertEqual(len(policy_records["CORRECTION_AWARE"].records), 2)

    def test_resume_reconstructs_durable_attempts_without_duplicate_writes(self) -> None:
        runner, task, _repo, isolation = self.make_runner()
        first = runner.run(task, seed=0)
        workspace = isolation.workspace("C", first.run_id)
        latest = CheckpointStore(workspace.checkpoints).latest(run_id=first.run_id)
        self.assertIsNotNone(latest)
        checkpoint_path, checkpoint = latest  # type: ignore[misc]
        before = self.ledger.ledger_head_hash()
        resumed = runner.run(task, seed=0, resume_from=checkpoint_path)
        self.assertEqual(resumed.terminal_status, "PROMOTED")
        self.assertEqual([item.candidate_id for item in resumed.attempts], [item.candidate_id for item in first.attempts])
        self.assertEqual(resumed.ledger_head_hash, before)
        self.assertEqual(checkpoint.status, "PROMOTED")
        self.assertEqual(len(self.ledger.receipts()), 6)

    def test_running_checkpoint_resumes_after_generator_interruption(self) -> None:
        class InterruptingGenerator(DeterministicFixtureGenerator):
            interrupt = True

            def propose(self, context):
                if self.interrupt and context.attempt_index == 2:
                    raise RuntimeError("simulated trainer interruption")
                return super().propose(context)

        holder = {}

        def generator_factory(task, repo, model_digest):
            public_source = dict(repo.source_files)["src/task.py"]
            generator = InterruptingGenerator(
                {task.task_id: (public_source, repo.corrected_source)},
                public_locus={task.task_id: task.public_locus},
                model_digest=model_digest,
            )
            holder["generator"] = generator
            return generator

        runner, task, repo, isolation = self.make_runner(generator_factory=generator_factory)
        with self.assertRaises(RuntimeError):
            runner.run(task, seed=0)
        run_id = runner._run_id(task.task_id, 0)
        workspace = isolation.workspace("C", run_id)
        latest = CheckpointStore(workspace.checkpoints).latest(run_id=run_id)
        self.assertIsNotNone(latest)
        checkpoint_path, checkpoint = latest  # type: ignore[misc]
        self.assertEqual(checkpoint.status, "RUNNING")
        holder["generator"].interrupt = False
        resumed = runner.run(task, seed=0, resume_from=checkpoint_path)
        self.assertTrue(resumed.promoted)
        self.assertEqual(len(resumed.attempts), 2)

    def test_infrastructure_loss_is_incident_bound_and_stops_the_loop(self) -> None:
        incident = digest_for("variation-infrastructure-incident")

        class InfrastructureGateway:
            enforceable = False
            evaluator_revision = "variation-infrastructure-test"
            evaluator_digest = digest_for(evaluator_revision)

            def __init__(self, hidden_runner):
                self.hidden_runner = hidden_runner

            def evaluate(self, *, candidate_id, task_id, source, **_kwargs):
                record = self.hidden_runner.public_record(task_id)
                return EvaluationResult(
                    candidate_id=candidate_id,
                    task_id=task_id,
                    candidate_artifact_digest=digest_bytes(source),
                    diagnostic_enum="INTERNAL_ERROR",
                    resource_bucket="UNDER_25",
                    disposition="ABSTAINED",
                    infrastructure_loss=True,
                    receipt_ids=(),
                    output_digest=digest_bytes(b""),
                    infrastructure_incident_id=incident,
                    failure_family_root=failure_family_root(
                        record["family_id"],
                        "INTERNAL_ERROR",
                        record["public_locus"],
                        record["public_rule_id"],
                        infrastructure_incident_id=incident,
                    ),
                )

        def evaluator_factory(corpus, ledger, policy_digest, campaign_id):
            fixture = FixtureEvaluationGateway(
                corpus,
                ledger,
                self.root / "infrastructure-fixture-evaluator",
                policy_digest=policy_digest,
                campaign_id=campaign_id,
            )
            return InfrastructureGateway(fixture.hidden_runner)

        runner, task, _repo, isolation = self.make_runner(evaluator_factory=evaluator_factory)
        report = runner.run(task, seed=0)
        self.assertEqual(report.terminal_status, "FAILED")
        self.assertEqual(len(report.attempts), 1)
        self.assertEqual(report.attempts[0].diagnostic_enum, "INTERNAL_ERROR")
        latest = CheckpointStore(isolation.workspace("C", report.run_id).checkpoints).latest(run_id=report.run_id)
        self.assertIsNotNone(latest)
        self.assertEqual(latest[1].status, "FAILED")  # type: ignore[index]

    def test_checkpoint_store_rejects_path_outside_isolation(self) -> None:
        store = CheckpointStore(self.root / "checkpoints")
        outside = self.root / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        with self.assertRaises(VariationCheckpointError):
            store.load(outside)

    def test_checkpoint_store_rejects_same_name_with_tampered_payload(self) -> None:
        runner, task, _repo, isolation = self.make_runner()
        report = runner.run(task, seed=0)
        workspace = isolation.workspace("C", report.run_id)
        latest = CheckpointStore(workspace.checkpoints).latest(run_id=report.run_id)
        self.assertIsNotNone(latest)
        _path, checkpoint = latest  # type: ignore[misc]
        tampered_root = self.root / "tampered-checkpoints"
        tampered_store = CheckpointStore(tampered_root)
        tampered = checkpoint.to_dict()
        tampered["status"] = "FAILED"
        tampered_path = tampered_root / "checkpoint-{}.json".format(checkpoint.digest)
        tampered_path.write_text(canonical_json(tampered) + "\n", encoding="utf-8")
        with self.assertRaises(VariationCheckpointError):
            tampered_store.load(tampered_path)

    def test_budget_and_lora_requirements_fail_closed(self) -> None:
        with self.assertRaises(VariationBudgetError):
            self.make_runner(max_attempts=13)
        with self.assertRaises(VariationDependencyError):
            self.make_runner(arm_id="E")

    def test_test_only_generator_cannot_enter_non_fixture_path(self) -> None:
        with self.assertRaises(VariationDependencyError):
            self.make_runner(fixture_mode=False)

    def test_candidate_contract_rejects_forged_locus_evidence_and_size(self) -> None:
        context = CandidateContext(
            "campaign",
            "run",
            0,
            "C",
            "task",
            "PURE_FUNCTION",
            "src/task.py:main",
            "rule-public",
            1,
            None,
            ({"event_id": "evt-1"},),
            digest_for("retrieval"),
            digest_for("model"),
            None,
            digest_for("prompt"),
        )
        valid = CandidateProposal(b"def main(value):\n    return value\n", context.public_locus, "READ_ONLY", ("evt-1",), digest_bytes(b"def main(value):\n    return value\n"), {})
        valid.validate(context, source_limit=CANDIDATE_SOURCE_LIMIT)
        with self.assertRaises(VariationConfigurationError):
            CandidateProposal(valid.source, "wrong", valid.requested_authority, valid.evidence_ids, valid.mutation_digest, {}).validate(
                context, source_limit=CANDIDATE_SOURCE_LIMIT
            )
        with self.assertRaises(VariationConfigurationError):
            CandidateProposal(valid.source, valid.declared_locus, valid.requested_authority, ("evt-2",), valid.mutation_digest, {}).validate(
                context, source_limit=CANDIDATE_SOURCE_LIMIT
            )
        with self.assertRaises(VariationConfigurationError):
            CandidateProposal(b"x" * (CANDIDATE_SOURCE_LIMIT + 1), context.public_locus, "READ_ONLY", (), digest_for("large"), {}).validate(
                context, source_limit=CANDIDATE_SOURCE_LIMIT
            )

    def test_model_prompt_and_source_only_repair_are_closed(self) -> None:
        context = CandidateContext(
            "campaign", "run", 0, "B", "task", "PURE_FUNCTION", "src/task.py:solve", "rule",
            1, None, ({"event_id": "evt-2"}, {"event_id": "evt-1"}), digest_for("retrieval"),
            digest_for("model"), None, digest_for("prompt"),
        )
        prompt = render_candidate_prompt(context)
        self.assertIn('declared_locus must equal "src/task.py:solve"', prompt)
        self.assertIn('requested_authority must equal "EXECUTE_CANDIDATE"', prompt)
        self.assertIn('["evt-1", "evt-2"]', prompt)
        self.assertIn("Do not use Markdown fences", prompt)

        initial_source = "def solve(value):\n    return value"
        source_context = replace(
            context,
            task_statement="Repair the bounded function without changing its public API.",
            initial_source=initial_source,
            initial_source_digest=digest_bytes(initial_source.encode("utf-8")),
        )
        source_prompt = render_candidate_prompt(
            source_context,
            response_contract="source-only-v1",
        )
        self.assertIn(initial_source, source_prompt)
        self.assertIn(source_context.task_statement, source_prompt)
        self.assertIn("trusted host supplies", source_prompt)
        with self.assertRaises(VariationConfigurationError):
            render_candidate_prompt(context, response_contract="source-only-v1")

        source = "def solve(value):\n    return value + 1"
        repaired = ModelCandidateGenerator._parse_response(source, context)
        self.assertEqual(repaired.source, source.encode("utf-8"))
        self.assertEqual(repaired.declared_locus, context.public_locus)
        self.assertEqual(repaired.requested_authority, "EXECUTE_CANDIDATE")
        self.assertEqual(repaired.evidence_ids, ())
        self.assertEqual(repaired.metadata["response_contract"], "source-only-repair-v1")
        self.assertEqual(repaired.metadata["raw_response_digest"], digest_bytes(repaired.source))
        repaired.validate(context, source_limit=CANDIDATE_SOURCE_LIMIT)

        source_only = ModelCandidateGenerator._parse_response(
            source,
            source_context,
            response_contract="source-only-v1",
        )
        self.assertEqual(source_only.source, source.encode("utf-8"))
        self.assertEqual(source_only.declared_locus, source_context.public_locus)
        self.assertEqual(source_only.requested_authority, "EXECUTE_CANDIDATE")
        self.assertEqual(source_only.evidence_ids, ("evt-1", "evt-2"))
        self.assertEqual(source_only.metadata["response_contract"], "source-only-v1")
        prefilled = ModelCandidateGenerator._parse_response(
            source,
            source_context,
            response_contract="source-only-prefill-v1",
        )
        self.assertEqual(prefilled.source, source.encode("utf-8"))
        self.assertEqual(prefilled.metadata["response_contract"], "source-only-prefill-v1")
        for raw in (
            "Here is the answer.",
            "```python\n" + source + "\n```",
            "value = 1",
            "def wrong_locus(value):\n    return value",
        ):
            with self.subTest(source_only_raw=raw), self.assertRaises(VariationDependencyError):
                ModelCandidateGenerator._parse_response(
                    raw,
                    source_context,
                    response_contract="source-only-v1",
                )

        rejected = (
            "Here is the answer.",
            "```python\ndef solve(value):\n    return value\n```",
            "{'source': 'not JSON'}",
            "value = 1",
            '{"source":"def solve(): pass"} trailing',
        )
        for raw in rejected:
            with self.subTest(raw=raw), self.assertRaises(VariationDependencyError):
                ModelCandidateGenerator._parse_response(raw, context)

        bad_metadata = canonical_json({
            "source": source, "declared_locus": context.public_locus,
            "requested_authority": "EXECUTE_CANDIDATE", "evidence_ids": [], "metadata": [],
        })
        with self.assertRaises(VariationDependencyError):
            ModelCandidateGenerator._parse_response(bad_metadata, context)
        wrong_authority = canonical_json({
            "source": source, "declared_locus": context.public_locus,
            "requested_authority": "READ_ONLY", "evidence_ids": [], "metadata": {},
        })
        with self.assertRaisesRegex(VariationDependencyError, "wrong authority"):
            ModelCandidateGenerator._parse_response(wrong_authority, context)

    def test_source_prefill_uses_continue_final_message_and_exact_chat_digest(self) -> None:
        base, model_digest = self.make_production_generator()

        class ChatTokenizer:
            chat_template = "frozen-test-chat-template"

            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return "<exact-chat>" + messages[-1]["content"]

        tokenizer = ChatTokenizer()
        loaded = LoadedPinnedModel(
            model=base.model,
            tokenizer=tokenizer,
            manifest=base.loaded_model.manifest,
            manifest_digest=model_digest,
            file_hashes=base.loaded_model.file_hashes,
            load_report={},
            base_state_digest=base.loaded_model.base_state_digest,
        )
        generator = ModelCandidateGenerator(
            loaded,
            model_digest=model_digest,
            response_contract="source-only-prefill-v1",
        )
        initial_source = "def solve(value):\n    return value"
        context = CandidateContext(
            "campaign", "run", 0, "B", "task", "PURE_FUNCTION", "src/task.py:solve", "rule",
            1, None, (), digest_for("retrieval"), model_digest, None, digest_for("placeholder"),
            "Repair the bounded function.", initial_source, digest_bytes(initial_source.encode("utf-8")),
            "source-only-prefill-v1", generator.response_contract_digest,
            generator.generation_profile_digest,
        )
        digest = generator.prompt_digest_for(context)
        self.assertEqual(len(digest), 64)
        self.assertEqual(tokenizer.messages[-1], {"role": "assistant", "content": "def solve(value):\n"})
        self.assertTrue(tokenizer.kwargs["continue_final_message"])
        self.assertFalse(tokenizer.kwargs["add_generation_prompt"])
        self.assertFalse(tokenizer.kwargs["tokenize"])
        with self.assertRaises(AttributeError):
            generator.prompt_registry = PromptRegistry()

    def test_source_prefill_propose_reassembles_and_seals_exact_generation(self) -> None:
        class Device:
            type = "cuda"

        device = Device()

        class Tensor:
            def __init__(self, values, tensor_device=None):
                self.values = list(values)
                self.device = tensor_device
                self.shape = (1, len(self.values))

            def to(self, *, device):
                return Tensor(self.values, device)

        class Model:
            def parameters(self):
                return iter((SimpleNamespace(device=device),))

            def generate(self, **kwargs):
                self.kwargs = kwargs
                input_values = kwargs["input_ids"].values
                return [input_values + [91, 92]]

        class Tokenizer:
            chat_template = "frozen-prefill-template-v1"

            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.chat_kwargs = kwargs
                return "<chat>" + messages[-1]["content"]

            def __call__(self, text, **kwargs):
                self.rendered = text
                self.tokenize_kwargs = kwargs
                return {"input_ids": Tensor([1, 2]), "attention_mask": Tensor([1, 1])}

            def decode(self, tokens, *, skip_special_tokens):
                self.decoded_tokens = list(tokens)
                self.skip_special_tokens = skip_special_tokens
                return "    return value + 1\n"

        base, model_digest = self.make_production_generator()
        model = Model()
        tokenizer = Tokenizer()
        loaded = LoadedPinnedModel(
            model=model,
            tokenizer=tokenizer,
            manifest=base.loaded_model.manifest,
            manifest_digest=model_digest,
            file_hashes=base.loaded_model.file_hashes,
            load_report={},
            base_state_digest=digest_for("base-state"),
        )
        generator = ModelCandidateGenerator(
            loaded,
            model_digest=model_digest,
            response_contract="source-only-prefill-v1",
        )
        initial_source = "def solve(value):\n    return value"
        context = CandidateContext(
            "campaign", "run", 0, "B", "task", "PURE_FUNCTION", "src/task.py:solve", "rule",
            1, None, ({"event_id": "evt-2"}, {"event_id": "evt-1"}), digest_for("retrieval"),
            model_digest, None, digest_for("placeholder"), "Repair the bounded function.",
            initial_source, digest_bytes(initial_source.encode("utf-8")),
            "source-only-prefill-v1", generator.response_contract_digest,
            generator.generation_profile_digest,
        )
        context = replace(context, prompt_digest=generator.prompt_digest_for(context))
        generation = generator.propose_with_evidence(context)
        proposal = generation.proposal
        expected = b"def solve(value):\n    return value + 1"
        self.assertEqual(generation.decoded_model_response, b"    return value + 1\n")
        self.assertEqual(
            generation.decoded_model_response_digest,
            digest_bytes(b"    return value + 1\n"),
        )
        self.assertEqual(generation.contract_response, expected + b"\n")
        self.assertEqual(generation.contract_response_digest, digest_bytes(expected + b"\n"))
        self.assertTrue(generation.rendered_prompt)
        self.assertEqual(digest_bytes(generation.rendered_prompt), context.prompt_digest)
        self.assertEqual(generation.rendered_prompt_digest, context.prompt_digest)
        self.assertEqual(generation.response_contract, "source-only-prefill-v1")
        self.assertEqual(proposal.source, expected)
        self.assertEqual(proposal.evidence_ids, ("evt-1", "evt-2"))
        self.assertEqual(proposal.metadata["contract_response_digest"], digest_bytes(expected + b"\n"))
        self.assertEqual(proposal.metadata["normalized_source_digest"], digest_bytes(expected))
        self.assertEqual(tokenizer.decoded_tokens, [91, 92])
        self.assertTrue(tokenizer.chat_kwargs["continue_final_message"])
        self.assertFalse(tokenizer.chat_kwargs["add_generation_prompt"])
        self.assertEqual(generator.propose(context), proposal)
        unrelated = b"def solve(value):\n    return value - 1\n"
        with self.assertRaises(VariationDependencyError):
            replace(
                generation,
                contract_response=unrelated,
                contract_response_digest=digest_bytes(unrelated),
            ).validate(context)
        with self.assertRaises(VariationDependencyError):
            replace(
                generation,
                decoded_model_response=b"    return value - 1\n",
                decoded_model_response_digest=digest_bytes(b"    return value - 1\n"),
            ).validate(context)
        with self.assertRaises(VariationDependencyError):
            replace(generation, response_contract="source-only-v1").validate(context)
        different_source = b"def solve(value):\n    return value + 2"
        with self.assertRaises(VariationDependencyError):
            replace(
                generation,
                proposal=replace(
                    proposal,
                    source=different_source,
                    mutation_digest=digest_bytes(different_source),
                ),
            ).validate(context)
        with self.assertRaises(VariationDependencyError):
            replace(
                generation,
                proposal=replace(proposal, metadata={**proposal.metadata, "contract_response_digest": "0" * 64}),
            ).validate(context)
        generator.validate_production_integrity()
        original_evidence_validate = CandidateGenerationEvidence.validate
        try:
            CandidateGenerationEvidence.validate = lambda self, context: None
            with self.assertRaisesRegex(VariationDependencyError, "evidence-generation method was altered"):
                generator.validate_production_integrity()
        finally:
            CandidateGenerationEvidence.validate = original_evidence_validate
        tokenizer.chat_template = "mutated-template"
        with self.assertRaisesRegex(VariationDependencyError, "chat-template bytes changed"):
            generator.validate_production_integrity()

    def test_source_contract_invalid_first_attempt_advances_to_valid_second_attempt(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, repo, generator, tokenizer, model, private_store = self.make_source_contract_runner(
            ["not valid Python", valid],
            max_attempts=2,
        )
        self.assertEqual(repo.corrected_source.decode("utf-8"), valid)
        report = runner.run(task, seed=0)
        self.assertEqual(report.terminal_status, "PROMOTED")
        self.assertEqual([item.attempt_index for item in report.attempts], [2])
        failures = private_store.source_contract_failures(
            run_id=report.run_id,
            task_id=task.task_id,
            arm_id="B",
        )
        self.assertEqual([item["attempt_index"] for item in failures], [1])
        self.assertEqual(tokenizer.decode_calls, 2)
        self.assertEqual(model.generate_calls, 2)
        row = self.ledger.connection.execute(
            "SELECT candidate_json,prompt_hash FROM candidates WHERE candidate_id=?",
            (report.attempts[0].candidate_id,),
        ).fetchone()
        metadata = json.loads(row["candidate_json"])["metadata"]
        self.assertEqual(metadata["generation_profile_digest"], generator.generation_profile_digest)
        private_attempt = private_store.load_attempts([report.attempts[0].candidate_id])[0]
        self.assertEqual(digest_bytes(private_attempt.rendered_prompt), row["prompt_hash"])

    def test_source_contract_resume_after_persisted_invalid_response_starts_at_second_attempt(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, private_store = self.make_source_contract_runner(
            ["not valid Python", valid],
            max_attempts=2,
        )
        original = PrivateTrajectoryStore.record_generation_failure
        crashed = {"value": False}

        def persist_then_crash(store, **kwargs):
            result = original(store, **kwargs)
            if store is private_store and not crashed["value"]:
                crashed["value"] = True
                raise RuntimeError("injected crash after immutable failure persistence")
            return result

        with patch.object(PrivateTrajectoryStore, "record_generation_failure", new=persist_then_crash):
            with self.assertRaisesRegex(RuntimeError, "injected crash"):
                runner.run(task, seed=0)
        self.assertEqual(
            self.ledger.connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0],
            0,
        )
        report = runner.run(task, seed=0)
        self.assertEqual(report.terminal_status, "PROMOTED")
        self.assertEqual([item.attempt_index for item in report.attempts], [2])
        self.assertEqual(tokenizer.decode_calls, 2)
        self.assertEqual(model.generate_calls, 2)

    def test_source_contract_resume_after_generation_evidence_does_not_regenerate(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )
        self.assertFalse(private_store.legacy_orphan_manifests.exists())
        original = PrivateTrajectoryStore.record_generation_success
        crashed = {"value": False}

        def persist_then_crash(store, **kwargs):
            result = original(store, **kwargs)
            if store is private_store and not crashed["value"]:
                crashed["value"] = True
                raise RuntimeError("crash after generation evidence")
            return result

        with patch.object(PrivateTrajectoryStore, "record_generation_success", new=persist_then_crash):
            with self.assertRaisesRegex(RuntimeError, "generation evidence"):
                runner.run(task, seed=0)
            report = runner.run(task, seed=0)
        self.assertTrue(report.promoted)
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), (1, 1))
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0], 1)
        self.assertEqual(len(report.attempts), 1)

    def test_source_contract_generation_start_quarantines_unpersisted_model_result(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, _private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )

        def drop_before_evidence(_store, **_kwargs):
            raise RuntimeError("drop before generation evidence")

        with patch.object(
            PrivateTrajectoryStore,
            "record_generation_success",
            new=drop_before_evidence,
        ):
            with self.assertRaisesRegex(RuntimeError, "before generation evidence"):
                runner.run(task, seed=0)
            calls = (model.generate_calls, tokenizer.decode_calls)
            with self.assertRaisesRegex(VariationCheckpointError, "starts, intents, and records"):
                runner.run(task, seed=0)
        self.assertEqual(calls, (1, 1))
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), calls)
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0], 0)

    def test_source_contract_generation_intent_recovers_after_first_blob_write(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )
        original = PrivateTrajectoryStore._put_artifact
        crashed = {"value": False}

        def first_blob_then_crash(store, data, **kwargs):
            result = original(store, data, **kwargs)
            if (
                store is private_store
                and kwargs.get("role") == "private-rendered-prompt"
                and not crashed["value"]
            ):
                crashed["value"] = True
                raise RuntimeError("crash after first generation blob")
            return result

        with patch.object(PrivateTrajectoryStore, "_put_artifact", new=first_blob_then_crash):
            with self.assertRaisesRegex(RuntimeError, "first generation blob"):
                runner.run(task, seed=0)
            report = runner.run(task, seed=0)
        self.assertTrue(report.promoted)
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), (1, 1))

    def test_crossed_pending_generation_bundle_fails_before_model_regeneration(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )
        original = PrivateTrajectoryStore.record_generation_success
        crashed = {"value": False}

        def evidence_then_crash(store, **kwargs):
            result = original(store, **kwargs)
            if store is private_store and not crashed["value"]:
                crashed["value"] = True
                raise RuntimeError("crash after generation bundle")
            return result

        with patch.object(PrivateTrajectoryStore, "record_generation_success", new=evidence_then_crash):
            with self.assertRaisesRegex(RuntimeError, "generation bundle"):
                runner.run(task, seed=0)
        candidate_id = next(path.stem for path in private_store.generation_starts.glob("*.json"))
        for root in (
            private_store.generation_starts,
            private_store.generation_intents,
            private_store.generation_records,
        ):
            path = root / (candidate_id + ".json")
            value = json.loads(path.read_text(encoding="utf-8"))
            value["context"]["run_id"] = "crossed-run"
            path.chmod(0o600)
            path.write_bytes(canonical_bytes(value))
        calls = (model.generate_calls, tokenizer.decode_calls)
        with self.assertRaisesRegex(VariationCheckpointError, "conflicts with durable content"):
            runner.run(task, seed=0)
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), calls)

    def test_source_contract_resume_after_candidate_artifact_does_not_regenerate(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, _private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )
        original = variation_loop.BoundedCandidateLoop._write_candidate_artifact
        crashed = {"value": False}

        def persist_then_crash(loop, workspace, source):
            result = original(loop, workspace, source)
            if loop is runner and not crashed["value"]:
                crashed["value"] = True
                raise RuntimeError("crash after candidate artifact")
            return result

        with patch.object(
            variation_loop.BoundedCandidateLoop,
            "_write_candidate_artifact",
            new=persist_then_crash,
        ):
            with self.assertRaisesRegex(RuntimeError, "candidate artifact"):
                runner.run(task, seed=0)
            report = runner.run(task, seed=0)
        self.assertTrue(report.promoted)
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), (1, 1))
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0], 1)

    def test_source_contract_resume_after_candidate_row_reconstructs_sidecar(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )
        original = EvidenceLedger.append_candidate
        crashed = {"value": False}

        def persist_then_crash(ledger, candidate_id, **kwargs):
            result = original(ledger, candidate_id, **kwargs)
            if ledger is self.ledger and not crashed["value"]:
                crashed["value"] = True
                raise RuntimeError("crash after candidate row")
            return result

        with patch.object(EvidenceLedger, "append_candidate", new=persist_then_crash):
            with self.assertRaisesRegex(RuntimeError, "candidate row"):
                runner.run(task, seed=0)
            candidate_id = str(
                self.ledger.connection.execute("SELECT candidate_id FROM candidates").fetchone()[0]
            )
            self.assertFalse((private_store.records / (candidate_id + ".json")).exists())
            report = runner.run(task, seed=0)
        self.assertTrue(report.promoted)
        self.assertTrue((private_store.records / (candidate_id + ".json")).is_file())
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), (1, 1))

    def test_source_contract_resume_after_sidecar_reconciles_exact_dependencies(self) -> None:
        repo = self.corpus.split("train")[0]
        runner, task, _repo, _generator, tokenizer, model, _private_store = self.make_source_contract_runner(
            [dict(repo.source_files)["src/task.py"].decode("utf-8"), repo.corrected_source.decode("utf-8")],
            max_attempts=2,
        )
        original = EvidenceLedger.append_dependency
        crashed = {"value": False}

        def persist_then_crash(ledger, parent_id, child_id, **kwargs):
            result = original(ledger, parent_id, child_id, **kwargs)
            row = ledger.connection.execute(
                "SELECT candidate_json FROM candidates WHERE candidate_id=?", (child_id,)
            ).fetchone()
            attempt_index = json.loads(row["candidate_json"])["metadata"]["attempt_index"] if row else 0
            if ledger is self.ledger and attempt_index == 2 and not crashed["value"]:
                crashed["value"] = True
                raise RuntimeError("crash after candidate dependency")
            return result

        with patch.object(EvidenceLedger, "append_dependency", new=persist_then_crash):
            with self.assertRaisesRegex(RuntimeError, "candidate dependency"):
                runner.run(task, seed=0)
            latest = CheckpointStore(
                runner.isolation.workspace("B", runner._run_id(task.task_id, 0)).checkpoints
            ).latest(run_id=runner._run_id(task.task_id, 0))
            self.assertIsNotNone(latest)
            report = runner.run(task, seed=0, resume_from=latest[0])  # type: ignore[index]
        self.assertTrue(report.promoted)
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), (2, 2))
        second = report.attempts[-1]
        edges = self.ledger.connection.execute(
            "SELECT parent_id,edge_type FROM dependencies WHERE child_id=?", (second.candidate_id,)
        ).fetchall()
        self.assertEqual(
            {(str(row["parent_id"]), str(row["edge_type"])) for row in edges},
            {(evidence_id, "EVIDENCE_USED") for evidence_id in second.evidence_ids},
        )

    def test_source_contract_resume_replays_cached_evaluator_after_receipts(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, _private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )
        gateway_type = type(runner.evaluator)
        original = gateway_type.evaluate
        state = {"calls": 0, "result": None}

        def effect_then_drop_response(gateway, **kwargs):
            state["calls"] += 1
            if state["result"] is None:
                state["result"] = original(gateway, **kwargs)
                raise VariationDependencyError("simulated transport loss after evaluator receipts")
            return state["result"]

        with patch.object(gateway_type, "evaluate", new=effect_then_drop_response):
            with self.assertRaisesRegex(VariationDependencyError, "transport loss"):
                runner.run(task, seed=0)
            self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM receipts").fetchone()[0], 3)
            self.assertEqual(
                sum(event["event_type"] == "VARIATION_ATTEMPT" for event in self.ledger.current_valid_events()),
                0,
            )
            report = runner.run(task, seed=0)
        self.assertTrue(report.promoted)
        self.assertEqual(state["calls"], 2)
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), (1, 1))
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM receipts").fetchone()[0], 3)
        self.assertEqual(
            sum(event["event_type"] == "VARIATION_ATTEMPT" for event in self.ledger.current_valid_events()),
            1,
        )
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0], 1)

    def test_source_contract_remote_gateway_replays_signed_cached_result_end_to_end(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        _fixture, task, repo, generator, tokenizer, model, private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )
        signer = ReceiptSigner(b"Z" * 32)
        public_key = self.root / "remote-replay.pub"
        public_key.write_bytes(signer.public_key_raw)
        command = self.root / "remote-replay.py"
        command.write_text("# durable remote replay fixture\n", encoding="utf-8")
        task_record = repo.public_manifest_record()
        task = VariationTask.from_public_record(task_record)
        campaign_id = "source-contract-retry-campaign"
        policy_digest = AuthorityPolicy.candidate_execution().digest
        docker = DockerSandboxConfig()
        unsigned_manifest = {
            "schema_version": REMOTE_VARIATION_SERVICE_SCHEMA,
            "campaign_id": campaign_id,
            "model_digest": generator.model_digest,
            "protocol_digest": variation_loop.VARIATION_PROTOCOL_DIGEST,
            "policy_digest": policy_digest,
            "data_manifest_digest": self.corpus.manifest_digest(),
            "task_manifest_digest": digest_for([task_record]),
            "task_bindings": [task_record],
            "evaluator_revision": "durable-remote-replay-v1",
            "evaluator_digest": digest_for("durable-remote-replay-v1"),
            "docker_image_digest": docker.pinned_image_id,
            "docker_config_digest": digest_for(dict(docker.__dict__)),
            "authority_policy_digest": policy_digest,
            "command_digest": hashlib.sha256(command.read_bytes()).hexdigest(),
            "evaluator_key_id": signer.key_id,
            "evaluator_public_key_digest": hashlib.sha256(public_key.read_bytes()).hexdigest(),
        }
        manifest_value = {
            **unsigned_manifest,
            "service_manifest_digest": digest_for(unsigned_manifest),
        }
        manifest = self.root / "remote-replay-service.json"
        manifest.write_text(canonical_json(manifest_value) + "\n", encoding="utf-8")
        gateway = RemoteControllerEvaluationGateway(
            ledger=self.ledger,
            manifest_path=manifest,
            public_key_path=public_key,
            command=command,
        )
        runner = BoundedCandidateLoop(
            ledger=self.ledger,
            evaluator=gateway,
            generator=generator,
            isolation=ArmIsolation(self.root / "remote-replay-arm", campaign_id=campaign_id),
            workspace_root=self.root / "remote-replay-state",
            campaign_id=campaign_id,
            source_commit="remote-replay-source",
            model_revision=MODEL_REVISION,
            model_digest=generator.model_digest,
            data_manifest_digest=self.corpus.manifest_digest(),
            policy_digest=policy_digest,
            arm_id="B",
            max_attempts=1,
            seed_set=(0,),
            private_store=private_store,
            initial_source=dict(repo.source_files)["src/task.py"],
            response_contract_digest=generator.response_contract_digest,
            generation_profile_digest=generator.generation_profile_digest,
        )
        remote_cache = {}
        counters = {"invocations": 0, "effects": 0, "dropped": False}

        def durable_remote_command(_invocation, request_text):
            counters["invocations"] += 1
            request = json.loads(request_text)
            cached = remote_cache.get(request["operation_digest"])
            if cached is None:
                counters["effects"] += 1
                common = {
                    "campaign_id": request["campaign_id"],
                    "run_id": request["run_id"],
                    "task_id": request["task_id"],
                    "candidate_id": request["candidate_id"],
                    "candidate_artifact_digest": request["candidate_artifact_digest"],
                    "protocol_digest": request["protocol_digest"],
                    "policy_digest": request["policy_digest"],
                    "arm_policy_digest": request["arm_policy_digest"],
                    "evaluator_digest": request["service_manifest_digest"],
                    "task_family": request["public_task_binding"]["family_id"],
                    "normalized_public_locus": request["public_task_binding"]["public_locus"],
                    "public_rule_id": request["public_task_binding"]["public_rule_id"],
                }
                sequence = request["receipt_sequence_start"]
                previous = request["previous_receipt_hash"]
                receipts = []
                output_digest = digest_bytes(b"ok")
                for receipt_type, fields in (
                    ("AUTHORITY", {
                        "request_id": "request-authority-" + request["candidate_id"],
                        "decision": "ALLOW",
                    }),
                    ("VERDICT", {
                        "request_id": "request-verdict-" + request["candidate_id"],
                        "decision": "PASS",
                        "diagnostic_enum": "PASS",
                        "resource_bucket": "UNDER_25",
                        "exit_status_class": "SUCCESS",
                        "input_digest": digest_for("private-input"),
                        "output_digest": output_digest,
                    }),
                    ("EFFECT", {
                        "request_id": "request-effect-" + request["candidate_id"],
                        "decision": "ALLOW",
                        "diagnostic_enum": "PASS",
                        "normalized_action_hash": digest_for({
                            "action": "execute_candidate", "locus": request["declared_locus"]
                        }),
                        "sandbox_id": "sealed-sandbox",
                        "started_at": "2026-08-24T00:00:00Z",
                        "finished_at": "2026-08-24T00:00:01Z",
                        "exit_status_class": "SUCCESS",
                        "output_digest": output_digest,
                        "environment_diff_digest": digest_for({}),
                    }),
                ):
                    receipt = signer.sign_receipt(
                        {**common, "receipt_type": receipt_type, **fields},
                        sequence=sequence,
                        previous_receipt_hash=previous,
                        idempotency_key=content_id(
                            "remote-replay",
                            {"operation": request["operation_digest"], "type": receipt_type},
                        ),
                    )
                    receipts.append(receipt)
                    previous = variation_loop.receipt_hash(receipt)
                    sequence += 1
                result = {
                    "candidate_id": request["candidate_id"],
                    "task_id": request["task_id"],
                    "candidate_artifact_digest": request["candidate_artifact_digest"],
                    "diagnostic_enum": "PASS",
                    "resource_bucket": "UNDER_25",
                    "disposition": "PROMOTED",
                    "infrastructure_loss": False,
                    "receipt_ids": [item["receipt_id"] for item in receipts],
                    "output_digest": output_digest,
                }
                unsigned = {
                    "schema_version": "egv-remote-variation-response-v1",
                    "operation_digest": request["operation_digest"],
                    "request_digest": request["request_digest"],
                    "service_manifest_digest": request["service_manifest_digest"],
                    "result": result,
                    "receipts": receipts,
                    "signing_key_id": signer.key_id,
                }
                cached = {**unsigned, "signature": signer.sign_bytes(canonical_bytes(unsigned))}
                remote_cache[request["operation_digest"]] = cached
            return 0, canonical_json(cached).encode("utf-8"), b""

        original_materialize = variation_loop.BoundedCandidateLoop._materialize_result

        def drop_first_verified_result(loop, **kwargs):
            if loop is runner and not counters["dropped"]:
                counters["dropped"] = True
                raise VariationDependencyError("drop verified remote result")
            return original_materialize(loop, **kwargs)

        with patch("egv.variation.remote._run_bounded_command", side_effect=durable_remote_command), patch.object(
            variation_loop.BoundedCandidateLoop,
            "_materialize_result",
            new=drop_first_verified_result,
        ):
            with self.assertRaisesRegex(VariationDependencyError, "drop verified"):
                runner.run(task, seed=0)
            self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM receipts").fetchone()[0], 3)
            candidate_id = str(
                self.ledger.connection.execute("SELECT candidate_id FROM candidates").fetchone()[0]
            )
            for root in (private_store.generation_starts, private_store.generation_intents):
                path = root / (candidate_id + ".json")
                path.chmod(0o600)
                path.unlink()
            self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0], 0)
            self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM effect_receipts").fetchone()[0], 0)
            self.assertEqual(
                sum(event["event_type"] == "VARIATION_ATTEMPT" for event in self.ledger.current_valid_events()),
                0,
            )
            calls_before_recovery = (model.generate_calls, tokenizer.decode_calls)
            with self.assertRaisesRegex(
                VariationCheckpointError,
                "exactly three distinct referenced artifacts and one distinct orphan artifact",
            ):
                runner.run(task, seed=0)
            self.assertEqual((model.generate_calls, tokenizer.decode_calls), calls_before_recovery)
            self.assertEqual(counters, {"invocations": 1, "effects": 1, "dropped": True})
            orphan_ref = private_store._put_artifact(
                b"commissioned-live-shape-orphan",
                media_type="application/octet-stream",
                role="commissioned-legacy-orphan",
            )
            report = runner.run(task, seed=0)
        self.assertTrue(report.promoted)
        self.assertEqual(counters, {"invocations": 2, "effects": 1, "dropped": True})
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), (1, 1))
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM receipts").fetchone()[0], 3)
        self.assertEqual(
            sum(event["event_type"] == "VARIATION_ATTEMPT" for event in self.ledger.current_valid_events()),
            1,
        )
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0], 1)
        manifest = json.loads(
            (private_store.legacy_orphan_manifests / (candidate_id + ".json")).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["orphan_artifact_digests"], [orphan_ref.digest])
        self.assertEqual(len(manifest["referenced_artifact_digests"]), 3)

    def test_source_contract_resume_revalidates_attempt_before_missing_checkpoint(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, _private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )
        gateway_type = type(runner.evaluator)
        original_evaluate = gateway_type.evaluate
        original_materialize = variation_loop.BoundedCandidateLoop._materialize_result
        state = {"calls": 0, "result": None, "crashed": False}

        def cached_evaluate(gateway, **kwargs):
            state["calls"] += 1
            if state["result"] is None:
                state["result"] = original_evaluate(gateway, **kwargs)
            return state["result"]

        def materialize_then_crash(loop, **kwargs):
            result = original_materialize(loop, **kwargs)
            if loop is runner and not state["crashed"]:
                state["crashed"] = True
                raise RuntimeError("crash after Variation attempt")
            return result

        with patch.object(gateway_type, "evaluate", new=cached_evaluate), patch.object(
            variation_loop.BoundedCandidateLoop,
            "_materialize_result",
            new=materialize_then_crash,
        ):
            with self.assertRaisesRegex(RuntimeError, "Variation attempt"):
                runner.run(task, seed=0)
            self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0], 0)
            report = runner.run(task, seed=0)
        self.assertTrue(report.promoted)
        self.assertEqual(state["calls"], 2)
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), (1, 1))
        self.assertEqual(
            sum(event["event_type"] == "VARIATION_ATTEMPT" for event in self.ledger.current_valid_events()),
            1,
        )
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0], 1)

    def test_legacy_generation_record_migrates_only_after_authenticated_cache_replay(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )
        gateway_type = type(runner.evaluator)
        original_evaluate = gateway_type.evaluate
        original_materialize = variation_loop.BoundedCandidateLoop._materialize_result
        state = {"calls": 0, "result": None, "crashed": False}

        def cached_evaluate(gateway, **kwargs):
            state["calls"] += 1
            if state["result"] is None:
                state["result"] = original_evaluate(gateway, **kwargs)
            return state["result"]

        def materialize_then_crash(loop, **kwargs):
            if loop is runner and not state["crashed"]:
                state["crashed"] = True
                raise RuntimeError("crash before legacy materialization")
            return original_materialize(loop, **kwargs)

        with patch.object(gateway_type, "evaluate", new=cached_evaluate), patch.object(
            variation_loop.BoundedCandidateLoop,
            "_materialize_result",
            new=materialize_then_crash,
        ):
            with self.assertRaisesRegex(RuntimeError, "legacy materialization"):
                runner.run(task, seed=0)
            candidate_id = str(
                self.ledger.connection.execute("SELECT candidate_id FROM candidates").fetchone()[0]
            )
            for root in (private_store.generation_starts, private_store.generation_intents):
                path = root / (candidate_id + ".json")
                path.chmod(0o600)
                path.unlink()
            self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM receipts").fetchone()[0], 3)
            self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0], 0)
            self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0], 0)
            self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM effect_receipts").fetchone()[0], 0)
            self.assertEqual(
                sum(event["event_type"] == "VARIATION_ATTEMPT" for event in self.ledger.current_valid_events()),
                0,
            )
            report = runner.run(task, seed=0)
        self.assertTrue(report.promoted)
        self.assertEqual(state["calls"], 2)
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), (1, 1))
        self.assertTrue((private_store.generation_starts / (candidate_id + ".json")).is_file())
        self.assertTrue((private_store.generation_intents / (candidate_id + ".json")).is_file())
        self.assertFalse(private_store.legacy_orphan_manifests.exists())
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM receipts").fetchone()[0], 3)
        self.assertEqual(
            sum(event["event_type"] == "VARIATION_ATTEMPT" for event in self.ledger.current_valid_events()),
            1,
        )
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0], 1)

    def test_legacy_generation_does_not_migrate_when_cached_replay_fails(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )
        gateway_type = type(runner.evaluator)
        original_evaluate = gateway_type.evaluate
        original_materialize = variation_loop.BoundedCandidateLoop._materialize_result
        state = {"result": None, "crashed": False}

        def cached_evaluate(gateway, **kwargs):
            if state["result"] is None:
                state["result"] = original_evaluate(gateway, **kwargs)
            return state["result"]

        def crash_before_materialize(loop, **kwargs):
            if loop is runner and not state["crashed"]:
                state["crashed"] = True
                raise RuntimeError("crash before legacy replay proof")
            return original_materialize(loop, **kwargs)

        with patch.object(gateway_type, "evaluate", new=cached_evaluate), patch.object(
            variation_loop.BoundedCandidateLoop,
            "_materialize_result",
            new=crash_before_materialize,
        ):
            with self.assertRaisesRegex(RuntimeError, "legacy replay proof"):
                runner.run(task, seed=0)
        candidate_id = str(
            self.ledger.connection.execute("SELECT candidate_id FROM candidates").fetchone()[0]
        )
        for root in (private_store.generation_starts, private_store.generation_intents):
            path = root / (candidate_id + ".json")
            path.chmod(0o600)
            path.unlink()
        calls = (model.generate_calls, tokenizer.decode_calls)

        def unavailable_replay(_gateway, **_kwargs):
            raise VariationDependencyError("cached evaluator replay unavailable")

        with patch.object(gateway_type, "evaluate", new=unavailable_replay):
            with self.assertRaisesRegex(VariationDependencyError, "replay unavailable"):
                runner.run(task, seed=0)
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), calls)
        self.assertFalse((private_store.generation_starts / (candidate_id + ".json")).exists())
        self.assertFalse((private_store.generation_intents / (candidate_id + ".json")).exists())
        self.assertFalse(private_store.legacy_orphan_manifests.exists())
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0], 0)
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM effect_receipts").fetchone()[0], 0)
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0], 0)

    def test_legacy_rejected_generation_migrates_without_repeating_attempt_one(self) -> None:
        repo = self.corpus.split("train")[0]
        rejected_source = dict(repo.source_files)["src/task.py"].decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, private_store = self.make_source_contract_runner(
            [rejected_source], max_attempts=1
        )
        gateway_type = type(runner.evaluator)
        original_evaluate = gateway_type.evaluate
        original_materialize = variation_loop.BoundedCandidateLoop._materialize_result
        state = {"calls": 0, "result": None, "crashed": False}

        def cached_evaluate(gateway, **kwargs):
            state["calls"] += 1
            if state["result"] is None:
                state["result"] = original_evaluate(gateway, **kwargs)
            return state["result"]

        def materialize_then_crash(loop, **kwargs):
            if loop is runner and not state["crashed"]:
                state["crashed"] = True
                raise RuntimeError("crash before rejected legacy materialization")
            return original_materialize(loop, **kwargs)

        with patch.object(gateway_type, "evaluate", new=cached_evaluate), patch.object(
            variation_loop.BoundedCandidateLoop,
            "_materialize_result",
            new=materialize_then_crash,
        ):
            with self.assertRaisesRegex(RuntimeError, "rejected legacy materialization"):
                runner.run(task, seed=0)
            self.assertEqual(
                sum(event["event_type"] == "VARIATION_ATTEMPT" for event in self.ledger.current_valid_events()),
                0,
            )
            candidate_id = str(
                self.ledger.connection.execute("SELECT candidate_id FROM candidates").fetchone()[0]
            )
            for root in (private_store.generation_starts, private_store.generation_intents):
                path = root / (candidate_id + ".json")
                path.chmod(0o600)
                path.unlink()
            report = runner.run(task, seed=0)
        self.assertEqual(report.terminal_status, "BUDGET_EXHAUSTED")
        self.assertEqual(state["calls"], 2)
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), (1, 1))
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM receipts").fetchone()[0], 3)
        self.assertEqual(
            sum(event["event_type"] == "VARIATION_ATTEMPT" for event in self.ledger.current_valid_events()),
            1,
        )
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0], 1)

    def test_legacy_rejected_generation_continues_to_attempt_two_in_same_process(self) -> None:
        repo = self.corpus.split("train")[0]
        rejected_source = dict(repo.source_files)["src/task.py"].decode("utf-8")
        promoted_source = repo.corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, private_store = self.make_source_contract_runner(
            [rejected_source, promoted_source], max_attempts=2
        )
        gateway_type = type(runner.evaluator)
        original_evaluate = gateway_type.evaluate
        original_materialize = variation_loop.BoundedCandidateLoop._materialize_result
        state = {"cache": {}, "effects": 0, "calls": 0, "crashed": False}

        def cached_evaluate(gateway, **kwargs):
            state["calls"] += 1
            candidate_id = kwargs["candidate_id"]
            if candidate_id not in state["cache"]:
                state["effects"] += 1
                state["cache"][candidate_id] = original_evaluate(gateway, **kwargs)
            return state["cache"][candidate_id]

        def crash_before_first_materialization(loop, **kwargs):
            if loop is runner and not state["crashed"]:
                state["crashed"] = True
                raise RuntimeError("crash before rejected attempt one materialization")
            return original_materialize(loop, **kwargs)

        with patch.object(gateway_type, "evaluate", new=cached_evaluate), patch.object(
            variation_loop.BoundedCandidateLoop,
            "_materialize_result",
            new=crash_before_first_materialization,
        ):
            with self.assertRaisesRegex(RuntimeError, "attempt one materialization"):
                runner.run(task, seed=0)
        first_candidate_id = str(
            self.ledger.connection.execute("SELECT candidate_id FROM candidates").fetchone()[0]
        )
        first_receipts = tuple(canonical_bytes(receipt) for receipt in self.ledger.receipts())
        for root in (private_store.generation_starts, private_store.generation_intents):
            path = root / (first_candidate_id + ".json")
            path.chmod(0o600)
            path.unlink()
        with patch.object(gateway_type, "evaluate", new=cached_evaluate):
            report = runner.run(task, seed=0)
        self.assertTrue(report.promoted)
        self.assertEqual(report.terminal_status, "PROMOTED")
        self.assertEqual([attempt.attempt_index for attempt in report.attempts], [1, 2])
        self.assertEqual([attempt.disposition for attempt in report.attempts], ["REJECTED", "PROMOTED"])
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), (2, 2))
        self.assertEqual((state["calls"], state["effects"]), (3, 2))
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM receipts").fetchone()[0], 6)
        self.assertEqual(
            tuple(canonical_bytes(receipt) for receipt in self.ledger.receipts()[:3]),
            first_receipts,
        )
        self.assertEqual(
            sum(event["event_type"] == "VARIATION_ATTEMPT" for event in self.ledger.current_valid_events()),
            2,
        )
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0], 2)

    def test_legacy_generation_migration_resumes_after_start_write(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )
        gateway_type = type(runner.evaluator)
        original_evaluate = gateway_type.evaluate
        original_materialize = variation_loop.BoundedCandidateLoop._materialize_result
        crashed = {"value": False, "result": None}

        def cached_evaluate(gateway, **kwargs):
            if crashed["result"] is None:
                crashed["result"] = original_evaluate(gateway, **kwargs)
            return crashed["result"]

        def materialize_then_crash(loop, **kwargs):
            if loop is runner and not crashed["value"]:
                crashed["value"] = True
                raise RuntimeError("crash before legacy migration")
            return original_materialize(loop, **kwargs)

        with patch.object(gateway_type, "evaluate", new=cached_evaluate), patch.object(
            variation_loop.BoundedCandidateLoop,
            "_materialize_result",
            new=materialize_then_crash,
        ):
            with self.assertRaisesRegex(RuntimeError, "legacy migration"):
                runner.run(task, seed=0)
        candidate_id = str(
            self.ledger.connection.execute("SELECT candidate_id FROM candidates").fetchone()[0]
        )
        for root in (private_store.generation_starts, private_store.generation_intents):
            path = root / (candidate_id + ".json")
            path.chmod(0o600)
            path.unlink()
        original_write_once = PrivateTrajectoryStore._write_once
        migration_drop = {"value": False}

        def drop_before_intent(store, path, value, **kwargs):
            if (
                store is private_store
                and path.parent == private_store.generation_intents
                and not migration_drop["value"]
            ):
                migration_drop["value"] = True
                raise RuntimeError("drop after migrated start")
            return original_write_once(store, path, value, **kwargs)

        calls = (model.generate_calls, tokenizer.decode_calls)
        with patch.object(gateway_type, "evaluate", new=cached_evaluate), patch.object(
            PrivateTrajectoryStore, "_write_once", new=drop_before_intent
        ):
            with self.assertRaisesRegex(RuntimeError, "migrated start"):
                runner.run(task, seed=0)
        self.assertTrue((private_store.generation_starts / (candidate_id + ".json")).is_file())
        self.assertFalse((private_store.generation_intents / (candidate_id + ".json")).exists())
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), calls)
        with patch.object(gateway_type, "evaluate", new=cached_evaluate):
            report = runner.run(task, seed=0)
        self.assertTrue(report.promoted)
        self.assertTrue((private_store.generation_intents / (candidate_id + ".json")).is_file())
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), calls)
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM receipts").fetchone()[0], 3)
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0], 1)

    def test_legacy_generation_migration_rejects_intent_without_start(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )
        original_materialize = variation_loop.BoundedCandidateLoop._materialize_result
        crashed = {"value": False}

        def materialize_then_crash(loop, **kwargs):
            if loop is runner and not crashed["value"]:
                crashed["value"] = True
                raise RuntimeError("crash before partial legacy set")
            return original_materialize(loop, **kwargs)

        with patch.object(
            variation_loop.BoundedCandidateLoop,
            "_materialize_result",
            new=materialize_then_crash,
        ):
            with self.assertRaisesRegex(RuntimeError, "partial legacy set"):
                runner.run(task, seed=0)
        candidate_id = str(
            self.ledger.connection.execute("SELECT candidate_id FROM candidates").fetchone()[0]
        )
        start_path = private_store.generation_starts / (candidate_id + ".json")
        start_path.chmod(0o600)
        start_path.unlink()
        calls = (model.generate_calls, tokenizer.decode_calls)
        with self.assertRaisesRegex(VariationCheckpointError, "impossible partial set"):
            runner.run(task, seed=0)
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), calls)
        self.assertFalse(start_path.exists())
        self.assertTrue((private_store.generation_intents / (candidate_id + ".json")).is_file())
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0], 0)

    def test_legacy_generation_migration_rejects_mismatched_existing_start(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )
        original_materialize = variation_loop.BoundedCandidateLoop._materialize_result
        crashed = {"value": False}

        def materialize_then_crash(loop, **kwargs):
            if loop is runner and not crashed["value"]:
                crashed["value"] = True
                raise RuntimeError("crash before mismatched migrated start")
            return original_materialize(loop, **kwargs)

        with patch.object(
            variation_loop.BoundedCandidateLoop,
            "_materialize_result",
            new=materialize_then_crash,
        ):
            with self.assertRaisesRegex(RuntimeError, "mismatched migrated start"):
                runner.run(task, seed=0)
        candidate_id = str(
            self.ledger.connection.execute("SELECT candidate_id FROM candidates").fetchone()[0]
        )
        intent_path = private_store.generation_intents / (candidate_id + ".json")
        intent_path.chmod(0o600)
        intent_path.unlink()
        start_path = private_store.generation_starts / (candidate_id + ".json")
        start = json.loads(start_path.read_text(encoding="utf-8"))
        start["context"]["task_id"] = "crossed-start-task"
        start_path.chmod(0o600)
        start_path.write_bytes(canonical_bytes(start))
        calls = (model.generate_calls, tokenizer.decode_calls)
        with self.assertRaisesRegex(VariationCheckpointError, "crossed its durable start"):
            runner.run(task, seed=0)
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), calls)
        self.assertFalse(intent_path.exists())
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0], 0)

    def test_legacy_generation_migration_rejects_missing_raw_cas_before_write(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )
        original_materialize = variation_loop.BoundedCandidateLoop._materialize_result
        crashed = {"value": False}

        def materialize_then_crash(loop, **kwargs):
            if loop is runner and not crashed["value"]:
                crashed["value"] = True
                raise RuntimeError("crash before missing legacy CAS")
            return original_materialize(loop, **kwargs)

        with patch.object(
            variation_loop.BoundedCandidateLoop,
            "_materialize_result",
            new=materialize_then_crash,
        ):
            with self.assertRaisesRegex(RuntimeError, "missing legacy CAS"):
                runner.run(task, seed=0)
        candidate_id = str(
            self.ledger.connection.execute("SELECT candidate_id FROM candidates").fetchone()[0]
        )
        for root in (private_store.generation_starts, private_store.generation_intents):
            path = root / (candidate_id + ".json")
            path.chmod(0o600)
            path.unlink()
        generation = json.loads(
            (private_store.generation_records / (candidate_id + ".json")).read_text(encoding="utf-8")
        )
        missing_digest = generation["contract_response_digest"]
        missing_path = (
            private_store.artifacts.root
            / "blobs"
            / "sha256"
            / missing_digest[:2]
            / missing_digest[2:4]
            / missing_digest
        )
        missing_path.chmod(0o600)
        missing_path.unlink()
        calls = (model.generate_calls, tokenizer.decode_calls)
        with self.assertRaises(VariationCheckpointError):
            runner.run(task, seed=0)
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), calls)
        self.assertFalse((private_store.generation_starts / (candidate_id + ".json")).exists())
        self.assertFalse((private_store.generation_intents / (candidate_id + ".json")).exists())
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0], 0)

    def test_legacy_generation_migration_rejects_extra_sidecar_before_write(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )
        original_materialize = variation_loop.BoundedCandidateLoop._materialize_result
        crashed = {"value": False}

        def crash_before_materialize(loop, **kwargs):
            if loop is runner and not crashed["value"]:
                crashed["value"] = True
                raise RuntimeError("crash before extra legacy sidecar")
            return original_materialize(loop, **kwargs)

        with patch.object(
            variation_loop.BoundedCandidateLoop,
            "_materialize_result",
            new=crash_before_materialize,
        ):
            with self.assertRaisesRegex(RuntimeError, "extra legacy sidecar"):
                runner.run(task, seed=0)
        candidate_id = str(
            self.ledger.connection.execute("SELECT candidate_id FROM candidates").fetchone()[0]
        )
        for root in (private_store.generation_starts, private_store.generation_intents):
            path = root / (candidate_id + ".json")
            path.chmod(0o600)
            path.unlink()
        source_sidecar = private_store.records / (candidate_id + ".json")
        (private_store.records / "unexpected-sidecar.json").write_bytes(source_sidecar.read_bytes())
        calls = (model.generate_calls, tokenizer.decode_calls)
        with self.assertRaisesRegex(VariationCheckpointError, "mixed with another generation state"):
            runner.run(task, seed=0)
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), calls)
        self.assertFalse((private_store.generation_starts / (candidate_id + ".json")).exists())
        self.assertFalse((private_store.generation_intents / (candidate_id + ".json")).exists())
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0], 0)
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM effect_receipts").fetchone()[0], 0)

    def test_legacy_generation_migration_manifests_single_orphan_cas(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )
        gateway_type = type(runner.evaluator)
        original_evaluate = gateway_type.evaluate
        original_materialize = variation_loop.BoundedCandidateLoop._materialize_result
        state = {"result": None, "crashed": False}

        def cached_evaluate(gateway, **kwargs):
            if state["result"] is None:
                state["result"] = original_evaluate(gateway, **kwargs)
            return state["result"]

        def crash_before_materialize(loop, **kwargs):
            if loop is runner and not state["crashed"]:
                state["crashed"] = True
                raise RuntimeError("crash before orphan legacy CAS")
            return original_materialize(loop, **kwargs)

        with patch.object(gateway_type, "evaluate", new=cached_evaluate), patch.object(
            variation_loop.BoundedCandidateLoop,
            "_materialize_result",
            new=crash_before_materialize,
        ):
            with self.assertRaisesRegex(RuntimeError, "orphan legacy CAS"):
                runner.run(task, seed=0)
        candidate_id = str(
            self.ledger.connection.execute("SELECT candidate_id FROM candidates").fetchone()[0]
        )
        for root in (private_store.generation_starts, private_store.generation_intents):
            path = root / (candidate_id + ".json")
            path.chmod(0o600)
            path.unlink()
        orphan_ref = private_store._put_artifact(
            b"orphan-private-cas",
            media_type="application/octet-stream",
            role="adversarial-orphan",
        )
        calls = (model.generate_calls, tokenizer.decode_calls)
        with patch.object(gateway_type, "evaluate", new=cached_evaluate):
            report = runner.run(task, seed=0)
        self.assertTrue(report.promoted)
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), calls)
        manifest_path = private_store.legacy_orphan_manifests / (candidate_id + ".json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["orphan_artifact_digests"], [orphan_ref.digest])
        self.assertTrue((private_store.generation_starts / (candidate_id + ".json")).is_file())
        self.assertTrue((private_store.generation_intents / (candidate_id + ".json")).is_file())
        self.assertEqual(
            len(private_store.successful_generations(run_id=report.run_id, task_id=task.task_id, arm_id="B")),
            1,
        )
        manifest_raw = manifest_path.read_bytes()
        manifest_path.chmod(0o600)
        manifest_path.unlink()
        with self.assertRaisesRegex(VariationCheckpointError, "lacks its orphan-artifact manifest"):
            private_store.successful_generations(run_id=report.run_id, task_id=task.task_id, arm_id="B")
        manifest_path.write_bytes(manifest_raw)
        tampered_manifest = json.loads(manifest_raw.decode("utf-8"))
        tampered_manifest["replay_proof_digest"] = "0" * 64
        manifest_path.write_bytes(canonical_bytes(tampered_manifest))
        with self.assertRaisesRegex(VariationCheckpointError, "manifest binding is invalid"):
            private_store.successful_generations(run_id=report.run_id, task_id=task.task_id, arm_id="B")

    def test_legacy_generation_migration_rejects_second_orphan_cas_before_write(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )
        original_materialize = variation_loop.BoundedCandidateLoop._materialize_result
        crashed = {"value": False}

        def crash_before_materialize(loop, **kwargs):
            if loop is runner and not crashed["value"]:
                crashed["value"] = True
                raise RuntimeError("crash before second orphan legacy CAS")
            return original_materialize(loop, **kwargs)

        with patch.object(
            variation_loop.BoundedCandidateLoop,
            "_materialize_result",
            new=crash_before_materialize,
        ):
            with self.assertRaisesRegex(RuntimeError, "second orphan legacy CAS"):
                runner.run(task, seed=0)
        candidate_id = str(
            self.ledger.connection.execute("SELECT candidate_id FROM candidates").fetchone()[0]
        )
        for root in (private_store.generation_starts, private_store.generation_intents):
            path = root / (candidate_id + ".json")
            path.chmod(0o600)
            path.unlink()
        for index in (1, 2):
            private_store._put_artifact(
                "orphan-private-cas-{}".format(index).encode("utf-8"),
                media_type="application/octet-stream",
                role="adversarial-orphan",
            )
        calls = (model.generate_calls, tokenizer.decode_calls)
        with self.assertRaisesRegex(VariationCheckpointError, "exceeds the preserved boundary"):
            runner.run(task, seed=0)
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), calls)
        self.assertFalse((private_store.generation_starts / (candidate_id + ".json")).exists())
        self.assertFalse((private_store.generation_intents / (candidate_id + ".json")).exists())
        self.assertFalse((private_store.legacy_orphan_manifests / (candidate_id + ".json")).exists())

    def test_legacy_generation_migration_resumes_after_manifest_before_start(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )
        gateway_type = type(runner.evaluator)
        original_evaluate = gateway_type.evaluate
        original_materialize = variation_loop.BoundedCandidateLoop._materialize_result
        state = {"result": None, "crashed": False}

        def cached_evaluate(gateway, **kwargs):
            if state["result"] is None:
                state["result"] = original_evaluate(gateway, **kwargs)
            return state["result"]

        def crash_before_materialize(loop, **kwargs):
            if loop is runner and not state["crashed"]:
                state["crashed"] = True
                raise RuntimeError("crash before orphan-manifest migration")
            return original_materialize(loop, **kwargs)

        with patch.object(gateway_type, "evaluate", new=cached_evaluate), patch.object(
            variation_loop.BoundedCandidateLoop,
            "_materialize_result",
            new=crash_before_materialize,
        ):
            with self.assertRaisesRegex(RuntimeError, "orphan-manifest migration"):
                runner.run(task, seed=0)
        candidate_id = str(
            self.ledger.connection.execute("SELECT candidate_id FROM candidates").fetchone()[0]
        )
        for root in (private_store.generation_starts, private_store.generation_intents):
            path = root / (candidate_id + ".json")
            path.chmod(0o600)
            path.unlink()
        private_store._put_artifact(
            b"preserved-orphan-before-start",
            media_type="application/octet-stream",
            role="adversarial-orphan",
        )
        original_record_start = PrivateTrajectoryStore.record_generation_start
        dropped = {"value": False}

        def drop_after_manifest(store, **kwargs):
            if store is private_store and not dropped["value"]:
                dropped["value"] = True
                manifest_path = store.legacy_orphan_manifests / (candidate_id + ".json")
                self.assertTrue(manifest_path.is_file())
                raise RuntimeError("drop after orphan manifest")
            return original_record_start(store, **kwargs)

        calls = (model.generate_calls, tokenizer.decode_calls)
        with patch.object(gateway_type, "evaluate", new=cached_evaluate), patch.object(
            PrivateTrajectoryStore,
            "record_generation_start",
            new=drop_after_manifest,
        ):
            with self.assertRaisesRegex(RuntimeError, "after orphan manifest"):
                runner.run(task, seed=0)
        manifest_path = private_store.legacy_orphan_manifests / (candidate_id + ".json")
        self.assertTrue(manifest_path.is_file())
        self.assertFalse((private_store.generation_starts / (candidate_id + ".json")).exists())
        self.assertFalse((private_store.generation_intents / (candidate_id + ".json")).exists())
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), calls)
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0], 0)
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM effect_receipts").fetchone()[0], 0)
        with patch.object(gateway_type, "evaluate", new=cached_evaluate):
            report = runner.run(task, seed=0)
        self.assertTrue(report.promoted)
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), calls)
        self.assertTrue((private_store.generation_starts / (candidate_id + ".json")).is_file())
        self.assertTrue((private_store.generation_intents / (candidate_id + ".json")).is_file())

    def test_legacy_generation_migration_rejects_crossed_sidecar_before_write(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )
        gateway_type = type(runner.evaluator)
        original_evaluate = gateway_type.evaluate
        original_materialize = variation_loop.BoundedCandidateLoop._materialize_result
        state = {"result": None, "crashed": False}

        def cached_evaluate(gateway, **kwargs):
            if state["result"] is None:
                state["result"] = original_evaluate(gateway, **kwargs)
            return state["result"]

        def materialize_then_crash(loop, **kwargs):
            if loop is runner and not state["crashed"]:
                state["crashed"] = True
                raise RuntimeError("crash before legacy tamper")
            return original_materialize(loop, **kwargs)

        with patch.object(gateway_type, "evaluate", new=cached_evaluate), patch.object(
            variation_loop.BoundedCandidateLoop,
            "_materialize_result",
            new=materialize_then_crash,
        ):
            with self.assertRaisesRegex(RuntimeError, "legacy tamper"):
                runner.run(task, seed=0)
            candidate_id = str(
                self.ledger.connection.execute("SELECT candidate_id FROM candidates").fetchone()[0]
            )
            for root in (private_store.generation_starts, private_store.generation_intents):
                path = root / (candidate_id + ".json")
                path.chmod(0o600)
                path.unlink()
            sidecar_path = private_store.records / (candidate_id + ".json")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["context"]["run_id"] = "crossed-legacy-run"
            sidecar_path.chmod(0o600)
            sidecar_path.write_bytes(canonical_bytes(sidecar))
            calls = (model.generate_calls, tokenizer.decode_calls)
            with self.assertRaisesRegex(VariationCheckpointError, "crossed its frozen context"):
                runner.run(task, seed=0)
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), calls)
        self.assertFalse((private_store.generation_starts / (candidate_id + ".json")).exists())
        self.assertFalse((private_store.generation_intents / (candidate_id + ".json")).exists())
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM receipts").fetchone()[0], 3)
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0], 0)

    def test_legacy_generation_migration_rejects_missing_receipt_before_write(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )
        original_materialize = variation_loop.BoundedCandidateLoop._materialize_result
        crashed = {"value": False}

        def materialize_then_crash(loop, **kwargs):
            if loop is runner and not crashed["value"]:
                crashed["value"] = True
                raise RuntimeError("crash before receipt deletion")
            return original_materialize(loop, **kwargs)

        with patch.object(
            variation_loop.BoundedCandidateLoop,
            "_materialize_result",
            new=materialize_then_crash,
        ):
            with self.assertRaisesRegex(RuntimeError, "receipt deletion"):
                runner.run(task, seed=0)
        candidate_id = str(
            self.ledger.connection.execute("SELECT candidate_id FROM candidates").fetchone()[0]
        )
        for root in (private_store.generation_starts, private_store.generation_intents):
            path = root / (candidate_id + ".json")
            path.chmod(0o600)
            path.unlink()
        self.ledger.connection.execute("DROP TRIGGER receipts_no_delete")
        self.ledger.connection.execute(
            "DELETE FROM receipts WHERE receipt_id=(SELECT receipt_id FROM receipts ORDER BY sequence LIMIT 1)"
        )
        calls = (model.generate_calls, tokenizer.decode_calls)
        with self.assertRaisesRegex(VariationCheckpointError, "pre-materialization boundary"):
            runner.run(task, seed=0)
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), calls)
        self.assertFalse((private_store.generation_starts / (candidate_id + ".json")).exists())
        self.assertFalse((private_store.generation_intents / (candidate_id + ".json")).exists())
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0], 0)

    def test_source_contract_resume_rejects_changed_result_with_unchanged_receipts_and_attempt(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, _private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )
        gateway_type = type(runner.evaluator)
        original_evaluate = gateway_type.evaluate
        original_materialize = variation_loop.BoundedCandidateLoop._materialize_result
        state = {"result": None, "crashed": False}

        def cached_evaluate(gateway, **kwargs):
            if state["result"] is None:
                state["result"] = original_evaluate(gateway, **kwargs)
            return state["result"]

        def materialize_then_crash(loop, **kwargs):
            result = original_materialize(loop, **kwargs)
            if loop is runner and not state["crashed"]:
                state["crashed"] = True
                raise RuntimeError("crash after durable attempt")
            return result

        with patch.object(gateway_type, "evaluate", new=cached_evaluate), patch.object(
            variation_loop.BoundedCandidateLoop,
            "_materialize_result",
            new=materialize_then_crash,
        ):
            with self.assertRaisesRegex(RuntimeError, "durable attempt"):
                runner.run(task, seed=0)
            state["result"] = replace(state["result"], resource_bucket="25_TO_50")
            with self.assertRaises(Exception):
                runner.run(task, seed=0)
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), (1, 1))
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM receipts").fetchone()[0], 3)
        self.assertEqual(
            sum(event["event_type"] == "VARIATION_ATTEMPT" for event in self.ledger.current_valid_events()),
            1,
        )
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0], 0)

    def test_source_contract_resume_finishes_checkpoint_file_before_ledger_projection_boundary(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, _private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )
        original = EvidenceLedger.add_checkpoint
        crashed = {"value": False}

        def checkpoint_projection_crash(ledger, checkpoint_id, **kwargs):
            if ledger is self.ledger and not crashed["value"]:
                crashed["value"] = True
                raise RuntimeError("crash before checkpoint projection")
            return original(ledger, checkpoint_id, **kwargs)

        with patch.object(EvidenceLedger, "add_checkpoint", new=checkpoint_projection_crash):
            with self.assertRaisesRegex(RuntimeError, "checkpoint projection"):
                runner.run(task, seed=0)
            workspace = runner.isolation.workspace("B", runner._run_id(task.task_id, 0))
            latest = CheckpointStore(workspace.checkpoints).latest(run_id=runner._run_id(task.task_id, 0))
            self.assertIsNotNone(latest)
            self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0], 0)
            report = runner.run(task, seed=0, resume_from=latest[0])  # type: ignore[index]
        self.assertTrue(report.promoted)
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), (1, 1))
        self.assertEqual(self.ledger.connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0], 1)

    def test_resume_rejects_hash_valid_crossed_variation_attempt_envelope(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, _tokenizer, _model, _private_store = self.make_source_contract_runner(
            [valid], max_attempts=2
        )
        report = runner.run(task, seed=0)
        attempt_event = next(
            event for event in self.ledger.current_valid_events()
            if event["event_type"] == "VARIATION_ATTEMPT"
        )
        payload = dict(attempt_event["payload"])
        payload["attempt_index"] = 2
        self.ledger.append_event(
            "VARIATION_ATTEMPT",
            payload,
            campaign_id="crossed-campaign",
            run_id=report.run_id,
            task_id=task.task_id,
            subject_id=payload["candidate_id"],
            source_class="GENERATOR",
            disposition=payload["disposition"],
            idempotency_key="variation-attempt:{}:2".format(payload["candidate_id"]),
        )
        with self.assertRaisesRegex(VariationCheckpointError, "native fields"):
            runner.run(task, seed=0, resume_from=Path(report.checkpoint_path))

    def test_budget_terminal_resume_rejects_crossed_authentic_receipt_suffix(self) -> None:
        repo = self.corpus.split("train")[0]
        invalid = dict(repo.source_files)["src/task.py"].decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, _private_store = self.make_source_contract_runner(
            [invalid, invalid], max_attempts=2
        )
        original_checkpoint = variation_loop.BoundedCandidateLoop._checkpoint

        def crash_before_budget_checkpoint(loop, **kwargs):
            if loop is runner and kwargs["attempt"].attempt_index == 2:
                raise RuntimeError("crash before budget checkpoint")
            return original_checkpoint(loop, **kwargs)

        with patch.object(
            variation_loop.BoundedCandidateLoop,
            "_checkpoint",
            new=crash_before_budget_checkpoint,
        ):
            with self.assertRaisesRegex(RuntimeError, "budget checkpoint"):
                runner.run(task, seed=0)
        attempt_events = [
            event for event in self.ledger.current_valid_events()
            if event["event_type"] == "VARIATION_ATTEMPT"
        ]
        self.assertEqual(len(attempt_events), 2)
        first, target = attempt_events
        crossed_payload = dict(target["payload"])
        crossed_payload["receipt_ids"] = list(first["payload"]["receipt_ids"])
        payload_json = canonical_json(crossed_payload)
        payload_hash = digest_for(payload_json.encode("utf-8"))
        immutable = {
            key: target.get(key)
            for key in (
                "event_id", "campaign_id", "run_id", "task_id", "event_type", "valid_time",
                "subject_id", "payload_hash", "payload_json", "blob_digest", "source_class",
                "disposition", "evaluator_identity", "idempotency_key",
            )
        }
        immutable["payload_hash"] = payload_hash
        immutable["payload_json"] = payload_json
        immutable["event_id"] = None
        new_event_id = content_id(
            "evt",
            {key: immutable[key] for key in immutable if key != "event_id"},
        )
        values = dict(target)
        values.update({
            "event_id": new_event_id,
            "payload_hash": payload_hash,
            "payload_json": payload_json,
        })
        new_event_hash = chain_digest(
            target["previous_hash"],
            self.ledger._event_record_for_hash(values),
        )
        self.ledger.connection.execute("DROP TRIGGER events_no_update")
        self.ledger.connection.execute(
            """UPDATE events SET event_id=?,payload_hash=?,payload_json=?,event_hash=?
               WHERE sequence=?""",
            (new_event_id, payload_hash, payload_json, new_event_hash, target["sequence"]),
        )
        self.ledger.connection.execute(
            "UPDATE meta SET value=? WHERE key='ledger_head_hash'",
            (new_event_hash,),
        )
        self.ledger.connection.commit()
        self.ledger.verify_integrity()
        workspace = runner.isolation.workspace("B", runner._run_id(task.task_id, 0))
        latest = CheckpointStore(workspace.checkpoints).latest(run_id=runner._run_id(task.task_id, 0))
        self.assertIsNotNone(latest)
        calls = (model.generate_calls, tokenizer.decode_calls)
        with self.assertRaisesRegex(VariationCheckpointError, "receipt suffix crossed"):
            runner.run(task, seed=0, resume_from=latest[0])  # type: ignore[index]
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), calls)

    def test_source_contract_resume_after_checkpoint_does_not_reexecute(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, tokenizer, model, _private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )
        original = variation_loop.BoundedCandidateLoop._checkpoint
        crashed = {"value": False}

        def checkpoint_then_crash(loop, **kwargs):
            result = original(loop, **kwargs)
            if loop is runner and not crashed["value"]:
                crashed["value"] = True
                raise RuntimeError("crash after checkpoint")
            return result

        with patch.object(variation_loop.BoundedCandidateLoop, "_checkpoint", new=checkpoint_then_crash):
            with self.assertRaisesRegex(RuntimeError, "checkpoint"):
                runner.run(task, seed=0)
        workspace = runner.isolation.workspace("B", runner._run_id(task.task_id, 0))
        latest = CheckpointStore(workspace.checkpoints).latest(run_id=runner._run_id(task.task_id, 0))
        self.assertIsNotNone(latest)
        report = runner.run(task, seed=0, resume_from=latest[0])  # type: ignore[index]
        self.assertTrue(report.promoted)
        self.assertEqual((model.generate_calls, tokenizer.decode_calls), (1, 1))

    def test_resume_rejects_deleted_historical_checkpoint_file(self) -> None:
        repo = self.corpus.split("train")[0]
        initial = dict(repo.source_files)["src/task.py"].decode("utf-8")
        runner, task, _repo, _generator, _tokenizer, _model, _private_store = self.make_source_contract_runner(
            [initial, repo.corrected_source.decode("utf-8")], max_attempts=2
        )
        report = runner.run(task, seed=0)
        store = CheckpointStore(Path(report.checkpoint_path).parent)
        history = store.inventory(run_id=report.run_id)
        self.assertEqual([item[1].attempt_index for item in history], [1, 2])
        history[0][0].chmod(0o644)
        history[0][0].unlink()
        with self.assertRaisesRegex(VariationCheckpointError, "complete prefix|durable ledger projections"):
            runner.run(task, seed=0, resume_from=history[-1][0])

    def test_resume_rejects_multiple_checkpoint_states_for_one_attempt(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, _tokenizer, _model, _private_store = self.make_source_contract_runner(
            [valid], max_attempts=1
        )
        report = runner.run(task, seed=0)
        store = CheckpointStore(Path(report.checkpoint_path).parent)
        latest = store.latest(run_id=report.run_id)
        self.assertIsNotNone(latest)
        forked = replace(latest[1], state_digest=digest_for("forked-checkpoint-state"))  # type: ignore[index]
        store.save(forked)
        with self.assertRaisesRegex(VariationCheckpointError, "multiple states"):
            runner.run(task, seed=0, resume_from=Path(report.checkpoint_path))

    def test_source_contract_twelve_invalid_responses_fail_closed_and_resume_does_not_regenerate(self) -> None:
        runner, task, _repo, _generator, tokenizer, model, private_store = self.make_source_contract_runner(
            ["not valid Python"] * 12,
            max_attempts=12,
        )
        with self.assertRaises(SourceContractBudgetExhausted) as raised:
            runner.run(task, seed=0)
        self.assertEqual(raised.exception.failure_count, 12)
        failures = private_store.source_contract_failures(
            run_id=raised.exception.run_id,
            task_id=task.task_id,
            arm_id="B",
        )
        self.assertEqual([item["attempt_index"] for item in failures], list(range(1, 13)))
        self.assertEqual(
            self.ledger.connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0],
            0,
        )
        calls = (tokenizer.decode_calls, model.generate_calls)
        with self.assertRaises(SourceContractBudgetExhausted):
            runner.run(task, seed=0)
        self.assertEqual((tokenizer.decode_calls, model.generate_calls), calls)

    def test_source_contract_generation_evidence_is_bound_to_ledger_and_exact_freeze_prompt(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, generator, _tokenizer, _model, private_store = self.make_source_contract_runner(
            [valid],
            max_attempts=1,
        )
        report = runner.run(task, seed=0)
        private = private_store.load_attempts([report.attempts[0].candidate_id])[0]
        row = self.ledger.connection.execute(
            "SELECT candidate_json FROM candidates WHERE candidate_id=?",
            (private.candidate_id,),
        ).fetchone()
        metadata = json.loads(row["candidate_json"])["metadata"]
        self.assertEqual(private.generation_evidence_digest, metadata["generation_evidence_digest"])
        cutoff = LedgerCutoff.capture(self.ledger, campaign_id=report.campaign_id)
        builder = TrajectoryDatasetBuilder(
            self.ledger,
            self.corpus,
            token_counter=lambda prompt, target: len(prompt) + len(target) + 1,
            receipt_public_key=runner.evaluator.signer.public_key,
            require_complete=False,
            generation_profile_digest=generator.generation_profile_digest,
        )
        dataset = builder.build(cutoff, [private])
        self.assertEqual(dataset.examples[0].prompt.encode("utf-8"), private.rendered_prompt)
        forged = replace(private, generation_evidence_digest=digest_for("forged-generation-record"))
        with self.assertRaisesRegex(ValueError, "generation evidence"):
            builder.build(cutoff, [forged])

    def test_private_generation_and_trajectory_hardlinks_fail_closed(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, _tokenizer, _model, private_store = self.make_source_contract_runner(
            [valid],
            max_attempts=1,
        )
        report = runner.run(task, seed=0)
        candidate_id = report.attempts[0].candidate_id
        record = private_store.records / (candidate_id + ".json")
        record_alias = self.root / "record-hardlink.json"
        try:
            os.link(record, record_alias)
        except OSError as exc:
            self.skipTest("hardlink creation is unavailable: {}".format(exc))
        with self.assertRaisesRegex(Exception, "linked or not regular"):
            private_store.load_attempts([candidate_id])
        record_alias.chmod(0o600)
        record_alias.unlink()
        record.chmod(0o400)
        generation = private_store.generation_records / (candidate_id + ".json")
        generation_alias = self.root / "generation-hardlink.json"
        os.link(generation, generation_alias)
        with self.assertRaisesRegex(Exception, "linked or not regular"):
            private_store.load_attempts([candidate_id])
        generation_alias.chmod(0o600)
        generation_alias.unlink()
        generation.chmod(0o400)

    def test_private_generation_proposal_source_substitution_fails_closed(self) -> None:
        valid = self.corpus.split("train")[0].corrected_source.decode("utf-8")
        runner, task, _repo, _generator, _tokenizer, _model, private_store = self.make_source_contract_runner(
            [valid],
            max_attempts=1,
        )
        report = runner.run(task, seed=0)
        candidate_id = report.attempts[0].candidate_id
        path = private_store.generation_records / (candidate_id + ".json")
        value = json.loads(path.read_text(encoding="utf-8"))
        value["proposal_source_digest"] = digest_for("substituted-proposal-source")
        path.chmod(0o600)
        path.write_text(canonical_json(value), encoding="utf-8")
        with self.assertRaisesRegex(Exception, "proposal differs"):
            private_store.load_attempts([candidate_id])

    def test_forged_task_metadata_cannot_cross_the_evaluation_manifest_boundary(self) -> None:
        runner, task, _repo, _isolation = self.make_runner()
        forged = replace(task, public_rule_id="rule-forged")
        with self.assertRaises(VariationConfigurationError):
            runner.run(forged, seed=0)

    def test_pinned_model_manifest_is_local_hashed_and_fails_closed_on_tamper(self) -> None:
        model_root = self.root / "model"
        model_root.mkdir()
        weights = model_root / "weights.safetensors"
        weights.write_bytes(b"test-only-model-bytes")
        manifest = build_local_manifest(
            model_root,
            license_name="test-license",
            license_source="test-source",
        )
        (model_root / "model-manifest.json").write_text(canonical_json(manifest.to_dict()) + "\n", encoding="utf-8")
        preflight = PinnedModelLoader(model_root).preflight()
        self.assertEqual(preflight["repository"], MODEL_REPOSITORY)
        self.assertEqual(preflight["revision"], MODEL_REVISION)
        self.assertEqual(preflight["architecture"], MODEL_ARCHITECTURE)
        self.assertEqual(preflight["config_class"], MODEL_CONFIG_CLASS)
        self.assertEqual(preflight["network"], "offline-environment-scoped-preflight")
        self.assertEqual(set(preflight["offline_environment"].values()), {"1"})

        (model_root / "unlisted-extra.bin").write_bytes(b"must-not-be-ignored")
        with self.assertRaises(VariationConfigurationError):
            PinnedModelLoader(model_root).preflight()
        (model_root / "unlisted-extra.bin").unlink()

        calls = {}

        class Qwen3_5TextConfig:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                calls["config"] = (args, kwargs)
                calls.setdefault("offline", []).append(("config", __import__("os").environ.get("HF_HUB_OFFLINE"), __import__("os").environ.get("TRANSFORMERS_OFFLINE")))
                return cls()

        class Qwen3_5ForCausalLM:
            def __init__(self):
                self.config = Qwen3_5TextConfig()

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                calls["model"] = (args, kwargs)
                calls.setdefault("offline", []).append(("model", __import__("os").environ.get("HF_HUB_OFFLINE"), __import__("os").environ.get("TRANSFORMERS_OFFLINE")))
                return cls(), {"missing_keys": [], "unexpected_keys": []}

            def state_dict(self):
                return {}

        class AutoTokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                calls["tokenizer"] = (args, kwargs)
                calls.setdefault("offline", []).append(("tokenizer", __import__("os").environ.get("HF_HUB_OFFLINE"), __import__("os").environ.get("TRANSFORMERS_OFFLINE")))
                return cls()

        fake_transformers = SimpleNamespace(
            Qwen3_5TextConfig=Qwen3_5TextConfig,
            Qwen3_5ForCausalLM=Qwen3_5ForCausalLM,
            AutoTokenizer=AutoTokenizer,
        )
        with patch("egv.variation.model.importlib_metadata.version", return_value="5.5.0"), patch(
            "egv.variation.model.importlib.import_module", return_value=fake_transformers
        ):
            loaded = PinnedModelLoader(model_root).load()
        self.assertEqual(loaded.manifest_digest, manifest.digest())
        self.assertTrue(calls["config"][1]["local_files_only"])
        self.assertFalse(calls["config"][1]["trust_remote_code"])
        self.assertEqual(calls["config"][1]["revision"], MODEL_REVISION)
        self.assertTrue(calls["model"][1]["local_files_only"])
        self.assertFalse(calls["model"][1]["trust_remote_code"])
        self.assertEqual(calls["tokenizer"][1]["revision"], MODEL_REVISION)
        self.assertEqual(calls["offline"], [("config", "1", "1"), ("model", "1", "1"), ("tokenizer", "1", "1")])

        weights.write_bytes(b"tampered")
        with self.assertRaises(VariationConfigurationError):
            PinnedModelLoader(model_root).preflight()

        external_manifest = self.root / "external-model-manifest.json"
        external_manifest.write_text(canonical_json(manifest.to_dict()) + "\n", encoding="utf-8")
        with self.assertRaises(VariationConfigurationError):
            PinnedModelLoader(model_root, manifest_path=external_manifest).verify_manifest()

    def test_pinned_model_manifest_rejects_wrong_revision_and_loader_dependency(self) -> None:
        value = {
            "schema_version": MODEL_MANIFEST_SCHEMA,
            "repository": MODEL_REPOSITORY,
            "revision": "mutable-main",
            "architecture": MODEL_ARCHITECTURE,
            "config_class": MODEL_CONFIG_CLASS,
            "transformers_version": "5.5.0",
            "files": {"weights": digest_bytes(b"bytes")},
            "license": {"name": "test", "source": "test"},
        }
        with self.assertRaises(VariationConfigurationError):
            from egv.variation.model import PinnedModelManifest

            PinnedModelManifest.from_mapping(value)

    def test_loader_attests_real_adapter_application_and_generator_rechecks_state(self) -> None:
        model_root = self.root / "attested-model"
        model_root.mkdir()
        (model_root / "weights.safetensors").write_bytes(b"attestation-model")
        manifest = build_local_manifest(model_root, license_name="test", license_source="test")
        (model_root / "model-manifest.json").write_text(canonical_json(manifest.to_dict()) + "\n", encoding="utf-8")
        artifact = self.make_adapter_artifact()

        class Qwen3_5TextConfig:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        class BaseModel:
            def __init__(self):
                self.config = Qwen3_5TextConfig()

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls(), {"missing_keys": [], "unexpected_keys": []}

            def state_dict(self):
                return {"base": b"base-state"}

        class AutoTokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        class AdapterConfig:
            def to_dict(self):
                return {"lora_alpha": 8, "peft_type": "LORA", "r": 4}

        class PeftModel:
            def __init__(self, model):
                self.config = model.config
                self.peft_config = {"default": AdapterConfig()}
                self.active_adapters = ["default"]

            def state_dict(self):
                return {"base": b"base-state", "lora": b"applied-lora"}

            @classmethod
            def from_pretrained(cls, model, root, **kwargs):
                return cls(model)

        fake_transformers = SimpleNamespace(
            Qwen3_5TextConfig=Qwen3_5TextConfig,
            Qwen3_5ForCausalLM=BaseModel,
            AutoTokenizer=AutoTokenizer,
        )
        def import_local(name):
            return fake_transformers if name == "transformers" else sys.modules[name]

        with patch("egv.variation.model.importlib_metadata.version", return_value="5.5.0"), patch(
            "egv.variation.model.importlib.import_module", side_effect=import_local
        ), patch.dict(sys.modules, {"peft": SimpleNamespace(PeftModel=PeftModel)}):
            loaded = PinnedModelLoader(model_root).load(adapter_artifact=artifact)
        self.assertIsNotNone(loaded.adapter_attestation)
        loaded.adapter_attestation.validate()  # type: ignore[union-attr]
        self.assertEqual(loaded.adapter_attestation.adapter_digest, artifact.digest)  # type: ignore[union-attr]
        with patch.dict(sys.modules, {"peft": SimpleNamespace(PeftModel=PeftModel)}):
            generator = ModelCandidateGenerator(
                loaded,
                model_digest=manifest.digest(),
                adapter_digest=artifact.digest,
                adapter_artifact=artifact,
            )
            generator.validate_production_integrity()

    def test_variation_smoke_separates_private_state_and_public_report(self) -> None:
        output = self.root / "smoke-output"
        report = run_variation_smoke(output)
        self.assertEqual(report["smoke"], "PASS")
        self.assertEqual(report["runtime_tier"], "floor-fixture")
        self.assertTrue((output / "private" / "evaluator-private" / "corpus-seed.bin").is_file())
        public_path = output / "public" / "variation-smoke-report.json"
        self.assertTrue(public_path.is_file())
        public_text = public_path.read_text(encoding="utf-8")
        self.assertNotIn("corpus-seed.bin", public_text)
        self.assertNotIn('"expected_output"', public_text)
        self.assertNotIn("golden_patch", public_text)
        self.assertNotIn("/home/", public_text)
        self.assertNotIn("egv-pure_function-heldout-1-v1", public_text)
        self.assertNotIn("receipt_ids", public_text)
        self.assertNotIn('"seed":', public_text)
        self.assertNotIn("candidate_artifact_digest", public_text)
        self.assertNotIn("corrected_source_digest", public_text)
        self.assertFalse(report["campaign_path_exercised"])
        self.assertTrue(report["public_scan"]["checked"])
        self.assertEqual(public_path.stat().st_mode & 0o222, 0)

    def test_variation_smoke_public_artifact_is_byte_identical_across_fresh_roots(self) -> None:
        first_root = self.root / "smoke-one"
        second_root = self.root / "smoke-two"
        run_variation_smoke(first_root)
        run_variation_smoke(second_root)
        first = (first_root / "public" / "variation-smoke-report.json").read_bytes()
        second = (second_root / "public" / "variation-smoke-report.json").read_bytes()
        self.assertEqual(first, second)

    def test_cli_variation_smoke_is_truthful_and_redacted(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "egv", "variation", "smoke", "--json"],
            cwd=str(Path(__file__).resolve().parents[1]),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["runtime_tier"], "floor-fixture")
        self.assertEqual(report["variation"]["terminal_status"], "PROMOTED")
        self.assertFalse(report["authority"]["enforceable"])
        self.assertNotIn("corpus-seed.bin", completed.stdout)
        self.assertNotIn('"expected_output"', completed.stdout)
        self.assertNotIn("egv-pure_function-heldout-1-v1", completed.stdout)
        self.assertNotIn("/home/", completed.stdout)
        self.assertNotIn('"seed":', completed.stdout)
        self.assertNotIn("candidate_artifact_digest", completed.stdout)
        self.assertFalse(report["campaign_path_exercised"])

    def test_cli_model_preflight_verifies_local_manifest_without_loading(self) -> None:
        model_root = self.root / "cli-model"
        model_root.mkdir()
        (model_root / "weights.safetensors").write_bytes(b"cli-test-model")
        manifest = build_local_manifest(model_root, license_name="test", license_source="test")
        (model_root / "model-manifest.json").write_text(canonical_json(manifest.to_dict()) + "\n", encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, "-m", "egv", "variation", "model-preflight", "--model-root", str(model_root), "--json"],
            cwd=str(Path(__file__).resolve().parents[1]),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["revision"], MODEL_REVISION)
        self.assertEqual(report["network"], "offline-environment-scoped-preflight")
        self.assertEqual(set(report["offline_environment"].values()), {"1"})

    def test_public_variation_scanner_rejects_seed_and_candidate_source_fields(self) -> None:
        with self.assertRaises(LeakageError):
            scan_public_variation_report({"seed": 0})
        with self.assertRaises(LeakageError):
            scan_public_variation_report({"attempts": [{"candidate_artifact_digest": digest_for("source")}]})

    def test_public_variation_docs_match_the_closed_runtime_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        slice_doc = (root / "docs" / "egv-variation-slice.md").read_text(encoding="utf-8")
        runbook = (root / "docs" / "runbooks" / "evidence-governed-variation-dual-spark.md").read_text(
            encoding="utf-8"
        )
        for text in (slice_doc, runbook):
            self.assertIn("floor-fixture", text)
            self.assertIn("variation smoke", text)
            self.assertIn(MODEL_REVISION, text)
            self.assertIn("fail closed", text)
            self.assertNotIn("41001", text)
            self.assertNotIn("41002", text)
        self.assertIn("Training and Campaign", slice_doc)


if __name__ == "__main__":
    unittest.main()
