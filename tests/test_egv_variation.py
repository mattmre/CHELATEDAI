"""Focused tests for the bounded EGV Variation slice."""

from __future__ import annotations

import json
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

from egv.canonical import canonical_json, digest_bytes, digest_for, failure_family_root
from egv.evaluation.authority import AuthorityBroker
from egv.evaluation.controller import EvaluationResult, EvaluatorController, HiddenEvaluatorRunner
from egv.evaluation.dataset import EvaluationCorpus, EVALUATOR_SEED_BYTES, FAMILY_SPECS
from egv.evaluation.sandbox import DockerCandidateSandbox
from egv.evaluation.errors import DockerConfigurationError, LeakageError
from egv.ledger import EvidenceLedger
from egv.receipts import ReceiptJournal, ReceiptSigner
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
    SealedAdapterArtifact,
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
from egv.variation.generator import CandidateContext, CandidateProposal, render_candidate_prompt
import egv.variation.loop as variation_loop
from egv.variation.loop import CANDIDATE_SOURCE_LIMIT
from egv.variation.model import MODEL_ARCHITECTURE, MODEL_CONFIG_CLASS, MODEL_MANIFEST_SCHEMA, MODEL_REPOSITORY


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
            True,
            fixture_evaluator,
            fixture_generator,
            fixture_isolation,
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

        source = "def solve(value):\n    return value + 1"
        repaired = ModelCandidateGenerator._parse_response(source, context)
        self.assertEqual(repaired.source, source.encode("utf-8"))
        self.assertEqual(repaired.declared_locus, context.public_locus)
        self.assertEqual(repaired.requested_authority, "EXECUTE_CANDIDATE")
        self.assertEqual(repaired.evidence_ids, ())
        self.assertEqual(repaired.metadata["response_contract"], "source-only-repair-v1")
        self.assertEqual(repaired.metadata["raw_response_digest"], digest_bytes(repaired.source))
        repaired.validate(context, source_limit=CANDIDATE_SOURCE_LIMIT)

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
