"""Focused tests for the bounded EGV Variation slice."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from egv.canonical import canonical_json, digest_bytes, digest_for, failure_family_root
from egv.evaluation.authority import AuthorityBroker
from egv.evaluation.controller import EvaluationResult, EvaluatorController, HiddenEvaluatorRunner
from egv.evaluation.dataset import EvaluationCorpus, EVALUATOR_SEED_BYTES, FAMILY_SPECS
from egv.evaluation.sandbox import DockerCandidateSandbox
from egv.evaluation.errors import LeakageError
from egv.ledger import EvidenceLedger
from egv.receipts import ReceiptJournal, ReceiptSigner
from egv.variation import (
    ADAPTER_MANIFEST_NAME,
    ARM_IDS,
    ArmIsolation,
    BoundedCandidateLoop,
    CheckpointStore,
    ControllerEvaluationGateway,
    DeterministicFixtureGenerator,
    MODEL_REVISION,
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
from egv.variation.generator import CandidateContext, CandidateProposal
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
        evaluator = FixtureEvaluationGateway(
            self.corpus,
            self.ledger,
            self.root / "fixture-evaluator",
            policy_digest=policy_digest,
            campaign_id=campaign_id,
        )
        isolation = ArmIsolation(self.root / "arm-state", campaign_id=campaign_id)
        generator = DeterministicFixtureGenerator(
            {task.task_id: tuple(sources)},
            public_locus={task.task_id: task.public_locus},
            model_digest=model_digest,
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
        (root / "adapter_config.json").write_text('{"r":4,"lora_alpha":8}\n', encoding="utf-8")
        (root / "adapter_model.safetensors").write_bytes(b"sealed-training-output")
        manifest = build_local_adapter_manifest(root)
        (root / ADAPTER_MANIFEST_NAME).write_text(canonical_json(manifest.to_dict()) + "\n", encoding="utf-8")
        return SealedAdapterArtifact(root)

    def test_controller_gateway_binds_real_controller_hidden_runner_and_docker(self) -> None:
        hidden_runner = HiddenEvaluatorRunner.from_corpus(self.corpus, evaluator_revision="gateway-test-evaluator")
        docker_sandbox = object.__new__(DockerCandidateSandbox)
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
        controller.sandbox = docker_sandbox
        controller.hidden_runner = SimpleNamespace(evaluator_revision="forged")
        with self.assertRaises(VariationDependencyError):
            gateway.validate_runtime()

    def test_production_run_revalidates_controller_and_docker_boundary(self) -> None:
        hidden_runner = HiddenEvaluatorRunner.from_corpus(self.corpus, evaluator_revision="run-boundary-evaluator")
        docker_sandbox = object.__new__(DockerCandidateSandbox)
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
        model_digest = digest_for({"production_model": MODEL_REVISION})
        generator = SimpleNamespace(model_digest=model_digest, adapter_digest=None, test_only=False)
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
        runner, task, repo, isolation = self.make_runner(max_attempts=3)

        class StopBeforeSecond(DeterministicFixtureGenerator):
            def propose(self, context):
                if context.attempt_index == 2:
                    raise RuntimeError("intentional interruption")
                return super().propose(context)

        public_source = dict(repo.source_files)["src/task.py"]
        runner.generator = StopBeforeSecond(
            {task.task_id: (public_source, repo.corrected_source)},
            public_locus={task.task_id: task.public_locus},
            model_digest=runner.model_digest,
        )
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

        class PeftModel:
            @classmethod
            def from_pretrained(cls, model, root, **kwargs):
                calls["args"] = (model, root, kwargs)
                return "adapted-model"

        with patch.dict(sys.modules, {"peft": SimpleNamespace(PeftModel=PeftModel)}):
            self.assertEqual(artifact.apply_to("base-model"), "adapted-model")
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
        runner, task, repo, isolation = self.make_runner()

        class InterruptingGenerator(DeterministicFixtureGenerator):
            def propose(self, context):
                if context.attempt_index == 2:
                    raise RuntimeError("simulated trainer interruption")
                return super().propose(context)

        public_source = dict(repo.source_files)["src/task.py"]
        runner.generator = InterruptingGenerator(
            {task.task_id: (public_source, repo.corrected_source)},
            public_locus={task.task_id: task.public_locus},
            model_digest=runner.model_digest,
        )
        with self.assertRaises(RuntimeError):
            runner.run(task, seed=0)
        run_id = runner._run_id(task.task_id, 0)
        workspace = isolation.workspace("C", run_id)
        latest = CheckpointStore(workspace.checkpoints).latest(run_id=run_id)
        self.assertIsNotNone(latest)
        checkpoint_path, checkpoint = latest  # type: ignore[misc]
        self.assertEqual(checkpoint.status, "RUNNING")
        runner.generator = DeterministicFixtureGenerator(
            {task.task_id: (public_source, repo.corrected_source)},
            public_locus={task.task_id: task.public_locus},
            model_digest=runner.model_digest,
        )
        resumed = runner.run(task, seed=0, resume_from=checkpoint_path)
        self.assertTrue(resumed.promoted)
        self.assertEqual(len(resumed.attempts), 2)

    def test_infrastructure_loss_is_incident_bound_and_stops_the_loop(self) -> None:
        runner, task, _repo, isolation = self.make_runner()
        fixture = runner.evaluator
        incident = digest_for("variation-infrastructure-incident")

        class InfrastructureGateway:
            enforceable = False
            evaluator_revision = "variation-infrastructure-test"
            evaluator_digest = digest_for(evaluator_revision)

            def __init__(self, hidden_runner):
                self.hidden_runner = hidden_runner

            def evaluate(self, *, candidate_id, task_id, source, **_kwargs):
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
                        task.family_id,
                        "INTERNAL_ERROR",
                        task.public_locus,
                        task.public_rule_id,
                        infrastructure_incident_id=incident,
                    ),
                )

        runner.evaluator = InfrastructureGateway(fixture.hidden_runner)
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
        self.assertEqual(preflight["network"], "disabled-local-files-only")

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
        self.assertEqual(report["network"], "disabled-local-files-only")

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
