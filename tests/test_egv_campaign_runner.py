from __future__ import annotations

import json
import io
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stdout

from egv.canonical import canonical_bytes, canonical_json, content_id, digest_for
from egv.campaign.commissioning import prepare_commissioning
from egv.campaign.runner import (
    CommissioningRunError,
    CommissioningRunJournal,
    CommissioningTrainerInputs,
    CommissioningTrainerSources,
)
from egv.cli import main
from egv.evaluation.dataset import EvaluationCorpus
from egv.identities import commissioning_run_id
from egv.training.contracts import FrozenTrainingDataset, LedgerCutoff
from egv.training.dataset import SEALED_RUNTIME_DATASET_SCHEMA, seal_runtime_dataset
from egv.variation.arms import arm_policy
from egv.variation.generator import CandidateContext, model_generation_profile_digest
from egv.variation.loop import BoundedCandidateLoop
from egv.variation.private import PrivateTrajectoryStore


class CommissioningRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.seed = self.root / "seed.bin"
        self.seed.write_bytes(bytes(range(32)))
        self.corpus = EvaluationCorpus.generate(secret_seed_file=self.seed)
        self.model_digest = digest_for("model")
        self.generation_profile_digest = model_generation_profile_digest(
            "source-only-v1",
            model_manifest_digest=self.model_digest,
            chat_template_digest=digest_for("pinned-chat-template"),
        )
        self.plan = prepare_commissioning(
            self.corpus,
            campaign_id="campaign-public",
            model_manifest_digest=self.model_digest,
            generation_profile_digest=self.generation_profile_digest,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_shared_run_identity_matches_every_frozen_request_and_loop(self) -> None:
        for request in self.plan.generation_requests:
            expected = commissioning_run_id(
                campaign_id=request.campaign_id,
                task_id=request.task_id,
                arm_id=request.arm_id,
                seed=request.seed,
            )
            fake = SimpleNamespace(campaign_id=request.campaign_id, policy=arm_policy(request.arm_id))
            self.assertEqual(request.run_id, expected)
            self.assertEqual(BoundedCandidateLoop._run_id(fake, request.task_id, request.seed), expected)

    def test_trainer_bundle_is_closed_exact_and_contains_no_dev_identity(self) -> None:
        value = self.plan.trainer_inputs()
        loaded = CommissioningTrainerInputs(value)
        self.assertEqual(len(loaded.requests), 80)
        encoded = json.dumps(value, sort_keys=True)
        for repo in self.corpus.split("dev") + self.corpus.split("heldout"):
            self.assertNotIn(repo.template_id, encoded)
            self.assertNotIn(str(repo.evaluator_input), encoded)
        changed = dict(value)
        changed["campaign_id"] = "different"
        with self.assertRaises(CommissioningRunError):
            CommissioningTrainerInputs(changed)

    @staticmethod
    def _redigest_trainer_inputs(value: dict) -> dict:
        value.pop("trainer_inputs_digest", None)
        value["trainer_inputs_digest"] = digest_for(value)
        return value

    @staticmethod
    def _redigest_trainer_sources(value: dict) -> dict:
        value.pop("trainer_sources_digest", None)
        value["trainer_sources_digest"] = digest_for(value)
        return value

    @staticmethod
    def _write_canonical(path: Path, value: dict) -> None:
        path.write_text(canonical_json(value) + "\n", encoding="utf-8")

    def test_trainer_bundle_rejects_duplicate_request_envelope(self) -> None:
        value = json.loads(json.dumps(self.plan.trainer_inputs()))
        requests = value["request_manifest"]["requests"]
        requests[0] = dict(requests[1])
        requests.sort(key=lambda item: item["request_id"])
        self._redigest_trainer_inputs(value)
        with self.assertRaisesRegex(CommissioningRunError, "matrix or canonical bindings"):
            CommissioningTrainerInputs(value)

    def test_trainer_bundle_rejects_substituted_run_identity(self) -> None:
        value = json.loads(json.dumps(self.plan.trainer_inputs()))
        request = dict(value["request_manifest"]["requests"][0])
        request["run_id"] = "egv-run-substituted"
        request.pop("request_id")
        request["request_id"] = content_id("genreq", request)
        value["request_manifest"]["requests"][0] = request
        value["request_manifest"]["requests"].sort(key=lambda item: item["request_id"])
        self._redigest_trainer_inputs(value)
        with self.assertRaisesRegex(CommissioningRunError, "matrix or canonical bindings"):
            CommissioningTrainerInputs(value)

    def test_trainer_bundle_rejects_missing_or_substituted_declared_seed(self) -> None:
        for seeds in ([0], [0, 2]):
            with self.subTest(seeds=seeds):
                value = json.loads(json.dumps(self.plan.trainer_inputs()))
                value["request_manifest"]["seeds"] = seeds
                self._redigest_trainer_inputs(value)
                with self.assertRaisesRegex(CommissioningRunError, "request manifest is invalid"):
                    CommissioningTrainerInputs(value)

    def test_trainer_sources_reject_tampered_exact_source_even_when_bundle_is_redigested(self) -> None:
        sources = json.loads(json.dumps(self.plan.trainer_source_manifest))
        sources["tasks"][0]["source_files"][1]["content_utf8"] += "\n# tampered\n"
        self._redigest_trainer_sources(sources)
        trainer = json.loads(json.dumps(self.plan.trainer_inputs()))
        trainer["trainer_sources_digest"] = sources["trainer_sources_digest"]
        self._redigest_trainer_inputs(trainer)
        inputs = CommissioningTrainerInputs(trainer)
        with self.assertRaisesRegex(CommissioningRunError, "source bytes differ"):
            CommissioningTrainerSources(sources, trainer_inputs=inputs)

    def test_trainer_sources_reject_missing_task_even_when_bundle_is_redigested(self) -> None:
        sources = json.loads(json.dumps(self.plan.trainer_source_manifest))
        sources["tasks"].pop()
        sources["task_count"] = len(sources["tasks"])
        self._redigest_trainer_sources(sources)
        trainer = json.loads(json.dumps(self.plan.trainer_inputs()))
        trainer["trainer_sources_digest"] = sources["trainer_sources_digest"]
        self._redigest_trainer_inputs(trainer)
        inputs = CommissioningTrainerInputs(trainer)
        with self.assertRaisesRegex(CommissioningRunError, "digest or identity"):
            CommissioningTrainerSources(sources, trainer_inputs=inputs)

    def test_trainer_sources_reject_symlink_bundle(self) -> None:
        backing = self.root / "trainer-sources-backing.json"
        link = self.root / "trainer-sources-link.json"
        self._write_canonical(backing, self.plan.trainer_source_manifest)
        try:
            link.symlink_to(backing)
        except OSError as exc:
            self.skipTest("source-bundle symlink creation is unavailable: {}".format(exc))
        with self.assertRaisesRegex(CommissioningRunError, "regular non-symlink"):
            CommissioningTrainerSources.from_path(
                link,
                trainer_inputs=CommissioningTrainerInputs(self.plan.trainer_inputs()),
            )

    def test_trainer_inputs_reject_response_contract_and_generation_profile_mismatch(self) -> None:
        for field, replacement in (
            ("response_contract", "closed-json-v1"),
            ("generation_profile_digest", digest_for("substituted-generation-profile")),
        ):
            with self.subTest(field=field):
                trainer = json.loads(json.dumps(self.plan.trainer_inputs()))
                trainer[field] = replacement
                self._redigest_trainer_inputs(trainer)
                with self.assertRaises(CommissioningRunError):
                    CommissioningTrainerInputs(trainer)

    def test_training_evaluator_cli_forwards_adapter_store(self) -> None:
        paths = {
            "service_manifest": self.root / "service.json",
            "model_root": self.root / "model",
            "development_dataset": self.root / "development.json",
            "private_key": self.root / "private.key",
            "adapter_store": self.root / "adapters",
        }
        output = io.StringIO()
        with mock.patch(
            "egv.cli.run_external_evaluator_once", autospec=True, return_value={"status": "ok"}
        ) as evaluator, mock.patch("sys.stdin", io.StringIO('{"request": "fixture"}')), redirect_stdout(output):
            code = main([
                "training", "evaluator-once",
                "--service-manifest", str(paths["service_manifest"]),
                "--model-root", str(paths["model_root"]),
                "--development-dataset", str(paths["development_dataset"]),
                "--private-key", str(paths["private_key"]),
                "--adapter-store", str(paths["adapter_store"]),
                "--device", "cpu",
            ])
        self.assertEqual(code, 0)
        evaluator.assert_called_once_with(
            {"request": "fixture"},
            service_manifest=paths["service_manifest"],
            model_root=paths["model_root"],
            development_dataset=paths["development_dataset"],
            evaluator_private_key=paths["private_key"],
            adapter_store=paths["adapter_store"],
            device="cpu",
        )
        self.assertEqual(json.loads(output.getvalue()), {"status": "ok"})

    def test_private_sidecar_round_trip_is_content_bound_and_conflict_rejected(self) -> None:
        store = PrivateTrajectoryStore(self.root / "private")
        context = CandidateContext(
            "campaign-public", "egv-run-123", 0, "B", "task-1", "PURE_FUNCTION",
            "module.py:solve", "rule", 1, None, tuple(), digest_for([]),
            digest_for("model"), None, digest_for("prompt"),
        )
        store.record(candidate_id="candidate-1", context=context, source=b"return 1\n")
        attempt = store.load_attempts(["candidate-1"])[0]
        self.assertEqual(attempt.context, context)
        self.assertEqual(attempt.candidate_source, b"return 1\n")
        store.record(candidate_id="candidate-1", context=context, source=b"return 1\n")
        with self.assertRaises(Exception):
            store.record(candidate_id="candidate-1", context=context, source=b"return 2\n")

    def test_private_store_rejects_symlink_root(self) -> None:
        backing = self.root / "private-backing"
        backing.mkdir()
        linked_root = self.root / "private-linked-root"
        try:
            linked_root.symlink_to(backing, target_is_directory=True)
        except OSError as exc:
            self.skipTest("directory symlink creation is unavailable: {}".format(exc))
        with self.assertRaisesRegex(Exception, "link or reparse"):
            PrivateTrajectoryStore(linked_root)

    def test_private_store_rejects_precreated_linked_record_directory(self) -> None:
        root = self.root / "private-linked-records"
        external = self.root / "external-records"
        root.mkdir()
        external.mkdir()
        try:
            (root / "records").symlink_to(external, target_is_directory=True)
        except OSError as exc:
            self.skipTest("directory symlink creation is unavailable: {}".format(exc))
        with self.assertRaisesRegex(Exception, "contained directory"):
            PrivateTrajectoryStore(root)

    def test_run_journal_is_idempotent_and_bound_to_inputs(self) -> None:
        request = self.plan.generation_requests[0]
        path = self.root / "run.json"
        journal = CommissioningRunJournal(
            path,
            trainer_inputs_digest=self.plan.trainer_inputs()["trainer_inputs_digest"],
            requests=self.plan.generation_requests,
        )
        result = {
            "request_id": request.request_id,
            "run_id": request.run_id,
            "terminal_status": "BUDGET_EXHAUSTED",
            "report_digest": None,
            "response": None,
            "generation_failure_digest": digest_for("last-source-contract-failure"),
            "source_contract_failure_count": 1,
        }
        journal.commit(request, result)
        journal.commit(request, result)
        self.assertEqual(journal.completed(request.request_id), result)
        with self.assertRaises(CommissioningRunError):
            journal.commit(
                request,
                {**result, "generation_failure_digest": digest_for("conflicting-failure")},
            )
        with self.assertRaises(CommissioningRunError):
            CommissioningRunJournal(
                path,
                trainer_inputs_digest=digest_for("other"),
                requests=self.plan.generation_requests,
            )

    def test_run_journal_rejects_unknown_partial_and_interrupted_records(self) -> None:
        requests = self.plan.generation_requests
        trainer_digest = self.plan.trainer_inputs()["trainer_inputs_digest"]
        path = self.root / "strict-run.json"
        journal = CommissioningRunJournal(
            path,
            trainer_inputs_digest=trainer_digest,
            requests=requests,
        )
        value = journal.load()
        value["completed"] = {"unknown-request": {}}
        path.write_bytes(canonical_bytes(value))
        with self.assertRaisesRegex(CommissioningRunError, "unknown request"):
            journal.load()

        value["completed"] = {requests[0].request_id: {"request_id": requests[0].request_id}}
        path.write_bytes(canonical_bytes(value))
        with self.assertRaisesRegex(CommissioningRunError, "closed record"):
            journal.load()

        value["completed"] = {}
        path.write_bytes(canonical_bytes(value))
        (self.root / ".commissioning-run-interrupted").write_text("partial", encoding="utf-8")
        with self.assertRaisesRegex(CommissioningRunError, "interrupted ambiguous"):
            journal.load()

    def test_prepare_cli_writes_separate_public_trainer_and_private_evaluator_inputs(self) -> None:
        trainer = self.root / "trainer.json"
        sources = self.root / "trainer-sources.json"
        evaluator = self.root / "evaluator.json"
        with mock.patch(
            "egv.cli._commissioning_generation_profile",
            return_value=self.generation_profile_digest,
        ):
            code = main([
                "commissioning", "prepare", "--campaign-id", "campaign-public",
                "--model-digest", self.model_digest, "--model-root", str(self.root / "model"),
                "--evaluator-seed", str(self.seed), "--trainer-output", str(trainer),
                "--trainer-sources-output", str(sources), "--evaluator-output", str(evaluator),
            ])
        self.assertEqual(code, 0)
        CommissioningTrainerInputs.from_path(trainer)
        CommissioningTrainerSources.from_path(
            sources,
            trainer_inputs=CommissioningTrainerInputs.from_path(trainer),
        )
        trainer_text = trainer.read_text(encoding="utf-8")
        source_text = sources.read_text(encoding="utf-8")
        private_text = evaluator.read_text(encoding="utf-8")
        for repo in self.corpus.split("dev"):
            self.assertNotIn(repo.template_id, trainer_text)
            self.assertNotIn(repo.template_id, source_text)
            self.assertIn(repo.template_id, private_text)

    def test_prepare_cli_rejects_same_output_before_writing_private_material(self) -> None:
        shared = self.root / "shared.json"
        with mock.patch(
            "egv.cli._commissioning_generation_profile",
            return_value=self.generation_profile_digest,
        ):
            code = main([
                "commissioning", "prepare", "--campaign-id", "campaign-public",
                "--model-digest", self.model_digest, "--model-root", str(self.root / "model"),
                "--evaluator-seed", str(self.seed), "--trainer-output", str(shared),
                "--trainer-sources-output", str(self.root / "sources.json"),
                "--evaluator-output", str(shared),
            ])
        self.assertEqual(code, 1)
        self.assertFalse(shared.exists())

    def test_prepare_cli_rejects_symlink_and_hardlink_output_aliases(self) -> None:
        backing = self.root / "backing.json"
        backing.write_text("preserve", encoding="utf-8")
        symlink = self.root / "trainer-link.json"
        try:
            symlink.symlink_to(backing)
        except OSError:
            symlink = None
        if symlink is not None:
            private = self.root / "private.json"
            with mock.patch(
                "egv.cli._commissioning_generation_profile",
                return_value=self.generation_profile_digest,
            ):
                code = main([
                    "commissioning", "prepare", "--campaign-id", "campaign-public",
                    "--model-digest", self.model_digest, "--model-root", str(self.root / "model"),
                    "--evaluator-seed", str(self.seed), "--trainer-output", str(symlink),
                    "--trainer-sources-output", str(self.root / "sources-symlink.json"),
                    "--evaluator-output", str(private),
                ])
            self.assertEqual(code, 1)
            self.assertFalse(private.exists())
            self.assertEqual(backing.read_text(encoding="utf-8"), "preserve")
        first = self.root / "first-hardlink.json"
        second = self.root / "second-hardlink.json"
        first.write_text("preserve", encoding="utf-8")
        try:
            os.link(first, second)
        except OSError:
            return
        with mock.patch(
            "egv.cli._commissioning_generation_profile",
            return_value=self.generation_profile_digest,
        ):
            code = main([
                "commissioning", "prepare", "--campaign-id", "campaign-public",
                "--model-digest", self.model_digest, "--model-root", str(self.root / "model"),
                "--evaluator-seed", str(self.seed), "--trainer-output", str(first),
                "--trainer-sources-output", str(self.root / "sources-hardlink.json"),
                "--evaluator-output", str(second),
            ])
        self.assertEqual(code, 1)
        self.assertEqual(first.read_text(encoding="utf-8"), "preserve")

    def test_sealed_runtime_dataset_uses_exact_train_lora_schema(self) -> None:
        cutoff = LedgerCutoff(
            "campaign-public", 1, "event-1", digest_for("event"), digest_for("receipt"), 1, "key-1"
        )
        dataset = FrozenTrainingDataset(cutoff, tuple(), {})
        output = self.root / "training.json"
        report = seal_runtime_dataset(dataset, output)
        value = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(value["schema_version"], SEALED_RUNTIME_DATASET_SCHEMA)
        self.assertEqual(value["manifest"], dataset.manifest())
        self.assertEqual(value["private_rows"], [])
        self.assertEqual(report["dataset_digest"], dataset.digest)


if __name__ == "__main__":
    unittest.main()
