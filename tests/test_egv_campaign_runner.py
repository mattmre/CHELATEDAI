from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from egv.canonical import digest_for
from egv.campaign.commissioning import prepare_commissioning
from egv.campaign.runner import (
    CommissioningRunError,
    CommissioningRunJournal,
    CommissioningTrainerInputs,
)
from egv.cli import main
from egv.evaluation.dataset import EvaluationCorpus
from egv.identities import commissioning_run_id
from egv.training.contracts import FrozenTrainingDataset, LedgerCutoff
from egv.training.dataset import SEALED_RUNTIME_DATASET_SCHEMA, seal_runtime_dataset
from egv.variation.arms import arm_policy
from egv.variation.generator import CandidateContext
from egv.variation.loop import BoundedCandidateLoop
from egv.variation.private import PrivateTrajectoryStore


class CommissioningRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.seed = self.root / "seed.bin"
        self.seed.write_bytes(bytes(range(32)))
        self.corpus = EvaluationCorpus.generate(secret_seed_file=self.seed)
        self.plan = prepare_commissioning(
            self.corpus,
            campaign_id="campaign-public",
            model_manifest_digest=digest_for("model"),
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

    def test_run_journal_is_idempotent_and_bound_to_inputs(self) -> None:
        request = self.plan.generation_requests[0]
        path = self.root / "run.json"
        journal = CommissioningRunJournal(path, trainer_inputs_digest=self.plan.trainer_inputs()["trainer_inputs_digest"])
        result = {"request_id": request.request_id, "terminal_status": "PROMOTED"}
        journal.commit(request, result)
        journal.commit(request, result)
        self.assertEqual(journal.completed(request.request_id), result)
        with self.assertRaises(CommissioningRunError):
            journal.commit(request, {**result, "terminal_status": "FAILED"})
        with self.assertRaises(CommissioningRunError):
            CommissioningRunJournal(path, trainer_inputs_digest=digest_for("other"))

    def test_prepare_cli_writes_separate_public_trainer_and_private_evaluator_inputs(self) -> None:
        trainer = self.root / "trainer.json"
        evaluator = self.root / "evaluator.json"
        code = main([
            "commissioning", "prepare", "--campaign-id", "campaign-public",
            "--model-digest", digest_for("model"), "--evaluator-seed", str(self.seed),
            "--trainer-output", str(trainer), "--evaluator-output", str(evaluator),
        ])
        self.assertEqual(code, 0)
        CommissioningTrainerInputs.from_path(trainer)
        trainer_text = trainer.read_text(encoding="utf-8")
        private_text = evaluator.read_text(encoding="utf-8")
        for repo in self.corpus.split("dev"):
            self.assertNotIn(repo.template_id, trainer_text)
            self.assertIn(repo.template_id, private_text)

    def test_prepare_cli_rejects_same_output_before_writing_private_material(self) -> None:
        shared = self.root / "shared.json"
        code = main([
            "commissioning", "prepare", "--campaign-id", "campaign-public",
            "--model-digest", digest_for("model"), "--evaluator-seed", str(self.seed),
            "--trainer-output", str(shared), "--evaluator-output", str(shared),
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
            code = main([
                "commissioning", "prepare", "--campaign-id", "campaign-public",
                "--model-digest", digest_for("model"), "--evaluator-seed", str(self.seed),
                "--trainer-output", str(symlink), "--evaluator-output", str(private),
            ])
            self.assertEqual(code, 1)
            self.assertFalse(private.exists())
            self.assertEqual(backing.read_text(encoding="utf-8"), "preserve")
        first = self.root / "first-hardlink.json"
        second = self.root / "second-hardlink.json"
        first.write_text("preserve", encoding="utf-8")
        try:
            import os

            os.link(first, second)
        except OSError:
            return
        code = main([
            "commissioning", "prepare", "--campaign-id", "campaign-public",
            "--model-digest", digest_for("model"), "--evaluator-seed", str(self.seed),
            "--trainer-output", str(first), "--evaluator-output", str(second),
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
