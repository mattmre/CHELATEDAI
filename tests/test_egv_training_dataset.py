"""CPU and adversarial tests for the frozen EGV training dataset."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from egv.canonical import canonical_bytes, canonical_json, digest_bytes, digest_for
from egv.evaluation.controller import HiddenEvaluatorRunner
from egv.evaluation.dataset import EVALUATOR_SEED_BYTES, EvaluationCorpus
from egv.evaluation.sft import SFT_FIELDS
from egv.ledger import EvidenceLedger
from egv.errors import ReceiptVerificationError
from egv.receipts import ReceiptSigner
from egv.training.contracts import LedgerCutoff, PrivateTrajectoryAttempt
from egv.training.dataset import TrajectoryDatasetBuilder
from egv.variation.arms import ArmIsolation
from egv.variation.fixture import FixtureEvaluationGateway
from egv.variation.generator import CandidateContext, DeterministicFixtureGenerator
from egv.variation.loop import BoundedCandidateLoop, VariationTask
from egv.variation.model import MODEL_REVISION


class _RunBoundFixtureGateway(FixtureEvaluationGateway):
    """Fixture receipts use the candidate's real durable run binding."""

    def _common(self, *, task_id: str, candidate_id: str, artifact_digest: str):
        values = super()._common(task_id=task_id, candidate_id=candidate_id, artifact_digest=artifact_digest)
        row = self.ledger.connection.execute(
            "SELECT run_id FROM candidates WHERE candidate_id=?", (candidate_id,)
        ).fetchone()
        values["run_id"] = str(row["run_id"])
        return values


class _RecordingGenerator:
    test_only = True

    def __init__(self, inner):
        self.inner = inner
        self.model_digest = inner.model_digest
        self.adapter_digest = inner.adapter_digest
        self.contexts = []

    def propose(self, context: CandidateContext):
        self.contexts.append(context)
        return self.inner.propose(context)


class TrainingDatasetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="egv-training-data-")
        self.root = Path(self.tempdir.name)
        seed = self.root / "seed.bin"
        seed.write_bytes(b"T" * EVALUATOR_SEED_BYTES)
        seed.chmod(0o600)
        self.corpus = EvaluationCorpus.generate(secret_seed_file=seed)
        self.ledgers = []
        self.receipt_keys = {}

    def tearDown(self) -> None:
        for ledger in self.ledgers:
            ledger.close()
        self.tempdir.cleanup()

    @staticmethod
    def token_count(text: str) -> int:
        return max(1, len(text.split()))

    def build_run(self, *, source=None, arm="B", seed=3):
        campaign = "training-data-test"
        repo = self.corpus.split("train")[0]
        task = VariationTask.from_microrepo(repo)
        source = bytes(source if source is not None else repo.corrected_source)
        ledger = EvidenceLedger(
            ":memory:",
            blob_root=self.root / "blobs-{}-{}".format(arm, len(self.ledgers)),
            clock=lambda: "2026-08-22T12:00:00Z",
        )
        self.ledgers.append(ledger)
        policy = digest_for("training-test-policy")
        evaluator = _RunBoundFixtureGateway(
            self.corpus,
            ledger,
            self.root / "evaluator-{}-{}".format(arm, len(self.ledgers)),
            policy_digest=policy,
            campaign_id=campaign,
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
            public_records={item.template_id: item.public_manifest_record() for item in self.corpus.repositories},
        )
        self.receipt_keys[id(ledger)] = evaluator.signer.public_key
        inner = DeterministicFixtureGenerator(
            {task.task_id: (source,)},
            public_locus={task.task_id: task.public_locus},
            model_digest=digest_for({"fixture": MODEL_REVISION}),
        )
        generator = _RecordingGenerator(inner)
        runner = BoundedCandidateLoop(
            ledger=ledger,
            evaluator=evaluator,
            generator=generator,
            isolation=ArmIsolation(self.root / "arms-{}-{}".format(arm, len(self.ledgers)), campaign_id=campaign),
            workspace_root=self.root / "workspace-{}-{}".format(arm, len(self.ledgers)),
            campaign_id=campaign,
            source_commit="training-test-source",
            model_revision=MODEL_REVISION,
            model_digest=inner.model_digest,
            data_manifest_digest=self.corpus.manifest_digest(),
            policy_digest=policy,
            arm_id=arm,
            max_attempts=1,
            seed_set=(seed,),
            fixture_mode=True,
        )
        report = runner.run(task, seed=seed)
        cutoff = LedgerCutoff.capture(ledger, campaign_id=campaign)
        private = PrivateTrajectoryAttempt(report.attempts[0].candidate_id, generator.contexts[0], source)
        return ledger, cutoff, private

    def builder(self, ledger, *, token_counter=None):
        return TrajectoryDatasetBuilder(
            ledger,
            self.corpus,
            token_counter=token_counter or self.token_count,
            receipt_public_key=self.receipt_keys[id(ledger)],
            require_complete=False,
        )

    def test_builds_private_digest_bound_row(self):
        ledger, cutoff, private = self.build_run()
        dataset = self.builder(ledger).build(cutoff, [private])
        self.assertEqual(len(dataset.examples), 1)
        row = dataset.examples[0]
        self.assertEqual(row.arm_id, "B")
        self.assertEqual(row.cutoff_digest, cutoff.digest)
        self.assertIn('"source":', row.target)
        private_record = row.private_record()
        self.assertEqual(private_record["sft_row"]["schema_version"], "egv-sft-row-v1")
        self.assertEqual(private_record["sft_row"]["diagnostic_enum"], "PASS")
        self.assertEqual(private_record["sft_row"]["promotion_disposition"], "PROMOTED")
        self.assertEqual(private_record["sft_row"]["candidate_artifact_digest"], row.source_digest)
        self.assertEqual(set(private_record["sft_row"]), SFT_FIELDS)
        manifest_text = str(dataset.manifest())
        self.assertNotIn(row.prompt, manifest_text)
        self.assertNotIn(row.target, manifest_text)

    def test_dataset_is_deterministic_and_rows_are_sorted(self):
        ledger, cutoff, private = self.build_run()
        first = self.builder(ledger).build(cutoff, [private])
        second = self.builder(ledger).build(cutoff, [private])
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.examples, second.examples)

    def test_rejects_source_digest_substitution(self):
        ledger, cutoff, private = self.build_run()
        forged = replace(private, candidate_source=b"def forged():\n    return True\n")
        with self.assertRaisesRegex(ValueError, "source digest"):
            self.builder(ledger).build(cutoff, [forged])

    def test_rejects_prompt_context_identity_substitution(self):
        ledger, cutoff, private = self.build_run()
        forged = replace(private, context=replace(private.context, seed=999))
        with self.assertRaisesRegex(ValueError, "authoritative run"):
            self.builder(ledger).build(cutoff, [forged])

    def test_complete_freeze_requires_exact_frozen_request_run_coordinates(self):
        ledger, cutoff, private = self.build_run()
        expected = {
            (private.context.run_id, private.context.task_id, private.context.arm_id, private.context.seed)
        }
        expected.update(
            ("frozen-run-{}".format(index), "frozen-task-{}".format(index), "B", index)
            for index in range(79)
        )
        builder = TrajectoryDatasetBuilder(
            ledger,
            self.corpus,
            token_counter=self.token_count,
            receipt_public_key=self.receipt_keys[id(ledger)],
            require_complete=True,
            expected_runs=expected,
        )
        with self.assertRaisesRegex(ValueError, "exact frozen commissioning requests"):
            builder.build(cutoff, [private])

    def test_rejects_prompt_retrieval_digest_substitution(self):
        ledger, cutoff, private = self.build_run()
        forged = replace(private, context=replace(private.context, retrieval_digest=digest_for("forged")))
        with self.assertRaisesRegex(ValueError, "retrieval digest"):
            self.builder(ledger).build(cutoff, [forged])

    def test_ledger_advance_after_cutoff_fails_closed(self):
        ledger, cutoff, private = self.build_run()
        ledger.append_event(
            "AUDIT_NOTE",
            {"note": "after cutoff"},
            campaign_id=cutoff.campaign_id,
            idempotency_key="after-cutoff",
        )
        with self.assertRaisesRegex(ValueError, "advanced"):
            self.builder(ledger).build(cutoff, [private])

    def test_missing_receipt_is_filtered(self):
        ledger, cutoff, private = self.build_run()
        builder = self.builder(ledger)
        actual = builder._receipts(private.candidate_id)
        actual.pop("EFFECT")
        with patch.object(builder, "_receipts", return_value=actual):
            dataset = builder.build(cutoff, [private])
        self.assertEqual(dataset.examples, ())
        self.assertEqual(dataset.excluded_counts["invalid_or_ineligible_attempt"], 1)

    def test_rejects_development_identity_leakage_in_target(self):
        leaked = self.corpus.split("dev")[0].template_id
        source = self.corpus.split("train")[0].corrected_source + ("\n# {}\n".format(leaked)).encode()
        ledger, cutoff, private = self.build_run(source=source)
        with self.assertRaisesRegex(ValueError, "development or held-out"):
            self.builder(ledger).build(cutoff, [private])

    def test_rejects_host_path_in_target(self):
        source = self.corpus.split("train")[0].corrected_source + b"\n# /home/operator/private\n"
        ledger, cutoff, private = self.build_run(source=source)
        with self.assertRaisesRegex(ValueError, "host-specific path"):
            self.builder(ledger).build(cutoff, [private])

    def test_rejects_overlength_instead_of_truncating(self):
        ledger, cutoff, private = self.build_run()
        calls = []

        def boundary_sensitive_counter(text):
            calls.append(text)
            return 4097 if "Return exactly one JSON object" in text and '\"source\"' in text else 1

        with self.assertRaisesRegex(ValueError, "4096-token"):
            self.builder(ledger, token_counter=boundary_sensitive_counter).build(cutoff, [private])
        self.assertEqual(len(calls), 1)

    def test_failed_rejected_candidate_is_never_a_positive_target(self):
        repo = self.corpus.split("train")[0]
        original = dict(repo.source_files)["src/task.py"]
        ledger, cutoff, private = self.build_run(source=original)
        dataset = self.builder(ledger).build(cutoff, [private])
        self.assertEqual(dataset.examples, ())
        self.assertEqual(dataset.excluded_counts["invalid_or_ineligible_attempt"], 1)

    def test_wrong_receipt_public_key_fails_signature_verification(self):
        ledger, cutoff, private = self.build_run()
        builder = TrajectoryDatasetBuilder(
            ledger,
            self.corpus,
            token_counter=self.token_count,
            receipt_public_key=ReceiptSigner.generate().public_key,
            require_complete=False,
        )
        with self.assertRaisesRegex(ValueError, "public key"):
            builder.build(cutoff, [private])

    def test_tampered_receipt_signature_fails_before_row_construction(self):
        ledger, cutoff, private = self.build_run()
        builder = self.builder(ledger)
        with patch.object(
            ledger,
            "verify_receipt_chain",
            side_effect=ReceiptVerificationError("invalid signature"),
        ):
            with self.assertRaisesRegex(Exception, "signature"):
                builder.build(cutoff, [private])

    def test_private_example_rejects_sft_envelope_downgrade(self):
        ledger, cutoff, private = self.build_run()
        row = self.builder(ledger).build(cutoff, [private]).examples[0]
        sft_row = row.private_record()["sft_row"]
        sft_row["diagnostic_enum"] = "WRONG_OUTPUT"
        forged_json = canonical_json(sft_row)
        with self.assertRaisesRegex(ValueError, "PASS/PROMOTED"):
            replace(
                row,
                sft_row_json=forged_json,
                sft_row_digest=digest_bytes(forged_json.encode("utf-8")),
            )

    def test_tampered_receipt_chain_cutoff_fails_closed(self):
        ledger, cutoff, private = self.build_run()
        forged = replace(cutoff, receipt_chain_head=digest_for("forged receipt head"))
        with self.assertRaisesRegex(ValueError, "receipt chain"):
            self.builder(ledger).build(forged, [private])

    def test_rejects_non_training_arm(self):
        ledger, cutoff, private = self.build_run(arm="C")
        with self.assertRaisesRegex(ValueError, r"outside the frozen B\+D"):
            self.builder(ledger).build(cutoff, [private])

    def test_private_attempts_must_be_unique(self):
        ledger, cutoff, private = self.build_run()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.builder(ledger).build(cutoff, [private, private])


if __name__ == "__main__":
    unittest.main()
