"""Adversarial and CPU-fixture tests for the EGV Training runtime."""

from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import io
import json
import sys
import unittest
from unittest.mock import patch

from egv.canonical import digest_for
from egv.cli import main
from egv.receipts import ReceiptSigner
from egv.training import (
    DevelopmentLossGateway,
    LoRATrainer,
    TrainingCheckpoint,
    TrainingConfigurationError,
    TrainingDataManifest,
    TrainingDependencyError,
    TrainingIntegrityError,
    TrainingLeakageError,
    TrainingProtocol,
    TrainingRow,
    build_training_batch,
    model_state_digest_for_training,
    seal_training_inputs,
    tokenize_training_row,
)
from egv.training import trainer as trainer_module
from egv.variation.model import (
    MODEL_ARCHITECTURE,
    MODEL_CONFIG_CLASS,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    LoadedPinnedModel,
    PinnedModelManifest,
)


class _Tokenizer:
    eos_token_id = 99

    def __call__(self, text, *, add_special_tokens, truncation):
        if add_special_tokens or truncation:
            raise AssertionError("mutable tokenizer options reached the fixture")
        return {"input_ids": [index + 1 for index, _ in enumerate(text.split())] or [1]}


class _Model:
    def __init__(self):
        self.steps = 0
        self.state = b"state-0"

    def state_dict(self):
        return {"weight": self.state}

    def train_step(self, _example):
        self.steps += 1
        self.state = "state-{}".format(self.steps).encode("ascii")
        return 1.0 / self.steps


class TrainingRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.protocol = TrainingProtocol()
        self.train_rows = (
            TrainingRow.create(
                task_id="egv-pure_function-train-1-v1",
                task_family="PURE_FUNCTION",
                split="train",
                prompt="train prompt one",
                target="train target one",
            ),
            TrainingRow.create(
                task_id="egv-pure_function-train-2-v1",
                task_family="PURE_FUNCTION",
                split="train",
                prompt="train prompt two",
                target="train target two",
            ),
        )
        self.dev_rows = (
            TrainingRow.create(
                task_id="egv-pure_function-dev-1-v1",
                task_family="PURE_FUNCTION",
                split="dev",
                prompt="private development prompt",
                target="private development target",
            ),
        )
        self.manifest = TrainingDataManifest.from_rows(self.train_rows, self.dev_rows)
        self.model = _Model()
        self.model_digest = digest_for("fixture-base-model")
        self.signer = ReceiptSigner(b"\x42" * 32)

        def evaluate(model, _rows, _protocol):
            return 1.0 / (model.steps + 1)

        self.gateway = DevelopmentLossGateway(
            self.dev_rows,
            data_manifest=self.manifest,
            model_digest=self.model_digest,
            protocol=self.protocol,
            signer=self.signer,
            evaluator=evaluate,
            production=False,
        )
        self.inputs = seal_training_inputs(
            self.model,
            model_digest=self.model_digest,
            train_rows=self.train_rows,
            data_manifest=self.manifest,
            protocol=self.protocol,
            fixture_only=True,
        )

    def _checkpoint(self, *, epoch=1, global_step=1, model_digest=None, data_digest=None, protocol_digest=None):
        state_digest = model_state_digest_for_training(self.model)
        return TrainingCheckpoint.create(
            epoch=epoch,
            global_step=global_step,
            model_digest=model_digest or self.model_digest,
            protocol_digest=protocol_digest or self.protocol.digest,
            data_manifest_digest=data_digest or self.manifest.digest,
            adapter_digest=digest_for({"adapter": epoch}),
            state_digest=state_digest,
            payload_digest=digest_for({"payload": epoch, "state": state_digest}),
        )

    def test_protocol_is_frozen_and_digest_changes_only_for_valid_default(self):
        self.protocol.validate()
        self.assertEqual(self.protocol.lora_rank, 16)
        self.assertEqual(self.protocol.lora_alpha, 32)
        self.assertEqual(self.protocol.lora_dropout, 0.05)
        self.assertEqual(self.protocol.learning_rate, 2e-4)
        self.assertEqual(self.protocol.max_epochs, 3)
        self.assertEqual(self.protocol.max_sequence_length, 4096)
        self.assertFalse(self.protocol.packing)
        self.assertEqual(self.protocol.gradient_accumulation_steps, 8)
        with self.assertRaises(TrainingConfigurationError):
            replace(self.protocol, learning_rate=1e-4).validate()
        with self.assertRaises(TrainingConfigurationError):
            TrainingProtocol.from_mapping({"schema_version": self.protocol.schema_version})

    def test_training_row_binds_immutable_evaluation_sft_record(self):
        source = {
            "schema_version": "egv-sft-row-v1",
            "row_id": "sft_train_row_1",
            "task_id": "egv-pure_function-train-1-v1",
            "task_family": "PURE_FUNCTION",
            "split": "train",
            "retrieved_evidence_ids": ["event_1"],
        }
        row = TrainingRow.from_sft_record(source, prompt="private prompt", target="private target")
        self.assertEqual(row.source_sft_row_digest, digest_for(source))
        self.assertEqual(row.source_event_ids, ("event_1",))
        with self.assertRaises((TrainingLeakageError, TrainingConfigurationError)):
            TrainingRow.from_sft_record(dict(source, split="dev"), prompt="p", target="t")

    def test_prompt_labels_are_masked_and_no_packing_or_truncation_occurs(self):
        row = self.train_rows[0]
        tokenized = tokenize_training_row(row, _Tokenizer(), self.protocol)
        prompt_length = 3
        self.assertEqual(tokenized.labels[:prompt_length], (-100, -100, -100))
        self.assertTrue(all(value != -100 for value in tokenized.labels[prompt_length:]))
        self.assertEqual(tokenized.input_ids[-1], 99)
        with self.assertRaises(TrainingConfigurationError):
            build_training_batch((self.train_rows[0], self.train_rows[1]), _Tokenizer(), self.protocol)

    def test_overlength_row_is_rejected_instead_of_truncated(self):
        row = TrainingRow.create(
            task_id="egv-pure_function-train-long-v1",
            task_family="PURE_FUNCTION",
            split="train",
            prompt=" ".join("token" for _ in range(4096)),
            target="one target",
        )
        with self.assertRaises(TrainingConfigurationError):
            tokenize_training_row(row, _Tokenizer(), self.protocol)

    def test_heldout_rows_and_split_overlap_are_rejected(self):
        heldout = TrainingRow(
            row_id="heldout-row",
            task_id="egv-pure_function-heldout-1-v1",
            task_family="PURE_FUNCTION",
            split="heldout",
            prompt="hidden",
            target="hidden",
        )
        with self.assertRaises((TrainingLeakageError, TrainingConfigurationError)):
            heldout.validate()
        with self.assertRaises((TrainingLeakageError, TrainingConfigurationError)):
            TrainingDataManifest.from_rows(self.train_rows, (replace(self.dev_rows[0], row_id=self.train_rows[0].row_id),))

    def test_development_gateway_signs_closed_receipt_without_content(self):
        checkpoint = self._checkpoint()
        evaluation = self.gateway.evaluate(checkpoint, model=self.model)
        self.gateway.verify_evaluation(
            evaluation,
            checkpoint=checkpoint,
            expected_model_digest=self.model_digest,
            expected_data_manifest_digest=self.manifest.digest,
            expected_protocol_digest=self.protocol.digest,
        )
        encoded = json.dumps(evaluation.to_dict(), sort_keys=True)
        self.assertNotIn("private development prompt", encoded)
        self.assertNotIn("private development target", encoded)
        self.assertNotIn("heldout", encoded.lower())
        self.assertEqual(evaluation.receipt["input_digest"], self.manifest.development_digest)
        self.assertEqual(evaluation.receipt["task_id"], "DEVELOPMENT_LOSS")

    def test_forged_loss_signature_and_stale_checkpoint_are_rejected(self):
        checkpoint = self._checkpoint()
        evaluation = self.gateway.evaluate(checkpoint, model=self.model)
        with self.assertRaises(TrainingIntegrityError):
            self.gateway.verify_evaluation(
                replace(evaluation, loss=evaluation.loss + 1.0),
                checkpoint=checkpoint,
                expected_model_digest=self.model_digest,
                expected_data_manifest_digest=self.manifest.digest,
                expected_protocol_digest=self.protocol.digest,
            )
        forged_receipt = dict(evaluation.receipt)
        forged_receipt["signature"] = "forged"
        with self.assertRaises(Exception):
            self.gateway.verify_evaluation(
                replace(evaluation, receipt=forged_receipt),
                checkpoint=checkpoint,
                expected_model_digest=self.model_digest,
                expected_data_manifest_digest=self.manifest.digest,
                expected_protocol_digest=self.protocol.digest,
            )
        with self.assertRaises(TrainingIntegrityError):
            self.gateway.evaluate(
                self._checkpoint(data_digest=digest_for("stale-data")),
                model=self.model,
            )

    def test_wrong_model_and_protocol_receipts_are_rejected(self):
        with self.assertRaises(TrainingIntegrityError):
            self.gateway.evaluate(self._checkpoint(model_digest=digest_for("other-model")), model=self.model)
        with self.assertRaises(TrainingIntegrityError):
            self.gateway.evaluate(self._checkpoint(protocol_digest=digest_for("old-protocol")), model=self.model)

    def test_lowest_valid_development_loss_is_selected(self):
        report = LoRATrainer(self.inputs, self.protocol, self.gateway, _Tokenizer(), production=False).run()
        self.assertEqual(report.selected_loss, min(item.loss for item in report.development_evaluations))
        self.assertEqual(report.selected_checkpoint_digest, report.development_evaluations[-1].checkpoint_digest)
        self.assertEqual(report.promotion_disposition, "NOT_PROMOTED_TRAINING_ARTIFACT")
        self.assertFalse(report.real_qwen_execution_claimed)

    def test_selection_report_has_no_development_or_heldout_content(self):
        report = LoRATrainer(self.inputs, self.protocol, self.gateway, _Tokenizer(), production=False).run()
        encoded = json.dumps(report.to_dict(), sort_keys=True)
        self.assertNotIn("private development prompt", encoded)
        self.assertNotIn("private development target", encoded)
        self.assertNotIn("heldout", encoded.lower())

    def test_missing_gateway_and_unsealed_inputs_fail_closed(self):
        with self.assertRaises(TrainingDependencyError):
            LoRATrainer(self.inputs, self.protocol, None, _Tokenizer(), production=False)
        unsealed = replace(self.inputs, _seal_token=None)
        with self.assertRaises(TrainingIntegrityError):
            LoRATrainer(unsealed, self.protocol, self.gateway, _Tokenizer(), production=False)

    def test_production_fixture_gateway_and_python_floor_fail_closed(self):
        with self.assertRaises(TrainingDependencyError):
            LoRATrainer(self.inputs, self.protocol, self.gateway, _Tokenizer(), production=True)

        production_gateway = DevelopmentLossGateway(
            self.dev_rows,
            data_manifest=self.manifest,
            model_digest=self.model_digest,
            protocol=self.protocol,
            signer=self.signer,
            evaluator=lambda _model, _rows, _protocol: 1.0,
            production=True,
        )
        production_inputs = seal_training_inputs(
            self.model,
            model_digest=self.model_digest,
            train_rows=self.train_rows,
            data_manifest=self.manifest,
            protocol=self.protocol,
            fixture_only=False,
        )
        trainer = LoRATrainer(production_inputs, self.protocol, production_gateway, _Tokenizer(), production=True)
        with patch.object(trainer_module.sys, "version_info", (3, 9)):
            with self.assertRaises(TrainingDependencyError):
                trainer.run()

    def test_production_missing_peft_is_not_replaced_by_fixture(self):
        manifest = PinnedModelManifest(
            repository=MODEL_REPOSITORY,
            revision=MODEL_REVISION,
            architecture=MODEL_ARCHITECTURE,
            config_class=MODEL_CONFIG_CLASS,
            transformers_version="5.5.0",
            files={"config.json": digest_for("config")},
            license={"name": "test-license", "source": "local"},
        )
        loaded = LoadedPinnedModel(
            model=self.model,
            tokenizer=_Tokenizer(),
            manifest=manifest,
            manifest_digest=manifest.digest(),
            file_hashes={"config.json": digest_for("config")},
            load_report={},
            base_state_digest=model_state_digest_for_training(self.model),
        )
        production_inputs = seal_training_inputs(
            loaded,
            model_digest=manifest.digest(),
            train_rows=self.train_rows,
            data_manifest=self.manifest,
            protocol=self.protocol,
            fixture_only=False,
        )
        gateway = DevelopmentLossGateway(
            self.dev_rows,
            data_manifest=self.manifest,
            model_digest=manifest.digest(),
            protocol=self.protocol,
            signer=self.signer,
            evaluator=lambda _model, _rows, _protocol: 1.0,
            production=True,
        )
        trainer = LoRATrainer(production_inputs, self.protocol, gateway, _Tokenizer(), production=True)
        with patch.dict(sys.modules, {"peft": None}):
            with self.assertRaises(TrainingDependencyError):
                trainer.run()

    def test_cli_training_smoke_and_train_lora_contract(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["training", "smoke", "--json"]), 0)
        smoke = json.loads(output.getvalue())
        self.assertEqual(smoke["smoke"], "PASS")
        self.assertFalse(smoke["real_qwen_execution_claimed"])
        errors = io.StringIO()
        with redirect_stderr(errors):
            self.assertEqual(
                main(
                    [
                        "train-lora",
                        "--model-root",
                        "/sealed/model",
                        "--train-manifest",
                        "/sealed/train.json",
                        "--development-manifest",
                        "/private/dev.json",
                    ]
                ),
                1,
            )
        self.assertIn("fail closed", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
