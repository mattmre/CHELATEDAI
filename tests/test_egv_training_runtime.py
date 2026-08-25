"""Adversarial and CPU-fixture tests for the EGV Training runtime."""

from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
import base64
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import egv.training.trainer as training_trainer
from egv.canonical import GENESIS_HASH, canonical_json, content_id, digest_for
from egv.cli import main
from egv.evaluation.dataset import EvaluationCorpus
from egv.receipts import ReceiptSigner, key_id_for_public_key
from egv.training import (
    DevelopmentLossGateway,
    DevelopmentLossEvaluation,
    ExternalDevelopmentLossGateway,
    LoRATrainer,
    TrainingCheckpoint,
    TrainingConfigurationError,
    TrainingDataManifest,
    TrainingDependencyError,
    TrainingIntegrityError,
    TrainingLeakageError,
    TrainingProtocol,
    TrainingRow,
    build_lora_change_attestation,
    build_lora_sealed_binding,
    build_training_batch,
    build_private_development_runtime,
    model_state_digest_for_training,
    seal_training_inputs,
    tokenize_training_row,
    run_production_training,
    receive_external_adapter,
)
from egv.training.contracts import FrozenTrainingDataset, LedgerCutoff
from egv.training.dataset import seal_runtime_dataset
from egv.training.development import _decode_external_signature
from egv.training.targets import (
    FROZEN_LORA_TARGETS,
    LORA_TARGET_MANIFEST_SCHEMA,
    LoraTargetManifest,
)
from egv.training.trainer import _restore_selected_lora_state
from egv.variation.adapter import ADAPTER_MANIFEST_NAME, SealedAdapterArtifact, build_local_adapter_manifest
from egv.variation.model import (
    MODEL_ARCHITECTURE,
    MODEL_CONFIG_CLASS,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    LoadedPinnedModel,
    PinnedModelManifest,
)


requires_production_python = unittest.skipIf(
    sys.version_info < (3, 10),
    "post-floor production boundary requires Python >=3.10",
)


class ProductionPythonFloorTests(unittest.TestCase):
    def test_production_training_rejects_python_39_before_reading_artifacts(self):
        with (
            patch("egv.training.trainer.sys.version_info", (3, 9, 0)),
            self.assertRaisesRegex(TrainingDependencyError, "Python >=3.10"),
        ):
            run_production_training(
                model_root=Path("missing-model"), training_dataset=Path("missing-train"),
                evaluator_manifest=Path("missing-service"), evaluator_public_key=Path("missing-key"),
                evaluator_command=Path("missing-command"),
                evaluator_transfer_command=Path("missing-transfer-command"),
                output_root=Path("missing-output"),
                expected_training_artifact_sha256=digest_for("artifact"),
                expected_training_dataset_digest=digest_for("dataset"), device="cuda",
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


class _AdapterStagingModel(_Model):
    def save_pretrained(self, root, *, safe_serialization):
        if not safe_serialization:
            raise AssertionError("adapter staging must use safetensors")
        destination = Path(root)
        (destination / "adapter_config.json").write_text("{}", encoding="utf-8")
        (destination / "adapter_model.safetensors").write_bytes(b"sealed-fixture")


def _frozen_target_manifest(*, model_digest=None) -> LoraTargetManifest:
    unsigned = {
        "schema_version": LORA_TARGET_MANIFEST_SCHEMA,
        "model_manifest_digest": model_digest or digest_for("attestation-model"),
        "architecture": MODEL_ARCHITECTURE,
        "config_class": MODEL_CONFIG_CLASS,
        "module_names": list(FROZEN_LORA_TARGETS),
        "module_types": ["torch.nn.Linear"] * len(FROZEN_LORA_TARGETS),
    }
    return LoraTargetManifest(
        schema_version=LORA_TARGET_MANIFEST_SCHEMA,
        model_manifest_digest=unsigned["model_manifest_digest"],
        architecture=MODEL_ARCHITECTURE,
        config_class=MODEL_CONFIG_CLASS,
        module_names=FROZEN_LORA_TARGETS,
        module_types=tuple(unsigned["module_types"]),
        digest=digest_for(unsigned),
    )


def _initial_lora_state():
    import torch

    state = {}
    for target_name in FROZEN_LORA_TARGETS:
        prefix = "base_model.model." + target_name
        state[prefix + ".lora_A.default.weight"] = torch.ones(
            (16, 3), dtype=torch.float32
        )
        state[prefix + ".lora_B.default.weight"] = torch.zeros(
            (4, 16), dtype=torch.float32
        )
    return state


def _clone_lora_state(state):
    return {name: value.clone() for name, value in state.items()}


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

    def _attach_selected_development_evidence(self, trainer):
        checkpoint = self._checkpoint()
        evaluation = self.gateway.evaluate(checkpoint, model=self.model)
        evaluation_snapshot = canonical_json(evaluation.to_dict())
        checkpoint_snapshot = canonical_json(checkpoint.to_dict())
        object.__setattr__(trainer, "inputs", self.inputs)
        object.__setattr__(trainer, "gateway", self.gateway)
        object.__setattr__(trainer, "_selected_evaluation", evaluation)
        object.__setattr__(trainer, "_selected_evaluation_snapshot", evaluation_snapshot)
        object.__setattr__(
            trainer,
            "_selected_evaluation_snapshot_digest",
            digest_for(json.loads(evaluation_snapshot)),
        )
        object.__setattr__(trainer, "_selected_checkpoint_snapshot", checkpoint_snapshot)
        object.__setattr__(
            trainer,
            "_selected_checkpoint_snapshot_digest",
            digest_for(json.loads(checkpoint_snapshot)),
        )
        object.__setattr__(
            trainer, "_selected_checkpoint_artifact_digest", evaluation.checkpoint_digest
        )
        return evaluation

    @staticmethod
    def _external_scratch(root):
        tree = training_trainer._create_private_training_tree(Path(root) / "gateway-output")
        scratch_root = tree.root / "evaluator-scratch"
        scratch_root.mkdir()
        training_trainer._verify_private_training_tree(tree)

        def validate_scratch():
            training_trainer._verify_private_training_tree(tree)

        return tree, {
            "scratch_root": scratch_root,
            "scratch_validator": validate_scratch,
        }

    def _external_verification_fixture(self, root):
        root = Path(root)
        command = root / "verify-evaluator.py"
        command.write_text("print('{}')\n", encoding="utf-8")
        public_key = root / "verify-evaluator.pub"
        public_key.write_bytes(self.signer.public_key_raw)
        unsigned = {
            "schema_version": "egv-external-development-service-v1",
            "campaign_id": "campaign-verify",
            "evaluator_key_id": self.signer.key_id,
            "evaluator_public_key_digest": hashlib.sha256(
                self.signer.public_key_raw
            ).hexdigest(),
            "endpoint_digest": hashlib.sha256(command.read_bytes()).hexdigest(),
            "transfer_endpoint_digest": hashlib.sha256(command.read_bytes()).hexdigest(),
            "development_manifest_digest": self.manifest.development_digest,
            "development_task_count": 8,
            "development_row_ids": ["dev-{:02d}".format(index) for index in range(8)],
            "model_digest": self.model_digest,
            "protocol_digest": self.protocol.digest,
        }
        service = {**unsigned, "service_manifest_digest": digest_for(unsigned)}
        service_path = root / "verify-service.json"
        service_path.write_text(canonical_json(service), encoding="utf-8")
        tree, scratch = self._external_scratch(root)
        gateway = ExternalDevelopmentLossGateway(
            service_path,
            public_key_path=public_key,
            command=command,
            transfer_command=command,
            **scratch,
        )
        checkpoint = self._checkpoint()
        checkpoint_artifact_digest = digest_for("verify-foundation-checkpoint")
        candidate_adapter_digest = digest_for("verify-candidate-adapter")
        loss = 0.25
        sample_count = 8
        output_digest = digest_for({
            "checkpoint_digest": checkpoint_artifact_digest,
            "adapter_digest": candidate_adapter_digest,
            "loss": loss,
            "sample_count": sample_count,
        })
        receipt_fields = {
            "receipt_type": "VERDICT",
            "campaign_id": service["campaign_id"],
            "run_id": "development-loss",
            "task_id": "DEVELOPMENT_LOSS",
            "request_id": content_id(
                "development-request", checkpoint_artifact_digest
            ),
            "candidate_id": content_id(
                "development-adapter", candidate_adapter_digest
            ),
            "candidate_artifact_digest": candidate_adapter_digest,
            "protocol_digest": self.protocol.digest,
            "evaluator_digest": service["service_manifest_digest"],
            "decision": "PASS",
            "diagnostic_enum": "PASS",
            "resource_bucket": "UNDER_25",
            "exit_status_class": "SUCCESS",
            "input_digest": self.manifest.development_digest,
            "output_digest": output_digest,
            "effect_kind": "DEVELOPMENT_LOSS",
        }
        receipt = self.signer.sign_receipt(
            receipt_fields,
            sequence=1,
            previous_receipt_hash=GENESIS_HASH,
            idempotency_key=content_id(
                "development-idempotency", checkpoint_artifact_digest
            ),
        )
        evaluation = DevelopmentLossEvaluation(
            checkpoint_digest=checkpoint_artifact_digest,
            checkpoint_id=checkpoint.checkpoint_id,
            model_digest=self.model_digest,
            data_manifest_digest=self.manifest.digest,
            development_manifest_digest=self.manifest.development_digest,
            protocol_digest=self.protocol.digest,
            gateway_digest=service["service_manifest_digest"],
            loss=loss,
            sample_count=sample_count,
            receipt=receipt,
        )
        return SimpleNamespace(
            tree=tree,
            scratch_root=scratch["scratch_root"],
            gateway=gateway,
            service_path=service_path,
            public_key_path=public_key,
            command_path=command,
            checkpoint=checkpoint,
            checkpoint_artifact_digest=checkpoint_artifact_digest,
            candidate_adapter_digest=candidate_adapter_digest,
            receipt_fields=receipt_fields,
            evaluation=evaluation,
        )

    def _direct_production_contract(self, tree, gateway):
        inputs = seal_training_inputs(
            self.model,
            model_digest=self.model_digest,
            train_rows=self.train_rows,
            data_manifest=self.manifest,
            protocol=self.protocol,
            fixture_only=False,
        )
        frozen_dataset = FrozenTrainingDataset(
            LedgerCutoff(
                "direct-production",
                1,
                "event",
                digest_for("direct-production-event"),
                digest_for("direct-production-receipts"),
                1,
                self.signer.key_id,
            ),
            (),
            {},
        )
        return {
            "inputs": inputs,
            "protocol": self.protocol,
            "gateway": gateway,
            "tokenizer": _Tokenizer(),
            "production": True,
            "frozen_dataset": frozen_dataset,
            "target_manifest": _frozen_target_manifest(model_digest=self.model_digest),
            "checkpoint_root": tree.root / "checkpoints",
            "private_training_tree": tree,
        }

    def test_private_development_builder_and_cli_freeze_exact_eight_rows(self):
        with tempfile.TemporaryDirectory(prefix="egv-dev-freeze-") as temporary:
            root = Path(temporary)
            seed = root / "seed.bin"
            seed.write_bytes(b"D" * 32)
            corpus = EvaluationCorpus.generate(secret_seed_file=seed)
            runtime = build_private_development_runtime(corpus)
            self.assertEqual(len(runtime["rows"]), 8)
            self.assertEqual({"dev"}, {corpus.get(row["task_id"]).split for row in runtime["rows"]})
            public_key = root / "evaluator.pub"
            public_key.write_bytes(self.signer.public_key_raw)
            command = root / "evaluator.py"
            command.write_text("print('sealed evaluator')\n", encoding="utf-8")
            private_output = root / "private-development.json"
            service_output = root / "service.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main([
                    "training", "freeze-evaluator-service",
                    "--campaign-id", "campaign-dev-freeze",
                    "--model-digest", digest_for("model"),
                    "--evaluator-seed", str(seed),
                    "--public-key", str(public_key),
                    "--command", str(command),
                    "--transfer-command", str(command),
                    "--private-output", str(private_output),
                    "--service-output", str(service_output),
                ])
            self.assertEqual(code, 0)
            private_value = json.loads(private_output.read_text(encoding="utf-8"))
            service_value = json.loads(service_output.read_text(encoding="utf-8"))
            self.assertEqual(private_value, runtime)
            self.assertEqual(service_value["development_task_count"], 8)
            self.assertEqual(service_value["development_row_ids"], sorted(row["row_id"] for row in runtime["rows"]))
            serialized_service = canonical_json(service_value)
            self.assertNotIn(str(seed), serialized_service)
            self.assertNotIn("prompt", serialized_service)
            self.assertNotIn("target", serialized_service)
            self.assertNotIn("heldout", canonical_json(private_value).lower())
            aliased_output = root / "aliased-private-public.json"
            errors = io.StringIO()
            with redirect_stderr(errors):
                code = main([
                    "training", "freeze-evaluator-service",
                    "--campaign-id", "campaign-dev-freeze",
                    "--model-digest", digest_for("model"),
                    "--evaluator-seed", str(seed),
                    "--public-key", str(public_key),
                    "--command", str(command),
                    "--transfer-command", str(command),
                    "--private-output", str(aliased_output),
                    "--service-output", str(aliased_output),
                ])
            self.assertEqual(code, 1)
            self.assertFalse(aliased_output.exists())

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
        self.assertEqual(self.protocol.device, "cuda")
        self.assertEqual(self.protocol.precision, "bfloat16")
        self.assertEqual(self.protocol.trainable_adapter_precision, "float32")
        self.assertEqual(self.protocol.optimizer_state_precision, "float32")
        with self.assertRaises(TrainingConfigurationError):
            replace(self.protocol, learning_rate=1e-4).validate()
        with self.assertRaises(TrainingConfigurationError):
            TrainingProtocol.from_mapping({"schema_version": self.protocol.schema_version})

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
        with self.assertRaises(TrainingDependencyError):
            LoRATrainer(production_inputs, self.protocol, production_gateway, _Tokenizer(), production=True)

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
        with self.assertRaises(TrainingDependencyError):
            LoRATrainer(production_inputs, self.protocol, gateway, _Tokenizer(), production=True)

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
                        "--sealed-training-artifact-sha256",
                        digest_for("artifact"),
                        "--sealed-training-dataset-digest",
                        digest_for("dataset"),
                        "--development-manifest",
                        "/private/dev.json",
                    ]
                ),
                1,
            )
        self.assertIn("fail closed", errors.getvalue())
        errors = io.StringIO()
        with redirect_stderr(errors), self.assertRaises(SystemExit) as raised:
            main(
                [
                    "train-lora",
                    "--model-root", "/sealed/model",
                    "--train-manifest", "/sealed/train.json",
                    "--development-manifest", "/private/dev.json",
                    "--evaluator-public-key", "/private/evaluator.pub",
                    "--evaluator-command", "/private/evaluate",
                    "--evaluator-transfer-command", "/private/transfer",
                    "--output", "/new/training-output",
                ]
            )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--sealed-training-artifact-sha256", errors.getvalue())
        self.assertIn("--sealed-training-dataset-digest", errors.getvalue())

    def test_external_evaluator_manifest_rejects_endpoint_and_key_substitution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = root / "evaluator"
            command.write_bytes(b"#!/bin/sh\nexit 0\n")
            public_key = root / "evaluator.pub"
            public_key.write_bytes(self.signer.public_key_raw)
            unsigned = {
                "schema_version": "egv-external-development-service-v1",
                "campaign_id": "campaign",
                "evaluator_key_id": key_id_for_public_key(self.signer.public_key_raw),
                "evaluator_public_key_digest": hashlib.sha256(self.signer.public_key_raw).hexdigest(),
                "endpoint_digest": hashlib.sha256(command.read_bytes()).hexdigest(),
                "transfer_endpoint_digest": hashlib.sha256(command.read_bytes()).hexdigest(),
                "development_manifest_digest": self.manifest.development_digest,
                "development_task_count": 8,
                "development_row_ids": ["dev-{:02d}".format(index) for index in range(8)],
                "model_digest": self.model_digest,
                "protocol_digest": self.protocol.digest,
            }
            manifest = {**unsigned, "service_manifest_digest": digest_for(unsigned)}
            manifest_path = root / "service.json"
            manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
            _scratch_tree, scratch = self._external_scratch(root)
            gateway = ExternalDevelopmentLossGateway(
                manifest_path, public_key_path=public_key, command=command,
                transfer_command=command, **scratch,
            )
            gateway.validate_production_boundary(
                expected_model_digest=self.model_digest, expected_protocol_digest=self.protocol.digest
            )
            with self.assertRaises(AttributeError):
                gateway._command_bytes = b"mutable endpoint"
            self.assertFalse(hasattr(gateway, "__dict__"))
            with self.assertRaises(TypeError):
                gateway._manifest["model_digest"] = digest_for("substituted-model")
            gateway.validate_production_boundary(
                expected_model_digest=self.model_digest, expected_protocol_digest=self.protocol.digest
            )
            original_invoke = ExternalDevelopmentLossGateway._invoke_pinned
            try:
                ExternalDevelopmentLossGateway._invoke_pinned = lambda self, **_kwargs: {}
                with self.assertRaises(TrainingDependencyError):
                    gateway.validate_production_boundary(
                        expected_model_digest=self.model_digest, expected_protocol_digest=self.protocol.digest
                    )
            finally:
                ExternalDevelopmentLossGateway._invoke_pinned = original_invoke
            gateway.validate_production_boundary(
                expected_model_digest=self.model_digest, expected_protocol_digest=self.protocol.digest
            )
            command.write_bytes(b"#!/bin/sh\nexit 9\n")
            with self.assertRaises(TrainingIntegrityError):
                ExternalDevelopmentLossGateway(
                    manifest_path, public_key_path=public_key, command=command,
                    transfer_command=command, **scratch,
                )
            command.write_bytes(b"#!/bin/sh\nexit 0\n")
            public_key.write_bytes(b"forged")
            with self.assertRaises(TrainingIntegrityError):
                ExternalDevelopmentLossGateway(
                    manifest_path, public_key_path=public_key, command=command,
                    transfer_command=command, **scratch,
                )

    def test_external_signed_envelope_rejects_noncanonical_ed25519_alias(self) -> None:
        encoded = self.signer.sign_bytes(b"external-envelope")
        self.assertEqual(len(_decode_external_signature(encoded)), 64)
        tail_alias = {"A": "B", "Q": "R", "g": "h", "w": "x"}
        with self.assertRaisesRegex(TrainingIntegrityError, "canonical"):
            _decode_external_signature(encoded[:-1] + tail_alias[encoded[-1]])

    @requires_production_python
    def test_direct_production_constructor_binds_exact_private_evaluator_scratch(self):
        with tempfile.TemporaryDirectory(prefix="egv-direct-production-boundary-") as temporary:
            root = Path(temporary)
            resources = root / "resources"
            resources.mkdir()
            source = self._external_verification_fixture(resources)
            intended_tree = training_trainer._create_private_training_tree(
                root / "intended-output"
            )

            with (
                patch(
                    "egv.training.development.subprocess.run",
                    side_effect=AssertionError("constructor must not invoke evaluator"),
                ),
                self.assertRaisesRegex(TrainingIntegrityError, "exact private staging descendant"),
            ):
                LoRATrainer(**self._direct_production_contract(intended_tree, source.gateway))
            self.assertEqual(list(source.scratch_root.iterdir()), [])
            self.assertFalse((intended_tree.root / "checkpoints").exists())

            outside_scratch = root / "outside-scratch"
            outside_scratch.mkdir()
            outside_gateway = ExternalDevelopmentLossGateway(
                source.service_path,
                public_key_path=source.public_key_path,
                command=source.command_path,
                transfer_command=source.command_path,
                scratch_root=outside_scratch,
                scratch_validator=lambda: None,
            )
            with (
                patch(
                    "egv.training.development.subprocess.run",
                    side_effect=AssertionError("constructor must not invoke evaluator"),
                ),
                self.assertRaisesRegex(TrainingIntegrityError, "exact private staging descendant"),
            ):
                LoRATrainer(
                    **self._direct_production_contract(intended_tree, outside_gateway)
                )
            self.assertEqual(list(outside_scratch.iterdir()), [])

            intended_scratch = intended_tree.root / "evaluator-scratch"
            intended_scratch.mkdir()

            def validate_intended_tree():
                training_trainer._verify_private_training_tree(intended_tree)

            intended_gateway = ExternalDevelopmentLossGateway(
                source.service_path,
                public_key_path=source.public_key_path,
                command=source.command_path,
                transfer_command=source.command_path,
                scratch_root=intended_scratch,
                scratch_validator=validate_intended_tree,
            )
            with patch(
                "egv.training.development.subprocess.run",
                side_effect=AssertionError("constructor must not invoke evaluator"),
            ):
                trainer = LoRATrainer(
                    **self._direct_production_contract(intended_tree, intended_gateway)
                )
            self.assertIs(trainer.gateway, intended_gateway)
            self.assertEqual(trainer._private_training_tree, intended_tree)
            self.assertEqual(list(intended_scratch.iterdir()), [])
            with patch.object(
                LoRATrainer, "_validate_production_model", autospec=True
            ) as validate_model:
                trainer._validate_run_boundary()
            validate_model.assert_called_once_with(trainer)

            retained_tree = root / "retained-intended-tree"
            intended_tree.root.replace(retained_tree)
            intended_tree.root.mkdir()
            root_marker = intended_tree.root / "foreign-root-marker"
            root_marker.write_text("foreign-root", encoding="utf-8")
            with (
                patch.object(
                    LoRATrainer,
                    "_validate_production_model",
                    side_effect=AssertionError("substitution must fail before model access"),
                ),
                self.assertRaisesRegex(TrainingIntegrityError, "identity changed"),
            ):
                trainer._validate_run_boundary()
            self.assertEqual(root_marker.read_text(encoding="utf-8"), "foreign-root")

            substituted_tree = training_trainer._create_private_training_tree(
                root / "substituted-output"
            )
            substituted_scratch = substituted_tree.root / "evaluator-scratch"
            substituted_scratch.mkdir()

            def validate_substituted_tree():
                training_trainer._verify_private_training_tree(substituted_tree)

            substituted_gateway = ExternalDevelopmentLossGateway(
                source.service_path,
                public_key_path=source.public_key_path,
                command=source.command_path,
                transfer_command=source.command_path,
                scratch_root=substituted_scratch,
                scratch_validator=validate_substituted_tree,
            )
            retained = substituted_tree.root / "retained-evaluator-scratch"
            substituted_scratch.replace(retained)
            substituted_scratch.mkdir()
            marker = substituted_scratch / "foreign-marker"
            marker.write_text("foreign", encoding="utf-8")
            with (
                patch(
                    "egv.training.development.subprocess.run",
                    side_effect=AssertionError("constructor must not invoke evaluator"),
                ),
                self.assertRaisesRegex(TrainingIntegrityError, "identity differs"),
            ):
                LoRATrainer(
                    **self._direct_production_contract(
                        substituted_tree, substituted_gateway
                    )
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "foreign")
            self.assertFalse((substituted_tree.root / "checkpoints").exists())

    def test_external_evaluation_exact_checkpoint_and_receipt_path_passes(self):
        with tempfile.TemporaryDirectory(prefix="egv-external-verify-positive-") as temporary:
            fixture = self._external_verification_fixture(temporary)
            fixture.gateway.verify_evaluation(
                fixture.evaluation,
                checkpoint=fixture.checkpoint,
                expected_checkpoint_artifact_digest=fixture.checkpoint_artifact_digest,
                expected_candidate_adapter_digest=fixture.candidate_adapter_digest,
                expected_model_digest=self.model_digest,
                expected_data_manifest_digest=self.manifest.digest,
                expected_protocol_digest=self.protocol.digest,
            )

    def test_external_evaluation_rejects_each_frozen_binding_mismatch(self):
        with tempfile.TemporaryDirectory(prefix="egv-external-verify-adversarial-") as temporary:
            fixture = self._external_verification_fixture(temporary)

            def verify(evaluation=None, checkpoint=None, **overrides):
                arguments = {
                    "checkpoint": checkpoint or fixture.checkpoint,
                    "expected_checkpoint_artifact_digest": fixture.checkpoint_artifact_digest,
                    "expected_candidate_adapter_digest": fixture.candidate_adapter_digest,
                    "expected_model_digest": self.model_digest,
                    "expected_data_manifest_digest": self.manifest.digest,
                    "expected_protocol_digest": self.protocol.digest,
                }
                arguments.update(overrides)
                fixture.gateway.verify_evaluation(
                    evaluation or fixture.evaluation,
                    **arguments,
                )

            outer_mismatches = {
                "checkpoint_id": replace(fixture.evaluation, checkpoint_id="foreign-checkpoint"),
                "checkpoint_artifact_digest": replace(
                    fixture.evaluation, checkpoint_digest=digest_for("foreign-artifact")
                ),
                "evaluation_model": replace(
                    fixture.evaluation, model_digest=digest_for("foreign-model")
                ),
                "evaluation_data": replace(
                    fixture.evaluation,
                    data_manifest_digest=digest_for("foreign-data"),
                ),
                "evaluation_protocol": replace(
                    fixture.evaluation, protocol_digest=digest_for("foreign-protocol")
                ),
                "gateway": replace(
                    fixture.evaluation, gateway_digest=digest_for("foreign-gateway")
                ),
                "development": replace(
                    fixture.evaluation,
                    development_manifest_digest=digest_for("foreign-development"),
                ),
                "negative_loss": replace(fixture.evaluation, loss=-0.1),
                "nonfinite_loss": replace(fixture.evaluation, loss=float("nan")),
                "loss_output_binding": replace(
                    fixture.evaluation, loss=fixture.evaluation.loss + 0.1
                ),
                "sample_count": replace(fixture.evaluation, sample_count=7),
            }
            for label, evaluation in outer_mismatches.items():
                with self.subTest(binding=label), self.assertRaises(TrainingIntegrityError):
                    verify(evaluation=evaluation)

            checkpoint_mismatches = {
                "checkpoint_model": self._checkpoint(
                    model_digest=digest_for("foreign-checkpoint-model")
                ),
                "checkpoint_data": self._checkpoint(
                    data_digest=digest_for("foreign-checkpoint-data")
                ),
                "checkpoint_protocol": self._checkpoint(
                    protocol_digest=digest_for("foreign-checkpoint-protocol")
                ),
            }
            for label, checkpoint in checkpoint_mismatches.items():
                with self.subTest(binding=label), self.assertRaises(TrainingIntegrityError):
                    verify(checkpoint=checkpoint)

            with self.subTest(binding="expected_checkpoint_digest"), self.assertRaises(
                TrainingIntegrityError
            ):
                verify(
                    expected_checkpoint_artifact_digest=digest_for(
                        "wrong-expected-checkpoint"
                    )
                )
            with self.subTest(binding="expected_candidate_digest"), self.assertRaises(
                TrainingIntegrityError
            ):
                verify(
                    expected_candidate_adapter_digest=digest_for(
                        "wrong-expected-adapter"
                    )
                )

            def sign(fields=None, *, signer=None, sequence=1, previous=GENESIS_HASH,
                     idempotency_key=None):
                return (signer or self.signer).sign_receipt(
                    fields or fixture.receipt_fields,
                    sequence=sequence,
                    previous_receipt_hash=previous,
                    idempotency_key=(
                        idempotency_key
                        or content_id(
                            "development-idempotency",
                            fixture.checkpoint_artifact_digest,
                        )
                    ),
                )

            semantic_mutations = {
                "campaign_id": {"campaign_id": "foreign-campaign"},
                "run_id": {"run_id": "foreign-run"},
                "task_id": {"task_id": "FOREIGN_LOSS"},
                "request_id": {"request_id": "foreign-request"},
                "candidate_id": {"candidate_id": "foreign-candidate"},
                "protocol_digest": {"protocol_digest": digest_for("foreign-protocol")},
                "evaluator_digest": {"evaluator_digest": digest_for("foreign-evaluator")},
                "decision": {"decision": "FAIL"},
                "diagnostic_enum": {"diagnostic_enum": "FAIL"},
                "resource_bucket": {"resource_bucket": "OVER_25"},
                "exit_status_class": {"exit_status_class": "NONZERO"},
                "input_digest": {"input_digest": digest_for("foreign-input")},
                "effect_kind": {"effect_kind": "FOREIGN_LOSS"},
            }
            for label, changes in semantic_mutations.items():
                fields = {**fixture.receipt_fields, **changes}
                receipt = sign(fields)
                with self.subTest(receipt_field=label), self.assertRaises(
                    TrainingIntegrityError
                ):
                    verify(evaluation=replace(fixture.evaluation, receipt=receipt))

            foreign_candidate = digest_for("foreign-candidate-adapter")
            candidate_fields = {
                **fixture.receipt_fields,
                "candidate_id": content_id("development-adapter", foreign_candidate),
                "candidate_artifact_digest": foreign_candidate,
                "output_digest": digest_for({
                    "checkpoint_digest": fixture.checkpoint_artifact_digest,
                    "adapter_digest": foreign_candidate,
                    "loss": fixture.evaluation.loss,
                    "sample_count": fixture.evaluation.sample_count,
                }),
            }
            signed_receipt_cases = {
                "receipt_type": sign({
                    **fixture.receipt_fields,
                    "receipt_type": "AUTHORITY",
                    "decision": "ALLOW",
                }),
                "candidate_adapter_digest": sign(candidate_fields),
                "output_digest": sign({
                    **fixture.receipt_fields,
                    "output_digest": digest_for("foreign-output"),
                }),
                "sequence": sign(sequence=2),
                "previous_receipt_hash": sign(previous=digest_for("foreign-previous")),
                "idempotency_key": sign(idempotency_key="foreign-idempotency"),
                "extra_field": sign({
                    **fixture.receipt_fields,
                    "started_at": "2026-08-24T00:00:00Z",
                }),
            }
            for label, receipt in signed_receipt_cases.items():
                with self.subTest(receipt_field=label), self.assertRaises(
                    TrainingIntegrityError
                ):
                    verify(evaluation=replace(fixture.evaluation, receipt=receipt))

            invalid_schema = dict(fixture.evaluation.receipt)
            invalid_schema["schema_version"] = "egv-receipt-v0"
            invalid_signature = dict(fixture.evaluation.receipt)
            invalid_signature["signature"] = "forged"
            other_signer = ReceiptSigner.generate()
            invalid_key = sign(signer=other_signer)
            for label, receipt in {
                "schema": invalid_schema,
                "signature": invalid_signature,
                "signing_key": invalid_key,
            }.items():
                with self.subTest(receipt_field=label), self.assertRaises(
                    TrainingIntegrityError
                ):
                    verify(evaluation=replace(fixture.evaluation, receipt=receipt))

    def test_external_evaluator_scratch_is_private_retained_and_substitution_safe(self):
        with tempfile.TemporaryDirectory(prefix="egv-external-scratch-") as temporary:
            root = Path(temporary)
            command = root / "evaluator.py"
            transfer_command = root / "transfer.py"
            private_key_b64 = base64.b64encode(self.signer.private_key_raw).decode("ascii")
            repository_root = str(Path.cwd())
            command.write_text(
                "import base64,json,sys\n"
                "from pathlib import Path\n"
                "Path('cwd-marker').write_text('evaluate',encoding='utf-8')\n"
                "sys.path.insert(0,{!r})\n"
                "from egv.canonical import content_id,digest_for\n"
                "from egv.receipts import GENESIS_HASH,ReceiptSigner\n"
                "request=json.loads(sys.stdin.read())\n"
                "signer=ReceiptSigner(base64.b64decode({!r}))\n"
                "loss=0.25\n"
                "out=digest_for({{'checkpoint_digest':request['checkpoint_digest'],"
                "'adapter_digest':request['adapter_digest'],'loss':loss,'sample_count':8}})\n"
                "receipt=signer.sign_receipt({{'receipt_type':'VERDICT',"
                "'campaign_id':request['campaign_id'],'run_id':'development-loss',"
                "'task_id':'DEVELOPMENT_LOSS','request_id':content_id('development-request',"
                "request['checkpoint_digest']),'candidate_id':content_id('development-adapter',"
                "request['adapter_digest']),'candidate_artifact_digest':request['adapter_digest'],"
                "'protocol_digest':request['protocol_digest'],"
                "'evaluator_digest':request['service_manifest_digest'],'decision':'PASS',"
                "'diagnostic_enum':'PASS','resource_bucket':'UNDER_25',"
                "'exit_status_class':'SUCCESS','input_digest':request['development_manifest_digest'],"
                "'output_digest':out,'effect_kind':'DEVELOPMENT_LOSS'}},sequence=1,"
                "previous_receipt_hash=GENESIS_HASH,idempotency_key=content_id("
                "'development-idempotency',request['checkpoint_digest']))\n"
                "print(json.dumps({{'schema_version':'egv-development-loss-response-v1',"
                "'checkpoint_digest':request['checkpoint_digest'],"
                "'adapter_digest':request['adapter_digest'],'loss':loss,"
                "'sample_count':8,'receipt':receipt}}))\n".format(
                    repository_root,
                    private_key_b64,
                ),
                encoding="utf-8",
            )
            transfer_command.write_text(
                "import base64,json,sys\n"
                "from pathlib import Path\n"
                "Path('cwd-marker').write_text('transfer',encoding='utf-8')\n"
                "sys.path.insert(0,{!r})\n"
                "from egv.canonical import canonical_bytes\n"
                "from egv.receipts import ReceiptSigner\n"
                "request=json.loads(sys.stdin.read())\n"
                "signer=ReceiptSigner(base64.b64decode({!r}))\n"
                "unsigned={{'schema_version':'egv-adapter-transfer-response-v1',"
                "'service_manifest_digest':request['service_manifest_digest'],"
                "'request_digest':request['request_digest'],"
                "'adapter_digest':request['adapter_digest'],"
                "'adapter_reference':request['adapter_digest'],"
                "'signing_key_id':signer.key_id}}\n"
                "print(json.dumps({{**unsigned,'signature':signer.sign_bytes("
                "canonical_bytes(unsigned))}}))\n".format(
                    repository_root,
                    private_key_b64,
                ),
                encoding="utf-8",
            )
            public_key = root / "evaluator.pub"
            public_key.write_bytes(self.signer.public_key_raw)
            unsigned = {
                "schema_version": "egv-external-development-service-v1",
                "campaign_id": "campaign-scratch",
                "evaluator_key_id": self.signer.key_id,
                "evaluator_public_key_digest": hashlib.sha256(
                    self.signer.public_key_raw
                ).hexdigest(),
                "endpoint_digest": hashlib.sha256(command.read_bytes()).hexdigest(),
                "transfer_endpoint_digest": hashlib.sha256(
                    transfer_command.read_bytes()
                ).hexdigest(),
                "development_manifest_digest": self.manifest.development_digest,
                "development_task_count": 8,
                "development_row_ids": [
                    "dev-{:02d}".format(index) for index in range(8)
                ],
                "model_digest": self.model_digest,
                "protocol_digest": self.protocol.digest,
            }
            service_path = root / "service.json"
            service_path.write_text(
                canonical_json({
                    **unsigned,
                    "service_manifest_digest": digest_for(unsigned),
                }),
                encoding="utf-8",
            )
            tree, scratch = self._external_scratch(root)
            gateway = ExternalDevelopmentLossGateway(
                service_path,
                public_key_path=public_key,
                command=command,
                transfer_command=transfer_command,
                **scratch,
            )

            class RecordingAdapter(_AdapterStagingModel):
                def save_pretrained(self, output, *, safe_serialization):
                    self.saved_root = Path(output)
                    super().save_pretrained(output, safe_serialization=safe_serialization)

            model = RecordingAdapter()
            checkpoint = self._checkpoint()
            checkpoint_artifact_digest = digest_for("scratch-foundation-checkpoint")
            with patch(
                "egv.training.development.tempfile.TemporaryDirectory",
                side_effect=AssertionError("automatic scratch cleanup is forbidden"),
            ):
                evaluation = gateway.evaluate(
                    checkpoint,
                    model=model,
                    checkpoint_artifact_digest=checkpoint_artifact_digest,
                )
            self.assertEqual(model.saved_root.parent, scratch["scratch_root"])
            self.assertTrue(model.saved_root.is_dir())
            retained_directories = sorted(
                path for path in scratch["scratch_root"].iterdir() if path.is_dir()
            )
            self.assertEqual(len(retained_directories), 3)
            self.assertEqual(
                len(list(scratch["scratch_root"].rglob("cwd-marker"))),
                2,
            )
            gateway.verify_evaluation(
                evaluation,
                checkpoint=checkpoint,
                expected_checkpoint_artifact_digest=checkpoint_artifact_digest,
                expected_candidate_adapter_digest=evaluation.receipt[
                    "candidate_artifact_digest"
                ],
                expected_model_digest=self.model_digest,
                expected_data_manifest_digest=self.manifest.digest,
                expected_protocol_digest=self.protocol.digest,
            )

            retained_scratch = tree.root / "retained-evaluator-scratch"
            scratch["scratch_root"].replace(retained_scratch)
            scratch["scratch_root"].mkdir()
            foreign_marker = scratch["scratch_root"] / "foreign-marker"
            foreign_marker.write_text("foreign", encoding="utf-8")

            class NeverWriteAdapter(_AdapterStagingModel):
                save_reached = False

                def save_pretrained(self, output, *, safe_serialization):
                    self.save_reached = True
                    super().save_pretrained(output, safe_serialization=safe_serialization)

            never_write = NeverWriteAdapter()
            with self.assertRaisesRegex(TrainingIntegrityError, "scratch root identity changed"):
                gateway.evaluate(
                    checkpoint,
                    model=never_write,
                    checkpoint_artifact_digest=checkpoint_artifact_digest,
                )
            self.assertFalse(never_write.save_reached)
            self.assertEqual(foreign_marker.read_text(encoding="utf-8"), "foreign")

    def test_adapter_transfer_installs_content_addressed_tree_without_source_path(self):
        with tempfile.TemporaryDirectory(prefix="egv-adapter-transfer-") as temporary:
            root = Path(temporary)
            adapter_root = root / "source-adapter"
            adapter_root.mkdir()
            (adapter_root / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter_root / "adapter_model.safetensors").write_bytes(b"sealed-transfer")
            manifest = build_local_adapter_manifest(adapter_root)
            (adapter_root / ADAPTER_MANIFEST_NAME).write_text(
                canonical_json(manifest.to_dict()) + "\n", encoding="utf-8"
            )
            artifact = SealedAdapterArtifact(adapter_root)
            artifact.verify()
            command = root / "command.py"
            command.write_text("print('unused')\n", encoding="utf-8")
            public_key = root / "evaluator.pub"
            private_key = root / "evaluator.key"
            public_key.write_bytes(self.signer.public_key_raw)
            private_key.write_bytes(self.signer.private_key_raw)
            unsigned_service = {
                "schema_version": "egv-external-development-service-v1",
                "campaign_id": "campaign-transfer",
                "evaluator_key_id": self.signer.key_id,
                "evaluator_public_key_digest": hashlib.sha256(self.signer.public_key_raw).hexdigest(),
                "endpoint_digest": hashlib.sha256(command.read_bytes()).hexdigest(),
                "transfer_endpoint_digest": hashlib.sha256(command.read_bytes()).hexdigest(),
                "development_manifest_digest": digest_for("dev"),
                "development_task_count": 8,
                "development_row_ids": ["dev-{:02d}".format(index) for index in range(8)],
                "model_digest": self.model_digest,
                "protocol_digest": self.protocol.digest,
            }
            service = {**unsigned_service, "service_manifest_digest": digest_for(unsigned_service)}
            service_path = root / "service.json"
            service_path.write_text(canonical_json(service), encoding="utf-8")
            records = []
            for path in sorted(adapter_root.rglob("*")):
                if path.is_file():
                    payload = path.read_bytes()
                    records.append({
                        "path": path.relative_to(adapter_root).as_posix(),
                        "digest": hashlib.sha256(payload).hexdigest(),
                        "content_b64": base64.urlsafe_b64encode(payload).decode().rstrip("="),
                    })
            body = {
                "schema_version": "egv-adapter-transfer-request-v1",
                "service_manifest_digest": service["service_manifest_digest"],
                "adapter_digest": artifact.digest,
                "files": records,
            }
            request = {**body, "request_digest": digest_for(body)}
            response = receive_external_adapter(
                request,
                service_manifest=service_path,
                adapter_store=root / "spark2-store",
                evaluator_private_key=private_key,
            )
            self.assertEqual(response["adapter_reference"], artifact.digest)
            installed = SealedAdapterArtifact(root / "spark2-store" / artifact.digest)
            installed.verify()
            self.assertEqual(installed.digest, artifact.digest)
            self.assertNotIn(str(adapter_root), canonical_json(request))
    def test_external_evaluator_rehashes_endpoint_immediately_before_spawn(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = root / "evaluator"
            command.write_bytes(b"#!/bin/sh\nexit 0\n")
            public_key = root / "evaluator.pub"
            public_key.write_bytes(self.signer.public_key_raw)
            unsigned = {
                "schema_version": "egv-external-development-service-v1",
                "campaign_id": "campaign",
                "evaluator_key_id": key_id_for_public_key(self.signer.public_key_raw),
                "evaluator_public_key_digest": hashlib.sha256(self.signer.public_key_raw).hexdigest(),
                "endpoint_digest": hashlib.sha256(command.read_bytes()).hexdigest(),
                "transfer_endpoint_digest": hashlib.sha256(command.read_bytes()).hexdigest(),
                "development_manifest_digest": self.manifest.development_digest,
                "development_task_count": 8,
                "development_row_ids": ["dev-{:02d}".format(index) for index in range(8)],
                "model_digest": self.model_digest,
                "protocol_digest": self.protocol.digest,
            }
            manifest_path = root / "service.json"
            manifest_path.write_text(
                canonical_json({**unsigned, "service_manifest_digest": digest_for(unsigned)}), encoding="utf-8"
            )
            gateway = ExternalDevelopmentLossGateway(
                manifest_path, public_key_path=public_key, command=command,
                transfer_command=command, **self._external_scratch(root)[1],
            )
            command.write_bytes(b"#!/bin/sh\nexit 7\n")
            with self.assertRaisesRegex(TrainingIntegrityError, "frozen resources changed"):
                gateway.evaluate(
                    self._checkpoint(), model=_AdapterStagingModel(),
                    checkpoint_artifact_digest=digest_for("foundation-checkpoint"),
                )

    def test_production_training_rejects_cpu_before_reading_artifacts(self):
        with self.assertRaisesRegex(TrainingConfigurationError, "frozen to the CUDA"):
            run_production_training(
                model_root=Path("missing-model"), training_dataset=Path("missing-train"),
                evaluator_manifest=Path("missing-service"), evaluator_public_key=Path("missing-key"),
                evaluator_command=Path("missing-command"),
                evaluator_transfer_command=Path("missing-transfer-command"),
                output_root=Path("missing-output"),
                expected_training_artifact_sha256=digest_for("artifact"),
                expected_training_dataset_digest=digest_for("dataset"), device="cpu",
            )

    @requires_production_python
    def test_production_training_requires_a_new_output_root_before_input_read(self):
        with tempfile.TemporaryDirectory(prefix="egv-training-output-") as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            marker = output / "prior-adapter"
            marker.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(TrainingIntegrityError, "newly absent"):
                run_production_training(
                    model_root=root / "missing-model", training_dataset=root / "missing-train",
                    evaluator_manifest=root / "missing-service", evaluator_public_key=root / "missing-key",
                    evaluator_command=root / "missing-command",
                    evaluator_transfer_command=root / "missing-transfer-command",
                    output_root=output,
                    expected_training_artifact_sha256=digest_for("artifact"),
                    expected_training_dataset_digest=digest_for("dataset"), device="cuda",
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_private_training_publish_rejects_destination_claim_without_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="egv-training-publish-race-") as temporary:
            root = Path(temporary)
            destination = root / "output"
            tree = training_trainer._create_private_training_tree(destination)
            payload = tree.root / "completed.txt"
            payload.write_text("owned-completed-tree", encoding="utf-8")
            completed_entries = training_trainer._verify_private_training_tree(tree)
            move_noreplace = training_trainer._move_training_root_noreplace

            def claim_then_publish(source, target):
                Path(target).mkdir()
                (Path(target) / "foreign-marker").write_text("foreign", encoding="utf-8")
                return move_noreplace(source, target)

            with patch.object(
                training_trainer,
                "_move_training_root_noreplace",
                side_effect=claim_then_publish,
            ):
                with self.assertRaisesRegex(TrainingIntegrityError, "claimed before publication"):
                    training_trainer._publish_private_training_tree(
                        tree,
                        destination,
                        expected_entries=completed_entries,
                    )
            self.assertEqual(
                (destination / "foreign-marker").read_text(encoding="utf-8"), "foreign"
            )
            self.assertEqual(payload.read_text(encoding="utf-8"), "owned-completed-tree")

    def test_private_training_publish_moves_completed_tree_to_final_path(self):
        with tempfile.TemporaryDirectory(prefix="egv-training-publish-success-") as temporary:
            root = Path(temporary)
            destination = root / "output"
            tree = training_trainer._create_private_training_tree(destination)
            staging_root = tree.root
            (staging_root / "adapter").mkdir()
            payload = staging_root / "adapter" / "adapter_model.safetensors"
            payload.write_bytes(b"completed-adapter")
            completed_entries = training_trainer._verify_private_training_tree(tree)
            published = training_trainer._publish_private_training_tree(
                tree,
                destination,
                expected_entries=completed_entries,
            )
            self.assertEqual(published.root, destination)
            self.assertFalse(staging_root.exists())
            self.assertEqual(
                (destination / "adapter" / "adapter_model.safetensors").read_bytes(),
                b"completed-adapter",
            )

    def test_private_training_staging_substitution_fails_before_write_or_publish(self):
        with tempfile.TemporaryDirectory(prefix="egv-training-stage-swap-") as temporary:
            root = Path(temporary)
            destination = root / "output"
            tree = training_trainer._create_private_training_tree(destination)
            completed_entries = training_trainer._verify_private_training_tree(tree)
            retained = root / "retained-owned-staging"
            tree.root.replace(retained)
            tree.root.mkdir()
            marker = tree.root / "foreign-marker"
            marker.write_text("foreign", encoding="utf-8")
            writes = []

            def guarded_write():
                training_trainer._verify_private_training_tree(tree)
                writes.append(True)

            with self.assertRaisesRegex(TrainingIntegrityError, "identity changed"):
                guarded_write()
            with self.assertRaisesRegex(TrainingIntegrityError, "identity changed"):
                training_trainer._publish_private_training_tree(
                    tree,
                    destination,
                    expected_entries=completed_entries,
                )
            self.assertEqual(writes, [])
            self.assertEqual(marker.read_text(encoding="utf-8"), "foreign")
            self.assertFalse(destination.exists())

    @requires_production_python
    def test_genuine_zero_row_freezer_artifact_fails_before_model_or_output(self):
        with tempfile.TemporaryDirectory(prefix="egv-training-zero-") as temporary:
            root = Path(temporary)
            dataset = FrozenTrainingDataset(
                LedgerCutoff(
                    "zero-row-campaign", 1, "event", digest_for("zero-event"),
                    digest_for("zero-receipts"), 1, "evaluator-key",
                ),
                (),
                {"invalid_or_ineligible_attempt": 80},
            )
            artifact = root / "sealed-training.json"
            freezer_report = seal_runtime_dataset(dataset, artifact)
            output = root / "output"
            with (
                patch.object(Path, "read_bytes", side_effect=AssertionError("path reread")),
                patch.object(Path, "read_text", side_effect=AssertionError("path reread")),
                patch(
                    "egv.training.trainer._create_private_training_tree",
                    side_effect=AssertionError("private staging reached"),
                ),
                self.assertRaisesRegex(TrainingIntegrityError, "exactly 20"),
            ):
                run_production_training(
                    model_root=root / "missing-model", training_dataset=artifact,
                    evaluator_manifest=root / "missing-service", evaluator_public_key=root / "missing-key",
                    evaluator_command=root / "missing-command",
                    evaluator_transfer_command=root / "missing-transfer-command",
                    output_root=output,
                    expected_training_artifact_sha256=freezer_report["output_digest"],
                    expected_training_dataset_digest=freezer_report["dataset_digest"], device="cuda",
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".egv-training-private-*")), [])

    @requires_production_python
    def test_sealed_training_handoff_rejects_raw_and_semantic_substitution(self):
        with tempfile.TemporaryDirectory(prefix="egv-training-substitution-") as temporary:
            root = Path(temporary)

            def freeze(name):
                dataset = FrozenTrainingDataset(
                    LedgerCutoff(
                        name, 1, "event", digest_for({"event": name}),
                        digest_for({"receipts": name}), 1, "evaluator-key",
                    ),
                    (),
                    {},
                )
                path = root / "{}.json".format(name)
                return path, seal_runtime_dataset(dataset, path)

            _first_path, first = freeze("first")
            second_path, second = freeze("second")
            common = {
                "model_root": root / "missing-model",
                "training_dataset": second_path,
                "evaluator_manifest": root / "missing-service",
                "evaluator_public_key": root / "missing-key",
                "evaluator_command": root / "missing-command",
                "evaluator_transfer_command": root / "missing-transfer-command",
                "device": "cuda",
            }
            with self.assertRaisesRegex(TrainingIntegrityError, "artifact SHA-256"):
                run_production_training(
                    **common, output_root=root / "raw-output",
                    expected_training_artifact_sha256=first["output_digest"],
                    expected_training_dataset_digest=first["dataset_digest"],
                )
            with self.assertRaisesRegex(TrainingIntegrityError, "semantic dataset digest"):
                run_production_training(
                    **common, output_root=root / "semantic-output",
                    expected_training_artifact_sha256=second["output_digest"],
                    expected_training_dataset_digest=first["dataset_digest"],
                )
            self.assertFalse((root / "raw-output").exists())
            self.assertFalse((root / "semantic-output").exists())

    @requires_production_python
    def test_sealed_training_reader_rejects_hardlinks_before_model_or_output(self):
        with tempfile.TemporaryDirectory(prefix="egv-training-hardlink-") as temporary:
            root = Path(temporary)
            dataset = FrozenTrainingDataset(
                LedgerCutoff(
                    "hardlink-campaign", 1, "event", digest_for("hardlink-event"),
                    digest_for("hardlink-receipts"), 1, "evaluator-key",
                ),
                (),
                {},
            )
            artifact = root / "sealed-training.json"
            freezer_report = seal_runtime_dataset(dataset, artifact)
            alias = root / "hardlink-alias.json"
            os.link(artifact, alias)
            output = root / "output"
            with self.assertRaisesRegex(TrainingIntegrityError, "single-link"):
                run_production_training(
                    model_root=root / "missing-model", training_dataset=alias,
                    evaluator_manifest=root / "missing-service", evaluator_public_key=root / "missing-key",
                    evaluator_command=root / "missing-command",
                    evaluator_transfer_command=root / "missing-transfer-command",
                    output_root=output,
                    expected_training_artifact_sha256=freezer_report["output_digest"],
                    expected_training_dataset_digest=freezer_report["dataset_digest"], device="cuda",
                )
            self.assertFalse(output.exists())

    @requires_production_python
    def test_sealed_training_reader_rejects_leaf_swap_before_open(self):
        with tempfile.TemporaryDirectory(prefix="egv-training-swap-") as temporary:
            root = Path(temporary)
            dataset = FrozenTrainingDataset(
                LedgerCutoff(
                    "swap-campaign", 1, "event", digest_for("swap-event"),
                    digest_for("swap-receipts"), 1, "evaluator-key",
                ),
                (),
                {},
            )
            artifact = root / "sealed-training.json"
            freezer_report = seal_runtime_dataset(dataset, artifact)
            replacement = root / "replacement.json"
            replacement.write_bytes(artifact.read_bytes())
            real_open = os.open
            swapped = False

            def swap_then_open(path, flags, *args, **kwargs):
                nonlocal swapped
                if Path(path) == artifact and not swapped:
                    swapped = True
                    artifact.unlink()
                    replacement.replace(artifact)
                return real_open(path, flags, *args, **kwargs)

            output = root / "output"
            with patch("egv.training.trainer.os.open", side_effect=swap_then_open):
                with self.assertRaisesRegex(TrainingIntegrityError, "changed during admission"):
                    run_production_training(
                        model_root=root / "missing-model", training_dataset=artifact,
                        evaluator_manifest=root / "missing-service", evaluator_public_key=root / "missing-key",
                        evaluator_command=root / "missing-command",
                        evaluator_transfer_command=root / "missing-transfer-command",
                        output_root=output,
                        expected_training_artifact_sha256=freezer_report["output_digest"],
                        expected_training_dataset_digest=freezer_report["dataset_digest"], device="cuda",
                    )
            self.assertTrue(swapped)
            self.assertFalse(output.exists())

    @requires_production_python
    def test_sealed_training_reader_rejects_parent_and_leaf_links_or_reparse_points(self):
        with tempfile.TemporaryDirectory(prefix="egv-training-reparse-") as temporary:
            root = Path(temporary)
            dataset = FrozenTrainingDataset(
                LedgerCutoff(
                    "reparse-campaign", 1, "event", digest_for("reparse-event"),
                    digest_for("reparse-receipts"), 1, "evaluator-key",
                ),
                (),
                {},
            )
            artifact = root / "sealed-training.json"
            freezer_report = seal_runtime_dataset(dataset, artifact)
            real_lstat = Path.lstat

            def reparse_stat(value):
                return SimpleNamespace(
                    st_mode=value.st_mode,
                    st_dev=value.st_dev,
                    st_ino=value.st_ino,
                    st_size=value.st_size,
                    st_nlink=value.st_nlink,
                    st_file_attributes=getattr(value, "st_file_attributes", 0) | 0x400,
                )

            def symlink_stat(value):
                return SimpleNamespace(
                    st_mode=stat.S_IFLNK | stat.S_IMODE(value.st_mode),
                    st_dev=value.st_dev,
                    st_ino=value.st_ino,
                    st_size=value.st_size,
                    st_nlink=value.st_nlink,
                    st_file_attributes=getattr(value, "st_file_attributes", 0),
                )

            def invoke_with_fake_link(link_path, output, transform):
                def fake_lstat(path, *args, **kwargs):
                    value = real_lstat(path, *args, **kwargs)
                    return transform(value) if path == link_path else value

                with patch.object(Path, "lstat", fake_lstat):
                    run_production_training(
                        model_root=root / "missing-model", training_dataset=artifact,
                        evaluator_manifest=root / "missing-service", evaluator_public_key=root / "missing-key",
                        evaluator_command=root / "missing-command",
                        evaluator_transfer_command=root / "missing-transfer-command",
                        output_root=output,
                        expected_training_artifact_sha256=freezer_report["output_digest"],
                        expected_training_dataset_digest=freezer_report["dataset_digest"], device="cuda",
                    )

            for label, link_path, output, transform, error in (
                ("parent-reparse", root, root / "parent-reparse-output", reparse_stat, "symlink or reparse ancestor"),
                ("parent-symlink", root, root / "parent-symlink-output", symlink_stat, "symlink or reparse ancestor"),
                ("leaf-reparse", artifact, root / "leaf-reparse-output", reparse_stat, "regular non-reparse"),
                ("leaf-symlink", artifact, root / "leaf-symlink-output", symlink_stat, "regular non-reparse"),
            ):
                with self.subTest(label=label):
                    with self.assertRaisesRegex(TrainingIntegrityError, error):
                        invoke_with_fake_link(link_path, output, transform)
                    self.assertFalse(output.exists())

    @requires_production_python
    def test_sealed_training_reader_enforces_explicit_byte_ceiling(self):
        with tempfile.TemporaryDirectory(prefix="egv-training-ceiling-") as temporary:
            root = Path(temporary)
            artifact = root / "oversized.json"
            artifact.write_bytes(b"x" * 65)
            output = root / "output"
            with patch("egv.training.trainer.SEALED_RUNTIME_DATASET_MAX_BYTES", 64):
                with self.assertRaisesRegex(TrainingIntegrityError, "byte ceiling"):
                    run_production_training(
                        model_root=root / "missing-model", training_dataset=artifact,
                        evaluator_manifest=root / "missing-service", evaluator_public_key=root / "missing-key",
                        evaluator_command=root / "missing-command",
                        evaluator_transfer_command=root / "missing-transfer-command",
                        output_root=output,
                        expected_training_artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        expected_training_dataset_digest=digest_for("dataset"), device="cuda",
                    )
            self.assertFalse(output.exists())

    def test_lora_change_attestation_rejects_noop_and_changed_a_with_zero_b(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")
        manifest = _frozen_target_manifest()
        initial = _initial_lora_state()
        with self.assertRaisesRegex(TrainingIntegrityError, "nonzero effective delta"):
            build_lora_change_attestation(
                initial,
                _clone_lora_state(initial),
                target_manifest=manifest,
                protocol=self.protocol,
            )
        selected = _clone_lora_state(initial)
        a_name = (
            "base_model.model." + FROZEN_LORA_TARGETS[0]
            + ".lora_A.default.weight"
        )
        selected[a_name] = torch.full_like(selected[a_name], 2.0)
        with self.assertRaisesRegex(TrainingIntegrityError, "nonzero effective delta"):
            build_lora_change_attestation(
                initial,
                selected,
                target_manifest=manifest,
                protocol=self.protocol,
            )

    def test_lora_change_attestation_accepts_valid_effective_delta_deterministically(self):
        try:
            __import__("torch")
        except ImportError:
            self.skipTest("torch is unavailable")
        manifest = _frozen_target_manifest()
        initial = _initial_lora_state()
        selected = _clone_lora_state(initial)
        target_name = FROZEN_LORA_TARGETS[0]
        b_name = "base_model.model." + target_name + ".lora_B.default.weight"
        selected[b_name][0, 0] = 1.0
        first = build_lora_change_attestation(
            initial, selected, target_manifest=manifest, protocol=self.protocol
        )
        second = build_lora_change_attestation(
            _clone_lora_state(initial),
            _clone_lora_state(selected),
            target_manifest=manifest,
            protocol=self.protocol,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["parameter_count"], 2 * len(FROZEN_LORA_TARGETS))
        self.assertEqual(first["target_pair_count"], len(FROZEN_LORA_TARGETS))
        self.assertEqual(first["effective_delta_changed_target_count"], 1)
        self.assertEqual(
            first["effective_delta_changed_target_set_digest"],
            digest_for([target_name]),
        )

    def test_lora_change_attestation_uses_coherent_private_tensor_snapshot(self):
        try:
            __import__("torch")
        except ImportError:
            self.skipTest("torch is unavailable")
        manifest = _frozen_target_manifest()
        initial = _initial_lora_state()
        selected = _clone_lora_state(initial)
        b_name = (
            "base_model.model." + FROZEN_LORA_TARGETS[0]
            + ".lora_B.default.weight"
        )
        selected[b_name][0, 0] = 1.0
        expected = build_lora_change_attestation(
            initial,
            _clone_lora_state(selected),
            target_manifest=manifest,
            protocol=self.protocol,
        )
        tensor_digest = training_trainer._attestation_tensor_digest
        digest_calls = 0

        def mutate_caller_after_digest(value):
            nonlocal digest_calls
            result = tensor_digest(value)
            digest_calls += 1
            if digest_calls == 1:
                selected[b_name][0, 0] = 0.0
            return result

        with patch.object(
            training_trainer,
            "_attestation_tensor_digest",
            side_effect=mutate_caller_after_digest,
        ):
            actual = build_lora_change_attestation(
                initial,
                selected,
                target_manifest=manifest,
                protocol=self.protocol,
            )
        self.assertEqual(actual, expected)
        self.assertEqual(float(selected[b_name][0, 0]), 0.0)

    def test_lora_change_attestation_rejects_missing_and_extra_target_pairs(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")
        manifest = _frozen_target_manifest()
        initial = _initial_lora_state()
        b_name = (
            "base_model.model." + FROZEN_LORA_TARGETS[0]
            + ".lora_B.default.weight"
        )
        missing_initial = _clone_lora_state(initial)
        del missing_initial[b_name]
        with self.assertRaisesRegex(TrainingIntegrityError, "exact A/B pair"):
            build_lora_change_attestation(
                missing_initial,
                _clone_lora_state(missing_initial),
                target_manifest=manifest,
                protocol=self.protocol,
            )
        extra_initial = _clone_lora_state(initial)
        extra_initial["base_model.model.extra.lora_A.default.weight"] = torch.ones(
            (16, 3), dtype=torch.float32
        )
        extra_initial["base_model.model.extra.lora_B.default.weight"] = torch.zeros(
            (4, 16), dtype=torch.float32
        )
        with self.assertRaisesRegex(TrainingIntegrityError, "missing or extra adapter"):
            build_lora_change_attestation(
                extra_initial,
                _clone_lora_state(extra_initial),
                target_manifest=manifest,
                protocol=self.protocol,
            )

    def test_lora_change_attestation_rejects_shape_dtype_and_nonfinite_drift(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")
        manifest = _frozen_target_manifest()
        initial = _initial_lora_state()
        a_name = (
            "base_model.model." + FROZEN_LORA_TARGETS[0]
            + ".lora_A.default.weight"
        )
        b_name = (
            "base_model.model." + FROZEN_LORA_TARGETS[0]
            + ".lora_B.default.weight"
        )
        cases = []
        shape_state = _clone_lora_state(initial)
        shape_state[a_name] = torch.ones((16, 4), dtype=torch.float32)
        cases.append(("shape", shape_state, "shape changed"))
        dtype_state = _clone_lora_state(initial)
        dtype_state[a_name] = dtype_state[a_name].to(dtype=torch.float64)
        cases.append(("dtype", dtype_state, "float32 dtype"))
        nonfinite_state = _clone_lora_state(initial)
        nonfinite_state[b_name][0, 0] = float("nan")
        cases.append(("nonfinite", nonfinite_state, "must be finite"))
        for label, selected, error in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(TrainingIntegrityError, error):
                    build_lora_change_attestation(
                        initial,
                        selected,
                        target_manifest=manifest,
                        protocol=self.protocol,
                    )

    def test_lora_sealed_binding_binds_selected_checkpoint_and_reloaded_state(self):
        try:
            __import__("torch")
        except ImportError:
            self.skipTest("torch is unavailable")
        manifest = _frozen_target_manifest()
        initial = _initial_lora_state()
        selected = _clone_lora_state(initial)
        b_name = (
            "base_model.model." + FROZEN_LORA_TARGETS[0]
            + ".lora_B.default.weight"
        )
        selected[b_name][0, 0] = 1.0
        selected_attestation = build_lora_change_attestation(
            initial, selected, target_manifest=manifest, protocol=self.protocol
        )
        sealed_adapter_digest = digest_for("sealed-adapter")
        binding = build_lora_sealed_binding(
            selected_change_attestation=selected_attestation,
            reloaded_change_attestation=dict(selected_attestation),
            selected_checkpoint_artifact_digest=digest_for("selected-checkpoint"),
            base_immutability_proof_digest=digest_for("immutability-proof"),
            sealed_adapter_digest=sealed_adapter_digest,
            evaluator_approved_adapter_digest=sealed_adapter_digest,
            selected_development_receipt_digest=digest_for("evaluator-receipt"),
            reloaded_applied_model_state_digest=digest_for("reloaded-state"),
        )
        self.assertEqual(
            binding["lora_change_attestation_digest"],
            selected_attestation["attestation_digest"],
        )
        self.assertEqual(binding["sealed_adapter_digest"], sealed_adapter_digest)
        self.assertEqual(binding["evaluator_approved_adapter_digest"], sealed_adapter_digest)
        self.assertEqual(
            binding["binding_digest"],
            digest_for({key: value for key, value in binding.items() if key != "binding_digest"}),
        )
        changed_reload = _clone_lora_state(selected)
        changed_reload[b_name][0, 1] = 2.0
        changed_reload_attestation = build_lora_change_attestation(
            initial, changed_reload, target_manifest=manifest, protocol=self.protocol
        )
        with self.assertRaisesRegex(TrainingIntegrityError, "reloaded LoRA effective-delta proof differs"):
            build_lora_sealed_binding(
                selected_change_attestation=selected_attestation,
                reloaded_change_attestation=changed_reload_attestation,
                selected_checkpoint_artifact_digest=digest_for("selected-checkpoint"),
                base_immutability_proof_digest=digest_for("immutability-proof"),
                sealed_adapter_digest=sealed_adapter_digest,
                evaluator_approved_adapter_digest=sealed_adapter_digest,
                selected_development_receipt_digest=digest_for("evaluator-receipt"),
                reloaded_applied_model_state_digest=digest_for("reloaded-state"),
            )
        with self.assertRaisesRegex(TrainingIntegrityError, "evaluator-approved"):
            build_lora_sealed_binding(
                selected_change_attestation=selected_attestation,
                reloaded_change_attestation=selected_attestation,
                selected_checkpoint_artifact_digest=digest_for("selected-checkpoint"),
                base_immutability_proof_digest=digest_for("immutability-proof"),
                sealed_adapter_digest=sealed_adapter_digest,
                evaluator_approved_adapter_digest=digest_for("different-adapter"),
                selected_development_receipt_digest=digest_for("evaluator-receipt"),
                reloaded_applied_model_state_digest=digest_for("reloaded-state"),
            )

    def test_selected_lora_restore_requires_expected_keys_but_allows_base_omissions(self):
        state = {"lora.a": object(), "lora.b": object()}

        class RestoreModel:
            def __init__(self, *, missing=(), unexpected=()):
                self.missing = tuple(missing)
                self.unexpected = tuple(unexpected)
                self.strict = None

            def load_state_dict(self, supplied, *, strict):
                self.supplied = supplied
                self.strict = strict
                return SimpleNamespace(
                    missing_keys=self.missing,
                    unexpected_keys=self.unexpected,
                )

        accepted = RestoreModel(missing=("base.weight",))
        _restore_selected_lora_state(accepted, state, tuple(state))
        self.assertIs(accepted.supplied, state)
        self.assertFalse(accepted.strict)
        with self.assertRaisesRegex(TrainingIntegrityError, "expected LoRA parameters missing"):
            _restore_selected_lora_state(
                RestoreModel(missing=("base.weight", "lora.a")),
                state,
                tuple(state),
            )
        with self.assertRaisesRegex(TrainingIntegrityError, "unexpected parameters"):
            _restore_selected_lora_state(
                RestoreModel(unexpected=("unknown.adapter",)),
                state,
                tuple(state),
            )

    def test_adapter_seal_recomputes_effective_delta_and_rejects_noop(self):
        try:
            __import__("torch")
        except ImportError:
            self.skipTest("torch is unavailable")
        manifest = _frozen_target_manifest()
        initial = _initial_lora_state()
        selected = _clone_lora_state(initial)
        b_name = (
            "base_model.model." + FROZEN_LORA_TARGETS[0]
            + ".lora_B.default.weight"
        )
        selected[b_name][0, 0] = 1.0
        selected_attestation = build_lora_change_attestation(
            initial, selected, target_manifest=manifest, protocol=self.protocol
        )
        with tempfile.TemporaryDirectory(prefix="egv-lora-noop-") as temporary:
            class NoopModel:
                def state_dict(self):
                    return _clone_lora_state(initial)

            trainer = object.__new__(LoRATrainer)
            object.__setattr__(trainer, "production", True)
            object.__setattr__(trainer, "protocol", self.protocol)
            object.__setattr__(trainer, "_trained_model", NoopModel())
            object.__setattr__(trainer, "_initial_lora_state", initial)
            object.__setattr__(trainer, "_target_manifest", manifest)
            object.__setattr__(trainer, "_lora_change_attestation", dict(selected_attestation))
            tree = training_trainer._create_private_training_tree(
                Path(temporary) / "output"
            )
            object.__setattr__(trainer, "_private_training_tree", tree)
            output = tree.root / "adapter"
            with self.assertRaisesRegex(TrainingIntegrityError, "nonzero effective delta"):
                trainer.seal_adapter(output)
            self.assertFalse(output.exists())

    def test_adapter_seal_rejects_preexisting_empty_destination(self):
        try:
            __import__("torch")
        except ImportError:
            self.skipTest("torch is unavailable")
        manifest = _frozen_target_manifest()
        initial = _initial_lora_state()
        selected = _clone_lora_state(initial)
        b_name = (
            "base_model.model." + FROZEN_LORA_TARGETS[0]
            + ".lora_B.default.weight"
        )
        selected[b_name][0, 0] = 1.0
        selected_attestation = build_lora_change_attestation(
            initial, selected, target_manifest=manifest, protocol=self.protocol
        )

        class SelectedModel:
            def state_dict(self):
                return _clone_lora_state(selected)

            def save_pretrained(self, *_args, **_kwargs):
                raise AssertionError("preexisting destination must fail before PEFT save")

        trainer = object.__new__(LoRATrainer)
        object.__setattr__(trainer, "production", True)
        object.__setattr__(trainer, "protocol", self.protocol)
        object.__setattr__(trainer, "_trained_model", SelectedModel())
        object.__setattr__(trainer, "_initial_lora_state", initial)
        object.__setattr__(trainer, "_target_manifest", manifest)
        object.__setattr__(trainer, "_lora_change_attestation", dict(selected_attestation))
        with tempfile.TemporaryDirectory(prefix="egv-lora-stale-") as temporary:
            tree = training_trainer._create_private_training_tree(
                Path(temporary) / "output"
            )
            object.__setattr__(trainer, "_private_training_tree", tree)
            destination = tree.root / "adapter"
            destination.mkdir()
            with self.assertRaisesRegex(TrainingIntegrityError, "newly absent"):
                trainer.seal_adapter(destination)

    def test_direct_adapter_seal_requires_exact_owned_private_leaf_before_model_access(self):
        try:
            __import__("torch")
        except ImportError:
            self.skipTest("torch is unavailable")
        manifest = _frozen_target_manifest()
        initial = _initial_lora_state()
        selected = _clone_lora_state(initial)
        b_name = (
            "base_model.model." + FROZEN_LORA_TARGETS[0]
            + ".lora_B.default.weight"
        )
        selected[b_name][0, 0] = 1.0
        selected_attestation = build_lora_change_attestation(
            initial, selected, target_manifest=manifest, protocol=self.protocol
        )

        def build_trainer(tree):
            state_calls = []
            save_calls = []

            class SelectedModel:
                def state_dict(self):
                    state_calls.append(True)
                    return _clone_lora_state(selected)

                def save_pretrained(self, root, *, safe_serialization):
                    save_calls.append((Path(root), safe_serialization))
                    raise RuntimeError("bounded positive-path stop")

            trainer = object.__new__(LoRATrainer)
            object.__setattr__(trainer, "production", True)
            object.__setattr__(trainer, "protocol", self.protocol)
            object.__setattr__(trainer, "_trained_model", SelectedModel())
            object.__setattr__(trainer, "_initial_lora_state", initial)
            object.__setattr__(trainer, "_target_manifest", manifest)
            object.__setattr__(
                trainer, "_lora_change_attestation", dict(selected_attestation)
            )
            object.__setattr__(trainer, "_private_training_tree", tree)
            self._attach_selected_development_evidence(trainer)
            return trainer, state_calls, save_calls

        with tempfile.TemporaryDirectory(prefix="egv-lora-owned-leaf-") as temporary:
            root = Path(temporary)

            intended_tree = training_trainer._create_private_training_tree(
                root / "intended-output"
            )
            trainer, state_calls, save_calls = build_trainer(intended_tree)
            exact_destination = intended_tree.root / "adapter"
            with self.assertRaisesRegex(TrainingDependencyError, "could not write"):
                trainer.seal_adapter(exact_destination)
            self.assertEqual(state_calls, [True])
            self.assertEqual(save_calls, [(exact_destination, True)])
            self.assertTrue(exact_destination.is_dir())

            cross_tree = training_trainer._create_private_training_tree(
                root / "cross-output"
            )
            foreign_tree = training_trainer._create_private_training_tree(
                root / "foreign-output"
            )
            trainer, state_calls, save_calls = build_trainer(cross_tree)
            foreign_destination = foreign_tree.root / "adapter"
            with self.assertRaisesRegex(TrainingIntegrityError, "exact private staging"):
                trainer.seal_adapter(foreign_destination)
            self.assertEqual(state_calls, [])
            self.assertEqual(save_calls, [])
            self.assertFalse(foreign_destination.exists())

            outside_tree = training_trainer._create_private_training_tree(
                root / "outside-output"
            )
            trainer, state_calls, save_calls = build_trainer(outside_tree)
            outside_destination = root / "outside-adapter"
            with self.assertRaisesRegex(TrainingIntegrityError, "exact private staging"):
                trainer.seal_adapter(outside_destination)
            self.assertEqual(state_calls, [])
            self.assertEqual(save_calls, [])
            self.assertFalse(outside_destination.exists())

            substituted_tree = training_trainer._create_private_training_tree(
                root / "substituted-output"
            )
            trainer, state_calls, save_calls = build_trainer(substituted_tree)
            retained = root / "retained-private-tree"
            substituted_tree.root.replace(retained)
            substituted_tree.root.mkdir()
            marker = substituted_tree.root / "foreign-marker"
            marker.write_text("foreign", encoding="utf-8")
            substituted_destination = substituted_tree.root / "adapter"
            with self.assertRaisesRegex(TrainingIntegrityError, "identity changed"):
                trainer.seal_adapter(substituted_destination)
            self.assertEqual(state_calls, [])
            self.assertEqual(save_calls, [])
            self.assertEqual(marker.read_text(encoding="utf-8"), "foreign")
            self.assertFalse(substituted_destination.exists())

    def test_adapter_seal_rejects_mutated_selected_receipt_before_save(self):
        try:
            __import__("torch")
        except ImportError:
            self.skipTest("torch is unavailable")
        manifest = _frozen_target_manifest()
        initial = _initial_lora_state()
        selected = _clone_lora_state(initial)
        b_name = (
            "base_model.model." + FROZEN_LORA_TARGETS[0]
            + ".lora_B.default.weight"
        )
        selected[b_name][0, 0] = 1.0
        selected_attestation = build_lora_change_attestation(
            initial, selected, target_manifest=manifest, protocol=self.protocol
        )
        save_calls = []

        class SelectedModel:
            def state_dict(self):
                return _clone_lora_state(selected)

            def save_pretrained(self, *_args, **_kwargs):
                save_calls.append(True)

        trainer = LoRATrainer(
            self.inputs,
            self.protocol,
            self.gateway,
            _Tokenizer(),
            production=False,
        )
        trainer.run()
        object.__setattr__(trainer, "production", True)
        object.__setattr__(trainer, "_trained_model", SelectedModel())
        object.__setattr__(trainer, "_initial_lora_state", initial)
        object.__setattr__(trainer, "_target_manifest", manifest)
        object.__setattr__(trainer, "_lora_change_attestation", dict(selected_attestation))
        evaluation = trainer._selected_evaluation
        object.__setattr__(
            trainer, "_selected_checkpoint_artifact_digest", evaluation.checkpoint_digest
        )
        candidate_digest = evaluation.receipt["candidate_artifact_digest"]
        evaluation.receipt["resource_bucket"] = "OVER_25"
        self.assertEqual(evaluation.receipt["candidate_artifact_digest"], candidate_digest)
        with tempfile.TemporaryDirectory(prefix="egv-lora-receipt-mutation-") as temporary:
            published = Path(temporary) / "output"
            tree = training_trainer._create_private_training_tree(published)
            object.__setattr__(trainer, "_private_training_tree", tree)
            destination = tree.root / "adapter"
            with self.assertRaisesRegex(
                TrainingIntegrityError, "evaluation changed after selection"
            ):
                trainer.seal_adapter(destination)
            self.assertFalse(destination.exists())
            self.assertEqual(save_calls, [])

    def test_adapter_seal_rejects_competing_destination_claim_before_save(self):
        try:
            __import__("torch")
        except ImportError:
            self.skipTest("torch is unavailable")
        manifest = _frozen_target_manifest()
        initial = _initial_lora_state()
        selected = _clone_lora_state(initial)
        b_name = (
            "base_model.model." + FROZEN_LORA_TARGETS[0]
            + ".lora_B.default.weight"
        )
        selected[b_name][0, 0] = 1.0
        selected_attestation = build_lora_change_attestation(
            initial, selected, target_manifest=manifest, protocol=self.protocol
        )
        save_calls = []

        class SelectedModel:
            def state_dict(self):
                return _clone_lora_state(selected)

            def save_pretrained(self, *_args, **_kwargs):
                save_calls.append(True)

        trainer = object.__new__(LoRATrainer)
        object.__setattr__(trainer, "production", True)
        object.__setattr__(trainer, "protocol", self.protocol)
        object.__setattr__(trainer, "_trained_model", SelectedModel())
        object.__setattr__(trainer, "_initial_lora_state", initial)
        object.__setattr__(trainer, "_target_manifest", manifest)
        object.__setattr__(trainer, "_lora_change_attestation", dict(selected_attestation))
        with tempfile.TemporaryDirectory(prefix="egv-lora-race-") as temporary:
            published = Path(temporary) / "output"
            tree = training_trainer._create_private_training_tree(published)
            object.__setattr__(trainer, "_private_training_tree", tree)
            self._attach_selected_development_evidence(trainer)
            destination = tree.root / "adapter"
            original_lexists = os.path.lexists

            def claim_after_precheck(path):
                exists = original_lexists(path)
                if Path(path) == destination and not exists:
                    destination.mkdir()
                return exists

            verify_evaluation_calls = []
            original_verify_evaluation = DevelopmentLossGateway.verify_evaluation

            def count_verify_evaluation(gateway, *args, **kwargs):
                verify_evaluation_calls.append(True)
                return original_verify_evaluation(gateway, *args, **kwargs)

            with (
                patch("egv.training.trainer.os.path.lexists", side_effect=claim_after_precheck),
                patch.object(
                    DevelopmentLossGateway,
                    "verify_evaluation",
                    new=count_verify_evaluation,
                ),
            ):
                with self.assertRaisesRegex(TrainingIntegrityError, "newly absent"):
                    trainer.seal_adapter(destination)
            self.assertTrue(destination.is_dir())
            self.assertEqual(len(verify_evaluation_calls), 1)
            self.assertEqual(save_calls, [])


if __name__ == "__main__":
    unittest.main()
