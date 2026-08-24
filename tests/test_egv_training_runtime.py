"""Adversarial and CPU-fixture tests for the EGV Training runtime."""

from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
import base64
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from egv.canonical import canonical_json, digest_for
from egv.cli import main
from egv.evaluation.dataset import EvaluationCorpus
from egv.receipts import ReceiptSigner, key_id_for_public_key
from egv.training import (
    DevelopmentLossGateway,
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
    build_training_batch,
    build_private_development_runtime,
    model_state_digest_for_training,
    seal_training_inputs,
    tokenize_training_row,
    run_production_training,
    receive_external_adapter,
)
from egv.training.development import _decode_external_signature
from egv.variation.adapter import ADAPTER_MANIFEST_NAME, SealedAdapterArtifact, build_local_adapter_manifest
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


class _AdapterStagingModel(_Model):
    def save_pretrained(self, root, *, safe_serialization):
        if not safe_serialization:
            raise AssertionError("adapter staging must use safetensors")
        destination = Path(root)
        (destination / "adapter_config.json").write_text("{}", encoding="utf-8")
        (destination / "adapter_model.safetensors").write_bytes(b"sealed-fixture")


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
                        "--development-manifest",
                        "/private/dev.json",
                    ]
                ),
                1,
            )
        self.assertIn("fail closed", errors.getvalue())

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
            gateway = ExternalDevelopmentLossGateway(
                manifest_path, public_key_path=public_key, command=command, transfer_command=command
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
                ExternalDevelopmentLossGateway._invoke_pinned = staticmethod(lambda **_kwargs: {})
                with self.assertRaises(TrainingDependencyError):
                    gateway.validate_production_boundary(
                        expected_model_digest=self.model_digest, expected_protocol_digest=self.protocol.digest
                    )
            finally:
                ExternalDevelopmentLossGateway._invoke_pinned = staticmethod(original_invoke)
            gateway.validate_production_boundary(
                expected_model_digest=self.model_digest, expected_protocol_digest=self.protocol.digest
            )
            command.write_bytes(b"#!/bin/sh\nexit 9\n")
            with self.assertRaises(TrainingIntegrityError):
                ExternalDevelopmentLossGateway(
                    manifest_path, public_key_path=public_key, command=command, transfer_command=command
                )
            command.write_bytes(b"#!/bin/sh\nexit 0\n")
            public_key.write_bytes(b"forged")
            with self.assertRaises(TrainingIntegrityError):
                ExternalDevelopmentLossGateway(
                    manifest_path, public_key_path=public_key, command=command, transfer_command=command
                )

    def test_external_signed_envelope_rejects_noncanonical_ed25519_alias(self) -> None:
        encoded = self.signer.sign_bytes(b"external-envelope")
        self.assertEqual(len(_decode_external_signature(encoded)), 64)
        tail_alias = {"A": "B", "Q": "R", "g": "h", "w": "x"}
        with self.assertRaisesRegex(TrainingIntegrityError, "canonical"):
            _decode_external_signature(encoded[:-1] + tail_alias[encoded[-1]])

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
                manifest_path, public_key_path=public_key, command=command, transfer_command=command
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
                output_root=Path("missing-output"), device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
