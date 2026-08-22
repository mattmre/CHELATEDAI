"""Bounded LoRA training runtime and deterministic CPU fixture smoke."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..canonical import canonical_json, content_id, digest_for
from ..receipts import ReceiptSigner
from .development import (
    DevelopmentLossEvaluation,
    DevelopmentLossGateway,
    ExternalDevelopmentLossGateway,
    TrainingCheckpoint,
    model_state_digest_for_training,
)
from .protocol import (
    SealedTrainingInputs,
    TokenizedTrainingExample,
    TrainingConfigurationError,
    TrainingDependencyError,
    TrainingError,
    TrainingIntegrityError,
    TrainingProtocol,
    TrainingRow,
    build_training_batch,
    seal_training_inputs,
)


TRAINING_RUNTIME_SCHEMA = "egv-training-runtime-v1"
SEALED_RUNTIME_DATASET_SCHEMA = "egv-sealed-training-runtime-input-v1"
SEALED_DEVELOPMENT_DATASET_SCHEMA = "egv-sealed-development-runtime-input-v1"


def _require_finite_loss(value: Any, source: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise TrainingIntegrityError("{} returned a non-finite loss".format(source))
    loss = float(value)
    if loss < 0:
        raise TrainingIntegrityError("{} returned a negative loss".format(source))
    return loss


@dataclass(frozen=True)
class TrainingRunReport:
    protocol_digest: str
    model_digest: str
    data_manifest_digest: str
    checkpoint_digests: Tuple[str, ...]
    development_evaluations: Tuple[DevelopmentLossEvaluation, ...]
    selected_checkpoint_digest: str
    selected_loss: float
    production: bool
    promotion_disposition: str
    real_qwen_execution_claimed: bool = False
    schema_version: str = TRAINING_RUNTIME_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "protocol_digest": self.protocol_digest,
            "model_digest": self.model_digest,
            "data_manifest_digest": self.data_manifest_digest,
            "checkpoint_digests": list(self.checkpoint_digests),
            "development_evaluations": [
                {
                    "checkpoint_digest": evaluation.checkpoint_digest,
                    "checkpoint_id": evaluation.checkpoint_id,
                    "model_digest": evaluation.model_digest,
                    "data_manifest_digest": evaluation.data_manifest_digest,
                    "development_manifest_digest": evaluation.development_manifest_digest,
                    "protocol_digest": evaluation.protocol_digest,
                    "gateway_digest": evaluation.gateway_digest,
                    "loss": evaluation.loss,
                    "sample_count": evaluation.sample_count,
                    "receipt": dict(evaluation.receipt),
                }
                for evaluation in self.development_evaluations
            ],
            "selected_checkpoint_digest": self.selected_checkpoint_digest,
            "selected_loss": self.selected_loss,
            "production": self.production,
            "promotion_disposition": self.promotion_disposition,
            "real_qwen_execution_claimed": self.real_qwen_execution_claimed,
        }

    def validate(self) -> None:
        if self.schema_version != TRAINING_RUNTIME_SCHEMA:
            raise TrainingIntegrityError("unsupported Training runtime report schema")
        for name in ("protocol_digest", "model_digest", "data_manifest_digest", "selected_checkpoint_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise TrainingIntegrityError("Training report {} is not a SHA-256 digest".format(name))
        if not self.checkpoint_digests or not self.development_evaluations:
            raise TrainingIntegrityError("Training report has no checkpoint/evaluation evidence")
        if self.selected_checkpoint_digest not in self.checkpoint_digests:
            raise TrainingIntegrityError("selected checkpoint is not in the evaluated checkpoint set")
        _require_finite_loss(self.selected_loss, "selected development loss")
        if self.promotion_disposition != "NOT_PROMOTED_TRAINING_ARTIFACT":
            raise TrainingIntegrityError("Training cannot emit a Variation promotion disposition")
        if self.real_qwen_execution_claimed != self.production:
            raise TrainingIntegrityError("Qwen execution claim must exactly match the production runtime tier")


class LoRATrainer:
    """Run the frozen LoRA protocol against sealed inputs and a dev gateway."""

    def __init__(
        self,
        inputs: SealedTrainingInputs,
        protocol: Optional[TrainingProtocol],
        gateway: Optional[Any],
        tokenizer: Any,
        *,
        production: bool = True,
        model_root: Optional[Path] = None,
        frozen_dataset: Optional[Any] = None,
        target_manifest: Optional[Any] = None,
        checkpoint_root: Optional[Path] = None,
    ) -> None:
        if type(inputs) is not SealedTrainingInputs:
            raise TrainingIntegrityError("LoRATrainer requires sealed Training inputs")
        if type(protocol) is not TrainingProtocol:
            raise TrainingConfigurationError("LoRATrainer requires the exact frozen TrainingProtocol")
        expected_gateway_type = ExternalDevelopmentLossGateway if production else DevelopmentLossGateway
        if type(gateway) is not expected_gateway_type:
            raise TrainingDependencyError(
                "production requires the external evaluator client" if production
                else "fixture runtime requires the local fixture gateway"
            )
        if not callable(tokenizer):
            raise TrainingDependencyError("LoRATrainer requires a tokenizer callable")
        protocol.validate()
        inputs.validate(protocol)
        if production:
            from .contracts import FrozenTrainingDataset
            from .targets import LoraTargetManifest

            if type(frozen_dataset) is not FrozenTrainingDataset or type(target_manifest) is not LoraTargetManifest:
                raise TrainingIntegrityError(
                    "production Training requires the canonical frozen dataset and sealed target manifest"
                )
            if checkpoint_root is None:
                raise TrainingConfigurationError("production Training requires a checkpoint artifact root")
            target_manifest.verify()
            if target_manifest.model_manifest_digest != inputs.model_digest:
                raise TrainingIntegrityError("target manifest is bound to a different pinned model")
        gateway.validate_production_boundary(
            expected_model_digest=inputs.model_digest,
            expected_protocol_digest=protocol.digest,
        ) if production else self._validate_fixture_gateway(gateway, inputs, protocol)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "gateway", gateway)
        object.__setattr__(self, "tokenizer", tokenizer)
        object.__setattr__(self, "production", bool(production))
        object.__setattr__(self, "model_root", Path(model_root).resolve() if model_root is not None else None)
        object.__setattr__(self, "frozen_dataset", frozen_dataset)
        object.__setattr__(self, "target_manifest", target_manifest)
        object.__setattr__(self, "checkpoint_root", Path(checkpoint_root).resolve() if checkpoint_root is not None else None)
        object.__setattr__(
            self,
            "_contract_digest",
            digest_for(
                {
                    "inputs_seal": inputs.seal_digest,
                    "protocol_digest": protocol.digest,
                    "gateway_digest": gateway.gateway_digest,
                    "frozen_dataset_digest": frozen_dataset.digest if frozen_dataset is not None else None,
                    "target_manifest_digest": target_manifest.digest if target_manifest is not None else None,
                    "tokenizer_id": id(tokenizer),
                    "production": bool(production),
                }
            ),
        )
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_initialized", False) and name in {
            "inputs",
            "protocol",
            "gateway",
            "tokenizer",
            "production",
            "model_root",
            "frozen_dataset",
            "target_manifest",
            "checkpoint_root",
            "_contract_digest",
        }:
            raise AttributeError("LoRATrainer runtime contract is immutable")
        object.__setattr__(self, name, value)

    @staticmethod
    def _validate_fixture_gateway(
        gateway: DevelopmentLossGateway, inputs: SealedTrainingInputs, protocol: TrainingProtocol
    ) -> None:
        if gateway.production:
            raise TrainingDependencyError("fixture Training runtime requires a non-production gateway")
        if gateway.model_digest != inputs.model_digest or gateway.protocol_digest != protocol.digest:
            raise TrainingIntegrityError("fixture gateway does not bind the sealed model/protocol")

    def _validate_run_boundary(self) -> None:
        if type(self) is not LoRATrainer:
            raise TrainingDependencyError("production requires the exact LoRATrainer type")
        self.protocol.validate()
        self.inputs.validate(self.protocol)
        if digest_for(
            {
                "inputs_seal": self.inputs.seal_digest,
                "protocol_digest": self.protocol.digest,
                "gateway_digest": self.gateway.gateway_digest,
                "frozen_dataset_digest": self.frozen_dataset.digest if self.frozen_dataset is not None else None,
                "target_manifest_digest": self.target_manifest.digest if self.target_manifest is not None else None,
                "tokenizer_id": id(self.tokenizer),
                "production": self.production,
            }
        ) != self._contract_digest:
            raise TrainingIntegrityError("Training runtime contract changed after construction")
        if self.production:
            if sys.version_info < (3, 10):
                raise TrainingDependencyError("production LoRA training requires Python >=3.10")
            if self.inputs.fixture_only:
                raise TrainingDependencyError("fixture-only Training inputs cannot enter production")
            self.gateway.validate_production_boundary(
                expected_model_digest=self.inputs.model_digest,
                expected_protocol_digest=self.protocol.digest,
            )
            self._validate_production_model()
        else:
            if not self.inputs.fixture_only:
                raise TrainingDependencyError("fixture Training runtime requires fixture-only sealed inputs")
            self._validate_fixture_gateway(self.gateway, self.inputs, self.protocol)

    def _validate_production_model(self) -> None:
        try:
            from ..variation.model import LoadedPinnedModel, PinnedModelManifest
        except ImportError as exc:
            raise TrainingDependencyError("production Training requires the merged Variation model contract") from exc
        if type(self.inputs.model) is not LoadedPinnedModel:
            raise TrainingDependencyError("production Training requires a LoadedPinnedModel, not a mock or digest")
        loaded = self.inputs.model
        if type(loaded.manifest) is not PinnedModelManifest:
            raise TrainingDependencyError("production Training requires the exact pinned model manifest")
        loaded.manifest.validate_contract()
        if loaded.manifest_digest != self.inputs.model_digest:
            raise TrainingIntegrityError("sealed Training model digest does not match the pinned manifest")
        try:
            peft = importlib.import_module("peft")
        except ImportError as exc:
            raise TrainingDependencyError("production LoRA training requires PEFT") from exc
        for name in ("LoraConfig", "get_peft_model"):
            if not callable(getattr(peft, name, None)):
                raise TrainingDependencyError("production PEFT runtime lacks {}".format(name))

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
        except ImportError:
            return

    def _apply_production_lora(self) -> Any:
        self._validate_production_model()
        peft = importlib.import_module("peft")
        base_model = self.inputs.model.model
        try:
            import torch
        except ImportError as exc:
            raise TrainingDependencyError("production Training requires torch") from exc
        floating_base = [parameter for parameter in base_model.parameters() if parameter.is_floating_point()]
        if not floating_base or any(
            parameter.device.type != self.protocol.device or parameter.dtype is not torch.bfloat16
            for parameter in floating_base
        ):
            raise TrainingIntegrityError("pinned base model is not entirely CUDA bfloat16")
        from .immutability import capture_base_model_snapshot
        from .targets import validate_lora_target_modules

        try:
            validate_lora_target_modules(base_model)
        except Exception as exc:
            raise TrainingIntegrityError("pinned model differs from the sealed 114-target manifest") from exc
        target_manifest = self.target_manifest
        if self.model_root is None:
            raise TrainingConfigurationError("production Training requires the sealed base-model root")
        try:
            snapshot = capture_base_model_snapshot(
                base_model,
                model_root=self.model_root,
                model_manifest_digest=self.inputs.model.manifest_digest,
                target_manifest=target_manifest,
                file_hashes=self.inputs.model.file_hashes,
            )
        except Exception as exc:
            raise TrainingIntegrityError("base-model immutability snapshot failed") from exc
        config = peft.LoraConfig(
            r=self.protocol.lora_rank,
            lora_alpha=self.protocol.lora_alpha,
            lora_dropout=self.protocol.lora_dropout,
            target_modules=list(target_manifest.module_names),
            bias="none",
            task_type="CAUSAL_LM",
        )
        try:
            model = peft.get_peft_model(base_model, config)
        except Exception as exc:
            raise TrainingDependencyError("the frozen PEFT LoRA configuration could not be applied") from exc
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(dtype=torch.float32)
        for name, parameter in model.named_parameters():
            if not parameter.is_floating_point():
                continue
            expected_dtype = torch.float32 if parameter.requires_grad else torch.bfloat16
            if parameter.device.type != self.protocol.device or parameter.dtype is not expected_dtype:
                raise TrainingIntegrityError(
                    "production parameter device/dtype differs from CUDA BF16 plus FP32 LoRA exception: {}".format(name)
                )

        from .immutability import assert_lora_only_trainable

        try:
            assert_lora_only_trainable(model, target_manifest)
        except Exception as exc:
            raise TrainingIntegrityError("PEFT trainability differs from the sealed LoRA targets") from exc
        object.__setattr__(self, "_target_manifest", target_manifest)
        object.__setattr__(self, "_base_snapshot", snapshot)
        return model

    @staticmethod
    def _fixture_train_step(model: Any, example: TokenizedTrainingExample) -> float:
        train_step = getattr(model, "train_step", None)
        if not callable(train_step):
            raise TrainingDependencyError("fixture model must expose train_step()")
        return _require_finite_loss(train_step(example), "fixture train_step")

    def _production_train_step(
        self, model: Any, example: TokenizedTrainingExample, optimizer: Any, *, accumulation_divisor: int
    ) -> float:
        try:
            import torch
        except ImportError as exc:
            raise TrainingDependencyError("production LoRA training requires torch") from exc
        model.train()
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError) as exc:
            raise TrainingDependencyError("production model exposes no device-bound parameters") from exc
        input_ids = torch.tensor([list(example.input_ids)], dtype=torch.long, device=device)
        attention_mask = torch.tensor([list(example.attention_mask)], dtype=torch.long, device=device)
        labels = torch.tensor([list(example.labels)], dtype=torch.long, device=device)
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
            value = _require_finite_loss(float(loss.detach().cpu().item()), "production model")
            if accumulation_divisor < 1:
                raise TrainingConfigurationError("gradient accumulation divisor must be positive")
            (loss / accumulation_divisor).backward()
            return value
        except TrainingError:
            raise
        except Exception as exc:
            raise TrainingDependencyError("production LoRA forward/backward failed") from exc

    def _make_checkpoint(self, model: Any, *, epoch: int, global_step: int) -> TrainingCheckpoint:
        state_digest = model_state_digest_for_training(model)
        adapter_digest = digest_for(
            {
                "model_digest": self.inputs.model_digest,
                "protocol_digest": self.protocol.digest,
                "state_digest": state_digest,
                "epoch": epoch,
            }
        )
        payload_digest = digest_for(
            {
                "adapter_digest": adapter_digest,
                "state_digest": state_digest,
                "global_step": global_step,
            }
        )
        return TrainingCheckpoint.create(
            epoch=epoch,
            global_step=global_step,
            model_digest=self.inputs.model_digest,
            protocol_digest=self.protocol.digest,
            data_manifest_digest=self.inputs.data_manifest.digest,
            adapter_digest=adapter_digest,
            state_digest=state_digest,
            payload_digest=payload_digest,
        )

    def run(self) -> TrainingRunReport:
        self._validate_run_boundary()
        self._seed_everything(self.protocol.train_seed)
        batches = [
            build_training_batch((row,), self.tokenizer, self.protocol)[0]
            for row in self.inputs.train_rows
        ]
        if len(batches) > self.protocol.max_train_rows:
            raise TrainingConfigurationError("sealed Training rows exceed the frozen row ceiling")
        model = self._apply_production_lora() if self.production else self.inputs.model
        optimizer = None
        scheduler = None
        if self.production:
            try:
                import torch

                trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
                optimizer = torch.optim.AdamW(
                    trainable,
                    lr=self.protocol.learning_rate,
                    weight_decay=self.protocol.weight_decay,
                )
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lambda step: max(0.0, 1.0 - (float(step) / float(self.protocol.max_train_steps))),
                )
            except TrainingError:
                raise
            except Exception as exc:
                raise TrainingDependencyError("frozen optimizer/scheduler could not be constructed") from exc
        checkpoints: List[TrainingCheckpoint] = []
        evaluations: List[DevelopmentLossEvaluation] = []
        best: Optional[DevelopmentLossEvaluation] = None
        no_improvement = 0
        global_step = 0
        accumulation = 0
        for epoch in range(1, self.protocol.max_epochs + 1):
            for batch_index, example in enumerate(batches):
                if self.production:
                    chunk_start = batch_index - (batch_index % self.protocol.gradient_accumulation_steps)
                    accumulation_divisor = min(
                        self.protocol.gradient_accumulation_steps, len(batches) - chunk_start
                    )
                    loss = self._production_train_step(
                        model, example, optimizer, accumulation_divisor=accumulation_divisor
                    )
                else:
                    loss = self._fixture_train_step(model, example)
                _require_finite_loss(loss, "training")
                accumulation += 1
                if accumulation >= self.protocol.gradient_accumulation_steps or example is batches[-1]:
                    if self.production:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                    global_step += 1
                    accumulation = 0
                    if global_step >= self.protocol.max_train_steps:
                        break
            checkpoint = self._make_checkpoint(model, epoch=epoch, global_step=max(global_step, 1))
            if self.production:
                object.__setattr__(self, "_trained_model", model)
                object.__setattr__(self, "_optimizer", optimizer)
                object.__setattr__(self, "_scheduler", scheduler)
                object.__setattr__(self, "_global_step", global_step)
                object.__setattr__(self, "_epoch_count", epoch)
                parent = getattr(self, "_checkpoint_head_digest", None)
                head = self._save_foundation_checkpoint(self.checkpoint_root, parent_artifact_digest=parent)
                object.__setattr__(self, "_checkpoint_head_digest", head)
                evaluation = self.gateway.evaluate(
                    checkpoint, model=model, checkpoint_artifact_digest=head
                )
            else:
                evaluation = self.gateway.evaluate(checkpoint, model=model)
            self.gateway.verify_evaluation(
                evaluation,
                checkpoint=checkpoint,
                expected_model_digest=self.inputs.model_digest,
                expected_data_manifest_digest=self.inputs.data_manifest.digest,
                expected_protocol_digest=self.protocol.digest,
            )
            checkpoints.append(checkpoint)
            evaluations.append(evaluation)
            if best is None or (evaluation.loss, evaluation.checkpoint_digest) < (best.loss, best.checkpoint_digest):
                best = evaluation
                if self.production:
                    best_adapter_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items() if "lora_" in name
                    }
                    object.__setattr__(self, "_best_adapter_state", best_adapter_state)
                no_improvement = 0
            else:
                no_improvement += 1
            if no_improvement > self.protocol.early_stopping_patience or global_step >= self.protocol.max_train_steps:
                break
        if best is None:
            raise TrainingIntegrityError("Training produced no valid development evaluation")
        if self.production:
            import torch
            from .checkpoint import TrainingCheckpointStore

            selected_artifact_digest = best.checkpoint_digest
            authoritative = TrainingCheckpointStore(self.checkpoint_root).resume(
                expected_artifact_digest=selected_artifact_digest
            )
            expected_adapter = self._best_adapter_state
            actual_adapter = authoritative.adapter_weights
            if set(actual_adapter) != set(expected_adapter) or any(
                not torch.equal(actual_adapter[name].detach().cpu(), expected_adapter[name])
                for name in expected_adapter
            ):
                raise TrainingIntegrityError("selected checkpoint adapter tensors differ from the evaluator-approved state")
            incompatible = model.load_state_dict(actual_adapter, strict=False)
            unexpected = tuple(getattr(incompatible, "unexpected_keys", ()))
            if unexpected:
                raise TrainingIntegrityError("selected adapter state contains unexpected parameters")
            object.__setattr__(self, "_selected_checkpoint_artifact_digest", selected_artifact_digest)
            object.__setattr__(self, "_selected_evaluation", best)
            object.__setattr__(self, "_selected_authoritative_adapter", dict(actual_adapter))
        report = TrainingRunReport(
            protocol_digest=self.protocol.digest,
            model_digest=self.inputs.model_digest,
            data_manifest_digest=self.inputs.data_manifest.digest,
            checkpoint_digests=tuple(
                evaluation.checkpoint_digest for evaluation in evaluations
            ) if self.production else tuple(checkpoint.digest for checkpoint in checkpoints),
            development_evaluations=tuple(evaluations),
            selected_checkpoint_digest=best.checkpoint_digest,
            selected_loss=best.loss,
            production=self.production,
            promotion_disposition="NOT_PROMOTED_TRAINING_ARTIFACT",
            real_qwen_execution_claimed=self.production,
        )
        report.validate()
        if self.production:
            from .immutability import validate_lora_only_training_state

            try:
                proof = validate_lora_only_training_state(
                    self._base_snapshot,
                    model,
                    target_manifest=self._target_manifest,
                    optimizer=optimizer,
                )
            except Exception as exc:
                raise TrainingIntegrityError("post-training base/optimizer immutability proof failed") from exc
            object.__setattr__(self, "_trained_model", model)
            object.__setattr__(self, "_immutability_proof", proof)
            object.__setattr__(self, "_optimizer", optimizer)
            object.__setattr__(self, "_scheduler", scheduler)
            object.__setattr__(self, "_global_step", global_step)
            object.__setattr__(self, "_epoch_count", len(evaluations))
        return report

    @staticmethod
    def _json_list(value: Any) -> Any:
        if isinstance(value, tuple):
            return [LoRATrainer._json_list(item) for item in value]
        if isinstance(value, list):
            return [LoRATrainer._json_list(item) for item in value]
        return value

    def _save_foundation_checkpoint(
        self, output_root: Path, *, parent_artifact_digest: Optional[str] = None
    ) -> str:
        from importlib import metadata as importlib_metadata
        import numpy as np
        import torch

        from .checkpoint import TrainingCheckpoint as FoundationCheckpoint
        from .checkpoint import TrainingCheckpointStore

        model = self._trained_model
        named = {id(parameter): name for name, parameter in model.named_parameters() if parameter.requires_grad}
        parameter_names = sorted(named.values())
        optimizer_tensors = {}
        tensor_parameter_names = {}
        for parameter, state in self._optimizer.state.items():
            parameter_name = named.get(id(parameter))
            if parameter_name is None:
                raise TrainingIntegrityError("optimizer state references a non-LoRA parameter")
            for field, value in sorted(state.items()):
                if torch.is_tensor(value):
                    if value.is_floating_point() and value.dtype is not torch.float32:
                        raise TrainingIntegrityError("optimizer floating state is not the frozen FP32 exception")
                    tensor_name = "{}.{}".format(parameter_name, field)
                    optimizer_tensors[tensor_name] = value.detach().cpu()
                    tensor_parameter_names[tensor_name] = parameter_name
        if not optimizer_tensors:
            raise TrainingIntegrityError("production optimizer emitted no resumable tensor state")
        groups = []
        for group in self._optimizer.param_groups:
            group_names = sorted(named[id(parameter)] for parameter in group["params"])
            groups.append({
                "parameter_names": group_names,
                "lr": float(group["lr"]),
                "betas": [float(item) for item in group["betas"]],
                "eps": float(group["eps"]),
                "weight_decay": float(group["weight_decay"]),
            })
        py_state = random.getstate()
        np_state = np.random.get_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        rng_tensors = {"torch_cpu_rng": torch.get_rng_state().detach().cpu()}
        for index, value in enumerate(cuda_states):
            rng_tensors["torch_cuda_rng.{}".format(index)] = value.detach().cpu()
        scheduler_state = self._scheduler.state_dict()
        tokenizer_digest = digest_for({
            "class": type(self.tokenizer).__name__,
            "name_or_path": str(getattr(self.tokenizer, "name_or_path", "local-pinned")),
            "vocab_size": int(getattr(self.tokenizer, "vocab_size", 0)),
        })
        versions = {}
        for package in ("torch", "transformers", "peft", "safetensors", "numpy"):
            try:
                versions[package] = importlib_metadata.version(package)
            except importlib_metadata.PackageNotFoundError as exc:
                raise TrainingDependencyError("production software manifest is incomplete") from exc
        checkpoint = FoundationCheckpoint(
            campaign_id=self.frozen_dataset.cutoff.campaign_id,
            run_id=content_id("training-run", {
                "dataset": self.frozen_dataset.digest,
                "protocol": self.protocol.digest,
                "target": self.target_manifest.digest,
            }),
            global_step=self._global_step,
            frozen_dataset_digest=self.frozen_dataset.digest,
            target_manifest_digest=self.target_manifest.digest,
            tokenizer_manifest_digest=tokenizer_digest,
            software_manifest_digest=digest_for(versions),
            base_model_manifest_hash=self.inputs.model_digest,
            base_snapshot_digest=self._base_snapshot.digest,
            ledger_cutoff_hash=self.frozen_dataset.cutoff.digest,
            protocol_hash=self.protocol.digest,
            data_cursor={
                "schema_version": "egv-training-cursor-v1", "epoch": self._epoch_count,
                "row_index": 0, "consumed_examples": self._epoch_count * len(self.inputs.train_rows),
                "sampler_seed": self.protocol.train_seed, "dataset_digest": self.frozen_dataset.digest,
            },
            scheduler_state={
                "schema_version": "egv-training-scheduler-v1", "scheduler_class": type(self._scheduler).__name__,
                "last_epoch": max(0, int(scheduler_state.get("last_epoch", 0))),
                "step_count": self._global_step,
                "base_lrs": [float(item) for item in scheduler_state.get("base_lrs", [self.protocol.learning_rate])],
                "last_lrs": [float(item) for item in self._scheduler.get_last_lr()],
            },
            rng_state={
                "schema_version": "egv-training-rng-v1",
                "python_state": {"version": py_state[0], "state": list(py_state[1]), "gauss": py_state[2]},
                "numpy_state": {
                    "bit_generator": str(np_state[0]), "keys": [int(item) for item in np_state[1]],
                    "position": int(np_state[2]), "has_gauss": int(np_state[3]),
                    "cached_gaussian": float(np_state[4]),
                },
                "torch_cpu_tensor_name": "torch_cpu_rng",
                "torch_cuda_tensor_names": ["torch_cuda_rng.{}".format(index) for index in range(len(cuda_states))],
                "cuda_device_count": len(cuda_states),
            },
            optimizer_metadata={
                "schema_version": "egv-training-optimizer-v1", "optimizer_class": type(self._optimizer).__name__,
                "parameter_names": parameter_names, "parameter_groups": groups,
                "tensor_parameter_names": tensor_parameter_names,
            },
            parent_artifact_digest=parent_artifact_digest,
        )
        adapter_weights = {
            name: value.detach().cpu() for name, value in model.state_dict().items() if "lora_" in name
        }
        _path, artifact_digest = TrainingCheckpointStore(Path(output_root)).save(
            checkpoint, adapter_weights=adapter_weights, optimizer_tensors=optimizer_tensors, rng_tensors=rng_tensors
        )
        return artifact_digest

    def seal_adapter(self, output_root: Path) -> Mapping[str, Any]:
        """Persist and re-open the real PEFT adapter as a sealed Variation artifact."""

        if not self.production or not hasattr(self, "_trained_model"):
            raise TrainingIntegrityError("adapter sealing requires a completed production run")
        destination = Path(output_root)
        if destination.exists() and any(destination.iterdir()):
            raise TrainingIntegrityError("adapter output directory must be absent or empty")
        destination.mkdir(parents=True, exist_ok=True)
        try:
            self._trained_model.save_pretrained(str(destination), safe_serialization=True)
        except Exception as exc:
            raise TrainingDependencyError("PEFT could not write the adapter artifact") from exc
        from ..variation.adapter import (
            ADAPTER_MANIFEST_NAME,
            SealedAdapterArtifact,
            build_local_adapter_manifest,
        )

        manifest = build_local_adapter_manifest(destination)
        (destination / ADAPTER_MANIFEST_NAME).write_text(
            canonical_json(manifest.to_dict()) + "\n", encoding="utf-8"
        )
        artifact = SealedAdapterArtifact(destination)
        artifact.verify()
        approved_adapter_digest = self._selected_evaluation.receipt.get("candidate_artifact_digest")
        if artifact.digest != approved_adapter_digest:
            raise TrainingIntegrityError(
                "published adapter digest differs from the evaluator-approved selected adapter"
            )
        from ..variation.model import PinnedModelLoader
        try:
            import torch
            reloaded = PinnedModelLoader(self.model_root).load(
                device=self.protocol.device,
                torch_dtype=torch.bfloat16,
                adapter_artifact=artifact,
            )
        except Exception as exc:
            raise TrainingIntegrityError("sealed adapter failed real pinned-model reload") from exc
        attestation = reloaded.adapter_attestation
        if (
            attestation is None
            or attestation.adapter_digest != artifact.digest
            or attestation.base_model_manifest_digest != self.inputs.model_digest
            or reloaded.adapter_digest != artifact.digest
        ):
            raise TrainingIntegrityError("reloaded adapter attestation differs from the selected sealed artifact")
        checkpoint_digest = self._selected_checkpoint_artifact_digest
        return {
            "adapter_manifest_digest": artifact.digest,
            "target_manifest_digest": self._target_manifest.digest,
            "base_immutability_proof_digest": self._immutability_proof["snapshot_digest"],
            "adapter_root": str(destination.resolve()),
            "checkpoint_artifact_digest": checkpoint_digest,
        }


class _FixtureTokenizer:
    eos_token_id = 0

    def __call__(self, text: str, *, add_special_tokens: bool, truncation: bool) -> Mapping[str, Sequence[int]]:
        if add_special_tokens or truncation:
            raise AssertionError("fixture tokenizer received a mutable tokenization mode")
        return {"input_ids": [index + 1 for index, _ in enumerate(text.split())] or [1]}


class _FixtureModel:
    def __init__(self) -> None:
        self.steps = 0
        self._state = b"fixture-state-0"

    def state_dict(self) -> Mapping[str, bytes]:
        return {"fixture.weight": self._state}

    def train_step(self, _example: TokenizedTrainingExample) -> float:
        self.steps += 1
        self._state = "fixture-state-{}".format(self.steps).encode("ascii")
        return 1.0 / float(self.steps)


def run_training_smoke() -> Dict[str, Any]:
    """Run a CPU-only fixture smoke without claiming Qwen or promotion."""

    protocol = TrainingProtocol()
    train_rows = (
        TrainingRow.create(
            task_id="egv-pure_function-train-1-v1",
            task_family="PURE_FUNCTION",
            split="train",
            prompt="repair the pure function",
            target="return the corrected result",
        ),
        TrainingRow.create(
            task_id="egv-pure_function-train-2-v1",
            task_family="PURE_FUNCTION",
            split="train",
            prompt="repair the second pure function",
            target="return the corrected result",
        ),
    )
    development_rows = (
        TrainingRow.create(
            task_id="egv-pure_function-dev-1-v1",
            task_family="PURE_FUNCTION",
            split="dev",
            prompt="development prompt",
            target="development target",
        ),
    )
    from .protocol import TrainingDataManifest

    manifest = TrainingDataManifest.from_rows(train_rows, development_rows)
    model = _FixtureModel()
    model_digest = digest_for("fixture-base-model")

    def evaluate(_model: Any, _rows: Sequence[TrainingRow], _protocol: TrainingProtocol) -> float:
        return 1.0 / float(getattr(_model, "steps", 0) + 1)

    signer = ReceiptSigner(b"egv-training-fixture-signer-32bytes"[:32])
    gateway = DevelopmentLossGateway(
        development_rows,
        data_manifest=manifest,
        model_digest=model_digest,
        protocol=protocol,
        signer=signer,
        evaluator=evaluate,
        production=False,
    )
    inputs = seal_training_inputs(
        model,
        model_digest=model_digest,
        train_rows=train_rows,
        data_manifest=manifest,
        protocol=protocol,
        fixture_only=True,
    )
    report = LoRATrainer(inputs, protocol, gateway, _FixtureTokenizer(), production=False).run()
    result = report.to_dict()
    result.update(
        {
            "smoke": "PASS",
            "runtime_tier": "floor-cpu-training-fixture",
            "real_qwen_execution_claimed": False,
            "promotion_claimed": False,
            "private_development_content_published": False,
            "limitations": [
                "CPU fixture only; the pinned Qwen checkpoint was not loaded",
                "PEFT production execution is fail-closed and unclaimed here",
                "no Variation promotion is emitted by the Training slice",
            ],
        }
    )
    return result


def _load_frozen_runtime_dataset(path: Path) -> Any:
    """Load the private export of the canonical ``FrozenTrainingDataset``."""

    from .contracts import FrozenTrainingDataset, LedgerCutoff, TrainingExample

    try:
        value = __import__("json").loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrainingConfigurationError("sealed training dataset cannot be decoded") from exc
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "manifest", "private_rows"}:
        raise TrainingConfigurationError("sealed training dataset is not a closed artifact")
    if value["schema_version"] != SEALED_RUNTIME_DATASET_SCHEMA:
        raise TrainingConfigurationError("sealed training dataset schema is unsupported")
    manifest = value["manifest"]
    rows = value["private_rows"]
    if not isinstance(manifest, Mapping) or not isinstance(rows, list):
        raise TrainingConfigurationError("sealed training dataset payload is malformed")
    cutoff_value = manifest.get("cutoff")
    if not isinstance(cutoff_value, Mapping):
        raise TrainingConfigurationError("sealed training dataset cutoff is missing")
    try:
        cutoff = LedgerCutoff(**dict(cutoff_value))
        examples = []
        for row in rows:
            if not isinstance(row, Mapping) or "sft_row" not in row:
                raise TrainingConfigurationError("sealed training private row is malformed")
            payload = dict(row)
            sft_row = payload.pop("sft_row")
            schema_version = payload.pop("schema_version", None)
            if schema_version != "egv-private-training-example-v1":
                raise TrainingConfigurationError("sealed training private row schema is unsupported")
            sft_json = canonical_json(sft_row)
            payload["sft_row_json"] = sft_json
            examples.append(TrainingExample(**payload))
        dataset = FrozenTrainingDataset(
            cutoff=cutoff,
            examples=tuple(examples),
            excluded_counts=manifest.get("excluded_counts", {}),
        )
    except (TypeError, ValueError) as exc:
        raise TrainingIntegrityError("sealed canonical training dataset failed validation") from exc
    if dataset.manifest() != dict(manifest):
        raise TrainingIntegrityError("sealed training private rows differ from the frozen manifest")
    tasks = sorted({row.task_id for row in dataset.examples})
    if len(tasks) != 20:
        raise TrainingIntegrityError("production selection requires exactly 20 frozen training tasks")
    return dataset


def run_production_training(
    *,
    model_root: Path,
    training_dataset: Path,
    evaluator_manifest: Path,
    evaluator_public_key: Path,
    evaluator_command: Path,
    output_root: Path,
    device: str = "cuda",
) -> Mapping[str, Any]:
    """Execute the bounded real-Qwen LoRA lane from sealed local artifacts."""

    if device != "cuda":
        raise TrainingConfigurationError("production LoRA training is frozen to the CUDA device")
    if sys.version_info < (3, 10):
        raise TrainingDependencyError("production LoRA training requires Python >=3.10")
    from ..variation.model import PinnedModelLoader
    from .development import ExternalDevelopmentLossGateway
    from .protocol import TrainingDataManifest

    dataset = _load_frozen_runtime_dataset(training_dataset)
    protocol = TrainingProtocol()
    gateway = ExternalDevelopmentLossGateway(
        evaluator_manifest, public_key_path=evaluator_public_key, command=evaluator_command
    )
    try:
        import torch
    except ImportError as exc:
        raise TrainingDependencyError("production LoRA training requires torch") from exc
    if not torch.cuda.is_available():
        raise TrainingDependencyError("production LoRA training requires an available CUDA device")
    loaded = PinnedModelLoader(model_root).load(device=device, torch_dtype=torch.bfloat16)
    if gateway.model_digest != loaded.manifest_digest:
        raise TrainingIntegrityError("external evaluator and pinned model manifest differ")
    # The canonical dataset remains the authority. The deterministic policy
    # chooses the earliest accepted B/D trajectory for each of the 20 tasks.
    selected = []
    for task_id in sorted({row.task_id for row in dataset.examples}):
        candidates = [row for row in dataset.examples if row.task_id == task_id]
        selected.append(min(candidates, key=lambda row: (row.attempt_index, row.arm_id, row.seed, row.candidate_id)))
    rows = tuple(
        TrainingRow.create(
            task_id=row.task_id,
            task_family=__import__("json").loads(row.sft_row_json)["task_family"],
            split="train",
            prompt=row.prompt,
            target=row.target,
            source_event_ids=(row.sft_row_digest,),
            row_id=row.row_id,
        )
        for row in selected
    )
    ordered = tuple(sorted(rows, key=lambda row: row.row_id))
    train_digest = __import__("egv.canonical", fromlist=["collection_digest"]).collection_digest(
        row.public_binding() for row in ordered
    )
    runtime_manifest = TrainingDataManifest(
        train_row_ids=tuple(row.row_id for row in ordered),
        development_row_ids=gateway.development_row_ids,
        train_digest=train_digest,
        development_digest=gateway.development_manifest_digest,
    )
    inputs = seal_training_inputs(
        loaded,
        model_digest=loaded.manifest_digest,
        train_rows=ordered,
        data_manifest=runtime_manifest,
        protocol=protocol,
        fixture_only=False,
    )
    from .targets import build_lora_target_manifest

    target_manifest = build_lora_target_manifest(
        loaded.model, model_manifest_digest=loaded.manifest_digest
    )
    trainer = LoRATrainer(
        inputs, protocol, gateway, loaded.tokenizer, production=True, model_root=model_root,
        frozen_dataset=dataset, target_manifest=target_manifest,
        checkpoint_root=Path(output_root) / "checkpoints",
    )
    report = trainer.run()
    sealed = trainer.seal_adapter(Path(output_root) / "adapter")
    return {
        **report.to_dict(),
        **sealed,
        "training_dataset_digest": dataset.digest,
        "selected_training_row_count": len(selected),
        "selection_policy": "EARLIEST_ACCEPTED_BD_PER_TASK_V1",
    }


__all__ = [
    "LoRATrainer",
    "TRAINING_RUNTIME_SCHEMA",
    "TrainingRunReport",
    "run_training_smoke",
    "run_production_training",
]
