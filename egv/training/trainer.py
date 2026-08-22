"""Bounded LoRA training runtime and deterministic CPU fixture smoke."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import random
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..canonical import digest_for
from ..receipts import ReceiptSigner
from .development import (
    DevelopmentLossEvaluation,
    DevelopmentLossGateway,
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
        if self.real_qwen_execution_claimed:
            raise TrainingIntegrityError("this bounded runtime cannot claim real Qwen execution")


class LoRATrainer:
    """Run the frozen LoRA protocol against sealed inputs and a dev gateway."""

    def __init__(
        self,
        inputs: SealedTrainingInputs,
        protocol: Optional[TrainingProtocol],
        gateway: Optional[DevelopmentLossGateway],
        tokenizer: Any,
        *,
        production: bool = True,
    ) -> None:
        if type(inputs) is not SealedTrainingInputs:
            raise TrainingIntegrityError("LoRATrainer requires sealed Training inputs")
        if type(protocol) is not TrainingProtocol:
            raise TrainingConfigurationError("LoRATrainer requires the exact frozen TrainingProtocol")
        if type(gateway) is not DevelopmentLossGateway:
            raise TrainingDependencyError("LoRATrainer requires the production DevelopmentLossGateway")
        if not callable(tokenizer):
            raise TrainingDependencyError("LoRATrainer requires a tokenizer callable")
        protocol.validate()
        inputs.validate(protocol)
        gateway.validate_production_boundary(
            expected_model_digest=inputs.model_digest,
            expected_protocol_digest=protocol.digest,
        ) if production else self._validate_fixture_gateway(gateway, inputs, protocol)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "gateway", gateway)
        object.__setattr__(self, "tokenizer", tokenizer)
        object.__setattr__(self, "production", bool(production))
        object.__setattr__(
            self,
            "_contract_digest",
            digest_for(
                {
                    "inputs_seal": inputs.seal_digest,
                    "protocol_digest": protocol.digest,
                    "gateway_digest": gateway.gateway_digest,
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
        config = peft.LoraConfig(
            r=self.protocol.lora_rank,
            lora_alpha=self.protocol.lora_alpha,
            lora_dropout=self.protocol.lora_dropout,
            target_modules=list(self.protocol.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        try:
            model = peft.get_peft_model(base_model, config)
        except Exception as exc:
            raise TrainingDependencyError("the frozen PEFT LoRA configuration could not be applied") from exc
        trainable = []
        named_parameters = getattr(model, "named_parameters", None)
        if not callable(named_parameters):
            raise TrainingDependencyError("production LoRA model lacks named_parameters()")
        for name, parameter in named_parameters():
            if bool(getattr(parameter, "requires_grad", False)):
                trainable.append(str(name))
        if not trainable or any("lora" not in name.lower() for name in trainable):
            raise TrainingIntegrityError("production LoRA application did not isolate adapter parameters")
        return model

    @staticmethod
    def _fixture_train_step(model: Any, example: TokenizedTrainingExample) -> float:
        train_step = getattr(model, "train_step", None)
        if not callable(train_step):
            raise TrainingDependencyError("fixture model must expose train_step()")
        return _require_finite_loss(train_step(example), "fixture train_step")

    def _production_train_step(self, model: Any, example: TokenizedTrainingExample, optimizer: Any) -> float:
        try:
            import torch
        except ImportError as exc:
            raise TrainingDependencyError("production LoRA training requires torch") from exc
        model.train()
        input_ids = torch.tensor([list(example.input_ids)], dtype=torch.long)
        attention_mask = torch.tensor([list(example.attention_mask)], dtype=torch.long)
        labels = torch.tensor([list(example.labels)], dtype=torch.long)
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
            value = _require_finite_loss(float(loss.detach().cpu().item()), "production model")
            loss.backward()
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
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
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
            for example in batches:
                if self.production:
                    loss = self._production_train_step(model, example, optimizer)
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
                no_improvement = 0
            else:
                no_improvement += 1
            if no_improvement > self.protocol.early_stopping_patience or global_step >= self.protocol.max_train_steps:
                break
        if best is None:
            raise TrainingIntegrityError("Training produced no valid development evaluation")
        report = TrainingRunReport(
            protocol_digest=self.protocol.digest,
            model_digest=self.inputs.model_digest,
            data_manifest_digest=self.inputs.data_manifest.digest,
            checkpoint_digests=tuple(checkpoint.digest for checkpoint in checkpoints),
            development_evaluations=tuple(evaluations),
            selected_checkpoint_digest=best.checkpoint_digest,
            selected_loss=best.loss,
            production=self.production,
            promotion_disposition="NOT_PROMOTED_TRAINING_ARTIFACT",
        )
        report.validate()
        return report


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


__all__ = [
    "LoRATrainer",
    "TRAINING_RUNTIME_SCHEMA",
    "TrainingRunReport",
    "run_training_smoke",
]
