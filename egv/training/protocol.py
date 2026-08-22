"""Frozen, content-addressed contracts for the EGV Training slice.

The training runtime is deliberately stricter than a generic Transformers
wrapper.  A protocol digest identifies every optimizer, batching, precision,
seed, evaluation, and LoRA choice.  Training rows are private inputs; their
public bindings contain only content digests and bounded identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ..canonical import collection_digest, content_id, digest_for, validate_sha256
from ..errors import EGVError


TRAINING_PROTOCOL_SCHEMA = "egv-training-protocol-v1"
TRAINING_ROW_SCHEMA = "egv-training-row-v1"
TRAINING_MANIFEST_SCHEMA = "egv-training-data-manifest-v1"
TRAINING_INPUT_SCHEMA = "egv-sealed-training-input-v1"


class TrainingError(EGVError):
    """Base class for bounded Training-slice failures."""


class TrainingConfigurationError(TrainingError, ValueError):
    """A frozen Training contract or input is invalid."""


class TrainingDependencyError(TrainingError):
    """A required production dependency or gateway is unavailable."""


class TrainingIntegrityError(TrainingError):
    """A signed binding, sealed input, or immutable artifact is invalid."""


class TrainingLeakageError(TrainingError):
    """A development/held-out/private boundary was crossed."""


def _target_modules() -> Tuple[str, ...]:
    full_attention_layers = (3, 7, 11, 15, 19, 23)
    full_attention_names = ("q_proj", "k_proj", "v_proj", "o_proj")
    linear_attention_names = (
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_a",
        "in_proj_b",
        "out_proj",
    )
    values = [
        "model.layers.{}.self_attn.{}".format(layer, name)
        for layer in full_attention_layers
        for name in full_attention_names
    ]
    values.extend(
        "model.layers.{}.linear_attn.{}".format(layer, name)
        for layer in range(24)
        if layer not in full_attention_layers
        for name in linear_attention_names
    )
    return tuple(values)


LORA_TARGET_MODULES = _target_modules()


@dataclass(frozen=True)
class TrainingProtocol:
    """The complete initial LoRA training protocol.

    Defaults are the only accepted values in this slice.  Keeping all values
    in one immutable record prevents a caller from changing a scheduler or
    seed while retaining an old protocol digest.
    """

    schema_version: str = TRAINING_PROTOCOL_SCHEMA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    max_epochs: int = 3
    max_sequence_length: int = 4096
    packing: bool = False
    optimizer: str = "adamw_torch"
    scheduler: str = "linear"
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    device: str = "cuda"
    precision: str = "bfloat16"
    trainable_adapter_precision: str = "float32"
    optimizer_state_precision: str = "float32"
    eval_cadence: str = "epoch"
    eval_every_n_steps: int = 1
    early_stopping_patience: int = 1
    train_seed: int = 20260822
    evaluation_seed: int = 20260823
    max_train_rows: int = 20
    max_train_steps: int = 60
    target_modules: Tuple[str, ...] = field(default_factory=lambda: LORA_TARGET_MODULES)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "max_sequence_length": self.max_sequence_length,
            "packing": self.packing,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "per_device_batch_size": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "device": self.device,
            "precision": self.precision,
            "trainable_adapter_precision": self.trainable_adapter_precision,
            "optimizer_state_precision": self.optimizer_state_precision,
            "eval_cadence": self.eval_cadence,
            "eval_every_n_steps": self.eval_every_n_steps,
            "early_stopping_patience": self.early_stopping_patience,
            "train_seed": self.train_seed,
            "evaluation_seed": self.evaluation_seed,
            "max_train_rows": self.max_train_rows,
            "max_train_steps": self.max_train_steps,
            "target_modules": list(self.target_modules),
        }

    @property
    def digest(self) -> str:
        self.validate()
        return digest_for(self.to_dict())

    def validate(self) -> None:
        expected = TrainingProtocol()
        if self.schema_version != TRAINING_PROTOCOL_SCHEMA:
            raise TrainingConfigurationError("unsupported Training protocol schema")
        if self.target_modules != expected.target_modules:
            raise TrainingConfigurationError("LoRA target module allowlist differs from the frozen protocol")
        integer_fields = (
            "lora_rank",
            "lora_alpha",
            "max_epochs",
            "max_sequence_length",
            "per_device_batch_size",
            "gradient_accumulation_steps",
            "eval_every_n_steps",
            "early_stopping_patience",
            "train_seed",
            "evaluation_seed",
            "max_train_rows",
            "max_train_steps",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TrainingConfigurationError("{} must be an integer".format(name))
        float_fields = ("lora_dropout", "learning_rate", "weight_decay", "warmup_ratio")
        for name in float_fields:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise TrainingConfigurationError("{} must be a finite number".format(name))
        if self != expected:
            mismatched = [
                name
                for name in self.to_dict()
                if self.to_dict()[name] != expected.to_dict()[name]
            ]
            raise TrainingConfigurationError(
                "Training protocol drift is not permitted: {}".format(", ".join(sorted(mismatched)))
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrainingProtocol":
        required = set(cls().to_dict())
        if set(value) != required:
            raise TrainingConfigurationError("Training protocol has an unexpected field set")
        payload = dict(value)
        target_modules = payload.get("target_modules")
        if not isinstance(target_modules, (list, tuple)):
            raise TrainingConfigurationError("target_modules must be an ordered array")
        payload["target_modules"] = tuple(target_modules)
        result = cls(**payload)
        result.validate()
        return result


def _require_nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise TrainingConfigurationError("{} must be a non-empty text value".format(field_name))
    return value


@dataclass(frozen=True)
class TrainingRow:
    """One private prompt/completion pair used only by the training path."""

    row_id: str
    task_id: str
    task_family: str
    split: str
    prompt: str
    target: str
    source_event_ids: Tuple[str, ...] = ()
    schema_version: str = TRAINING_ROW_SCHEMA

    def validate(self, *, expected_split: Optional[str] = None) -> None:
        if self.schema_version != TRAINING_ROW_SCHEMA:
            raise TrainingConfigurationError("unsupported Training row schema")
        for value, name in (
            (self.row_id, "row_id"),
            (self.task_id, "task_id"),
            (self.task_family, "task_family"),
            (self.prompt, "prompt"),
            (self.target, "target"),
        ):
            _require_nonempty_text(value, name)
        if self.split not in {"train", "dev"}:
            if self.split == "heldout" or "heldout" in self.task_id.lower():
                raise TrainingLeakageError("held-out content cannot enter the Training runtime")
            raise TrainingConfigurationError("Training rows must be train or dev rows")
        if expected_split is not None and self.split != expected_split:
            raise TrainingConfigurationError("Training row split does not match the requested boundary")
        if "heldout" in self.task_id.lower():
            raise TrainingLeakageError("held-out task identity is forbidden in Training inputs")
        if tuple(sorted(set(self.source_event_ids))) != self.source_event_ids:
            raise TrainingConfigurationError("source_event_ids must be sorted and unique")
        if not all(isinstance(item, str) and item for item in self.source_event_ids):
            raise TrainingConfigurationError("source_event_ids must contain non-empty IDs")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "row_id": self.row_id,
            "task_id": self.task_id,
            "task_family": self.task_family,
            "split": self.split,
            "prompt": self.prompt,
            "target": self.target,
            "source_event_ids": list(self.source_event_ids),
        }

    @property
    def digest(self) -> str:
        return digest_for(self.to_dict())

    def public_binding(self) -> Dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "row_id": self.row_id,
            "task_id": self.task_id,
            "task_family": self.task_family,
            "split": self.split,
            "prompt_digest": digest_for(self.prompt),
            "target_digest": digest_for(self.target),
            "row_digest": self.digest,
            "source_event_ids": list(self.source_event_ids),
        }

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        task_family: str,
        split: str,
        prompt: str,
        target: str,
        source_event_ids: Sequence[str] = (),
        row_id: Optional[str] = None,
    ) -> "TrainingRow":
        normalized_ids = tuple(sorted(set(source_event_ids)))
        body = {
            "task_id": task_id,
            "task_family": task_family,
            "split": split,
            "prompt": prompt,
            "target": target,
            "source_event_ids": list(normalized_ids),
        }
        return cls(
            row_id=row_id or content_id("train-row", body),
            task_id=task_id,
            task_family=task_family,
            split=split,
            prompt=prompt,
            target=target,
            source_event_ids=normalized_ids,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrainingRow":
        required = {"schema_version", "row_id", "task_id", "task_family", "split", "prompt", "target", "source_event_ids"}
        if set(value) != required:
            raise TrainingConfigurationError("Training row has an unexpected field set")
        source_ids = value["source_event_ids"]
        if not isinstance(source_ids, (list, tuple)):
            raise TrainingConfigurationError("source_event_ids must be an array")
        row = cls(
            row_id=value["row_id"],
            task_id=value["task_id"],
            task_family=value["task_family"],
            split=value["split"],
            prompt=value["prompt"],
            target=value["target"],
            source_event_ids=tuple(source_ids),
            schema_version=value["schema_version"],
        )
        row.validate()
        return row


def _split_digest(rows: Iterable[TrainingRow], split: str) -> Tuple[Tuple[TrainingRow, ...], str]:
    normalized = tuple(rows)
    for row in normalized:
        if type(row) is not TrainingRow:
            raise TrainingConfigurationError("Training manifests require exact TrainingRow values")
        row.validate(expected_split=split)
    row_ids = [row.row_id for row in normalized]
    if len(row_ids) != len(set(row_ids)):
        raise TrainingConfigurationError("Training row IDs must be unique within a split")
    ordered = tuple(sorted(normalized, key=lambda row: row.row_id))
    return ordered, collection_digest(row.public_binding() for row in ordered)


@dataclass(frozen=True)
class TrainingDataManifest:
    """A public binding for private train/development row content."""

    train_row_ids: Tuple[str, ...]
    development_row_ids: Tuple[str, ...]
    train_digest: str
    development_digest: str
    schema_version: str = TRAINING_MANIFEST_SCHEMA

    @classmethod
    def from_rows(cls, train_rows: Iterable[TrainingRow], development_rows: Iterable[TrainingRow]) -> "TrainingDataManifest":
        train, train_digest = _split_digest(train_rows, "train")
        development, development_digest = _split_digest(development_rows, "dev")
        train_ids = tuple(row.row_id for row in train)
        development_ids = tuple(row.row_id for row in development)
        overlap = set(train_ids) & set(development_ids)
        if overlap:
            raise TrainingLeakageError("train and development rows share IDs: {}".format(sorted(overlap)))
        if set(row.digest for row in train) & set(row.digest for row in development):
            raise TrainingLeakageError("train and development rows share content")
        return cls(train_ids, development_ids, train_digest, development_digest)

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "train_row_ids": list(self.train_row_ids),
            "development_row_ids": list(self.development_row_ids),
            "train_digest": self.train_digest,
            "development_digest": self.development_digest,
        }

    @property
    def digest(self) -> str:
        self.validate()
        return digest_for(self.to_dict())

    def validate(self) -> None:
        if self.schema_version != TRAINING_MANIFEST_SCHEMA:
            raise TrainingConfigurationError("unsupported Training data-manifest schema")
        if not self.train_row_ids or not self.development_row_ids:
            raise TrainingConfigurationError("Training data manifest must bind train and development rows")
        if tuple(sorted(self.train_row_ids)) != self.train_row_ids or tuple(sorted(self.development_row_ids)) != self.development_row_ids:
            raise TrainingConfigurationError("Training manifest row IDs must be sorted")
        if set(self.train_row_ids) & set(self.development_row_ids):
            raise TrainingLeakageError("Training manifest train/development IDs overlap")
        for name in ("train_digest", "development_digest"):
            try:
                validate_sha256(getattr(self, name), name)
            except Exception as exc:
                raise TrainingConfigurationError(str(exc)) from exc


_INPUT_SEAL_TOKEN = object()


@dataclass(frozen=True)
class SealedTrainingInputs:
    """Immutable input envelope required by :class:`LoRATrainer`."""

    model: Any = field(repr=False, compare=False)
    model_digest: str
    train_rows: Tuple[TrainingRow, ...]
    data_manifest: TrainingDataManifest
    protocol_digest: str
    seal_digest: str
    fixture_only: bool = False
    schema_version: str = TRAINING_INPUT_SCHEMA
    _seal_token: Any = field(default=None, repr=False, compare=False)

    def validate(self, protocol: TrainingProtocol) -> None:
        if self.schema_version != TRAINING_INPUT_SCHEMA:
            raise TrainingIntegrityError("unsupported sealed Training input schema")
        if self._seal_token is not _INPUT_SEAL_TOKEN:
            raise TrainingIntegrityError("Training inputs were not sealed by the runtime")
        if type(self.data_manifest) is not TrainingDataManifest:
            raise TrainingIntegrityError("Training inputs require the exact immutable data manifest")
        self.data_manifest.validate()
        protocol.validate()
        try:
            validate_sha256(self.model_digest, "model_digest")
            validate_sha256(self.protocol_digest, "protocol_digest")
            validate_sha256(self.seal_digest, "seal_digest")
        except Exception as exc:
            raise TrainingIntegrityError(str(exc)) from exc
        if self.protocol_digest != protocol.digest:
            raise TrainingIntegrityError("sealed Training inputs use a different protocol")
        normalized = tuple(self.train_rows)
        if not normalized or tuple(sorted(row.row_id for row in normalized)) != self.data_manifest.train_row_ids:
            raise TrainingIntegrityError("sealed Training rows do not match the immutable manifest")
        for row in normalized:
            if type(row) is not TrainingRow:
                raise TrainingIntegrityError("sealed Training rows require exact TrainingRow values")
            row.validate(expected_split="train")
        actual_train_digest = collection_digest(
            row.public_binding() for row in sorted(normalized, key=lambda item: item.row_id)
        )
        if actual_train_digest != self.data_manifest.train_digest:
            raise TrainingIntegrityError("sealed Training row content differs from the data manifest")
        expected_seal = digest_for(
            {
                "schema_version": self.schema_version,
                "model_digest": self.model_digest,
                "data_manifest_digest": self.data_manifest.digest,
                "protocol_digest": self.protocol_digest,
                "fixture_only": self.fixture_only,
            }
        )
        if self.seal_digest != expected_seal:
            raise TrainingIntegrityError("sealed Training input digest does not match its content")

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_digest": self.model_digest,
            "data_manifest_digest": self.data_manifest.digest,
            "protocol_digest": self.protocol_digest,
            "train_row_count": len(self.train_rows),
            "fixture_only": self.fixture_only,
            "seal_digest": self.seal_digest,
        }


def seal_training_inputs(
    model: Any,
    *,
    model_digest: str,
    train_rows: Iterable[TrainingRow],
    data_manifest: TrainingDataManifest,
    protocol: TrainingProtocol,
    fixture_only: bool = False,
) -> SealedTrainingInputs:
    protocol.validate()
    data_manifest.validate()
    normalized = tuple(train_rows)
    envelope = SealedTrainingInputs(
        model=model,
        model_digest=model_digest,
        train_rows=normalized,
        data_manifest=data_manifest,
        protocol_digest=protocol.digest,
        seal_digest="0" * 64,
        fixture_only=fixture_only,
        _seal_token=_INPUT_SEAL_TOKEN,
    )
    expected_seal = digest_for(
        {
            "schema_version": envelope.schema_version,
            "model_digest": model_digest,
            "data_manifest_digest": data_manifest.digest,
            "protocol_digest": protocol.digest,
            "fixture_only": fixture_only,
        }
    )
    return SealedTrainingInputs(
        model=model,
        model_digest=model_digest,
        train_rows=normalized,
        data_manifest=data_manifest,
        protocol_digest=protocol.digest,
        seal_digest=expected_seal,
        fixture_only=fixture_only,
        _seal_token=_INPUT_SEAL_TOKEN,
    )


@dataclass(frozen=True)
class TokenizedTrainingExample:
    row_id: str
    input_ids: Tuple[int, ...]
    attention_mask: Tuple[int, ...]
    labels: Tuple[int, ...]

    def validate(self, protocol: TrainingProtocol) -> None:
        protocol.validate()
        if not self.input_ids or len(self.input_ids) != len(self.labels) or len(self.input_ids) != len(self.attention_mask):
            raise TrainingConfigurationError("tokenized Training example has inconsistent lengths")
        if len(self.input_ids) > protocol.max_sequence_length:
            raise TrainingConfigurationError("tokenized Training example exceeds the frozen sequence limit")
        if not any(label != -100 for label in self.labels):
            raise TrainingConfigurationError("tokenized Training example has no target labels")


def _token_ids(tokenizer: Any, text: str) -> Tuple[int, ...]:
    try:
        encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    except TypeError as exc:
        raise TrainingConfigurationError("tokenizer must accept truncation=False without implicit truncation") from exc
    if isinstance(encoded, Mapping):
        values = encoded.get("input_ids")
    else:
        values = encoded
    if isinstance(values, (list, tuple)) and values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise TrainingConfigurationError("tokenizer returned a packed/multi-row sequence")
        values = values[0]
    if not isinstance(values, (list, tuple)) or not values or not all(isinstance(item, int) for item in values):
        raise TrainingConfigurationError("tokenizer did not return a non-empty integer token sequence")
    return tuple(values)


def tokenize_training_row(row: TrainingRow, tokenizer: Any, protocol: TrainingProtocol) -> TokenizedTrainingExample:
    row.validate(expected_split="train")
    protocol.validate()
    prompt_ids = _token_ids(tokenizer, row.prompt)
    target_ids = list(_token_ids(tokenizer, row.target))
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int) and (not target_ids or target_ids[-1] != eos_token_id):
        target_ids.append(eos_token_id)
    total = len(prompt_ids) + len(target_ids)
    if total > protocol.max_sequence_length:
        raise TrainingConfigurationError(
            "Training row {} exceeds {} tokens; truncation is forbidden".format(row.row_id, protocol.max_sequence_length)
        )
    example = TokenizedTrainingExample(
        row_id=row.row_id,
        input_ids=prompt_ids + tuple(target_ids),
        attention_mask=(1,) * total,
        labels=(-100,) * len(prompt_ids) + tuple(target_ids),
    )
    example.validate(protocol)
    return example


def build_training_batch(
    rows: Iterable[TrainingRow], tokenizer: Any, protocol: TrainingProtocol
) -> Tuple[TokenizedTrainingExample, ...]:
    protocol.validate()
    if protocol.packing:
        raise TrainingConfigurationError("sequence packing is forbidden by the frozen Training protocol")
    normalized = tuple(rows)
    if not normalized or len(normalized) > protocol.per_device_batch_size:
        raise TrainingConfigurationError("Training batch exceeds the frozen per-device batch size")
    return tuple(tokenize_training_row(row, tokenizer, protocol) for row in normalized)


__all__ = [
    "LORA_TARGET_MODULES",
    "SealedTrainingInputs",
    "TokenizedTrainingExample",
    "TRAINING_INPUT_SCHEMA",
    "TRAINING_MANIFEST_SCHEMA",
    "TRAINING_PROTOCOL_SCHEMA",
    "TRAINING_ROW_SCHEMA",
    "TrainingConfigurationError",
    "TrainingDataManifest",
    "TrainingError",
    "TrainingIntegrityError",
    "TrainingLeakageError",
    "TrainingProtocol",
    "TrainingRow",
    "TrainingDependencyError",
    "build_training_batch",
    "seal_training_inputs",
    "tokenize_training_row",
]
