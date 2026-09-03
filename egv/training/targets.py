"""Frozen attention-only LoRA target inventory for EGV Training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from ..canonical import canonical_json, digest_for, validate_sha256
from ..variation.model import MODEL_ARCHITECTURE, MODEL_CONFIG_CLASS


FULL_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23)
MODEL_LAYER_COUNT = 24
FULL_ATTENTION_PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)
LINEAR_ATTENTION_PROJECTIONS = (
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
    "linear_attn.out_proj",
)
LORA_TARGET_COUNT = 114
LORA_TARGET_SCHEMA = "egv-qwen35-attention-lora-targets-v1"
LORA_TARGET_MANIFEST_SCHEMA = "egv-qwen35-lora-target-manifest-v1"


class TrainingTargetError(ValueError):
    """The local model does not match the frozen EGV LoRA target contract."""


def frozen_lora_target_names(*, layer_prefix: str = "model.layers") -> Tuple[str, ...]:
    """Return the ordered, fully-qualified 114-module target allowlist."""

    if not isinstance(layer_prefix, str) or not layer_prefix or layer_prefix.startswith(".") or layer_prefix.endswith("."):
        raise TrainingTargetError("layer prefix must be a non-empty dotted module path")
    names = []
    full_layers = frozenset(FULL_ATTENTION_LAYERS)
    for layer in range(MODEL_LAYER_COUNT):
        projections = FULL_ATTENTION_PROJECTIONS if layer in full_layers else LINEAR_ATTENTION_PROJECTIONS
        names.extend("{}.{}.{}".format(layer_prefix, layer, projection) for projection in projections)
    result = tuple(names)
    if len(result) != LORA_TARGET_COUNT or len(set(result)) != LORA_TARGET_COUNT:
        raise AssertionError("internal EGV LoRA target inventory is not closed")
    return result


FROZEN_LORA_TARGETS = frozen_lora_target_names()


@dataclass(frozen=True)
class TargetValidation:
    schema_version: str
    module_names: Tuple[str, ...]
    module_type: str

    @property
    def count(self) -> int:
        return len(self.module_names)


@dataclass(frozen=True)
class LoraTargetManifest:
    schema_version: str
    model_manifest_digest: str
    architecture: str
    config_class: str
    module_names: Tuple[str, ...]
    module_types: Tuple[str, ...]
    digest: str

    def unsigned_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_manifest_digest": self.model_manifest_digest,
            "architecture": self.architecture,
            "config_class": self.config_class,
            "module_names": list(self.module_names),
            "module_types": list(self.module_types),
        }

    def to_dict(self) -> Mapping[str, Any]:
        return {**self.unsigned_dict(), "digest": self.digest}

    def canonical_json(self) -> str:
        self.verify()
        return canonical_json(self.to_dict())

    def verify(self) -> None:
        if self.schema_version != LORA_TARGET_MANIFEST_SCHEMA:
            raise TrainingTargetError("LoRA target manifest schema is unsupported")
        try:
            validate_sha256(self.model_manifest_digest, "model_manifest_digest")
            validate_sha256(self.digest, "target manifest digest")
        except ValueError as exc:
            raise TrainingTargetError(str(exc)) from exc
        if self.architecture != MODEL_ARCHITECTURE or self.config_class != MODEL_CONFIG_CLASS:
            raise TrainingTargetError("LoRA target manifest is not bound to the pinned model architecture")
        if self.module_names != FROZEN_LORA_TARGETS:
            raise TrainingTargetError("LoRA target manifest names differ from the frozen 114 targets")
        if self.module_types != ("torch.nn.Linear",) * LORA_TARGET_COUNT:
            raise TrainingTargetError("LoRA target manifest types differ from the exact Linear contract")
        if self.digest != digest_for(self.unsigned_dict()):
            raise TrainingTargetError("LoRA target manifest digest is not content-derived")


def build_lora_target_manifest(
    model: Any,
    *,
    model_manifest_digest: str,
    architecture: str = MODEL_ARCHITECTURE,
    config_class: str = MODEL_CONFIG_CLASS,
) -> LoraTargetManifest:
    """Validate and seal the exact target inventory against the pinned model."""

    validation = validate_lora_target_modules(model)
    unsigned = {
        "schema_version": LORA_TARGET_MANIFEST_SCHEMA,
        "model_manifest_digest": validate_sha256(model_manifest_digest, "model_manifest_digest"),
        "architecture": architecture,
        "config_class": config_class,
        "module_names": list(validation.module_names),
        "module_types": [validation.module_type] * validation.count,
    }
    manifest = LoraTargetManifest(
        LORA_TARGET_MANIFEST_SCHEMA,
        unsigned["model_manifest_digest"],
        architecture,
        config_class,
        validation.module_names,
        tuple(unsigned["module_types"]),
        digest_for(unsigned),
    )
    manifest.verify()
    return manifest


def validate_lora_target_modules(model: Any, *, layer_prefix: str = "model.layers") -> TargetValidation:
    """Fail closed unless the checkpoint exposes exactly the frozen Linear targets.

    The scan also rejects target-shaped projections in unexpected layers.  This
    catches architecture drift instead of silently training a partial adapter.
    """

    try:
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - Training installs torch
        raise TrainingTargetError("PyTorch is required to validate LoRA targets") from exc
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        raise TrainingTargetError("base model does not expose named_modules")
    modules: Mapping[str, Any] = dict(named_modules())
    expected = frozen_lora_target_names(layer_prefix=layer_prefix)
    expected_set = frozenset(expected)
    leaf_names = frozenset(
        projection.rsplit(".", 1)[-1]
        for projection in FULL_ATTENTION_PROJECTIONS + LINEAR_ATTENTION_PROJECTIONS
    )
    target_shaped = frozenset(
        name
        for name in modules
        if name.startswith(layer_prefix + ".") and name.rsplit(".", 1)[-1] in leaf_names
    )
    missing = sorted(expected_set - set(modules))
    additional = sorted(target_shaped - expected_set)
    if missing or additional:
        raise TrainingTargetError(
            "LoRA target inventory differs from the frozen checkpoint (missing={}, additional={})".format(
                missing, additional
            )
        )
    wrong_type = sorted(name for name in expected if type(modules[name]) is not nn.Linear)
    if wrong_type:
        raise TrainingTargetError("LoRA targets must be exact torch.nn.Linear modules: {}".format(wrong_type))
    return TargetValidation(LORA_TARGET_SCHEMA, expected, "torch.nn.Linear")


__all__ = [
    "FROZEN_LORA_TARGETS",
    "FULL_ATTENTION_LAYERS",
    "FULL_ATTENTION_PROJECTIONS",
    "LINEAR_ATTENTION_PROJECTIONS",
    "LORA_TARGET_COUNT",
    "LORA_TARGET_SCHEMA",
    "LORA_TARGET_MANIFEST_SCHEMA",
    "MODEL_LAYER_COUNT",
    "TargetValidation",
    "LoraTargetManifest",
    "TrainingTargetError",
    "frozen_lora_target_names",
    "build_lora_target_manifest",
    "validate_lora_target_modules",
]
