"""Local model runtime for Model-Scope observation and artifact emission."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, Mapping

import torch

from chelation_logger import get_logger
from config import ChelationConfig
from model_hook_bus import HookObservationConfig, ModelHookBus, _json_safe
from model_scope_features import FallbackActivationFeatureExtractor
from model_scope_artifacts import (
    MODEL_SCOPE_ARTIFACT_SCHEMA_VERSION,
    build_model_scope_artifact,
    load_model_scope_artifact as _load_model_scope_artifact,
    write_model_scope_artifact,
)


@dataclass
class ModelScopeRuntimeConfig:
    """Configuration for local model loading and hook capture."""

    model_name: str
    device: str | None = None
    max_input_tokens: int = 256
    layer_indices: list[int] | None = None
    summary_top_dimensions: int = 8
    hook_target: str = "residual_stream"
    artifact_dir: str | None = None
    model_family: str = "qwen_like"
    trust_remote_code: bool = False


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_")
    return slug or "model_scope"


class ModelScopeRuntime:
    """Load a local model, capture internal activations, and emit observation artifacts."""

    def __init__(
        self,
        config: ModelScopeRuntimeConfig,
        *,
        logger=None,
        model: Any | None = None,
        tokenizer: Any | None = None,
        model_loader: Callable[[str], Any] | None = None,
        tokenizer_loader: Callable[[str], Any] | None = None,
        hook_bus: ModelHookBus | None = None,
        feature_extractor=None,
        steerer=None,
        memory_store=None,
        expectation_comparator=None,
        eager_load: bool = False,
    ):
        self.config = config
        self.logger = logger or get_logger()
        self.model = model
        self.tokenizer = tokenizer
        self._model_loader = model_loader or self._default_model_loader
        self._tokenizer_loader = tokenizer_loader or self._default_tokenizer_loader
        self._device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._artifact_dir = Path(config.artifact_dir) if config.artifact_dir else None
        resolved_feature_extractor = feature_extractor or FallbackActivationFeatureExtractor(
            top_dimensions=config.summary_top_dimensions
        )
        self._hook_bus = hook_bus or ModelHookBus(
            HookObservationConfig(
                layer_indices=list(config.layer_indices) if config.layer_indices is not None else None,
                summary_top_dimensions=config.summary_top_dimensions,
                hook_target=config.hook_target,
                model_family=config.model_family,
            ),
            logger=self.logger,
            feature_extractor=resolved_feature_extractor,
        )
        self._steerer = steerer
        self._memory_store = memory_store
        self._expectation_comparator = expectation_comparator
        self._last_artifact: Dict[str, Any] | None = None

        if eager_load:
            self.load()

    def _default_model_loader(self, model_name: str) -> Any:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=self.config.trust_remote_code,
        )
        if hasattr(model, "to"):
            model = model.to(self._device)
        if hasattr(model, "eval"):
            model.eval()
        return model

    def _default_tokenizer_loader(self, model_name: str) -> Any:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load(self) -> None:
        if self.model is None:
            self.model = self._model_loader(self.config.model_name)
        if self.tokenizer is None:
            self.tokenizer = self._tokenizer_loader(self.config.model_name)
        self.logger.log_event(
            "model_scope_runtime_loaded",
            "Model-Scope runtime loaded local model and tokenizer",
            model_name=self.config.model_name,
            device=self._device,
            level="DEBUG",
        )

    def describe_runtime(self) -> Dict[str, Any]:
        return {
            "schema_version": MODEL_SCOPE_ARTIFACT_SCHEMA_VERSION,
            "model_name": self.config.model_name,
            "device": self._device,
            "max_input_tokens": int(self.config.max_input_tokens),
            "layer_indices": list(self.config.layer_indices) if self.config.layer_indices is not None else None,
            "summary_top_dimensions": int(self.config.summary_top_dimensions),
            "artifact_dir": str(self._artifact_dir) if self._artifact_dir is not None else None,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "memory_enabled": self._memory_store is not None,
            "expectation_comparator_enabled": self._expectation_comparator is not None,
        }

    def _prepare_inputs(self, text: str) -> Dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("tokenizer is not loaded")
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_tokens,
        )
        inputs = dict(encoded)
        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(self._device)
        return inputs

    def _artifact_output_path(self) -> Path | None:
        if self._artifact_dir is None:
            return None
        filename = f"{_safe_slug(self.config.model_name)}-{_timestamp_slug()}.json"
        return self._artifact_dir / filename

    def observe_text(
        self,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        output_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        self.load()
        inputs = self._prepare_inputs(text)
        capture = self._hook_bus.capture(
            self.model,
            inputs,
            model_name=self.config.model_name,
            prompt_text=text,
            metadata=metadata,
            layer_indices=self.config.layer_indices,
        )
        artifact = build_model_scope_artifact(runtime=self.describe_runtime(), capture=capture)
        if self._steerer is not None:
            artifact["steering"] = self._steerer.evaluate_capture(artifact)
        expectation_profile_id = None if metadata is None else metadata.get("expectation_profile_id")
        if (
            expectation_profile_id is not None
            and self._memory_store is not None
            and self._expectation_comparator is not None
        ):
            profile = self._memory_store.get_expectation_profile(str(expectation_profile_id))
            if profile is not None:
                artifact["expectation_comparison"] = self._expectation_comparator.compare_to_profile(
                    artifact,
                    profile,
                    candidate_id=str(expectation_profile_id),
                )
        if self._memory_store is not None:
            artifact["memory"] = self._memory_store.record_observation(
                artifact,
                query_text=text,
                metadata=metadata,
                expectation_profile_id=expectation_profile_id,
            )
        target_path = Path(output_path) if output_path is not None else self._artifact_output_path()
        if target_path is not None:
            written = write_model_scope_artifact(target_path, artifact)
            artifact["output_path"] = str(written)
        self._last_artifact = _json_safe(artifact)
        return dict(self._last_artifact)

    def get_last_artifact(self) -> Dict[str, Any] | None:
        if self._last_artifact is None:
            return None
        return dict(self._last_artifact)

    def close(self) -> None:
        if self.model is not None and hasattr(self.model, "cpu"):
            try:
                self.model.cpu()
            except Exception:
                pass
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_model_scope_artifact(path: str | Path) -> Dict[str, Any]:
    """Load a versioned Model-Scope artifact from disk."""

    return _load_model_scope_artifact(path)


def create_model_scope_runtime(
    model_name: str | None = None,
    *,
    layer_indices: Iterable[int] | None = None,
    max_input_tokens: int | None = None,
    summary_top_dimensions: int | None = None,
    artifact_dir: str | None = None,
    eager_load: bool = False,
    logger=None,
) -> ModelScopeRuntime:
    config = ModelScopeRuntimeConfig(
        model_name=model_name or ChelationConfig.MODEL_SCOPE_DEBUG_MODEL_NAME,
        layer_indices=list(layer_indices) if layer_indices is not None else None,
        max_input_tokens=max_input_tokens or ChelationConfig.MODEL_SCOPE_MAX_INPUT_TOKENS,
        summary_top_dimensions=summary_top_dimensions or ChelationConfig.MODEL_SCOPE_SUMMARY_TOP_DIMENSIONS,
        artifact_dir=artifact_dir or str(ChelationConfig.MODEL_SCOPE_ARTIFACT_ROOT),
    )
    return ModelScopeRuntime(config, logger=logger, eager_load=eager_load)
