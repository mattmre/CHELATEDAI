"""Observation-only hook bus for Model-Scope activation capture."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch

from chelation_logger import get_logger


MODEL_SCOPE_HOOK_SCHEMA_VERSION = 1


@dataclass
class HookObservationConfig:
    """Configuration for residual-stream capture and summary emission."""

    layer_indices: List[int] | None = None
    summary_top_dimensions: int = 8
    hook_target: str = "residual_stream"
    model_family: str = "generic_transformer"
    artifact_type: str = "model_scope_observation"
    metadata: Dict[str, Any] = field(default_factory=dict)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _resolve_transformer_layers(model: Any) -> Sequence[Any]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("model does not expose transformer layers via .model.layers or .layers")


def _extract_hidden_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    if hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, torch.Tensor):
        return output.last_hidden_state
    raise TypeError(f"unsupported layer output type for hook capture: {type(output).__name__}")


def _normalize_layer_indices(layer_indices: Iterable[int] | None, layer_count: int) -> List[int]:
    if layer_count < 1:
        raise ValueError("model exposes no transformer layers")
    if layer_indices is None:
        return list(range(layer_count))
    normalized = sorted({int(index) for index in layer_indices})
    for index in normalized:
        if index < 0 or index >= layer_count:
            raise ValueError(f"layer index {index} is out of range for {layer_count} layers")
    return normalized


def _activation_summary(
    tensor: torch.Tensor,
    *,
    layer_index: int,
    hook_target: str,
    top_dimensions: int,
) -> Dict[str, Any]:
    value = tensor.detach().float().cpu()
    if value.dim() == 1:
        value = value.unsqueeze(0).unsqueeze(0)
    elif value.dim() == 2:
        value = value.unsqueeze(0)
    if value.dim() < 3:
        raise ValueError(f"expected activation tensor with >= 2 feature axes, got shape {tuple(value.shape)}")

    flat_tokens = value.reshape(-1, value.shape[-1])
    token_norms = torch.linalg.vector_norm(flat_tokens, dim=-1)
    dim_abs_mean = flat_tokens.abs().mean(dim=0)
    top_k = min(max(int(top_dimensions), 0), dim_abs_mean.numel())
    if top_k > 0:
        top_values, top_indices = torch.topk(dim_abs_mean, k=top_k)
        top_dims = [
            {
                "dimension": int(index),
                "mean_abs_activation": float(score),
            }
            for index, score in zip(top_indices.tolist(), top_values.tolist())
        ]
    else:
        top_dims = []

    return {
        "layer_index": int(layer_index),
        "hook_target": hook_target,
        "shape": [int(dim) for dim in value.shape],
        "token_count": int(flat_tokens.shape[0]),
        "sequence_length": int(value.shape[-2]),
        "hidden_size": int(value.shape[-1]),
        "dtype": str(value.dtype),
        "value_mean": float(value.mean().item()),
        "value_std": float(value.std(unbiased=False).item()),
        "abs_mean": float(value.abs().mean().item()),
        "abs_max": float(value.abs().max().item()),
        "token_norm_mean": float(token_norms.mean().item()),
        "token_norm_std": float(token_norms.std(unbiased=False).item()),
        "top_dimensions": top_dims,
    }


class ModelHookBus:
    """Capture residual-stream summaries from selected transformer layers."""

    def __init__(self, config: HookObservationConfig | None = None, logger=None, feature_extractor=None):
        self.config = config or HookObservationConfig()
        self.logger = logger or get_logger()
        self.feature_extractor = feature_extractor
        self._last_capture: Dict[str, Any] | None = None

    def capture(
        self,
        model: Any,
        model_inputs: Mapping[str, Any],
        *,
        model_name: str,
        prompt_text: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        layer_indices: Iterable[int] | None = None,
    ) -> Dict[str, Any]:
        layers = _resolve_transformer_layers(model)
        selected_layers = _normalize_layer_indices(
            layer_indices if layer_indices is not None else self.config.layer_indices,
            len(layers),
        )
        captures: Dict[int, Dict[str, Any]] = {}
        handles = []

        def _make_hook(index: int):
            def _hook(_module: Any, _inputs: Any, output: Any) -> None:
                hidden_tensor = _extract_hidden_tensor(output)
                summary = _activation_summary(
                    hidden_tensor,
                    layer_index=index,
                    hook_target=self.config.hook_target,
                    top_dimensions=self.config.summary_top_dimensions,
                )
                if self.feature_extractor is not None:
                    feature_summary = self.feature_extractor.summarize(
                        layer_index=index,
                        activation=hidden_tensor,
                    )
                    if feature_summary is not None:
                        summary["feature_summary"] = _json_safe(feature_summary)
                captures[index] = summary

            return _hook

        for index in selected_layers:
            handles.append(layers[index].register_forward_hook(_make_hook(index)))

        started = time.time()
        try:
            with torch.no_grad():
                outputs = model(**model_inputs)
        finally:
            for handle in handles:
                handle.remove()

        input_ids = model_inputs.get("input_ids")
        token_count = int(input_ids.shape[-1]) if isinstance(input_ids, torch.Tensor) and input_ids.ndim >= 2 else None
        artifact = {
            "schema_version": MODEL_SCOPE_HOOK_SCHEMA_VERSION,
            "artifact_type": self.config.artifact_type,
            "hook_target": self.config.hook_target,
            "model_family": self.config.model_family,
            "model_name": model_name,
            "prompt_hash": hashlib.sha256(str(prompt_text or "").encode("utf-8")).hexdigest()[:16],
            "prompt_length": len(str(prompt_text or "")),
            "token_count": token_count,
            "layer_indices": list(selected_layers),
            "captured_layer_count": len(captures),
            "capture_runtime_ms": float((time.time() - started) * 1000.0),
            "observations": [captures[index] for index in selected_layers if index in captures],
            "metadata": _json_safe({**dict(self.config.metadata), **dict(metadata or {})}),
            "model_output_type": type(outputs).__name__,
        }
        self._last_capture = _json_safe(artifact)
        self.logger.log_event(
            "model_scope_capture",
            "Captured Model-Scope residual-stream summaries",
            model_name=model_name,
            layer_count=len(selected_layers),
            token_count=token_count,
            level="DEBUG",
        )
        return dict(self._last_capture)

    def get_last_capture(self) -> Dict[str, Any] | None:
        if self._last_capture is None:
            return None
        return dict(self._last_capture)
