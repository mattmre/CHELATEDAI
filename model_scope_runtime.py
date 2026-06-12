"""Local model runtime for observation-only activation capture.

NOTE ON ACTIVATION CAPTURE DESIGN
----------------------------------
``ActivationEvent`` stores activation *statistics* (mean, norm) and the tensor
shape, NOT the full residual tensor.  Storing full tensors for every layer on
every inference step would exhaust memory in long-running observation sessions.
The ``raw_tensor_shape`` field records the exact shape of the captured tensor so
callers know the true dimensionality without holding the data.

If you need the full tensor for downstream processing (e.g. feeding an SAE),
call ``QwenScopeLayerSAE.encode()`` *directly inside the hook* and store only
the sparse output -- see ``qwen_scope_adapter.py`` for the coupling contract.
"""

from __future__ import annotations

import json
import uuid
import warnings
import numpy as np
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


try:
    from chelated_shim_research import (
        promoted_sip_apply,
        bump_stall_counter,
        research_enabled,
        research_preflight_metadata,
    )
except ModuleNotFoundError:
    def promoted_sip_apply(v):
        return np.array(v, dtype=float).copy(), None

    def research_enabled() -> bool:
        return False

    def bump_stall_counter(counter: int, *, has_work: bool) -> int:
        return counter

    def research_preflight_metadata(
        *,
        seam: str,
        stall_count: int,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "research_shim_guard": False,
            "research_stall_count": stall_count,
            "sip_seam": seam,
            **(dict(extra or {})),
        }


@dataclass
class ActivationEvent:
    """One captured activation from a model layer during inference.

    Fields
    ------
    schema_version  : semver string for forward-compat deserialisation.
    model_id        : HuggingFace model identifier used at load time.
    layer_id        : dotted attribute path (e.g. ``model.layers.15``).
    token_count     : number of tokens in the sequence dimension.
    shape           : full tensor shape at the point of capture.
    mean_activation : global mean of the captured tensor (float scalar).
    norm_activation : Frobenius / L2 norm of the captured tensor (float scalar).
    raw_tensor_shape: alias of ``shape`` preserved explicitly so downstream code
                      can document that only the *shape* is retained, not the
                      data itself.  Always equal to ``shape``.
    captured_at     : ISO-8601 UTC timestamp.
    run_id          : UUID string grouping events from one ``run_inference`` call.
    """

    schema_version: str
    model_id: str
    layer_id: str
    token_count: int
    shape: tuple
    mean_activation: float
    norm_activation: float
    captured_at: str
    run_id: str
    # raw_tensor_shape records the actual tensor dimensions at capture time.
    # The full tensor is NOT stored here; only its shape is retained for
    # memory efficiency.  Downstream consumers that need the tensor data must
    # register their own hooks or use QwenScopeLayerSAE.encode() inline.
    raw_tensor_shape: Optional[tuple] = field(default=None)

    def __post_init__(self) -> None:
        # Keep raw_tensor_shape in sync with shape when not explicitly set.
        if self.raw_tensor_shape is None:
            object.__setattr__(self, "raw_tensor_shape", self.shape)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["shape"] = list(d["shape"])
        if d.get("raw_tensor_shape") is not None:
            d["raw_tensor_shape"] = list(d["raw_tensor_shape"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ActivationEvent:
        d = dict(d)
        d["shape"] = tuple(d["shape"])
        if d.get("raw_tensor_shape") is not None:
            d["raw_tensor_shape"] = tuple(d["raw_tensor_shape"])
        return cls(**d)


def _normalize_layer_id(layer_spec: Any) -> str:
    """Convert an integer-like layer spec into the runtime layer path."""
    if isinstance(layer_spec, int):
        return f"model.layers.{layer_spec}"
    if isinstance(layer_spec, str) and layer_spec.strip():
        stripped = layer_spec.strip()
        if stripped.isdigit():
            return f"model.layers.{stripped}"
        return stripped
    raise ValueError(f"invalid layer spec: {layer_spec!r}")


def _resolve_module(model: Any, layer_id: str) -> Any:
    """Traverse dotted layer_id to retrieve a submodule. Returns None if not found."""
    parts = layer_id.split(".")
    obj = model
    for part in parts:
        if hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            return None
    return obj


class LocalModelRuntime:
    """Loads a causal-LM and captures residual-stream activations via hooks."""

    def __init__(
        self,
        model_name: str,
        *,
        hook_layers: list | None = None,
        device: str = "cpu",
        layer_indices: list | None = None,
        max_input_tokens: int = 256,
        summary_top_dimensions: int = 8,
        artifact_dir: str = "experiment_runs/model_scope",
        eager_load: bool = False,
        logger: Any = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._hook_layers: list = list(hook_layers or [])
        self._layer_indices: list = list(layer_indices or [])
        self._max_input_tokens = int(max_input_tokens)
        self._summary_top_dimensions = int(summary_top_dimensions)
        self._artifact_dir = Path(artifact_dir)
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logger
        self._model: Any = None
        self._events: list = []
        self._tokenizer: Any = None
        self._tokenizer_tried = False
        self._research_observe_stall_count: int = 0
        self._last_research_shim_meta: dict[str, Any] | None = None

        for layer_id in self._layer_indices:
            normalized = _normalize_layer_id(layer_id)
            if normalized not in self._hook_layers:
                self._hook_layers.append(normalized)

        if eager_load:
            self.load()

    def load(self, *, model_loader: Callable = None) -> None:
        """Load the model. Accepts an optional loader callable for injection in tests."""
        if model_loader is not None:
            self._model = model_loader(self.model_name)
        else:
            try:
                from transformers import AutoModelForCausalLM  # type: ignore[import]
            except ModuleNotFoundError as exc:
                warnings.warn(
                    "transformers unavailable; falling back to a no-op runtime model for "
                    f"observation-only mode ({type(exc).__name__}: {exc})",
                    UserWarning,
                    stacklevel=2,
                )
                class _FallbackModel:
                    """Minimal stand-in used when optional dependencies are unavailable."""

                    def to(self, *_args, **_kwargs):
                        return self

                    def named_modules(self):
                        return []

                    def __call__(self, *_args, **_kwargs):
                        return None

                self._model = _FallbackModel()
            else:
                self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self._model.to(self.device)
        self._load_default_tokenizer()

    def _load_default_tokenizer(self) -> None:
        if self._tokenizer_tried:
            return
        self._tokenizer_tried = True
        if self._tokenizer is not None:
            return
        try:
            from transformers import AutoTokenizer  # type: ignore[import]

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception:
            self._tokenizer = None

    def register_hook_layers(self, layer_ids: list) -> None:
        """Add layer IDs to the set that will be hooked during inference."""
        for lid in layer_ids:
            normalized = _normalize_layer_id(lid)
            if normalized not in self._hook_layers:
                self._hook_layers.append(normalized)

    def _resolve_token_count(self, input_ids: Any) -> int:
        """Best-effort token-count extraction for model-agnostic inputs."""
        if hasattr(input_ids, "shape"):
            shape = tuple(input_ids.shape)
            if len(shape) >= 2:
                return int(shape[1])
            if len(shape) == 1:
                return int(shape[0])
        if isinstance(input_ids, (list, tuple)):
            if not input_ids:
                return 0
            first = input_ids[0]
            if isinstance(first, (list, tuple)):
                return int(len(first))
            return int(len(input_ids))
        try:
            return int(len(input_ids))
        except Exception:
            return 0

    def _prepare_input_ids(self, query_text: str) -> tuple[Any, int]:
        """Tokenize/fallback text and return (input_ids, token_count)."""
        if isinstance(query_text, bytes):
            query_text = query_text.decode("utf-8", errors="ignore")
        text = str(query_text)

        if self._tokenizer is None:
            self._load_default_tokenizer()

        if self._tokenizer is not None:
            try:
                encoded = self._tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self._max_input_tokens,
                )
                input_ids = encoded["input_ids"]
                token_count = self._resolve_token_count(input_ids)
                return input_ids, token_count
            except Exception as tok_err:
                warnings.warn(
                    f"tokenizer path failed for model_scope runtime; "
                    f"falling back to basic split: {type(tok_err).__name__}: {tok_err}",
                    stacklevel=2,
                )

        # Fallback when tokenizer unavailable (expected in lean test rigs).
        tokens = text.split()
        token_count = min(len(tokens), self._max_input_tokens)
        token_ids = list(range(token_count))
        try:
            import torch  # type: ignore[import]

            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        except Exception:
            input_ids = [token_ids]
        return input_ids, token_count

    def _build_capture(self, events: list[ActivationEvent], token_count: int, run_id: str, metadata: Mapping[str, Any] | None) -> dict:
        captured_layer_ids = [event.layer_id for event in events]
        return {
            "runtime_id": run_id,
            "token_count": token_count,
            "captured": bool(events),
            "captured_layer_count": len(events),
            "captured_layer_ids": captured_layer_ids,
            "layer_indices": self._layer_indices,
            "summary_top_dimensions": self._summary_top_dimensions,
            "top_dimensions": {
                "requested": self._summary_top_dimensions,
                "captured": min(len(captured_layer_ids), self._summary_top_dimensions),
            },
            "metadata": dict(metadata or {}),
        }

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, Mapping):
            return {str(k): LocalModelRuntime._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [LocalModelRuntime._json_safe(v) for v in value]
        return str(value)

    def observe_text(self, query_text: str, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Observe a text query and persist a model-scope artifact."""
        run_id = str(uuid.uuid4())
        if self._model is None:
            self.load()
        self.clear_events()

        input_ids, token_count = self._prepare_input_ids(query_text)
        events = self.run_inference(input_ids, run_id=run_id)
        try:
            from model_scope_artifacts import build_model_scope_artifact
        except Exception:
            def build_model_scope_artifact(*, runtime, capture, output_path: str | None = None, trace: Mapping[str, Any] | None = None, evidence: Mapping[str, Any] | None = None, **kwargs):
                return {
                    "schema_version": 1,
                    "artifact_type": "model_scope_runtime_observation",
                    "runtime": LocalModelRuntime._json_safe(runtime),
                    "capture": LocalModelRuntime._json_safe(capture),
                    **({"output_path": str(output_path)} if output_path is not None else {}),
                    **({"trace": LocalModelRuntime._json_safe(trace)} if trace is not None else {}),
                    **({"evidence": LocalModelRuntime._json_safe(evidence)} if evidence is not None else {}),
                }

        artifact = build_model_scope_artifact(
            runtime=self.describe_runtime(),
            capture=self._build_capture(events, token_count=token_count, run_id=run_id, metadata=metadata),
            trace={
                "query_length": len(str(query_text)),
            },
            evidence={
                "event_id": run_id,
                "run_id": run_id,
            },
        )
        event_count = len(events)
        research_payload = {
            "events_captured": event_count > 0,
            "captured_layer_count": event_count,
        }
        if research_enabled() and events:
            control_values = np.array(
                [float(events[0].mean_activation), float(events[0].norm_activation), float(events[0].token_count)],
                dtype=float,
            )
            modified_values, promoted_meta = promoted_sip_apply(control_values)
            if promoted_meta is not None:
                events[0].mean_activation = float(modified_values[0])
                events[0].norm_activation = float(modified_values[1])
                events[0].token_count = int(max(0, round(modified_values[2])))
                research_payload["research_shim_apply_applied"] = True
                research_payload["promoted_sip_apply"] = promoted_meta

        artifact["steering"] = None
        artifact["memory"] = None
        artifact["expectation_comparison"] = {
            "status": "observation_only",
            "captured": bool(events),
        }

        output_path = self._artifact_dir / f"observation_{run_id}.json"
        try:
            output_path.write_text(json.dumps(self._json_safe(artifact), indent=2), encoding="utf-8")
        except Exception as exc:
            warnings.warn(
                f"Model-Scope artifact persistence failed: {type(exc).__name__}: {exc}",
                stacklevel=2,
            )
        artifact["output_path"] = str(output_path)
        if research_enabled():
            self._research_observe_stall_count = bump_stall_counter(
                self._research_observe_stall_count,
                has_work=bool(events),
            )
            if bool(research_payload):
                _, promoted_meta = promoted_sip_apply(np.zeros(8, dtype=float))
                if promoted_meta is not None:
                    research_payload["promoted_sip_apply"] = promoted_meta
                self._last_research_shim_meta = research_preflight_metadata(
                    seam="LocalModelRuntime.observe_text",
                    stall_count=self._research_observe_stall_count,
                    extra=research_payload,
                )
                artifact["research_shim_meta"] = self._last_research_shim_meta
        else:
            self._last_research_shim_meta = None
            artifact["research_shim_meta"] = None

        return artifact

    def get_last_research_shim_meta(self) -> dict[str, Any] | None:
        """Return the last preflight shim metadata emitted by observe_text()."""
        return self._last_research_shim_meta

    def describe_runtime(self) -> dict[str, Any]:
        """Return a runtime telemetry bundle for diagnostics."""
        return {
            "schema_version": "1.0",
            "model_name": self.model_name,
            "device": self.device,
            "layer_indices": self._layer_indices,
            "hook_layers": list(self._hook_layers),
            "max_input_tokens": self._max_input_tokens,
            "summary_top_dimensions": self._summary_top_dimensions,
            "artifact_dir": str(self._artifact_dir),
        }

    def run_inference(
        self,
        input_ids: Any,
        *,
        run_id: str = None,
    ) -> list[ActivationEvent]:
        """Run a forward pass, capture activation stats, return new events."""
        if self._model is None:
            raise RuntimeError("Model not loaded; call load() first.")

        effective_run_id = run_id if run_id is not None else str(uuid.uuid4())
        run_events: list[ActivationEvent] = []
        hooks: list = []

        def _make_hook(layer_id: str) -> Callable:
            def hook(module: Any, _inp: Any, output: Any) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                try:
                    import torch  # type: ignore[import]

                    t_float = tensor.float()
                    mean_val = float(t_float.mean().item())
                    try:
                        norm_val = float(torch.linalg.norm(t_float).item())
                    except Exception as norm_err:
                        # Fallback for mock tensors that do not implement
                        # linalg.norm. Expected in tests; unexpected in
                        # production.
                        warnings.warn(
                            f"torch.linalg.norm fallback triggered ({type(norm_err).__name__}: {norm_err}); "
                            "expected for mock tensors in tests, unexpected in production",
                            stacklevel=2,
                        )
                        norm_val = float(tensor.norm())
                except Exception as outer_err:
                    # Fallback for mock tensors that do not implement full
                    # tensor protocol. This may include test fakes.
                    warnings.warn(
                        "torch tensor protocol fallback triggered in hook "
                        f"(mock-tensor compatible path; unexpected in production): "
                        f"{outer_err!r}",
                        UserWarning,
                        stacklevel=2,
                    )
                    mean_val = float(tensor.mean())
                    norm_val = float(tensor.norm())
                shape = tuple(tensor.shape)
                token_count = shape[1] if len(shape) >= 2 else shape[0]

                event = ActivationEvent(
                    schema_version="1.0",
                    model_id=self.model_name,
                    layer_id=layer_id,
                    token_count=token_count,
                    shape=shape,
                    mean_activation=mean_val,
                    norm_activation=norm_val,
                    captured_at=datetime.now(timezone.utc).isoformat(),
                    run_id=effective_run_id,
                    # raw_tensor_shape explicitly records the actual tensor
                    # dimensions captured by this hook invocation.
                    raw_tensor_shape=shape,
                )
                run_events.append(event)

            return hook

        for layer_id in self._hook_layers:
            module = _resolve_module(self._model, layer_id)
            if module is not None:
                hooks.append(module.register_forward_hook(_make_hook(layer_id)))
            else:
                try:
                    module_names = list(n for n, _ in self._model.named_modules())
                except Exception:
                    module_names = []
                warnings.warn(
                    f"Layer '{layer_id}' not found in model; hook will be skipped. "
                    f"Available submodules: {module_names}",
                    UserWarning,
                    stacklevel=2,
                )

        try:
            self._model(input_ids)
        finally:
            for h in hooks:
                h.remove()

        self._events.extend(run_events)
        return run_events

    def get_events(self) -> list:
        """Return all accumulated activation events."""
        return list(self._events)

    def clear_events(self) -> None:
        """Reset the accumulated event list."""
        self._events = []

    def is_loaded(self) -> bool:
        """Return True if the model has been loaded."""
        return self._model is not None

    def save_events(self, path: Any) -> None:
        """Write events as JSON Lines (one event per line)."""
        path = Path(path)
        with path.open("w", encoding="utf-8") as fh:
            for event in self._events:
                fh.write(json.dumps(event.to_dict()) + "\n")

    @classmethod
    def load_events(cls, path: Any) -> list[ActivationEvent]:
        """Reload ActivationEvent list from a JSON Lines file."""
        path = Path(path)
        events: list[ActivationEvent] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    events.append(ActivationEvent.from_dict(json.loads(line)))
        return events


def create_model_scope_runtime(
    model_name: str,
    *,
    layer_indices: list | None = None,
    max_input_tokens: int = 256,
    summary_top_dimensions: int = 8,
    artifact_dir: str = "experiment_runs/model_scope",
    eager_load: bool = False,
    logger: Any = None,
) -> LocalModelRuntime:
    """Create and (optionally) eagerly load a local model-scope runtime."""
    return LocalModelRuntime(
        model_name=model_name,
        layer_indices=layer_indices,
        max_input_tokens=max_input_tokens,
        summary_top_dimensions=summary_top_dimensions,
        artifact_dir=artifact_dir,
        eager_load=eager_load,
        logger=logger,
    )
