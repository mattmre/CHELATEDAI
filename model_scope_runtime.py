"""Local model runtime for observation-only activation capture."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


@dataclass
class ActivationEvent:
    """One captured activation from a model layer during inference."""

    schema_version: str
    model_id: str
    layer_id: str
    token_count: int
    shape: tuple
    mean_activation: float
    norm_activation: float
    captured_at: str
    run_id: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["shape"] = list(d["shape"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ActivationEvent:
        d = dict(d)
        d["shape"] = tuple(d["shape"])
        return cls(**d)


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
        hook_layers: list = None,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._hook_layers: list = list(hook_layers or [])
        self._model: Any = None
        self._events: list = []

    def load(self, *, model_loader: Callable = None) -> None:
        """Load the model. Accepts an optional loader callable for injection in tests."""
        if model_loader is not None:
            self._model = model_loader(self.model_name)
        else:
            from transformers import AutoModelForCausalLM  # type: ignore[import]

            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.to(self.device)

    def register_hook_layers(self, layer_ids: list) -> None:
        """Add layer IDs to the set that will be hooked during inference."""
        for lid in layer_ids:
            if lid not in self._hook_layers:
                self._hook_layers.append(lid)

    def run_inference(
        self,
        input_ids: Any,
        *,
        run_id: str = None,
    ) -> list:
        """Run a forward pass, capture activation stats, return new events."""
        if self._model is None:
            raise RuntimeError("Model not loaded; call load() first.")

        effective_run_id = run_id if run_id is not None else str(uuid.uuid4())
        run_events: list = []
        hooks: list = []

        def _make_hook(layer_id: str) -> Callable:
            def hook(module: Any, _inp: Any, output: Any) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                try:
                    import torch  # type: ignore[import]

                    t_float = tensor.float()
                    mean_val = float(t_float.mean().item())
                    norm_val = float(torch.linalg.norm(t_float).item())
                    shape = tuple(tensor.shape)
                    token_count = shape[1] if len(shape) >= 2 else shape[0]
                except Exception:
                    # Fallback for mock tensors exposing .mean / .norm as callables
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
                )
                run_events.append(event)

            return hook

        for lid in self._hook_layers:
            module = _resolve_module(self._model, lid)
            if module is not None:
                h = module.register_forward_hook(_make_hook(lid))
                hooks.append(h)

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
    def load_events(cls, path: Any) -> list:
        """Reload ActivationEvent list from a JSON Lines file."""
        path = Path(path)
        events: list = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    events.append(ActivationEvent.from_dict(json.loads(line)))
        return events
