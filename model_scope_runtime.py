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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class ActivationEvent:
    """One captured activation from a model layer during inference.

    Fields
    ------
    schema_version  : semver string for forward-compat deserialisation.
    model_id        : HuggingFace model identifier used at load time.
    layer_id        : dotted attribute path (e.g. ``model.layers.15``).
    token_count     : number of tokens in the sequence dimension.
    shape           : full tensor shape at the point of capture, e.g.
                      ``(batch, seq_len, hidden_size)``.
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
                    try:
                        norm_val = float(torch.linalg.norm(t_float).item())
                    except Exception as _e:
                        # Fallback for mock tensors that do not implement linalg.norm.
                        # Expected in tests; unexpected in production (e.g. CUDA OOM
                        # would appear here — the warning makes silent swallowing
                        # visible).
                        warnings.warn(
                            f"torch.linalg.norm fallback triggered"
                            f" ({type(_e).__name__}: {_e}); "
                            "expected for mock tensors in tests,"
                            " unexpected in production",
                            stacklevel=2,
                        )
                        norm_val = float(tensor.norm())
                except Exception:
                    # Fallback for mock tensors that do not implement the full
                    # torch tensor protocol (e.g. no .item() on .mean() result).
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
                    # dimensions captured by this hook invocation.  The full
                    # tensor data is NOT retained here (memory efficiency).
                    raw_tensor_shape=shape,
                )
                run_events.append(event)

            return hook

        for lid in self._hook_layers:
            module = _resolve_module(self._model, lid)
            if module is not None:
                h = module.register_forward_hook(_make_hook(lid))
                hooks.append(h)
            else:
                warnings.warn(
                    f"Layer '{lid}' not found in model; hook will be skipped. "
                    f"Available submodules: {[n for n, _ in self._model.named_modules()]}",
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
