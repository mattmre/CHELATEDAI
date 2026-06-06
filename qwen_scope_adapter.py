"""Qwen-Scope SAE adapter for sparse feature extraction from activation events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import numpy as np

try:
    import torch  # type: ignore
    _HAS_TORCH = True
except ModuleNotFoundError:  # pragma: no cover - lean environments without torch
    torch = None
    _HAS_TORCH = False

from model_scope_runtime import ActivationEvent


@dataclass
class QwenScopeLayerSAE:
    """One layer of the official Qwen-Scope SAE contract."""

    layer_index: int
    w_enc: torch.Tensor
    b_enc: torch.Tensor
    top_k: int = 100

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, Any], *, layer_index: int, top_k: int = 100):
        if torch is None:
            raise ModuleNotFoundError("torch is required for QwenScopeLayerSAE")
        if "W_enc" not in state_dict or "b_enc" not in state_dict:
            raise ValueError("Qwen-Scope checkpoint must contain W_enc and b_enc")
        w_enc = torch.as_tensor(state_dict["W_enc"], dtype=torch.float32).detach().cpu()
        b_enc = torch.as_tensor(state_dict["b_enc"], dtype=torch.float32).detach().cpu()
        if w_enc.ndim != 2:
            raise ValueError(f"W_enc must be rank-2, got shape {tuple(w_enc.shape)}")
        if b_enc.ndim != 1:
            raise ValueError(f"b_enc must be rank-1, got shape {tuple(b_enc.shape)}")
        if w_enc.shape[0] != b_enc.shape[0]:
            raise ValueError(
                f"W_enc/b_enc feature dimension mismatch: {tuple(w_enc.shape)} vs {tuple(b_enc.shape)}"
            )
        return cls(layer_index=layer_index, w_enc=w_enc, b_enc=b_enc, top_k=int(top_k))

    @classmethod
    def from_file(cls, path: str | Path, *, layer_index: int, top_k: int = 100):
        if torch is None:
            raise ModuleNotFoundError("torch is required for QwenScopeLayerSAE.from_file")
        state_dict = torch.load(Path(path), map_location="cpu")
        return cls.from_state_dict(state_dict, layer_index=layer_index, top_k=top_k)

    @property
    def d_sae(self) -> int:
        return int(self.w_enc.shape[0])

    @property
    def d_model(self) -> int:
        return int(self.w_enc.shape[1])

    def encode(self, residual: torch.Tensor) -> torch.Tensor:
        """Apply the official encoder path and top-k sparsification."""
        if torch is None:
            raise ModuleNotFoundError("torch is required for QwenScopeLayerSAE.encode")

        value = torch.as_tensor(residual, dtype=torch.float32).detach().cpu()
        if value.shape[-1] != self.d_model:
            raise ValueError(
                f"residual hidden size {value.shape[-1]} does not match SAE d_model {self.d_model}"
            )
        pre_acts = value @ self.w_enc.t() + self.b_enc
        top_k = min(max(int(self.top_k), 1), pre_acts.shape[-1])
        top_values, top_indices = pre_acts.topk(top_k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, top_indices, top_values)
        return acts

    def summarize_last_token(self, residual: torch.Tensor, *, top_features: int = 8) -> Dict[str, Any]:
        if torch is None:
            raise ModuleNotFoundError("torch is required for QwenScopeLayerSAE.summarize_last_token")
        acts = self.encode(residual)
        if acts.ndim == 3:
            last_token = acts[0, -1]
        elif acts.ndim == 2:
            last_token = acts[-1]
        else:
            last_token = acts.reshape(-1)
        active_idx = last_token.nonzero(as_tuple=True)[0]
        active_vals = last_token[active_idx]
        top_features = min(max(int(top_features), 0), active_idx.numel())
        if top_features > 0:
            top_vals, top_order = torch.topk(active_vals, k=top_features, dim=-1)
            top_idx = active_idx[top_order]
            features = [
                {
                    "feature_id": int(feature_id),
                    "value": float(score),
                }
                for feature_id, score in zip(top_idx.tolist(), top_vals.tolist())
            ]
        else:
            features = []
        return {
            "feature_space": "qwen_scope_sae",
            "layer_index": int(self.layer_index),
            "d_model": self.d_model,
            "d_sae": self.d_sae,
            "sae_top_k": int(self.top_k),
            "active_feature_count": int(active_idx.numel()),
            "token_selector": "last_token",
            "active_features": features,
        }


@dataclass
class SAECheckpointMetadata:
    """Metadata for a loaded Sparse Autoencoder checkpoint."""

    model_family: str
    layer_id: str
    feature_count: int
    checkpoint_path: str
    loaded_at: str
    schema_version: str = "1.0"


class QwenScopeAdapter:
    """Loads Qwen-Scope SAE checkpoints and extracts sparse feature activations."""

    def __init__(self, model_family: str = "Qwen3.5") -> None:
        self.model_family = model_family
        self._weights: np.ndarray | None = None
        self._metadata: SAECheckpointMetadata | None = None

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        checkpoint_loader: Callable | None = None,
    ) -> SAECheckpointMetadata:
        """Load SAE weights from path; optional loader callable for test injection."""
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_loader is not None:
            raw = checkpoint_loader(checkpoint_path)
        else:
            raw = np.load(str(checkpoint_path))

        layer_id = checkpoint_path.stem
        if isinstance(raw, dict):
            layer_id = str(raw.get("layer_id", checkpoint_path.stem))
            weights_candidate = raw.get("weights")
            if weights_candidate is None:
                for v in raw.values():
                    if hasattr(v, "shape"):
                        weights_candidate = v
                        break
            if weights_candidate is None:
                raise ValueError("Checkpoint dict contains no array with .shape attribute.")
            weights = np.asarray(weights_candidate, dtype=np.float64)
        else:
            weights = np.asarray(raw, dtype=np.float64)

        self._weights = weights
        self._metadata = SAECheckpointMetadata(
            model_family=self.model_family,
            layer_id=layer_id,
            feature_count=int(weights.shape[0]),
            checkpoint_path=str(checkpoint_path),
            loaded_at=datetime.now(timezone.utc).isoformat(),
        )
        return self._metadata

    def is_loaded(self) -> bool:
        """Return True if a checkpoint has been loaded."""
        return self._weights is not None

    def extract_features(self, activation: ActivationEvent) -> dict[str, float]:
        """Project activation statistics through SAE weights and return a sparse feature dict.

        IMPORTANT — STATISTICS-BASED PATH, NOT TENSOR-BASED
        -----------------------------------------------------
        This method operates on the *scalar statistics* stored in ``ActivationEvent``
        (``mean_activation``, ``norm_activation``, ``token_count``, ``shape[0]``),
        NOT on a raw residual tensor.  ``ActivationEvent`` intentionally does not
        store the full tensor for memory-efficiency reasons (see module docstring in
        ``model_scope_runtime.py``).

        As a result the SAE projection here is an approximation suitable for
        lightweight monitoring only.  It is NOT the same as calling
        ``QwenScopeLayerSAE.encode(residual)`` on the full hidden-state tensor.
        If you need the true SAE feature activations, use ``QwenScopeLayerSAE``
        directly inside your hook and pass the raw ``torch.Tensor`` residual.

        The input vector constructed here is:
            [mean_activation, norm_activation, token_count, shape[0], 0.0, ...]
        padded or truncated to match ``weights.shape[1]``.

        Returns
        -------
        dict[str, float]
            Sparse dict of ``"feature_<i>": value`` for all positive-valued outputs.
        """
        if not self.is_loaded():
            raise RuntimeError("SAE checkpoint not loaded; call load_checkpoint() first.")

        weights = self._weights  # shape (feature_count, input_dim)
        input_dim = int(weights.shape[1])
        raw: list[float] = [
            float(activation.mean_activation),
            float(activation.norm_activation),
            float(activation.token_count),
        ]
        if activation.shape:
            raw.append(float(activation.shape[0]))

        if len(raw) < input_dim:
            raw = raw + [0.0] * (input_dim - len(raw))
        else:
            raw = raw[:input_dim]

        input_vec = np.array(raw, dtype=np.float64)
        output: np.ndarray = weights @ input_vec
        return {
            f"feature_{i}": float(val)
            for i, val in enumerate(output)
            if float(val) > 0.0
        }

    def feature_count(self) -> int:
        """Return number of SAE features. Raises RuntimeError if not loaded."""
        if not self.is_loaded():
            raise RuntimeError("SAE checkpoint not loaded; call load_checkpoint() first.")
        return int(self._weights.shape[0])

    def supports_model(self, model_id: str) -> bool:
        """Return True if model_family substring appears in model_id."""
        return self.model_family in model_id
