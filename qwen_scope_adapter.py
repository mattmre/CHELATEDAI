"""Load and apply official Qwen-Scope SAE checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import torch


@dataclass
class QwenScopeLayerSAE:
    """One layer of the official Qwen-Scope SAE contract."""

    layer_index: int
    w_enc: torch.Tensor
    b_enc: torch.Tensor
    top_k: int = 100

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, Any], *, layer_index: int, top_k: int = 100):
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
