"""Query-side encoder-upgrade drift for retrieval-recovery experiments.

This models the realistic failure mode an encoder upgrade introduces: the eval
queries move to a *new* encoder while the document store remains cached in the
*original* encoder's space. Retrieval degrades because queries and documents are
no longer in the same representation.

Crucially, this drift defeats the "re-embed maintenance" oracle (condition C2).
C2 re-embeds the documents' unchanged text with the *original* frozen encoder,
which reproduces the cached document vectors exactly — but those vectors are
still in the old space and therefore still misaligned with the upgraded query
space, so C2 does not recover retrieval. Only an adapter that re-aligns the
cached document vectors toward the new query space (condition C3), or a full
re-embed of the corpus with the *new* encoder (the expensive oracle upper
bound), can recover. See
docs/waypoint-research-2026-06-09/finding-oracle-survives-document-model-swap.md.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

import numpy as np


class QueryEncoderDrift:
    """Embeds query text with a swapped encoder projected into the store's space.

    The projection is a FROZEN, seeded ``DimensionProjection`` and is part of the
    drift definition — it is never trained. Outcomes are a deterministic function
    of ``(seed, swap model, query text)``.
    """

    def __init__(
        self,
        store_dim: int,
        swap_model_name: str = "all-mpnet-base-v2",
        seed: int = 0,
        swap_backend: Any = None,
    ):
        self.store_dim = int(store_dim)
        self.swap_model_name = str(swap_model_name)
        self.seed = int(seed)
        self._swap_backend = swap_backend
        self._projection = None
        self._swap_dim: Optional[int] = None

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        """Return drifted query embeddings of shape (len(texts), store_dim), L2-normalized."""
        if not isinstance(texts, (list, tuple)) or len(texts) == 0:
            raise ValueError("texts must be a non-empty sequence of strings")
        backend = self._backend()
        raw = np.asarray(backend.embed_raw(list(texts)), dtype=np.float32)
        if raw.ndim != 2 or raw.shape[0] != len(texts):
            raise ValueError(
                f"swap backend returned shape {raw.shape}; expected ({len(texts)}, swap_dim)"
            )
        self._ensure_projection(int(raw.shape[1]))
        projected = self._projection.project_numpy(raw)
        return self._l2_normalize(np.asarray(projected, dtype=np.float32))

    def manifest(self) -> dict:
        """Reproducibility record. Call after at least one embed_queries()."""
        if self._projection is None or self._swap_dim is None:
            raise ValueError("manifest() requires at least one embed_queries() call first")
        return {
            "drift": "query_encoder_swap",
            "swap_model": self.swap_model_name,
            "projection_seed": self.seed,
            "projection_checksum": self._projection_checksum(self._projection),
            "swap_dim": int(self._swap_dim),
            "store_dim": int(self.store_dim),
        }

    def _backend(self) -> Any:
        if self._swap_backend is None:
            from embedding_backend import create_embedding_backend

            self._swap_backend = create_embedding_backend(self.swap_model_name)
        return self._swap_backend

    def _ensure_projection(self, swap_dim: int) -> None:
        if self._projection is not None:
            if swap_dim != self._swap_dim:
                raise ValueError(
                    f"swap backend changed dimension from {self._swap_dim} to {swap_dim}"
                )
            return
        import torch

        from teacher_distillation import DimensionProjection

        # Seed torch's global RNG so DimensionProjection's near-identity init
        # (which uses torch.randn_like) is reproducible. Frozen — never trained.
        torch.manual_seed(int(self.seed))
        projection = DimensionProjection(int(swap_dim), int(self.store_dim))
        projection.eval()
        for param in projection.parameters():
            param.requires_grad_(False)
        self._projection = projection
        self._swap_dim = int(swap_dim)

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms > 0.0, norms, 1.0)
        return (matrix / norms).astype(np.float32)

    @staticmethod
    def _projection_checksum(projection: Any) -> str:
        digest = hashlib.sha256()
        state: Dict[str, Any] = projection.state_dict()
        for name in sorted(state.keys()):
            digest.update(name.encode("utf-8"))
            digest.update(
                np.ascontiguousarray(state[name].detach().cpu().numpy(), dtype=np.float32).tobytes()
            )
        return digest.hexdigest()
