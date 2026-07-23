"""Frozen array artifacts for offline drift-recovery statistics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


@dataclass
class EmbeddingPack:
    """All arrays needed to rerun D1 statistics without model inference.

    The required core is ``Do``, ``Dor``, ``Qd``, qrels, ``fit_idx``, and
    per-query method scores. Extra arrays hold non-eval supervision pools so the
    learning curves remain leakage-free while using the same fixed eval IDs.
    """

    Do: np.ndarray
    Dor: np.ndarray
    Qd: np.ndarray
    doc_ids: np.ndarray
    query_ids: np.ndarray
    qrels: Mapping[str, Mapping[str, float]]
    fit_idx: np.ndarray
    per_query_scores: Mapping[str, np.ndarray]
    extra_arrays: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.Do.shape != self.Dor.shape or self.Do.ndim != 2:
            raise ValueError("Do and Dor must be same-shape matrices")
        if self.Qd.ndim != 2 or self.Qd.shape[1] != self.Do.shape[1]:
            raise ValueError("Qd must share the embedding dimension")
        if len(self.doc_ids) != self.Do.shape[0]:
            raise ValueError("doc_ids length mismatch")
        if len(self.query_ids) != self.Qd.shape[0]:
            raise ValueError("query_ids length mismatch")
        if np.any(self.fit_idx < 0) or np.any(self.fit_idx >= self.Do.shape[0]):
            raise ValueError("fit_idx is out of bounds")
        if len(np.unique(self.fit_idx)) != len(self.fit_idx):
            raise ValueError("fit_idx contains duplicates")
        for method, scores in self.per_query_scores.items():
            array = np.asarray(scores)
            if array.shape != (len(self.query_ids),):
                raise ValueError(f"{method} per-query score shape mismatch: {array.shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{method} contains non-finite scores")
        missing = set(map(str, self.query_ids)).difference(map(str, self.qrels))
        if missing:
            raise ValueError(f"qrels missing eval queries: {sorted(missing)[:5]}")

    def save(self, prefix: Path) -> Dict[str, str]:
        self.validate()
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        arrays_path = prefix.with_suffix(".npz")
        metadata_path = prefix.with_suffix(".json")
        arrays: Dict[str, np.ndarray] = {
            "Do": np.asarray(self.Do),
            "Dor": np.asarray(self.Dor),
            "Qd": np.asarray(self.Qd),
            "doc_ids": np.asarray(self.doc_ids, dtype=str),
            "query_ids": np.asarray(self.query_ids, dtype=str),
            "fit_idx": np.asarray(self.fit_idx, dtype=np.int64),
        }
        arrays.update({f"score__{key}": np.asarray(value) for key, value in self.per_query_scores.items()})
        arrays.update({f"extra__{key}": np.asarray(value) for key, value in self.extra_arrays.items()})
        np.savez_compressed(arrays_path, **arrays)
        payload = {
            "record_type": "embedding_pack",
            "format_version": 1,
            "arrays_file": arrays_path.name,
            "qrels": _json_ready(self.qrels),
            "metadata": _json_ready(self.metadata),
            "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        }
        metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        digest = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
        payload["arrays_sha256"] = digest
        metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return {"arrays": str(arrays_path), "metadata": str(metadata_path), "sha256": digest}

    @classmethod
    def load(cls, prefix: Path) -> "EmbeddingPack":
        prefix = Path(prefix)
        arrays_path = prefix.with_suffix(".npz")
        metadata_path = prefix.with_suffix(".json")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
        if digest != payload.get("arrays_sha256"):
            raise ValueError("EmbeddingPack array checksum mismatch")
        with np.load(arrays_path, allow_pickle=False) as stored:
            scores = {key[7:]: stored[key].copy() for key in stored.files if key.startswith("score__")}
            extras = {key[7:]: stored[key].copy() for key in stored.files if key.startswith("extra__")}
            pack = cls(
                Do=stored["Do"].copy(),
                Dor=stored["Dor"].copy(),
                Qd=stored["Qd"].copy(),
                doc_ids=stored["doc_ids"].copy(),
                query_ids=stored["query_ids"].copy(),
                qrels=payload["qrels"],
                fit_idx=stored["fit_idx"].copy(),
                per_query_scores=scores,
                extra_arrays=extras,
                metadata=payload.get("metadata", {}),
            )
        pack.validate()
        return pack


@dataclass
class D2CellPack:
    """Compact five-seed D2 scores used for all grouped cell statistics."""

    seed_ids: Sequence[int]
    query_ids: Sequence[str]
    per_seed_scores: Mapping[str, np.ndarray]
    detector_probabilities: np.ndarray
    detector_harm_losses: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        seed_count = len(self.seed_ids)
        query_count = len(self.query_ids)
        if seed_count < 1 or query_count < 2:
            raise ValueError("D2 cell pack requires seeds and at least two queries")
        if len(set(map(int, self.seed_ids))) != seed_count:
            raise ValueError("D2 seed IDs must be unique")
        if len(set(map(str, self.query_ids))) != query_count:
            raise ValueError("D2 query IDs must be unique")
        required = {"floor", "oracle"}
        if not required.issubset(self.per_seed_scores):
            raise ValueError("D2 scores require floor and oracle")
        for method, values in self.per_seed_scores.items():
            array = np.asarray(values, dtype=np.float64)
            if array.shape != (seed_count, query_count):
                raise ValueError(f"{method} D2 score shape mismatch: {array.shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{method} D2 scores contain non-finite values")
        probabilities = np.asarray(self.detector_probabilities, dtype=np.float64)
        losses = np.asarray(self.detector_harm_losses, dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[0] != seed_count:
            raise ValueError("detector probability shape mismatch")
        if losses.shape != probabilities.shape:
            raise ValueError("detector harm shape mismatch")
        if not np.all(np.isfinite(probabilities)) or not np.all(np.isfinite(losses)):
            raise ValueError("detector arrays contain non-finite values")
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise ValueError("detector probabilities must be in [0, 1]")
        if np.any(losses < 0.0):
            raise ValueError("detector harm losses must be nonnegative")

    def save(self, prefix: Path) -> Dict[str, str]:
        self.validate()
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        arrays_path = prefix.with_suffix(".npz")
        metadata_path = prefix.with_suffix(".json")
        arrays = {
            "seed_ids": np.asarray(self.seed_ids, dtype=np.int64),
            "query_ids": np.asarray(self.query_ids, dtype=str),
            "detector_probabilities": np.asarray(self.detector_probabilities, dtype=np.float64),
            "detector_harm_losses": np.asarray(self.detector_harm_losses, dtype=np.float64),
        }
        arrays.update(
            {
                f"score__{method}": np.asarray(values, dtype=np.float64)
                for method, values in self.per_seed_scores.items()
            }
        )
        np.savez_compressed(arrays_path, **arrays)
        digest = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
        payload = {
            "record_type": "d2_cell_pack",
            "format_version": 1,
            "arrays_file": arrays_path.name,
            "arrays_sha256": digest,
            "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
            "metadata": _json_ready(self.metadata),
        }
        metadata_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        return {"arrays": str(arrays_path), "metadata": str(metadata_path), "sha256": digest}

    @classmethod
    def load(cls, prefix: Path) -> "D2CellPack":
        prefix = Path(prefix)
        arrays_path = prefix.with_suffix(".npz")
        metadata_path = prefix.with_suffix(".json")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
        if digest != payload.get("arrays_sha256"):
            raise ValueError("D2CellPack array checksum mismatch")
        with np.load(arrays_path, allow_pickle=False) as stored:
            scores = {
                name[len("score__") :]: stored[name].copy()
                for name in stored.files
                if name.startswith("score__")
            }
            pack = cls(
                seed_ids=stored["seed_ids"].astype(np.int64).tolist(),
                query_ids=stored["query_ids"].astype(str).tolist(),
                per_seed_scores=scores,
                detector_probabilities=stored["detector_probabilities"].copy(),
                detector_harm_losses=stored["detector_harm_losses"].copy(),
                metadata=payload.get("metadata", {}),
            )
        pack.validate()
        return pack
