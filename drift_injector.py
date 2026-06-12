"""Deterministic vector drift injection for retrieval-recovery experiments."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from qdrant_client.models import PointStruct


class DriftInjector:
    """Injects seeded drift into vectors stored by an AntigravityEngine.

    Determinism contract: outcomes are a pure function of ``(seed, ordered
    sequence of injection calls)``. The shared RNG advances on every call, so
    the second injection on one instance differs from the first injection on a
    fresh instance with the same seed. Every manifest records
    ``injection_index`` (0-based position in this instance's call sequence);
    reproducing any injection requires replaying the same seed and call
    sequence from a fresh instance. Validation errors raise before any RNG is
    consumed, so a raising call advances neither the RNG state nor
    ``injection_index`` — only successful injections count toward the sequence.
    """

    def __init__(self, engine, seed: int):
        self.engine = engine
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._injection_index = 0

    def inject_rotation_drift(
        self,
        fraction: float,
        angle_degrees: float,
        dims: Optional[list] = None,
    ) -> dict:
        """Apply seeded Givens rotations to a deterministic fraction of stored vectors."""

        records = self._load_records()
        # All validation must precede any RNG consumption so that a raising
        # call leaves both the RNG state and injection_index untouched.
        vector_size = self._vector_size(records)
        selected_dims = self._validate_dims(dims, vector_size)
        theta = math.radians(float(angle_degrees))
        if not math.isfinite(theta):
            raise ValueError("angle_degrees must be finite")
        self._validate_fraction(fraction)

        affected_records = self._select_affected(records, fraction)
        rotation_pairs = self._rotation_pairs(selected_dims)

        checksum_before = self._checksum(records)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        for record in affected_records:
            rotated = record["vector"].copy()
            for dim_a, dim_b in rotation_pairs:
                value_a = rotated[dim_a]
                value_b = rotated[dim_b]
                rotated[dim_a] = (cos_theta * value_a) - (sin_theta * value_b)
                rotated[dim_b] = (sin_theta * value_a) + (cos_theta * value_b)
            record["vector"] = rotated.astype(np.float32)

        self._upsert_records(affected_records)
        updated_records = self._records_with_updates(records, affected_records)

        return self._manifest(
            mode="rotation",
            fraction=fraction,
            affected_records=affected_records,
            checksum_before=checksum_before,
            checksum_after=self._checksum(updated_records),
            dims=selected_dims,
            angle_degrees=float(angle_degrees),
            sigma=None,
            rotation_pairs=rotation_pairs,
        )

    def inject_noise_drift(self, fraction: float, sigma: float) -> dict:
        """Add seeded Gaussian noise to a deterministic fraction of stored vectors."""

        records = self._load_records()
        # All validation must precede any RNG consumption so that a raising
        # call leaves both the RNG state and injection_index untouched.
        sigma_value = float(sigma)
        if not math.isfinite(sigma_value) or sigma_value < 0.0:
            raise ValueError("sigma must be a non-negative finite number")
        self._validate_fraction(fraction)

        affected_records = self._select_affected(records, fraction)
        checksum_before = self._checksum(records)

        for record in affected_records:
            vector = record["vector"].copy()
            noise = self._rng.normal(0.0, sigma_value, size=vector.shape).astype(np.float32)
            drifted = vector + noise
            norm = float(np.linalg.norm(drifted))
            if norm > 0.0:
                drifted = drifted / norm
            record["vector"] = drifted.astype(np.float32)

        self._upsert_records(affected_records)
        updated_records = self._records_with_updates(records, affected_records)

        return self._manifest(
            mode="noise",
            fraction=fraction,
            affected_records=affected_records,
            checksum_before=checksum_before,
            checksum_after=self._checksum(updated_records),
            dims=list(range(self._vector_size(records))),
            angle_degrees=None,
            sigma=float(sigma),
            rotation_pairs=[],
        )

    def _load_records(self) -> List[Dict[str, Any]]:
        records = []
        offset = None
        while True:
            points, next_offset = self.engine.qdrant.scroll(
                collection_name=self.engine.collection_name,
                limit=256,
                with_vectors=True,
                with_payload=True,
                offset=offset,
            )
            for point in points:
                vector, vector_name = self._as_vector(point.vector)
                records.append(
                    {
                        "id": point.id,
                        "vector": vector,
                        "vector_name": vector_name,
                        "payload": point.payload or {},
                    }
                )
            if next_offset is None or not points:
                break
            offset = next_offset

        records.sort(key=lambda item: repr(item["id"]))
        if not records:
            raise ValueError("Cannot inject drift into an empty vector store")
        return records

    @staticmethod
    def _validate_fraction(fraction: float) -> None:
        if not 0.0 <= float(fraction) <= 1.0:
            raise ValueError("fraction must be between 0.0 and 1.0")

    def _select_affected(self, records: Sequence[Dict[str, Any]], fraction: float) -> List[Dict[str, Any]]:
        self._validate_fraction(fraction)
        count = int(math.floor((len(records) * float(fraction)) + 0.5))
        count = max(0, min(len(records), count))
        if count == 0:
            return []
        indices = np.sort(self._rng.choice(len(records), size=count, replace=False))
        return [dict(records[int(index)], vector=records[int(index)]["vector"].copy()) for index in indices]

    def _upsert_records(self, records: Sequence[Dict[str, Any]]) -> None:
        if not records:
            return
        points = []
        for record in records:
            vector_values = record["vector"].astype(np.float32).tolist()
            vector_name = record.get("vector_name")
            vector_payload = {vector_name: vector_values} if vector_name is not None else vector_values
            points.append(
                PointStruct(
                    id=record["id"],
                    vector=vector_payload,
                    payload=record["payload"],
                )
            )
        self.engine.qdrant.upsert(collection_name=self.engine.collection_name, points=points)

    def _records_with_updates(
        self,
        records: Sequence[Dict[str, Any]],
        updates: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        updates_by_id = {record["id"]: record for record in updates}
        updated_records = []
        for record in records:
            replacement = updates_by_id.get(record["id"])
            if replacement is None:
                updated_records.append(record)
            else:
                updated_records.append(replacement)
        return updated_records

    def _manifest(
        self,
        mode: str,
        fraction: float,
        affected_records: Sequence[Dict[str, Any]],
        checksum_before: str,
        checksum_after: str,
        dims: List[int],
        angle_degrees: Optional[float],
        sigma: Optional[float],
        rotation_pairs: Sequence[Tuple[int, int]],
    ) -> dict:
        injection_index = self._injection_index
        self._injection_index += 1
        return {
            "mode": mode,
            "seed": self.seed,
            "injection_index": injection_index,
            "fraction": float(fraction),
            "affected_ids": [record["id"] for record in affected_records],
            "affected_count": len(affected_records),
            "angle_degrees": angle_degrees,
            "sigma": sigma,
            "dims": list(dims),
            "rotation_pairs": [[int(a), int(b)] for a, b in rotation_pairs],
            "checksum_before": checksum_before,
            "checksum_after": checksum_after,
        }

    def _rotation_pairs(self, dims: Sequence[int]) -> List[Tuple[int, int]]:
        shuffled = np.array(list(dims), dtype=int)
        self._rng.shuffle(shuffled)
        return [
            (int(shuffled[index]), int(shuffled[index + 1]))
            for index in range(0, len(shuffled) - 1, 2)
        ]

    @staticmethod
    def _validate_dims(dims: Optional[list], vector_size: int) -> List[int]:
        selected = list(range(vector_size)) if dims is None else [int(dim) for dim in dims]
        if len(selected) < 2:
            raise ValueError("At least two dimensions are required for rotation drift")
        if len(set(selected)) != len(selected):
            raise ValueError("dims must not contain duplicates")
        if min(selected) < 0 or max(selected) >= vector_size:
            raise ValueError(f"dims must be within [0, {vector_size})")
        return selected

    @staticmethod
    def _vector_size(records: Sequence[Dict[str, Any]]) -> int:
        return int(records[0]["vector"].shape[0])

    @staticmethod
    def _as_vector(vector: Any) -> Tuple[np.ndarray, Optional[str]]:
        vector_name = None
        if isinstance(vector, dict):
            if len(vector) != 1:
                raise ValueError("Multi-vector Qdrant records are not supported by DriftInjector")
            vector_name, vector = next(iter(vector.items()))
            if vector_name == "":
                vector_name = None
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError(f"Expected one-dimensional vector, got shape {array.shape}")
        return array, vector_name

    @staticmethod
    def _checksum(records: Sequence[Dict[str, Any]]) -> str:
        digest = hashlib.sha256()
        for record in sorted(records, key=lambda item: repr(item["id"])):
            digest.update(repr(record["id"]).encode("utf-8"))
            digest.update(np.asarray(record["vector"], dtype=np.float32).tobytes())
        return digest.hexdigest()
