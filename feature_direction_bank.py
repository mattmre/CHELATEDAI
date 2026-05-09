"""
Feature direction bank for principled steering vector assignment.

Replaces hash-based one-hot directions with seeded Gaussian unit vectors
that provide full hypersphere coverage with no axis-alignment bias.
Deterministic per feature_id. Supports optional upgrade to real SAE decoder rows.
"""
from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np


class FeatureDirectionBank:
    """Maps feature IDs to unit-norm steering directions in embedding space.

    Default: seeded Gaussian unit vectors (full hypersphere coverage,
    no axis-alignment, deterministic per feature_id string).

    Upgrade path: call update_from_activation(feature_id, decoder_row) to
    replace the Gaussian approximation with real SAE decoder rows when
    they become available.
    """

    def __init__(self, dim: int, seed_salt: str = "chelated_direction_bank_v1") -> None:
        self._dim = dim
        self._salt = seed_salt
        self._overrides: Dict[str, np.ndarray] = {}

    def get_direction(self, feature_id: str) -> np.ndarray:
        """Return a unit-norm direction for feature_id.

        Uses override (real SAE decoder row) if available, otherwise
        generates a deterministic seeded Gaussian unit vector.
        """
        if feature_id in self._overrides:
            return self._overrides[feature_id].copy()
        return self._gaussian_unit_vector(feature_id)

    def update_from_activation(self, feature_id: str, decoder_row: np.ndarray) -> None:
        """Register a real SAE decoder row for a feature_id.

        Normalizes to unit norm before storing. Once registered,
        get_direction() returns this row instead of the Gaussian approximation.
        """
        row = np.array(decoder_row, dtype=float)
        norm = np.linalg.norm(row)
        if norm < 1e-8:
            raise ValueError(f"decoder_row for {feature_id!r} is near-zero (norm={norm})")
        self._overrides[feature_id] = row / norm

    def _gaussian_unit_vector(self, feature_id: str) -> np.ndarray:
        """Deterministic seeded Gaussian unit vector for feature_id.

        Uses SHA-256 of (salt + feature_id) as a 32-byte seed for numpy's
        default_rng to ensure full hypersphere coverage and reproducibility
        across Python versions (hash() is not stable across processes).
        """
        digest = hashlib.sha256(f"{self._salt}:{feature_id}".encode()).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self._dim)
        norm = np.linalg.norm(v)
        if norm < 1e-8:  # astronomically unlikely but guard it
            v = np.zeros(self._dim)
            v[0] = 1.0
            return v
        return v / norm

    @property
    def dim(self) -> int:
        return self._dim

    def override_count(self) -> int:
        return len(self._overrides)
