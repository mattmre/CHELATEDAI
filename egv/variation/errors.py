"""Variation-slice error taxonomy."""

from __future__ import annotations

from ..errors import EGVError, PhaseUnavailable


class VariationError(EGVError):
    """Base class for bounded Variation failures."""


class VariationConfigurationError(VariationError, ValueError):
    """The frozen Variation contract or an immutable input is invalid."""


class VariationDependencyError(VariationError):
    """A required local model/runtime dependency is absent or incompatible."""


class VariationIsolationError(VariationError):
    """An arm, workspace, retrieval, or artifact boundary was crossed."""


class VariationCheckpointError(VariationError):
    """A checkpoint is missing, corrupt, stale, or inconsistent."""


class VariationBudgetError(VariationError):
    """A requested loop budget exceeds the frozen bounded contract."""


__all__ = [
    "PhaseUnavailable",
    "VariationBudgetError",
    "VariationCheckpointError",
    "VariationConfigurationError",
    "VariationDependencyError",
    "VariationError",
    "VariationIsolationError",
]
