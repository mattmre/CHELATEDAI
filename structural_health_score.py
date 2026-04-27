"""Structural-health scoring for optional ES fitness penalties."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from chelation_logger import get_logger


@dataclass
class StructuralHealthResult:
    """Normalized structural-health score and components."""

    score: float
    components: Dict[str, float] = field(default_factory=dict)

    def penalty_multiplier(self, weight: float) -> float:
        if weight < 0:
            raise ValueError("weight must be non-negative")
        bounded = max(0.0, min(1.0, self.score))
        return max(0.0, 1.0 - float(weight) * (1.0 - bounded))


class StructuralHealthScore:
    """Combine topology, isomer, and stability diagnostics into a [0, 1] score."""

    def __init__(
        self,
        collapse_weight: float = 0.4,
        isomer_weight: float = 0.3,
        topology_weight: float = 0.3,
        logger=None,
    ):
        if collapse_weight < 0 or isomer_weight < 0 or topology_weight < 0:
            raise ValueError("weights must be non-negative")
        total = collapse_weight + isomer_weight + topology_weight
        if total <= 0:
            raise ValueError("at least one weight must be positive")
        self.collapse_weight = collapse_weight / total
        self.isomer_weight = isomer_weight / total
        self.topology_weight = topology_weight / total
        self.logger = logger or get_logger()

    def evaluate(
        self,
        persistent_collapse_ratio: float = 0.0,
        isomer_ratio: float = 0.0,
        topology_drift: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StructuralHealthResult:
        collapse_health = 1.0 - self._bounded(persistent_collapse_ratio)
        isomer_health = 1.0 - self._bounded(isomer_ratio)
        topology_health = 1.0 - self._bounded(topology_drift)
        score = (
            self.collapse_weight * collapse_health
            + self.isomer_weight * isomer_health
            + self.topology_weight * topology_health
        )
        components = {
            "collapse_health": collapse_health,
            "isomer_health": isomer_health,
            "topology_health": topology_health,
        }
        if metadata:
            components.update({f"metadata_{key}": value for key, value in metadata.items() if isinstance(value, (int, float))})
        result = StructuralHealthResult(score=float(score), components=components)
        self.logger.log_event(
            "structural_health_score_evaluated",
            "Evaluated structural health score",
            score=result.score,
            collapse_health=collapse_health,
            isomer_health=isomer_health,
            topology_health=topology_health,
            level="DEBUG",
        )
        return result

    @staticmethod
    def _bounded(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

