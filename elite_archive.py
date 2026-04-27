"""Elite candidate archive for Evolution Strategies runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class EliteCandidate:
    """Snapshot of a high-fitness ES candidate."""

    candidate_id: str
    fitness: float
    generation: int
    parameters: List[torch.Tensor]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clone_parameters(self) -> List[torch.Tensor]:
        return [param.detach().clone() for param in self.parameters]

    def to_summary(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "fitness": self.fitness,
            "generation": self.generation,
            "metadata": dict(self.metadata),
        }


class EliteArchive:
    """Keep a bounded best-k archive of candidate parameter snapshots."""

    def __init__(self, max_size: int = 3):
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.max_size = int(max_size)
        self._candidates: List[EliteCandidate] = []

    def add_candidate(
        self,
        candidate_id: str,
        fitness: float,
        generation: int,
        parameters: List[torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        candidate = EliteCandidate(
            candidate_id=candidate_id,
            fitness=float(fitness),
            generation=int(generation),
            parameters=[param.detach().clone() for param in parameters],
            metadata=metadata or {},
        )
        self._candidates.append(candidate)
        self._candidates.sort(key=lambda item: item.fitness, reverse=True)
        del self._candidates[self.max_size :]

    def best(self) -> Optional[EliteCandidate]:
        return self._candidates[0] if self._candidates else None

    def summaries(self) -> List[Dict[str, Any]]:
        return [candidate.to_summary() for candidate in self._candidates]

    def restore_best(self, params: List[torch.Tensor]) -> bool:
        best = self.best()
        if best is None:
            return False
        with torch.no_grad():
            for param, value in zip(params, best.parameters):
                param.copy_(value.to(device=param.device, dtype=param.dtype))
        return True

