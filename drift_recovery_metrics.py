"""Metrics for drift-recovery experiment trajectories."""

from __future__ import annotations

import math
import statistics
from typing import Dict, List, Optional, Tuple


def ndcg_at_k(ranked_ids, relevant_ids, k=10) -> float:
    """Compute binary nDCG@k with IDCG from the complete positive-qrel set."""

    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    ranked = list(ranked_ids)[:k]
    if len(set(ranked)) != len(ranked):
        raise ValueError("ranked_ids must not contain duplicate document IDs")
    dcg = sum(1.0 / math.log2(rank + 2.0) for rank, doc_id in enumerate(ranked) if doc_id in relevant_set)
    ideal_count = min(k, len(relevant_set))
    idcg = sum(1.0 / math.log2(rank + 2.0) for rank in range(ideal_count))
    return float(dcg / idcg)


class RecoveryTracker:
    def __init__(self, baseline_ndcg: float, recovery_threshold: float = 0.95):
        if baseline_ndcg < 0.0:
            raise ValueError("baseline_ndcg must be >= 0")
        if recovery_threshold <= 0.0:
            raise ValueError("recovery_threshold must be > 0")
        self.baseline_ndcg = float(baseline_ndcg)
        self.recovery_threshold = float(recovery_threshold)
        self._trajectory: List[Tuple[int, float, Dict]] = []

    def record_cycle(self, cycle_index: int, ndcg: float, metadata: dict) -> None:
        self._trajectory.append((int(cycle_index), float(ndcg), dict(metadata)))

    def recovery_cycle(self) -> Optional[int]:
        """First cycle where NDCG reaches threshold for two consecutive cycles."""

        target = self.recovery_threshold * self.baseline_ndcg
        ordered = sorted(self._trajectory, key=lambda item: item[0])
        for index in range(len(ordered) - 1):
            current_cycle, current_ndcg, _current_meta = ordered[index]
            next_cycle, next_ndcg, _next_meta = ordered[index + 1]
            if next_cycle != current_cycle + 1:
                continue
            if current_ndcg >= target and next_ndcg >= target:
                return current_cycle
        return None

    def post_recovery_stability(self) -> Optional[float]:
        recovery = self.recovery_cycle()
        if recovery is None:
            return None
        values = [ndcg for cycle, ndcg, _metadata in self._trajectory if cycle >= recovery]
        if len(values) < 2:
            return 0.0
        return float(statistics.pstdev(values))

    def trajectory(self) -> list:
        return [(cycle, ndcg, dict(metadata)) for cycle, ndcg, metadata in self._trajectory]

    def to_json(self) -> dict:
        return {
            "baseline_ndcg": self.baseline_ndcg,
            "recovery_threshold": self.recovery_threshold,
            "recovery_cycle": self.recovery_cycle(),
            "post_recovery_stability": self.post_recovery_stability(),
            "trajectory": [
                {
                    "cycle_index": cycle,
                    "ndcg": ndcg,
                    "metadata": dict(metadata),
                }
                for cycle, ndcg, metadata in self._trajectory
            ],
        }
