"""Opt-in adapter routing utilities for future MoE-style retrieval adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np

from chelation_logger import get_logger


@dataclass
class AdapterRoute:
    """Selected adapter route and score."""

    key: str
    score: float
    adapter: Any


class AdapterRouter:
    """Route query vectors to registered adapters by centroid similarity."""

    def __init__(self, logger=None):
        self._routes: Dict[str, tuple[np.ndarray, Any]] = {}
        self.logger = logger or get_logger()

    def register(self, key: str, centroid: Iterable[float], adapter: Any) -> None:
        vector = np.array(list(centroid), dtype=float)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("centroid must be a non-empty 1D vector")
        self._routes[key] = (vector, adapter)

    def select(self, query_vector: Iterable[float], fallback: Optional[Callable[[], Any]] = None) -> AdapterRoute:
        query = np.array(list(query_vector), dtype=float)
        if query.ndim != 1 or query.size == 0:
            raise ValueError("query_vector must be a non-empty 1D vector")
        if not self._routes:
            if fallback is None:
                raise ValueError("no adapters registered and no fallback provided")
            route = AdapterRoute(key="fallback", score=0.0, adapter=fallback())
            self.logger.log_event(
                "adapter_route_selected",
                "Selected fallback adapter route",
                route_key=route.key,
                route_score=route.score,
                level="DEBUG",
            )
            return route

        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            raise ValueError("query_vector must be non-zero")

        best_key = None
        best_score = -float("inf")
        best_adapter = None
        for key, (centroid, adapter) in self._routes.items():
            centroid_norm = np.linalg.norm(centroid)
            score = 0.0 if centroid_norm == 0 else float(np.dot(query, centroid) / (query_norm * centroid_norm))
            if score > best_score:
                best_key = key
                best_score = score
                best_adapter = adapter

        route = AdapterRoute(key=str(best_key), score=best_score, adapter=best_adapter)
        self.logger.log_event(
            "adapter_route_selected",
            "Selected adapter route",
            route_key=route.key,
            route_score=route.score,
            route_count=len(self._routes),
            level="DEBUG",
        )
        return route
