"""Opt-in adapter routing utilities for future MoE-style retrieval adaptation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np

from chelation_logger import get_logger


@dataclass
class AdapterRoute:
    """Selected adapter route and score."""

    key: str
    score: float
    adapter: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return route metadata without serializing the adapter object."""

        adapter_type = type(self.adapter).__name__ if self.adapter is not None else None
        return {
            "key": self.key,
            "score": float(self.score),
            "adapter_type": adapter_type,
            "metadata": dict(self.metadata),
        }


class AdapterRouter:
    """Route query vectors to registered adapters by centroid similarity."""

    def __init__(self, logger=None):
        self._routes: Dict[str, tuple[np.ndarray, Any]] = {}
        self._lock = Lock()
        self._last_route_outcome: Optional[Dict[str, Any]] = None
        self._route_history = deque(maxlen=256)
        self.logger = logger or get_logger()

    def register(self, key: str, centroid: Iterable[float], adapter: Any) -> None:
        vector = np.array(list(centroid), dtype=float)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("centroid must be a non-empty 1D vector")
        with self._lock:
            self._routes[key] = (vector, adapter)

    def select(self, query_vector: Iterable[float], fallback: Optional[Callable[[], Any]] = None) -> AdapterRoute:
        query = np.array(list(query_vector), dtype=float)
        if query.ndim != 1 or query.size == 0:
            raise ValueError("query_vector must be a non-empty 1D vector")
        with self._lock:
            routes = list(self._routes.items())
        if not routes:
            if fallback is None:
                raise ValueError("no adapters registered and no fallback provided")
            route = AdapterRoute(key="fallback", score=0.0, adapter=fallback(), metadata={"route_count": 0})
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
        for key, (centroid, adapter) in routes:
            centroid_norm = np.linalg.norm(centroid)
            score = 0.0 if centroid_norm == 0 else float(np.dot(query, centroid) / (query_norm * centroid_norm))
            if score > best_score:
                best_key = key
                best_score = score
                best_adapter = adapter

        route = AdapterRoute(
            key=str(best_key),
            score=best_score,
            adapter=best_adapter,
            metadata={"route_count": len(routes)},
        )
        self.logger.log_event(
            "adapter_route_selected",
            "Selected adapter route",
            route_key=route.key,
            route_score=route.score,
            route_count=len(routes),
            level="DEBUG",
        )
        return route

    def record_outcome(self, route_key: str, jaccard: float, latency_ms: Optional[float] = None) -> Dict[str, Any]:
        """Record the observed effectiveness of a selected route."""

        outcome = {
            "route_key": str(route_key),
            "jaccard": float(jaccard),
        }
        if latency_ms is not None:
            outcome["latency_ms"] = float(latency_ms)
        with self._lock:
            self._last_route_outcome = outcome
            self._route_history.append(outcome)
        self.logger.log_event(
            "adapter_route_outcome",
            "Recorded adapter route outcome",
            **outcome,
            level="DEBUG",
        )
        return dict(outcome)

    def get_last_route_outcome(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._last_route_outcome) if self._last_route_outcome is not None else None

    def get_route_history(self) -> list[Dict[str, Any]]:
        with self._lock:
            return [dict(outcome) for outcome in self._route_history]

    def get_route_effectiveness(self) -> Dict[str, Any]:
        with self._lock:
            history = [dict(outcome) for outcome in self._route_history]
        by_route: Dict[str, Dict[str, Any]] = {}
        for outcome in history:
            key = outcome["route_key"]
            stats = by_route.setdefault(
                key,
                {"count": 0, "mean_jaccard": 0.0, "mean_latency_ms": None},
            )
            stats["count"] += 1
            stats["mean_jaccard"] += outcome["jaccard"]
            if "latency_ms" in outcome:
                current = stats["mean_latency_ms"]
                stats["mean_latency_ms"] = outcome["latency_ms"] if current is None else current + outcome["latency_ms"]
        for stats in by_route.values():
            count = max(1, stats["count"])
            stats["mean_jaccard"] = float(stats["mean_jaccard"] / count)
            if stats["mean_latency_ms"] is not None:
                stats["mean_latency_ms"] = float(stats["mean_latency_ms"] / count)
        return {
            "total_routes_observed": len(history),
            "last_route_outcome": history[-1] if history else None,
            "routes": by_route,
        }
