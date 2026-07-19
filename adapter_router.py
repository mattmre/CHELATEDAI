"""Opt-in adapter routing utilities for future MoE-style retrieval adaptation."""

from __future__ import annotations

import hashlib
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
    """Route query vectors to registered adapters by centroid similarity.

    A router may also carry a separately registered global adapter.  When it
    does, a cluster route is applied only when its cosine advantage over the
    global centroid reaches ``margin_delta``; otherwise selection fails safely
    to the global adapter.  Usage is recorded only when callers provide an
    explicit scope (for example ``SELECT`` or ``REPORT``), keeping ordinary
    engine diagnostics separate from promotion evidence.
    """

    def __init__(self, margin_delta: float = 0.0, logger=None):
        if margin_delta < 0.0:
            raise ValueError("margin_delta must be non-negative")
        self._routes: Dict[str, tuple[np.ndarray, Any]] = {}
        self._global_route: Optional[tuple[np.ndarray, Any]] = None
        self._margin_delta = float(margin_delta)
        self._frozen = False
        self._lock = Lock()
        self._last_route_outcome: Optional[Dict[str, Any]] = None
        self._route_history = deque(maxlen=256)
        self._usage_by_scope: Dict[str, Dict[str, int]] = {}
        self.logger = logger or get_logger()

    @property
    def margin_delta(self) -> float:
        return self._margin_delta

    @margin_delta.setter
    def margin_delta(self, value: float) -> None:
        if self._frozen:
            raise RuntimeError("adapter router is frozen")
        if float(value) < 0.0:
            raise ValueError("margin_delta must be non-negative")
        self._margin_delta = float(value)

    def register(self, key: str, centroid: Iterable[float], adapter: Any) -> None:
        if self._frozen:
            raise RuntimeError("adapter router is frozen")
        if str(key) == "global":
            raise ValueError("'global' is reserved; use register_global()")
        vector = np.array(list(centroid), dtype=float)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("centroid must be a non-empty 1D vector")
        if not np.all(np.isfinite(vector)):
            raise ValueError("centroid must contain only finite values")
        with self._lock:
            self._routes[key] = (vector, adapter)

    def register_global(self, centroid: Iterable[float], adapter: Any) -> None:
        """Register the single-global adapter used by the margin fallback."""

        if self._frozen:
            raise RuntimeError("adapter router is frozen")
        vector = np.array(list(centroid), dtype=float)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("global centroid must be a non-empty 1D vector")
        if not np.all(np.isfinite(vector)):
            raise ValueError("global centroid must contain only finite values")
        with self._lock:
            self._global_route = (vector, adapter)

    def freeze(self) -> str:
        """Lock route membership, centroids, and margin; return their checksum."""

        with self._lock:
            self._frozen = True
        return self.state_checksum()

    @property
    def frozen(self) -> bool:
        return self._frozen

    def state_checksum(self) -> str:
        """Hash promotion-critical routing state, excluding usage counters."""

        digest = hashlib.sha256()
        with self._lock:
            routes = list(sorted(self._routes.items()))
            global_route = self._global_route
            margin_delta = self._margin_delta
            frozen = self._frozen
        digest.update(repr(float(margin_delta)).encode("utf-8"))
        digest.update(b"frozen=" + str(bool(frozen)).encode("ascii"))
        for key, (centroid, _adapter) in routes:
            digest.update(str(key).encode("utf-8"))
            digest.update(np.ascontiguousarray(centroid, dtype=np.float64).tobytes())
        if global_route is not None:
            digest.update(b"global")
            digest.update(np.ascontiguousarray(global_route[0], dtype=np.float64).tobytes())
        return digest.hexdigest()

    @staticmethod
    def _cosine(query: np.ndarray, centroid: np.ndarray, query_norm: float) -> float:
        if centroid.shape != query.shape:
            raise ValueError(
                f"centroid dimension {centroid.shape} does not match query dimension {query.shape}"
            )
        centroid_norm = np.linalg.norm(centroid)
        return 0.0 if centroid_norm == 0 else float(np.dot(query, centroid) / (query_norm * centroid_norm))

    def select(
        self,
        query_vector: Iterable[float],
        fallback: Optional[Callable[[], Any]] = None,
        usage_scope: Optional[str] = None,
    ) -> AdapterRoute:
        query = np.array(list(query_vector), dtype=float)
        if query.ndim != 1 or query.size == 0:
            raise ValueError("query_vector must be a non-empty 1D vector")
        if not np.all(np.isfinite(query)):
            raise ValueError("query_vector must contain only finite values")
        with self._lock:
            routes = list(self._routes.items())
            global_route = self._global_route
        if not routes and global_route is None:
            if fallback is None:
                raise ValueError("no adapters registered and no fallback provided")
            route = AdapterRoute(key="fallback", score=0.0, adapter=fallback(), metadata={"route_count": 0})
            self._record_usage(route.key, usage_scope)
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
            score = self._cosine(query, centroid, query_norm)
            if score > best_score:
                best_key = key
                best_score = score
                best_adapter = adapter

        global_score = None
        margin = None
        used_margin_fallback = False
        if global_route is not None:
            global_centroid, global_adapter = global_route
            global_score = self._cosine(query, global_centroid, query_norm)
            margin = best_score - global_score if routes else None
            if not routes or (margin is not None and margin < self.margin_delta):
                best_key = "global"
                best_score = global_score
                best_adapter = global_adapter
                used_margin_fallback = True

        route = AdapterRoute(
            key=str(best_key),
            score=best_score,
            adapter=best_adapter,
            metadata={
                "route_count": len(routes),
                "global_score": global_score,
                "margin": margin,
                "margin_delta": self.margin_delta,
                "used_margin_fallback": used_margin_fallback,
            },
        )
        self._record_usage(route.key, usage_scope)
        self.logger.log_event(
            "adapter_route_selected",
            "Selected adapter route",
            route_key=route.key,
            route_score=route.score,
            route_count=len(routes),
            global_score=global_score,
            margin=margin,
            margin_delta=self.margin_delta,
            used_margin_fallback=used_margin_fallback,
            usage_scope=usage_scope,
            level="DEBUG",
        )
        return route

    def _record_usage(self, route_key: str, usage_scope: Optional[str]) -> None:
        if usage_scope is None:
            return
        scope = str(usage_scope).strip().upper()
        if not scope:
            raise ValueError("usage_scope must be non-empty when provided")
        with self._lock:
            histogram = self._usage_by_scope.setdefault(scope, {})
            histogram[str(route_key)] = histogram.get(str(route_key), 0) + 1

    def reset_usage(self, usage_scope: Optional[str] = None) -> None:
        """Clear all usage evidence, or one named split scope."""

        with self._lock:
            if usage_scope is None:
                self._usage_by_scope.clear()
            else:
                self._usage_by_scope.pop(str(usage_scope).strip().upper(), None)

    def get_usage_summary(self, usage_scope: Optional[str] = None) -> Dict[str, Any]:
        """Return histogram, probabilities, entropy, and distinct-route count."""

        with self._lock:
            if usage_scope is None:
                histogram: Dict[str, int] = {}
                for scoped in self._usage_by_scope.values():
                    for key, count in scoped.items():
                        histogram[key] = histogram.get(key, 0) + int(count)
                scope_name = "ALL"
            else:
                scope_name = str(usage_scope).strip().upper()
                histogram = dict(self._usage_by_scope.get(scope_name, {}))
        total = int(sum(histogram.values()))
        probabilities = {
            key: (float(count) / total if total else 0.0)
            for key, count in sorted(histogram.items())
        }
        entropy = -sum(probability * float(np.log(probability)) for probability in probabilities.values() if probability > 0.0)
        route_histogram = {key: count for key, count in sorted(histogram.items()) if key not in {"global", "fallback"}}
        route_probabilities = {key: probabilities[key] for key in route_histogram}
        return {
            "scope": scope_name,
            "total": total,
            "histogram": dict(sorted(histogram.items())),
            "p_k": probabilities,
            "entropy": float(entropy),
            "entropy_nats": float(entropy),
            "n_used": len(histogram),
            "route_histogram": route_histogram,
            "route_p_k": route_probabilities,
            "n_routes_used": len(route_histogram),
        }

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
