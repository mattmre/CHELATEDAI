"""Steering-post bank with a prune / re-anneal lifecycle (Phase II — H5a).

A bank of correction *posts* over a retrieval store. Each post pairs a cluster
**centroid** with a **correction operator** (any callable ``vec -> vec``; in
production a bounded near-identity adapter). The bank routes each vector to its
nearest post by centroid cosine similarity and applies that post's correction.

What makes this a *living* bank rather than a static steering bank (SVF) or a
one-shot router (MoLE): posts carry a **fitness**, are **pruned** when fitness
drops below a temperature-modulated threshold (disintegration), and
**re-annealed** from a factory when drift fires — and every mutation is logged so
a caller can prove the store changed.

Deliberately **decoupled from torch / the engine**: a post's correction is any
callable and centroids are plain numpy, so the lifecycle is fully unit-testable
with stub corrections. The drift-recovery harness conditions (C5 / C5s / C5r)
supply the real bounded adapters and the store read/write. This module owns the
bank + lifecycle only — no GNN, no learning loop here.

This is the H5a substrate; the head-to-head campaign that decides whether the
living bank beats the static-bank and one-shot-router baselines is a separate,
GPU-gated slice and is NOT claimed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

Correction = Callable[[np.ndarray], np.ndarray]
PostFactory = Callable[[str], Tuple[Iterable[float], Correction]]


@dataclass
class SteeringPost:
    """One post: a centroid + a correction operator + lifecycle bookkeeping."""

    key: str
    centroid: np.ndarray
    correct: Correction
    fitness: float = 0.0
    age: int = 0
    origin: str = "registered"  # "registered" | "re-annealed"

    def summary(self) -> Dict[str, Any]:
        """JSON-safe summary (no centroid vector / no callable)."""
        return {
            "key": self.key,
            "dim": int(self.centroid.size),
            "fitness": float(self.fitness),
            "age": int(self.age),
            "origin": self.origin,
        }


class SteeringPostBank:
    """A living bank of steering posts with a prune / re-anneal lifecycle.

    Args:
        input_dim: expected centroid / vector dimension.
        prune_below: base fitness threshold; a post is a prune candidate when its
            fitness is strictly below the *effective* threshold (see anneal_step).
        min_posts: never prune below this many posts (keep the fittest).
        temperature: lifecycle temperature in [0, 1]. High = explore (lenient
            pruning), low = stabilize (strict pruning). The effective prune
            threshold is ``prune_below * (1 - temperature)``, so at T=1 nothing is
            pruned for low fitness and at T=0 the full threshold applies.
    """

    def __init__(
        self,
        input_dim: int,
        prune_below: float = 0.0,
        min_posts: int = 1,
        temperature: float = 0.0,
    ) -> None:
        if int(input_dim) <= 0:
            raise ValueError("input_dim must be a positive integer")
        self.input_dim = int(input_dim)
        self.prune_below = float(prune_below)
        self.min_posts = max(0, int(min_posts))
        self._temperature = self._clamp_temperature(temperature)
        self._posts: Dict[str, SteeringPost] = {}
        self._log: List[Dict[str, Any]] = []

    # -- temperature ---------------------------------------------------------
    @staticmethod
    def _clamp_temperature(t: float) -> float:
        return float(min(1.0, max(0.0, float(t))))

    @property
    def temperature(self) -> float:
        return self._temperature

    def anneal_step(self, temperature: float) -> float:
        """Set the lifecycle temperature (explore<->stabilize) and log it.

        Returns the effective prune threshold now in force.
        """
        self._temperature = self._clamp_temperature(temperature)
        eff = self.effective_prune_threshold()
        self._log.append(
            {"action": "anneal", "temperature": self._temperature, "effective_prune_threshold": eff}
        )
        return eff

    def effective_prune_threshold(self) -> float:
        return self.prune_below * (1.0 - self._temperature)

    # -- registration --------------------------------------------------------
    @property
    def posts(self) -> List[SteeringPost]:
        return list(self._posts.values())

    def __len__(self) -> int:
        return len(self._posts)

    def post(self, key: str) -> Optional[SteeringPost]:
        return self._posts.get(str(key))

    def register_post(
        self,
        key: str,
        centroid: Iterable[float],
        correct: Correction,
        fitness: float = 0.0,
        origin: str = "registered",
    ) -> SteeringPost:
        vector = np.asarray(list(centroid), dtype=float)
        if vector.ndim != 1 or vector.size != self.input_dim:
            raise ValueError(f"centroid must be a 1D vector of dim {self.input_dim}")
        if not callable(correct):
            raise ValueError("correct must be callable")
        post = SteeringPost(str(key), vector, correct, float(fitness), 0, str(origin))
        self._posts[post.key] = post
        self._log.append({"action": "register", "key": post.key, "origin": post.origin})
        return post

    # -- routing + apply -----------------------------------------------------
    def route(self, vector: Iterable[float]) -> SteeringPost:
        """Return the post whose centroid is most cosine-similar to ``vector``."""
        if not self._posts:
            raise ValueError("no posts registered")
        query = np.asarray(list(vector), dtype=float)
        if query.ndim != 1 or query.size != self.input_dim:
            raise ValueError(f"vector must be a 1D vector of dim {self.input_dim}")
        qnorm = float(np.linalg.norm(query))
        if qnorm == 0.0:
            raise ValueError("vector must be non-zero")
        best: Optional[SteeringPost] = None
        best_score = -np.inf
        # Deterministic tie-break by key (sorted) so routing is reproducible.
        for key in sorted(self._posts):
            post = self._posts[key]
            cnorm = float(np.linalg.norm(post.centroid))
            score = 0.0 if cnorm == 0.0 else float(np.dot(query, post.centroid) / (qnorm * cnorm))
            if score > best_score:
                best_score = score
                best = post
        assert best is not None
        return best

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        """Route every row of ``vectors`` to its post and apply that correction.

        Returns the corrected array (same shape) and logs a store-mutation record
        with per-post hit counts and the mean per-vector correction norm — the
        evidence that the bank actually changed the store.
        """
        arr = np.asarray(vectors, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != self.input_dim:
            raise ValueError(f"vectors must be (N, {self.input_dim})")
        if not self._posts:
            raise ValueError("no posts registered")
        out = np.empty_like(arr)
        hits: Dict[str, int] = {}
        total_norm = 0.0
        for i in range(arr.shape[0]):
            post = self.route(arr[i])
            corrected = np.asarray(post.correct(arr[i]), dtype=float)
            if corrected.shape != arr[i].shape:
                raise ValueError(
                    f"post {post.key!r} correction changed shape {arr[i].shape}->{corrected.shape}"
                )
            out[i] = corrected
            hits[post.key] = hits.get(post.key, 0) + 1
            total_norm += float(np.linalg.norm(corrected - arr[i]))
        mean_norm = total_norm / arr.shape[0] if arr.shape[0] else 0.0
        self._log.append(
            {
                "action": "apply",
                "n_vectors": int(arr.shape[0]),
                "per_post_hits": dict(hits),
                "mean_correction_norm": float(mean_norm),
            }
        )
        return out

    # -- lifecycle: fitness -> prune -> re-anneal ----------------------------
    def record_fitness(self, key: str, fitness: float, decay: float = 0.0) -> float:
        """Update a post's fitness (optionally as an EMA via ``decay`` in [0,1))."""
        post = self._posts.get(str(key))
        if post is None:
            raise KeyError(f"no post {key!r}")
        d = float(min(1.0, max(0.0, decay)))
        post.fitness = (d * post.fitness) + ((1.0 - d) * float(fitness)) if d > 0.0 else float(fitness)
        post.age += 1
        return post.fitness

    def prune(self) -> List[str]:
        """Remove posts whose fitness is below the effective (temperature-scaled)
        threshold, never dropping below ``min_posts`` (keeps the fittest).

        Returns the pruned keys (the disintegration set), and logs them.
        """
        threshold = self.effective_prune_threshold()
        candidates = [k for k, p in self._posts.items() if p.fitness < threshold]
        if not candidates:
            self._log.append({"action": "prune", "pruned": [], "threshold": threshold})
            return []
        # Respect min_posts: keep the fittest overall.
        keep_floor = self.min_posts
        if len(self._posts) - len(candidates) < keep_floor:
            # Sort candidates by fitness ascending; only prune the weakest until
            # we would hit the floor.
            ranked = sorted(candidates, key=lambda k: self._posts[k].fitness)
            max_prunable = max(0, len(self._posts) - keep_floor)
            candidates = ranked[:max_prunable]
        pruned: List[str] = []
        for key in candidates:
            del self._posts[key]
            pruned.append(key)
        self._log.append({"action": "prune", "pruned": pruned, "threshold": threshold})
        return pruned

    def re_anneal(self, factory: PostFactory, keys: Iterable[str]) -> List[str]:
        """Recreate posts for ``keys`` via ``factory(key) -> (centroid, correct)``.

        Re-annealed posts start at fitness 0 / age 0 with origin "re-annealed".
        Returns the keys actually re-annealed; logs them.
        """
        rebuilt: List[str] = []
        for key in keys:
            centroid, correct = factory(str(key))
            self.register_post(str(key), centroid, correct, fitness=0.0, origin="re-annealed")
            rebuilt.append(str(key))
        self._log.append({"action": "re_anneal", "keys": rebuilt})
        return rebuilt

    def prune_and_reanneal(
        self, factory: PostFactory, drift_fired: bool
    ) -> Dict[str, List[str]]:
        """One disintegration + re-anneal cycle: when ``drift_fired`` is True,
        prune low-fitness posts and re-anneal them from ``factory``.

        Returns ``{"pruned": [...], "reannealed": [...]}``. A no-op when drift did
        not fire (the lifecycle only mutates under a real drift signal — mirrors
        the engine's detector-gated correction).
        """
        if not drift_fired:
            self._log.append({"action": "prune_and_reanneal", "drift_fired": False})
            return {"pruned": [], "reannealed": []}
        pruned = self.prune()
        reannealed = self.re_anneal(factory, pruned)
        return {"pruned": pruned, "reannealed": reannealed}

    # -- introspection -------------------------------------------------------
    @property
    def lifecycle_log(self) -> List[Dict[str, Any]]:
        return [dict(entry) for entry in self._log]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_type": "steering_post_bank",
            "input_dim": self.input_dim,
            "prune_below": self.prune_below,
            "min_posts": self.min_posts,
            "temperature": self._temperature,
            "posts": [p.summary() for p in self._posts.values()],
            "lifecycle_log": self.lifecycle_log,
        }
