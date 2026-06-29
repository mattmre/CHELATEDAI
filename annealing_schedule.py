"""Annealing temperature schedule for the steering-post bank lifecycle
(Phase II — H6, rung 11).

One temperature schedule drives the bank's **explore -> stabilize** phases: high
temperature early (explore — lenient pruning, the bank keeps and tries many
posts), cooling to low temperature (stabilize — strict pruning, only the fittest
posts survive). Temperature is in ``[0, 1]`` to match
``SteeringPostBank.anneal_step`` (1 = full explore, 0 = full stabilize).

This is the rung-11 "one temperature schedule" — distinct from the engine's
scalar ``set_temperature``, which only rescales similarity scores and schedules
no lifecycle. Five schedule types mirror the teacher-weight scheduler family:
``constant`` / ``linear`` / ``cosine`` / ``step`` / ``adaptive``. The first four
are deterministic functions of the cycle index; ``adaptive`` maps a live drift
magnitude to temperature (more drift -> hotter -> explore). stdlib-only (``math``),
no torch / numpy, so it is CI-cheap and import-safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

SCHEDULES = ("constant", "linear", "cosine", "step", "adaptive")


def _clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, float(x))))


@dataclass
class AnnealingSchedule:
    """A temperature schedule in [0,1] for the post-bank lifecycle.

    Args:
        schedule: one of ``constant`` / ``linear`` / ``cosine`` / ``step`` /
            ``adaptive``.
        n_cycles: number of correction cycles the schedule spans (>= 1).
        t_start: explore-end temperature (cycle 0), default 1.0.
        t_end: stabilize-end temperature (final cycle), default 0.0.
        step_fraction: for ``step``, the fraction of cycles spent at ``t_start``
            before dropping to ``t_end`` (in [0, 1]).
        drift_reference: for ``adaptive``, the drift magnitude that maps to
            ``t_start`` (full explore); drift >= this saturates at ``t_start``.
    """

    schedule: str = "cosine"
    n_cycles: int = 12
    t_start: float = 1.0
    t_end: float = 0.0
    step_fraction: float = 0.5
    drift_reference: float = 1.0

    def __post_init__(self) -> None:
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}, got {self.schedule!r}")
        if int(self.n_cycles) <= 0:
            raise ValueError("n_cycles must be a positive integer")
        if not (0.0 <= float(self.step_fraction) <= 1.0):
            raise ValueError("step_fraction must be in [0, 1]")
        if float(self.drift_reference) <= 0.0:
            raise ValueError("drift_reference must be positive")
        self.n_cycles = int(self.n_cycles)
        self.t_start = _clamp01(self.t_start)
        self.t_end = _clamp01(self.t_end)
        self.step_fraction = float(self.step_fraction)
        self.drift_reference = float(self.drift_reference)

    def _progress(self, cycle: int) -> float:
        """Map a cycle index to progress in [0, 1] (0 = first cycle, 1 = last)."""
        if self.n_cycles <= 1:
            return 1.0
        c = min(max(int(cycle), 0), self.n_cycles - 1)
        return c / (self.n_cycles - 1)

    def temperature(self, cycle: int, drift_magnitude: Optional[float] = None) -> float:
        """Temperature for ``cycle``.

        For the four deterministic schedules ``drift_magnitude`` is ignored. For
        ``adaptive`` it is required and drives the temperature (cycle is ignored).
        """
        if self.schedule == "adaptive":
            if drift_magnitude is None:
                raise ValueError("adaptive schedule requires a drift_magnitude")
            ratio = min(1.0, abs(float(drift_magnitude)) / self.drift_reference)
            return _clamp01(self.t_end + (self.t_start - self.t_end) * ratio)

        p = self._progress(cycle)
        if self.schedule == "constant":
            return self.t_start
        if self.schedule == "linear":
            return _clamp01(self.t_start + (self.t_end - self.t_start) * p)
        if self.schedule == "cosine":
            # cos(0)=1 -> t_start at p=0 (explore); cos(pi)=-1 -> t_end at p=1.
            return _clamp01(self.t_end + 0.5 * (self.t_start - self.t_end) * (1.0 + math.cos(math.pi * p)))
        if self.schedule == "step":
            return self.t_start if p < self.step_fraction else self.t_end
        raise ValueError(f"unhandled schedule {self.schedule!r}")  # pragma: no cover

    def schedule_over_cycles(self) -> List[float]:
        """The deterministic temperature for every cycle (not valid for adaptive)."""
        if self.schedule == "adaptive":
            raise ValueError("adaptive schedule depends on live drift, not the cycle index")
        return [self.temperature(c) for c in range(self.n_cycles)]

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "record_type": "annealing_schedule",
            "schedule": self.schedule,
            "n_cycles": self.n_cycles,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "step_fraction": self.step_fraction,
            "drift_reference": self.drift_reference,
        }
        if self.schedule != "adaptive":
            d["temperatures"] = self.schedule_over_cycles()
        return d


def create_annealing_schedule(schedule: str = "cosine", n_cycles: int = 12, **kwargs: Any) -> AnnealingSchedule:
    """Factory mirroring the teacher-weight-scheduler API."""
    return AnnealingSchedule(schedule=schedule, n_cycles=int(n_cycles), **kwargs)
