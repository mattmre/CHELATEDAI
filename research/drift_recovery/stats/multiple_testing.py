"""Multiple-testing corrections for preregistered contrast families."""

from __future__ import annotations

from typing import Dict, Mapping


def holm_adjust(p_values: Mapping[str, float], alpha: float = 0.05) -> Dict[str, dict]:
    """Return step-down Holm adjusted p-values and rejection decisions."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    ordered = sorted((float(p), name) for name, p in p_values.items())
    if any(p < 0.0 or p > 1.0 for p, _ in ordered):
        raise ValueError("p-values must be in [0, 1]")
    count = len(ordered)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for rank, (p_value, name) in enumerate(ordered, start=1):
        running = max(running, (count - rank + 1) * p_value)
        adjusted[name] = min(1.0, running)
    still_rejecting = True
    output: Dict[str, dict] = {}
    for rank, (p_value, name) in enumerate(ordered, start=1):
        threshold = alpha / (count - rank + 1)
        reject = still_rejecting and p_value <= threshold
        if not reject:
            still_rejecting = False
        output[name] = {
            "p_value": p_value,
            "holm_adjusted_p": adjusted[name],
            "holm_threshold": threshold,
            "reject": reject,
            "rank": rank,
            "family_size": count,
        }
    return output
