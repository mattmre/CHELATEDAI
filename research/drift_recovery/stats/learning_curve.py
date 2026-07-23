"""Nested-prefix learning-curve planning and aggregation."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np


COUNTS = (8, 16, 32, 64, 128)
ORDER_SEEDS = (7, 42, 1337, 2026, 9001)


def nested_prefixes(
    pool_size: int,
    counts: Sequence[int] = COUNTS,
    seeds: Sequence[int] = ORDER_SEEDS,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Create one permutation per seed and return its requested nested prefixes."""

    if max(counts) > pool_size:
        raise ValueError(f"pool of {pool_size} cannot supply prefix {max(counts)}")
    output: Dict[int, Dict[int, np.ndarray]] = {}
    for seed in seeds:
        order = np.random.default_rng(seed).permutation(pool_size)
        output[int(seed)] = {int(count): order[: int(count)].copy() for count in counts}
        previous = set()
        for count in sorted(counts):
            current = set(output[int(seed)][int(count)].tolist())
            if not previous.issubset(current):
                raise AssertionError("learning-curve prefixes are not nested")
            previous = current
    return output


def summarize_learning_curve(rows: Iterable[dict]) -> list:
    """Aggregate seed rows without mixing supervision types or methods."""

    grouped: Dict[tuple, list] = {}
    for row in rows:
        key = (row["supervision_type"], row["method"], int(row["supervision_count"]))
        grouped.setdefault(key, []).append(float(row["ndcg"]))
    summary = []
    for (supervision_type, method, count), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        summary.append(
            {
                "supervision_type": supervision_type,
                "method": method,
                "supervision_count": count,
                "seed_count": len(values),
                "mean_ndcg": float(array.mean()),
                "std_ndcg": float(array.std(ddof=1)) if len(values) > 1 else 0.0,
                "min_ndcg": float(array.min()),
                "max_ndcg": float(array.max()),
            }
        )
    return summary
