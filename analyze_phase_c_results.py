"""Phase C results analysis script.

Loads ``phase_c_results.json`` produced by ``run_phase_c_eval.py``, computes
per-query NDCG\\@10 deltas against the baseline candidate, builds candidate
summaries, and identifies promotable candidates.

Usage::

    python analyze_phase_c_results.py
    python analyze_phase_c_results.py \\
        --results phase_c_results.json \\
        --output  phase_c_analysis.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ANALYSIS_SCHEMA_VERSION = "1.0"

DEFAULT_RESULTS_PATH = "phase_c_results.json"
DEFAULT_OUTPUT_PATH = "phase_c_analysis.json"

_REQUIRED_TOP_KEYS = {"candidates", "datasets", "per_query_results"}


# ---------------------------------------------------------------------------
# 1. Load + validate
# ---------------------------------------------------------------------------


def load_phase_c_results(path: str) -> Dict[str, Any]:
    """Load and validate ``phase_c_results.json`` from *path*.

    Raises:
        FileNotFoundError: if *path* does not exist.
        ValueError: if the document is missing required keys or has wrong types.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(p, encoding="utf-8") as fh:
        data = json.load(fh)
    missing = _REQUIRED_TOP_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Results file missing required keys: {sorted(missing)}")
    if not isinstance(data.get("per_query_results"), list):
        raise ValueError("'per_query_results' must be a list")
    if not isinstance(data.get("candidates"), list):
        raise ValueError("'candidates' must be a list")
    if not isinstance(data.get("datasets"), list):
        raise ValueError("'datasets' must be a list")
    return data


# ---------------------------------------------------------------------------
# 2. Per-query deltas
# ---------------------------------------------------------------------------


def compute_per_query_deltas(
    results: Dict[str, Any],
    baseline_candidate: str = "baseline",
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    r"""Compute per-query NDCG\@10 deltas for each non-baseline candidate.

    Returns:
        Nested dict ``{candidate_id: {dataset: [row, ...]}}`` where each row
        contains ``query_id``, ``dataset``, ``ndcg_at_10``, and
        ``delta_ndcg_at_10`` (``None`` if no baseline row for that query).
    """
    baseline_index: Dict[Tuple[str, str], float] = {}
    for row in results["per_query_results"]:
        if row.get("candidate_id") == baseline_candidate:
            key = (str(row["dataset"]), str(row["query_id"]))
            baseline_index[key] = float(row["ndcg_at_10"])

    per_query: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for row in results["per_query_results"]:
        cid = row.get("candidate_id", "")
        if cid == baseline_candidate:
            continue
        dataset = str(row["dataset"])
        qid = str(row["query_id"])
        baseline_ndcg = baseline_index.get((dataset, qid))
        delta: Optional[float] = (
            float(row["ndcg_at_10"]) - baseline_ndcg
            if baseline_ndcg is not None
            else None
        )
        entry: Dict[str, Any] = {
            "query_id": qid,
            "dataset": dataset,
            "ndcg_at_10": float(row["ndcg_at_10"]),
            "delta_ndcg_at_10": delta,
        }
        per_query.setdefault(cid, {}).setdefault(dataset, []).append(entry)

    return per_query


# ---------------------------------------------------------------------------
# 3. Candidate summary
# ---------------------------------------------------------------------------


def compute_candidate_summary(
    results: Dict[str, Any],
    baseline_candidate: str = "baseline",
) -> Dict[str, Dict[str, Any]]:
    r"""Build summary statistics for every candidate (including the baseline).

    Returns a dict keyed by candidate_id, each value containing:

    * ``mean_delta_ndcg_at_10`` — mean delta vs baseline across all queries
    * ``win_count``, ``loss_count``, ``tie_count`` — query-level counts
    * ``win_rate`` — win_count / total
    * ``datasets`` — sorted list of evaluated datasets
    * ``mean_ndcg_at_10`` — absolute mean NDCG\@10
    """
    deltas = compute_per_query_deltas(results, baseline_candidate)

    all_ndcg: Dict[str, List[float]] = {}
    all_datasets: Dict[str, set] = {}
    for row in results["per_query_results"]:
        cid = row["candidate_id"]
        all_ndcg.setdefault(cid, []).append(float(row["ndcg_at_10"]))
        all_datasets.setdefault(cid, set()).add(str(row["dataset"]))

    summaries: Dict[str, Dict[str, Any]] = {}

    # Baseline gets zero-delta stats
    if baseline_candidate in all_ndcg:
        bvals = all_ndcg[baseline_candidate]
        summaries[baseline_candidate] = {
            "mean_delta_ndcg_at_10": 0.0,
            "win_count": 0,
            "loss_count": 0,
            "tie_count": len(bvals),
            "win_rate": 0.0,
            "datasets": sorted(all_datasets.get(baseline_candidate, set())),
            "mean_ndcg_at_10": sum(bvals) / len(bvals) if bvals else 0.0,
        }

    for cid, dataset_rows in deltas.items():
        delta_values: List[float] = []
        win_count = 0
        loss_count = 0
        tie_count = 0
        datasets: List[str] = []
        for dataset, rows in dataset_rows.items():
            datasets.append(dataset)
            for r in rows:
                d = r["delta_ndcg_at_10"]
                if d is None:
                    tie_count += 1
                    continue
                delta_values.append(d)
                if d > 0.0:
                    win_count += 1
                elif d < 0.0:
                    loss_count += 1
                else:
                    tie_count += 1

        total = win_count + loss_count + tie_count
        win_rate = win_count / total if total > 0 else 0.0
        mean_delta = sum(delta_values) / len(delta_values) if delta_values else 0.0
        ndcg_vals = all_ndcg.get(cid, [])
        mean_ndcg = sum(ndcg_vals) / len(ndcg_vals) if ndcg_vals else 0.0

        summaries[cid] = {
            "mean_delta_ndcg_at_10": mean_delta,
            "win_count": win_count,
            "loss_count": loss_count,
            "tie_count": tie_count,
            "win_rate": win_rate,
            "datasets": sorted(set(datasets)),
            "mean_ndcg_at_10": mean_ndcg,
        }

    return summaries


# ---------------------------------------------------------------------------
# 4. Identify promotable candidates
# ---------------------------------------------------------------------------


def identify_promotable_candidates(
    summaries: Dict[str, Dict[str, Any]],
    min_win_rate: float = 0.52,
    min_mean_delta: float = 0.005,
    baseline_candidate: str = "baseline",
) -> List[str]:
    """Return candidates exceeding both promotion thresholds (strict >).

    Both conditions must hold:
    * ``win_rate > min_win_rate``
    * ``mean_delta_ndcg_at_10 > min_mean_delta``

    The baseline itself is excluded from consideration.
    """
    promotable: List[str] = []
    for cid, summary in summaries.items():
        if cid == baseline_candidate:
            continue
        if (
            summary["win_rate"] > min_win_rate
            and summary["mean_delta_ndcg_at_10"] > min_mean_delta
        ):
            promotable.append(cid)
    return promotable


# ---------------------------------------------------------------------------
# 5. Build analysis report
# ---------------------------------------------------------------------------


def build_analysis_report(
    results_path: str,
    output_path: str,
) -> Dict[str, Any]:
    """Full pipeline: load → compute → identify → write JSON report.

    Raises:
        FileNotFoundError: propagated from :func:`load_phase_c_results`.

    Returns:
        The report dict (also written to *output_path* as pretty-printed JSON).
    """
    results = load_phase_c_results(results_path)
    summaries = compute_candidate_summary(results)
    promotable = identify_promotable_candidates(summaries)

    baseline_summary = summaries.get("baseline", {})
    baseline_mean_ndcg = float(baseline_summary.get("mean_ndcg_at_10", 0.0))
    recommendation = "no_default_change" if not promotable else "review_required"

    report: Dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "results_path": str(results_path),
        "candidate_summaries": summaries,
        "promotable_candidates": promotable,
        "recommendation": recommendation,
        "baseline_mean_ndcg_at_10": baseline_mean_ndcg,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    return report


# ---------------------------------------------------------------------------
# 6. CLI helpers
# ---------------------------------------------------------------------------


def _print_summary_table(report: Dict[str, Any]) -> None:
    """Print a formatted candidate summary table to stdout."""
    summaries = report["candidate_summaries"]
    promotable_set = set(report["promotable_candidates"])

    header = (
        f"{'Candidate':<35} {'NDCG@10':>8}  {'Delta':>8}  {'Win%':>6}  Status"
    )
    print(f"\nPhase C Analysis — {report['analysis_timestamp']}")
    print(f"Recommendation : {report['recommendation']}")
    print(f"Promotable     : {report['promotable_candidates'] or 'none'}")
    print()
    print(header)
    print("-" * len(header))
    for cid, s in summaries.items():
        if cid in promotable_set:
            status = "PROMOTE"
        elif cid == "baseline":
            status = "baseline"
        else:
            status = "hold"
        print(
            f"{cid:<35} {s['mean_ndcg_at_10']:>8.4f}  "
            f"{s['mean_delta_ndcg_at_10']:>+8.4f}  "
            f"{s['win_rate']:>5.1%}  {status}"
        )
    print()


# ---------------------------------------------------------------------------
# 7. CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments, run analysis, return exit code."""
    parser = argparse.ArgumentParser(
        description="Analyse Phase C four-candidate evaluation results."
    )
    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS_PATH,
        help="Path to phase_c_results.json (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write phase_c_analysis.json (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    try:
        report = build_analysis_report(args.results, args.output)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _print_summary_table(report)
    print(f"Analysis written to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
