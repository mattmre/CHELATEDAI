"""Boundary-parity audit for the PRW numerical-tie contract (PRW-T1R linkage).

For ``p in {11, 19, 31}``, both transmitted types of the frozen two-type,
eight-layer ``typed16`` Legendre bank, and each of the 56 leading wrong-type
competitors at Hamming distance ``(7p - 1) / 2``, this module builds the
deterministic midpoint observation: the canonical truth state with the first
``d / 2`` coordinates (layer-major order) of its disagreement set flipped
toward the competitor.  Each midpoint is a mathematical cross-type tie, so the
frozen lowest-canonical-type rule must select type zero.

Three scorers are compared on every case:

* ``exact_hamming`` -- integer minimum Hamming distance per type over all
  ``16p`` shift/mask states;
* ``direct`` -- integer dot products ``sum m * roll(template) * query`` per
  state, no FFT;
* ``contract_fft`` -- the production campaign path
  (``_correlation_bank`` + ``_decode_dense_bank``) under
  ``PRW-NUMERICAL-TIE-CONTRACT-001``.

``legacy_float_fft`` replays the pre-contract scorer (float FFT correlations,
float einsum objectives, bit-exact float tie test) only as the documented
failing baseline; it is not used by the campaign.

Each midpoint also yields two adjacent strictly-unequal cases: one extra
disagreement flip toward the competitor, and one fewer.  Those differ by two
Hamming units (four correlation counts) and must rank strictly in every path.

This is a METHOD_DEV boundary check.  It makes no retrieval, production, or
novelty claim and does not test finite class-error rates.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from prime_ring_waypoint import policy_masks
from run_prime_ring_waypoint_experiment import (
    PRW_NUMERICAL_TIE_CONTRACT_ID,
    _build_carrier_templates,
    _correlation_bank,
    _decode_dense_bank,
    _phase_codebook,
)


AUDIT_ID = "PRW-T1R-NUMERICAL-TIE-BOUNDARY-AUDIT-001"
AUDIT_PRIMES = (11, 19, 31)
AUDIT_LAYERS = 8
AUDIT_SEED = 7
LEADING_COMPETITOR_COUNT = 56


def _bank(p: int) -> Tuple[np.ndarray, np.ndarray]:
    signatures, _manifest = _phase_codebook(p, AUDIT_LAYERS, 2, AUDIT_SEED)
    templates, _base, _provenance, _rings = _build_carrier_templates(
        p, AUDIT_LAYERS, "legendre", signatures, AUDIT_SEED
    )
    masks = np.asarray(policy_masks("typed16", AUDIT_LAYERS), dtype=np.int64)
    return np.asarray(templates, dtype=np.float64), masks


def _states(template: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Return every ``(shift, mask)`` state, shift-major then mask order."""

    p = template.shape[1]
    rolled = np.stack([np.roll(template, shift, axis=1) for shift in range(p)])
    states = masks[np.newaxis, :, :, np.newaxis] * rolled[:, np.newaxis, :, :]
    return states.reshape(-1, template.shape[0], p).astype(np.int64)


def _lowest_type_winner(values: Sequence[int], *, larger_is_better: bool) -> Tuple[int, int]:
    best = max(values) if larger_is_better else min(values)
    tied = [index for index, value in enumerate(values) if value == best]
    return tied[0], len(tied)


def _exact_hamming(query: np.ndarray, states: Sequence[np.ndarray]) -> Dict[str, Any]:
    distances = [int(np.min(np.sum(bank != query, axis=(1, 2)))) for bank in states]
    winner, tie_count = _lowest_type_winner(distances, larger_is_better=False)
    return {"predicted_type": winner, "tie_count": tie_count, "values": distances}


def _direct(query: np.ndarray, states: Sequence[np.ndarray]) -> Dict[str, Any]:
    objectives = [int(np.max(np.sum(bank * query, axis=(1, 2)))) for bank in states]
    winner, tie_count = _lowest_type_winner(objectives, larger_is_better=True)
    return {"predicted_type": winner, "tie_count": tie_count, "values": objectives}


def _contract_fft(query: np.ndarray, templates: np.ndarray) -> Dict[str, Any]:
    reference_ffts = np.fft.rfft(templates, axis=2)
    reference_norms = np.linalg.norm(templates, axis=2)
    correlations = _correlation_bank(
        query.astype(np.float64), reference_ffts, reference_norms, templates.shape[2]
    )
    decoded = _decode_dense_bank(correlations, "typed16")
    return {
        "predicted_type": decoded["predicted_type"],
        "tie_count": decoded["tie_count"],
        "margin": decoded["margin"],
    }


def _legacy_float_fft(query: np.ndarray, templates: np.ndarray) -> Dict[str, Any]:
    """Pre-contract scorer, replayed verbatim as the failing baseline."""

    length = templates.shape[2]
    reference_ffts = np.fft.rfft(templates, axis=2)
    reference_norms = np.linalg.norm(templates, axis=2)
    query = query.astype(np.float64)
    query_norms = np.linalg.norm(query, axis=1)
    products = np.fft.rfft(query, axis=1)[np.newaxis, :, :] * np.conjugate(
        reference_ffts
    )
    correlations = np.fft.irfft(products, n=length, axis=2)
    correlations /= (reference_norms * query_norms[np.newaxis, :])[:, :, np.newaxis]
    correlations = np.clip(correlations, -1.0, 1.0)
    allowed = np.asarray(policy_masks("typed16", AUDIT_LAYERS), dtype=np.float64)
    objectives = np.einsum("ml,tln->tmn", allowed, correlations, optimize=True) / AUDIT_LAYERS
    flattened = objectives.reshape(templates.shape[0], -1)
    scores = flattened[np.arange(templates.shape[0]), np.argmax(flattened, axis=1)]
    maximum = float(np.max(scores))
    ties = np.flatnonzero(scores == maximum)
    ordered = np.sort(scores)
    return {
        "predicted_type": int(ties[0]),
        "tie_count": int(ties.size),
        "margin": float(ordered[-1] - ordered[-2]),
    }


def _agrees(reference: Dict[str, Any], other: Dict[str, Any]) -> bool:
    return (
        reference["predicted_type"] == other["predicted_type"]
        and reference["tie_count"] == other["tie_count"]
    )


def audit_prime(p: int) -> List[Dict[str, Any]]:
    """Return one summary row per truth type for prime ``p``."""

    templates, masks = _bank(p)
    integer_templates = templates.astype(np.int64)
    states = [_states(integer_templates[t], masks) for t in range(2)]
    leading_distance = (7 * p - 1) // 2
    rows = []
    for truth_type in (0, 1):
        truth = states[truth_type][0]  # shift 0, all-positive mask
        wrong = states[1 - truth_type]
        wrong_distances = np.sum(wrong != truth, axis=(1, 2))
        competitors = wrong[wrong_distances == leading_distance]
        if competitors.shape[0] != LEADING_COMPETITOR_COUNT:
            raise RuntimeError(
                "expected 56 leading competitors, found {}".format(competitors.shape[0])
            )
        row = {
            "p": p,
            "truth_type": truth_type,
            "cases": int(competitors.shape[0]),
            "exact_hamming_ties": 0,
            "direct_ties": 0,
            "contract_fft_ties": 0,
            "legacy_float_fft_ties": 0,
            "legacy_float_fft_winner_type_0": 0,
            "legacy_float_fft_winner_type_1": 0,
            "legacy_float_fft_max_margin": 0.0,
            "midpoint_three_path_agreement": 0,
            "canonical_winner_matches_exact_hamming": 0,
            "contract_fft_max_tie_margin": 0.0,
            "adjacent_cases": 0,
            "adjacent_three_path_strict_agreement": 0,
        }
        for competitor in competitors:
            disagreements = np.flatnonzero((competitor != truth).ravel())
            half = leading_distance // 2
            midpoint = _flip(truth, disagreements[:half])
            exact = _exact_hamming(midpoint, states)
            direct = _direct(midpoint, states)
            contract = _contract_fft(midpoint, templates)
            legacy = _legacy_float_fft(midpoint, templates)
            row["exact_hamming_ties"] += exact["tie_count"] == 2
            row["direct_ties"] += direct["tie_count"] == 2
            row["contract_fft_ties"] += contract["tie_count"] == 2
            row["legacy_float_fft_ties"] += legacy["tie_count"] == 2
            row["legacy_float_fft_winner_type_{}".format(legacy["predicted_type"])] += 1
            row["legacy_float_fft_max_margin"] = max(
                row["legacy_float_fft_max_margin"], legacy["margin"]
            )
            row["contract_fft_max_tie_margin"] = max(
                row["contract_fft_max_tie_margin"], contract["margin"]
            )
            row["midpoint_three_path_agreement"] += _agrees(exact, direct) and _agrees(
                exact, contract
            )
            row["canonical_winner_matches_exact_hamming"] += (
                contract["predicted_type"] == exact["predicted_type"] == 0
            )
            for flips, expected_type in (
                (disagreements[: half + 1], 1 - truth_type),
                (disagreements[: half - 1], truth_type),
            ):
                adjacent = _flip(truth, flips)
                exact_adjacent = _exact_hamming(adjacent, states)
                direct_adjacent = _direct(adjacent, states)
                contract_adjacent = _contract_fft(adjacent, templates)
                row["adjacent_cases"] += 1
                row["adjacent_three_path_strict_agreement"] += (
                    exact_adjacent["tie_count"] == 1
                    and exact_adjacent["predicted_type"] == expected_type
                    and _agrees(exact_adjacent, direct_adjacent)
                    and _agrees(exact_adjacent, contract_adjacent)
                )
        rows.append(row)
    return rows


def _flip(truth: np.ndarray, flat_indices: np.ndarray) -> np.ndarray:
    query = truth.ravel().copy()
    query[flat_indices] *= -1
    return query.reshape(truth.shape)


def run_audit(primes: Sequence[int] = AUDIT_PRIMES) -> Dict[str, Any]:
    rows = [row for p in primes for row in audit_prime(int(p))]
    totals = {
        key: sum(int(row[key]) for row in rows)
        for key in (
            "cases",
            "exact_hamming_ties",
            "direct_ties",
            "contract_fft_ties",
            "legacy_float_fft_ties",
            "midpoint_three_path_agreement",
            "canonical_winner_matches_exact_hamming",
            "adjacent_cases",
            "adjacent_three_path_strict_agreement",
        )
    }
    passed = (
        totals["contract_fft_ties"] == totals["cases"]
        and totals["midpoint_three_path_agreement"] == totals["cases"]
        and totals["canonical_winner_matches_exact_hamming"] == totals["cases"]
        and totals["adjacent_three_path_strict_agreement"] == totals["adjacent_cases"]
    )
    return {
        "audit_id": AUDIT_ID,
        "contract_id": PRW_NUMERICAL_TIE_CONTRACT_ID,
        "evidence_mode": "METHOD_DEV",
        "primes": [int(p) for p in primes],
        "rows": rows,
        "totals": totals,
        "contract_fft_boundary_parity": "PASSED" if passed else "FAILED",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--primes", type=int, nargs="+", default=list(AUDIT_PRIMES))
    args = parser.parse_args(argv)
    result = run_audit(tuple(args.primes))
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if result["contract_fft_boundary_parity"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
