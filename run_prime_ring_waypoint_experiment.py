"""Bounded synthetic METHOD_DEV campaign for prime-ring waypoint routing.

The campaign deliberately separates four objects that are easy to conflate:

* a dense cyclic carrier used to estimate one shared shift;
* a measured relative-phase (OPPW-equivalent) type code;
* a separately generated semantic payload collision group; and
* a decoder mask policy that is crossed with, not inferred from, the planted
  mask family.

This is synthetic method development.  It is not production evidence, a
novelty claim, or a Rader-transform benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from prime_ring_waypoint import (
    crt_encode_integer,
    crt_reconstruct_integer,
    factor_integer_exact,
    has_ideal_legendre_autocorrelation,
    is_prime_exact,
    legendre_carrier,
    make_template,
    observe_template,
    policy_masks,
    rademacher_carrier,
)


PROTOCOL_ID = "CHELATEDAI-PRW-v0.1"
RECORD_TYPE = "prime_ring_waypoint_method_dev_campaign"
EVIDENCE_MODE = "METHOD_DEV"
FROZEN_SEEDS = (7, 42, 1337)
MANDATORY_LENGTHS = (4091, 4096, 4691)
CARRIER_FAMILIES = ("legendre", "rademacher")
PLANTED_MASK_FAMILIES = ("none", "shared", "typed16", "free256")
DECODER_MASK_POLICIES = ("none", "shared", "typed16", "free256")
SPLITS = ("SELECT", "REPORT")
DEFAULT_OUTPUT = (
    Path("artifacts")
    / "method-dev"
    / "prime-ring"
    / "prime-ring-waypoint-v1.json"
)


class CampaignValidationError(ValueError):
    """Raised before execution when a campaign contract is malformed."""


@dataclass(frozen=True)
class CampaignConfig:
    """Frozen factors and trial budgets for one deterministic campaign.

    The fast default retains the required structural controls but is explicitly
    too small for a <=1% false-unlock claim.  ``full_config`` uses the frozen
    primary-cell budgets of 512 SELECT and 1024 REPORT groups per input class
    and seed.  Non-primary cells remain bounded descriptive controls.
    """

    seeds: Tuple[int, ...] = FROZEN_SEEDS
    lengths: Tuple[int, ...] = MANDATORY_LENGTHS
    carrier_families: Tuple[str, ...] = CARRIER_FAMILIES
    layers: Tuple[int, ...] = (1, 8)
    planted_mask_families: Tuple[str, ...] = PLANTED_MASK_FAMILIES
    decoder_mask_policies: Tuple[str, ...] = DECODER_MASK_POLICIES
    bit_flip_rates: Tuple[float, ...] = (0.45,)
    payload_noise_rates: Tuple[float, ...] = (0.25,)
    type_count: int = 4
    waypoints_per_type: int = 4
    select_planted: int = 2
    select_unrelated: int = 4
    report_planted: int = 3
    report_unrelated: int = 8
    recall_k: int = 5
    target_false_unlock_rate: float = 0.01
    confidence_level: float = 0.975
    report_stream_tag: str = "REPORT-v1"
    output: str = str(DEFAULT_OUTPUT)
    primary_select_planted: int = 2
    primary_select_unrelated: int = 4
    primary_report_planted: int = 3
    primary_report_unrelated: int = 8
    run_label: str = "SMOKE_NON_EVIDENTIARY"


def full_config(output: str = str(DEFAULT_OUTPUT)) -> CampaignConfig:
    """Return the expanded preregistered factor grid and adequate FUR budget."""

    return CampaignConfig(
        bit_flip_rates=(0.0, 0.20, 0.35, 0.45, 0.49),
        payload_noise_rates=(0.0, 0.10, 0.25, 0.50),
        type_count=16,
        waypoints_per_type=8,
        select_planted=2,
        select_unrelated=4,
        report_planted=3,
        report_unrelated=8,
        primary_select_planted=512,
        primary_select_unrelated=512,
        primary_report_planted=1024,
        primary_report_unrelated=1024,
        output=output,
        run_label="FULL_METHOD_DEV",
    )


def _plain_int(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise CampaignValidationError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise CampaignValidationError(f"{name} must be >= {minimum}")
    return result


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise CampaignValidationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CampaignValidationError(f"{name} must be a finite number")
    return result


def _validated_tuple(
    values: object,
    name: str,
    *,
    allowed: Optional[Sequence[str]] = None,
    minimum: Optional[int] = None,
) -> Tuple[Any, ...]:
    if isinstance(values, (str, bytes)):
        raise CampaignValidationError(f"{name} must be a non-empty tuple")
    try:
        result = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise CampaignValidationError(f"{name} must be a non-empty tuple") from exc
    if not result:
        raise CampaignValidationError(f"{name} must be a non-empty tuple")
    if len(set(result)) != len(result):
        raise CampaignValidationError(f"{name} must not contain duplicates")
    if allowed is not None:
        unknown = [value for value in result if value not in allowed]
        if unknown:
            raise CampaignValidationError(
                f"{name} contains unsupported values: {unknown}"
            )
    if minimum is not None:
        for index, value in enumerate(result):
            _plain_int(value, f"{name}[{index}]", minimum)
    return result


def validate_config(config: CampaignConfig) -> CampaignConfig:
    """Validate every factor up front; malformed campaigns fail closed."""

    if not isinstance(config, CampaignConfig):
        raise CampaignValidationError("config must be CampaignConfig")
    _validated_tuple(config.seeds, "seeds", minimum=0)
    _validated_tuple(config.lengths, "lengths", minimum=7)
    _validated_tuple(
        config.carrier_families,
        "carrier_families",
        allowed=CARRIER_FAMILIES,
    )
    _validated_tuple(config.layers, "layers", allowed=(1, 8))
    _validated_tuple(
        config.planted_mask_families,
        "planted_mask_families",
        allowed=PLANTED_MASK_FAMILIES,
    )
    _validated_tuple(
        config.decoder_mask_policies,
        "decoder_mask_policies",
        allowed=DECODER_MASK_POLICIES,
    )
    for name, rates, upper in (
        ("bit_flip_rates", config.bit_flip_rates, 0.5),
        ("payload_noise_rates", config.payload_noise_rates, math.inf),
    ):
        _validated_tuple(rates, name)
        for index, rate in enumerate(rates):
            number = _finite_float(rate, f"{name}[{index}]")
            if number < 0.0 or number >= upper:
                raise CampaignValidationError(
                    f"{name}[{index}] must be in [0, {upper})"
                )
    for name, value, minimum in (
        ("type_count", config.type_count, 2),
        ("waypoints_per_type", config.waypoints_per_type, 2),
        ("select_planted", config.select_planted, 1),
        ("select_unrelated", config.select_unrelated, 1),
        ("report_planted", config.report_planted, 1),
        ("report_unrelated", config.report_unrelated, 1),
        ("primary_select_planted", config.primary_select_planted, 1),
        ("primary_select_unrelated", config.primary_select_unrelated, 1),
        ("primary_report_planted", config.primary_report_planted, 1),
        ("primary_report_unrelated", config.primary_report_unrelated, 1),
        ("recall_k", config.recall_k, 1),
    ):
        _plain_int(value, name, minimum)
    if config.type_count > 64 or config.waypoints_per_type > 64:
        raise CampaignValidationError("type and waypoint counts are bounded at 64")
    target = _finite_float(
        config.target_false_unlock_rate, "target_false_unlock_rate"
    )
    if not 0.0 < target < 1.0:
        raise CampaignValidationError("target_false_unlock_rate must be in (0, 1)")
    confidence = _finite_float(config.confidence_level, "confidence_level")
    if not 0.5 < confidence < 1.0:
        raise CampaignValidationError("confidence_level must be in (0.5, 1)")
    if not isinstance(config.report_stream_tag, str) or not config.report_stream_tag:
        raise CampaignValidationError("report_stream_tag must be a non-empty string")
    if not isinstance(config.output, str) or not config.output:
        raise CampaignValidationError("output must be a non-empty path")
    if not isinstance(config.run_label, str) or not config.run_label:
        raise CampaignValidationError("run_label must be a non-empty string")
    return config


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _stable_digest(*parts: object) -> str:
    payload = "|".join(_canonical_json(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_rng(*parts: object) -> np.random.Generator:
    seed_bytes = bytes.fromhex(_stable_digest(*parts)[:32])
    return np.random.default_rng(int.from_bytes(seed_bytes, "big"))


def _stream_manifest(config: CampaignConfig) -> Dict[str, Any]:
    streams = {
        "BANK": _stable_digest(PROTOCOL_ID, "BANK")[:24],
        "SELECT": _stable_digest(PROTOCOL_ID, "SELECT")[:24],
        "REPORT": _stable_digest(
            PROTOCOL_ID, config.report_stream_tag
        )[:24],
    }
    return {
        "derivation": "sha256-domain-separated-numpy-pcg64",
        "stream_ids": streams,
        "select_report_disjoint": len(set(streams.values())) == 3,
        "report_stream_tag": config.report_stream_tag,
    }


def _mask_availability(layers: int, family: str, *, decoder: bool) -> Optional[str]:
    if layers == 8:
        return None
    if family == "none":
        return None
    kind = "decoder mask policy" if decoder else "planted mask family"
    return f"{kind} {family} requires exactly 8 layers"


def _carrier_availability(length: int, family: str) -> Optional[str]:
    if family == "legendre" and (
        length % 4 != 3 or not is_prime_exact(length)
    ):
        return "Legendre carrier requires an odd prime congruent to 3 modulo 4"
    return None


def _cell_id(spec: Mapping[str, Any]) -> str:
    return "PRW-CELL-" + _stable_digest(spec)[:20].upper()


def expected_cell_specs(config: CampaignConfig) -> Tuple[Dict[str, Any], ...]:
    """Return the complete deterministic Cartesian grid, including invalid cells."""

    validate_config(config)
    specs = []
    for (
        seed,
        length,
        carrier,
        layers,
        planted_mask,
        decoder_mask,
        bit_flip,
        payload_noise,
    ) in product(
        config.seeds,
        config.lengths,
        config.carrier_families,
        config.layers,
        config.planted_mask_families,
        config.decoder_mask_policies,
        config.bit_flip_rates,
        config.payload_noise_rates,
    ):
        specs.append(
            {
                "seed": int(seed),
                "length": int(length),
                "carrier_family": carrier,
                "layers": int(layers),
                "planted_mask_family": planted_mask,
                "decoder_mask_policy": decoder_mask,
                "bit_flip_rate": float(bit_flip),
                "payload_noise_rate": float(payload_noise),
            }
        )
    return tuple(specs)


def _cell_unavailable_reason(spec: Mapping[str, Any]) -> Optional[str]:
    reasons = [
        _carrier_availability(spec["length"], spec["carrier_family"]),
        _mask_availability(
            spec["layers"], spec["planted_mask_family"], decoder=False
        ),
        _mask_availability(
            spec["layers"], spec["decoder_mask_policy"], decoder=True
        ),
    ]
    retained = [reason for reason in reasons if reason is not None]
    return "; ".join(retained) if retained else None


def _quotient_overlap(left: np.ndarray, right: np.ndarray, length: int) -> int:
    """Return the largest coordinate agreement after one common cyclic offset."""

    differences = np.mod(right - left, length)
    _values, counts = np.unique(differences, return_counts=True)
    return int(np.max(counts))


def _measure_codebook(
    signatures: np.ndarray, length: int
) -> Tuple[int, Tuple[Dict[str, int], ...]]:
    pairs = []
    maximum = 0
    for left in range(signatures.shape[0]):
        for right in range(left + 1, signatures.shape[0]):
            overlap = _quotient_overlap(
                signatures[left], signatures[right], length
            )
            maximum = max(maximum, overlap)
            pairs.append({"left_type": left, "right_type": right, "kappa": overlap})
    return maximum, tuple(pairs)


def _phase_codebook(
    length: int,
    layers: int,
    type_count: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Construct and measure a gauge-fixed codebook with pairwise kappa <= 1.

    The affine candidate is accepted only after direct measurement.  This keeps
    the 4096 composite control matched instead of assuming field arithmetic.
    A bounded rejection sampler is retained for unusual caller-provided sizes.
    """

    multipliers = np.arange(layers, dtype=np.int64)
    signatures = np.mod(
        np.arange(type_count, dtype=np.int64)[:, np.newaxis]
        * multipliers[np.newaxis, :],
        length,
    )
    maximum, pairs = _measure_codebook(signatures, length)
    strategy = "measured_affine"
    attempts = 0
    if layers > 1 and maximum > 1:
        strategy = "bounded_rejection_sample"
        rng = _stable_rng(PROTOCOL_ID, "BANK", seed, "phase-codebook", length, layers)
        accepted = [np.zeros(layers, dtype=np.int64)]
        while len(accepted) < type_count and attempts < 100_000:
            attempts += 1
            candidate = np.zeros(layers, dtype=np.int64)
            candidate[1:] = rng.choice(
                length, size=layers - 1, replace=False
            ).astype(np.int64)
            if all(
                _quotient_overlap(existing, candidate, length) <= 1
                for existing in accepted
            ):
                accepted.append(candidate)
        if len(accepted) != type_count:
            raise CampaignValidationError(
                "failed to construct a measured kappa<=1 phase codebook "
                "within 100000 attempts"
            )
        signatures = np.vstack(accepted)
        maximum, pairs = _measure_codebook(signatures, length)
    if layers > 1 and maximum > 1:
        raise CampaignValidationError("phase codebook violates the kappa<=1 contract")
    signatures = np.ascontiguousarray(signatures, dtype=np.int64)
    signatures.setflags(write=False)
    histogram: Dict[str, int] = {}
    for pair in pairs:
        key = str(pair["kappa"])
        histogram[key] = histogram.get(key, 0) + 1
    return signatures, {
        "construction": strategy,
        "gauge_fixed_first_coordinate": bool(
            np.all(signatures[:, 0] == 0)
        ),
        "measured_max_kappa": maximum,
        "kappa_histogram": histogram,
        "pair_count": len(pairs),
        "rejection_attempts": attempts,
        "oppw_equivalent_phase_address": True,
        "phase_address_novelty_claim": False,
        "digest": _stable_digest(signatures.tolist()),
    }


def _planted_masks(family: str, layers: int) -> np.ndarray:
    if family == "none":
        masks = np.ones((1, layers), dtype=np.int8)
    elif family == "shared":
        masks = policy_masks("shared", layers)
    elif family == "typed16":
        masks = policy_masks("typed16", layers)
    elif family == "free256":
        masks = policy_masks("free256", layers)
    else:  # protected by validate_config; retained as a local fail-closed guard
        raise CampaignValidationError(f"unsupported planted mask family {family}")
    return np.asarray(masks, dtype=np.int8)


def _build_carrier_templates(
    length: int,
    layers: int,
    family: str,
    signatures: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Tuple[Any, ...]]:
    """Build canonical all-positive templates and retain their provenance."""

    dummy_payload = np.ones((layers, length), dtype=np.float64)
    if family == "legendre":
        base = legendre_carrier(length).astype(np.float64)
        ring_templates = tuple(
            make_template(
                f"PRW-WP-TYPE-{type_index:02d}",
                dummy_payload,
                tuple(int(value) for value in signatures[type_index]),
                carrier_family="legendre",
            )
            for type_index in range(signatures.shape[0])
        )
        templates = np.stack([template.carrier for template in ring_templates])
        provenance = {
            "representation": "prime_legendre_ring",
            "template_factory": "make_template",
            "ring_observation_labels_manually_constructed": False,
            "base_shared_across_types": True,
            "composite_control": False,
        }
    elif family == "rademacher":
        # This is the true generic array path, including n=4096.  One base is
        # shared across every type; independent bases would add an illicit type
        # code.
        control_seed = int(
            _stable_digest(
                PROTOCOL_ID, "BANK", seed, "rademacher-base", length
            )[:16],
            16,
        )
        base = rademacher_carrier(length, control_seed).astype(np.float64)
        ring_templates = tuple(
            make_template(
                f"PRW-WP-TYPE-{type_index:02d}",
                dummy_payload,
                tuple(int(value) for value in signatures[type_index]),
                carrier_family="rademacher",
                base_carrier=base,
            )
            for type_index in range(signatures.shape[0])
        )
        templates = np.stack([template.carrier for template in ring_templates])
        provenance = {
            "representation": (
                "random_composite_control"
                if length == 4096
                else "generic_rademacher_array_control"
            ),
            "template_factory": "make_template",
            "ring_observation_labels_manually_constructed": False,
            "base_shared_across_types": True,
            "composite_control": length == 4096,
        }
    else:
        raise CampaignValidationError(f"unsupported carrier family {family}")
    templates = np.ascontiguousarray(templates, dtype=np.float64)
    base = np.ascontiguousarray(base, dtype=np.float64)
    templates.setflags(write=False)
    base.setflags(write=False)
    provenance.update(
        {
            "template_dtype": str(templates.dtype),
            "template_shape": list(templates.shape),
            "template_nbytes": int(templates.nbytes),
            "base_nbytes": int(base.nbytes),
            "corrupted_queries_are_raw_array_copies": True,
        }
    )
    return templates, base, provenance, ring_templates


def _correlation_bank(
    query: np.ndarray,
    reference_ffts: np.ndarray,
    reference_norms: np.ndarray,
    length: int,
) -> np.ndarray:
    """Score a raw query against every candidate/layer/rotation."""

    if query.ndim != 2 or query.shape[1] != length:
        raise CampaignValidationError("query carrier shape violates bank contract")
    query = np.asarray(query, dtype=np.float64)
    if not np.all(np.isfinite(query)):
        raise CampaignValidationError("query carrier contains non-finite values")
    query_norms = np.linalg.norm(query, axis=1)
    if np.any(query_norms <= 0.0) or not np.all(np.isfinite(query_norms)):
        raise CampaignValidationError("query carrier layers must have positive norm")
    query_fft = np.fft.rfft(query, axis=1)
    products = query_fft[np.newaxis, :, :] * np.conjugate(reference_ffts)
    correlations = np.fft.irfft(products, n=length, axis=2)
    correlations /= (
        reference_norms * query_norms[np.newaxis, :]
    )[:, :, np.newaxis]
    if not np.all(np.isfinite(correlations)):
        raise CampaignValidationError("carrier correlation produced non-finite values")
    return np.clip(correlations, -1.0, 1.0)


def _top_two_semantics(scores: np.ndarray) -> Tuple[int, float, float, int]:
    """Frozen bank rule: first canonical type wins an exact numerical tie."""

    if scores.ndim != 1 or scores.size < 2:
        raise CampaignValidationError("bank score vector must contain >=2 types")
    maximum = float(np.max(scores))
    exact_ties = np.flatnonzero(scores == maximum)
    winner = int(exact_ties[0])
    ordered = np.sort(scores)
    margin = float(ordered[-1] - ordered[-2])
    return winner, maximum, margin, int(exact_ties.size)


def _decode_dense_bank(
    correlations: np.ndarray,
    decoder_policy: str,
) -> Dict[str, Any]:
    """Return per-bank dense shared-shift results for one mask policy."""

    type_count, layers, length = correlations.shape
    if decoder_policy == "free256":
        by_shift = np.mean(np.abs(correlations), axis=1)
        shifts = np.argmax(by_shift, axis=1)
        type_scores = by_shift[np.arange(type_count), shifts]
        masks = np.vstack(
            [
                np.where(correlations[t, :, shifts[t]] >= 0.0, 1, -1)
                for t in range(type_count)
            ]
        ).astype(np.int8)
        mask_candidate_count = 256
    else:
        allowed = np.asarray(policy_masks(decoder_policy, layers), dtype=np.float64)
        objectives = np.einsum(
            "ml,tln->tmn", allowed, correlations, optimize=True
        ) / layers
        flattened = objectives.reshape(type_count, -1)
        best = np.argmax(flattened, axis=1)
        mask_index = best // length
        shifts = best % length
        type_scores = flattened[np.arange(type_count), best]
        masks = allowed[mask_index].astype(np.int8)
        mask_candidate_count = int(allowed.shape[0])
    winner, top_score, margin, tie_count = _top_two_semantics(type_scores)
    return {
        "predicted_type": winner,
        "score": top_score,
        "margin": margin,
        "tie_count": tie_count,
        "shared_shift": int(shifts[winner]),
        "mask": tuple(int(value) for value in masks[winner]),
        "candidate_type_count": type_count,
        "mask_candidate_count": mask_candidate_count,
        "search_candidate_count": int(type_count * length * mask_candidate_count),
        "working_array_bytes": int(correlations.nbytes),
    }


def _decode_independent_layers(correlations: np.ndarray) -> Dict[str, Any]:
    """Degenerate factorized control; it intentionally has no shared phase."""

    layer_best = np.max(np.abs(correlations), axis=2)
    scores = np.mean(layer_best, axis=1)
    winner, top_score, margin, tie_count = _top_two_semantics(scores)
    return {
        "predicted_type": winner,
        "score": top_score,
        "margin": margin,
        "tie_count": tie_count,
        "phase_recovery": None,
        "phase_recovery_status": "NOT_DEFINED_FOR_INDEPENDENT_LAYER_CONTROL",
        "working_array_bytes": int(correlations.nbytes + layer_best.nbytes),
    }


def _decode_sparse_oppw(
    correlations: np.ndarray,
    signatures: np.ndarray,
    length: int,
) -> Dict[str, Any]:
    """Decode the OPPW code from peaks obtained by the dense correlation path.

    This is a mathematical-form control, not a native sparse resource baseline:
    it inherits the dense observation, FFT preprocessing, and correlation bank.
    """

    # Type zero is the gauge-fixed all-zero template.  Absolute correlation
    # removes planted polarity, then one pulse (the best shift) is retained per
    # layer.  This is the required known-form baseline for the phase address.
    layer_shifts = np.argmax(np.abs(correlations[0]), axis=1).astype(np.int64)
    relative = np.mod(layer_shifts - layer_shifts[0], length)
    overlaps = np.sum(signatures == relative[np.newaxis, :], axis=1) / signatures.shape[1]
    winner, top_score, margin, tie_count = _top_two_semantics(overlaps)
    return {
        "predicted_type": winner,
        "score": top_score,
        "margin": margin,
        "tie_count": tie_count,
        "shared_shift": int(layer_shifts[0]),
        "relative_signature": tuple(int(value) for value in relative),
        "working_array_bytes_excluding_inherited_dense_bank": int(
            layer_shifts.nbytes + relative.nbytes + overlaps.nbytes
        ),
        "inherited_dense_correlation_bytes": int(correlations.nbytes),
        "baseline_family": "OPPW_from_dense_phase_estimates",
        "native_sparse_resource_baseline": False,
        "native_sparse_baseline_status": (
            "MANDATORY_UNIMPLEMENTED_MATCHED_NOISE_ENERGY_CONTRACT"
        ),
        "resource_pareto_claim_eligible": False,
        "novelty_claim": False,
    }


def _trial_group_id(seed: int, split: str, input_class: str, index: int) -> str:
    if split not in SPLITS:
        raise CampaignValidationError(f"unknown split {split}")
    return f"PRW-{split}-S{seed}-{input_class.upper()}-{index:05d}"


def _trial_truth(
    config: CampaignConfig,
    seed: int,
    split: str,
    input_class: str,
    index: int,
    length: int,
) -> Dict[str, Any]:
    stream_tag = "SELECT" if split == "SELECT" else config.report_stream_tag
    rng = _stable_rng(
        PROTOCOL_ID, stream_tag, seed, input_class, index, "truth"
    )
    return {
        "group_id": _trial_group_id(seed, split, input_class, index),
        "true_type": int(rng.integers(config.type_count)),
        "true_waypoint": int(rng.integers(config.waypoints_per_type)),
        "global_shift": int(
            int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
            % length
        ),
        "mask_draw": int(rng.integers(0, 2**31 - 1)),
    }


def _draw_planted_mask(
    family: str, layers: int, mask_draw: int
) -> Tuple[int, ...]:
    masks = _planted_masks(family, layers)
    selected = masks[int(mask_draw) % masks.shape[0]]
    return tuple(int(value) for value in selected)


def _make_planted_query(
    config: CampaignConfig,
    seed: int,
    split: str,
    truth: Mapping[str, Any],
    templates: np.ndarray,
    ring_templates: Sequence[Any],
    planted_mask_family: str,
    bit_flip_rate: float,
) -> Tuple[np.ndarray, Tuple[int, ...], Dict[str, Any]]:
    """Create an explicitly corrupted raw ndarray from one canonical template."""

    source = templates[int(truth["true_type"])]
    shift = int(truth["global_shift"])
    mask = _draw_planted_mask(
        planted_mask_family, source.shape[0], int(truth["mask_draw"])
    )
    observed = observe_template(
        ring_templates[int(truth["true_type"])],
        waypoint_id=f"PRW-WP-QUERY-{int(truth['true_type']):02d}",
        global_shift=shift,
        orientation_mask=mask,
    )
    query = observed.carrier.copy()
    stream_tag = "SELECT" if split == "SELECT" else config.report_stream_tag
    corruption_rng = _stable_rng(
        PROTOCOL_ID,
        stream_tag,
        seed,
        truth["group_id"],
        "bit-flips",
        source.shape[1],
        source.shape[0],
    )
    flips = corruption_rng.random(query.shape) < float(bit_flip_rate)
    query[flips] *= -1.0
    return np.ascontiguousarray(query), mask, {
        "representation": "raw_float64_ndarray",
        "derived_from_canonical_template": True,
        "observation_factory": "observe_template",
        "corruption": "independent_bipolar_bit_flips",
        "bit_flip_count": int(np.count_nonzero(flips)),
        "query_nbytes": int(query.nbytes),
    }


def _make_unrelated_query(
    config: CampaignConfig,
    seed: int,
    split: str,
    group_id: str,
    layers: int,
    length: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    stream_tag = "SELECT" if split == "SELECT" else config.report_stream_tag
    rng = _stable_rng(
        PROTOCOL_ID,
        stream_tag,
        seed,
        group_id,
        "unrelated-bipolar-query",
        layers,
        length,
    )
    query = rng.choice(
        np.array([-1.0, 1.0]), size=(layers, length)
    ).astype(np.float64)
    return query, {
        "representation": "raw_float64_ndarray",
        "derived_from_canonical_template": False,
        "corruption": None,
        "query_nbytes": int(query.nbytes),
    }


def _build_payload_groups(
    seed: int,
    waypoints_per_type: int,
    layers: int,
    length: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build separate float64 ring payload planes shared across type collisions."""

    rng = _stable_rng(
        PROTOCOL_ID,
        "BANK",
        seed,
        "payload-collision-groups",
        waypoints_per_type,
        layers,
        length,
    )
    payloads = rng.normal(
        size=(waypoints_per_type, layers, length)
    ).astype(np.float64)
    norms = np.linalg.norm(payloads, axis=2, keepdims=True)
    payloads /= norms
    payloads = np.ascontiguousarray(payloads)
    payloads.setflags(write=False)
    return payloads, {
        "collision_group_count": waypoints_per_type,
        "identical_group_planes_shared_across_types": True,
        "payload_values_used_for_carrier_phase_estimation": False,
        "dtype": str(payloads.dtype),
        "shape": list(payloads.shape),
        "deduplicated_array_nbytes": int(payloads.nbytes),
        "digest": _stable_digest(
            np.round(payloads[:, :, : min(32, length)], 12).tolist()
        ),
    }


def _materialize_payload_collision_bank(
    payload_groups: np.ndarray,
    type_count: int,
) -> np.ndarray:
    """Retain one deduplicated type-ambiguous bank in canonical coordinates."""

    if type_count < 2:
        raise CampaignValidationError("payload collision bank requires >=2 types")
    bank = np.ascontiguousarray(payload_groups, dtype=np.float64)
    bank.setflags(write=False)
    return bank


def _payload_query(
    config: CampaignConfig,
    seed: int,
    split: str,
    truth: Mapping[str, Any],
    payload_type_bank: np.ndarray,
    payload_noise_rate: float,
) -> np.ndarray:
    source = payload_type_bank[int(truth["true_waypoint"])]
    query = np.roll(source, int(truth["global_shift"]), axis=1).copy()
    stream_tag = "SELECT" if split == "SELECT" else config.report_stream_tag
    rng = _stable_rng(
        PROTOCOL_ID,
        stream_tag,
        seed,
        truth["group_id"],
        "payload-noise",
    )
    if payload_noise_rate > 0.0:
        query += rng.normal(scale=float(payload_noise_rate), size=query.shape)
    norms = np.linalg.norm(query, axis=1, keepdims=True)
    if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
        raise CampaignValidationError("payload query normalization failed")
    query /= norms
    return np.ascontiguousarray(query)


def _payload_scores_at_shift(
    query: np.ndarray,
    payloads: np.ndarray,
    shift: int,
) -> np.ndarray:
    shifted = np.roll(payloads, int(shift), axis=2)
    query_norms = np.linalg.norm(query, axis=1)
    payload_norms = np.linalg.norm(shifted, axis=2)
    layer_scores = np.sum(
        shifted * query[np.newaxis, :, :], axis=2
    ) / (payload_norms * query_norms[np.newaxis, :])
    return np.mean(layer_scores, axis=1)


def _payload_only_scores(
    query: np.ndarray,
    payload_type_bank: np.ndarray,
    *,
    payload_ffts: Optional[np.ndarray] = None,
    payload_norms: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], np.ndarray]:
    query_fft = np.fft.rfft(query, axis=1)
    waypoints, layers, length = payload_type_bank.shape
    if payload_ffts is None:
        payload_fft = np.fft.rfft(payload_type_bank, axis=2)
    else:
        payload_fft = np.asarray(payload_ffts)
    if payload_norms is None:
        flattened_norms = np.linalg.norm(payload_type_bank, axis=2)
    else:
        flattened_norms = np.asarray(payload_norms)
    conjugated_payload_fft = np.conjugate(payload_fft)
    fft_products = query_fft[np.newaxis, :, :] * conjugated_payload_fft
    correlations = np.fft.irfft(
        fft_products,
        n=length,
        axis=2,
    )
    normalizers = (
        np.linalg.norm(query, axis=1)[np.newaxis, :]
        * flattened_norms
    )
    correlations /= normalizers[:, :, np.newaxis]
    by_shift = np.mean(correlations, axis=1)
    best_shift = np.argmax(by_shift, axis=1)
    scores = by_shift[np.arange(waypoints), best_shift]
    component_bytes = {
        "query_fft": int(query_fft.nbytes),
        "conjugated_cached_payload_fft": int(conjugated_payload_fft.nbytes),
        "fft_products": int(fft_products.nbytes),
        "correlation_bank": int(correlations.nbytes),
        "by_shift_objective": int(by_shift.nbytes),
        "normalizers": int(normalizers.nbytes),
        "best_shift": int(best_shift.nbytes),
        "scores": int(scores.nbytes),
    }
    working_estimate = {
        "value": int(sum(component_bytes.values())),
        "unit": "bytes",
        "kind": "conservative_simultaneous_live_peak_estimate",
        "measured_process_peak": False,
        "components": component_bytes,
    }
    return (
        scores,
        best_shift,
        working_estimate,
        by_shift,
    )


def _rank_descending(scores: np.ndarray) -> np.ndarray:
    # Stable sorting preserves the frozen lowest-canonical-ID exact-tie rule.
    return np.argsort(-scores, kind="stable")


def _prepare_payload_controls(
    config: CampaignConfig,
    query: np.ndarray,
    payload_type_bank: np.ndarray,
    true_type: int,
    true_waypoint: int,
    payload_ffts: np.ndarray,
    payload_norms: np.ndarray,
) -> Dict[str, Any]:
    global_started = time.perf_counter()
    payload_only_scores, _payload_shifts, global_working_bytes, _by_shift = (
        _payload_only_scores(
            query,
            payload_type_bank,
            payload_ffts=payload_ffts,
            payload_norms=payload_norms,
        )
    )
    global_ms = (time.perf_counter() - global_started) * 1000.0
    # The global bank repeats each collision group for every type.  Exact ties
    # resolve to the lowest canonical type, intentionally exposing ambiguity.
    global_scores = np.tile(payload_only_scores, config.type_count)
    global_order = _rank_descending(global_scores)
    true_flat = true_type * config.waypoints_per_type + true_waypoint
    global_prediction = int(global_order[0])

    direct_started = time.perf_counter()
    direct_order = _rank_descending(payload_only_scores)
    direct_waypoint = int(direct_order[0])
    direct_ms = (time.perf_counter() - direct_started) * 1000.0
    k_global = min(config.recall_k, int(global_scores.size))
    k_within = min(config.recall_k, int(payload_only_scores.size))
    return {
        "payload_only_recall_at_1": bool(global_prediction == true_flat),
        "payload_only_recall_at_k": bool(
            true_flat in global_order[:k_global]
        ),
        "direct_known_type_recall_at_1": bool(
            direct_waypoint == true_waypoint
        ),
        "direct_known_type_recall_at_k": bool(
            true_waypoint in direct_order[:k_within]
        ),
        "direct_known_type_requires_side_information": True,
        "payload_latency_ms": {
            "global_payload_only_warm_query_path": global_ms,
            "direct_known_type_post_shared_group_scan": direct_ms,
            "shared_unique_group_scan": global_ms,
        },
        "payload_decoder_working_array_bytes": {
            "global_payload_only": global_working_bytes,
            "direct_known_type": {
                "value": int(payload_only_scores.nbytes * 2),
                "unit": "bytes",
                "kind": "conservative_simultaneous_live_peak_estimate",
                "measured_process_peak": False,
                "components": {
                    "shared_group_scores": int(payload_only_scores.nbytes),
                    "stable_argsort_order": int(payload_only_scores.nbytes),
                },
            },
        },
        "payload_fft_preprocessing": (
            "static template spectra cached; query FFT and correlations timed"
        ),
        "payload_query_nbytes": int(query.nbytes),
        "_query": query,
        "_k_within": k_within,
    }


def _score_payload_route(
    prepared: Mapping[str, Any],
    payload_type_bank: np.ndarray,
    true_type: int,
    true_waypoint: int,
    routed_type: int,
    routed_shift: int,
) -> Dict[str, Any]:
    routed_started = time.perf_counter()
    routed_scores = _payload_scores_at_shift(
        prepared["_query"],
        payload_type_bank,
        int(routed_shift),
    )
    routed_order = _rank_descending(routed_scores)
    routed_ms = (time.perf_counter() - routed_started) * 1000.0
    result = {
        key: value
        for key, value in prepared.items()
        if not key.startswith("_")
    }
    result.update(
        {
            "routed_waypoint_recall_at_1": bool(
                routed_type == true_type
                and int(routed_order[0]) == true_waypoint
            ),
            "routed_waypoint_recall_at_k": bool(
                routed_type == true_type
                and true_waypoint
                in routed_order[: int(prepared["_k_within"])]
            ),
            "carrier_only_waypoint_recall_at_1": bool(
                routed_type == true_type and true_waypoint == 0
            ),
        }
    )
    result["payload_latency_ms"] = {
        **result["payload_latency_ms"],
        "routed_frozen_shift": routed_ms,
    }
    result["payload_decoder_working_array_bytes"] = {
        **result["payload_decoder_working_array_bytes"],
        "routed_frozen_shift": {
            "value": int(
                2 * payload_type_bank.nbytes
                + payload_type_bank.shape[0]
                * payload_type_bank.shape[1]
                * 8
                + prepared["_query"].shape[0] * 8
                + payload_type_bank.shape[0]
                * payload_type_bank.shape[1]
                * 8
                + 2 * payload_type_bank.shape[0] * 8
            ),
            "unit": "bytes",
            "kind": "conservative_simultaneous_live_peak_estimate",
            "measured_process_peak": False,
            "components": {
                "shifted_payload_bank": int(payload_type_bank.nbytes),
                "shifted_times_query_product": int(payload_type_bank.nbytes),
                "payload_layer_norms": int(
                    payload_type_bank.shape[0]
                    * payload_type_bank.shape[1]
                    * 8
                ),
                "query_layer_norms": int(
                    prepared["_query"].shape[0] * 8
                ),
                "per_waypoint_layer_scores": int(
                    payload_type_bank.shape[0]
                    * payload_type_bank.shape[1]
                    * 8
                ),
                "waypoint_scores": int(payload_type_bank.shape[0] * 8),
                "stable_argsort_order": int(
                    payload_type_bank.shape[0] * 8
                ),
            },
        },
    }
    return result


def _summary(values: Iterable[float]) -> Dict[str, Any]:
    retained = [float(value) for value in values]
    if not retained:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    array = np.asarray(retained, dtype=np.float64)
    return {
        "count": len(retained),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Numerically stable continued fraction for the incomplete beta."""

    max_iterations = 256
    epsilon = 3.0e-14
    floor = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        twice = 2 * iteration
        numerator = (
            iteration
            * (b - iteration)
            * x
            / ((qam + twice) * (a + twice))
        )
        d = 1.0 + numerator * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + numerator / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        result *= d * c
        numerator = -(
            (a + iteration)
            * (qab + iteration)
            * x
            / ((a + twice) * (qap + twice))
        )
        d = 1.0 + numerator * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + numerator / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise CampaignValidationError(
        "regularized incomplete beta continued fraction did not converge"
    )


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        result = front * _beta_continued_fraction(a, b, x) / a
    else:
        result = 1.0 - (
            front
            * _beta_continued_fraction(b, a, 1.0 - x)
            / b
        )
    return float(min(1.0, max(0.0, result)))


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 1.0 if k >= n else 0.0
    return _regularized_incomplete_beta(
        1.0 - probability,
        float(n - k),
        float(k + 1),
    )


def clopper_pearson_upper(
    events: int, trials: int, confidence_level: float = 0.975
) -> float:
    """One-sided exact Clopper-Pearson upper confidence bound."""

    successes = _plain_int(events, "events", 0)
    total = _plain_int(trials, "trials", 1)
    confidence = _finite_float(confidence_level, "confidence_level")
    if successes > total:
        raise CampaignValidationError("events cannot exceed trials")
    if not 0.5 < confidence < 1.0:
        raise CampaignValidationError("confidence_level must be in (0.5, 1)")
    if successes == total:
        return 1.0
    alpha = 1.0 - confidence
    if successes == 0:
        return float(1.0 - alpha ** (1.0 / total))
    low, high = 0.0, 1.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if _binomial_cdf(successes, total, midpoint) > alpha:
            low = midpoint
        else:
            high = midpoint
    return float(high)


def _threshold_grid() -> Tuple[Tuple[float, float], ...]:
    scores = tuple(round(-1.0 + 0.05 * index, 10) for index in range(41))
    margins = tuple(round(0.01 * index, 10) for index in range(101))
    return tuple(product(scores, margins))


def _is_unlocked(row: Mapping[str, Any], score: float, margin: float) -> bool:
    # The protocol says both values must exceed their frozen thresholds.
    return float(row["score"]) > score and float(row["margin"]) > margin


def fit_select_thresholds(
    planted_rows: Sequence[Mapping[str, Any]],
    unrelated_rows: Sequence[Mapping[str, Any]],
    *,
    target_false_unlock_rate: float,
    confidence_level: float,
    allow_smoke_point_estimate: bool = False,
) -> Dict[str, Any]:
    """Apply the frozen grid and lexicographic SELECT-only choice rule."""

    if not planted_rows or not unrelated_rows:
        raise CampaignValidationError(
            "threshold fitting requires SELECT planted and unrelated rows"
        )
    if any(row.get("split") != "SELECT" for row in planted_rows):
        raise CampaignValidationError("planted threshold rows must all be SELECT")
    if any(row.get("split") != "SELECT" for row in unrelated_rows):
        raise CampaignValidationError("unrelated threshold rows must all be SELECT")
    eligible = []
    cp_cache: Dict[Tuple[int, int, float], float] = {}
    for score_threshold, margin_threshold in _threshold_grid():
        false_events = sum(
            _is_unlocked(row, score_threshold, margin_threshold)
            for row in unrelated_rows
        )
        cache_key = (
            false_events,
            len(unrelated_rows),
            float(confidence_level),
        )
        if cache_key not in cp_cache:
            cp_cache[cache_key] = clopper_pearson_upper(
                false_events, len(unrelated_rows), confidence_level
            )
        upper = cp_cache[cache_key]
        if upper > target_false_unlock_rate:
            continue
        true_events = sum(
            _is_unlocked(row, score_threshold, margin_threshold)
            and bool(row["type_correct"])
            for row in planted_rows
        )
        recall = true_events / len(planted_rows)
        eligible.append(
            (
                recall,
                upper,
                score_threshold,
                margin_threshold,
                false_events,
                true_events,
            )
        )
    select_ids = sorted(
        [str(row["group_id"]) for row in planted_rows]
        + [str(row["group_id"]) for row in unrelated_rows]
    )
    if not eligible and allow_smoke_point_estimate:
        smoke_candidates = []
        for score_threshold, margin_threshold in _threshold_grid():
            false_events = sum(
                _is_unlocked(row, score_threshold, margin_threshold)
                for row in unrelated_rows
            )
            false_rate = false_events / len(unrelated_rows)
            if false_rate > target_false_unlock_rate:
                continue
            true_events = sum(
                _is_unlocked(row, score_threshold, margin_threshold)
                and bool(row["type_correct"])
                for row in planted_rows
            )
            recall = true_events / len(planted_rows)
            cache_key = (
                false_events,
                len(unrelated_rows),
                float(confidence_level),
            )
            if cache_key not in cp_cache:
                cp_cache[cache_key] = clopper_pearson_upper(
                    false_events, len(unrelated_rows), confidence_level
                )
            upper = cp_cache[cache_key]
            smoke_candidates.append(
                (
                    recall,
                    false_rate,
                    score_threshold,
                    margin_threshold,
                    false_events,
                    true_events,
                    upper,
                )
            )
        if smoke_candidates:
            best_smoke = max(
                smoke_candidates,
                key=lambda item: (
                    item[0],
                    -item[1],
                    item[2],
                    item[3],
                ),
            )
            (
                recall,
                false_rate,
                score,
                margin,
                false_events,
                true_events,
                upper,
            ) = best_smoke
            return {
                "status": "SMOKE_FROZEN_POINT_ESTIMATE_ONLY",
                "source_split": "SELECT",
                "score_threshold": score,
                "margin_threshold": margin,
                "select_true_unlock_recall": recall,
                "select_true_unlock_events": true_events,
                "select_planted_groups": len(planted_rows),
                "select_false_unlock_events": false_events,
                "select_false_unlock_point_estimate": false_rate,
                "select_unrelated_groups": len(unrelated_rows),
                "select_false_unlock_cp_upper_97_5": upper,
                "target_false_unlock_rate": target_false_unlock_rate,
                "confidence_level": confidence_level,
                "fur_inference_eligible": False,
                "grid_score_count": 41,
                "grid_margin_count": 101,
                "selection_group_count": len(select_ids),
                "selection_group_digest": _stable_digest(select_ids),
                "report_rows_consumed": 0,
                "method": (
                    "frozen_grid_empirical_SELECT_only_SMOKE_no_inference"
                ),
            }
    if not eligible:
        return {
            "status": "NO_ELIGIBLE_THRESHOLD_FAIL_CLOSED",
            "source_split": "SELECT",
            "score_threshold": None,
            "margin_threshold": None,
            "grid_score_count": 41,
            "grid_margin_count": 101,
            "selection_group_count": len(select_ids),
            "selection_group_digest": _stable_digest(select_ids),
            "report_rows_consumed": 0,
            "method": "frozen_grid_exact_CP_lexicographic",
        }
    # Maximum recall; then lower upper bound; then higher score; then higher
    # margin.  Python max implements the frozen lexicographic rule after the
    # upper bound is negated.
    best = max(eligible, key=lambda item: (item[0], -item[1], item[2], item[3]))
    recall, upper, score, margin, false_events, true_events = best
    return {
        "status": "FROZEN",
        "source_split": "SELECT",
        "score_threshold": score,
        "margin_threshold": margin,
        "select_true_unlock_recall": recall,
        "select_true_unlock_events": true_events,
        "select_planted_groups": len(planted_rows),
        "select_false_unlock_events": false_events,
        "select_unrelated_groups": len(unrelated_rows),
        "select_false_unlock_cp_upper_97_5": upper,
        "target_false_unlock_rate": target_false_unlock_rate,
        "confidence_level": confidence_level,
        "grid_score_count": 41,
        "grid_margin_count": 101,
        "selection_group_count": len(select_ids),
        "selection_group_digest": _stable_digest(select_ids),
        "report_rows_consumed": 0,
        "method": "frozen_grid_exact_CP_lexicographic",
    }


def _primary_base_group(
    config: CampaignConfig,
    *,
    length: int,
    carrier_family: str,
    layers: int,
    planted_mask_family: str,
    bit_flip_rate: float,
) -> bool:
    return bool(
        config.run_label == "FULL_METHOD_DEV"
        and length == 4691
        and carrier_family == "legendre"
        and layers == 8
        and planted_mask_family == "typed16"
        and math.isclose(bit_flip_rate, 0.45, rel_tol=0.0, abs_tol=1e-12)
        and "typed16" in config.decoder_mask_policies
        and any(
            math.isclose(value, 0.25, rel_tol=0.0, abs_tol=1e-12)
            for value in config.payload_noise_rates
        )
        and config.type_count == 16
        and config.waypoints_per_type == 8
    )


def _trial_budgets(
    config: CampaignConfig, *, primary_base_group: bool
) -> Dict[str, Dict[str, int]]:
    if primary_base_group:
        return {
            "SELECT": {
                "planted": config.primary_select_planted,
                "unrelated": config.primary_select_unrelated,
                "non_cyclic": config.select_planted,
            },
            "REPORT": {
                "planted": config.primary_report_planted,
                "unrelated": config.primary_report_unrelated,
                "non_cyclic": config.report_planted,
            },
        }
    return {
        "SELECT": {
            "planted": config.select_planted,
            "unrelated": config.select_unrelated,
            "non_cyclic": config.select_planted,
        },
        "REPORT": {
            "planted": config.report_planted,
            "unrelated": config.report_unrelated,
            "non_cyclic": config.report_planted,
        },
    }


def _circular_error(observed: int, expected: int, length: int) -> int:
    delta = abs(int(observed) - int(expected)) % length
    return int(min(delta, length - delta))


def _make_noncyclic_query(
    config: CampaignConfig,
    seed: int,
    split: str,
    truth: Mapping[str, Any],
    templates: np.ndarray,
    ring_templates: Sequence[Any],
    planted_mask_family: str,
    bit_flip_rate: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    query, _mask, provenance = _make_planted_query(
        config,
        seed,
        split,
        truth,
        templates,
        ring_templates,
        planted_mask_family,
        bit_flip_rate,
    )
    stream_tag = "SELECT" if split == "SELECT" else config.report_stream_tag
    rng = _stable_rng(
        PROTOCOL_ID,
        stream_tag,
        seed,
        truth["group_id"],
        "non-cyclic-layer-perturbation",
    )
    layer_offsets = rng.integers(
        0, query.shape[1], size=query.shape[0], dtype=np.int64
    )
    # Anchor the first layer and force at least one differential shift.
    layer_offsets -= layer_offsets[0]
    layer_offsets %= query.shape[1]
    if query.shape[0] > 1 and np.all(layer_offsets == 0):
        layer_offsets[1] = 1
    perturbed = np.vstack(
        [np.roll(query[layer], int(offset)) for layer, offset in enumerate(layer_offsets)]
    )
    provenance.update(
        {
            "corruption": "bit_flips_plus_independent_layer_rotations",
            "layer_offsets": [int(value) for value in layer_offsets],
        }
    )
    return np.ascontiguousarray(perturbed), provenance


def _run_base_group(
    config: CampaignConfig,
    *,
    seed: int,
    length: int,
    carrier_family: str,
    layers: int,
    planted_mask_family: str,
    bit_flip_rate: float,
    signatures: np.ndarray,
    templates: np.ndarray,
    ring_templates: Sequence[Any],
    payload_type_bank: np.ndarray,
    payload_ffts: np.ndarray,
    payload_norms: np.ndarray,
    reference_ffts: np.ndarray,
    reference_norms: np.ndarray,
) -> Dict[str, Any]:
    """Run one observation group and return rows for every compatible decoder."""

    decoder_policies = tuple(
        policy
        for policy in config.decoder_mask_policies
        if _mask_availability(layers, policy, decoder=True) is None
    )
    rows: Dict[str, Dict[str, Dict[str, list]]] = {
        policy: {
            split: {"planted": [], "unrelated": [], "non_cyclic": []}
            for split in SPLITS
        }
        for policy in decoder_policies
    }
    primary = _primary_base_group(
        config,
        length=length,
        carrier_family=carrier_family,
        layers=layers,
        planted_mask_family=planted_mask_family,
        bit_flip_rate=bit_flip_rate,
    )
    budgets = _trial_budgets(config, primary_base_group=primary)

    for split in SPLITS:
        for input_class in ("planted", "unrelated", "non_cyclic"):
            for index in range(budgets[split][input_class]):
                truth = _trial_truth(
                    config, seed, split, input_class, index, length
                )
                if input_class == "unrelated":
                    query, query_provenance = _make_unrelated_query(
                        config,
                        seed,
                        split,
                        truth["group_id"],
                        layers,
                        length,
                    )
                    planted_mask: Optional[Tuple[int, ...]] = None
                elif input_class == "non_cyclic":
                    query, query_provenance = _make_noncyclic_query(
                        config,
                        seed,
                        split,
                        truth,
                        templates,
                        ring_templates,
                        planted_mask_family,
                        bit_flip_rate,
                    )
                    planted_mask = _draw_planted_mask(
                        planted_mask_family, layers, int(truth["mask_draw"])
                    )
                else:
                    query, planted_mask, query_provenance = _make_planted_query(
                        config,
                        seed,
                        split,
                        truth,
                        templates,
                        ring_templates,
                        planted_mask_family,
                        bit_flip_rate,
                    )

                correlation_started = time.perf_counter()
                correlations = _correlation_bank(
                    query, reference_ffts, reference_norms, length
                )
                correlation_ms = (time.perf_counter() - correlation_started) * 1000.0

                control_started = time.perf_counter()
                independent = _decode_independent_layers(correlations)
                independent_ms = (
                    time.perf_counter() - control_started
                ) * 1000.0 + correlation_ms
                oppw_started = time.perf_counter()
                oppw = _decode_sparse_oppw(correlations, signatures, length)
                oppw_ms = (
                    time.perf_counter() - oppw_started
                ) * 1000.0 + correlation_ms

                decoded_by_policy: Dict[str, Dict[str, Any]] = {}
                for policy in decoder_policies:
                    decoder_started = time.perf_counter()
                    decoded = _decode_dense_bank(correlations, policy)
                    route_ms = (
                        time.perf_counter() - decoder_started
                    ) * 1000.0 + correlation_ms
                    decoded_by_policy[policy] = decoded
                    row: Dict[str, Any] = {
                        "split": split,
                        "input_class": input_class,
                        "group_id": truth["group_id"],
                        "score": decoded["score"],
                        "margin": decoded["margin"],
                        "predicted_type": decoded["predicted_type"],
                        "tie_count": decoded["tie_count"],
                        "shared_shift": decoded["shared_shift"],
                        "route_latency_ms": route_ms,
                        "decoder_working_array_bytes": decoded[
                            "working_array_bytes"
                        ],
                        "query_provenance": query_provenance,
                        "independent_control": {
                            **independent,
                            "latency_ms": independent_ms,
                        },
                        "oppw_form_control": {
                            **oppw,
                            "latency_ms": oppw_ms,
                        },
                    }
                    if input_class != "unrelated":
                        true_type = int(truth["true_type"])
                        true_shift = int(truth["global_shift"])
                        row.update(
                            {
                                "true_type": true_type,
                                "true_waypoint": int(truth["true_waypoint"]),
                                "true_shift": true_shift,
                                "type_correct": decoded["predicted_type"]
                                == true_type,
                                "phase_exact": (
                                    decoded["predicted_type"] == true_type
                                    and decoded["shared_shift"] == true_shift
                                ),
                                "phase_circular_error": _circular_error(
                                    decoded["shared_shift"], true_shift, length
                                ),
                                "mask_recovery_identified": planted_mask
                                is not None,
                                "mask_recovered": (
                                    planted_mask is not None
                                    and tuple(decoded["mask"]) == planted_mask
                                ),
                                "independent_type_correct": independent[
                                    "predicted_type"
                                ]
                                == true_type,
                                "oppw_type_correct": oppw["predicted_type"]
                                == true_type,
                                "oppw_phase_exact": (
                                    oppw["predicted_type"] == true_type
                                    and oppw["shared_shift"] == true_shift
                                ),
                            }
                        )
                    rows[policy][split][input_class].append(row)

                if input_class == "planted" and split == "REPORT":
                    for payload_noise in config.payload_noise_rates:
                        payload_query = _payload_query(
                            config,
                            seed,
                            split,
                            truth,
                            payload_type_bank,
                            payload_noise,
                        )
                        prepared_payload = _prepare_payload_controls(
                            config,
                            payload_query,
                            payload_type_bank,
                            int(truth["true_type"]),
                            int(truth["true_waypoint"]),
                            payload_ffts,
                            payload_norms,
                        )
                        for policy, decoded in decoded_by_policy.items():
                            payload_result = _score_payload_route(
                                prepared_payload,
                                payload_type_bank,
                                int(truth["true_type"]),
                                int(truth["true_waypoint"]),
                                int(decoded["predicted_type"]),
                                int(decoded["shared_shift"]),
                            )
                            rows[policy][split]["planted"][-1].setdefault(
                                "payload_by_noise", {}
                            )[format(float(payload_noise), ".6f")] = payload_result
    return {
        "primary_base_group": primary,
        "budgets": budgets,
        "rows": rows,
    }


def _control_threshold_rows(
    rows: Sequence[Mapping[str, Any]], key: str
) -> list:
    converted = []
    for row in rows:
        control = row[key]
        converted.append(
            {
                "split": row["split"],
                "group_id": row["group_id"],
                "score": control["score"],
                "margin": control["margin"],
                "type_correct": row.get(
                    "independent_type_correct"
                    if key == "independent_control"
                    else "oppw_type_correct",
                    False,
                ),
            }
        )
    return converted


def _apply_threshold(
    rows: Sequence[Mapping[str, Any]], threshold: Mapping[str, Any]
) -> Tuple[Optional[list], Optional[list]]:
    if threshold.get("status") not in (
        "FROZEN",
        "SMOKE_FROZEN_POINT_ESTIMATE_ONLY",
    ):
        return None, None
    score = float(threshold["score_threshold"])
    margin = float(threshold["margin_threshold"])
    unlocked = [_is_unlocked(row, score, margin) for row in rows]
    correct = [
        bool(unlock and row.get("type_correct", False))
        for row, unlock in zip(rows, unlocked)
    ]
    return unlocked, correct


def _mean_bool(values: Iterable[bool]) -> Optional[float]:
    retained = [bool(value) for value in values]
    return (
        float(sum(retained) / len(retained))
        if retained
        else None
    )


def _resource_accounting(
    *,
    length: int,
    layers: int,
    type_count: int,
    waypoints_per_type: int,
    decoder_policy: str,
    templates: np.ndarray,
    base: np.ndarray,
    signatures: np.ndarray,
    payload_groups: np.ndarray,
    payload_type_bank: np.ndarray,
    reference_ffts: np.ndarray,
    reference_norms: np.ndarray,
    payload_ffts: np.ndarray,
    payload_norms: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    bits_per_residue = max(1, (length - 1).bit_length())
    mask_count = int(policy_masks(decoder_policy, layers).shape[0])
    serialized_carrier_bytes = (length + 7) // 8
    serialized_phase_codebook_bytes = math.ceil(
        type_count * max(layers - 1, 0) * bits_per_residue / 8
    )
    serialized_mask_codebook_bytes = math.ceil(mask_count * layers / 8)
    logical_type_expanded_payload_bytes = (
        type_count * waypoints_per_type * layers * length * 8
    )
    deduplicated_serialized_payload_bytes = (
        waypoints_per_type * layers * length * 8
    )
    logical_type_expanded_total_bytes = (
        serialized_carrier_bytes
        + serialized_phase_codebook_bytes
        + serialized_mask_codebook_bytes
        + logical_type_expanded_payload_bytes
    )
    deduplicated_total_bytes = (
        serialized_carrier_bytes
        + serialized_phase_codebook_bytes
        + serialized_mask_codebook_bytes
        + deduplicated_serialized_payload_bytes
    )
    logical_type_expanded_theoretical_packed_payload_bytes = math.ceil(
        type_count * waypoints_per_type * layers * length / 8
    )
    deduplicated_theoretical_packed_payload_bytes = math.ceil(
        waypoints_per_type * layers * length / 8
    )
    latency = [float(row["route_latency_ms"]) for row in rows]
    working = [
        int(row["decoder_working_array_bytes"]) for row in rows
    ]
    query_nbytes = [
        int(row["query_provenance"]["query_nbytes"]) for row in rows
    ]
    resident_arrays = {
        "canonical_base": base,
        "carrier_type_templates": templates,
        "phase_codebook": signatures,
        "payload_groups": payload_groups,
        "payload_collision_bank": payload_type_bank,
        "reference_ffts": reference_ffts,
        "reference_norms": reference_norms,
        "payload_ffts": payload_ffts,
        "payload_norms": payload_norms,
    }
    unique_buffers: Dict[int, Dict[str, Any]] = {}
    aliases: Dict[str, str] = {}
    for name, array in resident_arrays.items():
        pointer = int(array.__array_interface__["data"][0])
        if pointer in unique_buffers:
            aliases[name] = unique_buffers[pointer]["canonical_name"]
            unique_buffers[pointer]["nbytes"] = max(
                unique_buffers[pointer]["nbytes"], int(array.nbytes)
            )
        else:
            unique_buffers[pointer] = {
                "canonical_name": name,
                "nbytes": int(array.nbytes),
            }
    frequency_bins = length // 2 + 1
    correlation_bytes = type_count * layers * length * 8
    query_fft_bytes = layers * frequency_bins * 16
    products_bytes = type_count * layers * frequency_bins * 16
    objective_bytes = (
        type_count * length * 8
        if decoder_policy == "free256"
        else type_count * mask_count * length * 8
    )
    estimated_peak_decoder_bytes = (
        correlation_bytes
        + query_fft_bytes
        + products_bytes
        + objective_bytes
    )
    return {
        "logical_carrier_bits": length,
        "serialized_carrier_bytes": serialized_carrier_bytes,
        "serialized_phase_codebook_bytes": serialized_phase_codebook_bytes,
        "serialized_mask_codebook_bytes": serialized_mask_codebook_bytes,
        "logical_type_expanded_payload_bytes_float64": (
            logical_type_expanded_payload_bytes
        ),
        "logical_type_expanded_total_serialized_bytes": (
            logical_type_expanded_total_bytes
        ),
        "deduplicated_serialized_payload_bytes_float64": (
            deduplicated_serialized_payload_bytes
        ),
        "deduplicated_total_serialized_method_bytes": (
            deduplicated_total_bytes
        ),
        "logical_type_expanded_theoretical_packed_payload_bytes": (
            logical_type_expanded_theoretical_packed_payload_bytes
        ),
        "deduplicated_theoretical_packed_payload_bytes": (
            deduplicated_theoretical_packed_payload_bytes
        ),
        "deduplicated_theoretical_packed_total_bytes": (
            serialized_carrier_bytes
            + serialized_phase_codebook_bytes
            + serialized_mask_codebook_bytes
            + deduplicated_theoretical_packed_payload_bytes
        ),
        "cost_verdict_uses_theoretical_packed_bytes": False,
        "actual_numpy_resident_array_bytes": {
            "arrays": {
                name: int(array.nbytes)
                for name, array in resident_arrays.items()
            },
            "aliases_not_double_counted": aliases,
            "unique_buffer_total": int(
                sum(item["nbytes"] for item in unique_buffers.values())
            ),
            "query": _summary(query_nbytes),
            "scorer_reported_correlation_bank": _summary(working),
        },
        "estimated_decoder_peak_temporary_bytes": {
            "value": estimated_peak_decoder_bytes,
            "components": {
                "correlation_bank": correlation_bytes,
                "query_fft": query_fft_bytes,
                "fft_products": products_bytes,
                "policy_objectives_or_by_shift": objective_bytes,
            },
            "method": (
                "shape-accounting conservative simultaneous-live estimate; "
                "not process RSS"
            ),
        },
        "route_latency_ms": _summary(latency),
        "process_peak_resident_memory": {
            "status": "NOT_MEASURED",
            "reason": "portable psutil dependency not introduced",
        },
    }


def _aggregate_control(
    select_planted: Sequence[Mapping[str, Any]],
    select_unrelated: Sequence[Mapping[str, Any]],
    report_planted: Sequence[Mapping[str, Any]],
    report_unrelated: Sequence[Mapping[str, Any]],
    *,
    target_false_unlock_rate: float,
    confidence_level: float,
    allow_smoke_point_estimate: bool,
) -> Dict[str, Any]:
    threshold = fit_select_thresholds(
        select_planted,
        select_unrelated,
        target_false_unlock_rate=target_false_unlock_rate,
        confidence_level=confidence_level,
        allow_smoke_point_estimate=allow_smoke_point_estimate,
    )
    unlocked_planted, correct_planted = _apply_threshold(
        report_planted, threshold
    )
    unlocked_unrelated, _ = _apply_threshold(report_unrelated, threshold)
    if unlocked_unrelated is None:
        cp_upper = None
        false_events = None
    else:
        false_events = int(sum(unlocked_unrelated))
        cp_upper = clopper_pearson_upper(
            false_events, len(unlocked_unrelated), confidence_level
        )
    return {
        "threshold": threshold,
        "report_type_accuracy": _mean_bool(
            row.get("type_correct", False) for row in report_planted
        ),
        "report_true_unlock_recall": (
            _mean_bool(correct_planted) if correct_planted is not None else None
        ),
        "report_false_unlock_rate": (
            _mean_bool(unlocked_unrelated)
            if unlocked_unrelated is not None
            else None
        ),
        "report_false_unlock_events": false_events,
        "report_unrelated_groups": len(report_unrelated),
        "report_false_unlock_cp_upper_97_5": cp_upper,
        "fur_gate_pass": bool(
            threshold.get("status") == "FROZEN"
            and
            cp_upper is not None
            and cp_upper <= target_false_unlock_rate
        ),
    }


def _relative_mask_words(policy: str, layers: int) -> np.ndarray:
    masks = np.asarray(policy_masks(policy, layers), dtype=np.int8)
    relative = {
        tuple(int(value) for value in left * right)
        for left in masks
        for right in masks
    }
    return np.asarray(sorted(relative), dtype=np.int8)


def _planted_decoder_relative_words(
    planted_mask_family: str,
    decoder_policy: str,
    layers: int,
) -> np.ndarray:
    planted = np.asarray(
        _planted_masks(planted_mask_family, layers), dtype=np.int8
    )
    decoded = np.asarray(policy_masks(decoder_policy, layers), dtype=np.int8)
    relative = {
        tuple(int(value) for value in source * candidate)
        for source in planted
        for candidate in decoded
    }
    return np.asarray(sorted(relative), dtype=np.int8)


def _binomial_upper_tail(distance: int, crossover: float) -> float:
    """Exact-in-form binomial pairwise ML tail, evaluated by log-sum-exp."""

    d = int(distance)
    q = float(crossover)
    if d <= 0:
        return 1.0
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    threshold = (d + 1) // 2
    logs = [
        math.lgamma(d + 1)
        - math.lgamma(k + 1)
        - math.lgamma(d - k + 1)
        + k * math.log(q)
        + (d - k) * math.log1p(-q)
        for k in range(threshold, d + 1)
    ]
    maximum = max(logs)
    return float(math.exp(maximum) * sum(math.exp(value - maximum) for value in logs))


def _hamming_distance_diagnostic(
    *,
    length: int,
    layers: int,
    type_count: int,
    policy: str,
    planted_mask_family: str,
    carrier_family: str,
    base: np.ndarray,
    signatures: np.ndarray,
    bit_flip_rates: Sequence[float],
    observed_error_count: int,
    observed_trial_count: int,
) -> Dict[str, Any]:
    decoder_relative_masks = _relative_mask_words(policy, layers)
    relative_masks = _planted_decoder_relative_words(
        planted_mask_family, policy, layers
    )
    decoder_words = {
        tuple(int(value) for value in row)
        for row in policy_masks(policy, layers)
    }
    planted_words = {
        tuple(int(value) for value in row)
        for row in _planted_masks(planted_mask_family, layers)
    }
    planted_bank_compatible = planted_words.issubset(decoder_words)
    hypothesis_count = (
        type_count
        * length
        * int(policy_masks(policy, layers).shape[0])
    )
    total_bits = layers * length
    if carrier_family == "legendre":
        wrong_mask_distances = set()
        wrong_shift_distances = set()
        wrong_type_distances = set()
        all_positive = np.ones(layers, dtype=np.int8)
        # Identical type/shift but a distinct mask.
        for relative in relative_masks:
            if np.array_equal(relative, all_positive):
                continue
            inner = int(length * np.sum(relative))
            wrong_mask_distances.add((total_bits - inner) // 2)
        # Same type, different global shift: every Legendre layer contributes -1.
        unaligned = -np.ones(layers, dtype=np.int64)
        for relative in relative_masks:
            inner = int(np.dot(relative, unaligned))
            wrong_shift_distances.add((total_bits - inner) // 2)
        # Different types: only the actually measured difference patterns are
        # used; no field property is assumed for the composite control.
        for left in range(type_count):
            for right in range(left + 1, type_count):
                differences = np.mod(
                    signatures[right] - signatures[left], length
                )
                for delta in np.unique(differences):
                    correlations = np.where(
                        differences == delta, length, -1
                    ).astype(np.int64)
                    for relative in relative_masks:
                        inner = int(np.dot(relative, correlations))
                        wrong_type_distances.add((total_bits - inner) // 2)
        wrong_mask_distances.discard(0)
        wrong_shift_distances.discard(0)
        all_distances = (
            wrong_mask_distances
            | wrong_shift_distances
            | wrong_type_distances
        )
        minimum = min(all_distances)
        wrong_type_minimum = min(wrong_type_distances)
        wrong_shift_minimum = min(wrong_shift_distances)
        wrong_mask_minimum = (
            min(wrong_mask_distances) if wrong_mask_distances else None
        )
        spectrum = sorted(int(value) for value in all_distances)
        exactness = "exact_legendre_representative_distance_values"
        sidelobe = 1
    else:
        autocorrelation = np.rint(
            np.fft.irfft(
                np.fft.rfft(base) * np.conjugate(np.fft.rfft(base)),
                n=length,
            )
        ).astype(np.int64)
        sidelobe = int(np.max(np.abs(autocorrelation[1:])))
        kappa = 1 if layers > 1 else layers
        inner_upper = kappa * length + (layers - kappa) * sidelobe
        different_type_lower = math.ceil((total_bits - inner_upper) / 2)
        wrong_shift_lower = math.ceil(
            (total_bits - layers * sidelobe) / 2
        )
        mask_distances = [
            int(np.count_nonzero(relative != 1) * length)
            for relative in relative_masks
            if np.any(relative != 1)
        ]
        candidates = [different_type_lower, wrong_shift_lower]
        candidates.extend(mask_distances)
        minimum = max(1, min(candidates))
        wrong_type_minimum = max(1, different_type_lower)
        wrong_shift_minimum = max(1, wrong_shift_lower)
        wrong_mask_minimum = min(mask_distances) if mask_distances else None
        spectrum = sorted(set(candidates))
        exactness = "conservative_rademacher_sidelobe_lower_bound"
    tails = {}
    for crossover in sorted(set(float(value) for value in bit_flip_rates)):
        pairwise = _binomial_upper_tail(wrong_type_minimum, crossover)
        wrong_type_competitors = (
            max(type_count - 1, 0)
            * length
            * int(policy_masks(policy, layers).shape[0])
        )
        tails[format(crossover, ".6f")] = {
            "pairwise_binomial_tail": pairwise,
            "wrong_type_competitor_count": wrong_type_competitors,
            "wrong_type_union_bound": min(
                1.0, wrong_type_competitors * pairwise
            ),
        }
    return {
        "status": "CONSTRUCTION_DIAGNOSTIC_ONLY",
        "metric": "clean_hypothesis_bank_hamming_distance",
        "distance_exactness": exactness,
        "total_bits_per_hypothesis": total_bits,
        "hypothesis_count": hypothesis_count,
        "relative_mask_word_count": int(relative_masks.shape[0]),
        "relative_mask_words": [
            "".join("+" if value == 1 else "-" for value in row)
            for row in relative_masks
        ],
        "relative_word_definition": (
            "planted_mask_word multiplied by decoder_mask_word"
        ),
        "decoder_hypothesis_pair_relative_word_count": int(
            decoder_relative_masks.shape[0]
        ),
        "measured_or_bounded_sidelobe": sidelobe,
        "distance_spectrum_values": spectrum,
        "observed_to_decoder_candidate_d_min": minimum,
        "wrong_type_d_min": wrong_type_minimum,
        "wrong_shift_d_min": wrong_shift_minimum,
        "wrong_mask_only_d_min": wrong_mask_minimum,
        "bsc_wrong_type_pairwise_and_union_bounds": tails,
        "bsc_observed_cell_applicable": planted_bank_compatible,
        "bsc_observed_cell_applicability_reason": (
            "planted mask family is contained in decoder hypothesis bank"
            if planted_bank_compatible
            else "planted mask family contains words outside decoder hypothesis bank"
        ),
        "observed_planted_type_errors": (
            observed_error_count if planted_bank_compatible else None
        ),
        "observed_planted_groups": (
            observed_trial_count if planted_bank_compatible else None
        ),
        "theory_novelty_claim": False,
    }


def _aggregate_cell(
    config: CampaignConfig,
    spec: Mapping[str, Any],
    base_result: Mapping[str, Any],
    *,
    signatures: np.ndarray,
    templates: np.ndarray,
    base: np.ndarray,
    payload_groups: np.ndarray,
    payload_type_bank: np.ndarray,
    reference_ffts: np.ndarray,
    reference_norms: np.ndarray,
    payload_ffts: np.ndarray,
    payload_norms: np.ndarray,
    codebook_manifest: Mapping[str, Any],
    carrier_provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    policy = str(spec["decoder_mask_policy"])
    rows = base_result["rows"][policy]
    select_planted = rows["SELECT"]["planted"]
    select_unrelated = rows["SELECT"]["unrelated"]
    report_planted = rows["REPORT"]["planted"]
    report_unrelated = rows["REPORT"]["unrelated"]
    report_noncyclic = rows["REPORT"]["non_cyclic"]
    threshold = fit_select_thresholds(
        select_planted,
        select_unrelated,
        target_false_unlock_rate=config.target_false_unlock_rate,
        confidence_level=config.confidence_level,
        allow_smoke_point_estimate=(
            config.run_label == "SMOKE_NON_EVIDENTIARY"
        ),
    )
    unlocked_planted, correct_planted = _apply_threshold(
        report_planted, threshold
    )
    unlocked_unrelated, _ = _apply_threshold(report_unrelated, threshold)
    unlocked_noncyclic, _ = _apply_threshold(report_noncyclic, threshold)
    if unlocked_unrelated is None:
        false_events = None
        cp_upper = None
    else:
        false_events = int(sum(unlocked_unrelated))
        cp_upper = clopper_pearson_upper(
            false_events, len(unlocked_unrelated), config.confidence_level
        )

    oppw = _aggregate_control(
        _control_threshold_rows(select_planted, "oppw_form_control"),
        _control_threshold_rows(select_unrelated, "oppw_form_control"),
        _control_threshold_rows(report_planted, "oppw_form_control"),
        _control_threshold_rows(report_unrelated, "oppw_form_control"),
        target_false_unlock_rate=config.target_false_unlock_rate,
        confidence_level=config.confidence_level,
        allow_smoke_point_estimate=(
            config.run_label == "SMOKE_NON_EVIDENTIARY"
        ),
    )
    independent = _aggregate_control(
        _control_threshold_rows(select_planted, "independent_control"),
        _control_threshold_rows(select_unrelated, "independent_control"),
        _control_threshold_rows(report_planted, "independent_control"),
        _control_threshold_rows(report_unrelated, "independent_control"),
        target_false_unlock_rate=config.target_false_unlock_rate,
        confidence_level=config.confidence_level,
        allow_smoke_point_estimate=(
            config.run_label == "SMOKE_NON_EVIDENTIARY"
        ),
    )
    independent["phase_recovery_status"] = (
        "NOT_DEFINED_FOR_INDEPENDENT_LAYER_CONTROL"
    )

    payload_key = format(float(spec["payload_noise_rate"]), ".6f")
    payload_rows = [
        row["payload_by_noise"][payload_key] for row in report_planted
    ]
    payload_latency_paths = sorted(
        payload_rows[0]["payload_latency_ms"]
    )
    payload_working_paths = sorted(
        payload_rows[0]["payload_decoder_working_array_bytes"]
    )
    raw_routed = _mean_bool(
        item["routed_waypoint_recall_at_1"] for item in payload_rows
    )
    payload_only = _mean_bool(
        item["payload_only_recall_at_1"] for item in payload_rows
    )
    direct_known = _mean_bool(
        item["direct_known_type_recall_at_1"] for item in payload_rows
    )
    if unlocked_planted is None:
        thresholded_routed = None
        carrier_only = None
    else:
        thresholded_routed = _mean_bool(
            unlock and item["routed_waypoint_recall_at_1"]
            for unlock, item in zip(unlocked_planted, payload_rows)
        )
        carrier_only = _mean_bool(
            unlock and item["carrier_only_waypoint_recall_at_1"]
            for unlock, item in zip(unlocked_planted, payload_rows)
        )
    gap_denominator = (
        None
        if direct_known is None or payload_only is None
        else direct_known - payload_only
    )
    gap_closure = (
        None
        if (
            thresholded_routed is None
            or payload_only is None
            or gap_denominator is None
            or gap_denominator <= 0.0
        )
        else (thresholded_routed - payload_only) / gap_denominator
    )

    resources = _resource_accounting(
        length=int(spec["length"]),
        layers=int(spec["layers"]),
        type_count=config.type_count,
        waypoints_per_type=config.waypoints_per_type,
        decoder_policy=policy,
        templates=templates,
        base=base,
        signatures=signatures,
        payload_groups=payload_groups,
        payload_type_bank=payload_type_bank,
        reference_ffts=reference_ffts,
        reference_norms=reference_norms,
        payload_ffts=payload_ffts,
        payload_norms=payload_norms,
        rows=report_planted + report_unrelated + report_noncyclic,
    )
    raw_type_accuracy = _mean_bool(
        row["type_correct"] for row in report_planted
    )
    raw_phase_accuracy = _mean_bool(
        row["phase_exact"] for row in report_planted
    )
    observed_errors = sum(not row["type_correct"] for row in report_planted)
    hamming = _hamming_distance_diagnostic(
        length=int(spec["length"]),
        layers=int(spec["layers"]),
        type_count=config.type_count,
        policy=policy,
        planted_mask_family=str(spec["planted_mask_family"]),
        carrier_family=str(spec["carrier_family"]),
        base=base,
        signatures=signatures,
        bit_flip_rates=(float(spec["bit_flip_rate"]),),
        observed_error_count=observed_errors,
        observed_trial_count=len(report_planted),
    )
    true_unlock_recall = (
        _mean_bool(correct_planted) if correct_planted is not None else None
    )
    false_unlock_rate = (
        _mean_bool(unlocked_unrelated)
        if unlocked_unrelated is not None
        else None
    )
    efficiency_actual = None
    mean_latency = resources["route_latency_ms"]["mean"]
    if (
        true_unlock_recall is not None
        and false_unlock_rate is not None
        and mean_latency is not None
        and mean_latency > 0.0
    ):
        efficiency_actual = (
            true_unlock_recall - false_unlock_rate
        ) / (
            resources["deduplicated_total_serialized_method_bytes"]
            * mean_latency
        )
    primary_component = bool(
        base_result["primary_base_group"]
        and policy == "typed16"
        and math.isclose(
            float(spec["payload_noise_rate"]), 0.25, rel_tol=0.0, abs_tol=1e-12
        )
    )
    return {
        "cell_id": _cell_id(spec),
        "config": dict(spec),
        "status": "completed",
        "primary_fur_cell": False,
        "primary_fur_seed_component": primary_component,
        "evidence_role": (
            "primary_seed_component_descriptive"
            if primary_component
            else "descriptive_control"
        ),
        "query_group_budgets": base_result["budgets"],
        "phase_codebook": dict(codebook_manifest),
        "carrier_provenance": dict(carrier_provenance),
        "threshold": threshold,
        "report": {
            "planted_group_count": len(report_planted),
            "unrelated_group_count": len(report_unrelated),
            "non_cyclic_group_count": len(report_noncyclic),
            "raw_type_accuracy": raw_type_accuracy,
            "phase_exact_accuracy": raw_phase_accuracy,
            "phase_circular_error": _summary(
                row["phase_circular_error"] for row in report_planted
            ),
            "mask_recovery_accuracy": _mean_bool(
                row["mask_recovered"] for row in report_planted
            ),
            "true_unlock_recall": true_unlock_recall,
            "false_unlock_rate": false_unlock_rate,
            "false_unlock_events": false_events,
            "false_unlock_cp_upper_97_5": cp_upper,
            "fur_gate_pass": bool(
                threshold.get("status") == "FROZEN"
                and
                cp_upper is not None
                and cp_upper <= config.target_false_unlock_rate
            ),
            "non_cyclic_unlock_rate": (
                _mean_bool(unlocked_noncyclic)
                if unlocked_noncyclic is not None
                else None
            ),
            "score_distribution_planted": _summary(
                row["score"] for row in report_planted
            ),
            "score_distribution_unrelated": _summary(
                row["score"] for row in report_unrelated
            ),
            "margin_distribution_planted": _summary(
                row["margin"] for row in report_planted
            ),
            "margin_distribution_unrelated": _summary(
                row["margin"] for row in report_unrelated
            ),
            "tie_rate_planted": _mean_bool(
                row["tie_count"] > 1 for row in report_planted
            ),
            "waypoint": {
                "raw_routed_recall_at_1": raw_routed,
                "thresholded_routed_recall_at_1": thresholded_routed,
                "routed_recall_at_k": _mean_bool(
                    item["routed_waypoint_recall_at_k"]
                    for item in payload_rows
                ),
                "payload_only_recall_at_1": payload_only,
                "payload_only_recall_at_k": _mean_bool(
                    item["payload_only_recall_at_k"] for item in payload_rows
                ),
                "direct_known_type_recall_at_1": direct_known,
                "direct_known_type_recall_at_k": _mean_bool(
                    item["direct_known_type_recall_at_k"]
                    for item in payload_rows
                ),
                "direct_known_type_requires_side_information": True,
                "carrier_only_recall_at_1": carrier_only,
                "gap_closure": gap_closure,
                "payload_transport": (
                    "separate_type_canonical_coordinates_plus_frozen_global_shift"
                ),
                "payload_latency_ms": {
                    path: _summary(
                        item["payload_latency_ms"][path]
                        for item in payload_rows
                    )
                    for path in payload_latency_paths
                },
                "payload_decoder_working_array_bytes": {
                    path: {
                        "estimate_bytes": _summary(
                            item["payload_decoder_working_array_bytes"][
                                path
                            ]["value"]
                            for item in payload_rows
                        ),
                        "kind": payload_rows[0][
                            "payload_decoder_working_array_bytes"
                        ][path]["kind"],
                        "measured_process_peak": False,
                        "component_breakdown": payload_rows[0][
                            "payload_decoder_working_array_bytes"
                        ][path]["components"],
                    }
                    for path in payload_working_paths
                },
                "payload_only_recall_at_k_caveat": {
                    "canonical_exact_tie_order_dependent": True,
                    "canonical_order": "type-major then waypoint-major",
                    "configured_k": config.recall_k,
                    "type_count": config.type_count,
                    "k_exceeds_type_count": (
                        config.recall_k > config.type_count
                    ),
                    "evidence_of_type_retrieval": False,
                },
                "payload_fft_preprocessing": payload_rows[0][
                    "payload_fft_preprocessing"
                ],
            },
        },
        "controls": {
            "independent_layers": independent,
            "oppw_phase_address_from_dense_estimates": oppw,
            "native_sparse_oppw": {
                "status": "MANDATORY_UNIMPLEMENTED",
                "reason": (
                    "matched sparse observation noise and energy contract "
                    "not implemented"
                ),
                "resource_pareto_claim_eligible": False,
            },
            "equal_channel_use_repeated_bit": {
                "status": "MANDATORY_UNIMPLEMENTED",
                "reason": (
                    "no frozen repeated-bit observation with matched channel "
                    "uses, energy, and decoder budget"
                ),
                "matched_resource_advantage_claim_eligible": False,
            },
        },
        "resources": resources,
        "construction_diagnostics": {
            "hamming_distance_and_bsc_reduction": hamming,
        },
        "descriptive_efficiency_actual_bytes": efficiency_actual,
        "descriptive_efficiency_denominator": (
            "deduplicated_total_serialized_method_bytes * mean_route_latency_ms"
        ),
        "promotion_eligible": False,
    }


def _primary_aggregate(
    config: CampaignConfig,
    components: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    expected_seeds = set(FROZEN_SEEDS)
    observed_seeds = {int(component["seed"]) for component in components}
    if config.run_label != "FULL_METHOD_DEV":
        return {
            "status": "NOT_RUN_SMOKE_NON_EVIDENTIARY",
            "required_seeds": list(FROZEN_SEEDS),
            "one_pooled_threshold": True,
            "promotion_eligible": False,
        }
    if observed_seeds != expected_seeds:
        return {
            "status": "INCOMPLETE_PRIMARY_SEED_SET",
            "required_seeds": list(FROZEN_SEEDS),
            "observed_seeds": sorted(observed_seeds),
            "one_pooled_threshold": True,
            "promotion_eligible": False,
        }
    select_planted = [
        row
        for component in components
        for row in component["rows"]["SELECT"]["planted"]
    ]
    select_unrelated = [
        row
        for component in components
        for row in component["rows"]["SELECT"]["unrelated"]
    ]
    report_planted = [
        row
        for component in components
        for row in component["rows"]["REPORT"]["planted"]
    ]
    report_unrelated = [
        row
        for component in components
        for row in component["rows"]["REPORT"]["unrelated"]
    ]
    frozen_counts = {
        "select_planted": 3 * 512,
        "select_unrelated": 3 * 512,
        "report_planted": 3 * 1024,
        "report_unrelated": 3 * 1024,
    }
    actual_counts = {
        "select_planted": len(select_planted),
        "select_unrelated": len(select_unrelated),
        "report_planted": len(report_planted),
        "report_unrelated": len(report_unrelated),
    }
    if actual_counts != frozen_counts:
        return {
            "status": "INVALID_PRIMARY_GROUP_COUNTS",
            "required_counts": frozen_counts,
            "actual_counts": actual_counts,
            "one_pooled_threshold": True,
            "promotion_eligible": False,
        }
    threshold = fit_select_thresholds(
        select_planted,
        select_unrelated,
        target_false_unlock_rate=config.target_false_unlock_rate,
        confidence_level=config.confidence_level,
    )
    report_unlocked, report_correct = _apply_threshold(
        report_planted, threshold
    )
    unrelated_unlocked, _ = _apply_threshold(report_unrelated, threshold)
    if unrelated_unlocked is None:
        false_events = None
        cp_upper = None
        fur_gate = False
    else:
        false_events = int(sum(unrelated_unlocked))
        cp_upper = clopper_pearson_upper(
            false_events, len(unrelated_unlocked), config.confidence_level
        )
        fur_gate = cp_upper <= config.target_false_unlock_rate
    return {
        "status": (
            "REPORT_EVALUATED"
            if threshold["status"] == "FROZEN"
            else "NO_ELIGIBLE_SELECT_THRESHOLD_FAIL_CLOSED"
        ),
        "required_seeds": list(FROZEN_SEEDS),
        "observed_seeds": sorted(observed_seeds),
        "one_pooled_threshold": True,
        "counts": actual_counts,
        "threshold": threshold,
        "report_true_unlock_recall": (
            _mean_bool(report_correct) if report_correct is not None else None
        ),
        "report_false_unlock_events": false_events,
        "report_false_unlock_rate": (
            _mean_bool(unrelated_unlocked)
            if unrelated_unlocked is not None
            else None
        ),
        "report_false_unlock_cp_upper_97_5": cp_upper,
        "fur_gate_pass": fur_gate,
        "synthetic_method_dev_only": True,
        "native_sparse_oppw_resource_control_closed": False,
        "promotion_eligible": False,
    }


def _construction_checks(config: CampaignConfig) -> Dict[str, Any]:
    legendre_checks = {}
    for length in config.lengths:
        if length % 4 == 3 and is_prime_exact(length):
            legendre_checks[str(length)] = {
                "prime": True,
                "mod_4": length % 4,
                "exact_ideal_autocorrelation": has_ideal_legendre_autocorrelation(
                    length
                ),
            }
        else:
            legendre_checks[str(length)] = {
                "prime": is_prime_exact(length),
                "mod_4": length % 4,
                "exact_ideal_autocorrelation": "STRUCTURALLY_UNAVAILABLE",
            }
    crt = {
        "status": "NOT_APPLICABLE",
        "length": 4691,
    }
    if 4691 in config.lengths:
        moduli = (2, 5, 7, 67)
        round_trips = {
            str(value): (
                crt_reconstruct_integer(
                    crt_encode_integer(value, moduli), moduli
                )
                == value % 4690
            )
            for value in (0, 1, 66, 67, 2345, 4689)
        }
        crt = {
            "status": "CHECKED",
            "length": 4691,
            "factorization": list(factor_integer_exact(4690)),
            "expected_factorization": list(moduli),
            "round_trips": round_trips,
            "all_pass": all(round_trips.values())
            and factor_integer_exact(4690) == moduli,
            "rader_claim": False,
        }
    all_legendre = all(
        value["exact_ideal_autocorrelation"] in (
            True,
            "STRUCTURALLY_UNAVAILABLE",
        )
        for value in legendre_checks.values()
    )
    return {
        "legendre": legendre_checks,
        "crt_4691": crt,
        "mask_codebook_shapes": {
            policy: list(policy_masks(policy, 8).shape)
            for policy in DECODER_MASK_POLICIES
        },
        "all_checked_invariants_pass": bool(
            all_legendre and (
                crt["status"] != "CHECKED" or crt["all_pass"]
            )
        ),
    }


def _campaign_verdicts(
    config: CampaignConfig,
    cells: Sequence[Mapping[str, Any]],
    construction: Mapping[str, Any],
    primary: Mapping[str, Any],
) -> Dict[str, Any]:
    completed = [cell for cell in cells if cell["status"] == "completed"]
    errors = [cell for cell in cells if cell["status"] == "error"]
    smoke = config.run_label == "SMOKE_NON_EVIDENTIARY"
    prw0 = (
        "SUPPORTED_CONSTRUCTION_METHOD_DEV"
        if construction["all_checked_invariants_pass"] and not errors
        else "IMPLEMENTATION_INVALID"
    )
    common_blockers = [
        "synthetic METHOD_DEV cannot establish real retrieval utility",
        (
            "native sparse OPPW baseline with matched noise/energy is "
            "mandatory-unimplemented"
        ),
        "phase address is known OPPW-equivalent mathematics",
    ]
    return {
        "PRW-0": {
            "verdict": prw0,
            "scope": "construction invariants only",
            "promotion_eligible": False,
        },
        "PRW-1": {
            "verdict": (
                "INCONCLUSIVE_SMOKE"
                if smoke
                else (
                    "INCONCLUSIVE_CONTROL_GAP"
                    if primary.get("fur_gate_pass")
                    else "NO_GO_PRIMARY_GATE"
                )
            ),
        "blockers": common_blockers,
            "promotion_eligible": False,
        },
        "PRW-2": {
            "verdict": (
                "INCONCLUSIVE_SMOKE"
                if smoke
                else "INCONCLUSIVE_SYNTHETIC_COLLISION_GROUPS"
            ),
            "direct_known_type_is_extra_side_information": True,
            "promotion_eligible": False,
        },
        "PRW-3": {
            "verdict": (
                "INCONCLUSIVE_SMOKE"
                if smoke
                else "INCONCLUSIVE_PENDING_POOLED_MATCHED_RECALL_COMPARISON"
            ),
            "promotion_eligible": False,
        },
        "PRW-4": {
            "verdict": "INCONCLUSIVE",
            "reason": (
                "no native sparse resource baseline and no repeated hardware "
                "timing campaign; equal-channel-use repeated-bit control is "
                "unimplemented; theoretical packed bytes excluded from cost verdict"
            ),
            "special_status_for_4691": False,
            "promotion_eligible": False,
        },
        "RADER-1": {
            "verdict": "NOT_TESTED_OUT_OF_SCOPE",
            "rader_implementation_present": False,
            "promotion_eligible": False,
        },
        "novelty": {
            "verdict": "NOT_ESTABLISHED",
            "phase_address_prior_art_class": "OPPW_2D_optical_orthogonal_code",
            "claim_of_novel_phase_math": False,
            "completed_cell_count": len(completed),
        },
    }


def run_campaign(
    config: CampaignConfig,
    *,
    write_artifact: bool = True,
) -> Dict[str, Any]:
    """Execute the complete retained grid and optionally write its JSON artifact."""

    config = validate_config(config)
    stream_manifest = _stream_manifest(config)
    if not stream_manifest["select_report_disjoint"]:
        raise CampaignValidationError("BANK/SELECT/REPORT streams are not disjoint")
    construction = _construction_checks(config)
    cell_specs = expected_cell_specs(config)
    cells_by_id: Dict[str, Dict[str, Any]] = {}
    for spec in cell_specs:
        reason = _cell_unavailable_reason(spec)
        cells_by_id[_cell_id(spec)] = {
            "cell_id": _cell_id(spec),
            "config": dict(spec),
            "status": "unavailable" if reason else "pending",
            "unavailable_reason": reason,
            "promotion_eligible": False,
        }

    primary_components = []
    base_groups = product(
        config.seeds,
        config.lengths,
        config.carrier_families,
        config.layers,
    )
    for seed, length, carrier_family, layers in base_groups:
        carrier_reason = _carrier_availability(length, carrier_family)
        if carrier_reason:
            continue
        try:
            signatures, codebook_manifest = _phase_codebook(
                length, layers, config.type_count, seed
            )
            templates, base, carrier_provenance, ring_templates = (
                _build_carrier_templates(
                    length,
                    layers,
                    carrier_family,
                    signatures,
                    seed,
                )
            )
            payload_groups, payload_manifest = _build_payload_groups(
                seed,
                config.waypoints_per_type,
                layers,
                length,
            )
            payload_type_bank = _materialize_payload_collision_bank(
                payload_groups, config.type_count
            )
            carrier_provenance = dict(carrier_provenance)
            carrier_provenance["payload_bank"] = {
                **payload_manifest,
                "type_relative_transport_materialized": False,
                "coordinate_contract": (
                    "separate type-canonical payload coordinates; "
                    "shared global transport only"
                ),
                "deduplicated_collision_bank_shape": list(
                    payload_type_bank.shape
                ),
                "deduplicated_collision_bank_nbytes": int(
                    payload_type_bank.nbytes
                ),
            }
            reference_ffts = np.fft.rfft(templates, axis=2)
            reference_norms = np.linalg.norm(templates, axis=2)
            payload_ffts = np.fft.rfft(payload_type_bank, axis=2)
            payload_norms = np.linalg.norm(payload_type_bank, axis=2)
            for planted_mask_family in config.planted_mask_families:
                if (
                    _mask_availability(
                        layers, planted_mask_family, decoder=False
                    )
                    is not None
                ):
                    continue
                for bit_flip_rate in config.bit_flip_rates:
                    base_result = _run_base_group(
                        config,
                        seed=seed,
                        length=length,
                        carrier_family=carrier_family,
                        layers=layers,
                        planted_mask_family=planted_mask_family,
                        bit_flip_rate=bit_flip_rate,
                        signatures=signatures,
                        templates=templates,
                        ring_templates=ring_templates,
                        payload_type_bank=payload_type_bank,
                        payload_ffts=payload_ffts,
                        payload_norms=payload_norms,
                        reference_ffts=reference_ffts,
                        reference_norms=reference_norms,
                    )
                    for decoder_policy in config.decoder_mask_policies:
                        if (
                            _mask_availability(
                                layers, decoder_policy, decoder=True
                            )
                            is not None
                        ):
                            continue
                        for payload_noise_rate in config.payload_noise_rates:
                            spec = {
                                "seed": int(seed),
                                "length": int(length),
                                "carrier_family": carrier_family,
                                "layers": int(layers),
                                "planted_mask_family": planted_mask_family,
                                "decoder_mask_policy": decoder_policy,
                                "bit_flip_rate": float(bit_flip_rate),
                                "payload_noise_rate": float(payload_noise_rate),
                            }
                            try:
                                cells_by_id[_cell_id(spec)] = _aggregate_cell(
                                    config,
                                    spec,
                                    base_result,
                                    signatures=signatures,
                                    templates=templates,
                                    base=base,
                                    payload_groups=payload_groups,
                                    payload_type_bank=payload_type_bank,
                                    reference_ffts=reference_ffts,
                                    reference_norms=reference_norms,
                                    payload_ffts=payload_ffts,
                                    payload_norms=payload_norms,
                                    codebook_manifest=codebook_manifest,
                                    carrier_provenance=carrier_provenance,
                                )
                            except Exception as exc:  # retain the cell; never omit
                                cells_by_id[_cell_id(spec)] = {
                                    "cell_id": _cell_id(spec),
                                    "config": spec,
                                    "status": "error",
                                    "error": {
                                        "type": type(exc).__name__,
                                        "message": str(exc),
                                    },
                                    "promotion_eligible": False,
                                }
                    if base_result["primary_base_group"]:
                        primary_components.append(
                            {
                                "seed": int(seed),
                                "rows": base_result["rows"]["typed16"],
                            }
                        )
        except Exception as exc:
            # Convert every pending descendant of this bank to an explicit error.
            for cell in cells_by_id.values():
                spec = cell["config"]
                if (
                    cell["status"] == "pending"
                    and spec["seed"] == seed
                    and spec["length"] == length
                    and spec["carrier_family"] == carrier_family
                    and spec["layers"] == layers
                ):
                    cell["status"] = "error"
                    cell["error"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }

    for cell in cells_by_id.values():
        if cell["status"] == "pending":
            cell["status"] = "error"
            cell["error"] = {
                "type": "CampaignExecutionError",
                "message": "grid cell remained pending after campaign traversal",
            }
    cells = [cells_by_id[_cell_id(spec)] for spec in cell_specs]
    primary = _primary_aggregate(config, primary_components)
    verdicts = _campaign_verdicts(config, cells, construction, primary)
    status_counts: Dict[str, int] = {}
    for cell in cells:
        status_counts[cell["status"]] = status_counts.get(cell["status"], 0) + 1
    artifact = {
        "schema_version": "1.0.0",
        "record_type": RECORD_TYPE,
        "protocol_id": PROTOCOL_ID,
        "evidence_mode": EVIDENCE_MODE,
        "run_label": config.run_label,
        "scientific_claim_status": "unconfirmed",
        "novelty_claim_status": "unconfirmed",
        "promotion_eligible": False,
        "production_integration_permitted": False,
        "rader_claim": False,
        "config": asdict(config),
        "rng_streams": stream_manifest,
        "threshold_governance": {
            "selection_split": "SELECT",
            "evaluation_split": "REPORT",
            "report_rows_used_for_threshold_fit": 0,
            "score_grid": {"start": -1.0, "stop": 1.0, "step": 0.05},
            "margin_grid": {"start": 0.0, "stop": 1.0, "step": 0.01},
            "confidence_method": "one-sided exact Clopper-Pearson",
            "confidence_level": config.confidence_level,
            "optional_stopping": False,
        },
        "construction_checks": construction,
        "grid_manifest": {
            "expected_cell_count": len(cell_specs),
            "retained_cell_count": len(cells),
            "status_counts": status_counts,
            "all_cells_retained": len(cells) == len(cell_specs),
            "cell_ids_unique": len({cell["cell_id"] for cell in cells})
            == len(cells),
        },
        "cells": cells,
        "primary_aggregate": primary,
        "verdicts": verdicts,
        "known_limitations": [
            "SMOKE_NON_EVIDENTIARY cannot support a <=1% false-unlock claim"
            if config.run_label == "SMOKE_NON_EVIDENTIARY"
            else "full campaign remains synthetic METHOD_DEV",
            (
                "OPPW phase-address control inherits dense phase estimates; "
                "native sparse matched-resource baseline is unimplemented"
            ),
            (
                "equal-channel-use repeated-bit control is unimplemented; "
                "matched-resource advantage is fail-closed"
            ),
            (
                "payload-only Recall@K is canonical-order dependent under "
                "exact cross-type collision ties; smoke K=5 exceeds T=4 and "
                "is not evidence of type retrieval"
            ),
            "latency fields are wall-clock descriptive and nondeterministic",
            "process peak resident memory was not measured",
        ],
        "timing_fields_nondeterministic": True,
    }
    if write_artifact:
        output = Path(config.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                artifact,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return artifact


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bounded prime-ring waypoint METHOD_DEV campaign"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "run the complete factor grid and frozen 512/1024 primary budgets; "
            "the default is SMOKE_NON_EVIDENTIARY"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSON artifact path",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    config = (
        full_config(args.output)
        if args.full
        else replace(CampaignConfig(), output=args.output)
    )
    artifact = run_campaign(config)
    print(
        _canonical_json(
            {
                "output": config.output,
                "run_label": artifact["run_label"],
                "grid_manifest": artifact["grid_manifest"],
                "verdicts": {
                    key: value["verdict"]
                    for key, value in artifact["verdicts"].items()
                    if isinstance(value, dict) and "verdict" in value
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
