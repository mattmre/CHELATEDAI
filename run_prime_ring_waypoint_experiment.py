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
import hmac
import json
import math
import os
import secrets
import tempfile
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
CORRUPTION_VARIANTS = ("iid", "block_correlated", "burst")
RAW_SANITY_REQUIRED_BIT_FLIP_RATES = (0.0, 0.20, 0.35, 0.45)
RAW_SANITY_REPORT_PLANTED_GROUPS = {
    0.0: 3,
    0.20: 3,
    0.35: 3,
    0.45: 1024,
}
CAMPAIGN_EVIDENCE_CONFIG_FIELDS = (
    "seeds",
    "lengths",
    "carrier_families",
    "layers",
    "planted_mask_families",
    "decoder_mask_policies",
    "bit_flip_rates",
    "payload_noise_rates",
    "type_count",
    "waypoints_per_type",
    "select_planted",
    "select_unrelated",
    "report_planted",
    "report_unrelated",
    "recall_k",
    "target_false_unlock_rate",
    "control_min_true_unlock_recall",
    "confidence_level",
    "report_stream_tag",
    "primary_select_planted",
    "primary_select_unrelated",
    "primary_report_planted",
    "primary_report_unrelated",
    "run_label",
)
RAW_SANITY_CELL_CONFIG_FIELDS = (
    "seed",
    "length",
    "carrier_family",
    "layers",
    "planted_mask_family",
    "decoder_mask_policy",
    "bit_flip_rate",
    "payload_noise_rate",
)
RAW_SANITY_LIVE_SEAL_ESTIMATED_BYTES = 2048
RAW_SANITY_LIVE_AUTHORITY_ESTIMATED_BYTES = 2048
CONTROL_HARD_MAX_ESTIMATED_BYTES = 512 * 1024 * 1024
CONTROL_HARD_MAX_WORK_UNITS = 50_000_000
CAMPAIGN_HARD_MAX_ESTIMATED_BYTES = 512 * 1024 * 1024
CAMPAIGN_DEFAULT_MAX_RETAINED_ROWS = 1_000_000
CAMPAIGN_DEFAULT_MAX_WALL_CLOCK_SECONDS = 24.0 * 60.0 * 60.0
CAMPAIGN_HARD_MAX_RETAINED_ROWS = CAMPAIGN_DEFAULT_MAX_RETAINED_ROWS
CAMPAIGN_HARD_MAX_WALL_CLOCK_SECONDS = (
    CAMPAIGN_DEFAULT_MAX_WALL_CLOCK_SECONDS
)
CAMPAIGN_HARD_MAX_INPUT_BITS = 64
CAMPAIGN_HARD_MAX_FACTOR_VALUES = 64
CAMPAIGN_HARD_MAX_RING_LENGTH = 4691
CAMPAIGN_HARD_MAX_FACTOR_STRING_CHARS = 128
CAMPAIGN_HARD_MAX_FACTOR_STRING_UTF8_BYTES = 512
CAMPAIGN_HARD_MAX_REPORT_STREAM_TAG_CHARS = 256
CAMPAIGN_HARD_MAX_REPORT_STREAM_TAG_UTF8_BYTES = 1024
CAMPAIGN_HARD_MAX_OUTPUT_PATH_CHARS = 4096
CAMPAIGN_HARD_MAX_OUTPUT_PATH_UTF8_BYTES = 16 * 1024
CAMPAIGN_HARD_MAX_RUN_LABEL_CHARS = 128
CAMPAIGN_HARD_MAX_RUN_LABEL_UTF8_BYTES = 512
CAMPAIGN_HARD_MAX_JSON_STRING_CHARS = 1024 * 1024
CAMPAIGN_HARD_MAX_JSON_NODES = 2_000_000
CAMPAIGN_CONFIG_STRING_ARTIFACT_COPIES = 8
CAMPAIGN_ESTIMATED_CELL_ARTIFACT_BYTES = 64 * 1024
CAMPAIGN_ESTIMATED_RAW_PROVENANCE_ROW_BYTES = 512
CAMPAIGN_ESTIMATED_LIVE_ROW_BYTES = 8 * 1024
CAMPAIGN_ESTIMATED_ARTIFACT_BASE_BYTES = 1024 * 1024
DEFAULT_CONTROL_MIN_TRUE_UNLOCK_RECALL = 0.50
DEFAULT_OUTPUT = (
    Path("artifacts")
    / "method-dev"
    / "prime-ring"
    / "prime-ring-waypoint-v1.json"
)


class CampaignValidationError(ValueError):
    """Raised before execution when a campaign contract is malformed."""


class CampaignExecutionLimitError(RuntimeError):
    """Raised when a preflight, deadline, or retained-row limit is crossed."""


@dataclass(frozen=True)
class CampaignExecutionBudget:
    """Hostile-safe hard limits for one campaign process."""

    max_wall_clock_seconds: float = (
        CAMPAIGN_DEFAULT_MAX_WALL_CLOCK_SECONDS
    )
    max_retained_rows: int = CAMPAIGN_DEFAULT_MAX_RETAINED_ROWS
    max_estimated_bytes: int = CAMPAIGN_HARD_MAX_ESTIMATED_BYTES


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
    control_min_true_unlock_recall: float = (
        DEFAULT_CONTROL_MIN_TRUE_UNLOCK_RECALL
    )
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
    result = (
        int.__int__(value)
        if isinstance(value, int)
        else int(value)
    )
    if result.bit_length() > CAMPAIGN_HARD_MAX_INPUT_BITS:
        raise CampaignValidationError(
            f"{name} exceeds the {CAMPAIGN_HARD_MAX_INPUT_BITS}-bit input ceiling"
        )
    if result < minimum:
        raise CampaignValidationError(f"{name} must be >= {minimum}")
    return result


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise CampaignValidationError(f"{name} must be a finite number")
    if isinstance(value, (int, np.integer)):
        result = float(_plain_int(value, name))
    elif isinstance(value, float):
        result = float.__float__(value)
    else:
        result = float(value)
    if not math.isfinite(result):
        raise CampaignValidationError(f"{name} must be a finite number")
    return result


def _plain_string(
    value: object,
    name: str,
    *,
    max_length: int = CAMPAIGN_HARD_MAX_FACTOR_STRING_CHARS,
    max_utf8_bytes: int = CAMPAIGN_HARD_MAX_FACTOR_STRING_UTF8_BYTES,
) -> str:
    if not isinstance(value, str):
        raise CampaignValidationError(f"{name} must be a string")
    result = str.__str__(value)
    if type(result) is not str or not result:
        raise CampaignValidationError(f"{name} must be a non-empty string")
    if len(result) > max_length:
        raise CampaignValidationError(
            f"{name} exceeds the hard {max_length}-character ceiling"
        )
    if len(result.encode("utf-8")) > max_utf8_bytes:
        raise CampaignValidationError(
            f"{name} exceeds the hard {max_utf8_bytes}-byte UTF-8 ceiling"
        )
    return result


def _bounded_raw_tuple(values: object, name: str) -> Tuple[Any, ...]:
    if type(values) is not tuple:
        raise CampaignValidationError(f"{name} must be a non-empty tuple")
    if not values:
        raise CampaignValidationError(f"{name} must be a non-empty tuple")
    if len(values) > CAMPAIGN_HARD_MAX_FACTOR_VALUES:
        raise CampaignValidationError(
            f"{name} exceeds the hard {CAMPAIGN_HARD_MAX_FACTOR_VALUES}-value ceiling"
        )
    return values


def _validated_tuple(
    values: object,
    name: str,
    *,
    allowed: Optional[Sequence[str]] = None,
    minimum: Optional[int] = None,
) -> Tuple[Any, ...]:
    raw = _bounded_raw_tuple(values, name)
    if minimum is not None:
        result = tuple(
            _plain_int(value, f"{name}[{index}]", minimum)
            for index, value in enumerate(raw)
        )
    elif allowed is not None and all(
        isinstance(value, str) for value in allowed
    ):
        result = tuple(
            _plain_string(value, f"{name}[{index}]")
            for index, value in enumerate(raw)
        )
    elif allowed is not None and all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in allowed
    ):
        result = tuple(
            _plain_int(value, f"{name}[{index}]")
            for index, value in enumerate(raw)
        )
    else:
        raise CampaignValidationError(
            f"{name} has no canonical tuple element contract"
        )
    if len(set(result)) != len(result):
        raise CampaignValidationError(f"{name} must not contain duplicates")
    if allowed is not None:
        unknown = [value for value in result if value not in allowed]
        if unknown:
            raise CampaignValidationError(
                f"{name} contains unsupported values: {unknown}"
            )
    return result


def _validated_float_tuple(
    values: object,
    name: str,
    *,
    upper: float,
) -> Tuple[float, ...]:
    raw = _bounded_raw_tuple(values, name)
    result = tuple(
        _finite_float(value, f"{name}[{index}]")
        for index, value in enumerate(raw)
    )
    if len(set(result)) != len(result):
        raise CampaignValidationError(f"{name} must not contain duplicates")
    for index, number in enumerate(result):
        if number < 0.0 or number >= upper:
            raise CampaignValidationError(
                f"{name}[{index}] must be in [0, {upper})"
            )
    return result


def validate_config(config: CampaignConfig) -> CampaignConfig:
    """Validate every factor up front; malformed campaigns fail closed."""

    if type(config) is not CampaignConfig:
        raise CampaignValidationError("config must be CampaignConfig")
    seeds = _validated_tuple(config.seeds, "seeds", minimum=0)
    lengths = _validated_tuple(config.lengths, "lengths", minimum=7)
    if any(length > CAMPAIGN_HARD_MAX_RING_LENGTH for length in lengths):
        raise CampaignValidationError(
            "lengths exceed the hard campaign ring-length ceiling of "
            f"{CAMPAIGN_HARD_MAX_RING_LENGTH}"
        )
    carrier_families = _validated_tuple(
        config.carrier_families,
        "carrier_families",
        allowed=CARRIER_FAMILIES,
    )
    layers = _validated_tuple(config.layers, "layers", allowed=(1, 8))
    planted_mask_families = _validated_tuple(
        config.planted_mask_families,
        "planted_mask_families",
        allowed=PLANTED_MASK_FAMILIES,
    )
    decoder_mask_policies = _validated_tuple(
        config.decoder_mask_policies,
        "decoder_mask_policies",
        allowed=DECODER_MASK_POLICIES,
    )
    bit_flip_rates = _validated_float_tuple(
        config.bit_flip_rates,
        "bit_flip_rates",
        upper=0.5,
    )
    payload_noise_rates = _validated_float_tuple(
        config.payload_noise_rates,
        "payload_noise_rates",
        upper=math.inf,
    )
    integer_values = {}
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
        integer_values[name] = _plain_int(value, name, minimum)
    if (
        integer_values["type_count"] > 64
        or integer_values["waypoints_per_type"] > 64
    ):
        raise CampaignValidationError("type and waypoint counts are bounded at 64")
    target = _finite_float(
        config.target_false_unlock_rate, "target_false_unlock_rate"
    )
    if not 0.0 < target < 1.0:
        raise CampaignValidationError("target_false_unlock_rate must be in (0, 1)")
    control_recall = _finite_float(
        config.control_min_true_unlock_recall,
        "control_min_true_unlock_recall",
    )
    if not 0.0 < control_recall <= 1.0:
        raise CampaignValidationError(
            "control_min_true_unlock_recall must be in (0, 1]"
        )
    confidence = _finite_float(config.confidence_level, "confidence_level")
    if not 0.5 < confidence < 1.0:
        raise CampaignValidationError("confidence_level must be in (0.5, 1)")
    report_stream_tag = _plain_string(
        config.report_stream_tag,
        "report_stream_tag",
        max_length=CAMPAIGN_HARD_MAX_REPORT_STREAM_TAG_CHARS,
        max_utf8_bytes=CAMPAIGN_HARD_MAX_REPORT_STREAM_TAG_UTF8_BYTES,
    )
    output = _plain_string(
        config.output,
        "output",
        max_length=CAMPAIGN_HARD_MAX_OUTPUT_PATH_CHARS,
        max_utf8_bytes=CAMPAIGN_HARD_MAX_OUTPUT_PATH_UTF8_BYTES,
    )
    run_label = _plain_string(
        config.run_label,
        "run_label",
        max_length=CAMPAIGN_HARD_MAX_RUN_LABEL_CHARS,
        max_utf8_bytes=CAMPAIGN_HARD_MAX_RUN_LABEL_UTF8_BYTES,
    )
    return CampaignConfig(
        seeds=seeds,
        lengths=lengths,
        carrier_families=carrier_families,
        layers=layers,
        planted_mask_families=planted_mask_families,
        decoder_mask_policies=decoder_mask_policies,
        bit_flip_rates=bit_flip_rates,
        payload_noise_rates=payload_noise_rates,
        type_count=integer_values["type_count"],
        waypoints_per_type=integer_values["waypoints_per_type"],
        select_planted=integer_values["select_planted"],
        select_unrelated=integer_values["select_unrelated"],
        report_planted=integer_values["report_planted"],
        report_unrelated=integer_values["report_unrelated"],
        recall_k=integer_values["recall_k"],
        target_false_unlock_rate=target,
        control_min_true_unlock_recall=control_recall,
        confidence_level=confidence,
        report_stream_tag=report_stream_tag,
        output=output,
        primary_select_planted=integer_values["primary_select_planted"],
        primary_select_unrelated=integer_values["primary_select_unrelated"],
        primary_report_planted=integer_values["primary_report_planted"],
        primary_report_unrelated=integer_values["primary_report_unrelated"],
        run_label=run_label,
    )


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


def _campaign_contract_manifest(config: CampaignConfig) -> Dict[str, Any]:
    """Bind evidentiary settings and RNG domains independently of output path."""

    config_values = asdict(config)
    evidence_config = {
        field: config_values[field]
        for field in CAMPAIGN_EVIDENCE_CONFIG_FIELDS
    }
    stream_manifest = _stream_manifest(config)
    payload = {
        "protocol_id": PROTOCOL_ID,
        "record_type": RECORD_TYPE,
        "evidence_mode": EVIDENCE_MODE,
        "evidence_config": evidence_config,
        "stream_manifest": stream_manifest,
    }
    return {
        "digest_algorithm": "sha256_canonical_json_v1",
        "digest": _stable_digest(payload),
        **payload,
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


def _control_resource_accounting(
    components: Mapping[str, int],
    *,
    estimated_work_units: int,
) -> Dict[str, Any]:
    normalized = {
        str(name): _plain_int(value, f"resource component {name}", 0)
        for name, value in components.items()
    }
    estimated_bytes = int(sum(normalized.values()))
    work_units = _plain_int(
        estimated_work_units,
        "estimated_work_units",
        0,
    )
    if estimated_bytes > CONTROL_HARD_MAX_ESTIMATED_BYTES:
        raise CampaignValidationError(
            "control preflight exceeds the hard 512 MiB estimate ceiling: "
            f"{estimated_bytes} > {CONTROL_HARD_MAX_ESTIMATED_BYTES}"
        )
    if work_units > CONTROL_HARD_MAX_WORK_UNITS:
        raise CampaignValidationError(
            "control preflight exceeds the hard work ceiling: "
            f"{work_units} > {CONTROL_HARD_MAX_WORK_UNITS}"
        )
    return {
        "kind": "conservative_simultaneous_live_array_estimate",
        "standalone_estimated_peak_bytes": estimated_bytes,
        "component_breakdown": normalized,
        "estimated_work_units": work_units,
        "hard_max_estimated_bytes": CONTROL_HARD_MAX_ESTIMATED_BYTES,
        "hard_max_work_units": CONTROL_HARD_MAX_WORK_UNITS,
        "measured_process_peak": False,
        "harness_co_resident_dense_context_included": False,
    }


def _with_harness_dense_context(
    resource: Mapping[str, Any],
    *,
    dense_context_bytes: int,
) -> Dict[str, Any]:
    context = _plain_int(
        dense_context_bytes,
        "dense_context_bytes",
        0,
    )
    combined = int(resource["standalone_estimated_peak_bytes"]) + context
    if combined > CONTROL_HARD_MAX_ESTIMATED_BYTES:
        raise CampaignValidationError(
            "control plus co-resident dense harness estimate exceeds the hard "
            f"512 MiB ceiling: {combined} > "
            f"{CONTROL_HARD_MAX_ESTIMATED_BYTES}"
        )
    return {
        **resource,
        "harness_co_resident_dense_context_included": True,
        "harness_co_resident_dense_context_bytes": context,
        "harness_estimated_peak_bytes": combined,
    }


def _unique_numpy_nbytes(*values: object) -> int:
    seen_objects = set()
    seen_arrays = set()

    def visit(value: object) -> int:
        object_id = id(value)
        if isinstance(value, np.ndarray):
            if object_id in seen_arrays:
                return 0
            seen_arrays.add(object_id)
            return int(value.nbytes)
        if object_id in seen_objects:
            return 0
        seen_objects.add(object_id)
        if isinstance(value, Mapping):
            return sum(visit(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(visit(item) for item in value)
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            return visit(attributes)
        return 0

    return int(sum(visit(value) for value in values))


def _native_observation_resource(
    signatures: np.ndarray,
) -> Dict[str, Any]:
    layers = int(signatures.shape[1])
    return _control_resource_accounting(
        {
            "phase_signature_bank_int64": int(signatures.nbytes),
            "positions_int64": layers * 8,
            "substitution_mask_bool": layers,
            "uniform_draw_float64": layers * 8,
            "offsets_and_fancy_index_temporaries": layers * 8 * 4,
        },
        estimated_work_units=layers,
    )


def _native_decoder_resource(
    signatures: np.ndarray,
) -> Dict[str, Any]:
    layers = int(signatures.shape[1])
    type_count = int(signatures.shape[0])
    return _control_resource_accounting(
        {
            "phase_signature_bank_int64": int(signatures.nbytes),
            "observed_positions_int64": layers * 8,
            "type_scores_float64": type_count * 8,
            "type_shifts_int64": type_count * 8,
            "per_type_unique_and_sort_workspace": layers * 8 * 8,
            "per_type_boolean_workspace": layers,
        },
        estimated_work_units=int(
            type_count
            * layers
            * max(1, math.ceil(math.log2(max(layers, 2))))
        ),
    )


def _repeated_bit_resource(
    codebook: np.ndarray,
    length: int,
    layers: int,
) -> Dict[str, Any]:
    total = layers * length
    return _control_resource_accounting(
        {
            "repeated_bit_codebook_int8": int(codebook.nbytes),
            "repeated_observation_float64": total * 8,
            "repeat_construction_int8": total,
            "iid_uniform_workspace_float64": total * 8,
            "flip_mask_bool": total,
            "fancy_index_selected_values_float64": total * 8,
            "float64_codebook_matmul_copy": int(codebook.size * 8),
            "layer_means_float64": layers * 8,
            "type_scores_float64": int(codebook.shape[0] * 8),
        },
        estimated_work_units=int(
            2 * total + codebook.shape[0] * layers
        ),
    )


def _native_oppw_contract(length: int, layers: int) -> Dict[str, Any]:
    return {
        "representation": "one_pulse_per_layer_positions",
        "physical_channel_uses": int(layers * length),
        "pulse_amplitude": float(math.sqrt(length)),
        "transmitted_energy": int(layers * length),
        "noise_channel": (
            "independent layer-symbol substitution with declared probability q; "
            "a substituted pulse moves uniformly to one of p-1 other positions"
        ),
        "actual_query_storage": (
            f"{layers} canonical residue positions represented as int64"
        ),
        "dense_phase_estimates_consumed": False,
        "native_sparse_resource_baseline": True,
    }


def _observe_native_oppw(
    config: CampaignConfig,
    seed: int,
    split: str,
    truth: Mapping[str, Any],
    signatures: np.ndarray,
    input_class: str,
    substitution_rate: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate a native sparse phase observation without dense carriers."""

    # The maximum signature need not span the ring, so obtain p from the
    # declared trial contract rather than inferring it from observed residues.
    length = int(truth["ring_length"])
    layers = int(signatures.shape[1])
    resource_accounting = _native_observation_resource(signatures)
    stream_tag = "SELECT" if split == "SELECT" else config.report_stream_tag
    rng = _stable_rng(
        PROTOCOL_ID,
        stream_tag,
        seed,
        truth["group_id"],
        "native-oppw-symbol-channel",
        layers,
        length,
    )
    if input_class == "unrelated":
        positions = rng.integers(
            0, length, size=layers, dtype=np.int64
        )
        substitutions = np.zeros(layers, dtype=bool)
    else:
        positions = np.mod(
            signatures[int(truth["true_type"])]
            + int(truth["global_shift"]),
            length,
        ).astype(np.int64)
        substitutions = rng.random(layers) < float(substitution_rate)
        if np.any(substitutions):
            offsets = rng.integers(
                1, length, size=layers, dtype=np.int64
            )
            positions = positions.copy()
            positions[substitutions] = (
                positions[substitutions] + offsets[substitutions]
            ) % length
    positions = np.ascontiguousarray(positions, dtype=np.int64)
    return positions, {
        **_native_oppw_contract(length, layers),
        "declared_symbol_substitution_rate": float(substitution_rate),
        "observed_symbol_substitution_rate": float(
            np.mean(substitutions)
        ),
        "substitution_count": int(np.count_nonzero(substitutions)),
        "query_nbytes": int(positions.nbytes),
        "resource_accounting": resource_accounting,
    }


def _decode_native_oppw(
    positions: np.ndarray,
    signatures: np.ndarray,
    length: int,
) -> Dict[str, Any]:
    """Maximum pulse-overlap decoder over type and one shared shift."""

    observed = np.asarray(positions, dtype=np.int64)
    if observed.ndim != 1 or observed.size != signatures.shape[1]:
        raise CampaignValidationError("native OPPW observation shape mismatch")
    resource_accounting = _native_decoder_resource(signatures)
    type_scores = np.empty(signatures.shape[0], dtype=np.float64)
    type_shifts = np.empty(signatures.shape[0], dtype=np.int64)
    for type_index, signature in enumerate(signatures):
        deltas = np.mod(observed - signature, length)
        values, counts = np.unique(deltas, return_counts=True)
        maximum = int(np.max(counts))
        candidate_shifts = values[counts == maximum]
        type_scores[type_index] = maximum / signatures.shape[1]
        type_shifts[type_index] = int(np.min(candidate_shifts))
    winner, score, margin, tie_count = _top_two_semantics(type_scores)
    return {
        "predicted_type": winner,
        "score": score,
        "margin": margin,
        "tie_count": tie_count,
        "shared_shift": int(type_shifts[winner]),
        "candidate_type_count": int(signatures.shape[0]),
        "working_array_bytes": resource_accounting[
            "standalone_estimated_peak_bytes"
        ],
        "resource_accounting": resource_accounting,
        "baseline_family": "native_sparse_OPPW",
        "native_sparse_resource_baseline": True,
        "dense_phase_estimates_consumed": False,
    }


def _repeated_bit_codebook(
    type_count: int, layers: int
) -> Optional[np.ndarray]:
    if layers == 1:
        if type_count > 2:
            return None
        return np.asarray([[-1], [1]][:type_count], dtype=np.int8)
    if layers == 8 and type_count <= 16:
        return np.asarray(
            policy_masks("typed16", 8)[:type_count], dtype=np.int8
        )
    return None


def _equal_channel_repeated_bit_contract(
    length: int, layers: int
) -> Dict[str, Any]:
    return {
        "representation": "one_type_code_bit_repeated_p_times_per_layer",
        "physical_channel_uses": int(layers * length),
        "chip_amplitude": 1.0,
        "transmitted_energy": int(layers * length),
        "noise_channel": (
            "independent Bernoulli bipolar chip flips with probability q"
        ),
        "global_phase_recovery": "NOT_DEFINED",
        "equal_channel_uses_to_dense_carrier": True,
        "equal_transmitted_energy_to_dense_carrier": True,
    }


def _observe_and_decode_repeated_bit(
    config: CampaignConfig,
    seed: int,
    split: str,
    truth: Mapping[str, Any],
    input_class: str,
    length: int,
    layers: int,
    flip_rate: float,
) -> Dict[str, Any]:
    codebook = _repeated_bit_codebook(config.type_count, layers)
    if codebook is None:
        return {
            "status": "STRUCTURALLY_UNAVAILABLE",
            "reason": (
                f"{config.type_count} types cannot be represented by the "
                f"declared {layers}-layer repeated-bit codebook"
            ),
            "contract": _equal_channel_repeated_bit_contract(length, layers),
        }
    resource_accounting = _repeated_bit_resource(
        codebook,
        length,
        layers,
    )
    stream_tag = "SELECT" if split == "SELECT" else config.report_stream_tag
    rng = _stable_rng(
        PROTOCOL_ID,
        stream_tag,
        seed,
        truth["group_id"],
        "equal-channel-repeated-bit",
        layers,
        length,
    )
    if input_class == "unrelated":
        observation = rng.choice(
            np.array([-1.0, 1.0]), size=(layers, length)
        )
        flip_count = 0
    else:
        observation = np.repeat(
            codebook[int(truth["true_type"]), :, np.newaxis],
            length,
            axis=1,
        ).astype(np.float64)
        flips = _draw_flip_mask(
            observation.shape, flip_rate, "iid", rng
        )
        observation[flips] *= -1.0
        flip_count = int(np.count_nonzero(flips))
    layer_means = np.mean(observation, axis=1)
    type_scores = (
        np.matmul(codebook.astype(np.float64), layer_means) / layers
    )
    winner, score, margin, tie_count = _top_two_semantics(type_scores)
    return {
        "status": "COMPLETED",
        "predicted_type": winner,
        "score": score,
        "margin": margin,
        "tie_count": tie_count,
        "phase_recovery": None,
        "phase_recovery_status": "NOT_DEFINED_FOR_REPEATED_BIT_CONTROL",
        "working_array_bytes": resource_accounting[
            "standalone_estimated_peak_bytes"
        ],
        "resource_accounting": resource_accounting,
        "observed_flip_count": flip_count,
        "observed_flip_rate": float(
            flip_count / (layers * length)
        ),
        "contract": _equal_channel_repeated_bit_contract(length, layers),
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
        "ring_length": int(length),
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
    noise_variant: str = "iid",
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
    if noise_variant not in CORRUPTION_VARIANTS:
        raise CampaignValidationError(
            f"noise_variant must be one of {CORRUPTION_VARIANTS}"
        )
    corruption_rng = _stable_rng(
        PROTOCOL_ID,
        stream_tag,
        seed,
        truth["group_id"],
        "bit-flips",
        source.shape[1],
        source.shape[0],
        noise_variant,
    )
    flips = _draw_flip_mask(
        query.shape,
        float(bit_flip_rate),
        noise_variant,
        corruption_rng,
    )
    query[flips] *= -1.0
    return np.ascontiguousarray(query), mask, {
        "representation": "raw_float64_ndarray",
        "derived_from_canonical_template": True,
        "observation_factory": "observe_template",
        "corruption": {
            "iid": "independent_bernoulli_bit_flips",
            "block_correlated": "constant_weight_block_correlated_bit_flips",
            "burst": "constant_weight_per_layer_cyclic_burst_flips",
        }[noise_variant],
        "noise_variant": noise_variant,
        "correlation_geometry": {
            "iid": "rate-independent random coordinate order",
            "block_correlated": (
                "rate-independent randomized contiguous blocks over the "
                "layer-major flattened carrier"
            ),
            "burst": (
                "one rate-independent cyclic contiguous burst origin per layer"
            ),
        }[noise_variant],
        "flip_budget_contract": {
            "iid": (
                "independent Bernoulli draws per coordinate; realized count "
                "is random and severities share nested uniforms"
            ),
            "block_correlated": (
                "exact round(declared_rate * layers * ring_length) coordinates"
            ),
            "burst": (
                "exact round(declared_rate * layers * ring_length) coordinates"
            ),
        }[noise_variant],
        "bsc_binomial_theory_applicable": noise_variant == "iid",
        "declared_global_flip_probability_or_fraction": float(
            bit_flip_rate
        ),
        "observed_flip_rate": float(np.mean(flips)),
        "bit_flip_count": int(np.count_nonzero(flips)),
        "query_nbytes": int(query.nbytes),
    }


def _draw_flip_mask(
    shape: Tuple[int, int],
    rate: float,
    variant: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return nested deterministic draws under the declared noise law.

    ``iid`` uses independent Bernoulli uniforms, preserving the BSC contract.
    Structured variants use an exact global flip budget.  Every variant uses a
    rate-independent base draw, so increasing severity nests the earlier set.
    """

    layers, length = shape
    total = layers * length
    if variant == "iid":
        return np.asarray(rng.random(shape) < float(rate), dtype=bool)
    target = int(round(float(rate) * total))
    target = max(0, min(target, total))
    mask = np.zeros(shape, dtype=bool)
    if target == 0:
        return mask
    if variant == "block_correlated":
        block_size = max(2, int(round(math.sqrt(length))))
        block_starts = np.arange(0, total, block_size, dtype=np.int64)
        ordered_blocks = block_starts[rng.permutation(block_starts.size)]
        flat = mask.reshape(-1)
        remaining = target
        for start in ordered_blocks:
            if remaining <= 0:
                break
            stop = min(int(start) + block_size, total)
            take = min(stop - int(start), remaining)
            flat[int(start) : int(start) + take] = True
            remaining -= take
        return mask
    if variant == "burst":
        base_quota, extra = divmod(target, layers)
        starts = rng.integers(0, length, size=layers, dtype=np.int64)
        # A rate-independent layer order receives the at-most-one extra chip.
        extra_layers = set(
            int(value) for value in rng.permutation(layers)[:extra]
        )
        for layer in range(layers):
            quota = min(
                length,
                base_quota + (1 if layer in extra_layers else 0),
            )
            if quota:
                indices = (
                    int(starts[layer]) + np.arange(quota, dtype=np.int64)
                ) % length
                mask[layer, indices] = True
        if int(np.count_nonzero(mask)) != target:
            raise CampaignValidationError("burst flip budget construction failed")
        return mask
    raise CampaignValidationError(
        f"noise_variant must be one of {CORRUPTION_VARIANTS}"
    )


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
                "block_correlated": config.select_planted,
                "burst": config.select_planted,
            },
            "REPORT": {
                "planted": config.primary_report_planted,
                "unrelated": config.primary_report_unrelated,
                "non_cyclic": config.report_planted,
                "block_correlated": config.report_planted,
                "burst": config.report_planted,
            },
        }
    return {
        "SELECT": {
            "planted": config.select_planted,
            "unrelated": config.select_unrelated,
            "non_cyclic": config.select_planted,
            "block_correlated": config.select_planted,
            "burst": config.select_planted,
        },
        "REPORT": {
            "planted": config.report_planted,
            "unrelated": config.report_unrelated,
            "non_cyclic": config.report_planted,
            "block_correlated": config.report_planted,
            "burst": config.report_planted,
        },
    }


def _canonical_execution_budget(
    budget: Optional[CampaignExecutionBudget],
) -> CampaignExecutionBudget:
    if budget is None:
        return CampaignExecutionBudget()
    if type(budget) is not CampaignExecutionBudget:
        raise CampaignValidationError(
            "execution_budget must be CampaignExecutionBudget"
        )
    wall_clock = budget.max_wall_clock_seconds
    if type(wall_clock) is int:
        if wall_clock.bit_length() > CAMPAIGN_HARD_MAX_INPUT_BITS:
            raise CampaignValidationError(
                "max_wall_clock_seconds exceeds the hard input-bit ceiling"
            )
        canonical_wall_clock = float(wall_clock)
    elif type(wall_clock) is float:
        canonical_wall_clock = float.__float__(wall_clock)
    else:
        raise CampaignValidationError(
            "max_wall_clock_seconds must be a positive finite number"
        )
    if (
        not math.isfinite(canonical_wall_clock)
        or canonical_wall_clock <= 0.0
        or canonical_wall_clock > CAMPAIGN_HARD_MAX_WALL_CLOCK_SECONDS
    ):
        raise CampaignValidationError(
            "max_wall_clock_seconds must be positive, finite, and no greater "
            "than the hard campaign ceiling"
        )
    if (
        type(budget.max_retained_rows) is not int
        or budget.max_retained_rows.bit_length()
        > CAMPAIGN_HARD_MAX_INPUT_BITS
        or budget.max_retained_rows <= 0
        or budget.max_retained_rows > CAMPAIGN_HARD_MAX_RETAINED_ROWS
    ):
        raise CampaignValidationError(
            "max_retained_rows must be a positive integer no greater than "
            "the hard campaign ceiling"
        )
    if (
        type(budget.max_estimated_bytes) is not int
        or budget.max_estimated_bytes.bit_length()
        > CAMPAIGN_HARD_MAX_INPUT_BITS
        or budget.max_estimated_bytes <= 0
        or budget.max_estimated_bytes
        > CAMPAIGN_HARD_MAX_ESTIMATED_BYTES
    ):
        raise CampaignValidationError(
            "max_estimated_bytes must be a positive integer no greater "
            "than the hard 512 MiB modeled ceiling"
        )
    return CampaignExecutionBudget(
        max_wall_clock_seconds=canonical_wall_clock,
        max_retained_rows=budget.max_retained_rows,
        max_estimated_bytes=budget.max_estimated_bytes,
    )


def _modeled_base_group_fixed_array_bytes(
    config: CampaignConfig,
    *,
    length: int,
    layers: int,
) -> Dict[str, int]:
    """Conservatively model fixed NumPy arrays without allocating them."""

    half_spectrum = length // 2 + 1
    signatures = config.type_count * layers * 8
    templates = config.type_count * layers * length * 8
    base = length * 8
    ring_template_copies = 2 * templates
    payload_groups = config.waypoints_per_type * layers * length * 8
    payload_bank = (
        config.type_count
        * config.waypoints_per_type
        * layers
        * length
        * 8
    )
    reference_ffts = (
        config.type_count * layers * half_spectrum * 16
    )
    reference_norms = config.type_count * layers * 8
    payload_ffts = (
        config.type_count
        * config.waypoints_per_type
        * layers
        * half_spectrum
        * 16
    )
    payload_norms = (
        config.type_count
        * config.waypoints_per_type
        * layers
        * 8
    )
    fft_and_decoder_headroom = 2 * max(
        templates,
        payload_bank,
        payload_ffts,
    )
    components = {
        "signatures_int64": signatures,
        "templates_float64": templates,
        "base_float64": base,
        "ring_template_copy_allowance": ring_template_copies,
        "payload_groups_float64": payload_groups,
        "payload_collision_bank_float64": payload_bank,
        "reference_ffts_complex128": reference_ffts,
        "reference_norms_float64": reference_norms,
        "payload_ffts_complex128": payload_ffts,
        "payload_norms_float64": payload_norms,
        "fft_and_decoder_headroom": fft_and_decoder_headroom,
    }
    return {**components, "total": sum(components.values())}


def _config_string_utf8_bytes(config: CampaignConfig) -> int:
    values = (
        *config.carrier_families,
        *config.planted_mask_families,
        *config.decoder_mask_policies,
        config.report_stream_tag,
        config.output,
        config.run_label,
    )
    return sum(len(value.encode("utf-8")) for value in values)


def _campaign_execution_preflight(
    config: CampaignConfig,
    execution_budget: Optional[CampaignExecutionBudget] = None,
) -> Dict[str, Any]:
    """Return an allocation-free campaign model before any construction."""

    config = validate_config(config)
    budget = _canonical_execution_budget(execution_budget)
    config_string_utf8_bytes = _config_string_utf8_bytes(config)
    config_string_artifact_bytes = (
        config_string_utf8_bytes
        * CAMPAIGN_CONFIG_STRING_ARTIFACT_COPIES
    )
    expected_cell_count = math.prod(
        (
            len(config.seeds),
            len(config.lengths),
            len(config.carrier_families),
            len(config.layers),
            len(config.planted_mask_families),
            len(config.decoder_mask_policies),
            len(config.bit_flip_rates),
            len(config.payload_noise_rates),
        )
    )
    expected_base_group_count = math.prod(
        (
            len(config.seeds),
            len(config.lengths),
            len(config.carrier_families),
            len(config.layers),
        )
    )
    refusal_reasons = []
    if expected_cell_count > 1_000_000:
        refusal_reasons.append(
            "expected cell count exceeds the hostile-safe preflight "
            "enumeration ceiling of 1,000,000"
        )
        return {
            "status": "REFUSED",
            "allowed": False,
            "refusal_reasons": refusal_reasons,
            "expected_cell_count": expected_cell_count,
            "available_cell_count": None,
            "expected_base_group_count": expected_base_group_count,
            "available_base_group_count": None,
            "modeled_trial_count": None,
            "modeled_retained_row_count": None,
            "modeled_decoder_retained_row_count": None,
            "modeled_serialized_raw_provenance_row_count": None,
            "modeled_peak_live_retained_rows": None,
            "modeled_fixed_array_peak_bytes": None,
            "modeled_fixed_array_peak_components": {},
            "modeled_live_row_peak_bytes": None,
            "modeled_artifact_bytes": (
                CAMPAIGN_ESTIMATED_ARTIFACT_BASE_BYTES
                + expected_cell_count
                * CAMPAIGN_ESTIMATED_CELL_ARTIFACT_BYTES
                + config_string_artifact_bytes
            ),
            "modeled_config_string_utf8_bytes": (
                config_string_utf8_bytes
            ),
            "modeled_config_string_artifact_bytes": (
                config_string_artifact_bytes
            ),
            "conservative_total_estimated_bytes": None,
            "budget": asdict(budget),
            "hard_modeled_ceiling_bytes": (
                CAMPAIGN_HARD_MAX_ESTIMATED_BYTES
            ),
            "campaign_arrays_allocated": False,
            "process_rss_measured": False,
            "process_rss_limit_enforced": False,
            "estimate_is_process_rss": False,
            "rss_limitation": (
                "the 512 MiB ceiling covers modeled fixed arrays, live row "
                "allowance, and artifact size; Python allocator, library "
                "workspace, and process RSS are not measured"
            ),
            "checkpoint_resume_status": (
                "BLOCKED_NOT_IMPLEMENTED_IN_THIS_SLICE"
            ),
        }

    available_cell_count = 0
    available_base_group_count = 0
    modeled_trial_count = 0
    modeled_retained_row_count = 0
    modeled_raw_provenance_rows = 0
    modeled_peak_live_retained_rows = 0
    fixed_array_peak = 0
    fixed_array_peak_components: Dict[str, int] = {}
    for seed, length, carrier_family, layers in product(
        config.seeds,
        config.lengths,
        config.carrier_families,
        config.layers,
    ):
        if _carrier_availability(length, carrier_family) is not None:
            continue
        available_base_group_count += 1
        fixed_components = _modeled_base_group_fixed_array_bytes(
            config,
            length=int(length),
            layers=int(layers),
        )
        if fixed_components["total"] > fixed_array_peak:
            fixed_array_peak = fixed_components["total"]
            fixed_array_peak_components = fixed_components
        decoder_count = sum(
            _mask_availability(layers, policy, decoder=True) is None
            for policy in config.decoder_mask_policies
        )
        for planted_mask_family in config.planted_mask_families:
            if (
                _mask_availability(
                    layers,
                    planted_mask_family,
                    decoder=False,
                )
                is not None
            ):
                continue
            for bit_flip_rate in config.bit_flip_rates:
                primary = _primary_base_group(
                    config,
                    length=length,
                    carrier_family=carrier_family,
                    layers=layers,
                    planted_mask_family=planted_mask_family,
                    bit_flip_rate=bit_flip_rate,
                )
                budgets = _trial_budgets(
                    config,
                    primary_base_group=primary,
                )
                trial_count = sum(
                    count
                    for split_budgets in budgets.values()
                    for count in split_budgets.values()
                )
                retained_count = trial_count * decoder_count
                modeled_trial_count += trial_count
                modeled_retained_row_count += retained_count
                modeled_peak_live_retained_rows = max(
                    modeled_peak_live_retained_rows,
                    retained_count,
                )
                available_cell_count += (
                    decoder_count * len(config.payload_noise_rates)
                )
                if (
                    config.run_label == "FULL_METHOD_DEV"
                    and seed in FROZEN_SEEDS
                    and length == 4691
                    and carrier_family == "legendre"
                    and layers == 8
                    and planted_mask_family == "typed16"
                    and type(bit_flip_rate) is float
                    and bit_flip_rate
                    in RAW_SANITY_REQUIRED_BIT_FLIP_RATES
                    and "typed16" in config.decoder_mask_policies
                    and 0.25 in config.payload_noise_rates
                ):
                    modeled_raw_provenance_rows += (
                        RAW_SANITY_REPORT_PLANTED_GROUPS[bit_flip_rate]
                    )

    modeled_decoder_retained_rows = modeled_retained_row_count
    modeled_retained_row_count += modeled_raw_provenance_rows
    modeled_peak_live_retained_rows += modeled_raw_provenance_rows
    live_row_peak_bytes = (
        modeled_peak_live_retained_rows
        * CAMPAIGN_ESTIMATED_LIVE_ROW_BYTES
    )
    artifact_bytes = (
        CAMPAIGN_ESTIMATED_ARTIFACT_BASE_BYTES
        + expected_cell_count * CAMPAIGN_ESTIMATED_CELL_ARTIFACT_BYTES
        + modeled_raw_provenance_rows
        * CAMPAIGN_ESTIMATED_RAW_PROVENANCE_ROW_BYTES
        + config_string_artifact_bytes
    )
    total_estimated_bytes = (
        fixed_array_peak + live_row_peak_bytes + artifact_bytes
    )
    if available_cell_count == 0:
        refusal_reasons.append("campaign has no structurally available cells")
    if modeled_retained_row_count > budget.max_retained_rows:
        refusal_reasons.append(
            "modeled retained-row count exceeds max_retained_rows"
        )
    if total_estimated_bytes > budget.max_estimated_bytes:
        refusal_reasons.append(
            "conservative fixed-array/live-row/artifact estimate exceeds "
            "max_estimated_bytes"
        )
    allowed = not refusal_reasons
    return {
        "status": "ALLOWED" if allowed else "REFUSED",
        "allowed": allowed,
        "refusal_reasons": refusal_reasons,
        "expected_cell_count": expected_cell_count,
        "available_cell_count": available_cell_count,
        "expected_base_group_count": expected_base_group_count,
        "available_base_group_count": available_base_group_count,
        "modeled_trial_count": modeled_trial_count,
        "modeled_retained_row_count": modeled_retained_row_count,
        "modeled_decoder_retained_row_count": (
            modeled_decoder_retained_rows
        ),
        "modeled_serialized_raw_provenance_row_count": (
            modeled_raw_provenance_rows
        ),
        "modeled_peak_live_retained_rows": (
            modeled_peak_live_retained_rows
        ),
        "modeled_fixed_array_peak_bytes": fixed_array_peak,
        "modeled_fixed_array_peak_components": (
            fixed_array_peak_components
        ),
        "modeled_live_row_peak_bytes": live_row_peak_bytes,
        "modeled_artifact_bytes": artifact_bytes,
        "modeled_config_string_utf8_bytes": config_string_utf8_bytes,
        "modeled_config_string_artifact_bytes": (
            config_string_artifact_bytes
        ),
        "conservative_total_estimated_bytes": total_estimated_bytes,
        "budget": asdict(budget),
        "hard_modeled_ceiling_bytes": (
            CAMPAIGN_HARD_MAX_ESTIMATED_BYTES
        ),
        "campaign_arrays_allocated": False,
        "process_rss_measured": False,
        "process_rss_limit_enforced": False,
        "estimate_is_process_rss": False,
        "rss_limitation": (
            "the 512 MiB ceiling covers modeled fixed arrays, live row "
            "allowance, and artifact size; Python allocator, library "
            "workspace, and process RSS are not measured or guaranteed"
        ),
        "checkpoint_resume_status": "BLOCKED_NOT_IMPLEMENTED_IN_THIS_SLICE",
    }


def _check_campaign_deadline(
    started_at: float,
    budget: CampaignExecutionBudget,
    checkpoint: str,
) -> float:
    elapsed = time.perf_counter() - started_at
    if elapsed > budget.max_wall_clock_seconds:
        raise CampaignExecutionLimitError(
            "campaign wall-clock deadline exceeded at "
            f"{checkpoint}: {elapsed:.6f}s > "
            f"{budget.max_wall_clock_seconds:.6f}s"
        )
    return elapsed


def _base_result_retained_row_count(
    base_result: Mapping[str, Any],
) -> int:
    rows = base_result.get("rows")
    if not isinstance(rows, Mapping):
        raise CampaignExecutionLimitError(
            "base result omitted retained row mappings"
        )
    count = 0
    for policy_rows in rows.values():
        if not isinstance(policy_rows, Mapping):
            raise CampaignExecutionLimitError(
                "base result policy rows are malformed"
            )
        for split_rows in policy_rows.values():
            if not isinstance(split_rows, Mapping):
                raise CampaignExecutionLimitError(
                    "base result split rows are malformed"
                )
            for retained_rows in split_rows.values():
                if not isinstance(retained_rows, list):
                    raise CampaignExecutionLimitError(
                        "base result retained rows must be lists"
                    )
                count += len(retained_rows)
    return count


def _enforce_retained_row_cap(
    retained_rows: int,
    budget: CampaignExecutionBudget,
) -> None:
    if retained_rows > budget.max_retained_rows:
        raise CampaignExecutionLimitError(
            "campaign retained-row hard cap exceeded: "
            f"{retained_rows} > {budget.max_retained_rows}"
        )


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
    resident_context_bytes: int,
    deadline_check: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run one observation group and return rows for every compatible decoder."""

    resident_numpy_context = _plain_int(
        resident_context_bytes,
        "resident_context_bytes",
        0,
    )
    decoder_policies = tuple(
        policy
        for policy in config.decoder_mask_policies
        if _mask_availability(layers, policy, decoder=True) is None
    )
    rows: Dict[str, Dict[str, Dict[str, list]]] = {
        policy: {
            split: {
                "planted": [],
                "unrelated": [],
                "non_cyclic": [],
                "block_correlated": [],
                "burst": [],
            }
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
        if deadline_check is not None:
            deadline_check(f"base_group_split_{split}")
        for input_class in (
            "planted",
            "unrelated",
            "non_cyclic",
            "block_correlated",
            "burst",
        ):
            for index in range(budgets[split][input_class]):
                if deadline_check is not None:
                    deadline_check(
                        f"base_group_{split}_{input_class}_{index}"
                    )
                truth_class = (
                    "planted"
                    if input_class in ("block_correlated", "burst")
                    else input_class
                )
                truth = _trial_truth(
                    config, seed, split, truth_class, index, length
                )
                if input_class in ("block_correlated", "burst"):
                    truth = {
                        **truth,
                        "group_id": _trial_group_id(
                            seed, split, input_class, index
                        ),
                    }
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
                elif input_class in ("block_correlated", "burst"):
                    query, planted_mask, query_provenance = _make_planted_query(
                        config,
                        seed,
                        split,
                        truth,
                        templates,
                        ring_templates,
                        planted_mask_family,
                        bit_flip_rate,
                        noise_variant=(
                            "block_correlated"
                            if input_class == "block_correlated"
                            else "burst"
                        ),
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

                if input_class in ("planted", "unrelated"):
                    control_context_bytes = int(
                        resident_numpy_context + query.nbytes
                    )
                    native_observation_estimate = (
                        _native_observation_resource(signatures)
                    )
                    native_decoder_estimate = _native_decoder_resource(
                        signatures
                    )
                    native_standalone_peak = max(
                        native_observation_estimate[
                            "standalone_estimated_peak_bytes"
                        ],
                        native_decoder_estimate[
                            "standalone_estimated_peak_bytes"
                        ],
                    )
                    native_preflight = _with_harness_dense_context(
                        {
                            **native_decoder_estimate,
                            "standalone_estimated_peak_bytes": int(
                                native_standalone_peak
                            ),
                        },
                        dense_context_bytes=control_context_bytes,
                    )
                    repeated_preflight = None
                    repeated_preflight_codebook = _repeated_bit_codebook(
                        config.type_count,
                        layers,
                    )
                    if repeated_preflight_codebook is not None:
                        repeated_preflight = _with_harness_dense_context(
                            _repeated_bit_resource(
                                repeated_preflight_codebook,
                                length,
                                layers,
                            ),
                            dense_context_bytes=control_context_bytes,
                        )
                    native_positions, native_provenance = _observe_native_oppw(
                        config,
                        seed,
                        split,
                        truth,
                        signatures,
                        input_class,
                        bit_flip_rate,
                    )
                    native_oppw = _decode_native_oppw(
                        native_positions, signatures, length
                    )
                    native_stage_peaks = (
                        native_provenance["resource_accounting"][
                            "standalone_estimated_peak_bytes"
                        ],
                        native_oppw["resource_accounting"][
                            "standalone_estimated_peak_bytes"
                        ],
                    )
                    native_oppw["resource_accounting"] = {
                        **native_preflight,
                        "observation_stage_estimated_peak_bytes": int(
                            native_stage_peaks[0]
                        ),
                        "decoder_stage_estimated_peak_bytes": int(
                            native_stage_peaks[1]
                        ),
                    }
                    native_oppw["observation_contract"] = native_provenance
                    repeated_bit = _observe_and_decode_repeated_bit(
                        config,
                        seed,
                        split,
                        truth,
                        input_class,
                        length,
                        layers,
                        bit_flip_rate,
                    )
                    if repeated_bit.get("status") == "COMPLETED":
                        if repeated_preflight is None:
                            raise CampaignValidationError(
                                "completed repeated-bit control lacked preflight"
                            )
                        repeated_bit["resource_accounting"] = repeated_preflight
                else:
                    native_oppw = {
                        "status": "NOT_APPLICABLE",
                        "reason": (
                            "control has its own frozen native symbol channel; "
                            "dense-only corruption variant is not reused"
                        ),
                    }
                    repeated_bit = {
                        "status": "NOT_APPLICABLE",
                        "reason": (
                            "dense corruption variant is descriptive and not "
                            "part of the repeated-bit threshold cell"
                        ),
                    }

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
                        "native_oppw_control": native_oppw,
                        "repeated_bit_control": repeated_bit,
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
                                "native_oppw_type_correct": (
                                    native_oppw.get("predicted_type")
                                    == true_type
                                    if native_oppw.get("status", "COMPLETED")
                                    != "NOT_APPLICABLE"
                                    else None
                                ),
                                "native_oppw_phase_exact": (
                                    native_oppw.get("predicted_type")
                                    == true_type
                                    and native_oppw.get("shared_shift")
                                    == true_shift
                                    if native_oppw.get("status", "COMPLETED")
                                    != "NOT_APPLICABLE"
                                    else None
                                ),
                                "repeated_bit_type_correct": (
                                    repeated_bit.get("predicted_type")
                                    == true_type
                                    if repeated_bit.get("status") == "COMPLETED"
                                    else None
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
    correctness_keys = {
        "independent_control": "independent_type_correct",
        "oppw_form_control": "oppw_type_correct",
        "native_oppw_control": "native_oppw_type_correct",
        "repeated_bit_control": "repeated_bit_type_correct",
    }
    if key not in correctness_keys:
        raise CampaignValidationError(f"unknown control threshold key {key}")
    converted = []
    for row in rows:
        control = row[key]
        if control.get("status") == "STRUCTURALLY_UNAVAILABLE":
            continue
        converted.append(
            {
                "split": row["split"],
                "group_id": row["group_id"],
                "score": control["score"],
                "margin": control["margin"],
                "type_correct": row.get(correctness_keys[key], False),
                "resource_accounting": control.get(
                    "resource_accounting"
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
    resources = [
        row["resource_accounting"]
        for row in (
            list(select_planted)
            + list(select_unrelated)
            + list(report_planted)
            + list(report_unrelated)
        )
        if row.get("resource_accounting") is not None
    ]
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
        "resource_accounting": {
            "status": "CONSERVATIVE_ESTIMATE_RETAINED",
            "standalone_estimated_peak_bytes": (
                _summary(
                    resource["standalone_estimated_peak_bytes"]
                    for resource in resources
                )
                if resources
                else None
            ),
            "harness_estimated_peak_bytes": (
                _summary(
                    resource["harness_estimated_peak_bytes"]
                    for resource in resources
                    if "harness_estimated_peak_bytes" in resource
                )
                if any(
                    "harness_estimated_peak_bytes" in resource
                    for resource in resources
                )
                else None
            ),
            "estimated_work_units": (
                _summary(
                    resource["estimated_work_units"]
                    for resource in resources
                )
                if resources
                else None
            ),
            "measured_process_peak": False,
            "hard_max_estimated_bytes": (
                CONTROL_HARD_MAX_ESTIMATED_BYTES
            ),
            "hard_max_work_units": CONTROL_HARD_MAX_WORK_UNITS,
        },
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
    report_block_correlated = rows["REPORT"]["block_correlated"]
    report_burst = rows["REPORT"]["burst"]
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
    unlocked_block_correlated, correct_block_correlated = _apply_threshold(
        report_block_correlated, threshold
    )
    unlocked_burst, correct_burst = _apply_threshold(report_burst, threshold)
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
    native_oppw = _aggregate_control(
        _control_threshold_rows(select_planted, "native_oppw_control"),
        _control_threshold_rows(select_unrelated, "native_oppw_control"),
        _control_threshold_rows(report_planted, "native_oppw_control"),
        _control_threshold_rows(report_unrelated, "native_oppw_control"),
        target_false_unlock_rate=config.target_false_unlock_rate,
        confidence_level=config.confidence_level,
        allow_smoke_point_estimate=(
            config.run_label == "SMOKE_NON_EVIDENTIARY"
        ),
    )
    native_oppw.update(
        {
            "status": "COMPLETED",
            "phase_exact_accuracy": _mean_bool(
                row["native_oppw_phase_exact"] for row in report_planted
            ),
            "contract": report_planted[0]["native_oppw_control"][
                "observation_contract"
            ],
            "dense_phase_estimates_consumed": False,
        }
    )
    repeated_available = (
        select_planted[0]["repeated_bit_control"].get("status")
        == "COMPLETED"
    )
    if repeated_available:
        repeated_bit = _aggregate_control(
            _control_threshold_rows(select_planted, "repeated_bit_control"),
            _control_threshold_rows(
                select_unrelated, "repeated_bit_control"
            ),
            _control_threshold_rows(report_planted, "repeated_bit_control"),
            _control_threshold_rows(
                report_unrelated, "repeated_bit_control"
            ),
            target_false_unlock_rate=config.target_false_unlock_rate,
            confidence_level=config.confidence_level,
            allow_smoke_point_estimate=(
                config.run_label == "SMOKE_NON_EVIDENTIARY"
            ),
        )
        repeated_bit.update(
            {
                "status": "COMPLETED",
                "phase_recovery_status": (
                    "NOT_DEFINED_FOR_REPEATED_BIT_CONTROL"
                ),
                "contract": report_planted[0]["repeated_bit_control"][
                    "contract"
                ],
            }
        )
    else:
        repeated_bit = {
            **select_planted[0]["repeated_bit_control"],
            "matched_resource_advantage_claim_eligible": False,
        }

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
    raw_type_accuracy_provenance = _raw_type_accuracy_provenance(
        config,
        spec,
        report_planted,
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
        "campaign_contract_digest": _campaign_contract_manifest(config)[
            "digest"
        ],
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
            "block_correlated_group_count": len(report_block_correlated),
            "burst_group_count": len(report_burst),
            "raw_type_accuracy": raw_type_accuracy,
            "raw_type_accuracy_provenance": (
                raw_type_accuracy_provenance
            ),
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
            "correlated_noise_variants": {
                "declared_global_flip_fraction": float(
                    spec["bit_flip_rate"]
                ),
                "bsc_binomial_theory_applicable": False,
                "block_correlated": {
                    "observed_flip_rate": _summary(
                        row["query_provenance"]["observed_flip_rate"]
                        for row in report_block_correlated
                    ),
                    "raw_type_accuracy": _mean_bool(
                        row["type_correct"]
                        for row in report_block_correlated
                    ),
                    "true_unlock_recall": (
                        _mean_bool(correct_block_correlated)
                        if correct_block_correlated is not None
                        else None
                    ),
                    "unlock_rate": (
                        _mean_bool(unlocked_block_correlated)
                        if unlocked_block_correlated is not None
                        else None
                    ),
                },
                "burst": {
                    "observed_flip_rate": _summary(
                        row["query_provenance"]["observed_flip_rate"]
                        for row in report_burst
                    ),
                    "raw_type_accuracy": _mean_bool(
                        row["type_correct"] for row in report_burst
                    ),
                    "true_unlock_recall": (
                        _mean_bool(correct_burst)
                        if correct_burst is not None
                        else None
                    ),
                    "unlock_rate": (
                        _mean_bool(unlocked_burst)
                        if unlocked_burst is not None
                        else None
                    ),
                },
                "threshold_source": "iid SELECT only",
                "promotion_eligible": False,
            },
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
            "native_sparse_oppw": native_oppw,
            "equal_channel_use_repeated_bit": repeated_bit,
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


@dataclass(frozen=True)
class _RawSanityLiveSeal:
    """Compact in-process attestation derived from actual REPORT rows."""

    schema_version: str
    authority_id: str
    cell_id: str
    campaign_contract_digest: str
    carrier_provenance_digest: str
    raw_provenance_digest: str
    trial_count: int
    correct_count: int
    raw_type_accuracy: float
    authentication_tag: str


class _RawSanityLiveAuthority:
    """Per-run authority that never enters the retained campaign artifact."""

    __slots__ = ("_authority_id", "_campaign_contract_digest", "_key")

    def __init__(self, campaign_contract_digest: str) -> None:
        if (
            type(campaign_contract_digest) is not str
            or len(campaign_contract_digest) != 64
        ):
            raise CampaignValidationError(
                "raw sanity authority requires one campaign contract digest"
            )
        self._campaign_contract_digest = campaign_contract_digest
        self._key = secrets.token_bytes(32)
        self._authority_id = hashlib.sha256(
            self._key + campaign_contract_digest.encode("ascii")
        ).hexdigest()

    @property
    def authority_id(self) -> str:
        return self._authority_id

    @property
    def campaign_contract_digest(self) -> str:
        return self._campaign_contract_digest

    def _authentication_tag(self, fields: Mapping[str, Any]) -> str:
        payload = json.dumps(
            fields,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hmac.new(self._key, payload, hashlib.sha256).hexdigest()

    def mint(
        self,
        *,
        cell_id: str,
        campaign_contract_digest: str,
        carrier_provenance_digest: str,
        raw_provenance_digest: str,
        trial_count: int,
        correct_count: int,
        raw_type_accuracy: float,
    ) -> _RawSanityLiveSeal:
        fields = {
            "schema_version": "1.1",
            "authority_id": self._authority_id,
            "cell_id": cell_id,
            "campaign_contract_digest": campaign_contract_digest,
            "carrier_provenance_digest": carrier_provenance_digest,
            "raw_provenance_digest": raw_provenance_digest,
            "trial_count": trial_count,
            "correct_count": correct_count,
            "raw_type_accuracy": raw_type_accuracy,
        }
        if campaign_contract_digest != self._campaign_contract_digest:
            raise CampaignValidationError(
                "raw sanity authority cannot mint across campaign contracts"
            )
        return _RawSanityLiveSeal(
            **fields,
            authentication_tag=self._authentication_tag(fields),
        )

    def verifies(self, seal: object) -> bool:
        if type(seal) is not _RawSanityLiveSeal:
            return False
        fields = {
            "schema_version": seal.schema_version,
            "authority_id": seal.authority_id,
            "cell_id": seal.cell_id,
            "campaign_contract_digest": seal.campaign_contract_digest,
            "carrier_provenance_digest": seal.carrier_provenance_digest,
            "raw_provenance_digest": seal.raw_provenance_digest,
            "trial_count": seal.trial_count,
            "correct_count": seal.correct_count,
            "raw_type_accuracy": seal.raw_type_accuracy,
        }
        return bool(
            seal.authority_id == self._authority_id
            and seal.campaign_contract_digest
            == self._campaign_contract_digest
            and hmac.compare_digest(
                seal.authentication_tag,
                self._authentication_tag(fields),
            )
        )


def _raw_sanity_cell_coordinates(
    cell_config: object,
) -> Optional[Tuple[int, float]]:
    """Return exact frozen coordinates only for the canonical sanity cell."""

    if not isinstance(cell_config, Mapping):
        return None
    if set(cell_config) != set(RAW_SANITY_CELL_CONFIG_FIELDS):
        return None
    seed = cell_config.get("seed")
    length = cell_config.get("length")
    carrier_family = cell_config.get("carrier_family")
    layers = cell_config.get("layers")
    planted_mask_family = cell_config.get("planted_mask_family")
    decoder_mask_policy = cell_config.get("decoder_mask_policy")
    bit_flip_rate = cell_config.get("bit_flip_rate")
    payload_noise_rate = cell_config.get("payload_noise_rate")
    if type(seed) is not int or seed not in FROZEN_SEEDS:
        return None
    if type(length) is not int or length != 4691:
        return None
    if type(carrier_family) is not str or carrier_family != "legendre":
        return None
    if type(layers) is not int or layers != 8:
        return None
    if (
        type(planted_mask_family) is not str
        or planted_mask_family != "typed16"
    ):
        return None
    if (
        type(decoder_mask_policy) is not str
        or decoder_mask_policy != "typed16"
    ):
        return None
    if (
        type(bit_flip_rate) is not float
        or bit_flip_rate not in RAW_SANITY_REQUIRED_BIT_FLIP_RATES
    ):
        return None
    if type(payload_noise_rate) is not float or payload_noise_rate != 0.25:
        return None
    return seed, bit_flip_rate


def _raw_sanity_group_ids(seed: int, trial_count: int) -> list:
    return [
        _trial_group_id(seed, "REPORT", "planted", index)
        for index in range(trial_count)
    ]


def _raw_type_accuracy_provenance(
    config: CampaignConfig,
    spec: Mapping[str, Any],
    report_rows: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Retain the exact pre-threshold IID REPORT rows behind raw accuracy."""

    if config.run_label != "FULL_METHOD_DEV":
        return None
    coordinates = _raw_sanity_cell_coordinates(spec)
    if coordinates is None:
        return None
    seed, bit_flip_rate = coordinates
    expected_count = RAW_SANITY_REPORT_PLANTED_GROUPS[bit_flip_rate]
    if len(report_rows) != expected_count:
        raise CampaignValidationError(
            "raw sanity provenance requires the frozen REPORT planted count"
        )
    expected_group_ids = _raw_sanity_group_ids(seed, expected_count)
    provenance_rows = []
    for index, row in enumerate(report_rows):
        if not isinstance(row, Mapping):
            raise CampaignValidationError(
                "raw sanity provenance rows must be mappings"
            )
        query_provenance = row.get("query_provenance")
        if not isinstance(query_provenance, Mapping):
            raise CampaignValidationError(
                "raw sanity provenance requires query provenance"
            )
        if (
            row.get("split") != "REPORT"
            or row.get("input_class") != "planted"
            or row.get("group_id") != expected_group_ids[index]
            or query_provenance.get("noise_variant") != "iid"
            or query_provenance.get("representation")
            != "raw_float64_ndarray"
            or query_provenance.get("corruption")
            != "independent_bernoulli_bit_flips"
        ):
            raise CampaignValidationError(
                "raw sanity provenance requires canonical planted IID REPORT rows"
            )
        declared_rate = query_provenance.get(
            "declared_global_flip_probability_or_fraction"
        )
        true_type = row.get("true_type")
        predicted_type = row.get("predicted_type")
        type_correct = row.get("type_correct")
        if type(declared_rate) is not float or declared_rate != bit_flip_rate:
            raise CampaignValidationError(
                "raw sanity row declared bit-flip rate is not exact"
            )
        if (
            type(true_type) is not int
            or type(predicted_type) is not int
            or type(type_correct) is not bool
            or type_correct is not (predicted_type == true_type)
        ):
            raise CampaignValidationError(
                "raw sanity row outcome types or correctness are invalid"
            )
        provenance_rows.append(
            {
                "group_id": expected_group_ids[index],
                "split": "REPORT",
                "input_class": "planted",
                "noise_variant": "iid",
                "representation": "raw_float64_ndarray",
                "corruption": "independent_bernoulli_bit_flips",
                "declared_bit_flip_rate": bit_flip_rate,
                "threshold_applied": False,
                "true_type": true_type,
                "predicted_type": predicted_type,
                "type_correct": type_correct,
            }
        )
    correct_count = sum(row["type_correct"] for row in provenance_rows)
    return {
        "schema_version": "1.0",
        "metric": "raw_type_accuracy",
        "campaign_contract_digest": _campaign_contract_manifest(config)[
            "digest"
        ],
        "source_split": "REPORT",
        "input_class": "planted",
        "noise_variant": "iid",
        "threshold_applied": False,
        "trial_count": expected_count,
        "correct_count": correct_count,
        "group_digest": _stable_digest(expected_group_ids),
        "row_digest": _stable_digest(provenance_rows),
        "rows": provenance_rows,
    }


def _raw_sanity_live_seal(
    config: CampaignConfig,
    spec: Mapping[str, Any],
    report_rows: Sequence[Mapping[str, Any]],
    carrier_provenance: Mapping[str, Any],
    *,
    authority: _RawSanityLiveAuthority,
) -> Optional[_RawSanityLiveSeal]:
    """Seal actual in-process rows without retaining them in the artifact."""

    provenance = _raw_type_accuracy_provenance(
        config,
        spec,
        report_rows,
    )
    if provenance is None:
        return None
    if not isinstance(carrier_provenance, Mapping):
        raise CampaignValidationError(
            "raw sanity live seal requires carrier provenance"
        )
    expected_shape = [config.type_count, 8, 4691]
    expected_template_nbytes = config.type_count * 8 * 4691 * 8
    carrier_checks = {
        "representation": "prime_legendre_ring",
        "template_factory": "make_template",
        "ring_observation_labels_manually_constructed": False,
        "base_shared_across_types": True,
        "composite_control": False,
        "template_dtype": "float64",
        "template_shape": expected_shape,
        "template_nbytes": expected_template_nbytes,
        "base_nbytes": 4691 * 8,
        "corrupted_queries_are_raw_array_copies": True,
    }
    if any(
        carrier_provenance.get(field) != expected
        for field, expected in carrier_checks.items()
    ):
        raise CampaignValidationError(
            "raw sanity live seal requires canonical 4691 Legendre provenance"
        )
    correct_count = provenance["correct_count"]
    trial_count = provenance["trial_count"]
    return authority.mint(
        cell_id=_cell_id(spec),
        campaign_contract_digest=provenance[
            "campaign_contract_digest"
        ],
        carrier_provenance_digest=_stable_digest(carrier_provenance),
        raw_provenance_digest=_stable_digest(provenance),
        trial_count=trial_count,
        correct_count=correct_count,
        raw_type_accuracy=float(correct_count / trial_count),
    )


def _raw_sanity_provenance_reasons(
    provenance: object,
    *,
    config: CampaignConfig,
    seed: int,
    bit_flip_rate: float,
    expected_count: int,
    expected_campaign_digest: str,
) -> Tuple[list, Optional[int], Optional[int]]:
    """Validate and recompute every retained raw-metric provenance field."""

    reasons = []
    if not isinstance(provenance, Mapping):
        return ["raw_type_accuracy_provenance_missing"], None, None
    expected_fields = {
        "schema_version",
        "metric",
        "campaign_contract_digest",
        "source_split",
        "input_class",
        "noise_variant",
        "threshold_applied",
        "trial_count",
        "correct_count",
        "group_digest",
        "row_digest",
        "rows",
    }
    if set(provenance) != expected_fields:
        reasons.append("raw_provenance_schema_mismatch")
    for field, expected in (
        ("schema_version", "1.0"),
        ("metric", "raw_type_accuracy"),
        ("campaign_contract_digest", expected_campaign_digest),
        ("source_split", "REPORT"),
        ("input_class", "planted"),
        ("noise_variant", "iid"),
    ):
        if type(provenance.get(field)) is not str or provenance.get(
            field
        ) != expected:
            reasons.append(f"raw_provenance_{field}_mismatch")
    if provenance.get("threshold_applied") is not False:
        reasons.append("raw_provenance_threshold_applied")

    trial_count = provenance.get("trial_count")
    correct_count = provenance.get("correct_count")
    if type(trial_count) is not int or trial_count != expected_count:
        reasons.append("raw_provenance_trial_count_mismatch")
        validated_trial_count = None
    else:
        validated_trial_count = trial_count
    if (
        type(correct_count) is not int
        or correct_count < 0
        or correct_count > expected_count
    ):
        reasons.append("raw_provenance_correct_count_invalid")
        validated_correct_count = None
    else:
        validated_correct_count = correct_count

    rows = provenance.get("rows")
    expected_group_ids = _raw_sanity_group_ids(seed, expected_count)
    expected_row_fields = {
        "group_id",
        "split",
        "input_class",
        "noise_variant",
        "representation",
        "corruption",
        "declared_bit_flip_rate",
        "threshold_applied",
        "true_type",
        "predicted_type",
        "type_correct",
    }
    rows_valid = isinstance(rows, list) and len(rows) == expected_count
    if not rows_valid:
        reasons.append("raw_provenance_rows_count_mismatch")
    else:
        observed_group_ids = []
        derived_correct_count = 0
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != expected_row_fields:
                rows_valid = False
                reasons.append("raw_provenance_row_schema_mismatch")
                break
            group_id = row.get("group_id")
            true_type = row.get("true_type")
            predicted_type = row.get("predicted_type")
            type_correct = row.get("type_correct")
            canonical_truth = _trial_truth(
                config,
                seed,
                "REPORT",
                "planted",
                index,
                4691,
            )
            if type(group_id) is not str:
                rows_valid = False
                reasons.append("raw_provenance_group_id_type_invalid")
                break
            observed_group_ids.append(group_id)
            if (
                group_id != expected_group_ids[index]
                or row.get("split") != "REPORT"
                or row.get("input_class") != "planted"
                or row.get("noise_variant") != "iid"
                or row.get("representation") != "raw_float64_ndarray"
                or row.get("corruption")
                != "independent_bernoulli_bit_flips"
                or type(row.get("declared_bit_flip_rate")) is not float
                or row.get("declared_bit_flip_rate") != bit_flip_rate
                or row.get("threshold_applied") is not False
                or type(true_type) is not int
                or type(predicted_type) is not int
                or not 0 <= true_type < 16
                or not 0 <= predicted_type < 16
                or type(type_correct) is not bool
                or type_correct is not (predicted_type == true_type)
            ):
                rows_valid = False
                reasons.append("raw_provenance_row_contract_mismatch")
                break
            if true_type != int(canonical_truth["true_type"]):
                rows_valid = False
                reasons.append("raw_provenance_true_type_not_canonical")
                break
            derived_correct_count += int(type_correct)
        if rows_valid:
            if observed_group_ids != expected_group_ids:
                rows_valid = False
                reasons.append("raw_provenance_group_sequence_mismatch")
            if provenance.get("group_digest") != _stable_digest(
                expected_group_ids
            ):
                reasons.append("raw_provenance_group_digest_mismatch")
            if provenance.get("row_digest") != _stable_digest(rows):
                reasons.append("raw_provenance_row_digest_mismatch")
            if validated_correct_count != derived_correct_count:
                reasons.append("raw_provenance_correct_count_mismatch")
    return reasons, validated_trial_count, validated_correct_count


def _raw_sanity_cell_key(seed: int, bit_flip_rate: float) -> str:
    return f"seed={seed}|q={bit_flip_rate:.2f}"


def _aggregate_raw_sanity_sweep(
    config: CampaignConfig,
    cells: Sequence[Mapping[str, Any]],
    *,
    live_seals: Optional[Mapping[str, _RawSanityLiveSeal]] = None,
    live_authority: Optional[_RawSanityLiveAuthority] = None,
) -> Dict[str, Any]:
    """Check the complete frozen dense raw-type sanity sweep, fail closed.

    This consumes already-aggregated retained cells. It does not generate
    observations, refit thresholds, or infer lower-rate behavior from q=0.45.
    """

    campaign_contract = _campaign_contract_manifest(config)
    expected_campaign_digest = campaign_contract["digest"]
    config_checks = {
        "full_method_dev_run_label": (
            type(config.run_label) is str
            and config.run_label == "FULL_METHOD_DEV"
        ),
        "select_report_streams_disjoint": bool(
            _stream_manifest(config)["select_report_disjoint"]
        ),
        "exact_live_seal_count": (
            isinstance(live_seals, Mapping)
            and len(live_seals)
            == len(FROZEN_SEEDS)
            * len(RAW_SANITY_REQUIRED_BIT_FLIP_RATES)
        ),
        "live_authority_matches_campaign": (
            type(live_authority) is _RawSanityLiveAuthority
            and live_authority.campaign_contract_digest
            == expected_campaign_digest
        ),
        "frozen_seed_order": (
            type(config.seeds) is tuple
            and all(type(seed) is int for seed in config.seeds)
            and config.seeds == FROZEN_SEEDS
        ),
        "all_required_rates_present": (
            type(config.bit_flip_rates) is tuple
            and all(
                type(rate) is float for rate in config.bit_flip_rates
            )
            and all(
                required in config.bit_flip_rates
                for required in RAW_SANITY_REQUIRED_BIT_FLIP_RATES
            )
        ),
        "length_4691_present": (
            type(config.lengths) is tuple
            and all(type(length) is int for length in config.lengths)
            and 4691 in config.lengths
        ),
        "legendre_present": (
            type(config.carrier_families) is tuple
            and all(
                type(family) is str
                for family in config.carrier_families
            )
            and "legendre" in config.carrier_families
        ),
        "eight_layers_present": (
            type(config.layers) is tuple
            and all(type(layers) is int for layers in config.layers)
            and 8 in config.layers
        ),
        "typed16_planted_present": (
            type(config.planted_mask_families) is tuple
            and all(
                type(family) is str
                for family in config.planted_mask_families
            )
            and "typed16" in config.planted_mask_families
        ),
        "typed16_decoder_present": (
            type(config.decoder_mask_policies) is tuple
            and all(
                type(policy) is str
                for policy in config.decoder_mask_policies
            )
            and "typed16" in config.decoder_mask_policies
        ),
        "payload_noise_0_25_present": (
            type(config.payload_noise_rates) is tuple
            and all(
                type(rate) is float
                for rate in config.payload_noise_rates
            )
            and 0.25 in config.payload_noise_rates
        ),
        "sixteen_types": (
            type(config.type_count) is int and config.type_count == 16
        ),
        "eight_waypoints_per_type": (
            type(config.waypoints_per_type) is int
            and config.waypoints_per_type == 8
        ),
        "lower_rate_report_budget_is_three": (
            type(config.report_planted) is int
            and config.report_planted == 3
        ),
        "q_0_45_report_budget_is_1024": (
            type(config.primary_report_planted) is int
            and config.primary_report_planted == 1024
        ),
    }
    config_contract_pass = all(config_checks.values())
    buckets: Dict[Tuple[int, float], list] = {
        (seed, rate): []
        for seed in FROZEN_SEEDS
        for rate in RAW_SANITY_REQUIRED_BIT_FLIP_RATES
    }
    for cell in cells:
        if not isinstance(cell, Mapping):
            continue
        cell_config = cell.get("config")
        coordinates = _raw_sanity_cell_coordinates(cell_config)
        if coordinates is None:
            continue
        buckets[coordinates].append(cell)

    summaries = []
    missing_keys = []
    duplicate_keys = []
    invalid_keys = []
    failed_accuracy_keys = []
    for seed in FROZEN_SEEDS:
        for rate in RAW_SANITY_REQUIRED_BIT_FLIP_RATES:
            key = _raw_sanity_cell_key(seed, rate)
            matched = buckets[(seed, rate)]
            expected_groups = RAW_SANITY_REPORT_PLANTED_GROUPS[rate]
            if not matched:
                missing_keys.append(key)
                summaries.append(
                    {
                        "key": key,
                        "seed": seed,
                        "bit_flip_rate": rate,
                        "status": "MISSING",
                        "expected_report_planted_groups": expected_groups,
                        "contract_record_valid": False,
                        "raw_sanity_pass": False,
                    }
                )
                continue
            if len(matched) != 1:
                duplicate_keys.append(key)
                summaries.append(
                    {
                        "key": key,
                        "seed": seed,
                        "bit_flip_rate": rate,
                        "status": "DUPLICATE",
                        "matched_cell_count": len(matched),
                        "expected_report_planted_groups": expected_groups,
                        "contract_record_valid": False,
                        "raw_sanity_pass": False,
                    }
                )
                continue
            cell = matched[0]
            cell_config = cell["config"]
            report = cell.get("report")
            threshold = cell.get("threshold")
            reasons = []
            if cell.get("status") != "completed":
                reasons.append("cell_not_completed")
            cell_id = cell.get("cell_id")
            if type(cell_id) is not str or cell_id != _cell_id(cell_config):
                reasons.append("cell_id_not_canonical")
            live_seal = (
                live_seals.get(cell_id)
                if isinstance(live_seals, Mapping)
                and type(cell_id) is str
                else None
            )
            if type(live_seal) is not _RawSanityLiveSeal:
                reasons.append("live_source_seal_missing_or_invalid")
                live_seal = None
            else:
                if live_seal.schema_version != "1.1":
                    reasons.append("live_source_seal_schema_mismatch")
                if (
                    type(live_authority) is not _RawSanityLiveAuthority
                    or not live_authority.verifies(live_seal)
                ):
                    reasons.append("live_source_seal_authentication_failed")
                if live_seal.cell_id != cell_id:
                    reasons.append("live_source_seal_cell_id_mismatch")
                if (
                    live_seal.campaign_contract_digest
                    != expected_campaign_digest
                ):
                    reasons.append(
                        "live_source_seal_campaign_digest_mismatch"
                    )
            if (
                type(cell.get("campaign_contract_digest")) is not str
                or cell.get("campaign_contract_digest")
                != expected_campaign_digest
            ):
                reasons.append("campaign_contract_digest_mismatch")
            carrier_provenance = cell.get("carrier_provenance")
            if not isinstance(carrier_provenance, Mapping):
                reasons.append("carrier_provenance_missing")
            elif (
                live_seal is not None
                and _stable_digest(carrier_provenance)
                != live_seal.carrier_provenance_digest
            ):
                reasons.append("carrier_provenance_live_seal_mismatch")
            if cell.get("promotion_eligible") is not False:
                reasons.append("cell_promotion_not_locked_false")
            expected_primary_component = rate == 0.45
            if (
                cell.get("primary_fur_seed_component")
                is not expected_primary_component
            ):
                reasons.append("primary_component_role_mismatch")
            if not isinstance(threshold, Mapping):
                reasons.append("threshold_metadata_missing")
            else:
                if (
                    type(threshold.get("source_split")) is not str
                    or threshold.get("source_split") != "SELECT"
                ):
                    reasons.append("threshold_source_not_select")
                if (
                    type(threshold.get("report_rows_consumed")) is not int
                    or threshold.get("report_rows_consumed") != 0
                ):
                    reasons.append(
                        "threshold_report_rows_consumed_not_zero"
                    )
            if not isinstance(report, Mapping):
                reasons.append("missing_report")
                observed_groups = None
                raw_accuracy = None
                provenance_trial_count = None
                provenance_correct_count = None
            else:
                observed_groups = report.get("planted_group_count")
                if (
                    type(observed_groups) is not int
                    or observed_groups != expected_groups
                ):
                    reasons.append("report_planted_group_count_mismatch")
                raw_value = report.get("raw_type_accuracy")
                if (
                    type(raw_value) is not float
                    or not math.isfinite(raw_value)
                    or not 0.0 <= raw_value <= 1.0
                ):
                    raw_accuracy = None
                    reasons.append("raw_type_accuracy_not_finite_float")
                else:
                    raw_accuracy = raw_value
                (
                    provenance_reasons,
                    provenance_trial_count,
                    provenance_correct_count,
                ) = _raw_sanity_provenance_reasons(
                    report.get("raw_type_accuracy_provenance"),
                    config=config,
                    seed=seed,
                    bit_flip_rate=rate,
                    expected_count=expected_groups,
                    expected_campaign_digest=expected_campaign_digest,
                )
                reasons.extend(provenance_reasons)
                serialized_provenance = report.get(
                    "raw_type_accuracy_provenance"
                )
                if (
                    live_seal is not None
                    and isinstance(serialized_provenance, Mapping)
                    and _stable_digest(serialized_provenance)
                    != live_seal.raw_provenance_digest
                ):
                    reasons.append(
                        "serialized_provenance_live_seal_mismatch"
                    )
                if (
                    raw_accuracy is not None
                    and provenance_trial_count is not None
                    and provenance_correct_count is not None
                    and raw_accuracy
                    != provenance_correct_count / provenance_trial_count
                ):
                    reasons.append(
                        "raw_type_accuracy_provenance_inconsistent"
                    )
                if (
                    live_seal is not None
                    and (
                        provenance_trial_count != live_seal.trial_count
                        or provenance_correct_count
                        != live_seal.correct_count
                        or raw_accuracy != live_seal.raw_type_accuracy
                    )
                ):
                    reasons.append(
                        "raw_metric_live_source_seal_mismatch"
                    )
            valid = not reasons
            raw_pass = bool(
                valid
                and raw_accuracy == 1.0
                and provenance_correct_count == expected_groups
            )
            if not valid:
                invalid_keys.append(key)
            elif not raw_pass:
                failed_accuracy_keys.append(key)
            summaries.append(
                {
                    "key": key,
                    "seed": seed,
                    "bit_flip_rate": rate,
                    "cell_id": cell.get("cell_id"),
                    "status": (
                        "PASS"
                        if raw_pass
                        else "RAW_ACCURACY_FAILURE"
                        if valid
                        else "INVALID"
                    ),
                    "expected_report_planted_groups": expected_groups,
                    "observed_report_planted_groups": observed_groups,
                    "raw_type_accuracy": raw_accuracy,
                    "provenance_trial_count": provenance_trial_count,
                    "provenance_correct_count": provenance_correct_count,
                    "campaign_contract_digest": expected_campaign_digest,
                    "contract_record_valid": valid,
                    "raw_sanity_pass": raw_pass,
                    "invalid_reasons": reasons,
                }
            )
    observed_cell_ids = [
        str(summary["cell_id"])
        for summary in summaries
        if type(summary.get("cell_id")) is str
    ]
    duplicate_cell_ids = sorted(
        {
            cell_id
            for cell_id in observed_cell_ids
            if observed_cell_ids.count(cell_id) > 1
        }
    )
    contract_complete = bool(
        config_contract_pass
        and not missing_keys
        and not duplicate_keys
        and not duplicate_cell_ids
        and not invalid_keys
        and len(summaries)
        == len(FROZEN_SEEDS) * len(RAW_SANITY_REQUIRED_BIT_FLIP_RATES)
    )
    gate_pass = bool(contract_complete and not failed_accuracy_keys)
    rate_summaries = {}
    for rate in RAW_SANITY_REQUIRED_BIT_FLIP_RATES:
        rate_key = f"{rate:.2f}"
        selected = [
            summary
            for summary in summaries
            if summary["bit_flip_rate"] == rate
        ]
        rate_summaries[rate_key] = {
            "required_seed_count": len(FROZEN_SEEDS),
            "contract_record_count": sum(
                bool(summary.get("contract_record_valid"))
                for summary in selected
            ),
            "raw_type_accuracies": [
                summary.get("raw_type_accuracy") for summary in selected
            ],
            "gate_pass": bool(
                len(selected) == len(FROZEN_SEEDS)
                and all(summary["raw_sanity_pass"] for summary in selected)
            ),
        }
    if gate_pass:
        status = "COMPLETE_PASS"
    elif contract_complete:
        status = "COMPLETE_RAW_SANITY_FAILURE"
    else:
        status = "INCOMPLETE_FAIL_CLOSED"
    return {
        "status": status,
        "metric": "dense_report_raw_type_accuracy",
        "required_raw_type_accuracy": 1.0,
        "required_bit_flip_rates": list(
            RAW_SANITY_REQUIRED_BIT_FLIP_RATES
        ),
        "required_seeds": list(FROZEN_SEEDS),
        "required_cell_count": (
            len(FROZEN_SEEDS) * len(RAW_SANITY_REQUIRED_BIT_FLIP_RATES)
        ),
        "expected_report_planted_groups_by_rate": {
            f"{rate:.2f}": RAW_SANITY_REPORT_PLANTED_GROUPS[rate]
            for rate in RAW_SANITY_REQUIRED_BIT_FLIP_RATES
        },
        "configuration_checks": config_checks,
        "configuration_contract_pass": config_contract_pass,
        "campaign_contract": campaign_contract,
        "campaign_contract_digest": expected_campaign_digest,
        "live_source_contract": {
            "required": True,
            "serialized_provenance_alone_sufficient": False,
            "authentication": "per_run_hmac_sha256",
            "authority_present": (
                type(live_authority) is _RawSanityLiveAuthority
            ),
            "authority_secret_serialized": False,
            "authority_id_serialized": False,
            "authority_secret_bytes": 32,
            "seal_count": (
                len(live_seals)
                if isinstance(live_seals, Mapping)
                else 0
            ),
            "estimated_retained_bytes": (
                len(live_seals) * RAW_SANITY_LIVE_SEAL_ESTIMATED_BYTES
                if isinstance(live_seals, Mapping)
                else 0
            ),
            "authority_estimated_retained_bytes": (
                RAW_SANITY_LIVE_AUTHORITY_ESTIMATED_BYTES
                if type(live_authority) is _RawSanityLiveAuthority
                else 0
            ),
            "total_live_attestation_estimated_bytes": (
                (
                    len(live_seals) * RAW_SANITY_LIVE_SEAL_ESTIMATED_BYTES
                    if isinstance(live_seals, Mapping)
                    else 0
                )
                + (
                    RAW_SANITY_LIVE_AUTHORITY_ESTIMATED_BYTES
                    if type(live_authority) is _RawSanityLiveAuthority
                    else 0
                )
            ),
            "estimated_retained_bytes_scope": (
                "live_seal_objects_only_not_cells_or_serialized_provenance"
            ),
            "estimated_retained_bytes_kind": (
                "conservative_python_object_bound_not_process_rss"
            ),
            "additional_full_report_rows_retained_for_live_seal": False,
            "serialized_raw_provenance_rows_retained_in_cells": True,
            "live_seals_serialized_into_artifact": False,
        },
        "cell_summaries": summaries,
        "rate_summaries": rate_summaries,
        "missing_cell_keys": missing_keys,
        "duplicate_cell_keys": duplicate_keys,
        "duplicate_cell_ids": duplicate_cell_ids,
        "invalid_cell_keys": invalid_keys,
        "failed_raw_accuracy_cell_keys": failed_accuracy_keys,
        "contract_complete": contract_complete,
        "gate_pass": gate_pass,
        "lower_rates_inferred_from_q_0_45": False,
        "threshold_refit_performed": False,
        "promotion_eligible": False,
    }


def _primary_aggregate(
    config: CampaignConfig,
    components: Sequence[Mapping[str, Any]],
    *,
    raw_sanity_cells: Optional[Sequence[Mapping[str, Any]]] = None,
    raw_sanity_live_seals: Optional[
        Mapping[str, _RawSanityLiveSeal]
    ] = None,
    raw_sanity_live_authority: Optional[_RawSanityLiveAuthority] = None,
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
    native_oppw = _aggregate_control(
        _control_threshold_rows(select_planted, "native_oppw_control"),
        _control_threshold_rows(select_unrelated, "native_oppw_control"),
        _control_threshold_rows(report_planted, "native_oppw_control"),
        _control_threshold_rows(report_unrelated, "native_oppw_control"),
        target_false_unlock_rate=config.target_false_unlock_rate,
        confidence_level=config.confidence_level,
        allow_smoke_point_estimate=False,
    )
    native_oppw.update(
        {
            "status": "COMPLETED",
            "phase_exact_accuracy": _mean_bool(
                row["native_oppw_phase_exact"] for row in report_planted
            ),
            "contract": report_planted[0]["native_oppw_control"][
                "observation_contract"
            ],
            "dense_phase_estimates_consumed": False,
        }
    )
    repeated_rows = (
        select_planted
        + select_unrelated
        + report_planted
        + report_unrelated
    )
    repeated_available = all(
        row["repeated_bit_control"].get("status") == "COMPLETED"
        for row in repeated_rows
    )
    if repeated_available:
        repeated_bit = _aggregate_control(
            _control_threshold_rows(select_planted, "repeated_bit_control"),
            _control_threshold_rows(select_unrelated, "repeated_bit_control"),
            _control_threshold_rows(report_planted, "repeated_bit_control"),
            _control_threshold_rows(report_unrelated, "repeated_bit_control"),
            target_false_unlock_rate=config.target_false_unlock_rate,
            confidence_level=config.confidence_level,
            allow_smoke_point_estimate=False,
        )
        repeated_bit.update(
            {
                "status": "COMPLETED",
                "phase_recovery_status": (
                    "NOT_DEFINED_FOR_REPEATED_BIT_CONTROL"
                ),
                "contract": report_planted[0]["repeated_bit_control"][
                    "contract"
                ],
            }
        )
    else:
        unavailable = next(
            row["repeated_bit_control"]
            for row in repeated_rows
            if row["repeated_bit_control"].get("status")
            != "COMPLETED"
        )
        repeated_bit = {
            **unavailable,
            "matched_resource_advantage_claim_eligible": False,
        }
    native_control_closed = bool(
        native_oppw["threshold"].get("status") == "FROZEN"
        and native_oppw["fur_gate_pass"]
        and native_oppw["report_true_unlock_recall"] is not None
        and native_oppw["report_true_unlock_recall"]
        >= config.control_min_true_unlock_recall
    )
    native_oppw["minimum_true_unlock_recall_for_closure"] = (
        config.control_min_true_unlock_recall
    )
    native_oppw["true_unlock_recall_floor_gate_pass"] = bool(
        native_oppw["report_true_unlock_recall"] is not None
        and native_oppw["report_true_unlock_recall"]
        >= config.control_min_true_unlock_recall
    )
    repeated_control_closed = bool(
        repeated_available
        and repeated_bit["threshold"].get("status") == "FROZEN"
        and repeated_bit["fur_gate_pass"]
        and repeated_bit["report_true_unlock_recall"] is not None
        and repeated_bit["report_true_unlock_recall"]
        >= config.control_min_true_unlock_recall
    )
    repeated_bit["minimum_true_unlock_recall_for_closure"] = (
        config.control_min_true_unlock_recall
    )
    repeated_bit["true_unlock_recall_floor_gate_pass"] = bool(
        repeated_available
        and repeated_bit.get("report_true_unlock_recall") is not None
        and repeated_bit["report_true_unlock_recall"]
        >= config.control_min_true_unlock_recall
    )
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
    dense_true_unlock_recall = (
        _mean_bool(report_correct) if report_correct is not None else None
    )
    dense_raw_type_accuracy = _mean_bool(
        row.get("type_correct", False) for row in report_planted
    )
    dense_raw_type_sanity_gate_pass = (
        dense_raw_type_accuracy == 1.0
    )
    raw_sanity_sweep = _aggregate_raw_sanity_sweep(
        config,
        () if raw_sanity_cells is None else raw_sanity_cells,
        live_seals=raw_sanity_live_seals,
        live_authority=raw_sanity_live_authority,
    )
    dense_primary_control_closed = bool(
        threshold.get("status") == "FROZEN"
        and fur_gate
        and dense_raw_type_sanity_gate_pass
        and raw_sanity_sweep["gate_pass"]
        and dense_true_unlock_recall is not None
        and dense_true_unlock_recall
        >= config.control_min_true_unlock_recall
    )
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
        "report_true_unlock_recall": dense_true_unlock_recall,
        "dense_raw_type_accuracy": dense_raw_type_accuracy,
        "dense_raw_type_sanity_gate_required": 1.0,
        "dense_raw_type_sanity_gate_scope": "pooled_primary_q_0_45_endpoint",
        "dense_raw_type_sanity_gate_pass": (
            dense_raw_type_sanity_gate_pass
        ),
        "raw_sanity_sweep": raw_sanity_sweep,
        "through_q_0_45_raw_sanity_contract_complete": (
            raw_sanity_sweep["contract_complete"]
        ),
        "through_q_0_45_raw_sanity_gate_complete": (
            raw_sanity_sweep["gate_pass"]
        ),
        "through_q_0_45_raw_sanity_gate_blocker": (
            None
            if raw_sanity_sweep["gate_pass"]
            else (
                "the exact 12-cell q={0.00,0.20,0.35,0.45} retained-cell "
                f"contract did not pass: {raw_sanity_sweep['status']}"
            )
        ),
        "minimum_true_unlock_recall_for_control_closure": (
            config.control_min_true_unlock_recall
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
        "controls": {
            "native_sparse_oppw": native_oppw,
            "equal_channel_use_repeated_bit": repeated_bit,
        },
        "native_sparse_oppw_resource_control_closed": native_control_closed,
        "equal_channel_use_repeated_bit_control_closed": (
            repeated_control_closed
        ),
        "dense_primary_control_closed": dense_primary_control_closed,
        "all_mandatory_controls_closed": bool(
            dense_primary_control_closed
            and native_control_closed
            and repeated_control_closed
        ),
        "pairwise_noise_law_match": {
            "dense_vs_equal_channel_repeated_bit": True,
            "dense_vs_native_sparse_oppw": False,
            "reason": (
                "dense and repeated-bit paths use independent Bernoulli "
                "bipolar chip flips; native OPPW uses symbol substitution"
            ),
        },
        "matched_noise_comparison_ready": False,
        "matched_noise_comparison_blocker": (
            "the dense-versus-repeated-bit comparison has a matched Bernoulli "
            "chip-flip law, but native OPPW uses a symbol-substitution channel; "
            "a three-way or dense-versus-native superiority statement is not "
            "a matched-noise claim"
        ),
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
            "rader_campaign_status": "NOT_TESTED_BY_THIS_CAMPAIGN",
            "rader_implementation_present": True,
            "separate_rader_harness_results_ingested": False,
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
            "native sparse OPPW and dense carriers have explicit equal "
            "channel-use/energy accounting but different declared noise laws"
        ),
        "phase address is known OPPW-equivalent mathematics",
    ]
    if smoke:
        common_blockers.append(
            "pooled full-run native OPPW and repeated-bit control gates were "
            "not evaluated in smoke"
        )
    elif not primary.get("all_mandatory_controls_closed", False):
        common_blockers.append(
            "one or more mandatory pooled control gates did not close"
        )
    if not primary.get(
        "through_q_0_45_raw_sanity_gate_complete",
        False,
    ):
        common_blockers.append(
            "the frozen 12-cell q={0.00,0.20,0.35,0.45} 100% raw-type "
            "sanity contract did not complete and pass"
        )
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
                    "INCONCLUSIVE_SYNTHETIC_CONTROLS_CLOSED"
                    if (
                        primary.get("fur_gate_pass")
                        and primary.get(
                            "all_mandatory_controls_closed", False
                        )
                        and primary.get(
                            "through_q_0_45_raw_sanity_gate_complete",
                            False,
                        )
                    )
                    else (
                        "INCONCLUSIVE_CONTROL_GATE"
                        if primary.get("fur_gate_pass")
                        else "NO_GO_PRIMARY_GATE"
                    )
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
                "native sparse OPPW and equal-channel-use repeated-bit controls "
                "are implemented with explicit contracts, but no full RB-1 or "
                "repeated hardware timing campaign has closed; native OPPW uses "
                "a distinct symbol-noise law; theoretical packed bytes remain "
                "excluded from the cost verdict"
            ),
            "special_status_for_4691": False,
            "promotion_eligible": False,
        },
        "RADER-1": {
            "verdict": "NOT_TESTED_BY_THIS_CAMPAIGN",
            "rader_implementation_present": True,
            "separate_rader_harness_results_ingested": False,
            "promotion_eligible": False,
        },
        "novelty": {
            "verdict": "NOT_ESTABLISHED",
            "phase_address_prior_art_class": "OPPW_2D_optical_orthogonal_code",
            "claim_of_novel_phase_math": False,
            "completed_cell_count": len(completed),
        },
    }


def _validate_json_tree_bounds(value: object) -> None:
    stack = [value]
    visited = 0
    while stack:
        current = stack.pop()
        visited += 1
        if visited > CAMPAIGN_HARD_MAX_JSON_NODES:
            raise CampaignExecutionLimitError(
                "JSON artifact exceeds the hard node-count ceiling"
            )
        if current is None or type(current) in (bool,):
            continue
        if type(current) is int:
            if current.bit_length() > CAMPAIGN_HARD_MAX_INPUT_BITS:
                raise CampaignExecutionLimitError(
                    "JSON artifact integer exceeds the hard input-bit ceiling"
                )
            continue
        if type(current) is float:
            if not math.isfinite(current):
                raise CampaignValidationError(
                    "JSON artifact contains a non-finite float"
                )
            continue
        if type(current) is str:
            if len(current) > CAMPAIGN_HARD_MAX_JSON_STRING_CHARS:
                raise CampaignExecutionLimitError(
                    "JSON artifact string exceeds the hard character ceiling"
                )
            continue
        if type(current) is dict:
            for key, item in current.items():
                if type(key) is not str:
                    raise CampaignValidationError(
                        "JSON artifact object keys must be plain strings"
                    )
                if len(key) > CAMPAIGN_HARD_MAX_JSON_STRING_CHARS:
                    raise CampaignExecutionLimitError(
                        "JSON artifact key exceeds the hard character ceiling"
                    )
                stack.append(key)
                stack.append(item)
            continue
        if type(current) in (list, tuple):
            stack.extend(current)
            continue
        raise CampaignValidationError(
            "JSON artifact contains a non-canonical value of type "
            f"{type(current).__name__}"
        )


def _atomic_write_json(
    output: Path,
    artifact: Mapping[str, Any],
    *,
    deadline_check: Optional[Any] = None,
    max_encoded_bytes: int = CAMPAIGN_HARD_MAX_ESTIMATED_BYTES,
) -> None:
    """Stream one bounded JSON artifact through a same-directory replace."""

    output = Path(output)
    output_text = str(output)
    if (
        len(output_text) > CAMPAIGN_HARD_MAX_OUTPUT_PATH_CHARS
        or len(output_text.encode("utf-8"))
        > CAMPAIGN_HARD_MAX_OUTPUT_PATH_UTF8_BYTES
    ):
        raise CampaignValidationError(
            "output path exceeds the hard path-length ceiling"
        )
    if (
        type(max_encoded_bytes) is not int
        or max_encoded_bytes.bit_length() > CAMPAIGN_HARD_MAX_INPUT_BITS
        or max_encoded_bytes <= 0
        or max_encoded_bytes > CAMPAIGN_HARD_MAX_ESTIMATED_BYTES
    ):
        raise CampaignValidationError(
            "max_encoded_bytes must be a positive plain integer no greater "
            "than the hard modeled ceiling"
        )
    if deadline_check is not None:
        deadline_check("before_artifact_json_encoding")
    _validate_json_tree_bounds(artifact)
    encoder = json.JSONEncoder(
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(output.parent),
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            encoded_bytes = 0
            for chunk in encoder.iterencode(artifact):
                encoded = chunk.encode("utf-8")
                encoded_bytes += len(encoded)
                if encoded_bytes > max_encoded_bytes:
                    raise CampaignExecutionLimitError(
                        "encoded JSON artifact exceeds max_encoded_bytes"
                    )
                temporary.write(encoded)
            if encoded_bytes + 1 > max_encoded_bytes:
                raise CampaignExecutionLimitError(
                    "encoded JSON artifact exceeds max_encoded_bytes"
                )
            temporary.write(b"\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        if deadline_check is not None:
            deadline_check("after_artifact_json_encoding")
        if deadline_check is not None:
            deadline_check("before_artifact_atomic_replace")
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def run_campaign(
    config: CampaignConfig,
    *,
    write_artifact: bool = True,
    execution_budget: Optional[CampaignExecutionBudget] = None,
) -> Dict[str, Any]:
    """Execute the complete retained grid and optionally write its JSON artifact."""

    started_at = time.perf_counter()
    config = validate_config(config)
    canonical_budget = _canonical_execution_budget(execution_budget)
    execution_preflight = _campaign_execution_preflight(
        config,
        canonical_budget,
    )
    if not execution_preflight["allowed"]:
        raise CampaignExecutionLimitError(
            "campaign refused by allocation-free preflight: "
            + "; ".join(execution_preflight["refusal_reasons"])
        )
    _check_campaign_deadline(
        started_at,
        canonical_budget,
        "after_preflight_before_construction",
    )
    stream_manifest = _stream_manifest(config)
    campaign_contract = _campaign_contract_manifest(config)
    raw_sanity_live_authority = _RawSanityLiveAuthority(
        campaign_contract["digest"]
    )
    if not stream_manifest["select_report_disjoint"]:
        raise CampaignValidationError("BANK/SELECT/REPORT streams are not disjoint")
    construction = _construction_checks(config)
    _check_campaign_deadline(
        started_at,
        canonical_budget,
        "after_construction_checks",
    )
    cell_specs = expected_cell_specs(config)
    cells_by_id: Dict[str, Dict[str, Any]] = {}
    raw_sanity_live_seals: Dict[str, _RawSanityLiveSeal] = {}
    for spec in cell_specs:
        _check_campaign_deadline(
            started_at,
            canonical_budget,
            "cell_manifest_initialization",
        )
        reason = _cell_unavailable_reason(spec)
        cells_by_id[_cell_id(spec)] = {
            "cell_id": _cell_id(spec),
            "config": dict(spec),
            "campaign_contract_digest": campaign_contract["digest"],
            "status": "unavailable" if reason else "pending",
            "unavailable_reason": reason,
            "promotion_eligible": False,
        }

    primary_components = []
    retained_row_count = 0
    base_groups = product(
        config.seeds,
        config.lengths,
        config.carrier_families,
        config.layers,
    )
    for seed, length, carrier_family, layers in base_groups:
        _check_campaign_deadline(
            started_at,
            canonical_budget,
            "base_group_start",
        )
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
            resident_context_bytes = _unique_numpy_nbytes(
                signatures,
                templates,
                base,
                ring_templates,
                payload_groups,
                payload_type_bank,
                reference_ffts,
                reference_norms,
                payload_ffts,
                payload_norms,
            )
            _check_campaign_deadline(
                started_at,
                canonical_budget,
                "base_group_fixed_arrays_ready",
            )
            for planted_mask_family in config.planted_mask_families:
                _check_campaign_deadline(
                    started_at,
                    canonical_budget,
                    "planted_mask_loop",
                )
                if (
                    _mask_availability(
                        layers, planted_mask_family, decoder=False
                    )
                    is not None
                ):
                    continue
                for bit_flip_rate in config.bit_flip_rates:
                    _check_campaign_deadline(
                        started_at,
                        canonical_budget,
                        "bit_flip_loop_before_base_result",
                    )
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
                        resident_context_bytes=resident_context_bytes,
                        deadline_check=lambda checkpoint: (
                            _check_campaign_deadline(
                                started_at,
                                canonical_budget,
                                checkpoint,
                            )
                        ),
                    )
                    retained_row_count += _base_result_retained_row_count(
                        base_result
                    )
                    _enforce_retained_row_cap(
                        retained_row_count,
                        canonical_budget,
                    )
                    _check_campaign_deadline(
                        started_at,
                        canonical_budget,
                        "bit_flip_loop_after_base_result",
                    )
                    for decoder_policy in config.decoder_mask_policies:
                        _check_campaign_deadline(
                            started_at,
                            canonical_budget,
                            "decoder_policy_loop",
                        )
                        if (
                            _mask_availability(
                                layers, decoder_policy, decoder=True
                            )
                            is not None
                        ):
                            continue
                        for payload_noise_rate in config.payload_noise_rates:
                            _check_campaign_deadline(
                                started_at,
                                canonical_budget,
                                "cell_loop_before_aggregation",
                            )
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
                                cell_id = _cell_id(spec)
                                completed_cell = _aggregate_cell(
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
                                cells_by_id[cell_id] = completed_cell
                                serialized_provenance = completed_cell.get(
                                    "report", {}
                                ).get("raw_type_accuracy_provenance")
                                if isinstance(
                                    serialized_provenance,
                                    Mapping,
                                ) and isinstance(
                                    serialized_provenance.get("rows"),
                                    list,
                                ):
                                    retained_row_count += len(
                                        serialized_provenance["rows"]
                                    )
                                    _enforce_retained_row_cap(
                                        retained_row_count,
                                        canonical_budget,
                                    )
                                live_seal = _raw_sanity_live_seal(
                                    config,
                                    spec,
                                    base_result["rows"][decoder_policy][
                                        "REPORT"
                                    ]["planted"],
                                    carrier_provenance,
                                    authority=raw_sanity_live_authority,
                                )
                                if live_seal is not None:
                                    raw_sanity_live_seals[cell_id] = live_seal
                                _check_campaign_deadline(
                                    started_at,
                                    canonical_budget,
                                    "cell_loop_after_aggregation",
                                )
                            except CampaignExecutionLimitError:
                                raise
                            except Exception as exc:  # retain the cell; never omit
                                cells_by_id[_cell_id(spec)] = {
                                    "cell_id": _cell_id(spec),
                                    "config": spec,
                                    "campaign_contract_digest": (
                                        campaign_contract["digest"]
                                    ),
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
        except CampaignExecutionLimitError:
            raise
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
        _check_campaign_deadline(
            started_at,
            canonical_budget,
            "pending_cell_finalization",
        )
        if cell["status"] == "pending":
            cell["status"] = "error"
            cell["error"] = {
                "type": "CampaignExecutionError",
                "message": "grid cell remained pending after campaign traversal",
            }
    cells = [cells_by_id[_cell_id(spec)] for spec in cell_specs]
    _check_campaign_deadline(
        started_at,
        canonical_budget,
        "before_primary_aggregation",
    )
    primary = _primary_aggregate(
        config,
        primary_components,
        raw_sanity_cells=cells,
        raw_sanity_live_seals=raw_sanity_live_seals,
        raw_sanity_live_authority=raw_sanity_live_authority,
    )
    _check_campaign_deadline(
        started_at,
        canonical_budget,
        "after_primary_aggregation",
    )
    verdicts = _campaign_verdicts(config, cells, construction, primary)
    status_counts: Dict[str, int] = {}
    for cell in cells:
        status_counts[cell["status"]] = status_counts.get(cell["status"], 0) + 1
    artifact = {
        "schema_version": "1.1.0",
        "record_type": RECORD_TYPE,
        "protocol_id": PROTOCOL_ID,
        "evidence_mode": EVIDENCE_MODE,
        "run_label": config.run_label,
        "scientific_claim_status": "unconfirmed",
        "novelty_claim_status": "unconfirmed",
        "promotion_eligible": False,
        "production_integration_permitted": False,
        "rader_claim": False,
        "rader_campaign_status": "NOT_TESTED_BY_THIS_CAMPAIGN",
        "rader_implementation_present": True,
        "separate_rader_harness_results_ingested": False,
        "config": asdict(config),
        "rng_streams": stream_manifest,
        "campaign_contract": campaign_contract,
        "execution_preflight": execution_preflight,
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
        "control_claim_governance": {
            "native_sparse_oppw_required": True,
            "equal_channel_use_repeated_bit_required": True,
            "full_pooled_control_gates_required": True,
            "dense_raw_sanity_required_bit_flip_rates": list(
                RAW_SANITY_REQUIRED_BIT_FLIP_RATES
            ),
            "dense_raw_sanity_required_cell_count": (
                len(FROZEN_SEEDS)
                * len(RAW_SANITY_REQUIRED_BIT_FLIP_RATES)
            ),
            "dense_raw_sanity_requires_exact_retained_cells": True,
            "dense_raw_sanity_lower_rates_may_be_inferred": False,
            "minimum_true_unlock_recall_for_control_closure": (
                config.control_min_true_unlock_recall
            ),
            "identical_noise_law_required_for_matched_noise_claim": True,
            "hardware_timing_required_for_speed_or_cost_claim": True,
            "matched_resource_advantage_claim_eligible": False,
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
                "native sparse OPPW is decoded from pulse positions without "
                "dense phase estimates, but its symbol-substitution channel is "
                "not identical to the dense bipolar chip-flip channel"
            ),
            (
                "equal-channel-use repeated-bit control represents at most "
                "2 types for L=1 or 16 types for L=8 and is fail-closed "
                "outside that declared codebook"
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
    _check_campaign_deadline(
        started_at,
        canonical_budget,
        "campaign_complete",
    )
    if write_artifact:
        output = Path(config.output)
        _atomic_write_json(
            output,
            artifact,
            deadline_check=lambda checkpoint: _check_campaign_deadline(
                started_at,
                canonical_budget,
                checkpoint,
            ),
            max_encoded_bytes=canonical_budget.max_estimated_bytes,
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
