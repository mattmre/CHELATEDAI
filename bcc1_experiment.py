"""Artifact-first BCC-1 bridge-choice experiment harness.

BCC-1 asks whether query-time, qrels-free observables can choose between a
reverse query bridge and a forward document bridge better than the best fixed
direction.  This module deliberately does not generate embeddings, rankings,
or evaluation data.  It accepts only a frozen, SHA-bound JSON pack whose
SELECT and REPORT blocks were assigned before analysis.

The anti-leakage boundary is structural:

* SELECT outcome scores may choose fixed baselines and train route policies.
* REPORT features may be passed to already-fitted policies.
* REPORT outcome scores are read only after all REPORT decisions are frozen.
* Features must have an explicit qrels-free, inference-available contract.
* A REPORT block is one-shot.  The file runner marks it consumed after writing
  the deterministic report and refuses to evaluate it again.

The implementation depends only on NumPy and the Python standard library and
is compatible with Python 3.9.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PACK_SCHEMA = "chelatedai.bcc1.pack.v1"
MANIFEST_SCHEMA = "chelatedai.bcc1.manifest.v1"
REPORT_SCHEMA = "chelatedai.bcc1.report.v1"
METRIC_BINDING_SCHEMA = "chelatedai.bcc1.metric-binding.v1"
OOF_RESIDUAL_AUDIT_SCHEMA = "chelatedai.bcc1.oof-residual-audit.v1"

VALIDATED_QUERY_ROLE_REVERSE_BRIDGE = "VALIDATED_QUERY_ROLE_REVERSE_BRIDGE"
UNVALIDATED_DOCUMENT_ROLE_REVERSE_PROXY = "UNVALIDATED_DOCUMENT_ROLE_REVERSE_PROXY"

BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 1729
BOOTSTRAP_CONFIDENCE = 0.95
POWER_REPLICATES = 10000
POWER_SEED = 1730
POWER_VARIANCE_INFLATION = 1.25
QUERY_FOLD_COUNT = 5
LOGISTIC_L2 = 1.0
LOGISTIC_MAX_ITERATIONS = 100
LOGISTIC_TOLERANCE = 1e-10
KNN_K = 5
ABSTAIN_CONFIDENCE = 0.80
TIE_TOLERANCE = 1e-12

_PACK_KEYS = {
    "schema_version",
    "pack_id",
    "evidence_mode",
    "sampled",
    "metric",
    "rows",
}
_METRIC_KEYS = {"name", "higher_is_better", "minimum", "maximum"}
_ROW_KEYS = {
    "query_id",
    "independence_group_id",
    "block_id",
    "scores",
    "features",
    "fusion_scores",
}
_BASE_REQUIRED_SCORE_KEYS = {"reverse", "forward", "native_new"}
_OPTIONAL_SCORE_KEYS = {"native_old", "mismatch"}
_MANIFEST_KEYS = {
    "schema_version",
    "pack_id",
    "evidence_mode",
    "sampled",
    "pack_sha256",
    "features",
    "blocks",
    "soft_fusion",
    "metric_contract",
    "reverse_route_role_validation",
}
_MANIFEST_ALLOWED_KEYS = _MANIFEST_KEYS | {"confirmatory_gates"}
_FEATURE_CONTRACT_KEYS = {
    "name",
    "qrels_free",
    "inference_available",
    "source",
    "implementation_sha256",
    "provenance_sha256",
    "availability_stage",
    "execution_cost_class",
}
_BLOCK_KEYS = {
    "block_id",
    "dataset_family_id",
    "transition_family_id",
    "role",
    "query_ids_sha256",
    "consumed",
    "consumption_report_sha256",
    "consumption_pack_sha256",
    "consumption_manifest_sha256",
    "consumption_output_identity_sha256",
    "consumption_transaction_id",
}
_SOFT_FUSION_KEYS = {
    "source_kind",
    "qrels_free_at_inference",
    "normalization",
    "candidates",
}
_FUSION_CANDIDATE_KEYS = {"id", "alpha"}
_CONFIRMATORY_GATE_KEYS = {
    "primary_method",
    "minimum_worthwhile_effect",
    "target_power",
    "one_sided_alpha",
    "minimum_report_queries",
    "minimum_report_blocks",
    "native_new_noninferiority_margin",
    "worst_block_noninferiority_margin",
    "minimum_overall_coverage",
    "minimum_block_coverage",
    "power_replicates",
    "power_seed",
    "power_variance_inflation",
}
_METRIC_CONTRACT_KEYS = {
    "metric_name",
    "implementation_sha256",
    "gain",
    "idcg_population",
    "cutoff",
    "query_inclusion",
    "qrels_sha256",
    "pack_qrels_binding_sha256",
    "ranking_tie_break",
}
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_FORBIDDEN_FEATURE_TOKENS = {
    "qrel",
    "qrels",
    "relevance",
    "relevant",
    "judgment",
    "judgement",
    "label",
    "labels",
    "winner",
    "target",
    "oracle",
    "headroom",
    "reward",
    "ndcg",
    "mrr",
    "map",
    "recall",
    "precision",
}
_FORBIDDEN_SOURCE_PHRASES = (
    "derived from qrel",
    "computed from qrel",
    "qrel count",
    "uses qrel",
    "relevance judgment",
    "relevance judgement",
    "route label",
    "direction label",
    "winning route",
    "oracle outcome",
    "report outcome",
)


class BCC1ValidationError(ValueError):
    """Raised when a BCC-1 artifact violates its frozen-data contract."""


@dataclass(frozen=True)
class ValidatedBCC1:
    """Validated, deterministically ordered BCC-1 inputs."""

    pack_id: str
    evidence_mode: str
    sampled: bool
    metric_name: str
    metric_minimum: float
    metric_maximum: float
    feature_names: Tuple[str, ...]
    dual_read_feature_names: Tuple[str, ...]
    blocks: Mapping[str, Mapping[str, Any]]
    select_rows: Tuple[Mapping[str, Any], ...]
    report_rows: Tuple[Mapping[str, Any], ...]
    method_dev_rows: Tuple[Mapping[str, Any], ...]
    score_names: Tuple[str, ...]
    metric_contract: Mapping[str, Any]
    reverse_route_role_validation: str
    soft_fusion: Optional[Mapping[str, Any]]
    confirmatory_gates: Optional[Mapping[str, Any]]
    pack_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class _LogisticModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    iterations: int


@dataclass(frozen=True)
class _KNNModel:
    mean: np.ndarray
    scale: np.ndarray
    x_train: np.ndarray
    y_train: np.ndarray
    query_ids: Tuple[str, ...]


def _reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BCC1ValidationError("duplicate JSON key: {!r}".format(key))
        result[key] = value
    return result


def decode_json_object(raw: bytes, *, artifact_name: str) -> Dict[str, Any]:
    """Decode UTF-8 JSON while rejecting duplicate keys and non-objects."""

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BCC1ValidationError("{} is not valid UTF-8: {}".format(artifact_name, exc)) from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_raise_nonfinite_json(token)),
        )
    except BCC1ValidationError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise BCC1ValidationError("{} is not valid JSON: {}".format(artifact_name, exc)) from exc
    if not isinstance(value, dict):
        raise BCC1ValidationError("{} root must be a JSON object".format(artifact_name))
    return value


def _raise_nonfinite_json(token: str) -> None:
    raise BCC1ValidationError("non-finite JSON number is forbidden: {}".format(token))


def canonical_json_bytes(value: Any, *, trailing_newline: bool = False) -> bytes:
    """Return the one canonical JSON encoding used for fingerprints and output."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BCC1ValidationError("value is not canonical-JSON encodable: {}".format(exc)) from exc
    if trailing_newline:
        encoded += b"\n"
    return encoded


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def query_ids_sha256(query_ids: Iterable[str]) -> str:
    """Fingerprint sorted query membership, independent of row ordering."""

    return sha256_bytes(canonical_json_bytes(sorted(query_ids)))


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: Iterable[str],
    allowed: Iterable[str],
    location: str,
) -> None:
    required_set = set(required)
    allowed_set = set(allowed)
    actual = set(value)
    missing = sorted(required_set - actual)
    extra = sorted(actual - allowed_set)
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing={}".format(missing))
        if extra:
            parts.append("undeclared={}".format(extra))
        raise BCC1ValidationError("{} has invalid fields ({})".format(location, ", ".join(parts)))


def _require_identifier(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise BCC1ValidationError("{} must be a non-empty stable identifier using [A-Za-z0-9._:/-]".format(location))
    return value


def _require_sha256(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not _HEX_64_RE.fullmatch(value):
        raise BCC1ValidationError("{} must be a lowercase SHA-256 hex digest".format(location))
    return value


def _require_finite_number(value: Any, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BCC1ValidationError("{} must be a finite JSON number".format(location))
    result = float(value)
    if not math.isfinite(result):
        raise BCC1ValidationError("{} must be finite".format(location))
    return result


def _require_score(
    value: Any,
    *,
    minimum: float,
    maximum: float,
    location: str,
) -> float:
    result = _require_finite_number(value, location=location)
    if result < minimum or result > maximum:
        raise BCC1ValidationError("{}={} is outside [{}, {}]".format(location, result, minimum, maximum))
    return result


def _feature_name_tokens(name: str) -> Tuple[str, ...]:
    return tuple(token for token in re.split(r"[^a-z0-9]+", name.lower()) if token)


def _validate_feature_contract(raw_features: Any, *, evidence_mode: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    if not isinstance(raw_features, list) or not raw_features:
        raise BCC1ValidationError("manifest.features must be a non-empty list")
    names: List[str] = []
    dual_read_names: List[str] = []
    for index, descriptor in enumerate(raw_features):
        location = "manifest.features[{}]".format(index)
        if not isinstance(descriptor, dict):
            raise BCC1ValidationError("{} must be an object".format(location))
        _require_exact_keys(
            descriptor,
            required=_FEATURE_CONTRACT_KEYS,
            allowed=_FEATURE_CONTRACT_KEYS,
            location=location,
        )
        name = _require_identifier(descriptor["name"], location=location + ".name")
        if name in names:
            raise BCC1ValidationError("duplicate feature name: {!r}".format(name))
        forbidden = sorted(set(_feature_name_tokens(name)) & _FORBIDDEN_FEATURE_TOKENS)
        if forbidden:
            raise BCC1ValidationError("feature {!r} has outcome/leakage token(s): {}".format(name, forbidden))
        if descriptor["qrels_free"] is not True:
            raise BCC1ValidationError("{} must declare qrels_free=true".format(location))
        if descriptor["inference_available"] is not True:
            raise BCC1ValidationError("{} must declare inference_available=true".format(location))
        source = descriptor["source"]
        if not isinstance(source, str) or not source.strip():
            raise BCC1ValidationError("{} requires a non-empty source".format(location))
        source_lower = source.lower()
        leak_phrases = [phrase for phrase in _FORBIDDEN_SOURCE_PHRASES if phrase in source_lower]
        if leak_phrases:
            raise BCC1ValidationError("{} source claims outcome-derived input: {}".format(location, leak_phrases))
        _require_sha256(
            descriptor["implementation_sha256"],
            location=location + ".implementation_sha256",
        )
        _require_sha256(
            descriptor["provenance_sha256"],
            location=location + ".provenance_sha256",
        )
        availability_stage = descriptor["availability_stage"]
        execution_cost_class = descriptor["execution_cost_class"]
        expected_costs = {
            "PRE_SEARCH": "zero_search",
            "DUAL_READ_METHOD_DEV": "dual_search_probe",
        }
        if availability_stage not in expected_costs:
            raise BCC1ValidationError(
                "{}.availability_stage must be PRE_SEARCH or " "DUAL_READ_METHOD_DEV".format(location)
            )
        if execution_cost_class != expected_costs[availability_stage]:
            raise BCC1ValidationError(
                "{}.execution_cost_class must be {!r} for {}".format(
                    location,
                    expected_costs[availability_stage],
                    availability_stage,
                )
            )
        if availability_stage == "DUAL_READ_METHOD_DEV":
            dual_read_names.append(name)
        names.append(name)
    if evidence_mode == "CONFIRMATORY" and dual_read_names:
        raise BCC1ValidationError(
            "CONFIRMATORY one-route policy forbids DUAL_READ_METHOD_DEV " "features: {}".format(sorted(dual_read_names))
        )
    return tuple(names), tuple(dual_read_names)


def _validate_soft_fusion(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise BCC1ValidationError("manifest.soft_fusion must be null or an object")
    _require_exact_keys(
        raw,
        required=_SOFT_FUSION_KEYS,
        allowed=_SOFT_FUSION_KEYS,
        location="manifest.soft_fusion",
    )
    if raw["source_kind"] != "precomputed_retrieval_score_fusion":
        raise BCC1ValidationError("soft fusion is justified only for precomputed retrieval-score fusion inputs")
    if raw["qrels_free_at_inference"] is not True:
        raise BCC1ValidationError("soft fusion must declare qrels_free_at_inference=true")
    normalization = raw["normalization"]
    if not isinstance(normalization, str) or not normalization.strip():
        raise BCC1ValidationError("soft fusion requires a declared score normalization")
    candidates = raw["candidates"]
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise BCC1ValidationError("soft fusion requires at least two frozen interior candidates")
    normalized_candidates: List[Dict[str, Any]] = []
    ids = set()
    alphas = set()
    for index, candidate in enumerate(candidates):
        location = "manifest.soft_fusion.candidates[{}]".format(index)
        if not isinstance(candidate, dict):
            raise BCC1ValidationError("{} must be an object".format(location))
        _require_exact_keys(
            candidate,
            required=_FUSION_CANDIDATE_KEYS,
            allowed=_FUSION_CANDIDATE_KEYS,
            location=location,
        )
        candidate_id = _require_identifier(candidate["id"], location=location + ".id")
        alpha = _require_finite_number(candidate["alpha"], location=location + ".alpha")
        if not 0.0 < alpha < 1.0:
            raise BCC1ValidationError("{} must be strictly between 0 and 1".format(location + ".alpha"))
        if candidate_id in ids:
            raise BCC1ValidationError("duplicate soft-fusion candidate id: {!r}".format(candidate_id))
        if alpha in alphas:
            raise BCC1ValidationError("duplicate soft-fusion alpha: {}".format(alpha))
        ids.add(candidate_id)
        alphas.add(alpha)
        normalized_candidates.append({"id": candidate_id, "alpha": alpha})
    normalized_candidates.sort(key=lambda item: (item["alpha"], item["id"]))
    return {
        "source_kind": raw["source_kind"],
        "qrels_free_at_inference": True,
        "normalization": normalization,
        "candidates": normalized_candidates,
    }


def _validate_metric(raw: Any) -> Tuple[str, float, float]:
    if not isinstance(raw, dict):
        raise BCC1ValidationError("pack.metric must be an object")
    _require_exact_keys(
        raw,
        required=_METRIC_KEYS,
        allowed=_METRIC_KEYS,
        location="pack.metric",
    )
    name = _require_identifier(raw["name"], location="pack.metric.name")
    if raw["higher_is_better"] is not True:
        raise BCC1ValidationError("BCC-1 v1 requires a higher-is-better metric")
    minimum = _require_finite_number(raw["minimum"], location="pack.metric.minimum")
    maximum = _require_finite_number(raw["maximum"], location="pack.metric.maximum")
    if not minimum < maximum:
        raise BCC1ValidationError("pack.metric.minimum must be less than maximum")
    return name, minimum, maximum


def _validate_metric_contract(
    raw: Any,
    *,
    metric_name: str,
    pack_sha256: str,
) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise BCC1ValidationError("manifest.metric_contract must be an object")
    _require_exact_keys(
        raw,
        required=_METRIC_CONTRACT_KEYS,
        allowed=_METRIC_CONTRACT_KEYS,
        location="manifest.metric_contract",
    )
    exact_values = {
        "metric_name": "ndcg_at_10",
        "gain": "binary_positive_qrel",
        "idcg_population": "all_positive_qrels_for_query",
        "cutoff": 10,
        "query_inclusion": "queries_with_at_least_one_positive_qrel",
        "ranking_tie_break": "score_desc_document_id_asc",
    }
    for field, expected in exact_values.items():
        if raw[field] != expected:
            raise BCC1ValidationError("manifest.metric_contract.{} must equal {!r}".format(field, expected))
    if metric_name != raw["metric_name"]:
        raise BCC1ValidationError("manifest.metric_contract.metric_name must match pack.metric.name")
    implementation_sha256 = _require_sha256(
        raw["implementation_sha256"],
        location="manifest.metric_contract.implementation_sha256",
    )
    qrels_sha256 = _require_sha256(
        raw["qrels_sha256"],
        location="manifest.metric_contract.qrels_sha256",
    )
    binding_sha256 = _require_sha256(
        raw["pack_qrels_binding_sha256"],
        location="manifest.metric_contract.pack_qrels_binding_sha256",
    )
    expected_binding = sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": METRIC_BINDING_SCHEMA,
                "pack_sha256": pack_sha256,
                "qrels_sha256": qrels_sha256,
            }
        )
    )
    if binding_sha256 != expected_binding:
        raise BCC1ValidationError(
            "manifest.metric_contract.pack_qrels_binding_sha256 does not bind " "the declared pack and qrels digests"
        )
    return {
        **exact_values,
        "implementation_sha256": implementation_sha256,
        "qrels_sha256": qrels_sha256,
        "pack_qrels_binding_sha256": binding_sha256,
    }


def _validate_reverse_route_role(
    raw: Any,
    *,
    evidence_mode: str,
) -> str:
    allowed = {
        VALIDATED_QUERY_ROLE_REVERSE_BRIDGE,
        UNVALIDATED_DOCUMENT_ROLE_REVERSE_PROXY,
    }
    if raw not in allowed:
        raise BCC1ValidationError(
            "manifest.reverse_route_role_validation must be one of: {}".format(", ".join(sorted(allowed)))
        )
    if evidence_mode == "CONFIRMATORY" and raw != VALIDATED_QUERY_ROLE_REVERSE_BRIDGE:
        raise BCC1ValidationError(
            "CONFIRMATORY requires a validated query-role reverse bridge; "
            "document-role reverse proxies are non-promotional"
        )
    return str(raw)


def _validate_confirmatory_gates(
    raw: Any,
    *,
    evidence_mode: str,
    metric_minimum: float,
    metric_maximum: float,
) -> Optional[Dict[str, Any]]:
    if evidence_mode == "METHOD_DEV":
        if raw is not None:
            raise BCC1ValidationError("METHOD_DEV manifests cannot declare confirmatory_gates")
        return None
    if not isinstance(raw, dict):
        raise BCC1ValidationError("CONFIRMATORY manifests must declare confirmatory_gates")
    _require_exact_keys(
        raw,
        required=_CONFIRMATORY_GATE_KEYS,
        allowed=_CONFIRMATORY_GATE_KEYS,
        location="manifest.confirmatory_gates",
    )
    primary_method = raw["primary_method"]
    if primary_method not in {"regularized_logistic", "knn"}:
        raise BCC1ValidationError("manifest.confirmatory_gates.primary_method must be " "regularized_logistic or knn")
    metric_span = metric_maximum - metric_minimum

    def bounded_nonnegative(name: str, *, strictly_positive: bool) -> float:
        value = _require_finite_number(raw[name], location="manifest.confirmatory_gates." + name)
        if strictly_positive:
            valid = 0.0 < value <= metric_span
            expected = "in (0, metric span]"
        else:
            valid = 0.0 <= value <= metric_span
            expected = "in [0, metric span]"
        if not valid:
            raise BCC1ValidationError("manifest.confirmatory_gates.{} must be {}".format(name, expected))
        return value

    minimum_worthwhile_effect = bounded_nonnegative("minimum_worthwhile_effect", strictly_positive=True)
    native_new_margin = bounded_nonnegative("native_new_noninferiority_margin", strictly_positive=False)
    worst_block_margin = bounded_nonnegative("worst_block_noninferiority_margin", strictly_positive=False)
    target_power = _require_finite_number(
        raw["target_power"],
        location="manifest.confirmatory_gates.target_power",
    )
    if not 0.5 < target_power < 1.0:
        raise BCC1ValidationError("manifest.confirmatory_gates.target_power must be in (0.5, 1)")
    one_sided_alpha = _require_finite_number(
        raw["one_sided_alpha"],
        location="manifest.confirmatory_gates.one_sided_alpha",
    )
    if not 0.0 < one_sided_alpha < 0.5:
        raise BCC1ValidationError("manifest.confirmatory_gates.one_sided_alpha must be in (0, 0.5)")
    coverage: Dict[str, float] = {}
    for name in ("minimum_overall_coverage", "minimum_block_coverage"):
        value = _require_finite_number(raw[name], location="manifest.confirmatory_gates." + name)
        if not 0.0 < value <= 1.0:
            raise BCC1ValidationError("manifest.confirmatory_gates.{} must be in (0, 1]".format(name))
        coverage[name] = value
    counts: Dict[str, int] = {}
    for name, lower_bound in (
        ("minimum_report_queries", 1),
        ("minimum_report_blocks", 2),
        ("power_replicates", 100),
        ("power_seed", 0),
    ):
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < lower_bound:
            raise BCC1ValidationError(
                "manifest.confirmatory_gates.{} must be an integer >= {}".format(name, lower_bound)
            )
        counts[name] = value
    power_variance_inflation = _require_finite_number(
        raw["power_variance_inflation"],
        location="manifest.confirmatory_gates.power_variance_inflation",
    )
    frozen_values = {
        "minimum_worthwhile_effect": (minimum_worthwhile_effect, 0.005),
        "target_power": (target_power, 0.80),
        "one_sided_alpha": (one_sided_alpha, 0.025),
        "native_new_noninferiority_margin": (native_new_margin, 0.02),
        "worst_block_noninferiority_margin": (worst_block_margin, 0.02),
        "minimum_overall_coverage": (
            coverage["minimum_overall_coverage"],
            0.10,
        ),
        "minimum_block_coverage": (
            coverage["minimum_block_coverage"],
            0.05,
        ),
        "power_variance_inflation": (
            power_variance_inflation,
            POWER_VARIANCE_INFLATION,
        ),
    }
    for name, (observed, expected) in frozen_values.items():
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15):
            raise BCC1ValidationError(
                "manifest.confirmatory_gates.{} must equal frozen BCC-1 v1 value {}".format(name, expected)
            )
    frozen_counts = {
        "minimum_report_queries": 2400,
        "minimum_report_blocks": 12,
        "power_replicates": POWER_REPLICATES,
        "power_seed": POWER_SEED,
    }
    for name, expected in frozen_counts.items():
        if counts[name] != expected:
            raise BCC1ValidationError(
                "manifest.confirmatory_gates.{} must equal frozen BCC-1 v1 value {}".format(name, expected)
            )
    return {
        "primary_method": primary_method,
        "minimum_worthwhile_effect": minimum_worthwhile_effect,
        "target_power": target_power,
        "one_sided_alpha": one_sided_alpha,
        "minimum_report_queries": counts["minimum_report_queries"],
        "minimum_report_blocks": counts["minimum_report_blocks"],
        "native_new_noninferiority_margin": native_new_margin,
        "worst_block_noninferiority_margin": worst_block_margin,
        "minimum_overall_coverage": coverage["minimum_overall_coverage"],
        "minimum_block_coverage": coverage["minimum_block_coverage"],
        "power_replicates": counts["power_replicates"],
        "power_seed": counts["power_seed"],
        "power_variance_inflation": power_variance_inflation,
    }


def _validate_blocks(
    raw_blocks: Any,
    *,
    evidence_mode: str,
    sampled: bool,
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise BCC1ValidationError("manifest.blocks must be a non-empty list")
    blocks: Dict[str, Dict[str, Any]] = {}
    roles = set()
    structural_cells: Dict[Tuple[str, str, str], str] = {}
    for index, block in enumerate(raw_blocks):
        location = "manifest.blocks[{}]".format(index)
        if not isinstance(block, dict):
            raise BCC1ValidationError("{} must be an object".format(location))
        _require_exact_keys(
            block,
            required={
                "block_id",
                "dataset_family_id",
                "transition_family_id",
                "role",
                "query_ids_sha256",
                "consumed",
            },
            allowed=_BLOCK_KEYS,
            location=location,
        )
        block_id = _require_identifier(block["block_id"], location=location + ".block_id")
        dataset_family_id = _require_identifier(block["dataset_family_id"], location=location + ".dataset_family_id")
        transition_family_id = _require_identifier(
            block["transition_family_id"],
            location=location + ".transition_family_id",
        )
        if block_id in blocks:
            raise BCC1ValidationError("duplicate block_id: {!r}".format(block_id))
        role = block["role"]
        if role not in {"SELECT", "REPORT", "METHOD_DEV"}:
            raise BCC1ValidationError("{} role must be SELECT, REPORT, or METHOD_DEV".format(location))
        structural_cell = (role, dataset_family_id, transition_family_id)
        prior_block_id = structural_cells.get(structural_cell)
        if prior_block_id is not None:
            raise BCC1ValidationError(
                "structural cell {} is declared by multiple block IDs: {!r}, {!r}".format(
                    structural_cell,
                    prior_block_id,
                    block_id,
                )
            )
        structural_cells[structural_cell] = block_id
        if not isinstance(block["consumed"], bool):
            raise BCC1ValidationError("{} consumed must be boolean".format(location))
        if sampled and role == "REPORT":
            raise BCC1ValidationError("sampled packs may not contain a REPORT block")
        if evidence_mode == "METHOD_DEV" and role != "METHOD_DEV":
            raise BCC1ValidationError("METHOD_DEV packs may contain only METHOD_DEV blocks, never SELECT/REPORT")
        if evidence_mode == "CONFIRMATORY" and role == "METHOD_DEV":
            raise BCC1ValidationError("CONFIRMATORY packs may contain only disjoint SELECT/REPORT blocks")
        if role == "METHOD_DEV" and block["consumed"]:
            raise BCC1ValidationError("METHOD_DEV blocks cannot be marked consumed")
        consumption_fields = {
            "consumption_report_sha256",
            "consumption_pack_sha256",
            "consumption_manifest_sha256",
            "consumption_output_identity_sha256",
            "consumption_transaction_id",
        }
        if role == "REPORT" and block["consumed"]:
            missing_consumption = sorted(consumption_fields - set(block))
            if missing_consumption:
                raise BCC1ValidationError(
                    "{} consumed REPORT is missing transaction fields {}".format(location, missing_consumption)
                )
            for field in consumption_fields:
                _require_sha256(block[field], location=location + "." + field)
            raise BCC1ValidationError("REPORT block {!r} was already consumed".format(block_id))
        present_consumption = sorted(consumption_fields & set(block))
        if role != "REPORT" and present_consumption:
            raise BCC1ValidationError("{} non-REPORT block cannot have consumption transaction fields".format(location))
        if role == "REPORT" and not block["consumed"] and present_consumption:
            raise BCC1ValidationError(
                "{} unconsumed REPORT cannot have consumption transaction fields".format(location)
            )
        _require_sha256(block["query_ids_sha256"], location=location + ".query_ids_sha256")
        normalized_block = dict(block)
        normalized_block["dataset_family_id"] = dataset_family_id
        normalized_block["transition_family_id"] = transition_family_id
        blocks[block_id] = normalized_block
        roles.add(role)
    if evidence_mode == "CONFIRMATORY" and roles != {"SELECT", "REPORT"}:
        raise BCC1ValidationError("CONFIRMATORY manifest.blocks must contain both SELECT and REPORT roles")
    if evidence_mode == "METHOD_DEV" and roles != {"METHOD_DEV"}:
        raise BCC1ValidationError("METHOD_DEV manifest.blocks must contain only METHOD_DEV roles")
    return blocks


def _mechanical_family_roles(
    family_ids: Iterable[str],
    *,
    family_kind: str,
) -> Dict[str, str]:
    if family_kind not in {"dataset", "transition"}:
        raise BCC1ValidationError("family_kind must be dataset or transition")
    normalized = tuple(sorted(set(family_ids)))
    ordered = sorted(
        normalized,
        key=lambda family_id: (
            hashlib.sha256(("bcc1-v1-{}-family|".format(family_kind) + family_id).encode("utf-8")).digest(),
            family_id,
        ),
    )
    return {family_id: ("SELECT" if rank % 2 == 0 else "REPORT") for rank, family_id in enumerate(ordered)}


def _validate_confirmatory_universe(
    *,
    blocks: Mapping[str, Mapping[str, Any]],
    select_rows: Sequence[Mapping[str, Any]],
    report_rows: Sequence[Mapping[str, Any]],
) -> None:
    role_rows = {"SELECT": select_rows, "REPORT": report_rows}
    role_cells: Dict[str, Dict[Tuple[str, str], str]] = {
        "SELECT": {},
        "REPORT": {},
    }
    for block_id, block in blocks.items():
        role = block["role"]
        if role not in role_cells:
            continue
        cell = (block["dataset_family_id"], block["transition_family_id"])
        role_cells[role][cell] = block_id

    datasets_by_role = {role: {dataset_id for dataset_id, _ in cells} for role, cells in role_cells.items()}
    transitions_by_role = {role: {transition_id for _, transition_id in cells} for role, cells in role_cells.items()}
    all_datasets = datasets_by_role["SELECT"] | datasets_by_role["REPORT"]
    all_transitions = transitions_by_role["SELECT"] | transitions_by_role["REPORT"]
    if len(all_datasets) < 8:
        raise BCC1ValidationError("CONFIRMATORY universe requires at least 8 distinct dataset families")
    if len(all_transitions) < 6:
        raise BCC1ValidationError("CONFIRMATORY universe requires at least 6 distinct transition families")
    dataset_overlap = datasets_by_role["SELECT"] & datasets_by_role["REPORT"]
    transition_overlap = transitions_by_role["SELECT"] & transitions_by_role["REPORT"]
    if dataset_overlap:
        raise BCC1ValidationError("SELECT and REPORT dataset-family axes overlap: {}".format(sorted(dataset_overlap)))
    if transition_overlap:
        raise BCC1ValidationError(
            "SELECT and REPORT transition-family axes overlap: {}".format(sorted(transition_overlap))
        )

    expected_dataset_roles = _mechanical_family_roles(
        all_datasets,
        family_kind="dataset",
    )
    expected_transition_roles = _mechanical_family_roles(
        all_transitions,
        family_kind="transition",
    )
    for role in ("SELECT", "REPORT"):
        expected_datasets = {
            family_id for family_id, expected_role in expected_dataset_roles.items() if expected_role == role
        }
        expected_transitions = {
            family_id for family_id, expected_role in expected_transition_roles.items() if expected_role == role
        }
        if datasets_by_role[role] != expected_datasets:
            raise BCC1ValidationError(
                "{} dataset-family role allocation differs from mechanical SHA parity".format(role)
            )
        if transitions_by_role[role] != expected_transitions:
            raise BCC1ValidationError(
                "{} transition-family role allocation differs from mechanical SHA parity".format(role)
            )
        expected_cells = {
            (dataset_id, transition_id) for dataset_id in expected_datasets for transition_id in expected_transitions
        }
        actual_cells = set(role_cells[role])
        if actual_cells != expected_cells:
            missing = sorted(expected_cells - actual_cells)
            extra = sorted(actual_cells - expected_cells)
            raise BCC1ValidationError(
                "{} cells are not the complete mechanical cross-product " "(missing_count={}, extra_count={})".format(
                    role,
                    len(missing),
                    len(extra),
                )
            )
        if len(actual_cells) < 12:
            raise BCC1ValidationError("{} requires at least 12 distinct dataset x transition cells".format(role))

        row_counts: Dict[Tuple[str, str], int] = {cell: 0 for cell in actual_cells}
        groups_by_cell: Dict[Tuple[str, str], set] = {cell: set() for cell in actual_cells}
        for row in role_rows[role]:
            block = blocks[row["block_id"]]
            cell = (
                block["dataset_family_id"],
                block["transition_family_id"],
            )
            row_counts[cell] += 1
            groups_by_cell[cell].add(row["independence_group_id"])
        undersized = {cell: count for cell, count in row_counts.items() if count < 200}
        if undersized:
            first = sorted(undersized.items())[:10]
            raise BCC1ValidationError(
                "{} requires at least 200 query rows per cell; first undersized={}".format(
                    role,
                    first,
                )
            )
        for dataset_id in sorted(expected_datasets):
            expected_groups: Optional[set] = None
            for transition_id in sorted(expected_transitions):
                observed_groups = groups_by_cell[(dataset_id, transition_id)]
                if expected_groups is None:
                    expected_groups = observed_groups
                elif observed_groups != expected_groups:
                    raise BCC1ValidationError(
                        "{} dataset family {!r} must expose identical "
                        "independence-group membership across transitions".format(
                            role,
                            dataset_id,
                        )
                    )


def _validate_rows(
    raw_rows: Any,
    *,
    feature_names: Tuple[str, ...],
    blocks: Mapping[str, Mapping[str, Any]],
    minimum: float,
    maximum: float,
    soft_fusion: Optional[Mapping[str, Any]],
    evidence_mode: str,
) -> Tuple[
    Tuple[Mapping[str, Any], ...],
    Tuple[Mapping[str, Any], ...],
    Tuple[Mapping[str, Any], ...],
    Tuple[str, ...],
]:
    if not isinstance(raw_rows, list) or not raw_rows:
        raise BCC1ValidationError("pack.rows must be a non-empty list")
    expected_features = set(feature_names)
    fusion_ids = tuple(candidate["id"] for candidate in soft_fusion["candidates"]) if soft_fusion is not None else ()
    seen_query_ids = set()
    seen_observation_keys = set()
    rows_by_block: Dict[str, List[Mapping[str, Any]]] = {block_id: [] for block_id in blocks}
    normalized_rows: List[Mapping[str, Any]] = []
    score_signature: Optional[Tuple[str, ...]] = None

    for index, raw_row in enumerate(raw_rows):
        location = "pack.rows[{}]".format(index)
        if not isinstance(raw_row, dict):
            raise BCC1ValidationError("{} must be an object".format(location))
        required_keys = {
            "query_id",
            "independence_group_id",
            "block_id",
            "scores",
            "features",
        }
        if soft_fusion is not None:
            required_keys.add("fusion_scores")
        _require_exact_keys(
            raw_row,
            required=required_keys,
            allowed=_ROW_KEYS,
            location=location,
        )
        query_id = _require_identifier(raw_row["query_id"], location=location + ".query_id")
        independence_group_id = _require_identifier(
            raw_row["independence_group_id"],
            location=location + ".independence_group_id",
        )
        block_id = _require_identifier(raw_row["block_id"], location=location + ".block_id")
        observation_key = (block_id, query_id)
        if observation_key in seen_observation_keys:
            raise BCC1ValidationError("query_id {!r} occurs more than once in block {!r}".format(query_id, block_id))
        seen_observation_keys.add(observation_key)
        if evidence_mode == "CONFIRMATORY" and query_id in seen_query_ids:
            raise BCC1ValidationError("query_id {!r} occurs in more than one row/block".format(query_id))
        seen_query_ids.add(query_id)
        if block_id not in blocks:
            raise BCC1ValidationError("{} references undeclared block {!r}".format(location, block_id))

        raw_scores = raw_row["scores"]
        if not isinstance(raw_scores, dict):
            raise BCC1ValidationError("{}.scores must be an object".format(location))
        required_score_keys = set(_BASE_REQUIRED_SCORE_KEYS)
        if evidence_mode == "METHOD_DEV":
            required_score_keys.add("mismatch")
        _require_exact_keys(
            raw_scores,
            required=required_score_keys,
            allowed=required_score_keys | _OPTIONAL_SCORE_KEYS,
            location=location + ".scores",
        )
        current_signature = tuple(sorted(raw_scores))
        if score_signature is None:
            score_signature = current_signature
        elif current_signature != score_signature:
            raise BCC1ValidationError(
                "all rows must expose the same optional score fields; expected {}, got {} at {}".format(
                    list(score_signature), list(current_signature), location
                )
            )
        scores = {
            name: _require_score(
                raw_scores[name],
                minimum=minimum,
                maximum=maximum,
                location="{}.scores.{}".format(location, name),
            )
            for name in current_signature
        }

        raw_feature_values = raw_row["features"]
        if not isinstance(raw_feature_values, dict):
            raise BCC1ValidationError("{}.features must be an object".format(location))
        if set(raw_feature_values) != expected_features:
            raise BCC1ValidationError(
                "{}.features must exactly match manifest names; missing={}, undeclared={}".format(
                    location,
                    sorted(expected_features - set(raw_feature_values)),
                    sorted(set(raw_feature_values) - expected_features),
                )
            )
        features = {
            name: _require_finite_number(
                raw_feature_values[name],
                location="{}.features.{}".format(location, name),
            )
            for name in feature_names
        }

        fusion_scores: Optional[Dict[str, float]] = None
        if soft_fusion is not None:
            raw_fusion_scores = raw_row["fusion_scores"]
            if not isinstance(raw_fusion_scores, dict):
                raise BCC1ValidationError("{}.fusion_scores must be an object".format(location))
            if set(raw_fusion_scores) != set(fusion_ids):
                raise BCC1ValidationError("{}.fusion_scores must exactly match frozen candidate ids".format(location))
            fusion_scores = {
                candidate_id: _require_score(
                    raw_fusion_scores[candidate_id],
                    minimum=minimum,
                    maximum=maximum,
                    location="{}.fusion_scores.{}".format(location, candidate_id),
                )
                for candidate_id in fusion_ids
            }
        elif "fusion_scores" in raw_row:
            raise BCC1ValidationError("{} has fusion_scores but manifest.soft_fusion is null".format(location))

        normalized = {
            "query_id": query_id,
            "independence_group_id": independence_group_id,
            "block_id": block_id,
            "scores": scores,
            "features": features,
        }
        if fusion_scores is not None:
            normalized["fusion_scores"] = fusion_scores
        normalized_rows.append(normalized)
        rows_by_block[block_id].append(normalized)

    for block_id, block_rows in rows_by_block.items():
        if not block_rows:
            raise BCC1ValidationError("declared block {!r} has no rows".format(block_id))
        actual_membership_hash = query_ids_sha256(row["query_id"] for row in block_rows)
        expected_membership_hash = blocks[block_id]["query_ids_sha256"]
        if actual_membership_hash != expected_membership_hash:
            raise BCC1ValidationError("block {!r} query membership hash mismatch".format(block_id))

    if score_signature is None:
        raise BCC1ValidationError("pack.rows unexpectedly contained no score signature")
    group_datasets: Dict[str, set] = {}
    group_roles: Dict[str, set] = {}
    seen_cell_groups: Dict[Tuple[str, str, str, str], str] = {}
    for row in normalized_rows:
        group_id = row["independence_group_id"]
        block = blocks[row["block_id"]]
        cell_group = (
            block["role"],
            block["dataset_family_id"],
            block["transition_family_id"],
            group_id,
        )
        prior_query_id = seen_cell_groups.get(cell_group)
        if prior_query_id is not None:
            raise BCC1ValidationError(
                "independence_group_id {!r} occurs more than once in structural "
                "cell {} (query_ids={!r}, {!r})".format(
                    group_id,
                    cell_group[:3],
                    prior_query_id,
                    row["query_id"],
                )
            )
        seen_cell_groups[cell_group] = row["query_id"]
        group_datasets.setdefault(group_id, set()).add(block["dataset_family_id"])
        group_roles.setdefault(group_id, set()).add(block["role"])
    for group_id, dataset_ids in group_datasets.items():
        if len(dataset_ids) != 1:
            raise BCC1ValidationError(
                "independence_group_id {!r} spans dataset families {}".format(group_id, sorted(dataset_ids))
            )
    if evidence_mode == "CONFIRMATORY":
        for group_id, roles in group_roles.items():
            if len(roles) != 1:
                raise BCC1ValidationError(
                    "independence_group_id {!r} leaks across SELECT/REPORT roles".format(group_id)
                )
    _reject_feature_value_leakage(
        normalized_rows,
        feature_names,
        fusion_ids,
        score_signature,
    )

    normalized_rows.sort(key=lambda row: (row["block_id"], row["query_id"]))
    select_rows = tuple(row for row in normalized_rows if blocks[row["block_id"]]["role"] == "SELECT")
    report_rows = tuple(row for row in normalized_rows if blocks[row["block_id"]]["role"] == "REPORT")
    method_dev_rows = tuple(row for row in normalized_rows if blocks[row["block_id"]]["role"] == "METHOD_DEV")
    if evidence_mode == "CONFIRMATORY" and (not select_rows or not report_rows):
        raise BCC1ValidationError("both SELECT and REPORT must contain at least one row")
    if evidence_mode == "METHOD_DEV" and not method_dev_rows:
        raise BCC1ValidationError("METHOD_DEV must contain at least one row")
    return select_rows, report_rows, method_dev_rows, score_signature


def _reject_feature_value_leakage(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    fusion_ids: Sequence[str],
    score_names: Sequence[str],
) -> None:
    """Reject exact copies of outcomes or derived direction labels.

    Provenance declarations are the primary barrier.  This content check
    catches common accidental exports such as putting NDCG or ``winner`` into
    a numerically renamed feature column.
    """

    outcome_columns: Dict[str, np.ndarray] = {}
    for score_name in score_names:
        outcome_columns[score_name] = np.asarray([row["scores"][score_name] for row in rows], dtype=np.float64)
    for candidate_id in fusion_ids:
        outcome_columns["fusion:{}".format(candidate_id)] = np.asarray(
            [row["fusion_scores"][candidate_id] for row in rows], dtype=np.float64
        )
    direction_labels = np.asarray(
        [1.0 if row["scores"]["reverse"] > row["scores"]["forward"] else 0.0 for row in rows],
        dtype=np.float64,
    )
    direction_has_two_classes = len(set(direction_labels.tolist())) == 2

    for feature_name in feature_names:
        feature = np.asarray([row["features"][feature_name] for row in rows], dtype=np.float64)
        for outcome_name, outcome in outcome_columns.items():
            if float(np.ptp(outcome)) > TIE_TOLERANCE and np.allclose(feature, outcome, rtol=0.0, atol=TIE_TOLERANCE):
                raise BCC1ValidationError(
                    "feature {!r} exactly copies outcome column {!r}".format(feature_name, outcome_name)
                )
        if direction_has_two_classes and (
            np.allclose(feature, direction_labels, rtol=0.0, atol=TIE_TOLERANCE)
            or np.allclose(feature, 1.0 - direction_labels, rtol=0.0, atol=TIE_TOLERANCE)
        ):
            raise BCC1ValidationError(
                "feature {!r} exactly encodes the reverse/forward route label".format(feature_name)
            )


def _validate_pack_and_manifest_common(
    pack: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    pack_raw_sha256: Optional[str] = None,
    manifest_raw_sha256: Optional[str] = None,
    enforce_confirmatory_universe: bool = True,
) -> ValidatedBCC1:
    """Validate and normalize a frozen BCC-1 pack/manifest pair."""

    if not isinstance(pack, dict) or not isinstance(manifest, dict):
        raise BCC1ValidationError("pack and manifest must both be JSON objects")
    _require_exact_keys(
        pack,
        required=_PACK_KEYS,
        allowed=_PACK_KEYS,
        location="pack",
    )
    _require_exact_keys(
        manifest,
        required=_MANIFEST_KEYS,
        allowed=_MANIFEST_ALLOWED_KEYS,
        location="manifest",
    )
    if pack["schema_version"] != PACK_SCHEMA:
        raise BCC1ValidationError("unsupported pack.schema_version: {!r}".format(pack["schema_version"]))
    if manifest["schema_version"] != MANIFEST_SCHEMA:
        raise BCC1ValidationError("unsupported manifest.schema_version: {!r}".format(manifest["schema_version"]))
    pack_id = _require_identifier(pack["pack_id"], location="pack.pack_id")
    manifest_pack_id = _require_identifier(manifest["pack_id"], location="manifest.pack_id")
    if pack_id != manifest_pack_id:
        raise BCC1ValidationError("pack_id mismatch between pack and manifest")
    evidence_mode = pack["evidence_mode"]
    if evidence_mode not in {"CONFIRMATORY", "METHOD_DEV"}:
        raise BCC1ValidationError("pack.evidence_mode must be CONFIRMATORY or METHOD_DEV")
    if manifest["evidence_mode"] != evidence_mode:
        raise BCC1ValidationError("evidence_mode mismatch between pack and manifest")
    sampled = pack["sampled"]
    if not isinstance(sampled, bool):
        raise BCC1ValidationError("pack.sampled must be boolean")
    if not isinstance(manifest["sampled"], bool):
        raise BCC1ValidationError("manifest.sampled must be boolean")
    if manifest["sampled"] != sampled:
        raise BCC1ValidationError("sampled mismatch between pack and manifest")
    if evidence_mode == "CONFIRMATORY" and sampled:
        raise BCC1ValidationError("sampled packs are non-promotional and cannot use CONFIRMATORY/REPORT mode")

    actual_pack_sha256 = pack_raw_sha256 or sha256_bytes(canonical_json_bytes(pack))
    _require_sha256(actual_pack_sha256, location="actual pack SHA-256")
    expected_pack_sha256 = _require_sha256(manifest["pack_sha256"], location="manifest.pack_sha256")
    if actual_pack_sha256 != expected_pack_sha256:
        raise BCC1ValidationError(
            "pack SHA-256 mismatch: expected {}, got {}".format(expected_pack_sha256, actual_pack_sha256)
        )
    actual_manifest_sha256 = manifest_raw_sha256 or sha256_bytes(canonical_json_bytes(manifest))
    _require_sha256(actual_manifest_sha256, location="actual manifest SHA-256")

    metric_name, metric_minimum, metric_maximum = _validate_metric(pack["metric"])
    metric_contract = _validate_metric_contract(
        manifest["metric_contract"],
        metric_name=metric_name,
        pack_sha256=actual_pack_sha256,
    )
    reverse_route_role_validation = _validate_reverse_route_role(
        manifest["reverse_route_role_validation"],
        evidence_mode=evidence_mode,
    )
    confirmatory_gates = _validate_confirmatory_gates(
        manifest.get("confirmatory_gates"),
        evidence_mode=evidence_mode,
        metric_minimum=metric_minimum,
        metric_maximum=metric_maximum,
    )
    feature_names, dual_read_feature_names = _validate_feature_contract(
        manifest["features"], evidence_mode=evidence_mode
    )
    blocks = _validate_blocks(
        manifest["blocks"],
        evidence_mode=evidence_mode,
        sampled=sampled,
    )
    soft_fusion = _validate_soft_fusion(manifest["soft_fusion"])
    if evidence_mode == "METHOD_DEV" and soft_fusion is not None:
        raise BCC1ValidationError("METHOD_DEV packs cannot select or evaluate soft-fusion candidates")
    select_rows, report_rows, method_dev_rows, score_names = _validate_rows(
        pack["rows"],
        feature_names=feature_names,
        blocks=blocks,
        minimum=metric_minimum,
        maximum=metric_maximum,
        soft_fusion=soft_fusion,
        evidence_mode=evidence_mode,
    )
    if evidence_mode == "CONFIRMATORY" and enforce_confirmatory_universe:
        _validate_confirmatory_universe(
            blocks=blocks,
            select_rows=select_rows,
            report_rows=report_rows,
        )
    return ValidatedBCC1(
        pack_id=pack_id,
        evidence_mode=evidence_mode,
        sampled=sampled,
        metric_name=metric_name,
        metric_minimum=metric_minimum,
        metric_maximum=metric_maximum,
        feature_names=feature_names,
        dual_read_feature_names=dual_read_feature_names,
        blocks=blocks,
        select_rows=select_rows,
        report_rows=report_rows,
        method_dev_rows=method_dev_rows,
        score_names=score_names,
        metric_contract=metric_contract,
        reverse_route_role_validation=reverse_route_role_validation,
        soft_fusion=soft_fusion,
        confirmatory_gates=confirmatory_gates,
        pack_sha256=actual_pack_sha256,
        manifest_sha256=actual_manifest_sha256,
    )


def validate_pack_and_manifest(
    pack: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    pack_raw_sha256: Optional[str] = None,
    manifest_raw_sha256: Optional[str] = None,
) -> ValidatedBCC1:
    """Public validator with unconditional confirmatory-universe enforcement."""

    return _validate_pack_and_manifest_common(
        pack,
        manifest,
        pack_raw_sha256=pack_raw_sha256,
        manifest_raw_sha256=manifest_raw_sha256,
        enforce_confirmatory_universe=True,
    )


def _round_float(value: float) -> float:
    result = round(float(value), 12)
    if result == 0.0:
        return 0.0
    return result


def _mean(values: np.ndarray) -> float:
    return _round_float(float(np.mean(values)))


def paired_bootstrap_mean_ci(
    candidate: Sequence[float],
    comparator: Sequence[float],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> Dict[str, Any]:
    """Paired percentile-bootstrap CI for a mean candidate-comparator delta."""

    candidate_array = np.asarray(candidate, dtype=np.float64)
    comparator_array = np.asarray(comparator, dtype=np.float64)
    if candidate_array.ndim != 1 or comparator_array.ndim != 1:
        raise BCC1ValidationError("bootstrap inputs must be one-dimensional")
    if candidate_array.shape != comparator_array.shape or candidate_array.size == 0:
        raise BCC1ValidationError("bootstrap inputs must be non-empty paired vectors")
    if not np.all(np.isfinite(candidate_array)) or not np.all(np.isfinite(comparator_array)):
        raise BCC1ValidationError("bootstrap inputs must be finite")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 100:
        raise BCC1ValidationError("bootstrap replicates must be an integer >= 100")
    if not 0.0 < confidence < 1.0:
        raise BCC1ValidationError("bootstrap confidence must be between 0 and 1")

    deltas = candidate_array - comparator_array
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    n = deltas.size
    for index in range(replicates):
        sample_indices = rng.integers(0, n, size=n)
        means[index] = float(np.mean(deltas[sample_indices]))
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return {
        "confidence": _round_float(confidence),
        "replicates": replicates,
        "seed": seed,
        "mean_difference": _round_float(float(np.mean(deltas))),
        "ci_lower": _round_float(float(lower)),
        "ci_upper": _round_float(float(upper)),
    }


def hierarchical_paired_bootstrap_mean_ci(
    candidate: Sequence[float],
    comparator: Sequence[float],
    block_ids: Sequence[str],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> Dict[str, Any]:
    """Block-then-query paired bootstrap for an equal-structural-block estimand."""

    candidate_array = np.asarray(candidate, dtype=np.float64)
    comparator_array = np.asarray(comparator, dtype=np.float64)
    if candidate_array.ndim != 1 or comparator_array.ndim != 1:
        raise BCC1ValidationError("hierarchical bootstrap inputs must be one-dimensional")
    if candidate_array.shape != comparator_array.shape or candidate_array.size == 0:
        raise BCC1ValidationError("hierarchical bootstrap inputs must be non-empty paired vectors")
    if len(block_ids) != candidate_array.size:
        raise BCC1ValidationError("hierarchical bootstrap block_ids must align with paired vectors")
    if not np.all(np.isfinite(candidate_array)) or not np.all(np.isfinite(comparator_array)):
        raise BCC1ValidationError("hierarchical bootstrap inputs must be finite")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 100:
        raise BCC1ValidationError("hierarchical bootstrap replicates must be an integer >= 100")
    if not 0.0 < confidence < 1.0:
        raise BCC1ValidationError("hierarchical bootstrap confidence must be between 0 and 1")
    normalized_blocks = []
    for index, block_id in enumerate(block_ids):
        normalized_blocks.append(
            _require_identifier(block_id, location="hierarchical bootstrap block_ids[{}]".format(index))
        )
    deltas = candidate_array - comparator_array
    unique_blocks = tuple(sorted(set(normalized_blocks)))
    indices_by_block = tuple(
        np.asarray(
            [index for index, observed_block in enumerate(normalized_blocks) if observed_block == block_id],
            dtype=np.int64,
        )
        for block_id in unique_blocks
    )
    observed_block_means = np.asarray(
        [float(np.mean(deltas[indices])) for indices in indices_by_block],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    block_count = len(indices_by_block)
    for replicate_index in range(replicates):
        sampled_blocks = rng.integers(0, block_count, size=block_count)
        sampled_means = np.empty(block_count, dtype=np.float64)
        for output_index, sampled_block in enumerate(sampled_blocks):
            source_indices = indices_by_block[int(sampled_block)]
            query_sample = rng.integers(0, source_indices.size, size=source_indices.size)
            sampled_means[output_index] = float(np.mean(deltas[source_indices[query_sample]]))
        means[replicate_index] = float(np.mean(sampled_means))
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return {
        "estimand": "equal-weighted mean of declared structural block means",
        "resampling": "sample blocks with replacement, then paired queries within each sampled block",
        "confidence": _round_float(confidence),
        "replicates": replicates,
        "seed": seed,
        "block_count": block_count,
        "query_count": int(candidate_array.size),
        "mean_difference": _round_float(float(np.mean(observed_block_means))),
        "ci_lower": _round_float(float(lower)),
        "ci_upper": _round_float(float(upper)),
    }


def grouped_paired_bootstrap_mean_ci(
    candidate: Sequence[float],
    comparator: Sequence[float],
    independence_group_ids: Sequence[str],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> Dict[str, Any]:
    """Paired cluster bootstrap that never splits an independence group."""

    candidate_array = np.asarray(candidate, dtype=np.float64)
    comparator_array = np.asarray(comparator, dtype=np.float64)
    if candidate_array.shape != comparator_array.shape or candidate_array.ndim != 1:
        raise BCC1ValidationError("grouped bootstrap inputs must be paired vectors")
    if candidate_array.size == 0 or len(independence_group_ids) != candidate_array.size:
        raise BCC1ValidationError("grouped bootstrap requires non-empty aligned independence groups")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 100:
        raise BCC1ValidationError("grouped bootstrap replicates must be an integer >= 100")
    if not 0.0 < confidence < 1.0:
        raise BCC1ValidationError("grouped bootstrap confidence must be between 0 and 1")
    normalized_groups = tuple(
        _require_identifier(value, location="grouped bootstrap independence_group_ids[{}]".format(index))
        for index, value in enumerate(independence_group_ids)
    )
    deltas = candidate_array - comparator_array
    unique_groups = tuple(sorted(set(normalized_groups)))
    group_means = np.asarray(
        [
            float(
                np.mean(
                    deltas[
                        np.asarray(
                            [index for index, observed in enumerate(normalized_groups) if observed == group_id],
                            dtype=np.int64,
                        )
                    ]
                )
            )
            for group_id in unique_groups
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, group_means.size, size=(replicates, group_means.size))
    bootstrap_means = np.mean(group_means[sampled], axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_means, [tail, 1.0 - tail])
    return {
        "estimand": "equal-weighted independence-group mean",
        "confidence": _round_float(confidence),
        "replicates": replicates,
        "seed": seed,
        "independence_group_count": len(unique_groups),
        "observation_count": int(candidate_array.size),
        "mean_difference": _round_float(float(np.mean(group_means))),
        "ci_lower": _round_float(float(lower)),
        "ci_upper": _round_float(float(upper)),
    }


def crossed_hierarchical_paired_bootstrap_mean_ci(
    candidate: Sequence[float],
    comparator: Sequence[float],
    rows: Sequence[Mapping[str, Any]],
    blocks: Mapping[str, Mapping[str, Any]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> Dict[str, Any]:
    """Cross dataset and transition axes while clustering repeated queries."""

    candidate_array = np.asarray(candidate, dtype=np.float64)
    comparator_array = np.asarray(comparator, dtype=np.float64)
    if (
        candidate_array.shape != comparator_array.shape
        or candidate_array.ndim != 1
        or candidate_array.size != len(rows)
        or candidate_array.size == 0
    ):
        raise BCC1ValidationError("crossed bootstrap requires non-empty aligned paired rows")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 100:
        raise BCC1ValidationError("crossed bootstrap replicates must be an integer >= 100")
    if not 0.0 < confidence < 1.0:
        raise BCC1ValidationError("crossed bootstrap confidence must be between 0 and 1")
    deltas = candidate_array - comparator_array
    cell_group_values: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    dataset_groups: Dict[str, set] = {}
    for index, row in enumerate(rows):
        block = blocks[row["block_id"]]
        dataset_id = block["dataset_family_id"]
        transition_id = block["transition_family_id"]
        group_id = row["independence_group_id"]
        cell_group_values.setdefault((dataset_id, transition_id), {}).setdefault(group_id, []).append(
            float(deltas[index])
        )
        dataset_groups.setdefault(dataset_id, set()).add(group_id)
    datasets = tuple(sorted(dataset_groups))
    transitions = tuple(sorted({transition for _, transition in cell_group_values}))
    missing_cells = [
        (dataset_id, transition_id)
        for dataset_id in datasets
        for transition_id in transitions
        if (dataset_id, transition_id) not in cell_group_values
    ]
    if missing_cells:
        raise BCC1ValidationError(
            "crossed bootstrap requires a complete dataset x transition grid; "
            "missing_count={}, first_missing={}".format(len(missing_cells), missing_cells[:10])
        )
    repeated_cell_groups = [
        (cell, group_id, len(values))
        for cell, grouped_values in cell_group_values.items()
        for group_id, values in grouped_values.items()
        if len(values) != 1
    ]
    if repeated_cell_groups:
        raise BCC1ValidationError(
            "crossed bootstrap requires exactly one query row per "
            "independence group in each structural cell; first={}".format(repeated_cell_groups[:10])
        )
    collapsed_cells: Dict[Tuple[str, str], Dict[str, float]] = {
        cell: {group_id: float(np.mean(values)) for group_id, values in grouped_values.items()}
        for cell, grouped_values in cell_group_values.items()
    }
    for dataset_id in datasets:
        expected_groups = set(dataset_groups[dataset_id])
        for transition_id in transitions:
            observed_groups = set(collapsed_cells[(dataset_id, transition_id)])
            if observed_groups != expected_groups:
                raise BCC1ValidationError(
                    "dataset family {!r} has inconsistent independence groups " "across transition {!r}".format(
                        dataset_id, transition_id
                    )
                )
    observed_cell_means = np.asarray(
        [
            float(np.mean(tuple(collapsed_cells[(dataset_id, transition_id)].values())))
            for dataset_id in datasets
            for transition_id in transitions
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(replicates, dtype=np.float64)
    for replicate_index in range(replicates):
        sampled_datasets = rng.integers(0, len(datasets), size=len(datasets))
        sampled_transitions = rng.integers(0, len(transitions), size=len(transitions))
        replicate_cells = []
        for dataset_index in sampled_datasets:
            dataset_id = datasets[int(dataset_index)]
            groups = tuple(sorted(dataset_groups[dataset_id]))
            sampled_groups = rng.integers(0, len(groups), size=len(groups))
            sampled_group_ids = tuple(groups[int(index)] for index in sampled_groups)
            for transition_index in sampled_transitions:
                transition_id = transitions[int(transition_index)]
                values_by_group = collapsed_cells[(dataset_id, transition_id)]
                replicate_cells.append(float(np.mean([values_by_group[group_id] for group_id in sampled_group_ids])))
        bootstrap_means[replicate_index] = float(np.mean(replicate_cells))
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_means, [tail, 1.0 - tail])
    return {
        "estimand": "equal-cell macro over crossed dataset and transition families",
        "resampling": (
            "sample dataset families and transition families independently, "
            "then sample coupled independence groups within dataset"
        ),
        "confidence": _round_float(confidence),
        "one_sided_lower_confidence": _round_float(1.0 - tail),
        "replicates": replicates,
        "seed": seed,
        "dataset_family_count": len(datasets),
        "transition_family_count": len(transitions),
        "cell_count": len(observed_cell_means),
        "independence_group_count": sum(len(values) for values in dataset_groups.values()),
        "observation_count": int(candidate_array.size),
        "mean_difference": _round_float(float(np.mean(observed_cell_means))),
        "ci_lower": _round_float(float(lower)),
        "ci_upper": _round_float(float(upper)),
    }


def _select_power_gate(
    *,
    policy_name: str,
    structural_scheme: Mapping[str, Any],
    report_rows: Sequence[Mapping[str, Any]],
    blocks: Mapping[str, Mapping[str, Any]],
    gates: Mapping[str, Any],
) -> Dict[str, Any]:
    """Use frozen OOF SELECT residuals to simulate the declared REPORT shape."""

    def blocked(reason: str, **audit: Any) -> Dict[str, Any]:
        return {
            "status": "blocked",
            "reason": reason,
            "estimated_power": None,
            "power_pass": False,
            "pass": False,
            **audit,
        }

    if structural_scheme.get("axis") != "transition_family_id":
        return blocked("power requires the preregistered crossed transition-family x query-fold OOF scheme")
    raw_oof_audit = structural_scheme.get("oof_residual_audit")
    if not isinstance(raw_oof_audit, list) or not raw_oof_audit:
        return blocked("power requires a non-empty frozen OOF residual audit")
    expected_oof_sha256 = structural_scheme.get("oof_residual_audit_sha256")
    if not isinstance(expected_oof_sha256, str):
        return blocked("power requires an OOF residual audit SHA-256")
    computed_oof_sha256 = sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": OOF_RESIDUAL_AUDIT_SCHEMA,
                "policy_name": policy_name,
                "axis": structural_scheme["axis"],
                "rows": raw_oof_audit,
            }
        )
    )
    if expected_oof_sha256 != computed_oof_sha256:
        return blocked("OOF residual audit SHA-256 mismatch")
    if not structural_scheme.get("every_row_evaluated_exactly_once", False):
        return blocked("OOF residual audit does not evaluate every SELECT row exactly once")
    invalid_oof = [
        item for item in raw_oof_audit if not isinstance(item, Mapping) or item.get("partition_valid") is not True
    ]
    if invalid_oof:
        return blocked(
            "OOF residual audit contains invalid structural partitions",
            invalid_oof_row_count=len(invalid_oof),
            oof_residual_audit_sha256=computed_oof_sha256,
        )

    report_cell_groups: Dict[Tuple[str, str], Dict[str, int]] = {}
    report_cell_blocks: Dict[Tuple[str, str], set] = {}
    report_dataset_groups: Dict[str, set] = {}
    for row in report_rows:
        block = blocks[row["block_id"]]
        if block["role"] != "REPORT":
            return blocked("power REPORT design contains a non-REPORT row")
        cell = (
            block["dataset_family_id"],
            block["transition_family_id"],
        )
        group_id = row["independence_group_id"]
        grouped = report_cell_groups.setdefault(cell, {})
        grouped[group_id] = grouped.get(group_id, 0) + 1
        report_cell_blocks.setdefault(cell, set()).add(row["block_id"])
        report_dataset_groups.setdefault(cell[0], set()).add(group_id)
    report_datasets = tuple(sorted(report_dataset_groups))
    report_transitions = tuple(sorted({transition_id for _, transition_id in report_cell_groups}))
    if not report_datasets or not report_transitions:
        return blocked("declared REPORT design is empty")
    missing_report_cells = [
        (dataset_id, transition_id)
        for dataset_id in report_datasets
        for transition_id in report_transitions
        if (dataset_id, transition_id) not in report_cell_groups
    ]
    if missing_report_cells:
        return blocked(
            "declared REPORT allocation is not a complete dataset x transition grid",
            missing_report_cell_count=len(missing_report_cells),
            first_missing_report_cells=missing_report_cells[:10],
        )
    inconsistent_report_cells = [
        (dataset_id, transition_id)
        for dataset_id in report_datasets
        for transition_id in report_transitions
        if set(report_cell_groups[(dataset_id, transition_id)]) != report_dataset_groups[dataset_id]
    ]
    if inconsistent_report_cells:
        return blocked(
            "declared REPORT cells do not contain identical eligible groups across transitions within each dataset",
            inconsistent_report_cell_count=len(inconsistent_report_cells),
            first_inconsistent_report_cells=inconsistent_report_cells[:10],
        )
    report_block_ids = {row["block_id"] for row in report_rows}
    report_query_count = len(report_rows)
    report_block_count = len(report_block_ids)
    query_count_pass = report_query_count >= gates["minimum_report_queries"]
    block_count_pass = report_block_count >= gates["minimum_report_blocks"]
    report_cell_design = [
        {
            "dataset_family_id": dataset_id,
            "transition_family_id": transition_id,
            "block_count": len(report_cell_blocks[(dataset_id, transition_id)]),
            "independence_group_count": len(report_cell_groups[(dataset_id, transition_id)]),
            "query_count": sum(report_cell_groups[(dataset_id, transition_id)].values()),
            "group_query_counts": [
                {
                    "independence_group_id": group_id,
                    "query_count": count,
                }
                for group_id, count in sorted(report_cell_groups[(dataset_id, transition_id)].items())
            ],
            "group_query_multiplicities": sorted(report_cell_groups[(dataset_id, transition_id)].values()),
        }
        for dataset_id in report_datasets
        for transition_id in report_transitions
    ]
    report_design = {
        "dataset_family_count": len(report_datasets),
        "transition_family_count": len(report_transitions),
        "cell_count": len(report_cell_design),
        "block_count": report_block_count,
        "query_count": report_query_count,
        "cells": report_cell_design,
    }
    if not query_count_pass or not block_count_pass:
        return blocked(
            "declared REPORT design does not meet frozen minimum query/block counts",
            report_design=report_design,
            known_report_query_count=report_query_count,
            required_report_query_count=gates["minimum_report_queries"],
            query_count_pass=query_count_pass,
            known_report_block_count=report_block_count,
            required_report_block_count=gates["minimum_report_blocks"],
            block_count_pass=block_count_pass,
            oof_residual_audit_sha256=computed_oof_sha256,
        )

    source_raw: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    source_dataset_groups: Dict[str, set] = {}
    for item in raw_oof_audit:
        try:
            cell = (
                str(item["dataset_family_id"]),
                str(item["transition_family_id"]),
            )
            group_id = str(item["independence_group_id"])
            residual = float(item["residual"])
        except (KeyError, TypeError, ValueError) as exc:
            return blocked(
                "OOF residual audit has an invalid row: {}".format(exc),
                oof_residual_audit_sha256=computed_oof_sha256,
            )
        if not math.isfinite(residual):
            return blocked(
                "OOF residual audit contains a non-finite residual",
                oof_residual_audit_sha256=computed_oof_sha256,
            )
        source_raw.setdefault(cell, {}).setdefault(group_id, []).append(residual)
        source_dataset_groups.setdefault(cell[0], set()).add(group_id)
    source_datasets = tuple(sorted(source_dataset_groups))
    source_transitions = tuple(sorted({transition_id for _, transition_id in source_raw}))
    missing_source_cells = [
        (dataset_id, transition_id)
        for dataset_id in source_datasets
        for transition_id in source_transitions
        if (dataset_id, transition_id) not in source_raw
    ]
    inconsistent_source_cells = [
        (dataset_id, transition_id)
        for dataset_id in source_datasets
        for transition_id in source_transitions
        if (dataset_id, transition_id) in source_raw
        and set(source_raw[(dataset_id, transition_id)]) != source_dataset_groups[dataset_id]
    ]
    if missing_source_cells or inconsistent_source_cells:
        return blocked(
            "OOF SELECT variance library is not a complete crossed grid with coupled groups",
            missing_source_cell_count=len(missing_source_cells),
            inconsistent_source_cell_count=len(inconsistent_source_cells),
            oof_residual_audit_sha256=computed_oof_sha256,
        )
    source_centered: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    for cell, grouped in source_raw.items():
        cell_values = [value for values in grouped.values() for value in values]
        if not cell_values:
            return blocked(
                "OOF SELECT variance library contains an empty source cell",
                oof_residual_audit_sha256=computed_oof_sha256,
            )
        cell_mean = float(np.mean(cell_values))
        source_centered[cell] = {
            group_id: (np.asarray(values, dtype=np.float64) - cell_mean) * gates["power_variance_inflation"]
            for group_id, values in grouped.items()
        }

    rng = np.random.default_rng(gates["power_seed"])
    effects = np.empty(gates["power_replicates"], dtype=np.float64)
    for replicate_index in range(gates["power_replicates"]):
        sampled_dataset_indices = rng.integers(
            0,
            len(source_datasets),
            size=len(report_datasets),
        )
        sampled_transition_indices = rng.integers(
            0,
            len(source_transitions),
            size=len(report_transitions),
        )
        replicate_cells = []
        for report_dataset_index, report_dataset_id in enumerate(report_datasets):
            source_dataset_id = source_datasets[int(sampled_dataset_indices[report_dataset_index])]
            target_group_ids = tuple(sorted(report_dataset_groups[report_dataset_id]))
            source_group_ids = tuple(sorted(source_dataset_groups[source_dataset_id]))
            sampled_group_indices = rng.integers(
                0,
                len(source_group_ids),
                size=len(target_group_ids),
            )
            target_to_source_group = {
                target_group_id: source_group_ids[int(source_group_index)]
                for target_group_id, source_group_index in zip(
                    target_group_ids,
                    sampled_group_indices,
                )
            }
            for report_transition_index, report_transition_id in enumerate(report_transitions):
                source_transition_id = source_transitions[int(sampled_transition_indices[report_transition_index])]
                source_groups = source_centered[(source_dataset_id, source_transition_id)]
                target_group_means = []
                target_counts = report_cell_groups[(report_dataset_id, report_transition_id)]
                for target_group_id in target_group_ids:
                    multiplicity = target_counts[target_group_id]
                    source_group_id = target_to_source_group[target_group_id]
                    source_values = source_groups[source_group_id]
                    sampled_value_indices = rng.integers(
                        0,
                        len(source_values),
                        size=multiplicity,
                    )
                    target_group_means.append(float(np.mean(source_values[sampled_value_indices])))
                replicate_cells.append(float(np.mean(target_group_means)))
        effects[replicate_index] = float(np.mean(replicate_cells))
    null_critical = float(np.quantile(effects, 0.975))
    estimated_power = float(np.mean(effects + gates["minimum_worthwhile_effect"] > null_critical))
    power_pass = estimated_power + TIE_TOLERANCE >= gates["target_power"]
    return {
        "status": "evaluated",
        "source": (
            "frozen crossed transition-family x query-fold OOF SELECT residuals; "
            "computed before REPORT outcomes are read"
        ),
        "policy_name": policy_name,
        "oof_structural_scheme": "crossed_transition_family_x_query_fold",
        "oof_residual_audit_sha256": computed_oof_sha256,
        "oof_residual_count": len(raw_oof_audit),
        "resampling": (
            "simulate the exact declared REPORT dataset/transition grid and "
            "per-cell group/query multiplicities by resampling centered OOF "
            "SELECT residual cells across corresponding axes; reuse one "
            "target-group to source-group mapping across all transitions "
            "within each sampled dataset"
        ),
        "source_variance_library": {
            "dataset_family_count": len(source_datasets),
            "transition_family_count": len(source_transitions),
            "cell_count": len(source_centered),
        },
        "report_design": report_design,
        "replicates": gates["power_replicates"],
        "seed": gates["power_seed"],
        "variance_inflation": _round_float(gates["power_variance_inflation"]),
        "minimum_worthwhile_effect": _round_float(gates["minimum_worthwhile_effect"]),
        "null_critical_97_5_percentile": _round_float(null_critical),
        "null_effect_standard_deviation": _round_float(float(np.std(effects))),
        "estimated_power": _round_float(estimated_power),
        "required_power": _round_float(gates["target_power"]),
        "power_pass": power_pass,
        "known_report_query_count": report_query_count,
        "required_report_query_count": gates["minimum_report_queries"],
        "query_count_pass": query_count_pass,
        "known_report_block_count": report_block_count,
        "required_report_block_count": gates["minimum_report_blocks"],
        "block_count_pass": block_count_pass,
        "pass": power_pass and query_count_pass and block_count_pass,
    }


def _matrix(rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [[row["features"][name] for name in feature_names] for row in rows],
        dtype=np.float64,
    )


def _direction_training_data(
    rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...], int]:
    x_rows = []
    labels = []
    query_ids = []
    ties = 0
    for row in rows:
        reverse = row["scores"]["reverse"]
        forward = row["scores"]["forward"]
        if abs(reverse - forward) <= TIE_TOLERANCE:
            ties += 1
            continue
        x_rows.append([row["features"][name] for name in feature_names])
        labels.append(1.0 if reverse > forward else 0.0)
        query_ids.append(row["query_id"])
    if x_rows:
        x = np.asarray(x_rows, dtype=np.float64)
    else:
        x = np.empty((0, len(feature_names)), dtype=np.float64)
    return x, np.asarray(labels, dtype=np.float64), tuple(query_ids), ties


def _standardizer(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale <= TIE_TOLERANCE, 1.0, scale)
    return mean, scale


def _fit_logistic(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[_LogisticModel], str]:
    class_counts = {
        0: int(np.sum(y == 0.0)),
        1: int(np.sum(y == 1.0)),
    }
    if x.shape[0] < 4 or min(class_counts.values()) < 2:
        return None, "requires at least two non-tied SELECT examples per direction"
    mean, scale = _standardizer(x)
    standardized = (x - mean) / scale
    design = np.column_stack([np.ones(standardized.shape[0]), standardized])
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    regularizer = np.eye(design.shape[1], dtype=np.float64)
    regularizer[0, 0] = 0.0
    iterations = 0
    converged = False
    try:
        for iterations in range(1, LOGISTIC_MAX_ITERATIONS + 1):
            logits = np.clip(design @ coefficients, -40.0, 40.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            gradient = (design.T @ (probabilities - y)) / y.size
            gradient += LOGISTIC_L2 * (regularizer @ coefficients)
            weights = probabilities * (1.0 - probabilities)
            hessian = (design.T @ (weights[:, None] * design)) / y.size
            hessian += LOGISTIC_L2 * regularizer
            hessian += np.eye(design.shape[1], dtype=np.float64) * 1e-12
            step = np.linalg.solve(hessian, gradient)
            coefficients -= step
            if float(np.max(np.abs(step))) <= LOGISTIC_TOLERANCE:
                converged = True
                break
    except np.linalg.LinAlgError:
        return None, "regularized logistic solve was singular"
    if not converged:
        return (
            None,
            "regularized logistic did not converge within {} iterations".format(LOGISTIC_MAX_ITERATIONS),
        )
    if not np.all(np.isfinite(coefficients)):
        return None, "regularized logistic coefficients were non-finite"
    return (
        _LogisticModel(
            mean=mean,
            scale=scale,
            coefficients=coefficients,
            iterations=iterations,
        ),
        "",
    )


def _predict_logistic(model: _LogisticModel, x: np.ndarray) -> np.ndarray:
    standardized = (x - model.mean) / model.scale
    design = np.column_stack([np.ones(standardized.shape[0]), standardized])
    logits = np.clip(design @ model.coefficients, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _fit_knn(x: np.ndarray, y: np.ndarray, query_ids: Tuple[str, ...]) -> Tuple[Optional[_KNNModel], str]:
    class_counts = {
        0: int(np.sum(y == 0.0)),
        1: int(np.sum(y == 1.0)),
    }
    if x.shape[0] < KNN_K or min(class_counts.values()) < 1:
        return None, "requires at least {} non-tied SELECT examples and both directions".format(KNN_K)
    mean, scale = _standardizer(x)
    return (
        _KNNModel(
            mean=mean,
            scale=scale,
            x_train=(x - mean) / scale,
            y_train=y,
            query_ids=query_ids,
        ),
        "",
    )


def _predict_knn(model: _KNNModel, x: np.ndarray) -> np.ndarray:
    standardized = (x - model.mean) / model.scale
    probabilities = np.empty(standardized.shape[0], dtype=np.float64)
    canonical_query_ids = np.asarray(model.query_ids, dtype=str)
    for row_index, row in enumerate(standardized):
        distances = np.sum((model.x_train - row) ** 2, axis=1)
        nearest = np.lexsort((canonical_query_ids, distances))[:KNN_K]
        probabilities[row_index] = float(np.mean(model.y_train[nearest]))
    return probabilities


def _fixed_direction(
    rows: Sequence[Mapping[str, Any]],
    blocks: Mapping[str, Mapping[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """Select the fixed route under the registered equal-cell macro estimand."""

    if not rows:
        raise BCC1ValidationError("fixed-direction selection requires at least one row")
    cells: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        block = blocks[row["block_id"]]
        cell = (
            block["dataset_family_id"],
            block["transition_family_id"],
        )
        cells.setdefault(cell, []).append(row)
    reverse_cell_means = [
        float(np.mean([row["scores"]["reverse"] for row in cell_rows])) for _, cell_rows in sorted(cells.items())
    ]
    forward_cell_means = [
        float(np.mean([row["scores"]["forward"] for row in cell_rows])) for _, cell_rows in sorted(cells.items())
    ]
    reverse_mean_raw = float(np.mean(reverse_cell_means))
    forward_mean_raw = float(np.mean(forward_cell_means))
    direction = "reverse" if reverse_mean_raw >= forward_mean_raw else "forward"
    return direction, {
        "reverse": _round_float(reverse_mean_raw),
        "forward": _round_float(forward_mean_raw),
        "estimand": "equal-weighted observed dataset-family x transition-family cell means",
        "cell_count": len(cells),
    }


def _directions_from_probabilities(
    probabilities: np.ndarray, fixed_direction: str
) -> Tuple[Tuple[str, ...], Tuple[bool, ...]]:
    directions = []
    abstentions = []
    for probability in probabilities:
        confidence = max(float(probability), 1.0 - float(probability))
        abstained = confidence < ABSTAIN_CONFIDENCE
        if abstained:
            direction = fixed_direction
        else:
            direction = "reverse" if probability >= 0.5 else "forward"
        directions.append(direction)
        abstentions.append(abstained)
    return tuple(directions), tuple(abstentions)


def _scores_for_directions(rows: Sequence[Mapping[str, Any]], directions: Sequence[str]) -> np.ndarray:
    if len(rows) != len(directions):
        raise BCC1ValidationError("policy produced {} directions for {} REPORT rows".format(len(directions), len(rows)))
    return np.asarray(
        [row["scores"][direction] for row, direction in zip(rows, directions)],
        dtype=np.float64,
    )


def _headroom_summary(
    method_scores: np.ndarray,
    fixed_scores: np.ndarray,
    oracle_scores: np.ndarray,
) -> Dict[str, Any]:
    fixed_mean = float(np.mean(fixed_scores))
    method_mean = float(np.mean(method_scores))
    oracle_mean = float(np.mean(oracle_scores))
    available = oracle_mean - fixed_mean
    closed = method_mean - fixed_mean
    if available <= TIE_TOLERANCE:
        fraction: Optional[float] = None
        status = "no_positive_oracle_headroom"
    else:
        fraction = _round_float(closed / available)
        status = "defined"
    return {
        "definition": "(method_mean-best_fixed_mean)/(direction_oracle_mean-best_fixed_mean)",
        "status": status,
        "available": _round_float(available),
        "closed": _round_float(closed),
        "fraction_closed": fraction,
    }


def _method_report(
    *,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    directions: Sequence[str],
    probabilities: Optional[Sequence[float]],
    abstentions: Sequence[bool],
    status: str,
    reason: Optional[str],
    fixed_scores: np.ndarray,
    oracle_scores: np.ndarray,
    training: Mapping[str, Any],
    blocks: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    method_scores = _scores_for_directions(rows, directions)
    block_ids = tuple(row["block_id"] for row in rows)
    decisions = []
    for index, (row, direction, score, abstained) in enumerate(zip(rows, directions, method_scores, abstentions)):
        decision = {
            "query_id": row["query_id"],
            "direction": direction,
            "abstained_to_best_fixed": bool(abstained),
            "selected_score": _round_float(score),
        }
        if probabilities is not None:
            decision["probability_reverse"] = _round_float(probabilities[index])
        decisions.append(decision)
    result = {
        "name": name,
        "status": status,
        "reason": reason,
        "training": dict(training),
        "abstention_count": int(sum(bool(value) for value in abstentions)),
        "mean_score": _mean(method_scores),
        "delta_vs_best_fixed": _round_float(float(np.mean(method_scores - fixed_scores))),
        "paired_bootstrap_vs_best_fixed": paired_bootstrap_mean_ci(method_scores, fixed_scores),
        "hierarchical_bootstrap_vs_best_fixed": hierarchical_paired_bootstrap_mean_ci(
            method_scores, fixed_scores, block_ids
        ),
        "crossed_hierarchical_bootstrap_vs_best_fixed": (
            crossed_hierarchical_paired_bootstrap_mean_ci(method_scores, fixed_scores, rows, blocks)
        ),
        "oracle_headroom": _headroom_summary(method_scores, fixed_scores, oracle_scores),
        "decisions": decisions,
    }
    return result


def _soft_fusion_report(
    validated: ValidatedBCC1,
    *,
    fixed_scores: np.ndarray,
    oracle_scores: np.ndarray,
) -> Dict[str, Any]:
    if validated.soft_fusion is None:
        return {
            "name": "soft_score_fusion",
            "status": "not_run",
            "reason": "manifest has no justified precomputed retrieval-score fusion inputs",
        }
    candidates = validated.soft_fusion["candidates"]
    select_means = []
    for candidate in candidates:
        candidate_id = candidate["id"]
        values = np.asarray(
            [row["fusion_scores"][candidate_id] for row in validated.select_rows],
            dtype=np.float64,
        )
        select_means.append(
            {
                "id": candidate_id,
                "alpha": _round_float(candidate["alpha"]),
                "mean_score": _mean(values),
            }
        )
    selected = sorted(
        select_means,
        key=lambda item: (-item["mean_score"], item["alpha"], item["id"]),
    )[0]
    selected_id = selected["id"]
    report_scores = np.asarray(
        [row["fusion_scores"][selected_id] for row in validated.report_rows],
        dtype=np.float64,
    )
    block_ids = tuple(row["block_id"] for row in validated.report_rows)
    return {
        "name": "soft_score_fusion",
        "status": "evaluated",
        "reason": None,
        "input_justification": {
            "source_kind": validated.soft_fusion["source_kind"],
            "qrels_free_at_inference": True,
            "normalization": validated.soft_fusion["normalization"],
        },
        "selection": {
            "selected_candidate_id": selected_id,
            "selected_alpha": selected["alpha"],
            "select_candidates": select_means,
        },
        "mean_score": _mean(report_scores),
        "delta_vs_best_fixed": _round_float(float(np.mean(report_scores - fixed_scores))),
        "paired_bootstrap_vs_best_fixed": paired_bootstrap_mean_ci(report_scores, fixed_scores),
        "hierarchical_bootstrap_vs_best_fixed": hierarchical_paired_bootstrap_mean_ci(
            report_scores, fixed_scores, block_ids
        ),
        "crossed_hierarchical_bootstrap_vs_best_fixed": (
            crossed_hierarchical_paired_bootstrap_mean_ci(
                report_scores, fixed_scores, validated.report_rows, validated.blocks
            )
        ),
        "oracle_headroom": _headroom_summary(report_scores, fixed_scores, oracle_scores),
        "per_query": [
            {
                "query_id": row["query_id"],
                "selected_score": _round_float(score),
            }
            for row, score in zip(validated.report_rows, report_scores)
        ],
    }


def _method_dev_reference(rows: Sequence[Mapping[str, Any]], score_name: str) -> Dict[str, Any]:
    if score_name not in rows[0]["scores"]:
        return {
            "status": "not_provided",
            "mean_score": None,
        }
    values = np.asarray(
        [row["scores"][score_name] for row in rows],
        dtype=np.float64,
    )
    return {
        "status": "provided",
        "mean_score": _mean(values),
    }


def _method_dev_block_summary(
    block_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    reverse = np.asarray([row["scores"]["reverse"] for row in rows], dtype=np.float64)
    forward = np.asarray([row["scores"]["forward"] for row in rows], dtype=np.float64)
    mismatch = np.asarray([row["scores"]["mismatch"] for row in rows], dtype=np.float64)
    native_new = np.asarray([row["scores"]["native_new"] for row in rows], dtype=np.float64)
    reverse_wins = int(np.sum(reverse > forward + TIE_TOLERANCE))
    forward_wins = int(np.sum(forward > reverse + TIE_TOLERANCE))
    ties = int(reverse.size - reverse_wins - forward_wins)
    best_observed_direction = "reverse" if float(np.mean(reverse)) >= float(np.mean(forward)) else "forward"
    best_fixed = reverse if best_observed_direction == "reverse" else forward
    oracle = np.maximum(reverse, forward)
    independence_group_ids = [row["independence_group_id"] for row in rows]
    return {
        "block_id": block_id,
        "query_count": len(rows),
        "reverse_mean": _mean(reverse),
        "forward_mean": _mean(forward),
        "mismatch_mean": _mean(mismatch),
        "native_new_mean": _mean(native_new),
        "native_old": _method_dev_reference(rows, "native_old"),
        "reverse_win_count": reverse_wins,
        "forward_win_count": forward_wins,
        "tie_count": ties,
        "best_observed_direction": best_observed_direction,
        "best_observed_direction_mean": _mean(best_fixed),
        "direction_oracle_mean": _mean(oracle),
        "descriptive_oracle_headroom": _round_float(float(np.mean(oracle - best_fixed))),
        "reverse_vs_forward": grouped_paired_bootstrap_mean_ci(reverse, forward, independence_group_ids),
        "reverse_vs_mismatch": grouped_paired_bootstrap_mean_ci(reverse, mismatch, independence_group_ids),
        "forward_vs_mismatch": grouped_paired_bootstrap_mean_ci(forward, mismatch, independence_group_ids),
        "native_new_vs_mismatch": grouped_paired_bootstrap_mean_ci(native_new, mismatch, independence_group_ids),
    }


def _method_dev_transfer_diagnostic(
    rows_by_block: Mapping[str, Sequence[Mapping[str, Any]]],
    feature_names: Sequence[str],
    blocks: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    block_ids = tuple(sorted(rows_by_block))
    if len(block_ids) < 2:
        return {
            "status": "not_run",
            "reason": "requires at least two declared METHOD_DEV structural blocks",
            "classification": "exploratory_non_promotional",
        }
    accumulators: Dict[str, Dict[str, Any]] = {
        "regularized_logistic": {
            "scores": [],
            "fixed": [],
            "oracle": [],
            "block_ids": [],
            "rows": [],
            "folds": [],
            "fallbacks": 0,
        },
        "knn": {
            "scores": [],
            "fixed": [],
            "oracle": [],
            "block_ids": [],
            "rows": [],
            "folds": [],
            "fallbacks": 0,
        },
    }
    for heldout_block in block_ids:
        heldout_rows = tuple(rows_by_block[heldout_block])
        heldout_group_ids = {row["independence_group_id"] for row in heldout_rows}
        training_rows = tuple(
            row
            for block_id in block_ids
            if block_id != heldout_block
            for row in rows_by_block[block_id]
            if row["independence_group_id"] not in heldout_group_ids
        )
        if not training_rows:
            for accumulator in accumulators.values():
                accumulator["folds"].append(
                    {
                        "heldout_block_id": heldout_block,
                        "status": "not_run",
                        "reason": "no non-overlapping training rows remain",
                    }
                )
                accumulator["fallbacks"] += 1
            continue
        fixed_direction, _ = _fixed_direction(training_rows, blocks)
        x_train, y_train, training_query_ids, ties = _direction_training_data(training_rows, feature_names)
        x_heldout = _matrix(heldout_rows, feature_names)
        fixed_scores = _scores_for_directions(heldout_rows, tuple(fixed_direction for _ in heldout_rows))
        heldout_reverse = np.asarray([row["scores"]["reverse"] for row in heldout_rows], dtype=np.float64)
        heldout_forward = np.asarray([row["scores"]["forward"] for row in heldout_rows], dtype=np.float64)
        oracle_scores = np.maximum(heldout_reverse, heldout_forward)

        logistic_model, logistic_reason = _fit_logistic(x_train, y_train)
        if logistic_model is None:
            logistic_probabilities = None
            logistic_directions = tuple(fixed_direction for _ in heldout_rows)
            logistic_abstentions = tuple(True for _ in heldout_rows)
            logistic_status = "fallback_only"
        else:
            logistic_probabilities = _predict_logistic(logistic_model, x_heldout)
            logistic_directions, logistic_abstentions = _directions_from_probabilities(
                logistic_probabilities, fixed_direction
            )
            logistic_status = "evaluated"
            logistic_reason = None

        knn_model, knn_reason = _fit_knn(x_train, y_train, training_query_ids)
        if knn_model is None:
            knn_probabilities = None
            knn_directions = tuple(fixed_direction for _ in heldout_rows)
            knn_abstentions = tuple(True for _ in heldout_rows)
            knn_status = "fallback_only"
        else:
            knn_probabilities = _predict_knn(knn_model, x_heldout)
            knn_directions, knn_abstentions = _directions_from_probabilities(knn_probabilities, fixed_direction)
            knn_status = "evaluated"
            knn_reason = None

        fold_policies = {
            "regularized_logistic": (
                logistic_directions,
                logistic_abstentions,
                logistic_status,
                logistic_reason,
            ),
            "knn": (
                knn_directions,
                knn_abstentions,
                knn_status,
                knn_reason,
            ),
        }
        for policy_name, (
            directions,
            abstentions,
            status,
            reason,
        ) in fold_policies.items():
            scores = _scores_for_directions(heldout_rows, directions)
            accumulator = accumulators[policy_name]
            accumulator["scores"].extend(float(value) for value in scores)
            accumulator["fixed"].extend(float(value) for value in fixed_scores)
            accumulator["oracle"].extend(float(value) for value in oracle_scores)
            accumulator["block_ids"].extend(heldout_block for _ in heldout_rows)
            accumulator["rows"].extend(heldout_rows)
            if status != "evaluated":
                accumulator["fallbacks"] += 1
            accumulator["folds"].append(
                {
                    "heldout_block_id": heldout_block,
                    "status": status,
                    "reason": reason,
                    "training_query_count": len(training_rows),
                    "training_ties_excluded": ties,
                    "heldout_query_count": len(heldout_rows),
                    "best_fixed_direction_from_training": fixed_direction,
                    "abstention_count": int(sum(bool(value) for value in abstentions)),
                    "mean_score": _mean(scores),
                    "best_fixed_mean": _mean(fixed_scores),
                    "delta_vs_fold_best_fixed": _round_float(float(np.mean(scores - fixed_scores))),
                }
            )

    policy_reports: Dict[str, Any] = {}
    for policy_name, accumulator in accumulators.items():
        if not accumulator["scores"]:
            policy_reports[policy_name] = {
                "status": "not_run",
                "reason": "no holdout fold had non-overlapping training rows",
                "folds": accumulator["folds"],
            }
            continue
        scores = np.asarray(accumulator["scores"], dtype=np.float64)
        fixed = np.asarray(accumulator["fixed"], dtype=np.float64)
        oracle = np.asarray(accumulator["oracle"], dtype=np.float64)
        fallback_count = int(accumulator["fallbacks"])
        if fallback_count == 0:
            status = "evaluated"
        elif fallback_count == len(block_ids):
            status = "fallback_only"
        else:
            status = "evaluated_with_fallbacks"
        policy_reports[policy_name] = {
            "status": status,
            "reason": None,
            "heldout_block_count": len(block_ids),
            "fallback_fold_count": fallback_count,
            "mean_score": _mean(scores),
            "best_fixed_mean": _mean(fixed),
            "delta_vs_fold_best_fixed": _round_float(float(np.mean(scores - fixed))),
            "hierarchical_bootstrap_vs_fold_best_fixed": (
                crossed_hierarchical_paired_bootstrap_mean_ci(scores, fixed, accumulator["rows"], blocks)
            ),
            "oracle_headroom": _headroom_summary(scores, fixed, oracle),
            "folds": accumulator["folds"],
        }
    generalizing_policies = [
        policy_name
        for policy_name, policy_report in policy_reports.items()
        if policy_report["status"] == "evaluated"
        and policy_report["delta_vs_fold_best_fixed"] > 0.0
        and policy_report["hierarchical_bootstrap_vs_fold_best_fixed"]["ci_lower"] > 0.0
    ]
    if generalizing_policies:
        disposition = "RECONDITION_THEN_PREREGISTER"
        disposition_reason = (
            "at least one qrels-free policy beat its fold-specific best fixed "
            "route across held-out structural blocks; this remains METHOD_DEV only"
        )
    else:
        disposition = "CUT_CURRENT_FEATURE_POLICY_SET"
        disposition_reason = (
            "no qrels-free policy produced a strictly positive held-out "
            "hierarchical lower bound versus fold-specific best fixed; oracle "
            "headroom alone is not routeability evidence"
        )
    return {
        "status": "evaluated",
        "classification": "exploratory_non_promotional",
        "holdout_unit": "one exact declared METHOD_DEV structural block",
        "overlap_guard": (
            "all rows sharing a heldout independence_group_id are excluded "
            "from training across encoder-transition blocks"
        ),
        "feature_boundary": "validated qrels-free inference-available features only",
        "outcome_access_order": (
            "fit and freeze each heldout policy from other blocks before " "scoring heldout reverse/forward outcomes"
        ),
        "promotion_eligible": False,
        "disposition": disposition,
        "disposition_reason": disposition_reason,
        "generalizing_policies": generalizing_policies,
        "policies": policy_reports,
    }


def _structural_split_independence_audit(
    rows: Sequence[Mapping[str, Any]],
    blocks: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    schemes: Dict[str, Any] = {}
    group_folds = _query_group_folds(rows, blocks)
    for axis_field in ("dataset_family_id", "transition_family_id"):
        axis_values = tuple(sorted({blocks[row["block_id"]][axis_field] for row in rows}))
        folds = []
        partitions = []
        empty_partition_count = 0
        expected_partition_count = (
            len(axis_values) * QUERY_FOLD_COUNT if axis_field == "transition_family_id" else len(axis_values)
        )
        if axis_field == "transition_family_id":
            for heldout_value in axis_values:
                for heldout_fold in range(QUERY_FOLD_COUNT):
                    heldout_rows = tuple(
                        row
                        for row in rows
                        if blocks[row["block_id"]][axis_field] == heldout_value
                        and group_folds[row["independence_group_id"]] == heldout_fold
                    )
                    if not heldout_rows:
                        empty_partition_count += 1
                        folds.append(
                            {
                                "heldout_family_id": heldout_value,
                                "heldout_query_fold": heldout_fold,
                                "heldout_observation_count": 0,
                                "heldout_independence_group_count": 0,
                                "non_overlapping_training_observation_count": 0,
                                "independence_group_overlap_count": 0,
                                "training_direction_class_count": 0,
                                "identifiable": False,
                                "failure_reason": "empty evaluation partition",
                            }
                        )
                        continue
                    training_rows = tuple(
                        row
                        for row in rows
                        if blocks[row["block_id"]][axis_field] != heldout_value
                        and group_folds[row["independence_group_id"]] != heldout_fold
                    )
                    partitions.append((heldout_value, heldout_fold, heldout_rows, training_rows))
        else:
            for heldout_value in axis_values:
                heldout_rows = tuple(row for row in rows if blocks[row["block_id"]][axis_field] == heldout_value)
                heldout_groups = {row["independence_group_id"] for row in heldout_rows}
                training_rows = tuple(
                    row
                    for row in rows
                    if blocks[row["block_id"]][axis_field] != heldout_value
                    and row["independence_group_id"] not in heldout_groups
                )
                partitions.append((heldout_value, None, heldout_rows, training_rows))
        for heldout_value, heldout_fold, heldout_rows, training_rows in partitions:
            heldout_groups = {row["independence_group_id"] for row in heldout_rows}
            training_groups = {row["independence_group_id"] for row in training_rows}
            _, training_labels, _, _ = _direction_training_data(training_rows, ())
            folds.append(
                {
                    "heldout_family_id": heldout_value,
                    "heldout_query_fold": heldout_fold,
                    "heldout_observation_count": len(heldout_rows),
                    "heldout_independence_group_count": len(heldout_groups),
                    "non_overlapping_training_observation_count": len(training_rows),
                    "independence_group_overlap_count": len(heldout_groups & training_groups),
                    "training_direction_class_count": len(set(float(value) for value in training_labels)),
                    "identifiable": (
                        bool(training_rows)
                        and not (heldout_groups & training_groups)
                        and len(set(float(value) for value in training_labels)) == 2
                    ),
                }
            )
        schemes[axis_field] = {
            "folds": folds,
            "query_group_fold_count": (QUERY_FOLD_COUNT if axis_field == "transition_family_id" else None),
            "expected_partition_count": expected_partition_count,
            "observed_partition_count": len(partitions),
            "empty_partition_count": empty_partition_count,
            "all_folds_identifiable": (
                len(folds) == expected_partition_count
                and empty_partition_count == 0
                and all(fold["identifiable"] for fold in folds)
            ),
        }
    all_identifiable = all(scheme["all_folds_identifiable"] for scheme in schemes.values())
    return {
        "grouping_unit": "independence_group_id",
        "same_group_cross_transition_leakage_forbidden": True,
        "schemes": schemes,
        "all_preregistered_schemes_identifiable": all_identifiable,
        "disposition": (
            "STRUCTURAL_SPLITS_IDENTIFIABLE" if all_identifiable else "BLOCKED_INDEPENDENCE_GROUP_SPLIT_DESIGN"
        ),
        "reason": (
            None
            if all_identifiable
            else (
                "at least one outer structural holdout has no training rows "
                "with both direction classes after independence-group "
                "exclusion; confirmatory use must remain blocked"
            )
        ),
    }


def _method_dev_cut_disposition(
    *,
    dual_read_blocked: bool,
    reverse_role_blocked: bool,
) -> str:
    if dual_read_blocked and reverse_role_blocked:
        return "CUT_ROLE_MISMATCH_AND_DUAL_READ_FEATURE_SET"
    if dual_read_blocked:
        return "CUT_DUAL_READ_FEATURE_SET"
    if reverse_role_blocked:
        return "CUT_UNVALIDATED_DOCUMENT_ROLE_REVERSE_PROXY"
    return "PRE_SEARCH_FEATURES_ONLY"


def _analyze_method_dev(validated: ValidatedBCC1) -> Dict[str, Any]:
    """Return an explicitly non-promotional diagnostic for METHOD_DEV rows."""

    rows = validated.method_dev_rows
    reverse_role_blocked = validated.reverse_route_role_validation == UNVALIDATED_DOCUMENT_ROLE_REVERSE_PROXY
    operational_cut = _method_dev_cut_disposition(
        dual_read_blocked=bool(validated.dual_read_feature_names),
        reverse_role_blocked=reverse_role_blocked,
    )
    rows_by_block: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        rows_by_block.setdefault(row["block_id"], []).append(row)
    block_summaries = [
        _method_dev_block_summary(block_id, rows_by_block[block_id]) for block_id in sorted(rows_by_block)
    ]
    aggregate_summary = _method_dev_block_summary("__all_method_dev__", rows)
    report: Dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "pack_id": validated.pack_id,
        "input_fingerprints": {
            "pack_sha256": validated.pack_sha256,
            "manifest_sha256": validated.manifest_sha256,
        },
        "evidence_classification": {
            "evidence_mode": "METHOD_DEV",
            "sampled": validated.sampled,
            "promotion_eligible": False,
            "confirmatory_report_evaluated": False,
            "reason": (
                "METHOD_DEV uses no disjoint REPORT block; all summaries are "
                "diagnostic and cannot satisfy a formal BCC gate"
            ),
            "operational_routeability": operational_cut,
        },
        "protocol": {
            "select_report_separation": "not_applicable_no_report_access",
            "supervised_policy_fitting": ("exploratory_logistic_and_knn_diagnostics_allowed_but_non_promotional"),
            "bootstrap": {
                "confidence": BOOTSTRAP_CONFIDENCE,
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
            },
        },
        "data_contract": {
            "metric": validated.metric_name,
            "metric_contract": dict(validated.metric_contract),
            "metric_range": [
                _round_float(validated.metric_minimum),
                _round_float(validated.metric_maximum),
            ],
            "feature_names": list(validated.feature_names),
            "dual_read_feature_names": list(validated.dual_read_feature_names),
            "reverse_route_role_contract": {
                "validation_state": validated.reverse_route_role_validation,
                "confirmatory_eligible": not reverse_role_blocked,
                "independent_cut": ("CUT_UNVALIDATED_DOCUMENT_ROLE_REVERSE_PROXY" if reverse_role_blocked else None),
            },
            "feature_execution_boundary": {
                "status": ("DUAL_READ_METHOD_DEV" if validated.dual_read_feature_names else "PRE_SEARCH_ONLY"),
                "operational_one_route_eligible": not bool(validated.dual_read_feature_names),
                "reason": (
                    "current features require both forward and reverse "
                    "retrievals before routing, violating one-authorized-search "
                    "and ABSTAIN semantics"
                    if validated.dual_read_feature_names
                    else None
                ),
            },
            "method_dev_query_count": len(rows),
            "report_query_count": 0,
            "report_consumed_by_this_analysis": False,
        },
        "method_dev_diagnostic": {
            "classification": "exploratory_non_promotional",
            "selection_and_evaluation_use_same_rows": True,
            "aggregate": aggregate_summary,
            "per_block": block_summaries,
            "structural_split_independence_audit": (_structural_split_independence_audit(rows, validated.blocks)),
            "independence_safe_axis_transfer": _method_dev_axis_transfer(validated),
            "leave_one_structural_block_out": _method_dev_transfer_diagnostic(
                rows_by_block, validated.feature_names, validated.blocks
            ),
        },
        "methods": {
            "regularized_logistic": {
                "status": "not_run",
                "reason": "requires frozen SELECT and untouched REPORT blocks",
            },
            "knn": {
                "status": "not_run",
                "reason": "requires frozen SELECT and untouched REPORT blocks",
            },
            "soft_score_fusion": {
                "status": "not_run",
                "reason": "candidate selection is forbidden in METHOD_DEV mode",
            },
        },
    }
    canonical_json_bytes(report)
    return report


def _query_group_folds(
    rows: Sequence[Mapping[str, Any]],
    blocks: Mapping[str, Mapping[str, Any]],
) -> Dict[str, int]:
    groups_by_dataset: Dict[str, set] = {}
    for row in rows:
        dataset_id = blocks[row["block_id"]]["dataset_family_id"]
        groups_by_dataset.setdefault(dataset_id, set()).add(row["independence_group_id"])
    assignments: Dict[str, int] = {}
    for dataset_id, groups in groups_by_dataset.items():
        ordered = sorted(
            groups,
            key=lambda group_id: (
                hashlib.sha256(("bcc1-v1-query-fold|" + dataset_id + "|" + group_id).encode("utf-8")).digest(),
                group_id,
            ),
        )
        for rank, group_id in enumerate(ordered):
            assignments[group_id] = rank % QUERY_FOLD_COUNT
    return assignments


def _select_structural_scheme(
    *,
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    blocks: Mapping[str, Mapping[str, Any]],
    policy_name: str,
    axis_field: str,
    gates: Mapping[str, Any],
) -> Dict[str, Any]:
    axis_values = tuple(sorted({blocks[row["block_id"]][axis_field] for row in rows}))
    predicted_scores: List[float] = []
    fixed_score_values: List[float] = []
    predicted_rows: List[Mapping[str, Any]] = []
    abstentions: List[bool] = []
    fallback_folds = 0
    invalid_partitions = 0
    partition_audit = []
    oof_residual_audit: List[Dict[str, Any]] = []
    group_folds = _query_group_folds(rows, blocks)
    partitions = []
    empty_partition_count = 0
    expected_partition_count = (
        len(axis_values) * QUERY_FOLD_COUNT if axis_field == "transition_family_id" else len(axis_values)
    )
    if axis_field == "transition_family_id":
        for heldout_value in axis_values:
            for heldout_fold in range(QUERY_FOLD_COUNT):
                heldout_rows = tuple(
                    row
                    for row in rows
                    if blocks[row["block_id"]][axis_field] == heldout_value
                    and group_folds[row["independence_group_id"]] == heldout_fold
                )
                if not heldout_rows:
                    empty_partition_count += 1
                    invalid_partitions += 1
                    partition_audit.append(
                        {
                            "heldout_family_id": heldout_value,
                            "heldout_query_fold": heldout_fold,
                            "training_observation_count": 0,
                            "training_non_tied_count": 0,
                            "training_ties_excluded": 0,
                            "evaluation_observation_count": 0,
                            "training_independence_group_count": 0,
                            "evaluation_independence_group_count": 0,
                            "independence_group_overlap_count": 0,
                            "training_direction_class_count": 0,
                            "fold_fixed_direction": None,
                            "failure_stage": "partition",
                            "failure_reason": "empty evaluation partition",
                            "valid": False,
                        }
                    )
                    continue
                training_rows = tuple(
                    row
                    for row in rows
                    if blocks[row["block_id"]][axis_field] != heldout_value
                    and group_folds[row["independence_group_id"]] != heldout_fold
                )
                partitions.append((heldout_value, heldout_fold, heldout_rows, training_rows))
    else:
        for heldout_value in axis_values:
            heldout_rows = tuple(row for row in rows if blocks[row["block_id"]][axis_field] == heldout_value)
            heldout_groups = {row["independence_group_id"] for row in heldout_rows}
            training_rows = tuple(
                row
                for row in rows
                if blocks[row["block_id"]][axis_field] != heldout_value
                and row["independence_group_id"] not in heldout_groups
            )
            partitions.append((heldout_value, None, heldout_rows, training_rows))
    for heldout_value, heldout_fold, heldout_rows, training_rows in partitions:
        training_groups = {row["independence_group_id"] for row in training_rows}
        heldout_groups = {row["independence_group_id"] for row in heldout_rows}
        overlap_count = len(training_groups & heldout_groups)
        x_train, y_train, training_query_ids, training_ties = _direction_training_data(
            training_rows,
            feature_names,
        )
        class_count = len(set(float(value) for value in y_train))
        fold_fixed_direction = _fixed_direction(training_rows, blocks)[0] if training_rows else "reverse"
        x_heldout = _matrix(heldout_rows, feature_names)
        failure_stage: Optional[str] = None
        failure_reason: Optional[str] = None
        if not training_rows:
            failure_stage = "partition"
            failure_reason = "no non-overlapping training rows remain"
        elif overlap_count:
            failure_stage = "partition"
            failure_reason = "training and evaluation independence groups overlap"
        elif class_count != 2:
            failure_stage = "partition"
            failure_reason = "training rows do not contain both strict direction classes"

        probabilities: Optional[np.ndarray] = None
        if failure_reason is None:
            try:
                if policy_name == "regularized_logistic":
                    model, fit_reason = _fit_logistic(x_train, y_train)
                else:
                    model, fit_reason = _fit_knn(
                        x_train,
                        y_train,
                        training_query_ids,
                    )
            except Exception as exc:
                model = None
                fit_reason = "{} fit raised {}: {}".format(
                    policy_name,
                    type(exc).__name__,
                    exc,
                )
            if model is None:
                failure_stage = "fit"
                failure_reason = fit_reason or "{} fit returned no model".format(policy_name)
            else:
                try:
                    if policy_name == "regularized_logistic":
                        predicted = _predict_logistic(model, x_heldout)
                    else:
                        predicted = _predict_knn(model, x_heldout)
                    probabilities = np.asarray(predicted, dtype=np.float64)
                    if probabilities.shape != (len(heldout_rows),):
                        raise BCC1ValidationError(
                            "prediction shape {} does not match heldout row count {}".format(
                                probabilities.shape,
                                len(heldout_rows),
                            )
                        )
                    if not np.all(np.isfinite(probabilities)):
                        raise BCC1ValidationError("predictions contain non-finite values")
                    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
                        raise BCC1ValidationError("predictions fall outside the closed interval [0, 1]")
                except Exception as exc:
                    probabilities = None
                    failure_stage = "predict"
                    failure_reason = "{} predict failed with {}: {}".format(
                        policy_name,
                        type(exc).__name__,
                        exc,
                    )

        partition_valid = failure_reason is None and probabilities is not None
        if not partition_valid:
            invalid_partitions += 1
        partition_audit.append(
            {
                "heldout_family_id": heldout_value,
                "heldout_query_fold": heldout_fold,
                "training_observation_count": len(training_rows),
                "training_non_tied_count": int(y_train.size),
                "training_ties_excluded": training_ties,
                "evaluation_observation_count": len(heldout_rows),
                "training_independence_group_count": len(training_groups),
                "evaluation_independence_group_count": len(heldout_groups),
                "independence_group_overlap_count": overlap_count,
                "training_direction_class_count": class_count,
                "fold_fixed_direction": fold_fixed_direction,
                "failure_stage": failure_stage,
                "failure_reason": failure_reason,
                "valid": partition_valid,
            }
        )
        if probabilities is None:
            directions = tuple(fold_fixed_direction for _ in heldout_rows)
            fold_abstentions = tuple(True for _ in heldout_rows)
            fallback_folds += 1
        else:
            directions, fold_abstentions = _directions_from_probabilities(probabilities, fold_fixed_direction)
        scores = _scores_for_directions(heldout_rows, directions)
        fixed = _scores_for_directions(heldout_rows, tuple(fold_fixed_direction for _ in heldout_rows))
        predicted_scores.extend(float(value) for value in scores)
        fixed_score_values.extend(float(value) for value in fixed)
        predicted_rows.extend(heldout_rows)
        abstentions.extend(bool(value) for value in fold_abstentions)
        for row, direction, abstained, candidate_score, fixed_score in zip(
            heldout_rows,
            directions,
            fold_abstentions,
            scores,
            fixed,
        ):
            block = blocks[row["block_id"]]
            oof_residual_audit.append(
                {
                    "block_id": row["block_id"],
                    "query_id": row["query_id"],
                    "independence_group_id": row["independence_group_id"],
                    "dataset_family_id": block["dataset_family_id"],
                    "transition_family_id": block["transition_family_id"],
                    "query_group_fold": group_folds[row["independence_group_id"]],
                    "heldout_family_id": heldout_value,
                    "heldout_query_fold": heldout_fold,
                    "partition_valid": partition_valid,
                    "partition_failure_stage": failure_stage,
                    "partition_failure_reason": failure_reason,
                    "fold_fixed_direction": fold_fixed_direction,
                    "direction": direction,
                    "abstained": bool(abstained),
                    "candidate_score": _round_float(float(candidate_score)),
                    "fixed_score": _round_float(float(fixed_score)),
                    "residual": _round_float(float(candidate_score - fixed_score)),
                }
            )
    candidate = np.asarray(predicted_scores, dtype=np.float64)
    fixed = np.asarray(fixed_score_values, dtype=np.float64)
    evaluation_keys = [(row["block_id"], row["query_id"]) for row in predicted_rows]
    expected_keys = {(row["block_id"], row["query_id"]) for row in rows}
    evaluated_exactly_once = (
        len(evaluation_keys) == len(expected_keys)
        and len(set(evaluation_keys)) == len(evaluation_keys)
        and set(evaluation_keys) == expected_keys
    )
    interval = crossed_hierarchical_paired_bootstrap_mean_ci(candidate, fixed, predicted_rows, blocks)
    block_ids = tuple(row["block_id"] for row in predicted_rows)
    non_abstained = np.asarray([not value for value in abstentions], dtype=np.bool_)
    overall_coverage = float(np.mean(non_abstained))
    block_results = []
    for block_id in sorted(set(block_ids)):
        indices = np.asarray(
            [index for index, observed in enumerate(block_ids) if observed == block_id],
            dtype=np.int64,
        )
        point_delta = float(np.mean(candidate[indices] - fixed[indices]))
        coverage = float(np.mean(non_abstained[indices]))
        block_results.append(
            {
                "block_id": block_id,
                "point_delta": _round_float(point_delta),
                "coverage": _round_float(coverage),
                "point_safety_pass": point_delta >= -0.02 - TIE_TOLERANCE,
                "coverage_pass": coverage + TIE_TOLERANCE >= gates["minimum_block_coverage"],
            }
        )
    checks = {
        "every_row_evaluated_exactly_once": evaluated_exactly_once,
        "all_partitions_valid": (
            invalid_partitions == 0 and empty_partition_count == 0 and len(partitions) == expected_partition_count
        ),
        "point_mwee": interval["mean_difference"] >= gates["minimum_worthwhile_effect"] - TIE_TOLERANCE,
        "hierarchical_lower_bound": interval["ci_lower"] > 0.0,
        "overall_coverage": overall_coverage + TIE_TOLERANCE >= gates["minimum_overall_coverage"],
        "block_coverage": all(item["coverage_pass"] for item in block_results),
        "block_point_safety": all(item["point_safety_pass"] for item in block_results),
    }
    ordered_oof_audit = sorted(
        oof_residual_audit,
        key=lambda item: (item["block_id"], item["query_id"]),
    )
    oof_audit_payload = {
        "schema_version": OOF_RESIDUAL_AUDIT_SCHEMA,
        "policy_name": policy_name,
        "axis": axis_field,
        "rows": ordered_oof_audit,
    }
    return {
        "axis": axis_field,
        "heldout_family_count": len(axis_values),
        "partition_count": len(partitions),
        "expected_partition_count": expected_partition_count,
        "empty_partition_count": empty_partition_count,
        "invalid_partition_count": invalid_partitions,
        "every_row_evaluated_exactly_once": evaluated_exactly_once,
        "fallback_fold_count": fallback_folds,
        "delta_vs_select_fixed": interval["mean_difference"],
        "crossed_hierarchical_bootstrap": interval,
        "non_abstained_coverage": _round_float(overall_coverage),
        "blocks": block_results,
        "partition_audit": partition_audit,
        "oof_residual_audit": ordered_oof_audit,
        "oof_residual_audit_sha256": sha256_bytes(canonical_json_bytes(oof_audit_payload)),
        "query_group_fold_contract": (
            {
                "fold_count": QUERY_FOLD_COUNT,
                "assignment": (
                    "within each dataset, sort by "
                    "SHA256('bcc1-v1-query-fold|' + dataset_family_id + '|' + "
                    "independence_group_id), then rank mod 5"
                ),
            }
            if axis_field == "transition_family_id"
            else None
        ),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _select_transfer_gate(
    validated: ValidatedBCC1,
) -> Dict[str, Any]:
    if validated.confirmatory_gates is None:
        raise BCC1ValidationError("missing confirmatory gates")
    policies: Dict[str, Any] = {}
    for policy_name in ("regularized_logistic", "knn"):
        dataset_scheme = _select_structural_scheme(
            rows=validated.select_rows,
            feature_names=validated.feature_names,
            blocks=validated.blocks,
            policy_name=policy_name,
            axis_field="dataset_family_id",
            gates=validated.confirmatory_gates,
        )
        transition_scheme = _select_structural_scheme(
            rows=validated.select_rows,
            feature_names=validated.feature_names,
            blocks=validated.blocks,
            policy_name=policy_name,
            axis_field="transition_family_id",
            gates=validated.confirmatory_gates,
        )
        policies[policy_name] = {
            "leave_dataset_family_out": dataset_scheme,
            "leave_transition_family_out": transition_scheme,
            "robust_score": min(
                dataset_scheme["delta_vs_select_fixed"],
                transition_scheme["delta_vs_select_fixed"],
            ),
            "pass": dataset_scheme["pass"] and transition_scheme["pass"],
        }
    selected_method = sorted(
        policies,
        key=lambda name: (
            -policies[name]["robust_score"],
            0 if name == "regularized_logistic" else 1,
        ),
    )[0]
    manifest_method = validated.confirmatory_gates["primary_method"]
    selection_matches = selected_method == manifest_method
    return {
        "source": "SELECT outcomes only",
        "selected_method": selected_method,
        "manifest_primary_method": manifest_method,
        "selection_matches_frozen_manifest": selection_matches,
        "policies": policies,
        "pass": selection_matches and policies[selected_method]["pass"],
    }


def _method_dev_axis_transfer(
    validated: ValidatedBCC1,
) -> Dict[str, Any]:
    fixed_direction, _ = _fixed_direction(
        validated.method_dev_rows,
        validated.blocks,
    )
    diagnostic_gates = {
        "minimum_worthwhile_effect": 0.005,
        "minimum_overall_coverage": 0.10,
        "minimum_block_coverage": 0.05,
    }
    policies: Dict[str, Any] = {}
    for policy_name in ("regularized_logistic", "knn"):
        dataset_scheme = _select_structural_scheme(
            rows=validated.method_dev_rows,
            feature_names=validated.feature_names,
            blocks=validated.blocks,
            policy_name=policy_name,
            axis_field="dataset_family_id",
            gates=diagnostic_gates,
        )
        transition_scheme = _select_structural_scheme(
            rows=validated.method_dev_rows,
            feature_names=validated.feature_names,
            blocks=validated.blocks,
            policy_name=policy_name,
            axis_field="transition_family_id",
            gates=diagnostic_gates,
        )
        policies[policy_name] = {
            "leave_dataset_family_out": dataset_scheme,
            "crossed_transition_family_x_query_fold": transition_scheme,
            "passes_both_structural_schemes": (dataset_scheme["pass"] and transition_scheme["pass"]),
        }
    passing = [name for name, report in policies.items() if report["passes_both_structural_schemes"]]
    dual_read_blocked = bool(validated.dual_read_feature_names)
    reverse_role_blocked = validated.reverse_route_role_validation == UNVALIDATED_DOCUMENT_ROLE_REVERSE_PROXY
    operational_cut = _method_dev_cut_disposition(
        dual_read_blocked=dual_read_blocked,
        reverse_role_blocked=reverse_role_blocked,
    )
    return {
        "classification": "exploratory_non_promotional",
        "promotion_eligible": False,
        "best_fixed_direction": fixed_direction,
        "query_group_fold_count": QUERY_FOLD_COUNT,
        "passing_policies": passing,
        "disposition": (
            operational_cut
            if dual_read_blocked or reverse_role_blocked
            else ("RECONDITION_THEN_PREREGISTER" if passing else "CUT_CURRENT_FEATURE_POLICY_SET")
        ),
        "disposition_reason": (
            (
                "one or more policy features require both directional "
                "retrievals before selection and therefore cannot implement "
                "one-authorized-search routing"
            )
            if dual_read_blocked
            else (
                "the recovered reverse score applies a document-role bridge "
                "to query vectors without independent query-role validation"
                if reverse_role_blocked
                else (
                    "at least one current policy passed both METHOD_DEV structural schemes"
                    if passing
                    else (
                        "no current qrels-free policy passed both independence-safe "
                        "dataset and crossed transition/query-fold schemes; oracle "
                        "headroom alone is insufficient"
                    )
                )
            )
        ),
        "reverse_route_role_validation": validated.reverse_route_role_validation,
        "reverse_route_role_independent_cut": reverse_role_blocked,
        "dual_read_feature_names": list(validated.dual_read_feature_names),
        "policies": policies,
    }


def _confirmatory_promotion_gate(
    *,
    rows: Sequence[Mapping[str, Any]],
    primary_method: str,
    primary_status: str,
    primary_scores: np.ndarray,
    fixed_scores: np.ndarray,
    native_new_scores: np.ndarray,
    gates: Mapping[str, Any],
    blocks: Mapping[str, Mapping[str, Any]],
    abstentions: Sequence[bool],
    prereport_power: Mapping[str, Any],
    select_transfer: Mapping[str, Any],
) -> Dict[str, Any]:
    block_ids = tuple(row["block_id"] for row in rows)
    superiority_ci = crossed_hierarchical_paired_bootstrap_mean_ci(primary_scores, fixed_scores, rows, blocks)
    superiority_materiality_pass = (
        superiority_ci["mean_difference"] >= gates["minimum_worthwhile_effect"] - TIE_TOLERANCE
    )
    superiority_uncertainty_pass = superiority_ci["ci_lower"] > 0.0
    power = dict(prereport_power)
    native_new_ci = crossed_hierarchical_paired_bootstrap_mean_ci(primary_scores, native_new_scores, rows, blocks)
    native_new_pass = native_new_ci["ci_lower"] > -gates["native_new_noninferiority_margin"]

    worst_block_results = []
    block_count = len(set(block_ids))
    per_block_alpha = 0.05 / block_count
    per_block_confidence = 1.0 - 2.0 * per_block_alpha
    for block_id in sorted(set(block_ids)):
        indices = np.asarray(
            [index for index, observed_block in enumerate(block_ids) if observed_block == block_id],
            dtype=np.int64,
        )
        block_seed = int.from_bytes(
            hashlib.sha256(("bcc1-v1-safety|" + block_id).encode("utf-8")).digest()[:4],
            byteorder="big",
            signed=False,
        )
        block_ci = grouped_paired_bootstrap_mean_ci(
            primary_scores[indices],
            fixed_scores[indices],
            [rows[index]["independence_group_id"] for index in indices],
            seed=block_seed,
            confidence=per_block_confidence,
        )
        block_pass = block_ci["ci_lower"] > -gates["worst_block_noninferiority_margin"]
        worst_block_results.append(
            {
                "block_id": block_id,
                "query_count": int(indices.size),
                "pass": block_pass,
                "bonferroni_grouped_bootstrap_vs_best_fixed": block_ci,
            }
        )
    worst_block_pass = all(item["pass"] for item in worst_block_results)
    non_abstained = np.asarray([not bool(value) for value in abstentions], dtype=np.bool_)
    overall_coverage = float(np.mean(non_abstained))
    block_coverage = []
    for block_id in sorted(set(block_ids)):
        indices = np.asarray(
            [index for index, observed_block in enumerate(block_ids) if observed_block == block_id],
            dtype=np.int64,
        )
        coverage = float(np.mean(non_abstained[indices]))
        block_coverage.append(
            {
                "block_id": block_id,
                "coverage": _round_float(coverage),
                "pass": coverage + TIE_TOLERANCE >= gates["minimum_block_coverage"],
            }
        )
    coverage_pass = overall_coverage + TIE_TOLERANCE >= gates["minimum_overall_coverage"] and all(
        item["pass"] for item in block_coverage
    )
    method_evaluated_pass = primary_status == "evaluated"
    checks = {
        "separated_decision_outcome_boundary": False,
        "mechanical_universe_role_recomputation": False,
        "independent_feature_generation_attestation": False,
        "abi_operational_conformance_evidence": False,
        "canonical_replay_evidence": False,
        "select_structural_transfer": bool(select_transfer["pass"]),
        "primary_method_evaluated": method_evaluated_pass,
        "mwee_materiality": superiority_materiality_pass,
        "superiority_uncertainty": superiority_uncertainty_pass,
        "mwee_power_adequacy": bool(power["pass"]),
        "native_new_noninferiority": native_new_pass,
        "worst_block_safety": worst_block_pass,
        "non_abstained_coverage": coverage_pass,
    }
    failures = [name for name, passed in checks.items() if not passed]
    eligible = not failures
    return {
        "decision": "ELIGIBLE" if eligible else "BLOCKED",
        "promotion_eligible": eligible,
        "one_shot_report": True,
        "primary_method": primary_method,
        "decision_rule": "all preregistered gates must pass; no oracle-headroom gate",
        "checks": checks,
        "failed_checks": failures,
        "protocol_status": "BLOCKED_PROTOCOL",
        "protocol_blockers": [
            (
                "CONFIRMATORY decisions and sealed REPORT outcomes do not yet "
                "have separate SHA-bound schemas and evaluator inputs"
            ),
            ("eligible-universe mechanical role allocation is not " "recomputed from a precommitted universe artifact"),
            (
                "feature digests bind declarations but no independent "
                "generation attestation/reproduction artifact is verified"
            ),
            (
                "ABI-valid fallback execution and zero wrong-space-search "
                "adversarial evidence are not bound into this report"
            ),
            "canonical replay evidence cannot exist before the report hash is written",
        ],
        "required_upstream_evidence": {
            "status": "unverified",
            "promotion_gate_forced_blocked": True,
            "artifacts": [
                "representation_manifest_sha256",
                "resident_and_materialized_index_manifest_sha256",
                "external_anchor_manifest_sha256",
                "bridge_contract_manifest_sha256",
                "external_anchor_validation_report_sha256",
                "feature_artifact_and_independent_attestation_sha256",
                "environment_lock_sha256",
                "exact_runner_sha256",
                "eligible_universe_and_role_allocation_sha256",
                "qrels_identity_sha256",
                "independent_evaluator_identity_sha256",
                "frozen_policy_sha256",
                "frozen_report_decisions_sha256",
                "resolver_adversarial_conformance_report_sha256",
                "canonical_replay_report_sha256",
            ],
        },
        "select_structural_transfer": dict(select_transfer),
        "superiority_vs_best_fixed": {
            "minimum_worthwhile_effect": _round_float(gates["minimum_worthwhile_effect"]),
            "materiality_pass": superiority_materiality_pass,
            "uncertainty_pass": superiority_uncertainty_pass,
            "hierarchical_bootstrap": superiority_ci,
        },
        "power_and_precision": power,
        "native_new_noninferiority": {
            "margin": _round_float(gates["native_new_noninferiority_margin"]),
            "pass": native_new_pass,
            "hierarchical_bootstrap": native_new_ci,
        },
        "worst_block_safety": {
            "noninferiority_margin": _round_float(gates["worst_block_noninferiority_margin"]),
            "pass": worst_block_pass,
            "familywise_confidence": 0.95,
            "per_block_alpha": _round_float(per_block_alpha),
            "blocks": worst_block_results,
        },
        "non_abstained_coverage": {
            "overall": _round_float(overall_coverage),
            "minimum_overall": _round_float(gates["minimum_overall_coverage"]),
            "minimum_per_block": _round_float(gates["minimum_block_coverage"]),
            "pass": coverage_pass,
            "blocks": block_coverage,
        },
    }


def _analyze_combined_blocked_protocol_diagnostic(
    pack: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    pack_raw_sha256: Optional[str] = None,
    manifest_raw_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a combined fixture as BLOCKED_PROTOCOL diagnostic evidence."""

    validated = _validate_pack_and_manifest_common(
        pack,
        manifest,
        pack_raw_sha256=pack_raw_sha256,
        manifest_raw_sha256=manifest_raw_sha256,
        enforce_confirmatory_universe=False,
    )
    if validated.evidence_mode == "METHOD_DEV":
        return _analyze_method_dev(validated)

    fixed_direction, select_direction_means = _fixed_direction(
        validated.select_rows,
        validated.blocks,
    )
    select_transfer = _select_transfer_gate(validated)

    # Freeze every REPORT decision using only SELECT outcomes plus REPORT
    # qrels-free features before reading REPORT outcomes below.
    x_select, y_select, training_query_ids, tie_count = _direction_training_data(
        validated.select_rows, validated.feature_names
    )
    x_report = _matrix(validated.report_rows, validated.feature_names)
    class_counts = {
        "forward": int(np.sum(y_select == 0.0)),
        "reverse": int(np.sum(y_select == 1.0)),
        "ties_excluded": tie_count,
    }

    logistic_model, logistic_reason = _fit_logistic(x_select, y_select)
    if logistic_model is None:
        logistic_probabilities: Optional[np.ndarray] = None
        logistic_directions = tuple(fixed_direction for _ in validated.report_rows)
        logistic_abstentions = tuple(True for _ in validated.report_rows)
        logistic_status = "fallback_only"
        logistic_training: Dict[str, Any] = {
            "class_counts": class_counts,
            "l2": LOGISTIC_L2,
            "training_count": int(y_select.size),
        }
    else:
        logistic_probabilities = _predict_logistic(logistic_model, x_report)
        logistic_directions, logistic_abstentions = _directions_from_probabilities(
            logistic_probabilities, fixed_direction
        )
        logistic_status = "evaluated"
        logistic_reason = None
        logistic_training = {
            "class_counts": class_counts,
            "l2": LOGISTIC_L2,
            "training_count": int(y_select.size),
            "iterations": logistic_model.iterations,
            "coefficients": [_round_float(value) for value in logistic_model.coefficients],
            "feature_mean": [_round_float(value) for value in logistic_model.mean],
            "feature_scale": [_round_float(value) for value in logistic_model.scale],
        }

    knn_model, knn_reason = _fit_knn(x_select, y_select, training_query_ids)
    if knn_model is None:
        knn_probabilities: Optional[np.ndarray] = None
        knn_directions = tuple(fixed_direction for _ in validated.report_rows)
        knn_abstentions = tuple(True for _ in validated.report_rows)
        knn_status = "fallback_only"
    else:
        knn_probabilities = _predict_knn(knn_model, x_report)
        knn_directions, knn_abstentions = _directions_from_probabilities(knn_probabilities, fixed_direction)
        knn_status = "evaluated"
        knn_reason = None

    if validated.confirmatory_gates is None:
        raise BCC1ValidationError("CONFIRMATORY analysis requires validated confirmatory_gates")
    primary_method = validated.confirmatory_gates["primary_method"]
    power_structural_scheme = select_transfer["policies"][primary_method]["leave_transition_family_out"]
    prereport_power = _select_power_gate(
        policy_name=primary_method,
        structural_scheme=power_structural_scheme,
        report_rows=validated.report_rows,
        blocks=validated.blocks,
        gates=validated.confirmatory_gates,
    )

    # REPORT outcomes become visible only after the preceding decisions exist.
    report_reverse = np.asarray(
        [row["scores"]["reverse"] for row in validated.report_rows],
        dtype=np.float64,
    )
    report_forward = np.asarray(
        [row["scores"]["forward"] for row in validated.report_rows],
        dtype=np.float64,
    )
    report_native_new = np.asarray(
        [row["scores"]["native_new"] for row in validated.report_rows],
        dtype=np.float64,
    )
    report_block_ids = tuple(row["block_id"] for row in validated.report_rows)
    fixed_scores = report_reverse if fixed_direction == "reverse" else report_forward
    oracle_scores = np.maximum(report_reverse, report_forward)
    if "native_old" in validated.score_names:
        report_native_old: Optional[np.ndarray] = np.asarray(
            [row["scores"]["native_old"] for row in validated.report_rows],
            dtype=np.float64,
        )
        native_references: Dict[str, Any] = {
            "native_old_status": "provided",
            "native_old_report_mean": _mean(report_native_old),
            "native_new_report_mean": _mean(report_native_new),
            "native_new_minus_old": _round_float(float(np.mean(report_native_new - report_native_old))),
            "paired_bootstrap_native_new_vs_old": paired_bootstrap_mean_ci(report_native_new, report_native_old),
            "hierarchical_bootstrap_native_new_vs_old": (
                hierarchical_paired_bootstrap_mean_ci(report_native_new, report_native_old, report_block_ids)
            ),
        }
    else:
        native_references = {
            "native_old_status": "not_provided",
            "native_old_report_mean": None,
            "native_new_report_mean": _mean(report_native_new),
            "native_new_minus_old": None,
            "paired_bootstrap_native_new_vs_old": None,
            "hierarchical_bootstrap_native_new_vs_old": None,
        }

    logistic_report = _method_report(
        name="regularized_logistic_direction_classifier",
        rows=validated.report_rows,
        directions=logistic_directions,
        probabilities=logistic_probabilities,
        abstentions=logistic_abstentions,
        status=logistic_status,
        reason=logistic_reason,
        fixed_scores=fixed_scores,
        oracle_scores=oracle_scores,
        training=logistic_training,
        blocks=validated.blocks,
    )
    knn_report = _method_report(
        name="knn_direction_classifier",
        rows=validated.report_rows,
        directions=knn_directions,
        probabilities=knn_probabilities,
        abstentions=knn_abstentions,
        status=knn_status,
        reason=knn_reason,
        fixed_scores=fixed_scores,
        oracle_scores=oracle_scores,
        training={
            "class_counts": class_counts,
            "k": KNN_K,
            "training_count": int(y_select.size),
        },
        blocks=validated.blocks,
    )
    soft_fusion_report = _soft_fusion_report(
        validated,
        fixed_scores=fixed_scores,
        oracle_scores=oracle_scores,
    )
    primary_inputs = {
        "regularized_logistic": (
            logistic_report["status"],
            _scores_for_directions(validated.report_rows, logistic_directions),
            logistic_abstentions,
        ),
        "knn": (
            knn_report["status"],
            _scores_for_directions(validated.report_rows, knn_directions),
            knn_abstentions,
        ),
    }
    primary_status, primary_scores, primary_abstentions = primary_inputs[primary_method]
    promotion_gate = _confirmatory_promotion_gate(
        rows=validated.report_rows,
        primary_method=primary_method,
        primary_status=primary_status,
        primary_scores=primary_scores,
        fixed_scores=fixed_scores,
        native_new_scores=report_native_new,
        gates=validated.confirmatory_gates,
        blocks=validated.blocks,
        abstentions=primary_abstentions,
        prereport_power=prereport_power,
        select_transfer=select_transfer,
    )

    report: Dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "pack_id": validated.pack_id,
        "input_fingerprints": {
            "pack_sha256": validated.pack_sha256,
            "manifest_sha256": validated.manifest_sha256,
        },
        "evidence_classification": {
            "evidence_mode": "CONFIRMATORY",
            "sampled": False,
            "promotion_eligible": promotion_gate["promotion_eligible"],
            "confirmatory_report_evaluated": True,
            "reason": (
                "all preregistered one-shot gates passed"
                if promotion_gate["promotion_eligible"]
                else "one or more preregistered one-shot gates failed"
            ),
        },
        "protocol": {
            "select_report_separation": "fit/choose on SELECT; predict from REPORT qrels-free features; evaluate REPORT outcomes last",
            "bootstrap": {
                "confidence": BOOTSTRAP_CONFIDENCE,
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
            },
            "logistic_l2": LOGISTIC_L2,
            "knn_k": KNN_K,
            "abstain_confidence": ABSTAIN_CONFIDENCE,
            "tie_tolerance": TIE_TOLERANCE,
            "hierarchical_estimand": ("equal-weighted declared structural REPORT block means"),
            "confirmatory_gates": dict(validated.confirmatory_gates),
        },
        "data_contract": {
            "metric": validated.metric_name,
            "metric_contract": dict(validated.metric_contract),
            "metric_range": [
                _round_float(validated.metric_minimum),
                _round_float(validated.metric_maximum),
            ],
            "reverse_route_role_contract": {
                "validation_state": validated.reverse_route_role_validation,
                "confirmatory_eligible": True,
                "independent_cut": None,
            },
            "feature_names": list(validated.feature_names),
            "select_query_count": len(validated.select_rows),
            "report_query_count": len(validated.report_rows),
            "report_consumed_by_this_analysis": True,
        },
        "best_fixed": {
            "selection_rule": (
                "higher equal-weighted SELECT dataset-family x " "transition-family cell mean; reverse wins exact ties"
            ),
            "selected_direction": fixed_direction,
            "select_means": select_direction_means,
            "report_mean_score": _mean(fixed_scores),
        },
        "direction_oracle": {
            "definition": (
                "descriptive ex-post per-query max(reverse_score, forward_score); "
                "not a routeability or promotion estimand"
            ),
            "report_mean_score": _mean(oracle_scores),
            "descriptive_ex_post_choice_regret_upper_bound": _round_float(float(np.mean(oracle_scores - fixed_scores))),
        },
        "native_references": native_references,
        "direction_contrast": {
            "reverse_minus_forward_report_mean": _round_float(float(np.mean(report_reverse - report_forward))),
            "paired_bootstrap_reverse_vs_forward": paired_bootstrap_mean_ci(report_reverse, report_forward),
            "hierarchical_bootstrap_reverse_vs_forward": (
                hierarchical_paired_bootstrap_mean_ci(report_reverse, report_forward, report_block_ids)
            ),
        },
        "promotion_gate": promotion_gate,
        "methods": {
            "regularized_logistic": logistic_report,
            "knn": knn_report,
            "soft_score_fusion": soft_fusion_report,
        },
    }
    # A final encode is an executable assertion that no non-finite or
    # implementation-specific object escaped into the artifact.
    canonical_json_bytes(report)
    return report


def analyze_bcc1(
    pack: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    pack_raw_sha256: Optional[str] = None,
    manifest_raw_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze METHOD_DEV only; CONFIRMATORY access requires the file runner."""

    if isinstance(pack, Mapping) and pack.get("evidence_mode") == "CONFIRMATORY":
        raise BCC1ValidationError(
            "CONFIRMATORY REPORT cannot be evaluated in memory; use "
            "run_bcc1_files for lock-bound one-shot consumption"
        )
    validated = validate_pack_and_manifest(
        pack,
        manifest,
        pack_raw_sha256=pack_raw_sha256,
        manifest_raw_sha256=manifest_raw_sha256,
    )
    if validated.evidence_mode == "CONFIRMATORY":
        raise BCC1ValidationError(
            "CONFIRMATORY REPORT cannot be evaluated in memory; use "
            "run_bcc1_files for lock-bound one-shot consumption"
        )
    return _analyze_method_dev(validated)


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=path.name + ".",
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(str(temporary_path), str(path))
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def _output_identity_sha256(path: Path) -> str:
    canonical_path = os.path.normcase(str(path.resolve()))
    return sha256_bytes(
        canonical_json_bytes(
            {
                "kind": "bcc1-canonical-report-output-v1",
                "absolute_path": canonical_path,
            }
        )
    )


def _create_preopen_claim(
    lock_path: Path,
    claim: Mapping[str, Any],
) -> None:
    """Durably acquire the one-shot claim before any pack byte is opened."""

    try:
        lock_descriptor = os.open(
            str(lock_path),
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError as exc:
        raise BCC1ValidationError("BCC-1 manifest is locked or protocol-blocked: {}".format(lock_path)) from exc
    try:
        raw_claim = canonical_json_bytes(claim, trailing_newline=True)
        os.write(lock_descriptor, raw_claim)
        os.fsync(lock_descriptor)
    finally:
        os.close(lock_descriptor)


def run_bcc1_files(
    *,
    pack_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Run METHOD_DEV or write a durable pre-open CONFIRMATORY block claim.

    The function never creates or regenerates a pack.  Combined CONFIRMATORY
    packs are never opened; the public manifest envelope is claimed first and
    the call fails ``BLOCKED_PROTOCOL_PREOPEN``.
    """

    pack_path = Path(pack_path)
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    if not manifest_path.is_file():
        raise BCC1ValidationError("manifest does not exist: {}".format(manifest_path))
    try:
        public_manifest_raw = manifest_path.read_bytes()
    except OSError as exc:
        raise BCC1ValidationError("unable to read public BCC-1 manifest envelope: {}".format(exc)) from exc
    public_manifest = decode_json_object(public_manifest_raw, artifact_name="manifest")
    public_blocks = public_manifest.get("blocks")
    safe_method_dev_envelope = (
        public_manifest.get("evidence_mode") == "METHOD_DEV"
        and public_manifest.get("sampled") is True
        and isinstance(public_blocks, list)
        and bool(public_blocks)
        and all(
            isinstance(block, dict) and block.get("role") == "METHOD_DEV" and block.get("consumed") is False
            for block in public_blocks
        )
    )
    lock_path = manifest_path.with_name(manifest_path.name + ".bcc1.lock")
    if not safe_method_dev_envelope:
        _create_preopen_claim(
            lock_path,
            {
                "schema_version": "chelatedai.bcc1.lock.v1",
                "state": "BLOCKED_PROTOCOL_PREOPEN",
                "manifest_sha256": sha256_bytes(public_manifest_raw),
                "pack_path_identity_sha256": _output_identity_sha256(pack_path),
                "output_identity_sha256": _output_identity_sha256(output_path),
                "report_outcomes_opened": False,
                "reason": (
                    "BCC-1 v1 requires separate SHA-bound frozen decisions "
                    "and sealed REPORT outcomes plus operational evidence; "
                    "the single-pack evaluator is disabled"
                ),
            },
        )
        raise BCC1ValidationError(
            "BLOCKED_PROTOCOL_PREOPEN: CONFIRMATORY evaluation requires "
            "separate SHA-bound decisions/outcomes and verified operational "
            "evidence; no pack bytes were read"
        )
    if not pack_path.is_file():
        raise BCC1ValidationError("frozen pack does not exist: {}".format(pack_path))
    resolved_paths = {
        pack_path.resolve(),
        manifest_path.resolve(),
        output_path.resolve(),
    }
    if len(resolved_paths) != 3:
        raise BCC1ValidationError("pack, manifest, and output paths must be distinct")
    if output_path.exists():
        raise BCC1ValidationError("refusing to overwrite existing report artifact: {}".format(output_path))

    claim_base = {
        "schema_version": "chelatedai.bcc1.lock.v1",
        "manifest_sha256": sha256_bytes(public_manifest_raw),
        "declared_pack_sha256": public_manifest.get("pack_sha256"),
        "pack_path_identity_sha256": _output_identity_sha256(pack_path),
        "output_identity_sha256": _output_identity_sha256(output_path),
    }
    _create_preopen_claim(
        lock_path,
        {
            **claim_base,
            "state": "CLAIMED_UNCLASSIFIED_PREOPEN",
            "consumption_state": "CONSUMED_ON_CLAIM",
            "pack_bytes_opened": False,
            "report_outcomes_opened": None,
            "reason": (
                "a METHOD_DEV-looking public envelope is not trusted until "
                "the claimed pack is opened and strictly validated; the "
                "one-shot claim prevents an unaccounted restricted-pack read"
            ),
        },
    )

    initial_pack_raw: Optional[bytes] = None
    try:
        initial_pack_raw = pack_path.read_bytes()
        initial_pack = decode_json_object(initial_pack_raw, artifact_name="pack")
        if initial_pack.get("evidence_mode") != "METHOD_DEV":
            raise BCC1ValidationError(
                "public METHOD_DEV envelope does not match pack.evidence_mode; "
                "restricted pack was consumed under the durable pre-open claim"
            )
        report = analyze_bcc1(
            initial_pack,
            public_manifest,
            pack_raw_sha256=sha256_bytes(initial_pack_raw),
            manifest_raw_sha256=sha256_bytes(public_manifest_raw),
        )
        _atomic_write(output_path, canonical_json_bytes(report, trailing_newline=True))
    except Exception as exc:
        failure_claim = {
            **claim_base,
            "state": "BLOCKED_AFTER_CLAIM",
            "consumption_state": "CONSUMED_ON_CLAIM",
            "pack_bytes_opened": initial_pack_raw is not None,
            "report_outcomes_opened": initial_pack_raw is not None,
            "observed_pack_sha256": (sha256_bytes(initial_pack_raw) if initial_pack_raw is not None else None),
            "reason": (
                "pack classification or strict validation failed after the "
                "durable one-shot pre-open claim; outcome exposure is treated "
                "conservatively as consumed"
            ),
            "failure_type": type(exc).__name__,
        }
        try:
            _atomic_write(
                lock_path,
                canonical_json_bytes(failure_claim, trailing_newline=True),
            )
        except OSError as claim_exc:
            raise BCC1ValidationError(
                "BCC-1 failed after pre-open claim and the blocked claim " "could not be finalized: {}".format(
                    claim_exc
                )
            ) from exc
        raise

    _atomic_write(
        lock_path,
        canonical_json_bytes(
            {
                **claim_base,
                "state": "METHOD_DEV_COMPLETE",
                "consumption_state": "NON_CONFIRMATORY_METHOD_DEV_COMPLETE",
                "pack_bytes_opened": True,
                "report_outcomes_opened": False,
                "observed_pack_sha256": sha256_bytes(initial_pack_raw),
                "output_sha256": sha256_bytes(output_path.read_bytes()),
            },
            trailing_newline=True,
        ),
    )
    return report


__all__ = [
    "ABSTAIN_CONFIDENCE",
    "BCC1ValidationError",
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "KNN_K",
    "LOGISTIC_L2",
    "MANIFEST_SCHEMA",
    "PACK_SCHEMA",
    "REPORT_SCHEMA",
    "analyze_bcc1",
    "canonical_json_bytes",
    "crossed_hierarchical_paired_bootstrap_mean_ci",
    "decode_json_object",
    "grouped_paired_bootstrap_mean_ci",
    "hierarchical_paired_bootstrap_mean_ci",
    "paired_bootstrap_mean_ci",
    "query_ids_sha256",
    "run_bcc1_files",
    "sha256_bytes",
    "validate_pack_and_manifest",
]
