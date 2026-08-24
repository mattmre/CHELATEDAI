"""Fail-closed routing for searches across versioned representation spaces.

The resolver returns an explicit plan.  It never treats equal vector dimensions
as compatibility and never authorizes a search after an unresolved mismatch.
Transform execution and per-query machine-learned routing are deliberately outside
this module; a scorer protocol allows the latter to be injected without weakening
the deterministic safety boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple

from representation_space import (
    BridgeContract,
    BridgeCost,
    BridgeValidationState,
    RepresentationDescriptor,
    RepresentationRole,
    ResidentIndexDescriptor,
)

ROUTE_SCORER_CONTRACT_VERSION = 1
ROUTE_SELECTION_RULE_ID = "effective-confidence-v1"
MATERIALIZATION_CONTRACT_VERSION = 1


class ResolutionError(ValueError):
    """Raised for invalid resolver configuration or use of an abstained plan."""


class ResolutionMode(str, Enum):
    """Executable semantic-cache routing modes."""

    NATIVE = "native"
    REVERSE = "reverse"
    FORWARD = "forward"
    ABSTAIN = "abstain"


class ResolutionReason(str, Enum):
    """Stable reason codes for routing telemetry and policy enforcement."""

    IDENTICAL_SPACE = "identical_space"
    VALIDATED_REVERSE_BRIDGE = "validated_reverse_bridge"
    VALIDATED_FORWARD_BRIDGE = "validated_forward_bridge"
    NO_REPRESENTATION_BRIDGE = "no_representation_bridge"
    NO_VALIDATED_BRIDGE = "no_validated_bridge"
    INVALID_BRIDGE_COST = "invalid_bridge_cost"
    MATERIALIZED_INDEX_UNAVAILABLE = "materialized_index_unavailable"
    LOW_ROUTE_CONFIDENCE = "low_route_confidence"
    ROUTER_FAILURE = "router_failure"
    ROUTER_REQUIRED = "router_required"
    UNTRUSTED_BRIDGE_EVIDENCE = "untrusted_bridge_evidence"
    UNTRUSTED_BRIDGE_CONTRACT = "untrusted_bridge_contract"
    UNTRUSTED_MATERIALIZATION_CONTRACT = "untrusted_materialization_contract"


def _opaque_id(value: object, field_name: str, *, maximum_length: int = 1024) -> str:
    if not isinstance(value, str):
        raise ResolutionError(f"{field_name} must be a string")
    if value != value.strip():
        raise ResolutionError(f"{field_name} must not contain leading or trailing whitespace")
    normalized = unicodedata.normalize("NFC", value)
    if not normalized:
        raise ResolutionError(f"{field_name} must not be empty")
    if len(normalized) > maximum_length:
        raise ResolutionError(f"{field_name} must be at most {maximum_length} characters")
    if any(unicodedata.category(character).startswith("C") for character in normalized):
        raise ResolutionError(f"{field_name} must not contain control characters")
    return normalized


def _score(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResolutionError(f"{field_name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0 or converted > 1.0:
        raise ResolutionError(f"{field_name} must be in the closed interval [0, 1]")
    return converted


def _sha256_digest(value: object, field_name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ResolutionError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _content_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_json_value(value: object, path: str = "value") -> object:
    """Return an unaliased, deterministic JSON value or reject ambiguity."""

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResolutionError(f"{path} must not contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        normalized: Dict[str, object] = {}
        for raw_key, raw_value in value.items():
            key = _opaque_id(raw_key, f"{path} key")
            if key in normalized:
                raise ResolutionError(f"{path} contains a duplicate canonical key: {key}")
            normalized[key] = _canonical_json_value(raw_value, f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise ResolutionError(f"{path} must contain only JSON scalar, mapping, list, or tuple values")


_FEATURE_VALUE_TYPES = frozenset({"bool", "float", "int", "str"})
_ROUTE_FEATURE_LEAKAGE_FRAGMENTS = (
    "groundtruth",
    "judgment",
    "label",
    "ndcg",
    "qrel",
    "relevance",
    "reward",
    "winner",
)
_ROUTE_FEATURE_LEAKAGE_TOKENS = frozenset(
    {
        "clicked",
        "gold",
        "map",
        "mrr",
        "outcome",
        "precision",
        "rank",
        "recall",
        "result",
        "success",
    }
)


@dataclass(frozen=True)
class RouteFeatureDeclaration:
    """Content-addressed provenance and execution policy for one route feature."""

    name: str
    type: str
    qrels_free: bool
    availability_stage: str
    execution_cost_class: str
    implementation_sha256: str
    provenance_sha256: str

    def __post_init__(self) -> None:
        name = _opaque_id(self.name, "feature name")
        value_type = _opaque_id(self.type, f"feature type for {name}")
        if value_type not in _FEATURE_VALUE_TYPES:
            raise ResolutionError(f"feature type for {name} must be one of: " + ", ".join(sorted(_FEATURE_VALUE_TYPES)))
        if type(self.qrels_free) is not bool:
            raise ResolutionError(f"qrels_free for feature {name} must be a boolean")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "type", value_type)
        object.__setattr__(
            self,
            "availability_stage",
            _opaque_id(
                self.availability_stage,
                f"availability_stage for feature {name}",
            ),
        )
        object.__setattr__(
            self,
            "execution_cost_class",
            _opaque_id(
                self.execution_cost_class,
                f"execution_cost_class for feature {name}",
            ),
        )
        for field_name in ("implementation_sha256", "provenance_sha256"):
            object.__setattr__(
                self,
                field_name,
                _sha256_digest(
                    getattr(self, field_name),
                    f"{field_name} for feature {name}",
                ),
            )

    def canonical_payload(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "type": self.type,
            "qrels_free": self.qrels_free,
            "availability_stage": self.availability_stage,
            "execution_cost_class": self.execution_cost_class,
            "implementation_sha256": self.implementation_sha256,
            "provenance_sha256": self.provenance_sha256,
        }


def _normalize_feature_schema(
    feature_schema: object,
) -> Tuple[RouteFeatureDeclaration, ...]:
    if isinstance(feature_schema, (str, bytes)) or not isinstance(feature_schema, Sequence):
        raise ResolutionError("feature_schema must be a sequence of RouteFeatureDeclaration values")
    normalized: Dict[str, RouteFeatureDeclaration] = {}
    for item in feature_schema:
        if not isinstance(item, RouteFeatureDeclaration):
            raise ResolutionError("feature_schema entries must be RouteFeatureDeclaration values")
        if item.name in normalized:
            raise ResolutionError(f"duplicate feature schema entry: {item.name}")
        normalized[item.name] = item
    return tuple(normalized[name] for name in sorted(normalized))


def _route_feature_name_tokens(name: str) -> Tuple[str, ...]:
    camel_separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return tuple(token for token in re.split(r"[^a-z0-9]+", camel_separated.casefold()) if token)


def _validate_route_feature_policy(
    feature_schema: Tuple[RouteFeatureDeclaration, ...],
) -> None:
    for declaration in feature_schema:
        name = declaration.name
        if declaration.qrels_free is not True:
            raise ResolutionError(f"route feature {name} must declare qrels_free=true")
        if declaration.availability_stage != "PRE_SEARCH":
            raise ResolutionError(f"route feature {name} must have availability_stage PRE_SEARCH")
        if declaration.execution_cost_class != "zero_search":
            raise ResolutionError(f"route feature {name} must have execution_cost_class zero_search")
        tokens = _route_feature_name_tokens(name)
        compact_name = "".join(tokens)
        forbidden = sorted(
            {token for token in tokens if token in _ROUTE_FEATURE_LEAKAGE_TOKENS}
            | {fragment for fragment in _ROUTE_FEATURE_LEAKAGE_FRAGMENTS if fragment in compact_name}
        )
        if forbidden:
            raise ResolutionError(f"route feature {name} contains forbidden leakage token: " + ", ".join(forbidden))


def route_feature_manifest_sha256(feature_schema: object) -> str:
    """Return the content digest for an exact typed route-feature allowlist."""

    normalized = _normalize_feature_schema(feature_schema)
    return _content_sha256(
        {
            "record_type": "route_feature_manifest",
            "features": [declaration.canonical_payload() for declaration in normalized],
        }
    )


def route_confidence_rule_sha256(minimum_route_confidence: object) -> str:
    """Digest the executable threshold, confidence calculation, and tie rule."""

    minimum = _score(
        minimum_route_confidence,
        "minimum_route_confidence",
    )
    return _content_sha256(
        {
            "record_type": "route_confidence_rule",
            "selection_rule_id": ROUTE_SELECTION_RULE_ID,
            "minimum_route_confidence": minimum,
            "effective_confidence": "min(validation_confidence,route_score)",
            "qualification": "effective_confidence>=minimum_route_confidence",
            "ranking": [
                "effective_confidence_desc",
                "zero_write_reverse_before_materialized_forward",
                "bridge_stable_id_asc",
            ],
        }
    )


def _feature_schema_from_payload(
    raw: object,
) -> Tuple[RouteFeatureDeclaration, ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ResolutionError("feature_schema payload must be a sequence")
    declarations = []
    required_fields = {
        "name",
        "type",
        "qrels_free",
        "availability_stage",
        "execution_cost_class",
        "implementation_sha256",
        "provenance_sha256",
    }
    for entry in raw:
        if not isinstance(entry, Mapping) or set(entry) != required_fields:
            raise ResolutionError(
                "feature_schema payload entries require exactly "
                "name, type, qrels_free, availability_stage, "
                "execution_cost_class, implementation_sha256, and "
                "provenance_sha256"
            )
        declarations.append(
            RouteFeatureDeclaration(
                name=entry["name"],
                type=entry["type"],
                qrels_free=entry["qrels_free"],
                availability_stage=entry["availability_stage"],
                execution_cost_class=entry["execution_cost_class"],
                implementation_sha256=entry["implementation_sha256"],
                provenance_sha256=entry["provenance_sha256"],
            )
        )
    return _normalize_feature_schema(declarations)


def _validate_route_query_context(
    raw_context: object,
    feature_schema: Tuple[RouteFeatureDeclaration, ...],
) -> Dict[str, object]:
    canonical = _canonical_json_value(raw_context, "query_context")
    if not isinstance(canonical, dict):
        raise ResolutionError("query_context must be a mapping")
    required_context_keys = {"features", "query_input"}
    if set(canonical) != required_context_keys:
        raise ResolutionError("query_context keys must be exactly: features, query_input")
    query_input = canonical["query_input"]
    if not isinstance(query_input, str):
        raise ResolutionError("query_context.query_input must be a string")
    features = canonical["features"]
    if not isinstance(features, dict):
        raise ResolutionError("query_context.features must be a mapping")
    expected_types = {declaration.name: declaration.type for declaration in feature_schema}
    if set(features) != set(expected_types):
        missing = set(expected_types) - set(features)
        extra = set(features) - set(expected_types)
        detail = []
        if missing:
            detail.append("missing: " + ", ".join(sorted(missing)))
        if extra:
            detail.append("extra: " + ", ".join(sorted(extra)))
        raise ResolutionError(
            "query_context.features must exactly match the contract allowlist"
            + (" (" + "; ".join(detail) + ")" if detail else "")
        )
    expected_python_types = {
        "bool": bool,
        "float": float,
        "int": int,
        "str": str,
    }
    for declaration in feature_schema:
        value = features[declaration.name]
        if type(value) is not expected_python_types[declaration.type]:
            raise ResolutionError(f"query feature {declaration.name} must have exact type " f"{declaration.type}")
    return canonical


@dataclass(frozen=True)
class RouteScorerContract:
    """Immutable content-addressed authority for a per-query route policy."""

    policy_id: str
    scorer_stable_id: str
    model_sha256: str
    scaler_sha256: str
    feature_manifest_sha256: str
    feature_schema: Tuple[RouteFeatureDeclaration, ...]
    confidence_rule_sha256: str
    minimum_route_confidence: float
    implementation_sha256: str
    evidence_id: str
    evidence_sha256: str
    selection_rule_id: str = ROUTE_SELECTION_RULE_ID
    contract_version: int = ROUTE_SCORER_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.contract_version, bool)
            or not isinstance(self.contract_version, int)
            or self.contract_version != ROUTE_SCORER_CONTRACT_VERSION
        ):
            raise ResolutionError(
                "unsupported route scorer contract version "
                f"{self.contract_version!r}; runtime supports "
                f"{ROUTE_SCORER_CONTRACT_VERSION}"
            )
        object.__setattr__(self, "policy_id", _opaque_id(self.policy_id, "policy_id"))
        object.__setattr__(
            self,
            "scorer_stable_id",
            _opaque_id(self.scorer_stable_id, "scorer_stable_id"),
        )
        selection_rule_id = _opaque_id(
            self.selection_rule_id,
            "selection_rule_id",
        )
        if selection_rule_id != ROUTE_SELECTION_RULE_ID:
            raise ResolutionError(f"unsupported route selection rule: {selection_rule_id}")
        object.__setattr__(self, "selection_rule_id", selection_rule_id)
        minimum_route_confidence = _score(
            self.minimum_route_confidence,
            "minimum_route_confidence",
        )
        object.__setattr__(
            self,
            "minimum_route_confidence",
            minimum_route_confidence,
        )
        object.__setattr__(
            self,
            "evidence_id",
            _opaque_id(self.evidence_id, "evidence_id", maximum_length=1024),
        )
        for field_name in (
            "model_sha256",
            "scaler_sha256",
            "feature_manifest_sha256",
            "confidence_rule_sha256",
            "implementation_sha256",
            "evidence_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _sha256_digest(getattr(self, field_name), field_name),
            )
        normalized_feature_schema = _normalize_feature_schema(self.feature_schema)
        object.__setattr__(self, "feature_schema", normalized_feature_schema)
        expected_manifest_sha256 = route_feature_manifest_sha256(normalized_feature_schema)
        if self.feature_manifest_sha256 != expected_manifest_sha256:
            raise ResolutionError("feature_manifest_sha256 does not match feature_schema")
        _validate_route_feature_policy(normalized_feature_schema)
        expected_confidence_rule_sha256 = route_confidence_rule_sha256(minimum_route_confidence)
        if self.confidence_rule_sha256 != expected_confidence_rule_sha256:
            raise ResolutionError("confidence_rule_sha256 does not match threshold and selection rule")
        expected_scorer_id = (
            f"route-scorer-implementation:v{self.contract_version}:" f"sha256:{self.implementation_sha256}"
        )
        if self.scorer_stable_id != expected_scorer_id:
            raise ResolutionError("scorer_stable_id does not bind implementation_sha256")

    def canonical_payload(self) -> Dict[str, object]:
        return {
            "record_type": "route_scorer_contract",
            "contract_version": self.contract_version,
            "policy_id": self.policy_id,
            "scorer_stable_id": self.scorer_stable_id,
            "model_sha256": self.model_sha256,
            "scaler_sha256": self.scaler_sha256,
            "feature_manifest_sha256": self.feature_manifest_sha256,
            "feature_schema": [declaration.canonical_payload() for declaration in self.feature_schema],
            "confidence_rule_sha256": self.confidence_rule_sha256,
            "minimum_route_confidence": self.minimum_route_confidence,
            "selection_rule_id": self.selection_rule_id,
            "implementation_sha256": self.implementation_sha256,
            "evidence_id": self.evidence_id,
            "evidence_sha256": self.evidence_sha256,
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.canonical_payload())

    @property
    def content_sha256(self) -> str:
        return _content_sha256(self.canonical_payload())

    @property
    def stable_id(self) -> str:
        return f"route-scorer:v{self.contract_version}:" f"sha256:{self.content_sha256}"

    def to_dict(self) -> Dict[str, object]:
        payload = self.canonical_payload()
        payload["content_sha256"] = self.content_sha256
        payload["stable_id"] = self.stable_id
        return payload

    def expected_runtime_attestation(self) -> "RouteScorerRuntimeAttestation":
        """Return the complete immutable artifact identity required at runtime."""

        return RouteScorerRuntimeAttestation(
            contract_content_sha256=self.content_sha256,
            scorer_stable_id=self.scorer_stable_id,
            model_sha256=self.model_sha256,
            scaler_sha256=self.scaler_sha256,
            feature_manifest_sha256=self.feature_manifest_sha256,
            confidence_rule_sha256=self.confidence_rule_sha256,
            minimum_route_confidence=self.minimum_route_confidence,
            selection_rule_id=self.selection_rule_id,
            implementation_sha256=self.implementation_sha256,
            evidence_sha256=self.evidence_sha256,
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RouteScorerContract":
        if not isinstance(raw, Mapping):
            raise ResolutionError("route scorer contract must be a mapping")
        required = {
            "record_type",
            "contract_version",
            "policy_id",
            "scorer_stable_id",
            "model_sha256",
            "scaler_sha256",
            "feature_manifest_sha256",
            "feature_schema",
            "confidence_rule_sha256",
            "minimum_route_confidence",
            "selection_rule_id",
            "implementation_sha256",
            "evidence_id",
            "evidence_sha256",
        }
        optional = {"content_sha256", "stable_id"}
        keys = set(raw)
        missing = required - keys
        extra = keys - required - optional
        if missing:
            raise ResolutionError("route scorer contract is missing fields: " + ", ".join(sorted(missing)))
        if extra:
            raise ResolutionError("route scorer contract has unknown fields: " + ", ".join(sorted(extra)))
        if raw["record_type"] != "route_scorer_contract":
            raise ResolutionError("route scorer contract record_type must be route_scorer_contract")
        contract = cls(
            policy_id=raw["policy_id"],
            scorer_stable_id=raw["scorer_stable_id"],
            model_sha256=raw["model_sha256"],
            scaler_sha256=raw["scaler_sha256"],
            feature_manifest_sha256=raw["feature_manifest_sha256"],
            feature_schema=_feature_schema_from_payload(raw["feature_schema"]),
            confidence_rule_sha256=raw["confidence_rule_sha256"],
            minimum_route_confidence=raw["minimum_route_confidence"],
            implementation_sha256=raw["implementation_sha256"],
            evidence_id=raw["evidence_id"],
            evidence_sha256=raw["evidence_sha256"],
            selection_rule_id=raw["selection_rule_id"],
            contract_version=raw["contract_version"],
        )
        if "content_sha256" in raw:
            claimed_digest = _sha256_digest(
                raw["content_sha256"],
                "content_sha256",
            )
            if claimed_digest != contract.content_sha256:
                raise ResolutionError("route scorer contract content_sha256 does not match content")
        if "stable_id" in raw and raw["stable_id"] != contract.stable_id:
            raise ResolutionError("route scorer contract stable_id does not match content")
        return contract


@dataclass(frozen=True)
class RouteScorerRuntimeAttestation:
    """Complete runtime identity of every scorer artifact bound by its contract."""

    contract_content_sha256: str
    scorer_stable_id: str
    model_sha256: str
    scaler_sha256: str
    feature_manifest_sha256: str
    confidence_rule_sha256: str
    minimum_route_confidence: float
    selection_rule_id: str
    implementation_sha256: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scorer_stable_id",
            _opaque_id(self.scorer_stable_id, "scorer_stable_id"),
        )
        selection_rule_id = _opaque_id(
            self.selection_rule_id,
            "selection_rule_id",
        )
        if selection_rule_id != ROUTE_SELECTION_RULE_ID:
            raise ResolutionError(f"unsupported runtime route selection rule: {selection_rule_id}")
        object.__setattr__(self, "selection_rule_id", selection_rule_id)
        object.__setattr__(
            self,
            "minimum_route_confidence",
            _score(
                self.minimum_route_confidence,
                "minimum_route_confidence",
            ),
        )
        for field_name in (
            "contract_content_sha256",
            "model_sha256",
            "scaler_sha256",
            "feature_manifest_sha256",
            "confidence_rule_sha256",
            "implementation_sha256",
            "evidence_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _sha256_digest(getattr(self, field_name), field_name),
            )
        expected_scorer_id = (
            f"route-scorer-implementation:v{ROUTE_SCORER_CONTRACT_VERSION}:" f"sha256:{self.implementation_sha256}"
        )
        if self.scorer_stable_id != expected_scorer_id:
            raise ResolutionError("runtime scorer stable ID does not bind implementation_sha256")

    def canonical_payload(self) -> Dict[str, object]:
        return {
            "record_type": "route_scorer_runtime_attestation",
            "contract_content_sha256": self.contract_content_sha256,
            "scorer_stable_id": self.scorer_stable_id,
            "model_sha256": self.model_sha256,
            "scaler_sha256": self.scaler_sha256,
            "feature_manifest_sha256": self.feature_manifest_sha256,
            "confidence_rule_sha256": self.confidence_rule_sha256,
            "minimum_route_confidence": self.minimum_route_confidence,
            "selection_rule_id": self.selection_rule_id,
            "implementation_sha256": self.implementation_sha256,
            "evidence_sha256": self.evidence_sha256,
        }

    @property
    def content_sha256(self) -> str:
        return _content_sha256(self.canonical_payload())

    def to_dict(self) -> Dict[str, object]:
        payload = self.canonical_payload()
        payload["content_sha256"] = self.content_sha256
        return payload


@dataclass(frozen=True)
class MaterializedIndexRef:
    """Content-addressed contract for an already-built target-space index.

    A forward bridge is not searchable merely because its transform exists.  The
    resolver requires this exact contract in its materialization trust store, tied
    to the bridge, source snapshot, target index, build, corpus, and manifest.
    """

    bridge_id: str
    source_index_id: str
    index: ResidentIndexDescriptor
    manifest_id: str
    manifest_sha256: str
    contract_version: int = MATERIALIZATION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.index, ResidentIndexDescriptor):
            raise ResolutionError("materialized index must be a ResidentIndexDescriptor")
        if (
            isinstance(self.contract_version, bool)
            or not isinstance(self.contract_version, int)
            or self.contract_version != MATERIALIZATION_CONTRACT_VERSION
        ):
            raise ResolutionError(
                "unsupported materialization contract version "
                f"{self.contract_version!r}; runtime supports "
                f"{MATERIALIZATION_CONTRACT_VERSION}"
            )
        object.__setattr__(self, "bridge_id", _opaque_id(self.bridge_id, "bridge_id"))
        object.__setattr__(
            self,
            "source_index_id",
            _opaque_id(self.source_index_id, "source_index_id"),
        )
        object.__setattr__(self, "manifest_id", _opaque_id(self.manifest_id, "manifest_id"))
        object.__setattr__(
            self,
            "manifest_sha256",
            _sha256_digest(self.manifest_sha256, "manifest_sha256"),
        )

    @property
    def index_id(self) -> str:
        return self.index.index_id

    @property
    def representation(self) -> RepresentationDescriptor:
        return self.index.representation

    def canonical_payload(self) -> Dict[str, object]:
        return {
            "record_type": "materialized_index_contract",
            "contract_version": self.contract_version,
            "bridge_stable_id": self.bridge_id,
            "source_resident_index_stable_id": self.source_index_id,
            "target_materialized_index_stable_id": self.index.stable_id,
            "target_index_id": self.index.index_id,
            "target_document_space_id": self.index.document_space.stable_id,
            "target_query_space_id": self.index.query_space.stable_id,
            "corpus_snapshot_id": self.index.corpus_snapshot_id,
            "corpus_snapshot_sha256": self.index.corpus_snapshot_sha256,
            "vector_build_id": self.index.vector_build_id,
            "vector_build_sha256": self.index.vector_build_sha256,
            "document_count": self.index.document_count,
            "manifest_id": self.manifest_id,
            "manifest_sha256": self.manifest_sha256,
        }

    @property
    def content_sha256(self) -> str:
        return _content_sha256(self.canonical_payload())

    @property
    def stable_id(self) -> str:
        return f"materialization:v{self.contract_version}:" f"sha256:{self.content_sha256}"

    def to_dict(self) -> Dict[str, object]:
        payload = self.canonical_payload()
        payload["content_sha256"] = self.content_sha256
        payload["stable_id"] = self.stable_id
        return payload


class PerQueryRouteScorer(Protocol):
    """Pluggable interface for a per-query router; no ML implementation is implied."""

    runtime_attestation: RouteScorerRuntimeAttestation

    def score_routes(
        self,
        *,
        query_space: RepresentationDescriptor,
        resident_index_space: RepresentationDescriptor,
        candidates: Sequence[BridgeContract],
        query_context: Mapping[str, Any],
    ) -> Mapping[str, float]:
        """Return confidence by bridge ID for the supplied query context."""


@dataclass(frozen=True)
class ResolutionDecision:
    """Validated, explicit search plan produced by ``SemanticCacheResolver``."""

    mode: ResolutionMode
    reason: ResolutionReason
    query_space: RepresentationDescriptor
    resident_index: ResidentIndexDescriptor
    search_index: Optional[ResidentIndexDescriptor]
    confidence: float
    detail: str
    bridge: Optional[BridgeContract] = None
    materialized_index: Optional[MaterializedIndexRef] = None
    route_policy_id: Optional[str] = None
    route_policy_sha256: Optional[str] = None
    route_scorer_id: Optional[str] = None
    route_scorer_implementation_sha256: Optional[str] = None
    route_minimum_confidence: Optional[float] = None
    route_selection_rule_id: Optional[str] = None
    route_confidence_rule_sha256: Optional[str] = None
    route_query_input_sha256: Optional[str] = None
    route_invocation_sha256: Optional[str] = None
    route_candidate_ids: Tuple[str, ...] = ()
    route_scores: Tuple[Tuple[str, float], ...] = ()
    route_failure_class: Optional[str] = None

    def __post_init__(self) -> None:
        try:
            mode = ResolutionMode(self.mode)
        except (TypeError, ValueError) as exc:
            raise ResolutionError(f"unknown resolution mode: {self.mode!r}") from exc
        try:
            reason = ResolutionReason(self.reason)
        except (TypeError, ValueError) as exc:
            raise ResolutionError(f"unknown resolution reason: {self.reason!r}") from exc
        if not isinstance(self.query_space, RepresentationDescriptor):
            raise ResolutionError("query_space must be a RepresentationDescriptor")
        if not isinstance(self.resident_index, ResidentIndexDescriptor):
            raise ResolutionError("resident_index must be a ResidentIndexDescriptor")
        if self.search_index is not None and not isinstance(self.search_index, ResidentIndexDescriptor):
            raise ResolutionError("search_index must be a ResidentIndexDescriptor or None")
        confidence = _score(self.confidence, "confidence")
        detail = _opaque_id(self.detail, "detail", maximum_length=2048)
        route_policy_fields = (
            self.route_policy_id,
            self.route_policy_sha256,
            self.route_scorer_id,
            self.route_scorer_implementation_sha256,
            self.route_minimum_confidence,
            self.route_selection_rule_id,
            self.route_confidence_rule_sha256,
            self.route_query_input_sha256,
            self.route_invocation_sha256,
        )
        if all(value is None for value in route_policy_fields):
            if self.route_candidate_ids or self.route_scores or self.route_failure_class is not None:
                raise ResolutionError("route replay data requires complete route policy authority")
            normalized_candidate_ids: Tuple[str, ...] = ()
            normalized_route_scores: Tuple[Tuple[str, float], ...] = ()
            route_failure_class = None
        elif any(value is None for value in route_policy_fields):
            raise ResolutionError("route policy, scorer, query, and invocation digests are all required")
        else:
            route_policy_id = _opaque_id(
                self.route_policy_id,
                "route_policy_id",
            )
            route_policy_sha256 = _sha256_digest(
                self.route_policy_sha256,
                "route_policy_sha256",
            )
            route_scorer_id = _opaque_id(
                self.route_scorer_id,
                "route_scorer_id",
            )
            route_scorer_implementation_sha256 = _sha256_digest(
                self.route_scorer_implementation_sha256,
                "route_scorer_implementation_sha256",
            )
            route_minimum_confidence = _score(
                self.route_minimum_confidence,
                "route_minimum_confidence",
            )
            route_selection_rule_id = _opaque_id(
                self.route_selection_rule_id,
                "route_selection_rule_id",
            )
            if route_selection_rule_id != ROUTE_SELECTION_RULE_ID:
                raise ResolutionError(f"unsupported route selection rule: {route_selection_rule_id}")
            route_confidence_rule_digest = _sha256_digest(
                self.route_confidence_rule_sha256,
                "route_confidence_rule_sha256",
            )
            if route_confidence_rule_digest != route_confidence_rule_sha256(route_minimum_confidence):
                raise ResolutionError("route confidence rule digest does not match replay threshold")
            route_query_input_sha256 = _sha256_digest(
                self.route_query_input_sha256,
                "route_query_input_sha256",
            )
            route_invocation_sha256 = _sha256_digest(
                self.route_invocation_sha256,
                "route_invocation_sha256",
            )
            candidate_ids = set()
            for raw_candidate_id in self.route_candidate_ids:
                candidate_id = _opaque_id(
                    raw_candidate_id,
                    "route candidate ID",
                )
                if candidate_id in candidate_ids:
                    raise ResolutionError(f"duplicate route candidate ID: {candidate_id}")
                candidate_ids.add(candidate_id)
            if not candidate_ids:
                raise ResolutionError("route replay data requires eligible candidate IDs")
            normalized_candidate_ids = tuple(sorted(candidate_ids))
            normalized = {}
            for item in self.route_scores:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise ResolutionError("route_scores entries must be (bridge_id, confidence) pairs")
                bridge_id = _opaque_id(item[0], "route score bridge_id")
                if bridge_id in normalized:
                    raise ResolutionError(f"duplicate route score for bridge: {bridge_id}")
                normalized[bridge_id] = _score(
                    item[1],
                    f"route score for {bridge_id}",
                )
            normalized_route_scores = tuple(sorted(normalized.items()))
            if set(normalized) - candidate_ids:
                raise ResolutionError("route score vector contains an ineligible candidate")
            expected_policy_id = f"route-scorer:v{ROUTE_SCORER_CONTRACT_VERSION}:" f"sha256:{route_policy_sha256}"
            if route_policy_id != expected_policy_id:
                raise ResolutionError("route_policy_id does not match route_policy_sha256")
            expected_scorer_id = (
                f"route-scorer-implementation:v{ROUTE_SCORER_CONTRACT_VERSION}:"
                f"sha256:{route_scorer_implementation_sha256}"
            )
            if route_scorer_id != expected_scorer_id:
                raise ResolutionError("route_scorer_id does not match its implementation digest")
            route_failure_class = (
                _opaque_id(
                    self.route_failure_class,
                    "route_failure_class",
                    maximum_length=256,
                )
                if self.route_failure_class is not None
                else None
            )
            object.__setattr__(self, "route_policy_id", route_policy_id)
            object.__setattr__(self, "route_policy_sha256", route_policy_sha256)
            object.__setattr__(self, "route_scorer_id", route_scorer_id)
            object.__setattr__(
                self,
                "route_scorer_implementation_sha256",
                route_scorer_implementation_sha256,
            )
            object.__setattr__(
                self,
                "route_minimum_confidence",
                route_minimum_confidence,
            )
            object.__setattr__(
                self,
                "route_selection_rule_id",
                route_selection_rule_id,
            )
            object.__setattr__(
                self,
                "route_confidence_rule_sha256",
                route_confidence_rule_digest,
            )
            object.__setattr__(
                self,
                "route_query_input_sha256",
                route_query_input_sha256,
            )
            object.__setattr__(
                self,
                "route_invocation_sha256",
                route_invocation_sha256,
            )

        query_id = self.query_space.stable_id
        resident_id = self.resident_index.query_space.stable_id
        search_id = self.search_index.query_space.stable_id if self.search_index is not None else None

        if mode is ResolutionMode.ABSTAIN:
            if self.search_index is not None or self.bridge is not None or self.materialized_index is not None:
                raise ResolutionError("abstain decisions must not contain an executable search route")
        elif mode is ResolutionMode.NATIVE:
            if query_id != resident_id or search_id != query_id:
                raise ResolutionError("native decisions require identical query, resident, and search spaces")
            if self.bridge is not None or self.materialized_index is not None:
                raise ResolutionError("native decisions must not include a bridge or materialized index")
            if self.search_index.stable_id != self.resident_index.stable_id:
                raise ResolutionError("native decisions must search the resident index")
            if reason is not ResolutionReason.IDENTICAL_SPACE:
                raise ResolutionError("native decisions require reason=identical_space")
        elif mode is ResolutionMode.REVERSE:
            if not isinstance(self.bridge, BridgeContract):
                raise ResolutionError("reverse decisions require a BridgeContract")
            if self.bridge.cost is not BridgeCost.ZERO_WRITE:
                raise ResolutionError("reverse decisions require a zero_write bridge")
            if self.bridge.role not in {RepresentationRole.QUERY, RepresentationRole.SHARED}:
                raise ResolutionError("reverse decisions require a query-role bridge")
            if self.bridge.validation_state is not BridgeValidationState.VALIDATED:
                raise ResolutionError("reverse decisions require a validated bridge")
            if self.bridge.source.stable_id != query_id or self.bridge.target.stable_id != resident_id:
                raise ResolutionError("reverse bridge direction must be query_space -> resident_index_space")
            if search_id != resident_id:
                raise ResolutionError("reverse decisions must search the resident index space")
            if self.search_index.stable_id != self.resident_index.stable_id:
                raise ResolutionError("reverse decisions must search the resident index")
            if self.materialized_index is not None:
                raise ResolutionError("reverse decisions cannot include a materialized index")
            if reason is not ResolutionReason.VALIDATED_REVERSE_BRIDGE:
                raise ResolutionError("reverse decisions require reason=validated_reverse_bridge")
        else:
            if not isinstance(self.bridge, BridgeContract):
                raise ResolutionError("forward decisions require a BridgeContract")
            if self.bridge.cost is not BridgeCost.MATERIALIZED_INDEX:
                raise ResolutionError("forward decisions require a materialized_index bridge")
            if self.bridge.role not in {RepresentationRole.DOCUMENT, RepresentationRole.SHARED}:
                raise ResolutionError("forward decisions require a document-role bridge")
            if self.bridge.validation_state is not BridgeValidationState.VALIDATED:
                raise ResolutionError("forward decisions require a validated bridge")
            if self.bridge.source.stable_id != self.resident_index.document_space.stable_id:
                raise ResolutionError("forward bridge source must match the resident document space")
            if search_id != query_id:
                raise ResolutionError("forward decisions must search in the query representation space")
            if not isinstance(self.materialized_index, MaterializedIndexRef):
                raise ResolutionError("forward decisions require a MaterializedIndexRef")
            if self.materialized_index.bridge_id != self.bridge.stable_id:
                raise ResolutionError("materialized index does not belong to the selected bridge")
            if self.materialized_index.index.document_space.stable_id != self.bridge.target.stable_id:
                raise ResolutionError("materialized document space does not match the bridge target")
            if self.materialized_index.index.query_space.stable_id != query_id:
                raise ResolutionError("materialized index query space does not match query_space")
            if self.search_index.stable_id != self.materialized_index.index.stable_id:
                raise ResolutionError("forward search_index must be the referenced materialized index")
            if self.materialized_index.source_index_id != self.resident_index.stable_id:
                raise ResolutionError("materialized index was not built from the resident index snapshot")
            if self.materialized_index.index.corpus_snapshot_sha256 != self.resident_index.corpus_snapshot_sha256:
                raise ResolutionError("materialized index corpus snapshot differs from the resident index")
            if self.materialized_index.index.document_count != self.resident_index.document_count:
                raise ResolutionError("materialized index document count differs from the resident index")
            if reason is not ResolutionReason.VALIDATED_FORWARD_BRIDGE:
                raise ResolutionError("forward decisions require reason=validated_forward_bridge")

        if normalized_candidate_ids:
            scored_candidate_ids = {bridge_id for bridge_id, _score_value in normalized_route_scores}
            if mode in {ResolutionMode.REVERSE, ResolutionMode.FORWARD}:
                if route_failure_class is not None:
                    raise ResolutionError("executable route decisions cannot contain a scorer failure")
                if scored_candidate_ids != set(normalized_candidate_ids):
                    raise ResolutionError("executable route decisions require a full score vector")
                if self.bridge.stable_id not in scored_candidate_ids:
                    raise ResolutionError("route score vector does not contain the selected bridge")
            elif mode is ResolutionMode.ABSTAIN:
                if reason is ResolutionReason.LOW_ROUTE_CONFIDENCE:
                    if route_failure_class is not None:
                        raise ResolutionError("low-confidence abstention cannot contain a scorer failure")
                    if scored_candidate_ids != set(normalized_candidate_ids):
                        raise ResolutionError("low-confidence abstention requires a full score vector")
                elif reason is ResolutionReason.ROUTER_FAILURE:
                    if route_failure_class is None:
                        raise ResolutionError("router-failure abstention requires a failure class")
                    if scored_candidate_ids and scored_candidate_ids != set(normalized_candidate_ids):
                        raise ResolutionError("router-failure score evidence must be empty or complete")
                else:
                    raise ResolutionError("route replay data is not valid for this abstention reason")
            else:
                raise ResolutionError("route replay data is valid only for bridged or scorer-abstain decisions")
        elif reason is ResolutionReason.ROUTER_FAILURE:
            raise ResolutionError("router-failure abstention requires route replay authority")

        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "detail", detail)
        object.__setattr__(self, "route_candidate_ids", normalized_candidate_ids)
        object.__setattr__(self, "route_scores", normalized_route_scores)
        object.__setattr__(self, "route_failure_class", route_failure_class)

    @property
    def can_search(self) -> bool:
        """True only when the decision contains a representation-safe search plan."""

        return self.mode is not ResolutionMode.ABSTAIN

    @property
    def requires_query_transform(self) -> bool:
        return self.mode is ResolutionMode.REVERSE

    @property
    def requires_materialized_index(self) -> bool:
        return self.mode is ResolutionMode.FORWARD

    @property
    def resident_index_space(self) -> RepresentationDescriptor:
        return self.resident_index.query_space

    @property
    def search_space(self) -> Optional[RepresentationDescriptor]:
        return self.search_index.query_space if self.search_index is not None else None

    @property
    def corpus_write_cost(self) -> BridgeCost:
        return BridgeCost.MATERIALIZED_INDEX if self.mode is ResolutionMode.FORWARD else BridgeCost.ZERO_WRITE

    def assert_search_authorized(self) -> None:
        """Raise instead of allowing a caller to silently search after abstention."""

        if not self.can_search:
            raise ResolutionError(f"semantic-cache search is not authorized: {self.reason.value}: {self.detail}")

    def to_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode.value,
            "reason": self.reason.value,
            "query_space_id": self.query_space.stable_id,
            "resident_index_id": self.resident_index.stable_id,
            "resident_index_query_space_id": self.resident_index.query_space.stable_id,
            "resident_index_document_space_id": self.resident_index.document_space.stable_id,
            "search_index_id": self.search_index.stable_id if self.search_index is not None else None,
            "search_space_id": (self.search_index.query_space.stable_id if self.search_index is not None else None),
            "confidence": self.confidence,
            "detail": self.detail,
            "bridge_id": self.bridge.stable_id if self.bridge is not None else None,
            "evidence_id": self.bridge.evidence_id if self.bridge is not None else None,
            "evidence_sha256": (self.bridge.evidence_sha256 if self.bridge is not None else None),
            "materialized_index": (self.materialized_index.to_dict() if self.materialized_index is not None else None),
            "route_policy_id": self.route_policy_id,
            "route_policy_sha256": self.route_policy_sha256,
            "route_scorer_id": self.route_scorer_id,
            "route_scorer_implementation_sha256": self.route_scorer_implementation_sha256,
            "route_minimum_confidence": self.route_minimum_confidence,
            "route_selection_rule_id": self.route_selection_rule_id,
            "route_confidence_rule_sha256": self.route_confidence_rule_sha256,
            "route_query_input_sha256": self.route_query_input_sha256,
            "route_invocation_sha256": self.route_invocation_sha256,
            "route_candidate_ids": list(self.route_candidate_ids),
            "route_scores": [{"bridge_id": bridge_id, "confidence": score} for bridge_id, score in self.route_scores],
            "route_failure_class": self.route_failure_class,
            "can_search": self.can_search,
            "corpus_write_cost": self.corpus_write_cost.value,
        }


@dataclass(frozen=True)
class _Candidate:
    mode: ResolutionMode
    bridge: BridgeContract
    materialized_index: Optional[MaterializedIndexRef]


class SemanticCacheResolver:
    """Resolve native or bridged search plans with deterministic fail-closed rules."""

    def __init__(
        self,
        bridges: Iterable[BridgeContract] = (),
        *,
        minimum_route_confidence: float = 0.80,
        route_scorer: Optional[PerQueryRouteScorer] = None,
        route_scorer_contract: Optional[RouteScorerContract] = None,
        trusted_evidence_sha256s: Iterable[str] = (),
        trusted_bridge_contract_ids: Iterable[str] = (),
        trusted_materialization_contracts: Optional[Mapping[str, str]] = None,
        trusted_route_scorer_contract_sha256s: Iterable[str] = (),
    ) -> None:
        self._minimum_route_confidence = _score(
            minimum_route_confidence,
            "minimum_route_confidence",
        )
        self._trusted_evidence_sha256s = frozenset(
            _sha256_digest(value, "trusted_evidence_sha256") for value in trusted_evidence_sha256s
        )
        self._trusted_bridge_contract_ids = frozenset(
            self._trusted_bridge_contract_id(value) for value in trusted_bridge_contract_ids
        )
        self._trusted_materialization_contracts = self._normalize_trusted_materialization_contracts(
            trusted_materialization_contracts
        )
        self._trusted_route_scorer_contract_sha256s = frozenset(
            _sha256_digest(value, "trusted_route_scorer_contract_sha256")
            for value in trusted_route_scorer_contract_sha256s
        )
        if route_scorer is None:
            if route_scorer_contract is not None:
                raise ResolutionError("route_scorer_contract requires a configured route_scorer")
        else:
            if not callable(getattr(route_scorer, "score_routes", None)):
                raise ResolutionError("route_scorer must provide callable score_routes")
            if not isinstance(route_scorer_contract, RouteScorerContract):
                raise ResolutionError("configured route_scorer requires a RouteScorerContract")
            if route_scorer_contract.content_sha256 not in self._trusted_route_scorer_contract_sha256s:
                raise ResolutionError(
                    "route scorer contract content digest is not present in the " "runtime trust store"
                )
            if self._minimum_route_confidence != route_scorer_contract.minimum_route_confidence:
                raise ResolutionError("minimum_route_confidence does not match the trusted " "route scorer contract")
            self._assert_route_scorer_attestation(
                route_scorer,
                route_scorer_contract,
            )
        self._route_scorer = route_scorer
        self._route_scorer_contract = route_scorer_contract
        self._route_scorer_contract_snapshot = (
            RouteScorerContract.from_dict(route_scorer_contract.to_dict())
            if route_scorer_contract is not None
            else None
        )
        by_id: Dict[str, BridgeContract] = {}
        for bridge in bridges:
            if not isinstance(bridge, BridgeContract):
                raise ResolutionError("bridges must contain only BridgeContract instances")
            if bridge.stable_id in by_id:
                raise ResolutionError(f"duplicate bridge contract: {bridge.stable_id}")
            by_id[bridge.stable_id] = bridge
        self._bridges: Tuple[BridgeContract, ...] = tuple(by_id[bridge_id] for bridge_id in sorted(by_id))

    @property
    def minimum_route_confidence(self) -> float:
        return self._minimum_route_confidence

    @property
    def registered_bridges(self) -> Tuple[BridgeContract, ...]:
        return self._bridges

    @property
    def trusted_evidence_sha256s(self) -> frozenset:
        """Content digests explicitly trusted by the resolver's configuration."""

        return self._trusted_evidence_sha256s

    @property
    def trusted_bridge_contract_ids(self) -> frozenset:
        """Exact content-addressed bridge contracts trusted for execution."""

        return self._trusted_bridge_contract_ids

    @property
    def trusted_materialization_contracts(self) -> Dict[str, str]:
        """Exact materialization contract ID-to-content-digest trust pairs."""

        return dict(self._trusted_materialization_contracts)

    @property
    def route_scorer_contract(self) -> Optional[RouteScorerContract]:
        return self._route_scorer_contract

    @property
    def trusted_route_scorer_contract_sha256s(self) -> frozenset:
        return self._trusted_route_scorer_contract_sha256s

    @staticmethod
    def _assert_route_scorer_attestation(
        route_scorer: PerQueryRouteScorer,
        contract: RouteScorerContract,
    ) -> None:
        attestation = getattr(route_scorer, "runtime_attestation", None)
        if not isinstance(attestation, RouteScorerRuntimeAttestation):
            raise ResolutionError("route scorer requires a RouteScorerRuntimeAttestation")
        expected = contract.expected_runtime_attestation()
        for field_name in (
            "contract_content_sha256",
            "scorer_stable_id",
            "model_sha256",
            "scaler_sha256",
            "feature_manifest_sha256",
            "confidence_rule_sha256",
            "minimum_route_confidence",
            "selection_rule_id",
            "implementation_sha256",
            "evidence_sha256",
        ):
            if getattr(attestation, field_name) != getattr(expected, field_name):
                raise ResolutionError("route scorer runtime attestation mismatch for " f"{field_name}")

    @staticmethod
    def _trusted_bridge_contract_id(value: object) -> str:
        bridge_id = _opaque_id(value, "trusted_bridge_contract_id")
        if re.fullmatch(r"bridge:v[1-9][0-9]*:sha256:[0-9a-f]{64}", bridge_id) is None:
            raise ResolutionError("trusted_bridge_contract_id must be a content-addressed bridge stable ID")
        return bridge_id

    @staticmethod
    def _normalize_trusted_materialization_contracts(
        raw_contracts: Optional[Mapping[str, str]],
    ) -> Dict[str, str]:
        if raw_contracts is None:
            return {}
        if not isinstance(raw_contracts, Mapping):
            raise ResolutionError(
                "trusted_materialization_contracts must be a mapping " "from stable ID to content digest"
            )
        normalized: Dict[str, str] = {}
        for raw_stable_id, raw_content_sha256 in raw_contracts.items():
            stable_id = _opaque_id(
                raw_stable_id,
                "trusted materialization stable ID",
            )
            if (
                re.fullmatch(
                    r"materialization:v[1-9][0-9]*:sha256:[0-9a-f]{64}",
                    stable_id,
                )
                is None
            ):
                raise ResolutionError("trusted materialization stable ID must be content-addressed")
            content_sha256 = _sha256_digest(
                raw_content_sha256,
                "trusted materialization content digest",
            )
            expected_stable_id = f"materialization:v{MATERIALIZATION_CONTRACT_VERSION}:" f"sha256:{content_sha256}"
            if stable_id != expected_stable_id:
                raise ResolutionError("trusted materialization stable ID does not bind its " "content digest")
            if stable_id in normalized:
                raise ResolutionError(f"duplicate trusted materialization contract: {stable_id}")
            normalized[stable_id] = content_sha256
        return normalized

    def resolve(
        self,
        query_space: RepresentationDescriptor,
        resident_index: ResidentIndexDescriptor,
        *,
        materialized_indexes: Iterable[MaterializedIndexRef] = (),
        route_scores: Optional[Mapping[str, float]] = None,
        query_context: Optional[Mapping[str, Any]] = None,
    ) -> ResolutionDecision:
        """Return one explicit search plan, or ``ABSTAIN``.

        Route selection is deterministic: effective confidence descending, then
        zero-write reverse before materialized forward, then canonical bridge ID.
        Validation confidence is a ceiling on any per-query score.
        """

        if not isinstance(query_space, RepresentationDescriptor):
            raise ResolutionError("query_space must be a RepresentationDescriptor")
        if not isinstance(resident_index, ResidentIndexDescriptor):
            raise ResolutionError("resident_index must be a ResidentIndexDescriptor")
        if route_scores is not None:
            raise ResolutionError("direct route_scores are not accepted; configure a trusted route_scorer")
        if self._route_scorer is None and query_context is not None and not isinstance(query_context, Mapping):
            raise ResolutionError("query_context must be a mapping or None")

        materialized = self._materialization_registry(materialized_indexes)
        registered_ids = {bridge.stable_id for bridge in self._bridges}
        unknown_materializations = set(materialized) - registered_ids
        if unknown_materializations:
            raise ResolutionError(
                "materialized indexes reference unregistered bridges: " + ", ".join(sorted(unknown_materializations))
            )

        if query_space.stable_id == resident_index.query_space.stable_id:
            return ResolutionDecision(
                mode=ResolutionMode.NATIVE,
                reason=ResolutionReason.IDENTICAL_SPACE,
                query_space=query_space,
                resident_index=resident_index,
                search_index=resident_index,
                confidence=1.0,
                detail="query and resident index use the same representation ABI",
            )

        candidates = []
        relevant = []
        wrong_cost = []
        unavailable_materialization = []
        untrusted_materialization = []
        for bridge in self._bridges:
            reverse_match = (
                bridge.source.stable_id == query_space.stable_id
                and bridge.target.stable_id == resident_index.query_space.stable_id
                and bridge.role in {RepresentationRole.QUERY, RepresentationRole.SHARED}
            )
            forward_match = (
                bridge.source.stable_id == resident_index.document_space.stable_id
                and bridge.target.compatibility_domain_sha256 == query_space.compatibility_domain_sha256
                and bridge.role in {RepresentationRole.DOCUMENT, RepresentationRole.SHARED}
            )
            if not reverse_match and not forward_match:
                continue
            relevant.append(bridge)

            expected_cost = BridgeCost.ZERO_WRITE if reverse_match else BridgeCost.MATERIALIZED_INDEX
            if bridge.cost is not expected_cost:
                wrong_cost.append(bridge)
                continue
            if bridge.validation_state is not BridgeValidationState.VALIDATED:
                continue
            if bridge.evidence_sha256 not in self._trusted_evidence_sha256s:
                continue
            if bridge.stable_id not in self._trusted_bridge_contract_ids:
                continue
            if reverse_match:
                candidates.append(_Candidate(ResolutionMode.REVERSE, bridge, None))
                continue

            materialized_index = materialized.get(bridge.stable_id)
            if materialized_index is None:
                unavailable_materialization.append(bridge)
                continue
            trusted_materialization_sha256 = self._trusted_materialization_contracts.get(materialized_index.stable_id)
            if trusted_materialization_sha256 != materialized_index.content_sha256:
                untrusted_materialization.append(materialized_index)
                continue
            if materialized_index.source_index_id != resident_index.stable_id:
                raise ResolutionError(
                    f"materialized index {materialized_index.index_id!r} was built from "
                    "a different resident index snapshot"
                )
            if materialized_index.index.document_space.stable_id != bridge.target.stable_id:
                raise ResolutionError(
                    f"materialized index {materialized_index.index_id!r} has document space "
                    f"{materialized_index.index.document_space.stable_id}, expected {bridge.target.stable_id}"
                )
            if materialized_index.index.query_space.stable_id != query_space.stable_id:
                raise ResolutionError(
                    f"materialized index {materialized_index.index_id!r} has query space "
                    f"{materialized_index.index.query_space.stable_id}, expected {query_space.stable_id}"
                )
            if materialized_index.index.corpus_snapshot_sha256 != resident_index.corpus_snapshot_sha256:
                raise ResolutionError(
                    f"materialized index {materialized_index.index_id!r} uses a different corpus snapshot"
                )
            if materialized_index.index.document_count != resident_index.document_count:
                raise ResolutionError(
                    f"materialized index {materialized_index.index_id!r} has a different document count"
                )
            candidates.append(_Candidate(ResolutionMode.FORWARD, bridge, materialized_index))

        if not relevant:
            return self._abstain(
                query_space,
                resident_index,
                ResolutionReason.NO_REPRESENTATION_BRIDGE,
                "no registered bridge connects the query and resident index spaces",
            )
        cost_correct = [bridge for bridge in relevant if bridge not in wrong_cost]
        if not cost_correct:
            return self._abstain(
                query_space,
                resident_index,
                ResolutionReason.INVALID_BRIDGE_COST,
                "matching bridges have deployment costs incompatible with their direction",
            )
        validated = [bridge for bridge in cost_correct if bridge.validation_state is BridgeValidationState.VALIDATED]
        if not validated:
            return self._abstain(
                query_space,
                resident_index,
                ResolutionReason.NO_VALIDATED_BRIDGE,
                "matching bridges exist but none has validated evidence",
            )
        evidence_trusted_validated = [
            bridge for bridge in validated if bridge.evidence_sha256 in self._trusted_evidence_sha256s
        ]
        if not evidence_trusted_validated:
            return self._abstain(
                query_space,
                resident_index,
                ResolutionReason.UNTRUSTED_BRIDGE_EVIDENCE,
                "matching validated bridges are not present in the runtime evidence trust store",
            )
        contract_trusted_validated = [
            bridge for bridge in evidence_trusted_validated if bridge.stable_id in self._trusted_bridge_contract_ids
        ]
        if not contract_trusted_validated:
            return self._abstain(
                query_space,
                resident_index,
                ResolutionReason.UNTRUSTED_BRIDGE_CONTRACT,
                "matching validated bridges are not present in the exact-contract trust store",
            )
        if not candidates and untrusted_materialization:
            return self._abstain(
                query_space,
                resident_index,
                ResolutionReason.UNTRUSTED_MATERIALIZATION_CONTRACT,
                "matching materialized indexes are not present in the exact " "materialization contract trust store",
            )
        if not candidates and unavailable_materialization:
            return self._abstain(
                query_space,
                resident_index,
                ResolutionReason.MATERIALIZED_INDEX_UNAVAILABLE,
                "a validated forward bridge exists but its target-space index is not materialized",
            )
        if not candidates:
            return self._abstain(
                query_space,
                resident_index,
                ResolutionReason.NO_VALIDATED_BRIDGE,
                "no validated bridge is deployable for this query/index pair",
            )

        if len(candidates) > 1 and self._route_scorer is None:
            return self._abstain(
                query_space,
                resident_index,
                ResolutionReason.ROUTER_REQUIRED,
                "multiple validated routes require a configured per-query route scorer",
            )

        normalized_scores: Optional[Dict[str, float]] = None
        route_fields: Dict[str, object] = {}
        if self._route_scorer is not None:
            contract = self._route_scorer_contract_snapshot
            if contract is None:
                raise ResolutionError("configured route scorer lost its required contract")
            candidate_ids = tuple(sorted(candidate.bridge.stable_id for candidate in candidates))
            raw_query_context = query_context if query_context is not None else {"query_input": "", "features": {}}
            try:
                invocation_context = _canonical_json_value(
                    raw_query_context,
                    "query_context",
                )
            except Exception as exc:
                invocation_context = {
                    "invalid_context_failure_class": type(exc).__name__,
                }
            route_query_input_sha256 = _content_sha256(invocation_context)
            invocation_payload = {
                "record_type": "route_scorer_invocation",
                "route_policy_id": contract.stable_id,
                "route_policy_sha256": contract.content_sha256,
                "route_scorer_id": contract.scorer_stable_id,
                "route_scorer_implementation_sha256": contract.implementation_sha256,
                "minimum_route_confidence": contract.minimum_route_confidence,
                "selection_rule_id": contract.selection_rule_id,
                "confidence_rule_sha256": contract.confidence_rule_sha256,
                "query_space_id": query_space.stable_id,
                "resident_index_id": resident_index.stable_id,
                "resident_index_query_space_id": resident_index.query_space.stable_id,
                "eligible_candidates": [
                    {
                        "bridge_id": candidate.bridge.stable_id,
                        "materialized_index_id": (
                            candidate.materialized_index.index.stable_id
                            if candidate.materialized_index is not None
                            else None
                        ),
                        "materialization_contract_id": (
                            candidate.materialized_index.stable_id if candidate.materialized_index is not None else None
                        ),
                        "materialization_contract_sha256": (
                            candidate.materialized_index.content_sha256
                            if candidate.materialized_index is not None
                            else None
                        ),
                        "mode": candidate.mode.value,
                    }
                    for candidate in sorted(
                        candidates,
                        key=lambda candidate: candidate.bridge.stable_id,
                    )
                ],
                "query_context": invocation_context,
            }
            route_fields = {
                "route_policy_id": contract.stable_id,
                "route_policy_sha256": contract.content_sha256,
                "route_scorer_id": contract.scorer_stable_id,
                "route_scorer_implementation_sha256": contract.implementation_sha256,
                "route_minimum_confidence": contract.minimum_route_confidence,
                "route_selection_rule_id": contract.selection_rule_id,
                "route_confidence_rule_sha256": contract.confidence_rule_sha256,
                "route_query_input_sha256": route_query_input_sha256,
                "route_invocation_sha256": _content_sha256(invocation_payload),
                "route_candidate_ids": candidate_ids,
            }
            try:
                if (
                    self._route_scorer_contract is None
                    or self._route_scorer_contract.content_sha256 != contract.content_sha256
                    or contract.content_sha256 not in self._trusted_route_scorer_contract_sha256s
                ):
                    raise ResolutionError("route scorer contract is no longer present in the " "runtime trust store")
                self._assert_route_scorer_attestation(
                    self._route_scorer,
                    contract,
                )
                canonical_query_context = _validate_route_query_context(
                    raw_query_context,
                    contract.feature_schema,
                )
                scorer_output = self._route_scorer.score_routes(
                    query_space=query_space,
                    resident_index_space=resident_index.query_space,
                    candidates=tuple(candidate.bridge for candidate in candidates),
                    query_context=canonical_query_context,
                )
                normalized_scores = self._normalize_route_scores(scorer_output, candidates)
            except Exception as exc:  # The plugin boundary must fail closed for arbitrary scorer failures.
                return self._abstain(
                    query_space,
                    resident_index,
                    ResolutionReason.ROUTER_FAILURE,
                    f"per-query route scorer failed closed ({type(exc).__name__})",
                    route_failure_class=type(exc).__name__,
                    **route_fields,
                )

        ranked = []
        for candidate in candidates:
            validation_confidence = candidate.bridge.validation_confidence
            query_confidence = (
                normalized_scores.get(candidate.bridge.stable_id, 0.0)
                if normalized_scores is not None
                else validation_confidence
            )
            effective_confidence = min(validation_confidence, query_confidence)
            ranked.append((effective_confidence, candidate))

        qualified = [item for item in ranked if item[0] >= self._minimum_route_confidence]
        if not qualified:
            observed = max((item[0] for item in ranked), default=0.0)
            return self._abstain(
                query_space,
                resident_index,
                ResolutionReason.LOW_ROUTE_CONFIDENCE,
                (
                    f"best effective route confidence {observed:.6f} is below "
                    f"minimum {self._minimum_route_confidence:.6f}"
                ),
                confidence=observed,
                route_scores=(tuple(sorted(normalized_scores.items())) if normalized_scores is not None else ()),
                **route_fields,
            )

        direction_order = {ResolutionMode.REVERSE: 0, ResolutionMode.FORWARD: 1}
        qualified.sort(
            key=lambda item: (
                -item[0],
                direction_order[item[1].mode],
                item[1].bridge.stable_id,
            )
        )
        confidence, selected = qualified[0]
        if normalized_scores is not None:
            route_fields["route_scores"] = tuple(sorted(normalized_scores.items()))
        if selected.mode is ResolutionMode.REVERSE:
            return ResolutionDecision(
                mode=ResolutionMode.REVERSE,
                reason=ResolutionReason.VALIDATED_REVERSE_BRIDGE,
                query_space=query_space,
                resident_index=resident_index,
                search_index=resident_index,
                confidence=confidence,
                detail="transform query into the resident index representation; corpus writes are zero",
                bridge=selected.bridge,
                **route_fields,
            )
        return ResolutionDecision(
            mode=ResolutionMode.FORWARD,
            reason=ResolutionReason.VALIDATED_FORWARD_BRIDGE,
            query_space=query_space,
            resident_index=resident_index,
            search_index=selected.materialized_index.index,
            confidence=confidence,
            detail="search the validated materialized target-space index",
            bridge=selected.bridge,
            materialized_index=selected.materialized_index,
            **route_fields,
        )

    @staticmethod
    def _materialization_registry(
        materialized_indexes: Iterable[MaterializedIndexRef],
    ) -> Dict[str, MaterializedIndexRef]:
        registry: Dict[str, MaterializedIndexRef] = {}
        for reference in materialized_indexes:
            if not isinstance(reference, MaterializedIndexRef):
                raise ResolutionError("materialized_indexes must contain only MaterializedIndexRef instances")
            if reference.bridge_id in registry:
                raise ResolutionError(f"duplicate materialized index for bridge {reference.bridge_id}")
            registry[reference.bridge_id] = reference
        return registry

    @staticmethod
    def _normalize_route_scores(
        raw_scores: Mapping[str, float],
        candidates: Sequence[_Candidate],
    ) -> Dict[str, float]:
        if not isinstance(raw_scores, Mapping):
            raise ResolutionError("route scorer output must be a mapping from bridge ID to confidence")
        candidate_ids = {candidate.bridge.stable_id for candidate in candidates}
        normalized: Dict[str, float] = {}
        for raw_bridge_id, raw_confidence in raw_scores.items():
            bridge_id = _opaque_id(raw_bridge_id, "route score bridge_id")
            if bridge_id not in candidate_ids:
                raise ResolutionError(f"route score references an ineligible bridge: {bridge_id}")
            if bridge_id in normalized:
                raise ResolutionError(f"duplicate route score for bridge: {bridge_id}")
            normalized[bridge_id] = _score(
                raw_confidence,
                f"route score for {bridge_id}",
            )
        missing = candidate_ids - set(normalized)
        if missing:
            raise ResolutionError("route scorer output is missing eligible bridges: " + ", ".join(sorted(missing)))
        if set(normalized) != candidate_ids:
            raise ResolutionError("route scorer output coverage must exactly match eligible bridge IDs")
        return normalized

    @staticmethod
    def _abstain(
        query_space: RepresentationDescriptor,
        resident_index: ResidentIndexDescriptor,
        reason: ResolutionReason,
        detail: str,
        *,
        confidence: float = 0.0,
        route_policy_id: Optional[str] = None,
        route_policy_sha256: Optional[str] = None,
        route_scorer_id: Optional[str] = None,
        route_scorer_implementation_sha256: Optional[str] = None,
        route_minimum_confidence: Optional[float] = None,
        route_selection_rule_id: Optional[str] = None,
        route_confidence_rule_sha256: Optional[str] = None,
        route_query_input_sha256: Optional[str] = None,
        route_invocation_sha256: Optional[str] = None,
        route_candidate_ids: Tuple[str, ...] = (),
        route_scores: Tuple[Tuple[str, float], ...] = (),
        route_failure_class: Optional[str] = None,
    ) -> ResolutionDecision:
        return ResolutionDecision(
            mode=ResolutionMode.ABSTAIN,
            reason=reason,
            query_space=query_space,
            resident_index=resident_index,
            search_index=None,
            confidence=confidence,
            detail=detail,
            route_policy_id=route_policy_id,
            route_policy_sha256=route_policy_sha256,
            route_scorer_id=route_scorer_id,
            route_scorer_implementation_sha256=route_scorer_implementation_sha256,
            route_minimum_confidence=route_minimum_confidence,
            route_selection_rule_id=route_selection_rule_id,
            route_confidence_rule_sha256=route_confidence_rule_sha256,
            route_query_input_sha256=route_query_input_sha256,
            route_invocation_sha256=route_invocation_sha256,
            route_candidate_ids=route_candidate_ids,
            route_scores=route_scores,
            route_failure_class=route_failure_class,
        )
