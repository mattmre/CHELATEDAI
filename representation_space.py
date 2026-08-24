"""Versioned representation-space ABI and validated bridge contracts.

An embedding vector is meaningful only inside the coordinate system that produced
it.  Shape equality is not compatibility: two encoders can both emit 768 values
while assigning unrelated meanings to every coordinate.  This module makes that
usually implicit contract explicit and content-addressed.

The types here are dependency-free, immutable, and safe to import in storage and
query processes.  They describe compatibility; they do not load models or execute
bridges.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Type, TypeVar


REPRESENTATION_ABI_VERSION = 1
BRIDGE_CONTRACT_VERSION = 1
RESIDENT_INDEX_CONTRACT_VERSION = 1


class RepresentationABIError(ValueError):
    """Raised when a representation descriptor violates the ABI."""


class BridgeContractError(RepresentationABIError):
    """Raised when a bridge contract is incomplete or internally inconsistent."""


class Normalization(str, Enum):
    """Vector normalization applied by the encoder pipeline."""

    NONE = "none"
    L2 = "l2"


class DistanceMetric(str, Enum):
    """Similarity/distance operation expected by the representation."""

    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


class VectorDType(str, Enum):
    """Canonical on-wire/storage vector element types."""

    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT8 = "int8"
    UINT8 = "uint8"
    BINARY = "binary"


class RepresentationRole(str, Enum):
    """Encoder-side role whose prompts/pooling produced the vectors."""

    QUERY = "query"
    DOCUMENT = "document"
    SHARED = "shared"


class PoolingStrategy(str, Enum):
    """Token-to-vector pooling operation."""

    MEAN = "mean"
    CLS = "cls"
    LAST_TOKEN = "last_token"
    WEIGHTED_MEAN = "weighted_mean"
    CUSTOM = "custom"


class BridgeValidationState(str, Enum):
    """Lifecycle state of empirical evidence for a bridge."""

    UNVALIDATED = "unvalidated"
    VALIDATED = "validated"
    REJECTED = "rejected"
    EXPIRED = "expired"


class BridgeCost(str, Enum):
    """Operational deployment cost of a directed bridge."""

    ZERO_WRITE = "zero_write"
    MATERIALIZED_INDEX = "materialized_index"


_EnumT = TypeVar("_EnumT", bound=Enum)


def _canonical_text(value: object, field_name: str, *, maximum_length: int = 512) -> str:
    if not isinstance(value, str):
        raise RepresentationABIError(f"{field_name} must be a string")
    if value != value.strip():
        raise RepresentationABIError(f"{field_name} must not contain leading or trailing whitespace")
    normalized = unicodedata.normalize("NFC", value)
    if not normalized:
        raise RepresentationABIError(f"{field_name} must not be empty")
    if len(normalized) > maximum_length:
        raise RepresentationABIError(f"{field_name} must be at most {maximum_length} characters")
    if any(unicodedata.category(character).startswith("C") for character in normalized):
        raise RepresentationABIError(f"{field_name} must not contain control characters")
    return normalized


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RepresentationABIError(f"{field_name} must be an integer")
    if value < 1:
        raise RepresentationABIError(f"{field_name} must be >= 1")
    return value


def _nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RepresentationABIError(f"{field_name} must be an integer")
    if value < 0:
        raise RepresentationABIError(f"{field_name} must be >= 0")
    return value


def _sha256_digest(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise RepresentationABIError(f"{field_name} must be a SHA-256 hex string")
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise RepresentationABIError(f"{field_name} must contain exactly 64 hexadecimal characters")
    return normalized


def _confidence(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BridgeContractError(f"{field_name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0 or converted > 1.0:
        raise BridgeContractError(f"{field_name} must be in the closed interval [0, 1]")
    return converted


def _enum_value(enum_type: Type[_EnumT], value: object, field_name: str) -> _EnumT:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(sorted(str(member.value) for member in enum_type))
        raise RepresentationABIError(f"{field_name} must be one of: {allowed}") from exc


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_id(prefix: str, version: int, payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{prefix}:v{version}:sha256:{digest}"


@dataclass(frozen=True)
class RepresentationDescriptor:
    """Immutable identity of one embedding coordinate system.

    Every field that can change vector meaning participates in ``stable_id``.
    Deployments should persist that ID next to each vector index and compare it
    to the query descriptor before search.
    """

    encoder_id: str
    encoder_revision: str
    encoder_weights_sha256: str
    tokenizer_id: str
    tokenizer_sha256: str
    role: RepresentationRole
    compatibility_domain_id: str
    compatibility_domain_sha256: str
    instruction_sha256: str
    pooling: PoolingStrategy
    dimension: int
    preprocessing_id: str
    preprocessing_sha256: str
    projection_chain_id: str
    projection_chain_sha256: str
    adapter_chain_id: str
    adapter_chain_sha256: str
    quantization_id: str
    quantization_sha256: str
    normalization: Normalization = Normalization.L2
    metric: DistanceMetric = DistanceMetric.COSINE
    dtype: VectorDType = VectorDType.FLOAT32
    abi_version: int = REPRESENTATION_ABI_VERSION

    def __post_init__(self) -> None:
        version = _positive_int(self.abi_version, "abi_version")
        if version != REPRESENTATION_ABI_VERSION:
            raise RepresentationABIError(
                f"unsupported representation ABI version {version}; "
                f"this runtime supports {REPRESENTATION_ABI_VERSION}"
            )
        dimension = _positive_int(self.dimension, "dimension")
        if dimension > 2_147_483_647:
            raise RepresentationABIError("dimension exceeds the signed 32-bit ABI limit")

        object.__setattr__(self, "encoder_id", _canonical_text(self.encoder_id, "encoder_id"))
        object.__setattr__(
            self,
            "encoder_revision",
            _canonical_text(self.encoder_revision, "encoder_revision"),
        )
        object.__setattr__(
            self,
            "encoder_weights_sha256",
            _sha256_digest(self.encoder_weights_sha256, "encoder_weights_sha256"),
        )
        object.__setattr__(self, "tokenizer_id", _canonical_text(self.tokenizer_id, "tokenizer_id"))
        object.__setattr__(
            self,
            "tokenizer_sha256",
            _sha256_digest(self.tokenizer_sha256, "tokenizer_sha256"),
        )
        object.__setattr__(self, "role", _enum_value(RepresentationRole, self.role, "role"))
        object.__setattr__(
            self,
            "compatibility_domain_id",
            _canonical_text(self.compatibility_domain_id, "compatibility_domain_id"),
        )
        object.__setattr__(
            self,
            "compatibility_domain_sha256",
            _sha256_digest(self.compatibility_domain_sha256, "compatibility_domain_sha256"),
        )
        object.__setattr__(
            self,
            "instruction_sha256",
            _sha256_digest(self.instruction_sha256, "instruction_sha256"),
        )
        object.__setattr__(self, "pooling", _enum_value(PoolingStrategy, self.pooling, "pooling"))
        object.__setattr__(
            self,
            "preprocessing_id",
            _canonical_text(self.preprocessing_id, "preprocessing_id"),
        )
        object.__setattr__(
            self,
            "preprocessing_sha256",
            _sha256_digest(self.preprocessing_sha256, "preprocessing_sha256"),
        )
        object.__setattr__(
            self,
            "projection_chain_id",
            _canonical_text(self.projection_chain_id, "projection_chain_id"),
        )
        object.__setattr__(
            self,
            "projection_chain_sha256",
            _sha256_digest(self.projection_chain_sha256, "projection_chain_sha256"),
        )
        object.__setattr__(
            self,
            "adapter_chain_id",
            _canonical_text(self.adapter_chain_id, "adapter_chain_id"),
        )
        object.__setattr__(
            self,
            "adapter_chain_sha256",
            _sha256_digest(self.adapter_chain_sha256, "adapter_chain_sha256"),
        )
        object.__setattr__(
            self,
            "quantization_id",
            _canonical_text(self.quantization_id, "quantization_id"),
        )
        object.__setattr__(
            self,
            "quantization_sha256",
            _sha256_digest(self.quantization_sha256, "quantization_sha256"),
        )
        object.__setattr__(
            self,
            "normalization",
            _enum_value(Normalization, self.normalization, "normalization"),
        )
        object.__setattr__(self, "metric", _enum_value(DistanceMetric, self.metric, "metric"))
        object.__setattr__(self, "dtype", _enum_value(VectorDType, self.dtype, "dtype"))
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "abi_version", version)

    def canonical_payload(self) -> Dict[str, object]:
        """Return the exact content-addressed ABI payload."""

        return {
            "record_type": "representation_descriptor",
            "abi_version": self.abi_version,
            "encoder_id": self.encoder_id,
            "encoder_revision": self.encoder_revision,
            "encoder_weights_sha256": self.encoder_weights_sha256,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_sha256": self.tokenizer_sha256,
            "role": self.role.value,
            "compatibility_domain_id": self.compatibility_domain_id,
            "compatibility_domain_sha256": self.compatibility_domain_sha256,
            "instruction_sha256": self.instruction_sha256,
            "pooling": self.pooling.value,
            "dimension": self.dimension,
            "preprocessing_id": self.preprocessing_id,
            "preprocessing_sha256": self.preprocessing_sha256,
            "projection_chain_id": self.projection_chain_id,
            "projection_chain_sha256": self.projection_chain_sha256,
            "adapter_chain_id": self.adapter_chain_id,
            "adapter_chain_sha256": self.adapter_chain_sha256,
            "quantization_id": self.quantization_id,
            "quantization_sha256": self.quantization_sha256,
            "normalization": self.normalization.value,
            "metric": self.metric.value,
            "dtype": self.dtype.value,
        }

    def canonical_json(self) -> str:
        """Return deterministic JSON used to derive ``stable_id``."""

        return _canonical_json(self.canonical_payload())

    @property
    def stable_id(self) -> str:
        """Content-addressed identity stable across processes and key orderings."""

        return _stable_id("rep", self.abi_version, self.canonical_payload())

    def to_dict(self) -> Dict[str, object]:
        payload = self.canonical_payload()
        payload["stable_id"] = self.stable_id
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RepresentationDescriptor":
        """Deserialize and verify a descriptor, including an optional claimed ID."""

        if not isinstance(raw, Mapping):
            raise RepresentationABIError("representation descriptor must be a mapping")
        required = {
            "record_type",
            "abi_version",
            "encoder_id",
            "encoder_revision",
            "encoder_weights_sha256",
            "tokenizer_id",
            "tokenizer_sha256",
            "role",
            "compatibility_domain_id",
            "compatibility_domain_sha256",
            "instruction_sha256",
            "pooling",
            "dimension",
            "preprocessing_id",
            "preprocessing_sha256",
            "projection_chain_id",
            "projection_chain_sha256",
            "adapter_chain_id",
            "adapter_chain_sha256",
            "quantization_id",
            "quantization_sha256",
            "normalization",
            "metric",
            "dtype",
        }
        allowed = required | {"stable_id"}
        missing = required - set(raw)
        unknown = set(raw) - allowed
        if missing:
            raise RepresentationABIError("representation descriptor is missing fields: " + ", ".join(sorted(missing)))
        if unknown:
            raise RepresentationABIError("representation descriptor has unknown fields: " + ", ".join(sorted(unknown)))
        if raw["record_type"] != "representation_descriptor":
            raise RepresentationABIError("record_type must be 'representation_descriptor'")

        descriptor = cls(
            encoder_id=raw["encoder_id"],
            encoder_revision=raw["encoder_revision"],
            encoder_weights_sha256=raw["encoder_weights_sha256"],
            tokenizer_id=raw["tokenizer_id"],
            tokenizer_sha256=raw["tokenizer_sha256"],
            role=raw["role"],
            compatibility_domain_id=raw["compatibility_domain_id"],
            compatibility_domain_sha256=raw["compatibility_domain_sha256"],
            instruction_sha256=raw["instruction_sha256"],
            pooling=raw["pooling"],
            dimension=raw["dimension"],
            preprocessing_id=raw["preprocessing_id"],
            preprocessing_sha256=raw["preprocessing_sha256"],
            projection_chain_id=raw["projection_chain_id"],
            projection_chain_sha256=raw["projection_chain_sha256"],
            adapter_chain_id=raw["adapter_chain_id"],
            adapter_chain_sha256=raw["adapter_chain_sha256"],
            quantization_id=raw["quantization_id"],
            quantization_sha256=raw["quantization_sha256"],
            normalization=raw["normalization"],
            metric=raw["metric"],
            dtype=raw["dtype"],
            abi_version=raw["abi_version"],
        )
        claimed_id = raw.get("stable_id")
        if claimed_id is not None and claimed_id != descriptor.stable_id:
            raise RepresentationABIError(
                f"stable_id mismatch: claimed {claimed_id!r}, computed {descriptor.stable_id!r}"
            )
        return descriptor


@dataclass(frozen=True)
class ResidentIndexDescriptor:
    """Immutable identity of a concrete vector-store snapshot.

    Representation identity alone cannot identify a semantic cache: two stores
    built by the same encoder can contain different corpora or use different
    build parameters.  This contract binds both and is therefore required at the
    resolver boundary.

    A native query/document pair is deliberately fail-closed: its dimensions,
    distance metrics, normalization contracts, and vector dtypes must match
    exactly.  Although particular backends may support implicit normalization or
    dtype conversion, this ABI does not authorize either operation.  Such a
    conversion must be represented by a separately validated bridge or by a new
    content-addressed representation descriptor.
    """

    index_id: str
    document_space: RepresentationDescriptor
    query_space: RepresentationDescriptor
    corpus_snapshot_id: str
    corpus_snapshot_sha256: str
    vector_build_id: str
    vector_build_sha256: str
    document_count: int
    contract_version: int = RESIDENT_INDEX_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.document_space, RepresentationDescriptor):
            raise RepresentationABIError("document_space must be a RepresentationDescriptor")
        if not isinstance(self.query_space, RepresentationDescriptor):
            raise RepresentationABIError("query_space must be a RepresentationDescriptor")
        if self.document_space.role not in {RepresentationRole.DOCUMENT, RepresentationRole.SHARED}:
            raise RepresentationABIError("document_space role must be document or shared")
        if self.query_space.role not in {RepresentationRole.QUERY, RepresentationRole.SHARED}:
            raise RepresentationABIError("query_space role must be query or shared")
        document_domain = (
            self.document_space.compatibility_domain_id,
            self.document_space.compatibility_domain_sha256,
        )
        query_domain = (
            self.query_space.compatibility_domain_id,
            self.query_space.compatibility_domain_sha256,
        )
        if document_domain != query_domain:
            raise RepresentationABIError(
                "document_space and query_space must share an exact compatibility domain ID and digest"
            )
        if self.document_space.dimension != self.query_space.dimension:
            raise RepresentationABIError("document_space and query_space must have equal dimensions for native search")
        if self.document_space.metric is not self.query_space.metric:
            raise RepresentationABIError(
                "document_space and query_space must use the same distance metric for native search"
            )
        if self.document_space.normalization is not self.query_space.normalization:
            raise RepresentationABIError(
                "document_space and query_space must use the same normalization for native search"
            )
        if self.document_space.dtype is not self.query_space.dtype:
            raise RepresentationABIError(
                "document_space and query_space must use the same vector dtype for native search"
            )
        version = _positive_int(self.contract_version, "contract_version")
        if version != RESIDENT_INDEX_CONTRACT_VERSION:
            raise RepresentationABIError(
                f"unsupported resident-index contract version {version}; "
                f"this runtime supports {RESIDENT_INDEX_CONTRACT_VERSION}"
            )
        object.__setattr__(self, "index_id", _canonical_text(self.index_id, "index_id"))
        object.__setattr__(
            self,
            "corpus_snapshot_id",
            _canonical_text(self.corpus_snapshot_id, "corpus_snapshot_id"),
        )
        object.__setattr__(
            self,
            "corpus_snapshot_sha256",
            _sha256_digest(self.corpus_snapshot_sha256, "corpus_snapshot_sha256"),
        )
        object.__setattr__(
            self,
            "vector_build_id",
            _canonical_text(self.vector_build_id, "vector_build_id"),
        )
        object.__setattr__(
            self,
            "vector_build_sha256",
            _sha256_digest(self.vector_build_sha256, "vector_build_sha256"),
        )
        object.__setattr__(self, "document_count", _nonnegative_int(self.document_count, "document_count"))
        object.__setattr__(self, "contract_version", version)

    def canonical_payload(self) -> Dict[str, object]:
        return {
            "record_type": "resident_vector_index",
            "contract_version": self.contract_version,
            "index_id": self.index_id,
            "document_space_id": self.document_space.stable_id,
            "query_space_id": self.query_space.stable_id,
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "corpus_snapshot_sha256": self.corpus_snapshot_sha256,
            "vector_build_id": self.vector_build_id,
            "vector_build_sha256": self.vector_build_sha256,
            "document_count": self.document_count,
        }

    @property
    def stable_id(self) -> str:
        return _stable_id("index", self.contract_version, self.canonical_payload())

    def to_dict(self) -> Dict[str, object]:
        payload = self.canonical_payload()
        payload["document_space"] = self.document_space.to_dict()
        payload["query_space"] = self.query_space.to_dict()
        payload["stable_id"] = self.stable_id
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ResidentIndexDescriptor":
        if not isinstance(raw, Mapping):
            raise RepresentationABIError("resident index descriptor must be a mapping")
        required = {
            "record_type",
            "contract_version",
            "index_id",
            "document_space_id",
            "document_space",
            "query_space_id",
            "query_space",
            "corpus_snapshot_id",
            "corpus_snapshot_sha256",
            "vector_build_id",
            "vector_build_sha256",
            "document_count",
        }
        allowed = required | {"stable_id"}
        missing = required - set(raw)
        unknown = set(raw) - allowed
        if missing:
            raise RepresentationABIError("resident index descriptor is missing fields: " + ", ".join(sorted(missing)))
        if unknown:
            raise RepresentationABIError("resident index descriptor has unknown fields: " + ", ".join(sorted(unknown)))
        if raw["record_type"] != "resident_vector_index":
            raise RepresentationABIError("record_type must be 'resident_vector_index'")

        document_space = RepresentationDescriptor.from_dict(raw["document_space"])
        query_space = RepresentationDescriptor.from_dict(raw["query_space"])
        if raw["document_space_id"] != document_space.stable_id:
            raise RepresentationABIError("document_space_id does not match the nested document-space descriptor")
        if raw["query_space_id"] != query_space.stable_id:
            raise RepresentationABIError("query_space_id does not match the nested query-space descriptor")
        descriptor = cls(
            index_id=raw["index_id"],
            document_space=document_space,
            query_space=query_space,
            corpus_snapshot_id=raw["corpus_snapshot_id"],
            corpus_snapshot_sha256=raw["corpus_snapshot_sha256"],
            vector_build_id=raw["vector_build_id"],
            vector_build_sha256=raw["vector_build_sha256"],
            document_count=raw["document_count"],
            contract_version=raw["contract_version"],
        )
        claimed_id = raw.get("stable_id")
        if claimed_id is not None and claimed_id != descriptor.stable_id:
            raise RepresentationABIError(
                f"stable_id mismatch: claimed {claimed_id!r}, computed {descriptor.stable_id!r}"
            )
        return descriptor

    @property
    def representation(self) -> RepresentationDescriptor:
        """Stored document-vector representation (compatibility alias)."""

        return self.document_space


@dataclass(frozen=True)
class BridgeContract:
    """Immutable, directed contract for a representation-space transform.

    ``source`` and ``target`` are directional.  ``cost`` states whether the
    transform is safe to apply to a query without corpus writes or instead
    requires a separately materialized target-space index.  Only ``VALIDATED``
    contracts with evidence are eligible for search routing.
    """

    source: RepresentationDescriptor
    target: RepresentationDescriptor
    role: RepresentationRole
    transform_family: str
    transform_id: str
    weights_sha256: str
    hyperparameters_sha256: str
    fit_anchor_manifest_sha256: str
    fit_count: int
    validation_count: int
    validation_state: BridgeValidationState
    cost: BridgeCost
    evidence_id: Optional[str] = None
    evidence_sha256: Optional[str] = None
    validation_confidence: float = 0.0
    contract_version: int = BRIDGE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.source, RepresentationDescriptor):
            raise BridgeContractError("source must be a RepresentationDescriptor")
        if not isinstance(self.target, RepresentationDescriptor):
            raise BridgeContractError("target must be a RepresentationDescriptor")
        if self.source.stable_id == self.target.stable_id:
            raise BridgeContractError("bridge source and target must be distinct representation spaces")
        role = _enum_value(RepresentationRole, self.role, "role")
        if self.source.role is not role or self.target.role is not role:
            raise BridgeContractError("bridge role must exactly match both source and target representation roles")

        version = _positive_int(self.contract_version, "contract_version")
        if version != BRIDGE_CONTRACT_VERSION:
            raise BridgeContractError(
                f"unsupported bridge contract version {version}; " f"this runtime supports {BRIDGE_CONTRACT_VERSION}"
            )
        transform_family = _canonical_text(self.transform_family, "transform_family")
        transform_id = _canonical_text(self.transform_id, "transform_id")
        weights_sha256 = _sha256_digest(self.weights_sha256, "weights_sha256")
        hyperparameters_sha256 = _sha256_digest(
            self.hyperparameters_sha256,
            "hyperparameters_sha256",
        )
        fit_anchor_manifest_sha256 = _sha256_digest(
            self.fit_anchor_manifest_sha256,
            "fit_anchor_manifest_sha256",
        )
        fit_count = _nonnegative_int(self.fit_count, "fit_count")
        validation_count = _nonnegative_int(self.validation_count, "validation_count")
        state = _enum_value(BridgeValidationState, self.validation_state, "validation_state")
        cost = _enum_value(BridgeCost, self.cost, "cost")
        if role is RepresentationRole.QUERY and cost is not BridgeCost.ZERO_WRITE:
            raise BridgeContractError("query-role bridges must have zero_write cost")
        if role is RepresentationRole.DOCUMENT and cost is not BridgeCost.MATERIALIZED_INDEX:
            raise BridgeContractError("document-role bridges must have materialized_index cost")
        confidence = _confidence(self.validation_confidence, "validation_confidence")
        evidence_id = self.evidence_id
        if evidence_id is not None:
            evidence_id = _canonical_text(evidence_id, "evidence_id", maximum_length=1024)
        evidence_sha256 = self.evidence_sha256
        if evidence_sha256 is not None:
            evidence_sha256 = _sha256_digest(evidence_sha256, "evidence_sha256")

        if state is BridgeValidationState.UNVALIDATED:
            if evidence_id is not None or evidence_sha256 is not None:
                raise BridgeContractError("unvalidated bridges must not claim an evidence_id or evidence_sha256")
            if confidence != 0.0:
                raise BridgeContractError("unvalidated bridges must have validation_confidence=0")
        elif state is BridgeValidationState.VALIDATED:
            if evidence_id is None:
                raise BridgeContractError("validated bridges require a non-empty evidence_id")
            if evidence_sha256 is None:
                raise BridgeContractError("validated bridges require evidence_sha256")
            if confidence <= 0.0:
                raise BridgeContractError("validated bridges require validation_confidence > 0")
            if fit_count <= 0:
                raise BridgeContractError("validated bridges require fit_count > 0")
            if validation_count <= 0:
                raise BridgeContractError("validated bridges require validation_count > 0")
        else:
            if evidence_id is None or evidence_sha256 is None:
                raise BridgeContractError(
                    f"{state.value} bridges require content-addressed evidence explaining that state"
                )
            if confidence != 0.0:
                raise BridgeContractError(f"{state.value} bridges must have validation_confidence=0")

        object.__setattr__(self, "role", role)
        object.__setattr__(self, "transform_family", transform_family)
        object.__setattr__(self, "transform_id", transform_id)
        object.__setattr__(self, "weights_sha256", weights_sha256)
        object.__setattr__(self, "hyperparameters_sha256", hyperparameters_sha256)
        object.__setattr__(self, "fit_anchor_manifest_sha256", fit_anchor_manifest_sha256)
        object.__setattr__(self, "fit_count", fit_count)
        object.__setattr__(self, "validation_count", validation_count)
        object.__setattr__(self, "validation_state", state)
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "evidence_id", evidence_id)
        object.__setattr__(self, "evidence_sha256", evidence_sha256)
        object.__setattr__(self, "validation_confidence", confidence)
        object.__setattr__(self, "contract_version", version)

    @property
    def is_usable(self) -> bool:
        return self.validation_state is BridgeValidationState.VALIDATED

    def canonical_payload(self) -> Dict[str, object]:
        return {
            "record_type": "representation_bridge",
            "contract_version": self.contract_version,
            "source_space_id": self.source.stable_id,
            "target_space_id": self.target.stable_id,
            "role": self.role.value,
            "transform_family": self.transform_family,
            "transform_id": self.transform_id,
            "weights_sha256": self.weights_sha256,
            "hyperparameters_sha256": self.hyperparameters_sha256,
            "fit_anchor_manifest_sha256": self.fit_anchor_manifest_sha256,
            "fit_count": self.fit_count,
            "validation_count": self.validation_count,
            "validation_state": self.validation_state.value,
            "cost": self.cost.value,
            "evidence_id": self.evidence_id,
            "evidence_sha256": self.evidence_sha256,
            "validation_confidence": self.validation_confidence,
        }

    @property
    def stable_id(self) -> str:
        return _stable_id("bridge", self.contract_version, self.canonical_payload())

    @property
    def bridge_id(self) -> str:
        """Alias used by resolvers and materialization registries."""

        return self.stable_id

    def to_dict(self) -> Dict[str, object]:
        payload = self.canonical_payload()
        payload.update(
            {
                "source": self.source.to_dict(),
                "target": self.target.to_dict(),
                "stable_id": self.stable_id,
            }
        )
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BridgeContract":
        """Deserialize a bridge and reject altered nested descriptors or IDs."""

        if not isinstance(raw, Mapping):
            raise BridgeContractError("bridge contract must be a mapping")
        required = {
            "record_type",
            "contract_version",
            "source_space_id",
            "target_space_id",
            "source",
            "target",
            "role",
            "transform_family",
            "transform_id",
            "weights_sha256",
            "hyperparameters_sha256",
            "fit_anchor_manifest_sha256",
            "fit_count",
            "validation_count",
            "validation_state",
            "cost",
            "evidence_id",
            "evidence_sha256",
            "validation_confidence",
        }
        allowed = required | {"stable_id"}
        missing = required - set(raw)
        unknown = set(raw) - allowed
        if missing:
            raise BridgeContractError("bridge contract is missing fields: " + ", ".join(sorted(missing)))
        if unknown:
            raise BridgeContractError("bridge contract has unknown fields: " + ", ".join(sorted(unknown)))
        if raw["record_type"] != "representation_bridge":
            raise BridgeContractError("record_type must be 'representation_bridge'")

        source = RepresentationDescriptor.from_dict(raw["source"])
        target = RepresentationDescriptor.from_dict(raw["target"])
        if raw["source_space_id"] != source.stable_id:
            raise BridgeContractError("source_space_id does not match the nested source descriptor")
        if raw["target_space_id"] != target.stable_id:
            raise BridgeContractError("target_space_id does not match the nested target descriptor")

        contract = cls(
            source=source,
            target=target,
            role=raw["role"],
            transform_family=raw["transform_family"],
            transform_id=raw["transform_id"],
            weights_sha256=raw["weights_sha256"],
            hyperparameters_sha256=raw["hyperparameters_sha256"],
            fit_anchor_manifest_sha256=raw["fit_anchor_manifest_sha256"],
            fit_count=raw["fit_count"],
            validation_count=raw["validation_count"],
            validation_state=raw["validation_state"],
            cost=raw["cost"],
            evidence_id=raw["evidence_id"],
            evidence_sha256=raw["evidence_sha256"],
            validation_confidence=raw["validation_confidence"],
            contract_version=raw["contract_version"],
        )
        claimed_id = raw.get("stable_id")
        if claimed_id is not None and claimed_id != contract.stable_id:
            raise BridgeContractError(f"stable_id mismatch: claimed {claimed_id!r}, computed {contract.stable_id!r}")
        return contract
