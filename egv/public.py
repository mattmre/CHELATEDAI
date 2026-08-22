"""Closed public projections and public cryptographic decision replay.

This module is intentionally separate from the private ledger.  It accepts only
the allowlisted public records defined by ADR-0001 and never attempts to infer
hidden-test correctness, physical effect execution, or private content from a
digest.  A recorded disposition is checked only after the verifier computes its
own disposition from signed facts and the corrected dependency graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from .canonical import (
    GENESIS_HASH,
    canonical_bytes,
    canonical_json,
    collection_digest,
    content_id,
    digest_for,
    failure_family_root,
    parse_canonical_jsonl,
    validate_sha256,
    write_canonical_jsonl,
)
from .errors import PublicReplayError, PublicSchemaError, ReceiptVerificationError
from .receipts import (
    PUBLIC_RECEIPT_SCHEMA_VERSION,
    PUBLIC_EXIT_STATUS_CLASSES,
    RECEIPT_DECISIONS,
    RECEIPT_TYPES,
    key_id_for_public_key,
    load_public_key,
    receipt_hash,
)


PUBLIC_CANDIDATE_SCHEMA_VERSION = "egv-public-candidate-v1"
PUBLIC_DEPENDENCY_SCHEMA_VERSION = "egv-public-dependency-v1"
PUBLIC_EVENT_SCHEMA_VERSION = "egv-public-event-v1"
PUBLIC_RESTORE_SCHEMA_VERSION = "egv-public-restore-v1"

PUBLIC_DIAGNOSTICS = frozenset(
    {
        "PASS",
        "WRONG_OUTPUT",
        "SYNTAX_OR_IMPORT",
        "RUNTIME_EXCEPTION",
        "TIMEOUT",
        "RESOURCE_LIMIT",
        "AUTHORITY_DENIED",
        "MUTATION_LOCUS_VIOLATION",
        "PROTOCOL_VIOLATION",
        "INTERNAL_ERROR",
    }
)
RESOURCE_BUCKETS = frozenset(
    {"UNDER_25", "25_TO_50", "50_TO_75", "75_TO_100", "LIMIT_REACHED", "OUTPUT_LIMIT"}
)
EXIT_STATUS_CLASSES = PUBLIC_EXIT_STATUS_CLASSES
PUBLIC_REASONS = frozenset(
    {
        "EVALUATOR_RULE_CORRECTED",
        "PREMISE_RETRACTED",
        "INVALID_RECEIPT",
        "STALE_DEPENDENCY",
        "PROTOCOL_INVALID",
    }
)
PUBLIC_DISPOSITIONS = frozenset({"PROMOTED", "REJECTED", "ABSTAINED", "STALE_DEPENDENT"})
PUBLIC_SOURCE_CLASSES = frozenset({"FROZEN_EVALUATOR", "FROZEN_PROTOCOL"})
PUBLIC_EDGE_TYPES = frozenset({"EVIDENCE_USE", "RETRIEVED", "DEPENDS_ON", "CORRECTED_BY", "DERIVED_FROM"})

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}$")
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_TOPOLOGY_SINGLE_LABELS = frozenset(
    {
        "db",
        "database",
        "evaluator",
        "gateway",
        "host",
        "hostname",
        "ip6-localhost",
        "localhost",
        "postgres",
        "qdrant",
        "redis",
        "router",
        "spark",
        "trainer",
    }
)
_TOPOLOGY_TOKENS = _TOPOLOGY_SINGLE_LABELS | frozenset(
    {
        "container",
        "node",
        "pod",
        "service",
        "worker",
    }
)


def _require_safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise PublicSchemaError(f"{field} must be a bounded pseudonymous ID")
    if ".." in value or "/" in value or "\\" in value:
        raise PublicSchemaError(f"{field} contains a prohibited path-like value")
    return value


def _require_pseudonymous_id(value: Any, field: str) -> str:
    """Require an opaque public ID that cannot encode service topology."""

    value = _require_safe_id(value, field)
    lowered = value.lower()
    try:
        ipaddress.ip_address(value)
    except ValueError:
        pass
    else:
        raise PublicSchemaError(f"{field} must be pseudonymous, not an IP address")
    if value.isdecimal():
        raise PublicSchemaError(f"{field} must be pseudonymous, not a port-shaped value")
    if ":" in value or _URI_SCHEME.match(value):
        raise PublicSchemaError(f"{field} must be pseudonymous, not a URI or host-port")
    if "." in value:
        raise PublicSchemaError(f"{field} must be pseudonymous, not a hostname-shaped value")
    if lowered in _TOPOLOGY_SINGLE_LABELS:
        raise PublicSchemaError(f"{field} must be pseudonymous, not a topology label")
    tokens = [token for token in re.split(r"[-_]", lowered) if token]
    if any(token in _TOPOLOGY_TOKENS for token in tokens):
        raise PublicSchemaError(f"{field} must be pseudonymous, not a topology-shaped value")
    if re.fullmatch(r"(?:host|node|worker|spark|qdrant|evaluator|trainer|db|redis|postgres|service|container|pod|gateway|router)\d+", lowered):
        raise PublicSchemaError(f"{field} must be pseudonymous, not a topology-shaped hostname")
    return value


def _require_digest(value: Any, field: str) -> str:
    try:
        return validate_sha256(value, field)
    except Exception as exc:
        raise PublicSchemaError(str(exc)) from exc


def _strict_fields(record: Mapping[str, Any], allowed: Set[str], required: Set[str], label: str) -> None:
    if not isinstance(record, Mapping):
        raise PublicSchemaError(f"{label} must be an object")
    missing = sorted(required - set(record))
    extra = sorted(set(record) - allowed)
    if missing:
        raise PublicSchemaError(f"{label} is missing fields: {', '.join(missing)}")
    if extra:
        raise PublicSchemaError(f"{label} contains forbidden fields: {', '.join(extra)}")


PUBLIC_CANDIDATE_FIELDS = {
    "campaign_id",
    "run_id",
    "task_id",
    "arm",
    "attempt_index",
    "candidate_id",
    "parent_candidate_id",
    "candidate_artifact_digest",
    "model_digest",
    "adapter_digest",
    "prompt_template_digests",
    "mutation_family",
    "normalized_public_locus",
    "requested_authority",
    "declared_public_evidence_ids",
    "public_dependency_ids",
}
PUBLIC_CANDIDATE_REQUIRED = set(PUBLIC_CANDIDATE_FIELDS)


def validate_public_candidate(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate one closed public candidate record and return a detached dict."""

    _strict_fields(record, PUBLIC_CANDIDATE_FIELDS, PUBLIC_CANDIDATE_REQUIRED, "public candidate record")
    for field in ("campaign_id", "run_id", "task_id", "arm", "candidate_id"):
        _require_pseudonymous_id(record[field], field)
    for field in ("mutation_family", "requested_authority"):
        _require_safe_id(record[field], field)
    if record["parent_candidate_id"] is not None:
        _require_pseudonymous_id(record["parent_candidate_id"], "parent_candidate_id")
    if not isinstance(record["attempt_index"], int) or isinstance(record["attempt_index"], bool) or record["attempt_index"] < 0:
        raise PublicSchemaError("attempt_index must be a nonnegative integer")
    for field in ("candidate_artifact_digest", "model_digest", "adapter_digest"):
        _require_digest(record[field], field)
    if not isinstance(record["prompt_template_digests"], list) or not record["prompt_template_digests"]:
        raise PublicSchemaError("prompt_template_digests must be a non-empty array")
    for digest in record["prompt_template_digests"]:
        _require_digest(digest, "prompt_template_digests item")
    _require_safe_id(record["normalized_public_locus"], "normalized_public_locus")
    if not isinstance(record["declared_public_evidence_ids"], list):
        raise PublicSchemaError("declared_public_evidence_ids must be an array")
    if not isinstance(record["public_dependency_ids"], list):
        raise PublicSchemaError("public_dependency_ids must be an array")
    for field in ("declared_public_evidence_ids", "public_dependency_ids"):
        values = record[field]
        if field == "public_dependency_ids" and values != sorted(values):
            raise PublicSchemaError(f"{field} must be lexicographically sorted")
        for value in values:
            _require_pseudonymous_id(value, f"{field} item")
    return dict(record)


def public_candidate_digest(record: Mapping[str, Any]) -> str:
    return digest_for(validate_public_candidate(record))


PUBLIC_DEPENDENCY_FIELDS = {
    "parent_id",
    "child_id",
    "edge_type",
    "insertion_receipt_id",
}
PUBLIC_DEPENDENCY_REQUIRED = set(PUBLIC_DEPENDENCY_FIELDS)


def validate_public_dependency(record: Mapping[str, Any]) -> Dict[str, Any]:
    _strict_fields(record, PUBLIC_DEPENDENCY_FIELDS, PUBLIC_DEPENDENCY_REQUIRED, "public dependency record")
    for field in ("parent_id", "child_id", "insertion_receipt_id"):
        _require_pseudonymous_id(record[field], field)
    if record["edge_type"] not in PUBLIC_EDGE_TYPES:
        raise PublicSchemaError(f"unsupported public dependency edge type: {record['edge_type']!r}")
    if record["parent_id"] == record["child_id"]:
        raise PublicSchemaError("public dependency cannot point to itself")
    return dict(record)


def public_dependency_id(record: Mapping[str, Any]) -> str:
    return content_id("dep", validate_public_dependency(record))


def public_dependency_set_digest(records: Iterable[Mapping[str, Any]]) -> str:
    normalized = [validate_public_dependency(record) for record in records]
    normalized.sort(key=public_dependency_id)
    return collection_digest(normalized)


def public_candidate_collection_digest(records: Iterable[Mapping[str, Any]]) -> str:
    normalized = [validate_public_candidate(record) for record in records]
    normalized.sort(key=lambda record: record["candidate_id"])
    return collection_digest(normalized)


def public_dependency_collection_digest(records: Iterable[Mapping[str, Any]]) -> str:
    normalized = [validate_public_dependency(record) for record in records]
    normalized.sort(key=public_dependency_id)
    return collection_digest(normalized)


PUBLIC_RECEIPT_OPTIONAL_FIELDS = {
    "diagnostic_enum",
    "resource_bucket",
    "exit_status_class",
    "input_digest",
    "output_digest",
    "environment_diff_digest",
    "infrastructure_incident_id",
    "failure_family_root",
    "task_family",
    "normalized_public_locus",
    "public_rule_id",
}
PUBLIC_RECEIPT_FIELDS = {
    "schema_version",
    "receipt_type",
    "campaign_id",
    "run_id",
    "task_id",
    "receipt_id",
    "request_id",
    "candidate_id",
    "candidate_artifact_digest",
    "protocol_digest",
    "policy_digest",
    "evaluator_digest",
    "public_candidate_record_digest",
    "public_dependency_set_digest",
    "decision",
    "diagnostic_enum",
    "resource_bucket",
    "exit_status_class",
    "input_digest",
    "output_digest",
    "environment_diff_digest",
    "infrastructure_incident_id",
    "failure_family_root",
    "task_family",
    "normalized_public_locus",
    "public_rule_id",
    "public_sequence",
    "previous_public_receipt_digest",
    "signing_key_id",
    "signature",
}
PUBLIC_RECEIPT_REQUIRED = {
    "schema_version",
    "receipt_type",
    "campaign_id",
    "run_id",
    "task_id",
    "receipt_id",
    "request_id",
    "candidate_id",
    "candidate_artifact_digest",
    "protocol_digest",
    "policy_digest",
    "evaluator_digest",
    "public_candidate_record_digest",
    "public_dependency_set_digest",
    "decision",
    "public_sequence",
    "previous_public_receipt_digest",
    "signing_key_id",
    "signature",
}


def validate_public_receipt(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    _strict_fields(receipt, PUBLIC_RECEIPT_FIELDS, PUBLIC_RECEIPT_REQUIRED, "public receipt envelope")
    if receipt["schema_version"] != PUBLIC_RECEIPT_SCHEMA_VERSION:
        raise PublicSchemaError("unsupported public receipt schema")
    if receipt["receipt_type"] not in RECEIPT_TYPES:
        raise PublicSchemaError("unsupported public receipt type")
    if receipt["decision"] not in RECEIPT_DECISIONS[receipt["receipt_type"]]:
        raise PublicSchemaError("receipt decision is invalid for its type")
    for field in ("campaign_id", "run_id", "task_id", "receipt_id", "request_id", "candidate_id", "signing_key_id"):
        _require_pseudonymous_id(receipt[field], field)
    for field in (
        "candidate_artifact_digest",
        "protocol_digest",
        "policy_digest",
        "evaluator_digest",
        "public_candidate_record_digest",
        "public_dependency_set_digest",
        "input_digest",
        "output_digest",
        "environment_diff_digest",
        "failure_family_root",
    ):
        if field in receipt:
            _require_digest(receipt[field], field)
    if not isinstance(receipt["public_sequence"], int) or isinstance(receipt["public_sequence"], bool) or receipt["public_sequence"] < 1:
        raise PublicSchemaError("public_sequence must be a positive integer")
    previous = receipt["previous_public_receipt_digest"]
    if previous != GENESIS_HASH:
        _require_digest(previous, "previous_public_receipt_digest")
    if "diagnostic_enum" in receipt and receipt["diagnostic_enum"] not in PUBLIC_DIAGNOSTICS:
        raise PublicSchemaError("unknown diagnostic enum")
    if "resource_bucket" in receipt and receipt["resource_bucket"] not in RESOURCE_BUCKETS:
        raise PublicSchemaError("unknown resource bucket")
    if "exit_status_class" in receipt and receipt["exit_status_class"] not in EXIT_STATUS_CLASSES:
        raise PublicSchemaError("unknown exit status class")
    if "infrastructure_incident_id" in receipt:
        _require_pseudonymous_id(receipt["infrastructure_incident_id"], "infrastructure_incident_id")
    diagnostic = receipt.get("diagnostic_enum")
    incident = receipt.get("infrastructure_incident_id")
    root = receipt.get("failure_family_root")
    if diagnostic == "INTERNAL_ERROR":
        if not isinstance(incident, str) or not incident:
            raise PublicSchemaError("INTERNAL_ERROR public receipts require infrastructure_incident_id")
        if not isinstance(root, str) or not root:
            raise PublicSchemaError("INTERNAL_ERROR public receipts require an incident-bound failure_family_root")
        canonical_fields = ("task_family", "normalized_public_locus", "public_rule_id")
        missing_canonical = [field for field in canonical_fields if not isinstance(receipt.get(field), str) or not receipt[field]]
        if missing_canonical:
            raise PublicSchemaError(
                "INTERNAL_ERROR public receipts require canonical fields: {}".format(", ".join(missing_canonical))
            )
        _require_pseudonymous_id(receipt["task_family"], "task_family")
        _require_safe_id(receipt["normalized_public_locus"], "normalized_public_locus")
        _require_pseudonymous_id(receipt["public_rule_id"], "public_rule_id")
        expected_root = failure_family_root(
            receipt["task_family"],
            diagnostic,
            receipt["normalized_public_locus"],
            receipt["public_rule_id"],
            infrastructure_incident_id=incident,
        )
        if root != expected_root:
            raise PublicSchemaError("failure_family_root is not the exact incident-bound canonical root")
    elif incident is not None or root is not None:
        raise PublicSchemaError("infrastructure incident and failure root are reserved for INTERNAL_ERROR")
    if not isinstance(receipt["signature"], str) or not receipt["signature"]:
        raise PublicSchemaError("signature must be non-empty")
    return dict(receipt)


def build_public_receipt_envelope(payload: Mapping[str, Any], signer: Any) -> Dict[str, Any]:
    """Build and sign a public receipt from public fields only."""

    candidate = dict(payload)
    candidate["schema_version"] = PUBLIC_RECEIPT_SCHEMA_VERSION
    candidate["signing_key_id"] = signer.key_id
    candidate.pop("signature", None)
    supplied_id = candidate.pop("receipt_id", None)
    derived_id = content_id("pubrcpt", candidate)
    if supplied_id is not None and supplied_id != derived_id:
        raise PublicSchemaError("receipt_id is not content-derived from its public envelope")
    candidate["receipt_id"] = derived_id
    validate_public_receipt({**candidate, "signature": "placeholder"})
    signature = signer.sign_bytes(canonical_bytes(candidate))
    result = {**candidate, "signature": signature}
    validate_public_receipt(result)
    return result


def public_receipt_digest(receipt: Mapping[str, Any]) -> str:
    return digest_for(validate_public_receipt(receipt))


def verify_public_receipt(receipt: Mapping[str, Any], public_key: Any, *, expected_key_id: Optional[str] = None) -> str:
    from .receipts import _b64_decode

    normalized = validate_public_receipt(receipt)
    key = load_public_key(public_key)
    actual_key_id = key_id_for_public_key(key)
    if normalized["signing_key_id"] != actual_key_id:
        raise ReceiptVerificationError("public receipt key ID does not match public key")
    if expected_key_id is not None and normalized["signing_key_id"] != expected_key_id:
        raise ReceiptVerificationError("public receipt uses an unexpected key")
    unsigned = dict(normalized)
    unsigned.pop("signature")
    expected_id = content_id("pubrcpt", {key_name: value for key_name, value in unsigned.items() if key_name != "receipt_id"})
    if normalized["receipt_id"] != expected_id:
        raise PublicSchemaError("public receipt ID does not match signed content")
    try:
        key.verify(_b64_decode(normalized["signature"]), canonical_bytes(unsigned))
    except Exception as exc:
        if exc.__class__.__name__ == "InvalidSignature":
            raise ReceiptVerificationError("invalid public receipt signature") from exc
        raise
    return public_receipt_digest(normalized)


PUBLIC_EVENT_COMMON = {
    "schema_version",
    "event_type",
    "campaign_id",
    "run_id",
    "task_id",
    "event_id",
    "public_sequence",
    "previous_public_event_digest",
    "protocol_digest",
    "evaluator_digest",
    "signing_key_id",
    "signature",
}
PUBLIC_EVENT_FIELDS = PUBLIC_EVENT_COMMON | {
    "subject_id",
    "superseded_id",
    "replacement_id",
    "reason_code",
    "effective_after_attempt",
    "source_class",
    "authorizing_public_receipt_id",
    "recorded_disposition",
    "public_candidate_record_digest",
    "public_dependency_set_digest",
    "public_receipt_head_digest",
    "public_event_preseal_head_digest",
    "public_candidate_collection_digest",
    "public_dependency_collection_digest",
    "public_restore_receipt_digest",
}


def _event_type_fields(event_type: str) -> Tuple[Set[str], Set[str]]:
    if event_type == "CORRECTION":
        fields = PUBLIC_EVENT_COMMON | {
            "subject_id",
            "superseded_id",
            "replacement_id",
            "reason_code",
            "effective_after_attempt",
            "source_class",
            "authorizing_public_receipt_id",
        }
        return fields, fields - {"signature", "authorizing_public_receipt_id"}
    if event_type == "RETRACTION":
        fields = PUBLIC_EVENT_COMMON | {
            "subject_id",
            "reason_code",
            "effective_after_attempt",
            "source_class",
            "authorizing_public_receipt_id",
        }
        return fields, fields - {"signature", "authorizing_public_receipt_id"}
    if event_type == "RECORDED_DISPOSITION":
        fields = PUBLIC_EVENT_COMMON | {
            "subject_id",
            "source_class",
            "authorizing_public_receipt_id",
            "recorded_disposition",
            "public_candidate_record_digest",
            "public_dependency_set_digest",
        }
        return fields, fields - {"signature", "authorizing_public_receipt_id"}
    if event_type == "PUBLIC_CHAIN_SEAL":
        fields = PUBLIC_EVENT_COMMON | {
            "public_receipt_head_digest",
            "public_event_preseal_head_digest",
            "public_candidate_collection_digest",
            "public_dependency_collection_digest",
            "public_restore_receipt_digest",
        }
        return fields, fields - {"signature"}
    raise PublicSchemaError(f"unsupported public event type: {event_type!r}")


def validate_public_event(event: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(event, Mapping):
        raise PublicSchemaError("public event must be an object")
    event_type = event.get("event_type")
    allowed, required = _event_type_fields(event_type)
    _strict_fields(event, allowed, required, f"public {event_type.lower()} event")
    if event["schema_version"] != PUBLIC_EVENT_SCHEMA_VERSION:
        raise PublicSchemaError("unsupported public event schema")
    for field in ("campaign_id", "run_id", "task_id", "event_id", "signing_key_id"):
        _require_pseudonymous_id(event[field], field)
    for field in ("protocol_digest", "evaluator_digest"):
        _require_digest(event[field], field)
    if not isinstance(event["public_sequence"], int) or isinstance(event["public_sequence"], bool) or event["public_sequence"] < 1:
        raise PublicSchemaError("public event sequence must be a positive integer")
    previous = event["previous_public_event_digest"]
    if previous != GENESIS_HASH:
        _require_digest(previous, "previous_public_event_digest")
    if event_type != "PUBLIC_CHAIN_SEAL":
        _require_pseudonymous_id(event["subject_id"], "subject_id")
        if event["source_class"] not in PUBLIC_SOURCE_CLASSES:
            raise PublicSchemaError("unknown public event source class")
        if event["source_class"] == "FROZEN_EVALUATOR":
            if "authorizing_public_receipt_id" not in event:
                raise PublicSchemaError("evaluator-observed event requires an authorizing receipt")
            _require_pseudonymous_id(event["authorizing_public_receipt_id"], "authorizing_public_receipt_id")
        elif "authorizing_public_receipt_id" in event:
            raise PublicSchemaError("protocol-authorized event cannot carry evaluator receipt")
    if event_type in {"CORRECTION", "RETRACTION"}:
        _require_safe_id(event["reason_code"], "reason_code")
        if event["reason_code"] not in PUBLIC_REASONS:
            raise PublicSchemaError("unknown public lifecycle reason code")
        if not isinstance(event["effective_after_attempt"], int) or isinstance(event["effective_after_attempt"], bool) or event["effective_after_attempt"] < 0:
            raise PublicSchemaError("effective_after_attempt must be a nonnegative integer")
    if event_type == "CORRECTION":
        _require_pseudonymous_id(event["superseded_id"], "superseded_id")
        _require_pseudonymous_id(event["replacement_id"], "replacement_id")
    if event_type == "RECORDED_DISPOSITION":
        if event["recorded_disposition"] not in PUBLIC_DISPOSITIONS:
            raise PublicSchemaError("unknown recorded disposition")
        _require_digest(event["public_candidate_record_digest"], "public_candidate_record_digest")
        _require_digest(event["public_dependency_set_digest"], "public_dependency_set_digest")
    if event_type == "PUBLIC_CHAIN_SEAL":
        for field in (
            "public_receipt_head_digest",
            "public_event_preseal_head_digest",
            "public_candidate_collection_digest",
            "public_dependency_collection_digest",
            "public_restore_receipt_digest",
        ):
            _require_digest(event[field], field)
        if "subject_id" in event:
            raise PublicSchemaError("public chain seal cannot have subject_id")
    if not isinstance(event["signature"], str) or not event["signature"]:
        raise PublicSchemaError("public event signature must be non-empty")
    return dict(event)


def public_event_digest(event: Mapping[str, Any]) -> str:
    return digest_for(validate_public_event(event))


def build_public_event(payload: Mapping[str, Any], signer: Any) -> Dict[str, Any]:
    candidate = dict(payload)
    candidate["schema_version"] = PUBLIC_EVENT_SCHEMA_VERSION
    candidate["signing_key_id"] = signer.key_id
    candidate.pop("signature", None)
    supplied_id = candidate.pop("event_id", None)
    derived_id = content_id("pevt", candidate)
    if supplied_id is not None and supplied_id != derived_id:
        raise PublicSchemaError("event_id is not content-derived from its signed fields")
    candidate["event_id"] = derived_id
    validate_public_event({**candidate, "signature": "placeholder"})
    signed = signer.sign_bytes(canonical_bytes(candidate))
    result = {**candidate, "signature": signed}
    validate_public_event(result)
    return result


def verify_public_event(event: Mapping[str, Any], public_key: Any, *, expected_key_id: Optional[str] = None) -> str:
    from .receipts import _b64_decode

    normalized = validate_public_event(event)
    key = load_public_key(public_key)
    actual_key_id = key_id_for_public_key(key)
    if normalized["signing_key_id"] != actual_key_id:
        raise ReceiptVerificationError("public event key ID does not match public key")
    if expected_key_id is not None and normalized["signing_key_id"] != expected_key_id:
        raise ReceiptVerificationError("public event uses an unexpected key")
    unsigned = dict(normalized)
    unsigned.pop("signature")
    expected_id = content_id("pevt", {key_name: value for key_name, value in unsigned.items() if key_name != "event_id"})
    if normalized["event_id"] != expected_id:
        raise PublicSchemaError("public event ID does not match signed content")
    try:
        key.verify(_b64_decode(normalized["signature"]), canonical_bytes(unsigned))
    except Exception as exc:
        if exc.__class__.__name__ == "InvalidSignature":
            raise ReceiptVerificationError("invalid public event signature") from exc
        raise
    return public_event_digest(normalized)


RESTORE_FIELDS = {
    "schema_version",
    "campaign_id",
    "receipt_id",
    "logical_service_set_id",
    "logical_service_ids",
    "private_inventory_digest",
    "service_definition_set_digest",
    "model_set_digest",
    "configuration_set_digest",
    "executable_or_image_set_digest",
    "expected_service_count",
    "restored_service_count",
    "health_check_count",
    "health_pass_count",
    "all_health_checks_passed",
    "smoke_input_digest",
    "smoke_output_digest",
    "smoke_matches_baseline",
    "restoration_outcome",
    "signing_key_id",
    "signature",
}


def validate_public_restore_receipt(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    _strict_fields(receipt, RESTORE_FIELDS, RESTORE_FIELDS, "public restore receipt")
    if receipt["schema_version"] != PUBLIC_RESTORE_SCHEMA_VERSION:
        raise PublicSchemaError("unsupported public restore receipt schema")
    for field in ("campaign_id", "receipt_id", "logical_service_set_id", "signing_key_id"):
        _require_pseudonymous_id(receipt[field], field)
    for field in (
        "private_inventory_digest",
        "service_definition_set_digest",
        "model_set_digest",
        "configuration_set_digest",
        "executable_or_image_set_digest",
        "smoke_input_digest",
        "smoke_output_digest",
    ):
        _require_digest(receipt[field], field)
    services = receipt["logical_service_ids"]
    if not isinstance(services, list) or services != sorted(services) or not services:
        raise PublicSchemaError("logical_service_ids must be a sorted non-empty array")
    for service_id in services:
        _require_pseudonymous_id(service_id, "logical_service_ids item")
    for field in ("expected_service_count", "restored_service_count", "health_check_count", "health_pass_count"):
        if not isinstance(receipt[field], int) or isinstance(receipt[field], bool) or receipt[field] < 0:
            raise PublicSchemaError(f"{field} must be a nonnegative integer")
    if receipt["health_pass_count"] > receipt["health_check_count"]:
        raise PublicSchemaError("health_pass_count cannot exceed health_check_count")
    if receipt["restored_service_count"] > receipt["expected_service_count"]:
        raise PublicSchemaError("restored_service_count cannot exceed expected_service_count")
    if not isinstance(receipt["all_health_checks_passed"], bool) or not isinstance(receipt["smoke_matches_baseline"], bool):
        raise PublicSchemaError("restore booleans must be booleans")
    if receipt["restoration_outcome"] not in {"RESTORED", "PARTIAL", "FAILED"}:
        raise PublicSchemaError("unknown restoration outcome")
    return dict(receipt)


def build_public_restore_receipt(payload: Mapping[str, Any], signer: Any) -> Dict[str, Any]:
    candidate = dict(payload)
    candidate["schema_version"] = PUBLIC_RESTORE_SCHEMA_VERSION
    candidate["signing_key_id"] = signer.key_id
    candidate.pop("signature", None)
    supplied_id = candidate.pop("receipt_id", None)
    derived_id = content_id("restore", candidate)
    if supplied_id is not None and supplied_id != derived_id:
        raise PublicSchemaError("restore receipt ID is not content-derived")
    candidate["receipt_id"] = derived_id
    validate_public_restore_receipt({**candidate, "signature": "placeholder"})
    candidate["signature"] = signer.sign_bytes(canonical_bytes(candidate))
    validate_public_restore_receipt(candidate)
    return candidate


def public_restore_receipt_digest(receipt: Mapping[str, Any]) -> str:
    return digest_for(validate_public_restore_receipt(receipt))


def verify_public_restore_receipt(receipt: Mapping[str, Any], public_key: Any, *, expected_key_id: Optional[str] = None) -> str:
    from .receipts import _b64_decode

    normalized = validate_public_restore_receipt(receipt)
    key = load_public_key(public_key)
    actual_key_id = key_id_for_public_key(key)
    if normalized["signing_key_id"] != actual_key_id:
        raise ReceiptVerificationError("restore receipt key ID does not match public key")
    if expected_key_id is not None and normalized["signing_key_id"] != expected_key_id:
        raise ReceiptVerificationError("restore receipt uses an unexpected key")
    unsigned = dict(normalized)
    unsigned.pop("signature")
    expected_id = content_id("restore", {key_name: value for key_name, value in unsigned.items() if key_name != "receipt_id"})
    if normalized["receipt_id"] != expected_id:
        raise PublicSchemaError("restore receipt ID does not match signed content")
    try:
        key.verify(_b64_decode(normalized["signature"]), canonical_bytes(unsigned))
    except Exception as exc:
        if exc.__class__.__name__ == "InvalidSignature":
            raise ReceiptVerificationError("invalid restore receipt signature") from exc
        raise
    return public_restore_receipt_digest(normalized)


class PublicEventChain:
    """Builder/validator for the per-campaign signed public event chain."""

    def __init__(self, signer: Any, *, campaign_id: str, protocol_digest: str, evaluator_digest: str) -> None:
        self.signer = signer
        self.campaign_id = _require_pseudonymous_id(campaign_id, "campaign_id")
        self.protocol_digest = _require_digest(protocol_digest, "protocol_digest")
        self.evaluator_digest = _require_digest(evaluator_digest, "evaluator_digest")
        self._events: List[Dict[str, Any]] = []

    @property
    def events(self) -> List[Dict[str, Any]]:
        return [dict(event) for event in self._events]

    @property
    def head(self) -> str:
        return public_event_digest(self._events[-1]) if self._events else GENESIS_HASH

    def append(self, event: Mapping[str, Any]) -> Dict[str, Any]:
        normalized = validate_public_event(event)
        expected_sequence = len(self._events) + 1
        if normalized["campaign_id"] != self.campaign_id:
            raise PublicSchemaError("event campaign does not match chain")
        if normalized["protocol_digest"] != self.protocol_digest:
            raise PublicSchemaError("event protocol digest does not match chain")
        if normalized["evaluator_digest"] != self.evaluator_digest:
            raise PublicSchemaError("event evaluator digest does not match chain")
        if normalized["public_sequence"] != expected_sequence:
            raise PublicSchemaError("public event sequence is not contiguous")
        previous = self.head
        if normalized["previous_public_event_digest"] != previous:
            raise PublicSchemaError("public event previous digest does not match chain head")
        verify_public_event(normalized, self.signer.public_key, expected_key_id=self.signer.key_id)
        self._events.append(normalized)
        return dict(normalized)

    def _base(self, *, event_type: str, run_id: str, task_id: str) -> Dict[str, Any]:
        return {
            "event_type": event_type,
            "campaign_id": self.campaign_id,
            "run_id": _require_pseudonymous_id(run_id, "run_id"),
            "task_id": _require_pseudonymous_id(task_id, "task_id"),
            "public_sequence": len(self._events) + 1,
            "previous_public_event_digest": self.head,
            "protocol_digest": self.protocol_digest,
            "evaluator_digest": self.evaluator_digest,
        }

    def append_recorded_disposition(
        self,
        *,
        run_id: str,
        task_id: str,
        candidate_id: str,
        disposition: str,
        candidate_digest: str,
        dependency_digest: str,
        source_class: str = "FROZEN_PROTOCOL",
        authorizing_public_receipt_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = self._base(event_type="RECORDED_DISPOSITION", run_id=run_id, task_id=task_id)
        payload.update(
            {
                "subject_id": _require_pseudonymous_id(candidate_id, "candidate_id"),
                "recorded_disposition": disposition,
                "public_candidate_record_digest": candidate_digest,
                "public_dependency_set_digest": dependency_digest,
                "source_class": source_class,
            }
        )
        if authorizing_public_receipt_id is not None:
            payload["authorizing_public_receipt_id"] = authorizing_public_receipt_id
        return self.append(build_public_event(payload, self.signer))

    def append_correction(
        self,
        *,
        run_id: str,
        task_id: str,
        superseded_id: str,
        replacement_id: str,
        reason_code: str,
        effective_after_attempt: int,
        source_class: str = "FROZEN_EVALUATOR",
        authorizing_public_receipt_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = self._base(event_type="CORRECTION", run_id=run_id, task_id=task_id)
        payload.update(
            {
                "subject_id": superseded_id,
                "superseded_id": superseded_id,
                "replacement_id": replacement_id,
                "reason_code": reason_code,
                "effective_after_attempt": effective_after_attempt,
                "source_class": source_class,
            }
        )
        if authorizing_public_receipt_id is not None:
            payload["authorizing_public_receipt_id"] = authorizing_public_receipt_id
        return self.append(build_public_event(payload, self.signer))

    def append_retraction(
        self,
        *,
        run_id: str,
        task_id: str,
        subject_id: str,
        reason_code: str,
        effective_after_attempt: int,
        source_class: str = "FROZEN_EVALUATOR",
        authorizing_public_receipt_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = self._base(event_type="RETRACTION", run_id=run_id, task_id=task_id)
        payload.update(
            {
                "subject_id": subject_id,
                "reason_code": reason_code,
                "effective_after_attempt": effective_after_attempt,
                "source_class": source_class,
            }
        )
        if authorizing_public_receipt_id is not None:
            payload["authorizing_public_receipt_id"] = authorizing_public_receipt_id
        return self.append(build_public_event(payload, self.signer))

    def seal(
        self,
        *,
        run_id: str,
        task_id: str,
        receipt_head_digest: str,
        candidate_records: Iterable[Mapping[str, Any]],
        dependency_records: Iterable[Mapping[str, Any]],
        restore_receipt: Mapping[str, Any],
    ) -> Dict[str, Any]:
        candidates = [validate_public_candidate(record) for record in candidate_records]
        dependencies = [validate_public_dependency(record) for record in dependency_records]
        restore_digest = public_restore_receipt_digest(restore_receipt)
        payload = self._base(event_type="PUBLIC_CHAIN_SEAL", run_id=run_id, task_id=task_id)
        payload.update(
            {
                "public_receipt_head_digest": _require_digest(receipt_head_digest, "public_receipt_head_digest"),
                "public_event_preseal_head_digest": self.head,
                "public_candidate_collection_digest": public_candidate_collection_digest(candidates),
                "public_dependency_collection_digest": public_dependency_collection_digest(dependencies),
                "public_restore_receipt_digest": restore_digest,
            }
        )
        return self.append(build_public_event(payload, self.signer))


@dataclass(frozen=True)
class PublicReplayReport:
    """Result of a successful public cryptographic decision replay."""

    decisions: Dict[str, str]
    recorded_decisions: Dict[str, str]
    receipt_count: int
    event_count: int
    candidate_count: int
    dependency_count: int
    public_receipt_head_digest: str
    public_event_head_digest: str
    terminal_seal_verified: bool
    limitations: Tuple[str, ...] = (
        "does not rerun hidden tests",
        "does not prove a physical effect occurred",
        "does not reconstruct private content from digests",
    )

    @property
    def valid(self) -> bool:
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "replay_type": "public cryptographic decision replay",
            "decisions": dict(self.decisions),
            "recorded_decisions": dict(self.recorded_decisions),
            "receipt_count": self.receipt_count,
            "event_count": self.event_count,
            "candidate_count": self.candidate_count,
            "dependency_count": self.dependency_count,
            "public_receipt_head_digest": self.public_receipt_head_digest,
            "public_event_head_digest": self.public_event_head_digest,
            "terminal_seal_verified": self.terminal_seal_verified,
            "limitations": list(self.limitations),
        }


class PublicCryptographicVerifier:
    """Verify closed public records and derive dispositions from signed facts."""

    def __init__(
        self,
        public_key: Any,
        *,
        protocol_digest: str,
        evaluator_digest: Optional[str] = None,
        require_terminal_seal: bool = True,
    ) -> None:
        self.public_key = load_public_key(public_key)
        self.key_id = key_id_for_public_key(self.public_key)
        self.protocol_digest = _require_digest(protocol_digest, "protocol_digest")
        self.evaluator_digest = _require_digest(evaluator_digest, "evaluator_digest") if evaluator_digest else None
        self.require_terminal_seal = require_terminal_seal

    def _verify_receipt_chain(self, receipts: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
        normalized = [validate_public_receipt(receipt) for receipt in receipts]
        normalized.sort(key=lambda receipt: receipt["public_sequence"])
        previous = GENESIS_HASH
        seen_ids: Set[str] = set()
        for expected_sequence, receipt in enumerate(normalized, start=1):
            if receipt["public_sequence"] != expected_sequence:
                raise PublicReplayError("public receipt chain has a gap or duplicate sequence")
            if receipt["receipt_id"] in seen_ids:
                raise PublicReplayError("public receipt chain contains duplicate receipt ID")
            seen_ids.add(receipt["receipt_id"])
            if receipt["previous_public_receipt_digest"] != previous:
                raise PublicReplayError("public receipt chain previous digest mismatch")
            if receipt["protocol_digest"] != self.protocol_digest:
                raise PublicReplayError("public receipt uses an alternate protocol")
            if self.evaluator_digest and receipt["evaluator_digest"] != self.evaluator_digest:
                raise PublicReplayError("public receipt uses an alternate evaluator")
            previous = verify_public_receipt(receipt, self.public_key, expected_key_id=self.key_id)
        return normalized, previous

    def _verify_event_chain(self, events: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], str, Optional[Dict[str, Any]]]:
        normalized = [validate_public_event(event) for event in events]
        normalized.sort(key=lambda event: event["public_sequence"])
        previous = GENESIS_HASH
        seen_ids: Set[str] = set()
        seal: Optional[Dict[str, Any]] = None
        for expected_sequence, event in enumerate(normalized, start=1):
            if event["public_sequence"] != expected_sequence:
                raise PublicReplayError("public event chain has a gap or duplicate sequence")
            if event["event_id"] in seen_ids:
                raise PublicReplayError("public event chain contains duplicate event ID")
            seen_ids.add(event["event_id"])
            if event["previous_public_event_digest"] != previous:
                raise PublicReplayError("public event chain previous digest mismatch")
            if event["protocol_digest"] != self.protocol_digest:
                raise PublicReplayError("public event uses an alternate protocol")
            if self.evaluator_digest and event["evaluator_digest"] != self.evaluator_digest:
                raise PublicReplayError("public event uses an alternate evaluator")
            previous = verify_public_event(event, self.public_key, expected_key_id=self.key_id)
            if event["event_type"] == "PUBLIC_CHAIN_SEAL":
                if seal is not None:
                    raise PublicReplayError("public event chain contains multiple terminal seals")
                seal = event
                if expected_sequence != len(normalized):
                    raise PublicReplayError("public chain seal is not terminal")
        if self.require_terminal_seal and seal is None:
            raise PublicReplayError("public replay requires one terminal PUBLIC_CHAIN_SEAL")
        if seal is not None and normalized[-1]["event_type"] != "PUBLIC_CHAIN_SEAL":
            raise PublicReplayError("public chain has records after its terminal seal")
        return normalized, previous, seal

    @staticmethod
    def _candidate_stale(
        candidate_id: str,
        candidate: Mapping[str, Any],
        dependencies: Sequence[Mapping[str, Any]],
        invalid_roots: Set[str],
    ) -> bool:
        # Staleness flows from an invalid premise to its descendants. Walk
        # reverse edges from the candidate to its ancestors; walking forward
        # would incorrectly stale a parent or an unrelated sibling when a
        # child is retracted.
        reverse_adjacency: Dict[str, List[str]] = {}
        for dependency in dependencies:
            reverse_adjacency.setdefault(dependency["child_id"], []).append(dependency["parent_id"])
        stack = [candidate_id]
        visited: Set[str] = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            if node in invalid_roots:
                return True
            stack.extend(reverse_adjacency.get(node, []))
        return False

    @staticmethod
    def _validate_dependency_graph(
        dependencies: Sequence[Mapping[str, Any]],
        known_endpoint_ids: Set[str],
    ) -> None:
        adjacency: Dict[str, List[str]] = {}
        for dependency in dependencies:
            parent_id = dependency["parent_id"]
            child_id = dependency["child_id"]
            if parent_id not in known_endpoint_ids or child_id not in known_endpoint_ids:
                raise PublicReplayError("public dependency references an unknown endpoint")
            adjacency.setdefault(parent_id, []).append(child_id)
        visiting: Set[str] = set()
        visited: Set[str] = set()

        def visit(node: str) -> None:
            if node in visiting:
                raise PublicReplayError("public dependency graph contains a cycle")
            if node in visited:
                return
            visiting.add(node)
            for child in adjacency.get(node, []):
                visit(child)
            visiting.remove(node)
            visited.add(node)

        for node in adjacency:
            visit(node)

    @staticmethod
    def _compute_dispositions(
        candidates: Sequence[Mapping[str, Any]],
        dependencies: Sequence[Mapping[str, Any]],
        receipts: Sequence[Mapping[str, Any]],
        invalid_roots: Set[str],
    ) -> Dict[str, str]:
        receipts_by_candidate: Dict[str, List[Mapping[str, Any]]] = {}
        for receipt in receipts:
            receipts_by_candidate.setdefault(receipt["candidate_id"], []).append(receipt)
        decisions: Dict[str, str] = {}
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            if PublicCryptographicVerifier._candidate_stale(candidate_id, candidate, dependencies, invalid_roots):
                decisions[candidate_id] = "STALE_DEPENDENT"
                continue
            facts = receipts_by_candidate.get(candidate_id, [])
            if any(receipt["receipt_id"] in invalid_roots for receipt in facts):
                decisions[candidate_id] = "STALE_DEPENDENT"
                continue
            verdicts = [receipt for receipt in facts if receipt["receipt_type"] == "VERDICT"]
            authorities = [receipt for receipt in facts if receipt["receipt_type"] == "AUTHORITY"]
            effects = [receipt for receipt in facts if receipt["receipt_type"] == "EFFECT"]
            if any(receipt["decision"] == "DENY" for receipt in authorities + effects):
                decisions[candidate_id] = "REJECTED"
                continue
            if not verdicts:
                decisions[candidate_id] = "ABSTAINED"
                continue
            if any(receipt["decision"] == "ERROR" for receipt in facts):
                decisions[candidate_id] = "ABSTAINED"
                continue
            if any(receipt["decision"] == "FAIL" for receipt in verdicts):
                decisions[candidate_id] = "REJECTED"
                continue
            requested_authority = candidate["requested_authority"]
            if requested_authority not in {"NONE", "READ", "READ_ONLY"} and not authorities:
                decisions[candidate_id] = "ABSTAINED"
                continue
            requires_effect = requested_authority not in {"NONE", "READ", "READ_ONLY"}
            if requires_effect and not effects:
                decisions[candidate_id] = "ABSTAINED"
                continue
            if any(receipt["decision"] == "ERROR" for receipt in effects):
                decisions[candidate_id] = "ABSTAINED"
                continue
            if any(receipt["decision"] != "PASS" for receipt in verdicts):
                decisions[candidate_id] = "ABSTAINED"
                continue
            decisions[candidate_id] = "PROMOTED"
        return decisions

    def verify(
        self,
        *,
        candidates: Iterable[Mapping[str, Any]],
        dependencies: Iterable[Mapping[str, Any]],
        receipts: Iterable[Mapping[str, Any]],
        events: Iterable[Mapping[str, Any]],
        restore_receipt: Optional[Mapping[str, Any]] = None,
    ) -> PublicReplayReport:
        candidate_records = [validate_public_candidate(record) for record in candidates]
        dependency_records = [validate_public_dependency(record) for record in dependencies]
        event_input = list(events)
        candidate_records.sort(key=lambda record: record["candidate_id"])
        dependency_records.sort(key=public_dependency_id)
        candidates_by_id = {record["candidate_id"]: record for record in candidate_records}
        if len(candidates_by_id) != len(candidate_records):
            raise PublicReplayError("duplicate public candidate ID")
        dependency_ids = {public_dependency_id(record) for record in dependency_records}
        if len(dependency_ids) != len(dependency_records):
            raise PublicReplayError("duplicate public dependency record")
        campaign_ids = {record["campaign_id"] for record in candidate_records}
        if len(campaign_ids) > 1:
            raise PublicReplayError("public candidate projection mixes campaigns")
        candidate_ids = set(candidates_by_id)
        for candidate in candidate_records:
            parent_id = candidate["parent_candidate_id"]
            if parent_id is not None and parent_id not in candidate_ids:
                raise PublicReplayError("public candidate references an unknown parent candidate")
            if parent_id == candidate["candidate_id"]:
                raise PublicReplayError("public candidate cannot be its own parent")
            if len(candidate["public_dependency_ids"]) != len(set(candidate["public_dependency_ids"])):
                raise PublicReplayError("candidate carries duplicate public dependency IDs")
            unknown = set(candidate["public_dependency_ids"]) - dependency_ids
            if unknown:
                raise PublicReplayError(f"candidate references unknown dependency IDs: {sorted(unknown)}")
            expected_set_digest = public_dependency_set_digest(
                record for record in dependency_records if public_dependency_id(record) in set(candidate["public_dependency_ids"])
            )
            # The candidate itself has no set digest; the receipt binds it below.
            _ = expected_set_digest
        receipt_records, receipt_head = self._verify_receipt_chain(list(receipts))
        receipt_ids = {receipt["receipt_id"] for receipt in receipt_records}
        lifecycle_event_ids = {validate_public_event(event)["event_id"] for event in event_input}
        receipt_ids_for_dependencies = {receipt["receipt_id"] for receipt in receipt_records}
        if any(dependency["insertion_receipt_id"] not in receipt_ids_for_dependencies for dependency in dependency_records):
            raise PublicReplayError("public dependency references an unknown insertion receipt")
        self._validate_dependency_graph(
            dependency_records,
            candidate_ids | receipt_ids | lifecycle_event_ids,
        )
        for receipt in receipt_records:
            if campaign_ids and receipt["campaign_id"] not in campaign_ids:
                raise PublicReplayError("public receipt campaign does not match candidate projection")
            candidate = candidates_by_id.get(receipt["candidate_id"])
            if candidate is None:
                raise PublicReplayError("public receipt references an unknown candidate")
            if receipt["run_id"] != candidate["run_id"] or receipt["task_id"] != candidate["task_id"]:
                raise PublicReplayError("public receipt run/task binding mismatch")
            if receipt["candidate_artifact_digest"] != candidate["candidate_artifact_digest"]:
                raise PublicReplayError("public receipt candidate artifact binding mismatch")
            if receipt["public_candidate_record_digest"] != public_candidate_digest(candidate):
                raise PublicReplayError("public receipt candidate digest binding mismatch")
            candidate_dependencies = [
                record
                for record in dependency_records
                if public_dependency_id(record) in set(candidate["public_dependency_ids"])
            ]
            if receipt["public_dependency_set_digest"] != public_dependency_set_digest(candidate_dependencies):
                raise PublicReplayError("public receipt dependency-set digest binding mismatch")
        event_records, event_head, seal = self._verify_event_chain(event_input)
        if campaign_ids and any(event["campaign_id"] not in campaign_ids for event in event_records):
            raise PublicReplayError("public event campaign does not match candidate projection")
        receipts_by_id = {receipt["receipt_id"]: receipt for receipt in receipt_records}
        known_ids = set(candidates_by_id) | dependency_ids | set(receipts_by_id)
        invalid_roots: Set[str] = set()
        recorded: Dict[str, Mapping[str, Any]] = {}
        last_boundary: Dict[Tuple[str, str], int] = {}
        for event in event_records:
            event_type = event["event_type"]
            if event_type == "PUBLIC_CHAIN_SEAL":
                continue
            if event["source_class"] == "FROZEN_EVALUATOR":
                authorizing_id = event["authorizing_public_receipt_id"]
                authorizing_receipt = receipts_by_id.get(authorizing_id)
                if authorizing_receipt is None:
                    raise PublicReplayError("evaluator-observed event lacks an existing authorizing receipt")
                if (
                    authorizing_receipt["campaign_id"] != event["campaign_id"]
                    or authorizing_receipt["run_id"] != event["run_id"]
                    or authorizing_receipt["task_id"] != event["task_id"]
                ):
                    raise PublicReplayError("lifecycle authorizing receipt scope mismatch")
            if event["event_type"] in {"CORRECTION", "RETRACTION"}:
                boundary_key = (event["run_id"], event["task_id"])
                boundary = event["effective_after_attempt"]
                if boundary < last_boundary.get(boundary_key, 0):
                    raise PublicReplayError("public lifecycle boundary is not monotonic")
                last_boundary[boundary_key] = boundary
            if event_type == "CORRECTION":
                if event["superseded_id"] not in known_ids or event["replacement_id"] not in known_ids:
                    raise PublicReplayError("correction references an unknown public ID")
                invalid_roots.add(event["superseded_id"])
            elif event_type == "RETRACTION":
                if event["subject_id"] not in known_ids:
                    raise PublicReplayError("retraction references an unknown public ID")
                invalid_roots.add(event["subject_id"])
            elif event_type == "RECORDED_DISPOSITION":
                candidate = candidates_by_id.get(event["subject_id"])
                if candidate is None:
                    raise PublicReplayError("recorded disposition references an unknown candidate")
                if event["public_candidate_record_digest"] != public_candidate_digest(candidate):
                    raise PublicReplayError("recorded disposition candidate digest mismatch")
                candidate_dependencies = [
                    record
                    for record in dependency_records
                    if public_dependency_id(record) in set(candidate["public_dependency_ids"])
                ]
                if event["public_dependency_set_digest"] != public_dependency_set_digest(candidate_dependencies):
                    raise PublicReplayError("recorded disposition dependency digest mismatch")
                if event["subject_id"] in recorded:
                    raise PublicReplayError("candidate has multiple recorded dispositions")
                recorded[event["subject_id"]] = event
            known_ids.add(event["event_id"])
        # This is the decisive step: recorded_disposition is deliberately not
        # consulted while computing the disposition from signed facts.
        computed = self._compute_dispositions(candidate_records, dependency_records, receipt_records, invalid_roots)
        if set(recorded) != set(computed):
            raise PublicReplayError("public projection is missing a recorded disposition")
        recorded_decisions = {candidate_id: event["recorded_disposition"] for candidate_id, event in recorded.items()}
        mismatches = {
            candidate_id: (computed[candidate_id], recorded_decisions[candidate_id])
            for candidate_id in computed
            if computed[candidate_id] != recorded_decisions[candidate_id]
        }
        if mismatches:
            raise PublicReplayError(f"recorded dispositions disagree with cryptographic replay: {mismatches}")
        if seal is not None:
            if restore_receipt is None:
                raise PublicReplayError("terminal public chain seal requires the strict public restore receipt")
            restore_digest = verify_public_restore_receipt(restore_receipt, self.public_key, expected_key_id=self.key_id)
            if campaign_ids and restore_receipt.get("campaign_id") not in campaign_ids:
                raise PublicReplayError("restore receipt campaign does not match public projection")
            if seal["public_receipt_head_digest"] != receipt_head:
                raise PublicReplayError("terminal seal receipt head mismatch")
            preseal_events = event_records[:-1]
            preseal_head = public_event_digest(preseal_events[-1]) if preseal_events else GENESIS_HASH
            if seal["public_event_preseal_head_digest"] != preseal_head:
                raise PublicReplayError("terminal seal preseal head mismatch")
            if seal["public_candidate_collection_digest"] != public_candidate_collection_digest(candidate_records):
                raise PublicReplayError("terminal seal candidate collection mismatch")
            if seal["public_dependency_collection_digest"] != public_dependency_collection_digest(dependency_records):
                raise PublicReplayError("terminal seal dependency collection mismatch")
            if seal["public_restore_receipt_digest"] != restore_digest:
                raise PublicReplayError("terminal seal restore receipt mismatch")
        return PublicReplayReport(
            decisions=computed,
            recorded_decisions=recorded_decisions,
            receipt_count=len(receipt_records),
            event_count=len(event_records),
            candidate_count=len(candidate_records),
            dependency_count=len(dependency_records),
            public_receipt_head_digest=receipt_head,
            public_event_head_digest=event_head,
            terminal_seal_verified=seal is not None,
        )


class PublicProjection:
    """Deterministic file-backed bundle for closed public records."""

    def __init__(self) -> None:
        self.candidates: Dict[str, Dict[str, Any]] = {}
        self.dependencies: Dict[str, Dict[str, Any]] = {}
        self.receipts: Dict[str, Dict[str, Any]] = {}
        self.events: List[Dict[str, Any]] = []
        self.restore_receipt: Optional[Dict[str, Any]] = None
        self.public_key_pem: Optional[bytes] = None

    def add_candidate(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        normalized = validate_public_candidate(record)
        candidate_id = normalized["candidate_id"]
        if candidate_id in self.candidates and canonical_json(self.candidates[candidate_id]) != canonical_json(normalized):
            raise PublicSchemaError("conflicting public candidate record")
        self.candidates[candidate_id] = normalized
        return dict(normalized)

    def add_dependency(self, record: Mapping[str, Any]) -> str:
        normalized = validate_public_dependency(record)
        dependency_id = public_dependency_id(normalized)
        if dependency_id in self.dependencies and canonical_json(self.dependencies[dependency_id]) != canonical_json(normalized):
            raise PublicSchemaError("conflicting public dependency record")
        self.dependencies[dependency_id] = normalized
        return dependency_id

    def add_receipt(self, receipt: Mapping[str, Any]) -> str:
        normalized = validate_public_receipt(receipt)
        receipt_id = normalized["receipt_id"]
        if receipt_id in self.receipts and canonical_json(self.receipts[receipt_id]) != canonical_json(normalized):
            raise PublicSchemaError("conflicting public receipt envelope")
        self.receipts[receipt_id] = normalized
        return receipt_id

    def add_event(self, event: Mapping[str, Any]) -> Dict[str, Any]:
        normalized = validate_public_event(event)
        if self.events and normalized["public_sequence"] != self.events[-1]["public_sequence"] + 1:
            raise PublicSchemaError("public event sequence is not contiguous")
        if not self.events and normalized["public_sequence"] != 1:
            raise PublicSchemaError("public event sequence must begin at one")
        if (self.events and normalized["previous_public_event_digest"] != public_event_digest(self.events[-1])) or (
            not self.events and normalized["previous_public_event_digest"] != GENESIS_HASH
        ):
            raise PublicSchemaError("public event chain head mismatch")
        self.events.append(normalized)
        return dict(normalized)

    def set_restore_receipt(self, receipt: Mapping[str, Any]) -> Dict[str, Any]:
        self.restore_receipt = validate_public_restore_receipt(receipt)
        return dict(self.restore_receipt)

    def set_public_key(self, public_key_pem: bytes) -> None:
        if not isinstance(public_key_pem, bytes) or not public_key_pem:
            raise PublicSchemaError("public key export must be non-empty PEM bytes")
        self.public_key_pem = public_key_pem

    def export(self, directory: Union[str, Path], *, source_ledger: Any = None) -> Path:
        root = Path(directory)
        if source_ledger is not None:
            source_ledger.export_public_blobs(root)
        write_canonical_jsonl(root / "ledger" / "public-candidates.jsonl", [self.candidates[key] for key in sorted(self.candidates)])
        write_canonical_jsonl(
            root / "ledger" / "public-dependencies.jsonl",
            [self.dependencies[key] for key in sorted(self.dependencies)],
        )
        write_canonical_jsonl(root / "receipts" / "public-signed-envelopes.jsonl", [self.receipts[key] for key in sorted(self.receipts, key=lambda key: self.receipts[key]["public_sequence"])])
        write_canonical_jsonl(root / "ledger" / "public-events.jsonl", self.events)
        if self.public_key_pem is not None:
            key_path = root / "receipts" / "evaluator-public-key.pem"
            key_path.parent.mkdir(parents=True, exist_ok=True)
            key_path.write_bytes(self.public_key_pem)
        if self.restore_receipt is not None:
            restore_path = root / "restore" / "public-restore-receipt.json"
            restore_path.parent.mkdir(parents=True, exist_ok=True)
            restore_path.write_text(canonical_json(self.restore_receipt) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": "egv-public-projection-manifest-v1",
            "candidate_count": len(self.candidates),
            "dependency_count": len(self.dependencies),
            "receipt_count": len(self.receipts),
            "event_count": len(self.events),
            "candidate_collection_digest": public_candidate_collection_digest(self.candidates.values()),
            "dependency_collection_digest": public_dependency_collection_digest(self.dependencies.values()),
            "receipt_head_digest": receipt_hash(next(reversed(sorted(self.receipts.values(), key=lambda item: item["public_sequence"])))) if self.receipts else GENESIS_HASH,
            "event_head_digest": public_event_digest(self.events[-1]) if self.events else GENESIS_HASH,
            "restore_receipt_digest": public_restore_receipt_digest(self.restore_receipt) if self.restore_receipt is not None else None,
        }
        manifest_path = root / "projection-manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
        return root


def load_public_projection(directory: Union[str, Path]) -> PublicProjection:
    root = Path(directory)
    projection = PublicProjection()
    candidate_path = root / "ledger" / "public-candidates.jsonl"
    dependency_path = root / "ledger" / "public-dependencies.jsonl"
    receipt_path = root / "receipts" / "public-signed-envelopes.jsonl"
    event_path = root / "ledger" / "public-events.jsonl"
    key_path = root / "receipts" / "evaluator-public-key.pem"
    if key_path.exists():
        projection.set_public_key(key_path.read_bytes())
    for record in parse_canonical_jsonl(candidate_path.read_text(encoding="utf-8")):
        projection.add_candidate(record)
    for record in parse_canonical_jsonl(dependency_path.read_text(encoding="utf-8")):
        projection.add_dependency(record)
    for record in parse_canonical_jsonl(receipt_path.read_text(encoding="utf-8")):
        projection.add_receipt(record)
    for record in parse_canonical_jsonl(event_path.read_text(encoding="utf-8")):
        projection.add_event(record)
    restore_path = root / "restore" / "public-restore-receipt.json"
    if restore_path.exists():
        restore_records = parse_canonical_jsonl(restore_path.read_text(encoding="utf-8"))
        if len(restore_records) != 1:
            raise PublicSchemaError("public restore receipt file must contain exactly one canonical record")
        projection.set_restore_receipt(restore_records[0])
    return projection


def verify_public_projection(
    projection: Union[PublicProjection, str, Path],
    public_key: Any,
    *,
    protocol_digest: str,
    evaluator_digest: Optional[str] = None,
    require_terminal_seal: bool = True,
) -> PublicReplayReport:
    """Verify an in-memory or exported closed public projection."""

    loaded = load_public_projection(projection) if isinstance(projection, (str, Path)) else projection
    verifier = PublicCryptographicVerifier(
        public_key,
        protocol_digest=protocol_digest,
        evaluator_digest=evaluator_digest,
        require_terminal_seal=require_terminal_seal,
    )
    return verifier.verify(
        candidates=loaded.candidates.values(),
        dependencies=loaded.dependencies.values(),
        receipts=loaded.receipts.values(),
        events=loaded.events,
        restore_receipt=loaded.restore_receipt,
    )


PublicVerifier = PublicCryptographicVerifier


__all__ = [
    "PUBLIC_CANDIDATE_SCHEMA_VERSION",
    "PUBLIC_DEPENDENCY_SCHEMA_VERSION",
    "PUBLIC_EVENT_SCHEMA_VERSION",
    "PUBLIC_RESTORE_SCHEMA_VERSION",
    "PublicCryptographicVerifier",
    "PublicEventChain",
    "PublicProjection",
    "PublicReplayReport",
    "build_public_event",
    "build_public_receipt_envelope",
    "build_public_restore_receipt",
    "load_public_projection",
    "public_candidate_collection_digest",
    "public_candidate_digest",
    "public_dependency_collection_digest",
    "public_dependency_id",
    "public_dependency_set_digest",
    "public_event_digest",
    "public_receipt_digest",
    "public_restore_receipt_digest",
    "validate_public_candidate",
    "validate_public_dependency",
    "validate_public_event",
    "validate_public_receipt",
    "validate_public_restore_receipt",
    "verify_public_projection",
    "verify_public_event",
    "verify_public_receipt",
    "verify_public_restore_receipt",
    "PublicVerifier",
]
