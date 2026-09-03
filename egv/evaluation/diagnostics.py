"""Closed evaluator diagnostics and failure-family identities.

The evaluator exposes only this bounded vocabulary.  In particular,
``INTERNAL_ERROR`` means that the evaluator infrastructure could not produce a
model result; it is never converted into a model failure or collapsed with a
different infrastructure incident.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from ..canonical import failure_family_root


class Diagnostic(str, Enum):
    PASS = "PASS"
    WRONG_OUTPUT = "WRONG_OUTPUT"
    SYNTAX_OR_IMPORT = "SYNTAX_OR_IMPORT"
    RUNTIME_EXCEPTION = "RUNTIME_EXCEPTION"
    TIMEOUT = "TIMEOUT"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    AUTHORITY_DENIED = "AUTHORITY_DENIED"
    MUTATION_LOCUS_VIOLATION = "MUTATION_LOCUS_VIOLATION"
    PROTOCOL_VIOLATION = "PROTOCOL_VIOLATION"
    INTERNAL_ERROR = "INTERNAL_ERROR"


DIAGNOSTIC_ENUM = tuple(item.value for item in Diagnostic)
RESOURCE_BUCKETS = (
    "UNDER_25",
    "25_TO_50",
    "50_TO_75",
    "75_TO_100",
    "LIMIT_REACHED",
    "OUTPUT_LIMIT",
)
PROMOTION_DISPOSITIONS = ("PROMOTED", "REJECTED", "ABSTAINED", "STALE_DEPENDENT")
REQUESTED_AUTHORITIES = ("NONE", "READ_ONLY", "EXECUTE_CANDIDATE")


def validate_diagnostic(value: object) -> str:
    """Return a closed diagnostic value or fail closed."""

    if isinstance(value, Diagnostic):
        return value.value
    if not isinstance(value, str) or value not in DIAGNOSTIC_ENUM:
        raise ValueError("diagnostic_enum must be one of the frozen EGV values")
    return value


def validate_resource_bucket(value: object) -> str:
    if not isinstance(value, str) or value not in RESOURCE_BUCKETS:
        raise ValueError("resource_bucket is outside the frozen bounded vocabulary")
    return value


def validate_disposition(value: object) -> str:
    if not isinstance(value, str) or value not in PROMOTION_DISPOSITIONS:
        raise ValueError("promotion disposition is outside the frozen vocabulary")
    return value


def validate_requested_authority(value: object) -> str:
    if not isinstance(value, str) or value not in REQUESTED_AUTHORITIES:
        raise ValueError("requested authority is outside the frozen vocabulary")
    return value


def failure_family_for_attempt(
    task_family: str,
    diagnostic_enum: str,
    normalized_public_locus: str,
    public_rule_id: str,
    *,
    attempt_id: Optional[str] = None,
    infrastructure_incident_id: Optional[str] = None,
) -> str:
    """Derive the public failure root and bind internal loss to its incident."""

    diagnostic = validate_diagnostic(diagnostic_enum)
    if diagnostic == Diagnostic.INTERNAL_ERROR.value:
        if not infrastructure_incident_id:
            raise ValueError("INTERNAL_ERROR requires a unique infrastructure incident ID")
        return failure_family_root(
            task_family,
            diagnostic,
            normalized_public_locus,
            public_rule_id,
            infrastructure_incident_id=infrastructure_incident_id,
        )
    if infrastructure_incident_id is not None:
        raise ValueError("non-INTERNAL_ERROR roots cannot include an infrastructure incident ID")
    return failure_family_root(task_family, diagnostic, normalized_public_locus, public_rule_id)


__all__ = [
    "DIAGNOSTIC_ENUM",
    "Diagnostic",
    "PROMOTION_DISPOSITIONS",
    "REQUESTED_AUTHORITIES",
    "RESOURCE_BUCKETS",
    "failure_family_for_attempt",
    "validate_diagnostic",
    "validate_disposition",
    "validate_requested_authority",
    "validate_resource_bucket",
]
