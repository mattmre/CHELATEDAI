"""Closed, aggressively redacted public projection for EGV Training."""

from __future__ import annotations

import base64
import binascii
import html
import json
import math
import re
from typing import Any, Dict, List, Mapping, Tuple
from urllib.parse import unquote

from ..canonical import canonical_json


PUBLIC_TRAINING_PREFLIGHT_SCHEMA = "egv-training-public-preflight-v1"
PUBLIC_TRAINING_COMPLETED_SCHEMA = "egv-training-public-completed-v1"
PUBLIC_TRAINING_SCHEMA = PUBLIC_TRAINING_COMPLETED_SCHEMA
_COMMON_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "protocol_digest",
        "model_manifest_digest",
        "data_manifest_digest",
        "target_manifest_digest",
        "software_manifest_digest",
        "aggregate_metrics",
        "limitations",
    }
)
PUBLIC_TRAINING_PREFLIGHT_FIELDS = _COMMON_FIELDS
PUBLIC_TRAINING_COMPLETED_FIELDS = frozenset(
    {
        *_COMMON_FIELDS,
        "training_dataset_digest",
        "adapter_manifest_digest",
        "base_immutability_proof_digest",
        "development_receipt_digest",
        "development_receipt_binding_digest",
    }
)
PUBLIC_TRAINING_FIELDS = PUBLIC_TRAINING_COMPLETED_FIELDS
AGGREGATE_FIELDS = frozenset(
    {
        "training_row_count",
        "training_task_count",
        "development_task_count",
        "epoch_count",
        "optimizer_step_count",
        "base_tensor_count",
        "trainable_adapter_parameter_count",
        "aggregate_training_loss",
        "aggregate_development_loss",
    }
)
PUBLIC_STATUSES = frozenset({"READY", "COMPLETED", "FAILED", "ABSTAINED"})
PUBLIC_LIMITATIONS = frozenset(
    {
        "RESEARCH_ONLY",
        "SMALL_MODEL_PILOT",
        "NO_PRODUCTION_CLAIM",
        "NO_GENERAL_UTILITY_CLAIM",
        "EVALUATION_RESULTS_EXCLUDED",
        "HARDWARE_SCOPED",
    }
)
_NON_DIGEST_FIELDS = {"schema_version", "status", "aggregate_metrics", "limitations"}
_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_BASE64ISH = re.compile(r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/_-]{12,}={0,2}(?![A-Za-z0-9+/=_-])")

# Patterns apply to both field names and recursively decoded string values.
_FORBIDDEN = (
    re.compile(r"(?i)raw[_ -]?prompt|\bprompt\b|training[_ -]?target|target[_ -]?text|\btarget\b|candidate[_ -]?source|source[_ -]?(?:code|text|file)|\bsource\b|dataset[_ -]?(?:example|sample|row)"),
    re.compile(r"(?i)password|passphrase|api[_ -]?key|access[_ -]?token|refresh[_ -]?token|auth[_ -]?token|\btoken\b|bearer\s+|client[_ -]?secret|private[_ -]?key|BEGIN\s+(?:RSA|OPENSSH|EC|ED25519)"),
    re.compile(r"(?i)username|user[_ -]?name|\buser\b|hostname|host[_ -]?name|\bhost\b|machine[_ -]?(?:name|id)|node[_ -]?(?:name|id)|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    re.compile(r"(?i)(?:[A-Z]:[\\/]|/(?:home|root|Users|tmp|var|etc|proc|run|mnt)/|\\\\[A-Za-z0-9_.-]+\\|file://|ssh://|https?://)"),
    re.compile(r"(?i)(?:localhost|\.local\b|(?:dgx|gx\d+)[-_A-Za-z0-9]*|spark[_ -]?(?:trainer|evaluator))"),
    re.compile(r"(?<!\d)(?:10\.(?:\d{1,3}\.){2}\d{1,3}|192\.168\.(?:\d{1,3}\.)\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.(?:\d{1,3}\.)\d{1,3}|127\.(?:\d{1,3}\.){2}\d{1,3}|169\.254\.(?:\d{1,3}\.)\d{1,3})(?!\d)"),
    re.compile(r"(?i)(?:\[?(?:fc|fd|fe80)[0-9a-f:]+\]?|(?:^|\s)(?:\d{1,3}\.){3}\d{1,3}:\d{2,5}\b|\bport\b(?:\s*[:=]\s*\d{1,5})?)"),
    re.compile(r"(?i)timestamp|started[_ -]?at|ended[_ -]?at|wall[_ -]?clock|duration|latency|throughput|tokens?[_ -]?per[_ -]?second|gpu[_ -]?(?:util|memory|temperature)|vram|cpu[_ -]?util|process[_ -]?id|\bpid\b"),
    re.compile(r"(?i)checkpoint[_ -]?(?:path|state)|optimizer[_ -]?state|scheduler[_ -]?state|rng[_ -]?state|random[_ -]?state|data[_ -]?cursor"),
    re.compile(r"(?i)per[_ -]?task|task[_ -]?(?:id|identity|loss)|development[_ -]?(?:identity|example|row|task[_ -]?(?:id|loss))|dev[_ -]?(?:identity|example|row|task[_ -]?(?:id|loss))"),
    re.compile(r"(?i)hidden[_ -]?(?:test|rule|path|output|material)|held[_ -]?out|heldout|golden[_ -]?patch|expected[_ -]?output|evaluator[_ -]?private"),
)


class PublicTrainingSafetyError(ValueError):
    """The proposed public Training artifact is unsafe or outside schema."""


def _decoded_views(value: str) -> Tuple[str, ...]:
    """Expose common low-effort encodings without treating digests as content."""

    pending = [value]
    views: List[str] = []
    seen = set()
    while pending and len(views) < 24:
        current = pending.pop(0)
        if current in seen:
            continue
        seen.add(current)
        views.append(current)
        for decoded in (unquote(current), html.unescape(current)):
            if decoded != current:
                pending.append(decoded)
        for match in _BASE64ISH.findall(current):
            if _HEX_DIGEST.fullmatch(match.lower()):
                continue
            padded = match.replace("-", "+").replace("_", "/")
            padded += "=" * ((-len(padded)) % 4)
            try:
                raw = base64.b64decode(padded, validate=True)
                decoded = raw.decode("utf-8")
            except (binascii.Error, UnicodeDecodeError, ValueError):
                continue
            if decoded and sum(character.isprintable() for character in decoded) / len(decoded) >= 0.9:
                pending.append(decoded)
    return tuple(views)


def _scan(payload: Any) -> List[str]:
    findings: List[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                key_text = str(key)
                for view in _decoded_views(key_text):
                    if any(pattern.search(view) for pattern in _FORBIDDEN):
                        findings.append("forbidden field at {}/{}".format(path, key_text[:48]))
                        break
                visit(item, "{}/{}".format(path, key_text[:48]))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, "{}/{}".format(path, index))
        elif isinstance(value, str):
            if _HEX_DIGEST.fullmatch(value):
                return
            for view in _decoded_views(value):
                if any(pattern.search(view) for pattern in _FORBIDDEN):
                    findings.append("forbidden value at {}".format(path))
                    break

    visit(payload, "$")
    return sorted(set(findings))


def _require_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _HEX_DIGEST.fullmatch(value):
        raise PublicTrainingSafetyError("{} must be a lowercase SHA-256 digest".format(field))
    return value


def validate_public_training_summary(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate, detach, and normalize the one allowed public Training shape."""

    findings = _scan(payload)
    if findings:
        raise PublicTrainingSafetyError("; ".join(findings))
    if not isinstance(payload, Mapping):
        raise PublicTrainingSafetyError("public Training summary must be an object")
    schema = payload.get("schema_version")
    if schema == PUBLIC_TRAINING_PREFLIGHT_SCHEMA:
        expected_fields = PUBLIC_TRAINING_PREFLIGHT_FIELDS
        allowed_statuses = frozenset({"READY", "FAILED", "ABSTAINED"})
    elif schema == PUBLIC_TRAINING_COMPLETED_SCHEMA:
        expected_fields = PUBLIC_TRAINING_COMPLETED_FIELDS
        allowed_statuses = frozenset({"COMPLETED"})
    else:
        raise PublicTrainingSafetyError("public Training summary schema is unsupported")
    if set(payload) != expected_fields:
        raise PublicTrainingSafetyError("public Training summary has an unexpected field set")
    status = payload.get("status")
    if status not in allowed_statuses:
        raise PublicTrainingSafetyError("public Training status does not match its state schema")
    digests = {
        field: _require_digest(payload[field], field)
        for field in sorted(expected_fields - _NON_DIGEST_FIELDS)
    }
    metrics = payload.get("aggregate_metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != AGGREGATE_FIELDS:
        raise PublicTrainingSafetyError("aggregate metrics have an unexpected field set")
    normalized_metrics: Dict[str, Any] = {}
    for field in sorted(AGGREGATE_FIELDS):
        value = metrics[field]
        if field.startswith("aggregate_") and field.endswith("_loss"):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                raise PublicTrainingSafetyError("{} must be a finite nonnegative aggregate".format(field))
            normalized = round(float(value), 6)
            if float(value) != normalized:
                raise PublicTrainingSafetyError("aggregate loss must be rounded to at most six decimal places")
            normalized_metrics[field] = normalized
        else:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise PublicTrainingSafetyError("{} must be a nonnegative aggregate count".format(field))
            normalized_metrics[field] = value
    if schema == PUBLIC_TRAINING_COMPLETED_SCHEMA:
        if normalized_metrics["training_task_count"] != 20:
            raise PublicTrainingSafetyError("COMPLETED requires exactly 20 training tasks")
        if normalized_metrics["development_task_count"] != 8:
            raise PublicTrainingSafetyError("COMPLETED requires exactly 8 development tasks")
        if not 1 <= normalized_metrics["epoch_count"] <= 3:
            raise PublicTrainingSafetyError("COMPLETED requires one to three epochs")
        for field in (
            "training_row_count",
            "optimizer_step_count",
            "base_tensor_count",
            "trainable_adapter_parameter_count",
        ):
            if normalized_metrics[field] <= 0:
                raise PublicTrainingSafetyError("COMPLETED requires nonzero {}".format(field))
        if digests["development_receipt_digest"] == digests["development_receipt_binding_digest"]:
            raise PublicTrainingSafetyError("COMPLETED development receipt and binding digests must be distinct")
    limitations = payload.get("limitations")
    if not isinstance(limitations, list) or limitations != sorted(set(limitations)):
        raise PublicTrainingSafetyError("public limitations must be a sorted unique JSON array")
    if not limitations or any(item not in PUBLIC_LIMITATIONS for item in limitations):
        raise PublicTrainingSafetyError("public limitations are outside the closed enum")
    result = {
        "schema_version": schema,
        "status": status,
        **digests,
        "aggregate_metrics": normalized_metrics,
        "limitations": list(limitations),
    }
    # Reorder through the canonical contract, independent of caller mapping order.
    return json.loads(canonical_json(result))


def scan_public_training_summary(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized = validate_public_training_summary(payload)
    return {
        "scanner": normalized["schema_version"],
        "checked": True,
        "field_count": len(normalized),
        "findings": [],
    }


def public_training_summary_json(payload: Mapping[str, Any]) -> str:
    return canonical_json(validate_public_training_summary(payload)) + "\n"


def load_public_training_summary_json(value: str) -> Dict[str, Any]:
    try:
        payload = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise PublicTrainingSafetyError("public Training summary is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise PublicTrainingSafetyError("public Training summary JSON must contain an object")
    return validate_public_training_summary(payload)


__all__ = [
    "AGGREGATE_FIELDS",
    "PUBLIC_LIMITATIONS",
    "PUBLIC_STATUSES",
    "PUBLIC_TRAINING_FIELDS",
    "PUBLIC_TRAINING_SCHEMA",
    "PUBLIC_TRAINING_PREFLIGHT_FIELDS",
    "PUBLIC_TRAINING_PREFLIGHT_SCHEMA",
    "PUBLIC_TRAINING_COMPLETED_FIELDS",
    "PUBLIC_TRAINING_COMPLETED_SCHEMA",
    "PublicTrainingSafetyError",
    "load_public_training_summary_json",
    "public_training_summary_json",
    "scan_public_training_summary",
    "validate_public_training_summary",
]
