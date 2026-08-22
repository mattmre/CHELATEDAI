"""Canonical JSON and content-addressing primitives used by every EGV layer.

The evidence contract uses one serialization rule for IDs, hashes, JSONL, hash
chains, and public record bindings.  The implementation deliberately rejects
implicit stringification, unordered sets, NaN, and infinity: silently making
those values JSON-safe would create identities that cannot be independently
reproduced.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

from .errors import CanonicalizationError


CANONICAL_JSON_SEPARATORS = (",", ":")
GENESIS_HASH = "0" * 64


def _normalize(value: Any, path: str = "$", *, allow_dataclass: bool = True) -> Any:
    """Return a JSON-compatible value without introducing nondeterminism."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError(f"non-finite float at {path}")
        return value
    if isinstance(value, Path):
        raise CanonicalizationError(f"filesystem path is not canonical JSON at {path}")
    if allow_dataclass and is_dataclass(value) and not isinstance(value, type):
        return _normalize(asdict(value), path, allow_dataclass=False)
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError(f"object key at {path} is not a string: {key!r}")
            normalized[key] = _normalize(item, f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        raise CanonicalizationError(f"unordered set at {path} cannot be canonicalized")
    raise CanonicalizationError(
        f"unsupported value at {path}: {type(value).__module__}.{type(value).__qualname__}"
    )


def canonical_value(value: Any) -> Any:
    """Validate and return a detached canonical JSON-compatible value."""

    return _normalize(value)


def canonical_json(value: Any) -> str:
    """Serialize *value* deterministically as UTF-8 JSON text."""

    normalized = canonical_value(value)
    try:
        return json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=CANONICAL_JSON_SEPARATORS,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:  # defensive: _normalize already checked values
        raise CanonicalizationError(str(exc)) from exc


def canonical_bytes(value: Any) -> bytes:
    """Serialize *value* using :func:`canonical_json` and UTF-8."""

    return canonical_json(value).encode("utf-8")


# Descriptive aliases keep the contract vocabulary discoverable without
# creating a second serialization implementation.
canonical_json_bytes = canonical_bytes


def digest_bytes(data: bytes) -> str:
    """Return the full lowercase SHA-256 digest for bytes."""

    return hashlib.sha256(data).hexdigest()


def digest_for(value: Any) -> str:
    """Return the SHA-256 digest of canonical JSON for a JSON value."""

    if isinstance(value, bytes):
        return digest_bytes(value)
    return digest_bytes(canonical_bytes(value))


sha256_digest = digest_for


def content_id(prefix: str, value: Any) -> str:
    """Return a readable, content-derived ID with a full SHA-256 suffix."""

    if not prefix or not isinstance(prefix, str):
        raise CanonicalizationError("content-address prefix must be a non-empty string")
    if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in prefix):
        raise CanonicalizationError(f"invalid content-address prefix: {prefix!r}")
    return f"{prefix}_{digest_for(value)}"


def chain_digest(previous_digest: str, record: Any) -> str:
    """Hash one record into a deterministic hash chain."""

    if not isinstance(previous_digest, str) or len(previous_digest) != 64:
        raise CanonicalizationError("previous chain digest must be a 64-character SHA-256 hex value")
    try:
        int(previous_digest, 16)
    except ValueError as exc:
        raise CanonicalizationError("previous chain digest is not hexadecimal") from exc
    return digest_for({"previous": previous_digest, "record": record})


def canonical_jsonl(records: Iterable[Any]) -> str:
    """Serialize records as deterministic JSONL with one final newline.

    The caller controls ordering.  Exporters sort records by their contract
    sequence before calling this function; this primitive does not guess an
    ordering that might change semantic meaning.
    """

    return "".join(canonical_json(record) + "\n" for record in records)


def parse_canonical_jsonl(text: str, *, require_canonical: bool = True) -> list[Any]:
    """Parse JSONL and optionally reject non-canonical line encodings."""

    if not isinstance(text, str):
        raise CanonicalizationError("JSONL input must be text")
    if text == "":
        return []
    lines = text.splitlines()
    records = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise CanonicalizationError(f"blank JSONL line at {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CanonicalizationError(f"invalid JSONL line {line_number}: {exc}") from exc
        if require_canonical and canonical_json(record) != line:
            raise CanonicalizationError(f"non-canonical JSONL line at {line_number}")
        records.append(record)
    return records


def read_canonical_jsonl(path: Union[str, Path], *, require_canonical: bool = True) -> list[Any]:
    """Read and parse a canonical JSONL file."""

    return parse_canonical_jsonl(Path(path).read_text(encoding="utf-8"), require_canonical=require_canonical)


def write_canonical_jsonl(path: Union[str, Path], records: Iterable[Any]) -> Path:
    """Write canonical JSONL with a durable parent directory."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_jsonl(records), encoding="utf-8")
    return output


def collection_digest(records: Iterable[Any]) -> str:
    """Hash an already semantically ordered collection as canonical JSONL."""

    return digest_bytes(canonical_jsonl(records).encode("utf-8"))


def failure_family_root(
    task_family: str,
    diagnostic_enum: str,
    normalized_public_locus: str,
    public_rule_id: str,
    *,
    infrastructure_incident_id: Optional[str] = None,
) -> str:
    """Return the ADR-0001 failure-family root digest.

    The public failure identity is the ordered four-tuple from ADR-0001.  An
    ``INTERNAL_ERROR`` is infrastructure loss rather than a model failure, so
    its signed incident ID is appended as a fifth canonical component.  This
    keeps ordinary model failures grouped while preventing infrastructure
    incidents from collapsing into one public family.
    """

    identity = [task_family, diagnostic_enum, normalized_public_locus, public_rule_id]
    if diagnostic_enum == "INTERNAL_ERROR":
        if not isinstance(infrastructure_incident_id, str) or not infrastructure_incident_id:
            raise CanonicalizationError("INTERNAL_ERROR failure roots require a signed incident ID")
        identity.append(infrastructure_incident_id)
    elif infrastructure_incident_id is not None:
        raise CanonicalizationError("only INTERNAL_ERROR failure roots may include an incident ID")
    return digest_for(identity)


def utc_now_iso() -> str:
    """Return an explicit UTC timestamp for private ledger metadata."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def validate_sha256(value: Any, field: str = "digest") -> str:
    """Validate and return a lowercase SHA-256 hex string."""

    if not isinstance(value, str) or len(value) != 64:
        raise CanonicalizationError(f"{field} must be a full SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CanonicalizationError(f"{field} must be hexadecimal") from exc
    return value.lower()


__all__ = [
    "CANONICAL_JSON_SEPARATORS",
    "GENESIS_HASH",
    "canonical_bytes",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_jsonl",
    "canonical_value",
    "chain_digest",
    "collection_digest",
    "content_id",
    "digest_bytes",
    "digest_for",
    "failure_family_root",
    "parse_canonical_jsonl",
    "read_canonical_jsonl",
    "sha256_digest",
    "utc_now_iso",
    "validate_sha256",
    "write_canonical_jsonl",
]
