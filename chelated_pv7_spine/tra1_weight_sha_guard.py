"""TRA-1 state identity that binds weight revisions, not only config and tokenizer.

MemT-style ``_state_identity()`` (arXiv:2610.26825v5) can hash config, tokenizer,
and shape while omitting base weights and LoRA adapter weights. A later
fine-tune then reuses KV pages, cartridges, or cached state in silence.
``TRA1StateIdentityContract`` refuses that reuse.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping, Union

from chelated_pv7_spine import _canonical_json

_SHA256 = "0123456789abcdef"
Number = Union[int, float]
BytesLike = Union[bytes, bytearray, memoryview]


class StaleWeightRevisionError(RuntimeError):
    """KV, cartridge, or runtime state does not match the weight revision."""


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_digest(payload: bytes) -> str:
    """Hex SHA-256 of raw weight or tokenizer bytes."""
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise StaleWeightRevisionError("weight digest input must be bytes")
    return _sha256_hex(bytes(payload))


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _SHA256 for char in value):
        raise StaleWeightRevisionError(f"{name} must be 64 lowercase hex characters")
    return value


def _require_token(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or any(char.isspace() for char in value):
        raise StaleWeightRevisionError(f"{name} must be a non-empty token")
    return value


def _canonicalize_rope_base(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StaleWeightRevisionError("rope_freq_base must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise StaleWeightRevisionError("rope_freq_base must be a positive finite number")
    return format(number, ".16g")


def config_only_state_identity(
    config_sha256: str,
    tokenizer_sha256: str,
    shape: tuple,
) -> str:
    """Reproduce the MemT omission: config, tokenizer, and shape, with no weights.

    This helper exists so tests can show the broken identity staying stable
    across a weight patch. The cache below never calls it.
    """
    _require_sha256("config_sha256", config_sha256)
    _require_sha256("tokenizer_sha256", tokenizer_sha256)
    if not isinstance(shape, tuple) or not shape or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 1 for dim in shape):
        raise StaleWeightRevisionError("shape must be a tuple of positive integers")
    payload = _canonical_json(
        {
            "identity": "memt_config_tokenizer_shape",
            "config_sha256": config_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "shape": list(shape),
        }
    )
    return _sha256_hex(payload)


@dataclass(frozen=True)
class TRA1StateIdentityContract:
    """Fail-closed cryptographic identity of one weight and rope revision."""

    base_weights_sha256: str
    adapter_weights_sha256: str
    config_sha256: str
    tokenizer_sha256: str
    kv_or_state_dtype: str
    rope_freq_base: Number
    rope_scaling_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_weights_sha256", _require_sha256("base_weights_sha256", self.base_weights_sha256))
        object.__setattr__(
            self,
            "adapter_weights_sha256",
            _require_sha256("adapter_weights_sha256", self.adapter_weights_sha256),
        )
        object.__setattr__(self, "config_sha256", _require_sha256("config_sha256", self.config_sha256))
        object.__setattr__(self, "tokenizer_sha256", _require_sha256("tokenizer_sha256", self.tokenizer_sha256))
        object.__setattr__(self, "kv_or_state_dtype", _require_token("kv_or_state_dtype", self.kv_or_state_dtype))
        object.__setattr__(self, "rope_freq_base", _canonicalize_rope_base(self.rope_freq_base))
        object.__setattr__(self, "rope_scaling_hash", _require_sha256("rope_scaling_hash", self.rope_scaling_hash))

    def material(self) -> dict:
        return {
            "base_weights_sha256": self.base_weights_sha256,
            "adapter_weights_sha256": self.adapter_weights_sha256,
            "config_sha256": self.config_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "kv_or_state_dtype": self.kv_or_state_dtype,
            "rope_freq_base": self.rope_freq_base,
            "rope_scaling_hash": self.rope_scaling_hash,
        }

    def digest(self) -> str:
        """SHA-256 over the seven bound fields. Weight bytes are inside the hash."""
        return _sha256_hex(_canonical_json(self.material()))


class TRA1BoundStateCache:
    """State pages sealed to a full weight-revision digest."""

    def __init__(self) -> None:
        self._pages: dict[tuple[bytes, str], bytes] = {}

    def seal(self, page_id: bytes, contract: TRA1StateIdentityContract, payload: bytes) -> str:
        if not isinstance(page_id, (bytes, bytearray)) or not page_id:
            raise StaleWeightRevisionError("page_id must be non-empty bytes")
        if not isinstance(contract, TRA1StateIdentityContract):
            raise StaleWeightRevisionError("seal requires a TRA1StateIdentityContract")
        if not isinstance(payload, (bytes, bytearray)):
            raise StaleWeightRevisionError("state payload must be bytes")
        digest = contract.digest()
        self._pages[(bytes(page_id), digest)] = bytes(payload)
        return digest

    def reuse(self, page_id: bytes, contract: TRA1StateIdentityContract) -> bytes:
        """Return the sealed page or raise. Config-only identity is not consulted."""
        if not isinstance(page_id, (bytes, bytearray)) or not page_id:
            raise StaleWeightRevisionError("page_id must be non-empty bytes")
        if not isinstance(contract, TRA1StateIdentityContract):
            raise StaleWeightRevisionError("reuse requires a TRA1StateIdentityContract")
        digest = contract.digest()
        try:
            return self._pages[(bytes(page_id), digest)]
        except KeyError as exc:
            raise StaleWeightRevisionError(
                "stale weight revision: refusing KV, cartridge, or state reuse"
            ) from exc


def reproduce_memt_weight_omission(
    *,
    config_sha256: str,
    tokenizer_sha256: str,
    shape: tuple,
    base_weights_a: bytes,
    base_weights_b: bytes,
    adapter_weights_a: bytes,
    adapter_weights_b: bytes,
    kv_or_state_dtype: str,
    rope_freq_base: Number,
    rope_scaling_hash: str,
    page_id: bytes = b"kv-page-0",
    payload: bytes = b"cached-state",
) -> dict:
    """Show config-only identity matching after a weight patch, then fail closed.

    Returns a report whose ``silent_state_reuses`` is 0. A successful reuse of
    the patched contract would raise ``StaleWeightRevisionError`` instead of
    being counted.
    """
    if base_weights_a == base_weights_b and adapter_weights_a == adapter_weights_b:
        raise StaleWeightRevisionError("the omission fixture must patch base or adapter weights")
    omitted_a = config_only_state_identity(config_sha256, tokenizer_sha256, shape)
    omitted_b = config_only_state_identity(config_sha256, tokenizer_sha256, shape)
    if omitted_a != omitted_b:
        raise StaleWeightRevisionError("config-only identity moved without a config change")
    original = TRA1StateIdentityContract(
        base_weights_sha256=sha256_digest(base_weights_a),
        adapter_weights_sha256=sha256_digest(adapter_weights_a),
        config_sha256=config_sha256,
        tokenizer_sha256=tokenizer_sha256,
        kv_or_state_dtype=kv_or_state_dtype,
        rope_freq_base=rope_freq_base,
        rope_scaling_hash=rope_scaling_hash,
    )
    patched = TRA1StateIdentityContract(
        base_weights_sha256=sha256_digest(base_weights_b),
        adapter_weights_sha256=sha256_digest(adapter_weights_b),
        config_sha256=config_sha256,
        tokenizer_sha256=tokenizer_sha256,
        kv_or_state_dtype=kv_or_state_dtype,
        rope_freq_base=rope_freq_base,
        rope_scaling_hash=rope_scaling_hash,
    )
    if original.digest() == patched.digest():
        raise StaleWeightRevisionError("full state identity ignored a weight revision")
    cache = TRA1BoundStateCache()
    cache.seal(page_id, original, payload)
    reused = cache.reuse(page_id, original)
    if reused != payload:
        raise StaleWeightRevisionError("same-revision reuse returned the wrong page")
    silent = 0
    for label, candidate in (
        ("patched_base_or_adapter", patched),
        (
            "patched_base_only",
            TRA1StateIdentityContract(
                base_weights_sha256=sha256_digest(base_weights_b),
                adapter_weights_sha256=original.adapter_weights_sha256,
                config_sha256=config_sha256,
                tokenizer_sha256=tokenizer_sha256,
                kv_or_state_dtype=kv_or_state_dtype,
                rope_freq_base=rope_freq_base,
                rope_scaling_hash=rope_scaling_hash,
            ),
        ),
        (
            "patched_adapter_only",
            TRA1StateIdentityContract(
                base_weights_sha256=original.base_weights_sha256,
                adapter_weights_sha256=sha256_digest(adapter_weights_b),
                config_sha256=config_sha256,
                tokenizer_sha256=tokenizer_sha256,
                kv_or_state_dtype=kv_or_state_dtype,
                rope_freq_base=rope_freq_base,
                rope_scaling_hash=rope_scaling_hash,
            ),
        ),
    ):
        if candidate.digest() == original.digest():
            if label == "patched_base_or_adapter":
                continue
            if label == "patched_base_only" and base_weights_a == base_weights_b:
                continue
            if label == "patched_adapter_only" and adapter_weights_a == adapter_weights_b:
                continue
        try:
            cache.reuse(page_id, candidate)
        except StaleWeightRevisionError:
            continue
        silent += 1
    if silent:
        raise StaleWeightRevisionError(f"silent state reuse across weight revisions: {silent}")
    return {
        "failure_class": "memt_config_only_state_identity",
        "config_tokenizer_shape_digest": omitted_a,
        "config_only_identity_stable_across_weight_patch": True,
        "original_digest": original.digest(),
        "patched_digest": patched.digest(),
        "digests_differ": True,
        "silent_state_reuses": 0,
        "reuse_error": "StaleWeightRevisionError",
    }


def contract_from_mapping(fields: Mapping[str, object]) -> TRA1StateIdentityContract:
    """Build a contract from the seven canonical field names."""
    required = (
        "base_weights_sha256",
        "adapter_weights_sha256",
        "config_sha256",
        "tokenizer_sha256",
        "kv_or_state_dtype",
        "rope_freq_base",
        "rope_scaling_hash",
    )
    missing = [name for name in required if name not in fields]
    if missing:
        raise StaleWeightRevisionError(f"missing identity fields: {', '.join(missing)}")
    rope_base = fields["rope_freq_base"]
    if not isinstance(rope_base, (int, float)) or isinstance(rope_base, bool):
        raise StaleWeightRevisionError("rope_freq_base must be a finite number")
    return TRA1StateIdentityContract(
        base_weights_sha256=str(fields["base_weights_sha256"]),
        adapter_weights_sha256=str(fields["adapter_weights_sha256"]),
        config_sha256=str(fields["config_sha256"]),
        tokenizer_sha256=str(fields["tokenizer_sha256"]),
        kv_or_state_dtype=str(fields["kv_or_state_dtype"]),
        rope_freq_base=rope_base,
        rope_scaling_hash=str(fields["rope_scaling_hash"]),
    )


EMPTY_ADAPTER_SHA256 = sha256_digest(b"")
"""Digest of an absent LoRA adapter. A later adapter patch must not reuse state."""
