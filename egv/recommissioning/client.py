"""Receipt-bound, loopback-only model transport for recommissioning pilots.

Production construction always creates the no-proxy transport internally and
requires an externally issued trusted launch receipt. Tests have a separate,
visibly test-only constructor. Generation consumes an immutable prompt
envelope and a pinned-tokenizer preflight rather than arbitrary messages.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple, Union
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from ..canonical import canonical_json, canonical_value, digest_bytes, digest_for
from ..evaluation.sandbox import validate_candidate_source_contract
from .protocol import PromptEnvelope, TokenBudget


GenerationBudget = TokenBudget


class RecommissioningClientError(RuntimeError):
    """The bounded model-service contract could not be proven."""


_DIGEST = re.compile(r"[0-9a-f]{64}")
_ACTION_IDS = ("PROPOSE", "EDIT", "CRITIQUE", "COMMIT")
_RESPONSE_FIELDS = {"action", "source", "summary", "evidence_ids"}
_MAX_SOURCE_CHARACTERS = 64 * 1024
_MAX_SUMMARY_CHARACTERS = 1024
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_MAX_JSON_DEPTH = 16
_MAX_JSON_NODES = 8192
_CONSTRUCTION_TOKEN = object()


def _require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise RecommissioningClientError("{} is not a full lowercase SHA-256 digest".format(label))
    return value


def _require_text(value: Any, label: str, *, max_characters: int, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()) or len(value) > max_characters:
        raise RecommissioningClientError("{} is empty or exceeds its character ceiling".format(label))
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise RecommissioningClientError("{} contains invalid Unicode".format(label)) from exc
    if "\x00" in value:
        raise RecommissioningClientError("{} contains a NUL character".format(label))
    return value


def _bounded_json(value: Any, label: str) -> Any:
    nodes = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise RecommissioningClientError("{} exceeds the JSON node ceiling".format(label))
        if depth > _MAX_JSON_DEPTH:
            raise RecommissioningClientError("{} exceeds the JSON depth ceiling".format(label))
        if item is None or type(item) in {bool, int}:
            return
        if isinstance(item, float):
            raise RecommissioningClientError("{} contains an unapproved floating-point value".format(label))
        if isinstance(item, str):
            _require_text(item, label + " string", max_characters=2 * 1024 * 1024, allow_empty=True)
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                _require_text(key, label + " key", max_characters=256)
                visit(child, depth + 1)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child, depth + 1)
            return
        raise RecommissioningClientError("{} contains a non-JSON value".format(label))

    visit(value, 0)
    return canonical_value(value)


def _strict_json_loads(body: bytes, label: str) -> Any:
    if not isinstance(body, bytes) or len(body) > _MAX_RESPONSE_BYTES:
        raise RecommissioningClientError("{} exceeds the response byte ceiling".format(label))

    def closed_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        value: Dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise RecommissioningClientError("{} contains duplicate JSON keys".format(label))
            value[key] = item
        return value

    try:
        decoded = body.decode("utf-8", errors="strict")
        return _bounded_json(json.loads(decoded, object_pairs_hook=closed_pairs), label)
    except RecommissioningClientError:
        raise
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise RecommissioningClientError("{} did not return strict UTF-8 JSON".format(label)) from exc


@dataclass(frozen=True)
class ModelServiceProfile:
    model_bundle: str
    model_id: str
    model_revision: str
    engine_image_digest: str
    max_model_len: int
    precision: str
    kv_cache_dtype: str
    tokenizer_digest: str
    chat_template_digest: str

    def validate(self) -> None:
        if self.model_bundle not in {"LIGHTNING", "SUPER"}:
            raise RecommissioningClientError("model profile uses an unknown bundle")
        for value, label in (
            (self.model_id, "model ID"),
            (self.precision, "precision"),
            (self.kv_cache_dtype, "KV-cache dtype"),
        ):
            _require_text(value, label, max_characters=256)
        if not isinstance(self.model_revision, str) or re.fullmatch(r"[0-9a-f]{40,64}", self.model_revision) is None:
            raise RecommissioningClientError("model revision is not an immutable lowercase identity")
        if not isinstance(self.engine_image_digest, str) or re.fullmatch(
            r"sha256:[0-9a-f]{64}", self.engine_image_digest
        ) is None:
            raise RecommissioningClientError("engine image must use a sha256 repository digest")
        for value, label in (
            (self.tokenizer_digest, "tokenizer digest"),
            (self.chat_template_digest, "chat-template digest"),
        ):
            _require_digest(value, label)
        if type(self.max_model_len) is not int or not 512 <= self.max_model_len <= 1048576:
            raise RecommissioningClientError("maximum model length is outside the bounded contract")

    def public_manifest(self) -> Dict[str, Any]:
        self.validate()
        return {
            "schema_version": "egv-recommissioning-model-service-v2",
            "model_bundle": self.model_bundle,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "engine_image_digest": self.engine_image_digest,
            "max_model_len": self.max_model_len,
            "precision": self.precision,
            "kv_cache_dtype": self.kv_cache_dtype,
            "tokenizer_digest": self.tokenizer_digest,
            "chat_template_digest": self.chat_template_digest,
        }

    @property
    def digest(self) -> str:
        return digest_for(self.public_manifest())


@dataclass(frozen=True, init=False)
class TrustedLaunchReceipt:
    """Validated binding to a launch receipt issued by a trusted controller.

    This type never creates a production launch receipt. ``from_controller_record``
    only accepts an already-issued digest and a private issuer token supplied by
    the closed controller. ``for_test`` is the sole self-generated fixture path.
    """

    model_profile_digest: str
    runtime_config_digest: str
    launch_receipt_digest: str
    service_nonce: str
    issuer_token: bytes = field(repr=False)
    test_only: bool

    @classmethod
    def _construct(
        cls,
        construction_token: object,
        record: Mapping[str, Any],
        issuer_token: bytes,
        *,
        test_only: bool,
    ) -> "TrustedLaunchReceipt":
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise RecommissioningClientError("trusted launch receipt construction is a closed controller boundary")
        expected = {
            "schema_version",
            "model_profile_digest",
            "runtime_config_digest",
            "launch_receipt_digest",
            "service_nonce",
        }
        if not isinstance(record, Mapping) or set(record) != expected:
            raise RecommissioningClientError("trusted launch receipt is not a closed controller record")
        if record.get("schema_version") != "egv-trusted-launch-receipt-v1":
            raise RecommissioningClientError("trusted launch receipt schema is unsupported")
        if not isinstance(issuer_token, bytes) or len(issuer_token) < 32:
            raise RecommissioningClientError("trusted launch issuer token must be at least 256 private bits")
        values = {
            "model_profile_digest": _require_digest(record["model_profile_digest"], "launch model-profile digest"),
            "runtime_config_digest": _require_digest(record["runtime_config_digest"], "runtime-config digest"),
            "launch_receipt_digest": _require_digest(record["launch_receipt_digest"], "trusted launch receipt digest"),
            "service_nonce": _require_text(record["service_nonce"], "service nonce", max_characters=128),
            "issuer_token": bytes(issuer_token),
            "test_only": test_only,
        }
        result = object.__new__(cls)
        for field_name in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            object.__setattr__(result, field_name, values[field_name])
        return result

    @classmethod
    def from_controller_record(
        cls, record: Mapping[str, Any], *, issuer_token: bytes
    ) -> "TrustedLaunchReceipt":
        return cls._construct(_CONSTRUCTION_TOKEN, record, issuer_token, test_only=False)

    @classmethod
    def for_test(cls, profile: ModelServiceProfile) -> "TrustedLaunchReceipt":
        profile.validate()
        record = {
            "schema_version": "egv-trusted-launch-receipt-v1",
            "model_profile_digest": profile.digest,
            "runtime_config_digest": digest_for("test-runtime-config"),
            "launch_receipt_digest": digest_for("test-launch-receipt-issued-outside-endpoint"),
            "service_nonce": "test-service-nonce",
        }
        return cls._construct(
            _CONSTRUCTION_TOKEN,
            record,
            b"test-only-issuer-token-material!" * 2,
            test_only=True,
        )

    def public_binding_record(self) -> Dict[str, Any]:
        return {
            "model_profile_digest": self.model_profile_digest,
            "runtime_config_digest": self.runtime_config_digest,
            "launch_receipt_digest": self.launch_receipt_digest,
            "service_nonce": self.service_nonce,
        }


@dataclass(frozen=True, init=False)
class ServiceAttestation:
    model_profile_digest: str
    runtime_config_digest: str
    launch_receipt_digest: str
    observed_model_id: str
    service_nonce: str
    health_body_digest: str
    model_list_digest: str
    issuer_proof: str
    attestation_digest: str
    issuer_token: bytes = field(repr=False)

    @classmethod
    def _from_observation(
        cls,
        trusted: TrustedLaunchReceipt,
        *,
        observed_model_id: str,
        health_body: bytes,
        model_list_body: bytes,
    ) -> "ServiceAttestation":
        core = {
            "schema_version": "egv-recommissioning-service-attestation-v2",
            **trusted.public_binding_record(),
            "observed_model_id": observed_model_id,
            "health_body_digest": digest_bytes(health_body),
            "model_list_digest": digest_bytes(model_list_body),
        }
        proof = hmac.new(trusted.issuer_token, canonical_json(core).encode("utf-8"), hashlib.sha256).hexdigest()
        public = {**core, "issuer_proof": proof}
        values = {
            **{key: public[key] for key in public if key != "schema_version"},
            "attestation_digest": digest_for(public),
            "issuer_token": trusted.issuer_token,
        }
        result = object.__new__(cls)
        for field_name in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            object.__setattr__(result, field_name, values[field_name])
        result.validate()
        return result

    def core_record(self) -> Dict[str, Any]:
        return {
            "schema_version": "egv-recommissioning-service-attestation-v2",
            "model_profile_digest": self.model_profile_digest,
            "runtime_config_digest": self.runtime_config_digest,
            "launch_receipt_digest": self.launch_receipt_digest,
            "service_nonce": self.service_nonce,
            "observed_model_id": self.observed_model_id,
            "health_body_digest": self.health_body_digest,
            "model_list_digest": self.model_list_digest,
        }

    def public_record(self) -> Dict[str, Any]:
        self.validate()
        return {**self.core_record(), "issuer_proof": self.issuer_proof, "attestation_digest": self.attestation_digest}

    def validate(self) -> None:
        for value, label in (
            (self.model_profile_digest, "attestation model-profile digest"),
            (self.runtime_config_digest, "attestation runtime-config digest"),
            (self.launch_receipt_digest, "attestation launch-receipt digest"),
            (self.health_body_digest, "attestation health-body digest"),
            (self.model_list_digest, "attestation model-list digest"),
            (self.issuer_proof, "attestation issuer proof"),
            (self.attestation_digest, "service-attestation digest"),
        ):
            _require_digest(value, label)
        _require_text(self.observed_model_id, "observed model ID", max_characters=256)
        _require_text(self.service_nonce, "service nonce", max_characters=128)
        if not isinstance(self.issuer_token, bytes) or len(self.issuer_token) < 32:
            raise RecommissioningClientError("service attestation lost its private issuer binding")
        expected_proof = hmac.new(
            self.issuer_token, canonical_json(self.core_record()).encode("utf-8"), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected_proof, self.issuer_proof):
            raise RecommissioningClientError("service attestation issuer proof does not bind the observation")
        expected_digest = digest_for({**self.core_record(), "issuer_proof": self.issuer_proof})
        if expected_digest != self.attestation_digest:
            raise RecommissioningClientError("service attestation digest does not bind the observation")


class Transport(Protocol):
    def __call__(self, method: str, path: str, body: Optional[bytes], timeout: float) -> Tuple[int, bytes]:
        ...


class _RejectRedirectHandler(urllib_request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


class UrllibLoopbackTransport:
    """Exact production HTTP transport: numeric loopback, no proxy, no redirects."""

    def __init__(self, endpoint: str) -> None:
        parsed = urllib_parse.urlsplit(endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
            or parsed.port is None
        ):
            raise RecommissioningClientError("model endpoint must be an explicit loopback-only HTTP origin")
        self._origin = "http://{}:{}".format("[::1]" if parsed.hostname == "::1" else parsed.hostname, parsed.port)
        # An empty ProxyHandler is essential: urllib otherwise inherits
        # environment proxy configuration even for an explicit loopback URL.
        self._opener = urllib_request.build_opener(urllib_request.ProxyHandler({}), _RejectRedirectHandler())

    def __call__(self, method: str, path: str, body: Optional[bytes], timeout: float) -> Tuple[int, bytes]:
        if not path.startswith("/") or "?" in path or "#" in path:
            raise RecommissioningClientError("model request path is not closed")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib_request.Request(self._origin + path, data=body, headers=headers, method=method)
        try:
            with self._opener.open(request, timeout=timeout) as response:
                payload = response.read(_MAX_RESPONSE_BYTES + 1)
                status = int(response.status)
        except (urllib_error.URLError, OSError, TimeoutError) as exc:
            raise RecommissioningClientError("bounded model-service request failed") from exc
        if len(payload) > _MAX_RESPONSE_BYTES:
            raise RecommissioningClientError("model-service response exceeds the byte ceiling")
        return status, payload


def _observe_service(
    transport: Transport,
    profile: ModelServiceProfile,
    trusted: TrustedLaunchReceipt,
    *,
    timeout_seconds: float,
) -> ServiceAttestation:
    health_status, health_body = transport("GET", "/health", None, timeout_seconds)
    if health_status != 200:
        raise RecommissioningClientError("model health endpoint is not ready")
    models_status, models_body = transport("GET", "/v1/models", None, timeout_seconds)
    if models_status != 200:
        raise RecommissioningClientError("model-list endpoint is not ready")
    models = _strict_json_loads(models_body, "model-list endpoint")
    records = models.get("data") if isinstance(models, Mapping) else None
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], Mapping):
        raise RecommissioningClientError("model-list response does not identify exactly one served model")
    if records[0].get("id") != profile.model_id:
        raise RecommissioningClientError("served model identity differs from the frozen profile")
    if trusted.model_profile_digest != profile.digest:
        raise RecommissioningClientError("trusted launch receipt differs from the model profile")
    return ServiceAttestation._from_observation(
        trusted,
        observed_model_id=profile.model_id,
        health_body=health_body,
        model_list_body=models_body,
    )


@dataclass(frozen=True, init=False)
class TokenizationPreflight:
    prompt_digest: str
    model_profile_digest: str
    tokenizer_digest: str
    chat_template_digest: str
    rendered_prompt_digest: str
    token_ids_digest: str
    prompt_tokens: int
    preflight_digest: str

    @classmethod
    def from_pinned_renderer(
        cls,
        envelope: PromptEnvelope,
        profile: ModelServiceProfile,
        *,
        rendered_prompt: Union[str, bytes],
        token_ids: Sequence[int],
    ) -> "TokenizationPreflight":
        envelope.validate(model_profile_digest=profile.digest)
        profile.validate()
        if isinstance(rendered_prompt, str):
            try:
                rendered = rendered_prompt.encode("utf-8", errors="strict")
            except UnicodeEncodeError as exc:
                raise RecommissioningClientError("rendered prompt contains invalid Unicode") from exc
        elif isinstance(rendered_prompt, bytes):
            rendered = bytes(rendered_prompt)
            try:
                rendered.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise RecommissioningClientError("rendered prompt is not strict UTF-8") from exc
        else:
            raise RecommissioningClientError("rendered prompt must be exact UTF-8 bytes or text")
        if not rendered or len(rendered) > 4 * 1024 * 1024:
            raise RecommissioningClientError("rendered prompt is empty or exceeds the byte ceiling")
        if not isinstance(token_ids, (tuple, list)) or not token_ids:
            raise RecommissioningClientError("pinned tokenizer returned no token IDs")
        if any(type(token) is not int or token < 0 or token > 2**31 - 1 for token in token_ids):
            raise RecommissioningClientError("pinned tokenizer returned a malformed token ID")
        values = {
            "prompt_digest": envelope.prompt_digest,
            "model_profile_digest": profile.digest,
            "tokenizer_digest": profile.tokenizer_digest,
            "chat_template_digest": profile.chat_template_digest,
            "rendered_prompt_digest": digest_bytes(rendered),
            "token_ids_digest": digest_for(list(token_ids)),
            "prompt_tokens": len(token_ids),
        }
        core = {"schema_version": "egv-recommissioning-tokenization-preflight-v1", **values}
        values["preflight_digest"] = digest_for(core)
        result = object.__new__(cls)
        for field_name in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            object.__setattr__(result, field_name, values[field_name])
        result.validate(envelope=envelope, profile=profile)
        return result

    def core_record(self) -> Dict[str, Any]:
        return {
            "schema_version": "egv-recommissioning-tokenization-preflight-v1",
            "prompt_digest": self.prompt_digest,
            "model_profile_digest": self.model_profile_digest,
            "tokenizer_digest": self.tokenizer_digest,
            "chat_template_digest": self.chat_template_digest,
            "rendered_prompt_digest": self.rendered_prompt_digest,
            "token_ids_digest": self.token_ids_digest,
            "prompt_tokens": self.prompt_tokens,
        }

    def validate(
        self,
        *,
        envelope: Optional[PromptEnvelope] = None,
        profile: Optional[ModelServiceProfile] = None,
    ) -> None:
        for value, label in (
            (self.prompt_digest, "tokenization prompt digest"),
            (self.model_profile_digest, "tokenization model-profile digest"),
            (self.tokenizer_digest, "tokenizer digest"),
            (self.chat_template_digest, "chat-template digest"),
            (self.rendered_prompt_digest, "rendered-prompt digest"),
            (self.token_ids_digest, "token-ID digest"),
            (self.preflight_digest, "tokenization-preflight digest"),
        ):
            _require_digest(value, label)
        if type(self.prompt_tokens) is not int or self.prompt_tokens < 1:
            raise RecommissioningClientError("prompt-token preflight is malformed")
        if digest_for(self.core_record()) != self.preflight_digest:
            raise RecommissioningClientError("tokenization preflight digest does not bind the record")
        if envelope is not None and self.prompt_digest != envelope.prompt_digest:
            raise RecommissioningClientError("tokenization preflight differs from the prompt envelope")
        if profile is not None:
            profile.validate()
            if (
                self.model_profile_digest != profile.digest
                or self.tokenizer_digest != profile.tokenizer_digest
                or self.chat_template_digest != profile.chat_template_digest
            ):
                raise RecommissioningClientError("tokenization preflight differs from the pinned model profile")


@dataclass(frozen=True)
class CoordinateTokenState:
    coordinate_id: str
    budget_digest: str
    cumulative_reserved_tokens: int
    debit_count: int
    last_debit_digest: Optional[str]
    state_digest: str

    @classmethod
    def genesis(cls, coordinate_id: str, budget_digest: str) -> "CoordinateTokenState":
        core = {
            "schema_version": "egv-recommissioning-coordinate-token-state-v1",
            "coordinate_id": _require_digest(coordinate_id, "coordinate digest"),
            "budget_digest": _require_digest(budget_digest, "coordinate budget digest"),
            "cumulative_reserved_tokens": 0,
            "debit_count": 0,
            "last_debit_digest": None,
        }
        return cls(**{key: value for key, value in core.items() if key != "schema_version"}, state_digest=digest_for(core))

    def core_record(self) -> Dict[str, Any]:
        return {
            "schema_version": "egv-recommissioning-coordinate-token-state-v1",
            "coordinate_id": self.coordinate_id,
            "budget_digest": self.budget_digest,
            "cumulative_reserved_tokens": self.cumulative_reserved_tokens,
            "debit_count": self.debit_count,
            "last_debit_digest": self.last_debit_digest,
        }

    def record(self) -> Dict[str, Any]:
        self.validate()
        return {**self.core_record(), "state_digest": self.state_digest}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "CoordinateTokenState":
        expected = set(cls.genesis("0" * 64, "0" * 64).record())
        if not isinstance(record, Mapping) or set(record) != expected:
            raise RecommissioningClientError("coordinate token state is not a closed record")
        value = cls(
            coordinate_id=record["coordinate_id"],
            budget_digest=record["budget_digest"],
            cumulative_reserved_tokens=record["cumulative_reserved_tokens"],
            debit_count=record["debit_count"],
            last_debit_digest=record["last_debit_digest"],
            state_digest=record["state_digest"],
        )
        value.validate()
        return value

    def validate(self) -> None:
        _require_digest(self.coordinate_id, "coordinate token-state digest")
        _require_digest(self.budget_digest, "coordinate token-state budget digest")
        _require_digest(self.state_digest, "coordinate state digest")
        if self.last_debit_digest is not None:
            _require_digest(self.last_debit_digest, "last coordinate debit digest")
        if type(self.cumulative_reserved_tokens) is not int or self.cumulative_reserved_tokens < 0:
            raise RecommissioningClientError("coordinate cumulative token debit is malformed")
        if type(self.debit_count) is not int or self.debit_count < 0:
            raise RecommissioningClientError("coordinate debit count is malformed")
        if (self.debit_count == 0) != (self.last_debit_digest is None):
            raise RecommissioningClientError("coordinate token-state chain is inconsistent")
        if digest_for(self.core_record()) != self.state_digest:
            raise RecommissioningClientError("coordinate state digest does not bind the record")


@dataclass(frozen=True)
class CoordinateTokenDebit:
    coordinate_id: str
    budget_digest: str
    request_digest: str
    prior_state_digest: str
    reserved_tokens: int
    cumulative_reserved_tokens: int
    debit_count: int
    debit_digest: str
    state_digest: str

    def core_record(self) -> Dict[str, Any]:
        return {
            "schema_version": "egv-recommissioning-coordinate-token-debit-v1",
            "coordinate_id": self.coordinate_id,
            "budget_digest": self.budget_digest,
            "request_digest": self.request_digest,
            "prior_state_digest": self.prior_state_digest,
            "reserved_tokens": self.reserved_tokens,
            "cumulative_reserved_tokens": self.cumulative_reserved_tokens,
            "debit_count": self.debit_count,
        }

    def public_record(self) -> Dict[str, Any]:
        self.validate()
        return {**self.core_record(), "debit_digest": self.debit_digest, "state_digest": self.state_digest}

    def validate(self) -> None:
        for value, label in (
            (self.coordinate_id, "debit coordinate digest"),
            (self.budget_digest, "debit budget digest"),
            (self.request_digest, "debit request digest"),
            (self.prior_state_digest, "debit prior-state digest"),
            (self.debit_digest, "coordinate debit digest"),
            (self.state_digest, "coordinate post-debit state digest"),
        ):
            _require_digest(value, label)
        for value, label in (
            (self.reserved_tokens, "reserved token count"),
            (self.cumulative_reserved_tokens, "cumulative reserved token count"),
            (self.debit_count, "coordinate debit count"),
        ):
            if type(value) is not int or value < 1:
                raise RecommissioningClientError("{} is malformed".format(label))
        if digest_for(self.core_record()) != self.debit_digest:
            raise RecommissioningClientError("coordinate debit digest does not bind the record")
        state_core = {
            "schema_version": "egv-recommissioning-coordinate-token-state-v1",
            "coordinate_id": self.coordinate_id,
            "budget_digest": self.budget_digest,
            "cumulative_reserved_tokens": self.cumulative_reserved_tokens,
            "debit_count": self.debit_count,
            "last_debit_digest": self.debit_digest,
        }
        if digest_for(state_core) != self.state_digest:
            raise RecommissioningClientError("coordinate debit does not bind the post-debit state")


class CoordinateTokenLedger(Protocol):
    durable: bool

    def snapshot(self, coordinate_id: str, budget: TokenBudget) -> CoordinateTokenState:
        ...

    def debit(self, request: "GenerationRequest") -> CoordinateTokenDebit:
        ...


def _next_debit(request: "GenerationRequest", current: CoordinateTokenState) -> Tuple[CoordinateTokenDebit, CoordinateTokenState]:
    current.validate()
    if current.state_digest != request.coordinate_state.state_digest:
        raise RecommissioningClientError("coordinate token state changed after request construction")
    reserved = request.tokenization.prompt_tokens + request.budget.max_completion_tokens
    cumulative = current.cumulative_reserved_tokens + reserved
    if cumulative > request.budget.max_coordinate_tokens:
        raise RecommissioningClientError("coordinate token ceiling would be exceeded")
    debit_core = {
        "schema_version": "egv-recommissioning-coordinate-token-debit-v1",
        "coordinate_id": request.envelope.coordinate_id,
        "budget_digest": request.budget.digest,
        "request_digest": request.request_digest,
        "prior_state_digest": current.state_digest,
        "reserved_tokens": reserved,
        "cumulative_reserved_tokens": cumulative,
        "debit_count": current.debit_count + 1,
    }
    debit_digest = digest_for(debit_core)
    state_core = {
        "schema_version": "egv-recommissioning-coordinate-token-state-v1",
        "coordinate_id": request.envelope.coordinate_id,
        "budget_digest": request.budget.digest,
        "cumulative_reserved_tokens": cumulative,
        "debit_count": current.debit_count + 1,
        "last_debit_digest": debit_digest,
    }
    state = CoordinateTokenState(
        coordinate_id=request.envelope.coordinate_id,
        budget_digest=request.budget.digest,
        cumulative_reserved_tokens=cumulative,
        debit_count=current.debit_count + 1,
        last_debit_digest=debit_digest,
        state_digest=digest_for(state_core),
    )
    state.validate()
    debit = CoordinateTokenDebit(
        coordinate_id=request.envelope.coordinate_id,
        budget_digest=request.budget.digest,
        request_digest=request.request_digest,
        prior_state_digest=current.state_digest,
        reserved_tokens=reserved,
        cumulative_reserved_tokens=cumulative,
        debit_count=current.debit_count + 1,
        debit_digest=debit_digest,
        state_digest=state.state_digest,
    )
    debit.validate()
    return debit, state


class FileCoordinateTokenLedger:
    """Fail-closed, durable, cross-process coordinate token debit store."""

    durable = True

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise RecommissioningClientError("coordinate token ledger root must be a real directory")

    def _state_path(self, coordinate_id: str) -> Path:
        _require_digest(coordinate_id, "coordinate digest")
        return self.root / (coordinate_id + ".json")

    def snapshot(self, coordinate_id: str, budget: TokenBudget) -> CoordinateTokenState:
        budget.validate()
        path = self._state_path(coordinate_id)
        if not path.exists():
            return CoordinateTokenState.genesis(coordinate_id, budget.digest)
        if path.is_symlink() or not path.is_file():
            raise RecommissioningClientError("coordinate token state path is not a regular file")
        try:
            raw = path.read_bytes()
            record = _strict_json_loads(raw, "coordinate token state")
            state = CoordinateTokenState.from_record(record)
        except OSError as exc:
            raise RecommissioningClientError("coordinate token state could not be read") from exc
        if state.coordinate_id != coordinate_id or state.budget_digest != budget.digest:
            raise RecommissioningClientError("coordinate token state differs from the frozen coordinate or budget")
        return state

    def debit(self, request: "GenerationRequest") -> CoordinateTokenDebit:
        request.validate()
        lock_path = self.root / (request.envelope.coordinate_id + ".lock")
        try:
            descriptor = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise RecommissioningClientError("coordinate token ledger is locked or requires manual crash recovery") from exc
        try:
            os.write(descriptor, request.request_digest.encode("ascii"))
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            current = self.snapshot(request.envelope.coordinate_id, request.budget)
            debit, state = _next_debit(request, current)
            destination = self._state_path(request.envelope.coordinate_id)
            temporary = self.root / (request.envelope.coordinate_id + "." + request.request_digest + ".tmp")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            temporary_descriptor = os.open(str(temporary), flags, 0o600)
            try:
                payload = canonical_json(state.record()).encode("utf-8")
                os.write(temporary_descriptor, payload)
                os.fsync(temporary_descriptor)
            finally:
                os.close(temporary_descriptor)
            os.replace(str(temporary), str(destination))
            return debit
        except Exception:
            # A completed state replace is never rolled back. The lock is only
            # coordination; each debit itself is the durable evidence.
            raise
        finally:
            if "descriptor" in locals() and descriptor >= 0:
                os.close(descriptor)
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


class InMemoryCoordinateTokenLedger:
    """Explicitly test-only non-durable ledger."""

    durable = False

    def __init__(self, token: object) -> None:
        if token is not _CONSTRUCTION_TOKEN:
            raise RecommissioningClientError("in-memory token ledger is test-only")
        self._states: Dict[str, CoordinateTokenState] = {}

    @classmethod
    def for_test(cls) -> "InMemoryCoordinateTokenLedger":
        return cls(_CONSTRUCTION_TOKEN)

    def snapshot(self, coordinate_id: str, budget: TokenBudget) -> CoordinateTokenState:
        budget.validate()
        state = self._states.get(coordinate_id) or CoordinateTokenState.genesis(coordinate_id, budget.digest)
        if state.budget_digest != budget.digest:
            raise RecommissioningClientError("test coordinate state differs from the frozen budget")
        return state

    def debit(self, request: "GenerationRequest") -> CoordinateTokenDebit:
        current = self.snapshot(request.envelope.coordinate_id, request.budget)
        debit, state = _next_debit(request, current)
        self._states[request.envelope.coordinate_id] = state
        return debit


@dataclass(frozen=True, init=False)
class GenerationRequest:
    envelope: PromptEnvelope
    budget: TokenBudget
    attestation: ServiceAttestation
    tokenization: TokenizationPreflight
    coordinate_state: CoordinateTokenState
    wire_request_digest: str
    request_digest: str

    @classmethod
    def build(
        cls,
        envelope: PromptEnvelope,
        budget: TokenBudget,
        service_attestation: ServiceAttestation,
        tokenization: TokenizationPreflight,
        coordinate_state: CoordinateTokenState,
        *,
        profile: ModelServiceProfile,
    ) -> "GenerationRequest":
        values = {
            "envelope": envelope,
            "budget": budget,
            "attestation": service_attestation,
            "tokenization": tokenization,
            "coordinate_state": coordinate_state,
        }
        wire = cls._wire_request_for(envelope, budget, profile)
        values["wire_request_digest"] = digest_for(wire)
        core = cls._binding_record_for(
            envelope,
            budget,
            service_attestation,
            tokenization,
            coordinate_state,
            values["wire_request_digest"],
        )
        values["request_digest"] = digest_for(core)
        result = object.__new__(cls)
        for field_name in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            object.__setattr__(result, field_name, values[field_name])
        result.validate(profile=profile, attestation=service_attestation)
        return result

    @staticmethod
    def _wire_request_for(
        envelope: PromptEnvelope, budget: TokenBudget, profile: ModelServiceProfile
    ) -> Dict[str, Any]:
        return {
            "model": profile.model_id,
            "messages": [message.record() for message in envelope.messages],
            "temperature": 0.0,
            "seed": budget.seed,
            "max_tokens": budget.max_completion_tokens,
            "n": 1,
            "response_format": {
                "type": "json_schema",
                "json_schema": candidate_response_schema(envelope.visible_evidence_ids),
            },
        }

    @staticmethod
    def _binding_record_for(
        envelope: PromptEnvelope,
        budget: TokenBudget,
        attestation: ServiceAttestation,
        tokenization: TokenizationPreflight,
        coordinate_state: CoordinateTokenState,
        wire_request_digest: str,
    ) -> Dict[str, Any]:
        return {
            "schema_version": "egv-recommissioning-generation-request-v2",
            "coordinate_id": envelope.coordinate_id,
            "model_bundle": envelope.model_bundle,
            "task_id": envelope.task_id,
            "public_locus": envelope.public_locus,
            "task_adapter_kind": envelope.task_adapter_kind,
            "task_package_digest": envelope.task_package_digest,
            "model_profile_digest": envelope.model_profile_digest,
            "budget_digest": budget.digest,
            "seed": budget.seed,
            "attempt_index": envelope.attempt_index,
            "prompt_digest": envelope.prompt_digest,
            "service_attestation_digest": attestation.attestation_digest,
            "runtime_config_digest": attestation.runtime_config_digest,
            "launch_receipt_digest": attestation.launch_receipt_digest,
            "tokenization_preflight_digest": tokenization.preflight_digest,
            "prompt_tokens": tokenization.prompt_tokens,
            "prior_coordinate_state_digest": coordinate_state.state_digest,
            "prior_coordinate_reserved_tokens": coordinate_state.cumulative_reserved_tokens,
            "wire_request_digest": wire_request_digest,
        }

    def wire_request(self, profile: ModelServiceProfile) -> Dict[str, Any]:
        self.validate(profile=profile)
        return self._wire_request_for(self.envelope, self.budget, profile)

    def binding_record(self) -> Dict[str, Any]:
        return self._binding_record_for(
            self.envelope,
            self.budget,
            self.attestation,
            self.tokenization,
            self.coordinate_state,
            self.wire_request_digest,
        )

    def validate(
        self,
        *,
        profile: Optional[ModelServiceProfile] = None,
        attestation: Optional[ServiceAttestation] = None,
    ) -> None:
        self.envelope.validate(budget=self.budget)
        self.budget.validate(max_model_len=profile.max_model_len if profile is not None else None)
        self.attestation.validate()
        self.tokenization.validate(envelope=self.envelope, profile=profile)
        self.coordinate_state.validate()
        if self.coordinate_state.coordinate_id != self.envelope.coordinate_id:
            raise RecommissioningClientError("coordinate state differs from the prompt coordinate")
        if self.coordinate_state.budget_digest != self.budget.digest:
            raise RecommissioningClientError("coordinate state differs from the prompt budget")
        if self.tokenization.prompt_tokens > self.budget.max_prompt_tokens:
            raise RecommissioningClientError("exact rendered prompt exceeds the prompt-token ceiling")
        if self.tokenization.prompt_tokens + self.budget.max_completion_tokens > self.budget.max_total_tokens:
            raise RecommissioningClientError("exact rendered prompt leaves insufficient total-token budget")
        reserved = self.tokenization.prompt_tokens + self.budget.max_completion_tokens
        if self.coordinate_state.cumulative_reserved_tokens + reserved > self.budget.max_coordinate_tokens:
            raise RecommissioningClientError("generation request exceeds the cumulative coordinate token ceiling")
        if profile is not None:
            profile.validate()
            if self.envelope.model_profile_digest != profile.digest:
                raise RecommissioningClientError("prompt envelope differs from the supplied model profile")
            if self.envelope.model_bundle != profile.model_bundle:
                raise RecommissioningClientError("prompt bundle differs from the supplied model profile")
            if self.attestation.model_profile_digest != profile.digest:
                raise RecommissioningClientError("service attestation differs from the supplied model profile")
            wire = self._wire_request_for(self.envelope, self.budget, profile)
            if digest_for(wire) != self.wire_request_digest:
                raise RecommissioningClientError("wire-request digest does not bind the exact request")
        else:
            _require_digest(self.wire_request_digest, "wire-request digest")
        if attestation is not None and self.attestation.attestation_digest != attestation.attestation_digest:
            raise RecommissioningClientError("request does not use the service's current attestation")
        if digest_for(self.binding_record()) != self.request_digest:
            raise RecommissioningClientError("generation-request digest does not bind every request boundary")


@dataclass(frozen=True)
class TypedCandidateResponse:
    action: str
    source: str
    summary: str
    evidence_ids: Tuple[str, ...]

    @property
    def source_digest(self) -> str:
        return digest_bytes(self.source.encode("utf-8"))

    @classmethod
    def parse(
        cls,
        value: Any,
        *,
        visible_evidence_ids: Sequence[str],
        public_locus: str,
    ) -> "TypedCandidateResponse":
        if not isinstance(value, Mapping) or set(value) != _RESPONSE_FIELDS:
            raise RecommissioningClientError("candidate response does not use the closed field set")
        evidence = value.get("evidence_ids")
        if not isinstance(evidence, list):
            raise RecommissioningClientError("candidate evidence IDs must be a JSON array")
        result = cls(
            action=value.get("action"),
            source=value.get("source"),
            summary=value.get("summary"),
            evidence_ids=tuple(evidence),
        )
        result.validate(public_locus=public_locus, visible_evidence_ids=visible_evidence_ids)
        return result

    def validate(self, *, public_locus: str, visible_evidence_ids: Sequence[str]) -> None:
        if self.action not in _ACTION_IDS:
            raise RecommissioningClientError("candidate response uses an unknown action")
        _require_text(self.source, "candidate source", max_characters=_MAX_SOURCE_CHARACTERS)
        try:
            module = ast.parse(self.source, mode="exec")
        except (SyntaxError, ValueError) as exc:
            raise RecommissioningClientError("candidate source is not one complete Python module") from exc
        if public_locus != "module:main":
            raise RecommissioningClientError("candidate public locus is outside the learnable-v2 contract")
        mains = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main"]
        if len(mains) != 1:
            raise RecommissioningClientError("candidate source does not implement exactly one public main locus")
        arguments = mains[0].args
        if (
            len(arguments.args) != 1
            or arguments.args[0].arg != "value"
            or arguments.defaults
            or arguments.kw_defaults
        ):
            raise RecommissioningClientError("candidate main locus must use the exact main(value) contract")
        violation = validate_candidate_source_contract(self.source.encode("utf-8"))
        if violation is not None:
            raise RecommissioningClientError("candidate source violates pure-return-v1: {}".format(violation))
        _require_text(
            self.summary,
            "candidate summary",
            max_characters=_MAX_SUMMARY_CHARACTERS,
            allow_empty=True,
        )
        if len(self.evidence_ids) > 6 or list(self.evidence_ids) != sorted(set(self.evidence_ids)):
            raise RecommissioningClientError("candidate evidence IDs are not bounded, sorted, and unique")
        if any(not isinstance(item, str) or not item or len(item) > 96 for item in self.evidence_ids):
            raise RecommissioningClientError("candidate evidence ID is malformed")
        allowed = set(visible_evidence_ids)
        if any(item not in allowed for item in self.evidence_ids):
            raise RecommissioningClientError("candidate cited evidence that was not model-visible")

    def private_record(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "source": self.source,
            "summary": self.summary,
            "evidence_ids": list(self.evidence_ids),
        }

    def public_redacted_record(self) -> Dict[str, Any]:
        return {
            "schema_version": "egv-recommissioning-public-candidate-v2",
            "action": self.action,
            "source_digest": self.source_digest,
            "summary_digest": digest_bytes(self.summary.encode("utf-8")),
            "evidence_ids": list(self.evidence_ids),
        }

    def public_record(self) -> Dict[str, Any]:
        """Compatibility name for the explicitly redacted public record."""

        return self.public_redacted_record()


def candidate_response_schema(visible_evidence_ids: Sequence[str] = ()) -> Dict[str, Any]:
    if list(visible_evidence_ids) != sorted(set(visible_evidence_ids)):
        raise RecommissioningClientError("response schema evidence IDs must be sorted and unique")
    evidence_items: Dict[str, Any] = {"type": "string", "minLength": 1, "maxLength": 96}
    if visible_evidence_ids:
        evidence_items["enum"] = list(visible_evidence_ids)
    return {
        "name": "egv_candidate_v2",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": sorted(_RESPONSE_FIELDS),
            "properties": {
                "action": {"type": "string", "enum": list(_ACTION_IDS)},
                "source": {"type": "string", "minLength": 1, "maxLength": _MAX_SOURCE_CHARACTERS},
                "summary": {"type": "string", "maxLength": _MAX_SUMMARY_CHARACTERS},
                "evidence_ids": {
                    "type": "array",
                    "items": evidence_items,
                    "minItems": 0,
                    "maxItems": min(6, len(visible_evidence_ids)),
                    "uniqueItems": True,
                },
            },
        },
    }


class ReceiptSigner(Protocol):
    key_id: str

    def sign_bytes(self, payload: bytes) -> str:
        ...


_PUBLIC_RECEIPT_FIELDS = {
    "schema_version",
    "coordinate_id",
    "model_bundle",
    "task_id",
    "public_locus",
    "task_adapter_kind",
    "task_package_digest",
    "model_profile_digest",
    "budget_digest",
    "seed",
    "attempt_index",
    "prompt_digest",
    "wire_request_digest",
    "request_digest",
    "service_attestation_digest",
    "runtime_config_digest",
    "launch_receipt_digest",
    "tokenization_preflight_digest",
    "rendered_prompt_digest",
    "response_digest",
    "candidate_digest",
    "source_digest",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "max_prompt_tokens",
    "max_completion_tokens",
    "max_total_tokens",
    "max_coordinate_tokens",
    "hard_timeout_ms",
    "elapsed_ms",
    "coordinate_debit_digest",
    "coordinate_reserved_tokens",
    "coordinate_cumulative_reserved_tokens",
    "coordinate_state_digest",
    "signing_key_id",
    "receipt_digest",
    "signature",
}


@dataclass(frozen=True, init=False)
class PublicGenerationReceipt:
    """Signed, self-digested, replay-complete public generation receipt."""

    record_json: str

    @classmethod
    def build(cls, core: Mapping[str, Any], signer: ReceiptSigner) -> "PublicGenerationReceipt":
        if not hasattr(signer, "key_id") or not callable(getattr(signer, "sign_bytes", None)):
            raise RecommissioningClientError("public receipt signer does not use the closed interface")
        _require_text(signer.key_id, "public receipt signing key ID", max_characters=128)
        unsigned = {**canonical_value(core), "signing_key_id": signer.key_id}
        receipt_digest = digest_for(unsigned)
        signable = {**unsigned, "receipt_digest": receipt_digest}
        signature = signer.sign_bytes(canonical_json(signable).encode("utf-8"))
        _require_text(signature, "public receipt signature", max_characters=1024)
        record = {**signable, "signature": signature}
        cls.validate_record(record)
        result = object.__new__(cls)
        object.__setattr__(result, "record_json", canonical_json(record))
        return result

    @classmethod
    def validate_record(cls, record: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(record, Mapping) or set(record) != _PUBLIC_RECEIPT_FIELDS:
            raise RecommissioningClientError("public generation receipt is not a closed record")
        if record.get("schema_version") != "egv-recommissioning-public-generation-receipt-v2":
            raise RecommissioningClientError("public generation receipt schema is unsupported")
        for field_name in (
            "coordinate_id",
            "task_package_digest",
            "model_profile_digest",
            "budget_digest",
            "prompt_digest",
            "wire_request_digest",
            "request_digest",
            "service_attestation_digest",
            "runtime_config_digest",
            "launch_receipt_digest",
            "tokenization_preflight_digest",
            "rendered_prompt_digest",
            "response_digest",
            "candidate_digest",
            "source_digest",
            "coordinate_debit_digest",
            "coordinate_state_digest",
            "receipt_digest",
        ):
            _require_digest(record[field_name], "public receipt " + field_name)
        for field_name in (
            "seed",
            "attempt_index",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "max_prompt_tokens",
            "max_completion_tokens",
            "max_total_tokens",
            "max_coordinate_tokens",
            "hard_timeout_ms",
            "elapsed_ms",
            "coordinate_reserved_tokens",
            "coordinate_cumulative_reserved_tokens",
        ):
            if type(record[field_name]) is not int or record[field_name] < 0:
                raise RecommissioningClientError("public receipt {} is malformed".format(field_name))
        for field_name in (
            "model_bundle",
            "task_id",
            "public_locus",
            "task_adapter_kind",
            "finish_reason",
            "signing_key_id",
            "signature",
        ):
            _require_text(record[field_name], "public receipt " + field_name, max_characters=1024)
        unsigned = {key: value for key, value in record.items() if key not in {"receipt_digest", "signature"}}
        if digest_for(unsigned) != record["receipt_digest"]:
            raise RecommissioningClientError("public generation receipt digest does not bind the record")
        if record["total_tokens"] != record["prompt_tokens"] + record["completion_tokens"]:
            raise RecommissioningClientError("public generation receipt total-token accounting is inconsistent")
        if not 0 <= record["seed"] <= 2**31 - 1 or not 1 <= record["attempt_index"] <= 6:
            raise RecommissioningClientError("public generation receipt coordinate budget is invalid")
        if (
            record["prompt_tokens"] < 1
            or record["completion_tokens"] < 1
            or record["prompt_tokens"] > record["max_prompt_tokens"]
            or record["completion_tokens"] > record["max_completion_tokens"]
            or record["total_tokens"] > record["max_total_tokens"]
            or record["coordinate_cumulative_reserved_tokens"] > record["max_coordinate_tokens"]
        ):
            raise RecommissioningClientError("public generation receipt violates its token ceilings")
        if record["coordinate_reserved_tokens"] != record["prompt_tokens"] + record["max_completion_tokens"]:
            raise RecommissioningClientError("public generation receipt coordinate reservation is inconsistent")
        if record["hard_timeout_ms"] < 1 or record["elapsed_ms"] > record["hard_timeout_ms"]:
            raise RecommissioningClientError("public generation receipt violates its hard elapsed-time ceiling")
        if record["finish_reason"] != "stop":
            raise RecommissioningClientError("public generation receipt has an unapproved finish disposition")
        if record["model_bundle"] not in {"LIGHTNING", "SUPER"} or record["public_locus"] != "module:main":
            raise RecommissioningClientError("public generation receipt has an invalid model/locus binding")
        if record["task_adapter_kind"] != "LEARNABLE_V2" and record["task_adapter_kind"] != "TEST_ONLY":
            raise RecommissioningClientError("public generation receipt has an invalid task-adapter binding")
        return canonical_value(record)

    @property
    def receipt_digest(self) -> str:
        return self.public_record()["receipt_digest"]

    def public_record(self) -> Dict[str, Any]:
        value = json.loads(self.record_json)
        return self.validate_record(value)


GenerationReceipt = PublicGenerationReceipt


@dataclass(frozen=True)
class PrivateGenerationRecord:
    """Private raw model I/O; never emitted by a public projector."""

    request_json: str
    response_json: str
    candidate_json: str
    attestation_private_json: str
    private_record_digest: str

    @classmethod
    def build(
        cls,
        *,
        request_bytes: bytes,
        response_bytes: bytes,
        candidate: TypedCandidateResponse,
        attestation: ServiceAttestation,
    ) -> "PrivateGenerationRecord":
        try:
            request_json = request_bytes.decode("utf-8", errors="strict")
            response_json = response_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RecommissioningClientError("private raw model I/O is not UTF-8") from exc
        # The issuer token is intentionally not persisted even in the private
        # trajectory record. It remains live controller authority only.
        attestation_private = {
            **attestation.public_record(),
            "issuer_token_digest": digest_bytes(attestation.issuer_token),
        }
        core = {
            "schema_version": "egv-recommissioning-private-generation-record-v2",
            "request_json": request_json,
            "response_json": response_json,
            "candidate_json": canonical_json(candidate.private_record()),
            "attestation_private_json": canonical_json(attestation_private),
        }
        return cls(
            request_json=core["request_json"],
            response_json=core["response_json"],
            candidate_json=core["candidate_json"],
            attestation_private_json=core["attestation_private_json"],
            private_record_digest=digest_for(core),
        )

    def private_record(self) -> Dict[str, Any]:
        core = {
            "schema_version": "egv-recommissioning-private-generation-record-v2",
            "request_json": self.request_json,
            "response_json": self.response_json,
            "candidate_json": self.candidate_json,
            "attestation_private_json": self.attestation_private_json,
        }
        if digest_for(core) != self.private_record_digest:
            raise RecommissioningClientError("private generation-record digest does not bind the raw I/O")
        return {**core, "private_record_digest": self.private_record_digest}


@dataclass(frozen=True)
class GenerationResult:
    candidate: TypedCandidateResponse
    private_record: PrivateGenerationRecord
    public_receipt: PublicGenerationReceipt
    coordinate_debit: CoordinateTokenDebit

    @property
    def receipt(self) -> PublicGenerationReceipt:
        """Compatibility name for callers of the first draft."""

        return self.public_receipt


class AttestedLoopbackService:
    """Exact production service boundary with an explicit test-only path."""

    def __init__(
        self,
        token: object,
        profile: ModelServiceProfile,
        transport: Transport,
        trusted: TrustedLaunchReceipt,
        attestation: ServiceAttestation,
        *,
        test_only: bool,
    ) -> None:
        if token is not _CONSTRUCTION_TOKEN:
            raise RecommissioningClientError("use connect() or explicit for_test() construction")
        self.profile = profile
        self._transport = transport
        self._trusted = trusted
        self.attestation = attestation
        self._test_only = test_only

    @classmethod
    def connect(
        cls,
        endpoint: str,
        profile: ModelServiceProfile,
        trusted_launch_receipt: TrustedLaunchReceipt,
        *,
        attestation_timeout_seconds: float = 10.0,
    ) -> "AttestedLoopbackService":
        profile.validate()
        if trusted_launch_receipt.test_only:
            raise RecommissioningClientError("production connection rejects a test-only launch receipt")
        transport = UrllibLoopbackTransport(endpoint)
        attestation = _observe_service(
            transport,
            profile,
            trusted_launch_receipt,
            timeout_seconds=attestation_timeout_seconds,
        )
        return cls(
            _CONSTRUCTION_TOKEN,
            profile,
            transport,
            trusted_launch_receipt,
            attestation,
            test_only=False,
        )

    @classmethod
    def for_test(
        cls,
        profile: ModelServiceProfile,
        transport: Transport,
        trusted_launch_receipt: TrustedLaunchReceipt,
    ) -> "AttestedLoopbackService":
        profile.validate()
        if not trusted_launch_receipt.test_only or not callable(transport):
            raise RecommissioningClientError("test service requires explicit test-only launch authority and transport")
        attestation = _observe_service(
            transport,
            profile,
            trusted_launch_receipt,
            timeout_seconds=1.0,
        )
        return cls(
            _CONSTRUCTION_TOKEN,
            profile,
            transport,
            trusted_launch_receipt,
            attestation,
            test_only=True,
        )

    def attest(self) -> Dict[str, Any]:
        """Return the endpoint-free public attestation record."""

        return self.attestation.public_record()

    def build_request(
        self,
        envelope: PromptEnvelope,
        budget: TokenBudget,
        tokenization: TokenizationPreflight,
        ledger: CoordinateTokenLedger,
    ) -> GenerationRequest:
        if not self._test_only and not getattr(ledger, "durable", False):
            raise RecommissioningClientError("production generation requires a durable coordinate token ledger")
        if not self._test_only and envelope.task_adapter_kind != "LEARNABLE_V2":
            raise RecommissioningClientError("production generation requires the exact learnable-v2 task adapter")
        state = ledger.snapshot(envelope.coordinate_id, budget)
        return GenerationRequest.build(
            envelope,
            budget,
            self.attestation,
            tokenization,
            state,
            profile=self.profile,
        )

    def _revalidate_service(self) -> None:
        fresh = _observe_service(
            self._transport,
            self.profile,
            self._trusted,
            timeout_seconds=10.0,
        )
        if fresh.attestation_digest != self.attestation.attestation_digest:
            raise RecommissioningClientError("service observation changed after trusted attestation")

    def generate(
        self,
        request: GenerationRequest,
        *,
        ledger: CoordinateTokenLedger,
        signer: ReceiptSigner,
    ) -> GenerationResult:
        if not isinstance(request, GenerationRequest):
            raise RecommissioningClientError("generation consumes a typed GenerationRequest only")
        if not self._test_only and not getattr(ledger, "durable", False):
            raise RecommissioningClientError("production generation requires a durable coordinate token ledger")
        if not self._test_only and request.envelope.task_adapter_kind != "LEARNABLE_V2":
            raise RecommissioningClientError("production generation requires the exact learnable-v2 task adapter")
        request.validate(profile=self.profile, attestation=self.attestation)
        self._revalidate_service()
        coordinate_debit = ledger.debit(request)
        wire = request.wire_request(self.profile)
        request_bytes = canonical_json(wire).encode("utf-8")
        if digest_for(wire) != request.wire_request_digest:
            raise RecommissioningClientError("wire request changed after immutable construction")
        started_ns = time.monotonic_ns()
        status, response_bytes = self._transport(
            "POST",
            "/v1/chat/completions",
            request_bytes,
            request.budget.hard_timeout_ms / 1000.0,
        )
        elapsed_ms = max(0, (time.monotonic_ns() - started_ns + 999999) // 1000000)
        if elapsed_ms > request.budget.hard_timeout_ms:
            raise RecommissioningClientError("generation exceeded the hard elapsed-time budget")
        if status != 200:
            raise RecommissioningClientError("chat-completions endpoint returned HTTP {}".format(status))
        response = _strict_json_loads(response_bytes, "chat-completions endpoint")
        if not isinstance(response, Mapping) or response.get("model") != self.profile.model_id:
            raise RecommissioningClientError("completion response model differs from the frozen profile")
        choices = response.get("choices")
        usage = response.get("usage")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
            raise RecommissioningClientError("completion response does not contain exactly one choice")
        choice = choices[0]
        message = choice.get("message")
        finish_reason = choice.get("finish_reason")
        if not isinstance(message, Mapping):
            raise RecommissioningClientError("completion message is not an object")
        if "tool_calls" in message or "function_call" in message:
            raise RecommissioningClientError("completion attempted a forbidden tool call")
        if set(message) - {"role", "content", "reasoning_content"}:
            raise RecommissioningClientError("completion message contains an unexpected field")
        if message.get("role") != "assistant" or not isinstance(message.get("content"), str):
            raise RecommissioningClientError("completion message is not a typed assistant response")
        reasoning = message.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise RecommissioningClientError("private reasoning content has an unexpected type")
        if finish_reason != "stop":
            raise RecommissioningClientError("completion did not terminate with the frozen stop disposition")
        candidate_value = _strict_json_loads(message["content"].encode("utf-8"), "candidate content")
        candidate = TypedCandidateResponse.parse(
            candidate_value,
            visible_evidence_ids=request.envelope.visible_evidence_ids,
            public_locus=request.envelope.public_locus,
        )
        if not isinstance(usage, Mapping) or set(usage) < {"prompt_tokens", "completion_tokens"}:
            raise RecommissioningClientError("completion response omitted token accounting")
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if type(prompt_tokens) is not int or prompt_tokens != request.tokenization.prompt_tokens:
            raise RecommissioningClientError("endpoint prompt-token count differs from pinned preflight")
        if type(completion_tokens) is not int or not 1 <= completion_tokens <= request.budget.max_completion_tokens:
            raise RecommissioningClientError("completion-token accounting violates the budget")
        total_tokens = prompt_tokens + completion_tokens
        if total_tokens > request.budget.max_total_tokens:
            raise RecommissioningClientError("endpoint total-token accounting violates the budget")
        private_record = PrivateGenerationRecord.build(
            request_bytes=request_bytes,
            response_bytes=response_bytes,
            candidate=candidate,
            attestation=self.attestation,
        )
        public_candidate = candidate.public_redacted_record()
        receipt_core = {
            "schema_version": "egv-recommissioning-public-generation-receipt-v2",
            "coordinate_id": request.envelope.coordinate_id,
            "model_bundle": request.envelope.model_bundle,
            "task_id": request.envelope.task_id,
            "public_locus": request.envelope.public_locus,
            "task_adapter_kind": request.envelope.task_adapter_kind,
            "task_package_digest": request.envelope.task_package_digest,
            "model_profile_digest": self.profile.digest,
            "budget_digest": request.budget.digest,
            "seed": request.budget.seed,
            "attempt_index": request.envelope.attempt_index,
            "prompt_digest": request.envelope.prompt_digest,
            "wire_request_digest": request.wire_request_digest,
            "request_digest": request.request_digest,
            "service_attestation_digest": self.attestation.attestation_digest,
            "runtime_config_digest": self.attestation.runtime_config_digest,
            "launch_receipt_digest": self.attestation.launch_receipt_digest,
            "tokenization_preflight_digest": request.tokenization.preflight_digest,
            "rendered_prompt_digest": request.tokenization.rendered_prompt_digest,
            "response_digest": digest_bytes(response_bytes),
            "candidate_digest": digest_for(public_candidate),
            "source_digest": candidate.source_digest,
            "finish_reason": finish_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "max_prompt_tokens": request.budget.max_prompt_tokens,
            "max_completion_tokens": request.budget.max_completion_tokens,
            "max_total_tokens": request.budget.max_total_tokens,
            "max_coordinate_tokens": request.budget.max_coordinate_tokens,
            "hard_timeout_ms": request.budget.hard_timeout_ms,
            "elapsed_ms": elapsed_ms,
            "coordinate_debit_digest": coordinate_debit.debit_digest,
            "coordinate_reserved_tokens": coordinate_debit.reserved_tokens,
            "coordinate_cumulative_reserved_tokens": coordinate_debit.cumulative_reserved_tokens,
            "coordinate_state_digest": coordinate_debit.state_digest,
        }
        public_receipt = PublicGenerationReceipt.build(receipt_core, signer)
        return GenerationResult(candidate, private_record, public_receipt, coordinate_debit)


# The old draft name now resolves to the hardened boundary. Its constructor is
# intentionally unusable; callers must select connect() or explicit for_test().
BoundedModelClient = AttestedLoopbackService


__all__ = [
    "AttestedLoopbackService",
    "BoundedModelClient",
    "CoordinateTokenDebit",
    "CoordinateTokenLedger",
    "CoordinateTokenState",
    "FileCoordinateTokenLedger",
    "GenerationReceipt",
    "GenerationBudget",
    "GenerationRequest",
    "GenerationResult",
    "InMemoryCoordinateTokenLedger",
    "ModelServiceProfile",
    "PrivateGenerationRecord",
    "PublicGenerationReceipt",
    "RecommissioningClientError",
    "ServiceAttestation",
    "TokenizationPreflight",
    "TrustedLaunchReceipt",
    "TypedCandidateResponse",
    "UrllibLoopbackTransport",
    "candidate_response_schema",
]
