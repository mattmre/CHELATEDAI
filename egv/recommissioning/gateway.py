"""Loopback-only attestation gateway for production recommissioning services.

The controller launch issuer and the live service signer are deliberately
separate.  Neither path creates signing keys, and the public records contain
only content-derived identities and digests.  Private endpoint and file
locations remain inputs to the deployment process, never fields in a launch or
service-observation record.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import stat
import threading
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from ..canonical import canonical_json, canonical_value, digest_for
from ..receipts import ReceiptSigner, key_id_for_public_key, load_public_key, public_key_bytes
from .client import ModelServiceProfile, ServiceAttestation, TrustedLaunchReceipt


class RecommissioningGatewayError(RuntimeError):
    """The controller issuer or live gateway failed a closed trust boundary."""


_DIGEST = re.compile(r"[0-9a-f]{64}")
_NONCE = re.compile(r"nonce_[0-9a-f]{64}")
_LAUNCH_SCHEMA = "egv-trusted-launch-receipt-v2"
_OBSERVATION_SCHEMA = "egv-recommissioning-service-observation-v1"
_CONFIG_SCHEMA = "egv-recommissioning-gateway-config-v1"
_RUNTIME_SCHEMA = "egv-recommissioning-gateway-runtime-v1"
_MAX_CONFIG_BYTES = 128 * 1024
_MAX_PUBLIC_RECORD_BYTES = 256 * 1024
_MAX_KEY_BYTES = 8 * 1024
_MAX_HTTP_BODY = 4 * 1024 * 1024
_MAX_JSON_DEPTH = 16
_MAX_JSON_NODES = 8192
_ERROR_BODY = canonical_json({"error": "gateway_fail_closed"}).encode("utf-8")
_NOT_FOUND_BODY = canonical_json({"error": "route_not_found"}).encode("utf-8")
_METHOD_BODY = canonical_json({"error": "method_not_allowed"}).encode("utf-8")


def _require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise RecommissioningGatewayError("{} must be a full lowercase SHA-256 digest".format(label))
    return value


def _require_nonce(value: Any) -> str:
    if not isinstance(value, str) or _NONCE.fullmatch(value) is None:
        raise RecommissioningGatewayError("service nonce must be an externally provisioned opaque 256-bit value")
    return value


def _public_identity(value: Any, label: str) -> Tuple[bytes, str]:
    try:
        raw = public_key_bytes(value)
        return raw, key_id_for_public_key(raw)
    except Exception as exc:
        raise RecommissioningGatewayError("{} is not an Ed25519 public key".format(label)) from exc


def _signer_identity(signer: Any, label: str) -> Tuple[bytes, str]:
    if signer is None or not callable(getattr(signer, "sign_bytes", None)):
        raise RecommissioningGatewayError("{} is not an externally supplied signer".format(label))
    try:
        raw, derived = _public_identity(signer.public_key, label + " public key")
        supplied = signer.key_id
    except AttributeError as exc:
        raise RecommissioningGatewayError("{} does not expose its public identity".format(label)) from exc
    if supplied != derived:
        raise RecommissioningGatewayError("{} key ID is not content-derived".format(label))
    return raw, derived


def issue_trusted_launch_receipt(
    *,
    profile: ModelServiceProfile,
    runtime_config_digest: str,
    service_nonce: str,
    controller_signer: Any,
    service_public_key: Any,
    generation_receipt_public_key: Any,
) -> Dict[str, Any]:
    """Issue the exact controller-signed launch record without creating keys."""

    if not isinstance(profile, ModelServiceProfile):
        raise RecommissioningGatewayError("trusted launch requires a typed model-service profile")
    profile.validate()
    runtime_digest = _require_digest(runtime_config_digest, "runtime-config digest")
    nonce = _require_nonce(service_nonce)
    controller_raw, controller_key_id = _signer_identity(controller_signer, "controller signer")
    service_raw, service_key_id = _public_identity(service_public_key, "service public key")
    generation_raw, generation_key_id = _public_identity(
        generation_receipt_public_key,
        "generation-receipt public key",
    )
    if len({controller_raw, service_raw, generation_raw}) != 3:
        raise RecommissioningGatewayError("controller, service, and generation-receipt keys must be distinct")
    unsigned = {
        "schema_version": _LAUNCH_SCHEMA,
        "model_profile_digest": profile.digest,
        "runtime_config_digest": runtime_digest,
        "service_nonce": nonce,
        "service_public_key": service_raw.hex(),
        "service_key_id": service_key_id,
        "generation_receipt_public_key": generation_raw.hex(),
        "generation_receipt_key_id": generation_key_id,
        "controller_key_id": controller_key_id,
    }
    signable = {**unsigned, "launch_receipt_digest": digest_for(unsigned)}
    signature = controller_signer.sign_bytes(canonical_json(signable).encode("utf-8"))
    record = {**signable, "signature": signature}
    try:
        trusted = TrustedLaunchReceipt.from_controller_record(record, controller_public_key=controller_raw)
        return trusted.public_record()
    except Exception as exc:
        raise RecommissioningGatewayError("controller signer did not produce a valid trusted launch receipt") from exc


def sign_service_observation(
    *,
    profile: ModelServiceProfile,
    trusted_launch_receipt: TrustedLaunchReceipt,
    service_signer: Any,
) -> Dict[str, Any]:
    """Sign the live service observation rooted in one exact controller launch."""

    if not isinstance(profile, ModelServiceProfile) or not isinstance(
        trusted_launch_receipt,
        TrustedLaunchReceipt,
    ):
        raise RecommissioningGatewayError("service observation requires typed immutable launch inputs")
    profile.validate()
    trusted_launch_receipt.validate()
    if trusted_launch_receipt.test_only:
        raise RecommissioningGatewayError("production service observations reject test-only launch authority")
    service_raw, service_key_id = _signer_identity(service_signer, "service signer")
    if (
        service_raw != trusted_launch_receipt.service_public_key_raw
        or service_key_id != trusted_launch_receipt.service_key_id
    ):
        raise RecommissioningGatewayError("service signer differs from the controller-pinned service key")
    if profile.digest != trusted_launch_receipt.model_profile_digest:
        raise RecommissioningGatewayError("service profile differs from the controller-pinned model profile")
    unsigned = {
        "schema_version": _OBSERVATION_SCHEMA,
        "model_profile": profile.public_manifest(),
        "model_profile_digest": profile.digest,
        "runtime_config_digest": trusted_launch_receipt.runtime_config_digest,
        "launch_receipt_digest": trusted_launch_receipt.launch_receipt_digest,
        "service_nonce": trusted_launch_receipt.service_nonce,
        "service_key_id": service_key_id,
    }
    signable = {**unsigned, "observation_digest": digest_for(unsigned)}
    record = {
        **signable,
        "signature": service_signer.sign_bytes(canonical_json(signable).encode("utf-8")),
    }
    try:
        return ServiceAttestation._validate_service_observation(  # type: ignore[attr-defined]
            record,
            trusted=trusted_launch_receipt,
            expected_profile=profile,
        )
    except Exception as exc:
        raise RecommissioningGatewayError("service signer did not produce a valid runtime observation") from exc


def _normalize_loopback_origin(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) > 256:
        raise RecommissioningGatewayError("{} must be a bounded HTTP origin".format(label))
    parsed = urllib_parse.urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or parsed.port is None
        or not 1 <= parsed.port <= 65535
    ):
        raise RecommissioningGatewayError("{} must be an explicit numeric loopback-only HTTP origin".format(label))
    normalized = "http://{}:{}".format("[::1]" if parsed.hostname == "::1" else "127.0.0.1", parsed.port)
    if value != normalized:
        raise RecommissioningGatewayError("{} must use its exact normalized loopback origin".format(label))
    return normalized


def _strict_json_bytes(value: bytes, label: str, *, byte_ceiling: int) -> Any:
    if not isinstance(value, bytes) or len(value) > byte_ceiling:
        raise RecommissioningGatewayError("{} exceeds its byte ceiling".format(label))

    def closed_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise RecommissioningGatewayError("{} contains duplicate JSON keys".format(label))
            result[key] = item
        return result

    def reject_constant(_value: str) -> None:
        raise RecommissioningGatewayError("{} contains a non-finite number".format(label))

    try:
        text = value.decode("utf-8", errors="strict")
        parsed = json.loads(text, object_pairs_hook=closed_pairs, parse_constant=reject_constant)
    except RecommissioningGatewayError:
        raise
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise RecommissioningGatewayError("{} is not strict UTF-8 JSON".format(label)) from exc
    nodes = 0
    pending = [(parsed, 0)]
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise RecommissioningGatewayError("{} exceeds the JSON node ceiling".format(label))
        if depth > _MAX_JSON_DEPTH:
            raise RecommissioningGatewayError("{} exceeds the JSON depth ceiling".format(label))
        if item is None or type(item) in {bool, int, str}:
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                raise RecommissioningGatewayError("{} contains a non-finite number".format(label))
            continue
        if isinstance(item, Mapping):
            pending.extend((child, depth + 1) for child in item.values())
            continue
        if isinstance(item, (list, tuple)):
            pending.extend((child, depth + 1) for child in item)
            continue
        raise RecommissioningGatewayError("{} contains a non-JSON value".format(label))
    try:
        return canonical_value(parsed)
    except Exception as exc:
        raise RecommissioningGatewayError("{} cannot be canonicalized".format(label)) from exc


@dataclass(frozen=True)
class GatewayConfig:
    """Private deployment configuration whose contents never enter public records."""

    bind_host: str
    bind_port: int
    upstream_origin: str
    upstream_timeout_ms: int
    max_request_bytes: int
    max_response_bytes: int
    max_concurrency: int
    service_nonce: str
    profile: ModelServiceProfile

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "GatewayConfig":
        fields = {
            "schema_version",
            "bind_host",
            "bind_port",
            "upstream_origin",
            "upstream_timeout_ms",
            "max_request_bytes",
            "max_response_bytes",
            "max_concurrency",
            "service_nonce",
            "model_profile",
        }
        if not isinstance(record, Mapping) or set(record) != fields:
            raise RecommissioningGatewayError("gateway configuration is not a closed record")
        if record.get("schema_version") != _CONFIG_SCHEMA:
            raise RecommissioningGatewayError("gateway configuration schema is unsupported")
        bind_host = record["bind_host"]
        if bind_host not in {"127.0.0.1", "::1"}:
            raise RecommissioningGatewayError("gateway bind host must be numeric loopback")
        bind_port = record["bind_port"]
        if type(bind_port) is not int or not 1 <= bind_port <= 65535:
            raise RecommissioningGatewayError("gateway bind port is outside the bounded range")
        upstream_origin = _normalize_loopback_origin(record["upstream_origin"], "upstream origin")
        integer_limits = {
            "upstream_timeout_ms": (100, 300_000),
            "max_request_bytes": (1024, _MAX_HTTP_BODY),
            "max_response_bytes": (1024, _MAX_HTTP_BODY),
            "max_concurrency": (1, 64),
        }
        normalized_limits: Dict[str, int] = {}
        for field_name, bounds in integer_limits.items():
            current = record[field_name]
            if type(current) is not int or not bounds[0] <= current <= bounds[1]:
                raise RecommissioningGatewayError("{} is outside the bounded range".format(field_name))
            normalized_limits[field_name] = current
        try:
            profile = ModelServiceProfile.from_public_manifest(record["model_profile"])
        except Exception as exc:
            raise RecommissioningGatewayError("gateway model profile is invalid") from exc
        result = cls(
            bind_host=bind_host,
            bind_port=bind_port,
            upstream_origin=upstream_origin,
            service_nonce=_require_nonce(record["service_nonce"]),
            profile=profile,
            **normalized_limits,
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.bind_host not in {"127.0.0.1", "::1"}:
            raise RecommissioningGatewayError("gateway bind host must be numeric loopback")
        if type(self.bind_port) is not int or not 1 <= self.bind_port <= 65535:
            raise RecommissioningGatewayError("gateway bind port is outside the bounded range")
        _normalize_loopback_origin(self.upstream_origin, "upstream origin")
        limits = (
            (self.upstream_timeout_ms, 100, 300_000, "upstream_timeout_ms"),
            (self.max_request_bytes, 1024, _MAX_HTTP_BODY, "max_request_bytes"),
            (self.max_response_bytes, 1024, _MAX_HTTP_BODY, "max_response_bytes"),
            (self.max_concurrency, 1, 64, "max_concurrency"),
        )
        for value, minimum, maximum, label in limits:
            if type(value) is not int or not minimum <= value <= maximum:
                raise RecommissioningGatewayError("{} is outside the bounded range".format(label))
        _require_nonce(self.service_nonce)
        if not isinstance(self.profile, ModelServiceProfile):
            raise RecommissioningGatewayError("gateway model profile is not typed")
        self.profile.validate()

    def private_record(self) -> Dict[str, Any]:
        return {
            "schema_version": _CONFIG_SCHEMA,
            "bind_host": self.bind_host,
            "bind_port": self.bind_port,
            "upstream_origin": self.upstream_origin,
            "upstream_timeout_ms": self.upstream_timeout_ms,
            "max_request_bytes": self.max_request_bytes,
            "max_response_bytes": self.max_response_bytes,
            "max_concurrency": self.max_concurrency,
            "service_nonce": self.service_nonce,
            "model_profile": self.profile.public_manifest(),
        }

    def private_runtime_record(self) -> Dict[str, Any]:
        """Return the private runtime preimage; publish only its digest."""

        return {
            "schema_version": _RUNTIME_SCHEMA,
            "bind_host": self.bind_host,
            "bind_port": self.bind_port,
            "upstream_origin": self.upstream_origin,
            "upstream_timeout_ms": self.upstream_timeout_ms,
            "max_request_bytes": self.max_request_bytes,
            "max_response_bytes": self.max_response_bytes,
            "max_concurrency": self.max_concurrency,
            "model_profile_digest": self.profile.digest,
        }

    @property
    def runtime_config_digest(self) -> str:
        self.validate()
        return digest_for(self.private_runtime_record())


class _RejectRedirectHandler(urllib_request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


class _UpstreamClient:
    def __init__(self, origin: str, *, timeout_seconds: float, max_response_bytes: int) -> None:
        self._origin = _normalize_loopback_origin(origin, "upstream origin")
        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes
        self._opener = urllib_request.build_opener(urllib_request.ProxyHandler({}), _RejectRedirectHandler())

    def request(self, method: str, path: str, body: Optional[bytes] = None) -> bytes:
        allowed = {
            ("GET", "/health"),
            ("GET", "/v1/models"),
            ("POST", "/v1/chat/completions"),
        }
        if (method, path) not in allowed:
            raise RecommissioningGatewayError("gateway attempted an unapproved upstream route")
        headers = {"Accept": "application/json", "Connection": "close"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib_request.Request(self._origin + path, data=body, headers=headers, method=method)
        try:
            with self._opener.open(request, timeout=self._timeout_seconds) as response:
                if int(response.status) != 200:
                    raise RecommissioningGatewayError("upstream returned a non-success status")
                payload = response.read(self._max_response_bytes + 1)
        except RecommissioningGatewayError:
            raise
        except (urllib_error.HTTPError, urllib_error.URLError, OSError, TimeoutError) as exc:
            raise RecommissioningGatewayError("bounded upstream request failed") from exc
        if len(payload) > self._max_response_bytes:
            raise RecommissioningGatewayError("upstream response exceeds the configured byte ceiling")
        return payload


class LoopbackAttestationGateway:
    """Frozen upstream observer and fail-closed chat proxy."""

    def __init__(
        self,
        *,
        profile: ModelServiceProfile,
        trusted_launch_receipt: TrustedLaunchReceipt,
        service_signer: Any,
        runtime_config_digest: str,
        service_nonce: str,
        upstream_origin: str,
        upstream_timeout_seconds: float,
        max_request_bytes: int,
        max_response_bytes: int,
        max_concurrency: int,
    ) -> None:
        if not isinstance(profile, ModelServiceProfile) or not isinstance(
            trusted_launch_receipt,
            TrustedLaunchReceipt,
        ):
            raise RecommissioningGatewayError("gateway requires typed profile and launch authority")
        profile.validate()
        trusted_launch_receipt.validate()
        if trusted_launch_receipt.test_only:
            raise RecommissioningGatewayError("live gateway rejects test-only launch authority")
        if profile.digest != trusted_launch_receipt.model_profile_digest:
            raise RecommissioningGatewayError("gateway profile differs from the controller launch")
        if (
            _require_digest(runtime_config_digest, "runtime-config digest")
            != trusted_launch_receipt.runtime_config_digest
        ):
            raise RecommissioningGatewayError("gateway runtime configuration differs from the controller launch")
        if _require_nonce(service_nonce) != trusted_launch_receipt.service_nonce:
            raise RecommissioningGatewayError("gateway service nonce differs from the controller launch")
        if type(max_request_bytes) is not int or not 1 <= max_request_bytes <= _MAX_HTTP_BODY:
            raise RecommissioningGatewayError("gateway request ceiling is outside the bounded range")
        if type(max_response_bytes) is not int or not 1 <= max_response_bytes <= _MAX_HTTP_BODY:
            raise RecommissioningGatewayError("gateway response ceiling is outside the bounded range")
        if type(max_concurrency) is not int or not 1 <= max_concurrency <= 64:
            raise RecommissioningGatewayError("gateway concurrency ceiling is outside the bounded range")
        if not isinstance(upstream_timeout_seconds, (int, float)) or isinstance(upstream_timeout_seconds, bool):
            raise RecommissioningGatewayError("gateway upstream timeout is invalid")
        if not 0.1 <= float(upstream_timeout_seconds) <= 300.0:
            raise RecommissioningGatewayError("gateway upstream timeout is outside the bounded range")
        self.profile = profile
        self.trusted_launch_receipt = trusted_launch_receipt
        self.max_request_bytes = max_request_bytes
        self.max_response_bytes = max_response_bytes
        self.request_timeout_seconds = float(upstream_timeout_seconds)
        self._observation = sign_service_observation(
            profile=profile,
            trusted_launch_receipt=trusted_launch_receipt,
            service_signer=service_signer,
        )
        self._observation_body = canonical_json(self._observation).encode("utf-8")
        if len(self._observation_body) > max_response_bytes:
            raise RecommissioningGatewayError("signed service observation exceeds the response byte ceiling")
        self._upstream = _UpstreamClient(
            upstream_origin,
            timeout_seconds=self.request_timeout_seconds,
            max_response_bytes=max_response_bytes,
        )
        self._drift_lock = threading.RLock()
        self._latched_reason: Optional[str] = None
        self._capacity = threading.BoundedSemaphore(max_concurrency)
        initial_snapshot = self._read_and_validate_snapshot()
        confirming_snapshot = self._read_and_validate_snapshot()
        if initial_snapshot != confirming_snapshot:
            raise RecommissioningGatewayError("upstream snapshot was not stable during gateway startup")
        self._health_body, self._models_body = initial_snapshot

    def _validate_models(self, body: bytes) -> None:
        value = _strict_json_bytes(body, "upstream model-list response", byte_ceiling=self.max_response_bytes)
        records = value.get("data") if isinstance(value, Mapping) else None
        if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], Mapping):
            raise RecommissioningGatewayError("upstream model-list response does not identify exactly one model")
        if records[0].get("id") != self.profile.model_id:
            raise RecommissioningGatewayError("upstream model identity differs from the controller profile")

    def _read_and_validate_snapshot(self) -> Tuple[bytes, bytes]:
        health = self._upstream.request("GET", "/health")
        models = self._upstream.request("GET", "/v1/models")
        self._validate_models(models)
        return health, models

    def _latch(self, reason: str) -> None:
        with self._drift_lock:
            if self._latched_reason is None:
                self._latched_reason = reason

    @property
    def failed_closed(self) -> bool:
        with self._drift_lock:
            return self._latched_reason is not None

    def _verify_snapshot(self) -> None:
        with self._drift_lock:
            if self._latched_reason is not None:
                raise RecommissioningGatewayError("gateway is permanently failed closed")
            try:
                health, models = self._read_and_validate_snapshot()
                if health != self._health_body or models != self._models_body:
                    raise RecommissioningGatewayError("upstream immutable snapshot drifted")
            except Exception as exc:
                self._latch("immutable upstream snapshot verification failed")
                raise RecommissioningGatewayError("upstream immutable snapshot verification failed") from exc

    def acquire_capacity(self) -> bool:
        return self._capacity.acquire(blocking=False)

    def release_capacity(self) -> None:
        self._capacity.release()

    def public_body(self, path: str) -> bytes:
        if path not in {"/health", "/v1/models", "/v1/egv/attestation"}:
            raise RecommissioningGatewayError("gateway public route is not closed")
        self._verify_snapshot()
        if path == "/health":
            return self._health_body
        if path == "/v1/models":
            return self._models_body
        return self._observation_body

    def proxy_chat(self, body: bytes) -> bytes:
        if not isinstance(body, bytes) or not 1 <= len(body) <= self.max_request_bytes:
            raise RecommissioningGatewayError("chat request is outside the configured byte ceiling")
        request_value = _strict_json_bytes(body, "chat request", byte_ceiling=self.max_request_bytes)
        if not isinstance(request_value, Mapping) or request_value.get("model") != self.profile.model_id:
            raise RecommissioningGatewayError("chat request differs from the controller model profile")
        if canonical_json(request_value).encode("utf-8") != body:
            raise RecommissioningGatewayError("chat request is not canonical JSON")
        self._verify_snapshot()
        response = self._upstream.request("POST", "/v1/chat/completions", body)
        self._verify_snapshot()
        return response

    def latch_internal_failure(self) -> None:
        self._latch("gateway internal failure")


class _GatewayRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = ""
    sys_version = ""

    @property
    def gateway(self) -> LoopbackAttestationGateway:
        return self.server.gateway  # type: ignore[attr-defined,no-any-return]

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.gateway.request_timeout_seconds)

    def log_message(self, format, *args):  # type: ignore[no-untyped-def]
        return

    def _reply(self, status: int, payload: bytes) -> None:
        self.send_response_only(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)
        self.close_connection = True

    def _request_is_loopback(self) -> bool:
        try:
            peer = ipaddress.ip_address(str(self.client_address[0]).split("%")[0])
        except ValueError:
            return False
        if not peer.is_loopback:
            return False
        expected_hosts = self.server.expected_host_headers  # type: ignore[attr-defined]
        hosts = self.headers.get_all("Host", failobj=[])
        return len(hosts) == 1 and hosts[0] in expected_hosts

    def _run_bounded(self, operation) -> None:  # type: ignore[no-untyped-def]
        if not self._request_is_loopback():
            self._reply(403, _ERROR_BODY)
            return
        if not self.gateway.acquire_capacity():
            self._reply(503, _ERROR_BODY)
            return
        try:
            operation()
        except RecommissioningGatewayError:
            self._reply(503, _ERROR_BODY)
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            self.close_connection = True
        except Exception:
            self.gateway.latch_internal_failure()
            self._reply(503, _ERROR_BODY)
        finally:
            self.gateway.release_capacity()

    def do_GET(self) -> None:  # noqa: N802
        def operation() -> None:
            if self.headers.get("Transfer-Encoding") is not None or self.headers.get("Content-Length") is not None:
                self._reply(400, _ERROR_BODY)
                return
            if self.path not in {"/health", "/v1/models", "/v1/egv/attestation"}:
                self._reply(404, _NOT_FOUND_BODY)
                return
            self._reply(200, self.gateway.public_body(self.path))

        self._run_bounded(operation)

    def do_POST(self) -> None:  # noqa: N802
        def operation() -> None:
            if self.path != "/v1/chat/completions":
                self._reply(404, _NOT_FOUND_BODY)
                return
            if self.headers.get("Transfer-Encoding") is not None or self.headers.get("Expect") is not None:
                self._reply(400, _ERROR_BODY)
                return
            if self.headers.get("Content-Encoding") is not None:
                self._reply(400, _ERROR_BODY)
                return
            content_types = self.headers.get_all("Content-Type", failobj=[])
            if content_types != ["application/json"]:
                self._reply(415, _ERROR_BODY)
                return
            lengths = self.headers.get_all("Content-Length", failobj=[])
            if len(lengths) != 1 or not lengths[0].isdigit():
                self._reply(411, _ERROR_BODY)
                return
            length = int(lengths[0])
            if not 1 <= length <= self.gateway.max_request_bytes:
                self._reply(413, _ERROR_BODY)
                return
            body = self.rfile.read(length)
            if len(body) != length:
                self._reply(400, _ERROR_BODY)
                return
            self._reply(200, self.gateway.proxy_chat(body))

        self._run_bounded(operation)

    def _method_not_allowed(self) -> None:
        self._run_bounded(lambda: self._reply(405, _METHOD_BODY))

    do_HEAD = _method_not_allowed
    do_PUT = _method_not_allowed
    do_DELETE = _method_not_allowed
    do_PATCH = _method_not_allowed
    do_OPTIONS = _method_not_allowed
    do_CONNECT = _method_not_allowed
    do_TRACE = _method_not_allowed


class _GatewayHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = False
    daemon_threads = True
    request_queue_size = 32

    def __init__(
        self,
        server_address: Tuple[str, int],
        gateway: LoopbackAttestationGateway,
        *,
        bind_and_activate: bool = True,
    ) -> None:
        self.gateway = gateway
        super().__init__(server_address, _GatewayRequestHandler, bind_and_activate=bind_and_activate)
        host, port = self.server_address[:2]
        host_literal = "[::1]" if host == "::1" else host
        self.expected_host_headers = {"{}:{}".format(host_literal, port)}
        if port == 80:
            self.expected_host_headers.add(host_literal)

    def handle_error(self, request, client_address):  # type: ignore[no-untyped-def]
        self.gateway.latch_internal_failure()


class _GatewayHTTPServerV6(_GatewayHTTPServer):
    address_family = socket.AF_INET6


def build_gateway_server(
    gateway: LoopbackAttestationGateway,
    *,
    bind_host: str,
    bind_port: int,
    allow_ephemeral_port: bool = False,
) -> ThreadingHTTPServer:
    """Bind the gateway only to an explicit numeric loopback address."""

    if bind_host not in {"127.0.0.1", "::1"}:
        raise RecommissioningGatewayError("gateway server may bind only to numeric loopback")
    if not isinstance(gateway, LoopbackAttestationGateway):
        raise RecommissioningGatewayError("gateway server requires a typed attestation gateway")
    if type(bind_port) is not int or not (0 if allow_ephemeral_port else 1) <= bind_port <= 65535:
        raise RecommissioningGatewayError("gateway server port is outside the bounded range")
    server_type = _GatewayHTTPServerV6 if bind_host == "::1" else _GatewayHTTPServer
    try:
        return server_type((bind_host, bind_port), gateway)
    except OSError as exc:
        raise RecommissioningGatewayError("gateway loopback bind failed") from exc


def _read_file(path_value: str, *, byte_ceiling: int, label: str, private: bool = False) -> bytes:
    path = Path(path_value)
    if path.is_symlink():
        raise RecommissioningGatewayError("{} may not be a symlink".format(label))
    flags = os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOINHERIT", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise RecommissioningGatewayError("{} is unavailable".format(label)) from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > byte_ceiling:
            raise RecommissioningGatewayError("{} is not a bounded regular file".format(label))
        if private and os.name != "nt" and info.st_mode & 0o077:
            raise RecommissioningGatewayError("{} permissions expose private key material".format(label))
        chunks = []
        remaining = byte_ceiling + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        result = b"".join(chunks)
        if len(result) > byte_ceiling:
            raise RecommissioningGatewayError("{} exceeds its byte ceiling".format(label))
        return result
    except RecommissioningGatewayError:
        raise
    except OSError as exc:
        raise RecommissioningGatewayError("{} could not be read".format(label)) from exc
    finally:
        os.close(descriptor)


def _load_mapping_file(path_value: str, *, label: str, byte_ceiling: int) -> Dict[str, Any]:
    value = _strict_json_bytes(
        _read_file(path_value, byte_ceiling=byte_ceiling, label=label),
        label,
        byte_ceiling=byte_ceiling,
    )
    if not isinstance(value, Mapping):
        raise RecommissioningGatewayError("{} must contain one JSON object".format(label))
    return dict(value)


def _load_private_signer(path_value: str, label: str) -> ReceiptSigner:
    raw = _read_file(path_value, byte_ceiling=32, label=label, private=True)
    if len(raw) != 32:
        raise RecommissioningGatewayError("{} must contain exactly 32 raw Ed25519 bytes".format(label))
    try:
        return ReceiptSigner(raw)
    except Exception as exc:
        raise RecommissioningGatewayError("{} is not a valid Ed25519 private key".format(label)) from exc


def _load_public_key_file(path_value: str, label: str) -> Any:
    raw = _read_file(path_value, byte_ceiling=_MAX_KEY_BYTES, label=label)
    try:
        return load_public_key(raw)
    except Exception as exc:
        raise RecommissioningGatewayError("{} is not a valid Ed25519 public key".format(label)) from exc


def _write_new_atomic(path_value: str, payload: bytes) -> None:
    target = Path(path_value)
    parent = target.parent
    if parent.is_symlink() or not parent.is_dir() or target.name in {"", ".", ".."}:
        raise RecommissioningGatewayError("public launch output parent is not a stable directory")
    if target.exists() or target.is_symlink():
        raise RecommissioningGatewayError("public launch output already exists")
    temporary = parent / ".{}.{}.tmp".format(target.name, secrets.token_hex(12))
    descriptor: Optional[int] = None
    temporary_created = False
    try:
        descriptor = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        temporary_created = True
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(str(temporary), str(target))
        if os.name != "nt":
            directory_descriptor = os.open(str(parent), os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    except FileExistsError as exc:
        raise RecommissioningGatewayError("public launch output already exists") from exc
    except OSError as exc:
        raise RecommissioningGatewayError("public launch output could not be committed atomically") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_created:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise RecommissioningGatewayError("temporary public launch output could not be removed") from exc


def _load_config(path_value: str) -> GatewayConfig:
    return GatewayConfig.from_record(
        _load_mapping_file(path_value, label="gateway configuration", byte_ceiling=_MAX_CONFIG_BYTES)
    )


def _issue_from_args(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    controller_signer = _load_private_signer(args.controller_private_key, "controller private key")
    service_public_key = _load_public_key_file(args.service_public_key, "service public key")
    generation_public_key = _load_public_key_file(
        args.generation_receipt_public_key,
        "generation-receipt public key",
    )
    record = issue_trusted_launch_receipt(
        profile=config.profile,
        runtime_config_digest=config.runtime_config_digest,
        service_nonce=config.service_nonce,
        controller_signer=controller_signer,
        service_public_key=service_public_key,
        generation_receipt_public_key=generation_public_key,
    )
    _write_new_atomic(args.output, canonical_json(record).encode("utf-8") + b"\n")
    return 0


def _serve_from_args(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    controller_public_key = _load_public_key_file(args.controller_public_key, "controller public key")
    launch_record = _load_mapping_file(
        args.trusted_launch_receipt,
        label="trusted launch receipt",
        byte_ceiling=_MAX_PUBLIC_RECORD_BYTES,
    )
    try:
        trusted = TrustedLaunchReceipt.from_controller_record(
            launch_record,
            controller_public_key=controller_public_key,
        )
    except Exception as exc:
        raise RecommissioningGatewayError("trusted launch receipt failed controller verification") from exc
    service_signer = _load_private_signer(args.service_private_key, "service private key")
    gateway = LoopbackAttestationGateway(
        profile=config.profile,
        trusted_launch_receipt=trusted,
        service_signer=service_signer,
        runtime_config_digest=config.runtime_config_digest,
        service_nonce=config.service_nonce,
        upstream_origin=config.upstream_origin,
        upstream_timeout_seconds=config.upstream_timeout_ms / 1000.0,
        max_request_bytes=config.max_request_bytes,
        max_response_bytes=config.max_response_bytes,
        max_concurrency=config.max_concurrency,
    )
    server = build_gateway_server(
        gateway,
        bind_host=config.bind_host,
        bind_port=config.bind_port,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="egv-recommissioning-gateway")
    subparsers = parser.add_subparsers(dest="command", required=True)
    issue = subparsers.add_parser("issue-launch", help="issue one no-overwrite public launch receipt")
    issue.add_argument("--config", required=True)
    issue.add_argument("--controller-private-key", required=True)
    issue.add_argument("--service-public-key", required=True)
    issue.add_argument("--generation-receipt-public-key", required=True)
    issue.add_argument("--output", required=True)
    issue.set_defaults(handler=_issue_from_args)

    serve = subparsers.add_parser("serve", help="serve the loopback-only signed attestation gateway")
    serve.add_argument("--config", required=True)
    serve.add_argument("--controller-public-key", required=True)
    serve.add_argument("--service-private-key", required=True)
    serve.add_argument("--trusted-launch-receipt", required=True)
    serve.set_defaults(handler=_serve_from_args)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except RecommissioningGatewayError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":  # pragma: no cover - exercised through the installed entry point
    raise SystemExit(main())


__all__ = [
    "GatewayConfig",
    "LoopbackAttestationGateway",
    "RecommissioningGatewayError",
    "build_gateway_server",
    "issue_trusted_launch_receipt",
    "main",
    "sign_service_observation",
]
