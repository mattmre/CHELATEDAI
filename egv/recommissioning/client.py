"""Fail-closed OpenAI-compatible client for bounded recommissioning pilots.

The service address is runtime-private and is deliberately excluded from every
serializable receipt.  This module authenticates model identity and response
shape; it does not treat a healthy endpoint as model-quality evidence.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import re
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from ..canonical import canonical_json, digest_bytes, digest_for


class RecommissioningClientError(RuntimeError):
    """The bounded model service contract could not be proven."""


Transport = Callable[[str, str, Optional[bytes], float], Tuple[int, bytes]]


_ACTION_IDS = ("PROPOSE", "EDIT", "CRITIQUE", "COMMIT")
_RESPONSE_FIELDS = {"action", "source", "summary", "evidence_ids"}


@dataclass(frozen=True)
class ModelServiceProfile:
    model_id: str
    model_revision: str
    engine_image_digest: str
    max_model_len: int
    precision: str
    kv_cache_dtype: str

    def validate(self) -> None:
        for value, label in (
            (self.model_id, "model ID"),
            (self.model_revision, "model revision"),
            (self.engine_image_digest, "engine image digest"),
            (self.precision, "precision"),
            (self.kv_cache_dtype, "KV-cache dtype"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise RecommissioningClientError("{} must be a non-empty string".format(label))
        if re.fullmatch(r"[0-9a-f]{40,64}", self.model_revision) is None:
            raise RecommissioningClientError("model revision is not an immutable identity")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.engine_image_digest) is None:
            raise RecommissioningClientError("engine image must use a sha256 repository digest")
        if not isinstance(self.max_model_len, int) or self.max_model_len < 512:
            raise RecommissioningClientError("maximum model length is outside the bounded contract")

    def public_manifest(self) -> Dict[str, Any]:
        self.validate()
        return {
            "schema_version": "egv-recommissioning-model-service-v1",
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "engine_image_digest": self.engine_image_digest,
            "max_model_len": self.max_model_len,
            "precision": self.precision,
            "kv_cache_dtype": self.kv_cache_dtype,
        }

    @property
    def digest(self) -> str:
        return digest_for(self.public_manifest())


@dataclass(frozen=True)
class GenerationBudget:
    max_completion_tokens: int = 2048
    timeout_seconds: float = 120.0
    seed: int = 0

    def validate(self, profile: ModelServiceProfile) -> None:
        if (
            not isinstance(self.max_completion_tokens, int)
            or self.max_completion_tokens < 1
            or self.max_completion_tokens > min(8192, profile.max_model_len - 128)
        ):
            raise RecommissioningClientError("completion-token budget is outside the frozen envelope")
        if not isinstance(self.timeout_seconds, (int, float)) or not (1.0 <= float(self.timeout_seconds) <= 600.0):
            raise RecommissioningClientError("generation timeout is outside the frozen envelope")
        if not isinstance(self.seed, int) or self.seed < 0 or self.seed > 2**31 - 1:
            raise RecommissioningClientError("generation seed is outside the frozen envelope")

    def manifest(self) -> Dict[str, Any]:
        return {
            "max_completion_tokens": self.max_completion_tokens,
            "timeout_seconds": float(self.timeout_seconds),
            "seed": self.seed,
            "temperature": 0.0,
            "n": 1,
        }


@dataclass(frozen=True)
class TypedCandidateResponse:
    action: str
    source: str
    summary: str
    evidence_ids: Tuple[str, ...]

    @classmethod
    def parse(cls, value: Any, *, visible_evidence_ids: Sequence[str]) -> "TypedCandidateResponse":
        if not isinstance(value, Mapping) or set(value) != _RESPONSE_FIELDS:
            raise RecommissioningClientError("candidate response does not use the closed field set")
        action = value.get("action")
        source = value.get("source")
        summary = value.get("summary")
        evidence = value.get("evidence_ids")
        if action not in _ACTION_IDS:
            raise RecommissioningClientError("candidate response uses an unknown action")
        if not isinstance(source, str) or not source.strip() or len(source.encode("utf-8")) > 256 * 1024:
            raise RecommissioningClientError("candidate source is empty or exceeds the byte ceiling")
        try:
            module = ast.parse(source)
        except (SyntaxError, ValueError) as exc:
            raise RecommissioningClientError("candidate source is not one complete Python module") from exc
        if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in module.body):
            raise RecommissioningClientError("candidate source does not define a top-level function")
        if not isinstance(summary, str) or len(summary.encode("utf-8")) > 512:
            raise RecommissioningClientError("candidate summary exceeds the bounded metadata ceiling")
        if (
            not isinstance(evidence, list)
            or not all(isinstance(item, str) and item for item in evidence)
            or evidence != sorted(set(evidence))
        ):
            raise RecommissioningClientError("candidate evidence IDs are not sorted and unique")
        allowed = set(visible_evidence_ids)
        if any(item not in allowed for item in evidence):
            raise RecommissioningClientError("candidate cited evidence that was not model-visible")
        return cls(str(action), source, summary, tuple(evidence))

    def private_record(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "source": self.source,
            "summary": self.summary,
            "evidence_ids": list(self.evidence_ids),
        }

    def public_record(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "source_digest": digest_bytes(self.source.encode("utf-8")),
            "summary_digest": digest_bytes(self.summary.encode("utf-8")),
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class GenerationReceipt:
    model_profile_digest: str
    budget_digest: str
    request_digest: str
    response_digest: str
    candidate_digest: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_ms: int

    def public_record(self) -> Dict[str, Any]:
        return {
            "schema_version": "egv-recommissioning-generation-receipt-v1",
            "model_profile_digest": self.model_profile_digest,
            "budget_digest": self.budget_digest,
            "request_digest": self.request_digest,
            "response_digest": self.response_digest,
            "candidate_digest": self.candidate_digest,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass(frozen=True)
class GenerationResult:
    candidate: TypedCandidateResponse
    receipt: GenerationReceipt


def candidate_response_schema() -> Dict[str, Any]:
    return {
        "name": "egv_candidate_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": sorted(_RESPONSE_FIELDS),
            "properties": {
                "action": {"type": "string", "enum": list(_ACTION_IDS)},
                "source": {"type": "string", "minLength": 1},
                "summary": {"type": "string", "maxLength": 512},
                "evidence_ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "uniqueItems": True,
                },
            },
        },
    }


class UrllibLoopbackTransport:
    """Minimal transport that refuses non-loopback service addresses."""

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
        self._opener = urllib_request.build_opener(_RejectRedirectHandler())

    def __call__(self, method: str, path: str, body: Optional[bytes], timeout: float) -> Tuple[int, bytes]:
        if not path.startswith("/") or "?" in path or "#" in path:
            raise RecommissioningClientError("model request path is not closed")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib_request.Request(self._origin + path, data=body, headers=headers, method=method)
        try:
            with self._opener.open(request, timeout=timeout) as response:
                payload = response.read(4 * 1024 * 1024 + 1)
                status = int(response.status)
        except (urllib_error.URLError, OSError, TimeoutError) as exc:
            raise RecommissioningClientError("bounded model service request failed") from exc
        if len(payload) > 4 * 1024 * 1024:
            raise RecommissioningClientError("model service response exceeds the byte ceiling")
        return status, payload


class _RejectRedirectHandler(urllib_request.HTTPRedirectHandler):
    """Keep every request on the attested loopback origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


class BoundedModelClient:
    """Identity-bound, single-request client with no hidden automatic retries."""

    def __init__(self, profile: ModelServiceProfile, transport: Transport) -> None:
        profile.validate()
        if not callable(transport):
            raise RecommissioningClientError("model transport is not callable")
        self.profile = profile
        self._transport = transport

    @staticmethod
    def _decode_json(status: int, body: bytes, label: str) -> Any:
        if status != 200:
            raise RecommissioningClientError("{} returned HTTP {}".format(label, status))
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise RecommissioningClientError("{} did not return valid UTF-8 JSON".format(label)) from exc

    def attest(self, *, timeout_seconds: float = 10.0) -> Dict[str, Any]:
        health_status, health_body = self._transport("GET", "/health", None, timeout_seconds)
        if health_status != 200:
            raise RecommissioningClientError("model health endpoint is not ready")
        models_status, models_body = self._transport("GET", "/v1/models", None, timeout_seconds)
        models = self._decode_json(models_status, models_body, "model-list endpoint")
        records = models.get("data") if isinstance(models, Mapping) else None
        if not isinstance(records, list):
            raise RecommissioningClientError("model-list response is not closed enough to authenticate")
        identities = [item.get("id") for item in records if isinstance(item, Mapping)]
        if identities != [self.profile.model_id]:
            raise RecommissioningClientError("served model identity differs from the frozen profile")
        return {
            "schema_version": "egv-recommissioning-service-attestation-v1",
            "model_profile_digest": self.profile.digest,
            "model_id": self.profile.model_id,
            "health_body_digest": digest_bytes(health_body),
            "model_list_digest": digest_bytes(models_body),
        }

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        visible_evidence_ids: Sequence[str],
        budget: GenerationBudget,
    ) -> GenerationResult:
        budget.validate(self.profile)
        if (
            not isinstance(messages, (tuple, list))
            or not messages
            or any(
                not isinstance(message, Mapping)
                or set(message) != {"role", "content"}
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
                or not message.get("content")
                for message in messages
            )
        ):
            raise RecommissioningClientError("chat messages do not use the closed prompt envelope")
        if list(visible_evidence_ids) != sorted(set(visible_evidence_ids)):
            raise RecommissioningClientError("visible evidence IDs must be sorted and unique")
        request_value = {
            "model": self.profile.model_id,
            "messages": [dict(message) for message in messages],
            "temperature": 0.0,
            "seed": budget.seed,
            "max_tokens": budget.max_completion_tokens,
            "n": 1,
            "response_format": {"type": "json_schema", "json_schema": candidate_response_schema()},
        }
        request_bytes = canonical_json(request_value).encode("utf-8")
        started = time.monotonic()
        status, response_bytes = self._transport(
            "POST", "/v1/chat/completions", request_bytes, float(budget.timeout_seconds)
        )
        elapsed_ms = max(0, int((time.monotonic() - started) * 1000))
        response = self._decode_json(status, response_bytes, "chat-completions endpoint")
        if not isinstance(response, Mapping) or response.get("model") != self.profile.model_id:
            raise RecommissioningClientError("completion response model differs from the frozen profile")
        choices = response.get("choices")
        usage = response.get("usage")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
            raise RecommissioningClientError("completion response does not contain exactly one choice")
        choice = choices[0]
        message = choice.get("message")
        finish_reason = choice.get("finish_reason")
        if not isinstance(message, Mapping) or set(message) - {"role", "content", "reasoning_content", "tool_calls"}:
            raise RecommissioningClientError("completion message contains an unexpected field")
        if message.get("role") != "assistant" or not isinstance(message.get("content"), str):
            raise RecommissioningClientError("completion message is not a typed assistant response")
        if finish_reason != "stop":
            raise RecommissioningClientError("completion did not terminate with the frozen stop disposition")
        try:
            candidate_value = json.loads(message["content"])
        except ValueError as exc:
            raise RecommissioningClientError("schema-constrained content is not valid JSON") from exc
        candidate = TypedCandidateResponse.parse(
            candidate_value, visible_evidence_ids=visible_evidence_ids
        )
        if not isinstance(usage, Mapping):
            raise RecommissioningClientError("completion response omitted token accounting")
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if (
            not isinstance(prompt_tokens, int)
            or prompt_tokens < 1
            or not isinstance(completion_tokens, int)
            or completion_tokens < 1
            or completion_tokens > budget.max_completion_tokens
        ):
            raise RecommissioningClientError("completion token accounting violates the budget")
        receipt = GenerationReceipt(
            model_profile_digest=self.profile.digest,
            budget_digest=digest_for(budget.manifest()),
            request_digest=digest_bytes(request_bytes),
            response_digest=digest_bytes(response_bytes),
            candidate_digest=digest_for(candidate.public_record()),
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            elapsed_ms=elapsed_ms,
        )
        return GenerationResult(candidate, receipt)


__all__ = [
    "BoundedModelClient",
    "GenerationBudget",
    "GenerationReceipt",
    "GenerationResult",
    "ModelServiceProfile",
    "RecommissioningClientError",
    "TypedCandidateResponse",
    "UrllibLoopbackTransport",
    "candidate_response_schema",
]
