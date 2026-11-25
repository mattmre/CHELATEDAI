"""Streaming holdback guard. Restricted spans are hashed, never copied into receipts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator

from chelated_kernel.errors import StreamGuardError, TenantIsolationError
from chelated_kernel.profiles import get_stream_profile, require_tenant_id


def _normalize_spans(restricted_spans: object, tenant_id: str) -> list[tuple[str, ...]]:
    if isinstance(restricted_spans, (str, bytes)) or not isinstance(restricted_spans, (list, tuple)):
        raise StreamGuardError("restricted_spans must be a sequence of spans")
    spans: list[tuple[str, ...]] = []
    for span in restricted_spans:
        if isinstance(span, dict):
            tokens = span.get("tokens", span.get("span"))
            if not isinstance(tokens, (list, tuple)) or len(tokens) == 0:
                raise StreamGuardError("empty restricted span")
            foreign = span.get("tenant_id")
            if foreign is not None and str(foreign) != tenant_id:
                continue
            spans.append(tuple(str(token) for token in tokens))
            continue
        if isinstance(span, (list, tuple)):
            if len(span) == 0:
                raise StreamGuardError("empty restricted span")
            spans.append(tuple(str(token) for token in span))
            continue
        raise StreamGuardError("each span must be a token sequence or a dict")
    return spans


class StreamGuard:
    """Iterator that suppresses restricted spans and records an audit receipt."""

    def __init__(
        self,
        token_iter: object,
        *,
        ledger: object,
        spans: list[tuple[str, ...]],
        mode: str,
        profile_code: int,
        tenant_id: str,
    ) -> None:
        self._source = iter(token_iter)  # type: ignore[arg-type]
        self._ledger = ledger
        self._spans = spans
        self._mode = mode
        self._profile_code = int(profile_code)
        self._tenant_id = tenant_id
        self._hold = max((len(span) for span in spans), default=0)
        self._buf: list[str] = []
        self._ready: list[str] = []
        self._exhausted = False
        self._seq = 0
        self.suppressed_spans = 0
        self.llm_write_tokens = 0

    def __iter__(self) -> StreamGuard:
        return self

    def __next__(self) -> str:
        while True:
            item = self._pop()
            if item is not None:
                return item
            if self._exhausted:
                raise StopIteration
            try:
                token = next(self._source)
            except StopIteration:
                self._exhausted = True
                continue
            if not isinstance(token, str):
                raise StreamGuardError("stream tokens must be strings")
            self._accept(token)

    def _accept(self, token: str) -> None:
        self._buf.append(token)
        if self._mode != "async_armed":
            return
        current = tuple(self._buf)
        for span in self._spans:
            if current == span:
                self._suppress(self._buf)
                self._buf.clear()
                return
        if any(span[: len(current)] == current for span in self._spans):
            return
        self._ready.extend(self._buf)
        self._buf.clear()

    def _pop(self) -> str | None:
        if self._mode == "async_armed":
            if self._ready:
                return self._ready.pop(0)
            if self._exhausted and self._buf:
                self._ready.extend(self._buf)
                self._buf.clear()
                return self._ready.pop(0)
            return None
        self._drop_prefixes()
        if len(self._buf) > self._hold or (self._exhausted and self._buf):
            self._drop_prefixes()
            if not self._buf:
                return None
            if len(self._buf) > self._hold or self._exhausted:
                return self._buf.pop(0)
        return None

    def _drop_prefixes(self) -> None:
        changed = True
        while changed and self._buf:
            changed = False
            for span in self._spans:
                width = len(span)
                if width and tuple(self._buf[:width]) == span:
                    self._suppress(self._buf[:width])
                    del self._buf[:width]
                    changed = True
                    break

    def _suppress(self, tokens: list[str]) -> None:
        digest = hashlib.sha256(json.dumps(list(tokens), separators=(",", ":")).encode("utf-8")).hexdigest()
        self._seq += 1
        self._ledger.ingest(  # type: ignore[attr-defined]
            {
                "specversion": "1.0",
                "type": "com.chelated.stream.suppression",
                "id": f"sup-{digest[:20]}",
                "source": "chelated://stream-guard",
                "tenant_id": self._tenant_id,
                "time": self._seq,
                "data": {
                    "span_sha256": digest,
                    "profile_code": self._profile_code,
                    "cites_audit": True,
                },
            }
        )
        self.suppressed_spans += 1
        self.llm_write_tokens = int(getattr(self._ledger, "llm_write_tokens", 0))


def wrap_stream(
    token_iter: Iterator[str] | object,
    *,
    ledger: object,
    restricted_spans: object,
    profile: str = "strict_holdback",
    tenant_id: str = "default",
) -> StreamGuard:
    """Hold back or suppress restricted spans and append an Ed25519 ledger receipt."""

    tenant = require_tenant_id(tenant_id)
    spec = get_stream_profile(profile)
    ledger_tenant = getattr(ledger, "tenant_id", None)
    if ledger_tenant != tenant:
        raise TenantIsolationError("stream ledger tenant_id does not match")
    if not hasattr(ledger, "ingest"):
        raise StreamGuardError("ledger must accept ingest")
    spans = _normalize_spans(restricted_spans, tenant)
    return StreamGuard(
        token_iter,
        ledger=ledger,
        spans=spans,
        mode=spec.mode,
        profile_code=spec.code,
        tenant_id=tenant,
    )
