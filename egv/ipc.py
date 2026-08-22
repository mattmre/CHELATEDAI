"""Small authenticated local IPC adapter for the single ledger writer.

The service is intentionally narrow: clients can submit typed append requests
but cannot obtain a writable SQLite handle.  It is suitable for local smoke and
operator wiring; a production deployment still supplies the protected process
identity and socket permissions described by the runbook.
"""

from __future__ import annotations

import hmac
import json
from pathlib import Path
import secrets
import socket
import socketserver
import threading
from typing import Any, Dict, Mapping, Optional

from .canonical import canonical_json
from .errors import LedgerError
from .ledger import EvidenceLedger


_WRITER_METHODS = frozenset(
    {
        "append_event",
        "create_campaign",
        "create_run",
        "append_candidate",
        "append_verdict",
        "append_effect_receipt",
        "append_dependency",
        "append_correction",
        "append_retraction",
        "ingest_receipt",
        "add_checkpoint",
        "enqueue_projection_rebuild",
    }
)


class LedgerWriterService:
    """Threaded Unix-domain service that owns one writable ledger handle.

    The trainer/operator token may call the full typed writer surface. An
    evaluator receives a separate token and is mechanically restricted to
    signed ``ingest_receipt`` calls; verdict/effect materialization remains a
    trainer-owned ledger operation.
    """

    def __init__(
        self,
        ledger: EvidenceLedger,
        socket_path: str | Path,
        auth_token: str,
        *,
        evaluator_auth_token: Optional[str] = None,
    ) -> None:
        if ledger.mode != "writer":
            raise LedgerError("LedgerWriterService requires a writable ledger")
        if not auth_token:
            raise LedgerError("LedgerWriterService requires a non-empty authentication token")
        self.ledger = ledger
        self.socket_path = Path(socket_path)
        self.auth_token = auth_token
        self.evaluator_auth_token = evaluator_auth_token
        self._server: Optional[socketserver.ThreadingUnixStreamServer] = None
        self._thread: Optional[threading.Thread] = None

    def _dispatch(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        client_role = request.get("client_role", "trainer")
        method = request.get("method")
        if client_role == "trainer":
            expected_token = self.auth_token
        elif client_role == "evaluator":
            if self.evaluator_auth_token is None:
                raise LedgerError("evaluator IPC is not enabled")
            if method != "ingest_receipt":
                raise LedgerError("evaluator IPC is restricted to ingest_receipt")
            expected_token = self.evaluator_auth_token
        else:
            raise LedgerError(f"unsupported ledger client role: {client_role!r}")
        if not hmac.compare_digest(str(request.get("auth_token", "")), expected_token):
            raise LedgerError("unauthenticated ledger append request")
        arguments = dict(request.get("arguments") or {})
        if method not in _WRITER_METHODS:
            raise LedgerError(f"unsupported ledger writer method: {method!r}")
        return dict(getattr(self.ledger, str(method))(**arguments))

    def start(self) -> None:
        if self._server is not None:
            return
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()
        owner = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                line = self.rfile.readline(1024 * 1024)
                if not line:
                    return
                try:
                    request = json.loads(line.decode("utf-8"))
                    result = owner._dispatch(request)
                    response = {"ok": True, "result": result}
                except Exception as exc:  # the protocol must return a bounded error, not swallow it
                    response = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
                self.wfile.write((canonical_json(response) + "\n").encode("utf-8"))
                self.wfile.flush()

        self._server = socketserver.ThreadingUnixStreamServer(str(self.socket_path), Handler)
        self.socket_path.chmod(0o600)
        self._thread = threading.Thread(target=self._server.serve_forever, name="egv-ledger-writer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self.socket_path.exists():
            self.socket_path.unlink()

    def __enter__(self) -> "LedgerWriterService":
        self.start()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.stop()


class LedgerClient:
    """Authenticated append-only client with no SQLite access."""

    def __init__(self, socket_path: str | Path, auth_token: str, *, role: str = "trainer") -> None:
        if role not in {"trainer", "evaluator"}:
            raise LedgerError("ledger client role must be 'trainer' or 'evaluator'")
        self.socket_path = str(socket_path)
        self.auth_token = auth_token
        self.role = role

    def call(self, method: str, **arguments: Any) -> Dict[str, Any]:
        request = {
            "auth_token": self.auth_token,
            "client_role": self.role,
            "method": method,
            "arguments": arguments,
        }
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(10)
            connection.connect(self.socket_path)
            connection.sendall((canonical_json(request) + "\n").encode("utf-8"))
            data = b""
            while not data.endswith(b"\n"):
                chunk = connection.recv(1024 * 1024)
                if not chunk:
                    break
                data += chunk
        if not data:
            raise LedgerError("ledger writer returned no response")
        response = json.loads(data.decode("utf-8"))
        if not response.get("ok"):
            raise LedgerError(f"ledger writer rejected request: {response.get('error')}: {response.get('message')}")
        return dict(response["result"])

    def append_event(self, **arguments: Any) -> Dict[str, Any]:
        return self.call("append_event", **arguments)

    def append_dependency(self, **arguments: Any) -> Dict[str, Any]:
        return self.call("append_dependency", **arguments)

    def create_campaign(self, **arguments: Any) -> Dict[str, Any]:
        return self.call("create_campaign", **arguments)

    def create_run(self, **arguments: Any) -> Dict[str, Any]:
        return self.call("create_run", **arguments)

    def append_candidate(self, **arguments: Any) -> Dict[str, Any]:
        return self.call("append_candidate", **arguments)

    def append_verdict(self, **arguments: Any) -> Dict[str, Any]:
        return self.call("append_verdict", **arguments)

    def append_effect_receipt(self, **arguments: Any) -> Dict[str, Any]:
        return self.call("append_effect_receipt", **arguments)

    def append_correction(self, **arguments: Any) -> Dict[str, Any]:
        return self.call("append_correction", **arguments)

    def append_retraction(self, **arguments: Any) -> Dict[str, Any]:
        return self.call("append_retraction", **arguments)

    def add_checkpoint(self, **arguments: Any) -> Dict[str, Any]:
        return self.call("add_checkpoint", **arguments)

    def ingest_receipt(self, **arguments: Any) -> Dict[str, Any]:
        return self.call("ingest_receipt", **arguments)


def new_auth_token() -> str:
    """Create an in-memory IPC bearer token for an operator-owned service."""

    return secrets.token_urlsafe(32)


__all__ = ["LedgerClient", "LedgerWriterService", "new_auth_token"]
