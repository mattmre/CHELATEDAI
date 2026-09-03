"""Deny-by-default authority boundary for the Evaluation slice."""

from __future__ import annotations

from dataclasses import dataclass
import shutil
from typing import Any, Dict, FrozenSet, Optional, Tuple

from ..canonical import digest_for
from .errors import AuthorityDenied, DockerConfigurationError


ALLOWED_ACTIONS = frozenset({"execute_candidate"})


@dataclass(frozen=True)
class RuntimeStatus:
    backend: str
    available: bool
    enforceable: bool
    claimed: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "available": self.available,
            "enforceable": self.enforceable,
            "claimed": self.claimed,
            "reason": self.reason,
        }


class AuthorityRuntime:
    """Runtime interface used by the broker and sandbox."""

    status = RuntimeStatus("unavailable", False, False, False, "no enforceable runtime selected")

    def require_enforceable(self) -> None:
        if not self.status.enforceable:
            raise AuthorityDenied("AUTHORITY_RUNTIME_UNAVAILABLE: {}".format(self.status.reason))


class OpenShellRuntime(AuthorityRuntime):
    """Read-only detector; this implementation never claims OpenShell support."""

    def __init__(self) -> None:
        binary = shutil.which("openshell")
        self.status = RuntimeStatus(
            backend="openshell",
            available=binary is not None,
            enforceable=False,
            claimed=False,
            reason="OpenShell policy ABI is not implemented by this slice" if binary else "openshell executable absent",
        )


class LocalEnforcedRuntime(AuthorityRuntime):
    """Non-enforcing helper retained only for unit tests.

    The old audit-hook helper is deliberately not an authority runtime.  It
    cannot authorize a production candidate and is never used by the ceiling
    smoke.
    """

    status = RuntimeStatus(
        backend="local-test-helper",
        available=True,
        enforceable=False,
        claimed=False,
        reason="test-only helper; kernel isolation is not provided",
    )


class DockerEnforcedRuntime(AuthorityRuntime):
    """Pinned Docker runtime status used by production Evaluation smoke."""

    def __init__(self, config: Optional[Any] = None) -> None:
        try:
            from .sandbox import DockerSandboxConfig

            self.config = config or DockerSandboxConfig.from_environment()
            image_id = self.config.verify_image()
            self.status = RuntimeStatus(
                backend="docker-enforced-v1",
                available=True,
                enforceable=True,
                claimed=True,
                reason="cached pinned image verified; network none, read-only root, dropped capabilities",
            )
            self.image_id = image_id
        except (DockerConfigurationError, OSError, ValueError) as exc:
            self.config = config
            self.image_id = None
            self.status = RuntimeStatus(
                backend="docker-enforced-v1",
                available=False,
                enforceable=False,
                claimed=False,
                reason="configured Docker isolation is unavailable: {}".format(type(exc).__name__),
            )


def detect_authority_runtime() -> RuntimeStatus:
    """Report the configured enforceable runtime without falling back."""

    return DockerEnforcedRuntime().status


@dataclass(frozen=True)
class AuthorityPolicy:
    allowed_actions: FrozenSet[str] = frozenset()
    credential_scopes: Tuple[str, ...] = tuple()
    process_creation: bool = False
    network: bool = False

    def __post_init__(self) -> None:
        if not set(self.allowed_actions).issubset(ALLOWED_ACTIONS):
            raise ValueError("authority policy contains an unknown action")
        if self.credential_scopes or self.process_creation or self.network:
            raise ValueError("Evaluation authority policy cannot enable credentials, process, or network access")

    @classmethod
    def candidate_execution(cls) -> "AuthorityPolicy":
        return cls(
            allowed_actions=frozenset({"execute_candidate"}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed_actions": sorted(self.allowed_actions),
            "credential_scopes": list(self.credential_scopes),
            "process_creation": self.process_creation,
            "network": self.network,
        }

    @property
    def digest(self) -> str:
        return digest_for(self.to_dict())


@dataclass(frozen=True)
class AuthorityDecision:
    allowed: bool
    action: str
    candidate_id: str
    policy_digest: str
    reason_code: str


class AuthorityBroker:
    """Typed least-privilege broker; every action starts denied."""

    def __init__(self, runtime: Optional[AuthorityRuntime], policy: Optional[AuthorityPolicy] = None) -> None:
        self.runtime = runtime
        self.policy = policy or AuthorityPolicy()

    def authorize(
        self,
        *,
        action: str,
        candidate_id: str,
        requested_authority: str,
        artifact_digest: str,
        declared_locus: str = "module:solve",
    ) -> AuthorityDecision:
        if self.runtime is None or not self.runtime.status.enforceable:
            raise AuthorityDenied("AUTHORITY_RUNTIME_UNAVAILABLE")
        if action not in self.policy.allowed_actions:
            raise AuthorityDenied("AUTHORITY_DENIED")
        if requested_authority != "EXECUTE_CANDIDATE" or action != "execute_candidate":
            raise AuthorityDenied("AUTHORITY_DENIED")
        if not candidate_id or not artifact_digest or not declared_locus:
            raise AuthorityDenied("PROTOCOL_VIOLATION")
        self.runtime.require_enforceable()
        return AuthorityDecision(True, action, candidate_id, self.policy.digest, "ALLOW")

    def deny(self, *, action: str, candidate_id: str, reason_code: str = "AUTHORITY_DENIED") -> AuthorityDecision:
        return AuthorityDecision(False, action, candidate_id, self.policy.digest, reason_code)


__all__ = [
    "ALLOWED_ACTIONS",
    "AuthorityBroker",
    "AuthorityDecision",
    "AuthorityPolicy",
    "AuthorityRuntime",
    "DockerEnforcedRuntime",
    "LocalEnforcedRuntime",
    "OpenShellRuntime",
    "RuntimeStatus",
    "detect_authority_runtime",
]
