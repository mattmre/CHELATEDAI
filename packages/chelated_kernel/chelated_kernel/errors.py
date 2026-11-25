"""Typed failures for the public chelated_kernel surface."""

from __future__ import annotations


class ChelatedKernelError(Exception):
    """Base error for every public kernel failure."""


class UnsupportedCapabilityError(ChelatedKernelError):
    """Raised when a requested capability is outside the wheel."""

    def __init__(self, capability_name: str, *, reason: str) -> None:
        self.capability_name = str(capability_name)
        self.reason = str(reason)
        super().__init__(f"unsupported capability {self.capability_name}: {self.reason}")


class Tra1VerificationError(ChelatedKernelError):
    """Fail-closed representation-contract rejection."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"TRA-1 {self.code}: {self.detail}")


class TenantIsolationError(ChelatedKernelError):
    """Raised when a call would cross a tenant partition."""

    def __init__(self, detail: str) -> None:
        self.detail = str(detail)
        super().__init__(self.detail)


class EnvelopeError(ChelatedKernelError):
    """Raised when an ingested envelope is not one of the supported families."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


class ProfileError(ChelatedKernelError):
    """Raised when a profile or ledger policy name is unknown."""

    def __init__(self, detail: str) -> None:
        self.detail = str(detail)
        super().__init__(self.detail)


class StreamGuardError(ChelatedKernelError):
    """Raised when a stream guard cannot be constructed or advanced."""

    def __init__(self, detail: str) -> None:
        self.detail = str(detail)
        super().__init__(self.detail)


class ReceiptVerificationError(ChelatedKernelError):
    """Raised when an Ed25519 hash-chained receipt does not verify."""

    def __init__(self, detail: str) -> None:
        self.detail = str(detail)
        super().__init__(self.detail)
