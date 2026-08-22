"""Error types for the Evidence-Governed Variation evidence core."""

from __future__ import annotations


class EGVError(Exception):
    """Base class for errors raised by the EGV evidence core."""


class CanonicalizationError(EGVError, ValueError):
    """Raised when a value cannot be represented canonically as JSON."""


class LedgerError(EGVError):
    """Base class for authoritative-ledger errors."""


class LedgerBusyError(LedgerError):
    """Raised when another process already owns the writable ledger lock."""


class LedgerReadOnlyError(LedgerError):
    """Raised when a write is attempted through a read-only ledger handle."""


class AppendOnlyViolation(LedgerError):
    """Raised when immutable evidence history is changed or deleted."""


class IntegrityError(LedgerError):
    """Raised when a ledger hash chain or stored payload is inconsistent."""


class IdempotencyConflictError(LedgerError):
    """Raised when one idempotency key is reused for different content."""


class UnknownReferenceError(LedgerError):
    """Raised when a dependency or lifecycle record references an unknown ID."""


class DependencyCycleError(LedgerError):
    """Raised when an inserted dependency would create a cycle."""


class ReceiptError(EGVError):
    """Base class for receipt signing, journaling, and ingest errors."""


class ReceiptVerificationError(ReceiptError):
    """Raised when a receipt signature, schema, or chain binding is invalid."""


class ReceiptConflictError(ReceiptError):
    """Raised when duplicate receipt delivery carries conflicting content."""


class OptionalDependencyError(EGVError):
    """Raised when an explicitly selected optional backend is unavailable."""


class ProjectionError(EGVError):
    """Raised when a disposable retrieval projection cannot be built or read."""


class PublicSchemaError(EGVError, ValueError):
    """Raised when a public projection record violates its closed schema."""


class PublicReplayError(EGVError):
    """Raised when public cryptographic replay cannot be completed honestly."""


class PhaseUnavailable(EGVError):
    """Raised for runbook phases intentionally outside the Slice 2 core."""
