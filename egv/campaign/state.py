"""Atomic monotonic P0-P13 campaign lifecycle state."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

from ..canonical import canonical_bytes, digest_for, validate_sha256
from ..public import verify_public_restore_receipt
from .errors import CampaignStateError


CAMPAIGN_STATE_SCHEMA = "egv-campaign-lifecycle-state-v1"
PHASES: Tuple[str, ...] = tuple("P{}".format(index) for index in range(14))
_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "phase",
        "sequence",
        "previous_state_digest",
        "restore_required",
        "service_state",
        "inventory_digest",
        "last_receipt_digest",
    }
)
@dataclass(frozen=True)
class CampaignState:
    campaign_id: str
    phase: str
    sequence: int
    previous_state_digest: Optional[str]
    restore_required: bool
    service_state: str
    inventory_digest: Optional[str] = None
    last_receipt_digest: Optional[str] = None
    schema_version: str = CAMPAIGN_STATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CAMPAIGN_STATE_SCHEMA:
            raise CampaignStateError("unsupported campaign lifecycle state schema")
        if not isinstance(self.campaign_id, str) or not self.campaign_id:
            raise CampaignStateError("campaign state requires a non-empty campaign ID")
        if self.phase not in PHASES:
            raise CampaignStateError("campaign state phase is outside P0-P13")
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence < 0:
            raise CampaignStateError("campaign state sequence must be nonnegative")
        if not isinstance(self.restore_required, bool):
            raise CampaignStateError("restore_required must be boolean")
        if self.service_state not in {"RUNNING", "STOPPING", "STOPPED", "PARTIAL", "RESTORING", "RESTORED"}:
            raise CampaignStateError("service_state is outside the closed vocabulary")
        if self.restore_required and self.service_state not in {"STOPPING", "STOPPED", "PARTIAL", "RESTORING"}:
            raise CampaignStateError("restore-required state must represent interrupted services")
        if not self.restore_required and self.service_state in {"STOPPING", "PARTIAL", "RESTORING"}:
            raise CampaignStateError("interrupted services must require restoration")
        for field in ("previous_state_digest", "inventory_digest", "last_receipt_digest"):
            value = getattr(self, field)
            if value is not None:
                try:
                    validate_sha256(value, field)
                except Exception as exc:
                    raise CampaignStateError(str(exc)) from exc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "phase": self.phase,
            "sequence": self.sequence,
            "previous_state_digest": self.previous_state_digest,
            "restore_required": self.restore_required,
            "service_state": self.service_state,
            "inventory_digest": self.inventory_digest,
            "last_receipt_digest": self.last_receipt_digest,
        }

    @property
    def digest(self) -> str:
        return digest_for(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CampaignState":
        if not isinstance(value, Mapping) or set(value) != _FIELDS:
            raise CampaignStateError("campaign lifecycle state is not a closed schema")
        return cls(**{field: value[field] for field in _FIELDS})


class CampaignStateStore:
    """Durably replace one state record while enforcing phase monotonicity."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    def load(self, *, expected_digest: Optional[str] = None) -> Optional[CampaignState]:
        with _exclusive_path_lock(self.lock_path):
            return self._load_unlocked(expected_digest=expected_digest)

    def _load_unlocked(self, *, expected_digest: Optional[str] = None) -> Optional[CampaignState]:
        if not self.path.exists():
            return None
        if self.path.is_symlink() or not self.path.is_file():
            raise CampaignStateError("campaign state must be a regular non-link file")
        try:
            raw = self.path.read_bytes()
            value = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise CampaignStateError("campaign lifecycle state cannot be decoded") from exc
        state = CampaignState.from_mapping(value)
        if raw != canonical_bytes(state.to_dict()):
            raise CampaignStateError("campaign lifecycle state is not canonical JSON")
        if expected_digest is not None and state.digest != validate_sha256(expected_digest, "expected state digest"):
            raise CampaignStateError("campaign lifecycle rollback or replacement detected")
        return state

    def initialize(self, *, campaign_id: str, inventory_digest: Optional[str] = None) -> CampaignState:
        with _exclusive_path_lock(self.lock_path):
            if self._load_unlocked() is not None:
                raise CampaignStateError("campaign lifecycle is already initialized")
            state = CampaignState(campaign_id, "P0", 0, None, False, "RUNNING", inventory_digest)
            self._atomic_write(state)
            return state

    def advance(
        self,
        next_phase: str,
        *,
        expected_digest: str,
        last_receipt_digest: Optional[str] = None,
    ) -> CampaignState:
        with _exclusive_path_lock(self.lock_path):
            current = self._load_unlocked(expected_digest=expected_digest)
            if current is None:
                raise CampaignStateError("campaign lifecycle is not initialized")
            current_index = PHASES.index(current.phase)
            if current.service_state in {"STOPPING", "PARTIAL", "RESTORING"}:
                raise CampaignStateError("campaign cannot advance from an incomplete service transition")
            if next_phase not in PHASES or PHASES.index(next_phase) != current_index + 1:
                raise CampaignStateError("campaign phase cannot skip or regress")
            state = CampaignState(
                campaign_id=current.campaign_id,
                phase=next_phase,
                sequence=current.sequence + 1,
                previous_state_digest=current.digest,
                restore_required=current.restore_required,
                service_state=current.service_state,
                inventory_digest=current.inventory_digest,
                last_receipt_digest=last_receipt_digest,
            )
            self._atomic_write(state)
            return state

    def require_restore(self, *, expected_digest: str, service_state: str = "PARTIAL") -> CampaignState:
        with _exclusive_path_lock(self.lock_path):
            current = self._load_unlocked(expected_digest=expected_digest)
            if current is None:
                raise CampaignStateError("campaign lifecycle is not initialized")
            if PHASES.index(current.phase) < 3:
                raise CampaignStateError("service interruption cannot precede P3")
            if service_state not in {"STOPPING", "STOPPED", "PARTIAL", "RESTORING"}:
                raise CampaignStateError("restore-required service state is invalid")
            state = CampaignState(
                current.campaign_id,
                current.phase,
                current.sequence + 1,
                current.digest,
                True,
                service_state,
                current.inventory_digest,
                current.last_receipt_digest,
            )
            self._atomic_write(state)
            return state

    def mark_restored(
        self,
        *,
        expected_digest: str,
        public_receipt: Mapping[str, Any],
        evaluator_public_key: Any,
        expected_payload: Mapping[str, Any],
    ) -> CampaignState:
        with _exclusive_path_lock(self.lock_path):
            current = self._load_unlocked(expected_digest=expected_digest)
            if current is None or not current.restore_required:
                raise CampaignStateError("restoration can complete only from restore-required state")
            try:
                receipt_digest = verify_public_restore_receipt(public_receipt, evaluator_public_key)
            except Exception as exc:
                raise CampaignStateError("restoration requires a valid canonical signed receipt") from exc
            envelope_fields = {"schema_version", "receipt_id", "signing_key_id", "signature"}
            observed_payload = {
                key: value for key, value in public_receipt.items() if key not in envelope_fields
            }
            if not isinstance(expected_payload, Mapping) or observed_payload != dict(expected_payload):
                raise CampaignStateError("restoration receipt does not match the verified snapshot payload")
            if expected_payload.get("campaign_id") != current.campaign_id:
                raise CampaignStateError("restoration receipt campaign differs from lifecycle state")
            if expected_payload.get("private_inventory_digest") != current.inventory_digest:
                raise CampaignStateError("restoration receipt inventory differs from lifecycle state")
            # Normal completion restores at P12. A failed/partial P3 stop restores
            # immediately without fabricating skipped P4-P12 gates.
            if current.phase != "P12" and current.service_state not in {"PARTIAL", "RESTORING"}:
                raise CampaignStateError("non-emergency restoration completion requires P12")
            state = CampaignState(
                current.campaign_id,
                current.phase,
                current.sequence + 1,
                current.digest,
                False,
                "RESTORED",
                current.inventory_digest,
                receipt_digest,
            )
            self._atomic_write(state)
            return state

    def _atomic_write(self, state: CampaignState) -> None:
        descriptor, name = tempfile.mkstemp(prefix=".campaign-state-", dir=str(self.path.parent))
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(canonical_bytes(state.to_dict()))
                handle.flush()
                os.fsync(handle.fileno())
            _durable_replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()


@contextmanager
def _exclusive_path_lock(path: Path) -> Iterator[None]:
    """Cross-process advisory lock retained for the complete CAS transaction."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _durable_replace(source: Path, destination: Path) -> None:
    """Replace atomically and durably commit the containing directory entry."""

    if os.name == "nt":
        move_file = ctypes.windll.kernel32.MoveFileExW
        move_file.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_ulong]
        move_file.restype = ctypes.c_int
        # REPLACE_EXISTING | WRITE_THROUGH is Windows' durable rename primitive.
        if not move_file(str(source), str(destination), 0x1 | 0x8):
            raise OSError(ctypes.get_last_error(), "durable state replacement failed")
        return
    source.replace(destination)
    directory_fd = os.open(str(destination.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


__all__ = ["CAMPAIGN_STATE_SCHEMA", "PHASES", "CampaignState", "CampaignStateStore"]
