"""Authoritative append-only SQLite evidence ledger.

The ledger is deliberately boring: one writable handle owns a process lock,
SQLite runs in WAL/FULL-sync mode, every history row is inserted into a hash
chain, and corrections/retractions are new rows.  Read-only handles use
SQLite's ``mode=ro`` URI and ``query_only`` pragma.  The Qdrant layer consumes
this module but never becomes its source of truth.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from .canonical import (
    GENESIS_HASH,
    canonical_bytes,
    canonical_json,
    chain_digest,
    content_id,
    digest_for,
    parse_canonical_jsonl,
    utc_now_iso,
)
from .errors import (
    DependencyCycleError,
    IdempotencyConflictError,
    IntegrityError,
    LedgerBusyError,
    LedgerError,
    LedgerReadOnlyError,
    ReceiptConflictError,
    ReceiptVerificationError,
    UnknownReferenceError,
)
from .receipts import key_id_for_public_key, receipt_hash, verify_receipt


LEDGER_SCHEMA_VERSION = 1
INLINE_PAYLOAD_LIMIT = 16 * 1024
PUBLIC_BLOB_ROLES = frozenset(
    {
        "public-metrics",
        "public-projection",
        "protocol",
        "model-manifest",
        "data-manifest",
        "adapter",
        "restore-receipt",
    }
)
RETRACTED = "RETRACTED"
STALE_DEPENDENT = "STALE_DEPENDENT"
PROMOTED = "PROMOTED"
REJECTED = "REJECTED"
ABSTAINED = "ABSTAINED"

_ACTIVE_WRITER_COUNT = 0
_ACTIVE_WRITER_LOCK = threading.Lock()


def active_writer_count() -> int:
    """Return the number of writable ledger handles in this process.

    Production evaluator construction uses this as a co-hosting guard.  The
    count is process-local by design; the actual ledger writer remains a
    separate OS process and is protected by the existing file lock.
    """

    with _ACTIVE_WRITER_LOCK:
        return _ACTIVE_WRITER_COUNT


def _register_writer() -> None:
    global _ACTIVE_WRITER_COUNT
    with _ACTIVE_WRITER_LOCK:
        _ACTIVE_WRITER_COUNT += 1


def _unregister_writer() -> None:
    global _ACTIVE_WRITER_COUNT
    with _ACTIVE_WRITER_LOCK:
        _ACTIVE_WRITER_COUNT = max(0, _ACTIVE_WRITER_COUNT - 1)

_EVENT_COLUMNS = (
    "sequence",
    "event_id",
    "campaign_id",
    "run_id",
    "task_id",
    "event_type",
    "transaction_time",
    "valid_time",
    "subject_id",
    "payload_hash",
    "payload_json",
    "blob_digest",
    "source_class",
    "disposition",
    "evaluator_identity",
    "idempotency_key",
    "previous_hash",
    "event_hash",
)


_SCHEMA = f"""
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    sequence INTEGER PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE,
    campaign_id TEXT,
    run_id TEXT,
    task_id TEXT,
    event_type TEXT NOT NULL,
    transaction_time TEXT NOT NULL,
    valid_time TEXT,
    subject_id TEXT,
    payload_hash TEXT NOT NULL,
    payload_json TEXT,
    blob_digest TEXT,
    source_class TEXT,
    disposition TEXT,
    evaluator_identity TEXT,
    idempotency_key TEXT UNIQUE,
    previous_hash TEXT NOT NULL,
    event_hash TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id),
    protocol_hash TEXT NOT NULL,
    source_commit TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    data_manifest_hash TEXT NOT NULL,
    evaluator_hash TEXT NOT NULL,
    policy_hash TEXT NOT NULL,
    seed_set_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id),
    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
    arm TEXT NOT NULL,
    task_id TEXT NOT NULL,
    seed INTEGER NOT NULL,
    parent_checkpoint TEXT,
    start_state TEXT NOT NULL,
    end_state TEXT,
    host_role TEXT NOT NULL,
    software_manifest_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id),
    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    task_id TEXT NOT NULL,
    parent_candidate_id TEXT REFERENCES candidates(candidate_id),
    mutation_family TEXT NOT NULL,
    patch_hash TEXT NOT NULL,
    requested_authority TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    model_hash TEXT NOT NULL,
    adapter_hash TEXT,
    candidate_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS verdicts (
    verdict_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id),
    candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
    receipt_id TEXT,
    correctness INTEGER,
    performance_json TEXT,
    hidden_test_set_hash TEXT NOT NULL,
    evaluator_revision TEXT NOT NULL,
    signed_receipt_hash TEXT
);
CREATE TABLE IF NOT EXISTS effect_receipts (
    request_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id),
    candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
    identity TEXT NOT NULL,
    normalized_action_hash TEXT NOT NULL,
    decision TEXT NOT NULL,
    policy_hash TEXT NOT NULL,
    sandbox_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    exit_status_class TEXT NOT NULL,
    output_hash TEXT,
    environment_diff_hash TEXT,
    signature TEXT NOT NULL,
    receipt_id TEXT
);
CREATE TABLE IF NOT EXISTS dependencies (
    dependency_id TEXT PRIMARY KEY,
    parent_id TEXT NOT NULL,
    child_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    insertion_event_id TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS corrections (
    correction_id TEXT PRIMARY KEY,
    superseded_id TEXT NOT NULL,
    replacement_id TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    correction_source TEXT NOT NULL,
    event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id)
);
CREATE TABLE IF NOT EXISTS retractions (
    retraction_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    retraction_source TEXT NOT NULL,
    event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id)
);
CREATE TABLE IF NOT EXISTS receipts (
    receipt_id TEXT PRIMARY KEY,
    sequence INTEGER NOT NULL UNIQUE,
    receipt_hash TEXT NOT NULL UNIQUE,
    receipt_type TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    previous_receipt_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    event_id TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
    last_completed_phase TEXT NOT NULL,
    last_durable_event_id TEXT NOT NULL,
    ledger_hash TEXT NOT NULL,
    projection_generation TEXT NOT NULL,
    artifact_manifest_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS blobs (
    digest TEXT PRIMARY KEY,
    media_type TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    visibility TEXT NOT NULL,
    logical_role TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS projection_queue (
    queue_id TEXT PRIMARY KEY,
    ledger_head_hash TEXT NOT NULL,
    projection_generation TEXT,
    reason_code TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS quarantines (
    quarantine_id TEXT PRIMARY KEY,
    reason_code TEXT NOT NULL,
    subject_id TEXT,
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL
);
INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', '{LEDGER_SCHEMA_VERSION}');
INSERT OR IGNORE INTO meta(key, value) VALUES ('ledger_head_hash', '{GENESIS_HASH}');
INSERT OR IGNORE INTO meta(key, value) VALUES ('quarantined', '0');

CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: events cannot be updated'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: events cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS campaigns_no_update BEFORE UPDATE ON campaigns
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: campaigns cannot be updated'); END;
CREATE TRIGGER IF NOT EXISTS campaigns_no_delete BEFORE DELETE ON campaigns
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: campaigns cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS runs_no_update BEFORE UPDATE ON runs
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: runs cannot be updated'); END;
CREATE TRIGGER IF NOT EXISTS runs_no_delete BEFORE DELETE ON runs
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: runs cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS candidates_no_update BEFORE UPDATE ON candidates
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: candidates cannot be updated'); END;
CREATE TRIGGER IF NOT EXISTS candidates_no_delete BEFORE DELETE ON candidates
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: candidates cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS verdicts_no_update BEFORE UPDATE ON verdicts
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: verdicts cannot be updated'); END;
CREATE TRIGGER IF NOT EXISTS verdicts_no_delete BEFORE DELETE ON verdicts
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: verdicts cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS effects_no_update BEFORE UPDATE ON effect_receipts
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: effect receipts cannot be updated'); END;
CREATE TRIGGER IF NOT EXISTS effects_no_delete BEFORE DELETE ON effect_receipts
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: effect receipts cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS dependencies_no_update BEFORE UPDATE ON dependencies
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: dependencies cannot be updated'); END;
CREATE TRIGGER IF NOT EXISTS dependencies_no_delete BEFORE DELETE ON dependencies
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: dependencies cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS corrections_no_update BEFORE UPDATE ON corrections
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: corrections cannot be updated'); END;
CREATE TRIGGER IF NOT EXISTS corrections_no_delete BEFORE DELETE ON corrections
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: corrections cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS retractions_no_update BEFORE UPDATE ON retractions
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: retractions cannot be updated'); END;
CREATE TRIGGER IF NOT EXISTS retractions_no_delete BEFORE DELETE ON retractions
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: retractions cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS receipts_no_update BEFORE UPDATE ON receipts
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: receipts cannot be updated'); END;
CREATE TRIGGER IF NOT EXISTS receipts_no_delete BEFORE DELETE ON receipts
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: receipts cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS checkpoints_no_update BEFORE UPDATE ON checkpoints
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: checkpoints cannot be updated'); END;
CREATE TRIGGER IF NOT EXISTS checkpoints_no_delete BEFORE DELETE ON checkpoints
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: checkpoints cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS blobs_no_update BEFORE UPDATE ON blobs
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: blobs cannot be updated'); END;
CREATE TRIGGER IF NOT EXISTS blobs_no_delete BEFORE DELETE ON blobs
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: blobs cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS projection_queue_no_update BEFORE UPDATE ON projection_queue
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: projection queue cannot be updated'); END;
CREATE TRIGGER IF NOT EXISTS projection_queue_no_delete BEFORE DELETE ON projection_queue
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: projection queue cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS quarantines_no_update BEFORE UPDATE ON quarantines
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: quarantines cannot be updated'); END;
CREATE TRIGGER IF NOT EXISTS quarantines_no_delete BEFORE DELETE ON quarantines
BEGIN SELECT RAISE(ABORT, 'append-only evidence table: quarantines cannot be deleted'); END;
"""


class BlobStore:
    """Private content-addressed blob layout used for large payloads."""

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)

    def put(self, data: bytes) -> str:
        digest = digest_for(data)
        target = self.root / "blobs" / "sha256" / digest[:2] / digest[2:4] / digest
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != data:
                raise IntegrityError(f"content-addressed blob collision at {target}")
            return digest
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}-{threading.get_ident()}")
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        target.chmod(0o444)
        return digest

    def get(self, digest: str) -> bytes:
        if not isinstance(digest, str) or len(digest) != 64:
            raise IntegrityError("blob digest is not a full SHA-256 value")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise IntegrityError("blob digest is not hexadecimal") from exc
        target = self.root / "blobs" / "sha256" / digest[:2] / digest[2:4] / digest
        try:
            data = target.read_bytes()
        except FileNotFoundError as exc:
            raise IntegrityError(f"content-addressed blob is missing at {target}") from exc
        except OSError as exc:
            raise IntegrityError(f"content-addressed blob cannot be read at {target}: {exc}") from exc
        if digest_for(data) != digest:
            raise IntegrityError(f"blob digest mismatch at {target}")
        return data

    def path_for(self, digest: str) -> Path:
        return self.root / "blobs" / "sha256" / digest[:2] / digest[2:4] / digest


class EvidenceLedger:
    """Single-writer append-only SQLite ledger with read-only replay support."""

    def __init__(
        self,
        path: Union[str, Path],
        *,
        mode: str = "writer",
        blob_root: Optional[Union[str, Path]] = None,
        create: bool = True,
        clock: Callable[[], str] = utc_now_iso,
    ) -> None:
        if mode not in {"writer", "read_only"}:
            raise ValueError("ledger mode must be 'writer' or 'read_only'")
        self.path = str(path)
        self.mode = mode
        self._writable = mode == "writer"
        self._mutex = threading.RLock()
        self._lock_handle: Any = None
        self._closed = False
        self._clock = clock
        if self.path != ":memory:":
            db_path = Path(self.path)
            if mode == "writer":
                if create:
                    db_path.parent.mkdir(parents=True, exist_ok=True)
                elif not db_path.exists():
                    raise LedgerError(f"ledger does not exist: {db_path}")
                self._acquire_writer_lock(db_path)
                self._conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
            else:
                if not db_path.exists():
                    raise LedgerError(f"ledger does not exist: {db_path}")
                uri = f"file:{db_path.absolute()}?mode=ro"
                self._conn = sqlite3.connect(uri, uri=True, isolation_level=None, check_same_thread=False)
        else:
            if mode == "read_only":
                raise LedgerError("read_only mode cannot open a private :memory: database")
            self._conn = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        if self._writable:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.executescript(_SCHEMA)
        else:
            self._conn.execute("PRAGMA query_only=ON")
        self.blob_store = BlobStore(blob_root) if blob_root is not None else None
        self._writer_registered = False
        if self._writable:
            _register_writer()
            self._writer_registered = True

    def _acquire_writer_lock(self, db_path: Path) -> None:
        lock_path = Path(str(db_path) + ".writer.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+")
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (ImportError, BlockingIOError, OSError) as exc:
            handle.close()
            raise LedgerBusyError(f"another process owns the ledger writer lock: {lock_path}") from exc
        self._lock_handle = handle

    def close(self) -> None:
        if self._closed:
            return
        self._conn.close()
        if self._lock_handle is not None:
            try:
                import fcntl

                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock_handle.close()
        if self._writer_registered:
            _unregister_writer()
            self._writer_registered = False
        self._closed = True

    def __enter__(self) -> "EvidenceLedger":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        """Expose the connection for read-only diagnostics and trigger tests."""

        return self._conn

    def _require_writer(self) -> None:
        if not self._writable:
            raise LedgerReadOnlyError("this ledger handle is read-only")
        if self._closed:
            raise LedgerError("ledger handle is closed")

    @contextmanager
    def _write_transaction(self) -> Iterator[sqlite3.Connection]:
        self._require_writer()
        with self._mutex:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def _prepare_payload(self, payload: Mapping[str, Any]) -> Tuple[str, Optional[str], Optional[str]]:
        encoded = canonical_bytes(payload)
        payload_hash = digest_for(encoded)
        if len(encoded) <= INLINE_PAYLOAD_LIMIT:
            return payload_hash, encoded.decode("utf-8"), None
        if self.blob_store is None:
            raise LedgerError(
                f"payload is {len(encoded)} bytes, above the {INLINE_PAYLOAD_LIMIT}-byte inline limit; configure blob_root"
            )
        digest = self.blob_store.put(encoded)
        return payload_hash, None, digest

    def _row_payload(self, row: sqlite3.Row) -> Optional[Dict[str, Any]]:
        if row["payload_json"] is not None:
            return json.loads(row["payload_json"])
        if row["blob_digest"] is not None and self.blob_store is not None:
            return json.loads(self.blob_store.get(row["blob_digest"]).decode("utf-8"))
        return None

    @staticmethod
    def _immutable_event_view(event: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            key: event.get(key)
            for key in (
                "event_id",
                "campaign_id",
                "run_id",
                "task_id",
                "event_type",
                "valid_time",
                "subject_id",
                "payload_hash",
                "payload_json",
                "blob_digest",
                "source_class",
                "disposition",
                "evaluator_identity",
                "idempotency_key",
            )
        }

    def _event_record_for_hash(self, values: Mapping[str, Any]) -> Dict[str, Any]:
        return {"schema_version": LEDGER_SCHEMA_VERSION, **{key: values.get(key) for key in _EVENT_COLUMNS[:-1]}}

    def _row_to_event(self, row: sqlite3.Row) -> Dict[str, Any]:
        event = {column: row[column] for column in _EVENT_COLUMNS}
        if event["blob_digest"] is not None:
            metadata = self.blob_metadata(event["blob_digest"])
            if metadata is not None:
                event["blob_media_type"] = metadata["media_type"]
        payload = self._row_payload(row)
        if payload is not None:
            event["payload"] = payload
        return event

    def _insert_event_tx(
        self,
        conn: sqlite3.Connection,
        *,
        event_type: str,
        payload: Mapping[str, Any],
        campaign_id: Optional[str] = None,
        run_id: Optional[str] = None,
        task_id: Optional[str] = None,
        valid_time: Optional[str] = None,
        subject_id: Optional[str] = None,
        source_class: Optional[str] = None,
        disposition: Optional[str] = None,
        evaluator_identity: Optional[str] = None,
        event_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        transaction_time: Optional[str] = None,
        blob_visibility: str = "private",
        blob_role: str = "event-payload",
        blob_media_type: str = "application/json",
        payload_hash: Optional[str] = None,
        payload_json: Optional[str] = None,
        blob_digest: Optional[str] = None,
        sequence: Optional[int] = None,
        previous_hash: Optional[str] = None,
        event_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        if payload_hash is None or (payload_json is None and blob_digest is None):
            payload_hash, payload_json, blob_digest = self._prepare_payload(payload)
        if blob_digest is not None and blob_visibility not in {"private", "public-eligible"}:
            raise LedgerError("blob visibility must be 'private' or 'public-eligible'")
        if not isinstance(blob_media_type, str) or not blob_media_type.strip():
            raise LedgerError("blob media type must be a non-empty string")
        immutable = {
            "event_type": event_type,
            "campaign_id": campaign_id,
            "run_id": run_id,
            "task_id": task_id,
            "valid_time": valid_time,
            "subject_id": subject_id,
            "payload_hash": payload_hash,
            "payload_json": payload_json,
            "blob_digest": blob_digest,
            "source_class": source_class,
            "disposition": disposition,
            "evaluator_identity": evaluator_identity,
            "idempotency_key": idempotency_key,
        }
        derived_event_id = content_id("evt", immutable)
        if event_id is not None and event_id != derived_event_id:
            raise LedgerError("event_id must be content-derived from immutable event fields")
        event_id = derived_event_id
        existing = conn.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
        if existing is not None:
            existing_event = self._row_to_event(existing)
            if self._immutable_event_view(existing_event) != self._immutable_event_view({**immutable, "event_id": event_id}):
                raise IdempotencyConflictError(f"event ID {event_id} was delivered with conflicting content")
            return existing_event
        if idempotency_key is not None:
            existing = conn.execute("SELECT * FROM events WHERE idempotency_key=?", (idempotency_key,)).fetchone()
            if existing is not None:
                existing_event = self._row_to_event(existing)
                if self._immutable_event_view(existing_event) != self._immutable_event_view({**immutable, "event_id": event_id}):
                    raise IdempotencyConflictError(f"event idempotency key {idempotency_key} was reused")
                return existing_event
        if sequence is None:
            sequence = int(conn.execute("SELECT COALESCE(MAX(sequence), 0) + 1 FROM events").fetchone()[0])
        expected_previous = conn.execute("SELECT event_hash FROM events WHERE sequence=?", (sequence - 1,)).fetchone()
        expected_previous_hash = expected_previous[0] if expected_previous is not None else GENESIS_HASH
        if previous_hash is None:
            previous_hash = expected_previous_hash
        if previous_hash != expected_previous_hash:
            raise IntegrityError("event previous hash does not match the ledger head")
        transaction_time = transaction_time or self._clock()
        values = {
            "sequence": sequence,
            "event_id": event_id,
            "campaign_id": campaign_id,
            "run_id": run_id,
            "task_id": task_id,
            "event_type": event_type,
            "transaction_time": transaction_time,
            "valid_time": valid_time,
            "subject_id": subject_id,
            "payload_hash": payload_hash,
            "payload_json": payload_json,
            "blob_digest": blob_digest,
            "source_class": source_class,
            "disposition": disposition,
            "evaluator_identity": evaluator_identity,
            "idempotency_key": idempotency_key,
            "previous_hash": previous_hash,
        }
        expected_event_hash = chain_digest(previous_hash, self._event_record_for_hash(values))
        if event_hash is not None and event_hash != expected_event_hash:
            raise IntegrityError("event hash does not match canonical event content")
        event_hash = expected_event_hash
        conn.execute(
            """INSERT INTO events(sequence,event_id,campaign_id,run_id,task_id,event_type,
               transaction_time,valid_time,subject_id,payload_hash,payload_json,blob_digest,
               source_class,disposition,evaluator_identity,idempotency_key,previous_hash,event_hash)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            tuple(values[column] for column in _EVENT_COLUMNS[:-1]) + (event_hash,),
        )
        if blob_digest is not None:
            if payload_json is None:
                if self.blob_store is None:
                    raise LedgerError("blob-backed event requires a configured blob store")
                encoded_payload = canonical_bytes(payload)
                if digest_for(encoded_payload) != payload_hash or digest_for(encoded_payload) != blob_digest:
                    raise IntegrityError("blob-backed event payload does not match its declared digest")
                target = self.blob_store.path_for(blob_digest)
                if target.exists():
                    blob_size = len(self.blob_store.get(blob_digest))
                else:
                    materialized_digest = self.blob_store.put(encoded_payload)
                    if materialized_digest != blob_digest:
                        raise IntegrityError("replayed blob content address does not match the event")
                    blob_size = len(encoded_payload)
            else:
                blob_size = len(canonical_bytes(payload))
            conn.execute(
                "INSERT OR IGNORE INTO blobs(digest,media_type,byte_size,visibility,logical_role) VALUES(?,?,?,?,?)",
                (blob_digest, blob_media_type, blob_size, blob_visibility, blob_role),
            )
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('ledger_head_hash',?)", (event_hash,))
        row = conn.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
        return self._row_to_event(row)

    def append_event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        campaign_id: Optional[str] = None,
        run_id: Optional[str] = None,
        task_id: Optional[str] = None,
        valid_time: Optional[str] = None,
        subject_id: Optional[str] = None,
        source_class: Optional[str] = None,
        disposition: Optional[str] = None,
        evaluator_identity: Optional[str] = None,
        event_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        transaction_time: Optional[str] = None,
        blob_visibility: str = "private",
        blob_role: str = "event-payload",
        blob_media_type: str = "application/json",
    ) -> Dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise LedgerError("event payload must be a mapping")
        with self._write_transaction() as conn:
            return self._insert_event_tx(
                conn,
                event_type=event_type,
                payload=dict(payload),
                campaign_id=campaign_id,
                run_id=run_id,
                task_id=task_id,
                valid_time=valid_time,
                subject_id=subject_id,
                source_class=source_class,
                disposition=disposition,
                evaluator_identity=evaluator_identity,
                event_id=event_id,
                idempotency_key=idempotency_key,
                transaction_time=transaction_time,
                blob_visibility=blob_visibility,
                blob_role=blob_role,
                blob_media_type=blob_media_type,
            )

    append_evidence_event = append_event

    # ---- required logical records -----------------------------------------
    def create_campaign(
        self,
        campaign_id: str,
        *,
        protocol_hash: str,
        source_commit: str,
        model_revision: str,
        data_manifest_hash: str,
        evaluator_hash: str,
        policy_hash: str,
        seed_set: Sequence[int],
        created_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "campaign_id": campaign_id,
            "protocol_hash": protocol_hash,
            "source_commit": source_commit,
            "model_revision": model_revision,
            "data_manifest_hash": data_manifest_hash,
            "evaluator_hash": evaluator_hash,
            "policy_hash": policy_hash,
            "seed_set": list(seed_set),
            "created_at": created_at or self._clock(),
        }
        with self._write_transaction() as conn:
            existing = conn.execute("SELECT * FROM campaigns WHERE campaign_id=?", (campaign_id,)).fetchone()
            if existing is not None:
                existing_payload = {
                    "campaign_id": existing["campaign_id"],
                    "protocol_hash": existing["protocol_hash"],
                    "source_commit": existing["source_commit"],
                    "model_revision": existing["model_revision"],
                    "data_manifest_hash": existing["data_manifest_hash"],
                    "evaluator_hash": existing["evaluator_hash"],
                    "policy_hash": existing["policy_hash"],
                    "seed_set": json.loads(existing["seed_set_json"]),
                    "created_at": existing["created_at"],
                }
                comparable_payload = dict(payload)
                comparable_existing = dict(existing_payload)
                if created_at is None:
                    comparable_payload.pop("created_at")
                    comparable_existing.pop("created_at")
                if canonical_json(comparable_payload) != canonical_json(comparable_existing):
                    raise IdempotencyConflictError(f"campaign {campaign_id} was delivered with conflicting content")
                return dict(existing)
            event = self._insert_event_tx(conn, event_type="CAMPAIGN", payload=payload, campaign_id=campaign_id, subject_id=campaign_id, disposition="OBSERVED")
            conn.execute(
                """INSERT INTO campaigns(campaign_id,event_id,protocol_hash,source_commit,model_revision,
                   data_manifest_hash,evaluator_hash,policy_hash,seed_set_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    campaign_id,
                    event["event_id"],
                    protocol_hash,
                    source_commit,
                    model_revision,
                    data_manifest_hash,
                    evaluator_hash,
                    policy_hash,
                    canonical_json(list(seed_set)),
                    payload["created_at"],
                ),
            )
            return dict(conn.execute("SELECT * FROM campaigns WHERE campaign_id=?", (campaign_id,)).fetchone())

    def create_run(
        self,
        run_id: str,
        *,
        campaign_id: str,
        arm: str,
        task_id: str,
        seed: int,
        parent_checkpoint: Optional[str],
        start_state: str,
        host_role: str,
        software_manifest_hash: str,
        end_state: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "run_id": run_id,
            "campaign_id": campaign_id,
            "arm": arm,
            "task_id": task_id,
            "seed": seed,
            "parent_checkpoint": parent_checkpoint,
            "start_state": start_state,
            "end_state": end_state,
            "host_role": host_role,
            "software_manifest_hash": software_manifest_hash,
            "created_at": created_at or self._clock(),
        }
        with self._write_transaction() as conn:
            existing = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if existing is not None:
                existing_payload = {
                    "run_id": existing["run_id"],
                    "campaign_id": existing["campaign_id"],
                    "arm": existing["arm"],
                    "task_id": existing["task_id"],
                    "seed": existing["seed"],
                    "parent_checkpoint": existing["parent_checkpoint"],
                    "start_state": existing["start_state"],
                    "end_state": existing["end_state"],
                    "host_role": existing["host_role"],
                    "software_manifest_hash": existing["software_manifest_hash"],
                    "created_at": existing["created_at"],
                }
                comparable_payload = dict(payload)
                comparable_existing = dict(existing_payload)
                if created_at is None:
                    comparable_payload.pop("created_at")
                    comparable_existing.pop("created_at")
                if canonical_json(comparable_existing) != canonical_json(comparable_payload):
                    raise IdempotencyConflictError(f"run {run_id} was delivered with conflicting content")
                return dict(existing)
            if conn.execute("SELECT 1 FROM campaigns WHERE campaign_id=?", (campaign_id,)).fetchone() is None:
                raise UnknownReferenceError(f"unknown campaign: {campaign_id}")
            event = self._insert_event_tx(conn, event_type="RUN", payload=payload, campaign_id=campaign_id, run_id=run_id, task_id=task_id, subject_id=run_id, disposition="OBSERVED")
            conn.execute(
                """INSERT INTO runs(run_id,event_id,campaign_id,arm,task_id,seed,parent_checkpoint,
                   start_state,end_state,host_role,software_manifest_hash,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    event["event_id"],
                    campaign_id,
                    arm,
                    task_id,
                    seed,
                    parent_checkpoint,
                    start_state,
                    end_state,
                    host_role,
                    software_manifest_hash,
                    payload["created_at"],
                ),
            )
            return dict(conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone())

    append_campaign = create_campaign
    append_run = create_run

    def append_candidate(
        self,
        candidate_id: str,
        *,
        campaign_id: str,
        run_id: str,
        task_id: str,
        parent_candidate_id: Optional[str],
        mutation_family: str,
        patch_hash: str,
        requested_authority: str,
        prompt_hash: str,
        model_hash: str,
        adapter_hash: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "candidate_id": candidate_id,
            "campaign_id": campaign_id,
            "run_id": run_id,
            "task_id": task_id,
            "parent_candidate_id": parent_candidate_id,
            "mutation_family": mutation_family,
            "patch_hash": patch_hash,
            "requested_authority": requested_authority,
            "prompt_hash": prompt_hash,
            "model_hash": model_hash,
            "adapter_hash": adapter_hash,
            "metadata": dict(metadata or {}),
        }
        with self._write_transaction() as conn:
            existing = conn.execute("SELECT * FROM candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
            if existing is not None:
                if existing["candidate_json"] != canonical_json(payload):
                    raise IdempotencyConflictError(f"candidate {candidate_id} was delivered with conflicting content")
                return dict(existing)
            if conn.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone() is None:
                raise UnknownReferenceError(f"unknown run: {run_id}")
            event = self._insert_event_tx(
                conn,
                event_type="CANDIDATE",
                payload=payload,
                campaign_id=campaign_id,
                run_id=run_id,
                task_id=task_id,
                subject_id=candidate_id,
                disposition="OBSERVED",
            )
            conn.execute(
                """INSERT INTO candidates(candidate_id,event_id,campaign_id,run_id,task_id,parent_candidate_id,
                   mutation_family,patch_hash,requested_authority,prompt_hash,model_hash,adapter_hash,candidate_json)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    candidate_id,
                    event["event_id"],
                    campaign_id,
                    run_id,
                    task_id,
                    parent_candidate_id,
                    mutation_family,
                    patch_hash,
                    requested_authority,
                    prompt_hash,
                    model_hash,
                    adapter_hash,
                    canonical_json(payload),
                ),
            )
            return dict(conn.execute("SELECT * FROM candidates WHERE candidate_id=?", (candidate_id,)).fetchone())

    def append_verdict(
        self,
        verdict_id: str,
        *,
        candidate_id: str,
        correctness: Optional[bool],
        performance: Optional[Mapping[str, Any]],
        hidden_test_set_hash: str,
        evaluator_revision: str,
        receipt_id: Optional[str] = None,
        signed_receipt_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self._known_id(candidate_id):
            raise UnknownReferenceError(f"unknown candidate: {candidate_id}")
        linked_receipt = self.receipt_by_id(receipt_id) if receipt_id is not None else None
        if linked_receipt is not None:
            receipt = linked_receipt["receipt"]
            if receipt.get("receipt_type") != "VERDICT" or receipt.get("candidate_id") != candidate_id:
                raise ReceiptVerificationError("linked receipt is not a verdict for this candidate")
            if signed_receipt_hash is not None and signed_receipt_hash != receipt_hash(receipt):
                raise ReceiptVerificationError("signed verdict hash does not match the linked receipt")
        payload = {
            "verdict_id": verdict_id,
            "candidate_id": candidate_id,
            "correctness": correctness,
            "performance": dict(performance or {}),
            "hidden_test_set_hash": hidden_test_set_hash,
            "evaluator_revision": evaluator_revision,
            "receipt_id": receipt_id,
            "signed_receipt_hash": signed_receipt_hash,
        }
        with self._write_transaction() as conn:
            existing = conn.execute("SELECT * FROM verdicts WHERE verdict_id=?", (verdict_id,)).fetchone()
            if existing is not None:
                existing_payload = {
                    "verdict_id": existing["verdict_id"],
                    "candidate_id": existing["candidate_id"],
                    "correctness": None if existing["correctness"] is None else bool(existing["correctness"]),
                    "performance": json.loads(existing["performance_json"]),
                    "hidden_test_set_hash": existing["hidden_test_set_hash"],
                    "evaluator_revision": existing["evaluator_revision"],
                    "receipt_id": existing["receipt_id"],
                    "signed_receipt_hash": existing["signed_receipt_hash"],
                }
                if canonical_json(existing_payload) != canonical_json(payload):
                    raise IdempotencyConflictError(f"verdict {verdict_id} was delivered with conflicting content")
                return dict(existing)
            candidate = conn.execute("SELECT * FROM candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
            event = self._insert_event_tx(
                conn,
                event_type="VERDICT",
                payload=payload,
                campaign_id=candidate["campaign_id"],
                run_id=candidate["run_id"],
                task_id=candidate["task_id"],
                subject_id=candidate_id,
                source_class="FROZEN_EVALUATOR",
                disposition="VERIFIED" if receipt_id is not None and self.receipt_by_id(receipt_id) is not None else "OBSERVED",
            )
            conn.execute(
                """INSERT INTO verdicts(verdict_id,event_id,candidate_id,receipt_id,correctness,performance_json,
                   hidden_test_set_hash,evaluator_revision,signed_receipt_hash)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    verdict_id,
                    event["event_id"],
                    candidate_id,
                    receipt_id,
                    None if correctness is None else int(bool(correctness)),
                    canonical_json(dict(performance or {})),
                    hidden_test_set_hash,
                    evaluator_revision,
                    signed_receipt_hash,
                ),
            )
            return dict(conn.execute("SELECT * FROM verdicts WHERE verdict_id=?", (verdict_id,)).fetchone())

    def append_effect_receipt(
        self,
        request_id: str,
        *,
        candidate_id: str,
        identity: str,
        normalized_action_hash: str,
        decision: str,
        policy_hash: str,
        sandbox_id: str,
        started_at: str,
        finished_at: str,
        exit_status_class: str,
        output_hash: Optional[str],
        environment_diff_hash: Optional[str],
        signature: str,
        receipt_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self._known_id(candidate_id):
            raise UnknownReferenceError(f"unknown candidate: {candidate_id}")
        linked_receipt = self.receipt_by_id(receipt_id) if receipt_id is not None else None
        if linked_receipt is not None:
            receipt = linked_receipt["receipt"]
            checks = {
                "receipt_type": (receipt.get("receipt_type"), "EFFECT"),
                "candidate_id": (receipt.get("candidate_id"), candidate_id),
                "request_id": (receipt.get("request_id"), request_id),
                "decision": (receipt.get("decision"), decision),
                "normalized_action_hash": (receipt.get("normalized_action_hash"), normalized_action_hash),
                "policy_digest": (receipt.get("policy_digest"), policy_hash),
                "sandbox_id": (receipt.get("sandbox_id"), sandbox_id),
                "started_at": (receipt.get("started_at"), started_at),
                "finished_at": (receipt.get("finished_at"), finished_at),
                "exit_status_class": (receipt.get("exit_status_class"), exit_status_class),
                "signature": (signature, receipt.get("signature")),
            }
            if output_hash is not None:
                checks["output_digest"] = (receipt.get("output_digest"), output_hash)
            if environment_diff_hash is not None:
                checks["environment_diff_digest"] = (receipt.get("environment_diff_digest"), environment_diff_hash)
            mismatches = [field for field, (actual, expected) in checks.items() if actual != expected]
            if mismatches:
                raise ReceiptVerificationError(f"linked effect receipt fields disagree: {', '.join(mismatches)}")
        payload = {
            "request_id": request_id,
            "candidate_id": candidate_id,
            "identity": identity,
            "normalized_action_hash": normalized_action_hash,
            "decision": decision,
            "policy_hash": policy_hash,
            "sandbox_id": sandbox_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_status_class": exit_status_class,
            "output_hash": output_hash,
            "environment_diff_hash": environment_diff_hash,
            "signature": signature,
            "receipt_id": receipt_id,
        }
        with self._write_transaction() as conn:
            existing = conn.execute("SELECT * FROM effect_receipts WHERE request_id=?", (request_id,)).fetchone()
            if existing is not None:
                existing_payload = {
                    "request_id": existing["request_id"],
                    "candidate_id": existing["candidate_id"],
                    "identity": existing["identity"],
                    "normalized_action_hash": existing["normalized_action_hash"],
                    "decision": existing["decision"],
                    "policy_hash": existing["policy_hash"],
                    "sandbox_id": existing["sandbox_id"],
                    "started_at": existing["started_at"],
                    "finished_at": existing["finished_at"],
                    "exit_status_class": existing["exit_status_class"],
                    "output_hash": existing["output_hash"],
                    "environment_diff_hash": existing["environment_diff_hash"],
                    "signature": existing["signature"],
                    "receipt_id": existing["receipt_id"],
                }
                if canonical_json(existing_payload) != canonical_json(payload):
                    raise IdempotencyConflictError(f"effect request {request_id} was delivered with conflicting content")
                return dict(existing)
            candidate = conn.execute("SELECT * FROM candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
            event = self._insert_event_tx(
                conn,
                event_type="EFFECT_RECEIPT",
                payload=payload,
                campaign_id=candidate["campaign_id"],
                run_id=candidate["run_id"],
                task_id=candidate["task_id"],
                subject_id=candidate_id,
                source_class="FROZEN_EVALUATOR",
                disposition="VERIFIED" if receipt_id is not None and self.receipt_by_id(receipt_id) is not None else "OBSERVED",
            )
            conn.execute(
                """INSERT INTO effect_receipts(request_id,event_id,candidate_id,identity,normalized_action_hash,
                   decision,policy_hash,sandbox_id,started_at,finished_at,exit_status_class,output_hash,
                   environment_diff_hash,signature,receipt_id)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    request_id,
                    event["event_id"],
                    candidate_id,
                    identity,
                    normalized_action_hash,
                    decision,
                    policy_hash,
                    sandbox_id,
                    started_at,
                    finished_at,
                    exit_status_class,
                    output_hash,
                    environment_diff_hash,
                    signature,
                    receipt_id,
                ),
            )
            return dict(conn.execute("SELECT * FROM effect_receipts WHERE request_id=?", (request_id,)).fetchone())

    # ---- dependencies and lifecycle --------------------------------------
    def _known_id(self, identifier: str) -> bool:
        return self._conn.execute(
            """SELECT 1 WHERE EXISTS(SELECT 1 FROM events WHERE event_id=? OR subject_id=?)
               OR EXISTS(SELECT 1 FROM candidates WHERE candidate_id=?)
               OR EXISTS(SELECT 1 FROM receipts WHERE receipt_id=?)""",
            (identifier, identifier, identifier, identifier),
        ).fetchone() is not None

    def _known_id_tx(self, conn: sqlite3.Connection, identifier: str) -> bool:
        return conn.execute(
            """SELECT 1 WHERE EXISTS(SELECT 1 FROM events WHERE event_id=? OR subject_id=?)
               OR EXISTS(SELECT 1 FROM candidates WHERE candidate_id=?)
               OR EXISTS(SELECT 1 FROM receipts WHERE receipt_id=?)""",
            (identifier, identifier, identifier, identifier),
        ).fetchone() is not None

    def _identifier_aliases(self, identifier: str) -> set[str]:
        """Return the closed event/subject/candidate/receipt identity class.

        Lifecycle rows are history, not aliases for the entity they target. A
        retraction addressed by candidate ID therefore invalidates the
        candidate event and candidate ID without invalidating the retraction
        event itself.
        """

        if not isinstance(identifier, str) or not identifier:
            return set()
        aliases: set[str] = {identifier}
        pending = [identifier]
        processed: set[str] = set()
        while pending:
            value = pending.pop()
            if value in processed:
                continue
            processed.add(value)
            found_typed_row = False
            for row in self._conn.execute(
                "SELECT candidate_id,event_id FROM candidates WHERE candidate_id=? OR event_id=?",
                (value, value),
            ):
                found_typed_row = True
                for alias in (row["candidate_id"], row["event_id"]):
                    if alias and alias not in aliases:
                        aliases.add(alias)
                        pending.append(alias)
            for row in self._conn.execute("SELECT event_id,subject_id,event_type FROM events WHERE event_id=?", (value,)):
                found_typed_row = True
                event_aliases = (row["event_id"],) if row["event_type"] == "RECEIPT" else (row["event_id"], row["subject_id"])
                for alias in event_aliases:
                    if alias and alias not in aliases:
                        aliases.add(alias)
                        pending.append(alias)
            for row in self._conn.execute(
                "SELECT receipt_id,event_id,candidate_id FROM receipts WHERE receipt_id=? OR event_id=?",
                (value, value),
            ):
                found_typed_row = True
                for alias in (row["receipt_id"], row["event_id"]):
                    if alias and alias not in aliases:
                        aliases.add(alias)
                        pending.append(alias)
            if not found_typed_row:
                # A bare subject ID may identify one or more ordinary evidence
                # events. Do not pull CORRECTION/RETRACTION history rows into
                # the entity alias class.
                for row in self._conn.execute(
                    """SELECT event_id,subject_id FROM events
                       WHERE subject_id=? AND event_type NOT IN ('CORRECTION','RETRACTION')""",
                    (value,),
                ):
                    for alias in (row["event_id"], row["subject_id"]):
                        if alias and alias not in aliases:
                            aliases.add(alias)
                            pending.append(alias)
        return aliases

    def _event_id_for_identifier(self, identifier: str) -> Optional[str]:
        candidate = self._conn.execute(
            "SELECT event_id FROM candidates WHERE candidate_id=? OR event_id=? ORDER BY candidate_id LIMIT 1",
            (identifier, identifier),
        ).fetchone()
        if candidate is not None:
            return str(candidate[0])
        event = self._conn.execute("SELECT event_id FROM events WHERE event_id=?", (identifier,)).fetchone()
        if event is not None:
            return str(event[0])
        receipt = self._conn.execute(
            "SELECT event_id FROM receipts WHERE receipt_id=? OR event_id=? ORDER BY sequence DESC LIMIT 1",
            (identifier, identifier),
        ).fetchone()
        if receipt is not None:
            return str(receipt[0])
        event = self._conn.execute(
            """SELECT event_id FROM events
               WHERE subject_id=? AND event_type NOT IN ('CORRECTION','RETRACTION')
               ORDER BY sequence DESC LIMIT 1""",
            (identifier,),
        ).fetchone()
        return str(event[0]) if event is not None else None

    @staticmethod
    def _would_cycle(edges: Sequence[Tuple[str, str]], parent_id: str, child_id: str) -> bool:
        adjacency: Dict[str, List[str]] = {}
        for parent, child in edges:
            adjacency.setdefault(parent, []).append(child)
        stack = [child_id]
        seen = set()
        while stack:
            current = stack.pop()
            if current == parent_id:
                return True
            if current in seen:
                continue
            seen.add(current)
            stack.extend(adjacency.get(current, []))
        return False

    def append_dependency(
        self,
        parent_id: str,
        child_id: str,
        *,
        edge_type: str,
        insertion_event_id: Optional[str] = None,
        campaign_id: Optional[str] = None,
        run_id: Optional[str] = None,
        task_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        dependency_payload = {"parent_id": parent_id, "child_id": child_id, "edge_type": edge_type}
        dependency_id = content_id("dep", dependency_payload)
        with self._write_transaction() as conn:
            existing = conn.execute("SELECT * FROM dependencies WHERE dependency_id=?", (dependency_id,)).fetchone()
            if existing is not None:
                return dict(existing)
            if not self._known_id_tx(conn, parent_id):
                raise UnknownReferenceError(f"unknown dependency parent: {parent_id}")
            if not self._known_id_tx(conn, child_id):
                raise UnknownReferenceError(f"unknown dependency child: {child_id}")
            rows = conn.execute("SELECT parent_id,child_id FROM dependencies").fetchall()
            if self._would_cycle([(row[0], row[1]) for row in rows], parent_id, child_id):
                raise DependencyCycleError(f"dependency {parent_id} -> {child_id} would create a cycle")
            event = self._insert_event_tx(
                conn,
                event_type="DEPENDENCY",
                payload=dependency_payload,
                campaign_id=campaign_id,
                run_id=run_id,
                task_id=task_id,
                subject_id=child_id,
                source_class="FROZEN_PROTOCOL",
                disposition="OBSERVED",
                event_id=insertion_event_id,
                idempotency_key=idempotency_key,
            )
            if event["event_id"] != insertion_event_id and insertion_event_id is not None:
                raise LedgerError("insertion_event_id does not match dependency event content")
            conn.execute(
                "INSERT INTO dependencies(dependency_id,parent_id,child_id,edge_type,insertion_event_id) VALUES(?,?,?,?,?)",
                (dependency_id, parent_id, child_id, edge_type, event["event_id"]),
            )
            return dict(conn.execute("SELECT * FROM dependencies WHERE dependency_id=?", (dependency_id,)).fetchone())

    add_dependency = append_dependency

    def append_correction(
        self,
        superseded_event_id: str,
        replacement_event_id: str,
        *,
        reason_code: str,
        correction_source: str,
        campaign_id: Optional[str] = None,
        run_id: Optional[str] = None,
        task_id: Optional[str] = None,
        effective_after_attempt: int = 0,
        authorizing_receipt_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "superseded_event_id": superseded_event_id,
            "replacement_event_id": replacement_event_id,
            "reason_code": reason_code,
            "correction_source": correction_source,
            "effective_after_attempt": effective_after_attempt,
            "authorizing_receipt_id": authorizing_receipt_id,
        }
        correction_id = content_id("correction", payload)
        with self._write_transaction() as conn:
            if not self._known_id_tx(conn, superseded_event_id) or not self._known_id_tx(conn, replacement_event_id):
                raise UnknownReferenceError("correction references an unknown event")
            existing = conn.execute("SELECT * FROM corrections WHERE correction_id=?", (correction_id,)).fetchone()
            if existing is not None:
                return dict(existing)
            event = self._insert_event_tx(
                conn,
                event_type="CORRECTION",
                payload=payload,
                campaign_id=campaign_id,
                run_id=run_id,
                task_id=task_id,
                subject_id=superseded_event_id,
                source_class=correction_source,
                disposition="OBSERVED",
            )
            conn.execute(
                """INSERT INTO corrections(correction_id,superseded_id,replacement_id,reason_code,
                   correction_source,event_id) VALUES(?,?,?,?,?,?)""",
                (correction_id, superseded_event_id, replacement_event_id, reason_code, correction_source, event["event_id"]),
            )
            return dict(conn.execute("SELECT * FROM corrections WHERE correction_id=?", (correction_id,)).fetchone())

    record_correction = append_correction

    def append_retraction(
        self,
        subject_id: str,
        *,
        reason_code: str,
        retraction_source: str,
        campaign_id: Optional[str] = None,
        run_id: Optional[str] = None,
        task_id: Optional[str] = None,
        effective_after_attempt: int = 0,
        authorizing_receipt_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "subject_id": subject_id,
            "reason_code": reason_code,
            "retraction_source": retraction_source,
            "effective_after_attempt": effective_after_attempt,
            "authorizing_receipt_id": authorizing_receipt_id,
        }
        retraction_id = content_id("retraction", payload)
        with self._write_transaction() as conn:
            if not self._known_id_tx(conn, subject_id):
                raise UnknownReferenceError(f"retraction references an unknown subject: {subject_id}")
            existing = conn.execute("SELECT * FROM retractions WHERE retraction_id=?", (retraction_id,)).fetchone()
            if existing is not None:
                return dict(existing)
            event = self._insert_event_tx(
                conn,
                event_type="RETRACTION",
                payload=payload,
                campaign_id=campaign_id,
                run_id=run_id,
                task_id=task_id,
                subject_id=subject_id,
                source_class=retraction_source,
                disposition="OBSERVED",
            )
            conn.execute(
                "INSERT INTO retractions(retraction_id,subject_id,reason_code,retraction_source,event_id) VALUES(?,?,?,?,?)",
                (retraction_id, subject_id, reason_code, retraction_source, event["event_id"]),
            )
            return dict(conn.execute("SELECT * FROM retractions WHERE retraction_id=?", (retraction_id,)).fetchone())

    record_retraction = append_retraction

    def _invalid_roots(self) -> set[str]:
        roots = {row[0] for row in self._conn.execute("SELECT superseded_id FROM corrections")}
        roots.update(row[0] for row in self._conn.execute("SELECT subject_id FROM retractions"))
        aliases: set[str] = set()
        for root in roots:
            aliases.update(self._identifier_aliases(root))
        return aliases

    def _stale_nodes(self) -> set[str]:
        adjacency: Dict[str, List[str]] = {}
        for row in self._conn.execute("SELECT parent_id,child_id FROM dependencies"):
            parent_aliases = self._identifier_aliases(row[0])
            child_aliases = self._identifier_aliases(row[1])
            for parent in parent_aliases:
                adjacency.setdefault(parent, []).extend(child_aliases)
        roots = self._invalid_roots()
        stale: set[str] = set()
        stack = list(roots)
        while stack:
            current = stack.pop()
            for child in adjacency.get(current, []):
                if child not in stale:
                    stale.add(child)
                    stack.append(child)
        return stale

    def event_disposition(self, event_id: str) -> str:
        if not self._known_id(event_id):
            raise UnknownReferenceError(f"unknown event or subject: {event_id}")
        invalid = self._invalid_roots()
        stale = self._stale_nodes()
        resolved_event_id = self._event_id_for_identifier(event_id)
        if event_id in invalid or (resolved_event_id is not None and resolved_event_id in invalid):
            return RETRACTED
        if event_id in stale or (resolved_event_id is not None and resolved_event_id in stale):
            return STALE_DEPENDENT
        row = self._conn.execute("SELECT disposition FROM events WHERE event_id=?", (resolved_event_id,)).fetchone() if resolved_event_id else None
        value = row[0] if row else None
        return str(value or "OBSERVED")

    def stale_dependents(self, subject_id: str) -> List[str]:
        adjacency: Dict[str, List[str]] = {}
        for row in self._conn.execute("SELECT parent_id,child_id FROM dependencies"):
            for parent in self._identifier_aliases(row[0]):
                adjacency.setdefault(parent, []).extend(self._identifier_aliases(row[1]))
        result: set[str] = set()
        stack = list(self._identifier_aliases(subject_id))
        while stack:
            current = stack.pop()
            for child in adjacency.get(current, []):
                if child not in result:
                    result.add(child)
                    stack.append(child)
        return sorted(result)

    def candidate_disposition(self, candidate_id: str) -> str:
        candidate = self._conn.execute("SELECT * FROM candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
        if candidate is None:
            raise UnknownReferenceError(f"unknown candidate: {candidate_id}")
        event_id = candidate["event_id"]
        base = self.event_disposition(event_id)
        if base == RETRACTED:
            return RETRACTED
        if base == STALE_DEPENDENT:
            return STALE_DEPENDENT
        invalid = self._invalid_roots()
        stale = self._stale_nodes()
        verdicts = self._conn.execute("SELECT * FROM verdicts WHERE candidate_id=? ORDER BY rowid", (candidate_id,)).fetchall()
        receipt_verdicts = self._conn.execute(
            "SELECT payload_json FROM receipts WHERE candidate_id=? AND receipt_type='VERDICT' ORDER BY sequence", (candidate_id,)
        ).fetchall()
        authorities = self._conn.execute(
            "SELECT payload_json FROM receipts WHERE candidate_id=? AND receipt_type='AUTHORITY' ORDER BY sequence", (candidate_id,)
        ).fetchall()
        effects = self._conn.execute(
            "SELECT payload_json FROM receipts WHERE candidate_id=? AND receipt_type='EFFECT' ORDER BY sequence", (candidate_id,)
        ).fetchall()
        authority_decisions = [json.loads(row[0]).get("decision") for row in authorities]
        effect_receipts = [json.loads(row[0]) for row in effects]
        effect_decisions = [receipt.get("decision") for receipt in effect_receipts]
        linked_verdict_decisions = []
        for row in verdicts:
            if row["event_id"] in invalid or row["event_id"] in stale:
                return STALE_DEPENDENT
            if row["receipt_id"] is not None:
                if row["receipt_id"] in invalid or row["receipt_id"] in stale:
                    return STALE_DEPENDENT
                receipt = self.receipt_by_id(row["receipt_id"])
                if receipt is not None:
                    linked_verdict_decisions.append(receipt["receipt"].get("decision"))
        effect_rows = self._conn.execute("SELECT * FROM effect_receipts WHERE candidate_id=? ORDER BY rowid", (candidate_id,)).fetchall()
        if any(row["event_id"] in invalid or row["event_id"] in stale for row in effect_rows):
            return STALE_DEPENDENT
        if any(row["receipt_id"] is not None and (row["receipt_id"] in invalid or row["receipt_id"] in stale) for row in effect_rows):
            return STALE_DEPENDENT
        effect_row_decisions = [row["decision"] for row in effect_rows]
        if "DENY" in authority_decisions or "DENY" in effect_decisions or "DENY" in effect_row_decisions:
            return REJECTED
        if not verdicts and not receipt_verdicts:
            return ABSTAINED
        if any(row["receipt_id"] is None or self.receipt_by_id(row["receipt_id"]) is None for row in verdicts):
            return ABSTAINED
        if not verdicts:
            if any(json.loads(row[0]).get("decision") == "FAIL" for row in receipt_verdicts):
                return REJECTED
            if any(json.loads(row[0]).get("decision") != "PASS" for row in receipt_verdicts):
                return ABSTAINED
        if any(row["correctness"] == 0 for row in verdicts):
            return REJECTED
        if any(row["correctness"] is None for row in verdicts):
            return ABSTAINED
        if "FAIL" in linked_verdict_decisions:
            return REJECTED
        if any(decision not in {"PASS"} for decision in linked_verdict_decisions):
            return ABSTAINED
        requires_effect = candidate["requested_authority"] not in {"NONE", "READ", "READ_ONLY"}
        if requires_effect and not authority_decisions:
            return ABSTAINED
        if requires_effect and not effect_rows:
            return ABSTAINED
        if any(row["receipt_id"] is None or self.receipt_by_id(row["receipt_id"]) is None for row in effect_rows):
            return ABSTAINED
        if requires_effect:
            linked_effect_ids = {row["receipt_id"] for row in effect_rows}
            linked_effects = [receipt for receipt in effect_receipts if receipt["receipt_id"] in linked_effect_ids]
            if not linked_effects:
                return ABSTAINED
            if any(receipt["decision"] == "ERROR" for receipt in linked_effects):
                return ABSTAINED
        return PROMOTED

    def current_valid_events(self) -> List[Dict[str, Any]]:
        invalid = self._invalid_roots()
        stale = self._stale_nodes()
        result = []
        for row in self._conn.execute("SELECT * FROM events ORDER BY sequence"):
            event = self._row_to_event(row)
            if event["event_id"] in invalid or event["event_id"] in stale:
                continue
            result.append(event)
        return result

    # ---- receipts ---------------------------------------------------------
    def receipt_by_id(self, receipt_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute("SELECT * FROM receipts WHERE receipt_id=?", (receipt_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["receipt"] = json.loads(result.pop("payload_json"))
        return result

    def receipts(self) -> List[Dict[str, Any]]:
        result = []
        for row in self._conn.execute("SELECT payload_json FROM receipts ORDER BY sequence"):
            result.append(json.loads(row[0]))
        return result

    def receipt_head(self) -> str:
        row = self._conn.execute("SELECT receipt_hash FROM receipts ORDER BY sequence DESC LIMIT 1").fetchone()
        return row[0] if row is not None else GENESIS_HASH

    def _quarantine(self, reason_code: str, detail: str, subject_id: Optional[str] = None) -> None:
        self._require_writer()
        quarantine_id = content_id("quarantine", {"reason_code": reason_code, "detail": detail, "subject_id": subject_id})
        with self._write_transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO quarantines(quarantine_id,reason_code,subject_id,detail,created_at) VALUES(?,?,?,?,?)",
                (quarantine_id, reason_code, subject_id, detail, self._clock()),
            )
            conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('quarantined','1')")

    def ingest_receipt(self, receipt: Mapping[str, Any], public_key: Any) -> Dict[str, Any]:
        self._require_writer()
        candidate = dict(receipt)
        try:
            complete_hash = verify_receipt(candidate, public_key)
            supplied_key_id = key_id_for_public_key(public_key)
        except Exception as exc:
            self._quarantine("INVALID_RECEIPT", str(exc), candidate.get("receipt_id"))
            raise
        try:
            with self._write_transaction() as conn:
                existing = conn.execute(
                    "SELECT * FROM receipts WHERE receipt_id=? OR idempotency_key=?",
                    (candidate["receipt_id"], candidate["idempotency_key"]),
                ).fetchone()
                if existing is not None:
                    existing_payload = json.loads(existing["payload_json"])
                    if canonical_json(existing_payload) != canonical_json(candidate):
                        raise ReceiptConflictError("receipt ID or idempotency key conflicts with stored content")
                    return self._row_to_event(conn.execute("SELECT * FROM events WHERE event_id=?", (existing["event_id"],)).fetchone())
                pinned_row = conn.execute("SELECT value FROM meta WHERE key='evaluator_key_id'").fetchone()
                pinned_key_id = pinned_row[0] if pinned_row is not None else None
                if pinned_key_id is not None and pinned_key_id != supplied_key_id:
                    raise ReceiptVerificationError("ledger is pinned to a different evaluator signing key")
                last = conn.execute("SELECT sequence,receipt_hash FROM receipts ORDER BY sequence DESC LIMIT 1").fetchone()
                expected_sequence = int(last[0]) + 1 if last is not None else 1
                expected_previous = last[1] if last is not None else GENESIS_HASH
                verify_receipt(
                    candidate,
                    public_key,
                    expected_key_id=supplied_key_id,
                    expected_sequence=expected_sequence,
                    expected_previous_hash=expected_previous,
                )
                if pinned_key_id is None:
                    conn.execute(
                        "INSERT OR IGNORE INTO meta(key,value) VALUES('evaluator_key_id',?)",
                        (supplied_key_id,),
                    )
                payload = {"receipt": candidate, "receipt_hash": complete_hash}
                event = self._insert_event_tx(
                    conn,
                    event_type="RECEIPT",
                    payload=payload,
                    campaign_id=candidate["campaign_id"],
                    run_id=candidate["run_id"],
                    task_id=candidate["task_id"],
                    subject_id=candidate["candidate_id"],
                    source_class="FROZEN_EVALUATOR",
                    disposition="VERIFIED",
                    idempotency_key=f"receipt:{candidate['idempotency_key']}",
                )
                conn.execute(
                    """INSERT INTO receipts(receipt_id,sequence,receipt_hash,receipt_type,campaign_id,run_id,task_id,
                       candidate_id,idempotency_key,previous_receipt_hash,payload_json,event_id)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        candidate["receipt_id"],
                        candidate["sequence"],
                        complete_hash,
                        candidate["receipt_type"],
                        candidate["campaign_id"],
                        candidate["run_id"],
                        candidate["task_id"],
                        candidate["candidate_id"],
                        candidate["idempotency_key"],
                        candidate["previous_receipt_hash"],
                        canonical_json(candidate),
                        event["event_id"],
                    ),
                )
                return event
        except (ReceiptConflictError, ReceiptVerificationError, IntegrityError) as exc:
            self._quarantine("RECEIPT_CONFLICT_OR_CHAIN", str(exc), candidate.get("receipt_id"))
            raise

    def ingest_receipts(self, receipts: Iterable[Mapping[str, Any]], public_key: Any) -> int:
        count = 0
        for receipt in receipts:
            before = self.receipt_by_id(receipt.get("receipt_id")) if receipt.get("receipt_id") else None
            self.ingest_receipt(receipt, public_key)
            if before is None:
                count += 1
        return count

    def verify_receipt_chain(self, public_key: Any) -> Dict[str, Any]:
        """Verify every ingested receipt against its Ed25519 key and chain."""

        supplied_key_id = key_id_for_public_key(public_key)
        pinned_row = self._conn.execute("SELECT value FROM meta WHERE key='evaluator_key_id'").fetchone()
        pinned_key_id = pinned_row[0] if pinned_row is not None else supplied_key_id
        if pinned_key_id != supplied_key_id:
            raise ReceiptVerificationError("ledger is pinned to a different evaluator signing key")
        previous = GENESIS_HASH
        count = 0
        for row in self._conn.execute("SELECT * FROM receipts ORDER BY sequence"):
            receipt = json.loads(row["payload_json"])
            complete_hash = verify_receipt(
                receipt,
                public_key,
                expected_key_id=pinned_key_id,
                expected_sequence=row["sequence"],
                expected_previous_hash=previous,
            )
            if complete_hash != row["receipt_hash"]:
                raise IntegrityError(f"receipt {row['receipt_id']} hash mismatch")
            previous = complete_hash
            count += 1
        return {"receipt_count": count, "receipt_head_hash": previous}

    # ---- checkpoint, projection queue, integrity -------------------------
    def add_checkpoint(
        self,
        checkpoint_id: str,
        *,
        campaign_id: str,
        last_completed_phase: str,
        projection_generation: str,
        artifact_manifest_hash: str,
        created_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._conn.execute("SELECT 1 FROM campaigns WHERE campaign_id=?", (campaign_id,)).fetchone() is None:
            raise UnknownReferenceError(f"unknown campaign: {campaign_id}")
        created = created_at or self._clock()
        with self._write_transaction() as conn:
            existing = conn.execute("SELECT * FROM checkpoints WHERE checkpoint_id=?", (checkpoint_id,)).fetchone()
            head = self.ledger_head_hash()
            event_id = self.ledger_head_event_id() or "GENESIS"
            requested = {
                "checkpoint_id": checkpoint_id,
                "campaign_id": campaign_id,
                "last_completed_phase": last_completed_phase,
                "last_durable_event_id": event_id,
                "ledger_hash": head,
                "projection_generation": projection_generation,
                "artifact_manifest_hash": artifact_manifest_hash,
                "created_at": created,
            }
            if existing is not None:
                existing_dict = dict(existing)
                compare_keys = set(requested) if created_at is not None else set(requested) - {"created_at"}
                if canonical_json({key: existing_dict[key] for key in compare_keys}) != canonical_json({key: requested[key] for key in compare_keys}):
                    raise IdempotencyConflictError(f"checkpoint {checkpoint_id} was delivered with conflicting content")
                return dict(existing)
            conn.execute(
                """INSERT INTO checkpoints(checkpoint_id,campaign_id,last_completed_phase,last_durable_event_id,
                   ledger_hash,projection_generation,artifact_manifest_hash,created_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (checkpoint_id, campaign_id, last_completed_phase, event_id, head, projection_generation, artifact_manifest_hash, created),
            )
            return dict(conn.execute("SELECT * FROM checkpoints WHERE checkpoint_id=?", (checkpoint_id,)).fetchone())

    def enqueue_projection_rebuild(self, *, projection_generation: Optional[str], reason_code: str) -> Dict[str, Any]:
        with self._write_transaction() as conn:
            head = self.ledger_head_hash()
            queue_id = content_id("projection", {"ledger_head_hash": head, "projection_generation": projection_generation, "reason_code": reason_code})
            conn.execute(
                "INSERT OR IGNORE INTO projection_queue(queue_id,ledger_head_hash,projection_generation,reason_code,created_at) VALUES(?,?,?,?,?)",
                (queue_id, head, projection_generation, reason_code, self._clock()),
            )
            return dict(conn.execute("SELECT * FROM projection_queue WHERE queue_id=?", (queue_id,)).fetchone())

    def projection_rebuild_queue(self) -> List[Dict[str, Any]]:
        return [dict(row) for row in self._conn.execute("SELECT * FROM projection_queue ORDER BY rowid")]

    def blob_metadata(self, digest: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute("SELECT * FROM blobs WHERE digest=?", (digest,)).fetchone()
        return dict(row) if row is not None else None

    def export_public_blobs(self, directory: Union[str, Path]) -> List[Dict[str, Any]]:
        """Materialize only the closed public-eligible blob allowlist."""

        if self.blob_store is None:
            raise LedgerError("public blob export requires the private blob store")
        root = Path(directory)
        manifest: List[Dict[str, Any]] = []
        for row in self._conn.execute("SELECT * FROM blobs WHERE visibility='public-eligible' ORDER BY digest"):
            if row["logical_role"] not in PUBLIC_BLOB_ROLES:
                raise LedgerError(f"public blob role is not allowlisted: {row['logical_role']}")
            data = self.blob_store.get(row["digest"])
            if row["byte_size"] != len(data):
                raise IntegrityError(f"blob metadata size mismatch for {row['digest']}")
            target = root / "blobs" / "sha256" / row["digest"][:2] / row["digest"][2:4] / row["digest"]
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.public-tmp-{os.getpid()}-{threading.get_ident()}")
            with temporary.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            target.chmod(0o444)
            manifest.append(
                {
                    "digest": row["digest"],
                    "byte_size": len(data),
                    "media_type": row["media_type"],
                    "logical_role": row["logical_role"],
                }
            )
        return manifest

    def quarantines(self) -> List[Dict[str, Any]]:
        return [dict(row) for row in self._conn.execute("SELECT * FROM quarantines ORDER BY rowid")]

    def ledger_head_hash(self) -> str:
        row = self._conn.execute("SELECT value FROM meta WHERE key='ledger_head_hash'").fetchone()
        if row is not None:
            return row[0]
        row = self._conn.execute("SELECT event_hash FROM events ORDER BY sequence DESC LIMIT 1").fetchone()
        return row[0] if row is not None else GENESIS_HASH

    def ledger_head_event_id(self) -> Optional[str]:
        row = self._conn.execute("SELECT event_id FROM events ORDER BY sequence DESC LIMIT 1").fetchone()
        return row[0] if row is not None else None

    def events(self) -> List[Dict[str, Any]]:
        return [self._row_to_event(row) for row in self._conn.execute("SELECT * FROM events ORDER BY sequence")]

    def _verify_blob_row(self, row: sqlite3.Row) -> None:
        digest = row["blob_digest"]
        if digest is None:
            return
        if self.blob_store is None:
            raise IntegrityError(f"event {row['event_id']} references a blob without a configured blob store")
        metadata = self.blob_metadata(digest)
        if metadata is None:
            raise IntegrityError(f"event {row['event_id']} references an unregistered blob {digest}")
        data = self.blob_store.get(digest)
        if metadata["byte_size"] != len(data):
            raise IntegrityError(f"blob metadata size mismatch for {digest}")
        if digest_for(data) != row["payload_hash"]:
            raise IntegrityError(f"event {row['event_id']} blob bytes do not match payload hash")

    def verify_integrity(self) -> Dict[str, Any]:
        previous = GENESIS_HASH
        events = []
        for raw_row in self._conn.execute("SELECT * FROM events ORDER BY sequence"):
            self._verify_blob_row(raw_row)
            events.append(self._row_to_event(raw_row))
        for expected_sequence, event in enumerate(events, start=1):
            if event["sequence"] != expected_sequence:
                raise IntegrityError("event sequence is not contiguous")
            if event["previous_hash"] != previous:
                raise IntegrityError(f"event {event['event_id']} has an invalid previous hash")
            payload = event.get("payload")
            if payload is not None and digest_for(canonical_bytes(payload)) != event["payload_hash"]:
                raise IntegrityError(f"event {event['event_id']} payload hash mismatch")
            expected_hash = chain_digest(previous, self._event_record_for_hash(event))
            if event["event_hash"] != expected_hash:
                raise IntegrityError(f"event {event['event_id']} hash mismatch")
            previous = event["event_hash"]
        if self.ledger_head_hash() != previous:
            raise IntegrityError("stored ledger head hash does not match event chain")
        receipt_key_ids = {
            json.loads(row["payload_json"]).get("signing_key_id")
            for row in self._conn.execute("SELECT payload_json FROM receipts ORDER BY sequence")
        }
        receipt_key_ids.discard(None)
        if len(receipt_key_ids) > 1:
            raise IntegrityError("receipt table contains mixed evaluator signing keys")
        pinned_key_row = self._conn.execute("SELECT value FROM meta WHERE key='evaluator_key_id'").fetchone()
        if pinned_key_row is not None and receipt_key_ids and receipt_key_ids != {pinned_key_row[0]}:
            raise IntegrityError("receipt table does not match the pinned evaluator signing key")
        # A topological check is part of integrity, not just insertion hygiene.
        edges = [(row[0], row[1]) for row in self._conn.execute("SELECT parent_id,child_id FROM dependencies")]
        for parent, child in edges:
            if self._would_cycle([edge for edge in edges if edge != (parent, child)], parent, child):
                raise IntegrityError("dependency table contains a cycle")
        for checkpoint in self._conn.execute("SELECT * FROM checkpoints ORDER BY checkpoint_id"):
            if self._conn.execute("SELECT 1 FROM campaigns WHERE campaign_id=?", (checkpoint["campaign_id"],)).fetchone() is None:
                raise IntegrityError(f"checkpoint {checkpoint['checkpoint_id']} references an unknown campaign")
            if checkpoint["last_durable_event_id"] == "GENESIS":
                expected_checkpoint_hash = GENESIS_HASH
            else:
                durable_event = self._conn.execute(
                    "SELECT event_hash FROM events WHERE event_id=?", (checkpoint["last_durable_event_id"],)
                ).fetchone()
                if durable_event is None:
                    raise IntegrityError(f"checkpoint {checkpoint['checkpoint_id']} references an unknown durable event")
                expected_checkpoint_hash = durable_event[0]
            if checkpoint["ledger_hash"] != expected_checkpoint_hash:
                raise IntegrityError(f"checkpoint {checkpoint['checkpoint_id']} ledger hash mismatch")
        return {
            "event_count": len(events),
            "ledger_head_hash": previous,
            "receipt_count": self._conn.execute("SELECT COUNT(*) FROM receipts").fetchone()[0],
            "chain_valid": True,
        }

    # ---- deterministic export/replay -------------------------------------
    def export_jsonl(self, path: Optional[Union[str, Path]] = None) -> str:
        self.verify_integrity()
        records = []
        for event in self.events():
            record = {"record_type": "EVENT", "ledger_schema_version": LEDGER_SCHEMA_VERSION}
            record.update({key: event[key] for key in _EVENT_COLUMNS})
            if "payload" in event:
                record["payload"] = event["payload"]
            if "blob_media_type" in event:
                record["blob_media_type"] = event["blob_media_type"]
            records.append(record)
        for checkpoint in self._conn.execute("SELECT * FROM checkpoints ORDER BY checkpoint_id"):
            records.append(
                {
                    "record_type": "CHECKPOINT",
                    "ledger_schema_version": LEDGER_SCHEMA_VERSION,
                    "checkpoint": dict(checkpoint),
                }
            )
        text = "".join(canonical_json(record) + "\n" for record in records)
        if path is not None:
            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
        return text

    export_events_jsonl = export_jsonl
    export_deterministic_jsonl = export_jsonl

    @classmethod
    def replay_jsonl(
        cls,
        export: Union[str, Path],
        target_path: Union[str, Path],
        *,
        blob_root: Optional[Union[str, Path]] = None,
    ) -> "EvidenceLedger":
        if isinstance(export, Path):
            text = export.read_text(encoding="utf-8")
        else:
            export_text = str(export)
            looks_like_inline_jsonl = "\n" in export_text or export_text.lstrip().startswith("{")
            if not looks_like_inline_jsonl and len(export_text) < 4096 and Path(export_text).exists():
                text = Path(export_text).read_text(encoding="utf-8")
            else:
                text = export_text
        records = parse_canonical_jsonl(text)
        target = Path(target_path)
        if target.exists() and target.stat().st_size:
            raise LedgerError(f"replay target is not empty: {target}")
        effective_blob_root = blob_root
        if effective_blob_root is None and any(record.get("blob_digest") for record in records):
            # A self-contained export carries the canonical payload alongside
            # its blob address. Materialize replay blobs beside the target
            # unless the caller supplied an explicit private root.
            effective_blob_root = target.parent / f"{target.name}.blobs"
        ledger = cls(target, blob_root=effective_blob_root)
        try:
            with ledger._write_transaction() as conn:
                for record in records:
                    if record.get("ledger_schema_version") != LEDGER_SCHEMA_VERSION:
                        raise IntegrityError("unsupported ledger export record")
                    if record.get("record_type") == "CHECKPOINT":
                        ledger._materialize_replayed_checkpoint_tx(conn, record.get("checkpoint"))
                        continue
                    if record.get("record_type") != "EVENT":
                        raise IntegrityError("unsupported ledger export record")
                    payload = record.get("payload")
                    if payload is None:
                        payload = {}
                        if record.get("blob_digest") is None:
                            raise IntegrityError("event export omitted both payload and blob digest")
                    event = ledger._insert_event_tx(
                        conn,
                        event_type=record["event_type"],
                        payload=payload,
                        campaign_id=record.get("campaign_id"),
                        run_id=record.get("run_id"),
                        task_id=record.get("task_id"),
                        valid_time=record.get("valid_time"),
                        subject_id=record.get("subject_id"),
                        source_class=record.get("source_class"),
                        disposition=record.get("disposition"),
                        evaluator_identity=record.get("evaluator_identity"),
                        event_id=record["event_id"],
                        idempotency_key=record.get("idempotency_key"),
                        transaction_time=record["transaction_time"],
                        payload_hash=record["payload_hash"],
                        payload_json=record.get("payload_json"),
                        blob_digest=record.get("blob_digest"),
                        blob_media_type=record.get("blob_media_type", "application/json"),
                        sequence=record["sequence"],
                        previous_hash=record["previous_hash"],
                        event_hash=record["event_hash"],
                    )
                    ledger._materialize_replayed_event_tx(conn, event)
            ledger.verify_integrity()
        except Exception:
            ledger.close()
            raise
        return ledger

    def _materialize_replayed_checkpoint_tx(
        self,
        conn: sqlite3.Connection,
        checkpoint: Any,
    ) -> None:
        if not isinstance(checkpoint, Mapping):
            raise IntegrityError("checkpoint export record must contain an object")
        required = {
            "checkpoint_id",
            "campaign_id",
            "last_completed_phase",
            "last_durable_event_id",
            "ledger_hash",
            "projection_generation",
            "artifact_manifest_hash",
            "created_at",
        }
        if set(checkpoint) != required:
            raise IntegrityError("checkpoint export record has an unexpected schema")
        if conn.execute("SELECT 1 FROM campaigns WHERE campaign_id=?", (checkpoint["campaign_id"],)).fetchone() is None:
            raise IntegrityError(f"checkpoint {checkpoint['checkpoint_id']} references an unknown campaign")
        if checkpoint["last_durable_event_id"] == "GENESIS":
            expected_hash = GENESIS_HASH
        else:
            event = conn.execute(
                "SELECT event_hash FROM events WHERE event_id=?", (checkpoint["last_durable_event_id"],)
            ).fetchone()
            if event is None:
                raise IntegrityError(f"checkpoint {checkpoint['checkpoint_id']} references an unknown durable event")
            expected_hash = event[0]
        if checkpoint["ledger_hash"] != expected_hash:
            raise IntegrityError(f"checkpoint {checkpoint['checkpoint_id']} ledger hash mismatch")
        existing = conn.execute(
            "SELECT * FROM checkpoints WHERE checkpoint_id=?", (checkpoint["checkpoint_id"],)
        ).fetchone()
        if existing is not None:
            if canonical_json(dict(existing)) != canonical_json(dict(checkpoint)):
                raise IdempotencyConflictError(f"checkpoint {checkpoint['checkpoint_id']} was delivered with conflicting content")
            return
        conn.execute(
            """INSERT INTO checkpoints(checkpoint_id,campaign_id,last_completed_phase,last_durable_event_id,
               ledger_hash,projection_generation,artifact_manifest_hash,created_at)
               VALUES(?,?,?,?,?,?,?,?)""",
            (
                checkpoint["checkpoint_id"],
                checkpoint["campaign_id"],
                checkpoint["last_completed_phase"],
                checkpoint["last_durable_event_id"],
                checkpoint["ledger_hash"],
                checkpoint["projection_generation"],
                checkpoint["artifact_manifest_hash"],
                checkpoint["created_at"],
            ),
        )

    def _materialize_replayed_event_tx(self, conn: sqlite3.Connection, event: Mapping[str, Any]) -> None:
        payload = event.get("payload") or {}
        event_type = event["event_type"]
        if event_type == "CAMPAIGN":
            conn.execute(
                """INSERT OR IGNORE INTO campaigns(campaign_id,event_id,protocol_hash,source_commit,model_revision,
                   data_manifest_hash,evaluator_hash,policy_hash,seed_set_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (payload["campaign_id"], event["event_id"], payload["protocol_hash"], payload["source_commit"], payload["model_revision"], payload["data_manifest_hash"], payload["evaluator_hash"], payload["policy_hash"], canonical_json(payload["seed_set"]), payload["created_at"]),
            )
        elif event_type == "RUN":
            conn.execute(
                """INSERT OR IGNORE INTO runs(run_id,event_id,campaign_id,arm,task_id,seed,parent_checkpoint,
                   start_state,end_state,host_role,software_manifest_hash,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (payload["run_id"], event["event_id"], payload["campaign_id"], payload["arm"], payload["task_id"], payload["seed"], payload.get("parent_checkpoint"), payload["start_state"], payload.get("end_state"), payload["host_role"], payload["software_manifest_hash"], payload["created_at"]),
            )
        elif event_type == "CANDIDATE":
            conn.execute(
                """INSERT OR IGNORE INTO candidates(candidate_id,event_id,campaign_id,run_id,task_id,parent_candidate_id,
                   mutation_family,patch_hash,requested_authority,prompt_hash,model_hash,adapter_hash,candidate_json)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (payload["candidate_id"], event["event_id"], payload["campaign_id"], payload["run_id"], payload["task_id"], payload.get("parent_candidate_id"), payload["mutation_family"], payload["patch_hash"], payload["requested_authority"], payload["prompt_hash"], payload["model_hash"], payload.get("adapter_hash"), canonical_json(payload)),
            )
        elif event_type == "DEPENDENCY":
            dependency_id = content_id("dep", {"parent_id": payload["parent_id"], "child_id": payload["child_id"], "edge_type": payload["edge_type"]})
            conn.execute("INSERT OR IGNORE INTO dependencies(dependency_id,parent_id,child_id,edge_type,insertion_event_id) VALUES(?,?,?,?,?)", (dependency_id, payload["parent_id"], payload["child_id"], payload["edge_type"], event["event_id"]))
        elif event_type == "CORRECTION":
            correction_id = content_id("correction", payload)
            conn.execute("INSERT OR IGNORE INTO corrections(correction_id,superseded_id,replacement_id,reason_code,correction_source,event_id) VALUES(?,?,?,?,?,?)", (correction_id, payload["superseded_event_id"], payload["replacement_event_id"], payload["reason_code"], payload["correction_source"], event["event_id"]))
        elif event_type == "RETRACTION":
            retraction_id = content_id("retraction", payload)
            conn.execute("INSERT OR IGNORE INTO retractions(retraction_id,subject_id,reason_code,retraction_source,event_id) VALUES(?,?,?,?,?)", (retraction_id, payload["subject_id"], payload["reason_code"], payload["retraction_source"], event["event_id"]))
        elif event_type == "RECEIPT":
            receipt = payload["receipt"]
            pinned_key_row = conn.execute("SELECT value FROM meta WHERE key='evaluator_key_id'").fetchone()
            receipt_key_id = receipt.get("signing_key_id")
            if pinned_key_row is not None and pinned_key_row[0] != receipt_key_id:
                raise IntegrityError("replay contains mixed evaluator signing keys")
            if pinned_key_row is None and receipt_key_id:
                conn.execute(
                    "INSERT OR IGNORE INTO meta(key,value) VALUES('evaluator_key_id',?)",
                    (receipt_key_id,),
                )
            conn.execute("INSERT OR IGNORE INTO receipts(receipt_id,sequence,receipt_hash,receipt_type,campaign_id,run_id,task_id,candidate_id,idempotency_key,previous_receipt_hash,payload_json,event_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (receipt["receipt_id"], receipt["sequence"], payload["receipt_hash"], receipt["receipt_type"], receipt["campaign_id"], receipt["run_id"], receipt["task_id"], receipt["candidate_id"], receipt["idempotency_key"], receipt["previous_receipt_hash"], canonical_json(receipt), event["event_id"]))
        elif event_type == "VERDICT":
            conn.execute(
                """INSERT OR IGNORE INTO verdicts(verdict_id,event_id,candidate_id,receipt_id,correctness,
                   performance_json,hidden_test_set_hash,evaluator_revision,signed_receipt_hash)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (payload["verdict_id"], event["event_id"], payload["candidate_id"], payload.get("receipt_id"), None if payload.get("correctness") is None else int(bool(payload["correctness"])), canonical_json(payload.get("performance") or {}), payload["hidden_test_set_hash"], payload["evaluator_revision"], payload.get("signed_receipt_hash")),
            )
        elif event_type == "EFFECT_RECEIPT":
            conn.execute(
                """INSERT OR IGNORE INTO effect_receipts(request_id,event_id,candidate_id,identity,normalized_action_hash,
                   decision,policy_hash,sandbox_id,started_at,finished_at,exit_status_class,output_hash,
                   environment_diff_hash,signature,receipt_id)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (payload["request_id"], event["event_id"], payload["candidate_id"], payload["identity"], payload["normalized_action_hash"], payload["decision"], payload["policy_hash"], payload["sandbox_id"], payload["started_at"], payload["finished_at"], payload["exit_status_class"], payload.get("output_hash"), payload.get("environment_diff_hash"), payload["signature"], payload.get("receipt_id")),
            )

    def status(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "mode": self.mode,
            "ledger_head_hash": self.ledger_head_hash(),
            "ledger_head_event_id": self.ledger_head_event_id(),
            "event_count": self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
            "candidate_count": self._conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0],
            "receipt_count": self._conn.execute("SELECT COUNT(*) FROM receipts").fetchone()[0],
            "projection_rebuild_queue_count": self._conn.execute("SELECT COUNT(*) FROM projection_queue").fetchone()[0],
            "quarantine_count": self._conn.execute("SELECT COUNT(*) FROM quarantines").fetchone()[0],
            "quarantined": self._conn.execute("SELECT value FROM meta WHERE key='quarantined'").fetchone()[0] == "1" if self._conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'").fetchone() else False,
        }


SingleWriterLedger = EvidenceLedger


__all__ = [
    "ABSTAINED",
    "BlobStore",
    "EvidenceLedger",
    "INLINE_PAYLOAD_LIMIT",
    "LEDGER_SCHEMA_VERSION",
    "PUBLIC_BLOB_ROLES",
    "PROMOTED",
    "REJECTED",
    "RETRACTED",
    "STALE_DEPENDENT",
    "SingleWriterLedger",
]
