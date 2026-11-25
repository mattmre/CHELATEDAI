"""Independent conventional bitemporal + authority SQL challenger ledger.

Wave 16 / bundle #47 spine component (master plan section 2.2).  This module
is the *challenger* against which the EK5 v3 stratified evidence kernel is
compared (6C head-to-head, 2D low-memory SQLite, 4, 5.2).  The comparison is
only meaningful if the challenger shares **zero decision code** with EK5, so:

* this module imports nothing from ``ek5_v3_stratified`` or any ``ek5*``
  module (enforced by an AST assertion in the unit test), and
* every decision (visibility, authority precedence, retraction propagation,
  conflict escalation) is expressed in plain SQL over conventional tables,
  with only a tiny deterministic Python reduction step on the returned rows.

Semantics (identical across the SQLite and PostgreSQL 17 engines and across
the ``recursive_cte`` and ``closure_table`` evaluation modes):

``tenant isolation``
    Every statement is scoped by ``tenant_id``.  An edge never resolves a
    record id in another tenant (a cross-tenant parent is a *missing* parent).
``bitemporal visibility``
    An operation is visible iff ``tx_time <= query tx_time`` (and, when a
    ``policy_epoch`` is supplied, ``policy_epoch <= that epoch``).  A record's
    attributes come from its latest visible ASSERT/CONFLICT revision; it is a
    candidate for ``valid_time`` iff ``valid_from <= valid_time < valid_to``
    (``valid_to`` NULL = open ended).
``retraction / reinstatement``
    The last *effective* WITHDRAW/REINSTATE op wins.  A status op is
    effective only if its ``authority_rank`` is >= the target record's
    authority, so a low-authority actor cannot retract a high-authority claim.
``derivation propagation``
    ``DERIVES(src, dst)`` means *src is derived from dst*.  A candidate is
    dead if it, or any transitive ancestor, is withdrawn or not (yet) asserted
    (a *missing parent*).  Cycles are tolerated: they only ever terminate the
    walk, they do not by themselves kill anything.
``authority precedence``
    Among live candidates the highest ``authority_rank`` wins.  If the winning
    rank holds more than one distinct claim value, or any winner is a
    CONFLICT-flagged revision, the entity is ``DISPUTED`` (no winner is
    invented).  Retracting one side of an equal-authority conflict resolves
    it; a higher-authority assertion overrides it.

Two evaluators exist for derivation propagation and must agree exactly:

``recursive_cte``
    Query-time ``WITH RECURSIVE`` ancestor walk with a ``visited_path``
    string guard (SQLite) / ``CYCLE record_id SET is_cycle USING path``
    (PostgreSQL 17).  Cost is proportional to the number of *simple* ancestor
    paths of the candidate records.
``closure_table``
    An incrementally maintained ``edge_closure`` table holding, per
    ``(tenant, src, dst)``, the bottleneck ``min_tx`` (the earliest tx time at
    which some fully-visible derivation path exists).  Maintained on every
    edge insert in Python (SQLite) or by an ``AFTER INSERT`` trigger
    (PostgreSQL 17).

Limits that are deliberate and documented rather than hidden:

* ``policy_epoch`` gates record/status operations only; edges are structural
  and are not epoch-filtered (a precomputed closure cannot be epoch-sliced).
* Edges are immutable and only tx-time filtered (no valid-time on edges).
* Times are non-negative integer ticks.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import re
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "AsOfQueryResult",
    "ChallengerBitemporalLedger",
    "LedgerBackend",
    "LedgerEdgeEnvelope",
    "LedgerError",
    "LedgerRecordEnvelope",
    "OpKind",
    "Postgres17ChallengerBackend",
    "Postgres17SQLDialectBuilder",
    "QUERY_MODES",
    "SQLiteChallengerBackend",
    "resolve_candidates",
]

QUERY_MODES = ("recursive_cte", "closure_table")

STATUS_RESOLVED = "RESOLVED"
STATUS_DISPUTED = "DISPUTED"
STATUS_RETRACTED = "RETRACTED"
STATUS_ABSENT = "ABSENT"

_ID_FORBIDDEN = re.compile(r"[,\x00]")
_SQLITE_MAX_CACHE_MIB = 960


class LedgerError(ValueError):
    """Raised for invalid envelopes, bad query arguments and integrity faults."""


class OpKind(str, Enum):
    """Operation kinds carried by ledger envelopes."""

    ASSERT = "ASSERT"
    WITHDRAW = "WITHDRAW"
    REINSTATE = "REINSTATE"
    CONFLICT = "CONFLICT"
    DERIVES = "DERIVES"


_RECORD_OPS = frozenset(
    {OpKind.ASSERT, OpKind.WITHDRAW, OpKind.REINSTATE, OpKind.CONFLICT}
)
_ASSERTING_OPS = frozenset({OpKind.ASSERT, OpKind.CONFLICT})


def _check_id(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise LedgerError(f"{name} must be a non-empty string")
    if _ID_FORBIDDEN.search(value):
        raise LedgerError(f"{name} must not contain ',' or NUL: {value!r}")
    return value


def _check_tick(name: str, value: Any, *, allow_none: bool = False) -> Optional[int]:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise LedgerError(f"{name} must be an int tick, got {value!r}")
    if value < 0:
        raise LedgerError(f"{name} must be >= 0, got {value}")
    return value


@dataclass(frozen=True)
class LedgerRecordEnvelope:
    """One record-level operation (ASSERT / WITHDRAW / REINSTATE / CONFLICT).

    ``record_id`` names the *target* record: WITHDRAW/REINSTATE rows refer to
    an earlier-asserted record and a repeated ASSERT/CONFLICT for the same
    ``record_id`` is a bitemporal revision (latest ``tx_time``/insert order
    wins).  ``authority_rank`` on a WITHDRAW/REINSTATE is the authority of the
    actor performing it.
    """

    tenant_id: str
    record_id: str
    entity_key: str
    authority_rank: int
    valid_from: int
    valid_to: Optional[int]
    tx_time: int
    op_kind: OpKind
    policy_epoch: int = 0
    claim_value: str = ""

    def __post_init__(self) -> None:
        _check_id("tenant_id", self.tenant_id)
        _check_id("record_id", self.record_id)
        _check_id("entity_key", self.entity_key)
        try:
            kind = OpKind(self.op_kind)
        except ValueError as exc:
            raise LedgerError(f"unknown op_kind {self.op_kind!r}") from exc
        object.__setattr__(self, "op_kind", kind)
        if kind not in _RECORD_OPS:
            raise LedgerError(f"{kind.value} is not a record operation")
        _check_tick("authority_rank", self.authority_rank)
        _check_tick("valid_from", self.valid_from)
        _check_tick("valid_to", self.valid_to, allow_none=True)
        _check_tick("tx_time", self.tx_time)
        _check_tick("policy_epoch", self.policy_epoch)
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise LedgerError("valid_to must be > valid_from (half-open [from, to))")
        if not isinstance(self.claim_value, str):
            raise LedgerError("claim_value must be a string")
        if kind in _ASSERTING_OPS and not self.claim_value:
            raise LedgerError(f"{kind.value} requires a non-empty claim_value")


@dataclass(frozen=True)
class LedgerEdgeEnvelope:
    """``DERIVES(src_id, dst_id)``: *src_id is derived from dst_id*."""

    tenant_id: str
    edge_id: str
    src_id: str
    dst_id: str
    tx_time: int
    policy_epoch: int = 0
    op_kind: OpKind = OpKind.DERIVES

    def __post_init__(self) -> None:
        _check_id("tenant_id", self.tenant_id)
        _check_id("edge_id", self.edge_id)
        _check_id("src_id", self.src_id)
        _check_id("dst_id", self.dst_id)
        if OpKind(self.op_kind) is not OpKind.DERIVES:
            raise LedgerError("edge envelopes must use op_kind DERIVES")
        object.__setattr__(self, "op_kind", OpKind.DERIVES)
        _check_tick("tx_time", self.tx_time)
        _check_tick("policy_epoch", self.policy_epoch)


@dataclass(frozen=True)
class AsOfQueryResult:
    """Deterministic answer to one bitemporal as-of query."""

    tenant_id: str
    entity_key: str
    valid_time: int
    tx_time: int
    policy_epoch: Optional[int]
    status: str
    winner_record_id: Optional[str]
    winner_claim_value: Optional[str]
    winner_authority_rank: Optional[int]
    contending_record_ids: Tuple[str, ...]
    live_record_ids: Tuple[str, ...]
    retracted_record_ids: Tuple[str, ...]
    mode: str

    def semantic_key(self) -> Tuple[Any, ...]:
        """Everything that must be identical across evaluation modes/engines."""
        return (
            self.tenant_id,
            self.entity_key,
            self.valid_time,
            self.tx_time,
            self.policy_epoch,
            self.status,
            self.winner_record_id,
            self.winner_claim_value,
            self.winner_authority_rank,
            self.contending_record_ids,
            self.live_record_ids,
            self.retracted_record_ids,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "entity_key": self.entity_key,
            "valid_time": self.valid_time,
            "tx_time": self.tx_time,
            "policy_epoch": self.policy_epoch,
            "status": self.status,
            "winner_record_id": self.winner_record_id,
            "winner_claim_value": self.winner_claim_value,
            "winner_authority_rank": self.winner_authority_rank,
            "contending_record_ids": list(self.contending_record_ids),
            "live_record_ids": list(self.live_record_ids),
            "retracted_record_ids": list(self.retracted_record_ids),
            "mode": self.mode,
        }

    def digest(self) -> str:
        """SHA-256 over the mode-independent semantic content."""
        payload = json.dumps(
            list(self.semantic_key()), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Shared SQL composition.  Pure text; one place defines the decision logic so
# the SQLite and PostgreSQL 17 engines cannot drift apart.  Named parameters
# are written ``:name`` and converted to ``%(name)s`` for psycopg.
# --------------------------------------------------------------------------

_EPOCH_OK = "(CAST(:epoch AS BIGINT) IS NULL OR {alias}.policy_epoch <= :epoch)"


def _epoch_ok(alias: str) -> str:
    return _EPOCH_OK.format(alias=alias)


def _dead_predicate(node_ref: str) -> str:
    """SQL boolean: the record named by ``node_ref`` is retracted or missing."""
    return f"""(
      NOT EXISTS (
        SELECT 1 FROM ledger_ops a
        WHERE a.tenant_id = :tenant AND a.record_id = {node_ref}
          AND a.op_kind IN ('ASSERT', 'CONFLICT')
          AND a.tx_time <= :tx AND {_epoch_ok('a')})
      OR COALESCE((
        SELECT s.op_kind FROM ledger_ops s
        WHERE s.tenant_id = :tenant AND s.record_id = {node_ref}
          AND s.op_kind IN ('WITHDRAW', 'REINSTATE')
          AND s.tx_time <= :tx AND {_epoch_ok('s')}
          AND s.authority_rank >= COALESCE((
            SELECT a2.authority_rank FROM ledger_ops a2
            WHERE a2.tenant_id = :tenant AND a2.record_id = {node_ref}
              AND a2.op_kind IN ('ASSERT', 'CONFLICT')
              AND a2.tx_time <= :tx AND {_epoch_ok('a2')}
            ORDER BY a2.tx_time DESC, a2.seq DESC LIMIT 1), 0)
        ORDER BY s.tx_time DESC, s.seq DESC LIMIT 1), '') = 'WITHDRAW'
    )"""


_CANDIDATE_CTES = f"""
cand_ids AS (
  SELECT DISTINCT o.record_id
  FROM ledger_ops o
  WHERE o.tenant_id = :tenant AND o.entity_key = :entity
    AND o.op_kind IN ('ASSERT', 'CONFLICT')
    AND o.tx_time <= :tx AND {_epoch_ok('o')}
),
latest AS (
  SELECT o.record_id, o.entity_key, o.authority_rank, o.valid_from, o.valid_to,
         o.claim_value, o.op_kind, o.tx_time, o.seq
  FROM ledger_ops o
  JOIN cand_ids c ON c.record_id = o.record_id
  WHERE o.tenant_id = :tenant AND o.op_kind IN ('ASSERT', 'CONFLICT')
    AND o.tx_time <= :tx AND {_epoch_ok('o')}
    AND NOT EXISTS (
      SELECT 1 FROM ledger_ops n
      WHERE n.tenant_id = o.tenant_id AND n.record_id = o.record_id
        AND n.op_kind IN ('ASSERT', 'CONFLICT')
        AND n.tx_time <= :tx AND {_epoch_ok('n')}
        AND (n.tx_time > o.tx_time OR (n.tx_time = o.tx_time AND n.seq > o.seq)))
),
cand AS (
  SELECT * FROM latest
  WHERE entity_key = :entity AND valid_from <= :vt
    AND (valid_to IS NULL OR valid_to > :vt)
)"""

_FINAL_SELECT = """
SELECT c.record_id, c.entity_key, c.authority_rank, c.valid_from, c.valid_to,
       c.claim_value, c.op_kind, c.tx_time, c.seq,
       CASE WHEN EXISTS (SELECT 1 FROM dead d WHERE d.root_id = c.record_id)
            THEN 1 ELSE 0 END AS is_dead
FROM cand c
ORDER BY c.authority_rank DESC, c.tx_time DESC, c.seq DESC, c.record_id
"""


def _sqlite_as_of_sql(mode: str) -> str:
    if mode == "recursive_cte":
        walk = """
anc(root_id, node_id, visited_path) AS (
  SELECT record_id, record_id, ',' || record_id || ',' FROM cand
  UNION ALL
  SELECT a.root_id, e.dst_id, a.visited_path || e.dst_id || ','
  FROM anc a
  JOIN ledger_edges e
    ON e.tenant_id = :tenant AND e.src_id = a.node_id AND e.tx_time <= :tx
  WHERE instr(a.visited_path, ',' || e.dst_id || ',') = 0
)"""
    elif mode == "closure_table":
        walk = """
anc(root_id, node_id) AS (
  SELECT record_id, record_id FROM cand
  UNION
  SELECT k.src_id, k.dst_id
  FROM edge_closure k
  JOIN cand c ON k.tenant_id = :tenant AND k.src_id = c.record_id
  WHERE k.min_tx <= :tx
)"""
    else:
        raise LedgerError(f"unknown mode {mode!r}; expected one of {QUERY_MODES}")
    dead = f"""
dead AS (
  SELECT DISTINCT w.root_id FROM anc w WHERE {_dead_predicate('w.node_id')}
)"""
    return f"WITH RECURSIVE {_CANDIDATE_CTES},{walk},{dead}{_FINAL_SELECT}"


def _pg_as_of_sql(mode: str) -> str:
    if mode == "recursive_cte":
        walk = """
anc(root_id, record_id) AS (
  SELECT c.record_id, c.record_id FROM cand c
  UNION ALL
  SELECT a.root_id, e.dst_id
  FROM anc a
  JOIN ledger_edges e
    ON e.tenant_id = :tenant AND e.src_id = a.record_id AND e.tx_time <= :tx
) CYCLE record_id SET is_cycle USING path"""
        dead = f"""
dead AS (
  SELECT DISTINCT w.root_id FROM anc w
  WHERE NOT w.is_cycle AND {_dead_predicate('w.record_id')}
)"""
    elif mode == "closure_table":
        walk = """
anc(root_id, record_id) AS (
  SELECT c.record_id, c.record_id FROM cand c
  UNION
  SELECT k.src_id, k.dst_id
  FROM edge_closure k
  JOIN cand c ON k.tenant_id = :tenant AND k.src_id = c.record_id
  WHERE k.min_tx <= :tx
)"""
        dead = f"""
dead AS (
  SELECT DISTINCT w.root_id FROM anc w WHERE {_dead_predicate('w.record_id')}
)"""
    else:
        raise LedgerError(f"unknown mode {mode!r}; expected one of {QUERY_MODES}")
    return f"WITH RECURSIVE {_CANDIDATE_CTES},{walk},{dead}{_FINAL_SELECT}"


_NAMED_PARAM = re.compile(r":([a-z_]+)")
_PG_PARAM = re.compile(r"%\(([a-z_]+)\)s")


def _to_psycopg(sql: str) -> str:
    """Convert ``:name`` markers to ``%(name)s`` (no ``::`` casts are used)."""
    if "::" in sql:
        raise LedgerError("shared SQL must not contain '::' casts")
    return _NAMED_PARAM.sub(r"%(\1)s", sql)


# --------------------------------------------------------------------------
# Pure-Python reduction of candidate rows -> AsOfQueryResult
# --------------------------------------------------------------------------


def resolve_candidates(
    *,
    tenant_id: str,
    entity_key: str,
    valid_time: int,
    tx_time: int,
    policy_epoch: Optional[int],
    mode: str,
    rows: Sequence[Dict[str, Any]],
) -> AsOfQueryResult:
    """Apply authority precedence / DISPUTED escalation to SQL candidate rows."""
    live = [r for r in rows if not r["is_dead"]]
    dead = [r for r in rows if r["is_dead"]]
    retracted = tuple(sorted(r["record_id"] for r in dead))
    live_ids = tuple(sorted(r["record_id"] for r in live))
    common = dict(
        tenant_id=tenant_id,
        entity_key=entity_key,
        valid_time=valid_time,
        tx_time=tx_time,
        policy_epoch=policy_epoch,
        retracted_record_ids=retracted,
        live_record_ids=live_ids,
        mode=mode,
    )
    if not live:
        status = STATUS_RETRACTED if rows else STATUS_ABSENT
        return AsOfQueryResult(
            status=status,
            winner_record_id=None,
            winner_claim_value=None,
            winner_authority_rank=None,
            contending_record_ids=(),
            **common,
        )
    top = max(int(r["authority_rank"]) for r in live)
    winners = [r for r in live if int(r["authority_rank"]) == top]
    winners.sort(
        key=lambda r: (-int(r["tx_time"]), -int(r["seq"]), r["record_id"])
    )
    claims = {r["claim_value"] for r in winners}
    conflicted = any(OpKind(r["op_kind"]) is OpKind.CONFLICT for r in winners)
    contenders = tuple(sorted(r["record_id"] for r in winners))
    if len(claims) > 1 or conflicted:
        return AsOfQueryResult(
            status=STATUS_DISPUTED,
            winner_record_id=None,
            winner_claim_value=None,
            winner_authority_rank=top,
            contending_record_ids=contenders,
            **common,
        )
    head = winners[0]
    return AsOfQueryResult(
        status=STATUS_RESOLVED,
        winner_record_id=head["record_id"],
        winner_claim_value=head["claim_value"],
        winner_authority_rank=top,
        contending_record_ids=contenders,
        **common,
    )


# --------------------------------------------------------------------------
# Backend interface
# --------------------------------------------------------------------------


class LedgerBackend(ABC):
    """Storage/evaluation engine behind :class:`ChallengerBitemporalLedger`."""

    @abstractmethod
    def append_record(self, env: LedgerRecordEnvelope) -> None: ...

    @abstractmethod
    def append_edge(self, env: LedgerEdgeEnvelope) -> None: ...

    @abstractmethod
    def fetch_candidates(
        self,
        *,
        tenant_id: str,
        entity_key: str,
        valid_time: int,
        tx_time: int,
        policy_epoch: Optional[int],
        mode: str,
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def counts(self) -> Dict[str, int]: ...

    @abstractmethod
    def close(self) -> None: ...


class ChallengerBitemporalLedger:
    """Unified bitemporal + authority ledger over a pluggable SQL engine."""

    def __init__(self, backend: Optional[LedgerBackend] = None) -> None:
        self.backend: LedgerBackend = backend or SQLiteChallengerBackend()

    def append(self, env: LedgerRecordEnvelope) -> None:
        if not isinstance(env, LedgerRecordEnvelope):
            raise LedgerError("append() takes a LedgerRecordEnvelope")
        self.backend.append_record(env)

    def add_edge(self, env: LedgerEdgeEnvelope) -> None:
        if not isinstance(env, LedgerEdgeEnvelope):
            raise LedgerError("add_edge() takes a LedgerEdgeEnvelope")
        self.backend.append_edge(env)

    def query_as_of(
        self,
        tenant_id: str,
        entity_key: str,
        valid_time: int,
        tx_time: int,
        mode: str = "recursive_cte",
        policy_epoch: Optional[int] = None,
    ) -> AsOfQueryResult:
        _check_id("tenant_id", tenant_id)
        _check_id("entity_key", entity_key)
        _check_tick("valid_time", valid_time)
        _check_tick("tx_time", tx_time)
        _check_tick("policy_epoch", policy_epoch, allow_none=True)
        if mode not in QUERY_MODES:
            raise LedgerError(f"unknown mode {mode!r}; expected one of {QUERY_MODES}")
        rows = self.backend.fetch_candidates(
            tenant_id=tenant_id,
            entity_key=entity_key,
            valid_time=valid_time,
            tx_time=tx_time,
            policy_epoch=policy_epoch,
            mode=mode,
        )
        return resolve_candidates(
            tenant_id=tenant_id,
            entity_key=entity_key,
            valid_time=valid_time,
            tx_time=tx_time,
            policy_epoch=policy_epoch,
            mode=mode,
            rows=rows,
        )

    def counts(self) -> Dict[str, int]:
        return self.backend.counts()

    def close(self) -> None:
        self.backend.close()

    def __enter__(self) -> "ChallengerBitemporalLedger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --------------------------------------------------------------------------
# Closure maintenance helpers (engine independent, pure Python)
# --------------------------------------------------------------------------


def bottleneck_closure(
    edges: Iterable[Tuple[str, str, int]],
) -> Dict[Tuple[str, str], int]:
    """Reference closure: ``(src, dst) -> min over paths of max edge tx``.

    Bottleneck Dijkstra from every source.  ``(s, s)`` is present iff ``s``
    lies on a cycle.  Used by :meth:`SQLiteChallengerBackend.rebuild_closure`
    and by tests as the oracle for the incrementally maintained table.
    """
    out: Dict[str, List[Tuple[str, int]]] = {}
    for src, dst, tx in edges:
        out.setdefault(src, []).append((dst, tx))
    result: Dict[Tuple[str, str], int] = {}
    for source in out:
        settled: Dict[str, int] = {}
        heap: List[Tuple[int, str]] = [(tx, dst) for dst, tx in out[source]]
        heapq.heapify(heap)
        while heap:
            cost, node = heapq.heappop(heap)
            if node in settled:
                continue
            settled[node] = cost
            for nxt, tx in out.get(node, ()):
                if nxt not in settled:
                    heapq.heappush(heap, (max(cost, tx), nxt))
        for node, cost in settled.items():
            result[(source, node)] = cost
    return result


# --------------------------------------------------------------------------
# SQLite engine
# --------------------------------------------------------------------------

_SQLITE_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS ledger_ops (
      seq            INTEGER PRIMARY KEY AUTOINCREMENT,
      tenant_id      TEXT    NOT NULL,
      record_id      TEXT    NOT NULL,
      entity_key     TEXT    NOT NULL,
      authority_rank INTEGER NOT NULL CHECK (authority_rank >= 0),
      valid_from     INTEGER NOT NULL CHECK (valid_from >= 0),
      valid_to       INTEGER CHECK (valid_to IS NULL OR valid_to > valid_from),
      tx_time        INTEGER NOT NULL CHECK (tx_time >= 0),
      op_kind        TEXT    NOT NULL
                     CHECK (op_kind IN ('ASSERT','WITHDRAW','REINSTATE','CONFLICT')),
      policy_epoch   INTEGER NOT NULL DEFAULT 0 CHECK (policy_epoch >= 0),
      claim_value    TEXT    NOT NULL DEFAULT ''
    )""",
    "CREATE INDEX IF NOT EXISTS idx_ops_entity_bitemporal "
    "ON ledger_ops (tenant_id, entity_key, tx_time, valid_from)",
    "CREATE INDEX IF NOT EXISTS idx_ops_record_tx "
    "ON ledger_ops (tenant_id, record_id, tx_time, seq)",
    """
    CREATE TABLE IF NOT EXISTS ledger_edges (
      tenant_id    TEXT    NOT NULL,
      edge_id      TEXT    NOT NULL,
      src_id       TEXT    NOT NULL,
      dst_id       TEXT    NOT NULL,
      tx_time      INTEGER NOT NULL CHECK (tx_time >= 0),
      policy_epoch INTEGER NOT NULL DEFAULT 0 CHECK (policy_epoch >= 0),
      PRIMARY KEY (tenant_id, edge_id)
    ) WITHOUT ROWID""",
    "CREATE INDEX IF NOT EXISTS idx_edges_src "
    "ON ledger_edges (tenant_id, src_id, tx_time)",
    "CREATE INDEX IF NOT EXISTS idx_edges_dst "
    "ON ledger_edges (tenant_id, dst_id, tx_time)",
    """
    CREATE TABLE IF NOT EXISTS edge_closure (
      tenant_id TEXT    NOT NULL,
      src_id    TEXT    NOT NULL,
      dst_id    TEXT    NOT NULL,
      min_tx    INTEGER NOT NULL CHECK (min_tx >= 0),
      PRIMARY KEY (tenant_id, src_id, dst_id)
    ) WITHOUT ROWID""",
    "CREATE INDEX IF NOT EXISTS idx_closure_dst "
    "ON edge_closure (tenant_id, dst_id, src_id)",
)

_CLOSURE_UPSERT_SQLITE = (
    "INSERT INTO edge_closure (tenant_id, src_id, dst_id, min_tx) "
    "VALUES (?, ?, ?, ?) "
    "ON CONFLICT (tenant_id, src_id, dst_id) "
    "DO UPDATE SET min_tx = MIN(edge_closure.min_tx, excluded.min_tx)"
)


class SQLiteChallengerBackend(LedgerBackend):
    """In-process sqlite3 engine (the <960 MiB 2D configuration).

    ``cache_mib`` is the page-cache budget and must stay below 960 MiB.  File
    databases use WAL.  Both evaluation modes read the same tables.
    """

    def __init__(self, path: str = ":memory:", *, cache_mib: int = 64) -> None:
        if not isinstance(cache_mib, int) or not 1 <= cache_mib < _SQLITE_MAX_CACHE_MIB:
            raise LedgerError(
                f"cache_mib must be an int in [1, {_SQLITE_MAX_CACHE_MIB})"
            )
        self.path = path
        self.cache_mib = cache_mib
        self._conn = sqlite3.connect(path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(f"PRAGMA cache_size = {-cache_mib * 1024}")
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
        for stmt in _SQLITE_SCHEMA:
            self._conn.execute(stmt)

    # -- writes ----------------------------------------------------------
    def append_record(self, env: LedgerRecordEnvelope) -> None:
        try:
            self._conn.execute(
                "INSERT INTO ledger_ops (tenant_id, record_id, entity_key, "
                "authority_rank, valid_from, valid_to, tx_time, op_kind, "
                "policy_epoch, claim_value) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    env.tenant_id,
                    env.record_id,
                    env.entity_key,
                    env.authority_rank,
                    env.valid_from,
                    env.valid_to,
                    env.tx_time,
                    env.op_kind.value,
                    env.policy_epoch,
                    env.claim_value,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise LedgerError(f"record op rejected: {exc}") from exc

    def append_edge(self, env: LedgerEdgeEnvelope) -> None:
        """Insert the edge and incrementally maintain ``edge_closure``.

        Every path that uses the new edge ``s -> d`` factors as
        ``a ~> s`` + edge + ``d ~> b`` where the two halves avoid the new
        edge, so the already-maintained closure rows suffice.  The pair's
        ``min_tx`` is the bottleneck ``max(tx(a~>s), tx(edge), tx(d~>b))``.
        """
        conn = self._conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "INSERT INTO ledger_edges (tenant_id, edge_id, src_id, dst_id, "
                "tx_time, policy_epoch) VALUES (?,?,?,?,?,?)",
                (
                    env.tenant_id,
                    env.edge_id,
                    env.src_id,
                    env.dst_id,
                    env.tx_time,
                    env.policy_epoch,
                ),
            )
            ups: List[Tuple[str, int]] = [(env.src_id, 0)]
            ups.extend(
                (r["src_id"], r["min_tx"])
                for r in conn.execute(
                    "SELECT src_id, min_tx FROM edge_closure "
                    "WHERE tenant_id = ? AND dst_id = ?",
                    (env.tenant_id, env.src_id),
                )
            )
            downs: List[Tuple[str, int]] = [(env.dst_id, 0)]
            downs.extend(
                (r["dst_id"], r["min_tx"])
                for r in conn.execute(
                    "SELECT dst_id, min_tx FROM edge_closure "
                    "WHERE tenant_id = ? AND src_id = ?",
                    (env.tenant_id, env.dst_id),
                )
            )
            best: Dict[Tuple[str, str], int] = {}
            for a, a_tx in ups:
                for b, b_tx in downs:
                    tx = max(a_tx, env.tx_time, b_tx)
                    key = (a, b)
                    if key not in best or tx < best[key]:
                        best[key] = tx
            conn.executemany(
                _CLOSURE_UPSERT_SQLITE,
                [(env.tenant_id, a, b, tx) for (a, b), tx in best.items()],
            )
            conn.execute("COMMIT")
        except sqlite3.IntegrityError as exc:
            conn.execute("ROLLBACK")
            raise LedgerError(f"edge rejected: {exc}") from exc
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    # -- reads -----------------------------------------------------------
    @staticmethod
    def _params(
        tenant_id: str,
        entity_key: str,
        valid_time: int,
        tx_time: int,
        policy_epoch: Optional[int],
    ) -> Dict[str, Any]:
        return {
            "tenant": tenant_id,
            "entity": entity_key,
            "vt": valid_time,
            "tx": tx_time,
            "epoch": policy_epoch,
        }

    def fetch_candidates(
        self,
        *,
        tenant_id: str,
        entity_key: str,
        valid_time: int,
        tx_time: int,
        policy_epoch: Optional[int],
        mode: str,
    ) -> List[Dict[str, Any]]:
        sql = _sqlite_as_of_sql(mode)
        params = self._params(tenant_id, entity_key, valid_time, tx_time, policy_epoch)
        return [dict(r) for r in self._conn.execute(sql, params)]

    def explain_query_plan(self, mode: str = "recursive_cte") -> List[str]:
        """``EXPLAIN QUERY PLAN`` detail lines (evidence that indexes are used)."""
        sql = "EXPLAIN QUERY PLAN " + _sqlite_as_of_sql(mode)
        params = self._params("t", "e", 0, 0, None)
        return [str(r["detail"]) for r in self._conn.execute(sql, params)]

    # -- closure maintenance --------------------------------------------
    def closure_snapshot(self, tenant_id: Optional[str] = None) -> Dict[Tuple[str, str, str], int]:
        if tenant_id is None:
            cur = self._conn.execute(
                "SELECT tenant_id, src_id, dst_id, min_tx FROM edge_closure"
            )
        else:
            cur = self._conn.execute(
                "SELECT tenant_id, src_id, dst_id, min_tx FROM edge_closure "
                "WHERE tenant_id = ?",
                (tenant_id,),
            )
        return {
            (r["tenant_id"], r["src_id"], r["dst_id"]): r["min_tx"] for r in cur
        }

    def reference_closure(self) -> Dict[Tuple[str, str, str], int]:
        """Closure recomputed from ``ledger_edges`` alone (oracle, no writes)."""
        per_tenant: Dict[str, List[Tuple[str, str, int]]] = {}
        for r in self._conn.execute(
            "SELECT tenant_id, src_id, dst_id, tx_time FROM ledger_edges"
        ):
            per_tenant.setdefault(r["tenant_id"], []).append(
                (r["src_id"], r["dst_id"], r["tx_time"])
            )
        out: Dict[Tuple[str, str, str], int] = {}
        for tenant, edges in per_tenant.items():
            for (src, dst), tx in bottleneck_closure(edges).items():
                out[(tenant, src, dst)] = tx
        return out

    def rebuild_closure(self) -> int:
        """Drop and recompute ``edge_closure`` from the edge log; returns rows."""
        ref = self.reference_closure()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute("DELETE FROM edge_closure")
            self._conn.executemany(
                "INSERT INTO edge_closure (tenant_id, src_id, dst_id, min_tx) "
                "VALUES (?,?,?,?)",
                [(t, s, d, tx) for (t, s, d), tx in sorted(ref.items())],
            )
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        return len(ref)

    # -- misc -------------------------------------------------------------
    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for table in ("ledger_ops", "ledger_edges", "edge_closure"):
            out[table] = self._conn.execute(
                f"SELECT COUNT(*) AS n FROM {table}"
            ).fetchone()["n"]
        return out

    def close(self) -> None:
        self._conn.close()


# --------------------------------------------------------------------------
# PostgreSQL 17 dialect + optional-psycopg engine
# --------------------------------------------------------------------------


class Postgres17SQLDialectBuilder:
    """Generates the PostgreSQL 17 DDL / DML / query text for the challenger.

    Pure string generation: importing and using it never needs psycopg or a
    server.  Differences from the SQLite engine are limited to dialect:
    ``WITH RECURSIVE ... CYCLE record_id SET is_cycle USING path``,
    ``GENERATED ... AS IDENTITY``, a stored ``int8range`` column with GiST
    indexes (for the 6C range-overlap workload), and an ``AFTER INSERT``
    trigger that maintains ``edge_closure`` incrementally.
    """

    MIN_SERVER_MAJOR = 17

    def schema_statements(self) -> List[str]:
        return [
            "CREATE EXTENSION IF NOT EXISTS btree_gist",
            """CREATE TABLE IF NOT EXISTS ledger_ops (
  seq            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  tenant_id      TEXT   NOT NULL,
  record_id      TEXT   NOT NULL,
  entity_key     TEXT   NOT NULL,
  authority_rank BIGINT NOT NULL CHECK (authority_rank >= 0),
  valid_from     BIGINT NOT NULL CHECK (valid_from >= 0),
  valid_to       BIGINT CHECK (valid_to IS NULL OR valid_to > valid_from),
  tx_time        BIGINT NOT NULL CHECK (tx_time >= 0),
  op_kind        TEXT   NOT NULL
                 CHECK (op_kind IN ('ASSERT','WITHDRAW','REINSTATE','CONFLICT')),
  policy_epoch   BIGINT NOT NULL DEFAULT 0 CHECK (policy_epoch >= 0),
  claim_value    TEXT   NOT NULL DEFAULT '',
  valid_range    int8range GENERATED ALWAYS AS (int8range(valid_from, valid_to, '[)')) STORED
)""",
            "CREATE INDEX IF NOT EXISTS idx_ops_entity_bitemporal "
            "ON ledger_ops (tenant_id, entity_key, tx_time, valid_from)",
            "CREATE INDEX IF NOT EXISTS idx_ops_record_tx "
            "ON ledger_ops (tenant_id, record_id, tx_time, seq)",
            "CREATE INDEX IF NOT EXISTS idx_ops_valid_gist "
            "ON ledger_ops USING gist (tenant_id, entity_key, valid_range)",
            """CREATE TABLE IF NOT EXISTS ledger_edges (
  tenant_id    TEXT   NOT NULL,
  edge_id      TEXT   NOT NULL,
  src_id       TEXT   NOT NULL,
  dst_id       TEXT   NOT NULL,
  tx_time      BIGINT NOT NULL CHECK (tx_time >= 0),
  policy_epoch BIGINT NOT NULL DEFAULT 0 CHECK (policy_epoch >= 0),
  PRIMARY KEY (tenant_id, edge_id)
)""",
            "CREATE INDEX IF NOT EXISTS idx_edges_src "
            "ON ledger_edges (tenant_id, src_id, tx_time)",
            "CREATE INDEX IF NOT EXISTS idx_edges_dst "
            "ON ledger_edges (tenant_id, dst_id, tx_time)",
            """CREATE TABLE IF NOT EXISTS edge_closure (
  tenant_id TEXT   NOT NULL,
  src_id    TEXT   NOT NULL,
  dst_id    TEXT   NOT NULL,
  min_tx    BIGINT NOT NULL CHECK (min_tx >= 0),
  PRIMARY KEY (tenant_id, src_id, dst_id)
)""",
            "CREATE INDEX IF NOT EXISTS idx_closure_dst "
            "ON edge_closure (tenant_id, dst_id, src_id)",
            self.closure_trigger_function_ddl(),
            "DROP TRIGGER IF EXISTS trg_edge_closure_maintain ON ledger_edges",
            "CREATE TRIGGER trg_edge_closure_maintain "
            "AFTER INSERT ON ledger_edges FOR EACH ROW "
            "EXECUTE FUNCTION chelated_edge_closure_maintain()",
        ]

    def closure_trigger_function_ddl(self) -> str:
        """Incremental closure maintenance (same factoring as the SQLite path)."""
        return """CREATE OR REPLACE FUNCTION chelated_edge_closure_maintain()
RETURNS trigger LANGUAGE plpgsql AS $fn$
BEGIN
  INSERT INTO edge_closure (tenant_id, src_id, dst_id, min_tx)
  SELECT NEW.tenant_id, u.src_id, d.dst_id,
         MIN(GREATEST(u.min_tx, NEW.tx_time, d.min_tx))
  FROM (
    SELECT NEW.src_id AS src_id, 0::bigint AS min_tx
    UNION ALL
    SELECT k.src_id, k.min_tx FROM edge_closure k
    WHERE k.tenant_id = NEW.tenant_id AND k.dst_id = NEW.src_id
  ) u
  CROSS JOIN (
    SELECT NEW.dst_id AS dst_id, 0::bigint AS min_tx
    UNION ALL
    SELECT k.dst_id, k.min_tx FROM edge_closure k
    WHERE k.tenant_id = NEW.tenant_id AND k.src_id = NEW.dst_id
  ) d
  GROUP BY u.src_id, d.dst_id
  ON CONFLICT (tenant_id, src_id, dst_id)
  DO UPDATE SET min_tx = LEAST(edge_closure.min_tx, EXCLUDED.min_tx);
  RETURN NEW;
END
$fn$"""

    def closure_rebuild_sql(self) -> str:
        """Full closure recompute with a cycle-safe recursive walk.

        The ``DELETE`` comes first so stale rows (edges/paths that no longer exist in the edge log) cannot
        survive: the ``LEAST``-upsert alone only ever LOWERS ``min_tx`` and never removes a row, so a rebuild
        without the DELETE is not a rebuild (mirrors SQLite ``rebuild_closure``'s DELETE-then-INSERT).
        """
        return """DELETE FROM edge_closure;
INSERT INTO edge_closure (tenant_id, src_id, dst_id, min_tx)
SELECT w.tenant_id, w.src_id, w.dst_id, MIN(w.tx_max)
FROM (
  WITH RECURSIVE walk(tenant_id, src_id, dst_id, tx_max) AS (
    SELECT e.tenant_id, e.src_id, e.dst_id, e.tx_time FROM ledger_edges e
    UNION ALL
    SELECT w.tenant_id, w.src_id, e.dst_id, GREATEST(w.tx_max, e.tx_time)
    FROM walk w
    JOIN ledger_edges e ON e.tenant_id = w.tenant_id AND e.src_id = w.dst_id
  ) CYCLE dst_id SET is_cycle USING path
  SELECT tenant_id, src_id, dst_id, tx_max FROM walk
) w
GROUP BY w.tenant_id, w.src_id, w.dst_id
ON CONFLICT (tenant_id, src_id, dst_id)
DO UPDATE SET min_tx = LEAST(edge_closure.min_tx, EXCLUDED.min_tx)"""

    def insert_record_sql(self) -> str:
        return (
            "INSERT INTO ledger_ops (tenant_id, record_id, entity_key, "
            "authority_rank, valid_from, valid_to, tx_time, op_kind, "
            "policy_epoch, claim_value) VALUES (%(tenant)s, %(record)s, "
            "%(entity)s, %(authority)s, %(vfrom)s, %(vto)s, %(tx)s, %(op)s, "
            "%(epoch)s, %(claim)s)"
        )

    def insert_edge_sql(self) -> str:
        return (
            "INSERT INTO ledger_edges (tenant_id, edge_id, src_id, dst_id, "
            "tx_time, policy_epoch) VALUES (%(tenant)s, %(edge)s, %(src)s, "
            "%(dst)s, %(tx)s, %(epoch)s)"
        )

    def as_of_sql(self, mode: str) -> str:
        return _to_psycopg(_pg_as_of_sql(mode))


class Postgres17ChallengerBackend(LedgerBackend):
    """PostgreSQL 17 engine.  Requires ``psycopg`` (v3) and a reachable server.

    ``connect`` may be injected (a callable returning a DB-API-style
    connection) for tests; by default ``psycopg.connect(dsn)`` is imported
    lazily so this module stays importable without psycopg installed.
    """

    def __init__(
        self,
        dsn: str,
        *,
        connect: Optional[Callable[[str], Any]] = None,
        install_schema: bool = True,
        enforce_server_version: bool = True,
    ) -> None:
        self.dsn = dsn
        self.dialect = Postgres17SQLDialectBuilder()
        if connect is None:
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise LedgerError(
                    "Postgres17ChallengerBackend requires the optional 'psycopg' package"
                ) from exc
            self._conn = psycopg.connect(dsn, row_factory=dict_row)
        else:
            self._conn = connect(dsn)
        if enforce_server_version:
            major = self.server_major_version()
            if major < self.dialect.MIN_SERVER_MAJOR:
                raise LedgerError(
                    f"PostgreSQL >= {self.dialect.MIN_SERVER_MAJOR} required, got {major}"
                )
        if install_schema:
            self.install_schema()

    def server_major_version(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SHOW server_version_num")
            row = cur.fetchone()
        value = row["server_version_num"] if isinstance(row, dict) else row[0]
        return int(value) // 10000

    def install_schema(self) -> None:
        with self._conn.cursor() as cur:
            for stmt in self.dialect.schema_statements():
                cur.execute(stmt)
        self._conn.commit()

    def append_record(self, env: LedgerRecordEnvelope) -> None:
        params = {
            "tenant": env.tenant_id,
            "record": env.record_id,
            "entity": env.entity_key,
            "authority": env.authority_rank,
            "vfrom": env.valid_from,
            "vto": env.valid_to,
            "tx": env.tx_time,
            "op": env.op_kind.value,
            "epoch": env.policy_epoch,
            "claim": env.claim_value,
        }
        self._exec_write(self.dialect.insert_record_sql(), params)

    def append_edge(self, env: LedgerEdgeEnvelope) -> None:
        params = {
            "tenant": env.tenant_id,
            "edge": env.edge_id,
            "src": env.src_id,
            "dst": env.dst_id,
            "tx": env.tx_time,
            "epoch": env.policy_epoch,
        }
        self._exec_write(self.dialect.insert_edge_sql(), params)

    def _exec_write(self, sql: str, params: Dict[str, Any]) -> None:
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, params)
            self._conn.commit()
        except Exception as exc:
            self._conn.rollback()
            raise LedgerError(f"postgres write rejected: {exc}") from exc

    def fetch_candidates(
        self,
        *,
        tenant_id: str,
        entity_key: str,
        valid_time: int,
        tx_time: int,
        policy_epoch: Optional[int],
        mode: str,
    ) -> List[Dict[str, Any]]:
        sql = self.dialect.as_of_sql(mode)
        params = {
            "tenant": tenant_id,
            "entity": entity_key,
            "vt": valid_time,
            "tx": tx_time,
            "epoch": policy_epoch,
        }
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        self._conn.rollback()
        return [dict(r) for r in rows]

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        with self._conn.cursor() as cur:
            for table in ("ledger_ops", "ledger_edges", "edge_closure"):
                cur.execute(f"SELECT COUNT(*) AS n FROM {table}")
                row = cur.fetchone()
                out[table] = int(row["n"] if isinstance(row, dict) else row[0])
        self._conn.rollback()
        return out

    def close(self) -> None:
        self._conn.close()
