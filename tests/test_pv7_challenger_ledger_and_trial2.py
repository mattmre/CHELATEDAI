"""Tests: ChallengerLedger (bundle #47), EK5 reason canonicalizer, Trial 2 kit (bundle #64).

unittest only (no pytest).  Run with ``CUDA_VISIBLE_DEVICES=""`` -- nothing here touches a GPU.
The PostgreSQL execution test is opt-in via ``CHELATED_PG17_DSN`` and is skipped otherwise.
"""

from __future__ import annotations

import ast
import json
import os
import random
import re
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_pv7_spine import challenger_ledger as cl  # noqa: E402
from chelated_pv7_spine.challenger_ledger import (  # noqa: E402
    ChallengerBitemporalLedger,
    LedgerEdgeEnvelope,
    LedgerError,
    LedgerRecordEnvelope,
    OpKind,
    Postgres17ChallengerBackend,
    Postgres17SQLDialectBuilder,
    SQLiteChallengerBackend,
)
from chelated_pv7_spine.ek5_reason_canonicalizer import (  # noqa: E402
    CAUSE_CLASSES,
    canonical_cause,
    compare_with_fresh_rebuild,
    export_canonical_ek5_audit_trace,
)
from ek5_v3_stratified import (  # noqa: E402
    EdgeAttestation,
    StratifiedEvent,
    StratifiedEvidenceStore,
)
from partner_kits import trial2_records_governance_kit as t2  # noqa: E402

LEDGER_PATH = REPO_ROOT / "chelated_pv7_spine" / "challenger_ledger.py"
MODES = cl.QUERY_MODES


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def rec(
    record_id: str,
    claim: str,
    *,
    tenant: str = "t",
    entity: str = "e",
    rank: int = 1,
    vfrom: int = 0,
    vto: Optional[int] = None,
    tx: int = 1,
    op: OpKind = OpKind.ASSERT,
    epoch: int = 0,
) -> LedgerRecordEnvelope:
    return LedgerRecordEnvelope(
        tenant_id=tenant,
        record_id=record_id,
        entity_key=entity,
        authority_rank=rank,
        valid_from=vfrom,
        valid_to=vto,
        tx_time=tx,
        op_kind=op,
        policy_epoch=epoch,
        claim_value=claim if op in (OpKind.ASSERT, OpKind.CONFLICT) else "",
    )


def status_op(
    record_id: str, op: OpKind, *, rank: int, tx: int, tenant: str = "t", entity: str = "e", epoch: int = 0
) -> LedgerRecordEnvelope:
    return LedgerRecordEnvelope(
        tenant_id=tenant,
        record_id=record_id,
        entity_key=entity,
        authority_rank=rank,
        valid_from=0,
        valid_to=None,
        tx_time=tx,
        op_kind=op,
        policy_epoch=epoch,
    )


def edge(edge_id: str, src: str, dst: str, tx: int = 1, tenant: str = "t") -> LedgerEdgeEnvelope:
    return LedgerEdgeEnvelope(tenant_id=tenant, edge_id=edge_id, src_id=src, dst_id=dst, tx_time=tx)


def both_modes(ledger: ChallengerBitemporalLedger, *args: Any, **kwargs: Any) -> cl.AsOfQueryResult:
    """Query both evaluation modes, assert they agree exactly, and return one result."""
    a = ledger.query_as_of(*args, mode="recursive_cte", **kwargs)
    b = ledger.query_as_of(*args, mode="closure_table", **kwargs)
    assert a.semantic_key() == b.semantic_key(), (a.to_dict(), b.to_dict())
    assert a.digest() == b.digest()
    return a


# ---------------------------------------------------------------------------
# 0. Independence of the challenger from EK5 (AST assertion)
# ---------------------------------------------------------------------------
class TestChallengerIndependence(unittest.TestCase):
    def test_no_ek5_imports_in_challenger_ledger(self) -> None:
        tree = ast.parse(LEDGER_PATH.read_text(encoding="utf-8"))
        imported: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
                imported.extend(alias.name for alias in node.names)
        self.assertTrue(imported, "AST walk found no imports at all (parser broken?)")
        offenders = [name for name in imported if re.search(r"(^|\.)ek5", name)]
        self.assertEqual(offenders, [], f"challenger_ledger must not import EK5: {offenders}")
        self.assertNotIn("ek5_v3_stratified", imported)

    def test_no_ek5_names_referenced(self) -> None:
        tree = ast.parse(LEDGER_PATH.read_text(encoding="utf-8"))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        self.assertFalse({n for n in names if n.lower().startswith("ek5") or "Stratified" in n})

    def test_module_importable_without_psycopg(self) -> None:
        self.assertNotIn("psycopg", sys.modules.get("chelated_pv7_spine.challenger_ledger").__dict__)


# ---------------------------------------------------------------------------
# 1. Envelope validation
# ---------------------------------------------------------------------------
class TestEnvelopeValidation(unittest.TestCase):
    def test_bad_inputs_rejected(self) -> None:
        with self.assertRaises(LedgerError):
            rec("r,1", "x")  # comma forbidden
        with self.assertRaises(LedgerError):
            rec("", "x")
        with self.assertRaises(LedgerError):
            rec("r", "x", vfrom=5, vto=5)
        with self.assertRaises(LedgerError):
            rec("r", "")  # ASSERT needs a claim
        with self.assertRaises(LedgerError):
            rec("r", "x", tx=-1)
        with self.assertRaises(LedgerError):
            LedgerEdgeEnvelope("t", "e1", "a", "b", tx_time=True)  # type: ignore[arg-type]
        with self.assertRaises(LedgerError):
            LedgerEdgeEnvelope("t", "e1", "a", "b", tx_time=1, op_kind=OpKind.ASSERT)

    def test_query_argument_validation(self) -> None:
        with ChallengerBitemporalLedger() as ledger:
            with self.assertRaises(LedgerError):
                ledger.query_as_of("t", "e", 0, 0, mode="nope")
            with self.assertRaises(LedgerError):
                ledger.query_as_of("t", "e", -1, 0)
            with self.assertRaises(LedgerError):
                ledger.append(edge("e1", "a", "b"))  # type: ignore[arg-type]
            with self.assertRaises(LedgerError):
                ledger.add_edge(rec("r", "x"))  # type: ignore[arg-type]

    def test_duplicate_edge_id_rejected(self) -> None:
        with ChallengerBitemporalLedger() as ledger:
            ledger.add_edge(edge("e1", "a", "b"))
            with self.assertRaises(LedgerError):
                ledger.add_edge(edge("e1", "c", "d"))

    def test_cache_budget_enforced(self) -> None:
        with self.assertRaises(LedgerError):
            SQLiteChallengerBackend(cache_mib=960)
        with self.assertRaises(LedgerError):
            SQLiteChallengerBackend(cache_mib=0)


# ---------------------------------------------------------------------------
# 2. Ledger semantics (every case runs under BOTH evaluators)
# ---------------------------------------------------------------------------
class TestLedgerSemantics(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = ChallengerBitemporalLedger()
        self.addCleanup(self.ledger.close)

    def test_absent_and_bitemporal_visibility(self) -> None:
        L = self.ledger
        self.assertEqual(both_modes(L, "t", "e", 5, 5).status, "ABSENT")
        L.append(rec("r1", "old", vfrom=10, vto=20, tx=3))
        self.assertEqual(both_modes(L, "t", "e", 15, 2).status, "ABSENT")  # not yet known
        self.assertEqual(both_modes(L, "t", "e", 15, 3).winner_claim_value, "old")
        self.assertEqual(both_modes(L, "t", "e", 9, 9).status, "ABSENT")  # before valid_from
        self.assertEqual(both_modes(L, "t", "e", 20, 9).status, "ABSENT")  # valid_to is exclusive
        self.assertEqual(both_modes(L, "t", "e", 19, 9).status, "RESOLVED")

    def test_revision_latest_tx_wins_for_same_record(self) -> None:
        L = self.ledger
        L.append(rec("r1", "v1", tx=1))
        L.append(rec("r1", "v2", tx=5))
        self.assertEqual(both_modes(L, "t", "e", 0, 4).winner_claim_value, "v1")
        self.assertEqual(both_modes(L, "t", "e", 0, 5).winner_claim_value, "v2")

    def test_higher_authority_wins(self) -> None:
        L = self.ledger
        L.append(rec("lo", "low", rank=1, tx=1))
        L.append(rec("hi", "high", rank=5, tx=2))
        r = both_modes(L, "t", "e", 0, 9)
        self.assertEqual((r.status, r.winner_record_id, r.winner_authority_rank), ("RESOLVED", "hi", 5))

    def test_equal_authority_different_claims_escalates_to_disputed(self) -> None:
        L = self.ledger
        L.append(rec("a", "p", rank=3, tx=1))
        L.append(rec("b", "q", rank=3, tx=2))
        r = both_modes(L, "t", "e", 0, 9)
        self.assertEqual(r.status, "DISPUTED")
        self.assertIsNone(r.winner_record_id)  # no winner is invented
        self.assertEqual(r.contending_record_ids, ("a", "b"))
        # retracting one side resolves the dispute
        L.append(status_op("b", OpKind.WITHDRAW, rank=3, tx=3))
        r2 = both_modes(L, "t", "e", 0, 9)
        self.assertEqual((r2.status, r2.winner_record_id), ("RESOLVED", "a"))
        # a higher-authority assertion overrides an unresolved dispute
        L.append(rec("c", "r", rank=9, tx=4))
        self.assertEqual(both_modes(L, "t", "e", 0, 9).winner_record_id, "c")

    def test_equal_authority_same_claim_is_not_disputed(self) -> None:
        L = self.ledger
        L.append(rec("a", "same", rank=2, tx=1))
        L.append(rec("b", "same", rank=2, tx=2))
        self.assertEqual(both_modes(L, "t", "e", 0, 9).status, "RESOLVED")

    def test_conflict_flag_forces_disputed(self) -> None:
        L = self.ledger
        L.append(rec("a", "p", rank=2, tx=1, op=OpKind.CONFLICT))
        r = both_modes(L, "t", "e", 0, 9)
        self.assertEqual((r.status, r.winner_record_id), ("DISPUTED", None))

    def test_withdraw_reinstate_propagation(self) -> None:
        L = self.ledger
        L.append(rec("parent", "base", rank=2, entity="pe", tx=1))
        L.append(rec("child", "derived", rank=2, entity="ce", tx=2))
        L.add_edge(edge("d1", "child", "parent", tx=2))  # child derived from parent
        self.assertEqual(both_modes(L, "t", "ce", 0, 9).status, "RESOLVED")
        L.append(status_op("parent", OpKind.WITHDRAW, rank=2, tx=4, entity="pe"))
        r = both_modes(L, "t", "ce", 0, 9)
        self.assertEqual((r.status, r.retracted_record_ids), ("RETRACTED", ("child",)))
        self.assertEqual(both_modes(L, "t", "pe", 0, 9).status, "RETRACTED")
        # as of before the withdrawal everything is still live
        self.assertEqual(both_modes(L, "t", "ce", 0, 3).status, "RESOLVED")
        L.append(status_op("parent", OpKind.REINSTATE, rank=2, tx=6, entity="pe"))
        self.assertEqual(both_modes(L, "t", "ce", 0, 9).status, "RESOLVED")
        self.assertEqual(both_modes(L, "t", "ce", 0, 5).status, "RETRACTED")

    def test_transitive_ancestor_death_and_missing_parent(self) -> None:
        L = self.ledger
        L.append(rec("a", "A", entity="ea", tx=1))
        L.append(rec("b", "B", entity="eb", tx=1))
        L.append(rec("c", "C", entity="ec", tx=1))
        L.add_edge(edge("e1", "c", "b"))
        L.add_edge(edge("e2", "b", "a"))
        self.assertEqual(both_modes(L, "t", "ec", 0, 9).status, "RESOLVED")
        L.append(status_op("a", OpKind.WITHDRAW, rank=1, tx=3, entity="ea"))
        self.assertEqual(both_modes(L, "t", "ec", 0, 9).status, "RETRACTED")
        # an edge to a record that was never asserted: missing parent => dead
        L.add_edge(edge("e3", "c", "ghost", tx=4))
        L.append(status_op("a", OpKind.REINSTATE, rank=1, tx=5, entity="ea"))
        self.assertEqual(both_modes(L, "t", "ec", 0, 2).status, "RESOLVED")  # withdraw + ghost edge not yet visible
        self.assertEqual(both_modes(L, "t", "ec", 0, 3).status, "RETRACTED")  # withdraw of 'a' now visible
        # reinstated at tx 5, so only the dangling edge to 'ghost' (visible from tx 4) keeps 'c' dead
        self.assertEqual(both_modes(L, "t", "ec", 0, 9).status, "RETRACTED")
        self.assertEqual(both_modes(L, "t", "eb", 0, 9).status, "RESOLVED")  # 'b' never touches 'ghost'

    def test_low_authority_withdraw_is_ignored(self) -> None:
        L = self.ledger
        L.append(rec("hi", "claim", rank=5, tx=1))
        L.append(status_op("hi", OpKind.WITHDRAW, rank=2, tx=2))  # rank 2 < 5
        r = both_modes(L, "t", "e", 0, 9)
        self.assertEqual((r.status, r.winner_record_id), ("RESOLVED", "hi"))
        L.append(status_op("hi", OpKind.WITHDRAW, rank=5, tx=3))  # equal authority is enough
        self.assertEqual(both_modes(L, "t", "e", 0, 9).status, "RETRACTED")
        # a low-authority REINSTATE cannot undo it either
        L.append(status_op("hi", OpKind.REINSTATE, rank=1, tx=4))
        self.assertEqual(both_modes(L, "t", "e", 0, 9).status, "RETRACTED")

    def test_cycle_tolerance(self) -> None:
        L = self.ledger
        for rid in ("a", "b", "c"):
            L.append(rec(rid, rid.upper(), entity=f"e_{rid}", tx=1))
        L.add_edge(edge("x1", "a", "b"))
        L.add_edge(edge("x2", "b", "c"))
        L.add_edge(edge("x3", "c", "a"))  # a cycle never kills by itself
        L.add_edge(edge("x4", "a", "a"))  # self loop
        for ent in ("e_a", "e_b", "e_c"):
            self.assertEqual(both_modes(L, "t", ent, 0, 9).status, "RESOLVED")
        L.append(status_op("c", OpKind.WITHDRAW, rank=1, tx=5, entity="e_c"))
        for ent in ("e_a", "e_b", "e_c"):  # every member of the cycle reaches the dead node
            self.assertEqual(both_modes(L, "t", ent, 0, 9).status, "RETRACTED")

    def test_tenant_isolation(self) -> None:
        L = self.ledger
        L.append(rec("r1", "alpha", tenant="A", rank=1, tx=1))
        L.append(rec("r1", "beta", tenant="B", rank=1, tx=1))
        self.assertEqual(both_modes(L, "A", "e", 0, 9).winner_claim_value, "alpha")
        self.assertEqual(both_modes(L, "B", "e", 0, 9).winner_claim_value, "beta")
        # withdrawal in B must not touch A
        L.append(status_op("r1", OpKind.WITHDRAW, rank=9, tx=2, tenant="B"))
        self.assertEqual(both_modes(L, "A", "e", 0, 9).status, "RESOLVED")
        self.assertEqual(both_modes(L, "B", "e", 0, 9).status, "RETRACTED")
        # a cross-tenant parent is a MISSING parent, not a live one
        L.append(rec("child", "c", tenant="A", entity="ce", tx=3))
        L.append(rec("only_b", "x", tenant="B", entity="xe", tx=3))
        L.add_edge(edge("cross", "child", "only_b", tx=3, tenant="A"))
        self.assertEqual(both_modes(L, "A", "ce", 0, 9).status, "RETRACTED")
        self.assertEqual(both_modes(L, "B", "xe", 0, 9).status, "RESOLVED")
        self.assertEqual(both_modes(L, "A", "xe", 0, 9).status, "ABSENT")

    def test_policy_epoch_gates_record_ops_only(self) -> None:
        L = self.ledger
        L.append(rec("r1", "e0", tx=1, epoch=0))
        L.append(rec("r1", "e2", tx=2, epoch=2))
        self.assertEqual(both_modes(L, "t", "e", 0, 9, policy_epoch=1).winner_claim_value, "e0")
        self.assertEqual(both_modes(L, "t", "e", 0, 9, policy_epoch=2).winner_claim_value, "e2")
        self.assertEqual(both_modes(L, "t", "e", 0, 9).winner_claim_value, "e2")

    def test_counts_and_digest_is_mode_independent(self) -> None:
        L = self.ledger
        L.append(rec("a", "A"))
        L.add_edge(edge("e1", "a", "z"))
        c = L.counts()
        self.assertEqual((c["ledger_ops"], c["ledger_edges"], c["edge_closure"]), (1, 1, 1))
        r1 = L.query_as_of("t", "e", 0, 9, mode="recursive_cte")
        r2 = L.query_as_of("t", "e", 0, 9, mode="closure_table")
        self.assertNotEqual(r1.mode, r2.mode)
        self.assertEqual(r1.digest(), r2.digest())

    def test_query_plans_use_indexes(self) -> None:
        backend = self.ledger.backend
        assert isinstance(backend, SQLiteChallengerBackend)
        for mode in MODES:
            plan = " | ".join(backend.explain_query_plan(mode))
            self.assertIn("idx_", plan, f"{mode} plan uses no index: {plan}")

    def test_file_backend_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "ledger.sqlite")
            with ChallengerBitemporalLedger(SQLiteChallengerBackend(path)) as L:
                L.append(rec("a", "A"))
            with ChallengerBitemporalLedger(SQLiteChallengerBackend(path)) as L2:
                self.assertEqual(both_modes(L2, "t", "e", 0, 9).winner_claim_value, "A")


# ---------------------------------------------------------------------------
# 3. recursive_cte vs closure_table parity on random workloads + closure oracle
# ---------------------------------------------------------------------------
def _random_ledger_ops(rng: random.Random) -> Tuple[List[LedgerRecordEnvelope], List[LedgerEdgeEnvelope]]:
    ids = [f"r{i}" for i in range(8)]
    ents = ["e1", "e2", "e3"]
    tenants = ["A", "B"]
    owner = {(t, r): rng.choice(ents) for t in tenants for r in ids}
    recs: List[LedgerRecordEnvelope] = []
    for _ in range(rng.randint(6, 16)):
        t, r = rng.choice(tenants), rng.choice(ids)
        kind = rng.choice([OpKind.ASSERT] * 4 + [OpKind.CONFLICT, OpKind.WITHDRAW, OpKind.WITHDRAW, OpKind.REINSTATE])
        vfrom = rng.randint(0, 5)
        recs.append(
            LedgerRecordEnvelope(
                tenant_id=t,
                record_id=r,
                entity_key=owner[(t, r)],
                authority_rank=rng.randint(0, 3),
                valid_from=vfrom,
                valid_to=rng.choice([None, vfrom + rng.randint(1, 8)]),
                tx_time=rng.randint(0, 9),
                op_kind=kind,
                policy_epoch=rng.randint(0, 2),
                claim_value=rng.choice(["p", "q"]) if kind in (OpKind.ASSERT, OpKind.CONFLICT) else "",
            )
        )
    edges = [
        LedgerEdgeEnvelope(
            tenant_id=rng.choice(tenants),
            edge_id=f"x{i}",
            src_id=rng.choice(ids),
            dst_id=rng.choice(ids + ["ghost"]),
            tx_time=rng.randint(0, 9),
        )
        for i in range(rng.randint(0, 14))
    ]
    return recs, edges


class TestModeParityAndClosure(unittest.TestCase):
    def test_sqlite_recursive_cte_vs_closure_table_parity_random(self) -> None:
        checked = 0
        for seed in range(60):
            rng = random.Random(seed)
            recs, edges = _random_ledger_ops(rng)
            with ChallengerBitemporalLedger() as L:
                for env in recs:
                    L.append(env)
                for env in edges:
                    L.add_edge(env)
                for tenant in ("A", "B"):
                    for ent in ("e1", "e2", "e3"):
                        for vt, tx, ep in ((0, 9, None), (3, 5, 1), (6, 2, 2), (2, 9, 0)):
                            a = L.query_as_of(tenant, ent, vt, tx, "recursive_cte", ep)
                            b = L.query_as_of(tenant, ent, vt, tx, "closure_table", ep)
                            self.assertEqual(a.semantic_key(), b.semantic_key(), f"seed={seed}")
                            checked += 1
        self.assertEqual(checked, 60 * 2 * 3 * 4)

    def test_incremental_closure_equals_reference_and_rebuild_random_order_with_cycles(self) -> None:
        nodes = [f"n{i}" for i in range(7)]
        for seed in range(80):
            rng = random.Random(1000 + seed)
            edges = [
                LedgerEdgeEnvelope(
                    tenant_id=rng.choice(["A", "B"]),
                    edge_id=f"x{i}",
                    src_id=rng.choice(nodes),
                    dst_id=rng.choice(nodes),  # src==dst and back-edges => cycles
                    tx_time=rng.randint(0, 6),
                )
                for i in range(rng.randint(1, 18))
            ]
            backend = SQLiteChallengerBackend()
            self.addCleanup(backend.close)
            for env in edges:
                backend.append_edge(env)
                # the invariant must hold after EVERY insert, not just at the end
                self.assertEqual(backend.closure_snapshot(), backend.reference_closure(), f"seed={seed}")
            incremental = backend.closure_snapshot()
            n_rows = backend.rebuild_closure()
            self.assertEqual(n_rows, len(incremental))
            self.assertEqual(backend.closure_snapshot(), incremental, f"seed={seed}")
            # insertion order must not matter
            shuffled = edges[:]
            rng.shuffle(shuffled)
            other = SQLiteChallengerBackend()
            self.addCleanup(other.close)
            for env in shuffled:
                other.append_edge(env)
            self.assertEqual(other.closure_snapshot(), incremental, f"seed={seed} (order)")

    def test_closure_bottleneck_tx_semantics(self) -> None:
        b = SQLiteChallengerBackend()
        self.addCleanup(b.close)
        b.append_edge(edge("e1", "a", "b", tx=2))
        b.append_edge(edge("e2", "b", "c", tx=7))
        b.append_edge(edge("e3", "a", "c", tx=9))
        snap = b.closure_snapshot("t")
        self.assertEqual(snap[("t", "a", "c")], 7)  # min over paths of max edge tx
        self.assertEqual(snap[("t", "a", "b")], 2)
        b.append_edge(edge("e4", "c", "a", tx=1))  # closes a cycle
        self.assertIn(("t", "a", "a"), b.closure_snapshot("t"))
        self.assertEqual(b.closure_snapshot(), b.reference_closure())


# ---------------------------------------------------------------------------
# 4. PostgreSQL 17 dialect (string-level, no server) + injected fake connection
# ---------------------------------------------------------------------------
class _FakeCursor:
    def __init__(self, conn: "_FakeConn") -> None:
        self.conn = conn
        self._rows: List[Any] = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: Optional[Dict[str, Any]] = None) -> None:
        self.conn.executed.append((sql, params))
        if sql.strip() == "SHOW server_version_num":
            self._rows = [{"server_version_num": str(self.conn.version_num)}]
        elif sql.lstrip().upper().startswith("WITH RECURSIVE"):
            self._rows = []
        elif sql.lstrip().upper().startswith("SELECT COUNT"):
            self._rows = [{"n": 0}]
        else:
            self._rows = []

    def fetchone(self) -> Any:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> List[Any]:
        return list(self._rows)


class _FakeConn:
    def __init__(self, version_num: int = 170002) -> None:
        self.version_num = version_num
        self.executed: List[Tuple[str, Optional[Dict[str, Any]]]] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


_PG_KEYS = re.compile(r"%\(([a-z_]+)\)s")


class TestPostgres17Dialect(unittest.TestCase):
    def setUp(self) -> None:
        self.d = Postgres17SQLDialectBuilder()

    def test_recursive_cte_uses_pg17_cycle_clause(self) -> None:
        sql = self.d.as_of_sql("recursive_cte")
        self.assertIn("CYCLE record_id SET is_cycle USING path", sql)
        self.assertIn("WITH RECURSIVE", sql)
        self.assertNotIn("visited_path", sql)  # the SQLite-only string guard must not leak
        self.assertNotIn("instr(", sql)

    def test_closure_mode_has_no_recursive_walk_of_edges(self) -> None:
        sql = self.d.as_of_sql("closure_table")
        self.assertIn("edge_closure", sql)
        self.assertNotIn("CYCLE record_id", sql)
        self.assertNotIn("JOIN ledger_edges", sql)

    def test_schema_has_gist_trigger_on_conflict(self) -> None:
        stmts = self.d.schema_statements()
        blob = "\n".join(stmts)
        self.assertIn("USING gist", blob)
        self.assertIn("btree_gist", blob)
        self.assertIn("int8range", blob)
        self.assertIn("GENERATED ALWAYS AS IDENTITY", blob)
        self.assertIn("CREATE TRIGGER trg_edge_closure_maintain", blob)
        self.assertIn("AFTER INSERT ON ledger_edges", blob)
        self.assertIn("ON CONFLICT (tenant_id, src_id, dst_id)", self.d.closure_trigger_function_ddl())
        self.assertIn("ON CONFLICT (tenant_id, src_id, dst_id)", self.d.closure_rebuild_sql())
        self.assertIn("CYCLE dst_id SET is_cycle USING path", self.d.closure_rebuild_sql())
        # the trigger DDL must come before the CREATE TRIGGER that references it
        fn_idx = next(i for i, s in enumerate(stmts) if "CREATE OR REPLACE FUNCTION" in s)
        trg_idx = next(i for i, s in enumerate(stmts) if s.startswith("CREATE TRIGGER"))
        self.assertLess(fn_idx, trg_idx)

    def test_placeholders_are_psycopg_style_and_have_no_casts(self) -> None:
        for mode in MODES:
            sql = self.d.as_of_sql(mode)
            self.assertNotRegex(sql, r":[a-z_]+\b(?<!::)", "named ':x' marker left unconverted")
            self.assertNotIn("::", sql)
            self.assertIn("%(tenant)s", sql)

    def test_param_key_consistency_across_all_statements(self) -> None:
        record_keys = set(_PG_KEYS.findall(self.d.insert_record_sql()))
        edge_keys = set(_PG_KEYS.findall(self.d.insert_edge_sql()))
        self.assertEqual(record_keys, {"tenant", "record", "entity", "authority", "vfrom", "vto", "tx", "op", "epoch", "claim"})
        self.assertEqual(edge_keys, {"tenant", "edge", "src", "dst", "tx", "epoch"})
        for mode in MODES:
            self.assertEqual(set(_PG_KEYS.findall(self.d.as_of_sql(mode))), {"tenant", "entity", "vt", "tx", "epoch"})

    def test_backend_binds_exactly_the_keys_the_sql_expects(self) -> None:
        conn = _FakeConn()
        backend = Postgres17ChallengerBackend("fake://dsn", connect=lambda dsn: conn)
        ledger = ChallengerBitemporalLedger(backend)
        ledger.append(rec("r1", "x"))
        ledger.add_edge(edge("e1", "r1", "r0"))
        for mode in MODES:
            ledger.query_as_of("t", "e", 1, 1, mode=mode, policy_epoch=3)
        writes = [(s, p) for s, p in conn.executed if s.startswith("INSERT INTO ledger_")]
        queries = [(s, p) for s, p in conn.executed if s.lstrip().startswith("WITH RECURSIVE")]
        self.assertEqual(len(writes), 2)
        self.assertEqual(len(queries), 2)
        for sql, params in writes + queries:
            assert params is not None
            self.assertEqual(set(_PG_KEYS.findall(sql)), set(params), sql[:60])
        self.assertGreaterEqual(conn.commits, 3)  # schema + 2 writes
        ledger.close()
        self.assertTrue(conn.closed)

    def test_server_version_enforced(self) -> None:
        with self.assertRaises(LedgerError):
            Postgres17ChallengerBackend("fake://", connect=lambda dsn: _FakeConn(160009))
        Postgres17ChallengerBackend("fake://", connect=lambda dsn: _FakeConn(170000))
        Postgres17ChallengerBackend("fake://", connect=lambda dsn: _FakeConn(160009), enforce_server_version=False)

    def test_write_failure_rolls_back_and_raises_ledger_error(self) -> None:
        class Boom(_FakeConn):
            def cursor(self) -> _FakeCursor:
                cur = super().cursor()
                original = cur.execute

                def execute(sql: str, params: Optional[Dict[str, Any]] = None) -> None:
                    if sql.startswith("INSERT INTO ledger_ops"):
                        raise RuntimeError("constraint violated")
                    original(sql, params)

                cur.execute = execute  # type: ignore[method-assign]
                return cur

        conn = Boom()
        backend = Postgres17ChallengerBackend("fake://", connect=lambda dsn: conn, install_schema=False)
        with self.assertRaises(LedgerError):
            backend.append_record(rec("r1", "x"))
        self.assertEqual(conn.rollbacks, 1)

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(LedgerError):
            self.d.as_of_sql("bogus")


@unittest.skipUnless(os.environ.get("CHELATED_PG17_DSN"), "set CHELATED_PG17_DSN to run against a live PostgreSQL >= 17")
class TestPostgres17Live(unittest.TestCase):
    """Opt-in: executes the real dialect against a server and checks SQLite parity."""

    def test_pg_matches_sqlite_on_random_workloads(self) -> None:
        dsn = os.environ["CHELATED_PG17_DSN"]
        for seed in range(8):
            rng = random.Random(seed)
            recs, edges = _random_ledger_ops(rng)
            pg = Postgres17ChallengerBackend(dsn)
            with pg._conn.cursor() as cur:
                cur.execute("TRUNCATE ledger_ops, ledger_edges, edge_closure RESTART IDENTITY")
            pg._conn.commit()
            pg_ledger = ChallengerBitemporalLedger(pg)
            sq_ledger = ChallengerBitemporalLedger()
            try:
                for env in recs:
                    pg_ledger.append(env)
                    sq_ledger.append(env)
                for env in edges:
                    pg_ledger.add_edge(env)
                    sq_ledger.add_edge(env)
                for tenant in ("A", "B"):
                    for ent in ("e1", "e2", "e3"):
                        for mode in MODES:
                            a = pg_ledger.query_as_of(tenant, ent, 3, 6, mode, 2)
                            b = sq_ledger.query_as_of(tenant, ent, 3, 6, mode, 2)
                            self.assertEqual(a.semantic_key(), b.semantic_key(), f"seed={seed} {tenant}/{ent}/{mode}")
            finally:
                pg_ledger.close()
                sq_ledger.close()


# ---------------------------------------------------------------------------
# 5. EK5 reason canonicalizer
# ---------------------------------------------------------------------------
_DENSE_ETYPES = [
    "SUPERSEDE",
    "VERSION_LINEAGE_SUPERSEDE",
    "VERSION_LINEAGE_SUPERSEDE",
    "REFUTE",
    "DERIVES_CURRENT",
    "COPY_OF",
    "AMENDS",
]
_MIXED_ETYPES = [
    "REFUTE",
    "REFUTE",
    "SUPERSEDE",
    "VERSION_LINEAGE_SUPERSEDE",
    "COPY_OF",
    "DERIVES_CURRENT",
    "AMENDS",
    "DERIVES_SNAPSHOT",
    "CITES_AUDIT",
]


def run_random_ek5_sequence(seed: int, strict: bool, dense: bool) -> Optional[StratifiedEvidenceStore]:
    """12 random incremental mutations on <= 10 events; None if the engine rejected the batch."""
    rng = random.Random(seed)
    st = StratifiedEvidenceStore(strict_dag_supersede=strict)
    n_ev = 0
    n_att = 0
    for _ in range(12):
        evs: List[StratifiedEvent] = []
        for _ in range(rng.choice([0, 1, 1, 2])):
            if n_ev >= 10:
                break
            evs.append(
                StratifiedEvent(
                    event_id=f"e{n_ev}",
                    entity="a" if dense else rng.choice("ab"),
                    slot="x" if dense else rng.choice("xy"),
                    claim_value=rng.choice(["p", "q"]),
                    tenant="t",
                    domain=rng.choice(["general", "legal", "med"]),
                    authority_tier=rng.randint(0, 3),
                )
            )
            n_ev += 1
        pool = sorted(st.events) + [e.event_id for e in evs]
        atts: List[EdgeAttestation] = []
        sup_used = False
        if len(pool) >= 2:
            for _ in range(rng.choice([0, 1, 2, 3, 4] if dense else [0, 1, 2, 3])):
                s, d = rng.sample(pool, 2) if rng.random() < 0.97 else (pool[0], pool[0])
                et = rng.choice(_DENSE_ETYPES if dense else _MIXED_ETYPES)
                lin = None
                if et in ("SUPERSEDE", "VERSION_LINEAGE_SUPERSEDE"):
                    if s == d:
                        continue
                    if strict:
                        if sup_used:
                            continue
                        if s in st.events and d in st.events:
                            out = st._raw_sup_out if et == "SUPERSEDE" else st._raw_lsup_out
                            if st._would_close_supersede_cycle(s, d, out):
                                continue
                        sup_used = True
                    if et == "VERSION_LINEAGE_SUPERSEDE":
                        evmap = {e.event_id: e for e in evs}
                        a = st.events.get(s) or evmap.get(s)
                        b = st.events.get(d) or evmap.get(d)
                        if (a.entity, a.slot) != (b.entity, b.slot):
                            continue
                        lin = "L1"
                n_att += 1
                atts.append(EdgeAttestation(f"a{n_att}", s, d, et, lineage_id=lin))
        active = [a for a, x in st.attestations.items() if x.active]
        withdrawn = rng.sample(active, min(len(active), rng.choice([0, 0, 1, 2]))) if active else []
        try:
            st.apply_mutation(evs, atts, withdrawn)
        except Exception:  # noqa: BLE001 - engine-rejected batch (e.g. strict cycle); skip this seed
            return None
    return st


class TestReasonCanonicalizer(unittest.TestCase):
    def test_canonical_cause_mapping_is_total_and_keeps_causes_distinct(self) -> None:
        self.assertEqual(canonical_cause("UNFOUNDED_CYCLE"), "UNFOUNDED_POSITIVE_CYCLE")
        self.assertEqual(canonical_cause("SUPERSEDED_ANCESTOR"), "SUPERSEDED")
        self.assertEqual(canonical_cause("EPISTEMIC_CONTESTED"), "PEER_CONFLICT_DISPUTED")
        self.assertEqual(canonical_cause("TEMPORAL_ANCESTOR_CONTESTED"), "TEMPORAL_DISPUTED")
        with self.assertRaises(ValueError):
            canonical_cause("NOT_A_REASON")
        must_stay_distinct = {
            canonical_cause("SUPERSEDED"),
            canonical_cause("REFUTED"),
            canonical_cause("UNFOUNDED_CYCLE"),
            canonical_cause("EPISTEMIC_CONTESTED"),
            canonical_cause("TEMPORAL_CONTESTED"),
            canonical_cause("MISSING_MANDATORY_PARENT"),
        }
        self.assertEqual(len(must_stay_distinct), 6)
        self.assertLessEqual(must_stay_distinct, CAUSE_CLASSES)

    def _trace(self, st: StratifiedEvidenceStore) -> Dict[str, Any]:
        trace = export_canonical_ek5_audit_trace(st)
        cmp = compare_with_fresh_rebuild(st, trace)
        self.assertEqual(cmp.semantic_diffs, ())
        self.assertEqual(cmp.canonical_reason_diffs, ())
        self.assertEqual(cmp.canonical_pos_cyc_diffs, ())
        self.assertEqual(cmp.canonical_semantic_mismatches, ())
        return trace.by_id()

    def test_handcrafted_each_cause_class_is_distinct(self) -> None:
        ev = StratifiedEvent
        att = EdgeAttestation

        refuted = StratifiedEvidenceStore()
        refuted.apply_mutation([ev("a", claim_value="p", tenant="t"), ev("b", claim_value="q", tenant="t")], [att("r1", "b", "a", "REFUTE")])
        n = self._trace(refuted)
        self.assertEqual((n["a"].reason, n["a"].cause, n["a"].val), ("REFUTED", "REFUTED", "F"))

        superseded = StratifiedEvidenceStore()
        superseded.apply_mutation(
            [ev("a", claim_value="p", tenant="t"), ev("b", claim_value="q", tenant="t"), ev("c", claim_value="z", entity="z", tenant="t")],
            [att("s1", "b", "a", "SUPERSEDE"), att("d1", "c", "a", "DERIVES_CURRENT")],
        )
        n = self._trace(superseded)
        self.assertEqual((n["a"].reason, n["a"].cause), ("SUPERSEDED", "SUPERSEDED"))
        self.assertEqual((n["c"].reason, n["c"].cause), ("SUPERSEDED_ANCESTOR", "SUPERSEDED"))
        self.assertEqual(n["b"].reason, "CURRENT")

        cyc = StratifiedEvidenceStore()
        cyc.apply_mutation([ev("a", claim_value="p", tenant="t"), ev("b", claim_value="p", tenant="t")], [att("c1", "a", "b", "COPY_OF"), att("c2", "b", "a", "COPY_OF")])
        trace = export_canonical_ek5_audit_trace(cyc)
        n = trace.by_id()
        self.assertEqual((n["a"].cause, n["a"].pos_cyc, n["a"].val), ("UNFOUNDED_POSITIVE_CYCLE", True, "F"))
        self.assertEqual(trace.positive_cycles, (("a", "b"),))
        self._trace(cyc)

        peer = StratifiedEvidenceStore()
        peer.apply_mutation(
            [ev("a", claim_value="p", tenant="t", authority_tier=1), ev("b", claim_value="q", tenant="t", authority_tier=1)],
            [att("r1", "a", "b", "REFUTE"), att("r2", "b", "a", "REFUTE")],
        )
        n = self._trace(peer)
        self.assertEqual((n["a"].cause, n["a"].val), ("PEER_CONFLICT_DISPUTED", "U"))

        supcyc = StratifiedEvidenceStore(strict_dag_supersede=False)
        supcyc.apply_mutation(
            [ev("a", claim_value="p", tenant="t"), ev("b", claim_value="q", tenant="t"), ev("c", claim_value="z", entity="z", tenant="t")],
            [att("s1", "b", "a", "SUPERSEDE"), att("s2", "a", "b", "SUPERSEDE"), att("d1", "c", "a", "DERIVES_CURRENT")],
        )
        trace = export_canonical_ek5_audit_trace(supcyc)
        self.assertEqual(trace.supersede_cycles, (("a", "b"),))
        n = self._trace(supcyc)
        self.assertEqual((n["a"].reason, n["a"].cause), ("TEMPORAL_CONTESTED", "TEMPORAL_DISPUTED"))
        self.assertEqual((n["c"].reason, n["c"].cause), ("TEMPORAL_ANCESTOR_CONTESTED", "TEMPORAL_DISPUTED"))

    def test_missing_mandatory_parent_via_as_of_snapshot(self) -> None:
        ev, att = StratifiedEvent, EdgeAttestation
        st = StratifiedEvidenceStore()
        st.apply_mutation(
            [ev("a", claim_value="p", tenant="t", valid_end=5), ev("b", claim_value="q", entity="z", tenant="t")],
            [att("d1", "b", "a", "DERIVES_CURRENT")],
        )
        snap = st.as_of(10, 10)  # the parent's validity ended before the cutoff
        n = self._trace(snap)
        self.assertEqual((n["b"].reason, n["b"].cause, n["b"].val), ("MISSING_MANDATORY_PARENT", "MISSING_MANDATORY_PARENT", "F"))
        self.assertNotIn("a", n)

    def test_deterministic_digest_excludes_timing(self) -> None:
        st = run_random_ek5_sequence(3, strict=False, dense=True)
        assert st is not None
        t1 = export_canonical_ek5_audit_trace(st)
        t2_ = export_canonical_ek5_audit_trace(st)
        self.assertEqual(t1.digest(), t2_.digest())
        self.assertEqual(json.dumps(t1.to_dict(), sort_keys=True).count("export_ms"), json.dumps(t2_.to_dict(), sort_keys=True).count("export_ms"))

    def test_randomized_sequences_match_fresh_rebuild(self) -> None:
        runs = 0
        nodes = 0
        raw_divergent_runs = 0
        causes_seen: set = set()
        worst_ms = 0.0
        for seed in range(420):
            strict = seed % 2 == 0
            dense = (seed // 2) % 2 == 0
            st = run_random_ek5_sequence(seed, strict, dense)
            if st is None:
                continue
            trace = export_canonical_ek5_audit_trace(st)
            cmp = compare_with_fresh_rebuild(st, trace)
            runs += 1
            nodes += cmp.node_count
            worst_ms = max(worst_ms, trace.export_ms)
            self.assertEqual(cmp.semantic_diffs, (), f"seed={seed}")
            self.assertEqual(cmp.canonical_reason_diffs, (), f"seed={seed}")
            self.assertEqual(cmp.canonical_pos_cyc_diffs, (), f"seed={seed}")
            self.assertEqual(cmp.canonical_semantic_mismatches, (), f"seed={seed}")
            if cmp.raw_reason_diffs or cmp.raw_pos_cyc_diffs:
                raw_divergent_runs += 1
            for node in trace.nodes:
                causes_seen.add(node.cause)
                # a stored cause must always be the mapping of the stored reason
                self.assertEqual(node.cause, canonical_cause(node.reason))
        self.assertGreaterEqual(runs, 200, f"only {runs} usable sequences")
        self.assertGreater(nodes, 800)
        # property: the generator exercises more than one distinct cause; none collapse to a single class
        self.assertGreaterEqual(len(causes_seen), 4, f"causes seen: {sorted(causes_seen)}")
        for must in ("CURRENT", "SUPERSEDED", "REFUTED"):
            self.assertIn(must, causes_seen)
        # honest-limit probe (not a pass condition on correctness): small stores export fast
        self.assertLess(worst_ms, 250.0, "canonical export unexpectedly slow on <=10-event stores")
        # the raw incremental telemetry is the thing being fixed; record how often it diverges
        self.assertGreaterEqual(raw_divergent_runs, 0)

    def test_distinct_causes_never_collapse_in_aggregate(self) -> None:
        seen_reason_to_cause: Dict[str, set] = {}
        for seed in range(0, 420, 3):
            st = run_random_ek5_sequence(seed, seed % 2 == 0, (seed // 2) % 2 == 0)
            if st is None:
                continue
            for node in export_canonical_ek5_audit_trace(st).nodes:
                seen_reason_to_cause.setdefault(node.reason, set()).add(node.cause)
        for reason, causes in seen_reason_to_cause.items():
            self.assertEqual(len(causes), 1, f"{reason} mapped to several causes: {causes}")
        # SUPERSEDED / REFUTED / UNFOUNDED_POSITIVE_CYCLE / PEER_CONFLICT_DISPUTED are 4 distinct classes
        four = {canonical_cause(r) for r in ("SUPERSEDED", "REFUTED", "UNFOUNDED_CYCLE", "EPISTEMIC_CONTESTED")}
        self.assertEqual(len(four), 4)

    def test_canonicalizer_does_not_mutate_store(self) -> None:
        st = run_random_ek5_sequence(11, strict=False, dense=False)
        assert st is not None
        before = (dict(st.val), dict(st.cur), dict(st.eligibility), dict(st.reason), dict(st.pos_cyc))
        export_canonical_ek5_audit_trace(st)
        compare_with_fresh_rebuild(st)
        after = (dict(st.val), dict(st.cur), dict(st.eligibility), dict(st.reason), dict(st.pos_cyc))
        self.assertEqual(before, after)


# ---------------------------------------------------------------------------
# 6. Trial 2 records-owner governance kit
# ---------------------------------------------------------------------------
def _score(scenario: str) -> Dict[str, Any]:
    rows, base_rate, key = t2.build_open_dev_fixture(scenario)
    return t2.score_trial2_rows(rows, declared_base_rate=base_rate, unblinding_key=key, data_origin="open_dev_fixture", fixture_label=scenario)


class TestTrial2Kit(unittest.TestCase):
    def test_status_is_protocol_ready_for_every_open_dev_fixture(self) -> None:
        for scenario in t2.DEV_FIXTURE_SCENARIOS:
            r = _score(scenario)
            self.assertEqual(r["status"], "protocol-ready")
            self.assertEqual(r["claim_tier"], "T4")
            self.assertEqual(r["bundle_id"], 64)
            self.assertEqual(r["data_origin"], "open_dev_fixture")
            self.assertEqual(r["protocol_digest"], t2.PROTOCOL_DIGEST)

    def test_win_loss_tie_timeout_insufficient_outcomes(self) -> None:
        expected = {
            "win": ("win", "EK5_PREMIUM_RETAINED"),
            "loss": ("loss", "EK5_PREMIUM_RETIRED"),
            "tie": ("tie", "EK5_PREMIUM_RETIRED"),
            "timeout": ("timeout", "EK5_PREMIUM_RETIRED"),
            "insufficient_natural_conflicts": ("insufficient_natural_conflicts", "INSUFFICIENT_EVIDENCE"),
        }
        for scenario, (outcome, verdict) in expected.items():
            r = _score(scenario)
            self.assertEqual((r["outcome"], r["verdict"]), (outcome, verdict), scenario)

    def test_win_requires_both_comparisons_both_passes_and_faithfulness(self) -> None:
        r = _score("win")
        self.assertEqual(r["ek5_premium"], "retained")
        self.assertEqual(r["retirement_causes"], [])
        for comp in r["comparisons"].values():
            for p in ("attribution_pass", "operational_pass"):
                self.assertTrue(comp[p]["clears_margin"])
                self.assertGreaterEqual(comp[p]["lower_confidence_bound"], comp[p]["margin"])
        self.assertTrue(r["faithfulness_noninferior_to_challenger"])

    def test_loss_and_tie_retire_the_premium_with_causes(self) -> None:
        loss = _score("loss")
        self.assertIn("FAITHFULNESS_INFERIOR_TO_CHALLENGER", loss["retirement_causes"])
        self.assertIn("VS_CHALLENGER_ATTRIBUTION_MARGIN_NOT_CLEARED", loss["retirement_causes"])
        self.assertLess(loss["comparisons"]["ek5_on_vs_challenger"]["attribution_pass"]["mean_advantage"], 0)
        tie = _score("tie")
        self.assertEqual(tie["comparisons"]["ek5_on_vs_challenger"]["attribution_pass"]["mean_advantage"], 0.0)
        self.assertEqual(tie["ek5_premium"], "retired")  # a tie is a miss: sealed as verified-loss

    def test_timeout_scenario_hits_ceiling_even_though_loss_advantage_is_positive(self) -> None:
        r = _score("timeout")
        self.assertGreater(r["timeout_rate_by_arm"]["ek5_on"], t2.MAX_TIMEOUT_RATE)
        self.assertEqual(r["retirement_causes"], ["EK5_ON_TIMEOUT_RATE_EXCEEDS_LIMIT"])
        self.assertGreater(r["comparisons"]["ek5_on_vs_challenger"]["attribution_pass"]["lower_confidence_bound"], 0)

    def test_insufficient_natural_conflicts_never_scores(self) -> None:
        r = _score("insufficient_natural_conflicts")
        self.assertEqual(r["sizing"]["natural_conflicts"], 20)
        self.assertFalse(r["sizing"]["sizing_met"])
        self.assertEqual(r["ek5_premium"], "undetermined")
        self.assertNotIn("comparisons", r)

    def test_joint_rule_one_comparison_failing_retires_premium(self) -> None:
        rows, base_rate, key = t2.build_open_dev_fixture("win")
        # make ek5_off as good as ek5_on => EK5-on vs EK5-off no longer clears
        patched: List[t2.MatterRow] = []
        for row in rows:
            arms = dict(row.arms)
            if row.natural_conflict:
                arms[t2.ARM_EK5_OFF] = t2.ArmResult(
                    arms[t2.ARM_EK5_ON].outcome, arms[t2.ARM_EK5_ON].extraction_minutes, arms[t2.ARM_EK5_ON].review_minutes
                )
            patched.append(t2.MatterRow(row.matter_seq, row.matter_id, row.natural_conflict, arms, row.rubric))
        r = t2.score_trial2_rows(patched, declared_base_rate=base_rate, unblinding_key=key, data_origin="open_dev_fixture")
        self.assertEqual(r["ek5_premium"], "retired")
        self.assertTrue(r["comparisons"]["ek5_on_vs_challenger"]["attribution_pass"]["clears_margin"])
        self.assertFalse(r["comparisons"]["ek5_on_vs_ek5_off"]["attribution_pass"]["clears_margin"])
        self.assertIn("VS_EK5_OFF_ATTRIBUTION_MARGIN_NOT_CLEARED", r["retirement_causes"])
        self.assertNotIn("VS_CHALLENGER_ATTRIBUTION_MARGIN_NOT_CLEARED", r["retirement_causes"])

    def test_operational_pass_can_overturn_an_attribution_win(self) -> None:
        rows, base_rate, key = t2.build_open_dev_fixture("win")
        patched: List[t2.MatterRow] = []
        for row in rows:
            arms = dict(row.arms)
            if row.natural_conflict:
                on = arms[t2.ARM_EK5_ON]
                arms[t2.ARM_EK5_ON] = t2.ArmResult(on.outcome, 500.0, 500.0)  # crushing labor cost
            patched.append(t2.MatterRow(row.matter_seq, row.matter_id, row.natural_conflict, arms, row.rubric))
        r = t2.score_trial2_rows(patched, declared_base_rate=base_rate, unblinding_key=key, data_origin="open_dev_fixture")
        c = r["comparisons"]["ek5_on_vs_challenger"]
        self.assertTrue(c["attribution_pass"]["clears_margin"])
        self.assertFalse(c["operational_pass"]["clears_margin"])
        self.assertEqual(r["ek5_premium"], "retired")

    def test_base_rate_sizing_is_exact_ceil(self) -> None:
        self.assertEqual(t2.required_consecutive_matters(0.3), 100)  # float 0.3 must not round to 101
        self.assertEqual(t2.required_consecutive_matters(0.1), 300)
        self.assertEqual(t2.required_consecutive_matters(1), 30)
        self.assertEqual(t2.required_consecutive_matters(Fraction(1, 3)), 90)
        self.assertEqual(t2.required_consecutive_matters(0.07), 429)  # 30/0.07 = 428.57
        for bad in (0, -0.1, 1.01, True, "0.3"):
            with self.assertRaises(t2.TrialProtocolError):
                t2.required_consecutive_matters(bad)  # type: ignore[arg-type]

    def test_too_few_matters_for_declared_base_rate_is_insufficient(self) -> None:
        rows, _, key = t2.build_open_dev_fixture("win")
        r = t2.score_trial2_rows(rows, declared_base_rate=0.2, unblinding_key=key, data_origin="open_dev_fixture")  # needs 150
        self.assertEqual(r["verdict"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(r["sizing"]["required_consecutive_matters"], 150)

    def test_gaps_duplicates_and_empty_input_are_protocol_errors(self) -> None:
        rows, base_rate, key = t2.build_open_dev_fixture("win")
        with self.assertRaises(t2.TrialProtocolError):
            t2.score_trial2_rows(rows[:10] + rows[11:], declared_base_rate=base_rate, unblinding_key=key, data_origin="partner")
        with self.assertRaises(t2.TrialProtocolError):
            t2.score_trial2_rows([rows[0], rows[0]], declared_base_rate=base_rate, unblinding_key=key, data_origin="partner")
        with self.assertRaises(t2.TrialProtocolError):
            t2.score_trial2_rows([], declared_base_rate=base_rate, unblinding_key=key, data_origin="partner")
        with self.assertRaises(t2.TrialProtocolError):
            t2.score_trial2_rows(rows, declared_base_rate=base_rate, unblinding_key=key, data_origin="made_up")

    def test_row_and_rubric_validation(self) -> None:
        good = {a: t2.ArmResult("correct") for a in t2.MANDATORY_ARMS}
        with self.assertRaises(t2.TrialProtocolError):
            t2.ArmResult("maybe")
        with self.assertRaises(t2.TrialProtocolError):
            t2.ArmResult("correct", extraction_minutes=-1)
        with self.assertRaises(t2.TrialProtocolError):
            t2.MatterRow(0, "m", True, {t2.ARM_EK5_ON: good[t2.ARM_EK5_ON]}, {})  # missing arms
        with self.assertRaises(t2.TrialProtocolError):
            # rubric keyed by arm name would unblind the reviewer
            t2.MatterRow(0, "m", True, good, {"ek5_on": {c: 1 for c in t2.RUBRIC_CRITERIA}})
        with self.assertRaises(t2.TrialProtocolError):
            t2.MatterRow(0, "m", True, good, {"X1": {c: 2 for c in t2.RUBRIC_CRITERIA}})

    def test_blinding_is_deterministic_and_a_bijection(self) -> None:
        a = t2.blind_codes("matter-1", "salt")
        self.assertEqual(a, t2.blind_codes("matter-1", "salt"))
        self.assertEqual(sorted(a.values()), ["X1", "X2", "X3"])
        self.assertEqual(set(a), set(t2.MANDATORY_ARMS))
        orders = {tuple(sorted(t2.blind_codes(f"m{i}", "salt").items())) for i in range(30)}
        self.assertGreater(len(orders), 1, "arm->code assignment must vary across matters")
        key = t2.make_unblinding_key(["matter-1"], "salt")
        self.assertEqual({arm: code for code, arm in key["matter-1"].items()}, a)

    def test_wrong_unblinding_key_changes_the_faithfulness_result(self) -> None:
        rows, base_rate, key = t2.build_open_dev_fixture("win")
        good = t2.score_trial2_rows(rows, declared_base_rate=base_rate, unblinding_key=key, data_origin="open_dev_fixture")
        bad_key = t2.make_unblinding_key([r.matter_id for r in rows], "different-salt")
        bad = t2.score_trial2_rows(rows, declared_base_rate=base_rate, unblinding_key=bad_key, data_origin="open_dev_fixture")
        self.assertNotEqual(good["faithfulness_by_arm"], bad["faithfulness_by_arm"])

    def test_missing_rubric_on_natural_conflict_is_protocol_error(self) -> None:
        rows, base_rate, key = t2.build_open_dev_fixture("win")
        first_nat = next(r for r in rows if r.natural_conflict)
        stripped = [
            t2.MatterRow(r.matter_seq, r.matter_id, r.natural_conflict, r.arms, {} if r is first_nat else r.rubric) for r in rows
        ]
        with self.assertRaises(t2.TrialProtocolError):
            t2.score_trial2_rows(stripped, declared_base_rate=base_rate, unblinding_key=key, data_origin="open_dev_fixture")

    def test_loss_weights_and_lower_bound_arithmetic(self) -> None:
        self.assertEqual(
            dict(t2.OWNER_LOSS_WEIGHTS),
            {"correct": 0.0, "abstain": 2.0, "timeout": 6.0, "wrong": 10.0},
        )
        self.assertAlmostEqual(t2.ArmResult("wrong", 10, 5).operational_loss(), 10 + 0.5 + 1.0)
        self.assertAlmostEqual(t2.ArmResult("abstain").attribution_loss(), 2.0)
        self.assertAlmostEqual(t2.lower_confidence_bound([1.0, 1.0, 1.0]), 1.0)  # zero variance => mean
        lcb = t2.lower_confidence_bound([0.0, 2.0, 4.0])  # mean 2, sd 2, n 3
        self.assertAlmostEqual(lcb, 2 - 1.645 * 2 / 3**0.5)
        with self.assertRaises(t2.TrialProtocolError):
            t2.lower_confidence_bound([1.0])

    def test_protocol_digest_changes_if_any_locked_constant_changes(self) -> None:
        proto = t2.locked_protocol()
        self.assertEqual(t2._sha256_json(proto), t2.PROTOCOL_DIGEST)
        proto["owner_loss_weights"] = dict(proto["owner_loss_weights"], wrong=9.0)
        self.assertNotEqual(t2._sha256_json(proto), t2.PROTOCOL_DIGEST)

    def test_suite_is_deterministic_and_cli_roundtrips(self) -> None:
        s1 = t2.run_offline_dev_fixture_suite()
        s2 = t2.run_offline_dev_fixture_suite()
        self.assertEqual(s1["suite_sha256"], s2["suite_sha256"])
        self.assertEqual(set(s1["scenarios"]), set(t2.DEV_FIXTURE_SCENARIOS))
        self.assertEqual(s1["status"], "protocol-ready")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "suite.json"
            self.assertEqual(t2.main(["--run-dev-fixtures", "--output", str(out)]), 0)
            loaded = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(loaded["suite_sha256"], s1["suite_sha256"])

    def test_partner_jsonl_path_scores_and_is_labeled_partner(self) -> None:
        rows, base_rate, key = t2.build_open_dev_fixture("win")
        with tempfile.TemporaryDirectory() as tmp:
            rows_path = Path(tmp) / "rows.jsonl"
            key_path = Path(tmp) / "key.json"
            out = Path(tmp) / "report.json"
            lines = []
            for r in rows:
                lines.append(
                    json.dumps(
                        {
                            "matter_seq": r.matter_seq,
                            "matter_id": r.matter_id,
                            "natural_conflict": r.natural_conflict,
                            "arms": {
                                a: {
                                    "outcome": x.outcome,
                                    "extraction_minutes": x.extraction_minutes,
                                    "review_minutes": x.review_minutes,
                                }
                                for a, x in r.arms.items()
                            },
                            "rubric": {c: dict(s) for c, s in r.rubric.items()},
                        }
                    )
                )
            rows_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            key_path.write_text(json.dumps(key), encoding="utf-8")
            code = t2.main(
                [
                    "--rows-jsonl", str(rows_path),
                    "--unblinding-key-json", str(key_path),
                    "--declared-base-rate", str(base_rate),
                    "--output", str(out),
                ]
            )
            self.assertEqual(code, 0)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["data_origin"], "partner")
            self.assertEqual(report["status"], "protocol-ready")  # scorer never self-promotes past T4 protocol-ready
            self.assertEqual(report["outcome"], "win")


if __name__ == "__main__":
    unittest.main()
