"""Tests for evidence_dag.py — the typed evidence-graph contract (Phase II rung 12)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from evidence_dag import (
    EVIDENCE_DAG_JSON_SCHEMA,
    EdgeType,
    EvidenceDAG,
    EvidenceEdge,
    NodeType,
    from_attribution_pool,
    validate_evidence_dag,
)


def _valid_dag() -> EvidenceDAG:
    dag = EvidenceDAG()
    dag.add_node("q1", NodeType.QUERY)
    dag.add_node("cl1", NodeType.CLUSTER)
    dag.add_node("a1", NodeType.ACTUATOR)
    dag.add_edge("q1", "cl1", EdgeType.RETRIEVED_IN)
    dag.add_edge("q1", "a1", EdgeType.CORRECTED_BY)
    dag.add_edge("a1", "cl1", EdgeType.OPERATES_ON)
    return dag


class _DeepcopyBomb:
    def __init__(self):
        self.calls = 0

    def __deepcopy__(self, _memo):
        self.calls += 1
        raise AssertionError("arbitrary __deepcopy__ hook executed")


class TestEvidenceDagContract(unittest.TestCase):
    def test_valid_dag_passes_validation_and_is_acyclic(self):
        dag = _valid_dag()
        self.assertEqual(validate_evidence_dag(dag), [])
        self.assertTrue(dag.is_acyclic())

    def test_json_round_trip_preserves_nodes_and_edges(self):
        dag = _valid_dag()
        restored = EvidenceDAG.from_json(dag.to_json())
        self.assertEqual(restored.to_dict(), dag.to_dict())
        self.assertEqual(validate_evidence_dag(restored), [])

    def test_dangling_edge_endpoint_is_flagged(self):
        dag = EvidenceDAG()
        dag.add_node("q1", NodeType.QUERY)
        dag.add_node("cl1", NodeType.CLUSTER)
        dag.add_edge("q1", "missing", EdgeType.RETRIEVED_IN)
        violations = validate_evidence_dag(dag)
        self.assertTrue(any("is not a node" in v for v in violations), violations)

    def test_wrong_endpoint_types_violate_contract(self):
        # RETRIEVED_IN must be QUERY->CLUSTER; here it is CLUSTER->QUERY.
        dag = EvidenceDAG()
        dag.add_node("q1", NodeType.QUERY)
        dag.add_node("cl1", NodeType.CLUSTER)
        dag.add_edge("cl1", "q1", EdgeType.RETRIEVED_IN)
        violations = validate_evidence_dag(dag)
        self.assertTrue(any("violate contract" in v for v in violations), violations)

    def test_cycle_is_detected(self):
        # Force a cycle by abusing OPERATES_ON / CORRECTED_BY direction across
        # mistyped nodes so the structural cycle check (not just typing) fires.
        dag = EvidenceDAG()
        dag.add_node("a1", NodeType.ACTUATOR)
        dag.add_node("a2", NodeType.ACTUATOR)
        # Two actuator nodes pointing at each other via OPERATES_ON (endpoint-type
        # invalid too, but we specifically assert the acyclicity violation here).
        dag.add_edge("a1", "a2", EdgeType.OPERATES_ON)
        dag.add_edge("a2", "a1", EdgeType.OPERATES_ON)
        self.assertFalse(dag.is_acyclic())
        self.assertIsNotNone(dag.find_cycle())
        violations = validate_evidence_dag(dag)
        self.assertTrue(any("not acyclic" in v for v in violations), violations)

    def test_conflicting_node_retype_raises(self):
        dag = EvidenceDAG()
        dag.add_node("x", NodeType.QUERY)
        with self.assertRaises(ValueError):
            dag.add_node("x", NodeType.CLUSTER)

    def test_duplicate_node_id_in_serialized_form_raises(self):
        data = {
            "record_type": "evidence_dag",
            "nodes": [
                {"id": "q1", "type": "query", "attrs": {}},
                {"id": "q1", "type": "query", "attrs": {}},
            ],
            "edges": [],
        }
        with self.assertRaises(ValueError):
            EvidenceDAG.from_dict(data)

    def test_json_schema_describes_node_and_edge_enums(self):
        node_types = set(EVIDENCE_DAG_JSON_SCHEMA["properties"]["nodes"]["items"]["properties"]["type"]["enum"])
        self.assertEqual(node_types, {t.value for t in NodeType})
        edge_types = set(EVIDENCE_DAG_JSON_SCHEMA["properties"]["edges"]["items"]["properties"]["type"]["enum"])
        self.assertEqual(edge_types, {t.value for t in EdgeType})

    def test_attrs_are_canonicalized_to_detached_plain_json_values(self):
        nested = {"layers": [{"weight": 1.0}], "enabled": True}
        dag = EvidenceDAG()
        node = dag.add_node("q1", NodeType.QUERY, payload=nested)
        dag.add_node("cl1", NodeType.CLUSTER)
        edge = dag.add_edge("q1", "cl1", EdgeType.RETRIEVED_IN, payload=nested)

        nested["layers"][0]["weight"] = 9.0
        self.assertEqual(node.attrs["payload"]["layers"][0]["weight"], 1.0)
        self.assertEqual(edge.attrs["payload"]["layers"][0]["weight"], 1.0)

        exported = dag.to_dict()
        exported["nodes"][0]["attrs"]["payload"]["layers"][0]["weight"] = 8.0
        exported["edges"][0]["attrs"]["payload"]["layers"][0]["weight"] = 7.0
        self.assertEqual(dag.node("q1").attrs["payload"]["layers"][0]["weight"], 1.0)
        self.assertEqual(dag.edges[0].attrs["payload"]["layers"][0]["weight"], 1.0)

    def test_arbitrary_attr_objects_are_rejected_without_deepcopy_hooks(self):
        for constructor in ("node", "edge", "direct_edge"):
            with self.subTest(constructor=constructor):
                bomb = _DeepcopyBomb()
                dag = EvidenceDAG()
                dag.add_node("q1", NodeType.QUERY)
                dag.add_node("cl1", NodeType.CLUSTER)
                with self.assertRaisesRegex(TypeError, "plain JSON-compatible"):
                    if constructor == "node":
                        dag.add_node("bad", NodeType.QUERY, payload=bomb)
                    elif constructor == "edge":
                        dag.add_edge("q1", "cl1", EdgeType.RETRIEVED_IN, payload=bomb)
                    else:
                        EvidenceEdge(
                            "q1",
                            "cl1",
                            EdgeType.RETRIEVED_IN,
                            {"payload": bomb},
                        )
                self.assertEqual(bomb.calls, 0)

    def test_non_json_attr_shapes_fail_closed(self):
        cyclic = []
        cyclic.append(cyclic)
        cases = (
            ("tuple", (1, 2), TypeError),
            ("non_string_key", {1: "value"}, TypeError),
            ("non_finite", float("nan"), ValueError),
            ("cycle", cyclic, ValueError),
        )
        for label, value, error_type in cases:
            with self.subTest(label=label):
                dag = EvidenceDAG()
                with self.assertRaises(error_type):
                    dag.add_node("q1", NodeType.QUERY, payload=value)

    def test_transaction_preflight_rejects_injected_hostile_attrs_without_hooks(self):
        dag = _valid_dag()
        bomb = _DeepcopyBomb()
        dag.edges[0].attrs["payload"] = bomb
        scorer = MagicMock(return_value=1.0)

        with self.assertRaisesRegex(TypeError, "plain JSON-compatible"):
            dag.prune_edges(scorer, threshold=0.5)

        scorer.assert_not_called()
        self.assertEqual(bomb.calls, 0)

    def test_transaction_rejects_callback_injected_hostile_attrs_without_hooks(self):
        dag = _valid_dag()
        before = dag.to_dict()
        bomb = _DeepcopyBomb()
        mutated = False

        def hostile_scorer(edge):
            nonlocal mutated
            if not mutated:
                mutated = True
                edge.attrs["payload"] = bomb
            return 1.0

        with self.assertRaisesRegex(RuntimeError, "non-canonical"):
            dag.prune_edges(hostile_scorer, threshold=0.5)

        self.assertEqual(dag.to_dict(), before)
        self.assertEqual(bomb.calls, 0)


class TestFromAttributionPool(unittest.TestCase):
    def _pool(self):
        # Uses the REAL per-row action vocabulary emitted by the pipeline
        # (FAST = no correction; CHELATE / REFORMULATE / CHELATE_ALWAYS = corrections).
        # "REFORM" is deliberately NOT used here — it is an action_mix count key, not a
        # per-row action, so a builder matching it would silently drop real corrections.
        return {
            "query_attribution_rows": [
                {
                    "strategy": "probeA",
                    "query_id": "q-1",
                    "task": "SciFact",
                    "profile": "default",
                    "action": "FAST",
                    "fault_class": None,
                },
                {
                    "strategy": "probeA",
                    "query_id": "q-2",
                    "task": "SciFact",
                    "profile": "default",
                    "action": "CHELATE",
                    "fault_class": "drift",
                },
                {
                    "strategy": "probeB",
                    "query_id": "q-3",
                    "task": "NFCorpus",
                    "profile": "hotter",
                    "action": "REFORMULATE",
                    "fault_class": "collapse",
                },
                {
                    "strategy": "probeB",
                    "query_id": "q-4",
                    "task": "NFCorpus",
                    "profile": "hotter",
                    "action": "CHELATE_ALWAYS",
                    "fault_class": "drift",
                },
            ]
        }

    def test_builder_produces_valid_acyclic_typed_dag(self):
        dag = from_attribution_pool(self._pool())
        self.assertEqual(validate_evidence_dag(dag), [])
        self.assertTrue(dag.is_acyclic())

    def test_fast_query_has_no_actuator_edge_corrected_query_does(self):
        dag = from_attribution_pool(self._pool())
        types = {n.node_type for n in dag.nodes}
        self.assertIn(NodeType.ACTUATOR, types)
        # The three correcting rows (CHELATE, REFORMULATE, CHELATE_ALWAYS) each created an
        # actuator; the single FAST row created none. This would fail (count 1, not 3) if
        # the builder matched the fictional "REFORM" token instead of the real vocabulary.
        actuators = [n for n in dag.nodes if n.node_type == NodeType.ACTUATOR]
        self.assertEqual(len(actuators), 3)
        corrected_by = [e for e in dag.edges if e.edge_type == EdgeType.CORRECTED_BY]
        self.assertEqual(len(corrected_by), 3)
        operates_on = [e for e in dag.edges if e.edge_type == EdgeType.OPERATES_ON]
        self.assertEqual(len(operates_on), 3)

    def test_builder_is_deterministic(self):
        a = from_attribution_pool(self._pool()).to_dict()
        b = from_attribution_pool(self._pool()).to_dict()
        self.assertEqual(a, b)

    def test_empty_pool_yields_empty_valid_dag(self):
        dag = from_attribution_pool({})
        self.assertEqual(dag.nodes, [])
        self.assertEqual(validate_evidence_dag(dag), [])


if __name__ == "__main__":
    unittest.main()
