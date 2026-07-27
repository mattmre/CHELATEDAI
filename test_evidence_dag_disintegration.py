"""Offline unittest coverage for the rung-13 Evidence-DAG disintegration loop."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from convergence_monitor import ConvergenceMonitor
from evidence_dag import (
    EdgeType,
    EvidenceDAG,
    EvidenceEdge,
    NodeType,
    validate_evidence_dag,
)
from evidence_dag_disintegration import (
    detector_signals_from_outputs,
    reanneal_edges,
    score_edge_fitness,
    write_disintegration_artifact,
)
from isomer_detector import IsomerDetector


def _dag() -> EvidenceDAG:
    dag = EvidenceDAG()
    dag.add_node("q1", NodeType.QUERY, query_id="qid-1", query_text="collapsed query")
    dag.add_node("q2", NodeType.QUERY, query_id="qid-2", query_text="healthy query")
    dag.add_node("cl1", NodeType.CLUSTER)
    dag.add_node("cl2", NodeType.CLUSTER)
    dag.add_node("a1", NodeType.ACTUATOR)
    dag.add_edge("q1", "cl1", EdgeType.RETRIEVED_IN)
    dag.add_edge("q2", "cl2", EdgeType.RETRIEVED_IN)
    dag.add_edge("q2", "a1", EdgeType.CORRECTED_BY)
    dag.add_edge("a1", "cl2", EdgeType.OPERATES_ON)
    return dag


def _isomer_output(collapsed: bool = True):
    pre = {
        "collapsed query": ["a", "b"],
        "healthy query": ["x", "y"],
    }
    post = {
        "collapsed query": ["c", "d"] if collapsed else ["a", "b"],
        "healthy query": ["x", "y"],
    }
    with patch("isomer_detector.get_logger", return_value=MagicMock()):
        return IsomerDetector(strength_threshold=0.5).detect_sedimentation_isomers(pre, post)


class TestEvidenceDagDisintegration(unittest.TestCase):
    def test_flagged_edge_is_pruned_healthy_edges_remain_and_dag_is_valid(self):
        dag = _dag()
        signals = detector_signals_from_outputs(dag, _isomer_output(collapsed=True))
        record = dag.prune_edges(lambda edge: score_edge_fitness(edge, signals), threshold=0.5)

        self.assertEqual(record["edges_before"], 4)
        self.assertEqual(record["edges_after"], 3)
        self.assertEqual(len(record["pruned"]), 1)
        self.assertEqual(record["pruned"][0]["src"], "q1")
        self.assertEqual({edge.src for edge in dag.edges}, {"q2", "a1"})
        self.assertEqual(validate_evidence_dag(dag), [])

    def test_all_healthy_dag_is_noop(self):
        dag = _dag()
        before = dag.to_dict()
        signals = detector_signals_from_outputs(dag, _isomer_output(collapsed=False))
        record = dag.prune_edges(lambda edge: score_edge_fitness(edge, signals), threshold=0.5)
        self.assertEqual(record["pruned"], [])
        self.assertEqual(record["edges_before"], record["edges_after"])
        self.assertEqual(dag.to_dict(), before)
        self.assertEqual(dag.pruned_edge_ledger, [])

    def test_dry_run_reports_same_decision_without_mutation(self):
        dag = _dag()
        before = dag.to_dict()
        signals = detector_signals_from_outputs(dag, _isomer_output(collapsed=True))
        dry = dag.prune_edges(lambda edge: score_edge_fitness(edge, signals), threshold=0.5, dry_run=True)
        self.assertEqual(dag.to_dict(), before)
        self.assertEqual(dag.pruned_edge_ledger, [])

        actual = dag.prune_edges(lambda edge: score_edge_fitness(edge, signals), threshold=0.5)
        self.assertEqual(dry["scores"], actual["scores"])
        self.assertEqual(dry["pruned"], actual["pruned"])
        self.assertEqual(dry["edges_after"], actual["edges_after"])

    def test_reanneal_restores_edge_after_real_convergence_signal_recovers(self):
        dag = _dag()
        monitor = ConvergenceMonitor(patience=3, min_epochs=1)
        monitor.record_loss(1.0)
        monitor.record_loss(1.0)
        monitor.record_loss(1.0)
        unstable = detector_signals_from_outputs(dag, _isomer_output(collapsed=False), {"cl1": monitor.get_summary()})
        prune = dag.prune_edges(lambda edge: score_edge_fitness(edge, unstable), threshold=0.5)
        self.assertEqual(len(prune["pruned"]), 1)

        self.assertTrue(monitor.record_loss(1.0))
        recovered = detector_signals_from_outputs(dag, _isomer_output(collapsed=False), {"cl1": monitor.get_summary()})
        # The "recovery" here is a stable early-stop plateau (converged=True with a
        # flat loss), which the disintegration doc treats as stable/healthy. Assert
        # that semantics explicitly so the test name is not read as loss-improvement.
        self.assertEqual(recovered["provenance"]["convergence_monitor"]["states"]["cl1"], "converged")
        reanneal = reanneal_edges(dag, score_edge_fitness, 0.5, recovered)
        self.assertEqual(len(reanneal["reannealed"]), 1)
        self.assertEqual(len(dag.edges), 4)
        self.assertEqual(dag.pruned_edge_ledger, [])
        self.assertEqual(validate_evidence_dag(dag), [])

    def test_artifact_records_fitness_lifecycle_and_detector_provenance(self):
        dag = _dag()
        low = detector_signals_from_outputs(
            dag, _isomer_output(collapsed=True), provenance={"run_id": "offline-fixture"}
        )
        prune = dag.prune_edges(lambda edge: score_edge_fitness(edge, low), 0.5)
        recovered = detector_signals_from_outputs(dag, _isomer_output(collapsed=False))
        reanneal = reanneal_edges(dag, score_edge_fitness, 0.5, recovered)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rung13.json"
            artifact = write_disintegration_artifact(str(path), prune, reanneal, detector_provenance=low["provenance"])
            persisted = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(persisted, artifact)
        self.assertTrue(artifact["pruned"])
        self.assertTrue(artifact["reannealed"])
        self.assertIn("fitness_before", artifact["edge_fitness"][0])
        self.assertIn("fitness_after", artifact["edge_fitness"][0])
        self.assertIn("isomer_detector", artifact["detector_signal_provenance"])
        self.assertIn("convergence_monitor", artifact["detector_signal_provenance"])

    def test_structural_edge_is_protected_even_when_scorer_is_low(self):
        dag = EvidenceDAG()
        dag.add_node("a1", NodeType.ACTUATOR)
        dag.add_node("cl1", NodeType.CLUSTER)
        dag.add_edge("a1", "cl1", EdgeType.OPERATES_ON)
        before = dag.to_dict()
        record = dag.prune_edges(lambda _edge: 0.0, threshold=0.5)
        self.assertEqual(dag.to_dict(), before)
        self.assertEqual(record["pruned"], [])
        self.assertEqual(len(record["protected_skips"]), 1)

    def test_reanneal_scorer_failure_is_transactional(self):
        dag = EvidenceDAG()
        for query_id in ("q1", "q2"):
            dag.add_node(query_id, NodeType.QUERY)
        dag.add_node("cl1", NodeType.CLUSTER)
        dag.add_edge("q1", "cl1", EdgeType.RETRIEVED_IN)
        dag.add_edge("q2", "cl1", EdgeType.RETRIEVED_IN)
        dag.prune_edges(lambda _edge: 0.0, 0.5)
        before_edges = dag.to_dict()
        before_ledger = dag.pruned_edge_ledger
        calls = 0

        def raising_scorer(_edge, _signals):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("detector failure")
            return 1.0

        with self.assertRaisesRegex(RuntimeError, "detector failure"):
            reanneal_edges(dag, raising_scorer, 0.5, {})
        self.assertEqual(dag.to_dict(), before_edges)
        self.assertEqual(dag.pruned_edge_ledger, before_ledger)

    def test_reanneal_rejects_an_invalid_dag_before_scoring(self):
        dag = _dag()
        dag.prune_edges(
            lambda edge: 0.0 if edge.src == "q1" else 1.0,
            threshold=0.5,
        )
        dag.add_edge("q2", "missing-cluster", EdgeType.RETRIEVED_IN)
        before_edges = dag.to_dict()
        before_ledger = dag.pruned_edge_ledger
        scorer = MagicMock(return_value=1.0)

        with self.assertRaisesRegex(ValueError, "cannot re-anneal on an invalid EvidenceDAG"):
            reanneal_edges(dag, scorer, 0.5, {})

        scorer.assert_not_called()
        self.assertEqual(dag.to_dict(), before_edges)
        self.assertEqual(dag.pruned_edge_ledger, before_ledger)

    def test_reanneal_keeps_an_individually_invalid_ledger_edge_pruned(self):
        dag = EvidenceDAG()
        dag.add_node("q1", NodeType.QUERY)
        dag.add_node("cl1", NodeType.CLUSTER)
        dag.add_edge("q1", "cl1", EdgeType.RETRIEVED_IN)
        dag.prune_edges(lambda _edge: 0.0, threshold=0.5)
        dag._nodes.pop("q1")
        self.assertEqual(validate_evidence_dag(dag), [])

        record = reanneal_edges(dag, lambda _edge, _signals: 1.0, 0.5, {})

        self.assertEqual(record["reannealed"], [])
        self.assertEqual(len(record["skipped_invalid"]), 1)
        self.assertEqual(len(dag.pruned_edge_ledger), 1)

    def test_duplicate_edges_are_all_reannealed(self):
        dag = EvidenceDAG()
        dag.add_node("q1", NodeType.QUERY)
        dag.add_node("cl1", NodeType.CLUSTER)
        dag.add_edge("q1", "cl1", EdgeType.RETRIEVED_IN)
        dag.add_edge("q1", "cl1", EdgeType.RETRIEVED_IN)
        prune = dag.prune_edges(lambda _edge: 0.0, 0.5)
        self.assertEqual(len(prune["pruned"]), 2)
        reanneal = reanneal_edges(dag, lambda _edge, _signals: 1.0, 0.5, {})
        self.assertEqual(len(reanneal["reannealed"]), 2)
        self.assertEqual(len(dag.edges), 2)
        self.assertEqual(dag.pruned_edge_ledger, [])

    def test_artifact_rejects_missing_detector_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "detector_provenance"):
                write_disintegration_artifact(
                    str(Path(tmp) / "invalid.json"),
                    {"scores": [], "pruned": [], "threshold": 0.5},
                    detector_provenance={},
                )


class TestFailClosedRegressionSurface(unittest.TestCase):
    """Load-bearing claim: absent / unmatched / immature evidence can NEVER prune."""

    def _prune_count(self, dag, signals):
        record = dag.prune_edges(lambda edge: score_edge_fitness(edge, signals), threshold=0.5)
        return record["pruned"]

    def test_empty_isomer_output_prunes_nothing(self):
        dag = _dag()
        signals = detector_signals_from_outputs(dag, {})
        self.assertEqual(self._prune_count(dag, signals), [])
        self.assertEqual(len(dag.edges), 4)

    def test_unmatched_query_key_prunes_nothing(self):
        dag = _dag()
        # A real-shaped isomer output whose query does not join any DAG node.
        ghost = {
            "isomers": [{"query": "ghost-query", "strength": 1.0}],
            "non_isomers": [],
            "mode": "sedimentation",
        }
        signals = detector_signals_from_outputs(dag, ghost)
        self.assertEqual(
            signals["provenance"]["isomer_detector"]["unmatched_query_keys"],
            ["ghost-query"],
        )
        self.assertEqual(signals["isomer_query_fitness"], {})
        self.assertEqual(self._prune_count(dag, signals), [])

    def test_immature_convergence_prunes_nothing(self):
        dag = _dag()
        monitor = ConvergenceMonitor(patience=3, min_epochs=5)
        monitor.record_loss(1.0)
        monitor.record_loss(1.0)  # total_epochs (2) < min_epochs (5)
        signals = detector_signals_from_outputs(dag, _isomer_output(collapsed=False), {"cl1": monitor.get_summary()})
        self.assertEqual(
            signals["provenance"]["convergence_monitor"]["states"]["cl1"],
            "insufficient_evidence",
        )
        self.assertEqual(self._prune_count(dag, signals), [])

    def test_wrong_cluster_id_convergence_prunes_nothing(self):
        dag = _dag()
        monitor = ConvergenceMonitor(patience=3, min_epochs=1)
        for _ in range(4):
            monitor.record_loss(1.0)  # mature, non-converged, low fitness IF matched
        signals = detector_signals_from_outputs(
            dag, _isomer_output(collapsed=False), {"cluster-does-not-exist": monitor.get_summary()}
        )
        self.assertIn(
            "cluster-does-not-exist",
            signals["provenance"]["convergence_monitor"]["unmatched_cluster_ids"],
        )
        self.assertEqual(signals["convergence_cluster_fitness"], {})
        self.assertEqual(self._prune_count(dag, signals), [])

    def test_non_sedimentation_mode_is_rejected(self):
        dag = _dag()
        chelation = {
            "isomers": [{"query": "collapsed query", "strength": 1.0}],
            "non_isomers": [],
            "mode": "chelation",
        }
        with self.assertRaisesRegex(ValueError, "sedimentation"):
            detector_signals_from_outputs(dag, chelation)

    def test_prune_rejects_a_mutating_scorer(self):
        dag = _dag()
        before = dag.edges

        def mutating_scorer(edge):
            # A buggy caller that mutates the DAG mid-prune must be rejected, not
            # allowed to silently drop unscored/unledgered edges.
            if edge.src == "q2" and edge.dst == "cl2":
                dag._edges.pop()
            return 1.0

        with self.assertRaisesRegex(RuntimeError, "mutated the DAG edge set"):
            dag.prune_edges(mutating_scorer, threshold=0.5)
        self.assertEqual(dag.edges, before)

    def test_prune_rejects_same_length_edge_replacement(self):
        dag = _dag()
        before = dag.edges
        mutated = False

        def replacing_scorer(_edge):
            nonlocal mutated
            if not mutated:
                mutated = True
                dag._edges[-1] = EvidenceEdge("q1", "cl1", EdgeType.RETRIEVED_IN, {"replacement": True})
            return 1.0

        with self.assertRaisesRegex(RuntimeError, "mutated the DAG edge set"):
            dag.prune_edges(replacing_scorer, threshold=0.5)
        self.assertEqual(dag.edges, before)

    def test_prune_scorer_exception_restores_edge_order(self):
        dag = _dag()
        before = dag.edges

        def raising_scorer(_edge):
            dag._edges.reverse()
            raise RuntimeError("scorer failure")

        with self.assertRaisesRegex(RuntimeError, "scorer failure"):
            dag.prune_edges(raising_scorer, threshold=0.5)
        self.assertEqual(dag.edges, before)

    def test_dry_run_rejects_same_length_edge_reordering(self):
        dag = _dag()
        before = dag.edges
        mutated = False

        def reordering_scorer(_edge):
            nonlocal mutated
            if not mutated:
                mutated = True
                dag._edges.reverse()
            return 1.0

        with self.assertRaisesRegex(RuntimeError, "mutated the DAG edge set"):
            dag.prune_edges(reordering_scorer, threshold=0.5, dry_run=True)
        self.assertEqual(dag.edges, before)


if __name__ == "__main__":
    unittest.main()
