
"""Bundle 48 offline behavior and fail-closed native telemetry contracts."""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import httpx  # noqa: E402
import numpy as np  # noqa: E402

from chelated_pv7_spine.fragment_compiler import compile_fragments  # noqa: E402
from chelated_pv7_spine.hnsw_prefilter_pushdown import (  # noqa: E402
    Admission,
    OpenSearchPrefilter,
    PgvectorPrefilter,
    QdrantPrefilter,
    SearchResult,
    SimulatedHnsw,
    adaptive_ef,
    parse_opensearch_plan,
    parse_pgvector_plan,
    qdrant_graph_visits,
    qdrant_route_evidence,
    qdrant_route_snapshot,
    require_version,
)
from scripts.run_pv7_w17_6a_engine_closure import (  # noqa: E402
    DRY_DIMENSION,
    DRY_POPULATION,
    DRY_REQUESTS,
    RETRACTION_PERCENTAGES,
    SPINE_DIGEST,
    build_fixture,
    execute,
    run_cell,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = "docs/research/registry.d/48-t6a-engine-closure.json"
VERIFIER = "scripts/verifiers.d/48_verifier.py"


class TestNativeQueryContracts(unittest.TestCase):
    def test_adaptive_budget_has_floor_and_cap(self):
        self.assertEqual([adaptive_ef(frac) for frac in (1, 0.9, 0.5, 0.1, 0.001, 0)],
                         [100, 112, 200, 1000, 1000, 1000])
        self.assertEqual(adaptive_ef(0, base_ef=4, ef_cap=500), 200)
        for bad in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                adaptive_ef(bad)

    def test_pinned_versions(self):
        for engine, current in (("qdrant", "1.20.2"), ("opensearch", "3.9.0"), ("pgvector", "0.8.7")):
            with self.subTest(engine=engine):
                require_version(engine, current)
                with self.assertRaises(RuntimeError):
                    require_version(engine, "10.20.7")

    def test_qdrant_sends_epoch_and_ids_inside_native_filter(self):
        observed = []

        def transport(request):
            observed.append(json.loads(request.content))
            return httpx.Response(200, json={
                "status": "ok", "result": {"points": [{"id": 7}]},
                "usage": {"visited_graph_candidates": 37, "cpu": 1},
            })

        adapter = QdrantPrefilter("fixture", "http://qdrant.example")
        adapter.client.close()
        adapter.client = httpx.Client(base_url="http://qdrant.example", transport=httpx.MockTransport(transport))
        self.addCleanup(adapter.close)
        admission = Admission(2, 10, 100, (7, 8))
        result = adapter.search(np.array([0.5, 0.5]), admission)
        self.assertEqual(result, SearchResult((7,), 37))
        adapter.search(np.array([0.5, 0.5]), admission, exact=True)
        self.assertEqual(observed[0]["filter"]["must"], [
            {"key": "admitted_epoch", "range": {"lte": 2}}, {"has_id": [7, 8]},
        ])
        self.assertEqual(observed[0]["params"], {"hnsw_ef": 1000, "exact": False})
        self.assertTrue(observed[1]["params"]["exact"])
        self.assertIsNone(qdrant_graph_visits({"usage": {"cpu": 99, "vector_io_read": 100}}))

    def test_opensearch_uses_knn_filter_and_explicit_exact_control(self):
        observed = []

        def transport(request):
            observed.append(json.loads(request.content))
            return httpx.Response(200, json={
                "timed_out": False, "_shards": {"failed": 0},
                "hits": {"hits": [{"_source": {"point_id": 7}}]},
            })

        adapter = OpenSearchPrefilter("fixture", 2, "http://opensearch.example")
        adapter.client.close()
        adapter.client = httpx.Client(base_url="http://opensearch.example", transport=httpx.MockTransport(transport))
        self.addCleanup(adapter.close)
        admission = Admission(2, 2, 100, (7, 8))
        for exact in (True, False):
            self.assertEqual(adapter.search(np.array([0.5, 0.5]), admission, exact=exact).ids, (7,))
        ann = observed[1]["query"]["knn"]["embedding"]
        self.assertEqual(ann["filter"]["bool"]["filter"], [
            {"range": {"admitted_epoch": {"lte": 2}}}, {"terms": {"point_id": [7, 8]}},
        ])
        self.assertEqual(ann["method_parameters"]["ef_search"], 1000)
        self.assertEqual(observed[0]["query"]["script_score"]["query"], ann["filter"])


    def test_pgvector_executes_scoped_settings_and_bound_admission_in_one_query(self):
        adapter = PgvectorPrefilter.__new__(PgvectorPrefilter)
        adapter.table = "fixture"
        adapter.conn = MagicMock()
        cursor = adapter.conn.cursor.return_value.__enter__.return_value
        cursor.nextset.return_value = True
        cursor.fetchall.return_value = [("seven", 7, 2, 0.5)]
        admission = Admission(2, 2, 100, (7, 8))
        vector = np.array([0.5, 0.5])
        self.assertEqual(adapter.search(vector, admission).ids, (7,))
        sql, params = cursor.execute.call_args.args
        self.assertIn("SET LOCAL hnsw.iterative_scan = 'relaxed_order'", sql)
        self.assertIn("admitted_epoch <= %(epoch)s AND point_id = ANY(%(allowed)s)", sql)
        self.assertEqual(params["ef_search"], 1000)
        self.assertEqual(params["allowed"], [7, 8])

        adapter.search(vector, admission, exact=True)
        exact_sql, exact_params = cursor.execute.call_args.args
        self.assertIn("SET LOCAL enable_indexscan = off", exact_sql)
        self.assertNotIn("hnsw.iterative_scan", exact_sql)
        self.assertEqual(exact_params["allowed"], [7, 8])


class TestNativeTelemetry(unittest.TestCase):
    @staticmethod
    def qdrant_payload(counters):
        return {"result": {"collections": {"collections": [
            {"id": "fixture", "shards": [{"segments": [
                {"info": {"num_points": 100}, "vector_index_searches": [counters]},
                {"info": {"num_points": 0}, "vector_index_searches": [{"filtered_plain": {"count": 99}}]},
            ]}]},
            {"id": "someone_else", "shards": [{"segments": [
                {"info": {"num_points": 300}, "vector_index_searches": [{"filtered_plain": {"count": 99}}]},
            ]}]},
        ]}}}

    def test_qdrant_counts_graph_and_excludes_exact_controls(self):
        before = qdrant_route_snapshot(self.qdrant_payload({}), "fixture")
        after = qdrant_route_snapshot(self.qdrant_payload({
            "filtered_large_cardinality": {"count": 80}, "filtered_acorn": {"count": 20},
            "filtered_exact": {"count": 100},
        }), "fixture")
        evidence = qdrant_route_evidence(before, after, 100, exact_controls=100)
        self.assertEqual(evidence["route"], "filtered_hnsw")
        self.assertIs(evidence["exact_fallback_triggered"], False)
        self.assertEqual(evidence["route_counts"]["filtered_exact"], 100)
        self.assertIsNone(qdrant_route_evidence(None, after, 100)["exact_fallback_triggered"])

    def test_qdrant_small_cardinality_is_plain_scan(self):
        before = qdrant_route_snapshot(self.qdrant_payload({}), "fixture")
        after = qdrant_route_snapshot(
            self.qdrant_payload({"filtered_small_cardinality": {"count": 1}}), "fixture",
        )
        self.assertIs(qdrant_route_evidence(before, after, 1)["exact_fallback_triggered"], True)

    def test_opensearch_requires_executed_ann_and_bitset(self):
        breakdown = {"ann_search_count": 1, "exact_search_count": 0, "bitset_creation_count": 1, "cardinality": 100}
        profile = {"profile": {"shards": [{"searches": [{"query": [{"type": "KNNQuery", "breakdown": breakdown}]}]}]}}
        explain = {"explanation": {"details": [{
            "description": "the type of knn search executed at leaf was Approximate-NN, "
                           "it is not falling back to exact search after Approximate-NN search",
        }]}}
        self.assertEqual(parse_opensearch_plan(profile, explain)["route"], "filtered_hnsw")
        self.assertEqual(parse_opensearch_plan({}, explain)["route"], "unknown")
        breakdown["bitset_creation_count"] = 0
        self.assertEqual(parse_opensearch_plan(profile, explain)["route"], "unknown")
        breakdown["exact_search_count"] = 1
        self.assertIs(parse_opensearch_plan(profile, explain)["exact_fallback_triggered"], True)

    def test_opensearch_explicit_exact_explanation_is_rejected(self):
        explain = {"explanation": {
            "description": "the type of knn search executed at leaf was Exact since filteredIds = 4",
        }}
        self.assertEqual(parse_opensearch_plan({}, explain)["route"], "exact")

    def test_pgvector_distinguishes_index_bitmap_and_sequential_scans(self):
        for node, expected_route, fallback in (
            ({"Node Type": "Index Scan", "Index Name": "fixture_hnsw"}, "iterative_hnsw", 0),
            ({"Node Type": "Index Scan", "Index Name": "fixture_pkey"}, "exact_scan", 1),
            ({"Node Type": "Bitmap Heap Scan"}, "bitmap_scan", 1),
            ({"Node Type": "Seq Scan"}, "exact_scan", 1),
        ):
            with self.subTest(node=node):
                document = [{"Plan": {
                    "Node Type": "Sort", "Actual Rows": 8, "Actual Loops": 1, "Shared Hit Blocks": 42,
                    "Plans": [{**node, "Actual Loops": 1, "Relation Name": "fixture",
                               "Rows Removed by Filter": 15}],
                }}]
                evidence = parse_pgvector_plan(
                    document, hnsw_index="fixture_hnsw", iterative_scan="relaxed_order", expected_hits=10,
                )
                self.assertEqual(evidence["route"], expected_route)
                self.assertEqual(evidence["plan_node_type"], node["Node Type"])
                self.assertEqual(evidence["exact_fallback_count"], fallback)
                self.assertEqual(evidence["short_results"], 2)
                self.assertEqual(evidence["shared_hit_blocks"], 42)
                self.assertEqual(evidence["rows_removed_by_filter"], 15)


class TestOfflineBundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = execute("local-drycheck")

    def test_all_drycheck_cells_report_simulator_evidence(self):
        self.assertTrue(self.report["passed"])
        self.assertEqual(self.report["engines"][0]["engine"], "simulated_hnsw")
        cells = self.report["engines"][0]["cells"]
        self.assertEqual(tuple(cell["retraction_percent"] for cell in cells), RETRACTION_PERCENTAGES)
        self.assertEqual([cell["admitted_points"] for cell in cells], [240, 216, 120, 24])
        for cell in cells:
            self.assertEqual(cell["paired_requests"], DRY_REQUESTS)
            self.assertTrue(cell["graph_telemetry_complete"])
            self.assertGreater(cell["visited_graph_candidates"], 0)
            self.assertIsNone(cell["performance_passed"])
            self.assertEqual(cell["invalid_hits"], 0)
            self.assertGreaterEqual(cell["ann_latency"]["p99_ms"], cell["ann_latency"]["p95_ms"])
        self.assertGreaterEqual(cells[-1]["recall_at_10"], 0.99)

    def test_all_mutation_snapshots_match_fresh_ek5(self):
        audit = self.report["ek5"]
        self.assertEqual((audit["sequences"], audit["steps_per_sequence"]), (1000, 12))
        self.assertEqual(audit["snapshots_compared"], 12_000)
        self.assertGreater(audit["nodes_compared"], 50_000)
        self.assertEqual(audit["semantic_diffs"], 0)
        self.assertEqual(audit["reason_export_diffs"], 0)
        self.assertEqual(audit["pos_cyc_export_diffs"], 0)
        self.assertTrue({"CURRENT", "REFUTED", "SUPERSEDED", "UNFOUNDED_POSITIVE_CYCLE",
                         "PEER_CONFLICT_DISPUTED", "TEMPORAL_DISPUTED"} <= audit["canonical_cause_counts"].keys())

    def test_timeout_stays_in_paired_denominator(self):
        vectors, queries, epochs, admissions = build_fixture(DRY_POPULATION, DRY_DIMENSION, 2)

        class Timeouts(SimulatedHnsw):
            def search(self, query, admission, *, exact=False, limit=10):
                if not exact:
                    raise TimeoutError
                return super().search(query, admission, exact=exact, limit=limit)

            def profile(self, query, admission):
                return {"route": "simulated_filtered_graph", "exact_fallback_triggered": False}

        cell = run_cell(Timeouts(vectors, epochs), queries, epochs, 90, admissions[90], live=True)
        self.assertEqual(cell["paired_requests"], 2)
        self.assertEqual(cell["timeouts"], 2)
        self.assertEqual(cell["exact_timeouts"], 0)
        self.assertEqual(cell["recall_at_10"], 0.0)
        self.assertFalse(cell["passed"])

    def test_fragments_use_compiler_discovery_names(self):
        source = {path: (ROOT / path).read_bytes() for path in (REGISTRY, VERIFIER)}
        compiled = compile_fragments(ROOT, SPINE_DIGEST, source)
        self.assertEqual(compiled["dependency_order"], [48])
        self.assertEqual(compiled["registry"][0]["status"], "protocol-ready")
        self.assertEqual(compiled["verifiers"][0]["path"], VERIFIER)


if __name__ == "__main__":
    unittest.main()
