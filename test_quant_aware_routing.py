import inspect
import math
import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from adapter_router import AdapterRouter
from antigravity_engine import AntigravityEngine
from quant_aware_routing import (
    QuantAwareRoutingPlane,
    RoutingPlaneConfig,
    ThreeWaySplit,
    fit_quant_aware_plane,
    paired_query_bootstrap_ci,
    three_way_seeded_split,
)
from run_quant_aware_routing_campaign import _source_provenance, run_arena


class FixedLinear(torch.nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.register_buffer("matrix", torch.tensor(matrix, dtype=torch.float32))

    def forward(self, inputs):
        return inputs @ self.matrix.T


class TestSourceProvenance(unittest.TestCase):
    def test_source_provenance_fails_closed_when_git_or_head_is_unavailable(self):
        preregistration = Path(__file__).resolve().with_name("prereg_rung16.json")
        failures = (
            FileNotFoundError("git executable is unavailable"),
            subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"]),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                with patch(
                    "run_quant_aware_routing_campaign.subprocess.run",
                    side_effect=failure,
                ):
                    with self.assertRaises(type(failure)):
                        _source_provenance(preregistration)

    def test_source_provenance_fails_closed_when_status_capture_fails(self):
        preregistration = Path(__file__).resolve().with_name("prereg_rung16.json")
        status_failure = subprocess.CalledProcessError(
            128,
            ["git", "status", "--porcelain"],
        )
        with patch(
            "run_quant_aware_routing_campaign.subprocess.run",
            side_effect=(
                subprocess.CompletedProcess(
                    ["git", "rev-parse", "HEAD"],
                    0,
                    stdout="deadbeef\n",
                ),
                status_failure,
            ),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                _source_provenance(preregistration)


def _permutation(first, second, dim=4):
    matrix = np.eye(dim, dtype=np.float32)
    matrix[[first, second]] = matrix[[second, first]]
    return matrix


def _passing_plane(*, preregistration=None, include_unseen_bad_route=False):
    document_ids = ("doc-a", "decoy-a", "doc-b", "decoy-b")
    documents = np.asarray(
        [
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    identity = FixedLinear(np.eye(4, dtype=np.float32))
    route_a = FixedLinear(_permutation(0, 2))
    route_b = FixedLinear(_permutation(1, 3))
    global_adapter = FixedLinear(_permutation(0, 2))
    router = AdapterRouter(margin_delta=0.02, logger=MagicMock())
    router.register_global([-1.0, -1.0, -1.0, -1.0], global_adapter)
    router.register("route-a", [1.0, 0.0, 0.0, 1.0], route_a)
    router.register("route-b", [0.0, 1.0, 1.0, 0.0], route_b)
    adapters = {"global": global_adapter, "route-a": route_a, "route-b": route_b}
    if include_unseen_bad_route:
        router.register("route-c", [-1.0, -1.0, -1.0, -1.0], identity)
        adapters["route-c"] = identity

    anchor_ids = tuple(f"anchor-{index}" for index in range(4))
    select_ids = tuple(f"select-{index}" for index in range(20))
    report_ids = tuple(f"report-{index}" for index in range(20))
    split = ThreeWaySplit(anchor_ids, select_ids, report_ids, seed=1616)
    qrels = {}
    vectors = {}
    for phase_ids in (select_ids, report_ids):
        for index, query_id in enumerate(phase_ids):
            if index < 10:
                if index < 8:
                    vectors[query_id] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    qrels[query_id] = {"doc-a": 1.0}
                else:
                    vectors[query_id] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                    qrels[query_id] = {"doc-b": 1.0}
            else:
                if index < 18:
                    vectors[query_id] = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
                    qrels[query_id] = {"doc-b": 1.0}
                else:
                    vectors[query_id] = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
                    qrels[query_id] = {"doc-a": 1.0}
    config = RoutingPlaneConfig(
        k=1,
        bootstrap_resamples=500,
        min_lift=0.005,
        max_floor_loss=0.005,
        quant_minimum_fp32_gain=0.01,
        min_report_routes=2,
        min_report_route_fraction=0.10,
    )
    plane = QuantAwareRoutingPlane(
        router=router,
        adapters=adapters,
        document_ids=document_ids,
        document_embeddings=documents,
        qrels=qrels,
        split=split,
        config=config,
        preregistration=preregistration
        or {"record_type": "test_prereg", "status": "FROZEN_PRE_REPORT"},
        anchor_query_ids_used=anchor_ids,
        anchor_document_ids_used=("doc-a", "doc-b"),
        oracle_document_embeddings=documents,
    )
    select_vectors = {query_id: vectors[query_id] for query_id in select_ids}
    report_vectors = {query_id: vectors[query_id] for query_id in report_ids}
    return plane, select_vectors, report_vectors


class TestThreeWaySplit(unittest.TestCase):
    def test_three_way_split_is_deterministic_disjoint_and_40_30_30(self):
        ids = [f"q-{index}" for index in range(10)]
        first = three_way_seeded_split(reversed(ids), seed=1616)
        second = three_way_seeded_split(ids, seed=1616)
        self.assertEqual(first, second)
        self.assertEqual((len(first.anchor_ids), len(first.select_ids), len(first.report_ids)), (4, 3, 3))
        self.assertFalse(set(first.anchor_ids) & set(first.select_ids))
        self.assertFalse(set(first.anchor_ids) & set(first.report_ids))
        self.assertFalse(set(first.select_ids) & set(first.report_ids))
        self.assertEqual(set(first.anchor_ids) | set(first.select_ids) | set(first.report_ids), set(ids))

    def test_stratified_split_preserves_each_domain(self):
        ids = [f"a-{index}" for index in range(10)] + [f"b-{index}" for index in range(10)]
        strata = {query_id: query_id[0] for query_id in ids}
        split = three_way_seeded_split(ids, seed=1616, strata=strata)
        for phase in (split.anchor_ids, split.select_ids, split.report_ids):
            self.assertEqual({strata[query_id] for query_id in phase}, {"a", "b"})

    def test_campaign_locks_select_before_embedding_report_queries(self):
        source = inspect.getsource(run_arena)
        lock_write = source.index("_exclusive_write_json(lock_path, lock_payload)")
        report_embed = source.index("report_vectors = _embed_phase_queries")
        self.assertLess(lock_write, report_embed)
        self.assertNotIn("split.report_ids", source[:lock_write])


class TestPairedBootstrap(unittest.TestCase):
    def test_constant_positive_passes_and_zero_does_not(self):
        baseline = {f"q-{index}": 0.2 for index in range(20)}
        positive = {query_id: 0.3 for query_id in baseline}
        zero = dict(baseline)
        strong = paired_query_bootstrap_ci(
            positive,
            baseline,
            confidence_level=0.95,
            resamples=200,
            seed=1617,
        )
        tied = paired_query_bootstrap_ci(
            zero,
            baseline,
            confidence_level=0.95,
            resamples=200,
            seed=1617,
        )
        self.assertGreater(strong.lower, 0.005)
        self.assertEqual(tied.lower, 0.0)
        self.assertFalse(tied.lower > 0.005)

    def test_pairing_rejects_mismatched_or_nonfinite_queries(self):
        with self.assertRaises(ValueError):
            paired_query_bootstrap_ci(
                {"q1": 1.0},
                {"q2": 0.0},
                confidence_level=0.95,
                resamples=10,
                seed=1,
            )
        with self.assertRaises(ValueError):
            paired_query_bootstrap_ci(
                {"q1": math.nan},
                {"q1": 0.0},
                confidence_level=0.95,
                resamples=10,
                seed=1,
            )


class TestQuantAwarePromotionPlane(unittest.TestCase):
    def test_plane_level_ci_quant_baselines_and_multi_route_binding_promote(self):
        plane, select_vectors, report_vectors = _passing_plane()
        select = plane.select(select_vectors)
        self.assertTrue(select["select_gate_passed"])
        self.assertTrue(select["ci_passed"])
        self.assertEqual(set(select["quant_decisions"]), {"global", "route-a", "route-b"})
        for route_key, decision in select["quant_decisions"].items():
            expected_source = (
                "no_route_full_select_for_unseen_retained_adapter"
                if route_key == "global"
                else "no_route_same_routed_select_subset"
            )
            self.assertEqual(decision["baseline_source"], expected_source)
            self.assertAlmostEqual(decision["baseline_fitness"], 0.2)
            self.assertGreater(decision["fp32_gain"], 0.01)
            self.assertTrue(decision["passed"])
        self.assertEqual(select["usage"]["route_p_k"], {"route-a": 0.5, "route-b": 0.5})

        report = plane.report(report_vectors)
        self.assertEqual(report["verdict"], "PROMOTED")
        self.assertTrue(report["multi_route_binding"]["passed"])
        self.assertEqual(plane.verdict, "PROMOTED")
        self.assertTrue(plane.promoted)
        self.assertTrue(plane.provenance()["leakage_safe"])

    def test_select_quant_gates_unseen_retained_route(self):
        plane, select_vectors, _report_vectors = _passing_plane(include_unseen_bad_route=True)
        select = plane.select(select_vectors)
        unseen = select["quant_decisions"]["route-c"]
        self.assertFalse(unseen["selected_on_select"])
        self.assertEqual(unseen["query_count"], len(select_vectors))
        self.assertEqual(
            unseen["baseline_source"],
            "no_route_full_select_for_unseen_retained_adapter",
        )
        self.assertFalse(unseen["passed"])
        self.assertFalse(select["select_gate_passed"])

    def test_select_freezes_router_and_detects_direct_state_mutation(self):
        plane, select_vectors, report_vectors = _passing_plane()
        plane.select(select_vectors)
        with self.assertRaises(RuntimeError):
            plane.router.margin_delta = 0.5
        plane.router._margin_delta = 0.5
        with self.assertRaises(RuntimeError):
            plane.report(report_vectors)

    def test_provenance_cannot_mutate_failed_select_into_promotion(self):
        plane, select_vectors, report_vectors = _passing_plane(include_unseen_bad_route=True)
        select = plane.select(select_vectors)
        self.assertFalse(select["select_gate_passed"])
        exposed = plane.provenance()
        exposed["select"]["select_gate_passed"] = True
        exposed["select"]["quant_passed"] = True
        report = plane.report(report_vectors)
        self.assertEqual(report["verdict"], "FAIL-CLOSED")
        self.assertFalse(plane.select_gate_passed)
        self.assertFalse(plane.provenance()["select"]["quant_passed"])

    def test_direct_select_decision_mutation_fails_closed(self):
        plane, select_vectors, report_vectors = _passing_plane()
        plane.select(select_vectors)
        plane._select_decision["select_gate_passed"] = False
        with self.assertRaises(RuntimeError):
            plane.report(report_vectors)

    def test_engine_rejects_corrupted_promoted_plane_before_enable(self):
        plane, select_vectors, report_vectors = _passing_plane()
        plane.select(select_vectors)
        plane.report(report_vectors)
        plane.qrels["report-0"]["doc-a"] = 99.0
        engine = object.__new__(AntigravityEngine)
        engine.logger = MagicMock()

        def provider(_text):
            return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        with self.assertRaises(RuntimeError):
            engine.enable_quant_aware_routing(plane, query_vector_provider=provider)

    def test_mutated_cached_document_view_fails_closed(self):
        plane, select_vectors, report_vectors = _passing_plane()
        plane.select(select_vectors)
        plane.report(report_vectors)
        plane._view_cache[("route-a", True)] = plane._base_view()
        with self.assertRaises(RuntimeError):
            plane.retrieve([1.0, 0.0, 0.0, 0.0], quantized=True)

    def test_report_degeneracy_downgrades_even_after_select_pass(self):
        plane, select_vectors, report_vectors = _passing_plane()
        plane.select(select_vectors)
        all_route_a = {
            query_id: np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            for query_id in report_vectors
        }
        report = plane.report(all_route_a)
        self.assertEqual(report["verdict"], "DEGENERATE")
        self.assertFalse(plane.promoted)
        self.assertEqual(report["usage"]["n_routes_used"], 1)

    def test_report_is_one_shot_and_cannot_select_or_fit(self):
        plane, select_vectors, report_vectors = _passing_plane()
        with self.assertRaises(ValueError):
            plane.select(report_vectors)
        plane.select(select_vectors)
        plane.report(report_vectors)
        with self.assertRaises(RuntimeError):
            plane.report(report_vectors)
        with self.assertRaises(RuntimeError):
            plane.select(select_vectors)
        provenance = plane.provenance()
        self.assertEqual(provenance["fit"]["anchor_query_count"], 4)
        self.assertTrue(provenance["report_consumed"])

    def test_fit_rejects_report_query_before_adapter_construction(self):
        split = ThreeWaySplit(("anchor",), ("select",), ("report",), seed=1616)
        config = RoutingPlaneConfig(
            min_cluster_documents=1,
            adapter_steps=1,
            adapter_batch_size=1,
        )
        bad_pair = {
            "query_id": "report",
            "doc_id": "doc",
            "doc_vector": np.asarray([1.0, 0.0], dtype=np.float32),
            "query_vector": np.asarray([1.0, 0.0], dtype=np.float32),
            "route_key": "route",
        }
        with self.assertRaises(ValueError):
            fit_quant_aware_plane(
                document_ids=("doc",),
                document_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
                qrels={"report": {"doc": 1.0}},
                split=split,
                anchor_pairs=[bad_pair],
                config=config,
                preregistration={"status": "FROZEN_PRE_REPORT"},
                device="cpu",
            )

    def test_engine_accepts_only_final_promoted_plane(self):
        plane, select_vectors, report_vectors = _passing_plane()
        engine = object.__new__(AntigravityEngine)
        engine.logger = MagicMock()
        with self.assertRaises(ValueError):
            engine.enable_quant_aware_routing(plane)
        plane.select(select_vectors)
        plane.report(report_vectors)
        provenance = engine.enable_quant_aware_routing(plane)
        self.assertEqual(provenance["verdict"], "PROMOTED")
        self.assertIs(engine._quant_aware_routing_plane, plane)
        self.assertEqual(engine.get_quant_aware_routing_provenance()["verdict"], "PROMOTED")

    def test_engine_requires_explicit_query_geometry_for_encoder_swap(self):
        preregistration = {
            "status": "FROZEN_PRE_REPORT",
            "models": {
                "document_encoder": "mini-lm",
                "swapped_query_encoder": "mpnet",
            },
        }
        plane, select_vectors, report_vectors = _passing_plane(preregistration=preregistration)
        plane.select(select_vectors)
        plane.report(report_vectors)
        engine = object.__new__(AntigravityEngine)
        engine.logger = MagicMock()
        with self.assertRaises(ValueError):
            engine.enable_quant_aware_routing(plane)

        def provider(_text):
            return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        provenance = engine.enable_quant_aware_routing(plane, query_vector_provider=provider)
        self.assertEqual(provenance["serving"]["query_vector_source"], "explicit_query_vector_provider")

    def test_engine_runtime_uses_provider_vector_and_route_document_view(self):
        plane, select_vectors, report_vectors = _passing_plane()
        plane.select(select_vectors)
        plane.report(report_vectors)
        engine = object.__new__(AntigravityEngine)
        engine.logger = MagicMock()
        engine.vector_size = 4
        engine.mode = "local"
        engine.model_name = "fixture"
        engine._runtime_telemetry = {}
        engine._model_scope_runtime = None
        engine._last_runtime_diagnostics = None

        def forbidden_embed(_text):
            raise AssertionError("the promoted plane must not use the already-adapted engine embed path")

        engine.embed = forbidden_embed
        def provider(_text):
            return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        engine.enable_quant_aware_routing(plane, query_vector_provider=provider)
        standard, final, _mask, jaccard = engine.run_inference("route this")
        diagnostics = engine.get_last_runtime_diagnostics()
        self.assertEqual(standard[0], "doc-a")
        self.assertEqual(final[0], "doc-a")
        self.assertEqual(standard, final)
        self.assertEqual(jaccard, 1.0)
        self.assertEqual(diagnostics["route"]["key"], "route-a")
        self.assertEqual(diagnostics["retrieval_policy"]["policy"], "quant_aware_routing_plane")


if __name__ == "__main__":
    unittest.main()
