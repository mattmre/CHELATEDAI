import hashlib
import inspect
import json
import math
import subprocess
import threading
import unittest
from dataclasses import replace
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
    graded_ndcg_at_k,
    paired_query_bootstrap_ci,
    three_way_seeded_split,
)
from quantization_promotion_gate import QuantizationPromotionGate
from run_quant_aware_routing_campaign import _source_provenance, run_arena


class FixedLinear(torch.nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.register_buffer("matrix", torch.tensor(matrix, dtype=torch.float32))

    def forward(self, inputs):
        return inputs @ self.matrix.T


class InjectedCancellation(BaseException):
    """Hostile non-Exception cancellation signal used by state-machine tests."""


class TestSourceProvenance(unittest.TestCase):
    def _git_runner(self, *, dirty_status="", failing_command=None):
        repo_root = str(Path(__file__).resolve().parent)
        tracked_files = "adapter_router.py\0quantization_promotion_gate.py\0"

        def run(command, **_kwargs):
            arguments = tuple(str(argument) for argument in command[1:])
            if arguments == failing_command:
                raise subprocess.CalledProcessError(128, command)
            if arguments == ("rev-parse", "--show-toplevel"):
                output = f"{repo_root}\n"
            elif arguments[:3] == ("ls-files", "--error-unmatch", "--"):
                output = ""
            elif arguments == ("rev-parse", "HEAD"):
                output = "deadbeef\n"
            elif arguments == ("rev-parse", "HEAD^{tree}"):
                output = "treebeef\n"
            elif arguments == ("ls-files", "-z"):
                output = tracked_files
            elif arguments == ("status", "--porcelain=v1", "--untracked-files=no"):
                output = dirty_status
            else:
                raise AssertionError(f"unexpected Git command: {command!r}")
            return subprocess.CompletedProcess(command, 0, stdout=output)

        return run

    def test_source_provenance_fails_closed_when_git_or_head_is_unavailable(self):
        preregistration = Path(__file__).resolve().with_name("prereg_rung16.json")
        with patch(
            "run_quant_aware_routing_campaign.subprocess.run",
            side_effect=FileNotFoundError("git executable is unavailable"),
        ):
            with self.assertRaises(FileNotFoundError):
                _source_provenance(preregistration)

        with patch(
            "run_quant_aware_routing_campaign.subprocess.run",
            side_effect=self._git_runner(failing_command=("rev-parse", "HEAD")),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                _source_provenance(preregistration)

    def test_source_provenance_fails_closed_when_status_capture_fails(self):
        preregistration = Path(__file__).resolve().with_name("prereg_rung16.json")
        with patch(
            "run_quant_aware_routing_campaign.subprocess.run",
            side_effect=self._git_runner(
                failing_command=("status", "--porcelain=v1", "--untracked-files=no")
            ),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                _source_provenance(preregistration)

    def test_source_provenance_rejects_dirty_tracked_dependency(self):
        preregistration = Path(__file__).resolve().with_name("prereg_rung16.json")
        with patch(
            "run_quant_aware_routing_campaign.subprocess.run",
            side_effect=self._git_runner(
                dirty_status=" M quantization_promotion_gate.py\n"
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "quantization_promotion_gate"):
                _source_provenance(preregistration)

    def test_source_provenance_binds_complete_clean_tracked_tree(self):
        preregistration = Path(__file__).resolve().with_name("prereg_rung16.json")
        with patch(
            "run_quant_aware_routing_campaign.subprocess.run",
            side_effect=self._git_runner(),
        ):
            provenance = _source_provenance(preregistration)
        self.assertEqual(provenance["git_head"], "deadbeef")
        self.assertEqual(provenance["git_tree"], "treebeef")
        self.assertEqual(provenance["tracked_worktree_scope"], "entire_repository")
        self.assertTrue(provenance["tracked_worktree_clean"])
        self.assertEqual(provenance["tracked_file_count"], 2)
        self.assertIn("quant_aware_routing.py", provenance["sha256"])


def _permutation(first, second, dim=4):
    matrix = np.eye(dim, dtype=np.float32)
    matrix[[first, second]] = matrix[[second, first]]
    return matrix


def _passing_plane(
    *,
    preregistration=None,
    include_unseen_bad_route=False,
    relevance_override=None,
):
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
    if relevance_override is not None:
        qrels[select_ids[0]] = {"doc-a": relevance_override}
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


class TestFinitePromotionParameters(unittest.TestCase):
    def test_every_floating_routing_parameter_rejects_nonfinite_values(self):
        config = RoutingPlaneConfig()
        floating_fields = tuple(
            name
            for name, value in vars(config).items()
            if isinstance(value, float)
        )
        self.assertEqual(
            set(floating_fields),
            {
                "anchor_fraction",
                "select_fraction",
                "report_fraction",
                "margin_delta",
                "min_report_route_fraction",
                "confidence_level",
                "min_lift",
                "max_floor_loss",
                "quant_retained_gain",
                "quant_minimum_fp32_gain",
                "quantile",
                "adapter_learning_rate",
                "adapter_min_correction",
                "adapter_max_correction",
            },
        )
        for field_name in floating_fields:
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(field=field_name, value=value):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        replace(config, **{field_name: value}).validate()

    def test_quantization_gate_rejects_nonfinite_configuration_and_inputs(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(parameter="retained_gain_threshold", value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    QuantizationPromotionGate(retained_gain_threshold=value)
            with self.subTest(parameter="minimum_fp32_gain", value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    QuantizationPromotionGate(minimum_fp32_gain=value)

        gate = QuantizationPromotionGate()
        for position in range(3):
            for value in (float("nan"), float("inf"), -float("inf")):
                metrics = [1.0, 0.9, 0.0]
                metrics[position] = value
                with self.subTest(metric_position=position, value=value):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        gate.evaluate(*metrics[:2], baseline_fitness=metrics[2])

    def test_quantization_gate_rejects_nonfinite_derived_values(self):
        gate = QuantizationPromotionGate(logger=MagicMock())
        with self.assertRaisesRegex(ValueError, "derived gains.*finite"):
            gate.evaluate(1e308, 1e308, baseline_fitness=-1e308)
        with self.assertRaisesRegex(ValueError, "retained gain ratio.*finite"):
            gate.evaluate(5e-324, 1e308, baseline_fitness=0.0)

    def test_k_requires_an_exact_positive_builtin_integer(self):
        wrong_types = (None, True, False, 1.0, np.int64(1), "1")
        for value in wrong_types:
            with self.subTest(surface="ndcg", value=value):
                with self.assertRaisesRegex(TypeError, "built-in int"):
                    graded_ndcg_at_k(["doc"], {"doc": 1.0}, value)
            for field_name in ("k", "route_k"):
                with self.subTest(surface="config", field=field_name, value=value):
                    with self.assertRaisesRegex(TypeError, "built-in int"):
                        replace(
                            RoutingPlaneConfig(),
                            **{field_name: value},
                        ).validate()
        for value in (0, -1):
            with self.subTest(surface="ndcg", value=value):
                with self.assertRaisesRegex(ValueError, "must be >= 1"):
                    graded_ndcg_at_k(["doc"], {"doc": 1.0}, value)
            for field_name in ("k", "route_k"):
                with self.subTest(surface="config", field=field_name, value=value):
                    with self.assertRaisesRegex(ValueError, "must be >= 1"):
                        replace(
                            RoutingPlaneConfig(),
                            **{field_name: value},
                        ).validate()

    def test_preregistration_route_k_is_not_silently_coerced(self):
        preregistration = json.loads(
            Path(__file__).resolve().with_name("prereg_rung16.json").read_text(
                encoding="utf-8"
            )
        )
        preregistration["routing"]["k"] = 1.9
        with self.assertRaisesRegex(TypeError, "route_k must be a built-in int"):
            RoutingPlaneConfig.from_preregistration(preregistration)


class TestRelevanceValidation(unittest.TestCase):
    def test_relevance_rejects_nonfinite_and_unrepresentable_gains(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(surface="ndcg", value=value):
                with self.assertRaisesRegex(ValueError, "relevance scores.*finite"):
                    graded_ndcg_at_k(["doc"], {"doc": value}, 1)
            with self.subTest(surface="plane", value=value):
                with self.assertRaisesRegex(ValueError, "relevance scores.*finite"):
                    _passing_plane(relevance_override=value)
        with self.assertRaisesRegex(ValueError, "relevance gains.*finite"):
            graded_ndcg_at_k(["doc"], {"doc": 1024.0}, 1)
        with self.assertRaisesRegex(ValueError, "relevance gains.*finite"):
            _passing_plane(relevance_override=1024.0)

    def test_finite_relevance_that_overflows_aggregate_dcg_is_rejected(self):
        relevance = {
            "doc-a": 1023.0,
            "doc-b": 1023.0,
            "doc-c": 1023.0,
        }
        with self.assertRaisesRegex(ValueError, "DCG.*finite"):
            graded_ndcg_at_k(
                ["doc-a", "doc-b", "doc-c"],
                relevance,
                3,
            )
        with self.assertRaisesRegex(ValueError, "IDCG.*finite"):
            graded_ndcg_at_k([], relevance, 3)


class TestSelectionLockForensics(unittest.TestCase):
    def test_reconciliation_matches_preserved_historical_bytes(self):
        root = Path(__file__).resolve().parent
        reconciliation_path = (
            root / "docs" / "rung16-selection-lock-hash-reconciliation-2026-07.json"
        )
        reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))
        self.assertEqual(reconciliation["status"], "FORENSIC_MISMATCH_DISCLOSED")
        self.assertEqual(len(reconciliation["records"]), 2)
        for record in reconciliation["records"]:
            with self.subTest(arena=record["arena"]):
                lock_path = root / record["selection_lock_path"]
                marker_path = root / record["marker_path"]
                actual_lock_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest()
                actual_marker_hash = hashlib.sha256(marker_path.read_bytes()).hexdigest()
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    actual_lock_hash,
                    record["selection_lock_actual_sha256"],
                )
                self.assertEqual(actual_marker_hash, record["marker_file_sha256"])
                self.assertEqual(
                    marker["selection_lock_sha256"],
                    record["marker_claimed_selection_lock_sha256"],
                )
                self.assertNotEqual(
                    actual_lock_hash,
                    marker["selection_lock_sha256"],
                )
                self.assertFalse(record["match"])


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

    def test_select_reservation_freezes_router_before_any_evaluation(self):
        plane, select_vectors, report_vectors = _passing_plane()
        entered = threading.Event()
        release = threading.Event()
        original_evaluate = plane._evaluate
        results = []
        errors = []

        def blocked_evaluate(*args, **kwargs):
            if not entered.is_set():
                if not plane.router.frozen:
                    raise AssertionError("router was not frozen before SELECT evaluation")
                if plane._router_checksum != plane.router.state_checksum():
                    raise AssertionError("SELECT did not retain the frozen router checksum")
                entered.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("test did not release SELECT evaluation")
            return original_evaluate(*args, **kwargs)

        def run_select():
            try:
                results.append(plane.select(select_vectors))
            except Exception as exc:
                errors.append(exc)

        plane._evaluate = blocked_evaluate
        worker = threading.Thread(target=run_select)
        worker.start()
        try:
            self.assertTrue(entered.wait(timeout=2))
            self.assertEqual(plane.state, "SELECTING")
            frozen_checksum = plane.router.state_checksum()
            competing_mutations = (
                lambda: plane.router.register(
                    "late-route",
                    [1.0, 1.0, 0.0, 0.0],
                    FixedLinear(np.eye(4, dtype=np.float32)),
                ),
                lambda: plane.router.register_global(
                    [1.0, 1.0, 1.0, 1.0],
                    FixedLinear(np.eye(4, dtype=np.float32)),
                ),
                lambda: setattr(plane.router, "margin_delta", 0.5),
            )
            for mutation in competing_mutations:
                with self.assertRaisesRegex(RuntimeError, "frozen"):
                    mutation()
                self.assertEqual(plane.router.state_checksum(), frozen_checksum)
            with self.assertRaisesRegex(RuntimeError, "SELECTING"):
                plane.select(select_vectors)
            with self.assertRaisesRegex(RuntimeError, "SELECTING"):
                plane.report(report_vectors)
        finally:
            release.set()
            worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 1)
        self.assertEqual(plane.state, "SELECT_FROZEN")
        self.assertEqual(results[0]["router_state_sha256"], frozen_checksum)

    def test_failed_select_rolls_back_for_retry_but_retains_frozen_router(self):
        plane, select_vectors, _report_vectors = _passing_plane()
        original_evaluate = plane._evaluate

        def fail_select(*_args, **_kwargs):
            raise RuntimeError("injected SELECT evaluation failure")

        plane._evaluate = fail_select
        with self.assertRaisesRegex(RuntimeError, "injected SELECT"):
            plane.select(select_vectors)
        frozen_checksum = plane.router.state_checksum()
        self.assertTrue(plane.router.frozen)
        self.assertEqual(plane.state, "ANCHOR_READY")
        self.assertEqual(plane._router_checksum, frozen_checksum)
        self.assertFalse(plane.report_consumed)
        self.assertIsNone(plane.provenance()["select"])

        plane._evaluate = original_evaluate
        result = plane.select(select_vectors)
        self.assertEqual(plane.state, "SELECT_FROZEN")
        self.assertEqual(result["router_state_sha256"], frozen_checksum)

    def test_select_baseexceptions_cleanup_and_reraise_original_signal(self):
        for signal in (
            KeyboardInterrupt("keyboard cancellation"),
            SystemExit(17),
            InjectedCancellation("injected cancellation"),
        ):
            with self.subTest(signal=type(signal).__name__):
                plane, select_vectors, _report_vectors = _passing_plane()
                original_evaluate = plane._evaluate

                def interrupt_select(*_args, **_kwargs):
                    raise signal

                plane._evaluate = interrupt_select
                caught = None
                try:
                    plane.select(select_vectors)
                except BaseException as exc:
                    caught = exc
                self.assertIs(caught, signal)
                frozen_checksum = plane.router.state_checksum()
                self.assertEqual(plane.state, "ANCHOR_READY")
                self.assertTrue(plane.router.frozen)
                self.assertEqual(plane._router_checksum, frozen_checksum)
                self.assertFalse(plane.report_consumed)
                self.assertIsNone(plane.provenance()["select"])

                plane._evaluate = original_evaluate
                retry = plane.select(select_vectors)
                self.assertEqual(plane.state, "SELECT_FROZEN")
                self.assertEqual(retry["router_state_sha256"], frozen_checksum)

    def test_concurrent_report_is_reserved_once_and_finalized_atomically(self):
        plane, select_vectors, report_vectors = _passing_plane()
        plane.select(select_vectors)
        entered = threading.Event()
        release = threading.Event()
        original_evaluate = plane._evaluate
        results = []
        errors = []

        def blocked_evaluate(*args, **kwargs):
            if not entered.is_set():
                entered.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("test did not release REPORT evaluation")
            return original_evaluate(*args, **kwargs)

        def run_report():
            try:
                results.append(plane.report(report_vectors))
            except Exception as exc:
                errors.append(exc)

        plane._evaluate = blocked_evaluate
        worker = threading.Thread(target=run_report)
        worker.start()
        try:
            self.assertTrue(entered.wait(timeout=2))
            self.assertEqual(plane.state, "REPORTING")
            self.assertTrue(plane.report_consumed)
            with self.assertRaisesRegex(RuntimeError, "REPORTING"):
                plane.report(report_vectors)
            with self.assertRaisesRegex(RuntimeError, "REPORTING"):
                plane.provenance()
            with self.assertRaisesRegex(RuntimeError, "REPORTING"):
                _ = plane.verdict
        finally:
            release.set()
            worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["verdict"], "PROMOTED")
        self.assertEqual(plane.state, "FINALIZED")
        self.assertEqual(plane.verdict, "PROMOTED")

    def test_report_exception_is_terminal_fail_closed_and_not_retryable(self):
        plane, select_vectors, report_vectors = _passing_plane()
        plane.select(select_vectors)

        def fail_report(*_args, **_kwargs):
            raise RuntimeError("injected REPORT evaluation failure")

        plane._evaluate = fail_report
        with self.assertRaisesRegex(RuntimeError, "injected REPORT"):
            plane.report(report_vectors)
        self.assertEqual(plane.state, "REPORT_FAILED_CLOSED")
        self.assertTrue(plane.report_consumed)
        self.assertEqual(plane.verdict, "FAIL-CLOSED")
        provenance = plane.provenance()
        self.assertEqual(provenance["report"]["error_type"], "RuntimeError")
        self.assertEqual(provenance["report"]["verdict"], "FAIL-CLOSED")
        with self.assertRaisesRegex(RuntimeError, "REPORT_FAILED_CLOSED"):
            plane.report(report_vectors)

    def test_report_baseexceptions_finalize_fail_closed_and_reraise_original_signal(self):
        for signal in (
            KeyboardInterrupt("keyboard cancellation"),
            SystemExit(23),
            InjectedCancellation("injected cancellation"),
        ):
            with self.subTest(signal=type(signal).__name__):
                plane, select_vectors, report_vectors = _passing_plane()
                plane.select(select_vectors)

                def interrupt_report(*_args, **_kwargs):
                    raise signal

                plane._evaluate = interrupt_report
                caught = None
                try:
                    plane.report(report_vectors)
                except BaseException as exc:
                    caught = exc
                self.assertIs(caught, signal)
                self.assertEqual(plane.state, "REPORT_FAILED_CLOSED")
                self.assertTrue(plane.report_consumed)
                self.assertEqual(plane.verdict, "FAIL-CLOSED")
                provenance = plane.provenance()
                self.assertEqual(
                    provenance["report"]["error_type"],
                    type(signal).__name__,
                )
                self.assertEqual(provenance["report"]["verdict"], "FAIL-CLOSED")
                with self.assertRaisesRegex(RuntimeError, "REPORT_FAILED_CLOSED"):
                    plane.report(report_vectors)

    def test_invalid_report_vectors_do_not_consume_the_one_shot_report(self):
        plane, select_vectors, report_vectors = _passing_plane()
        plane.select(select_vectors)
        invalid_vectors = dict(report_vectors)
        invalid_vectors[next(iter(invalid_vectors))] = np.asarray(
            [float("nan"), 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        with self.assertRaisesRegex(ValueError, "invalid REPORT query vector"):
            plane.report(invalid_vectors)
        self.assertEqual(plane.state, "SELECT_FROZEN")
        self.assertFalse(plane.report_consumed)
        self.assertEqual(plane.report(report_vectors)["verdict"], "PROMOTED")

    def test_retrieve_k_none_is_default_and_invalid_values_have_no_route_side_effect(self):
        plane, select_vectors, report_vectors = _passing_plane()
        plane.select(select_vectors)
        plane.report(report_vectors)

        default_result = plane.retrieve([1.0, 0.0, 0.0, 0.0], k=None)
        explicit_result = plane.retrieve([1.0, 0.0, 0.0, 0.0], k=2)
        self.assertEqual(len(default_result["ids"]), plane.config.k)
        self.assertEqual(len(explicit_result["ids"]), 2)
        usage_before = plane.router.get_usage_summary("SERVE")

        for value in (True, False, 1.0, np.int64(1), "1"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "built-in int"):
                    plane.retrieve([1.0, 0.0, 0.0, 0.0], k=value)
                self.assertEqual(
                    plane.router.get_usage_summary("SERVE"),
                    usage_before,
                )
        for value in (0, -1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "must be >= 1"):
                    plane.retrieve([1.0, 0.0, 0.0, 0.0], k=value)
                self.assertEqual(
                    plane.router.get_usage_summary("SERVE"),
                    usage_before,
                )

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
