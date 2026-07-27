import math
import threading
import unittest
from functools import partial
from unittest.mock import MagicMock

from adapter_router import AdapterRouter


class _BlockingIterable:
    def __init__(self, entered, release):
        self.entered = entered
        self.release = release

    def __iter__(self):
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release blocked centroid conversion")
        yield 0.0
        yield 1.0


class _BlockingFloat:
    def __init__(self, entered, release, value):
        self.entered = entered
        self.release = release
        self.value = value

    def __float__(self):
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release blocked margin conversion")
        return self.value


class TestAdapterRouter(unittest.TestCase):
    def _freeze_during_mutation_preparation(self, router, mutation, entered, release):
        errors = []

        def mutate():
            try:
                mutation()
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=mutate)
        worker.start()
        try:
            self.assertTrue(entered.wait(timeout=2))
            frozen_checksum = router.freeze()
        finally:
            release.set()
            worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertEqual(str(errors[0]), "adapter router is frozen")
        self.assertEqual(router.state_checksum(), frozen_checksum)

    def test_legacy_nearest_centroid_and_empty_fallback(self):
        router = AdapterRouter(logger=MagicMock())
        fallback = router.select([1.0, 0.0], fallback=lambda: "default")
        self.assertEqual(fallback.key, "fallback")

        router.register("x", [1.0, 0.0], "adapter-x")
        router.register("y", [0.0, 1.0], "adapter-y")
        selected = router.select([0.9, 0.1])
        self.assertEqual(selected.key, "x")
        self.assertEqual(selected.adapter, "adapter-x")

    def test_margin_equal_delta_routes_and_below_delta_falls_back_global(self):
        exact = AdapterRouter(margin_delta=1.0, logger=MagicMock())
        exact.register("specialist", [1.0, 0.0], "specialist")
        exact.register_global([0.0, 1.0], "global")
        routed = exact.select([1.0, 0.0])
        self.assertEqual(routed.key, "specialist")
        self.assertAlmostEqual(routed.metadata["margin"], 1.0)
        self.assertFalse(routed.metadata["used_margin_fallback"])

        strict = AdapterRouter(margin_delta=1.01, logger=MagicMock())
        strict.register("specialist", [1.0, 0.0], "specialist")
        strict.register_global([0.0, 1.0], "global")
        fallback = strict.select([1.0, 0.0])
        self.assertEqual(fallback.key, "global")
        self.assertAlmostEqual(fallback.metadata["global_score"], 0.0)
        self.assertAlmostEqual(fallback.metadata["margin"], 1.0)
        self.assertTrue(fallback.metadata["used_margin_fallback"])

    def test_split_scoped_usage_is_not_history_capped(self):
        router = AdapterRouter(logger=MagicMock())
        router.register("x", [1.0, 0.0], "x")
        router.register("y", [0.0, 1.0], "y")
        for index in range(600):
            query = [1.0, 0.0] if index % 2 == 0 else [0.0, 1.0]
            router.select(query, usage_scope="REPORT")
        usage = router.get_usage_summary("REPORT")
        self.assertEqual(usage["total"], 600)
        self.assertEqual(usage["histogram"], {"x": 300, "y": 300})
        self.assertEqual(usage["p_k"], {"x": 0.5, "y": 0.5})
        self.assertAlmostEqual(usage["entropy_nats"], math.log(2.0))
        self.assertEqual(usage["n_used"], 2)
        self.assertEqual(usage["n_routes_used"], 2)

    def test_global_fallback_is_excluded_from_specialist_route_count(self):
        router = AdapterRouter(margin_delta=0.5, logger=MagicMock())
        router.register("x", [1.0, 0.0], "x")
        router.register_global([0.0, 1.0], "global")
        router.select([1.0, 0.0], usage_scope="REPORT")
        router.select([0.0, 1.0], usage_scope="REPORT")
        usage = router.get_usage_summary("REPORT")
        self.assertEqual(usage["n_used"], 2)
        self.assertEqual(usage["n_routes_used"], 1)
        self.assertEqual(usage["route_histogram"], {"x": 1})

    def test_invalid_vectors_fail_explicitly(self):
        router = AdapterRouter(logger=MagicMock())
        with self.assertRaises(ValueError):
            router.register("empty", [], "adapter")
        with self.assertRaises(ValueError):
            router.register("nan", [float("nan"), 0.0], "adapter")
        router.register("x", [1.0, 0.0], "adapter")
        with self.assertRaises(ValueError):
            router.select([0.0, 0.0])
        with self.assertRaises(ValueError):
            router.select([float("inf"), 0.0])
        with self.assertRaises(ValueError):
            router.select([1.0, 0.0, 0.0])

    def test_freeze_locks_margin_membership_and_centroids(self):
        router = AdapterRouter(margin_delta=0.2, logger=MagicMock())
        router.register("x", [1.0, 0.0], "x")
        router.register_global([0.0, 1.0], "global")
        checksum = router.freeze()
        self.assertTrue(router.frozen)
        self.assertEqual(router.state_checksum(), checksum)
        with self.assertRaises(RuntimeError):
            router.margin_delta = 0.3
        with self.assertRaises(RuntimeError):
            router.register("y", [0.0, 1.0], "y")
        with self.assertRaises(RuntimeError):
            router.register_global([1.0, 1.0], "global-2")

    def test_freeze_is_atomic_with_route_registration(self):
        for registration in ("specialist", "global"):
            with self.subTest(registration=registration):
                router = AdapterRouter(logger=MagicMock())
                entered = threading.Event()
                release = threading.Event()
                centroid = _BlockingIterable(entered, release)
                if registration == "specialist":
                    mutation = partial(router.register, "late", centroid, "adapter")
                else:
                    mutation = partial(router.register_global, centroid, "adapter")

                self._freeze_during_mutation_preparation(
                    router,
                    mutation,
                    entered,
                    release,
                )
                with self.assertRaises(ValueError):
                    router.select([0.0, 1.0])

    def test_freeze_is_atomic_with_margin_mutation(self):
        router = AdapterRouter(margin_delta=0.2, logger=MagicMock())
        entered = threading.Event()
        release = threading.Event()
        value = _BlockingFloat(entered, release, 0.4)

        def mutate_margin():
            router.margin_delta = value

        self._freeze_during_mutation_preparation(
            router,
            mutate_margin,
            entered,
            release,
        )
        self.assertEqual(router.margin_delta, 0.2)


if __name__ == "__main__":
    unittest.main()
