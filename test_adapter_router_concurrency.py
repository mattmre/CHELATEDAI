"""Concurrency regression test for AdapterRouter (PR295-01 verdict lock-in).

The "registry race" premise was falsified both statically (every shared-state
access holds `self._lock`) and empirically (16-thread hammer: 0 errors).
This bounded variant runs in CI to prevent regressions: mixed
select/record/register/getter traffic must produce no errors, no stuck
threads, an intact registry, and a correctly capped history.
"""

import threading
import unittest

from adapter_router import AdapterRouter


class TestAdapterRouterConcurrency(unittest.TestCase):
    def test_mixed_concurrent_traffic(self):
        router = AdapterRouter()
        dim = 8
        for i in range(8):
            router.register(f"a{i}", [float(i + 1)] * dim, object())

        errors = []
        threads = 8
        ops = 100

        def worker(w):
            try:
                for i in range(ops):
                    query = [float((i + w) % 8 + 1)] * dim
                    route = router.select(query)
                    self.assertIn(route.key, {f"a{j}" for j in range(8)})
                    router.record_outcome(route.key, 0.5, 1.0)
                    if i % 25 == 0:
                        router.register(
                            f"a{(i + w) % 8}",
                            [float((i + w) % 8 + 1)] * dim,
                            object(),
                        )
                        router.get_route_history()
                        router.get_last_route_outcome()
                        router.get_route_effectiveness()
            except Exception as exc:  # noqa: BLE001 — harness records failures
                errors.append(repr(exc))

        workers = [threading.Thread(target=worker, args=(w,)) for w in range(threads)]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join(120)
        self.assertFalse([t for t in workers if t.is_alive()], "stuck threads")
        self.assertEqual(errors, [])
        self.assertEqual(len(router._routes), 8)
        self.assertLessEqual(len(router.get_route_history()), 256)


if __name__ == "__main__":
    unittest.main()
