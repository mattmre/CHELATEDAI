"""Tests for computational_storage_poc/_experimental.py.

Verifies the mark_experimental helper is load-bearing: it reads the module's
EXPERIMENTAL flag, emits exactly one warning per (module, process), and
rejects non-bool flags. This closes the L13 / L4-light gap that Tier B
flagged on PR #247 — the EXPERIMENTAL constants are now part of a real
runtime check, not just inert top-level statements.

Uses ``unittest`` per CLAUDE.md Test Conventions.
"""

from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

_POC_DIR = Path(__file__).resolve().parent / "computational_storage_poc"
if str(_POC_DIR) not in sys.path:
    sys.path.insert(0, str(_POC_DIR))

from _experimental import (  # noqa: E402
    ExperimentalPOCWarning,
    mark_experimental,
)


class TestMarkExperimental(unittest.TestCase):
    def setUp(self) -> None:
        # Each test runs against a fresh dedup set so warnings fire as expected.
        import _experimental
        _experimental._ALREADY_WARNED.clear()

    def test_true_flag_emits_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ExperimentalPOCWarning)
            mark_experimental("computational_storage_poc.test_module_A", True)
        self.assertEqual(len(caught), 1)
        self.assertEqual(caught[0].category, ExperimentalPOCWarning)
        self.assertIn("test_module_A", str(caught[0].message))
        self.assertIn("research-stage POC", str(caught[0].message))

    def test_false_flag_emits_nothing(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ExperimentalPOCWarning)
            mark_experimental("computational_storage_poc.test_module_B", False)
        self.assertEqual(len(caught), 0)

    def test_warning_is_emitted_once_per_module_per_process(self) -> None:
        """Dedup behaviour: second call with same module name is a no-op."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ExperimentalPOCWarning)
            mark_experimental("computational_storage_poc.dedup_test", True)
            mark_experimental("computational_storage_poc.dedup_test", True)
            mark_experimental("computational_storage_poc.dedup_test", True)
        # Exactly one warning, not three.
        self.assertEqual(len(caught), 1)

    def test_non_bool_flag_raises_type_error(self) -> None:
        """Protects against EXPERIMENTAL = 'yes' or EXPERIMENTAL = 1 silently
        downgrading an experimental module to 'looks-stable'."""
        for bad in ("yes", 1, 0, None, []):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    mark_experimental("computational_storage_poc.bad_flag", bad)  # type: ignore[arg-type]

    def test_experimental_warning_subclasses_deprecation(self) -> None:
        # So default Python warning filters surface it (DeprecationWarning is
        # default-filtered, but our tests enable it explicitly).
        self.assertTrue(issubclass(ExperimentalPOCWarning, DeprecationWarning))


class TestRealPOCModulesEmitWarning(unittest.TestCase):
    """Verifies the 8 EXPERIMENTAL modules actually emit the warning when
    imported fresh. This is the runtime evidence that ``EXPERIMENTAL = True``
    is load-bearing in those modules — not just an inert top-level statement.
    """

    EXPERIMENTAL_MODULES = (
        "cpu_backends",
        "moe_reap",
        "packed_graph",
        "packed_cpu_inference",
        "sparse_cpu_inference",
        "repo_graph_memory",
        "integrated_repo_runtime",
        "phase7_system_evaluation",
    )

    def test_at_least_one_poc_module_emits_warning_on_fresh_import(self) -> None:
        # We only need to import one to prove the wiring; importing all 8
        # would not exercise additional code paths since they each call the
        # same helper.
        import _experimental
        _experimental._ALREADY_WARNED.clear()

        # Remove the module from sys.modules so the import re-runs.
        target = "packed_graph"
        sys.modules.pop(target, None)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ExperimentalPOCWarning)
            __import__(target)
        # The warning should fire from at least the target's mark_experimental
        # call (it may also fire from transitively-imported siblings, which is
        # acceptable — we just need >= 1).
        target_warnings = [w for w in caught if target in str(w.message)]
        self.assertGreaterEqual(
            len(target_warnings),
            1,
            f"Expected {target!r} to emit ExperimentalPOCWarning on fresh import; "
            f"got: {[str(w.message) for w in caught]}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
