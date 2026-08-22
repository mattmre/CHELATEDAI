"""tests/test_e2e_smoke.py — Stage 1 surface-boot smoke for ``scripts/smoke.sh``.

This is the file ``scripts/smoke.sh`` Stage 1 invokes. Its sole job is to verify
that the load-bearing production surfaces of the ChelatedAI library import
cleanly without raising. It catches:

  - L1 scaffold-as-feature (module referenced but not present),
  - L9 dependency-phantom (a module references a dep not actually installed),
  - L10 broken imports / circular-import regressions.

It does NOT exercise any real model, GPU, embedding service, or fixture. That
is the ceiling-tier job tracked as ``CD-001`` against
``scripts/smoke_pipeline.py``.

Test conventions per ``CLAUDE.md``:
  - ``unittest`` only (CI does NOT install ``pytest``).
  - Invoked via ``python -m unittest tests.test_e2e_smoke``.
  - Flat repo layout: target modules live at the repo root, so we insert
    the repo root onto ``sys.path`` before importing.
"""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The flat repo layout puts every production module at the project root.
# Ensure that root is importable regardless of where pytest/unittest is run.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Modules whose import-time failure would invalidate every other smoke claim.
# Picked from the CLAUDE.md dependency graph: central engine, core deps, and
# the surfaces operators interact with (dashboard, orchestrator, sedimentation).
SURFACE_MODULES = (
    "egv",
    "egv.cli",
    "antigravity_engine",
    "chelation_adapter",
    "config",
    "chelation_logger",
    "embedding_backend",
    "vector_store",
    "sedimentation",
    "aep_orchestrator",
    "dashboard_server",
)


class SurfaceBootSmoke(unittest.TestCase):
    """Stage 1 surface-boot: every load-bearing module must import cleanly."""

    def test_surface_modules_import(self) -> None:
        failures: list[str] = []
        for name in SURFACE_MODULES:
            try:
                importlib.import_module(name)
            except Exception as exc:  # noqa: BLE001 — we want every failure recorded
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
        if failures:
            self.fail(
                "Surface-boot smoke failed; the following production modules "
                "did not import cleanly:\n  - " + "\n  - ".join(failures)
            )

    def test_central_entry_point_exposed(self) -> None:
        """``AntigravityEngine`` must remain the documented entry point."""
        module = importlib.import_module("antigravity_engine")
        self.assertTrue(
            hasattr(module, "AntigravityEngine"),
            "antigravity_engine module no longer exposes AntigravityEngine; "
            "this is L1 scaffold-as-feature against the documented entry point.",
        )


if __name__ == "__main__":
    unittest.main()
