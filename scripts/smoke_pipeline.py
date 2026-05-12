#!/usr/bin/env python3
"""scripts/smoke_pipeline.py — Stage 2 of the v3.3 Rule 5 smoke gate.

REPO-SPECIFIC FILL for ChelatedAI (mattmre/CHELATEDAI).

Production module surface identified from CLAUDE.md dependency graph:
  - Primary entry point: antigravity_engine.AntigravityEngine
  - Core deps: chelation_adapter, embedding_backend, vector_store, config
  - Architecture: library/class-based API (not a CLI script with main())
  - No public module-level constants (class-based config via ChelationConfig)
  - No input-scanner function (embedding requests come in via API call)

Placeholder substitutions (filled from real CLAUDE.md / source, not guessed):
  {{PRODUCTION_MODULE}}      → antigravity_engine
  {{EXPECTED_CONSTANTS}}     → []  (class-based module, no module-level constants)
  {{ENTRY_POINT_NAMES}}      → ["AntigravityEngine"]  (the central class)
  {{SCANNER_NAMES}}          → []  (no input-iteration function; library API model)
  {{ALSO_REQUIRED_IMPORTS}}  → ["chelation_adapter", "embedding_backend",
                                "vector_store", "config"]

Ceiling-tier gap: CD-001 in docs/next-session.md — run_ceiling_smoke() is
NOT YET IMPLEMENTED (returns sentinel 2). The PR body SMOKE: line must state
"floor-tier only" and reference CD-001.

Two-tier framing per Brutal Honesty Rulebook v3.3 §1 + §10 Rule 5:
  Floor (this file, active):
    Imports the production module, verifies its documented surface (entry
    point class, core deps). Catches L1 scaffold-as-feature, L9
    dependency-phantom, L10 broken imports. Acceptable as the v3.3 minimum
    but NOT a substitute for ceiling-tier verification.
  Ceiling (target — CD-001):
    Runs AntigravityEngine against a real embedding fixture, calls embed()
    or run_inference(), asserts returned vector has expected shape and is
    not all-zeros. Not yet implemented; tracked as CD-001.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _print(msg: str) -> None:
    print(msg, flush=True)


def _fail(reason: str) -> int:
    _print(f"SMOKE_PIPELINE FAIL: {reason}")
    return 1


def _ok(msg: str) -> None:
    _print(f"  OK: {msg}")


def run_floor_smoke() -> int:
    """Floor-tier smoke: import + surface verification through production code.

    Returns 0 on PASS, 1 on FAIL. Does NOT require GPU, models, or external
    services. Acceptable as the v3.3 minimum but not a substitute for ceiling.
    """
    _print(f"smoke_pipeline.py (floor tier) — REPO_ROOT={REPO_ROOT}")

    # --- Make the repo importable -------------------------------------------
    sys.path.insert(0, str(REPO_ROOT))

    # --- 1. Import the primary production module ----------------------------
    production_module_name = "antigravity_engine"
    try:
        import importlib
        production_module = importlib.import_module(production_module_name)
        _ok(f"imported {production_module_name}")
    except ImportError as exc:
        return _fail(
            f"cannot import production module {production_module_name!r}: {exc}\n"
            f"  Production deps are not installed in the current Python env.\n"
            f"  Stage 2 cannot honestly verify the production path without them.\n"
            f"  Fix: pip install -r requirements.txt"
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(
            f"{production_module_name} import raised non-ImportError: {exc!r}\n"
            f"{traceback.format_exc()}"
        )

    # --- 2. Import additional required production modules -------------------
    also_required: list[str] = [
        "chelation_adapter",
        "embedding_backend",
        "vector_store",
        "config",
    ]
    for mod_name in also_required:
        try:
            importlib.import_module(mod_name)
            _ok(f"imported {mod_name}")
        except ImportError as exc:
            return _fail(f"cannot import required production module {mod_name!r}: {exc}")
        except Exception as exc:  # noqa: BLE001
            return _fail(
                f"{mod_name} import raised non-ImportError: {exc!r}\n"
                f"{traceback.format_exc()}"
            )

    # --- 3. Verify module-level constants (none documented for this repo) --
    expected_constants: list[str] = []
    if expected_constants:
        missing = [c for c in expected_constants if not hasattr(production_module, c)]
        if missing:
            return _fail(
                f"{production_module_name} is missing documented constants: {missing}\n"
                f"  Missing them is L4 partial-with-claim-of-complete."
            )
        _ok(f"production constants present: {expected_constants}")

    # --- 4. Verify the central entry-point class exists --------------------
    entry_point_names: list[str] = ["AntigravityEngine"]
    if entry_point_names:
        has_entry = next(
            (n for n in entry_point_names if hasattr(production_module, n)),
            None,
        )
        if has_entry is None:
            return _fail(
                f"{production_module_name} exposes no entry point "
                f"(looked for {entry_point_names}). This is L1 scaffold-as-feature."
            )
        _ok(f"production entry point exposed: {has_entry}")

    # --- 5. Verify input-iteration function (none for library API model) ---
    scanner_names: list[str] = []
    if scanner_names:
        has_scanner = [n for n in scanner_names if hasattr(production_module, n)]
        if not has_scanner:
            return _fail(
                f"{production_module_name} exposes no input-iteration function "
                f"(looked for {scanner_names}). This is L1 scaffold-as-feature."
            )
        _ok(f"production input-iteration functions exposed: {has_scanner}")

    return 0


def run_ceiling_smoke() -> int:
    """Ceiling-tier smoke: real end-to-end exercise against a real fixture.

    NOT YET IMPLEMENTED — returns sentinel 2 (deferred, not a failure).
    Tracked as CD-001 in docs/next-session.md.

    Target implementation:
        engine = AntigravityEngine(model_name="all-MiniLM-L6-v2",
                                   qdrant_location=":memory:")
        result = engine.embed("smoke test sentence")
        assert len(result) > 0, "embed() returned empty vector"
        assert any(v != 0.0 for v in result), "embed() returned all-zeros (likely broken)"
        _ok("ceiling-tier: embed() produced non-zero vector")
        return 0
    """
    return 2


def main() -> int:
    floor_status = run_floor_smoke()
    if floor_status != 0:
        return floor_status

    ceiling_status = run_ceiling_smoke()

    _print("")
    if ceiling_status == 2:
        # HONEST DISCLOSURE: floor-tier PASS, ceiling-tier NOT IMPLEMENTED.
        # This block is load-bearing per Rulebook v3.3 §1 Rule 5 — do not delete it.
        _print("HONEST DISCLOSURE (per Brutal Honesty Rulebook v3.3 §1 Rule 5):")
        _print("  This smoke ran the FLOOR tier only (import + surface-check through")
        _print("  production code paths). The ceiling tier (real end-to-end against a")
        _print("  fixture) is NOT YET IMPLEMENTED in run_ceiling_smoke().")
        _print("  Per Rule 5, claiming 'smoke passed' at ceiling-tier when only floor")
        _print("  ran is L4 partial-with-claim-of-complete. The PR body SMOKE: line")
        _print("  must name 'floor-tier only' AND the ceiling-tier gap appears as")
        _print("  CD-001 in docs/next-session.md per §6.3.")
        _print("")
        _print("SMOKE_PIPELINE PASS (floor tier; ceiling tier deferred as CD-001)")
        return 0
    if ceiling_status != 0:
        return ceiling_status

    _print("")
    _print("SMOKE_PIPELINE PASS (floor + ceiling tiers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
