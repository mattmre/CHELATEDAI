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

Ceiling-tier (CD-001 closure): runs AntigravityEngine end-to-end against an
in-memory Qdrant + local SentenceTransformer fixture. Ingests a small corpus,
computes a chelated query vector, and asserts non-trivial output. If torch /
sentence-transformers / antigravity_engine deps are missing in this Python
environment, the ceiling tier prints an explicit SKIP and exits 0 (an honest
skip is acceptable per Rule 5; a fake pass is L4). Per CLAUDE.md, the CI
matrix installs torch + sentence-transformers, so the ceiling tier SHOULD
run (not skip) in CI.

Two-tier framing per Brutal Honesty Rulebook v3.3 §1 + §10 Rule 5:
  Floor (active):
    Imports the production module, verifies its documented surface (entry
    point class, core deps). Catches L1 scaffold-as-feature, L9
    dependency-phantom, L10 broken imports. Acceptable as the v3.3 minimum
    but NOT a substitute for ceiling-tier verification.
  Ceiling (active — CD-001 closed):
    Constructs AntigravityEngine(qdrant_location=":memory:",
    model_name="all-MiniLM-L6-v2"), ingests 3 documents, runs
    get_chelated_vector() on a real query, and asserts the returned vector
    is a numpy array of the expected dim with non-zero norm.
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


CEILING_SKIPPED = 3
"""Sentinel exit status for ceiling-tier honest skip (missing heavy deps).

Distinct from sentinel 2 (which used to mean "not implemented"). A skip
exits the process with 0 because the surface gap is honest, not a failure;
this constant is internal-only and never escapes ``main()``.
"""


def run_ceiling_smoke() -> int:
    """Ceiling-tier smoke: real end-to-end exercise against AntigravityEngine.

    Closes CD-001. Returns:
        0                 — ceiling tier PASS (engine built, ingested, queried).
        1                 — ceiling tier FAIL (an assertion or runtime error fired).
        CEILING_SKIPPED   — heavy deps (torch / sentence-transformers / qdrant)
                            unavailable in this Python env. Honest skip; main()
                            translates this to a 0 exit with a SKIP banner.

    Per Rule 5, the ceiling tier MUST exercise a real fixture through the
    production code path. It does NOT mock the embedding backend or Qdrant —
    if those won't import, that is a dependency-phantom (L9) call, and skipping
    honestly is the correct response. Faking a pass would be L4.

    Per CLAUDE.md, the CI matrix installs torch + sentence-transformers; this
    function therefore runs (does not skip) on the standard CI path.
    """
    _print("")
    _print("--- ceiling tier: AntigravityEngine end-to-end exercise ---")

    # Heavy deps gate. Import errors here are honest skips, not failures.
    # We DO NOT broad-except: an import that raises something other than
    # ImportError is a real bug (L10 broken module) and must surface as FAIL.
    try:
        import numpy as np  # noqa: F401
        import torch  # noqa: F401
        import sentence_transformers  # noqa: F401
        import qdrant_client  # noqa: F401
    except ImportError as exc:
        _print(f"  SKIP: heavy dep missing ({exc.__class__.__name__}: {exc})")
        _print("  Honest skip per Rule 5; install requirements.txt to enable.")
        return CEILING_SKIPPED

    try:
        from antigravity_engine import AntigravityEngine
    except ImportError as exc:
        _print(f"  SKIP: antigravity_engine import failed ({exc})")
        _print("  Honest skip per Rule 5; install requirements.txt to enable.")
        return CEILING_SKIPPED

    import numpy as np

    # Build engine against the lightest viable config: in-memory Qdrant +
    # local SentenceTransformer (no Ollama HTTP call, no teacher distillation,
    # no quantization).
    try:
        engine = AntigravityEngine(
            qdrant_location=":memory:",
            model_name="all-MiniLM-L6-v2",
            training_mode="baseline",
        )
    except Exception as exc:  # noqa: BLE001 - surface ANY build failure as FAIL
        _print(f"  FAIL: AntigravityEngine construction raised: {exc!r}")
        _print(traceback.format_exc())
        return 1

    _ok(f"engine built (vector_size={engine.vector_size}, mode={engine.mode})")

    # Ingest a small real corpus — 4 short sentences spanning two topics so
    # the chelation mask has some actual variance to work against.
    corpus = [
        "Cats are small carnivorous mammals often kept as pets.",
        "Dogs are loyal animals frequently used as service companions.",
        "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
    ]
    try:
        engine.ingest(corpus)
    except Exception as exc:  # noqa: BLE001
        _print(f"  FAIL: engine.ingest raised: {exc!r}")
        _print(traceback.format_exc())
        return 1

    _ok(f"ingested {len(corpus)} documents")

    # Run a real query through the production retrieval entry point.
    try:
        chelated = engine.get_chelated_vector("What pets do people keep?")
    except Exception as exc:  # noqa: BLE001
        _print(f"  FAIL: engine.get_chelated_vector raised: {exc!r}")
        _print(traceback.format_exc())
        return 1

    chelated_arr = np.asarray(chelated)
    if chelated_arr.ndim != 1:
        return _fail(
            f"ceiling: get_chelated_vector returned ndim={chelated_arr.ndim}, "
            f"expected 1 (shape={chelated_arr.shape})"
        )
    if chelated_arr.shape[0] != engine.vector_size:
        return _fail(
            f"ceiling: get_chelated_vector returned len={chelated_arr.shape[0]}, "
            f"expected vector_size={engine.vector_size}"
        )
    norm = float(np.linalg.norm(chelated_arr))
    if not np.isfinite(norm):
        return _fail(f"ceiling: chelated vector norm is non-finite ({norm})")
    if norm == 0.0:
        return _fail("ceiling: chelated vector is all-zeros (broken pipeline)")

    _ok(
        f"get_chelated_vector PASS (shape={chelated_arr.shape}, "
        f"norm={norm:.4f}, dtype={chelated_arr.dtype})"
    )

    # Surface-check embed() too — it is the second production entry point and
    # is exercised by the recursive decomposer / dashboard.
    try:
        batch = engine.embed(["smoke test sentence", "another smoke sentence"])
    except Exception as exc:  # noqa: BLE001
        _print(f"  FAIL: engine.embed raised: {exc!r}")
        _print(traceback.format_exc())
        return 1

    batch_arr = np.asarray(batch)
    if batch_arr.shape != (2, engine.vector_size):
        return _fail(
            f"ceiling: embed returned shape {batch_arr.shape}, "
            f"expected (2, {engine.vector_size})"
        )
    if float(np.linalg.norm(batch_arr)) == 0.0:
        return _fail("ceiling: embed batch is all-zeros (broken pipeline)")

    _ok(f"embed batch PASS (shape={batch_arr.shape})")
    return 0


def main() -> int:
    floor_status = run_floor_smoke()
    if floor_status != 0:
        return floor_status

    ceiling_status = run_ceiling_smoke()

    _print("")
    if ceiling_status == CEILING_SKIPPED:
        # HONEST DISCLOSURE: floor-tier PASS, ceiling-tier SKIPPED because the
        # current Python env is missing heavy deps (torch / sentence-transformers
        # / qdrant). Skipping is honest; faking a pass would be L4. PR body
        # SMOKE: line must name 'floor-tier only (ceiling skipped: deps)'.
        _print("HONEST DISCLOSURE (per Brutal Honesty Rulebook v3.3 §1 Rule 5):")
        _print("  Floor tier PASS. Ceiling tier SKIPPED because heavy deps")
        _print("  (torch / sentence-transformers / qdrant) are not installed in")
        _print("  this Python environment. On the standard CI matrix these deps")
        _print("  ARE installed and the ceiling tier will run.")
        _print("")
        _print("SMOKE_PIPELINE PASS (floor tier; ceiling tier honestly skipped)")
        return 0
    if ceiling_status != 0:
        _print(f"SMOKE_PIPELINE FAIL: ceiling tier exit={ceiling_status}")
        return ceiling_status

    _print("")
    _print("SMOKE_PIPELINE PASS (floor + ceiling tiers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
