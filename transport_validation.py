"""
Standalone validation for REM-C1: VectorTransport target registration fix.

Proves end-to-end that:
1. register_target / target_count / clear_targets work correctly.
2. transport() fires (was_transported=True) for a vector with sim < 0.85 to nearest target.
3. transport() skips (was_transported=False) for a vector with sim >= 0.85 (already close).
4. enable_tts() on AntigravityEngine wires transport targets when transport_state_path is given.
5. add_tts_transport_target() and load_tts_transport_state() work on a live engine.
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np

# ── helpers ──────────────────────────────────────────────────────────────────

_TRANSPORT_LOGGER_PATCH = "vector_transport.get_logger"


def _make_transport(**cfg_kwargs):
    from vector_transport import TransportConfig, VectorTransport

    with patch(_TRANSPORT_LOGGER_PATCH, return_value=MagicMock()):
        return VectorTransport(TransportConfig(**cfg_kwargs))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        print(f"  FAIL: {msg}")
        sys.exit(1)
    print(f"  PASS: {msg}")


# ── Section 1: register_target / target_count / clear_targets ────────────────

def validate_registration_api() -> None:
    print("\n[1] register_target / target_count / clear_targets")
    tp = _make_transport()
    assert_true(tp.target_count() == 0, "fresh transport has 0 targets")

    centroid_a = _unit(np.array([1.0, 0.0, 0.0, 0.0]))
    centroid_b = _unit(np.array([0.0, 1.0, 0.0, 0.0]))

    tp.register_target("a", centroid_a, "cluster-A")
    assert_true(tp.target_count() == 1, "target_count == 1 after first registration")

    tp.register_target("b", centroid_b, "cluster-B")
    assert_true(tp.target_count() == 2, "target_count == 2 after second registration")

    # overwrite is idempotent
    tp.register_target("a", centroid_a, "cluster-A-updated")
    assert_true(tp.target_count() == 2, "target_count stays 2 on overwrite")

    tp.clear_targets()
    assert_true(tp.target_count() == 0, "target_count == 0 after clear_targets")


# ── Section 2: transport fires when sim < 0.85 ──────────────────────────────

def validate_transport_fires() -> None:
    print("\n[2] transport() fires for sim < 0.85 to nearest target")
    tp = _make_transport()  # default min_similarity_for_transport = 0.85

    # Build two orthogonal targets
    target_a = _unit(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    target_b = _unit(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    tp.register_target("a", target_a, "cluster-A")
    tp.register_target("b", target_b, "cluster-B")

    # A query that is somewhat similar to target_a but below 0.85
    # sim([1,0.5,...], [1,0,...]) < 1.0 due to the 0.5 component; tweak to be ~0.7
    query = _unit(np.array([1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    sim_to_a = _cosine_sim(query, target_a)
    sim_to_b = _cosine_sim(query, target_b)
    nearest_sim = max(sim_to_a, sim_to_b)

    print(f"     query sim_to_a={sim_to_a:.4f}, sim_to_b={sim_to_b:.4f}, nearest={nearest_sim:.4f}")
    assert_true(nearest_sim < 0.85, f"pre-condition: nearest_sim ({nearest_sim:.4f}) < 0.85")

    result = tp.transport(query)
    assert_true(result.was_transported, "was_transported=True when sim < 0.85")
    assert_true(result.target_id is not None, "target_id is set")
    assert_true(result.weight_used > 0.0, "weight_used > 0")

    # transported vector should be closer to target than original
    sim_after = _cosine_sim(result.transported, target_a if result.target_id == "a" else target_b)
    assert_true(sim_after > nearest_sim, "transported vector is closer to target than original")


# ── Section 3: transport skips when sim >= 0.85 ──────────────────────────────

def validate_transport_skips_close() -> None:
    print("\n[3] transport() skips (was_transported=False) when sim >= 0.85")
    tp = _make_transport()

    target = _unit(np.array([1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    tp.register_target("t", target, "near-target")

    # Vector that is very close to target (sim > 0.99)
    query = _unit(target + np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    sim = _cosine_sim(query, target)
    print(f"     sim to target = {sim:.6f}")
    assert_true(sim >= 0.85, f"pre-condition: sim ({sim:.6f}) >= 0.85")

    result = tp.transport(query)
    assert_true(not result.was_transported, "was_transported=False when already very close")
    assert_true(result.weight_used == 0.0, "weight_used==0 for skipped transport")
    np.testing.assert_array_almost_equal(result.transported, query, decimal=8)
    print("     transported vector == original (no change) ✓")


# ── Section 4: empty targets → always passthrough ───────────────────────────

def validate_empty_passthrough() -> None:
    print("\n[4] empty targets always passthrough (was_transported=False)")
    tp = _make_transport()
    query = _unit(np.random.rand(8))
    result = tp.transport(query)
    assert_true(not result.was_transported, "passthrough when no targets registered")


# ── Section 5: enable_tts() wires targets via transport_state_path ────────────

def validate_engine_enable_tts_with_state() -> None:
    print("\n[5] AntigravityEngine.enable_tts() wires targets from transport_state_path")

    state = {
        "targets": [
            {"id": "biomedical_cluster_0", "centroid": [0.1, -0.2, 0.3], "label": "biomedical"},
            {"id": "legal_cluster_1", "centroid": [-0.1, 0.4, 0.2], "label": "legal"},
        ]
    }
    state_file = "transport_validation_state.json"
    with open(state_file, "w") as fh:
        json.dump(state, fh)

    try:
        from antigravity_engine import AntigravityEngine

        patches = [
            patch("antigravity_engine.get_logger", return_value=MagicMock()),
            patch("chelation_logger.get_logger", return_value=MagicMock()),
            patch("vector_transport.get_logger", return_value=MagicMock()),
            patch("vector_translator.get_logger", return_value=MagicMock()),
            patch("tts_pipeline.get_logger", return_value=MagicMock()),
            patch("dashboard_server.update_tts_dashboard_state", return_value=None),
        ]
        for p in patches:
            p.start()

        engine = AntigravityEngine.__new__(AntigravityEngine)
        engine.logger = MagicMock()
        engine.vector_size = 3
        engine._tts_pipeline = None
        engine._last_tts_result = None

        engine.enable_tts(transport_state_path=state_file)

        transport = engine._tts_pipeline._transport
        assert_true(transport.target_count() == 2, "engine transport has 2 registered targets after enable_tts")

        for p in patches:
            p.stop()
    finally:
        if os.path.exists(state_file):
            os.remove(state_file)


# ── Section 6: add_tts_transport_target and load_tts_transport_state ─────────

def validate_engine_transport_methods() -> None:
    print("\n[6] add_tts_transport_target / load_tts_transport_state")

    patches = [
        patch("antigravity_engine.get_logger", return_value=MagicMock()),
        patch("chelation_logger.get_logger", return_value=MagicMock()),
        patch("vector_transport.get_logger", return_value=MagicMock()),
        patch("vector_translator.get_logger", return_value=MagicMock()),
        patch("tts_pipeline.get_logger", return_value=MagicMock()),
        patch("dashboard_server.update_tts_dashboard_state", return_value=None),
    ]
    for p in patches:
        p.start()

    try:
        from antigravity_engine import AntigravityEngine

        engine = AntigravityEngine.__new__(AntigravityEngine)
        engine.logger = MagicMock()
        engine.vector_size = 4
        engine._tts_pipeline = None
        engine._last_tts_result = None

        # Guard: raise before enable_tts
        try:
            engine.add_tts_transport_target("x", np.zeros(4))
            assert_true(False, "should have raised RuntimeError")
        except RuntimeError:
            assert_true(True, "RuntimeError raised when TTS not enabled (add_tts_transport_target)")

        try:
            engine.load_tts_transport_state("nonexistent.json")
            assert_true(False, "should have raised RuntimeError")
        except RuntimeError:
            assert_true(True, "RuntimeError raised when TTS not enabled (load_tts_transport_state)")

        engine.enable_tts()
        transport = engine._tts_pipeline._transport
        assert_true(transport.target_count() == 0, "no targets initially after enable_tts()")

        centroid = _unit(np.array([1.0, 0.0, 0.0, 0.0]))
        engine.add_tts_transport_target("medical", centroid, "medical-domain")
        assert_true(transport.target_count() == 1, "target_count == 1 after add_tts_transport_target")

        state = {
            "targets": [
                {"id": "legal", "centroid": [0.0, 1.0, 0.0, 0.0], "label": "legal"},
                {"id": "finance", "centroid": [0.0, 0.0, 1.0, 0.0], "label": "finance"},
            ]
        }
        state_file = "transport_validation_load_state.json"
        with open(state_file, "w") as fh:
            json.dump(state, fh)
        try:
            engine.load_tts_transport_state(state_file)
            assert_true(transport.target_count() == 3, "target_count == 3 after load_tts_transport_state")
        finally:
            if os.path.exists(state_file):
                os.remove(state_file)
    finally:
        for p in patches:
            p.stop()


# ── Section 7: build_default with transport_state_path ───────────────────────

def validate_build_default_with_state() -> None:
    print("\n[7] TTSPipeline.build_default() wires targets from transport_state_path")

    state = {
        "targets": [
            {"id": "cluster_0", "centroid": [1.0, 0.0, 0.0, 0.0], "label": "domain-0"},
        ]
    }
    state_file = "transport_validation_build_default_state.json"
    with open(state_file, "w") as fh:
        json.dump(state, fh)

    patches = [
        patch("vector_transport.get_logger", return_value=MagicMock()),
        patch("vector_translator.get_logger", return_value=MagicMock()),
        patch("tts_pipeline.get_logger", return_value=MagicMock()),
    ]
    for p in patches:
        p.start()

    try:
        from tts_pipeline import TTSPipeline

        pipeline = TTSPipeline.build_default(dim=4, transport_state_path=state_file)
        assert_true(pipeline._transport.target_count() == 1, "build_default loads 1 target from state file")

        pipeline_no_state = TTSPipeline.build_default(dim=4)
        assert_true(pipeline_no_state._transport.target_count() == 0, "build_default with no path has 0 targets")
    finally:
        for p in patches:
            p.stop()
        if os.path.exists(state_file):
            os.remove(state_file)


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("REM-C1 VectorTransport passthrough fix — validation")
    print("=" * 60)

    np.random.seed(42)

    validate_registration_api()
    validate_transport_fires()
    validate_transport_skips_close()
    validate_empty_passthrough()
    validate_engine_enable_tts_with_state()
    validate_engine_transport_methods()
    validate_build_default_with_state()

    print("\n" + "=" * 60)
    print("ALL VALIDATIONS PASSED")
    print("=" * 60)
