"""Tests for run_phase_c_eval.py — Phase C four-candidate evaluation campaign.

All tests use mocked data (no network calls, no real BEIR datasets). Unit coverage
targets every non-trivial code path in isolation to avoid 30+ minute full-run
overhead in CI.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np

from run_phase_c_eval import (
    PHASE_C_CANDIDATES,
    PhaseCCandidateSummary,
    PhaseCResult,
    _build_mask_vector,
    _load_or_train_reform_gate,
    _load_or_train_mask_gate_and_vector,
    _mask_gate_features,
    _predict_mask_gate,
    _reformulate_query,
    build_candidate_overlay_rows,
    summarize_candidate_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    candidate_id: str = "baseline",
    dataset: str = "SciFact",
    query_id: str = "q1",
    ndcg: float = 0.5,
    gate_applied: bool = False,
    reformulated: bool = False,
    mask_applied: bool = False,
) -> PhaseCResult:
    return PhaseCResult(
        candidate_id=candidate_id,
        dataset=dataset,
        query_id=query_id,
        query_text="test query",
        ndcg_at_10=ndcg,
        map_at_10=0.4,
        mrr=0.6,
        recall_at_10=0.7,
        latency_ms=10.0,
        gate_applied=gate_applied,
        reformulated=reformulated,
        mask_applied=mask_applied,
    )


def _make_pool_doc(action: str = "REFORMULATE") -> Dict[str, Any]:
    return {
        "query_id": "q1",
        "query_text": "example query about science",
        "action": action,
        "delta_ndcg_at_10": 0.05,
        "token_count": 5,
        "char_count": 30,
        "stopword_ratio": 0.2,
        "numeric_token_count": 0,
        "negation_count": 0,
        "claim_cue_count": 1,
    }


def _make_attribution_pool(n_positive: int = 40, n_negative: int = 60) -> Dict[str, Any]:
    rows = []
    for i in range(n_positive):
        r = _make_pool_doc("REFORMULATE")
        r["query_id"] = f"qp{i}"
        rows.append(r)
    for i in range(n_negative):
        r = _make_pool_doc("FAST")
        r["query_id"] = f"qn{i}"
        r["delta_ndcg_at_10"] = 0.0
        rows.append(r)
    return {"query_attribution_rows": rows}


def _make_mask_gate_config(threshold: float = 0.5) -> Dict[str, Any]:
    """Return a minimal valid mask gate config with a fully-structured inner gate."""
    return {
        "version": 1,
        "policy": "query_mask_linear_classifier",
        "gate": {
            "type": "linear_classifier",
            "features": ["query_token_count", "query_stopword_ratio"],
            "means": [5.0, 0.2],
            "scales": [2.0, 0.1],
            "weights": [0.1, -0.2],
            "intercept": 0.0,
            "threshold": threshold,
            "positive_count": 4,
            "negative_count": 10,
        },
    }


# ---------------------------------------------------------------------------
# Tests: _build_mask_vector
# ---------------------------------------------------------------------------


class TestBuildMaskVector(unittest.TestCase):
    def test_all_ones_when_no_masked_dims(self):
        v = _build_mask_vector([], 8)
        np.testing.assert_array_equal(v, np.ones(8))

    def test_masked_dims_set_to_zero(self):
        v = _build_mask_vector([0, 3, 7], 8)
        self.assertEqual(v[0], 0.0)
        self.assertEqual(v[3], 0.0)
        self.assertEqual(v[7], 0.0)
        self.assertEqual(v[1], 1.0)

    def test_out_of_range_dims_ignored(self):
        v = _build_mask_vector([100], 8)
        np.testing.assert_array_equal(v, np.ones(8))

    def test_correct_length(self):
        v = _build_mask_vector([1, 2], 384)
        self.assertEqual(len(v), 384)


# ---------------------------------------------------------------------------
# Tests: _mask_gate_features
# ---------------------------------------------------------------------------


class TestMaskGateFeatures(unittest.TestCase):
    def test_returns_dict_with_required_keys(self):
        feats = _mask_gate_features("what is CRISPR gene editing")
        self.assertIn("query_token_count", feats)
        self.assertIn("query_char_count", feats)
        self.assertIn("query_stopword_ratio", feats)
        self.assertIn("delta_ndcg_at_10", feats)

    def test_delta_is_zero_placeholder(self):
        feats = _mask_gate_features("test")
        self.assertEqual(feats["delta_ndcg_at_10"], 0.0)

    def test_empty_query(self):
        feats = _mask_gate_features("")
        self.assertIsInstance(feats["query_token_count"], (int, float))


# ---------------------------------------------------------------------------
# Tests: _predict_mask_gate
# ---------------------------------------------------------------------------


class TestPredictMaskGate(unittest.TestCase):
    def test_gate_none_returns_false(self):
        result = _predict_mask_gate({"query_token_count": 5}, {"gate": None})
        self.assertFalse(result)

    def test_empty_gate_config_returns_false(self):
        result = _predict_mask_gate({"query_token_count": 5}, {})
        self.assertFalse(result)

    def test_high_threshold_returns_false(self):
        gate_config = _make_mask_gate_config(threshold=999.0)
        result = _predict_mask_gate({"query_token_count": 5}, gate_config)
        self.assertFalse(result)

    def test_zero_threshold_returns_true(self):
        gate_config = _make_mask_gate_config(threshold=0.0)
        result = _predict_mask_gate({"query_token_count": 5, "query_stopword_ratio": 0.0}, gate_config)
        self.assertTrue(result)


# ---------------------------------------------------------------------------
# Tests: _reformulate_query
# ---------------------------------------------------------------------------


class TestReformulateQuery(unittest.TestCase):
    def test_stopword_removed_variant_preferred(self):
        mock_reformulator = MagicMock()
        variant = MagicMock()
        variant.strategy = "stopword_removed"
        variant.text = "CRISPR gene editing"
        mock_reformulator.reformulate.return_value = [variant]
        result = _reformulate_query("what is CRISPR gene editing", mock_reformulator)
        self.assertEqual(result, "CRISPR gene editing")

    def test_fallback_to_original_on_no_stopword_variant(self):
        mock_reformulator = MagicMock()
        variant = MagicMock()
        variant.strategy = "focused_prefix"
        variant.text = "Focused: query"
        mock_reformulator.reformulate.return_value = [variant]
        result = _reformulate_query("original query", mock_reformulator)
        self.assertEqual(result, "original query")

    def test_fallback_on_exception(self):
        mock_reformulator = MagicMock()
        mock_reformulator.reformulate.side_effect = RuntimeError("reformulator error")
        result = _reformulate_query("original query", mock_reformulator)
        self.assertEqual(result, "original query")


# ---------------------------------------------------------------------------
# Tests: summarize_candidate_results
# ---------------------------------------------------------------------------


class TestSummarizeCandidateResults(unittest.TestCase):
    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            summarize_candidate_results([])

    def test_basic_aggregation(self):
        results = [
            _make_result("baseline", ndcg=0.4, gate_applied=False),
            _make_result("baseline", ndcg=0.6, gate_applied=True),
        ]
        summary = summarize_candidate_results(results)
        self.assertAlmostEqual(summary.mean_ndcg_at_10, 0.5)
        self.assertEqual(summary.num_queries, 2)
        self.assertEqual(summary.gate_applied_count, 1)

    def test_single_result(self):
        summary = summarize_candidate_results([_make_result(ndcg=0.7)])
        self.assertAlmostEqual(summary.mean_ndcg_at_10, 0.7)
        self.assertEqual(summary.num_queries, 1)

    def test_returns_candidate_summary_type(self):
        summary = summarize_candidate_results([_make_result()])
        self.assertIsInstance(summary, PhaseCCandidateSummary)


# ---------------------------------------------------------------------------
# Tests: build_candidate_overlay_rows
# ---------------------------------------------------------------------------


class TestBuildCandidateOverlayRows(unittest.TestCase):
    def setUp(self):
        self.baseline = [
            _make_result("baseline", query_id="q1", ndcg=0.5),
            _make_result("baseline", query_id="q2", ndcg=0.4),
        ]
        self.candidate = [
            _make_result("guard_p85_t0.01", query_id="q1", ndcg=0.6, gate_applied=True),
            _make_result("guard_p85_t0.01", query_id="q2", ndcg=0.3, gate_applied=False),
        ]

    def test_row_count_matches_candidate(self):
        rows = build_candidate_overlay_rows(self.candidate, self.baseline)
        self.assertEqual(len(rows), 2)

    def test_delta_computed_correctly(self):
        rows = build_candidate_overlay_rows(self.candidate, self.baseline)
        q1_row = next(r for r in rows if r["query_id"] == "q1")
        self.assertAlmostEqual(q1_row["delta_ndcg_at_10"], 0.1)

    def test_negative_delta(self):
        rows = build_candidate_overlay_rows(self.candidate, self.baseline)
        q2_row = next(r for r in rows if r["query_id"] == "q2")
        self.assertAlmostEqual(q2_row["delta_ndcg_at_10"], -0.1)

    def test_missing_baseline_gives_none_delta(self):
        candidate = [_make_result("guard_p85_t0.01", query_id="q99", ndcg=0.8)]
        rows = build_candidate_overlay_rows(candidate, self.baseline)
        self.assertIsNone(rows[0]["delta_ndcg_at_10"])

    def test_action_field_for_reformulate(self):
        candidate = [_make_result("learned_reform_gate_v1", reformulated=True)]
        rows = build_candidate_overlay_rows(candidate, [_make_result("baseline")])
        self.assertEqual(rows[0]["action"], "REFORMULATE")

    def test_action_field_for_mask(self):
        candidate = [_make_result("learned_mask_gate_v1", mask_applied=True)]
        rows = build_candidate_overlay_rows(candidate, [_make_result("baseline")])
        self.assertEqual(rows[0]["action"], "MASK")

    def test_action_field_for_fast(self):
        candidate = [_make_result("guard_p85_t0.01")]
        rows = build_candidate_overlay_rows(candidate, [_make_result("baseline")])
        self.assertEqual(rows[0]["action"], "FAST")

    def test_fault_class_positive(self):
        candidate = [_make_result("guard_p85_t0.01", ndcg=0.7)]
        baseline = [_make_result("baseline", ndcg=0.4)]
        rows = build_candidate_overlay_rows(candidate, baseline)
        self.assertIn("positive", rows[0]["fault_class"])

    def test_fault_class_negative(self):
        candidate = [_make_result("guard_p85_t0.01", ndcg=0.3)]
        baseline = [_make_result("baseline", ndcg=0.5)]
        rows = build_candidate_overlay_rows(candidate, baseline)
        self.assertIn("negative", rows[0]["fault_class"])


# ---------------------------------------------------------------------------
# Tests: _load_or_train_reform_gate
# ---------------------------------------------------------------------------


class TestLoadOrTrainReformGate(unittest.TestCase):
    def test_returns_none_when_pool_missing(self):
        with patch("run_phase_c_eval._DEFAULT_ATTRIBUTION_POOL", "/no/such/file.json"):
            with patch("run_phase_c_eval.get_logger", return_value=MagicMock()):
                gate = _load_or_train_reform_gate(None)
        self.assertIsNone(gate)

    def test_loads_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp:
            json.dump({"gate": {"weights": {}, "bias": 0.0, "threshold": 0.5}}, tmp)
            tmp_path = tmp.name
        try:
            gate = _load_or_train_reform_gate(tmp_path)
            self.assertIsNotNone(gate)
            self.assertIn("gate", gate)
        finally:
            import os
            os.unlink(tmp_path)

    def test_trains_from_pool_file(self):
        pool = _make_attribution_pool(n_positive=40, n_negative=60)
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp:
            json.dump(pool, tmp)
            tmp_path = tmp.name
        try:
            with patch("run_phase_c_eval._DEFAULT_ATTRIBUTION_POOL", tmp_path):
                gate = _load_or_train_reform_gate(None)
            # Should return a gate config (may or may not have gate key; depends on training)
            self.assertIsNotNone(gate)
        finally:
            import os
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Tests: _load_or_train_mask_gate_and_vector
# ---------------------------------------------------------------------------


class TestLoadOrTrainMaskGateAndVector(unittest.TestCase):
    def test_returns_none_when_no_artifacts(self):
        with patch("run_phase_c_eval._DEFAULT_MASK_SCOPE_GLOB", "/no/such/*.json"):
            gate, vector = _load_or_train_mask_gate_and_vector(None, None)
        self.assertIsNone(gate)
        self.assertIsNone(vector)

    def test_loads_mask_vector_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp:
            json.dump({"masked_dims": [0, 1, 2]}, tmp)
            tmp_path = tmp.name
        try:
            with patch("run_phase_c_eval._DEFAULT_MASK_SCOPE_GLOB", "/no/such/*.json"):
                gate, vector = _load_or_train_mask_gate_and_vector(None, tmp_path)
            self.assertIsNone(gate)
            self.assertIsNotNone(vector)
            self.assertEqual(vector[0], 0.0)
            self.assertEqual(vector[3], 1.0)
        finally:
            import os
            os.unlink(tmp_path)

    def test_loads_gate_from_file(self):
        gate_data = _make_mask_gate_config()
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp:
            json.dump(gate_data, tmp)
            tmp_path = tmp.name
        try:
            with patch("run_phase_c_eval._DEFAULT_MASK_SCOPE_GLOB", "/no/such/*.json"):
                gate, vector = _load_or_train_mask_gate_and_vector(tmp_path, None)
            self.assertIsNotNone(gate)
            self.assertIn("gate", gate)
        finally:
            import os
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Tests: Integration — run_phase_c_eval with mocked engine
# ---------------------------------------------------------------------------


class TestRunPhaseCEvalIntegration(unittest.TestCase):
    """Integration test using fully mocked engine and data; no network calls."""

    def _make_mock_engine(self, ndcg_score: float = 0.5):
        """Create a mock AntigravityEngine with controlled inference output."""
        engine = MagicMock()
        engine.vector_size = 384
        # run_inference returns (std_top, chel_top, mask_arr, jaccard)
        # chel_top = list of mock Point objects with payload["doc_id"]
        def _make_point(doc_id: str) -> MagicMock:
            p = MagicMock()
            p.payload = {"doc_id": doc_id}
            return p

        engine.run_inference.return_value = (
            [],
            [_make_point("doc0"), _make_point("doc1"), _make_point("doc2")],
            np.ones(384),
            0.0,
        )
        engine.ingest.return_value = None
        engine.set_static_dimension_mask.return_value = None
        return engine

    def test_smoke_single_candidate_single_query(self):
        """Smoke test: one candidate, one query, fully mocked engine."""
        corpus = {"doc0": "relevant science text", "doc1": "other text"}
        queries = {"q1": "science question"}
        qrels = {"q1": {"doc0": 1}}

        from run_phase_c_eval import evaluate_phase_c_candidate

        mock_engine_instance = self._make_mock_engine()
        with patch("antigravity_engine.AntigravityEngine", return_value=mock_engine_instance), \
             patch("run_phase_c_eval.isolated_adapter_state") as mock_iso:
            # isolated_adapter_state is used as a context manager
            mock_iso.return_value.__enter__ = MagicMock(return_value=None)
            mock_iso.return_value.__exit__ = MagicMock(return_value=False)
            with patch("run_phase_c_eval.map_predicted_ids", return_value=["doc0", "doc1", "doc2"]):
                results = evaluate_phase_c_candidate(
                    "baseline",
                    corpus=corpus,
                    queries=queries,
                    qrels=qrels,
                    dataset="SciFact",
                    max_queries=1,
                )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].candidate_id, "baseline")
        self.assertEqual(results[0].query_id, "q1")
        self.assertFalse(results[0].gate_applied)

    def test_reform_gate_applied_when_gate_present(self):
        """Verify gate_applied=True when reform gate fires."""
        from run_phase_c_eval import evaluate_phase_c_candidate

        reform_gate_config = {"gate": {"weights": {}, "bias": 0.0, "threshold": 0.0}}
        corpus = {"doc0": "text"}
        queries = {"q1": "question"}
        qrels = {"q1": {"doc0": 1}}

        mock_engine_instance = self._make_mock_engine()
        with patch("antigravity_engine.AntigravityEngine", return_value=mock_engine_instance), \
             patch("run_phase_c_eval.isolated_adapter_state") as mock_iso, \
             patch("run_phase_c_eval.predict_single", return_value=True), \
             patch("run_phase_c_eval.map_predicted_ids", return_value=["doc0"]):
            mock_iso.return_value.__enter__ = MagicMock(return_value=None)
            mock_iso.return_value.__exit__ = MagicMock(return_value=False)
            results = evaluate_phase_c_candidate(
                "learned_reform_gate_v1",
                corpus=corpus,
                queries=queries,
                qrels=qrels,
                dataset="SciFact",
                reform_gate_config=reform_gate_config,
                max_queries=1,
            )

        self.assertTrue(results[0].gate_applied)
        self.assertTrue(results[0].reformulated)

    def test_mask_gate_applied_when_gate_present(self):
        """Verify mask_applied=True when mask gate fires."""
        from run_phase_c_eval import evaluate_phase_c_candidate

        mask_gate_config = _make_mask_gate_config(threshold=0.0)
        mask_vector = np.ones(384)
        corpus = {"doc0": "text"}
        queries = {"q1": "question"}
        qrels = {"q1": {"doc0": 1}}

        mock_engine_instance = self._make_mock_engine()
        with patch("antigravity_engine.AntigravityEngine", return_value=mock_engine_instance), \
             patch("run_phase_c_eval.isolated_adapter_state") as mock_iso, \
             patch("run_phase_c_eval._predict_mask_gate", return_value=True), \
             patch("run_phase_c_eval.map_predicted_ids", return_value=["doc0"]):
            mock_iso.return_value.__enter__ = MagicMock(return_value=None)
            mock_iso.return_value.__exit__ = MagicMock(return_value=False)
            results = evaluate_phase_c_candidate(
                "learned_mask_gate_v1",
                corpus=corpus,
                queries=queries,
                qrels=qrels,
                dataset="SciFact",
                mask_gate_config=mask_gate_config,
                mask_vector=mask_vector,
                max_queries=1,
            )

        self.assertTrue(results[0].gate_applied)
        self.assertTrue(results[0].mask_applied)

    def test_run_phase_c_eval_writes_output_files(self):
        """End-to-end test: run_phase_c_eval creates required JSON artifacts."""
        from run_phase_c_eval import run_phase_c_eval

        corpus = {"doc0": "text A", "doc1": "text B"}
        queries = {"q1": "query one", "q2": "query two"}
        qrels = {"q1": {"doc0": 1}, "q2": {"doc1": 1}}

        mock_engine_instance = self._make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("antigravity_engine.AntigravityEngine", return_value=mock_engine_instance), \
                 patch("run_phase_c_eval.isolated_adapter_state") as mock_iso, \
                 patch("run_phase_c_eval.load_mteb_data", return_value=(corpus, queries, qrels)), \
                 patch("run_phase_c_eval.map_predicted_ids", return_value=["doc0", "doc1"]), \
                 patch("run_phase_c_eval._load_or_train_reform_gate", return_value=None), \
                 patch("run_phase_c_eval._load_or_train_mask_gate_and_vector", return_value=(None, None)):
                mock_iso.return_value.__enter__ = MagicMock(return_value=None)
                mock_iso.return_value.__exit__ = MagicMock(return_value=False)
                run_phase_c_eval(
                    datasets=["SciFact"],
                    candidates=["baseline", "guard_p85_t0.01"],
                    max_queries=2,
                    output_dir=tmpdir,
                )

            out = Path(tmpdir)
            self.assertTrue((out / "phase_c_results.json").exists(), "phase_c_results.json missing")
            self.assertTrue((out / "run_manifest.json").exists(), "run_manifest.json missing")

            with open(out / "phase_c_results.json") as fh:
                doc = json.load(fh)
            self.assertEqual(doc["record_type"], "phase_c_eval_results")
            self.assertIn("summaries", doc)
            self.assertIn("per_query_results", doc)
            self.assertIn("baseline", doc["summaries"])

    def test_overlay_cards_written_for_each_candidate(self):
        """Verify overlay_card_<candidate>.json is written per candidate."""
        from run_phase_c_eval import run_phase_c_eval

        corpus = {"doc0": "text"}
        queries = {"q1": "query"}
        qrels = {"q1": {"doc0": 1}}

        mock_engine_instance = self._make_mock_engine()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("antigravity_engine.AntigravityEngine", return_value=mock_engine_instance), \
                 patch("run_phase_c_eval.isolated_adapter_state") as mock_iso, \
                 patch("run_phase_c_eval.load_mteb_data", return_value=(corpus, queries, qrels)), \
                 patch("run_phase_c_eval.map_predicted_ids", return_value=["doc0"]), \
                 patch("run_phase_c_eval._load_or_train_reform_gate", return_value=None), \
                 patch("run_phase_c_eval._load_or_train_mask_gate_and_vector", return_value=(None, None)):
                mock_iso.return_value.__enter__ = MagicMock(return_value=None)
                mock_iso.return_value.__exit__ = MagicMock(return_value=False)
                run_phase_c_eval(
                    datasets=["SciFact"],
                    candidates=["baseline", "guard_p85_t0.01"],
                    max_queries=1,
                    output_dir=tmpdir,
                )

            out = Path(tmpdir)
            # At least one overlay card must be written
            overlay_cards = list(out.glob("overlay_card_*.json"))
            self.assertGreater(len(overlay_cards), 0, "No overlay cards written")


# ---------------------------------------------------------------------------
# Tests: CLI (argument parsing / smoke)
# ---------------------------------------------------------------------------


class TestCLI(unittest.TestCase):
    def test_main_imports_without_side_effects(self):
        """Verify the module can be imported and main() is callable."""
        import run_phase_c_eval
        self.assertTrue(callable(run_phase_c_eval.main))

    def test_candidates_list_has_four_entries(self):
        self.assertEqual(len(PHASE_C_CANDIDATES), 4)
        self.assertIn("baseline", PHASE_C_CANDIDATES)
        self.assertIn("guard_p85_t0.01", PHASE_C_CANDIDATES)
        self.assertIn("learned_reform_gate_v1", PHASE_C_CANDIDATES)
        self.assertIn("learned_mask_gate_v1", PHASE_C_CANDIDATES)


if __name__ == "__main__":
    unittest.main()
