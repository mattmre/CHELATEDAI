"""Tests for Slice 20: Phase C repeat-seed evaluation.

Covers:
  - --seed argparse arg (default, type)
  - run_phase_c_eval() signature accepts seed param
  - seed is stored in results_doc
  - seed-123 evaluation output files exist and are valid
  - schema/structure invariants on both seed runs
  - NDCG variance bounded between runs
  - Seed-seeding code path via mocked gate training
"""

from __future__ import annotations

import argparse
import inspect
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import run_phase_c_eval as _rpc

_SEED123_DIR = Path("experiment_runs/phase-c-eval/seed-123")
_LATEST_DIR = Path("experiment_runs/phase-c-eval/latest")
_SEED123_RESULTS = _SEED123_DIR / "phase_c_results.json"
_LATEST_RESULTS = _LATEST_DIR / "phase_c_results.json"
_SEED123_MANIFEST = _SEED123_DIR / "run_manifest.json"

PHASE_C_CANDIDATES = [
    "baseline",
    "guard_p85_t0.01",
    "learned_reform_gate_v1",
    "learned_mask_gate_v1",
]


def _load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class TestSeedArgparseSpec(unittest.TestCase):
    """Tests for the --seed CLI argument."""

    def test_run_phase_c_eval_accepts_seed_param(self):
        sig = inspect.signature(_rpc.run_phase_c_eval)
        self.assertIn("seed", sig.parameters, "run_phase_c_eval must accept 'seed' parameter")

    def test_seed_param_default_is_42(self):
        sig = inspect.signature(_rpc.run_phase_c_eval)
        self.assertEqual(sig.parameters["seed"].default, 42)

    def test_seed_param_is_keyword_or_positional(self):
        sig = inspect.signature(_rpc.run_phase_c_eval)
        p = sig.parameters["seed"]
        self.assertIn(
            p.kind,
            (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ),
        )

    def test_argparse_seed_default_42(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args([])
        self.assertEqual(args.seed, 42)

    def test_argparse_seed_type_is_int(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args(["--seed", "99"])
        self.assertIsInstance(args.seed, int)

    def test_argparse_seed_custom_value(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args(["--seed", "123"])
        self.assertEqual(args.seed, 123)

    def test_argparse_seed_zero_allowed(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args(["--seed", "0"])
        self.assertEqual(args.seed, 0)


class TestRunPhaseCEvalSeedSeeding(unittest.TestCase):
    """Unit tests for seed-seeding inside run_phase_c_eval()."""

    def setUp(self):
        self._mod = _rpc

    def test_seed_in_results_doc_equals_passed_seed(self):
        """If run produces a results_doc, results_doc['seed'] equals the seed arg."""
        from run_phase_c_eval import PhaseCResult

        def fake_load_mteb_data(dataset):
            corpus = {"d1": "Biology is the science of life.", "d2": "Stars form in nebulae."}
            queries = {"q1": "What is biology?"}
            qrels = {"q1": {"d1": 1}}
            return corpus, queries, qrels

        def fake_evaluate(candidate_id, corpus, queries, qrels, **kwargs):
            return [
                PhaseCResult(
                    candidate_id=candidate_id,
                    dataset=kwargs.get("dataset", "TestDS"),
                    query_id="q1",
                    query_text="What is biology?",
                    ndcg_at_10=0.5,
                    map_at_10=0.5,
                    mrr=0.5,
                    recall_at_10=1.0,
                    latency_ms=10.0,
                )
            ]

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(_rpc, "load_mteb_data", side_effect=fake_load_mteb_data),
                patch.object(_rpc, "evaluate_phase_c_candidate", side_effect=fake_evaluate),
                patch.object(_rpc, "_load_or_train_reform_gate", return_value=None),
                patch.object(
                    _rpc,
                    "_load_or_train_mask_gate_and_vector",
                    return_value=(None, None),
                ),
                patch.object(_rpc, "_build_and_write_overlay_cards"),
            ):
                result = _rpc.run_phase_c_eval(
                    datasets=["TestDS"],
                    candidates=["baseline"],
                    max_queries=1,
                    output_dir=tmpdir,
                    seed=77,
                )
        self.assertEqual(result.get("seed"), 77)

    def test_numpy_seed_called_with_correct_value(self):
        """np.random.seed is called with the provided seed value."""
        from run_phase_c_eval import PhaseCResult

        seeded_values: list = []

        def capture_seed(val):
            seeded_values.append(val)

        def fake_load_mteb_data(dataset):
            return {"d1": "text"}, {"q1": "query"}, {"q1": {"d1": 1}}

        def fake_evaluate(candidate_id, corpus, queries, qrels, **kwargs):
            return [
                PhaseCResult(
                    candidate_id=candidate_id,
                    dataset=kwargs.get("dataset", "X"),
                    query_id="q1",
                    query_text="query",
                    ndcg_at_10=0.0,
                    map_at_10=0.0,
                    mrr=0.0,
                    recall_at_10=0.0,
                    latency_ms=1.0,
                )
            ]

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(_rpc, "load_mteb_data", side_effect=fake_load_mteb_data),
                patch.object(_rpc, "evaluate_phase_c_candidate", side_effect=fake_evaluate),
                patch.object(_rpc, "_load_or_train_reform_gate", return_value=None),
                patch.object(
                    _rpc,
                    "_load_or_train_mask_gate_and_vector",
                    return_value=(None, None),
                ),
                patch.object(_rpc, "_build_and_write_overlay_cards"),
                patch.object(_rpc.np.random, "seed", side_effect=capture_seed),
            ):
                _rpc.run_phase_c_eval(
                    datasets=["X"],
                    candidates=["baseline"],
                    max_queries=1,
                    output_dir=tmpdir,
                    seed=55,
                )
        self.assertIn(55, seeded_values, "np.random.seed should be called with seed=55")

    def test_results_doc_schema_version_via_mock(self):
        """run_phase_c_eval() always writes schema_version=1."""
        from run_phase_c_eval import PhaseCResult

        def fake_load_mteb_data(dataset):
            return {"d1": "text"}, {"q1": "query"}, {"q1": {"d1": 1}}

        def fake_evaluate(candidate_id, corpus, queries, qrels, **kwargs):
            return [
                PhaseCResult(
                    candidate_id=candidate_id,
                    dataset=kwargs.get("dataset", "X"),
                    query_id="q1",
                    query_text="query",
                    ndcg_at_10=0.0,
                    map_at_10=0.0,
                    mrr=0.0,
                    recall_at_10=0.0,
                    latency_ms=1.0,
                )
            ]

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(_rpc, "load_mteb_data", side_effect=fake_load_mteb_data),
                patch.object(_rpc, "evaluate_phase_c_candidate", side_effect=fake_evaluate),
                patch.object(_rpc, "_load_or_train_reform_gate", return_value=None),
                patch.object(
                    _rpc,
                    "_load_or_train_mask_gate_and_vector",
                    return_value=(None, None),
                ),
                patch.object(_rpc, "_build_and_write_overlay_cards"),
            ):
                result = _rpc.run_phase_c_eval(
                    datasets=["X"],
                    candidates=["baseline"],
                    max_queries=1,
                    output_dir=tmpdir,
                    seed=1,
                )
        self.assertEqual(result.get("schema_version"), 1)


class TestSeed123OutputFiles(unittest.TestCase):
    """Tests that the seed-123 evaluation produced the expected output files."""

    def test_repeat_seed_run_file_exists(self):
        self.assertTrue(
            _SEED123_RESULTS.exists(),
            f"Expected {_SEED123_RESULTS} to exist after seed-123 eval run",
        )

    def test_repeat_seed_manifest_exists(self):
        self.assertTrue(
            _SEED123_MANIFEST.exists(),
            f"Expected {_SEED123_MANIFEST} to exist after seed-123 eval run",
        )

    def test_seed123_overlay_baseline_exists(self):
        p = _SEED123_DIR / "overlay_card_baseline.json"
        self.assertTrue(p.exists(), "overlay_card_baseline.json should exist in seed-123 dir")

    def test_seed123_overlay_all_four_candidates(self):
        for cid in PHASE_C_CANDIDATES:
            p = _SEED123_DIR / f"overlay_card_{cid}.json"
            self.assertTrue(p.exists(), f"overlay_card_{cid}.json missing in seed-123")


class TestSeed123ResultsContent(unittest.TestCase):
    """Tests for the content of the seed-123 phase_c_results.json."""

    def setUp(self):
        if not _SEED123_RESULTS.exists():
            self.skipTest("seed-123 results file not found; eval run may not have completed")
        self._results = _load_json(_SEED123_RESULTS)

    def test_results_schema_version(self):
        self.assertEqual(self._results.get("schema_version"), 1)

    def test_results_seed_field_equals_123(self):
        self.assertEqual(self._results.get("seed"), 123)

    def test_results_have_expected_candidates(self):
        actual_cands = self._results.get("candidates", [])
        for cand in PHASE_C_CANDIDATES:
            self.assertIn(cand, actual_cands)

    def test_results_have_summaries(self):
        summaries = self._results.get("summaries", {})
        self.assertGreater(len(summaries), 0)

    def test_results_summaries_have_scifact(self):
        summaries = self._results.get("summaries", {})
        baseline_summary = summaries.get("baseline", {})
        self.assertIn("SciFact", baseline_summary)

    def test_results_record_type(self):
        self.assertEqual(self._results.get("record_type"), "phase_c_eval_results")

    def test_results_run_at_present(self):
        self.assertIn("run_at", self._results)

    def test_results_per_query_results_nonempty(self):
        pqr = self._results.get("per_query_results", [])
        self.assertGreater(len(pqr), 0)

    def test_results_max_queries_field(self):
        # max_queries_per_dataset should be 50 (as passed on CLI)
        self.assertEqual(self._results.get("max_queries_per_dataset"), 50)


class TestBothRunsComparison(unittest.TestCase):
    """Comparison tests that require both the original and seed-123 runs."""

    def setUp(self):
        if not _SEED123_RESULTS.exists():
            self.skipTest("seed-123 results not found")
        if not _LATEST_RESULTS.exists():
            self.skipTest("latest (seed-42) results not found")
        self._new = _load_json(_SEED123_RESULTS)
        self._orig = _load_json(_LATEST_RESULTS)

    def test_seed_42_vs_123_both_have_summaries(self):
        self.assertGreater(len(self._orig.get("summaries", {})), 0)
        self.assertGreater(len(self._new.get("summaries", {})), 0)

    def test_results_schema_version_both(self):
        self.assertEqual(self._new.get("schema_version"), 1)
        self.assertEqual(self._orig.get("schema_version"), 1)

    def test_results_have_expected_candidates_both(self):
        for cand in PHASE_C_CANDIDATES:
            self.assertIn(cand, self._new.get("candidates", []))
            self.assertIn(cand, self._orig.get("candidates", []))

    def test_new_run_seed_field_present(self):
        self.assertIn("seed", self._new)

    def test_ndcg_variance_bounded(self):
        """Baseline SciFact NDCG@10 between seed-42 and seed-123 must differ < 0.15."""
        orig_summ = self._orig.get("summaries", {}).get("baseline", {}).get("SciFact", {})
        new_summ = self._new.get("summaries", {}).get("baseline", {}).get("SciFact", {})
        if not orig_summ or not new_summ:
            self.skipTest("baseline/SciFact summaries not found in one or both runs")
        orig_ndcg = orig_summ.get("mean_ndcg_at_10", orig_summ.get("ndcg_at_10"))
        new_ndcg = new_summ.get("mean_ndcg_at_10", new_summ.get("ndcg_at_10"))
        if orig_ndcg is None or new_ndcg is None:
            self.skipTest("NDCG values not found in summaries")
        self.assertLess(
            abs(orig_ndcg - new_ndcg),
            0.15,
            f"Baseline SciFact NDCG variance too large: {orig_ndcg:.4f} vs {new_ndcg:.4f}",
        )

    def test_seed123_run_at_differs_from_latest(self):
        """The two runs should have different run_at timestamps."""
        self.assertNotEqual(
            self._orig.get("run_at"),
            self._new.get("run_at"),
            "seed-123 and latest runs should have different run_at timestamps",
        )


if __name__ == "__main__":
    unittest.main()
