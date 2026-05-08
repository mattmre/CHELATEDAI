"""Tests for analyze_phase_c_results.py — Phase C results analysis.

All tests use mocked or in-memory data (no real BEIR datasets, no network
calls, no filesystem side-effects except via tempfile).  Uses ``unittest``
only — NO pytest imports.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from analyze_phase_c_results import (
    ANALYSIS_SCHEMA_VERSION,
    build_analysis_report,
    compute_candidate_summary,
    compute_per_query_deltas,
    identify_promotable_candidates,
    load_phase_c_results,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_query_row(
    candidate_id: str = "baseline",
    dataset: str = "SciFact",
    query_id: str = "q1",
    ndcg: float = 0.5,
) -> Dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "dataset": dataset,
        "query_id": query_id,
        "query_text": "test query",
        "ndcg_at_10": ndcg,
        "map_at_10": 0.4,
        "mrr": 0.6,
        "recall_at_10": 0.7,
        "latency_ms": 10.0,
        "gate_applied": False,
        "reformulated": False,
        "mask_applied": False,
    }


def _make_results_doc(
    candidates: List[str] = None,
    datasets: List[str] = None,
    per_query_rows: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if candidates is None:
        candidates = ["baseline", "guard_p85_t0.01"]
    if datasets is None:
        datasets = ["SciFact"]
    if per_query_rows is None:
        per_query_rows = [
            _make_query_row("baseline", "SciFact", "q1", 0.5),
            _make_query_row("guard_p85_t0.01", "SciFact", "q1", 0.55),
        ]
    return {
        "schema_version": 1,
        "record_type": "phase_c_eval_results",
        "run_at": "2026-01-01T00:00:00Z",
        "candidates": candidates,
        "datasets": datasets,
        "max_queries_per_dataset": None,
        "gate_summary": {},
        "summaries": {},
        "per_query_results": per_query_rows,
    }


def _write_results_to_file(
    tmp_dir: str,
    doc: Dict[str, Any],
    filename: str = "phase_c_results.json",
) -> str:
    path = str(Path(tmp_dir) / filename)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    return path


# ---------------------------------------------------------------------------
# Tests: load_phase_c_results
# ---------------------------------------------------------------------------


class TestLoadPhaseCResults(unittest.TestCase):

    def test_load_valid_file(self):
        doc = _make_results_doc()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_results_to_file(tmp, doc)
            result = load_phase_c_results(path)
        self.assertEqual(result["candidates"], ["baseline", "guard_p85_t0.01"])
        self.assertEqual(result["datasets"], ["SciFact"])

    def test_file_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_phase_c_results("nonexistent_path/phase_c_results.json")

    def test_missing_candidates_key_raises(self):
        doc = _make_results_doc()
        del doc["candidates"]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_results_to_file(tmp, doc)
            with self.assertRaises(ValueError) as ctx:
                load_phase_c_results(path)
        self.assertIn("candidates", str(ctx.exception))

    def test_missing_datasets_key_raises(self):
        doc = _make_results_doc()
        del doc["datasets"]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_results_to_file(tmp, doc)
            with self.assertRaises(ValueError):
                load_phase_c_results(path)

    def test_missing_per_query_results_raises(self):
        doc = _make_results_doc()
        del doc["per_query_results"]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_results_to_file(tmp, doc)
            with self.assertRaises(ValueError):
                load_phase_c_results(path)

    def test_per_query_results_wrong_type_raises(self):
        doc = _make_results_doc()
        doc["per_query_results"] = "not-a-list"
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_results_to_file(tmp, doc)
            with self.assertRaises(ValueError):
                load_phase_c_results(path)

    def test_candidates_wrong_type_raises(self):
        doc = _make_results_doc()
        doc["candidates"] = "not-a-list"
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_results_to_file(tmp, doc)
            with self.assertRaises(ValueError):
                load_phase_c_results(path)

    def test_datasets_wrong_type_raises(self):
        doc = _make_results_doc()
        doc["datasets"] = 42
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_results_to_file(tmp, doc)
            with self.assertRaises(ValueError):
                load_phase_c_results(path)

    def test_empty_per_query_results_is_valid(self):
        doc = _make_results_doc(per_query_rows=[])
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_results_to_file(tmp, doc)
            result = load_phase_c_results(path)
        self.assertEqual(result["per_query_results"], [])


# ---------------------------------------------------------------------------
# Tests: compute_per_query_deltas
# ---------------------------------------------------------------------------


class TestComputePerQueryDeltas(unittest.TestCase):

    def test_basic_delta_computation(self):
        rows = [
            _make_query_row("baseline", "SciFact", "q1", 0.5),
            _make_query_row("guard_p85_t0.01", "SciFact", "q1", 0.6),
        ]
        results = _make_results_doc(per_query_rows=rows)
        deltas = compute_per_query_deltas(results)
        guard_rows = deltas["guard_p85_t0.01"]["SciFact"]
        self.assertEqual(len(guard_rows), 1)
        self.assertAlmostEqual(guard_rows[0]["delta_ndcg_at_10"], 0.1, places=6)

    def test_baseline_excluded_from_deltas(self):
        rows = [
            _make_query_row("baseline", "SciFact", "q1", 0.5),
            _make_query_row("guard_p85_t0.01", "SciFact", "q1", 0.55),
        ]
        results = _make_results_doc(per_query_rows=rows)
        deltas = compute_per_query_deltas(results)
        self.assertNotIn("baseline", deltas)

    def test_missing_baseline_gives_none_delta(self):
        rows = [
            _make_query_row("guard_p85_t0.01", "SciFact", "q_no_baseline", 0.6),
        ]
        results = _make_results_doc(per_query_rows=rows)
        deltas = compute_per_query_deltas(results)
        self.assertIsNone(
            deltas["guard_p85_t0.01"]["SciFact"][0]["delta_ndcg_at_10"]
        )

    def test_negative_delta(self):
        rows = [
            _make_query_row("baseline", "SciFact", "q1", 0.8),
            _make_query_row("guard_p85_t0.01", "SciFact", "q1", 0.5),
        ]
        results = _make_results_doc(per_query_rows=rows)
        deltas = compute_per_query_deltas(results)
        self.assertAlmostEqual(
            deltas["guard_p85_t0.01"]["SciFact"][0]["delta_ndcg_at_10"],
            -0.3,
            places=6,
        )

    def test_empty_per_query_results(self):
        results = _make_results_doc(per_query_rows=[])
        deltas = compute_per_query_deltas(results)
        self.assertEqual(deltas, {})

    def test_multiple_datasets(self):
        rows = [
            _make_query_row("baseline", "SciFact", "q1", 0.5),
            _make_query_row("baseline", "NFCorpus", "q2", 0.4),
            _make_query_row("guard_p85_t0.01", "SciFact", "q1", 0.6),
            _make_query_row("guard_p85_t0.01", "NFCorpus", "q2", 0.45),
        ]
        results = _make_results_doc(per_query_rows=rows)
        deltas = compute_per_query_deltas(results)
        self.assertIn("SciFact", deltas["guard_p85_t0.01"])
        self.assertIn("NFCorpus", deltas["guard_p85_t0.01"])
        self.assertAlmostEqual(
            deltas["guard_p85_t0.01"]["NFCorpus"][0]["delta_ndcg_at_10"],
            0.05,
            places=6,
        )

    def test_custom_baseline_candidate(self):
        rows = [
            _make_query_row("guard_p85_t0.01", "SciFact", "q1", 0.5),
            _make_query_row("learned_reform_gate_v1", "SciFact", "q1", 0.6),
        ]
        results = _make_results_doc(per_query_rows=rows)
        deltas = compute_per_query_deltas(results, baseline_candidate="guard_p85_t0.01")
        self.assertIn("learned_reform_gate_v1", deltas)
        self.assertNotIn("guard_p85_t0.01", deltas)


# ---------------------------------------------------------------------------
# Tests: compute_candidate_summary
# ---------------------------------------------------------------------------


class TestComputeCandidateSummary(unittest.TestCase):

    def test_baseline_has_zero_delta(self):
        rows = [
            _make_query_row("baseline", "SciFact", "q1", 0.5),
            _make_query_row("guard_p85_t0.01", "SciFact", "q1", 0.6),
        ]
        results = _make_results_doc(per_query_rows=rows)
        summaries = compute_candidate_summary(results)
        self.assertEqual(summaries["baseline"]["mean_delta_ndcg_at_10"], 0.0)

    def test_win_count_incremented_correctly(self):
        rows = [
            _make_query_row("baseline", "SciFact", "q1", 0.5),
            _make_query_row("baseline", "SciFact", "q2", 0.5),
            _make_query_row("guard_p85_t0.01", "SciFact", "q1", 0.6),  # win
            _make_query_row("guard_p85_t0.01", "SciFact", "q2", 0.4),  # loss
        ]
        results = _make_results_doc(per_query_rows=rows)
        summaries = compute_candidate_summary(results)
        s = summaries["guard_p85_t0.01"]
        self.assertEqual(s["win_count"], 1)
        self.assertEqual(s["loss_count"], 1)
        self.assertEqual(s["tie_count"], 0)

    def test_win_rate_calculation(self):
        rows = [
            _make_query_row("baseline", "SciFact", "q1", 0.5),
            _make_query_row("baseline", "SciFact", "q2", 0.5),
            _make_query_row("baseline", "SciFact", "q3", 0.5),
            _make_query_row("baseline", "SciFact", "q4", 0.5),
            _make_query_row("guard_p85_t0.01", "SciFact", "q1", 0.6),  # win
            _make_query_row("guard_p85_t0.01", "SciFact", "q2", 0.6),  # win
            _make_query_row("guard_p85_t0.01", "SciFact", "q3", 0.6),  # win
            _make_query_row("guard_p85_t0.01", "SciFact", "q4", 0.4),  # loss
        ]
        results = _make_results_doc(per_query_rows=rows)
        summaries = compute_candidate_summary(results)
        self.assertAlmostEqual(summaries["guard_p85_t0.01"]["win_rate"], 0.75, places=6)

    def test_all_ties_when_no_baseline(self):
        rows = [
            _make_query_row("guard_p85_t0.01", "SciFact", "q1", 0.6),
            _make_query_row("guard_p85_t0.01", "SciFact", "q2", 0.7),
        ]
        results = _make_results_doc(per_query_rows=rows)
        summaries = compute_candidate_summary(results)
        s = summaries.get("guard_p85_t0.01", {})
        self.assertEqual(s.get("win_count", 0), 0)
        self.assertEqual(s.get("loss_count", 0), 0)
        self.assertEqual(s.get("win_rate", 0.0), 0.0)

    def test_mean_ndcg_at_10_absolute(self):
        rows = [
            _make_query_row("baseline", "SciFact", "q1", 0.4),
            _make_query_row("baseline", "SciFact", "q2", 0.6),
        ]
        results = _make_results_doc(per_query_rows=rows)
        summaries = compute_candidate_summary(results)
        self.assertAlmostEqual(summaries["baseline"]["mean_ndcg_at_10"], 0.5, places=6)

    def test_datasets_field_populated(self):
        rows = [
            _make_query_row("baseline", "SciFact", "q1", 0.5),
            _make_query_row("guard_p85_t0.01", "SciFact", "q1", 0.6),
        ]
        results = _make_results_doc(per_query_rows=rows)
        summaries = compute_candidate_summary(results)
        self.assertIn("SciFact", summaries["guard_p85_t0.01"]["datasets"])

    def test_empty_results_gives_empty_summaries(self):
        results = _make_results_doc(per_query_rows=[])
        summaries = compute_candidate_summary(results)
        self.assertNotIn("baseline", summaries)
        self.assertNotIn("guard_p85_t0.01", summaries)


# ---------------------------------------------------------------------------
# Tests: identify_promotable_candidates
# ---------------------------------------------------------------------------


class TestIdentifyPromotableCandidates(unittest.TestCase):

    def _make_summary(
        self,
        win_rate: float = 0.6,
        mean_delta: float = 0.01,
    ) -> Dict[str, Any]:
        return {
            "win_rate": win_rate,
            "mean_delta_ndcg_at_10": mean_delta,
            "win_count": 6,
            "loss_count": 4,
            "tie_count": 0,
            "mean_ndcg_at_10": 0.6,
            "datasets": ["SciFact"],
        }

    def test_candidate_above_both_thresholds_is_promotable(self):
        summaries = {
            "baseline": self._make_summary(0.0, 0.0),
            "guard_p85_t0.01": self._make_summary(0.6, 0.01),
        }
        promotable = identify_promotable_candidates(summaries)
        self.assertIn("guard_p85_t0.01", promotable)

    def test_baseline_never_promotable(self):
        summaries = {
            "baseline": self._make_summary(0.9, 0.1),
        }
        promotable = identify_promotable_candidates(summaries)
        self.assertEqual(promotable, [])

    def test_candidate_below_win_rate_threshold_not_promotable(self):
        summaries = {
            "baseline": self._make_summary(0.0, 0.0),
            "guard_p85_t0.01": self._make_summary(0.5, 0.02),  # 0.5 <= 0.52
        }
        promotable = identify_promotable_candidates(summaries)
        self.assertNotIn("guard_p85_t0.01", promotable)

    def test_candidate_below_delta_threshold_not_promotable(self):
        summaries = {
            "baseline": self._make_summary(0.0, 0.0),
            "guard_p85_t0.01": self._make_summary(0.6, 0.004),  # 0.004 <= 0.005
        }
        promotable = identify_promotable_candidates(summaries)
        self.assertNotIn("guard_p85_t0.01", promotable)

    def test_exact_threshold_not_promoted(self):
        # strict >, not >=
        summaries = {
            "baseline": self._make_summary(0.0, 0.0),
            "guard_p85_t0.01": self._make_summary(0.52, 0.005),
        }
        promotable = identify_promotable_candidates(summaries)
        self.assertNotIn("guard_p85_t0.01", promotable)

    def test_empty_summaries_returns_empty(self):
        self.assertEqual(identify_promotable_candidates({}), [])

    def test_custom_thresholds(self):
        summaries = {
            "baseline": self._make_summary(0.0, 0.0),
            "guard_p85_t0.01": self._make_summary(0.55, 0.003),
        }
        # With relaxed thresholds, guard should qualify
        promotable = identify_promotable_candidates(
            summaries, min_win_rate=0.50, min_mean_delta=0.002
        )
        self.assertIn("guard_p85_t0.01", promotable)

    def test_multiple_promotable_candidates(self):
        summaries = {
            "baseline": self._make_summary(0.0, 0.0),
            "guard_p85_t0.01": self._make_summary(0.6, 0.01),
            "learned_reform_gate_v1": self._make_summary(0.7, 0.02),
        }
        promotable = identify_promotable_candidates(summaries)
        self.assertEqual(len(promotable), 2)


# ---------------------------------------------------------------------------
# Tests: build_analysis_report
# ---------------------------------------------------------------------------


class TestBuildAnalysisReport(unittest.TestCase):

    def _write_and_run(
        self, tmp: str, rows: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        doc = _make_results_doc(per_query_rows=rows)
        results_path = _write_results_to_file(tmp, doc)
        output_path = str(Path(tmp) / "phase_c_analysis.json")
        return build_analysis_report(results_path, output_path)

    def test_report_has_required_top_level_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self._write_and_run(tmp)
        for key in [
            "schema_version",
            "analysis_timestamp",
            "results_path",
            "candidate_summaries",
            "promotable_candidates",
            "recommendation",
            "baseline_mean_ndcg_at_10",
        ]:
            self.assertIn(key, report, f"Missing key: {key}")

    def test_schema_version_correct(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self._write_and_run(tmp)
        self.assertEqual(report["schema_version"], ANALYSIS_SCHEMA_VERSION)

    def test_recommendation_no_default_change(self):
        # guard wins only 1 of 2 queries → win_rate 0.5 → below threshold
        rows = [
            _make_query_row("baseline", "SciFact", "q1", 0.5),
            _make_query_row("baseline", "SciFact", "q2", 0.5),
            _make_query_row("guard_p85_t0.01", "SciFact", "q1", 0.51),  # tiny win
            _make_query_row("guard_p85_t0.01", "SciFact", "q2", 0.49),  # tiny loss
        ]
        with tempfile.TemporaryDirectory() as tmp:
            report = self._write_and_run(tmp, rows)
        self.assertEqual(report["recommendation"], "no_default_change")

    def test_recommendation_review_required(self):
        # Many wins above threshold
        rows = []
        for i in range(10):
            qid = f"q{i}"
            rows.append(_make_query_row("baseline", "SciFact", qid, 0.5))
            rows.append(_make_query_row("guard_p85_t0.01", "SciFact", qid, 0.52))
        with tempfile.TemporaryDirectory() as tmp:
            report = self._write_and_run(tmp, rows)
        self.assertEqual(report["recommendation"], "review_required")
        self.assertIn("guard_p85_t0.01", report["promotable_candidates"])

    def test_output_file_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = _make_results_doc()
            results_path = _write_results_to_file(tmp, doc)
            output_path = str(Path(tmp) / "analysis.json")
            build_analysis_report(results_path, output_path)
            self.assertTrue(Path(output_path).exists())

    def test_output_file_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = _make_results_doc()
            results_path = _write_results_to_file(tmp, doc)
            output_path = str(Path(tmp) / "analysis.json")
            build_analysis_report(results_path, output_path)
            with open(output_path, encoding="utf-8") as fh:
                loaded = json.load(fh)
        self.assertIn("schema_version", loaded)

    def test_missing_results_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            build_analysis_report("no_such_file.json", "out.json")

    def test_output_dir_created_if_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = _make_results_doc()
            results_path = _write_results_to_file(tmp, doc)
            nested_output = str(Path(tmp) / "nested" / "deep" / "analysis.json")
            build_analysis_report(results_path, nested_output)
            self.assertTrue(Path(nested_output).exists())

    def test_baseline_mean_ndcg_scalar(self):
        rows = [
            _make_query_row("baseline", "SciFact", "q1", 0.4),
            _make_query_row("baseline", "SciFact", "q2", 0.6),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            report = self._write_and_run(tmp, rows)
        self.assertAlmostEqual(report["baseline_mean_ndcg_at_10"], 0.5, places=6)


# ---------------------------------------------------------------------------
# Tests: CLI (main function)
# ---------------------------------------------------------------------------


class TestMainCLI(unittest.TestCase):

    def test_main_missing_results_returns_nonzero(self):
        ret = main(["--results", "does_not_exist.json", "--output", "out.json"])
        self.assertEqual(ret, 1)

    def test_main_success_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = _make_results_doc()
            results_path = _write_results_to_file(tmp, doc)
            output_path = str(Path(tmp) / "analysis.json")
            ret = main(["--results", results_path, "--output", output_path])
        self.assertEqual(ret, 0)

    def test_main_writes_output_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = _make_results_doc()
            results_path = _write_results_to_file(tmp, doc)
            output_path = str(Path(tmp) / "analysis.json")
            main(["--results", results_path, "--output", output_path])
            self.assertTrue(Path(output_path).exists())

    def test_main_subprocess_exit_zero(self):
        """Verify the script exits 0 when run via subprocess."""
        with tempfile.TemporaryDirectory() as tmp:
            doc = _make_results_doc()
            results_path = _write_results_to_file(tmp, doc)
            output_path = str(Path(tmp) / "analysis.json")
            proc = subprocess.run(
                [
                    sys.executable,
                    "analyze_phase_c_results.py",
                    "--results",
                    results_path,
                    "--output",
                    output_path,
                ],
                capture_output=True,
                text=True,
            )
        self.assertEqual(proc.returncode, 0)

    def test_main_subprocess_exit_one_on_missing_file(self):
        proc = subprocess.run(
            [
                sys.executable,
                "analyze_phase_c_results.py",
                "--results",
                "nonexistent.json",
                "--output",
                "out.json",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("ERROR", proc.stderr)

    def test_main_prints_summary_table(self, *_):
        with tempfile.TemporaryDirectory() as tmp:
            doc = _make_results_doc()
            results_path = _write_results_to_file(tmp, doc)
            output_path = str(Path(tmp) / "analysis.json")
            proc = subprocess.run(
                [
                    sys.executable,
                    "analyze_phase_c_results.py",
                    "--results",
                    results_path,
                    "--output",
                    output_path,
                ],
                capture_output=True,
                text=True,
            )
        self.assertIn("Phase C Analysis", proc.stdout)
        self.assertIn("Recommendation", proc.stdout)


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases(unittest.TestCase):

    def test_all_ties_no_baseline_present(self):
        """Only non-baseline candidates — all deltas are None (tie)."""
        rows = [
            _make_query_row("guard_p85_t0.01", "SciFact", "q1", 0.6),
            _make_query_row("guard_p85_t0.01", "SciFact", "q2", 0.7),
        ]
        results = _make_results_doc(per_query_rows=rows)
        summaries = compute_candidate_summary(results)
        s = summaries.get("guard_p85_t0.01", {})
        self.assertEqual(s.get("win_count", 0), 0)
        self.assertEqual(s.get("win_rate", 0.0), 0.0)

    def test_promotable_list_is_empty_when_no_winner(self):
        rows = [
            _make_query_row("baseline", "SciFact", "q1", 0.5),
            _make_query_row("guard_p85_t0.01", "SciFact", "q1", 0.5),  # tie
        ]
        results = _make_results_doc(per_query_rows=rows)
        summaries = compute_candidate_summary(results)
        promotable = identify_promotable_candidates(summaries)
        self.assertEqual(promotable, [])

    def test_results_path_stored_in_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = _make_results_doc()
            results_path = _write_results_to_file(tmp, doc)
            output_path = str(Path(tmp) / "analysis.json")
            report = build_analysis_report(results_path, output_path)
        self.assertIn(results_path, report["results_path"])

    def test_analysis_timestamp_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = _make_results_doc()
            results_path = _write_results_to_file(tmp, doc)
            output_path = str(Path(tmp) / "analysis.json")
            report = build_analysis_report(results_path, output_path)
        # Should be ISO 8601 ending in Z
        ts = report["analysis_timestamp"]
        self.assertTrue(ts.endswith("Z"), f"Unexpected timestamp format: {ts}")

    def test_zero_total_queries_win_rate_is_zero(self):
        """Edge case: candidate in candidates list but no per_query_results rows."""
        summaries = {
            "guard_p85_t0.01": {
                "win_rate": 0.0,
                "mean_delta_ndcg_at_10": 0.0,
                "mean_ndcg_at_10": 0.0,
                "datasets": [],
                "win_count": 0,
                "loss_count": 0,
                "tie_count": 0,
            }
        }
        promotable = identify_promotable_candidates(summaries)
        self.assertEqual(promotable, [])

    def test_four_candidates_full_flow(self):
        """Smoke test matching the real four-candidate schema."""
        candidates = [
            "baseline",
            "guard_p85_t0.01",
            "learned_reform_gate_v1",
            "learned_mask_gate_v1",
        ]
        rows = []
        for i in range(6):
            qid = f"q{i}"
            rows.append(_make_query_row("baseline", "SciFact", qid, 0.5))
            rows.append(_make_query_row("guard_p85_t0.01", "SciFact", qid, 0.52))
            rows.append(_make_query_row("learned_reform_gate_v1", "SciFact", qid, 0.53))
            rows.append(_make_query_row("learned_mask_gate_v1", "SciFact", qid, 0.48))
        doc = _make_results_doc(candidates=candidates, per_query_rows=rows)
        with tempfile.TemporaryDirectory() as tmp:
            results_path = _write_results_to_file(tmp, doc)
            output_path = str(Path(tmp) / "analysis.json")
            report = build_analysis_report(results_path, output_path)
        summaries = report["candidate_summaries"]
        self.assertIn("baseline", summaries)
        self.assertIn("guard_p85_t0.01", summaries)
        self.assertIn("learned_reform_gate_v1", summaries)
        self.assertIn("learned_mask_gate_v1", summaries)
        # mask gate loses consistently — should not be promotable
        self.assertNotIn("learned_mask_gate_v1", report["promotable_candidates"])


if __name__ == "__main__":
    unittest.main()
