"""Tests for run_gate_train.py — gate persistence (Slice 10)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REFORM_GATE_CONFIG = {
    "version": 1,
    "policy": "query_reformulation_action_classifier",
    "feature_space": "attribution_pool",
    "deployment_mode": "advisory_only",
    "runtime_compatible": False,
    "gate": {
        "type": "linear_classifier",
        "threshold": 0.6,
        "features": ["query_token_count"],
        "weights": [0.5],
        "intercept": 0.1,
        "means": [4.0],
        "scales": [1.0],
    },
    "accepted": [],
    "rejected_count": 0,
    "criteria": {},
    "training_summary": {"train_rows": 10, "holdout_rows": 2},
}

_MASK_GATE_CONFIG = {
    "version": 1,
    "policy": "query_mask_linear_classifier",
    "feature_space": "engine_scope",
    "deployment_mode": "advisory_only",
    "gate": {
        "type": "linear_classifier",
        "threshold": 0.5,
        "features": ["baseline_score_margin"],
        "weights": [0.4],
        "intercept": 0.0,
        "means": [0.2],
        "scales": [0.1],
    },
    "accepted": [],
    "rejected_count": 0,
    "criteria": {},
    "training_summary": {"train_rows": 8, "holdout_rows": 2},
}


def _make_pool(
    n_query_rows: int = 15,
    n_mask_rows: int = 8,
) -> dict:
    """Build a minimal attribution pool dict."""
    query_row = {
        "query_id": "q1",
        "task": "",
        "action": "REFORMULATE",
        "delta_ndcg_at_10": 0.05,
        "fault_class": "reference",
        "profile": "reform_rrf_v2",
    }
    mask_row = {
        "query_id": "",
        "task": "",
        "delta_ndcg_at_10": 0.02,
        "split": "train",
    }
    return {
        "record_type": "attribution_pool",
        "built_at": "2026-01-01T00:00:00Z",
        "query_attribution_rows": [dict(query_row) for _ in range(n_query_rows)],
        "gate_feature_rows": [],
        "attnres_profile_rows": [],
        "mask_probe_rows": [dict(mask_row) for _ in range(n_mask_rows)],
        "summary": {
            "query_attribution_count": n_query_rows,
            "mask_probe_count": n_mask_rows,
        },
    }


def _write_pool(path: Path, pool: dict) -> None:
    path.write_text(json.dumps(pool, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMissingPoolFile(unittest.TestCase):
    """Missing pool file must produce exit code 1."""

    def test_missing_pool_file_exits_1(self) -> None:
        from run_gate_train import main

        with tempfile.TemporaryDirectory() as tmpdir:
            missing = str(Path(tmpdir) / "does_not_exist.json")
            with self.assertRaises(SystemExit) as ctx:
                sys.exit(main(["--pool", missing, "--out-dir", tmpdir]))
        self.assertEqual(ctx.exception.code, 1)

    def test_missing_pool_file_no_artifacts_written(self) -> None:
        from run_gate_train import main

        with tempfile.TemporaryDirectory() as tmpdir:
            missing = str(Path(tmpdir) / "does_not_exist.json")
            main(["--pool", missing, "--out-dir", tmpdir])
            self.assertFalse((Path(tmpdir) / "reform_gate_v1.json").exists())
            self.assertFalse((Path(tmpdir) / "mask_gate_v1.json").exists())


class TestInsufficientSamples(unittest.TestCase):
    """When pool has fewer than --min-samples rows, skip training but exit 0."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        # Pool with only 3 total rows (< default min-samples of 10)
        pool = _make_pool(n_query_rows=2, n_mask_rows=1)
        self.pool_path = self.tmpdir / "attribution_pool.json"
        _write_pool(self.pool_path, pool)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_insufficient_samples_exits_0(self) -> None:
        from run_gate_train import main

        result = main([
            "--pool", str(self.pool_path),
            "--out-dir", str(self.tmpdir),
            "--min-samples", "10",
        ])
        self.assertEqual(result, 0)

    def test_insufficient_samples_no_reform_gate_file(self) -> None:
        from run_gate_train import main

        main([
            "--pool", str(self.pool_path),
            "--out-dir", str(self.tmpdir),
            "--min-samples", "10",
        ])
        self.assertFalse((self.tmpdir / "reform_gate_v1.json").exists())

    def test_insufficient_samples_no_mask_gate_file(self) -> None:
        from run_gate_train import main

        main([
            "--pool", str(self.pool_path),
            "--out-dir", str(self.tmpdir),
            "--min-samples", "10",
        ])
        self.assertFalse((self.tmpdir / "mask_gate_v1.json").exists())

    def test_insufficient_samples_writes_summary(self) -> None:
        from run_gate_train import main

        main([
            "--pool", str(self.pool_path),
            "--out-dir", str(self.tmpdir),
            "--min-samples", "10",
        ])
        summary_path = self.tmpdir / "gate_train_summary.json"
        self.assertTrue(summary_path.exists())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["recommendation"], "insufficient_data")
        self.assertEqual(summary["reform_gate"]["outcome"], "skipped")
        self.assertEqual(summary["mask_gate"]["outcome"], "skipped")


class TestSuccessfulTraining(unittest.TestCase):
    """Full pool → both gate files written with correct schema."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        pool = _make_pool(n_query_rows=15, n_mask_rows=10)
        self.pool_path = self.tmpdir / "attribution_pool.json"
        _write_pool(self.pool_path, pool)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run_with_mocks(self) -> int:
        """Run main with gate trainers mocked to avoid engine-scope import chain."""
        from run_gate_train import main

        with (
            patch("run_gate_train.train_gate_from_pool", return_value=_REFORM_GATE_CONFIG),
            patch("run_gate_train.train_mask_gate", return_value=_MASK_GATE_CONFIG),
            patch("run_gate_train.enrich_mask_rows", side_effect=lambda rows: rows),
        ):
            return main([
                "--pool", str(self.pool_path),
                "--out-dir", str(self.tmpdir),
                "--min-samples", "10",
            ])

    def test_successful_training_exits_0(self) -> None:
        self.assertEqual(self._run_with_mocks(), 0)

    def test_reform_gate_file_written(self) -> None:
        self._run_with_mocks()
        self.assertTrue((self.tmpdir / "reform_gate_v1.json").exists())

    def test_mask_gate_file_written(self) -> None:
        self._run_with_mocks()
        self.assertTrue((self.tmpdir / "mask_gate_v1.json").exists())

    def test_reform_gate_schema_fields(self) -> None:
        self._run_with_mocks()
        data = json.loads((self.tmpdir / "reform_gate_v1.json").read_text(encoding="utf-8"))
        for field in ("trained_at", "gate_type", "n_training_samples", "model_params", "schema_version"):
            self.assertIn(field, data, f"missing field: {field}")

    def test_mask_gate_schema_fields(self) -> None:
        self._run_with_mocks()
        data = json.loads((self.tmpdir / "mask_gate_v1.json").read_text(encoding="utf-8"))
        for field in ("trained_at", "gate_type", "n_training_samples", "model_params", "schema_version"):
            self.assertIn(field, data, f"missing field: {field}")

    def test_reform_gate_type_value(self) -> None:
        self._run_with_mocks()
        data = json.loads((self.tmpdir / "reform_gate_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(data["gate_type"], "reform_gate")

    def test_mask_gate_type_value(self) -> None:
        self._run_with_mocks()
        data = json.loads((self.tmpdir / "mask_gate_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(data["gate_type"], "mask_gate")

    def test_schema_version_reform_gate(self) -> None:
        self._run_with_mocks()
        data = json.loads((self.tmpdir / "reform_gate_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], "1.0")

    def test_schema_version_mask_gate(self) -> None:
        self._run_with_mocks()
        data = json.loads((self.tmpdir / "mask_gate_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], "1.0")

    def test_n_training_samples_reform_gate(self) -> None:
        self._run_with_mocks()
        data = json.loads((self.tmpdir / "reform_gate_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(data["n_training_samples"], 15)

    def test_n_training_samples_mask_gate(self) -> None:
        self._run_with_mocks()
        data = json.loads((self.tmpdir / "mask_gate_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(data["n_training_samples"], 10)

    def test_model_params_is_dict(self) -> None:
        self._run_with_mocks()
        for fname in ("reform_gate_v1.json", "mask_gate_v1.json"):
            data = json.loads((self.tmpdir / fname).read_text(encoding="utf-8"))
            self.assertIsInstance(data["model_params"], dict, f"{fname}: model_params should be dict")


class TestSummaryFile(unittest.TestCase):
    """gate_train_summary.json must be written with correct fields."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        pool = _make_pool(n_query_rows=15, n_mask_rows=10)
        self.pool_path = self.tmpdir / "attribution_pool.json"
        _write_pool(self.pool_path, pool)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self) -> None:
        from run_gate_train import main

        with (
            patch("run_gate_train.train_gate_from_pool", return_value=_REFORM_GATE_CONFIG),
            patch("run_gate_train.train_mask_gate", return_value=_MASK_GATE_CONFIG),
            patch("run_gate_train.enrich_mask_rows", side_effect=lambda rows: rows),
        ):
            main([
                "--pool", str(self.pool_path),
                "--out-dir", str(self.tmpdir),
                "--min-samples", "10",
            ])

    def test_summary_file_written(self) -> None:
        self._run()
        self.assertTrue((self.tmpdir / "gate_train_summary.json").exists())

    def test_summary_recommendation_gates_saved(self) -> None:
        self._run()
        data = json.loads((self.tmpdir / "gate_train_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(data["recommendation"], "gates_saved")

    def test_summary_schema_fields(self) -> None:
        self._run()
        data = json.loads((self.tmpdir / "gate_train_summary.json").read_text(encoding="utf-8"))
        for field in ("trained_at", "schema_version", "n_total_samples", "reform_gate", "mask_gate", "recommendation"):
            self.assertIn(field, data, f"missing summary field: {field}")

    def test_summary_schema_version(self) -> None:
        self._run()
        data = json.loads((self.tmpdir / "gate_train_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], "1.0")

    def test_summary_n_total_samples(self) -> None:
        self._run()
        data = json.loads((self.tmpdir / "gate_train_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(data["n_total_samples"], 25)  # 15 + 10

    def test_summary_custom_out_path(self) -> None:
        """--summary-out overrides the default summary path."""
        from run_gate_train import main

        custom = self.tmpdir / "custom_summary.json"
        with (
            patch("run_gate_train.train_gate_from_pool", return_value=_REFORM_GATE_CONFIG),
            patch("run_gate_train.train_mask_gate", return_value=_MASK_GATE_CONFIG),
            patch("run_gate_train.enrich_mask_rows", side_effect=lambda rows: rows),
        ):
            main([
                "--pool", str(self.pool_path),
                "--out-dir", str(self.tmpdir),
                "--min-samples", "10",
                "--summary-out", str(custom),
            ])
        self.assertTrue(custom.exists())


class TestCLIArguments(unittest.TestCase):
    """CLI argument parsing is handled correctly."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        pool = _make_pool(n_query_rows=15, n_mask_rows=10)
        self.pool_path = self.tmpdir / "attribution_pool.json"
        _write_pool(self.pool_path, pool)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, extra_args: list[str] | None = None) -> int:
        from run_gate_train import main

        with (
            patch("run_gate_train.train_gate_from_pool", return_value=_REFORM_GATE_CONFIG),
            patch("run_gate_train.train_mask_gate", return_value=_MASK_GATE_CONFIG),
            patch("run_gate_train.enrich_mask_rows", side_effect=lambda rows: rows),
        ):
            return main([
                "--pool", str(self.pool_path),
                "--out-dir", str(self.tmpdir),
                *(extra_args or []),
            ])

    def test_default_min_samples_allows_25_rows(self) -> None:
        """Default --min-samples is 10; pool with 25 rows should train."""
        result = self._run()
        self.assertEqual(result, 0)
        self.assertTrue((self.tmpdir / "reform_gate_v1.json").exists())

    def test_min_samples_threshold_respected(self) -> None:
        """When --min-samples exceeds pool size, training is skipped."""
        result = self._run(extra_args=["--min-samples", "9999"])
        self.assertEqual(result, 0)
        self.assertFalse((self.tmpdir / "reform_gate_v1.json").exists())

    def test_out_dir_creates_subdirectory(self) -> None:
        """--out-dir creates the target directory if it doesn't exist."""
        sub = self.tmpdir / "nested" / "output"
        self._run(extra_args=["--out-dir", str(sub)])
        self.assertTrue(sub.exists())

    def test_pool_argument_is_used(self) -> None:
        """Specifying a non-existent --pool path fails with exit 1."""
        from run_gate_train import main

        result = main(["--pool", "nonexistent_pool.json", "--out-dir", str(self.tmpdir)])
        self.assertEqual(result, 1)


class TestTrainedAtTimestamp(unittest.TestCase):
    """trained_at must be a valid ISO 8601 UTC timestamp."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        pool = _make_pool(n_query_rows=15, n_mask_rows=10)
        self.pool_path = self.tmpdir / "attribution_pool.json"
        _write_pool(self.pool_path, pool)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_trained_at_format_reform_gate(self) -> None:
        from run_gate_train import main

        with (
            patch("run_gate_train.train_gate_from_pool", return_value=_REFORM_GATE_CONFIG),
            patch("run_gate_train.train_mask_gate", return_value=_MASK_GATE_CONFIG),
            patch("run_gate_train.enrich_mask_rows", side_effect=lambda rows: rows),
        ):
            main(["--pool", str(self.pool_path), "--out-dir", str(self.tmpdir)])

        data = json.loads((self.tmpdir / "reform_gate_v1.json").read_text(encoding="utf-8"))
        ts = data["trained_at"]
        # Must be parseable as ISO 8601 and end with Z
        self.assertTrue(ts.endswith("Z"), f"timestamp should end with Z: {ts}")
        self.assertRegex(ts, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


if __name__ == "__main__":
    unittest.main()
