"""Tests for build_attribution_pool.py."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from typing import Any, Dict


def _write_json(path: pathlib.Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class TestCollectQueryAttributionRows(unittest.TestCase):
    """Unit tests for _collect_query_attribution_rows."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.art_dir = pathlib.Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _row(self, **kwargs: Any) -> Dict[str, Any]:
        defaults = {
            "loop": 1,
            "window": 1,
            "global_window": 1,
            "task": "SciFact",
            "query_id": "42",
            "profile": "baseline",
            "delta_ndcg_at_10": 0.0,
            "delta_mrr": 0.0,
            "delta_recall_at_10": 0.0,
            "rank_delta": 0,
            "top10_overlap_with_baseline": 10,
            "top_doc_changed": False,
            "action": "FAST",
            "global_variance": 0.002,
            "jaccard": 1.0,
            "mask_density": 1.0,
            "reformulation_variant_count": 0,
            "reformulation_changed": False,
            "fault_class": "reference",
        }
        defaults.update(kwargs)
        return defaults

    def test_collects_from_probe_artifact(self) -> None:
        from build_attribution_pool import _collect_query_attribution_rows

        art = {
            "query_attribution_rows": [self._row(query_id="1"), self._row(query_id="2")],
            "seed": 42,
        }
        _write_json(self.art_dir / "attribution_probe_100.json", art)
        rows = _collect_query_attribution_rows(self.art_dir)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["source_artifact"], "attribution_probe_100.json")
        self.assertEqual(rows[0]["strategy"], "attribution_probe_100")
        self.assertEqual(rows[0]["seed"], 42)

    def test_skips_single_row_artifacts(self) -> None:
        """Sentinel / research-pathway artifacts with 1 row are skipped."""
        from build_attribution_pool import _collect_query_attribution_rows

        _write_json(
            self.art_dir / "research_pathway_probe_100.json",
            {"query_attribution_rows": [self._row()]},
        )
        rows = _collect_query_attribution_rows(self.art_dir)
        self.assertEqual(rows, [])

    def test_normalises_optional_rich_fields(self) -> None:
        """Optional query-text fields default to None when absent."""
        from build_attribution_pool import _collect_query_attribution_rows

        art = {
            "query_attribution_rows": [self._row(), self._row()],
        }
        _write_json(self.art_dir / "probe.json", art)
        rows = _collect_query_attribution_rows(self.art_dir)
        self.assertIsNone(rows[0]["query_text"])
        self.assertIsNone(rows[0]["query_token_count"])

    def test_rich_schema_query_text_preserved(self) -> None:
        """query_text from reform_policy artifacts is preserved."""
        from build_attribution_pool import _collect_query_attribution_rows

        row = self._row(query_text="Hello world", query_token_count=2)
        art = {"query_attribution_rows": [row, self._row()]}
        _write_json(self.art_dir / "reform_policy.json", art)
        rows = _collect_query_attribution_rows(self.art_dir)
        self.assertEqual(rows[0]["query_text"], "Hello world")
        self.assertEqual(rows[0]["query_token_count"], 2)

    def test_skips_malformed_json(self) -> None:
        """Malformed JSON files are silently skipped."""
        from build_attribution_pool import _collect_query_attribution_rows

        (self.art_dir / "bad.json").write_text("{not valid json", encoding="utf-8")
        art = {
            "query_attribution_rows": [self._row(), self._row()],
        }
        _write_json(self.art_dir / "good.json", art)
        rows = _collect_query_attribution_rows(self.art_dir)
        self.assertEqual(len(rows), 2)


class TestCollectGateFeatureRows(unittest.TestCase):
    """Unit tests for _collect_gate_feature_rows."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.art_dir = pathlib.Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _gate_row(self, **kwargs: Any) -> Dict[str, Any]:
        defaults = {
            "loop": 1,
            "window": 1,
            "global_window": 1,
            "task": "SciFact",
            "profile": "baseline",
            "delta_vs_baseline": 0.0,
            "ndcg_at_10": 0.9,
            "fault_class": "reference",
            "promotion_blocker": False,
            "chelate_count": 0,
            "reformulate_count": 0,
            "fast_count": 50,
            "variance_mean": 0.002,
            "jaccard_mean": 1.0,
            "mask_density_mean": 1.0,
            "reformulation_changed_count": 0,
        }
        defaults.update(kwargs)
        return defaults

    def test_collects_gate_rows(self) -> None:
        from build_attribution_pool import _collect_gate_feature_rows

        art = {
            "gate_feature_rows": [self._gate_row(), self._gate_row(), self._gate_row()],
        }
        _write_json(self.art_dir / "adaptive_fivek.json", art)
        rows = _collect_gate_feature_rows(self.art_dir)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["source_artifact"], "adaptive_fivek.json")
        self.assertEqual(rows[0]["strategy"], "adaptive_fivek")

    def test_skips_single_row_gate_artifacts(self) -> None:
        from build_attribution_pool import _collect_gate_feature_rows

        art = {"gate_feature_rows": [self._gate_row()]}
        _write_json(self.art_dir / "tiny.json", art)
        rows = _collect_gate_feature_rows(self.art_dir)
        self.assertEqual(rows, [])


class TestCollectAttnresProfileRows(unittest.TestCase):
    """Unit tests for _collect_attnres_profile_rows."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.art_dir = pathlib.Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _profile_result(self, profile: str = "baseline", ndcg: float = 0.8) -> Dict[str, Any]:
        return {
            "profile": profile,
            "metrics": {
                "ndcg_at_10": ndcg,
                "map_at_10": 0.75,
                "mrr": 0.75,
                "recall_at_10": 0.95,
                "evaluated_queries": 20.0,
            },
            "action_mix": {"FAST": 20},
            "latency_ms_mean": 15.0,
        }

    def test_collects_attnres_rows(self) -> None:
        from build_attribution_pool import _collect_attnres_profile_rows

        art = {
            "task": "SciFact",
            "seed": 42,
            "corpus_size": 1200,
            "query_count": 20,
            "seed_gate": True,
            "quantization_survival": True,
            "profile_results": [
                self._profile_result("baseline", 0.82),
                self._profile_result("chelation", 0.79),
            ],
        }
        _write_json(self.art_dir / "attnres_trained_scifact_seed42.json", art)
        rows = _collect_attnres_profile_rows(self.art_dir)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["task"], "SciFact")
        self.assertEqual(rows[0]["seed"], 42)
        self.assertEqual(rows[0]["profile"], "baseline")
        self.assertAlmostEqual(rows[0]["ndcg_at_10"], 0.82, places=5)
        self.assertEqual(rows[0]["action_fast"], 20)
        self.assertTrue(rows[0]["seed_gate"])

    def test_ignores_non_attnres_files(self) -> None:
        """Files not matching attnres_*.json are ignored."""
        from build_attribution_pool import _collect_attnres_profile_rows

        art = {"task": "SciFact", "seed": 42, "profile_results": [self._profile_result()]}
        _write_json(self.art_dir / "other_artifact.json", art)
        rows = _collect_attnres_profile_rows(self.art_dir)
        self.assertEqual(rows, [])

    def test_missing_action_mix_defaults_to_zero(self) -> None:
        from build_attribution_pool import _collect_attnres_profile_rows

        pr = self._profile_result()
        pr.pop("action_mix")
        art = {
            "task": "SciFact",
            "seed": 42,
            "corpus_size": 100,
            "query_count": 20,
            "profile_results": [pr],
        }
        _write_json(self.art_dir / "attnres_nfcorpus_seed42.json", art)
        rows = _collect_attnres_profile_rows(self.art_dir)
        self.assertEqual(rows[0]["action_fast"], 0)
        self.assertEqual(rows[0]["action_chelate"], 0)


class TestCollectMaskProbeRows(unittest.TestCase):
    """Unit tests for _collect_mask_probe_rows."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.art_dir = pathlib.Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _mask_artifact(
        self,
        seed: int = 250,
        task: str = "SciFact",
        promotion_candidate: bool = False,
        train_ndcg: float = 0.89,
        holdout_ndcg: float = 0.88,
    ) -> Dict[str, Any]:
        split = {
            "baseline": {
                "ndcg_at_10": train_ndcg,
                "map_at_10": 0.85,
                "mrr": 0.85,
                "recall_at_10": 0.96,
            },
            "masked": {
                "ndcg_at_10": train_ndcg - 0.001,
                "map_at_10": 0.85,
                "mrr": 0.85,
                "recall_at_10": 0.96,
            },
            "delta_ndcg_at_10": -0.001,
        }
        return {
            "task": task,
            "seed": seed,
            "max_queries": 150,
            "mask": [1, 0, 1, 0],
            "promotion_candidate": promotion_candidate,
            "conditional_promotion_candidate": False,
            "train": [split],
            "holdout": [
                {
                    "baseline": {"ndcg_at_10": holdout_ndcg, "map_at_10": 0.84, "mrr": 0.84, "recall_at_10": 0.95},
                    "masked": {"ndcg_at_10": holdout_ndcg, "map_at_10": 0.84, "mrr": 0.84, "recall_at_10": 0.95},
                    "delta_ndcg_at_10": 0.0,
                }
            ],
            "conditional": None,
        }

    def test_collects_train_and_holdout_rows(self) -> None:
        from build_attribution_pool import _collect_mask_probe_rows

        art = self._mask_artifact(seed=250)
        _write_json(
            self.art_dir / "classifier_conditional_mask_dense_scifact_150_seed250.json",
            art,
        )
        rows = _collect_mask_probe_rows(self.art_dir)
        splits = {r["split"] for r in rows}
        self.assertIn("train", splits)
        self.assertIn("holdout", splits)
        # conditional is None → no rows
        self.assertNotIn("conditional", splits)

    def test_mask_density_computed(self) -> None:
        from build_attribution_pool import _collect_mask_probe_rows

        art = self._mask_artifact(seed=250)
        _write_json(
            self.art_dir / "classifier_conditional_mask_dense_scifact_150_seed250.json",
            art,
        )
        rows = _collect_mask_probe_rows(self.art_dir)
        # mask=[1,0,1,0] → density=0.5
        for r in rows:
            self.assertAlmostEqual(r["mask_density"], 0.5, places=5)
            self.assertEqual(r["mask_dims"], 4)

    def test_ignores_non_classifier_files(self) -> None:
        from build_attribution_pool import _collect_mask_probe_rows

        _write_json(self.art_dir / "other_probe.json", {"mask": [1, 0]})
        rows = _collect_mask_probe_rows(self.art_dir)
        self.assertEqual(rows, [])

    def test_promotion_candidate_propagated(self) -> None:
        from build_attribution_pool import _collect_mask_probe_rows

        art = self._mask_artifact(seed=251, promotion_candidate=True)
        _write_json(
            self.art_dir / "classifier_conditional_mask_dense_scifact_150_seed251.json",
            art,
        )
        rows = _collect_mask_probe_rows(self.art_dir)
        for r in rows:
            self.assertTrue(r["promotion_candidate"])


class TestBuildAttributionPool(unittest.TestCase):
    """Integration tests for build_attribution_pool()."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.art_dir = pathlib.Path(self._tmp.name) / "artifacts"
        self.out_dir = pathlib.Path(self._tmp.name) / "pool"
        self.art_dir.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_probe_artifact(self, name: str, n_rows: int = 5) -> None:
        rows = [
            {
                "loop": 1, "window": 1, "global_window": i,
                "task": "SciFact", "query_id": str(i),
                "profile": "baseline",
                "delta_ndcg_at_10": 0.0, "delta_mrr": 0.0,
                "delta_recall_at_10": 0.0, "rank_delta": 0,
                "top10_overlap_with_baseline": 10, "top_doc_changed": False,
                "action": "FAST", "global_variance": 0.002,
                "jaccard": 1.0, "mask_density": 1.0,
                "reformulation_variant_count": 0,
                "reformulation_changed": False, "fault_class": "reference",
            }
            for i in range(n_rows)
        ]
        _write_json(self.art_dir / name, {"query_attribution_rows": rows})

    def _write_attnres_artifact(self) -> None:
        art = {
            "task": "SciFact", "seed": 42,
            "corpus_size": 1200, "query_count": 20,
            "seed_gate": True, "quantization_survival": False,
            "profile_results": [
                {
                    "profile": "baseline",
                    "metrics": {"ndcg_at_10": 0.82, "map_at_10": 0.78, "mrr": 0.78, "recall_at_10": 0.95, "evaluated_queries": 20.0},
                    "action_mix": {"FAST": 20},
                    "latency_ms_mean": 15.0,
                }
            ],
        }
        _write_json(self.art_dir / "attnres_trained_scifact_seed42.json", art)

    def test_pool_written_to_disk(self) -> None:
        from build_attribution_pool import build_attribution_pool

        self._write_probe_artifact("probe_a.json", n_rows=3)
        self._write_probe_artifact("probe_b.json", n_rows=4)
        self._write_attnres_artifact()

        pool = build_attribution_pool(self.art_dir, self.out_dir)
        output_path = self.out_dir / "attribution_pool.json"
        self.assertTrue(output_path.exists())
        on_disk = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["record_type"], "attribution_pool")
        self.assertEqual(
            on_disk["summary"]["query_attribution_count"],
            pool["summary"]["query_attribution_count"],
        )

    def test_summary_counts_correct(self) -> None:
        from build_attribution_pool import build_attribution_pool

        self._write_probe_artifact("probe_a.json", n_rows=5)
        self._write_probe_artifact("probe_b.json", n_rows=7)
        self._write_attnres_artifact()

        pool = build_attribution_pool(self.art_dir, self.out_dir)
        self.assertEqual(pool["summary"]["query_attribution_count"], 12)
        self.assertEqual(pool["summary"]["attnres_profile_count"], 1)

    def test_output_dir_created_if_missing(self) -> None:
        from build_attribution_pool import build_attribution_pool

        deep_dir = self.out_dir / "sub" / "dir"
        self.assertFalse(deep_dir.exists())
        self._write_probe_artifact("probe.json", n_rows=2)
        build_attribution_pool(self.art_dir, deep_dir)
        self.assertTrue((deep_dir / "attribution_pool.json").exists())

    def test_pool_record_type(self) -> None:
        from build_attribution_pool import build_attribution_pool

        self._write_probe_artifact("probe.json", n_rows=2)
        pool = build_attribution_pool(self.art_dir, self.out_dir)
        self.assertEqual(pool["record_type"], "attribution_pool")

    def test_pool_contains_built_at_timestamp(self) -> None:
        from build_attribution_pool import build_attribution_pool

        self._write_probe_artifact("probe.json", n_rows=2)
        pool = build_attribution_pool(self.art_dir, self.out_dir)
        self.assertIn("built_at", pool)
        self.assertRegex(pool["built_at"], r"\d{4}-\d{2}-\d{2}T")

    def test_empty_artifact_dir_produces_empty_pool(self) -> None:
        from build_attribution_pool import build_attribution_pool

        pool = build_attribution_pool(self.art_dir, self.out_dir)
        self.assertEqual(pool["summary"]["query_attribution_count"], 0)
        self.assertEqual(pool["summary"]["gate_feature_count"], 0)
        self.assertEqual(pool["summary"]["attnres_profile_count"], 0)
        self.assertEqual(pool["summary"]["mask_probe_count"], 0)

    def test_tasks_in_summary(self) -> None:
        from build_attribution_pool import build_attribution_pool

        self._write_probe_artifact("probe.json", n_rows=3)
        pool = build_attribution_pool(self.art_dir, self.out_dir)
        self.assertIn("SciFact", pool["summary"]["tasks"])

    def test_fault_classes_in_summary(self) -> None:
        from build_attribution_pool import build_attribution_pool

        self._write_probe_artifact("probe.json", n_rows=3)
        pool = build_attribution_pool(self.art_dir, self.out_dir)
        self.assertIn("reference", pool["summary"]["fault_classes"])


if __name__ == "__main__":
    unittest.main()
