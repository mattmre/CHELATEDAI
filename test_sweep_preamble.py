"""Sweep baseline preamble. Fake engine only: no SciFact download and no grid."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
import uuid
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch.nn as nn

from config import ChelationConfig
from run_large_sweep import run_large_parameter_sweep
from run_sweep import prepare_sweep_baseline, run_parameter_sweep
from sweep_corpus_restore import snapshot_collection


CORPUS = {"10": "alpha", "b": "beta"}
QUERIES = {"q": "find alpha"}
QRELS = {"q": {"10": 1}}


class _Qdrant:
    def __init__(self, points_count):
        self.points_count = points_count
        self.upserts = []
        self._points = []
        self.scrolls = 0

    def get_collection(self, name):
        return type("Info", (), {"points_count": self.points_count})()

    def upsert(self, collection_name, points):
        self.upserts.append(list(points))
        self._points.extend(points)
        self.points_count = len(self._points)

    def scroll(self, collection_name, limit, with_vectors, with_payload, offset):
        self.scrolls += 1
        if offset is not None:
            return [], None
        return list(self._points), None


class _Backend:
    def __init__(self, embed_raw):
        self._embed_raw = embed_raw
        self.calls = []

    def embed_raw(self, texts):
        copied = list(texts)
        self.calls.append(copied)
        return self._embed_raw(copied)


class _Engine:
    def __init__(self, points_count, embed_raw):
        self.qdrant = _Qdrant(points_count)
        self.collection_name = "docs"
        self.vector_size = 8
        self.adapter = object()
        self.loaded_adapter = self.adapter
        self.embedding_backend = _Backend(embed_raw)
        self.embed = MagicMock(side_effect=AssertionError("ingest must not apply the adapter"))
        self.chelation_log = MagicMock()
        self.run_sedimentation_cycle = MagicMock()


def _vectors(texts):
    return [[float(index)] * 8 for index, _text in enumerate(texts)]


class TestSweepPreamble(unittest.TestCase):
    def setUp(self):
        self._cwd = os.getcwd()
        self._config = {
            "NOISE_INJECTION_ENABLED": ChelationConfig.NOISE_INJECTION_ENABLED,
            "NOISE_INJECTION_BASE_SCALE": ChelationConfig.NOISE_INJECTION_BASE_SCALE,
            "HOMEOSTATIC_PUSH_MAGNITUDE": ChelationConfig.HOMEOSTATIC_PUSH_MAGNITUDE,
        }

    def tearDown(self):
        os.chdir(self._cwd)
        for name, value in self._config.items():
            setattr(ChelationConfig, name, value)

    @contextmanager
    def _cwd_and_weights(self):
        with tempfile.TemporaryDirectory() as cwd_dir, tempfile.TemporaryDirectory() as weights_dir:
            os.chdir(cwd_dir)
            weights = Path(weights_dir) / "adapter_weights.pt"
            weights.write_bytes(b"loaded-adapter")
            decoy = Path(cwd_dir) / "adapter_weights.pt"
            decoy.write_bytes(b"cwd-relative")
            yield Path(cwd_dir), weights, decoy

    def test_baseline_evaluation_sees_fresh_adapter_and_deletes_configured_weights(self):
        engine = _Engine(points_count=len(CORPUS), embed_raw=_vectors)
        seen = {}

        def evaluate(received, queries, qrels, max_queries=None):
            seen["adapter"] = received.adapter
            seen["weights_exist"] = Path(ChelationConfig.ADAPTER_WEIGHTS_PATH).exists()
            seen["max_queries"] = max_queries
            return 0.25

        with self._cwd_and_weights() as (_cwd, weights, decoy):
            with patch.object(ChelationConfig, "ADAPTER_WEIGHTS_PATH", weights), patch(
                "run_sweep.evaluate_ndcg", side_effect=evaluate
            ):
                score, snapshot = prepare_sweep_baseline(
                    engine, CORPUS, QUERIES, QRELS, max_queries=4
                )

            self.assertNotEqual(Path(weights).resolve().parent, Path(_cwd).resolve())
            self.assertFalse(weights.exists())
            self.assertEqual(decoy.read_bytes(), b"cwd-relative")

        self.assertEqual(score, 0.25)
        self.assertEqual(seen["max_queries"], 4)
        self.assertFalse(seen["weights_exist"])
        self.assertIs(seen["adapter"], engine.adapter)
        self.assertIsNot(seen["adapter"], engine.loaded_adapter)
        self.assertIsInstance(seen["adapter"], nn.Module)
        self.assertEqual(engine.embedding_backend.calls, [list(CORPUS.values())])
        engine.embed.assert_not_called()
        self.assertEqual(len(engine.qdrant.upserts), 1)
        self.assertEqual(
            [_as_list(point.vector) for point in engine.qdrant.upserts[0]],
            _vectors(list(CORPUS.values())),
        )
        self.assertEqual(snapshot[0]["id"], 10)
        self.assertEqual(engine.qdrant.scrolls, 1)

    def test_short_collection_ingests_embed_raw_vectors_not_adapter_embed(self):
        engine = _Engine(points_count=1, embed_raw=_vectors)

        def evaluate(received, queries, qrels, max_queries=None):
            self.assertIsNot(received.adapter, received.loaded_adapter)
            return 0.5

        with self._cwd_and_weights() as (_cwd, weights, _decoy):
            with patch.object(ChelationConfig, "ADAPTER_WEIGHTS_PATH", weights), patch(
                "run_sweep.evaluate_ndcg", side_effect=evaluate
            ):
                _score, snapshot = prepare_sweep_baseline(engine, CORPUS, QUERIES, QRELS)
            self.assertFalse(weights.exists())

        self.assertEqual(engine.embedding_backend.calls, [list(CORPUS.values())])
        engine.embed.assert_not_called()
        self.assertEqual(len(engine.qdrant.upserts), 1)
        stored = {point.id: point for point in engine.qdrant.upserts[0]}
        self.assertEqual(stored[10].payload, {"text": "alpha", "original_id": "10"})
        beta_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "b"))
        self.assertEqual(stored[beta_id].payload, {"text": "beta", "original_id": "b"})
        self.assertEqual([_as_list(point.vector) for point in engine.qdrant.upserts[0]], _vectors(list(CORPUS.values())))
        self.assertEqual(snapshot[0]["id"], 10)
        self.assertEqual(snapshot[0]["payload"]["original_id"], "10")
        self.assertEqual(snapshot[1]["id"], beta_id)

    def test_raising_batch_skips_snapshot_baseline_and_weight_delete(self):
        calls = {"n": 0}

        def embed_raw(texts):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("batch blew up")
            return _vectors(texts)

        engine = _Engine(points_count=0, embed_raw=embed_raw)
        evaluated = {"hit": False}

        def evaluate(received, queries, qrels, max_queries=None):
            evaluated["hit"] = True
            return 0.0

        with self._cwd_and_weights() as (_cwd, weights, decoy):
            with patch.object(ChelationConfig, "ADAPTER_WEIGHTS_PATH", weights), patch(
                "run_sweep.evaluate_ndcg", side_effect=evaluate
            ):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = prepare_sweep_baseline(
                        engine, CORPUS, QUERIES, QRELS, batch_size=1
                    )

            self.assertTrue(weights.exists())
            self.assertEqual(decoy.read_bytes(), b"cwd-relative")

        self.assertEqual(code, 1)
        self.assertIn("Ingestion failed for batch 1: batch blew up", output.getvalue())
        self.assertFalse(evaluated["hit"])
        self.assertEqual(engine.qdrant.scrolls, 0)
        self.assertEqual(len(engine.qdrant.upserts), 1)
        self.assertIs(engine.adapter, engine.loaded_adapter)
        engine.embed.assert_not_called()

    def test_run_sweep_failed_ingest_writes_no_results(self):
        engine = _Engine(points_count=0, embed_raw=_raise_ingest)
        with self._cwd_and_weights() as (cwd, weights, decoy):
            with patch.object(ChelationConfig, "ADAPTER_WEIGHTS_PATH", weights), patch(
                "run_sweep.load_mteb_data", return_value=(CORPUS, QUERIES, QRELS)
            ), patch("run_sweep.AntigravityEngine", return_value=engine), patch(
                "run_sweep.evaluate_ndcg", side_effect=AssertionError("baseline should not run")
            ):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = run_parameter_sweep(task_name="NotSciFact", db_path=str(cwd / "db"))
            self.assertFalse((cwd / "sweep_results.json").exists())
            self.assertFalse(list(cwd.glob("sweep_results*")))
            self.assertEqual(decoy.read_bytes(), b"cwd-relative")
            self.assertTrue(weights.exists())
        self.assertEqual(code, 1)
        self.assertIn("Ingestion failed for batch 0", output.getvalue())
        self.assertEqual(engine.qdrant.scrolls, 0)
        self.assertEqual(engine.embedding_backend.calls, [list(CORPUS.values())])
        engine.embed.assert_not_called()

    def test_run_large_sweep_failed_ingest_writes_no_result_files(self):
        engine = _Engine(points_count=0, embed_raw=_raise_ingest)
        with self._cwd_and_weights() as (cwd, weights, decoy):
            with patch.object(ChelationConfig, "ADAPTER_WEIGHTS_PATH", weights), patch(
                "run_large_sweep.load_mteb_data", return_value=(CORPUS, QUERIES, QRELS)
            ), patch("run_large_sweep.AntigravityEngine", return_value=engine), patch(
                "run_sweep.evaluate_ndcg", side_effect=AssertionError("baseline should not run")
            ):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = run_large_parameter_sweep(task_name="NotSciFact", db_path=str(cwd / "db"))
            self.assertFalse((cwd / "large_sweep_results.csv").exists())
            self.assertFalse((cwd / "large_sweep_results.json").exists())
            self.assertFalse((cwd / "large_sweep_results.jsonl").exists())
            self.assertEqual(decoy.read_bytes(), b"cwd-relative")
            self.assertTrue(weights.exists())
        self.assertEqual(code, 1)
        self.assertIn("Ingestion failed for batch 0", output.getvalue())
        self.assertEqual(engine.qdrant.scrolls, 0)
        self.assertEqual(engine.embedding_backend.calls, [list(CORPUS.values())])
        engine.embed.assert_not_called()

    def test_run_large_sweep_empty_collection_ingests_then_snapshots_before_results(self):
        engine = _Engine(points_count=0, embed_raw=_vectors)
        seen = {}

        def baseline(received, queries, qrels, max_queries=None):
            seen["adapter"] = received.adapter
            seen["weights_exist"] = Path(ChelationConfig.ADAPTER_WEIGHTS_PATH).exists()
            return 0.25

        def loop_eval(received, queries, qrels, max_queries=None):
            return 0.4

        def snap(client, collection_name, page_size=256):
            seen["snapshot_before_results"] = not Path("large_sweep_results.csv").exists()
            seen["snapshot_before_json"] = not Path("large_sweep_results.json").exists()
            seen["snapshot_before_jsonl"] = not Path("large_sweep_results.jsonl").exists()
            seen["ingested_before_snapshot"] = engine.embedding_backend.calls == [list(CORPUS.values())]
            return snapshot_collection(client, collection_name, page_size=page_size)

        with self._cwd_and_weights() as (cwd, weights, decoy):
            with patch.object(ChelationConfig, "ADAPTER_WEIGHTS_PATH", weights), patch(
                "run_large_sweep.load_mteb_data", return_value=(CORPUS, QUERIES, QRELS)
            ), patch("run_large_sweep.AntigravityEngine", return_value=engine), patch(
                "run_sweep.evaluate_ndcg", side_effect=baseline
            ), patch("run_large_sweep.evaluate_ndcg", side_effect=loop_eval), patch(
                "run_large_sweep.snapshot_collection", side_effect=snap
            ), patch(
                "run_large_sweep.itertools.product",
                return_value=[(0.01, 1, 0.0, 1, 0.05)],
            ):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = run_large_parameter_sweep(
                        task_name="NotSciFact", max_queries=2, db_path=str(cwd / "db")
                    )
            self.assertTrue((cwd / "large_sweep_results.csv").exists())
            self.assertTrue((cwd / "large_sweep_results.json").exists())
            self.assertTrue((cwd / "large_sweep_results.jsonl").exists())
            self.assertEqual(decoy.read_bytes(), b"cwd-relative")

        self.assertIsNone(code)
        self.assertTrue(seen["snapshot_before_results"])
        self.assertTrue(seen["snapshot_before_json"])
        self.assertTrue(seen["snapshot_before_jsonl"])
        self.assertTrue(seen["ingested_before_snapshot"])
        self.assertFalse(seen["weights_exist"])
        self.assertIsNot(seen["adapter"], engine.loaded_adapter)
        self.assertIsInstance(seen["adapter"], nn.Module)
        engine.embed.assert_not_called()
        engine.run_sedimentation_cycle.assert_called_once()

    def test_scripts_do_not_delete_a_cwd_relative_adapter_file(self):
        for name in ("run_sweep.py", "run_large_sweep.py"):
            source = Path(name).read_text(encoding="utf-8")
            self.assertNotIn('os.remove("adapter_weights.pt")', source)
            self.assertNotIn("os.remove('adapter_weights.pt')", source)
            self.assertNotIn('os.path.exists("adapter_weights.pt")', source)
            self.assertIn("prepare_sweep_baseline(", source)
            self.assertIn("remove_configured_adapter_weights(", source)


def _raise_ingest(_texts):
    raise RuntimeError("disk full")


def _as_list(vector):
    if hasattr(vector, "tolist"):
        return vector.tolist()
    return list(vector)


if __name__ == "__main__":
    unittest.main()
