"""Partial Qdrant upserts write the pre-cycle vectors back."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sedimentation_trainer import sync_vectors_to_qdrant

try:
    import torch
except ImportError:
    torch = None


class _Point:
    def __init__(self, point_id, vector, payload):
        self.id = point_id
        self.vector = vector
        self.payload = payload


class _Client:
    def __init__(self):
        self.points = {}
        self.upserts = 0

    def upsert(self, collection_name, points):
        self.upserts += 1
        if any(point.id == "b" for point in points):
            raise RuntimeError("second chunk failed")
        for point in points:
            self.points[point.id] = (list(point.vector), dict(point.payload or {}))

    def retrieve(self, collection_name, ids, with_vectors=False):
        return []


class _ScriptedClient:
    """Succeeds a fixed number of upserts, then raises."""

    def __init__(self, succeed_upserts, fail_retrieve_after_upserts=None):
        self.succeed_upserts = succeed_upserts
        self.fail_retrieve_after_upserts = fail_retrieve_after_upserts
        self.upserts = 0
        self.points = {}

    def upsert(self, collection_name, points):
        self.upserts += 1
        if self.upserts > self.succeed_upserts:
            raise RuntimeError("upsert failed")
        for point in points:
            self.points[point.id] = (list(point.vector), dict(point.payload or {}))

    def retrieve(self, collection_name, ids, with_vectors=False):
        if (
            self.fail_retrieve_after_upserts is not None
            and self.upserts >= self.fail_retrieve_after_upserts
        ):
            raise RuntimeError("injected retrieve failure during compensation")
        found = []
        for doc_id in ids:
            if doc_id not in self.points:
                continue
            vector, payload = self.points[doc_id]
            found.append(_Point(doc_id, vector, payload))
        return found


class _Logger:
    def __init__(self):
        self.errors = []
        self.completions = []

    def log_error(self, kind, message, **kwargs):
        self.errors.append((kind, message))

    def log_event(self, *args, **kwargs):
        return None

    def log_training_start(self, *args, **kwargs):
        return None

    def log_training_epoch(self, *args, **kwargs):
        return None

    def log_training_complete(self, *args, **kwargs):
        self.completions.append(kwargs)

    def log_checkpoint(self, *args, **kwargs):
        return None


class TestCorpusRollback(unittest.TestCase):
    def test_second_chunk_failure_restores_the_first_id(self):
        client = _Client()
        logger = _Logger()
        originals = np.array([[0.25], [0.5]], dtype=np.float32)
        adapted = np.array([[9.0], [9.0]], dtype=np.float32)
        total, failed, compensated = sync_vectors_to_qdrant(
            client,
            "docs",
            ["a", "b"],
            adapted,
            1,
            logger,
            {"a": {"text": "keep"}, "b": {"text": "later"}},
            original_vectors_np=originals,
        )
        self.assertEqual(failed, 1)
        self.assertTrue(compensated)
        self.assertEqual(client.points["a"][0], [0.25])
        self.assertEqual(client.points["a"][1], {"text": "keep"})
        self.assertNotIn("b", client.points)
        self.assertFalse(any("Rolling back" in message for _, message in logger.errors))
        self.assertGreater(total, 0)

    def test_compensating_upsert_leaves_the_adapted_vector(self):
        client = _ScriptedClient(succeed_upserts=1)
        logger = _Logger()
        originals = np.array([[0.25], [0.5]], dtype=np.float32)
        adapted = np.array([[9.0], [9.0]], dtype=np.float32)
        total, failed, compensated = sync_vectors_to_qdrant(
            client,
            "docs",
            ["a", "b"],
            adapted,
            1,
            logger,
            {"a": {"text": "keep"}, "b": {"text": "later"}},
            original_vectors_np=originals,
        )
        self.assertEqual(failed, 1)
        self.assertGreater(total, 0)
        self.assertFalse(compensated)
        self.assertEqual(client.points["a"][0], [9.0])
        self.assertNotIn("b", client.points)
        messages = [message for _, message in logger.errors]
        self.assertIn("Corpus was not restored.", messages)
        self.assertFalse(any("written back" in message for message in messages))
        self.assertFalse(any("Rolling back" in message for message in messages))

    def test_compensation_retrieve_failure_does_not_escape(self):
        client = _ScriptedClient(succeed_upserts=1, fail_retrieve_after_upserts=2)
        logger = _Logger()
        originals = np.array([[0.25], [0.5]], dtype=np.float32)
        adapted = np.array([[9.0], [9.0]], dtype=np.float32)
        total, failed, compensated = sync_vectors_to_qdrant(
            client,
            "docs",
            ["a", "b"],
            adapted,
            1,
            logger,
            None,
            original_vectors_np=originals,
        )
        self.assertEqual(failed, 1)
        self.assertGreater(total, 0)
        self.assertFalse(compensated)
        self.assertEqual(client.points["a"][0], [9.0])
        messages = [message for _, message in logger.errors]
        self.assertIn("Corpus was not restored.", messages)
        self.assertFalse(any("written back" in message for message in messages))


class _StoredPoint:
    def __init__(self, point_id, vector, payload):
        self.id = point_id
        self.vector = [float(value) for value in vector]
        self.payload = dict(payload or {})


class _Corpus:
    def __init__(self, rows, adapter_path, fail_on_id="b", fail_compensation=False,
                 fail_compensation_retrieve=False):
        self.points = {
            row_id: _StoredPoint(row_id, vector, payload)
            for row_id, vector, payload in rows
        }
        self.adapter_path = adapter_path
        self.fail_on_id = fail_on_id
        self.fail_compensation = fail_compensation
        self.fail_compensation_retrieve = fail_compensation_retrieve
        self.second_failed = False
        self.trained_adapter_bytes = None
        self.first_written = {}
        self.upsert_ids = []

    def scroll(self, collection_name, limit, with_vectors, with_payload, offset):
        return list(self.points.values()), None

    def retrieve(self, collection_name, ids, with_vectors=False):
        if self.fail_compensation_retrieve and self.second_failed and not with_vectors:
            raise RuntimeError("injected retrieve failure during compensation")
        return [self.points[doc_id] for doc_id in ids]

    def upsert(self, collection_name, points):
        if self.trained_adapter_bytes is None:
            self.trained_adapter_bytes = Path(self.adapter_path).read_bytes()
        ids = [point.id for point in points]
        self.upsert_ids.append(ids)
        if self.fail_on_id is not None and any(point.id == self.fail_on_id for point in points):
            self.second_failed = True
            raise RuntimeError("second chunk failed")
        if self.fail_compensation and self.second_failed:
            raise RuntimeError("compensation failed")
        for point in points:
            vector = [float(value) for value in point.vector]
            if point.id not in self.first_written:
                self.first_written[point.id] = vector
            self.points[point.id] = _StoredPoint(point.id, vector, point.payload)


def _clone_state(adapter):
    return {key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()}


def _assert_adapter_state(test, adapter, expected):
    current = adapter.state_dict()
    test.assertEqual(set(current), set(expected))
    for key, value in expected.items():
        test.assertTrue(torch.equal(current[key].detach().cpu(), value), key)


def _assert_file_state(test, path, expected):
    loaded = torch.load(path, weights_only=True)
    test.assertEqual(set(loaded), set(expected))
    for key, value in expected.items():
        test.assertTrue(torch.equal(loaded[key].detach().cpu(), value), key)


class _Teacher:
    _projection_enabled = False
    _projection = None

    def check_dimension_compatibility(self, student_dim):
        return student_dim == 4

    def generate_distillation_targets(self, texts, current_embeddings, teacher_weight=1.0):
        current = np.asarray(current_embeddings, dtype=np.float32)
        target = np.zeros_like(current)
        target[:, -1] = 1.0
        return target


@unittest.skipUnless(torch is not None, "adapter reload tests need torch")
class TestProductionAdapterRollback(unittest.TestCase):
    DIM = 4

    def setUp(self):
        from config import ChelationConfig
        self._chunk_size = ChelationConfig.CHUNK_SIZE
        ChelationConfig.CHUNK_SIZE = 1
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        from config import ChelationConfig
        ChelationConfig.CHUNK_SIZE = self._chunk_size
        self._tmp.cleanup()

    def _basis(self, index):
        values = [0.0] * self.DIM
        values[index] = 1.0
        return values

    def _rows(self):
        payload = {"a": {"text": "alpha"}, "b": {"text": "beta"}}
        return [
            (doc_id, self._basis(0), payload[doc_id])
            for doc_id in ("a", "b")
        ]

    def _new_adapter(self):
        from chelation_adapter import ChelationAdapter
        torch.manual_seed(0)
        return ChelationAdapter(self.DIM)

    def _run_sedimentation(self, client, adapter, path, logger):
        from types import SimpleNamespace
        from antigravity_engine import AntigravityEngine
        from checkpoint_manager import CheckpointManager
        noise = np.array(self._basis(1), dtype=np.float32)
        chelation_log = {
            "a": [noise.copy()],
            "b": [noise.copy()],
        }
        engine = SimpleNamespace(
            training_mode="baseline",
            teacher_helper=None,
            chelation_log=chelation_log,
            logger=logger,
            qdrant=client,
            collection_name="docs",
            adapter=adapter,
            adapter_path=path,
            checkpoint_manager=CheckpointManager(self.tmp / "checkpoints"),
            _stability_tracker=None,
            _sedimentation_optimizer_type="adam",
            _sedimentation_loss_type="mse",
            _sedimentation_loss_kwargs={},
            _convergence_enabled=False,
            _kalman_lr_enabled=False,
            _weight_scheduler=None,
            _annealing_controller=None,
        )
        AntigravityEngine.run_sedimentation_cycle(
            engine,
            threshold=1,
            learning_rate=1.0,
            epochs=5,
            noise_injection=0.0,
        )
        return chelation_log

    def _assert_failure_keeps_pre_cycle_weights(self, adapter, path, pre_state, trained_bytes):
        self.assertNotEqual(path.read_bytes(), trained_bytes)
        _assert_file_state(self, path, pre_state)
        _assert_adapter_state(self, adapter, pre_state)

    def test_failed_sedimentation_restores_prefix_and_keeps_log(self):
        path = self.tmp / "adapter.pt"
        adapter = self._new_adapter()
        pre_state = _clone_state(adapter)
        logger = _Logger()
        client = _Corpus(self._rows(), path)
        chelation_log = self._run_sedimentation(client, adapter, path, logger)
        messages = [message for _, message in logger.errors]
        self.assertIn("a", chelation_log)
        self.assertIn("b", chelation_log)
        self.assertTrue(any("written back" in message for message in messages))
        self.assertFalse(any("Rolling back" in message for message in messages))
        self.assertIn("sedimentation_partial_failure", [kind for kind, _ in logger.errors])
        np.testing.assert_allclose(client.points["a"].vector, self._basis(0), atol=1e-5)
        np.testing.assert_allclose(client.points["b"].vector, self._basis(0), atol=1e-5)
        self._assert_failure_keeps_pre_cycle_weights(
            adapter, path, pre_state, client.trained_adapter_bytes
        )

    def test_compensation_failure_does_not_claim_writeback(self):
        path = self.tmp / "adapter.pt"
        adapter = self._new_adapter()
        pre_state = _clone_state(adapter)
        logger = _Logger()
        client = _Corpus(self._rows(), path, fail_compensation=True)
        chelation_log = self._run_sedimentation(client, adapter, path, logger)
        messages = [message for _, message in logger.errors]
        kinds = [kind for kind, _ in logger.errors]
        self.assertIn("a", chelation_log)
        self.assertIn("corpus_not_restored", kinds)
        self.assertIn("sedimentation_partial_failure", kinds)
        self.assertIn("Corpus was not restored.", messages)
        self.assertFalse(any("written back" in message for message in messages))
        self.assertFalse(any("Rolling back" in message for message in messages))
        self.assertEqual(client.points["a"].vector, client.first_written["a"])
        self.assertFalse(np.allclose(client.points["a"].vector, self._basis(0), atol=1e-5))
        self._assert_failure_keeps_pre_cycle_weights(
            adapter, path, pre_state, client.trained_adapter_bytes
        )

    def test_successful_sedimentation_clears_chelation_log(self):
        path = self.tmp / "adapter.pt"
        adapter = self._new_adapter()
        pre_state = _clone_state(adapter)
        logger = _Logger()
        client = _Corpus(self._rows(), path, fail_on_id=None)
        chelation_log = self._run_sedimentation(client, adapter, path, logger)
        self.assertEqual(chelation_log, {})
        self.assertFalse(logger.errors)
        self.assertEqual(client.upsert_ids, [["a"], ["b"]])
        self.assertEqual(path.read_bytes(), client.trained_adapter_bytes)
        loaded = torch.load(path, weights_only=True)
        self.assertFalse(torch.equal(
            loaded["correction_net.0.weight"].detach().cpu(),
            pre_state["correction_net.0.weight"],
        ))

    def test_existing_adapter_file_is_not_overwritten_before_checkpoint(self):
        path = self.tmp / "adapter.pt"
        adapter = self._new_adapter()
        adapter.save(path)
        pre_bytes = path.read_bytes()
        with torch.no_grad():
            adapter.correction_net[0].weight.add_(1.0)
        logger = _Logger()
        client = _Corpus(self._rows(), path, fail_compensation=True)
        self._run_sedimentation(client, adapter, path, logger)
        self.assertEqual(path.read_bytes(), pre_bytes)
        _assert_adapter_state(self, adapter, torch.load(path, weights_only=True))
        self.assertFalse(any("written back" in message for _, message in logger.errors))

    def test_hierarchical_without_adapter_file_reloads_pre_step_weights(self):
        from checkpoint_manager import CheckpointManager
        from sedimentation import HierarchicalSedimentationEngine
        path = self.tmp / "hierarchical.pt"
        adapter = self._new_adapter()
        pre_state = _clone_state(adapter)
        logger = _Logger()
        client = _Corpus(self._rows(), path)
        noise = np.array(self._basis(1), dtype=np.float32)
        engine = type("Engine", (), {})()
        engine.chelation_log = {"a": [noise.copy()], "b": [noise.copy()]}
        engine.qdrant = client
        engine.collection_name = "docs"
        engine.adapter = adapter
        engine.adapter_path = path
        cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            hierarchical = HierarchicalSedimentationEngine(engine)
            hierarchical.logger = logger
            hierarchical.checkpoint_manager = CheckpointManager(self.tmp / "hier-checkpoints")
            hierarchical.run_hierarchical_sedimentation(
                threshold=1,
                learning_rate=1.0,
                epochs=4,
            )
        finally:
            os.chdir(cwd)
        self.assertIn("a", engine.chelation_log)
        self.assertIn("b", engine.chelation_log)
        kinds = [kind for kind, _ in logger.errors]
        self.assertIn("hierarchical_sedimentation_partial_failure", kinds)
        self.assertFalse(any("written back" in message for _, message in logger.errors))
        self.assertFalse(any("Rolling back" in message for _, message in logger.errors))
        self._assert_failure_keeps_pre_cycle_weights(
            adapter, path, pre_state, client.trained_adapter_bytes
        )

    def _run_offline(self, client, adapter, path, logger):
        from types import SimpleNamespace
        from antigravity_engine import AntigravityEngine
        engine = SimpleNamespace(
            teacher_helper=_Teacher(),
            logger=logger,
            training_mode="offline",
            vector_size=self.DIM,
            qdrant=client,
            collection_name="docs",
            adapter=adapter,
            adapter_path=path,
            _sedimentation_optimizer_type="adam",
            _sedimentation_loss_type="mse",
            _sedimentation_loss_kwargs={},
            _convergence_enabled=False,
            _kalman_lr_enabled=False,
            _weight_scheduler=None,
        )
        AntigravityEngine.run_offline_distillation(
            engine,
            batch_size=10,
            learning_rate=1.0,
            epochs=5,
        )

    def test_offline_failed_second_upsert_reloads_existing_file(self):
        path = self.tmp / "adapter.pt"
        adapter = self._new_adapter()
        adapter.save(path)
        pre_bytes = path.read_bytes()
        pre_state = _clone_state(adapter)
        logger = _Logger()
        client = _Corpus(self._rows(), path)
        self._run_offline(client, adapter, path, logger)
        self.assertNotEqual(client.trained_adapter_bytes, pre_bytes)
        self.assertEqual(path.read_bytes(), pre_bytes)
        _assert_file_state(self, path, pre_state)
        _assert_adapter_state(self, adapter, pre_state)
        np.testing.assert_allclose(client.points["a"].vector, self._basis(0), atol=1e-5)

    def test_offline_missing_file_reloads_weights_saved_before_training(self):
        path = self.tmp / "adapter.pt"
        adapter = self._new_adapter()
        pre_state = _clone_state(adapter)
        logger = _Logger()
        client = _Corpus(self._rows(), path)
        self.assertFalse(path.exists())
        self._run_offline(client, adapter, path, logger)
        self.assertTrue(path.exists())
        self._assert_failure_keeps_pre_cycle_weights(
            adapter, path, pre_state, client.trained_adapter_bytes
        )

    def test_offline_compensation_retrieve_failure_reloads_adapter(self):
        path = self.tmp / "adapter.pt"
        adapter = self._new_adapter()
        adapter.save(path)
        pre_bytes = path.read_bytes()
        pre_state = _clone_state(adapter)
        logger = _Logger()
        client = _Corpus(self._rows(), path, fail_compensation_retrieve=True)
        self._run_offline(client, adapter, path, logger)
        messages = [message for _, message in logger.errors]
        kinds = [kind for kind, _ in logger.errors]
        self.assertIn("Corpus was not restored.", messages)
        self.assertIn("offline_distillation_mixed_store", kinds)
        self.assertEqual(logger.completions, [])
        self.assertFalse(any("written back" in message for message in messages))
        self.assertEqual(path.read_bytes(), pre_bytes)
        _assert_adapter_state(self, adapter, pre_state)
        self.assertEqual(client.points["a"].vector, client.first_written["a"])
        self.assertFalse(np.allclose(client.points["a"].vector, self._basis(0), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
