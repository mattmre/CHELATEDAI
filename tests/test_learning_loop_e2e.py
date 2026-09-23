"""Floor-tier learning loop: ingest, sedimentation, ranked-id delta.

The embedding backend is a dim-8 fixture stand-in, not MiniLM and not a BEIR
run. It is not a production encoder.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from antigravity_engine import AntigravityEngine  # noqa: E402
from benchmark_utils import isolated_adapter_state  # noqa: E402
from chelation_adapter import ChelationAdapter  # noqa: E402

_DOCUMENTS = [f"doc-{index}" for index in range(6)]
_QUERY = "query-seed"
_INGESTED_IDS = list(range(len(_DOCUMENTS)))


class _FloorDim8Backend:
    """Floor-tier stand-in encoder. Not MiniLM. Not a BEIR run.

    For text i in this call, component i % 8 is 1.0 and component 0 is 0.25.
    Component 0 is written second, so i % 8 == 0 leaves 0.25 there.
    """

    vector_size = 8

    def embed_raw(self, texts):
        rows = []
        for index, _text in enumerate(texts):
            vector = np.zeros(self.vector_size, dtype=np.float32)
            vector[index % 8] = 1.0
            vector[0] = 0.25
            rows.append(vector)
        return np.vstack(rows)


@contextmanager
def _learning_loop_engine():
    """Real engine, real in-memory Qdrant, real ChelationAdapter.

    CheckpointManager writes ./checkpoints relative to the process cwd, and
    AntigravityEngine constructs that manager immediately. The directory
    change happens before construction, which is before sedimentation, so
    those files are not written into the repo. isolated_adapter_state keeps
    ChelationConfig.ADAPTER_WEIGHTS_PATH from being left behind.
    """
    import torch

    cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="learning-loop-") as temp_dir:
        os.chdir(temp_dir)
        try:
            with isolated_adapter_state():
                with patch("antigravity_engine.get_logger", return_value=MagicMock()), patch(
                    "antigravity_engine.create_embedding_backend",
                    return_value=_FloorDim8Backend(),
                ):
                    torch.manual_seed(0)
                    np.random.seed(0)
                    engine = AntigravityEngine(
                        qdrant_location=":memory:",
                        use_centering=True,
                        training_mode="baseline",
                        store_full_text_payload=True,
                        model_name="floor-dim8-fixture",
                    )
                    yield engine
        finally:
            os.chdir(cwd)


def _ranked_ids(engine):
    """Chelated ranking from run_inference (final ids when centering is on)."""
    result = engine.run_inference(_QUERY)
    return list(result[1])


class TestLearningLoopE2E(unittest.TestCase):
    def test_sedimentation_moves_ranked_ids_and_query_embedding(self):
        with _learning_loop_engine() as engine:
            self.assertIsInstance(engine.adapter, ChelationAdapter)
            engine.ingest(_DOCUMENTS)
            ranked_before = _ranked_ids(engine)
            self.assertGreater(len(engine.chelation_log), 0)
            log_ids = sorted(engine.chelation_log)
            log_counts = [len(engine.chelation_log[doc_id]) for doc_id in log_ids]
            before_embedding = np.asarray(engine.embed(_QUERY), dtype=np.float64)[0]
            engine.run_sedimentation_cycle(threshold=1, learning_rate=0.5, epochs=8)
            ranked_after = _ranked_ids(engine)
            after_embedding = np.asarray(engine.embed(_QUERY), dtype=np.float64)[0]
            delta = float(np.linalg.norm(after_embedding - before_embedding))
            print(
                "LEARNING_LOOP_POSITIVE "
                f"ranked_before={ranked_before} ranked_after={ranked_after} "
                f"l2={delta:.8f} log_ids={log_ids} log_counts={log_counts} "
                f"adapter={type(engine.adapter).__name__}"
            )
            self.assertNotEqual(ranked_before, ranked_after)
            self.assertTrue(np.isfinite(delta))
            self.assertGreater(delta, 0.0)
            self.assertTrue(set(_INGESTED_IDS).issubset(ranked_before))
            self.assertTrue(set(_INGESTED_IDS).issubset(ranked_after))

    def test_zero_epochs_leaves_ranked_ids_unchanged(self):
        with _learning_loop_engine() as engine:
            self.assertIsInstance(engine.adapter, ChelationAdapter)
            engine.ingest(_DOCUMENTS)
            ranked_before = _ranked_ids(engine)
            engine.run_sedimentation_cycle(threshold=1, learning_rate=0.5, epochs=0)
            ranked_after = _ranked_ids(engine)
            print(
                "LEARNING_LOOP_ZERO "
                f"ranked_before={ranked_before} ranked_after={ranked_after}"
            )
            self.assertEqual(ranked_before, ranked_after)
            self.assertTrue(set(_INGESTED_IDS).issubset(ranked_before))
            self.assertTrue(set(_INGESTED_IDS).issubset(ranked_after))


if __name__ == "__main__":
    unittest.main()
