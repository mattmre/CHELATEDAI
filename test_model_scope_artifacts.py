import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from model_scope_artifacts import (
    MODEL_SCOPE_ARTIFACT_SCHEMA_VERSION,
    ArtifactStore,
    build_model_scope_artifact,
    feature_event_from_dict,
    feature_event_to_dict,
    load_model_scope_artifact,
    model_scope_artifact_to_evidence_event,
    summarize_model_scope_artifact,
    write_model_scope_artifact,
)
from model_scope_features import SparseFeatureEvent
from model_scope_runtime import ActivationEvent


class TestModelScopeArtifacts(unittest.TestCase):
    def test_round_trip_and_summary(self):
        artifact = build_model_scope_artifact(
            runtime={
                "model_name": "Qwen/Qwen3.5-2B",
                "device": "cpu",
            },
            capture={
                "metadata": {"query_id": "q1"},
                "token_count": 4,
                "captured_layer_count": 2,
                "layer_indices": [0, 1],
                "observations": [
                    {"layer_index": 0},
                    {"layer_index": 1},
                ],
            },
            trace={"run_id": "unit"},
            evidence={"event_id": "evt_existing"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_model_scope_artifact(Path(tmpdir) / "artifact.json", artifact)
            loaded = load_model_scope_artifact(path)

        self.assertEqual(loaded["schema_version"], MODEL_SCOPE_ARTIFACT_SCHEMA_VERSION)
        self.assertEqual(loaded["runtime"]["model_name"], "Qwen/Qwen3.5-2B")
        summary = summarize_model_scope_artifact(loaded)
        self.assertEqual(summary["captured_layer_count"], 2)
        self.assertEqual(summary["token_count"], 4)
        self.assertEqual(summary["layer_indices"], [0, 1])
        self.assertEqual(summary["evidence_event_id"], "evt_existing")

        event = model_scope_artifact_to_evidence_event(loaded)
        self.assertEqual(event["surface"], "model_scope")
        self.assertEqual(event["query_id"], "q1")
        self.assertEqual(event["trace"]["run_id"], "unit")


def _make_activation_event(run_id: str = "run-001", layer_id: str = "layer.0") -> ActivationEvent:
    return ActivationEvent(
        schema_version="1.0",
        model_id="Qwen3.5-7B",
        layer_id=layer_id,
        token_count=10,
        shape=(1, 10, 4096),
        mean_activation=0.5,
        norm_activation=1.2,
        captured_at=datetime.now(timezone.utc).isoformat(),
        run_id=run_id,
    )


def _make_feature_event(run_id: str = "run-001", layer_id: str = "layer.0") -> SparseFeatureEvent:
    activation = _make_activation_event(run_id=run_id, layer_id=layer_id)
    features = {
        "mean_activation": 0.5,
        "norm_activation": 1.2,
        "token_count": 10.0,
        "shape_0": 1.0,
    }
    return SparseFeatureEvent(
        source_activation=activation,
        feature_source="raw_stats",
        features=features,
        feature_count=4,
        nonzero_count=4,
        extracted_at=datetime.now(timezone.utc).isoformat(),
    )


class TestFeatureEventSerialisation(unittest.TestCase):
    def test_to_dict_returns_dict(self):
        event = _make_feature_event()
        d = feature_event_to_dict(event)
        self.assertIsInstance(d, dict)

    def test_to_dict_has_schema_version(self):
        event = _make_feature_event()
        d = feature_event_to_dict(event)
        self.assertEqual(d["schema_version"], "1.0")

    def test_to_dict_has_source_activation(self):
        event = _make_feature_event()
        d = feature_event_to_dict(event)
        self.assertIn("source_activation", d)
        self.assertIsInstance(d["source_activation"], dict)

    def test_round_trip_schema_version(self):
        event = _make_feature_event()
        restored = feature_event_from_dict(feature_event_to_dict(event))
        self.assertEqual(restored.schema_version, event.schema_version)

    def test_round_trip_feature_source(self):
        event = _make_feature_event()
        restored = feature_event_from_dict(feature_event_to_dict(event))
        self.assertEqual(restored.feature_source, event.feature_source)

    def test_round_trip_features(self):
        event = _make_feature_event()
        restored = feature_event_from_dict(feature_event_to_dict(event))
        self.assertEqual(restored.features, event.features)

    def test_round_trip_nonzero_count(self):
        event = _make_feature_event()
        restored = feature_event_from_dict(feature_event_to_dict(event))
        self.assertEqual(restored.nonzero_count, event.nonzero_count)

    def test_round_trip_nested_activation_model_id(self):
        event = _make_feature_event()
        restored = feature_event_from_dict(feature_event_to_dict(event))
        self.assertEqual(
            restored.source_activation.model_id, event.source_activation.model_id
        )

    def test_round_trip_nested_activation_shape(self):
        event = _make_feature_event()
        restored = feature_event_from_dict(feature_event_to_dict(event))
        self.assertEqual(
            restored.source_activation.shape, event.source_activation.shape
        )


class TestArtifactStore(unittest.TestCase):
    def test_save_returns_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(base_dir=tmpdir)
            event = _make_feature_event()
            path = store.save_feature_event(event)
            self.assertIsInstance(path, Path)

    def test_save_creates_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(base_dir=tmpdir)
            event = _make_feature_event()
            path = store.save_feature_event(event)
            self.assertTrue(path.exists())

    def test_save_with_custom_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(base_dir=tmpdir)
            event = _make_feature_event()
            path = store.save_feature_event(event, name="custom_event.json")
            self.assertEqual(path.name, "custom_event.json")

    def test_load_restores_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(base_dir=tmpdir)
            event = _make_feature_event()
            path = store.save_feature_event(event)
            loaded = store.load_feature_event(path)
            self.assertEqual(loaded.feature_source, event.feature_source)

    def test_save_load_round_trip_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(base_dir=tmpdir)
            event = _make_feature_event()
            path = store.save_feature_event(event)
            loaded = store.load_feature_event(path)
            self.assertEqual(loaded.features, event.features)

    def test_save_load_preserves_nested_activation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(base_dir=tmpdir)
            event = _make_feature_event(run_id="my-run", layer_id="layer.3")
            path = store.save_feature_event(event)
            loaded = store.load_feature_event(path)
            self.assertEqual(loaded.source_activation.run_id, "my-run")
            self.assertEqual(loaded.source_activation.layer_id, "layer.3")

    def test_save_batch_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(base_dir=tmpdir)
            events = [_make_feature_event(run_id=f"run-{i}") for i in range(3)]
            manifest_path = store.save_batch(events)
            self.assertTrue(manifest_path.exists())

    def test_save_batch_returns_manifest_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(base_dir=tmpdir)
            events = [_make_feature_event(run_id=f"run-{i}") for i in range(2)]
            manifest_path = store.save_batch(events)
            self.assertIsInstance(manifest_path, Path)

    def test_save_batch_writes_individual_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(base_dir=tmpdir)
            events = [_make_feature_event(run_id=f"run-{i}") for i in range(3)]
            store.save_batch(events)
            json_files = list(Path(tmpdir).glob("feature_event_*.json"))
            self.assertEqual(len(json_files), 3)

    def test_load_batch_reloads_all_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(base_dir=tmpdir)
            events = [_make_feature_event(run_id=f"run-{i}") for i in range(3)]
            manifest_path = store.save_batch(events)
            loaded = store.load_batch(manifest_path)
            self.assertEqual(len(loaded), 3)

    def test_load_batch_preserves_feature_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(base_dir=tmpdir)
            events = [_make_feature_event(run_id=f"run-{i}") for i in range(2)]
            manifest_path = store.save_batch(events)
            loaded = store.load_batch(manifest_path)
            for ev in loaded:
                self.assertEqual(ev.feature_source, "raw_stats")

    def test_list_artifacts_returns_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(base_dir=tmpdir)
            events = [_make_feature_event(run_id=f"run-{i}") for i in range(3)]
            store.save_batch(events)
            artifacts = store.list_artifacts()
            self.assertEqual(len(artifacts), 3)

    def test_list_artifacts_returns_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(base_dir=tmpdir)
            artifacts = store.list_artifacts()
            self.assertIsInstance(artifacts, list)

    def test_list_artifacts_custom_pattern(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(base_dir=tmpdir)
            event = _make_feature_event()
            store.save_feature_event(event, name="custom_file.json")
            artifacts = store.list_artifacts(pattern="custom_*.json")
            self.assertEqual(len(artifacts), 1)

    def test_manifest_json_is_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(base_dir=tmpdir)
            events = [_make_feature_event(run_id=f"run-{i}") for i in range(2)]
            manifest_path = store.save_batch(events)
            with manifest_path.open("r") as fh:
                manifest = json.load(fh)
            self.assertIn("artifact_paths", manifest)
            self.assertEqual(manifest["count"], 2)


if __name__ == "__main__":
    unittest.main()
