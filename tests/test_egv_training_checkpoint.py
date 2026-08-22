from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from egv.canonical import digest_for
from egv.training import artifacts as artifact_module
from egv.training.checkpoint import (
    ADAPTER_TENSORS_NAME,
    RNG_TENSORS_NAME,
    STATE_NAME,
    TrainingCheckpoint,
    TrainingCheckpointStore,
)
from egv.training.errors import TrainingArtifactError, TrainingCheckpointError


def _sha(label: str) -> str:
    return digest_for(label)


def _cursor(step: int) -> dict:
    return {
        "schema_version": "egv-training-cursor-v1",
        "epoch": step // 2,
        "row_index": step * 4,
        "consumed_examples": step * 4,
        "sampler_seed": 17,
        "dataset_digest": _sha("dataset"),
    }


def _scheduler(step: int) -> dict:
    return {
        "schema_version": "egv-training-scheduler-v1",
        "scheduler_class": "LinearLR",
        "last_epoch": step,
        "step_count": step,
        "base_lrs": [0.001],
        "last_lrs": [0.001],
    }


def _rng() -> dict:
    return {
        "schema_version": "egv-training-rng-v1",
        "python_state": {"version": 3, "state": [1, 2, 3], "gauss": None},
        "numpy_state": {
            "bit_generator": "MT19937",
            "keys": [4, 5, 6],
            "position": 2,
            "has_gauss": 0,
            "cached_gaussian": 0.0,
        },
        "torch_cpu_tensor_name": "torch_cpu_rng",
        "torch_cuda_tensor_names": [],
        "cuda_device_count": 0,
    }


def _optimizer() -> dict:
    parameter = "layer.lora_A.weight"
    return {
        "schema_version": "egv-training-optimizer-v1",
        "optimizer_class": "AdamW",
        "parameter_names": [parameter],
        "parameter_groups": [{
            "parameter_names": [parameter], "lr": 0.001, "betas": [0.9, 0.999],
            "eps": 1e-8, "weight_decay": 0.01,
        }],
        "tensor_parameter_names": {"state.layer.lora_A.weight.exp_avg": parameter},
    }


def _checkpoint(step: int, parent: str = None, **changes) -> TrainingCheckpoint:
    values = {
        "campaign_id": "campaign-private-1",
        "run_id": "run-private-1",
        "global_step": step,
        "frozen_dataset_digest": _sha("dataset"),
        "target_manifest_digest": _sha("targets"),
        "tokenizer_manifest_digest": _sha("tokenizer"),
        "software_manifest_digest": _sha("software"),
        "base_model_manifest_hash": _sha("base"),
        "base_snapshot_digest": _sha("snapshot"),
        "ledger_cutoff_hash": _sha("ledger"),
        "protocol_hash": _sha("protocol"),
        "parent_artifact_digest": parent,
        "data_cursor": _cursor(step),
        "scheduler_state": _scheduler(step),
        "rng_state": _rng(),
        "optimizer_metadata": _optimizer(),
    }
    values.update(changes)
    return TrainingCheckpoint(**values)


def _save(store: TrainingCheckpointStore, step: int, parent: str = None, rng_tensor=None, **changes):
    return store.save(
        _checkpoint(step, parent, **changes),
        adapter_weights={"layer.lora_A.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)},
        optimizer_tensors={"state.layer.lora_A.weight.exp_avg": torch.ones(2, 3)},
        rng_tensors={"torch_cpu_rng": torch.get_rng_state() if rng_tensor is None else rng_tensor},
    )


class TrainingCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_round_trip_is_content_addressed_and_idempotent(self) -> None:
        store = TrainingCheckpointStore(self.root)
        path, digest = _save(store, 4)
        second_path, second_digest = _save(store, 4)
        self.assertEqual((path, digest), (second_path, second_digest))
        loaded = store.resume(expected_artifact_digest=digest)
        self.assertEqual(loaded.checkpoint, _checkpoint(4))
        self.assertTrue(torch.equal(loaded.optimizer_tensors["state.layer.lora_A.weight.exp_avg"], torch.ones(2, 3)))
        self.assertNotEqual((path / ADAPTER_TENSORS_NAME).read_bytes()[:2], b"\x80\x04")
        self.assertTrue((path / RNG_TENSORS_NAME).is_file())

    def test_all_provenance_digests_are_frozen(self) -> None:
        checkpoint = _checkpoint(1)
        for field in (
            "frozen_dataset_digest", "target_manifest_digest", "tokenizer_manifest_digest",
            "software_manifest_digest", "base_model_manifest_hash", "base_snapshot_digest",
            "ledger_cutoff_hash", "protocol_hash",
        ):
            self.assertEqual(len(getattr(checkpoint, field)), 64)

    def test_arbitrary_nested_json_is_rejected(self) -> None:
        bad = _scheduler(1)
        bad["arbitrary"] = {"payload": [1, 2, 3]}
        with self.assertRaisesRegex(TrainingCheckpointError, "closed schema"):
            _checkpoint(1, scheduler_state=bad)

    def test_optimizer_parameter_topology_is_exact(self) -> None:
        store = TrainingCheckpointStore(self.root)
        with self.assertRaisesRegex(TrainingCheckpointError, "topology"):
            store.save(
                _checkpoint(1),
                adapter_weights={"layer.lora_A.weight": torch.ones(1)},
                optimizer_tensors={"hidden.base.weight": torch.ones(1)},
                rng_tensors={"torch_cpu_rng": torch.get_rng_state()},
            )

    def test_cpu_and_all_declared_cuda_rng_tensors_are_required(self) -> None:
        state = _rng()
        state["cuda_device_count"] = 2
        state["torch_cuda_tensor_names"] = ["torch_cuda_rng.0", "torch_cuda_rng.1"]
        store = TrainingCheckpointStore(self.root)
        with self.assertRaisesRegex(TrainingCheckpointError, "CPU/all-CUDA"):
            store.save(
                _checkpoint(1, rng_state=state),
                adapter_weights={"layer.lora_A.weight": torch.ones(1)},
                optimizer_tensors={"state.layer.lora_A.weight.exp_avg": torch.ones(1)},
                rng_tensors={"torch_cpu_rng": torch.get_rng_state(), "torch_cuda_rng.0": torch.zeros(4, dtype=torch.uint8)},
            )

    def test_deterministic_resume_equivalence_restores_torch_rng(self) -> None:
        store = TrainingCheckpointStore(self.root)
        torch.manual_seed(1701)
        saved_rng = torch.get_rng_state().clone()
        _expected_first = torch.rand(5)
        expected_second = torch.rand(5)
        _path, digest = _save(store, 1, rng_tensor=saved_rng)
        torch.manual_seed(9999)
        loaded = store.resume(expected_artifact_digest=digest)
        torch.set_rng_state(loaded.rng_tensors["torch_cpu_rng"])
        actual_first = torch.rand(5)
        actual_second = torch.rand(5)
        torch.set_rng_state(saved_rng)
        self.assertTrue(torch.equal(actual_first, torch.rand(5)))
        self.assertTrue(torch.equal(actual_second, expected_second))

    def test_production_latest_requires_authoritative_digest(self) -> None:
        store = TrainingCheckpointStore(self.root)
        with self.assertRaisesRegex(TrainingCheckpointError, "authoritative"):
            store.latest()

    def test_injected_higher_step_cannot_override_authoritative_head(self) -> None:
        store = TrainingCheckpointStore(self.root)
        _path, trusted = _save(store, 1)
        _save(store, 999)  # attacker-controlled disconnected checkpoint
        self.assertEqual(store.latest(expected_artifact_digest=trusted).checkpoint.global_step, 1)
        self.assertEqual(store.scan_latest_diagnostic().checkpoint.global_step, 999)

    def test_recursive_lineage_is_verified(self) -> None:
        store = TrainingCheckpointStore(self.root)
        _p1, first = _save(store, 1)
        _p2, second = _save(store, 2, first)
        _p3, third = _save(store, 3, second)
        self.assertEqual(store.resume(expected_artifact_digest=third).lineage, (third, second, first))

    def test_provenance_change_in_lineage_is_rejected_before_save(self) -> None:
        store = TrainingCheckpointStore(self.root)
        _path, parent = _save(store, 1)
        with self.assertRaisesRegex(TrainingCheckpointError, "lineage"):
            _save(store, 2, parent, software_manifest_digest=_sha("other-software"))

    def test_state_extra_file_and_replacement_are_rejected(self) -> None:
        store = TrainingCheckpointStore(self.root)
        path, digest = _save(store, 1)
        state = json.loads((path / STATE_NAME).read_text(encoding="utf-8"))
        state["untrusted_extra"] = True
        (path / STATE_NAME).write_text(json.dumps(state), encoding="utf-8")
        with self.assertRaises(TrainingCheckpointError):
            store.load(digest)

    def test_tensor_replacement_and_extra_file_are_rejected(self) -> None:
        store = TrainingCheckpointStore(self.root)
        path, digest = _save(store, 1)
        (path / ADAPTER_TENSORS_NAME).write_bytes(b"replacement")
        with self.assertRaisesRegex(TrainingCheckpointError, "digest changed"):
            store.load(digest)
        path, digest = _save(TrainingCheckpointStore(self.root / "other"), 1)
        (path / "hidden.pkl").write_bytes(b"unsafe")
        with self.assertRaisesRegex(TrainingCheckpointError, "not exhaustive"):
            TrainingCheckpointStore(self.root / "other").load(digest)

    def test_hardlink_and_symlink_attacks_fail_closed(self) -> None:
        store = TrainingCheckpointStore(self.root / "hard")
        path, digest = _save(store, 1)
        member = path / STATE_NAME
        outside = self.root / "outside.json"
        outside.write_bytes(member.read_bytes())
        member.unlink()
        os.link(outside, member)
        with self.assertRaisesRegex(TrainingCheckpointError, "hard-linked"):
            store.load(digest)
        target = self.root / "target"
        target.mkdir()
        link = self.root / "linked-store"
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            return
        with self.assertRaisesRegex(TrainingArtifactError, "symlink"):
            TrainingCheckpointStore(link)

    def test_interrupted_publication_is_not_authoritative(self) -> None:
        store = TrainingCheckpointStore(self.root)
        with mock.patch.object(artifact_module, "_fsync_directory", side_effect=OSError("simulated crash")):
            with self.assertRaisesRegex(OSError, "simulated crash"):
                _save(store, 1)
        self.assertIsNone(store.scan_latest_diagnostic())
        self.assertFalse(list(self.root.rglob(".publishing-*")))


if __name__ == "__main__":
    unittest.main()
