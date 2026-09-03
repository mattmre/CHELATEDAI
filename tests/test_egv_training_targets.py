from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest

import torch
from torch import nn

from egv.canonical import digest_for
from egv.training.immutability import (
    TrainingImmutabilityError,
    assert_base_model_immutable,
    assert_lora_only_trainable,
    assert_optimizer_lora_only,
    capture_base_model_snapshot,
    validate_lora_only_training_state,
)
from egv.training.targets import (
    FROZEN_LORA_TARGETS,
    FULL_ATTENTION_LAYERS,
    LORA_TARGET_COUNT,
    LoraTargetManifest,
    TrainingTargetError,
    build_lora_target_manifest,
    frozen_lora_target_names,
    validate_lora_target_modules,
)
from egv.variation.model import MODEL_ARCHITECTURE, MODEL_CONFIG_CLASS, model_tensor_hashes


class FullAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(2, 2)
        self.k_proj = nn.Linear(2, 2)
        self.v_proj = nn.Linear(2, 2)
        self.o_proj = nn.Linear(2, 2)


class LinearAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = nn.Linear(2, 2)
        self.in_proj_z = nn.Linear(2, 2)
        self.in_proj_a = nn.Linear(2, 2)
        self.in_proj_b = nn.Linear(2, 2)
        self.out_proj = nn.Linear(2, 2)


class Layer(nn.Module):
    def __init__(self, index):
        super().__init__()
        if index in FULL_ATTENTION_LAYERS:
            self.self_attn = FullAttention()
        else:
            self.linear_attn = LinearAttention()
        self.mlp = nn.Linear(2, 2)


class TinyFrozenShape(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Layer(index) for index in range(24)])
        self.register_buffer("proof_buffer", torch.tensor([7.0]))


class TrainingTargetTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="egv-target-tests-")
        self.root = Path(self.tempdir.name)
        self.model_digest = digest_for("manifest")

    def tearDown(self):
        self.tempdir.cleanup()

    @staticmethod
    def build_model():
        torch.manual_seed(4)
        return TinyFrozenShape()

    def manifest(self, model):
        return build_lora_target_manifest(model, model_manifest_digest=self.model_digest)

    @staticmethod
    def add_adapters(model):
        for target in FROZEN_LORA_TARGETS:
            module = model.get_submodule(target)
            module.lora_A = nn.ModuleDict({"default": nn.Linear(2, 1, bias=False)})
            module.lora_B = nn.ModuleDict({"default": nn.Linear(1, 2, bias=False)})
        for name, parameter in model.named_parameters():
            parameter.requires_grad = ".lora_A." in name or ".lora_B." in name
        return model

    def snapshot(self, model, manifest):
        root = self.root / "model"
        root.mkdir()
        checkpoint = root / "weights.bin"
        checkpoint.write_bytes(b"immutable checkpoint")
        files = {"weights.bin": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        return capture_base_model_snapshot(
            model,
            model_root=root,
            model_manifest_digest=self.model_digest,
            target_manifest=manifest,
            file_hashes=files,
        )

    def reset_model_root(self):
        path = self.root / "model"
        if path.exists():
            shutil.rmtree(path)

    def test_frozen_target_inventory_is_exact_and_deterministic(self):
        self.assertEqual(frozen_lora_target_names(), FROZEN_LORA_TARGETS)
        self.assertEqual(len(FROZEN_LORA_TARGETS), LORA_TARGET_COUNT)
        self.assertEqual(LORA_TARGET_COUNT, 114)
        self.assertEqual(len(set(FROZEN_LORA_TARGETS)), 114)
        self.assertIn("model.layers.3.self_attn.q_proj", FROZEN_LORA_TARGETS)
        self.assertIn("model.layers.0.linear_attn.in_proj_qkv", FROZEN_LORA_TARGETS)

    def test_target_manifest_is_canonical_sealed_and_model_bound(self):
        model = self.build_model()
        first = self.manifest(model)
        second = self.manifest(model)
        self.assertEqual(first, second)
        self.assertEqual(first.architecture, MODEL_ARCHITECTURE)
        self.assertEqual(first.config_class, MODEL_CONFIG_CLASS)
        self.assertEqual(len(first.module_names), 114)
        self.assertEqual(len(first.module_types), 114)
        self.assertEqual(first.canonical_json(), second.canonical_json())

    def test_manifest_substitution_and_wrong_model_identity_fail(self):
        model = self.build_model()
        manifest = self.manifest(model)
        forged = LoraTargetManifest(
            manifest.schema_version,
            digest_for("other-model"),
            manifest.architecture,
            manifest.config_class,
            manifest.module_names,
            manifest.module_types,
            manifest.digest,
        )
        with self.assertRaisesRegex(TrainingTargetError, "digest"):
            forged.verify()
        with self.assertRaisesRegex(TrainingTargetError, "architecture"):
            build_lora_target_manifest(
                model, model_manifest_digest=self.model_digest, architecture="SubstitutedArchitecture"
            )

    def test_target_validator_accepts_only_exact_linear_inventory(self):
        report = validate_lora_target_modules(self.build_model())
        self.assertEqual(report.count, 114)
        self.assertEqual(report.module_names, FROZEN_LORA_TARGETS)

    def test_target_validator_rejects_missing_additional_and_wrong_type(self):
        missing = self.build_model()
        del missing.model.layers[0].linear_attn.in_proj_a
        with self.assertRaisesRegex(TrainingTargetError, "missing"):
            validate_lora_target_modules(missing)
        additional = self.build_model()
        additional.model.layers.append(Layer(24))
        with self.assertRaisesRegex(TrainingTargetError, "additional"):
            validate_lora_target_modules(additional)
        wrong = self.build_model()
        wrong.model.layers[3].self_attn.q_proj = nn.Identity()
        with self.assertRaisesRegex(TrainingTargetError, "exact torch.nn.Linear"):
            validate_lora_target_modules(wrong)

    def test_snapshot_and_lora_only_success(self):
        model = self.build_model()
        manifest = self.manifest(model)
        proof = self.snapshot(model, manifest)
        self.add_adapters(model)
        adapter_parameters = assert_lora_only_trainable(model, manifest)
        optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad])
        assert_optimizer_lora_only(optimizer, model, manifest)
        result = validate_lora_only_training_state(proof, model, target_manifest=manifest, optimizer=optimizer)
        self.assertEqual(result["snapshot_digest"], proof.digest)
        self.assertEqual(result["target_manifest_digest"], manifest.digest)
        self.assertEqual(len(adapter_parameters), 228)

    def test_base_parameter_and_buffer_mutation_fail(self):
        for mutate in ("parameter", "buffer"):
            with self.subTest(mutate=mutate):
                self.reset_model_root()
                model = self.build_model()
                manifest = self.manifest(model)
                proof = self.snapshot(model, manifest)
                self.add_adapters(model)
                with torch.no_grad():
                    if mutate == "parameter":
                        model.model.layers[0].mlp.weight.add_(1)
                    else:
                        model.proof_buffer.add_(1)
                with self.assertRaisesRegex(TrainingImmutabilityError, "tensor changed"):
                    assert_base_model_immutable(proof, model, manifest)

    def test_checkpoint_file_mutation_fails(self):
        model = self.build_model()
        manifest = self.manifest(model)
        proof = self.snapshot(model, manifest)
        self.add_adapters(model)
        (proof.model_root / "weights.bin").write_bytes(b"tampered")
        with self.assertRaisesRegex(TrainingImmutabilityError, "file digest changed"):
            assert_base_model_immutable(proof, model, manifest)

    def test_non_lora_trainability_and_optimizer_smuggling_fail(self):
        model = self.build_model()
        manifest = self.manifest(model)
        self.add_adapters(model)
        model.model.layers[0].mlp.weight.requires_grad = True
        with self.assertRaisesRegex(TrainingImmutabilityError, "non-allowlisted"):
            assert_lora_only_trainable(model, manifest)
        model.model.layers[0].mlp.weight.requires_grad = False
        optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad])
        optimizer.param_groups[0]["params"].append(model.model.layers[0].mlp.weight)
        with self.assertRaisesRegex(TrainingImmutabilityError, "not exactly"):
            assert_optimizer_lora_only(optimizer, model, manifest)

    def test_suffix_overmatch_cannot_smuggle_adapter_target(self):
        model = self.build_model()
        manifest = self.manifest(model)
        self.add_adapters(model)
        model.evil = nn.Module()
        model.evil.model = nn.Module()
        model.evil.model.layers = nn.ModuleList([nn.Module()])
        model.evil.model.layers[0].linear_attn = nn.Module()
        target = nn.Module()
        model.evil.model.layers[0].linear_attn.in_proj_qkv = target
        target.lora_A = nn.ModuleDict({"default": nn.Linear(2, 1, bias=False)})
        with self.assertRaisesRegex(TrainingImmutabilityError, "non-allowlisted"):
            assert_lora_only_trainable(model, manifest)

    def test_detaching_original_base_tensor_inventory_fails(self):
        model = self.build_model()
        manifest = self.manifest(model)
        proof = self.snapshot(model, manifest)
        self.add_adapters(model)
        model.model.layers[0].mlp.weight = nn.Parameter(
            model.model.layers[0].mlp.weight.detach().clone(), requires_grad=False
        )
        with self.assertRaisesRegex(TrainingImmutabilityError, "no longer attached"):
            assert_base_model_immutable(proof, model, manifest)

    def test_snapshot_and_manifest_substitution_fail(self):
        model = self.build_model()
        manifest = self.manifest(model)
        proof = self.snapshot(model, manifest)
        self.add_adapters(model)
        substituted = build_lora_target_manifest(
            model, model_manifest_digest=digest_for("substituted-model-manifest")
        )
        with self.assertRaisesRegex(TrainingImmutabilityError, "different target/model manifest"):
            assert_base_model_immutable(proof, model, substituted)
        with self.assertRaises(TypeError):
            proof.file_hashes["weights.bin"] = "0" * 64
        object.__setattr__(proof, "model_manifest_digest", "0" * 64)
        with self.assertRaisesRegex(TrainingImmutabilityError, "target/model manifest|metadata"):
            assert_base_model_immutable(proof, model, manifest)

    def test_per_tensor_hashes_change_and_reject_empty_state(self):
        model = self.build_model()
        before = model_tensor_hashes(model)
        with torch.no_grad():
            model.model.layers[0].mlp.bias.add_(3)
        after = model_tensor_hashes(model)
        self.assertEqual(before.keys(), after.keys())
        self.assertNotEqual(before["model.layers.0.mlp.bias"], after["model.layers.0.mlp.bias"])
        with self.assertRaisesRegex(Exception, "no measurable tensors"):
            model_tensor_hashes(nn.Module())


if __name__ == "__main__":
    unittest.main()
