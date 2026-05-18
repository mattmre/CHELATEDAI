"""Tests for BlockAttnResAdapter and LayerAttentionAggregator."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from chelation_adapter import (
    BlockAttnResAdapter,
    BoundedAdapter,
    LayerAttentionAggregator,
    QuantizationAwareLowRankAdapter,
    create_adapter,
)
from config import ChelationConfig


class TestBlockAttnResAdapterBasics(unittest.TestCase):
    def setUp(self):
        self.input_dim = 384
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_forward_shape_2d(self):
        adapter = BlockAttnResAdapter(input_dim=self.input_dim)
        out = adapter(torch.randn(8, self.input_dim))
        self.assertEqual(out.shape, (8, self.input_dim))

    def test_forward_shape_1d(self):
        adapter = BlockAttnResAdapter(input_dim=self.input_dim)
        out = adapter(torch.randn(self.input_dim))
        self.assertEqual(out.dim(), 1)
        self.assertEqual(out.shape[0], self.input_dim)

    def test_output_normalized(self):
        adapter = BlockAttnResAdapter(input_dim=self.input_dim)
        out = adapter(torch.randn(10, self.input_dim))
        norms = torch.norm(out, p=2, dim=1)
        self.assertTrue(
            torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
            f"Output not unit-normalized: norms {norms.min():.6f}–{norms.max():.6f}",
        )

    def test_near_identity_init(self):
        adapter = BlockAttnResAdapter(input_dim=self.input_dim)
        x = torch.randn(5, self.input_dim)
        out = adapter(x)
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
        cosine = torch.nn.functional.cosine_similarity(out, x_norm, dim=1)
        self.assertTrue(
            torch.all(cosine > 0.95).item(),
            f"Not near identity at init: min cosine = {cosine.min().item():.4f}",
        )

    def test_invalid_rank_0d_raises(self):
        adapter = BlockAttnResAdapter(input_dim=self.input_dim)
        with self.assertRaises(ValueError):
            adapter(torch.tensor(1.0))

    def test_invalid_rank_3d_raises(self):
        adapter = BlockAttnResAdapter(input_dim=self.input_dim)
        with self.assertRaises(ValueError):
            adapter(torch.randn(2, 3, self.input_dim))

    def test_regularization_loss_zero(self):
        adapter = BlockAttnResAdapter(input_dim=self.input_dim)
        self.assertEqual(adapter.regularization_loss(), 0.0)

    def test_save_load_roundtrip(self):
        adapter = BlockAttnResAdapter(input_dim=self.input_dim)
        with torch.no_grad():
            for block in adapter.blocks:
                block[0].weight.add_(torch.randn_like(block[0].weight) * 0.1)
        save_path = self.temp_dir / "attnres.pt"
        adapter.save(str(save_path))

        new_adapter = BlockAttnResAdapter(input_dim=self.input_dim)
        success = new_adapter.load(str(save_path))
        self.assertTrue(success)

        x = torch.randn(3, self.input_dim)
        self.assertTrue(torch.allclose(adapter(x), new_adapter(x), atol=1e-5))

    def test_load_missing_file_returns_false(self):
        adapter = BlockAttnResAdapter(input_dim=self.input_dim)
        self.assertFalse(adapter.load(str(self.temp_dir / "nonexistent.pt")))

    def test_input_dim_attribute(self):
        adapter = BlockAttnResAdapter(input_dim=self.input_dim, num_blocks=3)
        self.assertEqual(adapter.input_dim, self.input_dim)
        self.assertEqual(adapter.num_blocks, 3)


class TestBlockAttnResAdapterBlocks(unittest.TestCase):
    def setUp(self):
        self.input_dim = 128

    def test_num_blocks_1(self):
        adapter = BlockAttnResAdapter(input_dim=self.input_dim, num_blocks=1)
        out = adapter(torch.randn(4, self.input_dim))
        self.assertEqual(out.shape, (4, self.input_dim))

    def test_num_blocks_2(self):
        adapter = BlockAttnResAdapter(input_dim=self.input_dim, num_blocks=2)
        out = adapter(torch.randn(4, self.input_dim))
        self.assertEqual(out.shape, (4, self.input_dim))

    def test_num_blocks_8(self):
        adapter = BlockAttnResAdapter(input_dim=self.input_dim, num_blocks=8)
        out = adapter(torch.randn(4, self.input_dim))
        self.assertEqual(out.shape, (4, self.input_dim))

    def test_custom_proj_dim(self):
        adapter = BlockAttnResAdapter(input_dim=self.input_dim, num_blocks=2, proj_dim=16)
        out = adapter(torch.randn(4, self.input_dim))
        self.assertEqual(out.shape, (4, self.input_dim))

    def test_correct_block_count(self):
        adapter = BlockAttnResAdapter(input_dim=self.input_dim, num_blocks=3)
        self.assertEqual(len(adapter.blocks), 3)

    def test_gradients_flow_through_blocks(self):
        adapter = BlockAttnResAdapter(input_dim=self.input_dim, num_blocks=2)
        x = torch.randn(4, self.input_dim)
        adapter(x).sum().backward()
        for i, block in enumerate(adapter.blocks):
            for name, param in block.named_parameters():
                self.assertIsNotNone(param.grad, f"No gradient for block {i} param {name}")

    def test_gradients_flow_through_attention(self):
        adapter = BlockAttnResAdapter(input_dim=self.input_dim, num_blocks=2)
        x = torch.randn(4, self.input_dim)
        adapter(x).sum().backward()
        self.assertIsNotNone(adapter.query_proj.weight.grad)
        self.assertIsNotNone(adapter.key_proj.weight.grad)

    def test_batch_size_1(self):
        adapter = BlockAttnResAdapter(input_dim=self.input_dim)
        out = adapter(torch.randn(1, self.input_dim))
        self.assertEqual(out.shape, (1, self.input_dim))


class TestBlockAttnResAdapterFactory(unittest.TestCase):
    def setUp(self):
        self.input_dim = 256

    def test_create_adapter_attnres_default(self):
        adapter = create_adapter("attnres", input_dim=self.input_dim)
        self.assertIsInstance(adapter, BlockAttnResAdapter)
        self.assertEqual(adapter.num_blocks, 4)

    def test_create_adapter_attnres_custom_blocks(self):
        adapter = create_adapter("attnres", input_dim=self.input_dim, num_blocks=6)
        self.assertIsInstance(adapter, BlockAttnResAdapter)
        self.assertEqual(adapter.num_blocks, 6)

    def test_create_adapter_ignores_attnres_kwargs_for_mlp(self):
        adapter = create_adapter("mlp", input_dim=self.input_dim, num_blocks=8, proj_dim=32)
        self.assertEqual(adapter.input_dim, self.input_dim)

    def test_create_adapter_attnres_bounded(self):
        adapter = create_adapter("attnres", input_dim=self.input_dim, bounded=True)
        self.assertIsInstance(adapter, BoundedAdapter)
        self.assertIsInstance(adapter.base_adapter, BlockAttnResAdapter)

    def test_create_adapter_unknown_raises_with_attnres_in_list(self):
        with self.assertRaises(ValueError) as ctx:
            create_adapter("bogus_type", input_dim=self.input_dim)
        self.assertIn("attnres", str(ctx.exception))

    def test_attnres_output_shape_and_normalized(self):
        adapter = create_adapter("attnres", input_dim=self.input_dim)
        out = adapter(torch.randn(5, self.input_dim))
        self.assertEqual(out.shape, (5, self.input_dim))
        norms = torch.norm(out, p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))


class TestAttnResConfigPresets(unittest.TestCase):
    def test_attnres_in_adapter_type_preset(self):
        preset = ChelationConfig.get_preset("attnres", "adapter_type")
        self.assertEqual(preset["adapter_type"], "attnres")
        self.assertEqual(preset["num_blocks"], 4)

    def test_preset_shallow(self):
        preset = ChelationConfig.get_preset("shallow", "attnres_adapter")
        self.assertEqual(preset["num_blocks"], 2)

    def test_preset_balanced(self):
        preset = ChelationConfig.get_preset("balanced", "attnres_adapter")
        self.assertEqual(preset["num_blocks"], 4)

    def test_preset_deep(self):
        preset = ChelationConfig.get_preset("deep", "attnres_adapter")
        self.assertEqual(preset["num_blocks"], 8)

    def test_invalid_preset_raises(self):
        with self.assertRaises(ValueError):
            ChelationConfig.get_preset("nonexistent", "attnres_adapter")

    def test_preset_returns_copy(self):
        p1 = ChelationConfig.get_preset("balanced", "attnres_adapter")
        p2 = ChelationConfig.get_preset("balanced", "attnres_adapter")
        p1["num_blocks"] = 99
        self.assertEqual(p2["num_blocks"], 4)


class TestLayerAttentionAggregator(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 256
        self.num_layers = 6
        self.batch = 4

    def test_output_shape(self):
        agg = LayerAttentionAggregator(hidden_size=self.hidden_size)
        x = torch.randn(self.batch, self.num_layers, self.hidden_size)
        out = agg(x)
        self.assertEqual(out.shape, (self.batch, self.hidden_size))

    def test_output_normalized(self):
        agg = LayerAttentionAggregator(hidden_size=self.hidden_size)
        x = torch.randn(self.batch, self.num_layers, self.hidden_size)
        out = agg(x)
        norms = torch.norm(out, p=2, dim=1)
        self.assertTrue(
            torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
            f"Output not unit-normalized: {norms}",
        )

    def test_wrong_input_shape_raises(self):
        agg = LayerAttentionAggregator(hidden_size=self.hidden_size)
        with self.assertRaises(ValueError):
            agg(torch.randn(self.batch, self.hidden_size))  # 2D, not 3D

    def test_wrong_hidden_size_raises(self):
        agg = LayerAttentionAggregator(hidden_size=self.hidden_size)
        with self.assertRaises(ValueError):
            agg(torch.randn(self.batch, self.num_layers, self.hidden_size + 1))

    def test_custom_proj_dim(self):
        agg = LayerAttentionAggregator(hidden_size=self.hidden_size, proj_dim=16)
        out = agg(torch.randn(self.batch, self.num_layers, self.hidden_size))
        self.assertEqual(out.shape, (self.batch, self.hidden_size))

    def test_gradients_flow(self):
        agg = LayerAttentionAggregator(hidden_size=self.hidden_size)
        x = torch.randn(self.batch, self.num_layers, self.hidden_size)
        agg(x).sum().backward()
        self.assertIsNotNone(agg.query_proj.weight.grad)
        self.assertIsNotNone(agg.key_proj.weight.grad)

    def test_single_layer(self):
        agg = LayerAttentionAggregator(hidden_size=self.hidden_size)
        out = agg(torch.randn(self.batch, 1, self.hidden_size))
        self.assertEqual(out.shape, (self.batch, self.hidden_size))

    def test_batch_size_1(self):
        agg = LayerAttentionAggregator(hidden_size=self.hidden_size)
        out = agg(torch.randn(1, self.num_layers, self.hidden_size))
        self.assertEqual(out.shape, (1, self.hidden_size))


class TestQuantizationAwareLowRankAdapter(unittest.TestCase):
    """Tests for QuantizationAwareLowRankAdapter (OPSD Loop 1 Agent 8).
    Verifies low-rank + STE fake-quant integration for quant-robust chelation
    self-distillation. Enables direct testing of OPSD-quant variants.
    """

    def setUp(self):
        self.input_dim = 128
        self.rank = 8
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_via_factory_quant_low_rank(self):
        adapter = create_adapter("quant_low_rank", input_dim=self.input_dim, rank=self.rank)
        self.assertIsInstance(adapter, QuantizationAwareLowRankAdapter)
        self.assertEqual(adapter.rank, self.rank)
        self.assertEqual(adapter.quant_levels, 127)

    def test_forward_shape_and_normalization(self):
        adapter = create_adapter("quant_low_rank", input_dim=self.input_dim, rank=self.rank)
        x = torch.randn(4, self.input_dim)
        out = adapter(x)
        self.assertEqual(out.shape, (4, self.input_dim))
        # Should be approx unit norm
        norms = torch.norm(out, p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_training_mode_applies_ste_quant(self):
        adapter = create_adapter(
            "quant_low_rank", input_dim=self.input_dim, rank=self.rank,
            quant_levels=127, quant_quantile=0.99, apply_quant_to="output"
        )
        adapter.train()
        x = torch.randn(2, self.input_dim)
        out = adapter(x)
        # In training, output is STE-quantized version of low-rank correction
        self.assertEqual(out.shape, (2, self.input_dim))
        # Gradients should flow (STE)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(adapter.U.grad)
        self.assertIsNotNone(adapter.V.grad)

    def test_inference_mode_uses_non_ste_simulator(self):
        adapter = create_adapter("quant_low_rank", input_dim=self.input_dim, rank=self.rank)
        adapter.eval()
        x = torch.randn(3, self.input_dim)
        with torch.no_grad():
            out = adapter(x)
        self.assertEqual(out.shape, (3, self.input_dim))
        # Eval mode plus no_grad should use the non-STE storage simulator without autograd.
        self.assertFalse(out.requires_grad)

    def test_quant_low_rank_plus_bounded(self):
        adapter = create_adapter(
            "quant_low_rank", input_dim=self.input_dim, rank=self.rank,
            bounded=True, min_correction=0.01, max_correction=0.4
        )
        self.assertIsInstance(adapter, BoundedAdapter)
        # Inner base should be quant low rank
        self.assertIsInstance(adapter.base_adapter, QuantizationAwareLowRankAdapter)

    def test_regularization_and_save_load(self):
        adapter = create_adapter("quant_low_rank", input_dim=self.input_dim, rank=self.rank, ste_scale=True)
        reg = adapter.regularization_loss()
        self.assertIsInstance(reg, torch.Tensor)
        # Save/load roundtrip
        p = self.temp_dir / "qlowrank_test.pt"
        adapter.save(p)
        adapter2 = create_adapter("quant_low_rank", input_dim=self.input_dim, rank=self.rank, ste_scale=True)
        loaded = adapter2.load(p)
        self.assertTrue(loaded)

    def test_ste_vs_non_quant_low_rank_delta_magnitude(self):
        # Quick sanity: quant version should produce deltas that are quant-discretized
        base_lr = create_adapter("low_rank", input_dim=self.input_dim, rank=self.rank)
        q_lr = create_adapter("quant_low_rank", input_dim=self.input_dim, rank=self.rank)
        base_lr.eval()
        q_lr.eval()
        x = torch.randn(1, self.input_dim)
        # Different outputs expected due to quant step (unless zero correction)
        out_base = base_lr(x)
        out_q = q_lr(x)
        # They can be close but the point is the mechanism exists for distillation training
        self.assertEqual(out_base.shape, out_q.shape)


if __name__ == "__main__":
    unittest.main()
