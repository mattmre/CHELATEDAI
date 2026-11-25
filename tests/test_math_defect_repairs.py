"""Regression tests for verified math defects D-03, D-04, D-06, D-07."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np
import torch

from antigravity_engine import AntigravityEngine
from chelation_adapter import BoundedAdapter, LowRankAffineAdapter
from sedimentation_loss import (
    FilteredClippedDistillLoss,
    OPSDAymmetricPrivilegedLoss,
    OPSDJSDSimilarityDistillLoss,
)


class TestSedimentationLossMathFixes(unittest.TestCase):
    def test_jsd_loss_is_zero_at_identity_and_sensitive_to_neighbors(self):
        torch.manual_seed(0)
        x = torch.randn(6, 16)
        loss_fn = OPSDJSDSimilarityDistillLoss(temperature=0.07, beta=0.3, lambda_ret=0.2)
        at_identity = loss_fn(x, x.clone(), x.clone())
        self.assertAlmostEqual(float(at_identity.item()), 0.0, places=6)

        perturbed = x.clone()
        perturbed[0] = -perturbed[0]
        changed = loss_fn(perturbed, x, x)
        self.assertGreater(float(changed.item()), 0.0)

        # Singleton batch guard
        single = loss_fn(x[:1], x[:1], x[:1])
        self.assertAlmostEqual(float(single.item()), 0.0, places=6)

    def test_asymmetric_loss_has_nonzero_gradient_above_clip_delta(self):
        base = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        # Orthogonal vector -> ||student_norm - base_norm|| = sqrt(2) > clip_delta=0.5
        student = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
        loss_fn = OPSDAymmetricPrivilegedLoss(lambda_ret=0.0, lambda_reg=0.0, clip_delta=0.5)
        # Zero out corr_loss contribution by checking excess delta directly via isolated backward:
        # Compare clip_delta=0.5 vs clip_delta=5.0 -> penalty must be strictly larger for clip_delta=0.5
        loss_small_cap = loss_fn(student, base, student.detach())
        loss_small_cap.backward()
        grad_norm = float(student.grad.norm().item())
        self.assertGreater(grad_norm, 0.0)

        student2 = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
        loss_large_cap = OPSDAymmetricPrivilegedLoss(lambda_ret=0.0, lambda_reg=0.0, clip_delta=5.0)(
            student2, base, student2.detach()
        )
        self.assertGreater(float(loss_small_cap.item()), float(loss_large_cap.item()))

    def test_filtered_clipped_loss_bounds_and_all_zero_mask(self):
        torch.manual_seed(1)
        s = torch.randn(4, 8, requires_grad=True)
        t = torch.randn(4, 8)
        b = s.detach().clone()
        loss_fn = FilteredClippedDistillLoss(r_min=0.05, r_max=5.0, tau=2.0, kl_beta=0.05)

        # Realistic positive gain=0.02, delta_norm=0.1 -> ratio=0.2 in (0.05, 5.0) => accepted!
        gains = torch.tensor([0.02, -0.01, 0.02, 0.02])
        norms = torch.tensor([0.1, 0.1, 0.1, 0.1])
        quants = torch.tensor([1, 1, 0, 1])
        val = loss_fn(s, t, gains, norms, quants, b)
        self.assertTrue(torch.isfinite(val).item())
        self.assertGreater(float(val.item()), 0.0)

        # All-zero mask remains finite (no NaN divide-by-zero)
        val_zero = loss_fn(s, t, torch.full((4,), -1.0), norms, quants, b)
        self.assertTrue(torch.isfinite(val_zero).item())


class TestAdapterAndMaskMathFixes(unittest.TestCase):
    def test_bounded_adapter_preserves_near_identity_when_floor_disabled(self):
        base = LowRankAffineAdapter(input_dim=16, rank=4)
        with torch.no_grad():
            base.V.fill_(1e-8)
        x = torch.randn(4, 16)
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)

        # min_correction=0.0 disables floor automatically
        bounded_zero = BoundedAdapter(base, min_correction=0.0, max_correction=0.5)
        with torch.no_grad():
            out_zero = bounded_zero(x)
        self.assertLess(torch.norm(out_zero - x_norm, dim=1).max().item(), 1e-4)

        # Explicit enforce_min_floor=False disables floor even for larger tiny deltas
        with torch.no_grad():
            base.V.fill_(1e-3)
        bounded_no_floor = BoundedAdapter(base, min_correction=0.05, max_correction=0.5, enforce_min_floor=False)
        with torch.no_grad():
            out_nf = bounded_no_floor(x)
        self.assertLess(torch.norm(out_nf - x_norm, dim=1).max().item(), 0.01)

    def test_chelate_toxicity_and_get_chelated_vector_never_return_zero_vector(self):
        engine = AntigravityEngine.__new__(AntigravityEngine)
        engine.vector_size = 8
        engine.chelation_p = 85
        engine.invert_chelation = False
        engine.collection_name = "test"
        engine.logger = MagicMock()

        # Singleton cluster -> all-ones fallback
        mask_single = engine._chelate_toxicity(np.array([[1.0] * 8]))
        self.assertTrue(np.allclose(mask_single, np.ones(8)))

        # Constant/equal-variance cluster -> all-ones fallback (previously all-zeros!)
        same_cluster = np.tile(np.linspace(0.1, 0.8, 8), (5, 1))
        mask_const = engine._chelate_toxicity(same_cluster)
        self.assertTrue(np.allclose(mask_const, np.ones(8)))

        # Inverted branch on constant cluster -> also all-ones fallback
        engine.invert_chelation = True
        mask_inv = engine._chelate_toxicity(same_cluster)
        self.assertTrue(np.allclose(mask_inv, np.ones(8)))

        # get_chelated_vector returns unit-norm vector on degenerate cluster
        q = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=float)
        q = q / np.linalg.norm(q)
        engine.embed = MagicMock(return_value=np.array([q]))
        hit = MagicMock()
        hit.vector = same_cluster[0].tolist()
        resp = MagicMock()
        resp.points = [hit, hit, hit]
        engine.qdrant = MagicMock()
        engine.qdrant.query_points.return_value = resp
        out = engine.get_chelated_vector("hello")
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
