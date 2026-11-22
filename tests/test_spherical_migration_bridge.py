"""Unit tests for the packaged Flagship #1 closed-form spherical migration stack."""

from __future__ import annotations

import hashlib
import unittest

import numpy as np

from representation_space import (
    DistanceMetric,
    Normalization,
    PoolingStrategy,
    RepresentationDescriptor,
    RepresentationRole,
    ResidentIndexDescriptor,
    VectorDType,
)
from semantic_cache_resolver import ResolutionMode
from spherical_migration_bridge import (
    CenteredProcrustesBridge,
    WhitenedSphericalProcrustesBridge,
    build_bridge_contract,
    combine_exposure_and_distortion_order,
    compose_centered_bridges,
    compute_spherical_distortion_scores,
    l2_normalize,
    resolve_with_tra1,
    score_quantile_calibrated_backfill,
    select_kcenter_anchors,
    select_spherical_kmeanspp_anchors,
)


def _h64(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _mk_desc(enc: str, dim: int, role: RepresentationRole) -> RepresentationDescriptor:
    return RepresentationDescriptor(
        encoder_id=enc,
        encoder_revision="r1",
        encoder_weights_sha256=_h64(f"w:{enc}"),
        tokenizer_id=enc,
        tokenizer_sha256=_h64(f"t:{enc}"),
        role=role,
        compatibility_domain_id="bench",
        compatibility_domain_sha256=_h64("dom:bench"),
        instruction_sha256=_h64(f"inst:{enc}:{role.value}"),
        pooling=PoolingStrategy.MEAN,
        dimension=dim,
        preprocessing_id="std",
        preprocessing_sha256=_h64("prep:std"),
        projection_chain_id="none",
        projection_chain_sha256=_h64("proj:none"),
        adapter_chain_id="none",
        adapter_chain_sha256=_h64("ad:none"),
        quantization_id="fp32",
        quantization_sha256=_h64("q:fp32"),
        normalization=Normalization.L2,
        metric=DistanceMetric.COSINE,
        dtype=VectorDType.FLOAT32,
    )


class TestSphericalMigrationBridge(unittest.TestCase):
    def test_centered_procrustes_recovers_shifted_cone_rotation_and_composes_isometrically(self):
        rng = np.random.default_rng(7)
        # Simulate narrow shifted cone (like BGE/E5) mapped to isotropic target via orthogonal rotation
        base = rng.normal(scale=0.15, size=(128, 16))
        cone_shift = np.zeros((1, 16))
        cone_shift[0, 0] = 2.0
        x_cone = base + cone_shift
        q1, _ = np.linalg.qr(rng.normal(size=(16, 16)))
        q2, _ = np.linalg.qr(rng.normal(size=(16, 16)))
        y_mid = (x_cone - x_cone.mean(axis=0, keepdims=True)) @ q1 + np.array([[0.0, 1.5] + [0.0] * 14])
        y_end = (y_mid - y_mid.mean(axis=0, keepdims=True)) @ q2

        b1 = CenteredProcrustesBridge.fit(x_cone, y_mid)
        b2 = CenteredProcrustesBridge.fit(y_mid, y_end)
        self.assertAlmostEqual(b1.condition_number, 1.0, places=8)
        self.assertAlmostEqual(b2.condition_number, 1.0, places=8)

        b_comp = compose_centered_bridges([b1, b2])
        self.assertAlmostEqual(b_comp.condition_number, 1.0, places=8)
        pred = b_comp.transform(x_cone, normalize=False)
        self.assertLess(float(np.max(np.abs(pred - y_end))), 1e-10)

    def test_conformal_radius_scaling_recovers_narrow_cone_to_isotropic_multihop(self):
        rng = np.random.default_rng(13)
        # Simulate extreme narrow cone (radius 0.4) -> medium cone (0.65) -> isotropic (1.0)
        z = rng.normal(size=(160, 16))
        x_narrow = 0.4 * z + np.array([[2.0] + [0.0] * 15])
        q1, _ = np.linalg.qr(rng.normal(size=(16, 16)))
        q2, _ = np.linalg.qr(rng.normal(size=(16, 16)))
        y_mid = 0.65 * (z @ q1) + np.array([[0.0, 1.5] + [0.0] * 14])
        y_iso = 1.0 * ((z @ q1) @ q2) + np.array([[0.0, 0.0, 0.4] + [0.0] * 13])

        b1 = CenteredProcrustesBridge.fit(x_narrow, y_mid, match_radius=True)
        b2 = CenteredProcrustesBridge.fit(y_mid, y_iso, match_radius=True)
        b_comp = compose_centered_bridges([b1, b2])

        self.assertAlmostEqual(b_comp.condition_number, 1.0, places=8)
        self.assertAlmostEqual(b_comp.scale, 1.0 / 0.4, places=5)
        pred = b_comp.transform(x_narrow, normalize=False)
        self.assertLess(float(np.max(np.abs(pred - y_iso))), 1e-10)

    def test_whitened_spherical_procrustes_recovers_anisotropic_stretch(self):
        rng = np.random.default_rng(11)
        z = rng.normal(size=(256, 12))
        scales = np.linspace(0.2, 3.0, 12)
        q_rot, _ = np.linalg.qr(rng.normal(size=(12, 12)))
        x = (z * scales) @ q_rot + 0.5
        y = z + 0.1
        bridge = WhitenedSphericalProcrustesBridge.fit(x, y, shrink=0.01)
        pred = bridge.transform(x, normalize=True)
        y_n = l2_normalize(y)
        mean_cos = float(np.mean(np.sum(pred * y_n, axis=1)))
        self.assertGreater(mean_cos, 0.99)

    def test_anchor_selectors_are_deterministic_and_unique(self):
        rng = np.random.default_rng(19)
        vecs = l2_normalize(rng.normal(size=(200, 16)))
        km1 = select_spherical_kmeanspp_anchors(vecs, 32, seed=42)
        km2 = select_spherical_kmeanspp_anchors(vecs, 32, seed=42)
        self.assertTrue(np.array_equal(km1, km2))
        self.assertEqual(len(km1), 32)
        self.assertEqual(len(set(km1.tolist())), 32)

        kc1 = select_kcenter_anchors(vecs, 32, seed=42)
        kc2 = select_kcenter_anchors(vecs, 32, seed=42)
        self.assertTrue(np.array_equal(kc1, kc2))
        self.assertEqual(len(kc1), 32)
        self.assertEqual(len(set(kc1.tolist())), 32)

    def test_quantile_calibrated_backfill_and_distortion_order(self):
        rng = np.random.default_rng(23)
        d_old = l2_normalize(rng.normal(size=(60, 16)))
        # Create a narrow-cone shifted d_new whose raw cosines are systematically +0.7 higher
        d_new = l2_normalize(d_old * 0.2 + np.eye(1, 16)[0])
        q_new = d_new[:10].copy()
        q_map = d_old[:10].copy()
        doc_ids = [f"d{i}" for i in range(60)]
        qids = [f"q{i}" for i in range(10)]

        dist_order, dist_scores = compute_spherical_distortion_scores(
            d_new, d_old, d_old, anchor_pool_idx=np.arange(30)
        )
        self.assertEqual(len(dist_order), 60)
        self.assertTrue(np.all(np.isfinite(dist_scores)))

        hybrid = combine_exposure_and_distortion_order(np.arange(60), dist_order)
        self.assertEqual(len(hybrid), 60)
        self.assertEqual(len(set(hybrid.tolist())), 60)

        rankings = score_quantile_calibrated_backfill(
            q_new, q_map, d_old, d_new, backfill_indices=hybrid[:12],
            query_ids=qids, doc_ids=doc_ids,
            dev_queries_new=q_new, dev_queries_mapped=q_map, k=5,
        )
        self.assertEqual(len(rankings), 10)
        for qid, rlist in rankings.items():
            self.assertEqual(len(rlist), 5)
            self.assertEqual(len(set(rlist)), 5)

    def test_tra1_bridge_contract_authorizes_valid_and_rejects_underpowered(self):
        rng = np.random.default_rng(31)
        x = l2_normalize(rng.normal(size=(64, 16)))
        y = l2_normalize(rng.normal(size=(64, 16)))
        bridge_ok = WhitenedSphericalProcrustesBridge.fit(x, y, shrink=0.2)
        bridge_weak = WhitenedSphericalProcrustesBridge.fit(x[:8], y[:8], shrink=0.2)

        q_e1 = _mk_desc("E1", 16, RepresentationRole.QUERY)
        q_e0 = _mk_desc("E0", 16, RepresentationRole.QUERY)
        d_e0 = _mk_desc("E0", 16, RepresentationRole.DOCUMENT)
        resident = ResidentIndexDescriptor(
            index_id="idx-e0",
            document_space=d_e0,
            query_space=q_e0,
            corpus_snapshot_id="snap0",
            corpus_snapshot_sha256=_h64("snap0"),
            vector_build_id="build0",
            vector_build_sha256=_h64("build0"),
            document_count=64,
        )

        c_ok = build_bridge_contract(
            source=q_e1, target=q_e0, bridge=bridge_ok,
            transform_id="t-ok", anchor_manifest_sha256=_h64("anc-ok"),
            validation_count=32, validation_confidence=0.92,
            evidence_id="ev-ok", evidence_sha256=_h64("ev-ok"),
        )
        dec_ok = resolve_with_tra1(
            query_space=q_e1, resident_index=resident, bridges=[c_ok],
            trusted_evidence_sha256s=[c_ok.evidence_sha256],
            trusted_bridge_contract_ids=[c_ok.stable_id],
        )
        self.assertEqual(dec_ok.mode, ResolutionMode.REVERSE)
        self.assertTrue(dec_ok.can_search)

        c_weak = build_bridge_contract(
            source=q_e1, target=q_e0, bridge=bridge_weak,
            transform_id="t-weak", anchor_manifest_sha256=_h64("anc-weak"),
            validation_count=4, validation_confidence=0.92,
            evidence_id="ev-weak", evidence_sha256=_h64("ev-weak"),
        )
        dec_weak = resolve_with_tra1(
            query_space=q_e1, resident_index=resident, bridges=[c_weak],
            trusted_evidence_sha256s=[c_weak.evidence_sha256],
            trusted_bridge_contract_ids=[c_weak.stable_id],
        )
        self.assertEqual(dec_weak.mode, ResolutionMode.ABSTAIN)
        self.assertFalse(dec_weak.can_search)

    def test_audit_edge_cases_constant_target_ties_and_pool_cap(self):
        # 1. Constant target -> scale=0 -> condition_number must be inf (never 0.0)
        rng = np.random.default_rng(77)
        x = l2_normalize(rng.normal(size=(32, 8)))
        y_const = np.tile(np.array([[1.0] + [0.0] * 7]), (32, 1))
        b_const = CenteredProcrustesBridge.fit(x, y_const, match_radius=True)
        self.assertEqual(b_const.scale, 0.0)
        self.assertEqual(b_const.condition_number, float("inf"))

        # 2. Requesting n_anchors > candidate_pool_cap returns exact unique count without error
        vecs = l2_normalize(rng.normal(size=(120, 8)))
        km = select_spherical_kmeanspp_anchors(vecs, 80, seed=5, candidate_pool_cap=32)
        kc = select_kcenter_anchors(vecs, 80, seed=5, candidate_pool_cap=32)
        self.assertEqual(len(km), 80)
        self.assertEqual(len(set(km.tolist())), 80)
        self.assertEqual(len(kc), 80)
        self.assertEqual(len(set(kc.tolist())), 80)

        # 3. 160 equal-score docs in reverse ID order must return lexicographically first "000" at rank 1
        from m1_baseline_ladder import score_dense_rankings

        d_eq = np.tile(np.array([[1.0] + [0.0] * 7]), (160, 1))
        q_eq = np.array([[1.0] + [0.0] * 7])
        rev_ids = [f"{i:03d}" for i in reversed(range(160))]
        rk_dense, _ = score_dense_rankings(q_eq, d_eq, ["q0"], rev_ids, k=10)
        self.assertEqual(rk_dense["q0"][0], "000")
        self.assertEqual(rk_dense["q0"][:5], ["000", "001", "002", "003", "004"])

        rk_quant = score_quantile_calibrated_backfill(
            q_eq, q_eq, d_eq, d_eq, backfill_indices=[0, 1, 2],
            query_ids=["q0"], doc_ids=rev_ids, dev_queries_new=q_eq, dev_queries_mapped=q_eq, k=10,
        )
        self.assertEqual(rk_quant["q0"][0], "000")
        self.assertEqual(rk_quant["q0"][:5], ["000", "001", "002", "003", "004"])


if __name__ == "__main__":
    unittest.main()
