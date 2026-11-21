"""Unit tests for Stage 1 M1 baseline ladder, SVD Procrustes, and fixed M2 gap preflight."""

from __future__ import annotations

import unittest

import numpy as np

from m1_baseline_ladder import (
    emit_candidate_row,
    fit_closed_form_ridge,
    fit_closed_form_svd_procrustes,
    m2_gap_preflight,
    required_m1_rungs,
    split_query_ids,
)


class TestM1M2Ladder(unittest.TestCase):
    def test_emit_candidate_row_fails_closed_when_any_rung_missing(self):
        required = required_m1_rungs(old_encoder_retired=True, budget=0.0)
        ladder = {
            rung: {
                "split_sha256": "s1",
                "metric_contract_sha256": "m1",
                "mean_ndcg_at_10": 0.5,
                "status": "OK",
            }
            for rung in required
        }
        # Complete ladder succeeds
        emitted = emit_candidate_row(
            candidate_name="test_cand",
            candidate_row={"mean_ndcg_at_10": 0.6},
            ladder_rows=ladder,
            required_rungs=required,
            split_sha256="s1",
            metric_contract_sha256="m1",
        )
        self.assertEqual(emitted["candidate_name"], "test_cand")

        # Removing any single rung fails closed
        del ladder["closed_form_svd_procrustes"]
        with self.assertRaises(RuntimeError):
            emit_candidate_row(
                candidate_name="test_cand",
                candidate_row={"mean_ndcg_at_10": 0.6},
                ladder_rows=ladder,
                required_rungs=required,
                split_sha256="s1",
                metric_contract_sha256="m1",
            )

    def test_m2_admits_positive_gap_fixture(self):
        # Proves D-02 is fixed: reference=0.90, best affordable=0.70 => gap +0.20 => ADMIT
        qids = [f"q{i}" for i in range(40)]
        dev_scores = {
            "full_reembed_c2o": {qid: 0.90 for qid in qids},
            "closed_form_ridge": {qid: 0.70 for qid in qids},
            "closed_form_svd_procrustes": {qid: 0.65 for qid in qids},
        }
        res = m2_gap_preflight(
            dev_scores,
            split="DEVELOPMENT",
            reference_arm="full_reembed_c2o",
            eligible_arms=["closed_form_ridge", "closed_form_svd_procrustes"],
            delta=0.02,
            n_bootstrap=1000,
            seed=42,
        )
        self.assertEqual(res["decision"], "ADMIT")
        self.assertAlmostEqual(res["point_gap"], 0.20, places=10)
        self.assertGreater(res["lcb_95_fixed"], 0.0)
        self.assertGreater(res["lcb_95_selection_adjusted"], 0.0)

    def test_m2_refuses_null_h5_replay_fixture(self):
        qids = [f"q{i}" for i in range(40)]
        dev_scores = {
            "c5_living": {qid: 0.131 for qid in qids},
            "c5s_static": {qid: 0.131 for qid in qids},
        }
        res = m2_gap_preflight(
            dev_scores,
            split="DEVELOPMENT",
            reference_arm="c5_living",
            eligible_arms=["c5s_static"],
            delta=0.02,
            n_bootstrap=1000,
            seed=42,
        )
        self.assertEqual(res["decision"], "REFUSE")
        self.assertAlmostEqual(res["point_gap"], 0.0, places=12)

    def test_m2_rejects_reference_inside_eligible_and_non_dev_splits(self):
        dev_scores = {
            "full_reembed_c2o": {"q0": 0.9, "q1": 0.9},
            "closed_form_ridge": {"q0": 0.7, "q1": 0.7},
        }
        with self.assertRaises(ValueError):
            m2_gap_preflight(
                dev_scores,
                split="DEVELOPMENT",
                reference_arm="full_reembed_c2o",
                eligible_arms=["closed_form_ridge", "full_reembed_c2o"],
                delta=0.02,
                n_bootstrap=200,
            )
        for bad_split in ("SELECT", "REPORT"):
            with self.assertRaises(ValueError):
                m2_gap_preflight(
                    dev_scores,
                    split=bad_split,
                    reference_arm="full_reembed_c2o",
                    eligible_arms=["closed_form_ridge"],
                    delta=0.02,
                    n_bootstrap=200,
                )

    def test_closed_form_svd_procrustes_preserves_gram_and_recovers_rotation(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(64, 16))
        q_orth, _ = np.linalg.qr(rng.normal(size=(16, 16)))
        y = x @ q_orth
        w_hat = fit_closed_form_svd_procrustes(x, y)
        # Orthogonality & exact Gram preservation
        self.assertLess(float(np.max(np.abs(w_hat.T @ w_hat - np.eye(16)))), 1e-10)
        gram_err = float(np.max(np.abs((x @ w_hat) @ (x @ w_hat).T - x @ x.T)))
        self.assertLess(gram_err, 1e-10)
        self.assertLess(float(np.max(np.abs(x @ w_hat - y))), 1e-10)

        # Ridge also recovers closely on noiseless full-rank data
        w_ridge = fit_closed_form_ridge(x, y, regularization=1e-8)
        self.assertLess(float(np.max(np.abs(x @ w_ridge - y))), 1e-6)

    def test_deterministic_query_splits_are_disjoint(self):
        qids = [f"q{i}" for i in range(100)]
        s1 = split_query_ids(qids)
        s2 = split_query_ids(qids)
        self.assertEqual(s1, s2)
        all_ids = s1["DEVELOPMENT"] + s1["SELECT"] + s1["REPORT"]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(all_ids), set(qids))

    def test_d08_bridge_production_floor_fails_closed(self):
        import hashlib
        from representation_space import (
            BridgeContract,
            BridgeCost,
            BridgeValidationState,
            DistanceMetric,
            Normalization,
            PoolingStrategy,
            RepresentationDescriptor,
            RepresentationRole,
            ResidentIndexDescriptor,
            VectorDType,
        )
        from semantic_cache_resolver import ResolutionMode, SemanticCacheResolver

        def h64(tag: str) -> str:
            return hashlib.sha256(tag.encode("utf-8")).hexdigest()

        def mk_desc(enc: str, dim: int) -> RepresentationDescriptor:
            return RepresentationDescriptor(
                encoder_id=enc,
                encoder_revision="r1",
                encoder_weights_sha256=h64(f"w:{enc}"),
                tokenizer_id=enc,
                tokenizer_sha256=h64(f"t:{enc}"),
                role=RepresentationRole.SHARED,
                compatibility_domain_id="d1",
                compatibility_domain_sha256=h64("dom:d1"),
                instruction_sha256=h64(f"i:{enc}"),
                pooling=PoolingStrategy.MEAN,
                dimension=dim,
                preprocessing_id="p1",
                preprocessing_sha256=h64("prep:p1"),
                projection_chain_id="none",
                projection_chain_sha256=h64("proj:none"),
                adapter_chain_id="none",
                adapter_chain_sha256=h64("ad:none"),
                quantization_id="fp32",
                quantization_sha256=h64("q:fp32"),
                normalization=Normalization.L2,
                metric=DistanceMetric.COSINE,
                dtype=VectorDType.FLOAT32,
            )

        src = mk_desc("e1", 384)
        dst = mk_desc("e0", 384)
        resident = ResidentIndexDescriptor(
            index_id="idx0",
            document_space=dst,
            query_space=dst,
            corpus_snapshot_id="snap0",
            corpus_snapshot_sha256=h64("snap0"),
            vector_build_id="b0",
            vector_build_sha256=h64("b0"),
            document_count=100,
        )
        weak_bridge = BridgeContract(
            source=src,
            target=dst,
            role=RepresentationRole.SHARED,
            transform_family="procrustes",
            transform_id="weak",
            weights_sha256=h64("w:weak"),
            hyperparameters_sha256=h64("hp:weak"),
            fit_anchor_manifest_sha256=h64("anc:weak"),
            fit_count=1,
            validation_count=1,
            validation_state=BridgeValidationState.VALIDATED,
            cost=BridgeCost.ZERO_WRITE,
            evidence_id="ev:weak",
            evidence_sha256=h64("ev:weak"),
            validation_confidence=0.95,
        )
        self.assertFalse(weak_bridge.meets_production_floor())
        resolver = SemanticCacheResolver(
            bridges=[weak_bridge],
            minimum_route_confidence=0.70,
            trusted_evidence_sha256s=[weak_bridge.evidence_sha256],
            trusted_bridge_contract_ids=[weak_bridge.stable_id],
            enforce_production_floor=True,
        )
        decision = resolver.resolve(src, resident)
        self.assertEqual(decision.mode, ResolutionMode.ABSTAIN)
        self.assertFalse(decision.can_search)

        strong_bridge = BridgeContract(
            source=src,
            target=dst,
            role=RepresentationRole.SHARED,
            transform_family="procrustes",
            transform_id="strong",
            weights_sha256=h64("w:strong"),
            hyperparameters_sha256=h64("hp:strong"),
            fit_anchor_manifest_sha256=h64("anc:strong"),
            fit_count=64,
            validation_count=32,
            validation_state=BridgeValidationState.VALIDATED,
            cost=BridgeCost.ZERO_WRITE,
            evidence_id="ev:strong",
            evidence_sha256=h64("ev:strong"),
            validation_confidence=0.95,
        )
        self.assertTrue(strong_bridge.meets_production_floor())
        resolver_ok = SemanticCacheResolver(
            bridges=[strong_bridge],
            minimum_route_confidence=0.70,
            trusted_evidence_sha256s=[strong_bridge.evidence_sha256],
            trusted_bridge_contract_ids=[strong_bridge.stable_id],
            enforce_production_floor=True,
        )
        decision_ok = resolver_ok.resolve(src, resident)
        self.assertEqual(decision_ok.mode, ResolutionMode.REVERSE)
        self.assertTrue(decision_ok.can_search)


if __name__ == "__main__":
    unittest.main()
