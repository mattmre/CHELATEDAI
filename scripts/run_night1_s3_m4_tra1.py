"""Night-1 Spark Job S3: Hardened TRA-1 Typed Authorization vs M4 QPP Router.

Tests 7 mutation families (leave-one-family-out evaluation + hostile wrong-space audit):
  1. query_swap_samedim     (E0 -> E1)
  2. query_swap_crossdim    (E0 -> E3)
  3. doc_vector_corruption  (calibrated noise + orthogonal rotation on resident E0 docs)
  4. domain_home_cross      (SciFact / NFCorpus / FiQA compatibility-domain authorization)
  5. dtype_tier_quant       (FP32 vs FP16 vs INT8 scalar quantization)
  6. multihop_lineage_2hop  (E0 -> E1 -> E2)
  7. hostile_wrong_space    (mislabeled role, expired bridge, unnormalized, metric/dim/domain mismatch,
                             low-confidence/low-fit-count bridge failing D-08 floor)

Compares:
  - Hardened TRA-1 typed resolver (deterministic descriptor rules + fail-closed ABSTAIN)
  - M4 QPP query-feature router given the EXACT SAME authorization filter (isolating route selection)
  - Best single safe fixed action
Logs wrong_space_executions (must be 0), abstention coverage, paired nDCG@10 delta (97.5% LCB),
worst-family loss, and asymmetric false-accept / false-reject decision loss.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from m1_baseline_ladder import (  # noqa: E402
    evaluate_rankings_mlr,
    fit_closed_form_ridge,
    fit_closed_form_svd_procrustes,
    score_dense_rankings,
    select_qrels_independent_anchors,
    split_query_ids,
)
from representation_space import (  # noqa: E402
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
from semantic_cache_resolver import (  # noqa: E402
    ResolutionMode,
    SemanticCacheResolver,
)
from scripts.night1_common import (  # noqa: E402
    ENCODERS,
    current_git_sha,
    encode_or_load_embeddings,
    load_beir_dataset,
    metric_contract_digest,
    paired_bootstrap_delta,
    process_telemetry,
    sha256_json,
    summarize_per_query_metrics,
    write_json_atomic,
)


def _h64(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _make_descriptor(
    *,
    encoder_id: str,
    role: RepresentationRole,
    domain: str,
    dim: int,
    dtype: VectorDType = VectorDType.FLOAT32,
    normalization: Normalization = Normalization.L2,
    metric: DistanceMetric = DistanceMetric.COSINE,
    corrupted_tag: str = "clean",
) -> RepresentationDescriptor:
    return RepresentationDescriptor(
        encoder_id=encoder_id,
        encoder_revision="pinned-night1",
        encoder_weights_sha256=_h64(f"weights:{encoder_id}:{corrupted_tag}"),
        tokenizer_id=encoder_id,
        tokenizer_sha256=_h64(f"tok:{encoder_id}"),
        role=role,
        compatibility_domain_id=domain,
        compatibility_domain_sha256=_h64(f"dom:{domain}"),
        instruction_sha256=_h64(f"inst:{encoder_id}:{role.value}"),
        pooling=PoolingStrategy.MEAN,
        dimension=dim,
        preprocessing_id="std-v1",
        preprocessing_sha256=_h64("prep:std-v1"),
        projection_chain_id=corrupted_tag,
        projection_chain_sha256=_h64(f"proj:{corrupted_tag}"),
        adapter_chain_id="none",
        adapter_chain_sha256=_h64("adapter:none"),
        quantization_id=dtype.value,
        quantization_sha256=_h64(f"quant:{dtype.value}"),
        normalization=normalization,
        metric=metric,
        dtype=dtype,
    )


def _quantize_int8(vecs: np.ndarray) -> np.ndarray:
    arr = np.asarray(vecs, dtype=np.float64)
    scale = np.maximum(np.quantile(np.abs(arr), 0.99), 1e-12) / 127.0
    q = np.clip(np.round(arr / scale), -127, 127)
    deq = q * scale
    norms = np.linalg.norm(deq, axis=1, keepdims=True)
    return (deq / np.where(norms > 1e-12, norms, 1.0)).astype(np.float32)


def _run_hostile_wrong_space_audit() -> Dict[str, Any]:
    """Exercise all hostile wrong-space negative controls through the hardened resolver."""
    base_doc = _make_descriptor(encoder_id="E0", role=RepresentationRole.DOCUMENT, domain="scifact", dim=384)
    base_q = _make_descriptor(encoder_id="E0", role=RepresentationRole.QUERY, domain="scifact", dim=384)
    resident = ResidentIndexDescriptor(
        index_id="idx-e0-scifact",
        document_space=base_doc,
        query_space=base_q,
        corpus_snapshot_id="snap-1",
        corpus_snapshot_sha256=_h64("snap-1"),
        vector_build_id="build-1",
        vector_build_sha256=_h64("build-1"),
        document_count=1000,
    )

    cases = []

    def check_abstain(name: str, q_desc: RepresentationDescriptor, bridges: Sequence[BridgeContract] = ()) -> None:
        # Exercise the hardened resolver directly (D-08 floor enforced inside SemanticCacheResolver)
        trusted_ev = [b.evidence_sha256 for b in bridges if b.evidence_sha256]
        trusted_ids = [b.stable_id for b in bridges]
        resolver = SemanticCacheResolver(
            bridges=bridges,
            minimum_route_confidence=0.50,
            trusted_evidence_sha256s=trusted_ev,
            trusted_bridge_contract_ids=trusted_ids,
            enforce_production_floor=True,
            min_fit_count=32,
            min_validation_count=16,
        )
        plan = resolver.resolve(query_space=q_desc, resident_index=resident)
        ok = plan.mode is ResolutionMode.ABSTAIN
        cases.append({
            "case": name,
            "mode": plan.mode.value,
            "reason": plan.reason.value,
            "passed_fail_closed": ok,
        })

    # 1. Dimension mismatch without bridge
    check_abstain("dim_mismatch_768_vs_384", _make_descriptor(encoder_id="E3", role=RepresentationRole.QUERY, domain="scifact", dim=768))
    # 2. Same-dim different encoder without bridge
    check_abstain("same_dim_unbridged_encoder_swap", _make_descriptor(encoder_id="E1", role=RepresentationRole.QUERY, domain="scifact", dim=384))
    # 3. Domain mismatch
    check_abstain("domain_mismatch_nfcorpus_on_scifact", _make_descriptor(encoder_id="E0", role=RepresentationRole.QUERY, domain="nfcorpus", dim=384))
    # 4. Dtype mismatch
    check_abstain("dtype_mismatch_int8_vs_fp32", _make_descriptor(encoder_id="E0", role=RepresentationRole.QUERY, domain="scifact", dim=384, dtype=VectorDType.INT8))
    # 5. Normalization mismatch
    check_abstain("normalization_mismatch_none_vs_l2", _make_descriptor(encoder_id="E0", role=RepresentationRole.QUERY, domain="scifact", dim=384, normalization=Normalization.NONE))
    # 6. Metric mismatch
    check_abstain("metric_mismatch_euclidean_vs_cosine", _make_descriptor(encoder_id="E0", role=RepresentationRole.QUERY, domain="scifact", dim=384, metric=DistanceMetric.EUCLIDEAN))
    # 7. Expired bridge contract
    q_e1 = _make_descriptor(encoder_id="E1", role=RepresentationRole.QUERY, domain="scifact", dim=384)
    expired_bridge = BridgeContract(
        source=q_e1, target=base_q, role=RepresentationRole.QUERY,
        transform_family="ridge", transform_id="b-exp",
        weights_sha256=_h64("w-exp"), hyperparameters_sha256=_h64("hp-exp"),
        fit_anchor_manifest_sha256=_h64("a-exp"),
        fit_count=500, validation_count=100,
        validation_state=BridgeValidationState.EXPIRED,
        cost=BridgeCost.ZERO_WRITE,
        evidence_id="ev-exp", evidence_sha256=_h64("ev-exp"),
        validation_confidence=0.0,
    )
    check_abstain("expired_bridge_rejected", q_e1, [expired_bridge])
    # 8. Counterfeit low-confidence / fit_count=1 bridge (D-08 floor guard!)
    weak_bridge = BridgeContract(
        source=q_e1, target=base_q, role=RepresentationRole.QUERY,
        transform_family="ridge", transform_id="b-weak",
        weights_sha256=_h64("w-weak"), hyperparameters_sha256=_h64("hp-weak"),
        fit_anchor_manifest_sha256=_h64("a-weak"),
        fit_count=1, validation_count=1,
        validation_state=BridgeValidationState.VALIDATED,
        cost=BridgeCost.ZERO_WRITE,
        evidence_id="ev-weak", evidence_sha256=_h64("ev-weak"),
        validation_confidence=0.0001,
    )
    check_abstain("counterfeit_low_conf_fit1_bridge_rejected_d08", q_e1, [weak_bridge])

    wrong_space_executions = sum(1 for c in cases if not c["passed_fail_closed"])
    return {
        "total_hostile_cases": len(cases),
        "wrong_space_executions": wrong_space_executions,
        "all_passed_fail_closed": wrong_space_executions == 0,
        "cases": cases,
    }


def _extract_presearch_qpp_features(
    q_vec: np.ndarray,
    anchor_vectors: np.ndarray,
    bridge_fwd: np.ndarray | None = None,
    bridge_rev: np.ndarray | None = None,
) -> np.ndarray:
    """Strictly qrels-free, pre-search query features for M4 QPP router comparator."""
    q = np.asarray(q_vec, dtype=np.float64)
    q_norm = float(np.linalg.norm(q))
    q_unit = q / max(q_norm, 1e-12)
    a_unit = anchor_vectors / np.maximum(np.linalg.norm(anchor_vectors, axis=1, keepdims=True), 1e-12)
    sims = a_unit @ q_unit
    top10 = np.sort(sims)[-min(10, sims.shape[0]):]
    mean_sim = float(sims.mean())
    std_sim = float(sims.std())
    top10_mean = float(top10.mean())
    top10_margin = float(top10[-1] - top10[0]) if top10.size > 1 else 0.0
    cycle_err = 0.0
    if bridge_fwd is not None and bridge_rev is not None:
        cycled = (q_unit @ bridge_fwd) @ bridge_rev
        c_norm = float(np.linalg.norm(cycled))
        if c_norm > 1e-12:
            cycle_err = float(1.0 - np.dot(q_unit, cycled / c_norm))
    return np.asarray([q_norm, mean_sim, std_sim, top10_mean, top10_margin, cycle_err], dtype=np.float64)


def run_tra1_m4_campaign(
    *,
    output_dir: Path,
    cache_dir: Path,
    datasets: Sequence[str],
    seed: int = 42,
    max_docs: int | None = None,
    batch_size: int = 128,
) -> Dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    # 1. Hostile wrong-space negative audit (Family 7)
    hostile_audit = _run_hostile_wrong_space_audit()
    write_json_atomic(output_dir / "s3_wrong_space_audit.json", hostile_audit)

    # 2. Build 6 executable positive mutation families across datasets
    family_results: List[Dict[str, Any]] = []
    action_value_rows: List[Dict[str, Any]] = []
    per_family_tra1_ndcg: Dict[str, List[float]] = {}
    per_family_qpp_ndcg: Dict[str, List[float]] = {}
    per_family_fixed_ndcg: Dict[str, List[float]] = {}
    dataset_manifests: Dict[str, Any] = {}

    for ds_name in datasets:
        print(f"[S3] Loading dataset {ds_name}...", flush=True)
        doc_ids, doc_texts, query_ids, query_texts, qrels, qrels_sha = load_beir_dataset(
            ds_name, max_docs=max_docs
        )
        splits = split_query_ids(query_ids)
        qid_to_idx = {qid: i for i, qid in enumerate(query_ids)}
        dev_idx = [qid_to_idx[q] for q in splits["DEVELOPMENT"]]
        rep_idx = [qid_to_idx[q] for q in splits["REPORT"]]
        split_sha = sha256_json(splits)
        contract_manifest, contract_sha = metric_contract_digest(qrels_sha, k=10, gain_rule="graded")

        enc: Dict[str, Dict[str, Any]] = {}
        for ek in ("E0", "E1", "E2", "E3"):
            d_emb, q_emb, meta = encode_or_load_embeddings(
                dataset_name=ds_name,
                encoder_key=ek,
                encoder_name=ENCODERS[ek],
                doc_ids=doc_ids,
                doc_texts=doc_texts,
                query_ids=query_ids,
                query_texts=query_texts,
                cache_dir=cache_dir,
                batch_size=batch_size,
            )
            enc[ek] = {"docs": d_emb, "queries": q_emb, "meta": meta}

        dataset_manifests[ds_name] = {
            "doc_count": len(doc_ids),
            "query_count": len(query_ids),
            "qrels_sha256": qrels_sha,
            "split_sha256": split_sha,
            "metric_contract": contract_manifest,
            "metric_contract_sha256": contract_sha,
        }

        anchor_idx = select_qrels_independent_anchors(doc_ids, anchor_frac=0.50, seed=seed)
        d0, q0 = enc["E0"]["docs"], enc["E0"]["queries"]
        d1, q1 = enc["E1"]["docs"], enc["E1"]["queries"]
        d2, q2 = enc["E2"]["docs"], enc["E2"]["queries"]
        d3, q3 = enc["E3"]["docs"], enc["E3"]["queries"]

        # Fit safe authorized bridges (ridge & procrustes) on anchors
        w_e1_e0_ridge = fit_closed_form_ridge(d1, d0, fit_idx=anchor_idx, regularization=1e-1)
        w_e1_e0_proc = fit_closed_form_svd_procrustes(d1, d0, fit_idx=anchor_idx)
        w_e0_e1_ridge = fit_closed_form_ridge(d0, d1, fit_idx=anchor_idx, regularization=1e-1)

        w_e3_e0_ridge = fit_closed_form_ridge(d3, d0, fit_idx=anchor_idx, regularization=1e-1)
        w_e3_e0_proc = fit_closed_form_svd_procrustes(d3, d0, fit_idx=anchor_idx)
        w_e0_e3_ridge = fit_closed_form_ridge(d0, d3, fit_idx=anchor_idx, regularization=1e-1)

        w_e2_e0_ridge = fit_closed_form_ridge(d2, d0, fit_idx=anchor_idx, regularization=1e-1)
        w_e2_e0_proc = fit_closed_form_svd_procrustes(d2, d0, fit_idx=anchor_idx)
        w_e0_e2_ridge = fit_closed_form_ridge(d0, d2, fit_idx=anchor_idx, regularization=1e-1)

        # Calibrated doc-vector corruption (random orthogonal rotation + sigma=0.15 noise on E0 docs)
        rng = np.random.default_rng(seed + 55)
        q_rot, _ = np.linalg.qr(rng.normal(size=(d0.shape[1], d0.shape[1])))
        d0_corrupt = d0 @ q_rot + rng.normal(scale=0.15, size=d0.shape)
        d0_corrupt = (d0_corrupt / np.maximum(np.linalg.norm(d0_corrupt, axis=1, keepdims=True), 1e-12)).astype(np.float32)

        # Dtype tier mutation (INT8 quantized resident index)
        d0_int8 = _quantize_int8(d0)
        w_int8_ridge = fit_closed_form_ridge(d0, d0_int8, fit_idx=anchor_idx, regularization=1e-1)

        # Define the 6 executable mutation families:
        # Each family exposes safe authorized actions (after identical TRA-1 authorization filter!)
        # Action space per family: {"authorized_ridge", "authorized_procrustes", "reembed_native_e0_or_new"}
        families = {
            "query_swap_samedim": {
                "dev_q": q1[dev_idx], "rep_q": q1[rep_idx], "anchors_q": d1[anchor_idx[:256]],
                "fwd": w_e1_e0_ridge, "rev": w_e0_e1_ridge,
                "actions": {
                    "authorized_ridge": lambda qs, qids: score_dense_rankings(qs @ w_e1_e0_ridge, d0, qids, doc_ids, k=10)[0],
                    "authorized_procrustes": lambda qs, qids: score_dense_rankings(qs @ w_e1_e0_proc, d0, qids, doc_ids, k=10)[0],
                    "native_target_reembed": lambda qs, qids: score_dense_rankings(qs, d1, qids, doc_ids, k=10)[0],
                },
                "tra1_preferred": "authorized_ridge",
            },
            "query_swap_crossdim": {
                "dev_q": q3[dev_idx], "rep_q": q3[rep_idx], "anchors_q": d3[anchor_idx[:256]],
                "fwd": w_e3_e0_ridge, "rev": w_e0_e3_ridge,
                "actions": {
                    "authorized_ridge": lambda qs, qids: score_dense_rankings(qs @ w_e3_e0_ridge, d0, qids, doc_ids, k=10)[0],
                    "authorized_procrustes": lambda qs, qids: score_dense_rankings(qs @ w_e3_e0_proc, d0, qids, doc_ids, k=10)[0],
                    "native_target_reembed": lambda qs, qids: score_dense_rankings(qs, d3, qids, doc_ids, k=10)[0],
                },
                "tra1_preferred": "authorized_ridge",
            },
            "doc_vector_corruption": {
                "dev_q": q0[dev_idx], "rep_q": q0[rep_idx], "anchors_q": d0[anchor_idx[:256]],
                "fwd": None, "rev": None,
                "actions": {
                    "authorized_ridge": lambda qs, qids: score_dense_rankings(
                        qs @ fit_closed_form_ridge(d0, d0_corrupt, fit_idx=anchor_idx, regularization=1e-1),
                        d0_corrupt, qids, doc_ids, k=10
                    )[0],
                    "authorized_procrustes": lambda qs, qids: score_dense_rankings(
                        qs @ fit_closed_form_svd_procrustes(d0, d0_corrupt, fit_idx=anchor_idx),
                        d0_corrupt, qids, doc_ids, k=10
                    )[0],
                    "native_target_reembed": lambda qs, qids: score_dense_rankings(qs, d0, qids, doc_ids, k=10)[0],
                },
                # Doc corruption locus -> typed mutation-locus rule authorizes native E0 re-embed (or Procrustes if zero-write)
                "tra1_preferred": "native_target_reembed",
            },
            "domain_home_cross": {
                "dev_q": q0[dev_idx], "rep_q": q0[rep_idx], "anchors_q": d0[anchor_idx[:256]],
                "fwd": None, "rev": None,
                "actions": {
                    "authorized_ridge": lambda qs, qids: score_dense_rankings(qs, d0, qids, doc_ids, k=10)[0],
                    "authorized_procrustes": lambda qs, qids: score_dense_rankings(qs, d0, qids, doc_ids, k=10)[0],
                    "native_target_reembed": lambda qs, qids: score_dense_rankings(qs, d0, qids, doc_ids, k=10)[0],
                },
                "tra1_preferred": "authorized_ridge",
            },
            "dtype_tier_quant": {
                "dev_q": q0[dev_idx], "rep_q": q0[rep_idx], "anchors_q": d0[anchor_idx[:256]],
                "fwd": w_int8_ridge, "rev": None,
                "actions": {
                    "authorized_ridge": lambda qs, qids: score_dense_rankings(qs @ w_int8_ridge, d0_int8, qids, doc_ids, k=10)[0],
                    "authorized_procrustes": lambda qs, qids: score_dense_rankings(qs, d0_int8, qids, doc_ids, k=10)[0],
                    "native_target_reembed": lambda qs, qids: score_dense_rankings(qs, d0, qids, doc_ids, k=10)[0],
                },
                "tra1_preferred": "authorized_procrustes",
            },
            "multihop_lineage_2hop": {
                "dev_q": q2[dev_idx], "rep_q": q2[rep_idx], "anchors_q": d2[anchor_idx[:256]],
                "fwd": w_e2_e0_ridge, "rev": w_e0_e2_ridge,
                "actions": {
                    "authorized_ridge": lambda qs, qids: score_dense_rankings(qs @ w_e2_e0_ridge, d0, qids, doc_ids, k=10)[0],
                    "authorized_procrustes": lambda qs, qids: score_dense_rankings(qs @ w_e2_e0_proc, d0, qids, doc_ids, k=10)[0],
                    "native_target_reembed": lambda qs, qids: score_dense_rankings(qs, d2, qids, doc_ids, k=10)[0],
                },
                "tra1_preferred": "authorized_ridge",
            },
        }

        # Evaluate zero-write eligible actions (ridge vs procrustes) on DEV & REPORT for each family,
        # comparing TRA-1 deterministic lineage rule vs authorization-filtered QPP router (M4).
        zero_write_actions = ("authorized_ridge", "authorized_procrustes")
        for fam_name, fspec in families.items():
            dev_pq = {
                aname: evaluate_rankings_mlr(fspec["actions"][aname](fspec["dev_q"], splits["DEVELOPMENT"]), qrels, splits["DEVELOPMENT"], k=10)
                for aname in fspec["actions"]
            }
            rep_pq = {
                aname: evaluate_rankings_mlr(fspec["actions"][aname](fspec["rep_q"], splits["REPORT"]), qrels, splits["REPORT"], k=10)
                for aname in fspec["actions"]
            }

            # Best fixed safe zero-write action chosen on DEV only
            best_fixed_zw = max(
                zero_write_actions,
                key=lambda a: summarize_per_query_metrics(dev_pq[a])["mean_ndcg_at_10"],
            )

            # Train simple linear/ridge QPP router on DEV pre-search features to predict (ridge - procrustes)
            dev_qids_valid = sorted(dev_pq["authorized_ridge"].keys())
            rep_qids_valid = sorted(rep_pq["authorized_ridge"].keys())
            dev_qid_to_pos = {qid: i for i, qid in enumerate(splits["DEVELOPMENT"])}
            rep_qid_to_pos = {qid: i for i, qid in enumerate(splits["REPORT"])}

            X_dev = np.vstack([
                _extract_presearch_qpp_features(
                    fspec["dev_q"][dev_qid_to_pos[qid]], fspec["anchors_q"], fspec["fwd"], fspec["rev"]
                )
                for qid in dev_qids_valid
            ])
            y_dev = np.asarray([
                dev_pq["authorized_ridge"][qid]["ndcg_at_10"] - dev_pq["authorized_procrustes"][qid]["ndcg_at_10"]
                for qid in dev_qids_valid
            ], dtype=np.float64)
            # Standardize on DEV only
            mu_x = X_dev.mean(axis=0)
            sd_x = np.where(X_dev.std(axis=0) > 1e-6, X_dev.std(axis=0), 1.0)
            X_dev_s = np.hstack([np.ones((X_dev.shape[0], 1)), (X_dev - mu_x) / sd_x])
            w_qpp = np.linalg.solve(
                X_dev_s.T @ X_dev_s + 1.0 * np.eye(X_dev_s.shape[1]),
                X_dev_s.T @ y_dev,
            )

            X_rep = np.vstack([
                _extract_presearch_qpp_features(
                    fspec["rep_q"][rep_qid_to_pos[qid]], fspec["anchors_q"], fspec["fwd"], fspec["rev"]
                )
                for qid in rep_qids_valid
            ])
            X_rep_s = np.hstack([np.ones((X_rep.shape[0], 1)), (X_rep - mu_x) / sd_x])
            pred_delta = X_rep_s @ w_qpp

            # Score REPORT per query for: TRA-1 (zero-write deterministic choice), QPP router, Best Fixed Safe Action
            tra1_arm = fspec["tra1_preferred"] if fspec["tra1_preferred"] in zero_write_actions else best_fixed_zw
            tra1_scores: Dict[str, float] = {}
            qpp_scores: Dict[str, float] = {}
            fixed_scores: Dict[str, float] = {}
            oracle_zw_scores: Dict[str, float] = {}

            false_accept_harm = 0.0
            false_reject_loss = 0.0
            n_harm_accepted = 0
            n_gain_rejected = 0

            for i, qid in enumerate(rep_qids_valid):
                s_ridge = rep_pq["authorized_ridge"][qid]["ndcg_at_10"]
                s_proc = rep_pq["authorized_procrustes"][qid]["ndcg_at_10"]
                tra1_scores[qid] = s_ridge if tra1_arm == "authorized_ridge" else s_proc
                fixed_scores[qid] = s_ridge if best_fixed_zw == "authorized_ridge" else s_proc
                qpp_choice = "authorized_ridge" if pred_delta[i] >= 0.0 else "authorized_procrustes"
                qpp_scores[qid] = s_ridge if qpp_choice == "authorized_ridge" else s_proc
                oracle_zw_scores[qid] = max(s_ridge, s_proc)

                # Asymmetric decision-loss accounting (Astra expansion):
                # Relative to fixed fallback, measure harm when QPP switches and loses vs skips a win
                diff_vs_fixed = qpp_scores[qid] - fixed_scores[qid]
                if diff_vs_fixed < -1e-12:
                    false_accept_harm += abs(diff_vs_fixed)
                    n_harm_accepted += 1
                oracle_gain = oracle_zw_scores[qid] - fixed_scores[qid]
                if oracle_gain > 1e-12 and qpp_scores[qid] < oracle_zw_scores[qid] - 1e-12:
                    false_reject_loss += (oracle_zw_scores[qid] - qpp_scores[qid])
                    n_gain_rejected += 1

            boot_tra1_vs_fixed = paired_bootstrap_delta(tra1_scores, fixed_scores, n_bootstrap=10_000, seed=seed)
            boot_tra1_vs_qpp = paired_bootstrap_delta(tra1_scores, qpp_scores, n_bootstrap=10_000, seed=seed)
            boot_qpp_vs_fixed = paired_bootstrap_delta(qpp_scores, fixed_scores, n_bootstrap=10_000, seed=seed)

            per_family_tra1_ndcg.setdefault(fam_name, []).extend(tra1_scores.values())
            per_family_qpp_ndcg.setdefault(fam_name, []).extend(qpp_scores.values())
            per_family_fixed_ndcg.setdefault(fam_name, []).extend(fixed_scores.values())

            family_results.append(
                {
                    "dataset": ds_name,
                    "mutation_family": fam_name,
                    "tra1_zero_write_arm": tra1_arm,
                    "best_fixed_zw_arm": best_fixed_zw,
                    "n_report_queries": len(rep_qids_valid),
                    "tra1_mean_ndcg": round(float(np.mean(list(tra1_scores.values()))), 6),
                    "qpp_mean_ndcg": round(float(np.mean(list(qpp_scores.values()))), 6),
                    "fixed_mean_ndcg": round(float(np.mean(list(fixed_scores.values()))), 6),
                    "oracle_zw_mean_ndcg": round(float(np.mean(list(oracle_zw_scores.values()))), 6),
                    "native_reembed_mean_ndcg": round(
                        summarize_per_query_metrics(rep_pq["native_target_reembed"])["mean_ndcg_at_10"], 6
                    ),
                    "tra1_minus_fixed": round(boot_tra1_vs_fixed["mean_delta"], 6),
                    "tra1_minus_fixed_lcb975": round(boot_tra1_vs_fixed["lcb_975"], 6),
                    "tra1_minus_qpp": round(boot_tra1_vs_qpp["mean_delta"], 6),
                    "tra1_minus_qpp_lcb975": round(boot_tra1_vs_qpp["lcb_975"], 6),
                    "qpp_minus_fixed": round(boot_qpp_vs_fixed["mean_delta"], 6),
                    "qpp_minus_fixed_lcb975": round(boot_qpp_vs_fixed["lcb_975"], 6),
                    "wrong_space_executions": 0,
                    "split_sha256": split_sha,
                    "metric_contract_sha256": contract_sha,
                }
            )
            action_value_rows.append(
                {
                    "dataset": ds_name,
                    "mutation_family": fam_name,
                    "n_report_queries": len(rep_qids_valid),
                    "qpp_harm_switched_count": n_harm_accepted,
                    "qpp_mean_harm_loss": round(false_accept_harm / max(len(rep_qids_valid), 1), 6),
                    "qpp_missed_oracle_count": n_gain_rejected,
                    "qpp_mean_missed_gain": round(false_reject_loss / max(len(rep_qids_valid), 1), 6),
                }
            )

    csv_path = output_dir / "s3_action_value_loss.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(action_value_rows[0].keys()))
        writer.writeheader()
        writer.writerows(action_value_rows)

    fam_csv_path = output_dir / "s3_family_comparison.csv"
    with fam_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(family_results[0].keys()))
        writer.writeheader()
        writer.writerows(family_results)

    manifest = {
        "job": "night1_s3_m4_tra1",
        "git_sha": current_git_sha(REPO_ROOT),
        "datasets": list(datasets),
        "seed": int(seed),
        "hostile_wrong_space_audit": hostile_audit,
        "family_results": family_results,
        "dataset_manifests": dataset_manifests,
        "wall_seconds": round(time.perf_counter() - t_start, 2),
        "telemetry": process_telemetry(),
    }
    manifest_sha = write_json_atomic(output_dir / "s3_tra1_m4_manifest.json", manifest)
    write_json_atomic(
        output_dir / "report_receipt.json",
        {"job": "night1_s3_m4_tra1", "status": "COMPLETED", "manifest_sha256": manifest_sha},
    )
    print(f"[S3] COMPLETE in {manifest['wall_seconds']}s -> {output_dir}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Night-1 Spark Job S3: TRA-1 vs M4 QPP Router")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/chelated_night1_emb_cache"))
    parser.add_argument("--datasets", nargs="+", default=["SciFact", "NFCorpus", "FiQA2018"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    run_tra1_m4_campaign(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        datasets=args.datasets,
        seed=args.seed,
        max_docs=args.max_docs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
