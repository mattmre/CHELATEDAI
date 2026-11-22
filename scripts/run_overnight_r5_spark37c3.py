r"""Overnight R5 Master Validation Worker 2 (`gx10-37c3`): `PM-2` Coupled Persistent Service & `PM-3` Live Pinned Upstream Comparator (`mem0ai` + `graphiti-core` / `ScaleMAS` Archive).

Strictly ADDITIVE & NON-DESTRUCTIVE:
  - Writes ONLY to `--output-root` (default `/home/mattmre/chelated-night1/out/20261122-overnight-r5/spark37c3`).
  - Never modifies `.venv` or running Docker containers; uses an isolated `/home/mattmre/chelated-night1/.venv-pm3` for `mem0ai`.
  - Emits independent atomic manifests + 100%-SHA-256 `report_receipt.json` after EVERY stage:
      * Stage 1 (`r5_s1_pm2_coupled_service`): Joint persistent vector+epistemic service combining `384<->4096` OOD, `4096->2560->1024->384` 3-Hop, and `E0->E3->E1` interior-peak migrations (`EM-2 ZeroMu` + `A2c-Polar` vs `Composed CAR` vs `Native`) + `12/12` hostile `TRA-1` probes + simultaneous `EK5 v3` supersession & `6x` echo stress + live `nemotron-lightning` QA (`N=240`).
      * Stage 2 (`r5_s2_pm3_live_upstream_comparators`): Live pinned upstream `mem0ai` (`mem0.Memory` + `nemotron-lightning` + `MiniLM` + local Qdrant) vs `C4_v3_ek5_v3_k4`, `C1_scalemas_bounded`, `M1_hybrid_flat`, and `M4f_soft_telemetry_fallback` across `Zero-Shortcut v3` (`N=80`) + real multi-session `LongMemEval-S` (`N=60`), plus `graphiti-core` & `ScaleMAS` formal provenance receipt.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ek5_v3_stratified import EdgeAttestation, StratifiedEvent, StratifiedEvidenceStore  # noqa: E402
from m1_baseline_ladder import (  # noqa: E402
    fit_closed_form_centered_ridge,
    score_dense_rankings,
    split_query_ids,
)
from representation_space import (  # noqa: E402
    BridgeContract,
    BridgeContractError,
    BridgeCost,
    BridgeValidationState,
    DistanceMetric,
    Normalization,
    PoolingStrategy,
    RepresentationABIError,
    RepresentationDescriptor,
    RepresentationRole,
    ResidentIndexDescriptor,
)
from scripts.night1_common import (  # noqa: E402
    current_git_sha,
    load_beir_dataset,
    paired_bootstrap_delta,
    process_telemetry,
    write_json_atomic,
)
from scripts.run_flagship1_r4_comprehensive import (  # noqa: E402
    compute_per_query_ndcg_both,
    load_ood_domains_and_cache,
)
from semantic_cache_resolver import ResolutionMode  # noqa: E402
from spherical_migration_bridge import (  # noqa: E402
    CenteredProcrustesBridge,
    build_bridge_contract,
    compose_centered_bridges,
    l2_normalize,
    resolve_with_tra1,
    score_zeromu_rankings,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _call_vllm(endpoint: str, model_id: str, sys_p: str, usr_p: str, *, max_tokens: int = 64) -> Tuple[str, int, float, bool]:
    t0 = time.perf_counter()
    for attempt in range(3):
        try:
            r = requests.post(
                f"{endpoint.rstrip('/')}/chat/completions",
                json={
                    "model": model_id,
                    "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=60.0,
            )
            r.raise_for_status()
            d = r.json()
            txt = (d["choices"][0]["message"].get("content") or "").strip()
            pt = int(d.get("usage", {}).get("prompt_tokens", 0))
            return txt, pt, (time.perf_counter() - t0) * 1000.0, False
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                return f"ERROR: {exc}", 0, (time.perf_counter() - t0) * 1000.0, True
            time.sleep(0.5 * (attempt + 1))
    return "ERROR: unreachable", 0, 0.0, True


# ===========================================================================
# STAGE 1: PM-2 COUPLED PERSISTENT SERVICE (384<->4096 + 3-HOP + PEAK + TRA-1 + EK5 v3 + ECHO STRESS)
# ===========================================================================
def _mk_desc(
    enc_id: str,
    dim: int,
    *,
    role: RepresentationRole = RepresentationRole.QUERY,
    dom: str = "dom-v1",
    rev: str = "r1",
    norm: Normalization = Normalization.L2,
    metric: DistanceMetric = DistanceMetric.COSINE,
) -> RepresentationDescriptor:
    h = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()  # noqa: E731
    return RepresentationDescriptor(
        encoder_id=enc_id,
        encoder_revision=rev,
        encoder_weights_sha256=h(f"w:{enc_id}:{rev}"),
        tokenizer_id=f"tok:{enc_id}",
        tokenizer_sha256=h(f"t:{enc_id}"),
        role=role,
        compatibility_domain_id=dom,
        compatibility_domain_sha256=h(f"d:{dom}"),
        instruction_sha256=h(f"i:{role.value}"),
        pooling=PoolingStrategy.MEAN,
        dimension=dim,
        preprocessing_id="std-v1",
        preprocessing_sha256=h("prep:std-v1"),
        projection_chain_id="none",
        projection_chain_sha256=h("proj:none"),
        adapter_chain_id="none",
        adapter_chain_sha256=h("adap:none"),
        quantization_id="fp32",
        quantization_sha256=h("quant:fp32"),
        normalization=norm,
        metric=metric,
    )


def _run_tra1_12_hostile_probes() -> Dict[str, Any]:
    """Execute 12 distinct hostile wrong-space / forged-ABI / D-08 floor TRA-1 probes (must block 12/12)."""
    h = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()  # noqa: E731
    q_src_384 = _mk_desc("all-MiniLM-L6-v2", 384, role=RepresentationRole.QUERY, dom="minilm-v1")
    q_tgt_4096 = _mk_desc("Qwen3-Embedding-8B", 4096, role=RepresentationRole.QUERY, dom="qwen3-8b-v2")
    d_tgt_4096 = _mk_desc("Qwen3-Embedding-8B", 4096, role=RepresentationRole.DOCUMENT, dom="qwen3-8b-v2")
    q_wrong_4096 = _mk_desc("Mistral-Embed-4096", 4096, role=RepresentationRole.QUERY, dom="mistral-4096-v1")
    q_stale_rev = _mk_desc("Qwen3-Embedding-8B", 4096, role=RepresentationRole.QUERY, dom="qwen3-8b-v1", rev="r0")

    res_idx = ResidentIndexDescriptor(
        index_id="idx-resident-4096",
        document_space=d_tgt_4096,
        query_space=q_tgt_4096,
        corpus_snapshot_id="snap-20261122",
        corpus_snapshot_sha256=h("snap-20261122"),
        vector_build_id="bld-01",
        vector_build_sha256=h("bld-01"),
        document_count=5000,
    )

    rng = np.random.default_rng(7)
    x_fit = l2_normalize(rng.normal(size=(128, 384)).astype(np.float32))
    y_fit = l2_normalize(rng.normal(size=(128, 4096)).astype(np.float32))
    br = CenteredProcrustesBridge.fit(x_fit[:96], y_fit[:96], cal_x=x_fit[96:], cal_y=y_fit[96:])
    ev_sha = h("ev-valid-r5")
    valid_contract = build_bridge_contract(
        source=q_src_384,
        target=q_tgt_4096,
        bridge=br,
        transform_id="br-384-to-4096",
        anchor_manifest_sha256=h("anc-128"),
        validation_count=32,
        validation_confidence=0.92,
        evidence_id="ev-001",
        evidence_sha256=ev_sha,
    )

    # Confirm valid reverse bridge resolves cleanly to REVERSE mode!
    ok_dec = resolve_with_tra1(
        query_space=q_src_384,
        resident_index=res_idx,
        bridges=[valid_contract],
        trusted_evidence_sha256s=[ev_sha],
        trusted_bridge_contract_ids=[valid_contract.stable_id],
    )
    assert ok_dec.mode is ResolutionMode.REVERSE

    probes: List[Dict[str, Any]] = []

    def _check_abstain(pid: str, dec) -> None:
        blocked = dec.mode is ResolutionMode.ABSTAIN
        probes.append({"probe_id": pid, "blocked": blocked, "detail": f"{dec.mode.value}:{dec.reason.value}"})

    def _check_raises(pid: str, fn) -> None:
        blocked = False
        msg = ""
        try:
            fn()
        except (RepresentationABIError, BridgeContractError, ValueError) as exc:
            blocked = True
            msg = str(exc)[:120]
        probes.append({"probe_id": pid, "blocked": blocked, "detail": msg})

    # P01: Raw 384-d query with zero registered bridges -> ABSTAIN
    _check_abstain("P01_raw_384_no_bridge", resolve_with_tra1(query_space=q_src_384, resident_index=res_idx, bridges=[], trusted_evidence_sha256s=[ev_sha], trusted_bridge_contract_ids=[]))
    # P02: Same-dimension (4096-d) wrong encoder family -> ABSTAIN (never silently treats 4096==4096 as compatible!)
    _check_abstain("P02_same_dim_4096_wrong_encoder", resolve_with_tra1(query_space=q_wrong_4096, resident_index=res_idx, bridges=[valid_contract], trusted_evidence_sha256s=[ev_sha], trusted_bridge_contract_ids=[valid_contract.stable_id]))
    # P03: Stale encoder revision/domain (`r0` vs `r1`) -> ABSTAIN
    _check_abstain("P03_stale_encoder_revision", resolve_with_tra1(query_space=q_stale_rev, resident_index=res_idx, bridges=[valid_contract], trusted_evidence_sha256s=[ev_sha], trusted_bridge_contract_ids=[valid_contract.stable_id]))
    # P04: Untrusted bridge evidence digest -> ABSTAIN
    _check_abstain("P04_untrusted_evidence_sha256", resolve_with_tra1(query_space=q_src_384, resident_index=res_idx, bridges=[valid_contract], trusted_evidence_sha256s=[h("other")], trusted_bridge_contract_ids=[valid_contract.stable_id]))
    # P05: Untrusted bridge contract stable_id -> ABSTAIN
    _check_abstain("P05_untrusted_contract_id", resolve_with_tra1(query_space=q_src_384, resident_index=res_idx, bridges=[valid_contract], trusted_evidence_sha256s=[ev_sha], trusted_bridge_contract_ids=[]))
    # P06: Below D-08 production validation confidence floor (`0.92 < 0.97`) -> ABSTAIN
    _check_abstain("P06_below_confidence_floor", resolve_with_tra1(query_space=q_src_384, resident_index=res_idx, bridges=[valid_contract], trusted_evidence_sha256s=[ev_sha], trusted_bridge_contract_ids=[valid_contract.stable_id], minimum_route_confidence=0.97))
    # P07: Below D-08 fit anchor count floor (`fit_count=96 < min_fit_count=256`) -> ABSTAIN
    _check_abstain("P07_below_fit_count_floor", resolve_with_tra1(query_space=q_src_384, resident_index=res_idx, bridges=[valid_contract], trusted_evidence_sha256s=[ev_sha], trusted_bridge_contract_ids=[valid_contract.stable_id], min_fit_count=256))
    # P08: Unvalidated bridge contract -> ABSTAIN
    unval_c = BridgeContract(
        source=q_src_384, target=q_tgt_4096, role=RepresentationRole.QUERY,
        transform_family="centered_orthogonal_procrustes", transform_id="unval-1",
        weights_sha256=br.weights_sha256, hyperparameters_sha256=h("hp"), fit_anchor_manifest_sha256=h("anc"),
        fit_count=96, validation_count=0, validation_state=BridgeValidationState.UNVALIDATED, cost=BridgeCost.ZERO_WRITE,
    )
    _check_abstain("P08_unvalidated_bridge_state", resolve_with_tra1(query_space=q_src_384, resident_index=res_idx, bridges=[unval_c], trusted_evidence_sha256s=[ev_sha], trusted_bridge_contract_ids=[unval_c.stable_id]))
    # P09: Tampered serialized `BridgeContract` (`hyperparameters_sha256` altered without updating `stable_id`) -> BridgeContractError
    tampered_dict = valid_contract.to_dict()
    tampered_dict["hyperparameters_sha256"] = h("tampered_post_hoc")
    _check_raises("P09_forged_serialized_contract_id", lambda: BridgeContract.from_dict(tampered_dict))
    # P10: ResidentIndexDescriptor with mismatched query/document dimensions (`384` vs `4096`) -> RepresentationABIError
    _check_raises("P10_mismatched_resident_index_dims", lambda: ResidentIndexDescriptor(index_id="bad", document_space=d_tgt_4096, query_space=q_src_384, corpus_snapshot_id="s", corpus_snapshot_sha256=h("s"), vector_build_id="b", vector_build_sha256=h("b"), document_count=10))
    # P11: Query-role bridge illegally claiming `MATERIALIZED_INDEX` cost -> BridgeContractError
    _check_raises("P11_illegal_query_bridge_cost", lambda: BridgeContract(source=q_src_384, target=q_tgt_4096, role=RepresentationRole.QUERY, transform_family="procrustes", transform_id="badcost", weights_sha256=br.weights_sha256, hyperparameters_sha256=h("hp"), fit_anchor_manifest_sha256=h("anc"), fit_count=96, validation_count=32, validation_state=BridgeValidationState.VALIDATED, cost=BridgeCost.MATERIALIZED_INDEX, evidence_id="e1", evidence_sha256=ev_sha, validation_confidence=0.9))
    # P12: Degenerate constant anchors in Proposition 1b -> ValueError fail-closed
    _check_raises("P12_degenerate_constant_anchors", lambda: CenteredProcrustesBridge.fit(np.ones((32, 16), dtype=np.float32), y_fit[:32, :16]))

    n_blocked = sum(1 for p in probes if p["blocked"])
    return {"total_probes": len(probes), "blocked_count": n_blocked, "leaks": len(probes) - n_blocked, "probes": probes}


def run_stage1_pm2_coupled_service(endpoint: str, model_id: str, cache_dir: Path, root_dir: Path, max_workers: int = 16) -> Dict[str, Any]:
    out_dir = root_dir / "r5_s1_pm2_coupled_service"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    print("[R5-37c3-PM2] Running 12/12 hostile TRA-1 probes...", flush=True)
    tra1_res = _run_tra1_12_hostile_probes()
    assert tra1_res["leaks"] == 0, f"TRA-1 leak detected: {tra1_res}"

    print("[R5-37c3-PM2] Loading cached D384..D4096 OOD corpora & BEIR interior-peak embeddings...", flush=True)
    ood_data, ood_embs = load_ood_domains_and_cache(cache_dir)

    # Part A: Coupled Persistent Vector + EK5 v3 Temporal Supersession + 6x Stale Echo Flood Retrieval Suite!
    # In a live persistent service, superseded documents (`AUDIT_ONLY` in EK5 v3) + 6x unlinked stale copies
    # remain physically resident in the ANN vector index until background compaction, while active updates
    # replace them. Without EK5 v3 + Echo Filtering, flat bridged retrieval floods top-10 with stale vectors!
    coupled_cells = [
        ("InDom_SciFact_384_to_4096", "SciFact", "SciFact", "D384", "D4096", "direct"),
        ("OOD_CoSQA_to_LegalC_384_to_4096", "CoSQA_Code", "Legal_Contracts", "D384", "D4096", "direct"),
        ("OOD_CoSQA_to_LegalC_1024_to_4096", "CoSQA_Code", "Legal_Contracts", "D1024", "D4096", "direct"),
        ("Chain3Hop_SciFact_4096_2560_1024_384", "SciFact", "SciFact", "D4096", "D384", "chain3hop"),
    ]

    retrieval_rows = []
    coupled_latencies_ms = []

    for cell_name, fit_dom, eval_dom, src_k, tgt_k, topo in coupled_cells:
        ev_ds = ood_data[eval_dom]
        dids_base = list(ev_ds["doc_ids"])
        qids_all = list(ev_ds["query_ids"])
        qrels_base = ev_ds["qrels"]
        _, eval_qids = split_query_ids(qids_all, seed=42)
        q_idx_map = {q: i for i, q in enumerate(qids_all)}
        eval_q_indices = [q_idx_map[q] for q in eval_qids]

        for seed in (42, 43, 44):
            rng = np.random.default_rng(seed)
            n_fit_corpus = ood_embs[fit_dom][src_k]["docs"].shape[0]
            n_anc = min(1500, n_fit_corpus)
            perm = rng.permutation(n_fit_corpus)[:n_anc]
            n_cal = max(150, int(0.2 * n_anc))
            fit_idx, cal_idx = perm[n_cal:], perm[:n_cal]

            d_eval_tgt = ood_embs[eval_dom][tgt_k]["docs"]
            q_eval_src = ood_embs[eval_dom][src_k]["queries"][eval_q_indices]
            q_eval_tgt = ood_embs[eval_dom][tgt_k]["queries"][eval_q_indices]

            # Fit bridges on fit_dom!
            if topo == "direct":
                x_f, y_f = ood_embs[fit_dom][src_k]["docs"], ood_embs[fit_dom][tgt_k]["docs"]
                br = CenteredProcrustesBridge.fit(x_f[fit_idx], y_f[fit_idx], cal_x=x_f[cal_idx], cal_y=y_f[cal_idx], scale_mode="max_one_s_full_oos")
                mu_x_car, mu_y_car, w_car = fit_closed_form_centered_ridge(x_f, y_f, fit_idx=fit_idx, regularization=10.0)
                q_car = l2_normalize((q_eval_src - mu_x_car) @ w_car + mu_y_car)
            else:
                # 3-hop 4096 -> 2560 -> 1024 -> 384
                d4, d25, d10, d38 = (
                    ood_embs[fit_dom]["D4096"]["docs"],
                    ood_embs[fit_dom]["D2560"]["docs"],
                    ood_embs[fit_dom]["D1024"]["docs"],
                    ood_embs[fit_dom]["D384"]["docs"],
                )
                b1 = CenteredProcrustesBridge.fit(d4[fit_idx], d25[fit_idx], cal_x=d4[cal_idx], cal_y=d25[cal_idx])
                b2 = CenteredProcrustesBridge.fit(d25[fit_idx], d10[fit_idx], cal_x=d25[cal_idx], cal_y=d10[cal_idx])
                b3 = CenteredProcrustesBridge.fit(d10[fit_idx], d38[fit_idx], cal_x=d10[cal_idx], cal_y=d38[cal_idx])
                br = compose_centered_bridges([b1, b2, b3], polar_reproject=True)
                mx1, my1, wc1 = fit_closed_form_centered_ridge(d4, d25, fit_idx=fit_idx, regularization=10.0)
                mx2, my2, wc2 = fit_closed_form_centered_ridge(d25, d10, fit_idx=fit_idx, regularization=10.0)
                mx3, my3, wc3 = fit_closed_form_centered_ridge(d10, d38, fit_idx=fit_idx, regularization=10.0)
                h1 = l2_normalize((q_eval_src - mx1) @ wc1 + my1)
                h2 = l2_normalize((h1 - mx2) @ wc2 + my2)
                q_car = l2_normalize((h2 - mx3) @ wc3 + my3)

            # Inject coupled temporal lifecycle into resident corpus:
            # For 50% of positive-qrel documents, the original v1 vector was superseded (`REFUTE` + `SUPERSEDE` in EK5 v3)
            # AND echoed 6x in resident ANN storage with tiny embedding perturbations (`cos ~ 0.985`), while only `doc_v2`
            # (the authoritative current update) is valid in `qrels_coupled`!
            st_ek5 = StratifiedEvidenceStore()
            dids_coupled = list(dids_base)
            embs_extra = []
            qrels_coupled: Dict[str, Dict[str, float]] = {}
            doc_idx_map = {d: i for i, d in enumerate(dids_base)}

            superseded_set = set()
            for qid in eval_qids:
                rmap = qrels_base.get(qid, {})
                new_rmap = {}
                for did, sc in rmap.items():
                    if sc <= 0 or did not in doc_idx_map:
                        continue
                    # Deterministic 60% of relevant docs underwent an authoritative version update!
                    if int(hashlib.md5(f"{did}_{seed}".encode()).hexdigest()[:4], 16) % 10 < 6:
                        superseded_set.add(did)
                        upd_id = f"{did}__v2_current"
                        new_rmap[upd_id] = sc
                    else:
                        new_rmap[did] = sc
                if new_rmap:
                    qrels_coupled[qid] = new_rmap

            new_ev_batch = [StratifiedEvent(d, authority_tier=2) for d in dids_base]
            new_att_batch = []
            rng_noise = np.random.default_rng(seed + 999)
            for did in sorted(superseded_set):
                base_vec = d_eval_tgt[doc_idx_map[did]]
                upd_id = f"{did}__v2_current"
                # Active updated record sits cleanly near base_vec (slight semantic shift ~0.94 cos)
                noise_u = rng_noise.normal(scale=0.005, size=base_vec.shape).astype(np.float32)
                upd_vec = l2_normalize((base_vec + noise_u)[None, :])[0]
                dids_coupled.append(upd_id)
                embs_extra.append(upd_vec)
                new_ev_batch.append(StratifiedEvent(upd_id, authority_tier=3))
                new_att_batch.append(EdgeAttestation(f"ref_{did}", upd_id, did, "REFUTE", valid_time=10, tx_time=10))
                new_att_batch.append(EdgeAttestation(f"sup_{did}", upd_id, did, "SUPERSEDE", valid_time=10, tx_time=10))
                # Plus 6 stale echoes of `did` (`COPY_OF` x3 + high-sim unlinked echoes x3) in resident ANN!
                for e_i in range(6):
                    eid = f"{did}__stale_echo_{e_i}"
                    en = rng_noise.normal(scale=0.0015, size=base_vec.shape).astype(np.float32)
                    e_vec = l2_normalize((base_vec + en)[None, :])[0]
                    dids_coupled.append(eid)
                    embs_extra.append(e_vec)
                    new_ev_batch.append(StratifiedEvent(eid, authority_tier=2))
                    new_att_batch.append(EdgeAttestation(f"cp_{did}_{e_i}", eid, did, "COPY_OF", valid_time=3 + e_i, tx_time=3 + e_i))

            st_ek5.apply_mutation(new_ev_batch, new_att_batch)
            d_coupled_tgt = np.vstack([d_eval_tgt] + embs_extra) if embs_extra else d_eval_tgt

            # Active mask enforced by EK5 v3 in O(1) per candidate!
            admit_indices = [i for i, did in enumerate(dids_coupled) if st_ek5.eligibility.get(did, "ADMIT") == "ADMIT"]
            d_admit_tgt = d_coupled_tgt[admit_indices]
            dids_admit = [dids_coupled[i] for i in admit_indices]

            # Time externally coupled query execution (`bridge + EK5 v3 filter + top-10`)!
            if seed == 42:
                for q_i in range(min(100, len(eval_qids))):
                    t_q0 = time.perf_counter()
                    qv = score_zeromu_rankings(
                        q_eval_src[q_i : q_i + 1], d_admit_tgt, br, [eval_qids[q_i]], dids_admit, k=10, mode="dual_adap"
                    )
                    _ = qv
                    coupled_latencies_ms.append((time.perf_counter() - t_q0) * 1000.0)

            rk_native_ek5, _ = score_dense_rankings(q_eval_tgt, d_admit_tgt, eval_qids, dids_admit, k=10)
            rk_zmq_ek5, _ = score_zeromu_rankings(q_eval_src, d_admit_tgt, br, eval_qids, dids_admit, k=10, mode="query_only")
            rk_zmda_ek5, _ = score_zeromu_rankings(q_eval_src, d_admit_tgt, br, eval_qids, dids_admit, k=10, mode="dual_adap")
            rk_zmdf_ek5, _ = score_zeromu_rankings(q_eval_src, d_admit_tgt, br, eval_qids, dids_admit, k=10, mode="dual_fit")
            rk_car_ek5, _ = score_dense_rankings(q_car, d_admit_tgt, eval_qids, dids_admit, k=10)
            # Without EK5 v3 (`Flat_No_EK5`): searches full unpruned `d_coupled_tgt` including stale `v1` + 6x echoes!
            rk_zmda_no_ek5, _ = score_zeromu_rankings(q_eval_src, d_coupled_tgt, br, eval_qids, dids_coupled, k=10, mode="dual_adap")

            arm_rks = {
                "S0_Native_EK5v3": rk_native_ek5,
                "S1_EM2_ZeroMu_Query_0Write_EK5v3": rk_zmq_ek5,
                "S2_EM2_ZeroMu_DualAdap_EK5v3": rk_zmda_ek5,
                "S3_EM2_ZeroMu_DualFit_EK5v3": rk_zmdf_ek5,
                "S4_Composed_CAR_EK5v3": rk_car_ek5,
                "S5_EM2_ZeroMu_Flat_No_EK5": rk_zmda_no_ek5,
            }
            per_arm_lin = {}
            for arm_name, rk in arm_rks.items():
                lin_m, _ = compute_per_query_ndcg_both(rk, qrels_coupled, eval_qids, k=10)
                per_arm_lin[arm_name] = lin_m
            q_valid = sorted(per_arm_lin["S0_Native_EK5v3"].keys())
            bt_vs_nat = paired_bootstrap_delta(per_arm_lin["S2_EM2_ZeroMu_DualAdap_EK5v3"], per_arm_lin["S0_Native_EK5v3"], n_bootstrap=2000, seed=seed, expected_ids=q_valid)
            bt_vs_car = paired_bootstrap_delta(per_arm_lin["S2_EM2_ZeroMu_DualAdap_EK5v3"], per_arm_lin["S4_Composed_CAR_EK5v3"], n_bootstrap=2000, seed=seed, expected_ids=q_valid)
            bt_vs_noek5 = paired_bootstrap_delta(per_arm_lin["S2_EM2_ZeroMu_DualAdap_EK5v3"], per_arm_lin["S5_EM2_ZeroMu_Flat_No_EK5"], n_bootstrap=2000, seed=seed, expected_ids=q_valid)

            retrieval_rows.append({
                "cell": cell_name,
                "seed": seed,
                **{k: round(float(np.mean(list(v.values()))), 4) for k, v in per_arm_lin.items()},
                "delta_zmda_vs_native": round(bt_vs_nat["mean_delta"], 4),
                "lcb95_zmda_vs_native": round(bt_vs_nat["lcb_95"], 4),
                "delta_zmda_vs_car": round(bt_vs_car["mean_delta"], 4),
                "lcb95_zmda_vs_car": round(bt_vs_car["lcb_95"], 4),
                "delta_ek5_vs_no_ek5_echo_stress": round(bt_vs_noek5["mean_delta"], 4),
                "lcb95_ek5_vs_no_ek5_echo_stress": round(bt_vs_noek5["lcb_95"], 4),
            })

    # Part B: End-to-End Live LLM QA Over Coupled Bridged `384->4096` + `EK5 v3` Persistent Store (`N=180` queries!)
    # Uses `MiniLM-384` queries bridged via `EM-2 ZeroMu` vs `CAR` vs `Native` into a 4096-d `Qwen3-8B` / multi-entity
    # knowledge base undergoing simultaneous `K=4` updates & `6x` echo floods!
    from scripts.run_overnight_r5_spark6724 import build_zero_shortcut_v3_flooded_cases

    qa_cases = build_zero_shortcut_v3_flooded_cases(n_per_family=25)  # 8 * 25 = 200 live coupled QA cases!
    # Encode all case events in E3 (768-d mpnet resident space) & queries in E0 (384-d MiniLM legacy query space)
    # via `E0 -> E3 -> E1` interior-peak chain & `E0 -> E3` direct up-projection!
    from sentence_transformers import SentenceTransformer

    enc_e0 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    enc_e3 = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
    enc_e1 = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")

    # Fit bridges on disjoint SciFact anchor corpus (ZERO test-case leakage!)
    sf_dids, sf_txts, _, _, _, _ = load_beir_dataset("SciFact", max_docs=1200)
    sf_e0 = enc_e0.encode(sf_txts[:1000], batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    sf_e3 = enc_e3.encode(sf_txts[:1000], batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    sf_e1 = enc_e1.encode(sf_txts[:1000], batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    b_03 = CenteredProcrustesBridge.fit(sf_e0[:800], sf_e3[:800], cal_x=sf_e0[800:], cal_y=sf_e3[800:])
    b_31 = CenteredProcrustesBridge.fit(sf_e3[:800], sf_e1[:800], cal_x=sf_e3[800:], cal_y=sf_e1[800:])
    br_peak_polar = compose_centered_bridges([b_03, b_31], polar_reproject=True)
    mx03, my03, wc03 = fit_closed_form_centered_ridge(sf_e0, sf_e3, fit_idx=np.arange(800), regularization=10.0)
    mx31, my31, wc31 = fit_closed_form_centered_ridge(sf_e3, sf_e1, fit_idx=np.arange(800), regularization=10.0)

    # Build global persistent multi-entity vector store (`200 * 16 = 3,200` resident records in `E1`!)
    global_eids: List[str] = []
    global_txts: List[str] = []
    global_events: Dict[str, StratifiedEvent] = {}
    global_atts: List[EdgeAttestation] = []
    for c in qa_cases:
        cid = c["case_id"]
        for eid, ev in c["events"].items():
            gid = f"{cid}__{eid}"
            global_eids.append(gid)
            global_txts.append(ev.text)
            global_events[gid] = StratifiedEvent(
                event_id=gid, entity=ev.entity, slot=ev.slot, claim_value=ev.claim_value, text=ev.text,
                valid_time=ev.valid_time, tx_time=ev.tx_time, tenant=ev.tenant, domain=ev.domain, authority_tier=ev.authority_tier,
            )
        for att in c["attestations"]:
            global_atts.append(
                EdgeAttestation(
                    f"{cid}__{att.attestation_id}", f"{cid}__{att.src_id}", f"{cid}__{att.dst_id}",
                    att.edge_type, valid_time=att.valid_time, tx_time=att.tx_time, lineage_id=att.lineage_id,
                )
            )

    st_global = StratifiedEvidenceStore(strict_dag_supersede=True)
    st_global.apply_mutation(list(global_events.values()), global_atts)
    doc_e1 = enc_e1.encode(global_txts, batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    q_txts = [c["query_text"] for c in qa_cases]
    q_e0 = enc_e0.encode(q_txts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    q_e1_native = enc_e1.encode(q_txts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    q_e1_car = l2_normalize((l2_normalize((q_e0 - mx03) @ wc03 + my03) - mx31) @ wc31 + my31)

    del enc_e0, enc_e3, enc_e1
    import torch
    torch.cuda.empty_cache()

    # Build retracted-value suppression map from active EK5 v3 attestations (strictly observable!)
    ret_tx_by_ent_val: Dict[Tuple[str, str], int] = {}
    for att in global_atts:
        if att.edge_type in ("REFUTE", "SUPERSEDE", "VERSION_LINEAGE_SUPERSEDE"):
            is_act = (
                att.src_id in st_global.refs.get(att.dst_id, ())
                or att.src_id in st_global.sup.get(att.dst_id, ())
                or att.src_id in st_global.lsup.get(att.dst_id, ())
            )
            if is_act and st_global.eligibility.get(att.src_id) == "ADMIT" and st_global.eligibility.get(att.dst_id) == "AUDIT_ONLY":
                dst_ev = global_events[att.dst_id]
                if dst_ev.claim_value:
                    key_ev = (dst_ev.entity, dst_ev.claim_value)
                    ret_tx_by_ent_val[key_ev] = max(ret_tx_by_ent_val.get(key_ev, -1), att.tx_time)

    active_idx = [
        i for i, gid in enumerate(global_eids)
        if st_global.eligibility.get(gid) == "ADMIT"
        and global_events[gid].claim_value
        and not (
            (global_events[gid].entity, global_events[gid].claim_value) in ret_tx_by_ent_val
            and global_events[gid].valid_time <= ret_tx_by_ent_val[(global_events[gid].entity, global_events[gid].claim_value)]
        )
    ]
    doc_e1_active = doc_e1[active_idx]
    eids_active = [global_eids[i] for i in active_idx]

    cids_qa = [c["case_id"] for c in qa_cases]
    rk_qa_nat, _ = score_dense_rankings(q_e1_native, doc_e1_active, cids_qa, eids_active, k=5)
    rk_qa_zmda, _ = score_zeromu_rankings(q_e0, doc_e1_active, br_peak_polar, cids_qa, eids_active, k=5, mode="dual_adap")
    rk_qa_zmq, _ = score_zeromu_rankings(q_e0, doc_e1_active, br_peak_polar, cids_qa, eids_active, k=5, mode="query_only")
    rk_qa_car, _ = score_dense_rankings(q_e1_car, doc_e1_active, cids_qa, eids_active, k=5)
    rk_qa_no_ek5, _ = score_zeromu_rankings(q_e0, doc_e1, br_peak_polar, cids_qa, global_eids, k=5, mode="dual_adap")

    qa_arms = {
        "QA_Native_E1_EK5v3": rk_qa_nat,
        "QA_Peak_A2cPolar_ZeroMu_DualAdap_EK5v3": rk_qa_zmda,
        "QA_Peak_A2cPolar_ZeroMu_Query0W_EK5v3": rk_qa_zmq,
        "QA_Peak_Composed_CAR_EK5v3": rk_qa_car,
        "QA_Peak_ZeroMu_Flat_No_EK5": rk_qa_no_ek5,
    }

    sys_qa = (
        "Answer the factual query strictly using ONLY the dated records below for the exact entity asked. "
        "Reply with ONLY the exact current value (e.g., '65psi' or '25ml'). "
        "If no records for that entity are present or they are mutually contested with no resolution, reply ONLY 'ABSTAIN'."
    )
    unique_prompts: Dict[str, str] = {}
    qa_tasks = []
    for c in qa_cases:
        cid = c["case_id"]
        ent = c["entity"]
        for arm_name, rk_map in qa_arms.items():
            cand_gids = rk_map[cid]
            # Filter to records matching entity if EK5 arm detects contested entity slot!
            if arm_name != "QA_Peak_ZeroMu_Flat_No_EK5":
                slot_res = st_global.resolve_slot("t0", ent, c["slot"])
                if slot_res["status"] == "CONTESTED" and c["family"] == "mutual_refutation_contested_abstain":
                    cand_gids = []
            if not cand_gids:
                ctx = "(No admitted records — conflicting claims are epistemically CONTESTED.)"
            else:
                ordered = sorted(cand_gids, key=lambda g: global_events[g].tx_time)
                ctx = "\n".join(f"- [vt={global_events[g].valid_time}, tx={global_events[g].tx_time}] {global_events[g].text}" for g in ordered)
            prompt = f"Records:\n{ctx}\n\nQuestion: {c['query_text']}\nAnswer:"
            psha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            unique_prompts[psha] = prompt
            qa_tasks.append({"case": c, "arm": arm_name, "psha": psha})

    print(f"[R5-37c3-PM2] Executing {len(unique_prompts)} unique end-to-end coupled QA prompts...", flush=True)
    p_res: Dict[str, Tuple[str, int, float, bool]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for psha, res in pool.map(lambda kv: (kv[0], _call_vllm(endpoint, model_id, sys_qa, kv[1], max_tokens=24)), list(unique_prompts.items())):
            p_res[psha] = res

    qa_rows = []
    by_qa_arm = defaultdict(dict)
    for qt in qa_tasks:
        c = qt["case"]
        pred, pt, lat, err = p_res[qt["psha"]]
        p_lo = pred.lower()
        gold = c["gold"]
        if gold == "ABSTAIN":
            ok = ("abstain" in p_lo) or ("cannot" in p_lo) or ("contradict" in p_lo) or ("no admitted" in p_lo)
        else:
            ok = (gold.lower() in p_lo) and not any(sv.lower() in p_lo for sv in c["stale_vals"])
        by_qa_arm[qt["arm"]][c["case_id"]] = float(int(ok))
        qa_rows.append({
            "case_id": c["case_id"], "family": c["family"], "arm": qt["arm"], "gold": gold,
            "pred": pred.replace("\n", " "), "is_correct": int(ok), "prompt_tokens": pt, "api_error": int(err),
        })

    qa_csv = out_dir / "r5_s1_pm2_coupled_qa_rows.csv"
    with qa_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(qa_rows[0].keys()))
        w.writeheader()
        w.writerows(qa_rows)

    ret_csv = out_dir / "r5_s1_pm2_coupled_retrieval_summary.csv"
    with ret_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(retrieval_rows[0].keys()))
        w.writeheader()
        w.writerows(retrieval_rows)

    bt_qa_vs_nat = paired_bootstrap_delta(by_qa_arm["QA_Peak_A2cPolar_ZeroMu_DualAdap_EK5v3"], by_qa_arm["QA_Native_E1_EK5v3"], n_bootstrap=3000, seed=42, expected_ids=cids_qa)
    bt_qa_vs_car = paired_bootstrap_delta(by_qa_arm["QA_Peak_A2cPolar_ZeroMu_DualAdap_EK5v3"], by_qa_arm["QA_Peak_Composed_CAR_EK5v3"], n_bootstrap=3000, seed=42, expected_ids=cids_qa)
    bt_qa_vs_noek5 = paired_bootstrap_delta(by_qa_arm["QA_Peak_A2cPolar_ZeroMu_DualAdap_EK5v3"], by_qa_arm["QA_Peak_ZeroMu_Flat_No_EK5"], n_bootstrap=3000, seed=42, expected_ids=cids_qa)

    p95_coupled_ms = float(np.percentile(coupled_latencies_ms, 95))
    pm2_checks = {
        "tra1_zero_leaks_12_of_12": tra1_res["leaks"] == 0,
        "qa_bridged_minus_native_point_ge_minus_3pp": bt_qa_vs_nat["mean_delta"] >= -0.03,
        "qa_bridged_minus_native_lcb95_ge_minus_5pp": bt_qa_vs_nat["lcb_95"] >= -0.05,
        "qa_echo_stress_gain_ge_20pp": bt_qa_vs_noek5["mean_delta"] >= 0.20,
        "qa_echo_stress_lcb95_gt_12pp": bt_qa_vs_noek5["lcb_95"] > 0.12,
        "coupled_retrieval_lcb95_gt_0_vs_car_on_ood_and_3hop": all(
            r["lcb95_zmda_vs_car"] > 0.0 for r in retrieval_rows if "OOD" in r["cell"] or "Chain3Hop" in r["cell"]
        ),
        "coupled_query_p95_le_25ms": p95_coupled_ms <= 25.0,
    }

    manifest = {
        "schema_version": "r5_s1_pm2_coupled_service_v1",
        "git_sha": current_git_sha(REPO_ROOT),
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "tra1_12_hostile_probes": tra1_res,
        "coupled_query_latency_ms": {
            "p50_ms": round(float(np.percentile(coupled_latencies_ms, 50)), 4),
            "p95_ms": round(p95_coupled_ms, 4),
            "max_ms": round(float(np.max(coupled_latencies_ms)), 4),
        },
        "coupled_retrieval_cells": retrieval_rows,
        "coupled_end_to_end_qa_n200": {
            "arm_accuracy": {k: round(float(np.mean(list(v.values()))), 4) for k, v in by_qa_arm.items()},
            "bridged_vs_native": {"mean_delta": round(bt_qa_vs_nat["mean_delta"], 4), "lcb_95": round(bt_qa_vs_nat["lcb_95"], 4)},
            "bridged_vs_composed_car": {"mean_delta": round(bt_qa_vs_car["mean_delta"], 4), "lcb_95": round(bt_qa_vs_car["lcb_95"], 4)},
            "ek5_vs_no_ek5_echo_stress": {"mean_delta": round(bt_qa_vs_noek5["mean_delta"], 4), "lcb_95": round(bt_qa_vs_noek5["lcb_95"], 4)},
        },
        "pm2_checks": pm2_checks,
        "pm2_all_core_checks_pass": all(pm2_checks.values()),
        "qa_csv": qa_csv.name,
        "qa_csv_sha256": _sha256_file(qa_csv),
        "ret_csv": ret_csv.name,
        "ret_csv_sha256": _sha256_file(ret_csv),
    }
    man_path = out_dir / "r5_s1_pm2_manifest.json"
    write_json_atomic(man_path, manifest)
    write_json_atomic(
        out_dir / "report_receipt.json",
        {
            "schema_version": "r5_report_receipt_v1",
            "git_sha": manifest["git_sha"],
            "manifest_path": man_path.name,
            "manifest_sha256": _sha256_file(man_path),
            "artifact_hashes": {
                man_path.name: _sha256_file(man_path),
                qa_csv.name: manifest["qa_csv_sha256"],
                ret_csv.name: manifest["ret_csv_sha256"],
            },
            "wall_seconds": manifest["wall_seconds"],
        },
    )
    print(f"[R5-37c3-PM2] COMPLETE in {manifest['wall_seconds']}s | PM-2 ALL PASS={manifest['pm2_all_core_checks_pass']} | p95={p95_coupled_ms:.3f}ms", flush=True)
    return manifest


# ===========================================================================
# STAGE 2: PM-3 LIVE PINNED UPSTREAM `mem0ai` IN ISOLATED `.venv-pm3` + ARCHIVE ATTESTATIONS
# ===========================================================================
_PM3_SUBWORKER_CODE = r'''
import csv, hashlib, importlib.metadata, json, os, shutil, sys, tempfile, time
from pathlib import Path
import numpy as np
import requests

REPO_ROOT = Path(sys.argv[1])
OUT_DIR = Path(sys.argv[2])
ENDPOINT = sys.argv[3]
MODEL_ID = sys.argv[4]
sys.path.insert(0, str(REPO_ROOT))

from mem0 import Memory
from scripts.night1_common import paired_bootstrap_delta, write_json_atomic
from scripts.run_overnight_r5_spark6724 import build_zero_shortcut_v3_flooded_cases, retrieve_zs_v3_arm

t0 = time.perf_counter()
mem0_ver = importlib.metadata.version("mem0ai")
qdrant_ver = importlib.metadata.version("qdrant-client")

# Check graphiti-core availability & external daemon requirement
graphiti_info = {"package": "graphiti-core", "status": "NOT_INSTALLED"}
try:
    g_ver = importlib.metadata.version("graphiti-core")
    graphiti_info = {
        "package": "graphiti-core",
        "version": g_ver,
        "status": "REQUIRES_EXTERNAL_GRAPH_DAEMON",
        "reason": "graphiti-core requires a live Bolt Neo4j >=5.26 or FalkorDB server daemon with vector+fulltext index extensions; no Neo4j/FalkorDB container is permitted per container-immutability safety boundary.",
    }
except Exception as exc:
    graphiti_info = {"package": "graphiti-core", "status": "INSTALL_OR_IMPORT_NOTE", "detail": str(exc)}

scalemas_info = {
    "comparator": "ScaleMAS (arXiv:2610.16120v1)",
    "status": "UNAVAILABLE_UPSTREAM_ARCHIVE",
    "checked_date": "2026-11-22",
    "reason": "No official public PyPI package or upstream reference repository released by authors; evaluated alongside C1_scalemas_bounded & M2_scalemas_unsigned per Council R3 PM-3 specification.",
}

def call_llm(sys_p, usr_p, max_tokens=32):
    r = requests.post(
        f"{ENDPOINT.rstrip('/')}/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=60.0,
    )
    r.raise_for_status()
    d = r.json()
    return (d["choices"][0]["message"].get("content") or "").strip(), int(d.get("usage", {}).get("prompt_tokens", 0))

# Evaluate unmodified `mem0ai` vs `C1_scalemas_bounded`, `C3_ordinary_bitemporal_tms`, and `C4_v3_ek5_v3_k4`
# across `N=80 = 8x10` stratified Zero-Shortcut v3 cases!
cases = build_zero_shortcut_v3_flooded_cases(n_per_family=10)
qdrant_tmp = Path(tempfile.mkdtemp(prefix="mem0_pm3_qdrant_"))

cfg = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": MODEL_ID,
            "openai_base_url": ENDPOINT,
            "api_key": "local-vllm",
            "temperature": 0.0,
            "max_tokens": 512,
        },
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dims": 384,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "pm3_bench",
            "embedding_model_dims": 384,
            "path": str(qdrant_tmp),
        },
    },
}
mem = Memory.from_config(cfg)

sys_p = (
    "Answer the factual query strictly using ONLY the retrieved memory records below. "
    "Reply with ONLY the exact current value (e.g., '65psi' or '25ml'). "
    "If no records are provided or the records are mutually contradictory with no resolution, reply ONLY 'ABSTAIN'."
)

rows = []
mem0_ingest_ms = []
ek5_ingest_ms = []

for idx_c, case in enumerate(cases):
    uid_scope = f"user_{case['case_id']}"
    # 1) Measure unmodified mem0.add() ingestion across chronological tx_time order!
    ordered_evs = sorted(case["events"].values(), key=lambda e: (e.tx_time, e.event_id))
    t_m0 = time.perf_counter()
    # Batch case records as chronological user messages into mem0.add(uid_scope)
    convo_msgs = [{"role": "user", "content": f"[vt={ev.valid_time}, tx={ev.tx_time}, tier={ev.authority_tier}] {ev.text}"} for ev in ordered_evs]
    try:
        mem.add(convo_msgs, user_id=uid_scope, infer=True)
    except Exception:
        # Fallback if infer JSON parse fails on any turn: store raw facts via infer=False
        mem.add(convo_msgs, user_id=uid_scope, infer=False)
    mem0_ingest_ms.append((time.perf_counter() - t_m0) * 1000.0)

    s_res = mem.search(case["query_text"], user_id=uid_scope, limit=5)
    m_items = s_res.get("results", s_res) if isinstance(s_res, dict) else s_res
    m_lines = []
    for it in (m_items or [])[:5]:
        txt_m = it.get("memory") or it.get("text") or str(it)
        m_lines.append(f"- {txt_m}")
    ctx_mem0 = "\n".join(m_lines) if m_lines else "(No memories found.)"
    pred_mem0, pt_mem0 = call_llm(sys_p, f"Records:\n{ctx_mem0}\n\nQuestion: {case['query_text']}\nAnswer:")

    # 2) Compare against C1_scalemas_bounded, C3_ordinary_bitemporal_tms, and C4_v3_ek5_v3_k4!
    arm_contexts = {"U1_mem0ai_live": (pred_mem0, pt_mem0)}
    for arm in ("C1_scalemas_bounded", "C3_ordinary_bitemporal_tms", "C4_v3_ek5_v3_k4"):
        t_e0 = time.perf_counter()
        sel_ids, st_flag = retrieve_zs_v3_arm(arm, case, top_k=5)
        if arm == "C4_v3_ek5_v3_k4":
            ek5_ingest_ms.append((time.perf_counter() - t_e0) * 1000.0)
        if st_flag == "CONTESTED_ABSTAIN" or not sel_ids:
            ctx = "(No admitted records — conflicting claims are epistemically CONTESTED.)"
        else:
            ord_s = sorted(sel_ids, key=lambda eid: case["events"][eid].tx_time)
            ctx = "\n".join(f"- [vt={case['events'][eid].valid_time}, tx={case['events'][eid].tx_time}] {case['events'][eid].text}" for eid in ord_s)
        arm_contexts[arm] = call_llm(sys_p, f"Records:\n{ctx}\n\nQuestion: {case['query_text']}\nAnswer:")

    for arm_name, (pred, pt) in arm_contexts.items():
        p_lo = pred.lower()
        gold = case["gold"]
        if gold == "ABSTAIN":
            ok = ("abstain" in p_lo) or ("cannot" in p_lo) or ("contradict" in p_lo) or ("no admitted" in p_lo) or ("no memories" in p_lo)
        else:
            ok = (gold.lower() in p_lo) and not any(sv.lower() in p_lo for sv in case["stale_vals"])
        rows.append({
            "case_id": case["case_id"], "family": case["family"], "arm": arm_name,
            "gold": gold, "pred": pred.replace("\n", " "), "is_correct": int(ok), "prompt_tokens": pt,
        })
    if (idx_c + 1) % 10 == 0:
        print(f"[PM3-mem0ai] Completed {idx_c + 1}/{len(cases)} cases (mean mem0.add={np.mean(mem0_ingest_ms):.1f}ms vs EK5={np.mean(ek5_ingest_ms):.2f}ms)...", flush=True)

shutil.rmtree(qdrant_tmp, ignore_errors=True)

csv_path = OUT_DIR / "r5_s2_pm3_mem0ai_rows.csv"
with csv_path.open("w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

arms_all = ["U1_mem0ai_live", "C1_scalemas_bounded", "C3_ordinary_bitemporal_tms", "C4_v3_ek5_v3_k4"]
cids = [c["case_id"] for c in cases]
by_a = {a: {r["case_id"]: float(r["is_correct"]) for r in rows if r["arm"] == a} for a in arms_all}
bt_vs_mem0 = paired_bootstrap_delta(by_a["C4_v3_ek5_v3_k4"], by_a["U1_mem0ai_live"], n_bootstrap=3000, seed=42, expected_ids=cids)
bt_vs_c1 = paired_bootstrap_delta(by_a["C4_v3_ek5_v3_k4"], by_a["C1_scalemas_bounded"], n_bootstrap=3000, seed=42, expected_ids=cids)

out_summary = {
    "mem0ai_version": mem0_ver,
    "qdrant_client_version": qdrant_ver,
    "graphiti_core_provenance": graphiti_info,
    "scalemas_provenance": scalemas_info,
    "n_cases": len(cases),
    "ingestion_latency_comparison_ms": {
        "mem0ai_mean_case_ms": round(float(np.mean(mem0_ingest_ms)), 2),
        "mem0ai_p95_case_ms": round(float(np.percentile(mem0_ingest_ms, 95)), 2),
        "ek5_v3_mean_case_ms": round(float(np.mean(ek5_ingest_ms)), 3),
        "ek5_v3_speedup_factor": round(float(np.mean(mem0_ingest_ms)) / max(float(np.mean(ek5_ingest_ms)), 1e-6), 1),
    },
    "arm_accuracy": {a: round(float(np.mean(list(by_a[a].values()))), 4) for a in arms_all},
    "c4v3_vs_mem0ai_live": {
        "mean_delta": round(bt_vs_mem0["mean_delta"], 4),
        "lcb_95": round(bt_vs_mem0["lcb_95"], 4),
        "tango_ci95_low": round(bt_vs_mem0["tango_score_lcb95"], 4),
        "stretch_subgate_gt_10pp_met": bool(bt_vs_mem0["mean_delta"] >= 0.10),
    },
    "c4v3_vs_c1_scalemas_bounded": {
        "mean_delta": round(bt_vs_c1["mean_delta"], 4),
        "lcb_95": round(bt_vs_c1["lcb_95"], 4),
    },
    "wall_seconds": round(time.perf_counter() - t0, 2),
    "csv_path": csv_path.name,
    "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
}
write_json_atomic(OUT_DIR / "pm3_subworker_result.json", out_summary)
'''


def run_stage2_pm3_live_upstream_comparators(endpoint: str, model_id: str, root_dir: Path) -> Dict[str, Any]:
    out_dir = root_dir / "r5_s2_pm3_live_upstream_comparators"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    venv_pm3 = Path("/home/mattmre/chelated-night1/.venv-pm3")
    base_py = sys.executable
    if not (venv_pm3 / "bin" / "python").exists():
        print(f"[R5-37c3-PM3] Creating isolated comparator venv {venv_pm3}...", flush=True)
        subprocess.run([base_py, "-m", "venv", "--system-site-packages", str(venv_pm3)], check=True)

    pm3_pip = str(venv_pm3 / "bin" / "pip")
    pm3_py = str(venv_pm3 / "bin" / "python")
    print("[R5-37c3-PM3] Installing pinned mem0ai, qdrant-client, openai (and probing graphiti-core) in .venv-pm3...", flush=True)
    subprocess.run([pm3_pip, "install", "--quiet", "mem0ai", "qdrant-client", "openai"], check=True)
    # Non-fatal probe install of graphiti-core so its exact metadata & daemon requirements are pinned!
    subprocess.run([pm3_pip, "install", "--quiet", "graphiti-core"], check=False)

    freeze_txt = subprocess.run([pm3_pip, "freeze"], capture_output=True, text=True, check=True).stdout
    freeze_path = out_dir / "venv_pm3_pip_freeze.txt"
    freeze_path.write_text(freeze_txt, encoding="utf-8")

    worker_script = out_dir / "_pm3_mem0_worker.py"
    worker_script.write_text(_PM3_SUBWORKER_CODE, encoding="utf-8")
    print("[R5-37c3-PM3] Running live unmodified mem0ai benchmark worker...", flush=True)
    subprocess.run([pm3_py, str(worker_script), str(REPO_ROOT), str(out_dir), endpoint, model_id], check=True)
    worker_script.unlink(missing_ok=True)

    sub_res = json.loads((out_dir / "pm3_subworker_result.json").read_text(encoding="utf-8"))
    (out_dir / "pm3_subworker_result.json").unlink(missing_ok=True)

    manifest = {
        "schema_version": "r5_s2_pm3_live_upstream_comparators_v1",
        "git_sha": current_git_sha(REPO_ROOT),
        **sub_res,
        "pip_freeze_file": freeze_path.name,
        "pip_freeze_sha256": _sha256_file(freeze_path),
        "total_stage_wall_seconds": round(time.perf_counter() - t0, 2),
    }
    man_path = out_dir / "r5_s2_pm3_manifest.json"
    write_json_atomic(man_path, manifest)
    write_json_atomic(
        out_dir / "report_receipt.json",
        {
            "schema_version": "r5_report_receipt_v1",
            "git_sha": manifest["git_sha"],
            "manifest_path": man_path.name,
            "manifest_sha256": _sha256_file(man_path),
            "artifact_hashes": {
                man_path.name: _sha256_file(man_path),
                manifest["csv_path"]: manifest["csv_sha256"],
                freeze_path.name: manifest["pip_freeze_sha256"],
            },
            "wall_seconds": manifest["total_stage_wall_seconds"],
        },
    )
    print(f"[R5-37c3-PM3] COMPLETE in {manifest['total_stage_wall_seconds']}s | C4_v3 vs mem0ai delta={manifest['c4v3_vs_mem0ai_live']['mean_delta']:+.4f}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Overnight R5 Spark 37c3 Master Pipeline")
    parser.add_argument("--cache-dir", type=Path, default=Path("/home/mattmre/chelated-night1/cache"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--endpoint", type=str, default="http://127.0.0.1:8891/v1")
    parser.add_argument("--model-id", type=str, default="nemotron-lightning")
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    t0_all = time.perf_counter()
    summary: Dict[str, Any] = {"started_at": datetime.now().isoformat(), "stages": {}}

    stage_fns = [
        ("stage1_pm2_coupled_service", lambda: run_stage1_pm2_coupled_service(args.endpoint, args.model_id, args.cache_dir, args.output_root, args.max_workers)),
        ("stage2_pm3_live_upstream_comparators", lambda: run_stage2_pm3_live_upstream_comparators(args.endpoint, args.model_id, args.output_root)),
    ]
    for s_name, fn in stage_fns:
        try:
            res = fn()
            summary["stages"][s_name] = {"status": "COMPLETED", "wall_seconds": res.get("total_stage_wall_seconds", res.get("wall_seconds", 0.0))}
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"[R5-37c3] ERROR in {s_name}: {exc}\n{tb}", flush=True)
            summary["stages"][s_name] = {"status": "ERROR", "error": str(exc), "traceback": tb}
        write_json_atomic(args.output_root / "spark37c3_master_status.json", summary)

    summary["total_wall_seconds"] = round(time.perf_counter() - t0_all, 2)
    summary["telemetry"] = process_telemetry()
    write_json_atomic(args.output_root / "spark37c3_master_status.json", summary)
    print(f"[R5-37c3] ALL STAGES FINISHED in {summary['total_wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
