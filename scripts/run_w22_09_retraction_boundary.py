"""W22-09 runner: retraction boundary (Parts A, B, C + Crouzeix) per the preregistration.

Usage: python scripts/run_w22_09_retraction_boundary.py [--out-dir DIR] [--quick]

Writes ``<out>/w22_09_results.json``; bulk files (KV blocks, flat KV checkpoints) go to ``<out>/scratch/``.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import retraction_boundary as rb  # noqa: E402

KILL_LAT = 1.10
KILL_BYTES = 2.0


def _stats(xs: List[float]) -> Dict[str, float]:
    a = np.asarray(xs, dtype=float)
    return {"median": float(np.median(a)), "p90": float(np.quantile(a, 0.9)), "n": int(a.size)}


def _dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def _k2(treat: Dict[str, Any], base: Dict[str, Any], *, ek5: bool) -> Dict[str, Any]:
    lat_ok = base["latency_ms"]["median"] <= KILL_LAT * treat["latency_ms"]["median"]
    if ek5:
        err_ok = base["max_semantic_diffs"] == 0
    else:
        err_ok = base["max_err"] <= max(treat["max_err"], rb.EXACT_TOL)
    bytes_ok = base["bytes"] <= KILL_BYTES * treat["bytes"]
    return {
        "latency_ratio": base["latency_ms"]["median"] / treat["latency_ms"]["median"],
        "bytes_ratio": base["bytes"] / treat["bytes"],
        "latency_ok": bool(lat_ok),
        "error_ok": bool(err_ok),
        "bytes_ok": bool(bytes_ok),
        "fires": bool(lat_ok and err_ok and bytes_ok),
    }


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------
def part_a(scratch: Path, seeds: List[int], sizes: List[int]) -> Dict[str, Any]:
    import chelated_ssd_kv_repo_cache as kvc
    from ek5_v3_stratified import EdgeAttestation, StratifiedEvent, StratifiedEvidenceStore

    per_cfg: Dict[int, Dict[str, Any]] = {}
    per_seed: List[Dict[str, Any]] = []
    for N in sizes:
        acc = {k: [] for k in ("T", "T_mut", "FLAT", "REPLAY")}
        errs = {k: [] for k in ("T_logit", "T_kv", "FLAT_logit", "FLAT_kv")}
        t_bytes = f_bytes = r_bytes = 0
        splice_rows = []
        refusals = 0
        for s in seeds:
            model = kvc.build_model(s)
            contract = kvc.contract_for(model)
            blocks = rb.kv_prompt(s, N)
            store = StratifiedEvidenceStore()
            store.apply_mutation([StratifiedEvent(b.version_id, authority_tier=1) for b in blocks])
            cache = kvc.SSDKVCache(scratch / "kv" / f"s{s}_N{N}", contract, store)
            _lg, kv_full = kvc.serve_chelated(model, cache, blocks, kvc.CacheStats())
            base_index = {k: dict(v) for k, v in cache.index.items()}
            base_store = rb.deep(store)
            tb = cache.bytes_on_disk
            flat = scratch / "flat" / f"s{s}_N{N}"
            flat.parent.mkdir(parents=True, exist_ok=True)
            fb = rb.flat_checkpoint_write(flat, kv_full, np.concatenate([b.tokens for b in blocks]))
            rbytes = int(sum(len(b.tokens) for b in blocks) * 8)
            t_bytes += tb
            f_bytes += fb
            r_bytes += rbytes
            seed_lat = {"T": [], "FLAT": []}
            for k in range(N):
                cf = [b for j, b in enumerate(blocks) if j != k]
                ref_logit, ref_kv = kvc.full_prefill(model, cf)
                # treatment: EK5 retraction of version k, then serve the counterfactual prompt
                cache.index = {kk: dict(v) for kk, v in base_index.items()}
                cache.store = rb.deep(base_store)
                rid = f"retract::{blocks[k].version_id}"
                _, t_mut = rb.time_call(
                    cache.store.apply_mutation,
                    [StratifiedEvent(rid, authority_tier=5)],
                    [EdgeAttestation(f"att::{rid}", rid, blocks[k].version_id, "SUPERSEDE")],
                )
                st = kvc.CacheStats()
                (lg_t, kv_t), t_ms = rb.time_call(kvc.serve_chelated, model, cache, cf, st)
                refusals += st.refused_retracted_blocks
                acc["T"].append(t_ms)
                acc["T_mut"].append(t_mut)
                seed_lat["T"].append(t_ms)
                errs["T_logit"].append(float(np.max(np.abs(lg_t - ref_logit))))
                errs["T_kv"].append(max(float(np.max(np.abs(kv_t[i][j] - ref_kv[i][j]))) for i in range(kvc.N_LAYERS) for j in range(2)))
                # B-FLAT
                (lg_f, kv_f), f_ms = rb.time_call(rb.flat_checkpoint_delete, model, flat, blocks, k, scratch / "flat" / "out.npy")
                acc["FLAT"].append(f_ms)
                seed_lat["FLAT"].append(f_ms)
                errs["FLAT_logit"].append(float(np.max(np.abs(lg_f - ref_logit))))
                errs["FLAT_kv"].append(max(float(np.max(np.abs(kv_f[i][j] - ref_kv[i][j]))) for i in range(kvc.N_LAYERS) for j in range(2)))
                # B-REPLAY
                _, r_ms = rb.time_call(kvc.full_prefill, model, cf)
                acc["REPLAY"].append(r_ms)
                # A1 splice probe
                splice_rows.append(rb.splice_probe(model, blocks, k, kv_full, ref_kv))
            per_seed.append({"N": N, "seed": s, "T_bytes": tb, "FLAT_bytes": fb,
                             "T_ms": _stats(seed_lat["T"]), "FLAT_ms": _stats(seed_lat["FLAT"])})
        n_seeds = len(seeds)
        treat = {"latency_ms": _stats(acc["T"]), "max_err": max(max(errs["T_logit"]), max(errs["T_kv"])),
                 "bytes": t_bytes / n_seeds}
        flat_arm = {"latency_ms": _stats(acc["FLAT"]), "max_err": max(max(errs["FLAT_logit"]), max(errs["FLAT_kv"])),
                    "bytes": f_bytes / n_seeds}
        replay_arm = {"latency_ms": _stats(acc["REPLAY"]), "max_err": 0.0, "bytes": r_bytes / n_seeds}
        sup = [r for r in splice_rows if r["suffix_rows"] > 0]
        dirty_frac = {
            f"layer{i}": float(np.mean([r["per_layer"][i]["dirty_rows"] / r["suffix_rows"] for r in sup])) if sup else None
            for i in range(kvc.N_LAYERS)
        }
        per_cfg[N] = {
            "treatment_T60": treat,
            "treatment_retraction_mutation_ms": _stats(acc["T_mut"]),
            "treatment_refused_retracted_blocks": refusals,
            "B_FLAT": flat_arm,
            "B_REPLAY_context": replay_arm,
            "errors": {k: max(v) for k, v in errs.items()},
            "A1_splice": {
                "deletions_with_suffix": len(sup),
                "mean_dirty_fraction_of_suffix_rows": dirty_frac,
                "min_dirty_fraction_layer1": float(min(r["per_layer"][1]["dirty_rows"] / r["suffix_rows"] for r in sup)) if sup else None,
                "max_abs_err_layer1": float(max(r["per_layer"][1]["max_abs"] for r in sup)) if sup else None,
                "prefix_rows_exact_all": all(r["prefix_rows_exact"] for r in splice_rows),
            },
            "K2_B_FLAT": _k2(treat, flat_arm, ek5=False),
        }
    return {"configs": {str(k): v for k, v in per_cfg.items()}, "per_seed": per_seed,
            "A3": a3_probe(scratch)}


def a3_probe(scratch: Path) -> Dict[str, Any]:
    import chelated_ssd_kv_repo_cache as kvc
    from ek5_v3_stratified import EdgeAttestation, StratifiedEvent, StratifiedEvidenceStore

    model = kvc.build_model(6001)
    contract = kvc.contract_for(model)
    rng = np.random.default_rng(909)
    P0 = kvc.FileVersion("P0", "p.py", rng.integers(0, kvc.VOCAB, 24), 0)
    toks = rng.integers(0, kvc.VOCAB, 24)
    v1 = kvc.FileVersion("v1", "f.py", toks.copy(), 1)
    v2 = kvc.FileVersion("v2", "g.py", toks.copy(), 2)
    S = kvc.FileVersion("S", "s.py", rng.integers(0, kvc.VOCAB, 24), 3)
    out: Dict[str, Any] = {"v1_v2_tokens_identical": bool(np.array_equal(v1.tokens, v2.tokens)),
                           "chain_keys_collide": kvc._prompt_chain([P0, v1, S])[1][0] == kvc._prompt_chain([P0, v2, S])[1][0]}

    def fresh(tag: str):
        store = StratifiedEvidenceStore()
        store.apply_mutation([StratifiedEvent(x, authority_tier=1) for x in ("P0", "v1", "v2", "S")])
        return kvc.SSDKVCache(scratch / "a3" / tag, contract, store), store

    def retract(store, vid):
        rid = f"retract::{vid}"
        store.apply_mutation([StratifiedEvent(rid, authority_tier=5)], [EdgeAttestation(f"att::{rid}", rid, vid, "SUPERSEDE")])

    for order, first, second in (("i", [P0, v1, S], [P0, v2, S]),
                                 ("ii", [P0, v2, S], [P0, v1, S])):
        cache, store = fresh(order)
        kvc.serve_chelated(model, cache, first, kvc.CacheStats())
        retract(store, "v1")
        st = kvc.CacheStats()
        lg, kv = kvc.serve_chelated(model, cache, second, st)
        ref, _ = kvc.full_prefill(model, second)
        request_has_retracted = any(store.eligibility.get(b.version_id) != "ADMIT" for b in second)
        served_blocks = st.blocks_hit
        out[order] = {
            "logit_max_abs_err": float(np.max(np.abs(lg - ref))),
            "a_logits_exact": bool(np.max(np.abs(lg - ref)) <= rb.EXACT_TOL),
            "blocks_hit": served_blocks,
            "refused_retracted_blocks": st.refused_retracted_blocks,
            "stale_served_blocks_counter": st.stale_served_blocks,
            "request_contains_retracted_version": request_has_retracted,
            "served_cached_block_whose_request_prefix_contains_retracted": bool(request_has_retracted and served_blocks >= 2),
            "eligibility_v1": store.eligibility.get("v1"),
        }
    return out


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------
def part_b(beam_rs: List[int], chain_ns: List[int], rng_seeds: List[int], n_del: int) -> Dict[str, Any]:
    from chelated_pv7_spine.ek5_reason_canonicalizer import reference_fresh_rebuild
    from ek5_v3_stratified import EdgeAttestation, StratifiedEvent

    logs = [rb.beam_log(r) for r in beam_rs] + [rb.chain_log(n) for n in chain_ns]
    out: Dict[str, Any] = {}
    for log in logs:
        t0 = time.perf_counter()
        warmed = rb.build_incremental(log)
        warm_ref = reference_fresh_rebuild(warmed)
        t_bytes = rb.store_bytes(warmed)
        l_bytes = rb.log_bytes(log)
        ckpts, geo_bytes = rb.build_geo_checkpoints(log, int(KILL_BYTES * t_bytes))
        att_turn = log.attestation_turn()
        att_ids = sorted(att_turn)
        lat = {k: [] for k in ("T", "REBUILD", "GEO", "REFUTE_APPEND")}
        diffs = {k: [] for k in ("T", "REBUILD", "GEO")}
        dirty = []
        geo_replayed_turns = []
        for seed in rng_seeds:
            rng = np.random.default_rng([seed, len(log.turns)])
            chosen = rng.choice(len(att_ids), size=min(n_del, len(att_ids)), replace=False)
            for ci in chosen:
                a = att_ids[int(ci)]
                st = rb.deep(warmed)
                delta, t_ms = rb.time_call(st.apply_mutation, withdraw_attestation_ids=[a])
                ref = reference_fresh_rebuild(st)
                lat["T"].append(t_ms)
                diffs["T"].append(rb.semantic_diffs(st, ref))
                dirty.append(len(delta.transitioned))
                rbld, r_ms = rb.time_call(rb.build_batch, log, None, 0, a)
                lat["REBUILD"].append(r_ms)
                diffs["REBUILD"].append(rb.semantic_diffs(rbld, ref))
                (geo, p), g_ms = rb.time_call(rb.geo_ckpt_delete, log, ckpts, att_turn[a], a)
                lat["GEO"].append(g_ms)
                diffs["GEO"].append(rb.semantic_diffs(geo, ref))
                geo_replayed_turns.append(len(log.turns) - p)
                src = warmed.attestations[a].src_id
                st2 = rb.deep(warmed)
                rid = f"refute::{src}::{seed}"
                _, ra_ms = rb.time_call(st2.apply_mutation, [StratifiedEvent(rid, authority_tier=5)],
                                        [EdgeAttestation(f"att::{rid}", rid, src, "REFUTE")])
                lat["REFUTE_APPEND"].append(ra_ms)
        treat = {"latency_ms": _stats(lat["T"]), "max_semantic_diffs": max(diffs["T"]), "bytes": t_bytes}
        reb = {"latency_ms": _stats(lat["REBUILD"]), "max_semantic_diffs": max(diffs["REBUILD"]), "bytes": l_bytes}
        geo = {"latency_ms": _stats(lat["GEO"]), "max_semantic_diffs": max(diffs["GEO"]), "bytes": geo_bytes,
               "checkpoint_positions": sorted(ckpts), "median_replayed_turns": float(np.median(geo_replayed_turns))}
        out[log.name] = {
            "events": log.n_events,
            "turns": len(log.turns),
            "attestations": len(att_ids),
            "warm_incremental_vs_fresh_semantic_diffs": rb.semantic_diffs(warmed, warm_ref),
            "treatment_TEK5": treat,
            "treatment_transitioned_nodes": _stats(dirty),
            "B_REBUILD_context": reb,
            "B_GEO_CKPT": geo,
            "REFUTE_APPEND_context_ms": _stats(lat["REFUTE_APPEND"]),
            "K2_B_GEO_CKPT": _k2(treat, geo, ek5=True),
            "K2_B_REBUILD_context_only": _k2(treat, reb, ek5=True),
            "wall_s": time.perf_counter() - t0,
        }
        print(f"  [B] {log.name}: T={treat['latency_ms']['median']:.2f}ms GEO={geo['latency_ms']['median']:.2f}ms "
              f"REB={reb['latency_ms']['median']:.2f}ms diffs T/G/R={treat['max_semantic_diffs']}/"
              f"{geo['max_semantic_diffs']}/{reb['max_semantic_diffs']} ({out[log.name]['wall_s']:.1f}s)", flush=True)
    out["P51_DIGEST"] = rb.digest_collision_probe()
    out["PEK5_ASOF"] = rb.asof_collision_probe()
    return out


# ---------------------------------------------------------------------------
# Part C
# ---------------------------------------------------------------------------
def part_c(ds: List[int], Ts: List[int], seeds: List[int], n_del: int) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    collisions: List[Dict[str, Any]] = []
    for regime in rb.REGIMES:
        for d in ds:
            for T in Ts:
                lat: Dict[str, List[float]] = {}
                err: Dict[str, List[float]] = {}
                refused = 0
                inv_attempts = 0
                work: Dict[str, List[int]] = {}
                for seed in seeds:
                    As, bs, meta = rb.generate_regime(regime, d, T, seed)
                    x0 = np.zeros(d)
                    X = rb.states_trajectory(As, bs, x0)
                    xT = X[T]
                    tree = rb.AffineSegmentTree(As, bs)
                    Phi, c = rb.suffix_products(As, bs)
                    geo = rb.geometric_checkpoint_positions(T)
                    powers = rb.binary_powers(As[0], T) if regime == "nonnormal-LTI" else None
                    rng = np.random.default_rng([seed, d, T, 77])
                    ks = rng.choice(T, size=min(n_del, T), replace=False)
                    eye = np.eye(d)
                    for k in (int(v) for v in ks):
                        ref_skip = rb.replay(As, bs, x0, skip=k)
                        ref_in = rb.replay(As, bs, x0, zero_input=k)

                        def rec(name, val, ms, ref, w):
                            lat.setdefault(name, []).append(ms)
                            err.setdefault(name, []).append(rb.rel_err(val, ref))
                            work.setdefault(name, []).append(w)

                        v, ms = rb.time_call(rb.replay, As, bs, x0, skip=k)
                        rec("RP_skip", v, ms, ref_skip, T - 1)
                        p = max(q for q in geo if q <= k)
                        v, ms = rb.time_call(rb.replay, As, bs, X[p], skip=k, start=p)
                        rec("CK-geo_skip", v, ms, ref_skip, T - p - 1)
                        v, ms = rb.time_call(rb.replay, As, bs, X[k], skip=k, start=k)
                        rec("CK-matched_skip", v, ms, ref_skip, T - k - 1)
                        v, ms = rb.time_call(rb.replay, As, bs, X[k], zero_input=k, start=k)
                        rec("CK-matched_input", v, ms, ref_in, T - k)
                        p2 = max(q for q in geo if q <= k)
                        v, ms = rb.time_call(rb.replay, As, bs, X[p2], zero_input=k, start=p2)
                        rec("CK-geo_input", v, ms, ref_in, T - p2)

                        A_k, b_k = As[k].copy(), bs[k].copy()

                        def st_del(A_new, b_new):
                            tree.update(k, A_new, b_new)
                            return tree.apply_root(x0)

                        v, ms = rb.time_call(st_del, eye, np.zeros(d))
                        rec("ST_skip", v, ms, ref_skip, tree.matmuls_last_update)
                        tree.update(k, A_k, b_k)
                        v, ms = rb.time_call(st_del, A_k, np.zeros(d))
                        rec("ST_input", v, ms, ref_in, tree.matmuls_last_update)
                        tree.update(k, A_k, b_k)

                        v, ms = rb.time_call(lambda: xT - Phi[k] @ b_k)
                        rec("SUB-TV_input", v, ms, ref_in, 1)
                        if powers is not None:
                            def sub_lti():
                                y, _mv = rb.apply_power(powers, T - 1 - k, b_k)
                                return xT - y
                            v, ms = rb.time_call(sub_lti)
                            rec("SUB-LTI_input", v, ms, ref_in, bin(T - 1 - k).count("1"))
                        for cert in (True, False):
                            name = "INV-cert_skip" if cert else "INV-nocert_skip"
                            (v, info), ms = rb.time_call(rb.inv_retract_skip, xT, Phi[k], c[k], A_k, b_k, certified=cert)
                            if cert:
                                inv_attempts += 1
                            if v is None:
                                refused += 1
                                lat.setdefault(name, []).append(ms)
                                continue
                            rec(name, v, ms, ref_skip, 1)
                    if regime == "singular-TV" and d == ds[0] and T == Ts[0]:
                        for s_step in meta["singular_steps"][:3]:
                            pair = rb.singular_collision_pair(As, bs, s_step)
                            bs2 = pair["bs_prime"]
                            xT2 = rb.replay(As, bs2, x0)
                            Phi2, c2 = rb.suffix_products(As, bs2)
                            truth1 = rb.replay(As, bs, x0, skip=s_step)
                            truth2 = rb.replay(As, bs2, x0, skip=s_step)
                            res = {"seed": seed, "singular_step": s_step, "null_singular_value": pair["null_singular_value"],
                                   "xT_rel_diff": rb.rel_err(xT2, xT),
                                   "retained_inputs_identical": bool(np.allclose(Phi2[s_step], Phi[s_step]) and np.allclose(c2[s_step], c[s_step])),
                                   "truths_rel_diff": rb.rel_err(truth2, truth1)}
                            for cert in (True, False):
                                o1, i1 = rb.inv_retract_skip(xT, Phi[s_step], c[s_step], As[s_step], bs[s_step], certified=cert)
                                o2, i2 = rb.inv_retract_skip(xT2, Phi2[s_step], c2[s_step], As[s_step], bs2[s_step], certified=cert)
                                tag = "cert" if cert else "nocert"
                                res[f"{tag}_refused"] = bool(i1["refused"] and i2["refused"])
                                res[f"{tag}_cond"] = i1["cond"]
                                if o1 is not None and o2 is not None:
                                    res[f"{tag}_err_H"] = rb.rel_err(o1, truth1)
                                    res[f"{tag}_err_Hprime"] = rb.rel_err(o2, truth2)
                                    res[f"{tag}_outputs_rel_diff"] = rb.rel_err(o2, o1)
                            collisions.append(res)
                n_pow = len(rb.binary_powers(np.eye(d), T))
                byt = {
                    "RP_skip": 8 * (T * d * d + T * d + d),
                    "CK-geo_skip": 8 * (T * d * d + T * d + d + len(rb.geometric_checkpoint_positions(T)) * d),
                    "CK-matched_skip": 8 * (T * d * d + T * d + (T + 1) * d),
                    "ST_skip": rb.AffineSegmentTree(np.zeros((T, d, d)), np.zeros((T, d))).retained_bytes() + 8 * d,
                    "SUB-TV_input": 8 * (d + T * d * d),
                    "SUB-LTI_input": 8 * (d + n_pow * d * d),
                    "INV-cert_skip": 8 * (d + 2 * T * d * d + 2 * T * d),
                }
                byt["INV-nocert_skip"] = byt["INV-cert_skip"]
                byt["CK-matched_input"] = byt["CK-matched_skip"]
                byt["CK-geo_input"] = byt["CK-geo_skip"]
                byt["ST_input"] = byt["ST_skip"]
                arms = {}
                for name in lat:
                    e = err.get(name, [])
                    arms[name] = {"latency_ms": _stats(lat[name]), "max_err": float(max(e)) if e else None,
                                  "median_err": float(np.median(e)) if e else None, "n_answered": len(e),
                                  "bytes": byt.get(name), "median_work": float(np.median(work[name])) if name in work else None}
                k2 = {}
                for tname, sem in (("SUB-TV_input", "input"), ("SUB-LTI_input", "input"), ("INV-cert_skip", "skip")):
                    if tname not in arms or arms[tname]["max_err"] is None:
                        continue
                    for bname in (f"ST_{sem}", f"CK-geo_{sem}"):
                        k2[f"{bname}_vs_{tname}"] = _k2(arms[tname], arms[bname], ek5=False)
                key = f"{regime}|d{d}|T{T}"
                rows[key] = {"arms": arms, "inv_refused": refused, "inv_cert_attempts": inv_attempts, "K2": k2}
                print(f"  [C] {key}: SUB-TV {arms['SUB-TV_input']['latency_ms']['median']*1e3:.1f}us "
                      f"ST {arms['ST_skip']['latency_ms']['median']*1e3:.1f}us CKgeo {arms['CK-geo_skip']['latency_ms']['median']*1e3:.1f}us "
                      f"INV refused {refused}/{inv_attempts}", flush=True)
    return {"configs": rows, "collision_pairs": collisions}


def crouzeix_part(ds: List[int]) -> Dict[str, Any]:
    import chelated_ssd_kv_repo_cache as kvc

    mats: List[Any] = []
    for d in ds:
        A, _b, _m = rb.generate_regime("nonnormal-LTI", d, 8, 0)
        mats.append((f"nonnormal-LTI|d{d}", "transition", A[0]))
        for regime in ("normal-TV", "nonnormal-TV", "singular-TV"):
            As, _b, meta = rb.generate_regime(regime, d, 256, 0)
            idx = list(np.linspace(0, 255, 8).astype(int))
            if regime == "singular-TV":
                idx = meta["singular_steps"][:4] + idx[:4]
            for t in idx:
                mats.append((f"{regime}|d{d}|t{t}", "transition", As[t]))
    for s in (6001, 6002, 6003):
        m = kvc.build_model(s)
        for nm in ("wq", "wk", "wv", "wo"):
            for layer, W in enumerate(getattr(m, nm)):
                mats.append((f"TinyModel{s}.{nm}[{layer}]", "weights", W))
    rows = []
    for i, (name, kind, A) in enumerate(mats):
        r = rb.crouzeix_check(A, seed=i)
        r.update({"matrix": name, "kind": kind})
        rows.append(r)
    return {
        "constant_theorem_1_plus_sqrt2": rb.CROUZEIX_PALENCIA_CONSTANT,
        "constant_conjectured": rb.CROUZEIX_CONJECTURED_CONSTANT,
        "n_matrices": len(rows),
        "max_ratio_W": float(max(r["max_ratio_W"] for r in rows)),
        "all_hold_1_plus_sqrt2": all(r["bound_1_plus_sqrt2_holds"] for r in rows),
        "all_hold_2": all(r["conjectured_2_holds"] for r in rows),
        "max_power_ratio_rho": float(max(r["max_power_ratio_rho"] for r in rows)),
        "rows": rows,
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts" / "pv7_w23b_w22_09"))
    ap.add_argument("--quick", action="store_true", help="tiny grid for smoke testing")
    ap.add_argument("--beam-r64", action="store_true", help="add the optional R=64 W-BEAM size")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    # a fresh per-run subfolder: the runner never deletes anything, earlier runs stay until the owner removes them
    scratch = out / "scratch" / time.strftime("r%m%d%H%M%S")
    scratch.mkdir(parents=True, exist_ok=False)
    t0 = time.perf_counter()
    if args.quick:
        cfg = dict(a_seeds=[6001], a_sizes=[4], beam=[1], chain=[32], b_seeds=[0], b_del=4,
                   c_d=[4], c_T=[32], c_seeds=[0], c_del=4, cz_d=[4])
    else:
        cfg = dict(a_seeds=[6001, 6002, 6003], a_sizes=[8, 16, 32], beam=[1, 4, 16] + ([64] if args.beam_r64 else []),
                   chain=[256, 1024, 4096], b_seeds=[0, 1, 2], b_del=32, c_d=[8, 32], c_T=[256, 2048],
                   c_seeds=[0, 1, 2], c_del=64, cz_d=[8, 32])
    timings = {}
    t = time.perf_counter()
    print("Part A", flush=True)
    A = part_a(scratch, cfg["a_seeds"], cfg["a_sizes"])
    timings["A_s"] = time.perf_counter() - t
    t = time.perf_counter()
    print("Part B", flush=True)
    B = part_b(cfg["beam"], cfg["chain"], cfg["b_seeds"], cfg["b_del"])
    timings["B_s"] = time.perf_counter() - t
    t = time.perf_counter()
    print("Part C", flush=True)
    C = part_c(cfg["c_d"], cfg["c_T"], cfg["c_seeds"], cfg["c_del"])
    timings["C_s"] = time.perf_counter() - t
    t = time.perf_counter()
    print("Crouzeix", flush=True)
    CZ = crouzeix_part(cfg["cz_d"])
    timings["Crouzeix_s"] = time.perf_counter() - t

    a3 = A["A3"]
    k1 = {
        "P51_DIGEST_bundle_path": bool(B["P51_DIGEST"]["digest_equal"] and B["P51_DIGEST"]["questions_with_different_answers"] > 0),
        "PEK5_ASOF_native_api": bool(B["PEK5_ASOF"]["collision"] and not (B["PEK5_ASOF"]["h1_asof_exact"] and B["PEK5_ASOF"]["h2_asof_exact"])
                                    and not B["PEK5_ASOF"]["flag_raised"]),
        "A3_order_i_logits_inexact": not a3["i"]["a_logits_exact"],
        "A3_order_ii_logits_inexact": not a3["ii"]["a_logits_exact"],
        "A3_order_ii_request_prefix_reading_of_b": bool(a3["ii"]["served_cached_block_whose_request_prefix_contains_retracted"]
                                                        and a3["ii"]["refused_retracted_blocks"] == 0),
    }
    k2_a = {N: v["K2_B_FLAT"]["fires"] for N, v in A["configs"].items()}
    k2_b = {n: v["K2_B_GEO_CKPT"]["fires"] for n, v in B.items() if isinstance(v, dict) and "K2_B_GEO_CKPT" in v}
    k1_fires = k1["P51_DIGEST_bundle_path"] or k1["PEK5_ASOF_native_api"] or k1["A3_order_i_logits_inexact"] or k1["A3_order_ii_logits_inexact"]
    k2_fires = any(k2_a.values()) or any(k2_b.values())
    verdict = "KILLED" if (k1_fires or k2_fires) else "ADVANCE (scoped)"
    c_k2 = {k: [n for n, v in r["K2"].items() if v["fires"]] for k, r in C["configs"].items()}
    c_coll = C["collision_pairs"]
    result = {
        "item": "W22-09",
        "prereg_commit": "c72921e6",
        "quick": bool(args.quick),
        "config": cfg,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
                        "cpu_only": True},
        "verdict": verdict,
        "K1": {"components": k1, "fires": bool(k1_fires),
               "note": "A3 order-ii request-prefix reading of invariant (b) is reported but not counted; see report."},
        "K2": {"part_A_B_FLAT_by_N": k2_a, "part_B_B_GEO_CKPT_by_workload": k2_b, "fires": bool(k2_fires)},
        "part_C_secondary": {
            "K2_firing_pairs_by_config": c_k2,
            "K1_collision_nocert_breaks": any(("nocert_err_H" in r and max(r["nocert_err_H"], r["nocert_err_Hprime"]) > 1e-6) for r in c_coll),
            "K1_collision_cert_refuses_all": all(r.get("cert_refused", False) for r in c_coll) if c_coll else None,
        },
        "part_A": A,
        "part_B": B,
        "part_C": C,
        "crouzeix": CZ,
        "timings_s": {**timings, "total": time.perf_counter() - t0},
        "scratch": {"path": str(scratch.resolve()), "bytes": _dir_bytes(scratch)},
    }
    out.mkdir(parents=True, exist_ok=True)
    name = "w22_09_results_quick.json" if args.quick else "w22_09_results.json"
    (out / name).write_text(json.dumps(result, indent=1, sort_keys=False, default=float), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "K1": k1, "K2": result["K2"], "timings_s": result["timings_s"],
                      "scratch": result["scratch"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
