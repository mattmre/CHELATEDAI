"""W22-05 runner: same-input ordinary ledger comparator + real-graph negation-cycle census.

Writes ``artifacts/pv7_w23a_w22_05/w22_05_results.json``.  Bulk data (the OSV cache, and
per-row dumps) goes under ``artifacts/pv7_w23a_w22_05/scratch/``, which is never committed.
The kill/advance rule is the one frozen in ``PREREGISTRATION.md`` (commit cfd40d6d).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import run_pv7_w18_6c_wild_revisions as b55  # noqa: E402
from chelated_pv7_spine.challenger_ledger import SQLiteChallengerBackend  # noqa: E402
from chelated_wild_revision_streams import load_adjudicated_families, select_cohort  # noqa: E402
from prospective_realworld_drift_corpus import build_prospective_realworld_drift_suite  # noqa: E402
from virgin_conversational_holdout import _SCENARIOS, build_virgin_conversational_holdout, virgin_suite_digest  # noqa: E402

import w22_05_ledger_comparator as cmp  # noqa: E402

OUT_DIR = REPO_ROOT / "artifacts" / "pv7_w23a_w22_05"
SCRATCH = OUT_DIR / "scratch"
ARENA_SHA = "cb6bc0082b5850f4edfea668d1ecb2a73fe6f898b7476e34ed11036cb378765b"
CONV_DIGEST = "d395dd94d72291355382a4d1f389df234bbb3f9b32884e0aada78592b14a95e2"
B55_EXPECTED = 0.4375
B55_SNAPSHOTS = REPO_ROOT / "artifacts" / "pv7_w22e_b55_wild_live" / "frozen_snapshots"
CONV_AUDIT = REPO_ROOT / "artifacts" / "wave6_pm1b_v4_reranked_sota" / "wave6_virgin_blind_judge_audit.json"
EK5_ON_ARM = "Chelated_v4_Reranked_EK5v3"
EK5_OFF_ARM = "DualGran_Reranked_NoEK5"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_arena():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    cache = SCRATCH / "osv_cve_seeds.json"
    cache_existed = cache.is_file()
    cases, docs, sha = build_prospective_realworld_drift_suite(cache)
    by_case = {}
    for d in docs:
        cid = next(c["case_id"] for c in cases if d.doc_id.startswith(c["case_id"] + "_"))
        by_case.setdefault(cid, []).append(d)
    rows, census_graphs, items, div_ops = [], [], [], []
    pos_cyc_nodes = 0
    for case in cases:
        row = cmp.run_arena_case(case, by_case[case["case_id"]])
        exo = row.pop("_exo")
        nodes = list(exo.events.keys())
        edges = [(a.src_id, a.dst_id, a.edge_type) for a in exo.attestations]
        census = cmp.negation_cycle_census(nodes, edges)
        census["pos_cyc_nodes"] = sum(1 for v in exo.store.pos_cyc.values() if v)
        pos_cyc_nodes += census["pos_cyc_nodes"]
        census_graphs.append({"id": case["case_id"], "is_conflict": row["is_conflict"], "census": census})
        row["census"] = census
        rows.append(row)
        items.append(
            {
                "id": case["case_id"],
                "group": case["regime"],
                "is_conflict": row["is_conflict"],
                "ek5_verdict": tuple(row["ek5_verdict"]),
                "ledger_verdict": tuple(row["ledger_verdict"]),
                "ek5_correct": row["ek5_correct"],
                "ledger_correct": row["ledger_correct"],
                "ek5_covered": bool(row["ek5_verdict"]),
                "ledger_covered": bool(row["ledger_verdict"]),
            }
        )
        ops, keys = cmp.arena_ledger_ops(exo)
        ekey = next(iter(keys.values()))
        txs = sorted({env.tx_time for _k, env in ops})
        div_ops.append((ops, [(docs[0].tenant, ekey, cmp.LEDGER_QUERY_VALID_TIME, t) for t in txs]))
    # Ledger incremental vs fresh, per case (each case is an independent ledger).
    agg = {"queries": 0, "ops": 0, "inc_vs_fresh_ordered": 0, "inc_vs_fresh_shuffled": 0}
    for k, (ops, queries) in enumerate(div_ops):
        d = cmp.ledger_rebuild_divergence(ops, queries, seed=20261126 + k)
        agg["queries"] += d["queries"]
        agg["ops"] += d["ops"]
        agg["inc_vs_fresh_ordered"] += d["inc_vs_fresh_ordered"]["digest_mismatches"]
        agg["inc_vs_fresh_shuffled"] += d["inc_vs_fresh_shuffled"]["digest_mismatches"]
    ablation = {
        "ledger_authority_only_correct": sum(1 for r in rows if r["ledger_authority_only_correct"]),
        "by_regime": {},
    }
    by_regime = {}
    for r in rows:
        g = by_regime.setdefault(r["regime"], {"n": 0, "ek5": 0, "ledger": 0, "ledger_authority_only": 0, "parity": 0})
        g["n"] += 1
        g["ek5"] += int(r["ek5_correct"])
        g["ledger"] += int(r["ledger_correct"])
        g["ledger_authority_only"] += int(r["ledger_authority_only_correct"])
        g["parity"] += int(r["parity"])
    ablation["by_regime"] = by_regime
    return {
        "corpus_sha256": sha,
        "expected_sha256": ARENA_SHA,
        "gate_pass": sha == ARENA_SHA,
        "osv_cache": {"path": str(cache), "existed_before_run": cache_existed, "sha256": _sha_file(cache)},
        "n_cases": len(cases),
        "n_docs": len(docs),
        "by_regime": by_regime,
        "ablation": ablation,
        "census": cmp.census_summary(census_graphs),
        "pos_cyc_nodes_total": pos_cyc_nodes,
        "ledger_rebuild_divergence": agg,
    }, rows, items


def _b55_ops(family):
    return [("edge" if k == "edge" else "record", env) for k, env in b55.sql_envelopes(family)]


def run_b55():
    families = tuple(select_cohort(load_adjudicated_families(B55_SNAPSHOTS))["families"])
    store, _w = b55.build_ek5_store(families)
    ek5_rows = b55.score_ek5(store, families)
    ledger, _w2 = b55.build_sql_ledger(families, SQLiteChallengerBackend())
    sql_rows = {mode: b55.score_sql(ledger, families, mode) for mode in cmp.LEDGER_MODES}
    ledger.close()
    acc = {"ek5_v3": b55._accuracy(ek5_rows)}
    for mode, rs in sql_rows.items():
        acc[f"sqlite_{mode}"] = b55._accuracy(rs)
    mode_disagreements = sum(
        1
        for a, b in zip(sql_rows["recursive_cte"], sql_rows["closure_table"])
        if (a["predicted_disposition"], a["predicted_lineage"]) != (b["predicted_disposition"], b["predicted_lineage"])
    )
    gate = all(abs(v - B55_EXPECTED) < 1e-12 for v in (acc["ek5_v3"], acc["sqlite_closure_table"]))
    sql_by = {(r["family_id"], r["cut"]): r for r in sql_rows["closure_table"]}
    items, disp_parity = [], 0
    for e in ek5_rows:
        s = sql_by[(e["family_id"], e["cut"])]
        if (e["oracle_disposition"], e["oracle_lineage"]) != (s["oracle_disposition"], s["oracle_lineage"]):
            raise RuntimeError("oracle mismatch between arms")
        disp_parity += int(e["predicted_disposition"] == s["predicted_disposition"])
        items.append(
            {
                "id": f"{e['family_id']}@{e['cut']}",
                "group": e["stratum"],
                "cut": e["cut"],
                "is_conflict": e["oracle_disposition"] == "disputed",
                "ek5_verdict": (e["predicted_disposition"], tuple(e["predicted_lineage"])),
                "ledger_verdict": (s["predicted_disposition"], tuple(s["predicted_lineage"])),
                "ek5_correct": bool(e["hit"]),
                "ledger_correct": bool(s["hit"]),
                "ek5_covered": e["predicted_disposition"] != "retracted",
                "ledger_covered": s["predicted_disposition"] != "retracted",
            }
        )
    # Census per family on the EK5 graph actually projected by the bundle.
    graphs = []
    oracle_disp_by_family = {}
    for it in items:
        oracle_disp_by_family.setdefault(it["id"].split("@")[0], []).append(it["is_conflict"])
    for fam in families:
        events, copies, withdraws = b55.project_ek5(fam)
        census = cmp.negation_cycle_census(
            [ev.event_id for ev in events], [(a.src_id, a.dst_id, a.edge_type) for a in list(copies) + list(withdraws)]
        )
        graphs.append({"id": fam.family_id, "is_conflict": any(oracle_disp_by_family[fam.family_id]), "census": census})
    # Ledger incremental vs fresh over every queried cut.
    ops, queries = [], []
    for fam in families:
        ops.extend(_b55_ops(fam))
        queries.extend((fam.tenant_id, fam.entity_key, c.valid_time, c.tx_time) for c in fam.cuts)
    div = cmp.ledger_rebuild_divergence(ops, queries)
    strata = {}
    for it in items:
        g = strata.setdefault(it["group"], {"n": 0, "ek5": 0, "ledger": 0, "set_parity": 0, "conflict": 0})
        g["n"] += 1
        g["ek5"] += int(it["ek5_correct"])
        g["ledger"] += int(it["ledger_correct"])
        g["set_parity"] += int(it["ek5_verdict"] == it["ledger_verdict"])
        g["conflict"] += int(it["is_conflict"])
    return {
        "n_families": len(families),
        "n_cuts": len(items),
        "snapshot_sha256": b55.snapshot_directory_sha256(B55_SNAPSHOTS),
        "accuracy": acc,
        "expected_accuracy": B55_EXPECTED,
        "gate_pass": gate,
        "sqlite_mode_disagreements": mode_disagreements,
        "disposition_only_parity": disp_parity / len(items),
        "by_stratum": strata,
        "census": cmp.census_summary(graphs),
        "pos_cyc_nodes_global_store": sum(1 for v in store.pos_cyc.values() if v),
        "ledger_rebuild_divergence": div,
    }, items


def run_conv():
    convs = build_virgin_conversational_holdout()
    digest = virgin_suite_digest(convs)
    audit = json.loads(CONV_AUDIT.read_text(encoding="utf-8"))["audit_log"]
    on = cmp.conv_audit_per_item(audit, EK5_ON_ARM)
    off = cmp.conv_audit_per_item(audit, EK5_OFF_ARM)
    rows = cmp.conv_ku_ledger_rows(convs, _SCENARIOS)
    for r in rows:
        r["ek5_on_correct"] = on.get(r["qid"])
        r["ek5_off_correct"] = off.get(r["qid"])
    n = len(rows)

    def _cnt(k):
        return sum(1 for r in rows if r[k] is True)

    return {
        "digest": digest,
        "expected_digest": CONV_DIGEST,
        "gate_pass": digest == CONV_DIGEST,
        "audit_sha256": _sha_file(CONV_AUDIT),
        "n_ku_items": n,
        "n_items_total": sum(len(c.questions) for c in convs),
        "ledger_correct": _cnt("ledger_correct"),
        "ledger_authority_only_correct": _cnt("ledger_authority_only_correct"),
        "ek5_on_correct": _cnt("ek5_on_correct"),
        "ek5_off_correct": _cnt("ek5_off_correct"),
        "ek5_on_items_all_audit": sum(1 for v in on.values() if v),
        "ek5_off_items_all_audit": sum(1 for v in off.values() if v),
        "audit_items_per_arm": len(on),
        "set_level_parity": None,
        "note": "descriptive only; EK5 admitted sets were never persisted, so no matched comparison is possible",
    }, rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--conformance-sequences", type=int, default=1000)
    ap.add_argument("--conformance-steps", type=int, nargs="*", default=[6, 12])
    args = ap.parse_args(argv)
    t0 = time.time()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    arena, arena_rows, arena_items = run_arena()
    b55_block, b55_items = run_b55()
    conv, conv_rows = run_conv()
    gates = {"arena60": arena["gate_pass"], "b55": b55_block["gate_pass"]}
    blocks = {
        "arena60": cmp.paired_metrics(arena_items),
        "b55": cmp.paired_metrics(b55_items),
    }
    pooled = [it for k, its in (("arena60", arena_items), ("b55", b55_items)) if gates[k] for it in its]
    blocks["pooled"] = cmp.paired_metrics(pooled)
    decision = cmp.kill_rule_decision(blocks, gates)
    wins = {
        k: {
            "ek5_wins": [it["id"] for it in its if it["ek5_correct"] and not it["ledger_correct"]],
            "ledger_wins": [it["id"] for it in its if it["ledger_correct"] and not it["ek5_correct"]],
            "set_mismatches": [
                {"id": it["id"], "ek5": list(it["ek5_verdict"]), "ledger": list(it["ledger_verdict"])}
                for it in its
                if it["ek5_verdict"] != it["ledger_verdict"]
            ][:200],
        }
        for k, its in (("arena60", arena_items), ("b55", b55_items))
    }
    for k in wins:
        wins[k]["ek5_wins_by_group"] = {}
        wins[k]["ledger_wins_by_group"] = {}
    for k, its in (("arena60", arena_items), ("b55", b55_items)):
        for it in its:
            if it["ek5_correct"] and not it["ledger_correct"]:
                wins[k]["ek5_wins_by_group"][it["group"]] = wins[k]["ek5_wins_by_group"].get(it["group"], 0) + 1
            if it["ledger_correct"] and not it["ek5_correct"]:
                wins[k]["ledger_wins_by_group"][it["group"]] = wins[k]["ledger_wins_by_group"].get(it["group"], 0) + 1
    conformance = {}
    for steps in args.conformance_steps:
        t = time.time()
        conformance[str(steps)] = cmp.ek5_conformance_explanation_diffs(steps, args.conformance_sequences)
        conformance[str(steps)]["seconds"] = round(time.time() - t, 2)
    results = {
        "item": "W22-05",
        "preregistration_commit": "cfd40d6d",
        "decision": decision,
        "gates": gates,
        "metrics": blocks,
        "wins": wins,
        "arena60": arena,
        "b55": b55_block,
        "conv120": conv,
        "ek5_conformance_inc_vs_fresh_python": conformance,
        "postgres": "not used: PostgreSQL 18.6 present on host, PostgreSQL 17 absent; SQLite only",
        "unavailable_inputs": [
            "artifacts/.../wave6_pm3_v2_arena_rows.csv (arena per-case rows) - absent",
            "arena LLM claim extractions - absent; role-derived oracle extraction used for both arms",
            "Conv 120 EK5 per-item admitted sets - never persisted; graph census not possible",
            "formal 12-step conformance report (962) - absent; recomputed in Python here",
            "Lean ek5_ref_cli - not built; Lean-side comparisons not run",
        ],
        "runtime_seconds": round(time.time() - t0, 2),
    }

    def _ser(o):
        if isinstance(o, (set, frozenset, tuple)):
            return list(o)
        raise TypeError(type(o))

    (OUT_DIR / "w22_05_results.json").write_text(json.dumps(results, indent=2, sort_keys=True, default=_ser), encoding="utf-8")
    (SCRATCH / "arena_rows.json").write_text(json.dumps(arena_rows, indent=1, default=_ser), encoding="utf-8")
    (SCRATCH / "b55_items.json").write_text(json.dumps(b55_items, indent=1, default=_ser), encoding="utf-8")
    (SCRATCH / "conv_ku_rows.json").write_text(json.dumps(conv_rows, indent=1, default=_ser), encoding="utf-8")
    print(json.dumps({"decision": decision["verdict"], "gates": gates, "runtime_seconds": results["runtime_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
