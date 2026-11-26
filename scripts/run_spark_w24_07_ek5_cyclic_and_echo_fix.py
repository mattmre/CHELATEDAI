#!/usr/bin/env python3
"""Wave 24-07 (Lane 2): EK5 Acronym/Header Echo-Recall Fix Diagnostic + Non-Trivial Cyclic/Deletion Benchmark vs SQLite.

Runs on DGX Spark CPU (`gx10-6724`):
  Part A: Diagnoses why `classify_high_similarity_pair` got 0.99% echo recall on #55 frozen wild
          snapshots (`artifacts/pv7_w22e_b55_wild_live/frozen_snapshots`) and evaluates a clean
          proper-noun/header normalization fix (stripping structured source prefixes like
          `OSV mirror:`, `Red Hat CSAF VEX:`, `SEC EDGAR 8-K:` and treating sentence-initial
          boilerplate tags symmetrically) — reporting BEFORE vs AFTER echo recall, precision,
          and conflict preservation on both #55 wild pairs AND synthetic minimal-pair conflict tests.
  Part B: Non-trivial cyclic + multi-hop copy + dynamic deletion + CITES_AUDIT benchmark comparing
          `EK5_v3` (`StratifiedEvidenceStore`) vs ordinary SQLite bitemporal ledger
          (`ChallengerBitemporalLedger`) across graph scales N in (50, 200, 500, 2000), verifying
          exact oracle agreement (`compute_alternating_wfs_oracle`) on mutual-REFUTE 2-cycles,
          positive DERIVES_CURRENT cycles, CITES_AUDIT legal-hold edges, and dynamic deletions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ek5_v3_stratified as EK5  # noqa: E402
from chelated_pv7_spine.challenger_ledger import (  # noqa: E402
    ChallengerBitemporalLedger,
    LedgerEdgeEnvelope,
    LedgerRecordEnvelope,
    SQLiteChallengerBackend,
)
from chelated_wild_revision_streams import load_adjudicated_families  # noqa: E402


# Structured wire/mirror header patterns observed on real OSV / Red Hat CSAF / SEC EDGAR streams
_STRUCTURED_WIRE_PREFIX_RE = re.compile(
    r"^(?:"
    r"(?:osv|ghsa|nvd|csaf|vex|red\s+hat|sec|edgar|mirror|wire|digest|syndicated|feed|alert|advisory|bulletin|record)"
    r"(?:\s+[a-z0-9._/-]+){0,4}\s*[:\-–—]\s*"
    r")+",
    re.IGNORECASE,
)

# Framing acronyms/labels that name the reporting channel rather than the underlying claim entity
_CHANNEL_FRAMING_TOKENS: Set[str] = {
    "osv", "ghsa", "nvd", "csaf", "vex", "sec", "edgar", "mirror", "wire", "digest",
    "syndicated", "feed", "alert", "advisory", "bulletin", "repost", "copy", "relay",
    "red", "hat", "github", "filing", "form", "item", "report", "notice", "update",
}


def strip_structured_wire_headers(text: str) -> str:
    cleaned = EK5.strip_boilerplate_headers(text)
    for _ in range(3):
        nxt = _STRUCTURED_WIRE_PREFIX_RE.sub("", cleaned, count=1).strip()
        if nxt == cleaned:
            break
        cleaned = nxt
    return cleaned


def classify_high_similarity_pair_fixed(text_a: str, text_b: str) -> Tuple[str, str]:
    """Candidate fix for EK5 classify_high_similarity_pair:
    1. Strips structured channel/mirror wire prefixes before token extraction.
    2. Filters channel-framing acronyms (OSV/SEC/GHSA/Mirror/Wire) from entity conflict checks
       while retaining all real package/company/CVE/numeric/polarity/antonym checks.
    3. Keeps strict conflict preservation on numeric/code/time, polarity, antonym, real entities,
       and minimal-pair substitutions.
    """
    body_a = strip_structured_wire_headers(text_a)
    body_b = strip_structured_wire_headers(text_b)
    sigs_a, neg_a, st_a, prop_a, res_a = EK5._canonical_value_tokens(body_a)
    sigs_b, neg_b, st_b, prop_b, res_b = EK5._canonical_value_tokens(body_b)

    prop_a = {p for p in prop_a if p not in _CHANNEL_FRAMING_TOKENS}
    prop_b = {p for p in prop_b if p not in _CHANNEL_FRAMING_TOKENS}
    res_a = {r for r in res_a if r not in _CHANNEL_FRAMING_TOKENS}
    res_b = {r for r in res_b if r not in _CHANNEL_FRAMING_TOKENS}

    if sigs_a != sigs_b and (sigs_a or sigs_b):
        return "PRESERVE_CONFLICT", "CONFLICT_NUMERIC_OR_CODE"
    if neg_a != neg_b:
        return "PRESERVE_CONFLICT", "CONFLICT_POLARITY"
    for term in st_a:
        if EK5._ANTONYM_MAP.get(term) in st_b:
            return "PRESERVE_CONFLICT", "CONFLICT_ANTONYM"
    if prop_a != prop_b and prop_a and prop_b:
        return "PRESERVE_CONFLICT", "CONFLICT_PROPER_ENTITY"
    diff_a = res_a - res_b
    diff_b = res_b - res_a
    if diff_a and diff_b and len(diff_a) <= 2 and len(diff_b) <= 2:
        return "PRESERVE_CONFLICT", "CONFLICT_MINIMAL_PAIR_SUBSTITUTION"
    return "ECHO_COLLAPSE", "COMPATIBLE_ECHO"


def run_part_a_echo_fix_audit(snapshot_dir: Path) -> Dict[str, Any]:
    families = load_adjudicated_families(snapshot_dir)
    tp_base = fp_base = fn_base = tn_base = 0
    tp_fix = fp_fix = fn_fix = tn_fix = 0
    fn_reasons_base: Counter[str] = Counter()
    fn_reasons_fix: Counter[str] = Counter()
    sample_fn_base: List[Dict[str, str]] = []
    sample_fn_fix: List[Dict[str, str]] = []

    for fam in families:
        asserts = [r for r in fam.records if r.op_kind in ("ASSERT", "CONFLICT") and r.text]
        by_id = {r.record_id: r for r in asserts}
        for i, a in enumerate(asserts):
            for b in asserts[i + 1 :]:
                if a.tenant_id != b.tenant_id:
                    continue
                is_true_echo = (
                    (b.echo_of == a.record_id)
                    or (a.echo_of == b.record_id)
                    or (bool(a.echo_of) and a.echo_of == b.echo_of)
                )
                act_b, rsn_b = EK5.classify_high_similarity_pair(a.text, b.text)
                act_f, rsn_f = classify_high_similarity_pair_fixed(a.text, b.text)
                pred_b = act_b == "ECHO_COLLAPSE"
                pred_f = act_f == "ECHO_COLLAPSE"

                if is_true_echo:
                    if pred_b:
                        tp_base += 1
                    else:
                        fn_base += 1
                        fn_reasons_base[rsn_b] += 1
                        if len(sample_fn_base) < 5:
                            sample_fn_base.append({"reason": rsn_b, "a": a.text[:140], "b": b.text[:140]})
                    if pred_f:
                        tp_fix += 1
                    else:
                        fn_fix += 1
                        fn_reasons_fix[rsn_f] += 1
                        if len(sample_fn_fix) < 5:
                            sample_fn_fix.append({"reason": rsn_f, "a": a.text[:140], "b": b.text[:140]})
                else:
                    if pred_b:
                        fp_base += 1
                    else:
                        tn_base += 1
                    if pred_f:
                        fp_fix += 1
                    else:
                        tn_fix += 1

    # Conflict-preservation guard checks (must ALL return PRESERVE_CONFLICT)
    conflict_guard_cases = [
        ("CVE-2026-1234 affects package foo 1.2.3", "CVE-2026-1234 affects package foo 1.2.4"),
        ("OSV mirror: package foo is affected", "Red Hat CSAF VEX: package foo is not affected"),
        ("SEC EDGAR: Acme Corp revenue is $120M", "SEC EDGAR: Beta Corp revenue is $120M"),
        ("Advisory: widget is vulnerable", "Advisory: widget is patched"),
        ("Notice: status approved for release", "Notice: status rejected for release"),
    ]
    guard_passed = sum(
        1 for a, b in conflict_guard_cases if classify_high_similarity_pair_fixed(a, b)[0] == "PRESERVE_CONFLICT"
    )

    def _metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, Any]:
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 5),
            "recall": round(rec, 5),
            "f1": round(f1, 5),
        }

    base_m = _metrics(tp_base, fp_base, fn_base, tn_base)
    fix_m = _metrics(tp_fix, fp_fix, fn_fix, tn_fix)
    print(
        f"[w24-07:A] #55 echo classifier | BASE recall={base_m['recall']:.4f} prec={base_m['precision']:.4f} "
        f"-> FIXED recall={fix_m['recall']:.4f} prec={fix_m['precision']:.4f} | conflict_guards={guard_passed}/{len(conflict_guard_cases)}",
        flush=True,
    )
    return {
        "n_families": len(families),
        "baseline": base_m,
        "fixed": fix_m,
        "baseline_fn_reasons": dict(fn_reasons_base),
        "fixed_fn_reasons": dict(fn_reasons_fix),
        "sample_fn_baseline": sample_fn_base,
        "sample_fn_fixed": sample_fn_fix,
        "conflict_guard_passed": guard_passed,
        "conflict_guard_total": len(conflict_guard_cases),
    }


def run_part_b_cyclic_stress() -> Dict[str, Any]:
    """Benchmark EK5_v3 vs SQLite challenger on non-trivial cyclic + multi-hop + CITES_AUDIT + dynamic deletion graphs.

    Specifically exercises the 4 structures that were ABSENT from #55:
      1. Mutual-REFUTE 2-cycles (equal-authority disputes -> WFS U / CONTESTED).
      2. Positive DERIVES_CURRENT cycles (unfounded self-supporting echo rings -> WFS F / AUDIT_ONLY).
      3. CITES_AUDIT legal-hold edges (must remain ADMIT even when cited target is withdrawn!).
      4. Multi-hop COPY_OF cascades with dynamic deletion & reinstatement.
    """
    scales = (50, 200, 500, 2000)
    results_by_scale: List[Dict[str, Any]] = []

    for n_nodes in scales:
        rng = random.Random(240700 + n_nodes)
        store = EK5.StratifiedEvidenceStore()
        sql = ChallengerBitemporalLedger(SQLiteChallengerBackend(":memory:"))

        # Build modular blocks of 10 nodes each:
        #   0,1: mutual REFUTE 2-cycle (equal auth -> CONTESTED in WFS)
        #   2,3,4: positive DERIVES_CURRENT 3-cycle (unfounded -> F in WFS; naive SQL recursive CTE without WFS fails!)
        #   5: root claim -> 6,7: COPY_OF cascade -> 8: REFUTE withdrawing 5 (must cascade-withdraw 6,7)
        #   9: CITES_AUDIT referencing 5 (legal hold: MUST STAY ADMIT even after 5 is withdrawn!)
        n_blocks = n_nodes // 10
        events: List[EK5.StratifiedEvent] = []
        edges: List[EK5.EdgeAttestation] = []
        sql_ops: List[Tuple[str, Any]] = []
        t = 1
        eid = 0

        for blk in range(n_blocks):
            ent = f"ent_{blk}"
            slot = "status"
            ids = [f"b{blk}_n{k}" for k in range(10)]
            for k, nid in enumerate(ids):
                val = f"val_{blk}_{k}"
                events.append(
                    EK5.StratifiedEvent(
                        event_id=nid, entity=ent, slot=slot, claim_value=val,
                        text=f"Claim {nid} states {val}", valid_time=t, tx_time=t,
                        tenant="t0", domain="general", authority_tier=5,
                    )
                )
                sql_ops.append((
                    "assert",
                    LedgerRecordEnvelope(
                        tenant_id="t0", record_id=nid, entity_key=f"{ent}:{k//2}",
                        authority_rank=5, valid_from=t, valid_to=None, tx_time=t,
                        op_kind="ASSERT", claim_value=val,
                    ),
                ))
                t += 1

            # 1) Mutual REFUTE 2-cycle: n0 <-> n1
            edges.append(EK5.EdgeAttestation(f"e{eid}", ids[0], ids[1], "REFUTE", valid_time=t, tx_time=t)); eid += 1; t += 1
            edges.append(EK5.EdgeAttestation(f"e{eid}", ids[1], ids[0], "REFUTE", valid_time=t, tx_time=t)); eid += 1; t += 1
            # 2) Positive DERIVES_CURRENT 3-cycle: n2 -> n3 -> n4 -> n2 (unfounded!)
            for u, v in ((ids[2], ids[3]), (ids[3], ids[4]), (ids[4], ids[2])):
                edges.append(EK5.EdgeAttestation(f"e{eid}", u, v, "DERIVES_CURRENT", valid_time=t, tx_time=t)); eid += 1; t += 1
                sql_ops.append(("edge", LedgerEdgeEnvelope("t0", f"se{eid}", u, v, t)))
            # 3) Multi-hop COPY_OF cascade: n6 -> n5, n7 -> n6, plus n9 CITES_AUDIT n5
            edges.append(EK5.EdgeAttestation(f"e{eid}", ids[6], ids[5], "COPY_OF", valid_time=t, tx_time=t)); eid += 1; t += 1
            sql_ops.append(("edge", LedgerEdgeEnvelope("t0", f"se{eid}", ids[6], ids[5], t)))
            edges.append(EK5.EdgeAttestation(f"e{eid}", ids[7], ids[6], "COPY_OF", valid_time=t, tx_time=t)); eid += 1; t += 1
            sql_ops.append(("edge", LedgerEdgeEnvelope("t0", f"se{eid}", ids[7], ids[6], t)))
            edges.append(EK5.EdgeAttestation(f"e{eid}", ids[9], ids[5], "CITES_AUDIT", valid_time=t, tx_time=t)); eid += 1; t += 1
            # Note: SQL ledger has no CITES_AUDIT distinction (either omits edge or treats as derived dependency)
            # 4) Withdrawal of root n5 by n8
            edges.append(EK5.EdgeAttestation(f"e{eid}", ids[8], ids[5], "REFUTE", valid_time=t, tx_time=t)); eid += 1; t += 1
            sql_ops.append((
                "withdraw",
                LedgerRecordEnvelope(
                    tenant_id="t0",
                    record_id=ids[5],
                    entity_key=f"{ent}:2",
                    authority_rank=5,
                    valid_from=t,
                    valid_to=None,
                    tx_time=t,
                    op_kind="WITHDRAW",
                    claim_value="",
                ),
            ))

        # Benchmark EK5 ingest + incremental evaluation
        t_ek0 = time.perf_counter()
        store.apply_mutation(events, edges)
        ek5_ms = (time.perf_counter() - t_ek0) * 1000.0

        # Independent WFS oracle verification on entire graph
        oracle = EK5.compute_alternating_wfs_oracle(store.events.keys(), store.refs, store.deps_v)
        ek5_oracle_mismatches = sum(1 for nid, v in oracle.items() if store.val.get(nid) != v)

        # Check the 4 exact structural invariants on EK5
        ek5_mutual_refute_ok = sum(
            1 for b in range(n_blocks)
            if store.val[f"b{b}_n0"] == "U" and store.val[f"b{b}_n1"] == "U"
        )
        ek5_pos_cycle_ok = sum(
            1 for b in range(n_blocks)
            if all(store.val[f"b{b}_n{k}"] == "F" for k in (2, 3, 4))
        )
        ek5_copy_cascade_ok = sum(
            1 for b in range(n_blocks)
            if all(store.val[f"b{b}_n{k}"] == "F" for k in (5, 6, 7))
        )
        ek5_cites_audit_ok = sum(
            1 for b in range(n_blocks)
            if store.eligibility[f"b{b}_n9"] == "ADMIT"
        )

        # Benchmark SQLite challenger on the same ops
        t_sq0 = time.perf_counter()
        for kind, env in sql_ops:
            if kind == "assert":
                sql.append(env)
            elif kind == "edge":
                sql.add_edge(env)
            elif kind == "withdraw":
                sql.append(env)
        sql_ingest_ms = (time.perf_counter() - t_sq0) * 1000.0

        # Query positive-cycle entities on SQLite (n2,n3,n4) -> ordinary SQLite does NOT detect unfounded positive cycles!
        t_sqq0 = time.perf_counter()
        sql_pos_cycle_detected = 0
        sql_copy_cascade_ok = 0
        for b in range(n_blocks):
            res_cyc = sql.query_as_of("t0", f"ent_{b}:1", valid_time=t + 10, tx_time=t + 10, mode="recursive_cte")
            if res_cyc.status == "RETRACTED":
                sql_pos_cycle_detected += 1
            res_cop = sql.query_as_of("t0", f"ent_{b}:3", valid_time=t + 10, tx_time=t + 10, mode="recursive_cte")
            if res_cop.status == "RETRACTED":
                sql_copy_cascade_ok += 1
        sql_query_ms = (time.perf_counter() - t_sqq0) * 1000.0

        row = {
            "n_nodes": n_nodes,
            "n_blocks": n_blocks,
            "ek5_apply_ms": round(ek5_ms, 2),
            "ek5_oracle_mismatches": ek5_oracle_mismatches,
            "ek5_mutual_refute_2cycle_ok": f"{ek5_mutual_refute_ok}/{n_blocks}",
            "ek5_unfounded_pos_cycle_ok": f"{ek5_pos_cycle_ok}/{n_blocks}",
            "ek5_copy_cascade_ok": f"{ek5_copy_cascade_ok}/{n_blocks}",
            "ek5_cites_audit_preserved": f"{ek5_cites_audit_ok}/{n_blocks}",
            "sqlite_ingest_ms": round(sql_ingest_ms, 2),
            "sqlite_query_ms": round(sql_query_ms, 2),
            "sqlite_unfounded_pos_cycle_detected": f"{sql_pos_cycle_detected}/{n_blocks}",
            "sqlite_copy_cascade_ok": f"{sql_copy_cascade_ok}/{n_blocks}",
        }
        results_by_scale.append(row)
        print(
            f"[w24-07:B] N={n_nodes:4d} | EK5={ek5_ms:6.1f}ms (oracle_mism={ek5_oracle_mismatches}, "
            f"pos_cyc={ek5_pos_cycle_ok}/{n_blocks}, cites_ok={ek5_cites_audit_ok}/{n_blocks}) | "
            f"SQLite={sql_ingest_ms+sql_query_ms:6.1f}ms (pos_cyc={sql_pos_cycle_detected}/{n_blocks}, copy={sql_copy_cascade_ok}/{n_blocks})",
            flush=True,
        )

    return {"scales": results_by_scale}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot-dir", type=Path, default=REPO_ROOT / "artifacts" / "pv7_w22e_b55_wild_live" / "frozen_snapshots")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts" / "pv7_w24_07_ek5_cyclic_and_echo_fix")
    args = ap.parse_args()

    t0 = time.time()
    part_a = run_part_a_echo_fix_audit(args.snapshot_dir)
    part_b = run_part_b_cyclic_stress()

    payload = {
        "schema_version": "w24_07_ek5_cyclic_and_echo_fix_v1",
        "wave_item_id": "W24-07",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - t0, 2),
        "part_a_echo_classifier_fix": part_a,
        "part_b_cyclic_and_deletion_benchmark": part_b,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "w24_07_results.json"
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    out_path.write_bytes(raw)
    print(f"[w24-07] DONE sha256={hashlib.sha256(raw).hexdigest()[:12]} wall={payload['runtime_seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
