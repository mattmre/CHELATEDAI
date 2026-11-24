#!/usr/bin/env python3
"""EK5 v3 conformance runner + external-kernel (nanoda) tooling for the Wave-14A formal package.

Two jobs live here because ``scripts/verify_all_offline.py --full-formal`` invokes both from ``formal/``:

1. **Differential conformance** (default mode).  ``ek5_ref_cli`` (the Lean reference evaluator whose
   Stratum-2 core is kernel-bound to the well-founded semantics by ``evalStratum2Wfs_sound``) is compared with
   ``ek5_v3_stratified.StratifiedEvidenceStore`` over ``--sequences`` seeded multi-mutation sequences.  At every
   checkpoint four independent answers are produced for the *same* materialized graph:

   * ``inc``    - the live incremental store after the mutation batch (dirty-region maintenance),
   * ``fresh``  - a store rebuilt from scratch from the visible events/edges in shuffled order,
   * ``lean``   - the Lean CLI on the materialized snapshot (whole-snapshot functional evaluation),
   * ``oracle`` - ``compute_alternating_wfs_oracle`` (textbook Van Gelder alternating fixpoint) for ``val``.

   ``as_of`` checkpoints (bitemporal rewinds, including *missing mandatory parent* snapshots) are included.
   The runner also proves its own comparator is live by injecting deterministic faults and requiring 100%
   detection (a differential that cannot fail is not evidence).  It writes ``formal/conformance_report.json``
   and exits 0 only if every hard gate holds.

2. **External kernel re-check** (``--prepare-nanoda`` / ``--run-nanoda``).  ``lean4export`` serialises the Theorem-3
   declarations; ``nanoda_bin`` (an independent Rust kernel) re-checks them with ``unpermitted_axiom_hard_error``.
   The export is ~10 MB and is *generated*, not committed: ``python formal/run_conformance.py --prepare-nanoda``
   must run before ``verify_all_offline.py --full-formal`` reaches its nanoda step.

Python 3.9 compatible; never writes CRLF; never shells out through a redirect (UTF-16 trap on PowerShell).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

FORMAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = FORMAL_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ek5_v3_stratified as ek5  # noqa: E402  (needs REPO_ROOT on sys.path first)

SCHEMA_VERSION = 1
DEFAULT_SEED = 20261124
DEFAULT_SEQUENCES = 1000
DEFAULT_STEPS = 6
MIN_SEQUENCES = 1000
MIN_CHECKPOINTS = 5000
MIN_ASOF_MISSING_PARENT = 50
MAX_NODES = 11
PHANTOM = "__missing_mandatory_parent__"

MANIFEST_PATH = FORMAL_DIR / "nanoda_manifest.json"
REPORT_PATH = FORMAL_DIR / "conformance_report.json"

NODE_FIELDS: Tuple[str, ...] = ("eff_auth", "val", "cur", "eligibility", "reason", "pos_cyc")
SEMANTIC_FIELDS: Tuple[str, ...] = ("eff_auth", "val", "cur", "eligibility")
DOMAINS: Tuple[str, ...] = ("general", "general", "dom_a", "dom_b")
EDGE_WEIGHTS: Tuple[Tuple[str, int], ...] = (
    ("REFUTE", 30),
    ("SUPERSEDE", 14),
    ("VERSION_LINEAGE_SUPERSEDE", 9),
    ("COPY_OF", 5),
    ("DERIVES_CURRENT", 5),
    ("DERIVES_FROM", 6),
    ("AMENDS", 5),
    ("DERIVES_SNAPSHOT", 4),
    ("CITES_AUDIT", 4),
    ("SUPPORT", 4),
)
_EDGE_TYPES: Tuple[str, ...] = tuple(t for t, _ in EDGE_WEIGHTS)
_EDGE_CUM: Tuple[int, ...] = tuple(sum(w for _, w in EDGE_WEIGHTS[: i + 1]) for i in range(len(EDGE_WEIGHTS)))


# ---------------------------------------------------------------------------------------------------------------
# Generation model (tracks which attestations are active so generated batches are always legal)
# ---------------------------------------------------------------------------------------------------------------
class _Model:
    """Mirror of the attestation log, used only to keep generated mutations legal (no hidden state shared with the store)."""

    def __init__(self, strict: bool) -> None:
        self.strict = strict
        self.nodes: List[str] = []
        self.entity: Dict[str, str] = {}
        self.node_seq = 0
        self.att_seq = 0
        self.order: List[str] = []  # insertion order of every attestation id ever used (Python dict order)
        self.att: Dict[str, Tuple[str, str, str, Optional[str]]] = {}  # id -> (src, dst, type, lineage)
        self.active: Dict[str, bool] = {}
        self.dup_triples_seen = 0

    def active_ids(self) -> List[str]:
        return [a for a in self.order if self.active.get(a)]

    def mult(self, triple: Tuple[str, str, str]) -> int:
        return sum(1 for a in self.order if self.active.get(a) and self.att[a][:3] == triple)

    def edges(self, etype: str) -> Dict[str, Set[str]]:
        out: Dict[str, Set[str]] = {}
        for a in self.order:
            if self.active.get(a) and self.att[a][2] == etype:
                out.setdefault(self.att[a][0], set()).add(self.att[a][1])
        return out

    def would_close(self, etype: str, src: str, dst: str) -> bool:
        out = self.edges(etype)
        seen = {dst}
        stack = [dst]
        while stack:
            u = stack.pop()
            for w in out.get(u, ()):
                if w == src:
                    return True
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return False

    def add(self, att: "ek5.EdgeAttestation") -> None:
        if att.attestation_id not in self.att:
            self.order.append(att.attestation_id)
        self.att[att.attestation_id] = (att.src_id, att.dst_id, att.edge_type, att.lineage_id)
        if self.mult((att.src_id, att.dst_id, att.edge_type)) >= 1:
            self.dup_triples_seen += 1
        self.active[att.attestation_id] = True

    def deactivate(self, aid: str) -> None:
        self.active[aid] = False

    def remove_first(self, triple: Tuple[str, str, str]) -> None:
        for a in self.order:
            if self.active.get(a) and self.att[a][:3] == triple:
                self.active[a] = False
                return


def _pick_type(rng: random.Random) -> str:
    x = rng.randint(1, _EDGE_CUM[-1])
    for etype, cum in zip(_EDGE_TYPES, _EDGE_CUM):
        if x <= cum:
            return etype
    return _EDGE_TYPES[-1]


def _gen_batch(rng: random.Random, model: _Model, step: int) -> Dict[str, Any]:
    new_events: List[Any] = []
    n_new = rng.randint(3, 5) if step == 0 else rng.randint(0, 2)
    for _ in range(n_new):
        if len(model.nodes) >= MAX_NODES:
            break
        nid = "n%02d" % model.node_seq
        model.node_seq += 1
        ent = rng.choice(("e0", "e1"))
        vt = rng.randint(0, 4)
        ve = None if rng.random() < 0.7 else vt + rng.randint(1, 4)
        new_events.append(
            ek5.StratifiedEvent(
                event_id=nid,
                entity=ent,
                slot="s0",
                claim_value="v%d" % rng.randint(0, 2),
                valid_time=vt,
                tx_time=rng.randint(0, 4),
                valid_end=ve,
                tenant="t0",
                domain=rng.choice(DOMAINS),
                authority_tier=rng.randint(0, 4),
            )
        )
        model.nodes.append(nid)
        model.entity[nid] = ent

    withdraw: List[str] = []
    active_ids = model.active_ids()
    if active_ids and rng.random() < 0.35:
        for aid in rng.sample(active_ids, k=min(len(active_ids), rng.randint(1, 2))):
            withdraw.append(aid)
            model.deactivate(aid)
    remove: List[Tuple[str, str, str]] = []
    live = model.active_ids()
    if live and rng.random() < 0.15:
        triple = model.att[rng.choice(live)][:3]
        remove.append(triple)
        model.remove_first(triple)

    atts: List[Any] = []
    n_add = rng.randint(2, 6) if step == 0 else rng.randint(0, 4)
    if not model.strict and step in (1, 2) and len(model.nodes) >= 2 and rng.random() < 0.5:
        # imported-cycle mode: deliberately plant a SUPERSEDE 2-cycle so quarantine logic is exercised
        a_node, b_node = rng.sample(model.nodes, 2)
        for s_node, d_node in ((a_node, b_node), (b_node, a_node)):
            model.att_seq += 1
            cyc = ek5.EdgeAttestation(
                attestation_id="a%04d" % model.att_seq,
                src_id=s_node,
                dst_id=d_node,
                edge_type="SUPERSEDE",
                valid_time=rng.randint(0, 2),
                tx_time=rng.randint(0, 2),
            )
            model.add(cyc)
            atts.append(cyc)
    for _ in range(n_add):
        if len(model.nodes) < 2:
            break
        etype = _pick_type(rng)
        src = rng.choice(model.nodes)
        dst = rng.choice(model.nodes)
        lineage: Optional[str] = None
        if etype in ("SUPERSEDE", "VERSION_LINEAGE_SUPERSEDE"):
            if etype == "SUPERSEDE" and not model.strict and rng.random() < 0.4:
                existing = sorted((s, d) for s, ds in model.edges("SUPERSEDE").items() for d in ds)
                if existing:
                    src, dst = rng.choice(existing)
                    src, dst = dst, src  # deliberately close a cycle (imported-cycle mode only)
            if etype == "VERSION_LINEAGE_SUPERSEDE":
                peers = [n for n in model.nodes if n != src and model.entity[n] == model.entity[src]]
                if not peers:
                    continue
                dst = rng.choice(peers)
                lineage = "L%d" % rng.randint(0, 2)
            if src == dst:
                continue
            if model.strict and model.would_close(etype, src, dst):
                continue
        elif src == dst and rng.random() < 0.8:
            continue
        reusable = [a for a in model.order if not model.active.get(a)]
        if reusable and rng.random() < 0.1:
            aid = rng.choice(reusable)
            old = model.att[aid]
            src, dst, etype, lineage = old  # re-activating a withdrawn id must keep its identity
            if model.strict and etype in ("SUPERSEDE", "VERSION_LINEAGE_SUPERSEDE") and model.would_close(etype, src, dst):
                continue
        else:
            model.att_seq += 1
            aid = "a%04d" % model.att_seq
        att = ek5.EdgeAttestation(
            attestation_id=aid,
            src_id=src,
            dst_id=dst,
            edge_type=etype,
            valid_time=rng.randint(0, 4),
            tx_time=rng.randint(0, 4),
            lineage_id=lineage,
        )
        model.add(att)
        atts.append(att)
    return {"events": new_events, "atts": atts, "withdraw": withdraw, "remove": remove}


# ---------------------------------------------------------------------------------------------------------------
# Materialization, state extraction, oracle
# ---------------------------------------------------------------------------------------------------------------
def active_triples(store: "ek5.StratifiedEvidenceStore") -> List[Tuple[str, str, str]]:
    return sorted(k for k, v in store._edge_mult.items() if v > 0)


def materialize(store: "ek5.StratifiedEvidenceStore", snapshot_id: str) -> Dict[str, Any]:
    nodes = [
        {"id": nid, "tier": ev.authority_tier, "domain": ev.domain} for nid, ev in sorted(store.events.items())
    ]
    return {
        "snapshot_id": snapshot_id,
        "strict_dag_supersede": bool(store.strict_dag_supersede),
        "nodes": nodes,
        "missing": sorted(store._missing_dep_nodes),
        "edges": [{"src": s, "dst": d, "type": t} for s, d, t in active_triples(store)],
    }


def store_state(store: "ek5.StratifiedEvidenceStore") -> Dict[str, Any]:
    nodes = {
        nid: {
            "eff_auth": int(store.eff_auth[nid]),
            "val": store.val[nid],
            "cur": store.cur[nid],
            "eligibility": store.eligibility[nid],
            "reason": store.reason[nid],
            "pos_cyc": bool(store.pos_cyc[nid]),
        }
        for nid in store.events
    }
    tele = store.get_telemetry()
    return {
        "nodes": nodes,
        "blocked": sorted([list(t) for t in tele.blocked_authority_edges]),
        "quarantined": sorted([sorted(c) for c in tele.quarantined_supersede_cycles]),
    }


def lean_state(result: Mapping[str, Any]) -> Dict[str, Any]:
    nodes = {
        n["id"]: {k: n[k] for k in NODE_FIELDS} for n in result["nodes"]
    }
    return {
        "nodes": nodes,
        "blocked": sorted([list(t) for t in result["blocked_authority_edges"]]),
        "quarantined": sorted([sorted(c) for c in result["quarantined_supersede_cycles"]]),
    }


def fresh_from(store: "ek5.StratifiedEvidenceStore", rng: random.Random) -> "ek5.StratifiedEvidenceStore":
    """Rebuild the visible graph from scratch (shuffled insertion order) with the same strictness and missing set."""
    fresh = ek5.StratifiedEvidenceStore(strict_dag_supersede=store.strict_dag_supersede)
    fresh._missing_dep_nodes = set(store._missing_dep_nodes)
    events = list(store.events.values())
    rng.shuffle(events)
    seen: Set[Tuple[str, str, str]] = set()
    atts: List[Any] = []
    for att in store.attestations.values():
        key = (att.src_id, att.dst_id, att.edge_type)
        if att.active and key not in seen and store._edge_mult.get(key, 0) > 0:
            seen.add(key)
            atts.append(att)
    rng.shuffle(atts)
    fresh.apply_mutation(events, atts)
    return fresh


def oracle_vals(store: "ek5.StratifiedEvidenceStore", snapshot: Mapping[str, Any]) -> Dict[str, str]:
    """Textbook alternating-fixpoint WFS over the *snapshot's* dependency edges and the store's active refutes."""
    ids = [n["id"] for n in snapshot["nodes"]]
    deps_of: Dict[str, Set[str]] = {i: set() for i in ids}
    for e in snapshot["edges"]:
        if e["type"] in ek5.DEP_ALL_TYPES:
            deps_of[e["src"]].add(e["dst"])
    for m in snapshot["missing"]:
        deps_of[m].add(PHANTOM)
    deps_of[PHANTOM] = {PHANTOM}  # unsupportable atom: anything depending on it is unfounded
    refs_in = {w: set(store.refs.get(w, ())) for w in ids}
    out = ek5.compute_alternating_wfs_oracle(ids + [PHANTOM], refs_in, deps_of)
    return {i: out[i] for i in ids}


# ---------------------------------------------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------------------------------------------
def compare_states(
    a: Mapping[str, Any], b: Mapping[str, Any], fields: Sequence[str] = NODE_FIELDS, telemetry: bool = True
) -> List[Dict[str, Any]]:
    """Return the list of differences between two ``{nodes, blocked, quarantined}`` states."""
    diffs: List[Dict[str, Any]] = []
    na, nb = a["nodes"], b["nodes"]
    if set(na) != set(nb):
        diffs.append({"field": "node_set", "node": None, "a": sorted(set(na) - set(nb)), "b": sorted(set(nb) - set(na))})
    for nid in sorted(set(na) & set(nb)):
        for f in fields:
            if na[nid][f] != nb[nid][f]:
                diffs.append({"field": f, "node": nid, "a": na[nid][f], "b": nb[nid][f]})
    if telemetry:
        for f in ("blocked", "quarantined"):
            if a[f] != b[f]:
                diffs.append({"field": f, "node": None, "a": a[f], "b": b[f]})
    return diffs


def inject_fault(state: Mapping[str, Any], kind: int) -> Optional[Dict[str, Any]]:
    """Return a deterministic corruption of ``state`` (``None`` if the state has nothing to corrupt)."""
    nodes = {k: dict(v) for k, v in state["nodes"].items()}
    if not nodes:
        return None
    nid = sorted(nodes)[0]
    out: Dict[str, Any] = {"nodes": nodes, "blocked": [list(x) for x in state["blocked"]], "quarantined": [list(x) for x in state["quarantined"]]}
    k = kind % 9
    if k == 0:
        nodes[nid]["val"] = "F" if nodes[nid]["val"] != "F" else "T"
    elif k == 1:
        nodes[nid]["cur"] = "F" if nodes[nid]["cur"] != "F" else "T"
    elif k == 2:
        nodes[nid]["eligibility"] = "ADMIT" if nodes[nid]["eligibility"] != "ADMIT" else "AUDIT_ONLY"
    elif k == 3:
        nodes[nid]["eff_auth"] = nodes[nid]["eff_auth"] + 1
    elif k == 4:
        nodes[nid]["reason"] = nodes[nid]["reason"] + "_X"
    elif k == 5:
        nodes[nid]["pos_cyc"] = not nodes[nid]["pos_cyc"]
    elif k == 6:
        out["blocked"] = out["blocked"] + [["__x__", "__y__", "REFUTE"]]
    elif k == 7:
        out["quarantined"] = out["quarantined"] + [["__x__", "__y__"]]
    else:
        del nodes[nid]
    return out


# ---------------------------------------------------------------------------------------------------------------
# Sequence driver
# ---------------------------------------------------------------------------------------------------------------
def _visible_missing(store: "ek5.StratifiedEvidenceStore", vc: int, tc: int) -> Set[str]:
    vis = {
        ev.event_id
        for ev in store.events.values()
        if ev.tx_time <= tc and ev.valid_time <= vc and (ev.valid_end is None or vc < ev.valid_end)
    }
    miss: Set[str] = set()
    for att in store.attestations.values():
        if att.active and att.tx_time <= tc and att.valid_time <= vc:
            if att.edge_type in ek5.DEP_ALL_TYPES and att.src_id in vis and att.dst_id not in vis:
                miss.add(att.src_id)
    return miss


def _checkpoint(
    kind: str, seq_seed: int, step: int, store: "ek5.StratifiedEvidenceStore", rng: random.Random
) -> Dict[str, Any]:
    snapshot = materialize(store, "%d:%d:%s" % (seq_seed, step, kind))
    fresh = fresh_from(store, rng)
    return {
        "id": snapshot["snapshot_id"],
        "kind": kind,
        "snapshot": snapshot,
        "inc": store_state(store),
        "fresh": store_state(fresh),
        "oracle": {"python_inc": oracle_vals(store, snapshot), "python_fresh": oracle_vals(fresh, snapshot)},
    }


def run_sequence(seed: int, steps: int = DEFAULT_STEPS) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Run one seeded multi-mutation sequence; return its checkpoints and generation statistics."""
    rng = random.Random(seed)
    strict = rng.random() < 0.6
    store = ek5.StratifiedEvidenceStore(strict_dag_supersede=strict)
    model = _Model(strict)
    cps: List[Dict[str, Any]] = []
    for step in range(steps):
        batch = _gen_batch(rng, model, step)
        store.apply_mutation(batch["events"], batch["atts"], batch["withdraw"], remove_edges=batch["remove"])
        cps.append(_checkpoint("mutation", seed, step, store, rng))
    grid = [(vc, tc) for vc in range(0, 8) for tc in range(0, 8)]
    missing_grid = [g for g in grid if _visible_missing(store, g[0], g[1])]
    picks: List[Tuple[int, int]] = []
    if missing_grid:
        picks.append(rng.choice(missing_grid))
    picks.append(rng.choice(grid))
    for j, (vc, tc) in enumerate(picks):
        snap = store.as_of(vc, tc)
        cps.append(_checkpoint("as_of", seed, steps + j, snap, rng))
    stats = {
        "strict": 1 if strict else 0,
        "duplicate_triples": model.dup_triples_seen,
        "attestations": len(model.order),
        "withdrawn": sum(1 for a in model.order if not model.active.get(a)),
    }
    return cps, stats


# ---------------------------------------------------------------------------------------------------------------
# Lean CLI
# ---------------------------------------------------------------------------------------------------------------
def _find_lake() -> Optional[str]:
    found = shutil.which("lake")
    if found:
        return found
    for cand in (Path.home() / ".elan" / "bin" / "lake.exe", Path.home() / ".elan" / "bin" / "lake"):
        if cand.is_file():
            return str(cand)
    return None


def cli_path() -> Path:
    override = os.environ.get("EK5_REF_CLI")
    if override:
        return Path(override)
    exe = "ek5_ref_cli.exe" if os.name == "nt" else "ek5_ref_cli"
    return FORMAL_DIR / ".lake" / "build" / "bin" / exe


def ensure_cli(build: bool = True) -> Path:
    path = cli_path()
    if path.is_file():
        return path
    lake = _find_lake()
    if build and lake:
        subprocess.run([lake, "build", "ek5_ref_cli"], cwd=str(FORMAL_DIR), check=True, timeout=3600)
    if not path.is_file():
        raise FileNotFoundError("ek5_ref_cli not found at %s (build it with `lake build ek5_ref_cli` in formal/)" % path)
    return path


def run_lean_cli(cli: Path, snapshots: Sequence[Mapping[str, Any]], timeout: int = 900) -> List[Dict[str, Any]]:
    payload = ("\n".join(json.dumps(s, sort_keys=True, separators=(",", ":")) for s in snapshots) + "\n").encode("utf-8")
    proc = subprocess.run([str(cli)], input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
    if proc.returncode != 0:
        raise RuntimeError("ek5_ref_cli rc=%d: %s" % (proc.returncode, proc.stderr.decode("utf-8", "replace")[-400:]))
    lines = [ln for ln in proc.stdout.decode("utf-8").split("\n") if ln.strip()]
    if len(lines) != len(snapshots):
        raise RuntimeError("ek5_ref_cli returned %d lines for %d snapshots" % (len(lines), len(snapshots)))
    return [json.loads(ln) for ln in lines]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# Conformance
# ---------------------------------------------------------------------------------------------------------------
def verify_spec_lock() -> Dict[str, Any]:
    spec_meta_path = FORMAL_DIR / "spec_sha256.json"
    spec_lean_path = FORMAL_DIR / "ChelatedFormal" / "Spec.lean"
    if not spec_meta_path.is_file() or not spec_lean_path.is_file():
        return {"ok": False, "reason": "spec_sha256.json or Spec.lean missing"}
    meta = json.loads(spec_meta_path.read_text(encoding="utf-8"))
    actual = sha256_file(spec_lean_path)
    claimed = str(meta.get("spec_sha256", ""))
    locked = bool(meta.get("locked_before_proofs"))
    return {
        "ok": locked and actual == claimed,
        "locked_before_proofs": locked,
        "claimed_sha256": claimed,
        "actual_sha256": actual,
    }


def run_anchored_clipped_ema_burst_test(
    d: int = 384,
    steps: int = 100,
    rho: float = 0.05,
    alpha: float = 0.08,
    lam: float = 0.02,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Deterministic 100-query M6 adversarial query-burst test matching Theorem 4."""
    import math as _math

    rng = random.Random(seed)
    mu0 = [0.0] * d
    mu0[0] = 0.25
    adv_dir = [rng.gauss(0.0, 1.0) for _ in range(d)]
    adv_norm = _math.sqrt(sum(x * x for x in adv_dir))
    q_burst = [mu0[j] + 1.0 * (adv_dir[j] / adv_norm) for j in range(d)]

    mu_clip = list(mu0)
    mu_raw = list(mu0)
    max_clip_drift = 0.0
    max_raw_drift = 0.0
    for _ in range(steps):
        cand = [(1.0 - alpha - lam) * mu_clip[j] + alpha * q_burst[j] + lam * mu0[j] for j in range(d)]
        diff = [cand[j] - mu0[j] for j in range(d)]
        dn = _math.sqrt(sum(x * x for x in diff))
        if dn <= rho:
            mu_clip = cand
        else:
            scale = rho / dn
            mu_clip = [mu0[j] + scale * diff[j] for j in range(d)]
        clip_drift = _math.sqrt(sum((mu_clip[j] - mu0[j]) ** 2 for j in range(d)))
        max_clip_drift = max(max_clip_drift, clip_drift)

        mu_raw = [(1.0 - alpha) * mu_raw[j] + alpha * q_burst[j] for j in range(d)]
        raw_drift = _math.sqrt(sum((mu_raw[j] - mu0[j]) ** 2 for j in range(d)))
        max_raw_drift = max(max_raw_drift, raw_drift)

    return {
        "steps": steps,
        "dim": d,
        "rho": rho,
        "max_clipped_drift": round(max_clip_drift, 10),
        "max_unclipped_drift": round(max_raw_drift, 10),
        "clipped_within_rho": max_clip_drift <= rho + 1e-12,
        "unclipped_breached_rho": max_raw_drift > rho * 5.0,
    }


def _bump(d: Dict[str, int], key: str, n: int = 1) -> None:
    d[key] = d.get(key, 0) + n


def run_conformance(
    sequences: int = DEFAULT_SEQUENCES,
    steps: int = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
    cli: Optional[Path] = None,
    sample_limit: int = 12,
    enforce_minimums: bool = True,
) -> Dict[str, Any]:
    """Run the 3-way (+oracle) differential and return the report dictionary (``report["ok"]`` is the gate)."""
    t0 = time.perf_counter()
    cli = cli or ensure_cli()
    checkpoints: List[Dict[str, Any]] = []
    gen: Dict[str, int] = {"strict_sequences": 0, "nonstrict_sequences": 0, "duplicate_triples": 0, "attestations": 0, "withdrawn": 0}
    for i in range(sequences):
        cps, st = run_sequence(seed * 100003 + i, steps)
        checkpoints.extend(cps)
        gen["strict_sequences" if st["strict"] else "nonstrict_sequences"] += 1
        for k in ("duplicate_triples", "attestations", "withdrawn"):
            gen[k] += st[k]
    results = run_lean_cli(cli, [c["snapshot"] for c in checkpoints])

    pair_counts: Dict[str, Dict[str, int]] = {p: {} for p in ("lean_vs_fresh", "lean_vs_inc", "inc_vs_fresh")}
    samples: Dict[str, List[Dict[str, Any]]] = {p: [] for p in pair_counts}
    oracle_mismatch: Dict[str, int] = {"lean": 0, "python_inc": 0, "python_fresh": 0}
    oracle_samples: List[Dict[str, Any]] = []
    lean_not_ok = 0
    cov: Dict[str, int] = {}
    kinds = {"mutation": 0, "as_of": 0}
    asof_missing = 0
    lean_states: List[Optional[Dict[str, Any]]] = []

    for cp, res in zip(checkpoints, results):
        kinds[cp["kind"]] += 1
        snap = cp["snapshot"]
        if cp["kind"] == "as_of" and snap["missing"]:
            asof_missing += 1
        if not res.get("ok"):
            lean_not_ok += 1
            lean_states.append(None)
            continue
        lean = lean_state(res)
        lean_states.append(lean)
        for pair, (x, y) in (("lean_vs_fresh", (lean, cp["fresh"])), ("lean_vs_inc", (lean, cp["inc"])), ("inc_vs_fresh", (cp["inc"], cp["fresh"]))):
            for d in compare_states(x, y):
                key = d["field"]
                _bump(pair_counts[pair], key)
                if len(samples[pair]) < sample_limit:
                    samples[pair].append(dict(d, snapshot=cp["id"]))
        # textbook alternating fixpoint vs each implementation's three-valued ``val``
        for who, vals, oracle in (
            ("lean", {n: v["val"] for n, v in lean["nodes"].items()}, cp["oracle"]["python_inc"]),
            ("python_inc", {n: v["val"] for n, v in cp["inc"]["nodes"].items()}, cp["oracle"]["python_inc"]),
            ("python_fresh", {n: v["val"] for n, v in cp["fresh"]["nodes"].items()}, cp["oracle"]["python_fresh"]),
        ):
            if vals != oracle:
                oracle_mismatch[who] += 1
                if len(oracle_samples) < sample_limit:
                    oracle_samples.append({"snapshot": cp["id"], "who": who, "got": vals, "oracle": oracle})
        for nid, nd in lean["nodes"].items():
            _bump(cov, "val_" + nd["val"])
            _bump(cov, "cur_" + nd["cur"])
            _bump(cov, "elig_" + nd["eligibility"])
            _bump(cov, "reason_" + nd["reason"])
            if nd["pos_cyc"]:
                _bump(cov, "pos_cyc_nodes")
        _bump(cov, "blocked_authority_edges", len(lean["blocked"]))
        _bump(cov, "quarantined_supersede_cycles", len(lean["quarantined"]))
        _bump(cov, "strict_snapshots" if snap["strict_dag_supersede"] else "nonstrict_snapshots")
        _bump(cov, "missing_parent_nodes", len(snap["missing"]))
        _bump(cov, "nodes_total", len(snap["nodes"]))
        _bump(cov, "edges_total", len(snap["edges"]))

    # comparator liveness: every injected fault must be flagged
    injected = detected = 0
    for idx, (cp, lean) in enumerate(zip(checkpoints, lean_states)):
        if lean is None or idx % 5 != 0:
            continue
        bad = inject_fault(lean, idx // 5)
        if bad is None:
            continue
        injected += 1
        if compare_states(bad, cp["fresh"]):
            detected += 1

    semantic_div = {
        p: sum(c for f, c in pair_counts[p].items() if f in SEMANTIC_FIELDS or f == "node_set") for p in pair_counts
    }
    total_div = {p: sum(pair_counts[p].values()) for p in pair_counts}
    spec_lock = verify_spec_lock()
    ema_burst = run_anchored_clipped_ema_burst_test(seed=seed)
    gates: Dict[str, bool] = {
        "spec_sha256_locked": bool(spec_lock["ok"]),
        "ema_100_query_burst_bounded": bool(ema_burst["clipped_within_rho"] and ema_burst["unclipped_breached_rho"]),
        "lean_cli_all_snapshots_ok": lean_not_ok == 0,
        "min_sequences": (not enforce_minimums) or sequences >= MIN_SEQUENCES,
        "min_checkpoints": (not enforce_minimums) or len(checkpoints) >= MIN_CHECKPOINTS,
        "min_asof_missing_parent": (not enforce_minimums) or asof_missing >= MIN_ASOF_MISSING_PARENT,
        "lean_vs_fresh_exact": total_div["lean_vs_fresh"] == 0,
        "lean_vs_inc_semantic": semantic_div["lean_vs_inc"] == 0,
        "inc_vs_fresh_semantic": semantic_div["inc_vs_fresh"] == 0,
        "oracle_agrees_val": sum(oracle_mismatch.values()) == 0,
        "comparator_detects_all_injected_faults": injected > 0 and detected == injected,
        "coverage_three_valued": all(cov.get(k, 0) > 0 for k in ("val_T", "val_F", "val_U", "cur_T", "cur_F", "cur_U")),
        "coverage_structural": all(
            cov.get(k, 0) > 0 for k in ("pos_cyc_nodes", "blocked_authority_edges", "quarantined_supersede_cycles", "missing_parent_nodes")
        ),
    }
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": "formal/run_conformance.py",
        "tier": "ceiling-tier differential against the production ek5_v3_stratified.StratifiedEvidenceStore",
        "spec_lock": spec_lock,
        "ema_100_query_burst": ema_burst,
        "seed": seed,
        "sequences": sequences,
        "steps_per_sequence": steps,
        "checkpoints": {"total": len(checkpoints), "mutation": kinds["mutation"], "as_of": kinds["as_of"], "as_of_missing_parent": asof_missing},
        "generation": gen,
        "lean_cli": {"path": cli.name, "sha256": sha256_file(cli), "snapshots_evaluated": len(results), "not_ok": lean_not_ok},
        "divergences": {
            p: {"total": total_div[p], "semantic_fields": semantic_div[p], "by_field": dict(sorted(pair_counts[p].items()))}
            for p in pair_counts
        },
        "divergence_samples": samples,
        "oracle": {"mismatch_counts": oracle_mismatch, "samples": oracle_samples},
        "comparator_liveness": {"faults_injected": injected, "faults_detected": detected},
        "coverage": dict(sorted(cov.items())),
        "gates": gates,
        "ok": all(gates.values()),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }
    return report


# ---------------------------------------------------------------------------------------------------------------
# nanoda external kernel
# ---------------------------------------------------------------------------------------------------------------
def load_manifest(path: Path = MANIFEST_PATH) -> Dict[str, Any]:
    with open(str(path), "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or not data.get("nanoda_bin") or not isinstance(data.get("nanoda_args"), list):
        raise ValueError("nanoda_manifest.json must define nanoda_bin and nanoda_args")
    return data


def theorem_declarations(lean_file: Optional[Path] = None, namespace: str = "ChelatedFormal") -> List[str]:
    """Names of the ``theorem`` declarations in the Theorem-3 module (the set exported for the external kernel)."""
    lean_file = lean_file or (FORMAL_DIR / "ChelatedFormal" / "Theorem3_EK5_CyclicWFS.lean")
    text = lean_file.read_text(encoding="utf-8")
    names = re.findall(r"^theorem\s+([A-Za-z0-9_'.]+)", text, flags=re.M)
    return ["%s.%s" % (namespace, n) for n in names]


def _write_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "wb") as fh:
        fh.write(text.encode("utf-8"))


def prepare_nanoda(manifest: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Run ``lean4export`` (raw bytes, never a shell redirect) and write the nanoda config next to the export."""
    m = dict(manifest or load_manifest())
    export_path = FORMAL_DIR / str(m["export_file"])
    config_path = FORMAL_DIR / str(m["nanoda_args"][0])
    l4e = os.environ.get("LEAN4EXPORT_BIN") or str(m["lean4export_bin"])
    lake = _find_lake()
    if lake is None:
        raise FileNotFoundError("lake not found on PATH or in ~/.elan/bin")
    decls = list(m.get("export_declarations") or theorem_declarations())
    cmd = [lake, "env", l4e, str(m["lean4export_module"]), "--"] + decls
    raw = subprocess.check_output(cmd, cwd=str(FORMAL_DIR), timeout=1800)
    if not raw.startswith(b'{"meta"'):
        raise RuntimeError("lean4export produced no NDJSON header (got %r)" % raw[:60])
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(export_path), "wb") as fh:
        fh.write(raw)
    _write_lf(config_path, json.dumps(m["nanoda_config"], indent=2) + "\n")
    return {
        "export_file": str(m["export_file"]),
        "export_bytes": len(raw),
        "export_sha256": hashlib.sha256(raw).hexdigest(),
        "declarations_requested": len(decls),
        "config_file": str(m["nanoda_args"][0]),
    }


_NANODA_OK = re.compile(r"Checked (\d+) declarations with no typechecker errors")
_NANODA_PP = re.compile(r"(\d+) pretty printer errors")


def run_nanoda(manifest: Optional[Mapping[str, Any]] = None, timeout: int = 1800) -> Dict[str, Any]:
    m = dict(manifest or load_manifest())
    cmd = [str(m["nanoda_bin"])] + [str(a) for a in m["nanoda_args"]]
    proc = subprocess.run(cmd, cwd=str(FORMAL_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    out = proc.stdout.decode("utf-8", "replace")
    ok = _NANODA_OK.search(out)
    pp = _NANODA_PP.search(out)
    return {
        "rc": proc.returncode,
        "accepted": proc.returncode == 0 and ok is not None,
        "declarations_checked": int(ok.group(1)) if ok else None,
        "pretty_printer_errors": int(pp.group(1)) if pp else 0,
        "output_tail": out.strip()[-400:],
    }


# ---------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------
def write_report(report: Mapping[str, Any], path: Path = REPORT_PATH) -> None:
    _write_lf(path, json.dumps(report, indent=2, sort_keys=True) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sequences", type=int, default=DEFAULT_SEQUENCES)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--cli", type=Path, default=None, help="path to ek5_ref_cli (default: formal/.lake/build/bin)")
    ap.add_argument("--report", type=Path, default=REPORT_PATH)
    ap.add_argument("--prepare-nanoda", action="store_true", help="only (re)generate the lean4export file + nanoda config")
    ap.add_argument("--run-nanoda", action="store_true", help="only run nanoda_bin from the manifest")
    ap.add_argument("--with-nanoda", action="store_true", help="also prepare+run nanoda and embed the result in the report")
    args = ap.parse_args(argv)

    if args.prepare_nanoda or args.run_nanoda:
        rc = 0
        if args.prepare_nanoda:
            print(json.dumps(prepare_nanoda(), indent=2))
        if args.run_nanoda:
            res = run_nanoda()
            print(json.dumps(res, indent=2))
            rc = 0 if res["accepted"] else 1
        return rc

    report = run_conformance(sequences=args.sequences, steps=args.steps, seed=args.seed, cli=args.cli)
    if args.with_nanoda:
        try:
            prep = prepare_nanoda()
            report["nanoda"] = dict(run_nanoda(), **prep)
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
            report["nanoda"] = {"accepted": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        report["gates"]["nanoda_accepts_export"] = bool(report["nanoda"].get("accepted"))
        report["ok"] = all(report["gates"].values())
    write_report(report, args.report)
    cps = report["checkpoints"]
    print(
        "EK5 conformance: sequences=%d checkpoints=%d (as_of=%d, missing-parent=%d) ok=%s"
        % (report["sequences"], cps["total"], cps["as_of"], cps["as_of_missing_parent"], report["ok"])
    )
    for name, passed in sorted(report["gates"].items()):
        print("  [%s] %s" % ("PASS" if passed else "FAIL", name))
    for pair, info in sorted(report["divergences"].items()):
        if info["total"]:
            print("  divergence %s: total=%d semantic=%d by_field=%s" % (pair, info["total"], info["semantic_fields"], info["by_field"]))
    print("report: %s" % args.report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
