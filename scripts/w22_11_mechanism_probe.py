#!/usr/bin/env python3
"""W22-11 POST-HOC mechanism probe (not preregistered; written after the main run).

This probe explains *why* raw ``pos_cyc``/``reason`` differ between the incremental store and a fresh
rebuild. It wraps the incremental store's ``_solve_stratum2_region`` on the instance and records each
batch's dirty region. It does not change what the engine computes. It replays the preregistered tiers'
first N sequences and classifies every raw node diff at mutation checkpoints:

* ``pos_cyc`` with the node in this batch's dirty region:
  * ``partial_scc_in_dirty``: the node's full D_V SCC (from the snapshot) is not inside the dirty region,
    so the engine's dirty-restricted Tarjan missed the cycle;
  * ``unexplained_in_dirty``: anything else.
* ``pos_cyc`` with the node outside this batch's dirty region:
  * ``stale_not_resolved_this_batch``: the flag was set in an earlier batch and the node was not re-solved
    (pruned or outside the cone).
* ``reason`` diffs: split by whether the same node also has a ``pos_cyc`` diff.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ek5_v3_stratified as ek5  # noqa: E402
import w22_11_wfs_deletion_stress as w  # noqa: E402
from run_pv7_w23c_w22_11_wfs_deletion_stress import preregistered_tiers  # noqa: E402


def probe_sequence(cfg: w.TierConfig, i: int, out: Counter) -> None:
    seed = cfg.seq_seed(i)
    rng = random.Random(seed)
    strict = rng.random() < 0.6
    store = ek5.StratifiedEvidenceStore(strict_dag_supersede=strict)
    model = w.rc._Model(strict)
    dirty_log: list = []
    orig = store._solve_stratum2_region

    def wrapped(dirty, **kw):
        dirty_log.append(set(dirty))
        return orig(dirty, **kw)

    store._solve_stratum2_region = wrapped
    for step in range(cfg.steps):
        dirty_log.clear()
        batch = w.gen_batch_baseline(rng, model, step) if cfg.generator == "baseline" else w.gen_batch_stress(rng, model, step, cfg)
        store.apply_mutation(batch["events"], batch["atts"], batch["withdraw"], remove_edges=batch["remove"])
        snap = w.rc.model_snapshot(model, "p", None)
        fresh = w.rc.fresh_from_model(model, snap, rng, None)
        scc_of = {}
        deps = w.rc.snapshot_deps(snap)
        for comp in ek5._iterative_tarjan_sccs(set(deps), deps, subset=set(deps)):
            for u in comp:
                scc_of[u] = set(comp)
        dirty = set().union(*dirty_log) if dirty_log else set()
        pos_diff = set()
        for nid in store.events:
            a, b = bool(store.pos_cyc[nid]), bool(fresh.pos_cyc[nid])
            if a != b:
                pos_diff.add(nid)
                direction = "inc_missed_cycle" if b else "inc_stale_cycle"
                if nid in dirty:
                    cls = "partial_scc_in_dirty" if not scc_of[nid] <= dirty else "unexplained_in_dirty"
                else:
                    cls = "stale_not_resolved_this_batch"
                out["|".join((cfg.name, "pos_cyc", direction, cls))] += 1
        for nid in store.events:
            if store.reason[nid] != fresh.reason[nid]:
                tag = "with_pos_cyc_diff" if nid in pos_diff else "without_pos_cyc_diff"
                out["|".join((cfg.name, "reason", "%s->%s" % (store.reason[nid], fresh.reason[nid]), tag))] += 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts" / "pv7_w23c_w22_11"))
    ap.add_argument("--sequences", type=int, default=200)
    args = ap.parse_args(argv)
    out: Counter = Counter()
    for cfg in preregistered_tiers():
        if cfg.name in ("A0", "A1"):
            continue
        n = min(cfg.sequences, args.sequences if cfg.max_nodes <= 64 else max(1, args.sequences // 8))
        for i in range(n):
            probe_sequence(cfg, i, out)
        out[cfg.name + "|sequences_probed"] = n
    res = {"post_hoc": True, "note": __doc__.strip().splitlines()[0], "counts": dict(sorted(out.items()))}
    Path(args.out_dir, "w22_11_mechanism_probe.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
