# Codex build — Rung 13: real disintegration loop (drift-detector → Evidence-DAG edge prune)

Implement the genuine Phase-II rung-13 disintegration loop the ROADMAP names: drift detectors
(`isomer_detector` / `convergence_monitor`) drive pruning of low-fitness Evidence-DAG edges, with a
re-anneal path and a before/after fitness artifact. This is CPU-only, no GPU. Branch is already
checked out (`lattice/rung13-disintegration-20260714`).

## Read first (match real APIs — do not invent signatures)
- `evidence_dag.py` — `EvidenceDAG`, `EvidenceNode`, `EvidenceEdge`, `NodeType`, `EdgeType`,
  `add_node`, `add_edge`, `validate_evidence_dag`, `from_attribution_pool`, `to_json`.
- `isomer_detector.py` — what signal it exposes (isomer/collapse score per query/cluster).
- `convergence_monitor.py` — what stability/convergence signal it exposes.
- `stability_tracker.py` / `topology_analyzer.py` if the above reference them.

## Build
1. **Edge-fitness scorer.** A function that scores each Evidence-DAG edge's fitness from detector
   signals — e.g. an edge touching a cluster the `isomer_detector` flags as collapsed/isomeric, or one
   whose `convergence_monitor` signal shows non-convergence, gets a LOW fitness. Pure function of
   (edge, detector outputs); deterministic; documented mapping from detector signal → edge fitness.
2. **`EvidenceDAG.prune_edges(scorer, threshold, *, dry_run=False)`** (add to `evidence_dag.py`):
   removes edges whose fitness < threshold; returns a record with edges-before, edges-after, and the
   per-edge fitness scores. `dry_run=True` computes the record without mutating. Never removes a node,
   only edges; preserves DAG validity (`validate_evidence_dag` still passes after prune).
3. **Re-anneal path.** A `reanneal_edges(dag, scorer, threshold, recovered_signal)` that RE-ADDS a
   previously pruned edge if its cluster's detector signal has recovered above threshold — records the
   re-add in the artifact. Keep a small pruned-edge ledger so re-anneal can restore.
4. **Artifact.** A JSON record (fitness before/after per edge, pruned list, re-annealed list, thresholds,
   detector-signal provenance) writable to a path — the "records fitness before/after" the exit
   criteria require.
5. **Fail-closed guard.** If the scorer would prune an edge that is actually HIGH fitness by a
   downstream proxy (provide a `protected_predicate` hook, default: never prune an edge whose
   `EdgeType` is structural/required), skip it and log. The loop must never degrade an all-healthy DAG.

## Tests (`test_evidence_dag_disintegration.py`, plain unittest, offline)
- A DAG with one detector-flagged low-fitness edge → `prune_edges` removes exactly that edge; the
  healthy edges remain; `validate_evidence_dag` passes after.
- An all-healthy DAG → `prune_edges` is a **no-op** (baseline: pruning nothing when nothing is unfit).
- `dry_run=True` mutates nothing but returns the same record.
- Re-anneal: prune an edge, then feed a recovered signal → the edge is restored; artifact logs both.
- Artifact has fitness before/after and detector provenance.
- Fail-closed: a protected/structural edge is never pruned even if scored low.

## Deliverables
1. `evidence_dag.py` extended (`prune_edges`, `reanneal_edges`, pruned-edge ledger) + a
   `evidence_dag_disintegration.py` (or in-module) scorer wiring isomer/convergence signals.
2. `test_evidence_dag_disintegration.py`; run `python -m unittest test_evidence_dag_disintegration -v`
   AND `python -m unittest test_evidence_dag -v` (no regression); paste both results.
3. A short `docs/rung13-disintegration-loop.md` describing the detector→fitness mapping, the prune/
   re-anneal contract, and the fail-closed guard.

Be brutally honest: if `isomer_detector` / `convergence_monitor` do not expose a per-cluster/per-edge
signal you can honestly map to edge fitness, say so with file:line and propose the smallest real signal
(do NOT fabricate a mapping). No stubs, no `pass`-body placeholders. The prune must be driven by a real
detector signal, not a random or hardcoded score.
