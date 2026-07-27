# Rung 13: detector-driven Evidence-DAG disintegration

## Detector to fitness mapping

`IsomerDetector` exposes a real per-query signal: `compute_isomer_strength` is
`1 - Jaccard` (`isomer_detector.py:75`) and each immediate `detect_*` result retains
the exact query key and strength (`isomer_detector.py:135`, `:164`).
`detector_signals_from_outputs` joins that key exactly to an Evidence-DAG query
node's `node_id`, `query_id`, or `query_text`. Query fitness is:

```text
isomer fitness = 1 - isomer strength = Jaccard similarity
```

A cluster's isomer component is the minimum fitness of its joined member queries.
No exact join means neutral fitness `1.0`; an empty detector result with explicit
`mode="sedimentation"` cannot trigger pruning. Missing mode fails closed with an
exception, as does any mode override: no other sign convention is validated.

`ConvergenceMonitor` exposes one training-run summary (`convergence_monitor.py:128`),
not a per-query or per-cluster result. Selective scoring therefore requires callers
to maintain one real monitor per cluster and pass
`{cluster_node_id: monitor.get_summary()}`. The mapping is:

```text
before min_epochs                         -> 1.0 (insufficient evidence)
converged                                 -> 1.0 (stable early-stop plateau)
mature and not converged                  ->
    1 - epochs_without_improvement / patience
```

The edge score is the minimum applicable query and cluster component. A mature
run with zero stalled epochs therefore remains healthy. A run approaching its
patience boundary falls toward zero; once it reaches the monitor's real converged
state, that recovered stability can re-anneal a ledgered edge.

Two production-wiring limits remain explicit. The engine creates convergence
monitors as local training variables (`antigravity_engine.py:1989`, `:2323`) and
does not currently persist their summaries. It also retains only aggregate isomer
history, while selective scoring needs the immediate per-query `detect_*` return.
This rung supplies the genuine scorer/prune/re-anneal contract for those real
outputs; it does not claim the engine automatically invokes it yet.

## Prune and re-anneal contract

`EvidenceDAG.prune_edges(scorer, threshold, dry_run=False)` scores every edge once,
removes only scores strictly below the threshold, never removes nodes, validates
the graph before and after mutation, and records detached copies of removed edges
in a runtime-only ledger. `dry_run=True` reports identical proposed decisions without
changing the graph or ledger. Non-finite or out-of-range scores fail closed. A
scorer or protection callback that adds/removes/mutates nodes, replaces/removes/
reorders/mutates edges, or mutates the pruned ledger is rejected against a joint
state snapshot in both normal and dry-run modes. Nodes, active edges, and the
ledger are restored together on returned mutation or exception.

`reanneal_edges(dag, scorer, threshold, recovered_signal)` considers only ledgered
edges. It rejects an invalid starting DAG before scoring. Scoring itself runs
inside the same complete transaction: callback mutation or exceptions restore
nodes, active edges, and the ledger. It then restores scores at or above the
threshold, validates each restoration, and keeps failed/low or individually
invalid ledger edges in the ledger.

Node and edge attributes follow the serialized schema: plain JSON objects composed
only of string keys, null, booleans, strings, integers, finite floats, lists, and
objects. Inputs are recursively validated and detached at ingestion; custom Python
objects, container subclasses, tuples, non-finite floats, non-string keys, and
cycles fail closed. Transaction snapshots use the same hook-free copier rather
than Python `deepcopy`, so attribute objects cannot execute copy hooks. Ledgered
edges and returned prune/re-anneal records are detached from live edge and nested
attribute aliases.

`write_disintegration_artifact` writes the prune and recovery thresholds, detector
provenance, edge-level fitness before/after, pruned edges, re-annealed edges, and
protected skips as deterministic JSON using an atomic same-directory replace.

## Fail-closed guard

The default protection treats `EdgeType.OPERATES_ON` as the current schema's
structural actuator-to-cluster edge. Any edge with `required=True` or
`structural=True` is also protected. Mandatory protection is always ORed with a
caller-supplied downstream `protected_predicate`, so the custom predicate can only
add protection and cannot weaken it. Protected edges remain in the graph even when
their detector score is low, and the artifact records the skip. With all detector
signals healthy or with an explicitly sedimentation-mode empty result, every score
is `1.0` and pruning is a no-op.
