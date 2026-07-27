"""Detector-driven Evidence-DAG disintegration and re-annealing (rung 13).

The integration is intentionally explicit about detector scope.  IsomerDetector
emits per-query strengths, so those values are joined to DAG query nodes by exact
``node_id``, ``query_id``, or ``query_text``.  ConvergenceMonitor is run-global;
selective edge scoring therefore accepts only summaries already keyed by cluster
node id (one real monitor per cluster).  Missing, unmatched, and immature signals
are neutral fitness 1.0 so missing evidence can never cause destructive pruning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from evidence_dag import EdgeType, EvidenceDAG, EvidenceEdge, NodeType, validate_evidence_dag


def _bounded_float(value: Any, name: str) -> float:
    result = float(value)
    if result != result or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0.0, 1.0], got {value!r}")
    return result


def _query_join_keys(dag: EvidenceDAG) -> Dict[str, List[str]]:
    joined: Dict[str, List[str]] = {}
    for node in dag.nodes:
        if node.node_type is not NodeType.QUERY:
            continue
        keys = [node.node_id, node.attrs.get("query_id"), node.attrs.get("query_text")]
        for key in keys:
            if key is None or str(key) == "":
                continue
            joined.setdefault(str(key), []).append(node.node_id)
    return joined


def _convergence_fitness(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Map one real ``ConvergenceMonitor.get_summary()`` to stability fitness.

    Before ``min_epochs`` there is insufficient evidence, so fitness is 1.0.
    A converged (early-stop plateau) monitor is stable and scores 1.0.  A mature
    but not-yet-converged monitor scores ``1 - no_improvement / patience``.  Thus
    an unstable run approaching its patience boundary falls toward zero and can
    be pruned, while active improvement (zero stalled epochs) remains healthy.
    """
    total_epochs = int(summary.get("total_epochs", 0))
    min_epochs = max(1, int(summary.get("min_epochs", 1)))
    patience = max(1, int(summary.get("patience", 1)))
    stalled = max(0, int(summary.get("epochs_without_improvement", 0)))
    converged = bool(summary.get("converged", False))
    if total_epochs < min_epochs:
        return {"fitness": 1.0, "state": "insufficient_evidence"}
    if converged:
        return {"fitness": 1.0, "state": "converged"}
    return {
        "fitness": max(0.0, 1.0 - min(stalled, patience) / patience),
        "state": "mature_not_converged",
    }


def detector_signals_from_outputs(
    dag: EvidenceDAG,
    isomer_output: Mapping[str, Any],
    convergence_by_cluster: Optional[Mapping[str, Mapping[str, Any]]] = None,
    *,
    provenance: Optional[Mapping[str, Any]] = None,
    expected_isomer_mode: str = "sedimentation",
) -> Dict[str, Any]:
    """Normalize actual detector outputs into edge-addressable fitness signals.

    ``isomer_output`` is the dict returned by an IsomerDetector ``detect_*``
    method. ``convergence_by_cluster`` maps a DAG cluster node id to a real
    ``ConvergenceMonitor.get_summary()`` result.  A single global convergence
    summary is deliberately not accepted as selective per-edge evidence.

    The ``fitness = 1 - strength`` mapping is only correct-signed for
    ``mode="sedimentation"`` (high pre/post drift = unstable = low fitness). For
    ``mode="chelation"`` (standard vs chelated) a high strength can mean the
    correction *worked*, so that mapping would be inverted; this function
    fail-closes (raises) on a non-sedimentation mode rather than silently
    mis-signalling. Pass ``expected_isomer_mode`` explicitly only with a
    validated mapping for that mode.
    """
    mode = isomer_output.get("mode")
    if mode is not None and str(mode) != expected_isomer_mode:
        raise ValueError(
            f"isomer fitness mapping (fitness = 1 - strength) is defined only for "
            f"mode={expected_isomer_mode!r}; got mode={mode!r}. For 'chelation' mode a "
            f"high strength can mean the correction worked, not that the edge is unfit."
        )
    query_join = _query_join_keys(dag)
    query_fitness: Dict[str, float] = {}
    unmatched_queries: List[str] = []
    for bucket in ("isomers", "non_isomers"):
        for entry in isomer_output.get(bucket, []) or []:
            query_key = str(entry["query"])
            strength = _bounded_float(entry["strength"], f"isomer strength for {query_key}")
            matched = query_join.get(query_key, [])
            if not matched:
                unmatched_queries.append(query_key)
            for node_id in matched:
                fitness = 1.0 - strength
                query_fitness[node_id] = min(query_fitness.get(node_id, 1.0), fitness)

    cluster_members: Dict[str, List[str]] = {}
    for edge in dag.edges:
        if edge.edge_type is EdgeType.RETRIEVED_IN:
            cluster_members.setdefault(edge.dst, []).append(edge.src)
    isomer_cluster_fitness = {
        cluster_id: min(
            (query_fitness[qid] for qid in members if qid in query_fitness),
            default=1.0,
        )
        for cluster_id, members in cluster_members.items()
    }

    convergence_fitness: Dict[str, float] = {}
    convergence_states: Dict[str, str] = {}
    unmatched_clusters: List[str] = []
    cluster_ids = {node.node_id for node in dag.nodes if node.node_type is NodeType.CLUSTER}
    for cluster_id, summary in (convergence_by_cluster or {}).items():
        cluster_id = str(cluster_id)
        if cluster_id not in cluster_ids:
            unmatched_clusters.append(cluster_id)
            continue
        mapped = _convergence_fitness(summary)
        convergence_fitness[cluster_id] = mapped["fitness"]
        convergence_states[cluster_id] = mapped["state"]

    return {
        "isomer_query_fitness": query_fitness,
        "isomer_cluster_fitness": isomer_cluster_fitness,
        "convergence_cluster_fitness": convergence_fitness,
        "provenance": {
            "isomer_detector": {
                "module": "isomer_detector.IsomerDetector",
                "mode": isomer_output.get("mode"),
                "total_queries": isomer_output.get("total_queries"),
                "unmatched_query_keys": sorted(set(unmatched_queries)),
                "mapping": "fitness = 1 - emitted isomer strength (exact query join)",
            },
            "convergence_monitor": {
                "module": "convergence_monitor.ConvergenceMonitor.get_summary",
                "scope": "one summary per exact EvidenceDAG cluster node id",
                "states": convergence_states,
                "unmatched_cluster_ids": sorted(set(unmatched_clusters)),
                "mapping": (
                    "immature or converged = 1; mature/non-converged = " "1 - epochs_without_improvement / patience"
                ),
            },
            **dict(provenance or {}),
        },
    }


def score_edge_fitness(edge: EvidenceEdge, detector_outputs: Mapping[str, Any]) -> float:
    """Pure deterministic edge fitness: minimum applicable detector component.

    Query endpoints use per-query isomer fitness. Cluster endpoints use the
    worst member-query isomer fitness and the cluster-scoped convergence fitness.
    With no exact signal match, the edge remains at fail-closed fitness 1.0.
    """
    components = [1.0]
    query_scores = detector_outputs.get("isomer_query_fitness", {})
    cluster_isomer_scores = detector_outputs.get("isomer_cluster_fitness", {})
    cluster_convergence_scores = detector_outputs.get("convergence_cluster_fitness", {})
    for endpoint in (edge.src, edge.dst):
        if endpoint in query_scores:
            components.append(_bounded_float(query_scores[endpoint], endpoint))
        if endpoint in cluster_isomer_scores:
            components.append(_bounded_float(cluster_isomer_scores[endpoint], endpoint))
        if endpoint in cluster_convergence_scores:
            components.append(_bounded_float(cluster_convergence_scores[endpoint], endpoint))
    return min(components)


def reanneal_edges(
    dag: EvidenceDAG,
    scorer: Callable[[EvidenceEdge, Mapping[str, Any]], float],
    threshold: float,
    recovered_signal: Mapping[str, Any],
) -> Dict[str, Any]:
    """Restore ledgered edges whose recovered detector fitness clears threshold."""
    threshold = _bounded_float(threshold, "threshold")
    violations = validate_evidence_dag(dag)
    if violations:
        raise ValueError("cannot re-anneal on an invalid EvidenceDAG: " + "; ".join(violations))
    before = len(dag.edges)
    original_edges = list(dag._edges)
    original_ledger = list(dag._pruned_edge_ledger)
    remaining: List[Dict[str, Any]] = []
    scores: List[Dict[str, Any]] = []
    reannealed: List[Dict[str, Any]] = []
    skipped_invalid: List[Dict[str, Any]] = []

    # Score the complete ledger before mutation.  A bad scorer can therefore
    # never leave a half-restored graph with a stale ledger.
    scored_entries = []
    for ledger_entry in original_ledger:
        fitness = _bounded_float(scorer(ledger_entry["edge"], recovered_signal), ledger_entry["edge_id"])
        scored_entries.append((ledger_entry, fitness))

    try:
        for ledger_entry, fitness in scored_entries:
            edge = ledger_entry["edge"]
            score_record = {
                "edge_id": ledger_entry["edge_id"],
                "edge": edge.to_dict(),
                "fitness_before": ledger_entry["fitness_at_prune"],
                "fitness_after": fitness,
                "decision": "retain_pruned",
            }
            if fitness < threshold:
                remaining.append(ledger_entry)
                scores.append(score_record)
                continue

            # Duplicate equal edges are legal in EvidenceDAG.  Each ledger entry
            # represents one occurrence and must therefore be appended once.
            dag._edges.append(edge)
            violations = validate_evidence_dag(dag)
            if violations:
                dag._edges.pop()
                remaining.append(ledger_entry)
                score_record["decision"] = "skip_invalid"
                score_record["violations"] = violations
                skipped_invalid.append(edge.to_dict())
            else:
                score_record["decision"] = "reanneal"
                reannealed.append(edge.to_dict())
            scores.append(score_record)
        dag._pruned_edge_ledger = remaining
    except Exception:
        dag._edges = original_edges
        dag._pruned_edge_ledger = original_ledger
        raise
    return {
        "record_type": "evidence_dag_reanneal",
        "threshold": threshold,
        "edges_before": before,
        "edges_after": len(dag.edges),
        "scores": scores,
        "reannealed": reannealed,
        "retained_pruned": [entry["edge"].to_dict() for entry in remaining],
        "skipped_invalid": skipped_invalid,
    }


def write_disintegration_artifact(
    path: str,
    prune_record: Mapping[str, Any],
    reanneal_record: Optional[Mapping[str, Any]] = None,
    *,
    detector_provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    """Write a deterministic before/after lifecycle artifact and return it."""
    if not detector_provenance:
        raise ValueError("detector_provenance must be a non-empty mapping")
    recovered_by_id = {item["edge_id"]: item for item in (reanneal_record or {}).get("scores", [])}
    edge_fitness = []
    for item in prune_record.get("scores", []):
        recovered = recovered_by_id.get(item["edge_id"])
        edge_fitness.append(
            {
                "edge_id": item["edge_id"],
                "edge": item["edge"],
                "fitness_before": item["fitness_before"],
                "fitness_after": (recovered["fitness_after"] if recovered is not None else item["fitness_after"]),
            }
        )
    artifact = {
        "record_type": "evidence_dag_disintegration",
        "schema_version": 1,
        "thresholds": {
            "prune": prune_record.get("threshold"),
            "reanneal": (reanneal_record or {}).get("threshold"),
        },
        "detector_signal_provenance": dict(detector_provenance),
        "edges_before": prune_record.get("edges_before"),
        "edges_after": ((reanneal_record or {}).get("edges_after", prune_record.get("edges_after"))),
        "edge_fitness": edge_fitness,
        "pruned": list(prune_record.get("pruned", [])),
        "reannealed": list((reanneal_record or {}).get("reannealed", [])),
        "protected_skips": list(prune_record.get("protected_skips", [])),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return artifact
