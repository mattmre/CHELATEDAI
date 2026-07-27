"""Evidence DAG — typed graph contract over the attribution pool (Phase II, rung 12).

The attribution pool (``build_attribution_pool.py``) is a set of FLAT tables. This
module gives those relationships a typed, validated **directed acyclic graph**
contract — the substrate the disintegration loop (rung 13) prunes and a GNN
(rung 15) would later learn over. There is NO GNN here and no learning: this is
the schema + a structural validator + a deterministic builder from the flat pool.

Three node types and three typed edges, oriented query -> {cluster, actuator} and
actuator -> cluster so the graph is acyclic by construction:

    QUERY  --RETRIEVED_IN-->  CLUSTER
    QUERY  --CORRECTED_BY-->  ACTUATOR        (a correction-actuator: adapter/mask/route)
    ACTUATOR --OPERATES_ON--> CLUSTER

The validator enforces: unique node ids, known node/edge types, edges referencing
existing nodes, edge endpoint-type constraints, and acyclicity. Stdlib only — no
numpy/torch — so it is CI-cheap and import-safe everywhere.

Node and edge attributes are detached plain JSON values. This matches the external
schema and keeps transaction snapshots free of arbitrary Python object hooks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple


class NodeType(str, Enum):
    QUERY = "query"
    CLUSTER = "cluster"
    ACTUATOR = "actuator"  # correction-actuator: adapter / mask / steering route


class EdgeType(str, Enum):
    RETRIEVED_IN = "retrieved_in"  # QUERY -> CLUSTER
    CORRECTED_BY = "corrected_by"  # QUERY -> ACTUATOR
    OPERATES_ON = "operates_on"  # ACTUATOR -> CLUSTER


# Allowed (source, destination) node types per edge type. This is the typed-graph
# contract; the validator rejects any edge whose endpoints violate it.
EDGE_ENDPOINTS: Dict[EdgeType, Tuple[NodeType, NodeType]] = {
    EdgeType.RETRIEVED_IN: (NodeType.QUERY, NodeType.CLUSTER),
    EdgeType.CORRECTED_BY: (NodeType.QUERY, NodeType.ACTUATOR),
    EdgeType.OPERATES_ON: (NodeType.ACTUATOR, NodeType.CLUSTER),
}


def _copy_plain_json(value: Any, path: str, active: Optional[set] = None) -> Any:
    """Validate and detach a value from the supported plain-JSON attribute domain.

    Exact built-in types are required deliberately. Accepting arbitrary Mapping,
    Sequence, numeric, or string subclasses would let user-defined iteration,
    conversion, equality, or copy hooks execute inside graph transactions.
    """
    value_type = type(value)
    if value is None or value_type in (str, bool, int):
        return value
    if value_type is float:
        if not isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return value

    if value_type not in (list, dict):
        raise TypeError(f"{path} must contain only plain JSON-compatible values; " f"got {value_type.__name__}")

    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise ValueError(f"{path} must not contain cyclic containers")
    active.add(identity)
    try:
        if value_type is list:
            return [_copy_plain_json(item, f"{path}[{index}]", active) for index, item in enumerate(value)]

        copied: Dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} keys must be plain strings; got {type(key).__name__}")
            copied[key] = _copy_plain_json(item, f"{path}.{key}", active)
        return copied
    finally:
        active.remove(identity)


def _canonical_attrs(attrs: Mapping[str, Any], owner: str) -> Dict[str, Any]:
    copied = _copy_plain_json(attrs, f"{owner}.attrs")
    if type(copied) is not dict:  # defensive: attrs is an object in the schema
        raise TypeError(f"{owner}.attrs must be a plain JSON object")
    return copied


@dataclass(frozen=True)
class EvidenceNode:
    node_id: str
    node_type: NodeType
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attrs", _canonical_attrs(self.attrs, "EvidenceNode"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type.value,
            "attrs": _canonical_attrs(self.attrs, "EvidenceNode"),
        }


@dataclass(frozen=True)
class EvidenceEdge:
    src: str
    dst: str
    edge_type: EdgeType
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attrs", _canonical_attrs(self.attrs, "EvidenceEdge"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "type": self.edge_type.value,
            "attrs": _canonical_attrs(self.attrs, "EvidenceEdge"),
        }


_DAGState = Tuple[
    Dict[str, EvidenceNode],
    List[EvidenceEdge],
    List[Dict[str, Any]],
]


class EvidenceDAG:
    """A typed directed (intended-acyclic) evidence graph."""

    RECORD_TYPE = "evidence_dag"

    def __init__(self) -> None:
        self._nodes: Dict[str, EvidenceNode] = {}
        self._edges: List[EvidenceEdge] = []
        # Runtime-only ledger.  It is deliberately excluded from the rung-12 JSON
        # schema: a serialized DAG describes the current graph, while the rung-13
        # lifecycle artifact records why edges left or re-entered it.
        self._pruned_edge_ledger: List[Dict[str, Any]] = []

    # -- construction --------------------------------------------------------
    def add_node(self, node_id: str, node_type: NodeType, **attrs: Any) -> EvidenceNode:
        node = EvidenceNode(str(node_id), NodeType(node_type), dict(attrs))
        # Idempotent for identical (id, type); a conflicting re-add is a caller bug.
        existing = self._nodes.get(node.node_id)
        if existing is not None and existing.node_type != node.node_type:
            raise ValueError(
                f"node id {node.node_id!r} re-added with type {node.node_type.value!r} "
                f"(was {existing.node_type.value!r})"
            )
        self._nodes[node.node_id] = node
        return node

    def add_edge(self, src: str, dst: str, edge_type: EdgeType, **attrs: Any) -> EvidenceEdge:
        edge = EvidenceEdge(str(src), str(dst), EdgeType(edge_type), dict(attrs))
        self._edges.append(edge)
        return edge

    @staticmethod
    def _clone_node(node: EvidenceNode) -> EvidenceNode:
        if type(node) is not EvidenceNode:
            raise TypeError("EvidenceDAG node state must contain exact EvidenceNode instances")
        if type(node.node_id) is not str or type(node.node_type) is not NodeType:
            raise TypeError("EvidenceDAG node identity and type must be canonical")
        return EvidenceNode(node.node_id, node.node_type, node.attrs)

    @staticmethod
    def _clone_edge(edge: EvidenceEdge) -> EvidenceEdge:
        if type(edge) is not EvidenceEdge:
            raise TypeError("EvidenceDAG edge state must contain exact EvidenceEdge instances")
        if type(edge.src) is not str or type(edge.dst) is not str or type(edge.edge_type) is not EdgeType:
            raise TypeError("EvidenceDAG edge identity and type must be canonical")
        return EvidenceEdge(edge.src, edge.dst, edge.edge_type, edge.attrs)

    @classmethod
    def _clone_ledger_entry(cls, entry: Dict[str, Any]) -> Dict[str, Any]:
        if type(entry) is not dict:
            raise TypeError("EvidenceDAG pruned ledger entries must be plain dictionaries")
        expected = {"edge_id", "edge", "fitness_at_prune", "threshold"}
        if set(entry) != expected:
            raise TypeError(
                "EvidenceDAG pruned ledger entry fields must be exactly " "edge_id, edge, fitness_at_prune, threshold"
            )
        if type(entry["edge_id"]) is not str:
            raise TypeError("EvidenceDAG pruned ledger edge_id must be a plain string")
        return {
            "edge_id": entry["edge_id"],
            "edge": cls._clone_edge(entry["edge"]),
            "fitness_at_prune": _copy_plain_json(
                entry["fitness_at_prune"],
                "EvidenceDAG.pruned_edge_ledger.fitness_at_prune",
            ),
            "threshold": _copy_plain_json(
                entry["threshold"],
                "EvidenceDAG.pruned_edge_ledger.threshold",
            ),
        }

    @classmethod
    def _clone_ledger(cls, ledger: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if type(ledger) is not list:
            raise TypeError("EvidenceDAG pruned ledger must be a plain list")
        return [cls._clone_ledger_entry(entry) for entry in ledger]

    def _snapshot_state(self) -> _DAGState:
        """Return a detached, hook-free transaction snapshot of all graph state."""
        if type(self._nodes) is not dict or type(self._edges) is not list:
            raise TypeError("EvidenceDAG node and edge containers must be plain containers")

        nodes: Dict[str, EvidenceNode] = {}
        for node_id, node in self._nodes.items():
            if type(node_id) is not str:
                raise TypeError("EvidenceDAG node map keys must be plain strings")
            cloned_node = self._clone_node(node)
            if cloned_node.node_id != node_id:
                raise TypeError("EvidenceDAG node map keys must match canonical node ids")
            nodes[node_id] = cloned_node
        edges = [self._clone_edge(edge) for edge in self._edges]
        ledger = self._clone_ledger(self._pruned_edge_ledger)
        return nodes, edges, ledger

    def _restore_state(self, snapshot: _DAGState) -> None:
        self._nodes, self._edges, self._pruned_edge_ledger = snapshot

    def _state_matches(self, snapshot: _DAGState) -> bool:
        """Compare current state only after canonical validation and detachment."""
        return self._snapshot_state() == snapshot

    @property
    def nodes(self) -> List[EvidenceNode]:
        return list(self._nodes.values())

    @property
    def edges(self) -> List[EvidenceEdge]:
        return list(self._edges)

    def node(self, node_id: str) -> Optional[EvidenceNode]:
        return self._nodes.get(str(node_id))

    @property
    def pruned_edge_ledger(self) -> List[Dict[str, Any]]:
        """Return a defensive copy of edges eligible for re-annealing."""
        return self._clone_ledger(self._pruned_edge_ledger)

    @staticmethod
    def _edge_record_id(edge: EvidenceEdge, occurrence: int) -> str:
        """Stable-in-record identifier that also distinguishes duplicate edges."""
        return f"{edge.src}|{edge.edge_type.value}|{edge.dst}|{occurrence}"

    @staticmethod
    def default_protected_predicate(edge: EvidenceEdge) -> bool:
        """Protect required topology edges when no downstream proxy is supplied.

        ``EdgeType`` has no literal STRUCTURAL/REQUIRED member.  In the current
        schema, ``OPERATES_ON`` is the structural actuator-to-cluster link.  A
        caller may also mark any edge ``required=True`` or ``structural=True``.
        """
        return (
            edge.edge_type is EdgeType.OPERATES_ON
            or bool(edge.attrs.get("required"))
            or bool(edge.attrs.get("structural"))
        )

    def prune_edges(
        self,
        scorer: Callable[[EvidenceEdge], float],
        threshold: float,
        *,
        dry_run: bool = False,
        protected_predicate: Optional[Callable[[EvidenceEdge], bool]] = None,
    ) -> Dict[str, Any]:
        """Score and remove edges with fitness strictly below ``threshold``.

        The scorer is called once per edge and must return a finite value in
        ``[0, 1]``.  Nodes are never removed.  A dry run reports the same proposed
        decisions without mutating either the edge list or the pruned-edge ledger.
        Mandatory structural/required protection is always applied; a custom
        predicate can add protection but cannot weaken the mandatory predicate.
        """
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")

        # Canonicalize before structural validation so even a privately injected
        # object/subclass cannot execute attribute hooks through the validator.
        state_snapshot = self._snapshot_state()
        violations = validate_evidence_dag(self)
        if violations:
            raise ValueError("cannot prune an invalid EvidenceDAG: " + "; ".join(violations))

        occurrences: Dict[Tuple[str, str, str], int] = {}
        decisions: List[Tuple[EvidenceEdge, str, float, bool, bool]] = []

        # Preflight canonicalization is deliberately outside the callback boundary:
        # it rejects hostile pre-existing attrs without invoking arbitrary copy
        # hooks. Once callbacks begin, nodes, edges, and ledger are one transaction.
        live_edges = list(self._edges)
        try:
            for edge in live_edges:
                identity = (edge.src, edge.edge_type.value, edge.dst)
                occurrence = occurrences.get(identity, 0)
                occurrences[identity] = occurrence + 1
                edge_id = self._edge_record_id(edge, occurrence)
                fitness = float(scorer(edge))
                if fitness != fitness or not 0.0 <= fitness <= 1.0:
                    raise ValueError(
                        f"scorer returned non-finite/out-of-range fitness {fitness!r} " f"for edge {edge_id}"
                    )

                mandatory_protection = self.default_protected_predicate(edge)
                custom_protection = (
                    bool(protected_predicate(edge))
                    if protected_predicate is not None and not mandatory_protection
                    else False
                )
                is_protected = mandatory_protection or custom_protection
                would_prune = fitness < threshold and not is_protected
                decisions.append((edge, edge_id, fitness, is_protected, would_prune))

            try:
                state_unchanged = self._state_matches(state_snapshot)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "scorer or protection predicate left non-canonical EvidenceDAG " "state during prune_edges"
                ) from exc
            if not state_unchanged:
                raise RuntimeError(
                    "scorer or protection predicate mutated the DAG edge set, node "
                    "set, or pruned ledger state during prune_edges; refusing to "
                    "apply a prune computed over a stale snapshot"
                )
        except Exception:
            self._restore_state(state_snapshot)
            raise

        scores: List[Dict[str, Any]] = []
        retained: List[EvidenceEdge] = []
        proposed_ledger: List[Dict[str, Any]] = []
        for edge, edge_id, fitness, is_protected, would_prune in decisions:
            decision = (
                "prune" if would_prune else ("skip_protected" if fitness < threshold and is_protected else "keep")
            )
            edge_record = edge.to_dict()
            scores.append(
                {
                    "edge_id": edge_id,
                    "edge": edge_record,
                    "fitness_before": fitness,
                    "fitness_after": None if would_prune else fitness,
                    "decision": decision,
                    "protected": is_protected,
                }
            )
            if would_prune:
                proposed_ledger.append(
                    {
                        "edge_id": edge_id,
                        "edge": self._clone_edge(edge),
                        "fitness_at_prune": fitness,
                        "threshold": threshold,
                    }
                )
            else:
                retained.append(edge)

        record = {
            "record_type": "evidence_dag_prune",
            "threshold": threshold,
            "dry_run": bool(dry_run),
            "edges_before": len(live_edges),
            "edges_after": len(retained),
            "scores": scores,
            "pruned": [entry["edge"].to_dict() for entry in proposed_ledger],
            "protected_skips": [
                _copy_plain_json(item["edge"], "prune_record.protected_skips")
                for item in scores
                if item["decision"] == "skip_protected"
            ],
        }
        if dry_run:
            return _copy_plain_json(record, "prune_record")

        self._edges = retained
        post_violations = validate_evidence_dag(self)
        if post_violations:
            self._restore_state(state_snapshot)
            raise ValueError("prune would invalidate EvidenceDAG: " + "; ".join(post_violations))
        self._pruned_edge_ledger.extend(proposed_ledger)
        return _copy_plain_json(record, "prune_record")

    # -- serialization -------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_type": self.RECORD_TYPE,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
        }

    def to_json(self, **dumps_kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **dumps_kwargs)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceDAG":
        dag = cls()
        seen: set = set()
        for raw in data.get("nodes", []):
            nid = str(raw["id"])
            if nid in seen:
                raise ValueError(f"duplicate node id in serialized DAG: {nid!r}")
            seen.add(nid)
            attrs = raw.get("attrs", {})
            if type(attrs) is not dict:
                raise TypeError("serialized node attrs must be a plain JSON object")
            dag.add_node(nid, NodeType(raw["type"]), **attrs)
        for raw in data.get("edges", []):
            attrs = raw.get("attrs", {})
            if type(attrs) is not dict:
                raise TypeError("serialized edge attrs must be a plain JSON object")
            dag.add_edge(
                str(raw["src"]),
                str(raw["dst"]),
                EdgeType(raw["type"]),
                **attrs,
            )
        return dag

    @classmethod
    def from_json(cls, text: str) -> "EvidenceDAG":
        return cls.from_dict(json.loads(text))

    # -- analysis ------------------------------------------------------------
    def find_cycle(self) -> Optional[List[str]]:
        """Return one cycle as a node-id list, or None if acyclic."""
        adj: Dict[str, List[str]] = {nid: [] for nid in self._nodes}
        for e in self._edges:
            if e.src in adj and e.dst in self._nodes:
                adj[e.src].append(e.dst)
        WHITE, GREY, BLACK = 0, 1, 2
        color = {nid: WHITE for nid in self._nodes}
        stack: List[str] = []

        def dfs(node: str) -> Optional[List[str]]:
            color[node] = GREY
            stack.append(node)
            for nxt in adj[node]:
                if color[nxt] == GREY:
                    return stack[stack.index(nxt) :] + [nxt]
                if color[nxt] == WHITE:
                    found = dfs(nxt)
                    if found is not None:
                        return found
            color[node] = BLACK
            stack.pop()
            return None

        for nid in self._nodes:
            if color[nid] == WHITE:
                found = dfs(nid)
                if found is not None:
                    return found
        return None

    def is_acyclic(self) -> bool:
        return self.find_cycle() is None


def validate_evidence_dag(dag: EvidenceDAG) -> List[str]:
    """Structurally validate a DAG. Returns a list of violation strings (empty == valid)."""
    violations: List[str] = []
    ids = {n.node_id for n in dag.nodes}

    for node in dag.nodes:
        if not isinstance(node.node_type, NodeType):
            violations.append(f"node {node.node_id!r}: unknown node type {node.node_type!r}")

    for i, edge in enumerate(dag.edges):
        if not isinstance(edge.edge_type, EdgeType):
            violations.append(f"edge[{i}]: unknown edge type {edge.edge_type!r}")
            continue
        if edge.src not in ids:
            violations.append(f"edge[{i}] ({edge.edge_type.value}): src {edge.src!r} is not a node")
        if edge.dst not in ids:
            violations.append(f"edge[{i}] ({edge.edge_type.value}): dst {edge.dst!r} is not a node")
        if edge.src in ids and edge.dst in ids:
            want_src, want_dst = EDGE_ENDPOINTS[edge.edge_type]
            got_src = dag.node(edge.src).node_type
            got_dst = dag.node(edge.dst).node_type
            if got_src != want_src or got_dst != want_dst:
                violations.append(
                    f"edge[{i}] ({edge.edge_type.value}): endpoint types "
                    f"{got_src.value}->{got_dst.value} violate contract "
                    f"{want_src.value}->{want_dst.value}"
                )

    cycle = dag.find_cycle()
    if cycle is not None:
        violations.append("graph is not acyclic; cycle: " + " -> ".join(cycle))

    return violations


# ---------------------------------------------------------------------------
# JSON Schema (draft-07) for external validation of the serialized form.
# ---------------------------------------------------------------------------

EVIDENCE_DAG_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "EvidenceDAG",
    "type": "object",
    "required": ["record_type", "nodes", "edges"],
    "properties": {
        "record_type": {"const": "evidence_dag"},
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "type"],
                "properties": {
                    "id": {"type": "string"},
                    "type": {"enum": [t.value for t in NodeType]},
                    "attrs": {"type": "object"},
                },
                "additionalProperties": False,
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["src", "dst", "type"],
                "properties": {
                    "src": {"type": "string"},
                    "dst": {"type": "string"},
                    "type": {"enum": [t.value for t in EdgeType]},
                    "attrs": {"type": "object"},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Builder: flat attribution pool -> typed evidence DAG.
# ---------------------------------------------------------------------------


def from_attribution_pool(pool: Mapping[str, Any]) -> EvidenceDAG:
    """Build a typed evidence DAG from a flat attribution pool dict.

    Maps each ``query_attribution_rows`` entry to a QUERY node, the (task, profile)
    it ran under to a CLUSTER node, and the row's ``strategy``/``action`` to an
    ACTUATOR node, wiring QUERY-RETRIEVED_IN->CLUSTER, QUERY-CORRECTED_BY->ACTUATOR
    (only when the row's action denotes a correction), and ACTUATOR-OPERATES_ON->
    CLUSTER. Deterministic: node ids derive from stable row fields, so the same
    pool yields the same graph.
    """
    dag = EvidenceDAG()
    rows = pool.get("query_attribution_rows") or []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        task = row.get("task")
        profile = row.get("profile")
        strategy = row.get("strategy") or row.get("source_artifact") or "unknown"
        action = str(row.get("action") or "").upper()
        qid = row.get("query_id")
        query_node_id = f"q:{strategy}:{qid if qid is not None else idx}"
        cluster_node_id = f"c:{task}:{profile}"
        actuator_node_id = f"a:{strategy}:{action or 'none'}"

        dag.add_node(
            query_node_id,
            NodeType.QUERY,
            task=task,
            profile=profile,
            query_id=qid,
            query_text=row.get("query_text"),
            action=action or None,
            fault_class=row.get("fault_class"),
        )
        dag.add_node(cluster_node_id, NodeType.CLUSTER, task=task, profile=profile)
        dag.add_edge(query_node_id, cluster_node_id, EdgeType.RETRIEVED_IN)

        # A correction-actuator edge only when the action denotes a REAL correction.
        # The pipeline's per-row action vocabulary (canonical interpreter:
        # adaptive_overlay.infer_aggression_level; emitted by
        # antigravity_engine.run_inference) is FAST (no correction) vs the corrections
        # CHELATE / CHELATE_ALWAYS / REFORMULATE. NB "REFORM" is NOT a per-row action —
        # it is only an action_mix COUNT key in build_attribution_pool, so matching it
        # would silently drop every real reformulation/always correction.
        if action in {"CHELATE", "CHELATE_ALWAYS", "REFORMULATE"}:
            dag.add_node(actuator_node_id, NodeType.ACTUATOR, strategy=strategy, action=action)
            dag.add_edge(query_node_id, actuator_node_id, EdgeType.CORRECTED_BY)
            dag.add_edge(actuator_node_id, cluster_node_id, EdgeType.OPERATES_ON)
    return dag
