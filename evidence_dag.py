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
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple


class NodeType(str, Enum):
    QUERY = "query"
    CLUSTER = "cluster"
    ACTUATOR = "actuator"  # correction-actuator: adapter / mask / steering route


class EdgeType(str, Enum):
    RETRIEVED_IN = "retrieved_in"   # QUERY -> CLUSTER
    CORRECTED_BY = "corrected_by"   # QUERY -> ACTUATOR
    OPERATES_ON = "operates_on"     # ACTUATOR -> CLUSTER


# Allowed (source, destination) node types per edge type. This is the typed-graph
# contract; the validator rejects any edge whose endpoints violate it.
EDGE_ENDPOINTS: Dict[EdgeType, Tuple[NodeType, NodeType]] = {
    EdgeType.RETRIEVED_IN: (NodeType.QUERY, NodeType.CLUSTER),
    EdgeType.CORRECTED_BY: (NodeType.QUERY, NodeType.ACTUATOR),
    EdgeType.OPERATES_ON: (NodeType.ACTUATOR, NodeType.CLUSTER),
}


@dataclass(frozen=True)
class EvidenceNode:
    node_id: str
    node_type: NodeType
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.node_id, "type": self.node_type.value, "attrs": dict(self.attrs)}


@dataclass(frozen=True)
class EvidenceEdge:
    src: str
    dst: str
    edge_type: EdgeType
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "type": self.edge_type.value,
            "attrs": dict(self.attrs),
        }


class EvidenceDAG:
    """A typed directed (intended-acyclic) evidence graph."""

    RECORD_TYPE = "evidence_dag"

    def __init__(self) -> None:
        self._nodes: Dict[str, EvidenceNode] = {}
        self._edges: List[EvidenceEdge] = []

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

    @property
    def nodes(self) -> List[EvidenceNode]:
        return list(self._nodes.values())

    @property
    def edges(self) -> List[EvidenceEdge]:
        return list(self._edges)

    def node(self, node_id: str) -> Optional[EvidenceNode]:
        return self._nodes.get(str(node_id))

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
            dag.add_node(nid, NodeType(raw["type"]), **dict(raw.get("attrs", {})))
        for raw in data.get("edges", []):
            dag.add_edge(
                str(raw["src"]), str(raw["dst"]), EdgeType(raw["type"]),
                **dict(raw.get("attrs", {})),
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
                    return stack[stack.index(nxt):] + [nxt]
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

        dag.add_node(query_node_id, NodeType.QUERY,
                     task=task, profile=profile, action=action or None,
                     fault_class=row.get("fault_class"))
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
