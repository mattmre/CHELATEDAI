"""Tiny exact diagnostics for the PRW-G1 graph-fiber hypothesis.

Each node has one unary score for every phase in ``Z_p``.  The analyzer
enumerates one joint-assignment stream for three full-space scorers and one
explicit restricted comparator:

* independent: the sum of unary node scores;
* graph coupled: the unary sum minus one penalty per violated labeled edge;
* shuffled control: the same graph, degree sequence, edge count, label
  multiset, observation, and candidate assignments, with a deterministic
  non-trivial permutation of edge labels when one exists.
* global-phase comparator: when the original graph is connected and
  cycle-consistent, unary scoring restricted to
  ``phase[i] = z + offset[i] (mod p)`` for one shared ``z``.

This is ordinary finite-group synchronization.  It does not construct a
physical three-dimensional lattice, model gravity or resonance, establish
novelty, or provide retrieval evidence.  Exact enumeration is deliberately
restricted to tiny states.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations, product
import math
from numbers import Integral, Rational, Real
import time
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


MIB = 1024 * 1024
ALLOWED_MODULI = (7, 11, 31)

# These immutable ceilings do not exceed the existing finite-analysis caps in
# prime_ring_intersection.py.  Defaults below are intentionally much smaller.
HARD_MAX_ESTIMATED_BYTES = 512 * MIB
HARD_MAX_SECONDS = 120.0
HARD_MAX_WORK_UNITS = 50_000_000
HARD_MAX_HYPOTHESES = 1024
HARD_MAX_NODES = 12
HARD_MAX_EDGES = 18
HARD_MAX_RATIONAL_BITS = 256
HARD_MAX_LABEL_PERMUTATIONS = 1024

DEFAULT_MAX_ESTIMATED_BYTES = 8 * MIB
DEFAULT_MAX_SECONDS = 5.0
DEFAULT_MAX_WORK_UNITS = 1_000_000
DEFAULT_MAX_LABEL_PERMUTATIONS = 64

PROTOCOL_ID = "CHELATEDAI-PRW-G1-v0.2"
EVIDENCE_MODE = "NON_CONFIRMATORY_METHOD_DEV"

KILL_CRITERIA = (
    "candidate_or_observation_budget_mismatch",
    "label_shuffled_control_not_distinct",
    "planted_assignment_violates_declared_edges",
    "graph_coupling_not_better_than_independent_recovery",
    "graph_coupling_not_better_than_shuffled_recovery",
    "label_specific_planted_margin_gain_nonpositive",
    "connected_consistent_graph_is_one_global_phase_up_to_fixed_offsets",
)

LIMITATIONS = (
    "Consumes frozen unary phase scores; it does not construct or corrupt ring carriers.",
    "One tiny synthetic instance cannot estimate recall or false-unlock rates.",
    "The negative control is one deterministic structure-matched edge-label permutation, not a graph ensemble.",
    "Arbitrary label permutations can change cycle consistency; this analyzer rejects rather than compares such controls.",
    "Matching covers observations, assignments, topology, degrees, edge count, and label multiset; stored bytes, channel uses, and latency are not measured.",
    "Exact enumeration is limited to at most 1024 joint assignments.",
    "A connected consistent zero-violation manifold is one global phase up to fixed offsets; finite-penalty violating assignments are not.",
    "Tie-broken selected assignments are descriptive; recovery effects and gates require unique exact recovery.",
    "No non-cyclic product-key decoder is implemented in this slice.",
    "The global-phase comparator has p candidates rather than p**nodes; its exact rank is reported with that denominator and is not budget-matched to full-space rank.",
    "A soft graph penalty can prefer constraint-violating assignments and then differ from the exact zero-violation global-phase family.",
    "The deadline is cooperative and checked between bounded Python operations; it cannot preempt one hostile object method.",
    "Estimated bytes exclude arbitrary caller-container overhead and do not measure process peak memory.",
    "No physical 3D, gravity, resonance, production, or novelty claim is made.",
)


class GraphFiberValidationError(ValueError):
    """Raised when a PRW-G1 input contract is malformed."""


class GraphFiberResourceError(RuntimeError):
    """Raised before or during work when a resource boundary is exceeded."""


def _plain_int(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise GraphFiberValidationError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise GraphFiberValidationError(f"{name} must be >= {minimum}")
    return result


def _positive_seconds(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise GraphFiberValidationError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise GraphFiberValidationError(f"{name} must be a finite positive number")
    return result


def _bounded_fraction(value: object, name: str) -> Fraction:
    if isinstance(value, bool):
        raise GraphFiberValidationError(f"{name} must be a finite real number")
    if isinstance(value, Rational):
        result = Fraction(value)
    elif isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise GraphFiberValidationError(f"{name} must be finite")
        result = Fraction(str(numeric))
    else:
        raise GraphFiberValidationError(f"{name} must be a finite real number")
    if (
        abs(result.numerator).bit_length() > HARD_MAX_RATIONAL_BITS
        or result.denominator.bit_length() > HARD_MAX_RATIONAL_BITS
    ):
        raise GraphFiberValidationError(
            f"{name} exceeds the {HARD_MAX_RATIONAL_BITS}-bit rational limit"
        )
    return result


@dataclass(frozen=True, order=True)
class FiberEdge:
    """One canonically oriented edge ``source < target`` with phase in ``Z_p``."""

    source: int
    target: int
    phase: int


@dataclass(frozen=True)
class GraphFiberBudget:
    """Immutable small-state budget, bounded by repository-wide hard caps."""

    max_estimated_bytes: int = DEFAULT_MAX_ESTIMATED_BYTES
    max_seconds: float = DEFAULT_MAX_SECONDS
    max_work_units: int = DEFAULT_MAX_WORK_UNITS
    max_hypotheses: int = HARD_MAX_HYPOTHESES
    max_nodes: int = HARD_MAX_NODES
    max_edges: int = HARD_MAX_EDGES
    max_label_permutations: int = DEFAULT_MAX_LABEL_PERMUTATIONS

    def __post_init__(self) -> None:
        integer_limits = (
            ("max_estimated_bytes", self.max_estimated_bytes, HARD_MAX_ESTIMATED_BYTES),
            ("max_work_units", self.max_work_units, HARD_MAX_WORK_UNITS),
            ("max_hypotheses", self.max_hypotheses, HARD_MAX_HYPOTHESES),
            ("max_nodes", self.max_nodes, HARD_MAX_NODES),
            ("max_edges", self.max_edges, HARD_MAX_EDGES),
            (
                "max_label_permutations",
                self.max_label_permutations,
                HARD_MAX_LABEL_PERMUTATIONS,
            ),
        )
        for name, value, hard_maximum in integer_limits:
            checked = _plain_int(value, name, 1)
            if checked > hard_maximum:
                raise GraphFiberValidationError(
                    f"{name} cannot exceed immutable hard cap {hard_maximum}"
                )
            object.__setattr__(self, name, checked)
        seconds = _positive_seconds(self.max_seconds, "max_seconds")
        if seconds > HARD_MAX_SECONDS:
            raise GraphFiberValidationError(
                f"max_seconds cannot exceed immutable hard cap {HARD_MAX_SECONDS:g}"
            )
        object.__setattr__(self, "max_seconds", seconds)


@dataclass(frozen=True)
class GraphStructure:
    """Exact structural diagnostics for one labeled graph."""

    degrees: Tuple[int, ...]
    component_count: int
    cycle_rank: int
    cycle_consistent: bool
    satisfying_assignment_count: int
    connected: bool
    collapses_to_one_global_phase: bool
    all_edge_labels_zero: bool


@dataclass(frozen=True)
class DecoderOutcome:
    """Exact outcome and planted-assignment diagnostics for one scorer."""

    assignment: Tuple[int, ...]
    score: Fraction
    tie_count: int
    selected_assignment_matches_planted: bool
    unique_exact_recovery: bool
    planted_score: Fraction
    best_incorrect_score: Fraction
    planted_margin: Fraction
    planted_rank_min: int
    planted_rank_max: int
    planted_is_top_tied: bool


@dataclass(frozen=True)
class ResourceEstimate:
    """Conservative streaming-enumeration estimate."""

    candidate_assignments: int
    full_space_hypotheses_evaluated_per_decoder: int
    full_space_decoder_hypothesis_evaluations: int
    global_phase_comparator_hypotheses_upper_bound: int
    total_decoder_hypothesis_evaluations: int
    total_decoder_hypothesis_evaluations_is_upper_bound: bool
    label_permutation_candidates_upper_bound: int
    estimated_peak_bytes: int
    estimated_work_units: int
    component_bytes: Tuple[Tuple[str, int], ...]
    max_estimated_bytes: int
    max_work_units: int
    max_hypotheses: int
    max_label_permutations: int
    max_seconds: float
    streaming_assignments_not_materialized: bool
    caller_owned_inputs_included: bool
    measured_process_peak: bool


@dataclass(frozen=True)
class GraphFiberAnalysis:
    """Complete non-promotional result for one bounded PRW-G1 instance."""

    protocol_id: str
    evidence_mode: str
    modulus: int
    node_count: int
    coupling: Fraction
    planted_assignment: Tuple[int, ...]
    graph_edges: Tuple[FiberEdge, ...]
    shuffled_edges: Tuple[FiberEdge, ...]
    shuffled_control_status: str
    graph_structure: GraphStructure
    shuffled_structure: GraphStructure
    independent: DecoderOutcome
    graph_coupled: DecoderOutcome
    shuffled_graph: DecoderOutcome
    global_phase_offsets: Optional[Tuple[int, ...]]
    global_phase_comparator_status: str
    zero_violation_manifold_equals_global_phase_comparator: bool
    global_phase_comparator: Optional[DecoderOutcome]
    global_phase_hypotheses_evaluated: int
    resource_guard: ResourceEstimate
    comparison_contract: Mapping[str, Any]
    paired_effects: Mapping[str, Any]
    kill_criteria_triggered: Tuple[str, ...]
    limitations: Tuple[str, ...]
    hypothesis_status: str
    promotion_eligible: bool
    physical_claim: bool
    novelty_claim: bool


@dataclass
class _OutcomeAccumulator:
    planted: Tuple[int, ...]
    planted_score: Fraction
    best_assignment: Optional[Tuple[int, ...]] = None
    best_score: Optional[Fraction] = None
    tie_count: int = 0
    scores_above_planted: int = 0
    scores_equal_planted: int = 0
    best_incorrect_score: Optional[Fraction] = None

    def observe(self, assignment: Tuple[int, ...], score: Fraction) -> None:
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_assignment = assignment
            self.tie_count = 1
        elif score == self.best_score:
            self.tie_count += 1
            if self.best_assignment is None or assignment < self.best_assignment:
                self.best_assignment = assignment
        if score > self.planted_score:
            self.scores_above_planted += 1
        elif score == self.planted_score:
            self.scores_equal_planted += 1
        if assignment != self.planted and (
            self.best_incorrect_score is None or score > self.best_incorrect_score
        ):
            self.best_incorrect_score = score

    def finish(self) -> DecoderOutcome:
        if (
            self.best_assignment is None
            or self.best_score is None
            or self.best_incorrect_score is None
            or self.scores_equal_planted < 1
        ):
            raise GraphFiberValidationError("internal exact enumeration was incomplete")
        selected_match = self.best_assignment == self.planted
        return DecoderOutcome(
            assignment=self.best_assignment,
            score=self.best_score,
            tie_count=self.tie_count,
            selected_assignment_matches_planted=selected_match,
            unique_exact_recovery=selected_match and self.tie_count == 1,
            planted_score=self.planted_score,
            best_incorrect_score=self.best_incorrect_score,
            planted_margin=self.planted_score - self.best_incorrect_score,
            planted_rank_min=1 + self.scores_above_planted,
            planted_rank_max=(
                self.scores_above_planted + self.scores_equal_planted
            ),
            planted_is_top_tied=self.scores_above_planted == 0,
        )


def _validate_shape(
    unary_scores: object,
    edges: object,
    modulus: int,
    budget: GraphFiberBudget,
) -> Tuple[int, int, int]:
    if isinstance(unary_scores, (str, bytes)) or not isinstance(
        unary_scores, Sequence
    ):
        raise GraphFiberValidationError("unary_scores must be a sequence of rows")
    node_count = len(unary_scores)
    if node_count < 2:
        raise GraphFiberValidationError("unary_scores must contain at least two nodes")
    if node_count > budget.max_nodes:
        raise GraphFiberResourceError(
            f"node count exceeds budget: {node_count} > {budget.max_nodes}"
        )
    for node, row in enumerate(unary_scores):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise GraphFiberValidationError(
                f"unary_scores[{node}] must be a sequence"
            )
        if len(row) != modulus:
            raise GraphFiberValidationError(
                f"unary_scores[{node}] must contain exactly {modulus} phases"
            )
    if isinstance(edges, (str, bytes)) or not isinstance(edges, Sequence):
        raise GraphFiberValidationError("edges must be a sequence")
    edge_count = len(edges)
    if edge_count < 1:
        raise GraphFiberValidationError("at least one graph edge is required")
    if edge_count > budget.max_edges:
        raise GraphFiberResourceError(
            f"edge count exceeds budget: {edge_count} > {budget.max_edges}"
        )
    if edge_count > node_count * (node_count - 1) // 2:
        raise GraphFiberValidationError("too many edges for a simple graph")
    hypothesis_count = modulus**node_count
    if hypothesis_count > budget.max_hypotheses:
        raise GraphFiberResourceError(
            "joint assignment count exceeds hypothesis budget: "
            f"{hypothesis_count} > {budget.max_hypotheses}"
        )
    return node_count, edge_count, hypothesis_count


def _resource_estimate(
    *,
    modulus: int,
    node_count: int,
    edge_count: int,
    hypothesis_count: int,
    budget: GraphFiberBudget,
) -> ResourceEstimate:
    permutation_candidates = math.factorial(edge_count)
    if permutation_candidates > budget.max_label_permutations:
        raise GraphFiberResourceError(
            "edge-label permutation count exceeds control-search budget: "
            f"{permutation_candidates} > {budget.max_label_permutations}"
        )
    components = (
        (
            "normalized_unary_and_visible_caller_value_allowance",
            node_count * modulus * 2 * 256,
        ),
        (
            "normalized_edges_and_visible_caller_value_allowance",
            edge_count * 3 * 256,
        ),
        (
            "normalized_planted_and_visible_caller_value_allowance",
            node_count * 2 * 64,
        ),
        ("two_graph_adjacency_views", (node_count + 2 * edge_count) * 2 * 192),
        (
            "label_permutation_search_workspace",
            4096 + permutation_candidates * (256 + edge_count * 64),
        ),
        ("current_streamed_assignment", node_count * 64),
        ("four_decoder_accumulators_upper_bound", 4 * (2048 + node_count * 64)),
        ("immutable_result_and_diagnostics", 32 * 1024),
    )
    estimated_bytes = sum(value for _name, value in components)
    estimated_work = (
        hypothesis_count * (3 * node_count + 20 * edge_count + 30)
        + node_count * modulus
        + edge_count * 20
        + permutation_candidates * (10 * node_count + 40 * edge_count + 20)
        + hypothesis_count * (node_count + 2)
        + modulus * 20
    )
    if estimated_bytes > budget.max_estimated_bytes:
        raise GraphFiberResourceError(
            "estimated peak exceeds byte budget: "
            f"{estimated_bytes} > {budget.max_estimated_bytes}"
        )
    if estimated_work > budget.max_work_units:
        raise GraphFiberResourceError(
            "estimated work exceeds budget: "
            f"{estimated_work} > {budget.max_work_units}"
        )
    return ResourceEstimate(
        candidate_assignments=hypothesis_count,
        full_space_hypotheses_evaluated_per_decoder=hypothesis_count,
        full_space_decoder_hypothesis_evaluations=3 * hypothesis_count,
        global_phase_comparator_hypotheses_upper_bound=modulus,
        total_decoder_hypothesis_evaluations=(
            3 * hypothesis_count + modulus
        ),
        total_decoder_hypothesis_evaluations_is_upper_bound=True,
        label_permutation_candidates_upper_bound=permutation_candidates,
        estimated_peak_bytes=estimated_bytes,
        estimated_work_units=estimated_work,
        component_bytes=components,
        max_estimated_bytes=budget.max_estimated_bytes,
        max_work_units=budget.max_work_units,
        max_hypotheses=budget.max_hypotheses,
        max_label_permutations=budget.max_label_permutations,
        max_seconds=budget.max_seconds,
        streaming_assignments_not_materialized=True,
        caller_owned_inputs_included=False,
        measured_process_peak=False,
    )


def _normalize_unary_scores(
    unary_scores: Sequence[Sequence[object]],
) -> Tuple[Tuple[Fraction, ...], ...]:
    return tuple(
        tuple(
            _bounded_fraction(value, f"unary_scores[{node}][{phase}]")
            for phase, value in enumerate(row)
        )
        for node, row in enumerate(unary_scores)
    )


def _normalize_edges(
    edges: Sequence[object],
    *,
    node_count: int,
    modulus: int,
) -> Tuple[FiberEdge, ...]:
    normalized = []
    seen = set()
    for index, raw_edge in enumerate(edges):
        if isinstance(raw_edge, FiberEdge):
            source_raw, target_raw, phase_raw = (
                raw_edge.source,
                raw_edge.target,
                raw_edge.phase,
            )
        elif (
            isinstance(raw_edge, Sequence)
            and not isinstance(raw_edge, (str, bytes))
            and len(raw_edge) == 3
        ):
            source_raw, target_raw, phase_raw = raw_edge
        else:
            raise GraphFiberValidationError(
                f"edges[{index}] must be FiberEdge or a length-3 sequence"
            )
        source = _plain_int(source_raw, f"edges[{index}].source", 0)
        target = _plain_int(target_raw, f"edges[{index}].target", 0)
        phase = _plain_int(phase_raw, f"edges[{index}].phase", 0)
        if source >= node_count or target >= node_count:
            raise GraphFiberValidationError(
                f"edges[{index}] endpoint is outside the node range"
            )
        if source == target:
            raise GraphFiberValidationError("self edges are not allowed")
        if phase >= modulus:
            raise GraphFiberValidationError(
                f"edges[{index}].phase must be in [0, {modulus})"
            )
        if source > target:
            source, target = target, source
            phase = (-phase) % modulus
        key = (source, target)
        if key in seen:
            raise GraphFiberValidationError("duplicate undirected edges are not allowed")
        seen.add(key)
        normalized.append(FiberEdge(source, target, phase))
    return tuple(sorted(normalized))


def _normalize_assignment(
    planted_assignment: object,
    *,
    node_count: int,
    modulus: int,
) -> Tuple[int, ...]:
    if isinstance(planted_assignment, (str, bytes)) or not isinstance(
        planted_assignment, Sequence
    ):
        raise GraphFiberValidationError("planted_assignment must be a sequence")
    if len(planted_assignment) != node_count:
        raise GraphFiberValidationError(
            "planted_assignment must contain one phase per node"
        )
    values = tuple(
        _plain_int(value, f"planted_assignment[{node}]", 0)
        for node, value in enumerate(planted_assignment)
    )
    if any(value >= modulus for value in values):
        raise GraphFiberValidationError(
            f"planted_assignment phases must be in [0, {modulus})"
        )
    return values


def _shuffle_edge_labels(
    edges: Tuple[FiberEdge, ...],
    *,
    node_count: int,
    modulus: int,
    original_structure: GraphStructure,
    deadline: float,
    max_candidates: int,
) -> Tuple[Tuple[FiberEdge, ...], GraphStructure, str]:
    labels = tuple(edge.phase for edge in edges)
    seen = set()
    examined = 0
    for candidate_labels in permutations(labels):
        examined += 1
        if examined > max_candidates:
            raise GraphFiberResourceError(
                "edge-label control search exceeded its preflight candidate cap"
            )
        _check_deadline(deadline)
        if candidate_labels in seen:
            continue
        seen.add(candidate_labels)
        if candidate_labels == labels:
            continue
        candidate_edges = tuple(
            FiberEdge(edge.source, edge.target, candidate_labels[index])
            for index, edge in enumerate(edges)
        )
        candidate_structure = _graph_structure(
            node_count,
            candidate_edges,
            modulus,
        )
        if candidate_structure == original_structure:
            return (
                candidate_edges,
                candidate_structure,
                "COMPLETED_EXACT_STRUCTURE_MATCHED_LABEL_PERMUTATION",
            )
    return (
        edges,
        original_structure,
        "STRUCTURALLY_UNAVAILABLE_NO_DISTINCT_STRUCTURE_MATCHED_LABEL_PERMUTATION",
    )


def _graph_structure(
    node_count: int,
    edges: Tuple[FiberEdge, ...],
    modulus: int,
) -> GraphStructure:
    adjacency = [[] for _ in range(node_count)]
    degrees = [0] * node_count
    for edge in edges:
        adjacency[edge.source].append((edge.target, edge.phase))
        adjacency[edge.target].append((edge.source, (-edge.phase) % modulus))
        degrees[edge.source] += 1
        degrees[edge.target] += 1

    potentials: list = [None] * node_count
    components = 0
    consistent = True
    for root in range(node_count):
        if potentials[root] is not None:
            continue
        components += 1
        potentials[root] = 0
        stack = [root]
        while stack:
            node = stack.pop()
            for neighbor, delta in adjacency[node]:
                expected = (potentials[node] + delta) % modulus
                if potentials[neighbor] is None:
                    potentials[neighbor] = expected
                    stack.append(neighbor)
                elif potentials[neighbor] != expected:
                    consistent = False
    cycle_rank = len(edges) - node_count + components
    connected = components == 1
    return GraphStructure(
        degrees=tuple(degrees),
        component_count=components,
        cycle_rank=cycle_rank,
        cycle_consistent=consistent,
        satisfying_assignment_count=(modulus**components if consistent else 0),
        connected=connected,
        collapses_to_one_global_phase=(
            connected and consistent and bool(edges)
        ),
        all_edge_labels_zero=all(edge.phase == 0 for edge in edges),
    )


def _global_phase_offsets(
    node_count: int,
    edges: Tuple[FiberEdge, ...],
    modulus: int,
    structure: GraphStructure,
) -> Optional[Tuple[int, ...]]:
    """Return canonical offsets for the connected consistent edge manifold.

    With ``offset[0] = 0``, every zero-violation assignment is exactly
    ``phase[i] = z + offset[i] (mod p)`` for one ``z in Z_p``.  Conversely,
    each of those ``p`` assignments satisfies every edge.  A disconnected or
    contradictory graph has no one-global-phase representation and returns
    ``None``.
    """

    if not structure.connected or not structure.cycle_consistent:
        return None
    adjacency = [[] for _ in range(node_count)]
    for edge in edges:
        adjacency[edge.source].append((edge.target, edge.phase))
        adjacency[edge.target].append((edge.source, (-edge.phase) % modulus))
    offsets: list = [None] * node_count
    offsets[0] = 0
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor, delta in adjacency[node]:
            expected = (offsets[node] + delta) % modulus
            if offsets[neighbor] is None:
                offsets[neighbor] = expected
                stack.append(neighbor)
            elif offsets[neighbor] != expected:
                raise GraphFiberValidationError(
                    "graph structure and canonical-offset derivation disagree"
                )
    if any(value is None for value in offsets):
        raise GraphFiberValidationError(
            "connected graph did not yield one offset per node"
        )
    return tuple(int(value) for value in offsets)


def _matches_global_phase_family(
    assignment: Tuple[int, ...],
    offsets: Tuple[int, ...],
    modulus: int,
) -> bool:
    global_phase = assignment[0]
    return all(
        phase == (global_phase + offsets[node]) % modulus
        for node, phase in enumerate(assignment)
    )


def _violation_count(
    assignment: Tuple[int, ...],
    edges: Tuple[FiberEdge, ...],
    modulus: int,
) -> int:
    return sum(
        (
            assignment[edge.target]
            - assignment[edge.source]
            - edge.phase
        )
        % modulus
        != 0
        for edge in edges
    )


def _unary_score(
    assignment: Tuple[int, ...],
    unary_scores: Tuple[Tuple[Fraction, ...], ...],
) -> Fraction:
    return sum(
        (unary_scores[node][phase] for node, phase in enumerate(assignment)),
        Fraction(0),
    )


def _check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise GraphFiberResourceError(
            "exact graph-fiber analysis exceeded its wall-clock deadline"
        )


def analyze_graph_fibers(
    unary_scores: Sequence[Sequence[object]],
    edges: Sequence[object],
    planted_assignment: Sequence[object],
    *,
    modulus: int,
    coupling: object = 1,
    budget: GraphFiberBudget = GraphFiberBudget(),
) -> GraphFiberAnalysis:
    """Exactly compare three phase decoders on one frozen tiny observation.

    The graph penalty is the indicator ``rho(a) = 1[a != 0]``.  Edge
    ``(u, v, g)`` is satisfied exactly when
    ``phase[v] - phase[u] == g (mod p)``.

    For a connected cycle-consistent graph, canonical offsets ``o_i`` satisfy
    every edge with ``o_0 = 0``.  The explicit comparator searches

    ``C = {x(z): x_i(z) = z + o_i (mod p), z in Z_p}``

    and scores ``x(z)`` with the same unary observation.  Thus ``C`` is
    exactly the graph's zero-violation manifold, while the finite-penalty
    graph decoder may still select assignments outside ``C``.  Connectedness
    supplies a root path to every node, cycle consistency makes the resulting
    offsets path-independent, and subtracting ``x_0`` proves both inclusions.
    """

    if not isinstance(budget, GraphFiberBudget):
        raise GraphFiberValidationError("budget must be GraphFiberBudget")
    started = time.monotonic()
    deadline = started + budget.max_seconds
    prime = _plain_int(modulus, "modulus", 2)
    if prime not in ALLOWED_MODULI:
        raise GraphFiberValidationError(
            f"modulus must be one of the bounded set {ALLOWED_MODULI}"
        )
    _check_deadline(deadline)
    node_count, edge_count, hypothesis_count = _validate_shape(
        unary_scores,
        edges,
        prime,
        budget,
    )
    _check_deadline(deadline)
    resource = _resource_estimate(
        modulus=prime,
        node_count=node_count,
        edge_count=edge_count,
        hypothesis_count=hypothesis_count,
        budget=budget,
    )
    _check_deadline(deadline)
    weight = _bounded_fraction(coupling, "coupling")
    if weight <= 0:
        raise GraphFiberValidationError("coupling must be positive")
    _check_deadline(deadline)
    normalized_unary = _normalize_unary_scores(unary_scores)
    _check_deadline(deadline)
    normalized_edges = _normalize_edges(
        edges,
        node_count=node_count,
        modulus=prime,
    )
    _check_deadline(deadline)
    planted = _normalize_assignment(
        planted_assignment,
        node_count=node_count,
        modulus=prime,
    )
    _check_deadline(deadline)
    original_structure = _graph_structure(
        node_count,
        normalized_edges,
        prime,
    )
    global_phase_offsets = _global_phase_offsets(
        node_count,
        normalized_edges,
        prime,
        original_structure,
    )
    zero_violation_manifold_equals_comparator = (
        global_phase_offsets is not None
        and original_structure.satisfying_assignment_count == prime
    )
    if (
        global_phase_offsets is not None
        and not zero_violation_manifold_equals_comparator
    ):
        raise GraphFiberValidationError(
            "global-phase family size disagrees with graph structure"
        )
    shuffled_edges, shuffled_structure, shuffled_status = _shuffle_edge_labels(
        normalized_edges,
        node_count=node_count,
        modulus=prime,
        original_structure=original_structure,
        deadline=deadline,
        max_candidates=resource.label_permutation_candidates_upper_bound,
    )
    _check_deadline(deadline)

    planted_unary = _unary_score(planted, normalized_unary)
    graph_planted_violations = _violation_count(
        planted,
        normalized_edges,
        prime,
    )
    planted_graph = planted_unary - weight * graph_planted_violations
    planted_shuffled = planted_unary - weight * _violation_count(
        planted,
        shuffled_edges,
        prime,
    )
    if not original_structure.connected:
        global_phase_status = (
            "STRUCTURALLY_UNAVAILABLE_REQUIRES_CONNECTED_GRAPH"
        )
    elif not original_structure.cycle_consistent:
        global_phase_status = (
            "STRUCTURALLY_UNAVAILABLE_CONTRADICTORY_CYCLE"
        )
    elif graph_planted_violations:
        global_phase_status = (
            "REFUSED_PLANTED_ASSIGNMENT_OUTSIDE_ZERO_VIOLATION_MANIFOLD"
        )
    else:
        global_phase_status = (
            "COMPLETED_EXACT_ONE_GLOBAL_PHASE_PLUS_FIXED_OFFSETS"
        )
    accumulators = (
        _OutcomeAccumulator(planted, planted_unary),
        _OutcomeAccumulator(planted, planted_graph),
        _OutcomeAccumulator(planted, planted_shuffled),
    )
    global_phase_accumulator = (
        _OutcomeAccumulator(planted, planted_unary)
        if global_phase_status
        == "COMPLETED_EXACT_ONE_GLOBAL_PHASE_PLUS_FIXED_OFFSETS"
        else None
    )
    evaluated = 0
    global_phase_evaluated = 0
    for index, raw_assignment in enumerate(
        product(range(prime), repeat=node_count)
    ):
        if index % 64 == 0:
            _check_deadline(deadline)
        assignment = tuple(raw_assignment)
        unary = _unary_score(assignment, normalized_unary)
        graph_score = unary - weight * _violation_count(
            assignment,
            normalized_edges,
            prime,
        )
        shuffled_score = unary - weight * _violation_count(
            assignment,
            shuffled_edges,
            prime,
        )
        for accumulator, score in zip(
            accumulators,
            (unary, graph_score, shuffled_score),
        ):
            accumulator.observe(assignment, score)
        if (
            global_phase_accumulator is not None
            and global_phase_offsets is not None
            and _matches_global_phase_family(
                assignment,
                global_phase_offsets,
                prime,
            )
        ):
            global_phase_accumulator.observe(assignment, unary)
            global_phase_evaluated += 1
        evaluated += 1
    _check_deadline(deadline)
    if evaluated != hypothesis_count:
        raise GraphFiberValidationError(
            "streamed hypothesis count did not match the preflight contract"
        )

    independent, graph_coupled, shuffled_graph = (
        accumulator.finish() for accumulator in accumulators
    )
    if global_phase_accumulator is not None and global_phase_evaluated != prime:
        raise GraphFiberValidationError(
            "global-phase comparator did not evaluate exactly p hypotheses"
        )
    global_phase_comparator = (
        global_phase_accumulator.finish()
        if global_phase_accumulator is not None
        else None
    )
    selected_graph_violations = _violation_count(
        graph_coupled.assignment,
        normalized_edges,
        prime,
    )
    control_distinct = shuffled_edges != normalized_edges
    independent_unique = int(independent.unique_exact_recovery)
    graph_unique = int(graph_coupled.unique_exact_recovery)
    shuffled_unique = int(shuffled_graph.unique_exact_recovery)
    global_phase_unique = (
        int(global_phase_comparator.unique_exact_recovery)
        if global_phase_comparator is not None
        else None
    )
    selected_graph_within_global_phase_family = (
        _matches_global_phase_family(
            graph_coupled.assignment,
            global_phase_offsets,
            prime,
        )
        if global_phase_offsets is not None
        else None
    )
    paired_effects: Dict[str, Any] = {
        "graph_minus_independent_unique_exact_recovery": (
            graph_unique - independent_unique
        ),
        "graph_minus_shuffled_unique_exact_recovery": (
            graph_unique - shuffled_unique
        ),
        "graph_minus_independent_planted_margin": (
            graph_coupled.planted_margin - independent.planted_margin
        ),
        "graph_minus_shuffled_planted_margin": (
            graph_coupled.planted_margin - shuffled_graph.planted_margin
        ),
        "label_specific_coupling_interaction_on_unique_exact_recovery": (
            (graph_unique - independent_unique)
            - (shuffled_unique - independent_unique)
        ),
        "label_specific_coupling_interaction_on_planted_margin": (
            (graph_coupled.planted_margin - independent.planted_margin)
            - (shuffled_graph.planted_margin - independent.planted_margin)
        ),
        "graph_minus_global_phase_unique_exact_recovery": (
            graph_unique - global_phase_unique
            if global_phase_unique is not None
            else None
        ),
        "graph_minus_global_phase_planted_margin": (
            graph_coupled.planted_margin
            - global_phase_comparator.planted_margin
            if global_phase_comparator is not None
            else None
        ),
        "graph_minus_global_phase_tie_count": (
            graph_coupled.tie_count - global_phase_comparator.tie_count
            if global_phase_comparator is not None
            else None
        ),
        "graph_minus_global_phase_planted_rank_min": (
            graph_coupled.planted_rank_min
            - global_phase_comparator.planted_rank_min
            if global_phase_comparator is not None
            else None
        ),
        "graph_minus_global_phase_planted_rank_max": (
            graph_coupled.planted_rank_max
            - global_phase_comparator.planted_rank_max
            if global_phase_comparator is not None
            else None
        ),
        "soft_graph_outcome_equals_global_phase_comparator": (
            graph_coupled == global_phase_comparator
            if global_phase_comparator is not None
            else None
        ),
        "selected_graph_within_global_phase_family": (
            selected_graph_within_global_phase_family
        ),
        "original_planted_edge_violations": graph_planted_violations,
        "selected_graph_edge_violations": selected_graph_violations,
        "shuffled_planted_edge_violations": _violation_count(
            planted,
            shuffled_edges,
            prime,
        ),
    }

    triggered = []
    if not control_distinct:
        triggered.append("label_shuffled_control_not_distinct")
    if graph_planted_violations:
        triggered.append("planted_assignment_violates_declared_edges")
    if graph_unique <= independent_unique:
        triggered.append("graph_coupling_not_better_than_independent_recovery")
    if graph_unique <= shuffled_unique:
        triggered.append("graph_coupling_not_better_than_shuffled_recovery")
    if paired_effects["graph_minus_shuffled_planted_margin"] <= 0:
        triggered.append("label_specific_planted_margin_gain_nonpositive")
    if original_structure.collapses_to_one_global_phase:
        triggered.append(
            "connected_consistent_graph_is_one_global_phase_up_to_fixed_offsets"
        )

    comparison_contract: Dict[str, Any] = {
        "same_unary_observation": True,
        "same_assignment_order": True,
        "full_space_hypotheses_per_decoder": hypothesis_count,
        "full_space_decoders_same_hypothesis_budget": True,
        "global_phase_comparator_uses_same_assignment_stream": (
            global_phase_comparator is not None
        ),
        "global_phase_comparator_hypotheses_evaluated": (
            global_phase_evaluated
        ),
        "global_phase_comparator_expected_hypotheses": (
            prime if global_phase_comparator is not None else 0
        ),
        "global_phase_comparator_search_accounting_matches": (
            global_phase_evaluated
            == (prime if global_phase_comparator is not None else 0)
        ),
        "graph_and_global_phase_same_hypothesis_count": (
            hypothesis_count == global_phase_evaluated
            if global_phase_comparator is not None
            else None
        ),
        "graph_rank_denominator": hypothesis_count,
        "global_phase_rank_denominator": (
            global_phase_evaluated
            if global_phase_comparator is not None
            else None
        ),
        "zero_violation_manifold_equals_global_phase_comparator": (
            zero_violation_manifold_equals_comparator
        ),
        "global_phase_comparator_completed_when_comparable": (
            global_phase_comparator is not None
            if (
                zero_violation_manifold_equals_comparator
                and graph_planted_violations == 0
            )
            else True
        ),
        "graph_and_control_same_endpoints": tuple(
            (edge.source, edge.target) for edge in normalized_edges
        )
        == tuple((edge.source, edge.target) for edge in shuffled_edges),
        "graph_and_control_same_degree_sequence": (
            original_structure.degrees == shuffled_structure.degrees
        ),
        "graph_and_control_same_edge_count": (
            len(normalized_edges) == len(shuffled_edges)
        ),
        "graph_and_control_same_label_multiset": (
            sorted(edge.phase for edge in normalized_edges)
            == sorted(edge.phase for edge in shuffled_edges)
        ),
        "graph_and_control_same_cycle_consistency": (
            original_structure.cycle_consistent
            == shuffled_structure.cycle_consistent
        ),
        "graph_and_control_same_satisfying_assignment_count": (
            original_structure.satisfying_assignment_count
            == shuffled_structure.satisfying_assignment_count
        ),
        "negative_control_changes_only_label_placement": (
            control_distinct
            and original_structure.cycle_consistent
            == shuffled_structure.cycle_consistent
            and original_structure.satisfying_assignment_count
            == shuffled_structure.satisfying_assignment_count
        ),
        "matched_stored_bytes": False,
        "matched_channel_uses": False,
        "latency_measured": False,
        "false_unlock_rate_measured": False,
    }
    if not all(
        (
            comparison_contract["same_unary_observation"],
            comparison_contract["same_assignment_order"],
            comparison_contract[
                "full_space_decoders_same_hypothesis_budget"
            ],
            comparison_contract["graph_and_control_same_endpoints"],
            comparison_contract["graph_and_control_same_degree_sequence"],
            comparison_contract["graph_and_control_same_edge_count"],
            comparison_contract["graph_and_control_same_label_multiset"],
            comparison_contract["graph_and_control_same_cycle_consistency"],
            comparison_contract[
                "graph_and_control_same_satisfying_assignment_count"
            ],
            comparison_contract[
                "global_phase_comparator_search_accounting_matches"
            ],
            comparison_contract[
                "global_phase_comparator_completed_when_comparable"
            ],
        )
    ):
        triggered.append("candidate_or_observation_budget_mismatch")

    return GraphFiberAnalysis(
        protocol_id=PROTOCOL_ID,
        evidence_mode=EVIDENCE_MODE,
        modulus=prime,
        node_count=node_count,
        coupling=weight,
        planted_assignment=planted,
        graph_edges=normalized_edges,
        shuffled_edges=shuffled_edges,
        shuffled_control_status=shuffled_status,
        graph_structure=original_structure,
        shuffled_structure=shuffled_structure,
        independent=independent,
        graph_coupled=graph_coupled,
        shuffled_graph=shuffled_graph,
        global_phase_offsets=global_phase_offsets,
        global_phase_comparator_status=global_phase_status,
        zero_violation_manifold_equals_global_phase_comparator=(
            zero_violation_manifold_equals_comparator
        ),
        global_phase_comparator=global_phase_comparator,
        global_phase_hypotheses_evaluated=global_phase_evaluated,
        resource_guard=resource,
        comparison_contract=MappingProxyType(comparison_contract),
        paired_effects=MappingProxyType(paired_effects),
        kill_criteria_triggered=tuple(triggered),
        limitations=LIMITATIONS,
        hypothesis_status="NON_CONFIRMATORY_SINGLE_INSTANCE",
        promotion_eligible=False,
        physical_claim=False,
        novelty_claim=False,
    )


__all__ = [
    "ALLOWED_MODULI",
    "DEFAULT_MAX_ESTIMATED_BYTES",
    "DEFAULT_MAX_LABEL_PERMUTATIONS",
    "DEFAULT_MAX_SECONDS",
    "DEFAULT_MAX_WORK_UNITS",
    "DecoderOutcome",
    "EVIDENCE_MODE",
    "FiberEdge",
    "GraphFiberAnalysis",
    "GraphFiberBudget",
    "GraphFiberResourceError",
    "GraphFiberValidationError",
    "GraphStructure",
    "HARD_MAX_EDGES",
    "HARD_MAX_ESTIMATED_BYTES",
    "HARD_MAX_HYPOTHESES",
    "HARD_MAX_LABEL_PERMUTATIONS",
    "HARD_MAX_NODES",
    "HARD_MAX_RATIONAL_BITS",
    "HARD_MAX_SECONDS",
    "HARD_MAX_WORK_UNITS",
    "KILL_CRITERIA",
    "LIMITATIONS",
    "PROTOCOL_ID",
    "ResourceEstimate",
    "analyze_graph_fibers",
]
