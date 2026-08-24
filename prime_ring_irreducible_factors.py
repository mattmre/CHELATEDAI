"""Exact algebraic discriminator for the bounded PRW-G2 mesh hypothesis.

The screen answers only necessary algebraic questions:

* is a pair factor separable into unary terms;
* is a higher-order factor reducible to strict-subset terms on the original
  variables;
* does a query change an interaction rather than only a unary score; and
* does an exact optimum collapse to one global phase plus fixed offsets.

All fixtures use exact integer arithmetic and streamed enumeration.  A
surviving fixture is ordinary factor-graph behavior, not evidence of novelty,
retrieval utility, dimensional physics, or a systems advantage.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from itertools import permutations, product
from typing import Callable, Dict, Optional, Sequence, Tuple

from prime_ring_rb10_contract import (
    ArtifactEnvelope,
    Deadline,
    ExperimentBudget,
    ResourceEstimate,
    RunContract,
    preflight_experiment,
)


PROTOCOL_ID = "CHELATEDAI-PRW-G2-ALGEBRA-v1"
HYPOTHESIS_ID = "PRW-G2"
ALLOWED_MODULI = (7, 11)
COMMON_CHANNEL_ID = "BSC-Q-1-5-DECLARED-NOT-SAMPLED"
TIE_POLICY_ID = "LEXICOGRAPHIC-CANONICAL-ASSIGNMENT"
DECISION_RULE_ID = "EXACT-ALGEBRAIC-RESIDUAL-AND-STREAMED-MAP"
EVIDENCE_MODE = "EXACT_BOUNDED_ALGEBRAIC_METHOD_DEV"
MATCHED_RANDOM_SEED = 4691

LIMITATIONS = (
    "The algebraic screen does not sample the declared BSC channel.",
    "Nonzero interaction residuals are necessary but not sufficient for a useful mesh.",
    "Exact flat and factorized MAP equality is an invariant, not an advantage.",
    "The modular hyperfactor records a generic pairwise auxiliary upper bound, not a proven minimum.",
    "The independent, shuffled-label, wrong-grouping, and matched-random controls do not establish a static advantage.",
    "No approximate decoder, training loop, retrieval corpus, or scaling claim is included.",
    "Modeled bytes are not process RSS and the cooperative deadline is not preemptive.",
    "Factor graphs, hypergraphs, synchronization, and message passing are established prior art.",
    "No novelty, production, physical-dimensionality, gravity, or resonance claim is made.",
)


class G2ValidationError(ValueError):
    """Raised when a G2 algebraic input violates the frozen contract."""


@dataclass(frozen=True)
class PairResidualAnalysis:
    modulus: int
    nonzero_count: int
    maximum_absolute_residual: int
    separable_into_unaries: bool
    witness: Optional[Tuple[int, int, int]]


@dataclass(frozen=True)
class HyperResidualAnalysis:
    modulus: int
    nonzero_count: int
    maximum_absolute_residual: int
    strict_subset_reducible: bool
    witness: Optional[Tuple[int, int, int, int]]


@dataclass(frozen=True)
class OptimumAnalysis:
    modulus: int
    node_count: int
    optimum_score: int
    maximizer_count: int
    diagonal_orbit_count: int
    fixed_offset_reducible: bool
    canonical_offset_representatives: Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class TreewidthAnalysis:
    node_count: int
    factor_scopes: Tuple[Tuple[int, ...], ...]
    exact_treewidth: int
    minimizing_order: Tuple[int, ...]


@dataclass(frozen=True)
class G2AlgebraicResult:
    protocol_id: str
    hypothesis_id: str
    modulus: int
    common_channel_id: str
    channel_sampled: bool
    pair_separable_control: PairResidualAnalysis
    pair_orbit_factor: PairResidualAnalysis
    g1_consistent_triangle: OptimumAnalysis
    g2p_ambiguous_path: OptimumAnalysis
    frustrated_triangle: OptimumAnalysis
    independent_nodes_control: OptimumAnalysis
    shuffled_label_control: OptimumAnalysis
    wrong_grouping_control: OptimumAnalysis
    matched_random_factor_control: OptimumAnalysis
    lower_order_hyper_control: HyperResidualAnalysis
    mod_sum3_hyperfactor: HyperResidualAnalysis
    query_unary_only_change: PairResidualAnalysis
    query_interaction_change: PairResidualAnalysis
    flat_factorized_scores_equal: bool
    flat_factorized_selected_assignment_equal: bool
    pair_path_treewidth: TreewidthAnalysis
    triangle_treewidth: TreewidthAnalysis
    hyperedge_treewidth: TreewidthAnalysis
    pair_formula_bytes: int
    pair_table_bytes: int
    hyper_table_bytes: int
    flat_score_table_bytes: int
    generic_pairwise_auxiliary_upper_bound_states: int
    generic_pairwise_auxiliary_upper_bound_bytes: int
    minimal_auxiliary_cost_proved: bool
    execution_completed: bool
    control_screen_passed: bool
    static_control_advantage_established: bool
    pair_sublane_status: str
    hyper_sublane_status: str
    query_sublane_status: str
    overall_status: str
    promotion_eligible: bool
    novelty_claim: bool
    evidence_mode: str
    resource_guard: ResourceEstimate
    limitations: Tuple[str, ...]

    def as_dict(self) -> Dict[str, object]:
        """Return a deterministic JSON-compatible representation."""

        return asdict(self)

    def artifact_envelope(self, budget: ExperimentBudget) -> ArtifactEnvelope:
        """Bind this result to the shared RB-10 contract."""

        contract = make_run_contract(self.modulus, budget)
        return ArtifactEnvelope.create(
            stage_id="PRW-G2-ALGEBRA",
            status="COMPLETE",
            run_contract=contract,
            resource_estimate=self.resource_guard,
            result=self.as_dict(),
            limitations=self.limitations,
        )


def _plain_modulus(value: object) -> int:
    if type(value) is not int or value not in ALLOWED_MODULI:
        raise G2ValidationError("modulus must be a plain integer in {}".format(ALLOWED_MODULI))
    return value


def _exact_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise G2ValidationError("{} must be a plain integer".format(name))
    if value.bit_length() > 63:
        raise G2ValidationError("{} exceeds the 63-bit exact ceiling".format(name))
    return value


def _square_integer_table(
    table: object,
    *,
    name: str,
) -> Tuple[Tuple[int, ...], ...]:
    if type(table) is not tuple or not table:
        raise G2ValidationError("{} must be a non-empty plain tuple".format(name))
    modulus = len(table)
    rows = []
    for row_index, row in enumerate(table):
        if type(row) is not tuple or len(row) != modulus:
            raise G2ValidationError(
                "{}[{}] must be a plain row of length {}".format(
                    name,
                    row_index,
                    modulus,
                )
            )
        rows.append(
            tuple(_exact_int(value, "{}[{}][{}]".format(name, row_index, column)) for column, value in enumerate(row))
        )
    return tuple(rows)


def _cube_integer_table(
    table: object,
    *,
    name: str,
) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
    if type(table) is not tuple or not table:
        raise G2ValidationError("{} must be a non-empty plain tuple".format(name))
    modulus = len(table)
    planes = []
    for left, plane in enumerate(table):
        if type(plane) is not tuple or len(plane) != modulus:
            raise G2ValidationError("{}[{}] must contain {} rows".format(name, left, modulus))
        rows = []
        for middle, row in enumerate(plane):
            if type(row) is not tuple or len(row) != modulus:
                raise G2ValidationError(
                    "{}[{}][{}] must contain {} values".format(
                        name,
                        left,
                        middle,
                        modulus,
                    )
                )
            rows.append(
                tuple(
                    _exact_int(
                        value,
                        "{}[{}][{}][{}]".format(
                            name,
                            left,
                            middle,
                            right,
                        ),
                    )
                    for right, value in enumerate(row)
                )
            )
        planes.append(tuple(rows))
    return tuple(planes)


def analyze_pair_interaction(
    table: object,
    *,
    name: str = "pair_table",
) -> PairResidualAnalysis:
    """Compute the exact anchored mixed-difference residual."""

    values = _square_integer_table(table, name=name)
    modulus = len(values)
    if modulus not in ALLOWED_MODULI:
        raise G2ValidationError("{} must have modulus in {}".format(name, ALLOWED_MODULI))
    witness = None
    nonzero = 0
    maximum = 0
    origin = values[0][0]
    for left in range(modulus):
        for right in range(modulus):
            residual = values[left][right] - values[left][0] - values[0][right] + origin
            absolute = abs(residual)
            if residual:
                nonzero += 1
                if witness is None:
                    witness = (left, right, residual)
            maximum = max(maximum, absolute)
    return PairResidualAnalysis(
        modulus=modulus,
        nonzero_count=nonzero,
        maximum_absolute_residual=maximum,
        separable_into_unaries=(nonzero == 0),
        witness=witness,
    )


def analyze_hyper_interaction(
    table: object,
    *,
    name: str = "hyper_table",
) -> HyperResidualAnalysis:
    """Compute the exact anchored three-way Möbius residual."""

    values = _cube_integer_table(table, name=name)
    modulus = len(values)
    if modulus not in ALLOWED_MODULI:
        raise G2ValidationError("{} must have modulus in {}".format(name, ALLOWED_MODULI))
    witness = None
    nonzero = 0
    maximum = 0
    origin = values[0][0][0]
    for left in range(modulus):
        for middle in range(modulus):
            for right in range(modulus):
                residual = (
                    values[left][middle][right]
                    - values[left][middle][0]
                    - values[left][0][right]
                    - values[0][middle][right]
                    + values[left][0][0]
                    + values[0][middle][0]
                    + values[0][0][right]
                    - origin
                )
                absolute = abs(residual)
                if residual:
                    nonzero += 1
                    if witness is None:
                        witness = (left, middle, right, residual)
                maximum = max(maximum, absolute)
    return HyperResidualAnalysis(
        modulus=modulus,
        nonzero_count=nonzero,
        maximum_absolute_residual=maximum,
        strict_subset_reducible=(nonzero == 0),
        witness=witness,
    )


def _pair_table(
    modulus: int,
    function: Callable[[int, int], int],
) -> Tuple[Tuple[int, ...], ...]:
    return tuple(
        tuple(_exact_int(function(left, right), "pair_value") for right in range(modulus)) for left in range(modulus)
    )


def _hyper_table(
    modulus: int,
    function: Callable[[int, int, int], int],
) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
    return tuple(
        tuple(
            tuple(_exact_int(function(left, middle, right), "hyper_value") for right in range(modulus))
            for middle in range(modulus)
        )
        for left in range(modulus)
    )


def _normalized_scope(scope: Sequence[int], node_count: int) -> Tuple[int, ...]:
    if type(scope) is not tuple or not scope:
        raise G2ValidationError("factor scopes must be non-empty plain tuples")
    result = []
    for index, node in enumerate(scope):
        if type(node) is not int or not 0 <= node < node_count:
            raise G2ValidationError("scope node {} is outside [0, {})".format(index, node_count))
        result.append(node)
    if len(set(result)) != len(result):
        raise G2ValidationError("factor scopes cannot repeat a node")
    return tuple(sorted(result))


def exact_primal_treewidth(
    node_count: int,
    factor_scopes: Tuple[Tuple[int, ...], ...],
) -> TreewidthAnalysis:
    """Return exact primal-graph treewidth for at most six nodes."""

    if type(node_count) is not int or not 1 <= node_count <= 6:
        raise G2ValidationError("node_count must be a plain integer in [1, 6]")
    if type(factor_scopes) is not tuple:
        raise G2ValidationError("factor_scopes must be a plain tuple")
    scopes = tuple(_normalized_scope(scope, node_count) for scope in factor_scopes)
    base = {node: set() for node in range(node_count)}
    for scope in scopes:
        for left in scope:
            base[left].update(right for right in scope if right != left)

    best_width = node_count
    best_order = tuple(range(node_count))
    for order in permutations(range(node_count)):
        adjacency = {node: set(neighbors) for node, neighbors in base.items()}
        width = 0
        for node in order:
            live = adjacency[node]
            width = max(width, len(live))
            for left in live:
                adjacency[left].update(live - {left})
                adjacency[left].discard(node)
            adjacency[node].clear()
            if width >= best_width:
                break
        if width < best_width:
            best_width = width
            best_order = tuple(order)
    return TreewidthAnalysis(
        node_count=node_count,
        factor_scopes=scopes,
        exact_treewidth=best_width,
        minimizing_order=best_order,
    )


def _analyze_optimum(
    modulus: int,
    node_count: int,
    scorer: Callable[[Tuple[int, ...]], int],
    deadline: Deadline,
    *,
    stage: str,
) -> OptimumAnalysis:
    optimum = None
    maximizers = []
    for index, assignment in enumerate(product(range(modulus), repeat=node_count)):
        if index % 128 == 0:
            deadline.check(stage)
        score = _exact_int(scorer(tuple(assignment)), "{}_score".format(stage))
        if optimum is None or score > optimum:
            optimum = score
            maximizers = [tuple(assignment)]
        elif score == optimum:
            maximizers.append(tuple(assignment))
    if optimum is None:
        raise AssertionError("non-empty assignment stream produced no optimum")
    offsets = {tuple((value - assignment[0]) % modulus for value in assignment) for assignment in maximizers}
    representatives = tuple(sorted(offsets))
    fixed_offset = len(representatives) == 1 and len(maximizers) == modulus
    return OptimumAnalysis(
        modulus=modulus,
        node_count=node_count,
        optimum_score=optimum,
        maximizer_count=len(maximizers),
        diagonal_orbit_count=len(representatives),
        fixed_offset_reducible=fixed_offset,
        canonical_offset_representatives=representatives,
    )


def _flat_factorized_equivalence(
    modulus: int,
    deadline: Deadline,
) -> Tuple[bool, bool]:
    allowed = {1, modulus - 1}
    unary_left = tuple(-(value * value) for value in range(modulus))
    unary_middle = tuple(-2 * value for value in range(modulus))
    unary_right = tuple(-value for value in range(modulus))
    pair_left_middle = _pair_table(
        modulus,
        lambda left, middle: int((middle - left) % modulus in allowed),
    )
    pair_middle_right = _pair_table(
        modulus,
        lambda middle, right: int((right - middle) % modulus in allowed),
    )
    flat_scores = []
    for index, (left, middle, right) in enumerate(product(range(modulus), repeat=3)):
        if index % 128 == 0:
            deadline.check("g2_flat_table")
        flat_scores.append(
            unary_left[left]
            + unary_middle[middle]
            + unary_right[right]
            + pair_left_middle[left][middle]
            + pair_middle_right[middle][right]
        )

    score_equal = True
    factorized_optimum = None
    factorized_best = None
    for index, (left, middle, right) in enumerate(product(range(modulus), repeat=3)):
        if index % 128 == 0:
            deadline.check("g2_factorized_rescore")
        factorized_score = -(left * left + 2 * middle + right)
        factorized_score += int((middle - left) % modulus in allowed)
        factorized_score += int((right - middle) % modulus in allowed)
        if factorized_score != flat_scores[index]:
            score_equal = False
        assignment = (left, middle, right)
        if factorized_optimum is None or factorized_score > factorized_optimum:
            factorized_optimum = factorized_score
            factorized_best = assignment

    flat_optimum = max(flat_scores)
    flat_best = min(
        assignment
        for assignment, score in zip(
            product(range(modulus), repeat=3),
            flat_scores,
        )
        if score == flat_optimum
    )
    return (
        score_equal,
        flat_optimum == factorized_optimum and flat_best == factorized_best,
    )


def _matched_random_pair_tables(
    modulus: int,
) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]]:
    """Create two deterministic row-degree-two random pair controls."""

    generator = random.Random(MATCHED_RANDOM_SEED + modulus)
    tables = []
    for _edge in range(2):
        rows = []
        for _left in range(modulus):
            allowed_rights = set(generator.sample(range(modulus), 2))
            rows.append(tuple(int(right in allowed_rights) for right in range(modulus)))
        tables.append(tuple(rows))
    return tables[0], tables[1]


def estimate_g2_resources(
    modulus: object,
    budget: ExperimentBudget = ExperimentBudget(),
) -> ResourceEstimate:
    """Preflight the whole exact algebraic screen before fixture construction."""

    p = _plain_modulus(modulus)
    assignments = p**3
    pair_table_count = 12
    hyper_table_count = 2
    component_bytes = (
        ("pair_tables_python_upper_bound", pair_table_count * p * p * 48),
        (
            "hyper_tables_python_upper_bound",
            hyper_table_count * assignments * 48,
        ),
        ("flat_score_table_python_upper_bound", assignments * 48),
        (
            "maximizer_and_offset_python_upper_bound",
            assignments * 96 + p * p * 80,
        ),
        ("python_runtime_and_result_allowance", 1_048_576),
    )
    component_work = (
        ("pair_residuals", pair_table_count * p * p * 5),
        ("hyper_residuals", hyper_table_count * assignments * 12),
        ("streamed_optima", assignments * 9 * 8),
        ("flat_factorized_equivalence", assignments * 2 * 8),
        ("exact_treewidth", 3 * 720 * 36),
    )
    return preflight_experiment(
        assignment_count=assignments,
        node_count=3,
        factor_arities=(2, 2, 2, 3),
        component_bytes=component_bytes,
        component_work=component_work,
        budget=budget,
        streaming_assignments_not_materialized=False,
    )


def make_run_contract(
    modulus: object,
    budget: ExperimentBudget = ExperimentBudget(),
) -> RunContract:
    p = _plain_modulus(modulus)
    return RunContract.create(
        stage_id="PRW-G2-ALGEBRA",
        hypothesis_id=HYPOTHESIS_ID,
        decision_rule_id=DECISION_RULE_ID,
        channel_id=COMMON_CHANNEL_ID,
        tie_policy_id=TIE_POLICY_ID,
        control_ids=(
            "SEP_UNARY",
            "G1_CONSISTENT_TRIANGLE",
            "G2P_AMBIGUOUS_PATH",
            "FRUSTRATED_TRIANGLE",
            "INDEPENDENT_NODES",
            "SHUFFLED_LABEL_PATH",
            "WRONG_GROUPING_PATH",
            "MATCHED_RANDOM_ROW_DEGREE_TWO_PATH",
            "LOWER_ORDER_ONLY",
            "MOD_SUM3",
            "Q_UNARY_ONLY",
            "Q_INTERACTION",
            "EXACT_FLAT_MAP",
        ),
        seeds=(MATCHED_RANDOM_SEED + p,),
        parameters={
            "modulus": p,
            "node_count": 3,
            "arithmetic": "exact_integer",
            "channel_sampled": False,
        },
        budget=budget,
    )


def run_g2_algebraic_screen(
    modulus: object = 7,
    *,
    budget: ExperimentBudget = ExperimentBudget(),
) -> G2AlgebraicResult:
    """Run the complete tiny G2 algebraic discriminator."""

    p = _plain_modulus(modulus)
    resource = estimate_g2_resources(p, budget)
    deadline = Deadline.start(budget)
    deadline.check("g2_fixture_construction")

    separable = _pair_table(p, lambda left, right: 2 * left - 3 * right)
    allowed = {1, p - 1}
    orbit_pair = _pair_table(
        p,
        lambda left, right: int((right - left) % p in allowed),
    )
    query_unary = _pair_table(p, lambda left, _right: left)
    zero_pair = _pair_table(p, lambda _left, _right: 0)
    query_interaction_delta = tuple(
        tuple(orbit_pair[left][right] - zero_pair[left][right] for right in range(p)) for left in range(p)
    )

    lower_order = _hyper_table(
        p,
        lambda left, middle, right: left - 2 * middle + 3 * right + int((middle - left) % p in allowed),
    )
    modular_sum = _hyper_table(
        p,
        lambda left, middle, right: int((left + middle + right) % p == 0),
    )
    deadline.check("g2_residuals")

    pair_separable = analyze_pair_interaction(separable, name="SEP_UNARY")
    pair_orbit = analyze_pair_interaction(orbit_pair, name="G2P_AMBIGUOUS")
    lower_order_analysis = analyze_hyper_interaction(
        lower_order,
        name="LOWER_ORDER_ONLY",
    )
    mod_sum_analysis = analyze_hyper_interaction(modular_sum, name="MOD_SUM3")
    query_unary_analysis = analyze_pair_interaction(
        query_unary,
        name="Q_UNARY_ONLY",
    )
    query_interaction_analysis = analyze_pair_interaction(
        query_interaction_delta,
        name="Q_INTERACTION",
    )

    g1_edges = ((0, 1, 1), (1, 2, 2), (0, 2, 3))
    g1 = _analyze_optimum(
        p,
        3,
        lambda assignment: sum(
            int((assignment[target] - assignment[source]) % p == offset) for source, target, offset in g1_edges
        ),
        deadline,
        stage="g1_consistent_triangle",
    )
    ambiguous = _analyze_optimum(
        p,
        3,
        lambda assignment: (
            int((assignment[1] - assignment[0]) % p in allowed) + int((assignment[2] - assignment[1]) % p in allowed)
        ),
        deadline,
        stage="g2p_ambiguous_path",
    )
    frustrated_edges = ((0, 1, 1), (1, 2, 1), (0, 2, 3))
    frustrated = _analyze_optimum(
        p,
        3,
        lambda assignment: sum(
            int((assignment[target] - assignment[source]) % p == offset) for source, target, offset in frustrated_edges
        ),
        deadline,
        stage="frustrated_triangle",
    )
    independent = _analyze_optimum(
        p,
        3,
        lambda _assignment: 0,
        deadline,
        stage="independent_nodes_control",
    )
    shuffled_allowed = {2, p - 2}
    shuffled = _analyze_optimum(
        p,
        3,
        lambda assignment: (
            int((assignment[1] - assignment[0]) % p in shuffled_allowed)
            + int((assignment[2] - assignment[1]) % p in shuffled_allowed)
        ),
        deadline,
        stage="shuffled_label_control",
    )
    wrong_grouping = _analyze_optimum(
        p,
        3,
        lambda assignment: (
            int((assignment[1] - assignment[0]) % p in allowed) + int((assignment[2] - assignment[0]) % p in allowed)
        ),
        deadline,
        stage="wrong_grouping_control",
    )
    random_left_middle, random_middle_right = _matched_random_pair_tables(p)
    matched_random = _analyze_optimum(
        p,
        3,
        lambda assignment: (
            random_left_middle[assignment[0]][assignment[1]] + random_middle_right[assignment[1]][assignment[2]]
        ),
        deadline,
        stage="matched_random_factor_control",
    )
    mod_sum_optimum = _analyze_optimum(
        p,
        3,
        lambda assignment: int(sum(assignment) % p == 0),
        deadline,
        stage="mod_sum3_optimum",
    )
    if mod_sum_optimum.maximizer_count != p * p:
        raise AssertionError("MOD_SUM3 must have exactly p**2 maximizers")

    flat_equal, selected_equal = _flat_factorized_equivalence(p, deadline)
    pair_path_treewidth = exact_primal_treewidth(3, ((0, 1), (1, 2)))
    triangle_treewidth = exact_primal_treewidth(
        3,
        ((0, 1), (1, 2), (0, 2)),
    )
    hyperedge_treewidth = exact_primal_treewidth(3, ((0, 1, 2),))
    deadline.check("g2_finalize")

    pair_survives = (
        not pair_orbit.separable_into_unaries
        and not ambiguous.fixed_offset_reducible
        and ambiguous.diagonal_orbit_count > 1
    )
    hyper_survives = not mod_sum_analysis.strict_subset_reducible
    query_survives = (
        query_unary_analysis.separable_into_unaries and not query_interaction_analysis.separable_into_unaries
    )
    flat_invariant = flat_equal and selected_equal
    control_screen_passed = (
        independent.optimum_score == 0
        and independent.maximizer_count == p**3
        and independent.diagonal_orbit_count == p * p
        and shuffled.optimum_score == 2
        and shuffled.maximizer_count == 4 * p
        and wrong_grouping.optimum_score == 2
        and wrong_grouping.maximizer_count == 4 * p
        and matched_random.optimum_score == 2
        and matched_random.maximizer_count == 4 * p
    )
    overall_survives = pair_survives and hyper_survives and query_survives and flat_invariant and control_screen_passed

    return G2AlgebraicResult(
        protocol_id=PROTOCOL_ID,
        hypothesis_id=HYPOTHESIS_ID,
        modulus=p,
        common_channel_id=COMMON_CHANNEL_ID,
        channel_sampled=False,
        pair_separable_control=pair_separable,
        pair_orbit_factor=pair_orbit,
        g1_consistent_triangle=g1,
        g2p_ambiguous_path=ambiguous,
        frustrated_triangle=frustrated,
        independent_nodes_control=independent,
        shuffled_label_control=shuffled,
        wrong_grouping_control=wrong_grouping,
        matched_random_factor_control=matched_random,
        lower_order_hyper_control=lower_order_analysis,
        mod_sum3_hyperfactor=mod_sum_analysis,
        query_unary_only_change=query_unary_analysis,
        query_interaction_change=query_interaction_analysis,
        flat_factorized_scores_equal=flat_equal,
        flat_factorized_selected_assignment_equal=selected_equal,
        pair_path_treewidth=pair_path_treewidth,
        triangle_treewidth=triangle_treewidth,
        hyperedge_treewidth=hyperedge_treewidth,
        pair_formula_bytes=3 * 8,
        pair_table_bytes=p * p * 8,
        hyper_table_bytes=p**3 * 8,
        flat_score_table_bytes=p**3 * 8,
        generic_pairwise_auxiliary_upper_bound_states=p * p,
        generic_pairwise_auxiliary_upper_bound_bytes=3 * p**3 * 8,
        minimal_auxiliary_cost_proved=False,
        execution_completed=True,
        control_screen_passed=control_screen_passed,
        static_control_advantage_established=False,
        pair_sublane_status=(
            "SURVIVES_NECESSARY_NONSEPARABILITY_ONLY" if pair_survives else "KILLED_BY_ALGEBRAIC_SCREEN"
        ),
        hyper_sublane_status=(
            "SURVIVES_ORIGINAL_VARIABLE_MOBIUS_SCREEN_AUXILIARY_COST_PENDING"
            if hyper_survives
            else "KILLED_BY_STRICT_SUBSET_REDUCTION"
        ),
        query_sublane_status=(
            "SURVIVES_INTERACTION_CHANGE_NECESSITY_ONLY" if query_survives else "KILLED_AS_QUERY_UNARY_ONLY"
        ),
        overall_status=(
            "SURVIVES_ALGEBRAIC_SCREEN_ONLY_CONTROLS_MATCH"
            if overall_survives
            else "KILLED_BY_ALGEBRAIC_OR_CONTROL_SCREEN"
        ),
        promotion_eligible=False,
        novelty_claim=False,
        evidence_mode=EVIDENCE_MODE,
        resource_guard=resource,
        limitations=LIMITATIONS,
    )


__all__ = [
    "ALLOWED_MODULI",
    "COMMON_CHANNEL_ID",
    "G2AlgebraicResult",
    "G2ValidationError",
    "HyperResidualAnalysis",
    "LIMITATIONS",
    "MATCHED_RANDOM_SEED",
    "OptimumAnalysis",
    "PROTOCOL_ID",
    "PairResidualAnalysis",
    "TreewidthAnalysis",
    "analyze_hyper_interaction",
    "analyze_pair_interaction",
    "estimate_g2_resources",
    "exact_primal_treewidth",
    "make_run_contract",
    "run_g2_algebraic_screen",
]
