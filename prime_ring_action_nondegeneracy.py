"""Exact action-nondegeneracy gate for the bounded PRW-A1 lane.

Adaptive selection is established controlled sensing.  This module asks only
whether the frozen JO1 slope actions expose different semantic distance
fingerprints and contain an action-ranking crossover.  It deliberately does
not run a posterior policy or claim observation, error, or compute savings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

from prime_ring_joint_orbit_spectrum import (
    ACTION_IDS,
    COMMON_CHANNEL_ID,
    COORDINATES_PER_PLANK,
    OrbitState,
    STATES,
    STATE_COUNT,
    TYPE_ZERO_REPRESENTATIVE,
    orbit_word,
)
from prime_ring_rb10_contract import (
    ArtifactEnvelope,
    Deadline,
    ExperimentBudget,
    ResourceEstimate,
    RunContract,
    preflight_experiment,
)


PROTOCOL_ID = "CHELATEDAI-PRW-A1-NONDEGENERACY-P11-v1"
HYPOTHESIS_ID = "PRW-A1"
DECISION_RULE_ID = "EXACT-SEMANTIC-FINGERPRINT-CROSSOVER"
TIE_POLICY_ID = "PAIRWISE-HAMMING-DISTANCE"
EVIDENCE_MODE = "EXACT_BOUNDED_ACTION_NONDEGENERACY_METHOD_DEV"

LIMITATIONS = (
    "This gate proves only that fixed semantic labels can rank slope actions differently.",
    "All action fingerprints have the same distance multiset and are therefore related by a semantic competitor permutation.",
    "The cost and physical availability of semantic relabeling or slope selection are not measured.",
    "No posterior update, MaxEJS comparison, sequential stopping policy, or observation saving is tested.",
    "The declared BSC is not sampled because nondegeneracy depends only on exact Hamming distances.",
    "Modeled bytes are not process RSS and the cooperative deadline is not preemptive.",
    "Controlled sensing, active hypothesis testing, and MaxEJS are established prior art.",
    "No novelty, production, retrieval, training, physical-dimensionality, gravity, or resonance claim is made.",
)


class ActionNondegeneracyValidationError(ValueError):
    """Raised when the frozen A1 input or result contract is malformed."""


@dataclass(frozen=True)
class CrossoverWitness:
    truth: OrbitState
    first_action: int
    second_action: int
    first_competitor: OrbitState
    second_competitor: OrbitState
    first_competitor_distances: Tuple[int, int]
    second_competitor_distances: Tuple[int, int]


@dataclass(frozen=True)
class ActionNondegeneracyResult:
    protocol_id: str
    hypothesis_id: str
    truth: OrbitState
    action_count: int
    competitor_count: int
    unique_semantic_fingerprint_count: int
    unique_distance_multiset_count: int
    posterior_relevant_variable_pair_count: int
    action_pair_count: int
    action_pair_crossover_count: int
    universal_weakly_dominating_action: Optional[int]
    copied_action_fingerprint_equal: bool
    common_coordinate_roll_fingerprint_equal: bool
    all_actions_distance_multiset_equivalent: bool
    fixed_semantic_labels_nondegenerate: bool
    crossover_witness: CrossoverWitness
    action_status: str
    posterior_policy_executed: bool
    observation_saving_established: bool
    policy_compute_saving_established: bool
    promotion_eligible: bool
    novelty_claim: bool
    evidence_mode: str
    resource_guard: ResourceEstimate
    limitations: Tuple[str, ...]

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def _hamming(left: Tuple[int, ...], right: Tuple[int, ...]) -> int:
    if len(left) != len(right):
        raise AssertionError("internal words must have equal length")
    return sum(left_value != right_value for left_value, right_value in zip(left, right))


def _roll(word: Tuple[int, ...], shift: int) -> Tuple[int, ...]:
    normalized = shift % len(word)
    if normalized == 0:
        return word
    return word[-normalized:] + word[:-normalized]


def _fingerprint(
    action: int,
    truth: OrbitState,
    deadline: Deadline,
    *,
    common_roll: int = 0,
) -> Tuple[int, ...]:
    truth_word = orbit_word(action, truth)
    if common_roll:
        truth_word = _roll(truth_word, common_roll)
    values = []
    for index, state in enumerate(STATES):
        if index % 64 == 0:
            deadline.check("a1_semantic_fingerprint")
        if state == truth:
            continue
        candidate = orbit_word(action, state)
        if common_roll:
            candidate = _roll(candidate, common_roll)
        values.append(_hamming(truth_word, candidate))
    return tuple(values)


def _competitor_states(truth: OrbitState) -> Tuple[OrbitState, ...]:
    return tuple(state for state in STATES if state != truth)


def _dominant_action(
    fingerprints: Dict[int, Tuple[int, ...]],
) -> Optional[int]:
    for candidate_action, candidate in fingerprints.items():
        weakly_better_than_all = True
        strictly_better_somewhere = False
        for other_action, other in fingerprints.items():
            if other_action == candidate_action:
                continue
            if any(left < right for left, right in zip(candidate, other)):
                weakly_better_than_all = False
                break
            strictly_better_somewhere = strictly_better_somewhere or any(
                left > right for left, right in zip(candidate, other)
            )
        if weakly_better_than_all and strictly_better_somewhere:
            return candidate_action
    return None


def preflight_action_nondegeneracy(
    budget: ExperimentBudget = ExperimentBudget(),
) -> ResourceEstimate:
    competitor_count = STATE_COUNT - 1
    fingerprint_values = len(ACTION_IDS) * competitor_count
    component_bytes = (
        ("states_and_words", STATE_COUNT * 64 + 4 * COORDINATES_PER_PLANK * 32),
        ("semantic_fingerprints", fingerprint_values * 40),
        ("sorted_multiset_fingerprints", fingerprint_values * 40),
        ("result_and_metadata_allowance", 262_144),
    )
    component_work = (
        (
            "fingerprint_chip_comparisons",
            len(ACTION_IDS) * competitor_count * COORDINATES_PER_PLANK,
        ),
        (
            "action_pair_crossover_checks",
            (len(ACTION_IDS) * (len(ACTION_IDS) - 1) // 2) * competitor_count,
        ),
        (
            "dominance_and_variability_checks",
            len(ACTION_IDS) * len(ACTION_IDS) * competitor_count,
        ),
        (
            "isometry_control_chip_comparisons",
            2 * competitor_count * COORDINATES_PER_PLANK,
        ),
    )
    return preflight_experiment(
        assignment_count=STATE_COUNT,
        node_count=1,
        factor_arities=(),
        component_bytes=component_bytes,
        component_work=component_work,
        budget=budget,
    )


def make_run_contract(
    budget: ExperimentBudget = ExperimentBudget(),
) -> RunContract:
    return RunContract.create(
        stage_id="PRW-A1-NONDEGENERACY-P11",
        hypothesis_id=HYPOTHESIS_ID,
        decision_rule_id=DECISION_RULE_ID,
        channel_id=COMMON_CHANNEL_ID,
        tie_policy_id=TIE_POLICY_ID,
        control_ids=(
            "COPIED_ACTION",
            "COMMON_COORDINATE_ROLL_ISOMETRY",
            "DISTANCE_MULTISET_SEMANTIC_RELABELING",
            "SLOPE_1_VERSUS_5_CROSSOVER",
            "UNIVERSAL_DOMINANCE_CHECK",
        ),
        seeds=(),
        parameters={
            "truth": [0, 0, 0],
            "action_ids": list(ACTION_IDS),
            "state_count": STATE_COUNT,
            "posterior_policy_executed": False,
        },
        budget=budget,
    )


def analyze_action_nondegeneracy(
    *,
    budget: ExperimentBudget = ExperimentBudget(),
) -> ActionNondegeneracyResult:
    """Run the complete frozen semantic-fingerprint gate."""

    resource = preflight_action_nondegeneracy(budget)
    deadline = Deadline.start(budget)
    truth = TYPE_ZERO_REPRESENTATIVE
    competitors = _competitor_states(truth)
    fingerprints = {action: _fingerprint(action, truth, deadline) for action in ACTION_IDS}
    unique_fingerprints = set(fingerprints.values())
    multiset_fingerprints = {tuple(sorted(fingerprint)) for fingerprint in fingerprints.values()}
    variable_pairs = sum(
        len({fingerprints[action][index] for action in ACTION_IDS}) > 1 for index in range(len(competitors))
    )
    crossover_count = 0
    for left_index, left_action in enumerate(ACTION_IDS):
        for right_action in ACTION_IDS[left_index + 1 :]:
            differences = tuple(
                left - right
                for left, right in zip(
                    fingerprints[left_action],
                    fingerprints[right_action],
                )
            )
            if any(value > 0 for value in differences) and any(value < 0 for value in differences):
                crossover_count += 1

    first_competitor = OrbitState(1, 1, 0)
    second_competitor = OrbitState(1, 1, 3)
    first_index = competitors.index(first_competitor)
    second_index = competitors.index(second_competitor)
    witness = CrossoverWitness(
        truth=truth,
        first_action=1,
        second_action=5,
        first_competitor=first_competitor,
        second_competitor=second_competitor,
        first_competitor_distances=(
            fingerprints[1][first_index],
            fingerprints[5][first_index],
        ),
        second_competitor_distances=(
            fingerprints[1][second_index],
            fingerprints[5][second_index],
        ),
    )
    copied_equal = fingerprints[1] == _fingerprint(
        1,
        truth,
        deadline,
    )
    rolled_equal = fingerprints[1] == _fingerprint(
        1,
        truth,
        deadline,
        common_roll=1,
    )
    dominating = _dominant_action(fingerprints)
    fixed_label_nondegenerate = (
        len(unique_fingerprints) > 1 and variable_pairs > 0 and crossover_count > 0 and dominating is None
    )
    if witness.first_competitor_distances != (48, 42):
        raise AssertionError("first frozen crossover distances drifted")
    if witness.second_competitor_distances != (44, 50):
        raise AssertionError("second frozen crossover distances drifted")
    deadline.check("a1_finalize")

    if not fixed_label_nondegenerate:
        status = "ACTION_FAMILY_INFORMATIONALLY_DEGENERATE"
    elif len(multiset_fingerprints) == 1:
        status = "SURVIVES_FIXED_LABEL_CROSSOVER_SEMANTIC_RELABELING_COST_PENDING"
    else:
        status = "SURVIVES_ACTION_NONDEGENERACY_GATE"

    return ActionNondegeneracyResult(
        protocol_id=PROTOCOL_ID,
        hypothesis_id=HYPOTHESIS_ID,
        truth=truth,
        action_count=len(ACTION_IDS),
        competitor_count=len(competitors),
        unique_semantic_fingerprint_count=len(unique_fingerprints),
        unique_distance_multiset_count=len(multiset_fingerprints),
        posterior_relevant_variable_pair_count=variable_pairs,
        action_pair_count=len(ACTION_IDS) * (len(ACTION_IDS) - 1) // 2,
        action_pair_crossover_count=crossover_count,
        universal_weakly_dominating_action=dominating,
        copied_action_fingerprint_equal=copied_equal,
        common_coordinate_roll_fingerprint_equal=rolled_equal,
        all_actions_distance_multiset_equivalent=(len(multiset_fingerprints) == 1),
        fixed_semantic_labels_nondegenerate=fixed_label_nondegenerate,
        crossover_witness=witness,
        action_status=status,
        posterior_policy_executed=False,
        observation_saving_established=False,
        policy_compute_saving_established=False,
        promotion_eligible=False,
        novelty_claim=False,
        evidence_mode=EVIDENCE_MODE,
        resource_guard=resource,
        limitations=LIMITATIONS,
    )


def build_action_nondegeneracy_artifact(
    result: object,
    *,
    budget: ExperimentBudget = ExperimentBudget(),
) -> ArtifactEnvelope:
    if type(result) is not ActionNondegeneracyResult:
        raise ActionNondegeneracyValidationError("result must be exactly ActionNondegeneracyResult")
    return ArtifactEnvelope.create(
        stage_id="PRW-A1-NONDEGENERACY-P11",
        status="COMPLETE",
        run_contract=make_run_contract(budget),
        resource_estimate=result.resource_guard,
        result=result.as_dict(),
        limitations=result.limitations,
    )


__all__ = [
    "ActionNondegeneracyResult",
    "ActionNondegeneracyValidationError",
    "CrossoverWitness",
    "LIMITATIONS",
    "PROTOCOL_ID",
    "analyze_action_nondegeneracy",
    "build_action_nondegeneracy_artifact",
    "make_run_contract",
    "preflight_action_nondegeneracy",
]
