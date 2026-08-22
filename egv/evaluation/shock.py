"""Matched correction-shock fixtures and exact graph invalidation checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Set, Tuple

from ..canonical import content_id, digest_for
from .errors import ShockMismatchError


SHOCK_POLICIES = ("full-restart", "naive-reuse", "dependency-aware")
SHOCK_TASK_IDS = (
    "egv-shock-task-01",
    "egv-shock-task-02",
    "egv-shock-task-03",
    "egv-shock-task-04",
)
SHOCK_SEEDS = (11, 29, 47)
POST_SHOCK_ATTEMPTS = (1, 2, 3, 4, 5, 6)


def _oracle_affected_nodes(task_id: str, seed: int) -> Tuple[str, ...]:
    """Independent frozen oracle for the seeded correction closure.

    The oracle intentionally does not traverse ``DependencyGraph``.  It is
    derived from the fixture key and the contract's seeded candidate/descendant
    identities so a graph implementation can be wrong while the audit still
    detects it.
    """

    suffix = "{}-{}".format(task_id[-2:], seed)
    return tuple(sorted(("candidate-{}".format(suffix), "descendant-{}".format(suffix))))


@dataclass(frozen=True)
class DependencyGraph:
    edges: Tuple[Tuple[str, str], ...]

    def __post_init__(self) -> None:
        if len(set(self.edges)) != len(self.edges):
            raise ShockMismatchError("shock dependency graph contains duplicate edges")
        nodes = {item for edge in self.edges for item in edge}
        for parent, child in self.edges:
            if parent == child:
                raise ShockMismatchError("shock dependency graph contains a self-edge")
        for node in nodes:
            if self._reachable(node, node):
                raise ShockMismatchError("shock dependency graph contains a cycle")

    def _reachable(self, start: str, target: str) -> bool:
        frontier = [start]
        visited: Set[str] = set()
        while frontier:
            current = frontier.pop()
            for parent, child in self.edges:
                if parent != current or child in visited:
                    continue
                if child == target:
                    return True
                visited.add(child)
                frontier.append(child)
        return False

    def descendants(self, root: str) -> Set[str]:
        children: Dict[str, List[str]] = {}
        for parent, child in self.edges:
            children.setdefault(parent, []).append(child)
        found: Set[str] = set()
        frontier = list(children.get(root, []))
        while frontier:
            node = frontier.pop(0)
            if node in found:
                continue
            found.add(node)
            frontier.extend(children.get(node, []))
        return found

    @property
    def digest(self) -> str:
        return digest_for(list(self.edges))


@dataclass(frozen=True)
class ShockProfile:
    model_digest: str
    adapter_digest: str
    authority_policy_digest: str
    evaluator_digest: str
    rng_state_digest: str
    budget_profile_digest: str
    prompt_manifest_digest: str
    protocol_digest: str

    @property
    def digest(self) -> str:
        return digest_for(
            {
                "model_digest": self.model_digest,
                "adapter_digest": self.adapter_digest,
                "authority_policy_digest": self.authority_policy_digest,
                "evaluator_digest": self.evaluator_digest,
                "rng_state_digest": self.rng_state_digest,
                "budget_profile_digest": self.budget_profile_digest,
                "prompt_manifest_digest": self.prompt_manifest_digest,
                "protocol_digest": self.protocol_digest,
            }
        )


@dataclass(frozen=True)
class ShockFixture:
    task_id: str
    seed: int
    accepted_premise_id: str
    candidate_state: Mapping[str, Any]
    dependency_graph: DependencyGraph
    profile: ShockProfile
    correction_event_id: str
    correction_after_attempt: int = 6
    post_shock_attempts: Tuple[int, ...] = POST_SHOCK_ATTEMPTS
    oracle_affected_nodes: Tuple[str, ...] = tuple()

    def __post_init__(self) -> None:
        if self.correction_after_attempt != 6 or self.post_shock_attempts != POST_SHOCK_ATTEMPTS:
            raise ShockMismatchError("shock fixture does not use the frozen post-shock attempt window")
        if not self.oracle_affected_nodes:
            object.__setattr__(self, "oracle_affected_nodes", _oracle_affected_nodes(self.task_id, self.seed))
        if len(set(self.oracle_affected_nodes)) != len(self.oracle_affected_nodes):
            raise ShockMismatchError("shock oracle contains duplicate nodes")
        if self.accepted_premise_id in self.oracle_affected_nodes:
            raise ShockMismatchError("correction root cannot be an affected descendant")

    @classmethod
    def create(cls, task_id: str, seed: int) -> "ShockFixture":
        if task_id not in SHOCK_TASK_IDS or seed not in SHOCK_SEEDS:
            raise ShockMismatchError("shock fixture is outside the frozen task/seed design")
        accepted = "premise-{}-{}".format(task_id[-2:], seed)
        candidate = "candidate-{}-{}".format(task_id[-2:], seed)
        descendant = "descendant-{}-{}".format(task_id[-2:], seed)
        unrelated_root = "unrelated-root-{}-{}".format(task_id[-2:], seed)
        unrelated_child = "unrelated-child-{}-{}".format(task_id[-2:], seed)
        graph = DependencyGraph(
            (
                (accepted, candidate),
                (candidate, descendant),
                (unrelated_root, unrelated_child),
            )
        )
        profile = ShockProfile(
            model_digest=digest_for("shock-model-v1"),
            adapter_digest=digest_for("shock-adapter-v1"),
            authority_policy_digest=digest_for("shock-authority-policy-v1"),
            evaluator_digest=digest_for("shock-evaluator-v1"),
            rng_state_digest=digest_for({"task_id": task_id, "seed": seed, "boundary": "attempt-6"}),
            budget_profile_digest=digest_for({"max_attempts": 12, "post_shock_attempts": 6, "task_id": task_id}),
            prompt_manifest_digest=digest_for("shock-prompts-v1"),
            protocol_digest=digest_for("egv-evaluation-protocol-v1"),
        )
        correction_event_id = content_id(
            "correction",
            {"task_id": task_id, "seed": seed, "accepted_premise_id": accepted, "attempt": 6},
        )
        return cls(
            task_id=task_id,
            seed=seed,
            accepted_premise_id=accepted,
            candidate_state={"candidate_id": candidate, "attempt": 6, "status": "PROMOTED"},
            dependency_graph=graph,
            profile=profile,
            correction_event_id=correction_event_id,
        )

    @property
    def fixture_digest(self) -> str:
        return digest_for(
            {
                "task_id": self.task_id,
                "seed": self.seed,
                "accepted_premise_id": self.accepted_premise_id,
                "candidate_state": dict(self.candidate_state),
                "dependency_graph": list(self.dependency_graph.edges),
                "profile_digest": self.profile.digest,
                "correction_event_id": self.correction_event_id,
                "post_shock_attempts": list(self.post_shock_attempts),
                "oracle_affected_nodes": list(self.oracle_affected_nodes),
            }
        )

    def expected_affected_descendants(self) -> Set[str]:
        return set(self.oracle_affected_nodes)


@dataclass(frozen=True)
class ShockClone:
    fixture: ShockFixture
    policy: str
    stale_nodes: frozenset
    restarted: bool

    def __post_init__(self) -> None:
        if self.policy not in SHOCK_POLICIES:
            raise ShockMismatchError("unknown correction-shock policy")

    @property
    def profile_digest(self) -> str:
        return self.fixture.profile.digest

    def apply_correction(self, *, attempt: int = 6) -> "ShockClone":
        if attempt != self.fixture.correction_after_attempt:
            raise ShockMismatchError("correction must commit immediately after attempt 6")
        if self.policy == "dependency-aware":
            predicted = frozenset(self.fixture.dependency_graph.descendants(self.fixture.accepted_premise_id))
            return ShockClone(self.fixture, self.policy, predicted, False)
        if self.policy == "full-restart":
            return ShockClone(self.fixture, self.policy, frozenset(), True)
        return ShockClone(self.fixture, self.policy, frozenset(), False)


def graph_precision_recall(expected: Iterable[str], actual: Iterable[str]) -> Dict[str, float]:
    expected_set = set(expected)
    actual_set = set(actual)
    if not expected_set:
        raise ShockMismatchError("shock graph must contain a non-empty affected descendant set")
    true_positive = len(expected_set & actual_set)
    precision = true_positive / len(actual_set) if actual_set else 0.0
    recall = true_positive / len(expected_set)
    return {"precision": precision, "recall": recall}


class CorrectionShockSuite:
    """Generate and audit the exact 4-task x 3-seed x 3-policy factor."""

    def __init__(self) -> None:
        self.fixtures = tuple(ShockFixture.create(task_id, seed) for task_id in SHOCK_TASK_IDS for seed in SHOCK_SEEDS)

    def clones(self) -> Tuple[ShockClone, ...]:
        return tuple(
            ShockClone(fixture, policy, frozenset(), False)
            for fixture in self.fixtures
            for policy in SHOCK_POLICIES
        )

    def verify_exact_match(self, reference: ShockFixture, candidate: ShockFixture) -> None:
        if reference.fixture_digest != candidate.fixture_digest:
            raise ShockMismatchError("correction-shock clone differs in premise, state, graph, or frozen profile")

    def audit_dependency_aware(self) -> Dict[str, Any]:
        audited = []
        for fixture in self.fixtures:
            clone = ShockClone(fixture, "dependency-aware", frozenset(), False).apply_correction()
            metrics = graph_precision_recall(fixture.oracle_affected_nodes, clone.stale_nodes)
            if metrics != {"precision": 1.0, "recall": 1.0}:
                raise ShockMismatchError("dependency-aware correction did not invalidate exactly the seeded descendants")
            audited.append({"task_id": fixture.task_id, "seed": fixture.seed, **metrics})
        return {
            "fixture_count": len(self.fixtures),
            "clone_count": len(self.clones()),
            "policies": list(SHOCK_POLICIES),
            "correction_after_attempt": 6,
            "post_shock_attempts": list(POST_SHOCK_ATTEMPTS),
            "graph_precision": 1.0,
            "graph_recall": 1.0,
            "oracle": "independent-seeded-closure-v1",
            "audited_blocks": audited,
        }


__all__ = [
    "CorrectionShockSuite",
    "DependencyGraph",
    "SHOCK_POLICIES",
    "SHOCK_SEEDS",
    "SHOCK_TASK_IDS",
    "POST_SHOCK_ATTEMPTS",
    "ShockClone",
    "ShockFixture",
    "ShockMismatchError",
    "ShockProfile",
    "graph_precision_recall",
]
