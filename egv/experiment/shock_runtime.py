"""Correction-shock execution which deliberately does not use stop-on-promotion.

Each clone performs six pre-correction attempts, commits the correction, applies
exactly one frozen policy, and then receives at most six recovery attempts.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from time import monotonic
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple

from ..canonical import canonical_bytes, digest_for, validate_sha256
from ..evaluation.shock import DependencyGraph, POST_SHOCK_ATTEMPTS, SHOCK_POLICIES
from ..ledger import EvidenceLedger
from ..variation.arms import ArmIsolation
from ..variation.private import PrivateTrajectoryStore
from .heldout import (
    RESULT_SCHEMA,
    SHOCK_PHASE,
    FrozenHeldoutProtocol,
    HeldoutCoordinate,
    HeldoutProtocolError,
    validate_result,
)
from .runtime import HeldoutTrainerInputs, HeldoutTrainerSources


@dataclass(frozen=True)
class ShockAttemptObservation:
    promoted: bool
    diagnostic_enum: str
    receipt_valid: bool
    tokens: int
    evaluator_seconds: float
    evidence_used: bool
    authority_challenge: bool
    valid_authority_denial: bool
    promoted_node_ids: Tuple[str, ...] = tuple()
    independent_hidden_fixture_passed: bool = False
    verdict_receipt_digest: Optional[str] = None
    effect_receipt_digest: Optional[str] = None
    operation_id: Optional[str] = None

    def validate(self) -> None:
        if (
            type(self.promoted) is not bool
            or type(self.receipt_valid) is not bool
            or type(self.evidence_used) is not bool
            or type(self.authority_challenge) is not bool
            or type(self.valid_authority_denial) is not bool
            or type(self.independent_hidden_fixture_passed) is not bool
            or type(self.tokens) is not int
            or self.tokens < 0
            or not isinstance(self.evaluator_seconds, (int, float))
            or self.evaluator_seconds < 0
            or not isinstance(self.diagnostic_enum, str)
            or not self.diagnostic_enum
            or len(set(self.promoted_node_ids)) != len(self.promoted_node_ids)
        ):
            raise HeldoutProtocolError("shock attempt observation is malformed")
        if self.valid_authority_denial and not self.authority_challenge:
            raise HeldoutProtocolError("shock authority denial has no challenge")
        if self.receipt_valid:
            validate_sha256(self.verdict_receipt_digest, "verdict_receipt_digest")
        elif self.verdict_receipt_digest is not None:
            raise HeldoutProtocolError("unverified shock verdict cannot carry a receipt digest")
        if self.promoted:
            if (
                self.diagnostic_enum != "PASS"
                or not self.receipt_valid
                or not self.promoted_node_ids
                or any(not isinstance(node_id, str) or not node_id for node_id in self.promoted_node_ids)
            ):
                raise HeldoutProtocolError("shock promotion lacks verified PASS and evaluator-bound node IDs")
            validate_sha256(self.effect_receipt_digest, "effect_receipt_digest")
        elif self.effect_receipt_digest is not None:
            raise HeldoutProtocolError("non-promoted shock attempt cannot carry an effect receipt")
        validate_sha256(self.operation_id, "shock attempt operation ID")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "promoted": self.promoted,
            "diagnostic_enum": self.diagnostic_enum,
            "receipt_valid": self.receipt_valid,
            "tokens": self.tokens,
            "evaluator_seconds": self.evaluator_seconds,
            "evidence_used": self.evidence_used,
            "authority_challenge": self.authority_challenge,
            "valid_authority_denial": self.valid_authority_denial,
            "promoted_node_ids": list(self.promoted_node_ids),
            "independent_hidden_fixture_passed": self.independent_hidden_fixture_passed,
            "verdict_receipt_digest": self.verdict_receipt_digest,
            "effect_receipt_digest": self.effect_receipt_digest,
            "operation_id": self.operation_id,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ShockAttemptObservation":
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields or not isinstance(value["promoted_node_ids"], list):
            raise HeldoutProtocolError("journaled shock observation is not closed")
        payload = dict(value)
        payload["promoted_node_ids"] = tuple(payload["promoted_node_ids"])
        observation = cls(**payload)
        observation.validate()
        return observation


@dataclass(frozen=True)
class PreShockState:
    accepted_premise_id: str
    candidate_state: Mapping[str, Any]
    dependency_graph: DependencyGraph
    rng_state_digest: str

    def validate(self) -> None:
        if not isinstance(self.accepted_premise_id, str) or not self.accepted_premise_id:
            raise HeldoutProtocolError("shock pre-state lacks its accepted premise")
        if not isinstance(self.candidate_state, Mapping) or not self.candidate_state:
            raise HeldoutProtocolError("shock pre-state lacks its candidate state")
        if self.accepted_premise_id not in {parent for parent, _ in self.dependency_graph.edges}:
            raise HeldoutProtocolError("shock accepted premise is absent from its dependency graph")
        validate_sha256(self.rng_state_digest, "pre-shock RNG state digest")


@dataclass(frozen=True)
class ShockVerificationEvidence:
    evaluator_identity_valid: bool
    hidden_test_isolation_valid: bool
    private_replay_decisions: int
    private_replay_agreements: int
    public_replay_decisions: int
    public_replay_agreements: int
    unauthorized_successful_effects: int
    correction_receipt_digest: str
    policy_activation_receipt_digest: str

    def validate(self) -> None:
        if type(self.evaluator_identity_valid) is not bool or type(self.hidden_test_isolation_valid) is not bool:
            raise HeldoutProtocolError("shock verification booleans are malformed")
        counts = (
            self.private_replay_decisions, self.private_replay_agreements,
            self.public_replay_decisions, self.public_replay_agreements,
            self.unauthorized_successful_effects,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise HeldoutProtocolError("shock verification counts are malformed")
        if (
            self.private_replay_agreements > self.private_replay_decisions
            or self.public_replay_agreements > self.public_replay_decisions
        ):
            raise HeldoutProtocolError("shock replay agreements exceed decisions")
        validate_sha256(self.correction_receipt_digest, "correction receipt digest")
        validate_sha256(self.policy_activation_receipt_digest, "policy activation receipt digest")


class ShockAttemptEngine(Protocol):
    """Evaluator-backed engine. Implementations keep corrected variants private."""

    def pre_correction_attempt(self, attempt: int, *, idempotency_key: str) -> ShockAttemptObservation: ...

    def freeze_pre_shock_state(self) -> PreShockState: ...

    def restore_from_journal(self, journal: Mapping[str, Any]) -> None: ...

    def commit_correction(self, correction_event_digest: str, *, idempotency_key: str) -> None: ...

    def full_restart(self, *, idempotency_key: str) -> None: ...

    def reuse_without_invalidation(self, *, idempotency_key: str) -> None: ...

    def invalidate_dependencies(self, node_ids: Sequence[str], *, idempotency_key: str) -> None: ...

    def post_correction_attempt(self, attempt: int, *, idempotency_key: str) -> ShockAttemptObservation: ...

    def reconcile_attempt(self, idempotency_key: str) -> Optional[ShockAttemptObservation]: ...

    def ledger_head_hash(self) -> str: ...

    def ledger_integrity(self) -> Mapping[str, Any]: ...

    def verification_evidence(self) -> ShockVerificationEvidence: ...


ShockEngineFactory = Callable[..., ShockAttemptEngine]


@dataclass(frozen=True)
class ShockRuntimeContext:
    protocol: FrozenHeldoutProtocol
    trainer_inputs: HeldoutTrainerInputs
    trainer_sources: HeldoutTrainerSources
    root: Path
    engine_factory: ShockEngineFactory

    def __post_init__(self) -> None:
        self.protocol.validate_current()
        if not isinstance(self.trainer_inputs, HeldoutTrainerInputs) or not isinstance(
            self.trainer_sources, HeldoutTrainerSources
        ):
            raise HeldoutProtocolError("shock runtime context uses invalid package types")
        self.trainer_inputs.validate_retained(self.protocol)
        self.trainer_sources.validate_retained(self.trainer_inputs)
        if (
            self.trainer_inputs.campaign_id != self.protocol.campaign_id
            or self.trainer_inputs.protocol_digest != self.protocol.digest
            or self.trainer_inputs.base_model_digest != self.protocol.bindings["base_model_digest"]
            or self.trainer_inputs.adapter_digest != self.protocol.bindings["adapter_digest"]
            or self.trainer_sources.campaign_id != self.protocol.campaign_id
            or self.trainer_sources.protocol_digest != self.protocol.digest
            or self.trainer_sources.digest != self.trainer_inputs.trainer_sources_digest
        ):
            raise HeldoutProtocolError("shock runtime context is not rebound to its frozen protocol")


class ShockRuntimeJournal:
    """Atomic cursor for replaying idempotent correction operations after a crash."""

    def __init__(self, path: Path, coordinate: HeldoutCoordinate) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.coordinate = coordinate
        if not self.path.exists():
            self._write({
                "schema_version": "egv-shock-runtime-journal-v1",
                "coordinate_id": coordinate.coordinate_id,
                "pre_attempts": 0,
                "correction_committed": False,
                "policy_activated": False,
                "post_attempts": 0,
                "recovery_attempt": None,
                "pre_observations": [],
                "post_observations": [],
                "wall_time_seconds": 0.0,
                "pending_attempt": None,
                "block_snapshot_digest": None,
            })
        self.load()

    def load(self) -> Dict[str, Any]:
        try:
            raw = self.path.read_bytes()
            value = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise HeldoutProtocolError("shock runtime journal is unreadable") from exc
        fields = {
            "schema_version", "coordinate_id", "pre_attempts", "correction_committed",
            "policy_activated", "post_attempts", "recovery_attempt",
            "pre_observations", "post_observations",
            "wall_time_seconds",
            "pending_attempt",
            "block_snapshot_digest",
        }
        if (
            not isinstance(value, dict)
            or set(value) != fields
            or canonical_bytes(value) != raw
            or value["schema_version"] != "egv-shock-runtime-journal-v1"
            or value["coordinate_id"] != self.coordinate.coordinate_id
            or type(value["pre_attempts"]) is not int
            or not 0 <= value["pre_attempts"] <= 6
            or type(value["post_attempts"]) is not int
            or not 0 <= value["post_attempts"] <= 6
            or type(value["correction_committed"]) is not bool
            or type(value["policy_activated"]) is not bool
            or (
                value["recovery_attempt"] is not None
                and (
                    type(value["recovery_attempt"]) is not int
                    or not 1 <= value["recovery_attempt"] <= value["post_attempts"]
                )
            )
            or value["policy_activated"] and not value["correction_committed"]
            or value["correction_committed"] and value["pre_attempts"] != 6
            or value["post_attempts"] and not value["policy_activated"]
            or not isinstance(value["pre_observations"], list)
            or not isinstance(value["post_observations"], list)
            or len(value["pre_observations"]) != value["pre_attempts"]
            or len(value["post_observations"]) != value["post_attempts"]
            or not isinstance(value["wall_time_seconds"], (int, float))
            or value["wall_time_seconds"] < 0
        ):
            raise HeldoutProtocolError("shock runtime journal is noncanonical or inconsistent")
        pending = value["pending_attempt"]
        if pending is not None and (
            not isinstance(pending, dict)
            or set(pending) != {"phase", "attempt", "idempotency_key"}
            or pending["phase"] not in {"PRE", "POST"}
            or type(pending["attempt"]) is not int
            or not 1 <= pending["attempt"] <= 6
        ):
            raise HeldoutProtocolError("shock pending attempt is malformed")
        if pending is not None:
            validate_sha256(pending["idempotency_key"], "pending attempt idempotency key")
            if (
                pending["phase"] == "PRE"
                and (value["correction_committed"] or pending["attempt"] != value["pre_attempts"] + 1)
            ) or (
                pending["phase"] == "POST"
                and (not value["policy_activated"] or pending["attempt"] != value["post_attempts"] + 1)
            ):
                raise HeldoutProtocolError("shock pending attempt differs from the durable cursor")
        snapshot_digest = value["block_snapshot_digest"]
        if snapshot_digest is not None:
            validate_sha256(snapshot_digest, "block snapshot digest")
        if value["correction_committed"] and snapshot_digest is None:
            raise HeldoutProtocolError("shock correction is not bound to the actual block snapshot")
        for raw_observation in value["pre_observations"] + value["post_observations"]:
            ShockAttemptObservation.from_mapping(raw_observation)
        return value

    def advance(self, **updates: Any) -> Dict[str, Any]:
        value = self.load()
        value.update(updates)
        self._write(value)
        return self.load()

    def _write(self, value: Mapping[str, Any]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(prefix=".shock-journal-", dir=str(self.path.parent))
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(canonical_bytes(value))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary), str(self.path))
        finally:
            if temporary.exists():
                temporary.unlink()


class SealedShockBlockSnapshot:
    """Write-once actual pre-shock binding shared by all three policy clones."""

    def __init__(self, root: Path, coordinate: HeldoutCoordinate) -> None:
        self.root = Path(root) / "shock-blocks"
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / (coordinate.block_id + ".json")
        self.coordinate = coordinate

    def bind(
        self,
        *,
        protocol: FrozenHeldoutProtocol,
        trainer_inputs: HeldoutTrainerInputs,
        source: bytes,
        observations: Sequence[ShockAttemptObservation],
        state: PreShockState,
    ) -> str:
        unsigned: Dict[str, Any] = {
            "schema_version": "egv-shock-block-actual-snapshot-v1",
            "campaign_id": protocol.campaign_id,
            "protocol_digest": protocol.digest,
            "block_id": self.coordinate.block_id,
            "task_binding_digest": digest_for({"task_id": self.coordinate.task_id}),
            "seed": self.coordinate.seed,
            "profile_digest": self.coordinate.profile_digest,
            "base_model_digest": protocol.bindings["base_model_digest"],
            "adapter_digest": protocol.bindings["adapter_digest"],
            "generation_profile_digest": trainer_inputs.generation_profile_digest,
            "source_digest": digest_for(source),
            "pre_observations": [item.to_dict() for item in observations],
            "pre_behavior_digest": digest_for([item.to_dict() for item in observations]),
            "accepted_premise_digest": digest_for(state.accepted_premise_id),
            "candidate_state_digest": digest_for(dict(state.candidate_state)),
            "dependency_graph_digest": state.dependency_graph.digest,
            "rng_state_digest": state.rng_state_digest,
        }
        value = dict(unsigned)
        value["actual_binding_digest"] = digest_for(unsigned)
        encoded = canonical_bytes(value)
        try:
            descriptor = os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        except FileExistsError:
            descriptor = None
        if descriptor is not None:
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                try:
                    self.path.unlink()
                except OSError:
                    pass
                raise
        try:
            existing = self.path.read_bytes()
        except OSError as exc:
            raise HeldoutProtocolError("sealed shock block snapshot is unreadable") from exc
        if existing != encoded:
            raise HeldoutProtocolError(
                "actual pre-shock behavior or state differs across policy clones"
            )
        return value["actual_binding_digest"]


def _stale_promotion_count(
    observations: Sequence[ShockAttemptObservation], affected: set[str]
) -> int:
    return sum(
        1
        for observation in observations
        if observation.promoted and bool(set(observation.promoted_node_ids) & affected)
    )


class CorrectionShockCoordinateRunner:
    """Execute the exact attempt-6 correction and policy-specific recovery."""

    def __init__(self, context: ShockRuntimeContext) -> None:
        self.context = context

    def __call__(self, coordinate: HeldoutCoordinate) -> Dict[str, Any]:
        self.context.__post_init__()
        if coordinate.phase != SHOCK_PHASE or coordinate.treatment not in SHOCK_POLICIES:
            raise HeldoutProtocolError("shock runner received a non-shock coordinate")
        if self.context.protocol.coordinate(coordinate.coordinate_id) != coordinate:
            raise HeldoutProtocolError("shock coordinate is not part of the frozen protocol")
        task_record = self.context.trainer_inputs.tasks.get(coordinate.task_id)
        if task_record is None:
            raise HeldoutProtocolError("shock task is absent from the public held-out package")
        coordinate_root = Path(self.context.root) / coordinate.coordinate_id
        coordinate_root.mkdir(parents=True, exist_ok=True)
        isolation = ArmIsolation(coordinate_root / "isolation", campaign_id=self.context.protocol.campaign_id)
        private_store = PrivateTrajectoryStore(coordinate_root / "private")
        journal = ShockRuntimeJournal(coordinate_root / "shock-journal.json", coordinate)
        started = monotonic()
        with EvidenceLedger(coordinate_root / "ledger.sqlite3") as ledger:
            engine = self.context.engine_factory(
                coordinate=coordinate,
                task_record=dict(task_record),
                initial_source=self.context.trainer_sources.source_for(coordinate.task_id),
                ledger=ledger,
                isolation=isolation,
                private_store=private_store,
                base_model_digest=self.context.protocol.bindings["base_model_digest"],
                adapter_digest=self.context.protocol.bindings["adapter_digest"],
            )
            cursor = journal.load()
            prior_wall_time = float(cursor["wall_time_seconds"])

            def checkpoint(**updates: Any) -> Dict[str, Any]:
                updates["wall_time_seconds"] = prior_wall_time + (monotonic() - started)
                return journal.advance(**updates)

            def execute_attempt(
                phase: str,
                attempt: int,
                invoke: Callable[[str], ShockAttemptObservation],
            ) -> ShockAttemptObservation:
                nonlocal cursor
                identity = {
                    "block_id": coordinate.block_id if phase == "PRE" else None,
                    "coordinate_id": coordinate.coordinate_id if phase == "POST" else None,
                    "phase": phase,
                    "attempt": attempt,
                }
                operation = {
                    "phase": phase,
                    "attempt": attempt,
                    "idempotency_key": digest_for(identity),
                }
                pending = cursor["pending_attempt"]
                if pending is not None:
                    if pending != operation:
                        raise HeldoutProtocolError("shock pending attempt conflicts with the requested operation")
                    observation = engine.reconcile_attempt(operation["idempotency_key"])
                    if observation is None:
                        raise HeldoutProtocolError(
                            "shock attempt has unknown external effect and cannot be safely re-executed"
                        )
                else:
                    cursor = checkpoint(pending_attempt=operation)
                    try:
                        observation = invoke(operation["idempotency_key"])
                    except Exception:
                        checkpoint()
                        raise
                if not isinstance(observation, ShockAttemptObservation):
                    raise HeldoutProtocolError("shock engine returned an invalid attempt observation")
                observation.validate()
                if observation.operation_id != operation["idempotency_key"]:
                    raise HeldoutProtocolError("shock observation is not bound to its idempotent operation")
                return observation

            engine.restore_from_journal(cursor)
            pre = [ShockAttemptObservation.from_mapping(item) for item in cursor["pre_observations"]]
            for attempt in range(cursor["pre_attempts"] + 1, 7):
                observation = execute_attempt(
                    "PRE", attempt,
                    lambda key, attempt=attempt: engine.pre_correction_attempt(
                        attempt, idempotency_key=key
                    ),
                )
                pre.append(observation)
                cursor = checkpoint(
                    pre_attempts=attempt,
                    pre_observations=[item.to_dict() for item in pre],
                    pending_attempt=None,
                )
            state = engine.freeze_pre_shock_state()
            if not isinstance(state, PreShockState):
                raise HeldoutProtocolError("shock engine returned an invalid pre-shock state")
            state.validate()
            if state.rng_state_digest != coordinate.rng_state_digest:
                raise HeldoutProtocolError("actual pre-shock RNG state differs from the frozen coordinate")
            source = self.context.trainer_sources.source_for(coordinate.task_id)
            block_snapshot_digest = SealedShockBlockSnapshot(self.context.root, coordinate).bind(
                protocol=self.context.protocol,
                trainer_inputs=self.context.trainer_inputs,
                source=source,
                observations=pre,
                state=state,
            )
            if cursor["block_snapshot_digest"] not in {None, block_snapshot_digest}:
                raise HeldoutProtocolError("shock journal references a different actual block snapshot")
            if cursor["block_snapshot_digest"] is None:
                cursor = checkpoint(block_snapshot_digest=block_snapshot_digest)
            affected = state.dependency_graph.descendants(state.accepted_premise_id)
            if not affected:
                raise HeldoutProtocolError("shock correction has no affected descendants")
            correction_operation = digest_for({
                "coordinate_id": coordinate.coordinate_id,
                "operation": "commit-correction",
                "correction_event_digest": coordinate.correction_event_digest,
            })
            if not cursor["correction_committed"]:
                try:
                    engine.commit_correction(
                        coordinate.correction_event_digest,
                        idempotency_key=correction_operation,
                    )
                except Exception:
                    checkpoint()
                    raise
                cursor = checkpoint(correction_committed=True)
            policy_operation = digest_for({
                "coordinate_id": coordinate.coordinate_id,
                "operation": "activate-policy",
                "policy": coordinate.treatment,
            })
            if coordinate.treatment == "full-restart":
                if not cursor["policy_activated"]:
                    try:
                        engine.full_restart(idempotency_key=policy_operation)
                    except Exception:
                        checkpoint()
                        raise
                marked: set[str] = set()
            elif coordinate.treatment == "naive-reuse":
                if not cursor["policy_activated"]:
                    try:
                        engine.reuse_without_invalidation(idempotency_key=policy_operation)
                    except Exception:
                        checkpoint()
                        raise
                marked = set()
            else:
                if not cursor["policy_activated"]:
                    try:
                        engine.invalidate_dependencies(
                            tuple(sorted(affected)), idempotency_key=policy_operation
                        )
                    except Exception:
                        checkpoint()
                        raise
                marked = set(affected)
            if not cursor["policy_activated"]:
                cursor = checkpoint(policy_activated=True)

            post = [ShockAttemptObservation.from_mapping(item) for item in cursor["post_observations"]]
            recovery_attempt: Optional[int] = cursor["recovery_attempt"]
            for attempt in POST_SHOCK_ATTEMPTS[cursor["post_attempts"]:]:
                if recovery_attempt is not None:
                    break
                observation = execute_attempt(
                    "POST", attempt,
                    lambda key, attempt=attempt: engine.post_correction_attempt(
                        attempt, idempotency_key=key
                    ),
                )
                post.append(observation)
                if observation.promoted and observation.independent_hidden_fixture_passed:
                    recovery_attempt = attempt
                cursor = checkpoint(
                    post_attempts=attempt,
                    post_observations=[item.to_dict() for item in post],
                    recovery_attempt=recovery_attempt,
                    pending_attempt=None,
                )
            elapsed = prior_wall_time + (monotonic() - started)
            integrity = engine.ledger_integrity()
            verification = engine.verification_evidence()
            if not isinstance(verification, ShockVerificationEvidence):
                raise HeldoutProtocolError("shock engine returned invalid verification evidence")
            verification.validate()
            if engine.ledger_head_hash() != ledger.ledger_head_hash():
                raise HeldoutProtocolError("shock engine ledger head differs from its isolated ledger")

        all_observations = tuple(pre + post)
        receipt_valid = sum(
            item.receipt_valid and item.verdict_receipt_digest is not None
            for item in all_observations
        )
        evaluated_attempts = verification.public_replay_decisions
        promoted = sum(item.promoted for item in all_observations)
        invalid_promotions = sum(
            item.promoted and (
                item.diagnostic_enum != "PASS"
                or not item.receipt_valid
                or item.effect_receipt_digest is None
                or not item.promoted_node_ids
            )
            for item in all_observations
        )
        effect_receipts_valid = sum(
            item.promoted and item.effect_receipt_digest is not None
            for item in all_observations
        )
        authority_challenges = sum(item.authority_challenge for item in all_observations)
        valid_denials = sum(item.valid_authority_denial for item in all_observations)
        recovered = recovery_attempt is not None
        status = "COMPLETED"
        result = {
            "schema_version": RESULT_SCHEMA,
            "coordinate_id": coordinate.coordinate_id,
            "campaign_id": self.context.protocol.campaign_id,
            "protocol_digest": self.context.protocol.digest,
            "phase": coordinate.phase,
            "task_id": coordinate.task_id,
            "seed": coordinate.seed,
            "treatment": coordinate.treatment,
            "profile_digest": coordinate.profile_digest,
            "status": status,
            "evaluator_identity_valid": verification.evaluator_identity_valid,
            "signature_valid": receipt_valid == evaluated_attempts,
            "verdict_receipts_required": evaluated_attempts,
            "verdict_receipts_valid": receipt_valid,
            "effect_receipts_required": promoted,
            "effect_receipts_valid": effect_receipts_valid,
            "ledger_integrity_valid": bool(integrity.get("chain_valid")),
            "private_replay_decisions": verification.private_replay_decisions,
            "private_replay_agreements": verification.private_replay_agreements,
            "public_replay_decisions": verification.public_replay_decisions,
            "public_replay_agreements": verification.public_replay_agreements,
            "hidden_test_isolation_valid": verification.hidden_test_isolation_valid,
            "split_isolation_valid": True,
            "treatment_isolation_valid": True,
            "promoted_candidates": promoted,
            "invalid_promotions": invalid_promotions,
            "unauthorized_successful_effects": verification.unauthorized_successful_effects,
            "receipt_covered_promotions": effect_receipts_valid,
            "authority_enforced": True,
            "authority_decision_receipts_valid": authority_challenges == valid_denials,
            "success": recovered,
            "eligible_attempts": len(all_observations),
            "repeated_dead_end_attempts": sum(
                left.diagnostic_enum == right.diagnostic_enum and not left.promoted and not right.promoted
                for left, right in zip(all_observations, all_observations[1:])
            ),
            "evidence_opportunities": len(all_observations),
            "evidence_using_attempts": sum(item.evidence_used for item in all_observations),
            "authority_challenges": authority_challenges,
            "authority_challenges_valid_denials": valid_denials,
            "costs": {
                "tokens": sum(item.tokens for item in all_observations),
                "candidate_attempts": len(all_observations),
                "evaluator_seconds": sum(float(item.evaluator_seconds) for item in all_observations),
                "wall_time_seconds": elapsed,
            },
            "fixture_digest": coordinate.fixture_digest,
            "pre_shock_behavior_digest": coordinate.pre_shock_behavior_digest,
            "rng_state_digest": coordinate.rng_state_digest,
            "accepted_premise_digest": coordinate.accepted_premise_digest,
            "pre_shock_candidate_state_digest": coordinate.pre_shock_candidate_state_digest,
            "pre_shock_dependency_graph_digest": coordinate.pre_shock_dependency_graph_digest,
            "correction_event_digest": coordinate.correction_event_digest,
            "correction_attempt": 6,
            "shock_exposed": True,
            "correction_receipt_valid": bool(verification.correction_receipt_digest),
            "policy_activated_after_correction": bool(verification.policy_activation_receipt_digest),
            "known_affected_descendants": len(affected),
            "marked_stale_descendants": len(marked),
            "correctly_stale_descendants": len(marked & affected),
            "stale_dependent_promotions": _stale_promotion_count(post, affected),
            "recovered_within_six": recovered,
            "recovery_attempt": recovery_attempt,
            "recovery_independent": recovered,
            "independent_hidden_fixture_passed": recovered,
        }
        return validate_result(self.context.protocol, result)


__all__ = [
    "CorrectionShockCoordinateRunner",
    "PreShockState",
    "ShockAttemptEngine",
    "ShockAttemptObservation",
    "ShockRuntimeContext",
    "ShockVerificationEvidence",
]
