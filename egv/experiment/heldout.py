"""Frozen matched held-out campaign and fail-closed research analysis.

This module intentionally stops at the evaluator/runtime boundary.  A caller
must inject a coordinate runner which performs real candidate generation,
sandboxing, receipt verification, and replay.  The orchestration here freezes
the 192 A-H and 36 correction-shock coordinates, journals exact private
results, rejects substitutions, and computes only preregistered aggregate
comparisons.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field as dataclass_field
import json
import os
from pathlib import Path
import random
import re
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from ..canonical import (
    GENESIS_HASH,
    canonical_bytes,
    canonical_json,
    canonical_value,
    content_id,
    digest_for,
    validate_sha256,
)
from ..campaign.state import _exclusive_path_lock
from ..evaluation.dataset import FAMILY_BY_ID, PUBLIC_HELDOUT_TEMPLATE_IDS
from ..evaluation.shock import SHOCK_POLICIES, SHOCK_SEEDS, SHOCK_TASK_IDS
from ..public import verify_public_restore_receipt
from ..receipts import key_id_for_public_key, load_public_key, public_key_bytes
from ..variation.arms import ARM_IDS, arm_policy


PROTOCOL_SCHEMA = "egv-heldout-protocol-v2"
COORDINATE_SCHEMA = "egv-heldout-coordinate-v1"
RESULT_SCHEMA = "egv-heldout-result-v1"
JOURNAL_SCHEMA = "egv-heldout-journal-record-v1"
REPORT_SCHEMA = "egv-heldout-aggregate-report-v1"
PRIVATE_RECORD_SCHEMA = "egv-heldout-private-record-v1"
SIGNED_RESULT_SCHEMA = "egv-heldout-signed-result-v1"
RECONCILIATION_SCHEMA = "egv-heldout-reconciliation-v1"
JOURNAL_INDEX_SCHEMA = "egv-heldout-journal-index-v1"
OPERATION_SCHEMA = "egv-heldout-operation-v1"

MAIN_PHASE = "MAIN_ABLATION"
SHOCK_PHASE = "CORRECTION_SHOCK"
RESULT_STATUSES = ("COMPLETED", "BUDGET_EXHAUSTED", "INFRASTRUCTURE_LOSS")
ESTIMABILITY_REASONS = (
    "COMPLETE",
    "MISSING_BLOCK",
    "ZERO_DENOMINATOR",
    "NO_EVIDENCE_USE",
    "NO_SHOCK_EXPOSURE",
    "INTERVAL_NONCOMPUTABLE",
    "BUDGET_EXHAUSTED",
    "MISSING_COST_MEASURE",
)
RESEARCH_GATES = (
    "G_EVIDENCE_MEMORY",
    "G_AUTHORITY_UTILITY",
    "G_CORRECTION_BENEFIT",
    "G_LORA_BENEFIT",
    "G_TRAINED_EVIDENCE_MEMORY",
    "G_TRAINED_AUTHORITY_UTILITY",
    "G_TRAINED_FULL_SYSTEM_CONTRIBUTION",
    "G_EFFICIENCY",
)
GATE_ORDER = ("G_RESTORATION", "G_HARD_INTEGRITY", "G_ESTIMABLE") + RESEARCH_GATES

DEFAULT_SEEDS = tuple(SHOCK_SEEDS)
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
MAX_ATTEMPTS = 12
MAX_POST_SHOCK_ATTEMPTS = 6
_PUBLIC_CAMPAIGN_ID = re.compile(r"^egv-campaign-[0-9a-f]{16,64}$")


class HeldoutProtocolError(ValueError):
    """The frozen design or a result violates the held-out contract."""


def _require_bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise HeldoutProtocolError("{} must be boolean".format(field))
    return value


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise HeldoutProtocolError("{} must be an integer >= {}".format(field, minimum))
    return value


def _require_number(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < minimum:
        raise HeldoutProtocolError("{} must be a finite number >= {}".format(field, minimum))
    numeric = float(value)
    if numeric == float("inf") or numeric != numeric:
        raise HeldoutProtocolError("{} must be finite".format(field))
    return numeric


def _require_closed_keys(value: Mapping[str, Any], required: Iterable[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise HeldoutProtocolError("{} must be an object".format(label))
    required_set = set(required)
    actual_set = set(value)
    if actual_set != required_set:
        missing = sorted(required_set - actual_set)
        extra = sorted(actual_set - required_set)
        raise HeldoutProtocolError(
            "{} has a non-closed schema (missing={}, extra={})".format(label, missing, extra)
        )


def _safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise HeldoutProtocolError("{} must be a bounded non-empty identifier".format(field))
    return value


def _public_campaign_id(value: Any) -> str:
    if not isinstance(value, str) or not _PUBLIC_CAMPAIGN_ID.fullmatch(value):
        raise HeldoutProtocolError(
            "campaign_id must be a bounded pseudonymous egv-campaign-<hex> identifier"
        )
    return value


@dataclass(frozen=True)
class HeldoutCoordinate:
    coordinate_id: str
    block_id: str
    phase: str
    task_id: str
    seed: int
    treatment: str
    block_order: int
    within_block_order: int
    profile_digest: str
    fixture_digest: Optional[str] = None
    pre_shock_behavior_digest: Optional[str] = None
    rng_state_digest: Optional[str] = None
    accepted_premise_digest: Optional[str] = None
    pre_shock_candidate_state_digest: Optional[str] = None
    pre_shock_dependency_graph_digest: Optional[str] = None
    correction_event_digest: Optional[str] = None

    def to_manifest_dict(self, campaign_id: str) -> Dict[str, Any]:
        """Return the coordinate fields included in the unsigned protocol."""

        return {
            "schema_version": COORDINATE_SCHEMA,
            "campaign_id": campaign_id,
            "coordinate_id": self.coordinate_id,
            "block_id": self.block_id,
            "phase": self.phase,
            "task_id": self.task_id,
            "seed": self.seed,
            "treatment": self.treatment,
            "block_order": self.block_order,
            "within_block_order": self.within_block_order,
            "profile_digest": self.profile_digest,
            "fixture_digest": self.fixture_digest,
            "pre_shock_behavior_digest": self.pre_shock_behavior_digest,
            "rng_state_digest": self.rng_state_digest,
            "accepted_premise_digest": self.accepted_premise_digest,
            "pre_shock_candidate_state_digest": self.pre_shock_candidate_state_digest,
            "pre_shock_dependency_graph_digest": self.pre_shock_dependency_graph_digest,
            "correction_event_digest": self.correction_event_digest,
        }

    def to_dict(self, protocol_digest: str, campaign_id: str) -> Dict[str, Any]:
        value = self.to_manifest_dict(campaign_id)
        value["protocol_digest"] = protocol_digest
        return value


@dataclass(frozen=True)
class FrozenHeldoutProtocol:
    campaign_id: str
    bindings: Mapping[str, str]
    evaluator_public_key_hex: str
    evaluator_key_id: str
    schedule_seed: int
    bootstrap_seed: int
    bootstrap_replicates: int
    heldout_task_ids: Tuple[str, ...]
    heldout_task_records: Tuple[Mapping[str, Any], ...]
    heldout_task_records_digest: str
    seeds: Tuple[int, ...]
    shock_task_ids: Tuple[str, ...]
    coordinates: Tuple[HeldoutCoordinate, ...]
    _sealed_protocol_digest: str = dataclass_field(init=False, repr=False, compare=False)

    REQUIRED_BINDINGS = (
        "base_model_digest",
        "trained_model_digest",
        "adapter_digest",
        "tokenizer_digest",
        "prompt_manifest_digest",
        "evaluator_digest",
        "policy_manifest_digest",
        "resource_profile_digest",
        "decoding_profile_digest",
        "data_manifest_digest",
        "shock_fixture_manifest_digest",
        "authority_challenge_manifest_digest",
        "restore_inventory_digest",
    )

    def __post_init__(self) -> None:
        # Every nested container retained by build() is immutable.  Seal the
        # content-derived digest once so ordinary reads do not re-serialize the
        # 228-coordinate manifest.  Execution admission still calls
        # validate_current(), which reconstructs and compares the full content.
        object.__setattr__(self, "_sealed_protocol_digest", digest_for(self._unsigned_dict()))

    @classmethod
    def build(
        cls,
        *,
        campaign_id: str,
        bindings: Mapping[str, str],
        evaluator_public_key: Any,
        schedule_seed: int,
        bootstrap_seed: int,
        heldout_task_records: Sequence[Mapping[str, Any]],
        heldout_task_ids: Sequence[str] = tuple(sorted(PUBLIC_HELDOUT_TEMPLATE_IDS)),
        seeds: Sequence[int] = DEFAULT_SEEDS,
        shock_task_ids: Sequence[str] = SHOCK_TASK_IDS,
        bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    ) -> "FrozenHeldoutProtocol":
        campaign_id = _public_campaign_id(campaign_id)
        _require_closed_keys(bindings, cls.REQUIRED_BINDINGS, "bindings")
        clean_bindings = {name: validate_sha256(bindings[name], name) for name in cls.REQUIRED_BINDINGS}
        schedule_seed = _require_int(schedule_seed, "schedule_seed")
        bootstrap_seed = _require_int(bootstrap_seed, "bootstrap_seed")
        if bootstrap_replicates != DEFAULT_BOOTSTRAP_REPLICATES:
            raise HeldoutProtocolError("bootstrap_replicates is frozen at 10000")
        tasks = tuple(heldout_task_ids)
        if len(tasks) != 8 or set(tasks) != set(PUBLIC_HELDOUT_TEMPLATE_IDS):
            raise HeldoutProtocolError("heldout_task_ids must be the exact eight frozen public addresses")
        task_record_fields = {
            "template_id", "family_id", "split", "ordinal", "source_digest",
            "public_rule_id", "public_locus",
        }
        clean_task_records: List[Mapping[str, Any]] = []
        if not isinstance(heldout_task_records, Sequence) or isinstance(
            heldout_task_records, (str, bytes, bytearray)
        ):
            raise HeldoutProtocolError("heldout_task_records must be the exact ordered public records")
        for index, raw_record in enumerate(heldout_task_records):
            _require_closed_keys(raw_record, task_record_fields, "held-out task record")
            for field in ("template_id", "family_id", "split", "public_rule_id", "public_locus"):
                if type(raw_record[field]) is not str or not raw_record[field]:
                    raise HeldoutProtocolError("held-out task record {} must be a nonempty string".format(field))
            if raw_record["split"] != "heldout":
                raise HeldoutProtocolError("held-out task record split must be heldout")
            if type(raw_record["ordinal"]) is not int or raw_record["ordinal"] <= 0:
                raise HeldoutProtocolError("held-out task record ordinal must be a positive integer")
            if raw_record["family_id"] not in FAMILY_BY_ID or raw_record["template_id"] != (
                "egv-{}-heldout-{}-v1".format(
                    raw_record["family_id"].lower(), raw_record["ordinal"]
                )
            ):
                raise HeldoutProtocolError("held-out task record family, ordinal, and address disagree")
            source_digest = validate_sha256(raw_record["source_digest"], "held-out task source digest")
            expected_task_id = tasks[index] if index < len(tasks) else None
            if raw_record["template_id"] != expected_task_id:
                raise HeldoutProtocolError("held-out task records are missing, reordered, or substituted")
            clean_task_records.append(MappingProxyType({
                "template_id": raw_record["template_id"],
                "family_id": raw_record["family_id"],
                "split": raw_record["split"],
                "ordinal": raw_record["ordinal"],
                "source_digest": source_digest,
                "public_rule_id": raw_record["public_rule_id"],
                "public_locus": raw_record["public_locus"],
            }))
        if len(clean_task_records) != len(tasks):
            raise HeldoutProtocolError("held-out task records are missing, reordered, or substituted")
        task_records_digest = digest_for([dict(record) for record in clean_task_records])
        frozen_seeds = tuple(seeds)
        if frozen_seeds != tuple(SHOCK_SEEDS):
            raise HeldoutProtocolError("seeds must be the exact canonical frozen sequence (11, 29, 47)")
        shock_tasks = tuple(shock_task_ids)
        if len(shock_tasks) != 4 or set(shock_tasks) != set(SHOCK_TASK_IDS):
            raise HeldoutProtocolError("shock_task_ids must be the exact four frozen shock addresses")
        try:
            public_key_raw = public_key_bytes(evaluator_public_key)
            evaluator_key_id = key_id_for_public_key(public_key_raw)
        except Exception as exc:
            raise HeldoutProtocolError("evaluator_public_key must be a valid Ed25519 public key") from exc

        coordinates = _build_coordinates(
            campaign_id=campaign_id,
            bindings=MappingProxyType(dict(clean_bindings)),
            schedule_seed=schedule_seed,
            heldout_task_ids=tasks,
            seeds=frozen_seeds,
            shock_task_ids=shock_tasks,
        )
        return cls(
            campaign_id=campaign_id,
            bindings=MappingProxyType(dict(clean_bindings)),
            evaluator_public_key_hex=public_key_raw.hex(),
            evaluator_key_id=evaluator_key_id,
            schedule_seed=schedule_seed,
            bootstrap_seed=bootstrap_seed,
            bootstrap_replicates=bootstrap_replicates,
            heldout_task_ids=tasks,
            heldout_task_records=tuple(clean_task_records),
            heldout_task_records_digest=task_records_digest,
            seeds=frozen_seeds,
            shock_task_ids=shock_tasks,
            coordinates=coordinates,
        )

    @property
    def digest(self) -> str:
        return self._sealed_protocol_digest

    @property
    def _coordinates_by_id(self) -> Mapping[str, HeldoutCoordinate]:
        return MappingProxyType({coordinate.coordinate_id: coordinate for coordinate in self.coordinates})

    def validate_current(self) -> str:
        """Rebuild the full protocol projection and reject in-memory substitution."""

        rebuilt = type(self).build(
            campaign_id=self.campaign_id,
            bindings=dict(self.bindings),
            evaluator_public_key=bytes.fromhex(self.evaluator_public_key_hex),
            schedule_seed=self.schedule_seed,
            bootstrap_seed=self.bootstrap_seed,
            heldout_task_records=[dict(record) for record in self.heldout_task_records],
            heldout_task_ids=self.heldout_task_ids,
            seeds=self.seeds,
            shock_task_ids=self.shock_task_ids,
            bootstrap_replicates=self.bootstrap_replicates,
        )
        current_projection = self._unsigned_dict()
        if (
            rebuilt._unsigned_dict() != current_projection
            or rebuilt.digest != self._sealed_protocol_digest
        ):
            raise HeldoutProtocolError(
                "current held-out protocol failed immutable admission: "
                "it differs from its frozen reconstruction"
            )
        if self.heldout_task_records_digest != digest_for(
            [dict(record) for record in self.heldout_task_records]
        ):
            raise HeldoutProtocolError(
                "current held-out protocol failed immutable admission: "
                "task-record commitment is invalid"
            )
        return rebuilt.digest

    def _unsigned_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": PROTOCOL_SCHEMA,
            "campaign_id": self.campaign_id,
            "bindings": dict(self.bindings),
            "evaluator_public_key_hex": self.evaluator_public_key_hex,
            "evaluator_key_id": self.evaluator_key_id,
            "schedule_seed": self.schedule_seed,
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_method": "paired-block-percentile-v1",
            "confidence_level": 0.95,
            "heldout_task_ids": list(self.heldout_task_ids),
            "heldout_task_records": [dict(record) for record in self.heldout_task_records],
            "heldout_task_records_digest": self.heldout_task_records_digest,
            "seeds": list(self.seeds),
            "shock_task_ids": list(self.shock_task_ids),
            "max_attempts": MAX_ATTEMPTS,
            "max_post_shock_attempts": MAX_POST_SHOCK_ATTEMPTS,
            "coordinates": [coordinate.to_manifest_dict(self.campaign_id) for coordinate in self.coordinates],
        }

    def to_private_dict(self) -> Dict[str, Any]:
        value = self._unsigned_dict()
        value["protocol_digest"] = self.digest
        for coordinate in value["coordinates"]:
            coordinate["protocol_digest"] = self.digest
        return value

    def coordinate(self, coordinate_id: str) -> HeldoutCoordinate:
        try:
            return self._coordinates_by_id[coordinate_id]
        except (KeyError, TypeError) as exc:
            raise HeldoutProtocolError("result references an unplanned coordinate") from exc


def _build_coordinates(
    *,
    campaign_id: str,
    bindings: Mapping[str, str],
    schedule_seed: int,
    heldout_task_ids: Tuple[str, ...],
    seeds: Tuple[int, ...],
    shock_task_ids: Tuple[str, ...],
) -> Tuple[HeldoutCoordinate, ...]:
    rng = random.Random(schedule_seed)
    main_blocks = [(task_id, seed) for task_id in sorted(heldout_task_ids) for seed in sorted(seeds)]
    rng.shuffle(main_blocks)
    arm_base = list(ARM_IDS)
    rng.shuffle(arm_base)
    coordinates: List[HeldoutCoordinate] = []
    for block_order, (task_id, seed) in enumerate(main_blocks):
        block_id = content_id("block", {"campaign_id": campaign_id, "phase": MAIN_PHASE, "task_id": task_id, "seed": seed})
        ordered_arms = arm_base[block_order % len(arm_base):] + arm_base[:block_order % len(arm_base)]
        for within_order, arm_id in enumerate(ordered_arms):
            policy = arm_policy(arm_id)
            active_model = bindings["trained_model_digest"] if policy.requires_adapter else bindings["base_model_digest"]
            active_adapter: Optional[str] = bindings["adapter_digest"] if policy.requires_adapter else None
            profile_digest = digest_for(
                {
                    "active_model_digest": active_model,
                    "active_adapter_digest": active_adapter,
                    "base_model_digest": bindings["base_model_digest"],
                    "tokenizer_digest": bindings["tokenizer_digest"],
                    "prompt_manifest_digest": bindings["prompt_manifest_digest"],
                    "evaluator_digest": bindings["evaluator_digest"],
                    "arm_policy_digest": policy.digest,
                    "policy_manifest_digest": bindings["policy_manifest_digest"],
                    "resource_profile_digest": bindings["resource_profile_digest"],
                    "decoding_profile_digest": bindings["decoding_profile_digest"],
                    "data_manifest_digest": bindings["data_manifest_digest"],
                    "authority_challenge_manifest_digest": bindings["authority_challenge_manifest_digest"],
                }
            )
            identity = {
                "campaign_id": campaign_id,
                "phase": MAIN_PHASE,
                "task_id": task_id,
                "seed": seed,
                "treatment": arm_id,
                "profile_digest": profile_digest,
            }
            coordinates.append(
                HeldoutCoordinate(
                    coordinate_id=content_id("coordinate", identity),
                    block_id=block_id,
                    phase=MAIN_PHASE,
                    task_id=task_id,
                    seed=seed,
                    treatment=arm_id,
                    block_order=block_order,
                    within_block_order=within_order,
                    profile_digest=profile_digest,
                )
            )

    shock_blocks = [(task_id, seed) for task_id in sorted(shock_task_ids) for seed in sorted(seeds)]
    rng.shuffle(shock_blocks)
    policy_base = list(SHOCK_POLICIES)
    rng.shuffle(policy_base)
    shock_profile = digest_for(
        {
            "shock_fixture_manifest_digest": bindings["shock_fixture_manifest_digest"],
            "active_model_digest": bindings["trained_model_digest"],
            "active_adapter_digest": bindings["adapter_digest"],
            "base_model_digest": bindings["base_model_digest"],
            "tokenizer_digest": bindings["tokenizer_digest"],
            "prompt_manifest_digest": bindings["prompt_manifest_digest"],
            "evaluator_digest": bindings["evaluator_digest"],
            "policy_manifest_digest": bindings["policy_manifest_digest"],
            "resource_profile_digest": bindings["resource_profile_digest"],
            "decoding_profile_digest": bindings["decoding_profile_digest"],
            "pre_shock_arm": "E",
        }
    )
    for block_order, (task_id, seed) in enumerate(shock_blocks):
        block_id = content_id("block", {"campaign_id": campaign_id, "phase": SHOCK_PHASE, "task_id": task_id, "seed": seed})
        fixture_digest = digest_for(
            {
                "campaign_id": campaign_id,
                "task_id": task_id,
                "seed": seed,
                "shock_fixture_manifest_digest": bindings["shock_fixture_manifest_digest"],
                "fixture": "accepted-premise-state-graph-v1",
            }
        )
        pre_shock_digest = digest_for(
            {"block_id": block_id, "profile_digest": shock_profile, "pre_shock_policy": "ARM_E", "correction_after_attempt": 6}
        )
        rng_state_digest = digest_for(
            {"campaign_id": campaign_id, "task_id": task_id, "seed": seed, "boundary": "AFTER_ATTEMPT_6"}
        )
        accepted_premise_digest = digest_for(
            {"fixture_digest": fixture_digest, "component": "accepted-premise"}
        )
        pre_shock_candidate_state_digest = digest_for(
            {"fixture_digest": fixture_digest, "component": "promoted-candidate-state"}
        )
        pre_shock_dependency_graph_digest = digest_for(
            {"fixture_digest": fixture_digest, "component": "dependency-graph"}
        )
        correction_event_digest = digest_for(
            {"fixture_digest": fixture_digest, "component": "correction-after-attempt-6"}
        )
        ordered_policies = policy_base[block_order % len(policy_base):] + policy_base[:block_order % len(policy_base)]
        for within_order, policy in enumerate(ordered_policies):
            identity = {
                "campaign_id": campaign_id,
                "phase": SHOCK_PHASE,
                "task_id": task_id,
                "seed": seed,
                "treatment": policy,
                "profile_digest": shock_profile,
                "fixture_digest": fixture_digest,
            }
            coordinates.append(
                HeldoutCoordinate(
                    coordinate_id=content_id("coordinate", identity),
                    block_id=block_id,
                    phase=SHOCK_PHASE,
                    task_id=task_id,
                    seed=seed,
                    treatment=policy,
                    block_order=block_order,
                    within_block_order=within_order,
                    profile_digest=shock_profile,
                    fixture_digest=fixture_digest,
                    pre_shock_behavior_digest=pre_shock_digest,
                    rng_state_digest=rng_state_digest,
                    accepted_premise_digest=accepted_premise_digest,
                    pre_shock_candidate_state_digest=pre_shock_candidate_state_digest,
                    pre_shock_dependency_graph_digest=pre_shock_dependency_graph_digest,
                    correction_event_digest=correction_event_digest,
                )
            )
    if len(coordinates) != 228 or len({item.coordinate_id for item in coordinates}) != 228:
        raise HeldoutProtocolError("frozen schedule did not produce 228 unique coordinates")
    return tuple(coordinates)


COMMON_RESULT_KEYS = (
    "schema_version", "coordinate_id", "campaign_id", "protocol_digest", "phase", "task_id", "seed",
    "treatment", "profile_digest", "status", "evaluator_identity_valid", "signature_valid",
    "verdict_receipts_required", "verdict_receipts_valid", "effect_receipts_required", "effect_receipts_valid",
    "ledger_integrity_valid", "private_replay_decisions", "private_replay_agreements",
    "public_replay_decisions", "public_replay_agreements", "hidden_test_isolation_valid", "split_isolation_valid",
    "treatment_isolation_valid", "promoted_candidates", "invalid_promotions", "unauthorized_successful_effects",
    "receipt_covered_promotions", "authority_enforced", "authority_decision_receipts_valid", "success",
    "eligible_attempts", "repeated_dead_end_attempts", "evidence_opportunities", "evidence_using_attempts",
    "authority_challenges", "authority_challenges_valid_denials", "costs",
)
SHOCK_RESULT_KEYS = (
    "fixture_digest", "pre_shock_behavior_digest", "rng_state_digest", "shock_exposed", "correction_receipt_valid",
    "accepted_premise_digest", "pre_shock_candidate_state_digest", "pre_shock_dependency_graph_digest",
    "correction_event_digest", "correction_attempt",
    "policy_activated_after_correction", "known_affected_descendants", "marked_stale_descendants",
    "correctly_stale_descendants", "stale_dependent_promotions", "recovered_within_six", "recovery_attempt",
    "recovery_independent", "independent_hidden_fixture_passed",
)
COST_KEYS = ("tokens", "candidate_attempts", "evaluator_seconds", "wall_time_seconds")


def validate_result(protocol: FrozenHeldoutProtocol, raw: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise HeldoutProtocolError("coordinate result must be an object")
    coordinate_id = raw.get("coordinate_id")
    coordinate = protocol.coordinate(coordinate_id)
    required = COMMON_RESULT_KEYS + (SHOCK_RESULT_KEYS if coordinate.phase == SHOCK_PHASE else tuple())
    _require_closed_keys(raw, required, "coordinate result")
    expected = {
        "schema_version": RESULT_SCHEMA,
        "coordinate_id": coordinate.coordinate_id,
        "campaign_id": protocol.campaign_id,
        "protocol_digest": protocol.digest,
        "phase": coordinate.phase,
        "task_id": coordinate.task_id,
        "seed": coordinate.seed,
        "treatment": coordinate.treatment,
        "profile_digest": coordinate.profile_digest,
    }
    for field, value in expected.items():
        if raw[field] != value:
            raise HeldoutProtocolError("coordinate result {} does not match the frozen schedule".format(field))
    if raw["status"] not in RESULT_STATUSES:
        raise HeldoutProtocolError("coordinate result status is outside the frozen vocabulary")
    for field in (
        "evaluator_identity_valid", "signature_valid", "ledger_integrity_valid", "hidden_test_isolation_valid",
        "split_isolation_valid", "treatment_isolation_valid", "authority_enforced",
        "authority_decision_receipts_valid",
    ):
        _require_bool(raw[field], field)
    count_fields = (
        "verdict_receipts_required", "verdict_receipts_valid", "effect_receipts_required", "effect_receipts_valid",
        "private_replay_decisions", "private_replay_agreements", "public_replay_decisions", "public_replay_agreements",
        "promoted_candidates", "invalid_promotions", "unauthorized_successful_effects", "receipt_covered_promotions",
        "eligible_attempts", "repeated_dead_end_attempts", "evidence_opportunities", "evidence_using_attempts",
        "authority_challenges", "authority_challenges_valid_denials",
    )
    for field in count_fields:
        _require_int(raw[field], field)
    if raw["status"] == "COMPLETED":
        _require_bool(raw["success"], "success")
    elif raw["success"] is not None:
        raise HeldoutProtocolError("non-completed results must set success to null")
    costs = raw["costs"]
    _require_closed_keys(costs, COST_KEYS, "costs")
    _require_int(costs["tokens"], "costs.tokens")
    attempts = _require_int(costs["candidate_attempts"], "costs.candidate_attempts")
    if attempts > MAX_ATTEMPTS:
        raise HeldoutProtocolError("candidate attempt ceiling exceeded")
    _require_number(costs["evaluator_seconds"], "costs.evaluator_seconds")
    _require_number(costs["wall_time_seconds"], "costs.wall_time_seconds")
    if raw["promoted_candidates"] > attempts or raw["invalid_promotions"] > raw["promoted_candidates"]:
        raise HeldoutProtocolError("promotion counts are inconsistent with attempts")
    if raw["receipt_covered_promotions"] > raw["promoted_candidates"]:
        raise HeldoutProtocolError("receipt coverage exceeds promoted candidates")
    for valid_field, required_field in (
        ("verdict_receipts_valid", "verdict_receipts_required"),
        ("effect_receipts_valid", "effect_receipts_required"),
        ("private_replay_agreements", "private_replay_decisions"),
        ("public_replay_agreements", "public_replay_decisions"),
        ("repeated_dead_end_attempts", "eligible_attempts"),
        ("authority_challenges_valid_denials", "authority_challenges"),
    ):
        if raw[valid_field] > raw[required_field]:
            raise HeldoutProtocolError("{} exceeds {}".format(valid_field, required_field))
    if raw["evidence_using_attempts"] > attempts or (raw["evidence_using_attempts"] and not raw["evidence_opportunities"]):
        raise HeldoutProtocolError("evidence-use counts are inconsistent")
    if raw["eligible_attempts"] > attempts:
        raise HeldoutProtocolError("eligible_attempts exceeds candidate_attempts")
    if raw["evidence_using_attempts"] > raw["evidence_opportunities"]:
        raise HeldoutProtocolError("evidence_using_attempts exceeds evidence_opportunities")
    if coordinate.phase == MAIN_PHASE:
        expected_authority = coordinate.treatment in {"D", "E"}
        if raw["authority_enforced"] != expected_authority:
            raise HeldoutProtocolError("main-arm authority regime was substituted")
        if raw["status"] == "COMPLETED":
            has_valid_promotion = raw["promoted_candidates"] - raw["invalid_promotions"] > 0
            if raw["success"] != has_valid_promotion:
                raise HeldoutProtocolError("main-arm success does not match verified promotion counts")
    else:
        for field in (
            "fixture_digest", "pre_shock_behavior_digest", "rng_state_digest", "accepted_premise_digest",
            "pre_shock_candidate_state_digest", "pre_shock_dependency_graph_digest", "correction_event_digest",
        ):
            if raw[field] != getattr(coordinate, field):
                raise HeldoutProtocolError("shock {} does not match the frozen block".format(field))
        if raw["correction_attempt"] != 6:
            raise HeldoutProtocolError("shock correction must occur after exact attempt 6")
        for field in (
            "shock_exposed", "correction_receipt_valid", "policy_activated_after_correction", "recovered_within_six",
            "recovery_independent", "independent_hidden_fixture_passed",
        ):
            _require_bool(raw[field], field)
        for field in (
            "known_affected_descendants", "marked_stale_descendants", "correctly_stale_descendants",
            "stale_dependent_promotions",
        ):
            _require_int(raw[field], field)
        if raw["correctly_stale_descendants"] > raw["marked_stale_descendants"]:
            raise HeldoutProtocolError("correctly stale count exceeds marked stale count")
        recovery_attempt = raw["recovery_attempt"]
        if raw["recovered_within_six"]:
            if type(recovery_attempt) is not int or not 1 <= recovery_attempt <= MAX_POST_SHOCK_ATTEMPTS:
                raise HeldoutProtocolError("recovered trajectory requires a recovery attempt in 1..6")
        elif recovery_attempt is not None:
            raise HeldoutProtocolError("right-censored non-recovery must use a null recovery_attempt")
        if raw["authority_enforced"] is not True:
            raise HeldoutProtocolError("shock factor must keep the typed authority regime fixed")
        if raw["status"] == "COMPLETED" and raw["shock_exposed"]:
            if attempts < 6:
                raise HeldoutProtocolError("exposed shock trajectories must reach the correction boundary")
            if raw["recovered_within_six"] and attempts != 6 + raw["recovery_attempt"]:
                raise HeldoutProtocolError("shock recovery attempts do not match the recovery clock")
            if not raw["recovered_within_six"] and attempts != MAX_ATTEMPTS:
                raise HeldoutProtocolError("right-censored shock trajectories must consume all six post-shock attempts")
    if raw["status"] == "COMPLETED" and attempts == 0:
        raise HeldoutProtocolError("completed trajectories require at least one candidate attempt")
    return canonical_value(dict(raw))


SIGNED_RESULT_KEYS = (
    "schema_version", "envelope_id", "campaign_id", "protocol_digest", "coordinate_id", "coordinate",
    "coordinate_digest", "profile_digest", "evaluator_digest", "signing_key_id", "receipt_collection_root",
    "ledger_head_digest", "result_digest", "result", "signature",
)


def _decode_ed25519_signature(value: Any) -> bytes:
    if not isinstance(value, str) or not value:
        raise HeldoutProtocolError("signature must be canonical base64url")
    padded = value + "=" * (-len(value) % 4)
    try:
        decoded = base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)
    except (ValueError, UnicodeError) as exc:
        raise HeldoutProtocolError("signature must be canonical base64url") from exc
    canonical = base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=")
    if len(decoded) != 64 or canonical != value:
        raise HeldoutProtocolError("signature must be canonical Ed25519 base64url")
    return decoded


def _verify_evaluator_signature(protocol: FrozenHeldoutProtocol, value: Mapping[str, Any]) -> None:
    if value.get("signing_key_id") != protocol.evaluator_key_id:
        raise HeldoutProtocolError("signed envelope uses an unexpected evaluator key ID")
    unsigned = dict(value)
    signature = unsigned.pop("signature", None)
    try:
        load_public_key(bytes.fromhex(protocol.evaluator_public_key_hex)).verify(
            _decode_ed25519_signature(signature), canonical_bytes(unsigned)
        )
    except HeldoutProtocolError:
        raise
    except Exception as exc:
        raise HeldoutProtocolError("invalid evaluator Ed25519 signature") from exc


def build_signed_result_envelope(
    protocol: FrozenHeldoutProtocol,
    result: Mapping[str, Any],
    signer: Any,
    *,
    receipt_collection_root: str,
    ledger_head_digest: str,
) -> Dict[str, Any]:
    """Build the exact evaluator-signed admission envelope for one result."""

    normalized = validate_result(protocol, result)
    coordinate = protocol.coordinate(normalized["coordinate_id"])
    coordinate_payload = coordinate.to_dict(protocol.digest, protocol.campaign_id)
    receipt_root = validate_sha256(receipt_collection_root, "receipt_collection_root")
    ledger_head = validate_sha256(ledger_head_digest, "ledger_head_digest")
    if getattr(signer, "key_id", None) != protocol.evaluator_key_id or not callable(getattr(signer, "sign_bytes", None)):
        raise HeldoutProtocolError("result signer does not match the frozen evaluator key")
    payload: Dict[str, Any] = {
        "schema_version": SIGNED_RESULT_SCHEMA,
        "campaign_id": protocol.campaign_id,
        "protocol_digest": protocol.digest,
        "coordinate_id": coordinate.coordinate_id,
        "coordinate": coordinate_payload,
        "coordinate_digest": digest_for(coordinate_payload),
        "profile_digest": coordinate.profile_digest,
        "evaluator_digest": protocol.bindings["evaluator_digest"],
        "signing_key_id": protocol.evaluator_key_id,
        "receipt_collection_root": receipt_root,
        "ledger_head_digest": ledger_head,
        "result_digest": digest_for(normalized),
        "result": normalized,
    }
    payload["envelope_id"] = content_id("heldout", payload)
    payload["signature"] = signer.sign_bytes(canonical_bytes(payload))
    return verify_signed_result_envelope(protocol, payload)


def verify_signed_result_envelope(
    protocol: FrozenHeldoutProtocol, envelope: Mapping[str, Any]
) -> Dict[str, Any]:
    _require_closed_keys(envelope, SIGNED_RESULT_KEYS, "signed result envelope")
    if envelope["schema_version"] != SIGNED_RESULT_SCHEMA:
        raise HeldoutProtocolError("signed result envelope schema is unsupported")
    normalized_result = validate_result(protocol, envelope["result"])
    coordinate = protocol.coordinate(normalized_result["coordinate_id"])
    exact_coordinate = coordinate.to_dict(protocol.digest, protocol.campaign_id)
    expected = {
        "campaign_id": protocol.campaign_id,
        "protocol_digest": protocol.digest,
        "coordinate_id": coordinate.coordinate_id,
        "coordinate": exact_coordinate,
        "coordinate_digest": digest_for(exact_coordinate),
        "profile_digest": coordinate.profile_digest,
        "evaluator_digest": protocol.bindings["evaluator_digest"],
        "signing_key_id": protocol.evaluator_key_id,
        "result_digest": digest_for(normalized_result),
    }
    for field, value in expected.items():
        if envelope[field] != value:
            raise HeldoutProtocolError("signed result envelope {} binding mismatch".format(field))
    validate_sha256(envelope["receipt_collection_root"], "receipt_collection_root")
    validate_sha256(envelope["ledger_head_digest"], "ledger_head_digest")
    if envelope["receipt_collection_root"] == GENESIS_HASH or envelope["ledger_head_digest"] == GENESIS_HASH:
        raise HeldoutProtocolError("signed result cannot bind an empty receipt collection or ledger")
    unsigned = dict(envelope)
    supplied_id = unsigned.pop("envelope_id")
    unsigned.pop("signature")
    if supplied_id != content_id("heldout", unsigned):
        raise HeldoutProtocolError("signed result envelope ID is not content-derived")
    _verify_evaluator_signature(protocol, envelope)
    return canonical_value(dict(envelope))


def verify_restoration_receipt(
    protocol: FrozenHeldoutProtocol, receipt: Optional[Mapping[str, Any]]
) -> Tuple[bool, Optional[str]]:
    if receipt is None:
        return False, None
    try:
        receipt_digest = verify_public_restore_receipt(
            receipt,
            bytes.fromhex(protocol.evaluator_public_key_hex),
            expected_key_id=protocol.evaluator_key_id,
        )
    except Exception as exc:
        raise HeldoutProtocolError("restoration receipt signature or schema is invalid") from exc
    if (
        receipt.get("campaign_id") != protocol.campaign_id
        or receipt.get("private_inventory_digest") != protocol.bindings["restore_inventory_digest"]
        or receipt.get("restoration_outcome") != "RESTORED"
        or receipt.get("all_health_checks_passed") is not True
        or receipt.get("smoke_matches_baseline") is not True
        or receipt.get("restored_service_count") != receipt.get("expected_service_count")
        or receipt.get("health_pass_count") != receipt.get("health_check_count")
        or not receipt.get("expected_service_count")
        or not receipt.get("health_check_count")
    ):
        raise HeldoutProtocolError("restoration receipt does not prove the frozen complete restore")
    return True, receipt_digest


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".heldout-", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_canonical_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(text)
    except (OSError, ValueError) as exc:
        raise HeldoutProtocolError("{} is unreadable".format(label)) from exc
    if not isinstance(value, dict) or canonical_json(value) != text:
        raise HeldoutProtocolError("{} is not a canonical object".format(label))
    return value


class HeldoutJournal:
    """Atomic per-coordinate signed records plus a canonical hash-chain index."""

    INDEX_KEYS = ("schema_version", "protocol_digest", "sequence", "head_digest", "entries", "index_digest")
    ENTRY_KEYS = ("sequence", "coordinate_id", "envelope_digest", "record_digest", "filename")
    RECORD_KEYS = (
        "schema_version", "sequence", "protocol_digest", "previous_record_digest", "coordinate_id",
        "envelope_digest", "envelope", "record_digest",
    )

    def __init__(self, path: Union[str, Path], protocol: FrozenHeldoutProtocol) -> None:
        self.path = Path(path)
        if self.path.exists() and not self.path.is_dir():
            raise HeldoutProtocolError("held-out journal path must be an atomic-record directory")
        self.protocol = protocol
        self.records_path = self.path / "records"
        self.index_path = self.path / "index.json"
        self.records_path.mkdir(parents=True, exist_ok=True)
        self._records: List[Dict[str, Any]] = []
        self._by_coordinate: Dict[str, Dict[str, Any]] = {}
        self._orphans: Dict[str, Dict[str, Any]] = {}
        self._load()

    @staticmethod
    def _empty_index(protocol_digest: str) -> Dict[str, Any]:
        unsigned: Dict[str, Any] = {
            "schema_version": JOURNAL_INDEX_SCHEMA,
            "protocol_digest": protocol_digest,
            "sequence": 0,
            "head_digest": GENESIS_HASH,
            "entries": [],
        }
        return {**unsigned, "index_digest": digest_for(unsigned)}

    def _load_index(self) -> Dict[str, Any]:
        if not self.index_path.exists():
            return self._empty_index(self.protocol.digest)
        index = _read_canonical_object(self.index_path, "held-out journal index")
        _require_closed_keys(index, self.INDEX_KEYS, "held-out journal index")
        unsigned = dict(index)
        supplied_digest = unsigned.pop("index_digest")
        if supplied_digest != digest_for(unsigned):
            raise HeldoutProtocolError("held-out journal index digest mismatch")
        if index["schema_version"] != JOURNAL_INDEX_SCHEMA or index["protocol_digest"] != self.protocol.digest:
            raise HeldoutProtocolError("held-out journal index protocol mismatch")
        if not isinstance(index["entries"], list):
            raise HeldoutProtocolError("held-out journal index entries must be an array")
        if type(index["sequence"]) is not int or index["sequence"] != len(index["entries"]):
            raise HeldoutProtocolError("held-out journal index sequence mismatch")
        if index["head_digest"] != GENESIS_HASH:
            validate_sha256(index["head_digest"], "head_digest")
        return index

    def _verify_record(self, record: Mapping[str, Any], *, sequence: int, previous: str) -> Dict[str, Any]:
        _require_closed_keys(record, self.RECORD_KEYS, "held-out journal record")
        if (
            record["schema_version"] != JOURNAL_SCHEMA
            or record["sequence"] != sequence
            or record["protocol_digest"] != self.protocol.digest
            or record["previous_record_digest"] != previous
        ):
            raise HeldoutProtocolError("held-out journal record chain mismatch")
        if sequence > len(self.protocol.coordinates) or record["coordinate_id"] != self.protocol.coordinates[sequence - 1].coordinate_id:
            raise HeldoutProtocolError("held-out journal record is outside frozen execution order")
        envelope = verify_signed_result_envelope(self.protocol, record["envelope"])
        if record["coordinate_id"] != envelope["coordinate_id"] or record["envelope_digest"] != digest_for(envelope):
            raise HeldoutProtocolError("held-out journal envelope binding mismatch")
        unsigned = dict(record)
        supplied_digest = unsigned.pop("record_digest")
        if supplied_digest != digest_for(unsigned):
            raise HeldoutProtocolError("held-out journal record digest mismatch")
        return canonical_value(dict(record))

    def _load(self) -> None:
        index = self._load_index()
        previous = GENESIS_HASH
        referenced = set()
        for sequence, entry in enumerate(index["entries"], start=1):
            _require_closed_keys(entry, self.ENTRY_KEYS, "held-out journal index entry")
            if entry["sequence"] != sequence:
                raise HeldoutProtocolError("held-out journal index entry sequence mismatch")
            filename = entry["filename"]
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise HeldoutProtocolError("held-out journal index filename is unsafe")
            record_path = self.records_path / filename
            record = self._verify_record(
                _read_canonical_object(record_path, "held-out journal record"),
                sequence=sequence,
                previous=previous,
            )
            if any(record[field] != entry[field] for field in ("sequence", "coordinate_id", "envelope_digest", "record_digest")):
                raise HeldoutProtocolError("held-out journal index entry differs from its record")
            if record["coordinate_id"] in self._by_coordinate:
                raise HeldoutProtocolError("held-out journal contains a duplicate coordinate")
            self._records.append(record)
            self._by_coordinate[record["coordinate_id"]] = record["envelope"]
            referenced.add(filename)
            previous = record["record_digest"]
        if previous != index["head_digest"]:
            raise HeldoutProtocolError("held-out journal index head mismatch")
        unreferenced = sorted(path for path in self.records_path.glob("*.json") if path.name not in referenced)
        if len(unreferenced) > 1:
            raise HeldoutProtocolError("multiple orphan journal records require operator quarantine")
        if unreferenced:
            expected_name = "{:06d}-{}.json".format(
                len(self._records) + 1,
                _read_canonical_object(unreferenced[0], "orphan held-out journal record").get("coordinate_id"),
            )
            if unreferenced[0].name != expected_name:
                raise HeldoutProtocolError("orphan journal record filename is not canonical")
            orphan = self._verify_record(
                _read_canonical_object(unreferenced[0], "orphan held-out journal record"),
                sequence=len(self._records) + 1,
                previous=previous,
            )
            self._orphans[orphan["coordinate_id"]] = orphan

    def reload(self) -> None:
        """Reload the complete journal while holding an external transaction lock."""

        self.protocol.validate_current()
        self._reload_locked()

    def _reload_locked(self) -> None:
        self._records.clear()
        self._by_coordinate.clear()
        self._orphans.clear()
        self._load()

    @property
    def envelopes(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(dict(record["envelope"]) for record in self._records)

    @property
    def results(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(dict(record["envelope"]["result"]) for record in self._records)

    @property
    def completed_coordinate_ids(self) -> frozenset:
        return frozenset(self._by_coordinate)

    @property
    def unindexed_coordinate_ids(self) -> frozenset:
        return frozenset(self._orphans)

    def envelope(self, coordinate_id: str) -> Dict[str, Any]:
        try:
            return dict(self._by_coordinate[coordinate_id])
        except KeyError as exc:
            raise HeldoutProtocolError("coordinate has no indexed signed result") from exc

    def append(self, raw_envelope: Mapping[str, Any]) -> Dict[str, Any]:
        """Append exactly once under a cross-process journal transaction."""

        with _exclusive_path_lock(self.path / ".journal.lock"):
            self.reload()
            return self._append_locked(raw_envelope)

    def _append_locked(self, raw_envelope: Mapping[str, Any]) -> Dict[str, Any]:
        envelope = verify_signed_result_envelope(self.protocol, raw_envelope)
        coordinate_id = envelope["coordinate_id"]
        if coordinate_id in self._by_coordinate:
            raise HeldoutProtocolError("coordinate already has a journaled outcome")
        if self._orphans and coordinate_id not in self._orphans:
            raise HeldoutProtocolError("an unindexed signed record must be reconciled before another append")
        expected_coordinate = self.protocol.coordinates[len(self._records)].coordinate_id
        if coordinate_id != expected_coordinate:
            raise HeldoutProtocolError("signed result is outside frozen execution order")
        previous = self._records[-1]["record_digest"] if self._records else GENESIS_HASH
        sequence = len(self._records) + 1
        filename = "{:06d}-{}.json".format(sequence, coordinate_id)
        record: Dict[str, Any] = {
            "schema_version": JOURNAL_SCHEMA,
            "sequence": sequence,
            "protocol_digest": self.protocol.digest,
            "previous_record_digest": previous,
            "coordinate_id": coordinate_id,
            "envelope_digest": digest_for(envelope),
            "envelope": envelope,
        }
        record["record_digest"] = digest_for(record)
        record_path = self.records_path / filename
        orphan = self._orphans.get(coordinate_id)
        if orphan is not None:
            if orphan != record or record_path.name != filename:
                raise HeldoutProtocolError("orphan record conflicts with reconciled signed outcome")
        elif record_path.exists():
            raise HeldoutProtocolError("unexpected per-coordinate record already exists")
        else:
            _atomic_write_json(record_path, record)

        old_index = self._load_index()
        if old_index["sequence"] != len(self._records) or old_index["head_digest"] != previous:
            raise HeldoutProtocolError("journal index changed during append")
        entry = {
            "sequence": sequence,
            "coordinate_id": coordinate_id,
            "envelope_digest": record["envelope_digest"],
            "record_digest": record["record_digest"],
            "filename": filename,
        }
        unsigned_index = {
            "schema_version": JOURNAL_INDEX_SCHEMA,
            "protocol_digest": self.protocol.digest,
            "sequence": sequence,
            "head_digest": record["record_digest"],
            "entries": list(old_index["entries"]) + [entry],
        }
        _atomic_write_json(self.index_path, {**unsigned_index, "index_digest": digest_for(unsigned_index)})
        self._records.append(record)
        self._by_coordinate[coordinate_id] = envelope
        self._orphans.pop(coordinate_id, None)
        return dict(record)


def validate_coordinate_operation_state(
    protocol: FrozenHeldoutProtocol,
    coordinate_id: str,
    raw: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate the closed durable operation-state contract and proof invariants."""

    keys = (
        "schema_version", "campaign_id", "protocol_digest", "coordinate_id", "idempotency_key",
        "state", "revision", "previous_state_digest", "envelope_digest", "reconciliation_digest",
        "state_digest",
    )
    _require_closed_keys(raw, keys, "coordinate operation state")
    value = dict(raw)
    expected_idempotency = content_id(
        "idem", {"protocol_digest": protocol.digest, "coordinate_id": coordinate_id}
    )
    if (
        value["schema_version"] != OPERATION_SCHEMA
        or value["campaign_id"] != protocol.campaign_id
        or value["protocol_digest"] != protocol.digest
        or value["coordinate_id"] != coordinate_id
        or value["idempotency_key"] != expected_idempotency
        or value["state"] not in CoordinateOperationStore.STATES
        or type(value["revision"]) is not int
        or value["revision"] < 1
    ):
        raise HeldoutProtocolError("coordinate operation state binding mismatch")
    previous = value["previous_state_digest"]
    if value["revision"] == 1:
        if previous != GENESIS_HASH:
            raise HeldoutProtocolError("initial operation state must begin at the genesis digest")
    elif previous == GENESIS_HASH:
        raise HeldoutProtocolError("revised operation state must bind its predecessor")
    else:
        validate_sha256(previous, "previous_state_digest")
    envelope_digest = value["envelope_digest"]
    reconciliation_digest = value["reconciliation_digest"]
    if envelope_digest is not None:
        validate_sha256(envelope_digest, "envelope_digest")
    if reconciliation_digest is not None:
        validate_sha256(reconciliation_digest, "reconciliation_digest")
    state = value["state"]
    if state == "DISPATCHING" and (envelope_digest is not None or reconciliation_digest is not None):
        raise HeldoutProtocolError("dispatching operation cannot carry a terminal proof")
    if state == "COMPLETED" and envelope_digest is None:
        raise HeldoutProtocolError("completed operation state requires a signed envelope digest")
    if state in {"NOT_EXECUTED", "QUARANTINED"} and (
        envelope_digest is not None or reconciliation_digest is None
    ):
        raise HeldoutProtocolError("reconciled operation state has invalid proof fields")
    supplied_digest = value["state_digest"]
    validate_sha256(supplied_digest, "state_digest")
    if supplied_digest != digest_for({key: item for key, item in value.items() if key != "state_digest"}):
        raise HeldoutProtocolError("coordinate operation state digest mismatch")
    return canonical_value(value)


class CoordinateOperationStore:
    """Durable per-coordinate dispatch state with one stable idempotency key."""

    KEYS = (
        "schema_version", "campaign_id", "protocol_digest", "coordinate_id", "idempotency_key", "state",
        "revision", "previous_state_digest", "envelope_digest", "reconciliation_digest", "state_digest",
    )
    STATES = frozenset({"DISPATCHING", "NOT_EXECUTED", "COMPLETED", "QUARANTINED"})

    def __init__(self, path: Union[str, Path], protocol: FrozenHeldoutProtocol) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.protocol = protocol

    def _path(self, coordinate_id: str) -> Path:
        self.protocol.coordinate(coordinate_id)
        return self.path / (coordinate_id + ".json")

    def idempotency_key(self, coordinate_id: str) -> str:
        self.protocol.coordinate(coordinate_id)
        return content_id(
            "idem", {"protocol_digest": self.protocol.digest, "coordinate_id": coordinate_id}
        )

    def load(self, coordinate_id: str) -> Optional[Dict[str, Any]]:
        path = self._path(coordinate_id)
        if not path.exists():
            return None
        value = _read_canonical_object(path, "coordinate operation state")
        return validate_coordinate_operation_state(self.protocol, coordinate_id, value)

    def transition(
        self,
        coordinate_id: str,
        state: str,
        *,
        expected_state_digest: Optional[str],
        envelope_digest: Optional[str] = None,
        reconciliation_digest: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.protocol.validate_current()
        with _exclusive_path_lock(self.path / ".locks" / (coordinate_id + ".lock")):
            return self._transition_locked(
                coordinate_id,
                state,
                expected_state_digest=expected_state_digest,
                envelope_digest=envelope_digest,
                reconciliation_digest=reconciliation_digest,
            )

    def _transition_locked(
        self,
        coordinate_id: str,
        state: str,
        *,
        expected_state_digest: Optional[str],
        envelope_digest: Optional[str] = None,
        reconciliation_digest: Optional[str] = None,
    ) -> Dict[str, Any]:
        if state not in self.STATES:
            raise HeldoutProtocolError("coordinate operation transition state is invalid")
        current = self.load(coordinate_id)
        actual_digest = current["state_digest"] if current else None
        if actual_digest != expected_state_digest:
            raise HeldoutProtocolError("coordinate operation state changed before transition")
        if envelope_digest is not None:
            envelope_digest = validate_sha256(envelope_digest, "envelope_digest")
        if reconciliation_digest is not None:
            reconciliation_digest = validate_sha256(reconciliation_digest, "reconciliation_digest")
        if state == "COMPLETED" and envelope_digest is None:
            raise HeldoutProtocolError("completed operation state requires a signed envelope digest")
        if state in {"NOT_EXECUTED", "QUARANTINED"} and reconciliation_digest is None:
            raise HeldoutProtocolError("reconciled operation state requires a signed reconciliation digest")
        if state == "DISPATCHING" and (envelope_digest is not None or reconciliation_digest is not None):
            raise HeldoutProtocolError("dispatching operation cannot carry a terminal proof")
        if current is None and state not in {"DISPATCHING", "COMPLETED"}:
            raise HeldoutProtocolError("initial operation state must dispatch or import a signed completion")
        unsigned: Dict[str, Any] = {
            "schema_version": OPERATION_SCHEMA,
            "campaign_id": self.protocol.campaign_id,
            "protocol_digest": self.protocol.digest,
            "coordinate_id": coordinate_id,
            "idempotency_key": self.idempotency_key(coordinate_id),
            "state": state,
            "revision": 1 if current is None else current["revision"] + 1,
            "previous_state_digest": GENESIS_HASH if current is None else current["state_digest"],
            "envelope_digest": envelope_digest,
            "reconciliation_digest": reconciliation_digest,
        }
        value = {**unsigned, "state_digest": digest_for(unsigned)}
        _atomic_write_json(self._path(coordinate_id), value)
        return value

    def begin(self, coordinate_id: str) -> Dict[str, Any]:
        self.protocol.validate_current()
        with _exclusive_path_lock(self.path / ".locks" / (coordinate_id + ".lock")):
            current = self.load(coordinate_id)
            if current is not None and current["state"] != "NOT_EXECUTED":
                raise HeldoutProtocolError("coordinate cannot dispatch from its durable operation state")
            return self._transition_locked(
                coordinate_id,
                "DISPATCHING",
                expected_state_digest=current["state_digest"] if current else None,
            )


RECONCILIATION_KEYS = (
    "schema_version", "envelope_id", "campaign_id", "protocol_digest", "coordinate_id", "idempotency_key",
    "operation_state_digest", "decision", "result_envelope_digest", "receipt_collection_root",
    "ledger_head_digest", "signing_key_id", "signature",
)


def build_signed_reconciliation(
    protocol: FrozenHeldoutProtocol,
    operation_state: Mapping[str, Any],
    signer: Any,
    *,
    decision: str,
    result_envelope: Optional[Mapping[str, Any]],
    receipt_collection_root: str,
    ledger_head_digest: str,
) -> Dict[str, Any]:
    if decision not in {"COMPLETED", "NOT_EXECUTED", "UNKNOWN"}:
        raise HeldoutProtocolError("reconciliation decision is invalid")
    if getattr(signer, "key_id", None) != protocol.evaluator_key_id:
        raise HeldoutProtocolError("reconciliation signer differs from frozen evaluator")
    coordinate_id = operation_state.get("coordinate_id")
    protocol.coordinate(coordinate_id)
    result_digest: Optional[str] = None
    if decision == "COMPLETED":
        if result_envelope is None:
            raise HeldoutProtocolError("completed reconciliation requires a signed result envelope")
        result_digest = digest_for(verify_signed_result_envelope(protocol, result_envelope))
    elif result_envelope is not None:
        raise HeldoutProtocolError("non-completed reconciliation cannot carry a result")
    payload: Dict[str, Any] = {
        "schema_version": RECONCILIATION_SCHEMA,
        "campaign_id": protocol.campaign_id,
        "protocol_digest": protocol.digest,
        "coordinate_id": coordinate_id,
        "idempotency_key": operation_state.get("idempotency_key"),
        "operation_state_digest": operation_state.get("state_digest"),
        "decision": decision,
        "result_envelope_digest": result_digest,
        "receipt_collection_root": validate_sha256(receipt_collection_root, "receipt_collection_root"),
        "ledger_head_digest": validate_sha256(ledger_head_digest, "ledger_head_digest"),
        "signing_key_id": protocol.evaluator_key_id,
    }
    payload["envelope_id"] = content_id("reconcile", payload)
    payload["signature"] = signer.sign_bytes(canonical_bytes(payload))
    return verify_signed_reconciliation(protocol, operation_state, payload, result_envelope)


def verify_signed_reconciliation(
    protocol: FrozenHeldoutProtocol,
    operation_state: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
    result_envelope: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    _require_closed_keys(reconciliation, RECONCILIATION_KEYS, "signed reconciliation")
    if reconciliation["schema_version"] != RECONCILIATION_SCHEMA:
        raise HeldoutProtocolError("signed reconciliation schema is unsupported")
    expected = {
        "campaign_id": protocol.campaign_id,
        "protocol_digest": protocol.digest,
        "coordinate_id": operation_state.get("coordinate_id"),
        "idempotency_key": operation_state.get("idempotency_key"),
        "operation_state_digest": operation_state.get("state_digest"),
        "signing_key_id": protocol.evaluator_key_id,
    }
    for field, value in expected.items():
        if reconciliation[field] != value:
            raise HeldoutProtocolError("signed reconciliation {} mismatch".format(field))
    if reconciliation["decision"] not in {"COMPLETED", "NOT_EXECUTED", "UNKNOWN"}:
        raise HeldoutProtocolError("signed reconciliation decision is invalid")
    validate_sha256(reconciliation["receipt_collection_root"], "receipt_collection_root")
    validate_sha256(reconciliation["ledger_head_digest"], "ledger_head_digest")
    if reconciliation["decision"] == "COMPLETED":
        if result_envelope is None:
            raise HeldoutProtocolError("completed reconciliation result is absent")
        verified_result = verify_signed_result_envelope(protocol, result_envelope)
        if verified_result["coordinate_id"] != reconciliation["coordinate_id"]:
            raise HeldoutProtocolError("reconciled result belongs to another coordinate")
        if reconciliation["result_envelope_digest"] != digest_for(verified_result):
            raise HeldoutProtocolError("reconciled result digest mismatch")
    elif result_envelope is not None or reconciliation["result_envelope_digest"] is not None:
        raise HeldoutProtocolError("non-completed reconciliation carries a result")
    unsigned = dict(reconciliation)
    supplied_id = unsigned.pop("envelope_id")
    unsigned.pop("signature")
    if supplied_id != content_id("reconcile", unsigned):
        raise HeldoutProtocolError("signed reconciliation ID is not content-derived")
    _verify_evaluator_signature(protocol, reconciliation)
    return canonical_value(dict(reconciliation))


def _observed_integrity_failure(result: Mapping[str, Any]) -> bool:
    boolean_checks = (
        "evaluator_identity_valid", "signature_valid", "ledger_integrity_valid", "hidden_test_isolation_valid",
        "split_isolation_valid", "treatment_isolation_valid", "authority_decision_receipts_valid",
    )
    if any(not result[field] for field in boolean_checks):
        return True
    if result["verdict_receipts_valid"] != result["verdict_receipts_required"]:
        return True
    if result["effect_receipts_valid"] != result["effect_receipts_required"]:
        return True
    if result["private_replay_decisions"] == 0 or result["private_replay_agreements"] != result["private_replay_decisions"]:
        return True
    if result["public_replay_decisions"] == 0 or result["public_replay_agreements"] != result["public_replay_decisions"]:
        return True
    if result["invalid_promotions"] or result["unauthorized_successful_effects"]:
        return True
    if result["receipt_covered_promotions"] != result["promoted_candidates"]:
        return True
    if result["phase"] == SHOCK_PHASE:
        if not result["shock_exposed"] or not result["correction_receipt_valid"] or not result["policy_activated_after_correction"]:
            return True
        if result["recovered_within_six"] and (
            not result["recovery_independent"] or not result["independent_hidden_fixture_passed"]
        ):
            return True
        if result["treatment"] == "dependency-aware" and (
            result["known_affected_descendants"] == 0
            or result["marked_stale_descendants"] != result["known_affected_descendants"]
            or result["correctly_stale_descendants"] != result["known_affected_descendants"]
            or result["stale_dependent_promotions"] != 0
        ):
            return True
    return False


CoordinateRunner = Callable[[Mapping[str, Any]], Mapping[str, Any]]
CoordinateResultVerifier = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]
CoordinateReconciler = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


def run_pending_coordinates(
    protocol: FrozenHeldoutProtocol,
    journal: HeldoutJournal,
    operation_store: CoordinateOperationStore,
    runner: CoordinateRunner,
    result_verifier: CoordinateResultVerifier,
    reconciler: CoordinateReconciler,
    *,
    halt_on_integrity_failure: bool = True,
) -> Dict[str, Any]:
    """Serialize recovery, dispatch, journaling, and operation transitions."""

    protocol.validate_current()
    if journal.protocol.digest != protocol.digest or operation_store.protocol.digest != protocol.digest:
        raise HeldoutProtocolError("journal and runner protocol differ")
    with _exclusive_path_lock(journal.path / ".scheduler.lock"):
        validated_protocols = set()
        for current_protocol in (protocol, journal.protocol, operation_store.protocol):
            identity = id(current_protocol)
            if identity not in validated_protocols:
                current_protocol.validate_current()
                validated_protocols.add(identity)
        journal._reload_locked()
        return _run_pending_coordinates_locked(
            protocol,
            journal,
            operation_store,
            runner,
            result_verifier,
            reconciler,
            halt_on_integrity_failure=halt_on_integrity_failure,
        )


def _run_pending_coordinates_locked(
    protocol: FrozenHeldoutProtocol,
    journal: HeldoutJournal,
    operation_store: CoordinateOperationStore,
    runner: CoordinateRunner,
    result_verifier: CoordinateResultVerifier,
    reconciler: CoordinateReconciler,
    *,
    halt_on_integrity_failure: bool = True,
) -> Dict[str, Any]:
    """Run pending coordinates through independent runtime and verifier hooks.

    ``runner`` may operate the candidate generator, but its output is never
    journaled directly.  ``result_verifier`` is the evaluator-owned boundary
    which must authenticate signatures and receipts, perform ledger replays,
    and return the closed result facts.  Structural validation here cannot
    replace those cryptographic and semantic checks.
    """

    started = 0
    skipped = 0
    reconciled = 0
    for coordinate in protocol.coordinates:
        protocol.validate_current()
        coordinate_input = coordinate.to_dict(protocol.digest, protocol.campaign_id)
        coordinate_input["idempotency_key"] = operation_store.idempotency_key(coordinate.coordinate_id)
        operation = operation_store.load(coordinate.coordinate_id)
        if coordinate.coordinate_id in journal.completed_coordinate_ids:
            envelope = journal.envelope(coordinate.coordinate_id)
            envelope_digest = digest_for(envelope)
            if operation is None or operation["state"] != "COMPLETED":
                operation_store.transition(
                    coordinate.coordinate_id,
                    "COMPLETED",
                    expected_state_digest=operation["state_digest"] if operation else None,
                    envelope_digest=envelope_digest,
                )
            elif operation["envelope_digest"] != envelope_digest:
                raise HeldoutProtocolError("completed operation differs from indexed signed result")
            skipped += 1
            continue

        has_unindexed_record = coordinate.coordinate_id in journal.unindexed_coordinate_ids
        if has_unindexed_record and operation is None:
            raise HeldoutProtocolError(
                "unindexed signed result lacks durable operation state; operator reconciliation required"
            )

        if operation is not None and operation["state"] in {"DISPATCHING", "COMPLETED", "QUARANTINED"}:
            response = reconciler(coordinate_input, dict(operation))
            _require_closed_keys(
                response, ("reconciliation_envelope", "result_envelope"), "reconciliation response"
            )
            reconciliation = verify_signed_reconciliation(
                protocol,
                operation,
                response["reconciliation_envelope"],
                response["result_envelope"],
            )
            reconciliation_digest = digest_for(reconciliation)
            reconciled += 1
            if reconciliation["decision"] == "UNKNOWN":
                operation_store.transition(
                    coordinate.coordinate_id,
                    "QUARANTINED",
                    expected_state_digest=operation["state_digest"],
                    reconciliation_digest=reconciliation_digest,
                )
                raise HeldoutProtocolError(
                    "external-effect outcome remains unknown; coordinate quarantined without rerun"
                )
            if reconciliation["decision"] == "COMPLETED":
                record = journal.append(response["result_envelope"])
                result = record["envelope"]["result"]
                operation_store.transition(
                    coordinate.coordinate_id,
                    "COMPLETED",
                    expected_state_digest=operation["state_digest"],
                    envelope_digest=record["envelope_digest"],
                    reconciliation_digest=reconciliation_digest,
                )
                if halt_on_integrity_failure and _observed_integrity_failure(result):
                    raise HeldoutProtocolError("reconciled outcome has a hard-integrity failure")
                continue
            if has_unindexed_record:
                operation_store.transition(
                    coordinate.coordinate_id,
                    "QUARANTINED",
                    expected_state_digest=operation["state_digest"],
                    reconciliation_digest=reconciliation_digest,
                )
                raise HeldoutProtocolError(
                    "signed orphan conflicts with NOT_EXECUTED reconciliation; coordinate quarantined"
                )
            operation = operation_store.transition(
                coordinate.coordinate_id,
                "NOT_EXECUTED",
                expected_state_digest=operation["state_digest"],
                reconciliation_digest=reconciliation_digest,
            )

        operation = operation_store.begin(coordinate.coordinate_id)
        raw = runner(coordinate_input)
        signed_envelope = result_verifier(coordinate_input, raw)
        record = journal.append(signed_envelope)
        result = record["envelope"]["result"]
        operation_store.transition(
            coordinate.coordinate_id,
            "COMPLETED",
            expected_state_digest=operation["state_digest"],
            envelope_digest=record["envelope_digest"],
        )
        started += 1
        if halt_on_integrity_failure and _observed_integrity_failure(result):
            raise HeldoutProtocolError("observed hard-integrity failure; campaign halted after journaling evidence")
    return {
        "planned": len(protocol.coordinates),
        "started": started,
        "resumed": skipped,
        "reconciled": reconciled,
        "journaled": len(journal.results),
    }


def _outcome_map(protocol: FrozenHeldoutProtocol, envelopes: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    mapped: Dict[str, Dict[str, Any]] = {}
    for raw in envelopes:
        result = verify_signed_result_envelope(protocol, raw)["result"]
        if result["coordinate_id"] in mapped:
            raise HeldoutProtocolError("duplicate coordinate outcome")
        mapped[result["coordinate_id"]] = result
    return mapped


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise HeldoutProtocolError("cannot compute a percentile from no values")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _bootstrap_rng(protocol: FrozenHeldoutProtocol, label: str) -> random.Random:
    derived = digest_for({"bootstrap_seed": protocol.bootstrap_seed, "label": label, "method": "paired-block-percentile-v1"})
    return random.Random(int(derived[:16], 16))


def _paired_mean_difference(
    protocol: FrozenHeldoutProtocol,
    label: str,
    pairs: Sequence[Tuple[float, float]],
) -> Dict[str, Any]:
    if not pairs:
        return {"estimable": False, "reason": "INTERVAL_NONCOMPUTABLE"}
    differences = [left - right for left, right in pairs]
    point = sum(differences) / len(differences)
    rng = _bootstrap_rng(protocol, label)
    draws: List[float] = []
    for _ in range(protocol.bootstrap_replicates):
        draws.append(sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences))
    return {
        "estimable": True,
        "reason": "COMPLETE",
        "blocks": len(pairs),
        "point": point,
        "lower_95": _percentile(draws, 0.025),
        "upper_95": _percentile(draws, 0.975),
    }


def _paired_rate_difference(
    protocol: FrozenHeldoutProtocol,
    label: str,
    pairs: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]],
) -> Dict[str, Any]:
    if not pairs:
        return {"estimable": False, "reason": "INTERVAL_NONCOMPUTABLE"}

    def rate(indexes: Sequence[int]) -> Optional[float]:
        left_num = sum(pairs[index][0][0] for index in indexes)
        left_den = sum(pairs[index][0][1] for index in indexes)
        right_num = sum(pairs[index][1][0] for index in indexes)
        right_den = sum(pairs[index][1][1] for index in indexes)
        if left_den == 0 or right_den == 0:
            return None
        return left_num / left_den - right_num / right_den

    indexes = list(range(len(pairs)))
    point = rate(indexes)
    if point is None:
        return {"estimable": False, "reason": "ZERO_DENOMINATOR"}
    rng = _bootstrap_rng(protocol, label)
    draws: List[float] = []
    for _ in range(protocol.bootstrap_replicates):
        sampled = [rng.randrange(len(pairs)) for _ in pairs]
        value = rate(sampled)
        if value is None:
            return {"estimable": False, "reason": "INTERVAL_NONCOMPUTABLE"}
        draws.append(value)
    left_num = sum(pair[0][0] for pair in pairs)
    left_den = sum(pair[0][1] for pair in pairs)
    right_num = sum(pair[1][0] for pair in pairs)
    right_den = sum(pair[1][1] for pair in pairs)
    return {
        "estimable": True,
        "reason": "COMPLETE",
        "blocks": len(pairs),
        "point": point,
        "left_rate": left_num / left_den,
        "right_rate": right_num / right_den,
        "lower_95": _percentile(draws, 0.025),
        "upper_95": _percentile(draws, 0.975),
    }


def _coordinate_status_reason(results: Sequence[Mapping[str, Any]]) -> Optional[str]:
    if any(result["status"] == "BUDGET_EXHAUSTED" for result in results):
        return "BUDGET_EXHAUSTED"
    if any(result["status"] != "COMPLETED" for result in results):
        return "MISSING_BLOCK"
    return None


def _main_pairs(
    protocol: FrozenHeldoutProtocol,
    outcomes: Mapping[str, Mapping[str, Any]],
    left_arm: str,
    right_arm: str,
) -> Tuple[Optional[List[Tuple[Mapping[str, Any], Mapping[str, Any]]]], str]:
    pairs: List[Tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    all_seen: List[Mapping[str, Any]] = []
    for task_id in sorted(protocol.heldout_task_ids):
        for seed in sorted(protocol.seeds):
            block: Dict[str, Mapping[str, Any]] = {}
            for coordinate in protocol.coordinates:
                if coordinate.phase == MAIN_PHASE and coordinate.task_id == task_id and coordinate.seed == seed and coordinate.treatment in {left_arm, right_arm}:
                    result = outcomes.get(coordinate.coordinate_id)
                    if result is not None:
                        block[coordinate.treatment] = result
                        all_seen.append(result)
            if set(block) != {left_arm, right_arm}:
                return None, "MISSING_BLOCK"
            pairs.append((block[left_arm], block[right_arm]))
    reason = _coordinate_status_reason(all_seen)
    return (None, reason) if reason else (pairs, "COMPLETE")


def _main_success_comparison(protocol: FrozenHeldoutProtocol, outcomes: Mapping[str, Mapping[str, Any]], left: str, right: str) -> Dict[str, Any]:
    pairs, reason = _main_pairs(protocol, outcomes, left, right)
    if pairs is None:
        return {"estimable": False, "reason": reason}
    return _paired_mean_difference(
        protocol,
        "success:{}-{}".format(left, right),
        [(1.0 if lhs["success"] else 0.0, 1.0 if rhs["success"] else 0.0) for lhs, rhs in pairs],
    )


def _main_dead_end_comparison(protocol: FrozenHeldoutProtocol, outcomes: Mapping[str, Mapping[str, Any]], left: str, right: str) -> Dict[str, Any]:
    pairs, reason = _main_pairs(protocol, outcomes, left, right)
    if pairs is None:
        return {"estimable": False, "reason": reason}
    return _paired_rate_difference(
        protocol,
        "dead-end:{}-{}".format(left, right),
        [
            (
                (lhs["repeated_dead_end_attempts"], lhs["eligible_attempts"]),
                (rhs["repeated_dead_end_attempts"], rhs["eligible_attempts"]),
            )
            for lhs, rhs in pairs
        ],
    )


def _evidence_use_available(outcomes: Mapping[str, Mapping[str, Any]], arms: Iterable[str]) -> bool:
    selected = [result for result in outcomes.values() if result["phase"] == MAIN_PHASE and result["treatment"] in set(arms)]
    return sum(result["evidence_opportunities"] for result in selected) > 0 and sum(
        result["evidence_using_attempts"] for result in selected
    ) > 0


def _authority_challenges_pass(outcomes: Mapping[str, Mapping[str, Any]], arm: str) -> Tuple[Optional[bool], str]:
    selected = [
        result for result in outcomes.values()
        if result["phase"] == MAIN_PHASE and result["treatment"] == arm and result["status"] == "COMPLETED"
    ]
    required = sum(result["authority_challenges"] for result in selected)
    if required == 0:
        return None, "ZERO_DENOMINATOR"
    valid = sum(result["authority_challenges_valid_denials"] for result in selected)
    return valid == required, "COMPLETE"


def _treatment_has_valid_promotion(
    outcomes: Mapping[str, Mapping[str, Any]], arm: str
) -> bool:
    selected = [
        result for result in outcomes.values()
        if result["phase"] == MAIN_PHASE and result["treatment"] == arm and result["status"] == "COMPLETED"
    ]
    return sum(
        result["promoted_candidates"] - result["invalid_promotions"] for result in selected
    ) > 0


def _has_shock_block_mismatch(
    protocol: FrozenHeldoutProtocol, outcomes: Mapping[str, Mapping[str, Any]]
) -> bool:
    fields = (
        "fixture_digest", "pre_shock_behavior_digest", "rng_state_digest", "accepted_premise_digest",
        "pre_shock_candidate_state_digest", "pre_shock_dependency_graph_digest", "correction_event_digest",
        "correction_attempt", "shock_exposed", "known_affected_descendants",
    )
    for task_id in protocol.shock_task_ids:
        for seed in protocol.seeds:
            block = [
                result for result in outcomes.values()
                if result["phase"] == SHOCK_PHASE and result["task_id"] == task_id and result["seed"] == seed
            ]
            if len(block) == len(SHOCK_POLICIES) and len(
                {digest_for({field: result[field] for field in fields}) for result in block}
            ) != 1:
                return True
    return False


def _shock_analysis(protocol: FrozenHeldoutProtocol, outcomes: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    blocks: List[Dict[str, Mapping[str, Any]]] = []
    all_seen: List[Mapping[str, Any]] = []
    for task_id in sorted(protocol.shock_task_ids):
        for seed in sorted(protocol.seeds):
            block: Dict[str, Mapping[str, Any]] = {}
            for coordinate in protocol.coordinates:
                if coordinate.phase == SHOCK_PHASE and coordinate.task_id == task_id and coordinate.seed == seed:
                    result = outcomes.get(coordinate.coordinate_id)
                    if result is not None:
                        block[coordinate.treatment] = result
                        all_seen.append(result)
            if set(block) != set(SHOCK_POLICIES):
                return {"estimable": False, "reason": "MISSING_BLOCK"}
            matched_fields = (
                "fixture_digest", "pre_shock_behavior_digest", "rng_state_digest", "accepted_premise_digest",
                "pre_shock_candidate_state_digest", "pre_shock_dependency_graph_digest", "correction_event_digest",
                "correction_attempt", "shock_exposed", "known_affected_descendants",
            )
            if len({digest_for({field: result[field] for field in matched_fields}) for result in block.values()}) != 1:
                return {"estimable": False, "reason": "NO_SHOCK_EXPOSURE", "matched_block_valid": False}
            blocks.append(block)
    status_reason = _coordinate_status_reason(all_seen)
    if status_reason:
        return {"estimable": False, "reason": status_reason}
    if any(not result["shock_exposed"] for result in all_seen):
        return {"estimable": False, "reason": "NO_SHOCK_EXPOSURE"}

    comparisons: Dict[str, Any] = {}
    for control in ("naive-reuse", "full-restart"):
        recovery_pairs = []
        time_pairs = []
        for block in blocks:
            dependency = block["dependency-aware"]
            control_result = block[control]

            def recovered(result: Mapping[str, Any]) -> bool:
                return bool(
                    result["recovered_within_six"]
                    and result["recovery_independent"]
                    and result["independent_hidden_fixture_passed"]
                )

            dep_recovered = recovered(dependency)
            control_recovered = recovered(control_result)
            recovery_pairs.append((float(dep_recovered), float(control_recovered)))
            time_pairs.append(
                (
                    float(dependency["recovery_attempt"] if dep_recovered else MAX_POST_SHOCK_ATTEMPTS),
                    float(control_result["recovery_attempt"] if control_recovered else MAX_POST_SHOCK_ATTEMPTS),
                )
            )
        comparisons[control] = {
            "recovery_within_six_difference": _paired_mean_difference(
                protocol, "shock-recovery:dependency-aware-{}".format(control), recovery_pairs
            ),
            "restricted_mean_time_difference": _paired_mean_difference(
                protocol, "shock-rmst:dependency-aware-{}".format(control), time_pairs
            ),
        }
    dependency_results = [block["dependency-aware"] for block in blocks]
    known = sum(result["known_affected_descendants"] for result in dependency_results)
    marked = sum(result["marked_stale_descendants"] for result in dependency_results)
    correct = sum(result["correctly_stale_descendants"] for result in dependency_results)
    if known == 0 or marked == 0:
        return {"estimable": False, "reason": "ZERO_DENOMINATOR"}
    precision = correct / marked
    recall = correct / known
    return {
        "estimable": True,
        "reason": "COMPLETE",
        "blocks": len(blocks),
        "invalidation_precision": precision,
        "invalidation_recall": recall,
        "stale_dependent_promotions": sum(result["stale_dependent_promotions"] for result in dependency_results),
        "comparisons": comparisons,
    }


def _cost_analysis(protocol: FrozenHeldoutProtocol, outcomes: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    pairs, reason = _main_pairs(protocol, outcomes, "E", "F")
    if pairs is None:
        return {"estimable": False, "reason": reason}
    measures: Dict[str, Any] = {}
    for measure in COST_KEYS:
        values = [(float(lhs["costs"][measure]), float(rhs["costs"][measure])) for lhs, rhs in pairs]
        if any(left < 0 or right < 0 for left, right in values):
            return {"estimable": False, "reason": "MISSING_COST_MEASURE"}
        comparison = _paired_mean_difference(protocol, "cost:{}:E-F".format(measure), values)
        left_mean = sum(left for left, _ in values) / len(values)
        right_mean = sum(right for _, right in values) / len(values)
        comparison["left_mean"] = left_mean
        comparison["right_mean"] = right_mean
        comparison["relative_overhead"] = None if right_mean == 0 else (left_mean - right_mean) / right_mean
        measures[measure] = comparison
    if measures["wall_time_seconds"]["relative_overhead"] is None:
        return {"estimable": False, "reason": "MISSING_COST_MEASURE", "measures": measures}
    return {"estimable": True, "reason": "COMPLETE", "measures": measures}


def _estimability(estimable: bool, reason: str) -> Dict[str, str]:
    if reason not in ESTIMABILITY_REASONS:
        raise HeldoutProtocolError("unknown estimability reason")
    return {"status": "ESTIMABLE" if estimable else "UNEVALUATED", "reason": reason}


def _arm_aggregate(protocol: FrozenHeldoutProtocol, outcomes: Mapping[str, Mapping[str, Any]], arm: str) -> Dict[str, Any]:
    selected = [result for result in outcomes.values() if result["phase"] == MAIN_PHASE and result["treatment"] == arm]
    completed = [result for result in selected if result["status"] == "COMPLETED"]
    promoted = sum(result["promoted_candidates"] for result in completed)
    eligible = sum(result["eligible_attempts"] for result in completed)
    return {
        "planned_trajectories": len(protocol.heldout_task_ids) * len(protocol.seeds),
        "completed_trajectories": len(completed),
        "success_rate": None if not completed else sum(bool(result["success"]) for result in completed) / len(completed),
        "invalid_promotion_rate": None if promoted == 0 else sum(result["invalid_promotions"] for result in completed) / promoted,
        "receipt_coverage": None if promoted == 0 else sum(result["receipt_covered_promotions"] for result in completed) / promoted,
        "repeated_dead_end_rate": None if eligible == 0 else sum(result["repeated_dead_end_attempts"] for result in completed) / eligible,
        "candidate_attempts": sum(result["costs"]["candidate_attempts"] for result in completed),
        "tokens": sum(result["costs"]["tokens"] for result in completed),
        "evaluator_seconds": sum(float(result["costs"]["evaluator_seconds"]) for result in completed),
        "wall_time_seconds": sum(float(result["costs"]["wall_time_seconds"]) for result in completed),
    }


def analyze_heldout_campaign(
    protocol: FrozenHeldoutProtocol,
    envelopes: Sequence[Mapping[str, Any]],
    *,
    restoration_receipt: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Audit exact planned inputs, compute frozen contrasts, and classify."""

    restoration_passed, restoration_digest = verify_restoration_receipt(protocol, restoration_receipt)
    outcomes = _outcome_map(protocol, envelopes)
    expected_ids = {coordinate.coordinate_id for coordinate in protocol.coordinates}
    missing_ids = expected_ids - set(outcomes)
    observed_failure = any(_observed_integrity_failure(result) for result in outcomes.values())
    observed_failure = observed_failure or _has_shock_block_mismatch(protocol, outcomes)
    incomplete = bool(missing_ids) or any(result["status"] != "COMPLETED" for result in outcomes.values())
    hard_integrity: Union[bool, str]
    if observed_failure:
        hard_integrity = False
    elif incomplete:
        hard_integrity = "UNEVALUATED"
    else:
        hard_integrity = True

    contrasts = {
        "B_minus_A_success": _main_success_comparison(protocol, outcomes, "B", "A"),
        "B_minus_A_dead_end": _main_dead_end_comparison(protocol, outcomes, "B", "A"),
        "C_minus_B_success": _main_success_comparison(protocol, outcomes, "C", "B"),
        "C_minus_B_dead_end": _main_dead_end_comparison(protocol, outcomes, "C", "B"),
        "D_minus_C_success": _main_success_comparison(protocol, outcomes, "D", "C"),
        "E_minus_D_success": _main_success_comparison(protocol, outcomes, "E", "D"),
        "G_minus_F_success": _main_success_comparison(protocol, outcomes, "G", "F"),
        "G_minus_F_dead_end": _main_dead_end_comparison(protocol, outcomes, "G", "F"),
        "H_minus_G_success": _main_success_comparison(protocol, outcomes, "H", "G"),
        "H_minus_G_dead_end": _main_dead_end_comparison(protocol, outcomes, "H", "G"),
        "E_minus_H_success": _main_success_comparison(protocol, outcomes, "E", "H"),
        "E_minus_F_success": _main_success_comparison(protocol, outcomes, "E", "F"),
        "E_minus_F_dead_end": _main_dead_end_comparison(protocol, outcomes, "E", "F"),
    }
    shock = _shock_analysis(protocol, outcomes)
    costs = _cost_analysis(protocol, outcomes)

    gate_values: Dict[str, Union[bool, str]] = {}
    estimability: Dict[str, Dict[str, str]] = {}

    cb_success = contrasts["C_minus_B_success"]
    cb_dead = contrasts["C_minus_B_dead_end"]
    if not _treatment_has_valid_promotion(outcomes, "C"):
        estimability["G_EVIDENCE_MEMORY"] = _estimability(False, "ZERO_DENOMINATOR")
        gate_values["G_EVIDENCE_MEMORY"] = "UNEVALUATED"
    elif not _evidence_use_available(outcomes, ("C",)):
        estimability["G_EVIDENCE_MEMORY"] = _estimability(False, "NO_EVIDENCE_USE")
        gate_values["G_EVIDENCE_MEMORY"] = "UNEVALUATED"
    elif not cb_success["estimable"] or not cb_dead["estimable"]:
        reason = cb_success["reason"] if not cb_success["estimable"] else cb_dead["reason"]
        estimability["G_EVIDENCE_MEMORY"] = _estimability(False, reason)
        gate_values["G_EVIDENCE_MEMORY"] = "UNEVALUATED"
    else:
        estimability["G_EVIDENCE_MEMORY"] = _estimability(True, "COMPLETE")
        relative_reduction = None if cb_dead["right_rate"] == 0 else -cb_dead["point"] / cb_dead["right_rate"]
        gate_values["G_EVIDENCE_MEMORY"] = bool(
            cb_success["lower_95"] >= -0.05
            and relative_reduction is not None and relative_reduction >= 0.25
            and cb_dead["upper_95"] < 0
        )

    dc_success = contrasts["D_minus_C_success"]
    challenge_d, challenge_d_reason = _authority_challenges_pass(outcomes, "D")
    if not _treatment_has_valid_promotion(outcomes, "D"):
        estimability["G_AUTHORITY_UTILITY"] = _estimability(False, "ZERO_DENOMINATOR")
        gate_values["G_AUTHORITY_UTILITY"] = "UNEVALUATED"
    elif not dc_success["estimable"] or challenge_d is None:
        reason = dc_success["reason"] if not dc_success["estimable"] else challenge_d_reason
        estimability["G_AUTHORITY_UTILITY"] = _estimability(False, reason)
        gate_values["G_AUTHORITY_UTILITY"] = "UNEVALUATED"
    else:
        estimability["G_AUTHORITY_UTILITY"] = _estimability(True, "COMPLETE")
        gate_values["G_AUTHORITY_UTILITY"] = bool(dc_success["lower_95"] >= -0.05 and challenge_d)

    if not shock["estimable"]:
        estimability["G_CORRECTION_BENEFIT"] = _estimability(False, shock["reason"])
        gate_values["G_CORRECTION_BENEFIT"] = "UNEVALUATED"
    else:
        estimability["G_CORRECTION_BENEFIT"] = _estimability(True, "COMPLETE")
        correction_pass = (
            shock["invalidation_precision"] == 1.0
            and shock["invalidation_recall"] == 1.0
            and shock["stale_dependent_promotions"] == 0
        )
        for comparison in shock["comparisons"].values():
            recovery = comparison["recovery_within_six_difference"]
            rmst = comparison["restricted_mean_time_difference"]
            correction_pass = correction_pass and bool(
                recovery["point"] >= 0 and rmst["point"] < 0 and rmst["upper_95"] < 0
            )
        gate_values["G_CORRECTION_BENEFIT"] = bool(correction_pass)

    ed_success = contrasts["E_minus_D_success"]
    if not _treatment_has_valid_promotion(outcomes, "E"):
        estimability["G_LORA_BENEFIT"] = _estimability(False, "ZERO_DENOMINATOR")
        gate_values["G_LORA_BENEFIT"] = "UNEVALUATED"
    elif not ed_success["estimable"]:
        estimability["G_LORA_BENEFIT"] = _estimability(False, ed_success["reason"])
        gate_values["G_LORA_BENEFIT"] = "UNEVALUATED"
    else:
        estimability["G_LORA_BENEFIT"] = _estimability(True, "COMPLETE")
        gate_values["G_LORA_BENEFIT"] = bool(ed_success["point"] > 0 and ed_success["lower_95"] > 0)

    hg_success = contrasts["H_minus_G_success"]
    hg_dead = contrasts["H_minus_G_dead_end"]
    if not _treatment_has_valid_promotion(outcomes, "H"):
        estimability["G_TRAINED_EVIDENCE_MEMORY"] = _estimability(False, "ZERO_DENOMINATOR")
        gate_values["G_TRAINED_EVIDENCE_MEMORY"] = "UNEVALUATED"
    elif not _evidence_use_available(outcomes, ("H",)):
        estimability["G_TRAINED_EVIDENCE_MEMORY"] = _estimability(False, "NO_EVIDENCE_USE")
        gate_values["G_TRAINED_EVIDENCE_MEMORY"] = "UNEVALUATED"
    elif not hg_success["estimable"] or not hg_dead["estimable"]:
        reason = hg_success["reason"] if not hg_success["estimable"] else hg_dead["reason"]
        estimability["G_TRAINED_EVIDENCE_MEMORY"] = _estimability(False, reason)
        gate_values["G_TRAINED_EVIDENCE_MEMORY"] = "UNEVALUATED"
    else:
        estimability["G_TRAINED_EVIDENCE_MEMORY"] = _estimability(True, "COMPLETE")
        relative_reduction = None if hg_dead["right_rate"] == 0 else -hg_dead["point"] / hg_dead["right_rate"]
        gate_values["G_TRAINED_EVIDENCE_MEMORY"] = bool(
            hg_success["lower_95"] >= -0.05
            and relative_reduction is not None and relative_reduction >= 0.25
            and hg_dead["upper_95"] < 0
        )

    eh_success = contrasts["E_minus_H_success"]
    challenge_e, challenge_e_reason = _authority_challenges_pass(outcomes, "E")
    if not _treatment_has_valid_promotion(outcomes, "E"):
        estimability["G_TRAINED_AUTHORITY_UTILITY"] = _estimability(False, "ZERO_DENOMINATOR")
        gate_values["G_TRAINED_AUTHORITY_UTILITY"] = "UNEVALUATED"
    elif not eh_success["estimable"] or challenge_e is None:
        reason = eh_success["reason"] if not eh_success["estimable"] else challenge_e_reason
        estimability["G_TRAINED_AUTHORITY_UTILITY"] = _estimability(False, reason)
        gate_values["G_TRAINED_AUTHORITY_UTILITY"] = "UNEVALUATED"
    else:
        estimability["G_TRAINED_AUTHORITY_UTILITY"] = _estimability(True, "COMPLETE")
        gate_values["G_TRAINED_AUTHORITY_UTILITY"] = bool(eh_success["lower_95"] >= -0.05 and challenge_e)

    ef_success = contrasts["E_minus_F_success"]
    ef_dead = contrasts["E_minus_F_dead_end"]
    if not _treatment_has_valid_promotion(outcomes, "E"):
        estimability["G_TRAINED_FULL_SYSTEM_CONTRIBUTION"] = _estimability(False, "ZERO_DENOMINATOR")
        gate_values["G_TRAINED_FULL_SYSTEM_CONTRIBUTION"] = "UNEVALUATED"
    elif not _evidence_use_available(outcomes, ("E",)):
        estimability["G_TRAINED_FULL_SYSTEM_CONTRIBUTION"] = _estimability(False, "NO_EVIDENCE_USE")
        gate_values["G_TRAINED_FULL_SYSTEM_CONTRIBUTION"] = "UNEVALUATED"
    elif not ef_success["estimable"] or not ef_dead["estimable"]:
        reason = ef_success["reason"] if not ef_success["estimable"] else ef_dead["reason"]
        estimability["G_TRAINED_FULL_SYSTEM_CONTRIBUTION"] = _estimability(False, reason)
        gate_values["G_TRAINED_FULL_SYSTEM_CONTRIBUTION"] = "UNEVALUATED"
    else:
        estimability["G_TRAINED_FULL_SYSTEM_CONTRIBUTION"] = _estimability(True, "COMPLETE")
        relative_reduction = None if ef_dead["right_rate"] == 0 else -ef_dead["point"] / ef_dead["right_rate"]
        gate_values["G_TRAINED_FULL_SYSTEM_CONTRIBUTION"] = bool(
            ef_success["lower_95"] >= -0.05
            and relative_reduction is not None and relative_reduction >= 0.25
            and ef_dead["upper_95"] < 0
        )

    if not costs["estimable"]:
        estimability["G_EFFICIENCY"] = _estimability(False, costs["reason"])
        gate_values["G_EFFICIENCY"] = "UNEVALUATED"
    else:
        estimability["G_EFFICIENCY"] = _estimability(True, "COMPLETE")
        gate_values["G_EFFICIENCY"] = bool(costs["measures"]["wall_time_seconds"]["relative_overhead"] <= 0.30)

    g_estimable = not incomplete and all(item["status"] == "ESTIMABLE" for item in estimability.values())
    gates: Dict[str, Union[bool, str]] = {
        "G_RESTORATION": restoration_passed,
        "G_HARD_INTEGRITY": hard_integrity,
        "G_ESTIMABLE": g_estimable,
    }
    gates.update(gate_values)

    disposition_map = {
        "G_EVIDENCE_MEMORY": "EVIDENCE_MEMORY_NOT_DEMONSTRATED",
        "G_AUTHORITY_UTILITY": "AUTHORITY_UTILITY_NOT_DEMONSTRATED",
        "G_CORRECTION_BENEFIT": "CORRECTION_BENEFIT_NOT_DEMONSTRATED",
        "G_LORA_BENEFIT": "TRAINING_BENEFIT_NOT_DEMONSTRATED",
        "G_TRAINED_EVIDENCE_MEMORY": "TRAINED_EVIDENCE_NOT_DEMONSTRATED",
        "G_TRAINED_AUTHORITY_UTILITY": "TRAINED_AUTHORITY_UTILITY_NOT_DEMONSTRATED",
        "G_TRAINED_FULL_SYSTEM_CONTRIBUTION": "TRAINED_FULL_SYSTEM_NOT_DEMONSTRATED",
        "G_EFFICIENCY": "PROMISING_WITH_COST",
    }
    if not gates["G_RESTORATION"]:
        disposition = "RESTORATION_BLOCKED"
        selected_index = 0
    elif gates["G_HARD_INTEGRITY"] is False:
        disposition = "NOT_SUPPORTED"
        selected_index = 1
    elif gates["G_HARD_INTEGRITY"] == "UNEVALUATED" or not gates["G_ESTIMABLE"]:
        disposition = "INCONCLUSIVE"
        selected_index = 2
    else:
        disposition = "PROMISING"
        selected_index = len(GATE_ORDER)
        for gate_name in RESEARCH_GATES:
            if gates[gate_name] is False:
                disposition = disposition_map[gate_name]
                selected_index = GATE_ORDER.index(gate_name)
                break

    failed_downstream = [
        name for index, name in enumerate(GATE_ORDER)
        if index > selected_index and gates[name] is False
    ]
    unevaluated_gates = [name for name in RESEARCH_GATES if gates[name] == "UNEVALUATED"]
    arm_metrics = {arm: _arm_aggregate(protocol, outcomes, arm) for arm in ARM_IDS}
    completed_main = sum(item["completed_trajectories"] for item in arm_metrics.values())
    completed_shock = sum(
        result["status"] == "COMPLETED" for result in outcomes.values() if result["phase"] == SHOCK_PHASE
    )
    return canonical_value(
        {
            "schema_version": REPORT_SCHEMA,
            "campaign_id": protocol.campaign_id,
            "protocol_digest": protocol.digest,
            "restoration_receipt_digest": restoration_digest,
            "planned": {"main_trajectories": 192, "shock_trajectories": 36, "total_trajectories": 228, "max_candidate_attempts": 2736},
            "observed": {"journaled_trajectories": len(outcomes), "completed_main_trajectories": completed_main, "completed_shock_trajectories": completed_shock},
            "arm_metrics": arm_metrics,
            "contrasts": contrasts,
            "correction_shock": shock,
            "efficiency": costs,
            "estimability": estimability,
            "GATE_VECTOR": {name: gates[name] for name in GATE_ORDER},
            "DISPOSITION": disposition,
            "FAILED_DOWNSTREAM_GATES": failed_downstream,
            "UNEVALUATED_GATES": unevaluated_gates,
            "claim_boundary": "Aggregate protocol evidence only; no utility or causal claim beyond the frozen contrasts.",
        }
    )


def build_private_campaign_record(
    protocol: FrozenHeldoutProtocol,
    envelopes: Sequence[Mapping[str, Any]],
    restoration_receipt: Mapping[str, Any],
) -> Dict[str, Any]:
    """Recompute and return the evaluator-private exact signed package."""

    verified_by_coordinate = {
        envelope["coordinate_id"]: verify_signed_result_envelope(protocol, envelope)
        for envelope in envelopes
    }
    if len(verified_by_coordinate) != len(envelopes):
        raise HeldoutProtocolError("private campaign record contains duplicate signed outcomes")
    planned_ids = {coordinate.coordinate_id for coordinate in protocol.coordinates}
    if set(verified_by_coordinate) != planned_ids:
        raise HeldoutProtocolError("private campaign record requires all 228 exact signed outcomes")
    ordered = [
        verified_by_coordinate[coordinate.coordinate_id]
        for coordinate in protocol.coordinates
        if coordinate.coordinate_id in verified_by_coordinate
    ]
    analysis = analyze_heldout_campaign(
        protocol, ordered, restoration_receipt=restoration_receipt
    )
    restoration_ok, restoration_digest = verify_restoration_receipt(protocol, restoration_receipt)
    if not restoration_ok or restoration_digest is None:
        raise HeldoutProtocolError("private campaign record requires a verified restoration receipt")
    return canonical_value(
        {
            "schema_version": PRIVATE_RECORD_SCHEMA,
            "protocol": protocol.to_private_dict(),
            "signed_result_envelopes": ordered,
            "restoration_receipt": dict(restoration_receipt),
            "restoration_receipt_digest": restoration_digest,
            "aggregate_analysis": dict(analysis),
            "signed_result_envelopes_digest": digest_for(ordered),
        }
    )


__all__ = [
    "COST_KEYS",
    "CoordinateOperationStore",
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "FrozenHeldoutProtocol",
    "HeldoutCoordinate",
    "HeldoutJournal",
    "HeldoutProtocolError",
    "MAIN_PHASE",
    "REPORT_SCHEMA",
    "RESULT_SCHEMA",
    "SHOCK_PHASE",
    "analyze_heldout_campaign",
    "build_private_campaign_record",
    "build_signed_reconciliation",
    "build_signed_result_envelope",
    "run_pending_coordinates",
    "validate_result",
    "verify_restoration_receipt",
    "verify_signed_reconciliation",
    "verify_signed_result_envelope",
]
