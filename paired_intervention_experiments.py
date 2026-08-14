"""Exact paired perturbation evaluation for ChelatedAI method development.

This module implements only the dependency-light sanity boundary frozen in
``CHELATEDAI-PRW-ISI1-PAIRED-SANITY-v1``.  It scores nuisance-preserving and
material-answer-changing pairs together, then applies that evaluator to a
small synthetic retrieval fixture through the existing production variance
chelation method.

Passing this fixture is not evidence for full PRW-ISI1, natural-language
reasoning, production retrieval utility, or novelty.  Full PRW-ISI1 remains
blocked on PRW-EK3.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PAIRED_INTERVENTION_PROTOCOL_ID = "CHELATEDAI-PRW-ISI1-PAIRED-SANITY-v1"
PAIRED_INTERVENTION_STAGE_ID = "PRW-ISI1-PAIRED-SANITY"
NUISANCE_PAIR = "NUISANCE_INVARIANCE"
MATERIAL_PAIR = "MATERIAL_INTERVENTION"
DEFAULT_MAX_DIMENSION = 64
DEFAULT_MAX_PAIRS = 5_000
DEFAULT_MAX_ESTIMATED_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_WORK_UNITS = 5_000_000
DEFAULT_MAX_SECONDS = 30.0
DEFAULT_MAX_OUTPUT_BYTES = 8 * 1024 * 1024


class PairedInterventionValidationError(ValueError):
    """Raised when a paired case or fixture violates the frozen contract."""


class PairedInterventionResourceError(RuntimeError):
    """Raised before or during work when an immutable resource limit is crossed."""


def _plain_int(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        raise PairedInterventionValidationError(f"{name} must be a plain integer")
    if value < minimum or value > maximum:
        raise PairedInterventionValidationError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _plain_float(value: object, name: str, minimum: float, maximum: float) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise PairedInterventionValidationError(f"{name} must be a plain real number")
    result = float(value)
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise PairedInterventionValidationError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return result


def _nonempty_string(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise PairedInterventionValidationError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class PairedInterventionBudget:
    """Caller-lowerable limits under the immutable paired-sanity ceiling."""

    max_dimension: int = DEFAULT_MAX_DIMENSION
    max_pairs: int = DEFAULT_MAX_PAIRS
    max_estimated_bytes: int = DEFAULT_MAX_ESTIMATED_BYTES
    max_work_units: int = DEFAULT_MAX_WORK_UNITS
    max_seconds: float = DEFAULT_MAX_SECONDS
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES

    def __post_init__(self) -> None:
        for name, value, maximum in (
            ("max_dimension", self.max_dimension, DEFAULT_MAX_DIMENSION),
            ("max_pairs", self.max_pairs, DEFAULT_MAX_PAIRS),
            ("max_estimated_bytes", self.max_estimated_bytes, DEFAULT_MAX_ESTIMATED_BYTES),
            ("max_work_units", self.max_work_units, DEFAULT_MAX_WORK_UNITS),
            ("max_output_bytes", self.max_output_bytes, DEFAULT_MAX_OUTPUT_BYTES),
        ):
            object.__setattr__(self, name, _plain_int(value, name, 1, maximum))
        object.__setattr__(
            self,
            "max_seconds",
            _plain_float(self.max_seconds, "max_seconds", 0.001, DEFAULT_MAX_SECONDS),
        )


@dataclass(frozen=True)
class PairedInterventionResourceEstimate:
    dimension: int
    pair_count: int
    document_count: int
    estimated_bytes: int
    work_units: int
    max_seconds: float

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class _Deadline:
    def __init__(self, seconds: float) -> None:
        self._deadline = time.monotonic() + seconds

    def check(self, stage: str) -> None:
        if time.monotonic() > self._deadline:
            raise PairedInterventionResourceError(f"deadline exceeded at {stage}")


def estimate_paired_resources(
    *,
    dimension: int,
    pair_count: int,
    document_count: int,
    budget: PairedInterventionBudget = PairedInterventionBudget(),
) -> PairedInterventionResourceEstimate:
    """Allocation-free preflight for the synthetic paired retrieval suite."""

    d = _plain_int(dimension, "dimension", 2, budget.max_dimension)
    pairs = _plain_int(pair_count, "pair_count", 2, budget.max_pairs)
    documents = _plain_int(document_count, "document_count", 2, 10_000)
    vector_values = (pairs * 2 + documents + documents * 2) * d
    estimated_bytes = vector_values * 8 + pairs * 4_096 + 128 * 1_024
    work_units = pairs * documents * d * 8
    if estimated_bytes > budget.max_estimated_bytes:
        raise PairedInterventionResourceError(
            f"modeled allocation {estimated_bytes} exceeds {budget.max_estimated_bytes} bytes"
        )
    if work_units > budget.max_work_units:
        raise PairedInterventionResourceError(
            f"modeled work {work_units} exceeds {budget.max_work_units} units"
        )
    return PairedInterventionResourceEstimate(
        dimension=d,
        pair_count=pairs,
        document_count=documents,
        estimated_bytes=estimated_bytes,
        work_units=work_units,
        max_seconds=budget.max_seconds,
    )


@dataclass(frozen=True)
class PairedCase:
    """One canonical/perturbed prediction pair with a declared causal role."""

    case_id: str
    pair_kind: str
    canonical_label: str
    perturbed_label: str
    canonical_prediction: str
    perturbed_prediction: str
    changed_features: Tuple[str, ...]
    safety_critical: bool = False

    def __post_init__(self) -> None:
        for name in (
            "case_id",
            "canonical_label",
            "perturbed_label",
            "canonical_prediction",
            "perturbed_prediction",
        ):
            _nonempty_string(getattr(self, name), name)
        if self.pair_kind not in (NUISANCE_PAIR, MATERIAL_PAIR):
            raise PairedInterventionValidationError("pair_kind is not recognized")
        if type(self.changed_features) is not tuple or not self.changed_features:
            raise PairedInterventionValidationError("changed_features must be a non-empty tuple")
        checked_features = tuple(
            _nonempty_string(feature, "changed_features item") for feature in self.changed_features
        )
        if len(set(checked_features)) != len(checked_features):
            raise PairedInterventionValidationError("changed_features must be unique")
        if type(self.safety_critical) is not bool:
            raise PairedInterventionValidationError("safety_critical must be boolean")
        if self.pair_kind == NUISANCE_PAIR and self.canonical_label != self.perturbed_label:
            raise PairedInterventionValidationError("nuisance pairs must preserve the declared label")
        if self.pair_kind == MATERIAL_PAIR and self.canonical_label == self.perturbed_label:
            raise PairedInterventionValidationError("material pairs must change the declared label")

    @property
    def intervention_order(self) -> int:
        return len(self.changed_features)


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def score_paired_cases(cases: Sequence[PairedCase]) -> Dict[str, object]:
    """Score exact pair correctness without allowing either pair kind to vanish."""

    checked = tuple(cases)
    if not checked:
        raise PairedInterventionValidationError("cases cannot be empty")
    if any(not isinstance(case, PairedCase) for case in checked):
        raise PairedInterventionValidationError("every case must be a PairedCase")
    identifiers = [case.case_id for case in checked]
    if len(set(identifiers)) != len(identifiers):
        raise PairedInterventionValidationError("case_id values must be unique")
    nuisance_count = sum(case.pair_kind == NUISANCE_PAIR for case in checked)
    material_count = sum(case.pair_kind == MATERIAL_PAIR for case in checked)
    if nuisance_count == 0 or material_count == 0:
        raise PairedInterventionValidationError("both nuisance and material pairs are required")

    rows: List[Dict[str, object]] = []
    order_buckets: Dict[int, Dict[str, int]] = {}
    for case in checked:
        canonical_correct = case.canonical_prediction == case.canonical_label
        perturbed_correct = case.perturbed_prediction == case.perturbed_label
        expected_flip = case.pair_kind == MATERIAL_PAIR
        predicted_flip = case.canonical_prediction != case.perturbed_prediction
        relation_correct = predicted_flip == expected_flip
        strict_pair_correct = canonical_correct and perturbed_correct and relation_correct
        row = {
            "case_id": case.case_id,
            "pair_kind": case.pair_kind,
            "canonical_label": case.canonical_label,
            "perturbed_label": case.perturbed_label,
            "canonical_prediction": case.canonical_prediction,
            "perturbed_prediction": case.perturbed_prediction,
            "changed_features": list(case.changed_features),
            "intervention_order": case.intervention_order,
            "safety_critical": case.safety_critical,
            "canonical_correct": canonical_correct,
            "perturbed_correct": perturbed_correct,
            "expected_flip": expected_flip,
            "predicted_flip": predicted_flip,
            "relation_correct": relation_correct,
            "strict_pair_correct": strict_pair_correct,
        }
        rows.append(row)
        bucket = order_buckets.setdefault(
            case.intervention_order,
            {"count": 0, "strict_correct": 0, "relation_correct": 0},
        )
        bucket["count"] += 1
        bucket["strict_correct"] += int(strict_pair_correct)
        bucket["relation_correct"] += int(relation_correct)

    canonical_correct_count = sum(bool(row["canonical_correct"]) for row in rows)
    perturbed_correct_count = sum(bool(row["perturbed_correct"]) for row in rows)
    strict_correct_count = sum(bool(row["strict_pair_correct"]) for row in rows)
    relation_correct_count = sum(bool(row["relation_correct"]) for row in rows)
    nuisance_rows = [row for row in rows if row["pair_kind"] == NUISANCE_PAIR]
    material_rows = [row for row in rows if row["pair_kind"] == MATERIAL_PAIR]
    nuisance_strict_count = sum(bool(row["strict_pair_correct"]) for row in nuisance_rows)
    material_strict_count = sum(bool(row["strict_pair_correct"]) for row in material_rows)
    nuisance_relation_count = sum(bool(row["relation_correct"]) for row in nuisance_rows)
    material_relation_count = sum(bool(row["relation_correct"]) for row in material_rows)
    nuisance_violation_count = sum(bool(row["predicted_flip"]) for row in nuisance_rows)
    missed_intervention_count = sum(not bool(row["predicted_flip"]) for row in material_rows)
    nuisance_strict_accuracy = _rate(nuisance_strict_count, nuisance_count)
    material_strict_accuracy = _rate(material_strict_count, material_count)
    if nuisance_strict_accuracy and material_strict_accuracy:
        balanced_joint_score = (
            2.0
            * nuisance_strict_accuracy
            * material_strict_accuracy
            / (nuisance_strict_accuracy + material_strict_accuracy)
        )
    else:
        balanced_joint_score = 0.0
    critical_rows = [row for row in material_rows if row["safety_critical"]]
    safety_critical_miss_count = sum(not bool(row["strict_pair_correct"]) for row in critical_rows)

    by_intervention_order = {
        str(order): {
            "count": bucket["count"],
            "strict_accuracy": _rate(bucket["strict_correct"], bucket["count"]),
            "relation_accuracy": _rate(bucket["relation_correct"], bucket["count"]),
        }
        for order, bucket in sorted(order_buckets.items())
    }
    return {
        "pair_count": len(rows),
        "nuisance_pair_count": nuisance_count,
        "material_pair_count": material_count,
        "canonical_accuracy": _rate(canonical_correct_count, len(rows)),
        "perturbed_accuracy": _rate(perturbed_correct_count, len(rows)),
        "strict_paired_accuracy": _rate(strict_correct_count, len(rows)),
        "relation_accuracy": _rate(relation_correct_count, len(rows)),
        "nuisance_relation_accuracy": _rate(nuisance_relation_count, nuisance_count),
        "material_relation_accuracy": _rate(material_relation_count, material_count),
        "nuisance_strict_pair_accuracy": nuisance_strict_accuracy,
        "material_strict_pair_accuracy": material_strict_accuracy,
        "balanced_joint_score": balanced_joint_score,
        "joint_floor": min(nuisance_strict_accuracy, material_strict_accuracy),
        "nuisance_violation_count": nuisance_violation_count,
        "nuisance_violation_rate": _rate(nuisance_violation_count, nuisance_count),
        "false_intervention_count": nuisance_violation_count,
        "false_intervention_rate": _rate(nuisance_violation_count, nuisance_count),
        "missed_intervention_count": missed_intervention_count,
        "missed_intervention_rate": _rate(missed_intervention_count, material_count),
        "safety_critical_material_count": len(critical_rows),
        "safety_critical_miss_count": safety_critical_miss_count,
        "all_safety_critical_detected": safety_critical_miss_count == 0,
        "by_intervention_order": by_intervention_order,
        "rows": rows,
    }


def build_paired_chelation_fixture(
    *,
    topic_count: int = 4,
    collapse_strength: float = 4.0,
    nuisance_multiplier: float = 1.5,
) -> Dict[str, object]:
    """Create deterministic retrieval pairs with one declared nuisance axis."""

    topics = _plain_int(topic_count, "topic_count", 2, 32)
    collapse = _plain_float(collapse_strength, "collapse_strength", 1.01, 1_000.0)
    multiplier = _plain_float(nuisance_multiplier, "nuisance_multiplier", 1.01, 100.0)
    dimension = topics + 1
    collapse_dimension = topics
    documents: Dict[str, np.ndarray] = {
        "a_collapse": np.eye(dimension, dtype=np.float64)[collapse_dimension]
    }
    for topic in range(topics):
        vector = np.zeros(dimension, dtype=np.float64)
        vector[topic] = 1.0
        documents[f"topic_{topic:02d}"] = vector

    calibration_rows: List[np.ndarray] = []
    for topic in range(topics):
        for sign in (-1.0, 1.0):
            row = np.zeros(dimension, dtype=np.float64)
            row[topic] = 1.0
            row[collapse_dimension] = sign * collapse
            calibration_rows.append(row)

    pairs: List[Dict[str, object]] = []
    for topic in range(topics):
        next_topic = (topic + 1) % topics
        canonical_query = np.zeros(dimension, dtype=np.float64)
        canonical_query[topic] = 1.0
        canonical_query[collapse_dimension] = collapse
        nuisance_query = canonical_query.copy()
        nuisance_query[collapse_dimension] = collapse * multiplier
        material_query = np.zeros(dimension, dtype=np.float64)
        material_query[next_topic] = 1.0
        material_query[collapse_dimension] = collapse
        pairs.append(
            {
                "case_id": f"nuisance_{topic:02d}",
                "pair_kind": NUISANCE_PAIR,
                "canonical_query": canonical_query,
                "perturbed_query": nuisance_query,
                "canonical_label": f"topic_{topic:02d}",
                "perturbed_label": f"topic_{topic:02d}",
                "changed_features": ("nuisance_amplitude",),
                "safety_critical": False,
            }
        )
        pairs.append(
            {
                "case_id": f"material_{topic:02d}",
                "pair_kind": MATERIAL_PAIR,
                "canonical_query": canonical_query,
                "perturbed_query": material_query,
                "canonical_label": f"topic_{topic:02d}",
                "perturbed_label": f"topic_{next_topic:02d}",
                "changed_features": (f"topic_identity_{topic:02d}_to_{next_topic:02d}",),
                "safety_critical": topic == 0,
            }
        )

    coalition_target = 2 if topics > 2 else 1
    coalition_canonical = np.zeros(dimension, dtype=np.float64)
    coalition_canonical[0] = 1.0
    coalition_canonical[collapse_dimension] = collapse
    coalition_perturbed = np.zeros(dimension, dtype=np.float64)
    coalition_perturbed[coalition_target] = 1.0
    coalition_perturbed[collapse_dimension] = collapse
    pairs.append(
        {
            "case_id": "material_coalition_00",
            "pair_kind": MATERIAL_PAIR,
            "canonical_query": coalition_canonical,
            "perturbed_query": coalition_perturbed,
            "canonical_label": "topic_00",
            "perturbed_label": f"topic_{coalition_target:02d}",
            "changed_features": ("fact_alpha", "fact_beta"),
            "safety_critical": True,
        }
    )
    return {
        "dimension": dimension,
        "topic_count": topics,
        "collapse_dimension": collapse_dimension,
        "collapse_strength": collapse,
        "nuisance_multiplier": multiplier,
        "documents": documents,
        "calibration_cluster": np.vstack(calibration_rows),
        "pairs": pairs,
    }


def _cosine_score(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator else 0.0


def _top_prediction(
    query: np.ndarray,
    documents: Mapping[str, np.ndarray],
    mask: np.ndarray,
) -> str:
    masked_query = query * mask
    scored = [
        (document_id, _cosine_score(masked_query, vector * mask))
        for document_id, vector in documents.items()
    ]
    return min(scored, key=lambda item: (-item[1], item[0]))[0]


def _validate_mask(mask: object, dimension: int) -> np.ndarray:
    result = np.asarray(mask, dtype=np.float64)
    if result.shape != (dimension,):
        raise PairedInterventionValidationError(f"mask must have shape ({dimension},)")
    if not np.all(np.isfinite(result)):
        raise PairedInterventionValidationError("mask must be finite")
    if np.any(result < 0.0) or np.any(result > 1.0):
        raise PairedInterventionValidationError("mask values must be in [0, 1]")
    return result


def production_variance_chelation_mask(
    calibration_cluster: np.ndarray,
    *,
    chelation_p: int = 85,
) -> np.ndarray:
    """Invoke the existing production variance-mask method without engine setup."""

    cluster = np.asarray(calibration_cluster, dtype=np.float64)
    if cluster.ndim != 2 or cluster.shape[0] < 2 or cluster.shape[1] < 2:
        raise PairedInterventionValidationError("calibration_cluster must be a non-empty 2-D matrix")
    if not np.all(np.isfinite(cluster)):
        raise PairedInterventionValidationError("calibration_cluster must be finite")
    percentile = _plain_int(chelation_p, "chelation_p", 1, 99)
    from antigravity_engine import AntigravityEngine

    proxy = SimpleNamespace(vector_size=cluster.shape[1], chelation_p=percentile)
    mask = AntigravityEngine._chelate_toxicity(proxy, cluster)
    return _validate_mask(mask, cluster.shape[1])


def evaluate_synthetic_policy(
    fixture: Mapping[str, object],
    *,
    policy: str,
    mask: object,
) -> Dict[str, object]:
    """Evaluate one frozen mask policy through exact paired retrieval scoring."""

    policy_name = _nonempty_string(policy, "policy")
    dimension = _plain_int(fixture["dimension"], "fixture dimension", 2, DEFAULT_MAX_DIMENSION)
    checked_mask = _validate_mask(mask, dimension)
    documents = fixture["documents"]
    if not isinstance(documents, Mapping) or not documents:
        raise PairedInterventionValidationError("fixture documents must be a non-empty mapping")
    cases: List[PairedCase] = []
    for pair in fixture["pairs"]:
        canonical_prediction = _top_prediction(pair["canonical_query"], documents, checked_mask)
        perturbed_prediction = _top_prediction(pair["perturbed_query"], documents, checked_mask)
        cases.append(
            PairedCase(
                case_id=pair["case_id"],
                pair_kind=pair["pair_kind"],
                canonical_label=pair["canonical_label"],
                perturbed_label=pair["perturbed_label"],
                canonical_prediction=canonical_prediction,
                perturbed_prediction=perturbed_prediction,
                changed_features=pair["changed_features"],
                safety_critical=pair["safety_critical"],
            )
        )
    return {
        "policy": policy_name,
        "mask": [float(value) for value in checked_mask],
        "masked_dimensions": [int(index) for index in np.flatnonzero(checked_mask == 0.0)],
        "metrics": score_paired_cases(cases),
    }


def run_paired_chelation_sanity(
    *,
    topic_count: int = 4,
    collapse_strength: float = 4.0,
    nuisance_multiplier: float = 1.5,
    chelation_p: int = 85,
    budget: PairedInterventionBudget = PairedInterventionBudget(),
) -> Dict[str, object]:
    """Run the frozen paired chelation sanity controls and retain all outcomes."""

    expected_pair_count = _plain_int(topic_count, "topic_count", 2, 32) * 2 + 1
    expected_document_count = topic_count + 1
    estimate = estimate_paired_resources(
        dimension=topic_count + 1,
        pair_count=expected_pair_count,
        document_count=expected_document_count,
        budget=budget,
    )
    deadline = _Deadline(budget.max_seconds)
    fixture = build_paired_chelation_fixture(
        topic_count=topic_count,
        collapse_strength=collapse_strength,
        nuisance_multiplier=nuisance_multiplier,
    )
    deadline.check("fixture")
    dimension = int(fixture["dimension"])
    oracle_mask = np.ones(dimension, dtype=np.float64)
    oracle_mask[int(fixture["collapse_dimension"])] = 0.0
    production_mask = production_variance_chelation_mask(
        fixture["calibration_cluster"],
        chelation_p=chelation_p,
    )
    controls = (
        ("no_projection", np.ones(dimension, dtype=np.float64)),
        ("oracle_nuisance_mask", oracle_mask),
        ("production_variance_chelation", production_mask),
        ("over_chelation", np.zeros(dimension, dtype=np.float64)),
    )
    policies: List[Dict[str, object]] = []
    for policy, mask in controls:
        deadline.check(f"policy_{policy}")
        policies.append(evaluate_synthetic_policy(fixture, policy=policy, mask=mask))
    by_policy = {row["policy"]: row for row in policies}
    baseline = by_policy["no_projection"]["metrics"]
    oracle = by_policy["oracle_nuisance_mask"]["metrics"]
    production = by_policy["production_variance_chelation"]["metrics"]
    over_chelation = by_policy["over_chelation"]["metrics"]
    sanity = {
        "production_mask_matches_oracle": bool(np.array_equal(production_mask, oracle_mask)),
        "production_predictions_match_oracle": production["rows"] == oracle["rows"],
        "production_strict_paired_accuracy_is_one": production["strict_paired_accuracy"] == 1.0,
        "production_balanced_joint_score_is_one": production["balanced_joint_score"] == 1.0,
        "production_joint_floor_is_one": production["joint_floor"] == 1.0,
        "production_detects_all_safety_critical": production["all_safety_critical_detected"],
        "production_beats_no_projection_joint_score": (
            production["balanced_joint_score"] > baseline["balanced_joint_score"]
        ),
        "over_chelation_looks_nuisance_invariant": (
            over_chelation["nuisance_relation_accuracy"] == 1.0
        ),
        "over_chelation_misses_material_interventions": (
            over_chelation["material_relation_accuracy"] == 0.0
            and over_chelation["missed_intervention_rate"] == 1.0
        ),
        "balanced_score_rejects_over_chelation": over_chelation["balanced_joint_score"] == 0.0,
        "second_order_intervention_is_scored": (
            production["by_intervention_order"].get("2", {}).get("strict_accuracy") == 1.0
        ),
    }
    deadline.check("summary")
    return {
        "stage_id": PAIRED_INTERVENTION_STAGE_ID,
        "protocol_id": PAIRED_INTERVENTION_PROTOCOL_ID,
        "execution_mode": "bounded_cpu_synthetic_sanity",
        "relationship_to_video": "ADDITIVE_EVALUATION_AXIS_ONLY",
        "parent_card": "PRW-ISI1",
        "parent_dependency_status": "BLOCKED_ON_PRW-EK3",
        "evidence_state": "VALIDATED" if all(sanity.values()) else "REJECTED",
        "scientific_claim_status": "UNCONFIRMED",
        "novelty_claim_status": "UNCONFIRMED",
        "production_path_changed": False,
        "model_or_corpus_loaded": False,
        "fixture": {
            "topic_count": fixture["topic_count"],
            "dimension": fixture["dimension"],
            "pair_count": len(fixture["pairs"]),
            "document_count": len(fixture["documents"]),
            "collapse_dimension": fixture["collapse_dimension"],
            "collapse_strength": fixture["collapse_strength"],
            "nuisance_multiplier": fixture["nuisance_multiplier"],
            "chelation_p": chelation_p,
        },
        "resource_estimate": estimate.as_dict(),
        "oracle_mask": [float(value) for value in oracle_mask],
        "production_mask": [float(value) for value in production_mask],
        "policies": policies,
        "sanity": sanity,
    }


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest_payload(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PairedInterventionArtifact:
    stage_id: str
    protocol_id: str
    status: str
    evidence_state: str
    scientific_claim_status: str
    budget: Dict[str, object]
    result: Dict[str, object]
    limitations: Tuple[str, ...]
    artifact_digest: str

    @classmethod
    def create(
        cls,
        *,
        budget: PairedInterventionBudget,
        result: Mapping[str, object],
        limitations: Sequence[str],
    ) -> "PairedInterventionArtifact":
        if result.get("stage_id") != PAIRED_INTERVENTION_STAGE_ID:
            raise PairedInterventionValidationError("result stage does not match paired protocol")
        checked_limitations = tuple(_nonempty_string(item, "limitation") for item in limitations)
        payload: Dict[str, object] = {
            "stage_id": PAIRED_INTERVENTION_STAGE_ID,
            "protocol_id": PAIRED_INTERVENTION_PROTOCOL_ID,
            "status": "COMPLETE",
            "evidence_state": result["evidence_state"],
            "scientific_claim_status": result["scientific_claim_status"],
            "budget": asdict(budget),
            "result": dict(result),
            "limitations": list(checked_limitations),
        }
        return cls(
            stage_id=PAIRED_INTERVENTION_STAGE_ID,
            protocol_id=PAIRED_INTERVENTION_PROTOCOL_ID,
            status="COMPLETE",
            evidence_state=str(result["evidence_state"]),
            scientific_claim_status=str(result["scientific_claim_status"]),
            budget=asdict(budget),
            result=dict(result),
            limitations=checked_limitations,
            artifact_digest=_digest_payload(payload),
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "protocol_id": self.protocol_id,
            "status": self.status,
            "evidence_state": self.evidence_state,
            "scientific_claim_status": self.scientific_claim_status,
            "budget": self.budget,
            "result": self.result,
            "limitations": list(self.limitations),
            "artifact_digest": self.artifact_digest,
        }

    def write_json(self, output: object) -> Path:
        if isinstance(output, Path):
            path = output
        elif type(output) is str and output:
            path = Path(output)
        else:
            raise PairedInterventionValidationError("output must be a non-empty path")
        encoded = (_canonical_json(self.as_dict()) + "\n").encode("utf-8")
        if len(encoded) > int(self.budget["max_output_bytes"]):
            raise PairedInterventionResourceError("artifact exceeds max_output_bytes")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=str(path.parent),
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
        return path


def make_paired_intervention_artifact(
    *,
    budget: PairedInterventionBudget = PairedInterventionBudget(),
) -> PairedInterventionArtifact:
    result = run_paired_chelation_sanity(budget=budget)
    return PairedInterventionArtifact.create(
        budget=budget,
        result=result,
        limitations=(
            "frozen synthetic retrieval fixture only",
            "oracle mask and production variance mask are expected to coincide on this fixture",
            "no natural-language, model, corpus, RAG, training, or independent confirmation evidence",
            "full PRW-ISI1 remains blocked on PRW-EK3",
        ),
    )


__all__ = [
    "MATERIAL_PAIR",
    "NUISANCE_PAIR",
    "PAIRED_INTERVENTION_PROTOCOL_ID",
    "PAIRED_INTERVENTION_STAGE_ID",
    "PairedCase",
    "PairedInterventionArtifact",
    "PairedInterventionBudget",
    "PairedInterventionResourceError",
    "PairedInterventionValidationError",
    "build_paired_chelation_fixture",
    "estimate_paired_resources",
    "evaluate_synthetic_policy",
    "make_paired_intervention_artifact",
    "production_variance_chelation_mask",
    "run_paired_chelation_sanity",
    "score_paired_cases",
]
