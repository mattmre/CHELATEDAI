"""Closed ``egv-sft-row-v1`` training-row contract."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..canonical import content_id, digest_for, validate_sha256
from .dataset import EvaluationCorpus, FAMILY_BY_ID
from .diagnostics import (
    failure_family_for_attempt,
    validate_diagnostic,
    validate_disposition,
    validate_requested_authority,
    validate_resource_bucket,
)
from .prompts import PROMPT_IDS


SFT_SCHEMA_VERSION = "egv-sft-row-v1"
SFT_FIELDS = frozenset(
    {
        "schema_version",
        "row_id",
        "task_id",
        "task_family",
        "split",
        "arm",
        "attempt_index",
        "attempt_id",
        "task_record_digest",
        "task_manifest_digest",
        "public_locus",
        "public_rule_id",
        "prompt_template_ids",
        "prompt_digest",
        "retrieved_evidence_ids",
        "proposed_mutation_digest",
        "candidate_artifact_digest",
        "verdict_receipt_digest",
        "diagnostic_enum",
        "resource_bucket",
        "infrastructure_incident_id",
        "failure_family_root",
        "dependency_ids",
        "requested_authority",
        "promotion_disposition",
        "input_digest",
        "output_digest",
    }
)
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}$")
_FORBIDDEN_ROW_FIELDS = frozenset(
    {
        "prompt",
        "raw_prompt",
        "patch",
        "source",
        "candidate_source",
        "rationale",
        "reasoning",
        "stdout",
        "stderr",
        "hidden_test",
        "expected_output",
        "private_path",
        "credential",
    }
)


def _require_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ValueError("{} must be a bounded public ID".format(field))
    if "/" in value or "\\" in value or ".." in value:
        raise ValueError("{} contains a path-like value".format(field))
    return value


def _require_digest(value: Any, field: str) -> str:
    return validate_sha256(value, field)


def _require_digest_list(value: Any, field: str, *, sorted_required: bool = True) -> List[str]:
    if not isinstance(value, list):
        raise ValueError("{} must be a JSON array".format(field))
    normalized = [_require_digest(item, "{} item".format(field)) for item in value]
    if len(set(normalized)) != len(normalized):
        raise ValueError("{} must not contain duplicates".format(field))
    if sorted_required and normalized != sorted(normalized):
        raise ValueError("{} must be lexicographically sorted".format(field))
    return normalized


def _require_id_list(value: Any, field: str) -> List[str]:
    if not isinstance(value, list):
        raise ValueError("{} must be a JSON array".format(field))
    normalized = [_require_id(item, "{} item".format(field)) for item in value]
    if len(set(normalized)) != len(normalized):
        raise ValueError("{} must not contain duplicates".format(field))
    if normalized != sorted(normalized):
        raise ValueError("{} must be lexicographically sorted".format(field))
    return normalized


def _validate_task_binding(
    row: Mapping[str, Any],
    *,
    corpus: Optional[EvaluationCorpus],
    task_record: Optional[Mapping[str, Any]],
    task_manifest_digest: Optional[str],
) -> None:
    if corpus is None:
        raise ValueError("SFT validation requires the evaluator-owned EvaluationCorpus")
    authoritative = corpus.get(str(row.get("task_id"))).public_manifest_record()
    expected_manifest_digest = corpus.manifest_digest()
    if task_manifest_digest is None:
        task_manifest_digest = expected_manifest_digest
    _require_digest(task_manifest_digest, "task_manifest_digest")
    _require_digest(row["task_manifest_digest"], "task_manifest_digest")
    if task_manifest_digest != expected_manifest_digest or row["task_manifest_digest"] != expected_manifest_digest:
        raise ValueError("task_manifest_digest does not match the evaluator-owned immutable manifest")
    if task_record is not None and dict(task_record) != authoritative:
        raise ValueError("caller task_record does not match the evaluator-owned manifest lookup")
    _require_digest(row["task_record_digest"], "task_record_digest")
    if row["task_record_digest"] != digest_for(authoritative):
        raise ValueError("task_record_digest does not match the evaluator-owned task record")
    if authoritative["template_id"] != row["task_id"]:
        raise ValueError("task_id does not match the immutable task record")
    if authoritative["family_id"] != row["task_family"] or authoritative["split"] != row["split"]:
        raise ValueError("task family or split does not match the immutable task record")
    if authoritative["public_locus"] != row["public_locus"] or authoritative["public_rule_id"] != row["public_rule_id"]:
        raise ValueError("public locus or rule does not match the immutable task record")


def validate_sft_row(
    row: Mapping[str, Any],
    *,
    corpus: Optional[EvaluationCorpus] = None,
    task_record: Optional[Mapping[str, Any]] = None,
    task_manifest_digest: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate one closed, train-only row and return a detached mapping."""

    if not isinstance(row, Mapping):
        raise ValueError("SFT row must be an object")
    extra = set(row) - SFT_FIELDS
    missing = SFT_FIELDS - set(row)
    if extra:
        raise ValueError("SFT row contains forbidden fields: {}".format(", ".join(sorted(extra))))
    if missing:
        raise ValueError("SFT row is missing fields: {}".format(", ".join(sorted(missing))))
    if set(row) & _FORBIDDEN_ROW_FIELDS:
        raise ValueError("SFT row contains raw prompt, source, reasoning, telemetry, or private material")
    if row["schema_version"] != SFT_SCHEMA_VERSION:
        raise ValueError("unsupported SFT row schema")
    for field in ("row_id", "task_id", "task_family", "arm"):
        _require_id(row[field], field)
    for field in ("attempt_id", "public_locus", "public_rule_id"):
        _require_id(row[field], field)
    _validate_task_binding(
        row,
        corpus=corpus,
        task_record=task_record,
        task_manifest_digest=task_manifest_digest,
    )
    if row["task_family"] not in FAMILY_BY_ID:
        raise ValueError("task_family is outside the frozen evaluation family enum")
    if row["split"] != "train":
        raise ValueError("SFT rows may contain training tasks only; dev and held-out are evaluator inputs")
    task_parts = row["task_id"].split("-")
    if len(task_parts) != 5 or task_parts[0:2] != ["egv", row["task_family"].lower()] or task_parts[2] != "train" or task_parts[4] != "v1":
        raise ValueError("task_id does not match the frozen train-task ID contract")
    try:
        ordinal = int(task_parts[3])
    except ValueError as exc:
        raise ValueError("task_id ordinal is not an integer") from exc
    if ordinal < 1 or ordinal > FAMILY_BY_ID[row["task_family"]].train:
        raise ValueError("task_id is outside the frozen train-task allocation")
    if not isinstance(row["attempt_index"], int) or isinstance(row["attempt_index"], bool) or row["attempt_index"] < 1:
        raise ValueError("attempt_index must be a positive integer")
    if not isinstance(row["prompt_template_ids"], list) or not row["prompt_template_ids"]:
        raise ValueError("prompt_template_ids must be a non-empty array")
    prompt_ids = [_require_id(item, "prompt_template_ids item") for item in row["prompt_template_ids"]]
    if len(set(prompt_ids)) != len(prompt_ids) or any(item not in PROMPT_IDS for item in prompt_ids):
        raise ValueError("prompt_template_ids must be a unique subset of the five frozen prompts")
    if prompt_ids != sorted(prompt_ids, key=PROMPT_IDS.index):
        raise ValueError("prompt_template_ids must use frozen prompt order")
    for field in (
        "prompt_digest",
        "proposed_mutation_digest",
        "candidate_artifact_digest",
        "verdict_receipt_digest",
        "failure_family_root",
        "input_digest",
        "output_digest",
    ):
        _require_digest(row[field], field)
    _require_id_list(row["retrieved_evidence_ids"], "retrieved_evidence_ids")
    _require_id_list(row["dependency_ids"], "dependency_ids")
    validate_diagnostic(row["diagnostic_enum"])
    validate_resource_bucket(row["resource_bucket"])
    validate_requested_authority(row["requested_authority"])
    validate_disposition(row["promotion_disposition"])
    incident = row["infrastructure_incident_id"]
    if row["diagnostic_enum"] == "INTERNAL_ERROR":
        _require_id(incident, "infrastructure_incident_id")
    elif incident is not None:
        raise ValueError("non-INTERNAL_ERROR SFT rows cannot carry infrastructure_incident_id")
    expected_failure_root = failure_family_for_attempt(
        row["task_family"],
        row["diagnostic_enum"],
        row["public_locus"],
        row["public_rule_id"],
        attempt_id=row["attempt_id"],
        infrastructure_incident_id=incident,
    )
    if row["failure_family_root"] != expected_failure_root:
        raise ValueError("failure_family_root must bind the task's actual public locus and rule")
    unsigned = dict(row)
    supplied_row_id = unsigned.pop("row_id")
    expected_row_id = content_id("sft", unsigned)
    if supplied_row_id != expected_row_id:
        raise ValueError("row_id must be content-derived from the closed row")
    return dict(row)


def build_sft_row(
    *,
    task_id: str,
    task_family: str,
    arm: str,
    attempt_index: int,
    prompt_template_ids: Sequence[str],
    prompt_digest: str,
    retrieved_evidence_ids: Sequence[str],
    proposed_mutation_digest: str,
    candidate_artifact_digest: str,
    verdict_receipt_digest: str,
    diagnostic_enum: str,
    resource_bucket: str,
    dependency_ids: Sequence[str],
    requested_authority: str,
    promotion_disposition: str,
    input_digest: str,
    output_digest: str,
    public_locus: str,
    public_rule_id: str,
    corpus: Optional[EvaluationCorpus] = None,
    task_record: Optional[Mapping[str, Any]] = None,
    task_manifest_digest: Optional[str] = None,
    attempt_id: Optional[str] = None,
    infrastructure_incident_id: Optional[str] = None,
) -> Dict[str, Any]:
    if corpus is None:
        raise ValueError("building SFT rows requires the evaluator-owned EvaluationCorpus")
    authoritative = corpus.get(task_id).public_manifest_record()
    if task_record is not None and dict(task_record) != authoritative:
        raise ValueError("caller task_record does not match the evaluator-owned manifest lookup")
    expected_manifest_digest = corpus.manifest_digest()
    if task_manifest_digest is not None and task_manifest_digest != expected_manifest_digest:
        raise ValueError("task_manifest_digest does not match the evaluator-owned immutable manifest")
    task_manifest_digest = expected_manifest_digest
    diagnostic = validate_diagnostic(diagnostic_enum)
    if diagnostic == "INTERNAL_ERROR" and not infrastructure_incident_id:
        raise ValueError("INTERNAL_ERROR SFT rows require a unique infrastructure_incident_id")
    if diagnostic != "INTERNAL_ERROR" and infrastructure_incident_id is not None:
        raise ValueError("non-INTERNAL_ERROR SFT rows cannot carry infrastructure_incident_id")
    if attempt_id is None:
        attempt_id = content_id(
            "attempt",
            {
                "task_id": task_id,
                "attempt_index": attempt_index,
                "candidate_artifact_digest": candidate_artifact_digest,
            },
        )
    row: Dict[str, Any] = {
        "schema_version": SFT_SCHEMA_VERSION,
        "task_id": task_id,
        "task_family": task_family,
        "split": "train",
        "arm": arm,
        "attempt_index": attempt_index,
        "attempt_id": attempt_id,
        "task_record_digest": digest_for(authoritative),
        "task_manifest_digest": task_manifest_digest,
        "public_locus": public_locus,
        "public_rule_id": public_rule_id,
        "prompt_template_ids": list(prompt_template_ids),
        "prompt_digest": prompt_digest,
        "retrieved_evidence_ids": sorted(retrieved_evidence_ids),
        "proposed_mutation_digest": proposed_mutation_digest,
        "candidate_artifact_digest": candidate_artifact_digest,
        "verdict_receipt_digest": verdict_receipt_digest,
        "diagnostic_enum": diagnostic,
        "resource_bucket": resource_bucket,
        "infrastructure_incident_id": infrastructure_incident_id,
        "failure_family_root": failure_family_for_attempt(
            task_family,
            diagnostic,
            public_locus,
            public_rule_id,
            attempt_id=attempt_id,
            infrastructure_incident_id=infrastructure_incident_id,
        ),
        "dependency_ids": sorted(dependency_ids),
        "requested_authority": requested_authority,
        "promotion_disposition": promotion_disposition,
        "input_digest": input_digest,
        "output_digest": output_digest,
    }
    row["row_id"] = content_id("sft", row)
    return validate_sft_row(row, corpus=corpus, task_manifest_digest=task_manifest_digest)


def validate_sft_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    corpus: Optional[EvaluationCorpus] = None,
    task_records: Optional[Mapping[str, Mapping[str, Any]]] = None,
    task_manifest_digest: str,
) -> List[Dict[str, Any]]:
    if corpus is None:
        raise ValueError("validating SFT rows requires the evaluator-owned EvaluationCorpus")
    normalized = [
        validate_sft_row(
            row,
            corpus=corpus,
            task_record=(task_records or {}).get(str(row.get("task_id"))),
            task_manifest_digest=task_manifest_digest,
        )
        for row in rows
    ]
    ids = [row["row_id"] for row in normalized]
    if len(set(ids)) != len(ids):
        raise ValueError("SFT row IDs must be unique")
    return normalized


__all__ = [
    "SFT_FIELDS",
    "SFT_SCHEMA_VERSION",
    "build_sft_row",
    "validate_sft_row",
    "validate_sft_rows",
]
