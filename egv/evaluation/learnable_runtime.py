"""Receipt-backed production evaluation for the learnable-v2 corpus.

The public-only reference builder is deliberately invoked before evaluator
cases are addressed.  Candidate bytes then cross the existing exact
``DockerCandidateSandbox`` and ``EvaluatorController`` boundary.  Returned
proof records contain counts and digests only; hidden inputs and expected
outputs remain evaluator-private.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from ..canonical import canonical_bytes, digest_bytes, digest_for
from ..errors import EGVError
from .controller import EvaluationResult, EvaluatorController, HiddenEvaluatorRunner
from .learnable import (
    LearnableCorpusV2,
    LearnableTask,
    validate_learnable_manifest,
    validate_learnable_private_manifest,
    validate_model_visible_package,
)
from .learnable_freeze import CORPUS_VALIDATION_SCOPE
from .sandbox import DockerSandboxConfig


LEARNABLE_RUNTIME_VERSION = "egv-learnable-production-runtime-v2"
LEARNABLE_CORPUS_VALIDATION_VERSION = "egv-learnable-corpus-validation-proof-v2"
LEARNABLE_CORPUS_PERMIT_VERSION = "egv-learnable-corpus-validation-permit-v1"
PUBLIC_LOCUS = "module:main"


def _corpus_validation_accepted(
    *,
    reference_pass_count: int,
    reference_total: int,
    starter_rejection_count: int,
    starter_total: int,
    receipt_count: int,
    receipt_delivery_count: int,
    receipt_chain_valid: bool,
    negative_controls: Mapping[str, bool],
) -> bool:
    """Compute the production disposition from every required proof surface."""

    expected_receipt_count = 3 * (reference_total + starter_total)
    return bool(
        reference_total > 0
        and starter_total > 0
        and reference_pass_count == reference_total
        and starter_rejection_count == starter_total
        and receipt_chain_valid
        and receipt_count == expected_receipt_count
        and receipt_delivery_count == expected_receipt_count
        and negative_controls
        and all(negative_controls.values())
    )


@dataclass(frozen=True)
class LearnableCaseAddress:
    """Public address for one evaluator-private case."""

    controller_task_id: str
    public_task_id: str
    family_id: str
    case_ordinal: int
    public_rule_id: str

    def public_record(self) -> Dict[str, Any]:
        return {
            "schema_version": "egv-learnable-case-address-v1",
            "controller_task_id": self.controller_task_id,
            "public_task_id": self.public_task_id,
            "family_id": self.family_id,
            "case_ordinal": self.case_ordinal,
            "public_locus": PUBLIC_LOCUS,
            "public_rule_id": self.public_rule_id,
        }


@dataclass(frozen=True)
class CorpusValidationPermit:
    """Narrow Gate-A permit; it cannot authorize a model request."""

    public_manifest_digest: str
    private_manifest_digest: str
    runtime_manifest_digest: str
    scope: str
    permit_digest: str

    def _core(self) -> Dict[str, Any]:
        return {
            "schema_version": LEARNABLE_CORPUS_PERMIT_VERSION,
            "public_manifest_digest": self.public_manifest_digest,
            "private_manifest_digest": self.private_manifest_digest,
            "runtime_manifest_digest": self.runtime_manifest_digest,
            "scope": self.scope,
        }

    def validate(self, corpus: LearnableCorpusV2, config: DockerSandboxConfig) -> None:
        public_manifest = validate_learnable_manifest(corpus.public_manifest(), corpus)
        private_manifest = validate_learnable_private_manifest(corpus.private_manifest(), corpus)
        expected_runtime = runtime_manifest(config)
        if self.scope != CORPUS_VALIDATION_SCOPE:
            raise ValueError("corpus-validation permit has task-bearing scope")
        if self.public_manifest_digest != public_manifest["manifest_digest"]:
            raise ValueError("corpus-validation permit differs from the public corpus")
        if self.private_manifest_digest != private_manifest["manifest_digest"]:
            raise ValueError("corpus-validation permit differs from the private corpus")
        if self.runtime_manifest_digest != expected_runtime["manifest_digest"]:
            raise ValueError("corpus-validation permit differs from the production runtime")
        if self.permit_digest != digest_for(self._core()):
            raise ValueError("corpus-validation permit digest mismatch")

    def public_record(self) -> Dict[str, Any]:
        return {**self._core(), "permit_digest": self.permit_digest}


def runtime_manifest(config: DockerSandboxConfig) -> Dict[str, Any]:
    """Return the closed, endpoint-free production sandbox contract."""

    from .learnable_gate import build_gate_a_runtime_manifest

    return build_gate_a_runtime_manifest(config)


def build_corpus_validation_permit(
    corpus: LearnableCorpusV2, config: Optional[DockerSandboxConfig] = None
) -> CorpusValidationPermit:
    """Freeze the narrow permit needed to execute public reference bytes."""

    selected = config or DockerSandboxConfig.from_environment()
    public_manifest = validate_learnable_manifest(corpus.public_manifest(), corpus)
    private_manifest = validate_learnable_private_manifest(corpus.private_manifest(), corpus)
    runtime = runtime_manifest(selected)
    core = {
        "schema_version": LEARNABLE_CORPUS_PERMIT_VERSION,
        "public_manifest_digest": public_manifest["manifest_digest"],
        "private_manifest_digest": private_manifest["manifest_digest"],
        "runtime_manifest_digest": runtime["manifest_digest"],
        "scope": CORPUS_VALIDATION_SCOPE,
    }
    return CorpusValidationPermit(
        public_manifest_digest=core["public_manifest_digest"],
        private_manifest_digest=core["private_manifest_digest"],
        runtime_manifest_digest=core["runtime_manifest_digest"],
        scope=core["scope"],
        permit_digest=digest_for(core),
    )


def _case_task_id(task: LearnableTask, case_ordinal: int) -> str:
    return "egv-lv2-case-{}".format(
        digest_for(
            {
                "schema_version": "egv-learnable-controller-address-v1",
                "public_task_id": task.task_id,
                "case_ordinal": case_ordinal,
            }
        )[:32]
    )


def build_learnable_hidden_runner(
    tasks: Sequence[LearnableTask], *, evaluator_revision: str = LEARNABLE_RUNTIME_VERSION
) -> Tuple[HiddenEvaluatorRunner, Tuple[LearnableCaseAddress, ...]]:
    """Bind hidden cases to public-only controller addresses."""

    cases: Dict[str, Tuple[Any, bytes, Optional[str]]] = {}
    public_loci: Dict[str, str] = {}
    public_records: Dict[str, Mapping[str, str]] = {}
    addresses = []
    seen_public_tasks = set()
    for task in tasks:
        if task.task_id in seen_public_tasks:
            raise ValueError("learnable runtime received a duplicate public task")
        seen_public_tasks.add(task.task_id)
        package = validate_model_visible_package(
            {
                "schema_version": "egv-learnable-model-package-v2",
                "task": dict(task.public_record),
                "package_digest": digest_for(
                    {
                        "schema_version": "egv-learnable-model-package-v2",
                        "task": dict(task.public_record),
                    }
                ),
            }
        )
        public_rule_id = str(package["task"]["public_spec_digest"])
        for case in task.hidden_cases:
            controller_task_id = _case_task_id(task, case.ordinal)
            if controller_task_id in cases:
                raise ValueError("learnable runtime case address collision")
            cases[controller_task_id] = (
                case.input_value,
                canonical_bytes(case.expected_output) + b"\n",
                None,
            )
            public_loci[controller_task_id] = PUBLIC_LOCUS
            public_records[controller_task_id] = {
                "family_id": task.family_id,
                "public_locus": PUBLIC_LOCUS,
                "public_rule_id": public_rule_id,
            }
            addresses.append(
                LearnableCaseAddress(
                    controller_task_id,
                    task.task_id,
                    task.family_id,
                    case.ordinal,
                    public_rule_id,
                )
            )
    return (
        HiddenEvaluatorRunner(
            cases,
            evaluator_revision=evaluator_revision,
            public_loci=public_loci,
            public_records=public_records,
        ),
        tuple(addresses),
    )


def _evaluate_task_source(
    controller: EvaluatorController,
    task: LearnableTask,
    addresses: Sequence[LearnableCaseAddress],
    source: bytes,
    *,
    candidate_kind: str,
    case_limit: Optional[int] = None,
) -> Tuple[EvaluationResult, ...]:
    selected = [address for address in addresses if address.public_task_id == task.task_id]
    selected.sort(key=lambda address: address.case_ordinal)
    if case_limit is not None:
        selected = selected[:case_limit]
    if not selected:
        raise ValueError("learnable runtime has no cases for the public task")
    case_by_ordinal = {case.ordinal: case for case in task.hidden_cases}
    source_digest = digest_bytes(source)
    results = []
    for address in selected:
        case = case_by_ordinal[address.case_ordinal]
        candidate_id = build_learnable_candidate_id(
            candidate_kind=candidate_kind,
            public_task_id=task.task_id,
            source_digest=source_digest,
            case_ordinal=address.case_ordinal,
        )
        results.append(
            controller.evaluate(
                candidate_id=candidate_id,
                task_id=address.controller_task_id,
                source=source,
                opaque_input=case.input_value,
                declared_locus=PUBLIC_LOCUS,
            )
        )
    return tuple(results)


def build_learnable_candidate_id(
    *,
    candidate_kind: str,
    public_task_id: str,
    source_digest: str,
    case_ordinal: int,
) -> str:
    """Derive the exact candidate identity shared by execution and replay."""

    return "{}-{}-{}".format(
        candidate_kind,
        digest_for({"public_task_id": public_task_id, "source_digest": source_digest})[:24],
        case_ordinal,
    )


def run_production_solvability_proof(
    corpus: LearnableCorpusV2,
    workspace: Union[Path, str],
    *,
    permit: CorpusValidationPermit,
    splits: Sequence[str] = ("blind",),
    config: Optional[DockerSandboxConfig] = None,
) -> Dict[str, Any]:
    """Run public reference and starter controls through production Docker.

    The permit scope explicitly forbids model requests.  This function is a
    corpus/sandbox proof only and cannot establish model quality or issue the
    later task-bearing campaign gate.
    """

    raise EGVError(
        "direct in-process corpus proof is disabled; use run_gate_a_corpus_validation "
        "with external freeze/controller keys and three spawned stages"
    )


def run_gate_a_corpus_validation(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Compatibility export for the spawn-only Gate-A production entrypoint."""

    from .learnable_gate import run_gate_a_corpus_validation as run_spawned_gate

    return run_spawned_gate(*args, **kwargs)


__all__ = [
    "CORPUS_VALIDATION_SCOPE",
    "CorpusValidationPermit",
    "LEARNABLE_CORPUS_PERMIT_VERSION",
    "LEARNABLE_CORPUS_VALIDATION_VERSION",
    "LEARNABLE_RUNTIME_VERSION",
    "LearnableCaseAddress",
    "build_corpus_validation_permit",
    "build_learnable_candidate_id",
    "build_learnable_hidden_runner",
    "run_production_solvability_proof",
    "run_gate_a_corpus_validation",
    "runtime_manifest",
]
