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
from ..receipts import ReceiptJournal, ReceiptSigner
from .artifacts import ContentAddressedArtifactStore
from .authority import AuthorityBroker, AuthorityPolicy, DockerEnforcedRuntime
from .controller import EvaluationResult, EvaluatorController, HiddenEvaluatorRunner
from .diagnostics import Diagnostic
from .learnable import (
    SPLITS_V2,
    LearnableCorpusV2,
    LearnableTask,
    build_reference_source,
    validate_learnable_manifest,
    validate_learnable_private_manifest,
    validate_model_visible_package,
)
from .sandbox import DockerCandidateSandbox, DockerSandboxConfig


LEARNABLE_RUNTIME_VERSION = "egv-learnable-production-runtime-v2"
LEARNABLE_CORPUS_VALIDATION_VERSION = "egv-learnable-corpus-validation-proof-v2"
LEARNABLE_CORPUS_PERMIT_VERSION = "egv-learnable-corpus-validation-permit-v1"
CORPUS_VALIDATION_SCOPE = "REFERENCE_AND_STARTER_ONLY_NO_MODEL_REQUESTS"
PUBLIC_LOCUS = "module:main"


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

    config.validate()
    manifest: Dict[str, Any] = {
        "schema_version": LEARNABLE_RUNTIME_VERSION,
        "backend": "docker-enforced-v1",
        "image_ref": config.image_ref,
        "pinned_image_id": config.pinned_image_id,
        "candidate_contract": "pure-return-v1",
        "network": "none",
        "read_only_root": True,
        "cap_drop": "ALL",
        "no_new_privileges": True,
        "user": config.user,
        "pids_limit": config.pids_limit,
        "memory_limit": config.memory_limit,
        "tmpfs_size": config.tmpfs_size,
        "timeout_seconds": config.timeout_seconds,
        "output_limit": config.output_limit,
    }
    manifest["manifest_digest"] = digest_for(manifest)
    return manifest


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
        candidate_id = "{}-{}-{}".format(
            candidate_kind,
            digest_for({"public_task_id": task.task_id, "source_digest": source_digest})[:24],
            address.case_ordinal,
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

    selected_splits = tuple(splits)
    if selected_splits != ("blind",) or any(split not in SPLITS_V2 for split in selected_splits):
        raise ValueError("production corpus proof is frozen to the blind split")
    selected_config = config or DockerSandboxConfig.from_environment()
    permit.validate(corpus, selected_config)
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    tasks = tuple(task for split in selected_splits for task in corpus.split(split))

    # Build every reference solely from validated model-visible packages before
    # any hidden case is addressed by the controller adapter.
    reference_sources: Dict[str, bytes] = {}
    starter_sources: Dict[str, bytes] = {}
    for task in tasks:
        package = validate_model_visible_package(corpus.model_visible_package(task.task_id))
        reference_sources[task.task_id] = build_reference_source(package["task"])
        starter_sources[task.task_id] = str(package["task"]["starter_source"]).encode("utf-8")

    hidden_runner, addresses = build_learnable_hidden_runner(tasks)
    sandbox = DockerCandidateSandbox(root / "sandbox", config=selected_config)
    sandbox.validate_production_integrity()
    negative_controls = sandbox.negative_controls(root / "private" / "hidden-never-mounted.json")
    runtime = DockerEnforcedRuntime(selected_config)
    policy = AuthorityPolicy.candidate_execution()
    signer = ReceiptSigner.generate()
    journal = ReceiptJournal(root / "private" / "receipts.jsonl", signer.public_key)
    delivered: list[Dict[str, Any]] = []
    protocol_digest = digest_for(
        {
            "schema_version": LEARNABLE_CORPUS_VALIDATION_VERSION,
            "permit_digest": permit.permit_digest,
            "split": "blind",
        }
    )
    controller = EvaluatorController(
        sandbox=sandbox,
        hidden_runner=hidden_runner,
        broker=AuthorityBroker(runtime, policy),
        signer=signer,
        journal=journal,
        ingest=lambda receipt: delivered.append(dict(receipt)),
        campaign_id="egv-learnable-v2-corpus-validation",
        protocol_digest=protocol_digest,
        policy_digest=policy.digest,
        artifact_store=ContentAddressedArtifactStore(root / "private" / "artifacts"),
    )

    reference_results = []
    starter_results = []
    family_counts: Dict[str, Dict[str, int]] = {
        task.family_id: {"reference_passed": 0, "reference_total": 0, "starter_rejected": 0, "starter_total": 0}
        for task in tasks
    }
    for task in tasks:
        task_reference = _evaluate_task_source(
            controller,
            task,
            addresses,
            reference_sources[task.task_id],
            candidate_kind="reference",
        )
        task_starter = _evaluate_task_source(
            controller,
            task,
            addresses,
            starter_sources[task.task_id],
            candidate_kind="starter",
            case_limit=1,
        )
        reference_results.extend(task_reference)
        starter_results.extend(task_starter)
        family = family_counts[task.family_id]
        family["reference_total"] += len(task_reference)
        family["reference_passed"] += sum(result.diagnostic_enum == Diagnostic.PASS.value for result in task_reference)
        family["starter_total"] += len(task_starter)
        family["starter_rejected"] += sum(
            result.diagnostic_enum == Diagnostic.WRONG_OUTPUT.value for result in task_starter
        )

    journal_evidence = journal.verify()
    reference_pass_count = sum(result.diagnostic_enum == Diagnostic.PASS.value for result in reference_results)
    starter_rejection_count = sum(result.diagnostic_enum == Diagnostic.WRONG_OUTPUT.value for result in starter_results)
    report: Dict[str, Any] = {
        "schema_version": LEARNABLE_CORPUS_VALIDATION_VERSION,
        "evidence_scope": "production-corpus-validation-no-model-quality-claim",
        "production_acceptance": True,
        "task_bearing_model_requests_permitted": False,
        "permit_digest": permit.permit_digest,
        "public_manifest_digest": corpus.public_manifest()["manifest_digest"],
        "private_manifest_digest": corpus.private_manifest()["manifest_digest"],
        "runtime_manifest_digest": runtime_manifest(selected_config)["manifest_digest"],
        "split": "blind",
        "task_count": len(tasks),
        "hidden_case_count": len(reference_results),
        "reference_pass_count": reference_pass_count,
        "reference_all_hidden_cases_passed": reference_pass_count == len(reference_results),
        "starter_control_count": len(starter_results),
        "starter_wrong_output_count": starter_rejection_count,
        "starter_controls_rejected": starter_rejection_count == len(starter_results),
        "family_counts": family_counts,
        "builder_input": "validated-closed-model-visible-package-only",
        "hidden_cases_passed_to_builder": False,
        "receipt_count": journal_evidence["count"],
        "receipt_chain_head": journal_evidence["head"],
        "receipt_chain_valid": journal_evidence["chain_valid"],
        "receipt_delivery_count": len(delivered),
        "sandbox_negative_control_count": len(negative_controls),
        "sandbox_negative_controls_all_passed": all(negative_controls.values()),
        "sandbox_negative_controls_digest": digest_for(negative_controls),
        "hidden_values_exposed": False,
        "private_paths_exposed": False,
    }
    report["proof_digest"] = digest_for(report)
    return report


__all__ = [
    "CORPUS_VALIDATION_SCOPE",
    "CorpusValidationPermit",
    "LEARNABLE_CORPUS_PERMIT_VERSION",
    "LEARNABLE_CORPUS_VALIDATION_VERSION",
    "LEARNABLE_RUNTIME_VERSION",
    "LearnableCaseAddress",
    "build_corpus_validation_permit",
    "build_learnable_hidden_runner",
    "run_production_solvability_proof",
    "runtime_manifest",
]
