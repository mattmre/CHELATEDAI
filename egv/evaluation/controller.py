"""Evaluator controller, hidden runner, and receipt reconciliation boundary."""

from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

from ..canonical import GENESIS_HASH, canonical_bytes, digest_bytes, digest_for, failure_family_root, utc_now_iso
from ..ledger import active_writer_count
from ..receipts import ReceiptJournal, ReceiptSigner, receipt_hash
from .artifacts import ContentAddressedArtifactStore
from .authority import AuthorityBroker
from .boundary import install_evaluator_sqlite_jail
from .diagnostics import Diagnostic
from .errors import AuthorityDenied
from .sandbox import DockerCandidateSandbox, SandboxResult, validate_candidate_source_contract


@dataclass(frozen=True)
class HiddenVerdict:
    task_id: str
    diagnostic_enum: str
    output_digest: str
    infrastructure_loss: bool


class HiddenEvaluatorRunner:
    """Evaluator-only comparator that never returns expected output details."""

    def __init__(
        self,
        cases: Mapping[str, Union[Tuple[Any, bytes], Tuple[Any, bytes, Optional[str]]]],
        *,
        evaluator_revision: str,
        public_loci: Optional[Mapping[str, str]] = None,
        public_records: Optional[Mapping[str, Mapping[str, str]]] = None,
        resource_limits: Optional[Mapping[str, Optional[str]]] = None,
    ) -> None:
        self._cases: Dict[str, Tuple[Any, bytes, Optional[str]]] = {}
        for task_id, case in cases.items():
            if len(case) == 2:
                input_value, expected = case
                limit = None
            else:
                input_value, expected, limit = case  # type: ignore[misc]
            self._cases[task_id] = (input_value, expected, limit)
        self.evaluator_revision = evaluator_revision
        self._public_loci = dict(public_loci or {})
        self._public_records = {key: dict(value) for key, value in (public_records or {}).items()}
        if resource_limits:
            self._resource_limits = dict(resource_limits)
        else:
            self._resource_limits = {task_id: case[2] for task_id, case in self._cases.items()}

    @classmethod
    def from_corpus(cls, corpus: Any, *, evaluator_revision: str = "egv-evaluator-v1") -> "HiddenEvaluatorRunner":
        cases = {
            repo.template_id: (
                repo.evaluator_input,
                canonical_bytes(repo.expected_output) + b"\n",
                repo.hidden_spec.get("resource_limit"),
            )
            for repo in corpus.hidden_repositories()
        }
        public_loci = {repo.template_id: repo.public_locus for repo in corpus.hidden_repositories()}
        public_records = {repo.template_id: repo.public_manifest_record() for repo in corpus.hidden_repositories()}
        resource_limits = {repo.template_id: repo.hidden_spec.get("resource_limit") for repo in corpus.hidden_repositories()}
        return cls(
            cases,
            evaluator_revision=evaluator_revision,
            public_loci=public_loci,
            public_records=public_records,
            resource_limits=resource_limits,
        )

    def locus_matches(self, task_id: str, declared_locus: str) -> bool:
        return self._public_loci.get(task_id) == declared_locus

    def has_task(self, task_id: str) -> bool:
        return task_id in self._cases

    def resource_limit(self, task_id: str) -> Optional[str]:
        """Return the evaluator-private frozen resource profile for a task."""

        return self._resource_limits.get(task_id)

    def public_record(self, task_id: str) -> Optional[Dict[str, str]]:
        record = self._public_records.get(task_id)
        return dict(record) if record is not None else None

    def evaluate(self, task_id: str, output_bytes: bytes, *, opaque_input: Any = None) -> HiddenVerdict:
        case = self._cases.get(task_id)
        if case is None:
            return HiddenVerdict(task_id, Diagnostic.PROTOCOL_VIOLATION.value, digest_bytes(output_bytes), False)
        expected_input, expected, _resource_limit = case
        if opaque_input is not None and canonical_bytes(opaque_input) != canonical_bytes(expected_input):
            return HiddenVerdict(task_id, Diagnostic.PROTOCOL_VIOLATION.value, digest_bytes(output_bytes), False)
        diagnostic = Diagnostic.PASS.value if output_bytes == expected else Diagnostic.WRONG_OUTPUT.value
        return HiddenVerdict(task_id, diagnostic, digest_bytes(output_bytes), False)


@dataclass(frozen=True)
class EvaluationResult:
    candidate_id: str
    task_id: str
    candidate_artifact_digest: str
    diagnostic_enum: str
    resource_bucket: str
    disposition: str
    infrastructure_loss: bool
    receipt_ids: Tuple[str, ...]
    output_digest: str
    infrastructure_incident_id: Optional[str] = None
    failure_family_root: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "candidate_id": self.candidate_id,
            "task_id": self.task_id,
            "candidate_artifact_digest": self.candidate_artifact_digest,
            "diagnostic_enum": self.diagnostic_enum,
            "resource_bucket": self.resource_bucket,
            "disposition": self.disposition,
            "infrastructure_loss": self.infrastructure_loss,
            "receipt_ids": list(self.receipt_ids),
            "output_digest": self.output_digest,
        }
        if self.infrastructure_incident_id is not None:
            result["infrastructure_incident_id"] = self.infrastructure_incident_id
        if self.failure_family_root is not None:
            result["failure_family_root"] = self.failure_family_root
        return result


def reconcile_journal(
    journal: ReceiptJournal,
    ingest: Callable[[Mapping[str, Any]], Any],
    *,
    existing_receipt_ids: Optional[Sequence[str]] = None,
) -> int:
    """Deliver every journal record through an idempotent sole-writer sink."""

    existing = set(existing_receipt_ids or ())
    added = 0
    for receipt in journal.receipts():
        ingest(receipt)
        if receipt["receipt_id"] not in existing:
            added += 1
            existing.add(receipt["receipt_id"])
    return added


class EvaluatorController:
    """Controller-side orchestration with evaluator-private receipt durability."""

    def __init__(
        self,
        *,
        sandbox: DockerCandidateSandbox,
        hidden_runner: HiddenEvaluatorRunner,
        broker: AuthorityBroker,
        signer: ReceiptSigner,
        journal: ReceiptJournal,
        ingest: Callable[[Mapping[str, Any]], Any],
        campaign_id: str,
        protocol_digest: str,
        policy_digest: str,
        artifact_store: Optional[ContentAddressedArtifactStore] = None,
        clock: Callable[[], str] = utc_now_iso,
    ) -> None:
        if active_writer_count() > 0:
            raise AuthorityDenied("EVALUATOR_LEDGER_WRITER_COHOSTING_FORBIDDEN")
        # Install before any evaluator-owned work.  The spawned production
        # process installs the same immutable hook as its first action; this
        # second call makes direct production construction equally fail closed.
        install_evaluator_sqlite_jail()
        self.sandbox = sandbox
        if not getattr(sandbox, "enforceable", False):
            raise AuthorityDenied("AUTHORITY_RUNTIME_UNAVAILABLE: production Evaluation requires Docker isolation")
        self.hidden_runner = hidden_runner
        self.broker = broker
        self.artifact_store = artifact_store
        self.signer = signer
        self.journal = journal
        self.ingest = ingest
        self.campaign_id = campaign_id
        self.protocol_digest = protocol_digest
        self.policy_digest = policy_digest
        self.clock = clock
        self.quarantine: list[Dict[str, Any]] = []

    def _record_receipt(self, fields: Mapping[str, Any], *, idempotency_key: str) -> Dict[str, Any]:
        existing = self.journal.receipts()
        sequence = len(existing) + 1
        previous = receipt_hash(existing[-1]) if existing else GENESIS_HASH
        receipt = self.signer.sign_receipt(
            fields,
            sequence=sequence,
            previous_receipt_hash=previous,
            idempotency_key=idempotency_key,
        )
        try:
            stored = self.journal.append(receipt)
            self.ingest(stored)
            return stored
        except Exception as exc:
            self.quarantine.append({"idempotency_key": idempotency_key, "reason": type(exc).__name__})
            raise

    def _common(self, *, task_id: str, candidate_id: str, artifact_digest: str) -> Dict[str, Any]:
        common = {
            "campaign_id": self.campaign_id,
            "run_id": "run-evaluation",
            "task_id": task_id,
            "candidate_id": candidate_id,
            "candidate_artifact_digest": artifact_digest,
            "protocol_digest": self.protocol_digest,
            "policy_digest": self.policy_digest,
            "evaluator_digest": digest_for(self.hidden_runner.evaluator_revision),
        }
        record = self.hidden_runner.public_record(task_id)
        if record is not None:
            common.update(
                {
                    "task_family": record["family_id"],
                    "normalized_public_locus": record["public_locus"],
                    "public_rule_id": record["public_rule_id"],
                }
            )
        return common

    @staticmethod
    def _infrastructure_incident(candidate_id: str, reason: str) -> str:
        return digest_for(
            {
                "kind": "evaluation-infrastructure-loss",
                "candidate_id": candidate_id,
                "reason": reason,
                "nonce": uuid.uuid4().hex,
            }
        )

    def _failure_root(self, task_id: str, diagnostic: str, incident_id: str) -> str:
        record = self.hidden_runner.public_record(task_id)
        if record is not None:
            return failure_family_root(
                record["family_id"],
                diagnostic,
                record["public_locus"],
                record["public_rule_id"],
                infrastructure_incident_id=incident_id,
            )
        raise RuntimeError("INTERNAL_ERROR cannot be signed without the immutable public task record")

    def evaluate(
        self,
        *,
        candidate_id: str,
        task_id: str,
        source: bytes,
        opaque_input: Any,
        requested_authority: str = "EXECUTE_CANDIDATE",
        declared_locus: str,
        candidate_source_path: Optional[str] = None,
    ) -> EvaluationResult:
        artifact_digest = digest_bytes(source)
        if self.artifact_store is not None:
            stored = self.artifact_store.put(source, media_type="text/x-python", role="private-candidate-source")
            if stored.digest != artifact_digest:
                raise RuntimeError("content-addressed candidate artifact digest mismatch")
        common = self._common(task_id=task_id, candidate_id=candidate_id, artifact_digest=artifact_digest)
        if not self.hidden_runner.has_task(task_id):
            authority_receipt = self._record_receipt(
                {
                    **common,
                    "receipt_type": "AUTHORITY",
                    "request_id": "request-authority-{}".format(candidate_id),
                    "decision": "DENY",
                    "diagnostic_enum": Diagnostic.PROTOCOL_VIOLATION.value,
                },
                idempotency_key="authority:{}".format(candidate_id),
            )
            return EvaluationResult(
                candidate_id,
                task_id,
                artifact_digest,
                Diagnostic.PROTOCOL_VIOLATION.value,
                "UNDER_25",
                "REJECTED",
                False,
                (authority_receipt["receipt_id"],),
                digest_bytes(b""),
            )
        if not self.hidden_runner.locus_matches(task_id, declared_locus):
            authority_receipt = self._record_receipt(
                {
                    **common,
                    "receipt_type": "AUTHORITY",
                    "request_id": "request-authority-{}".format(candidate_id),
                    "decision": "DENY",
                    "diagnostic_enum": Diagnostic.MUTATION_LOCUS_VIOLATION.value,
                },
                idempotency_key="authority:{}".format(candidate_id),
            )
            return EvaluationResult(
                candidate_id,
                task_id,
                artifact_digest,
                Diagnostic.MUTATION_LOCUS_VIOLATION.value,
                "UNDER_25",
                "REJECTED",
                False,
                (authority_receipt["receipt_id"],),
                digest_bytes(b""),
            )
        try:
            self.broker.authorize(
                action="execute_candidate",
                candidate_id=candidate_id,
                requested_authority=requested_authority,
                artifact_digest=artifact_digest,
                declared_locus=declared_locus,
            )
            authority_decision = "ALLOW"
            infrastructure_loss = False
        except AuthorityDenied as exc:
            authority_reason = str(exc).split(":", 1)[0]
            authority_decision = "DENY"
            infrastructure_loss = "AUTHORITY_RUNTIME_UNAVAILABLE" in authority_reason
        infrastructure_incident_id: Optional[str] = None
        failure_family_root: Optional[str] = None
        if infrastructure_loss:
            infrastructure_incident_id = self._infrastructure_incident(candidate_id, "authority-unavailable")
            failure_family_root = self._failure_root(task_id, Diagnostic.INTERNAL_ERROR.value, infrastructure_incident_id)
        authority_fields = {
            **common,
            "receipt_type": "AUTHORITY",
            "request_id": "request-authority-{}".format(candidate_id),
            "decision": authority_decision,
        }
        if infrastructure_incident_id is not None:
            authority_fields.update(
                {
                    "diagnostic_enum": Diagnostic.INTERNAL_ERROR.value,
                    "infrastructure_incident_id": infrastructure_incident_id,
                    "failure_family_root": failure_family_root,
                }
            )
        authority_receipt = self._record_receipt(
            authority_fields,
            idempotency_key="authority:{}".format(candidate_id),
        )
        if authority_decision != "ALLOW":
            return EvaluationResult(
                candidate_id,
                task_id,
                artifact_digest,
                Diagnostic.INTERNAL_ERROR.value if infrastructure_loss else Diagnostic.AUTHORITY_DENIED.value,
                "UNDER_25",
                "ABSTAINED",
                infrastructure_loss,
                (authority_receipt["receipt_id"],),
                digest_bytes(b""),
                infrastructure_incident_id,
                failure_family_root,
            )

        sandbox_kwargs: Dict[str, Any] = {
            "artifact_digest": artifact_digest,
            "candidate_id": candidate_id,
            "source_path": candidate_source_path,
        }
        resource_limit = self.hidden_runner.resource_limit(task_id)
        if resource_limit is not None:
            sandbox_kwargs["memory_limit"] = resource_limit
        # Docker frames, stdout, return codes, and low-level probe results are
        # untrusted evidence, not an authenticated result. The controller
        # admits only the closed pure-return contract as a production
        # precondition before candidate bytes execute. The evaluator-private
        # hidden oracle is the decision authority; low-level adversaries can
        # never turn their evidence into a production PASS.
        contract_reason = validate_candidate_source_contract(source)
        if contract_reason is not None:
            sandbox_result = SandboxResult(
                digest_for({"candidate_id": candidate_id, "artifact_digest": artifact_digest, "input": opaque_input}),
                artifact_digest,
                Diagnostic.PROTOCOL_VIOLATION.value,
                "UNDER_25",
                b"",
                "PROTOCOL_VIOLATION",
                0,
                {
                    "backend": getattr(self.sandbox, "backend_name", "unknown"),
                    "candidate_contract": "pure-return-v1",
                    "contract_status": "REJECTED",
                    "contract_reason": contract_reason,
                },
            )
        else:
            sandbox_kwargs["candidate_contract"] = "pure-return-v1"
            sandbox_result = self.sandbox.execute(source, opaque_input, **sandbox_kwargs)
        hidden_verdict = self.hidden_runner.evaluate(task_id, sandbox_result.output_bytes, opaque_input=opaque_input)
        candidate_execution_eligible = (
            contract_reason is None
            and sandbox_result.diagnostic_enum == Diagnostic.PASS.value
            and sandbox_result.exit_status == "SUCCESS"
        )
        diagnostic = hidden_verdict.diagnostic_enum if candidate_execution_eligible else sandbox_result.diagnostic_enum
        infrastructure_loss = hidden_verdict.infrastructure_loss or diagnostic == Diagnostic.INTERNAL_ERROR.value
        if infrastructure_loss:
            infrastructure_incident_id = sandbox_result.incident_id or self._infrastructure_incident(candidate_id, "sandbox-infrastructure-loss")
            failure_family_root = self._failure_root(task_id, Diagnostic.INTERNAL_ERROR.value, infrastructure_incident_id)
        if infrastructure_loss:
            verdict_decision = "ERROR"
        elif diagnostic == Diagnostic.PASS.value:
            verdict_decision = "PASS"
        else:
            verdict_decision = "FAIL"
        receipt_fields = {
            **common,
            "receipt_type": "VERDICT",
            "request_id": "request-verdict-{}".format(candidate_id),
            "decision": verdict_decision,
            "diagnostic_enum": diagnostic,
            "resource_bucket": sandbox_result.resource_bucket,
            "exit_status_class": sandbox_result.exit_status_class,
            "input_digest": digest_for(opaque_input),
            "output_digest": hidden_verdict.output_digest,
        }
        if infrastructure_incident_id is not None:
            receipt_fields["infrastructure_incident_id"] = infrastructure_incident_id
        if failure_family_root is not None:
            receipt_fields["failure_family_root"] = failure_family_root
        verdict_receipt = self._record_receipt(
            receipt_fields,
            idempotency_key="verdict:{}".format(candidate_id),
        )
        effect_decision = "ERROR" if infrastructure_loss else "ALLOW"
        effect_fields = {
            **common,
            "receipt_type": "EFFECT",
            "request_id": "request-effect-{}".format(candidate_id),
            "decision": effect_decision,
            # An EFFECT carrying an infrastructure incident is itself part of
            # the INTERNAL_ERROR failure chain.  Keep the public diagnostic
            # tuple complete on every receipt, including the effect leg.
            "diagnostic_enum": Diagnostic.INTERNAL_ERROR.value if infrastructure_loss else diagnostic,
            "normalized_action_hash": digest_for({"action": "execute_candidate", "locus": declared_locus}),
            "sandbox_id": sandbox_result.sandbox_id,
            "started_at": self.clock(),
            "finished_at": self.clock(),
            "exit_status_class": sandbox_result.exit_status_class,
            "output_digest": sandbox_result.output_digest,
            "environment_diff_digest": sandbox_result.environment_diff_digest,
        }
        if infrastructure_incident_id is not None:
            effect_fields["infrastructure_incident_id"] = infrastructure_incident_id
        if failure_family_root is not None:
            effect_fields["failure_family_root"] = failure_family_root
        effect_receipt = self._record_receipt(
            effect_fields,
            idempotency_key="effect:{}".format(candidate_id),
        )
        # A non-read authority action is promotable only after its signed
        # EFFECT receipt has been durably recorded and explicitly allowed.
        if infrastructure_loss:
            disposition = "ABSTAINED"
        elif diagnostic == Diagnostic.PASS.value and requested_authority != "READ_ONLY" and effect_decision == "ALLOW":
            disposition = "PROMOTED"
        elif diagnostic == Diagnostic.PASS.value and requested_authority == "READ_ONLY":
            disposition = "PROMOTED" if effect_decision == "ALLOW" else "ABSTAINED"
        else:
            disposition = "REJECTED"
        return EvaluationResult(
            candidate_id,
            task_id,
            artifact_digest,
            diagnostic,
            sandbox_result.resource_bucket,
            disposition,
            infrastructure_loss,
            (authority_receipt["receipt_id"], verdict_receipt["receipt_id"], effect_receipt["receipt_id"]),
            hidden_verdict.output_digest,
            infrastructure_incident_id,
            failure_family_root,
        )


EvaluationController = EvaluatorController


__all__ = [
    "EvaluatorController",
    "EvaluationController",
    "EvaluationResult",
    "HiddenEvaluatorRunner",
    "HiddenVerdict",
    "reconcile_journal",
]
