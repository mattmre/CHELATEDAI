"""Test-only deterministic evaluator used by the bounded Variation smoke.

This module is intentionally not an enforcement backend.  Production Variation
requires :class:`ControllerEvaluationGateway` over the Docker evaluator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..canonical import GENESIS_HASH, digest_bytes, digest_for, failure_family_root
from ..evaluation.controller import EvaluationResult, HiddenEvaluatorRunner
from ..evaluation.dataset import EvaluationCorpus
from ..evaluation.diagnostics import Diagnostic
from ..evaluation.sandbox import LocalTestSandbox
from ..receipts import ReceiptJournal, ReceiptSigner, receipt_hash
from .errors import VariationConfigurationError
from .loop import VARIATION_PROTOCOL_DIGEST


class FixtureEvaluationGateway:
    """CPU-only fixture adapter; never accepted by production Variation."""

    enforceable = False
    test_only = True
    evaluator_revision = "egv-variation-fixture-evaluator-v1"

    def __init__(
        self,
        corpus: EvaluationCorpus,
        ledger: Any,
        root: Path,
        *,
        policy_digest: str,
        campaign_id: str = "variation-fixture-campaign",
    ) -> None:
        self.corpus = corpus
        self.ledger = ledger
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.sandbox = LocalTestSandbox(self.root / "local-test-sandbox")
        self.hidden_runner = HiddenEvaluatorRunner.from_corpus(corpus, evaluator_revision=self.evaluator_revision)
        self.signer = ReceiptSigner(b"\x07" * 32)
        self.journal = ReceiptJournal(self.root / "fixture-receipts.jsonl", self.signer.public_key)
        self.policy_digest = policy_digest
        self.campaign_id = campaign_id
        self.evaluator_digest = digest_for(self.evaluator_revision)

    def _common(self, *, task_id: str, candidate_id: str, artifact_digest: str) -> dict[str, Any]:
        record = self.hidden_runner.public_record(task_id)
        if record is None:
            raise VariationConfigurationError("fixture evaluator task is outside the frozen hidden corpus")
        return {
            "campaign_id": self.campaign_id,
            "run_id": "run-evaluation-fixture",
            "task_id": task_id,
            "candidate_id": candidate_id,
            "candidate_artifact_digest": artifact_digest,
            "protocol_digest": VARIATION_PROTOCOL_DIGEST,
            "policy_digest": self.policy_digest,
            "evaluator_digest": self.evaluator_digest,
            "task_family": record["family_id"],
            "normalized_public_locus": record["public_locus"],
            "public_rule_id": record["public_rule_id"],
        }

    def _append_receipt(self, fields: dict[str, Any], *, idempotency_key: str) -> dict[str, Any]:
        previous = self.journal.receipts()
        sequence = len(previous) + 1
        receipt = self.signer.sign_receipt(
            fields,
            sequence=sequence,
            previous_receipt_hash=receipt_hash(previous[-1]) if previous else GENESIS_HASH,
            idempotency_key=idempotency_key,
        )
        self.journal.append(receipt)
        self.ledger.ingest_receipt(receipt, self.signer.public_key)
        return receipt

    def evaluate(
        self,
        *,
        candidate_id: str,
        task_id: str,
        source: bytes,
        opaque_input: Any,
        requested_authority: str,
        declared_locus: str,
        candidate_source_path: Optional[str] = None,
    ) -> EvaluationResult:
        artifact_digest = digest_bytes(source)
        authority = self._append_receipt(
            {
                **self._common(task_id=task_id, candidate_id=candidate_id, artifact_digest=artifact_digest),
                "receipt_type": "AUTHORITY",
                "request_id": "fixture-authority-{}".format(candidate_id),
                "decision": "ALLOW",
            },
            idempotency_key="fixture-authority:{}".format(candidate_id),
        )
        if not self.hidden_runner.has_task(task_id) or not self.hidden_runner.locus_matches(task_id, declared_locus):
            diagnostic = Diagnostic.PROTOCOL_VIOLATION.value
            output = b""
            sandbox_status = "PROTOCOL_VIOLATION"
        else:
            sandbox_result = self.sandbox.execute(
                source,
                opaque_input,
                artifact_digest=artifact_digest,
                candidate_id=candidate_id,
                source_path=candidate_source_path,
            )
            verdict = self.hidden_runner.evaluate(task_id, sandbox_result.output_bytes, opaque_input=opaque_input)
            diagnostic = verdict.diagnostic_enum if sandbox_result.diagnostic_enum == Diagnostic.PASS.value else sandbox_result.diagnostic_enum
            output = sandbox_result.output_bytes
            sandbox_status = sandbox_result.exit_status_class
        incident = None
        root = None
        if diagnostic == Diagnostic.INTERNAL_ERROR.value:
            incident = digest_for({"candidate_id": candidate_id, "reason": "fixture-infrastructure"})
            record = self.hidden_runner.public_record(task_id)
            if record is not None:
                root = failure_family_root(
                    record["family_id"], diagnostic, record["public_locus"], record["public_rule_id"], infrastructure_incident_id=incident
                )
        common = self._common(task_id=task_id, candidate_id=candidate_id, artifact_digest=artifact_digest)
        verdict_fields = {
            **common,
            "receipt_type": "VERDICT",
            "request_id": "fixture-verdict-{}".format(candidate_id),
            "decision": "ERROR" if incident else ("PASS" if diagnostic == Diagnostic.PASS.value else "FAIL"),
            "diagnostic_enum": diagnostic,
            "resource_bucket": "UNDER_25",
            "exit_status_class": "SUCCESS" if diagnostic == Diagnostic.PASS.value else sandbox_status,
            "input_digest": digest_for(opaque_input),
            "output_digest": digest_bytes(output),
        }
        if incident is not None:
            verdict_fields.update({"infrastructure_incident_id": incident, "failure_family_root": root})
        verdict_receipt = self._append_receipt(verdict_fields, idempotency_key="fixture-verdict:{}".format(candidate_id))
        effect_fields = {
            **common,
            "receipt_type": "EFFECT",
            "request_id": "fixture-effect-{}".format(candidate_id),
            "decision": "ERROR" if incident else "ALLOW",
            "diagnostic_enum": Diagnostic.INTERNAL_ERROR.value if incident else diagnostic,
            "normalized_action_hash": digest_for({"action": "execute_candidate", "locus": declared_locus}),
            "sandbox_id": digest_for({"candidate_id": candidate_id, "artifact": artifact_digest}),
            "started_at": "2026-08-22T00:00:00Z",
            "finished_at": "2026-08-22T00:00:00Z",
            "exit_status_class": "SUCCESS" if diagnostic == Diagnostic.PASS.value else sandbox_status,
            "output_digest": digest_bytes(output),
            "environment_diff_digest": digest_for({"fixture": True, "candidate": artifact_digest}),
        }
        if incident is not None:
            effect_fields.update({"infrastructure_incident_id": incident, "failure_family_root": root})
        effect_receipt = self._append_receipt(effect_fields, idempotency_key="fixture-effect:{}".format(candidate_id))
        return EvaluationResult(
            candidate_id=candidate_id,
            task_id=task_id,
            candidate_artifact_digest=artifact_digest,
            diagnostic_enum=diagnostic,
            resource_bucket="UNDER_25",
            disposition="ABSTAINED" if incident else ("PROMOTED" if diagnostic == Diagnostic.PASS.value else "REJECTED"),
            infrastructure_loss=incident is not None,
            receipt_ids=(authority["receipt_id"], verdict_receipt["receipt_id"], effect_receipt["receipt_id"]),
            output_digest=digest_bytes(output),
            infrastructure_incident_id=incident,
            failure_family_root=root,
        )


__all__ = ["FixtureEvaluationGateway"]
