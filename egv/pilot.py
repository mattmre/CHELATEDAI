"""Bounded CPU-only trainer/evaluator process smoke for the Slice 2 core.

This is an integration harness, not a campaign runner.  The trainer process
owns the SQLite writer and local IPC service.  The evaluator process owns only
its private receipt journal and communicates with the trainer over the
authenticated socket.  No model or external service is started.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
import queue
import sqlite3
import tempfile
from typing import Any, Dict

from .canonical import digest_for
from .errors import EGVError
from .ipc import LedgerClient, LedgerWriterService
from .ledger import EvidenceLedger
from .receipts import ReceiptJournal, ReceiptSigner, receipt_hash


_CLOCK_VALUE = "2026-08-21T00:00:00Z"


def _put_error(channel: Any, exc: BaseException) -> None:
    channel.put({"error": type(exc).__name__, "message": str(exc)})


def _trainer_process(
    ledger_path: str,
    blob_root: str,
    socket_path: str,
    auth_token: str,
    evaluator_auth_token: str,
    ready: Any,
    done: Any,
    errors: Any,
) -> None:
    ledger = EvidenceLedger(ledger_path, blob_root=blob_root, clock=lambda: _CLOCK_VALUE)
    try:
        campaign_id = "campaign-dual-smoke"
        run_id = "run-dual-smoke"
        task_id = "task-dual-smoke"
        ledger.create_campaign(
            campaign_id,
            protocol_hash=digest_for("egv-dual-smoke-protocol"),
            source_commit="dual-smoke-commit",
            model_revision="cpu-fixture-no-model",
            data_manifest_hash=digest_for("egv-dual-smoke-data"),
            evaluator_hash=digest_for("egv-dual-smoke-evaluator"),
            policy_hash=digest_for("egv-dual-smoke-policy"),
            seed_set=[11],
            created_at=_CLOCK_VALUE,
        )
        ledger.create_run(
            run_id,
            campaign_id=campaign_id,
            arm="D",
            task_id=task_id,
            seed=11,
            parent_checkpoint=None,
            start_state="READY",
            host_role="trainer-fixture",
            software_manifest_hash=digest_for("egv-dual-smoke-software"),
            created_at=_CLOCK_VALUE,
        )
        ledger.append_candidate(
            "candidate-dual-smoke",
            campaign_id=campaign_id,
            run_id=run_id,
            task_id=task_id,
            parent_candidate_id=None,
            mutation_family="PURE_FUNCTION",
            patch_hash=digest_for("egv-dual-smoke-patch"),
            requested_authority="EXECUTE_CANDIDATE",
            prompt_hash=digest_for("egv-dual-smoke-prompt"),
            model_hash=digest_for("egv-dual-smoke-model"),
            adapter_hash=digest_for("egv-dual-smoke-adapter"),
        )
        with LedgerWriterService(
            ledger,
            socket_path,
            auth_token,
            evaluator_auth_token=evaluator_auth_token,
        ):
            ready.set()
            if not done.wait(30):
                raise EGVError("trainer fixture timed out waiting for evaluator")
            receipt_rows = ledger.connection.execute(
                "SELECT payload_json FROM receipts WHERE receipt_type IN ('VERDICT','EFFECT') ORDER BY sequence"
            ).fetchall()
            receipts_by_type = {json.loads(row[0])["receipt_type"]: json.loads(row[0]) for row in receipt_rows}
            verdict = receipts_by_type.get("VERDICT")
            effect = receipts_by_type.get("EFFECT")
            if verdict is None or effect is None:
                raise EGVError("trainer did not observe both verdict and effect receipts")
            ledger.append_verdict(
                "verdict-dual-smoke",
                candidate_id="candidate-dual-smoke",
                correctness=True,
                performance={"score": 1},
                hidden_test_set_hash=digest_for("egv-dual-smoke-hidden-tests"),
                evaluator_revision="dual-smoke-evaluator-v1",
                receipt_id=verdict["receipt_id"],
                signed_receipt_hash=receipt_hash(verdict),
            )
            ledger.append_effect_receipt(
                request_id=effect["request_id"],
                candidate_id="candidate-dual-smoke",
                identity="evaluator-fixture",
                normalized_action_hash=effect["normalized_action_hash"],
                decision=effect["decision"],
                policy_hash=effect["policy_digest"],
                sandbox_id=effect["sandbox_id"],
                started_at=effect["started_at"],
                finished_at=effect["finished_at"],
                exit_status_class=effect["exit_status_class"],
                output_hash=effect.get("output_digest"),
                environment_diff_hash=effect.get("environment_diff_digest"),
                signature=effect["signature"],
                receipt_id=effect["receipt_id"],
            )
    except BaseException as exc:  # process boundary: report and let the parent fail closed
        _put_error(errors, exc)
        ready.set()
    finally:
        ledger.close()


def _evaluator_process(
    socket_path: str,
    evaluator_auth_token: str,
    journal_path: str,
    ready: Any,
    result: Any,
    errors: Any,
) -> None:
    try:
        sqlite_connect_attempts: list[str] = []

        def denied_sqlite_connect(*arguments: Any, **_keywords: Any) -> Any:
            sqlite_connect_attempts.append(str(arguments[0]) if arguments else "")
            raise AssertionError("evaluator process attempted SQLite access")

        sqlite3.connect = denied_sqlite_connect  # type: ignore[assignment]
        if not ready.wait(30):
            raise EGVError("evaluator fixture timed out waiting for trainer IPC")
        signer = ReceiptSigner(b"\x06" * 32)
        campaign_id = "campaign-dual-smoke"
        run_id = "run-dual-smoke"
        task_id = "task-dual-smoke"
        candidate_id = "candidate-dual-smoke"
        protocol_digest = digest_for("egv-dual-smoke-protocol")
        policy_digest = digest_for("egv-dual-smoke-policy")
        evaluator_digest = digest_for("egv-dual-smoke-evaluator")
        common = {
            "campaign_id": campaign_id,
            "run_id": run_id,
            "task_id": task_id,
            "candidate_id": candidate_id,
            "candidate_artifact_digest": digest_for("egv-dual-smoke-artifact"),
            "protocol_digest": protocol_digest,
            "policy_digest": policy_digest,
            "evaluator_digest": evaluator_digest,
        }
        authority = signer.sign_receipt(
            {**common, "receipt_type": "AUTHORITY", "request_id": "dual-authority", "decision": "ALLOW"},
            sequence=1,
            idempotency_key="dual-authority",
        )
        verdict = signer.sign_receipt(
            {
                **common,
                "receipt_type": "VERDICT",
                "request_id": "dual-verdict",
                "decision": "PASS",
                "diagnostic_enum": "PASS",
                "resource_bucket": "UNDER_25",
                "exit_status_class": "SUCCESS",
                "output_digest": digest_for("egv-dual-smoke-output"),
            },
            sequence=2,
            previous_receipt_hash=receipt_hash(authority),
            idempotency_key="dual-verdict",
        )
        effect = signer.sign_receipt(
            {
                **common,
                "receipt_type": "EFFECT",
                "request_id": "dual-effect",
                "decision": "ALLOW",
                "normalized_action_hash": digest_for("egv-dual-smoke-action"),
                "sandbox_id": "dual-sandbox",
                "started_at": _CLOCK_VALUE,
                "finished_at": _CLOCK_VALUE,
                "exit_status_class": "SUCCESS",
                "output_digest": digest_for("egv-dual-smoke-effect-output"),
                "environment_diff_digest": digest_for("egv-dual-smoke-environment"),
            },
            sequence=3,
            previous_receipt_hash=receipt_hash(verdict),
            idempotency_key="dual-effect",
        )
        journal = ReceiptJournal(journal_path, signer.public_key)
        client = LedgerClient(socket_path, evaluator_auth_token, role="evaluator")
        public_key = signer.public_key_pem.decode("ascii")
        for receipt in (authority, verdict, effect):
            journal.append(receipt)
            client.ingest_receipt(receipt=receipt, public_key=public_key)
        non_receipt_ipc_rejected = False
        try:
            client.append_event(event_type="EVALUATOR_FORBIDDEN", payload={"value": 1}, subject_id="forbidden")
        except Exception as exc:
            non_receipt_ipc_rejected = isinstance(exc, Exception) and "restricted to ingest_receipt" in str(exc)
        if not non_receipt_ipc_rejected:
            raise EGVError("evaluator IPC accepted a non-receipt write")
        if sqlite_connect_attempts:
            raise EGVError("evaluator process attempted SQLite access")
        result.put(
            {
                "journal": journal.verify(),
                "evaluator_ipc_methods": ["ingest_receipt"],
                "evaluator_used_sqlite": bool(sqlite_connect_attempts),
                "evaluator_sqlite_connect_calls": len(sqlite_connect_attempts),
                "evaluator_non_receipt_ipc_rejected": non_receipt_ipc_rejected,
            }
        )
    except BaseException as exc:  # process boundary: report and let the parent fail closed
        _put_error(errors, exc)


def _drain_errors(errors: Any) -> list[Dict[str, str]]:
    found: list[Dict[str, str]] = []
    while True:
        try:
            value = errors.get_nowait()
        except queue.Empty:
            return found
        found.append(dict(value))


def run_two_process_smoke() -> Dict[str, Any]:
    """Run the CPU-only trainer/evaluator IPC fixture in separate processes."""

    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="egv-dual-smoke-") as directory:
        root = Path(directory)
        ledger_path = root / "ledger.sqlite"
        socket_path = root / "ledger-writer.sock"
        journal_path = root / "evaluator-receipts.jsonl"
        auth_token = "egv-dual-smoke-auth-token"
        evaluator_auth_token = "egv-dual-smoke-evaluator-token"
        ready = context.Event()
        done = context.Event()
        results = context.Queue()
        errors = context.Queue()
        trainer = context.Process(
            target=_trainer_process,
            args=(str(ledger_path), str(root / "private"), str(socket_path), auth_token, evaluator_auth_token, ready, done, errors),
            name="egv-trainer-fixture",
        )
        evaluator = context.Process(
            target=_evaluator_process,
            args=(str(socket_path), evaluator_auth_token, str(journal_path), ready, results, errors),
            name="egv-evaluator-fixture",
        )
        trainer.start()
        if not ready.wait(30):
            trainer.terminate()
            trainer.join(5)
            raise EGVError("trainer fixture did not expose its writer IPC endpoint")
        evaluator.start()
        evaluator.join(30)
        done.set()
        trainer.join(30)
        if evaluator.is_alive():
            evaluator.terminate()
            evaluator.join(5)
        if trainer.is_alive():
            trainer.terminate()
            trainer.join(5)
        process_errors = _drain_errors(errors)
        if process_errors or trainer.exitcode != 0 or evaluator.exitcode != 0:
            raise EGVError(
                f"two-process smoke failed: trainer_exit={trainer.exitcode}, "
                f"evaluator_exit={evaluator.exitcode}, errors={process_errors}"
            )
        try:
            evaluator_result = dict(results.get(timeout=2))
        except queue.Empty as exc:
            raise EGVError("evaluator fixture returned no result") from exc
        with EvidenceLedger(ledger_path, mode="read_only", blob_root=root / "private") as ledger:
            integrity = ledger.verify_integrity()
            disposition = ledger.candidate_disposition("candidate-dual-smoke")
            status = ledger.status()
        # Omit the temporary filesystem path so the CLI smoke is byte-for-byte
        # reproducible across runs while retaining all operational counters.
        status.pop("path", None)
        return {
            "mode": "two-process-cpu-only",
            "trainer_process_exit": trainer.exitcode,
            "evaluator_process_exit": evaluator.exitcode,
            "evaluator_used_sqlite": evaluator_result["evaluator_used_sqlite"],
            "evaluator_ipc_methods": evaluator_result["evaluator_ipc_methods"],
            "evaluator_sqlite_connect_calls": evaluator_result["evaluator_sqlite_connect_calls"],
            "evaluator_non_receipt_ipc_rejected": evaluator_result["evaluator_non_receipt_ipc_rejected"],
            "journal": evaluator_result["journal"],
            "ledger": integrity,
            "status": status,
            "candidate_disposition": disposition,
        }


__all__ = ["run_two_process_smoke"]
