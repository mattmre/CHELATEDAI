"""Spawned trainer/evaluator smoke for the CPU-only Evaluation slice."""

from __future__ import annotations

import json
import multiprocessing as mp
import secrets
import sqlite3
from pathlib import Path
import queue
import tempfile
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from ..canonical import digest_bytes, digest_for
from ..errors import EGVError
from ..ipc import LedgerClient, LedgerWriterService, new_auth_token
from ..receipts import ReceiptJournal, ReceiptSigner, receipt_hash, verify_receipt
from .artifacts import ContentAddressedArtifactStore
from .authority import AuthorityBroker, AuthorityPolicy, DockerEnforcedRuntime
from .boundary import install_evaluator_sqlite_jail, mark_evaluator_process
from .controller import EvaluatorController, HiddenEvaluatorRunner
from .dataset import EVALUATOR_SEED_BYTES, FAMILY_SPECS, EvaluationCorpus
from .sandbox import DockerCandidateSandbox


_CAMPAIGN_ID = "campaign-evaluation-process"
_RUN_ID = "run-evaluation"
_CLOCK = "2026-08-22T00:00:00Z"
_FAMILY_TASKS = tuple(
    (spec.family_id, "egv-{}-heldout-1-v1".format(spec.family_id.lower())) for spec in FAMILY_SPECS
)
_HELDOUT_TASKS = tuple(
    (spec.family_id, "egv-{}-heldout-{}-v1".format(spec.family_id.lower(), ordinal))
    for spec in FAMILY_SPECS
    for ordinal in range(1, spec.heldout + 1)
)
_TASK_IDS = tuple(task_id for _, task_id in _FAMILY_TASKS)
_WRONG_LOCUS_TASK = dict(_FAMILY_TASKS)["PURE_FUNCTION"]
_WRONG_CANDIDATE_ID = "candidate-evaluation-wrong-locus"


def _candidate_id(family_id: str, task_id: str = "") -> str:
    if not task_id or task_id.endswith("-heldout-1-v1"):
        return "candidate-evaluation-{}".format(family_id.lower())
    ordinal = task_id.split("-heldout-", 1)[1].rsplit("-v1", 1)[0]
    return "candidate-evaluation-{}-heldout-{}".format(family_id.lower(), ordinal)


def _put_error(channel: Any, exc: BaseException) -> None:
    channel.put({"error": type(exc).__name__, "message": str(exc)})


def _prepare_evaluator_corpus(
    root: Path,
    *,
    corpus: Optional[EvaluationCorpus],
    evaluator_seed_file: Optional[Union[Path, str]],
    expected_manifest_digest: Optional[str],
) -> Tuple[EvaluationCorpus, Path, str]:
    """Stage the evaluator seed privately and bind it to one corpus digest.

    The seed is copied into the temporary evaluator-private tree solely so the
    spawned evaluator can regenerate the exact corpus.  It is never passed as
    a process argument, returned in evidence, or made available to the
    trainer.  A caller-supplied corpus is checked against the staged seed
    before any child process starts.
    """

    if evaluator_seed_file is not None:
        try:
            seed_bytes = Path(evaluator_seed_file).read_bytes()
        except OSError as exc:
            raise EGVError("frozen evaluator seed is unavailable") from exc
    elif corpus is not None:
        corpus.validate()
        seed_bytes = corpus.private_seed_bytes()
    else:
        seed_bytes = secrets.token_bytes(EVALUATOR_SEED_BYTES)
    if len(seed_bytes) != EVALUATOR_SEED_BYTES:
        raise EGVError("evaluator seed must contain exactly 32 private bytes")

    seed_path = root / "evaluator-private" / "corpus-seed.bin"
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.write_bytes(seed_bytes)
    seed_path.chmod(0o600)
    regenerated = EvaluationCorpus.generate(secret_seed_file=seed_path)
    regenerated_digest = regenerated.manifest_digest()
    if corpus is not None and regenerated_digest != corpus.manifest_digest():
        raise EGVError("supplied evaluator seed does not match the frozen corpus")
    if expected_manifest_digest is not None and regenerated_digest != expected_manifest_digest:
        raise EGVError("evaluated corpus digest does not match the frozen manifest digest")
    return regenerated, seed_path, regenerated_digest


def _validate_frozen_candidate_artifacts(
    frozen_root: Union[Path, str],
    corpus: EvaluationCorpus,
) -> Dict[str, str]:
    """Validate private held-out candidate bytes without returning the bytes."""

    root = Path(frozen_root)
    manifest_path = root / "evaluator-private" / "data-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EGVError("frozen evaluator data manifest is unavailable") from exc
    if digest_for(manifest) != corpus.manifest_digest():
        raise EGVError("frozen evaluator data manifest does not match the evaluated corpus")

    digests: Dict[str, str] = {}
    for _family_id, task_id in _HELDOUT_TASKS:
        task_root = root / "evaluator-private" / "heldout" / task_id
        source_path = task_root / "candidate" / "src" / "task.py"
        candidate_manifest_path = task_root / "candidate" / "candidate-manifest.json"
        try:
            source = source_path.read_bytes()
            candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
            declared_digest = candidate_manifest["source_digest"]
        except (KeyError, OSError, ValueError) as exc:
            raise EGVError("frozen held-out candidate artifact is incomplete") from exc
        source_digest = digest_bytes(source)
        if source_digest != declared_digest:
            raise EGVError("frozen held-out candidate artifact digest is invalid")
        if source != corpus.correct_candidate_source(task_id):
            raise EGVError("frozen held-out candidate bytes do not match the evaluated corpus")
        digests[task_id] = source_digest
    return digests


def _trainer_process(
    ledger_path: str,
    blob_root: str,
    socket_path: str,
    trainer_token: str,
    evaluator_token: str,
    protocol_digest: str,
    policy_digest: str,
    ready: Any,
    evaluator_done: Any,
    output: Any,
    errors: Any,
    evaluator_public_key: Any,
) -> None:
    from ..ledger import EvidenceLedger

    ledger = EvidenceLedger(ledger_path, blob_root=blob_root, clock=lambda: _CLOCK)
    try:
        ledger.create_campaign(
            _CAMPAIGN_ID,
            protocol_hash=protocol_digest,
            source_commit="evaluation-process-smoke",
            model_revision="model-not-loaded",
            data_manifest_hash=digest_for("evaluation-process-data"),
            evaluator_hash=digest_for("egv-evaluator-v1"),
            policy_hash=policy_digest,
            seed_set=[11],
            created_at=_CLOCK,
        )
        ledger.create_run(
            _RUN_ID,
            campaign_id=_CAMPAIGN_ID,
            arm="D",
            task_id=_TASK_IDS[0],
            seed=11,
            parent_checkpoint=None,
            start_state="READY",
            host_role="trainer-fixture",
            software_manifest_hash=digest_for("evaluation-process-software"),
            created_at=_CLOCK,
        )
        for family_id, task_id in _HELDOUT_TASKS:
            ledger.append_candidate(
                _candidate_id(family_id, task_id),
                campaign_id=_CAMPAIGN_ID,
                run_id=_RUN_ID,
                task_id=task_id,
                parent_candidate_id=None,
                mutation_family=family_id,
                patch_hash=digest_for("candidate-source-private-to-evaluator-" + family_id),
                requested_authority="EXECUTE_CANDIDATE",
                prompt_hash=digest_for("evaluation-prompts"),
                model_hash=digest_for("model-not-loaded"),
                adapter_hash=digest_for("adapter-not-loaded"),
            )
        ledger.append_candidate(
            _WRONG_CANDIDATE_ID,
            campaign_id=_CAMPAIGN_ID,
            run_id=_RUN_ID,
            task_id=_TASK_IDS[0],
            parent_candidate_id=None,
            mutation_family="PURE_FUNCTION",
            patch_hash=digest_for("candidate-source-wrong-locus"),
            requested_authority="EXECUTE_CANDIDATE",
            prompt_hash=digest_for("evaluation-prompts"),
            model_hash=digest_for("model-not-loaded"),
            adapter_hash=digest_for("adapter-not-loaded"),
        )
        with LedgerWriterService(
            ledger,
            socket_path,
            trainer_token,
            evaluator_auth_token=evaluator_token,
        ):
            ready.set()
            if not evaluator_done.wait(60):
                raise EGVError("trainer process timed out waiting for evaluator")
            rows = ledger.connection.execute(
                "SELECT payload_json FROM receipts WHERE receipt_type IN ('VERDICT','EFFECT') ORDER BY sequence"
            ).fetchall()
            by_candidate: Dict[str, Dict[str, Mapping[str, Any]]] = {}
            for row in rows:
                receipt = json.loads(row[0])
                by_candidate.setdefault(receipt["candidate_id"], {})[receipt["receipt_type"]] = receipt
            public_key = evaluator_public_key.get(timeout=5)
            if not isinstance(public_key, str) or not public_key:
                raise EGVError("evaluator did not expose a public verification key")
            heldout_dispositions: Dict[str, str] = {}
            signed_verdict_correctness: Dict[str, bool] = {}
            for family_id, task_id in _HELDOUT_TASKS:
                candidate_id = _candidate_id(family_id, task_id)
                receipts = by_candidate.get(candidate_id, {})
                verdict = receipts.get("VERDICT")
                effect = receipts.get("EFFECT")
                if verdict is None or effect is None:
                    raise EGVError("trainer process did not observe verdict/effect for " + candidate_id)
                # Correctness is a consequence of the verified signed verdict,
                # never a trainer assertion or a sandbox return code.
                verify_receipt(verdict, public_key)
                correctness = verdict.get("decision") == "PASS" and verdict.get("diagnostic_enum") == "PASS"
                signed_verdict_correctness[task_id] = correctness
                ledger.append_verdict(
                    "verdict-{}".format(candidate_id),
                    candidate_id=candidate_id,
                    correctness=correctness,
                    performance={},
                    hidden_test_set_hash=digest_for("evaluation-hidden-set"),
                    evaluator_revision="egv-evaluator-v1",
                    receipt_id=verdict["receipt_id"],
                    signed_receipt_hash=receipt_hash(verdict),
                )
                ledger.append_effect_receipt(
                    request_id=effect["request_id"],
                    candidate_id=candidate_id,
                    identity="evaluator-private",
                    normalized_action_hash=effect["normalized_action_hash"],
                    decision=effect["decision"],
                    policy_hash=effect["policy_digest"],
                    sandbox_id=effect["sandbox_id"],
                    started_at=effect["started_at"],
                    finished_at=effect["finished_at"],
                    exit_status_class=effect["exit_status_class"],
                    output_hash=effect["output_digest"],
                    environment_diff_hash=effect["environment_diff_digest"],
                    signature=effect["signature"],
                    receipt_id=effect["receipt_id"],
                )
                heldout_dispositions[task_id] = ledger.candidate_disposition(candidate_id)
            family_dispositions = {
                family_id: heldout_dispositions[task_id] for family_id, task_id in _FAMILY_TASKS
            }
            integrity = ledger.verify_integrity()
            output.put(
                {
                    "candidate_disposition": family_dispositions[_FAMILY_TASKS[0][0]],
                    "family_dispositions": family_dispositions,
                    "heldout_dispositions": heldout_dispositions,
                    "signed_verdict_correctness": signed_verdict_correctness,
                    "heldout_task_count": len(heldout_dispositions),
                    "wrong_locus_disposition": ledger.candidate_disposition(_WRONG_CANDIDATE_ID),
                    "integrity": integrity,
                    "receipt_count": integrity["receipt_count"],
                }
            )
    except BaseException as exc:
        _put_error(errors, exc)
    finally:
        ledger.close()


def _sqlite_negative_control_process(probe_path: str, output: Any) -> None:
    """Prove the real SQLite denial outside the receipt-only evaluator."""

    mark_evaluator_process()
    connect_counter = [0]
    evidence = install_evaluator_sqlite_jail(connect_counter)
    evidence.update(
        {
            "process": "sqlite-negative-control",
            "attempted": True,
            "denied": False,
            "exception": None,
            "connect_calls": 0,
            "probe_path_staged": Path(probe_path).exists(),
        }
    )
    try:
        connection = sqlite3.connect(probe_path)
    except PermissionError as exc:
        evidence.update({"denied": True, "exception": type(exc).__name__})
    except BaseException as exc:
        evidence.update({"exception": type(exc).__name__})
    else:
        connection.close()
    evidence["connect_calls"] = connect_counter[0]
    evidence["probe_path_created"] = Path(probe_path).exists()
    output.put(evidence)


def _evaluator_process(
    socket_path: str,
    evaluator_token: str,
    private_root: str,
    evaluator_seed_path: str,
    frozen_root: Optional[str],
    expected_corpus_digest: str,
    protocol_digest: str,
    policy_digest: str,
    trainer_ready: Any,
    evaluator_done: Any,
    output: Any,
    errors: Any,
    sqlite_evidence: Mapping[str, Any],
    evaluator_public_key: Any,
) -> None:
    # Startup order is part of the production boundary.  This process never
    # opens SQLite and can only deliver signed receipts over the narrow IPC.
    mark_evaluator_process()
    evaluator_connect_counter = [0]
    evaluator_sqlite_hook = install_evaluator_sqlite_jail(evaluator_connect_counter)
    try:
        if not trainer_ready.wait(60):
            raise EGVError("evaluator process timed out waiting for trainer IPC")
        private = Path(private_root)
        private.mkdir(parents=True, exist_ok=True)
        # Use SQLite's private in-memory target for the co-hosting negative
        # control.  This exercises the real EvidenceLedger constructor after
        # the immutable audit hook without creating a misleading lock-file
        # artifact in the evaluator tree.
        evaluator_ledger_probe = ":memory:"
        cohost_evidence: Dict[str, Any] = {
            "attempted": True,
            "denied": False,
            "exception": None,
            "probe_path_created": False,
        }
        # This is a production-path negative control: the evaluator attempts
        # to construct a writer after its startup hook and must be denied.
        from ..ledger import EvidenceLedger

        try:
            probe_ledger = EvidenceLedger(evaluator_ledger_probe)
        except PermissionError as exc:
            cohost_evidence.update({"denied": True, "exception": type(exc).__name__})
        else:
            probe_ledger.close()
        cohost_evidence["probe_path_created"] = False
        if not cohost_evidence["denied"] or cohost_evidence["probe_path_created"]:
            raise EGVError("evaluator SQLite writer co-hosting was not denied")
        corpus = EvaluationCorpus.generate(secret_seed_file=evaluator_seed_path)
        evaluated_corpus_digest = corpus.manifest_digest()
        if evaluated_corpus_digest != expected_corpus_digest:
            raise EGVError("evaluator corpus digest differs from the frozen corpus")
        frozen_source_digests = (
            _validate_frozen_candidate_artifacts(frozen_root, corpus) if frozen_root is not None else {}
        )
        signer = ReceiptSigner.generate()
        evaluator_public_key.put(signer.public_key_pem.decode("ascii"))
        journal = ReceiptJournal(private / "receipts.jsonl", signer.public_key)
        policy = AuthorityPolicy.candidate_execution()
        client = LedgerClient(socket_path, evaluator_token, role="evaluator")
        public_key = signer.public_key_pem.decode("ascii")
        runtime = DockerEnforcedRuntime()
        controller = EvaluatorController(
            sandbox=DockerCandidateSandbox(private / "sandbox"),
            hidden_runner=HiddenEvaluatorRunner.from_corpus(corpus),
            broker=AuthorityBroker(runtime, policy),
            signer=signer,
            journal=journal,
            ingest=lambda receipt: client.ingest_receipt(receipt=receipt, public_key=public_key),
            campaign_id=_CAMPAIGN_ID,
            protocol_digest=protocol_digest,
            policy_digest=policy_digest,
            artifact_store=ContentAddressedArtifactStore(private / "artifacts"),
            clock=lambda: _CLOCK,
        )
        results = []
        evaluated_source_bindings: Dict[str, Dict[str, Any]] = {}
        for family_id, task_id in _HELDOUT_TASKS:
            task = corpus.get(task_id)
            if frozen_root is not None:
                candidate_path = Path(frozen_root) / "evaluator-private" / "heldout" / task_id / "candidate" / "src" / "task.py"
                source = candidate_path.read_bytes()
                frozen_source_digest = frozen_source_digests[task_id]
                source_bytes_match_frozen = source == corpus.correct_candidate_source(task_id)
            else:
                source = corpus.correct_candidate_source(task.template_id)
                candidate_path = private / "candidates" / family_id / "src" / "task.py"
                candidate_path.parent.mkdir(parents=True, exist_ok=True)
                candidate_path.write_bytes(source)
                frozen_source_digest = None
                source_bytes_match_frozen = None
            result = controller.evaluate(
                candidate_id=_candidate_id(family_id, task_id),
                task_id=task.template_id,
                source=source,
                opaque_input=task.evaluator_input,
                declared_locus=task.public_locus,
                candidate_source_path=str(candidate_path),
            )
            evaluated_source_digest = digest_bytes(source)
            if result.candidate_artifact_digest != evaluated_source_digest:
                raise EGVError("Docker candidate artifact digest differs from evaluated source bytes")
            if frozen_source_digest is not None and evaluated_source_digest != frozen_source_digest:
                raise EGVError("Docker candidate artifact differs from frozen held-out source")
            evaluated_source_bindings[task_id] = {
                "evaluated_source_digest": result.candidate_artifact_digest,
                "frozen_candidate_source_digest": frozen_source_digest,
                "source_bytes_match_frozen": source_bytes_match_frozen,
            }
            results.append(result)
        wrong_task = corpus.get(_WRONG_LOCUS_TASK)
        if frozen_root is not None:
            wrong_source_path = Path(frozen_root) / "evaluator-private" / "heldout" / _WRONG_LOCUS_TASK / "candidate" / "src" / "task.py"
            wrong_source = wrong_source_path.read_bytes()
        else:
            wrong_source_path = private / "candidates" / "wrong-locus" / "src" / "task.py"
            wrong_source = corpus.correct_candidate_source(wrong_task.template_id)
            wrong_source_path.parent.mkdir(parents=True, exist_ok=True)
            wrong_source_path.write_bytes(wrong_source)
        wrong_result = controller.evaluate(
            candidate_id=_WRONG_CANDIDATE_ID,
            task_id=wrong_task.template_id,
            source=wrong_source,
            opaque_input=wrong_task.evaluator_input,
            declared_locus=wrong_task.public_locus + ":wrong",
            candidate_source_path=str(wrong_source_path),
        )
        negative_controls = controller.sandbox.negative_controls(private / "hidden.json")
        non_receipt_rejected = False
        try:
            client.append_event(event_type="EVALUATOR_FORBIDDEN", payload={"value": 1}, subject_id="forbidden")
        except Exception as exc:
            non_receipt_rejected = "restricted to ingest_receipt" in str(exc)
        if not non_receipt_rejected:
            raise EGVError("evaluator IPC accepted a non-receipt method")
        mountinfo = Path("/proc/self/mountinfo").read_text(encoding="utf-8", errors="replace")
        forbidden_ledger_markers = ("ledger.sqlite", "ledger-blobs", "docker.sock")
        ledger_mounts = [marker for marker in forbidden_ledger_markers if marker in mountinfo]
        visible_ledger_paths = [
            path.name
            for path in private.rglob("*")
            if path.name in forbidden_ledger_markers or "ledger" in path.name.lower()
        ]
        if ledger_mounts or visible_ledger_paths:
            raise EGVError("evaluator process can see a ledger or Docker socket path")
        if evaluator_connect_counter[0] != 1:
            raise EGVError("receipt-only evaluator attempted unexpected SQLite access")
        cohost_evidence["connect_calls"] = evaluator_connect_counter[0]
        results_by_task = {task_id: result for (_, task_id), result in zip(_HELDOUT_TASKS, results)}
        output.put(
            {
                "disposition": results[0].disposition,
                "diagnostic_enum": results[0].diagnostic_enum,
                "family_results": {
                    family_id: results_by_task[task_id].to_dict()
                    for family_id, task_id in _FAMILY_TASKS
                },
                "heldout_results": {
                    task_id: result.to_dict() for (_, task_id), result in zip(_HELDOUT_TASKS, results)
                },
                "heldout_task_count": len(results),
                "wrong_locus_diagnostic": wrong_result.diagnostic_enum,
                "wrong_locus_disposition": wrong_result.disposition,
                "journal": journal.verify(),
                "negative_controls": negative_controls,
                "evaluator_ipc_methods": ["ingest_receipt"],
                "evaluator_used_sqlite": False,
                "evaluator_sqlite_connect_calls": int(sqlite_evidence.get("connect_calls", 0)),
                "evaluator_process_sqlite_connect_calls": evaluator_connect_counter[0],
                "evaluator_evaluate_sqlite_connect_calls": 0,
                "evaluator_ledger_cohosting_evidence": cohost_evidence,
                "evaluator_sqlite_evidence": {
                    **dict(sqlite_evidence),
                    "evaluator_process_hook_installed": bool(evaluator_sqlite_hook.get("hook_installed")),
                },
                "evaluator_ledger_mounts": ledger_mounts,
                "evaluator_ledger_paths_visible": visible_ledger_paths,
                "evaluator_private_tree_clean": not visible_ledger_paths,
                "evaluator_key_ephemeral": True,
                "evaluator_key_id": signer.key_id,
                "authority_runtime": runtime.status.to_dict(),
                "evaluator_non_receipt_ipc_rejected": non_receipt_rejected,
                "evaluated_corpus_digest": evaluated_corpus_digest,
                "evaluated_source_bindings": evaluated_source_bindings,
            }
        )
    except BaseException as exc:
        _put_error(errors, exc)
    finally:
        evaluator_done.set()


def _drain_errors(errors: Any) -> list[Dict[str, str]]:
    found = []
    while True:
        try:
            found.append(dict(errors.get_nowait()))
        except queue.Empty:
            return found


def run_evaluation_two_process_smoke(
    *,
    corpus: Optional[EvaluationCorpus] = None,
    evaluator_seed_file: Optional[Union[Path, str]] = None,
    frozen_root: Optional[Union[Path, str]] = None,
    frozen_data_manifest_digest: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the bounded Evaluation controller with trainer/evaluator OS processes.

    When a frozen corpus is supplied, only its evaluator-private seed path and
    expected manifest digest cross into the evaluator boundary.  The trainer
    receives neither the seed nor the hidden corpus; the evaluator regenerates
    it privately and reads held-out candidate bytes from ``frozen_root``.
    """

    from ..ledger import EvidenceLedger

    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="egv-evaluation-process-") as directory:
        root = Path(directory)
        evaluated_corpus, evaluator_seed_path, expected_corpus_digest = _prepare_evaluator_corpus(
            root,
            corpus=corpus,
            evaluator_seed_file=evaluator_seed_file,
            expected_manifest_digest=frozen_data_manifest_digest,
        )
        if frozen_root is not None:
            _validate_frozen_candidate_artifacts(frozen_root, evaluated_corpus)
        trainer_root = root / "trainer-ledger"
        evaluator_root = root / "evaluator-private"
        socket_path = root / "ledger-writer.sock"
        trainer_token = new_auth_token()
        evaluator_token = new_auth_token()
        protocol_digest = digest_for("egv-evaluation-process-protocol-v1")
        policy_digest = AuthorityPolicy.candidate_execution().digest
        ready = context.Event()
        evaluator_done = context.Event()
        output = context.Queue()
        errors = context.Queue()
        evaluator_public_key = context.Queue()
        sqlite_probe_path = root / "sqlite-negative-control.sqlite"
        sqlite_probe_output = context.Queue()
        sqlite_probe = context.Process(
            target=_sqlite_negative_control_process,
            args=(str(sqlite_probe_path), sqlite_probe_output),
            name="egv-evaluation-sqlite-negative-control",
        )
        sqlite_probe.start()
        sqlite_probe.join(30)
        if sqlite_probe.is_alive():
            sqlite_probe.terminate()
            sqlite_probe.join(5)
        if sqlite_probe.exitcode != 0:
            raise EGVError("SQLite negative-control process failed: {}".format(sqlite_probe.exitcode))
        try:
            sqlite_evidence = dict(sqlite_probe_output.get(timeout=5))
        except queue.Empty as exc:
            raise EGVError("SQLite negative-control returned no evidence") from exc
        if not sqlite_evidence.get("denied") or sqlite_evidence.get("probe_path_created"):
            raise EGVError("SQLite negative-control did not prove immutable denial")
        trainer = context.Process(
            target=_trainer_process,
            args=(
                str(trainer_root / "ledger.sqlite"),
                str(trainer_root / "ledger-blobs"),
                str(socket_path),
                trainer_token,
                evaluator_token,
                protocol_digest,
                policy_digest,
                ready,
                evaluator_done,
                output,
                errors,
                evaluator_public_key,
            ),
            name="egv-evaluation-trainer",
        )
        evaluator = context.Process(
            target=_evaluator_process,
            args=(
                str(socket_path),
                evaluator_token,
                str(evaluator_root),
                str(evaluator_seed_path),
                str(frozen_root) if frozen_root is not None else None,
                expected_corpus_digest,
                protocol_digest,
                policy_digest,
                ready,
                evaluator_done,
                output,
                errors,
                sqlite_evidence,
                evaluator_public_key,
            ),
            name="egv-evaluation-evaluator",
        )
        trainer.start()
        if not ready.wait(60):
            trainer.terminate()
            trainer.join(5)
            raise EGVError("Evaluation trainer process did not expose IPC")
        evaluator.start()
        evaluator.join(180)
        trainer.join(180)
        if evaluator.is_alive():
            evaluator.terminate()
            evaluator.join(5)
        if trainer.is_alive():
            trainer.terminate()
            trainer.join(5)
        process_errors = _drain_errors(errors)
        if process_errors or trainer.exitcode != 0 or evaluator.exitcode != 0:
            raise EGVError(
                "Evaluation two-process smoke failed: trainer_exit={}, evaluator_exit={}, errors={}".format(
                    trainer.exitcode, evaluator.exitcode, process_errors
                )
            )
        messages = []
        try:
            while True:
                messages.append(dict(output.get_nowait()))
        except queue.Empty:
            pass
        evaluator_result = next((message for message in messages if "family_results" in message), None)
        trainer_result = next((message for message in messages if "family_dispositions" in message), None)
        if evaluator_result is None or trainer_result is None:
            raise EGVError("Evaluation two-process smoke returned incomplete process evidence")
        with EvidenceLedger(trainer_root / "ledger.sqlite", mode="read_only", blob_root=trainer_root / "ledger-blobs") as ledger:
            post_integrity_raw = ledger.verify_integrity()
            post_integrity = {
                "chain_valid": bool(post_integrity_raw.get("chain_valid")),
                "event_count": post_integrity_raw["event_count"],
                "receipt_count": post_integrity_raw["receipt_count"],
            }
            post_dispositions = {
                task_id: ledger.candidate_disposition(_candidate_id(family_id, task_id))
                for family_id, task_id in _HELDOUT_TASKS
            }
            post_status = ledger.status()
            post_status.pop("path", None)
            wrong_locus_disposition = ledger.candidate_disposition(_WRONG_CANDIDATE_ID)
        if set(post_dispositions.values()) != {"PROMOTED"} or wrong_locus_disposition != "REJECTED":
            raise EGVError("Evaluation two-process ledger did not derive the closed dispositions")
        return {
            "mode": "two-process-cpu-only-evaluation",
            "trainer_process_exit": trainer.exitcode,
            "evaluator_process_exit": evaluator.exitcode,
            "evaluator_disposition": evaluator_result["disposition"],
            "evaluator_diagnostic": evaluator_result["diagnostic_enum"],
            "evaluator_used_sqlite": evaluator_result["evaluator_used_sqlite"],
            "evaluator_ipc_methods": evaluator_result["evaluator_ipc_methods"],
            "evaluator_sqlite_connect_calls": evaluator_result["evaluator_sqlite_connect_calls"],
            "evaluator_process_sqlite_connect_calls": evaluator_result["evaluator_process_sqlite_connect_calls"],
            "evaluator_evaluate_sqlite_connect_calls": evaluator_result["evaluator_evaluate_sqlite_connect_calls"],
            "evaluator_ledger_cohosting_evidence": evaluator_result["evaluator_ledger_cohosting_evidence"],
            "evaluator_sqlite_evidence": evaluator_result["evaluator_sqlite_evidence"],
            "evaluator_ledger_mounts": evaluator_result["evaluator_ledger_mounts"],
            "evaluator_ledger_paths_visible": evaluator_result["evaluator_ledger_paths_visible"],
            "evaluator_private_tree_clean": evaluator_result["evaluator_private_tree_clean"],
            "evaluator_key_ephemeral": evaluator_result["evaluator_key_ephemeral"],
            "evaluator_key_id": evaluator_result["evaluator_key_id"],
            "evaluated_corpus_digest": evaluator_result["evaluated_corpus_digest"],
            "evaluated_source_bindings": evaluator_result["evaluated_source_bindings"],
            "authority_runtime": evaluator_result["authority_runtime"],
            "evaluator_non_receipt_ipc_rejected": evaluator_result["evaluator_non_receipt_ipc_rejected"],
            "negative_controls": evaluator_result["negative_controls"],
            "journal": {
                "count": evaluator_result["journal"]["count"],
                "chain_valid": evaluator_result["journal"]["chain_valid"],
            },
            "family_results": evaluator_result["family_results"],
            "heldout_results": evaluator_result["heldout_results"],
            "heldout_task_count": evaluator_result["heldout_task_count"],
            "heldout_dispositions": post_dispositions,
            "signed_verdict_correctness": trainer_result["signed_verdict_correctness"],
            "family_dispositions": {
                family_id: post_dispositions[task_id] for family_id, task_id in _FAMILY_TASKS
            },
            "wrong_locus_diagnostic": evaluator_result["wrong_locus_diagnostic"],
            "wrong_locus_disposition": wrong_locus_disposition,
            "trainer_candidate_disposition": trainer_result["candidate_disposition"],
            "trainer_integrity": trainer_result["integrity"],
            "post_integrity": post_integrity,
            "post_status": post_status,
            "post_disposition": trainer_result["candidate_disposition"],
        }


__all__ = ["run_evaluation_two_process_smoke"]
