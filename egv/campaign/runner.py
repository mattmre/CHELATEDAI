"""Production commissioning runner joining frozen requests to Variation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Optional

from ..canonical import canonical_bytes, content_id, digest_bytes, digest_for
from ..evaluation.authority import AuthorityPolicy
from ..evaluation.dataset import EvaluationCorpus
from ..ledger import EvidenceLedger
from ..variation import (
    ArmIsolation,
    BoundedCandidateLoop,
    CheckpointStore,
    ModelCandidateGenerator,
    PinnedModelLoader,
    PrivateTrajectoryStore,
    RemoteControllerEvaluationGateway,
    VariationTask,
)
from ..variation.generator import (
    SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST,
    model_generation_profile_digest,
)
from ..variation.loop import SourceContractBudgetExhausted, VARIATION_PROTOCOL_DIGEST
from ..variation.model import MODEL_REVISION
from ..training.contracts import LedgerCutoff
from ..training.dataset import TrajectoryDatasetBuilder, seal_runtime_dataset
from ..training.protocol import training_sequence_token_count
from .commissioning import (
    GENERATION_REQUEST_COUNT,
    REQUEST_MANIFEST_SCHEMA,
    COMMISSIONING_RESPONSE_CONTRACT,
    TRAINER_INPUTS_SCHEMA,
    TRAIN_MANIFEST_SCHEMA,
    TRAIN_SOURCE_MANIFEST_SCHEMA,
)
from .errors import CampaignError
from .trajectories import (
    COMMISSIONING_ARMS,
    COMMISSIONING_SEEDS,
    CommissioningTrajectoryError,
    GenerationRequest,
    GenerationResponse,
    validate_accepted_response,
)


RUN_JOURNAL_SCHEMA = "egv-commissioning-run-journal-v1"
RUN_REPORT_SCHEMA = "egv-commissioning-run-report-v1"


class CommissioningRunError(CampaignError):
    """Frozen commissioning inputs cannot be executed or resumed safely."""


class CommissioningTrainerInputs:
    """Closed trainer-only projection of the 20/80 commissioning manifests."""

    def __init__(self, value: Mapping[str, Any]) -> None:
        fields = {
            "schema_version", "campaign_id", "corpus_manifest_digest", "model_manifest_digest",
            "variation_protocol_digest", "response_contract", "response_contract_digest",
            "generation_profile_digest", "trainer_sources_digest", "train_manifest", "request_manifest",
            "trainer_inputs_digest",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise CommissioningRunError("commissioning trainer inputs are not closed")
        unsigned = dict(value)
        supplied = unsigned.pop("trainer_inputs_digest")
        if value["schema_version"] != TRAINER_INPUTS_SCHEMA or digest_for(unsigned) != supplied:
            raise CommissioningRunError("commissioning trainer inputs digest is invalid")
        if value["variation_protocol_digest"] != VARIATION_PROTOCOL_DIGEST:
            raise CommissioningRunError("commissioning trainer protocol is stale")
        if (
            value["response_contract"] != COMMISSIONING_RESPONSE_CONTRACT
            or value["response_contract_digest"] != SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST
        ):
            raise CommissioningRunError("commissioning trainer response contract is stale")
        train = value["train_manifest"]
        requests = value["request_manifest"]
        if not isinstance(train, Mapping) or set(train) != {
            "schema_version", "corpus_manifest_digest", "task_count", "family_counts", "tasks"
        } or train["schema_version"] != TRAIN_MANIFEST_SCHEMA or train["task_count"] != 20:
            raise CommissioningRunError("commissioning training manifest is invalid")
        if train["corpus_manifest_digest"] != value["corpus_manifest_digest"]:
            raise CommissioningRunError("commissioning corpus bindings differ")
        if not isinstance(train["tasks"], list) or len(train["tasks"]) != 20:
            raise CommissioningRunError("commissioning requires exactly 20 public training tasks")
        task_fields = {
            "template_id", "family_id", "split", "ordinal", "source_digest", "public_rule_id", "public_locus"
        }
        if any(not isinstance(item, Mapping) or set(item) != task_fields or item.get("split") != "train"
               for item in train["tasks"]):
            raise CommissioningRunError("commissioning trainer task records are not closed train records")
        if not isinstance(requests, Mapping) or set(requests) != {
            "schema_version", "request_count", "arms", "seeds", "requests"
        } or (
            requests["schema_version"] != REQUEST_MANIFEST_SCHEMA
            or requests["request_count"] != GENERATION_REQUEST_COUNT
            or requests["arms"] != list(COMMISSIONING_ARMS)
            or requests["seeds"] != list(COMMISSIONING_SEEDS)
            or not isinstance(requests["requests"], list)
        ):
            raise CommissioningRunError("commissioning request manifest is invalid")
        try:
            parsed = tuple(GenerationRequest.from_mapping(item) for item in requests["requests"])
        except (CommissioningTrajectoryError, TypeError, ValueError) as exc:
            raise CommissioningRunError("commissioning request envelope is invalid") from exc
        if len(parsed) != GENERATION_REQUEST_COUNT or tuple(item.request_id for item in parsed) != tuple(
            sorted(item.request_id for item in parsed)
        ):
            raise CommissioningRunError("commissioning requests are incomplete or unordered")
        tasks = {item["template_id"]: dict(item) for item in train["tasks"]}
        if len(tasks) != 20:
            raise CommissioningRunError("commissioning training task identities are ambiguous")
        expected = []
        for task in train["tasks"]:
            for arm_id in COMMISSIONING_ARMS:
                for seed in COMMISSIONING_SEEDS:
                    expected.append(
                        GenerationRequest.build(
                            campaign_id=value["campaign_id"],
                            task_record=task,
                            corpus_manifest_digest=value["corpus_manifest_digest"],
                            arm_id=arm_id,
                            seed=seed,
                            model_manifest_digest=value["model_manifest_digest"],
                            variation_protocol_digest=value["variation_protocol_digest"],
                            generation_profile_digest=value["generation_profile_digest"],
                        )
                    )
        expected.sort(key=lambda item: item.request_id)
        if tuple(item.to_dict() for item in parsed) != tuple(item.to_dict() for item in expected):
            raise CommissioningRunError("commissioning request matrix or canonical bindings differ")
        for request in parsed:
            task = tasks.get(request.task_id)
            if (
                task is None
                or request.campaign_id != value["campaign_id"]
                or request.task_family != task["family_id"]
                or request.task_record_digest != digest_for(task)
                or request.corpus_manifest_digest != value["corpus_manifest_digest"]
                or request.model_manifest_digest != value["model_manifest_digest"]
                or request.response_contract != value["response_contract"]
                or request.response_contract_digest != value["response_contract_digest"]
                or request.generation_profile_digest != value["generation_profile_digest"]
            ):
                raise CommissioningRunError("commissioning request differs from trainer inputs")
        self._value = dict(value)
        self.requests = parsed
        self.tasks = tasks

    @classmethod
    def from_path(cls, path: Path) -> "CommissioningTrainerInputs":
        target = Path(path)
        if target.is_symlink() or not target.is_file():
            raise CommissioningRunError("commissioning trainer inputs must be a regular file")
        try:
            return cls(json.loads(target.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise CommissioningRunError("commissioning trainer inputs cannot be decoded") from exc

    @property
    def campaign_id(self) -> str:
        return self._value["campaign_id"]

    @property
    def corpus_manifest_digest(self) -> str:
        return self._value["corpus_manifest_digest"]

    @property
    def model_manifest_digest(self) -> str:
        return self._value["model_manifest_digest"]

    @property
    def trainer_sources_digest(self) -> str:
        return self._value["trainer_sources_digest"]

    @property
    def generation_profile_digest(self) -> str:
        return self._value["generation_profile_digest"]


class CommissioningTrainerSources:
    """Exact public train source bytes, separate from evaluator-private inputs."""

    def __init__(self, value: Mapping[str, Any], *, trainer_inputs: CommissioningTrainerInputs) -> None:
        fields = {
            "schema_version", "campaign_id", "corpus_manifest_digest", "task_count", "tasks",
            "trainer_sources_digest",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise CommissioningRunError("commissioning trainer source bundle is not closed")
        unsigned = dict(value)
        supplied = unsigned.pop("trainer_sources_digest")
        if (
            value["schema_version"] != TRAIN_SOURCE_MANIFEST_SCHEMA
            or digest_for(unsigned) != supplied
            or supplied != trainer_inputs.trainer_sources_digest
            or value["campaign_id"] != trainer_inputs.campaign_id
            or value["corpus_manifest_digest"] != trainer_inputs.corpus_manifest_digest
            or value["task_count"] != len(trainer_inputs.tasks)
            or not isinstance(value["tasks"], list)
        ):
            raise CommissioningRunError("commissioning trainer source bundle digest or identity is invalid")
        sources: Dict[str, str] = {}
        expected_ids = list(trainer_inputs.tasks)
        actual_ids = []
        for record in value["tasks"]:
            if not isinstance(record, Mapping) or set(record) != {
                "template_id", "repository_source_digest", "source_files"
            }:
                raise CommissioningRunError("commissioning trainer source record is not closed")
            task_id = record["template_id"]
            files = record["source_files"]
            if (
                not isinstance(task_id, str)
                or not isinstance(files, list)
                or len(files) != 2
                or any(not isinstance(item, Mapping) or set(item) != {"path", "content_utf8"} for item in files)
                or [item.get("path") for item in files] != ["README.md", "src/task.py"]
                or any(not isinstance(item.get("content_utf8"), str) or not item["content_utf8"] for item in files)
            ):
                raise CommissioningRunError("commissioning trainer source files are missing or invalid")
            source_map = {item["path"]: item["content_utf8"] for item in files}
            task_record = trainer_inputs.tasks.get(task_id)
            if (
                task_record is None
                or record["repository_source_digest"] != task_record["source_digest"]
                or digest_for(source_map) != task_record["source_digest"]
            ):
                raise CommissioningRunError("commissioning trainer source bytes differ from the public task record")
            actual_ids.append(task_id)
            sources[task_id] = source_map["src/task.py"]
        if actual_ids != expected_ids or len(sources) != len(expected_ids):
            raise CommissioningRunError("commissioning trainer source bundle is missing, duplicated, or unordered")
        self._value = dict(value)
        self.sources = sources

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        trainer_inputs: CommissioningTrainerInputs,
    ) -> "CommissioningTrainerSources":
        target = Path(path)
        if target.is_symlink() or not target.is_file():
            raise CommissioningRunError("commissioning trainer source bundle must be a regular non-symlink file")
        try:
            raw = target.read_bytes()
            value = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise CommissioningRunError("commissioning trainer source bundle cannot be decoded") from exc
        if canonical_bytes(value) + b"\n" != raw:
            raise CommissioningRunError("commissioning trainer source bundle is not canonical")
        return cls(value, trainer_inputs=trainer_inputs)

    def source_for(self, task_id: str) -> str:
        try:
            return self.sources[task_id]
        except KeyError as exc:
            raise CommissioningRunError("commissioning task has no exact trainer source") from exc


class CommissioningRunJournal:
    """Atomic idempotency journal; a completed request is never re-executed."""

    def __init__(self, path: Path, *, trainer_inputs_digest: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.trainer_inputs_digest = trainer_inputs_digest
        if not self.path.exists():
            self._write({
                "schema_version": RUN_JOURNAL_SCHEMA,
                "trainer_inputs_digest": trainer_inputs_digest,
                "completed": {},
            })
        self.load()

    def load(self) -> Dict[str, Any]:
        try:
            raw = self.path.read_bytes()
            value = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise CommissioningRunError("commissioning run journal cannot be decoded") from exc
        if canonical_bytes(value) != raw or set(value) != {"schema_version", "trainer_inputs_digest", "completed"}:
            raise CommissioningRunError("commissioning run journal is not canonical and closed")
        if (
            value["schema_version"] != RUN_JOURNAL_SCHEMA
            or value["trainer_inputs_digest"] != self.trainer_inputs_digest
            or not isinstance(value["completed"], Mapping)
        ):
            raise CommissioningRunError("commissioning run journal differs from frozen inputs")
        return value

    def completed(self, request_id: str) -> Optional[Mapping[str, Any]]:
        value = self.load()["completed"].get(request_id)
        return dict(value) if isinstance(value, Mapping) else None

    def commit(self, request: GenerationRequest, result: Mapping[str, Any]) -> None:
        value = self.load()
        completed = dict(value["completed"])
        record = dict(result)
        prior = completed.get(request.request_id)
        if prior is not None and prior != record:
            raise CommissioningRunError("commissioning request journal has conflicting completion")
        completed[request.request_id] = record
        value["completed"] = dict(sorted(completed.items()))
        self._write(value)

    def _write(self, value: Mapping[str, Any]) -> None:
        encoded = canonical_bytes(value)
        descriptor, name = tempfile.mkstemp(prefix=".commissioning-run-", dir=str(self.path.parent))
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary), str(self.path))
        finally:
            if temporary.exists():
                temporary.unlink()


def _task_for(request: GenerationRequest, record: Mapping[str, Any]) -> VariationTask:
    return VariationTask(
        task_id=request.task_id,
        family_id=request.task_family,
        public_locus=record["public_locus"],
        public_rule_id=record["public_rule_id"],
        task_statement="Repair the bounded {} task at {}.".format(
            request.task_family, record["public_locus"]
        ),
        evaluator_input=None,
    )


def _response_for(
    request: GenerationRequest,
    report: Any,
    *,
    ledger: EvidenceLedger,
    private_store: PrivateTrajectoryStore,
    evaluator_public_key: Any,
    evaluator_digest: str,
) -> Optional[Dict[str, Any]]:
    if not report.promoted:
        return None
    attempt = report.attempts[-1]
    receipts = []
    for receipt_id in attempt.receipt_ids:
        stored = ledger.receipt_by_id(receipt_id)
        if stored is not None:
            receipts.append(dict(stored["receipt"]))
    if [item.get("receipt_type") for item in receipts] != ["AUTHORITY", "VERDICT", "EFFECT"]:
        raise CommissioningRunError("promoted commissioning attempt lacks its exact receipt chain")
    private_attempt = private_store.load_attempts([attempt.candidate_id])[0]
    candidate_row = ledger.connection.execute(
        "SELECT candidate_json FROM candidates WHERE candidate_id=?",
        (attempt.candidate_id,),
    ).fetchone()
    try:
        candidate_metadata = json.loads(candidate_row["candidate_json"])["metadata"]
    except (KeyError, TypeError, ValueError) as exc:
        raise CommissioningRunError("promoted candidate lacks immutable generation metadata") from exc
    if (
        not isinstance(candidate_metadata, Mapping)
        or candidate_metadata.get("generation_evidence_digest")
        != private_attempt.generation_evidence_digest
    ):
        raise CommissioningRunError("promoted candidate differs from its private generation evidence")
    source = private_attempt.candidate_source
    payload = {
        "schema_version": "egv-commissioning-generation-response-v1",
        "request_id": request.request_id,
        "candidate_id": attempt.candidate_id,
        "candidate_artifact_digest": attempt.candidate_artifact_digest,
        "model_output_digest": attempt.candidate_artifact_digest,
        "output_byte_count": len(source),
        "disposition": "PROMOTED",
        "receipts": receipts,
    }
    payload["response_id"] = content_id("genresp", payload)
    response = GenerationResponse.from_mapping(payload)
    accepted = validate_accepted_response(
        request,
        response,
        evaluator_public_key=evaluator_public_key,
        evaluator_digest=evaluator_digest,
        expected_first_sequence=receipts[0]["sequence"],
        expected_previous_receipt_hash=receipts[0]["previous_receipt_hash"],
    )
    return {"response": response.to_dict(), "accepted_evidence": accepted}


def run_commissioning(
    *,
    trainer_inputs_path: Path,
    trainer_sources_path: Path,
    model_root: Path,
    ledger_path: Path,
    blob_root: Path,
    evaluator_manifest: Path,
    evaluator_public_key: Path,
    evaluator_command: Path,
    workspace_root: Path,
    journal_path: Path,
    source_commit: str,
    request_id: Optional[str] = None,
    max_attempts: int = 12,
    device: str = "cuda",
) -> Mapping[str, Any]:
    """Run one or all frozen requests; reruns skip journaled terminal requests."""

    inputs = CommissioningTrainerInputs.from_path(trainer_inputs_path)
    sources = CommissioningTrainerSources.from_path(
        trainer_sources_path,
        trainer_inputs=inputs,
    )
    selected = tuple(item for item in inputs.requests if request_id is None or item.request_id == request_id)
    if not selected:
        raise CommissioningRunError("requested commissioning request ID is not frozen")
    try:
        import torch
    except ImportError as exc:
        raise CommissioningRunError("commissioning runner requires torch") from exc
    if device != "cuda" or not torch.cuda.is_available():
        raise CommissioningRunError("commissioning production generation requires CUDA")
    loaded = PinnedModelLoader(model_root).load(device=device, torch_dtype=torch.bfloat16)
    if loaded.manifest_digest != inputs.model_manifest_digest:
        raise CommissioningRunError("loaded model differs from commissioning inputs")
    generator = ModelCandidateGenerator(
        loaded,
        model_digest=loaded.manifest_digest,
        response_contract=COMMISSIONING_RESPONSE_CONTRACT,
    )
    if (
        generator.response_contract_digest != inputs._value["response_contract_digest"]
        or generator.generation_profile_digest != inputs.generation_profile_digest
    ):
        raise CommissioningRunError(
            "loaded generator differs from the frozen commissioning response or generation profile"
        )
    isolation = ArmIsolation(workspace_root, campaign_id=inputs.campaign_id)
    private_store = PrivateTrajectoryStore(Path(workspace_root) / "private-trajectories")
    journal = CommissioningRunJournal(
        journal_path,
        trainer_inputs_digest=inputs._value["trainer_inputs_digest"],
    )
    results = []
    with EvidenceLedger(ledger_path, blob_root=blob_root) as ledger:
        gateway = RemoteControllerEvaluationGateway(
            ledger=ledger,
            manifest_path=evaluator_manifest,
            public_key_path=evaluator_public_key,
            command=evaluator_command,
        )
        ledger.verify_receipt_chain(evaluator_public_key.read_bytes())
        for request in selected:
            prior = journal.completed(request.request_id)
            if prior is not None:
                results.append(prior)
                continue
            task = _task_for(request, inputs.tasks[request.task_id])
            initial_source = sources.source_for(request.task_id)
            runner = BoundedCandidateLoop(
                ledger=ledger,
                evaluator=gateway,
                generator=generator,
                isolation=isolation,
                workspace_root=workspace_root,
                campaign_id=inputs.campaign_id,
                source_commit=source_commit,
                model_revision=loaded.manifest.revision,
                model_digest=loaded.manifest_digest,
                data_manifest_digest=inputs.corpus_manifest_digest,
                policy_digest=AuthorityPolicy.candidate_execution().digest,
                arm_id=request.arm_id,
                max_attempts=max_attempts,
                seed_set=(0, 1),
                private_store=private_store,
                initial_source=initial_source.encode("utf-8"),
                response_contract_digest=request.response_contract_digest,
                generation_profile_digest=request.generation_profile_digest,
            )
            if runner._run_id(task.task_id, request.seed) != request.run_id:
                raise CommissioningRunError("Variation run identity differs from frozen request")
            workspace = isolation.workspace(request.arm_id, request.run_id)
            latest = CheckpointStore(workspace.checkpoints).latest(run_id=request.run_id)
            resume_from = latest[0] if latest is not None else None
            try:
                report = runner.run(task, seed=request.seed, resume_from=resume_from)
            except SourceContractBudgetExhausted as exc:
                record = {
                    "request_id": request.request_id,
                    "run_id": request.run_id,
                    "terminal_status": "BUDGET_EXHAUSTED",
                    "report_digest": None,
                    "response": None,
                    "generation_failure_digest": exc.last_failure_digest,
                    "source_contract_failure_count": exc.failure_count,
                }
                journal.commit(request, record)
                results.append(record)
                continue
            ledger.verify_receipt_chain(evaluator_public_key.read_bytes())
            accepted = _response_for(
                request,
                report,
                ledger=ledger,
                private_store=private_store,
                evaluator_public_key=evaluator_public_key.read_bytes(),
                evaluator_digest=gateway.manifest.digest,
            )
            record = {
                "request_id": request.request_id,
                "run_id": request.run_id,
                "terminal_status": report.terminal_status,
                "report_digest": digest_for(report.to_dict()),
                "response": accepted,
                "generation_failure_digest": None,
                "source_contract_failure_count": len(
                    private_store.source_contract_failures(
                        run_id=request.run_id,
                        task_id=request.task_id,
                        arm_id=request.arm_id,
                    )
                ),
            }
            journal.commit(request, record)
            results.append(record)
    completed = len(journal.load()["completed"])
    failures = sum(
        bool(item.get("generation_failure_digest"))
        for item in journal.load()["completed"].values()
    )
    return {
        "schema_version": RUN_REPORT_SCHEMA,
        "campaign_id": inputs.campaign_id,
        "selected_request_count": len(selected),
        "completed_request_count": completed,
        "total_request_count": GENERATION_REQUEST_COUNT,
        "generation_failure_count": failures,
        "status": (
            "TERMINAL_WITH_FAILURES"
            if completed == GENERATION_REQUEST_COUNT and failures
            else "COMPLETE"
            if completed == GENERATION_REQUEST_COUNT
            else "PENDING"
        ),
        "results": results,
    }


def freeze_commissioning_dataset(
    *,
    trainer_inputs_path: Path,
    ledger_path: Path,
    blob_root: Path,
    private_store_root: Path,
    evaluator_seed: Path,
    evaluator_public_key: Path,
    model_root: Path,
    output: Path,
) -> Mapping[str, Any]:
    """Build and seal training input from a copied frozen ledger on the evaluator."""

    inputs = CommissioningTrainerInputs.from_path(trainer_inputs_path)
    corpus = EvaluationCorpus.generate(secret_seed_file=evaluator_seed)
    if corpus.manifest_digest() != inputs.corpus_manifest_digest:
        raise CommissioningRunError("evaluator corpus differs from commissioning inputs")
    loader = PinnedModelLoader(model_root)
    manifest, _files = loader.verify_manifest()
    if manifest.digest() != inputs.model_manifest_digest:
        raise CommissioningRunError("tokenizer model differs from commissioning inputs")
    try:
        from transformers import AutoTokenizer

        with loader._offline_environment():
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_root), revision=MODEL_REVISION, local_files_only=True, trust_remote_code=False
            )
    except Exception as exc:
        raise CommissioningRunError("frozen tokenizer could not be loaded") from exc
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise CommissioningRunError("frozen tokenizer lacks exact chat-template bytes")
    generation_profile_digest = model_generation_profile_digest(
        COMMISSIONING_RESPONSE_CONTRACT,
        model_manifest_digest=manifest.digest(),
        chat_template_digest=digest_bytes(chat_template.encode("utf-8")),
        max_new_tokens=512,
    )
    if generation_profile_digest != inputs.generation_profile_digest:
        raise CommissioningRunError("frozen tokenizer differs from the commissioning generation profile")

    def count_tokens(prompt: str, target: str) -> int:
        return training_sequence_token_count(prompt, target, tokenizer)

    private_store = PrivateTrajectoryStore(private_store_root)
    with EvidenceLedger(ledger_path, blob_root=blob_root, create=False) as ledger:
        cutoff = LedgerCutoff.capture(ledger, campaign_id=inputs.campaign_id)
        candidate_ids = [
            str(row["candidate_id"])
            for row in ledger.connection.execute(
                "SELECT candidate_id FROM candidates WHERE campaign_id=? ORDER BY candidate_id",
                (inputs.campaign_id,),
            ).fetchall()
        ]
        attempts = private_store.load_attempts(candidate_ids)
        dataset = TrajectoryDatasetBuilder(
            ledger,
            corpus,
            token_counter=count_tokens,
            receipt_public_key=evaluator_public_key.read_bytes(),
            require_complete=True,
            expected_runs=(
                (request.run_id, request.task_id, request.arm_id, request.seed)
                for request in inputs.requests
            ),
            generation_profile_digest=inputs.generation_profile_digest,
        ).build(cutoff, attempts)
        result = seal_runtime_dataset(dataset, output)
    return {
        **result,
        "campaign_id": inputs.campaign_id,
        "cutoff_digest": cutoff.digest,
        "candidate_count": len(candidate_ids),
    }


__all__ = [
    "CommissioningRunError", "CommissioningRunJournal", "CommissioningTrainerInputs",
    "CommissioningTrainerSources",
    "RUN_JOURNAL_SCHEMA", "RUN_REPORT_SCHEMA", "freeze_commissioning_dataset", "run_commissioning",
]
