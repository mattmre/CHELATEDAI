"""Build one deterministic, cutoff-bound EGV trajectory dataset."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Dict, Iterable, Mapping, Tuple

from ..canonical import canonical_json, content_id, digest_bytes, digest_for
from ..evaluation.dataset import EvaluationCorpus
from ..evaluation.prompts import PROMPT_IDS, PromptRegistry
from ..evaluation.sft import build_sft_row, validate_sft_row
from ..ledger import STALE_DEPENDENT, RETRACTED
from ..receipts import key_id_for_public_key, receipt_hash
from ..variation.generator import CandidateContext, render_candidate_prompt
from ..variation.loop import VariationTask
from .contracts import (
    FrozenTrainingDataset,
    LedgerCutoff,
    MAX_TRAINING_TOKENS,
    PrivateTrajectoryAttempt,
    TRAINING_ARMS,
    TrainingExample,
)


_HOST_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/home/|/Users/|/root/)")
_SECRET_RE = re.compile(
    r"(?i)(?:password|passwd|private[_ -]?key|api[_ -]?key|authorization)\s*[:=]\s*[^\s,;]{4,}"
)
SEALED_RUNTIME_DATASET_SCHEMA = "egv-sealed-training-runtime-input-v1"


class TrajectoryDatasetBuilder:
    """Validate private trajectories against the authoritative frozen ledger."""

    def __init__(
        self,
        ledger: Any,
        corpus: EvaluationCorpus,
        *,
        token_counter: Callable[[str], int],
        receipt_public_key: Any,
        max_input_tokens: int = MAX_TRAINING_TOKENS,
        require_complete: bool = True,
    ) -> None:
        if not callable(token_counter):
            raise TypeError("training dataset requires the pinned tokenizer's token counter")
        if max_input_tokens != MAX_TRAINING_TOKENS:
            raise ValueError("training token ceiling is frozen at 4096")
        self.ledger = ledger
        self.corpus = corpus
        self.token_counter = token_counter
        if receipt_public_key is None:
            raise ValueError("training dataset requires the pinned evaluator receipt public key")
        self.receipt_public_key = receipt_public_key
        self.max_input_tokens = max_input_tokens
        self.require_complete = bool(require_complete)
        self.prompt_registry = PromptRegistry()

    @contextmanager
    def _frozen_snapshot(self, cutoff: LedgerCutoff):
        """Hold a SQLite write-reserving snapshot for the complete freeze read."""

        connection = self.ledger.connection
        if connection.in_transaction:
            raise ValueError("training freeze requires a fresh SQLite snapshot transaction")
        connection.execute("BEGIN IMMEDIATE")
        try:
            cutoff.validate(self.ledger)
            yield
            cutoff.validate(self.ledger)
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    def _verify_receipt_chain(self, cutoff: LedgerCutoff) -> None:
        if key_id_for_public_key(self.receipt_public_key) != cutoff.evaluator_key_id:
            raise ValueError("supplied evaluator public key differs from the frozen ledger key")
        verified = self.ledger.verify_receipt_chain(self.receipt_public_key)
        if verified.get("receipt_count") != cutoff.receipt_count:
            raise ValueError("receipt chain length differs from the frozen cutoff")
        if verified.get("receipt_head_hash") != cutoff.receipt_chain_head:
            raise ValueError("verified receipt chain head differs from the frozen cutoff")

    def _candidate_row(self, candidate_id: str) -> Mapping[str, Any]:
        row = self.ledger.connection.execute(
            "SELECT * FROM candidates WHERE candidate_id=?", (candidate_id,)
        ).fetchone()
        if row is None:
            raise ValueError("private trajectory references an unknown ledger candidate")
        return row

    def _receipts(self, candidate_id: str) -> Dict[str, Mapping[str, Any]]:
        values = [item for item in self.ledger.receipts() if item.get("candidate_id") == candidate_id]
        by_type: Dict[str, Mapping[str, Any]] = {}
        for receipt in values:
            kind = str(receipt.get("receipt_type"))
            if kind in by_type:
                raise ValueError("training candidate has duplicate {} receipts".format(kind))
            by_type[kind] = receipt
        return by_type

    def _validate_campaign_surface(self, cutoff: LedgerCutoff) -> None:
        rows = self.ledger.connection.execute(
            "SELECT arm,task_id,seed FROM runs WHERE campaign_id=? ORDER BY task_id,arm,seed",
            (cutoff.campaign_id,),
        ).fetchall()
        seen = set()
        for row in rows:
            try:
                repo = self.corpus.get(str(row["task_id"]))
            except KeyError as exc:
                raise ValueError("campaign run references a task outside the frozen corpus") from exc
            if repo.split != "train":
                raise ValueError("development or held-out task identity crossed the training cutoff")
            if str(row["arm"]) not in TRAINING_ARMS:
                raise ValueError("training cutoff contains an arm outside the frozen B+D union")
            key = (str(row["task_id"]), str(row["arm"]), int(row["seed"]))
            if key in seen:
                raise ValueError("training cutoff contains a duplicate task/arm/seed trajectory")
            seen.add(key)
        if self.require_complete:
            train_ids = {repo.template_id for repo in self.corpus.split("train")}
            seeds = {seed for _task, _arm, seed in seen}
            if len(train_ids) != 20 or len(seeds) != 2:
                raise ValueError("complete training freeze requires 20 tasks and exactly two seeds")
            expected = {(task, arm, seed) for task in train_ids for arm in TRAINING_ARMS for seed in seeds}
            if seen != expected:
                raise ValueError("training freeze is not the complete 20 x 2 x 2 trajectory grid")

    def _validate_context(
        self,
        context: CandidateContext,
        row: Mapping[str, Any],
        metadata: Mapping[str, Any],
        cutoff: LedgerCutoff,
    ) -> None:
        repo = self.corpus.get(context.task_id)
        run = self.ledger.connection.execute(
            "SELECT * FROM runs WHERE run_id=?", (context.run_id,)
        ).fetchone()
        if run is None:
            raise ValueError("candidate context references an unknown run")
        expected = {
            "campaign_id": cutoff.campaign_id,
            "run_id": str(run["run_id"]),
            "task_id": str(run["task_id"]),
            "arm_id": str(run["arm"]),
            "seed": int(run["seed"]),
        }
        actual = {
            "campaign_id": context.campaign_id,
            "run_id": context.run_id,
            "task_id": context.task_id,
            "arm_id": context.arm_id,
            "seed": context.seed,
        }
        if actual != expected:
            raise ValueError("private prompt context differs from the authoritative run")
        if repo.split != "train" or context.arm_id not in TRAINING_ARMS:
            raise ValueError("private prompt crossed the frozen split/arm boundary")
        if (
            context.family_id != repo.family_id
            or context.public_locus != repo.public_locus
            or context.public_rule_id != repo.public_rule_id
        ):
            raise ValueError("private prompt context differs from immutable task metadata")
        if int(metadata.get("attempt_index", -1)) != context.attempt_index:
            raise ValueError("private prompt attempt index differs from candidate metadata")
        if metadata.get("arm_id") != context.arm_id:
            raise ValueError("private prompt arm differs from candidate metadata")
        if row["parent_candidate_id"] != context.parent_candidate_id:
            raise ValueError("private prompt parent differs from candidate lineage")
        if row["model_hash"] != context.model_digest or row["adapter_hash"] != context.adapter_digest:
            raise ValueError("private prompt model/adapter binding differs from candidate")
        records = tuple(context.retrieval_records)
        event_ids = [str(record.get("event_id")) for record in records]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("private prompt retrieval view contains duplicate evidence IDs")
        if event_ids != sorted(event_ids):
            raise ValueError("private prompt retrieval view is not deterministic")
        if digest_for([dict(record) for record in records]) != context.retrieval_digest:
            raise ValueError("private prompt retrieval digest is not bound to its records")
        for event_id in event_ids:
            event = self.ledger.connection.execute(
                "SELECT sequence,run_id FROM events WHERE event_id=?", (event_id,)
            ).fetchone()
            if event is None or int(event["sequence"]) > cutoff.sequence:
                raise ValueError("private prompt retrieves evidence outside the frozen cutoff")
            if event["run_id"]:
                source_run = self.ledger.connection.execute(
                    "SELECT campaign_id,arm FROM runs WHERE run_id=?", (event["run_id"],)
                ).fetchone()
                if source_run is None or source_run["campaign_id"] != cutoff.campaign_id or source_run["arm"] != context.arm_id:
                    raise ValueError("private prompt contains cross-arm or cross-campaign retrieval")
        if metadata.get("retrieval_digest") != context.retrieval_digest:
            raise ValueError("candidate metadata retrieval digest differs from private context")
        task = VariationTask.from_microrepo(repo)
        expected_context_digest = digest_for(
            {
                "schema_version": "egv-variation-prompt-context-v1",
                "prompt_ids": list(PROMPT_IDS),
                "prompt_manifest_digest": self.prompt_registry.manifest_digest(),
                "task": task.public_dict(),
                "seed": context.seed,
                "attempt_index": context.attempt_index,
                "retrieval_digest": context.retrieval_digest,
            }
        )
        if context.prompt_digest != expected_context_digest or row["prompt_hash"] != expected_context_digest:
            raise ValueError("candidate prompt context digest is not reproducible")

    def _forbidden_fragments(self) -> Tuple[str, ...]:
        values = set()
        for repo in self.corpus.repositories:
            if repo.split == "train":
                continue
            values.add(repo.template_id)
            for _path, data in repo.source_files:
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                if len(text) >= 16:
                    values.add(text)
            try:
                corrected = repo.corrected_source.decode("utf-8")
            except UnicodeDecodeError:
                corrected = ""
            if len(corrected) >= 16:
                values.add(corrected)
            for value in repo.private_material.values():
                if isinstance(value, str) and len(value) >= 12:
                    values.add(value)
        return tuple(sorted(values))

    def _reject_private_leakage(self, prompt: str, target: str) -> None:
        combined = prompt + "\n" + target
        if _HOST_PATH_RE.search(combined):
            raise ValueError("training row contains a host-specific path")
        if _SECRET_RE.search(combined):
            raise ValueError("training row contains credential-like material")
        if any(fragment in combined for fragment in self._forbidden_fragments()):
            raise ValueError("training row contains development or held-out material")

    def _build_example(
        self,
        private: PrivateTrajectoryAttempt,
        cutoff: LedgerCutoff,
    ) -> TrainingExample | None:
        context = private.context
        if type(context) is not CandidateContext:
            raise ValueError("private trajectory requires the exact frozen CandidateContext")
        row = self._candidate_row(private.candidate_id)
        if row["campaign_id"] != cutoff.campaign_id or row["candidate_id"] != private.candidate_id:
            raise ValueError("candidate is not bound to the frozen campaign")
        candidate_payload = json.loads(row["candidate_json"])
        metadata = candidate_payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("candidate is missing its closed Variation metadata")
        self._validate_context(context, row, metadata, cutoff)
        valid_event_ids = {event["event_id"] for event in self.ledger.current_valid_events()}
        if row["event_id"] not in valid_event_ids:
            return None
        disposition = self.ledger.candidate_disposition(private.candidate_id)
        if disposition in {RETRACTED, STALE_DEPENDENT}:
            return None

        source_digest = digest_bytes(private.candidate_source)
        if row["patch_hash"] != source_digest or metadata.get("candidate_artifact_digest") != source_digest:
            raise ValueError("private candidate bytes differ from the ledger source digest")
        cited = tuple(str(item) for item in metadata.get("evidence_ids", ()))
        dependencies = tuple(
            sorted(
                str(item["parent_id"])
                for item in self.ledger.connection.execute(
                    "SELECT parent_id,edge_type FROM dependencies WHERE child_id=? ORDER BY parent_id",
                    (private.candidate_id,),
                ).fetchall()
                if item["edge_type"] == "EVIDENCE_USED"
            )
        )
        if cited != dependencies or not set(cited).issubset(
            {str(record["event_id"]) for record in context.retrieval_records}
        ):
            raise ValueError("candidate evidence declarations differ from durable dependencies")

        receipts = self._receipts(private.candidate_id)
        if set(receipts) != {"AUTHORITY", "VERDICT", "EFFECT"}:
            return None
        authority, verdict, effect = (receipts[kind] for kind in ("AUTHORITY", "VERDICT", "EFFECT"))
        for receipt in (authority, verdict, effect):
            if (
                receipt.get("campaign_id") != cutoff.campaign_id
                or receipt.get("run_id") != context.run_id
                or receipt.get("task_id") != context.task_id
                or receipt.get("candidate_id") != private.candidate_id
                or receipt.get("candidate_artifact_digest") != source_digest
            ):
                raise ValueError("signed receipt is not bound to the training candidate")
        if authority.get("decision") != "ALLOW" or effect.get("decision") != "ALLOW":
            return None
        # Rejected/failed candidates remain evidence for later prompts. They
        # are never serialized as desired assistant targets.
        if verdict.get("decision") != "PASS" or disposition != "PROMOTED":
            return None
        if not effect.get("output_digest") or not effect.get("environment_diff_digest"):
            return None

        prompt = render_candidate_prompt(context, prompt_registry=self.prompt_registry)
        target = canonical_json(
            {
                "declared_locus": context.public_locus,
                "evidence_ids": list(cited),
                "requested_authority": row["requested_authority"],
                "source": private.candidate_source.decode("utf-8"),
            }
        )
        self._reject_private_leakage(prompt, target)
        # Count the exact concatenated training sequence once so tokenizer
        # boundary/special-token behavior cannot hide an overlength example.
        token_count = self.token_counter(prompt + target)
        if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count < 1:
            raise ValueError("pinned tokenizer returned an invalid token count")
        if token_count > self.max_input_tokens:
            raise ValueError("training row exceeds the frozen 4096-token ceiling")

        sft_row = build_sft_row(
            task_id=context.task_id,
            task_family=context.family_id,
            arm=context.arm_id,
            attempt_index=context.attempt_index,
            attempt_id=private.candidate_id,
            prompt_template_ids=PROMPT_IDS,
            prompt_digest=context.prompt_digest,
            retrieved_evidence_ids=[str(record["event_id"]) for record in context.retrieval_records],
            proposed_mutation_digest=str(row["patch_hash"]),
            candidate_artifact_digest=source_digest,
            verdict_receipt_digest=receipt_hash(verdict),
            diagnostic_enum=str(verdict["diagnostic_enum"]),
            resource_bucket=str(verdict["resource_bucket"]),
            dependency_ids=dependencies,
            requested_authority=str(row["requested_authority"]),
            promotion_disposition=disposition,
            input_digest=str(verdict["input_digest"]),
            output_digest=str(verdict["output_digest"]),
            public_locus=context.public_locus,
            public_rule_id=context.public_rule_id,
            corpus=self.corpus,
        )
        validate_sft_row(sft_row, corpus=self.corpus, task_manifest_digest=self.corpus.manifest_digest())
        sft_row_json = canonical_json(sft_row)
        values = {
            "campaign_id": context.campaign_id,
            "run_id": context.run_id,
            "task_id": context.task_id,
            "arm_id": context.arm_id,
            "seed": context.seed,
            "attempt_index": context.attempt_index,
            "candidate_id": private.candidate_id,
            "prompt": prompt,
            "target": target,
            "prompt_digest": digest_bytes(prompt.encode("utf-8")),
            "target_digest": digest_bytes(target.encode("utf-8")),
            "sft_row_json": sft_row_json,
            "sft_row_digest": digest_bytes(sft_row_json.encode("utf-8")),
            "prompt_context_digest": context.prompt_digest,
            "source_digest": source_digest,
            "verdict_receipt_digest": receipt_hash(verdict),
            "effect_receipt_digest": receipt_hash(effect),
            "cutoff_digest": cutoff.digest,
            "input_token_count": token_count,
        }
        manifest_values = {
            "schema_version": "egv-private-training-example-v1",
            **{
                key: value
                for key, value in values.items()
                if key not in {"prompt", "target", "sft_row_json"}
            },
        }
        values["row_id"] = content_id("trainrow", manifest_values)
        return TrainingExample(**values)

    def build(
        self,
        cutoff: LedgerCutoff,
        private_attempts: Iterable[PrivateTrajectoryAttempt],
    ) -> FrozenTrainingDataset:
        with self._frozen_snapshot(cutoff):
            self._verify_receipt_chain(cutoff)
            self.corpus.validate()
            self._validate_campaign_surface(cutoff)
            supplied = tuple(private_attempts)
            ids = [item.candidate_id for item in supplied]
            if len(ids) != len(set(ids)):
                raise ValueError("private training attempts contain duplicate candidates")
            if self.require_complete:
                expected_ids = {
                    str(row["candidate_id"])
                    for row in self.ledger.connection.execute(
                        "SELECT candidate_id FROM candidates WHERE campaign_id=?", (cutoff.campaign_id,)
                    ).fetchall()
                }
                if set(ids) != expected_ids:
                    raise ValueError("private training material does not cover every frozen campaign candidate")
            examples = []
            excluded = Counter()
            for private in supplied:
                example = self._build_example(private, cutoff)
                if example is None:
                    excluded["invalid_or_ineligible_attempt"] += 1
                else:
                    examples.append(example)
            examples.sort(key=lambda row: (row.task_id, row.arm_id, row.seed, row.attempt_index, row.candidate_id))
            return FrozenTrainingDataset(cutoff, tuple(examples), dict(excluded))


def seal_runtime_dataset(dataset: FrozenTrainingDataset, output: Path) -> Dict[str, Any]:
    """Atomically write the exact private runtime artifact consumed by train-lora."""

    if type(dataset) is not FrozenTrainingDataset:
        raise TypeError("runtime dataset sealing requires the exact frozen dataset")
    value = {
        "schema_version": SEALED_RUNTIME_DATASET_SCHEMA,
        "manifest": dataset.manifest(),
        "private_rows": [row.private_record() for row in dataset.examples],
    }
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".sealed-training-", dir=str(target.parent))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(target))
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "schema_version": "egv-sealed-training-runtime-report-v1",
        "dataset_digest": dataset.digest,
        "row_count": len(dataset.examples),
        "output_digest": digest_bytes(encoded),
    }


__all__ = ["SEALED_RUNTIME_DATASET_SCHEMA", "TrajectoryDatasetBuilder", "seal_runtime_dataset"]
