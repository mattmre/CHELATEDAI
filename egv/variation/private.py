"""Durable evaluator-private trajectory sidecars for training freeze."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from ..canonical import canonical_bytes, digest_bytes
from ..evaluation.artifacts import ContentAddressedArtifactStore
from .errors import VariationCheckpointError
from .generator import (
    CandidateContext,
    CandidateGenerationEvidence,
    CandidateGenerationFailureEvidence,
    ModelCandidateGenerator,
    model_response_contract_digest,
)


LEGACY_PRIVATE_TRAJECTORY_SCHEMA = "egv-private-trajectory-sidecar-v1"
PRIVATE_TRAJECTORY_SCHEMA = "egv-private-trajectory-sidecar-v2"
PRIVATE_GENERATION_SCHEMA = "egv-private-generation-evidence-v1"
PRIVATE_GENERATION_START_SCHEMA = "egv-private-generation-start-v1"
PRIVATE_GENERATION_INTENT_SCHEMA = "egv-private-generation-intent-v1"
_PRIVATE_GENERATION_FIELDS = frozenset({
    "schema_version", "candidate_id", "status", "context", "response_contract",
    "response_contract_digest", "generation_profile_digest", "rendered_prompt_digest",
    "decoded_model_response_digest", "contract_response_digest", "proposal_source_digest",
    "failure_stage", "error_code",
})
_PRIVATE_GENERATION_INTENT_FIELDS = frozenset({
    "schema_version", "candidate_id", "status", "context", "response_contract",
    "response_contract_digest", "generation_profile_digest", "rendered_prompt_b64",
    "decoded_model_response_b64", "contract_response_b64", "proposal_source_digest",
    "failure_stage", "error_code",
})
_PRIVATE_GENERATION_START_FIELDS = frozenset({"schema_version", "candidate_id", "context"})


def _context_dict(context: CandidateContext) -> Dict[str, Any]:
    return {
        "campaign_id": context.campaign_id, "run_id": context.run_id, "seed": context.seed,
        "arm_id": context.arm_id, "task_id": context.task_id, "family_id": context.family_id,
        "public_locus": context.public_locus, "public_rule_id": context.public_rule_id,
        "attempt_index": context.attempt_index, "parent_candidate_id": context.parent_candidate_id,
        "retrieval_records": [dict(item) for item in context.retrieval_records],
        "retrieval_digest": context.retrieval_digest, "model_digest": context.model_digest,
        "adapter_digest": context.adapter_digest, "prompt_digest": context.prompt_digest,
        "task_statement": context.task_statement, "initial_source": context.initial_source,
        "initial_source_digest": context.initial_source_digest,
        "response_contract": context.response_contract,
        "response_contract_digest": context.response_contract_digest,
        "generation_profile_digest": context.generation_profile_digest,
    }


def _context_from_mapping(value: Mapping[str, Any]) -> CandidateContext:
    fields = tuple(CandidateContext.__dataclass_fields__)
    legacy_fields = fields[:15]
    source_fields_without_profile = fields[:-1]
    if not isinstance(value, Mapping):
        raise VariationCheckpointError("private candidate context is not closed")
    supplied_fields = set(value)
    if supplied_fields not in (
        set(legacy_fields), set(source_fields_without_profile), set(fields)
    ):
        raise VariationCheckpointError("private candidate context is not closed")
    payload = dict(value)
    records = payload.get("retrieval_records")
    if not isinstance(records, list) or any(not isinstance(item, Mapping) for item in records):
        raise VariationCheckpointError("private candidate retrieval records are malformed")
    payload["retrieval_records"] = tuple(dict(item) for item in records)
    if supplied_fields == set(legacy_fields):
        payload.update({
            "task_statement": None,
            "initial_source": None,
            "initial_source_digest": None,
            "response_contract": "closed-json-v1",
            "response_contract_digest": model_response_contract_digest("closed-json-v1"),
            "generation_profile_digest": None,
        })
    elif supplied_fields == set(source_fields_without_profile):
        payload["generation_profile_digest"] = None
    elif (
        payload.get("response_contract_digest")
        != model_response_contract_digest(payload.get("response_contract"))
    ):
        raise VariationCheckpointError("private candidate response-contract digest is invalid")
    return CandidateContext(**payload)


class PrivateTrajectoryStore:
    """Write-once context/source records excluded from public projections."""

    def __init__(self, root: Path) -> None:
        supplied_root = Path(os.path.abspath(os.fspath(root)))
        self._reject_link_components(supplied_root)
        supplied_root.mkdir(parents=True, exist_ok=True)
        self._validate_directory(supplied_root)
        self.root = supplied_root.resolve(strict=True)
        self.records = self.root / "records"
        self._ensure_directory(self.records, containment_root=self.root)
        self.generation_records = self.root / "generation-records"
        self._ensure_directory(self.generation_records, containment_root=self.root)
        self.generation_starts = self.root / "generation-starts"
        self._ensure_directory(self.generation_starts, containment_root=self.root)
        self.generation_intents = self.root / "generation-intents"
        self._ensure_directory(self.generation_intents, containment_root=self.root)
        artifacts_root = self.root / "artifacts"
        self._ensure_directory(artifacts_root, containment_root=self.root)
        self.artifacts = ContentAddressedArtifactStore(artifacts_root)

    @staticmethod
    def _link_like(metadata: os.stat_result) -> bool:
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse)

    @classmethod
    def _reject_link_components(cls, path: Path) -> None:
        cursor = Path(os.path.abspath(os.fspath(path)))
        while True:
            try:
                metadata = os.lstat(cursor)
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise VariationCheckpointError("private trajectory path metadata is unreadable") from exc
            else:
                if cls._link_like(metadata):
                    raise VariationCheckpointError("private trajectory path contains a link or reparse point")
            if cursor.parent == cursor:
                break
            cursor = cursor.parent

    @classmethod
    def _validate_directory(cls, path: Path, *, containment_root: Path | None = None) -> None:
        try:
            metadata = os.lstat(path)
        except OSError as exc:
            raise VariationCheckpointError("private trajectory directory is missing or unreadable") from exc
        if cls._link_like(metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise VariationCheckpointError("private trajectory directory is not a regular contained directory")
        if containment_root is not None:
            try:
                path.resolve(strict=True).relative_to(containment_root.resolve(strict=True))
            except (OSError, ValueError) as exc:
                raise VariationCheckpointError("private trajectory directory escapes its store") from exc

    @classmethod
    def _ensure_directory(cls, path: Path, *, containment_root: Path) -> None:
        cls._validate_directory(containment_root)
        root = containment_root.resolve(strict=True)
        try:
            relative = Path(os.path.abspath(os.fspath(path))).relative_to(root)
        except ValueError as exc:
            raise VariationCheckpointError("private trajectory directory escapes its store") from exc
        cursor = root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.exists() or cursor.is_symlink():
                cls._validate_directory(cursor, containment_root=root)
            else:
                try:
                    cursor.mkdir()
                except OSError as exc:
                    raise VariationCheckpointError("private trajectory directory cannot be created") from exc
                cls._validate_directory(cursor, containment_root=root)

    @classmethod
    def _validate_directory_chain(cls, path: Path, *, containment_root: Path) -> None:
        cls._validate_directory(containment_root)
        root = containment_root.resolve(strict=True)
        absolute = Path(os.path.abspath(os.fspath(path)))
        try:
            relative = absolute.relative_to(root)
        except ValueError as exc:
            raise VariationCheckpointError("private trajectory directory escapes its store") from exc
        cursor = root
        for part in relative.parts:
            cursor = cursor / part
            cls._validate_directory(cursor, containment_root=root)

    @classmethod
    def _read_regular_file(cls, path: Path, *, containment_root: Path) -> bytes:
        root = containment_root.resolve(strict=True)
        absolute = Path(os.path.abspath(os.fspath(path)))
        try:
            absolute.relative_to(root)
        except ValueError as exc:
            raise VariationCheckpointError("private evidence path escapes its store") from exc
        cls._validate_directory_chain(absolute.parent, containment_root=root)
        try:
            metadata = os.lstat(absolute)
        except OSError as exc:
            raise VariationCheckpointError("private evidence file is missing or unreadable") from exc
        if (
            cls._link_like(metadata)
            or not stat.S_ISREG(metadata.st_mode)
            or int(getattr(metadata, "st_nlink", 0)) != 1
        ):
            raise VariationCheckpointError("private evidence file is linked or not regular")
        try:
            absolute.resolve(strict=True).relative_to(root)
            return absolute.read_bytes()
        except (OSError, ValueError) as exc:
            raise VariationCheckpointError("private evidence file cannot be read inside its store") from exc

    def _validate_store_layout(self) -> None:
        self._validate_directory(self.root)
        self._validate_directory_chain(self.records, containment_root=self.root)
        self._validate_directory_chain(self.generation_records, containment_root=self.root)
        self._validate_directory_chain(self.generation_starts, containment_root=self.root)
        self._validate_directory_chain(self.generation_intents, containment_root=self.root)
        self._validate_directory_chain(self.artifacts.root, containment_root=self.root)

    def _validate_no_interrupted_evidence_writes(self) -> None:
        for root, pattern in (
            (self.records, ".private-trajectory-*"),
            (self.generation_records, ".private-generation-*"),
            (self.generation_starts, ".private-generation-start-*"),
            (self.generation_intents, ".private-generation-intent-*"),
        ):
            if any(root.glob(pattern)):
                raise VariationCheckpointError("private evidence contains an interrupted ambiguous write")

    def _put_artifact(self, data: bytes, *, media_type: str, role: str):
        self._validate_store_layout()
        digest = digest_bytes(data)
        relative = Path("blobs") / "sha256" / digest[:2] / digest[2:4]
        self._ensure_directory(self.artifacts.root / relative, containment_root=self.artifacts.root)
        ref = self.artifacts.put(data, media_type=media_type, role=role)
        self._read_regular_file(self.artifacts.root / ref.relative_path, containment_root=self.artifacts.root)
        return ref

    def _read_artifact(self, digest: str) -> bytes:
        self._validate_store_layout()
        path = self.artifacts.root / "blobs" / "sha256" / digest[:2] / digest[2:4] / digest
        self._read_regular_file(path, containment_root=self.artifacts.root)
        return self.artifacts.read(digest)

    @staticmethod
    def _validate_candidate_id(candidate_id: str) -> None:
        if not isinstance(candidate_id, str) or not candidate_id or "/" in candidate_id or "\\" in candidate_id:
            raise VariationCheckpointError("private trajectory candidate ID is invalid")

    def _write_once(
        self,
        path: Path,
        value: Mapping[str, Any],
        *,
        prefix: str,
        containment_root: Path,
    ) -> None:
        encoded = canonical_bytes(value)
        if path.exists():
            if self._read_regular_file(path, containment_root=containment_root) != encoded:
                raise VariationCheckpointError("private evidence conflicts with durable content")
            return
        self._validate_directory(path.parent, containment_root=containment_root)
        descriptor, name = tempfile.mkstemp(prefix=prefix, dir=str(path.parent))
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o400)
            os.replace(str(temporary), str(path))
            self._read_regular_file(path, containment_root=containment_root)
        finally:
            if temporary.exists():
                temporary.unlink()

    def record(self, *, candidate_id: str, context: CandidateContext, source: bytes) -> None:
        self._validate_store_layout()
        self._validate_candidate_id(candidate_id)
        if type(context) is not CandidateContext or not isinstance(source, bytes) or not source:
            raise VariationCheckpointError("private trajectory sidecar input is not exact")
        ref = self._put_artifact(source, media_type="text/x-python", role="private-candidate-source")
        value = {
            "schema_version": PRIVATE_TRAJECTORY_SCHEMA,
            "candidate_id": candidate_id,
            "context": _context_dict(context),
            "source_digest": ref.digest,
        }
        path = self.records / (candidate_id + ".json")
        self._write_once(
            path,
            value,
            prefix=".private-trajectory-",
            containment_root=self.records,
        )

    def _put_optional(self, value: Any, *, role: str) -> Any:
        if value is None:
            return None
        return self._put_artifact(value, media_type="application/octet-stream", role=role).digest

    @staticmethod
    def _encode_intent_bytes(value: Any, field: str) -> Any:
        if value is None:
            return None
        if not isinstance(value, bytes):
            raise VariationCheckpointError("private generation intent {} is not bytes".format(field))
        return base64.b64encode(value).decode("ascii")

    @staticmethod
    def _decode_intent_bytes(value: Any, field: str) -> Optional[bytes]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise VariationCheckpointError("private generation intent {} is not encoded text".format(field))
        try:
            decoded = base64.b64decode(value.encode("ascii"), validate=True)
        except (UnicodeError, ValueError) as exc:
            raise VariationCheckpointError(
                "private generation intent {} is not canonical base64".format(field)
            ) from exc
        if base64.b64encode(decoded).decode("ascii") != value:
            raise VariationCheckpointError(
                "private generation intent {} is not canonical base64".format(field)
            )
        return decoded

    def record_generation_start(self, *, candidate_id: str, context: CandidateContext) -> str:
        """Persist the exact generation cursor before the non-idempotent model call."""

        self._validate_store_layout()
        self._validate_candidate_id(candidate_id)
        if type(context) is not CandidateContext:
            raise VariationCheckpointError("private generation start context is not exact")
        value = {
            "schema_version": PRIVATE_GENERATION_START_SCHEMA,
            "candidate_id": candidate_id,
            "context": _context_dict(context),
        }
        path = self.generation_starts / (candidate_id + ".json")
        self._write_once(
            path,
            value,
            prefix=".private-generation-start-",
            containment_root=self.generation_starts,
        )
        return digest_bytes(self._read_regular_file(path, containment_root=self.generation_starts))

    def _load_generation_start(self, candidate_id: str) -> Tuple[CandidateContext, str]:
        self._validate_candidate_id(candidate_id)
        path = self.generation_starts / (candidate_id + ".json")
        try:
            raw = self._read_regular_file(path, containment_root=self.generation_starts)
            value = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise VariationCheckpointError("private generation start is missing or unreadable") from exc
        if (
            canonical_bytes(value) != raw
            or set(value) != _PRIVATE_GENERATION_START_FIELDS
            or value.get("schema_version") != PRIVATE_GENERATION_START_SCHEMA
            or value.get("candidate_id") != candidate_id
            or not isinstance(value.get("context"), Mapping)
        ):
            raise VariationCheckpointError("private generation start is not canonical and closed")
        return _context_from_mapping(value["context"]), digest_bytes(raw)

    def _load_generation_intent(
        self,
        candidate_id: str,
    ) -> Tuple[Mapping[str, Any], CandidateContext, Mapping[str, Optional[bytes]]]:
        self._validate_candidate_id(candidate_id)
        path = self.generation_intents / (candidate_id + ".json")
        try:
            raw = self._read_regular_file(path, containment_root=self.generation_intents)
            value = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise VariationCheckpointError("private generation intent is missing or unreadable") from exc
        if (
            canonical_bytes(value) != raw
            or set(value) != _PRIVATE_GENERATION_INTENT_FIELDS
            or value.get("schema_version") != PRIVATE_GENERATION_INTENT_SCHEMA
            or value.get("candidate_id") != candidate_id
        ):
            raise VariationCheckpointError("private generation intent is not canonical and closed")
        context_value = value.get("context")
        if not isinstance(context_value, Mapping):
            raise VariationCheckpointError("private generation intent lacks its exact context")
        context = _context_from_mapping(context_value)
        start_context, _start_digest = self._load_generation_start(candidate_id)
        if start_context != context:
            raise VariationCheckpointError("private generation intent crossed its start context")
        if (
            value.get("response_contract") != context.response_contract
            or value.get("response_contract_digest") != context.response_contract_digest
            or value.get("generation_profile_digest") != context.generation_profile_digest
        ):
            raise VariationCheckpointError("private generation intent differs from its frozen context")
        raw_values = {
            "rendered_prompt": self._decode_intent_bytes(
                value.get("rendered_prompt_b64"), "rendered prompt"
            ),
            "decoded_model_response": self._decode_intent_bytes(
                value.get("decoded_model_response_b64"), "decoded response"
            ),
            "contract_response": self._decode_intent_bytes(
                value.get("contract_response_b64"), "contract response"
            ),
        }
        status = value.get("status")
        if status == "SUCCESS":
            if (
                value.get("failure_stage") is not None
                or value.get("error_code") is not None
                or not isinstance(value.get("proposal_source_digest"), str)
                or any(item is None for item in raw_values.values())
            ):
                raise VariationCheckpointError("private generation success intent is inconsistent")
        elif status == "FAILED":
            if (
                not isinstance(value.get("failure_stage"), str)
                or not isinstance(value.get("error_code"), str)
                or value.get("proposal_source_digest") is not None
            ):
                raise VariationCheckpointError("private generation failure intent is inconsistent")
        else:
            raise VariationCheckpointError("private generation intent status is outside the closed vocabulary")
        rendered = raw_values["rendered_prompt"]
        if rendered is not None and digest_bytes(rendered) != context.prompt_digest:
            raise VariationCheckpointError("private generation intent prompt differs from its context")
        return dict(value), context, raw_values

    def _complete_generation_intent(self, candidate_id: str) -> str:
        value, _context, raw_values = self._load_generation_intent(candidate_id)
        record = {
            "schema_version": PRIVATE_GENERATION_SCHEMA,
            "candidate_id": candidate_id,
            "status": value["status"],
            "context": value["context"],
            "response_contract": value["response_contract"],
            "response_contract_digest": value["response_contract_digest"],
            "generation_profile_digest": value["generation_profile_digest"],
            "rendered_prompt_digest": self._put_optional(
                raw_values["rendered_prompt"], role="private-rendered-prompt"
            ),
            "decoded_model_response_digest": self._put_optional(
                raw_values["decoded_model_response"], role="private-decoded-model-response"
            ),
            "contract_response_digest": self._put_optional(
                raw_values["contract_response"], role="private-contract-response"
            ),
            "proposal_source_digest": value["proposal_source_digest"],
            "failure_stage": value["failure_stage"],
            "error_code": value["error_code"],
        }
        path = self.generation_records / (candidate_id + ".json")
        if path.exists():
            try:
                existing_raw = self._read_regular_file(
                    path, containment_root=self.generation_records
                )
                existing = json.loads(existing_raw.decode("utf-8"))
            except (OSError, UnicodeError, ValueError) as exc:
                raise VariationCheckpointError(
                    "private generation record is unreadable during intent reconciliation"
                ) from exc
            if existing.get("proposal_source_digest") != record["proposal_source_digest"]:
                raise VariationCheckpointError(
                    "private generation proposal differs from its durable intent"
                )
        self._write_once(
            path,
            record,
            prefix=".private-generation-",
            containment_root=self.generation_records,
        )
        return digest_bytes(self._read_regular_file(path, containment_root=self.generation_records))

    def _reconcile_generation_intents(self) -> None:
        """Finish every intent, then require an exact closed artifact inventory."""

        self._validate_store_layout()
        self._validate_no_interrupted_evidence_writes()
        intent_ids = set()
        for path in sorted(self.generation_intents.glob("*.json")):
            candidate_id = path.stem
            self._load_generation_intent(candidate_id)
            self._complete_generation_intent(candidate_id)
            intent_ids.add(candidate_id)
        start_ids = set()
        for path in sorted(self.generation_starts.glob("*.json")):
            candidate_id = path.stem
            self._load_generation_start(candidate_id)
            start_ids.add(candidate_id)
        record_ids = {path.stem for path in self.generation_records.glob("*.json")}
        if record_ids != intent_ids or start_ids != intent_ids:
            raise VariationCheckpointError(
                "private generation starts, intents, and records are incomplete or ambiguous"
            )
        referenced = set()
        optional_referenced = set()
        for candidate_id in sorted(intent_ids):
            intent_value, _context, raw_values = self._load_generation_intent(candidate_id)
            referenced.update(digest_bytes(item) for item in raw_values.values() if item is not None)
            proposal_digest = intent_value.get("proposal_source_digest")
            if isinstance(proposal_digest, str):
                optional_referenced.add(proposal_digest)
        for path in sorted(self.records.glob("*.json")):
            try:
                raw = self._read_regular_file(path, containment_root=self.records)
                value = json.loads(raw.decode("utf-8"))
            except (OSError, UnicodeError, ValueError) as exc:
                raise VariationCheckpointError("private trajectory inventory is unreadable") from exc
            if (
                canonical_bytes(value) != raw
                or set(value) != {"schema_version", "candidate_id", "context", "source_digest"}
                or value.get("candidate_id") != path.stem
                or value.get("schema_version")
                not in {PRIVATE_TRAJECTORY_SCHEMA, LEGACY_PRIVATE_TRAJECTORY_SCHEMA}
                or not isinstance(value.get("source_digest"), str)
            ):
                raise VariationCheckpointError("private trajectory inventory is not canonical and closed")
            referenced.add(value["source_digest"])
        for path in sorted(self.artifacts.root.rglob("*.tmp")):
            digest = path.name[:-4]
            target = path.with_name(digest)
            if (
                digest not in referenced | optional_referenced
                or target.exists()
                or digest_bytes(self._read_regular_file(path, containment_root=self.artifacts.root)) != digest
            ):
                raise VariationCheckpointError("private artifact store contains an ambiguous interrupted write")
            os.replace(str(path), str(target))
            target.chmod(0o444)
        actual = set()
        for path in sorted(self.artifacts.root.rglob("*")):
            if not path.is_file():
                continue
            if path.name.endswith(".tmp"):
                raise VariationCheckpointError("private artifact store contains an interrupted write")
            digest = path.name
            expected = self.artifacts.root / "blobs" / "sha256" / digest[:2] / digest[2:4] / digest
            if (
                len(digest) != 64
                or digest != digest.lower()
                or path.resolve() != expected.resolve()
                or digest_bytes(self._read_regular_file(path, containment_root=self.artifacts.root)) != digest
            ):
                raise VariationCheckpointError("private artifact inventory contains an invalid object")
            try:
                int(digest, 16)
            except ValueError as exc:
                raise VariationCheckpointError("private artifact inventory digest is invalid") from exc
            actual.add(digest)
        if not referenced.issubset(actual) or not actual.issubset(referenced | optional_referenced):
            raise VariationCheckpointError("private artifact inventory contains missing or orphaned evidence")

    def _record_generation(
        self,
        *,
        candidate_id: str,
        context: CandidateContext,
        status: str,
        rendered_prompt: Any,
        decoded_model_response: Any,
        contract_response: Any,
        failure_stage: Any,
        error_code: Any,
        proposal_source_digest: Any,
    ) -> str:
        self._validate_store_layout()
        self._validate_candidate_id(candidate_id)
        start_context, _start_digest = self._load_generation_start(candidate_id)
        if start_context != context:
            raise VariationCheckpointError("private generation evidence crossed its durable start")
        if context.response_contract != "closed-json-v1":
            profile = context.generation_profile_digest
            if not isinstance(profile, str) or len(profile) != 64:
                raise VariationCheckpointError("private generation context lacks its generation profile")
            try:
                int(profile, 16)
            except ValueError as exc:
                raise VariationCheckpointError("private generation profile digest is not hexadecimal") from exc
        if status == "SUCCESS":
            if failure_stage is not None or error_code is not None or proposal_source_digest is None:
                raise VariationCheckpointError("private generation success fields are inconsistent")
        elif status == "FAILED":
            if not isinstance(failure_stage, str) or not isinstance(error_code, str) or proposal_source_digest is not None:
                raise VariationCheckpointError("private generation failure fields are inconsistent")
        else:
            raise VariationCheckpointError("private generation status is outside the closed vocabulary")
        intent = {
            "schema_version": PRIVATE_GENERATION_INTENT_SCHEMA,
            "candidate_id": candidate_id,
            "status": status,
            "context": _context_dict(context),
            "response_contract": context.response_contract,
            "response_contract_digest": context.response_contract_digest,
            "generation_profile_digest": context.generation_profile_digest,
            "rendered_prompt_b64": self._encode_intent_bytes(rendered_prompt, "rendered prompt"),
            "decoded_model_response_b64": self._encode_intent_bytes(
                decoded_model_response, "decoded response"
            ),
            "contract_response_b64": self._encode_intent_bytes(
                contract_response, "contract response"
            ),
            "proposal_source_digest": proposal_source_digest,
            "failure_stage": failure_stage,
            "error_code": error_code,
        }
        path = self.generation_intents / (candidate_id + ".json")
        self._write_once(
            path,
            intent,
            prefix=".private-generation-intent-",
            containment_root=self.generation_intents,
        )
        return self._complete_generation_intent(candidate_id)

    def record_generation_success(
        self,
        *,
        candidate_id: str,
        context: CandidateContext,
        evidence: CandidateGenerationEvidence,
    ) -> str:
        if type(evidence) is not CandidateGenerationEvidence:
            raise VariationCheckpointError("private generation success evidence type is invalid")
        evidence.validate(context)
        return self._record_generation(
            candidate_id=candidate_id,
            context=context,
            status="SUCCESS",
            rendered_prompt=evidence.rendered_prompt,
            decoded_model_response=evidence.decoded_model_response,
            contract_response=evidence.contract_response,
            failure_stage=None,
            error_code=None,
            proposal_source_digest=digest_bytes(evidence.proposal.source),
        )

    def record_generation_failure(
        self,
        *,
        candidate_id: str,
        context: CandidateContext,
        evidence: CandidateGenerationFailureEvidence,
    ) -> str:
        if type(evidence) is not CandidateGenerationFailureEvidence:
            raise VariationCheckpointError("private generation failure evidence type is invalid")
        evidence.validate(context)
        return self._record_generation(
            candidate_id=candidate_id,
            context=context,
            status="FAILED",
            rendered_prompt=evidence.rendered_prompt,
            decoded_model_response=evidence.decoded_model_response,
            contract_response=evidence.contract_response,
            failure_stage=evidence.stage,
            error_code=evidence.error_code,
            proposal_source_digest=None,
        )

    def generation_record_digest(self, candidate_id: str) -> str:
        self._reconcile_generation_intents()
        self._validate_store_layout()
        self._validate_candidate_id(candidate_id)
        path = self.generation_records / (candidate_id + ".json")
        try:
            raw = self._read_regular_file(path, containment_root=self.generation_records)
            value = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise VariationCheckpointError("private generation evidence is missing or unreadable") from exc
        if (
            canonical_bytes(value) != raw
            or set(value) != _PRIVATE_GENERATION_FIELDS
            or value.get("schema_version") != PRIVATE_GENERATION_SCHEMA
            or value.get("candidate_id") != candidate_id
        ):
            raise VariationCheckpointError("private generation evidence is not canonical or content-bound")
        return digest_bytes(raw)

    def has_generation_record(self, candidate_id: str) -> bool:
        """Return whether an exact private generation record exists.

        A link, hardlink, or non-regular object is evidence corruption rather
        than an absent record and therefore fails closed.
        """

        self._validate_store_layout()
        self._validate_candidate_id(candidate_id)
        path = self.generation_records / (candidate_id + ".json")
        try:
            os.lstat(path)
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise VariationCheckpointError("private generation evidence metadata is unreadable") from exc
        self._read_regular_file(path, containment_root=self.generation_records)
        return True

    def load_generation_success(
        self, candidate_id: str
    ) -> Tuple[CandidateContext, CandidateGenerationEvidence, str]:
        """Reconstruct one successful proposal from immutable raw evidence.

        This is the crash-recovery authority for the narrow boundary after
        model generation and before a Variation attempt checkpoint.  The
        proposal is reparsed from the retained contract-response bytes; no
        model generation is repeated and no proposal fields are guessed from
        receipts or mutable process state.
        """

        self._reconcile_generation_intents()
        self._validate_store_layout()
        self._validate_candidate_id(candidate_id)
        path = self.generation_records / (candidate_id + ".json")
        try:
            raw = self._read_regular_file(path, containment_root=self.generation_records)
            value = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise VariationCheckpointError("private generation evidence is missing or unreadable") from exc
        if (
            canonical_bytes(value) != raw
            or set(value) != _PRIVATE_GENERATION_FIELDS
            or value.get("schema_version") != PRIVATE_GENERATION_SCHEMA
            or value.get("candidate_id") != candidate_id
            or value.get("status") != "SUCCESS"
            or value.get("failure_stage") is not None
            or value.get("error_code") is not None
        ):
            raise VariationCheckpointError("private generation success is not canonical and closed")
        context_value = value.get("context")
        if not isinstance(context_value, Mapping):
            raise VariationCheckpointError("private generation success lacks its exact context")
        context = _context_from_mapping(context_value)
        if (
            context.response_contract == "closed-json-v1"
            or value.get("response_contract") != context.response_contract
            or value.get("response_contract_digest") != context.response_contract_digest
            or value.get("generation_profile_digest") != context.generation_profile_digest
            or value.get("rendered_prompt_digest") != context.prompt_digest
        ):
            raise VariationCheckpointError("private generation success differs from its frozen context")

        artifacts: Dict[str, bytes] = {}
        for field in (
            "rendered_prompt_digest",
            "decoded_model_response_digest",
            "contract_response_digest",
        ):
            artifact_digest = value.get(field)
            if not isinstance(artifact_digest, str):
                raise VariationCheckpointError("private generation success lacks raw evidence")
            artifact = self._read_artifact(artifact_digest)
            if digest_bytes(artifact) != artifact_digest:
                raise VariationCheckpointError("private generation success raw evidence is corrupt")
            artifacts[field] = artifact
        try:
            contract_text = artifacts["contract_response_digest"].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise VariationCheckpointError("private generation contract response is not UTF-8") from exc
        try:
            proposal = ModelCandidateGenerator._parse_response(
                contract_text,
                context,
                response_contract=context.response_contract,
            )
            evidence = CandidateGenerationEvidence(
                proposal=proposal,
                decoded_model_response=artifacts["decoded_model_response_digest"],
                decoded_model_response_digest=str(value["decoded_model_response_digest"]),
                contract_response=artifacts["contract_response_digest"],
                contract_response_digest=str(value["contract_response_digest"]),
                rendered_prompt=artifacts["rendered_prompt_digest"],
                rendered_prompt_digest=str(value["rendered_prompt_digest"]),
                response_contract=context.response_contract,
            )
            evidence.validate(context)
        except Exception as exc:
            if isinstance(exc, VariationCheckpointError):
                raise
            raise VariationCheckpointError("private generation success cannot reconstruct its exact proposal") from exc
        if value.get("proposal_source_digest") != digest_bytes(proposal.source):
            raise VariationCheckpointError("private generation proposal differs from its durable source digest")
        return context, evidence, digest_bytes(raw)

    def load_generation_failure(
        self, candidate_id: str
    ) -> Tuple[CandidateContext, CandidateGenerationFailureEvidence, str]:
        """Load and revalidate one exact failed generation record."""

        self._reconcile_generation_intents()
        self._validate_store_layout()
        self._validate_candidate_id(candidate_id)
        path = self.generation_records / (candidate_id + ".json")
        try:
            raw = self._read_regular_file(path, containment_root=self.generation_records)
            value = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise VariationCheckpointError("private generation failure is missing or unreadable") from exc
        if (
            canonical_bytes(value) != raw
            or set(value) != _PRIVATE_GENERATION_FIELDS
            or value.get("schema_version") != PRIVATE_GENERATION_SCHEMA
            or value.get("candidate_id") != candidate_id
            or value.get("status") != "FAILED"
            or value.get("proposal_source_digest") is not None
        ):
            raise VariationCheckpointError("private generation failure is not canonical and closed")
        context_value = value.get("context")
        if not isinstance(context_value, Mapping):
            raise VariationCheckpointError("private generation failure lacks its exact context")
        context = _context_from_mapping(context_value)

        def optional_artifact(field: str) -> Optional[bytes]:
            artifact_digest = value.get(field)
            if artifact_digest is None:
                return None
            if not isinstance(artifact_digest, str):
                raise VariationCheckpointError("private generation failure artifact digest is invalid")
            artifact = self._read_artifact(artifact_digest)
            if digest_bytes(artifact) != artifact_digest:
                raise VariationCheckpointError("private generation failure raw evidence is corrupt")
            return artifact

        rendered_prompt = optional_artifact("rendered_prompt_digest")
        decoded_response = optional_artifact("decoded_model_response_digest")
        contract_response = optional_artifact("contract_response_digest")
        try:
            evidence = CandidateGenerationFailureEvidence(
                stage=value.get("failure_stage"),
                response_contract=value.get("response_contract"),
                rendered_prompt=rendered_prompt,
                rendered_prompt_digest=value.get("rendered_prompt_digest"),
                decoded_model_response=decoded_response,
                decoded_model_response_digest=value.get("decoded_model_response_digest"),
                contract_response=contract_response,
                contract_response_digest=value.get("contract_response_digest"),
                error_code=value.get("error_code"),
            )
            evidence.validate(context)
        except Exception as exc:
            raise VariationCheckpointError("private generation failure cannot revalidate its raw evidence") from exc
        if (
            value.get("response_contract") != context.response_contract
            or value.get("response_contract_digest") != context.response_contract_digest
            or value.get("generation_profile_digest") != context.generation_profile_digest
        ):
            raise VariationCheckpointError("private generation failure differs from its frozen context")
        return context, evidence, digest_bytes(raw)

    def successful_generations(
        self,
        *,
        run_id: str,
        task_id: str,
        arm_id: str,
    ) -> Tuple[Tuple[str, CandidateContext, CandidateGenerationEvidence, str], ...]:
        """Load every successful generation bound to one trajectory."""

        self._reconcile_generation_intents()
        self._validate_store_layout()
        self._validate_no_interrupted_evidence_writes()
        found = []
        self._validate_directory(self.generation_records, containment_root=self.root)
        for path in sorted(self.generation_records.glob("*.json")):
            try:
                raw = self._read_regular_file(path, containment_root=self.generation_records)
                value = json.loads(raw.decode("utf-8"))
            except (OSError, UnicodeError, ValueError) as exc:
                raise VariationCheckpointError("private generation evidence is unreadable") from exc
            if (
                canonical_bytes(value) != raw
                or set(value) != _PRIVATE_GENERATION_FIELDS
                or value.get("schema_version") != PRIVATE_GENERATION_SCHEMA
            ):
                raise VariationCheckpointError("private generation evidence is not canonical and closed")
            context_value = value.get("context")
            if not isinstance(context_value, Mapping):
                raise VariationCheckpointError("private generation evidence lacks its exact context")
            context = _context_from_mapping(context_value)
            if context.run_id != run_id:
                continue
            if context.task_id != task_id:
                raise VariationCheckpointError("private generation evidence crossed its task boundary")
            candidate_id = value.get("candidate_id")
            if context.arm_id != arm_id or candidate_id != path.stem:
                raise VariationCheckpointError("private generation evidence crossed its run/arm boundary")
            if value.get("status") not in {"SUCCESS", "FAILED"}:
                raise VariationCheckpointError("private generation evidence has an unknown durable status")
            if value.get("status") != "SUCCESS":
                continue
            loaded_context, evidence, record_digest = self.load_generation_success(str(candidate_id))
            if loaded_context != context:
                raise VariationCheckpointError("private generation context changed while loading")
            found.append((str(candidate_id), context, evidence, record_digest))
        found.sort(key=lambda item: item[1].attempt_index)
        indices = [item[1].attempt_index for item in found]
        if len(indices) != len(set(indices)):
            raise VariationCheckpointError("successful private generation attempt indices are duplicated")
        return tuple(found)

    def source_contract_failures(
        self,
        *,
        run_id: str,
        task_id: str,
        arm_id: str,
    ) -> Tuple[Mapping[str, Any], ...]:
        """Load immutable source-contract failures for exact bounded resume."""

        self._reconcile_generation_intents()
        self._validate_store_layout()
        self._validate_no_interrupted_evidence_writes()
        failures = []
        self._validate_directory(self.generation_records, containment_root=self.root)
        for path in sorted(self.generation_records.glob("*.json")):
            try:
                raw = self._read_regular_file(path, containment_root=self.generation_records)
                value = json.loads(raw.decode("utf-8"))
            except (OSError, UnicodeError, ValueError) as exc:
                raise VariationCheckpointError("private generation evidence is unreadable") from exc
            if (
                canonical_bytes(value) != raw
                or set(value) != _PRIVATE_GENERATION_FIELDS
                or value.get("schema_version") != PRIVATE_GENERATION_SCHEMA
            ):
                raise VariationCheckpointError("private generation evidence is not canonical and closed")
            context_value = value.get("context")
            if not isinstance(context_value, Mapping):
                raise VariationCheckpointError("private generation evidence lacks its exact context")
            context = _context_from_mapping(context_value)
            if context.run_id != run_id:
                continue
            if context.task_id != task_id:
                raise VariationCheckpointError("private generation evidence crossed its task boundary")
            if context.arm_id != arm_id or value.get("candidate_id") != path.stem:
                raise VariationCheckpointError("private generation evidence crossed its run/arm boundary")
            if value.get("status") not in {"SUCCESS", "FAILED"}:
                raise VariationCheckpointError("private generation evidence has an unknown durable status")
            if value.get("status") != "FAILED":
                continue
            if value.get("failure_stage") != "RESPONSE_CONTRACT":
                raise VariationCheckpointError("fatal model generation failure prevents automatic resume")
            if (
                value.get("rendered_prompt_digest") != context.prompt_digest
                or value.get("response_contract") != context.response_contract
                or value.get("response_contract_digest") != context.response_contract_digest
                or value.get("generation_profile_digest") != context.generation_profile_digest
                or value.get("proposal_source_digest") is not None
            ):
                raise VariationCheckpointError("source-contract failure binding is invalid")
            for field in (
                "rendered_prompt_digest", "decoded_model_response_digest", "contract_response_digest"
            ):
                artifact_digest = value.get(field)
                if not isinstance(artifact_digest, str) or digest_bytes(self._read_artifact(artifact_digest)) != artifact_digest:
                    raise VariationCheckpointError("source-contract failure raw evidence is missing or corrupt")
            failures.append({
                "attempt_index": context.attempt_index,
                "candidate_id": value["candidate_id"],
                "parent_candidate_id": context.parent_candidate_id,
                "record_digest": digest_bytes(raw),
                "prompt_digest": context.prompt_digest,
            })
        failures.sort(key=lambda item: item["attempt_index"])
        indices = [item["attempt_index"] for item in failures]
        if len(indices) != len(set(indices)):
            raise VariationCheckpointError("source-contract failure attempt indices are duplicated")
        return tuple(failures)

    def load_attempts(self, candidate_ids: Iterable[str]) -> Tuple[object, ...]:
        from ..training.contracts import PrivateTrajectoryAttempt

        self._reconcile_generation_intents()
        self._validate_store_layout()
        attempts = []
        for candidate_id in sorted(set(candidate_ids)):
            self._validate_candidate_id(candidate_id)
            path = self.records / (candidate_id + ".json")
            try:
                raw = self._read_regular_file(path, containment_root=self.records)
                value = json.loads(raw.decode("utf-8"))
            except (OSError, UnicodeError, ValueError) as exc:
                raise VariationCheckpointError("private trajectory sidecar is missing or unreadable") from exc
            if canonical_bytes(value) != raw or set(value) != {
                "schema_version", "candidate_id", "context", "source_digest"
            }:
                raise VariationCheckpointError("private trajectory sidecar is not canonical and closed")
            if value["schema_version"] not in {PRIVATE_TRAJECTORY_SCHEMA, LEGACY_PRIVATE_TRAJECTORY_SCHEMA} or value["candidate_id"] != candidate_id:
                raise VariationCheckpointError("private trajectory sidecar identity is invalid")
            context = _context_from_mapping(value["context"])
            rendered_prompt = None
            if context.response_contract != "closed-json-v1":
                generation_path = self.generation_records / (candidate_id + ".json")
                try:
                    generation_raw = self._read_regular_file(
                        generation_path,
                        containment_root=self.generation_records,
                    )
                    generation = json.loads(generation_raw.decode("utf-8"))
                except (OSError, UnicodeError, ValueError) as exc:
                    raise VariationCheckpointError("source-only trajectory lacks generation evidence") from exc
                if (
                    canonical_bytes(generation) != generation_raw
                    or set(generation) != _PRIVATE_GENERATION_FIELDS
                    or generation.get("schema_version") != PRIVATE_GENERATION_SCHEMA
                    or generation.get("candidate_id") != candidate_id
                    or generation.get("status") != "SUCCESS"
                    or generation.get("context") != _context_dict(context)
                    or generation.get("response_contract") != context.response_contract
                    or generation.get("response_contract_digest") != context.response_contract_digest
                    or generation.get("rendered_prompt_digest") != context.prompt_digest
                    or generation.get("generation_profile_digest") != context.generation_profile_digest
                    or generation.get("failure_stage") is not None
                    or generation.get("error_code") is not None
                ):
                    raise VariationCheckpointError("source-only generation evidence differs from trajectory")
                for field in (
                    "rendered_prompt_digest", "decoded_model_response_digest", "contract_response_digest"
                ):
                    artifact_digest = generation.get(field)
                    if not isinstance(artifact_digest, str) or digest_bytes(self._read_artifact(artifact_digest)) != artifact_digest:
                        raise VariationCheckpointError("source-only generation raw evidence is missing or corrupt")
                rendered_prompt = self._read_artifact(generation["rendered_prompt_digest"])
                generation_evidence_digest = digest_bytes(generation_raw)
            else:
                generation_evidence_digest = None
            source = self._read_artifact(value["source_digest"])
            if digest_bytes(source) != value["source_digest"]:
                raise VariationCheckpointError("private trajectory source digest is invalid")
            if context.response_contract != "closed-json-v1" and (
                generation.get("proposal_source_digest") != value["source_digest"]
            ):
                raise VariationCheckpointError("source-only generation proposal differs from the trajectory source")
            attempts.append(
                PrivateTrajectoryAttempt(
                    candidate_id,
                    context,
                    source,
                    rendered_prompt,
                    generation_evidence_digest,
                )
            )
        return tuple(attempts)


__all__ = [
    "LEGACY_PRIVATE_TRAJECTORY_SCHEMA", "PRIVATE_GENERATION_SCHEMA",
    "PRIVATE_TRAJECTORY_SCHEMA", "PrivateTrajectoryStore",
]
