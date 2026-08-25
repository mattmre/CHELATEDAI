"""Durable evaluator-private trajectory sidecars for training freeze."""

from __future__ import annotations

import base64
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import stat
import sys
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
PRIVATE_GENERATION_SCHEMA = "egv-private-generation-evidence-v2"
PRIVATE_GENERATION_START_SCHEMA = "egv-private-generation-start-v1"
PRIVATE_GENERATION_INTENT_SCHEMA = "egv-private-generation-intent-v2"
LEGACY_PRIVATE_GENERATION_INTENT_SCHEMA = "egv-private-generation-intent-legacy-migration-v2"
LEGACY_ORPHAN_ARTIFACTS_SCHEMA = "egv-private-legacy-orphan-artifacts-v1"
_PRIVATE_GENERATION_FIELDS = frozenset({
    "schema_version", "candidate_id", "status", "context", "response_contract",
    "response_contract_digest", "generation_profile_digest", "rendered_prompt_digest",
    "decoded_model_response_digest", "contract_response_digest", "proposal_source_digest",
    "failure_stage", "error_code", "replay_error_chain",
})
_PRIVATE_GENERATION_INTENT_FIELDS = frozenset({
    "schema_version", "candidate_id", "status", "context", "response_contract",
    "response_contract_digest", "generation_profile_digest", "rendered_prompt_b64",
    "decoded_model_response_b64", "contract_response_b64", "proposal_source_digest",
    "failure_stage", "error_code", "replay_error_chain",
})
_LEGACY_PRIVATE_GENERATION_INTENT_FIELDS = frozenset(
    set(_PRIVATE_GENERATION_INTENT_FIELDS) | {"legacy_orphan_manifest_digest"}
)
_PRIVATE_GENERATION_START_FIELDS = frozenset({"schema_version", "candidate_id", "context"})
_REPLAY_EXCEPTION_CHAIN_LIMIT = 16


def replay_exception_chain_classification(
    error: BaseException,
    *,
    ambient_exception: Optional[BaseException] = None,
) -> Tuple[str, ...]:
    """Return one exact bounded outer-to-inner exception classification."""

    chain = []
    seen = set()
    cursor: Optional[BaseException] = error
    while cursor is not None:
        if id(cursor) in seen or len(chain) >= _REPLAY_EXCEPTION_CHAIN_LIMIT:
            raise VariationCheckpointError("response-contract replay exception chain is cyclic or too deep")
        seen.add(id(cursor))
        name = type(cursor).__name__
        if not name or len(name) > 128:
            raise VariationCheckpointError("response-contract replay exception type is invalid")
        chain.append(name)
        next_error = cursor.__cause__ if cursor.__cause__ is not None else cursor.__context__
        if next_error is ambient_exception:
            break
        cursor = next_error
    return tuple(chain)


def response_contract_replay_error_chain(
    context: CandidateContext,
    contract_response: bytes,
) -> Tuple[str, ...]:
    """Replay trainer-supplied contract bytes and classify the exact failure chain."""

    if not isinstance(contract_response, bytes) or not contract_response:
        raise VariationCheckpointError("response-contract replay lacks exact contract bytes")
    ambient_exception = sys.exc_info()[1]
    try:
        proposal = ModelCandidateGenerator._parse_response(
            contract_response.decode("utf-8"),
            context,
            response_contract=context.response_contract,
        )
        proposal.validate(context, source_limit=256 * 1024)
    except Exception as exc:
        return replay_exception_chain_classification(
            exc,
            ambient_exception=ambient_exception,
        )
    raise VariationCheckpointError("response-contract failure unexpectedly succeeds on replay")


def _generation_replay_error_chain(
    context: CandidateContext,
    failure_stage: Any,
    contract_response: Any,
) -> Optional[list[str]]:
    if failure_stage != "RESPONSE_CONTRACT":
        return None
    return list(response_contract_replay_error_chain(context, contract_response))


@dataclass(frozen=True)
class LegacyGenerationBundle:
    """Fully revalidated pre-write-ahead generation and trajectory evidence.

    This contract exists only to migrate the single preserved live orphan that
    predates generation starts/intents.  Ledger, candidate, artifact, and
    receipt authority remain the caller's responsibility before commit.
    """

    candidate_id: str
    context: CandidateContext
    evidence: CandidateGenerationEvidence
    generation_record_digest: str
    trajectory_record_digest: str
    orphan_artifact_digests: Tuple[str, ...] = tuple()


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
        self.legacy_orphan_manifests = self.root / "legacy-orphan-artifacts"
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
        if self.legacy_orphan_manifests.exists() or self.legacy_orphan_manifests.is_symlink():
            self._validate_directory_chain(
                self.legacy_orphan_manifests,
                containment_root=self.root,
            )
        self._validate_directory_chain(self.artifacts.root, containment_root=self.root)

    def _validate_no_interrupted_evidence_writes(self) -> None:
        for root, pattern in (
            (self.records, ".private-trajectory-*"),
            (self.generation_records, ".private-generation-*"),
            (self.generation_starts, ".private-generation-start-*"),
            (self.generation_intents, ".private-generation-intent-*"),
            (self.legacy_orphan_manifests, ".legacy-orphan-artifacts-*"),
        ):
            if root.exists() and any(root.glob(pattern)):
                raise VariationCheckpointError("private evidence contains an interrupted ambiguous write")

    def _closed_json_inventory(self, root: Path, label: str) -> Tuple[Path, ...]:
        """Return an exact single-link JSON-file inventory for one evidence root."""

        self._validate_directory_chain(root, containment_root=self.root)
        try:
            entries = tuple(root.iterdir())
        except OSError as exc:
            raise VariationCheckpointError(
                "{} inventory is unreadable".format(label)
            ) from exc
        for path in entries:
            try:
                metadata = os.lstat(path)
            except OSError as exc:
                raise VariationCheckpointError(
                    "{} inventory metadata is unreadable".format(label)
                ) from exc
            if (
                self._link_like(metadata)
                or not stat.S_ISREG(metadata.st_mode)
                or int(getattr(metadata, "st_nlink", 0)) != 1
            ):
                raise VariationCheckpointError(
                    "{} inventory entry is linked or not regular".format(label)
                )
            if path.suffix != ".json":
                raise VariationCheckpointError(
                    "{} inventory contains an unexpected entry".format(label)
                )
        return tuple(sorted(entries))

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

    def load_trajectory_read_only(
        self, candidate_id: str
    ) -> Tuple[CandidateContext, bytes, str]:
        """Validate one terminal trajectory sidecar without repairing evidence."""

        self._validate_store_layout()
        self._validate_no_interrupted_evidence_writes()
        self._validate_candidate_id(candidate_id)
        path = self.records / (candidate_id + ".json")
        try:
            raw = self._read_regular_file(path, containment_root=self.records)
            value = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise VariationCheckpointError(
                "private trajectory sidecar is missing or unreadable"
            ) from exc
        if (
            canonical_bytes(value) != raw
            or set(value)
            != {"schema_version", "candidate_id", "context", "source_digest"}
            or value.get("schema_version")
            not in {PRIVATE_TRAJECTORY_SCHEMA, LEGACY_PRIVATE_TRAJECTORY_SCHEMA}
            or value.get("candidate_id") != candidate_id
            or not isinstance(value.get("context"), Mapping)
            or not isinstance(value.get("source_digest"), str)
        ):
            raise VariationCheckpointError(
                "private trajectory sidecar is not canonical and closed"
            )
        context = _context_from_mapping(value["context"])
        source = self._read_artifact(value["source_digest"])
        if digest_bytes(source) != value["source_digest"]:
            raise VariationCheckpointError("private trajectory source digest is invalid")
        return context, source, digest_bytes(raw)

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
        schema_version = value.get("schema_version")
        standard_intent = (
            schema_version == PRIVATE_GENERATION_INTENT_SCHEMA
            and set(value) == _PRIVATE_GENERATION_INTENT_FIELDS
        )
        legacy_intent = (
            schema_version == LEGACY_PRIVATE_GENERATION_INTENT_SCHEMA
            and set(value) == _LEGACY_PRIVATE_GENERATION_INTENT_FIELDS
        )
        if (
            canonical_bytes(value) != raw
            or not (standard_intent or legacy_intent)
            or value.get("candidate_id") != candidate_id
        ):
            raise VariationCheckpointError("private generation intent is not canonical and closed")
        if legacy_intent:
            manifest_digest = value.get("legacy_orphan_manifest_digest")
            try:
                manifest_raw = self._read_regular_file(
                    self.legacy_orphan_manifests / (candidate_id + ".json"),
                    containment_root=self.legacy_orphan_manifests,
                )
            except VariationCheckpointError as exc:
                raise VariationCheckpointError(
                    "legacy generation intent lacks its orphan-artifact manifest"
                ) from exc
            if (
                not isinstance(manifest_digest, str)
                or len(manifest_digest) != 64
                or manifest_digest != manifest_digest.lower()
                or digest_bytes(manifest_raw) != manifest_digest
            ):
                raise VariationCheckpointError(
                    "legacy generation intent orphan-artifact manifest binding is invalid"
                )
            try:
                int(manifest_digest, 16)
            except ValueError as exc:
                raise VariationCheckpointError(
                    "legacy generation intent orphan-artifact manifest digest is invalid"
                ) from exc
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
                or value.get("replay_error_chain") is not None
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
        stage = value.get("failure_stage")
        expected_presence = {
            ("SUCCESS", None): (True, True, True),
            ("FAILED", "PROMPT_RENDER"): (False, False, False),
            ("FAILED", "PROMPT_INTEGRITY"): (True, False, False),
            ("FAILED", "MODEL_GENERATION"): (True, False, False),
            ("FAILED", "RESPONSE_CONTRACT"): (True, True, True),
        }.get((status, stage))
        if tuple(item is not None for item in raw_values.values()) != expected_presence:
            raise VariationCheckpointError("private generation intent artifact shape differs from its stage")
        if rendered is not None:
            rendered_digest = digest_bytes(rendered)
            if stage == "PROMPT_INTEGRITY":
                if rendered_digest == context.prompt_digest:
                    raise VariationCheckpointError(
                        "private prompt-integrity intent does not preserve differing prompt bytes"
                    )
            elif rendered_digest != context.prompt_digest:
                raise VariationCheckpointError("private generation intent prompt differs from its context")
        expected_replay_error_chain = _generation_replay_error_chain(
            context,
            stage,
            raw_values["contract_response"],
        )
        if value.get("replay_error_chain") != expected_replay_error_chain:
            raise VariationCheckpointError(
                "private generation intent replay exception chain is not exact"
            )
        if expected_replay_error_chain is not None and value.get("error_code") != expected_replay_error_chain[0]:
            raise VariationCheckpointError(
                "private generation intent outer failure classification is not exact"
            )
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
            "replay_error_chain": value["replay_error_chain"],
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

    def _load_legacy_generation_bundle(self, candidate_id: str) -> LegacyGenerationBundle:
        """Revalidate a complete success record plus exact trajectory sidecar.

        The method deliberately does not write.  A caller with ledger access
        must independently validate the candidate/effect boundary before
        :meth:`commit_legacy_generation_bundle` is allowed to synthesize the
        missing write-ahead records.
        """

        self._validate_candidate_id(candidate_id)
        generation_path = self.generation_records / (candidate_id + ".json")
        trajectory_path = self.records / (candidate_id + ".json")
        try:
            generation_raw = self._read_regular_file(
                generation_path, containment_root=self.generation_records
            )
            generation = json.loads(generation_raw.decode("utf-8"))
            trajectory_raw = self._read_regular_file(
                trajectory_path, containment_root=self.records
            )
            trajectory = json.loads(trajectory_raw.decode("utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise VariationCheckpointError(
                "legacy generation migration lacks complete private evidence"
            ) from exc
        if (
            canonical_bytes(generation) != generation_raw
            or set(generation) != _PRIVATE_GENERATION_FIELDS
            or generation.get("schema_version") != PRIVATE_GENERATION_SCHEMA
            or generation.get("candidate_id") != candidate_id
            or generation.get("status") != "SUCCESS"
            or generation.get("failure_stage") is not None
            or generation.get("error_code") is not None
            or generation.get("replay_error_chain") is not None
            or canonical_bytes(trajectory) != trajectory_raw
            or set(trajectory) != {"schema_version", "candidate_id", "context", "source_digest"}
            or trajectory.get("schema_version")
            not in {PRIVATE_TRAJECTORY_SCHEMA, LEGACY_PRIVATE_TRAJECTORY_SCHEMA}
            or trajectory.get("candidate_id") != candidate_id
        ):
            raise VariationCheckpointError(
                "legacy generation migration evidence is not canonical and closed"
            )
        context_value = generation.get("context")
        if not isinstance(context_value, Mapping):
            raise VariationCheckpointError("legacy generation migration lacks its exact context")
        context = _context_from_mapping(context_value)
        if (
            context.response_contract == "closed-json-v1"
            or trajectory.get("context") != _context_dict(context)
            or generation.get("response_contract") != context.response_contract
            or generation.get("response_contract_digest") != context.response_contract_digest
            or generation.get("generation_profile_digest") != context.generation_profile_digest
            or generation.get("rendered_prompt_digest") != context.prompt_digest
        ):
            raise VariationCheckpointError(
                "legacy generation migration crossed its frozen context"
            )
        artifacts: Dict[str, bytes] = {}
        for field in (
            "rendered_prompt_digest",
            "decoded_model_response_digest",
            "contract_response_digest",
        ):
            artifact_digest = generation.get(field)
            if not isinstance(artifact_digest, str):
                raise VariationCheckpointError(
                    "legacy generation migration lacks exact raw evidence"
                )
            artifact = self._read_artifact(artifact_digest)
            if digest_bytes(artifact) != artifact_digest:
                raise VariationCheckpointError(
                    "legacy generation migration raw evidence is corrupt"
                )
            artifacts[field] = artifact
        try:
            contract_text = artifacts["contract_response_digest"].decode("utf-8")
            proposal = ModelCandidateGenerator._parse_response(
                contract_text,
                context,
                response_contract=context.response_contract,
            )
            evidence = CandidateGenerationEvidence(
                proposal=proposal,
                decoded_model_response=artifacts["decoded_model_response_digest"],
                decoded_model_response_digest=str(generation["decoded_model_response_digest"]),
                contract_response=artifacts["contract_response_digest"],
                contract_response_digest=str(generation["contract_response_digest"]),
                rendered_prompt=artifacts["rendered_prompt_digest"],
                rendered_prompt_digest=str(generation["rendered_prompt_digest"]),
                response_contract=context.response_contract,
            )
            evidence.validate(context)
        except Exception as exc:
            raise VariationCheckpointError(
                "legacy generation migration cannot reconstruct its exact proposal"
            ) from exc
        source_digest = digest_bytes(proposal.source)
        if (
            generation.get("proposal_source_digest") != source_digest
            or trajectory.get("source_digest") != source_digest
            or self._read_artifact(source_digest) != proposal.source
        ):
            raise VariationCheckpointError(
                "legacy generation migration proposal differs from its trajectory"
            )
        return LegacyGenerationBundle(
            candidate_id=candidate_id,
            context=context,
            evidence=evidence,
            generation_record_digest=digest_bytes(generation_raw),
            trajectory_record_digest=digest_bytes(trajectory_raw),
        )

    @staticmethod
    def _legacy_bundle_artifact_digests(bundle: LegacyGenerationBundle) -> Tuple[str, ...]:
        return tuple(sorted({
            digest_bytes(bundle.evidence.rendered_prompt),
            digest_bytes(bundle.evidence.decoded_model_response),
            digest_bytes(bundle.evidence.contract_response),
            digest_bytes(bundle.evidence.proposal.source),
        }))

    def _load_legacy_orphan_manifest(
        self,
        bundle: LegacyGenerationBundle,
    ) -> Mapping[str, Any]:
        path = self.legacy_orphan_manifests / (bundle.candidate_id + ".json")
        try:
            raw = self._read_regular_file(
                path,
                containment_root=self.legacy_orphan_manifests,
            )
            value = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise VariationCheckpointError(
                "legacy orphan-artifact manifest is missing or unreadable"
            ) from exc
        required = {
            "schema_version",
            "candidate_id",
            "generation_record_digest",
            "trajectory_record_digest",
            "referenced_artifact_digests",
            "orphan_artifact_digests",
            "replay_proof_digest",
        }
        proof_digest = value.get("replay_proof_digest")
        if (
            canonical_bytes(value) != raw
            or set(value) != required
            or value.get("schema_version") != LEGACY_ORPHAN_ARTIFACTS_SCHEMA
            or value.get("candidate_id") != bundle.candidate_id
            or value.get("generation_record_digest") != bundle.generation_record_digest
            or value.get("trajectory_record_digest") != bundle.trajectory_record_digest
            or value.get("referenced_artifact_digests")
            != list(self._legacy_bundle_artifact_digests(bundle))
            or value.get("orphan_artifact_digests")
            != list(bundle.orphan_artifact_digests)
            or not isinstance(proof_digest, str)
            or len(proof_digest) != 64
            or proof_digest != proof_digest.lower()
        ):
            raise VariationCheckpointError(
                "legacy orphan-artifact manifest is not canonical and content-bound"
            )
        try:
            int(proof_digest, 16)
        except ValueError as exc:
            raise VariationCheckpointError(
                "legacy orphan-artifact replay proof digest is invalid"
            ) from exc
        return value

    def legacy_generation_bundles(
        self,
        *,
        require_preserved_orphan: bool = False,
    ) -> Tuple[LegacyGenerationBundle, ...]:
        """Return only complete legacy records with no durable intent.

        A record with an intent but no start is never legacy.  A record with a
        start but no intent is accepted only as a resumable interruption of
        this migration and is revalidated from the immutable record/sidecar.
        """

        if type(require_preserved_orphan) is not bool:
            raise VariationCheckpointError(
                "legacy generation preserved-orphan requirement is invalid"
            )
        self._validate_store_layout()
        self._validate_no_interrupted_evidence_writes()
        def closed_json_ids(root: Path) -> set[str]:
            entries = list(root.iterdir())
            if any(not path.is_file() or path.suffix != ".json" for path in entries):
                raise VariationCheckpointError(
                    "private generation legacy inventory contains an unexpected entry"
                )
            return {path.stem for path in entries}

        record_ids = closed_json_ids(self.generation_records)
        start_ids = closed_json_ids(self.generation_starts)
        intent_ids = closed_json_ids(self.generation_intents)
        trajectory_ids = closed_json_ids(self.records)
        if intent_ids - start_ids:
            raise VariationCheckpointError(
                "private generation legacy inventory contains an impossible partial set"
            )
        legacy_ids = record_ids - intent_ids
        if legacy_ids and (
            intent_ids
            or not start_ids.issubset(legacy_ids)
            or trajectory_ids != legacy_ids
        ):
            raise VariationCheckpointError(
                "private generation legacy inventory is mixed with another generation state"
            )
        bundles = []
        for candidate_id in sorted(legacy_ids):
            bundle = self._load_legacy_generation_bundle(candidate_id)
            if candidate_id in start_ids:
                start_context, _start_digest = self._load_generation_start(candidate_id)
                if start_context != bundle.context:
                    raise VariationCheckpointError(
                        "legacy generation migration crossed its durable start"
                    )
            bundles.append(bundle)
        if bundles:
            if len(bundles) != 1:
                raise VariationCheckpointError(
                    "private generation legacy inventory contains multiple orphans"
                )
            referenced_artifacts = self._legacy_bundle_artifact_digests(bundles[0])
            expected_artifacts = set(referenced_artifacts)
            artifact_files = [
                path for path in self.artifacts.root.rglob("*") if path.is_file()
            ]
            actual_artifacts = {path.name for path in artifact_files}
            for path in artifact_files:
                digest = path.name
                expected_path = (
                    self.artifacts.root
                    / "blobs"
                    / "sha256"
                    / digest[:2]
                    / digest[2:4]
                    / digest
                )
                try:
                    valid_digest = (
                        len(digest) == 64
                        and digest == digest.lower()
                        and path.resolve(strict=True) == expected_path.resolve(strict=True)
                        and digest_bytes(
                            self._read_regular_file(path, containment_root=self.artifacts.root)
                        )
                        == digest
                    )
                    int(digest, 16)
                except (OSError, ValueError):
                    valid_digest = False
                if not valid_digest:
                    raise VariationCheckpointError(
                        "private generation legacy artifact inventory contains an invalid object"
                    )
            orphan_artifacts = tuple(sorted(actual_artifacts - expected_artifacts))
            if (
                len(artifact_files) != len(actual_artifacts)
                or not expected_artifacts.issubset(actual_artifacts)
                or len(orphan_artifacts) > 1
            ):
                raise VariationCheckpointError(
                    "private generation legacy artifact inventory exceeds the preserved boundary"
                )
            if require_preserved_orphan and (
                len(referenced_artifacts) != 3
                or len(orphan_artifacts) != 1
                or len(actual_artifacts) != 4
            ):
                raise VariationCheckpointError(
                    "commissioned legacy recovery requires exactly three distinct referenced "
                    "artifacts and one distinct orphan artifact"
                )
            bundles[0] = replace(
                bundles[0],
                orphan_artifact_digests=orphan_artifacts,
            )
            manifest_ids = (
                closed_json_ids(self.legacy_orphan_manifests)
                if self.legacy_orphan_manifests.exists()
                else set()
            )
            expected_manifest_ids = {bundles[0].candidate_id} if orphan_artifacts else set()
            if not manifest_ids.issubset(expected_manifest_ids):
                raise VariationCheckpointError(
                    "legacy orphan-artifact manifest inventory is not exact"
                )
            if (
                require_preserved_orphan
                and (start_ids or manifest_ids)
                and manifest_ids != expected_manifest_ids
            ):
                raise VariationCheckpointError(
                    "commissioned legacy recovery lacks its mandatory orphan-artifact manifest"
                )
            if manifest_ids:
                self._load_legacy_orphan_manifest(bundles[0])
        return tuple(bundles)

    def commit_legacy_generation_bundle(
        self,
        bundle: LegacyGenerationBundle,
        *,
        generation_record_digest: str,
        trajectory_record_digest: str,
        replay_proof_digest: str,
        require_preserved_orphan: bool = False,
    ) -> str:
        """Write deterministic start/intent records after external proof."""

        if type(bundle) is not LegacyGenerationBundle:
            raise VariationCheckpointError("legacy generation migration bundle type is invalid")
        current_bundles = self.legacy_generation_bundles(
            require_preserved_orphan=require_preserved_orphan,
        )
        if (
            current_bundles != (bundle,)
            or generation_record_digest != bundle.generation_record_digest
            or trajectory_record_digest != bundle.trajectory_record_digest
            or not isinstance(replay_proof_digest, str)
            or len(replay_proof_digest) != 64
            or replay_proof_digest != replay_proof_digest.lower()
        ):
            raise VariationCheckpointError(
                "legacy generation migration authority differs from current evidence"
            )
        try:
            int(replay_proof_digest, 16)
        except ValueError as exc:
            raise VariationCheckpointError(
                "legacy generation migration replay proof digest is invalid"
            ) from exc
        if (self.generation_intents / (bundle.candidate_id + ".json")).exists():
            raise VariationCheckpointError("legacy generation migration intent already exists")
        legacy_manifest_digest = None
        if bundle.orphan_artifact_digests:
            self._ensure_directory(
                self.legacy_orphan_manifests,
                containment_root=self.root,
            )
            manifest_path = self.legacy_orphan_manifests / (bundle.candidate_id + ".json")
            self._write_once(
                manifest_path,
                {
                    "schema_version": LEGACY_ORPHAN_ARTIFACTS_SCHEMA,
                    "candidate_id": bundle.candidate_id,
                    "generation_record_digest": bundle.generation_record_digest,
                    "trajectory_record_digest": bundle.trajectory_record_digest,
                    "referenced_artifact_digests": list(
                        self._legacy_bundle_artifact_digests(bundle)
                    ),
                    "orphan_artifact_digests": list(bundle.orphan_artifact_digests),
                    "replay_proof_digest": replay_proof_digest,
                },
                prefix=".legacy-orphan-artifacts-",
                containment_root=self.legacy_orphan_manifests,
            )
            legacy_manifest_digest = digest_bytes(
                self._read_regular_file(
                    manifest_path,
                    containment_root=self.legacy_orphan_manifests,
                )
            )
        self.record_generation_start(candidate_id=bundle.candidate_id, context=bundle.context)
        intent = {
            "schema_version": (
                LEGACY_PRIVATE_GENERATION_INTENT_SCHEMA
                if legacy_manifest_digest is not None
                else PRIVATE_GENERATION_INTENT_SCHEMA
            ),
            "candidate_id": bundle.candidate_id,
            "status": "SUCCESS",
            "context": _context_dict(bundle.context),
            "response_contract": bundle.context.response_contract,
            "response_contract_digest": bundle.context.response_contract_digest,
            "generation_profile_digest": bundle.context.generation_profile_digest,
            "rendered_prompt_b64": self._encode_intent_bytes(
                bundle.evidence.rendered_prompt, "rendered prompt"
            ),
            "decoded_model_response_b64": self._encode_intent_bytes(
                bundle.evidence.decoded_model_response, "decoded response"
            ),
            "contract_response_b64": self._encode_intent_bytes(
                bundle.evidence.contract_response, "contract response"
            ),
            "proposal_source_digest": digest_bytes(bundle.evidence.proposal.source),
            "failure_stage": None,
            "error_code": None,
            "replay_error_chain": None,
        }
        if legacy_manifest_digest is not None:
            intent["legacy_orphan_manifest_digest"] = legacy_manifest_digest
        self._write_once(
            self.generation_intents / (bundle.candidate_id + ".json"),
            intent,
            prefix=".private-generation-intent-",
            containment_root=self.generation_intents,
        )
        reproduced = self._complete_generation_intent(bundle.candidate_id)
        if reproduced != bundle.generation_record_digest:
            raise VariationCheckpointError(
                "legacy generation migration did not reproduce its immutable record"
            )
        return reproduced

    def _reconcile_generation_intents(self, *, repair: bool = True) -> None:
        """Reconcile or read-only validate the exact closed generation inventory."""

        self._validate_store_layout()
        self._validate_no_interrupted_evidence_writes()
        intent_paths = self._closed_json_inventory(
            self.generation_intents,
            "private generation intent",
        )
        start_paths = self._closed_json_inventory(
            self.generation_starts,
            "private generation start",
        )
        record_paths = self._closed_json_inventory(
            self.generation_records,
            "private generation record",
        )
        trajectory_paths = self._closed_json_inventory(
            self.records,
            "private trajectory",
        )
        intent_ids = set()
        for path in intent_paths:
            candidate_id = path.stem
            self._load_generation_intent(candidate_id)
            if repair:
                self._complete_generation_intent(candidate_id)
            intent_ids.add(candidate_id)
        if repair:
            # Completing a durable intent publishes its generation and
            # trajectory records. Re-close both inventories after repair so
            # the equality check observes exactly what was durably published.
            record_paths = self._closed_json_inventory(
                self.generation_records,
                "private generation record",
            )
            trajectory_paths = self._closed_json_inventory(
                self.records,
                "private trajectory",
            )
        start_ids = set()
        for path in start_paths:
            candidate_id = path.stem
            self._load_generation_start(candidate_id)
            start_ids.add(candidate_id)
        record_ids = {path.stem for path in record_paths}
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
        for path in trajectory_paths:
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
        manifest_entries = (
            list(self.legacy_orphan_manifests.iterdir())
            if self.legacy_orphan_manifests.exists()
            else []
        )
        if any(not path.is_file() or path.suffix != ".json" for path in manifest_entries):
            raise VariationCheckpointError(
                "legacy orphan-artifact manifest inventory contains an unexpected entry"
            )
        manifest_orphans = set()
        for path in sorted(manifest_entries):
            candidate_id = path.stem
            if candidate_id not in intent_ids:
                raise VariationCheckpointError(
                    "legacy orphan-artifact manifest lacks a complete generation"
                )
            try:
                raw = self._read_regular_file(
                    path,
                    containment_root=self.legacy_orphan_manifests,
                )
                value = json.loads(raw.decode("utf-8"))
                generation_raw = self._read_regular_file(
                    self.generation_records / (candidate_id + ".json"),
                    containment_root=self.generation_records,
                )
                generation_value = json.loads(generation_raw.decode("utf-8"))
                trajectory_raw = self._read_regular_file(
                    self.records / (candidate_id + ".json"),
                    containment_root=self.records,
                )
                intent_value, _intent_context, _intent_raw_values = self._load_generation_intent(
                    candidate_id
                )
            except (OSError, UnicodeError, ValueError) as exc:
                raise VariationCheckpointError(
                    "legacy orphan-artifact manifest evidence is unreadable"
                ) from exc
            required = {
                "schema_version",
                "candidate_id",
                "generation_record_digest",
                "trajectory_record_digest",
                "referenced_artifact_digests",
                "orphan_artifact_digests",
                "replay_proof_digest",
            }
            referenced_values = [
                generation_value.get("rendered_prompt_digest"),
                generation_value.get("decoded_model_response_digest"),
                generation_value.get("contract_response_digest"),
                generation_value.get("proposal_source_digest"),
            ]
            if any(not isinstance(digest, str) for digest in referenced_values):
                raise VariationCheckpointError(
                    "legacy orphan-artifact generation references are invalid"
                )
            expected_referenced = sorted(set(referenced_values))
            orphan_values = value.get("orphan_artifact_digests")
            proof_digest = value.get("replay_proof_digest")
            if (
                canonical_bytes(value) != raw
                or set(value) != required
                or value.get("schema_version") != LEGACY_ORPHAN_ARTIFACTS_SCHEMA
                or value.get("candidate_id") != candidate_id
                or value.get("generation_record_digest") != digest_bytes(generation_raw)
                or value.get("trajectory_record_digest") != digest_bytes(trajectory_raw)
                or value.get("referenced_artifact_digests") != expected_referenced
                or generation_value.get("status") != "SUCCESS"
                or intent_value.get("schema_version")
                != LEGACY_PRIVATE_GENERATION_INTENT_SCHEMA
                or intent_value.get("legacy_orphan_manifest_digest") != digest_bytes(raw)
                or not isinstance(orphan_values, list)
                or len(orphan_values) != 1
                or orphan_values != sorted(set(orphan_values))
                or any(
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or digest != digest.lower()
                    or digest in expected_referenced
                    for digest in orphan_values
                )
                or not isinstance(proof_digest, str)
                or len(proof_digest) != 64
                or proof_digest != proof_digest.lower()
            ):
                raise VariationCheckpointError(
                    "legacy orphan-artifact manifest is not canonical and content-bound"
                )
            try:
                for digest in orphan_values + [proof_digest]:
                    int(digest, 16)
            except ValueError as exc:
                raise VariationCheckpointError(
                    "legacy orphan-artifact manifest contains an invalid digest"
                ) from exc
            if manifest_orphans.intersection(orphan_values):
                raise VariationCheckpointError(
                    "legacy orphan-artifact manifests overlap"
                )
            manifest_orphans.update(orphan_values)
        for path in sorted(self.artifacts.root.rglob("*.tmp")):
            if not repair:
                raise VariationCheckpointError(
                    "private artifact store contains an interrupted write"
                )
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
        if manifest_orphans.intersection(referenced | optional_referenced):
            raise VariationCheckpointError(
                "legacy orphan-artifact manifest overlaps operational evidence"
            )
        allowed_artifacts = referenced | optional_referenced | manifest_orphans
        if (
            not (referenced | manifest_orphans).issubset(actual)
            or not actual.issubset(allowed_artifacts)
        ):
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
        replay_error_chain: Any,
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
            if (
                failure_stage is not None
                or error_code is not None
                or replay_error_chain is not None
                or proposal_source_digest is None
            ):
                raise VariationCheckpointError("private generation success fields are inconsistent")
        elif status == "FAILED":
            if (
                not isinstance(failure_stage, str)
                or not isinstance(error_code, str)
                or proposal_source_digest is not None
                or (
                    failure_stage == "RESPONSE_CONTRACT"
                    and (
                        not isinstance(replay_error_chain, list)
                        or not replay_error_chain
                        or replay_error_chain[0] != error_code
                    )
                )
                or (failure_stage != "RESPONSE_CONTRACT" and replay_error_chain is not None)
            ):
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
            "replay_error_chain": replay_error_chain,
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
            replay_error_chain=None,
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
        replay_error_chain = _generation_replay_error_chain(
            context,
            evidence.stage,
            evidence.contract_response,
        )
        return self._record_generation(
            candidate_id=candidate_id,
            context=context,
            status="FAILED",
            rendered_prompt=evidence.rendered_prompt,
            decoded_model_response=evidence.decoded_model_response,
            contract_response=evidence.contract_response,
            failure_stage=evidence.stage,
            error_code=evidence.error_code,
            replay_error_chain=replay_error_chain,
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

    def _load_generation_success(
        self,
        candidate_id: str,
        *,
        repair: bool,
    ) -> Tuple[CandidateContext, CandidateGenerationEvidence, str]:
        """Reconstruct one successful proposal from immutable raw evidence.

        This is the crash-recovery authority for the narrow boundary after
        model generation and before a Variation attempt checkpoint.  The
        proposal is reparsed from the retained contract-response bytes; no
        model generation is repeated and no proposal fields are guessed from
        receipts or mutable process state.
        """

        self._reconcile_generation_intents(repair=repair)
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
            or value.get("replay_error_chain") is not None
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

    def load_generation_success(
        self, candidate_id: str
    ) -> Tuple[CandidateContext, CandidateGenerationEvidence, str]:
        return self._load_generation_success(candidate_id, repair=True)

    def load_generation_success_read_only(
        self, candidate_id: str
    ) -> Tuple[CandidateContext, CandidateGenerationEvidence, str]:
        """Validate terminal success evidence without completing intents or artifacts."""

        return self._load_generation_success(candidate_id, repair=False)

    def _load_generation_failure(
        self,
        candidate_id: str,
        *,
        repair: bool,
    ) -> Tuple[CandidateContext, CandidateGenerationFailureEvidence, str]:
        """Load and revalidate one exact failed generation record."""

        self._reconcile_generation_intents(repair=repair)
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
            expected_replay_error_chain = _generation_replay_error_chain(
                context,
                evidence.stage,
                evidence.contract_response,
            )
        except Exception as exc:
            raise VariationCheckpointError("private generation failure cannot revalidate its raw evidence") from exc
        if (
            value.get("response_contract") != context.response_contract
            or value.get("response_contract_digest") != context.response_contract_digest
            or value.get("generation_profile_digest") != context.generation_profile_digest
            or value.get("replay_error_chain") != expected_replay_error_chain
            or (
                expected_replay_error_chain is not None
                and value.get("error_code") != expected_replay_error_chain[0]
            )
        ):
            raise VariationCheckpointError("private generation failure differs from its frozen context")
        return context, evidence, digest_bytes(raw)

    def load_generation_failure(
        self, candidate_id: str
    ) -> Tuple[CandidateContext, CandidateGenerationFailureEvidence, str]:
        return self._load_generation_failure(candidate_id, repair=True)

    def load_generation_failure_read_only(
        self, candidate_id: str
    ) -> Tuple[CandidateContext, CandidateGenerationFailureEvidence, str]:
        """Validate terminal failure evidence without completing intents or artifacts."""

        return self._load_generation_failure(candidate_id, repair=False)

    def _successful_generations(
        self,
        *,
        run_id: str,
        task_id: str,
        arm_id: str,
        repair: bool,
    ) -> Tuple[Tuple[str, CandidateContext, CandidateGenerationEvidence, str], ...]:
        """Load every successful generation bound to one trajectory."""

        self._reconcile_generation_intents(repair=repair)
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
            loaded_context, evidence, record_digest = self._load_generation_success(
                str(candidate_id),
                repair=repair,
            )
            if loaded_context != context:
                raise VariationCheckpointError("private generation context changed while loading")
            found.append((str(candidate_id), context, evidence, record_digest))
        found.sort(key=lambda item: item[1].attempt_index)
        indices = [item[1].attempt_index for item in found]
        if len(indices) != len(set(indices)):
            raise VariationCheckpointError("successful private generation attempt indices are duplicated")
        return tuple(found)

    def successful_generations(
        self,
        *,
        run_id: str,
        task_id: str,
        arm_id: str,
    ) -> Tuple[Tuple[str, CandidateContext, CandidateGenerationEvidence, str], ...]:
        """Load successful generations, repairing an interrupted STARTED write."""

        return self._successful_generations(
            run_id=run_id,
            task_id=task_id,
            arm_id=arm_id,
            repair=True,
        )

    def successful_generations_read_only(
        self,
        *,
        run_id: str,
        task_id: str,
        arm_id: str,
    ) -> Tuple[Tuple[str, CandidateContext, CandidateGenerationEvidence, str], ...]:
        """Validate terminal successful-generation inventory without repairing it."""

        return self._successful_generations(
            run_id=run_id,
            task_id=task_id,
            arm_id=arm_id,
            repair=False,
        )

    def _source_contract_failures(
        self,
        *,
        run_id: str,
        task_id: str,
        arm_id: str,
        repair: bool,
    ) -> Tuple[Mapping[str, Any], ...]:
        """Load immutable source-contract failures for exact bounded resume."""

        self._reconcile_generation_intents(repair=repair)
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
            raw_artifacts = {}
            for field in (
                "rendered_prompt_digest", "decoded_model_response_digest", "contract_response_digest"
            ):
                artifact_digest = value.get(field)
                if not isinstance(artifact_digest, str):
                    raise VariationCheckpointError("source-contract failure raw evidence is missing or corrupt")
                artifact = self._read_artifact(artifact_digest)
                if digest_bytes(artifact) != artifact_digest:
                    raise VariationCheckpointError("source-contract failure raw evidence is missing or corrupt")
                raw_artifacts[field] = artifact
            expected_replay_error_chain = _generation_replay_error_chain(
                context,
                value.get("failure_stage"),
                raw_artifacts["contract_response_digest"],
            )
            if (
                value.get("replay_error_chain") != expected_replay_error_chain
                or value.get("error_code") != expected_replay_error_chain[0]
            ):
                raise VariationCheckpointError("source-contract failure replay classification is invalid")
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

    def source_contract_failures(
        self,
        *,
        run_id: str,
        task_id: str,
        arm_id: str,
    ) -> Tuple[Mapping[str, Any], ...]:
        """Load source-contract failures, repairing an interrupted STARTED write."""

        return self._source_contract_failures(
            run_id=run_id,
            task_id=task_id,
            arm_id=arm_id,
            repair=True,
        )

    def source_contract_failures_read_only(
        self,
        *,
        run_id: str,
        task_id: str,
        arm_id: str,
    ) -> Tuple[Mapping[str, Any], ...]:
        """Validate terminal source-contract-failure inventory without repairing it."""

        return self._source_contract_failures(
            run_id=run_id,
            task_id=task_id,
            arm_id=arm_id,
            repair=False,
        )

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
                    or generation.get("replay_error_chain") is not None
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
    "LEGACY_PRIVATE_TRAJECTORY_SCHEMA", "LegacyGenerationBundle", "PRIVATE_GENERATION_SCHEMA",
    "PRIVATE_TRAJECTORY_SCHEMA", "PrivateTrajectoryStore", "replay_exception_chain_classification",
    "response_contract_replay_error_chain",
]
