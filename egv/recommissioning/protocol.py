"""Immutable, model-visible contracts for the repaired Nemotron pilots.

The classes in this module deliberately separate model-visible material from
private evaluator records and public redacted receipts. A prompt is rendered
only from a validated learnable-v2 package and same-coordinate feedback. Its
content address binds the coordinate, task, model profile, token budget, seed,
and exact messages, so generation code never accepts caller-supplied chat
messages or evidence identifiers.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..canonical import canonical_bytes, canonical_json, canonical_value, digest_bytes, digest_for
from ..receipts import key_id_for_public_key, load_public_key, public_key_bytes


class RecommissioningProtocolError(ValueError):
    """A prompt, feedback record, or coordinate crossed its frozen boundary."""


PROTOCOL_IDS = ("P0", "P1")
MODEL_BUNDLE_IDS = ("LIGHTNING", "SUPER")
DIAGNOSTIC_IDS = (
    "PASS",
    "WRONG_OUTPUT",
    "SYNTAX_OR_IMPORT",
    "RUNTIME_EXCEPTION",
    "TIMEOUT",
    "RESOURCE_LIMIT",
    "MUTATION_LOCUS_VIOLATION",
    "PROTOCOL_VIOLATION",
)

_DIGEST = re.compile(r"[0-9a-f]{64}")
_PRIVATE_VOCABULARY = (
    "expected_output",
    "expected output",
    "golden_patch",
    "golden patch",
    "oracle_token",
    "oracle token",
    "evaluator_seed",
    "evaluator seed",
    "private_key",
    "private key",
    "hidden path",
)
_MAX_JSON_DEPTH = 16
_MAX_JSON_NODES = 4096
_MAX_PUBLIC_SPEC_BYTES = 64 * 1024
_MAX_SOURCE_CHARACTERS = 64 * 1024
_MAX_SUMMARY_CHARACTERS = 1024
_PUBLIC_TASK_REF = re.compile(r"task_[0-9a-f]{64}")
_SENSITIVE_PUBLIC_PATTERNS = (
    re.compile(r"(?i)\b(?:api[-_ ]?key|credential|password|passwd|secret|bearer|authorization)\b"),
    re.compile(r"(?i)\bsk-[a-z0-9_-]{8,}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)(?:[a-z]:[\\/]|\\\\|/(?:home|users|var|tmp|etc|opt|srv)/)"),
    re.compile(r"(?i)\b(?:https?|ssh|file)://|\blocalhost\b|\b127\.0\.0\.1\b|::1"),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b"),
    re.compile(r"(?i)\b(?:worker|node|spark|trainer|evaluator|router|gateway|postgres|qdrant|redis)[-_]?\d+\b"),
)


def _require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise RecommissioningProtocolError("{} is not a full lowercase SHA-256 digest".format(label))
    return value


def _require_key_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"key_[0-9a-f]{64}", value) is None:
        raise RecommissioningProtocolError("{} is not a content-derived Ed25519 key ID".format(label))
    return value


def _require_text(value: Any, label: str, *, max_characters: int, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()) or len(value) > max_characters:
        raise RecommissioningProtocolError("{} is empty or exceeds its character ceiling".format(label))
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise RecommissioningProtocolError("{} contains invalid Unicode".format(label)) from exc
    if "\x00" in value:
        raise RecommissioningProtocolError("{} contains a NUL character".format(label))
    return value


def _decode_ed25519_signature(value: Any, label: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise RecommissioningProtocolError("{} must be canonical Ed25519 base64url".format(label))
    padded = value + "=" * (-len(value) % 4)
    try:
        decoded = base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)
    except (ValueError, UnicodeError) as exc:
        raise RecommissioningProtocolError("{} must be canonical Ed25519 base64url".format(label)) from exc
    canonical = base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=")
    if len(decoded) != 64 or canonical != value:
        raise RecommissioningProtocolError("{} must be canonical Ed25519 base64url".format(label))
    return decoded


def _public_key_identity(public_key: Any, label: str) -> Tuple[bytes, str]:
    try:
        raw = public_key_bytes(public_key)
        return raw, key_id_for_public_key(raw)
    except Exception as exc:
        raise RecommissioningProtocolError("{} is not a valid Ed25519 public key".format(label)) from exc


def _signer_identity(signer: Any, label: str) -> Tuple[bytes, str]:
    if not callable(getattr(signer, "sign_bytes", None)) or not hasattr(signer, "public_key"):
        raise RecommissioningProtocolError("{} does not expose an Ed25519 signing interface".format(label))
    raw, expected_key_id = _public_key_identity(signer.public_key, label + " public key")
    if getattr(signer, "key_id", None) != expected_key_id:
        raise RecommissioningProtocolError("{} key ID is not derived from its public key".format(label))
    return raw, expected_key_id


def _verify_ed25519(
    *,
    public_key: Any,
    key_id: Any,
    signature: Any,
    payload: Mapping[str, Any],
    label: str,
) -> bytes:
    raw, expected_key_id = _public_key_identity(public_key, label + " public key")
    if key_id != expected_key_id:
        raise RecommissioningProtocolError("{} uses an unexpected signing key".format(label))
    try:
        load_public_key(raw).verify(
            _decode_ed25519_signature(signature, label + " signature"),
            canonical_bytes(payload),
        )
    except RecommissioningProtocolError:
        raise
    except Exception as exc:
        raise RecommissioningProtocolError("{} Ed25519 signature is invalid".format(label)) from exc
    return raw


def _reject_sensitive_public_text(value: str, label: str) -> None:
    _require_text(value, label, max_characters=_MAX_SUMMARY_CHARACTERS, allow_empty=True)
    lowered = value.lower()
    if any(token in lowered for token in _PRIVATE_VOCABULARY) or any(
        pattern.search(value) for pattern in _SENSITIVE_PUBLIC_PATTERNS
    ):
        raise RecommissioningProtocolError("{} contains prohibited private material".format(label))


def _scan_public_projection(value: Any, label: str) -> None:
    if isinstance(value, str):
        if any(pattern.search(value) for pattern in _SENSITIVE_PUBLIC_PATTERNS):
            raise RecommissioningProtocolError("{} contains prohibited public data".format(label))
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "signature" or key.endswith("_digest"):
                continue
            _scan_public_projection(child, label)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _scan_public_projection(child, label)


def public_task_ref(task_package_digest: str) -> str:
    """Return the only task identifier admitted to a public projection."""

    digest = _require_digest(task_package_digest, "task-package digest for public reference")
    return "task_" + digest_for({"task_package_digest": digest})


def _validate_bounded_json(value: Any, label: str, *, byte_ceiling: int) -> Any:
    """Validate a detached JSON tree with explicit depth, node, and UTF-8 bounds."""

    nodes = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise RecommissioningProtocolError("{} exceeds the JSON node ceiling".format(label))
        if depth > _MAX_JSON_DEPTH:
            raise RecommissioningProtocolError("{} exceeds the JSON depth ceiling".format(label))
        if item is None or type(item) in {bool, int}:  # bool is intentionally not treated as an integer.
            return
        if isinstance(item, float):
            raise RecommissioningProtocolError("{} may not contain floating-point values".format(label))
        if isinstance(item, str):
            _require_text(item, label + " string", max_characters=16 * 1024, allow_empty=True)
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                _require_text(key, label + " key", max_characters=256)
                visit(child, depth + 1)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child, depth + 1)
            return
        raise RecommissioningProtocolError("{} contains a non-JSON value".format(label))

    visit(value, 0)
    detached = canonical_value(value)
    if len(canonical_json(detached).encode("utf-8")) > byte_ceiling:
        raise RecommissioningProtocolError("{} exceeds the canonical UTF-8 byte ceiling".format(label))
    return detached


@dataclass(frozen=True)
class TokenBudget:
    """Frozen per-request and per-coordinate token/time limits."""

    max_prompt_tokens: int
    max_completion_tokens: int
    max_total_tokens: int
    max_coordinate_tokens: int
    hard_timeout_ms: int
    seed: int

    def validate(self, *, max_model_len: Optional[int] = None) -> None:
        values = (
            (self.max_prompt_tokens, "prompt-token", 1, 131072),
            (self.max_completion_tokens, "completion-token", 1, 8192),
            (self.max_total_tokens, "total-token", 2, 131072),
            (self.max_coordinate_tokens, "coordinate-token", 2, 1048576),
            (self.hard_timeout_ms, "hard-timeout", 1, 600000),
            (self.seed, "seed", 0, 2**31 - 1),
        )
        for value, label, minimum, maximum in values:
            if type(value) is not int or not minimum <= value <= maximum:
                raise RecommissioningProtocolError("{} budget is outside the frozen range".format(label))
        if self.max_prompt_tokens + self.max_completion_tokens > self.max_total_tokens:
            raise RecommissioningProtocolError("request token ceilings exceed the frozen total")
        if self.max_total_tokens > self.max_coordinate_tokens:
            raise RecommissioningProtocolError("request total exceeds the coordinate token ceiling")
        if max_model_len is not None:
            if type(max_model_len) is not int or max_model_len < 1:
                raise RecommissioningProtocolError("model context length is malformed")
            if self.max_total_tokens > max_model_len:
                raise RecommissioningProtocolError("request total exceeds the frozen model context")

    def manifest(self) -> Dict[str, Any]:
        self.validate()
        return {
            "schema_version": "egv-recommissioning-token-budget-v2",
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_completion_tokens": self.max_completion_tokens,
            "max_total_tokens": self.max_total_tokens,
            "max_coordinate_tokens": self.max_coordinate_tokens,
            "hard_timeout_ms": self.hard_timeout_ms,
            "seed": self.seed,
            "temperature": 0.0,
            "n": 1,
        }

    @property
    def digest(self) -> str:
        return digest_for(self.manifest())


# Backward-compatible spelling for callers of the first draft. The value is
# now the stronger prompt/total/coordinate budget, not the old partial budget.
GenerationBudget = TokenBudget


@dataclass(frozen=True)
class PilotCoordinate:
    campaign_id: str
    model_bundle: str
    protocol_id: str
    task_id: str
    task_family: str
    seed: int
    generation_receipt_key_id: str
    evaluator_key_id: str
    evaluator_profile_digest: str

    def validate(self) -> None:
        for value, label in (
            (self.campaign_id, "campaign ID"),
            (self.task_id, "task ID"),
            (self.task_family, "task family"),
        ):
            _require_text(value, label, max_characters=128)
            if any(character in value for character in ("/", "\\")):
                raise RecommissioningProtocolError("{} is not a bounded identity".format(label))
        if self.model_bundle not in MODEL_BUNDLE_IDS:
            raise RecommissioningProtocolError("coordinate uses an unknown model bundle")
        if self.protocol_id not in PROTOCOL_IDS:
            raise RecommissioningProtocolError("coordinate uses an unknown protocol")
        if type(self.seed) is not int or self.seed < 0 or self.seed > 2**31 - 1:
            raise RecommissioningProtocolError("coordinate seed is outside the frozen range")
        _require_key_id(self.generation_receipt_key_id, "coordinate generation-receipt key ID")
        _require_key_id(self.evaluator_key_id, "coordinate evaluator key ID")
        _require_digest(self.evaluator_profile_digest, "coordinate evaluator-profile digest")

    def identity_record(self) -> Dict[str, Any]:
        self.validate()
        return {
            "campaign_id": self.campaign_id,
            "model_bundle": self.model_bundle,
            "protocol_id": self.protocol_id,
            "task_id": self.task_id,
            "task_family": self.task_family,
            "seed": self.seed,
            "generation_receipt_key_id": self.generation_receipt_key_id,
            "evaluator_key_id": self.evaluator_key_id,
            "evaluator_profile_digest": self.evaluator_profile_digest,
        }

    @property
    def coordinate_id(self) -> str:
        return digest_for(self.identity_record())


@dataclass(frozen=True, init=False)
class VisibleTaskPackage:
    """Immutable adapter over one exact closed learnable-v2 model package."""

    task_id: str
    task_family: str
    split: str
    ordinal: int
    public_spec_json: str
    initial_source: str
    public_locus: str
    original_package_digest: str
    original_task_record_digest: str
    corpus_manifest_digest: str
    corpus_task_index: int
    corpus_membership_digest: str
    public_spec_digest: str
    initial_source_digest: str
    adapter_kind: str
    _validated_corpus_manifest_json: Optional[str] = field(repr=False, compare=False)

    @classmethod
    def _from_values(cls, **values: Any) -> "VisibleTaskPackage":
        result = object.__new__(cls)
        for field_name in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            object.__setattr__(result, field_name, values[field_name])
        result.validate()
        return result

    @classmethod
    def from_learnable_v2(
        cls, validated_package: Mapping[str, Any], validated_corpus_manifest: Mapping[str, Any]
    ) -> "VisibleTaskPackage":
        """Revalidate a package and prove exact membership in the full corpus manifest."""

        try:
            from ..evaluation.learnable import validate_learnable_manifest, validate_model_visible_package

            package = validate_model_visible_package(validated_package)
            corpus_manifest = validate_learnable_manifest(validated_corpus_manifest)
        except (ImportError, KeyError, ValueError, TypeError) as exc:
            raise RecommissioningProtocolError(
                "learnable-v2 package or corpus manifest failed closed validation"
            ) from exc
        task = package["task"]
        matching_indexes = [
            index for index, manifest_task in enumerate(corpus_manifest["tasks"]) if manifest_task == task
        ]
        if matching_indexes != [
            next(
                (
                    index
                    for index, manifest_task in enumerate(corpus_manifest["tasks"])
                    if manifest_task.get("task_id") == task["task_id"]
                ),
                -1,
            )
        ]:
            raise RecommissioningProtocolError("learnable-v2 package is not an exact unique corpus-manifest member")
        task_index = matching_indexes[0]
        corpus_manifest_digest = corpus_manifest["manifest_digest"]
        spec = task["public_spec"]
        return cls._from_values(
            task_id=task["task_id"],
            task_family=task["family_id"],
            split=task["split"],
            ordinal=task["ordinal"],
            public_spec_json=canonical_json(spec),
            initial_source=task["starter_source"],
            public_locus="module:main",
            original_package_digest=package["package_digest"],
            original_task_record_digest=task["record_digest"],
            corpus_manifest_digest=corpus_manifest_digest,
            corpus_task_index=task_index,
            corpus_membership_digest=digest_for(
                {
                    "schema_version": "egv-learnable-corpus-membership-v1",
                    "corpus_manifest_digest": corpus_manifest_digest,
                    "corpus_task_index": task_index,
                    "task_record_digest": task["record_digest"],
                    "package_digest": package["package_digest"],
                }
            ),
            public_spec_digest=task["public_spec_digest"],
            initial_source_digest=task["starter_source_digest"],
            adapter_kind="LEARNABLE_V2",
            _validated_corpus_manifest_json=canonical_json(corpus_manifest),
        )

    @classmethod
    def for_test(
        cls,
        *,
        task_id: str,
        task_family: str,
        public_spec: Mapping[str, Any],
        initial_source: str,
        split: str = "train",
        ordinal: int = 1,
        public_locus: str = "module:main",
    ) -> "VisibleTaskPackage":
        """Construct a visibly test-only fixture; production runners must not call this."""

        spec = _validate_bounded_json(public_spec, "test public specification", byte_ceiling=_MAX_PUBLIC_SPEC_BYTES)
        source_digest = digest_bytes(initial_source.encode("utf-8"))
        task_core = {
            "schema_version": "egv-test-visible-task-v1",
            "task_id": task_id,
            "family_id": task_family,
            "split": split,
            "ordinal": ordinal,
            "public_spec": spec,
            "public_spec_digest": digest_for(spec),
            "starter_source": initial_source,
            "starter_source_digest": source_digest,
        }
        record_digest = digest_for(task_core)
        package_core = {
            "schema_version": "egv-test-model-package-v1",
            "task": {**task_core, "record_digest": record_digest},
        }
        return cls._from_values(
            task_id=task_id,
            task_family=task_family,
            split=split,
            ordinal=ordinal,
            public_spec_json=canonical_json(spec),
            initial_source=initial_source,
            public_locus=public_locus,
            original_package_digest=digest_for(package_core),
            original_task_record_digest=record_digest,
            corpus_manifest_digest=digest_for("test-corpus-manifest"),
            corpus_task_index=0,
            corpus_membership_digest=digest_for(
                {
                    "schema_version": "egv-test-corpus-membership-v1",
                    "corpus_manifest_digest": digest_for("test-corpus-manifest"),
                    "corpus_task_index": 0,
                    "task_record_digest": record_digest,
                    "package_digest": digest_for(package_core),
                }
            ),
            public_spec_digest=digest_for(spec),
            initial_source_digest=source_digest,
            adapter_kind="TEST_ONLY",
            _validated_corpus_manifest_json=None,
        )

    @property
    def structured_public_spec(self) -> Dict[str, Any]:
        return json.loads(self.public_spec_json)

    def model_visible_record(self) -> Dict[str, Any]:
        return {
            "schema_version": "egv-recommissioning-model-visible-task-v2",
            "adapter_kind": self.adapter_kind,
            "task_id": self.task_id,
            "task_family": self.task_family,
            "split": self.split,
            "ordinal": self.ordinal,
            "public_spec": self.structured_public_spec,
            "initial_source": self.initial_source,
            "public_locus": self.public_locus,
            "original_package_digest": self.original_package_digest,
            "original_task_record_digest": self.original_task_record_digest,
            "corpus_manifest_digest": self.corpus_manifest_digest,
            "corpus_task_index": self.corpus_task_index,
            "corpus_membership_digest": self.corpus_membership_digest,
            "public_spec_digest": self.public_spec_digest,
            "initial_source_digest": self.initial_source_digest,
        }

    @property
    def task_package_digest(self) -> str:
        return digest_for(self.model_visible_record())

    @property
    def manifest_digest(self) -> str:
        """Compatibility name for the first draft."""

        return self.task_package_digest

    def validate(self, coordinate: Optional[PilotCoordinate] = None) -> None:
        _require_text(self.task_id, "task ID", max_characters=128)
        _require_text(self.task_family, "task family", max_characters=128)
        _require_text(self.split, "task split", max_characters=32)
        if type(self.ordinal) is not int or self.ordinal < 1 or self.ordinal > 10000:
            raise RecommissioningProtocolError("task ordinal is outside the frozen range")
        if type(self.corpus_task_index) is not int or not 0 <= self.corpus_task_index < 10000:
            raise RecommissioningProtocolError("corpus task index is outside the frozen range")
        if coordinate is not None:
            coordinate.validate()
            if self.task_id != coordinate.task_id or self.task_family != coordinate.task_family:
                raise RecommissioningProtocolError("visible task package differs from its coordinate")
        try:
            spec = json.loads(self.public_spec_json)
        except (UnicodeError, ValueError) as exc:
            raise RecommissioningProtocolError("public specification is not canonical JSON") from exc
        if canonical_json(spec) != self.public_spec_json:
            raise RecommissioningProtocolError("public specification JSON is not canonical")
        _validate_bounded_json(spec, "public specification", byte_ceiling=_MAX_PUBLIC_SPEC_BYTES)
        _require_text(self.initial_source, "initial source", max_characters=_MAX_SOURCE_CHARACTERS)
        _require_text(self.public_locus, "public locus", max_characters=128)
        if self.public_locus != "module:main":
            raise RecommissioningProtocolError("learnable-v2 public locus must be module:main")
        for value, label in (
            (self.original_package_digest, "original package digest"),
            (self.original_task_record_digest, "original task-record digest"),
            (self.corpus_manifest_digest, "corpus manifest digest"),
            (self.corpus_membership_digest, "corpus membership digest"),
            (self.public_spec_digest, "public-spec digest"),
            (self.initial_source_digest, "initial-source digest"),
        ):
            _require_digest(value, label)
        if self.public_spec_digest != digest_for(spec):
            raise RecommissioningProtocolError("public-spec digest does not bind the structured specification")
        if self.initial_source_digest != digest_bytes(self.initial_source.encode("utf-8")):
            raise RecommissioningProtocolError("initial-source digest does not bind the source")
        from ..evaluation.sandbox import validate_candidate_source_contract

        source_violation = validate_candidate_source_contract(self.initial_source.encode("utf-8"))
        if source_violation is not None:
            raise RecommissioningProtocolError("initial source violates pure-return-v1: {}".format(source_violation))
        if self.adapter_kind not in {"LEARNABLE_V2", "TEST_ONLY"}:
            raise RecommissioningProtocolError("task adapter kind is not closed")
        if self.adapter_kind == "LEARNABLE_V2":
            if not isinstance(self._validated_corpus_manifest_json, str):
                raise RecommissioningProtocolError("learnable-v2 task lost its full corpus-manifest proof")
            try:
                from ..evaluation.learnable import validate_learnable_manifest

                corpus_manifest = validate_learnable_manifest(json.loads(self._validated_corpus_manifest_json))
            except (ImportError, KeyError, ValueError, TypeError) as exc:
                raise RecommissioningProtocolError("learnable-v2 corpus-manifest proof failed validation") from exc
            if canonical_json(corpus_manifest) != self._validated_corpus_manifest_json:
                raise RecommissioningProtocolError("learnable-v2 corpus-manifest proof is not canonical")
            if corpus_manifest["manifest_digest"] != self.corpus_manifest_digest:
                raise RecommissioningProtocolError("task corpus-manifest digest differs from its full proof")
            try:
                manifest_task = corpus_manifest["tasks"][self.corpus_task_index]
            except (IndexError, KeyError, TypeError) as exc:
                raise RecommissioningProtocolError("task corpus membership index is invalid") from exc
            expected_task_bindings = {
                "task_id": self.task_id,
                "family_id": self.task_family,
                "split": self.split,
                "ordinal": self.ordinal,
                "public_spec": spec,
                "public_spec_digest": self.public_spec_digest,
                "starter_source": self.initial_source,
                "starter_source_digest": self.initial_source_digest,
                "record_digest": self.original_task_record_digest,
            }
            for field_name, expected_value in expected_task_bindings.items():
                if manifest_task.get(field_name) != expected_value:
                    raise RecommissioningProtocolError("task differs from corpus-manifest member {}".format(field_name))
            package_core = {"schema_version": "egv-learnable-model-package-v2", "task": manifest_task}
            if digest_for(package_core) != self.original_package_digest:
                raise RecommissioningProtocolError("task package digest differs from its corpus-manifest member")
        elif self._validated_corpus_manifest_json is not None:
            raise RecommissioningProtocolError("test task may not carry production corpus-manifest authority")
        membership_schema = (
            "egv-learnable-corpus-membership-v1"
            if self.adapter_kind == "LEARNABLE_V2"
            else "egv-test-corpus-membership-v1"
        )
        expected_membership = digest_for(
            {
                "schema_version": membership_schema,
                "corpus_manifest_digest": self.corpus_manifest_digest,
                "corpus_task_index": self.corpus_task_index,
                "task_record_digest": self.original_task_record_digest,
                "package_digest": self.original_package_digest,
            }
        )
        if self.corpus_membership_digest != expected_membership:
            raise RecommissioningProtocolError("task package lacks an exact corpus-membership proof")
        visible = canonical_json(self.model_visible_record()).lower()
        if any(token in visible for token in _PRIVATE_VOCABULARY):
            raise RecommissioningProtocolError("visible task package contains evaluator-private vocabulary")


@dataclass(frozen=True)
class ChatMessage:
    """One immutable model-visible chat message."""

    role: str
    content: str

    def validate(self) -> None:
        if self.role not in {"system", "user"}:
            raise RecommissioningProtocolError("prompt contains an unapproved chat role")
        _require_text(self.content, "chat content", max_characters=256 * 1024)

    def record(self) -> Dict[str, str]:
        self.validate()
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, init=False)
class PromptEnvelope:
    """Content-addressed immutable prompt and all experimental bindings."""

    coordinate_id: str
    model_bundle: str
    task_id: str
    public_locus: str
    task_adapter_kind: str
    task_package_digest: str
    model_profile_digest: str
    budget_digest: str
    seed: int
    generation_receipt_key_id: str
    evaluator_key_id: str
    evaluator_profile_digest: str
    protocol_id: str
    attempt_index: int
    visible_evidence_ids: Tuple[str, ...]
    messages: Tuple[ChatMessage, ...]
    prompt_digest: str
    _coordinate: PilotCoordinate = field(repr=False, compare=False)
    _task: VisibleTaskPackage = field(repr=False, compare=False)
    _budget: TokenBudget = field(repr=False, compare=False)
    _feedback: Tuple[AttemptFeedback, ...] = field(repr=False, compare=False)

    @classmethod
    def _build(
        cls,
        *,
        coordinate: PilotCoordinate,
        task: VisibleTaskPackage,
        model_profile_digest: str,
        budget: TokenBudget,
        attempt_index: int,
        feedback: Sequence[AttemptFeedback] = (),
    ) -> "PromptEnvelope":
        if not isinstance(coordinate, PilotCoordinate) or not isinstance(task, VisibleTaskPackage):
            raise RecommissioningProtocolError("prompt construction requires validated coordinate and task objects")
        coordinate.validate()
        task.validate(coordinate)
        _require_digest(model_profile_digest, "model-profile digest")
        budget.validate()
        if budget.seed != coordinate.seed:
            raise RecommissioningProtocolError("prompt budget seed differs from its coordinate")
        frozen_feedback = tuple(feedback)
        visible_evidence_ids, messages = _canonical_prompt_material(
            coordinate=coordinate,
            task=task,
            model_profile_digest=model_profile_digest,
            budget=budget,
            attempt_index=attempt_index,
            feedback=frozen_feedback,
        )
        values = {
            "coordinate_id": coordinate.coordinate_id,
            "model_bundle": coordinate.model_bundle,
            "task_id": task.task_id,
            "public_locus": task.public_locus,
            "task_adapter_kind": task.adapter_kind,
            "task_package_digest": task.task_package_digest,
            "model_profile_digest": model_profile_digest,
            "budget_digest": budget.digest,
            "seed": coordinate.seed,
            "generation_receipt_key_id": coordinate.generation_receipt_key_id,
            "evaluator_key_id": coordinate.evaluator_key_id,
            "evaluator_profile_digest": coordinate.evaluator_profile_digest,
            "protocol_id": coordinate.protocol_id,
            "attempt_index": attempt_index,
            "visible_evidence_ids": visible_evidence_ids,
            "messages": messages,
            "_coordinate": coordinate,
            "_task": task,
            "_budget": budget,
            "_feedback": frozen_feedback,
        }
        core = {
            "schema_version": "egv-recommissioning-prompt-envelope-v2",
            "coordinate_id": values["coordinate_id"],
            "model_bundle": values["model_bundle"],
            "task_id": values["task_id"],
            "public_locus": values["public_locus"],
            "task_adapter_kind": values["task_adapter_kind"],
            "task_package_digest": values["task_package_digest"],
            "model_profile_digest": values["model_profile_digest"],
            "budget_digest": values["budget_digest"],
            "seed": values["seed"],
            "generation_receipt_key_id": values["generation_receipt_key_id"],
            "evaluator_key_id": values["evaluator_key_id"],
            "evaluator_profile_digest": values["evaluator_profile_digest"],
            "protocol_id": values["protocol_id"],
            "attempt_index": values["attempt_index"],
            "visible_evidence_ids": list(values["visible_evidence_ids"]),
            "messages": [message.record() for message in values["messages"]],
        }
        values["prompt_digest"] = digest_for(core)
        result = object.__new__(cls)
        for field_name in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            object.__setattr__(result, field_name, values[field_name])
        result.validate(
            coordinate=coordinate,
            task=task,
            model_profile_digest=model_profile_digest,
            budget=budget,
        )
        return result

    def core_record(self) -> Dict[str, Any]:
        return {
            "schema_version": "egv-recommissioning-prompt-envelope-v2",
            "coordinate_id": self.coordinate_id,
            "model_bundle": self.model_bundle,
            "task_id": self.task_id,
            "public_locus": self.public_locus,
            "task_adapter_kind": self.task_adapter_kind,
            "task_package_digest": self.task_package_digest,
            "model_profile_digest": self.model_profile_digest,
            "budget_digest": self.budget_digest,
            "seed": self.seed,
            "generation_receipt_key_id": self.generation_receipt_key_id,
            "evaluator_key_id": self.evaluator_key_id,
            "evaluator_profile_digest": self.evaluator_profile_digest,
            "protocol_id": self.protocol_id,
            "attempt_index": self.attempt_index,
            "visible_evidence_ids": list(self.visible_evidence_ids),
            "messages": [message.record() for message in self.messages],
        }

    def validate(
        self,
        *,
        coordinate: Optional[PilotCoordinate] = None,
        task: Optional[VisibleTaskPackage] = None,
        model_profile_digest: Optional[str] = None,
        budget: Optional[TokenBudget] = None,
    ) -> None:
        if (
            not isinstance(self._coordinate, PilotCoordinate)
            or not isinstance(self._task, VisibleTaskPackage)
            or not isinstance(self._budget, TokenBudget)
            or not isinstance(self._feedback, tuple)
        ):
            raise RecommissioningProtocolError("prompt lost its validated canonical-render inputs")
        self._coordinate.validate()
        self._task.validate(self._coordinate)
        self._budget.validate()
        canonical_evidence_ids, canonical_messages = _canonical_prompt_material(
            coordinate=self._coordinate,
            task=self._task,
            model_profile_digest=self.model_profile_digest,
            budget=self._budget,
            attempt_index=self.attempt_index,
            feedback=self._feedback,
        )
        for value, label in (
            (self.coordinate_id, "prompt coordinate digest"),
            (self.task_package_digest, "prompt task-package digest"),
            (self.model_profile_digest, "prompt model-profile digest"),
            (self.budget_digest, "prompt budget digest"),
            (self.prompt_digest, "prompt digest"),
        ):
            _require_digest(value, label)
        _require_text(self.task_id, "prompt task ID", max_characters=128)
        if self.model_bundle not in MODEL_BUNDLE_IDS:
            raise RecommissioningProtocolError("prompt model bundle is not closed")
        if self.public_locus != "module:main":
            raise RecommissioningProtocolError("prompt public locus is outside learnable-v2")
        if self.task_adapter_kind not in {"LEARNABLE_V2", "TEST_ONLY"}:
            raise RecommissioningProtocolError("prompt task adapter kind is not closed")
        if self.protocol_id not in PROTOCOL_IDS:
            raise RecommissioningProtocolError("prompt protocol is not closed")
        if type(self.seed) is not int or self.seed < 0 or self.seed > 2**31 - 1:
            raise RecommissioningProtocolError("prompt seed is outside the frozen range")
        _require_key_id(self.generation_receipt_key_id, "prompt generation-receipt key ID")
        _require_key_id(self.evaluator_key_id, "prompt evaluator key ID")
        _require_digest(self.evaluator_profile_digest, "prompt evaluator-profile digest")
        if type(self.attempt_index) is not int or not 1 <= self.attempt_index <= 6:
            raise RecommissioningProtocolError("prompt attempt index is outside the six-candidate budget")
        if not isinstance(self.visible_evidence_ids, tuple) or list(self.visible_evidence_ids) != sorted(
            set(self.visible_evidence_ids)
        ):
            raise RecommissioningProtocolError("prompt evidence IDs are not sorted and unique")
        if any(not isinstance(value, str) or len(value) > 96 for value in self.visible_evidence_ids):
            raise RecommissioningProtocolError("prompt evidence ID is not bounded")
        if not isinstance(self.messages, tuple) or len(self.messages) != 2:
            raise RecommissioningProtocolError("prompt must contain exactly two immutable messages")
        for message in self.messages:
            if not isinstance(message, ChatMessage):
                raise RecommissioningProtocolError("prompt messages must use the immutable ChatMessage type")
            message.validate()
        if tuple(message.role for message in self.messages) != ("system", "user"):
            raise RecommissioningProtocolError("prompt roles differ from the frozen order")
        if self.visible_evidence_ids != canonical_evidence_ids or self.messages != canonical_messages:
            raise RecommissioningProtocolError("prompt messages differ from the canonical deterministic render")
        if digest_for(self.core_record()) != self.prompt_digest:
            raise RecommissioningProtocolError("prompt digest does not bind the complete immutable envelope")
        bound_coordinate = coordinate or self._coordinate
        bound_coordinate.validate()
        expected = (
            self._coordinate.coordinate_id,
            self._coordinate.model_bundle,
            self._coordinate.task_id,
            self._coordinate.protocol_id,
            self._coordinate.seed,
            self._coordinate.generation_receipt_key_id,
            self._coordinate.evaluator_key_id,
            self._coordinate.evaluator_profile_digest,
        )
        actual = (
            self.coordinate_id,
            self.model_bundle,
            self.task_id,
            self.protocol_id,
            self.seed,
            self.generation_receipt_key_id,
            self.evaluator_key_id,
            self.evaluator_profile_digest,
        )
        if actual != expected:
            raise RecommissioningProtocolError("prompt differs from its coordinate binding")
        if self._coordinate.identity_record() != bound_coordinate.identity_record():
            raise RecommissioningProtocolError("prompt canonical coordinate differs from its supplied coordinate")
        bound_task = task or self._task
        if task is not None and self._task.task_package_digest != task.task_package_digest:
            raise RecommissioningProtocolError("prompt canonical task-package differs from its supplied task")
        bound_task.validate(bound_coordinate)
        if (
            self.task_package_digest != self._task.task_package_digest
            or self.task_id != self._task.task_id
            or self.public_locus != self._task.public_locus
            or self.task_adapter_kind != self._task.adapter_kind
        ):
            raise RecommissioningProtocolError("prompt differs from its task-package binding")
        if model_profile_digest is not None and self.model_profile_digest != _require_digest(
            model_profile_digest, "expected model-profile digest"
        ):
            raise RecommissioningProtocolError("prompt differs from its model-profile binding")
        bound_budget = budget or self._budget
        bound_budget.validate()
        if self.budget_digest != self._budget.digest or self.seed != self._budget.seed:
            raise RecommissioningProtocolError("prompt differs from its budget or seed binding")
        if self._budget.digest != bound_budget.digest:
            raise RecommissioningProtocolError("prompt canonical budget differs from its supplied budget")


@dataclass(frozen=True, init=False)
class AttemptFeedback:
    """Model-visible feedback bound to the exact preceding private evaluation."""

    coordinate_id: str
    task_package_digest: str
    model_profile_digest: str
    evaluator_profile_digest: str
    attempt_index: int
    prior_prompt_digest: str
    generation_receipt_digest: str
    evaluator_receipt_digest: str
    candidate_digest: str
    source_digest: str
    source: str
    diagnostic: str
    public_summary: str
    previous_feedback_digest: Optional[str]
    feedback_digest: str

    @property
    def evidence_id(self) -> str:
        return "egv-feedback-" + self.feedback_digest

    def core_record(self) -> Dict[str, Any]:
        return {
            "schema_version": "egv-recommissioning-feedback-v2",
            "coordinate_id": self.coordinate_id,
            "task_package_digest": self.task_package_digest,
            "model_profile_digest": self.model_profile_digest,
            "evaluator_profile_digest": self.evaluator_profile_digest,
            "attempt_index": self.attempt_index,
            "prior_prompt_digest": self.prior_prompt_digest,
            "generation_receipt_digest": self.generation_receipt_digest,
            "evaluator_receipt_digest": self.evaluator_receipt_digest,
            "candidate_digest": self.candidate_digest,
            "source_digest": self.source_digest,
            "source": self.source,
            "diagnostic": self.diagnostic,
            "public_summary": self.public_summary,
            "previous_feedback_digest": self.previous_feedback_digest,
        }

    def model_visible_record(self) -> Dict[str, Any]:
        self.validate()
        return {**self.core_record(), "evidence_id": self.evidence_id, "feedback_digest": self.feedback_digest}

    def public_redacted_record(self) -> Dict[str, Any]:
        self.validate()
        value = self.core_record()
        value.pop("source")
        value.pop("public_summary")
        value["schema_version"] = "egv-recommissioning-public-feedback-v3"
        record = {
            **value,
            "public_summary_digest": digest_bytes(self.public_summary.encode("utf-8")),
            "evidence_id": self.evidence_id,
            "feedback_digest": self.feedback_digest,
        }
        _scan_public_projection(record, "public feedback projection")
        return record

    def validate(self) -> None:
        for value, label in (
            (self.coordinate_id, "feedback coordinate digest"),
            (self.task_package_digest, "feedback task-package digest"),
            (self.model_profile_digest, "feedback model-profile digest"),
            (self.evaluator_profile_digest, "feedback evaluator-profile digest"),
            (self.prior_prompt_digest, "feedback prior-prompt digest"),
            (self.generation_receipt_digest, "feedback generation-receipt digest"),
            (self.evaluator_receipt_digest, "feedback evaluator-receipt digest"),
            (self.candidate_digest, "feedback candidate digest"),
            (self.source_digest, "feedback source digest"),
            (self.feedback_digest, "feedback digest"),
        ):
            _require_digest(value, label)
        if self.previous_feedback_digest is not None:
            _require_digest(self.previous_feedback_digest, "previous feedback digest")
        if type(self.attempt_index) is not int or not 1 <= self.attempt_index <= 6:
            raise RecommissioningProtocolError("feedback attempt index is outside the frozen budget")
        if self.diagnostic not in DIAGNOSTIC_IDS:
            raise RecommissioningProtocolError("feedback diagnostic is outside the visible enum")
        _require_text(self.source, "feedback source", max_characters=_MAX_SOURCE_CHARACTERS)
        if digest_bytes(self.source.encode("utf-8")) != self.source_digest:
            raise RecommissioningProtocolError("feedback source bytes do not match the bounded digest")
        _reject_sensitive_public_text(self.public_summary, "feedback summary")
        if digest_for(self.core_record()) != self.feedback_digest:
            raise RecommissioningProtocolError("feedback digest does not bind the full lineage record")


def _feedback_records(
    feedback: Sequence[AttemptFeedback],
    *,
    coordinate: PilotCoordinate,
    task: VisibleTaskPackage,
    model_profile_digest: str,
) -> Tuple[Dict[str, Any], ...]:
    records = []
    previous: Optional[str] = None
    for expected_index, item in enumerate(feedback, start=1):
        if not isinstance(item, AttemptFeedback):
            raise RecommissioningProtocolError("feedback lineage contains an untyped record")
        item.validate()
        if (
            item.coordinate_id != coordinate.coordinate_id
            or item.task_package_digest != task.task_package_digest
            or item.model_profile_digest != model_profile_digest
            or item.evaluator_profile_digest != coordinate.evaluator_profile_digest
        ):
            raise RecommissioningProtocolError("feedback crossed an arm/task/seed/model coordinate boundary")
        if item.attempt_index != expected_index or item.previous_feedback_digest != previous:
            raise RecommissioningProtocolError("feedback is not a contiguous full-digest lineage")
        records.append(item.model_visible_record())
        previous = item.feedback_digest
    return tuple(records)


def _canonical_prompt_material(
    coordinate: PilotCoordinate,
    task: VisibleTaskPackage,
    *,
    model_profile_digest: str,
    budget: TokenBudget,
    attempt_index: int,
    feedback: Sequence[AttemptFeedback] = (),
) -> Tuple[Tuple[str, ...], Tuple[ChatMessage, ...]]:
    """Derive the only accepted model-visible messages from validated inputs."""
    coordinate.validate()
    task.validate(coordinate)
    _require_digest(model_profile_digest, "model-profile digest")
    budget.validate()
    if budget.seed != coordinate.seed:
        raise RecommissioningProtocolError("budget seed differs from the coordinate seed")
    if type(attempt_index) is not int or not 1 <= attempt_index <= 6:
        raise RecommissioningProtocolError("attempt index is outside the frozen six-candidate budget")
    records = _feedback_records(
        feedback,
        coordinate=coordinate,
        task=task,
        model_profile_digest=model_profile_digest,
    )
    if len(records) != attempt_index - 1:
        raise RecommissioningProtocolError("feedback lineage does not match the next attempt")
    if coordinate.protocol_id == "P0" and records:
        latest = dict(records[-1])
        # P0 is the static control: it receives the immediately preceding
        # candidate/diagnostic only, not even the prior-lineage pointer.
        latest.pop("previous_feedback_digest", None)
        visible_records = (latest,)
    else:
        visible_records = records
    visible_ids = tuple(sorted(record["evidence_id"] for record in visible_records))
    system = (
        "You are a bounded code-repair variation operator. The structured public specification is "
        "authoritative. Return one schema-constrained candidate only. Never emit tool calls or request "
        "hidden tests, expected values, credentials, network access, filesystem discovery, or evaluator authority."
    )
    protocol = {
        "P0": (
            "Static control: propose one complete repair from the current public task and, when present, "
            "only the immediately preceding bounded diagnostic. Do not infer persistent memory."
        ),
        "P1": (
            "AVO-inspired treatment: inspect the same-coordinate full-digest lineage, avoid repeated source "
            "digests, state a short falsifiable change summary, and choose PROPOSE, EDIT, CRITIQUE, or COMMIT. "
            "The supervisor is deterministic and supplies no extra model call or private information."
        ),
    }[coordinate.protocol_id]
    user_value = {
        "schema_version": "egv-recommissioning-model-visible-prompt-v2",
        "coordinate": {**coordinate.identity_record(), "coordinate_id": coordinate.coordinate_id},
        "task": task.model_visible_record(),
        "task_package_digest": task.task_package_digest,
        "model_profile_digest": model_profile_digest,
        "evaluator_profile_digest": coordinate.evaluator_profile_digest,
        "budget_digest": budget.digest,
        "attempt_index": attempt_index,
        "visible_feedback": list(visible_records),
        "response_contract": {
            "fields": ["action", "source", "summary", "evidence_ids"],
            "source_contract": "pure-return-v1 complete Python module",
            "public_locus": task.public_locus,
            "evidence_ids": list(visible_ids),
            "tool_calls": "forbidden",
        },
    }
    messages = (
        ChatMessage("system", system),
        ChatMessage("user", protocol + "\n" + canonical_json(user_value)),
    )
    return visible_ids, messages


def render_prompt(
    coordinate: PilotCoordinate,
    task: VisibleTaskPackage,
    *,
    model_profile_digest: str,
    budget: TokenBudget,
    attempt_index: int,
    feedback: Sequence[AttemptFeedback] = (),
) -> PromptEnvelope:
    """Render a frozen P0/P1 prompt from one exact coordinate and lineage."""

    return PromptEnvelope._build(
        coordinate=coordinate,
        task=task,
        model_profile_digest=model_profile_digest,
        budget=budget,
        attempt_index=attempt_index,
        feedback=feedback,
    )


_EVALUATOR_RECEIPT_FIELDS = {
    "schema_version",
    "coordinate_id",
    "task_package_digest",
    "model_profile_digest",
    "evaluator_profile_digest",
    "attempt_index",
    "prior_prompt_digest",
    "generation_receipt_digest",
    "candidate_digest",
    "source_digest",
    "diagnostic",
    "decision",
    "public_summary_digest",
    "signing_key_id",
    "receipt_digest",
    "signature",
}


@dataclass(frozen=True, init=False)
class SignedEvaluatorReceipt:
    """Closed Ed25519 evaluator verdict bound to one generated candidate."""

    record_json: str
    public_key_raw: bytes = field(repr=False)

    @classmethod
    def build(
        cls,
        *,
        coordinate_id: str,
        task_package_digest: str,
        model_profile_digest: str,
        evaluator_profile_digest: str,
        attempt_index: int,
        prior_prompt_digest: str,
        generation_receipt_digest: str,
        candidate_digest: str,
        source_digest: str,
        diagnostic: str,
        public_summary: str,
        signer: Any,
    ) -> "SignedEvaluatorReceipt":
        public_key_raw, signing_key_id = _signer_identity(signer, "evaluator receipt signer")
        _reject_sensitive_public_text(public_summary, "evaluator public summary")
        unsigned = {
            "schema_version": "egv-recommissioning-signed-evaluator-receipt-v1",
            "coordinate_id": coordinate_id,
            "task_package_digest": task_package_digest,
            "model_profile_digest": model_profile_digest,
            "evaluator_profile_digest": evaluator_profile_digest,
            "attempt_index": attempt_index,
            "prior_prompt_digest": prior_prompt_digest,
            "generation_receipt_digest": generation_receipt_digest,
            "candidate_digest": candidate_digest,
            "source_digest": source_digest,
            "diagnostic": diagnostic,
            "decision": "PASS" if diagnostic == "PASS" else "FAIL",
            "public_summary_digest": digest_bytes(public_summary.encode("utf-8")),
            "signing_key_id": signing_key_id,
        }
        receipt_digest = digest_for(unsigned)
        signable = {**unsigned, "receipt_digest": receipt_digest}
        signature = signer.sign_bytes(canonical_bytes(signable))
        return cls.from_record({**signable, "signature": signature}, public_key=public_key_raw)

    @classmethod
    def from_record(cls, record: Mapping[str, Any], *, public_key: Any) -> "SignedEvaluatorReceipt":
        normalized = cls.validate_record(record, public_key=public_key)
        raw, _key_id = _public_key_identity(public_key, "evaluator receipt public key")
        result = object.__new__(cls)
        object.__setattr__(result, "record_json", canonical_json(normalized))
        object.__setattr__(result, "public_key_raw", raw)
        return result

    @classmethod
    def validate_record(cls, record: Mapping[str, Any], *, public_key: Any) -> Dict[str, Any]:
        if not isinstance(record, Mapping) or set(record) != _EVALUATOR_RECEIPT_FIELDS:
            raise RecommissioningProtocolError("evaluator receipt is not a closed signed record")
        if record.get("schema_version") != "egv-recommissioning-signed-evaluator-receipt-v1":
            raise RecommissioningProtocolError("evaluator receipt schema is unsupported")
        for field_name in (
            "coordinate_id",
            "task_package_digest",
            "model_profile_digest",
            "evaluator_profile_digest",
            "prior_prompt_digest",
            "generation_receipt_digest",
            "candidate_digest",
            "source_digest",
            "public_summary_digest",
            "receipt_digest",
        ):
            _require_digest(record[field_name], "evaluator receipt " + field_name)
        if type(record["attempt_index"]) is not int or not 1 <= record["attempt_index"] <= 6:
            raise RecommissioningProtocolError("evaluator receipt attempt index is outside the frozen budget")
        if record["diagnostic"] not in DIAGNOSTIC_IDS:
            raise RecommissioningProtocolError("evaluator receipt diagnostic is outside the closed enum")
        expected_decision = "PASS" if record["diagnostic"] == "PASS" else "FAIL"
        if record["decision"] != expected_decision:
            raise RecommissioningProtocolError("evaluator receipt decision disagrees with its diagnostic")
        unsigned = {key: value for key, value in record.items() if key not in {"receipt_digest", "signature"}}
        if digest_for(unsigned) != record["receipt_digest"]:
            raise RecommissioningProtocolError("evaluator receipt digest does not bind the signed record")
        signable = {key: value for key, value in record.items() if key != "signature"}
        _verify_ed25519(
            public_key=public_key,
            key_id=record["signing_key_id"],
            signature=record["signature"],
            payload=signable,
            label="evaluator receipt",
        )
        return canonical_value(record)

    @property
    def receipt_digest(self) -> str:
        return self.public_record()["receipt_digest"]

    def public_record(self) -> Dict[str, Any]:
        return self.validate_record(json.loads(self.record_json), public_key=self.public_key_raw)


def lineage_feedback(
    *,
    coordinate: PilotCoordinate,
    task: VisibleTaskPackage,
    model_profile_digest: str,
    prior_prompt: PromptEnvelope,
    generation_receipt: Mapping[str, Any],
    generation_receipt_public_key: Any,
    service_attestation: Mapping[str, Any],
    controller_public_key: Any,
    evaluator_receipt: Mapping[str, Any],
    evaluator_receipt_public_key: Any,
    candidate: Any,
    diagnostic: str,
    public_summary: str,
    previous_feedback: Optional[AttemptFeedback] = None,
) -> AttemptFeedback:
    """Bind visible feedback to exact prompt, generation, evaluation, and source."""

    # Local import prevents a module cycle while still requiring the concrete
    # client types at the trust boundary.
    from .client import PublicGenerationReceipt, ServiceAttestation, TypedCandidateResponse

    coordinate.validate()
    task.validate(coordinate)
    _require_digest(model_profile_digest, "model-profile digest")
    prior_prompt.validate(
        coordinate=coordinate,
        task=task,
        model_profile_digest=model_profile_digest,
    )
    if not isinstance(candidate, TypedCandidateResponse):
        raise RecommissioningProtocolError("feedback candidate is not a typed response")
    candidate.validate(public_locus=task.public_locus, visible_evidence_ids=prior_prompt.visible_evidence_ids)
    public_candidate = candidate.public_redacted_record()
    candidate_digest = digest_for(public_candidate)
    receipt = PublicGenerationReceipt.validate_record(
        generation_receipt,
        public_key=generation_receipt_public_key,
    )
    if receipt["signing_key_id"] != coordinate.generation_receipt_key_id:
        raise RecommissioningProtocolError("generation receipt key differs from the frozen coordinate")
    try:
        attestation = ServiceAttestation.from_record(
            service_attestation,
            controller_public_key=controller_public_key,
        )
        attestation.validate()
    except Exception as exc:
        raise RecommissioningProtocolError("generation lineage lacks a controller-rooted service attestation") from exc
    generation_public_key_raw, generation_key_id = _public_key_identity(
        generation_receipt_public_key,
        "generation receipt public key",
    )
    if (
        generation_public_key_raw != attestation.generation_receipt_public_key_raw
        or generation_key_id != attestation.generation_receipt_key_id
        or attestation.generation_receipt_key_id != coordinate.generation_receipt_key_id
        or attestation.model_profile_digest != model_profile_digest
    ):
        raise RecommissioningProtocolError("generation lineage crossed its controller-rooted service authority")
    for field_name, expected in (
        ("coordinate_id", coordinate.coordinate_id),
        ("model_bundle", coordinate.model_bundle),
        ("task_ref", public_task_ref(task.task_package_digest)),
        ("public_locus", task.public_locus),
        ("task_adapter_kind", task.adapter_kind),
        ("task_package_digest", task.task_package_digest),
        ("model_profile_digest", model_profile_digest),
        ("service_attestation_digest", attestation.attestation_digest),
        ("service_observation_digest", attestation.service_observation["observation_digest"]),
        ("controller_key_id", attestation.launch_record["controller_key_id"]),
        ("service_key_id", attestation.launch_record["service_key_id"]),
        ("generation_receipt_key_id", attestation.generation_receipt_key_id),
        ("runtime_config_digest", attestation.runtime_config_digest),
        ("launch_receipt_digest", attestation.launch_receipt_digest),
        ("budget_digest", prior_prompt.budget_digest),
        ("seed", prior_prompt.seed),
        ("attempt_index", prior_prompt.attempt_index),
        ("prompt_digest", prior_prompt.prompt_digest),
        ("candidate_digest", candidate_digest),
        ("source_digest", candidate.source_digest),
    ):
        if receipt[field_name] != expected:
            raise RecommissioningProtocolError("generation receipt differs from feedback {} binding".format(field_name))
    evaluator = SignedEvaluatorReceipt.validate_record(
        evaluator_receipt,
        public_key=evaluator_receipt_public_key,
    )
    if evaluator["signing_key_id"] != coordinate.evaluator_key_id:
        raise RecommissioningProtocolError("evaluator receipt key differs from the frozen coordinate")
    evaluator_expected = {
        "coordinate_id": coordinate.coordinate_id,
        "task_package_digest": task.task_package_digest,
        "model_profile_digest": model_profile_digest,
        "evaluator_profile_digest": coordinate.evaluator_profile_digest,
        "attempt_index": prior_prompt.attempt_index,
        "prior_prompt_digest": prior_prompt.prompt_digest,
        "generation_receipt_digest": receipt["receipt_digest"],
        "candidate_digest": candidate_digest,
        "source_digest": candidate.source_digest,
        "diagnostic": diagnostic,
        "decision": "PASS" if diagnostic == "PASS" else "FAIL",
        "public_summary_digest": digest_bytes(public_summary.encode("utf-8")),
    }
    for field_name, expected in evaluator_expected.items():
        if evaluator[field_name] != expected:
            raise RecommissioningProtocolError("evaluator receipt differs from feedback {} binding".format(field_name))
    if previous_feedback is not None:
        previous_feedback.validate()
        if previous_feedback.coordinate_id != coordinate.coordinate_id:
            raise RecommissioningProtocolError("previous feedback crossed a coordinate boundary")
        if previous_feedback.attempt_index != prior_prompt.attempt_index - 1:
            raise RecommissioningProtocolError("previous feedback is not the immediate lineage parent")
        previous_digest = previous_feedback.feedback_digest
    else:
        if prior_prompt.attempt_index != 1:
            raise RecommissioningProtocolError("later feedback requires its immediate lineage parent")
        previous_digest = None
    values = {
        "coordinate_id": coordinate.coordinate_id,
        "task_package_digest": task.task_package_digest,
        "model_profile_digest": model_profile_digest,
        "evaluator_profile_digest": coordinate.evaluator_profile_digest,
        "attempt_index": prior_prompt.attempt_index,
        "prior_prompt_digest": prior_prompt.prompt_digest,
        "generation_receipt_digest": receipt["receipt_digest"],
        "evaluator_receipt_digest": digest_for(evaluator),
        "candidate_digest": candidate_digest,
        "source_digest": candidate.source_digest,
        "source": candidate.source,
        "diagnostic": diagnostic,
        "public_summary": public_summary,
        "previous_feedback_digest": previous_digest,
    }
    core = {"schema_version": "egv-recommissioning-feedback-v2", **values}
    values["feedback_digest"] = digest_for(core)
    result = object.__new__(AttemptFeedback)
    for field_name in AttemptFeedback.__dataclass_fields__:  # type: ignore[attr-defined]
        object.__setattr__(result, field_name, values[field_name])
    result.validate()
    return result


__all__ = [
    "AttemptFeedback",
    "ChatMessage",
    "DIAGNOSTIC_IDS",
    "GenerationBudget",
    "MODEL_BUNDLE_IDS",
    "PROTOCOL_IDS",
    "PilotCoordinate",
    "PromptEnvelope",
    "RecommissioningProtocolError",
    "SignedEvaluatorReceipt",
    "TokenBudget",
    "VisibleTaskPackage",
    "lineage_feedback",
    "public_task_ref",
    "render_prompt",
]
