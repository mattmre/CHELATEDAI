"""Sealed command bridge for an independent production Variation evaluator."""

from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..canonical import (
    GENESIS_HASH,
    canonical_bytes,
    canonical_json,
    content_id,
    digest_bytes,
    digest_for,
    failure_family_root,
    validate_sha256,
)
from ..evaluation.controller import EvaluationResult, EvaluatorController, HiddenEvaluatorRunner
from ..evaluation.diagnostics import Diagnostic, validate_diagnostic, validate_disposition, validate_resource_bucket
from ..ledger import EvidenceLedger
from ..receipts import ReceiptJournal, ReceiptSigner, key_id_for_public_key, load_public_key, receipt_hash, verify_receipt
from .errors import VariationConfigurationError, VariationDependencyError


REMOTE_VARIATION_SERVICE_SCHEMA = "egv-remote-variation-service-v1"
REMOTE_VARIATION_REQUEST_SCHEMA = "egv-remote-variation-request-v1"
REMOTE_VARIATION_RESPONSE_SCHEMA = "egv-remote-variation-response-v1"
REMOTE_VARIATION_TIMEOUT_SECONDS = 600
REMOTE_VARIATION_REQUEST_LIMIT = 512 * 1024
REMOTE_VARIATION_RESPONSE_LIMIT = 1024 * 1024
REMOTE_VARIATION_STATE_SCHEMA = "egv-remote-variation-state-v1"

_TASK_FIELDS = frozenset({"template_id", "family_id", "split", "ordinal", "source_digest", "public_rule_id", "public_locus"})
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "model_digest",
        "protocol_digest",
        "policy_digest",
        "data_manifest_digest",
        "task_manifest_digest",
        "task_bindings",
        "evaluator_revision",
        "evaluator_digest",
        "docker_image_digest",
        "docker_config_digest",
        "authority_policy_digest",
        "command_digest",
        "evaluator_key_id",
        "evaluator_public_key_digest",
        "service_manifest_digest",
    }
)
_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "operation_digest",
        "request_digest",
        "service_manifest_digest",
        "campaign_id",
        "model_digest",
        "protocol_digest",
        "policy_digest",
        "run_id",
        "arm_policy_digest",
        "data_manifest_digest",
        "task_manifest_digest",
        "evaluator_digest",
        "docker_image_digest",
        "candidate_id",
        "task_id",
        "public_task_binding",
        "candidate_artifact_digest",
        "candidate_source_b64",
        "requested_authority",
        "declared_locus",
        "receipt_sequence_start",
        "previous_receipt_hash",
    }
)
_RESPONSE_FIELDS = frozenset(
    {
        "schema_version",
        "operation_digest",
        "request_digest",
        "service_manifest_digest",
        "result",
        "receipts",
        "signing_key_id",
        "signature",
    }
)


def _require_digest(value: Any, name: str) -> str:
    try:
        return validate_sha256(value, name)
    except Exception as exc:
        raise VariationConfigurationError(str(exc)) from exc


def _decode_b64(value: Any, name: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise VariationConfigurationError("{} must be non-empty base64url".format(name))
    try:
        padded = value + "=" * (-len(value) % 4)
        return base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)
    except (ValueError, UnicodeError) as exc:
        raise VariationConfigurationError("{} is not valid base64url".format(name)) from exc


def _encode_b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


@contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write("0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _run_bounded_command(invocation: Sequence[str], request_text: str) -> Tuple[int, bytes, bytes]:
    """Execute with hard in-memory stdout/stderr caps and a frozen timeout."""

    try:
        process = subprocess.Popen(
            list(invocation),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise VariationDependencyError("remote Variation evaluator invocation failed") from exc
    stdout = bytearray()
    stderr = bytearray()
    overflow = []

    def drain(stream: Any, sink: bytearray, label: str) -> None:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            if len(sink) + len(chunk) > REMOTE_VARIATION_RESPONSE_LIMIT:
                overflow.append(label)
                process.kill()
                break
            sink.extend(chunk)

    threads = (
        threading.Thread(target=drain, args=(process.stdout, stdout, "stdout"), daemon=True),
        threading.Thread(target=drain, args=(process.stderr, stderr, "stderr"), daemon=True),
    )
    for thread in threads:
        thread.start()
    try:
        assert process.stdin is not None
        process.stdin.write(request_text.encode("utf-8"))
        process.stdin.close()
        process.wait(timeout=REMOTE_VARIATION_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.wait()
        raise VariationDependencyError("remote Variation evaluator timed out") from exc
    finally:
        for thread in threads:
            thread.join(timeout=5)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
    if overflow:
        raise VariationDependencyError("remote Variation evaluator {} exceeded the bounded output limit".format(overflow[0]))
    return int(process.returncode), bytes(stdout), bytes(stderr)


def _validate_task_binding(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TASK_FIELDS:
        raise VariationConfigurationError("remote evaluator public task binding is not closed")
    result = dict(value)
    for field in ("template_id", "family_id", "split", "public_rule_id", "public_locus"):
        if not isinstance(result[field], str) or not result[field]:
            raise VariationConfigurationError("remote evaluator public task binding is malformed")
    if result["split"] not in {"train", "heldout"}:
        raise VariationConfigurationError("remote evaluator may expose only frozen train or held-out public task bindings")
    if not isinstance(result["ordinal"], int) or isinstance(result["ordinal"], bool) or result["ordinal"] < 1:
        raise VariationConfigurationError("remote evaluator public task ordinal is invalid")
    _require_digest(result["source_digest"], "task source digest")
    return result


class RemoteEvaluatorServiceManifest:
    """Closed, self-digesting trust manifest for one evaluator command."""

    def __init__(self, value: Mapping[str, Any]) -> None:
        if not isinstance(value, Mapping) or set(value) != _MANIFEST_FIELDS:
            raise VariationConfigurationError("remote evaluator service manifest is not closed")
        normalized = dict(value)
        supplied = normalized.pop("service_manifest_digest")
        if normalized.get("schema_version") != REMOTE_VARIATION_SERVICE_SCHEMA or digest_for(normalized) != supplied:
            raise VariationConfigurationError("remote evaluator service manifest digest is invalid")
        for field in (
            "model_digest",
            "protocol_digest",
            "policy_digest",
            "data_manifest_digest",
            "task_manifest_digest",
            "evaluator_digest",
            "docker_config_digest",
            "authority_policy_digest",
            "command_digest",
            "evaluator_public_key_digest",
        ):
            _require_digest(value[field], field)
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(value["docker_image_digest"])):
            raise VariationConfigurationError("remote evaluator Docker image is not pinned by SHA-256 ID")
        if not isinstance(value["campaign_id"], str) or not value["campaign_id"]:
            raise VariationConfigurationError("remote evaluator campaign ID is missing")
        if not isinstance(value["evaluator_revision"], str) or not value["evaluator_revision"]:
            raise VariationConfigurationError("remote evaluator revision is missing")
        if value["evaluator_digest"] != digest_for(value["evaluator_revision"]):
            raise VariationConfigurationError("remote evaluator revision digest is invalid")
        tasks = value["task_bindings"]
        if not isinstance(tasks, list) or not tasks:
            raise VariationConfigurationError("remote evaluator task registry is empty")
        bindings = [_validate_task_binding(item) for item in tasks]
        if [item["template_id"] for item in bindings] != sorted(set(item["template_id"] for item in bindings)):
            raise VariationConfigurationError("remote evaluator task registry is not unique and sorted")
        if digest_for(bindings) != value["task_manifest_digest"]:
            raise VariationConfigurationError("remote evaluator task registry digest is invalid")
        self._value = dict(value)
        self._tasks = {item["template_id"]: item for item in bindings}

    @classmethod
    def from_path(cls, path: Path) -> "RemoteEvaluatorServiceManifest":
        target = Path(path)
        if target.is_symlink() or not target.is_file():
            raise VariationDependencyError("remote evaluator service manifest must be a regular file")
        try:
            value = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise VariationConfigurationError("remote evaluator service manifest cannot be decoded") from exc
        return cls(value)

    @property
    def digest(self) -> str:
        return self._value["service_manifest_digest"]

    def __getitem__(self, name: str) -> Any:
        return self._value[name]

    def public_record(self, task_id: str) -> Optional[Dict[str, Any]]:
        value = self._tasks.get(task_id)
        return dict(value) if value is not None else None

    def validate_integrity(self) -> None:
        current = dict(self._value)
        supplied = current.pop("service_manifest_digest", None)
        if supplied != digest_for(current):
            raise VariationDependencyError("remote evaluator service manifest changed after validation")
        bindings = [_validate_task_binding(item) for item in current["task_bindings"]]
        if digest_for(bindings) != current["task_manifest_digest"]:
            raise VariationDependencyError("remote evaluator task registry changed after validation")


def build_remote_evaluator_service_manifest(
    *,
    campaign_id: str,
    model_digest: str,
    protocol_digest: str,
    policy_digest: str,
    corpus: Any,
    evaluator_revision: str,
    public_key_path: Path,
    command: Path,
    docker_config: Any,
) -> Dict[str, Any]:
    """Freeze a public-only service manifest from evaluator-owned authority."""

    from ..evaluation.authority import AuthorityPolicy
    from ..evaluation.dataset import EvaluationCorpus
    from ..evaluation.sandbox import DockerSandboxConfig

    if type(corpus) is not EvaluationCorpus:
        raise VariationDependencyError("remote service freeze requires the exact evaluator corpus")
    if type(docker_config) is not DockerSandboxConfig:
        raise VariationDependencyError("remote service freeze requires the exact Docker configuration")
    authority_policy = AuthorityPolicy.candidate_execution()
    if policy_digest != authority_policy.digest:
        raise VariationConfigurationError("remote service policy differs from candidate-execution authority")
    public_key = Path(public_key_path)
    endpoint = Path(command)
    if any(path.is_symlink() or not path.is_file() for path in (public_key, endpoint)):
        raise VariationDependencyError("remote service public key and command must be regular files")
    image_id = docker_config.verify_image()
    if image_id != docker_config.pinned_image_id:
        raise VariationConfigurationError("remote service Docker inspection differs from its pinned image")
    task_bindings = sorted(
        (
            repo.public_manifest_record()
            for repo in corpus.repositories
            if repo.split in {"train", "heldout"}
        ),
        key=lambda item: item["template_id"],
    )
    public_key_bytes = public_key.read_bytes()
    endpoint_bytes = endpoint.read_bytes()
    unsigned = {
        "schema_version": REMOTE_VARIATION_SERVICE_SCHEMA,
        "campaign_id": campaign_id,
        "model_digest": _require_digest(model_digest, "model digest"),
        "protocol_digest": _require_digest(protocol_digest, "protocol digest"),
        "policy_digest": policy_digest,
        "data_manifest_digest": corpus.manifest_digest(),
        "task_manifest_digest": digest_for(task_bindings),
        "task_bindings": task_bindings,
        "evaluator_revision": evaluator_revision,
        "evaluator_digest": digest_for(evaluator_revision),
        "docker_image_digest": docker_config.pinned_image_id,
        "docker_config_digest": digest_for(dict(docker_config.__dict__)),
        "authority_policy_digest": authority_policy.digest,
        "command_digest": hashlib.sha256(endpoint_bytes).hexdigest(),
        "evaluator_key_id": key_id_for_public_key(public_key_bytes),
        "evaluator_public_key_digest": hashlib.sha256(public_key_bytes).hexdigest(),
    }
    result = {**unsigned, "service_manifest_digest": digest_for(unsigned)}
    RemoteEvaluatorServiceManifest(result)
    return result


def _operation_digest(request: Mapping[str, Any]) -> str:
    stable = dict(request)
    for field in ("request_digest", "operation_digest", "receipt_sequence_start", "previous_receipt_hash"):
        stable.pop(field, None)
    return digest_for(stable)


def _verify_response_envelope(
    response: Mapping[str, Any], manifest: RemoteEvaluatorServiceManifest, public_key: Any
) -> Dict[str, Any]:
    if set(response) != _RESPONSE_FIELDS or response.get("schema_version") != REMOTE_VARIATION_RESPONSE_SCHEMA:
        raise VariationConfigurationError("remote Variation response is not closed")
    if response.get("service_manifest_digest") != manifest.digest:
        raise VariationConfigurationError("remote Variation response has a stale service binding")
    if response.get("signing_key_id") != manifest["evaluator_key_id"]:
        raise VariationConfigurationError("remote Variation response uses the wrong evaluator key")
    unsigned = dict(response)
    signature = _decode_b64(unsigned.pop("signature"), "response signature")
    try:
        load_public_key(public_key).verify(signature, canonical_bytes(unsigned))
    except Exception as exc:
        raise VariationConfigurationError("remote Variation response signature is invalid") from exc
    return dict(response)


def _empty_remote_state(manifest: RemoteEvaluatorServiceManifest) -> Dict[str, Any]:
    body = {
        "schema_version": REMOTE_VARIATION_STATE_SCHEMA,
        "service_manifest_digest": manifest.digest,
        "evaluator_key_id": manifest["evaluator_key_id"],
        "next_sequence": 1,
        "receipt_head": GENESIS_HASH,
        "operation_order": [],
        "responses": {},
        "pending_operation": None,
    }
    return {**body, "state_digest": digest_for(body)}


def _load_remote_state(
    path: Path, manifest: RemoteEvaluatorServiceManifest, public_key: Any
) -> Dict[str, Any]:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise VariationDependencyError("remote evaluator has an ambiguous interrupted state commit")
    if not path.exists():
        return _empty_remote_state(manifest)
    if path.is_symlink() or not path.is_file():
        raise VariationDependencyError("remote evaluator state must be a regular file")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VariationDependencyError("remote evaluator state cannot be decoded") from exc
    fields = {
        "schema_version", "service_manifest_digest", "evaluator_key_id", "next_sequence",
        "receipt_head", "operation_order", "responses", "pending_operation", "state_digest",
    }
    if not isinstance(state, Mapping) or set(state) != fields:
        raise VariationDependencyError("remote evaluator state is not closed")
    body = dict(state)
    supplied = body.pop("state_digest")
    if supplied != digest_for(body):
        raise VariationDependencyError("remote evaluator state digest is invalid")
    if (
        body["schema_version"] != REMOTE_VARIATION_STATE_SCHEMA
        or body["service_manifest_digest"] != manifest.digest
        or body["evaluator_key_id"] != manifest["evaluator_key_id"]
    ):
        raise VariationDependencyError("remote evaluator state authority binding is stale")
    order = body["operation_order"]
    responses = body["responses"]
    if not isinstance(order, list) or order != list(dict.fromkeys(order)) or not isinstance(responses, Mapping):
        raise VariationDependencyError("remote evaluator operation cache is malformed")
    if set(order) != set(responses):
        raise VariationDependencyError("remote evaluator operation cache index differs from its responses")
    pending = body["pending_operation"]
    if pending is not None:
        pending_fields = {
            "operation_digest", "request_digest", "receipt_sequence_start", "previous_receipt_hash"
        }
        if not isinstance(pending, Mapping) or set(pending) != pending_fields:
            raise VariationDependencyError("remote evaluator pending operation is malformed")
        _require_digest(pending["operation_digest"], "pending operation digest")
        _require_digest(pending["request_digest"], "pending request digest")
        if (
            pending["receipt_sequence_start"] != body["next_sequence"]
            or pending["previous_receipt_hash"] != body["receipt_head"]
            or pending["operation_digest"] in responses
        ):
            raise VariationDependencyError("remote evaluator pending operation anchor is inconsistent")
    sequence = 1
    previous = GENESIS_HASH
    for operation in order:
        response = _verify_response_envelope(responses[operation], manifest, public_key)
        if response["operation_digest"] != operation:
            raise VariationDependencyError("remote evaluator cached operation binding is invalid")
        receipts = response["receipts"]
        if not isinstance(receipts, list) or not receipts:
            raise VariationDependencyError("remote evaluator cached receipt suffix is empty")
        for receipt in receipts:
            previous = verify_receipt(
                receipt,
                public_key,
                expected_key_id=manifest["evaluator_key_id"],
                expected_sequence=sequence,
                expected_previous_hash=previous,
            )
            sequence += 1
    if body["next_sequence"] != sequence or body["receipt_head"] != previous:
        raise VariationDependencyError("remote evaluator cached receipt head is inconsistent")
    return dict(state)


def _write_remote_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(state) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))
    if os.name != "nt":
        directory = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _durable_remote_response(
    request: Mapping[str, Any],
    *,
    manifest: RemoteEvaluatorServiceManifest,
    signer: ReceiptSigner,
    state_root: Path,
    build_response: Any,
    validate_response: Any = None,
) -> Mapping[str, Any]:
    root = Path(state_root)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "remote-evaluator-state.json"
    with _exclusive_lock(root / "remote-evaluator-state.lock"):
        state = _load_remote_state(state_path, manifest, signer.public_key)
        operation = request["operation_digest"]
        cached = state["responses"].get(operation)
        if cached is not None:
            if validate_response is not None:
                validate_response(cached)
            return dict(cached)
        pending = state["pending_operation"]
        if pending is not None:
            raise VariationDependencyError(
                "remote evaluator is quarantined after an ambiguous interrupted execution"
            )
        if (
            request["receipt_sequence_start"] != state["next_sequence"]
            or request["previous_receipt_hash"] != state["receipt_head"]
        ):
            raise VariationConfigurationError("remote evaluator rejected a stale or forked receipt anchor")
        pending_body = {key: value for key, value in state.items() if key != "state_digest"}
        pending_body["pending_operation"] = {
            "operation_digest": operation,
            "request_digest": request["request_digest"],
            "receipt_sequence_start": request["receipt_sequence_start"],
            "previous_receipt_hash": request["previous_receipt_hash"],
        }
        pending_state = {**pending_body, "state_digest": digest_for(pending_body)}
        # The execution intent is durable before invoking Docker. A crash after this
        # point deliberately quarantines the evaluator instead of re-executing an
        # action whose effects cannot be proven absent.
        _write_remote_state(state_path, pending_state)
        response = _verify_response_envelope(build_response(), manifest, signer.public_key)
        if response["operation_digest"] != operation or response["request_digest"] != request["request_digest"]:
            raise VariationConfigurationError("remote evaluator response is not bound to the current operation")
        sequence = state["next_sequence"]
        previous = state["receipt_head"]
        for receipt in response["receipts"]:
            previous = verify_receipt(
                receipt,
                signer.public_key,
                expected_key_id=manifest["evaluator_key_id"],
                expected_sequence=sequence,
                expected_previous_hash=previous,
            )
            sequence += 1
        if validate_response is not None:
            validate_response(response)
        next_body = {key: value for key, value in state.items() if key != "state_digest"}
        next_body["next_sequence"] = sequence
        next_body["receipt_head"] = previous
        next_body["operation_order"] = list(next_body["operation_order"]) + [operation]
        next_body["responses"] = {**dict(next_body["responses"]), operation: response}
        next_body["pending_operation"] = None
        next_state = {**next_body, "state_digest": digest_for(next_body)}
        _write_remote_state(state_path, next_state)
        return response


def _validate_remote_result_semantics(
    *,
    result_value: Any,
    receipts: Sequence[Mapping[str, Any]],
    manifest: RemoteEvaluatorServiceManifest,
    task_binding: Mapping[str, Any],
    candidate_id: str,
    task_id: str,
    artifact_digest: str,
    declared_locus: str,
    run_id: str,
    arm_policy_digest: str,
) -> EvaluationResult:
    """Validate the complete signed controller contract without mutating the ledger."""

    if not isinstance(result_value, Mapping):
        raise VariationConfigurationError("remote Variation result is not an object")
    try:
        normalized = dict(result_value)
        normalized["receipt_ids"] = tuple(normalized["receipt_ids"])
        result = EvaluationResult(**normalized)
        validate_diagnostic(result.diagnostic_enum)
        validate_resource_bucket(result.resource_bucket)
        validate_disposition(result.disposition)
        _require_digest(result.candidate_artifact_digest, "result candidate artifact digest")
        _require_digest(result.output_digest, "result output digest")
    except (KeyError, TypeError, ValueError) as exc:
        raise VariationConfigurationError("remote Variation result contract is invalid") from exc
    if type(result.infrastructure_loss) is not bool:
        raise VariationConfigurationError("remote Variation infrastructure flag is invalid")
    receipt_ids = tuple(receipt["receipt_id"] for receipt in receipts)
    if (
        result.candidate_id != candidate_id
        or result.task_id != task_id
        or result.candidate_artifact_digest != artifact_digest
        or result.receipt_ids != receipt_ids
    ):
        raise VariationConfigurationError("remote Variation result differs from its signed request or receipts")
    expected_common = {
        "campaign_id": manifest["campaign_id"],
        "run_id": run_id,
        "task_id": task_id,
        "candidate_id": candidate_id,
        "candidate_artifact_digest": artifact_digest,
        "protocol_digest": manifest["protocol_digest"],
        "policy_digest": manifest["policy_digest"],
        "arm_policy_digest": arm_policy_digest,
        "evaluator_digest": manifest.digest,
        "task_family": task_binding["family_id"],
        "normalized_public_locus": task_binding["public_locus"],
        "public_rule_id": task_binding["public_rule_id"],
    }
    for receipt in receipts:
        for field, expected in expected_common.items():
            if receipt.get(field) != expected:
                raise VariationConfigurationError("remote Variation receipt {} binding is invalid".format(field))
    types = [receipt["receipt_type"] for receipt in receipts]
    if types not in (["AUTHORITY"], ["AUTHORITY", "VERDICT", "EFFECT"]):
        raise VariationConfigurationError("remote Variation receipt chain has an invalid type order")
    authority = receipts[0]
    if authority.get("request_id") != "request-authority-{}".format(candidate_id):
        raise VariationConfigurationError("remote Variation authority request identity is invalid")
    empty_digest = digest_bytes(b"")
    if len(receipts) == 1:
        allowed = {
            Diagnostic.PROTOCOL_VIOLATION.value: "REJECTED",
            Diagnostic.MUTATION_LOCUS_VIOLATION.value: "REJECTED",
            Diagnostic.AUTHORITY_DENIED.value: "ABSTAINED",
            Diagnostic.INTERNAL_ERROR.value: "ABSTAINED",
        }
        if (
            authority.get("decision") != "DENY"
            or result.diagnostic_enum not in allowed
            or result.disposition != allowed[result.diagnostic_enum]
            or result.resource_bucket != "UNDER_25"
            or result.output_digest != empty_digest
        ):
            raise VariationConfigurationError("remote Variation authority-only result is inconsistent")
        receipt_diagnostic = authority.get("diagnostic_enum")
        if result.diagnostic_enum == Diagnostic.AUTHORITY_DENIED.value:
            if receipt_diagnostic is not None:
                raise VariationConfigurationError("authority denial receipt disclosed an invalid diagnostic")
        elif receipt_diagnostic != result.diagnostic_enum:
            raise VariationConfigurationError("authority denial diagnostic differs from its result")
    else:
        verdict, effect = receipts[1:]
        expected_action = digest_for({"action": "execute_candidate", "locus": declared_locus})
        diagnostic = result.diagnostic_enum
        infrastructure = diagnostic == Diagnostic.INTERNAL_ERROR.value
        expected_verdict = "ERROR" if infrastructure else ("PASS" if diagnostic == Diagnostic.PASS.value else "FAIL")
        expected_effect = "ERROR" if infrastructure else "ALLOW"
        expected_disposition = (
            "ABSTAINED" if infrastructure else ("PROMOTED" if diagnostic == Diagnostic.PASS.value else "REJECTED")
        )
        if (
            authority.get("decision") != "ALLOW"
            or verdict.get("request_id") != "request-verdict-{}".format(candidate_id)
            or effect.get("request_id") != "request-effect-{}".format(candidate_id)
            or verdict.get("decision") != expected_verdict
            or effect.get("decision") != expected_effect
            or verdict.get("diagnostic_enum") != diagnostic
            or effect.get("diagnostic_enum") != diagnostic
            or verdict.get("resource_bucket") != result.resource_bucket
            or verdict.get("output_digest") != result.output_digest
            or effect.get("normalized_action_hash") != expected_action
            or result.disposition != expected_disposition
        ):
            raise VariationConfigurationError("remote Variation result disagrees with its signed controller chain")
    incident = result.infrastructure_incident_id
    root = result.failure_family_root
    if result.diagnostic_enum == Diagnostic.INTERNAL_ERROR.value:
        if not result.infrastructure_loss or not incident or not root:
            raise VariationConfigurationError("remote Variation infrastructure loss is incomplete")
        expected_root = failure_family_root(
            task_binding["family_id"],
            result.diagnostic_enum,
            task_binding["public_locus"],
            task_binding["public_rule_id"],
            infrastructure_incident_id=incident,
        )
        if root != expected_root:
            raise VariationConfigurationError("remote Variation infrastructure root is invalid")
        for receipt in receipts:
            if receipt.get("infrastructure_incident_id") != incident or receipt.get("failure_family_root") != root:
                raise VariationConfigurationError("remote Variation receipt incident binding is inconsistent")
    elif result.infrastructure_loss or incident is not None or root is not None:
        raise VariationConfigurationError("remote Variation non-infrastructure result carries an incident")
    return result


class RemoteControllerEvaluationGateway:
    """Exact production client for a separately administered evaluator command."""

    enforceable = True

    def __init__(
        self,
        *,
        ledger: EvidenceLedger,
        manifest_path: Path,
        public_key_path: Path,
        command: Path,
        timeout_seconds: int = REMOTE_VARIATION_TIMEOUT_SECONDS,
    ) -> None:
        if type(self) is not RemoteControllerEvaluationGateway:
            raise VariationDependencyError("remote Variation evaluator type is not frozen")
        if type(ledger) is not EvidenceLedger:
            raise VariationDependencyError("remote Variation evaluator requires the authoritative EvidenceLedger")
        paths = tuple(Path(item) for item in (manifest_path, public_key_path, command))
        if any(path.is_symlink() or not path.is_file() for path in paths):
            raise VariationDependencyError("remote evaluator manifest, key, and command must be regular files")
        if timeout_seconds != REMOTE_VARIATION_TIMEOUT_SECONDS:
            raise VariationConfigurationError("remote evaluator timeout differs from the frozen boundary")
        manifest = RemoteEvaluatorServiceManifest.from_path(paths[0])
        public_key = paths[1].read_bytes()
        if hashlib.sha256(public_key).hexdigest() != manifest["evaluator_public_key_digest"]:
            raise VariationConfigurationError("remote evaluator public key differs from the service manifest")
        if key_id_for_public_key(public_key) != manifest["evaluator_key_id"]:
            raise VariationConfigurationError("remote evaluator key ID differs from its public key")
        command_bytes = paths[2].read_bytes()
        if hashlib.sha256(command_bytes).hexdigest() != manifest["command_digest"]:
            raise VariationConfigurationError("remote evaluator command differs from the service manifest")
        self.ledger = ledger
        self.manifest = manifest
        self._public_key = public_key
        self._command = paths[2].resolve()
        self._command_bytes = command_bytes
        self._command_suffix = paths[2].suffix.lower()
        self.evaluator_revision = manifest["evaluator_revision"]
        self.evaluator_digest = manifest["service_manifest_digest"]
        self.task_registry = manifest
        self._frozen_contract = digest_for(
            {
                "manifest": manifest.digest,
                "key": hashlib.sha256(public_key).hexdigest(),
                "command": hashlib.sha256(command_bytes).hexdigest(),
            }
        )
        self._pinned_ledger = ledger
        self._frozen = True

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("RemoteControllerEvaluationGateway is immutable")
        object.__setattr__(self, name, value)

    def validate_campaign_bindings(
        self,
        *,
        campaign_id: str,
        model_digest: str,
        protocol_digest: str,
        policy_digest: str,
        data_manifest_digest: str,
    ) -> None:
        expected = {
            "campaign_id": campaign_id,
            "model_digest": model_digest,
            "protocol_digest": protocol_digest,
            "policy_digest": policy_digest,
            "data_manifest_digest": data_manifest_digest,
        }
        for field, value in expected.items():
            if self.manifest[field] != value:
                raise VariationConfigurationError("remote evaluator has a stale {} binding".format(field))

    def validate_runtime(self) -> None:
        if type(self) is not RemoteControllerEvaluationGateway:
            raise VariationDependencyError("remote Variation evaluator type changed")
        if "evaluate" in self.__dict__ or "validate_runtime" in self.__dict__:
            raise VariationDependencyError("remote Variation evaluator methods cannot be overridden")
        if type(self).evaluate is not _ORIGINAL_REMOTE_EVALUATE or type(self).validate_runtime is not _ORIGINAL_REMOTE_VALIDATE:
            raise VariationDependencyError("remote Variation evaluator implementation changed")
        if type(self)._invoke is not _ORIGINAL_REMOTE_INVOKE:
            raise VariationDependencyError("remote Variation evaluator command boundary changed")
        if self.ledger is not self._pinned_ledger or type(self.ledger) is not EvidenceLedger:
            raise VariationDependencyError("remote Variation evaluator ledger binding changed")
        self.manifest.validate_integrity()
        current = self._command.read_bytes()
        if current != self._command_bytes or hashlib.sha256(current).hexdigest() != self.manifest["command_digest"]:
            raise VariationDependencyError("remote evaluator command changed after construction")
        if digest_for(
            {
                "manifest": self.manifest.digest,
                "key": hashlib.sha256(self._public_key).hexdigest(),
                "command": hashlib.sha256(current).hexdigest(),
            }
        ) != self._frozen_contract:
            raise VariationDependencyError("remote evaluator frozen contract changed")

    def _invoke(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        self.validate_runtime()
        request_text = canonical_json(request)
        if len(request_text.encode("utf-8")) > REMOTE_VARIATION_REQUEST_LIMIT:
            raise VariationConfigurationError("remote Variation request exceeds the bounded input limit")
        with tempfile.TemporaryDirectory(prefix="egv-pinned-variation-endpoint-") as temporary:
            endpoint = Path(temporary) / (self.manifest["command_digest"] + self._command_suffix)
            with endpoint.open("xb") as handle:
                handle.write(self._command_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            endpoint.chmod(0o500)
            if hashlib.sha256(endpoint.read_bytes()).hexdigest() != self.manifest["command_digest"]:
                raise VariationDependencyError("content-addressed remote evaluator copy failed verification")
            invocation = [sys.executable, str(endpoint)] if self._command_suffix == ".py" else [str(endpoint)]
            returncode, stdout, _stderr = _run_bounded_command(invocation, request_text)
        if returncode != 0:
            raise VariationDependencyError("remote Variation evaluator returned a nonzero exit status")
        try:
            response = json.loads(stdout.decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise VariationDependencyError("remote Variation evaluator returned invalid JSON") from exc
        if not isinstance(response, Mapping):
            raise VariationDependencyError("remote Variation evaluator returned no result object")
        return response

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
        del candidate_source_path
        if opaque_input is not None:
            raise VariationConfigurationError("remote evaluator boundary cannot receive evaluator-private input")
        task_binding = self.manifest.public_record(task_id)
        if task_binding is None:
            raise VariationConfigurationError("remote evaluator task is outside the frozen public registry")
        if task_binding["public_locus"] != declared_locus:
            raise VariationConfigurationError("remote evaluator task locus differs from the public registry")
        source_bytes = bytes(source)
        from .loop import CANDIDATE_SOURCE_LIMIT

        if not source_bytes or len(source_bytes) > CANDIDATE_SOURCE_LIMIT:
            raise VariationConfigurationError("remote Variation candidate source exceeds the frozen byte ceiling")
        artifact_digest = digest_bytes(source_bytes)
        candidate = self.ledger.connection.execute(
            "SELECT campaign_id,run_id,task_id,candidate_json FROM candidates WHERE candidate_id=?",
            (candidate_id,),
        ).fetchone()
        if candidate is None:
            run_id = "run-evaluation"
            arm_policy_digest = digest_for("unbound-remote-evaluation-arm")
        else:
            try:
                candidate_value = json.loads(candidate["candidate_json"])
                if candidate["campaign_id"] != self.manifest["campaign_id"] or candidate["task_id"] != task_id:
                    raise ValueError("candidate authority binding differs")
                if candidate_value["metadata"]["candidate_artifact_digest"] != artifact_digest:
                    raise ValueError("candidate artifact binding differs")
                arm_id = candidate_value["metadata"]["arm_id"]
                from .arms import arm_policy

                arm_policy_digest = arm_policy(str(arm_id)).digest
            except (KeyError, TypeError, ValueError) as exc:
                raise VariationConfigurationError("remote Variation candidate arm binding is invalid") from exc
            run_id = str(candidate["run_id"])
        stable_body = {
            "schema_version": REMOTE_VARIATION_REQUEST_SCHEMA,
            "service_manifest_digest": self.manifest.digest,
            "campaign_id": self.manifest["campaign_id"],
            "model_digest": self.manifest["model_digest"],
            "protocol_digest": self.manifest["protocol_digest"],
            "policy_digest": self.manifest["policy_digest"],
            "run_id": run_id,
            "arm_policy_digest": arm_policy_digest,
            "data_manifest_digest": self.manifest["data_manifest_digest"],
            "task_manifest_digest": self.manifest["task_manifest_digest"],
            "evaluator_digest": self.manifest["evaluator_digest"],
            "docker_image_digest": self.manifest["docker_image_digest"],
            "candidate_id": candidate_id,
            "task_id": task_id,
            "public_task_binding": task_binding,
            "candidate_artifact_digest": artifact_digest,
            "candidate_source_b64": _encode_b64(source_bytes),
            "requested_authority": requested_authority,
            "declared_locus": declared_locus,
        }
        operation = _operation_digest(stable_body)
        body = {
            **stable_body,
            "operation_digest": operation,
            "receipt_sequence_start": self.ledger.receipt_next_sequence(),
            "previous_receipt_hash": self.ledger.receipt_head(),
        }
        request = {**body, "request_digest": digest_for(body)}
        response = dict(self._invoke(request))
        response = _verify_response_envelope(response, self.manifest, self._public_key)
        if response["operation_digest"] != operation:
            raise VariationConfigurationError("remote Variation response has a stale operation binding")
        receipts = response["receipts"]
        if not isinstance(receipts, list) or not receipts:
            raise VariationConfigurationError("remote Variation response receipt chain is empty")
        first = receipts[0]
        if not isinstance(first, Mapping):
            raise VariationConfigurationError("remote Variation receipt is not an object")
        sequence = first.get("sequence")
        previous = first.get("previous_receipt_hash")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
            raise VariationConfigurationError("remote Variation receipt sequence is invalid")
        if previous != GENESIS_HASH:
            _require_digest(previous, "remote Variation previous receipt hash")
        receipt_values = []
        for receipt in receipts:
            if not isinstance(receipt, Mapping):
                raise VariationConfigurationError("remote Variation receipt is not an object")
            receipt_value = dict(receipt)
            complete = verify_receipt(
                receipt_value,
                self._public_key,
                expected_key_id=self.manifest["evaluator_key_id"],
                expected_sequence=sequence,
                expected_previous_hash=previous,
            )
            receipt_values.append(receipt_value)
            previous = complete
            sequence += 1
        result = _validate_remote_result_semantics(
            result_value=response["result"],
            receipts=receipt_values,
            manifest=self.manifest,
            task_binding=task_binding,
            candidate_id=candidate_id,
            task_id=task_id,
            artifact_digest=artifact_digest,
            declared_locus=declared_locus,
            run_id=run_id,
            arm_policy_digest=arm_policy_digest,
        )
        stored = [self.ledger.receipt_by_id(receipt["receipt_id"]) for receipt in receipt_values]
        present = [item is not None for item in stored]
        if any(present) and not all(present):
            raise VariationConfigurationError("remote Variation receipt suffix is only partially present")
        if all(present):
            for existing, receipt in zip(stored, receipt_values):
                assert existing is not None
                if canonical_bytes(existing["receipt"]) != canonical_bytes(receipt):
                    raise VariationConfigurationError("remote Variation cached receipt differs from the ledger")
            return result
        if response["request_digest"] != request["request_digest"]:
            raise VariationConfigurationError("remote Variation response has a stale request binding")
        if (
            receipt_values[0]["previous_receipt_hash"] != request["previous_receipt_hash"]
            or receipt_values[0]["sequence"] != request["receipt_sequence_start"]
            or self.ledger.receipt_head() != request["previous_receipt_hash"]
            or self.ledger.receipt_next_sequence() != request["receipt_sequence_start"]
        ):
            raise VariationConfigurationError("authoritative receipt head changed during remote evaluation")
        self.ledger.ingest_receipts_atomic(receipt_values, self._public_key)
        return result


_ORIGINAL_REMOTE_EVALUATE = RemoteControllerEvaluationGateway.evaluate
_ORIGINAL_REMOTE_VALIDATE = RemoteControllerEvaluationGateway.validate_runtime
_ORIGINAL_REMOTE_INVOKE = RemoteControllerEvaluationGateway._invoke


def run_remote_evaluator_once(
    request: Mapping[str, Any],
    *,
    service_manifest: Path,
    evaluator_seed: Path,
    evaluator_private_key: Path,
    workspace: Path,
    state_root: Path,
) -> Mapping[str, Any]:
    """Run one request with evaluator-owned hidden inputs and Docker authority."""

    if not isinstance(request, Mapping) or set(request) != _REQUEST_FIELDS:
        raise VariationConfigurationError("remote Variation request is not closed")
    request_value = dict(request)
    if len(canonical_bytes(request_value)) > REMOTE_VARIATION_REQUEST_LIMIT:
        raise VariationConfigurationError("remote Variation request exceeds the bounded input limit")
    if request_value.get("operation_digest") != _operation_digest(request_value):
        raise VariationConfigurationError("remote Variation operation digest is invalid")
    unsigned_request = dict(request_value)
    supplied_request_digest = unsigned_request.pop("request_digest")
    if digest_for(unsigned_request) != supplied_request_digest:
        raise VariationConfigurationError("remote Variation request digest is invalid")
    manifest = RemoteEvaluatorServiceManifest.from_path(service_manifest)
    for field in (
        "service_manifest_digest",
        "campaign_id",
        "model_digest",
        "protocol_digest",
        "policy_digest",
        "data_manifest_digest",
        "task_manifest_digest",
        "evaluator_digest",
        "docker_image_digest",
    ):
        expected = manifest.digest if field == "service_manifest_digest" else manifest[field]
        if request_value[field] != expected:
            raise VariationConfigurationError("remote Variation request {} binding is stale".format(field))
    if not isinstance(request_value["run_id"], str) or not request_value["run_id"]:
        raise VariationConfigurationError("remote Variation run binding is invalid")
    _require_digest(request_value["arm_policy_digest"], "remote Variation arm policy digest")
    task_binding = manifest.public_record(str(request_value["task_id"]))
    if task_binding is None or request_value["public_task_binding"] != task_binding:
        raise VariationConfigurationError("remote Variation request task binding is invalid")
    from .loop import CANDIDATE_SOURCE_LIMIT, ControllerEvaluationGateway

    encoded_source = request_value["candidate_source_b64"]
    encoded_limit = ((CANDIDATE_SOURCE_LIMIT + 2) // 3) * 4
    if not isinstance(encoded_source, str) or len(encoded_source) > encoded_limit:
        raise VariationConfigurationError("remote Variation encoded candidate exceeds the frozen byte ceiling")
    source = _decode_b64(encoded_source, "candidate source")
    if len(source) > CANDIDATE_SOURCE_LIMIT or digest_bytes(source) != request_value["candidate_artifact_digest"]:
        raise VariationConfigurationError("remote Variation candidate bytes violate the sealed request")
    seed_path = Path(evaluator_seed)
    key_path = Path(evaluator_private_key)
    if seed_path.is_symlink() or not seed_path.is_file() or key_path.is_symlink() or not key_path.is_file():
        raise VariationDependencyError("remote evaluator seed and private key must be regular local files")
    signer = ReceiptSigner(key_path.read_bytes())
    if signer.key_id != manifest["evaluator_key_id"]:
        raise VariationConfigurationError("remote evaluator private key differs from the service authority")
    from ..evaluation.authority import AuthorityBroker, AuthorityPolicy, DockerEnforcedRuntime
    from ..evaluation.dataset import EvaluationCorpus
    from ..evaluation.sandbox import DockerCandidateSandbox, DockerSandboxConfig

    corpus = EvaluationCorpus.generate(secret_seed_file=seed_path)
    if corpus.manifest_digest() != manifest["data_manifest_digest"]:
        raise VariationConfigurationError("remote evaluator private corpus differs from the frozen manifest")
    repo = corpus.get(str(request_value["task_id"]))
    if repo.public_manifest_record() != task_binding or repo.split not in {"train", "heldout"}:
        raise VariationConfigurationError("remote evaluator private task differs from its public binding")
    config = DockerSandboxConfig.from_environment()
    if config.pinned_image_id != manifest["docker_image_digest"] or digest_for(dict(config.__dict__)) != manifest["docker_config_digest"]:
        raise VariationConfigurationError("remote evaluator Docker configuration differs from the service manifest")
    policy = AuthorityPolicy.candidate_execution()
    if policy.digest != manifest["authority_policy_digest"] or policy.digest != manifest["policy_digest"]:
        raise VariationConfigurationError("remote evaluator authority policy differs from the frozen campaign")
    runtime = DockerEnforcedRuntime(config)
    sandbox = DockerCandidateSandbox(Path(workspace), config=config)
    runner = HiddenEvaluatorRunner.from_corpus(corpus, evaluator_revision=manifest["evaluator_revision"])
    sequence = request_value["receipt_sequence_start"]
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
        raise VariationConfigurationError("remote Variation receipt sequence anchor is invalid")
    previous = request_value["previous_receipt_hash"]
    if previous != GENESIS_HASH:
        _require_digest(previous, "previous receipt hash")
    def build_response() -> Mapping[str, Any]:
        collected = []
        with tempfile.TemporaryDirectory(prefix="egv-remote-receipts-") as journal_root:
            journal = ReceiptJournal(Path(journal_root) / "receipts.jsonl", signer.public_key)
            controller = EvaluatorController(
                sandbox=sandbox,
                hidden_runner=runner,
                broker=AuthorityBroker(runtime, policy),
                signer=signer,
                journal=journal,
                ingest=lambda receipt: collected.append(dict(receipt)),
                campaign_id=manifest["campaign_id"],
                protocol_digest=manifest["protocol_digest"],
                policy_digest=manifest["policy_digest"],
            )
            local_result = ControllerEvaluationGateway(controller).evaluate(
                candidate_id=str(request_value["candidate_id"]),
                task_id=repo.template_id,
                source=source,
                opaque_input=repo.evaluator_input,
                requested_authority=str(request_value["requested_authority"]),
                declared_locus=str(request_value["declared_locus"]),
            )
        signed_receipts = []
        current_sequence = sequence
        current_previous = previous
        for original in collected:
            fields = dict(original)
            for key in (
                "schema_version", "sequence", "previous_receipt_hash", "idempotency_key",
                "signing_key_id", "receipt_id", "signature",
            ):
                fields.pop(key, None)
            fields["evaluator_digest"] = manifest.digest
            fields["run_id"] = request_value["run_id"]
            fields["arm_policy_digest"] = request_value["arm_policy_digest"]
            receipt = signer.sign_receipt(
                fields,
                sequence=current_sequence,
                previous_receipt_hash=current_previous,
                idempotency_key=content_id(
                    "remote-variation-receipt",
                    {"operation_digest": request_value["operation_digest"], "receipt_type": fields["receipt_type"]},
                ),
            )
            signed_receipts.append(receipt)
            current_previous = receipt_hash(receipt)
            current_sequence += 1
        result_value = local_result.to_dict()
        result_value["receipt_ids"] = [receipt["receipt_id"] for receipt in signed_receipts]
        response_unsigned = {
            "schema_version": REMOTE_VARIATION_RESPONSE_SCHEMA,
            "operation_digest": request_value["operation_digest"],
            "request_digest": supplied_request_digest,
            "service_manifest_digest": manifest.digest,
            "result": result_value,
            "receipts": signed_receipts,
            "signing_key_id": signer.key_id,
        }
        return {**response_unsigned, "signature": signer.sign_bytes(canonical_bytes(response_unsigned))}

    return _durable_remote_response(
        request_value,
        manifest=manifest,
        signer=signer,
        state_root=state_root,
        build_response=build_response,
        validate_response=lambda response: _validate_remote_result_semantics(
            result_value=response["result"],
            receipts=response["receipts"],
            manifest=manifest,
            task_binding=task_binding,
            candidate_id=str(request_value["candidate_id"]),
            task_id=repo.template_id,
            artifact_digest=str(request_value["candidate_artifact_digest"]),
            declared_locus=str(request_value["declared_locus"]),
            run_id=str(request_value["run_id"]),
            arm_policy_digest=str(request_value["arm_policy_digest"]),
        ),
    )


__all__ = [
    "REMOTE_VARIATION_REQUEST_SCHEMA",
    "REMOTE_VARIATION_RESPONSE_SCHEMA",
    "REMOTE_VARIATION_SERVICE_SCHEMA",
    "RemoteControllerEvaluationGateway",
    "RemoteEvaluatorServiceManifest",
    "build_remote_evaluator_service_manifest",
    "run_remote_evaluator_once",
]
