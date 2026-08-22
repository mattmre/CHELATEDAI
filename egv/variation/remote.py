"""Sealed command bridge for an independent production Variation evaluator."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, Mapping, Optional

from ..canonical import GENESIS_HASH, canonical_bytes, canonical_json, content_id, digest_bytes, digest_for, validate_sha256
from ..evaluation.controller import EvaluationResult, EvaluatorController, HiddenEvaluatorRunner
from ..ledger import EvidenceLedger
from ..receipts import ReceiptJournal, ReceiptSigner, key_id_for_public_key, load_public_key, receipt_hash, verify_receipt
from .errors import VariationConfigurationError, VariationDependencyError


REMOTE_VARIATION_SERVICE_SCHEMA = "egv-remote-variation-service-v1"
REMOTE_VARIATION_REQUEST_SCHEMA = "egv-remote-variation-request-v1"
REMOTE_VARIATION_RESPONSE_SCHEMA = "egv-remote-variation-response-v1"
REMOTE_VARIATION_TIMEOUT_SECONDS = 600

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
        "request_digest",
        "service_manifest_digest",
        "campaign_id",
        "model_digest",
        "protocol_digest",
        "policy_digest",
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
        self._response_digests = set()
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
            try:
                completed = subprocess.run(
                    invocation,
                    input=canonical_json(request),
                    text=True,
                    capture_output=True,
                    timeout=REMOTE_VARIATION_TIMEOUT_SECONDS,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise VariationDependencyError("remote Variation evaluator invocation failed") from exc
        if completed.returncode != 0:
            raise VariationDependencyError("remote Variation evaluator returned a nonzero exit status")
        if len(completed.stdout.encode("utf-8")) > 1024 * 1024:
            raise VariationDependencyError("remote Variation evaluator response exceeds the bounded output limit")
        try:
            response = json.loads(completed.stdout)
        except ValueError as exc:
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
        del opaque_input, candidate_source_path
        task_binding = self.manifest.public_record(task_id)
        if task_binding is None:
            raise VariationConfigurationError("remote evaluator task is outside the frozen public registry")
        if task_binding["public_locus"] != declared_locus:
            raise VariationConfigurationError("remote evaluator task locus differs from the public registry")
        source_bytes = bytes(source)
        artifact_digest = digest_bytes(source_bytes)
        body = {
            "schema_version": REMOTE_VARIATION_REQUEST_SCHEMA,
            "service_manifest_digest": self.manifest.digest,
            "campaign_id": self.manifest["campaign_id"],
            "model_digest": self.manifest["model_digest"],
            "protocol_digest": self.manifest["protocol_digest"],
            "policy_digest": self.manifest["policy_digest"],
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
            "receipt_sequence_start": self.ledger.receipt_next_sequence(),
            "previous_receipt_hash": self.ledger.receipt_head(),
        }
        request = {**body, "request_digest": digest_for(body)}
        response = dict(self._invoke(request))
        if set(response) != _RESPONSE_FIELDS or response.get("schema_version") != REMOTE_VARIATION_RESPONSE_SCHEMA:
            raise VariationConfigurationError("remote Variation response is not closed")
        if response["request_digest"] != request["request_digest"] or response["service_manifest_digest"] != self.manifest.digest:
            raise VariationConfigurationError("remote Variation response has stale request or service bindings")
        if response["signing_key_id"] != self.manifest["evaluator_key_id"]:
            raise VariationConfigurationError("remote Variation response uses the wrong evaluator key")
        unsigned = dict(response)
        signature = _decode_b64(unsigned.pop("signature"), "response signature")
        try:
            load_public_key(self._public_key).verify(signature, canonical_bytes(unsigned))
        except Exception as exc:
            raise VariationConfigurationError("remote Variation response signature is invalid") from exc
        response_digest = digest_for(response)
        if response_digest in self._response_digests:
            raise VariationConfigurationError("remote Variation response replay detected")
        receipts = response["receipts"]
        if not isinstance(receipts, list) or not receipts:
            raise VariationConfigurationError("remote Variation response receipt chain is empty")
        previous = request["previous_receipt_hash"]
        sequence = request["receipt_sequence_start"]
        receipt_ids = []
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
            for field, expected_value in (
                ("campaign_id", self.manifest["campaign_id"]),
                ("task_id", task_id),
                ("candidate_id", candidate_id),
                ("candidate_artifact_digest", artifact_digest),
                ("protocol_digest", self.manifest["protocol_digest"]),
                ("policy_digest", self.manifest["policy_digest"]),
                ("evaluator_digest", self.manifest.digest),
            ):
                if receipt_value.get(field) != expected_value:
                    raise VariationConfigurationError("remote Variation receipt {} binding is invalid".format(field))
            if self.ledger.receipt_by_id(receipt_value["receipt_id"]) is not None:
                raise VariationConfigurationError("remote Variation receipt replay detected")
            receipt_ids.append(receipt_value["receipt_id"])
            receipt_values.append(receipt_value)
            previous = complete
            sequence += 1
        result_value = response["result"]
        if not isinstance(result_value, Mapping):
            raise VariationConfigurationError("remote Variation result is not an object")
        try:
            normalized_result = dict(result_value)
            normalized_result["receipt_ids"] = tuple(normalized_result["receipt_ids"])
            result = EvaluationResult(**normalized_result)
        except (KeyError, TypeError, ValueError) as exc:
            raise VariationConfigurationError("remote Variation result contract is invalid") from exc
        if (
            result.candidate_id != candidate_id
            or result.task_id != task_id
            or result.candidate_artifact_digest != artifact_digest
            or result.receipt_ids != tuple(receipt_ids)
        ):
            raise VariationConfigurationError("remote Variation result differs from its signed request or receipts")
        receipt_types = [receipt["receipt_type"] for receipt in receipt_values]
        if receipt_types not in (["AUTHORITY"], ["AUTHORITY", "VERDICT", "EFFECT"]):
            raise VariationConfigurationError("remote Variation receipt chain has an invalid type order")
        authority = receipt_values[0]
        if len(receipt_values) == 1:
            if authority["decision"] == "ALLOW" or result.disposition not in {"ABSTAINED", "REJECTED"}:
                raise VariationConfigurationError("remote Variation authority-only result is inconsistent")
        else:
            verdict, effect = receipt_values[1:]
            if authority["decision"] != "ALLOW":
                raise VariationConfigurationError("remote Variation executed after denied authority")
            if (
                verdict.get("diagnostic_enum") != result.diagnostic_enum
                or verdict.get("resource_bucket") != result.resource_bucket
                or verdict.get("output_digest") != result.output_digest
                or effect.get("diagnostic_enum") != result.diagnostic_enum
            ):
                raise VariationConfigurationError("remote Variation result disagrees with its signed verdict chain")
        if (
            self.ledger.receipt_head() != request["previous_receipt_hash"]
            or self.ledger.receipt_next_sequence() != request["receipt_sequence_start"]
        ):
            raise VariationConfigurationError("authoritative receipt head changed during remote evaluation")
        self.ledger.ingest_receipts_atomic(receipts, self._public_key)
        self._response_digests.add(response_digest)
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
) -> Mapping[str, Any]:
    """Run one request with evaluator-owned hidden inputs and Docker authority."""

    if not isinstance(request, Mapping) or set(request) != _REQUEST_FIELDS:
        raise VariationConfigurationError("remote Variation request is not closed")
    request_value = dict(request)
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
    task_binding = manifest.public_record(str(request_value["task_id"]))
    if task_binding is None or request_value["public_task_binding"] != task_binding:
        raise VariationConfigurationError("remote Variation request task binding is invalid")
    source = _decode_b64(request_value["candidate_source_b64"], "candidate source")
    from .loop import CANDIDATE_SOURCE_LIMIT, ControllerEvaluationGateway

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
    sequence = request_value["receipt_sequence_start"]
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
        raise VariationConfigurationError("remote Variation receipt sequence anchor is invalid")
    previous = request_value["previous_receipt_hash"]
    if previous != GENESIS_HASH:
        _require_digest(previous, "previous receipt hash")
    signed_receipts = []
    for original in collected:
        fields = dict(original)
        for key in ("schema_version", "sequence", "previous_receipt_hash", "idempotency_key", "signing_key_id", "receipt_id", "signature"):
            fields.pop(key, None)
        fields["evaluator_digest"] = manifest.digest
        receipt = signer.sign_receipt(
            fields,
            sequence=sequence,
            previous_receipt_hash=previous,
            idempotency_key=content_id(
                "remote-variation-receipt",
                {"request_digest": supplied_request_digest, "receipt_type": fields["receipt_type"]},
            ),
        )
        signed_receipts.append(receipt)
        previous = receipt_hash(receipt)
        sequence += 1
    result_value = local_result.to_dict()
    result_value["receipt_ids"] = [receipt["receipt_id"] for receipt in signed_receipts]
    response_unsigned = {
        "schema_version": REMOTE_VARIATION_RESPONSE_SCHEMA,
        "request_digest": supplied_request_digest,
        "service_manifest_digest": manifest.digest,
        "result": result_value,
        "receipts": signed_receipts,
        "signing_key_id": signer.key_id,
    }
    return {**response_unsigned, "signature": signer.sign_bytes(canonical_bytes(response_unsigned))}


__all__ = [
    "REMOTE_VARIATION_REQUEST_SCHEMA",
    "REMOTE_VARIATION_RESPONSE_SCHEMA",
    "REMOTE_VARIATION_SERVICE_SCHEMA",
    "RemoteControllerEvaluationGateway",
    "RemoteEvaluatorServiceManifest",
    "build_remote_evaluator_service_manifest",
    "run_remote_evaluator_once",
]
