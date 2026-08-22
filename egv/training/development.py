"""Private development-loss evaluation and signed receipt bindings.

The development gateway owns its rows.  The trainer receives only a bounded
loss and a receipt whose digests bind checkpoint, model, data, and protocol;
raw development or held-out content never enters the receipt or public report.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Dict, Mapping, Sequence

from ..canonical import GENESIS_HASH, canonical_bytes, canonical_json, collection_digest, content_id, digest_for, validate_sha256
from ..receipts import ReceiptSigner, key_id_for_public_key, receipt_hash, verify_receipt
from .protocol import (
    TrainingConfigurationError,
    TrainingDataManifest,
    TrainingIntegrityError,
    TrainingLeakageError,
    TrainingProtocol,
    TrainingRow,
    TrainingDependencyError,
    TrainingError,
)


CHECKPOINT_SCHEMA = "egv-training-checkpoint-v1"
DEVELOPMENT_RECEIPT_SCHEMA = "egv-development-loss-receipt-v1"
DEVELOPMENT_GATEWAY_SCHEMA = "egv-development-loss-gateway-v1"
EXTERNAL_DEVELOPMENT_SERVICE_SCHEMA = "egv-external-development-service-v1"


def model_state_digest_for_training(model: Any) -> str:
    """Hash a model state without serializing model objects or private paths."""

    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise TrainingDependencyError("training model must expose state_dict()")
    try:
        state = state_dict()
    except Exception as exc:
        raise TrainingDependencyError("training model state cannot be inspected") from exc
    if not isinstance(state, Mapping) or not state:
        raise TrainingDependencyError("training model state must be a non-empty mapping")
    records = []
    for name, value in sorted(state.items(), key=lambda item: str(item[0])):
        if not isinstance(name, str) or not name:
            raise TrainingDependencyError("training model state names must be non-empty strings")
        try:
            detached = value.detach().cpu()
            try:
                import torch

                material = detached.contiguous().view(dtype=torch.uint8).numpy().tobytes()
            except Exception:
                material = detached.numpy().tobytes()
            shape = tuple(int(item) for item in detached.shape)
            dtype = str(detached.dtype)
        except AttributeError:
            if isinstance(value, bytes):
                material = value
                shape = ()
                dtype = "bytes"
            elif isinstance(value, (str, int, float, bool, list, tuple, dict)):
                material = canonical_bytes(value)
                shape = ()
                dtype = type(value).__name__
            else:
                raise TrainingDependencyError("training model state contains an unsupported value")
        records.append(
            {
                "name": name,
                "shape": list(shape),
                "dtype": dtype,
                "digest": hashlib.sha256(material).hexdigest(),
            }
        )
    return digest_for(records)


def _require_digest(value: Any, name: str) -> str:
    try:
        return validate_sha256(value, name)
    except Exception as exc:
        raise TrainingIntegrityError(str(exc)) from exc


@dataclass(frozen=True)
class TrainingCheckpoint:
    """An immutable, content-addressed checkpoint binding one epoch."""

    checkpoint_id: str
    epoch: int
    global_step: int
    model_digest: str
    protocol_digest: str
    data_manifest_digest: str
    adapter_digest: str
    state_digest: str
    payload_digest: str
    schema_version: str = CHECKPOINT_SCHEMA

    def _body(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_digest": self.model_digest,
            "protocol_digest": self.protocol_digest,
            "data_manifest_digest": self.data_manifest_digest,
            "adapter_digest": self.adapter_digest,
            "state_digest": self.state_digest,
            "payload_digest": self.payload_digest,
        }

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {"checkpoint_id": self.checkpoint_id, **self._body()}

    @property
    def digest(self) -> str:
        self.validate()
        return digest_for(self.to_dict())

    def validate(self) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA:
            raise TrainingIntegrityError("unsupported Training checkpoint schema")
        if not isinstance(self.epoch, int) or isinstance(self.epoch, bool) or self.epoch < 1:
            raise TrainingIntegrityError("Training checkpoint epoch must be positive")
        if not isinstance(self.global_step, int) or isinstance(self.global_step, bool) or self.global_step < 1:
            raise TrainingIntegrityError("Training checkpoint global_step must be positive")
        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id:
            raise TrainingIntegrityError("Training checkpoint ID is missing")
        for name in (
            "model_digest",
            "protocol_digest",
            "data_manifest_digest",
            "adapter_digest",
            "state_digest",
            "payload_digest",
        ):
            _require_digest(getattr(self, name), name)
        expected_id = content_id("train-checkpoint", self._body())
        if self.checkpoint_id != expected_id:
            raise TrainingIntegrityError("Training checkpoint ID is not content-addressed")

    @classmethod
    def create(
        cls,
        *,
        epoch: int,
        global_step: int,
        model_digest: str,
        protocol_digest: str,
        data_manifest_digest: str,
        adapter_digest: str,
        state_digest: str,
        payload_digest: str,
    ) -> "TrainingCheckpoint":
        body = {
            "schema_version": CHECKPOINT_SCHEMA,
            "epoch": epoch,
            "global_step": global_step,
            "model_digest": model_digest,
            "protocol_digest": protocol_digest,
            "data_manifest_digest": data_manifest_digest,
            "adapter_digest": adapter_digest,
            "state_digest": state_digest,
            "payload_digest": payload_digest,
        }
        result = cls(checkpoint_id=content_id("train-checkpoint", body), **body)
        result.validate()
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrainingCheckpoint":
        required = {
            "checkpoint_id",
            "epoch",
            "global_step",
            "model_digest",
            "protocol_digest",
            "data_manifest_digest",
            "adapter_digest",
            "state_digest",
            "payload_digest",
            "schema_version",
        }
        if set(value) != required:
            raise TrainingIntegrityError("Training checkpoint has an unexpected field set")
        result = cls(**{key: value[key] for key in required})
        result.validate()
        return result


@dataclass(frozen=True)
class DevelopmentLossEvaluation:
    """Loss plus the signed, content-only receipt returned by the gateway."""

    checkpoint_digest: str
    checkpoint_id: str
    model_digest: str
    data_manifest_digest: str
    development_manifest_digest: str
    protocol_digest: str
    gateway_digest: str
    loss: float
    sample_count: int
    receipt: Mapping[str, Any]
    schema_version: str = DEVELOPMENT_RECEIPT_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "checkpoint_digest": self.checkpoint_digest,
            "checkpoint_id": self.checkpoint_id,
            "model_digest": self.model_digest,
            "data_manifest_digest": self.data_manifest_digest,
            "development_manifest_digest": self.development_manifest_digest,
            "protocol_digest": self.protocol_digest,
            "gateway_digest": self.gateway_digest,
            "loss": self.loss,
            "sample_count": self.sample_count,
            "receipt": dict(self.receipt),
        }

    def validate(self) -> None:
        if self.schema_version != DEVELOPMENT_RECEIPT_SCHEMA:
            raise TrainingIntegrityError("unsupported development-loss evaluation schema")
        for name in (
            "checkpoint_digest",
            "model_digest",
            "data_manifest_digest",
            "development_manifest_digest",
            "protocol_digest",
            "gateway_digest",
        ):
            _require_digest(getattr(self, name), name)
        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id:
            raise TrainingIntegrityError("development evaluation checkpoint ID is missing")
        if not isinstance(self.loss, (int, float)) or isinstance(self.loss, bool) or not math.isfinite(float(self.loss)):
            raise TrainingIntegrityError("development loss must be finite")
        if self.loss < 0:
            raise TrainingIntegrityError("development loss cannot be negative")
        if not isinstance(self.sample_count, int) or isinstance(self.sample_count, bool) or self.sample_count <= 0:
            raise TrainingIntegrityError("development sample count must be positive")
        if not isinstance(self.receipt, Mapping):
            raise TrainingIntegrityError("development evaluation receipt is missing")


def _loss_output_digest(evaluation: Mapping[str, Any]) -> str:
    return digest_for(
        {
            "checkpoint_digest": evaluation["checkpoint_digest"],
            "model_digest": evaluation["model_digest"],
            "data_manifest_digest": evaluation["data_manifest_digest"],
            "protocol_digest": evaluation["protocol_digest"],
            "loss": evaluation["loss"],
            "sample_count": evaluation["sample_count"],
        }
    )


def _closed_receipt_has_no_content(receipt: Mapping[str, Any]) -> None:
    allowed_task = "DEVELOPMENT_LOSS"
    if receipt.get("task_id") != allowed_task:
        raise TrainingIntegrityError("development receipt task identity is outside the closed contract")
    forbidden_fragments = ("prompt", "target", "heldout", "expected", "private", "source_event")
    for key, value in receipt.items():
        text = "{} {}".format(key, value).lower()
        if any(fragment in text for fragment in forbidden_fragments):
            raise TrainingLeakageError("development receipt contains private or held-out content")


class DevelopmentLossGateway:
    """Evaluator-owned development loss gateway with a signed receipt chain.

    ``evaluator`` is the only callable allowed to inspect the private rows.
    It receives ``(model, private_rows, protocol)`` and must return one finite
    scalar loss.  The gateway never exposes those rows through its public
    result, receipt, or digest fields.
    """

    def __init__(
        self,
        development_rows: Sequence[TrainingRow],
        *,
        data_manifest: TrainingDataManifest,
        model_digest: str,
        protocol: TrainingProtocol,
        signer: ReceiptSigner,
        evaluator: Callable[[Any, Sequence[TrainingRow], TrainingProtocol], float],
        production: bool = True,
    ) -> None:
        if type(self) is not DevelopmentLossGateway:
            raise TrainingDependencyError("production requires the exact DevelopmentLossGateway type")
        protocol.validate()
        data_manifest.validate()
        rows = tuple(development_rows)
        if not rows:
            raise TrainingConfigurationError("development gateway requires at least one development row")
        for row in rows:
            if type(row) is not TrainingRow:
                raise TrainingConfigurationError("development gateway requires exact TrainingRow values")
            row.validate(expected_split="dev")
        expected_ids = tuple(sorted(row.row_id for row in rows))
        if expected_ids != data_manifest.development_row_ids:
            raise TrainingIntegrityError("development rows do not match the immutable data manifest")
        if data_manifest.development_digest != _development_digest(rows):
            raise TrainingIntegrityError("development row content does not match the immutable data manifest")
        if not callable(evaluator):
            raise TrainingDependencyError("development gateway requires a callable evaluator")
        if not isinstance(signer, ReceiptSigner):
            raise TrainingDependencyError("development gateway requires an Ed25519 ReceiptSigner")
        _require_digest(model_digest, "model_digest")
        object.__setattr__(self, "_rows", rows)
        object.__setattr__(self, "_data_manifest", data_manifest)
        object.__setattr__(self, "_model_digest", model_digest)
        object.__setattr__(self, "_protocol", protocol)
        object.__setattr__(self, "_signer", signer)
        object.__setattr__(self, "_evaluator", evaluator)
        object.__setattr__(self, "_production", bool(production))
        gateway_digest = digest_for(
            {
                "schema_version": DEVELOPMENT_GATEWAY_SCHEMA,
                "model_digest": model_digest,
                "data_manifest_digest": data_manifest.digest,
                "development_manifest_digest": data_manifest.development_digest,
                "protocol_digest": protocol.digest,
                "signing_key_id": signer.key_id,
            }
        )
        object.__setattr__(self, "_gateway_digest", gateway_digest)
        object.__setattr__(self, "_receipts", {})
        object.__setattr__(self, "_receipt_order", [])
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("DevelopmentLossGateway is immutable")
        object.__setattr__(self, name, value)

    @property
    def production(self) -> bool:
        return self._production

    @property
    def model_digest(self) -> str:
        return self._model_digest

    @property
    def data_manifest_digest(self) -> str:
        return self._data_manifest.digest

    @property
    def development_manifest_digest(self) -> str:
        return self._data_manifest.development_digest

    @property
    def protocol_digest(self) -> str:
        return self._protocol.digest

    @property
    def gateway_digest(self) -> str:
        return self._gateway_digest

    @property
    def public_key(self) -> Any:
        return self._signer.public_key

    @property
    def row_count(self) -> int:
        return len(self._rows)

    def validate_production_boundary(self, *, expected_model_digest: str, expected_protocol_digest: str) -> None:
        if type(self) is not DevelopmentLossGateway:
            raise TrainingDependencyError("production development gateway type was changed")
        if not self.production:
            raise TrainingDependencyError("production training requires a production development gateway")
        if self.model_digest != expected_model_digest:
            raise TrainingIntegrityError("development gateway is bound to a different model")
        if self.protocol_digest != expected_protocol_digest:
            raise TrainingIntegrityError("development gateway is bound to a different protocol")
        if self._gateway_digest != digest_for(
            {
                "schema_version": DEVELOPMENT_GATEWAY_SCHEMA,
                "model_digest": self.model_digest,
                "data_manifest_digest": self.data_manifest_digest,
                "development_manifest_digest": self.development_manifest_digest,
                "protocol_digest": self.protocol_digest,
                "signing_key_id": self._signer.key_id,
            }
        ):
            raise TrainingIntegrityError("development gateway digest changed")

    def evaluate(self, checkpoint: TrainingCheckpoint, *, model: Any) -> DevelopmentLossEvaluation:
        checkpoint.validate()
        if checkpoint.model_digest != self.model_digest:
            raise TrainingIntegrityError("development evaluation received a wrong-model checkpoint")
        if checkpoint.protocol_digest != self.protocol_digest:
            raise TrainingIntegrityError("development evaluation received a stale protocol checkpoint")
        if checkpoint.data_manifest_digest != self.data_manifest_digest:
            raise TrainingIntegrityError("development evaluation received a stale data checkpoint")
        actual_state_digest = model_state_digest_for_training(model)
        if actual_state_digest != checkpoint.state_digest:
            raise TrainingIntegrityError("development evaluation model state does not match the checkpoint")
        checkpoint_digest = checkpoint.digest
        cached = self._receipts.get(checkpoint_digest)
        if cached is not None:
            return cached
        try:
            loss = self._evaluator(model, self._rows, self._protocol)
        except TrainingError:
            raise
        except Exception as exc:
            raise TrainingDependencyError("development evaluator failed") from exc
        if not isinstance(loss, (int, float)) or isinstance(loss, bool) or not math.isfinite(float(loss)):
            raise TrainingIntegrityError("development evaluator returned a non-finite loss")
        loss = float(loss)
        if loss < 0:
            raise TrainingIntegrityError("development evaluator returned a negative loss")
        output_digest = _loss_output_digest(
            {
                "checkpoint_digest": checkpoint_digest,
                "model_digest": checkpoint.model_digest,
                "data_manifest_digest": checkpoint.data_manifest_digest,
                "protocol_digest": checkpoint.protocol_digest,
                "loss": loss,
                "sample_count": len(self._rows),
            }
        )
        sequence = len(self._receipt_order) + 1
        previous = GENESIS_HASH
        if self._receipt_order:
            previous = receipt_hash(self._receipt_order[-1])
        receipt = self._signer.sign_receipt(
            {
                "receipt_type": "VERDICT",
                "campaign_id": "egv-training",
                "run_id": "development-loss",
                "task_id": "DEVELOPMENT_LOSS",
                "request_id": content_id("development-request", checkpoint_digest),
                "candidate_id": checkpoint.checkpoint_id,
                "candidate_artifact_digest": checkpoint_digest,
                "protocol_digest": checkpoint.protocol_digest,
                "evaluator_digest": self.gateway_digest,
                "decision": "PASS",
                "diagnostic_enum": "PASS",
                "resource_bucket": "UNDER_25",
                "exit_status_class": "SUCCESS",
                "input_digest": self.development_manifest_digest,
                "output_digest": output_digest,
                "effect_kind": "DEVELOPMENT_LOSS",
            },
            sequence=sequence,
            previous_receipt_hash=previous,
            idempotency_key=content_id("development-idempotency", checkpoint_digest),
        )
        _closed_receipt_has_no_content(receipt)
        result = DevelopmentLossEvaluation(
            checkpoint_digest=checkpoint_digest,
            checkpoint_id=checkpoint.checkpoint_id,
            model_digest=checkpoint.model_digest,
            data_manifest_digest=checkpoint.data_manifest_digest,
            development_manifest_digest=self.development_manifest_digest,
            protocol_digest=checkpoint.protocol_digest,
            gateway_digest=self.gateway_digest,
            loss=loss,
            sample_count=len(self._rows),
            receipt=receipt,
        )
        result.validate()
        self._receipts[checkpoint_digest] = result
        self._receipt_order.append(receipt)
        return result

    def verify_evaluation(
        self,
        evaluation: DevelopmentLossEvaluation,
        *,
        checkpoint: TrainingCheckpoint,
        expected_model_digest: str,
        expected_data_manifest_digest: str,
        expected_protocol_digest: str,
    ) -> None:
        if type(evaluation) is not DevelopmentLossEvaluation:
            raise TrainingIntegrityError("development evaluation type is not closed")
        evaluation.validate()
        checkpoint.validate()
        if evaluation.checkpoint_digest != checkpoint.digest or evaluation.checkpoint_id != checkpoint.checkpoint_id:
            raise TrainingIntegrityError("development receipt is bound to a different checkpoint")
        if evaluation.model_digest != expected_model_digest or checkpoint.model_digest != expected_model_digest:
            raise TrainingIntegrityError("development receipt is bound to a different model")
        if evaluation.data_manifest_digest != expected_data_manifest_digest:
            raise TrainingIntegrityError("development receipt is bound to a different data manifest")
        if evaluation.protocol_digest != expected_protocol_digest:
            raise TrainingIntegrityError("development receipt is bound to a different protocol")
        if evaluation.gateway_digest != self.gateway_digest:
            raise TrainingIntegrityError("development receipt is bound to a different gateway")
        receipt = dict(evaluation.receipt)
        _closed_receipt_has_no_content(receipt)
        verify_receipt(receipt, self.public_key, expected_key_id=self._signer.key_id)
        expected_output = _loss_output_digest(
            {
                "checkpoint_digest": evaluation.checkpoint_digest,
                "model_digest": evaluation.model_digest,
                "data_manifest_digest": evaluation.data_manifest_digest,
                "protocol_digest": evaluation.protocol_digest,
                "loss": evaluation.loss,
                "sample_count": evaluation.sample_count,
            }
        )
        if receipt.get("candidate_artifact_digest") != checkpoint.digest:
            raise TrainingIntegrityError("development receipt checkpoint digest is wrong")
        if receipt.get("protocol_digest") != expected_protocol_digest:
            raise TrainingIntegrityError("development receipt protocol digest is wrong")
        if receipt.get("evaluator_digest") != self.gateway_digest:
            raise TrainingIntegrityError("development receipt gateway digest is wrong")
        if receipt.get("input_digest") != self.development_manifest_digest:
            raise TrainingIntegrityError("development receipt data digest is wrong")
        if receipt.get("output_digest") != expected_output:
            raise TrainingIntegrityError("development receipt loss digest is wrong")
        if receipt.get("decision") != "PASS" or receipt.get("diagnostic_enum") != "PASS":
            raise TrainingIntegrityError("development receipt is not a valid loss result")


class ExternalDevelopmentLossGateway:
    """Production-only client for an evaluator-owned signed loss service."""

    _FIELDS = frozenset({
        "schema_version", "campaign_id", "evaluator_key_id", "evaluator_public_key_digest",
        "endpoint_digest", "development_manifest_digest", "development_task_count",
        "development_row_ids",
        "model_digest", "protocol_digest", "service_manifest_digest",
    })

    def __init__(self, manifest_path: Path, *, public_key_path: Path, command: Path) -> None:
        paths = tuple(Path(item) for item in (manifest_path, public_key_path, command))
        if any(path.is_symlink() or not path.is_file() for path in paths):
            raise TrainingDependencyError("external evaluator manifest, key, and command must be regular files")
        try:
            value = json.loads(paths[0].read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise TrainingIntegrityError("external evaluator service manifest cannot be decoded") from exc
        if not isinstance(value, Mapping) or set(value) != self._FIELDS:
            raise TrainingIntegrityError("external evaluator service manifest is not closed")
        unsigned = dict(value)
        supplied = unsigned.pop("service_manifest_digest")
        if value["schema_version"] != EXTERNAL_DEVELOPMENT_SERVICE_SCHEMA or digest_for(unsigned) != supplied:
            raise TrainingIntegrityError("external evaluator service manifest digest is invalid")
        if value["development_task_count"] != 8:
            raise TrainingIntegrityError("external evaluator must bind exactly eight development tasks")
        if (
            not isinstance(value["development_row_ids"], list)
            or len(value["development_row_ids"]) != 8
            or value["development_row_ids"] != sorted(set(value["development_row_ids"]))
            or any(not isinstance(item, str) or not item for item in value["development_row_ids"])
        ):
            raise TrainingIntegrityError("external evaluator development row IDs are not a sealed eight-row set")
        public_key = paths[1].read_bytes()
        if hashlib.sha256(public_key).hexdigest() != value["evaluator_public_key_digest"]:
            raise TrainingIntegrityError("external evaluator public key differs from the frozen manifest")
        if key_id_for_public_key(public_key) != value["evaluator_key_id"]:
            raise TrainingIntegrityError("external evaluator key ID differs from its public key")
        command_bytes = paths[2].read_bytes()
        if hashlib.sha256(command_bytes).hexdigest() != value["endpoint_digest"]:
            raise TrainingIntegrityError("external evaluator command identity differs from the frozen manifest")
        self._manifest = dict(value)
        self._public_key = public_key
        self._command = paths[2].resolve()
        self._command_bytes = command_bytes
        self._endpoint_digest = value["endpoint_digest"]
        self._command_suffix = paths[2].suffix.lower()

    @property
    def production(self) -> bool:
        return True

    @property
    def model_digest(self) -> str:
        return self._manifest["model_digest"]

    @property
    def protocol_digest(self) -> str:
        return self._manifest["protocol_digest"]

    @property
    def gateway_digest(self) -> str:
        return self._manifest["service_manifest_digest"]

    @property
    def development_manifest_digest(self) -> str:
        return self._manifest["development_manifest_digest"]

    @property
    def development_row_ids(self) -> tuple:
        return tuple(self._manifest["development_row_ids"])

    def validate_production_boundary(self, *, expected_model_digest: str, expected_protocol_digest: str) -> None:
        if type(self) is not ExternalDevelopmentLossGateway:
            raise TrainingDependencyError("production evaluator client type changed")
        if self.model_digest != expected_model_digest or self.protocol_digest != expected_protocol_digest:
            raise TrainingIntegrityError("external evaluator is bound to a different model or protocol")

    def evaluate(
        self, checkpoint: TrainingCheckpoint, *, model: Any, checkpoint_artifact_digest: str
    ) -> DevelopmentLossEvaluation:
        checkpoint.validate()
        self.validate_production_boundary(
            expected_model_digest=checkpoint.model_digest, expected_protocol_digest=checkpoint.protocol_digest
        )
        with tempfile.TemporaryDirectory(prefix="egv-dev-adapter-") as temporary:
            root = Path(temporary)
            try:
                model.save_pretrained(str(root), safe_serialization=True)
            except Exception as exc:
                raise TrainingDependencyError("cannot stage sealed adapter for external evaluation") from exc
            from ..variation.adapter import ADAPTER_MANIFEST_NAME, SealedAdapterArtifact, build_local_adapter_manifest

            adapter_manifest = build_local_adapter_manifest(root)
            (root / ADAPTER_MANIFEST_NAME).write_text(
                canonical_json(adapter_manifest.to_dict()) + "\n", encoding="utf-8"
            )
            artifact = SealedAdapterArtifact(root)
            artifact.verify()
            request = {
                "schema_version": "egv-development-loss-request-v1",
                "campaign_id": self._manifest["campaign_id"],
                "checkpoint_digest": validate_sha256(checkpoint_artifact_digest, "checkpoint_artifact_digest"),
                "adapter_digest": artifact.digest,
                "adapter_root": str(root.resolve()),
                "model_digest": checkpoint.model_digest,
                "protocol_digest": checkpoint.protocol_digest,
                "development_manifest_digest": self.development_manifest_digest,
                "service_manifest_digest": self.gateway_digest,
            }
            try:
                current_bytes = self._command.read_bytes()
                if (
                    current_bytes != self._command_bytes
                    or hashlib.sha256(current_bytes).hexdigest() != self._endpoint_digest
                ):
                    raise TrainingIntegrityError("external evaluator command changed before invocation")
                with tempfile.TemporaryDirectory(prefix="egv-pinned-endpoint-") as endpoint_temporary:
                    endpoint = Path(endpoint_temporary) / (self._endpoint_digest + self._command_suffix)
                    with endpoint.open("xb") as handle:
                        handle.write(self._command_bytes)
                        handle.flush()
                        __import__("os").fsync(handle.fileno())
                    endpoint.chmod(0o500)
                    if hashlib.sha256(endpoint.read_bytes()).hexdigest() != self._endpoint_digest:
                        raise TrainingIntegrityError("content-addressed evaluator copy failed verification")
                    invocation = [sys.executable, str(endpoint)] if self._command_suffix == ".py" else [str(endpoint)]
                    completed = subprocess.run(
                        invocation, input=canonical_json(request), text=True,
                        capture_output=True, timeout=3600, check=False,
                    )
                    if completed.returncode != 0:
                        raise TrainingDependencyError("external development evaluator returned a nonzero exit status")
                response = json.loads(completed.stdout)
            except TrainingError:
                raise
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                raise TrainingDependencyError("external development evaluator failed") from exc
            if completed.returncode != 0 or not isinstance(response, Mapping):
                raise TrainingDependencyError("external development evaluator returned no valid result")
            required = {"schema_version", "checkpoint_digest", "adapter_digest", "loss", "sample_count", "receipt"}
            if set(response) != required or response["schema_version"] != "egv-development-loss-response-v1":
                raise TrainingIntegrityError("external development response is not closed")
            if response["checkpoint_digest"] != checkpoint_artifact_digest or response["adapter_digest"] != artifact.digest:
                raise TrainingIntegrityError("external development response is bound to another artifact")
            loss = float(response["loss"])
            if not math.isfinite(loss) or loss < 0 or response["sample_count"] != 8:
                raise TrainingIntegrityError("external development response has invalid aggregate metrics")
            receipt = dict(response["receipt"])
            verify_receipt(receipt, self._public_key, expected_key_id=self._manifest["evaluator_key_id"])
            expected_output_digest = digest_for({
                "checkpoint_digest": checkpoint_artifact_digest, "adapter_digest": artifact.digest,
                "loss": loss, "sample_count": 8,
            })
            if (
                receipt.get("candidate_artifact_digest") != artifact.digest
                or receipt.get("protocol_digest") != checkpoint.protocol_digest
                or receipt.get("evaluator_digest") != self.gateway_digest
                or receipt.get("input_digest") != self.development_manifest_digest
                or receipt.get("output_digest") != expected_output_digest
                or receipt.get("decision") != "PASS"
            ):
                raise TrainingIntegrityError("external development receipt binding is invalid")
            result = DevelopmentLossEvaluation(
                checkpoint_artifact_digest, checkpoint.checkpoint_id, checkpoint.model_digest,
                checkpoint.data_manifest_digest, self.development_manifest_digest,
                checkpoint.protocol_digest, self.gateway_digest, loss, 8, receipt,
            )
            result.validate()
            return result

    def verify_evaluation(
        self, evaluation: DevelopmentLossEvaluation, *, checkpoint: TrainingCheckpoint,
        expected_model_digest: str, expected_data_manifest_digest: str, expected_protocol_digest: str,
    ) -> None:
        if (
            evaluation.model_digest != expected_model_digest
            or evaluation.data_manifest_digest != expected_data_manifest_digest
            or evaluation.protocol_digest != expected_protocol_digest
        ):
            raise TrainingIntegrityError("external development evaluation binding changed")
        verify_receipt(dict(evaluation.receipt), self._public_key, expected_key_id=self._manifest["evaluator_key_id"])


def run_external_evaluator_once(
    request: Mapping[str, Any], *, service_manifest: Path, model_root: Path,
    development_dataset: Path, evaluator_private_key: Path, device: str = "cuda",
) -> Mapping[str, Any]:
    """Evaluator-side one-shot command; private rows and signing key stay here."""

    if device != "cuda":
        raise TrainingConfigurationError("external development evaluation is frozen to CUDA")

    required_request = {
        "schema_version", "campaign_id", "checkpoint_digest", "adapter_digest", "adapter_root",
        "model_digest", "protocol_digest", "development_manifest_digest", "service_manifest_digest",
    }
    if not isinstance(request, Mapping) or set(request) != required_request:
        raise TrainingIntegrityError("development loss request is not closed")
    service_value = json.loads(Path(service_manifest).read_text(encoding="utf-8"))
    if not isinstance(service_value, Mapping) or set(service_value) != ExternalDevelopmentLossGateway._FIELDS:
        raise TrainingIntegrityError("evaluator service manifest is not closed")
    unsigned = dict(service_value)
    service_digest = unsigned.pop("service_manifest_digest")
    if digest_for(unsigned) != service_digest or request["service_manifest_digest"] != service_digest:
        raise TrainingIntegrityError("development request service binding is invalid")
    for field in ("campaign_id", "model_digest", "protocol_digest", "development_manifest_digest"):
        if request[field] != service_value[field]:
            raise TrainingIntegrityError("development request {} binding differs".format(field))
    key_path = Path(evaluator_private_key)
    if key_path.is_symlink() or not key_path.is_file():
        raise TrainingDependencyError("evaluator private key must be a regular local file")
    signer = ReceiptSigner(key_path.read_bytes())
    if signer.key_id != service_value["evaluator_key_id"]:
        raise TrainingIntegrityError("evaluator private key differs from the frozen authority")
    from ..variation.adapter import SealedAdapterArtifact
    from ..variation.model import PinnedModelLoader

    artifact = SealedAdapterArtifact(Path(str(request["adapter_root"])))
    artifact.verify()
    if artifact.digest != request["adapter_digest"]:
        raise TrainingIntegrityError("development request adapter digest differs from its sealed tree")
    try:
        private_value = json.loads(Path(development_dataset).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TrainingIntegrityError("private development dataset cannot be decoded") from exc
    if not isinstance(private_value, Mapping) or set(private_value) != {
        "schema_version", "development_manifest_digest", "rows"
    } or private_value["schema_version"] != "egv-private-development-runtime-v1":
        raise TrainingIntegrityError("private development dataset is not closed")
    rows = private_value["rows"]
    if not isinstance(rows, list) or len(rows) != 8:
        raise TrainingIntegrityError("private development dataset must contain exactly eight rows")
    normalized = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "row_id", "task_id", "task_family", "prompt", "target"
        } or any(not isinstance(row[field], str) or not row[field] for field in row):
            raise TrainingIntegrityError("private development row is malformed")
        normalized.append(dict(row))
    row_ids = sorted(row["row_id"] for row in normalized)
    if row_ids != service_value["development_row_ids"]:
        raise TrainingIntegrityError("private development row identities differ from the frozen service")
    public_digest = collection_digest({
        "row_id": row["row_id"], "task_id": row["task_id"], "task_family": row["task_family"],
        "prompt_digest": digest_for(row["prompt"]), "target_digest": digest_for(row["target"]),
    } for row in sorted(normalized, key=lambda item: item["row_id"]))
    if public_digest != private_value["development_manifest_digest"] or public_digest != service_value["development_manifest_digest"]:
        raise TrainingIntegrityError("private development content differs from its frozen digest")
    try:
        import torch
    except ImportError as exc:
        raise TrainingDependencyError("external development evaluator requires torch") from exc
    if not torch.cuda.is_available():
        raise TrainingDependencyError("external development evaluator requires an available CUDA device")
    loaded = PinnedModelLoader(model_root).load(
        device=device, torch_dtype=torch.bfloat16, adapter_artifact=artifact
    )
    if loaded.manifest_digest != request["model_digest"]:
        raise TrainingIntegrityError("development evaluator loaded a different pinned model")
    model = loaded.model
    model.eval()
    losses = []
    with torch.no_grad():
        for row in normalized:
            prompt_ids = loaded.tokenizer(row["prompt"], add_special_tokens=False, truncation=False)["input_ids"]
            target_ids = loaded.tokenizer(row["target"], add_special_tokens=False, truncation=False)["input_ids"]
            eos = getattr(loaded.tokenizer, "eos_token_id", None)
            if isinstance(eos, int) and (not target_ids or target_ids[-1] != eos):
                target_ids = list(target_ids) + [eos]
            ids = list(prompt_ids) + list(target_ids)
            if not ids or len(ids) > 4096:
                raise TrainingIntegrityError("private development row exceeds the frozen token boundary")
            device_value = next(model.parameters()).device
            input_ids = torch.tensor([ids], dtype=torch.long, device=device_value)
            labels = torch.tensor([[-100] * len(prompt_ids) + list(target_ids)], dtype=torch.long, device=device_value)
            output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=labels)
            loss = float(output.loss.detach().float().cpu().item())
            if not math.isfinite(loss) or loss < 0:
                raise TrainingIntegrityError("development model returned invalid loss")
            losses.append(loss)
    aggregate = sum(losses) / len(losses)
    output_digest = digest_for({
        "checkpoint_digest": request["checkpoint_digest"], "adapter_digest": artifact.digest,
        "loss": aggregate, "sample_count": 8,
    })
    receipt = signer.sign_receipt(
        {
            "receipt_type": "VERDICT", "campaign_id": request["campaign_id"],
            "run_id": "development-loss", "task_id": "DEVELOPMENT_LOSS",
            "request_id": content_id("development-request", request["checkpoint_digest"]),
            "candidate_id": content_id("development-adapter", artifact.digest),
            "candidate_artifact_digest": artifact.digest,
            "protocol_digest": request["protocol_digest"], "evaluator_digest": service_digest,
            "decision": "PASS", "diagnostic_enum": "PASS", "resource_bucket": "UNDER_25",
            "exit_status_class": "SUCCESS", "input_digest": public_digest,
            "output_digest": output_digest, "effect_kind": "DEVELOPMENT_LOSS",
        },
        sequence=1, previous_receipt_hash=GENESIS_HASH,
        idempotency_key=content_id("development-idempotency", request["checkpoint_digest"]),
    )
    return {
        "schema_version": "egv-development-loss-response-v1",
        "checkpoint_digest": request["checkpoint_digest"], "adapter_digest": artifact.digest,
        "loss": aggregate, "sample_count": 8, "receipt": receipt,
    }


def _development_digest(rows: Sequence[TrainingRow]) -> str:
    ordered = sorted(rows, key=lambda row: row.row_id)
    return collection_digest(row.public_binding() for row in ordered)


__all__ = [
    "CHECKPOINT_SCHEMA",
    "DEVELOPMENT_GATEWAY_SCHEMA",
    "DEVELOPMENT_RECEIPT_SCHEMA",
    "DevelopmentLossEvaluation",
    "DevelopmentLossGateway",
    "EXTERNAL_DEVELOPMENT_SERVICE_SCHEMA",
    "ExternalDevelopmentLossGateway",
    "run_external_evaluator_once",
    "TrainingCheckpoint",
    "model_state_digest_for_training",
]
