"""Private development-loss evaluation and signed receipt bindings.

The development gateway owns its rows.  The trainer receives only a bounded
loss and a receipt whose digests bind checkpoint, model, data, and protocol;
raw development or held-out content never enters the receipt or public report.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Dict, Mapping, Sequence

from ..canonical import GENESIS_HASH, canonical_bytes, collection_digest, content_id, digest_for, validate_sha256
from ..receipts import ReceiptSigner, receipt_hash, verify_receipt
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


def _development_digest(rows: Sequence[TrainingRow]) -> str:
    ordered = sorted(rows, key=lambda row: row.row_id)
    return collection_digest(row.public_binding() for row in ordered)


__all__ = [
    "CHECKPOINT_SCHEMA",
    "DEVELOPMENT_GATEWAY_SCHEMA",
    "DEVELOPMENT_RECEIPT_SCHEMA",
    "DevelopmentLossEvaluation",
    "DevelopmentLossGateway",
    "TrainingCheckpoint",
    "model_state_digest_for_training",
]
