"""Typed DeepSeek lifecycle plans executed only through an injected boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple

from ..canonical import digest_for, validate_sha256
from ..public import build_public_restore_receipt
from .errors import LifecycleExecutionError, ProtectedInventoryError
from .inventory import ProtectedRestoreInventory
from .state import CampaignState, CampaignStateStore


@dataclass(frozen=True)
class CommandPlan:
    action: str
    argv: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.action not in {"CAPTURE", "STOP", "RESTORE"}:
            raise LifecycleExecutionError("lifecycle command action is invalid")
        if not self.argv or any(not isinstance(item, str) or not item for item in self.argv):
            raise LifecycleExecutionError("lifecycle argv is malformed")


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    observation: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool):
            raise LifecycleExecutionError("executor exit code is invalid")
        if not isinstance(self.observation, Mapping):
            raise LifecycleExecutionError("executor observation must be typed data")


@dataclass(frozen=True)
class ServiceObservation:
    logical_id: str
    running: bool
    container_identity: str
    image_digest: str
    model_digest: str
    configuration_digest: str
    executable_digest: str
    health_status: int
    health_response_digest: str
    smoke_input_digest: str
    smoke_output_digest: str


@dataclass(frozen=True)
class VerificationSnapshot:
    services: Tuple[ServiceObservation, ...]
    resource_baseline_digest: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VerificationSnapshot":
        if not isinstance(value, Mapping) or set(value) != {"services", "resource_baseline_digest"}:
            raise LifecycleExecutionError("executor observation is not a closed snapshot")
        raw_services = value["services"]
        if not isinstance(raw_services, list):
            raise LifecycleExecutionError("executor service observations are malformed")
        services = []
        fields = {
            "logical_id", "running", "container_identity", "image_digest", "model_digest",
            "configuration_digest", "executable_digest", "health_status", "health_response_digest",
            "smoke_input_digest", "smoke_output_digest",
        }
        for raw in raw_services:
            if not isinstance(raw, Mapping) or set(raw) != fields:
                raise LifecycleExecutionError("executor service observation is not closed")
            if not isinstance(raw["running"], bool) or not isinstance(raw["health_status"], int):
                raise LifecycleExecutionError("executor service state is malformed")
            for field in (
                "image_digest", "model_digest", "configuration_digest", "executable_digest",
                "health_response_digest", "smoke_input_digest", "smoke_output_digest",
            ):
                try:
                    validate_sha256(raw[field], field)
                except Exception as exc:
                    raise LifecycleExecutionError(str(exc)) from exc
            services.append(ServiceObservation(**{field: raw[field] for field in fields}))
        try:
            baseline = validate_sha256(value["resource_baseline_digest"], "resource baseline digest")
        except Exception as exc:
            raise LifecycleExecutionError(str(exc)) from exc
        return cls(tuple(services), baseline)


def _compose_base(inventory: ProtectedRestoreInventory) -> Tuple[str, ...]:
    if type(inventory) is not ProtectedRestoreInventory:
        raise ProtectedInventoryError("lifecycle planning requires an exact validated inventory")
    return (
        "docker", "compose", "--env-file", ".env.dspark", "-f", "docker-compose.dspark.yml",
        "-p", inventory.compose_project,
    )


def capture_plan(inventory: ProtectedRestoreInventory) -> CommandPlan:
    return CommandPlan("CAPTURE", _compose_base(inventory) + ("ps", "--format", "json"))


def stop_plan(inventory: ProtectedRestoreInventory) -> CommandPlan:
    services = tuple(inventory.service(item).compose_service for item in reversed(inventory.dependency_order))
    return CommandPlan("STOP", _compose_base(inventory) + ("stop", "--timeout", "120") + services)


def restore_plan(inventory: ProtectedRestoreInventory) -> CommandPlan:
    services = tuple(inventory.service(item).compose_service for item in inventory.dependency_order)
    return CommandPlan("RESTORE", _compose_base(inventory) + ("up", "-d", "--no-build") + services)


def _verify_snapshot(inventory: ProtectedRestoreInventory, snapshot: VerificationSnapshot, *, expected_running: bool) -> None:
    if len(snapshot.services) != inventory.expected_service_count:
        raise LifecycleExecutionError("service count differs from protected inventory")
    by_id = {item.logical_id: item for item in snapshot.services}
    if len(by_id) != len(snapshot.services) or set(by_id) != set(inventory.dependency_order):
        raise LifecycleExecutionError("observed service identities are absent, extra, or ambiguous")
    for logical_id in inventory.dependency_order:
        expected = inventory.service(logical_id)
        actual = by_id[logical_id]
        if actual.running is not expected_running:
            raise LifecycleExecutionError("service running state differs from lifecycle target")
        if (
            actual.container_identity != expected.container_identity
            or actual.image_digest != expected.image_digest
            or actual.model_digest != expected.model_digest
            or actual.configuration_digest != expected.configuration_digest
            or actual.executable_digest != expected.executable_digest
        ):
            raise LifecycleExecutionError("service identity/image/model/configuration binding failed")
        if expected_running and (
            actual.health_status != expected.health.expected_status
            or actual.health_response_digest != expected.health.response_digest
            or actual.smoke_input_digest != expected.smoke.input_digest
            or actual.smoke_output_digest != expected.smoke.output_digest
        ):
            raise LifecycleExecutionError("service health or deterministic smoke binding failed")
    if snapshot.resource_baseline_digest != inventory.resource_baseline_digest:
        raise LifecycleExecutionError("resource baseline differs from protected inventory")


class DeepSeekLifecycleController:
    """Plan/verify lifecycle operations; the caller owns the injected executor."""

    def __init__(self, inventory: ProtectedRestoreInventory, state_store: CampaignStateStore, executor: Callable[[CommandPlan], CommandResult]) -> None:
        if type(inventory) is not ProtectedRestoreInventory:
            raise ProtectedInventoryError("controller requires an exact validated inventory")
        if not callable(executor):
            raise LifecycleExecutionError("lifecycle executor must be injected")
        self.inventory = inventory
        self.state_store = state_store
        self.executor = executor

    def _run(self, plan: CommandPlan) -> Tuple[CommandResult, VerificationSnapshot]:
        result = self.executor(plan)
        if type(result) is not CommandResult:
            raise LifecycleExecutionError("executor returned an untyped result")
        snapshot = VerificationSnapshot.from_mapping(result.observation)
        return result, snapshot

    def capture(self) -> VerificationSnapshot:
        result, snapshot = self._run(capture_plan(self.inventory))
        if result.exit_code != 0:
            raise LifecycleExecutionError("restore-point capture command failed")
        _verify_snapshot(self.inventory, snapshot, expected_running=True)
        return snapshot

    def stop(self, *, expected_state_digest: str) -> CampaignState:
        current = self.state_store.load(expected_digest=expected_state_digest)
        if current is None or current.phase != "P3" or current.inventory_digest != self.inventory.digest:
            raise LifecycleExecutionError("controlled stop requires exact P3 state and inventory binding")
        self.capture()  # exact P2 baseline must still match before mutation
        # The durable recovery obligation precedes dispatch. A process kill at
        # any later instruction leaves an unambiguous STOPPING recovery state.
        intent = self.state_store.require_restore(expected_digest=current.digest, service_state="STOPPING")
        try:
            # Once STOP is offered to the executor its effect can be unknown,
            # including when the executor crashes or returns malformed proof.
            # Every such path durably forces restoration.
            result, snapshot = self._run(stop_plan(self.inventory))
            if result.exit_code != 0:
                raise LifecycleExecutionError("controlled stop command failed")
            _verify_snapshot(self.inventory, snapshot, expected_running=False)
        except Exception:
            return self.state_store.require_restore(expected_digest=intent.digest, service_state="PARTIAL")
        return self.state_store.require_restore(expected_digest=intent.digest, service_state="STOPPED")

    def reconcile_stop_intent(self, *, expected_state_digest: str) -> CampaignState:
        """Convert a recovered STOPPING intent into mandatory restoration."""

        current = self.state_store.load(expected_digest=expected_state_digest)
        if current is None or not current.restore_required or current.service_state != "STOPPING":
            raise LifecycleExecutionError("stop reconciliation requires exact durable STOPPING intent")
        return self.state_store.require_restore(expected_digest=current.digest, service_state="PARTIAL")

    def restore(
        self,
        *,
        expected_state_digest: str,
        public_receipt: Mapping[str, Any],
        evaluator_public_key: Any,
    ) -> CampaignState:
        current = self.state_store.load(expected_digest=expected_state_digest)
        if current is None or not current.restore_required:
            raise LifecycleExecutionError("restore requires durable restore-required state")
        restoring = self.state_store.require_restore(expected_digest=current.digest, service_state="RESTORING")
        result, snapshot = self._run(restore_plan(self.inventory))
        if result.exit_code != 0:
            raise LifecycleExecutionError("restore command failed; restoration remains required")
        _verify_snapshot(self.inventory, snapshot, expected_running=True)
        expected_payload = public_restore_payload(
            self.inventory,
            snapshot,
            campaign_id=restoring.campaign_id,
        )
        try:
            return self.state_store.mark_restored(
                expected_digest=restoring.digest,
                public_receipt=public_receipt,
                evaluator_public_key=evaluator_public_key,
                expected_payload=expected_payload,
            )
        except Exception as exc:
            raise LifecycleExecutionError("signed public restoration receipt is invalid or mismatched") from exc


def public_restore_payload(
    inventory: ProtectedRestoreInventory,
    snapshot: VerificationSnapshot,
    *,
    campaign_id: str,
) -> Dict[str, Any]:
    """Return the exact payload for the canonical signed public receipt."""

    _verify_snapshot(inventory, snapshot, expected_running=True)
    logical_ids = [f"svc-{index:03d}" for index in range(1, inventory.expected_service_count + 1)]
    return {
        "campaign_id": campaign_id,
        "logical_service_set_id": "svcset-" + digest_for(logical_ids)[:16],
        "logical_service_ids": logical_ids,
        "private_inventory_digest": inventory.digest,
        "service_definition_set_digest": digest_for([service.executable_digest for service in inventory.services]),
        "model_set_digest": digest_for([service.model_digest for service in inventory.services]),
        "configuration_set_digest": digest_for([service.configuration_digest for service in inventory.services]),
        "executable_or_image_set_digest": digest_for([service.image_digest for service in inventory.services]),
        "expected_service_count": inventory.expected_service_count,
        "restored_service_count": len(snapshot.services),
        "health_check_count": len(snapshot.services),
        "health_pass_count": len(snapshot.services),
        "all_health_checks_passed": True,
        "smoke_input_digest": digest_for([service.smoke.input_digest for service in inventory.services]),
        "smoke_output_digest": digest_for([service.smoke.output_digest for service in inventory.services]),
        "smoke_matches_baseline": True,
        "restoration_outcome": "RESTORED",
    }


def public_restore_receipt(
    inventory: ProtectedRestoreInventory,
    snapshot: VerificationSnapshot,
    *,
    campaign_id: str,
    signer: Any,
) -> Dict[str, Any]:
    """Build the repository's canonical signed ``egv-public-restore-v1`` receipt."""

    return build_public_restore_receipt(
        public_restore_payload(inventory, snapshot, campaign_id=campaign_id),
        signer,
    )


__all__ = [
    "CommandPlan", "CommandResult", "DeepSeekLifecycleController", "ServiceObservation",
    "VerificationSnapshot", "capture_plan", "public_restore_payload", "public_restore_receipt",
    "restore_plan", "stop_plan",
]
