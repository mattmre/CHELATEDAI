from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from egv.canonical import canonical_bytes, digest_for
from egv.campaign.errors import CampaignStateError, LifecycleExecutionError, ProtectedInventoryError
from egv.campaign.inventory import ProtectedRestoreInventory, load_protected_inventory_fd
from egv.campaign.lifecycle import (
    CommandResult,
    DeepSeekLifecycleController,
    VerificationSnapshot,
    capture_plan,
    public_restore_receipt,
    restore_plan,
    stop_plan,
)
from egv.campaign.state import CampaignStateStore, _exclusive_path_lock
from egv.receipts import ReceiptSigner


def _sha(label: str) -> str:
    return digest_for(label)


def _inventory_mapping() -> dict:
    services = []
    for index, name in enumerate(("api", "worker"), start=1):
        services.append({
            "logical_id": f"private-{name}",
            "compose_service": name,
            "container_identity": f"deepseek-private-{name}",
            "image_digest": _sha(f"image-{index}"),
            "model_digest": _sha(f"model-{index}"),
            "configuration_digest": _sha(f"config-{index}"),
            "executable_digest": _sha(f"executable-{index}"),
            "health": {"method": "GET", "path": "/v1/models", "expected_status": 200, "response_digest": _sha(f"health-{index}")},
            "smoke": {"input_digest": _sha(f"smoke-in-{index}"), "output_digest": _sha(f"smoke-out-{index}")},
        })
    return {
        "schema_version": "egv-protected-restore-inventory-v1",
        "inventory_id": "deepseek-private-inventory",
        "compose_project": "deepseek-v4-flash",
        "compose_file": "docker-compose.dspark.yml",
        "env_file": ".env.dspark",
        "services": services,
        "dependency_order": ["private-api", "private-worker"],
        "expected_service_count": 2,
        "resource_baseline_digest": _sha("resources"),
    }


def _load_inventory(root: Path, value=None) -> ProtectedRestoreInventory:
    path = root / "protected.json"
    path.write_bytes(canonical_bytes(_inventory_mapping() if value is None else value))
    with path.open("rb") as handle:
        return load_protected_inventory_fd(handle.fileno())


def _snapshot(inventory: ProtectedRestoreInventory, running: bool = True, **overrides) -> dict:
    services = []
    for item in inventory.services:
        row = {
            "logical_id": item.logical_id,
            "running": running,
            "container_identity": item.container_identity,
            "image_digest": item.image_digest,
            "model_digest": item.model_digest,
            "configuration_digest": item.configuration_digest,
            "executable_digest": item.executable_digest,
            "health_status": item.health.expected_status,
            "health_response_digest": item.health.response_digest,
            "smoke_input_digest": item.smoke.input_digest,
            "smoke_output_digest": item.smoke.output_digest,
        }
        row.update(overrides)
        services.append(row)
    return {"services": services, "resource_baseline_digest": inventory.resource_baseline_digest}


class FakeExecutor:
    def __init__(self, results) -> None:
        self.results = list(results)
        self.plans = []

    def __call__(self, plan):
        self.plans.append(plan)
        if not self.results:
            raise AssertionError("unexpected command execution")
        return self.results.pop(0)


def _locked_advance_worker(path: str, digest: str, ready, result) -> None:
    ready.set()
    try:
        state = CampaignStateStore(Path(path)).advance("P1", expected_digest=digest)
        result.put(("ok", state.digest))
    except Exception as exc:
        result.put(("error", type(exc).__name__))


class CampaignLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.inventory = _load_inventory(self.root)
        self.signer = ReceiptSigner.generate()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _state_at(self, phase: int) -> tuple:
        store = CampaignStateStore(self.root / f"state-{phase}.json")
        state = store.initialize(campaign_id="campaign-public", inventory_digest=self.inventory.digest)
        for index in range(1, phase + 1):
            state = store.advance(f"P{index}", expected_digest=state.digest)
        return store, state

    def test_state_is_monotonic_without_skips_or_regression(self) -> None:
        store, state = self._state_at(1)
        with self.assertRaisesRegex(CampaignStateError, "skip or regress"):
            store.advance("P3", expected_digest=state.digest)
        with self.assertRaisesRegex(CampaignStateError, "skip or regress"):
            store.advance("P0", expected_digest=state.digest)
        advanced = store.advance("P2", expected_digest=state.digest)
        self.assertEqual((advanced.phase, advanced.sequence), ("P2", 2))

    def test_stale_state_digest_and_crash_before_replace_preserve_old_state(self) -> None:
        store, state = self._state_at(0)
        with self.assertRaises(CampaignStateError):
            store.advance("P1", expected_digest=_sha("stale"))
        with mock.patch("egv.campaign.state._durable_replace", side_effect=OSError("simulated crash")):
            with self.assertRaisesRegex(OSError, "simulated crash"):
                store.advance("P1", expected_digest=state.digest)
        self.assertEqual(store.load().digest, state.digest)
        self.assertFalse(list(self.root.glob(".campaign-state-*")))

    def test_cross_process_lock_covers_complete_compare_and_swap(self) -> None:
        import multiprocessing
        import queue

        store, state = self._state_at(0)
        context = multiprocessing.get_context("spawn")
        ready = context.Event()
        result = context.Queue()
        with _exclusive_path_lock(store.lock_path):
            process = context.Process(
                target=_locked_advance_worker,
                args=(str(store.path), state.digest, ready, result),
            )
            process.start()
            self.assertTrue(ready.wait(5))
            with self.assertRaises(queue.Empty):
                result.get(timeout=0.25)
        outcome = result.get(timeout=5)
        process.join(5)
        self.assertEqual((process.exitcode, outcome[0]), (0, "ok"))
        with self.assertRaises(CampaignStateError):
            store.advance("P1", expected_digest=state.digest)

    def test_inventory_requires_already_open_regular_fd(self) -> None:
        read_fd, write_fd = os.pipe()
        try:
            with self.assertRaisesRegex(ProtectedInventoryError, "regular file"):
                load_protected_inventory_fd(read_fd)
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_inventory_rejects_secret_and_command_line_fields(self) -> None:
        for field, value in (("pass" + "word", "redacted-fixture"), ("argv", ["docker", "rm", "all"])):
            raw = _inventory_mapping()
            raw[field] = value
            path = self.root / f"bad-{field}.json"
            path.write_bytes(canonical_bytes(raw))
            with path.open("rb") as handle:
                with self.assertRaises(ProtectedInventoryError):
                    load_protected_inventory_fd(handle.fileno())

    def test_ambiguous_inventory_executes_no_commands(self) -> None:
        raw = _inventory_mapping()
        raw["services"][1]["compose_service"] = raw["services"][0]["compose_service"]
        executor = FakeExecutor([])
        with self.assertRaisesRegex(ProtectedInventoryError, "ambiguous"):
            inventory = _load_inventory(self.root, raw)
            DeepSeekLifecycleController(inventory, CampaignStateStore(self.root / "state.json"), executor)
        self.assertEqual(executor.plans, [])

    def test_exact_compose_env_file_and_dependency_order_plans(self) -> None:
        prefix = (
            "docker", "compose", "--env-file", ".env.dspark", "-f",
            "docker-compose.dspark.yml", "-p", "deepseek-v4-flash",
        )
        self.assertEqual(capture_plan(self.inventory).argv, prefix + ("ps", "--format", "json"))
        self.assertEqual(stop_plan(self.inventory).argv, prefix + ("stop", "--timeout", "120", "worker", "api"))
        self.assertEqual(restore_plan(self.inventory).argv, prefix + ("up", "-d", "--no-build", "api", "worker"))

    def test_successful_stop_is_durably_restore_required(self) -> None:
        store, state = self._state_at(3)
        executor = FakeExecutor([
            CommandResult(0, _snapshot(self.inventory, True)),
            CommandResult(0, _snapshot(self.inventory, False)),
        ])
        final = DeepSeekLifecycleController(self.inventory, store, executor).stop(expected_state_digest=state.digest)
        self.assertTrue(final.restore_required)
        self.assertEqual(final.service_state, "STOPPED")
        self.assertEqual([plan.action for plan in executor.plans], ["CAPTURE", "STOP"])

    def test_partial_stop_failure_forces_restore_required_and_blocks_advance(self) -> None:
        store, state = self._state_at(3)
        executor = FakeExecutor([
            CommandResult(0, _snapshot(self.inventory, True)),
            CommandResult(9, _snapshot(self.inventory, False)),
        ])
        final = DeepSeekLifecycleController(self.inventory, store, executor).stop(expected_state_digest=state.digest)
        self.assertEqual((final.restore_required, final.service_state), (True, "PARTIAL"))
        with self.assertRaisesRegex(CampaignStateError, "incomplete"):
            store.advance("P4", expected_digest=final.digest)

    def test_unknown_stop_effect_after_executor_crash_forces_restore(self) -> None:
        store, state = self._state_at(3)

        class CrashingExecutor(FakeExecutor):
            def __call__(self, plan):
                self.plans.append(plan)
                if plan.action == "CAPTURE":
                    return CommandResult(0, _snapshot(self_inventory, True))
                raise OSError("transport lost after stop dispatch")

        self_inventory = self.inventory
        executor = CrashingExecutor([])
        final = DeepSeekLifecycleController(self.inventory, store, executor).stop(expected_state_digest=state.digest)
        self.assertEqual((final.restore_required, final.service_state), (True, "PARTIAL"))

    def test_stop_intent_is_durable_before_dispatch_and_reconciles_after_hard_crash(self) -> None:
        store, state = self._state_at(3)

        class HardCrashExecutor(FakeExecutor):
            def __call__(self, plan):
                self.plans.append(plan)
                if plan.action == "CAPTURE":
                    return CommandResult(0, _snapshot(self_inventory, True))
                observed = store.load()
                self_observed.append((observed.restore_required, observed.service_state))
                raise KeyboardInterrupt("simulated process death")

        self_inventory = self.inventory
        self_observed = []
        controller = DeepSeekLifecycleController(self.inventory, store, HardCrashExecutor([]))
        with self.assertRaises(KeyboardInterrupt):
            controller.stop(expected_state_digest=state.digest)
        intent = store.load()
        self.assertEqual(self_observed, [(True, "STOPPING")])
        self.assertEqual((intent.restore_required, intent.service_state), (True, "STOPPING"))
        reconciled = controller.reconcile_stop_intent(expected_state_digest=intent.digest)
        self.assertEqual((reconciled.restore_required, reconciled.service_state), (True, "PARTIAL"))

    def test_restore_verifies_all_bindings_and_clears_requirement(self) -> None:
        store, state = self._state_at(3)
        stopped = store.require_restore(expected_digest=state.digest, service_state="PARTIAL")
        executor = FakeExecutor([CommandResult(0, _snapshot(self.inventory, True))])
        snapshot = VerificationSnapshot.from_mapping(_snapshot(self.inventory, True))
        receipt = public_restore_receipt(
            self.inventory, snapshot, campaign_id=stopped.campaign_id, signer=self.signer
        )
        restored = DeepSeekLifecycleController(self.inventory, store, executor).restore(
            expected_state_digest=stopped.digest,
            public_receipt=receipt,
            evaluator_public_key=self.signer.public_key,
        )
        self.assertFalse(restored.restore_required)
        self.assertEqual((restored.phase, restored.service_state), ("P3", "RESTORED"))

    def test_state_store_rejects_legacy_bare_restore_digest(self) -> None:
        store, state = self._state_at(3)
        stopped = store.require_restore(expected_digest=state.digest, service_state="PARTIAL")
        with self.assertRaises(TypeError):
            store.mark_restored(expected_digest=stopped.digest, verified_receipt=_sha("unverified"))
        self.assertTrue(store.load().restore_required)

    def test_importable_restore_capability_forge_is_absent_and_cannot_clear_state(self) -> None:
        import egv.campaign.state as state_module

        self.assertFalse(hasattr(state_module, "_verified_restore_receipt"))
        self.assertFalse(hasattr(state_module, "VerifiedRestoreReceipt"))
        store, state = self._state_at(3)
        stopped = store.require_restore(expected_digest=state.digest, service_state="PARTIAL")
        forged = {"private_inventory_digest": "0" * 64, "campaign_id": stopped.campaign_id}
        with self.assertRaises(CampaignStateError):
            store.mark_restored(
                expected_digest=stopped.digest,
                public_receipt=forged,
                evaluator_public_key=b"0" * 32,
                expected_payload=forged,
            )
        self.assertEqual(store.load().digest, stopped.digest)

    def test_restore_identity_mismatch_fails_and_remains_required(self) -> None:
        store, state = self._state_at(3)
        stopped = store.require_restore(expected_digest=state.digest, service_state="PARTIAL")
        executor = FakeExecutor([CommandResult(0, _snapshot(self.inventory, True, image_digest=_sha("wrong")))])
        receipt = public_restore_receipt(
            self.inventory,
            VerificationSnapshot.from_mapping(_snapshot(self.inventory, True)),
            campaign_id=stopped.campaign_id,
            signer=self.signer,
        )
        with self.assertRaisesRegex(LifecycleExecutionError, "binding"):
            DeepSeekLifecycleController(self.inventory, store, executor).restore(
                expected_state_digest=stopped.digest,
                public_receipt=receipt,
                evaluator_public_key=self.signer.public_key,
            )
        self.assertTrue(store.load().restore_required)
        self.assertEqual(store.load().service_state, "RESTORING")

    def test_public_receipt_contains_no_private_identifiers_or_secrets(self) -> None:
        snapshot = VerificationSnapshot.from_mapping(_snapshot(self.inventory, True))
        receipt = public_restore_receipt(
            self.inventory, snapshot, campaign_id="campaign-public", signer=self.signer
        )
        encoded = json.dumps(receipt, sort_keys=True)
        for prohibited in ("deepseek-private", "private-api", "private-worker", "password", "token", ".env.dspark"):
            self.assertNotIn(prohibited, encoded)
        self.assertEqual(receipt["logical_service_ids"], ["svc-001", "svc-002"])
        self.assertEqual(receipt["schema_version"], "egv-public-restore-v1")

    def test_restore_rejects_wrong_key_tamper_and_snapshot_binding_without_clearing_intent(self) -> None:
        for mode in ("wrong-key", "tamper", "campaign"):
            with self.subTest(mode=mode):
                store = CampaignStateStore(self.root / ("restore-{}.json".format(mode)))
                state = store.initialize(
                    campaign_id="campaign-public",
                    inventory_digest=self.inventory.digest,
                )
                for index in range(1, 4):
                    state = store.advance("P{}".format(index), expected_digest=state.digest)
                stopped = store.require_restore(expected_digest=state.digest, service_state="PARTIAL")
                snapshot = VerificationSnapshot.from_mapping(_snapshot(self.inventory, True))
                receipt = public_restore_receipt(
                    self.inventory, snapshot, campaign_id=stopped.campaign_id, signer=self.signer
                )
                public_key = self.signer.public_key
                if mode == "wrong-key":
                    public_key = ReceiptSigner.generate().public_key
                elif mode == "tamper":
                    receipt = dict(receipt)
                    receipt["smoke_output_digest"] = _sha("substituted")
                else:
                    other = dict(receipt)
                    payload = {
                        key: value
                        for key, value in other.items()
                        if key not in {"schema_version", "receipt_id", "signing_key_id", "signature"}
                    }
                    payload["campaign_id"] = "campaign-other"
                    from egv.public import build_public_restore_receipt

                    receipt = build_public_restore_receipt(payload, self.signer)
                executor = FakeExecutor([CommandResult(0, _snapshot(self.inventory, True))])
                with self.assertRaises(LifecycleExecutionError):
                    DeepSeekLifecycleController(self.inventory, store, executor).restore(
                        expected_state_digest=stopped.digest,
                        public_receipt=receipt,
                        evaluator_public_key=public_key,
                    )
                final = store.load()
                self.assertTrue(final.restore_required)
                self.assertEqual(final.service_state, "RESTORING")


if __name__ == "__main__":
    unittest.main()
