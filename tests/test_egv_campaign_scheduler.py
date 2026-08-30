from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from egv.canonical import canonical_bytes, digest_for
import egv.campaign as campaign_package
from egv.campaign.scheduler import (
    AVOIDABLE_IDLE_DEFECT_MS,
    CLAIM_BOUNDARY,
    PILOT_CHECKPOINT_REQUEST,
    STALL_TIMEOUT_MS,
    CampaignScheduler,
    CampaignSchedulerError,
    SchedulerPlan,
    SchedulerTask,
    SchedulerWorker,
    validate_public_scheduler_report,
)


def _sha(label: str) -> str:
    return digest_for(label)


def _task(
    node_id: str,
    kind: str,
    resource_pool: str,
    dependencies=tuple(),
    *,
    arm_id=None,
    request_index=None,
    promotion_key=None,
) -> SchedulerTask:
    owner_by_kind = {
        "DIAGNOSTIC": "CONTROLLER",
        "EVALUATION": "EVALUATOR",
        "GENERATION": "TRAINER",
        "PROMOTION": "CONTROLLER",
        "REVIEW": "REVIEWER",
        "TRAINING": "TRAINER",
        "VALIDATION": "EVALUATOR",
    }
    return SchedulerTask(
        node_id=node_id,
        kind=kind,
        resource_pool=resource_pool,
        owner_role=owner_by_kind[kind],
        input_digest=_sha("input:{}".format(node_id)),
        dependencies=dependencies,
        arm_id=arm_id,
        request_index=request_index,
        promotion_key=promotion_key,
    )


def _node(scheduler: CampaignScheduler, node_id: str) -> dict:
    return next(node for node in scheduler.snapshot()["nodes"] if node["node_id"] == node_id)


def _arm(scheduler: CampaignScheduler, arm_id: str) -> dict:
    return next(arm for arm in scheduler.snapshot()["arms"] if arm["arm_id"] == arm_id)


class CampaignSchedulerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.state_path = self.root / "scheduler-state.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def initialize(self, workers, tasks) -> CampaignScheduler:
        plan = SchedulerPlan("campaign-scheduler-test", tuple(workers), tuple(tasks))
        return CampaignScheduler.initialize(self.state_path, plan, now_ms=0)

    def test_package_exports_scheduler_surface(self) -> None:
        self.assertIs(campaign_package.CampaignScheduler, CampaignScheduler)
        self.assertIs(campaign_package.CampaignSchedulerError, CampaignSchedulerError)
        self.assertIs(campaign_package.SchedulerPlan, SchedulerPlan)
        self.assertIs(campaign_package.SchedulerTask, SchedulerTask)
        self.assertIs(campaign_package.SchedulerWorker, SchedulerWorker)
        self.assertIs(campaign_package.validate_public_scheduler_report, validate_public_scheduler_report)

    def test_lane_evidence_persists_owner_input_times_resource_output_and_terminal_reason(self) -> None:
        task = _task("generation", "GENERATION", "ACCELERATOR")
        scheduler = self.initialize(
            (SchedulerWorker("accelerator-a", "ACCELERATOR"),),
            (task,),
        )
        dispatch = scheduler.dispatch_ready(now_ms=10)[0]
        output_digest = _sha("generation-output")
        scheduler.complete(
            "generation",
            worker_id=dispatch.worker_id,
            output_digest=output_digest,
            now_ms=20,
        )

        persisted = _node(CampaignScheduler(self.state_path), "generation")
        self.assertEqual(persisted["owner_role"], "TRAINER")
        self.assertEqual(persisted["input_digest"], task.input_digest)
        self.assertEqual(persisted["resource_pool"], "ACCELERATOR")
        self.assertEqual(persisted["assigned_worker_id"], "accelerator-a")
        self.assertEqual(persisted["started_ms"], 10)
        self.assertEqual(persisted["last_progress_ms"], 20)
        self.assertEqual(persisted["output_digest"], output_digest)
        self.assertEqual(persisted["terminal_reason"], "COMPLETED")

    def test_dependency_ready_evaluation_starts_while_unrelated_generation_is_running(self) -> None:
        workers = (
            SchedulerWorker("accelerator-a", "ACCELERATOR"),
            SchedulerWorker("accelerator-b", "ACCELERATOR"),
            SchedulerWorker("cpu-a", "CPU"),
        )
        tasks = [
            _task("generation-a", "GENERATION", "ACCELERATOR"),
            _task("generation-b", "GENERATION", "ACCELERATOR"),
        ]
        for request_index in range(1, PILOT_CHECKPOINT_REQUEST + 1):
            dependency = "generation-a" if request_index == 1 else "generation-b"
            tasks.append(
                _task(
                    "evaluation-{}".format(request_index),
                    "EVALUATION",
                    "CPU",
                    (dependency,),
                    arm_id="arm-l1",
                    request_index=request_index,
                )
            )
        scheduler = self.initialize(workers, tasks)

        initial = {dispatch.node_id: dispatch for dispatch in scheduler.dispatch_ready(now_ms=0)}
        self.assertEqual(set(initial), {"generation-a", "generation-b"})
        scheduler.complete(
            "generation-a",
            worker_id=initial["generation-a"].worker_id,
            output_digest=_sha("generation-a-output"),
            now_ms=100,
        )

        next_dispatches = scheduler.dispatch_ready(now_ms=100)
        self.assertEqual([dispatch.node_id for dispatch in next_dispatches], ["evaluation-1"])
        self.assertEqual(next_dispatches[0].worker_id, "cpu-a")
        self.assertEqual(_node(scheduler, "generation-b")["status"], "RUNNING")
        self.assertEqual(_node(scheduler, "evaluation-2")["status"], "PENDING")

    def test_stall_clock_requires_new_evidence_and_reassigns_once_then_fails_closed(self) -> None:
        scheduler = self.initialize(
            (
                SchedulerWorker("accelerator-a", "ACCELERATOR"),
                SchedulerWorker("accelerator-b", "ACCELERATOR"),
            ),
            (_task("generation", "GENERATION", "ACCELERATOR"),),
        )
        first = scheduler.dispatch_ready(now_ms=0)[0]
        self.assertEqual(first.worker_id, "accelerator-a")
        evidence_digest = _sha("progress-1")
        scheduler.record_progress(
            "generation",
            worker_id=first.worker_id,
            evidence_digest=evidence_digest,
            now_ms=1,
        )
        second_evidence_digest = _sha("progress-2")
        scheduler.record_progress(
            "generation",
            worker_id=first.worker_id,
            evidence_digest=second_evidence_digest,
            now_ms=2,
        )
        duplicate = scheduler.record_progress(
            "generation",
            worker_id=first.worker_id,
            evidence_digest=evidence_digest,
            now_ms=500_000,
        )
        self.assertEqual(duplicate["last_progress_ms"], 2)
        self.assertEqual(scheduler.recover_stalls(now_ms=STALL_TIMEOUT_MS), tuple())

        first_recovery = scheduler.recover_stalls(now_ms=STALL_TIMEOUT_MS + 2)
        self.assertEqual(len(first_recovery), 1)
        self.assertEqual(first_recovery[0].disposition, "RETRY_PENDING")
        scheduler = CampaignScheduler(self.state_path)
        self.assertEqual(_node(scheduler, "generation")["failed_worker_ids"], ["accelerator-a"])
        retry = scheduler.dispatch_ready(now_ms=STALL_TIMEOUT_MS + 2)[0]
        self.assertEqual(retry.worker_id, "accelerator-b")
        self.assertEqual(retry.attempt, 2)

        second_recovery = scheduler.recover_stalls(now_ms=(2 * STALL_TIMEOUT_MS) + 2)
        self.assertEqual(len(second_recovery), 1)
        self.assertEqual(second_recovery[0].disposition, "FAILED_CLOSED")
        terminal = _node(scheduler, "generation")
        self.assertEqual(terminal["status"], "FAILED")
        self.assertEqual(terminal["terminal_reason"], "STALL_RETRY_EXHAUSTED")
        self.assertEqual(terminal["failed_worker_ids"], ["accelerator-a", "accelerator-b"])
        self.assertEqual(scheduler.dispatch_ready(now_ms=(2 * STALL_TIMEOUT_MS) + 2), tuple())

    def test_stall_without_an_alternate_worker_fails_closed_at_first_timeout(self) -> None:
        scheduler = self.initialize(
            (SchedulerWorker("accelerator-a", "ACCELERATOR"),),
            (_task("generation", "GENERATION", "ACCELERATOR"),),
        )
        scheduler.dispatch_ready(now_ms=0)
        recovery = scheduler.recover_stalls(now_ms=STALL_TIMEOUT_MS)
        self.assertEqual(recovery[0].disposition, "FAILED_CLOSED")
        self.assertEqual(_node(scheduler, "generation")["terminal_reason"], "STALL_NO_REASSIGNMENT")

    def test_restart_does_not_duplicate_completion_or_promotion(self) -> None:
        promotion_key = _sha("candidate-and-policy")
        scheduler = self.initialize(
            (SchedulerWorker("cpu-a", "CPU"),),
            (_task("promotion", "PROMOTION", "CPU", promotion_key=promotion_key),),
        )
        dispatch = scheduler.dispatch_ready(now_ms=0)[0]
        output_digest = _sha("sealed-promotion")
        scheduler.complete(
            "promotion",
            worker_id=dispatch.worker_id,
            output_digest=output_digest,
            now_ms=10,
        )
        sequence = scheduler.snapshot()["sequence"]

        resumed = CampaignScheduler(self.state_path)
        self.assertEqual(resumed.dispatch_ready(now_ms=10), tuple())
        exact_retry = resumed.complete(
            "promotion",
            worker_id=dispatch.worker_id,
            output_digest=output_digest,
            now_ms=10,
        )
        self.assertEqual(exact_retry["status"], "COMPLETED")
        self.assertEqual(resumed.snapshot()["sequence"], sequence)
        self.assertEqual(
            sum(
                1
                for node in resumed.snapshot()["nodes"]
                if node["kind"] == "PROMOTION" and node["status"] == "COMPLETED"
            ),
            1,
        )
        with self.assertRaisesRegex(CampaignSchedulerError, "conflicts"):
            resumed.complete(
                "promotion",
                worker_id=dispatch.worker_id,
                output_digest=_sha("substituted-promotion"),
                now_ms=10,
            )

        raw = self.state_path.read_bytes()
        self.assertEqual(raw, canonical_bytes(json.loads(raw.decode("utf-8"))))

    def test_restart_exact_failure_retry_is_idempotent(self) -> None:
        scheduler = self.initialize(
            (SchedulerWorker("cpu-a", "CPU"),),
            (_task("review", "REVIEW", "CPU"),),
        )
        dispatch = scheduler.dispatch_ready(now_ms=0)[0]
        scheduler.fail("review", worker_id=dispatch.worker_id, now_ms=5)
        sequence = scheduler.snapshot()["sequence"]

        resumed = CampaignScheduler(self.state_path)
        retry = resumed.fail("review", worker_id=dispatch.worker_id, now_ms=5)
        self.assertEqual(retry["terminal_reason"], "ACTIVITY_FAILED")
        self.assertEqual(resumed.snapshot()["sequence"], sequence)

    def test_plan_rejects_duplicate_promotion_authority(self) -> None:
        key = _sha("one-promotion-authority")
        with self.assertRaisesRegex(CampaignSchedulerError, "duplicate promotion"):
            SchedulerPlan(
                "campaign-duplicate-promotion",
                (SchedulerWorker("cpu-a", "CPU"),),
                (
                    _task("promotion-a", "PROMOTION", "CPU", promotion_key=key),
                    _task("promotion-b", "PROMOTION", "CPU", promotion_key=key),
                ),
            )

    def test_plan_rejects_deadlock_and_string_dependency_smuggling(self) -> None:
        with self.assertRaisesRegex(CampaignSchedulerError, "ordered sequence of IDs"):
            _task("string-dependency", "REVIEW", "CPU", "review-a")
        with self.assertRaisesRegex(CampaignSchedulerError, "cycle"):
            SchedulerPlan(
                "campaign-cycle",
                (SchedulerWorker("cpu-a", "CPU"),),
                (
                    _task("review-a", "REVIEW", "CPU", ("review-b",)),
                    _task("review-b", "REVIEW", "CPU", ("review-a",)),
                ),
            )

    def _checkpoint_scheduler(self) -> CampaignScheduler:
        tasks = []
        evaluation_ids = []
        for request_index in range(1, PILOT_CHECKPOINT_REQUEST + 1):
            generation_id = "generation-{}".format(request_index)
            evaluation_id = "evaluation-{}".format(request_index)
            evaluation_ids.append(evaluation_id)
            tasks.extend(
                [
                    _task(
                        generation_id,
                        "GENERATION",
                        "ACCELERATOR",
                        arm_id="arm-l1",
                        request_index=request_index,
                    ),
                    _task(
                        evaluation_id,
                        "EVALUATION",
                        "CPU",
                        (generation_id,),
                        arm_id="arm-l1",
                        request_index=request_index,
                    ),
                ]
            )
        tasks.append(
            _task(
                "generation-7",
                "GENERATION",
                "ACCELERATOR",
                arm_id="arm-l1",
                request_index=PILOT_CHECKPOINT_REQUEST + 1,
            )
        )
        tasks.append(
            _task(
                "frozen-diagnostic",
                "DIAGNOSTIC",
                "CPU",
                tuple(evaluation_ids),
                arm_id="arm-l1",
            )
        )
        return self.initialize(
            (
                SchedulerWorker("accelerator-a", "ACCELERATOR"),
                SchedulerWorker("cpu-a", "CPU"),
            ),
            tasks,
        )

    def _run_to_checkpoint(self, scheduler: CampaignScheduler, passing_requests, failed_requests=()) -> int:
        now_ms = 0
        for _ in range(100):
            if _arm(scheduler, "arm-l1")["checkpoint_complete"]:
                return now_ms
            dispatches = scheduler.dispatch_ready(now_ms=now_ms)
            self.assertTrue(dispatches, "checkpoint fixture deadlocked")
            running = {node["node_id"]: node for node in scheduler.snapshot()["nodes"]}
            for dispatch in dispatches:
                now_ms += 1
                node = running[dispatch.node_id]
                if node["kind"] == "EVALUATION" and node["request_index"] in failed_requests:
                    scheduler.fail(dispatch.node_id, worker_id=dispatch.worker_id, now_ms=now_ms)
                    continue
                kwargs = {}
                if node["kind"] == "EVALUATION":
                    kwargs["semantic_passed"] = node["request_index"] in passing_requests
                scheduler.complete(
                    dispatch.node_id,
                    worker_id=dispatch.worker_id,
                    output_digest=_sha("{}-output".format(dispatch.node_id)),
                    now_ms=now_ms,
                    **kwargs,
                )
        self.fail("checkpoint fixture exceeded its deterministic step bound")

    def test_zero_semantic_signal_pauses_arm_at_frozen_request_six(self) -> None:
        scheduler = self._checkpoint_scheduler()
        now_ms = self._run_to_checkpoint(scheduler, passing_requests=set())
        arm = _arm(scheduler, "arm-l1")
        self.assertEqual(arm["status"], "PAUSED_ZERO_SIGNAL")
        self.assertEqual(arm["checkpoint_terminal_requests"], list(range(1, 7)))
        self.assertEqual(arm["checkpoint_semantic_passes"], 0)
        future = _node(scheduler, "generation-7")
        self.assertEqual(future["status"], "CANCELLED")
        self.assertEqual(future["terminal_reason"], "ZERO_SEMANTIC_SIGNAL_AT_FROZEN_CHECKPOINT")

        diagnostic = scheduler.dispatch_ready(now_ms=now_ms)
        self.assertEqual([dispatch.node_id for dispatch in diagnostic], ["frozen-diagnostic"])

    def test_frozen_diagnostic_runs_after_terminal_evaluator_failure(self) -> None:
        scheduler = self._checkpoint_scheduler()
        now_ms = self._run_to_checkpoint(scheduler, passing_requests=set(), failed_requests={3})
        self.assertEqual(_arm(scheduler, "arm-l1")["status"], "PAUSED_ZERO_SIGNAL")
        self.assertEqual(_node(scheduler, "evaluation-3")["status"], "FAILED")
        diagnostic = scheduler.dispatch_ready(now_ms=now_ms)
        self.assertEqual([dispatch.node_id for dispatch in diagnostic], ["frozen-diagnostic"])

    def test_nonzero_checkpoint_signal_unlocks_post_checkpoint_work(self) -> None:
        scheduler = self._checkpoint_scheduler()
        now_ms = self._run_to_checkpoint(scheduler, passing_requests={2})
        arm = _arm(scheduler, "arm-l1")
        self.assertEqual(arm["status"], "CONTINUING")
        self.assertEqual(arm["checkpoint_semantic_passes"], 1)
        next_dispatches = scheduler.dispatch_ready(now_ms=now_ms)
        self.assertEqual(
            {dispatch.node_id for dispatch in next_dispatches},
            {"frozen-diagnostic", "generation-7"},
        )

    def test_request_indexed_arm_requires_exact_frozen_checkpoint_evaluations(self) -> None:
        with self.assertRaisesRegex(CampaignSchedulerError, "frozen pilot checkpoint"):
            SchedulerPlan(
                "campaign-incomplete-checkpoint",
                (
                    SchedulerWorker("accelerator-a", "ACCELERATOR"),
                    SchedulerWorker("cpu-a", "CPU"),
                ),
                (
                    _task(
                        "generation-1",
                        "GENERATION",
                        "ACCELERATOR",
                        arm_id="arm-l1",
                        request_index=1,
                    ),
                    _task(
                        "evaluation-1",
                        "EVALUATION",
                        "CPU",
                        ("generation-1",),
                        arm_id="arm-l1",
                        request_index=1,
                    ),
                ),
            )

    def test_utilization_accounts_for_busy_idle_and_avoidable_idle_defect(self) -> None:
        scheduler = self.initialize(
            (SchedulerWorker("accelerator-a", "ACCELERATOR"),),
            (_task("generation", "GENERATION", "ACCELERATOR"),),
        )
        aliases = {"accelerator-a": "slot-a"}
        at_threshold = scheduler.public_report(
            now_ms=AVOIDABLE_IDLE_DEFECT_MS,
            worker_aliases=aliases,
        )
        self.assertFalse(at_threshold["aggregate"]["avoidable_idle_defect"])
        report = scheduler.public_report(
            now_ms=AVOIDABLE_IDLE_DEFECT_MS + 1,
            worker_aliases=aliases,
        )
        self.assertEqual(report["aggregate"]["avoidable_idle_ms"], AVOIDABLE_IDLE_DEFECT_MS + 1)
        self.assertTrue(report["aggregate"]["avoidable_idle_defect"])

        scheduler.dispatch_ready(now_ms=AVOIDABLE_IDLE_DEFECT_MS + 1)
        after_busy = scheduler.public_report(
            now_ms=AVOIDABLE_IDLE_DEFECT_MS + 101,
            worker_aliases=aliases,
        )
        self.assertEqual(after_busy["aggregate"]["busy_ms"], 100)
        self.assertEqual(
            after_busy["aggregate"]["busy_ms"] + after_busy["aggregate"]["idle_ms"],
            after_busy["aggregate"]["capacity_ms"],
        )
        self.assertEqual(after_busy["claim_boundary"], CLAIM_BOUNDARY)

    def test_avoidable_idle_matching_does_not_count_more_workers_than_runnable_tasks(self) -> None:
        scheduler = self.initialize(
            (
                SchedulerWorker("accelerator-a", "ACCELERATOR"),
                SchedulerWorker("accelerator-b", "ACCELERATOR"),
            ),
            (_task("generation", "GENERATION", "ACCELERATOR"),),
        )
        report = scheduler.public_report(
            now_ms=100,
            worker_aliases={"accelerator-a": "slot-a", "accelerator-b": "slot-b"},
        )
        self.assertEqual(report["aggregate"]["idle_ms"], 200)
        self.assertEqual(report["aggregate"]["avoidable_idle_ms"], 100)
        self.assertEqual(report["aggregate"]["legitimately_blocked_ms"], 100)

    def test_public_projection_rejects_path_secret_topology_and_schema_drift(self) -> None:
        scheduler = self.initialize(
            (SchedulerWorker("accelerator-a", "ACCELERATOR"),),
            (_task("generation", "GENERATION", "ACCELERATOR"),),
        )
        for unsafe_alias in (
            "10.0.0.24",
            "C:\\Users\\private",
            "private-slot",
            "spark-a",
            "api-token",
            "worker-a",
            "unstructuredalias",
        ):
            with self.subTest(alias=unsafe_alias):
                with self.assertRaises(CampaignSchedulerError):
                    scheduler.public_report(
                        now_ms=0,
                        worker_aliases={"accelerator-a": unsafe_alias},
                    )

        report = scheduler.public_report(now_ms=1, worker_aliases={"accelerator-a": "slot-a"})
        self.assertEqual(validate_public_scheduler_report(report), report)
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn("accelerator-a", serialized)
        self.assertNotIn("generation", serialized)
        self.assertNotIn("TRAINER", serialized)
        self.assertNotIn(_sha("input:generation"), serialized)
        leaked = deepcopy(report)
        leaked["private_path"] = "C:\\Users\\private"
        with self.assertRaisesRegex(CampaignSchedulerError, "closed schema"):
            validate_public_scheduler_report(leaked)
        secret_alias = deepcopy(report)
        secret_alias["workers"][0]["alias"] = "secret-value"
        with self.assertRaisesRegex(CampaignSchedulerError, "secret or topology"):
            validate_public_scheduler_report(secret_alias)
        drifted = deepcopy(report)
        drifted["aggregate"]["busy_ms"] += 1
        with self.assertRaises(CampaignSchedulerError):
            validate_public_scheduler_report(drifted)

        false_blocking = deepcopy(report)
        false_blocking["workers"][0]["legitimately_blocked_ms"] += 1
        false_blocking["aggregate"]["legitimately_blocked_ms"] += 1
        with self.assertRaisesRegex(CampaignSchedulerError, "legitimate blocking"):
            validate_public_scheduler_report(false_blocking)

    def test_noncanonical_or_open_persisted_state_is_rejected_on_restart(self) -> None:
        scheduler = self.initialize(
            (SchedulerWorker("cpu-a", "CPU"),),
            (_task("review", "REVIEW", "CPU"),),
        )
        value = scheduler.snapshot()
        value["nodes"][0]["private_path"] = "C:\\private"
        self.state_path.write_bytes(canonical_bytes(value))
        with self.assertRaisesRegex(CampaignSchedulerError, "closed schema"):
            CampaignScheduler(self.state_path)

    def test_persisted_worker_time_must_reconcile_to_scheduler_window(self) -> None:
        scheduler = self.initialize(
            (SchedulerWorker("cpu-a", "CPU"),),
            (_task("review", "REVIEW", "CPU"),),
        )
        scheduler.public_report(now_ms=10, worker_aliases={"cpu-a": "slot-a"})
        value = scheduler.snapshot()
        value["workers"][0]["idle_ms"] += 1
        self.state_path.write_bytes(canonical_bytes(value))
        with self.assertRaisesRegex(CampaignSchedulerError, "does not reconcile"):
            CampaignScheduler(self.state_path)


if __name__ == "__main__":
    unittest.main()
