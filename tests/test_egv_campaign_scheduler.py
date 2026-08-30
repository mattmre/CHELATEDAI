from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from egv.canonical import GENESIS_HASH, canonical_bytes, digest_for
import egv.campaign as campaign_package
from egv.campaign.scheduler import (
    AVOIDABLE_IDLE_DEFECT_MS,
    CLAIM_BOUNDARY,
    PILOT_CHECKPOINT_REQUEST,
    PROGRESS_CHECKPOINT_MS,
    STALL_TIMEOUT_MS,
    STATE_AUTHORITY,
    CampaignScheduler,
    CampaignSchedulerError,
    SchedulerPlan,
    SchedulerTask,
    SchedulerWorker,
    validate_public_scheduler_report,
)
from egv.receipts import ReceiptSigner, receipt_hash


def _sha(label: str) -> str:
    return digest_for(label)


def _task(
    node_id: str,
    kind: str,
    resource_pool: str,
    owner_role: str,
    *,
    stage: str = "SCREENING",
    dependencies=tuple(),
    arm_id=None,
    request_index=None,
    promotion_key=None,
    promotion_authority_node_id=None,
) -> SchedulerTask:
    return SchedulerTask(
        node_id=node_id,
        kind=kind,
        resource_pool=resource_pool,
        owner_role=owner_role,
        input_digest=_sha("input:{}".format(node_id)),
        stage=stage,
        dependencies=dependencies,
        arm_id=arm_id,
        request_index=request_index,
        promotion_key=promotion_key,
        promotion_authority_node_id=promotion_authority_node_id,
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
        self.signers = {}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def worker(self, worker_id, pool, role, capabilities) -> SchedulerWorker:
        signer = ReceiptSigner.generate()
        self.signers[worker_id] = signer
        return SchedulerWorker(worker_id, pool, role, tuple(capabilities), signer.public_key_raw)

    def initialize(self, workers, tasks) -> CampaignScheduler:
        return CampaignScheduler.initialize(
            self.state_path,
            SchedulerPlan("campaign-scheduler-test", tuple(workers), tuple(tasks)),
            now_ms=0,
        )

    def receipt(
        self,
        scheduler: CampaignScheduler,
        node_id: str,
        effect_kind: str,
        *,
        output_digest=None,
        decision=None,
        signer_id=None,
    ) -> dict:
        state = scheduler.snapshot()
        node = next(item for item in state["nodes"] if item["node_id"] == node_id)
        attempt = node["attempt_receipts"][-1]
        signer_id = signer_id or attempt["worker_id"]
        signer = self.signers[signer_id]
        sequence = len(attempt["receipts"]) + 1
        previous = receipt_hash(attempt["receipts"][-1]) if attempt["receipts"] else GENESIS_HASH
        receipt_type = "VERDICT" if effect_kind == "SCHEDULER_VERDICT" else "EFFECT"
        if decision is None:
            decision = {
                "SCHEDULER_PROGRESS": "ALLOW",
                "SCHEDULER_COMPLETION": "ALLOW",
                "SCHEDULER_FAILURE": "ERROR",
                "SCHEDULER_CANCELLATION": "ALLOW",
                "SCHEDULER_VERDICT": "PASS",
            }[effect_kind]
        if effect_kind == "SCHEDULER_PROGRESS" and output_digest is None:
            progress_kind = {
                "GENERATION": "GENERATED_ARTIFACT",
                "EVALUATION": "EVALUATION_ARTIFACT",
                "DIAGNOSTIC": "DIAGNOSTIC_ARTIFACT",
                "TRAINING": "TRAINING_CHECKPOINT",
                "VALIDATION": "VALIDATION_ARTIFACT",
                "PROMOTION": "PROMOTION_ARTIFACT",
                "REVIEW": "REVIEW_ARTIFACT",
            }[node["kind"]]
            output_digest = digest_for(
                {
                    "attempt_id": attempt["attempt_id"],
                    "node_progress_kind": progress_kind,
                    "progress_ordinal": sequence,
                    "previous_progress_digest": (
                        attempt["receipts"][-1]["output_digest"]
                        if attempt["receipts"] and attempt["receipts"][-1]["effect_kind"] == "SCHEDULER_PROGRESS"
                        else GENESIS_HASH
                    ),
                    "manifest_digest": attempt["manifest_digest"],
                }
            )
        fields = {
            "receipt_type": receipt_type,
            "campaign_id": state["campaign_id"],
            "run_id": state["campaign_id"],
            "task_id": node_id,
            "request_id": attempt["attempt_id"],
            "candidate_id": attempt["attempt_id"],
            "decision": decision,
            "input_digest": attempt["resolved_input_digest"],
            "output_digest": output_digest or _sha("{}:{}:{}".format(node_id, effect_kind, sequence)),
            "resource_bucket": node["resource_pool"],
            "effect_kind": effect_kind,
        }
        progress_kind = (
            {
                "GENERATION": "GENERATED_ARTIFACT",
                "EVALUATION": "EVALUATION_ARTIFACT",
                "DIAGNOSTIC": "DIAGNOSTIC_ARTIFACT",
                "TRAINING": "TRAINING_CHECKPOINT",
                "VALIDATION": "VALIDATION_ARTIFACT",
                "PROMOTION": "PROMOTION_ARTIFACT",
                "REVIEW": "REVIEW_ARTIFACT",
            }[node["kind"]]
            if effect_kind == "SCHEDULER_PROGRESS"
            else "NONE"
        )
        terminal_binding = digest_for(
            {
                "attempt_id": attempt["attempt_id"],
                "effect_kind": effect_kind,
                "decision": decision,
                "output_digest": fields["output_digest"],
                "release_digest": attempt["release_digest"],
            }
        )
        fields.update(
            {
                "protocol_digest": attempt["protocol_digest"],
                "policy_digest": digest_for(
                    {
                        field: attempt[field]
                        for field in (
                            "budget_limit",
                            "budget_used",
                            "predecessor_attempt_id",
                            "predecessor_receipt_hash",
                            "lease_id",
                            "release_digest",
                            "model_digest",
                            "monotonic_counter",
                        )
                    }
                ),
                "arm_policy_digest": attempt["checkpoint_digest"],
                "evaluator_digest": attempt["evaluator_digest"],
                "candidate_artifact_digest": fields["output_digest"],
                "environment_diff_digest": digest_for(
                    {field: attempt[field] for field in ("isolation_digest", "capability_digest", "sandbox_digest")}
                ),
                "normalized_action_hash": digest_for(
                    {"progress_kind": progress_kind, "terminal_binding": terminal_binding}
                ),
                "public_dependency_set_digest": attempt["manifest_digest"],
                "sandbox_id": attempt["lease_id"],
            }
        )
        return signer.sign_receipt(
            fields,
            sequence=sequence,
            previous_receipt_hash=previous,
            idempotency_key=digest_for(
                {
                    "attempt_id": attempt["attempt_id"],
                    "sequence": sequence,
                    "effect_kind": effect_kind,
                }
            ),
        )

    def complete_dispatch(self, scheduler, dispatch, *, now_ms, output_digest=None, semantic_passed=None):
        node = _node(scheduler, dispatch.node_id)
        effect = "SCHEDULER_VERDICT" if node["kind"] in {"EVALUATION", "VALIDATION"} else "SCHEDULER_COMPLETION"
        decision = None
        if effect == "SCHEDULER_VERDICT":
            decision = "PASS" if semantic_passed else "FAIL"
        receipt = self.receipt(
            scheduler,
            dispatch.node_id,
            effect,
            output_digest=output_digest,
            decision=decision,
        )
        return scheduler.complete(
            dispatch.node_id,
            worker_id=dispatch.worker_id,
            receipt=receipt,
            now_ms=now_ms,
        ), receipt

    def fail_dispatch(self, scheduler, dispatch, *, now_ms):
        receipt = self.receipt(scheduler, dispatch.node_id, "SCHEDULER_FAILURE")
        return scheduler.fail(
            dispatch.node_id,
            worker_id=dispatch.worker_id,
            receipt=receipt,
            now_ms=now_ms,
        )

    def test_package_exports_scheduler_surface(self) -> None:
        self.assertIs(campaign_package.CampaignScheduler, CampaignScheduler)
        self.assertIs(campaign_package.CampaignSchedulerError, CampaignSchedulerError)
        self.assertIs(campaign_package.SchedulerPlan, SchedulerPlan)
        self.assertIs(campaign_package.SchedulerTask, SchedulerTask)
        self.assertIs(campaign_package.SchedulerWorker, SchedulerWorker)

    def test_dependency_ready_evaluation_starts_while_unrelated_generation_runs(self) -> None:
        workers = (
            self.worker("trainer-a", "ACCELERATOR", "TRAINER", ("GENERATION",)),
            self.worker("trainer-b", "ACCELERATOR", "TRAINER", ("GENERATION",)),
            self.worker("evaluator-a", "CPU", "EVALUATOR", ("EVALUATION",)),
        )
        tasks = [
            _task("generation-a", "GENERATION", "ACCELERATOR", "TRAINER"),
            _task("generation-b", "GENERATION", "ACCELERATOR", "TRAINER"),
        ]
        for request_index in range(1, PILOT_CHECKPOINT_REQUEST + 1):
            tasks.append(
                _task(
                    "evaluation-{}".format(request_index),
                    "EVALUATION",
                    "CPU",
                    "EVALUATOR",
                    dependencies=("generation-a" if request_index == 1 else "generation-b",),
                    arm_id="arm-l1",
                    request_index=request_index,
                )
            )
        scheduler = self.initialize(workers, tasks)
        initial = {dispatch.node_id: dispatch for dispatch in scheduler.dispatch_ready(now_ms=0)}
        self.assertEqual(set(initial), {"generation-a", "generation-b"})
        self.complete_dispatch(scheduler, initial["generation-a"], now_ms=1)
        next_dispatch = scheduler.dispatch_ready(now_ms=1)
        self.assertEqual([item.node_id for item in next_dispatch], ["evaluation-1"])
        evaluation = next_dispatch[0]
        expected_binding = digest_for(
            {
                "declared_input_digest": _sha("input:evaluation-1"),
                "dependency_evidence": [
                    {
                        "node_id": "generation-a",
                        "status": "COMPLETED",
                        "output_digest": _node(scheduler, "generation-a")["output_digest"],
                        "terminal_reason": "COMPLETED",
                    }
                ],
            }
        )
        self.assertEqual(evaluation.resolved_input_digest, expected_binding)
        self.assertEqual(_node(scheduler, "generation-b")["status"], "RUNNING")

    def test_owner_role_capability_and_signed_verdict_are_enforced(self) -> None:
        workers = (
            self.worker("trainer-a", "ACCELERATOR", "TRAINER", ("GENERATION",)),
            self.worker("evaluator-a", "CPU", "EVALUATOR", ("EVALUATION",)),
        )
        tasks = []
        previous = None
        for request_index in range(1, 7):
            generation_id = "generation-{}".format(request_index)
            evaluation_id = "evaluation-{}".format(request_index)
            generation_dependencies = (previous,) if previous else tuple()
            tasks.extend(
                (
                    _task(
                        generation_id,
                        "GENERATION",
                        "ACCELERATOR",
                        "TRAINER",
                        dependencies=generation_dependencies,
                        arm_id="arm-l1",
                        request_index=request_index,
                    ),
                    _task(
                        evaluation_id,
                        "EVALUATION",
                        "CPU",
                        "EVALUATOR",
                        dependencies=(generation_id,),
                        arm_id="arm-l1",
                        request_index=request_index,
                    ),
                )
            )
            previous = evaluation_id
        scheduler = self.initialize(workers, tasks)
        generation = scheduler.dispatch_ready(now_ms=0)[0]
        self.complete_dispatch(scheduler, generation, now_ms=1)
        evaluation = scheduler.dispatch_ready(now_ms=1)[0]
        forged = self.receipt(
            scheduler,
            evaluation.node_id,
            "SCHEDULER_VERDICT",
            decision="PASS",
            signer_id="trainer-a",
        )
        with self.assertRaisesRegex(CampaignSchedulerError, "bound to this exact attempt|signature or chain"):
            scheduler.complete(
                evaluation.node_id,
                worker_id=evaluation.worker_id,
                receipt=forged,
                now_ms=2,
            )
        completed, signed = self.complete_dispatch(scheduler, evaluation, now_ms=2, semantic_passed=False)
        self.assertFalse(completed["semantic_passed"])
        self.assertEqual(completed["output_digest"], signed["output_digest"])

    def test_progress_requires_advancing_signed_evidence_and_emits_control_actions(self) -> None:
        scheduler = self.initialize(
            (self.worker("trainer-a", "ACCELERATOR", "TRAINER", ("GENERATION",)),),
            (_task("generation", "GENERATION", "ACCELERATOR", "TRAINER"),),
        )
        dispatch = scheduler.dispatch_ready(now_ms=0)[0]
        receipt = self.receipt(
            scheduler,
            "generation",
            "SCHEDULER_PROGRESS",
        )
        scheduler.record_progress(
            "generation",
            worker_id=dispatch.worker_id,
            receipt=receipt,
            now_ms=1,
        )
        exact_retry = scheduler.record_progress(
            "generation",
            worker_id=dispatch.worker_id,
            receipt=receipt,
            now_ms=1,
        )
        self.assertEqual(exact_retry["last_progress_ms"], 1)

        repeated = self.receipt(
            scheduler,
            "generation",
            "SCHEDULER_PROGRESS",
            output_digest=_sha("metadata-heartbeat-churn"),
        )
        with self.assertRaisesRegex(CampaignSchedulerError, "bound to this exact attempt"):
            scheduler.record_progress(
                "generation",
                worker_id=dispatch.worker_id,
                receipt=repeated,
                now_ms=2,
            )
        tampered = deepcopy(repeated)
        tampered["output_digest"] = _sha("tampered-after-signing")
        with self.assertRaisesRegex(CampaignSchedulerError, "bound to this exact attempt|signature or chain"):
            scheduler.record_progress(
                "generation",
                worker_id=dispatch.worker_id,
                receipt=tampered,
                now_ms=2,
            )
        self.assertEqual(scheduler.controller_actions(now_ms=PROGRESS_CHECKPOINT_MS), tuple())
        due = scheduler.controller_actions(now_ms=PROGRESS_CHECKPOINT_MS + 1)
        self.assertEqual(due[0].action, "EMIT_PROGRESS_CHECKPOINT")
        self.assertEqual(due[0].due_ms, PROGRESS_CHECKPOINT_MS + 1)
        stalled = scheduler.controller_actions(now_ms=STALL_TIMEOUT_MS + 1)
        self.assertEqual(stalled[0].action, "CANCEL_STALLED_ATTEMPT")

    def test_stall_cancellation_retry_receipts_survive_restart_and_fail_closed(self) -> None:
        workers = (
            self.worker("trainer-a", "ACCELERATOR", "TRAINER", ("GENERATION",)),
            self.worker("trainer-b", "ACCELERATOR", "TRAINER", ("GENERATION",)),
            self.worker("controller-a", "CPU", "CONTROLLER", ("CANCEL",)),
        )
        scheduler = self.initialize(
            workers,
            (_task("generation", "GENERATION", "ACCELERATOR", "TRAINER"),),
        )
        first = scheduler.dispatch_ready(now_ms=0)[0]
        with self.assertRaisesRegex(CampaignSchedulerError, "exact bounded cancellation"):
            scheduler.recover_stalls(now_ms=STALL_TIMEOUT_MS, cancellation_receipts={})

        forged = self.receipt(
            scheduler,
            "generation",
            "SCHEDULER_CANCELLATION",
            signer_id=first.worker_id,
        )
        with self.assertRaisesRegex(CampaignSchedulerError, "controller capability"):
            scheduler.recover_stalls(
                now_ms=STALL_TIMEOUT_MS,
                cancellation_receipts={"generation": forged},
            )
        cancellation = self.receipt(
            scheduler,
            "generation",
            "SCHEDULER_CANCELLATION",
            signer_id="controller-a",
        )
        recovery = scheduler.recover_stalls(
            now_ms=STALL_TIMEOUT_MS,
            cancellation_receipts={"generation": cancellation},
        )
        self.assertEqual(recovery[0].disposition, "RETRY_PENDING")
        scheduler = CampaignScheduler(self.state_path)
        retried_node = _node(scheduler, "generation")
        self.assertEqual(retried_node["attempt_receipts"][0]["disposition"], "STALL_RETRY")
        self.assertEqual(
            retried_node["attempt_receipts"][0]["receipts"][-1]["effect_kind"],
            "SCHEDULER_CANCELLATION",
        )
        second = scheduler.dispatch_ready(now_ms=STALL_TIMEOUT_MS)[0]
        self.assertNotEqual(second.worker_id, first.worker_id)
        second_cancel = self.receipt(
            scheduler,
            "generation",
            "SCHEDULER_CANCELLATION",
            signer_id="controller-a",
        )
        terminal = scheduler.recover_stalls(
            now_ms=2 * STALL_TIMEOUT_MS,
            cancellation_receipts={"generation": second_cancel},
        )
        self.assertEqual(terminal[0].disposition, "FAILED_CLOSED")
        self.assertEqual(_node(scheduler, "generation")["terminal_reason"], "STALL_RETRY_EXHAUSTED")
        self.assertEqual(len(_node(scheduler, "generation")["attempt_receipts"]), 2)

    def test_least_options_maximum_matching_respects_retry_exclusion(self) -> None:
        workers = (
            self.worker("a-flex", "ACCELERATOR", "TRAINER", ("GENERATION",)),
            self.worker("z-flex", "ACCELERATOR", "TRAINER", ("GENERATION",)),
            self.worker("reviewer-a", "CPU", "REVIEWER", ("REVIEW",)),
            self.worker("controller-a", "CPU", "CONTROLLER", ("CANCEL",)),
        )
        scheduler = self.initialize(
            workers,
            (
                _task("retry-node", "GENERATION", "ACCELERATOR", "TRAINER"),
                _task("gate", "REVIEW", "CPU", "REVIEWER"),
                _task(
                    "general-node",
                    "GENERATION",
                    "ACCELERATOR",
                    "TRAINER",
                    dependencies=("gate",),
                ),
            ),
        )
        initial = {item.node_id: item for item in scheduler.dispatch_ready(now_ms=0)}
        self.assertEqual(initial["retry-node"].worker_id, "a-flex")
        self.complete_dispatch(scheduler, initial["gate"], now_ms=STALL_TIMEOUT_MS)
        cancellation = self.receipt(
            scheduler,
            "retry-node",
            "SCHEDULER_CANCELLATION",
            signer_id="controller-a",
        )
        scheduler.recover_stalls(
            now_ms=STALL_TIMEOUT_MS,
            cancellation_receipts={"retry-node": cancellation},
        )
        matched = {item.node_id: item.worker_id for item in scheduler.dispatch_ready(now_ms=STALL_TIMEOUT_MS)}
        self.assertEqual(matched, {"general-node": "a-flex", "retry-node": "z-flex"})

    def test_process_local_mutex_prevents_duplicate_dispatch(self) -> None:
        scheduler = self.initialize(
            (self.worker("reviewer-a", "CPU", "REVIEWER", ("REVIEW",)),),
            (_task("review", "REVIEW", "CPU", "REVIEWER"),),
        )
        resumed = CampaignScheduler(self.state_path)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda item: item.dispatch_ready(now_ms=0), (scheduler, resumed)))
        self.assertEqual(sum(len(result) for result in results), 1)
        self.assertEqual(_node(scheduler, "review")["attempt_count"], 1)

    def checkpoint_plan_parts(self, *, promotion_key=None):
        workers = (
            self.worker("trainer-a", "ACCELERATOR", "TRAINER", ("GENERATION", "TRAINING")),
            self.worker("evaluator-a", "CPU", "EVALUATOR", ("EVALUATION", "VALIDATION")),
            self.worker(
                "controller-a",
                "CPU",
                "CONTROLLER",
                ("DIAGNOSTIC", "PROMOTION", "CANCEL"),
            ),
        )
        tasks = []
        evaluation_ids = []
        previous_evaluation = None
        for request_index in range(1, PILOT_CHECKPOINT_REQUEST + 1):
            generation_id = "generation-{}".format(request_index)
            evaluation_id = "evaluation-{}".format(request_index)
            evaluation_ids.append(evaluation_id)
            tasks.append(
                _task(
                    generation_id,
                    "GENERATION",
                    "ACCELERATOR",
                    "TRAINER",
                    dependencies=(previous_evaluation,) if previous_evaluation else tuple(),
                    arm_id="arm-l1",
                    request_index=request_index,
                )
            )
            tasks.append(
                _task(
                    evaluation_id,
                    "EVALUATION",
                    "CPU",
                    "EVALUATOR",
                    dependencies=(generation_id,),
                    arm_id="arm-l1",
                    request_index=request_index,
                )
            )
            previous_evaluation = evaluation_id
        checkpoint_dependencies = tuple(evaluation_ids)
        tasks.extend(
            (
                _task(
                    "generation-7",
                    "GENERATION",
                    "ACCELERATOR",
                    "TRAINER",
                    stage="POST_SCREEN",
                    dependencies=checkpoint_dependencies,
                    arm_id="arm-l1",
                    request_index=PILOT_CHECKPOINT_REQUEST + 1,
                ),
                _task(
                    "post-training",
                    "TRAINING",
                    "ACCELERATOR",
                    "TRAINER",
                    stage="POST_SCREEN",
                    dependencies=checkpoint_dependencies,
                    arm_id="arm-l1",
                ),
                _task(
                    "post-validation",
                    "VALIDATION",
                    "CPU",
                    "EVALUATOR",
                    stage="POST_SCREEN",
                    dependencies=("post-training",),
                    arm_id="arm-l1",
                ),
                _task(
                    "post-promotion",
                    "PROMOTION",
                    "CPU",
                    "CONTROLLER",
                    stage="POST_SCREEN",
                    dependencies=("post-validation",),
                    arm_id="arm-l1",
                    promotion_key=promotion_key or _sha("validated-candidate"),
                    promotion_authority_node_id="post-validation",
                ),
                _task(
                    "frozen-diagnostic",
                    "DIAGNOSTIC",
                    "CPU",
                    "CONTROLLER",
                    stage="DIAGNOSTIC",
                    dependencies=checkpoint_dependencies,
                    arm_id="arm-l1",
                ),
            )
        )
        return workers, tasks

    def checkpoint_scheduler(self, *, promotion_key=None) -> CampaignScheduler:
        workers, tasks = self.checkpoint_plan_parts(promotion_key=promotion_key)
        return self.initialize(workers, tasks)

    def run_checkpoint(self, scheduler, *, passing_requests=tuple(), failed_request=None):
        now_ms = 0
        for _ in range(100):
            arm = _arm(scheduler, "arm-l1")
            if arm["status"] != "SCREENING":
                return now_ms
            dispatches = scheduler.dispatch_ready(now_ms=now_ms)
            self.assertTrue(dispatches, "checkpoint fixture deadlocked")
            for dispatch in dispatches:
                node = _node(scheduler, dispatch.node_id)
                now_ms += 1
                if node["kind"] == "EVALUATION" and node["request_index"] == failed_request:
                    self.fail_dispatch(scheduler, dispatch, now_ms=now_ms)
                else:
                    self.complete_dispatch(
                        scheduler,
                        dispatch,
                        now_ms=now_ms,
                        semantic_passed=(
                            node["request_index"] in passing_requests if node["kind"] == "EVALUATION" else None
                        ),
                    )
        self.fail("checkpoint fixture exceeded deterministic bound")

    def test_post_screen_plan_bypass_is_rejected_and_screening_never_dispatches_it(self) -> None:
        with self.assertRaisesRegex(CampaignSchedulerError, "frozen request index"):
            _task(
                "omitted-request",
                "GENERATION",
                "ACCELERATOR",
                "TRAINER",
                arm_id="arm-l1",
            )
        with self.assertRaisesRegex(CampaignSchedulerError, "post-screen tasks"):
            _task("bad-training", "TRAINING", "ACCELERATOR", "TRAINER")
        with self.assertRaisesRegex(CampaignSchedulerError, "stage bypasses"):
            _task(
                "bad-request-7",
                "GENERATION",
                "ACCELERATOR",
                "TRAINER",
                arm_id="arm-l1",
                request_index=7,
            )
        workers, tasks = self.checkpoint_plan_parts()
        checkpoint_only = [task for task in tasks if task.stage != "POST_SCREEN"]
        checkpoint_only.append(
            _task(
                "rogue-training",
                "TRAINING",
                "ACCELERATOR",
                "TRAINER",
                stage="POST_SCREEN",
                arm_id="arm-l1",
            )
        )
        with self.assertRaisesRegex(CampaignSchedulerError, "bypasses the completed checkpoint"):
            SchedulerPlan("bypass", workers, tuple(checkpoint_only))

        scheduler = self.checkpoint_scheduler()
        dispatched = {item.node_id for item in scheduler.dispatch_ready(now_ms=0)}
        self.assertEqual(dispatched, {"generation-1"})
        for node_id in ("generation-7", "post-training", "post-validation", "post-promotion"):
            self.assertEqual(_node(scheduler, node_id)["status"], "PENDING")

    def test_transition_journal_rejects_valid_stale_state_rollback(self) -> None:
        scheduler = self.initialize(
            (self.worker("reviewer-a", "CPU", "REVIEWER", ("REVIEW",)),),
            (_task("review", "REVIEW", "CPU", "REVIEWER"),),
        )
        stale_valid_state = self.state_path.read_bytes()
        scheduler.dispatch_ready(now_ms=0)
        self.state_path.write_bytes(stale_valid_state)
        with self.assertRaisesRegex(CampaignSchedulerError, "transition journal head"):
            CampaignScheduler(self.state_path)

    def test_matching_is_lexicographically_minimal_under_sealed_priority(self) -> None:
        scheduler = self.initialize(
            (
                self.worker("a-reviewer", "CPU", "REVIEWER", ("REVIEW",)),
                self.worker("b-reviewer", "CPU", "REVIEWER", ("REVIEW",)),
            ),
            (
                _task("a-task", "REVIEW", "CPU", "REVIEWER"),
                _task("b-task", "REVIEW", "CPU", "REVIEWER"),
            ),
        )
        matched = {dispatch.node_id: dispatch.worker_id for dispatch in scheduler.dispatch_ready(now_ms=0)}
        self.assertEqual(matched, {"a-task": "a-reviewer", "b-task": "b-reviewer"})

    def test_retry_preserves_budget_lineage_and_all_frozen_control_bindings(self) -> None:
        scheduler = self.initialize(
            (
                self.worker("trainer-a", "ACCELERATOR", "TRAINER", ("GENERATION",)),
                self.worker("trainer-b", "ACCELERATOR", "TRAINER", ("GENERATION",)),
                self.worker("controller-a", "CPU", "CONTROLLER", ("CANCEL",)),
            ),
            (_task("generation", "GENERATION", "ACCELERATOR", "TRAINER"),),
        )
        scheduler.dispatch_ready(now_ms=0)
        cancellation = self.receipt(scheduler, "generation", "SCHEDULER_CANCELLATION", signer_id="controller-a")
        scheduler.recover_stalls(
            now_ms=STALL_TIMEOUT_MS,
            cancellation_receipts={"generation": cancellation},
        )
        scheduler.dispatch_ready(now_ms=STALL_TIMEOUT_MS)
        node = _node(scheduler, "generation")
        first, second = node["attempt_receipts"]
        self.assertEqual((first["budget_used"], second["budget_used"], second["budget_limit"]), (1, 2, 2))
        self.assertEqual(second["predecessor_attempt_id"], first["attempt_id"])
        self.assertEqual(second["predecessor_receipt_hash"], receipt_hash(first["receipts"][-1]))
        for field in (
            "checkpoint_digest",
            "isolation_digest",
            "lease_id",
            "release_digest",
            "capability_digest",
            "model_digest",
            "protocol_digest",
            "evaluator_digest",
            "sandbox_digest",
            "manifest_digest",
            "monotonic_counter",
        ):
            self.assertIsNotNone(second[field])
        forged = self.receipt(scheduler, "generation", "SCHEDULER_COMPLETION")
        forged["policy_digest"] = _sha("forged-budget-policy")
        with self.assertRaisesRegex(CampaignSchedulerError, "bound to this exact attempt"):
            scheduler.complete("generation", worker_id=second["worker_id"], receipt=forged, now_ms=STALL_TIMEOUT_MS + 1)

    def test_zero_signal_cancels_every_post_screen_task_and_allows_diagnostic(self) -> None:
        scheduler = self.checkpoint_scheduler()
        now_ms = self.run_checkpoint(scheduler)
        arm = _arm(scheduler, "arm-l1")
        self.assertEqual(arm["status"], "PAUSED_ZERO_SIGNAL")
        self.assertTrue(arm["checkpoint_complete"])
        self.assertEqual(arm["checkpoint_completed_requests"], list(range(1, 7)))
        for node_id in ("generation-7", "post-training", "post-validation", "post-promotion"):
            node = _node(scheduler, node_id)
            self.assertEqual(node["status"], "CANCELLED")
            self.assertEqual(node["terminal_reason"], "ZERO_SEMANTIC_SIGNAL_AT_FROZEN_CHECKPOINT")
        diagnostic = scheduler.dispatch_ready(now_ms=now_ms)
        self.assertEqual([item.node_id for item in diagnostic], ["frozen-diagnostic"])

    def test_failed_checkpoint_is_incomplete_fail_closed_not_semantic_zero_or_continue(self) -> None:
        scheduler = self.checkpoint_scheduler()
        now_ms = self.run_checkpoint(scheduler, passing_requests=(1,), failed_request=3)
        arm = _arm(scheduler, "arm-l1")
        self.assertEqual(arm["status"], "INCOMPLETE_FAIL_CLOSED")
        self.assertFalse(arm["checkpoint_complete"])
        self.assertEqual(arm["checkpoint_completed_requests"], [1, 2])
        self.assertEqual(arm["checkpoint_semantic_passes"], 1)
        for node_id in ("generation-7", "post-training", "post-validation", "post-promotion"):
            node = _node(scheduler, node_id)
            self.assertEqual(node["status"], "CANCELLED")
            self.assertEqual(node["terminal_reason"], "CHECKPOINT_INCOMPLETE_FAIL_CLOSED")
        diagnostic = scheduler.dispatch_ready(now_ms=now_ms)
        self.assertEqual([item.node_id for item in diagnostic], ["frozen-diagnostic"])

    def test_nonzero_completed_checkpoint_unlocks_post_screen_work(self) -> None:
        scheduler = self.checkpoint_scheduler()
        now_ms = self.run_checkpoint(scheduler, passing_requests=(2,))
        arm = _arm(scheduler, "arm-l1")
        self.assertEqual(arm["status"], "CONTINUING")
        self.assertTrue(arm["checkpoint_complete"])
        dispatches = scheduler.dispatch_ready(now_ms=now_ms)
        self.assertIn("frozen-diagnostic", {item.node_id for item in dispatches})
        self.assertTrue({"generation-7", "post-training"} & {item.node_id for item in dispatches})

    def drive_to_promotion(self, scheduler, *, validation_output):
        now_ms = self.run_checkpoint(scheduler, passing_requests=(1,))
        promotion_dispatch = None
        promotion_receipt = None
        for _ in range(30):
            dispatches = scheduler.dispatch_ready(now_ms=now_ms)
            if not dispatches:
                break
            for dispatch in dispatches:
                node = _node(scheduler, dispatch.node_id)
                now_ms += 1
                output = validation_output if dispatch.node_id == "post-validation" else None
                _, receipt = self.complete_dispatch(
                    scheduler,
                    dispatch,
                    now_ms=now_ms,
                    output_digest=output,
                    semantic_passed=(
                        True if node["kind"] == "VALIDATION" else False if node["kind"] == "EVALUATION" else None
                    ),
                )
                if dispatch.node_id == "post-promotion":
                    promotion_dispatch = dispatch
                    promotion_receipt = receipt
                    return now_ms, promotion_dispatch, promotion_receipt
        return now_ms, promotion_dispatch, promotion_receipt

    def test_promotion_requires_ancestor_candidate_evidence_and_restart_is_idempotent(self) -> None:
        candidate = _sha("validated-candidate")
        scheduler = self.checkpoint_scheduler(promotion_key=candidate)
        now_ms, promotion_dispatch, promotion_receipt = self.drive_to_promotion(
            scheduler,
            validation_output=candidate,
        )
        self.assertIsNotNone(promotion_dispatch)
        self.assertEqual(_node(scheduler, "post-promotion")["status"], "COMPLETED")
        sequence = scheduler.snapshot()["sequence"]
        resumed = CampaignScheduler(self.state_path)
        exact_retry = resumed.complete(
            "post-promotion",
            worker_id=promotion_dispatch.worker_id,
            receipt=promotion_receipt,
            now_ms=now_ms,
        )
        self.assertEqual(exact_retry["status"], "COMPLETED")
        self.assertEqual(resumed.snapshot()["sequence"], sequence)
        conflicting = deepcopy(promotion_receipt)
        conflicting["output_digest"] = _sha("substituted-promotion-output")
        with self.assertRaisesRegex(CampaignSchedulerError, "conflicts"):
            resumed.complete(
                "post-promotion",
                worker_id=promotion_dispatch.worker_id,
                receipt=conflicting,
                now_ms=now_ms,
            )

    def test_stale_promotion_key_never_dispatches(self) -> None:
        scheduler = self.checkpoint_scheduler(promotion_key=_sha("unobserved-candidate"))
        now_ms, promotion_dispatch, _ = self.drive_to_promotion(
            scheduler,
            validation_output=_sha("different-observed-candidate"),
        )
        self.assertIsNone(promotion_dispatch)
        self.assertEqual(_node(scheduler, "post-promotion")["status"], "PENDING")
        self.assertEqual(scheduler.dispatch_ready(now_ms=now_ms), tuple())

    def test_matching_candidate_without_positive_validation_authority_never_promotes(self) -> None:
        candidate = _sha("validated-candidate")
        scheduler = self.checkpoint_scheduler(promotion_key=candidate)
        now_ms = self.run_checkpoint(scheduler, passing_requests=(1,))
        for _ in range(30):
            dispatches = scheduler.dispatch_ready(now_ms=now_ms)
            if not dispatches:
                break
            for dispatch in dispatches:
                node = _node(scheduler, dispatch.node_id)
                now_ms += 1
                self.complete_dispatch(
                    scheduler,
                    dispatch,
                    now_ms=now_ms,
                    output_digest=candidate if dispatch.node_id == "post-validation" else None,
                    semantic_passed=(False if node["kind"] in {"EVALUATION", "VALIDATION"} else None),
                )
        validation = _node(scheduler, "post-validation")
        self.assertFalse(validation["semantic_passed"])
        self.assertIsNone(validation["positive_promotion_authority"])
        self.assertEqual(_node(scheduler, "post-promotion")["status"], "PENDING")

    def test_utilization_identifies_avoidable_and_legitimately_blocked_idle(self) -> None:
        scheduler = self.initialize(
            (
                self.worker("trainer-a", "ACCELERATOR", "TRAINER", ("GENERATION",)),
                self.worker("trainer-b", "ACCELERATOR", "TRAINER", ("GENERATION",)),
            ),
            (_task("generation", "GENERATION", "ACCELERATOR", "TRAINER"),),
        )
        aliases = {"trainer-a": "slot-a", "trainer-b": "slot-b"}
        report = scheduler.public_report(
            now_ms=AVOIDABLE_IDLE_DEFECT_MS,
            worker_aliases=aliases,
        )
        self.assertEqual(report["aggregate"]["idle_ms"], 2 * AVOIDABLE_IDLE_DEFECT_MS)
        self.assertEqual(report["aggregate"]["avoidable_idle_ms"], AVOIDABLE_IDLE_DEFECT_MS)
        self.assertEqual(report["aggregate"]["legitimately_blocked_ms"], AVOIDABLE_IDLE_DEFECT_MS)
        self.assertFalse(report["aggregate"]["avoidable_idle_defect"])
        report = scheduler.public_report(
            now_ms=AVOIDABLE_IDLE_DEFECT_MS + 1,
            worker_aliases=aliases,
        )
        self.assertTrue(report["aggregate"]["avoidable_idle_defect"])

    def test_public_projection_uses_rfc5737_fixture_and_rejects_leakage(self) -> None:
        scheduler = self.initialize(
            (self.worker("reviewer-a", "CPU", "REVIEWER", ("REVIEW",)),),
            (_task("review", "REVIEW", "CPU", "REVIEWER"),),
        )
        for unsafe_alias in (
            "192.0.2.24",
            "X:\\fixture\\private",
            "private-slot",
            "secret-value",
            "worker-a",
            "unstructuredalias",
        ):
            with self.subTest(alias=unsafe_alias):
                with self.assertRaises(CampaignSchedulerError):
                    scheduler.public_report(now_ms=0, worker_aliases={"reviewer-a": unsafe_alias})
        report = scheduler.public_report(now_ms=1, worker_aliases={"reviewer-a": "slot-a"})
        self.assertEqual(validate_public_scheduler_report(report), report)
        self.assertEqual(report["state_authority"], STATE_AUTHORITY)
        self.assertEqual(report["claim_boundary"], CLAIM_BOUNDARY)
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn("reviewer-a", serialized)
        self.assertNotIn("REVIEWER", serialized)
        self.assertNotIn(self.signers["reviewer-a"].key_id, serialized)
        leaked = deepcopy(report)
        leaked["private_path"] = "X:\\fixture\\private"
        with self.assertRaisesRegex(CampaignSchedulerError, "closed schema"):
            validate_public_scheduler_report(leaked)

    def test_trusted_state_is_explicitly_non_authoritative_and_tamper_fails_closed(self) -> None:
        scheduler = self.initialize(
            (self.worker("reviewer-a", "CPU", "REVIEWER", ("REVIEW",)),),
            (_task("review", "REVIEW", "CPU", "REVIEWER"),),
        )
        self.assertEqual(scheduler.snapshot()["state_authority"], STATE_AUTHORITY)
        value = scheduler.snapshot()
        value["state_authority"] = "AUTHORITATIVE"
        self.state_path.write_bytes(canonical_bytes(value))
        with self.assertRaisesRegex(CampaignSchedulerError, "authority"):
            CampaignScheduler(self.state_path)

    def test_persisted_signed_receipt_tamper_and_accounting_drift_fail_closed(self) -> None:
        scheduler = self.initialize(
            (self.worker("reviewer-a", "CPU", "REVIEWER", ("REVIEW",)),),
            (_task("review", "REVIEW", "CPU", "REVIEWER"),),
        )
        dispatch = scheduler.dispatch_ready(now_ms=0)[0]
        receipt = self.receipt(scheduler, "review", "SCHEDULER_PROGRESS")
        scheduler.record_progress(
            "review",
            worker_id=dispatch.worker_id,
            receipt=receipt,
            now_ms=1,
        )
        value = scheduler.snapshot()
        value["nodes"][0]["attempt_receipts"][0]["receipts"][0]["output_digest"] = _sha("tamper")
        self.state_path.write_bytes(canonical_bytes(value))
        with self.assertRaisesRegex(
            CampaignSchedulerError, "bound to this exact attempt|signature or chain|transition journal head"
        ):
            CampaignScheduler(self.state_path)

        accounting_path = self.root / "accounting-state.json"
        scheduler = CampaignScheduler.initialize(
            accounting_path,
            SchedulerPlan(
                "accounting-test",
                (self.worker("reviewer-b", "CPU", "REVIEWER", ("REVIEW",)),),
                (_task("review-b", "REVIEW", "CPU", "REVIEWER"),),
            ),
            now_ms=0,
        )
        scheduler.public_report(now_ms=10, worker_aliases={"reviewer-b": "slot-b"})
        value = scheduler.snapshot()
        value["workers"][0]["idle_ms"] += 1
        accounting_path.write_bytes(canonical_bytes(value))
        with self.assertRaisesRegex(CampaignSchedulerError, "does not reconcile"):
            CampaignScheduler(accounting_path)


if __name__ == "__main__":
    unittest.main()
