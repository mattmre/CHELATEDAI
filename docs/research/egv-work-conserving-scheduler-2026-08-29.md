# EGV work-conserving scheduler implementation boundary

Status: implemented and locally acceptance-tested control-plane slice for
[issue #307](https://github.com/mattmre/CHELATEDAI/issues/307). It has not run a
model, a live multi-worker campaign, or cross-host trainer/evaluator transport.

## What is implemented

`egv.campaign.scheduler.CampaignScheduler` is a deterministic, file-backed DAG
state machine. Callers give it logical workers and tasks, then execute the
returned dispatches outside the scheduler and return only progress/output
digests. Its durable transitions provide:

- dependency-aware concurrent dispatch across `ACCELERATOR` and `CPU` pools;
- per-lane binding of a closed owner role and content-addressed input digest,
  followed by persisted start/progress times, resource assignment, output
  digest, and terminal reason;
- immediate readiness of a dependent evaluation when its own generation
  finishes, without a global fan-in barrier;
- an exact 600,000 ms no-new-evidence stall boundary, one retry on a different
  logical worker, and a fail-closed terminal result when reassignment is
  unavailable or the retry stalls;
- idempotent exact completion after process restart and unique promotion
  authority per content digest;
- the preregistered request-6 checkpoint: post-checkpoint work stays gated,
  zero semantic passes pause the arm and cancel later requests, while the
  frozen diagnostic remains runnable;
- per-worker busy, legitimately blocked, avoidable-idle, current-streak, and
  maximum-streak accounting, with a defect only when an avoidable interval is
  strictly over 300,000 ms; and
- a closed aggregate public projection that requires pseudonymous aliases and
  rejects extra fields, path-shaped values, secret vocabulary, topology labels,
  duplicate aliases, and inconsistent accounting.

The state file is canonical JSON, replaced atomically under a cross-process
lock. The static task/worker inventory is bound by `plan_digest`; every load
revalidates the closed schema, DAG, retry ceiling, assignments, dependencies,
checkpoint projection, and terminal evidence before another transition can
run.

## Frozen scope

This slice schedules logical work only. It does not launch processes, call a
model endpoint, move bytes between machines, inspect hidden evaluator material,
decide scientific promotion, or prove accelerator utilization. Activity
adapters must enforce those runtime boundaries separately and return
content-addressed evidence to this scheduler.

This library also does not own a wall-clock polling loop. A live controller
must call the reporting/progress and stall-recovery surfaces on its scheduled
five-minute/ten-minute cadence; the deterministic fixtures advance an explicit
logical clock to prove the transition boundaries, not asynchronous polling.

The public report carries the literal claim boundary
`ENGINEERING_ONLY_NO_MODEL_QUALITY_OR_LIVE_DISTRIBUTED_CLAIM`. Internal worker
IDs, task IDs, arm IDs, filesystem paths, endpoints, credentials, prompts,
candidate source, host labels, and deployment topology are absent from that
projection.

## Local acceptance command

```text
python -m unittest -v tests.test_egv_campaign_scheduler
```

The fixtures use the production file-backed scheduler, explicit logical-clock
jumps, real restart/reload, and canonical state bytes. They use no model,
network, or distributed-runtime mock and make no model-quality claim.
