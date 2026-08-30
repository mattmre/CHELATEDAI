# EGV work-conserving scheduler implementation boundary

Status: implemented and locally acceptance-tested control-plane slice for
[issue #307](https://github.com/mattmre/CHELATEDAI/issues/307). It has not run a
model, a live multi-worker campaign, or cross-host trainer/evaluator transport.

## What is implemented

`egv.campaign.scheduler.CampaignScheduler` is a deterministic, file-backed DAG
state machine. Callers give it logical workers and tasks, execute the returned
dispatches outside the scheduler, and return closed Ed25519-signed activity
receipts. Initialization and restart require an explicit `effect_fence_path`
that is distinct from the scheduler state and transition-journal paths. Its
durable transitions provide:

- dependency-aware concurrent dispatch across `ACCELERATOR` and `CPU` pools,
  using the lexicographically minimal maximum bipartite matching under each
  task's sealed priority tuple (including preregistered manifest ordinal and
  counterbalance rank), owner-role/capability checks, and retry-worker
  exclusions;
- per-lane binding of a closed owner role, worker public key, declared input,
  dependency-output-derived input, start/progress times, resource assignment,
  output digest, and terminal reason;
- immediate readiness of a dependent evaluation when its own generation
  finishes, without a global fan-in barrier;
- an explicit 300,000 ms progress-checkpoint controller action and exact
  600,000 ms no-new-evidence cancellation action; progress advances time only
  when a cryptographically distinct `OBSERVE` authority signs a new
  content-addressed artifact and contiguous external counter;
- a signed bounded-cancellation receipt plus a separately signed external
  termination observation are required before one retry on a different logical
  worker or a fail-closed terminal result. The observation binds execution
  termination, lease revocation, a fencing token, runtime checkpoint evidence,
  resource teardown evidence, and an exit-status class. The cancellation and
  observation keys must be distinct from each other and from the attempt worker;
- an append-only per-attempt receipt journal, signature/chain replay on every
  load, independently observed node-kind progress commitments that reject
  scheduler-derived heartbeat churn, and idempotent exact completion after
  process restart;
- signed receipts that bind the frozen checkpoint, retry budget/counter,
  predecessor attempt and receipt, lease/release, logical isolation,
  capability, model/protocol/evaluator/sandbox/manifest identities, and
  monotonic attempt counter; retry preserves that lineage and excludes the
  cancelled worker;
- signed evaluator/validation verdicts as the only semantic-pass authority and
  a closed positive promotion record bound to the intended direct authority
  node, exact candidate digest, `PASS`, attempt, and receipt-chain head;
- the preregistered request-6 checkpoint: all six evaluator nodes must be
  `COMPLETED` before zero-pass or continue is possible; any failed/cancelled
  evaluator produces `INCOMPLETE_FAIL_CLOSED`; and every transitive post-screen
  task is gated and cancelled on zero or incomplete signal while the frozen
  diagnostic remains runnable;
- per-worker busy, legitimately blocked, avoidable-idle, current-streak, and
  maximum-streak accounting, with a defect only when an avoidable interval is
  strictly over 300,000 ms; and
- a closed aggregate public projection that requires pseudonymous aliases and
  rejects extra fields, path-shaped values, secret vocabulary, topology labels,
  duplicate aliases, and inconsistent accounting.

The state file is canonical JSON, replaced atomically under both a process-local
mutex and cross-process lock. Every committed state digest is also chained into
an append-only local transition journal; a valid older state file cannot be
reloaded against the later journal head. Dispatch additionally claims its exact
attempt binding in a caller-supplied external effect-fence journal before work
is returned. A random fence-instance identity is pinned into scheduler state,
so substituting another empty fence for the same campaign/plan fails closed.
Coherently restoring only the scheduler state and transition journal therefore
cannot redispatch that effect. The static task/worker inventory is bound by
`plan_digest`; every load revalidates the closed schema, DAG, retry ceiling,
assignments, dependency/output bindings, receipt signatures and chains,
checkpoint projection, utilization reconciliation, and terminal evidence
before another transition can run. This boundary uses scheduler-state schema
v4; prior v3 scheduler state is rejected rather than silently migrated.

## Frozen scope

This slice schedules logical work only. It does not launch processes, call a
model endpoint, move bytes between machines, inspect hidden evaluator material,
decide scientific promotion, or prove accelerator utilization. Activity
adapters must enforce those runtime boundaries separately and return signed,
content-addressed evidence to this scheduler. The scheduler verifies observer
signatures, counter continuity, and content addresses; it does not open the
referenced artifact bytes and therefore does not prove scientific correctness.

This library also does not own a wall-clock polling loop. A live controller
must call the reporting/progress and stall-recovery surfaces on its scheduled
five-minute/ten-minute cadence; the deterministic fixtures advance an explicit
logical clock to prove the transition boundaries, not asynchronous polling.

The state file is explicitly labeled
`TRUSTED_LOCAL_CONTROL_PLANE_NON_AUTHORITATIVE`. The append-only journal is
only a consistency check inside this trusted local control-plane boundary. It
has no external transparency anchor or detached state signature and therefore
must not be used as standalone scientific, publication, or
distributed-execution evidence. The caller must keep the effect-fence journal
in a rollback domain independent from the scheduler pair; rolling back all
three files defeats this local at-most-once protection. Signed activity and
observation receipts authenticate their bounded assertions; neither they nor
the journals prove external execution or model quality.

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
