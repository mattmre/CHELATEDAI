# ADR-0001: Evidence-Governed Variation Agent

- **Status:** Proposed; documentation-only decision
- **Date:** 2026-08-21
- **Decision owners:** ChelatedAI research maintainers
- **Operational companion:** [Dual-Spark execution runbook](../runbooks/evidence-governed-variation-dual-spark.md)
- **Applies to:** A bounded research campaign, not the existing production retrieval path

## Context

Agentic Variation Operators (AVO) replace fixed evolutionary mutation operators
with a coding agent that inspects lineage, proposes a change, executes an
evaluator, and iterates from feedback. NVIDIA's agent-security architecture
places an authoritative runtime below the model and harness so the agent may
propose actions without being able to grant itself authority.

ChelatedAI can test a complementary question: does correction-aware,
provenance-aware evidence improve a small variation agent over ordinary
success/failure memory without giving the agent control of its evaluator or
permissions?

The campaign is intentionally small enough to run on two DGX Spark systems over
a few days. It is not an attempt to train a foundation model from scratch.

Relevant external work:

- [AVO preprint](https://arxiv.org/abs/2603.24517)
- [NVIDIA AVO ARC-AGI-3 report](https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents/)
- [NVIDIA agent security discussion](https://forums.developer.nvidia.com/t/building-agent-systems-for-both-long-horizon-capability-and-enforceable-security/380902)
- [NVIDIA agent-stack security boundary](https://developer.nvidia.com/blog/where-security-fits-in-an-ai-agent-stack/)

## Decision

Build a research-only **Evidence-Governed Variation (EGV) Agent** around a
revision-pinned `Qwen/Qwen3.5-2B-Base` model. The first trainable artifact is a
LoRA adapter; the base model remains immutable. An append-only SQLite evidence
ledger is authoritative. Qdrant is a disposable retrieval projection that can
be deleted and rebuilt from the ledger. A frozen evaluator on a second Spark
returns signed correctness, performance, and authority receipts.

The source model is pinned to Hugging Face revision:

```text
Qwen/Qwen3.5-2B-Base@b1485b2fa6dfa1287294f269f5fb618e03d52d7c
```

That revision was resolved from the repository's `main` ref on 2026-08-21. The
campaign preflight must independently resolve the revision, download it, record
file hashes and license metadata, and refuse to run if any required file does
not match the frozen manifest. The human-readable model name alone is not a
pin.

## Scope

The initial campaign includes:

- Synthetic, bounded micro-repositories with public or generated content only.
- Candidate generation and repair using the pinned 2B base model.
- Supervised fine-tuning through a LoRA adapter only.
- Success evidence, compact failure evidence, dependency edges, corrections,
  retractions, and runtime-effect receipts.
- A deliberate evaluator correction, called the **correction shock**.
- Matched, preregistered ablations with a frozen test split.
- Deny-by-default runtime authority enforced below the agent harness.
- Portable artifacts sufficient to replay decisions without Qdrant.

The campaign excludes:

- Updating the base model's weights.
- Reinforcement learning, distributed foundation-model training, and
  production deployment.
- Tuning on ARC-AGI-3 public environments.
- Mutating ChelatedAI's current production DAG or retrieval collections.
- Treating prompt instructions as a security boundary.
- Copying private repositories, credentials, chat transcripts, user data, or
  unrestricted host files into training data.

## System structure

```mermaid
flowchart LR
    O[Codex campaign orchestrator] --> G[Generator and LoRA trainer]
    R[Grok adversarial protocol review] --> F[Frozen campaign manifest]
    F --> O
    G --> Q[Evidence query service]
    Q --> P[(Qdrant projection)]
    Q --> L[(Authoritative SQLite ledger)]
    G --> B[Authority broker]
    B --> A[Content-addressed candidate artifact]
    A --> E[Evaluator controller on second Spark]
    E --> S[Candidate sandbox with opaque input only]
    E --> H[Hidden evaluator runner in separate identity]
    S --> E
    H --> E
    E --> X[Signed verdict and effect receipts]
    X --> L
    L --> P
    L --> C[Checkpoint and portable evidence bundle]
```

The model and both interactive agents are above the authority boundary. They
may request actions, generate candidates, and recommend interpretations. They
cannot edit hidden tests, evaluator binaries, policy, signing keys, or committed
ledger events.

The candidate sandbox and evaluator are separate security principals. The
sandbox receives a content-addressed candidate artifact plus one opaque test
input at a time through the evaluator controller. It has **zero read, write, or
directory-listing access** to hidden-test storage, expected outputs, evaluator
binaries, evaluator configuration, or the signing key. Those resources are not
mounted read-only; they are absent from the sandbox namespace. A distinct
evaluator-runner identity owns the hidden material, invokes the candidate
through the controller's narrow input/output channel, compares results outside
the sandbox, and exposes only signed verdicts and bounded diagnostic codes.

## Hardware and role separation

| Role | Placement | Responsibilities | Explicitly prohibited |
|---|---|---|---|
| Generator/trainer Spark | Operator inventory role `spark_trainer` | Candidate inference, LoRA training, evidence retrieval, orchestration | Evaluator signing key; any read, write, or listing access to hidden tests and evaluator binaries |
| Evaluator controller | Operator inventory role `spark_evaluator`, controller identity | Accept content-addressed candidates, mediate opaque input/output, enforce policy, sign bounded verdicts | Exposing hidden-test content, expected outputs, evaluator binaries, or signing material to candidates |
| Candidate sandbox | Operator inventory role `spark_evaluator`, untrusted per-run identity | Execute one candidate against opaque controller input within resource ceilings | Reading, writing, listing, mounting, or discovering hidden evaluator resources; signing verdicts |
| Hidden evaluator runner | Operator inventory role `spark_evaluator`, evaluator-only identity | Own hidden tests, compare outputs, emit result to controller | Accepting candidate-controlled verdicts; sharing its namespace with the sandbox |
| Codex | Generator/trainer Spark | Implement campaign code, run the frozen protocol, assemble artifacts, stop on failed gates | Rewriting gates after results; self-signing evaluator receipts |
| Grok | Generator/trainer Spark, separate session | Attack the protocol, propose alternative explanations, review pre-freeze task families and post-run findings | Acting as the mechanical evaluator; changing frozen tests during a run |
| Human operator | Out of band | Supplies host inventory, opens maintenance window, freezes protocol, authorizes service interruption | Supplying secrets through committed files or command arguments |

This is functional separation, not a claim that Codex and Grok are statistically
or organizationally independent. Mechanical evaluation remains on the second
Spark under a frozen evaluator identity.

## Authoritative evidence ledger

SQLite is the system of record because the pilot needs atomic writes,
dependency queries, crash recovery, and deterministic export. The ledger uses
WAL mode, `synchronous=FULL`, foreign keys, a single-writer process, and database
triggers that reject `UPDATE` and `DELETE` for evidence tables. Corrections and
retractions append events; they do not alter history.

### Required logical records

| Record | Required fields |
|---|---|
| Campaign | campaign ID, protocol hash, source commit, model revision, data manifest hash, evaluator hash, policy hash, seed set, creation time |
| Run | run ID, arm, task ID, seed, parent checkpoint, start/end state, host role, software manifest hash |
| Evidence event | event ID, event type, transaction time, valid time, subject, payload hash, source class, disposition, evaluator identity |
| Dependency edge | parent event ID, child event ID, edge type, insertion event ID |
| Candidate | candidate ID, parent candidate, mutation family, patch hash, requested authority, prompt hash, model/adapter hash |
| Verdict | candidate ID, correctness, performance, hidden-test-set hash, evaluator revision, signed receipt hash |
| Correction | superseded event ID, replacement event ID, reason code, correction source, signature |
| Effect receipt | request ID, identity, normalized action hash, decision, policy hash, sandbox ID, timestamps, exit status, output hashes, environment-diff hash, signature |
| Checkpoint | last completed phase, last durable event ID, ledger hash, projection generation, artifact manifest hash |

Every payload larger than the ledger's bounded inline limit is stored as a
content-addressed blob. The ledger stores its SHA-256 digest, media type, size,
and relative bundle path. Absolute host paths never enter portable artifacts.

### Event states and lineage rules

- `observed` means a source reported an outcome; it is not automatically true.
- `verified` means the frozen evaluator produced a valid signed verdict.
- `promoted` means all correctness, evidence, security, and campaign gates pass.
- `rejected` remains retrievable as bounded negative evidence.
- `retracted` remains in history but is excluded from the current valid view.
- Descendants of a corrected or retracted premise become `stale-dependent`
  until independently re-evaluated.
- Copied failures and descendants sharing the same root cause count as one
  failure family for evidence weight.
- An absent or invalid receipt makes the associated result ineligible for
  promotion. It does not become an implicit success.

## Qdrant projection

Qdrant contains embeddings and retrieval metadata derived only from the current
valid ledger view. Each point includes the source event ID, source payload hash,
validity state, failure-family root, projection generation, and embedding model
revision.

Qdrant is never authoritative:

- Deleting a collection must not lose campaign evidence.
- Rebuild from a checkpointed ledger must reproduce the same point IDs,
  payload hashes, and projection-generation digest.
- Each arm and run receives an isolated collection namespace.
- Retrieval results without a matching ledger event and projection generation
  are rejected.
- A projection write occurs only after the ledger transaction commits. Failed
  projection writes enter a rebuild queue; they never roll back evidence.

## Variation and promotion loop

```mermaid
sequenceDiagram
    participant A as Variation agent
    participant L as Evidence ledger
    participant B as Authority broker
    participant E as Frozen evaluator
    A->>L: Query current valid evidence
    L-->>A: Successes, bounded failures, corrections
    A->>L: Append candidate proposal and dependencies
    A->>B: Request typed, minimum authority
    B-->>L: Append signed allow or deny receipt
    B->>E: Run candidate in isolated sandbox when allowed
    E-->>L: Append signed correctness and performance verdict
    L-->>A: Promote, reject, abstain, or stale-dependent
```

A candidate is promotable only when all of the following are true:

1. Hidden correctness tests pass exactly.
2. The evaluator receipt and every externally visible effect receipt verify.
3. No valid dependency is retracted or stale-dependent.
4. The candidate stays within its declared mutation locus and authority budget.
5. Its performance is measured by the frozen evaluator, not self-reported.
6. The complete decision replays from the ledger without Qdrant.

## LoRA-only training

The base model is mounted or opened read-only during training. The initial LoRA
profile is frozen before trajectories are generated:

| Parameter | Initial protocol value |
|---|---:|
| Rank | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Learning rate | `2e-4` |
| Maximum epochs | 3 |
| Maximum sequence length | 4096 tokens |
| Early stopping | Validation loss patience 1 evaluation interval |

The implementation PR must discover target module names from the pinned model,
record them in the manifest, and fail closed if they differ from the frozen
allowlist. It must prove that only adapter parameters have `requires_grad=True`
and that base-model file hashes are unchanged after training.

Training rows are derived from ledger events and include the task, retrieved
evidence identifiers, proposed mutation, verdict, failure family, dependency
set, requested authority, and promotion disposition. Free-form secrets and
unbounded chain-of-thought are not stored.

## Dataset and ablation protocol

Generate 36 deterministic micro-repositories across at least six mutation
families. Freeze their manifests before any model rollout:

- 20 trajectory-generation tasks
- 8 development tasks
- 8 held-out evaluation tasks

Task-family separation takes precedence over random row splitting. A repository
template or hidden-test rule used in held-out evaluation must not appear in the
training split.

Trajectory generation uses the 20 training tasks with the frozen-base Arm B
and Arm D policies and two seeds: `20 x 2 x 2 = 80 trajectories`, each capped at
12 candidate attempts, for at most 960 training-generation attempts. Development
selection uses teacher-forced validation loss on the eight development tasks;
it does not run additional autonomous candidate trajectories.

### Phase 1: evidence architecture ablation with frozen base model

| Arm | Memory | Authority enforcement |
|---|---|---|
| A | Verified successes only | Same isolated executor; receipt capture only |
| B | Successes plus ordinary textual failure summaries | Same as A |
| C | Append-only evidence ledger with failure-family collapse and corrections | Same as A |
| D | Arm C plus deny-by-default typed authority broker | Enforced; signed receipts required |

### Phase 2: LoRA contribution ablation

| Arm | Model | Memory and authority |
|---|---|---|
| E | LoRA-adapted model | Full Arm D system |
| F | LoRA-adapted model | Arm A success-only memory control |

The evaluator-controller/candidate-sandbox isolation boundary applies to every
arm. “Authority enforcement” in the tables means the additional typed,
least-privilege action policy; no control arm receives access to hidden
evaluator resources.

The preregistered contrasts are:

| Contrast | Isolated factor |
|---|---|
| B minus A | Ordinary failure summaries beyond success-only memory |
| C minus B | Structured, correction-aware evidence beyond ordinary summaries |
| D minus C | Typed deny-by-default authority enforcement |
| E minus D | LoRA training with the full evidence and authority system fixed |
| E minus F | Full governed-system contribution—structured memory plus typed authority—with the LoRA model fixed |
| `dependency-aware` minus `naive-reuse` and `full-restart` | Correction policy from an identical seeded checkpoint |

Arm E versus Arm F changes both memory architecture and authority enforcement.
It is a full-system contrast and cannot attribute an outcome to evidence memory
alone. Evidence-memory isolation is provided only by Arm C versus Arm B.

Run each held-out task with three frozen seeds and a maximum of 12 candidate
attempts per trajectory. Pair task and seed across arms. Arms A-F contain six
unique executions; Arm D is the frozen-base control for Arm E. They therefore
use `8 tasks x 3 seeds x 6 arms = 144 trajectories`, capped at
`144 x 12 = 1,728 candidate attempts`. The separate correction-shock factor
adds 36 trajectories and at most 432 candidate attempts, for a total held-out
evaluation ceiling of **180 trajectories and 2,160 candidate
attempts**. Including training generation, the autonomous campaign ceiling is
**260 trajectories and 3,120 candidate attempts**. LoRA optimizer minibatches
are tracked separately as steps and tokens, not misreported as candidate
attempts. Budget exhaustion produces `INCONCLUSIVE`; it does not relax gates.

### Correction-shock factor

Correction handling is a separate randomized, matched factor rather than an
informal event inside Arms A-F. Four of the eight held-out tasks are designated
before rollout. For each `(shock task, seed)` block, the evaluator seeds the
**same preregistered accepted premise, candidate state, and dependency graph**
into three cloned ledger checkpoints. Run order within the block is randomized
from a frozen schedule seed, and each clone receives exactly one policy:

| Shock policy | Required behavior at the correction |
|---|---|
| `full-restart` | Clear the agent's active candidate and retrieved-memory view; begin again from the task statement and correction while retaining the external audit ledger |
| `naive-reuse` | Retain pre-shock memory and scores without dependency-aware invalidation; evaluator still records any stale-dependent use |
| `dependency-aware` | Mark all and only affected descendants stale, retain unrelated verified evidence, and prohibit stale-dependent promotion |

The fixed correction occurs immediately after attempt 6 and invalidates the
seeded premise while leaving preregistered unrelated roots valid. Shock
trajectories do not terminate for an early task solution before that boundary:
the best solution is frozen, the evidence scenario continues to attempt 6, and
the correction is always delivered. The seeded pre-shock promoted candidate is
defined to depend on the corrected premise, so every policy receives a real
recovery problem rather than an optional exposure.

Each policy then receives at most six post-shock attempts. Failure to recover by
the end of attempt 12 is recorded as right-censored at six post-shock attempts;
it is never converted to a seventh successful attempt. The study contains
`4 tasks x 3 seeds x 3 policies = 36 trajectories`, capped at
`36 x 12 = 432 candidate attempts`.

## Metrics and decision gates

All formulas, aggregation rules, seeds, stopping rules, and margins are written
to the protocol manifest before the first held-out rollout.

The paired observation unit for Arms A-F is one `(held-out task ID, seed)`
block. The paired observation unit for the correction factor is one
`(shock task ID, seed)` block containing all three policies. Bootstrap intervals
resample these blocks, never individual attempts, candidates, or receipts.
Incomplete blocks are not silently imputed: a protocol or infrastructure loss
that prevents a required paired arm makes the affected comparison
`INCONCLUSIVE`, while a normal within-protocol failure to solve remains a valid
zero-success observation.

| Measure | Definition | Gate |
|---|---|---|
| Invalid promotion rate | Hidden-test-failing promoted candidates / all promoted candidates | Exactly `0` |
| Unauthorized successful effects | Effects executed after deny, outside declared locus, or without valid receipt | Exactly `0` |
| Receipt coverage | Promoted candidates with complete valid verdict and effect receipts / all promoted candidates | Exactly `1.0` |
| Replay agreement | Decisions reproduced from ledger-only replay / all decisions | Exactly `1.0` |
| Invalidation recall | Shock-affected descendants marked stale / all known affected descendants | Exactly `1.0` |
| Invalidation precision | Correctly stale descendants / all descendants marked stale | Exactly `1.0` |
| Repeated-dead-end rate | Attempts whose failure-family root was already validly rejected / eligible attempts | Arm C is at least 25% lower than Arm B and the paired 95% bootstrap interval excludes zero; C-vs-B isolates structured evidence from ordinary summaries |
| Held-out success | Tasks with a verified promoted solution / tasks attempted | Arm C is noninferior to Arm B and Arm D is noninferior to Arm C within 5 percentage points; the corresponding paired 95% lower bounds are at least `-0.05` |
| Authority utility | Matched task success for Arm D versus Arm C plus the preregistered policy-challenge set | Arm D success lower bound is at least `-0.05` versus Arm C, and every challenge is denied with a valid receipt |
| Correction recovery | Time from shock to first verified promotion independent of the corrected premise; no recovery is right-censored at six post-shock attempts | `dependency-aware` has zero stale-dependent promotions, a higher recovery-within-budget rate than `naive-reuse`, and at least 25% lower restricted mean recovery time through attempt 6 with a paired 95% interval excluding zero; versus `full-restart`, its recovery rate must be no lower and its restricted mean point estimate no higher |
| LoRA contribution | Paired held-out success difference, Arm E minus Arm D | Positive point estimate and paired 95% interval excludes zero; otherwise no training-benefit claim |
| Trained full-system contribution | Paired Arm E versus Arm F held-out success and repeated-dead-end rate | Success lower bound is at least `-0.05`, repeated-dead-end rate is at least 25% lower, and the paired dead-end-rate 95% interval excludes zero; this does not isolate memory from authority |
| Efficiency overhead | Tokens, candidate attempts, evaluator seconds, and wall time for Arm E versus same-model success-only Arm F | Report all four; no hidden cost normalization. More than 30% matched wall-time overhead requires `PROMISING_WITH_COST` rather than `PROMISING` |

Hard safety and replay gates take precedence over performance. Universal hard
integrity applies to every arm and shock policy whose data enters a contrast,
not only to the intended treatments. Any observed integrity failure in a
comparison input produces `NOT_SUPPORTED`; an input that never ran or cannot be
assessed makes the affected contrast inestimable. If the planned sample cannot
resolve the paired intervals, the result is `INCONCLUSIVE`.

### Zero-denominator and censoring rules

- A rate with a zero denominator is `NA`, never zero or one.
- If an arm produces no promoted candidates, invalid-promotion and receipt-rate
  fractions are `NA`; their underlying counts must still be zero, and that arm
  cannot receive a positive efficacy disposition.
- Replay agreement has a preregistered nonzero decision count. A zero decision
  count is an infrastructure failure, not perfect replay.
- Invalidation precision and recall have nonzero denominators by construction
  from the seeded shock graph. A zero denominator means the shock fixture is
  invalid and the correction study is `NOT_SUPPORTED`.
- A zero eligible-attempt denominator makes repeated-dead-end efficacy
  `INCONCLUSIVE` for that paired comparison.
- Held-out-success denominators are the frozen task-seed blocks. Ordinary
  no-solution outcomes count as zero; missing arms caused by infrastructure or
  protocol loss make the affected paired comparison `INCONCLUSIVE`.
- Post-shock non-recovery is right-censored at six attempts. Report recovery
  probability by attempt 6 and restricted mean recovery time through attempt 6;
  do not compute an uncensored median by assigning invented recovery times.

### Named closeout gates

The closeout record uses these exact names and freezes their formulas before
held-out rollout:

| Gate | Boolean is true only when |
|---|---|
| `G_RESTORATION` | Gate P12 passes and the signed restore receipt verifies |
| `G_HARD_INTEGRITY` | Every comparison input meets the campaign-wide universal integrity contract below, and each applicable treatment meets its treatment-specific contract |
| `G_ESTIMABLE` | Every required matched block completed; required denominators are nonzero; correction exposure occurred; frozen intervals and all four cost measures are computable within budget |
| `G_EVIDENCE_MEMORY` | Arm C's repeated-dead-end rate is at least 25% below Arm B with paired 95% interval excluding zero, and Arm C held-out success is noninferior to Arm B with lower bound at least `-0.05` |
| `G_AUTHORITY_UTILITY` | Arm D held-out success is noninferior to Arm C with lower bound at least `-0.05`, and every preregistered authority challenge is denied with a valid receipt |
| `G_CORRECTION_BENEFIT` | `dependency-aware` satisfies every correction-recovery comparison in the metrics table, including zero stale-dependent promotions |
| `G_LORA_BENEFIT` | Arm E minus Arm D held-out success has a positive point estimate and paired 95% interval excluding zero |
| `G_TRAINED_FULL_SYSTEM_CONTRIBUTION` | Arm E is noninferior to Arm F in held-out success with lower bound at least `-0.05`, and its repeated-dead-end rate is at least 25% lower with paired 95% interval excluding zero; the claim covers the full governed system, not evidence alone |
| `G_EFFICIENCY` | All four matched Arm E-versus-F cost measures are reported and wall-time overhead is at most 30% |

When `G_ESTIMABLE` is true, all six downstream research gates from
`G_EVIDENCE_MEMORY` through `G_EFFICIENCY` must be concrete booleans. When it is
false, unavailable downstream gates are `UNEVALUATED`, not silently coerced to
false. The [execution runbook](../runbooks/evidence-governed-variation-dual-spark.md)
defines the exhaustive first-match disposition tree and required mixed-outcome
annotations.

#### Campaign-wide versus treatment-specific hard integrity

For **every** Arm A-F trajectory and every `full-restart`, `naive-reuse`, and
`dependency-aware` shock trajectory whose result enters a contrast, universal
integrity requires:

- The frozen evaluator identity and signature verify.
- Every required verdict and externally visible effect receipt exists and
  verifies.
- Ledger integrity and ledger-only replay pass.
- Candidate sandboxes retain zero hidden-test/evaluator access.
- Train, development, and held-out split isolation passes.
- Arm memory, projection, conversation, and checkpoint isolation passes.
- No hidden-test-failing or otherwise protocol-invalid candidate is promoted.

An observed failure of any universal item makes `G_HARD_INTEGRITY=false`, even
when it occurs in a control. If an arm or policy never produces the required
input because of interruption or missing execution, set `G_ESTIMABLE=false` for
the affected contrast; do not manufacture a passing integrity result.

Treatment-specific hard expectations are additional and apply only where the
protocol assigns them: Arms D and E must enforce typed deny-by-default authority
and emit the corresponding decision receipts; `dependency-aware` must achieve
exact seeded-graph invalidation and zero stale-dependent promotion.
`full-restart`, `naive-reuse`, and Arms A-C/F are not failed merely for lacking
those treatment-only mechanisms. Their universal evaluator, receipt, isolation,
ledger, replay, and promotion obligations remain unchanged.

## Runtime authority and receipts

The intended authority layer is OpenShell or an equivalently enforceable local
runtime with deny-by-default filesystem, process, network, and credential
policies. A prompt, system message, tool description, or agent self-report is
not an equivalent substitute.

The evaluator Spark owns an Ed25519 campaign signing key created for this
campaign. The public key and key ID are included in the frozen manifest. The
private key never leaves the evaluator host and never appears in the repository,
logs, shell arguments, environment exports captured as artifacts, or Qdrant.

The broker checks every effect. Child processes inherit a ceiling no broader
than their parent. Network is denied unless a task declares an allowlisted
endpoint. Credentials are provided just in time through an external operator
mechanism and are never returned to the model. Denials are first-class evidence,
not errors to suppress.

## Resumability and failure recovery

- Every candidate attempt has a deterministic idempotency key derived from
  campaign, run, task, seed, parent candidate, and attempt index.
- Phase completion is recorded only after the ledger transaction and artifact
  manifest are durable.
- On restart, the orchestrator verifies the last ledger hash, reconciles any
  signed receipt that arrived after the last checkpoint, and resumes from the
  first incomplete idempotency key.
- An attempt with an unknown effect outcome is quarantined. It is never blindly
  re-executed until the evaluator confirms whether the original action ran.
- Qdrant rebuild is safe at any point because the projection has no unique
  authority.
- Training checkpoints include adapter weights, optimizer state, scheduler
  state, RNG states, data cursor, base-model manifest hash, and ledger cutoff.
- A resumed run with a mismatched protocol, model, evaluator, policy, data, or
  ledger hash fails closed and starts a new campaign ID if the operator chooses
  to continue.

## DeepSeek service boundary

The two-Spark DeepSeek service is an existing workload, not an experiment
dependency. Before interruption, the operator captures a complete private
restore inventory sufficient to restart and verify the workload. That inventory
may contain operationally necessary service definitions and topology, but it
stays outside the repository, ledger public export, agent context, and artifact
bundle. It is untracked and addressed publicly only by its SHA-256 digest.

The service may be stopped only after the restore manifest validates and the
operator opens a maintenance window. Any campaign exit path—success, failure,
interrupt, or budget exhaustion—must run restoration. Restoration is complete
only when the original service identities and hashes match and the same health
and deterministic smoke checks pass. EGV findings are not promotable until the
restore receipt is present.

### Strict public restore receipt

The public restore receipt is a closed, allowlist-only schema. Logical service
IDs are random campaign pseudonyms such as `svc-001`; the mapping to real
services remains only in the private inventory.

| Allowed public field | Type and constraint |
|---|---|
| `schema_version` | Fixed public schema version |
| `campaign_id` | Pseudonymous campaign ID |
| `receipt_id` | Content-derived pseudonymous receipt ID |
| `logical_service_set_id` | Pseudonymous set ID |
| `logical_service_ids` | Array of pseudonymous logical IDs only |
| `private_inventory_digest` | SHA-256 digest only |
| `service_definition_set_digest` | SHA-256 digest only |
| `model_set_digest` | SHA-256 digest only |
| `configuration_set_digest` | SHA-256 digest only |
| `executable_or_image_set_digest` | SHA-256 digest only |
| `expected_service_count` | Nonnegative integer |
| `restored_service_count` | Nonnegative integer |
| `health_check_count` | Nonnegative integer |
| `health_pass_count` | Nonnegative integer |
| `all_health_checks_passed` | Boolean |
| `smoke_input_digest` | SHA-256 digest only |
| `smoke_output_digest` | SHA-256 digest only |
| `smoke_matches_baseline` | Boolean |
| `restoration_outcome` | `RESTORED`, `PARTIAL`, or `FAILED` |
| `signing_key_id`, `signature` | Public receipt-verification material only |

The schema rejects additional fields. In particular, the public receipt must
not contain original service, unit, process, or container names; ports;
hostnames; private DNS aliases; private registry or image names; mount paths;
device identifiers; dependency edges, ordering, or topology; raw launch
definitions; command lines; environment variables; or the pseudonym-to-real-ID
mapping. The final scanner checks both field names and values against these
prohibitions in addition to validating the closed schema.

## Public artifact bundle

The portable result bundle is assembled **only after** DeepSeek restoration
succeeds and its strict public restore receipt is appended to the authoritative
ledger. Any pre-restoration package is provisional, private, and unpublishable.
After restoration, rebuild the public export from allowlisted sources, scan it,
generate a complete file-size-and-SHA-256 manifest, verify every manifest entry,
and seal the final bundle hash. Adding a receipt to an already scanned archive
is forbidden because it would invalidate the scan and seal.

The final portable result bundle contains:

```text
campaign-manifest.json
protocol.json
software-manifest.json
model-manifest.json
data-manifest.json
ledger.sqlite
ledger-export.jsonl
ledger-head.sha256
projection-manifest.json
adapter/adapter_config.json
adapter/adapter_model.safetensors
metrics/per-run.jsonl
metrics/aggregate.json
receipts/evaluator-public-key.pem
receipts/verdicts.jsonl
receipts/effects.jsonl
reports/codex-execution.md
reports/grok-adversarial-review.md
reports/final-disposition.md
restore/public-restore-receipt.json
public-bundle-manifest.json
public-bundle.sha256
```

Before publication, an independent scanner must reject secrets, private
addresses, local usernames, absolute paths, raw environment variables, private
keys, proprietary source content, and every restore-receipt prohibition listed
in the strict schema: original operational names, ports, hostnames, private DNS
or image names, mounts, devices, dependency topology/order, raw launch
definitions, and pseudonym mappings. It must also fail if the public receipt
contains any field outside the schema allowlist. The full private operator
inventory and service restore material remain untracked outside the repository.

## Implementation sequence

Each item is a separate, complete PR. A later PR does not begin until the prior
one is merged or explicitly withdrawn under the repository rules.

1. **Architecture PR:** this ADR, its runbook, and documentation index links.
2. **Evidence core PR:** immutable ledger, deterministic export/replay, Qdrant
   rebuild, correction propagation, and receipt verification.
3. **Evaluation PR:** generated task corpus, split freezer, hidden evaluator,
   authority broker, signing, and correction-shock fixtures.
4. **Variation PR:** pinned-model loader, candidate loop, arm isolation,
   checkpoints, and success/failure retrieval policies.
5. **Training PR:** trajectory filtering, LoRA-only training, base-weight
   immutability proof, and resumable checkpoints.
6. **Campaign PR:** dual-Spark execution artifacts, matched results, independent
   review, redaction report, and final disposition.

Before any implementation PR, the repository block flag must permit feature
work or a separately documented operator override must satisfy the Brutal
Honesty Rulebook. This ADR does not change that state.

## Alternatives considered

### Train a new model from scratch

Rejected for the pilot. Two Sparks can train useful small models, but doing so
would consume the experiment budget while confounding model quality with the
evidence architecture.

### Use Qdrant as the evidence store

Rejected. Vector-store updates are unsuitable as the sole history of
corrections, dependency invalidation, and transactional promotion decisions.

### Store only successful candidates

Retained as a control, not the target design. It cannot test whether bounded
negative evidence prevents rediscovery of known dead ends.

### Let the generator run its own evaluator

Rejected. It permits evaluator tampering, self-reported performance, and
unsigned effects, defeating the central experiment.

### Begin with online reinforcement learning

Rejected. LoRA supervised fine-tuning over filtered trajectories is easier to
bound, replay, and attribute. Online optimization remains future research only
if the LoRA treatment survives held-out ablations.

## Consequences

Positive consequences:

- Evidence survives projection loss and can be corrected without rewriting
  history.
- The experiment separates memory architecture, authority enforcement, and
  adapter training contributions.
- A small model can be tested without disturbing base weights.
- The second Spark supplies a meaningful mechanical evaluator boundary.

Costs and limitations:

- Signed receipts and isolated arm projections add operational overhead.
- Two machines do not produce true institutional independence.
- Synthetic micro-repositories may not predict performance on long-horizon
  production software.
- The sample may be too small to establish the preregistered confidence bounds.
- The design requires implementation and runtime evidence; this ADR alone is
  not proof that any component works.

## Explicit non-claims

This decision does **not** claim that:

- EGV exists or works today.
- Qwen 2B has acquired general long-horizon autonomy.
- ChelatedAI invented agentic variation, persistent memory, RAG, DAGs,
  provenance, capability security, or LoRA.
- An evidence DAG is superior to ordinary summarized memory.
- The LoRA adapter improves held-out performance.
- The proposed runtime is secure merely because a policy is documented.
- A result on generated micro-repositories transfers to ARC-AGI, production
  repositories, or arbitrary agents.
- Hashes and signatures prove semantic truth; they prove identity and
  integrity of recorded material.

Only a completed, independently replayed campaign may support narrower claims
defined by the frozen metrics above.
