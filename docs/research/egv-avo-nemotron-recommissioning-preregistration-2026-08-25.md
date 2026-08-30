# EGV AVO/Nemotron recommissioning preregistration

Status: amended preregistration; blocked pending the v2 corpus, frozen manifests,
blind solvability proof, and fresh exact-head independent review. No model,
training, or utility result exists.

Original freeze: 2026-08-25

Critical-review amendment: 2026-08-29

## Amendment 1: solvability and experimental-identification repair

A fresh Tier-B review scored the original draft **58/100** and identified a
critical experimental-validity defect: the inherited `source-only-v1` corpus
asks the candidate to recover an evaluator-only 128-bit oracle that is
independent of the public specification and runtime input. Diagnostic feedback
does not supply the missing information. The accepted output is therefore not
a function of what the model can observe.

This amendment was made before any task-bearing Nemotron request. It:

- disqualifies the v1 corpus from model, harness, and training-capability
  comparisons while retaining its historical runtime and contract diagnostics;
- requires a new learnable v2 corpus and a blind production-sandbox solvability
  gate;
- treats Lightning and Super as complete model-bundle contrasts, not as a clean
  parameter-count intervention;
- freezes task ordering, seeds, resets, feedback, budgets, stopping, retries,
  held-out analysis, training-treatment separation, and scheduler comparison;
  and
- blocks all task-bearing execution until this amendment and its required
  manifests receive an independent exact-head review.

Changing any confirmatory rule below after model-visible task execution begins
creates a new campaign ID and requires a dated amendment that reports what was
already observed. It may not be described as the same preregistered study.

## Why a new campaign is required

The completed EGV/Qwen B/D commissioning campaign used
`Qwen/Qwen3.5-2B-Base`, a source-only response contract, fixed B/D loops, and a
12-attempt budget. It completed 960 attempts. Of 343 candidates that crossed
the source contract, none passed the deterministic semantic evaluator. A
matched formatting repair made 387 failed responses parseable and still
recovered zero semantic passes. The resulting freezer correctly admitted zero
training rows.

The v1 public task material and opaque runtime input did not determine the
expected output. Each expected answer incorporated a separate 32-hex-character
(128-bit) oracle derived only inside the evaluator. The closed diagnostic
reported outcome classes but did not expose a learnable rule or the missing
oracle. Under that construction, a model-visible solver cannot infer the
accepted unseen answer except through leakage or chance. Formatting repair
could improve contract admission, but it could not create absent task
information.

The v1 result therefore establishes useful runtime facts: the generator often
violated the response contract, the repair path restored parseability, the
semantic evaluator rejected admitted wrong answers, and the freezer admitted
no unverified row. It does **not** establish that Qwen lacked the relevant
semantic capability, that B/D search was inferior to a persistent harness, or
that a larger model is necessary. The v1 corpus is permanently
`DIAGNOSTIC_ONLY_UNLEARNABLE` and is excluded from every Qwen/Nemotron,
Lightning/Super, `P0`/`P1`, trainability, and promotion comparison.

The new campaign first repairs corpus validity. Only after a blind solver proves
that model-visible material is sufficient may it compare an instruction-tuned
agent bundle inside static and persistent execution-grounded search protocols.

The prior result remains frozen. No artifact, label, or outcome from it will be
rewritten, retroactively promoted, or pooled with v2 observations.

## Source-grounded design rationale

The follow-up adapts, but does not claim to reproduce, three recent NVIDIA
research directions:

1. **Agentic Variation Operators (AVO).** AVO promotes the model from a fixed
   candidate generator to a self-directed variation operator that can inspect
   lineage, consult a knowledge base, act, evaluate, repair, critique, and
   verify. The published attention-kernel result ran for seven days on B200
   hardware; this campaign makes no equivalence claim about hardware, duration,
   or outcome.
2. **Persistent, typed harness state.** NVIDIA's NOOA report identifies typed
   inputs and outputs, code as action, explicit durable state, model-callable
   harness APIs, and curated relational memory as separately testable harness
   capabilities. We use these as protocol components rather than assuming that
   a longer prompt is sufficient.
3. **Model routing.** NVIDIA positions Nemotron 3.5 Lightning as a 30B-total,
   3B-active execution model that runs on DGX Spark and supports LoRA, SFT, and
   environment-based RL workflows. Nemotron 3 Super is a 120B-total,
   12B-active reasoning model intended for complex planning. The campaign
   measures a Lightning execution path and a Super planning ceiling instead of
   assigning every step to one model.

Primary sources:

- AVO paper: <https://arxiv.org/abs/2603.24517>
- NVIDIA AVO ARC-AGI-3 report:
  <https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents/>
- NVIDIA NOOA harness report:
  <https://developer.nvidia.com/blog/six-agent-harness-capabilities-for-higher-model-performance/>
- Nemotron 3.5 Lightning:
  <https://developer.nvidia.com/blog/nvidia-nemotron-3-5-lightning-delivers-fast-accurate-specialized-task-execution-for-long-running-agents/>
- Nemotron 3 Super:
  <https://developer.nvidia.com/blog/introducing-nemotron-3-super-an-open-hybrid-mamba-transformer-moe-for-agentic-reasoning/>
- Nemotron 3 family report: <https://arxiv.org/abs/2512.20856>

## Learnable v2 corpus and blind solvability gate

The confirmatory campaign uses a new corpus identity, `learnable-v2`. It must
not reuse the v1 expected-output construction. For every task, the following
functional-dependence contract must hold:

```text
expected_output = F(public_specification, runtime_input)
```

`F` may require reasoning, search, state tracking, or bounded execution, but it
must be deterministic and fully specified by model-visible material. Hidden
tests may select inputs and keep expected outputs private; they may not add an
independent target bit, secret schema value, separator, transition, dependency,
or rule that the public specification and supplied runtime input do not
determine. Corpus generation must reject two task records that expose identical
model-visible material but require different outputs.

Before any campaign arm sees a v2 task, a corpus manifest must freeze:

- corpus generator revision and environment digest;
- public task-family specification and per-family invariants;
- split membership and opaque task identities;
- model-visible file, prompt, tool, runtime-input, and feedback schemas;
- evaluator revision, sandbox policy, canonicalization, limits, and expected
  output digests; and
- a proof that evaluator-private material affects task selection and checking
  only, never the mapping from visible input to accepted output.

### Blind model-visible-only solvability proof

An implementation team that did not author the evaluator receives exactly the
public specification, candidate workspace, runtime input, and feedback channel
available to a campaign arm. It receives no expected output, corrected source,
hidden fixture, evaluator seed, private manifest field, or evaluator filesystem
access. Before seeing evaluator-private records, that team freezes a solver
source digest.

The frozen solver then runs through the **production candidate sandbox and
controller path**, not an in-process shortcut, on every task in a precommitted
18-task solvability slice containing three tasks from each of six families. The
gate passes only if all 18 tasks produce the exact accepted output, every
receipt binds the solver and task digests, and the prohibited-data scan is
clean. A single wrong, missing, timed-out, or boundary-violating result fails
the gate. The slice is corpus-validation evidence only and is excluded from
Stage A, Stage B, training, development, and held-out evaluation.

The evaluator-author oracle solution must also pass every v2 task, but that
white-box check cannot substitute for the blind gate. If either check fails,
the corpus is revised under a new digest and all prior v2 pilot observations
are discarded rather than used to tune the task construction.

## Research questions and hypotheses

The study tests a model-bundle factor and a protocol factor before training.

- **H1 — model-bundle capability:** the frozen Super bundle produces more exact
  semantic passes than the frozen Lightning bundle under the same protocol.
  This contrast includes weights, tokenizer, chat template, precision, engine
  compatibility, and bundle-specific serving settings; it is not identified as
  an effect of parameter count alone.
- **H2 — harness capability:** an AVO-inspired persistent variation loop
  produces more exact passes than a matched static generate-and-retry loop with
  the same model bundle and total model-token, model-action, candidate, and
  wall-clock ceilings.
- **H3 — interaction:** model-bundle and harness effects interact; the harness
  difference is not assumed to be constant across bundles.
- **H4 — trainability:** verified successful Lightning trajectories provide a
  non-empty, diverse dataset that can train a LoRA adapter without altering the
  base model.
- **H5 — held-out improvement:** a frozen Lightning LoRA improves paired
  held-out semantic correctness over the unmodified Lightning AVO arm without
  evaluator-boundary or task-family regression.
- **H6 — work conservation:** dependency-aware scheduling reduces avoidable
  idle accelerator time and exposes stalls without changing scientific
  outcomes.

H1-H3 can be answered even if H4-H5 fail. `Q0` is historical descriptive
context only and cannot answer any hypothesis. H6 is an engineering result and
must not be reported as model-quality evidence.

## Model bundles and immutable identity

The exact repository, revision, tokenizer, chat template, weights, precision,
inference engine, generation settings, and license digest must be frozen before
the first model-bearing request.

| ID | Intended role | Candidate family | Mutation |
|---|---|---|---|
| `L` | execution/LoRA | Nemotron 3.5 Lightning | frozen base |
| `S` | planning ceiling | Nemotron 3 Super | frozen base |
| `Q0` | history only | Qwen 3.5 2B Base | no rerun |

`L` and `S` are frozen as complete bundles. Their comparison necessarily
includes different weights, active capacity, tokenizer/chat-template behavior,
quantization or precision, compatible serving configuration, and any
bundle-required parser. The study may report a **bundle difference**. It may not
attribute that difference solely to model size, architecture, quantization, or
reasoning mode without a separate intervention that holds the other components
fixed. `Q0` uses an unlearnable corpus and a different response protocol; it is
reported in a historical table only and never enters an effect size, interval,
gate, or promotion decision.

The `S` arm is admitted only after a bounded local load and inference probe.
If it cannot fit or run reliably in the available envelope, it is reported as
`UNEVALUATED_RESOURCE_BOUND`; it is not silently replaced with a remote model.

## Protocols

### Static protocol `P0`

`P0` is a repaired control, not the historical source-only contract:

- instruction/chat template appropriate to the selected model;
- typed task input and typed final candidate envelope;
- reasoning and tool events retained privately, with candidate source emitted
  only through the final typed field;
- deterministic execution feedback after each admitted candidate;
- fixed total model-token, model-action, executed-candidate, and wall-clock
  budgets; and
- no persistent cross-attempt lineage beyond the immediately preceding result.

### AVO-inspired protocol `P1`

`P1` has the same external budgets as `P0` and adds:

- a content-addressed lineage graph of candidates, parentage, changes,
  hypotheses, execution results, and verifier receipts;
- agent-selected variation actions rather than a fixed B/D mutation prompt;
- typed inspect, propose, edit, run, compare, critique, and commit operations;
- persistent retrieval of relevant prior failures and successes;
- an explicit task-state/world-model record that is revised when predictions
  disagree with execution;
- novelty pressure against byte-identical or behaviorally identical repeats;
- a deterministic finite-state supervisor that detects stagnation, repetition,
  and budget misuse and redirects among a frozen action table without a model
  call, hidden output, learned policy, or free-text judgment; and
- deterministic evaluator authority below the model/harness boundary.

The `P1` supervisor consumes zero model tokens and zero model calls. Its states,
thresholds, transition order, tie-breaking, and permitted redirects are frozen
in the protocol manifest. It may change search strategy but may not change the
frozen task, bundle identity, evaluator, split, metric, or budget.

### Matched budgets

Before the first task-bearing request, a hashed protocol manifest records one
literal budget tuple for each bundle:

```text
(maximum model calls, maximum input tokens, maximum generated tokens,
 maximum model-selected actions, maximum executed candidates,
 maximum coordinate wall time)
```

For a given bundle, `P0` and `P1` receive the identical tuple. Planning,
critique, repair, tool selection, and finalization all debit the same aggregate
model-call and token ledgers. Harness bookkeeping and the deterministic `P1`
supervisor cannot issue an unmetered model call. A rejected duplicate still
debits the model-call, token, and action ledgers; only an executed candidate
debits the candidate ledger. Exceeding any ceiling terminates the coordinate.

The manifest may freeze different literal token ceilings for `L` and `S` only
where their tokenizers or context limits require it. Such a difference becomes
part of the model-bundle contrast and is reported. It cannot differ between
`P0` and `P1` for the same bundle.

### State, resets, and feedback visibility

The isolation key is `(campaign_id, split, arm, task_id, generation_seed)`. Each
coordinate starts with an empty conversation/KV cache, tool session, candidate
workspace, process namespace, retrieval namespace, lineage store, and
supervisor state. The serving process and immutable weight cache may remain
loaded, but no task content or model state may survive a coordinate reset.

Within one coordinate:

- `P0` may see the public task material and only the immediately preceding
  candidate plus its allowed feedback;
- `P1` may retain lineage and retrieval state from earlier attempts in that
  same coordinate; and
- both protocols see the same typed feedback: contract diagnostics, candidate
  source they emitted, bounded sandbox stdout/stderr, exit status,
  timeout/resource-policy class, canonical observed output, and one semantic
  pass/fail bit.

Neither protocol sees the expected output, evaluator seed, hidden test input,
private fixture, corrected source, oracle material, evaluator stack trace, or
another arm's state. Training and development coordinates do not write to a
held-out namespace. Held-out state resets after every `(task_id, seed)` and
cannot be retrieved by any later held-out coordinate, even within the same arm.
Cross-task or cross-seed persistent memory is a separate future treatment and
is not part of this experiment.

## Factorial capability screen

Before LoRA training, run the following matched 2 x 2 screen:

| Arm | Model | Protocol | Purpose |
|---|---|---|---|
| `L0` | Lightning | `P0` static | Lightning-bundle static baseline |
| `L1` | Lightning | `P1` AVO-inspired | harness effect on trainable model |
| `S0` | Super | `P0` static | Super-bundle static condition |
| `S1` | Super | `P1` AVO-inspired | persistent Super arm |

All arms receive the same public task material, seed identities, maximum
candidate executions, model-action ceiling, and coordinate wall-clock ceiling.
`P0` and `P1` also receive identical token ceilings within each bundle as
specified above. Hidden evaluator content is never model-visible. Historical
`Q0` is descriptive context and is not included in causal contrasts because its
corpus, protocol, and bundle differ.

### Stage A: bounded pilot

The hashed `stage-a-v2-manifest.json` must name six exact v2 task identities
spanning all six families. Its literal task order is `T1` through `T6`. The
generation seeds are frozen as decimal integers `1729` then `3253`. Every arm
runs the same coordinate order:

```text
T1/1729, T2/1729, T3/1729, T4/1729, T5/1729, T6/1729,
T1/3253, T2/3253, T3/3253, T4/3253, T5/3253, T6/3253
```

The scheduler may execute ready coordinates concurrently, but it may not change
that within-arm order. Each coordinate allows at most six executed candidates.
Wrong, timed-out, aborted, or missing terminal outcomes are failures.

The zero-signal rule is final: after coordinate 6, an arm with zero exact passes
is labeled `ZERO_SIGNAL_STAGE_A`, its remaining six coordinates are not run,
and those six outcomes are recorded as missing/fail for the Stage-A gate. Logs
may be examined after termination, but the arm cannot resume under this
campaign ID. Any repair requires a new amendment and campaign.

An arm that survives the checkpoint crosses the semantic floor only if all 12
coordinates finish and it produces at least three exact passes across at least
two distinct task identities. No discretionary diagnostic, budget extension,
or data-dependent task substitution is allowed.

### Stage B: training-split commissioning

Only arms that cross the Stage-A floor continue across the exact 20-task
training split frozen in `stage-b-v2-manifest.json` before Stage A begins. The
Stage-B seed list is exactly `1729`, then `3253`. Task order is the literal
manifest order; within each seed block that order is preserved. An arm is
training-data eligible only if it yields at least
24 independently verified successful trajectories spanning at least 10 tasks
and four task families. Missing coordinates are failures. These are
dataset-diversity thresholds, not a claim that the model is generally capable.

If `L1` misses the threshold but `S1` crosses it, verified `S1` trajectories may
be used as teacher demonstrations for Lightning. Each target must still execute
correctly under the deterministic evaluator, carry full teacher provenance,
and remain confined to the training split. Teacher generation and student
training become an explicitly separate treatment.

## Training treatments

Training begins only after the evaluator freezes a non-empty eligible dataset,
but the training design freezes **before any Stage-A trajectory is generated**.
A separately hashed `lightning-training-v2-manifest.json` must record:

- the exact base-bundle digest and literal LoRA target-module names and tensor
  shapes obtained from a non-task-bearing module inventory;
- rank, alpha, dropout, bias policy, precision, optimizer and all optimizer
  arguments, learning-rate schedule, warmup, batch size, accumulation, sequence
  length, loss mask, epochs or maximum steps, clipping, checkpoint interval,
  RNG seeds, shuffle algorithm, and library/container revisions;
- the eligibility condition, dataset query, canonical ordering, deduplication,
  and immutable digest construction for each treatment; and
- the single checkpoint rule: the final optimizer step is confirmatory.
  Intermediate checkpoints are diagnostics and never see the held-out split.

The confirmatory campaign uses one frozen hyperparameter configuration per
treatment. It performs no adaptive sweep. Target modules and hyperparameters
may be selected from architecture-only inspection and published implementation
guidance, but not from v2 task trajectories, development scores, or held-out
behavior. A desired target-module or hyperparameter change after Stage A begins
terminates the adapter treatment under this campaign ID and requires a new
preregistration.

The following datasets and adapters remain separate:

1. **`SELF_SFT`:** verified `L1` successes only, with exact typed final targets.
2. **`TEACHER_SFT`:** verified `S1` successes only, admitted solely under the
   predeclared `L1`-ineligible/`S1`-eligible condition. Every target carries
   teacher provenance and passes deterministic replay. It is never pooled with
   `SELF_SFT`.
3. **`PREFERENCE`:** verified success/failure pairs from the same frozen source
   treatment and coordinate-matching rule. It trains a separate preference
   adapter and is never merged, stacked, or averaged with either SFT adapter
   before evaluation.

If more than one treatment is eligible, each produces a separately named
adapter, dataset digest, optimizer ledger, and held-out arm. Development and
held-out evaluator outputs never enter optimizer inputs, retrieval memory, or
model prompts. Base weights are read-only. Adapter tensor inventory, data
cursor, optimizer state, RNG state, and exact dataset digests must survive
reload.

## Held-out evaluation

The held-out design is fixed at **120 unique v2 tasks**, exactly 20 from each of
six task families, with three generation seeds per task: decimal integers
`104729`, `130363`, and `155921`. This yields 360 paired `(task_id, seed)`
observations per arm and 120 independent task clusters. A 10-percentage-point
difference equals 36 paired observations, while task clustering prevents the
three seeds from being treated as 360 independent tasks.

The fixed-size planning alternative assumes 18% adapter-only wins, 8%
adapter-only losses, and within-task seed intraclass correlation of 0.10. Under
a normal approximation, 120 task clusters with three seeds provide about 86%
power for a positive lower endpoint even at the most conservative planned
multiplicity of three treatments. These are design assumptions, not observed
facts. If the real discordance or clustering is less favorable, the study stays
underpowered and cannot expand after seeing results.

Before Stage A begins, `heldout-v2-manifest.json` must freeze all 120 opaque task
identities, family labels, literal task order, the three seeds in the order
above, corpus/evaluator/sandbox digests, and arm order. Its private identity
projection is unavailable to every model, trainer, retrieval store, and
development process. No held-out task may appear in the solvability slice,
Stage A, Stage B, development, or training datasets.

The confirmatory held-out arms are:

- unmodified Lightning with `P0` (`L0`);
- unmodified Lightning with `P1` (`L1`);
- unmodified Super with `P0` (`S0`);
- unmodified Super with `P1` (`S1`); and
- every eligible, frozen Lightning training treatment with `P1`, each as a
  distinct arm.

The evaluator runs the same task/seed coordinates, order, reset rules, feedback,
and within-bundle budgets for every applicable arm. For the primary binary
outcome, an observation equals `1` if **any** candidate passes the exact
semantic evaluator within that coordinate's frozen aggregate budget; otherwise
it equals `0`. A missing, aborted, timed-out, crashed, or selectively skipped
coordinate is `0`. Selective reruns are forbidden. A predeclared exact rerun is
a complete new pass over all 120 tasks, three seeds, and confirmatory arms from
sealed initial state.

Primary metric: paired exact semantic pass rate. Secondary metrics: distinct
tasks solved, family-stratified pass rate, attempts and actions to first pass,
wrong-output and runtime-exception rates, repeated-candidate rate, model tokens,
wall time, evaluator time, and accelerator busy/idle time.

### Frozen paired analysis

For any trained arm `A`, define
`d(task, seed) = pass_A(task, seed) - pass_L1(task, seed)`. The reported paired
difference is the arithmetic mean of all 360 values. The deterministic
task-clustered bootstrap is:

1. initialize `numpy.random.Generator(PCG64(2026082901))`;
2. for each of 100,000 replicates, sample 120 task indices with replacement;
3. for every sampled task, retain all three seed-level `d` values and compute
   their mean; then compute the mean across the 120 sampled task means; and
4. sort the 100,000 replicate means ascending.

If `m` trained treatments enter held-out evaluation, the multiplicity-adjusted
lower endpoint for every treatment is the empirical quantile at
`0.025 / m`, selected by the nearest-rank rule (`ceil(p * 100000)`, one-based).
This is a conservative familywise 95% lower bound across `SELF_SFT`,
`TEACHER_SFT`, and `PREFERENCE`; ineligible treatments do not enter `m`.
Unadjusted 2.5% and 97.5% endpoints are also reported descriptively.

The model-bundle effect is reported separately under `P0` (`S0 - L0`) and `P1`
(`S1 - L1`). The protocol effect is reported separately within Lightning
(`L1 - L0`) and Super (`S1 - S0`). The difference-in-differences interaction is
`(S1 - S0) - (L1 - L0)`. Each uses the same task-clustered bootstrap algorithm
and reports its interval, but H1-H3 remain model-bundle/task-distribution
claims. The interaction is descriptive unless a separate power analysis was
precommitted; no post-hoc significance label is allowed.

A trained arm is **not** promoted unless all of these hold:

- paired exact-pass improvement over unmodified `L1` is at least 10 percentage
  points across all 360 observations;
- its multiplicity-adjusted task-clustered lower endpoint is greater than zero;
- no task family loses more than one exact pass;
- no evaluator-boundary, leakage, replay, base-mutation, scheduling-integrity,
  or restoration gate fails; and
- the result survives the one complete exact rerun from sealed artifacts under
  the same rule.

The count, identities, families, seeds, arms, budgets, metrics, and analysis may
not expand or change in response to observed results. If realized uncertainty
does not clear the interval criterion, the result remains a commissioning
observation. A larger study requires a new preregistration and new held-out
identities; these 120 tasks cannot be reused to support the new positive claim.

## Work-conserving lab architecture

Public artifacts use logical roles only:

- accelerator worker A runs Lightning inference and, when eligible, LoRA;
- accelerator worker B runs Super inference or an independent matched arm;
- trusted CPU workers run deterministic evaluation, orchestration, artifact
  validation, review preparation, and privacy/security scans; and
- the scheduler dispatches every ready DAG node without waiting for unrelated
  lanes.

Every lane persists owner, input digest, start time, last-progress time,
resource assignment, output digest, monotonically increasing progress counter,
heartbeat time, retry count, and terminal reason. Active jobs emit a signed
heartbeat every 30 seconds. Three consecutive missed heartbeats mark the job
`SUSPECT`; a 10-minute interval with neither a higher progress counter nor a new
sealed output receipt marks it `STALLED`. Repeated heartbeats with unchanged
progress do not postpone the stall deadline.

Each coordinate permits one infrastructure retry from its latest sealed
checkpoint with the same isolation key and generation seed. Model calls,
tokens, actions, and candidate receipts committed before the stall remain
charged; a restart does not reset scientific budgets. The scheduler suppresses
duplicates by the full coordinate/input/checkpoint digest. A second stall, an
unverifiable checkpoint, or a conflicting duplicate terminates the coordinate
as missing/fail. Scientific failures and evaluator rejections are never
classified as infrastructure retries.

### Model-load and service exceptions

Bundle admission occurs before task-bearing execution and permits at most two
load attempts, each with a 30-minute deadline, followed by one fixed non-task
health prompt. Load time and that prompt do not enter coordinate budgets. If a
bundle fails both attempts, it is `UNEVALUATED_RESOURCE_BOUND`; no arm using it
starts and no substitute model is allowed.

After held-out execution begins, one reload from the same immutable bundle is
allowed after an infrastructure loss. Coordinates already sealed remain
sealed; an active coordinate resumes only from a verified checkpoint under the
single-retry rule. Reload time is reported separately and does not erase
avoidable-idle time. A digest change, second reload failure, or inability to
restore a protected service blocks promotion. It cannot be repaired by
selectively rerunning failed held-out coordinates.

Any avoidable accelerator idle interval over five minutes while a compatible
runnable DAG node exists is a campaign defect. The measurable implementation
work is tracked in GitHub issue #307.

### Scheduling comparator for H6

H6 compares two deterministic schedulers over the same sealed campaign DAG:

- `FIFO_SERIAL`: one global manifest-order queue; it starts the next ready node
  only after the current node reaches a terminal state; and
- `WORK_CONSERVING`: whenever a compatible worker is free, dispatch every ready
  node in manifest ordinal order, breaking ties by arm ID and task ID.

The live campaign uses `WORK_CONSERVING`. Afterward, a deterministic
discrete-event replay applies both policies to the recorded node readiness,
compatibility, observed service durations, stalls, and retry durations. It
reports makespan, compatible-worker idle minutes while work was ready, queue
delay, retry recovery time, and duplicate-suppression count. The comparator may
not alter model outputs or be used as quality evidence. H6 is supported only if
the replay validator reproduces the observed work-conserving schedule and the
work-conserving policy has lower avoidable accelerator-idle time without a
higher missing-coordinate count. Otherwise H6 is negative or unevaluated.

This scheduling architecture does not claim live cross-node trainer/evaluator
transport. Such a claim requires observed exact-byte transfer, remote
execution, receipt continuity, and restart evidence.

## Security boundary

Models, agents, retrieval, and supervisors may propose actions. The runtime
binds identity, credentials, filesystem/network access, policy, isolation,
timeouts, and audit records. Candidate code executes below a denied-by-default
boundary with hidden tests and signing material inaccessible. No prompt or
harness instruction is treated as a security control.

Public bundles exclude credentials, private addresses, usernames, host labels,
private paths, endpoints, deployment topology, raw prompts, candidate source,
hidden tasks, evaluator seeds, and signing material. Prohibited-data and secret
scans run on every publishable artifact.

## Decision rules

- **The v2 blind solver misses any solvability task:** the corpus is invalid;
  no capability arm may run.
- **`L0` and `L1` both miss the Stage-A floor while `S` crosses it:** the
  Lightning bundle misses the measured floor. Apply the predeclared
  `TEACHER_SFT` condition without attributing the gap solely to size.
- **`L1` beats `L0` while the bundle is fixed:** report evidence for a harness
  contribution on this task distribution.
- **`S0` beats `L0` while the protocol is fixed:** report evidence for a
  model-bundle contribution on this task distribution.
- **`S1` exceeds the additive expectation of bundle and harness effects:**
  report the frozen interaction estimate and interval. Do not call it
  confirmatory unless separately powered.
- **Every arm fails Stage A after the blind gate passed:** stop, report a
  negative bundle/protocol result, and independently audit the production
  render and feedback path before a new campaign.
- **Eligible trajectories exist but LoRA does not improve held-out `L1`:**
  report a negative trainability/generalization result; do not promote.
- **LoRA improves but a safety, replay, or leakage gate fails:** quarantine the
  result; do not promote or publish it as validated.
- **A v1-to-v2 or Qwen-to-Nemotron descriptive gap appears:** report historical
  context only; do not compute a causal effect or promotion gate from it.

## Required pre-run proof

Before a task-bearing campaign request starts:

1. commit this amendment and freeze hashes for the corpus, blind-solvability,
   Stage-A, Stage-B, held-out, model-bundle, protocol/budget,
   Lightning-training, scheduler, evaluator, and sandbox manifests;
2. prove every v2 task has one deterministic expected output from its public
   specification and runtime input, reject visible collisions with different
   targets, and verify an evaluator-author oracle candidate on every task;
3. pass the 18/18 blind model-visible-only solvability gate through the exact
   production sandbox/controller path;
4. prove all four base-arm prompts, feedback projections, reset boundaries, and
   typed contracts render from the frozen manifests without private material;
5. prove bundle load, one non-task health generation, candidate execution,
   receipt binding, heartbeat/stall classification, restart, budget continuity,
   and duplicate-suppression paths;
6. freeze repository, model, tokenizer, chat template, engine, precision, task,
   evaluator, sandbox, seed, feedback, state, budget, LoRA, and analysis digests;
7. capture the protected-service restore state when a service stop is actually
   required; and
8. obtain a fresh independent exact-head review that explicitly returns
   `GO_TASK_BEARING` with no unresolved critical or major experimental-validity,
   leakage, promotion, restart, or analysis finding.

The adversarial review must attempt corpus underdetermination, task/split
leakage, hidden-feedback exposure, state carryover, stale promotion, evaluator
gaming, deadlock, heartbeat-without-progress, duplicate work, selective rerun,
budget reset, model-bundle drift, training-treatment pooling, and false progress.
An unreviewed manifest change revokes `GO_TASK_BEARING`.

## Amendment and independent-review record

The 2026-08-29 Tier-B review of the 2026-08-25 draft returned 58/100. Its
critical finding was information-theoretic corpus underdetermination: the
evaluator required an independent 128-bit oracle while the candidate received
only unrelated public/runtime material and diagnostic-only feedback. Secondary
findings concerned bundle attribution, unmetered supervisor latitude,
cross-coordinate state, under-specified held-out analysis, adaptive training
choices, and non-deterministic stall/retry handling.

At the time of this amendment, no task-bearing Nemotron request, v2 trajectory,
adapter training step, or held-out observation had run. Model inventory, weight
download, bundle load, and a fixed non-task health prompt do not count as
task-bearing only when they contain no corpus task, task-derived example,
evaluator feedback, or expected-output material.

This amendment is a **design repair**, not clearance to execute. The first
task-bearing request remains prohibited until every pre-run proof above is
sealed and a new reviewer evaluates the exact amended commit plus all frozen
manifest digests. The original 58/100 review remains part of the evidence
history and may not be replaced or hidden by the later disposition.

## Claim boundary

This document is a preregistration and architecture decision. It supports no
claim that Nemotron is better than Qwen on these tasks, that AVO mechanisms
improve EGV, that a dataset or LoRA exists, that two accelerators ran, or that
the research theory is confirmed. In particular, the historical v1 zero-pass
result is diagnostic evidence from an underdetermined corpus, not a Qwen
capability baseline. Those conclusions require the valid v2 frozen live
campaign and its sealed evidence.
