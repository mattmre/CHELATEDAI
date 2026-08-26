# EGV AVO/Nemotron recommissioning preregistration

Status: proposed follow-up campaign; no model, training, or utility result yet.

Date frozen: 2026-08-25

## Why a new campaign is required

The completed EGV/Qwen B/D commissioning campaign used
`Qwen/Qwen3.5-2B-Base`, a source-only response contract, fixed B/D loops, and a
12-attempt budget. It completed 960 attempts. Of 343 candidates that crossed
the source contract, none passed the deterministic semantic evaluator. A
matched formatting repair made 387 failed responses parseable and still
recovered zero semantic passes. The resulting freezer correctly admitted zero
training rows.

That result rejects the sufficiency of the tested model/protocol combination.
It does **not** identify model size as the sole cause, and it does not test an
instruction-tuned agent model inside a persistent, execution-grounded search
loop. This follow-up therefore changes both factors under a design that can
estimate their separate and joint effects.

The prior result remains frozen. No artifact, label, or outcome from it will be
rewritten or retroactively promoted.

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

## Research questions and hypotheses

The study tests a model factor and a protocol factor before training.

- **H1 — model capability:** an instruction-tuned Nemotron agent model produces
  more exact semantic passes than the historical 2B base-model configuration.
- **H2 — harness capability:** an AVO-inspired persistent variation loop
  produces more exact passes than a matched static generate-and-retry loop with
  the same model and inference budget.
- **H3 — interaction:** model and harness improvements interact; the harness
  benefit is not assumed to be constant across model sizes.
- **H4 — trainability:** verified successful Lightning trajectories provide a
  non-empty, diverse dataset that can train a LoRA adapter without altering the
  base model.
- **H5 — held-out improvement:** a frozen Lightning LoRA improves paired
  held-out semantic correctness over the unmodified Lightning AVO arm without
  evaluator-boundary or task-family regression.
- **H6 — work conservation:** dependency-aware scheduling reduces avoidable
  idle accelerator time and exposes stalls without changing scientific
  outcomes.

H1-H3 can be answered even if H4-H5 fail. H6 is an engineering result and must
not be reported as model-quality evidence.

## Models and immutable identity

The exact repository, revision, tokenizer, chat template, weights, precision,
inference engine, generation settings, and license digest must be frozen before
the first model-bearing request.

| ID | Intended role | Candidate family | Mutation |
|---|---|---|---|
| `L` | high-volume variation/execution and LoRA target | NVIDIA Nemotron 3.5 Lightning 30B-A3B | base immutable; LoRA only after eligibility |
| `S` | planning/supervision ceiling | NVIDIA Nemotron 3 Super 120B-A12B | none |
| `Q0` | historical reference only | frozen Qwen 3.5 2B Base campaign | none; no rerun required |

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
- fixed attempt, token, action, and wall-clock budgets; and
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
- a supervisor that detects stagnation, repetition, and budget misuse and can
  redirect strategy without seeing hidden expected outputs; and
- deterministic evaluator authority below the model/harness boundary.

The supervisor may change search strategy but may not change the frozen task,
model identity, evaluator, split, metric, or total budget.

## Factorial capability screen

Before LoRA training, run the following matched 2 x 2 screen:

| Arm | Model | Protocol | Purpose |
|---|---|---|---|
| `L0` | Lightning | `P0` static | small-model repaired baseline |
| `L1` | Lightning | `P1` AVO-inspired | harness effect on trainable model |
| `S0` | Super | `P0` static | model-scale/reasoning effect |
| `S1` | Super | `P1` AVO-inspired | planning ceiling and interaction |

All arms receive the same public task material, seeds, maximum candidate
executions, model-token ceiling, and wall-clock ceiling. Hidden evaluator
content is never model-visible. Historical `Q0` is descriptive context and is
not included in causal contrasts because its protocol and model differ.

### Stage A: bounded pilot

- Six preregistered tasks sampled across at least four task families.
- Two generation seeds per arm: 12 terminal requests per arm.
- Maximum six executed candidates per request.
- Hard checkpoint after request 6 and at Stage-A completion.

An arm crosses the semantic floor only if it produces at least three exact
passes across at least two distinct tasks. An arm with zero passes at request 6
is paused for a frozen diagnostic; it does not automatically consume the
remaining campaign budget.

### Stage B: training-split commissioning

Only arms that cross the Stage-A floor continue across the full 20-task training
split and frozen seeds. An arm is training-data eligible only if it yields at
least 24 independently verified successful trajectories spanning at least 10
tasks and four task families. These are dataset-diversity thresholds, not a
claim that the model is generally capable.

If `L1` misses the threshold but `S1` crosses it, verified `S1` trajectories may
be used as teacher demonstrations for Lightning. Each target must still execute
correctly under the deterministic evaluator, carry full teacher provenance,
and remain confined to the training split. Teacher generation and student
training become an explicitly separate treatment.

## Training treatments

Training begins only after the evaluator freezes a non-empty eligible dataset.

1. **SFT/LoRA:** train Lightning on verified successful trajectories with exact
   typed final targets.
2. **Preference ablation:** if enough matched failures exist, train a separate
   Lightning adapter on verified success-over-failure pairs. It must not be
   combined with the SFT adapter before the frozen comparison.
3. **No online hidden-test learning:** development and held-out evaluator
   outputs never enter optimizer inputs, retrieval memory, or model prompts.

Checkpoint choice uses a frozen development-only rule. Base weights are
read-only. Adapter tensor inventory, data cursor, optimizer state, RNG state,
and exact dataset digests must survive reload.

## Held-out evaluation

The minimum matched comparison is:

- unmodified Lightning with `P0`;
- unmodified Lightning with `P1`;
- Lightning SFT/LoRA with `P1`; and
- Lightning preference adapter with `P1`, if eligible.

The evaluator runs the same hidden tasks, order, seeds, and budgets for every
arm. Primary metric: exact semantic pass rate. Secondary metrics: distinct
tasks solved, family-stratified pass rate, attempts and actions to first pass,
wrong-output and runtime-exception rates, repeated-candidate rate, model tokens,
wall time, evaluator time, and accelerator busy/idle time.

A trained arm is **not** promoted unless all of these hold:

- paired exact-pass improvement over unmodified `L1` is at least 10 percentage
  points;
- the task-clustered 95% bootstrap lower bound for the paired difference is
  greater than zero;
- no task family loses more than one exact pass;
- no evaluator-boundary, leakage, replay, base-mutation, or restoration gate
  fails; and
- the result survives one exact rerun from sealed artifacts.

If the sample is too small for the interval criterion, the result remains a
commissioning observation and must be expanded before a positive research
claim.

## Work-conserving lab architecture

Public artifacts use logical roles only:

- accelerator worker A runs Lightning inference and, when eligible, LoRA;
- accelerator worker B runs Super inference or an independent matched arm;
- trusted CPU workers run deterministic evaluation, orchestration, artifact
  validation, review preparation, and privacy/security scans; and
- the scheduler dispatches every ready DAG node without waiting for unrelated
  lanes.

Every lane persists owner, input digest, start time, last-progress time,
resource assignment, output digest, and terminal reason. Active jobs are polled
asynchronously. A 10-minute no-evidence interval triggers a bounded retry,
reassignment, or fail-closed stop. Any avoidable accelerator idle interval over
five minutes while runnable work exists is a campaign defect. The measurable
implementation work is tracked in GitHub issue #307.

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

## Decision table

| Observation | Decision |
|---|---|
| `L0` and `L1` both fail the Stage-A floor while `S` crosses it | smaller model lacks the measured ceiling; test verified teacher distillation into Lightning |
| `L1` beats `L0` while model is held fixed | evidence for a harness contribution on this task distribution |
| `S0` beats `L0` while protocol is held fixed | evidence for a model contribution on this task distribution |
| `S1` exceeds the additive expectation of model and harness effects | evidence of interaction; report descriptively unless powered for inference |
| every arm fails Stage A | stop; audit task/evaluator validity with oracle solutions before any training |
| eligible trajectories exist but LoRA does not improve held-out `L1` | negative trainability/generalization result; do not promote |
| LoRA improves but a safety, replay, or leakage gate fails | quarantine result; do not promote or publish as validated |

## Required pre-run proof

Before a model-bearing campaign starts:

1. verify one oracle candidate per task against the exact evaluator;
2. prove all four arm prompts and typed contracts render from frozen manifests;
3. prove model load, one bounded generation, candidate execution, receipt,
   restart, and duplicate-suppression paths;
4. freeze model, tokenizer, engine, task, evaluator, seed, and budget digests;
5. capture the protected-service restore state when a service stop is actually
   required; and
6. run adversarial review against leakage, stale promotion, evaluator gaming,
   deadlock, duplicate work, and false progress.

## Claim boundary

This document is a preregistration and architecture decision. It supports no
claim that Nemotron is better than Qwen on these tasks, that AVO mechanisms
improve EGV, that a dataset or LoRA exists, that two accelerators ran, or that
the research theory is confirmed. Those conclusions require the frozen live
campaign and its sealed evidence.
