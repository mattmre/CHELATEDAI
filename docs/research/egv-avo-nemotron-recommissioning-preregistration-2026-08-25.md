# EGV AVO/Nemotron recommissioning preregistration

Status: amended preregistration; blocked pending the v2 corpus, frozen manifests,
blind solvability proof, and fresh exact-head independent review. No
model-capability, training, or utility result exists.

Original freeze: 2026-08-25

Critical-review amendment: 2026-08-29

Follow-up specification amendment: 2026-08-29

Determinism-review amendment: 2026-08-29

## Amendments 1-3: validity and deterministic-execution repair

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
  held-out analysis, training-treatment separation, scheduler comparison,
  resource isolation, and timing; and
- separates non-model corpus validation from model task-bearing execution so
  neither clearance depends circularly on work that it prohibits.

Changing any confirmatory rule below after model-visible task execution begins
creates a new campaign ID and requires a dated amendment that reports what was
already observed. Changing a corpus-validation rule after
`GO_CORPUS_VALIDATION` similarly requires a new corpus-validation run and new
receipt digests. Neither change may be described as the same preregistered
study.

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

### Two non-circular clearance gates

`GO_CORPUS_VALIDATION` and `GO_TASK_BEARING` are different, ordered decisions.
Neither label may be inferred from file presence, a self-review, a passing unit
test, or the other label.

Here, a **model-task-bearing request** is an inference request to `L` or `S`
that contains a v2 corpus task or task-derived material. The non-model oracle
and blind solver do execute corpus tasks, but only as the separately governed
Gate-A validation runs; they are never campaign-arm or model requests.

An independent exact-head reviewer may return `GO_CORPUS_VALIDATION` only after
the amendment plus the frozen corpus, 18-task slice, evaluator-author solver,
blind-solver source digest, evaluator, production sandbox/controller, receipt
schema, and prohibited-data-scan manifests have no unresolved critical or
major solvability, leakage, or boundary finding. That clearance permits only:

- the evaluator-author oracle check over the corpus; and
- the already-frozen, non-model blind solver over the 18-task slice through the
  production sandbox/controller path.

It does not permit a model call containing a corpus task, `P0` or `P1`
execution, Stage A, trajectory collection, training, development evaluation,
or held-out evaluation. Failure changes the corpus digest and revokes the
clearance; it is never repaired by inspecting expected outputs and rerunning
under the same identity.

`GO_TASK_BEARING` is considered only after the oracle and blind-solvability
receipts are sealed and validated. A fresh independent reviewer must inspect
the exact amended commit, all frozen campaign manifests, the 18/18 receipts,
the prohibited-data scan, rendered prompt/feedback projections, reset and
budget proofs, model-bundle admission receipts, and scheduler/restart proofs.
Only an explicit `GO_TASK_BEARING` with no unresolved critical or major
experimental-validity, leakage, promotion, restart, resource-isolation,
timing, or analysis finding permits the first model request containing v2 task
material.

### Blind model-visible-only solvability proof

An implementation team that did not author the evaluator receives exactly the
public specification, candidate workspace, runtime input, and feedback channel
available to a campaign arm. It receives no expected output, corrected source,
hidden fixture, evaluator seed, private manifest field, or evaluator filesystem
access. Before seeing evaluator-private records, that team freezes a solver
source digest.

After `GO_CORPUS_VALIDATION`, the frozen solver runs through the **production
candidate sandbox and controller path**, not an in-process shortcut, on every
task in a precommitted
18-task solvability slice containing three tasks from each of six families. The
gate passes only if all 18 tasks produce the exact accepted output, every
receipt binds the solver and task digests, and the prohibited-data scan is
clean. A wrong output, scientific timeout, or boundary violation fails the gate
immediately. An infrastructure-classified missing or timed-out node may consume
its single frozen retry; recurrence fails the gate. The slice is
corpus-validation evidence only and is excluded from Stage A, Stage B,
training, development, and held-out evaluation.

The evaluator-author oracle solution must also pass every v2 task, but that
white-box check cannot substitute for the blind gate. If either check fails,
the corpus is revised under a new digest and all prior v2 pilot observations
are discarded rather than used to tune the task construction.

## Research questions and hypotheses

The study tests a model-bundle factor and a protocol factor before training.

- **H1 — model-bundle capability:** the confirmatory endpoint is the paired
  exact-pass difference `S1 - L1` under the persistent protocol. The frozen
  Super bundle is predicted to exceed the frozen Lightning bundle. The
  corresponding static-protocol contrast `S0 - L0` is secondary and
  descriptive. These contrasts include weights, tokenizer, chat template,
  precision, engine compatibility, serving settings, and the frozen logical
  worker assignment; they are not identified as effects of parameter count
  alone.
- **H2 — harness capability:** the confirmatory endpoint is the paired
  exact-pass difference `L1 - L0` within the Lightning bundle. The
  AVO-inspired persistent variation loop is predicted to exceed the matched
  static generate-and-retry loop under the same total model-token,
  model-action, candidate, and wall-clock ceilings. The corresponding Super
  contrast `S1 - S0` is secondary and descriptive.
- **H3 — interaction:** model-bundle and harness effects interact; the harness
  difference is not assumed to be constant across bundles.
- **H4 — trainability:** verified successful Lightning trajectories provide a
  non-empty, diverse dataset that can train a LoRA adapter without altering the
  base model; that dataset does not exist yet and must not be quoted as a
  result.
- **H5 — held-out improvement:** a frozen Lightning LoRA improves paired
  held-out semantic correctness over the unmodified Lightning AVO arm without
  evaluator-boundary or task-family regression.
- **H6 — trace-conditioned deterministic replay:** conditional on the sealed
  campaign DAG and the observed readiness, compatibility, duration, stall, and
  retry trace, deterministic work-conserving replay yields less avoidable
  compatible-worker idle time than deterministic serial FIFO replay. Terminal
  scientific outcomes are held fixed by construction. H6 is an engineering
  replay result, not a randomized live-scheduler effect or model-quality
  claim.

H1-H3 can be answered even if H4-H5 fail. `Q0` is historical descriptive
context only and cannot answer any hypothesis. H6 is an engineering result and
must not be reported as model-quality evidence.

## Model bundles and immutable identity

The exact repository, revision, tokenizer, chat template, weights, precision,
inference engine, generation settings, and license digest must be frozen before
the first model-bearing request.

| ID | Intended role | Candidate family | Mutation |
| --- | --- | --- | --- |
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

Before the first model-task-bearing request, a hashed protocol manifest records
one literal budget tuple for each bundle:

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
| --- | --- | --- | --- |
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
is labeled `ZERO_SIGNAL_STAGE_A` and its remaining six coordinates are sealed
as `NOT_RUN_ZERO_SIGNAL`. They are not scored, counted as failures, or imputed
as zero-valued observations. The arm fails admission by rule, is excluded from
Stage B and held-out evaluation, and cannot resume under this campaign ID.
Logs may be examined after termination, but any repair requires a new amendment
and campaign.

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

An arm stopped in Stage A contributes no Stage-B or held-out observations.
Every contrast requiring that arm is `UNEVALUATED_STOPPED_ARM`; an arm whose
bundle never passed admission is instead `UNEVALUATED_RESOURCE_BOUND`. No unrun
coordinate is imputed and alpha is not reassigned. Specifically, `S1 - L1`
requires both `S1` and `L1`, `L1 - L0` requires both `L1` and `L0`, the two
secondary contrasts require their named pairs, and H3 requires all four base
arms.

Teacher SFT is allowed only when `L1` crossed Stage A, completed its prescribed
Stage-B coordinates without an unresolved integrity failure, and produced
fewer than 24 verified success trajectories, while `S1` crossed Stage A and met
the full 24-trajectory/10-task/four-family threshold in Stage B. Verified `S1`
trajectories may then be used as teacher demonstrations for Lightning. If `L1`
has at least 24 successes but misses only the task- or family-diversity
threshold, teacher SFT is not allowed. Each teacher target must still execute
correctly under the deterministic evaluator, carry full teacher provenance,
and remain confined to the training split. If `L1` fails Stage A, no teacher
SFT or other confirmatory Lightning-adapter promotion is allowed because the
predeclared unmodified `L1` held-out comparator is absent. Teacher generation
and student training are always an explicitly separate treatment.

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

### Canonical records, serialization, and objectives

A source coordinate is one Stage-B `(arm, task_id, generation_seed)` tuple.
For every source coordinate, the canonical success is the lowest attempt index
whose contract-admitted typed final candidate receives an exact semantic pass.
At most that one success and its model-visible same-coordinate prefix may enter
an SFT dataset; later or byte-distinct successes from the coordinate are
discarded from training. The canonical trajectory ends at that success and
contains only the public task/runtime material and the messages, actions,
candidate sources, and allowed feedback that the source arm itself saw before
and at that attempt. It contains no private evaluator field or cross-coordinate
state.

Every training record uses UTF-8 and LF newlines. Structured fields use RFC
8785 JSON Canonicalization Scheme bytes. The frozen Lightning chat-template
function renders the ordered normalized messages; the assistant target is the
canonical typed final envelope, and the tokenizer appends exactly one EOS. The
manifest binds the normalizer, chat-template, tokenizer, BOS/EOS behavior, and
resulting token-ID digest for every record. If prompt plus target exceeds the
frozen sequence length, earliest prompt tokens are removed from the left while
the complete target and EOS are retained. A target that cannot fit by itself is
ineligible and is reported; no task-dependent summarizer or alternate
truncation is allowed. If serialization losses reduce a treatment below its
eligibility threshold, that treatment is ineligible.

`SELF_SFT` uses only canonical `L1` successes and is eligible only when `L1`
meets the Stage-B 24-trajectory/10-task/four-family threshold. `TEACHER_SFT`
uses only canonical `S1` successes and is eligible only under the narrower
teacher condition specified in Stage B: `L1` passed Stage A and completed Stage
B with fewer than 24 successes, while `S1` met the full threshold. Teacher
messages are normalized
to the same typed role schema and rendered with the frozen Lightning template;
every target carries the `S1` model-bundle and source-coordinate digests and
passes deterministic replay. `SELF_SFT` and `TEACHER_SFT` are never pooled.

Both SFT treatments minimize target-only causal language-model cross-entropy:
for each optimizer update, sum negative log probabilities over every unmasked
final-assistant target token including EOS across all accumulated microbatches,
then divide once by that update's total unmasked-target-token count. All prompt
and padding tokens are masked. The language-model-loss coefficient is exactly
`1.0`, label smoothing is `0.0`, and preference, KL, and auxiliary-loss
coefficients are `0.0`.

`PREFERENCE` uses `L1` only. For each coordinate, the chosen response is the
canonical success. The rejected response is the greatest lower attempt index
that is contract-admitted, is an exact semantic failure, and precedes that
success. A coordinate lacking either response contributes no pair. Both
responses are typed final envelopes and share the frozen initial `P1` public
task/runtime prompt with no feedback or lineage appended. At most one pair per
coordinate is admitted. The treatment requires at least 24 pairs spanning at
least 10 tasks and four families; it is otherwise ineligible.

The preference objective is reference-based DPO with sequence log probability
equal to the sum over response tokens including EOS:

```text
-log sigmoid(0.1 * ((log p_adapter(chosen) - log p_base(chosen))
                  - (log p_adapter(rejected) - log p_base(rejected))))
```

The frozen, unmodified Lightning bundle with every adapter disabled is the
read-only reference policy. DPO coefficient is `1.0`, beta is `0.1`, label
smoothing is `0.0`, and auxiliary SFT, KL, and other loss coefficients are
`0.0`. Each optimizer update is the arithmetic mean of the displayed per-pair
loss over all pairs in its accumulated microbatches; the response log
probabilities in the formula are token sums, not length-normalized means. No
reference cache generated under a different bundle digest is valid.
The preference adapter is never merged, stacked, or averaged with either SFT
adapter before evaluation.

Each eligible treatment has exactly one confirmatory adapter replicate. Its
single RNG seed is `2026082911` for `SELF_SFT`, `2026082912` for
`TEACHER_SFT`, or `2026082913` for `PREFERENCE`; that value initializes Python,
NumPy, framework CPU, and accelerator RNGs. Canonically ordered records are
permuted once per epoch using `numpy.random.Generator(PCG64(seed +
zero_based_epoch))`. No second seed, checkpoint, or replicate is selected.
An infrastructure retry must resume the same sealed data cursor, optimizer,
scheduler, scaler, and RNG states; an unrecoverable run is `UNEVALUATED`, not
restarted as a new replicate.

If more than one treatment is eligible, each produces a separately named
adapter, dataset digest, optimizer ledger, and held-out arm. Development and
held-out evaluator outputs never enter optimizer inputs, retrieval memory, or
model prompts. Base weights are read-only. Adapter tensor inventory, data
cursor, optimizer state, RNG state, and exact dataset digests must survive
reload. The final optimizer step is the sole confirmatory checkpoint, with no
development-score or evaluator-driven selection.

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
above, corpus/evaluator/sandbox digests, the canonical arm list, and the
counterbalancing algorithm below. Its private identity projection is
unavailable to every model, trainer, retrieval store, and development process.
No held-out task may appear in the solvability slice, Stage A, Stage B,
development, or training datasets.

The eligible confirmatory held-out arms are:

- each of unmodified `L0`, `L1`, `S0`, and `S1` that crossed its own Stage-A
  floor; and
- every eligible, frozen Lightning training treatment with `P1`, each as a
  distinct arm.

The canonical held-out arm list is `L0`, `L1`, `S0`, `S1`, `SELF_SFT`,
`TEACHER_SFT`, `PREFERENCE`; filtering removes Stage-A-stopped or
resource-bound base arms and ineligible or failed training treatments without
changing the order. A removed arm contributes no held-out row. Every contrast
that depends on it is `UNEVALUATED` with the applicable stopped-arm,
resource-bound, or treatment-ineligible reason; no pass/fail value, confidence
interval, or synthetic comparator is imputed.

The evaluator runs the same task/seed coordinates, order, reset rules, feedback,
and within-bundle budgets for every applicable arm. For the primary binary
outcome, an observation equals `1` if **any** candidate passes the exact
semantic evaluator within that coordinate's frozen aggregate budget; otherwise
it equals `0`. Within an admitted held-out arm, a missing, aborted, timed-out,
crashed, or selectively skipped coordinate is `0`. Stage-A-stopped arms are not
admitted and therefore are not assigned 360 zeros. Selective reruns are
forbidden. A predeclared exact rerun is a complete new pass over all 120 tasks,
three seeds, and the same admitted confirmatory arms from sealed initial state.

Primary metric: paired exact semantic pass rate. Secondary metrics: distinct
tasks solved, family-stratified pass rate, attempts and actions to first pass,
wrong-output and runtime-exception rates, repeated-candidate rate, model tokens,
wall time, evaluator time, and accelerator busy/idle time.

### Frozen paired analysis

For any named contrast `A - B`, define
`d(task, seed) = pass_A(task, seed) - pass_B(task, seed)`. Its point estimate is
the arithmetic mean of all 360 paired values. The deterministic bootstrap is
both task-clustered and family-stratified:

1. order the six families and the 20 tasks within each family exactly as in the
   held-out manifest, and initialize a fresh
   `numpy.random.Generator(PCG64(contrast_seed))`;
2. for each of 100,000 replicates and each family in manifest order, sample 20
   task positions with replacement from that family;
3. retain all three seed-level `d` values for each sampled task, average those
   three values to a task mean, and take the unweighted mean of all 120 sampled
   task means, thereby retaining exactly one-sixth weight per family; and
4. sort the 100,000 replicate means ascending. For probability `p`, the
   nearest-rank quantile is element `ceil(p * 100000)` in one-based indexing.

The independent bootstrap streams are frozen as follows:

| Analysis key | Contrast seed |
| --- | ---: |
| `H1_P1_MODEL` (`S1 - L1`) | `2026082901` |
| `H2_L_HARNESS` (`L1 - L0`) | `2026082902` |
| `DESC_P0_MODEL` (`S0 - L0`) | `2026082903` |
| `DESC_S_HARNESS` (`S1 - S0`) | `2026082904` |
| `DESC_INTERACTION` | `2026082905` |
| `PROMOTE_SELF_SFT` (`SELF_SFT - L1`) | `2026082921` |
| `PROMOTE_TEACHER_SFT` (`TEACHER_SFT - L1`) | `2026082922` |
| `PROMOTE_PREFERENCE` (`PREFERENCE - L1`) | `2026082923` |

H1 and H2 form one confirmatory family with familywise one-sided alpha `0.05`.
A fixed Bonferroni allocation gives each hypothesis alpha `0.025`; alpha is not
recycled if the other contrast is unavailable. The decision interval for each
is the one-sided interval `[q_0.025, +infinity)`. H1 is supported only if the
`S1 - L1` lower endpoint is greater than zero; H2 is supported only if the
`L1 - L0` lower endpoint is greater than zero. If either named arm was excluded
after Stage A, that hypothesis is `UNEVALUATED_STOPPED_ARM`, not accepted or
rejected; a resource-bound arm produces the corresponding
`UNEVALUATED_RESOURCE_BOUND` status. The `S0 - L0` and `S1 - S0` contrasts and the
`(S1 - S0) - (L1 - L0)` interaction report descriptive `q_0.025` to
`q_0.975` intervals only. H3 has no confirmatory alpha or significance label.

For a trained arm `A`, the comparator is always unmodified `L1`. If `m`
trained treatments enter held-out evaluation, the multiplicity-adjusted lower
endpoint for every treatment is `q_(0.025 / m)` from its named bootstrap
stream. Ineligible treatments do not enter `m`, and alpha is not reassigned
after held-out execution begins. Unadjusted `q_0.025` and `q_0.975` endpoints
are also reported descriptively.

A trained arm is **not** promoted unless all of these hold:

- paired exact-pass improvement over unmodified `L1` is at least 10 percentage
  points across all 360 observations;
- its multiplicity-adjusted family-stratified task-clustered lower endpoint is
  greater than zero;
- for every family `f`, the paired net exact-pass count
  `sum_{20 tasks in f, 3 seeds}(pass_A - pass_L1)` is at least `-1`;
- no evaluator-boundary, leakage, replay, base-mutation, scheduling-integrity,
  or restoration gate fails; and
- the one complete exact rerun from sealed artifacts independently satisfies
  every rule above; original and rerun observations are reported separately
  and are never pooled.

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
- the scheduler dispatches as many ready DAG nodes as the currently free,
  compatible logical workers can accept without waiting for unrelated lanes.

At each scheduling event, the controller constructs a bipartite graph of ready
nodes and free workers from the frozen compatibility manifest. It dispatches
exactly the maximum-cardinality matching. If more than one maximum matching
exists, choose the lexicographically smallest ordered list of
`(node_priority_tuple, logical_worker_id)` pairs. The node priority tuple is
sealed in the DAG manifest and incorporates manifest ordinal and the frozen
counterbalance rank below. Thus `WORK_CONSERVING` dispatches up to the number
of compatible free workers; it never means that more nodes are launched than
the isolation policy permits.

The closed `scheduler-receipt-v2` JSON Schema and its digest are frozen before
`GO_TASK_BEARING`; unknown properties are rejected and every field is present,
with `null` permitted only where the schema explicitly allows it. Each receipt
binds at least the schema and campaign IDs, split/stage, full isolation key,
DAG-node and node-kind IDs, input and predecessor-receipt digests, model,
protocol, evaluator, sandbox, and manifest digests, logical worker and worker-
capability digests, resource-lease digest, sequence number, event kind,
controller monotonic nanoseconds, UTC audit timestamp, progress counter,
model-call/token/action/candidate budget counters, retry count, checkpoint and
output digests, terminal reason, and signature. Signing covers the RFC 8785
canonical private receipt with its required `signature` field set to `null`;
the signed value is then inserted and the final private-receipt digest is
computed. A public projection records that original receipt digest while
receiving its own projection digest after private topology fields are redacted.

An accepted progress receipt must increment both the sequence number and
progress counter by exactly one, name the preceding receipt digest, and satisfy
the frozen predicate for its node kind:

- `COORDINATE` progress is exactly one completed model call with bound
  request/response and debited token digests, one completed tool action with
  bound action/result/exit/resource digests, or one completed candidate
  evaluation with bound candidate/evaluator/outcome digests;
- `TRAINING_TREATMENT` progress is exactly one optimizer update committed
  **and** a forward-moving data cursor, with before/after optimizer, scheduler,
  scaler, RNG, cursor, and adapter-state digests;
- `BUNDLE_ADMISSION` or `SERVICE_RUNTIME` progress is one completed frozen load
  phase, health generation, or warmup generation with its attestation and
  output digest;
- `CORPUS_VALIDATION` progress is one completed oracle/blind-solver evaluation
  with solver, task, evaluator, sandbox, and outcome digests; and
- `ARTIFACT_VALIDATION` progress is one completed manifest-named artifact check
  with input, validator, and pass/fail output digests;
- `ANALYSIS_REPLAY` has no intermediate progress event: only its completed,
  schema-valid terminal replay report counts; and
- `CONTROLLER_RUNTIME` progress is one completed dispatch or terminalization
  state transition binding the ready-set, dependency, matching, worker-lease,
  and resulting scheduler-state digests.

A terminal event additionally binds the validated terminal output and reason.
A new timestamp, log line, heartbeat, reserialized metadata object, checkpoint
name, or content digest that does not satisfy the applicable completed-work
predicate is not progress. A checkpoint may accompany genuine progress but can
never create progress by itself. Unknown node kinds or progress-event kinds are
rejected.

Heartbeats use their own event kind and may advance receipt sequence but never
the progress counter. Active jobs emit a sealed heartbeat every 30 seconds.
Three consecutive missed heartbeats mark the job `SUSPECT`; a 10-minute
interval with neither an accepted genuine-progress receipt nor a terminal
receipt marks it `STALLED`. Repeated heartbeats and metadata-only checkpoint
churn do not postpone the stall deadline.

Every DAG node permits at most one infrastructure retry from its latest
accepted checkpoint receipt under the same input and isolation digests.
If no genuine-progress checkpoint exists, that sole retry restarts from the
sealed initial-node state; it never fabricates an intermediate checkpoint.
`BUNDLE_ADMISSION` therefore has two total load attempts: the initial attempt
plus one retry. After admission, each immutable bundle service has one reload
ceiling for the **entire campaign**, regardless of whether loss occurs before
Stage A, during Stage A/B or training, or during held-out evaluation. The
controller runtime likewise has at most one infrastructure restart for the
campaign. A retry or reload is a ceiling, not an entitlement: semantic failure,
evaluator rejection, an invalid output, or policy denial never triggers it.

The exhausted dispositions are frozen:

- `CORPUS_VALIDATION` becomes
  `CORPUS_VALIDATION_FAILED_INFRASTRUCTURE`; Gate A fails and no campaign arm
  starts.
- `BUNDLE_ADMISSION` becomes `UNEVALUATED_RESOURCE_BOUND`; all dependent arms
  are excluded without substitution.
- `SERVICE_RUNTIME` before held-out becomes `UNEVALUATED_SERVICE_LOSS`; the
  active consumer receives its own coordinate or training exhausted
  disposition, unstarted dependent nodes are
  `NOT_RUN_UPSTREAM_INFRASTRUCTURE`, and dependent held-out contrasts are
  unevaluated rather than imputed.
- `COORDINATE` becomes `MISSING_INFRASTRUCTURE_RETRY_EXHAUSTED`; it is a
  stage-gate failure, or `0` only when the arm was already admitted to held-out
  evaluation.
- `TRAINING_TREATMENT` becomes `UNEVALUATED_TRAINING_INFRASTRUCTURE`; no new
  seed, replicate, or checkpoint may replace it and no held-out adapter arm is
  created.
- `ARTIFACT_VALIDATION` or `ANALYSIS_REPLAY` becomes
  `UNEVALUATED_ARTIFACT_INVALID`; no promotion, hypothesis support, or replay
  claim may depend on it.
- `CONTROLLER_RUNTIME` becomes `ABORTED_CONTROLLER_RETRY_EXHAUSTED`; remaining
  nodes are not run and confirmatory hypotheses are unevaluated.

If fatal `SERVICE_RUNTIME` loss occurs after an arm enters held-out evaluation,
its unfinished coordinates remain missing and score `0` under the frozen
held-out rule, but every affected confirmatory contrast is
`UNEVALUATED_SERVICE_LOSS` and promotion is blocked. Sealed partial results are
reported descriptively. No selective rerun is allowed.

A valid checkpoint binds the conversation/lineage state, candidate workspace,
tool and sandbox state, supervisor state, generation and framework RNG states,
budget counters, training data cursor/optimizer state where applicable, and
previous output receipts. Committed calls, tokens, actions, candidates, and
optimizer updates remain charged; a restart does not reset scientific budgets
or wall time. The scheduler suppresses duplicates by the full
node/input/checkpoint digest and accepts only one successor for a predecessor
receipt. A second stall, unverifiable checkpoint, counter rollback, or
conflicting successor applies the exhausted disposition above.

### Timing, isolation, warmup, and arm order

All scientific durations use the controller's monotonic clock; UTC timestamps
are audit labels only. For a coordinate, queue delay begins when the node
becomes dependency-ready and ends at the coordinate start defined below,
including time awaiting a lease and the task-neutral reset/attestation. Bundle
load and warmup are separate prerequisite DAG nodes rather than coordinate
queue time. They are excluded from the coordinate wall-clock ceiling but
included in campaign makespan, service-overhead, and H6 replay metrics.

Only task-neutral work may occur before coordinate start: acquire the frozen
resource lease, reset an empty generic worker/process namespace, attest the
immutable bundle and runtime, and verify the already-completed warm-service
receipt. The controller then seals the coordinate-start receipt and monotonic
start timestamp **before** it reads or materializes task-specific payload bytes,
renders or tokenizes a task prompt, initializes `P0`/`P1`, a supervisor,
lineage, world state, or task workspace, queries or opens task retrieval state,
or performs any other protocol/task computation. Task identity and content
digests may be present in the scheduler envelope solely to select and bind the
node; their payloads may not be opened or transformed before the start receipt.

A coordinate stops at the first sealed exact pass or, if none occurs, when the
earliest model-call, token, action, executed-candidate, or wall-clock ceiling is
reached, the final allowed candidate is evaluated, or an unrecoverable terminal
failure is sealed. All task rendering/tokenization, protocol and supervisor
initialization, retrieval, lineage/world-state operations, model latency,
model-service waiting, harness/tool work, candidate execution, evaluator time,
checkpointing, stall detection, retry recovery, and reload time after start
count against coordinate wall time. The clock is never paused.

The resource manifest freezes each logical worker's hardware/driver/runtime
digest, model-service configuration, CPU and memory quota, accelerator lease,
and permitted concurrency. A model service has exclusive use of its assigned
accelerator during measured execution; training and inference never overlap on
that accelerator, no unrelated accelerator workload is permitted, and at most
one model-bearing coordinate runs per model service. Candidate sandboxes and
evaluators use identical frozen CPU/memory/process/time quotas across arms and
cannot access an accelerator. A lease conflict, unrecorded workload, quota
change, or compatibility drift is an integrity failure for affected
coordinates and cannot be fixed by a selective rerun. The fixed logical-worker
assignment remains part of the disclosed model-bundle contrast.

After every cold bundle load or allowed reload, run exactly one frozen non-task
health generation followed by two frozen non-task warmup generations with
generation seeds `2026082900` and `2026082901`, in that order. Their prompt and
rendered-token digests are frozen in the model manifest and contain no corpus,
task-derived, evaluator, or training material. They are excluded from
coordinate budgets and scientific outcomes but included in service-overhead
reporting. No per-arm or data-dependent warmup is allowed.

Arm execution is counterbalanced independently within each model service. For
the participating arms on a service, preserve their order from the canonical
list and assign zero-based indices. For `n` arms, define the first Williams row
as `w[0] = 0`; for `k = 1..n-1`, `w[k] = (k + 1) / 2` when `k` is odd and
`w[k] = n - (k / 2)` when `k` is even. Row `r` adds `r` modulo `n` to every
index. If `n` is odd, append the reverse of every row. A coordinate with
zero-based manifest ordinal `j` uses row
`j modulo row_count`. Every arm in that service's row for coordinate `j` must
reach a terminal state before the service begins coordinate `j + 1`; the two
different model services have no cross-service barrier and may run
concurrently. Stage A uses the original base-arm rows and removes a stopped arm
without recomputing them. Stage B and held-out execution compute and seal their
rows once from the admitted arms before that stage starts. This changes only
cross-arm dispatch priority; each arm's frozen task/seed order is unchanged.

### Model-load and service exceptions

Bundle admission occurs before model-task-bearing execution and permits at most
two load attempts, each with a 30-minute deadline, followed by the fixed health
and two-warmup sequence above. Load and warmup time do not enter coordinate
budgets. If a bundle fails both attempts, it is
`UNEVALUATED_RESOURCE_BOUND`; no arm using it starts and no substitute model is
allowed.

After admission, the one campaign-wide reload defined above may restore the
same immutable bundle after infrastructure loss in any stage, including before
held-out execution. Coordinates already sealed remain sealed; an active node
resumes only from its verified checkpoint and consumes that node's single
retry. Reload time is reported separately, counts against an already-started
coordinate's wall clock, and does not erase avoidable-idle time. A digest
change, exhausted reload, or inability to restore a protected service applies
the frozen `SERVICE_RUNTIME` disposition and cannot be repaired by selectively
rerunning coordinates.

Any avoidable accelerator idle interval over five minutes while a compatible
runnable DAG node exists is a campaign defect. The measurable implementation
work is tracked in GitHub issue #307.

### Scheduling comparator for H6

H6 compares two deterministic replays over the same sealed campaign DAG and
closed-schema receipt trace:

- `FIFO_SERIAL`: one global manifest-order queue; it starts the next ready node
  only after the current node reaches a terminal state; and
- `WORK_CONSERVING`: at each scheduling event, apply the frozen
  maximum-cardinality compatible matching and lexicographic tie-breaker above.

The live campaign uses `WORK_CONSERVING`. Afterward, a deterministic
discrete-event replay applies both policies to the sealed DAG edges, root-node
exogenous release times, compatibility, observed node service durations,
stalls, and retry durations. The only recorded calendar events replayed at
their original campaign-relative offsets are exogenous worker/service
availability and failure/recovery intervals. Recorded child dependency-release,
dispatch, and completion timestamps are validation targets, never readiness
inputs. A failure enters the exogenous stream only when its sealed
infrastructure classification is independent of node content; otherwise it
remains attached to that node's fixed service/terminal trace and is not replayed
as a calendar event.

For each policy, a root becomes ready at its frozen exogenous release. Every
non-root release is recomputed endogenously as the replayed completion time of
its last required predecessor; a failed required predecessor instead applies
the frozen upstream-failure disposition and never releases the child. A node
may dispatch only when every required predecessor has reached its required
terminal state and a compatible worker is available. The validator rejects any
trace in which `dispatch_time(node)` is earlier than any required predecessor's
replayed completion time, or in which worker availability, compatibility, or
the counterbalanced stage barrier is violated.

The replay copies every node's terminal reason and scientific output digest
unchanged; it may alter only dispatch and completion times implied by the frozen
durations and exogenous availability/failure intervals. Each observed duration
remains attached to its node even if replay assigns another compatible logical
worker; no worker-specific duration is estimated. It reports makespan,
compatible-worker idle minutes while work was ready, queue delay, retry
recovery time, and duplicate-suppression count.

H6 is supported only if a clean replay first reproduces the observed
work-conserving dispatch/terminal timeline from the sealed receipts and the
counterfactual work-conserving replay has strictly less avoidable
compatible-worker idle time than `FIFO_SERIAL`. Equal or higher idle time is a
negative H6 result; a pre-predecessor dispatch, receipt discontinuity, or failed
reproduction is `UNEVALUATED_REPLAY_INVALID`. No scheduler replay may be used
as quality or causal live-performance evidence.

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
- **`L1` misses the Stage-A floor:** exclude it from Stage B and held-out,
  prohibit every Lightning adapter treatment including teacher SFT, and mark
  every dependent contrast `UNEVALUATED_STOPPED_ARM`.
- **`L1` passes Stage A, completes Stage B with fewer than 24 verified
  successes, and `S1` yields at least 24 verified Stage-B trajectories spanning
  at least 10 tasks and four task families:** apply the predeclared
  `TEACHER_SFT` treatment without attributing the gap solely to model size. If
  `L1` has at least 24 successes but fails only diversity, teacher SFT remains
  ineligible.
- **The one-sided lower endpoint for `L1 - L0` is greater than zero:** support
  H2 for a harness contribution on this task distribution. A positive point
  estimate alone is insufficient.
- **The one-sided lower endpoint for `S1 - L1` is greater than zero:** support
  H1 for a model-bundle contribution on this task distribution. A positive
  `S0 - L0` point estimate or interval is descriptive only.
- **`S1` exceeds the additive expectation of bundle and harness effects:**
  report the frozen interaction estimate and interval. Do not call it
  confirmatory in this campaign.
- **Every arm fails Stage A after the blind gate passed:** stop, report that
  every tested arm missed the predeclared semantic floor, mark H1-H3
  unevaluated, and independently audit the production render and feedback path
  before a new campaign. Do not infer a comparative bundle or protocol effect.
- **Eligible trajectories exist but LoRA does not improve held-out `L1`:**
  report a negative trainability/generalization result; do not promote.
- **LoRA improves but a safety, replay, or leakage gate fails:** quarantine the
  result; do not promote or publish it as validated.
- **A v1-to-v2 or Qwen-to-Nemotron descriptive gap appears:** report historical
  context only; do not compute a causal effect or promotion gate from it.

## Required pre-run proof

### Gate A: `GO_CORPUS_VALIDATION`

Before any oracle or blind-solver production execution:

1. commit this amendment and freeze the corpus generator, public/private corpus
   manifests, 18-task slice, evaluator-author solver, blind-solver source,
   evaluator, production sandbox/controller, closed receipt schema, and
   prohibited-data-scan digests;
2. statically verify that every expected output is defined solely by its public
   specification and runtime input and reject model-visible collisions with
   different targets; and
3. obtain an independent exact-head review that explicitly returns
   `GO_CORPUS_VALIDATION` with no unresolved critical or major solvability,
   leakage, evaluator-boundary, sandbox, receipt-integrity, or scan finding.

That clearance permits the evaluator-author oracle run and the frozen 18-task
blind-solver run only. Both must use the exact production sandbox/controller,
and both must finish with sealed exact-output and prohibited-data-scan receipts.

### Gate B: `GO_TASK_BEARING`

Before the first model-bearing campaign request:

1. validate the evaluator-author oracle on every v2 task and the blind solver
   on all 18/18 solvability tasks, with receipt-chain and scan verification;
2. freeze hashes for the Stage-A, Stage-B, held-out, model-bundle,
   protocol/budget, Lightning-training, scheduler, evaluator, sandbox,
   resource, timing, warmup, counterbalance, receipt-schema, and analysis
   manifests;
3. prove all four base-arm prompts, feedback projections, reset boundaries, and
   typed contracts render without a model call from the frozen manifests and
   contain no private material;
4. prove bundle load, the fixed non-task health/warmup sequence, candidate
   execution on a frozen non-corpus synthetic fixture, receipt binding,
   heartbeat/stall classification, node-kind genuine-progress rejection of
   metadata-only churn, every retry/reload disposition, the pre-render
   coordinate clock boundary, budget and wall-clock continuity, duplicate
   suppression, maximum-cardinality dispatch, resource leases, endogenous
   dependency release with no pre-predecessor dispatch, and counterbalanced
   ordering on a non-corpus synthetic DAG;
5. prove on frozen non-corpus synthetic records that each training treatment's
   selection rule emits at most one canonical trajectory or pair per
   coordinate and uses the frozen serialization, objective, reference policy,
   coefficients, seeds, and no checkpoint/replicate selection;
6. freeze repository, model, tokenizer, chat template, engine, precision, task,
   evaluator, sandbox, seed, feedback, state, budget, LoRA, scheduler, resource,
   timing, and analysis digests;
7. capture the protected-service restore state when a service stop is actually
   required; and
8. obtain a fresh independent exact-head review that explicitly returns
   `GO_TASK_BEARING` with no unresolved critical or major experimental-validity,
   leakage, promotion, restart, resource-isolation, timing, scheduler, receipt,
   training, or analysis finding.

Gate-B rendering and systems proofs may use only their frozen non-corpus
synthetic fixtures. They may not contain a v2 task, task-derived example,
expected output, evaluator feedback, or trajectory; otherwise they are an
unauthorized corpus or model-task-bearing execution and cannot support
clearance.

The adversarial review must attempt corpus underdetermination, task/split
leakage, hidden-feedback exposure, state carryover, stale promotion, evaluator
gaming, deadlock, heartbeat-without-progress, metadata-only checkpoint churn,
task computation before coordinate start, uncovered node-kind retry, pre-heldout
service-loss ambiguity, duplicate work, selective rerun, budget or clock reset,
pre-predecessor dispatch, recorded-ready-time reuse, model-bundle drift,
resource contention, arm-order drift, training-treatment pooling,
pair/trajectory duplication, reference-policy drift, checkpoint selection, and
false progress. An unreviewed Gate-A artifact
change revokes `GO_CORPUS_VALIDATION` and invalidates its receipts. Any
unreviewed campaign-manifest or validated-proof change revokes
`GO_TASK_BEARING`.

## Amendment and independent-review record

The 2026-08-29 Tier-B review of the 2026-08-25 draft returned 58/100. Its
critical finding was information-theoretic corpus underdetermination: the
evaluator required an independent 128-bit oracle while the candidate received
only unrelated public/runtime material and diagnostic-only feedback. Secondary
findings concerned bundle attribution, unmetered supervisor latitude,
cross-coordinate state, under-specified held-out analysis, adaptive training
choices, and non-deterministic stall/retry handling.

A follow-up independent review of the first amended draft found two remaining
critical specification defects: corpus validation depended circularly on a
task-bearing clearance, and a Stage-A-stopped arm was nevertheless required in
held-out contrasts. It also required fixed confirmatory endpoints and alpha,
family-stratified resampling, exact training-record and objective rules,
receipt-bound replay semantics, and fixed timing/resource/order controls. This
second amendment addresses those design findings; it does not represent a
passing re-review.

The next exact-head review found remaining deterministic-execution gaps: the
coordinate clock began after potentially task-specific setup, checkpoint churn
could masquerade as progress, retry/reload handling did not cover training and
pre-heldout service loss, and H6 reused observed dependency-release times
instead of recomputing them. Amendment 3 freezes the earlier clock boundary,
node-kind work predicates, exhaustive retry dispositions, and endogenous replay
dependency releases. It likewise does not represent a passing re-review.

At the time of this amendment, no task-bearing Nemotron request, v2 trajectory,
adapter training step, or held-out observation had run. Model inventory, weight
download, bundle load, and the fixed non-task health/warmup sequence do not
count as model-task-bearing only when they contain no corpus task, task-derived
example, evaluator feedback, or expected-output material.

At the time of this follow-up amendment, the manifests, receipt chains, blind
production proof, scheduler replay, adapters, and held-out results required
above are obligations, not asserted artifacts. Neither
`GO_CORPUS_VALIDATION` nor `GO_TASK_BEARING` has been granted.

This amendment is a **design repair**, not clearance to execute. Gate A permits
only the two non-model corpus-validation runs after its own independent review;
the first model-bearing request remains prohibited until every Gate-B proof is
sealed and a new reviewer evaluates the exact amended commit plus all frozen
manifest and receipt digests. The original 58/100 review remains part of the
evidence history and may not be replaced or hidden by the later disposition.

## Claim boundary

This document is a preregistration and architecture decision. It supports no
claim that Nemotron is better than Qwen on these tasks, that AVO mechanisms
improve EGV, that a dataset or LoRA exists, that two accelerators ran, or that
the research theory is confirmed. In particular, the historical v1 zero-pass
result is diagnostic evidence from an underdetermined corpus, not a Qwen
capability baseline. Those conclusions require the valid v2 frozen live
campaign and its sealed evidence.
