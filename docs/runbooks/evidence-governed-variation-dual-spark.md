# Evidence-Governed Variation: Dual-Spark Execution Runbook

## Purpose and current status

This runbook defines how operators will stage, run, resume, and close the
Evidence-Governed Variation (EGV) campaign described in
[ADR-0001](../architecture/adr-0001-evidence-governed-variation-agent.md).

> **Current Slice 2 + Evaluation + Variation status:** The bounded evidence-core
> floor, complete CPU-only Evaluation slice, and bounded Variation wiring are
> runnable in this branch. The
> Evaluation smoke freezes 36 deterministic split-disjoint micro-repositories,
> runs a hidden held-out comparator through the pinned Docker isolation adapter,
> and exercises spawned trainer/evaluator IPC. It does not exercise the real
> campaign, Qdrant campaign projection, service control, DeepSeek, hosted
> models, or dual-Spark infrastructure. Variation's default smoke is explicitly
> `floor-fixture`: it exercises ledger/checkpoint/retrieval wiring with a
> test-only CPU gateway and does not claim production authority. Treat command
> output as runtime evidence only when it names its tier and gaps.

The runbook intentionally contains no host addresses, local usernames,
passwords, tokens, private keys, or private workspace paths. Operators provide
those through an untracked inventory and the existing credential mechanism.

## Outcomes

A successful campaign ends with all of the following:

- A revision-verified Qwen 2B base model and a LoRA-only adapter.
- Complete matched ablation results or an explicit disposition from the Phase
  10 exhaustive tree.
- A ledger-only deterministic replay result.
- A correction-shock audit.
- Signed evaluator and authority receipts.
- A redacted portable evidence bundle.
- The pre-existing DeepSeek service restored to its captured identity and
  health state.

Failure to restore the service or satisfy an exact safety gate prevents
promotion of experimental findings.

## Operator-supplied inventory

Keep the inventory outside the repository and outside the evidence bundle. It
must define the following logical values without exposing them to the models:

| Inventory key | Meaning |
|---|---|
| `spark_trainer` | SSH destination for the generator/trainer Spark |
| `spark_evaluator` | SSH destination for the evaluator Spark |
| `trainer_workspace` | Absolute, campaign-specific workspace on the trainer |
| `evaluator_workspace` | Absolute, campaign-specific workspace on the evaluator |
| `artifact_root` | Absolute destination for unredacted campaign artifacts |
| `deepseek_service_set` | Existing services that must be captured and restored |
| `authority_runtime` | OpenShell runtime identity and immutable version |
| `credential_provider` | Existing out-of-band credential mechanism |
| `bootstrap_signer` | Pre-existing out-of-band lifecycle signer; private key never enters either Spark |
| `bootstrap_trust_fingerprint` | Public-key fingerprint authorized at P0 for closed private bootstrap records |

Inventory validation must reject empty values, loopback destinations,
trainer/evaluator aliasing, broad filesystem roots, and values containing
embedded credentials. The automation accepts inventory by file descriptor or
protected local file, never a committed file or command-line secret.

## Slice 2 floor command subset

The currently runnable, read-only or bounded floor paths are:

```text
python -m egv status
python -m egv export
python -m egv replay
python -m egv verify-public
python -m egv smoke --json
python -m egv smoke --json --two-process
python -m egv evaluation freeze --output <new-empty-output-directory>
python -m egv evaluation smoke --json
python -m egv variation smoke --json
python -m egv variation model-preflight --model-root <staged-model-root> --json
```

`status` reads an authoritative ledger, `export` writes deterministic ledger
JSONL, and `replay` either verifies a ledger read-only or replays JSONL into a
new ledger. `verify-public` performs closed cryptographic decision replay from
an exported public projection. It requires a terminal public seal by default;
`--allow-provisional` is the explicit public-CLI option for verifying an
unsealed provisional projection.

The smoke commands are the only bounded execution paths in this slice.
`--two-process` starts the local CPU-only trainer/evaluator fixture and proves
that the evaluator can use only the `ingest_receipt` IPC method while the
trainer remains the single SQLite writer. Evaluator startup installs an
immutable `sqlite3.connect` audit denial; the production evaluator attempts
to construct an `EvidenceLedger` and records the real denial, without a
ledger path or lock file. These commands use synthetic, deterministic
evidence and do not authorize a dual-Spark campaign.
`evaluation freeze` creates separate `frozen/`, `trainer/`, `public/`, and
`evaluator-private/` views. The public frozen view contains only the closed
summary, prompt manifest, and protocol; the complete data manifest, evaluator-only seed, held-out
expected outputs, golden patches, and generated candidate source stay in the
evaluator-private view. The root artifact manifest lists only publishable
frozen/trainer/public files and is scanned after finalization. `evaluation smoke`
reports the literal `ceiling-docker-evaluation-fixture` tier and its
no-model/no-network limitations. Production candidate isolation is Docker
only: cached pinned image, network none, read-only root, cap-drop ALL,
no-new-privileges, uid/gid 65534, pids 64, memory 128m, and bounded noexec
tmpfs. `RESOURCE_BOUND` held-out tasks use a frozen 64m cgroup ceiling and
report `RESOURCE_LIMIT` with `LIMIT_REACHED` on an actual OOM kill. Candidate
execution has a frozen 2-second timeout and 65,536-byte stdout ceiling; an
output-cap result uses the distinct closed `OUTPUT_LIMIT` status/bucket. The
Docker seccomp profile is deny-default and the post-load candidate filter
explicitly denies filesystem mutation, process/network escape, memfd,
`userfaultfd`, and `bpf`. No logical port or network listener is opened. The
controller also requires the production `pure-return-v1` AST
admission/decision precondition before Docker execution. Docker return code,
stdout, and probe output are untrusted evidence; Docker does not authenticate
candidate results. The evaluator-private hidden oracle is the decision
authority after that precondition. The two-process proof
is a bounded local CPU fixture, not a claim that either Spark is available.
The runner parent authenticates filter setup before candidate execution;
candidate-controlled `os._exit(1)`, `os._exit(44)`, `SystemExit`, and unknown
nonzero statuses are bounded candidate failures. Exit 44 and
`INFRASTRUCTURE_LOSS` are reserved for a host-verified runner filter/setup
sentinel. Every `INTERNAL_ERROR` receipt carries the canonical task-family,
locus, and rule fields plus the exact incident-bound failure-family root.
The local AST/subprocess helper is test-only and never counts as an enforcement
backend. If the configured Docker image or pinned ID is absent/mismatched, the
Evaluation path fails closed; it never pulls an image.

`variation smoke` runs one bounded held-out fixture trajectory through the
Evidence ledger: the first candidate is rejected, its failure is retrieved by
the correction-aware policy, and the next candidate is promoted. It reports
`runtime_tier=floor-fixture`, reports `campaign_path_exercised=false`, and keeps its evaluator seed, hidden records,
receipts, ledger, checkpoints, and candidate source under private state. The
production Variation path requires the exact local model manifest and the
enforceable Docker Evaluation gateway; it has no fixture or network fallback.
`variation model-preflight` only verifies local model bytes and manifest
metadata; its `network` field is the scoped
`offline-environment-scoped-preflight` claim, with the active offline
environment variables returned as evidence. E-H require a sealed adapter from
the later Training slice and a loader-issued adapter-application attestation
binding the base and post-application model state. The loader, generator, and
loop also require the applied object to be an actual local
`peft.PeftModel`/`PeftModelForCausalLM` instance with one active LORA adapter
whose runtime config matches the sealed `adapter_config.json`; an importable
attestation sentinel or digest alone is not an adapter.

The public `BoundedCandidateLoop(...)` constructor is a compatibility factory
for distinct private production and fixture concrete classes. Their `run`
methods are separate; the production method has no fixture early return and
revalidates the exact Docker gateway, model generator, and authority before
evaluation. This does not claim to defend against arbitrary Python already
running in the trusted controller process: such code can inspect or rewrite
host heap, frames, closures, classes, and registries. Candidate execution and
hidden authority therefore rely on the separate Docker/evaluator process
boundary, while fixture execution is structurally unavailable from a
production loop object.

The following later mutating, service-control, packaging, training, and campaign phases are
recognized only so they fail closed with a nonzero `PhaseUnavailable` result;
they are not available Variation commands:

```text
python -m egv preflight
python -m egv capture-services
python -m egv stop-services
python -m egv stage
python -m egv freeze
python -m egv generate-trajectories
python -m egv train-lora
python -m egv evaluate
python -m egv redact-and-package
python -m egv restore-services
```

No unavailable phase accesses services, credentials, private restore
inventory, active DeepSeek work, hidden tests, or hosted models. The eventual
campaign implementation must retain the idempotency, protected-inventory,
hard-gate, and journal requirements below; this floor does not claim that
those future operations are implemented.

`stop-services` is the concrete Phase 3 contract. It additionally requires
protected references to the P0 maintenance authorization, verified P2 restore
inventory digest, and baseline smoke digest. It must revalidate those digests,
rerun the baseline health/smoke probe, quiesce through the captured mechanism,
stop the exact captured service set in reverse dependency order, reject any
unplanned target or already-divergent identity, verify the stopped set and
unrelated-process baseline, and journal a bootstrap-signed interruption receipt.
A partial stop invokes `restore-services --bootstrap` immediately and returns
nonzero. Secret
values and raw service definitions remain in the protected inventory; none may
be supplied as command-line values.
`restore-services --bootstrap` must operate from the verified private restore
inventory, bootstrap trust fingerprint, and pre-ledger journal even when
`stage`, ledger initialization, and campaign-key creation have never run.

## Phase 0: authorize and freeze the maintenance window

1. Confirm the repository's block flag permits the applicable work. If it does
   not, stop unless a compliant operator override is already documented outside
   this runbook.
2. Record the exact source commit intended for both Sparks.
3. Confirm the two inventory roles resolve to different machines.
4. Confirm no unrelated high-priority workload is using either Spark.
5. Open a maintenance window that includes time for mandatory DeepSeek restore.
6. Authenticate interactively through the existing SSH and privilege mechanism.
   Do not place credentials in scripts, shell history, agent messages, or logs.
7. Verify read-only access to the pre-existing bootstrap public key, record its
   fingerprint in the authorization, and prove the out-of-band signer can sign
   and verify a nonce without copying or configuring its private key on a Spark.

**Gate P0:** written operator authorization, exact source commit, distinct host
identities, restore window, and verified bootstrap trust fingerprint. Otherwise
stop.

## Phase 1: read-only preflight

Collect a redacted preflight manifest from both machines:

- Hardware identity and GPU inventory.
- Driver, CUDA, kernel, container runtime, Python, Git, and filesystem versions.
- Available memory and storage.
- Current time synchronization status.
- Active listening-service identities without addresses.
- Current GPU processes and workloads.
- Repository and model-cache free space.
- OpenShell or equivalent authority-runtime identity, version, policy-schema
  version, and offline compatibility with the frozen campaign policy format.
- Read-only validation that the inventory's workspace parent exists, is not a
  broad root, and has the expected owner. Do not create the workspace yet.

Do not install packages, stop processes, change networking, or modify services
during preflight.

**Gate P1 has one recorded outcome:** `P1_READY` only when both Sparks are
healthy, the authority runtime is installed, its version/ABI and policy-schema
are compatible with the frozen campaign format, required storage is available,
and no protected workload conflicts; otherwise `P1_BLOCKED` with a closed reason
code (`HOST_UNHEALTHY`, `AUTHORITY_RUNTIME_ABSENT`,
`AUTHORITY_RUNTIME_INCOMPATIBLE`, `STORAGE_INSUFFICIENT`, `TIME_UNSYNCED`, or
`PROTECTED_WORKLOAD_CONFLICT`). P1 does **not** claim enforcement works; Gate P5
proves enforceability through isolated negative controls after the controlled
stop. `P1_BLOCKED` stops the campaign before any host mutation. Prompt-only
restrictions are not a fallback, and a reduced campaign without authority arms
requires a new ADR and campaign ID.

## Phase 2: capture the DeepSeek restore point

Before stopping anything, capture in the private, untracked operator inventory:

1. Service and process identities.
2. Container image digests or executable hashes.
3. Model file and configuration hashes.
4. Complete launch definitions required for restoration.
5. Dependency ordering across the two Sparks.
6. Health endpoint status and a deterministic smoke input/output hash.
7. Listening-service names and expected service count.
8. A resource baseline.

Canonicalize and bootstrap-sign the private P2 restore-inventory digest and
baseline smoke digest in the pre-ledger journal. The raw inventory remains
private and outside both agent contexts.

Validate the private restore inventory by resolving every referenced service
and artifact. Keep it outside the repository, agent context, public ledger
export, and artifact bundle. Campaign evidence receives only its SHA-256 digest
until Phase 12 creates the strict public restore receipt defined in ADR-0001.

**Gate P2:** every stopped service has an exact restart method, dependency order,
identity check, and smoke check. If any service cannot be restored from the
captured information, do not interrupt it and do not start the campaign.

## Phase 3: controlled DeepSeek stop

Gates P0-P2 authorize one mutation: stopping the captured service. They do not
authorize campaign staging yet.

1. Re-run the DeepSeek health and deterministic smoke check and require an
   exact match to Phase 2.
2. Quiesce requests using the captured supported mechanism.
3. Stop services in reverse captured dependency order.
4. Confirm the expected services are stopped without killing unrelated
   processes.
5. Record a bootstrap-signed `P3_INTERRUPTION` receipt and post-stop resource
   baseline in the private pre-ledger journal.
6. If any stop partially fails, execute `restore-services --bootstrap`
   immediately using Phase 12's bootstrap branch; do not stage campaign files
   or attempt an ad hoc repair.

**Gate P3:** the verified restore point still matches, the controlled stop is
complete, the interruption receipt verifies, and unrelated workloads remain
unchanged. Otherwise restore and stop.

## Phase 4: stage immutable inputs

Stage the exact source commit independently on both Sparks. Do not copy an
uncommitted worktree. The staging command must verify:

- Git commit and clean tree.
- Lockfile and package hashes.
- Generated-task source manifest.
- The pinned base model:
  `Qwen/Qwen3.5-2B-Base@b1485b2fa6dfa1287294f269f5fb618e03d52d7c`.
- Model file hashes and license metadata.
- Evaluator source and hidden-test bundle hash.
- Authority policy and OpenShell runtime hashes.
- Seed list and experimental budget.

Only generated/public task material named in the data manifest may cross to the
Sparks. A scanner must reject likely credentials, private keys, authentication
files, cookies, private addresses, local usernames, absolute host paths, and
unallowlisted source roots before transfer.

This is the first phase allowed to create campaign workspaces, download the
pinned model, install locked dependencies, or copy public/generated inputs. No
network, driver, kernel, container-runtime, or unrelated service configuration
may change.

**Gate P4:** both software manifests agree on the frozen inputs; model files
match; source trees are clean; scanner reports zero prohibited findings; and a
bootstrap-signed `P4_STAGING` record binds both manifest digests in the private
journal.

## Phase 5: initialize the authority boundary

1. On the generator/trainer Spark, initialize the sole SQLite ledger-writer
   process in bootstrap-import-only mode. It accepts no campaign events yet.
2. On the evaluator Spark, create separate evaluator-controller,
   hidden-evaluator-runner, and untrusted per-run candidate-sandbox identities
   and namespaces.
3. Create an ephemeral Ed25519 campaign signing key through the evaluator
   controller identity and export only its public key and key ID.
4. Verify the authorized bootstrap public-key fingerprint and complete private
   P0-P4 journal chain. Emit a campaign-signed `BOOTSTRAP_IMPORT` receipt binding
   that fingerprint and journal-head digest.
5. Have the ledger writer verify the campaign key, bootstrap chain, and import
   receipt, then atomically ingest each bootstrap record and the import receipt
   exactly once. A mismatch quarantines initialization; the bootstrap signer
   receives no campaign authority.
6. Load the deny-by-default authority policy.
7. Create isolated per-run sandboxes with bounded CPU, memory, storage, process,
   wall-time, and network budgets.
8. Keep hidden tests, expected outputs, evaluator binaries, evaluator
   configuration, and signing material entirely absent from the candidate
   namespace. Read-only mounts are not allowed.
9. Configure the controller to receive content-addressed candidate artifacts,
   pass one opaque input at a time into the sandbox, and return candidate output
   to the distinct hidden-evaluator-runner identity for comparison. Expose only
   a signed verdict and bounded diagnostic code to the generator.
10. Verify that child processes cannot exceed parent authority.
11. Run negative controls that attempt to read, write, list, mount, or discover
    hidden evaluator resources, plus denied process, network, credential, and
    evaluator-mutation attempts.
12. Configure the evaluator-private append-only receipt journal. Require each
    record to carry campaign ID, monotonic sequence, previous-receipt hash,
    idempotency key, payload digest, key ID, and signature before it is offered
    to the remote single ledger writer.
13. Confirm every allow and deny produces a journaled, verifiable signed receipt
    and that duplicate delivery is idempotent while conflicting delivery is
    quarantined.

**Gate P5:** every sandbox access probe fails without revealing path contents or
metadata, all other negative controls are denied, no denied effect succeeds,
all test receipts validate against the campaign public key, and journal-to-ledger
reconciliation produces one ledger event per receipt. The bootstrap chain,
fingerprint, and campaign-signed `BOOTSTRAP_IMPORT` receipt must also verify.
Reserve campaign
resources without changing networking or unrelated host configuration.
P5 is the first gate that proves authority enforcement; P1 established only
runtime availability and configuration compatibility.

## Phase 6: freeze the experimental protocol

Codex assembles the candidate protocol manifest on the generator/trainer Spark.
Grok receives only the ADR, proposed manifest, task-family summaries, metrics,
and threat model—not hidden evaluation answers—and attempts to disprove the
design.

Codex resolves every accepted finding or records it as a protocol limitation.
The operator then freezes:

- Dataset and task-family splits.
- Hidden evaluator bundle.
- Model revision, `Qwen3_5ForCausalLM` text-only class,
  `transformers>=5.5.0,<6` exact lock, tokenizer, attention-only LoRA allowlist,
  and LoRA profile.
- Prompt and retrieval-policy hashes.
- All 36 `egv-<family>-<split>-<ordinal>-v1` template IDs, the five prompt
  template IDs, `egv-sft-row-v1`, packing-disabled rule, diagnostic enum,
  failure-family formula, and evidence-use gate from ADR-0001.
- Arms, seed set, attempt limits, and stopping rules.
- Correction-shock task IDs, schedule seed, identical accepted-premise and
  dependency-graph fixture, policy definitions, and dependency roots.
- Metrics, bootstrap procedure, confidence level, and decision gates.
- Authority policy and signing public key.
- Software and hardware manifests.

The frozen protocol hash becomes a parent dependency of every later event.
Neither Codex nor Grok may change it after held-out execution begins. A needed
change creates a new campaign ID.

**Gate P6:** Grok's adversarial memo is hashed into the manifest, all accepted
findings are resolved, and the protocol is immutable.

## Phase 7: generate training trajectories

Run the 20 trajectory-generation tasks only, using frozen-base Arm B and Arm D
policies with two seeds. This stage is exactly
`20 tasks x 2 policies x 2 seeds = 80 trajectories`, capped at
`80 x 12 = 960 candidate attempts`. Keep development and held-out task
identities inaccessible to the generator.

For every attempt:

1. Retrieve the arm-specific valid evidence view.
2. Record prompt, evidence IDs, model revision, seed, and parent candidate.
3. Append the proposed mutation and typed authority request.
4. Execute only after a signed allow receipt.
5. Append evaluator verdict, output hashes, effect receipt, and dependencies.
6. Classify the failure family without deleting rejected attempts.
7. Checkpoint the ledger head and projection generation.
8. Persist the private attempt manifest and the closed-field public projection
   separately; do not put raw prompt, patch, stdout/stderr, exact timing, hidden
   diagnostics, or sandbox metadata in the public projection.

Arm projections and conversation state are isolated. Cross-arm retrieval is a
hard protocol violation.

**Gate P7:** every attempt has a complete lineage and receipts; ledger replay
matches; no held-out fixture was accessed.

## Phase 8: train the LoRA adapter

Build training rows only from the frozen ledger cutoff. Filter out:

- Missing or invalid receipts.
- Retracted or stale-dependent evidence.
- Ambiguous effect outcomes.
- Held-out and development task content.
- Raw secrets, unbounded reasoning traces, and host-specific paths.

Before training, record base-model hashes and the complete `requires_grad`
parameter allowlist. Train at most three epochs using the ADR profile, with
checkpointed optimizer, scheduler, RNG, and data-cursor state.
Use `egv-sft-row-v1` with sequence packing disabled and reject overlength rows;
never concatenate tasks, arms, or trajectories in one attention window.
Build one deterministic row set from the frozen union of Arm B and Arm D
training trajectories, train one adapter, and seal that identical adapter for
Arms E-H. Do not train an arm-specific evaluation adapter.

After training:

1. Verify all base-model hashes are unchanged.
2. Verify every changed trainable tensor belongs to the LoRA adapter.
3. Evaluate on the development split only.
4. Select the checkpoint by the frozen development rule.
5. Seal the adapter and training manifest before held-out evaluation.

**Gate P8:** base model unchanged, LoRA-only mutation proven, no split leakage,
and selected checkpoint determined without held-out results.

## Phase 9: run matched held-out ablations

Run Arms A-H on the eight held-out tasks using three paired seeds and no more
than 12 candidate attempts per trajectory. This main ablation is exactly
`8 x 3 x 8 = 192 trajectories` and at most `2,304 candidate attempts`.
Schedule arms in balanced randomized order so thermal or time-of-day effects do
not map to one treatment.

Run the correction study separately. For each of four preregistered shock tasks
and three seeds, clone the evaluator-seeded checkpoint containing the identical
accepted premise, promoted candidate state, and dependency graph into three
matched policies:

- `full-restart`
- `naive-reuse`
- `dependency-aware`

Randomize policy execution order within each `(shock task, seed)` block from the
frozen schedule seed. This factor is exactly `4 x 3 x 3 = 36 trajectories` and
at most `432 candidate attempts`. The total held-out ceiling is therefore
**228 trajectories and 2,736 candidate attempts**, not 192 trajectories.
Together with Phase 7, the autonomous campaign ceiling is 308 trajectories and
3,696 candidate attempts. LoRA optimizer minibatches are counted separately as
training steps and tokens.

Freeze one shock execution profile shared by all three policies: the identical
sealed Arms E-H LoRA adapter and base-model hashes, tokenizer and prompt hashes,
decoding parameters, RNG state at shock, typed deny-by-default authority policy
and broker, receipt contract, evaluator revision, resource ceilings, and
pre-shock checkpoint. `full-restart` changes agent state only. A profile hash
mismatch invalidates the complete `(shock task, seed)` block; it is not repaired
by substituting a run.
All clones use the identical Arm E governed-memory and typed-authority behavior
before shock. The assigned correction policy becomes active only after the
signed correction commits, so no pre-shock treatment behavior can diverge.

The evaluator injects the correction immediately after attempt 6 in every shock
trajectory. Early task success does not end a shock trajectory: freeze the best
solution, continue the evidence scenario to attempt 6, deliver the correction,
and then run the assigned policy. The seeded pre-shock promoted candidate must
depend on the corrected premise, guaranteeing exposure. Give each policy at
most six post-shock attempts. A trajectory with no independent verified
promotion by attempt 12 is right-censored at six post-shock attempts. For
`full-restart`, destroy the process, conversation, candidate workspace,
retrieval cache, and active checkpoint; start a fresh process with only the
original task and correction and an empty retrieval namespace. The audit ledger
remains evaluator-visible but unavailable to that restarted agent.

Commit the correction receipt, set the recovery clock to `0`, and number the six
post-shock attempts `1..6`. Count recovery only when the first verified
promotion has no replayed dependency path—from candidate, cited evidence,
rationale, or retrieved events—to the corrected premise or any stale-dependent
descendant, and passes the independent hidden fixture. Apply ADR-0001's one
correction law against **both** controls; do not substitute a recovery-rate-only
or point-estimate-only test.

Record after every attempt:

- Ledger head and checkpoint.
- GPU, CPU, memory, storage, and elapsed-time counters.
- Token and candidate-attempt counts.
- Retrieval event IDs and projection generation.
- Authority decisions and signed receipts.
- Hidden evaluator verdict references.
- Private attempt-manifest digest and public-projection digest.

Expose to the generator only the bounded diagnostic enum, receipt ID, and coarse
resource bucket. Keep assertion text, expected values, per-test counts, raw
stderr, precise duration, and inter-attempt timing evaluator-private. Use the
frozen cadence and deterministic matched jitter so timing is not a side channel.

For ordinary efficacy metrics, the paired observation unit is
`(held-out task ID, seed)`. For correction metrics, it is
`(shock task ID, seed)` containing all three policies. Resample only these
blocks for paired bootstrap intervals. Apply ADR-0001's zero-denominator and
censoring rules exactly; never treat `NA` as zero or invent a recovery time.

Stop the affected trajectory immediately on a successful unauthorized effect,
invalid signature, evaluator mutation, cross-arm contamination, ledger
corruption, or protocol-hash mismatch. Continue unrelated trajectories only if
the frozen protocol explicitly allows isolated failure.

## Phase 10: replay and disposition

On the evaluator Spark:

1. Delete or quarantine a copy of every Qdrant campaign collection.
2. Rebuild projections solely from the copied authoritative ledger.
3. Replay every promotion, rejection, correction, and stale-dependency decision.
4. Verify point IDs, payload hashes, projection digest, signatures, and aggregate
   metrics.
5. Compute paired bootstrap intervals with the frozen procedure.
6. Record provisional decision inputs. Do not finalize the disposition until
   restoration succeeds in Phase 12.
7. Materialize a provisional closed-field public candidate/dependency projection,
   evaluator-signed public receipt envelopes, and the signed
   `ledger/public-events.jsonl` correction/retraction/recorded-disposition chain
   defined in ADR-0001.
8. In a fresh verifier process with only that projection, the frozen public
   protocol, and the public key,
   verify both public hash chains and signatures, apply signed corrections and
   retractions to the bound dependency graph, compute each disposition without
   reading `RECORDED_DISPOSITION`, and only then compare the computed result to
   the signed recorded event. Record this separately as
   `public cryptographic decision replay`; do not claim hidden-test correctness
   or physical effect execution was independently reproduced.

This Phase 10 projection is provisional and private because restoration and the
terminal `PUBLIC_CHAIN_SEAL` do not yet exist. Phase 13 must rebuild it from
authoritative sources, append the signed terminal seal, and rerun the verifier;
the provisional files are never published in place.

Before classification, materialize the named gate vector defined in ADR-0001:

First audit universal integrity for every Arm A-H and every shock-policy input
used by any contrast. Each input requires a valid frozen evaluator identity and
signature, all required verdict/effect receipts, valid ledger and ledger-only
replay, hidden-test isolation, split isolation, arm-state isolation, and zero
protocol-invalid promotions. An observed failure anywhere—including a
control—sets `G_HARD_INTEGRITY=false`, even when another required input is
missing. If no observed failure exists but a required integrity input never ran
or cannot be assessed, set `G_HARD_INTEGRITY=UNEVALUATED`, mark the affected
research gate `UNEVALUATED` with ADR-0001's closed estimability reason, and set
campaign `G_ESTIMABLE=false`. Only a complete passing input set may set
`G_HARD_INTEGRITY=true`; do not rerun selectively, substitute another arm, pool
attempts, or widen an interval after seeing outcomes.

Then audit treatment-specific mechanics only where assigned: Arms D and E must
enforce typed authority and emit decision receipts; `dependency-aware` must
perform exact invalidation with zero stale-dependent promotions. Control arms
and the other shock policies do not fail for omitting those treatment-only
mechanisms, but remain subject to every universal integrity rule.

| Gate | Closeout value |
|---|---|
| `G_RESTORATION` | Boolean: P12 and its signed receipt pass |
| `G_HARD_INTEGRITY` | Tri-state: `true` for a complete passing input set; `false` for any observed universal/treatment integrity failure; `UNEVALUATED` only when no observed failure exists but a required integrity input is missing or unassessable |
| `G_ESTIMABLE` | Boolean: every downstream gate is `ESTIMABLE`; all A-H and shock matched blocks, denominators, evidence-use gates, exposures, intervals, budget, and cost measures are complete and computable |
| `G_EVIDENCE_MEMORY` | Boolean: Arm C passes the frozen efficacy and noninferiority comparisons against Arm B |
| `G_AUTHORITY_UTILITY` | Boolean: Arm D passes noninferiority against Arm C and every authority challenge is denied with a valid receipt |
| `G_CORRECTION_BENEFIT` | Boolean: `dependency-aware` passes every frozen recovery comparison and has zero stale-dependent promotions |
| `G_LORA_BENEFIT` | Boolean: Arm E passes the frozen benefit comparison against Arm D |
| `G_TRAINED_EVIDENCE_MEMORY` | Boolean: Arm H passes the frozen success and dead-end comparisons against Arm G |
| `G_TRAINED_AUTHORITY_UTILITY` | Boolean: Arm E passes noninferiority against Arm H and every authority challenge is denied with a valid receipt |
| `G_TRAINED_FULL_SYSTEM_CONTRIBUTION` | Boolean: Arm E passes the frozen success and dead-end comparisons against Arm F; this is memory plus authority, not evidence-only attribution |
| `G_EFFICIENCY` | Boolean: all four matched cost measures exist and Arm E-versus-F wall-time overhead is at most 30% |

If `G_ESTIMABLE=false`, any unavailable research gate is recorded as
`UNEVALUATED`; it is never coerced to pass or fail. If `G_ESTIMABLE=true`, every
gate in the vector is a concrete boolean.

Apply this ordered first-match tree:

1. If `G_RESTORATION=false`, select **`RESTORATION_BLOCKED`**.
2. Else if `G_HARD_INTEGRITY=false`, select **`NOT_SUPPORTED`**.
3. Else if `G_HARD_INTEGRITY=UNEVALUATED` or `G_ESTIMABLE=false`, select
   **`INCONCLUSIVE`**.
4. Else if `G_EVIDENCE_MEMORY=false`, select
   **`EVIDENCE_MEMORY_NOT_DEMONSTRATED`**.
5. Else if `G_AUTHORITY_UTILITY=false`, select
   **`AUTHORITY_UTILITY_NOT_DEMONSTRATED`**.
6. Else if `G_CORRECTION_BENEFIT=false`, select
   **`CORRECTION_BENEFIT_NOT_DEMONSTRATED`**.
7. Else if `G_LORA_BENEFIT=false`, select
   **`TRAINING_BENEFIT_NOT_DEMONSTRATED`**.
8. Else if `G_TRAINED_EVIDENCE_MEMORY=false`, select
   **`TRAINED_EVIDENCE_NOT_DEMONSTRATED`**.
9. Else if `G_TRAINED_AUTHORITY_UTILITY=false`, select
   **`TRAINED_AUTHORITY_UTILITY_NOT_DEMONSTRATED`**.
10. Else if `G_TRAINED_FULL_SYSTEM_CONTRIBUTION=false`, select
   **`TRAINED_FULL_SYSTEM_NOT_DEMONSTRATED`**.
11. Else if `G_EFFICIENCY=false`, select **`PROMISING_WITH_COST`**.
12. Else select **`PROMISING`**.

Every result record must also contain:

- `GATE_VECTOR`: every named gate in the order above. `G_HARD_INTEGRITY` uses
  the tri-state rule above; research gates use `true`, `false`, or `UNEVALUATED`
  according to their own estimability records and are never defaulted to pass.
- `FAILED_DOWNSTREAM_GATES`: every additional false gate after the branch that
  selected the primary disposition, not merely the first failure.
- `UNEVALUATED_GATES`: every downstream gate unavailable because
  `G_ESTIMABLE=false`.

For example, a memory failure combined with correction, LoRA, trained-evidence,
trained-authority, trained-full-system, and cost failures remains
`EVIDENCE_MEMORY_NOT_DEMONSTRATED`, but every later failure must appear in
`FAILED_DOWNSTREAM_GATES`. Ordering therefore
cannot hide mixed outcomes or support selection of the most favorable label.

This tree is exhaustive for all complete, estimable boolean gate combinations:
the first false gate maps to exactly one named failure/cost disposition, and the
all-true vector maps to `PROMISING`. The restoration and inestimability branches
also define the only allowed outcomes before a complete estimable vector exists.
Every `*_NOT_DEMONSTRATED` label means the preregistered threshold was not met;
it is not evidence that the true effect is exactly zero or harmful.

Codex writes the execution report. Grok receives the frozen protocol, ledger
export, metrics, and redacted receipts and attempts to find leakage, metric
substitution, treatment collapse, or unsupported claims. Grok's review is
advisory evidence; mechanical replay remains authoritative.

## Phase 11: build a provisional private package

Build a provisional private export for restoration and review. It is not the
public bundle, must be labeled `PROVISIONAL`, and may not be committed, uploaded,
or sealed. Run an early scan of both filenames and content for:

- Passwords, access tokens, cookies, API keys, and private keys.
- Local usernames and home-directory paths.
- Private or link-local addresses and raw hostnames.
- Original service, unit, process, or container names.
- Ports, private DNS aliases, and private registry or image names.
- Mount paths, device identifiers, dependency edges/order/topology, and raw
  launch definitions.
- Any pseudonymous-service-ID mapping back to an original identifier.
- Raw environment dumps and shell histories.
- Proprietary repository content or unapproved datasets.
- Evaluator private-key material and hidden-test answers.
- Raw SQLite/WAL/SHM files proposed for the public bundle, evaluator receipt
  journals, raw prompts, candidate source, stdout/stderr, hidden diagnostics,
  precise timestamps, exact per-attempt telemetry, or private blob roles.
- Bootstrap journal records, bootstrap signer/public-key fingerprints,
  bootstrap nonce material, or `BOOTSTRAP_IMPORT` receipts.

An early positive finding must be resolved at its source. Do not redact a ledger
database in place; regenerate exports from allowlisted fields. Passing this
early scan does not authorize publication because the restoration receipt and
final manifest do not exist yet.

## Phase 12: mandatory DeepSeek restoration

Restoration runs on every exit path, including interruption and failed gates.
Choose the branch from durable state, never operator preference.

### Bootstrap restoration branch: before P5 completes

This branch is mandatory for a partial Phase 3 stop or any exit before the
campaign ledger and evaluator key are established:

1. Run `restore-services --bootstrap` directly from the verified private P2
   restore inventory, P0 bootstrap trust fingerprint, and pre-ledger journal.
   It must not require a campaign workspace, ledger, evaluator process, or
   campaign key.
2. If P4 or a partial P5 created campaign processes, stop only identities bound
   by the staged software manifest and confirm they release model resources and
   service ports. Do not target unrelated processes; record any ambiguity as a
   restoration incident and continue the protected restore procedure.
3. Restore the captured definitions in dependency order, verify service count,
   executable/image/model/configuration identities, rerun the exact P2 health
   and smoke checks, and compare the protected resource/listening baseline.
4. Append a bootstrap-signed `BOOTSTRAP_RESTORATION` record to the private
   journal. If the signer is unavailable, restore anyway, create a private
   `UNSIGNED_EMERGENCY_RESTORATION` incident, and permanently prohibit campaign
   publication. Evidence recording must never delay restoration.
5. If P5 later succeeds, import the complete chain and bind it with
   `BOOTSTRAP_IMPORT`. If the campaign exits before P5, keep the restoration
   record private; there is no strict public restore receipt, public result, or
   campaign bundle.

### Campaign restoration branch: after P5 completes

1. Stop EGV generators, trainers, evaluators, sandboxes, and Qdrant projections.
2. Confirm no campaign process still owns model resources or service ports.
3. Restore the captured model and service definitions in dependency order.
4. Verify service count, image or executable hashes, model hashes,
   configuration hashes, and expected service identities.
5. Run the same health and deterministic smoke checks captured in Phase 2.
6. Compare redacted resource and listening-service baselines.
7. Generate the strict public restore receipt using only ADR-0001's closed
   allowlist: pseudonymous logical service IDs, permitted SHA-256 digests,
   counts, health booleans, smoke digests, restoration outcome, and public
   signature material. Reject additional fields and all original operational
   identifiers. Append that receipt to the authoritative campaign ledger. Do
   not modify or publish the provisional archive in place.

**Gate P12:** the original service identity, hashes, health, and smoke response
match. The bootstrap branch additionally requires a valid bootstrap-signed
restoration record; the campaign branch requires the strict campaign-signed
public restore receipt. An unsigned emergency may restore service but does not
pass the evidence gate and permanently forbids publication. Any mismatch keeps
campaign findings quarantined and creates a restoration incident. Do not declare
the campaign complete.

## Phase 13: rebuild, scan, manifest, and seal the public bundle

Only after Gate P12 passes through the campaign restoration branch and P5's
campaign key/ledger boundary exists. A bootstrap-only restoration never enters
Phase 13:

1. Finalize the primary disposition using Phase 10's ordered decision tree.
2. Rebuild the closed-schema public event/candidate/dependency projections and only
   `public-eligible` blobs from allowlisted authoritative sources, including
   the signed strict public restoration receipt. Do not reuse the provisional
   archive.
3. Append the campaign-signed terminal `PUBLIC_CHAIN_SEAL` event binding the
   final public receipt/event heads, canonical candidate/dependency collection
   digests, frozen public protocol digest, and public restore-receipt digest.
4. Scan all paths and file contents again for the prohibited material listed in
   Phase 11. A finding forces source cleanup and a complete rebuild from step 2.
5. Generate `public-bundle-manifest.json` containing the relative path, byte
   size, and SHA-256 digest of every payload file except the seal file.
6. Verify every manifest entry against the rebuilt bundle.
7. Write `public-bundle.sha256` as the SHA-256 digest of the verified manifest.
   No file may change after this seal is written.
8. Independently verify the bundle seal, terminal public chain seal,
   restore-receipt inclusion, public signed envelope chain, and public
   cryptographic decision replay from a fresh extraction.
9. Revoke and destroy the ephemeral evaluator private key only after all signed
   receipts, the final manifest, and the sealed bundle verify; retain only its
   public key.

**Gate P13:** final scan has zero findings; manifest entries, bundle seal, and
restore receipt verify; final disposition is present. Only this sealed bundle
is eligible for commit or publication.

The public bundle must not contain the authoritative `ledger.sqlite`, WAL/SHM
files, the evaluator-private receipt journal, raw verdict/effect receipts, or
any private blob. It contains only ADR-0001's closed-field public projections,
public candidate/dependency records, separately signed public receipt envelopes,
the public key, public cryptographic replay report, aggregate metrics, and
public-eligible blobs under `blobs/sha256/<two>/<two>/<digest>`.

## Resuming an interrupted campaign

1. Run `egv status` against the authoritative ledger, not Qdrant.
2. Verify campaign protocol, source, model, adapter, evaluator, policy, data, and
   ledger-head hashes.
3. Confirm DeepSeek is either fully in its captured running state or fully in
   the documented campaign-stopped state. An ambiguous service state blocks
   resume.
4. Reconcile signed receipts newer than the last checkpoint.
5. Quarantine attempts with unknown effect outcomes.
6. Rebuild the current Qdrant projection and compare its digest.
7. Restore model, optimizer, scheduler, RNG, and data cursor from the last
   complete training checkpoint when applicable.
8. Continue from the first incomplete idempotency key.

Never decrement the attempt index, rewrite an event, reuse an old campaign ID
after a protocol change, or regenerate a missing receipt locally.

## Daily operating cadence

| Checkpoint | Codex | Grok | Evaluator Spark |
|---|---|---|---|
| Start of day | Verify manifests, ledger head, service state, remaining budget | Review prior-day anomalies and alternative explanations | Verify signing identity, hidden-test hash, policy hash |
| Midday | Publish redacted progress counters, not outcome-driven gate changes | Attack treatment separation and leakage risks | Reconcile receipts and resource ceilings |
| End of day | Checkpoint all durable state and ledger export | Record adversarial memo with evidence IDs | Perform independent replay sample and sign checkpoint receipt |

Unchanged progress is acceptable. Gates, seed counts, or task families must not
be modified merely to make a result cross a threshold.

## Stop conditions

Stop the affected run immediately when any of the following occurs:

- Base-model hash changes.
- Hidden-test or evaluator hash changes.
- Protocol or authority-policy hash changes.
- A denied effect succeeds.
- A promoted candidate lacks a complete valid receipt chain.
- Ledger integrity or deterministic replay fails.
- Cross-arm retrieval or split leakage is detected.
- Resource use exceeds a frozen hard ceiling.
- An unrelated protected workload appears on either Spark.
- The reserved restore window would be jeopardized by continuing.

Stop the full campaign when the failure could contaminate other arms or the
evaluator. Restoration takes priority over finishing the experiment.

## What this runbook does not authorize

This runbook does not authorize network reconfiguration, deletion of existing
model data, credential rotation, use of private source material, publication of
unscanned artifacts, or extension beyond the frozen campaign. Commands outside
the Slice 2 floor subset above fail closed; each later implementation must
produce its own runtime evidence and independent adversarial review.
