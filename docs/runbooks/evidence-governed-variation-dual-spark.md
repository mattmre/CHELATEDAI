# Evidence-Governed Variation: Dual-Spark Execution Runbook

## Purpose and current status

This runbook defines how operators will stage, run, resume, and close the
Evidence-Governed Variation (EGV) campaign described in
[ADR-0001](../architecture/adr-0001-evidence-governed-variation-agent.md).

> **Important:** This is an activation plan for future implementation PRs. The
> `egv` commands named below are the required command-line contract; they do not
> exist in this documentation-only PR. Do not interpret this document as
> runtime evidence.

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

Inventory validation must reject empty values, loopback destinations,
trainer/evaluator aliasing, broad filesystem roots, and values containing
embedded credentials. The automation accepts inventory by file descriptor or
protected local file, never a committed file or command-line secret.

## Required implementation command contract

Later PRs must provide one top-level command surface with these subcommands:

```text
python -m egv preflight
python -m egv capture-services
python -m egv stage
python -m egv freeze
python -m egv generate-trajectories
python -m egv train-lora
python -m egv evaluate
python -m egv replay
python -m egv redact-and-package
python -m egv restore-services
python -m egv status
```

Every subcommand must be idempotent, accept a campaign ID and protected
inventory reference, write structured results to the authoritative ledger, and
return nonzero on an unmet hard gate. No command may print credentials or dump
the complete environment.

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

**Gate P0:** written operator authorization, exact source commit, distinct host
identities, and a restore window. Otherwise stop.

## Phase 1: read-only preflight

Collect a redacted preflight manifest from both machines:

- Hardware identity and GPU inventory.
- Driver, CUDA, kernel, container runtime, Python, Git, and filesystem versions.
- Available memory and storage.
- Current time synchronization status.
- Active listening-service identities without addresses.
- Current GPU processes and workloads.
- Repository and model-cache free space.
- OpenShell or equivalent authority-runtime identity and version.
- Ability to create an isolated campaign workspace under the exact inventory
  path.

Do not install packages, stop processes, change networking, or modify services
during preflight.

**Gate P1:** both Sparks are healthy; the authority runtime is present and
enforceable; required storage is available; no protected workload conflicts.
If OpenShell or an equivalent below-harness enforcement layer is absent, the
full authority arm is blocked. Prompt-only restrictions are not a fallback.

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

Validate the private restore inventory by resolving every referenced service
and artifact. Keep it outside the repository, agent context, public ledger
export, and artifact bundle. Campaign evidence receives only its SHA-256 digest
until Phase 12 creates the strict public restore receipt defined in ADR-0001.

**Gate P2:** every stopped service has an exact restart method, dependency order,
identity check, and smoke check. If any service cannot be restored from the
captured information, do not interrupt it and do not start the campaign.

## Phase 3: stage immutable inputs

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

Create distinct Unix identities or runtime identities for generator and
evaluator duties. The generator and candidate-sandbox identities must have no
read, write, directory-listing, mount, or discovery access to hidden tests,
evaluator binaries, expected outputs, or the evaluator signing key.

**Gate P3:** both software manifests agree on the frozen inputs; model files
match; source trees are clean; scanner reports zero prohibited findings.

## Phase 4: initialize the authority boundary

On the evaluator Spark:

1. Create separate evaluator-controller, hidden-evaluator-runner, and untrusted
   per-run candidate-sandbox identities and namespaces.
2. Create an ephemeral Ed25519 campaign signing key through the evaluator
   controller identity.
3. Export only the public key and key ID to the campaign manifest.
4. Load the deny-by-default authority policy.
5. Create isolated per-run sandboxes with bounded CPU, memory, storage, process,
   wall-time, and network budgets.
6. Keep hidden tests, expected outputs, evaluator binaries, evaluator
   configuration, and signing material entirely absent from the candidate
   namespace. Read-only mounts are not allowed.
7. Configure the controller to receive content-addressed candidate artifacts,
   pass one opaque input at a time into the sandbox, and return candidate output
   to the distinct hidden-evaluator-runner identity for comparison. Expose only
   a signed verdict and bounded diagnostic code to the generator.
8. Verify that child processes cannot exceed parent authority.
9. Run negative controls that attempt to read, write, list, mount, or discover
   hidden evaluator resources, plus denied process, network, credential, and
   evaluator-mutation attempts.
10. Confirm every allow and deny produces a verifiable signed receipt.

**Gate P4:** every sandbox access probe fails without revealing path contents or
metadata, all other negative controls are denied, no denied effect succeeds,
and all test receipts validate against the campaign public key.

## Phase 5: capture, stop, and reserve the current model service

Only after gates P0-P4 pass:

1. Re-run the DeepSeek smoke check and confirm it matches Phase 2.
2. Quiesce requests using the service's supported mechanism.
3. Stop services in reverse dependency order.
4. Confirm the expected services are stopped without killing unrelated
   processes.
5. Record a signed interruption receipt and resource baseline.
6. Reserve the campaign resources; do not change networking or unrelated host
   configuration.

If any stop operation partially fails, immediately enter the restoration phase.

## Phase 6: freeze the experimental protocol

Codex assembles the candidate protocol manifest on the generator/trainer Spark.
Grok receives only the ADR, proposed manifest, task-family summaries, metrics,
and threat model—not hidden evaluation answers—and attempts to disprove the
design.

Codex resolves every accepted finding or records it as a protocol limitation.
The operator then freezes:

- Dataset and task-family splits.
- Hidden evaluator bundle.
- Model and LoRA profile.
- Prompt and retrieval-policy hashes.
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

After training:

1. Verify all base-model hashes are unchanged.
2. Verify every changed trainable tensor belongs to the LoRA adapter.
3. Evaluate on the development split only.
4. Select the checkpoint by the frozen development rule.
5. Seal the adapter and training manifest before held-out evaluation.

**Gate P8:** base model unchanged, LoRA-only mutation proven, no split leakage,
and selected checkpoint determined without held-out results.

## Phase 9: run matched held-out ablations

Run Arms A-F on the eight held-out tasks using three paired seeds and no more
than 12 candidate attempts per trajectory. This main ablation is exactly
`8 x 3 x 6 = 144 trajectories` and at most `1,728 candidate attempts`.
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
**180 trajectories and 2,160 candidate attempts**, not 144 trajectories.
Together with Phase 7, the autonomous campaign ceiling is 260 trajectories and
3,120 candidate attempts. LoRA optimizer minibatches are counted separately as
training steps and tokens.

The evaluator injects the correction immediately after attempt 6 in every shock
trajectory. Early task success does not end a shock trajectory: freeze the best
solution, continue the evidence scenario to attempt 6, deliver the correction,
and then run the assigned policy. The seeded pre-shock promoted candidate must
depend on the corrected premise, guaranteeing exposure. Give each policy at
most six post-shock attempts. A trajectory with no independent verified
promotion by attempt 12 is right-censored at six post-shock attempts.

Record after every attempt:

- Ledger head and checkpoint.
- GPU, CPU, memory, storage, and elapsed-time counters.
- Token and candidate-attempt counts.
- Retrieval event IDs and projection generation.
- Authority decisions and signed receipts.
- Hidden evaluator verdict references.

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

Before classification, materialize the named gate vector defined in ADR-0001:

First audit universal integrity for every Arm A-F and every shock-policy input
used by any contrast. Each input requires a valid frozen evaluator identity and
signature, all required verdict/effect receipts, valid ledger and ledger-only
replay, hidden-test isolation, split isolation, arm-state isolation, and zero
protocol-invalid promotions. An observed failure anywhere—including a
control—sets `G_HARD_INTEGRITY=false`. If a required input never ran or cannot
be assessed, set `G_ESTIMABLE=false` for its contrast instead of treating
integrity as passed.

Then audit treatment-specific mechanics only where assigned: Arms D and E must
enforce typed authority and emit decision receipts; `dependency-aware` must
perform exact invalidation with zero stale-dependent promotions. Control arms
and the other shock policies do not fail for omitting those treatment-only
mechanisms, but remain subject to every universal integrity rule.

| Gate | Closeout value |
|---|---|
| `G_RESTORATION` | Boolean: P12 and its signed receipt pass |
| `G_HARD_INTEGRITY` | Boolean: every comparison input passes universal integrity and each applicable treatment passes its assigned typed-authority or invalidation contract |
| `G_ESTIMABLE` | Boolean: required matched blocks, denominators, exposures, intervals, budget, and cost measures are complete and computable |
| `G_EVIDENCE_MEMORY` | Boolean: Arm C passes the frozen efficacy and noninferiority comparisons against Arm B |
| `G_AUTHORITY_UTILITY` | Boolean: Arm D passes noninferiority against Arm C and every authority challenge is denied with a valid receipt |
| `G_CORRECTION_BENEFIT` | Boolean: `dependency-aware` passes every frozen recovery comparison and has zero stale-dependent promotions |
| `G_LORA_BENEFIT` | Boolean: Arm E passes the frozen benefit comparison against Arm D |
| `G_TRAINED_FULL_SYSTEM_CONTRIBUTION` | Boolean: Arm E passes the frozen success and dead-end comparisons against Arm F; this is memory plus authority, not evidence-only attribution |
| `G_EFFICIENCY` | Boolean: all four matched cost measures exist and Arm E-versus-F wall-time overhead is at most 30% |

If `G_ESTIMABLE=false`, any unavailable research gate is recorded as
`UNEVALUATED`; it is never coerced to pass or fail. If `G_ESTIMABLE=true`, every
gate in the vector is a concrete boolean.

Apply this ordered first-match tree:

1. If `G_RESTORATION=false`, select **`RESTORATION_BLOCKED`**.
2. Else if `G_HARD_INTEGRITY=false`, select **`NOT_SUPPORTED`**.
3. Else if `G_ESTIMABLE=false`, select **`INCONCLUSIVE`**.
4. Else if `G_EVIDENCE_MEMORY=false`, select
   **`REDUCES_TO_ORDINARY_MEMORY`**.
5. Else if `G_AUTHORITY_UTILITY=false`, select
   **`AUTHORITY_UTILITY_LOSS`**.
6. Else if `G_CORRECTION_BENEFIT=false`, select
   **`NO_CORRECTION_BENEFIT`**.
7. Else if `G_LORA_BENEFIT=false`, select **`NO_TRAINING_BENEFIT`**.
8. Else if `G_TRAINED_FULL_SYSTEM_CONTRIBUTION=false`, select
   **`NO_TRAINED_FULL_SYSTEM_CONTRIBUTION`**.
9. Else if `G_EFFICIENCY=false`, select **`PROMISING_WITH_COST`**.
10. Else select **`PROMISING`**.

Every result record must also contain:

- `GATE_VECTOR`: every named gate in the order above, using `true`, `false`, or
  `UNEVALUATED` only as permitted by `G_ESTIMABLE`.
- `FAILED_DOWNSTREAM_GATES`: every additional false gate after the branch that
  selected the primary disposition, not merely the first failure.
- `UNEVALUATED_GATES`: every downstream gate unavailable because
  `G_ESTIMABLE=false`.

For example, a memory failure combined with correction, LoRA, trained
full-system, and cost failures remains `REDUCES_TO_ORDINARY_MEMORY`, but all
four later failures must appear in `FAILED_DOWNSTREAM_GATES`. Ordering therefore
cannot hide mixed outcomes or support selection of the most favorable label.

This tree is exhaustive for all complete, estimable boolean gate combinations:
the first false gate maps to exactly one named failure/cost disposition, and the
all-true vector maps to `PROMISING`. The restoration and inestimability branches
also define the only allowed outcomes before a complete estimable vector exists.

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

An early positive finding must be resolved at its source. Do not redact a ledger
database in place; regenerate exports from allowlisted fields. Passing this
early scan does not authorize publication because the restoration receipt and
final manifest do not exist yet.

## Phase 12: mandatory DeepSeek restoration

Restoration runs on every exit path, including interruption and failed gates:

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
match. If they do not, campaign findings remain quarantined and the operator
receives a restoration incident report. Do not declare the campaign complete.

## Phase 13: rebuild, scan, manifest, and seal the public bundle

Only after Gate P12 passes:

1. Finalize the primary disposition using Phase 10's ordered decision tree.
2. Rebuild every public export from allowlisted authoritative sources, including
   the signed strict public restoration receipt. Do not reuse the provisional
   archive.
3. Scan all paths and file contents again for the prohibited material listed in
   Phase 11. A finding forces source cleanup and a complete rebuild from step 2.
4. Generate `public-bundle-manifest.json` containing the relative path, byte
   size, and SHA-256 digest of every payload file except the seal file.
5. Verify every manifest entry against the rebuilt bundle.
6. Write `public-bundle.sha256` as the SHA-256 digest of the verified manifest.
   No file may change after this seal is written.
7. Independently verify the seal and restore-receipt inclusion from a fresh
   extraction.
8. Revoke and destroy the ephemeral evaluator private key only after all signed
   receipts, the final manifest, and the sealed bundle verify; retain only its
   public key.

**Gate P13:** final scan has zero findings; manifest entries, bundle seal, and
restore receipt verify; final disposition is present. Only this sealed bundle
is eligible for commit or publication.

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
unscanned artifacts, or extension beyond the frozen campaign. It does not
assert that the future `egv` commands work. Each implementation PR must produce
its own runtime evidence and independent adversarial review.
