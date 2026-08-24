# EGV live pilot evidence — 2026-08-22

## Scope

This note records the sanitized result of the first controlled, two-node live
pilot for the Evidence-Governed Variation commissioning path. The tested source
was commit `3b79aacb4affc2855d1155cf7b34717f6a49e6ef` with tree
`d04ad8ccc5f47e2dfed25139cd459909ad6885c0`.

The pilot was deliberately limited to one frozen Arm B request. It did not run
Arm D, the full 80-request commissioning matrix, LoRA training, or private
development selection.

## Controls demonstrated

Before the live model attempt, the operator controls demonstrated:

- an exclusive, one-use launch authorization acquired before service stop;
- clean two-rank service shutdown with exit codes `0/0` and no forced exit;
- immutable container-image and exact-source bindings;
- GPU injection through the platform's supported device-request mechanism;
- content-addressed evaluator manifest, key, and command dependencies;
- an unprivileged model process with a private writable home and cache;
- owned-container cleanup and proof that no Qwen GPU process remained;
- mandatory restoration of the pre-existing model service on every terminal
  path; and
- private-output ownership and mode enforcement, followed by an exhaustive
  post-run correction audit.

The final permission audit reduced 19 nonconforming private-file modes to zero
without changing any file contents. The correction receipt digest is
`f7223e73fcf9912d7af187492d0652b2571867292105349eff5615b67b01bea6`.

## Live result

The pinned Qwen model loaded all `320/320` weight shards and performed one
deterministic generation for the exact frozen Arm B request. Infrastructure,
identity/cache, and evaluator-dependency gates all passed.

The generated response did not satisfy the closed candidate-JSON contract.
The run therefore failed closed before evaluator execution. The private ledger
remained internally valid and recorded one campaign, one exact Arm B run, and
two consecutive events, but zero candidates, verdicts, authority receipts,
effect receipts, or quarantines. No signed candidate decision exists and no
promotion claim is made.

The terminal receipt digest is
`278d24917a1f12036f322b418fb24e3fe346c3c00698005ab89160ceffda5eb7`.
The receipt records model reach, owned-container cleanup, absence of a Qwen GPU
process, and successful restoration.

## Restoration result

After the model attempt, both pre-existing service ranks returned with zero
restarts, the expected model was advertised, request queues were `0/0`, and the
protected semantic baseline matched exactly. No Qwen and DeepSeek workload
overlap was observed.

## Evidence-supported conclusion

The live pilot now separates infrastructure readiness from model-output
readiness:

- The bounded two-node control path reached the pinned model and restored the
  original service safely.
- The current deterministic Qwen response is not yet a valid closed candidate
  envelope.
- No evidence supports scaling to Arm D, all 80 requests, or training.

The next bounded test is a private raw-generation diagnostic that records only
the completion in protected storage and publishes shape and digest metadata.
That evidence will determine whether the next change belongs in prompt/schema
construction, constrained decoding, or a narrowly justified parser rule.

## Private raw-generation diagnostic

The bounded diagnostic completed after this note was first published. The
pinned model produced a short structured response that did not reach the token
ceiling. It contained one syntactically valid JSON root followed by a malformed
structured suffix.
The valid root had the wrong closed field set and failed the required type,
locus, authority, evidence, and final proposal gates. The suffix could not be
partitioned into additional JSON roots, and the entire response was not a
Python module.

This rules out fragment extraction as a safe repair. The evidence supports a
prompt/schema-adherence problem: the base model was asked both to generate a
complete Python file and to encode that file inside a governance envelope.
The next canary therefore separates responsibilities. Qwen generates only the
complete Python module; trusted deterministic code supplies and validates the
locus, authority, evidence dependency set, and content digest. The existing
JSON commissioning contract remains unchanged until the source-only canary has
independent evidence and a fully versioned integration path.

The diagnostic terminal receipt digest is
`f1ba143074574fcc595e2f48589049e0e7f69668968ab33833c17f95d8a4bc0b`.
The closed shape receipt digest is
`879b0b9cc8934effe900d1061a0f3e9c8ad450653614dcc826d6381c6ac7382e`.
Raw completion bytes remain private and are not published.

## 2026-08-23 controlled rerun

A fresh set of offline gates preceded the next live attempt. The main bundle
reported `103/103` worker checks and `112/112` lifecycle checks. The probe
bundle reported `26/26` builder checks and `47/47` self-tests. The direct
precheck bundle reported `112/112` checks.

The live precheck returned `GO` within its bounded scope. It recorded `11/11`
custody checks, seven expected model files, two pre-existing service ranks on
two GPUs, and idle request queues. Qwen was absent both before and after the
precheck. A create-only GPU probe exercised device injection without starting
model inference. A separate no-GPU, no-model probe imported the real
Transformers and Qwen symbols. Owned-resource cleanup was confirmed after the
checks.

The live one-shot Qwen attempt emitted one candidate with the internal state
`proposal_valid`. The ledger counters were `generation_count=1`,
`retry_count=0`, `repair_count=0`, and `evaluation_count=0`. The terminal
reason was `GENERATION_VALID_EVAL_UNAUTHORIZED`. The candidate content remains
private; its SHA-256 digest is
`10be95499b335f55a530c4b3214782e0af1fe050d1d812987c7b8079a575a0c4`.

This state demonstrates only that the bounded generation path produced an
envelope accepted by the proposal-validity gate. It is not correctness,
utility, or promotion evidence by itself.

### Independent signed evaluation

The previously commissioned evaluator state was not reused because it was
bound to an earlier source revision. A fresh evaluator identity, empty state,
service manifest, and controller ledger were commissioned for the current
source commit `5fb6e7598a980f204a11d42b692863b3362bec3e` and tree
`a3090e423c08fe3792bfed673b351019f86f27f1`. Only the candidate bytes and
public bindings crossed the evaluator boundary; the prompt, raw completion,
hidden input, expected output, signing key, and private paths did not.

The evaluator completed without infrastructure loss and returned
`disposition=REJECTED`, `diagnostic_enum=WRONG_OUTPUT`, and
`resource_bucket=UNDER_25`. The controller ledger contains one candidate and
the ordered `AUTHORITY`, `VERDICT`, and `EFFECT` receipt chain. Its integrity
check reported six events, three receipts, a valid chain, and final receipt
head
`5dfcf35651d45426ea2118204ca0e08060b8bfe5ca7bcb8c174d347e72dbf9e5`.
The evaluator state recorded one completed operation, one response, and no
pending operation. The sanitized result summary digest is
`7c552907797d0d4714b003840243576d174ebc5e69dbacb40c662fa68fc680d1`.

This is a useful negative finding: the smaller model reached the governed
candidate boundary, but this single candidate did not solve its frozen task.
It was not promoted, retried, repaired, or regenerated after the signed
decision. This canary does not establish aggregate model quality, training
benefit, or campaign-level utility.

### Independent two-node CPU and Qdrant smoke

The same sealed source archive was exercised independently on both compute
nodes. It was derived from the commit and tree above, contained 917 tracked
files, and had SHA-256
`5d48efd1e2f1d0a083fd42fdb2d6a59a12137dcb16d1ebd85bf3faa9b43bf8e3`.
Both nodes used Python 3.12.3 and an isolated dependency environment with the
same freeze digest
`e78f9e8e29ffd7b5819d1d26e1903da8a6b83896908b2aa196ce619f1439aec3`.

On each node, the deterministic two-process CPU fixture passed with trainer
and evaluator exit code zero, a three-receipt chain, no evaluator SQLite
access, rejection of non-receipt evaluator IPC, successful ledger replay, and
an empty projection rebuild queue. Separately, the installed-Qdrant 1.17.1
regressions for deterministic exact replacement and persistent reopen/rebuild
both passed on each node.

These are two independent same-source node smokes. They do not demonstrate
cross-node trainer/evaluator transport. The CPU fixture is synthetic and the
Qdrant checks are projection regressions; neither ran Qwen, a real campaign,
or a production Qdrant service.

### Service restoration

The lifecycle-owned automatic restore failed after successful candidate
generation because its isolated home setting redirected the pre-existing
service to an empty model-cache location. The lifecycle terminal preserved
that failure and required manual restoration. The service was then started
from its normal service environment. Both ranks returned without container
restarts, the model and metrics endpoints returned HTTP 200, and a minimal
OpenAI-compatible chat request succeeded. The candidate evaluation and CPU /
Qdrant smokes ran without requiring another service interruption.

## Public boundary

This document intentionally excludes machine identities, addresses, operator
paths, credentials, keys, prompts, raw model output, private rows, hidden
inputs, service inventory, and operational logs. The listed digests identify
private evidence without publishing it.
