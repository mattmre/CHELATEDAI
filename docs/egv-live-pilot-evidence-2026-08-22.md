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

## Public boundary

This document intentionally excludes machine identities, addresses, operator
paths, credentials, keys, prompts, raw model output, private rows, hidden
inputs, service inventory, and operational logs. The listed digests identify
private evidence without publishing it.
