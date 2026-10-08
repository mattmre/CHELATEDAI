# Qwen-Scope causal-intervention infrastructure addendum v2

**Protocol ID:** `CHELATEDAI-QSCCI-v2`

**Status:** frozen before any v2 model execution

**Base scientific protocol:**
`docs/research/qwen-scope-chelated-causal-intervention-preregistration-2026-08-16.md`,
SHA-256 `f7417b022dd93b96d523f6e8ca4a12c8a915288b5b4621e9ef130ac4f52848f1`

**Base artifact schema:**
`docs/research/qwen-scope-chelated-causal-intervention-artifact-schema-v1.json`,
SHA-256 `032f06fefc53aa04f13a410ce5aec93b7039c7c042bd0a2cedfe8f4745cb653a`

**Immutable fixture:**
`docs/research/qwen-scope-chelated-causal-intervention-fixture-v1.json`,
SHA-256 `d9f873d5a0e00d0330b87e5ea053aaf0343d4c1636d7402020490719d1336f83`

## Reason for the addendum

Official v1 run-003 failed closed solely because verification of the restored
DeepSeek service exceeded v1's frozen 600-second supervisor/restoration
ceiling. No v1 result was published and no invalid scientific artifact was
inspected. The attempted v1 evidence bytes remain immutable and governed by
`CHELATEDAI-QSCCI-v1`; this addendum does not relabel, repair, or re-evaluate
that attempt.

The service restoration helper's configured retry/chat envelope is nominally
at most 2,060 seconds: 100 attempts each allow up to five seconds for `curl`
and then sleep 15 seconds, followed by a final chat probe allowing up to 60
seconds. That is not an inner hard bound: command startup and surrounding
Docker, SSH, compose, and log operations do not all have their own mechanical
timeouts. V2 sets the outer supervisor/restoration ceiling to 2,400 seconds as
the enforced experiment boundary, approximately 340 seconds above the nominal
retry/chat envelope. Exceeding that outer boundary still fails the run closed.

## Sole permitted change

In section 7 of the base protocol, replace only this sentence:

> Supervisor/restoration evidence has a separate 10-minute ceiling.

with:

> Supervisor/restoration evidence has a separate 2,400-second ceiling.

The runner constant is therefore exactly `RESTORATION_SECONDS = 2400.0`.

## Scientific invariance

Every other byte-defined or mathematically defined requirement of the base
protocol remains binding without reinterpretation. In particular, v2 changes
none of the fixture text, prompt IDs, labels, order, SELECT/REPORT split,
estimands, formulas, feature selection, controls, seeds, tuning grid, gates,
status precedence, model identity, SAE identity, numerical precision,
scientific resource ceilings, child TERM/KILL ceilings, artifact leaves,
verifier reconstruction, claim boundary, or publication rules.

The v2 artifact schema differs from the v1 schema only in its schema `$id` and
the required `protocol_id` constant. A v1 artifact remains a v1 artifact and
must be rejected by the v2 runner/verifier. A v2 artifact cannot be presented
as evidence for any v1 attempt.
