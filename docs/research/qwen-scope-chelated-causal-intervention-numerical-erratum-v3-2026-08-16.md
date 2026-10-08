# Qwen-Scope causal-intervention numerical erratum v3

**Protocol ID:** `CHELATEDAI-QSCCI-v3`

**Status:** frozen before any v3 model execution

**Base scientific protocol:**
`docs/research/qwen-scope-chelated-causal-intervention-preregistration-2026-08-16.md`,
SHA-256 `f7417b022dd93b96d523f6e8ca4a12c8a915288b5b4621e9ef130ac4f52848f1`

**Base artifact schema:**
`docs/research/qwen-scope-chelated-causal-intervention-artifact-schema-v1.json`,
SHA-256 `032f06fefc53aa04f13a410ce5aec93b7039c7c042bd0a2cedfe8f4745cb653a`

**Immutable fixture:**
`docs/research/qwen-scope-chelated-causal-intervention-fixture-v1.json`,
SHA-256 `d9f873d5a0e00d0330b87e5ea053aaf0343d4c1636d7402020490719d1336f83`

## Trigger and evidence boundary

An authorized live diagnostic of the exact v1 worker suppressed scientific
stdout and opaquely destroyed its stage. The only observed child diagnostic
was the exception category/message `direction cosine must be in [-1, 1]`.
No disposition, scientific stdout, artifact, endpoint, feature, direction, or
gate value was inspected or retained. This erratum is justified solely by that
numerical boundary exception.

The infrastructure-only `CHELATEDAI-QSCCI-v2` draft, addendum SHA-256
`3bf01a1b2e140fa1ee57816dca15d204ad7fdc240804d421116261f207e6a467`
and schema SHA-256
`9a1ec63343772955d3f7057beffaded5e64c10ccfaf57b7c36e14d02a65c5c80`,
was never executed and is not an ancestor of v3. V3 inherits v1's original
600-second restoration ceiling.

## Sole numerical erratum

The direction-cosine raw value is still computed in binary64 using the frozen
v1 dot-product and vector-norm formula and frozen accumulation order. Define:

The frozen absolute envelope is exactly binary64 `1e-12`. Binary64 unit
roundoff is `u=2^-53`, about `1.11e-16`. For a 2,048-term reduction,
`gamma_2048 = (2048*u)/(1-2048*u)` is about `2.27e-13`; conservatively allowing
about three such error contributions for the dot product, norms, product, and
final ratio gives about `6.82e-13`. The `1e-12` envelope is a simple decimal
bound above that derived scale and is not fitted to an observed scientific
value.

The raw cosine must be binary64 finite and lie in the inclusive interval
`[-1-1e-12, 1+1e-12]`. A raw value in `[-1-1e-12, -1)` is stored as
exact binary64 `-1.0`. A raw value in `(1, 1+1e-12]` is stored as exact
binary64 `1.0`. Values already in `[-1, 1]` are unchanged. NaN, infinities, or
values beyond the inclusive envelope invalidate the run. Producer, verifier,
and treatment-separation gate use the same canonicalization helper. The
retained cosine is the canonical value, and verifier reconstruction compares
against that canonical value exactly.

## Invariants

Every other v1 requirement remains byte- or formula-binding without
reinterpretation, including fixture text and hashes, prompt IDs and order,
SELECT/REPORT split, labels, estimands, feature selection, controls, seeds,
tuning grid, model and SAE identities, intervention formula, endpoint
reductions, scientific gates and thresholds, status precedence, resource
ceilings, child supervision, original 600-second restoration ceiling,
artifact reconstruction, publication, and claim boundaries.

The v3 artifact schema differs from the v1 schema only in its schema `$id` and
required `protocol_id` constant. V1 and the unexecuted v2 draft remain distinct
protocol identities and must be rejected by the v3 verifier.
