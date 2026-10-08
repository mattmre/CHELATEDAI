# Qwen-Scope causal-intervention delta-norm erratum v4

**Protocol ID:** `CHELATEDAI-QSCCI-v4`

**Status:** frozen before any v4 model execution

**Immediate base:** `CHELATEDAI-QSCCI-v3`, numerical erratum SHA-256
`2d5375feb8f3b6b4c8da4d8640554d50f83f36fca63f696b8835f65fd3139a30`
and artifact schema SHA-256
`237a3a678c2bf7cafc4c86b2ec22657f67aeab39dd49c2cc1bd687e02d38d9c8`.

**Immutable v1 scientific base:** protocol SHA-256
`f7417b022dd93b96d523f6e8ca4a12c8a915288b5b4621e9ef130ac4f52848f1`,
schema SHA-256
`032f06fefc53aa04f13a410ce5aec93b7039c7c042bd0a2cedfe8f4745cb653a`,
and fixture SHA-256
`d9f873d5a0e00d0330b87e5ea053aaf0343d4c1636d7402020490719d1336f83`.

## Trigger and evidence boundary

V3 run-002 failed closed after the child. The only exposed child diagnostic
was `QSCCIError: retained delta norms do not reconstruct from
scale/direction/alpha`. DeepSeek and the recovery flag, mask, log, and public
output postconditions were independently verified. One failed-run quarantine
remains opaque. No quarantine member, artifact, scientific stdout, endpoint,
feature, direction, norm, gate, or disposition value was inspected.

## Exact invariant mismatch

V3's producer computed `shared_delta_norm` and `cast_delta_norm` with CUDA
`torch.linalg.vector_norm` reductions over CUDA-constructed deltas. Its
verifier reconstructed the same conceptual quantities with CPU tensor
arithmetic and CPU reductions, then required exact binary64 equality. V3 did
not freeze equivalent device reduction trees or intermediate arithmetic.
Bit-for-bit equality across those two reduction paths was therefore not a
valid reconstructable invariant. This erratum does not change the intervention
components or the BF16 hook update; it changes only where their retained norm
evidence is reduced.

## Sole erratum

The producer still constructs the actual per-prompt FP32 delta on the hook
device with v1's frozen expression
`label * alpha * scale * direction`, casts that exact delta to BF16 on the hook
device, and applies that BF16 tensor. After both actual tensors exist, the
producer copies their exact components to detached CPU FP32 and BF16 tensors.

V4 defines one shared CPU reduction helper used by producer and verifier. It
requires rank-one, equal-width FP32 and BF16 component vectors; requires every
component finite; computes `shared_delta_norm` as CPU
`torch.linalg.vector_norm(delta_fp32_cpu)`; computes `cast_delta_norm` as CPU
`torch.linalg.vector_norm(delta_bf16_cpu.float())`; and requires both results
finite and strictly positive.

The producer additionally reconstructs the reference components on CPU from
the retained direction, frozen positive `alpha`, and retained FP32 residual RMS
scale using the same left-associated arithmetic. It requires the copied actual
FP32 and BF16 components to equal that reference exactly before retaining any
norm. A component or cast disagreement invalidates the run. The producer then
reduces each copied actual signed vector with the shared helper and requires
the resulting norms to be invariant under the frozen positive and negative
prompt labels.

The verifier reconstructs the same CPU reference FP32 and BF16 vectors, calls
the same reduction helper, and requires exact equality with every retained
norm. Thus the retained fields describe the actual applied delta rather than a
separate synthetic tensor. `realized_update_norm` remains measured from the
actual BF16 hidden-state update on the hook device and remains
model-execution-attested; it is not replaced by the canonical reduction.

## Invariants

Every other v3 and v1 requirement remains binding, including v3 cosine
canonicalization, fixture and split, selection and controls, tuning, model/SAE
identities, intervention components and hook update, realized-update
measurement, endpoint formulas,
scientific gates, statuses, resources, 600-second restoration ceiling,
artifact publication, and claim boundary. The v4 schema differs from v3 only
in `$id` and `protocol_id`. V1, the unexecuted v2 draft, and v3 remain distinct
and must be rejected by the v4 verifier.
