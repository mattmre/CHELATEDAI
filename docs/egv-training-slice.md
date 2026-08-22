# EGV Training runtime

This bounded slice implements the runtime contract for the first LoRA training
artifact. The CPU smoke makes no Qwen claim. A successful production command
truthfully records a local pinned-Qwen execution, but still emits
`NOT_PROMOTED_TRAINING_ARTIFACT` and makes no deployment claim.

## Frozen protocol

`egv.training.TrainingProtocol` is an immutable, content-addressed record. Its
initial values are:

| Setting | Frozen value |
|---|---:|
| LoRA rank / alpha / dropout | `16 / 32 / 0.05` |
| Learning rate | `2e-4` |
| Maximum epochs | `3` |
| Maximum sequence length | `4096` |
| Packing | disabled |
| Optimizer / scheduler | `adamw_torch` / true linear decay |
| Per-device batch / accumulation | `1 / 8` |
| Device / base precision | CUDA / `bfloat16` |
| LoRA + optimizer-state exception | `float32` |
| Evaluation cadence / patience | every epoch / 1 interval |
| Seeds | train `20260822`, evaluation `20260823` |

The protocol also freezes weight decay, warmup, row/step ceilings, and the
attention-only target-module allowlist from ADR-0001. Any missing, additional,
or changed field fails validation and changes no accepted protocol digest.

## Data and labels

Production consumes the canonical private export of `FrozenTrainingDataset`.
Every positive target must retain its validated `egv-sft-row-v1` envelope and
PASS/PROMOTED evidence binding. The frozen selection policy chooses exactly one
earliest accepted B/D trajectory for each of the 20 training tasks. Development
rows never enter the trainer; the external evaluator manifest binds exactly
eight private development row IDs and their aggregate digest.

The tokenizer is called with `truncation=False`. Prompt tokens receive `-100`
labels; target tokens receive their token IDs and one EOS token when the
tokenizer supplies one. A sequence over 4,096 tokens is rejected rather than
truncated or packed. The public report contains row counts and digests, never
prompt, target, expected-output, or held-out content.

## Development selection

The production gateway is an external evaluator client. Its executable bytes,
adapter-transfer executable bytes, public key/key ID, service manifest, model, protocol, campaign, and private
development digest are frozen before training. It evaluates a sealed adapter
and returns a signed Ed25519 `VERDICT` receipt. The receipt is closed and binds the
checkpoint, model, full data manifest, development manifest, protocol, gateway
configuration, loss digest, and sample count. It contains no development row
IDs or content. A stale, forged, wrong-model, wrong-data, wrong-protocol, or
loss-mismatched receipt is rejected. The trainer selects the lowest valid
development loss with a deterministic digest tie-break.

Training emits `NOT_PROMOTED_TRAINING_ARTIFACT`; promotion remains a separate
Variation/Evidence decision and is never synthesized by this runtime.

## Production boundary and smoke

Production execution requires Python 3.10+, an exact `LoadedPinnedModel`, the
canonical frozen dataset, the sealed 114-module target manifest, local PEFT,
and an evaluator-owned external service. It captures the base files/tensors
before PEFT, requires CUDA BF16 base tensors, restricts the optimizer to FP32
LoRA A/B tensors and FP32 optimizer state, applies actual linear LR decay, rechecks the base
after training, writes a foundation `TrainingCheckpointStore` safetensors
checkpoint at every epoch, restores the evaluator-approved checkpoint, compares
its exact adapter tensors, saves a real PEFT adapter, requires its sealed digest
to equal the signed development receipt, and reopens it through the pinned
Variation model loader with a real PEFT attestation. The external evaluator
command is rehashed before every call and executed only from a fresh
content-addressed copy. The trainer sends adapter bytes to the independently
administered evaluator through a second digest-pinned command. Spark2 verifies
and installs the complete sealed tree under its adapter digest, signs the
content reference, and resolves that evaluator-local reference for loss
evaluation. No Spark1-local adapter path crosses the boundary. Missing or substituted
authority, data, target, model, checkpoint, or PEFT state fails closed.

The honest CPU smoke is:

```bash
python -m egv training smoke --json
```

The production contract surface is exposed as:

```bash
python -m egv train-lora \
  --model-root PATH \
  --train-manifest PRIVATE_FROZEN_DATASET.json \
  --development-manifest EVALUATOR_SERVICE.json \
  --evaluator-public-key EVALUATOR.pub \
  --evaluator-command EVALUATOR_EXECUTABLE \
  --evaluator-transfer-command ADAPTER_TRANSFER_EXECUTABLE \
  --output OUTPUT_DIR \
  --device cuda \
  --json
```

Spark2 freezes its exact eight-row private runtime and path-free public service
manifest with `training freeze-evaluator-service`. Its transfer executable
invokes `training receive-adapter --adapter-store ...`; its loss executable
invokes `training evaluator-once --adapter-store ...`. Both commands are
generic command boundaries: host addresses, credentials, evaluator seeds, and
private filesystem paths are operator inputs and are never embedded in the
repository or service manifest.

The command is executable only when every sealed local artifact and external
authority binding validates. It never falls back to a fixture. It does not
stop or restore DeepSeek itself and it does not publish credentials or held-out
material.
