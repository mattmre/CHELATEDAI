# EGV Training runtime

This bounded slice implements the runtime contract for the first LoRA training
artifact. It does not claim that the pinned Qwen checkpoint has been loaded,
that PEFT training has run on GBA1, or that any adapter has been promoted.

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
| Optimizer / scheduler | `adamw_torch` / `linear` |
| Per-device batch / accumulation | `1 / 8` |
| Precision | `fp32` |
| Evaluation cadence / patience | every epoch / 1 interval |
| Seeds | train `20260822`, evaluation `20260823` |

The protocol also freezes weight decay, warmup, row/step ceilings, and the
attention-only target-module allowlist from ADR-0001. Any missing, additional,
or changed field fails validation and changes no accepted protocol digest.

## Data and labels

Training rows are exact `egv-training-row-v1` wrappers around immutable
`egv-sft-row-v1` Evaluation records and must be in the `train` split.
Development rows are owned by `DevelopmentLossGateway` and must
be in the `dev` split. Held-out rows, held-out identifiers, and train/dev
overlap are rejected. A sealed input envelope binds the model digest, private
train-row bindings, development manifest, and protocol digest.

The tokenizer is called with `truncation=False`. Prompt tokens receive `-100`
labels; target tokens receive their token IDs and one EOS token when the
tokenizer supplies one. A sequence over 4,096 tokens is rejected rather than
truncated or packed. The public report contains row counts and digests, never
prompt, target, expected-output, or held-out content.

## Development selection

The gateway evaluates a checkpoint through its private development rows and
returns a signed Ed25519 `VERDICT` receipt. The receipt is closed and binds the
checkpoint, model, full data manifest, development manifest, protocol, gateway
configuration, loss digest, and sample count. It contains no development row
IDs or content. A stale, forged, wrong-model, wrong-data, wrong-protocol, or
loss-mismatched receipt is rejected. The trainer selects the lowest valid
development loss with a deterministic digest tie-break.

Training emits `NOT_PROMOTED_TRAINING_ARTIFACT`; promotion remains a separate
Variation/Evidence decision and is never synthesized by this runtime.

## Production boundary and smoke

Production execution requires Python 3.10+, an exact `LoadedPinnedModel` with
the merged pinned-model manifest, the local PEFT runtime, sealed inputs, and an
exact production `DevelopmentLossGateway`. The gateway and PEFT application are
revalidated at construction and at every run boundary. Missing PEFT, an
unsealed input, a fixture model, or an absent/non-production gateway fails
closed.

The honest CPU smoke is:

```bash
python -m egv training smoke --json
```

The production contract surface is exposed as:

```bash
python -m egv train-lora --model-root PATH --train-manifest PATH --development-manifest PATH
```

That command fails closed until the sealed production orchestration supplies
the runtime objects; it does not fall back to a fixture or claim Qwen
execution. No DeepSeek process, service configuration, network, credential, or
held-out asset is touched by this slice.
