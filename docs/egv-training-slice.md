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
The evaluator-side freezer reports both the artifact's exact raw SHA-256 as
`output_digest` and the semantic dataset digest as `dataset_digest`. The
operator must capture both values from that trusted run, carry them across the
handoff, and supply them to `train-lora`. These two pins are manual operator
trust anchors: they are not a signature, an evaluator identity proof, or
cryptographic proof of freezer provenance. Relative to those trusted pins, the
trainer rejects an internally self-consistent replacement dataset before it
constructs the evaluator client, checks CUDA, or loads the model. A future
protocol may add a separately specified signed freezer receipt; this version
does not accept or claim one.

The trainer admits the private export through one bounded file descriptor. It
rejects symlink/reparse ancestors, a symlink/reparse leaf, hardlink aliases,
identity changes during admission, non-canonical bytes, and artifacts larger
than 128 MiB. Raw hashing, UTF-8 decoding, JSON parsing, and semantic validation
all use the same bytes read from that single handle.

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
its exact adapter tensors, and requires an exact 114-target A/B inventory with
unchanged names, shapes, and FP32 dtypes. Every tensor must be finite. A bounded,
chunked proof computes the frozen-scaled difference between selected and
initial `B @ A` products and requires at least one nonzero effective target;
changing only `A` while `B` remains zero is not training evidence. The proof
first clones the complete caller-owned tensor inventories into private CPU
storage, then performs every shape, dtype, finite, digest, and effective-delta
check from those coherent snapshots. It records deterministic
initial/selected-state digests plus the changed-target set digest and count,
and is recomputed immediately before sealing.

The trainer saves a real PEFT adapter, requires its sealed digest to equal the
signed development receipt, and reopens it through the pinned Variation model
loader with a real PEFT attestation. It then extracts the actual reloaded LoRA
tensors, recomputes the effective-delta proof, and requires exact equality to
the evaluator-selected state. Selection itself freezes a canonical deep
snapshot and digest of the complete `DevelopmentLossEvaluation` plus its exact
matching `TrainingCheckpoint`. Immediately before adapter sealing, the trainer
requires the still-stored evaluation to equal that snapshot, reconstructs both
objects, and asks the gateway to verify the signed evaluation again against the
selected checkpoint artifact and logical checkpoint ID, model, dataset,
protocol, gateway, and development manifest. Reauthorization also requires a
finite nonnegative loss, the exact eight-row sample count, the evaluator-seen
candidate adapter, its recomputed output digest, the closed receipt semantics,
and the exact Ed25519 key, signature, and one-shot chain position. Only the
frozen receipt's candidate-adapter digest and receipt digest enter sealing and
final binding.
The final binding digest coherently covers that proof, the selected checkpoint
artifact, the post-training base-immutability proof, the selected evaluator
receipt, the sealed/evaluator-approved adapter, and the reloaded
applied-model-state digest.

The external evaluator command is rehashed before every call and executed only
from a fresh content-addressed copy. Every trainer-controlled evaluator write
uses an exclusive, identity-snapshotted subdirectory of the private training
tree: the temporary PEFT adapter and manifest, both pinned command copies, and
each command's working directory. These directories are validated before and
after writes and process execution and are retained; no automatic temporary-
directory cleanup can follow a substituted pathname. The exported production
`LoRATrainer` constructor also requires the gateway scratch root and its stable
identity to be the exact `private_training_tree/evaluator-scratch` child before
any evaluator invocation or trainer write; cross-tree, out-of-tree, and
substituted scratch roots fail closed. The production contract can invoke a
separately administered evaluator through a second digest-pinned command. When
that command is independently configured on Spark2, the intended protocol is to
send adapter bytes, verify and install the complete sealed tree under its
digest, sign the content reference, and resolve only that evaluator-local
reference for loss evaluation. No live cross-host trainer/evaluator route has
been executed or evidenced in this campaign; the current tests establish the
local command, digest, and containment contracts only. Missing or substituted
authority, data, target, model, checkpoint, or PEFT state fails closed.

The requested production output root must be newly absent. After the data
handoff passes, the trainer creates a cryptographically random, exclusive
private sibling staging root under the output parent and restricts it to the
owner on POSIX. Checkpoints, evaluator scratch, and the final adapter are
written only inside that tree. The trainer repeatedly verifies the stable
staging-root and ancestor identities and rejects symlink, reparse,
cross-device, hardlinked-file, or non-regular tree entries through completion.
Direct `seal_adapter` calls accept only the exact newly absent
`private_training_tree/adapter` leaf and reject cross-tree, out-of-tree, or
substituted roots before model access or filesystem writes.
It freezes the completed tree's identities, then publishes the entire root
atomically without replacement:
`MoveFileExW` with flags `0` on Windows or
`renameat2(RENAME_NOREPLACE)` on Linux. If that primitive is unavailable, or a
competitor claims the requested root, publication fails closed without
overwriting the competitor. Failed private staging trees are retained rather
than destructively cleaned through a pathname whose identity may have changed;
an operator must inspect them and choose a fresh destination. The returned
`adapter_root` always names the successfully published path.

This filesystem boundary assumes the training process memory and same-user
debugger control are not compromised; it does not claim protection from an
attacker who can arbitrarily rewrite live process objects. It does defend the
documented pathname-substitution boundary with private unpredictable staging,
owner-only access where supported, repeated identity checks, closed tree
validation, and no-replace publication.

The honest CPU smoke is:

```bash
python -m egv training smoke --json
```

The production contract surface is exposed as:

```bash
python -m egv train-lora \
  --model-root PATH \
  --train-manifest PRIVATE_FROZEN_DATASET.json \
  --sealed-training-artifact-sha256 FREEZER_ARTIFACT_SHA256 \
  --sealed-training-dataset-digest FREEZER_DATASET_DIGEST \
  --development-manifest EVALUATOR_SERVICE.json \
  --evaluator-public-key EVALUATOR.pub \
  --evaluator-command EVALUATOR_EXECUTABLE \
  --evaluator-transfer-command ADAPTER_TRANSFER_EXECUTABLE \
  --output OUTPUT_DIR \
  --device cuda \
  --json
```

In the intended two-Spark deployment, the Spark2 operator first freezes an exact
eight-row private runtime and path-free public service manifest with `training
freeze-evaluator-service`. A separately configured transfer executable can
invoke `training receive-adapter --adapter-store ...`, and a loss executable can
invoke `training evaluator-once --adapter-store ...`. These are generic command
boundaries: host addresses, credentials, evaluator seeds, and private filesystem
paths are operator inputs and are never embedded in the repository or service
manifest. This cross-host deployment has not yet been executed in the campaign.

The command is executable only when every sealed local artifact and external
authority binding validates. It never falls back to a fixture. It does not
stop or restore DeepSeek itself and it does not publish credentials or held-out
material.
