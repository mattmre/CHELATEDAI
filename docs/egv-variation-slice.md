# EGV Variation slice

This document describes the bounded Variation implementation in this branch.
It composes the merged Evidence ledger/receipt boundary with the merged
Evaluation controller. Training and Campaign remain unavailable here.

## Frozen inputs

The production generator is local-only and revision-pinned:

| Input | Frozen value |
| --- | --- |
| Repository | `Qwen/Qwen3.5-2B-Base` |
| Revision | `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` |
| Model class | `Qwen3_5ForCausalLM` |
| Config class | `Qwen3_5TextConfig` |
| Transformers | `>=5.5,<6` |

The core package retains Python 3.9 syntax compatibility. The pinned model
stack is declared only for Python 3.10+ because Transformers 5.5+ does not
publish a Python 3.9 runtime; Python 3.9 can run the dependency-light core
and fixture paths, while model loading fails closed there.

`PinnedModelLoader` requires a local `model-manifest.json` with SHA-256 hashes
for every staged file and license metadata. It passes
`local_files_only=True`, `trust_remote_code=False`, and the pinned revision to
the exact text-only classes. Missing, changed, incompatible, or incomplete
model material fails closed; this branch does not download a model or include
model weights.

## Candidate contract

`BoundedCandidateLoop` runs one `(campaign, arm, task, seed)` trajectory with a
maximum of 12 attempts. Each proposal is bounded Python source, the immutable
public locus, a closed requested-authority value, and a sorted subset of IDs
from that arm's retrieval view. Candidate source is stored in a write-once,
content-addressed artifact below the arm/run workspace. The generator receives
public task metadata and retrieved event summaries only; evaluator input and
expected output remain in the evaluator process.

The loop creates or reuses the Evidence campaign/run, appends candidate and
dependency records, sends the opaque input through the Evaluation gateway,
materializes signed authority/verdict/effect receipts, and checks that the
ledger disposition agrees with the evaluator result. A non-read authority
promotion therefore requires the signed EFFECT receipt accepted by the ledger.
Every attempt creates an immutable JSON checkpoint plus the ledger checkpoint
record. Resume validates the campaign, arm, task, seed, model, adapter,
retrieval policy, protocol, artifact state, and ledger head before continuing.

## Arms and retrieval

The eight arm policies are frozen as follows:

| Arm | Model | Retrieval | Authority |
| --- | --- | --- | --- |
| A | base | verified successes only | receipt capture |
| B | base | ordinary failure summaries | receipt capture |
| C | base | correction-aware | receipt capture |
| D | base | correction-aware | typed authority enforced |
| E | sealed LoRA | correction-aware | typed authority enforced |
| F | sealed LoRA | verified successes only | receipt capture |
| G | sealed LoRA | ordinary failure summaries | receipt capture |
| H | sealed LoRA | correction-aware | receipt capture |

Each arm has a disjoint filesystem namespace and candidate-ID prefix. Retrieval
filters by campaign, run, task, current ledger validity, and arm before a
generator can cite an event. E-H require an exhaustive, content-addressed
`SealedAdapterArtifact` bound to the frozen base-model revision. A hexadecimal
digest alone is rejected; the later Training slice is responsible for producing
and applying that sealed adapter, so an absent adapter fails closed.

## Evaluation boundary

Production Variation accepts only `ControllerEvaluationGateway` over the
enforceable Docker Evaluation sandbox. The controller is the sole path to the
hidden evaluator and receipt ingestion; a test-only local helper is available
only for the explicitly labelled floor fixture. Docker isolation and the
Evaluation hidden oracle remain the authority boundary. A fixture smoke never
claims Docker enforcement, model inference, a GPU, Qdrant, Spark, a campaign,
or a hosted service. A missing enforceable backend must fail closed.

## CLI and smoke

```text
python -m egv variation smoke --json
python -m egv variation smoke --output <new-output-root> --json
python -m egv variation model-preflight --model-root <staged-model-root> --json
```

The Variation smoke is `runtime_tier=floor-fixture`: it generates the frozen
36-task corpus from an evaluator-private seed, runs one held-out task through
the deterministic test-only gateway, demonstrates failure retrieval followed
by promotion, and writes private state under `private/` and the redacted report
under `public/`. `campaign_path_exercised=false` means the later Campaign phase
was not exercised; the fixture only creates the ledger records needed to test
the Variation trajectory. The private seed, hidden values, receipts, ledger,
and candidate source are not publishable artifacts. The closed public scanner
re-reads the final JSON artifact and rejects seed fields, candidate-source
digests, held-out IDs, private paths, and hidden-answer markers. A real production run requires
the local pinned model and the configured digest-pinned Docker runtime; there
is no network or image-pull fallback.

`scripts/smoke.sh --api-only` exits before Variation/Evaluation execution.
The full script reports the Variation tier separately from the existing
Evidence, Evaluation, and legacy surface stages; missing unrelated legacy
dependencies are not silently relabelled as Variation success.

No DeepSeek service is stopped, started, inspected, or reconfigured by this
slice. The Training and Campaign phases remain fail-closed and are not
implemented on this branch.
