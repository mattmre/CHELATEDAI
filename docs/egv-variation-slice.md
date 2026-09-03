# EGV Variation slice

Production Variation supports the original local Docker controller and an
exact sealed independent evaluator command. See
`docs/egv-remote-variation-evaluator.md` for its trust manifest, private-input
separation, receipt-chain protocol, deployment commands, and runtime evidence
boundary.

This document describes the bounded Variation implementation in this branch.
It composes the merged Evidence ledger/receipt, Evaluation, Training, and
Campaign foundations.

## Frozen inputs

The production generator is local-only and revision-pinned:

| Input | Frozen value |
| --- | --- |
| Repository | `Qwen/Qwen3.5-2B-Base` |
| Revision | `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` |
| Model class | `Qwen3_5ForCausalLM` |
| Config class | `Qwen3_5TextConfig` |
| Transformers | `>=5.5,<6` |

PEFT is declared as `>=0.15.2,<1` for Python 3.10+; it is not an optional
test shim for production LoRA arms.

The core package retains Python 3.9 syntax compatibility. The pinned model
stack is declared only for Python 3.10+ because Transformers 5.5+ does not
publish a Python 3.9 runtime; Python 3.9 can run the dependency-light core
and fixture paths, while model loading fails closed there.

`PinnedModelLoader` requires a local `model-manifest.json` with SHA-256 hashes
for every staged file and license metadata. It passes
`local_files_only=True`, `trust_remote_code=False`, and the pinned revision to
the exact text-only classes. Missing, changed, incompatible, or incomplete
model material fails closed; this branch does not download a model or include
model weights. `model-preflight` reports
`network=offline-environment-scoped-preflight` together with the four offline
environment variables active during that verification window; it does not
claim a process-wide network firewall.

## Candidate contract

`BoundedCandidateLoop` runs one `(campaign, arm, task, seed)` trajectory with a
maximum of 12 attempts. Each proposal is bounded Python source, the immutable
public locus, a closed requested-authority value, and a sorted subset of IDs
from that arm's retrieval view. Candidate source is stored in a write-once,
content-addressed artifact below the arm/run workspace. The generator receives
public task metadata and retrieved event summaries only; evaluator input and
expected output remain in the evaluator process.

The loop creates or reuses the Evidence campaign/run, appends candidate and
dependency records, invokes the Evaluation gateway (with opaque input retained
only by a local evaluator or resolved on the independent evaluator),
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
`SealedAdapterArtifact` bound to the frozen base-model revision plus a
`PinnedModelLoader`-issued application attestation binding the adapter digest,
base manifest/state, and applied model state. Loader, generator, and loop
validation additionally require an actual local
`peft.PeftModel`/`PeftModelForCausalLM` instance with one active LORA adapter
whose runtime config matches the sealed `adapter_config.json`. A hexadecimal
digest, dummy artifact, importable sentinel, or non-applied adapter is
rejected; the Training runtime produces and applies that sealed adapter, while
an absent or unverified adapter fails closed.

## Evaluation boundary

Production Variation accepts only the exact local
`ControllerEvaluationGateway` or exact sealed
`RemoteControllerEvaluationGateway`. Both terminate in the enforceable Docker
Evaluation sandbox. The evaluator controller is the sole path to the hidden
oracle; remote receipts cross back only after complete signature/binding checks
and atomic ledger admission. A test-only local helper remains available only
for the explicitly labelled floor fixture. Docker isolation and the
Evaluation hidden oracle remain the authority boundary. A fixture smoke never
claims Docker enforcement, model inference, a GPU, Qdrant, Spark, a campaign,
or a hosted service. A missing enforceable backend must fail closed.

`BoundedCandidateLoop(...)` is a compatibility factory: it returns distinct
private production and fixture concrete classes with distinct `run` methods.
The production method has no fixture early-return path and unconditionally
revalidates the sealed gateway, model generator, and authority before any
evaluation. The trusted-host boundary is explicit: arbitrary Python already
executing in the controller process can inspect or rewrite that process's
heap, frames, closures, classes, and registries. In-process seals are not a
defense against that capability; candidate execution and hidden authority
therefore remain separate Docker/evaluator process boundaries, and fixture
execution is structurally unavailable from a production loop object.

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
