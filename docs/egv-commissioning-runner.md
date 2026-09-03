# EGV commissioning runner

This runner connects the frozen 20/8/80 commissioning inputs to the production
Variation and Training primitives. It does not claim a completed model run.

## Trust split

- The evaluator prepares the corpus-derived inputs. The trainer receives only
  the 20 public training records, their exact public `README.md` and
  `src/task.py` bytes, and 80 content-derived B/D requests. Source bytes are in
  a separate digest-bound bundle; no development or held-out repository enters
  either trainer file.
- Development and held-out task identities, inputs, expected outputs, and the
  evaluator seed remain on the evaluator.
- Every exact rendered-chat prompt, raw model continuation, contract response,
  candidate context, and candidate source is retained in a private,
  write-once content-addressed sidecar. Public reports never include those
  values.
- The authoritative evaluator receipt ledger remains the evidence source. A
  runner journal prevents a terminal request from being executed twice.

The remote evaluator gateway must provide durable evaluator-side request
idempotency and a locked receipt-chain head. Do not run commissioning against a
stateless evaluator endpoint.

## Prepare inputs

Run preparation on the evaluator:

```bash
python -m egv commissioning prepare \
  --campaign-id CAMPAIGN_ID \
  --model-digest MODEL_MANIFEST_SHA256 \
  --model-root PINNED_MODEL \
  --evaluator-seed EVALUATOR_SEED \
  --trainer-output TRAINER_INPUTS.json \
  --trainer-sources-output TRAINER_SOURCES.json \
  --evaluator-output EVALUATOR_PRIVATE_INPUTS.json
```

Transfer only `TRAINER_INPUTS.json` and `TRAINER_SOURCES.json` to the trainer.
The source bundle contains exactly two UTF-8 public files per training task and
is transitively bound to each public task `source_digest`, the trainer-input
digest, and every request's task-record digest. Preparation loads the exact
offline pinned tokenizer and freezes `source-only-v1` together with the model
manifest, prompt manifest, tokenizer chat-template bytes, 512-token generation
ceiling, disabled thinking, greedy decoding, and special-token decode setting
as one generation-profile digest. This is not a runtime CLI choice.

## Run and resume

Start with one request ID. Omit `--request-id` only after the one-request proof
passes.

```bash
python -m egv commissioning run \
  --trainer-inputs TRAINER_INPUTS.json \
  --trainer-sources TRAINER_SOURCES.json \
  --model-root PINNED_MODEL \
  --ledger PRIVATE_LEDGER.sqlite \
  --blob-root PRIVATE_BLOBS \
  --evaluator-manifest EVALUATOR_SERVICE.json \
  --evaluator-public-key EVALUATOR.pub \
  --evaluator-command CONTENT_BOUND_REMOTE_COMMAND \
  --workspace PRIVATE_VARIATION_WORKSPACE \
  --journal PRIVATE_RUN_JOURNAL.json \
  --source-commit SOURCE_COMMIT \
  --request-id FROZEN_REQUEST_ID \
  --device cuda \
  --json
```

The command validates both trainer files before loading Torch, loads the pinned
model in BF16, and verifies the actual tokenizer/generator profile before the
ledger is opened. It then constructs the exact production generator and remote
evaluator gateway and validates every request binding. Requests already present
in the atomic completion journal are skipped.

`RESPONSE_CONTRACT` failures consume one of the frozen 12 attempts. Each failed
attempt persists the exact prompt and raw response privately, creates no
candidate or evaluator receipt, and advances deterministically. A restart with
only such failures derives its next cursor from those immutable records; after
12 failures the request is journaled `BUDGET_EXHAUSTED`. Prompt-render,
prompt-integrity, and model-generation failures are durable but fatal and are
never automatically retried. Candidate checkpoints remain the legacy v1
schema. Therefore failure-only prefixes resume from private write-once records,
while exactly-once model inference cannot be claimed for a process loss before
the failure/success record itself reaches durable storage.

## Freeze the training dataset

After all 80 terminal runs, transfer a stopped, content-verified ledger
snapshot, its blob tree, and the private sidecar tree to the evaluator. Run:

```bash
python -m egv commissioning freeze-training \
  --trainer-inputs TRAINER_INPUTS.json \
  --ledger FROZEN_LEDGER.sqlite \
  --blob-root FROZEN_BLOBS \
  --private-store PRIVATE_VARIATION_WORKSPACE/private-trajectories \
  --evaluator-seed EVALUATOR_SEED \
  --evaluator-public-key EVALUATOR.pub \
  --model-root PINNED_MODEL \
  --output PRIVATE_FROZEN_DATASET.json \
  --json
```

This command rebuilds the evaluator corpus, reloads the exact offline tokenizer
and rejects generation-profile drift, verifies the full receipt chain and
ledger cutoff, and requires private sidecars for every candidate. For
source-only rows it binds the private generation-record digest to immutable
ledger metadata, binds the proposal source digest to the sidecar/ledger source,
and uses the exact stored rendered-chat bytes as the SFT prompt. Its token
ceiling uses the same separate prompt/target tokenization plus target EOS as the
training runtime. It excludes ineligible attempts, scans against private
development and held-out material, and emits the exact
`egv-sealed-training-runtime-input-v1` artifact consumed by
`python -m egv train-lora`.

## Initial go/no-go sequence

The earlier sanitized live-pilot result and its evidence boundary are recorded
in [EGV live pilot evidence — 2026-08-22](egv-live-pilot-evidence-2026-08-22.md).
That run failed closed at the candidate-JSON contract. The versioned
`source-only-v1` integration described here resolves the transport-contract
path in code; it does not claim a successful new live model or evaluator run.

1. Verify the corrected remote gateway's persistent idempotency and receipt
   chain state.
2. Run one Arm B request and verify its journal, checkpoint, sidecar, ledger,
   and three-receipt result.
3. Run the paired Arm D request for the same task and seed.
4. Exercise interruption and resume once; confirm no duplicate evaluation.
5. Scale to 80 requests only after those proofs pass.
6. Freeze on the evaluator and verify the sealed dataset before training.
