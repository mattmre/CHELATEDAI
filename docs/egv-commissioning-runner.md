# EGV commissioning runner

This runner connects the frozen 20/8/80 commissioning inputs to the production
Variation and Training primitives. It does not claim a completed model run.

## Trust split

- The evaluator prepares the corpus-derived inputs. The trainer receives only
  the 20 public training records and 80 content-derived B/D requests.
- Development and held-out task identities, inputs, expected outputs, and the
  evaluator seed remain on the evaluator.
- Every candidate prompt context and source is retained in a private,
  content-addressed sidecar. Public reports never include either value.
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
  --evaluator-seed EVALUATOR_SEED \
  --trainer-output TRAINER_INPUTS.json \
  --evaluator-output EVALUATOR_PRIVATE_INPUTS.json
```

Transfer only `TRAINER_INPUTS.json` to the trainer. It contains exactly 20
public training records and 80 pending requests. Each request uses the same
`commissioning_run_id` implementation as the Variation loop.

## Run and resume

Start with one request ID. Omit `--request-id` only after the one-request proof
passes.

```bash
python -m egv commissioning run \
  --trainer-inputs TRAINER_INPUTS.json \
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

The command loads the pinned model in BF16, constructs the exact production
model generator and remote evaluator gateway, validates every request binding,
and uses the existing Variation checkpoint for an interrupted active request.
Requests already present in the atomic completion journal are skipped.

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

This command rebuilds the evaluator corpus, verifies the full receipt chain and
ledger cutoff, requires private sidecars for every candidate, excludes
ineligible attempts, scans against private development and held-out material,
and emits the exact `egv-sealed-training-runtime-input-v1` artifact consumed by
`python -m egv train-lora`.

## Initial go/no-go sequence

The current sanitized live-pilot result and its evidence boundary are recorded
in [EGV live pilot evidence — 2026-08-22](egv-live-pilot-evidence-2026-08-22.md).
The first full model-reach attempt failed closed at the candidate-JSON contract,
so the sequence below has not advanced past step 2.

1. Verify the corrected remote gateway's persistent idempotency and receipt
   chain state.
2. Run one Arm B request and verify its journal, checkpoint, sidecar, ledger,
   and three-receipt result.
3. Run the paired Arm D request for the same task and seed.
4. Exercise interruption and resume once; confirm no duplicate evaluation.
5. Scale to 80 requests only after those proofs pass.
6. Freeze on the evaluator and verify the sealed dataset before training.
