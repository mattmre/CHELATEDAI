# Session 2026-08-24 — EGV Qwen B/D campaign completion boundary

## Objective

Complete the frozen two-arm Qwen commissioning matrix, admit training rows only
through verified promotion evidence, harden the production training and held-out
paths, and publish only validated public-safe attestations.

## Research outcome

- The primary matrix reached 80 of 80 terminal requests.
- All 80 requests exhausted the frozen 12-attempt budget.
- The model produced 960 attempts: 617 response-contract failures and 343
  source-contract-accepted candidates.
- All 343 evaluated candidates were incorrect: 321 wrong outputs and 22 runtime
  exceptions.
- There were zero promotions, zero queued projections, and zero quarantines.
- The freezer covered all 343 candidates, excluded all 343 as ineligible, and
  emitted zero rows across zero distinct tasks.
- Final research disposition: `NO_ADMISSIBLE_TRAINING_SET`. No research adapter
  exists, and trained arms plus matched trained-versus-base evaluation remain
  `UNEVALUATED`.

## Diagnostic finding

The frozen cutoff-57 formatting diagnostic repaired the source transport format
for all 387 matched failures but recovered zero semantic passes in either the
original or repaired condition. This is diagnostic evidence that formatting was
a transport defect but not a sufficient correctness intervention. It is not a
promotion, training, or causal arm-comparison result.

## Implementation hardening

The production training overlay now covers raw and semantic input pins,
single-open bounded canonical admission, zero-row rejection before private
staging/evaluator/CUDA/model/output, complete LoRA tensor inventory and effective
delta proof, checkpoint/evaluation reauthorization, base immutability, reload
binding, private-tree scratch containment, and native no-replace whole-tree
publication.

The production held-out overlay now covers full receipt and ledger closure,
durable prompt-integrity failure evidence, exact ordered exception-chain replay
from trainer-supplied raw bytes, explicit non-resumable legacy schemas, bounded
process containment, and fail-closed recovery. Replay evidence does not prove
that the recorded bytes originated from a model.

Fresh adversarial review of the corrected 20-path composite found no remaining
code, privacy, identity, reproducibility-language, governance-state, or
cross-host-claim defect. That exact candidate was committed as
`1dd45c6ee567ffa48ea41d74959582f1a3015cfa`. The first hosted matrix then found
one test-harness mismatch: eight post-production-floor tests expected later
guards under Python 3.9 even though production correctly requires Python 3.10+.
Commit `8669dcd77821d43a54601fffdcbd4668c8315e2f` skips only those post-floor
assertions on 3.9 and adds a dependency-independent test of the floor itself.
Independent review matched the eight methods to all 11 hosted errors and issued
`GO`; production code was unchanged.

The exact `8669dcd7` source archive was byte-matched across both accelerators.
Each offline, read-only-source container passed six real PEFT/CUDA and Linux
publication tests. The genuine 343-candidate, zero-row freezer artifact was
then supplied with both exact digests and deliberately missing downstream
dependencies. Both runs reached the exact 20-task rejection and produced no
adapter output. These are implementation and negative-gate proofs, not a
research adapter or model-quality result.

The stacked held-out lane separately repaired the original process cleanup,
inventory, and native-handle findings. A later independent adversarial pass
found one additional cleanup-only multi-error evidence loss in both local and
remote boundaries. That defect and its fault-injection tests are repaired in
the current PR #306 overlay; a fresh exact-tree review remains required before
that overlay is committed.

## Infrastructure at wrap

Both accelerator nodes were observed ready and completed the exact acceptance
band. The operator-owned
inference service remains intentionally stopped while the campaign gates are
open. Its restore identity and verification procedure are retained only in the
private local handoff; no private topology or credentials are present here.

## Public evidence

The seven machine-readable evidence records are canonical, closed-schema, and
exact-byte pinned. The two reports and validator are required bundle members;
the seven records, two reports, and `validation.json` are targeted-privacy-scanned,
and the full bundle including the validator is Gitleaks-clean. The final
evidence addendum still requires fresh exact review before publication. It
deliberately distinguishes publicly exposed, schema/self-consistency-validated
counters from private seal context, historical exact-source rejection from the
later hardened proof, and tiny-model implementation acceptance from research
model evidence.
The redacted bundle does not contain row-level campaign inputs and cannot
independently recompute the private campaign result.

Reviewed artifact hashes:

- primary aggregate: `32370e4c85f52d1d6a77a718a41dff77b969582af8e0579894c07463b56b35b2`
- exact-source freezer gate: `36b86371819baa699eab2af20156892b8aaedf143ab2b92c0b2a31925db4419b`
- cutoff-57 diagnostic: `dfff9543251c58406f12e3aa49b493dd3c852823a7061ab2364f188a50027b95`
- first accelerator CPU/Qdrant smoke: `6163661009da9770d7cdcc1d786e3f9842c8e0e11cea6cb2900fdbdb50303c77`
- second accelerator CPU/Qdrant smoke: `c208be009d8d2c8c5fa0be1dd7dd7637e4ae942b30dc27268ff977dd87d87a0b`
- second accelerator focused recovery validation: `38a8a19dfcd3f8b16a7551a90d7d7728d8e4cc10fa6ae791549dd4c932d99b1e`
- dual-accelerator GPU/Linux acceptance: `6c2a87543c134f364a1da844bacf3156d3718f2a755efda4406a29999c88565d`
- validation record: `f44e2ca70ed7ce3bfe022880d1c845b5df6221f2d0222ae2932921f42d7471dc`
- validator: `9f7ed92cfc186f74a22fb4c33fc492614bb17d471a1b8711640dad9d747a7ddf`

## Pull requests at wrap

- PR #305 remains the main commissioning/training PR.
- PR #306 remains a stacked draft for production held-out/shock integration.
- PR #305's implementation candidate and Python-floor repair received independent
  `GO`, and its exact dual-accelerator acceptance is recorded. The evidence-only
  addendum and hosted exact-head matrix remain to be frozen and reviewed.
- PR #306's repaired overlay remains uncommitted pending fresh exact review,
  rebase onto the final PR #305 head, and Linux/process acceptance.
- No administrative bypass substitutes for those technical gates.

## Next session

1. Freeze, independently review, commit, and run the hosted matrix for the final
   PR #305 evidence addendum.
2. Finish the fresh exact-tree review of the repaired PR #306 overlay, then
   commit it and rebase it onto the final PR #305 head.
3. Run exact held-out Linux/process acceptance for PR #306 and repeat exact-head
   review after the rebase.
4. Update both PR bodies with exact results and explicit claim boundaries.
5. Merge only after exact technical gates. Restore and verify the operator's
   pre-existing inference service as the final infrastructure step.

## Explicit boundaries

- No LoRA research training occurred because the admissible dataset was empty.
- No trained-versus-base held-out result exists.
- No live cross-host trainer/evaluator transport is claimed.
- CPU/Qdrant and synthetic/tiny-model acceptance do not establish Qwen utility.
- Private prompts, generated source, task identities, credentials, endpoints,
  paths, host labels, and deployment topology are excluded from public files.
