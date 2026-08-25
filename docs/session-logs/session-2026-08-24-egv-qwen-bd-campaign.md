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

Fresh adversarial review found additional direct-API/private-tree, Windows
cleanup, and Python 3.9/3.10 compatibility gaps during session wrap. They were
returned to their implementers before publication. The repaired seven-path
training implementation previously received exact-diff `GO`; its
ResourceWarning-as-error suite passed 45 tests with 3 genuine PEFT/CUDA skips,
and its wider suite passed 99 tests with the same 3 skips. Later documentation
and public-safety corrections superseded that reviewed diff identity. The
current composite candidate must receive a new exact review before it is called
approved for Spark acceptance or publication. The held-out reviewer found
additional inventory/cleanup-evidence gaps after its first repaired freeze. The
exact snapshot at raw diff SHA-256
`ec56ef561a0b36ba7130cb8ecdf3e28d55abad164ef665e82860bf3bd5a332a0`
received `NEEDS CHANGES`: post-`Popen` child cleanup can leak, simultaneous
cleanup failures can lose evidence, unexpected non-JSON inventory entries are
ignored, and some native handle closes are unchecked. Its broad suite passed
112 tests with 6 platform skips, but the targeted fault evidence keeps it HOLD.

## Infrastructure at wrap

Both accelerator nodes were last observed ready and idle. The operator-owned
inference service remains intentionally stopped while the campaign gates are
open. Its restore identity and verification procedure are retained only in the
private local handoff; no private topology or credentials are present here.

## Public evidence

The six machine-readable evidence records are canonical, closed-schema, and
exact-byte pinned. The two reports and validator are required bundle members;
the six records, two reports, and `validation.json` are targeted-privacy-scanned,
and the full bundle including the validator is Gitleaks-clean. The current
composite candidate still requires fresh exact review after the later
documentation and scanner corrections. It deliberately distinguishes publicly exposed,
schema/self-consistency-validated counters from private seal context and
historical exact-source rejection from the pending hardened pre-model proof.
The redacted bundle does not contain row-level campaign inputs and cannot
independently recompute the private campaign result.

Reviewed artifact hashes:

- primary aggregate: `32370e4c85f52d1d6a77a718a41dff77b969582af8e0579894c07463b56b35b2`
- exact-source freezer gate: `36b86371819baa699eab2af20156892b8aaedf143ab2b92c0b2a31925db4419b`
- cutoff-57 diagnostic: `dfff9543251c58406f12e3aa49b493dd3c852823a7061ab2364f188a50027b95`
- first accelerator CPU/Qdrant smoke: `6163661009da9770d7cdcc1d786e3f9842c8e0e11cea6cb2900fdbdb50303c77`
- second accelerator CPU/Qdrant smoke: `c208be009d8d2c8c5fa0be1dd7dd7637e4ae942b30dc27268ff977dd87d87a0b`
- second accelerator focused recovery validation: `38a8a19dfcd3f8b16a7551a90d7d7728d8e4cc10fa6ae791549dd4c932d99b1e`
- validation record: `c62b03fe79186c0675294000a439a0afb58d9ec72e2bb3935efb5fef50c6ad07`
- validator: `43042ba7fea82da69220a412ff7c1aee321b4e04c1b864507466a8d09f5602cc`

## Pull requests at wrap

- PR #305 remains the main commissioning/training PR.
- PR #306 remains a stacked draft for production held-out/shock integration.
- PR #305's historical training-only candidate received exact-diff `GO`, but
  the current composite candidate has later documentation/privacy corrections
  and requires fresh exact review. Neither PR is merge-ready until that review
  and the remaining accelerator/Linux acceptance are recorded, and PR #306's
  rejected held-out candidate is repaired and freshly reviewed.
- No administrative bypass substitutes for those technical gates.

## Next session

1. Commit the reviewed training/evidence candidate on PR #305 and preserve its
   exact reviewed identity in the commit evidence.
2. On the exact PR #305 SHA, run the genuine zero-row artifact and prove rejection
   before private staging, evaluator, CUDA, model load, or output creation.
3. Run bounded real-PEFT/CUDA and Linux native no-replace implementation
   acceptance. These are implementation tests, not research-adapter evidence.
4. Push/check PR #305. Repair the four held-out blockers, freeze and independently
   review the new candidate, then rebase PR #306 onto the final PR #305 head.
5. Run the exact held-out Linux/process acceptance for PR #306 and repeat its
   exact-diff review after the rebase.
6. Publish the reviewed public-safe bundle and update both PR bodies with exact
   results and explicit boundaries.
7. Merge only after exact technical gates. Restore and verify the operator's
   pre-existing inference service as the final infrastructure step.

## Explicit boundaries

- No LoRA research training occurred because the admissible dataset was empty.
- No trained-versus-base held-out result exists.
- No live cross-host trainer/evaluator transport is claimed.
- CPU/Qdrant and synthetic/tiny-model acceptance do not establish Qwen utility.
- Private prompts, generated source, task identities, credentials, endpoints,
  paths, host labels, and deployment topology are excluded from public files.
