# EGV Qwen B/D commissioning campaign: final public-safe result

Status: completed negative commissioning result; no admissible LoRA dataset and no trained-model comparison.

The frozen campaign completed all 80 planned requests on source commit
`8f1955abe8a214cc6f469547d3cc1bbddbf7d1ec`. The design crossed 20 tasks,
two prespecified arms (B and D), and two seeds. Each request had a fixed budget
of 12 candidate attempts. This report contains aggregate results only; private
prompts, candidate source, task identities, evaluator material, credentials,
paths, endpoints, and deployment topology are excluded.

## Primary result

| Measure | Result |
|---|---:|
| Terminal requests | 80 / 80 |
| Terminal disposition | 80 `BUDGET_EXHAUSTED` |
| Candidate attempts | 960 |
| Source-contract failures | 617 |
| Source-contract-accepted candidates | 343 |
| Correct candidates | 0 / 343 |
| Wrong-output verdicts | 321 |
| Runtime-exception verdicts | 22 |
| Promotions | 0 |
| Projection queue / quarantine | 0 / 0 |

Arm B produced 174 evaluated candidates and arm D produced 169. Every one was
incorrect. When retrieval evidence was available, the generated candidate
cited it in 154 of 154 arm-B opportunities and 149 of 149 arm-D opportunities.
That shows that the evidence-use mechanism was exercised; it does not show that
retrieval improved correctness, because neither arm produced a passing
candidate in this campaign.

The private evaluator/ledger seal recorded 343 cached operations, 1,029 signed
receipts, 4,826 chained events, 80 runs, no pending operation, and matching
evaluator/primary receipt heads. Those private seal details were checked during
the campaign but are not reproduced as fields in the public aggregate. The
public artifact independently exposes the 343 candidate/checkpoint/effect
counts, 1,029 receipt count, and zero queue/quarantine counts; it must not be
used to claim public reproduction of the omitted private reconciliation fields.

## Formatting diagnostic

A separate diagnostic was frozen earlier at primary cutoff 57. It replayed all
387 exact-single-fence source-contract failures available at that cutoff under
matched original and formatting-repaired conditions. Formatting repair made
the source contract parseable in all 387 repaired cases, but both conditions
still produced zero semantic passes. This supports a narrow conclusion:
response formatting was a real transport defect, but repairing it was not
sufficient to recover correct programs in that frozen failure census.

The diagnostic is not an iid sample, does not cover the later primary failures,
and did not write to or retroactively change the primary campaign. It supplies
no promotion or training-eligibility evidence.

## Training freezer decision

The evaluator freezer consumed a byte-identical copied ledger while the sealed
original remained read-only. Its pre/post ledger hash and table counts were
unchanged and its integrity check passed.

| Freezer measure | Result |
|---|---:|
| Frozen candidates covered | 343 |
| `invalid_or_ineligible_attempt` | 343 |
| Admissible private training rows | 0 |
| Distinct represented tasks | 0 |
| Training disposition | `NO_ADMISSIBLE_TRAINING_SET` |

The zero-row artifact is canonical and digest-pinned. Under the frozen protocol,
only independently verified `PASS` / `PROMOTED` trajectories can become desired
LoRA targets. Because there were none, starting LoRA optimization, fabricating
an adapter, substituting base-model output for a trained arm, or reporting a
trained-versus-base held-out comparison would be invalid. No research adapter
was created and the trained arms remain unevaluated.

This exact-source gate is historical evidence of the zero-row rejection, not
the still-pending hardened pre-model proof. Its machine record deliberately
sets `accepted_pre_model_rejection=false`, `pre_model_no_model_claimed=false`,
and `hardened_overlay_pending=true`: the original source ordered evaluator,
CUDA, and model initialization before the task-count rejection. The final
hardened overlay must be tested separately on an exact committed revision and
must prove rejection before evaluator, CUDA, model loading, or output creation.
Until that separate artifact exists, no stronger pre-model claim is made.

## What the result supports

This campaign is evidence against the sufficiency of the privately sealed
tested configuration: its pinned Qwen model, source-only prompt/decoding
contract, fixed B/D agent loops, retrieval surface, evaluator, and 12-attempt
budget did not produce a single correct promotable candidate across the
80-request matrix. The public bundle cross-binds the exact source commit across
the freezer, diagnostic, and both Spark acceptance records, and the diagnostic
publishes its model revision/manifest. It does not independently reproduce the
primary campaign's private prompt, decoding, evaluator, or protocol manifests.
The formatting diagnostic further indicates that transport repair alone is not
the missing ingredient in its separate frozen cutoff-57 census.

The result does not establish that evidence-governed variation, retrieval,
DAG-based control, LoRA, Qwen models generally, or the underlying research
theory cannot work. It does not compare B with D causally, because both arms had
zero successes. It makes no model-quality, safety, utility, cost, novelty, or
production-readiness claim.

A scientifically valid follow-up must be preregistered as a new campaign. It
may change the base model, candidate scaffold, prompt contract, budget, or
treatment, but must not retroactively alter this frozen negative result.

## Public-safe attestation artifacts

- `primary-aggregate-80of80.json` — terminal, attempt, evaluator, evidence-use,
  and integrity aggregates; SHA-256
  `32370e4c85f52d1d6a77a718a41dff77b969582af8e0579894c07463b56b35b2`.
- `freezer-gate-80of80.json` — copied-ledger integrity, exact freezer digests,
  exclusions, row count, and bounded training-gate status; current SHA-256
  `36b86371819baa699eab2af20156892b8aaedf143ab2b92c0b2a31925db4419b`.
- `r5-full-census.aggregate.json` — matched cutoff-57 formatting diagnostic;
  SHA-256 `dfff9543251c58406f12e3aa49b493dd3c852823a7061ab2364f188a50027b95`.
- `spark1-cpu-qdrant-smoke.json` and `spark2-cpu-qdrant-smoke.json` — independent
  deterministic CPU and local Qdrant acceptance only. These do not claim model
  inference or cross-node trainer/evaluator transport.

Every public JSON artifact uses a closed schema and is scanned for raw
prompt/source fields, private identifiers, credentials, paths, endpoints, and
topology. The final artifact validator pins the exact artifact bytes and must
pass before publication. Because row-level inputs and raw execution outputs are
redacted, this bundle validates the exposed bytes, schema, bindings, and
self-consistency; it cannot independently recompute the private campaign result.
