# Paired chelation intervention sanity protocol

**Protocol ID:** `CHELATEDAI-PRW-ISI1-PAIRED-SANITY-v1`  
**Stage ID:** `PRW-ISI1-PAIRED-SANITY`  
**Frozen:** 2026-08-14, before the retained runner artifact  
**Parent card:** `PRW-ISI1`  
**Parent dependency status:** unchanged; full `PRW-ISI1` remains blocked on
`PRW-EK3`.

## Question and scope

Does exact paired evaluation add a useful guard to ChelatedAI's original
dimension-binding idea?

The guard separates two requirements that an ordinary retrieval score can
confound:

1. a nuisance-only change must preserve the answer; and
2. a material change must change the answer in the declared way.

This is an evaluation-axis sanity check, not a new representation-learning
method. It does not execute the blocked `PRW-ISI1` controls, train a model,
load a corpus, or make a scientific or novelty claim. The public Sophontic
video and evaluation page independently motivate the exact-pair framing, but
contrast sets and behavioral perturbation testing predate both projects:

- <https://www.youtube.com/watch?v=4S8I22ybG2c>
- <https://sophontic.ai/evals/>
- <https://aclanthology.org/2020.findings-emnlp.117/>
- <https://aclanthology.org/2020.acl-main.442/>

## Frozen fixture

The deterministic fixture has four semantic topic axes and one high-amplitude
nuisance axis. It contains:

- one relevant document per topic;
- one nuisance-aligned collapse distractor;
- four nuisance pairs that change only nuisance amplitude while retaining the
  relevant topic;
- four singleton material pairs that change the relevant topic; and
- one two-atom material pair so the evaluator cannot silently assume that all
  interventions are singleton.

The calibration cluster varies the nuisance axis much more than the semantic
axes. The production `AntigravityEngine._chelate_toxicity` method is invoked
without constructing an engine, model, vector database, or learned predictor.

## Frozen controls

Controls run in this order:

1. `no_projection`: all dimensions retained;
2. `oracle_nuisance_mask`: only the declared nuisance axis removed;
3. `production_variance_chelation`: mask returned by the existing production
   variance-chelation method at `chelation_p=85`; and
4. `over_chelation`: all dimensions removed, producing a constant-output
   failure control.

The oracle is a ceiling/sanity control, not a deployable method. The
production control is not trained or selected on outcome labels.

## Frozen metrics

For every pair, both the canonical and perturbed predictions must equal their
declared labels. Relation-only agreement is reported separately because a
constant-output system can appear perfectly nuisance-invariant while missing
every material intervention.

The aggregate reports:

- canonical, perturbed, and strict paired accuracy;
- nuisance relation accuracy and nuisance violation rate;
- material relation accuracy and missed-intervention rate;
- strict nuisance-pair and material-pair accuracy;
- the harmonic mean of the two strict group accuracies as
  `balanced_joint_score`;
- their minimum as `joint_floor`;
- safety-critical miss count; and
- strict accuracy by declared intervention order.

The balanced score is zero if either group has zero strict accuracy. Both pair
kinds are required; an evaluator input containing only one kind is rejected.

## Required sanity boundary

The harness is valid only if all of the following hold:

1. the production variance mask is exactly equal to the oracle nuisance mask;
2. the production and oracle controls produce identical predictions;
3. the production control has strict paired accuracy, balanced joint score,
   and joint floor equal to `1.0` on the frozen fixture;
4. it detects every frozen safety-critical material intervention;
5. the no-projection control scores below the production control; and
6. the over-chelation control has perfect relation-only nuisance invariance
   but zero material relation accuracy and zero balanced joint score.

Failure rejects the sanity harness or the fixture assumption. Passing does
not establish an advantage on natural language, live embeddings, RAG, unseen
interventions, or independent data.

## Resource and durability contract

The first pre-retention focused test rejected the draft 10-second ceiling
because a cold import of the existing production engine crossed it. No
official artifact had been retained. The ceiling was reconditioned to 30
seconds and explicitly includes that cold import; the numerical fixture itself
remains unchanged.

- Python standard library plus NumPy and the repository's existing production
  import surface only;
- dimension at most 64 and pair count at most 5,000;
- modeled allocation at most 64 MiB;
- cooperative deadline at most 30 seconds, including the first cold import of
  the production engine surface;
- no model, corpus, GPU, network, or vector database;
- atomic canonical JSON output; and
- retained output under
  `artifacts/method-dev/isi1-paired-intervention-sanity/` with byte count and
  SHA-256 in its manifest.

## Claim boundary

The only possible positive statement from this protocol is:

> The exact-pair evaluator and the existing production variance mask behave
> consistently on the frozen synthetic chelation fixture, and the evaluator
> catches a constant-output over-chelation failure.

Scientific status and novelty status remain `UNCONFIRMED`. Full `PRW-ISI1`
candidate-survival work remains blocked on `PRW-EK3` and requires a separate
SELECT/REPORT protocol, independent transformation/evaluator generators, and
a confirmation set.
