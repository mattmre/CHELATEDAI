# Paired chelation intervention sanity protocol v2

**Protocol ID:** `CHELATEDAI-PRW-ISI1-PAIRED-SANITY-v2`

**Status:** frozen evidence-integrity reconditioning of v1; dependency-light
synthetic sanity only

## 1. Preserved predecessor

Version 1 remains byte-preserved at
`docs/research/paired-chelation-intervention-sanity-protocol-2026-08.md` with
SHA-256
`75aab66730a5d4ef4055ce66086705c6fd79c4b057807fdba20cd4181bd11ce9`.
Its two retained output directories remain historical audit evidence and are
not rewritten.

Tier-B returned v1 at 70/critical because it had no verifier, could overwrite
a nonempty output directory nontransactionally, trusted claimed result
semantics during artifact construction, and called a declaration-only feature
count a second-order intervention. V2 changes evidence integrity and wording;
it does not change the frozen numeric retrieval fixture, masks, policy
predictions, or bounded scientific disposition.

## 2. Scope and unchanged fixture

V2 retains v1's exact deterministic configuration:

- four topic axes plus one collapse/nuisance axis (`dimension=5`);
- four nuisance pairs, four ordinary material pairs, and one additional
  material pair (`pair_count=9`);
- no-projection, oracle-nuisance-mask, production-variance-chelation, and
  over-chelation policies;
- `collapse_strength=4.0`, `nuisance_multiplier=1.5`, and `chelation_p=85`;
- the exact paired metrics, constant-output over-chelation control, resource
  ceilings, 30-second cooperative deadline, and claim boundaries from v1.

The production and oracle masks are expected to equal `[1,1,1,1,0]` on this
constructed fixture. Passing therefore validates fixture, mask, and metric
semantics only. It is not natural-language, model, corpus, RAG, training,
utility, causal-mechanism, or novelty evidence. Full `PRW-ISI1` remains
`BLOCKED_ON_PRW-EK3`.

## 3. Corrected intervention-count semantics

`changed_features` is declared metadata supplied with each synthetic case. V2
does not infer it from the numeric query transformation and does not call its
length an intervention order. The retained metric is named
`declared_changed_feature_count`, and the optional grouping is named
`by_declared_changed_feature_count`.

The additional material pair declares two metadata labels but performs the
same class of one-topic-coordinate replacement as the ordinary material
pairs. It is retained only as another safety-critical material case. It is not
evidence of a second-order, factorial, coalition, or interacting intervention,
and no v2 sanity gate depends on the declared count.

## 4. Artifact semantics

The only valid v2 artifact is the deterministic result regenerated from the
frozen default `PairedInterventionBudget` and exact fixture. Artifact
construction must reject a caller-supplied result unless its complete
canonical structure equals an independently regenerated result under that
budget.

Required claim fields are exact:

- artifact and result status: `COMPLETE`;
- evidence state: `VALIDATED` only when every frozen v2 sanity predicate is
  true, otherwise `REJECTED`;
- scientific and novelty claim status: `UNCONFIRMED`;
- production path changed: `false`;
- model or corpus loaded: `false`.

Every numeric leaf must be finite. The artifact digest is SHA-256 of the exact
canonical artifact payload excluding only the `artifact_digest` field.

## 5. Publication transaction

The official runner accepts only an absent output path. Existing files,
directories, symlinks, junction-like paths, and nonempty targets are rejected
before executing the fixture. It writes `paired_intervention_sanity.json` and
`manifest.json` into one unique sibling staging directory, verifies the full
staged set, and atomically renames that directory to the final absent path.
Any failure before rename removes the staging directory and leaves no final
evidence. Successful directory rename is the publication commit point; no
fallible evidence mutation follows it.

Public verification rejects hidden staging paths and requires exactly two
ordinary nonsymlink files. It checks canonical newline-terminated UTF-8 JSON,
exact schemas, the v1 predecessor digest, this v2 protocol digest, artifact
digest, file SHA-256 and byte count, all claim boundaries, the exact regenerated
artifact, and the exact regenerated manifest. Unexpected files or coherent
resealing are rejected.

## 6. Official v2 output

The v2 output directory is:

```text
artifacts/method-dev/isi1-paired-intervention-sanity-v2/
```

Version 1 directories are preserved and must not be relabeled as verified v2
evidence. V2 does not upgrade the scientific finding: the constructed
production mask still matches the oracle and the over-chelation control still
shows why invariance without material response is insufficient.

## 7. Frozen disposition

If the exact v2 artifact verifies, its status is
`VALIDATED_SYNTHETIC_SANITY_ONLY`. This means the evaluator rejects a dead
constant-output policy on the exact fixture. It does not promote `PRW-ISI1`,
does not establish second-order intervention handling, and does not supply
independent SELECT/REPORT evidence.
