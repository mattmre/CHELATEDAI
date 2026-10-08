# Paired chelation intervention sanity protocol v3

**Protocol ID:** `CHELATEDAI-PRW-ISI1-PAIRED-SANITY-v3`

**Status:** frozen final evidence-integrity reconditioning; dependency-light
synthetic sanity only

## Preserved lineage

V1 protocol SHA-256 is
`75aab66730a5d4ef4055ce66086705c6fd79c4b057807fdba20cd4181bd11ce9`.
Git retains one v1 output directory containing its artifact and manifest. An
empty local replay sibling is not evidence and is not claimed.

V2 protocol SHA-256 is
`0e6938ec6b6b3f5ade0ea1e82cf38d60aece8850d3a2121339de5c28ce28bda1`.
Its output remains byte-preserved. V2 repaired semantic regeneration,
canonical verification, and absent-target transactional publication, but its
protocol incorrectly required a result-level `status` that its retained result
did not contain. Its public verifier also did not reject Windows junctions,
allowing a hidden staging copy to be exposed through a public-named reparse
path. V3 corrects both without rewriting v1 or v2 bytes.

## Unchanged synthetic experiment

V3 retains the exact v2 numeric fixture, budget, four policies, masks,
predictions, metrics, limitations, and scientific disposition. It makes no
new model, corpus, RAG, training, utility, causal-mechanism, second-order, or
novelty claim. Declared changed-feature counts remain metadata only. Full
`PRW-ISI1` remains `BLOCKED_ON_PRW-EK3`.

The deterministic result now explicitly contains `status="COMPLETE"`. The
artifact also contains `status="COMPLETE"`. Evidence state is `VALIDATED` only
when every frozen synthetic sanity predicate is true. Scientific and novelty
claim statuses remain `UNCONFIRMED`.

## Verification and publication

V3 binds the exact v1, v2, and v3 protocol SHA-256 values in the artifact and
manifest. Artifact construction accepts only the complete result regenerated
from the frozen default budget and fixture. The artifact digest excludes only
its own digest field. Public verification requires canonical newline-
terminated UTF-8 JSON, exact regenerated artifact and manifest, exact digests,
and exactly two ordinary files.

On Windows, the root and every member are rejected when `os.lstat()` reports
`FILE_ATTRIBUTE_REPARSE_POINT` (`0x400`), covering junctions and other reparse
objects that `Path.is_symlink()` misses. On every platform, resolved member
paths must remain direct children of the resolved result root. Symlink or
reparse roots and members are invalid. Hidden `.stage-*` paths are never public
evidence, including when reached through an alias.

The runner accepts only an absent target, writes both files into a unique
sibling staging directory, verifies the staged set, and atomically promotes it
without replacement. Any pre-promotion failure removes the stage and leaves no
final result. Successful promotion is the publication commit point.

## Official v3 output and disposition

The official output is a new directory:

```text
artifacts/method-dev/isi1-paired-intervention-sanity-v3/
```

A verified output has manifest status
`VALIDATED_SYNTHETIC_SANITY_ONLY`. This validates only that the exact evaluator
and production variance mask pass the constructed fixture while the dead
over-chelation control fails its material-response endpoint. It does not
promote the parent research card.
