# QSCCI Blind Follow-up Fixture v1

Status: **review-draft**. It is not frozen and is not evidence of model behavior.

## Blind scope and task

This fixture was authored without consulting any prior QSCCI fixture, protocol, result, or artifact. Its creation and validation use no model, SAE, GPU, network, or service. For each self-contained item, answer `YES` exactly when the proposed conclusion is logically forced by all stated facts; otherwise answer `NO`.

The JSON has 60 stably ordered rows. IDs are opaque random-looking tokens: they contain neither partition nor ordinal nor label. Each partition owns a fixed, validator-pinned ID sequence and a separately fixed, validator-pinned non-alternating label sequence. This describes the committed contract, not provenance for how a shuffle was generated.

| Partition | Rows | YES | NO | Rows per logic family |
|---|---:|---:|---:|---:|
| DEVELOPMENT | 12 | 6 | 6 | 2 |
| SELECT | 24 | 12 | 12 | 4 |
| REPORT | 24 | 12 | 12 | 4 |

Every row contains three distinct texts: canonical, label-preserving nuisance, and label-inverting material. Nuisance variants change presentation, ordering, vocabulary, names, or formatting while retaining all load-bearing facts. Material variants change a load-bearing fact or proposed conclusion. Labels are explicit.

## Exact schema and policy

The validator rejects missing or unknown fields. The top-level fields are exactly:

```text
fixture_id, schema_version, status, task, partition_order,
label_order_policy, near_duplicate_policy, structural_policy, rows
```

Every row has exactly:

```text
id, partition, logic_family, logic_subtype, surface_signature,
canonical_text, canonical_label, nuisance_text, nuisance_label,
material_text, material_label
```

Policy objects are closed values, not extension points. `schema_version` is integer `2`; status is `review-draft`; labels are `YES` or `NO`; all text and identity fields are strings. The validator pins the exact task, partition order, label policy, duplicate policy, structural policy, ID order, and ID-to-partition mapping.

## Structural leakage control

Lexical distance alone cannot prevent a label from correlating with a single reasoning template. Every partition therefore has an equal, label-balanced quota for six distinct logic families:

- transitive chains;
- conjunctive rules;
- disjunctive elimination;
- temporal ordering;
- cardinality constraints;
- mutual exclusion.

Within each partition and family, half the rows are `YES` and half are `NO`. The label order is neither alternating nor encoded by ID or ordinal. The validator rejects family quota, family-label balance, ID mapping, or fixed pinned label-order drift.

Each row also declares a closed `logic_subtype` and a mechanically derived `surface_signature`. A canonical text must contain the contiguous normalized token pair `proposed conclusion` exactly once; `proposed answer`, separated tokens, missing pairs, and duplicate pairs are rejected. The signature is `NEGATED_CONCLUSION` exactly when tokens after that unique pair contain `not`; otherwise it is `AFFIRMATIVE_CONCLUSION`.

For every family and partition, the validator requires identical subtype and surface-signature distributions for `YES` and `NO`. Cardinality subtype is derived only from the first rule sentence: exactly one contiguous `exactly when` pair and no threshold pair means `EXACT_COUNT`; exactly one `at least` or `at most` pair and no `exactly when` pair means `THRESHOLD`; every other pattern is rejected. Marker words in later fact or note sentences therefore cannot authorize a subtype or relabel. Conjunctive canonical texts mechanically forbid the token `but`; this removes the reviewed surface cue rather than merely documenting it.

## Near-duplicate gate

`validate_qscci_blind_followup_fixture.py` is dependency-free and model-independent:

1. Apply Unicode NFKC, then Unicode `casefold()`.
2. Scan left to right. Unicode letters (`L*`) and numbers (`N*`) extend a token. Combining marks (`M*`) extend a token only after a letter or number. Every other code point separates tokens; empty tokens are discarded.
3. Treat tokens as multisets. Similarity is multiset Sorensen-Dice: `2 * sum(min(A[t], B[t])) / (sum(A.values()) + sum(B.values()))`. Two empty streams score `1.0`.
4. Compare canonical, nuisance, and material variants across distinct base-row IDs. Designed siblings are excluded from cross-row similarity but must be distinct after normalization/tokenization.
5. Reject similarity greater than or equal to `0.86`; the boundary is inclusive.

The validator handles malformed JSON/root types without a traceback and additionally enforces exact schemas, types, opaque IDs, stable order, ID ownership, fixed pinned label sequences, family and subtype balances, derived surface signatures, sibling distinctness, transform-label relations, and an eight-token minimum.

Run the bounded checks with:

```text
python validate_qscci_blind_followup_fixture.py
python -m unittest test_qscci_blind_followup_fixture.py -v
python -m ruff check validate_qscci_blind_followup_fixture.py test_qscci_blind_followup_fixture.py
```
