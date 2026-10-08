# Qwen-Scope nuisance-rank separation follow-up

**Status:** `REVIEW-DRAFT / NOT FROZEN / NOT AUTHORIZED FOR EXECUTION`

**Scientific and novelty status:** `UNCONFIRMED`

## Motivation and contamination boundary

QSCCI v4 produced a valid negative result. Its nuisance-penalized and ordinary
contrastive selectors chose the same four features and therefore the same
direction. The shared direction strongly exceeded random controls, but no
chelation-specific effect was identified.

Both v4 SELECT and REPORT evidence have now been inspected. That fixture is
exhausted for confirmatory use. Any rerun on it is exploratory replication
only. This follow-up requires a wholly new fixture frozen before target-model
or SAE probing. The lambda grid and hypotheses below are explicitly motivated
by the exploratory v4 result.

## Prospective fixture

An independent authoring pass must create three mutually disjoint partitions:

- DEVELOPMENT: 12 balanced base rows, implementation checks only;
- SELECT: 24 balanced base rows, deterministic configuration selection; and
- REPORT: 24 balanced base rows, inaccessible until selection is locked.

Each base row has a canonical form, a label-preserving nuisance transform, and
a label-changing material transform. Before execution, freeze all texts,
labels, transformations, ordering, exclusions, and hashes. Reuse, deterministic
paraphrases, and model-independent near-duplicates of v4 rows are prohibited.
Authors must remain blind to target-model activations, SAE features, and
outcomes. A protocol change after DEVELOPMENT requires a new version and fresh
untouched SELECT and REPORT partitions.

The independent blind-fixture authoring pass is complete. The bindings below
were inserted only after the authored bytes and their model-independent
validator were fixed. The fixture author did not inspect any prior QSCCI
fixture, protocol, result, artifact, model activation, SAE feature, or service.
These bindings do not by themselves authorize runner implementation or
execution; this document remains a review draft until its final mechanical
freeze.

| Binding | Required frozen value |
|---|---|
| DEVELOPMENT fixture path / SHA-256 | `docs/research/qscci-blind-followup-fixture-v1-development.json` / `6ea3a5f927be266074a3875d5e793b1a801688970c3ce0cd6f7ca549422c88f5` |
| SELECT fixture path / SHA-256 | `docs/research/qscci-blind-followup-fixture-v1-select.json` / `18197b51af39d81639ab730675330507c2b78fb58f74c3ae7dc36096550a59df` |
| REPORT fixture path / SHA-256 | `docs/research/qscci-blind-followup-fixture-v1-report.json` / `26263306fbcfedbe106d6335d434f960b6128c8e2b6894b31d188b0b505de64e` |
| Blind authoring source / SHA-256 | `docs/research/qscci-blind-followup-fixture-review-draft-v1.json` / `877e2a27ec42afeb2a0b3100f76576aa45fee1c435d08e4748c71efb6338820c` |
| Prompt template bytes / SHA-256 | exact 171 UTF-8 bytes below, without BOM or trailing LF / `54bc90ed5be052328c82dadae8fb86279668f8be51958b887f2bb5ecb1c5c013` |
| Blind-source schema authority / SHA-256 | `validate_qscci_blind_followup_fixture.py` / `43f869fe86d7a75ea1b839d926b5cdcd88077a3c0508b17956cba19adb50e148` |
| Frozen split/schema transform / SHA-256 | `freeze_qscci_blind_followup_fixture.py` / `13f83591cbb0ff09c3cb30849b62a00e29531a891660e0a23cf6e3ee8105594a` |
| Human-readable fixture contract / SHA-256 | `docs/research/qscci-blind-followup-fixture-review-draft-v1.md` / `aebca1acbb81d89b985b589841908dc78d5d9eb7fddec303bea2d3db036948e1` |
| Blind hostile-test suite / SHA-256 | `test_qscci_blind_followup_fixture.py` / `3635daa391127dbedfc5702c2ebf6dedbcf1748c9ea773066700c685ea5669eb` |
| Split and cross-fixture boundary tests / SHA-256 | `test_qscci_followup_fixture_boundary.py` / `600a82bf2aeac662e83893ff37931ca51412f728319aee4dbe73efa744fc6100` |
| Near-duplicate normalization, metric, and threshold | Unicode NFKC, then `casefold()`; left-to-right Unicode `L*`/`N*` tokens with following `M*` marks; multiset Sorensen-Dice; reject similarity `>= 0.86` |

The exact prompt template is the following content with LF separators. It has
one literal `{text}` marker and no trailing LF:

```text
Determine whether the proposed conclusion is logically forced by all stated facts.
Respond with positive if it is forced, otherwise respond with negative.

{text}

Answer:
```

Fixture labels map mechanically as `YES -> +1 -> " positive"` and
`NO -> -1 -> " negative"`. Canonical and nuisance labels must be equal;
the material label must be the opposite. No other label mapping is allowed.
The bound cross-fixture test performs all `180 * 36 = 6,480` comparisons
between the new fixture's three variants and every SELECT/REPORT variant in
the exhausted v4 fixture. The maximum observed similarity is the exact
token-count fraction `8/26 = 4/13` (binary64 `0.3076923076923077`), below the
frozen inclusive rejection threshold.

The three frozen capability inputs are canonical compact sorted-key JSON,
UTF-8 without BOM, LF-only, with one final LF. Each binds its source digest,
partition identity, exact rows, policies, and label mapping and has status
`FROZEN`. The combined `review-draft`-labelled file remains immutable authoring
custody evidence; it is not passed to DEVELOPMENT, SELECT, or REPORT. The
frozen inputs must contain exactly 12 DEVELOPMENT, 24 SELECT, and 24
REPORT base rows, respectively balanced `6/6`, `12/12`, and `12/12` between
labels `-1` and `+1` after the mapping above. Reformatting, key reordering, or
semantic resealing changes a bound digest and is not permitted.

## Inherited v4 bindings

Unless this document explicitly replaces feature ranking, control construction,
selection, or REPORT comparisons, all QSCCI v4 definitions remain binding.
The follow-up must reject any mismatch in these inherited identities:

| Item | Frozen value |
|---|---|
| v4 protocol ID | `CHELATEDAI-QSCCI-v4` |
| v4 core SHA-256 | `0659c40efa4866486f94a0c2470c0b0af7ba1064aba3d8c39206814914c1b4db` |
| v4 runner SHA-256 | `29d91881ca825b3bf37f37b81cc8702e4b6d9cb911a25056639fefa70361b9f0` |
| v4 erratum SHA-256 | `b28e978fc61238df9439348355da92d95329092794b12b754bcb6c990e8aa27e` |
| v4 artifact-schema SHA-256 | `a30fee50976ebc1d04c04d9ca7c1f1cc2c37cb1caed06ac56aae77e1d2a5eb46` |
| immutable v1 protocol SHA-256 | `f7417b022dd93b96d523f6e8ca4a12c8a915288b5b4621e9ef130ac4f52848f1` |
| immutable v1 schema SHA-256 | `032f06fefc53aa04f13a410ce5aec93b7039c7c042bd0a2cedfe8f4745cb653a` |
| model / revision | `Qwen/Qwen3.5-2B-Base` / `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` |
| SAE / revision | `Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100` / `027267657257a8d490296286e8fab41e1c1a1a3d` |
| SAE file / SHA-256 | `layer11.sae.pt` / `d1828ace348b13cca9104f61fb47672e439e963d9d5fc5496f4c6b068a06499f` |
| hook / widths / TopK | post-layer-11 residual / `2048 / 32768 / 100` |
| target strings / token IDs | `" positive"`, `" negative"` / `6572`, `7968` |

The new fixture hashes and the eventual follow-up protocol, runner, and schema
hashes are additional bindings; they do not replace these ancestry bindings.
Inherited `Q` is exactly
`Decimal.from_float(float(x)).quantize(Decimal("0.000000001"),
rounding=ROUND_HALF_EVEN)`. It is applied only after the declared binary64
aggregate or complete candidate-minus-control difference has been computed.

### Exact cardinality, order, and batching overrides

The new partition sizes replace every inherited v4 literal tied to six rows or
18 prompts. Within each frozen partition, row order is the JSON array order
and variant order is exactly canonical, nuisance, material. DEVELOPMENT has
12 rows and 36 prompts. SELECT and REPORT each have 24 rows and 72 prompts.
No ID sort may replace the frozen array order.

On SELECT, `contrast_j` and `nuisance_j` each average exactly 24 row terms in
frozen row order. Eligibility uses exactly the 48 canonical/material prompts;
`active` means the feature appears in deterministic TopK100 for at least 12 of
those 48 prompts. Nuisance prompts remain excluded from eligibility. Mean
absolute activation and activation-decile construction also use exactly those
48 canonical/material prompts. The retained sparse SELECT evidence has exactly
72 ordered prompt rows, each with 100 feature IDs and signed FP32 values, and
reconstructs one implicit `72 x 32768` activation matrix.

Residual scale is partition-local. Sort all 72 SELECT residual RMS values in
ascending FP32 order, convert zero-based elements 35 and 36 to binary64,
average them in binary64, and round once to FP32. REPORT repeats that exact
72-value operation using REPORT only; it does not reuse or change the locked
SELECT `alpha`. DEVELOPMENT uses its 36 values and zero-based elements 17 and
18 only for implementation validation and never contributes to selection or a
scientific endpoint.

Every SELECT/REPORT prompt-level endpoint, strict accuracy, mean KL, maximum
KL population, collateral fraction, and per-prompt norm contract contains
exactly 72 prompt instances. A row endpoint reduces its three prompts first
and then reduces exactly 24 rows. Material completeness has denominator 24.
One microbatch contains all 36 DEVELOPMENT prompts or all 72 prompts for the
relevant SELECT/REPORT partition; right padding and all inherited deterministic
runtime settings remain binding. Missing, duplicate, reordered, or cross-phase
prompt records are `INVALID_RUN`. The inherited wall, RSS, CUDA allocated,
CUDA reserved, disk, service-restoration, and external-supervisor ceilings are
unchanged and apply separately to each phase; a failed preflight or measured
breach is `INVALID_RUN`, never a scientific failure.

## Ranking treatment

Retain v4 eligibility, raw material score `m_i`, raw nuisance score `n_i`,
contrast sign, model, layer, SAE, and intervention definitions. Change only
feature ranking.

For the `N` eligible SELECT features, let raw `m_i` and `n_i` be the finite
binary64 values produced by inherited v4 ordered `math.fsum` reductions. Raw
equality means Python binary64 `==`; neither raw score is quantized before
equality classes are counted. Define integer midrank numerators:

```text
rM_i = 2 * count(m_j < m_i) + count(m_j = m_i)
rN_i = 2 * count(n_j < n_i) + count(n_j = n_i)
M_i  = rM_i / (2N)
N_i  = rN_i / (2N)
```

Every rank, rank mean, and rank gate is exact rational arithmetic over Python
integers. Encode `lambda` as the reduced integer pair `(p,q)` in this exact
order:

```text
(0,1), (1,4), (1,2), (1,1), (2,1), (4,1), (8,1), (16,1)
k      in {1, 4, 8}
alpha  in {0.5, 1, 2, 4}
```

For fixed `(p,q)`, rank by the exact integer
`T_i=q*rM_i-p*rN_i`, which is proportional to
`S_lambda(i)=M_i-lambda*N_i`. Sort by descending `T_i`, descending `rM_i`,
ascending `rN_i`, then ascending integer feature ID. No floating-point rank or
lambda operation is permitted. `lambda=(0,1)` is the pure contrastive control.
Retain `rM_i`, `rN_i`, `N`, `p`, `q`, and `T_i` as JSON integers; derived
fractions are serialized as reduced `{numerator,denominator}` objects, never
JSON floats.

Feature `i` has locked sign `+1` when inherited v4 `contrast_i > 0` and `-1`
when it is `< 0`; exact zero remains ineligible. Its decoder direction is v4's
finite positive-norm `W_dec[:,i]` column, unit-normalized, multiplied by that
locked sign. Selected signed columns are summed in ascending feature-ID order
with the inherited FP32 construction and normalized once. REPORT must reuse
the SELECT feature IDs, per-feature signs, aggregate direction, `k`, and
`alpha` byte-for-byte; REPORT activations cannot change any of them.

## SELECT manipulation and admission gates

A `(lambda,k)` treatment with `lambda>0` is admissible only relative to
`lambda=0` at the same `k`. Let `A` be the treatment top-k set and `B` the
contrastive top-k set. All set cardinalities are exact integers. Let
`mean_R(X)=sum_{i in X} rR_i/(2Nk)` for `R in {M,N}`, reduced exactly. Define
all improvements as control minus treatment unless explicitly called a causal
or completeness difference:

- replacement fraction `|A symmetric_difference B| / (2k) >= 1/2`;
- Jaccard `|A intersect B| / |A union B| <= 1/3`;
- `abs(canonicalize_cosine(cos(d_A,d_B))) <= 0.95`, inclusive, using the v3/v4
  `1e-12` cosine envelope and clamp;
- nuisance-rank reduction `mean_N(B)-mean_N(A) >= 1/10`;
- material-rank loss `mean_M(B)-mean_M(A) <= 1/10`; and
- raw nuisance reduction
  `Q(mean_raw_nuisance(B)-mean_raw_nuisance(A)) > 0.000000000`.

Raw selected means use `math.fsum` over ascending feature ID and divide in
binary64 before applying inherited v4 `Q`. Every displayed decimal threshold
is inclusive except the explicitly strict raw-nuisance comparison. A nonfinite
operand invalidates the run.

These are manipulation checks, not scientific success. If no treatment is
admissible, stop as `NO_SEPARATED_NUISANCE_RANKING_ON_SELECT` and never open
REPORT.

At matched `k,alpha`, an admissible treatment must then qualify on SELECT:

- mismatch reduction
  `Q(mismatch_contrastive-mismatch_treatment) >= 0.050000000`;
- causal-contrast difference
  `Q(causal_treatment-causal_contrastive) >= -0.050000000`;
- completeness difference
  `completeness_treatment-completeness_contrastive >= 0` using exact row-count
  fractions;
- unsteered strict full-vocabulary baseline accuracy `>= 0.75`;
- `Q(mean_correct_gain) > 0.000000000` and
  `Q(mean_wrong_gain) < 0.000000000`; and
- mean KL `<= 0.050000000` nat, maximum KL `<= 0.250000000` nat, and
  outside-target top-1 collateral-change fraction `<= 1/6`.

Choose exactly one treatment lexicographically by largest mismatch reduction,
largest causal contrast, largest material completeness, lowest mean KL, lower
`lambda` by exact rational comparison, lower `k`, then lower `alpha`. Endpoint
keys use inherited `Q`; exact equality advances to the next key. Publish and
hash-bind the full SELECT
grid and the chosen configuration in a selection receipt before REPORT may be
loaded. The runner must mechanically refuse REPORT without that receipt.

If separated treatments exist but none qualifies, stop as
`NO_SELECT_QUALIFYING_TREATMENT` and never open REPORT.

## Deterministic controls

All controls are constructed from SELECT only after the treatment is locked.
Seeds are exactly `1701`, `1709`, and `1721`. SHA-256 inputs are UTF-8 ASCII;
integers are unsigned base-10 without leading zeros; raw 32-byte digests are
ordered lexicographically, and a digest collision is broken by lower feature
ID. Duplicate feature selection is never permitted within a control draw.

For nuisance permutation seed `s`, sort eligible source feature IDs ascending,
sort the same IDs by
`(SHA256("nuisance_permutation|s|feature_id"), feature_id)`, and assign the
`rN` value at ascending-source position `r` to the feature at digest-order
position `r`. Rank with the locked treatment `(lambda,k)` using the permuted
`rN`, while retaining each feature's own `rM` and own SELECT contrast sign.
Each seed produces one frozen top-k set. A permutation that is the identity or
collides in top-k set with another required nuisance control is retained,
flagged, and makes the run `INVALID_RUN`; it is never silently redrawn.

Activation deciles and random sampling inherit v4 exactly, except that the
locked treatment top-k supplies the activation-decile multiset. Family strings
are exactly `activation_matched_random` and `uniform_random`; digest inputs are
`family|seed|k|feature_id`. A random top-k collision with another required
random control or the treatment is retained and flagged but is not invalid;
overlap is an intended measured property. A digest collision uses lower
feature ID. Decile shortfall, fewer than eight eligible features, duplicate
selection, or a missing draw is `INVALID_RUN`. Every random feature uses its
own SELECT contrast sign.

## Selection receipt and REPORT barrier

The SELECT process receives no REPORT path, bytes, labels, or handle. It writes
only a hidden staging directory containing canonical JSON files. Required
files are:

- `select-grid.json`: every candidate, including inadmissible and nonqualifying
  cells, exact rank integers/fractions, feature IDs/signs, endpoints, gates,
  and deterministic-control definitions;
- `selection-lock.json`: the one locked treatment, contrastive comparator, all
  nine control feature sets/signs, exact `k`, `alpha`, lambda pair, directions,
  and selection key; and
- `SELECT-COMMIT.json`: the receipt and sole SELECT commit point.

`SELECT-COMMIT.json` is a closed object with exactly these keys and types:
`record_type` string `qscci_nuisance_rank_selection_receipt`;
`schema_version` string; `protocol_sha256`, `runner_sha256`, and
`result_schema_sha256` lowercase 64-hex strings; `inherited_v4` closed object;
`fixture_bindings` closed object containing all eleven frozen fixture-boundary
bindings; `model_identity` and `sae_identity` closed objects; `members` closed
object with exactly `select-grid.json` and `selection-lock.json`, each mapped
to a closed `{byte_size: nonnegative integer, sha256: lowercase 64-hex}` object;
`status` string exactly `SELECT_LOCKED`; `chosen_candidate_id` nonempty ASCII
string; `seeds` integer array exactly `[1701,1709,1721]`; `lifecycle_nonce`
lowercase 32-hex string; `report_opened` Boolean exactly `false`; and
`receipt_sha256` lowercase 64-hex string. The internal digest is computed over
the canonical object with only `receipt_sha256` omitted. Every nested object is
closed by the eventual schema. Unknown or missing keys invalidate the receipt.

The producer writes and fsyncs the two data files, writes and fsyncs a hidden
receipt candidate, fsyncs the staging directory, then atomically publishes the
receipt with same-filesystem `os.replace` and fsyncs the directory again. The
receipt's appearance is the SELECT commit point. Overwrite, symlinks, hard
links, non-regular files, stale targets, path escapes, or a pre-existing final
receipt are refused.

REPORT runs in a new process. Before its first REPORT file open, it must verify
canonical bytes, closed schemas, hashes, sizes, nonce, status, all bindings,
the complete grid, deterministic reconstruction of the winner and controls,
and `report_opened:false`. It then creates an exclusive regular-file
`REPORT-OPENED` marker containing the receipt SHA-256 and lifecycle nonce,
fsyncs it and its directory, and only then opens the exact hash-bound REPORT
path. Missing/invalid receipt, a pre-existing marker, or any attempted REPORT
access before that marker is `INVALID_RUN`. REPORT cannot invoke ranking,
selection, sign choice, direction construction, or tuning code.

## Locked REPORT evaluation

Run only the locked treatment and these controls, all at the treatment's exact
`k,alpha` with no family-specific retuning:

- pure contrastive `lambda=0`;
- three frozen deterministic nuisance-score permutations;
- three activation-matched random controls; and
- three uniform-random controls.

Confirmatory success is the conjunction of:

- SELECT and REPORT baseline accuracy `>= 0.75`;
- treatment-versus-contrastive Jaccard is `<= 1/3` and absolute canonicalized
  direction cosine is `<= 0.95`, both inclusive;
- for contrastive and each nuisance permutation `c`,
  `Q(mismatch_c-mismatch_treatment) >= 0.050000000`;
- for every comparator `c` in the exact set consisting of contrastive, all
  three nuisance permutations, all three activation-matched random controls,
  and all three uniform-random controls,
  `Q(causal_treatment-causal_c) >= -0.050000000`;
- material completeness is `>= 5/6` and no lower than every control;
- for every activation-matched or uniform-random control `r`,
  `Q(causal_treatment-causal_r) >= 0.050000000`;
- for every random control `r`,
  `Q(mismatch_r-mismatch_treatment) >= 0.000000000`;
- correct-direction gain is positive and wrong-direction gain is negative; and
- mean KL `<= 0.050000000` nat, maximum KL `<= 0.250000000` nat, and
  outside-target top-1 collateral-change fraction `<= 1/6`.

Completeness comparisons use exact row-count fractions; the treatment must be
at least `5/6`, and `completeness_treatment-completeness_control >= 0` for every
control. A complete row retains v4's absolute requirements: canonical and
material correct-sign gains are each `>= 0.020000000` logit and row material
bidirectional causal contrast is `>= 0.040000000` logit. Correct-direction gain
means `Q(mean_correct_gain) > 0.000000000`; wrong-direction gain means
`Q(mean_wrong_gain) < 0.000000000`. All inequalities are inclusive except
these two explicit sign tests.

Failure of any conjunct falsifies the treatment on this fixture. The v4
absolute completeness threshold remains binding and may not be weakened.

## Anti-cherry-picking and interpretation

Publish every SELECT candidate, including inadmissible candidates, the locked
selection receipt, and every REPORT control. Never substitute another lambda,
seed, `k`, or `alpha` after REPORT begins. Do not promote a secondary endpoint
after primary failure. A completed negative run cannot be replaced by another
seed or fixture. Infrastructure reruns are allowed only when fail-closed
evidence proves that no REPORT scientific leaves were exposed.

Terminal statuses are mutually exclusive and use this precedence:

1. `INVALID_RUN`: invocation, binding, identity, arithmetic, nonfinite,
   resource, infrastructure, restoration, phase-barrier, incomplete
   output/artifact, or integrity failure, regardless of any visible endpoint;
2. `INVALID_TASK`: no earlier invalidity and SELECT or REPORT unsteered strict
   full-vocabulary baseline accuracy is `< 0.75`;
3. `NO_SEPARATED_NUISANCE_RANKING_ON_SELECT`: SELECT-valid but no treatment
   passes every manipulation check;
4. `NO_SELECT_QUALIFYING_TREATMENT`: at least one separated treatment exists
   but none passes every SELECT qualification gate;
5. `DOES_NOT_SURVIVE_NUISANCE_RANK_FIXTURE`: REPORT was validly opened and at
   least one confirmatory conjunct failed; or
6. `SURVIVES_NUISANCE_RANK_FIXTURE`: every confirmatory conjunct passed.

Statuses 3 and 4 prohibit REPORT access. `SELECT-COMMIT.json` exists only on
the REPORT-eligible path and has status `SELECT_LOCKED`; the final artifact and
final commit receipt repeat the one terminal status. Their disagreement is
`INVALID_RUN`.

The committed inventory is phase-dependent and closed. A valid SELECT-terminal
status (`INVALID_TASK` detected before REPORT, status 3, or status 4) contains
exactly `select-grid.json`, `result.json`, `manifest.json`, and `COMMIT.json`;
it contains no selection receipt, REPORT marker, or REPORT records. A
REPORT-terminal status contains exactly `select-grid.json`,
`selection-lock.json`, `SELECT-COMMIT.json`, `REPORT-OPENED`,
`report-records.json`, `result.json`, `manifest.json`, and `COMMIT.json`.
`REPORT-OPENED` is canonical JSON. `manifest.json` binds path, byte size, and
SHA-256 for every preceding member.
`COMMIT.json` is the sole final atomic publication point and binds the manifest,
terminal status, lifecycle nonce, and internal result digest. Canonical JSON,
safe-path, regular-file, no-link, refuse-overwrite, staging, fsync, atomic
rename, quarantine-on-failure, and independent-verifier requirements inherit
v4. A verifier recomputes every rank, selection, control, endpoint, gate, and
status from retained leaves; model-execution-attested leaves remain bounded as
in v4. `INVALID_RUN` never publishes `COMMIT.json`; its stage is quarantined.
Material-response completeness below `5/6` is a scientific conjunct failure
and therefore status 5 after a valid REPORT; it is never the incomplete-output
condition in status 1. If SELECT baseline accuracy is `<0.75`, candidate
evaluation does not begin: `select-grid.json` contains its exact identity and
binding header, the 72 ordered unsteered prompt records, baseline endpoint,
zero candidate cells, zero control definitions, and status `INVALID_TASK`.

Even a pass would establish only prospective separation and robustness on one
model, layer, SAE, and small authored fixture. It would not establish general
utility, training-cost reduction, architectural novelty, or a new theory of
intelligence. Independent replication across new fixtures, models, layers,
and SAE checkpoints remains required.

## Required work before freeze

This draft remains `REVIEW-DRAFT / NOT FROZEN / NOT AUTHORIZED FOR EXECUTION`.
The independent fixture authoring and adversarial review are complete: the
fixture validator passes, its 24 hostile unit tests pass, the four integration
tests pass, the label balances and prompt mapping are bound above, and the
cross-fixture near-duplicate boundary passes. No placeholder remains.

The only remaining freeze step is a fresh whole-document exactness review of
these integrated bytes. If that review approves them, perform a mechanical
freeze into new, never-reused follow-up protocol and schema identities, bind
their exact hashes in the runner and tests, and rerun all validators before any
DEVELOPMENT model execution. SELECT and REPORT remain prohibited until the
mechanical phase barriers defined above are implemented and independently
disproved.

For equal-size top-k sets, replacement fraction `>=0.5` and Jaccard `<=1/3`
are algebraically equivalent; at `k=1` they require total replacement. Both
gates are intentionally retained as independently reconstructed audit fields,
and the `k=1` total-replacement behavior is intentional. Disagreement between
their Boolean outcomes is `INVALID_RUN`.
