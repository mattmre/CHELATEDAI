# BCC-1 preregistered confirmatory protocol

**Protocol ID:** `CHELATEDAI-BCC1-v1`

**Protocol status:** draft retained after METHOD_DEV cut; no confirmatory
`REPORT` exists or may be opened from the current feature family

**Freeze date target:** 2026-07

**Primary metric:** binary nDCG@10, higher is better

**Scientific claim status:** unconfirmed

**Novelty claim status:** unconfirmed and deliberately narrower than any single
bridge, conversion, backfilling, or routing technique

## 1. Brutally honest status

The repaired 12-block artifact is **METHOD_DEV only**. It is sampled,
transductive, uses the same rows for descriptive selection and evaluation, has
no disjoint `REPORT`, and is explicitly non-promotional. It cannot confirm
BCC-1. Its current feature and policy family is cut. The recovered inputs also
contain only old/new document-role anchors: the oldward matrix is fitted on
document embeddings and then applied to query embeddings. Its stored
`reverse` arm is therefore an `UNVALIDATED_DOCUMENT_ROLE_REVERSE_PROXY`, not
a validated reverse-query bridge. That is an independent protocol cut.

The regenerated pack
`bcc1-method-dev-derivation-d5f8ce2b43ebe9f47d414d0bc7bda19325dedf25dde4fe9a002f3eaa61b4e1c8`
contains 720 query-block observations but only 240 independent dataset-query
groups: 4 datasets (`arguana`, `fiqa2018`, `nfcorpus`, and `scifact`) crossed
with 3 target encoder families. Pack SHA-256 is
`19acc83532b53a103da350fa37d82b724e9aa929778bcea43940d96b7ccdb5a2`,
manifest SHA-256 is
`d0280f7d8fbfa397651adebc13210ed70d3e86030afdf61bd3170a379404728f`,
and result SHA-256 is
`25b345eb3233b7d82d684b355d7d12ba41efa75577870e89afa35e05408094b7`.
The path-bound one-shot lock is
`docs/bcc1-method-dev-manifest-2026-07.json.bcc1.lock`, SHA-256
`0f41ff9eb08a5b5d56129472ae8c12740532b23f6d21bff0f752b7c2ab72b72e`;
it records `METHOD_DEV_COMPLETE` and `report_outcomes_opened=false`.
Its aggregate means are:

| Strategy/reference | Mean binary nDCG@10 |
|---|---:|
| Forward document conversion | 0.528363905261 |
| Document-role-derived reverse proxy | 0.460306747309 |
| Ex-post per-query maximum | 0.571264636826 |
| Native-new retrieval | 0.651793476970 |
| Unconverted mismatch | 0.008544128231 |

The descriptive ex-post choice-regret upper bound is 0.042900731565 above the
better observed fixed direction. It is a max-after-outcomes statistic, not a
deployable or inferential effect. All 12 sampled blocks have positive raw
headroom, ranging from 0.013813201871 to 0.078409180214. There are 275 forward
wins, 150 reverse-proxy wins, and 295 ties.

Nineteen of the 21 qrels-free features use resident-corpus scores, ranks,
cross-route comparisons, or half-corpus anchor scans. They require both route
probes and are now declared `DUAL_READ_METHOD_DEV` with cost class
`dual_search_probe`. Only query-cycle error and reverse-query norm ratio are
`PRE_SEARCH`/`zero_search`. A result selector that has already run both
retrievals cannot implement the question in Section 2 or the resolver's
one-authorized-search abstention semantics.

The grouped structural harness also fails to recover the raw proxy envelope even
with the richer dual-read surface. The disposition is
`CUT_ROLE_MISMATCH_AND_DUAL_READ_FEATURE_SET`: logistic regression effectively falls back
to the fixed route, while 5-NN is negative under the dataset-family and crossed
transition/query-group schemes. The operational feature-stage violation is an
independent cut. No confirmatory universe, qrels, or `REPORT` may be created or
opened from this derivation.

| Policy and structural scheme | Lift vs fold-fixed route | One-sided LCB | Non-abstained coverage |
|---|---:|---:|---:|
| Logistic, leave dataset family out | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| Logistic, crossed transition/query fold | 0.000000000000 | 0.000000000000 | 0.019444444444 |
| 5-NN, leave dataset family out | -0.001560515325 | -0.007684386566 | 0.520833333333 |
| 5-NN, crossed transition/query fold | -0.001753665104 | -0.012838525276 | 0.490277777778 |

All partitions were valid, group-disjoint, and evaluated exactly once. Zero
policies passed both schemes.

The current 12 blocks, their qrels, and every feature invented after inspecting
them are permanently classified as `METHOD_DEV`. They may be used to
recondition or cut the method, but never as `SELECT` or `REPORT` evidence for
this protocol.

## 2. Question and formal hypothesis

BCC-1 asks a narrow operational question:

> For a query encoded in a new representation space and an existing semantic
> cache/index encoded in an old space, can a frozen, qrels-free, per-query
> policy safely choose between a zero-write reverse-query bridge and an
> exactly materialized forward-document bridge better than the best fixed
> direction on entirely unseen dataset and encoder-transition families?

For REPORT structural block \(b\), query \(q\), strategy \(s\), let
\(Y_{bq}^{s}\) be binary nDCG@10. Define:

- \(A_r\): transform the new-space query into the resident old query space and
  search the unchanged resident index;
- \(A_f\): search an exact, materialized target-space index produced by
  transforming every document in the resident corpus;
- \(A_g\): the single better fixed direction selected using `SELECT` only,
  with an exact tie resolved to `reverse`;
- \(A_\pi\): the frozen per-query router; if its confidence is below 0.80, it
  falls back to \(A_g\), but only when that fixed route is independently
  ABI-valid and authorized;
- \(A_n\): native-new retrieval after re-encoding the complete same corpus in
  the target document space;
- \(A_o\): the per-query direction oracle
  \(\max(Y^{A_r},Y^{A_f})\), used only as a descriptive upper bound.

The primary estimand is the equal-cell macro effect across the crossed REPORT
dataset families \(D_R\) and encoder-transition families \(T_R\):

\[
\Delta =
\frac{1}{|D_R||T_R|}
\sum_{d \in D_R}\sum_{t \in T_R}
\frac{1}{n_{dt}}\sum_q
\left(Y_{dtq}^{A_\pi}-Y_{dtq}^{A_g}\right).
\]

Each `independence_group_id` represents exactly one dataset query. There must
be exactly one row for that group in each structural
`role × dataset × transition` cell, and every transition within a
role/dataset must expose the identical group set. Duplicate rows within a cell
fail validation. Under this invariant, the query-weighted cell mean and the
equal-weighted independence-group cell mean are identical; repeated views
remain coupled across transitions during resampling.

The primary null is \(H_0: \Delta \le 0\). The directional alternative is
\(H_1: \Delta > 0\). The preregistered minimum worthwhile effect is
\(\delta_{\text{MWEE}}=0.005\) absolute binary nDCG@10. The scientific
prediction is therefore both statistical superiority and
\(\widehat{\Delta}\ge0.005\); neither condition alone is a pass.

For query \(q\), binary relevance is `1` exactly when its frozen qrel is
strictly greater than zero. At rank \(i\), gain is
`relevant_i / log2(i + 1)`. DCG is the sum through rank 10; IDCG uses the same
formula with all positive documents first; nDCG is `DCG / IDCG`. Queries with
zero positive documents are ineligible rather than assigned an invented zero.
Ranking is descending cosine score with canonical document ID as the final
tie-break. Every accepted artifact must content-bind the metric name, cutoff,
binary gain, all-positive-qrels IDCG population, query-inclusion rule, qrels
snapshot SHA-256, metric implementation SHA-256, tie-break, and pack/qrels
binding. A label such as `nDCG@10` without that lineage is insufficient. The
shared legacy metric defect and regeneration policy are specified in
`docs/research/metric-lineage-repair-protocol-2026-07.md`.

The hypothesis concerns safe operational composition. It does not assert that
reverse translation, forward conversion, per-query routing, representation
metadata, or backfilling is individually new.

## 3. Representation ABI and fail-closed cache semantics

Every vector-producing or vector-consuming component must carry an immutable,
content-addressed representation descriptor. Its identity includes, at
minimum:

- encoder ID, exact revision, and weights SHA-256;
- tokenizer ID and SHA-256;
- query/document/shared role;
- compatibility-domain ID and SHA-256;
- instruction SHA-256, pooling, dimension, preprocessing ID and SHA-256;
- projection and adapter chain IDs and SHA-256 values;
- quantization ID and SHA-256;
- normalization, metric, dtype, and ABI version.

Every resident or materialized index descriptor must additionally bind the
document and query representation IDs, corpus snapshot ID and SHA-256,
vector-build ID and SHA-256, exact document count, index ID, and contract
version.

Each bridge is directional and evidence-bound. Its contract fixes source and
target descriptor IDs, role, transform family and ID, weights and
hyperparameter SHA-256 values, external-anchor manifest SHA-256, fit and
validation counts, validation state, cost class, evidence ID, and validation
confidence.

The resolver semantics are:

1. Exact query-space equality with the resident query-space ID permits
   `NATIVE`.
2. `REVERSE` requires a validated query-role bridge from the exact incoming
   query descriptor to the exact resident query descriptor, with
   `zero_write` cost. It searches the unchanged resident index.
3. `FORWARD` requires a validated document-role bridge from the exact resident
   document descriptor to the target document descriptor, with
   `materialized_index` cost. The materialized index must name that exact
   bridge, source resident-index ID, corpus SHA-256, target query/document
   spaces, and document count.
4. Unknown ABI versions, descriptor/hash mismatches, wrong roles or direction,
   wrong cost classes, unvalidated/rejected/retired bridges, missing or stale
   materializations, unknown route IDs, non-finite scores, scorer exceptions,
   and confidence below the frozen threshold produce `ABSTAIN`.
5. `ABSTAIN` does not authorize search. A statistical fallback to \(A_g\) is
   executable only if \(A_g\)'s route independently satisfies rules 2 or 3.
   Otherwise the final operational result remains `ABSTAIN`.
6. Validation confidence caps per-query confidence. Both must be at least
   0.80.

The operational conformance gate is zero wrong-space searches across the full
adversarial resolver suite. A single case that searches after an invalid
descriptor, stale materialization, router exception, or explicit abstention
fails BCC-1 regardless of retrieval metrics.

## 4. Directional strategies and bridge construction

The confirmatory bridge family is frozen before `SELECT` outcomes are opened:

- **Forward:** fit a document-role ridge map from old document anchors to new
  document anchors, then transform and L2-normalize every resident document.
- **Reverse:** fit a query-role ridge map from new query anchors to old query
  anchors, then transform and L2-normalize each incoming query.
- **Ridge definition:** for source matrix \(X\), target matrix \(Z\), and
  identity matrix \(I\), \(W=(X^\top X+1.0I)^{-1}X^\top Z\), computed in
  float64. No REPORT-derived refit or regularization change is permitted.
- **Retrieval:** cosine similarity over normalized vectors, exact full-corpus
  top 10, stable descending-score order with canonical document ID as the
  deterministic tie-breaker.

Role-specific anchor encodings are mandatory. A document-role transform may
not be reused as a query-role transform when the descriptor roles,
instructions, or preprocessing differ.

Each bridge must pass an external-anchor validation before it can be marked
`VALIDATED`:

1. At least 4,096 unique fit anchors and 1,024 unique validation anchors.
2. Anchor train/validation assignment is the first 80%/last 20% after sorting
   normalized anchor text by
   `SHA256("bcc1-v1-anchor|" + normalized_text)`.
3. The anchor corpus is independently sourced, SHA-bound, and contains no
   normalized text from any evaluated query, corpus document, or qrel record.
   It is not selected using retrieval outcomes.
4. On validation anchors, mean matched-pair cosine after transformation is at
   least 0.80, its one-sided 95% bootstrap lower bound is at least 0.80, and
   the one-sided 95% lower bound for its paired improvement over the
   untransformed mismatch is greater than 0.
5. The bridge's validation confidence is that mean-cosine lower bound, clipped
   to [0, 1]. A bridge below 0.80 is not a candidate.

The current METHOD_DEV builder's corpus-derived "leakage-safe matched anchors"
and sampled packs do not satisfy this external-anchor contract.

## 5. Eligible universe and structural holdout

A structural block is the content-addressed tuple:

`dataset_family_id × source_representation_family_id ×
target_representation_family_id × corpus_snapshot_sha256 × bridge_protocol_id`.

Aliases, revisions, or mirrors of the same dataset are one dataset family.
Models sharing the same underlying training lineage are one representation
family. The family map, its evidence, and all exact descriptor IDs are frozen
before any `SELECT` or `REPORT` outcome is read.

Eligibility requires:

- a full, unsampled corpus snapshot;
- the full set of judged queries having at least one positive qrel in that
  corpus, with at least 200 such queries per cell;
- exact pinned encoder/tokenizer revisions available locally;
- both directional bridges passing Section 4;
- native-new full-corpus retrieval for the same query and document snapshot;
- no dataset, query, corpus snapshot, encoder transition, or qrel previously
  used as METHOD_DEV for the eventual REPORT families.

The frozen universe must contain at least 8 eligible dataset families and 6
eligible ordered encoder-transition families. Role assignment is mechanical:

1. Sort dataset families by
   `SHA256("bcc1-v1-dataset-family|" + dataset_family_id)`. Using zero-based
   ranks, even ranks are `SELECT_DATASET`; odd ranks are `REPORT_DATASET`.
2. Sort ordered transition families by
   `SHA256("bcc1-v1-transition-family|" + transition_family_id)`. Using
   zero-based ranks, even ranks are `SELECT_TRANSITION`; odd ranks are
   `REPORT_TRANSITION`.
3. `SELECT` is the complete cross-product of `SELECT_DATASET` and
   `SELECT_TRANSITION`.
4. `REPORT` is the complete cross-product of `REPORT_DATASET` and
   `REPORT_TRANSITION`.
5. Mixed-role cells are `RESERVE` and are not accessed in BCC-1 v1.

This yields at least 12 SELECT and 12 REPORT cells, at least 4 unseen REPORT
dataset families, and at least 3 unseen REPORT transition families. Every
REPORT cell is held out along both structural axes, not merely by query row.
The universe manifest and role allocation are canonical JSON, SHA-bound, and
committed before `SELECT` analysis.

## 6. No-qrels and no-REPORT-leakage boundary

The primary BCC-1 router authorizes exactly one resident-corpus search.
Inference features must therefore be available **before either candidate
search runs**. They may use query vectors, frozen bridge matrices,
external-anchor vectors and similarities, bridge reconstruction or cycle
residuals, representation descriptors, and non-outcome operational metadata.
They may not use resident-index candidate IDs, document score vectors,
top-\(k\) scores or margins, rank overlap, cross-route score agreement, or any
other quantity that requires executing either candidate retrieval. Feature
names, sources, execution phase, and computations must be in a frozen
manifest. Qrel counts, relevance labels, nDCG, winners, oracle regret, native
outcomes, or any transformations of them are also forbidden.

A policy that first executes both candidate searches and then chooses a result
set is a **dual-read result selector**, not the one-route BCC-1 policy. It
requires a separate typed authorization for both reads, a latency/cost
estimand, and post-search abstention semantics. It cannot be promoted under
this protocol or used to support the zero-write one-search operational claim.
The repaired METHOD_DEV pack's score, margin, top-\(k\), and rank-agreement
features are in this dual-read category. They remain useful only as a cut
diagnostic.

`SELECT` qrels may be used only for fixed-direction selection, router fitting,
and the prespecified METHOD_DEV-to-confirmatory gates below. `REPORT` qrels and
all scores derived from them must not be present in any artifact accessible to
method developers before the following are frozen and SHA-bound:

- representation, index, anchor, and bridge manifests;
- ordered feature manifest and feature implementation SHA;
- SELECT/REPORT structural allocation;
- fitted scaler, model parameters, confidence rule, fixed fallback, and every
  per-REPORT-query route decision;
- environment lock, exact runner SHA, seeds, estimands, and decision gates.

An independent evaluator or equivalent access boundary joins frozen REPORT
decisions to sealed REPORT qrels only after those hashes exist. A JSON parser
that merely delays reading REPORT score fields inside a file already visible
to method developers is not a sufficient confirmatory access boundary.

Every REPORT block is one-shot. Successful or failed evaluation atomically
records `consumed=true` and the output report SHA-256 in its manifest. A crash
after REPORT outcome access is still consumption; recovery may replay only
from the already-frozen decision artifact, never refit or choose a model.

## 7. Frozen router family and abstention

The only BCC-1 v1 candidate routers over the frozen **pre-search** feature
manifest are:

1. L2-regularized logistic regression with standardized SELECT features,
   \(L2=1.0\), at most 100 Newton iterations, and convergence tolerance
   \(10^{-10}\).
2. Euclidean 5-nearest-neighbor classification over the same standardized
   features, with canonical query ID as the deterministic final tie-breaker.

The training label is `reverse` only when reverse nDCG@10 is strictly greater
than forward nDCG@10 by more than \(10^{-12}\); it is `forward` for the
opposite inequality. Direction ties are excluded from classifier fitting but
remain in every policy-score denominator.

Candidate evaluation on SELECT uses two outer structural schemes:

1. **Leave one dataset family out.** Train on every other dataset family and
   evaluate the held-out dataset family. Because independence-group IDs are
   dataset-family scoped, no evaluation group may occur in training.
2. **Leave one transition family and one query-group fold out.** Within each
   dataset family, sort its unique independence-group IDs by
   `SHA256("bcc1-v1-query-fold|" + dataset_family_id + "|" +
   independence_group_id)`, breaking an impossible digest tie by the literal
   group ID. Assign zero-based sorted rank modulo 5 as `query_group_fold`.
   For every ordered pair `(heldout_transition, heldout_group_fold)`, evaluate
   rows having both values and train only on rows whose transition differs
   from the held-out transition **and** whose query-group fold differs from
   the held-out fold. Aggregate the out-of-fold predictions so each SELECT row
   is evaluated exactly once.

The second scheme is required because the same underlying dataset query is
observed under every transition. A literal leave-one-transition-out fit would
see the held-out queries under other transitions and leak query-specific
outcomes; excluding all repeated query groups would instead leave zero
training rows. The crossed five-fold construction tests an unseen transition
and unseen query groups simultaneously while retaining a non-overlapping
training partition.

Within each fold, fit the scaler and classifier and select the fixed fallback
using only the training partition. Apply those frozen objects to the held-out
partition. The held-out outcomes may score the fold but may not affect its
scaler, classifier, confidence, or fallback. Every fold records train/eval
group IDs and must prove their intersection is empty. An empty partition, a
training partition without both strict direction labels, fewer than five
non-tied training examples for 5-NN, a missing out-of-fold prediction, or any
group overlap fails that candidate and structural scheme closed.

For each candidate, compute macro lift over the SELECT-chosen fixed direction
under both schemes. Its robust selection score is the smaller of those two
lifts. The candidate with the larger robust score is selected; an exact tie
selects logistic regression. It is then refit once on all SELECT rows.

For logistic regression, confidence is `max(p(reverse), 1-p(reverse))`. For
5-NN, confidence is the winning vote fraction. Confidence below 0.80 abstains
to the SELECT-chosen fixed route, subject to the ABI authorization rule in
Section 3. Soft fusion, additional models, feature selection using REPORT,
threshold tuning, per-dataset tuning, and per-encoder tuning are outside the
BCC-1 v1 confirmatory family.

Before a REPORT artifact may be generated, the selected candidate must have:

- point lift at least 0.005 in both SELECT structural schemes;
- a one-sided 97.5% hierarchical lower bound greater than 0 in both schemes;
- non-abstained coverage at least 10% overall and 5% in every SELECT block;
- no SELECT block with point loss below -0.02.

Failure stops before REPORT. It is a method-selection failure, not permission
to inspect REPORT and recondition.

## 8. Power and minimum worthwhile effect

The REPORT power gate is computed using SELECT only, after the router is frozen
and before REPORT qrels or outcomes are available.

1. Form paired **out-of-fold** SELECT residuals
   \(R_{dtq}=Y_{dtq}^{A_\pi}-Y_{dtq}^{A_g}\) from the frozen structural scheme
   selected in Section 7. Do not predict SELECT rows with a model fitted on
   those same rows for the power calculation. Bind every OOF decision,
   fallback, group assignment, and residual to the power artifact.
2. Center residuals within each structural cell and multiply the centered
   residuals by 1.25 as a preregistered variance inflation.
3. Simulate the declared qrels-free REPORT design, not the SELECT shape. Use
   the exact count of REPORT dataset families, REPORT transition families, and
   eligible query groups in every REPORT cell from the frozen universe
   manifest. For each replicate, sample SELECT residual cells as a variance
   library with replacement across the corresponding dataset and transition
   axes, then sample centered OOF residuals to the declared group count of each
   REPORT cell. Missing REPORT membership, an empty source cell, or a
   non-crossed REPORT allocation fails the power gate closed.
4. Use 10,000 Monte Carlo replicates with seed 1730. The null critical value is
   the 97.5th percentile of resampled null macro effects. Estimated power at
   the MWEE is the fraction of the same residual replicates shifted by +0.005
   that exceed that null critical value.
5. Required power is at least 0.80.

If power is below 0.80, do not open REPORT. The only allowed response is to
freeze a larger eligible universe under a new protocol minor version, repeat
the mechanical family allocation, and rerun the power calculation without
REPORT outcomes. Lowering the MWEE, confidence level, variance inflation, or
structural holdout requirements after seeing power is forbidden.

## 9. Confirmatory inference and safety boundaries

The actual REPORT primary interval uses 10,000 crossed hierarchical bootstrap
replicates with seed 1729: independently resample REPORT dataset families and
transition families, cross them, and then resample queries within each selected
cell. The one-sided 97.5% lower confidence bound is the 2.5th percentile.

### Primary superiority and practical-effect gates

- one-sided 97.5% lower bound for \(\Delta\) is greater than 0; and
- \(\widehat{\Delta}\) is at least 0.005.

### Native-new quality boundary

Define

\[
\Gamma =
\operatorname{macro}_{d,t,q}
\left(Y_{dtq}^{A_\pi}-Y_{dtq}^{A_n}\right).
\]

The routed compatibility path is non-inferior to native-new only if the
one-sided 97.5% lower bound for \(\Gamma\) is greater than -0.02. This boundary
prevents a small gain over a weak compatibility baseline from being presented
as acceptable when rebuilding natively would recover materially more quality.

### Worst-block safety

For every REPORT cell \(b\), estimate
\(\Delta_b=\operatorname{mean}_q(Y_{bq}^{A_\pi}-Y_{bq}^{A_g})\) using 10,000
query bootstrap replicates. Use one-sided Bonferroni-adjusted 95% familywise
lower bounds with per-block alpha `0.05 / number_of_REPORT_blocks`. Each bound
must be greater than -0.02. The seed for block \(b\) is the unsigned integer
encoded by the first 8 hex digits of
`SHA256("bcc1-v1-safety|" + block_id)`.

### Abstention and operational safety

- non-abstained coverage is at least 10% overall and 5% in every REPORT block;
- every abstained policy decision executes only an ABI-valid fixed fallback or
  returns operational `ABSTAIN` with no search;
- all ABI/resolver adversarial cases pass with zero wrong-space searches;
- canonical replay from the frozen inputs reproduces the exact report SHA-256.

Query-weighted effects, direction-oracle headroom captured, reverse-route rate,
native-old comparisons, latency, write amplification, and cost are secondary
descriptive estimands. They cannot rescue a failed primary or safety gate.

## 10. One-shot decision rule

BCC-1 v1 passes only if all of the following are true:

1. all artifact hashes, exact revisions, full-corpus requirements, external
   anchors, and structural roles validate;
2. the anti-leakage and independent REPORT access boundary is documented and
   independently reviewed;
3. the pre-REPORT SELECT transfer and power gates pass;
4. the primary statistical and 0.005 MWEE gates pass;
5. the native-new non-inferiority boundary passes;
6. every worst-block safety boundary passes;
7. abstention coverage and fail-closed operational conformance pass;
8. deterministic replay produces the exact report hash.

If a valid one-shot REPORT is consumed and any item 4 through 8 fails, record
`BCC1_V1_REJECTED`. Do not tune on that REPORT. Any new attempt requires a new
protocol ID, new feature/policy hash, and untouched dataset and transition
families.

If items 1 through 3 fail before REPORT access, record
`BCC1_V1_BLOCKED_PROTOCOL`; the scientific hypothesis remains untested. Passing
all gates makes the implementation eligible for an adversarial promotion
review. It does not itself authorize deployment, establish novelty, or support
a broad claim about computing or humanity.

## 11. Reconditioning and cut rules

The current 21-feature policy family has already been reproduced into a
canonical, SHA-bound METHOD_DEV pack and has already been cut. That derivation
is complete; it cannot be renamed, rescored, or promoted after this protocol
was written.

Any continuation starts a new METHOD_DEV derivation ID. At most three new
derivation cycles are allowed before the BCC-1 v1 policy lane is cut. Each
cycle must:

1. predeclare one mechanistic, qrels-free, **pre-search** failure-fingerprint
   feature family and its implementation hash before evaluating that
   derivation;
2. retain exactly the Section 7 leave-dataset-family-out and crossed
   transition-family/query-group-fold evaluations;
3. report every block, abstention rate, selected direction rate, descriptive
   oracle headroom, and lift over the same fold-selected fixed baseline; and
4. advance only if both structural schemes meet the Section 7 pre-REPORT
   gates.

Permissible primary mechanisms include externally anchored
out-of-distribution distance and bridge residual/cycle-consistency diagnostics
that are computable before either resident-index search. Cross-direction rank
instability requires both candidate retrievals and may appear only as a
`DUAL_READ_METHOD_DEV` diagnostic; it is not an eligible primary BCC-1
pre-search feature. Outcome proxies, report-family identifiers,
dataset-specific rules, post-hoc thresholds, and undeclared search probes are
forbidden.

If the same transfer failure survives two consecutive new derivations, stop
immediately rather than spending the third. If no new derivation passes after
three cycles, record `BCC1_ROUTER_CUT`. The representation ABI and
deterministic fixed-route resolver may remain useful independently, but there
is no evidence for the query-conditional BCC-1 hypothesis.

After REPORT consumption there is no reconditioning within v1. A failed native
boundary means cut the operational-promotion claim even if router lift is
positive. A failed worst-block or fail-closed gate means cut deployment
eligibility even if average lift is positive.

## 12. Prior-art boundary and novelty falsifier

A non-exhaustive primary-source audit dated 2026-07-23 already identifies:

- reverse query translation in
  [Drift-Adapter](https://aclanthology.org/2025.emnlp-main.805/) and
  [Query Drift Compensation](https://arxiv.org/abs/2506.00037), plus
  query-side cross-space transformation in
  [STEER](https://arxiv.org/abs/2507.18518) and
  [Trans-RAG](https://arxiv.org/abs/2604.09541);
- forward embedding conversion in
  [Embedding-Converter](https://aclanthology.org/2025.acl-long.1237/) and
  [Forward Compatible Training](https://arxiv.org/abs/2112.02805);
- bidirectional or merged upgrade paths in
  [BiCT](https://arxiv.org/abs/2204.13919) and
  [Online Backfilling with No Regret](https://openreview.net/pdf?id=KoSduEDl5M),
  plus separate old/new retrieval paths, distance rank merge, and a reverse
  transformation in
  [Metric Compatible Training](https://openreview.net/forum?id=1RmeKnkwsB);
- uncertainty-guided or selectively compatible backfill in
  [FastFill](https://arxiv.org/abs/2303.04766) and
  [Darwinian Model Upgrades](https://arxiv.org/abs/2210.06954);
- query-specific retriever routing in
  [Mixture of Retrievers](https://aclanthology.org/2025.emnlp-main.601/) and
  [R3AG](https://aclanthology.org/2026.acl-long.939/); and
- production index identity using an index-definition hash and version in
  [MongoDB automated embedding](https://www.mongodb.com/docs/vector-search/crud-embeddings/automated-embedding/overview/);
- an explicit embedding ABI, directional compatibility operators, and
  confidence-gated transformed-query routing to the old index with fallback to
  dual-read or the target index in
  [vectormigrate](https://github.com/goutamadwant/vectormigrate/blob/main/docs/paper_system_model.md);
- backfill-free backward-compatible search plus optional gradual backfill in
  issued patent
  [US11216697B1](https://patents.google.com/patent/US11216697B1/en); and
- translation and predetermined or dynamically determined traversal across
  embedding spaces in pending application
  [US20260111496A1](https://patents.google.com/patent/US20260111496A1/en),
  which records a 2024-10-22 priority date.

The machine-readable audit is
`docs/research/bcc1-prior-art-manifest-2026-07.json`, SHA-256
`a44fab246246feca4e5e6589934ede623bffd4639251d414608ae254f2c91994`.

Therefore none of those constituents may be described as the BCC-1 novelty.
The closest systems disclosure found is vectormigrate. The narrow remaining
distinction under investigation is not generic ABI, translation, confidence
routing, fallback, or backfill. It is evidence-bound choice between a
zero-write reverse-query route and a separately materialized
forward-transformed legacy corpus, with fail-closed authorization and
untouched two-axis one-shot evaluation. Even that combination is
**unconfirmed**. The patent entries are search evidence, not legal analysis;
this protocol makes no patent, priority, infringement, validity, or other legal
opinion.

Before any novelty statement, freeze a bibliography manifest and complete
backward/forward citation searches from every source above. The novelty lane is
falsified and cut if a primary source predating this protocol discloses, in one
system:

1. exact/versioned query, document, and resident-index representation identity;
2. evidence-bound directional bridge contracts;
3. per-query choice between zero-write reverse translation and an exactly
   materialized forward index;
4. fail-closed abstention for identity, evidence, confidence, or materialization
   failures; and
5. an operational evaluation separating method development from structurally
   unseen evidence.

If the falsifier is met, retain only an honest replication, integration, or
benchmarking contribution if the evidence warrants it. Performance success
cannot override prior art. Conversely, failure to find the combination in this
non-exhaustive audit is not proof of novelty.

## 13. Exact execution plan and required artifacts

The current derivation terminates after step 4:

1. **Recover and reproduce METHOD_DEV — complete.** The builder ignores the
   inherited qrels-derived fit index, uses deterministic qrels-independent
   anchors, and assigns stable independent query groups across transitions.
2. **Classify execution stages — complete.** Two features are pre-search;
   nineteen are dual-read probes. The latter cannot support the primary
   operational claim.
3. **Run structural transfer — complete.** Evaluate both frozen policy families
   with leave-dataset-family-out and crossed transition/query-group folds,
   fitting scalers, fallbacks, and policies inside each training partition.
4. **Cut — complete.** Record
   `CUT_ROLE_MISMATCH_AND_DUAL_READ_FEATURE_SET`, preserve the
   pack/result/checksums, and stop. The recovered oldward route is explicitly
   a document-role reverse proxy, and nineteen features require dual reads. No
   confirmatory universe or REPORT may be built from this derivation.

Any continuation is a new METHOD_DEV derivation with a new ID and these
preconditions:

5. **State a new pre-search feature hypothesis.** Use only query, bridge, ABI,
   and explicitly external calibration evidence that can be computed before
   either resident-corpus search. Bind feature type, source, cost, and
   implementation. Do not reuse this derivation's outcome-driven feature
   choices as confirmatory evidence.
6. **Repeat both structural gates.** Advance only if one policy independently
   satisfies both Section 7 transfer schemes, coverage, effect, lower-bound,
   and worst-block gates. Otherwise cut the new derivation too.
7. **Freeze implementation.** Record repository commit, environment lock,
   descriptor/bridge schemas, runner SHA, feature code SHA, seeds, and this
   protocol SHA.
8. **Freeze the universe.** Produce
   `docs/bcc1-confirmatory-universe-v1.json` with family evidence, exact
   descriptor IDs, corpus hashes, eligibility, mechanical roles, and its own
   SHA-256.
9. **Build external bridges.** Produce an anchor manifest, validation report,
   and immutable bridge contracts. Reject any bridge below Section 4.
10. **Build full-corpus SELECT.** Fit/select the router, freeze
   `docs/bcc1-policy-v1.json`, and run the Section 7 gate.
11. **Run power.** Write a canonical power artifact. If it fails, do not create
   or unlock REPORT.
12. **Freeze REPORT decisions.** Generate qrels-free REPORT features and route
   decisions; commit their IDs and SHA values before the evaluator can access
   REPORT outcomes.
13. **Consume REPORT once.** The independent evaluator joins sealed outcomes,
   writes the canonical report, and atomically consumes every REPORT block.
14. **Replay and disprove.** A fresh reviewer attempts artifact substitution,
    qrel leakage, family leakage, stale materialization, wrong-space search,
    selective block omission, seed drift, and decision-rule drift.
15. **Record the result.** Use exactly `BCC1_V1_REJECTED`,
    `BCC1_V1_BLOCKED_PROTOCOL`, or `BCC1_V1_PASSED_REVIEW_ELIGIBLE`. Never
    silently convert a failed gate into a caveat.

The full-corpus run must use the exact pinned encoder revisions. If those
weights are not locally available, execution is blocked until their hashes are
restored or an explicit, audited download is authorized. Substituting a
different revision creates a different representation descriptor and requires
regenerating the frozen universe; it is not a transparent retry.
