# Latent option value audit

**Audit ID:** `CHELATEDAI-LOVA-2026-07`

**Audit date:** 2026-07-23

**Status:** bounded manual multi-worktree audit; no routeability, causal, or
novelty claim confirmed

## 1. The missed work

ChelatedAI repeatedly compared several admissible methods on the same query or
seed-query instances, selected or rejected one method by its aggregate mean,
and stopped. The retained row-level outcomes show that different methods often
win on different instances even when one method is best on average.

That is a real missed evaluation step, but it is not a new field. It is the
classical per-instance algorithm-selection problem: compare the single best
fixed option with a virtual best or oracle selector, then test whether
pre-outcome instance features can recover any of that upper bound on untouched
instances. The repository-specific gap is that ChelatedAI did not make this
audit a required gate and often did not retain the feature/provenance surface
needed to test routeability.

The important distinction is:

> Positive ex-post oracle headroom proves only that the options made different
> errors. It does not prove that a deployable policy can know which option will
> win before the outcome is observed.

## 2. Formal hypothesis

For independent instance group \(i\), structural cell \(b\), and admissible
option \(a\in\mathcal A\), let \(Y_{bi}^{a}\) be the frozen outcome, with higher
values better. Let \(a_g\) be the best single fixed option selected using
development data only. Define the descriptive ex-post upper bound

\[
O_{bi}=\max_{a\in\mathcal A}Y_{bi}^{a}
\]

and the raw virtual-best envelope

\[
L_{\mathrm{raw}} =
\operatorname{macro}_{b,i}
\left(O_{bi}-Y_{bi}^{a_g}\right).
\]

Because this maximum is taken after observing every option outcome,
\(L_{\mathrm{raw}}\) is pointwise non-negative. It is therefore a descriptive
screening statistic, not an ordinary inferential estimand: a null
\(L_{\mathrm{raw}}\le0\) collapses to exact equality, and repeated stochastic
runs can create a max-of-noise envelope even when the options have equal
expected value.

For a population-level estimand, define
\(\mu_a(x,b)=\mathbb E[Y^a\mid X=x,B=b]\) and

\[
V^* =
\mathbb E_{X,B}\left[\max_{a\in\mathcal A}\mu_a(X,B)\right]
-
\max_{a\in\mathcal A}\mathbb E_{X,B}\left[\mu_a(X,B)\right].
\]

Estimate \(V^*\) only with repeated-run averaging or cross-fitting that keeps
action selection outside the scored observation. Report it separately from
\(L_{\mathrm{raw}}\), and open a routing lane only when both exceed a
preregistered materiality threshold.

For qrels-free,
pre-outcome features \(X_{bi}\), a frozen policy
\(\pi:X\rightarrow\mathcal A\cup\{\mathrm{abstain}\}\), and an independently
valid fixed fallback \(a_g\), define

\[
\Delta_{\pi} =
\operatorname{macro}_{b,i}
\left(Y_{bi}^{\pi(X_{bi})}-Y_{bi}^{a_g}\right).
\]

The deployable hypothesis is:

\[
H_{1,\mathrm{route}}:
\Delta_{\pi}\ge\delta_{\mathrm{MWEE}},
\quad
\operatorname{LCB}_{97.5\%}(\Delta_{\pi})>0,
\]

subject to preregistered structural holdout, non-inferiority, worst-cell,
coverage, cost, ABI, evidence, one-shot, and replay gates. Any policy trained,
selected, or thresholded using the evaluation outcomes does not test this
hypothesis.

## 3. Repository-wide result

The bounded scan covered the checkout, five linked worktrees, both verified
recovery bundles, the preserved raw ZIPs, and 1,064 JSON artifacts
deduplicated by content. It was a manual evidence audit, not a reproducible
general-purpose scanner. It prioritized artifacts with aligned per-option
query outcomes and excluded unreadable, invalidated, aggregate-only, missing,
or non-alignable evidence. Exact included sources and known exclusions are
listed below.

The scan found the same latent-option-value pattern in many lanes, but the
quantitative comparison exposed a more fundamental repository-wide defect:
the legacy nDCG helper computed ideal DCG from the retrieved relevance vector
instead of from every positive qrel. For a two-positive query where only the
first positive is retrieved at rank 1, the legacy function returns `1.0`; the
correct binary nDCG@2 is `0.6131471927654584`. The defect inflates queries with
multiple positive judgments and contaminates any aggregate or option envelope
that used that implementation.

The audit therefore separates the newly corrected BCC-1 derivation from legacy
lanes whose metric lineage has not yet been regenerated:

| Lane | Evidence grain | Quantitative status | Honest disposition |
|---|---:|---|---|
| BCC-1 repaired METHOD_DEV | 720 observations / 240 independent dataset-query groups | corrected all-positive-qrels IDCG; exact regenerated values are bound to the durable artifacts below | sampled, transductive; current feature family cut |
| D2 primary arms | 1,200 observations / 120 base groups repeated across collapse and seed views | `LEGACY_METRIC_LINEAGE_BLOCKED` | authoritative recovery study already negative/inconclusive |
| H5 post-bank lifecycle | aligned query outcomes in SciFact and NFCorpus | `LEGACY_METRIC_LINEAGE_BLOCKED` | descriptive option pattern only |
| Query-encoder swap and budget | aligned query outcomes in SciFact and NFCorpus | `LEGACY_METRIC_LINEAGE_BLOCKED` | selection/evaluation reuse remains independently disqualifying |
| D1 ridge vs Procrustes | 60 aligned SciFact queries | `LEGACY_METRIC_LINEAGE_BLOCKED` | qrels-dependent METHOD_DEV |
| Rung-16 Arenas A/B | 30 and 90 aligned routes | metric lineage unverified; stored values quarantined | structural home/cross-domain reversal remains hypothesis-generating only |
| Phase C and calibrated C0-C4 | aligned SciFact outcomes | `LEGACY_METRIC_LINEAGE_BLOCKED` | consumed development evidence |
| C3 knob grid | 100 aligned SciFact queries | `LEGACY_METRIC_LINEAGE_BLOCKED` | mostly inert duplicate options |
| H4 compound cycles | 60 aligned SciFact queries | `LEGACY_METRIC_LINEAGE_BLOCKED` | qualitative no-complementarity control pending regeneration |

The rows are not directly comparable because their sampling, option sets,
cells, and independence structures differ. Until each lane binds a corrected
metric implementation and is regenerated, its stored numerical nDCG,
headroom, and policy-lift values are historical diagnostics only and are not
accepted scientific evidence.

### 3.1 Repository-wide metric-lineage failure

The discovered faulty legacy call chain included
`drift_recovery_metrics.ndcg_at_k` to `benchmark_utils.ndcg_at_k`. The latter
sorts only the relevance values attached to retrieved documents when forming
IDCG. The current slice repairs the drift-recovery wrapper because it already
receives complete relevant IDs. Known direct or historical consumers of the
susceptible lineage include:

- `benchmark_comparative.py`;
- `benchmark_rlm.py`;
- `benchmark_distillation.py` (duplicated implementation and call);
- `benchmark_evolution.py`;
- `retrieval_fitness_evaluator.py`;
- `run_drift_recovery_experiment.py`;
- `run_phase_c_eval.py`;
- `run_road_course_campaign.py`;
- `run_thousand_query_tuning.py`; and
- the wrapper lineage in `drift_recovery_metrics.py`.

Single-positive queries are numerically unaffected by this particular defect;
multi-positive queries, including ordinary FiQA and NFCorpus cases, are not.
Dataset-level claims are still blocked until the exact evaluated qrels and code
lineage prove that every included query was single-positive or the result is
regenerated.

`synthetic_collapse_benchmark.py` is a known safe exception because it
constructs exactly one relevant document per query. Quarantine follows actual
metric call and qrels lineage, not the presence of the string `nDCG` alone.

An analogous hostile input exposes the same missing ranking-identity invariant
in legacy AP/MAP: duplicated retrieval `["a", "a"]` against positive set
`{"a"}` returns `2.0`. Recall de-duplicates and MRR is unaffected by this
exact case, but the repair campaign must centralize unique-document ranking
validation for every bounded retrieval metric rather than patching nDCG alone.

This finding changes the required experimental ABI. Every future result must
bind the metric name, implementation SHA-256, gain rule, cutoff, ideal-ranking
population, qrels snapshot, query inclusion rule, and canonical document-ID
tie-break. The corrected BCC-1 builder and drift-recovery wrapper implement
that rule locally. The ambiguous shared gain-vector API, remaining caller
migrations, and regeneration campaign are a separate P1 follow-up; no legacy
result may be promoted while that work is pending. The exact repair protocol
is `docs/research/metric-lineage-repair-protocol-2026-07.md`.

## 4. Evidence and caveats by lane

### 4.1 BCC-1 bidirectional cache coherence

Sources:

- `docs/bcc1-method-dev-pack-2026-07.json`
  (SHA-256 `19acc83532b53a103da350fa37d82b724e9aa929778bcea43940d96b7ccdb5a2`)
- `docs/bcc1-method-dev-manifest-2026-07.json`
  (SHA-256 `d0280f7d8fbfa397651adebc13210ed70d3e86030afdf61bd3170a379404728f`)
- `docs/bcc1-method-dev-results-2026-07.json`
  (SHA-256 `25b345eb3233b7d82d684b355d7d12ba41efa75577870e89afa35e05408094b7`)
- `docs/bcc1-method-dev-manifest-2026-07.json.bcc1.lock`
  (SHA-256 `0f41ff9eb08a5b5d56129472ae8c12740532b23f6d21bff0f752b7c2ab72b72e`)
- `scripts/build_bcc1_method_dev_pack.py`

The repaired builder ignores inherited qrels-derived fit indices, chooses a
deterministic qrels-independent half-corpus anchor set from dataset-family and
document IDs, and assigns one independence-group ID to the same raw query
across encoder transitions. The pack contains 720 rows but only 240 independent
dataset-query groups. Forward wins 275 rows, the document-role-derived reverse
proxy wins 150, and 295 tie.
All 12 cells have positive descriptive headroom, from 0.01381320 to
0.07840918.

The pack remains sampled METHOD_DEV. Its bridges are trained transductively on
the evaluated corpus, its encoder revisions and weights are not a complete
representation ABI, and there is no untouched REPORT. More fundamentally, the
recovered packs retain old/new **document-role** anchor vectors but no
old/new query-role anchor pairs. The builder therefore fits its oldward matrix
on document embeddings and applies that matrix to new query embeddings. The
stored `reverse` score is an
`UNVALIDATED_DOCUMENT_ROLE_REVERSE_PROXY`, not evidence for a role-correct
reverse-query bridge. Asymmetric query/document instructions or preprocessing
can change the result. This role mismatch is an independent cut even if every
routing feature were pre-search and predictive.

The feature audit further cuts the current routing family. Nineteen of 21
features require forward and/or reverse resident-corpus score vectors,
candidate ranks, cross-route comparisons, or half-corpus anchor scans. They are
`DUAL_READ_METHOD_DEV`, not pre-search features for a one-authorized-search
resolver. Only query-cycle error and reverse-query norm ratio are true
zero-search features. Even with the richer inadmissible dual-read surface, the
grouped structural harness returns
`CUT_ROLE_MISMATCH_AND_DUAL_READ_FEATURE_SET`: logistic
abstains to the fixed route, while 5-NN is negative under both structural
schemes. The positive envelope justified this test; the test did not establish
routeability. No confirmatory campaign may be opened from this feature family.

### 4.2 D2 primary arms

Sources:

- `research/drift_recovery/out/d2/d2_summary.json`
- `research/drift_recovery/out/d2/cells/scifact_c8/cell_pack.npz`
- `research/drift_recovery/out/d2/cells/scifact_c16/cell_pack.npz`
- `research/drift_recovery/out/d2/cells/nfcorpus_c8/cell_pack.npz`
- `research/drift_recovery/out/d2/cells/nfcorpus_c16/cell_pack.npz`

**Metric-lineage disposition:** `LEGACY_METRIC_LINEAGE_BLOCKED`. Every number
in this subsection is a stored historical diagnostic, not accepted nDCG
evidence, until the D2 artifacts are regenerated with the corrected metric.

The admissible arms were `global_ridge`, `cbie`, `hubness`, and
`chelation_primary`. The stored legacy pooled means were 0.65367580,
0.66895840, 0.65275999, and 0.66437843. Across 1,200 legacy
cell-seed-query rows, the stored unique-win counts were 53, 128, 76, and 27;
916 rows tied.

The raw 1,200-observation envelope is not independent evidence: the same 120
dataset-query groups are repeated across two collapse settings and five seeds.
Averaging collapse seeds first leaves 240 dataset-collapse-query groups and
headroom 0.02742379. Averaging both collapse settings and seeds leaves 120
dataset-query groups and headroom 0.02713700. Known structural metadata
explains only 0.00525843: choosing chelation for SciFact c8, ridge for SciFact
c16, and CBIE for both NFCorpus cells yields 0.67421682 versus the pooled CBIE
mean 0.66895840. That same-data structural choice captures 17.5% of the raw
row-wise envelope and is not an untouched policy result.

More importantly, the authoritative D2 report makes this a negative/inconclusive
recovery control rather than the next positive routing candidate. Its
recoverable oracle gap relative to the corruption floor is approximately zero
or negative in all four cells; chelation is exactly the floor on both SciFact
cells and moves NFCorpus NDCG by at most 0.005. The fixed
\(\beta=0.10\) corruption regime was non-discriminating, so no G3 verdict was
possible. Choosing among adapters can still create an ex-post maximum, but
that is not evidence of recovery when the underlying treatment barely causes
recoverable harm.

Forty validation queries were separate from the 60-query evaluation set,
leakage audits reported empty contaminated-ID sets and zero positive fit
documents, and the four cells have these raw method-choice envelopes:

| Cell | Best fixed | Headroom |
|---|---|---:|
| SciFact c8 | chelation_primary | 0.01118466 |
| SciFact c16 | global_ridge | 0.00817332 |
| NFCorpus c8 | cbie | 0.03878533 |
| NFCorpus c16 | cbie | 0.04090455 |

The retained cluster detector probabilities and harm losses are `5 x 32`, but
the artifacts do not retain query feature vectors or query-to-cluster
assignments. They cannot support a leakage-safe per-query router. D2b
sparse-local is closed in its tested CPU preflight, while D3 and powered
Objective A v2 are negative for the tested simple recoverability estimators.
The lane stays closed unless a new preregistered severity calibration first
creates a meaningful recovery opportunity; a large max-of-options statistic
alone cannot reopen it.

### 4.3 H5 post-bank lifecycle

Sources:

- `docs/drift-recovery-post-bank-headtohead-results-2026-06.md`
- `docs/drift-recovery-post-bank-headtohead-nfcorpus-results-2026-06.md`
- recovery ZIP members under
  `drift-recovery/post-bank-headtohead*/headtohead/*.json`

**Metric-lineage disposition:** `LEGACY_METRIC_LINEAGE_BLOCKED`; the stored
per-query comparisons are hypothesis-generating only.

`C5` living and `C5s` static are bit-identical in aggregate in both datasets.
SciFact favors `C5r` one-shot globally; NFCorpus slightly favors static/living.
The per-query envelope is nevertheless above either fixed choice. The raw
artifacts retain identifiers and outcomes but not a frozen qrels-free query
feature matrix or independent routing holdout.

### 4.4 Query-encoder-swap main and budget arms

Sources:

- `experiment_runs/drift-recovery/swap/swap-campaign-manifest-2026-06.json`
- `experiment_runs/drift-recovery/swap-nfcorpus/swap-campaign-manifest-2026-06.json`
- recovery ZIP members under `drift-recovery/swap*/main/*.json`
- recovery ZIP members under `drift-recovery/swap*/budget/*.json`

**Metric-lineage disposition:** `LEGACY_METRIC_LINEAGE_BLOCKED`; selection and
evaluation reuse would remain disqualifying even after metric regeneration.

For the main comparison, the admissible deployable arms are `C0`, `C2`, `C3a`,
and `C4a`; `C2O` is excluded because it is explicitly an oracle re-embedding
condition. For the budget comparison, the four cells are `30@0.01`,
`200@0.05`, `1000@0.1`, and `2000@0.1`.

Selection and envelope measurement reused seed-42 evaluation outcomes in the
budget sweep. Three-seed confirmation tested only the globally chosen budget.
The 40% anchor split is separate from the 60-query evaluation subset, but there
is no independent query split for option selection. Current-root manifests are
a later generation and do not reproduce the archived rows; the version-locked
semantic-cache worktree manifests and recovery ZIP are the evidence pair.

### 4.5 D1 method complementarity

Source:

- `research/drift_recovery/out/d1/scifact_evalsplit_pack`
- `research/drift_recovery/out/d1/leakage_sensitivity.json`

**Metric-lineage disposition:** `LEGACY_METRIC_LINEAGE_BLOCKED`. The stored
leakage-sensitivity scorer yields ridge 0.6373186238395446,
Procrustes 0.5967599203911357, and their ex-post per-query maximum
0.6704739242031438; those values are not accepted metric evidence. Its stored
win counts are ridge 15, Procrustes 7, and 38 ties.

The retained "safe" fit index has no evaluation-positive documents, but that
index was itself constructed by consulting evaluation qrels. This is
qrels-dependent METHOD_DEV, not a confirmatory or routeable result. Older
stored score arrays report different means and must not be joined to this
recalculation.

### 4.6 Phase C, calibrated C0-C4, and C3 knob grid

Sources:

- `phase_c_results.json`
- `phase_c_analysis.json`
- `docs/drift-recovery-calibrated-results-2026-06.md`
- `experiment_runs/drift-recovery/knob-sweep/knob-sweep-manifest-2026-06.json`
- `experiment_runs/drift-recovery/knob-sweep/grid/*.json`

**Metric-lineage disposition:** `LEGACY_METRIC_LINEAGE_BLOCKED`; all exact
values in these sources require corrected regeneration.

Phase C compared baseline, guard, reformulation, and mask on one 50-query
SciFact slice. Only 3, 3, and 4 queries strictly improved over baseline for the
three non-baseline arms.

The calibrated rotation and noise grids show positive ex-post envelopes, but
the current evidence does not establish a pre-outcome structural router. The
C3 knob grid is especially weak: correction and sedimentation never triggered,
threshold/profile dimensions are inert duplicates, and all 100 queries
participate in a tie. Its small headroom should not be prioritized.

### 4.7 H4 negative control

Sources:

- `docs/drift-recovery-h4-compound-cycles-ablation-2026-07.md`
- `experiment_runs/drift-recovery/h4-compound/C4a_compound0_seed42.json`
- `experiment_runs/drift-recovery/h4-compound/C4a_compound1_seed42.json`

**Metric-lineage disposition:** `LEGACY_METRIC_LINEAGE_BLOCKED`. The
bit-identical/dominance pattern is retained as a qualitative negative control,
not as accepted nDCG.

`compound_cycles=false` dominates `true`; their oracle equals the fixed
non-compounding mean. This lane contains no latent option value and should stay
closed.

### 4.8 Rung-16 domain routing and weight-refinement reversal

Rung-16 source at exact local ref
`lattice/rung16-routing-20260714@2c78195d6d3eb398228bfccf6ae07d97b67a9ea5`:

- `docs/rung16-quant-aware-routing-manifest-2026-07.json`
  (Git blob `4626f902ba0c2eb5a7645ede76a939f8bf673193`, raw SHA-256
  `77572d070a282a17b5a0afc38de2f5c5a56279969d087cddab8a9a296fc526ba`)

**Metric-lineage disposition:** unverified and quarantined. The exact ref and
raw artifact remain useful provenance, but the numeric results below are
stored historical diagnostics until their metric implementation is audited or
regenerated.

Arena A contains 30 unique SciFact queries. In the stored results, plane wins
one, global wins two,
and 27 tie; its raw two-option headroom is 0.01435589. Arena B contains 90
unique queries, 30 each from SciFact, NFCorpus, and FiQA2018. Plane wins three,
global wins six, and 81 tie; raw headroom is 0.00731555.

The more important Arena B result is structural rather than lexical. For 21
queries routed to their own domain plane, plane-minus-global is +0.025669. For
39 cross-domain plane routes it is -0.030860. Thirty margin fallbacks use the
global route and have zero plane/global policy delta. Domain correspondence is
observable before relevance and reverses the aggregate, but the result is
post-hoc: both Arena A and Arena B frozen verdicts are fail-closed, and the
preregistered gate did not pass. The current PR must not be merged as a
positive routing result.

The retained weight-refinement report
`docs/weight-refinement-follow-up-results-2026-04-25-session33.md` supplies a
coarser version of the same motif: the tested candidate changed SciFact by
+0.0226 and NFCorpus by -0.0046, while the aggregate was +0.009017 and transfer
failed. Its raw run directories no longer survive and it retains only two
dataset outcomes, not aligned query rows, so it is structural warning evidence
rather than a routeability dataset.

### 4.9 Mutation-locus recovery policy

Two older lanes expose a higher-level condition that is cheaper and safer than
predicting per-query relevance:

- under calibrated **document-vector corruption**, C2 re-embedding with the
  original encoder exactly restores the baseline in all three rotation seeds
  and all three noise seeds;
- under a **query-encoder swap**, that same document-side C2 action is a no-op;
  the separately labeled C2O/native-new-space re-embed is the action that
  addresses the changed representation role.

This supports a typed recovery-policy hypothesis: first identify which
representation role, lineage, or domain contract changed, then authorize only
actions whose input/output descriptor types repair that locus. Formally, for a
content-addressed mutation descriptor \(L\), an action \(a\) is eligible only
when its declared precondition accepts \(L\) and its postcondition produces the
required serving ABI. A deterministic policy over those types should have zero
wrong-space executions and be non-inferior to the best safe eligible action on
unseen mutation families.

More exactly, let representation state be
\(S=(r,\ell,c,m,d,n,\tau)\): role, lineage, compatibility domain, metric,
dimension, normalization, and dtype. Let
\(L=(S_{\mathrm{before}},S_{\mathrm{after}},\text{mutation evidence})\), and
let each action contract be
\(a=(P_a,Q_a,E_a)\), where \(P_a(L)\) is its typed precondition, \(Q_a(L)\) is
its promised output state, and \(E_a\) is content-addressed validation
evidence. The resolver may execute \(a\) only when:

1. \(P_a(L)\) is true;
2. \(Q_a(L)\) exactly equals the resident serving ABI;
3. the action and evidence digests are trusted;
4. the decision inputs replay to the same digest; and
5. exactly one authorized action wins the frozen deterministic rule.

Otherwise it returns operational `ABSTAIN` without a wrong-space search. The
formal held-out hypothesis, `TRA-1`, is zero wrong-space executions plus a
one-sided 97.5% lower bound above `-0.02` for quality relative to the best safe
eligible fixed action, with entire mutation families held out. The first
campaign must cross document corruption, query-encoder replacement,
document-encoder replacement, compatibility-domain change, and compound
mutations; it must include deliberately mislabeled-role, stale-evidence,
dimension, metric, normalization, dtype, and domain negative controls.

That may be the most operationally meaningful missed integration: route at the
failure-mechanism level before attempting a query-level selector. It is mostly
a representation-ABI correctness rule, not a defensible novelty claim, and it
still requires a new preregistered cross-mutation validation.

### 4.10 Audit boundary and exclusions

The manual scan used these exact Git roots:

- main checkout at `9473e9f7d192299e6333608d91f3a659c63eda8d`;
- `agent-build` at recovered commit
  `b831493a325b4df93615c8e5e4f0989c112eaaa6`;
- `h2-rerun` at `6c3e1847830ec231157a421735a5a34d7d71a658`;
- `relaxed-wozniak-271e04` at
  `bf23a47f754241e77fdc10b91c3f9deb6dfb8943`;
- this `semantic-cache-h1` worktree based on
  `eb7509583e81b6a62d13b82587398562e4bba09a`; and
- `waypoint-recovery` at
  `65ae99a4ae9b90c56febc6d8b84e1447c522ea3f`.

It also inspected the two verified recovery bundles and the raw archives
`CHELATEDAI-relaxed-raw-drift-evidence-20260722.zip` (SHA-256
`8830679e29352d03adb277480f8351cd6b5f0b70b2b5a19b5f4bda6eb91527f0`)
and `CHELATEDAI-weight-refinement-session29-20260722.zip` (SHA-256
`7fe2e845b7128bb2dfbbb96e18df059430145bf2e2d62176ecf766c7ba94342b`).
The 1,064 deduplicated JSON count is a manual scan tally, not a generated
ledger.

Explicit exclusions were invalidated D2 outputs, old BCC-1 artifacts whose
fit-index construction consulted qrels, roadcourse evidence without alignable
raw option rows, aggregate-only reports without query identifiers, C2O when it
was defined as an oracle/non-deployable arm, and current-root manifests that
did not byte-match the archived run generation. These exclusions prevent this
document from being a systematic review. It is a bounded hypothesis-finding
audit with named evidence, not proof that every repository artifact was
classified.

## 5. What the audit falsifies

The audit falsifies six tempting claims:

1. **The oracle envelope is not novel.** Virtual-best versus single-best
   comparison and per-instance algorithm selection are longstanding work.
2. **Positive headroom is not deployability.** In this repository it repeatedly
   coexists with qrels contamination, consumed development data, missing
   features, failed transfer, or no independent holdout.
3. **More options are not automatically better.** Larger portfolios increase
   headroom but also increase the max-of-noise effect, selection bias, and
   overfitting risk. Inert, harmful, or invalid options must be removed before
   an audit.
4. **The pattern is not universal.** H4 has zero option headroom; D2/D2b/D3
   show that an ex-post method maximum can coexist with no meaningful
   recoverable perturbation and negative simple estimators.
5. **The most routeable conditions may be structural.** Domain correspondence
   and representation mutation locus are observable before retrieval and can
   reverse which action is safe. This does not make the rule novel, but it
   changes experiment priority.
6. **A metric label is not metric identity.** Historical artifacts that say
   `nDCG@10` are not comparable evidence unless they bind the implementation,
   ideal-ranking population, qrels snapshot, inclusion rule, and tie-break.
   The legacy multi-positive implementation fails that requirement.

The durable prior-art boundary is recorded in
`docs/research/bcc1-prior-art-manifest-2026-07.json` (SHA-256
`a44fab246246feca4e5e6589934ede623bffd4639251d414608ae254f2c91994`).

## 6. Exact policy for every future experiment

Every experiment with at least two admissible options and aligned instance
outcomes must now pass the following decision sequence:

1. Freeze option eligibility, metric, independent group ID, structural cell ID,
   costs, safety constraints, and outcome-independent feature schema.
2. Separate `METHOD_DEV`, `SELECT`, and sealed one-shot `REPORT` along the
   strongest plausible dataset, transition, task, seed, or time axes.
3. Report the single best fixed option, ex-post oracle upper bound, headroom,
   per-option unique wins, ties, worst-cell results, and counts at both raw
   observation and independent base-group grain.
4. Average repeated seeds or cross-fit option-value estimates before screening.
   Stop when material headroom disappears, is compatible with max-of-noise, or
   is concentrated in an invalid, inert, or harmful option.
5. When headroom survives, fit only a bounded policy family using qrels-free
   features and development outcomes. Use group-disjoint structural
   cross-validation; never split repeated views of one instance.
6. Compare the frozen policy against a fixed fallback selected inside each
   training fold. Held-out outcomes cannot choose the fallback, threshold,
   model, features, or option set.
7. Require practical lift, a one-sided lower bound above zero, native/reference
   non-inferiority, worst-cell bounds, minimum non-abstained coverage, and
   cost/latency limits.
8. Declare every feature's availability stage and execution cost. A one-search
   policy may use only pre-search features; a dual-read result selector is a
   different policy with both reads authorized and costed. Bind the feature
   generator, policy, bridge, representation, materialization, input digest,
   complete candidate score vector, and decision to content-addressed
   contracts trusted by the evaluator.
9. Open REPORT once. A failed or crashed post-open evaluation remains consumed;
   reconditioning requires a new protocol and untouched evidence.
10. If the policy fails, keep the honest fixed winner and record
    `ORACLE_HEADROOM_NOT_ROUTEABLE`. Do not reinterpret the oracle as a product
    benefit.

## 7. Execution priority

1. **Close the current BCC-1 feature family.** Preserve the repaired pack,
   structural harness, and exact cut. Do not open confirmatory evidence. A
   future pre-search-only hypothesis is a new METHOD_DEV derivation, not a
   re-interpretation of this run.
2. **Test typed mutation-locus authorization.** Freeze mutation descriptors and
   action pre/postconditions, include query-swap, document-corruption,
   domain-home, domain-cross, and wrong-space negative cases, and hold out
   entire mutation families. The current `representation_space.py` and
   `semantic_cache_resolver.py` implement content-addressed descriptors,
   resident-index compatibility, bridge/action trust, feature-stage
   enforcement, scorer attestation, abstention, and replay foundations; they
   do not replace the required cross-mutation scientific campaign.
3. **Recondition Rung-16 only under a new protocol.** Domain correspondence is
   a real pre-outcome feature, but the current report is consumed and
   fail-closed. Any new run must freeze home/cross-domain authorization before
   outcomes.
4. **Rebuild H5/swap only if a pre-search policy survives structural
   development gates.** They have larger raw envelopes but weaker retained
   feature and holdout surfaces.
5. **Keep D2/D2b/D3, H4, and the inert C3 dimensions closed.** D2 may reopen
   only after independently preregistered severity calibration creates a
   meaningful recoverable gap.

No lane is authorized for deployment or a historical-computing claim from this
audit. The immediate contribution is a falsifiable evaluation discipline and
an evidence-bound implementation path, not proof that the ex-post option value
can be captured.
