# Qwen-Scope chelated causal-intervention preregistration

**Protocol ID:** `CHELATEDAI-QSCCI-v1`

**Status:** frozen design; not implemented or executed; repository block gate
remains binding

**Fixture file:**
`docs/research/qwen-scope-chelated-causal-intervention-fixture-v1.json`,
SHA-256 `d9f873d5a0e00d0330b87e5ea053aaf0343d4c1636d7402020490719d1336f83`

**Artifact-envelope schema:**
`docs/research/qwen-scope-chelated-causal-intervention-artifact-schema-v1.json`,
SHA-256 `032f06fefc53aa04f13a410ce5aec93b7039c7c042bd0a2cedfe8f4745cb653a`

**Model boundary:** pinned `Qwen/Qwen3.5-2B-Base` plus the official layer-11
Qwen-Scope TopK SAE already verified by `QWEN-SCOPE-REAL-SMOKE`

**Claim boundary:** a passing run would be preliminary causal-intervention
evidence on one small sentiment fixture. It would not establish general
chelation utility, interpretability, training benefit, retrieval benefit,
production readiness, or novelty.

## 1. Question and formal hypothesis

Qwen-Scope already describes contrastive feature discovery followed by
residual-stream steering. The narrower ChelatedAI addition tested here is
whether penalizing a feature for sensitivity to meaning-preserving nuisance
rewrites improves held-out causal steering relative to ordinary contrastive
selection.

For SAE activation `z_j(x)` of feature `j`, paired canonical/nuisance prompts
`(x_i, n_i)` with label `y_i in {-1,+1}`, and a material counterpart `m_i`
with label `-y_i`, define on SELECT only:

```text
contrast_j = mean_i[y_i * (z_j(x_i) - z_j(m_i))]
material_j = abs(contrast_j)
nuisance_j = mean_i[abs(z_j(x_i) - z_j(n_i))]
eligible_j = active on at least 25% of SELECT canonical or material prompts
contrastive_score_j = material_j
chelated_score_j = material_j - nuisance_j
```

Here `m_i` has label `-y_i`. Captured residuals are copied to CPU and SAE
activation is computed there in FP32 as
`pre = h_cpu_float32 @ W_enc_cpu_float32.T + b_enc_cpu_float32`.
SAE/statistic/direction discovery never runs on CUDA. The process sets PyTorch
intra-op and inter-op threads to one, enables deterministic algorithms, and
disables CUDA TF32 globally before any model pass; CUDA is used only for the
pinned BF16 model forward and hook update.
Every `h`, `pre`, and derived activation must be finite. TopK is defined
independently of `torch.topk`: sort all 32,768 pairs by descending FP32
preactivation and then lower feature ID, select the first 100, preserve their
signed values, and set every other `z_j` to exact zero. No ReLU is applied.
A feature is `active` when its ID is in that TopK set. The eligibility
denominator is the 12 SELECT canonical/material prompts; at least three must
select the feature. Nuisance prompts do not change eligibility. Exact zero
`contrast_j` is ineligible. Any nonfinite activation, contrast, nuisance, or
score invalidates the whole run; it is never converted into ineligibility.

All score rankings use descending score and then lower feature ID. No REPORT
activation, label, logit, or metric may influence eligibility, ranking,
feature count, direction sign, or steering strength.

Let `d_j` be column `j` of the checkpoint's `W_dec`, normalized to unit L2
norm. The column must be finite and have positive finite norm. Each selected
direction is multiplied by `sign(contrast_j)`, summed, and normalized once
more to unit L2 norm. A zero or nonfinite summed norm invalidates that cell.
For prompt instance `t` in row `i`, set `y_it=y_i` for canonical/nuisance and
`y_it=-y_i` for material. The label-oracle intervention at the last
non-padding token after transformer layer 11 is:

```text
h' = h + y_it * alpha * s * d
```

where each residual RMS uses FP32 square, sum, division, and square root over
hidden width. Sort the 18 FP32 values ascending; convert elements 8 and 9
(zero-based) to binary64, average them in binary64, and round once to FP32 to
obtain the even-sample median `s`. `alpha` is chosen only on SELECT. `s` must
be positive and finite.

The primary alternative hypothesis is:

> On disjoint REPORT pairs, the equally tuned chelated pipeline produces a
> larger mean improvement in label-oracle signed positive-versus-negative
> next-token logit margin than equally tuned ordinary contrastive,
> activation-matched random, and uniform-random pipelines,
> while satisfying frozen nuisance-stability and collateral gates.

If the chelated pipeline does not beat every control on the frozen REPORT
endpoints, it fails this preregistered finite-fixture gate. This is not an
inferential hypothesis test and closes only this small selection rule on this
fixture.

## 2. Fixed identities

The runner must reuse and re-verify the exact identities from the passed real
smoke:

| Item | Frozen value |
|---|---|
| Model | `Qwen/Qwen3.5-2B-Base` |
| Model revision | `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` |
| SAE repository | `Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100` |
| SAE revision | `027267657257a8d490296286e8fab41e1c1a1a3d` |
| SAE file | `layer11.sae.pt` |
| SAE SHA-256 | `d1828ace348b13cca9104f61fb47672e439e963d9d5fc5496f4c6b068a06499f` |
| Hook | post-layer-11 residual stream |
| Hidden width / SAE width / TopK | `2048 / 32768 / 100` |
| Target token strings | `" positive"`, `" negative"` |
| Observed pinned token IDs | `6572`, `7968` |

Token strings must each encode to exactly one token and to the frozen IDs. Any
identity, tensor-shape, tokenization, or checkpoint-digest mismatch invalidates
the run before model inference.

## 3. Fixed task and splits

Every prompt uses the same two-shot prefix:

```text
Review: The meal was wonderful.
Sentiment: positive
Review: The meal was awful.
Sentiment: negative
Review: {review}
Sentiment:
```

Each row contains a canonical review, a meaning-preserving nuisance rewrite,
and a material opposite. Labels refer to the canonical and nuisance reviews;
the material review has the opposite label.

### SELECT rows

| ID | y | canonical | nuisance | material opposite |
|---|---:|---|---|---|
| S01 | +1 | The service was excellent and the staff were kind. | The staff were kind; the service was excellent. | The service was terrible and the staff were rude. |
| S02 | -1 | The book was dull and painfully confusing. | Painfully confusing, the book was dull. | The book was engaging and wonderfully clear. |
| S03 | +1 | The hotel room was spotless and comfortable. | Comfortable and spotless: that was the hotel room. | The hotel room was filthy and uncomfortable. |
| S04 | -1 | The software was frustrating and unreliable. | Unreliable and frustrating, the software was. | The software was dependable and easy to use. |
| S05 | +1 | The concert was energetic and memorable. | Memorable and energetic, the concert was. | The concert was lifeless and forgettable. |
| S06 | -1 | The delivery was late and the package was damaged. | The package was damaged, and the delivery was late. | The delivery was prompt and the package was perfect. |

### REPORT rows

| ID | y | canonical | nuisance | material opposite |
|---|---:|---|---|---|
| R01 | +1 | The museum tour was fascinating and informative. | Informative and fascinating, the museum tour was. | The museum tour was tedious and uninformative. |
| R02 | -1 | The headphones sounded muddy and uncomfortable. | Uncomfortable and muddy, the headphones sounded. | The headphones sounded crisp and felt comfortable. |
| R03 | +1 | The support agent was patient and helpful. | Helpful and patient, the support agent was. | The support agent was impatient and unhelpful. |
| R04 | -1 | The course was disorganized and shallow. | Shallow and disorganized, the course was. | The course was structured and deeply informative. |
| R05 | +1 | The camera was sturdy and produced beautiful photos. | Sturdy, the camera produced beautiful photos. | The camera was flimsy and produced ugly photos. |
| R06 | -1 | The commute was exhausting and unpleasant. | Unpleasant and exhausting, the commute was. | The commute was relaxing and pleasant. |

The text, punctuation, order, split, and labels are immutable. The fixture
file above is the machine-readable authority; prompts are formed by replacing
the single literal `{review}` marker in its template with the declared review
text, without adding or removing any byte.

## 4. Candidate and controls

The direction families are:

1. `chelated`: top features by `material_j - nuisance_j`;
2. `contrastive`: top features by `material_j`, matching Qwen-Scope's broad
   contrastive discovery route;
3. `activation_matched_random`: three fixed-seed draws from eligible features,
   matched without replacement to the selected chelated features by SELECT
   mean absolute activation decile;
4. `uniform_random`: three fixed-seed draws without replacement from the same
   eligible pool; and
5. `zero_intervention`: the unchanged model.

Random seeds are `1701`, `1709`, and `1721`. Feature counts are `k in
{1,4,8}`. Steering coefficients are `alpha in {0.5,1,2,4}`. Direction signs,
`k`, and `alpha` are chosen on SELECT only.

Random controls are library-independent. Eligible feature IDs are sorted
ascending. For each feature, SELECT mean absolute activation is the arithmetic
mean of `abs(z_j)` over exactly the 12 canonical/material prompts. Sort eligible
features by ascending mean absolute activation and then lower ID; for zero-based
rank `r` among `N` eligible features, define activation decile
`min(9, floor(10*r/N))`. Thus tied values are split deterministically by ID.
For every family, seed, `k`, and eligible feature ID, compute SHA-256 of the
UTF-8 ASCII string `family|seed|k|feature_id`, where integers are unsigned
base-10 with no leading zeros and family is exactly the lowercase protocol
identifier. Digests are compared as their raw 32 bytes (lowercase hex is only
the retained display form). Uniform random selects the `k` lowest
`(digest_bytes, feature_id)` pairs. Activation-matched random first takes the
multiset of deciles occupied by the chelated top-`k`, then, within every
required decile, takes the required count of lowest `(digest, feature_id)`
pairs. Sampling is without replacement within a draw. Chelated and
contrastive IDs remain in the eligible random pool; overlap is measured, not
forbidden. Fewer than eight eligible features, any decile shortfall, or any
missing required draw invalidates the whole run.

Every contrastive or random feature receives its sign from its own SELECT
material contrast using the same formula as the chelated features. Controls
therefore differ only in feature selection, not in access to labels or the
direction-sign rule.

For each nonzero direction family, the runner evaluates label-oracle
class-corrective `+y_it*alpha` and deliberately wrong-sign `-y_it*alpha`
interventions. This tests bidirectional controllability with an oracle label;
it is not an autonomous classifier, correction policy, or deployment path.
Intervened accuracy is therefore diagnostic only and must never be described
as predictive task improvement. A causal direction must improve the declared
class margin in the correct direction and worsen it in the wrong direction.
One-sided movement alone is insufficient. Every one of the required 12
`(k, alpha)` cells must be finite and construct a finite nonzero aggregate
direction for every nonzero family and random seed. No invalid cell may be
dropped before tuning; any missing, zero-norm, or nonfinite cell makes the
whole run `INVALID_RUN`.

## 5. Endpoints

For prompt instance `(i,t)` with label `y_it`, define:

```text
margin(x) = logit(" positive") - logit(" negative")
signed_margin(i,t) = y_it * margin(i,t)
gain(i,t) = signed_margin_intervened(i,t) - signed_margin_baseline(i,t)
```

All prompt-level endpoints use the 18 prompt instances with equal weight. A
row-level endpoint first averages its declared prompt values within a row and
then averages the six row values. Random directions remain separate by seed;
no seed is silently pooled before comparison. Per direction family, seed,
`k`, `alpha`, and intervention sign, retain:

- mean correct-sign gain across all canonical, nuisance, and material prompts;
- mean wrong-sign gain;
- bidirectional causal contrast = correct-sign gain minus wrong-sign gain;
- strict full-vocabulary next-token accuracy before and after intervention:
  correct only when the full-vocabulary argmax equals the frozen target token,
  with an exact-logit tie resolved to the lower token ID;
- canonical/nuisance effect mismatch, the mean absolute difference between
  their gains within each row;
- material response completeness, the fraction of rows where both the
  canonical and material prompt move in their declared corrective direction;
- mean and maximum `D_KL(p_baseline || p_intervened)` over the 18 prompts;
- fraction of prompts whose baseline top-1 token is outside the two target
  tokens and changes to a different outside-target token; and
- wall time, process peak RSS, CUDA peak allocated/reserved, failure count,
  and every rejected/nonfinite cell.

For the material-completeness gate, the per-row material bidirectional causal
contrast uses canonical and material prompts only:

```text
row_material_bidirectional(i) =
  ((gain_correct(i,canonical) + gain_correct(i,material))
   - (gain_wrong(i,canonical) + gain_wrong(i,material))) / 2
```

The nuisance prompt is excluded from this particular row quantity. The
overall bidirectional endpoint above continues to use all 18 prompts.

Canonical/nuisance mismatch, KL, collateral-change fraction, material response
completeness, and their gates use correct-sign interventions only. Wrong-sign
rows are retained and used only for the bidirectional causal contrast and
wrong-sign endpoint. Maximum KL is the maximum over the 18 correct-sign prompt
instances for that exact family/seed/`k`/`alpha` cell.

Except for the explicitly FP32 SAE matmul/TopK leaves, residual RMS, direction
construction/normalization, BF16 hook update, and KL formula, every ordered
statistic and endpoint reduction over retained FP32 leaves uses Python
`math.fsum` in frozen prompt/feature order and divides in binary64. Stored
binary64 aggregates use canonical JSON's shortest round-trip decimal encoding.
Define the sole comparison helper as
`Q(x)=Decimal.from_float(float(x)).quantize(Decimal("0.000000001"),
rounding=ROUND_HALF_EVEN)`. Raw contrast, nuisance, material, chelated score,
endpoint, and tuning-key scalars are fully computed first and then passed to
`Q`; chelated ranking therefore uses `Q(material_j-nuisance_j)`, not
`Q(material_j)-Q(nuisance_j)`. Comparative gates quantize the complete
difference, for example `Q(candidate_contrast-control_contrast) >=
Decimal("0.050000000")`. Absolute gates analogously use
`Q(abs(candidate-control))`; all stated `at least`/`at most` gates are
inclusive. Equality in a tuning key advances to the next declared key.
Feature-ID and TopK comparisons remain exact. Jaccard uses exact integer set
counts; direction cosine uses binary64 accumulation and then `Q`. Nonfinite
leaves or aggregates invalidate the run.

The artifact retains the SHA-bound sparse SELECT activation rows for all 18
SELECT prompts: prompt ID plus exactly 100 ordered feature IDs and their signed
FP32 values. The verifier reconstructs the implicit `18 x 32768` zero-filled
matrix from these rows. It then reconstructs one record for every one of the
32,768 feature
IDs: active count, signed contrast, material score, nuisance score, eligibility
boolean and exclusion reason, mean absolute activation, activation decile when
eligible, sign, and both family scores. It also retains every selected feature
ID, random digest rank, aggregate direction norm, and feature-set overlap.
Every per-prompt record retains baseline and intervened margins, gains, KL,
actual baseline and intervened full-vocabulary top-1 token IDs, direction
family, seed, `k`, `alpha`, and intervention sign. Aggregate endpoints,
collateral flags, and gates are recomputed from those records. Full-vocabulary
logits and dense hidden states are not retained. Sparse SAE rows,
baseline/intervened margins, KL scalars, and top-1 token IDs therefore remain
model-execution-attested. The verifier checks their schemas, finiteness,
ranges, identities, and cross-links, then independently reconstructs all
downstream feature statistics, rankings, selected directions, aggregate
endpoints, and dispositions. It does not claim to reconstruct the attested
leaves without rerunning the pinned model.

Every family and every random seed independently chooses its own `(k, alpha)`
on SELECT using the identical 12-cell grid and the same lexicographic order:

1. highest bidirectional causal contrast;
2. lowest canonical/nuisance effect mismatch;
3. lowest mean KL;
4. smaller `alpha`;
5. smaller `k`.

The primary REPORT comparison uses those independently SELECT-tuned pipelines.
A secondary matched-operating-point ablation evaluates every family at the
externally frozen `k=4, alpha=2`; it is reported separately and cannot rescue
a failed primary gate. Feature-set Jaccard overlap and aggregate-direction
cosine are retained for chelated versus contrastive and every random control.
An identical feature set or absolute direction cosine above `0.999` between
the independently SELECT-selected primary directions is
`NO_TREATMENT_SEPARATION`, not evidence for an additive selection rule. The
secondary fixed operating point and unselected grid cells do not determine
this status.

## 6. Frozen gates

The run is `INVALID_TASK` unless unsteered strict full-vocabulary accuracy over
the 18 prompts is at least `0.75` on SELECT and separately at least `0.75` on
REPORT. REPORT task validity is evaluated once after SELECT choices freeze; it
cannot trigger retuning or a new selection pass.

The candidate receives `SURVIVES_SMALL_LABEL_ORACLE_FIXTURE` only if all
conditions hold on REPORT:

1. mean correct-sign gain is positive and mean wrong-sign gain is negative;
2. bidirectional causal contrast exceeds ordinary contrastive by at least
   `0.05` logit and exceeds the best of all six random controls by at least
   `0.05` logit;
3. canonical/nuisance effect mismatch is no greater than the mismatch of
   ordinary contrastive or any of the six random controls;
4. material response completeness is at least `5/6` and no lower than every
   control, where a row is complete only if both its canonical and material
   correct-sign gains are at least `0.02` logit and its row bidirectional
   causal contrast is at least `0.04` logit;
5. mean KL is at most `0.05` nat and maximum KL is at most `0.25` nat;
6. outside-target top-1 collateral-change fraction is at most `1/6`;
7. every retained value is finite and every fixture, identity, resource, and
   artifact-integrity check passes; and
8. SELECT and REPORT feature IDs, parameter choices, and raw endpoints are
   retained so the decision can be independently recomputed.

Status precedence is deterministic. Any invocation, identity, infrastructure,
resource, restoration, nonfinite, incomplete-output, or integrity failure
yields `INVALID_RUN` regardless of any concurrently observable task or
scientific endpoint. Otherwise a baseline task failure yields `INVALID_TASK`.
Otherwise any failed scientific gate or `NO_TREATMENT_SEPARATION` yields
`DOES_NOT_SURVIVE_SMALL_LABEL_ORACLE_FIXTURE`; only then can every passed gate
yield `SURVIVES_SMALL_LABEL_ORACLE_FIXTURE`. These statuses are mutually
exclusive.

## 7. Resource and execution contract

- Offline pinned-cache execution only; no download during the run.
- One Spark GPU is expected to be sufficient based only on the earlier load
  smoke; this campaign must still pass its own preflight and retained resource
  gates. Do not attempt concurrent model copies across both Sparks.
- Cooperative Python ceiling: 30 minutes.
- External supervisor ceiling: TERM at 1,900 seconds, KILL 30 seconds later.
- Maximum process peak RSS: 24 GiB.
- Minimum free disk before load: 12 GiB.
- Minimum free CUDA memory before load: 12 GiB.
- Maximum retained CUDA allocated: 8 GiB; maximum reserved: 10 GiB.
- Python 3.12, PyTorch `2.13.0+cu130`, Transformers `5.15.0`, CUDA, BF16 model
  weights, float32 SAE arithmetic and logits, `attn_implementation="sdpa"`,
  tokenizer right padding, and EOS-as-pad only if the tokenizer has no pad ID.
- Prompt order is row ID ascending and, within each row, canonical, nuisance,
  material. One microbatch contains all 18 prompts. Every pass uses
  `torch.inference_mode()`.
- A tuple layer output is replaced by `(modified_hidden, *output[1:])`; a
  tensor output is replaced by the modified tensor. Only the last
  non-padding-token residual is changed.
- Residual RMS `s`, direction sums, normalization, and intervention scaling
  are accumulated in FP32. The exact hook update is
  `delta_bf16=(y_it*alpha*s*d_fp32).to(dtype=h.dtype)` followed by
  `modified_hidden=h+delta_bf16`; it may not promote the BF16 hidden state.
  Each cell retains the shared FP32 delta norm before casting and, for every
  one of its 18 prompts, the cast BF16 delta norm plus the actual realized
  update norm `norm((modified_hidden-h).float())` after BF16 addition. Every
  per-prompt norm must be finite and positive; one vanished prompt update
  invalidates the whole cell and therefore the run.
- KL uses FP32 logits in vocabulary-token order:
  `logp0=log_softmax(base.float())`, `logp1=log_softmax(changed.float())`,
  `p0=exp(logp0)`, and `sum(p0*(logp0-logp1), dtype=float32)`. Any nonfinite
  logit or intermediate invalidates the run.
- CUDA is synchronized before and after every measured forward pass. Peak
  CUDA statistics are reset after model/SAE load and before the first task
  pass; the reset baseline and final allocated/reserved peaks are retained.
- Batched forward inference only; no generation, optimizer, gradient, or model
  weight update.
- Atomic result-directory publication with canonical JSON, file digests, and
  an independently verifiable commit receipt. Refuse overwrite and stale
  partial directories.

An out-of-process stdlib supervisor, not the research worker, owns the entire
service lifecycle. It records service state, optionally stops DeepSeek, starts
the research child in its own process group, applies TERM at 1,900 seconds and
KILL 30 seconds later to that child group only, and remains alive. In a
`finally` path it restores and verifies DeepSeek when required. The child may
write only to a hidden staging directory; only the supervisor can publish it,
and only after successful restoration or a verified `NOT_APPLICABLE` service
state. Supervisor/restoration evidence has a separate 10-minute ceiling; its
failure produces `INVALID_RUN` and quarantines the child stage.

If the two-node DeepSeek service must be stopped to free CUDA memory, first
record both container identities and idle state. After the run, restore the
same pinned service and verify both containers, `/health`, `/v1/models`, one
real completion, and zero running/waiting requests. A research result is not
complete until restoration evidence is retained. If the service was never
stopped, restoration status is `NOT_APPLICABLE` and the retained before/after
health observations must show it remained available. If required restoration
fails, all otherwise generated research bytes are quarantined and the run is
`INVALID_RUN`; no result directory is published.

## 8. Implementation gate

Independent design review reached `100/100` on the frozen estimands, controls,
numerics, evidence boundary, lifecycle, and claim boundary. This design
approval does not override the repository block flag. Implementation or model
execution must remain research-only and may begin only after the operator
explicitly authorizes work under that still-`BLOCKED` state. Any implementation
must mechanically confirm:

1. the signed material and nuisance estimands match the fixture semantics;
2. decoder-column orientation and residual hook replacement match the official
   Qwen-Scope equation;
3. no REPORT information enters selection or tuning;
4. random and contrastive controls have equal search and intervention budgets;
5. the bidirectional and collateral gates cannot be satisfied by a dead or
   indiscriminately disruptive direction;
6. the artifact verifier reconstructs every gate rather than trusting stored
   dispositions; and
7. the experiment remains research-only while the repository block flag is
   `BLOCKED`.

## 9. Primary sources and prior-art boundary

- Qwen-Scope model card and four-tensor SAE contract:
  <https://huggingface.co/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100>
- Qwen-Scope technical report, especially contrastive feature discovery and
  residual steering `h' = h + alpha*d`:
  <https://arxiv.org/abs/2605.11887>

The general SAE steering mechanism and contrastive feature discovery are
Qwen-Scope/prior art. The only ChelatedAI-specific candidate here is the
nuisance-penalized selection rule and its paired invariance/material-response
gate. Even a pass would establish a narrow empirical addition, not a new
general theory of memory, dimensionality, lattices, resonance, or training.
