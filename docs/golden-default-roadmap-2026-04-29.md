# Golden Default And Autopilot Roadmap - 2026-04-29

## Scope

This report analyzes the work completed on `2026-04-28` and `2026-04-29`, with emphasis on:

- road-course and 1,000/5,000-query tuning results
- the six-path alternative validation work added on `2026-04-29`
- follow-on reformulation, static-mask, conditional-mask, regularized-mask, and classifier-mask probes
- the self-healing advisory layer and other submodules now connected to `AntigravityEngine`

It is intended to answer two questions:

1. What is the best evidence-backed default or golden path right now?
2. What should the next 1-2 days of autopilot work do to maximize the chance of finding a genuinely positive default?

## Executive conclusion

There is no evidence-backed **golden improving default** yet.

There is, however, a clear **golden safe default**:

- keep the runtime on the baseline retrieval path
- treat `ChelationConfig.DEFAULT_CHELATION_THRESHOLD = 0.01` as a guardrail, not as proof of active chelation lift
- keep active chelation, deterministic reformulation, and learned masking in evaluation or shadow mode only

The strongest conclusion from the last two days is not "we found the winning setting." It is:

- global active chelation thresholds are still unsafe
- deterministic reformulation is at best weakly neutral and usually negative
- synthetic-collapse and learned-mask smoke tests validate the premise in controlled fixtures
- real-data uplift will likely require **query-conditional controls trained from richer per-query attribution**, not another global threshold sweep

## Current golden safe default

### Runtime default

Use this as the practical default stack for the engine today:

| Component | Recommended default | Why |
| --- | --- | --- |
| Core retrieval path | baseline / FAST path | Best real-data behavior remains the baseline path |
| `use_centering` | `False` | Always-on centered chelation regressed on the road-course runs |
| `use_quantization` | `False` at runtime | Quantization is useful as a promotion gate, not as a default actuator |
| `chelation_p` | `85` | Keep the balanced percentile; threshold guardrail prevents over-activation |
| `chelation_threshold` | `0.01` | Current best safety guardrail; preserved baseline on small-model road-course runs |
| `ADAPTIVE_THRESHOLD_ENABLED` | `False` | No evidence yet that adaptive thresholding improves holdout quality |
| Query reformulation | disabled by default | Current heuristic policies do not justify always-on use |
| Masking | disabled by default | Static and conditional masks are not stable enough for production use |
| Self-healing | advisory only | Good planning and diagnostics surface, not a proven runtime actuator |
| Persistent self-healing updates | `False` | No retention/transfer proof for persistence |
| Adaptive gate actions | advisory only | The gate logic is useful, but current evidence does not justify automatic mutation |

### Promotion gates that should remain hard requirements

Any candidate default must still clear all of these:

- repeatability across seeds
- transfer across at least `SciFact`, `NFCorpus`, and `FiQA2018`
- no `actuator_active_negative` fault blockers
- no `metric_changed_without_actuator` blocker
- quantization retained-gain >= `0.80`
- positive FP32 gain before quantization is even considered
- structural health >= `0.70`
- no unacceptable latency or norm-drift regressions

## What the last two days actually proved

### 1. Road-course and long-loop tuning disproved the obvious global defaults

The strongest positive-looking global chelation profile remained unstable:

- `adaptive_p99_t0.0015` produced large upside windows
- but it also produced larger negative windows and many active-negative fault events

That pattern means:

- it is useful as **training data**
- it is not usable as a **default**

`guard_p85_t0.01` stayed safe, but its value is conservative preservation, not quality lift.

### 2. The six-path extension clarified where the signal is

The six-path work was the correct move. It separated "is the premise real?" from "is the current runtime configuration good?"

#### Path 1: Query-level attribution rows

Value:

- highest-leverage addition from the two-day batch
- moves learning from 50-query window averages down to the actual query level where actuators help or hurt

Conclusion:

- keep expanding this dataset
- use it as the primary supervision surface for future learned gates

#### Path 2: Synthetic collapse benchmark

Value:

- proves the masking premise on a deterministic controlled fixture

Conclusion:

- the core idea is valid in principle
- it does **not** justify production masking defaults by itself

#### Path 3: Learned mask smoke

Value:

- recovers the synthetic collapse dimension without hand-labeling it directly

Conclusion:

- learned masking is plausible
- current real-data features and supervision are still too weak

#### Path 4: Selective reformulation policy

Value:

- established that simple lexical heuristics can gate reformulation

Conclusion:

- the current heuristics are not strong enough
- always-on and rule-based selective reformulation both remain too weak or too negative

#### Path 5: Benchmark-family meta-analysis

Value:

- gave a compact way to compare profile behavior by task family across artifacts

Conclusion:

- no task-stable winner emerged
- this is a useful ranking/reporting tool, not a source of defaults by itself

#### Path 6: Candidate-profile proposals

Value:

- turns attribution summaries into concrete "retest" vs "training-data-only" outputs

Conclusion:

- this is the right automation layer for triage
- it is not yet producing a promotable candidate

### 3. The follow-on mask and reformulation probes narrowed the search space further

The follow-on probes were useful because they killed several weak branches quickly:

- reformulation policy search removed the remaining optimism around simple heuristic reformulation
- static supervised masks overfit and hurt holdout
- simple conditional mask gates avoided some damage but did not create a repeatable lift
- regularized conditional masks produced one tiny local positive but did not repeat
- classifier-gated masks only became safe when they failed closed and did nothing
- later dense/pooled classifier sweeps also failed closed with zero positive candidates

This is a strong directional result:

- **do not** spend more time on one-feature or shallow heuristic gates as candidate defaults
- **do** spend time on pooled, query-level, supervised gates trained over richer features and broader examples

### 4. Self-healing is promising as infrastructure, not as a current default actuator

The self-healing layer is useful because it gives the project:

- candidate generation
- provenance
- shadow execution seams
- retrieval-scored evaluation
- retention and quantization gates

But it should stay in this role for now:

- advisory planner
- shadow evaluator
- candidate ledger producer

It should **not** become a persistent automatic behavior-changing default yet.

## Best current interpretation of a "positive default"

The term "positive default" should be defined carefully.

Right now the best positive default is not "the most active system." It is:

- the safest baseline configuration
- plus the richest diagnostic and shadow-evaluation surfaces
- plus a search loop that learns when *not* to activate risky modules

That means the real positive default is operationally:

1. baseline retrieval stays primary
2. `0.01` remains the chelation guardrail
3. active modules run only as shadow/evaluation candidates
4. query-level attribution keeps accumulating
5. promotion requires broad, repeatable evidence

## Recommended golden path for engine submodules

### Core engine

Keep as default:

- `use_centering=False`
- `use_quantization=False`
- `chelation_threshold=0.01`
- `chelation_p=85`

Interpretation:

- leave the engine in a low-regret state
- allow experiments to turn on active modules explicitly

### Query reformulator

Current recommendation:

- not a default actuator
- keep as a candidate-family generator

Best next use:

- learned query-conditional reformulation
- weighted or confidence-aware fusion instead of fixed RRF alone

### Quantization promotion gate

Current recommendation:

- keep it mandatory for all candidate promotion

Why:

- it is correctly failing closed on candidates whose FP32 gains disappear
- this is already one of the cleanest hard constraints in the system

### Adaptive gate orchestrator

Current recommendation:

- keep in `advisory` mode conceptually
- use its outputs as search-control signals, not direct runtime mutation

Best next use:

- stop or down-rank candidate families that keep producing active-negative faults

### Self-healing chelation

Current recommendation:

- advisory only
- no persistent updates
- use it to generate and log candidate actions

Best next use:

- produce candidate repair directives for offline or shadow evaluation
- feed successful patterns into future adapter-bank or route-limited experiments

### Masking modules

Current recommendation:

- no default masking

Best next use:

- pooled query-level supervised experiments
- only with strict fail-closed behavior and external holdout

## Speculative changes most worth trying next

These are the speculative branches with the highest expected value now:

### 1. Query-level learned reformulation gate

Replace heuristic reformulation policies with a small supervised gate that predicts whether reformulation is worth trying for a query.

Likely useful features:

- baseline score margin
- top-score and top-2 gap
- query token count
- stopword ratio
- negation count
- claim-cue count
- route-selection metadata if available

Important design choice:

- do not just predict on/off
- predict whether reformulation should run, and possibly how much weight the fused ranking should receive

### 2. Query-level learned mask gate

Replace global mask-fraction policies with a gate that decides:

- whether to mask
- how much to mask
- only when predicted risk is low and expected gain is positive

Important design choice:

- train on pooled attribution across tasks and seeds
- keep strict fail-closed behavior

### 3. Candidate weighting instead of hard switching

The current probes mostly compare baseline vs actuator. A better next step is:

- baseline ranking as anchor
- reformulated ranking as auxiliary candidate
- masked ranking as auxiliary candidate
- learned weighting or re-ranking confidence instead of binary replacement

### 4. Broader pooled supervision

The classifier-mask experiments strongly suggest the problem is sparse positive supervision.

The next loop should pool:

- multiple tasks
- multiple seeds
- multiple windows
- multiple candidate families

The current train slices are too small for a stable learned gate.

### 5. Route-limited candidate application

If future candidates become positive, apply them narrowly first:

- by task family
- by route cluster
- by query shape

Do not jump from "one weak positive branch" to "global runtime default."

## Autopilot roadmap for the next 1-2 days

### Phase A: Consolidate training data

Goal:

- build one pooled training table from existing artifacts

Actions:

1. Gather `query_attribution_rows` from all relevant tuning artifacts.
2. Gather window-level `gate_feature_rows` for promotion/fault labels.
3. Gather mask-probe outcomes from conditional, regularized, classifier, dense, and pooled sweeps.
4. Build one normalized analysis table keyed by task, seed, window, query, profile, and fault class.

Success condition:

- one pooled dataset suitable for query-level supervised experiments

### Phase B: Train better learned gates

Goal:

- replace simple thresholds with pooled supervised gates

Actions:

1. Train a query-level reformulation gate on pooled attribution.
2. Train a query-level mask gate on pooled attribution plus mask-probe outcomes.
3. Evaluate both with strict holdout by seed or by task.
4. Reject any gate that leaks active-negative behavior on holdout.

Success condition:

- at least one learned gate produces positive mean holdout delta with zero meaningful regressions on the selected evaluation slice

### Phase C: Evaluate only four candidate stacks

Do not open the search space again immediately. Evaluate only:

1. `baseline`
2. `guard_p85_t0.01`
3. `learned_reform_gate_v1`
4. `learned_mask_gate_v1`

Optional fifth candidate only if one of the above is already stable:

5. `learned_reform_gate_v1 + learned_mask_gate_v1`

Success condition:

- a candidate shows repeatable positive deltas without active-negative blockers

### Phase D: Run broader validation only after a candidate survives holdout

If a candidate survives Phase C:

1. run a 1,000-query validation across `SciFact`, `NFCorpus`, and `FiQA2018`
2. run quantization retention
3. run structural-health and latency checks
4. rerun with a second seed family

Only after that should the candidate be treated as a default discussion.

## Stop conditions

Autopilot should stop or pivot immediately if any of these happen:

- the candidate has any repeated `actuator_active_negative` blocker on holdout
- the candidate regresses on transfer tasks after looking positive on the source task
- the candidate fails quantization retention
- the candidate only becomes safe by failing closed and doing nothing
- the candidate requires a task-specific rule that does not generalize

## What not to do next

Avoid these for the next pass:

- another pure global threshold sweep
- another heuristic-only reformulation policy sweep
- another global static-mask search
- persistent self-healing updates
- promoting any candidate from synthetic-collapse success alone

## Final recommendation

The right interpretation of the last two days is:

- the project has not cracked a golden default yet
- but it has successfully identified the correct search surface

That surface is:

- pooled query-level attribution
- learned query-conditional gating
- strict fail-closed promotion criteria
- baseline-plus-guard as the stable runtime anchor

So the working default should remain conservative, while the next autopilot pass focuses on learning **when** to activate modules rather than pushing harder on a single global setting.
