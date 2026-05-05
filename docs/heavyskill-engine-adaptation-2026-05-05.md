# HeavySkill Engine Adaptation Review

Date: 2026-05-05

Sources reviewed:

- Paper: `HeavySkill: Heavy Thinking as the Inner Skill in Agentic Harness`, arXiv:2605.02396v1, 2026-05-04, `https://arxiv.org/abs/2605.02396`.
- HTML paper view: `https://arxiv.org/html/2605.02396v1`.
- Reference repository: `https://github.com/wjn1996/HeavySkill`, HEAD `649a5b4af6e941c358f0ad6f019010c5acd0e28c`.
- Related work inspected:
  - `Self-Consistency Improves Chain of Thought Reasoning in Language Models`, arXiv:2203.11171.
  - `Tree of Thoughts: Deliberate Problem Solving with Large Language Models`, arXiv:2305.10601.
  - `ParaThinker: Native Parallel Thinking as a New Paradigm to Scale LLM Test-time Compute`, arXiv:2509.04475.
  - `LongCat-Flash-Thinking-2601 Technical Report`, arXiv:2601.16725.
  - `Group Sequence Policy Optimization`, arXiv:2507.18071.
  - `Agent Skills for Large Language Models: Architecture, Acquisition, Security, and the Path Forward`, arXiv:2602.12430.
  - `Towards Secure Agent Skills: Architecture, Threat Taxonomy, and Security Analysis`, arXiv:2604.02837.
  - OpenClaw agent runtime documentation, `https://docs.openclaw.ai/concepts/agent`.

## Executive Read

HeavySkill argues that much of the useful behavior inside modern agentic harnesses can be reduced to a two-stage test-time scaling pattern:

1. generate multiple independent reasoning trajectories for the same task
2. serialize those trajectories into a memory cache and ask a deliberation model to synthesize the final answer

The important transfer into ChelatedAI is not "spawn more agents." HeavySkill is primarily a harness paper. Its core mechanism lives outside the model weights and outside the retrieval engine: sample several independent attempts, serialize them as context, and run a deliberator over that context.

For this repo, that means HeavySkill should not become a full agent harness implementation. The useful transfer is an engine-adjacent adaptation layer: a replayable trajectory/cache evidence format plus hook points that let the engine vary, segment, protect, or amplify different channels during runtime. That layer can later support harness-like deliberation, but the first implementation should stay focused on engine tuning and fail-closed adaptive variation.

The clean name for the user idea is:

> adaptive channel overlay

This means a runtime overlay that sits above a base model or retrieval engine, divides behavior into typed channels, and learns when to route more aggression toward a channel, damp it, protect it, or fork it into alternate variants. It is similar to RAG only in the sense that it injects external state at inference time. It is not static context retrieval; it is adaptive variation over engine/model control surfaces.

## Paper Claims That Matter

HeavySkill decomposes inference into `parallel reasoning` and `sequential deliberation`. The parallel phase samples `K` independent trajectories. The deliberation phase receives a serialized memory cache built from those trajectories and produces one or more summary answers.

The paper reports a performance hierarchy on verifiable STEM tasks:

`Heavy-Pass@K >= Heavy-Mean@K >= Vote@K >= Mean@K`

That hierarchy is directly relevant to our evaluation posture because it separates four things we often blur:

1. single-run average quality
2. whether any sampled trajectory contains a correct answer
3. whether a voting heuristic can recover that answer
4. whether deliberation can recover or re-derive it

The paper also claims that quality and diversity of the sampled trajectories are key contributors, that the deliberation model can be different from the reasoning model, and that RLVR can optimize both width and depth.

## Related Work Interpretation

HeavySkill is best understood as a consolidation paper over several adjacent lines:

| Line of work | Methodology | What it contributes to HeavySkill | What it means for ChelatedAI |
| --- | --- | --- | --- |
| Self-consistency | Sample diverse reasoning paths, then marginalize/select the most consistent answer. | Establishes that multiple sampled paths expose latent model capability beyond greedy decoding. | Useful metric split: sample availability vs selection quality. Not enough by itself because retrieval metrics do not have a single boxed answer. |
| Tree of Thoughts | Explore intermediate "thought" states with self-evaluation, lookahead, and backtracking. | Shows that deliberate search over reasoning states helps tasks requiring planning. | Relevant as a search/control analogy, but too harness-heavy to import directly. Our analogue is branch search over engine profiles and hooks. |
| ParaThinker | Train a model to produce multiple parallel paths and synthesize them, targeting test-time compute bottlenecks and "tunnel vision." | Supports the width-vs-depth framing and the value of path diversity. | Long-term insight only. We should not train native parallel cognition; we should log engine branch diversity first. |
| LongCat Heavy Thinking | Combines reasoning depth and width in an agentic/tool-capable model family. | Confirms that heavy thinking is now a model/harness product feature, not only a prompt trick. | Useful as a benchmark direction, but the repo should remain provider/model agnostic. |
| GSPO / RLVR | Sequence-level RL optimization for LLM training and verifiable reward settings. | HeavySkill uses this as the bridge from harness behavior to trainable policy. | Defer. ChelatedAI can use verifiable rewards for overlay policy and engine tuning, not base-model rewriting. |
| Agent Skills survey | Skills are dynamic capability packages loaded on demand without retraining. | Explains why HeavySkill is packaged as `heavyskill.md`. | Use as a protocol inspiration only. Engine hooks need typed schemas, permissions, and promotion gates. |
| Secure Agent Skills | Skills create lifecycle and data-instruction boundary risks. | Warns that a readable skill is also an attack surface. | Another reason not to drop a free-form harness skill into the engine path. Use typed, gated policies instead. |
| OpenClaw-style runtime | Workspace files, skills, memory, tools, sessions, and steering are loaded by an outer runtime. | Shows the operational shape of a harness. | Confirms this is outside CHELATEDAI's main scope unless we expose a narrow integration hook. |

The net is straightforward: HeavySkill is a test-time orchestration/control pattern. It is not an engine-tuning method in the same sense as Qwen-Scope, AttnRes, or our road-course promotion gates. Its best fit is to improve how we observe and steer adaptive branches, not to become the product architecture.

## Methodology Caveats

HeavySkill's reported gains are strongest when a verifier can determine correctness cleanly. That includes math, coding, and constrained reasoning. CHELATEDAI's main surface is retrieval and adaptive engine tuning, where a "correct" answer is usually a metric distribution over ranked documents, not a boxed scalar answer.

Important caveats:

1. `Pass@K` is an upper bound on whether at least one branch found a good answer; it is not evidence that the system can pick the good branch.
2. `Vote@K` works best when answers normalize cleanly. Retrieval branches may disagree by useful complementarity rather than a single majority winner.
3. Deliberation can regress strong branches by over-summarizing or preferring consensus. The engine must record active-negative cases where synthesis hides a bad intervention.
4. RLVR claims should not be applied to this repo until reward adapters are verifiable and replayable.
5. The reference repo stores raw trajectories as strings. That is acceptable for a demo harness, but insufficient for engine promotion because it omits provenance, clipping, score, and replay boundaries.

## Reference Repo Findings

The GitHub implementation is small and confirms the paper's practical shape:

| Component | Repo implementation | Notes for us |
| --- | --- | --- |
| Parallel generation | `workflow/parallel_reasoning.py` calls an OpenAI-compatible agent with `n=config.reason_k`. | Good minimal runner pattern, but it emits raw strings rather than replayable typed records. |
| Memory cache | `workflow/memory_cache.py` stores trajectories and deliberation history. Selection supports `random`, answer-frequency, and answer-diversity modes, then shuffles selected entries. | Selection and shuffle policy should be persisted for reproducibility. |
| Token budget | `workflow/utils.py` estimates tokens and clips trajectories when the cache exceeds budget. | We need explicit omitted-span hashes and clipping provenance. |
| Deliberation | `workflow/sequential_deliberation.py` builds a summary prompt and samples `summary_k` outputs. | We need a deliberation record with input-cache hash, output hashes, score, and verifier notes. |
| Skill mode | `skill/heavyskill.md` tells a harness to spawn independent agents, collect trajectories, deliberate locally, and output only the final answer. | Useful as an engine-readable protocol, but CHELATEDAI needs schema and gates before a skill is enough. |
| Evaluation | `scripts/evaluate.py` normalizes boxed answers and exact/numeric matches. | Too narrow for retrieval and engine tasks; we need task-specific verifier adapters. |

## Matrix Comparison Against ChelatedAI

| HeavySkill concept | Current CHELATEDAI analogue | Gap | Priority |
| --- | --- | --- | --- |
| Independent trajectory set | Road-course profile comparisons, reformulation variants, Model-Scope observations | No first-class `TrajectoryRecord` with prompt, seed, model, answer, artifact pointers, and score | P0 |
| Serialized memory cache | Model-Scope artifact writer, evidence events, experiment JSON reports | No cache manifest with included trajectories, shuffle seed, clipping policy, omitted hashes, and context budget | P0 |
| Sequential deliberation | Comparator reports and promotion summaries | No deliberator model/stage that consumes a cache and emits a replayable synthesis | P3 |
| Iterative deliberation | Autopilot loops and repeat-seed validation | No per-query depth state or convergence/stop reason for deliberation loops | P4 |
| Heavy-Mean/Pass/Vote metrics | NDCG@10, MAP@10, MRR, recall, quantization gate | No heavy-thinking metrics that separate sample availability, voting recovery, and deliberation recovery | P0 |
| Verifiable rewards | Promotion contract, holdout reports, quantization promotion gate | Need verifier adapters per task family before RLVR claims are safe | P0 |
| Trajectory diversity | Balanced AttnRes profile set and active-negative mining | No semantic/answer diversity diagnostics over candidate trajectories | P1 |
| Deliberator model selection | Model-Scope runtime and provider probes | No policy for cheap vs strong deliberators or cross-model pairing | P4 |
| Heavy-mode-aware RL | Sedimentation, adapter tuning, fail-closed promotion | Too early until logs prove which width/depth decisions correlate with reward | P5 |
| Readable harness skill | Local docs and Codex skill-like handoffs | Out of main scope; only useful as a future integration spec | Defer |
| Adaptive channel overlay | Engine-Scope gates, Model-Scope hooks, masks, reformulation, AttnRes profiles | No unified runtime vocabulary for routing, damping, protecting, or forking channels | P0 |

## Adaptive Channel Overlay

The better implementation target is not "HeavySkill for CHELATEDAI." It is an adaptive channel overlay that can use HeavySkill-style evidence when useful.

In this architecture:

- a `channel` is a bounded control surface such as retrieval baseline, reformulation, mask gate, AttnRes profile, Model-Scope hook, feature steering overlay, verifier, or safety blocker
- a `variation` is one concrete setting of that channel for a query or batch
- an `intake` is the current evidence packet: query features, prior outcomes, memory summaries, profile scores, safety signals, and budget state
- a `router` decides which channels should receive more aggression, damping, protection, or branching
- a `promotion gate` decides whether any changed channel policy can persist

This is RAG-like only in that external state affects inference. The difference is that the state is not primarily retrieved content. It is adaptive control state over engine behavior.

The key verbs are:

| Verb | Engine meaning | Example |
| --- | --- | --- |
| Route | Send a query toward a channel or profile family | Prefer reformulation variants on ambiguous short queries |
| Amplify | Increase channel influence or sampling width | Try more reformulation candidates when feature coverage is low |
| Damp | Reduce a channel's influence | Suppress masking after active-negative blockers |
| Protect | Prevent a channel from mutating or affecting output | Freeze baseline retrieval and safety gates during experiments |
| Fork | Create controlled alternate variants | Compare baseline, mask, reformulation, and AttnRes branches |
| Merge | Combine evidence from variants | Report candidate deltas and consensus/contradiction |
| Promote | Persist a policy only after replay and holdout | Make a channel router default only after quantization and repeat-seed gates |

## Information To Add To The Engine

### 1. `TrajectoryRecord`

One row per generated attempt, regardless of whether it is later selected into a cache.

Required fields:

- `run_id`, `query_id`, `trajectory_id`, `parent_task_id`
- `task_family`, `dataset`, `split`, `seed`, `offset`
- `reasoning_model`, `provider`, `model_revision`, `prompt_id`, `prompt_hash`
- `sampling`: temperature, top_p, max_tokens, stop policy
- `input_hash`, `output_hash`, `answer_text`, `answer_hash`
- `rationale_hash` or `rationale_summary`; avoid storing unrestricted long chain text by default
- `artifact_paths`: Model-Scope captures, retrieval traces, evidence events
- `status`: completed, filtered_repetitive, timeout, provider_error, clipped
- `score`: task verifier result, retrieval metrics, comparator deltas when available
- `telemetry`: latency, input/output tokens, cost estimate, device/memory stats

### 2. `TrajectoryCacheManifest`

One manifest per deliberation input cache.

Required fields:

- `cache_id`, `run_id`, `query_id`
- `width_k_requested`, `width_k_available`, `width_k_selected`
- `selection_strategy`: random, answer_frequency, diversity, profile_policy
- `shuffle_seed`, `selected_trajectory_ids`, `selection_order`
- `context_budget_tokens`, `estimated_cache_tokens`
- `clipping_policy`, `clipped_trajectory_ids`, `omitted_span_hashes`
- `cache_hash`, `serialized_prompt_hash`
- `prior_deliberation_ids` for iterative depth

### 3. `DeliberationRecord`

One row per summary/deliberation output.

Required fields:

- `deliberation_id`, `cache_id`, `iteration_index`, `depth_n`
- `deliberator_model`, `provider`, `model_revision`, `prompt_id`, `prompt_hash`
- `summary_k_index`, `output_hash`, `final_answer_text`, `final_answer_hash`
- `selected_or_rederived`: selected, merged, rederived, abstained
- `compared_trajectory_ids`
- `verifier_result`, `score`, `failure_class`
- `telemetry`: latency, tokens, cost, stop reason
- `promotion_evidence_event_id`

### 4. `HeavyThinkingRun`

The run-level envelope that lets promotion and replay reason about breadth and depth.

Required fields:

- `run_id`, `task_family`, `dataset`, `split`, `created_at`
- `width_k`, `summary_k`, `depth_n`, `max_budget`
- `reasoning_policy`, `cache_policy`, `deliberation_policy`
- `baseline_score`, `mean_at_k`, `pass_at_k`, `vote_at_k`, `heavy_mean_at_k`, `heavy_pass_at_k`
- `budget_normalized_lift`
- `active_negative_count`, `regression_count`
- `promotion_decision_id`, `promotion_ready`, `blockers`

### 5. Diversity And Quality Diagnostics

The paper's strongest operational claim is that trajectory quality and diversity matter. We need diagnostics rather than just more samples:

- answer distribution entropy
- semantic cluster count over answers and rationales
- pairwise contradiction flags
- retrieval-result overlap between trajectories
- profile/source diversity when trajectories come from different engine branches
- low-pass-rate cohort labels so we can test whether deliberation actually rescues hard queries

### 6. `ChannelVariationRecord`

This is the bridge from HeavySkill-style trajectories into the engine-tuning scope.

Required fields:

- `channel_id`, `channel_type`, `variation_id`
- `aggression_level`: disabled, observe, soft, normal, high
- `protection_level`: mutable, guarded, frozen, safety_critical
- `input_features_hash`, `intake_id`, `policy_id`
- `baseline_output_hash`, `variation_output_hash`
- `metric_delta`, `active_negative_flags`, `safety_flags`
- `decision`: route, amplify, damp, protect, fork, merge, promote, reject
- `decision_reason`, `budget_impact`, `replay_group_id`

### 7. `AdaptiveOverlayIntake`

One typed evidence packet used by the router before it changes channel behavior.

Required fields:

- query/task metadata
- engine feature summary
- model-scope observation summary when present
- recent memory/evidence profile matches
- current channel states
- safety and blocker state
- budget state
- eligible variation policies
- forbidden mutations

## Adaptation Plan

### P0: Observation-Only Adaptive Overlay Evidence

Add dataclasses or typed JSON builders for `ChannelVariationRecord`, `AdaptiveOverlayIntake`, `TrajectoryRecord`, `TrajectoryCacheManifest`, and `HeavyThinkingRun`. Wire them to the existing evidence/artifact style without changing any default retrieval behavior.

Acceptance bar:

- deterministic hashes for records, intakes, and cache manifests
- JSON schema versioning
- unit tests for serialization, routing/protection fields, clipping provenance, and hash stability
- no promotion path enabled

### P1: Channel Variation Emitter

Emit observation records for existing retrieval branches: baseline, reformulation variants, mask variants, AttnRes profiles, and Model-Scope-observed runs. Do not add an LLM deliberator yet.

Acceptance bar:

- emits channel variation records for every branch
- records route/amplify/damp/protect/fork/merge decisions when they are known
- preserves baseline protection state
- can replay a fixed query slice with identical manifests

### P2: Heavy Metrics Over Engine Branches

Compute HeavySkill-inspired metrics over branch sets without importing a harness:

- `Mean@K`: average branch score
- `Pass@K`: whether any branch exceeds a success threshold
- `Vote@K`: whether a simple consensus/selection heuristic recovers a good branch
- `Overlay-Mean@K`: score after the overlay router picks a branch
- `Oracle gap`: `Pass@K - Overlay-Mean@K`

Acceptance bar:

- computes `Mean@K`, `Pass@K`, and `Vote@K` analogues for retrieval metrics
- reports oracle gap and active-negative counts
- fails closed when thresholds are task-incompatible

### P3: Optional Deliberation Adapter

Only after P0-P2, add a deliberation stage that consumes a cache manifest and emits a final candidate answer or retrieval action. Start with text-only deliberation over summaries and metric traces, not raw hidden-state dumps.

Acceptance bar:

- deliberator prompt is versioned
- output is verifier-scored
- failure classes include `majority_wrong`, `minority_recovered`, `all_wrong_rederived`, and `deliberation_regressed`
- default remains disabled

### P4: Promotion Gate Extension

Extend the promotion contract to understand adaptive overlay evidence.

Acceptance bar:

- promotion fails closed when replay, holdout, verifier, quantization, or budget-normalized lift is missing
- overlay lift must survive repeat seeds
- no active-negative blocker can be masked by a high aggregate score
- protected channels cannot be mutated by promotion

### P5: Coverage-Aware Width And Channel Search

Use diversity and low-pass-rate cohort labels to decide where more branch width, a channel fork, or another deliberation iteration is worth the budget.

Acceptance bar:

- stop conditions are explicit
- width/channel changes are justified by observed uncertainty or disagreement
- budget-normalized lift is reported beside raw lift

### P6: Heavy-Mode-Aware Training

Only after the above exists, train policies that choose width, channel source mix, cache selection strategy, and optional deliberator model.

Acceptance bar:

- reward functions are task-verifiable
- training is replayable from records
- learned policy remains advisory until promotion gates pass

## Claim Boundaries For This Repo

HeavySkill's most convincing results are on verifiable reasoning tasks. CHELATEDAI's current strongest benchmark surface is retrieval. That means we should adapt the evidence architecture and metrics, not claim direct performance transfer.

Safe claim:

> ChelatedAI can evaluate HeavySkill-style breadth as typed, replayable adaptive overlay evidence over engine channels, while keeping harness-level deliberation optional and disabled by default.

Unsafe claim for now:

> HeavySkill proves that more agents or more samples will improve our retrieval engine by default.

The next practical slice is P0: typed adaptive overlay evidence and cache manifests. It is small, testable, and sets up every later experiment without changing production defaults.

## Implementation Decision

Do not implement a full HeavySkill harness in this repo now.

Most of the adaptive channel overlay is already part of the current ChelatedAI direction under existing names:

| Overlay idea | Existing repo surface | Status |
| --- | --- | --- |
| Channels | Road-course profiles, reformulation variants, mask probes, AttnRes profiles, Model-Scope hook runtime | Already present |
| Intakes | Engine-Scope rows, Model-Scope artifacts, evidence bundles, query attribution rows | Already present |
| Routing | Learned reformulation gates, learned mask gates, profile selection in tuning/autopilot | Partially present |
| Damp/protect behavior | Active-negative blockers, fail-closed gates, safe default recommendation, baseline preservation | Already present |
| Promotion | `promotion_contract.py`, quantization gate, repeat-seed docs, holdout gates | Already present |
| Replayable evidence | `evidence_contract.py`, Engine-Scope row loaders, Model-Scope artifacts | Already present |
| Unified overlay vocabulary | Cross-channel channel variation records | Missing before this review |

The only near-term adaptation worth pulling from HeavySkill is the normalized branch-set vocabulary: make each existing profile/gate/hook branch visible as a channel variation, then compute metrics over those branch sets. That is useful because it lets existing engine work ask HeavySkill-like questions without copying the harness:

1. Did any channel variant work for this query?
2. Did the router pick the working variant?
3. Did a negative channel need damping or protection?
4. Is the oracle gap shrinking across learned gates?
5. Does wider branch search add value after budget and active-negative costs?

Implemented in this follow-up:

1. `adaptive_overlay.py` adds observation-only `ChannelVariationRecord` and `AdaptiveOverlayIntake` builders.
2. Engine-Scope rows can now be normalized into channel records with `channel_type`, `aggression_level`, `protection_level`, `decision`, `active_negative_flags`, and stable record hashes.
3. The module includes summaries over channel type, decision, blockers, and active-negative records.
4. Tests cover active-negative reformulation damping, frozen baseline protection, intake construction, and summary output.

Continue to implement:

1. Promotion-gate extensions only after the observation layer shows stable signal.
2. Optional router/oracle-gap dashboards if branch metrics show enough signal.
3. Coverage-aware channel search using branch metrics plus existing Engine-Scope coverage rows.

Follow-up implementation slices completed after the overlap audit:

1. Added HeavySkill-inspired branch-set metrics over channel variation records: `pass_at_k_rate`, `safe_pass_at_k_rate`, `regressed_at_k_rate`, `mean_best_delta`, `mean_branch_delta`, and `mean_oracle_gap`.
2. Added `build_overlay_report()` so Engine-Scope rows can produce channel records, summaries, and branch metrics through one call.
3. Wired `adaptive_overlay` reports into `run_thousand_query_tuning.py` final and checkpoint artifacts.
4. Wired `adaptive_overlay` reports into golden-default autopilot reform validation, mask validation, and hard-negative replay artifacts.
5. Added regression tests for channel records, branch metrics, overlay report construction, and tuning-report integration.

Defer:

1. spawning agent swarms
2. readable skill execution inside the engine
3. base-model RLVR or heavy-mode-aware model training
4. any default route that lets a deliberator override retrieval without verifier evidence
