# HeavySkill Engine Adaptation Review

Date: 2026-05-05

Sources reviewed:

- Paper: `HeavySkill: Heavy Thinking as the Inner Skill in Agentic Harness`, arXiv:2605.02396v1, 2026-05-04.
- Reference repository: `https://github.com/wjn1996/HeavySkill`, HEAD `649a5b4af6e941c358f0ad6f019010c5acd0e28c`.

## Executive Read

HeavySkill argues that much of the useful behavior inside modern agentic harnesses can be reduced to a two-stage test-time scaling pattern:

1. generate multiple independent reasoning trajectories for the same task
2. serialize those trajectories into a memory cache and ask a deliberation model to synthesize the final answer

The important transfer into ChelatedAI is not "spawn more agents." The useful transfer is a new evidence unit: a replayable trajectory set plus a deliberation record, scored by verifiable outcomes, budget telemetry, diversity diagnostics, and fail-closed promotion gates.

For this repo, HeavySkill should become an observation-first engine layer that sits beside the current Model-Scope and Engine-Scope work. It should not promote a new default until it can prove that trajectory width and deliberation depth beat the existing baseline on held-out retrieval and reasoning tasks without quantization, latency, or active-negative regressions.

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
| Sequential deliberation | Comparator reports and promotion summaries | No deliberator model/stage that consumes a cache and emits a replayable synthesis | P1 |
| Iterative deliberation | Autopilot loops and repeat-seed validation | No per-query depth state or convergence/stop reason for deliberation loops | P1 |
| Heavy-Mean/Pass/Vote metrics | NDCG@10, MAP@10, MRR, recall, quantization gate | No heavy-thinking metrics that separate sample availability, voting recovery, and deliberation recovery | P0 |
| Verifiable rewards | Promotion contract, holdout reports, quantization promotion gate | Need verifier adapters per task family before RLVR claims are safe | P0 |
| Trajectory diversity | Balanced AttnRes profile set and active-negative mining | No semantic/answer diversity diagnostics over candidate trajectories | P1 |
| Deliberator model selection | Model-Scope runtime and provider probes | No policy for cheap vs strong deliberators or cross-model pairing | P2 |
| Heavy-mode-aware RL | Sedimentation, adapter tuning, fail-closed promotion | Too early until logs prove which width/depth decisions correlate with reward | P3 |
| Readable harness skill | Local docs and Codex skill-like handoffs | Need a repo-native protocol file only after schemas are stable | P2 |

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

## Adaptation Plan

### P0: Observation-Only Heavy-Thinking Evidence

Add dataclasses or typed JSON builders for `TrajectoryRecord`, `TrajectoryCacheManifest`, `DeliberationRecord`, and `HeavyThinkingRun`. Wire them to the existing evidence/artifact style without changing any default retrieval behavior.

Acceptance bar:

- deterministic hashes for records and cache manifests
- JSON schema versioning
- unit tests for serialization, clipping provenance, and cache hash stability
- no promotion path enabled

### P1: Retrieval-Aware Parallel Trajectory Runner

Build a small runner that treats existing retrieval branches as trajectories: baseline, reformulation variants, mask variants, AttnRes profiles, and Model-Scope-observed runs.

Acceptance bar:

- emits trajectory records for every branch
- computes `Mean@K`, `Pass@K`, and `Vote@K` analogues for retrieval metrics
- can replay a fixed query slice with identical cache manifests

### P2: Deliberation Adapter

Add a deliberation stage that consumes a cache manifest and emits a final candidate answer or retrieval action. Start with text-only deliberation over summaries and metric traces, not raw hidden-state dumps.

Acceptance bar:

- deliberator prompt is versioned
- output is verifier-scored
- failure classes include `majority_wrong`, `minority_recovered`, `all_wrong_rederived`, and `deliberation_regressed`

### P3: Promotion Gate Extension

Extend the promotion contract to understand heavy-thinking evidence.

Acceptance bar:

- promotion fails closed when replay, holdout, verifier, quantization, or budget-normalized lift is missing
- heavy-thinking lift must survive repeat seeds
- no active-negative blocker can be masked by a high aggregate score

### P4: Coverage-Aware Width And Depth Search

Use diversity and low-pass-rate cohort labels to decide where more width or another deliberation iteration is worth the budget.

Acceptance bar:

- stop conditions are explicit
- width/depth increases are justified by observed uncertainty or disagreement
- budget-normalized lift is reported beside raw lift

### P5: Heavy-Mode-Aware Training

Only after the above exists, train policies that choose width, trajectory source mix, cache selection strategy, and deliberator model.

Acceptance bar:

- reward functions are task-verifiable
- training is replayable from records
- learned policy remains advisory until promotion gates pass

## Claim Boundaries For This Repo

HeavySkill's most convincing results are on verifiable reasoning tasks. CHELATEDAI's current strongest benchmark surface is retrieval. That means we should adapt the evidence architecture and metrics, not claim direct performance transfer.

Safe claim:

> ChelatedAI can evaluate heavy-thinking style breadth and deliberation as a typed, replayable, fail-closed engine mode.

Unsafe claim for now:

> HeavySkill proves that more agents or more samples will improve our retrieval engine by default.

The next practical slice is P0: typed heavy-thinking evidence and cache manifests. It is small, testable, and sets up every later experiment without changing production defaults.
