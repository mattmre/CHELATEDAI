# Shared test contract for IQ-20260912

**Status:** `PLANNING_REQUIREMENTS / NOT_A_MECHANICAL_GATE / NO_RUN_AUTHORIZATION`.

Applies to all six packets in the [queue](README.md). Existing frozen protocols, data access rules, resource ceilings, and [repository restrictions](../../next-session.md) take precedence. A proposed threshold below is a design choice to review before results, not an experimentally justified constant or a retrospective amendment.

## G0 — separate construction and execution admission

`G0-BUILD` authorizes a specifically scoped harness-construction task only after applicable repository restrictions are resolved/dispositioned, source scope and experiment design are reviewed, DEVELOPMENT-only access is defined, and an implementer/reviewer is assigned. It does not require final executable hashes before code exists, and does not authorize a scientific campaign or REPORT access. Focused correctness tests must stay within that separately approved construction scope.

`G0-RUN` follows harness correctness and review. It requires the complete record below, including exact executable source, frozen data/analysis, measured resource limits, and operator run approval. A packet saying "after G0" means the gate for that task's actual action; a build approval never substitutes for run approval. Neither gate has been granted in this queue-preparation task.

The future run owner must provide one reviewed record containing:

- exact authorized scope: literature/reduction, harness construction, METHOD_DEV, SELECT, REPORT, replication, or hardware; permitted actions and prohibited services;
- canonical worktree and clean execution checkout or explicitly reviewed complete payload, Git SHA, file digests, environment/package lock, model/tokenizer/checkpoint revisions and licenses;
- applicable block/debt and predecessor dispositions, with evidence pointers; no status inferred from a green test or file existence;
- exact operators, baselines, primary estimand, practical-effect threshold, noninferiority margins, independence units, power/precision design, multiplicity policy, and deterministic selection rule;
- new fixture manifest, split/group assignment procedure, access roles, observation-timing table, evaluator contract and solvability evidence; no recycled exhausted REPORT;
- exact executable entry point and verification command **after they exist**; exact output path/run ID with absent-target check, artifact schema and verifier;
- assigned implementer, independent reviewer, operator approval reference, hardware identity, measured pilot profile, finite wall-time/CPU/RSS/VRAM/storage/model-call limits, checkpoint and stop procedure;
- a declaration that existing services will not be stopped/reconfigured, data will not be exported, and paid resources will not be provisioned without separate authorization.

No launch command is supplied in these documents because the new harnesses are not implemented or source-frozen. Existing modules are reuse candidates, not asserted working entry points for these experiments.

## Data and observation boundaries

Use MODEL-TRAIN to fit model/controller parameters, DEVELOPMENT for fixtures and tuning design, SELECT for permitted policy/feature selection, and an unopened REPORT for final evidence. A source protocol that assigns predictor fitting to SELECT retains that assignment. Proposed new lanes default to disjoint generation seeds, semantic templates, entities, and source lineages; splits are made at the independent problem family/group before expanding into repeated views, variants, chains, or tokens.

| Availability class | Permitted use | Prohibited substitution |
| --- | --- | --- |
| `PRE_COMPOSITION` | Static component metadata and approved probes on individual components or explicitly allowed calibration pairs. | Measured outcomes of the held-out candidate chain. |
| `PRE_NEXT_STEP` | Prefix and already-computed state available when choosing the next update; count this computation. | Future tokens, final correctness, retrospective trajectory labels. |
| `EVALUATOR_ONLY` | Scoring, oracle upper bounds clearly segregated from deployment controls. | Gate inputs, route hints, confidence calibration, shared-basis fitting. |
| `POST_OUTCOME` | Explanatory diagnostics and error analysis after evaluation. | Advertising the same values as prospective predictors. |

When query groups alone are disjoint but all use the same adapters, report **new-query performance**, not unseen-adapter generalization. Distinguish held-out combinations of known components, unseen component identities, unseen domains, and unseen model families. They require different partitions.

Any candidate-only metadata or target-derived representation invalidates an information-matched utility claim. Intentional oracle and negative controls must be visibly labeled and cannot win the deployable method selection.

## Evaluation and inference

- Metric validity comes first. Retrieval uses the [metric-lineage protocol](../metric-lineage-repair-protocol-2026-07.md), complete qrels, canonical unique IDs, and frozen tie handling. No legacy quarantined score supplies a threshold.
- New binary task outcomes use paired per-problem comparisons with fixed scoring. Model-generated answers require an independent scoring rule, not a candidate generating and judging its own evidence.
- Count uncertainty at the independent unit. Resample full related chains/variants together; report between-seed/family variation separately. Three seeds are a METHOD_DEV robustness screen, not evidence of independent task generalization.
- Defaults for new lanes: predeclare one primary comparison against the strongest SELECT-chosen ordinary control, paired group-bootstrap intervals with 10,000 resamples, and a 95% two-sided interval. Secondary comparisons remain descriptive unless multiplicity is specified before REPORT. Preserve the CRSV protocol's one-sided 97.5% hierarchy and nested bootstrap instead of applying this default to it.
- For proposed effect gates, require both the minimum point effect and the stated confidence/guardrail condition. For noninferiority, the relevant interval—not just its point estimate—must respect the margin. Do not convert inadequate precision into a mechanistic negative.
- Prospective power or precision simulation uses DEVELOPMENT/SELECT dispersion, never REPORT. If the required sample size exceeds the admitted cap, stop as `INCONCLUSIVE/NOT_ADMITTED`; do not quietly shrink the scientific claim or enlarge the budget.
- No early stopping for significance, repeated REPORT-driven tuning, pooling invalid runs as model errors, or replacing the preregistered endpoint with a more favorable one. A code defect after REPORT exposure needs a documented amendment and appropriate fresh confirmation data.

## Resource and comparison accounting

Report training tokens, parameter counts, tuning attempts, failed runs, selection data, probe calls, preprocessing, total inference/replay operations, warm/cold latency, p95 latency, peak process-tree RSS, VRAM, and all stored auxiliary bytes. State both parameter-matched and compute-matched comparisons if exact matching of both is impossible.

The initial budgeting proposal for a **new** learned pilot is one accelerator, no distributed training, and at most 24 accelerator-hours total including failed attempts. This is a proposed ceiling to accept or replace explicitly after a DEVELOPMENT profile, not a reservation or throughput prediction. Each packet further limits problem size/configurations. No paid-cloud spend is approved. CPU-only screening gets its own finite limit; existing RB-13/RHPC caps remain stricter where specified.

A pilot uses one candidate family, at most four preregistered tunable configurations per trainable arm and three training seeds unless its original protocol says otherwise. All attempts count. No hyperparameter search begins until the complete matrix fits the approved limits. A small existing model need not be loaded if a deterministic or tiny learned test can answer the question first.

Matched FLOPs are not measured latency or joules. GPU/Spark results cannot prove Z1 energy efficiency. Projecting after a dense block does not erase that block's cost. Identical clock counts do not establish effective independent sample throughput.

## Evidence artifacts to implement later

The future run must retain a manifest binding protocol/code/environment/data/model/config/result bytes; a complete per-independent-unit result table with split and seed; aggregate estimates and uncertainty; full resource/timing records; admission receipt; failure/partial-run disposition; and an independent replay/verification report. These are requirements, not a claim that a shared exporter or validator already exists.

Keep append-only run IDs and absent final output targets; never overwrite evidence. Preserve checkpoints and failures with non-success status. Verifiers must test leakage, wrong source/digest, repeated REPORT use, incomplete cells, invalid numeric values, mismatched controls, and resource termination. A hash-shaped string is not authenticated provenance.

## Evidence levels and continuation

### Mathematical extensions and numerical evidence — 2026-10-08

Selected findings use [MA0–MA7](math-findings-integration-2026-10-07.md), [source cards OM0–OM8](sources.md#mathematical-findings-integration--2026-10-07), and the [case register](math-test-cases-2026-10-07.csv). A queued case is a specification, not passing test coverage. New feature/claim admission must record the exact mathematical statement and definitions, source verification state, applicable operator assumptions, precision/normalization/cache boundary, information availability, strongest ordinary control, and certificate/probe cost.

Prefer the elementary same-input perturbation bound and established Crouzeix–Palencia constant initially. The released constant-two bound or other new classifications cannot be labeled locally verified from source hashes or challenge templates. Fixed-linear, switching, nonlinear, coordinate-protection, greedy-output, stochastic-output, and semantic claims require separate justifications. Learned confidence and sampled numerical-range points are not certified bounds.

Retained-byte claims require a complete physical/logical state ledger including shared bases, precision metadata, caches, and external replay. Numerical receipts must bind the exact source/build and arithmetic/tie policy to all expected case IDs and every fallback record. Missing/duplicate cells, arithmetic mismatch, or resource exhaustion cannot produce complete-certificate status. A new theorem or numerical receipt does not replace metric-lineage repair, ordinary controls, or independent task-utility evidence. These additions apply to reviewed extensions; frozen protocols are unchanged.

1. Algebraic identity/reduction: can close an overbroad claim without training.
2. Harness correctness: supports execution behavior only.
3. METHOD_DEV signal: justifies reviewing a frozen confirmation proposal, not efficacy or novelty.
4. Frozen held-out endpoint: supports only its declared population and estimand.
5. Independent task/model-family replication: supports a broader empirical claim within tested limits.
6. Specialist prior-art and equivalence review: needed before novelty language; never guaranteed by an unsuccessful literature search.

The existing [QSCCI follow-up](../qwen-scope-nuisance-rank-separation-preregistration-draft-2026-08-16.md) remains separately queued and review-only. Oracle-assisted controllability cannot substitute for label-blind correction. Existing EGV and BCC redesign gates and closed prime-ring/RB-15 negatives are not reopened here.
