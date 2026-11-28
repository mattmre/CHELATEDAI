<!--
{
  "taskId": "node:delegated-task:command%3Amcp%3Addeb746b-9684-4a04-bcfe-727030442015%3Adelegate-task%3Apost-v7-panel-r1-fable52-medium-20261124",
  "link": "[Post-v7 Panel R1 Fable 5.2 Medium : Anti-Overfitting, Statistical Ri...](t3-thread://v2/c5bb8594-9c79-4948-9251-bd83e8527cf6/thread:delegated-task:command%3Amcp%3Addeb746b-9684-4a04-bcfe-727030442015%3Adelegate-task%3Apost-v7-panel-r1-fable52-medium-20261124)",
  "model": "claude-fable-5-2",
  "options": [{"id": "effort", "value":"medium"}],
  "status": "completed"
}
-->

```yaml
panel_type: panel-of-experts (cross-model critical & architectural review)
artifact: Post-v7.0 Expansion Wave Queue (Queued Items 1–6, additive multi-branch mandate)
run_date: 2026-11-24
run_id: post-v7-wave-queue-r1-fable52-medium
round: 1_of_2
challenge_depth: critical
active_experts: [principal_diligence_auditor, econometrician, epistemic_probe_methodologist]
```

**Basis:** I worked from the specification in the brief only. I opened no files and ran no commands, so every figure quoted from it (`-0.0572`, `62/70`, `940 ms`, `+0.0269`, `cos=1.0`) is unverified by me. Nothing below prunes or parks a track.

**Count defect:** the enumerated sub-tracks total 17 (1 + 5 + 4 + 1 + 2 + 4), not 18. Either split Item 1 into Scenario A and Scenario B, or correct the number; a wrong denominator in a pre-registration is the first thing hostile diligence finds.

## Q1. Topology, namespacing, trunk isolation

- **One spine branch.** Cut `pv7/spine` (Wave 16-Core) from the sealed `v7.0` tag. Every track branch (`pv7/t1-fabric`, `pv7/t2a-diskmoe`, … `pv7/t6d-sdk`) forks from a tagged spine release and never merges sideways.
- **No track edits the three hot files.** Spine converts them once to discovery:
  - `verified_registry.json` becomes a generated file compiled from `registry.d/<bundle>.json`.
  - `server.py` loads routes from `routes.d/`.
  - `verify_all_offline.py` enumerates verifier manifests.
  - Each track then only adds files, so collisions cannot occur.
- **Bundle IDs are reserved before any data runs.** `#47` is the spine plus Claim Registry; `#48–#64` go one per sub-track in fixed order. An ID is never vacated or reassigned: a null or failed track seals as a negative bundle under its own ID. This is the structural defence against cherry-picking.
- **Waves:** 16 spine; 17 core (6A, 6D); 18 packets (3A–3D, 6B, 6C); 19 storage (2A–2E); 20 verticals (1, 4; D12, D13); 21 partner kits (5.1, 5.2).
- **Spine changes** after tag `spine-1.0` need a version bump, and every already-sealed dependent bundle re-verifies.

## Q2. Wave 16 Core Spine (build once)

| Spine module | Consumers | Fairness requirement |
|---|---|---|
| `challenger_ledger`: Postgres 17 bitemporal + authority ledger | 1, 2D, 4, 5.2, 6C | Two forms: query-time `WITH RECURSIVE` (with `CYCLE` guard and parent/valid-time indexes) and an incrementally maintained closure table. Tuned by a fresh agent with the same budget as `EK5`. A SQLite parity build serves 2D's 1 GiB cap. |
| `bridge_merge_harness` | 3A, 5.1, 6A, 6B | Full factorial of every bridge by every merge, including raw. One anchor sampler; no bridge is denied `CSLS`. |
| `span_profiler` | 5.1, 6A, 2C, 4 | Same span taxonomy everywhere. |
| `ek5_reason_canonicalizer` | all `EK5` exporters | Lands before any export, or the 153–366 diffs propagate into six bundles. |
| `stats`: paired and cluster bootstrap, non-inferiority, Holm, exact binomial | all | One implementation; no per-track test selection. |
| `burn_ledger`: hash manifest of every doc/query ID ever used for tuning | 3A, 6B, 2C, 2D | Enforces "untouched" mechanically. |
| `cpu_io_cage`: cgroup, cpuset, `io.max`, cache drop, `/proc/<pid>/io` | 2A–2E | One runner for all five. |
| `claim_registry` + prereg hasher | all | See Q5. |
| `chelated-kernel` (6D) | 1, 2B–2E, 4, 5.x | Verticals import only the public SDK surface. Any import of internals is logged as a services dependency. |

## Q3. Track-by-track traps and in-track fixes

| Track | Trap | Fix inside the track |
|---|---|---|
| 1 | Four of five arms are strawmen. `wfs_oracle_mismatches == 0` is self-consistency against an oracle you authored. `\|κ−1\| < 1e-5` holds by construction for orthogonal maps. A 384→4096 map from 500 anchors has rank ≤ 500. | Add spine ledger and `Procrustes+CSLS` arms. Have a separate agent author the scenarios before arms exist. Label κ as an invariant. Report 4096-d hop quality against a direct fit. |
| 2A | "≥3.0x" has no named comparator among six rungs. `HumanEval` at N=164 has roughly ±7 pp resolution. `dm-delay` adds constant per-bio delay, so it cannot reward cylinder ordering. `O_DIRECT` does not govern mmap paths. | Comparator is the best of rungs 1–4 after equal tuning. Add greedy token-exact match against the Q4 reference. Make cylinder-order claims only on physical `ROTA=1`. |
| 2B | Title says 1,000 cartridges, build says 250 "real/seeded". "0.00%" has no N. | Headline the real count. Report seeded clones separately. State the exact-binomial upper bound. |
| 2C | `Qdrant on_disk` HNSW on spindles is a layout strawman. Speed without matched recall is meaningless. | Add an on-disk IVF-SQ8 challenger with the same extent size and RAM budget. Compare at matched recall@10. |
| 2D | `EK7` is in-house and has been seen before. C3's tuning budget is unstated. | Fresh held-out split via `burn_ledger`; equal tuning budget for C3. |
| 2E | One repo is n=1 with autocorrelated commits. Non-prefix KV reuse is not equivalent to re-prefill. | At least five repos chosen by a hash rule; cluster bootstrap by repo. Add a logit-equivalence gate against full re-prefill, and a stable-files-first prefix-LRU challenger. |
| 3A | 100 queries cannot resolve a 0.0072 gap in a lower bound. Rerunning until it crosses is optional stopping. | Compute n from the paired SD in PM-2 and fix it. One analysis. PM-2 stays recorded as a miss; the new corpus is a new claim. |
| 3B | 400 questions sit in 20 conversations, so the effective N is 20. | Cluster bootstrap by conversation; descriptive tier only. |
| 3C | `cos=1.0` means the nuisance penalty was inert. The k sweep times five endpoints times two bases is a forking-paths design. | Positive control first: plant a nuisance-correlated feature and show the selector rejects it. Primary is k=1 with one endpoint. Norm-matched dosing. |
| 3D | 149 families gives a 1.99% one-sided bound only at zero false positives; one event gives about 3.1%. | Use at least 238 families to tolerate one event, or 183 for a two-sided bound at zero. Apply Holm across the 1-unit and 4-unit probes. |
| 4 | One scenario template. Live LLM variance swamps a per-minute rate. Stage 2 thresholds risk leaking into Stage 3. | At least 24 held-out templates. Record and replay worker outputs for attribution. Single primary contrast: the interaction under `both` at pool 4. |
| 5.1 | "Matches" is elastic. `+0.0269` is retrospective. | Pre-register numeric tie bands. Report on the engine before and after 6A. |
| 5.2 | Natural conflict rate is unknown, so N is unknown. | Size N as 30 divided by the base rate. Freeze the owner's loss prices before scoring. |
| 6A | The gain is repair of your own fallback path, not novelty. Filtered HNSW recall degrades at 90% retraction. A canonicalizer can erase real differences. | Report recall against exact search and p99 at each retraction level. Property-test that distinct causes stay distinct. Confirm the pinned `OpenSearch 3.9` and `pgvector 0.8.7` releases exist before pre-registration. |
| 6B | Several of the twelve are BEIR sets this project has plausibly touched. Encoder pairs are a free parameter. | Admit sets only through `burn_ledger`. Fix encoder pairs in the hashed config. For the 4-hop test, add a direct v1→v4 comparator. |
| 6C | If truth is the final advisory state, any correct ledger scores 100%, so a tie is the likely result. | Evaluate as-of `tx_time`. Pre-register the differentiator: echo collapse, incremental write cost, or explanation. |
| 6D | Line count and glue lines are gameable. | Cold-start test: fresh agents with docs only integrate each vertical; score success rate and time. |

## Q4. Fleet schedule

I do not know which hosts carry the resident Nemotron containers or their footprints, so this is a constraint-first schedule.

- **Timing rule.** Any bundle with a latency or throughput endpoint runs in an exclusive window on its host. cgroup memory caps do not isolate last-level cache, memory bandwidth, or disk queues, so pin cpusets, set `io.max`, and record host load in the bundle.
- **Phase 0, Desktop:** spine, Claim Registry, `burn_ledger`. Nothing else starts until `spine-1.0` is tagged.
- **Phase 1, parallel:**
  - Desktop: 6D, then 3C and 3D on GPU.
  - Spark #1: 6A (the databases live there), then 3A.
  - Spark #2: 2A cage calibration.
  - Spark #3: 6C ingest and the Postgres challenger.
- **Phase 2:**
  - Spark #1: 2C, then 6B, then 5.1 (all need the 6A engine and the MSMARCO shards).
  - Spark #2: 2A, 2B, 2E in sequence.
  - Spark #3: 2D, 3B, 6C replay.
  - Desktop: 5.2 kit.
- **Phase 3:** Item 4 on Spark #2 for live workers, with replay on Spark #3; Item 1 on Spark #3. Both depend on 6D and the spine ledger.
- **Critical path:** spine → 6A → {2C, 5.1, 6B}, and spine → 6D → {1, 4}.

## Q5. Tiered Claim Registry and falsification scorecard

Commit the registry to spine, hashed, before any track touches data. Each track gets one primary endpoint, one comparator, one N, and one retirement trigger.

| Tier | Meaning | Members | Error control |
|---|---|---|---|
| T0 | Invariants | κ identity, incremental==full parity, zero revoked execution | Any failure is a defect; no alpha spent |
| T1 | Confirmatory pure-tech claims | 6B, 3A, 6C, 2D, 6D cold-start | Holm at family-wise 0.05 |
| T2 | Reference architectures | 1, 2A, 2B, 2C, 2E, 4, 6A | Descriptive, with intervals; no "win" language |
| T3 | Exploratory probes | 3B, 3C, 3D | Hypothesis-generating only |
| T4 | Protocol-ready | 5.1, 5.2 | No claim until an outside owner runs it |

Scorecard rules:

- **Publish every row** as win, loss, tie, or timeout against the pre-registered trigger.
- **6B passes on a sign test** of `ZeroMu+CSLS` against `Procrustes+CSLS`, needing at least 10 of 12 domains (9 of 12 is p ≈ 0.07 one-sided).
- **Add an A/A null track** through the full harness to measure its own false-positive rate.
- **Carry prior retired claims** on the same scorecard.

Two limits to state in the scorecard itself. "Non-overfitted" cannot be exhaustively proven; it can only survive pre-registered attempts to break it. And zero-services cannot be shown by the team that wrote the code; only 6D cold-start and an outside-run Item 5 bear on it.

## Q6. Additive hardening inside Item 6

- **6B, anchor poisoning.** Replace 1, 5, 10, and 20% of the 500 anchors with adversarial or off-domain text. Report the degradation curve and whether `TRA-1` blocks the contract.
- **6B, anchor variance.** Twenty anchor resamples per domain. A result from one draw is one draw.
- **6C, echo precision and recall.** Use naturally occurring mirror pairs with provenance ground truth. Score false collapse of independent but near-identical corroborations, and missed paraphrased echoes.
- **6D, multi-tenant isolation.** Two tenants on one process: prove no cross-tenant anchor or ledger read, and measure noisy-neighbour p95.
- **6D, asset cleanliness.** Reproducible wheel build, dependency count, SBOM, and licence scan of code and benchmark data. An acquirer buys provenance as much as performance.
- **6A, recall floor.** A hard gate that the sub-35 ms result holds at the same recall as the exact-scan path it replaces.

**Round 1 verdict:** the queue is executable additively. It is not yet pre-registerable until four things are fixed: the track count, the unnamed comparators (2A, 5.1), the under-sized Ns (3A, 3D, 5.2), and the 3C selector positive control.
