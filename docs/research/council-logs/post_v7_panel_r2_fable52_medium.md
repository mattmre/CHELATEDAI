<!--
{
  "taskId": "node:delegated-task:command%3Amcp%3Addeb746b-9684-4a04-bcfe-727030442015%3Adelegate-task%3Apost-v7-panel-r2-fable52-medium-20261124",
  "link": "[Post-v7 Panel R2 Fable 5.2 Medium : Cross-Model Challenge of Astra 6...](t3-thread://v2/c5bb8594-9c79-4948-9251-bd83e8527cf6/thread:delegated-task:command%3Amcp%3Addeb746b-9684-4a04-bcfe-727030442015%3Adelegate-task%3Apost-v7-panel-r2-fable52-medium-20261124)",
  "model": "claude-fable-5-2",
  "options": [{"id": "effort", "value": "medium"}],
  "status": "completed"
}
-->

---
panel_type: cross-model challenge and consensus lock
artifact: "Post-v7.0 expansion wave queue, Items 1–6"
run_date: 2026-11-24
run_id: post-v7-r2-fable52-statistics
round: 2_of_2
challenge_depth: critical
peer_examined: "Astra 6.2 Medium, Round 1"
tracks_retained: 17
bundle_range: "#47–#64"
tools_used: none
---

**All 17 tracks and the `#47` spine are retained.** I worked from Astra's verbatim report and the brief's summary of my own Round 1; I did not re-read either thread. The "verified facts" are taken as given, not re-checked.

## What Astra caught that I missed

- **Denominator.** Seventeen empirical tracks plus one engineering packet. Counting Core as an experiment inflates the evidence.
- **Independent oracle.** Parity between two implementations that share decision code proves nothing. This governs 6C and 3B.
- **Sampling units.** 3B's 400 questions sit in 20 conversations, so n = 20.
- **Hop arithmetic.** `v1→v4` is three transitions; four needs a fifth encoder.
- **2E.** An unchanged AST cannot certify KV reuse after an earlier-prefix edit.
- **2D.** `1024M` is exactly 1 GiB, so a "sub-1 GiB" claim needs a strictly smaller limit.

## Where Astra's rules misfire without refinement

1. **Holm across three primary families.** "Governance loss at natural prevalence" depends on partner data (5.2) that may never arrive, and "zero-services adoption" on unaffiliated humans we do not have. Two of the three primaries would stay permanently open. T1 must contain only claims runnable in-house.
2. **"Audit previous exposure" (6B).** A manual audit is what let `ArguAna` through; neither report caught it mechanically. The gate must be a `burn_ledger` verifier that fails closed on any dataset ID in any prior run script.
3. **3A sample sizing.** The disjoint pool is finite. If the required paired N exceeds it, the track reports "underpowered"; the −0.0500 margin may not be loosened afterwards.
4. **3D's "<2%" bound.** Arithmetically right (1 − 0.05^(1/149) ≈ 1.99%), but it bounds false alarms only. It says nothing about sensitivity or lead time and must not be quoted as detector quality.
5. **Spark #2 for 3C/3D.** The verified topology names Spark #3 as the clean host for the `Qwen3.5-2B-Base` probes.
6. **"Preserve resident containers" plus exclusive leases.** A resident ~24 GiB Nemotron on unified memory competes for memory bandwidth even when idle. Each timed receipt must log resident request counts over the window; a nonzero count voids the timing.

## R2-Q1: canonical reservation table

I adopt Astra's IDs verbatim and withdraw my six-wave numbering. IDs are permanent and are not execution order.

| ID | Track | ID | Track |
|---|---|---|---|
| `#47` | Core spine + A/A fixtures | `#56` | 4 |
| `#48` | 6A | `#57–60` | 3A, 3B, 3C, 3D |
| `#49` | 6D | `#61` | 6B |
| `#50` | 1 (`D12`, `/fabric`) | `#62` | 6C |
| `#51–55` | 2A–2E | `#63–64` | 5.1, 5.2 |

**A/A placement.** It gets no ID of its own, since `#65` would become an eighteenth "experiment". The A/A null lives in `#47` as a verifier fixture. Every harness (retrieval, storage, ledger) runs identical-arm pairs, and each track manifest must cite an A/A receipt on its own host showing a null-rejection rate within the binomial band of nominal α. A missing receipt leaves the bundle unverified.

**Fragment contract.**
- `registry.d/<id>-<track>.json` holds bundle ID, Core wheel digest, tier, prereg hash and status. Status is one of `verified-pass`, `verified-loss`, `unsupported`, `protocol-ready`.
- `routes.d/<id>.json` holds route declarations only.
- `verifiers.d/<id>.py` holds one offline entry point.
- Only the integration worktree writes the merged files. It rejects duplicate IDs, routes or paths, dependency cycles and digest mismatches.
- `verified-loss` keeps its ID permanently.

## R2-Q2: 6B hypotheses

I withdraw my Round 1 rule (`ZeroMu+CSLS` > `UncenteredProcrustes+CSLS` on ≥10/12 static domains). It pre-registered a superiority claim that Wave 6 evidence already contradicts, which invites a post-hoc endpoint swap. The replacement is three decomposed hypotheses:

- **`H6B_merge` (T1, superiority).** CSLS vs CalibratedZScore with the bridge held fixed, on paired nDCG@10. It passes iff the pooled cluster-bootstrap lower bound is above 0 and at least 10 of 12 units are positive. The sign-test p for 10/12 is 79/4096 ≈ 0.019.
- **`H6B_bridge_static` (guard, non-inferiority).** `ZeroMu+CSLS` vs `UncenteredProcrustes+CSLS` on unshifted 1-hop data. The margin is frozen before REPORT; I propose −0.5 pp nDCG@10.
- **`H6B_bridge_shift_4hop` (T1, superiority).** The same pair, without refit, under two conditions:
  - a frozen list of public RAG prompt wrappers and tenant templates, chosen before any shifted run;
  - four-transition composition `v1→…→v5`, direct vs composed at each transition.

Two mandatory comparators neither report had:
- **Mean-centred Procrustes.** If it ties ZeroMu, the result is "centring helps", not a ZeroMu claim.
- **Uncentred Procrustes refit on shifted anchors**, reported as the cost ceiling.

`TRA-1` ABI gating is a verifier check, not a hypothesis.

**Retirement.** If static fails, ZeroMu is not the static default. If shift/4-hop fails, ZeroMu is retired to "equivalent option". If merge fails, CSLS's `+2.69 pp` is recorded as not replicated.

**Dataset list.** Seven corpora are burned: `msmarco, quora, fiqa, scidocs, arguana, scifact, nfcorpus`. I cannot certify any candidate as unburned without the ledger, so this list is conditional on a mechanical `burn_ledger` pass:

- Candidates: TREC-COVID, Touché-2020, DBPedia-Entity, Climate-FEVER, one CQADupStack subforum, LitSearch, BioASQ-TaskB, PubMedQA, MIRACL-zh, MIRACL-ar, MIRACL-sw, CoIR-Apps.
- Alternates, in frozen order: NQ, HotpotQA, FEVER, Mr.TyDi-ja.
- BioASQ is registration-gated, so check its licence before use.
- The three MIRACL units share a Wikipedia source and count as one cluster in the bootstrap.

## R2-Q3: 6C endpoint

Agreed: an independently implemented PostgreSQL 17 challenger offering both indexed `WITH RECURSIVE … CYCLE` and an incremental closure table, with production indexes, equal durability and equal tuning hours.

One point neither Round 1 report stated: two correct implementations of the same `as_of` semantics must agree on dispositions. EK5 therefore cannot win on correctness unless the SQL challenger is handicapped. The primary endpoint is joint:

1. **Correctness guard.** On a blinded, adjudicated sample from frozen dated snapshots, `as_of(valid_time, tx_time)` disposition agreement is measured in two strata:
   - unlinked cross-mirror positive-support echo cycles;
   - equal-authority `VEX not_affected` vs `GHSA affected`, where the correct output is isolated `Disputed`.
   
   EK5 must be non-inferior to the better challenger (paired McNemar), with zero cross-tenant or stale-authority admissions.
2. **Cost superiority.** One endpoint, frozen in advance: p95 `as_of` latency under concurrent ingest, or incremental maintenance work per withdrawal event.

SEC is a separate stratum. Only true Item 4.02 non-reliance filings count as restatements; `10-K/A` and `8-K/A` amendments are tallied separately. If the adjudicated natural-prevalence count falls below the frozen minimum, the stratum is descriptive only.

**Retirement.** If the tuned challenger matches adjudicated dispositions within margin and sits within 1.25× on the cost endpoint, EK5's claim becomes "reference semantics, reproducible in stock SQL". That result is published as `verified-loss` in `#62`.

## R2-Q4: tiers and per-track lock

Astra's gates and my "no win language" rule are reconciled as follows:
- **T1** carries the family-wise α = 0.05 Holm claims.
- **T2** may state "superior to *named comparator* on *named endpoint*" with within-track Holm and paired intervals. It may not use unqualified "win" or cross-track aggregate language.
- **T3** is exploratory.
- **T4** is protocol-ready only.

The T1 family has four members: 3A, `H6B_merge`, `H6B_bridge_shift_4hop` and 6C. 6D is a deterministic gate, not a test.

N is locked as a rule wherever effect sizes are unverified. Inventing integers would be fabrication.

| Track | Tier | Unit and N | Primary endpoint vs comparator | Retirement |
|---|---|---|---|---|
| 1 | T2 | ≥1,000 randomized traces, all prefixes | Oracle disagreements = 0; per-hop recall vs direct bridge | Any disagreement |
| 2A | T2 | ≥30 paired prompts × cold/warm | Tokens/s at exact greedy equality vs same-target uncapped | Equality fails |
| 2B | T2 | 250 materialized payloads | Time to first usable inference vs bandwidth bound; revoked-execution upper CI | Any revoked execution |
| 2C | T2 | ≥5 seeds × withdrawal grid | Cold-vector rewrites at matched recall vs tuned Qdrant | Tie or loss |
| 2D | T2 | Held-out source families, ≥30 sessions | Paired stale admission and clean-recall loss vs DEV-selected C3 | CI includes 0 |
| 2E | T2 | ≥200 replayed edits | TTFT, conditional on logit tolerance, vs full prefill | Any tolerance breach |
| 3A | T1 | Power-sized disjoint queries | Non-inferiority at −0.0500 vs canonical | Underpowered or lower bound below margin |
| 3B | T3 | 20 conversations | Decision agreement with oracle after each event | Descriptive only |
| 3C | T3 | One fresh REPORT | Norm-matched causal effect vs random direction | CI includes 0 |
| 3D | T3 | 149 benign families + events | False-alarm upper bound; lead time reported separately | No efficacy claim |
| 4 | T2 | Each eligible job once | On-time-correct rate, factorial, vs no-bridge/no-governance | Interaction CI includes 0 |
| 5.1 | T4 | Partner-frozen | Equivalence within partner margin vs partner alternative | Protocol-ready only |
| 5.2 | T4 | Consecutive matters | Paired owner-weighted loss | Protocol-ready only |
| 6A | T2 | Per pinned backend | End-to-end p95 at matched recall and load, with pushdown receipt | No pushdown evidence |
| 6B | T1 | 12 units, cluster-adjusted | See R2-Q2 | See R2-Q2 |
| 6C | T1 | Adjudicated sample, frozen minimum | See R2-Q3 | 1.25× tie rule |
| 6D | Gate | 36 runs | See R2-Q5 | Below threshold |

## R2-Q5: 6D cold-start benchmark

- **Design.** Three model families (at least one with no part in building the SDK) × four withheld vertical fixtures × three seeds = 36 runs.
- **Environment.** A clean container with an offline package mirror and no repo access. Inputs are the frozen `.whl`, the public docs at the same digest, and the task statement.
- **Sealing order.** Fixture hashes are sealed before the wheel freezes, and the docs freeze before fixtures are unsealed. A similarity check rejects any fixture answerable by copying a docs example.
- **Canary.** One sub-task needs a capability the wheel lacks. The only correct outcome is a typed `unsupported`.
- **Scoring.** A hidden scorer records elapsed time, caller LOC, configuration lines, errors and failed attempts. Private imports are caught by a static scan plus a runtime import hook.
- **Pass rule.** At least 30 of 36 complete, every family ≥ 8/12, zero private imports, zero canary false successes.
- **Tenant isolation.** A two-tenant fixture carries planted canary records; any leak fails the gate.
- **SBOM/licence.** The SBOM is generated from the wheel, checked against a licence deny-list, and publishes the transitive dependency count. Upgrade, rollback and offline install each run once per platform.

Agents are not buyers. The claim is labelled "agent cold-start, zero author assistance". Astra's unaffiliated-human tier stays open until real operators exist.

## R2-Q6: execution DAG

- **Phase 0, Desktop.** Verify the v6.0 45/45 and v7.0 46/46 manifests. Build `#47`, the `burn_ledger` verifier and the A/A fixtures. Freeze preregistrations.
- **Phase 1, parallel.**
  - Spark #1: A/A, then 2C, then 6A.
  - Spark #2: A/A, then 2A, then 2B, then 2E.
  - Spark #3: A/A, Item 1 embedding precompute, then 3C, then 3D.
  - Desktop: Item 1 replay, 5.1 and 5.2 kits.
- **Phase 2.**
  - Spark #1: 6C, then 3B.
  - Spark #3: 2D, then 3A, then 6B.
  - Desktop: wheel freeze.
- **Phase 3.**
  - Spark #3: Item 4, alone on the whole worker pool.
  - Spark #2 (Linux containers) and Desktop (Windows): 6D.
  - Open confirmatory REPORTs once, then run integration and `verify_all_offline`.

All `cgroup v2` and `dm-delay` work runs on the Sparks under exclusive leases. `dm-delay` results are labelled as an emulation tier.

## Residual dissent

1. The 1.25× tie band and the −0.5 pp margin are my proposals, not evidence. The owner must freeze them before any run.
2. Twelve corpora cannot support "universal" language under any outcome.
3. T2 gains two parallel-ready tracks only if Spark #2's resident `glm53-remote-nfs` is quiet. If it is not, 2A and 2B move to Spark #3 and Phase 2 slips.
