<!--
{
  "taskId": "node:delegated-task:command%3Amcp%3Addeb746b-9684-4a04-bcfe-727030442015%3Adelegate-task%3Apost-v7-panel-r2-astra62-medium-20261124",
  "link": "[Post-v7 Panel R2 Astra 6.2 Medium : Cross-Model Challenge of Fable 5...](t3-thread://v2/c5bb8594-9c79-4948-9251-bd83e8527cf6/thread:delegated-task:command%3Amcp%3Addeb746b-9684-4a04-bcfe-727030442015%3Adelegate-task%3Apost-v7-panel-r2-astra62-medium-20261124)",
  "model": "gpt-6.2-astra",
  "options": [{"id": "reasoningEffort", "value": "medium"}],
  "status": "completed"
}
-->

```yaml
panel_type: panel-of-experts
artifact: Post-v7.0 expansion queue and Wave 16-Core
run_date: 2026-11-24
run_id: post-v7-panel-r2-astra62-medium
round: 2_of_2
challenge_depth: critical
active_experts: ["Astra 6.2 Medium reviewing Fable 5.2 Medium and Carbon 4 Medium findings"]
persist_version: null
decision: lock_protocol_design
empirical_status: untested
```

**Lock all 17 empirical tracks and the `#47` spine.** Fable caught two things my first review missed. The 3C selector needs a planted positive control before its null result is interpretable. The exposure ledger also needs veto power over corpus admission. The supplied D10 source finding makes that concrete for ArguAna. Fable’s clean-domain 10-of-12 rule, however, could retire a useful bridge for losing a contest in which its proposed advantage never arises. Conversely, it could certify the bridge without testing a single shifted query.

This is the Astra reviewer’s final synthesis of the two Round 1 reports and the supplied repository and fleet findings. No shell commands ran or files changed. The numerical designs below are proposed preregistration decisions. Corpus clearance, achievable power, outside adoption, and performance remain unverified.

## 1. Permanent reservations and integration contract

Adopt Fable’s six-wave publication order. It gives the shared spine its own release and places both customer protocols after the engines they evaluate. Keep Astra’s permanent IDs per deliverable and distinguish publication order from execution dependencies.

| Wave | Permanent reservations |
|---|---|
| 16 | `#47` Core, claim registry, exposure ledger, and A/A calibration |
| 17 | `#48` 6A served retrieval, `#49` 6D SDK |
| 18 | `#50` 3A, `#51` 3B, `#52` 3C, `#53` 3D, `#54` 6B, `#55` 6C |
| 19 | `#56` 2A, `#57` 2B, `#58` 2C, `#59` 2D, `#60` 2E |
| 20 | `#61` Item 1 `/fabric`, `#62` Item 4 pooled workers |
| 21 | `#63` 5.1 migration partner kit, `#64` 5.2 governance partner kit |

Each spoke forks from a tagged spine descended from sealed `v7.0`. `registry.d/<id>.json` declares track, spine digest, dependencies, preregistration hash, artifact hashes, outcome, and verification status. `routes.d/<id>.json` declares method, path, and public handler. `verifiers.d/<id>.json` declares offline entry point, input digests, and expected receipt schema. Empty route lists are valid. Only integration generates the registry and wires the loaders. Reject duplicate IDs, method/path pairs, escaping paths, cycles, and unpinned inputs. Preserve failures under their original IDs. Run identical-arm A/A replays inside `#47` across frozen seeds, reporting false alarms and uncertainty. One clean replay establishes only that replay’s result.

## 2. 6B decomposes the migration claim

Use equal-domain-weighted nDCG@10, paired queries, twenty frozen 500-anchor draws per domain, and identical search budgets. Anchor draws quantify sensitivity within a domain. They do not multiply the domain count. Freeze encoder revisions, fit chronology, wrapper text, and the full bridge-by-merge grid before scoring.

| Hypothesis | Prespecified contrast and decision |
|---|---|
| `H6B_merge` | On clean one-hop queries, CSLS must improve over CalibratedZScore averaged equally across the two named bridges. Report each bridge-specific effect and their interaction. |
| `H6B_bridge_static` | At fixed CSLS, ZeroMu’s lower confidence bound against UncenteredProcrustes must exceed −0.010 absolute nDCG@10. |
| `H6B_bridge_shift_and_4hop` | At fixed CSLS, require positive paired effects separately under held-out tenant/prompt wrappers without refitting and under clean four-hop composition. Report their crossed cell and direct `v1→v5` fit. Five frozen encoder states supply four transitions. |

Inference-time centering does not guarantee immunity to a nonlinear encoder’s response to a textual prefix. Measure the observed mean displacement and retrieval loss. Count ABI refusals and fallback in the served denominator, with accepted-only quality reported separately. The joint 6B claim requires every component to pass. Retain individual effects when it fails.

**Exact proposed admission slate:** TREC-COVID, NFCorpus-free BioASQ Task B (2024 document retrieval), LitSearch, MIRACL-zh, CoIR-Apps, Touché-2020, TREC-CAR, DBPedia-Entity v2, Robust04, Signal-1M, CQADupStack-Android, and FEVER passage retrieval. “NFCorpus-free” requires an overlap check on BioASQ documents and queries. The supplied D10 finding establishes prior use of MSMARCO, Quora, FiQA, SCIDOCS, ArguAna, SciFact, and NFCorpus. It cannot establish twelve clean replacements. Before outcome access, bind revisions and qrels, resolve aliases and derivatives, and compare document/query fingerprints against historical use. Any collision or unverifiable history leaves admission open and requires a dated replacement amendment before scoring. A verified-clean twelve-domain list is still outstanding.

## 3. 6C gives PostgreSQL a genuine opportunity to tie

Independently implement indexed PostgreSQL 17 `WITH RECURSIVE … CYCLE` and incremental closure. Share input envelopes, policy specifications, and durability settings. Give SQL independent decision code, equal tuning time, and DEV-only selection between implementations. Preserve both results.

Freeze **240 independently adjudicated families**, sixty each for unlinked mirror echoes, equal-authority VEX/GHSA conflicts, SEC Item 4.02 non-reliance disclosures, and amendment controls. Preserve wild NVD/GHSA/OSV/VEX snapshots and filing accession histories. For four frozen `(valid_time, tx_time)` cuts per family, the oracle admits only records received by transaction time and effective at valid time. It returns disposition and independent support lineages. A mirror cycle contributes no new corroboration. Equal-authority incompatible assertions about the same product/version yield `Disputed`. An amended filing alone cannot establish a restatement.

The primary endpoint is the fraction of families correct at **all four cuts**, including disposition and lineage. EK5 requires a positive adjusted lower bound on paired advantage over the selected SQL challenger. Equal family accuracy retires the superiority claim for this packet. An inconclusive interval also withholds the claim. Write cost and explanation quality retain separate rows. Sparse natural strata produce an incomplete receipt, never synthetic observations labeled wild.

## 4. Scorecard and retirement rules

T0 invariants fail on any counterexample. T1 applies Holm family-wise α=.05 across 3A, 6B, 6C, and 6D, using the largest component p-value for a conjunctive claim. T2 permits named workload-specific engineering wins under paired intervals and within-track Holm correction. T3 reports exploratory estimates. T4 requires outside execution. The following numbers lock enrollment targets, not post hoc power claims. Missing targets, failed gates, and timeouts stay visible.

| Track | Enrollment, primary comparison, and retirement rule |
|---|---|
| 1 · T2 | 100 paired traces per DPU/LEO scenario. Compare correct eligible deliveries against ledger plus Procrustes/CSLS. Require positive lower bound and zero oracle violations. Use public TLS flow records, real encoders, and dated CelesTrak schedules. Label software replay. |
| 2A · T2 | 164 HumanEval prompts, three paired runs each. Under 8 GiB require throughput-ratio lower bound ≥3 against the DEV-best of rungs 1–4 and greedy Q4 token parity. Mechanical-order claims require physical rotating media. |
| 2B · T2 | 250 materialized cartridges, four randomized swaps each. Require lower first-usable-inference latency than cold reload and zero post-enforcement revoked executions. Report unique payloads and clustered uncertainty. |
| 2C · T2 | 30 paired withdrawal traces per layout. Require fewer total device bytes written through quiescence versus matched-extent IVF-SQ8 at recall@10 within 0.01. Include tuned Qdrant. |
| 2D · T2 | 200 held-out source families. Require fewer stale admissions than DEV-selected C3, clean recall loss ≤0.01, and whole-appliance memory below 1 GiB. |
| 2E · T2 | 20 hash-selected repositories × 50 consecutive revisions. Require faster TTFT than stable-files-first prefix LRU and logit agreement with full prefill within a DEV-frozen tolerance. Pair by repository. |
| 3A · T1 | 1,000 disjoint paired migration queries against the frozen direct-fit reference. Require lower bound above −0.0500. Verify power from PM-2 paired variance before enrollment. Preserve the original miss. |
| 3B · T3 | 20 conversations, 400 questions. Estimate withdrawal-aware correctness against full replay, clustered by conversation. Any prefix disagreement retires incremental parity. |
| 3C · T3 | 200 held-out prompt pairs per base. Compare k=1 removal against norm-matched sham on the frozen target-logit endpoint. Failed planted control retires selector interpretation. |
| 3D · T3 | 300 independent benign families per shadow probe, paired against the frozen baseline alert. Retire the <2% false-alert characterization if either multiplicity-adjusted upper bound reaches 2%. |
| 4 · T2 | 24 held-out templates × four arrival schedules. Require positive paired bridge-by-governance interaction in correct on-time jobs/minute at pool four. Replay identical worker outputs and report live variance separately. |
| 5.1 · T4 | 1,000 partner queries versus its deployable CSLS adapter. Require quality equivalence within ±0.010 nDCG and the partner’s frozen served SLA. Failure retires equivalence. |
| 5.2 · T4 | 3,000 consecutive matters versus incumbent decisions and independent SQL. Require lower owner-priced loss against both. Fewer than 30 natural conflicts yields insufficient evidence. |
| 6A · T2 | 10,000 paired requests per backend/retraction cell. Require served p95 below 35 ms and recall within 0.01 of exact fallback. Publish p99, short responses, and timeouts. |
| 6B · T1 | Twelve cleared domains. Retire the joint claim on any failed Section 2 component. |
| 6C · T1 | 240 families. Apply Section 3’s SQL tie and superiority rules. |
| 6D · T1 | 24 cold starts. Apply the adoption and isolation gates below. |

## 5. 6D measures actual integration work

Cross three unrelated model families, four withheld vertical fixtures, and two independently reset sessions per cell. An unaffiliated operator controls allocation and hidden acceptance tests. In paired clean environments compare wheel plus public docs with a frozen public reference adapter. Counterbalance order using fresh contexts. Provide identical fixtures, dependency mirrors, and sixty-minute budgets. Capture elapsed time, caller code, configuration, errors, attempted private imports, and assistance. Score every assigned session, including installation failures.

Require all 24 SDK sessions to pass hidden functional tests without private imports or author help, plus a positive paired time-saving bound over the adapter, accounting for repeated fixtures. Narrow inference to these families and fixtures. Two concurrent tenants must show zero unauthorized reads across ledger, anchors, caches, and replay. Publish noisy-neighbour p95. Require reproducible wheel hashes, offline install, dependency SBOM, and license disposition. Independent customer adoption remains a separate T4 outcome.

## 6. Four-node execution DAG

| Phase | Locked placement and dependencies |
|---|---|
| 0 | Windows Desktop seals `#47`, reservations, exposure inventory, scorers, and A/A calibration. Sparks record resident load and allocatable resources. |
| 1 | Desktop builds 6D wheel and reference kit. Spark #1 runs 6A against staged shards and pinned databases. Spark #2 runs 2A. Spark #3 serializes 3C then 3D under GPU leases. |
| 2 | After 6A, Spark #1 serializes 2C → 6B → 6C. Spark #2 runs 2B → 2E. Spark #3 runs 2D → 3A → 3B. SDK consumers pin the frozen wheel. Desktop coordinates blinded 6D sessions and prepares both partner kits. |
| 3 | Desktop runs Item 1 software replay. Spark #3 reserves the worker pool for Item 4. Desktop seals 5.1 and 5.2 after dependency receipts arrive. Partner execution has its own clock. |

All five storage tracks use exclusive Linux timing leases. Preserve resident services, record interference, and reschedule contaminated measurements. The fleet establishes NVMe availability. Physical-spindle performance, twelve-domain clearance, SQL superiority, and partner outcomes remain open evidence fields under permanent reservations.
