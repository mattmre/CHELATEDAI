# W22-05 Preregistration: Same-Input Ordinary Bitemporal Ledger Comparator + Real-Graph Negation-Cycle Census

Status: **frozen before any W22-05 result was computed**. This file is committed on its own, ahead of the
comparator module, the runner, and every results file. Any later change to it shows up in `git log`.

- Queue item: `W22-05`, from `docs/research/wave22_queue_additions_2026-11-25.json` and the handoff packet
  `waves-handoff-wave22-breakthrough-audit-and-queue-additions-2026-11-25.md`, line 86.
- Compute class: CPU only. Zero LLM calls, zero embedding models.

## What I already knew before freezing (disclosed so it is not presented as a blind prediction)

- **Bundle #55.** The registry entry (`docs/research/registry.d/55-t6c-wild-revisions.json`) and
  `artifacts/pv7_w22e_b55_wild_live/pv7_w18_6c_wild_live_report.json` already report the following.
  - EK5_v3 and the SQLite challenger (closure_table) tie at a composite hit rate of 0.4375.
  - Family-cluster outcome: 240/240 ties.
  - So the #55 *aggregate* is not blind. What is new here is the per-trial set-level parity, the
    conflict-wins analysis, prevalence-weighted harm, and the negation-cycle census.
- **Arena 60.** The packet reports Chelated 58/60 vs Mem0 49/60. All 9 wins came from the 15 contested
  cases. Per-case arena rows are **not on disk**: `wave6_pm3_v2_arena_rows.csv` is absent, and the
  manifest holds aggregates only. The original arena extracted claims with an LLM; that output is not on
  disk either.
- **Conv 120.** Per-item correctness for EK5-on and EK5-off is on disk at
  `artifacts/wave6_pm1b_v4_reranked_sota/wave6_virgin_blind_judge_audit.json` (480 rows, 4 arms x 120).
  - EK5's per-item admitted sets were never persisted.
  - Recomputing them needs BGE-M3 plus a reranker. That is out of scope here: CPU only, no models.
- **The 153 explanation diffs.** These are in `formal/conformance_report.json`:
  - 6 steps: inc_vs_fresh = 153 = 49 reason + 104 pos_cyc.
  - No 12-step report (962 = 366 + 596) is on disk.
  - The Lean `ek5_ref_cli` is not built in this worktree.

## Inputs (same input to both arms)

| Corpus | Source | Reproduction gate |
|---|---|---|
| Arena 60 | `prospective_realworld_drift_corpus.build_prospective_realworld_drift_suite`, rebuilt into `scratch/`. The OSV cache is fetched live, because `cache/osv_cve_seeds.json` is absent. | corpus sha256 must equal `cb6bc0082b5850f4edfea668d1ecb2a73fe6f898b7476e34ed11036cb378765b`. A mismatch is reported, and the arena verdict is then labelled `corpus_hash_mismatch` (OSV summary text only touches the 5 CVE prelim docs). |
| #55 wild revisions | `artifacts/pv7_w22e_b55_wild_live/frozen_snapshots/` (4 strata, 240 families, 960 cuts) | Re-executing with the bundle's own projection and scoring functions must reproduce accuracy 0.4375 for both EK5_v3 and SQLite closure_table. |
| Conv 120 | `virgin_conversational_holdout.build_virgin_conversational_holdout()` | digest must equal `d395dd94…95e2`, which matches the manifest. |

### Matched extraction (Arena 60)

The original LLM extractions are not on disk, so both arms receive the **same deterministic role-derived
claims**.
- **entity:** the case subject.
- **slot:** one canonical slot per case.
- **value:** taken from the case fields that the doc-id role points to.
- **stance:** follows the extraction prompt V2 rules.
  - `prelim`, `echo`, `late_archive`, `seed_rumor`, `late_launder`, `v1`, `v1_echo` → ASSERT
  - `official_supersede`, `v2`, `v3_final` → SUPERSEDE
  - `official_debunk` → DEBUNK_RUMOR
  - `party_a`, `party_b` → REFUTE

This is oracle extraction. It removes extraction noise from **both** arms, so Arena-60 numbers here are a
clean-graph comparison and are not a reproduction of the LLM-extracted 58/60.

- **EK5 arm.** The claims go through `chelated_epistemic_proposer_v2.AuthorityDominatedExocortexV2`, using
  deterministic one-hot document and slot vectors.
  - The verdict is `ABSTAIN` if the CONTESTED rule fires (`slot_contested and max_cont_auth > max_adm_auth`).
  - Otherwise the verdict is the frozenset of normalized claim values of admitted docs on the case slot.
- **Ordinary ledger arm.** This is `chelated_pv7_spine.challenger_ledger` on SQLite, which shares no EK5
  code (enforced by AST).
  - Each claim becomes an `ASSERT` with authority = channel tier, `valid_from` = published_ts,
    `tx_time` = ingested_ts. REFUTE claims become a `CONFLICT` revision instead.
  - SUPERSEDE and DEBUNK_RUMOR become `WITHDRAW` ops, issued at the actor's tx and authority, against
    every earlier-ingested record in the slot with a different value.
  - Each echo of a value gets `DERIVES(echo → first-ingested doc with that value)`.
  - The query is as-of valid=10^6 and tx=max ingested.
  - Verdict: `RESOLVED` → {winner value}; `DISPUTED` → `ABSTAIN`; `RETRACTED`/`ABSENT` → empty set.
  - Both modes are run, `recursive_cte` and `closure_table`, and they must agree.
- **Ablation (reported, not part of the decision).** An authority-only ledger with no WITHDRAW, no
  CONFLICT and no DERIVES. It shows how much of the arena pure authority ranking decides.
- **Correct** means the verdict set equals {gold}. Gold ABSTAIN means the verdict must be ABSTAIN.

### #55

The bundle's own `project_ek5` / `build_ek5_store` / `score_ek5` and `sql_envelopes` / `build_sql_ledger` /
`score_sql` are used unchanged. Both arms get the bundle's identical public inputs, with `echo_of` stripped.
- **Set-level verdict per cut** = (disposition, lineage-root set), which is the bundle's own hit definition.
- **Disposition-only parity** is reported as a secondary measure.

### Conv 120 (not a matched-input comparison; declared in advance)

- **Coverage.** Only the 24 `knowledge_update_echo_flood` questions are ledger-shaped (one entity/slot
  whose value changes). The other 96 (needle, synthesis, attribution, abstention) are not.
- **Ledger arm.** It runs on the 24 KU items only, with role-derived claims:
  - `ku_initial` and `ku_echo_*` turns → ASSERT(stale value)
  - `ku_update` → SUPERSEDE, i.e. WITHDRAW the stale records and ASSERT the gold value
  - every record has the same authority (same speaker)
- **EK5 per-item results** come from the audit log: EK5-on = `Chelated_v4_Reranked_EK5v3`,
  EK5-off = `DualGran_Reranked_NoEK5`.
- **Set-level parity is NOT computable for Conv 120** (EK5 sets were not persisted). Conv 120 is excluded
  from the kill rule and reported descriptively only.

## Primary metrics

Each metric is computed per matched corpus (Arena 60, #55) and pooled over both.

1. **Set-level verdict parity** `P` = share of items where the ledger's set-level verdict equals EK5's.
2. **Conflict set** = items whose gold verdict is a conflict outcome: Arena gold `ABSTAIN` (15 cases), and
   #55 oracle disposition `disputed`. Everything else is determinate.
   - **EK5 conflict wins** `W` = conflict items where EK5 is correct and the ledger is wrong.
   - **Ledger conflict wins** = the reverse.
3. **Prevalence-weighted harm** for each arm, at p ∈ {0.01, 0.05, 0.25}:
   `harm(p) = p·err_conflict + (1−p)·err_determinate`
   where err = 1 − correct-rate on that subset. A subset with n = 0 is reported as undefined, not as 0.
4. **Coverage** = share of items where the arm returns a non-empty verdict (a value set or ABSTAIN).

## Kill / advance rule (frozen)

**RETIRE the EK5 clean-graph reasoning premium** only if all three conditions hold, on the pooled matched
set **and** on each matched corpus that passed its reproduction gate:

- (a) `P ≥ 0.99`;
- (b) `W = 0`, meaning the ledger matches every conflict item EK5 gets right;
- (c) equivalent harm and coverage: at each p ∈ {0.01, 0.05, 0.25}, `harm_ledger(p) ≤ harm_EK5(p) + 0.005`,
  and `coverage_ledger ≥ coverage_EK5 − 0.005`.

If those hold, F2 is narrowed to formal/audit proof (T-Rs) plus missing-edge/hidden-copy regimes (T-D).

**Otherwise ADVANCE (EK5 keeps a measured premium).** In that case the report lists every item and
stratum where EK5 beats the ledger, and every one where the ledger beats EK5.

If a corpus fails its reproduction gate, it is reported but excluded from the decision, and the verdict
carries the suffix `_PARTIAL_INPUTS`.

## Secondary measurements (reported, not part of the decision)

- **153–962 explanation diffs.** A Python-only re-run of the `/formal` conformance generator's
  inc_vs_fresh comparison at steps 6 and 12 (1000 sequences, seed 20261124). It uses
  `run_sequence` + `compare_states` and does not use the Lean CLI.
  - It reproduces the reason and pos_cyc diff counts. The expectation is 153 = 49 + 104 at 6 steps; at
    12 steps, 962 is what the packet cites.
  - **Ledger counterpart.** For every queried cut in Arena 60 and #55, the ledger is built
    incrementally, then rebuilt from scratch from the same op log. The count is the number of
    `AsOfQueryResult.digest()` mismatches.
  - Caveat: the ledger's explanation surface (status, winner, contending, live, retracted ids) is thinner
    than EK5's cause-class reasons. A zero count here does not mean it gives equivalent explanations.
- **Real-graph negation-cycle census.** Run on the EK5 attestation graphs actually built from Arena 60
  and from #55.
  - Count the SCCs of size > 1 (plus self-loops) that contain at least one negative edge (REFUTE,
    SUPERSEDE, VERSION_LINEAGE_SUPERSEDE).
  - Report the incidence per case/family, the number of mutual-REFUTE 2-cycles, and the store's own
    `pos_cyc` node count.
  - Report both raw incidence and incidence weighted by conflict prevalence (incidence among conflict
    items × p, plus incidence among determinate items × (1−p)).
  - The Conv 120 graph is not available (it needs the models); that is disclosed.
- **PostgreSQL.** The task asks for PostgreSQL 17. The host has PostgreSQL **18.6** (scoop), not 17. This
  run is SQLite-only, and Postgres is disclosed as not used.
