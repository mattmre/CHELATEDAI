# W22-05: EK5 vs a plain record ledger, compared on clean graphs

Date: 2026-11-26. This was a CPU-only run, SQLite only, and took 39.3 s. The preregistration was committed as
`cfd40d6d` before any result existed. The results are in `w22_05_results.json`.

## Verdict: `RETIRE_EK5_CLEAN_GRAPH_PREMIUM` (PROPOSED)

All three preregistered checks hold on `arena60`, on `b55`, and on the pooled set. On clean graphs, EK5 adds nothing over a
plain ledger:

- **(a)** set-level parity is at least 0.99;
- **(b)** EK5 has zero conflict-item wins;
- **(c)** coverage is equivalent, and harm is equivalent at p = 0.01, 0.05 and 0.25.

**How to read the verdict.** No clean-graph premium for EK5 was *detected*, but the inputs could hardly have shown
one. arena60 has no power to separate the arms. On b55, EK5's graphs contain no copy edges and no cycles, so its
distinctive machinery never runs (limits 1, 6 and 7 below). So this is the preregistered outcome. It is **not**
positive evidence that a plain ledger equals EK5 on graphs where copying or cycles matter. The status is PROPOSED;
round-1 Tier B scored it 80 (important).

## Parity

| Corpus | n | EK5 correct | Ledger correct | Set parity | EK5 wins | Ledger wins |
|---|---|---|---|---|---|---|
| arena60 | 60 | 60 | 60 | 1.000 | 0 | 0 |
| b55 | 960 | 420 | 420 | 1.000 | 0 | 0 |
| pooled | 1020 | 480 | 480 | 1.000 | 0 | 0 |

Pooled harm is the same for both arms: 0.6041 at p = 0.01, 0.5797 at p = 0.05 and 0.4576 at p = 0.25. On b55, the error
rate on determinate items is 0.643 for both arms. Both arms share the bundle's public projection, which strips
`echo_of` (`scripts/run_pv7_w18_6c_wild_revisions.py:100-106`). So the shared failure could come from that projection
as much as from the corpus. This run does not separate the two causes.

## Limits on what this shows (read before citing the verdict)

1. **arena60 has no power to separate the arms.** The authority-only ledger ablation also scores 60/60, in every
   regime (15/15 × 4). So the arena cannot tell EK5 from a much weaker baseline. What arena60 shows is only that the
   two arms do not differ. It is not evidence that the ledger matches EK5 on hard cases. All of the discriminating
   weight sits on b55, where both arms score 420/960.
2. **The 962 figure was not reproduced.** The cited 962 conformance diffs (incremental vs a fresh rebuild, 12 steps)
   did not reproduce. A Python recomputation with the same seed (20261124) over 1000 sequences gives:
   - **153** diffs at 6 steps (`pos_cyc` 104, `reason` 49);
   - **825** diffs at 12 steps (`pos_cyc` 548, `reason` 277).

   In both cases the semantic total is 0. The formal 12-step report and the Lean `ek5_ref_cli` are not present, so the
   gap from 825 to 962 is unexplained.
3. **Rebuilding the ledger depends on insertion order.**
   - On b55, rebuilding from a shuffled insertion order changes `winner_record_id` on 81 of 960 queries. The cause is
     tie-breaking by insertion order.
   - Rebuilding in the original order gives 0 mismatches.
   - arena60 has 0 mismatches either way.
   - The ledger's dispositions are stable, so parity is unaffected, but which record it names as the winner is not
     canonical.
4. **"Clean graph" holds only for b55.**
   - arena60: 30 of 60 graphs contain a negation cycle, made up of 210 mutual-REFUTE 2-cycles. This covers all 15
     conflict graphs and 15 of the 45 determinate graphs.
   - b55: 0 of 240 graphs contain one.
   - EK5 still ties the ledger on arena60, so these cycles did not give EK5 an advantage on this corpus.
5. **Conv120 is descriptive only.**
   - On the 24 knowledge-update items, the scores were:

     | Arm | Correct (of 24) |
     |---|---|
     | Ledger | 24 |
     | Authority-only ledger | 0 |
     | EK5-on | 23 (misses `vconv_10`) |
     | EK5-off | 24 |

   - EK5's admitted sets were never persisted, so no matched set-level comparison is possible.
   - The audit digest matches the expected value, `d395dd94…`.
6. **On b55, EK5's distinctive machinery never runs.**
   - EK5's b55 graphs have 360 edges, and every one is a negative supersede edge.
   - There are 0 COPY or DERIVES edges and 0 cycles.
   - So echo/copy collapse, refutation-cycle handling and `pos_cyc` never fire. On inputs where the
     highest-authority live record wins, the two engines are close to bound to agree.
7. **Condition (b) has almost no power on b55.**
   - All 120 b55 conflict items sit in one stratum, `equal_authority_vex_ghsa`, where both arms score 240/240.
   - The other three strata fail the same way for both arms, 180/240 each.
   - So "zero EK5 conflict wins" rests on items that both arms get trivially right.
8. **The b55 tie was known before the preregistration.** Bundle #55 had already reported EK5 = SQL at 0.4375, with
   240/240 family ties (`PREREGISTRATION.md:12-16`). The verdict rests on b55, so it largely re-runs a known result.
9. **The arena ledger's inputs come from EK5's pipeline.** `arena_ledger_ops` (`w22_05_ledger_comparator.py:154-229`)
   reads the events and cleaned claims that EK5's grounding step produced. Its WITHDRAW-on-SUPERSEDE/DEBUNK rule was
   written by hand. The ledger *engine* shares no EK5 code, but its inputs are not independent of EK5.
10. **There was an earlier run that was not recorded.** The OSV cache already existed when the committed run started
    (`existed_before_run: true`; it was created at 02:49:50, and the committed run started at 02:52:20). The
    implementation was committed after the results. So a code change made after seeing results cannot be ruled out.
    The b55 path uses the bundle's code unchanged, which limits how much this could matter.

## Inputs that were not available

- `wave6_pm3_v2_arena_rows.csv` (per-case arena rows). Because the arena LLM claim extractions are missing too, both arms
  used a role-derived oracle extraction.
- The per-item admitted sets for Conv120 EK5.
- The formal 12-step conformance report (the 962 figure).
- The Lean `ek5_ref_cli`, which has not been built.
- PostgreSQL 17. The host has 18.6, so the PostgreSQL arm was not run. Both SQLite modes (recursive CTE and closure
  table) agree with EK5 v3 on b55 at 0.4375, with 0 disagreements.

## Brutal Honesty (self-score 85)

- **L4 — partial work presented as complete:**
  - This report was written after the run, not by the runner.
  - The 962 figure was not reproduced.
  - Two choices were made after preregistration:
    - the timestamp used for Conv120 ordering;
    - mapping "retracted" to "not covered".
- **L5 — production path untested:** the runner functions in `scripts/run_pv7_w23a_w22_05_ledger_comparator.py` have no
  tests. The 18 tests cover only `w22_05_ledger_comparator.py`.
- **L11 — broad catch:** the OSV seed fetch uses a bare `except`. The run used a cache that already existed
  (`scratch/osv_cve_seeds.json`, sha256 `5cc7450c…`).
- **L13 — weaker than it reads:** arena60 cannot separate the arms (limit 1 above). The RETIRE verdict rests on b55
  parity together with the absence of any EK5 win.
- **Scratch:** `artifacts/pv7_w23a_w22_05/scratch/` is gitignored and can be removed after review. It holds:
  - `osv_cve_seeds.json`
  - `arena_rows.json`
  - `b55_items.json`
  - `conv_ku_rows.json`
  - `full_run_stdout.txt`
  - the Tier B rerun folder, `tierb_rerun/`

  On this host that folder is
  `C:\Users\mattm\AppData\Local\Temp\claude\439e695c92b40043\c9f4318c-455f-427a-844a-07562a0a3c42\scratchpad\w22-05\artifacts\pv7_w23a_w22_05\scratch\`
  (1.2 MB, including the rerun).
- **The runner can overwrite the committed results.** Its output directory is hard-coded
  (`scripts/run_pv7_w23a_w22_05_ledger_comparator.py:34-35`), so any re-run overwrites `w22_05_results.json`.
- **DEFERRED_SCOPE (queue items not delivered):**
  - the PostgreSQL 17 arm;
  - a matched Conv120 comparison, which needs EK5's admitted sets to be persisted;
  - the gap between 825 and 962 diffs, which needs the formal report or the Lean CLI.
