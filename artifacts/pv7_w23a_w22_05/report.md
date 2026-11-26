# W22-05: EK5 vs a plain record ledger, compared on clean graphs

Date: 2026-11-26. This was a CPU-only run, SQLite only, and took 39.3 s. The preregistration was committed as
`cfd40d6d` before any result existed. The results are in `w22_05_results.json`.

## Verdict: `RETIRE_EK5_CLEAN_GRAPH_PREMIUM` (PROPOSED)

All three preregistered checks hold on `arena60`, on `b55`, and on the pooled set. On clean graphs, EK5 adds nothing over a
plain ledger:

- **(a)** set-level parity is at least 0.99;
- **(b)** EK5 has zero conflict-item wins;
- **(c)** coverage is equivalent, and harm is equivalent at p = 0.01, 0.05 and 0.25.

The verdict is **PROPOSED** until a fresh adversarial Tier B review has run.

## Parity

| Corpus | n | EK5 correct | Ledger correct | Set parity | EK5 wins | Ledger wins |
|---|---|---|---|---|---|---|
| arena60 | 60 | 60 | 60 | 1.000 | 0 | 0 |
| b55 | 960 | 420 | 420 | 1.000 | 0 | 0 |
| pooled | 1020 | 480 | 480 | 1.000 | 0 | 0 |

Pooled harm is the same for both arms: 0.6041 at p = 0.01, 0.5797 at p = 0.05 and 0.4576 at p = 0.25. On b55, the error
rate on determinate items is 0.643 for both arms. That reflects the corpus, not a difference between the arms.

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
- **Scratch:** `artifacts/pv7_w23a_w22_05/scratch/` holds only the OSV cache and can be removed.
