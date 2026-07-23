# Fix task — apply the adversarial-review fixes to the Obj B preflight tests (and one production audit hole)

A reviewer found the new `research/drift_recovery/tests/test_sparse_local_preflight.py` overclaims in
two places, and found one production audit hole. Apply exactly these fixes:

## F1 (production, small): `_aggregate_cell` hardcodes `clean_ids_used_by_fit: False`
In `research/drift_recovery/d2b_preflight/preflight.py` (~L696), the sweep aggregation hardcodes the
flag instead of deriving it. Derive it as the OR over the per-run detector audits (same pattern as the
`run_preflight_cell` detector block at ~L616 which already derives from fit audits). A future leak must
flip the aggregate flag.

## F2 (test, behavioral leak-catcher): replace flag-reading with a routing-source check
Rename `test_fit_local_ridges_rejects_clean_ids` to what it actually asserts (audit contract), and ADD
a genuinely behavioral test: monkeypatch `infer_corrupted_clusters` to return a distinctive sentinel
labeling (e.g. a fixed permutation different from the clean IDs), run `run_preflight_cell`, and assert
the fitted local-ridge cluster keys match the SENTINEL labeling (not the clean IDs). This catches the
load-bearing leak the reviewer described: routing with `regime.clean_cluster_ids` while the flag stays
False would produce clean-ID cluster keys and FAIL this test.

## F3 (test, ladder margin): strengthen the sanity ladder
In `test_oracle_local_beats_global_on_residual`, construct the cell with a strong-enough warp
(higher gamma) that the ordering has a real margin, then assert: global residual >= 0.01 (not 1e-6) AND
local_ndcg - global_ndcg >= 0.01, AND the full ordering floor <= global < local <= oracle. Keep runtime
small. If the margins are not achievable on a small synthetic cell, use the largest stable margin and
say so in a comment — do not assert a margin the construction cannot guarantee deterministically.

## Then
1. Re-run the preflight sweep: `python -m research.drift_recovery.d2b_preflight.preflight` (offline env
   vars). Confirm the verdict is still CLOSE and the grid numbers are unchanged (F1 touches only audit
   metadata). If anything numeric changed, STOP and report — do not regenerate silently.
2. Run: `python -m unittest research.drift_recovery.tests.test_sparse_local_preflight research.drift_recovery.tests.test_d2_decisions research.drift_recovery.tests.test_synthetic_collapse -v`
   — paste the full pass/fail output.
3. Report each fix as file:line + what changed. Be brutally honest if F2's sentinel test reveals a real
   routing bug in production — report it, do not paper over it.
