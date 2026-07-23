# Implement the missing Obj B preflight test file (test debt)

The build of research/drift_recovery/d2b_preflight/ skipped its test deliverable. Write
`research/drift_recovery/tests/test_sparse_local_preflight.py` (plain unittest, no pytest; offline;
CPU-only; keep runtime under ~60s by using small N/d/K). Import from
research.drift_recovery.regimes.sparse_local_nonaffine and research.drift_recovery.d2b_preflight.preflight.
Read those modules first to match real APIs. Required tests:
1. Warp properties: harmed-cluster fraction matches s (sparse); displacement is zero on clean clusters;
   the within-cluster map is NON-AFFINE (fit the best least-squares affine map on a harmed cluster's
   clean->corrupted pairs and assert residual > tolerance) and NON-RADIAL (displacement not parallel to
   the radial direction from the cluster centroid) for BOTH families (quadratic, soft_fold).
2. No clean-ID leakage: fit_local_ridges training_audit has clean_cluster_ids_accepted == False, and
   run_preflight_cell's detector block reports clean_ids_used_by_fit == False and that value is derived
   from the fit audits (check it flips if you monkeypatch an audit to True — behavioral, not just a flag read).
3. Sanity ladder: on a synthetic cell, the best GLOBAL ridge leaves a residual while a dense local fit
   using oracle clean cluster IDs recovers strictly more (diagnostic upper bound ordering).
4. Hyperparameter hygiene: lambda/rank selection uses only anchor train/dev rows (selection_source ==
   'anchor_dev_mse_only'; eval indices disjoint from fit+dev indices).
5. New audit fields present: training_audit contains selected_lambda, selected_rank (None allowed for
   dense), selected_lambdas_per_cluster.
Then RUN: python -m unittest research.drift_recovery.tests.test_sparse_local_preflight -v (with
HF_HUB_OFFLINE=1) and paste the full pass/fail output. If any test fails, fix the TEST unless it reveals
a real bug in preflight.py — in that case report the bug with file:line and do NOT paper over it.
