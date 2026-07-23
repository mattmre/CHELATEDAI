# C1 adversarial-review fix report

Generated from the frozen pack and regenerated D1 artifacts; result numbers are not hand-entered.

| Fix | Exact evidence location | Result |
|---|---|---|
| F1 | `research/drift_recovery/d1/paper_stats.py:171`<br>`research/drift_recovery/out/d1/ci_report.md:26`<br>`research/drift_recovery/out/d1/paper_edits.md:19` | Literal fit: 28/62 eval-positive docs; safe fit: 0. ridge 78.9% safe / +5.4 pp, closed-form orthogonal Procrustes 73.9% safe / +8.4 pp, residual MLP 67.5% safe / +14.1 pp. |
| F2 | `research/drift_recovery/out/d1/paper_edits.md:31` | Ban plus forbidden/permitted lists emitted; ridge−MLP CI [-0.0333, 0.0773]. |
| F3 | `research/drift_recovery/out/d1/paper_edits.md:33` | Chair-only application guard is explicit; protected paper-draft files modified: 0. |
| F4 | `research/drift_recovery/d1/paper_stats.py:751`<br>`research/drift_recovery/out/d1/section3_draft.md:19` | Closed-form SVD ladder map is explicitly distinct from the trainable Cayley adapter. |
| F5 | `research/drift_recovery/d1/paper_stats.py:252`<br>`research/drift_recovery/out/d1/ladder.json:5` | Old cross-seed C3a scalar 0.1609 is superseded by seed-42 per-query 0.1570; recovery difference 0.0056 exceeds 0.001. |
| F6 | `research/drift_recovery/d1/paper_stats.py:500`<br>`research/drift_recovery/out/d1/contrasts_holm.json:46`<br>`research/drift_recovery/out/d1/ci_report.md:22` | Primary ridge−MLP half-width 0.0553; secondary family median 0.0887; rough scale 817 queries. |
| F7 | `research/drift_recovery/tests/test_d1_stats.py:31`<br>`research/drift_recovery/tests/test_d1_stats.py:52`<br>`research/drift_recovery/tests/test_d1_stats.py:117`<br>`research/drift_recovery/stats/paired_bootstrap.py:117` | Shared-draw audit exposed; 1% invalid allowed and >1% rejected; leakage assertions bind 28/62 and safe=0; max harness-parity delta 0.0e+00. |

Protocol-honesty note: C1's supplied F2 sentence called 0.089 the `median primary` half-width, while F6 requires ridge−MLP alone to be primary. The generated chair block therefore reports primary=0.055 and labels 0.089 as secondary.
