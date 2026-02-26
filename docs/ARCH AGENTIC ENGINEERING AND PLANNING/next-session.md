# Next Session Checklist

Purpose: Minimal context to resume the workflow in short sessions.

## Session Start
- Review `docs/research-2026-02-26-sweep-analysis.md` for the completed 81-config sweep analysis.
- The 81-config sweep (`sweep_results.json`) analysis is COMPLETE. See findings below.
- The 7,350-config large sweep (`large_sweep_results.json`) was never completed -- no output file exists. Decide whether to re-run or skip.
- Check tracker date and carryover items.

## Sweep Analysis Results (Completed 2026-02-26)
- **Zero configurations improved over baseline.** All 81 configs degraded SciFact NDCG@10.
- **Best config:** LR=0.01, threshold=1, noise=0.2, epochs=5 (-0.11% degradation).
- **63% of configs collapsed** to degenerate state (LR>=0.1 all catastrophically fail).
- **Noise injection at 0.05 is beneficial** -- best in 6/9 comparisons at LR=0.01.
- **Threshold=1 is strictly optimal** -- higher thresholds cause severe degradation.
- **Fewer epochs (5) are better** -- sedimentation objective diverges from retrieval quality.
- **Recommended safe preset:** LR=0.01, threshold=1, noise=0.05, epochs=5.

## Session Objectives
- Primary goal: Integrate sweep findings into `ChelationConfig` presets (update sedimentation presets, enable noise by default at 0.05, reduce default epochs to 5).
- Secondary goal: Investigate why sedimentation hurts retrieval -- the training objective is misaligned with NDCG@10. Consider loss function changes or retrieval-aware early stopping.
- Tertiary goal: Run finer-grained sweep in the 0.001-0.01 LR range with sub-integer thresholds [0.5, 1.0, 1.5] to find true optimum.
- Quaternary goal: Continue driving review/merge progression across the open stacked PR chain.

## Cycle ID
- AEP-2026-02-26 (Sweep Analysis & Config Integration)

## Hand-off Notes
- Session 19 completed noise injection regularization dynamically scaled by chelation event complexity.
- Session 20 completed infrastructure and research enhancements (committed on `feat/session20-infra-research-enhancements`).
- The `run_sweep.py` 81-config grid sweep completed successfully on 2026-02-23 (45 min runtime).
- The `run_large_sweep.py` 7,350-config sweep never produced output -- likely was interrupted or never started successfully.
- Comprehensive sweep analysis written to `docs/research-2026-02-26-sweep-analysis.md` with full tables, cross-parameter interactions, and recommended presets.
- Key config changes needed: "balanced" sedimentation preset uses threshold=3 which is dangerously aggressive per sweep data. Should be revised to threshold=1.
- Pytest test suite stable (529 passing on main per MEMORY.md).
