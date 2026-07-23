# Recon: H2 re-run commands (closes CD-H1-01)

Repo root is the cwd (branch lattice/phase2-continue-20260713 = origin/main @ 34ce4b56). Machine: RTX 3090, HF cache has all-MiniLM-L6-v2 / all-mpnet-base-v2 / bge-large / e5-base-v2; SciFact + NFCorpus + FiQA2018 + ArguAna datasets cached. Offline flags normally set; real-model tests gate on CHELATED_RUN_REAL_MODEL_TESTS=1. Deliver EXACT PowerShell commands, not prose.

CD-H1-01 says the committed swap-campaign result docs (docs/drift-recovery-swap-results-2026-06.md
+ docs/drift-recovery-swap-nfcorpus-results-2026-06.md) carry pre-H1-fix C3a numbers; closure = re-run
both campaigns (SciFact + NFCorpus) on post-#276 main and regenerate both auto-generated docs.
Read the campaign driver (PR #273 added it — likely run_swap_campaign.py or similar; find it), the H1
fix (#276, commit d5fc7dfa) to know what changed, and the doc-regeneration path (the docs say
auto-generated — find the generator). Deliver: (1) the exact command(s) to re-run BOTH campaigns with
the right seeds/conditions to reproduce the doc tables (C0/C2/C2O/C3a/C4a, 3 seeds, budget sweep if
included), (2) expected runtime + GPU footprint, (3) which files the run regenerates vs which need
manual number refresh (paper Section 5 mention), (4) gotchas (env vars, offline flags, output dirs
like experiment_runs/, whether the driver commits artifacts or writes gitignored dirs).
