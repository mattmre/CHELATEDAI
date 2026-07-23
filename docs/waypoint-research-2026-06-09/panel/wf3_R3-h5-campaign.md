# Recon: H5/H4 verdict campaign commands

Repo root is the cwd (branch lattice/phase2-continue-20260713 = origin/main @ 34ce4b56). Machine: RTX 3090, HF cache has all-MiniLM-L6-v2 / all-mpnet-base-v2 / bge-large / e5-base-v2; SciFact + NFCorpus + FiQA2018 + ArguAna datasets cached. Offline flags normally set; real-model tests gate on CHELATED_RUN_REAL_MODEL_TESTS=1. Deliver EXACT PowerShell commands, not prose.

PRs: #279 H5a steering-post bank + prune/re-anneal lifecycle; #281-283 H5b/S2 post-bank-from-anchors
builder + runtime glue + conditions C5/C5s/C5r; #290 H5 post-bank head-to-head campaign driver
(C5/C5s/C5r verdict); #291 H4 compound_cycles knob. No results(h5) commit exists — the verdict
campaign has not been run. Read the #290 driver. Deliver: (1) the exact command(s) to run the full
head-to-head (which conditions, seeds, datasets it covers; does it include H4 compound_cycles arms),
(2) expected runtime/GPU, (3) what verdict artifact/doc it writes and whether a results doc generator
exists, (4) what the preregistered decision rule is (what makes C5/C5s/C5r a win/loss vs C3a/C4a),
(5) gotchas. If the driver expects artifacts from the H2 re-run (fresh baselines), say so explicitly —
that decides run ordering.