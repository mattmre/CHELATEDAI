# Recon: CD-A2-01 closure requirements

Repo root is the cwd (branch lattice/phase2-continue-20260713 = origin/main @ 34ce4b56). Machine: RTX 3090, HF cache has all-MiniLM-L6-v2 / all-mpnet-base-v2 / bge-large / e5-base-v2; SciFact + NFCorpus + FiQA2018 + ArguAna datasets cached. Offline flags normally set; real-model tests gate on CHELATED_RUN_REAL_MODEL_TESTS=1. Deliver EXACT PowerShell commands, not prose.

CD-A2-01: the real swap-backend path (query_encoder_drift.QueryEncoderDrift._backend ->
embedding_backend.create_embedding_backend loading real all-mpnet-base-v2) is not exercised in default
CI; closure = "the PR-A4 real-model campaign actually running the all-mpnet-base-v2 swap path on a
networked/cached machine (CHELATED_RUN_REAL_MODEL_TESTS=1) and recording the artifact".
Read the gated tests (#271 made them opt-in) and the A4 campaign driver. Deliver: (1) the exact
command(s) that satisfy this closure (is the H2 campaign re-run sufficient by itself, or must the
gated unittest also run? name the test module), (2) what artifact must be recorded and where it should
live so the debt row can cite it, (3) the exact next-session.md row edit that closes it honestly.
