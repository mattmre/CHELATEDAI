# Experiment scripts — reproducibility index (2026-06-30)

These scripts were run from the temp scratchpad against the **`gpu-campaigns` worktree** (detached at
`origin/main`, commit 34ce4b56 / PR #291). They import the merged harness (`run_drift_recovery_*`,
`query_encoder_drift`, `run_road_course_campaign`). To reproduce, place a worktree/checkout of
`origin/main` at `D:\GITHUB\CHELATEDAI\.claude\worktrees\gpu-campaigns` (or edit the `WT` path in each
script) and run with `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` (cached MiniLM / mpnet / bge-large / Qwen2.5-0.5B).

## Which script produces which paper number (canonical = SciFact **eval-split**, binary NDCG)

| script | produces | headline |
|---|---|---|
| `ladder_evalsplit.py` | the §5.5 recovery ladder (Fig 1) | C3a 20%, lstsq 67%, **ridge 84.4%**, Procrustes 82.6%, MLP 81.2%, low-rank 49%; α-curve {0.9,2.1,11.8,58.7,84.4} → **4.2×** |
| `rank1_generality.py` | §5.5 generality (Fig 3) | eval-split: mpnet SciFact 84.4/2.1, NFCorpus 66.4/2.2; bge SciFact 78.9/0.0, NFCorpus 74.3/2.1 |
| `rank1_airtight.py` | the airtight matched-eval-split corrector-vs-fair (SciFact 4.2×) | |
| `s1_doc2query.py` | §5.6/§7 S1 clean test (true Qwen doc2query) | S1 **81%** on SciFact (all-qrels), but **dominated** by the oracle-pair (85.5%) |
| `make_figures.py` | fig1/2/3 PNGs (→ `../figures/`) | reads the canonical numbers above |
| `rank1_sweep.py`, `rank1_inarena.py`, `rank1_mlp.py` | the earlier all-qrels fair-baseline sweep (superseded by the eval-split ladder for consistency) | |
| `rank4_direction.py` | old→new vs new→old direction check (Rank 4) | |
| `fullscale_campaigns.py`, `fullscale_v2.py` | the full GPU campaign drivers (H5/H3/H2, 3 seeds) → `../campaign-evidence/` manifests | |
| `s1_hyde.py` | the first-K-words S1 proxy (68–84%, blurs into partial re-embed — superseded by `s1_doc2query.py`) | |
| `h2_rerun*.py`, `h1_confirm.py`, `h5_smoke.py`, `extract.py` | intermediate diagnostics | |

Canonical numbers and their provenance are also in `../review-notes-cleanup-pass.md`
(the 2026-06-30 "4-agent review panel" + "S1 clean form" + "citation verification" entries).
