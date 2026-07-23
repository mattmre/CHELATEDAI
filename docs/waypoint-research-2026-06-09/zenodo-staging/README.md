# Zenodo staging — drift-recovery methodology framework + three evaluation hazards

**STATUS: STAGED, NOT UPLOADED.** This directory is the operator's pre-upload bundle
for a versioned Zenodo DOI that timestamps the two *proven* methodology findings while
the full Liquified-Lattice program is completed. Nothing here is uploaded automatically.
Operator: review, confirm the license + author identity, zip the files listed in
`FILES.md` from a clean checkout, then upload via the Zenodo web UI or API.

This is **priority insurance**, not a paper. It banks the date on the two contributions
that are already runtime-verified, so finishing the program first costs no priority if a
competing paper lands mid-program. The full paper (two datasets, partial-recovery result)
stays a local draft until the program is whole.

## What this artifact is

An open, reproducible framework for evaluating closed-loop embedding-drift recovery, plus
three evaluation-design hazards it surfaces. All are backed by command-output evidence from
the production code path (not tests, not prose) — including the strongest, Finding 3, on
real full-scale 3-seed runs.

### Finding 1 — the C2-oracle hazard
The "obvious" maintenance baseline — re-embedding affected documents with the *original
frozen encoder* — is a **degenerate oracle** for synthetic geometric drift: it inverts the
perturbation by construction and returns to the exact pre-drift state, so it cannot serve
as a fair baseline. Only an encoder/query *upgrade* arena defeats it; document-side
model-swap does not. Verified: C2 == C0 at full float precision on both SciFact and
NFCorpus (max|baseline − final| = 0).

### Finding 2 — the supervision-signal mechanism
An *unsupervised* homeostatic correction loop has no signal pointing back toward the
pre-drift geometry, so its loss-optimal correction is near-identity ("do almost nothing").
Recovery **requires** an explicit pre-drift supervision channel (held-out anchor
query–doc pairs). Verified: unbounded unsupervised correction norm ≈ 1.9e-4; in the
synthetic-drift regime the unsupervised loop never engaged (correction_applied = 0/36,
store byte-identical).

### Finding 3 — the trivial-baseline hazard (2026-06-29, strongest; real full-scale numbers)
The whole elaborate corrector apparatus (bounded adapter, "living/annealed post-bank", MoE
router) is **beaten ~4.5× by a single `np.linalg.lstsq` call** in the exact same arena. Real,
3-seed, both datasets: our corrector recovers ~16% (SciFact) / ~30% (NFCorpus) of the oracle
gap; a regularized linear map (ridge / orthogonal Procrustes) recovers **~85% / ~80%**; a
residual MLP does not beat it (~85% is the post-hoc ceiling — the last ~15% is the oracle's
re-embed-raw-text advantage, unreachable from cached vectors). A bound→recovery α-ablation +
the unbounded corrector number (C4a) decompose the deficit honestly: **primarily a weak
supervision signal** (relevance-anchor InfoNCE vs paired-embedding least-squares: ~+60pts),
**secondarily the near-identity bound** (~−9pts; though a tight bound alone wrecks even a good
map). Lesson: *always benchmark the trivial regularized linear map before claiming a drift
corrector works* — the frozen-C0 straw man hides this. (Also: the H5 "living bank beats static"
thesis is falsified — C5 ≡ C5s to 15 s.f., real data; the prune/re-anneal lifecycle is a no-op.)

### Worked example (honest scope — CORRECTED 2026-06-29)
With the oracle defeated (encoder-upgrade arena) and explicit InfoNCE supervision, the
detection-triggered bounded adapter recovers only a *small* fraction (~16–30% of the oracle
gap) — and per Finding 3 it is **far below the achievable post-hoc ceiling (~85%)** reached by a
trivial regularized linear map. So the honest result is NOT "partial recovery is the ceiling" but
"our bounded corrector underperforms the trivial baseline; the ceiling is ~85% (linear), and the
remaining gap to oracle is irreducible from cached vectors." Real numbers: `review-notes-cleanup-pass.md`
(2026-06-29 entries) + `scratchpad/rank{1_inarena,1_sweep,1_mlp,4_direction}.py`.

## Reproduce

```
pip install -r requirements.txt
# offline model cache assumed; set HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -m unittest test_drift_injector test_drift_recovery_metrics \
    test_run_drift_recovery_experiment
# real-data swap campaign (GPU or CPU):
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python run_drift_recovery_swap_campaign.py
```

## Operator checklist before upload
- [ ] Confirm license (see `.zenodo.json` — placeholder `TBD` must be set).
- [ ] Confirm author name / ORCID / affiliation in `.zenodo.json`.
- [ ] Pull final §5 numbers from regenerated `docs/drift-recovery-swap*-results-2026-06.md`.
- [ ] Zip the files in `FILES.md` from a clean checkout at the chosen tag.
- [ ] This bundle is informational — the canonical code lives in the public repo; the
      Zenodo record should reference the repo + commit/tag for provenance.
