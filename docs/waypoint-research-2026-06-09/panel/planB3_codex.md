# Codex build task — lean D2: the one honest crossover kill-screen (P2, locked plan)

Build the lean crossover kill-screen in `research/drift_recovery/d2/` + `research/drift_recovery/
methods/` + `research/drift_recovery/regimes/`, on top of `harness_bridge.py`, `contracts.py`,
`artifacts.py`, and `stats/`. This is P2 — the last honest test of whether *any* detector-gated
bounded chelation-style corrector beats a trivial baseline in the regime chelation was designed for
(local, near-identity, cluster-scoped corrections). It is a KILL-SCREEN: the expected and acceptable
outcome is a clean negative. Do NOT engineer a win.

Work ONLY inside `research/drift_recovery/`. Reuse the single harness import boundary. Keep parity
≤1e-12. Offline (HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1), sequential, modest GPU (shared 3090).

## Regime-C (this is NOT the encoder-upgrade regime — it is the chelation home turf)
`regimes/synthetic_collapse.py`: inject **semantic collapse** by pulling a fraction of clusters
toward their centroid. Corrupt **8 and 16 clusters**, collapse strength **β=0.10** only. Seeds
{42,1337,7,101,202} (5 seeds — needed for the same-sign kill rule). SciFact + NFCorpus. This is a
LOCAL, bounded, near-identity-friendly corruption — exactly the regime the bounded/annealed chelation
premise claims to own. If chelation cannot win here, it cannot win anywhere.

## Methods (≤4 PRIMARY, param-matched)
`methods/`:
- `affine.py`: **global ridge** (the trivial baseline that must be beaten).
- `local_adapters.py`: **global + low-rank-local, soft-routed, parameter-matched** to the chelation
  adapter — this is the "local" diagnostic, NOT a chelation win (it is privileged/paired).
- `isotropy.py`: **CBIE** (cluster-based isotropy enhancement) + ZCA + all-but-top as baselines.
- `hubness.py`: one **hubness score-scaling** method (operates on SCORES via `ScoreTransform.fit/rank`,
  not on vectors — respect the two-interface split from the plan).
- `bounded_chelation.py`: the bounded adapter **+ detector gating**. Two arms:
  (a) **α=0.05** near-identity bound — reported as a SEPARATE design-premise stress test, NOT the
      primary kill arm.
  (b) **primary kill arm** — a detector-gated repair that can actually move mass (unbounded, OR α
      calibrated so mean ‖x'−x‖ under β=0.10 is feasible). The G3 kill is judged on THIS arm.

## Detector = an explicit cluster-level HARM scorer (Grok FP-A / FN-B — load-bearing)
`d2/detector_evaluation.py`: the detector must predict **harm labels = actual per-cluster retrieval
loss** (cluster's NDCG drop from clean→corrupted), NOT the k-means injection knob and NOT
result-list Jaccard. Features → P(retrieval loss). Measure **AUPRC against harm labels**. Do NOT
reuse `IsomerDetector`'s result-list Jaccard as the detector (API mismatch — it scores result lists,
not doc clusters). If you wrap any existing detector, adapt it to score doc-clusters against harm
labels or write a small purpose-built harm scorer.

## Leakage contract (binding)
No method sees eval qrels during fit. HP selection on a separate anchor-validation partition frozen
in preregistration. qrel-positive docs excluded from the anchor pool. All docs pass through the
learned map (never substitute known targets into the index).

## Gates (binding, from agreed-build-plan-v1.md)
- **G2 is a HARD gate before G3.** Compute the 10k paired-query bootstrap half-width for the primary
  ΔNDCG contrasts. If median half-width > ~0.015, do NOT interpret G3 — mark the cell
  **underpowered/negative** (or note the query budget needed). No kill claim on noise.
- **G3 chelation kill (dual-CI rule).** A chelation "win" requires ALL of: ≥5 recovery points AND
  ≥0.02 absolute NDCG AND 95% CI excludes 0 AND same sign in all 5 seeds AND the cell's min oracle
  gap ≥0.05 (Grok FP-D: 5 pts on a 0.03 gap = noise). Kill criteria apply ONLY to the unpaired /
  detector-gated primary arm — the privileged paired local-ridge is a DIAGNOSTIC, not a chelation win.
  Require BOTH axes: detection **AUPRC ≥ 0.80 on harm labels** AND correction beating baselines. A
  detection-only pass (good AUPRC, no correction win) → park chelation as a re-embed **router**, not
  a corrector; that is still a KILL of the corrector claim.
- **Holm family FROZEN to: chelation-vs-CBIE, chelation-vs-hubness** (± one preregistered contrast).
  Everything else is exploratory and labeled as such.

## Deliverables
1. `regimes/synthetic_collapse.py`, `methods/*` (above), `d2/crossover.py`,
   `d2/detector_evaluation.py`, `d2/decisions.py` (the dual-CI kill logic), all unit-tested.
2. `protocols/d2_crossover.yaml` — preregistered cells, seeds, Holm family, gates.
3. `research/drift_recovery/out/d2/` artifacts: per-cell bootstrap CIs, detector AUPRC vs harm
   labels, and `d2_decision.md` applying G2 then G3 and stating the verdict (expected: KILL).
4. `tests/test_synthetic_collapse.py` + `tests/test_d2_decisions.py` (dual-CI logic on synthetic
   inputs; detector-harm-label wiring; leakage guard). Run the suite; paste pass/fail.
5. `research/drift_recovery/out/d2/D2_REPORT.md` (≤1 page): the verdict, which arm was judged, the G2
   power status per cell, AUPRC, and the single most important caveat.

Be brutally honest. This is designed to kill the corrector claim cleanly; if chelation somehow wins a
cell under the full dual-CI rule, that is a real (surprising) positive — report it with the exact CI
and seed-sign evidence, do not dismiss it. If a cell is underpowered (G2 fail), say so and do NOT
report a G3 verdict for it. Report the smallest concrete thing that does not work.
