# Execution Plan: Drift-Recovery Evidence + Paper Draft (June 2026)

**Audience:** an implementing agent (assume lower reasoning capacity — follow steps
literally, do not improvise architecture, escalate per §8 when blocked).
**Owner:** mattmre · **Written:** 2026-06-11 · **Deadline:** evidence complete + paper
draft by 2026-06-30 (stretch goal, acknowledged).
**Location note:** this folder is git-excluded (local-only). Code PRs from this plan are
public as normal; the paper draft and this plan stay in this folder.

---

## §0. The one-sentence goal

Build and run ONE experiment — **detection-triggered bounded embedding correction
recovering retrieval quality after injected drift, vs. three baselines** — capture
artifacts, and draft an arXiv-ready preprint around it.

### Why this experiment (decided — do not re-litigate)

Per [novelty-assessment-five-concepts.md](novelty-assessment-five-concepts.md) and
[shims-steering-posts-deep-dive.md](shims-steering-posts-deep-dive.md): every component
of ChelatedAI is published piecewise (Search-Adaptor, DIME, Ada-IVF, SmartVector, SVF),
but **no published system closes the loop**: drift detection that *triggers* bounded
correction under an annealing schedule. Also verified (refuted-claim 0-3): bounded
corrections are NOT in the closest prior art. The drift-recovery experiment is therefore
the smallest piece of runtime evidence that demonstrates the novel claim. It is already
roadmapped as Phase II step 14 in `docs/ROADMAP_EXECUTION.md` — this plan fast-tracks a
minimal version of steps 11+13+14 only. Steering banks, evidence DAG, GNN, and
correction-geometry are explicitly OUT of scope this month (paper future-work section only).

---

## §1. Scope freeze

| In scope (June) | Out of scope (do not build) |
|---|---|
| `drift_injector.py` — seeded, parameterized drift | Steering-post bank / routes |
| `drift_recovery_metrics.py` — trajectory + recovery metrics | Evidence DAG schema or GNN |
| `annealing_controller.py` — minimal temperature controller | Disk/computational-storage integration |
| `run_drift_recovery_experiment.py` — conditions C0–C4 harness | Base-model-swap drift (D3) unless §6 stretch reached |
| Campaign runs on SciFact (+NFCorpus stretch) | New adapter types, new loss functions |
| Paper draft (local-only) | Conference submission (arXiv preprint only) |

Dataset: **SciFact** via the existing BEIR machinery (CPU-friendly; road-course scripts
already use `--max-queries 100 --sample-docs 1200`). Model: `all-MiniLM-L6-v2`.

---

## §2. Repo conventions the implementer MUST follow (verified 2026-06-11)

- Flat layout: new `.py` files at repo root. New tests `test_*.py` at root, `unittest`
  ONLY (no pytest imports). Python 3.9 compat: use `from __future__ import annotations`
  or `Optional[...]`, never bare `X | None`.
- Mock logger in tests: `patch('<module>.get_logger')` returning `MagicMock()`.
  In-memory Qdrant: `qdrant_location=":memory:"`.
- Verified engine APIs to build on (do NOT invent new ones without checking):
  - `AntigravityEngine.ingest(text_corpus, payloads)` — antigravity_engine.py:318
  - `AntigravityEngine.set_temperature(temperature)` — :800
  - `AntigravityEngine.enable_online_updates(...)` — :911
  - `AntigravityEngine.enable_stability_tracking()` — :1406
  - `AntigravityEngine.run_sedimentation_cycle(threshold, learning_rate, epochs, noise_injection)` — :1604
  - `AntigravityEngine.run_inference(query_text)` — :2323
  - `AntigravityEngine.enable_kalman_lr(...)` — :768
  - `create_adapter("mlp"|"procrustes"|"low_rank", input_dim, bounded=True)` — chelation_adapter.py
  - Drift signals: `isomer_detector.py`, `stability_tracker.py`, `topology_analyzer.py`
- Every PR: full implementation (no stubs/TODOs), BHS body per §7, lint `ruff check .`
  clean, all tests green locally before opening.
- **No RP2040/hardware claims. No fabricated metrics — every number in artifacts must
  come from an actual run (Session Rule 3).**

---

## §3. Work breakdown — five PRs, in order

### PR-1: `drift_injector.py` + `test_drift_injector.py` (~Days 1–3)

A deterministic, seeded drift instrument operating on an `AntigravityEngine`'s vector
store. Implement EXACTLY two drift modes:

```python
class DriftInjector:
    def __init__(self, engine, seed: int): ...
    def inject_rotation_drift(self, fraction: float, angle_degrees: float,
                              dims: "Optional[list]" = None) -> dict:
        """Apply a seeded random rotation (Givens rotations on random dim pairs,
        or dims if given) to `fraction` of stored vectors. Returns manifest:
        {affected_ids, angle, dims, seed, checksum_before, checksum_after}."""
    def inject_noise_drift(self, fraction: float, sigma: float) -> dict:
        """Add seeded Gaussian noise (std=sigma) to `fraction` of stored vectors,
        then re-normalize. Returns same manifest shape."""
```

Implementation notes: read vectors from the Qdrant store via the existing
`vector_store.py` abstraction, modify, upsert back. The manifest MUST be JSON-serializable
and saved by callers — it is the reproducibility record.

Acceptance (tests must prove, not assert-trivially):
1. Same seed → byte-identical post-drift vectors (determinism).
2. `fraction=0.3` → exactly 30% (±1) of points affected; the rest bit-identical.
3. Retrieval degradation is real: on a 50-doc in-memory corpus, NDCG@10 for a fixed
   query set drops by a measurable amount after `inject_rotation_drift(0.5, 25.0)`.
   (This is the runtime evidence for the PR.)
4. Manifest round-trips through JSON.

SMOKE: `python -m unittest test_drift_injector -v`

### PR-2: `drift_recovery_metrics.py` + `test_drift_recovery_metrics.py` (~Days 3–5)

```python
class RecoveryTracker:
    def __init__(self, baseline_ndcg: float, recovery_threshold: float = 0.95): ...
    def record_cycle(self, cycle_index: int, ndcg: float, metadata: dict) -> None: ...
    def recovery_cycle(self) -> "Optional[int]":
        """First cycle where ndcg >= recovery_threshold * baseline, sustained for
        2 consecutive cycles. None if never."""
    def post_recovery_stability(self) -> "Optional[float]":  # std of ndcg after recovery
    def trajectory(self) -> list:  # [(cycle, ndcg, metadata)] for plotting/JSON
    def to_json(self) -> dict: ...
```

Plus a pure function `ndcg_at_k(ranked_ids, relevant_ids, k=10) -> float` (or reuse the
one in `benchmark_utils.py` / `benchmark_beir.py` if present — CHECK FIRST with Grep, do
not duplicate; if reusing, this PR shrinks to RecoveryTracker only).

Acceptance: unit tests with hand-computed NDCG values; recovery_cycle correct on
synthetic trajectories (recovers / never recovers / recovers-then-dips).

SMOKE: `python -m unittest test_drift_recovery_metrics -v`

### PR-3: `annealing_controller.py` + engine hook + `test_annealing_controller.py` (~Days 5–9)

Minimal unified temperature controller (Phase II #11, minimal form):

```python
class AnnealingController:
    """Maps a drift signal to a temperature, cools it per cycle, and exposes
    the knob settings for each correction cycle."""
    def __init__(self, initial_temperature: float = 0.0, cooling_rate: float = 0.7,
                 trigger_threshold: float = 0.15, max_temperature: float = 1.0): ...
    def observe_drift(self, drift_magnitude: float) -> None:
        """Raise temperature proportional to drift if above trigger_threshold."""
    def should_correct(self) -> bool:  # temperature > epsilon
    def cycle_settings(self) -> dict:
        """{'learning_rate_scale': f(T), 'epochs': g(T), 'online_intensity': h(T)}
        High T -> aggressive (lr*1.0, epochs=3); low T -> gentle (lr*0.1, epochs=1)."""
    def end_cycle(self) -> None:  # temperature *= cooling_rate
```

Engine integration: add `AntigravityEngine.enable_annealing_controller(**kwargs)` which
wires controller settings into `run_sedimentation_cycle` calls (scale the passed
learning_rate/epochs) and `set_temperature`. Drift magnitude input: use the existing
stability/isomer signals — Grep `isomer_detector.py` and `stability_tracker.py` for their
public score outputs and pick the simplest scalar (document WHICH one in the PR body).
Do not modify the detectors themselves.

Acceptance: controller unit tests (trigger, cooling, monotone knob mapping); integration
test on a tiny in-memory engine showing a sedimentation cycle actually receives scaled
parameters (assert via log capture or return values, not mocks-only — at least one test
must run the real engine path with `qdrant_location=":memory:"`).

SMOKE: `python -m unittest test_annealing_controller -v`

### PR-4: `run_drift_recovery_experiment.py` + `test_run_drift_recovery_experiment.py` (~Days 9–13)

The harness. CLI:

```
python run_drift_recovery_experiment.py --task SciFact --max-queries 100
  --sample-docs 1200 --condition C3 --drift rotation --fraction 0.5 --angle 25
  --cycles 12 --seed 42 --output experiment_runs/drift-recovery/C3_rot_seed42.json
```

Conditions (each a function, all sharing one ingest+drift+measure skeleton):

| ID | Condition | Implementation |
|---|---|---|
| C0 | Frozen | No correction after drift. Lower bound. |
| C1 | Static adapter (Search-Adaptor analog) | Train adapter ONCE on pre-drift data via one `run_sedimentation_cycle`; freeze; no post-drift updates. |
| C2 | Maintenance-only (Ada-IVF analog) | After drift: re-ingest/re-index affected docs from raw text with the SAME frozen embeddings — index structure refresh, no embedding correction. |
| C3 | **Closed loop (OURS)** | `enable_annealing_controller` + `bounded=True` adapter + detection-triggered `run_sedimentation_cycle` per cycle while `should_correct()`. |
| C4 | Unbounded ablation | C3 with `bounded=False`. |

Protocol per run (fixed, seeded): ingest → measure baseline NDCG@10 over the query set →
inject drift (save manifest) → for cycle in 1..N: (condition-specific action) → measure →
record. Output JSON: config, drift manifest, full trajectory, recovery_cycle, stability,
correction-norm stats (C3/C4 — prove boundedness numerically), wall-clock.

Acceptance: smoke-scale end-to-end test (tiny synthetic corpus, 2 cycles, all five
conditions produce well-formed JSON through the REAL engine path); determinism test
(same seed twice → same trajectory).

SMOKE: `python -m unittest test_run_drift_recovery_experiment -v` AND one real tiny run:
`python run_drift_recovery_experiment.py --task SciFact --max-queries 10 --sample-docs 200 --condition C0 --cycles 2 --seed 1 --output experiment_runs/drift-recovery/smoke_C0.json`

### PR-5: Campaign execution + results doc (~Days 13–16)

No new product code. Run the matrix; commit artifacts + a results doc.

Matrix (30 runs): conditions {C0,C1,C2,C3,C4} × drift {rotation(0.5, 25°), noise(0.5,
σ=0.05)} × seeds {42, 1337, 7}. SciFact, `--max-queries 100 --sample-docs 1200
--cycles 12`. If a full run exceeds ~45 min on CPU, halve sample-docs and document it.

Deliverables: all JSONs under `experiment_runs/drift-recovery/`; plots (matplotlib,
NDCG-vs-cycle per condition, mean±std across seeds) under the same dir;
`docs/drift-recovery-results-2026-06.md` with the results table:

| Condition | Recovery@12 (rot) | Final NDCG (rot) | Recovery@12 (noise) | Final NDCG (noise) | Mean ‖correction‖ |

**Honesty rule: report whatever the numbers say.** If C3 does not beat C1/C2, the paper
becomes "an honest negative/mixed result + system description" — that is still a
publishable preprint and still demonstrates capability. Do NOT tune until it wins and
report only the winning config (that is L4/score-gaming). Pre-registered primary
comparison: C3 vs C1 and C3 vs C2 on recovery_cycle and final NDCG, 3 seeds.

---

## §4. Paper draft (parallel track, local-only)

Write in `docs/waypoint-research-2026-06-09/paper-draft/` (inside the excluded folder).
`main.md` first; convert to LaTeX (arXiv two-column) only after content freeze ~Day 17.

Working title: *"Detection-Triggered Bounded Embedding Correction Under Drift: A
Closed-Loop Approach to Retrieval Maintenance"*.

| Section | Source material | Can start |
|---|---|---|
| 1 Intro (problem: silent retrieval decay; gap: no closed loop) | novelty assessment | Day 1 |
| 2 Related work (Search-Adaptor, DIME, 2602.03306, 2603.21437, Ada-IVF, Quake, FreshDiskANN, SmartVector, SVF/FLAS one paragraph) | both research docs | Day 1 |
| 3 System (engine, detectors, bounded adapters, annealing controller) | repo + PR-3 | Day 9 |
| 4 Experiment design (protocol, conditions, metrics, seeds) | PR-4 | Day 11 |
| 5 Results (tables/plots from PR-5) | PR-5 artifacts | Day 16 |
| 6 Limitations (single dataset, synthetic drift, CPU scale, layman-led prototype) | — | Day 16 |
| 7 Future work (steering banks+lifecycle, evidence DAG w/ actuator nodes, correction geometry across model swaps) | shim deep-dive | Day 3 |

Tone rules: claim ONLY what PR-5 artifacts show; cite every adjacent system named in the
waypoint docs; never claim novelty without the qualifier "to our knowledge"; state
plainly that components build on published work and the contribution is the closed loop
+ bounded-correction evidence. Venue: **arXiv preprint** (cs.IR). Risk: first-time arXiv
submission may need endorsement — fallback in order: TechRxiv, Zenodo DOI + tagged
GitHub release. A workshop submission is a July+ decision, not June.

---

## §5. Timeline (2026-06-11 → 06-30)

| Days | Work |
|---|---|
| 1–3 | PR-1 drift injector · paper §§1–2 drafted |
| 3–5 | PR-2 metrics · paper §7 |
| 5–9 | PR-3 annealing controller (hardest PR — escalate early per §8) · paper §3 |
| 9–13 | PR-4 harness · paper §4 |
| 13–16 | PR-5 campaign runs (start overnight runs Day 13) · paper §§5–6 |
| 17–19 | LaTeX conversion, figure polish, full read-through, buffer |

If ≥2 days behind by Day 9: drop noise-drift (rotation only) and C4 (bounded ablation
becomes future work). If ≥4 days behind by Day 13: drop NFCorpus/stretch entirely, run
matrix at `--max-queries 50`, and the paper becomes a short technical report. Never
compress by faking: cutting scope is allowed, inventing evidence is not.

## §6. Stretch goals (ONLY if PR-5 done before Day 17)

1. NFCorpus transfer run (same matrix, 1 seed).
2. Masking ablation C5 (C3 + `enable_learned_masking`).
3. Model-swap drift D3 (re-embed half the corpus with `all-mpnet-base-v2` through
   `DimensionProjection`) — first evidence toward the correction-geometry claim.

## §7. BHS compliance (every PR)

Per CLAUDE.md: PR body ends with `## Brutal Honesty` section (stubs L1, escape
conditionals L2, mocks-in-prod L3, partial-as-complete L4, untested paths L5/L8/L12,
broad catches L11, with file:line) and the required lines `BHS_SELF_DRAFT`,
`BHS_SELF_DRAFT_AGENT`, `BHS_TIER_B`, `BHS_TIER_B_AGENT`, `BHS_TIER_B_SEVERITY`,
`BHS_OFFICIAL`, `CARRY_FORWARD`, `DEFERRED_SCOPE`, `LOOP_ITERATIONS`,
`OPERATOR_OVERRIDE`. `EVIDENCE:` = actual command output from the production path (e.g.,
PR-1's measured NDCG drop). `SMOKE:` = the tier actually run, named. Tier B = fresh
adversarial agent on the diff. Only `BHS_OFFICIAL = 100` merges. Run
`python scripts/validate_pr_brutal_honesty.py` and `ruff check .` before opening.

## §8. Escalation rules for the implementing agent

1. An acceptance test fails twice for the same root cause → STOP, write the failure +
   hypothesis to `docs/waypoint-research-2026-06-09/blockers.md`, move to the paper
   track, surface to operator at session end. Do not loop (BHS Tier A: same gap
   surviving 2 iterations → escalate).
2. A named API doesn't match this plan → trust the code, Grep the real signature, note
   the discrepancy in the PR body. Do not redesign.
3. A campaign run crashes mid-matrix → save partial JSONs, record the crash, continue
   with remaining runs; never hand-edit result files.
4. Anything tempting you to write a placeholder/stub to stay on schedule → cut scope per
   §5 instead and record it in `DEFERRED_SCOPE:`.
5. Never push, tag, or publish anything from this folder; paper publishing is an
   operator-only action.

## §9. Definition of "month succeeded"

Minimum bar (must): PR-1..PR-4 merged at BHS 100; ≥1 full seeded matrix for rotation
drift across C0–C3 committed under `experiment_runs/drift-recovery/`; results doc with
real numbers; paper `main.md` complete through §6 with real figures.
Target: full §3 matrix + LaTeX draft.
Stretch: §6 items + arXiv-ready PDF.
