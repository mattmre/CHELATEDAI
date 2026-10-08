# IQ-07 — Non-autoregressive UI decision engine ("Local Jev") and its Phase-0 battery

**Status:** `QUEUED_HARNESS_REPAIR_REVIEW / NOT_FROZEN / NOT_RUN / TRAINING_AND_PROCUREMENT_HELD`.

**Parents:** `P4-REPRESENTATION` (RHPC), `P3-UTILITY` + RB-13/RB-14 (evidence kernels), `P5-REDESIGN` (localized assimilation / chelation gate). Deliberately **not** a new top-level lane — see [Lane mapping](#lane-mapping-external-ids-are-not-this-repos-lanes).

**Sources:** [verbatim conversation export](07-source-conversation-gliner25-vs-jev.md) (Google Doc "GLiNER2.5 vs. Jev Comparison", Drive id `1Zxif_429SXmTtddBwB9SlRXqcACqmO_FkgWmcnivBkE`, modified 2026-09-17) and the [`build_bundle.py` paste as received](07-source-bundle-as-received.txt) in the same session (the copy with line structure intact; substantively identical to §8 of that export). Both are `EXTERNAL_SOURCE / UNVERIFIED`.

**Owner/reviewer:** unassigned harness engineer / independent protocol reviewer. Follow [G0](shared-test-contract.md).

---

## Why this entry exists

The source proposes a local sub-10 ms non-autoregressive DOM decision engine (ModernBERT backbone + GLiNER-style boundary pointers + Jev-style calibrated scalar heads), augmented with four CHELATEDAI mechanisms, and gates a rented-compute training run behind a three-battery "zero-training falsification suite" (`run_phase0.py`).

**The architecture question is live and worth queueing. The Phase-0 battery as written cannot answer it.** This packet queues the repaired battery, not the pasted one, and records why.

## Verified review findings

Reproduction: [`07-review-evidence/`](07-review-evidence/RESULTS.md) (python 3.11.9, torch 2.5.1+cu121, 2026-09-17). Those three scripts are hand-repaired transcriptions of the pasted harnesses; the paste itself does not run. **Disclosure:** running them was seconds of synthetic-random-data arithmetic for source review — no model download, no fixture access, no campaign. It authorizes nothing downstream.

### V1 — The pasted `build_bundle.py` cannot execute, so nothing was ever verified on disk

Two independent blockers. (a) Every `FILES[...] = """..."""` entry wraps file content that itself opens a bare `"""` docstring, terminating the outer literal early; a minimal reduction of that shape raises `IndentationError`/`SyntaxError` on compile. (b) Markdown transit mangled `__init__` → `**init**` and `__name__ == "__main__"` → `**name** == "**main**"` throughout, and the embedded README's fences are unbalanced. The paste's own instruction ("run the script below … It will build and verify") is therefore unsatisfiable as received. It was **not** executed here. `L4` — a build-and-verify claim attached to a file that does not compile.

### V2 — The RHPC cycle detector never fires, on any cycle

Hand-repaired harness, its own defaults (`dim=2048`, `window=6`, `threshold=0.88`), orthonormal random action vectors:

| Trace | README claim | Measured alarms |
| --- | --- | --- |
| Valid 5-step DAG `ABCDE` | 0 false positives | `0` |
| `ABABAB` (k=2) | "detects within 1 tick" | **`0`** |
| `ABCABC` (k=3) | detects | **`0`** |
| 12-step k=5 repeat | detects | **`0`** |
| 200 random 5-step DAGs | `= 0.0%` FP | `0/200 = 0.0%` |

`rhpc_passed = (len(valid_alarms) == 0) and (len(loop_alarms) > 0)` is therefore `False`. **The advertised 0.0% false-positive rate is achieved by a detector that outputs "no cycle" unconditionally** — a constant-NO predictor scores identically, which is the ordinary baseline the scorecard never states.

Mechanism, not tuning: `H_t = circconv(R · H_{t-1}, a_t)` is a function of the **entire prefix**, not of a revisitable state. Re-entering action `A` after `A,B,C` does not return `H` to any earlier value, so a cosine-similarity-to-history test has nothing to match. Lowering the threshold trades this for false positives rather than fixing it; the fix is a state/period-detecting statistic, which is a different mechanism and needs its own derivation. One prior hypothesis of mine was **wrong and is withdrawn**: I expected spectral magnitude products to collapse or explode the state norm — measured `|H|` stays within `[0.990, 1.003]` across the trace. Norm drift is not the problem.

### V3 — The EMA baseline the README compares against is never run, and would not be matched

`run_trace_leaky` is defined and never called by `run_phase0.py`. No RHPC-vs-baseline comparison happens. Even if called, RHPC uses `threshold=0.88` and the baseline `0.95` — different operating points, so the comparison would be uninterpretable without a threshold sweep or ROC. `L5` untested production path + `L4` claimed-but-absent control.

### V4 — The coalition gate is unreachable, and both probes lose to the trivial baseline

Pass condition: linear `< 0.80` **and** bilinear `≥ 0.95`. As shipped (`n=2000`, `d_model=1024`):

| Configuration | Linear | Bilinear | Gate |
| --- | --- | --- | --- |
| As shipped, in-sample, real AND labels | `0.6445` | `0.6810` | **FAILS** (bilinear ≪ 0.95) |
| As shipped, **random labels** | `0.6335` | `0.6560` | ~unchanged |
| Held-out split (1000/1000), real labels | `0.5300` | `0.5530` | FAILS |
| Honest regime `n=20000`, `d=64`, held-out | `0.7567` | `0.8340` | FAILS |

Three separate defects. (a) **Unreachable gate:** the bilinear probe never approaches 0.95. (b) **Near-zero label sensitivity:** substituting random labels at a matched base rate moves accuracy by ~0.03, so the probe is barely measuring the AND structure it claims to test. (c) **Below the trivial baseline:** base rate is `0.248`, so always predicting `False` scores `0.752` — both probes are *worse than a constant*, because ridge with `λ=1e-3` on a near-square system (2048 and 3072 columns against 2000 rows, no split) is dominated by fit geometry rather than signal. Also `sigmoid(Xw) > 0.5` is identical to `Xw > 0`; the sigmoid is decorative.

In the honest regime the interaction term *does* help (`0.757 → 0.834`), which is the real and unsurprising content: multiplicative features help on an AND target. That is a textbook property of linear models, established on synthetic Gaussians with no ModernBERT, no DOM, and no UI. It is not evidence about UI coalitions and not evidence for `nn.Bilinear` in this architecture.

### V5 — The chelation gate passes only where the answer is arithmetic

`Prune_5%` returns exactly `1.0000`, not merely `≥ 0.95`: pruning 51 of 1024 dimensions removes all 50 corrupted dimensions, after which the clean and noisy matrices are *identical*, so cosine is 1 by construction. Sensitivity checks:

| Variant | `0%` | `2%` | `5%` | Reading |
| --- | --- | --- | --- | --- |
| As shipped (50 dims, σ=12, oracle variance-ratio) | `0.3569` | `0.4441` | `1.0` | passes trivially |
| Same, **noisy-variance-only** ranking (no clean reference) | `0.3569` | `0.4455` | `1.0` | oracle is not load-bearing here |
| Noise at σ=1 (signal-scale) | `0.9766` | `0.9856` | `1.0` | **`0%` prune already passes a ≥0.95 gate — no chelation needed** |
| Dense non-axis-aligned (rotated) noise, σ=0.5 | `0.8945` | `0.8945` | `0.8945` | pruning does nothing; fails for any coordinate method |

So the gate is satisfied when the noise is enormous and axis-aligned, is satisfied *without the method* when the noise is realistic in scale, and is unsatisfiable when the noise is not coordinate-aligned. It discriminates the noise model, not the method. Note `evaluate_chelation_bounds` takes `clean_embeddings` to rank coordinates — an input unavailable at runtime — though V5 row 2 shows that for *this* noise model it is unnecessary; the oracle dependency becomes load-bearing precisely in the harder regimes where the method would need to earn its place. This is the same failure class as the `C2` oracle already recorded in this repo's drift-recovery arc: a control that inverts synthetic corruption by construction.

### V6 — Production-path defects in the model and orchestrator

- **`L3`, most serious.** `LocalDecisionOrchestrator.step` computes `action_vec = normalize(torch.randn(self.dim))` and ignores `preds["action"]` and `preds["selector"]` entirely. The runtime loop-blocker is driven by **fresh noise**, so even with a working RHPC update it could not detect a repeated action. The advertised safety property is a random walk.
- **`L1`/`L4`.** `schema_proj` and `coalition_bilinear` are constructed in `__init__` and never referenced in `forward()`. "Jev-style schema evaluation" and the Lane L-04 coalition tensor — two of the four headline mechanisms — are dead parameters.
- **Not connected.** `step` calls `self.model.infer(...)`, which `ChelatedDecisionModel` does not define, and expects `{"selector", "action", "confidence"}`, which `forward()` does not return. Model and orchestrator have never run together.
- **Two unrelated spaces.** Model `d_model = 1024`; RHPC `dim = 2048`. No projection is specified between them.
- **Dependency pin wrong.** `ModernBertModel` requires `transformers >= 4.48`; `requirements.txt` pins `>= 4.40`. (Local env is 4.57.3, so this would not surface here — it would surface on a clean install.)
- **No latency measurement anywhere**, despite "sub-10 ms" in §1.
- **Reproducibility is accidental.** `run_phase0.py` seeds only inside `TrajectoryFalsifier` and the coalition probe; the chelation section's `torch.randn` consumes whatever global state remains, so its numbers are stable only while call order is unchanged.

### V7 — Unresourced and internally inconsistent procurement claims

"Dual DGX Sparks (GB10)" in §1 versus "8x RTX 6000 Ada" training in §4 are unreconciled. "20 epochs across 350k synthetic/grounded interaction pairs" names no dataset, no generator, no source, and no licence — nothing to acquire it exists in the source or this repo. Under the shared contract this is a `G0-RUN` blocker, not a schedule item.

### V8 — Repo-fit

The proposed `fixtures/ models/ orchestrator/` tree contradicts this repo's flat root layout; there are no `unittest` files, contrary to convention; and the source's `L-01/L-03/L-04/L-09/L-10` identifiers are an external scheme, not this repo's lane registry.

## Lane mapping: external IDs are not this repo's lanes

| Source ID | Actual home here | Binding constraint |
| --- | --- | --- |
| `L-10` RHPC | `P4-REPRESENTATION`; [RHPC Stage A](sources.md) is `FROZEN_NOT_OFFICIALLY_RUN` | Same-information ordinary controls and the existing signed EGV seal/restore + Spark gates apply. **A "Local Jev" RHPC harness must not become a second, unreviewed RHPC lane.** |
| `L-04` evidence kernels | `P3-UTILITY`, RB-13/RB-14 | Every named EK1–EK7 predecessor disposition stands; see [IQ-04](04-temporal-evidence-correction.md). |
| `L-01` coordinate chelation | `P5-REDESIGN` (localized assimilation) | Needs a deployable, non-oracle coordinate-selection contract before any learned gate. |
| `L-09` EGV ledger | Existing EGV lane | Append-only ledger and external signed verification boundary unchanged; no EGV source/solvability decision is implied. |
| `L-03` liquified lattice | Phase II rungs 15–17 | Referenced by the source (adaptation 2) but absent from the bundle. Out of scope here. |

## Work items and dependencies

- [ ] **Y0 — battery repair and honest-baseline restatement (0.5–1 day, documentation + local arithmetic):** obtain a clean copy of the bundle (do not un-flatten the Docs export, and do not patch the mangled paste in place). Restate each of the four scorecard rows against the ordinary baseline it currently omits: constant-NO for cycle detection, majority-class for the coalition probe, and 0%-prune for chelation. A gate that the trivial predictor also passes is not a gate. Produce a corrected scorecard or record the battery as `INVALID` and stop.
- [ ] **Y1 — cycle detection, mechanism first (2–4 days, after Y0 and G0-BUILD):** decide whether the target is *state revisitation* or *action-sequence periodicity*; they need different statistics. Specify the candidate against explicit ordinary baselines — exact visited-state set/hash, action n-gram counter, Bloom filter, EMA-of-embedding — on traces with declared cycle period `k ∈ {2,3,6}`, declared near-miss non-cycles (the case that matters: a legitimately repeated action under different state), and declared trace lengths. Report a threshold sweep and ROC/AUC per method, not a single operating point. **Prediction to falsify:** exact state-hash detection is both cheaper and strictly better on synthetic traces, and RHPC's only defensible claim is bounded memory under approximate matching — which must then be measured as a memory/accuracy curve, not asserted.
- [ ] **Y2 — coalition necessity, honestly powered (1–2 days, after Y0):** `n ≫ d`, held-out split, majority-class and logistic baselines, and a real encoder's representations rather than `torch.randn`. If the question is whether ModernBERT already linearly encodes a two-field AND precondition, the probe must run on ModernBERT activations over real DOM text; on Gaussians it answers nothing about the architecture. Verdict is `nn.Bilinear` warranted / not warranted, at the measured margin.
- [ ] **Y3 — chelation with a deployable selector (2–4 days, after Y0):** replace the clean-reference ranking with one computable at inference. Evaluate across a declared noise grid — axis-aligned vs rotated, σ from signal-scale to 12× — and report the full surface including the regimes where coordinate masking provably cannot help (V5 row 4). Success requires beating 0%-prune at signal-scale noise, which the current fixture never tests.
- [ ] **Y4 — architecture integration, conditional on Y1–Y3 (1–2 weeks, after G0-RUN):** only for mechanisms that survived. Wire `forward()` to the heads it declares or delete them; give the orchestrator the real action vector; define the 1024↔2048 projection; add a measured p50/p99 latency harness before any "sub-10 ms" wording appears anywhere. Repo-flat layout and `unittest` files per convention.
- [ ] **Y5 — training run. HELD.** Requires Y1–Y4 surviving, a named dataset with provenance and licence, one reconciled hardware target, a cost estimate, and explicit operator run-admission. Nothing in this packet reserves compute or authorizes procurement.

Dependency edges: `Y0 -> Y1`, `Y0 -> Y2`, `Y0 -> Y3`, `{Y1,Y2,Y3} -> Y4 -> Y5`.

## Go/no-go boundaries

- **Every battery row must beat the trivial predictor named in Y0.** This is the binding gate; the pasted scorecard fails it on all three rows measured.
- **Cycle detection:** RHPC is `CLOSED_AS_DOMINATED` unless it beats exact state-hashing on a stated axis (memory at fixed recall, or recall under state aliasing). Matching a hash table while costing a 2048² matmul per step is not a result.
- **Coalition:** `nn.Bilinear` is not warranted unless the interaction term gives a held-out margin over a logistic baseline **on real encoder activations**. A synthetic-Gaussian margin does not transfer and cannot be cited.
- **Chelation:** must beat 0%-prune at signal-scale noise with an inference-time selector. Oracle-ranked recovery on 12σ axis-aligned noise is `CLOSED_AS_CONSTRUCTED`.
- **Latency:** "sub-10 ms" may not appear in any doc, README, or roadmap entry until measured p50/p99 on named hardware exists. Per CLAUDE.md Rule 2 (visible means verified), unmeasured latency must not be surfaced as a property.
- A repaired harness that passes is evidence about the *harness*, not about the engine. Y4 integration and Y5 training remain separately gated.

## Budget and handoff

### Selected mathematical extensions — 2026-10-08

Y0 must first repair the invalid supplied battery. If Y0 then selects a virtual-mixture, holographic-code, or compressed-memory mechanism, reuse [MA0/MA2/MA3/MA4/MA7](math-findings-integration-2026-10-07.md) and the same named cases under the mapped parent lane's admission. Fixed-linear certificates and codebook diagnostics do not validate the current RHPC/coalition/chelation gates, and no mathematical source authorizes Y5 training or procurement. Handoff records which MA extensions were selected versus related-only; it does not duplicate the shared case implementation or launch a new campaign.

Y0–Y3 are single-CPU-process local work; Y2 and Y3 need one small encoder already in the local HF cache, no download. Propose the shared contract's default ceiling: one process, 8 GiB RSS, ten minutes per battery after profiling, no network. Y4 needs one GPU for latency measurement only. Y5 cost is **unknown and unallocated**; the source's two hardware targets and absent dataset must both be resolved before an estimate exists.

Future artifacts: corrected scorecard with ordinary-baseline columns; per-method threshold sweeps and ROC; noise-grid surface for chelation; held-out coalition results on real activations; measured latency distribution; and one of `INVALID`, `INCONCLUSIVE`, `NEGATIVE_IN_SCOPE`, `SURVIVES_METHOD_DEV`, `SUPPORTS_FROZEN_ENDPOINT`.

Unresolved before dispatch: a clean copy of the bundle; whether the cycle target is state revisitation or action periodicity; the inference-time coordinate selector; the dataset and its licence; one hardware target; and operator run-admission. This packet is documentation, not runnable code or a frozen preregistration.
