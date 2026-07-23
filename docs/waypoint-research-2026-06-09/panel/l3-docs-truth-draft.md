# L3 docs-truth draft — Phase II steps 9–17 (chair applies; panel only)

**Role:** chair-ready paste blocks. Do **not** apply to `docs/ROADMAP_EXECUTION.md` / `CHANGELOG.md` until chair accepts.  
**Verified against:** `origin/main` @ `34ce4b56` (feat lattice H4 #291), plus this-slice H5 VERDICT artifacts on worktree `lattice/phase2-continue-20260713` (staged/local results docs; **not yet a merged results PR on `origin/main` as of verification**).  
**Main advance:** 33 commits after vision/roadmap land (`bf23a47f` → `origin/main` tip).  
**Sources:** `docs/ROADMAP_EXECUTION.md`, root `CHANGELOG.md`, `git log`/`git show` on merged PR SHAs, targeted greps for steps 15–17.

**Hard rules used:** no “delivered” without a merged PR (or explicit this-slice artifact path). Partial called partial. Speculating is a lie — unknown marked unknown.

---

## 0. Structural fact (edit shape)

`docs/ROADMAP_EXECUTION.md` Phase II table today has columns **Step | Track | Exit criteria** only — **there is no Status column**. Staleness is “program still reads as future work” (vision-era prose + CHANGELOG Unreleased Findings still talk about Phase I / SHIM on hold), not a wrong Status cell.

Chair must **add a Status column** (recommended) *or* append status parentheticals into Track. Below proposes **exact old → new** for the full Phase II table (Status column added).

---

## 1. Per-step truth (9–17) — git-verified

| Step | Track (roadmap) | True status | Delivering PRs / commits (merged on `origin/main` unless noted) | Exit-criteria honesty |
|------|-----------------|-------------|----------------------------------------------------------------|------------------------|
| **9** | Model-Scope shadow (close Phase I #7) | **DONE (fixture path)** | Model-Scope stack slices **#229** (Phase 6 engine+dashboard), **#228–#227** (phases 4–5), steering/actuator earlier slices; BHS close **#254** (`persist_records` / intervention caps / rollback). Tests on main: `test_model_scope_steering.py` (`persist_records` / `load_records` round-trip, `max_total_interventions_exceeded`); bridge uses `max_total_interventions`. | Exit criteria (fixture + persist/load + bridge cap in test) are met. **Not** a claim that a live Qwen3.5-9B shadow pilot on real weights is closed — CHANGELOG Findings still mark step 7 real-weights as fixture/integration-gated; that is separate from step 9 fixture DoD. |
| **10** | SHIM substrate DoD | **DONE (substrate + DoD correction)** | **#284** A1a executable steering rollback (`model_scope_steering` RollbackPlan path); **#285** A1b `steering_route_promotion.py` (quant-survival **and** actionable rollback); **#286** A1c `production_steering_control.py` + `docs/rung10-shim-substrate-dod.md`; **#289** DoD correction — diagnostics seam is **observation-only** (no live SOFT_SCALE/SUPPRESSION to un-guard). | Rung-10 **control-plane substrate** is complete and default-safe. **#289** explicitly: introducing live promotable routes overlaps **rung 16**, not a remaining rung-10 gap. Do **not** re-read old SHIM-CD-01/02/06 “production seams without env-only guards” wording as fully closed historical SHIM-CD rows — DoD is the promotable-route control plane, not a blanket SHIM-CD CLOSED claim. |
| **11** | Annealing controller | **DONE (two complementary modules)** | **#260** `annealing_controller.py` + engine `enable_annealing_controller` / `observe_annealing_drift` / temperature sync on sedimentation path (`antigravity_engine.py`); tests `test_annealing_controller.py`. **#280** `annealing_schedule.py` (H6) — cycle schedule (constant/linear/cosine/step/adaptive) for post-bank explore↔stabilize; tests `test_annealing_schedule.py`; wired via post-bank conditions (**#283**). | Exit text (“single module owns temperature schedule… wired to sedimentation + online_updater + ES”) is **slightly over-specified vs reality**: schedule ownership is **split** (engine scalar controller **#260** vs post-bank schedule **#280**). High-T/low-T behavior is unit-proven. Treat as **DONE for lattice rungs**, not as a single-module monopoly. |
| **12** | Evidence DAG schema | **DONE** | **#277** `evidence_dag.py` + `test_evidence_dag.py` — typed nodes/edges, JSON schema, validator, `from_attribution_pool()`. Explicitly **no GNN**. | Matches exit criteria. |
| **13** | Disintegration loop | **PARTIAL — lifecycle substrate DONE; literal trigger wiring NOT done** | **#279** H5a `steering_post_bank.py` — fitness → prune → re-anneal lifecycle, temperature-modulated thresholds, mutation log. **#281–#283** builder/runtime/C5 conditions exercise lifecycle end-to-end in harness tests. | Exit criteria name **`isomer_detector` / `convergence_monitor` triggers prune of graph edges or pool entries**. Grep: those modules are **not** wired to Evidence DAG edge prune or post-bank prune. Delivered path is **post-bank fitness prune/re-anneal**, not isomer/convergence-driven DAG disintegration. Artifact fitness before/after exists in lifecycle logs for the post-bank path. **Remaining:** wire (or formally re-scope) exit criteria to the post-bank mechanism, or implement isomer/convergence → DAG prune. |
| **14** | Concept-drift experiment | **DONE (apparatus + campaigns); H5 living-bank question CLOSED as negative (this slice)** | **Harness / injection:** **#258–#266** (injector, metrics, harness, campaign results, diagnostics, calibration, knob sweep) + **#261** replay contract. **Swap arena / H1–H2 line:** **#268–#276** (+ local `results(h2)` commit `2b0fa044` on worktree branch, **ahead of `origin/main`** — chair: confirm merge). **H3 teacher-supervised:** **#287** module + **#288** C3b harness wiring. **H4 compound knob:** **#291** `compound_cycles`. **H5 apparatus:** **#279** H5a; **#281/#282/#283** H5b builder/runtime/conditions; **#290** head-to-head driver (PR body deferred GPU run). **H5 VERDICT (this slice):** real GPU campaign artifacts + frozen docs (see §1.1) — **LIVING BANK WINS = False** on SciFact and NFCorpus. | Exit: injected drift + measurable recovery documentation. Harness + multiple result docs exist under `docs/drift-recovery-*.md`. **Honest outcome of the post-bank living-bank hypothesis (H5):** negative (see §1.1). H4 single-seed ablation: compounding catastrophic (~45× collapse). Do not promote living/annealed post-bank. |
| **15** | GNN prototype | **OPEN — not delivered** | No merged PR. | Grep `*.py`: no `torch_geometric` / `dgl` / GraphConv prototype. Only prose mentions in `evidence_dag.py` docstring (“a GNN (rung 15) would later learn over”). **Not delivered.** |
| **16** | Quant-aware shim routing | **OPEN — substrate pieces exist; integration as steering plane NOT delivered** | Pieces on main: `adapter_router.py` + `AntigravityEngine.enable_adapter_routing` (opt-in centroid routing); `QuantizationPromotionGate` used in ES/distillation/road-course/live-fire diagnostics; **#285** composes quant gate into **steering-route** promotion (Model-Scope routes), not adapter_router+retrieval-fitness as one steering plane. **#289** marks live promotable routes as future / rung-16 overlap. | Exit: “`adapter_router.py` + `QuantizationPromotionGate` **integrated as steering plane**; promotion requires quant survival **+ retrieval fitness**.” Grep does **not** show that integrated plane or a PR that closes it. **Do not mark DONE.** Closest partial: A1b quant+rollback for Model-Scope routes (#285), observation-only diagnostics path. |
| **17** | Disk pool slice | **OPEN — not delivered** | No merged PR for “one precomputed pool shard readable via `computational_storage_poc/block_graph.py` with host parity.” | `block_graph.py` exists for computational-storage graph execution (weights/activations), not a lattice precomputed **retrieval pool shard** parity path. No storage-track doc claiming step 17 closed. **Not delivered.** |

### 1.1 H5 VERDICT run (this slice) — numbers from frozen docs

Sources (worktree; chair should merge/publish before citing as `main` truth):

- `docs/drift-recovery-post-bank-headtohead-results-2026-06.md` (SciFact)
- `docs/drift-recovery-post-bank-headtohead-nfcorpus-results-2026-06.md` (NFCorpus)
- `docs/drift-recovery-h4-compound-cycles-ablation-2026-07.md` (H4; single seed 42)
- manifests under `experiment_runs/drift-recovery/post-bank-headtohead*/`

| Dataset | C5 living | C5s static | C5r one-shot | C5>C5s | C5>C5r | **LIVING BANK WINS** |
|---------|----------:|-----------:|-------------:|:------:|:------:|:--------------------:|
| SciFact | 0.131135 | 0.131135 | **0.180862** | False | False | **False** |
| NFCorpus | 0.046389 | 0.046389 | 0.045817 | False | True | **False** |

Gate (from #290 / result docs): living wins iff C5 beats **both** C5s and C5r. **Both datasets fail the gate.** SciFact: C5 **bit-identical** to C5s; C5r beats living. NFCorpus: C5==C5s; living edges one-shot slightly but gate still fails.

H4 (SciFact C4a, seed 42 only): `compound_cycles=False` final NDCG **0.236297** vs `True` **0.005258** (~45× collapse). Single-seed; direction unambiguous.

Apparatus PRs for the series (merged): #284–#286 + #289 (rung 10), #260+#280 (rung 11), #277 (rung 12), #279+#281–#283 (rung 13/H5a–b lifecycle+wiring), #287–#288 (H3), #291 (H4 knob), #290 (H5 driver). Verdict **execution** is this slice’s runtime evidence, not #290 alone.

---

## 2. EXACT text edits — `docs/ROADMAP_EXECUTION.md`

### 2.1 Phase II table — replace entire table block

**OLD:**

```markdown
| Step | Track | Exit criteria |
|------|--------|----------------|
| 9 | **Model-Scope shadow (close Phase I #7)** | One bounded steering policy on Qwen3.5-9B fixture; `persist_records` / `load_records` round-trip; bridge `max_total_interventions` exercised in test |
| 10 | **SHIM substrate DoD** | SHIM-CD-01/02/06 production seams wired without env-only guards; rollback test; restore `Blocking=YES` on open SHIM rows when re-entering |
| 11 | **Annealing controller** | Single module owns temperature schedule (explore ↔ stabilize); wired to sedimentation + `online_updater` + ES entrypoint; unit test proves high-T increases perturbation, low-T reduces it |
| 12 | **Evidence DAG schema** | Typed graph contract over `build_attribution_pool.py` output (nodes: query/cluster/actuator; edges: retrieval/intervention links); JSON schema + validator; no GNN required yet |
| 13 | **Disintegration loop** | `isomer_detector` / `convergence_monitor` triggers prune of low-fitness graph edges or pool entries; re-anneal path records fitness before/after in artifact |
| 14 | **Concept-drift experiment** | Injected drift fixture (extend `chelatedai-synthetic-collapse` or road-course window); measurable recovery within N anneal cycles documented in `CHANGELOG.md` |
| 15 | **GNN prototype** | Lightweight GNN over evidence DAG (PyG or DGL); only after steps 12–14 green; must beat flat-pool baseline on drift fixture or fail closed |
| 16 | **Quant-aware shim routing** | `adapter_router.py` + `QuantizationPromotionGate` integrated as steering plane; promotion requires quant survival + retrieval fitness |
| 17 | **Disk pool slice** | One precomputed pool shard readable via `computational_storage_poc/block_graph.py` with host parity check; documented in storage track docs |
```

**NEW:**

```markdown
| Step | Track | Status | Exit criteria |
|------|--------|--------|----------------|
| 9 | **Model-Scope shadow (close Phase I #7)** | **DONE** (fixture path; #229 stack + #254 persist/cap tests) | One bounded steering policy on Qwen3.5-9B fixture; `persist_records` / `load_records` round-trip; bridge `max_total_interventions` exercised in test |
| 10 | **SHIM substrate DoD** | **DONE** (rung 10: #284 A1a / #285 A1b / #286 A1c; DoD honesty #289 — diagnostics observation-only; live routes = future/rung 16) | Promotable steering route may run LIVE iff quant-survival + actionable rollback, gated by default-safe control plane (`docs/rung10-shim-substrate-dod.md`). Historical SHIM-CD-01/02/06 env-guard wording superseded by this DoD — do not claim those CD rows CLOSED without separate evidence. |
| 11 | **Annealing controller** | **DONE** (#260 engine controller + #280 post-bank schedule; schedule ownership is split across two modules) | Temperature schedule(s) drive explore ↔ stabilize; unit tests prove high-T vs low-T behavior; engine path wires controller into sedimentation temperature; post-bank path uses `annealing_schedule` in C5 lifecycle |
| 12 | **Evidence DAG schema** | **DONE** (#277 `evidence_dag.py` + validator + JSON schema; no GNN) | Typed graph contract over `build_attribution_pool.py` output (nodes: query/cluster/actuator; edges: retrieval/intervention links); JSON schema + validator; no GNN required yet |
| 13 | **Disintegration loop** | **PARTIAL** — post-bank prune/re-anneal **DONE** (#279 H5a, wired #281–#283); **NOT** isomer/convergence → Evidence-DAG edge prune | Delivered: fitness-gated prune + re-anneal on `SteeringPostBank` with lifecycle artifacts. Remaining vs original exit text: `isomer_detector` / `convergence_monitor` triggers on DAG edges/pool entries — **not implemented**. Re-scope exit criteria or implement that trigger. |
| 14 | **Concept-drift experiment** | **DONE apparatus** (#258–#266 harness line; swap arena #268–#276; H3 #287/#288; H4 #291; H5 driver #290). **H5 living-bank VERDICT: FAIL / non-promoted** (this slice; SciFact+NFCorpus) | Injected drift + recovery campaigns documented under `docs/drift-recovery-*.md`. Living annealed post-bank (C5) does **not** beat static bank + one-shot router gate; do not promote living post-bank. Compounding (`compound_cycles=True`) is catastrophic on single-seed H4 ablation. |
| 15 | **GNN prototype** | **OPEN** (no merged PR; no PyG/DGL code) | Lightweight GNN over evidence DAG (PyG or DGL); only after steps 12–14 green; must beat flat-pool baseline on drift fixture or fail closed |
| 16 | **Quant-aware shim routing** | **OPEN** (pieces exist: `adapter_router`, `QuantizationPromotionGate`, A1b route gate #285 — **not** integrated as one retrieval-fitness steering plane) | `adapter_router.py` + `QuantizationPromotionGate` integrated as steering plane; promotion requires quant survival + retrieval fitness |
| 17 | **Disk pool slice** | **OPEN** (no pool-shard parity via `block_graph`; storage POC is not this slice) | One precomputed pool shard readable via `computational_storage_poc/block_graph.py` with host parity check; documented in storage track docs |
```

### 2.2 Optional one-line under Phase II header (after “Same one-track rule…”)

**OLD:** *(no status blurb; table only)*

**NEW (insert after the one-track paragraph):**

```markdown
**Status snapshot (2026-07, git-verified):** rungs **10–12 DONE**; **13 PARTIAL** (post-bank lifecycle only); **14 apparatus DONE** with **H5 living-bank closed negative**; **15–17 OPEN**. Lattice apparatus PRs: #260, #277, #279–#291 (see panel L3 draft). Next executable feature work on this queue is step **15** only after chair re-scopes or closes step **13** remainder — or step **16/17** if chair prioritizes quant plane / disk pool over GNN.
```

### 2.3 “What we are not doing” bullet (line ~43) — truth update

**OLD:**

```markdown
- No GNN layer or disk-pool integration until Phase II steps 12–14 complete (schema + drift experiment first).
```

**NEW:**

```markdown
- No GNN layer or disk-pool integration until Phase II steps 12–14 are honestly closed (schema **#277 DONE**; drift apparatus **DONE**; disintegration **PARTIAL** — do not start step 15 while claiming 13 fully green without chair re-scope).
```

---

## 3. CHANGELOG note (chair-ready paste)

Place under `## [Unreleased]` → `### Findings (honest status)` **or** a new `### Research results` subsection. Prefer Findings so promotion language stays disciplined.

**PROPOSED NEW BULLETS** (add; do not delete older bullets without a separate hygiene pass — older Unreleased Findings are themselves stale vs lattice work and should be refreshed in a follow-up):

```markdown
- **Lattice rungs 10–14 apparatus (merged):** SHIM/promotable-route substrate **#284/#285/#286** (+ DoD correction **#289**); annealing controller **#260** + post-bank temperature schedule **#280**; Evidence DAG schema **#277**; disintegration/lifecycle as post-bank prune/re-anneal **#279** (+ builder/runtime/conditions **#281/#282/#283**); concept-drift harness and campaigns **#258–#276** with H3 teacher-supervised **#287/#288**, H4 `compound_cycles` **#291**, H5 head-to-head driver **#290**.
- **H5 living-bank VERDICT (closed negative):** SciFact C5==C5s (0.131135) and C5r one-shot 0.180862 beats living; NFCorpus C5==C5s (0.046389), living edges C5r (0.045817) but does not clear the dual gate. **LIVING BANK WINS = False** on both datasets — living/annealed lifecycle adds nothing over a frozen static bank; one-shot router is competitive-or-better on SciFact. Sources: `docs/drift-recovery-post-bank-headtohead-results-2026-06.md`, `docs/drift-recovery-post-bank-headtohead-nfcorpus-results-2026-06.md`.
- **H4 compound_cycles (single-seed SciFact C4a):** False 0.236297 vs True 0.005258 (~45× collapse). Compounding is not the recovery path. Source: `docs/drift-recovery-h4-compound-cycles-ablation-2026-07.md`.
- **Still OPEN on Phase II queue:** step **15** GNN prototype (no PyG/DGL code); step **16** quant-aware shim routing as integrated steering plane (`adapter_router` + `QuantizationPromotionGate` + retrieval fitness — not closed by #285 alone); step **17** disk pool shard via `block_graph`. Step **13** remains **partial** until isomer/convergence→DAG prune is implemented or exit criteria re-scoped to post-bank lifecycle.
```

**Also propose** (same edit session if chair agrees) softening the Unreleased “Phase II execution program” Added line so it does not read as “not started”:

**OLD:**

```markdown
- **Phase II execution program** — [docs/ROADMAP_EXECUTION.md](docs/ROADMAP_EXECUTION.md) steps 9–17: Model-Scope close-out, SHIM DoD, annealing controller, evidence DAG, disintegration loop, drift experiment, GNN prototype, quant shim routing, disk pool slice.
```

**NEW:**

```markdown
- **Phase II execution program** — [docs/ROADMAP_EXECUTION.md](docs/ROADMAP_EXECUTION.md) steps 9–17. **Delivered apparatus:** 9–12, 14 (with H5 living-bank fail-closed), 13 partial (post-bank lifecycle). **Open:** 15 GNN, 16 quant-aware shim routing plane, 17 disk pool slice. Status column is source of truth once L3 chair-applies.
```

---

## 4. Grep evidence for OPEN steps 15–17 (do not assume)

| Check | Result |
|-------|--------|
| `torch_geometric` / `import dgl` / `from dgl` in `*.py` | **No matches** (only docstring forward-refs in `evidence_dag.py`) |
| GNN training module / beat-flat-pool experiment PR | **None** on `origin/main` log |
| `adapter_router` + `QuantizationPromotionGate` co-used as **one** production steering plane with retrieval-fitness promotion | **Not found.** Both exist; engine can opt into adapter routing; quant gate used in ES/distill/diagnostics; A1b (#285) is Model-Scope route quant+rollback, not step-16 DoD |
| `resolve_production_mode` / `evaluate_steering_route_promotion` called from live inference path | **Not on engine live path** per #289 (diagnostics remain observation-only) |
| Precomputed lattice pool shard via `block_graph.py` + host parity | **Not found.** `block_graph` serves computational-storage weight/activation graphs, not step-17 pool shards |

---

## 5. Merge-state caveats (chair must not lie in docs)

1. **H5 VERDICT docs + manifests** are present on worktree branch `lattice/phase2-continue-20260713` (staged at verification); **`origin/main` tip is still `34ce4b56` (#291)** without a `results(h5)` merge. Chair should land results before treating CHANGELOG H5 bullets as mainline history.
2. **`results(h2)` commit `2b0fa044`** is on the worktree branch (**ahead of `origin/main`**); do not cite as merged until it is.
3. **Step 13** must not be marked fully DONE without either (a) isomer/convergence→DAG prune PR, or (b) explicit exit-criteria re-scope in ROADMAP (chair decision).
4. **Step 10** DONE is **rung-10 DoD**, not “all SHIM-CD rows CLOSED.”

---

## 6. Suggested chair apply order

1. Merge/publish H5 (+ H4) results docs if not already on main.  
2. Apply ROADMAP table edit (§2.1) + optional snapshot (§2.2–2.3).  
3. Paste CHANGELOG findings (§3).  
4. Optional follow-up L2/L3: refresh stale Unreleased Findings (Phase I / turn 3171 / “SHIM on hold only”) so they do not contradict lattice delivery.

---

## 7. One-line summary for panel

**Rungs 10–12 done; 13 partial (post-bank lifecycle, not isomer/DAG prune); 14 apparatus done and H5 living-bank closed as a hard negative; 15–17 still open after grep.** No delivery claims without merged PRs; H5 numbers are this-slice runtime evidence pending results merge.
