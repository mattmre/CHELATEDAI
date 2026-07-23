# Phase II Final-State Audit — GIT-VERIFIED

**Date:** 2026-07-19
**Auditor:** fresh agent, adversarial posture ("try to disprove the ROADMAP")
**Repo:** `D:\GITHUB\CHELATEDAI\.claude\worktrees\relaxed-wozniak-271e04`
**Working tree:** CLEAN (`git status --short` empty)
**HEAD:** `2c78195d` on `lattice/rung16-routing-20260714`
**origin/main HEAD:** `34ce4b56` (#291, merged 2026-06-29)
**HEAD vs origin/main:** 0 behind / **11 ahead** — none of this session's work is on `main`.

Method: every status claim in `docs/ROADMAP_EXECUTION.md` (lines 43, 54–73) was checked against
`git log`, `gh pr view/list`, `git cat-file`/`git ls-tree` file presence on `origin/main` vs `HEAD`,
and `git grep` on the actual code — not against the prose. New rung test modules were executed.

**Local runtime evidence (chair-run, this audit):**
`python -m unittest test_evidence_dag_disintegration test_evidence_dag test_pool_shard_parity test_quant_aware_routing test_adapter_router`
→ **Ran 57 tests, OK** (0.843s).

---

## The table

| Step | ROADMAP claimed status | GIT-VERIFIED status | Evidence (PR / commit / file) | Discrepancy |
|---|---|---|---|---|
| **9** — Model-Scope shadow | DONE (fixture path; #254 persist/cap tests) | **DONE (code+test), but fixture-only — no real-model campaign artifact** | PR **#254 MERGED** 2026-05-17 (`9962cbb2` on main). `model_scope_*.py` (7 modules) + 12 `test_model_scope_*.py` on `origin/main`. `test_model_scope_steering.py:519-548` = `persist_records`/`load_records` round-trip. `test_model_scope_engine_bridge.py:674-690` = `max_total_interventions` wired+capped. `config.py:109` `MODEL_SCOPE_PRIMARY_PILOT_MODEL = "Qwen/Qwen3.5-9B"`. | **Minor understatement of weakness.** Exit criteria say "on Qwen3.5-9B fixture" and the ROADMAP does hedge "(fixture path)" — honest. But there is **no `docs/model-scope-*-results-*.md` campaign artifact** anywhere in the tree; the Qwen3.5-9B reference is a config constant, and the tests are synthetic fixtures. Status is DONE *against the written exit criteria*, which are themselves fixture-level. Do not later read this row as "steering validated on a 9B model." |
| **10** — SHIM substrate DoD | DONE (#284/#285/#286; honesty correction #289) | **DONE — confirmed merged** | PRs **#284, #285, #286, #289 all MERGED** 2026-06-29 (`efe6d1be`, `49b36046`, `d8181f64`, `d1bfc090`). `docs/rung10-shim-substrate-dod.md`, `steering_route_promotion.py`, `production_steering_control.py` present on `origin/main`. | **None.** The row already self-limits ("diagnostics observation-only", "not a claim that SHIM-CD-01/02/06 are CLOSED"). That caveat is load-bearing and correctly stated. |
| **11** — Annealing controller | DONE (#260 engine controller + #280 post-bank schedule) | **DONE — confirmed merged, both exit clauses verified** | PR **#260 MERGED** (`f0c643ae`), PR **#280 MERGED** (`9db098ea`). `annealing_controller.py` + `annealing_schedule.py` on `origin/main`. Engine wiring: `antigravity_engine.py:824-832` `enable_annealing_controller()`. High-T vs low-T proof: `test_annealing_schedule.py:24-25` (explore 1.0 → stabilize 0.0), `:80-81` (no drift → 0.0, ref drift → 1.0). | **None.** The "schedule ownership split across two modules" caveat is accurate and matches the two-file reality. |
| **12** — Evidence DAG schema | DONE (#277 `evidence_dag.py` + validator + JSON schema; no GNN) | **DONE — confirmed merged** | PR **#277 MERGED** (`81e3614b`). `evidence_dag.py` on `origin/main`: `validate_evidence_dag()` at `:182`, `EVIDENCE_DAG_JSON_SCHEMA` at `:221`. `build_attribution_pool.py` + `test_evidence_dag.py` present. | **Cosmetic only.** "JSON schema" is a Python dict constant (`EVIDENCE_DAG_JSON_SCHEMA`), not a standalone `.json` file on disk. It is a real, machine-usable JSON-Schema document; the claim is not false, but a reader expecting a `.json` artifact will not find one. |
| **13** — Disintegration loop | DONE — detector-driven Evidence-DAG edge prune (rung-13 PR) | **DONE in code, but UNMERGED and CI-UNVALIDATED** | Commit `77de3283` (+ docs `98e9dec4`) — **PR #293 OPEN**, base `lattice/phase2-continue-20260713`, `mergeStateStatus: CLEAN`. New: `evidence_dag_disintegration.py` (300 L), `EvidenceDAG.prune_edges`/`reanneal` (+130 L), `test_evidence_dag_disintegration.py` (288 L), `docs/rung13-disintegration-loop.md`. Real detector wiring confirmed (`IsomerDetector` strength → `fitness=1-strength`; `ConvergenceMonitor.get_summary` → patience ratio). 27 rung-13 tests pass locally. | **Overstatement of merge state.** ROADMAP line 43 and the row read as settled fact; the code is **not on `main`**. Worse: **PR #293 has run NO repo CI** — `gh pr checks 293` returns only `GitGuardian Security Checks`. Cause verified: `.github/workflows/test.yml` triggers on `pull_request: branches: [main]` only, and #293 targets an intermediate branch. So lint / py3.9–3.12 matrix / Rule-5 smoke / §4 PR-body validator / §6.3 block-flag gate / v3.3 drift validator **have never run on rung 13**. Only chair-run local smoke exists. Disclosed `DEFERRED_SCOPE`: no engine-side auto-prune cadence (opt-in only). |
| **14** — Concept-drift experiment | DONE apparatus; H5 living-bank VERDICT: FAIL / non-promoted | **Apparatus DONE and merged. VERDICT ARTIFACTS ARE UNMERGED.** | Merged on `origin/main`: #258–#267, #268–#276, #287, #288, #290 (H5 driver), #291 (H4 `compound_cycles`). But `docs/drift-recovery-post-bank-headtohead-results-2026-06.md`, `...-nfcorpus-results-2026-06.md`, and `docs/drift-recovery-h4-compound-cycles-ablation-2026-07.md` exist **only on HEAD** (commit `7e6c9118`), i.e. inside **PR #292 — OPEN and `mergeStateStatus: BLOCKED`**. | **Real discrepancy.** The *capability* to run H4/H5 is merged; the *verdict* the ROADMAP cites (LIVING BANK WINS=False, SciFact+NFCorpus; H4 ~45× collapse) has **no evidence artifact on `main`**. Anyone doing a fresh `main` checkout cannot reproduce or even read the verdict. #292 CI is fully green (all 12 checks pass) — it is blocked by branch protection, not by failure. |
| **15** — GNN prototype | OPEN (no merged PR; no PyG/DGL code — only a docstring forward-ref) | **OPEN — confirmed, and confirmed on HEAD too** | `git grep -l -iE "torch_geometric\|import dgl\|GCNConv\|MessagePassing"` returns **empty on both `origin/main` and `HEAD`**. Only forward-references: `evidence_dag.py:5-6` ("There is NO GNN here and no learning"), `steering_post_bank.py:18`. No PR exists. | **None — this row is honest.** It is the single genuinely-open executable feature. Note its stated dependency `13 + 14 → 15` is satisfied only by *unmerged* work. |
| **16** — Quant-aware shim routing | DONE locally / NON-PROMOTED; both arenas FAIL-CLOSED; "change set/evidence still needs durable publication" | **DONE in code as an honest non-promotion — UNMERGED, CI-UNVALIDATED; and one ROADMAP number is STALE** | Commits `5a3fbcc2` + `2c78195d` — **PR #295 OPEN**, base `lattice/rung17-diskpool-20260714`, `CLEAN`. New: `quant_aware_routing.py` (1123 L), `run_quant_aware_routing_campaign.py` (615 L), `test_quant_aware_routing.py` + `test_adapter_router.py` (24 tests, pass locally), `adapter_router.py` (+185), `antigravity_engine.py` (+170), prereg `prereg_rung16.md/.json`, results doc + 55 606-line manifest + per-arena selection locks. Engine seam is **default-OFF** (verified: `antigravity_engine.py:164` `self._quant_aware_routing_plane = None`; opt-in `enable_quant_aware_routing()` at `:1198`). | **Two issues. (a) Stale Arena-A numbers in the ROADMAP row.** The table (line 72) states Arena A "SELECT delta 0.000000, CI [0.000000, 0.000000], quant pass 0.25" and Arena B "delta -0.000899". The commit message for `5a3fbcc2` reports Arena A as **-0.003332, CI [-0.051022, 0.041025]** and Arena B **-0.000899**; the session brief quotes -0.0033 / -0.0074. The ROADMAP's all-zeros Arena-A row does not match the campaign record and must be reconciled against `docs/rung16-quant-aware-routing-results-2026-07.md` before publication. **(b) The ROADMAP row omits the Tier-B correction entirely.** Commit `2c78195d` (Grok Tier B, severity *important*) established that Arena B's home-route purity was FiQA 9/30, NFCorpus 1/30, SciFact 11/30 — home-correct routing **HELPED** (n=21, **+0.0257**) and cross-domain misroutes (n=39, **-0.0309**) drove the loss. Arena B therefore falsified **the preregistered centroid-margin plane**, not domain routing in general. The ROADMAP still reads as blanket evidence against routing — an **overstatement of the negative**. Also: no repo CI has run (PR #295 → only GitGuardian, same intermediate-base cause as #293). Artifact-ordering caveat is self-disclosed in the commit and must survive into any publication. |
| **17** — Disk pool slice | DONE — block-graph pool-shard read with host parity (rung-17 PR) | **DONE in code, but UNMERGED and CI-UNVALIDATED** | Commits `e12b7668`, `d8ccf70c`, `12aa1be6` — **PR #294 OPEN**, base `lattice/rung13-disintegration-20260714`, `CLEAN`. New: `computational_storage_poc/pool_shard.py` (306 L), `test_pool_shard_parity.py` (103 L), `docs/rung17-disk-pool-slice.md`. Real traversal confirmed (`_decode_payload_via_block_graph` → `read_block`, next_offset link-walk, not an npz reload). 4 parity tests pass locally. | **Overstatement of merge state** (same class as 13/16): not on `main`, **no repo CI** (PR #294 → GitGuardian only). The row's own honesty is otherwise good: it discloses the 4× FP16 byte-lane storage overhead and `EXPERIMENTAL` / not-wired-into-live-retrieval status. |

---

## Cross-cutting findings

### F1 — The single biggest docs-truth problem: **nothing from this session is on `main`.**
`origin/main` is still at `34ce4b56` (#291, 2026-06-29). Four PRs are stacked and open:

```
main (34ce4b56)
 └─ #292  lattice/phase2-continue-20260713   OPEN, BLOCKED (branch protection), CI 12/12 GREEN
     └─ #293  rung13   OPEN, CLEAN, CI: GitGuardian only
         └─ #294  rung17   OPEN, CLEAN, CI: GitGuardian only
             └─ #295  rung16   OPEN, CLEAN, CI: GitGuardian only
```

The ROADMAP's own "Status snapshot (2026-07, **git-verified**)" header is itself part of `c6a698e4`,
which is **unmerged inside #292**. A fresh `main` checkout shows rungs 13/16/17 as not existing.

**Phase II cannot be truthfully declared closed while the closure evidence is unmerged.**

### F2 — Rungs 13, 16, 17 have never been through repo CI.
Verified root cause: `.github/workflows/test.yml` lines 3–7 trigger only on `push`/`pull_request` to
`main`. Stacked PRs targeting intermediate branches get **no lint, no py3.9–3.12 matrix, no Rule-5
smoke gate, no §4 PR-body validator, no §6.3 block-flag gate, no v3.3 schema-drift validator**.
The BHS merge gate is therefore *unexercised* on three of the four claimed-DONE session rungs.
Local chair-runs (57 tests OK, this audit) are real evidence but are **floor-tier and single-host**;
they do not substitute for the py3.9 compatibility leg, which historically breaks this repo
(`X | None` annotations — see CLAUDE.md). Note `quant_aware_routing.py`, `evidence_dag_disintegration.py`,
and `pool_shard.py` are all new modules that have never been imported under 3.9 in CI.

### F3 — ROADMAP **overstates** (must be corrected before closure):
1. Rungs **13, 16, 17** presented as settled DONE without stating "unmerged / no CI".
2. Rung **14**'s H5/H4 verdicts cited as established while their evidence docs live only in blocked #292.
3. Rung **16**'s Arena-A numbers (`delta 0.000000, CI [0.000000, 0.000000]`) contradict the campaign
   record in commit `5a3fbcc2` (`-0.003332`, CI `[-0.051022, 0.041025]`).
4. Rung **16** framed as evidence that routing loses on its home turf, **after** the Tier-B correction
   established the binding constraint was route *assignment* (65% misroute), not specialist *capacity*.
   This is the highest-value scientific nuance in the whole session and it is missing from the table.

### F4 — ROADMAP **understates** (worth crediting, but do not inflate):
1. Rung **13** is stronger than the row implies: transactional prune/re-anneal, a scorer-mutation
   `RuntimeError` guard, and fail-closed neutral-fitness on missing/unmatched/immature signals —
   absent evidence can never cause a destructive prune.
2. Rung **17**'s byte-lane encoding is genuinely bit-exact through an FP16-native format and the read
   path is proven to be real `read_block` traversal, not an array reload.
3. Rung **16**'s de-rigging (three-way ANCHOR/SELECT/REPORT split, single-global primary baseline,
   plane-level paired bootstrap, per-route "lift>0" promotion *banned*) is methodologically the
   strongest gate the program has built. The apparatus correctly declined to promote. That is a win.

### F5 — Consistency with the program's six fail-closeds.
Rung 16 is fail-closed #6 and fits the pattern (elaborate structure loses to a trivial global
baseline) — **with the caveat in F3.4**. Rung 14/H5 is fail-closed #4/#5. Rungs 9–13, 17 are
apparatus, not hypotheses, so they carry no promotion claim. Nothing in this audit found a
fabricated positive result; the failure mode in this repo is **premature merge-state claiming**,
not fabricated numbers.

---

## Verdict on "can we declare Phase II closed?"

**Not yet — three blocking conditions, all mechanical:**

| # | Blocker | Action |
|---|---|---|
| B1 | 4 PRs unmerged (#292 blocked by branch protection; #293/#294/#295 stacked behind it) | Admin-merge #292 (CI 12/12 green), then re-target or land #293 → #294 → #295. |
| B2 | Rungs 13/16/17 have zero repo CI | Re-base each onto `main` so `test.yml` fires, **or** land the stack and confirm the post-merge `push: main` run is green. Do not declare closure on chair-runs alone. |
| B3 | Rung-16 Arena-A numbers in the ROADMAP contradict the campaign record; Tier-B routing nuance absent | Reconcile line 72 against `docs/rung16-quant-aware-routing-results-2026-07.md` and add the purity/attribution correction from `2c78195d`. |

**After B1–B3:** Phase II closes with rungs **9–14, 16, 17 DONE** (16 as an explicit
non-promotion) and **15 (GNN) the sole OPEN item** — and "15 OPEN" is the honest thing to say,
not something to paper over. Given the program's six-for-six record of elaborate structure losing to
trivial linear baselines, and given that a GNN over the Evidence DAG is the *most* elaborate
structure yet proposed, rung 15 should not be started without the power math first — per the
standing lesson: **derive the win bar from the achievable n, and if the bar is not clearable, CUT.**

*No numbers in this document were estimated or inferred. Every figure is quoted from a git commit
message, a `gh` API response, or a file/line in the tree, each cited inline.*
