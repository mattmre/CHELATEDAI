# DRAFT — `docs/next-session.md` update for the endgame round (2026-07-19)

**Status:** draft for chair review. Nothing here is applied to `docs/next-session.md` yet.
**Schema discipline:** no headings renamed, no tables collapsed, no existing rows edited or deleted.
This draft only (a) **appends** rows to the existing `## Carried Debt` table, (b) **replaces the
`_none yet_` placeholder** in the existing `## Deferred Scope` table, (c) **adds one new
`## Disposition — …` section** (same pattern as the two that already exist), and (d) updates the two
footer lines.

**Block flag: verified CLEAR.** See §5 of this draft for the command output and the honest caveat
about the advisory WARNING the appended rows will trigger.

---

## 1. Carried Debt — rows this round genuinely opens

Append these four rows to the end of the existing `## Carried Debt` table body (after `CD-H1-01`).
Existing rows are untouched.

```markdown
| CD-LAT-01 | The entire endgame evidence base is unlanded: a four-deep stacked PR chain `#295 → #294 → #293 → #292 → main` (rung 16 → rung 17 → rung 13 → phase-2 verdicts). Every PR is opened with a BHS body and green required checks, but `#292` is `mergeStateStatus: BLOCKED` / `reviewDecision: REVIEW_REQUIRED` (branch protection), so nothing below it can merge. Until the stack lands, `docs/ROADMAP_EXECUTION.md` rows marking rungs 13 / 16 / 17 **DONE** describe code that does not exist on the default branch. | endgame round 2026-07-19 | 1 cycle | NO — Session Rule 2 is satisfied (each slice is opened and green); the gap is a branch-protection admin merge, and the ROADMAP row for 16 already says "on their feature branches". Escalate to YES if the stack survives a second cycle, because at that point the DONE claims become a Rule-2-of-the-rulebook "visible means verified" violation. | OPEN — needs admin merge of `#292`, then `#293`, `#294`, `#295` in order. Re-verify each PR's checks after each base-branch retarget. |
| CD-PR-01 | Three long-lived PRs remain open against `main` with no landing path: `#278` "docs(brain): file-map dossiers B0+B1" (BLOCKED, opened 2026-06-29), `#257` "feat(progress): live progress branch" (DRAFT + **DIRTY**, i.e. conflicted, opened 2026-06-06), `#256` "feat(unblock): SHIM-CD-01 first thin SIP probe design" (DRAFT + BLOCKED, opened 2026-05-28). Unlanded PRs are debt per Session Rule 2 regardless of how honest their contents are. | endgame round 2026-07-19 (stale-PR triage) | 1 cycle | NO — none is on a production code path; `#257`/`#256` are drafts | OPEN — this round's triage produced a per-PR CLOSE / REWORK / MERGE verdict for each. Row closes when all three are dispositioned in the repo (closed with a superseded-by note, or reworked and merged). Any PR verdicted REWORK converts into its own successor row with the rework scope named. |
| CD-R16-01 | The rung-16 opt-in serving surface has never run against a real engine or a real promoted plane. All seven `enable_quant_aware_routing` call sites in `test_quant_aware_routing.py` construct the engine as `object.__new__(AntigravityEngine)` with hand-set attributes (`logger`, `vector_size`, `mode`, `_runtime_telemetry`, …) — the constructor, embedding backend and Qdrant store are bypassed. `test_engine_runtime_uses_provider_vector_and_route_document_view` does drive `run_inference()` end-to-end, but on a synthetic 4-dimensional fixture plane. Separately, **both preregistered arenas FAIL-CLOSED, so no `PROMOTED` plane exists anywhere in the repo** — the success branch of the enable gate and the live serving path have only ever seen in-test planes. L5 (untested production path). | rung 16 (`5a3fbcc2`/`2c78195d`, PR `#295`) | 1 cycle | NO — the plane is opt-in, default OFF, and `enable_quant_aware_routing` raises on any non-`PROMOTED` verdict, so the untested branch is unreachable in production by construction | OPEN — closes when either (a) a campaign produces a genuinely `PROMOTED` plane and the serving path is exercised on a real `AntigravityEngine`, or (b) an integration test builds a real in-memory-Qdrant engine and drives a fixture-promoted plane through `run_inference()`. Do **not** close this by relaxing the promotion gate. |
| CD-R16-02 | Frozen-artifact ↔ code reason-code drift: the frozen rung-16 evidence (`docs/rung16-quant-aware-routing-manifest-2026-07.json`, both `rung16-arena_*-selection-lock-2026-07.json`) records the quant-gate reason `one_or_more_used_adapters_failed_quant_survival`, while current `quant_aware_routing.py:681` emits `one_or_more_retained_adapters_failed_quant_survival` (the post-campaign hardening broadened the gate from *used* to *retained* adapters). No metric, verdict, query ID or decision differs — this is a string rename only. The Tier B reviewer flagged it cosmetic and explicitly declined to reopen it ("still present as lineage note only"), so it is recorded here rather than silently dropped. v3.3 schema/prose/artifact drift class. | rung-16 Tier B (grok-4.5, 2026-07-19) | 1 cycle | NO — cosmetic; verdict arithmetic unaffected | OPEN — one-line fix: name the rename in the "Artifact note" paragraph of `docs/rung16-quant-aware-routing-results-2026-07.md` so a future replayer diffing reason codes against the frozen locks is not misled. REPORT must **not** be rerun to normalize the string. |
```

### Rows deliberately NOT opened (and why)

Being strict here matters more than being thorough — inflating the table devalues it.

| Candidate | Verdict | Reason |
|---|---|---|
| Rung-16 FAIL-CLOSED on both arenas | **Not debt** | A closed research question with a preregistered gate and a published negative. The gate worked. Debt would be the wrong signal entirely. |
| Arena B domain-purity collapse (65% misroute) | **Not debt** | Measured, recomputed by Tier B, and now disclosed in the results doc. The *follow-up ablation* is Deferred Scope (§2), not debt. |
| Rung-16 lineage limitation (manifest carries no contemporaneous code/Git hash; runner hardened after the run) | **Not debt** | Permanently unfixable without rerunning a consumed one-shot REPORT, which the design forbids. It is disclosed in the results doc's opening Artifact note. Recording it as debt would imply a fix exists. |
| HI-1 harmonic-invariance preregistration CUT before GPU | **Not debt** | A design killed before it consumed resources. That is the process working. Noted in the disposition (§3). |
| Rung-17 `array_equal` NaN false-negative (Tier B "low") | **Not debt** | Already fixed on the branch by `d8ccf70c` ("rung 17 parity uses byte-equality"). |
| Rung-13 residual notes (length-based mutation guard; `mode is None` fallback) | **Not debt** | Tier B scored 100 and explicitly recorded both as "not scoring gates" / "not a reopened named gap". Production detectors always set `mode`. |
| Rung-17 pool shard not wired into live retrieval | **Not debt** | Explicitly POC/EXPERIMENTAL-scoped in the rung-17 exit criteria and the ROADMAP row. Bounded scope, not a gap in a claimed capability. |

---

## 2. Deferred Scope

Replace the `_none yet_` placeholder row in the existing `## Deferred Scope` table with these three
rows. (Heading and column schema unchanged.)

```markdown
| DS-15-01 | **Rung 15 — GNN prototype over the Evidence DAG: CUT.** Phase II closes at 9–14 + 16 + 17 with rung 15 documented as a cut, not as abandoned or open. | endgame round 2026-07-19 (rung-15 CUT decision) | The endgame plan budgeted rung 15 as a coin-flip "leaning negative" before any work started. Two independent grounds now make it a cut rather than a run: (i) the same power problem that killed HI-1 — the drift fixture's held-out edge/cluster count cannot support a lower-bound-clears-zero comparison against the flat-pool baseline at any bar worth preregistering; (ii) the deep-research pass found the graph-learning cell dominated for this question. A seventh pre-narrated fail-closed has near-zero decision value. Reopen only with a fixture large enough to derive a clearable bar from power math FIRST. |
| DS-WEDGE-01 | **Within-space superposition/compressed-sensing wedge test — designed this round, NOT run.** A gradient-free nonlinear (ℓ1/CS-style) readout vs the strongest linear probe, on known superposed features inside ONE frozen embedding space. | endgame round 2026-07-19 (wedge design + power analysis) | This is the one genuinely unpursued cell the deep-research pass identified: the wedge itself is proven (Garg–Kleinberg–Peng 2026, arXiv:2602.11246 — quadratic gap, matching lower bound; corroborated by Engels et al., ICLR 2025, arXiv:2405.14860), but every demonstrated exploiter is gradient-trained, and sparse dictionary learning is provably non-identifiable (arXiv:2512.05534). No GPU was spent this round: per the HI-1 lesson, the **power math is derived first and the bar is derived from it**. The test is only queued if the power analysis names an operating point where the bar is clearable at achievable n; if it does not, the honest outcome is CUT, and that CUT is the deliverable. Do not carry this forward as an obligation to run. |
| DS-R16-01 | **Rung-16 oracle / domain-label routing ablation** (route by ground-truth domain label instead of centroid margin), plus the prerequisite question of whether domain-separable centroids can survive an encoder swap at all. | rung 16 (PR `#295`, DEFERRED_SCOPE amended per Tier B) | Arena B falsified the *preregistered centroid-margin* plane, not domain-specialized routing in general: home-correct specialist routes beat single-global (n=21, mean Δ **+0.0257**) while cross-domain misroutes (n=39, mean Δ **−0.0309**) dominated the −0.007383 total, at serve-time home-route purity FiQA 30.0% / NFCorpus 3.3% / SciFact 36.7%. The binding constraint measured was **route assignment under encoder-swap drift, not specialist capacity**. The ablation that would separate those two cannot be bolted onto a consumed one-shot REPORT — it needs its own preregistration and its own power math. Also carried forward from the original DEFERRED_SCOPE: the anti-aligned residual-drift regime. |
```

---

## 3. New disposition section

Insert after the existing `## Disposition — Rung 16 quant-aware routing plane` section and before
`## Aggregate BHS trend`.

```markdown
## Disposition — endgame round (2026-07-19)

**Phase II closed; the round's product is three gates that held and two designs killed before GPU.**

Landed as PRs (stacked, awaiting admin merge of `#292` — see CD-LAT-01): rung 13 disintegration loop
(`#293`, Tier B 100), rung 17 disk-pool slice (`#294`, Tier B 100), rung 16 quant-aware routing plane
(`#295`, Tier B 100 after the Arena B purity correction). Rung 16's own verdict is FAIL-CLOSED on both
arenas — the gate is the deliverable, not the lift.

Two designs were CUT before consuming GPU, and both cuts are counted as wins:

- **HI-1** (gradient-free harmonic operator vs ridge in OOD regions of a frozen cloud) — killed by
  design red-team on four independent grounds: underpowered by construction (realistic n ≈ 9–27
  UNKNOWN queries; paired 95% half-width ≈ 0.08–0.14, so clearing a +0.01 lower bound required a true
  effect of ~0.09–0.15 absolute NDCG — larger than rung-16's entire effect); operator-starved (the
  UNKNOWN stratum was *defined* as low-density, and Nyström extension into sparse regions is the
  textbook spectral failure mode); baseline too weak (best-of-ridge/Procrustes is weaker than the
  leakage-safe full-fit affine ridge this repo already ships, while the treatment carried six knobs to
  the baseline's one λ); and unfalsifiable as written (both outcomes were pre-narrated, so decision
  value ≈ 0). The last of those is a **chair design error**, not a reviewer catch to be proud of.
- **Rung 15** (GNN prototype) — see DS-15-01.

**The governing lesson, recorded so it binds the next round:** do the power math FIRST and derive the
win bar from it. If the bar is not clearable at the achievable n, the honest answer is CUT. Never
propose a bar and hope n cooperates. This program has now produced six consecutive fail-closeds in
which elaborate structure lost to a trivial global linear baseline (drift corrector beaten ~4.2×,
Holm p<0.001; recoverability estimator inverted from +0.886 to −0.706 between 6 and 12 cells and lost
to a gap-only null; sparse-local home-turf preflight admitted no cell out of 48; H5 living bank tied a
frozen static bank bit-identically and lost to a one-shot router; H4 compounding collapsed recovery
~45×; rung-16 routing plane fail-closed on both arenas). A seventh, produced by a design already known
to be underpowered, would teach nothing.

Not-yet-open questions are parked in Deferred Scope, not in Carried Debt. A closed research question
is not debt.
```

---

## 4. Footer line updates

Replace the two closing lines:

```markdown
**Last session**: 2026-07-19 — endgame round. Rungs 13 / 16 / 17 implemented, each fresh-Grok Tier B
at 100, stacked as PRs `#293`/`#294`/`#295` behind `#292` (admin merge pending — CD-LAT-01). Rung 16
FAIL-CLOSED on both arenas (no plane enabled); Arena B's loss traced to route assignment, not
specialist capacity. HI-1 CUT before GPU on power + falsifiability grounds; rung 15 CUT, Phase II
closed. Four new Carried Debt rows opened (CD-LAT-01, CD-PR-01, CD-R16-01, CD-R16-02); three Deferred
Scope rows recorded. Prior: 2026-07-14 — rung 16 run once on CUDA, evidence then unpublished.
**Last validated by `check_block_flag.py`**: 2026-07-19 — `CLEAR`, exit 0 (see §5 of the round draft;
the 4 new first-cycle rows trigger the script's advisory CLEAR-with-open-rows WARNING, which is
expected and does not change the exit code).
```

Leave the `**2026-05-17**` line as-is.

**Aggregate BHS trend / Operator overrides log:** no change proposed. The three endgame PRs each
scored `BHS_OFFICIAL = 100` with no override, so the overrides log correctly gains no row. The trend
table's `_current_` row should be filled by whoever performs the merge, once the stack actually lands
— filling it now would credit merges that have not happened.

---

## 5. Block-flag verification

Run in the worktree `D:\GITHUB\CHELATEDAI\.claude\worktrees\relaxed-wozniak-271e04`:

```text
$ python scripts/check_block_flag.py
======================================================================
Brutal Honesty Rulebook v3.3 — §6.3 block-flag gate
File: docs\next-session.md
======================================================================
Block flag state: CLEAR
Carried Debt row count: 0
RESULT: PASS — block flag CLEAR. New feature work is allowed.
EXIT=0
```

**The block flag stays `CLEAR`, and no edit to the `## Block flag` section is proposed.**

Honest caveat, stated up front so nobody is surprised in CI: once the four rows in §1 are appended,
`count_carried_debt_rows` reports **4** instead of 0 and the script additionally prints an advisory
warning. This was **verified by simulation**, not reasoned about — the four rows were appended to a
scratch copy and the gate re-run against it:

```text
$ python scripts/check_block_flag.py --file <scratch>/next-session-simulated.md
Block flag state: CLEAR
Carried Debt row count: 4
WARNING: flag is CLEAR but 4 OPEN Carried Debt row(s) remain. Per §6.3, first-cycle items keep the
flag CLEAR; items surviving a full cycle (TTL=0) flip the flag to BLOCKED. [...]
RESULT: PASS — block flag CLEAR. New feature work is allowed.
EXIT=0
```

Per rulebook §6.3 and the script's own docstring, **first-cycle items legitimately keep the flag
CLEAR** — the script cannot determine cycle age and the warning does not affect the exit code, which
remains **0 / PASS**. All four rows are opened this cycle with `TTL = 1 cycle`, so the flag is
honestly CLEAR right now.

**The trap to avoid next cycle:** if any of these four rows is still OPEN at the next session
boundary, the session-wrap step must flip `**Current**:` to `BLOCKED` — the script will not do it, and
leaving it CLEAR would be exactly the "flag is lying" failure the row-count print exists to expose.
CD-LAT-01 is the likely offender, since it depends on an operator admin merge rather than on any work
this program controls.
```
