# Codex build task — apply Grok's C1 adversarial-review fixes to the D1 package

You built the D1 paired-bootstrap package at `research/drift_recovery/`. A fresh adversarial
reviewer (Grok, C1) validated it and returned **PASS-WITH-FIXES**. Numbers recompute exactly;
harness parity is real; 4/4 tests pass. But there are ship blockers on protocol honesty. Apply the
fixes below. Work ONLY inside `research/drift_recovery/`. **Do NOT touch any file under
`docs/waypoint-research-2026-06-09/paper-draft/`** — the human chair applies the paper prose edits.

Full C1 review text: `../../docs/waypoint-research-2026-06-09/panel/out_C1_grok.txt`
(relative to the worktree root it is `docs/waypoint-research-2026-06-09/panel/out_C1_grok.txt`).

## Fixes to apply (regenerate all machine artifacts; NO hand-entered numbers)

### F1 — BLOCKER: leakage magnitude must be first-class, not a soft sentence
The canonical ladder/bootstrap uses the literal waypoint half-corpus permutation which contains
28 eval-positive docs (45% of the 62 eval-positives). The 84.3% headline is therefore a leaky-fit
point estimate. Grok measured leakage-safe full-600 fits at ridge ~78.9% / procrustes ~73.9%
(+5.4pp / +8.4pp inflation).
- In `d1/paper_stats.py`, compute BOTH fits from the frozen pack: the literal-waypoint fit (current
  headline) AND a leakage-safe full-600 fit (eval-positive docs excluded from the fit set) for
  ridge, procrustes, mlp. Emit a **leakage sensitivity table** into `ci_report.md` and a dedicated
  leakage section into `paper_edits.md` stating: N eval-positive docs in the literal fit, the
  leakage-safe recovery for each method, and the per-method inflation in pp. Numbers must be
  code-computed, not copied from the review.
- Verify the "28 eval-positive docs / 62 total" audit counts programmatically from the pack
  (`ladder.json` `waypoint_audit` or recompute) and assert them in a test (see F7).

### F2 — BLOCKER: ban the "ridge≈MLP ⇒ linear ceiling" inference in paper_edits.md
`paper_edits.md` currently only does noun-phrase replacement. Add an explicit, location-agnostic
**inference-ban block** with this exact prohibition language for the chair to paste:
> Do not interpret ridge−MLP non-significance as evidence of a linear post-hoc ceiling or the
> absence of nonlinear benefit. The 95% CI on ΔNDCG(ridge−MLP) is [−0.0333, 0.0773] — it permits
> MLP better by up to ~4 recovery points and ridge better by up to ~10. G2 FAILED (median primary
> half-width 0.089 ≫ 0.015), so this study is underpowered for the equality contrast on 60 queries.
> Non-significance here is absence of evidence, not evidence of absence.
Also add the forbidden-claims list: no "linear post-hoc ceiling", no "no nonlinear benefit / capacity
is linear", no "84% is the recoverable upper bound", no "irreducible 16%", no G2-powered precision on
method contrasts. Permitted claims: point estimates + 95% CIs on the stated protocol; ridge is the
highest-observed here; ridge ≫ C3a (Holm-significant); ridge vs MLP not distinguishable at α=0.05
under this n; prefer "observed plateau under this protocol".

### F5 — Medium: fix false "within tolerance" on C3a
`paper_stats.py:182-188` — C3a canonical-tolerance check reports "within tolerance" but the actual
error is 0.0056 > the 0.001 threshold. Either enforce the tolerance (fail loudly) or drop the
"within tolerance" wording and state the supersede narrative honestly (old 0.1609 was cross-seed
mean; seed-42 per-query is 0.1570). Do not print a false pass.

### F6 — Medium: G2 primary half-width = ridge−MLP, not median of both contrasts
`paper_stats.py:300-309` currently takes the median half-width across {ridge−MLP, ridge−C3a}. The
preregistered ceiling contrast is ridge−MLP. Report the G2 primary half-width as ridge−MLP's alone
(0.055) AND keep the median as a secondary diagnostic. G2 still FAILS either way vs 0.015; make the
primary explicit and add the rough n≈800+ scale-up note to `ci_report.md`.

### F4 — High: name the ladder Procrustes correctly
In `section3_draft.md` (and any caption text the generator emits), the D1 ladder "procrustes" is a
closed-form SVD orthogonal Procrustes map — it is NOT the trainable Cayley adapter described in §3.1
of the paper. Label it "closed-form orthogonal Procrustes" wherever the ladder baseline is named so
the two objects are not conflated.

### F7 — Low: harden tests
In `tests/test_d1_stats.py` add: (a) assert floor/oracle/all-methods share the same bootstrap
indices; (b) an invalid-draw gate test (gap ≤ ε invalidates, >1% hard-fails); (c) assert
`eval_positive_docs_in_literal_fit == 28` and leakage-safe fit == 0 eval-positives; (d) assert
leakage-safe ridge recovery < leaky ridge recovery. All tests must pass.

## Deliverables
1. Updated code under `research/drift_recovery/` (paper_stats.py, tests).
2. Regenerated `research/drift_recovery/out/d1/ci_report.md` (now with leakage sensitivity table +
   G2 primary=ridge−MLP + scale-up note).
3. Regenerated `research/drift_recovery/out/d1/paper_edits.md` (now with the F1 leakage section +
   F2 inference-ban block + forbidden/permitted lists).
4. Run the test suite; paste the pass/fail summary.
5. A short `research/drift_recovery/out/d1/C1_FIX_REPORT.md` mapping each fix F1–F7 to the exact
   file:line changed and the resulting number, so the chair can verify without re-reading code.

Be brutally honest: if any fix cannot be done cleanly, say so with file:line rather than faking a
pass. Do not regress the harness-parity (must stay ≤1e-12) or the 4 existing tests.
