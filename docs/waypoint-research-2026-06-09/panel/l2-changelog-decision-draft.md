# L2 draft: H5 living-bank verdict + H4 ablation

**Status:** chair applies; draft to panel only. Not applied to `CHANGELOG.md` or `docs/next-session.md` by this artifact.

**Sources (numbers only from these frozen docs):**

- `docs/drift-recovery-post-bank-headtohead-results-2026-06.md` (SciFact H5)
- `docs/drift-recovery-post-bank-headtohead-nfcorpus-results-2026-06.md` (NFCorpus H5)
- `docs/drift-recovery-h4-compound-cycles-ablation-2026-07.md` (H4)

---

## Block A — paste under `CHANGELOG.md` → `## [Unreleased]`

```markdown
### Findings (honest status) — H5 post-bank head-to-head + H4 compound-cycles ablation

- **H5 living post-bank (preregistered gate: C5 must beat BOTH C5s and C5r; LIVING BANK WINS = both):** failed on both datasets. Frozen sources: `docs/drift-recovery-post-bank-headtohead-results-2026-06.md` (SciFact), `docs/drift-recovery-post-bank-headtohead-nfcorpus-results-2026-06.md` (NFCorpus). Arena `query_encoder_swap`, cycles 12, seeds [42, 1337, 7]; mean Final NDCG over seeds.
  - **SciFact:** C5 living Final NDCG **0.131135** == C5s frozen static **0.131135** (bit-identical means; living beats static: **False**). C5r one-shot Final NDCG **0.180862** beats living (living beats one-shot: **False**). **LIVING BANK WINS (beats both): False.**
  - **NFCorpus:** C5 living Final NDCG **0.046389** == C5s static **0.046389** (living beats static: **False**). Living edges C5r one-shot **0.045817** (living beats one-shot: **True**) but does not clear the both-comparators gate. **LIVING BANK WINS (beats both): False.**
  - **Honest negative:** the living/annealed bank lifecycle adds nothing over a frozen static bank (C5 ties C5s on both tasks). A one-shot router is competitive-or-better (SciFact C5r strictly above C5; NFCorpus C5 only narrowly above C5r while still tied with C5s). Do not promote the living bank on this evidence.
- **H4 compound-cycles ablation (single-seed):** SciFact C4a, seed **42**, cycles 12 — `docs/drift-recovery-h4-compound-cycles-ablation-2026-07.md`. `compound_cycles=False` (idempotent fixed point) Final NDCG@10 **0.236297**; `compound_cycles=True` (compounding) **0.005258** — recovery collapses ~**45×** (0.236 → 0.005), near the frozen floor. Compounding is catastrophic on this single-seed ablation; direction is unambiguous, exact magnitude is one seed.
```

---

## Block B — paste for `docs/next-session.md` (decision / disposition; not a new Carried Debt row)

```markdown
### Disposition — living / annealed post-bank corrector (H5; rung-13/14 evidence)

**NON-PROMOTED.** Park the living-bank / annealed-post-bank corrector line per its own preregistered H5 gate: C5 must beat **both** C5s (frozen static bank) and C5r (one-shot router). Frozen campaign means:

| Dataset | C5 living | C5s static | C5r one-shot | C5 > C5s | C5 > C5r | LIVING BANK WINS |
|---|---:|---:|---:|---|---|---|
| SciFact | 0.131135 | 0.131135 | 0.180862 | False | False | **False** |
| NFCorpus | 0.046389 | 0.046389 | 0.045817 | False | True | **False** |

Sources: `docs/drift-recovery-post-bank-headtohead-results-2026-06.md`, `docs/drift-recovery-post-bank-headtohead-nfcorpus-results-2026-06.md`. On SciFact, living beat neither comparator (tied static; one-shot higher). On NFCorpus, living beat only one-shot by a small margin and still tied static — gate still **False**.

**If any post-bank is kept at all:** the frozen static bank (**C5s**) is the honest baseline; the living/annealed lifecycle is not justified over C5s on this evidence. No new Carried Debt or Deferred-Scope row is required for this disposition — the line is closed as non-promoted research outcome, not an open implementation obligation. Adjacent H4 (single-seed SciFact C4a seed 42): `compound_cycles=True` **0.005258** vs `False` **0.236297** (~45× collapse) — compounding remains rejected design, not a follow-on promotion path (`docs/drift-recovery-h4-compound-cycles-ablation-2026-07.md`).
```

---

## Chair notes (not paste)

- **No Carried Debt / Deferred-Scope row proposed:** gate failed cleanly; disposition is park/NON-PROMOTED, not an unpaid implementation debt.
- **Hard bans observed:** no “wins” / “promising” / “validated” for the living bank; no ceiling (C2O) claims; H4 labeled **single-seed**.
- **Rounded display vs full means in sources:** tables use 6 dp as in frozen head-to-head tables (e.g. SciFact C5/C5s 0.131135, C5r 0.180862; NFCorpus 0.046389 / 0.045817); full floats remain in the SciFact/NFCorpus verdict bullets if chair wants bit-identical paste.
