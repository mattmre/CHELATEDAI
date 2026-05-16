# BHS Validator Rubric Scope

**Status**: load-bearing — read this before adding to `scripts/bhs_validator.py`.

**Origin**: CD-245-01 closure, 2026-05-16, after three Tier B iterations on PR #245 (BHS_TIER_B 92 → 96 → 85) converged on the same gap: the validator is described as a "honesty score" but is mechanically a structural check, and operators who know that can pad inputs to score 100 without committed prose.

## What the rubric IS

`scripts/bhs_validator.py` is a **structural** rubric. It scores AEP findings and phase summaries on 0–100 by mechanically checking that:

| Signal | What it catches | What it can't catch |
|---|---|---|
| Required fields present | L4 (missing data) | A field that's present but meaningless |
| Severity ∈ {CRITICAL, HIGH, MEDIUM, LOW} | L4 (uncommitted tier) | A severity that's correctly typed but wrong for the finding |
| Prose field ≥ 12 chars (whitespace-stripped) | Single-char placeholders ("." / "x") | 12 chars of single-character padding ("xxxxxxxxxxxx") |
| Char entropy ≥ 3.0 bits/char | Padded constant-char strings | "foo bar baz" — diverse but meaningless |
| ≥ 3 unique tokens per prose field | "fix fix fix fix" repetition | 3 unrelated tokens that don't describe the finding |
| No single token > 60% of prose field | "fix fix fix bug" dominant-token padding | Mix of fillers under 60% each |
| Evidence regex (file:line / PR# / artifact / sha) somewhere in text | Findings without any traceable pointer | A pointer to an unrelated file |
| No lie-marker keywords (stub / TODO / fake / ...) | Self-confessing scaffolds | Sophisticated euphemisms ("placeholder" → "interim implementation") |

## What the rubric is NOT

The rubric is **NOT a semantic judgement** of finding quality. It cannot tell you whether:

- The prose actually describes the finding.
- The evidence pointer points to the right file.
- The severity is calibrated correctly.
- The recommended fix would actually fix the problem.

These all require human judgement. A motivated operator who knows the rubric can write prose that passes every mechanical signal while saying nothing meaningful. Example:

```python
finding = {
    "id": "F-1",
    "severity": "HIGH",
    "impact": "foo bar baz qux at handler.py:42",
    "recommended_fix": "tweak the thing in module.py:10",
}
# Structural score: 100. Semantic content: zero.
```

`test_bhs_validator.py::test_diverse_but_meaningless_prose_acknowledged_gap_scores_100` asserts this score-100 outcome on purpose, so the gap is visible and any future "we closed it" claim has to actually change the rubric (and update this doc).

## How to bridge the gap honestly

`scripts/audit_findings.py` is an operator-driven sample audit. Weekly (or per AEP cycle, whichever is shorter), the operator:

1. Runs `python scripts/audit_findings.py --input <findings.json> --sample 12`
2. Reads each sampled finding and scores semantic quality 1–10
3. The script writes `artifacts/findings_audit_YYYY-MM-DD.json` with the structural-vs-semantic delta and surfaces any finding where `structural - semantic*10 > 30` (i.e., the rubric over-scored)
4. `artifacts/findings_audit_latest.json` carries the latest audit's timestamp + summary so the AEP closure can surface "Last audit: ..."

If a cycle's audit shows the same structural-passes-but-empty pattern twice, open a Carried Debt entry for rubric drift and consider whether a new mechanical signal would catch it without false-positiving real prose.

## What NOT to add to the rubric

Three things were rejected during CD-245-01 design (2026-05-16, three Tier B iterations + an external research agent's survey):

### 1. LLM-graded scoring

Rejected because:
- Non-deterministic across model versions (same input scores differently next month).
- Requires API key + network in CI (or a local model + GPU).
- Outage coupling: per Session Rule #1, the validator cannot have a "fall back to a stub" path. If the LLM is down, the validator is down, and the merge gate is down.
- Cost: even at Haiku rates, every AEP finding scored at every cycle adds up.
- Gameability shifts, doesn't close: a sophisticated operator could prompt-tune their prose against the grading prompt.

If you are tempted to add LLM grading anyway, open a Carried Debt entry first and get a Tier B reviewer to disprove these objections, not just affirm them.

### 2. Higher entropy / unique-token thresholds

Bumping `_MIN_CHAR_ENTROPY_BITS` to 3.5 or `_MIN_UNIQUE_TOKENS` to 5 would catch slightly more padding patterns, but every increase pulls in false-positives on legitimate short technical strings. The current thresholds (3.0 / 3 / 60%) were tuned against the literal CD-245-01 example and against real AEP findings from prior cycles. Tighten only with evidence that the false-positive rate stays acceptable.

### 3. Embedding-similarity to a known-good corpus

Rejected because:
- Requires curating + maintaining a corpus of "well-scored" findings (early findings define "good" — bootstrapping bias).
- Adds an embedding-model dependency.
- Determinism risk if the embedding model is updated.
- Higher cost than LLM grading for marginal benefit.

## TL;DR

The validator is structural. Human audit is the semantic layer. Don't conflate the two. Don't try to make the validator "smarter" without first running an audit and writing down what the audit caught that the rubric missed.
