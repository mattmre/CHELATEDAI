# Finding: the C2 oracle survives document-side model-swap drift (2026-06-13)

Local-only. Caught during PR-A1 implementation. Contradicts the prior next-experiment and
publishability analyses on one load-bearing point — verified concretely below.

## Claim (prior analyses)
"Model-swap drift — re-embed a fraction of the corpus with all-mpnet-base-v2 via a frozen
DimensionProjection — defeats the C2 re-embed oracle, so C2 becomes an honest baseline."

## Why it is WRONG for document-side drift
C2 is defined as: re-embed affected documents' raw text with the original frozen encoder.
That reproduces the **pre-drift** document vector regardless of how the stored vector was
corrupted, because the text and the original encoder are both still available.

Concrete trace (SciFact, MiniLM store):
- Baseline: `v_d = MiniLM(text_d)`, queries `q = MiniLM(text_q)`; aligned → high NDCG.
- Document-side model-swap drift: `v_d ← normalize(Proj_768→384(mpnet(text_d)))` for a
  fraction of docs. Affected docs now in a different space → NDCG drops. (real drift ✓)
- C2 maintenance: `v_d ← MiniLM(text_d)` = the exact pre-drift vector → store fully
  restored → NDCG returns to baseline. **C2 fully recovers → still an oracle.**
- `oracle_breaker_check = C2_final_NDCG − baseline_NDCG = 0` → PR-A4's own gate would
  (correctly) declare the arena BROKEN.

General statement: **any drift that only corrupts stored document vectors, while document
text and the original encoder remain available, is perfectly invertible by C2.** Switching
the documents to a different model does not change this — C2 does not chase the drifted
vector; it restores the baseline, and that is recovery.

## The arena that DOES defeat the oracle: encoder/query upgrade
Recovery target must NOT be "the original encoder's embedding of available text." That
holds when the **query/eval encoder is upgraded** and cached document vectors stay in the
old space:
- Stored docs unchanged: `v_d = MiniLM(text_d)` (cached; re-embedding all N is the cost we
  avoid).
- Drift: eval queries become `q' = normalize(Proj_768→384(mpnet(text_q)))` — a new space.
- Retrieval `cos(q', v_d)` misaligned → NDCG drops. (real drift ✓)
- C2 (re-embed docs with original MiniLM): docs unchanged, still misaligned with `q'` →
  **does NOT recover.** Oracle defeated ✓.
- C3 closed loop: learn a bounded adapter `A` so `cos(q', A(v_d))` is high for relevant
  pairs, trained on held-out `(q', relevant d)` anchors → a real, non-trivial target.

This is exactly **Concept 4** (correction geometry persisting across base-model upgrades) —
the most novel of the five concepts in the novelty assessment. The honest C2 upper bound
becomes "re-embed the entire corpus with the NEW model" (expensive); the realistic C2
baseline is "re-embed only a budget-limited subset"; C3 is the cheap adapter alternative.

## Implication for the plan
- PR-A1 as written (a `DriftInjector.inject_model_swap_drift` that mutates stored doc
  vectors) is a valid drift primitive but is NOT the oracle-defeating arena. Building the
  campaign on it would fail PR-A4's oracle-breaker gate.
- The oracle-defeating arena is a **query/eval-encoder-swap** at the harness level
  (embed eval queries with a second model projected to store dim), with documents cached in
  the old space. This re-scopes PR-A1 from an injector method to a harness eval-path option.
- The paper's C2-oracle insight (§4.3) gets STRONGER: the oracle survives even
  document-side model migration; only an encoder-upgrade (query-side) drift defeats it.

## Decision needed (operator)
Which arena to build for the next sprint:
(A) Query/encoder-upgrade drift (recommended) — defeats the oracle, tests the thesis,
    instantiates Concept 4.
(B) Keep document-side model-swap as an additional drift mode, but only as a
    severity/robustness probe, NOT as the thesis test (and document that C2 still bounds it).
(C) Both — A as the headline arena, B as a secondary robustness mode.
