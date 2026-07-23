# Waypoint Research — 2026-06-09

**LOCAL ONLY — DO NOT COMMIT OR PUSH.** This folder is excluded via `.git/info/exclude`
(`docs/waypoint-research-2026-06-09/`). It will not appear in `git status` and cannot be
accidentally staged. If you ever want to publish any of it, copy content out deliberately.

## What this waypoint is

A point-in-time novelty assessment of the ChelatedAI theory stack against the 2024–2026
literature, produced via a deep-research run (106 agents, 24 primary sources, 25 claims
adversarially verified — 19 confirmed at 3-0, 1 genuinely refuted 0-3, 5 killed by
abstention when the session limit hit) plus targeted follow-up searches on the shim /
steering-post concept.

## Files

| File | Contents |
|---|---|
| [novelty-assessment-five-concepts.md](novelty-assessment-five-concepts.md) | Per-concept assessment: semantic collapse correction, self-annealing pools, evidence DAG, correction geometry, computational storage — closest prior art, how close, what remains uncommon |
| [shims-steering-posts-deep-dive.md](shims-steering-posts-deep-dive.md) | Follow-up deep-dive on the shim / steering-post / annealed-route idea, including the Feb–May 2026 papers that got close (SVF, FLAS, Queryable LoRA, LD-MoLE) |
| [verified-claims-and-sources.md](verified-claims-and-sources.md) | Raw verified/refuted claims from the deep-research run with quotes, vote counts, and the full source list |
| [implementation-plan-june-2026-drift-paper.md](implementation-plan-june-2026-drift-paper.md) | Step-by-step June 2026 execution plan (written for a lower-capacity implementing agent): 5 PRs building the drift-recovery experiment + parallel arXiv preprint draft |
| `paper-draft/` | (created during execution) the local-only paper draft |

## Headline conclusions (2026-06-09)

1. **Every individual ingredient is published somewhere** — several only in Feb–May 2026
   (query-aware dimension masking, SmartVector lifecycle, Steering Vector Fields, FLAS).
   The field is converging on this neighborhood fast.
2. **The integrated closed loop is not published anywhere found**: drift detection that
   *triggers* bounded correction, index lifecycle coupled to correction *training* under
   one annealing/temperature controller, and correction-actuators as first-class graph
   entities.
3. **Three defensible novelty claims**:
   - Closed-loop detect → bounded-correct → anneal coupling (index lifecycle + correction training together)
   - Correction-actuators (adapters/masks/steering routes) as first-class nodes in an evidence DAG
   - Correction-geometry meta-space persisting across base-model swaps (most conceptually novel, least developed)
   - For shims specifically: a *living* steering bank — routes annealed/pruned under drift,
     applied to retrieval pools rather than LLM residual streams
4. **The proving experiment is already on the roadmap**: Phase II step 14 (drift-injection
   recovery). Baselines handed to us by the literature: Search-Adaptor, DIME, Ada-IVF/Quake,
   SmartVector, SVF (static bank), LD-MoLE (one-shot gating).

## Verification coverage caveats

- Concepts 1 and 2 were fully adversarially verified (3-0 votes).
- Concept 3 (Zep/Graphiti, HippoRAG 2) claims were extracted but verification votes were
  cut off by a session limit — treat as likely-but-unconfirmed.
- Concepts 4 and 5 were sourced but never reached verification; those sections lean on
  fetched sources plus model knowledge (cutoff Jan 2026) and are flagged inline.
- The shim deep-dive used direct web search (2026-06-09), not the adversarial pipeline.

## Repo state at time of writing

Local `main` == `origin/main` at `bf23a47` (Liquified Lattice vision/roadmap docs).
Working tree clean. Assessment inputs: `docs/VISION_LIQUIFIED_LATTICE.md`,
`docs/RESEARCH_TRACKS.md`, README Liquified Lattice framing, Session 30 dual-hemisphere
memory notes.
