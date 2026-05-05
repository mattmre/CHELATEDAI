# Roadmap Execution Queue - 2026-05-05

Purpose: Durable continuation queue after Engine-Scope, Model-Scope, and adaptive overlay implementation scaffolds landed on `main`.

## Status
- Engine-Scope cycle: complete as implementation scaffold; no default promotion.
- Model-Scope cycle: complete as implementation scaffold; promotion remains fail-closed.
- Adaptive overlay chain: implemented through records, metrics, readiness, promotion input, campaign reporting, integrated diagnostics, and dashboard summaries.

## Execution Rules
- Review open PR comments and checks before creating new branches.
- Keep one PR-sized slice per branch.
- Each PR must include focused validation and a phase summary or session-log update.
- Re-run a research/refinement pass before starting any high-uncertainty implementation path.
- Do not promote defaults from single-slice or single-seed evidence.

## Priority Queue
| rank | item | type | status | acceptance target |
| --- | --- | --- | --- | --- |
| 1 | Overlay artifact cards for promoted or candidate overlays | reporting | complete | compact card includes purpose, base, data, evals, safety checks, limitations, rollback |
| 2 | Broader adaptive-overlay replay and holdout validation | validation | complete | produce replay/holdout artifact proving ready or blocked status |
| 3 | Overlay trajectory health diagnostics | diagnostics | next | report loop burden, blocker recurrence, oracle-gap trend, and budget per safe pass |
| 4 | Budget-aware overlay collection policy | implementation | queued | collection policy chooses broader branch search by uncertainty, coverage novelty, and blocker history |
| 5 | Verifier/rubric evidence card integration | reporting | queued | verifier/rubric outputs are evidence cards, not default runtime controllers |
| 6 | Model-Scope smoke campaign with supplied overlay report | validation | queued | campaign report shows promotion decision and overlay summary together |
| 7 | Hard-negative replay expansion for overlay readiness | validation | queued | readiness summary includes stress replay blocker state |
| 8 | Repeat-seed AttnRes contrastive/quantization-aware experiment | research/validation | queued | result doc states promote/no-promote with repeat-seed evidence |
| 9 | Computational-storage real-hardware evidence capture | operational | externally blocked | hardware report captured on actual RP2040/Pico device |
| 10 | Dashboard campaign-history view for model-scope reports | reporting | queued | latest campaign reports visible without reading files manually |
| 11 | Promotion-contract artifact-card linkage | implementation | queued | promotion decision links candidate artifact card and rollback path |
| 12 | Long-run validation bundle | validation | queued | single command runs focused overlay/model-scope regression suite |
| 13 | ARCH-AEP tracker closure audit | documentation | complete | all current trackers/indexes agree on open/complete/external-blocked state |
| 14 | Session memory and resume summary refresh | documentation | recurring | next-session and phase summary stay current after each merged PR |
| 15 | Default-promotion decision review | governance | gated | only starts after validation items show repeatable positive evidence |

## Closed Implementation Scaffolds
- Engine-Scope row contract, internal gates, coverage analysis, hard-negative pipeline, and supervisor integration are implemented.
- Model-Scope hook runtime, feature extraction, shadow steering, segmented memory, overlay training/promotion, and engine integration are implemented.
- HeavySkill-derived adaptive overlay scaffolding is implemented as an observation-first engine-control layer, not as a full harness.

## Known Non-Goals
- No base-weight mutation.
- No default runtime route changes from weak or single-run evidence.
- No full HeavySkill-style agent harness unless a verifier-backed use case appears.

## Resume Note
Start the next branch from the highest-ranked `next` item that is not blocked. If research changes the ranking, update this file in the same PR that records the research.

## Research Refresh
- 2026-05-05: Frontier adaptive overlay/test-time scaling refresh recorded in `docs/frontier-adaptive-overlay-research-2026-05-05.md`. Queue updated to prioritize overlay artifact cards before broader branching.
- 2026-05-05: Overlay artifact cards implemented for Model-Scope campaign reports. Next slice remains broader replay/holdout validation using the card as the durable review object.
- 2026-05-05: Adaptive overlay validation reports implemented. Campaigns can now carry replay and holdout overlay reports and emit fail-closed validation evidence into the artifact card.
