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
| 3 | Overlay trajectory health diagnostics | diagnostics | complete | report loop burden, blocker recurrence, oracle-gap trend, and budget per safe pass |
| 4 | Budget-aware overlay collection policy | implementation | complete | collection policy chooses broader branch search by uncertainty, coverage novelty, and blocker history |
| 5 | Verifier/rubric evidence card integration | reporting | complete | verifier/rubric outputs are evidence cards, not default runtime controllers |
| 6 | Model-Scope smoke campaign with supplied overlay report | validation | complete | campaign report shows promotion decision and overlay summary together |
| 7 | Hard-negative replay expansion for overlay readiness | validation | complete | readiness summary includes stress replay blocker state |
| 8 | Repeat-seed AttnRes contrastive/quantization-aware experiment | research/validation | complete | result doc states promote/no-promote with repeat-seed evidence |
| 9 | Computational-storage real-hardware evidence capture | operational | externally blocked | hardware report captured on actual RP2040/Pico device |
| 10 | Dashboard campaign-history view for model-scope reports | reporting | complete | latest campaign reports visible without reading files manually |
| 11 | Promotion-contract artifact-card linkage | implementation | complete | promotion decision links candidate artifact card and rollback path |
| 12 | Long-run validation bundle | validation | complete | single command runs focused overlay/model-scope regression suite |
| 13 | ARCH-AEP tracker closure audit | documentation | complete | all current trackers/indexes agree on open/complete/external-blocked state |
| 14 | Session memory and resume summary refresh | documentation | recurring | next-session and phase summary stay current after each merged PR |
| 15 | Default-promotion decision review | governance | gated | only starts after validation items show repeatable positive evidence |
| 16 | Validation bundle dashboard ingestion | reporting | complete | dashboard can surface latest validation-bundle pass/fail state |
| 17 | Overlay validation bundle CI affordance | validation | complete | CI or local command can run the bundle with clear timeout/failure reporting |
| 18 | Promotion linkage audit report | governance | complete | existing campaign artifacts can be checked for artifact-card and rollback linkage |
| 19 | Model-Scope overlay bundle schema doc | documentation | complete | generated campaign bundle fields are documented for future operators |
| 20 | Default-promotion evidence preflight | governance | complete | preflight command states whether validation, audit, and repeat evidence are sufficient to start promotion review |
| 21 | Preflight dashboard ingestion | reporting | complete | dashboard can surface the latest default-promotion preflight decision and blockers |
| 22 | Preflight CI affordance | governance | complete | CI or local command can run/link validation, audit, repeat decision, and preflight with clear fail-closed status |
| 23 | Promotion evidence runbook | documentation | complete | operator doc explains the promotion evidence bundle, blocker meanings, and no-default-change path |
| 24 | Cross-artifact evidence index | reporting | complete | single generated index links validation, audit, repeat decision, preflight, campaign reports, and artifact cards |
| 25 | Evidence-index dashboard ingestion | reporting | complete | dashboard can surface the latest cross-artifact evidence index summary and artifact counts |
| 26 | Evidence index freshness audit | governance | complete | command reports whether evidence-index entries point to existing files and whether the index is stale against source artifacts |
| 27 | Evidence chain dashboard ingestion | reporting | next | dashboard can surface latest evidence-chain summaries without reading JSON manually |
| 28 | Evidence artifact retention policy | governance | queued | operator doc defines which generated evidence artifacts are retained, regenerated, or excluded from git |
| 29 | Evidence chain manifest schema doc | documentation | queued | documented schema for evidence-chain summary and evidence-index records |

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
- 2026-05-05: Overlay trajectory health diagnostics implemented. Overlay reports and artifact cards now expose loop burden, blocker recurrence, oracle-gap trend, and budget per safe pass.
- 2026-05-05: Budget-aware overlay collection policy implemented as advisory-only evidence. Campaigns now emit collection policy artifacts and include them in overlay artifact cards.
- 2026-05-05: Verifier/rubric evidence cards implemented as review-only artifacts. Campaigns now emit verifier cards and embed them in overlay artifact cards.
- 2026-05-05: Model-Scope overlay smoke runner implemented. The deterministic smoke command generates supplied overlay reports and verifies the full campaign overlay bundle.
- 2026-05-05: Hard-negative overlay replay readiness added. Overlay reports can now carry stress replay blockers, and readiness fails closed when they are present.
- 2026-05-05: Repeat-seed AttnRes decision codified from existing artifacts. Machine summary says `no_default_change` because SciFact has non-positive deltas and 5/6 quantization gates fail.
- 2026-05-05: Dashboard campaign-history API and tab added. Live model-scope campaign reports are discoverable from `experiment_runs` without reading files manually.
- 2026-05-05: Promotion decisions can now carry required artifact-card and rollback references. Model-Scope overlay campaigns require the linkage and fail closed if it is missing.
- 2026-05-05: Long-run overlay/model-scope validation bundle added as `run_overlay_model_scope_validation.py` and `chelatedai-overlay-model-scope-validation`. Queue expanded with the next concrete reporting, CI, audit, and schema-doc follow-ups.
- 2026-05-05: Dashboard validation-history ingestion added. The Campaign History tab can now show latest validation-bundle pass/fail state and failed command names.
- 2026-05-05: Manual GitHub Actions validation workflow added for the overlay/model-scope bundle. Local operators can also list the exact commands with `python run_overlay_model_scope_validation.py --list-commands`.
- 2026-05-05: Promotion-linkage audit command added. Campaign reports with overlay evidence or promotion-ready decisions now have a scanner for missing artifact-card and rollback references.
- 2026-05-05: Model-Scope overlay bundle schema documented for future operators, including campaign outputs, overlay sidecars, validation summaries, and linkage audit requirements.
- 2026-05-06: Default-promotion evidence preflight added as `default_promotion_preflight.py` and `chelatedai-default-promotion-preflight`. Current evidence passes validation and linkage audit but blocks promotion review because repeat-seed AttnRes evidence does not support a default change.
- 2026-05-06: Dashboard preflight-history ingestion added. The Campaign History tab and `/api/preflight_history` can surface latest default-promotion preflight status, blockers, artifact count, and report path.
- 2026-05-06: Default-promotion evidence-chain runner and manual workflow added. The chain links validation, promotion-linkage audit, repeat-seed decision, and preflight outputs while keeping no-default-change as an explicit fail-closed status.
- 2026-05-06: Default-promotion evidence runbook added. Operators now have one doc for the evidence-chain command, manual workflow, blocker meanings, and expected no-default-change path.
- 2026-05-06: Cross-artifact evidence index generator added. Operators can generate one compact index over validation, audit, repeat-decision, preflight, evidence-chain, campaign, and overlay artifact-card outputs.
- 2026-05-06: Evidence-index dashboard ingestion added. `/api/evidence_index` and the Campaign History tab can surface artifact counts, latest chain status, review status, blockers, and index path.
- 2026-05-06: Evidence-index freshness audit added. The audit verifies that indexed artifact paths exist and that the index is not older than its referenced artifacts.
