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
| 27 | Evidence chain dashboard ingestion | reporting | complete | dashboard can surface latest evidence-chain summaries without reading JSON manually |
| 28 | Evidence artifact retention policy | governance | complete | operator doc defines which generated evidence artifacts are retained, regenerated, or excluded from git |
| 29 | Evidence chain manifest schema doc | documentation | complete | documented schema for evidence-chain summary and evidence-index records |
| 30 | Dashboard fetch robustness follow-up | reporting | complete | dashboard fetch helpers check response status and avoid rendering malformed evidence payloads |
| 31 | Evidence API backend parse hardening | reporting | complete | evidence dashboard API loaders skip malformed JSON consistently and return stable empty payloads |
| 32 | Evidence chain workflow freshness audit | governance | complete | manual evidence workflow can generate the index and run freshness audit after evidence-chain collection |
| 33 | Evidence dashboard runbook refresh | documentation | complete | operator docs explain dashboard evidence panels and when to regenerate artifacts |
| 34 | Evidence artifact cleanup dry-run | governance | complete | command lists generated evidence candidates safe to delete without deleting them |
| 35 | Evidence cleanup dashboard/report ingestion | reporting | complete | dashboard or generated report can surface dry-run cleanup candidates without deleting files |
| 36 | Evidence cleanup plan schema doc | documentation | complete | cleanup dry-run output fields and dashboard interpretation are documented |
| 37 | Evidence cleanup plan CI artifact | governance | complete | manual evidence workflow can publish a cleanup dry-run plan artifact without deleting files |
| 38 | Evidence cleanup freshness linkage | governance | complete | cleanup plan generation can run after freshness audit and record index/freshness artifact references |
| 39 | Cleanup plan dashboard source linkage | reporting | complete | dashboard cleanup panel shows linked evidence index and freshness audit paths |
| 40 | Evidence cleanup plan stale-source audit | governance | complete | cleanup plan can report when linked evidence index or freshness audit artifacts are missing |
| 41 | Cleanup source status dashboard | reporting | complete | dashboard cleanup panel shows whether linked source artifacts are present |
| 42 | Cleanup source status schema refresh | documentation | complete | dashboard cleanup-plan response documents source status fields |
| 43 | Cleanup source status API regression | validation | complete | API handler preserves default cleanup source paths and validates query overrides |
| 44 | Cleanup source status runbook note | documentation | complete | operator docs explain present/missing source status on the cleanup dashboard |
| 45 | Cleanup stale-source fail-closed summary | governance | complete | cleanup plan summary flags missing linked source artifacts without deleting files |
| 46 | Cleanup fail-closed dashboard status | reporting | complete | dashboard cleanup cards show cleanup-review allowed or blocked by missing source artifacts |
| 47 | Cleanup fail-closed API schema | documentation | complete | dashboard cleanup-plan response documents cleanup_review_allowed and missing_source_artifacts |
| 48 | Cleanup planner blocked-review exit mode | governance | complete | cleanup planner can optionally exit nonzero when cleanup_review_allowed is false |
| 49 | Cleanup workflow blocked-review guard | governance | complete | manual evidence workflow documents or wires the blocked-review exit mode without deleting files |
| 50 | Cleanup workflow artifact upload resilience | governance | complete | evidence workflow preserves cleanup/index artifacts for debugging even when blocked-review guard fails |
| 51 | Cleanup workflow blocked-review diagnostics | reporting | complete | workflow emits a compact blocked-review diagnostic summary before failing |
| 52 | Cleanup workflow manual dispatch guardrail | governance | complete | manual evidence workflow exposes an explicit cleanup guard mode input with fail-closed default |
| 53 | Cleanup guard mode docs schema refresh | documentation | complete | cleanup workflow guard modes are documented in the cleanup-plan schema/runbook contract |
| 54 | Cleanup guard mode warning diagnostic | reporting | complete | warn-mode workflow runs still emit cleanup-review allowed/blocked summary without failing |
| 55 | Cleanup workflow diagnostic de-duplication | maintainability | complete | shared cleanup diagnostic generation avoids duplicated inline workflow scripts |
| 56 | Cleanup diagnostic script entry point | implementation | complete | cleanup-review diagnostic helper is exposed as a console script for local operators |
| 57 | Cleanup diagnostic schema doc fields | documentation | complete | cleanup diagnostic output fields are documented as a stable operator-facing contract |
| 58 | Cleanup diagnostic local command validation | validation | complete | local validation covers the installed console-script behavior or module equivalent |
| 59 | Cleanup diagnostic invalid-json handling | validation | complete | diagnostic helper reports malformed cleanup plans without traceback noise |
| 60 | Cleanup diagnostic workflow invalid-plan note | documentation | complete | workflow docs explain malformed cleanup-plan diagnostic output and operator response |
| 61 | Cleanup diagnostic unreadable-plan schema action | documentation | complete | cleanup schema gives the operator response for unreadable cleanup-plan diagnostics |
| 62 | Cleanup diagnostic missing-plan schema action | documentation | complete | cleanup schema gives the operator response for missing cleanup-plan diagnostics |
| 63 | Cleanup diagnostic missing-plan workflow note | documentation | complete | workflow docs explain missing cleanup-plan diagnostic output and operator response |
| 64 | Cleanup diagnostic missing-plan CLI validation | validation | complete | module CLI coverage proves missing cleanup-plan diagnostics exit cleanly without tracebacks |
| 65 | Cleanup diagnostic missing-plan summary append validation | validation | complete | GitHub summary append path is covered for missing cleanup-plan diagnostics |
| 66 | Cleanup diagnostic unreadable-plan summary append validation | validation | complete | GitHub summary append path is covered for unreadable cleanup-plan diagnostics |
| 67 | Cleanup diagnostic summary append no-env validation | validation | complete | GitHub summary mode exits cleanly when GITHUB_STEP_SUMMARY is unset |
| 68 | Cleanup diagnostic summary append no-env docs | documentation | complete | local diagnostic docs explain that --github-summary is inert without GITHUB_STEP_SUMMARY |
| 69 | Cleanup diagnostic summary append preexisting-content validation | validation | complete | GitHub summary append preserves preexisting summary content |
| 70 | Cleanup diagnostic append contract docs | documentation | complete | docs state summary diagnostics append after existing step summary content |
| 71 | Cleanup diagnostic append newline validation | validation | complete | GitHub summary append writes a trailing newline after cleanup diagnostics |
| 72 | Cleanup diagnostic append newline docs | documentation | complete | docs state summary append writes complete newline-terminated markdown blocks |
| 73 | Cleanup diagnostic summary parent-dir validation | validation | complete | GitHub summary append creates diagnostics when the summary file path is in an existing directory |
| 74 | Cleanup diagnostic summary path docs | documentation | complete | docs state GITHUB_STEP_SUMMARY parent directory must already exist |
| 75 | Cleanup diagnostic unwritable summary validation | validation | complete | GitHub summary append failure surfaces as an operator-visible exception |
| 76 | Cleanup diagnostic unwritable summary docs | documentation | next | docs state invalid summary paths fail visibly instead of silently dropping diagnostics |

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
- 2026-05-06: Evidence-chain dashboard ingestion added. `/api/evidence_chain_history` and the Campaign History tab can show latest chain pass/fail state, review status, artifacts, blockers, and report paths.
- 2026-05-06: Evidence artifact retention policy added. Generated evidence stays in local/CI artifacts unless curated as deterministic fixtures or source contracts.
- 2026-05-06: Evidence-chain/index schema doc added. The evidence-chain summary, evidence-index, freshness-audit, and dashboard read-only surfaces are now documented.
- 2026-05-06: Dashboard evidence fetch robustness added. Evidence panels now use checked JSON fetches and defensive object/array normalization before rendering.
- 2026-05-06: Evidence dashboard backend parse hardening added. Dashboard loaders now require JSON objects, skip malformed history files consistently, and return stable empty evidence-index payloads.
- 2026-05-06: Manual evidence-chain workflow freshness audit added. The workflow now regenerates the evidence index, audits freshness, and uploads both chain and index artifacts after collection.
- 2026-05-06: Evidence dashboard runbook added. Operators now have panel/API/source-artifact mapping plus regeneration triggers for evidence-chain, index, and freshness-audit outputs.
- 2026-05-06: Evidence artifact cleanup dry-run planner added. Operators can list generated evidence cleanup candidates while retaining the latest artifact per type, with no deletion behavior.
- 2026-05-06: Evidence cleanup dashboard ingestion added. The dashboard now exposes a dry-run cleanup plan API and panel with candidate counts, retained counts, candidate bytes, and candidate paths.
- 2026-05-06: Evidence cleanup plan schema documented. The dashboard cleanup-plan response is documented as compact and row-limited while the CLI output retains full audit fields.
- 2026-05-06: Evidence cleanup plan CI artifact added. The manual evidence workflow now writes and uploads a dry-run cleanup plan alongside evidence-chain and evidence-index artifacts.
- 2026-05-06: Evidence cleanup freshness linkage added. Cleanup plans now record the evidence-index and freshness-audit paths they were generated after.
- 2026-05-06: Cleanup dashboard source linkage added. The cleanup panel now shows the evidence-index and freshness-audit paths linked by the dry-run cleanup plan.
- 2026-05-06: Cleanup stale-source audit added. Cleanup plans now report presence status for linked evidence-index and freshness-audit artifacts.
- 2026-05-06: Cleanup source status dashboard added. The cleanup panel now shows whether linked evidence-index and freshness-audit artifacts are present or missing.
- 2026-05-06: Cleanup source status schema refreshed. The dashboard cleanup-plan response now documents source artifact paths and presence fields, and the API wrapper preserves planner defaults.
- 2026-05-06: Cleanup source status API regression added. Empty source-path overrides now preserve planner defaults instead of resolving to the current directory.
- 2026-05-06: Cleanup source status runbook note added. Operator docs now explain present/missing cleanup source status and the regenerate-before-delete response.
- 2026-05-06: Cleanup stale-source fail-closed summary added. Cleanup plans now expose `cleanup_review_allowed` and missing linked source artifact names.
- 2026-05-06: Cleanup fail-closed dashboard status added. The cleanup dashboard now shows cleanup-review allowed/blocked status and missing source artifact names.
- 2026-05-06: Cleanup fail-closed API schema refreshed. The dashboard cleanup-plan response now documents `cleanup_review_allowed` and `missing_source_artifacts`.
- 2026-05-06: Cleanup planner blocked-review exit mode added. `--fail-on-blocked-review` lets cleanup-plan callers exit nonzero when linked source artifacts are missing while preserving default read-only behavior.
- 2026-05-06: Cleanup workflow blocked-review guard added. The manual evidence workflow now fails cleanup planning when linked source artifacts are missing, while cleanup remains dry-run only.
- 2026-05-06: Cleanup workflow artifact upload resilience added. The manual evidence workflow now uploads available evidence artifacts even when the cleanup review guard fails.
- 2026-05-06: Cleanup workflow blocked-review diagnostics added. Blocked cleanup review now emits missing source artifacts, source status, and candidate counts before failing.
- 2026-05-06: Cleanup workflow manual dispatch guardrail added. The manual evidence workflow now exposes `cleanup-guard-mode` with fail-closed default and explicit warn mode.
- 2026-05-06: Cleanup guard mode docs schema refreshed. The cleanup-plan schema now documents workflow `fail` and `warn` guard-mode behavior.
- 2026-05-06: Cleanup guard mode warning diagnostic added. Warn-mode workflow runs now emit cleanup-review allowed/blocked status and source status without failing.
- 2026-05-06: Cleanup workflow diagnostic de-duplication added. Cleanup-review diagnostics now come from `cleanup_review_diagnostic.py` instead of duplicated inline workflow scripts.
- 2026-05-06: Cleanup diagnostic script entry point added. Local operators can run `chelatedai-cleanup-review-diagnostic` for the same cleanup-review diagnostic used by CI.
- 2026-05-06: Cleanup diagnostic schema fields documented. The cleanup-plan schema now maps diagnostic lines to source fields and operator meaning.
- 2026-05-06: Cleanup diagnostic local command validation added. Unit coverage now exercises `python -m cleanup_review_diagnostic` as the source-checkout equivalent to the console script.
- 2026-05-06: Cleanup diagnostic invalid-json handling added. Malformed cleanup plans now render clean unreadable-plan diagnostics without traceback noise.
- 2026-05-06: Cleanup diagnostic workflow invalid-plan note added. The default-promotion evidence runbook now explains how to respond when workflow diagnostics report an unreadable cleanup plan.
- 2026-05-06: Cleanup diagnostic unreadable-plan schema action added. The cleanup-plan schema now states that unreadable cleanup-plan diagnostics require full evidence regeneration before cleanup decisions.
- 2026-05-06: Cleanup diagnostic missing-plan schema action added. The cleanup-plan schema now distinguishes missing cleanup-plan artifacts from unreadable artifacts and keeps both fail-closed.
- 2026-05-06: Cleanup diagnostic missing-plan workflow note added. The default-promotion evidence runbook now explains how to respond when workflow diagnostics report a missing cleanup plan.
- 2026-05-06: Cleanup diagnostic missing-plan CLI validation added. Module CLI coverage now proves missing cleanup-plan diagnostics exit cleanly without traceback output.
- 2026-05-06: Cleanup diagnostic missing-plan summary append validation added. The GitHub summary append path now covers missing cleanup-plan diagnostics without traceback output.
- 2026-05-06: Cleanup diagnostic unreadable-plan summary append validation added. The GitHub summary append path now covers malformed cleanup-plan diagnostics without traceback output.
- 2026-05-06: Cleanup diagnostic summary append no-env validation added. `--github-summary` now has regression coverage for local runs without `GITHUB_STEP_SUMMARY`.
- 2026-05-06: Cleanup diagnostic summary append no-env docs added. Operator docs now explain that `--github-summary` appends only when `GITHUB_STEP_SUMMARY` is set.
- 2026-05-06: Cleanup diagnostic summary append preexisting-content validation added. Regression coverage now verifies cleanup diagnostics append after existing GitHub step-summary content.
- 2026-05-06: Cleanup diagnostic append contract docs added. Operator docs now state summary writes are append-only and preserve existing step-summary content.
- 2026-05-06: Cleanup diagnostic append newline validation added. GitHub summary append coverage now verifies cleanup diagnostics are written as newline-terminated blocks.
- 2026-05-06: Cleanup diagnostic append newline docs added. Operator docs now state GitHub summary appends are complete newline-terminated markdown blocks.
- 2026-05-06: Cleanup diagnostic summary parent-dir validation added. Coverage now verifies `--github-summary` creates the summary file when its parent directory already exists.
- 2026-05-06: Cleanup diagnostic summary path docs added. Operator docs now state `GITHUB_STEP_SUMMARY` should point to a writable file in an existing directory.
- 2026-05-06: Cleanup diagnostic unwritable summary validation added. Coverage now verifies invalid summary paths fail visibly instead of silently dropping diagnostics.
