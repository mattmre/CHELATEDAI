# Next Session

<!--
  This file is the Tier C (cross-PR / cross-session) state surface for the
  Brutal Honesty Rulebook v3.2. It is read by:
    - scripts/check_block_flag.py  (the merge gate for new feature work)
    - the Tier B reviewer of the next PR you open
    - whoever picks the work back up after a session boundary

  Schema is fixed. Do not rename headings. Do not collapse the tables. Empty
  state is honest state — empty rows do not get deleted, they get filled with
  the "_none yet_" placeholder which the validator recognizes.

  Cycle definition (per rulebook §6.1): 1 cycle = the next operator-initiated
  session OR 5 calendar days, whichever comes FIRST. Items with TTL=1 cycle
  must be cleared before the next cycle starts or the block flag flips to
  BLOCKED automatically.
-->

## Block flag

**Current**: `CLEAR` — no Carried Debt items have expired.

When the flag is `BLOCKED`, no new feature work may merge until Carried Debt
is empty. The flag is set automatically by `scripts/check_block_flag.py`:
- `CLEAR` if no Carried Debt rows OR all open rows are still in their first
  cycle (TTL not yet expired).
- `BLOCKED` if any open Carried Debt row has survived a full cycle without
  being closed.

The script does not know cycle age — that is set by the operator at session-
wrap by inspecting the TTL column. The `**Current**:` line above is the
authoritative source; everything else is advisory.

## Carried Debt

| ID | Item | Source | TTL | Blocking | Status |
|----|------|--------|-----|----------|--------|
| CD-001 | smoke_pipeline.py ceiling-tier not yet implemented; floor-tier only (`run_ceiling_smoke()` returns sentinel 2). Ceiling gap = no real end-to-end fixture exercise of AntigravityEngine | kit install 2026-05-10 | 1 cycle | NO — honestly disclosed per Rule 5 | **CLOSED** by PR <pending consolidation PR> — `run_ceiling_smoke()` now constructs `AntigravityEngine(qdrant_location=":memory:", model_name="all-MiniLM-L6-v2")`, ingests 4 docs, runs `get_chelated_vector()` + `embed()` against the production code path, asserts non-zero vector with `vector_size=384`; honest skip path retained for envs missing torch / sentence-transformers / qdrant; covered by `test_smoke_pipeline_ceiling.py` (8 tests) |
| CD-002 | `scripts/smoke.sh` Stage 1 exits non-zero: `tests/test_e2e_smoke.py` does not exist; smoke.sh is the `bash`-mode entry point but the repo has no e2e smoke test file. The Python `smoke_pipeline.py` path (used by CI and operator) is unaffected. | kit v3.3 upgrade 2026-05-12 | 1 cycle | NO — CI uses `smoke_pipeline.py` directly; gap is only in the `bash scripts/smoke.sh` code path | **CLOSED** by PR <pending consolidation PR> — added `tests/test_e2e_smoke.py` (unittest surface-boot covering `antigravity_engine` + 8 load-bearing modules and the `AntigravityEngine` entry-point class); swapped `smoke.sh` Stage 1 invocation from `python -m pytest tests/test_e2e_smoke.py` to `python -m unittest -v tests.test_e2e_smoke` per CLAUDE.md (CI has no pytest) |
| CD-244-01 | `scripts/bhs_validator.py:43-50` `validate_pr_brutal_honesty()` returns hardcoded `BHSResult(score=0.0)`; `:53-58` `run_smoke_pipeline()` always returns `True`. AEP orchestrator hooks call these so `summary["avg_bhs_score"]` is always `0.0`. L1 + L4. Violates Session Rule #1. | PR #244 (2026-05-16) | 1 cycle | YES — load-bearing stub | **CLOSED** by PR #245 |
| CD-244-02 | `aep_orchestrator.py:679, 745, 919-926` consume `bhs_metadata` from stub call; `summary["avg_bhs_score"]` always `0.0`. After CD-244-01 lands, verify the score actually varies with finding content and is surfaced in operator-facing closure summary. | PR #244 (2026-05-16) | 1 cycle | YES — depends on CD-244-01 | **CLOSED** by PR #245 |
| CD-244-03 | New comp-storage modules (`computational_storage_poc/moe_reap.py`, `sparse_cpu_inference.py`, `packed_graph.py`, `packed_cpu_inference.py`, `repo_graph_memory.py`, `integrated_repo_runtime.py`, `phase7_system_evaluation.py`, `disk_llm_estimator.py`, `cpu_backends.py` + benchmarks) have unit tests but zero references from production paths. L4 + L8. | PR #244 (2026-05-16) | 1 cycle | NO — POC scoped + honestly disclosed | **CLOSED** by PR #247 — `disk_llm_estimator` wired into `dashboard_server.py` `/api/disk_llm_estimate` rendered in dashboard campaigns-tab panel; other 8 modules marked `EXPERIMENTAL = True` + load-bearing `mark_experimental()` import-time check; tabulated in `computational_storage_poc/README.md` Status section |
| CD-244-04 | `aep_orchestrator.py:33` catches `Exception` (not `ImportError`) around the BHS import; any future runtime error in `scripts.bhs_validator` is silently absorbed. L11 risk. | PR #244 (2026-05-16) | 1 cycle | NO — post-CD-244-01 cleanup | **CLOSED** by PR #245 |
| CD-244-05 | `computational_storage_poc/model.cspg` binary artifact tracked in repo via PR #244; decide ignore/LFS/remove. | PR #244 (2026-05-16) | 1 cycle | NO — hygiene | **CLOSED** by PR #246 — Option 3 (untrack + gitignore `*.cspg`); README documents regeneration commands |
| CD-245-01 | `scripts/bhs_validator.py` `_score_finding` rubric is length-based + keyword-based, not semantic. Three Tier B iterations (92/96/85) converged on this: `"xxxxxxxxxxxx"` (12 identical chars) passes the min-content check; padded keyword-bait can hit score 100 without committed prose. L13 (soft-prose-claimed-as-mechanical). Per rulebook §6.1, same gap class surviving 2 iterations escalates — flagging as Tier C debt rather than looping further. | PR #245 (2026-05-16) | 1 cycle | NO — rubric depth research, not an L1/L4 in production data flow | **CLOSED** by PR #248 — Candidate A + B per research-agent recommendation: entropy/unique-token/dominant-token content-quality penalty in `scripts/bhs_validator.py::_content_quality_penalty` (drops `"xxxxxxxxxxxx"` to 85), `_score_finding` renamed to `_score_finding_structure` with backwards-compat alias, scope documented in `docs/bhs-rubric-scope.md`, operator audit script at `scripts/audit_findings.py` for periodic human sample-grade. Residual diverse-but-meaningless gap (`"foo bar baz qux at handler.py:42"` still scores 100) is honestly acknowledged in the scope doc and asserted in `test_bhs_validator.py::test_diverse_but_meaningless_prose_acknowledged_gap_scores_100` so any future "we closed it" claim must actually change the rubric. |
| CD-247-01 | `aep_orchestrator.py:678, 966` carry inline `# BHS v3.3 placeholder hook` / `# BHS v3.3 placeholder (to be expanded)` comments; PR #245 replaced the underlying validator with a real implementation but the comments now misrepresent the integration state. Stale-docs L13-light. | post-consolidation audit (2026-05-16) | 1 cycle | NO — docs drift, no behaviour change | **CLOSED** by PR #267 (Track 0 hygiene) — both comments refreshed: `aep_orchestrator.py:678` now states the validator is the real PR #245 implementation attaching advisory scores (gating not enforced), and `:966` states the floor-tier smoke gate runs the real `run_smoke_pipeline`. |
| CD-247-02 | Six broad-`except Exception` swallow sites across production code (`antigravity_engine.py:1080-1081, 2460-2461`; `run_phase_c_eval.py:433-434`; `checkpoint_manager.py:372`; `engine_scope_coverage.py:35`; `run_weight_refinement_campaign.py:443/589/626`) silently absorb errors with `pass` or `return None`. Per L11, even benign optional-path swallows should at minimum log at debug. | post-consolidation audit (2026-05-16) | 1 cycle | NO — audited as deliberate non-critical paths | **CLOSED** by PR #267 (Track 0 hygiene) — audited all six citations against current `main`; only `run_phase_c_eval.py:433-434` was a genuine broad swallow and is now narrowed to log the exception type. The other five citations were stale/over-flagged and verified to need no change: `antigravity_engine.py` sites already `log_error` or are L11-disclosed (lines moved; current broad-excepts at 283/1135/1326/2541/2576/2808/2815 all log); `checkpoint_manager.py:371` is `except ValueError: pass` inside a `__main__` demo block; `engine_scope_coverage.py:34` is already a narrow `except (TypeError, ValueError)` with a fallback return; `run_weight_refinement_campaign.py` contains no `except Exception` at the cited lines. Honest disposition: no cosmetic edits manufactured to satisfy stale citations. |
| CD-A2-01 | The real swap-backend resolution path (`query_encoder_drift.QueryEncoderDrift._backend` → `embedding_backend.create_embedding_backend(swap_model_name)`, loading the real `all-mpnet-base-v2`) is no longer exercised by any default-CI test after the real-model smoke was made opt-in (gated on `CHELATED_RUN_REAL_MODEL_TESTS=1`) to stop a ~17-min HF-connection CI hang. It is covered only by mocked unit tests and the non-gating PR-A4 real-model campaign. L5 (untested production path in default CI). | PR for real-model-smoke opt-in (2026-06-14) | 1 cycle | NO — logic covered by stubs; real path deferred to PR-A4 campaign | **CLOSED — narrow procedural path proof only.** Committed log blob `e5ebca641969ca20c813045f5f0898e2881272d6` at commit `efac48daae5358599eeeb055906083e423eb5072` records `Initializing local backend: all-mpnet-base-v2` followed by `Local backend loaded. Vector size: 768`; the production code path reaches that factory through `run_drift_recovery_experiment._build_query_encoder_drift()` and `QueryEncoderDrift.embed_queries()` / `_backend()`. The exact anchor and required markers are validated by `scripts/validate_metric_lineage_quarantine.py` on local candidate `f2e41d42`. This proves that the real backend resolved during the retained NFCorpus campaign, not that the H2 re-run specifically resolved it and not that any stored nDCG value is valid. The default-CI skip remains intentional. |
| CD-MOD-001 | MOD-1 audit: `model_scope_runtime.LocalModelRuntime.load()` real `transformers.AutoModelForCausalLM.from_pretrained` branch has zero test coverage; every test injects a `MagicMock` loader. The `Qwen3.5-9B` pilot load (Phase 1 AC1) is unverified end-to-end. L5 + L8. | BHS Scope B audit 2026-05-16 | 1 cycle | NO — Model-Scope cycle phase scored 72; tracked under cycle-wide remediation | **CLOSED**: PR #250 — `ActivationEvent` gains `raw_tensor_shape` field; `TestRunInferenceTryBranchDiscrimination` covers real try-branch via sentinel patch; integration test added under `@skipUnless(CHELATED_INTEGRATION_MODEL)` for actual `AutoModelForCausalLM.from_pretrained` path with `sshleifer/tiny-gpt2` |
| CD-MOD-002 | MOD-2 audit: `qwen_scope_adapter.py` has no `hf_hub_download`, no HuggingFace repo string, no checksum; every SAE test uses `np.random.default_rng(42).random(...)` as the weight matrix. `QwenScopeAdapter.extract_features` projects only 3-4 scalar activation stats, not residual-stream tensors. Phase 2 AC ("Qwen3.5-9B sparse features from hooked residual states") not demonstrated. L4 + L1 + L5. | BHS Scope B audit 2026-05-16 | 1 cycle | NO — critical-severity but POC-bounded; honestly disclosed | **CLOSED**: PR #250 — `QwenScopeLayerSAE.from_file` real checkpoint path covered by `TestQwenScopeLayerSAEFromFile` (creates synthetic `.pt` fixture via `torch.save`, loads, calls `encode`, asserts output shapes and top-k sparsity); `extract_features()` docstring explicitly discloses operation on activation STATISTICS, not raw tensors (L13 resolved) |
| CD-MOD-003 | MOD-3 audit: `model_scope_steering` `InterventionRecord` lives in `self._records: List[...]` only; no `persist()`/`to_disk()` method. On fresh checkout all provenance is lost. Phase 3 AC3 ("provenance records show what feature, layer, and policy caused intervention") half-met: structure exists, durability does not. L4. | BHS Scope B audit 2026-05-16 | 1 cycle | NO — cosmetic vs durability | **CLOSED**: PR #254 — `SteeringActuator` gains `persist_records(path)` / `load_records(path)` (JSON Lines); provenance survives process restart; 10 new tests including fresh-instance round-trip |
| CD-MOD-004 | MOD-6 audit: `model_scope_engine_bridge.py:61` constructs `SteeringActuator(registry, max_total_interventions=0)`; intervention count is hardcoded to never exceed 0 regardless of `enable_steering`. Dashboard "intervention evidence" panel can never display anything other than 0. L4. | BHS Scope B audit 2026-05-16 | 1 cycle | NO — one-line fix; structurally constrains acceptance criterion #3 | **CLOSED**: PR #254 — `ModelScopeBridgeConfig.max_total_interventions: int = 100` added and wired to `SteeringActuator`; hardcoded `0` removed; two runtime-behavior tests guard against regression |
| CD-MOD-005 | MOD-5 audit: `OverlayTrainer.evaluate_promotion(baseline_events, candidate_events)` evaluates on training-data pairs (`all_inputs`/`all_targets`); `baseline_score` is comparator-pass-rate with default `threshold=0.0` (both sides trivially pass); no `stress` references in trainer or campaign. `rollback()` zeros in-memory weights only — never restores a prior promoted overlay file. Phase 5 AC3 ("promoted artifacts beat baseline under replay and stress checks; failed candidates roll back cleanly") not met. L4 (×3). | BHS Scope B audit 2026-05-16 | 1 cycle | NO — critical-severity but does not block running campaigns | **CLOSED**: PR #254 — `promote_candidate()` writes `.backup` before overwriting overlay; new `rollback()` restores from backup; `run_campaign()` splits episodes 80/20 before training and passes held-out eval set to `evaluate_promotion()` |
| CD-ENG-001 | ENG-5 audit: `run_golden_default_autopilot._recommendation` hardcodes `default_change_allowed: False` and `safe_default_holds: True`; no code path emits "supported candidate" or "documented no-promotion" terminal artifact. `evidence_contract`/`promotion_contract`/`compute_budget_policy`/`evaluator_fabric` are not imported by the supervisor. Phase 5 AC3 unimplemented. L4 + L13. | BHS Scope B audit 2026-05-16 | 1 cycle | NO — supervisor produces ongoing recommendations; gap is terminal artifact | **CLOSED**: PR #251 — `_recommendation()` now sets `default_change_allowed = bool(reform_candidate or mask_candidate)`; `main()` writes `terminal_decision.json` artifact after loop exits; `test_main_writes_terminal_decision_json` verifies all required fields |
| CD-ENG-002 | ENG-4 audit: `engine_scope_negatives` "clustering" is dict-bucketing on exact-string signature equality, not a real distance/cluster algorithm; module hard-codes `generator` field but has no synthesis path (only mining); no committed `golden_runs/` artifact and no replay-twice-and-diff determinism test. Phase 4 ACs met only via mining path; the "synthetic" framing is prose-soft. L13 + L5. | BHS Scope B audit 2026-05-16 | 1 cycle | NO — mining-only is acceptable per AC; framing is the gap | **CLOSED**: PR #251 — `mine_hard_negative_families()` docstring explicitly names algorithm "deterministic fault-class grouping" (not clustering); `test_build_hard_negative_replay_artifact_is_deterministic` calls builder twice and asserts identical `family_id` assignments and row ordering; label-fix regression guard added |
| CD-TTS-001 | TTS-2 audit: grep of all 17 `run_*.py` runners for `enable_tts` or `--enable-tts` returns zero matches. TTSPipeline is library-callable only; no campaign runner can populate the dashboard TTS panel without external glue code. Visible-without-evidence pattern: panel + API exist, but no operationally-reachable activation path. L4 + L5. | BHS Scope B audit 2026-05-16 | 1 cycle | NO — short fix; dashboard already shows honest "not enabled" empty state | **CLOSED**: PR #252 — `--enable-tts`, `--no-tts-translation`, and `--no-tts-transport` flags wired into `run_road_course_campaign.py`; CLI wiring tested through real `main()` in `TestRunRoadCourseCampaignCLIWiring` |
| CD-TTS-002 | TTS-1 audit: REM-C2 (per-inference signal clearing in `tts_pipeline.py:213-218`) and REM-H2 (`FeatureDirectionBank` Gaussian unit vectors) are fixed in code but no test would fail if either were reverted. Two of the four bug classes that triggered the post-merge remediation wave can silently regress. L5. | BHS Scope B audit 2026-05-16 | 1 cycle | NO — code is fixed; regression coverage is the gap | **CLOSED**: PR #252 — regression tests added for REM-C2 (cross-inference signal-accumulation guard) and REM-H2 (Gaussian direction-bank distribution test); both tests would fail if the corresponding fixes were reverted |
| CD-H1-01 | The committed swap-campaign reports included C3a metrics generated before H1 commit `5b2379b` removed a build-time bounded adapter from the C3a construction path. The active report lineage therefore needed post-fix procedural supersession; the stored pre- or post-fix quantitative values are not accepted evidence. L13 (stale-docs). | H1 (PR pending) | 1 cycle | NO — report supersession was required; quantitative magnitude is not accepted | **CLOSED — procedural supersession only.** After the H1 code fix landed, the swap manifests and report files were replaced by the H2 lineage and the pre-H1 reports ceased to be the active report surface. This closure makes no quantitative claim: no stored baseline equality, final nDCG, effect size, condition ordering, or “run noise” attribution is accepted. All such values are `LEGACY_METRIC_LINEAGE_BLOCKED`; corrected caller migration and regeneration are tracked by CD-MLR-01 and DS-MLR-01. |
| CD-MLR-01 | Repair drift-recovery metric lineage end to end: replace the retrieved-list IDCG helper with a qrels-complete graded/binary nDCG contract; migrate every drift caller; add regression coverage where relevant documents exceed retrieved hits; regenerate or explicitly retire every affected quantitative campaign whose claims are to be restored (including H2/H4/H5) with preserved environment/run provenance; and publish an explicit 113-entry old-artifact → corrected/retired-artifact supersession map with hashes. Until all acceptance points are met, exact values, comparator orderings, gates, promotions, rejections, and paper claims are prohibited. | PR #292 Tier-B reconditioning (2026-07-27) | 1 cycle | YES — blocks quantitative promotion/rejection and publication claims; first-cycle remediation debt | **OPEN — first cycle.** Complete immutable quarantine map candidate: `artifacts/legacy-ndcg-quarantine-index-v2.json` at local `f2e41d42` (113/113 affected tracked artifacts; v1 retained as an incomplete audit predecessor); fail-closed validator: `python scripts/validate_metric_lineage_quarantine.py`; explicit CI gate plus hostile mutation coverage: `test_validate_metric_lineage_quarantine.py`. Public PR #292 remains at `eb750958`, so this control is not yet GitHub-durable or merge-accepted. |
| CD-R13-01 | Rejected PR #293 cannot safely provide callback-driven DAG mutation, pruning/re-annealing, transaction rollback, or caller-supplied trusted provenance: exact iteration-six probes reproduced BaseException partial mutation, concurrent lost updates, and forged/aliased provenance. | PR #293 Tier-B scope reduction (2026-07-27) | 1 cycle | NO — the unsafe implementation is excluded; this becomes a hard gate only if the scope is reintroduced | **OPEN — first cycle.** Preserve `454e4a32` for audit; any future implementation must start from a fresh branch and independently prove atomic mutation, rollback, concurrency, and owned immutable provenance. |
| CD-R16-01 | Rejected PR #295 cannot safely provide a reusable promotion plane or engine activation path: exact iteration-five probes reproduced mixed-state publication/cancellation races, false promotion with incomplete REPORT qrels or no router-global route, invalid metric edge cases, publication-time provenance TOCTOU, and dimension-mismatched activation. | PR #295 Tier-B scope reduction (2026-07-27) | 1 cycle | NO — the unsafe implementation is excluded; this becomes a hard gate only if the scope is reintroduced | **OPEN — first cycle.** Preserve `730b305e` for audit; any future runtime implementation must independently prove atomic transitions/readers, complete qrels/global binding, bounded metrics, publication-time provenance, engine dimension/corpus identity, and an exact-head runtime smoke. |

**Schema**:
- `ID`: stable identifier, prefix `CD-` + sequential number (CD-001, CD-002, ...).
- `Item`: one short sentence stating the gap. Reference file:line where useful.
- `Source`: PR number or session ID that opened the debt.
- `TTL`: `1 cycle`, `expired`, or `—` (placeholder row only).
- `Blocking`: `YES` (forbids new feature work in next cycle if not cleared)
  or `NO — <reason>` (cosmetic / honestly-disclosed gap that does not block).
- `Status`: `OPEN — <note>`, `**CLOSED** by PR #N`, or `_—_` (placeholder).
  Rows starting with `CLOSED` (case-insensitive, stripping `**` markdown bold)
  are filtered out of the active-debt count by `check_block_flag.py`.

## Deferred Scope

Items from this cycle's PRs whose `DEFERRED_SCOPE:` field captured ≥25% of
original scope. These are NOT debt — they are honestly-bounded
not-in-scope items. They appear here so the next planner sees them.

| ID | Item | Source PR | Why deferred |
|----|------|-----------|--------------|
| DS-MLR-01 | Valid quantitative H2/H4/H5 conclusions | #292 | Metric-helper contract and caller migration, corrected campaign regeneration, and hash-linked supersession/provenance map remain for the research-validity owner in the next remediation cycle (TTL governed by CD-MLR-01). |
| DS-PRW-001 | Full 11,520-cell prime-ring campaign, real-corpus retrieval, repeated hardware timing, and promotion-grade scientific evidence | bounded prime-ring method-development branch (local) | The bounded 12-cell `p=4691` raw-sanity campaign and separate 4091/4691 Rader harness executed on 2026-07-24. RB-9 on 2026-07-25 falsified the 56-state nearest-only event-union normalization and internally proved the repaired 64-state asymptotic for the frozen exact-Hamming bank. RB-10 derived the frozen 50/50 analytic tie corollary and ran 14 formula tests. RB-11 then found that all `336/336` constructed leading midpoints tie under exact-Hamming/direct scoring but only `117/336` remain bit-exact ties through the current float-FFT scorer; the analytic theorem survives, while production-decoder linkage fails pending a declared numerical-tie contract and boundary parity test. The bounded claim chart now includes the direct 1992 Legendre-inner construction collision; specialist proof-equivalence/citation review remains required. The full grid remains refused at 1,059,556,800 modeled bytes. After decoder scope is resolved, remaining work is generic noisy/nonlinear quotient learning, matched transcript tests, common-channel carrier controls, process-tree RSS/checkpoint/kill-resume, and then bounded flat real-corpus validation. No novelty or production promotion is established. |
| DS-PRW-002 | Joint-plank orbit-spectrum coding, posterior-guided acquisition, and irreducible orbit-coded factor/hypergraph meshes (`PRW-JO1`, `PRW-A1`, `PRW-G2`, `PRW-G2A`) | RB-10 bounded implementation and execution (local) | Three exact cells executed on 2026-07-25 under a shared bounded artifact/resource contract and revalidated on 2026-07-26. `PRW-JO1` shell shaping is real but ties 15 restricted schedules plus complementary/random controls; stack and flat distances are identical for all 38,610 audited comparisons, closing stacking/dimensional novelty. Only an optional ordinary constrained-code comparison against known-design and unrestricted matched-cost controls remains. `PRW-A1` has fixed-label crossovers in all 45 action pairs and identical radial multisets, but this proves only truth-local relabelings—not one global prior-preserving channel isometry. Its next gate is full action-channel conjugacy plus real semantic action availability/cost; a policy runs only if that survives. `PRW-G2` has necessary pair/hyper/query interactions at `p=7` and bounded `p=11` unit evidence, while static advantage is false. Stage 4 is blocked until pairwise auxiliary minimality and a concrete orbit-specific representation/decoder advantage are specified. `PRW-G2A`, large meshes, and real relational retrieval remain unauthorized. |
| DS-R13-01 | Callback-driven evidence-DAG mutation, prune/re-anneal transactions, rollback guarantees, and trusted caller provenance | rejected PR #293 iteration six | The current implementation is disproved and will be withdrawn rather than repaired again. Do not open a normalization-only replacement now because no authorized production consumer exists. If a real consumer is later approved, a fresh normalizer may accept detector output and emit detached deterministic maps, but it must exclude graph mutation, re-annealing, transaction claims, caller-owned trusted provenance, and `DONE` wording. |
| DS-R16-01 | Reusable routing promotion plane, campaign runner, engine interception/activation, CLI/package exposure, and production runtime claims | rejected PR #295 iteration five | Withdraw current #295. The four consumed lock/REPORT artifacts (`rung16-arena_a_default_swap-selection-lock`, `rung16-arena_b_multi_domain-selection-lock`, and their two `report-consumed-48bad48fe78b` files) must remain byte-preserved; their reconciliation file records an actual-versus-claimed lock mismatch and audit parent `7a79b0ec`, while the results admit no contemporaneous Git/code hash. Any new run requires new IDs/paths and cannot upgrade those artifacts into exact-current-head evidence. A separate nine-file evidence-only replacement may archive eight immutable records plus one newly written disposition document. Separately, a later fresh component PR may contain only isolated router freeze/mutation locking and scale-first finite cosine/margin arithmetic with dedicated tests; it may not inherit promotion-plane or artifact claims. |

### Preservation and merge resume order

The exact research/evidence tree has a local pre-reconciliation bundle anchor
at `7dec564d`, with evidence commit `186590c8` and a verified private
full-history bundle. This later documentation-only reconciliation is bound
post-commit in the private recovery index without rewriting evidence. Private
V3 at snapshot head `8c7446fb` binds V2 and records 2,854 entries,
1,829,320,221 bytes, ten total worktrees, two stashes, zero errors, and
`cleanup_authorized: false`. No cleanup is authorized.

1. Complete candidate `f2e41d42` supplies PR #292's 113/113 metric-lineage
   reconditioning and earned independent local candidate Tier B 100, making it
   content-ready for owner-approved publication/CI. Current public-state BHS
   remains 70/Critical because #292 is still at `eb750958`; after explicit
   approval, fast-forward/update it, pass hosted checks, and obtain fresh
   exact-public-head review.
2. After explicit approval, withdraw rejected #293. Create a new #294
   transplant branch/replacement PR from accepted #292 using only
   `e12b7668`, `d8ccf70c`, `9d211613`, and `6e78cf41`; exclude stale
   `12aa1be6`, and do not force-push or retarget public #294.
3. After explicit approval, withdraw current #295. Optionally create a separate
   nine-file evidence-only replacement directly from final #292: eight
   byte-exact historical records plus one newly written narrowed disposition
   document. Do not cherry-pick any #295 commit or run a sixth repair iteration.
4. Reconcile all final tracker/next-session surfaces, generate a V3 private
   residual manifest and complete-history recovery bundle, and validate the
   exact committed primary tree.
5. Ask for explicit approval of each exact public mutation—pushes, PR
   title/body/base edits, closes/withdrawals, replacement PR opens, and
   merges—then prove remote equality, run hosted checks, and merge in
   dependency order.
6. After merged-state verification, regenerate the residual ledger and request
   owner signoff separately for each cleanup candidate.

### Prime-ring METHOD_DEV resume order

The exact RB-10/RB-11 source, tests, protocols, and four JSON evidence files
are preserved in local commit
`186590c8bde8311f39c17167110d5ba30b13a4fd`. The manifest validates against
those bytes, but the branch is not yet verified on the remote.
Before another experiment:

1. After the preservation merge lane above is technically green and the exact
   public payload is explicitly approved, publish the committed exact tree and
   prove remote SHA equality without regenerating or editing the four JSON
   artifacts.
2. Preserve `PRW-T1R` as an internal exact-Hamming theorem and resolve the
   production numerical-tie contract with exact-Hamming/direct/FFT boundary
   parity. Do not call the current float-FFT path a proved decoder.
3. Run the A1 global action-conjugacy/semantic-cost gate and the G2H
   auxiliary-minimality/named-candidate gate. Kill only the sublane whose own
   gate fails.
4. Treat JO1 known/unrestricted controls as optional ordinary-code research,
   not as evidence for stacking, dimensions, resonance, or living memory.
5. Run noisy/nonlinear quotient, matched transcript, common-channel carrier,
   process-tree RSS/checkpoint fault injection, and bounded flat real retrieval
   in that order. Do not open G2A or relational retrieval before a static pass.

### RB-13/RB-14 METHOD_DEV status

RB-13 remains a draft dependency queue, not Carried Debt, Deferred Scope, a
runnable preregistration, or evidence that a mechanism works. Its
status/dependency source of truth is Section 1.1 of
`docs/research/evidence-kernel-masked-subplane-experiment-queue-2026-07.md`.
The non-RB-14 cards still require their own protocol freeze and disposition.

On 2026-08-04 the user authorized a bounded implementation pass for RB-14. The
new module `observability_experiments.py` and runner
`run_rb14_observability.py` execute only deterministic NumPy/stdlib synthetic
cells. The focused RB-14 suite plus existing CRSV tests pass (`50 tests`, `OK`),
and the runner wrote five atomic artifacts under
`artifacts/method-dev/rb14-observability/` with manifest digests. Every stage
is `VALIDATED` only for its exact synthetic sanity boundary; every stage keeps
scientific and novelty status `UNCONFIRMED`.
The same frozen grid was rerun at seed `11` under the `seed-11` child directory
with exact digest recomputation; this is robustness of the fixture, not a
disjoint confirmation set.

- `PRW-OBS1` passes the in-span null and bounded full-rank recovery control; it
  does not establish a frontier-policy advantage.
- `PRW-COA1` passes the below-degree blindness and eligible pair/triad controls;
  it does not rename the prior G2 interaction result or establish utility.
- `PRW-CTX1` rejects incompatible singleton-witness composition, promotes a
  compatible shared-context fixture, and abstains on missing propensity
  support. Its real dependency on `PRW-EK2` remains.
- `PRW-SPU1-TRANSPORT` demonstrates raw coordinate drift and exact recovery by
  a known inverse transport. It remains a confound-control subcell blocked on
  the full `PRW-SPU1` disposition.
- `PRW-EK7-COALITION` shows a matched-token coalition surrogate recovering a
  conjunction missed by individual top-k. It is not an EK7 result and remains
  gated by `PRW-COA1`, the explicit CTX1 disposition, and the unchanged EK7
  entry gate.

Candidate-survival contrasts, disjoint confirmation sets, live corpus/model
evaluation, and production integration are outstanding. The first full
repository discovery attempt was stopped at roughly 709 MB RSS to honor the
resource guard; it emitted no final summary, so full-suite status is
unverified.

The queue has independently disposed families and cross-cutting additions:

- `PRW-EK0`--`PRW-EK7`: current-memory characterization, a Donto-adjacent
  append-only bitemporal evidence-kernel bridge, source-genealogy collapse,
  claim-typed gates, truth/authority separation, local demotion, reversible
  language views, and a survivor-only agent/RAG composite;
- `PRW-BIL1`, `PRW-DDF1`, `PRW-RCM1`, `PRW-ISI1`, `PRW-SPU0`, `PRW-SPU1`,
  `PRW-VAR1`, and `PRW-REV1`: evidence-bilattice merge semantics, selective
  deference, propagation deflection,
  lineage-idempotent recurrent masks, paired invariance/intervention tests,
  flat-reduction and heterogeneous-subplane controls, variational method
  selection, and correction-reversible culling; and
- `PRW-OBS1`, `PRW-COA1`, `PRW-CTX1`, `PRW-SPU1-TRANSPORT`, and
  `PRW-EK7-COALITION`: off-span support, lifted interaction support,
  context-compatible composition, coordinate-transport confounding, and an
  optional matched-token coalition-RAG ablation. These now have bounded
  synthetic implementation/sanity artifacts, but no candidate-survival or
  production evidence.

Resume RB-13 only after honoring the preservation/publication order above:

1. Freeze separate SELECT/REPORT protocols for `PRW-EK0`, `PRW-BIL1`,
   `PRW-SPU0`, and exact-small `PRW-VAR1`. A protocol must preserve the
   Section 1.1 dependency table and its card's resource ceiling.
2. Treat Wave 0B's `PRW-OBS1` and `PRW-COA1` harnesses as implemented sanity
   guards, then freeze their candidate-survival contrasts separately after
   Wave 0A and the required predecessor reviews.
3. Only after the relevant protocol and preflight are frozen, run one card at a
   time; report the entire scout grid and a disjoint confirmation set. Do not
   treat the current synthetic artifacts as production or novelty evidence.
4. Preserve JO1's stacking closure, G2's necessary-interaction/static-null
   disposition, and the existing CRSV/SRS-1 diagnostics as
   predecessor controls. Constant masks, fixed projectors, flat
   concatenation, principal angles, signed interference, commutators,
   finite-horizon gain, and scale-local peaks are not new RB-13 mechanisms.
5. Never count repeated walks as independent evidence, scalar-cancel
   support/refutation, or let query-time masks mutate stored evidence.
6. Do not run `PRW-EK7` unless every required component has an explicit
   disposition and the process-tree RSS/checkpoint/kill-resume guard is green.
7. Keep `PRW-SPU1-TRANSPORT` a confound control and
   `PRW-EK7-COALITION` an optional survivor ablation; neither may promote or
   block an unrelated component by association.

### RB-15 nonlinear-neutraliser transfer status

The supplied Frances Fulton presentation has been traced to the 2025
single-neutraliser journal model and reviewed in full. The exact source audit,
formal graph-sidecar transfer, prior-art collision map, and frozen first-run
fixture are in
`docs/research/nonlinear-neutraliser-subspace-transfer-2026-08.md`.

`PRW-RCM1-NLN` is an optional `PRW-RCM1` subcell, not a new promoted theory
family. Candidate-survival work remains blocked on `PRW-RCM1` and
`PRW-ISI1`. The CPU-only Stage-A mathematical sanity run is complete for run
IDs 7 and 11. Both exact artifacts retain one frozen failure: the high-drive
forward scalar maximum is boundary-censored at 1.60, so bidirectional hardening
is unresolved. Linear, sampled-storage, protected-channel, co-location
equivalence, RSS, and wall gates passed; scientific and novelty claims remain
`UNCONFIRMED`.

The promising descriptive lead is distributed self-grading: identical
attachments experienced materially different local amplitudes, phases, and
effective stiffness, while the distributed graph cell was less branch-sensitive
than the single/co-located controls. This is not yet attributable to
nonlinearity because the frozen grid lacked a distributed-linear control.

Resume order for RB-15:

1. Preserve and verify both manifests under
   `artifacts/method-dev/rb15-nonlinear-neutralizer/`; do not widen or rerun the
   completed frozen grid as if it were confirmatory evidence.
2. Complete the independent adversarial review of implementation, artifacts,
   and the retained negative result.
3. If a new synthetic run is later authorized, preregister a distinct
   distribution-by-nonlinearity factorial with matched distributed-linear
   control and aggregate/worst-case/hysteresis endpoints before output.
4. Do not open Stage B until `PRW-RCM1` and `PRW-ISI1` have explicit
   dispositions and a separate SELECT/REPORT protocol is frozen.

The mathematical disposition is already clear enough to prevent one category
error: Gelfand/Fomin and classical Euler-Lagrange are valid for a declared
smooth continuum functional. A finite graph/sheaf energy has an ordinary
finite-dimensional stationarity equation. Neither supplies a whole-system
optimizer for binary masks, top-k selection, graph rewiring, variable
dimensions, or lattice-valued epistemic state without the relaxation and
integrality/rounding-gap audit in `PRW-VAR1`.

## Disposition — living / annealed post-bank corrector (H5)

**SUPERSEDED METRIC EVIDENCE — STILL NON-PROMOTED.** The July 2026
metric-lineage audit found that the legacy nDCG helper forms IDCG from
retrieved relevance instead of all positive qrels. H5's exact values and
comparator ordering are therefore `LEGACY_METRIC_LINEAGE_BLOCKED` and are not
accepted scientific evidence until regenerated. See
`docs/research/latent-option-value-audit-2026-07.md`, Section 3.1, and
`docs/research/metric-lineage-repair-protocol-2026-07.md`. The existing
non-promotion remains the conservative disposition; this note does not reopen
or promote H5.

The living-bank / annealed-post-bank corrector line was parked per its own
preregistered H5 gate: C5 (living) had to beat **both** C5s (frozen static
bank) and C5r (one-shot router). These are quarantined stored campaign values,
not current accepted metrics (query-encoder-swap arena, cycles 12, seeds
[42,1337,7]):

| Dataset | C5 living | C5s static | C5r one-shot | C5 > C5s | C5 > C5r | LIVING BANK WINS |
|---|---:|---:|---:|:---:|:---:|:---:|
| SciFact | 0.131135 | 0.131135 | 0.180862 | False | False | **False** |
| NFCorpus | 0.046389 | 0.046389 | 0.045817 | False | True | **False** |

Sources: `docs/drift-recovery-post-bank-headtohead-results-2026-06.md`,
`docs/drift-recovery-post-bank-headtohead-nfcorpus-results-2026-06.md`.
Do not treat the legacy ordering as proof that C5s is the confirmed honest
baseline. The line remains closed and non-promoted because it has no valid
positive evidence, **not** because the quarantined comparator means have been
reconfirmed. Corrected metric migration, regeneration, and supersession
provenance are Carried Debt CD-MLR-01 / Deferred Scope DS-MLR-01. Adjacent H4
exact nDCG values are quarantined by the same lineage audit; compounding remains
non-promoted pending any separately justified corrected regeneration.

## Aggregate BHS trend

Track `BHS_OFFICIAL` per merged PR over the last 5 cycles. Falling trend = the
remediation loop is succeeding. Flat-at-100 trend = either real progress or the
loop is being gamed (Tier B agents not adversarial enough). Spot-check the
Tier B reports if the trend looks suspicious.

| Cycle | PRs merged | Avg BHS_OFFICIAL | Avg LOOP_ITERATIONS | OPERATOR_OVERRIDE count |
|-------|------------|------------------|---------------------|------------------------|
| _current_ | _—_ | _—_ | _—_ | _—_ |

## Operator overrides log

Every PR merged at `BHS_OFFICIAL < 100` (i.e. with `OPERATOR_OVERRIDE:`
populated) gets a permanent row here. The override creates an automatic top-
priority Carried Debt entry; this log is the audit trail.

| PR | BHS_OFFICIAL at merge | Override reason | Override author | Out-of-band ref |
|----|----------------------|-----------------|-----------------|-----------------|
| #244 | 55 | reconciliation foundation must land so follow-up cycle can implement CD-244-01..05 against canonical main | mattmre | `docs/next-session.md` Carried Debt CD-244-01..05 |
| #245 | 85 | three Tier B iterations converged on rubric-depth gameability (length-based not semantic); per §6.1 same-gap-2-iterations rule, escalating to Tier C as CD-245-01 rather than looping further | mattmre | `docs/next-session.md` Carried Debt CD-245-01 |

---

**Last session**: 2026-07-27 — completed the lossless primary/residual audit. The exact 38-path evidence/source allowlist is local at `186590c8`; pre-reconciliation durability anchor `7dec564d` and a verified full-history bundle protect the primary line, while this later documentation-only checkpoint is recorded post-commit in the private recovery index. Private V2 binds 2,584 ignored/untracked files, 1,678,118,521 bytes, ten total worktrees, two stashes, and no cleanup authority; exact archive refs/bundles and verified evidence/source ZIPs close the immediate local-loss gaps. PR #292 `835f6199` failed at Tier B 70/Critical; replacement `f2e41d42` now covers 113/113 affected artifacts, preserves all 103 historical JSON/PNG bytes, passes 28 hostile validator cases, and is independently recovery-bundled. Fresh review found no local artifact blocker but retained BHS 70/Critical because public #292 remains at `eb750958` with stale invalid claims and no `f2e41d42` hosted runs. PR #293 `454e4a32` failed at 70/Critical after six iterations and current #295 `730b305e` failed at 60/Critical after five; both are archive-ref preserved and must be withdrawn after explicit approval, not repaired again. PR #294 `6e78cf41` reached 100 only at its original stacked/pre-transplant head and requires a new transplant branch/replacement PR plus fresh review; do not force-push or retarget public #294. Nothing was pushed, publicly edited/closed/opened, merged, deleted, pruned, restored, reset, or cleaned. Prior scientific disposition remains: T1R is an internal exact-Hamming theorem, current float-FFT production linkage failed, JO1 stacking is closed by flat identity, A1/G2H are conditional, and G2A is blocked.
**2026-05-17**: PRs #249–#254 merged; 9 BHS Scope B audit Carried Debt rows (CD-MOD-001 through CD-TTS-002) closed.
**Last validated by `check_block_flag.py`**: 2026-07-27 local bookkeeping validation; `CLEAR`, three first-cycle open carried-debt rows (`CD-MLR-01`, `CD-R13-01`, `CD-R16-01`), none expired.
