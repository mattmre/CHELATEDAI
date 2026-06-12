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

**Current**: `CLEAR` — operator reprioritization 2026-06-03: active work follows `docs/ROADMAP_EXECUTION.md` (ML correctness → infra → Model-Scope → E2E). SHIM-CD-01, 02, 06, 08, 09 remain **OPEN** but **deferred last**; resume shim program only after queue step 8 and those rows are re-entered as `Blocking=YES`. SHIM-CD-05 **CLOSED**. Re-run `python scripts/check_block_flag.py` after any session-wrap edit.

**Prior state (pre-Cycle 4 D transcription)**: `CLEAR` (only because SHIM-CDs were absent from this table despite repeated "mandatory" declarations in dashboard/cycle summaries).

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
| CD-247-01 | `aep_orchestrator.py:678, 966` carry inline `# BHS v3.3 placeholder hook` / `# BHS v3.3 placeholder (to be expanded)` comments; PR #245 replaced the underlying validator with a real implementation but the comments now misrepresent the integration state. Stale-docs L13-light. | post-consolidation audit (2026-05-16) | 1 cycle | NO — docs drift, no behaviour change | **CLOSED** (2026-06-02) — comments refreshed to state `validate_pr_brutal_honesty` and `run_smoke_pipeline` are live integrations |
| CD-247-02 | Six broad-`except Exception` swallow sites across production code (`antigravity_engine.py:1080-1081, 2460-2461`; `run_phase_c_eval.py:433-434`; `checkpoint_manager.py:372`; `engine_scope_coverage.py:35`; `run_weight_refinement_campaign.py:443/589/626`) silently absorb errors with `pass` or `return None`. Per L11, even benign optional-path swallows should at minimum log at debug. | post-consolidation audit (2026-05-16) | 1 cycle | NO — audited as deliberate non-critical paths | **CLOSED** (2026-06-02) — re-audit: `run_phase_c_eval.py` reformulate + centroid paths now log at DEBUG; `antigravity_engine.py` TTS dashboard path logs DEBUG + warn; remaining cited lines are narrow `TypeError`/`ValueError` sorts or control-flow `pass`, not silent `Exception` swallows |
| CD-MOD-001 | MOD-1 audit: `model_scope_runtime.LocalModelRuntime.load()` real `transformers.AutoModelForCausalLM.from_pretrained` branch has zero test coverage; every test injects a `MagicMock` loader. The `Qwen3.5-9B` pilot load (Phase 1 AC1) is unverified end-to-end. L5 + L8. | BHS Scope B audit 2026-05-16 | 1 cycle | NO — Model-Scope cycle phase scored 72; tracked under cycle-wide remediation | **CLOSED**: PR #250 — `ActivationEvent` gains `raw_tensor_shape` field; `TestRunInferenceTryBranchDiscrimination` covers real try-branch via sentinel patch; integration test added under `@skipUnless(CHELATED_INTEGRATION_MODEL)` for actual `AutoModelForCausalLM.from_pretrained` path with `sshleifer/tiny-gpt2` |
| CD-MOD-002 | MOD-2 audit: `qwen_scope_adapter.py` has no `hf_hub_download`, no HuggingFace repo string, no checksum; every SAE test uses `np.random.default_rng(42).random(...)` as the weight matrix. `QwenScopeAdapter.extract_features` projects only 3-4 scalar activation stats, not residual-stream tensors. Phase 2 AC ("Qwen3.5-9B sparse features from hooked residual states") not demonstrated. L4 + L1 + L5. | BHS Scope B audit 2026-05-16 | 1 cycle | NO — critical-severity but POC-bounded; honestly disclosed | **CLOSED**: PR #250 — `QwenScopeLayerSAE.from_file` real checkpoint path covered by `TestQwenScopeLayerSAEFromFile` (creates synthetic `.pt` fixture via `torch.save`, loads, calls `encode`, asserts output shapes and top-k sparsity); `extract_features()` docstring explicitly discloses operation on activation STATISTICS, not raw tensors (L13 resolved) |
| CD-MOD-003 | MOD-3 audit: `model_scope_steering` `InterventionRecord` lives in `self._records: List[...]` only; no `persist()`/`to_disk()` method. On fresh checkout all provenance is lost. Phase 3 AC3 ("provenance records show what feature, layer, and policy caused intervention") half-met: structure exists, durability does not. L4. | BHS Scope B audit 2026-05-16 | 1 cycle | NO — cosmetic vs durability | **CLOSED**: PR #254 — `SteeringActuator` gains `persist_records(path)` / `load_records(path)` (JSON Lines); provenance survives process restart; 10 new tests including fresh-instance round-trip |
| CD-MOD-004 | MOD-6 audit: `model_scope_engine_bridge.py:61` constructs `SteeringActuator(registry, max_total_interventions=0)`; intervention count is hardcoded to never exceed 0 regardless of `enable_steering`. Dashboard "intervention evidence" panel can never display anything other than 0. L4. | BHS Scope B audit 2026-05-16 | 1 cycle | NO — one-line fix; structurally constrains acceptance criterion #3 | **CLOSED**: PR #254 — `ModelScopeBridgeConfig.max_total_interventions: int = 100` added and wired to `SteeringActuator`; hardcoded `0` removed; two runtime-behavior tests guard against regression |
| CD-MOD-005 | MOD-5 audit: `OverlayTrainer.evaluate_promotion(baseline_events, candidate_events)` evaluates on training-data pairs (`all_inputs`/`all_targets`); `baseline_score` is comparator-pass-rate with default `threshold=0.0` (both sides trivially pass); no `stress` references in trainer or campaign. `rollback()` zeros in-memory weights only — never restores a prior promoted overlay file. Phase 5 AC3 ("promoted artifacts beat baseline under replay and stress checks; failed candidates roll back cleanly") not met. L4 (×3). | BHS Scope B audit 2026-05-16 | 1 cycle | NO — critical-severity but does not block running campaigns | **CLOSED**: PR #254 — `promote_candidate()` writes `.backup` before overwriting overlay; new `rollback()` restores from backup; `run_campaign()` splits episodes 80/20 before training and passes held-out eval set to `evaluate_promotion()` |
| CD-ENG-001 | ENG-5 audit: `run_golden_default_autopilot._recommendation` hardcodes `default_change_allowed: False` and `safe_default_holds: True`; no code path emits "supported candidate" or "documented no-promotion" terminal artifact. `evidence_contract`/`promotion_contract`/`compute_budget_policy`/`evaluator_fabric` are not imported by the supervisor. Phase 5 AC3 unimplemented. L4 + L13. | BHS Scope B audit 2026-05-16 | 1 cycle | NO — supervisor produces ongoing recommendations; gap is terminal artifact | **CLOSED**: PR #251 — `_recommendation()` now sets `default_change_allowed = bool(reform_candidate or mask_candidate)`; `main()` writes `terminal_decision.json` artifact after loop exits; `test_main_writes_terminal_decision_json` verifies all required fields |
| CD-ENG-002 | ENG-4 audit: `engine_scope_negatives` "clustering" is dict-bucketing on exact-string signature equality, not a real distance/cluster algorithm; module hard-codes `generator` field but has no synthesis path (only mining); no committed `golden_runs/` artifact and no replay-twice-and-diff determinism test. Phase 4 ACs met only via mining path; the "synthetic" framing is prose-soft. L13 + L5. | BHS Scope B audit 2026-05-16 | 1 cycle | NO — mining-only is acceptable per AC; framing is the gap | **CLOSED**: PR #251 — `mine_hard_negative_families()` docstring explicitly names algorithm "deterministic fault-class grouping" (not clustering); `test_build_hard_negative_replay_artifact_is_deterministic` calls builder twice and asserts identical `family_id` assignments and row ordering; label-fix regression guard added |
| CD-TTS-001 | TTS-2 audit: grep of all 17 `run_*.py` runners for `enable_tts` or `--enable-tts` returns zero matches. TTSPipeline is library-callable only; no campaign runner can populate the dashboard TTS panel without external glue code. Visible-without-evidence pattern: panel + API exist, but no operationally-reachable activation path. L4 + L5. | BHS Scope B audit 2026-05-16 | 1 cycle | NO — short fix; dashboard already shows honest "not enabled" empty state | **CLOSED**: PR #252 — `--enable-tts`, `--no-tts-translation`, and `--no-tts-transport` flags wired into `run_road_course_campaign.py`; CLI wiring tested through real `main()` in `TestRunRoadCourseCampaignCLIWiring` |
| CD-TTS-002 | TTS-1 audit: REM-C2 (per-inference signal clearing in `tts_pipeline.py:213-218`) and REM-H2 (`FeatureDirectionBank` Gaussian unit vectors) are fixed in code but no test would fail if either were reverted. Two of the four bug classes that triggered the post-merge remediation wave can silently regress. L5. | BHS Scope B audit 2026-05-16 | 1 cycle | NO — code is fixed; regression coverage is the gap | **CLOSED**: PR #252 — regression tests added for REM-C2 (cross-inference signal-accumulation guard) and REM-H2 (Gaussian direction-bank distribution test); both tests would fail if the corresponding fixes were reverted |
| SHIM-CD-01 | CRITICAL: Shim Insertion Points (SIPs) in production hosts — partial 2026-06-03: `promoted_sip_apply()` at `VectorSteerer.steer` + `get_chelated_vector` when both envs on. Still unwired: steering_policy, self_healing, model_scope_*, block_graph; live-fixture rollback test. L4 partial. | BHS 5MIN Shim Loop + 2026-06-03 | — | YES | OPEN — on hold until core queue step 8 |
| SHIM-CD-02 | CRITICAL: Shim primitives promotion — partial 2026-06-03: `shim_node_promoted.py` at repo root when `CHELATED_SHIM_PROMOTED=1`. Collapse/MockMTP remain research-only. | BHS 5MIN Shim Loop + 2026-06-03 | — | YES | OPEN — on hold until core queue step 8 |
| SHIM-CD-03 | IMPORTANT: All MTP Shim Lookahead, cascade compounding, usage refinement, efficiency logic is pure simulation (MockMTPShimLookahead dict patterns, placeholder tokens, no real head, no OPSD trace consumption). L3 per self-disclosure. | BHS 5MIN Shim Loop Cycle-001 D (dashboard:166) + shim_collapse...py:52 + nomenclature + gap_audit:52 | 1 cycle | NO — explicitly L3-scoped in harness BHS NOTES | OPEN — unchanged; Mock only |
| SHIM-CD-04 | IMPORTANT: Companion tests for shim — 2026-06-03: `tests/test_shim_*`, `test_shim_inference_evidence.py`, `test_shim_promoted_probe.py`; `bash scripts/verify_shim_development.sh` runs full gate including `run_inference` path. 10+ open TODOs in research artifacts remain. L5 partial. | BHS 5MIN Shim Loop + 2026-06-03 | 1 cycle | NO — research harness scoped | OPEN — prod + inference + promoted-probe tests; artifact TODOs remain |
| SHIM-CD-05 | CRITICAL: Production-path shim evidence — steer, TTS intercept, and `AntigravityEngine.run_inference` + `enable_tts` under `CHELATED_SHIM_RESEARCH=1` via `record_shim_*_evidence.py` + `tests/test_shim_inference_evidence.py`. | BHS 5MIN Shim Loop + 2026-06-03 | 1 cycle | YES | **CLOSED** (2026-06-03) — `record_shim_inference_evidence.py` writes `artifacts/bhs_shim_evidence_inference_*.json` with `research_shim_guard`; unittest guards script |
| SHIM-CD-06 | CRITICAL process: 5-agent model — partial 2026-06-03: in-repo `run_five_worker_shim_gate.py`. External scheduler 019e669bf1bb still unverified. | BHS 5MIN Shim Loop + 2026-06-03 | — | YES | OPEN — on hold until core queue step 8 |
| SHIM-CD-07 | IMPORTANT: BHS Research Program Score (shim workstream) showed 0 delta (35→22/100 flat) despite "first official cycle" + subsequent claims. No quantified self-improvement on any goal §77-83 metrics (SIPs=0, token acct=0, MTP=N/A, L4 risk reduction=0, benchmark families=0, cascade traces=0). L13 soft-prose vs reality. | BHS 5MIN Shim Loop Cycle-001 D (dashboard:170) + all cumulative sections + Cycle 2/3 reflections | 1 cycle | NO | OPEN — program score static post-transcription |
| SHIM-CD-08 | CRITICAL: Remediation loop Tier C / next-session.md transcription — SHIM rows maintained; block gate now operator-driven per execution queue. | BHS 5MIN Shim Loop + 2026-06-03 | — | YES | OPEN — on hold until core queue step 8 |
| SHIM-CD-09 | CRITICAL process: 10th cycle (Cycle-010 Agent 10 Integrator meta + background) of doc-only slice additions (min-max backlog #9/#10 inserted in goal:115-168 + #10 at 109-110; "Loop 10 Comparative Analysis Slice" + MinMax MSA vs SE-RDAG tables + min-max adaptation pseudocode + BHS disclosures appended to research_plan:199-272; "successful 10-agent model use" + Cycle-010 row + 4Q + §4 BH appended to dashboard:956-992; consolidated bhs_10agent_integrator_evidence_Cycle-010-20260527.json) **while core backlog #1 (first real minimal SIP wiring into any prod host) remains 0%** (fresh grep: 0 prod references outside 2 research artifacts/ files; tts_pipeline.py:47-80 / antigravity_engine.py:2452-2600/2566-2600 SIP seams all Wired=NO per A matrices + reconfirm; 10 cycles). 5-vs-10 narrative gap (goal:7/34/130/213-230 "Exactly 10 (A–J)" / "10-agent model begins with Cycle 009" / "successful use" vs scheduler 019e669bf1bb "still dispatches 5" + all prompts/history/009/010 reality = 5 or 0 execution; L4 + L13 core per 01/04/09_cycle009 audits + dashboard:3/10/31/973 + plan:268 + json:58/59) unclosed. §128 termination condition ("3 consecutive cycles with BHS Cycle Score < 60" — now 10x; avg ~4-10/100; 0 substrate deltas on §77-83; 0 SIPs; 0 SHIM-CD closures post-transcription) exceeded 7x+ with explicit repeated PAUSE/TERMINATE recs in every E/D/Agent9/0200 output ignored. Pattern of "adding more slices while core #1 0%" executed despite goal:157 explicit "risks further L9/L4" + Agent J role mandate to audit it as process L4. L4 (10-agent "successful" framing for 1-agent doc edits + "integrated outputs" / "Loop 10 slice" on 0 substrate) + L9 (doc-as-impl on "meta-work" + "remediation" + "model fidelity" while OPEN SHIM 01-08 + BLOCKED persist; 10-cycle transcription L9 escalation) + L13 (soft-prose "mechanical 10-agent" / "self-improving engine" / "successful use" vs runtime scheduler/prompts/0 tasks/0 fidelity/0 deltas + prose-vs-artifact drift on Cycle-010 "integration"). | Cycle-010 Agent 10 json:1-62 + dashboard:956-992 + goal Model Change Log:213-230 + research_plan:199-272 (new section) + 01_cycle009_audit.md:23/110/193 + 04_cycle009_d_audit.md:58/61/93/121/151 + 09_cycle009_agent9_bhs_compliance_audit.md:57/72-75/111/127/149/177 + BHS_5MIN_SHIM_LOOP_GOAL.md:157/166 + fresh prod-excluded grep (0 SIPs) + next-session:61-68 + check_block_flag.py:108-109 + Agent 8 audit 08_cycle010_agent8_bhs_process_gap_audit.md | — | YES | OPEN — on hold until core queue step 8; historical audit preserved |

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
| _none yet_ | _—_ | _—_ | _—_ |

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

**Last session**: 2026-05-16 — PR #244 reconciliation merge (BHS_OFFICIAL=55, OPERATOR_OVERRIDE)
**2026-05-17**: PRs #249–#254 merged; 9 BHS Scope B audit Carried Debt rows (CD-MOD-001 through CD-TTS-002) closed.
**Last validated by `check_block_flag.py`**: run after this commit
