# ARCH-AEP Remediation Sweep — 2026-06-03 (Turns 1-9 execution)

## 9-Turn Execution Plan

1. Establish controlled context (branch, working tree, script/tool availability).
2. Capture authoritative debt posture (`check_block_flag`, `next-session` parsing, schema drift checks).
3. Run production floor gates (`smoke_pipeline.py`, `test_aep_orchestrator.py`, test suites with high signal).
4. Execute shim evidence hardening command set (`verify_shim_development.sh`) and capture artifacts.
5. Validate model-scope and TTS runtime seams (`tests/test_model_scope_runtime.py`, `tests/test_antigravity_model_scope_tts.py`, shim evidence scripts).
6. Run focused structural scans for high-risk code smells (`broad except/pass`, optional dependency paths).
7. Run regression/diagnostic diagnostics (`run_live_fire_diagnostics.py`).
8. Cross-check process-tracking artifacts for drift (cycle pointers, tracker references).
9. Produce consolidated findings, risk-ranked remediation plan, and roadmap recommendations.

## Turn Execution Matrix

| Turn | Objective | Evidence | Outcome |
|---|---|---|---|
| 1 | Confirm branch and working tree | `git branch --show-current`, `git status --short` | PASS — branch `aep-arch-aep-remediation-turns9-exec-20260603` active; repo writable for edits/checks |
| 2 | Capture debt/state | `python scripts/check_block_flag.py`, `python scripts/check_block_flag.py --allow-debt-prs`, `python scripts/validate_v33_schema_drift.py` | PASS — gate open/closed mechanics valid; state: `BLOCKED` with 8 open / 5 blocking-YES rows |
| 3 | Run primary floor gates | `python -m unittest -q tests/test_e2e_smoke.py`, `python -m unittest -q tests/test_check_block_flag.py`, `python -m unittest -q test_aep_orchestrator.py`, `python -m unittest discover -s tests -p "test_*.py"` | PASS — all targeted suites pass (with module-path correction for `unittest` when needed) |
| 4 | Run shim integrity sweep | `bash scripts/verify_shim_development.sh` | PASS — 20 shim tests + 4 evidence artifacts, all passing; gate remains blocked due open debt |
| 5 | Validate runtime seams | `python -m unittest tests/test_antigravity_model_scope_tts.py tests/test_model_scope_runtime.py tests/test_shim_inference_evidence.py ...` | PASS — regression tests for runtime/bridge/TTS seams are green |
| 6 | Structural health checks | `python scripts/smoke_pipeline.py`, `python scripts/bhs_validator.py --report` | PASS on schema and smoke floors; `bhs_validator` floor checks return mixed historical samples (informational) |
| 7 | Runtime diagnostics | `python run_live_fire_diagnostics.py` | PASS exit, with explicit warnings: `chelate_rate_outside_target_for_tiny_fixture`, `live_fire_fixture_is_saturated` |
| 8 | Process artifact drift check | manual inspection of `docs/next-session.md`, `docs/ARCH AGENTIC ENGINEERING AND PLANNING/tracker-pointer.md`, cycle trackers | PASS; mismatch confirmed and `tracker-pointer.md` updated to route live authority to `docs/next-session.md` |
| 9 | Consolidate findings + roadmap | report drafting | PASS |

## Findings (risk-ranked)

### 1) Blocking carried debt is still active (P1 / gate-critical)
`docs/next-session.md` still contains five blocking SHIM rows as OPEN (`SHIM-CD-01, 02, 06, 08, 09`), and `scripts/check_block_flag.py` reports:
- `Carried Debt row count (OPEN): 8`
- `Carried Debt row count (OPEN + Blocking YES): 5`

`check_block_flag.py` exit is `FAIL` as designed. This is the current merge gate.

### 2) Shim production substrate remains partially wired (P1)
`SHIM-CD-01` and `SHIM-CD-02` remain OPEN with required production seams still incomplete (see `docs/next-session.md` rows `SHIM-CD-01`, `SHIM-CD-02`).

### 3) 5-agent execution model still partial in production envelope (P1)
`SHIM-CD-06` remains OPEN: in-repo five-worker evidence exists, but legacy scheduler `019e669bf1bb` has no completed evidence closure. This directly conflicts with goal framing that implies broader 10-agent operationality.

### 4) Process-tier debt closure logic still non-resolved (P1)
`SHIM-CD-08` and `SHIM-CD-09` remain OPEN: they are now process/infrastructure controls rather than code defects but still block safe progression and indicate repeated “documentation + slice expansion” tension without closing the root substrate.

### 5) Live diagnostics show useful but non-generalizable fixture behavior (P2)
`run_live_fire_diagnostics.py` completed successfully, but warns:
- chelate-rate control is outside target for tiny fixture,
- fixture saturation in live-fire guidance.
This means current diagnostics are healthy as a smoke, not a production-representative signal.

### 6) Architecture process drift (P2, partially remediated)
`docs/ARCH AGENTIC ENGINEERING AND PLANNING/tracker-pointer.md` previously pointed to a stale tracker (`cycles/2026-05-01`) while open debt and current work is carried in `docs/next-session.md`; this mismatch created governance ambiguity for new operators. Pointer now explicitly delegates to `docs/next-session.md`.

## Remediation Queue (next 9 turns if extended)

1. **Close SHIM-CD-01 first**: complete production SIP coverage in VectorSteerer/Antigravity/TTS seams (`clear_signals`, steering policy, self-healing hooks, rollback fixture).
2. **Close SHIM-CD-02**: finish shim primitive promotion surface (`shim_node_promoted` parity with runtime-critical modules, remove research-only leakage into production-paths).
3. **Close SHIM-CD-06**: run and evidence the external scheduler path (`019e669bf1bb`) end-to-end; record worker-count and successful completion proofs.
4. **Automate Blocked debt aging**: add optional TTL validation in `check_block_flag.py` (or explicit `session_wrap_age_days` enforcement) to reduce manual process drift.
5. **No-doc progress gate**: for BHS loop steps claiming progress, require at least one runnable code-path test in `tests/` and at least one production artifact under `artifacts/`.
6. **Align architecture docs**: create/update active tracker for the SHIM workstream and update `tracker-pointer.md` atomically when opening/closing debt-heavy cycles.
7. **Stabilize scheduler evidence**: publish a canonical “scheduler truth” artifact per cycle (command line, worker set, output digest, artifact hash) and make it a required checklist item before adding any new cycle rows.

## Suggested Architecture-Management Posture (Senior-Architect view)

- Treat SHIM as a **single release-tracked substrate goal**: no further new research slices until `SHIM-CD-01/02/06/08/09` are closed or formally deferred with a date-bound scope reduction.
- Require every PR that changes architecture semantics to touch one of: `docs/next-session.md`, a verification script, and a production-backed test.
- Preserve evidence integrity by keeping runtime evidence and process evidence as separate, explicit artifacts; block PRs that only update process docs.
- Keep 10-minute/BHS loop gates strict: if `check_block_flag` is `BLOCKED`, scope for the next tranche is debt-draining only.
