# Merge readiness — remediation branch → `main`

**Date:** 2026-06-03  
**Source branch:** `aep-arch-aep-remediation-turns9-exec-2-20260603`  
**Target:** `main` (`3595283` — feat(opsd) #255)  
**Reviewer:** pre-merge pass on current working tree (committed + uncommitted)

---

## Executive summary

| Area | Status | Notes |
|------|--------|--------|
| Merge conflicts (committed only) | **None expected** | `git merge main` → already up to date on branch tip |
| Merge gate (`check_block_flag`) | **PASS (CLEAR)** | 8 open SHIM rows, **0** `Blocking=YES` (deferred per operator reprioritize) |
| Production code remediation | **Uncommitted** | **41** modified tracked files — **must be committed** before merge or they stay local only |
| Committed tip vs `main` | **1 commit** | Mostly shim **research/docs** (~168 files, +24k lines), not the runtime remediation |
| Untracked noise | **~6.6k files** | ~6.3k under `artifacts/phase_loop/reports/` — **do not commit**; add to `.gitignore` |
| Verification (this tree) | **Green** | See checklist below |

**Bottom line:** Merging today without a remediation commit would land **documentation + loop artifacts policy**, not the **model-scope / TTS / block-flag / test** fixes sitting in the working tree. Stage a deliberate commit (or split commits) first, then merge.

---

## What is on the branch today

### Already committed (`4379b9c` vs `main`)

- Shim loop program docs, phase plans, 10-agent wave artifacts, `OPERATOR_OVERRIDE` delegation notes.
- `long_running_orchestrator_stub.py` v0.2 (auto-continue / wall gate).
- **No** production edits in that commit message scope (“Research-only spike. 0 prod edits”).

### Not committed (actual remediation — merge-critical)

**41 tracked files**, including:

- **Runtime / engine:** `model_scope_runtime.py`, `model_scope_engine_bridge.py`, `model_scope_steering.py`, `tts_pipeline.py`, `antigravity_engine.py` (if touched in index), `steering_policy.py`, `self_healing_chelation.py`, …
- **Gate / governance:** `scripts/check_block_flag.py`, `docs/next-session.md`, `tests/test_check_block_flag.py`
- **Tests:** `test_model_scope_*.py`, `test_tts_pipeline.py`, `test_computational_storage_poc.py`, `tests/test_e2e_smoke.py`, …

### Untracked (selective commit)

| Keep for merge | Skip / gitignore |
|----------------|------------------|
| `scripts/phase_development_loop.py`, `scripts/verify_shim_development.sh`, `scripts/loop_*.sh`, `scripts/run_10min_priority_bhs_loop.py`, shim record/promote scripts | `artifacts/phase_loop/reports/**` (6318 files) |
| `docs/ROADMAP_EXECUTION.md`, `docs/loop_workers/`, `chelated_shim_research.py`, `shim_node_promoted.py`, `tests/test_shim*.py` | `artifacts/bhs_10min_loop/workers/**` (271 files) |
| Selected evidence JSON (1–2 canonical, not every cycle) | `tmp_agent5_mtp_smoke.py`, `synthesis-research-only/` unless product asks |

---

## Verification run (current tree)

```text
python3 scripts/check_block_flag.py          → PASS (CLEAR)
python3 -m unittest tests.test_check_block_flag tests.test_e2e_smoke  → 23 OK
python3 -m unittest discover -s tests -p 'test_shim*.py'               → 22 OK
bash scripts/verify_shim_development.sh      → SHIM verification complete
python3 scripts/smoke_pipeline.py            → PASS (floor + ceiling)
python3 -m unittest test_model_scope_runtime test_model_scope_engine_bridge \
  test_model_scope_steering test_tts_pipeline                             → 245 OK (2 skipped)
```

**Warnings (non-blocking):**

- Smoke: adapter checkpoint shape mismatch log (384 vs 768) — ceiling path still passes with new adapter.
- Live-fire: tiny-fixture chelate-rate warnings (documented in remediation report).

---

## Alignment with operator policy

- `docs/next-session.md`: **CLEAR**; SHIM-CD-01/02/06/08/09 **OPEN**, `Blocking=NO`, **deferred last**.
- `docs/ROADMAP_EXECUTION.md`: core queue steps 1–8 before shim; use `loop_core_10m.sh` not shim BHS loop by default.
- Old remediation report (`reports/arch-aep-remediation-sweep-20260603-turns9-execution.md`) still describes **BLOCKED + 5 blocking YES** — **stale** relative to current `next-session.md`; do not use it as merge gate truth.

---

## Risks if merged as-is

1. **Lost remediation** — 41-file diff never reaches `main`.
2. **Artifact repo bloat** — accidental `git add artifacts/` adds thousands of turn reports.
3. **Policy regression** — merge only the doc commit; another agent re-flips SHIM rows to `Blocking=YES` without `ROADMAP_EXECUTION` context.
4. **Dual automation** — restart `phase_development_loop` + `loop_10m` on `main` without `CHELATED_EXEC_TRACK=core` → shim-first behavior returns.

---

## Recommended merge procedure

### 1. Hygiene (before commit)

```bash
cd /home/mattmre/CHELATEDAI

# Confirm branch
git branch --show-current   # expect aep-arch-aep-remediation-turns9-exec-2-20260603

# Optional: append to .gitignore (if not already)
#   artifacts/phase_loop/reports/
#   artifacts/bhs_10min_loop/workers/
#   artifacts/arch_aep_turns/

python3 scripts/check_block_flag.py
python3 scripts/smoke_pipeline.py
bash scripts/verify_shim_development.sh
```

### 2. Commit remediation (suggested split)

**Commit A — production + tests (required):**

```bash
git add \
  scripts/check_block_flag.py docs/next-session.md tests/test_check_block_flag.py \
  model_scope_runtime.py model_scope_engine_bridge.py model_scope_steering.py \
  model_scope_features.py qwen_scope_adapter.py \
  tts_pipeline.py steering_policy.py self_healing_chelation.py \
  test_model_scope_runtime.py test_model_scope_engine_bridge.py test_model_scope_steering.py \
  test_tts_pipeline.py test_aep_orchestrator.py \
  # …other modified production/test files from git diff HEAD --name-only
git commit -m "fix(remediation): model-scope runtime, TTS seams, block-flag gate, and regression tests"
```

**Commit B — operator queue + automation (optional same PR):**

```bash
git add docs/ROADMAP_EXECUTION.md docs/loop_workers/ \
  scripts/phase_development_loop.py scripts/loop_core_10m.sh scripts/verify_shim_development.sh \
  chelated_shim_research.py tests/test_shim*.py scripts/record_shim_*.py
git commit -m "chore(exec): core-first roadmap, phase loop, and shim evidence tooling"
```

Do **not** `git add artifacts/phase_loop/reports/` unless you intentionally want turn history in git.

### 3. Merge to `main`

```bash
git fetch origin
git checkout main
git pull origin main
git merge aep-arch-aep-remediation-turns9-exec-2-20260603
# resolve conflicts if any (unlikely on committed tip; possible if main moved)
python3 scripts/check_block_flag.py
python3 scripts/smoke_pipeline.py
git push origin main
```

### 4. Post-merge (first hour on `main`)

1. Re-run `python3 scripts/check_block_flag.py` — expect **CLEAR**, 0 blocking YES.
2. Start **one** driver: `bash scripts/loop_core_10m.sh 10` **or** phase loop with core slices — not both.
3. Execute `docs/ROADMAP_EXECUTION.md` step **1** (ML projection) — first core slice not yet in `artifacts/phase_loop/state.json`.
4. Archive or delete local `artifacts/phase_loop/reports/` on dev machines (optional; keep out of git).

---

## Merge checklist (copy for PR description)

- [ ] Remediation commit(s) include the **41** modified production/test files (not docs-only tip)
- [ ] `check_block_flag.py` → **CLEAR** on merged `docs/next-session.md`
- [ ] SHIM rows remain **deferred** (`Blocking=NO`) until core queue step 8
- [ ] `smoke_pipeline.py` PASS (floor + ceiling)
- [ ] `verify_shim_development.sh` PASS (if shim tooling merged)
- [ ] No bulk commit of `artifacts/phase_loop/reports/`
- [ ] `docs/ROADMAP_EXECUTION.md` present on `main`
- [ ] CI / unittest path matches `CLAUDE.md` (unittest, not pytest-only)

---

## After merge — next work (core track)

Per `docs/ROADMAP_EXECUTION.md`, in order:

1. ML projection / InfoNCE correctness + tests  
2. Infra hygiene (`isolated_adapter`, `*.pt` gitignore)  
3. Sweep/CI cost, packaging, docs truth  
4. Model-Scope shadow pilot  
5. E2E learning-loop test  
6. **Last:** SHIM substrate (re-enable `Blocking=YES` when re-entering program)

---

*Generated from pre-merge review on branch `aep-arch-aep-remediation-turns9-exec-2-20260603`.*