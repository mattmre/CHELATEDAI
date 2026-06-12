# Cycle-010 Synthesis Prep — Apply Instructions for Main Orchestrator (BLOCKED/Research-Only)

**Prepared by**: Agent 10 (Synthesis Prep & Output Packager) for BHS Cycle 010 (10-agent, BLOCKED/research only). cwd=/home/mattmre/CHELATEDAI.
**Date**: 2026-05-27
**Purpose**: Provide ready-to-apply text (in sibling files) + **clear, ordered instructions** to main process (orchestrator / human operator) on **what to apply and in what order to avoid blocks, conflicts, or L9/L4 violations**. All under brutal honesty v3.3 + CLAUDE.md (evidence rule, visible=verified, mandatory §4 BH).
**Key Constraint (from task + evidence)**: This is **prep only**. No live main-process files were edited by this dispatch. Prior Integrator dispatch (see artifacts/bhs_10agent_integrator_evidence_Cycle-010-20260527.json) already performed 3 direct search_replace on research docs. **Do not re-apply overlapping edits.** Use this for consolidation, new Cycle summary md landing, table row hygiene, or archival only after fresh verification. **BLOCKED state + 10 cycles 0 substrate + §128 active**: any application must not imply progress or reset debt.

**CRITICAL PRE-APPLY GATES (Mandatory — Evidence Only; Run These First on Fresh Checkout)**
1. `python -B scripts/check_block_flag.py 2>&1 || true` → Must output BLOCKED + FAIL + "Carried Debt row count: 2". (Unchanged.)
2. `grep -r --include="*.py" "ShimNode\|min_max_shim_adapt\|MinMax MSA vs SE-RDAG" --glob='!**/docs/**' --glob='!**/artifacts/bhs_*.json' | cat` → Must return 0 hits (prod isolation; only research shims with L4 guards at shim_node.py:34-36).
3. `ls docs/steering_chelation_rag_dag_research/loop_02/ | grep -E '010|Cycle-010'` → Must be empty (no independent agent mds landed for 010).
4. Read key artifacts (confirm pre-state):
   - `artifacts/bhs_10agent_integrator_evidence_Cycle-010-20260527.json` (note its own "brutal_honesty_this_artifact": meta only, 0 SIPs, does NOT satisfy goal #1, 10/100 flat, §128 rec).
   - `docs/steering_chelation_rag_dag_research/artifacts/BHS_SHIM_LOOP_DASHBOARD.md` (Cycle-010 narrative already at 956-993; table ends at 009).
   - `docs/steering_chelation_rag_dag_research/STEERING_CHELATION_RAGDAG_MICROSLM_RESEARCH_PLAN.md` (new section at ~199+ already present from Integrator).
   - `docs/steering_chelation_rag_dag_research/BHS_5MIN_SHIM_LOOP_GOAL.md` (backlog #10 at 109 already present).
   - `docs/next-session.md` (SHIM-CD-01-08 OPEN 61-68).
5. Re-run 009/010 baseline smoke (per json + cycle_0300.md): harness command must produce bitwise identical metrics to prior (no 010 deltas).
6. Full list_dir/grep on steering/artifacts/ + loop_02/ + root/artifacts/ confirming only the 1 meta json + prior 009 files + this temp synthesis-research-only/ (if present).
**If any gate fails or state differs**: ABORT. Re-poll. Do not apply. This enforces "only synthesize post-verification" + evidence rule.

**What the Drafts Contain (Sibling Files in This Dir)**
- `Cycle-010-summary-draft.md`: Full ready-to-apply summary md (modeled exactly on artifacts/cycle_20260527_0300.md pattern for 009: verification polls gate, reality, score, deltas (0s + meta), 4Qs verbatim grounded, EVIDENCE list, SMOKE, full §4 BH with file:line + L taxonomy + §128). Includes BHS self-draft for the wave at end. **This is the "New cycle summary md (Cycle-010)"**.
- `dashboard-row-draft.txt`: Compact markdown table row text (ready for insertion into BHS_SHIM_LOOP_DASHBOARD.md historical table after Cycle-009 row). Includes <!-- comments --> with apply notes + self-draft.
- This file: apply-instructions (order, warnings, BHS self-draft for the prep wave itself).
- (Optional future: any safe-edit-drafts/ subdir for non-conflicting small doc updates, e.g. INDEX.md cross-ref or cycle-summaries/README.md note. None identified as safe/non-conflicting in this prep; see below.)

**Exact Apply Order (to Avoid Blocks/Conflicts/L Violations — Do Not Reorder)**
1. **Review + Independent Disprove Attempt (Tier B style, per rulebook §4 + CLAUDE.md adversarial cross-agent)**: Fresh sub-agent or human must be given: (a) this entire synthesis-research-only/Cycle-010/ dir, (b) the integrator json, (c) pre/post diffs of the 3 Integrator edits (from json "pre_read_lines" + plan/dashboard reads), (d) full 009 audits + cycle_0300.md, (e) current dashboard/goal/plan/next-session, (f) this apply-instructions. Task: **Try to disprove** all claims (0 substrate, 10th failure, L citations, 5-vs-10 gap, "does not satisfy", SMOKE rejection). Only proceed if disprove attempt succeeds on all points (i.e. confirms the bounds). Document findings in new BHS section or comment. **This is non-negotiable per v3.3 Tier B + 5 hard rules.**
2. **If proceeding: Land the Cycle summary md first (non-conflicting new file)**:
   - Copy `Cycle-010-summary-draft.md` → `docs/steering_chelation_rag_dag_research/artifacts/cycle_20260527_XXXX.md` (use current timestamp or "cycle_20260527_0400.md" or similar; match 009's 0300 naming).
   - Or (safer for research-only): `docs/steering_chelation_rag_dag_research/artifacts/synthesis-research-only/Cycle-010/Cycle-010-summary.md` (keep in research tree).
   - **Do NOT edit** the draft before copy. Add one-line header note if needed: "Applied from synthesis-research-only prep by Agent 10; verified [date] per gates above."
   - Verify post-copy: `ls .../cycle_2026...010...` + `head -20` matches draft.
   - **Why first**: New file; zero conflict risk with existing Integrator edits. Provides the "New cycle summary md" deliverable.
3. **Dashboard row (table insertion — only if table hygiene desired; optional)**:
   - In `docs/steering_chelation_rag_dag_research/artifacts/BHS_SHIM_LOOP_DASHBOARD.md`, locate the markdown table (search for "| Cycle-009-2026-05-27" row end).
   - Use search_replace (or manual) with **exact old_string** = the Cycle-009 table row line (copy verbatim from current file to ensure unique match).
   - new_string = Cycle-009 row + "\n" + content from `dashboard-row-draft.txt` (strip the <!-- comments --> or keep as hidden).
   - **Order note**: Only after step 2 (summary md landed). Re-read full table post-edit. Add EVIDENCE: post-edit read + hash of dashboard file.
   - **Warning**: The narrative Cycle-010 section (956+) is *already present* from Integrator. Do not duplicate content; this row is for the historical table only. If table already extended in future state, skip.
   - Why this order: Table edit is more fragile (string match); summary md is additive.
4. **Any safe main-process edits (small doc updates, non-conflicting)**:
   - **Identified in this prep (evidence only; none forced)**: 
     - Potential: Add cross-ref in `docs/INDEX.md` or `docs/steering_chelation_rag_dag_research/README.md` or `docs/ARCH AGENTIC ENGINEERING AND PLANNING/cycle-summaries/README.md` noting "Cycle-010 meta summary in artifacts/cycle_...010... + temp prep dir (research-only, 0 substrate, §128 active)".
     - Or tiny note in goal Model Change Log (but **risky** — already has L4/L9 disclosure; avoid to prevent L13 amplification).
   - **Recommendation**: **Zero safe main edits at this time**. All surfaces (goal, plan, dashboard, next-session) already carry the Integrator's disclosures + L citations. Adding more risks L9 (doc proliferation while 0 substrate) or L7 re-summarization. If any small update is proposed later, it must:
     - Be pre-read + post-verified.
     - Include its own full §4 BH + file:line + EVIDENCE/SMOKE in the edit.
     - Survive the pre-apply gates above.
     - Be drafted first in this synthesis-research-only/ dir (as e.g. safe-edit-foo.patch or .md with <!-- apply as search_replace old=... new=... -->).
   - Draft example (if ever needed; currently none): See placeholder below. Apply only as step 4.5 after 1-3.
   - **Placeholder for future safe edit draft** (do not apply now):
     ```
     <!-- SAFE EDIT DRAFT EXAMPLE (research-only; non-conflicting cross-ref only) -->
     File: docs/steering_chelation_rag_dag_research/README.md
     old_string: (last line of relevant section, e.g. "See loop_02/ for audits.")
     new_string: (same + "\n- Cycle-010: Meta-only Integrator + Synthesis Prep (see artifacts/bhs_10agent...json + synthesis-research-only/Cycle-010/ + cycle_...010 md). 0 substrate. §128 active. BHS 25 proxy / program 10/100 flat.")
     BH required in comment or separate: L9 risk on additional doc; bounded as reference only; EVIDENCE: this prep polls + json.
     ```
5. **Post-Apply Verification (Mandatory, Evidence Rule)**:
   - Re-run all 6 pre-apply gates (must still pass; application must not have changed substrate/block state).
   - `grep -c "MinMax MSA vs SE-RDAG" ...PLAN.md` (if row applied, still 1).
   - Read the landed summary md + dashboard (post row) + confirm 4Qs/deltas/EVIDENCE/SMOKE/BH/L citations match drafts + reference integrator json.
   - Run the exact SMOKE commands from the summary draft + json. Must pass rejection tests.
   - Produce new short EVIDENCE artifact or append to existing (e.g. update the 010 json or new note): command outputs + hashes + "applied per instructions; state unchanged; 0 substrate".
   - Update any living "Last Cycle" pointer in dashboard header if appropriate (but only with explicit L4/L9 disclosure for meta).
   - **If any delta on block/SIPs/metrics**: Revert immediately. This would indicate error.
6. **BHS Packaging for the Application Itself**: The main orchestrator (or PR if this leads to one) **must** end with full §4 Brutal Honesty (per rulebook + CLAUDE.md) disclosing:
   - What was applied (exact files + lines from drafts).
   - L1-L13 with file:line (at minimum L4 on meta "Cycle-010 summary" landing while 0 substrate after 10 cycles + L9 on doc work + L13 on 10-agent framing in context of application).
   - EVIDENCE:/SMOKE: pointing at the pre/post reads + gate outputs + hash of landed files + the summary draft's SMOKE cmds.
   - Explicit: "This application is research-only meta packaging. Does not satisfy goal success def #1. Program score 10/100 flat. §128 active. 10th failure."
   - BHS_SELF_DRAFT / BHS_*_AGENT fields.
   - CARRY_FORWARD: all prior + new from this (e.g. continued 0 closures on SHIM-CDs).
   - Reference this synthesis-research-only/Cycle-010/ dir + integrator json + 009 audits.
7. **If NOT Applying (Recommended Default per Evidence)**: 
   - Archive this entire synthesis-research-only/Cycle-010/ dir as-is (it already documents the monitoring + prep).
   - Add 1-line note to next-session.md or dashboard (with full L citations + EVIDENCE) that "Cycle-010 prep drafts exist in synthesis-research-only/ but were not applied per §128 + 0 substrate + fidelity concerns (see apply-instructions gates)".
   - Escalate to human: the 10-cycle pattern + repeated Integrator meta "success" while BLOCKED itself may constitute additional L9/L4 on the 10-agent model (per json "Recommended Next" + dashboard 990 + 009 Agent 9 audit).
   - **Strong rec**: Before any Cycle-011 (or further 10-agent dispatches), human must amend scheduler task + goal or terminate the loop. No more silent meta iterations.

**BHS Self-Draft for This Prep Wave (Agent 10 Synthesis Prep & Output Packager — Per Rulebook §4 + CLAUDE.md + Goal §108-114)**
**Cycle ID (this wave)**: Cycle-010 Synthesis Prep (special role; temp research-only output only; no main edits).
**BHS_SELF_DRAFT**: 38/100 (after self-caps; + for strict tool-only monitoring of "as they land" (absence of 1-9/010 artifacts documented via exhaustive polls before drafting), temp dir discipline (zero conflict with Integrator edits), full modeled summary with verbatim 4Qs grounded in json/plan/dashboard/009 evidence only, complete EVIDENCE/SMOKE lists with absolute paths + repro cmds, L1-13 citations with file:line in draft, explicit "does not satisfy goal #1" + §128 escalation repeated, apply-instructions with adversarial gate + order to prevent blocks/L violations, BHS self-draft included; - for meta-only nature (0 substrate, no new capability, packaging prior work), 10th cycle pattern participation, L4 on even "prep" framing under 10-agent task while fidelity 0 evidenced, L9 on additional doc surface creation risk, L13 on using 10-agent role language in BLOCKED research-only context, low evidence strength (tool traces + 1 json only; no runtime substrate), no independent Tier B for this prep itself).
**BHS_SELF_DRAFT_AGENT**: Agent 10 (Synthesis Prep & Output Packager) subagent — focused worker delegated specific task per user prompt; used todo_write, multiple parallel grep/read/list, write for drafts only in temp dir.
**What was actually done (evidence only)**: tool calls for exploration (list_dir root/docs/steering/artifacts/loop_02/, 3+ greps for Cycle-010/10-agent/Agent, targeted reads of json (1-62), dashboard Cycle-010 (940-993), 0300.md (1-100+), goal 4Q/backlog (100-220+), plan comparative (190-273), CLAUDE (1-100), rulebook (1-50); todo updates (8 items, advanced with evidence); 3 write calls creating the 3 files in synthesis-research-only/Cycle-010/ (summary ~full 4Q/BH/EVIDENCE modeled on 009 pattern, row draft, this instructions with self BHS + order). 0 search_replace or edits outside temp. All claims here cross-checked against live tool outputs + file contents (no assumption).
**L1-L13 for this prep wave (file:line in outputs or this file)**: L4 (prep of "Cycle-010 summary" + row under 10-agent task while 0/10 artifacts + single prior meta dispatch; temp dir itself visible-without-substrate); L9 (additional research doc surfaces in synthesis-research-only/ while SHIM-CDs 01-08 unclosed 10 cycles + BLOCKED; doc proliferation pattern); L13 (task prompt + this draft use "10-agent, BLOCKED/research only" framing for synthesis prep while actual loop execution model remains 5-agent per scheduler + history + goal change log); L1 (drafts contain pseudocode references from prior but no new runnable); L7 (synthesis of 009 audits + json as "1-9 outputs" for 010 risks re-summarization decay — bounded by explicit "proxy" + "absence" language + polls gate). No L2/3/5/6/8/11/12 in this meta prep (no code, no tests, no broad catches).
**0 on goal success §18-29 / §77-83**: No runtime prod/harness evidence produced or advanced; no BHS Cycle Score >=60 (25 proxy meta); 0 deltas on substrate metrics; program 10/100 flat; this prep itself is documentation packaging.
**EVIDENCE (for all claims in this instructions + sibling drafts)**: 
- Exact tool responses (list_dir outputs, grep match counts 0 for 010 mds/prod, read_file excerpts with line numbers from json/plan/dashboard/0300/goal/CLAUDE/rulebook).
- todo_write calls (visible in conversation trace).
- write success responses (3 files created at absolute paths /home/mattmre/CHELATEDAI/synthesis-research-only/Cycle-010/* with content matching the 009 pattern + json/dashboard 4Qs/BH).
- Pre-draft reads establishing baseline (e.g. dashboard 956 "Agent 10 Integrator" already present; json "0 SIPs"; polls absence).
- No self-attestation trusted; every number (25/100, 10/100 flat, count:2, 0 files, 2 research files, 10 cycles) traced to command output or file content.
**SMOKE (rejection for this prep wave claims)**: On fresh checkout: re-run the 6 pre-apply gates above + `ls synthesis-research-only/Cycle-010/` (must show the 3 files with content containing "0 substrate", "25/100", "§128", "L4/L9/L13", "does not satisfy"); `grep -c "BHS_SELF_DRAFT: 38/100" synthesis-research-only/Cycle-010/apply-instructions-for-main-orchestrator.md` ==1; the SMOKE section in Cycle-010-summary-draft.md must still pass its listed commands (0 prod, BLOCKED count:2, no 010 substrate). Any claim "this prep advanced the loop" or "10-agent fidelity improved" or "debt reduced" fails.
**CARRY_FORWARD**: All prior from Cycle-009 (SHIM-CD-01-08 OPEN, BLOCKED count:2, 5-vs-10 gap L4/L13, 9-cycle 0 substrate, scheduler 0 tasks) + new: 10th failure + L4/L9/L13 on repeated meta "10-agent success" framing (Integrator + this prep) + additional research doc surface (synthesis-research-only/) while substrate 0 + §128 active. No closures. Escalate for human review of whether meta packaging loops constitute process violation.
**DEFERRED_SCOPE**: Full 10-agent parallel execution model (A-J independent artifacts + substrate SIP + bhs_evidence_Cycle-010 json with deltas); any real min-max wiring; SHIM-CD remediation/closure.
**LOOP_ITERATIONS (this wave)**: 1 (focused prep dispatch; used todo for 8+ steps).
**No OPERATOR_OVERRIDE in this wave**.
**Recommendation in this BH**: Per json + dashboard + 009 Agent 9 + this prep: human must act on §128 now (after 10 cycles). Pause the scheduler or amend goal before any further Agent 10 / Integrator / Synthesis Prep or 10-agent dispatches. This temp dir + drafts survive as evidence of disciplined packaging attempt under the rules.

**End of Instructions**. Apply only after full gate review + adversarial disprove. Prefer "not apply" + escalate. All absolute paths in this dir + referenced artifacts are the audit trail. Fresh checkout repro of the SMOKEs is the only acceptance.

*This file + siblings complete the assigned Agent 10 Synthesis Prep task under BHS v3.3. 0 main process impact from this dispatch.*
