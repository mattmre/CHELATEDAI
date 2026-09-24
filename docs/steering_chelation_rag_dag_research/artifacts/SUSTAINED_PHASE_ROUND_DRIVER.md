# Sustained Multi-Hour Phase Round Driver — CHELATEDAI BHS Research (v1 — 2026-05-27)

**Purpose**: This replaces the previous 3-minute fragmentation loop. It is designed for long-running, persistent, goal-like execution (multi-hour rounds) with full 10-agent (A-J) parallel dispatch per major development round. The short 3-min scheduler (019e6a78debf) was deleted because it prevented exactly the sustained, fully-implemented work the user requested.

**Governing North Star**: FULL_SHIM_LOOP_PHASE_PLAN.md (Phases 0-9). Progress is measured by concrete advancement of unblocked phases with runtime evidence, full 10-agent artifacts, and BHS v3.3 scoring — not by number of short cycles.

**Core Model**:
- A "Round" = one focused, fully-implemented development cycle on 1-2 high-leverage unblocked slices from the Phase Plan.
- A Round can run 30-120+ minutes as needed for quality (soft timebox with honest overrun logging).
- Every Round **must** dispatch and collect **all 10 agents (A-J)** with independent artifacts before synthesis (enforce the original 10-agent model that the short loop never delivered at runtime).
- Use the existing 10_AGENT_SAFE_MERGE_COORDINATION_PROTOCOL.md for all parallel work (mandatory re-reads, safe edit order, coordination notes on shared files, collection gates before E/J synthesis).
- Research guard + BLOCKED + SHIM-CDs remain in force until human changes them (OVERRIDE or debt clearance). No prod SIP wiring.

**When to Use This Driver**:
- Launch as a long-running background process (recommended for "run for hours" behavior).
- Or schedule via longer-interval scheduler (45-60 min recommended as bridge).
- Or run manually for a specific Phase Round when the operator wants deep sustained progress.

**Round Structure (adapt duration to the slice)**:
1. **Round Start (0-5 min)**: Orchestrator re-reads all governing docs (goal, phase plan, dashboard, next-session, protocol, recent artifacts, block script, 0-prod). Selects 1-2 concrete unblocked Phase Plan slices. Writes Round Plan + todo. Dispatches the 10 agents.
2. **Agent Execution (main duration)**: Full parallel dispatch of A-J (use spawn_subagent tool for each role with clear, narrow, high-value sub-tasks). Agents follow protocol §1-8. Produce independent artifacts (loop_02/NN_round_agentX_role.md + bhs json where applicable).
3. **Collection + Verification Gate (when agents report done)**: Enforce "all 10 present + independent" before any synthesis. Run full 0-prod, block check, BHS review (D role or adversarial pass). Merge safely using protocol.
4. **Synthesis + Phase Update (E + J + Orchestrator)**: Quantify deltas (even on research harness), update living dashboard + phase plan status, produce Round Summary artifact with brutal honesty, L-tax, 4Qs, "does not satisfy goal #1" (while BLOCKED/SHIM-CD-01), and explicit §128 recommendation.
5. **Round Close**: Decide next round (same phase or next unblocked slice) or pause for human input. Log any overruns as debt only if zero output was produced.

**10 Agent Roles (use exactly — adapt scope to the long round)**:
- A: Research & Mapping (deep literature + seam audit for the target phase)
- B: Build (narrow guarded implementation on research harness or new primitives)
- C: Test & Evidence (run harness, produce runtime EVIDENCE/SMOKE that survives fresh checkout)
- D: BHS Auditor (full rulebook scoring + L1-L13 table + carried debt delta + §128 assessment)
- E: Integration & Self-Improvement (cross-agent synthesis, dashboard/phase plan updates, quantified deltas)
- F: Literature (targeted 2025-2026 papers mapped to current phase)
- G: OPSD / Trace Work (synthetic privileged traces or generator improvements)
- H: Micro-SLM Policy Sketch (if relevant to phase)
- I: MTP Prototype (deepen lookahead, correlation, generator variance)
- J: Meta Auditor (fidelity of the 10-agent round itself, protocol health, Phase 2 "real usage" vs L9 theater assessment)

**BHS Invariants (non-negotiable in every round)**:
- Visible means verified (EVIDENCE:/SMOKE: + repro commands + hashes + file:line in every artifact).
- Full L1-L13 disclosure with severity caps.
- Explicit "0 substrate / does not satisfy goal success def #1" while SHIM-CD-01 + BLOCKED + research guard are active.
- No overclaim on Phase 3 progress.
- 10-agent fidelity is now load-bearing (0/10 = automatic L4 + score cap).

**Long-Running / Persistent Behavior**:
- This driver is intended to be launched once and allowed to manage multiple sequential or parallel rounds over hours.
- Use background execution + periodic status checks.
- The driver can internally decide to continue the next round without waiting for a short scheduler tick.
- When the user wants true "set and forget for hours", launch the supporting Python orchestrator (see companion script) that loops on this driver logic using the available spawn_subagent + scheduler tools.

**Transition from Old 3-Min Loop**:
- Old short scheduler (019e6a78debf) deleted 2026-05-27T14:23.
- Old 3-min prompt was a useful forcing function for BHS discipline and surfaced many L4/L9/L13 issues (5-vs-10 fidelity, meta volume while 0 SIPs, etc.).
- New model keeps all BHS rigor but removes the artificial 3-min fragmentation so full 10-agent implementation of real phase slices can actually complete.

**First Recommended Long Round Target (as of 2026-05-27)**:
Advance Phase 2 ("real usage" of pivot + resilience) + Phase 1/5 (MTP synthetic signal + MinMax correlation + trace generator variance work) with a full 10-agent wave. This is completely unblocked, has existing harness substrate from prior pivot work, and directly tests whether the new longer model can deliver the 10/10 fidelity the old loop never achieved at runtime.

**How to Launch**:
- Manual: Copy this prompt + current phase plan state into a long context and drive the round yourself or via multiple spawn_subagent calls.
- Scheduled: Create a 45-60min scheduler whose prompt is "Execute one complete Sustained Phase Round using the driver in SUSTAINED_PHASE_ROUND_DRIVER.md. Focus on [current unblocked slices]."
- Persistent: Run the Long-Running Orchestrator Python script in background (nohup ... &). It will manage internal rounds for hours.

**Zero-Wall Auto-Chain Mode (added 2026-05-27 per user request to eliminate idle wall time)**:
After a round completes (E/J synthesis, summary artifact, gates, dashboard update), the orchestrator **immediately** begins planning and dispatching the next round with no (or minimal <5s) sleep. The goal is continuous iteration like a persistent /goal process with no burning wall time between turns.

- The 10-minute scheduler (see updated 019e6ab0e6d0 successor) is now a **recovery / heartbeat backstop only** (re-awaken if the persistent stub crashes or the session ends).
- The long_running_orchestrator_stub.py is the primary vehicle for zero-wall behavior: it loops internally, re-verifies BLOCKED/OVERRIDE/0-prod at the start of every round, and chains the next state reload + 10-agent wave (via wired spawn_subagent or equivalent) as soon as the prior round's artifacts and reports are written.
- Mandatory gate: Before every auto-chained round (and on every scheduler fire), the orchestrator must re-read OPERATOR_OVERRIDE.md and run the full §1 re-read + block/0-prod checks. If OVERRIDE: NONE and §128 conditions are met (11+ cycles 0 substrate + BLOCKED count:2 + SHIM-CD-01 OPEN), it must **produce a fresh gate report artifact** (like the 019e6ab0e6d0 post-R04 example) and sleep the safety interval (default 10min) instead of dispatching agents. No silent continuation.
- Wall time accounting: The stub logs productive vs idle seconds per round and appends a "Process Improvement Note" (measured wall, fidelity trends, L9 risks surfaced, suggestions) to the round summary. This enables self-iteration on the loop itself.
- To run with true zero wall: `nohup python .../long_running_orchestrator_stub.py --auto-continue --max-wall-min 10 --phase-plan ... --driver ... > sustained.log 2>&1 &`

This directly addresses the request for "no wall time and you continue working. After you're done with whatever phase or your turn is complete, automatically start the next loop and begin again. Iterate and improve the process."

All BHS invariants remain: full 10-agent fidelity per round, "0 substrate / does not satisfy goal success def #1 while BLOCKED + SHIM-CD-01" in every artifact, research guard, etc.

**Version Note**: Created in direct response to user feedback on 2026-05-27 that the 3-min loop prevented full 10-agent implementation and sustained multi-hour development. Updated same day for zero-wall auto-chain + 10min recovery scheduler per explicit request to stop burning idle time between turns. BHS honesty preserved: we are still blocked on the core goal (real SIPs) until human intervention on OVERRIDE or debt clearance.