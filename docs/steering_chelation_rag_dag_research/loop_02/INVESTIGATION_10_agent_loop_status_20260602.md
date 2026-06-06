# Investigation: 10-agent / 5-min BHS shim loop (do not archive yet)

**Date**: 2026-06-02  
**Requested by**: operator — investigate before treating loop artifacts as historical-only.

## Summary

The BHS 5-minute shim workstream is **blocked by process and substrate**, not by a single
technical bug. Research harnesses and JSON evidence exist; **production SIP count was 0** until
the 2026-06-02 `VectorSteerer.steer` research guard landed. The **10-agent parallel model** was
**not reliably executed** in documented cycles; schedulers observed in loop_02 show a **10-minute
recovery task** (`019e6ba504ce`), not the original 5-agent / 5-minute ID (`019e669bf1bb`).

**Recommendation**: Keep `docs/steering_chelation_rag_dag_research/` and `artifacts/bhs_shim_evidence_*`
**active research**, not archived. Revisit archive only after: (1) at least one promoted SIP with
tests, (2) `check_block_flag` CLEAR or debt-draining PR merged, (3) explicit operator scope reduction.

## Why `BLOCKED` (merge gate)

1. **`docs/next-session.md` `**Current**:`** explicitly set to `BLOCKED` (SHIM-CD transcription +
   multi-cycle L9 failure narrative). `scripts/check_block_flag.py` reads this line — not only row count.
2. **Carried debt under-count**: blank line between `CD-TTS-002` and `SHIM-CD-01` in the debt table
   caused the parser to stop early; only **2** OPEN rows counted (`CD-247-01`, `CD-247-02`) while
   **9** OPEN `SHIM-CD-*` rows were omitted. Parser fixed 2026-06-02 to skip intra-table blank lines.
3. **SHIM-CD substance**: 0 production SIPs, artifacts-only primitives, no prod-path smoke in cycles
   001–010 (per audits and grep). Evidence JSON is harness-scoped (`shim_collapse_benchmark_extension.py`).

## 10-agent vs 5-agent fidelity (SHIM-CD-06, SHIM-CD-09)

| Claim | Finding |
|-------|---------|
| Goal: exactly 5 parallel agents (early goal text) | Partial outputs in cycle summaries; not always A–E |
| Goal: 10 agents A–J (later model change log) | Cycle 009/010 doc slices reference 10-agent framing |
| Scheduler `019e669bf1bb` (5-min, 5-agent) | Referenced in debt rows; **no live task** in loop_02 `scheduler_list` captures |
| Scheduler `019e6ba504ce` (10-min recovery) | **Present** in `scheduled_fire_019e6ba504ce_status_20260528*.md` — heartbeat / gate re-read only |
| §128 terminate after 3 cycles BHS &lt; 60 | Exceeded in prose; PAUSE/TERMINATE recs in E/D outputs **not** applied to schedulers |
| Core backlog #1 (first prod SIP) | **0%** through Cycle-010; first guarded probe only in 2026-06-02 session work |

## Evidence artifacts (not archive)

- `artifacts/bhs_shim_evidence_Cycle-002.json` … `Cycle-011-20260527_agentC.json`
- Content: synthetic harness metrics, `block_script_output` BLOCKED/FAIL, reproducibility notes
- **No** assertion of production-path exercise in `runtime_summary` fields reviewed for Cycle-010

## loop_02 unblock wave (2026-05-28)

Human review package and agent artifacts **21–24** (mapping, VectorSteerer diff, tests, rollback)
exist under `loop_02/`. Operator OVERRIDE documented as ACTIVE for delegated unblock while
`0-prod-import` guard held. This investigation aligns with that wave — promotion still requires
Tier B / debt closure per rulebook.

## Next actions (not archiving)

1. Land guarded `VectorSteerer.steer` SIP + `tests/test_shim_vector_steerer_research.py`.
2. Fix debt table blank line or rely on hardened `count_carried_debt_rows`.
3. Decide scheduler: terminate `019e6ba504ce` vs reduce to historical audit-only.
4. Update `next-session.md` SHIM-CD-01 status only when SIP is test-backed and operator accepts partial close.
5. Do **not** move `steering_chelation_rag_dag_research/` to archive until operator confirms scope reduction.