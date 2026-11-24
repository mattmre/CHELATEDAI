---
panel_type: panel-of-experts (4-model council, Round 3 of 3)
artifact: docs/research/council-logs/wave14-r2-synthesis-draft-plan.md
run_date: "2026-11-24"
run_id: wave14-r3-carbon4-high
challenge_depth: round-3-of-3-final
active_experts:
  - gemini-4-carbon (Architecture, Release Governance, & Final Convergence Audit)
persist_version: "1.1.0"
---

# Wave 14 Council — Round 3 Final Binding Audit (`gemini-4-carbon`, High)

**Scope & Identity:** Last-mile audit of `wave14-r2-synthesis-draft-plan.md` (`HEAD 4fae0b77`) against all four Round-2 peer memos (`fable52`, `astra62`, `neon4`, `carbon4`) and cited source lines in `chelated_graphiti_arena_v1.py`, `run_wave11_d9_fast_extractor_and_u_grpo.py`, `run_wave11_d10_5m_real_cutover_and_70b_scale.py`, `spherical_migration_bridge.py`, `ek5_v3_stratified.py`, and `bundle_registry_spec.json`. Reviewer: `gemini-4-carbon` (`high`), `round-3-of-3-final`; record: `docs/research/council-logs/wave14-r3-carbon4-high.md`. Read-only; zero runtime jobs or sealed-key reads.

**Binding Verdict:** **`APPROVED WITH FINAL AMENDMENTS`** — lock and execute upon applying Delta Edits `A1–A6`.

## Last-Mile Findings & Exact Delta Edits

1. **`A1` [CRITICAL — Tail-Latency Byte Leak & Unarmed `Profile B` Gap (`plan:99–102`)]:**
   - *Flaw:* Static buffering `H = 64 + 16 + ceil(L_p99)` leaks undecided bytes whenever Spark #3 RPC or batching exceeds `L_p99` (`1%` tail) and splits multi-byte UTF-8 across BPE tokens; `Profile B` unarmed leak stays bounded *only* if Tier 2B scores unarmed streams asynchronously.
   - *Delta Edit (§4.3):* Enforce **closed-loop verdict-ack release** (`release_cursor <= cleared_window_end - (W_dec + s - 2)` plus incomplete UTF-8 codepoint buffering; treat `80 + ceil(L_p99)` strictly as p99 TTFT telemetry). Run async Tier-2B (`s=16`) on `Profile B` unarmed streams with mid-stream abort, bounding leak at `max(0, W_dec + s - 2 + L_2B_actual - H_0)` tokens.
   - *Check:* Injected `500 ms` Spark #3 delay (`> L_p99`) emits `0` undecided client bytes.

2. **`A2` [CRITICAL — Lean 4 `Theorem1b` Transpose Mismatch & Residual (`plan:76–78`, `bridge:315`)]:**
   - *Flaw:* $X^\top Y = U\Sigma V^\top$ ($U\in\operatorname{St}(d_1,d_0)$, $d_0\le d_1$) types $X$ as $n\times d_1$ and $Y$ as $n\times d_0$, colliding with Corollary 2's $X_{\text{leg}}$ ($n\times d_0$), $Y_{\text{new}}$ ($n\times d_1$) (`Fin d_0`/`Fin d_1` build error) and omitting Fable `C5`'s Procrustes residual.
   - *Delta Edit (§3.2):* Bind $M = Y_{\text{new}}^\top X_{\text{leg}} = U\Sigma V^\top$ ($d_1\times d_0$, $d_0\le d_1$), type counterexample as $d_1=2\to d_0=1$ (`X=diag(1,10)`, `Y=(1,1)^T`), and log in-sample Procrustes residual beside `captured_energy_ratio`.
   - *Check:* `lake build Formal.Theorem1b_ProcrustesOptimality` compiles with satisfiability witness.

3. **`A3` [HIGH — `D7` Schema-Blind Retry on `json_object` Fallback (`plan:51–54`, `arena:308,365,382`)]:**
   - *Flaw:* Calls starting in `"json_schema"` skip `_append_schema` (`arena:308`); when `n_rep > 0` locks `"json_object"` (`arena:365`) and falls through into `ValidationError` retry (`arena:382`), `retry_messages` omits `response_model.model_json_schema()`.
   - *Delta Edit (§2.1):* Inject `response_model.model_json_schema()` into `retry_messages[-1]` whenever retry uses `"json_object"`, and count `finish_reason == "length"` on initial and retry calls.
   - *Check:* Replay test of repaired-invalid JSON asserts schema presence in `retry_messages`.

4. **`A4` [HIGH — `D9-B` Loop Leak, `step=0` Guard, & Peer Restorations (`plan:59–63`, `d9:2487,2641–2672`)]:**
   - *Flaw:* `EVAL_WAVE6_LOOK2` and `D1` (`d9:2641–2643`) remain inside the seed loop; `step=0` SFT makes `"none_eligible"` unreachable unless scoped to `step>0`; `d9:2487,2672,2948` hardcode quartiles and `run_steps*2`; Fable's `(137,2026)` mean and non-attribution rule were omitted.
   - *Delta Edit (§2.2):* Move **all** non-`DEV` evals post-`selection_frozen.json`; scope `"none_eligible"` to `step>0` (`fell_back_to_step0_sft`); set `eval_ckpts=range(10,61,10)` and `prompt_groups=run_steps*8`; report `(137,2026)` beside `(137,2026,31415)`; forbid single-knob causal claims across `Protocol v2`.
   - *Check:* Zero non-`DEV` metrics log before `selection_frozen.json` lock.

5. **`A5` [MEDIUM — `D10-B` Judge/Extractor Contract & `PM-6` Hygiene (`plan:67,105–106`, `d10:1249,1419,1570`)]:**
   - *Flaw:* `70B + Chelated` misses hardcoded `qwen_judge_scores` (`d10:1419`), shares `Nemotron-30B` extraction (`d10:1249`), and breaks `assemble-manifest` (`d10:1570`) on `NOT_RUN`. `PM-6` Arm `(5)` conflates `no-EK5` with `Raw Cosine`, and Phase-3 $\mu_{\text{lang}}$ risks test-batch leakage (`Astra C3`).
   - *Delta Edit (§2.3, §4.4):* Add `70B + Chelated` to `qwen_judge_scores`, disclose shared `30B` extractor (reader-only cross), and write schema-valid `status:"NOT_RUN"` JSON on load failure. Add zero-LLM-cost offline `Gate-1 + Gate-2 Raw Cosine` rescoring, compare `7B` CAA strictly against `7B` arms (`DEV`-calibrated multiplier), and fit $\mu_{\text{es/de}}$ using `0` `TEST` queries.
   - *Check:* `qwen_judge_scores` covers all `70B` arms; `TEST` vectors stay untouched pre-freeze.

6. **`A6` [MEDIUM — `PM-1b` Flag, `/formal` Strata, & `--fast` Notice (`plan:37,80,113`, `ek5:189,1180`)]:**
   - *Flaw:* `PM-1b` needs `assignment_gates_pass:false` (`Astra C6`); `ek5:189` oracle covers Stratum 2 while `ek5:717,1180` span domain-scoped Stratum 1 and Stratum 3 imported cycles; `--fast` omits Lean checks.
   - *Delta Edit (§1.2, §3.2, §5):* Keep `PM-1b` `assignment_gates_pass:false`; distinguish deductive Stratum 1–2 `Theorem3` from full Stratum 0–3 `ek5_ref_cli` + fresh-store conformance in `Statement.lean`; print `formal_tier:"SKIPPED_USE_FULL_FORMAL"` in `--fast`.
   - *Check:* `--fast` stdout flags skipped formal checks.

## Irreducible Limitations, Dissent, & Independent Acceptance
- **Irreducible Scope Limits:** `PM-6` provides hosted-gateway suppression, not open-weight erasure; `CITES_AUDIT` retention is auditable suppression, not physical blob erasure; unbounded semantic allegory remains empirically bounded (`R2b`), never zero-certified. `/formal` proves linear Stiefel properties (in-sample plain branch; downward Frobenius is false), not nonlinear normalization, HNSW ANN misses, or LLM extraction errors (`M6`). `PM-1b` remains underpowered (`N=120`, `assignment_gates_pass:false`).
- **Missing Runtime & Human Proofs:** Live `D7`, `D9-B`, `D10-A/B` (`70B` NF4 load), Spark #3 `Qwen3-8B` p99 latency, Phase-0 `7B` yield, cross-lingual `ROC-AUC`, and mandatory **human-only `PM-4b` grading** (`pm4_human_audit_100_blind_sheet.csv`) remain unexecuted.
