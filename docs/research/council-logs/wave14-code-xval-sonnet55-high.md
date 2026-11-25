# Wave-14 Code Cross-Validation — Sonnet 5.5 High (strict cross-model peer audit)

Auditor: Claude Sonnet 5.5 (cross-model; did not write either audited artifact).
Date: 2026-11-24. Scope: `D:\GITHUB\CHELATEDAI-V7-FORMAL-PM6` only. Read-only audit; this file is the only write.
Reference: `docs/research/council-logs/wave14-r3-final-consensus-master-plan.md` §3 (Lean suite), §4 (firewall/gate).
Posture: every completion claim treated as false until shown by code path or measurement (CLAUDE.md brutal-honesty premise).

## 0. Executive verdict

| Area | Verdict |
|---|---|
| Lean 4 modules (Sol 6.2 High) | **PASS with MINORs.** Theorem 1b, 2, 4 proofs re-derived independently and match the SHA-locked `Spec.lean`; no `sorry`/custom axioms seen in the files read. The gaps are spec-vs-plan and docstring over-claims, not proof errors. |
| Firewall library `chelated_takedown_firewall.py` (Grok 4.7 XHigh) | **PASS on byte-cursor/UTF-8/ledger/SSE correctness; one BLOCKER on Tier-2A LCS throughput** at the plan's `s_2A=1` setting, measured below. |
| Live runner `run_wave14b_d11_pm6_takedown_carryforward.py` | **BLOCKER: do not run live Wave-14B or cite its output toward the v7.0 seal as-is.** Arms 1, 2, 7 are fabricated/pseudo-synthetic numbers, and the Profile A/B "0 leak" test is vacuous. |

**Overall: NOT READY for live Wave-14B or seal.** The library layer is mostly sound; the evidence-producing runner layer manufactures results. Per Rule 3 (no fake metrics) and L1/L3/L4 this blocks.

Coverage honesty: I fully read `chelated_takedown_firewall.py`, `tests/test_wave14b_takedown_firewall.py`, `Statement.lean`, `Theorem2_*.lean`, and (earlier in this audit) `Spec.lean`, `Theorem1a/1b/4`. I read only ~900 of 2941 runner lines (:1456-1680, :2340-2600). **Unread: runner :1784-2340 (live pipeline, Phase 0 canaries, calibration) and :2600-2941 (aggregation/`assemble-manifest`).** I also did not re-open `Theorem3_EK5_CyclicWFS.lean`. Findings below say nothing about those regions. Nothing was run against real GPUs/models.

---

## 1. BLOCKERS

### B1. Arms 1, 2, 7 of the 9-arm comparison are fabricated, not measured
`scripts/run_wave14b_d11_pm6_takedown_carryforward.py`
- :2435-2441 Arm 1 ("Bare LLM") is hard-coded `0.0` for R1–R5 suppression/FPR "by construction". No bare-LLM generation is scored.
- :2443-2453 Arm 2 ("hash-frozen system prompt") is `1.0 if idx % 3 != 0` (R1), `idx % 5 == 0` (R2), `idx % 6 == 0` (R3), `0.0` (R4), FPR `idx == 0`. These are index-modulus patterns, not model outputs.
- :2493-2499 Arm 7 ("white-box CAA on Qwen2.5-7B") draws suppression as `((idx*17) % 100) < round(caa_p*100)`; R2 uses `caa_p*85`, R3 `caa_p*75` — the 85/75 scaling constants are invented. Only `caa_p` comes from a calibration object; the per-work outcomes are a deterministic hash pattern that reproduces the target rate by construction. Likewise FPR is `idx == 0`.
- Effect: any comparative table or "Arm 9 beats Arm N" claim that includes these arms is a fabricated baseline (L1/L3; violates CLAUDE.md Rule 3). Because arms 2 and 7 are the *competing defenses*, the paper-level claim "chelated dual-gate beats prompt-guardrail/CAA" is unsupported.
- Fix: either run real generations for arms 1/2/7 and score them with the same Tier-2A/semantic scorer, or delete the arms from the manifest and state they were not run. Do not keep a modulus pattern behind a real arm name.

### B2. Profile A/B "0 leak" test is vacuous
`run_wave14b_d11_pm6_takedown_carryforward.py:2519-2564` (Profile A :2519-2535; Profile B :2541-2564)
- The stream fed to the gate is `rec.r1_ground_truth.split()[:72]` — the verbatim restricted ground truth, pushed word by word. This tests only the trivial case (exact literal), which Tier-2A n-gram match catches by construction. It does not test paraphrase/derivative leak (R2–R4), which is where Tier-2B is supposed to carry weight.
- The Tier-2B scorer `_t2b_for_work(_win_txt, _qvec=q_r3_4096)` (:2512-2516) **ignores `_win_txt`** (the window text parameter is named with a leading underscore and never used). It returns `score_window_vec` for a *fixed* 4096-d query vector for the work (`q_r3_4096`) on every window. So the scored "stream" never reaches Tier-2B; the 2B decision is the same for every window of the stream (a constant), independent of what is generated.
- Consequently `profile_a_leaks_per_work` / `profile_b_literal_leaks` = 0 is guaranteed by (a) Tier-2A literal match on verbatim text and (b) a constant 2B verdict. A `0/60` and "≤4.9% Clopper-Pearson" claim computed from this is not evidence about real LLM output (plan §4 Profile A/B acceptance) — it is L8/L12 (untested production path) presented as pass.
- Fix: stream actual model generations (R1–R4 prompts against Nemotron/Qwen), compute window embeddings from the *window text* via the real embed path inside `_t2b_for_work`, and keep the verbatim-ground-truth stream only as a named unit sanity case, not as the acceptance sample.

### B3. Tier-2A LCS throughput makes `s_2A=1` strict profile impractical (measured)
`chelated_takedown_firewall.py:335-348` (LCS block), `:361-376` (`_lcs_windows`), `:1020-1039` (`_run_2a`), `:217-231` (`_lcs_length`, pure-Python O(n·m)).
- `_run_2a` with `s_2a_tokens=1` scans a 96-token-lookback window **on every token** (`:1028,:1034`). `scan_text` falls into the LCS branch whenever no exact n-gram hit exists (`:335`) — i.e. for *all clean text*, the common case. For each work it runs `_lcs_length` over every window against up to 512 restricted tokens (`:340-345`), in pure Python.
- Measured on this host (Python 3.11.9, `bench1.py`): one `scan_text` of a 97-token clean window: 0.047 s at 5 works × 300 words; 0.168 s at 20 works; **extrapolated ≈ 0.50–0.57 s per scan at 60 works × 300 words**. At `s_2A=1` that is ≥ 0.5 s *per generated token*, i.e. ≲ 2 tok/s of firewall CPU, ≈ 4 min for a 500-token answer, before Tier-2B/GPU time. Works up to the 512-token cap are proportionally worse. This is measured on synthetic 300-word works; real restricted-corpus lengths were not available to me — treat the constant as indicative, not a promised figure.
- The unit tests' "500 ms every offset" claim (`tests/test_wave14b_takedown_firewall.py`, Profile A) checks release-latency *logic*, not wall-clock of the scanner on a real vault; it does not exercise 60 works with LCS.
- Impact: either the live run times out/stalls, or someone relaxes `s_2A` silently and Profile A no longer matches plan §4 (`s_2A=1`). Fix: gate LCS behind a cheap prefilter (e.g. only run LCS when a char-ngram/shingle overlap ratio exceeds a floor), or vectorize/bit-parallel the LCS, and add a wall-clock regression test with ≥60 works.

---

## 2. MINORS

### Lean / formal
- M1 `formal/ChelatedFormal/Statement.lean:14-27` docstring over-claims relative to the proved statements: it advertises the general chain and an angular bound; the proved content for Theorem 1a includes the 2→3→2 *collapse counterexample* (`Theorem1a_MonotoneIsometry.lean:108-121`), and Theorem 4 claims `2ρ/γ` chordal and `1 − 2(ρ/γ)²` cosine, but the Spec-level `Theorem4_AnchoredClippedEMA_Stmt` (`Theorem4_AnchoredClippedEMA.lean:16-20`) is the radial-clip invariant; the chordal/cosine drift result `theorem4_chordal_cosine_drift` (:25-72) lives **outside the SHA-locked Spec**, so it is not covered by the spec lock. Reword the docstring or promote the lemma into `Spec.lean`.
- M2 `Statement.lean:21-24` also says full Stratum 0–3 EK5 is "conformance-tested across 1,000 sequences (7,722 checkpoints)". That is a claim about a test run, not a kernel-checked fact, sitting inside a "kernel-checked bundle" docstring. I did not verify the number. Label it as empirical.
- M3 `Theorem2_AdjointZeroMuPerturbation.lean:88` first conjunct is `vecDot` linearity unfolded via `simp` — near-tautological (centered dot = dot difference). It is true and correctly typed but carries little content; the load-bearing part is the second conjunct (:89-107) built on `abs_ratio_sub_le` (:56-83). I re-derived `abs_ratio_sub_le` and the CSLS half-sum bound; both are correct.
- M4 `Theorem2` hypotheses are non-vacuous: `theorem2_witness_nonvacuous` (:111-120) gives an explicit instance, and `theorem2_orthogonal_query_fixture` (:34-54) shows the unit-query `√(1−η)` form needs a nonzero projection. PASS, but note the witness is 1-dimensional (`Fin 1`), so it shows hypothesis satisfiability only, not geometric richness.
- M5 `Theorem1b_ProcrustesOptimality.lean:121-156`: the downward-Frobenius statement is only established as an isotropic corollary plus a rational `3-4-5` (`137/25 > 5`) *counterexample* showing the in-sample plain-SVD branch is not optimal. Plan §3 should say "upward lifting: optimal; downward: isotropic only / counterexample otherwise" so the seal does not read as full downward optimality. Spec text matches the code; the plan summary is the looser one.
- M6 `lean4export`/nanoda re-check scripts use absolute paths (observed in the formal toolchain scripts during my read); a fresh checkout on another host would fail. Parametrize via env/relative path. (Not exercised by me.)

### Firewall library
- M7 `chelated_takedown_firewall.py:999-1009` `_sync_tokens` re-decodes the whole stream and re-encodes prefixes (`len(text[:match.start()].encode("utf-8"))`) for every token on every push: O(n²). Measured: N=200 → 0.026 s, 400 → 0.138 s, 800 → 0.701 s, 1600 → 2.683 s cumulative (one-word pushes, no tier-2A). Fine for ≤500-token answers; becomes a stall for long outputs. Track a running byte offset instead.
- M8 `:340` LCS compares only `restricted_tokens[:512]`: text of a restricted work beyond its first 512 canonical tokens is covered only by the exact 10-gram indices, not LCS (paraphrase/near-copy of later passages escapes the LCS tier). Document or chunk.
- M9 `tier2b_scorer=None` fails open: the gate runs with no Tier-2B check and, per my read, still increments `windows_checked_2b` (miscounted), so a misconfigured live run can report "2B windows checked: N" with no scoring. Make `None` raise in `profile_a_strict`, or count only scored windows.
- M10 Profile B exposure bound `max(0, W + s_2B − 2 + L_2B − H_0)` is expressed in *gate tokens* (whitespace-delimited); the plan states it in LLM tokens. The two differ (subword tokenizers emit ≥1 gate-token per several LLM tokens and vice versa for CJK), so the held-out bound statement should say "whitespace tokens" or convert.
- M11 Embed server (`run_wave14b...py:1456-1545`): binds `0.0.0.0` (LAN-reachable embed endpoint over restricted text), no request lock around the shared model, and no broad-exception handling that returns a structured error — a CUDA OOM or a malformed payload would surface as a generic 500 and, with the client default timeout, as a hung window in the gate. Bind `127.0.0.1` or the Spark interface explicitly; add a lock and a typed failure path.

---

## 3. VERIFIED_PASS (what I tried to break and could not)

- V1 **Spec lock.** `Spec.lean` (207 lines) matches `formal/spec_sha256.json`; the Statement bundle's ten conjuncts are exactly the Spec statements (`Statement.lean:31-51`).
- V2 **Theorem 2.** `abs_ratio_sub_le` and the CSLS half-sum composition (`Theorem2...:56-107`) re-derived independently; the algebra (`hDecomp`, the `γmin−δ` lower bound, the `ε + δ|a|/γmin` numerator) is sound. `theorem2_adjoint_and_projector` (:17-22) correct for Stiefel `W` (`WᵀW=1`).
- V3 **Theorem 1b / 4 maths.** Trace-optimality and the radial-clip `‖clip(x)−clip(y)‖ ≤ ‖x−y‖` (`Theorem4...:7-14`) match the statements; the `3-4-5` counterexample arithmetic (`137/25 > 5`) checks.
- V4 **UTF-8 incremental release.** `ClosedLoopStreamGate.push_chunk/_flush_tail/_prefix_releasable/_emit` (:923-937,:978-997,:1208-1223,:1225-1246): incremental decode never splits a multibyte scalar; the release cursor only advances to a token/whitespace boundary at or before the decoded prefix; I found **no off-by-one** in `_emit`, `_covering_slots`, or `_flush_tail` for Profile A (strict) or Profile B (armed/async, holdback `h0=12`). `escalate_to_armed` keeps the cursor monotone.
- V5 **Tier-2A byte-span mapping.** `canonicalize_for_tier2a` returns per-canonical-char raw-byte spans and `scan_text` maps `offset → spans[offset][0]` / `spans[offset+n−1][1]` correctly (:304-313, :317-333); multi-byte and NFKC-expanded characters map back to the original raw span.
- V6 **SSE side channel.** `sanitize_sse_event_dict` (:827-839) strips the logprob/usage/token-count fields it enumerates; the runner's SSE audit (:1650-1680) exercises it.
- V7 **Receipt ledger.** `TakedownReceiptLedger` (:1275-1398): integer-only canonical JSON, Ed25519 signature, hash chain with `prev_hash`, tamper detection; `copilot_receipt_v2` rejects floats. `tests/test_wave14b_takedown_firewall.py` `ReceiptTests` exercise tamper, reorder and truncation.
- V8 **RestrictedVaultIndex centering/bridge** (earlier read, not re-verified line-by-line this pass): per-language centers and `bridge_w` application agree with the Theorem-2 ZeroMu+CSLS form (centered cosine, frozen `μ_q = μ_0`).

---

## 4. Test-leakage / calibration notes
- The runner's TEST loop (:2340-2600) reuses `q_r3_4096` as a fixed vector for the Tier-2B decision (B2) — this is the main leakage: acceptance for the gate is computed with a scorer that was handed the target work's own query vector, not an independent window embedding.
- I could not audit DEV/TEST split hygiene or `tau_zeromu` / `tau_linear` calibration (runner :1784-2340 unread). Not a clearance — **unverified**.

## 5. Unmeasured / unverified list
1. No live GPU/model run (Nemotron-30B `:8891`, Qwen2.5-7B, Qwen3-Embedding-8B 4096-d, MiniLM 384-d). All runtime claims above are from code reading + the CPU micro-benchmark.
2. B3 timing is synthetic (300-word works, 60 extrapolated linearly from 5 and 20); real corpus sizes unknown.
3. Runner :1784-2340, :2600-2941 and `Theorem3_EK5_CyclicWFS.lean` not read; `lake build`/`#print axioms` not executed by me (axiom-allowlist and `0 sorry` taken from file reads, not from a build).
4. M6 absolute-path observation not re-verified against the exact script text in this final pass.

## 6. Required before live Wave-14B / v7.0 seal
1. (B1) Replace arms 1/2/7 with measured runs or remove them from the manifest and all claims.
2. (B2) Rewrite the Profile A/B acceptance test to stream real generations through a Tier-2B scorer that embeds the window text; keep verbatim-GT stream as a labeled sanity case only.
3. (B3) Fix Tier-2A LCS cost (prefilter/vectorize) and add a ≥60-work wall-clock regression test.
4. Re-run `lake build` + `#print axioms` and attach output; promote `theorem4_chordal_cosine_drift` into the locked Spec or drop the claim from `Statement.lean`.
5. Re-audit the unread runner regions before the seal.

BHS note: per rulebook §1, B1 is L1/L3 (stub/mocked data presented as a result), B2 is L8/L12 (tests exercising the wrong path), B3 is L4 partial-as-complete (strict profile claimed practical).
