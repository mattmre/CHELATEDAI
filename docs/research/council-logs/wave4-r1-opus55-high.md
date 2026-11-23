# Wave-4 R1 Advisory Audit — Opus 5.5 (high)

Read-only. Audit A (PM1b / conv-v3), B (D2-4096d Qdrant+CSLS), C (PM2, proposer v2, Wave-2). No code, GPU or sealed-key access.

## 1. Verified strengths

- **Gate-parity guard:** `run_r6_pm1b_conversational_sota_closure.py:444-445` raises if an inactive gate changes the prompt.
- **Honest statistics:** paired bootstrap + Tango + Clopper-Pearson, LoCoMo cluster CI; `assignment_gates_pass=false` recorded, not hidden.
- **D2 hygiene:** fit set excludes report positives (`run_r6_pm3_and_d2_qdrant_csls_backfill.py:570`); backfill order (`:652`) and CSLS doc hubness (`:498-500`) come from DEV queries; Qdrant vectors unmodified; HNSW recall sweep reported.
- **Causal EMA is causal:** `chelated_raw_text_exocortex.py` projects before updating `mu_q`.
- **Proposer v2 query path infers the slot from retrieval** (`chelated_epistemic_proposer_v2.py:382-430`).
- **v3 complete-linkage echo clustering** (`chelated_conversational_v3.py:235-241`) blocks transitive bridging.

## 2. Findings

### CRITICAL

**C1 — PM2 "label-free" is false (L13).**
- `run_r6_pm2_labelfree_closure.py:252` calls `_resolve_slot_labelfree("t0", c["entity"], c["slot"])`: gold case metadata, hard-coded tenant. The `:250` comment claims neighborhood inference.
- `zero_family_labels_used` / `strictly_causal_online_ema_mu_q` are literal `True` (`:307-308`), not computed.
- The oracle forces CONTESTED_ABSTAIN for every EK5 arm but not `Flat_No_EK5`, so the +85 pp gap partly measures the oracle. Native = 1.000: task at ceiling.
- **Fix:** route all arms through `select_context_for_query`; derive the flags from code paths.

**C2 — The PM1b headline is credited to the wrong component.**
- EK5 adds exactly 0 on clean data (LME and LoCoMo Δ=0.000, n01=n10=0).
- On R5-KU stress, gated vs hybrid is +3.85 pp, LCB −6.4 pp. Of the +25.64 pp over M1, +21.79 pp comes from the ungated hybrid.
- Every comparison with M1 is context-volume confounded: about 6,099 vs 987 prompt tokens on LME (6.2×), 1,666 vs 413 on LoCoMo.
- Calling "+7.80 pp" an EK5/chelation result is L4.

### MAJOR

**M1 — LoCoMo Cat-5 regression (0.13 vs 0.39).**
- Anchor selection (`v3:321-327`) and window expansion (`:328-334`) ignore speakers. Speaker matching exists only in pairwise gating (`:191`).
- Ten anchors plus ±1 windows retrieve the swapped speaker's true statement (any-recall 0.855 vs 0.63), and the reader misattributes it.
- Flat top-5 scores 0.39 largely because it is starved of recall; its Cat-1/4 are correspondingly lower (0.48/0.58 vs 0.67/0.74). One budget knob trades abstention against answerable accuracy.

**M2 — SSU-70 dilution (Δ −5.7 pp, n01=5/n10=1).** Twelve anchors with a per-session cap of 3 (`v3:323`) force at least 4 sessions into single-session questions. Chronological re-rendering (`v3:336`) discards the rank signal.

**M3 — No visible tuning split.** Per-suite budgets (`runner:51-52`), session weight 0.45 (`v3:177`), the cap and the thresholds 0.80/0.55/0.62/0.42 (`v3:196-224`) are fixed on the reported sets.

**M4 — D2 CSLS backfill baseline is a strawman.**
- The uncalibrated merge (`pm3:486-491`) mixes centered ZeroMu scores with raw cosines and collapses to 0.243 at f=0.05.
- CSLS barely moves off its own f=0 value (0.532 → 0.537), so the "+29 pp" mostly undoes self-inflicted miscalibration.
- Needed baselines: per-shard z-score or min-max, and RRF.

**M5 — Coverage, and no ZeroMu win.**
- Only SciFact was scored (n=86); FiQA and NFCorpus are `MISSING_4096`.
- ZeroMu vs Procrustes: −0.99 pp at 4096-d and −1.0 pp at 1M.
- At 1M, QueryAffine recall is exactly 0.0, which suggests a bug, and it is the only arm ZeroMu significantly beats.

**M6 — Proposer v2's rules were patched from D1 v1 forensics** on the same generator (`proposer_v2:1-22,101-117`). D1 v2 (n=60) is in-distribution; vs Mem0 it is +5 pp, LCB −6.7 pp.

### MINOR

- **m1** LME recall is scored per session (`runner:448`) and is saturated (flat 0.968), so it does not discriminate.
- **m2** LoCoMo echoes are verbatim copies (`runner:354-368`), which makes the stress trivial.
- **m3** DualAdap uses a non-causal batch mean (`pm2:202`). Unshuffled queries make the EMA order-dependent.
- **m4** HNSW at k_os=10 reaches recall 0.86; k_os ≥ 32 is required.
- **m5** "1023× memory" ignores that the raw index stays resident and that native recall is 2× higher (0.59 vs 0.29).

## 3. Refinements and v4.0 disclosures

**R1 — Query-named-speaker gating (for Cat-5).**
- Match names in the question against the conversation's closed speaker set. This is inference-time metadata, not a category label.
- Score anchors as speaker-match × content overlap, and reserve ⌈k/2⌉ anchors for the named speaker.
- If named-speaker support is 0 but cross-speaker support is >0, add a neutral `[ATTRIBUTION: statements are by X, not Y]` header. Never force an abstention.
- Freeze the rule on LoCoMo conversations 0-4, report on 5-9, and report Cat-1/2/4 costs.

**R2 — Adaptive budget (for SSU).** Stop at τ·s₁ or at the largest RRF gap. If the top-3 anchors share one session by margin δ, collapse to that session plus its window. Calibrate τ and δ on a dev split.

**R3 — Rank-aware rendering:** mark the top-3 anchors, or order sessions by rank with chronology inside each.

**R4 — Budget-matched controls:** flat BGE-M3 at 26k/12k characters, and v3 at 5 units.

**Mandatory v4.0 disclosures**
1. `assignment_gates_pass=false`; four PM1b checks fail (SSU-70, LoCoMo −4 pp, LoCoMo stress, R5 LCB 0.141 < 0.15).
2. EK5 adds 0 on clean data; the gains come from hybrid retrieval plus 4–6× more context.
3. Cat-5 0.39 → 0.13, mechanism per M1.
4. D2: one domain scored, ZeroMu ≤ Procrustes, raw-mix CSLS baseline, QueryAffine at 0.0, HNSW needs k_os ≥ 32.
5. PM2: ceiling task, gold-slot resolver, literal provenance flags. Rerun before claiming "label-free".
6. D1 v2 is in-distribution (n=60), with no significant gain over Mem0.
