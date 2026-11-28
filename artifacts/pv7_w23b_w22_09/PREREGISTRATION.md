# W22-09 Retraction boundary: compressed state vs segment trees, checkpoints, Crouzeix bounds (#325) and bit ledger (#140) — preregistration

- Queue item: `W22-09` (wave slot W22B, PENCIL_AND_CPU). Queue entry:
  `D:\GITHUB\CHELATEDAI\docs\research\wave22_queue_additions_2026-11-25.json`.
- Branch `pv7/w23b-w22-09-retraction-state`, cut from spine commit `e5e0b037`.
- Written 2026-11-26. This file is committed **before** any result is computed. Any departure from it must be named
  in the report.
- Queue kill rule (verbatim): *"Kill if a state-collision pair breaks exactness without a certificate, or if an
  ordered affine segment tree / geometric checkpoint matches deletion latency & error at <=2x retained bytes."*

## What I had seen before writing this

- The queue entry, the W23A disposition (W22-05 RETIRE proposed, W22-08 KILLED proposed) and the W22-08 prereg format.
- The scorecard rows for #51 (`pv7_51-t3b-beam-memory`), #60 (`pv7_60-t2e-ssd-kv-repos`) and
  #62 (`pv7_62-t4-pooled-workers`).
- The source of `chelated_beam_128k_replay.py` (#51), `chelated_ssd_kv_repo_cache.py` (#60),
  `chelated_pooled_agent_runtime.py` (#62, headers and ledger adapters only), `ek5_v3_stratified.py`
  (`apply_mutation`, `as_of`) and `chelated_pv7_spine/ek5_reason_canonicalizer.py`.
- A feasibility pilot that only timed a fresh EK5 build of the #51 fixture replicated R = 1, 4, 16 times
  (140 / 560 / 2,240 events: 7.5 ms / 42 ms / 64 ms; pickle 69 kB / 230 kB / 878 kB). No arm was compared.

## The plain finding before any experiment: there is no compressed neural-state retraction in #51/#60/#62

I searched the spine (`retract`, `segment`, `checkpoint`, `crouzeix`, `dirty cone`, `affine`). None of the three
bundles holds a compressed (superposed, fixed-size) neural state from which an item is deleted. What they do hold:

| Bundle | Retained state | Retraction mechanism in the code |
|---|---|---|
| #51 beam memory | Uncompressed EK5 event log + derived WFS state (`StratifiedEvidenceStore`) | Append a `REFUTE` withdrawal event; incremental recompute of the dirty cone; "exactness" certified by equality of `export_canonical_ek5_audit_trace(...).digest()` with a fresh rebuild; checkpoint restore + replay certified the same way |
| #60 SSD KV repos | Per-block KV tensors of a tiny 2-layer attention model on disk, keyed by a causal-prefix SHA-256 chain | Mark the file version superseded in EK5; any cached block whose `sources` list contains a non-ADMIT version is refused and recomputed (prefix reuse + suffix replay) |
| #62 pooled workers | Same EK5 store as #51 (via `EK5LedgerAdapter`) | Same as #51 (append + incremental), plus selective per-field reopen of worker output |

So the item cannot be run as titled. I scope it to what is testable:

- **Part A (#60, real code):** the KV-cache retraction path, used as the treatment.
- **Part B (#51 and #62, real code):** the EK5 store's retraction paths, used as the treatment. #62 does not get a
  separate arm: its ledger is the same `StratifiedEvidenceStore`, and its worker-reopen layer is outside this item.
- **Part C (constructed, NOT bundle code):** a linear recurrent "compressed state" with affine transitions,
  `x_{t+1} = A_t x_t + b_t`. Here I build the segment tree, geometric checkpoints, a subtractive (superposition)
  retraction arm, the collision probe and the Crouzeix check. Part C is secondary. **It cannot rescue or kill the item
  verdict.** It only maps where exact deletion without replay is possible at all.

**Segment trees in Parts A/B:** these are N/A. Softmax-attention KV and EK5 WFS states are not an associative affine
monoid, so no ordered affine segment tree exists for them. This is stated, not tested.

## Common definitions

- **Deletion (counterfactual):** the state that would exist had item `k` never been in the history.
  - #60: the prompt without block `k`.
  - EK5: the log without attestation `k` (native `withdraw_attestation_ids`).
  - Part C: either step `k` skipped (`A_k, b_k -> I, 0`, "skip") or input `k` removed (`b_k -> 0`, "input").
- **Replay:** recomputing state from some earlier retained point forward.
- **Latency:** `time.perf_counter()` wall clock around the deletion operation only.
  - Excluded from the timing: building the warmed state, deep copies made so that deletions stay independent, and
    the reference computation.
  - One measurement per deletion. Per configuration I report the median and p90 across deletions.
  - All runs are CPU only and single process.
- **Retained bytes:**
  - #60 treatment: the on-disk bytes of every `.kvblk` file the warmed cache wrote for the prompt
    (`SSDKVCache.bytes_on_disk`). The in-memory index and `TRA1BoundStateCache` are excluded; I will disclose this.
  - EK5 treatment: `len(pickle.dumps(store, protocol=5))`.
  - Baselines: the bytes of every file or pickle they keep.
  - Part C: `8 *` the float64 count of every retained array.
- **Error:**
  - #60: max abs difference of the last-position logits and of every KV entry vs `full_prefill(counterfactual prompt)`.
  - EK5: the count of `val`/`cur`/`eligibility` differences vs `reference_fresh_rebuild` of the counterfactual log
    (must be 0).
  - Part C: relative 2-norm error vs a float64 full replay from `x_0` of the counterfactual sequence.

## Part A — #60 (treatment `T60` = `SSDKVCache` + `serve_chelated`, unmodified)

- **Workloads:**
  - Model seeds {6001, 6002, 6003} (`build_model`).
  - Prompts of N ∈ {8, 16, 32} blocks, block lengths 20–36 tokens, tokens drawn by `numpy.default_rng` seeded from
    `(seed, N)`.
  - Every block index `k ∈ [0, N)` is deleted, once, independently.
- **Treatment:** warm the cache by serving the full prompt, then time `serve_chelated(counterfactual prompt)`. Prefix
  blocks hit. Suffix blocks get new chain keys and are recomputed and written. That is how the bundle code behaves.
- **Baseline `B-FLAT` (checkpoint + replay):**
  - Retain one flat `.npy` of the full-prompt KV plus the token ids on disk. In a KV cache every prefix checkpoint is a
    prefix slice of this one array, so the full geometric/uniform checkpoint ladder costs no extra bytes.
  - Deletion: read the file, slice rows `[0, start_k)`, recompute the suffix with `TinyModel.forward(past=...)`, and
    write the new flat file. The write gives durability parity with the treatment.
- **Context arm `B-REPLAY`:** full prefill of the counterfactual prompt. It retains the token ids only.
- **A1 boundary probe — splice without replay:** drop block `k`'s KV rows, keep all other rows unchanged, then score
  the error. Report the fraction of suffix KV rows whose max abs diff vs the reference is > 1e-9 (the dirty-cone
  fraction), per layer.
- **A3 collision probe:**
  - Two distinct file versions `v1 ≠ v2` with identical tokens at the same position after the same prefix. Their chain
    keys collide by construction.
  - Orders: (i) `v1` cached first, then `v1` retracted, then serve `v2`; (ii) `v2` cached first, then `v1` retracted,
    then serve the `v1` prompt.
  - Exactness for #60 has two parts:
    - (a) the logits match full prefill to ≤ 1e-9;
    - (b) the bundle's stated invariant holds: no cached block whose causal prefix contains a retracted version is
      served. A refusal counts as fail-closed, which is acceptable.
  - The certificate: any refusal, exception, or counter (`refused_retracted_blocks`, `stale_served_blocks`) that marks
    the serve.

## Part B — #51/#62 (treatment `TEK5` = `StratifiedEvidenceStore.apply_mutation`, unmodified)

### Workloads

- **W-BEAM:** the #51 fixture `build_deterministic_beam_128k_fixture(seed=5101)`, derivative branch, replicated
  R ∈ {1, 4, 16} times with id prefixes (140 / 560 / 2,240 events). This is a pilot size; R = 64 is added only if the
  total runtime stays under 20 minutes.
- **W-CHAIN** (dirty-cone stress, synthetic):
  - `e_0 … e_{n-1}`, each `e_i` `DERIVES_CURRENT e_{i-1}`;
  - a tier-5 `r` that `REFUTE`s `e_0`;
  - n ∈ {256, 1024, 4096}.
  - Deleting edge `(e_k → e_{k-1})` re-admits `e_k…e_{n-1}`, so the dirty cone is n − k.
  - If n = 4096 cannot finish inside the budget it is dropped, and that is reported as a departure.

### Deletions and the treatment

- 32 attestations are chosen per (workload, size, rng seed ∈ {0, 1, 2}) uniformly without replacement from the
  attestations of edge-carrying turns. Deletions are independent: each starts from a deep copy of the warmed store.
- **Treatment:** time `apply_mutation(withdraw_attestation_ids=[a])` on that copy. The error is the semantic diff count
  vs `reference_fresh_rebuild`.
- **Secondary op:** the bundle-native `REFUTE`-append retraction, timed the same way. Appending at the end of the log
  is trivially matched by "checkpoint at end + replay one event", so K2 is not evaluated on it. I report it for context.

### Baselines

- **`B-REBUILD`:** retain the log only (pickle of the ordered events and attestations). Deletion means rebuilding
  from the log minus `a` in one atomic batch.
- **`B-GEO-CKPT`:**
  - Retain the log plus pickled store checkpoints at the geometric ladder `p_j = floor(n·(1 − 2^{-j}))`, j = 1, 2, …
  - Checkpoints are added in order of j while the total retained bytes stay ≤ 2 × the treatment bytes.
  - Deletion: unpickle the latest checkpoint whose position is before the turn that introduced `a`. Then replay the
    remaining log minus `a` in one atomic batch. If no checkpoint qualifies, rebuild from scratch.
  - The log is ordered by `tx_time`, then by fixture order.

### B2 collision probes

- **`P51-DIGEST`:**
  - Take a pair of stores that differ in one field the #51 question resolver reads (a replayed turn's `claim_value`,
    or its `tx_time` across a question's cutoff) but have identical structure.
  - Test three things: (i) equal canonical EK5 digest; (ii) different `_resolve_ek5_question` answers on the #51
    question set; (iii) whether #51's own parity criterion (canonical-digest equality, as in
    `_run_branch_with_prefix_parity` / `checkpoint_replay_exact`) reports them as exact.
  - (i) + (ii) + (iii) together = exactness broken without a certificate.
- **`PEK5-ASOF`:**
  - H1: a root `R`, a copy `C` with `COPY_OF R` at tx 2, a tier-5 `REFUTE` of `R` at tx 5, and a native withdrawal of
    the `COPY_OF` attestation at tx 10.
  - H2: the same, except the `COPY_OF` edge never existed.
  - Compare `as_of(valid=∞, tx=7)` for both against the true historical state: a fresh store built from only the items
    with tx ≤ 7, with the edge present in H1.
  - If both stores return the same `C` eligibility but the truths differ, and nothing flags it, the probe breaks
    exactness without a certificate. Here "the store" means the native deletion API, which #51/#62 do not call.

## Part C — constructed affine compressed state (secondary)

- **Grid:** d ∈ {8, 32}, T ∈ {256, 2048}, seeds {0, 1, 2}. 64 deletion positions are sampled per config.
- **Regimes:**
  - `normal-TV`: A_t = 0.97 · Q_t, with Q_t random orthogonal.
  - `nonnormal-TV`: A_t = 0.9 I + N_t, with N_t strictly upper triangular, entries `N(0, (1.5/sqrt(d))^2)`.
  - `nonnormal-LTI`: one fixed A of that form.
  - `singular-TV`: the `normal-TV` regime with 5% of the A_t replaced by a rank-(d−1) projection times 0.97.
  - In every regime b_t ~ N(0, I) and x_0 = 0.
- **Arms:**

| Arm | Retains | Delete op |
|---|---|---|
| `RP` replay | x_0 + all (A_t, b_t) | full forward replay |
| `CK-geo` | RP + states at positions `T − 2^j` (and 0) | restore the nearest checkpoint ≤ k−1, replay the suffix |
| `CK-matched` | RP + states at every position (budget ≈ treatment-comparable) | restore x_{k−1}, replay the suffix |
| `ST` segment tree | (2T−1) composed affine maps | point-update leaf k, recompose O(log T), apply the root to x_0 |
| `SUB-LTI` (treatment proxy, input semantics, LTI only) | x_T, A, A^{2^j} | x_T − A^{T−1−k} b_k, with b_k supplied by the retraction request |
| `SUB-TV` (treatment proxy, input semantics) | x_T + suffix products Φ(T,k) for all k | x_T − Φ(T,k) b_k |
| `INV` (treatment proxy, skip semantics) | x_T + Φ(T,k) + A_k, b_k | x_{k−1} = A_k^{-1}(Φ^{-1}(x_T − c_k) − b_k), then x_T' = Φ x_{k−1} + c_k |

- **`INV` certificate:** cond(Φ(T,k) A_k) < 1e8. Above that the arm refuses. The collision probe runs `INV` with and
  without the certificate on the `singular-TV` regime.
  - **Collision pair:** two histories that differ only before a singular step, so they have equal x_T but different
    skip-counterfactuals.
- **Work proxy (bit ledger, #140):** the matvec/matmul count per deletion and the dirty-cone length T − k. Reported with
  the bytes.
- **Crouzeix (#325):**
  - **Numerical range W(A):** boundary points `v_θ* A v_θ`, where v_θ is the top eigenvector of
    `(e^{iθ} A + e^{−iθ} A*)/2`, at 720 angles θ. The angles are refined 8× when a ratio comes within 0.1% of a
    bound.
  - **Polynomials:** z^n for n = 1…64, and 32 random complex polynomials of degree ≤ 8.
  - **Matrices:**
    - the `nonnormal-LTI` A;
    - 8 sampled A_t per TV regime;
    - the square `wq/wk/wv/wo` weights of the #60 `TinyModel`, seeds 6001–6003. These are labelled as weights, not as
      transitions.
  - **Constant used:** **1+√2** (Crouzeix–Palencia 2017, a theorem). I also report the ratio against **2**, the
    conjectured constant, and against the spectral-radius proxy `ρ(A)^n`.
  - Sampling the boundary can only underestimate `max_W |p|`. So a computed ratio above the bound is reported as a
    possible discretization artefact unless it survives refinement.

## Kill rule as operationalized (primary = Parts A and B only)

- **K1** fires if any A3/B2 probe shows a state-collision pair whose served/certified output is not exact, with no
  refusal or flag.
  - I report separately whether it fires on a path the bundle uses (`P51-DIGEST`, A3) or only on the native
    deletion API (`PEK5-ASOF`). **Both count for K1.**
- **K2** fires for a bundle treatment if, in at least one preregistered (workload, size) configuration, a baseline
  among {`B-FLAT`, `B-GEO-CKPT`} meets all three of:
  - median latency ≤ 1.10 × the treatment median;
  - error ≤ max(the treatment error, 1e-9) for #60, or 0 semantic diffs for EK5;
  - retained bytes ≤ 2 × the treatment's retained bytes.
  - `B-REBUILD` and `B-REPLAY` are context only.
- **Item verdict:**
  - `KILLED` if K1 or K2 fires on any bundle treatment.
  - Otherwise `ADVANCE (scoped)`. The scope is: retraction is exact and beats replay at matched bytes for the bundle
    arms that were testable.
  - Part C gets its own K1/K2 readout under the same thresholds, with `SUB-*`/`INV` as treatment. It is labelled
    secondary.

## Expectations written down before running (so they cannot be quietly rewritten)

- I expect K2 to fire for #60 by construction. Its retraction is prefix reuse + suffix replay, and `B-FLAT` holds the
  same KV without the per-block JSON header (which carries the full prefix token list). I am running it anyway, to
  measure the byte and latency gap rather than assume it.
- I do not know whether EK5 beats `B-GEO-CKPT` on W-CHAIN.
- I do not know whether `P51-DIGEST` / `PEK5-ASOF` actually reproduce. I have only read the code paths.

## Process

- Code:
  - `retraction_boundary.py`: the segment tree, checkpoints, Part C arms, Crouzeix, and probe helpers.
  - `scripts/run_w22_09_retraction_boundary.py`: the runner, with `--out-dir`, default `artifacts/pv7_w23b_w22_09`.
  - `tests/test_retraction_boundary.py`: unittest tests.
- Bulk data goes under `<out>/scratch/`, which is gitignored. Results go to `w22_09_results.json` and
  `W22_09_REPORT.md`.
- CPU only, deterministic seeds, target total runtime < 20 minutes.
