# W22-10 (#60 half only): receiver-activation rejection bound for #60's weight-bound KV guard vs selective top-layer re-prefill — preregistration

- Queue item: `W22-10` (wave slot W22B, PENCIL_AND_CPU). Queue entry:
  `D:\GITHUB\CHELATEDAI\docs\research\wave22_queue_additions_2026-11-25.json`.
- Branch `pv7/w23b-w22-10-kv-guard-bound`, cut from spine commit `2c408925`.
- Written 2026-11-26. This file is committed **alone, before** any result is computed. Any departure from it must be
  named in the report.
- Queue kill/advance rule (verbatim): *"Drop N3 if ordinary contagion matches held-out topology; keep T-K killed
  unless a non-vacuous receiver-level bound beats selective top-layer re-prefill."*

## Scope: the N3 half is NOT run

The queue makes the N3 swarm-threshold falsifier conditional on W22-07 passing. W22-07 killed T-D
(`artifacts/pv7_w23a_seal/W23A_DISPOSITION.md`, section "W22-10: what runs"). The condition is not met, so N3 is not
run, not simulated and not scored here. Only the "#60 KV-Guard Scope Bound" half is in scope.

## What I had seen before writing this

- The queue entry and the wave-22 handoff
  (`docs/research/waves-handoff-wave22-breakthrough-audit-and-queue-additions-2026-11-25.md`, lines 54, 69, 97).
  There, **T-K = "Cross-Model KV-Cache Transport"** (includes the killed N2 RoPE-equivariant transport idea), verdict
  `KILL_OR_DEFER` 3/3, reason: *"Transported-cache receiver KL loses to selective top-layer re-prefill / published
  linear mappers at equal compute"*. Row 97 restricts this item to *"a pencil bound on tiny decoder activations
  strictly to define what #60's weight-bound KV guard can safely reject/reuse (NOT a transfer paper)"*.
- The source of `chelated_ssd_kv_repo_cache.py` (bundle #60): `TinyModel`, `SSDKVCache.put/get/reload_persisted`,
  `serve_chelated`, `contract_for`, `build_repo`, `weight_revision_probe`.
- The registry row `docs/research/registry.d/60-t2e-ssd-kv-repos.json` (guard text) and the W22-09 module header
  (`git show pv7/w23b-w22-09-retraction-state:retraction_boundary.py`) for layout conventions only.
- **No result of any arm below.** No feasibility pilot was run.

## The plain finding before any experiment: what "#60's weight-bound KV guard" is

The guard exists as code, but it is a **binary identity check, not a bound**:

- `SSDKVCache.get` calls `TRA1BoundStateCache.reuse`, which raises `StaleWeightRevisionError` if the active
  `TRA1StateIdentityContract.digest()` differs from the sealed one.
- `SSDKVCache.reload_persisted` refuses any persisted block whose `contract_digest` is missing or differs.
- The contract digest binds the SHA-256 of **all** base weight bytes plus the adapter bytes. Any change to any weight
  (including ones the cached KV does not depend on) is refused. Every refusal falls back to full re-prefill.

No receiver-activation bound, tolerance or "safe reuse" relaxation exists anywhere in the tree. This item therefore
**constructs** one as a research instrument in a new module. It does **not** change `chelated_ssd_kv_repo_cache.py`
and does not wire any relaxation into #60. Any positive result is a recommendation, not a shipped guard change.

The model is #60's tiny deterministic float64 numpy model (2 layers, d=16, 1 head, vocab 64, absolute sinusoidal
positions). It is not an LLM. All conclusions are floor-tier and scoped to this model.

## Setup

### Model and notation

Layer l (l = 0, 1): `q = h Wq_l`, `k = h Wk_l`, `v = h Wv_l`, causal `p = softmax(q k^T / sqrt(d))`, `a = p v`,
`h_{l+1} = h_l + a Wo_l` (for l = 1, `Wo_1` includes the adapter if present), `logits = h_2 Wout`.
Sender weights θ (wrote the cache), receiver weights θ' (wants to reuse it). `||x||` = Euclidean norm of a row;
`||M||` = spectral norm; `Δ` = receiver minus sender; all "max" are over rows.

### Cases

- Repositories: `select_repo_names(6001)` → 20 repos, `build_repo(6001, name)` (50 commits each).
- Per repo, 5 commits chosen by hash: the 5 smallest `sha256("6001|<repo>|<c>")` for `c` in `1..49`. → **100 cases**.
- For a case, `blocks = commit.files` (stable-files-first order). **Prefix** = `blocks[:-1]` (the cached part),
  **suffix** = `blocks[-1].tokens` (the query the receiver computes).
- Sender cache: the prefix is served through the real `serve_chelated(sender, SSDKVCache, prefix_blocks)` path, so the
  stale KV bytes are exactly the `.kvblk` bytes #60 writes. One cache per (model seed), on disk under
  `artifacts/pv7_w23b_w22_10/scratch/`.
- Sender per-block statistics (computed at write time by an instrumented forward that must reproduce
  `TinyModel.forward` KV **bit-for-bit**; a test enforces this) are kept in a side table: per layer `H_l` (max input
  row norm), `Q_l` (max query row norm), `A_l` (max attention-output row norm before Wo).

### Weight-revision classes (receiver θ' relative to sender θ)

| Class | Receiver | Instances per model seed |
|---|---|---|
| R0 identity | θ' = θ | 1 |
| R1 patched-base | `model.patched_base()` (#60's own: `emb[0,0] += 1e-3`) | 1 |
| R2 last-layer adapter | `model.with_adapter(77)` (#60's own probe seed; delta on `Wo_1`) | 1 |
| R3 layer-0 adapter | same-construction dense delta, scale 0.05, added to `Wo_0`; seeds 1..5 | 5 |
| R4(ε) global perturbation | every base matrix M → M + ε·(||M||_F/||G||_F)·G, G Gaussian; ε ∈ {1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1}; seeds 1..5 | 30 |
| R5 cross-model | independent `build_model(s)`, s ∈ {7001..7005} (T-K transport proper) | 5 |
| R6 fp16 | every base matrix rounded to float16 and back | 1 |

"Realistic" classes (in-cone weights materially changed): **R3, R4 with ε ≥ 1e-4, R5, R6.**

Model seeds: **6001 (primary)**, 6002, 6003 (robustness). Cases are the same 100 for every model seed.
Total: 44 instances × 100 cases × 3 model seeds = 13,200 (instance, case) rows.

### Arms

- **A0 #60 guard as shipped.** `SSDKVCache.get(key, ptoks, receiver_contract)`; on `StaleWeightRevisionError` the
  prefix is fully re-prefilled. Exact. Recorded: refusal count.
- **A1 dependency-cone rule (exact, no numerics).** Prefix KV depends only on `emb[prefix tokens]`, `Wq_0, Wk_0, Wv_0,
  Wo_0`, `Wk_1, Wv_1`. Accept the stale block iff all of these are bitwise equal; else full re-prefill.
- **A2 receiver-activation bound guard (the item).** Compute the bound B below; accept the stale prefix KV iff
  `B ≤ τ`; else full re-prefill.
- **A3 TLR-h — selective top-layer re-prefill (conventional baseline, primary).** DroidSpeak-style: reuse the stale
  layer-0 KV, recompute the top layer's KV under θ' from the sender's stored layer-1 input hidden states:
  `K_1 = h_1^s Wk_1'`, `V_1 = h_1^s Wv_1'`. Requires storing `h_1^s` (+25% block bytes; disclosed).
- **A3r TLR-r (reported, not in the kill rule).** Recompute layer-1 KV from the receiver's own layer-0 pass over the
  stale layer-0 KV (no stored hidden state).
- **A4 naive stale reuse (no guard).** Reported to show what the guard protects against.

"True" receiver output = the receiver serving the prefix block-by-block (same procedure as `serve_chelated`) and then
the suffix with that KV. Error of an arm = max absolute difference of the suffix logits (all suffix positions × all
vocab) from true.

### The bound B (sound by construction; derivation below is what will be implemented)

Prefix stage (sender stats `H_l, Q_l, A_l`; `Kmax_l = max ||k_j||`, `R_l = max ||v_j − mean(v)||` from the cached
block):

1. `e_0 = max over prefix tokens t of ||Δemb[t]||` (positions are identical).
2. For l = 0, 1: `dK_l = e_l ||Wk_l'|| + H_l ||ΔWk_l||`, `dV_l = e_l ||Wv_l'|| + H_l ||ΔWv_l||`.
3. For l = 0 only (prefix KV needs no layer-1 attention): `dQ_0 = e_0 ||Wq_0'|| + H_0 ||ΔWq_0||`,
   `δ_0 = [(Q_0 + dQ_0) dK_0 + dQ_0 Kmax_0] / sqrt(d)`, `da_0 = dV_0 + min(e^{2δ_0} − 1, 2) R_0`,
   `e_1 = e_0 + da_0 ||Wo_0'|| + A_0 ||ΔWo_0||`.

Suffix stage (receiver's observed run with the stale cache; superscript o; all norms are observed and cheap):

4. Layer 0: suffix hidden states are exact. `δ'_0 = Qs_0 · dK_0 / sqrt(d)` (Qs = max suffix query norm),
   `ds_0 = dV_0 + min(e^{2δ'_0} − 1, 2) R^o_0` (R^o over all prefix+suffix value rows), `s_1 = ds_0 ||Wo_0'||`.
5. Layer 1: `DK = max(dK_1, s_1 ||Wk_1'||)`, `DV = max(dV_1, s_1 ||Wv_1'||)`, `dq = s_1 ||Wq_1'||`,
   `δ'_1 = [Qs_1 DK + dq (Kmax^o_1 + DK)] / sqrt(d)`, `ds_1 = DV + min(e^{2δ'_1} − 1, 2) R^o_1`,
   `s_2 = s_1 + ds_1 ||Wo_1'||`.
6. **`B = s_2 · max_v ||Wout'[:, v]||`** bounds every suffix logit error of reusing the stale prefix KV.

Lemmas used: `|p'_j − p_j| ≤ p_j (e^{2δ} − 1)` when every score moves by at most δ, and
`Σ p'_j v'_j − Σ p_j v_j = Σ p_j (v'_j − v_j) + Σ (p'_j − p_j)(v'_j − c)` for any constant c.

### Certification targets

- **Primary τ = 1e-4** = #60's own `LOGIT_TOL` (its logit-parity gate against full prefill).
- Secondary (report only): argmax certificate — at each suffix position, the observed top-1/top-2 logit margin `> 2B`
  certifies the same argmax as the true run.

### Cost model (primary: analytic FLOPs; secondary: wall clock)

Prefix length P, d = 16, multiply-add = 2 FLOPs, softmax/exp/residual adds ignored (disclosed):

- `F_full(P) = 3·2Pd² + 2·P(P+1)d + 2Pd² + 2·2Pd²` (all layer-0 work + layer-1 K/V projections).
- `F_tlrh(P) = 2·2Pd²`. `F_tlrr(P) = 2Pd² + 2·P(P+1)d + 2Pd² + 2·2Pd²`.
- `F_guard` = per-revision part (spectral norms: counted as `20·d³` per matrix, an estimate; ΔW and Δemb row
  norms) **amortized over the 100 cases of the instance** + per-case part (`P` lookups, observed-norm passes
  `~4(P+S)d·L`, ~100 scalar ops). Unamortized cost reported as sensitivity.
- A2 expected cost per case = `F_guard + (refused ? F_full : 0)`. A1 = `0 + (refused ? F_full : 0)`.
- Suffix compute is identical for every arm and excluded. Wall clock: median per arm, CPU, single process; at d=16
  this is dominated by Python overhead and is NOT used in any decision.

## Hypotheses

- **H1 (soundness).** B is an upper bound: across all 13,200 rows, `actual_error(A4) ≤ B + 1e-12 + 1e-9·B`.
- **H2 (scope map for #60).** Cone rule A1 and bound guard A2 accept R0, R2 (KV does not depend on `Wo_1`) and the R1
  cases whose prefix lacks token 0, exactly; the shipped guard A0 refuses all non-R0 instances.
- **H3 (T-K advance).** In at least one realistic class the bound is non-vacuous and beats TLR-h (definitions below).

## Definitions and decision rule

For class c, pooled over its instances and the 100 cases (model seed 6001 unless stated):

- `R_bound(c)` = fraction of rows with `B ≤ τ`; `R_cone(c)` = fraction accepted by A1.
- **Non-vacuous** in c: `R_bound(c) ≥ 0.5` AND `R_bound(c) > R_cone(c)` (accepts beyond the exact structural rule).
- **Beats TLR-h** in c: A2 Pareto-dominates A3 on (mean per-case cost, max per-case logit error over the class):
  mean cost(A2) < mean cost(A3) AND max error(A2) ≤ max error(A3).
- Tightness: median and p90 of `B / actual_error(A4)` over rows with `actual > 1e-15`.

**Kill / advance:**

- **K-SOUND:** if H1 fails on any row, the bound is invalid. T-K stays killed and no relaxation of #60 is
  recommended; the report locates the failure.
- **T-K ADVANCE** iff K-SOUND holds AND there is a realistic class c where the bound is non-vacuous AND beats TLR-h on
  model seed 6001, AND the same holds for that class on at least 2 of the 3 model seeds.
- **Otherwise T-K STAYS KILLED.** This is the default.
- Independently of T-K, the report states, per class, what #60's guard could safely reuse under A1 and under A2 at
  τ = 1e-4. That is the item's scope-bound deliverable. Recommendations only; #60's code is not changed here.

## Seeds (complete list)

Repo/case selection 6001; model seeds 6001, 6002, 6003; R2 adapter 77; R3 adapter seeds 1–5; R4 perturbation seeds
1–5; R5 receiver model seeds 7001–7005. No other randomness.

## Outputs

- Evidence (committed): `artifacts/pv7_w23b_w22_10/w22_10_results.json`, `artifacts/pv7_w23b_w22_10/W22_10_REPORT.md`.
- Bulk (gitignored by `artifacts/**/scratch/`): `artifacts/pv7_w23b_w22_10/scratch/` (the `.kvblk` caches). The report
  lists its full path and size as removable.
- Code: `kv_guard_scope_bound.py` (repo root), `scripts/run_w22_10_kv_guard_bound.py`,
  `tests/test_kv_guard_scope_bound.py` (unittest).
