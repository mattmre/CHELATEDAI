<!--
{
  "taskId": "node:delegated-task:command%3Amcp%3A3b84a70c-9824-441c-8750-b9902399a932%3Adelegate-task%3Acouncil-r2-neon-xhigh-20261121",
  "link": "[Council Refinement R2 Neon XHigh : Cross-Check Peer R1 Proofs & Conv...](t3-thread://v2/c5bb8594-9c79-4948-9251-bd83e8527cf6/thread:delegated-task:command%3Amcp%3A3b84a70c-9824-441c-8750-b9902399a932%3Adelegate-task%3Acouncil-r2-neon-xhigh-20261121)",
  "model": "gemini-4-neon",
  "executedModel": null,
  "options": [
    {
      "id": "thought_level",
      "value": "xhigh"
    }
  ],
  "status": "completed"
}
-->

# Tri-Model Doctoral Research Council — Refinement Round 2 Report (`Neon XHigh`)

**Reviewer:** `gemini-4-neon` (`xhigh`)  
**Target Checkout:** `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1` (`HEAD 18973789`, strictly read-only; all probes executed in-memory with `python -B`)  
**Audited Inputs:** [r2-synthesis.md](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r2-synthesis.md), [r1-astra.md](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-astra.md), [r1-carbon.md](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-carbon.md), [r1-neon.md](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-neon.md), [f1_4096_ood_manifest.json](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/artifacts/flagship1-night1/20261121-rigor-r3/f1_4096_ood/f1_4096_ood_manifest.json), all 15 `report_receipt.json` bundles, and [chelated-memory-spherical-migration-whitepaper-presentation.html](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/chelated-memory-spherical-migration-whitepaper-presentation.html).

---

## 1. Peer R1 Cross-Check Audit Table (Including Self-Audit Concessions)

| Peer | R1 Claim / Algorithm | Verdict | Executable Evidence / Edge Case Found |
|---|---|---|---|
| **Astra R1** | 3-node stale-rank counterexample (`56,568` rank-only edits; `48` false skips) & raw-Fitting separation ([r1-astra.md:54-76](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-astra.md#L54-L76)) | **CONFIRMED** *(generalized via positive-SCC $\rho=0$)* | Independently reproduced **exact `56,568`** rank-only edits across all `262,144` 3-node graphs. Confirmed Astra's raw-Fitting separation (`raw_rank=∞` on unfounded cycles) is 100% sound, though more conservative than positive-SCC $\rho=0$ (which also shields hubs downstream of positive cycles with `0` errors). |
| **Astra R1** | Direct-$B_i$ + Declared Version-Lineage + quarantined cycle member supplies *at most possible* force externally ([r1-astra.md:27-45](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-astra.md#L27-L45)) | **CONFIRMED & ADOPTED** | Probe 2 showed Neon R1's $B_i^{\text{cyc}}$ allowed quarantined cycle member `B` in `A↔B` with `B→E` to force `E=F` (definite kill!). Routing edges *leaving* a nontrivial `SUPERSEDE`-SCC into $B_i^{\text{cyc}}$ yields `E=U` (`CONTESTED`), matching Astra's invariant in pure WFS. |
| **Astra R1** | `f1_4096_ood` forensic corrections: no Legal-trained bridge, OOD Ridge `0.0000–0.1704`, Hop-2 `n-1=1999>1024` full thin rank, split discrepancy `0.6866` vs `0.6484`, no runner script ([r1-astra.md:105-119](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-astra.md#L105-L119)) | **CONFIRMED 4/4** | Verified on disk: `fit_domains=['CoSQA_Code','SciFact']`; all-18 OOD Ridge `[0.0000, 0.1704]` (`[0.0000, 0.1362]` on 12 `384↔4096` cells); `SciFact` D384 `native_old=0.6866` (in-domain) vs `0.6484` (OOD); `scratch/run_pristine_f1_4096_ood.py` absent from repo. |
| **Astra R1** | `A2g` non-conformality (`1.8` radial vs `1.2751` transverse at $s=1.8,\theta=0.9$) & aligned-pole exact composition inside $\|w\|<\pi$ ([r1-astra.md:138-144](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-astra.md#L138-L144)) | **CONFIRMED** | Verified `radial=1.8000`, `transverse=sin(1.62)/sin(0.9)=1.2751`; aligned-pole 2-hop composition matched sequential `Log→Exp→Log→Exp` to **`2.498e-16`** max error. |
| **Astra R1** | Exact `PM-1b` power (`N=1,921` for 1 pp HW; `N=3,925` for 80% power `LCB>-1pp` at $q=0.05$) & zero-discordance bootstrap degeneracy ([r1-astra.md:167-169](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-astra.md#L167-L169)) | **CONFIRMED & ADOPTED** | Exact normal check: `1920.7` and `3924.4`. At `N=500, q=0.05`: `LCB>=-3pp` + `point>=-1pp` gives **`84.1%` power** (vs `51.6%` at `-2pp`, `70.5%` at `-2.5pp`). At `0/0` discordance, percentile bootstrap gives degenerate `[0,0]`, whereas Newcombe-Wilson gives `-0.76 pp` (`N=500`) / `-5.20 pp` (`N=70`). |
| **Carbon R1** | `Theorem 1.1` Pitfall A cone pruning + Pitfall B $O(1)$ external shield delta + queue-driven $O(|D|+|E_{\text{inc}}(D)|)$ solve ([r1-carbon.md:58-71](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-carbon.md#L58-L71)) | **CONFIRMED (WITH 2 REFINEMENTS)** | Passed Astra's 3-node counterexample (`dirty=1`, `rho[B]:0→2`, Pitfall B updated `sigma[A]:2→1`!), **`0/1,018,224`** exhaustive 2-step rank-only tests, **`0/150,000`** single-edge ops, **`0/100,000`** atomic multi-edge batches (`50k` Dijkstra + `50k` **plain FIFO**!), and `dirty=1, edges=3` (`0.032 ms`) on 10k-in/10k-out hub! **Two mandatory refinements:** (1) asymmetric `rho==0` rule (`ref_out`/`del_ref` may skip/withhold positive-SCC `rho=0`, `dep_rev`/`del_dep` MUST require `rho>0`), (2) deduplicate opposite `(add,del)` edges in atomic batches before BFS. |
| **Carbon R1** | Stratum 2 prose claim that external `X→A` alone resolves both `A` and `B` in `A↔B` ([r1-carbon.md:46](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-carbon.md#L46)) | **PARTIALLY REFUTED (SCOPED)** | True ONLY for **Declared Version-Lineage** condensation DAGs. False for **ordinary Direct-$B_i$ claims**: `X→A` alone yields `X=T, A=F, B=U (CONTESTED)` so `A` cannot silently kill `B` without `X` covering `B`. |
| **Carbon R1** | Effective Authority Ceiling $\tilde{\alpha}(v) = \min(\alpha_{\text{raw}}(v), \min_{d \in D_V(v)} \tilde{\alpha}(d))$ ([r1-carbon.md:83-85](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-carbon.md#L83-L85)) | **CONFIRMED (WITH EDGE-ONTOLOGY & DELETE GUARD)** | Probe 2 showed $\tilde{\alpha}$ **must** cover all $D_V$ (both `COPY_OF` and arithmetic `AMENDS`, else `AMENDS`-laundering breaks it!), while auditors refuting/logging rumors MUST use `REFUTE` or non-epistemic `CITES`/`AUDITS_RECORD` ($\notin D_V$) so `auth=4` is not dragged to `0`. Under deletions/rewinds, `min` is non-monotone and needs $O(|D|+|E_{D_V}|)$ $L$-bucket recomputation + fail-closed `alpha=0` on dangling invisible parents. |
| **Carbon R1** | `A2g` `pole_floor = 0.25` & up-projection `s >= 1.0` cap ([r1-carbon.md:142,171](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-carbon.md#L142)) | **CONFIRMED & ADOPTED (BEATS NEON R1!)** | At `d=384, n=500`, `||mu||=0.15` still had **`16.62°`** mean pole jitter vs **`9.76°`** at `0.25`. On realistic heterogeneous-hubness up-projection (`384→4096`), **`max(1.0, s_oos)` swept `5/5` regimes**, beating `A2c_OOS` by `+0.052–0.107` CSLS on wide→narrow and `A2(1.0)` by `+0.171` CSLS on narrow→wide! |
| **Carbon R1** | `--warm: #7d2f0a` contrast listed as `7.84:1` on `#fdf2ec` ([r1-carbon.md:233](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-carbon.md#L233)) | **MINOR NUMERIC FIX (STRONGER)** | Exact WCAG 2.1 contrast of `#7d2f0a` on `#fdf2ec` is **`8.36:1` (AAA)** (`9.20:1` on `#ffffff`). Adopted `#7d2f0a` over Neon's `#8a3208` (`7.51:1`) for extra AAA margin. |
| **Neon R1 (Self)** | `A2g` smooth `0.95*pi * tanh(r / 0.95*pi)` cut-locus guard ([r1-neon.md:146](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-neon.md#L146)) | **SELF-REFUTED & WITHDRAWN** | Probe 10 proved `tanh` introduces **`-5.06°` to `-7.91°` radial error** inside Astra's exact angular-dilation test (`r=1.37..1.60`) and `-14.55°` at `r=2.0`, ruining `A2g`! Replaced with hard clamp `min(r, pi - 0.05)` (`0.00°` error on `[0, 177.1°]`). |
| **Neon R1 (Self)** | `s_subspace = ||Y_val,c P_Q||_F / ||X_val,c Q||_F` on up-projection ([r1-neon.md:173](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-neon.md#L173)) | **SELF-REFUTED & WITHDRAWN** | Probe 9A proved projecting numerator by $P_Q$ shrinks $s$ (`0.666 < 0.879 < 1.0`), worsening the signal-to-hubness ratio $s\langle(q-\mu_x)Q, d_j\rangle + \langle\mu_y, d_j\rangle$ and scoring **dead last (`0/5` wins)** under realistic document hubness (`0.2025` vs `0.3831` CSLS)! |
| **Neon R1 (Self)** | `UnifiedWFSStore` required $O(|D|\log|D|)$ Dijkstra heap & omitted `f1_4096_ood` `0.1704` upper OOD bound ([r1-neon.md:75,217](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-neon.md#L75)) | **SELF-CORRECTED** | Plain FIFO Lyapunov propagation inside `D` preserves invariant $\mathcal{I}$ in **strict $O(|D|+|E_{\text{inc}}(D)|)$** (`0/1,018,224` 2-step + `0/50,000` batch errors). Corrected all-18 OOD Ridge range to `0.0000–0.1704` (`0.0000–0.1362` on 12 `384↔4096` cells). |

---

## 2. Resolved Final Specifications for Targets 1–5 (Ready to Freeze in R3)

### Target 1 — Unified `EK5 v3` Exact Linear-Time Rank-Shielded WFS & Stratum 2 Specification

#### 1A. Unified Stratum 1 (`EK5 v3` Epistemic WFS): Pitfall A/B + Positive-SCC $\rho=0$ + Linear FIFO Lyapunov
- **Invariant $\mathcal{I}$ (Lyapunov Proof-Tree Potential):** Every node $v$ stores `val[v] ∈ {F, U, T}`, `rho[v] ∈ ℕ ∪ {∞}` (`rho[v]=∞ ⇔ val[v]=U`; `rho[v]=0` for positive-SCC `F`-nodes or unattacked root `T`-facts), and for `F`-nodes:
  $$\sigma[w] = \big|\{u \in R^{\text{auth}}(w) : \operatorname{val}[u]=\mathbf{T} \land \rho[u] < \rho[w]\}\big| + \big|\{d \in D_V(w) : \operatorname{val}[d]=\mathbf{F} \land \rho[d] < \rho[w]\}\big|$$
  *(Note: For positive-SCC nodes with $\rho[w]=0$, $\sigma[w]\equiv 0$ identically. Non-minimal lazy upper bounds $\rho$ remain exact in `int64` with optional periodic rank compaction on checkpoint/snapshot; `exact_min_rank=True` Dijkstra mode available for diagnostics.)*

```python
def apply_epistemic_batch(store, raw_add_refs, raw_del_refs, raw_add_deps, raw_del_deps):
    """Exact O(|Delta| + |D| + |E_incident(D)|) incremental WFS with Pitfall A/B & positive-SCC collapse."""
    # 0. Net-cancel opposite edges in the same atomic batch (after multiplicity check)
    add_refs = set(raw_add_refs) - set(raw_del_refs)
    del_refs = set(raw_del_refs) - set(raw_add_refs)
    add_deps = set(raw_add_deps) - set(raw_del_deps)
    del_deps = set(raw_del_deps) - set(raw_add_deps)

    seeds = set()
    # 1. Apply net topology changes + O(1) Lyapunov-sound seed skips
    for u, w in add_refs:
        if u in store.refs[w]: continue
        store.refs[w].add(u); store.ref_out[u].add(w)
        if store.val[w] == F and store.val[u] == T and store.rho[u] < store.rho[w]:
            store.sigma[w] += 1
        if not (store.val[w] == F or (store.val[u] == F and store.rho[u] < store.rho[w])):
            seeds.add(w)
    for u, w in del_refs:
        if u not in store.refs[w]: continue
        was = (store.val[w] == F and store.val[u] == T and store.rho[u] < store.rho[w])
        store.refs[w].remove(u); store.ref_out[u].remove(w)
        if was: store.sigma[w] -= 1
        # Positive-SCC (rho==0) cannot be broken by deleting REF -> safe to skip!
        if not (store.val[w] == T or (store.val[w] == F and (store.rho[w] == 0 or store.sigma[w] > 0))):
            seeds.add(w)
    for w, d in add_deps:
        if d in store.deps[w]: continue
        store.deps[w].add(d); store.dep_rev[d].add(w)
        if store.val[w] == F and store.val[d] == F and store.rho[d] < store.rho[w]:
            store.sigma[w] += 1
        if not (store.val[w] == F or (store.val[d] == T and store.rho[d] < store.rho[w])):
            seeds.add(w)
    for w, d in del_deps:
        if d not in store.deps[w]: continue
        was = (store.val[w] == F and store.val[d] == F and store.rho[d] < store.rho[w])
        store.deps[w].remove(d); store.dep_rev[d].remove(w)
        if was: store.sigma[w] -= 1
        # Positive-SCC (rho==0) CAN be broken by deleting DEP -> MUST require rho[w] > 0!
        if not (store.val[w] == T or (store.val[w] == F and store.rho[w] > 0 and store.sigma[w] > 0)):
            seeds.add(w)

    if not seeds:
        return MutationDelta(inserted=..., transitioned={}, removed=...)

    # 2. Pitfall A: Forward cone expansion withholding strictly shielded F-nodes
    D = set(seeds); q = deque(seeds); loss = {}
    while q:
        u = q.popleft()
        for w in store.ref_out[u]:
            if w in D: continue
            if store.val[w] == F:
                if store.val[u] == T and store.rho[u] < store.rho[w]:
                    loss[w] = loss.get(w, 0) + 1
                if store.rho[w] == 0 or (store.sigma[w] - loss.get(w, 0) > 0):
                    continue  # shielded F-node or pos-cycle on REF edge: withhold from D
            D.add(w); q.append(w)
        for w in store.dep_rev[u]:
            if w in D: continue
            if store.val[w] == F:
                if store.val[u] == F and store.rho[u] < store.rho[w]:
                    loss[w] = loss.get(w, 0) + 1
                if store.rho[w] > 0 and (store.sigma[w] - loss.get(w, 0) > 0):
                    continue  # strictly shielded non-cycle F-node: withhold from D
            D.add(w); q.append(w)

    old_val = {u: store.val[u] for u in D}
    old_rho = {u: store.rho[u] for u in D}

    # 3. Iterative Tarjan positive-SCC collapse inside D + boundary-frozen linear FIFO propagation
    pos_cyc = iterative_tarjan_nontrivial_sccs(D, store.deps)
    new_val = {u: U for u in D}; new_rho = {u: INF for u in D}; rem_sat = {}; fifo = deque()
    for u in D:
        if u in pos_cyc:
            new_val[u] = F; new_rho[u] = 0; fifo.append(u); continue
        rem = 0; best_f = INF; max_r = -1
        for a in store.refs[u]:
            if a not in D:
                if store.val[a] == T: best_f = min(best_f, store.rho[a] + 1)
                elif store.val[a] == F: max_r = max(max_r, store.rho[a])
                else: rem += 1
            else: rem += 1
        for d in store.deps[u]:
            if d not in D:
                if store.val[d] == F: best_f = min(best_f, store.rho[d] + 1)
                elif store.val[d] == T: max_r = max(max_r, store.rho[d])
                else: rem += 1
            else: rem += 1
        rem_sat[u] = [rem, max_r]
        if best_f < INF:
            new_val[u] = F; new_rho[u] = best_f; fifo.append(u)
        elif rem == 0:
            new_val[u] = T; new_rho[u] = 0 if (not store.refs[u] and not store.deps[u]) else (max_r + 1); fifo.append(u)

    proc = set()
    while fifo:
        u = fifo.popleft()
        if u in proc: continue
        proc.add(u); vu, r = new_val[u], new_rho[u]
        for w in store.ref_out[u]:
            if w not in D or w in proc or w in pos_cyc: continue
            if vu == T and new_val[w] != F:
                new_val[w] = F; new_rho[w] = r + 1; fifo.append(w)
            elif vu == F and new_val[w] == U:
                rem_sat[w][0] -= 1; rem_sat[w][1] = max(rem_sat[w][1], r)
                if rem_sat[w][0] == 0:
                    new_val[w] = T; new_rho[w] = rem_sat[w][1] + 1; fifo.append(w)
        for w in store.dep_rev[u]:
            if w not in D or w in proc or w in pos_cyc: continue
            if vu == F and new_val[w] != F:
                new_val[w] = F; new_rho[w] = r + 1; fifo.append(w)
            elif vu == T and new_val[w] == U:
                rem_sat[w][0] -= 1; rem_sat[w][1] = max(rem_sat[w][1], r)
                if rem_sat[w][0] == 0:
                    new_val[w] = T; new_rho[w] = rem_sat[w][1] + 1; fifo.append(w)

    for u in D:
        store.val[u], store.rho[u] = new_val[u], new_rho[u]

    # 4. Pitfall B: O(1) external shield delta (handles rank-only increases!) + local recompute in D
    for u in D:
        for w in store.ref_out[u]:
            if w not in D and store.val[w] == F:
                store.sigma[w] += int(store.val[u]==T and store.rho[u]<store.rho[w]) - int(old_val[u]==T and old_rho[u]<store.rho[w])
        for w in store.dep_rev[u]:
            if w not in D and store.val[w] == F:
                store.sigma[w] += int(store.val[u]==F and store.rho[u]<store.rho[w]) - int(old_val[u]==F and old_rho[u]<store.rho[w])
    for w in D:
        store.sigma[w] = (
            sum(1 for a in store.refs[w] if store.val[a]==T and store.rho[a]<store.rho[w]) +
            sum(1 for d in store.deps[w] if store.val[d]==F and store.rho[d]<store.rho[w])
        ) if store.val[w] == F else 0
```

#### 1B. Unified Stratum 2 (Temporal Currency): Direct-$B_i$ + Quarantined $B_i^{\text{cyc}}$ vs. Declared Version-Lineage
1. **Two Typed Replacement Relations:**
   - **Ordinary `SUPERSEDE` (Direct-$B_i$):** Active set $E_{\text{SUP}}^{\text{act}} = \{(j \to i) \in E_{\text{SUP}}^{\text{auth}} : V_j \neq \mathbf{F}\}$. Compute Tarjan SCCs of $E_{\text{SUP}}^{\text{act}}$; let $\mathcal{Q}_{\text{SUP}}$ be all nontrivial SCCs ($|\text{SCC}| \ge 2$ or self-loop). Route edge $(j \to i)$ to $B_i^{\text{cyc}}$ iff **`scc(j) == scc(i)` OR `scc(j) in Q_SUP`** (implementing Astra's *"quarantined member supplies at most possible replacement force externally"*!), otherwise to $B_i^{\text{ext}}$:
     $$B_i^{\text{ext}} \leftarrow V_j, \qquad B_i^{\text{cyc}} \leftarrow V_j \wedge \operatorname{not} B_i^{\text{cyc}}, \qquad C_i \leftarrow V_i \wedge \operatorname{not} B_i^{\text{ext}} \wedge \operatorname{not} B_i^{\text{cyc}} \wedge \bigwedge_{d \in D_i^{\text{cur}}} C_d$$
   - **Declared `VERSION_LINEAGE_SUPERSEDE`:** Requires identical `(tenant, entity, single_valued_slot, lineage_id)` + authorized `LINEAGE_SUP` capability. Durable version-history edge (persists even if issuer's factual claim is refuted). Evaluated linearly on the lineage condensation DAG via two monotone flags propagated forward: `has_def_anc` (strict ancestor SCC contains a non-quarantined node with $V_u = \mathbf{T}$) $\implies B_i^{\text{ext}}=\mathbf{T}$, and `has_poss_anc` (strict ancestor SCC contains a non-refuted quarantined or $V_u=\mathbf{U}$ node, or $i$ lies in a nontrivial lineage SCC with a non-refuted member) $\implies B_i^{\text{cyc}}=\mathbf{U}$.
2. **Write Boundary & Two-Mode Dependency Split:**
   - Reject cycle-forming ordinary/lineage `SUPERSEDE` writes online (`SupersedeCycleError`); imported/replayed cycles evaluate fail-closed to `C = U` (`CONTESTED(TEMPORAL_CYCLE)`).
   - **`D_C ⊆ D_V` (`COPY_OF`, `DERIVES_CURRENT`):** Inherit both $V$ and $C$ (so `C COPY_OF A` under `A↔B` becomes `C=U`, `CONTESTED(TEMPORAL_ANCESTOR)`). Legacy `DERIVES_FROM` defaults to `DERIVES_CURRENT` unless explicitly migrated (`Astra R1 §1` safety rule — no silent backfill mutation!).
   - **`D_V \ D_C` (`AMENDS`, `DERIVES_SNAPSHOT`):** Inherit $V$ only (so `B AMENDS+SUP A` keeps `B=ADMIT` when `A` is superseded, and collapses to `B=AUDIT_ONLY(REFUTED)` if `A` is refuted).
   - **Verified Truth-Table Lock:** (i) `A↔B + C COPY_OF A` $\to$ `{A:U, B:U, C:U}`; (ii) `A↔B + X→A` (ordinary) $\to$ `{X:T, A:F, B:U}` vs (declared lineage) $\to$ `{X:T, A:F, B:F}`; (iii) `A↔B + B→E` $\to$ `{A:U, B:U, E:U}`; (iv) `C REF+SUP B, B SUP A` (ordinary) $\to$ `{A:T, B:F, C:T}` (slot-conflict guard triggers `CONTESTED(MULTI_CURRENT)` unless explicit `C SUP A`) vs (declared lineage) $\to$ `{A:F, B:F, C:T}`.

---

### Target 2 — Scoped Anti-Laundering Authority Ceiling ($\tilde{\alpha}(v, \kappa)$) inside $K=4$ Stabilization

1. **Exact Edge-Ontology Separation (Solves the Auditor Refutation Paradox + Closes `AMENDS`-Laundering):**
   - **Epistemic Truth Dependencies $D_V$ (`COPY_OF`, `DERIVES_CURRENT`, `AMENDS`, `DERIVES_SNAPSHOT`):** Because `AMENDS` (`budget = r0 + $20`) logically relies on $V(r_0) = \mathbf{T}$, **$\tilde{\alpha}$ propagates unconditionally over ALL of $D_V$**, collapsing positive $D_V$-cycles to $\bot_{\text{auth}} = 0$.
   - **Non-Epistemic Audit/Refutation Edges ($\notin D_V$):** An auditor (`auth=4`) refuting a rumor emits `REFUTE(auditor, r_0)` and/or non-epistemic metadata reference **`CITES` / `AUDITS_RECORD`** ($\notin D_V, \notin D_C$). Thus the auditor keeps full `auth=4`! Independent re-verification of a rumor creates a new tier-4 primary record linked via `CITES` (never `D_V`).
2. **Combining Carbon's $\tilde{\alpha}$ with Astra's Slot/Capability-Scoped Authority (Zero $K=4$ Feedback):**
   - Define finite capability scope $\kappa = (\text{tenant}, \text{slot\_class}, \text{action} \in \{\texttt{ASSERT}, \texttt{REFUTE}, \texttt{SUPERSEDE}, \texttt{LINEAGE\_SUP}\})$ with raw authority $\alpha_{\text{raw}}(v, \kappa) \in \{0, \dots, L-1\}$ determined strictly by grants visible at `(vt_cut, tx_cut)` (scalar `authority_tier: int = 0` serves as backward-compatible uniform default).
   - Propagate independently per scope in **Stratum 2 (strictly AFTER Stratum 1 Visibility, strictly BEFORE Stratum 3 WFS)**:
     $$\tilde{\alpha}(v, \kappa) = \min\!\Big(\alpha_{\text{raw}}(v, \kappa),\; \min_{d \in D_V^{\text{vis}}(v)} \tilde{\alpha}(d, \kappa)\Big)$$
     with fail-closed rule: any visible $D_V$ edge pointing to an **invisible/missing parent** at `(vt_cut, tx_cut)` contributes $\bot_{\text{auth}} = 0$ (and $V_{\text{parent}} = \mathbf{U}$).
   - **Defeat Admissibility Gate:** Visible edge `(u, v, action)` enters $R^{\text{auth}}$ or $S^{\text{auth}}$ iff $\tilde{\alpha}(u, (\text{tenant}(v), \text{sc}(v), \text{action})) \ge \tilde{\alpha}(v, (\text{tenant}(v), \text{sc}(v), \texttt{ASSERT}))$; blocked edges emit `BLOCKED_INSUFFICIENT_AUTHORITY`.
   - **Incremental Maintenance under Deletions & Rewinds:** Because `min` is non-monotone on `del_dep` / rewind, maintain $\tilde{\alpha}$ on the forward $D_V$ dirty cone via iterative Tarjan positive-SCC collapse ($\to 0$) + **$L$-bucket monotone min-propagation** ($L \le 5$), guaranteeing strict **$O(|D_{\text{auth}}| + |E_{D_V}(D_{\text{auth}})|)$** linear time and exact bitemporal rewind parity while preserving the $K=4$ Stratified Stabilization Theorem.

---

### Target 3 — Locked Flagship #1 Equations (`A2c-Subspace+OOS`, `A2g`, `A2c-Polar`) & `f1_4096_ood` Status

1. **Locked `A2c-Subspace + OOS-Radius` (Default Multi-Hop & Cross-Dimension Spherical Bridge):**
   - Given L2-normalized fit pairs $(X_{\text{fit}}, Y_{\text{fit}})$ and disjoint calibration pairs $(X_{\text{cal}}, Y_{\text{cal}})$ (or internal 80/20 anchor split if none supplied):
     - Fail closed (`ValueError`) if $\|X_{\text{fit},c}\|_F$, $\|Y_{\text{fit},c}\|_F$, or $\sigma_1(X_{\text{fit},c}^\top Y_{\text{fit},c}) < 10^{-8}$.
     - Thin SVD $X_{\text{fit},c}^\top Y_{\text{fit},c} = U \Sigma V^\top$; retain rank $r = |\{i : \sigma_i > 10^{-6}\sigma_1\}|$ (optionally capped on DEV when $n_{\text{fit}}-1 < d_{\text{in}}$); set $Q_r = U_r V_r^\top$.
     - Calibrate held-out projected scale with **Carbon's verified up-projection floor** (replacing withdrawn `s_subspace`):
       $$s_{\text{oos}} = \frac{\|Y_{\text{cal},c}\|_F}{\max(\|X_{\text{cal},c} Q_r\|_F, 10^{-8})}, \qquad s = \begin{cases}\max(1.0, s_{\text{oos}}) & \text{if } d_{\text{in}} < d_{\text{out}} \\ s_{\text{oos}} & \text{if } d_{\text{in}} \ge d_{\text{out}}\end{cases}$$
     - Map $f_{\text{A2c}}(x) = \operatorname{norm}_2\big(s (x - \mu_x) Q_r + \mu_y\big)$ (with optional `A2c-Q` query-centroid $\mu_{q,\text{cal}}$ for asymmetric-prefix encoders and optional `CSLS` scoring).
2. **Locked `A2g` (`GeodesicPolarBridge` — Guarded Tangent Ablation):**
   - Poles $p_x = \bar{x}/\|\bar{x}\|_2$, $p_y = \bar{y}/\|\bar{y}\|_2$. **Mandatory Guard:** Fail over cleanly to `A2c-Subspace+OOS` (with receipt) if $\min(\|\bar{x}\|_2, \|\bar{y}\|_2) < \tau_{\text{pole}} = \mathbf{0.25}$ OR split-half pole drift $> 10^\circ$.
   - Construct orthonormal tangent bases $B_x \in \mathbb{R}^{d_x\times(d_x-1)}, B_y \in \mathbb{R}^{d_y\times(d_y-1)}$ via $O(d)$ **Householder reflection** ($p \mapsto e_1$, never forming $d\times d$ matrices!).
   - Exact `Log` / `Exp` via `atan2` and `sinc` ($\epsilon_{\text{cut}} = \mathbf{0.05}\text{ rad}$, **hard clamp — `tanh` withdrawn!**):
     $$c = \operatorname{clip}(x^\top p_x, -1, 1),\;\; h = x - c p_x,\;\; \theta = \min(\operatorname{atan2}(\|h\|_2, c), \pi - \epsilon_{\text{cut}}),\;\; \operatorname{Log}_{p_x}(x) = \frac{\theta}{\|h\|_2} h \;\; (\text{limit } h \text{ as } \|h\|_2 \to 0)$$
     Solve thin SVD of $(B_x^\top V_{x,\text{fit}}^\top)(V_{y,\text{fit}} B_y) = U_r \Sigma_r W_r^\top$ ($\sigma_i > 10^{-6}\sigma_1$), $Q_{\text{tan}} = U_r W_r^\top$, disjoint calibration $s_\theta = \|V_{y,\text{cal}}\|_F / \max(\|V_{x,\text{cal}} B_x Q_{\text{tan}}\|_F, 10^{-8})$, and for query $x$:
     $$w = s_\theta (\operatorname{Log}_{p_x}(x) B_x Q_{\text{tan}}) B_y^\top,\qquad r_{\text{eff}} = \min(\|w\|_2, \pi - \epsilon_{\text{cut}}),\qquad \operatorname{Exp}_{p_y}(w) = \cos(r_{\text{eff}})\,p_y + \frac{r_{\text{eff}}}{\max(\|w\|_2,10^{-12})}\operatorname{sinc}\!\left(\frac{r_{\text{eff}}}{\pi}\right) w$$
   - Log `frac_clamped`, split-half pole stability, retained rank, and sequential vs aligned-pole latency; explicitly state radial isometry / transverse non-conformality ($\sin(s_\theta\theta)/\sin\theta$). **Promotion rule:** Promote as optional narrow-cone mode **iff** paired `LCB95 > +0.005` over `A2c-Subspace+OOS` (and sequential `norm_2` `A2c`) on preregistered held-out queries including fallback coverage; otherwise record cleanly in the Falsification Ledger.
3. **`A2c-Polar` & `f1_4096_ood` Status:**
   - For $P = Q_1 \cdots Q_m = U\Sigma V^\top$, freeze rank cutoff on DEV, set $P_{\text{polar}} = U_r V_r^\top$, recalibrate endpoint scale/translation on disjoint CAL, log pre-polar $\sigma_{\min}$, $\kappa_+$, discarded energy, and require paired `LCB95 > 0` on peak chains (`384→1024→384`, `384→768→384`) for promotion.
   - `f1_4096_ood` (`ce81b930...`) remains registered in Appendix A Part II as **`MANIFEST_ONLY_PENDING_R4_RUNNER_AND_CSVS`**; its Hop-2 `-0.0255` loss is labeled as an empirical finding with falsifiable hypotheses (unobserved source complement + OOS radius + MP tail noise — explicitly NOT zero-singular SVD completion since $n-1=1999>1024$!).

---

### Target 4 — Locked FINAL `PM-1 .. PM-5` Pre-Release Gates (Zero Residual Numeric Drift)

| Gate | Locked Final Criterion (Unanimous Target for R3 Freeze) |
|---|---|
| **`PM-1a`** *(Tier-1 Structured `EK5 v3` $K=4$)* | **Oracle/Fuzzer:** `0` mismatches vs independent $\Gamma^2$ WFS oracle across exhaustive 2-node (`1,280`), 3-node (`262,144` static + `56,568` rank-only 2-step), and `100k` cyclic/batch/rewind/authority ops; `dirty<=1, edges<=5` on shielded `10k` in/out hubs. **Live `Zero-Shortcut v2` (`N=240 = 8×30`, persistent `EK5 v3` $K=4$, equalized fields & `top_k=5` token budget):** `C4_v3 >= 98.0%` acc, `0.0%` zombie & authority-DoS errors, `100%` oracle parity, beating `C2`, `C2b_jaccard`, `C3`, and Astra's 4-line rule each by two-sided 95% **`LCB95 > +15.0 pp`**. |
| **`PM-1b`** *(Tier-2 Blind Conversational)* | **Corpora:** Full `LongMemEval-S` (`N=500`, `_abs` reported separately) + 10-conv stratified `LoCoMo` (`N=300`, `30/conv`, compound IDs split on `[;,\s]+`, any-hit & all-hit recall, structured refusal parser; previously audited items tagged separately). Deterministic `batch_size=1` / controlled-order generation + prompt-SHA cache. **Clean Regime A Gate:** On primary `LongMemEval-S` full unmodified (`N=500`), **`point >= -1.0 pp` AND `LCB95 >= -3.0 pp`** vs frozen `M1_hybrid_flat` *(84.1% power at $q=0.05$; explicitly labeled "1.0 pp observed-loss cap with 3.0 pp 95% uncertainty bound", NOT statistical 1.0 pp non-inferiority which requires $N\approx 3,925$!)*. Report `single-session-user` slice (`point >= -1.0 pp`, exact matched-pair CI) and `LoCoMo` Regime A (`N=300`, `point >= -1.0 pp`, **10-conv cluster bootstrap CI**) separately. **Stress Regime B Gate (Question-Blind & Gold-Blind Neutral Echoes, zero `"from earlier chat"` cues):** **`>= +25.0 pp` AND `LCB95 > +15.0 pp`** vs frozen `M1_hybrid_flat` on `LongMemEval-S` KU (and **`LCB95 > +10.0 pp`** on `LoCoMo`). Include `M1c_hybrid_jaccard_dedup` & `C2b_bm25_jaccard_dedup`; claiming beyond-dedup novelty requires **`LCB95 > 0.0 pp` vs `M1c`**. |
| **`PM-2`** *(Coupled Dual-Drift Proof Engine)* | Single persistent Proof Engine v1 crossing `384↔1024↔2560↔4096` (both projection directions + interior peak `384→1024→384` + explicit fallback) + `TRA-1` (**`0/12` hostile leaks**) + persistent `EK5 v3` ($K=4$): bridged-minus-native accuracy **absolute `point >= -3.0 pp` AND `LCB95 >= -5.0 pp`** *(native ratio reported as diagnostic only)*; **`>= +20.0 pp` (`LCB95 > +12.0 pp`)** over bridged flat under echo stress; **`LCB95 > 0`** over Composed Centered Affine Ridge on preregistered primary multi-hop endpoint (both linear and $2^r-1$ nDCG@10 published; `B4` encode/write cost reported separately). |
| **`PM-3`** *(Pinned Upstream Comparators)* | Pin unmodified `mem0ai/mem0` and `getzep/graphiti` (`graphiti-core`) with exact commit SHAs, adapter revisions, matched embedder/LLM/writes/retrieved-token ceiling; check `ScaleMAS` availability at run time (emit dated `UNAVAILABLE_UPSTREAM_ARCHIVE` + `C1_bounded` mechanism comparator if absent); 1-factor-at-a-time `C4_v3` ablation. **Mandatory Original Gate:** positive stale-error reduction (`LCB95 > 0`) at matched-or-lower retrieved tokens. **Labeled `R1-STRETCH` Sub-Gate (non-retroactive):** `LCB95 > +10.0 pp` stale-error reduction (or `>= 25%` token reduction at non-inferior accuracy) vs upstream on multi-session conflict slices. |
| **`PM-4`** *(Multi-Generator & Lock-Free Blind Multi-Judge)* | `>= 2` generator families × `>= 2` independent blinded judge families (preregistered pair **Cohen's $\kappa \ge 0.85$** with 95% CI & confusion table) + **`100`-item stratified human audit** (25 agreements, 25 disagreements, 25 `_abs`/abstentions, 25 relative-date/negation). Two-phase offline lock-free grading over unique `(qid, norm_pred)` keys (`0` inconsistent keys!). `paired_bootstrap_delta` fails closed (`ValueError`) on missing/mismatched IDs; whenever discordant count $n_{10}+n_{01}<10$, enforce **exact matched-pair / Newcombe-Wilson score interval** alongside bootstrap. Primary win direction preserved in every cell (`LCB95 > 0` required per cell claiming superiority). |
| **`PM-5`** *(Service SLO, `1M` Kernel, & 100% Receipts)* | **(1) `100k`-vector End-to-End Service SLO:** externally timed `accept→WAL/EK5 v3/index→TRA-1→A2c/A3→retrieval→receipt` **`p95 <= 25.0 ms`** (report p50/p95/p99, QPS, RSS, restart parity). **(2) `1M`-node `EK5 v3` Kernel:** disjoint trees, power-law cyclic ($\gamma\approx 2.3$, 10% `REFUTE`, 2% cycles), `10k` in/out hubs, dense cascades (fallback ON/OFF) vs independent $\Gamma^2$ oracle: **`0` spurious full scans on sparse/no-op/shielded/authority-blocked writes**, reporting changed labels **and ranks**, incident edges, fallback reasons, and measured crossover range. **(3) Receipts:** `100%` SHA-256 coverage of all new-run CSVs/JSONLs/manifests/summaries + mandatory `git_sha`. |

---

### Target 5 — Locked Whitepaper v2 Chapter Order, HTML v2 Spec, & Single-Source Builder Schema

1. **Unified 8-Chapter + Appendix A Whitepaper v2 Order:**
   - **Front Matter:** Pre-Release Gate Banner, Abstract, and 3-Claim Scoped Boundary Table (separating Flagship #1 Transport, Flagship #2 Tier-1 `EK5 v3`, and Flagship #2 Tier-2 `M4c`; placing historical clean `42/50` vs `46/50` beside Regime B).
   - **Ch. 1:** Dual-Drift Thesis, Scoped Claims & Release Status.
   - **Ch. 2:** Metric Governance (`mlr-v1`, linear vs $2^r-1$ gain correction, two-sided 95% CI 2.5th percentile, sparse score CI), `TRA-1` ABI (`TRA1_OR_M4_NEEDS_REVIEW` + `8/8` hostile blocks), & Nested `DEV/SELECT/REPORT`.
   - **Ch. 3:** Flagship #1 Geometry (Three-Regime Stiefel Trichotomy Theorem 1 + `A2c-Polar`, Repaired Prop. 1b + OOS Radius + Up-Projection `max(1.0, s_oos)`, OOD Centroid-Collapse Proposition, `A3_subspace` $d_{\text{out}}>n$ Shrinkage Proof, Guarded `A2g` Non-Conformality & Aligned-Pole Theorem).
   - **Ch. 4:** Flagship #1 Empirical Results (6-BEIR, 4-hop, `f1_4096_ood` 10.7× jumps & 18-cell OOD with provenance caveats, 11-arm cached ablation, Scoped Secondary `B4` Backfill).
   - **Ch. 5:** Flagship #2 Semantics & Maintenance (`COPY_OF`/`AMENDS`/`CITES`, Direct-$B_i$+$B_i^{\text{cyc}}$ vs Declared Version-Lineage, Scoped Anti-Laundering $\tilde{\alpha}(v,\kappa)$, Stratified $K$-Axis Theorem $K\in\{2,4,6,8\}$, Linear-Time Positive-SCC + Lyapunov Pitfall A/B Theorem + 3-node/5-node stale-rank proofs).
   - **Ch. 6:** Flagship #2 Empirical Evaluation (`Zero-Shortcut v2`, `1M` Kernel + `100k` Service, full `LongMemEval-S` $N=500$ + 10-conv `LoCoMo` $N=300$ separating clean/stress/abstention + `M1c_hybrid_jaccard_dedup`).
   - **Ch. 7:** Failure Forensics, Archive Repair (`pristine-b37e6a55` BOM/LF vs `-exact`), Pruned Hypotheses (`A4/A6`, `P1`, `P7`, `M4 QPP`, `IQ-01`, `s_subspace`, `A2g tanh`), & Related Work.
   - **Ch. 8:** Integrated Proof Engine v1 & Locked `PM-1..PM-5` Pre-Release Gates.
   - **Appendix A:** Two-Part (+ Pending Part III) Cryptographic Evidence Ledger — **15/15** verified primary manifests (`12` Part I Historical incl. `s1` `1,226.40 s`, `2/2` S1 ancillary JSON digests `s1_m2_preflight.json` & `selection_lock.json`, `f2_iq04` full 64-char `148d80bf3faadd77f9ad0f9eeb1f1a089d8e87b89025c9f645d4a685ad7e619b`, `f2_longmemeval` `SUPERSEDED_BY: f2_longmemeval_v2`, `iq01` `FALSIFIER`, and `19` historical CSVs explicitly marked `UNHASHED_IN_HISTORICAL_RECEIPT`; `3` Part II R3 bundles with `6/6` verified `artifact_sha256` entries + `f1_4096_ood` marked `MANIFEST_ONLY_PENDING_R4`).
2. **Locked Presentation HTML v2 Specification:**
   - **Color Contrast (AAA):** Lock `--warm: #7d2f0a` (**`8.36:1` on `#fdf2ec`**, **`9.20:1` on `#ffffff`** — full WCAG AAA).
   - **Zero-JS Static Rendering & Accessibility:** All tables, KPIs, 4-decimal unrounded-float deltas, full 64-char digests (`overflow-wrap: anywhere`), native `<math display="block">` MathML (zero raw TeX literals), `.release-status-banner` inside `<header class="hero">` (always visible in `@media print`), `<div class="table-scroll" tabindex="0" role="region">` down to `375px`, ARIA `role="tablist"` keyboard tabs, and `"CI unavailable (aggregate-only manifest)"` disclosure for `f1_4096_ood`.
   - **Locked Static-SVG Figure List (6 Panels, Each Backed by a Semantic Table):** **Fig. 1A** 6-Domain BEIR 4-Hop `2×3` Small Multiples; **Fig. 1B** `384↔4096` Dimension-Jump (`query_dim → index_dim` labeled) & 18-Cell OOD Matrix; **Fig. 2** 4-Panel Aligned `0–100%` CI-Whisker Chart (Structured / Clean Regime A incl. `42/50` vs `46/50` / Stress Regime B incl. `M1c` / Abstention); **Fig. 3** Separate `1M` Kernel Topology + Crossover vs `100k` Service SLO; **Fig. 4A/4B** Accessible Concept SVGs (`A2c` vs `A2g` + `TRA-1`, and $K=4$/$K=8$ Stratum DAG).
3. **Locked Single-Source Builder (`scripts/build_whitepaper_assets.py` + `docs/research/bundle_policy.json`):**
   - Pure-stdlib 5-stage pipeline: `discover_attempts → verify_bytes → normalize_observations → validate_claims → render`.
   - Enforces: every `report_receipt.json` on disk catalogued in `bundle_policy.json`; SHA-256 recomputed on all `15/15` primary manifests, `6/6` R3 `artifact_sha256` files, and `2/2` S1 ancillary JSONs (`s1_m2_preflight.json`, `selection_lock.json`); `19` unhashed historical CSVs record `verification_status: "UNHASHED_IN_HISTORICAL_RECEIPT"` + separate `builder_observed_sha256` (never retroactively forged into historical receipts); canonical `docs/research/verified_registry.json` emitted with full observation tuple (`attempt_id, relative_path, manifest_sha256, git_sha, verification_status, method, dataset, fit_domain, eval_domain, fit_dim, query_dim, index_dim, hop, split, seed, metric, gain_convention, denominator, estimate, ci_method, ci_low, ci_high, derived_from_rounded, supersession_status, null_reason`); deterministic `--check` CI gate fails non-zero on any 1-byte drift across Markdown fenced blocks (`<!-- BEGIN:AUTO:<id> -->`) and HTML v2.

---

## 3. Remaining Micro-Dissent / Open Items for R3 Final Adjudication

All five R2 targets have achieved near-total technical convergence. Only **two tiny implementation-level choices** remain for unanimous R3 sign-off:
1. **`EK5 v3` Default Rank-Propagation Queue (`Plain FIFO` vs `Dijkstra / Bucket`):** Plain FIFO inside `D` is strictly $O(|D|+|E_{\text{inc}}(D)|)$ and passed `0/1,018,224` 2-step + `0/50,000` batch tests, but lazy upper-bound $\rho$ can drift upward over $M$ repeated edits (`O(M)`, harmless in `int64`, reset on checkpoint). Confirm in R3 whether default `EK5 v3` ships **Plain FIFO + periodic checkpoint rank compaction** (my recommendation for true linear worst-case bound) or **Dijkstra min-heap** ($O(|D|\log|D|)$ for strictly bounded $\rho \le |V|$).
2. **`PM-1b` Clean Reporting Floor on `LoCoMo` (`N=300`, 10 Conversations):** Because `10` conversations yield coarse cluster-bootstrap quantiles, confirm in R3 whether `LoCoMo` Regime A (`N=300`) enforces `point >= -1.0 pp` + **informational** 10-conv cluster CI (keeping the formal `LCB95 >= -3.0 pp` non-inferiority gate strictly on independent `LongMemEval-S N=500`), or sets an explicit numeric cluster-LCB floor on `LoCoMo`.
