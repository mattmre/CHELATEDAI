<!--
{
  "taskId": "node:delegated-task:command%3Amcp%3A3b84a70c-9824-441c-8750-b9902399a932%3Adelegate-task%3Acouncil-r2-carbon-20261121",
  "link": "[Council Refinement R2 Carbon : Cross-Check Peer R1 Proofs & Converge...](t3-thread://v2/c5bb8594-9c79-4948-9251-bd83e8527cf6/thread:delegated-task:command%3Amcp%3A3b84a70c-9824-441c-8750-b9902399a932%3Adelegate-task%3Acouncil-r2-carbon-20261121)",
  "model": "gemini-4-carbon",
  "executedModel": null,
  "options": [
    {
      "id": "thought_level",
      "value": "high"
    }
  ],
  "status": "completed"
}
-->

# Tri-Model Doctoral Research Council — Refinement Round 2 Report (`Carbon`)

**Council Member:** Gemini 4 Carbon (`thought_level=high`)  
**Stage:** **Refinement Round 2 of 3** (Peer R1 Proof Cross-Check, Adversarial Stress-Testing, & Final Specification Lock)  
**Target Checkout:** `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1` (`HEAD 18973789615ecab1268a77ebd3c69a283607241c`)  
**Inputs Cross-Checked:** [r2-synthesis.md](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r2-synthesis.md), [r1-astra.md](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-astra.md), [r1-neon.md](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-neon.md), [r1-carbon.md](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-carbon.md), [f1_4096_ood_manifest.json](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/artifacts/flagship1-night1/20261121-rigor-r3/f1_4096_ood/f1_4096_ood_manifest.json), and [s1_m1_m3_manifest.json](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/artifacts/flagship1-night1/20261121-flagship1-r1/s1/s1_m1_m3_manifest.json).  
**Read-Only Verification Discipline:** Zero repository edits, zero git state writes, zero subagents; all theorems, edge cases, and statistical power calculations were executed in memory via `python -B`.

---

## 1. Peer R1 Cross-Check Audit Table (Including Carbon Self-Audit)

| Peer / Source | R1 Claim / Proposed Algorithm | R2 Verdict | Executable Evidence / Edge Case Found (`python -B` Probes) |
|---|---|---|---|
| **Astra 6.2 XHigh** ([r1-astra.md:52-66](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-astra.md#L52-L66)) | 3-node stale-rank counterexample (`B→A, C→A` + `add A→B` + `del C→A`; `56,568` rank-only edits / `48` false skips) + separate raw Fitting rank (`rank = ∞` for GUS-eliminated positive cycles). | **Counterexample CONFIRMED; `raw_rank = ∞` NEEDS 1 REFINEMENT** | Confirmed against label-only propagation ([iq04_temporal_evidence.py:940](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/iq04_temporal_evidence.py#L940) / [xc-carbon.md:47](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/xc-carbon.md#L47)). **Subtle edge case found in Astra's `raw_rank = ∞`:** when a positive cycle $A \leftrightarrow_{\text{DEP}} B$ (`F` via GUS) attacks a `10k` out-hub (`A --REF--> hub --REF--> w_1..w_10000`), `raw_rank[A] = ∞` forces `raw_rank[hub] = ∞`, setting `shield_cnt[w_i] = 0` and disabling $O(1)$ fast cuts across the entire downstream subgraph. Setting `wfs_rank = 0` with an explicit `pos_cyc[v] = True` bit achieves **`0 / 100,000` single+batch mismatches** while preserving finite downstream shields (`rank[hub]=1, rank[w_i]=2`). |
| **Neon XHigh** ([r1-neon.md:69-85](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-neon.md#L69-L85)) | 5-node stale-rank counterexample against mixing unranked backward skips with `strict_t_att_cnt` + `UnifiedWFSStore` Lyapunov Dijkstra ranks. | **Counterexample CONFIRMED; Unpruned Cone NEEDS Pitfall A/B** | Verified `0` mismatches on Neon's 5-node witness. **Edge case found:** without Carbon's Pitfall A cone pruning (`shield_cnt[w] - loss_D[w] > 0`), a rank-only shift `x(F,1) --REF--> u(T,0)` on a `10k` out-hub whose targets `w_1..w_10000` are already shielded by `r0(T,0)` puts all `10,001` nodes into `D`; with Pitfall A+B, **`|D| = 1`** and external shield deltas update in $O(1)$ per outgoing edge of `u`. |
| **Carbon (Self-Audit)** ([r1-carbon.md:60-70](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-carbon.md#L60-L70)) | `Theorem 1.1` Pitfall A/B cone pruning under rank-only increases (`u: T(0) -> T(2)`) & atomic multi-edge batches. | **SURVIVES Probe A; Batch Ordering LOCKED** | Tested on Astra's 3-node witness (`D={'B'}`, `rank[B]: 0->2`, `shield[A]: 2->1`, then `del C->A` $\implies$ `A=U, B=U`) and a 12-node rank-only cascade (`x(F,1)->u(T,0)` with `shield[w]=1` from `u(T,0)` + non-shield `z(T,2)`): `loss_D[w]=1` yields `1 - 1 = 0 <= 0`, **correctly pulling `w, q, t` into `D`** (`ranks: 0->2, 1->3, 2->4, 3->5`, `0` label changes). **Batch catch:** atomic multi-edge batches/rewinds MUST apply deletions to `shield_cnt` *before* evaluating `add` skips and guard `pos_cyc` skips when `any_dep_del` holds (`0 / 100,000` batch mismatches). |
| **Neon XHigh** ([r1-neon.md:38-48](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-neon.md#L38-L48)) & **Carbon** ([r1-carbon.md:45-46](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-carbon.md#L45-L46)) | Mutual-`SUPERSEDE` pure-WFS rule $B_i^{\text{cyc}} \leftarrow V_j \wedge \operatorname{not} B_i^{\text{cyc}}$ vs. SCC external dominance. | **2 EDGE CASES CAUGHT & RESOLVED** | Executed in Python (`Probe 3`): (i) $A \leftrightarrow_{\text{SUP}} B + C \xrightarrow{\text{COPY\_OF}} A \implies \{A:\mathbf{U}, B:\mathbf{U}, C:\mathbf{U}\}$ (**PASS**). (ii-a) $A \leftrightarrow_{\text{SUP}} B + X \xrightarrow{\text{SUP}} A$ only $\implies \{X:\mathbf{T}, A:\mathbf{F}, B:\mathbf{U}\}$: **proves Carbon R1:46 was WRONG** to let $X \to A$ transitively erase $B$ via the corrupted cycle edge $(A \to B)$ (*Bob-branch erasure bug*!). (ii-c) $A \leftrightarrow_{\text{SUP}} B + B \xrightarrow{\text{SUP}} Y$: Neon's formula gave $C_Y = \mathbf{F}$ (`B_Y^ext = V_B = T`), violating **Astra R1:44** (*quarantined cycle members supply at most possible `U` replacement force externally*). Fixed in unified Stratum 2 below. |
| **Carbon (Self-Audit)** ([r1-carbon.md:83-85](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-carbon.md#L83-L85)) & **Astra** ([r1-astra.md:91](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-astra.md#L91)) | Effective authority ceiling $\tilde{\alpha}(v)$ over $D_V$ vs. auditor citations & capability-scoped authority. | **REFINED & UNIFIED** | **Self-catch:** An auditor (`auth=4`) refuting/quoting a rumor `r_0` (`auth=0`) MUST use `CITES_AUDIT` ($\notin D_V$), NEVER `DERIVES_SNAPSHOT` ($\in D_V$), because `r_0 ∈ D_V(aud)` + `aud REFUTE r_0` creates $V_{r_0} \leftarrow \neg V_{r_0}$ (`U` self-deadlock even in $K=2$!). Furthermore, defense armor MUST also use $\tilde{\alpha}_q(w, c_w)$ (not $\alpha_{\text{raw}}$), otherwise a rumor-summary (`raw=3, eff=0`) retains tier-3 armor against a tier-2 clinician refutation! |
| **Astra 6.2 XHigh** ([r1-astra.md:105-120](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-astra.md#L105-L120)) | `f1_4096_ood` precision catches: zero Legal-trained bridges, `0.0000–0.1704` OOD Ridge range, full thin rank `1024` at `SciFact` Hop 2 (`n-1=1999>1024`). | **100% CONFIRMED (+ NEW DISK CATCH)** | Verified in [f1_4096_ood_manifest.json](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/artifacts/flagship1-night1/20261121-rigor-r3/f1_4096_ood/f1_4096_ood_manifest.json). **New R2 forensic catch:** `scratch/run_pristine_f1_4096_ood.py` does **not** exist on disk; both `SciFact` (`0.6866/0.7266` vs `0.6484/0.7645`) and `CoSQA_Code` (`0.2827/0.3465` vs `0.2831/0.3461`) shift between `indomain_dim_jumps` (post-DEV split) and `zero_shot_ood` (full query set), AND [s1_per_query_paired.jsonl](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/artifacts/flagship1-night1/20261121-flagship1-r1/s1/s1_per_query_paired.jsonl)'s 300-query `all-MiniLM-L6-v2` SciFact score is **`0.6451 ≠ 0.6484`**, proving `D384` encoder/loader provenance in `f1_4096_ood` was undocumented! |
| **Astra** ([r1-astra.md:123-145](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-astra.md#L123-L145)) & **Neon** ([r1-neon.md:145-146](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-neon.md#L145-L146)) | `A2g` `atan2`, non-conformality (`s` vs `sin(sθ)/sin(θ)`), aligned-pole composition, vs. Neon's `0.95π tanh`. | **ASTRA CONFIRMED; NEON `0.95π tanh` FALSIFIED** | Verified `atan2`+2-step GS gives `0.000e+00` rel error at $\theta=10^{-9}$ vs **`100%` error** for `arccos`; aligned-pole 2-hop composition matches sequential relay to **`7.23e-16`**. **Crucial catch against Neon's global `0.95π tanh`:** on Astra's 1.8x angular dilation test (`A2c = 5.2653°`), Neon's `0.95π tanh` suffered **`3.4709°` shrinkage error**, whereas a **$C^1$ exact soft-collar** (`r <= π-0.10 ? r : r0 + 0.05 tanh((r-r0)/0.05)`) achieved **`3.96e-7°`** while remaining $C^1$ smooth and strictly monotone! |
| **Carbon** ([r1-carbon.md:171](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-carbon.md#L171)) vs. **Neon** ([r1-neon.md:173](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-neon.md#L173)) | Up-projection (`d_in < d_out`): Carbon's `s >= 1.0` cap vs. Neon's `s_subspace = ||Y_val,c P_Q||_F / ||X_val,c Q||_F`. | **NEON `s_subspace` FALSIFIED; CARBON `s>=1` SPLIT BY REGIME** | In `Probe 2` (`U1 384->4096` & `U2 wide->narrow`), **`s_subspace` lost on EVERY metric** (`U2` CSLS `0.1533` vs `0.2200` for `s=1.0` and `0.1900` for `s_oos_amb`) because shrinking $s$ inflates query-independent hub bias $\frac{1}{s}\langle\mu_y, d_j\rangle$. Meanwhile, `s >= 1.0` **wins 1-hop terminal retrieval** (even on wide$\to$narrow cones!) **but hurts sequential 2-hop cone geometry** (`0.9931` vs `0.9999` for `s_oos_amb` and `1.0000` for `Collapsed A2c-Polar+OOS`). |
| **Astra** ([r1-astra.md:167-170](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-astra.md#L167-L170)) vs. **Carbon/Neon** | `PM-1b` power math (`N=3,925` for `-1pp` LCB; `-3.0pp` on `N=500` vs Neon `-2.5pp` vs Carbon `N=800` pooled; zero-discordance bootstrap collapse). | **CONCEDED TO ASTRA (`-3.0 pp` on `N=500` + Tango Score CI)** | Exact multinomial probe at $N=500, q=0.05, \Delta=0$: `pwr(-1pp) = 17.3%`, `pwr(-2pp) = 50.9%`, Neon's `pwr(-2.5pp) = 69.6%` (`<80%`!), **Astra's `pwr(-3.0pp) = 84.1%` (`>=80%`)**! Carbon concedes i.i.d. `N=800` pooling because `LoCoMo`'s 300 items share 10 conversations (`df=9` clusters). Tango (1998) score CI gives `-0.76 pp` at `0/500` and `-5.20 pp` at `0/70` discordance vs naive bootstrap's degenerate `0.00 pp`. |
| **Astra** ([r1-astra.md:183](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-astra.md#L183)) & **Neon/Carbon** | `15/15` manifests, `6/6` R3 artifact SHAs, `2/2` S1 ancillary SHAs, `19` unhashed historical CSVs, WCAG AAA `--warm`. | **100% VERIFIED ON DISK** | Verified all `15` receipts, `6` R3 side-artifact SHAs, `2` S1 ancillary SHAs, and exact count of `19` unhashed historical CSVs. Exact sRGB contrast on `#fdf2ec`: `#7d2f0a` = **`8.36:1`** (`9.20:1` on `#fff`), `#8a3208` = **`7.51:1`** (`8.27:1` on `#fff`) — both pass WCAG AAA (`>= 7.0:1`). |

---

## 2. Resolved Final Specifications for Targets 1–5 (Ready for R3 Freeze)

### Target 1 — Unified `EK5 v3` Rank-Shielded Positive-SCC WFS & Stratum 2 Currency

#### 1A. Exact Stratum 1 Algorithm (`EK5 v3`: Positive-SCC + WFS-Rank + Pitfall A/B Cone Pruning)
Maintain for each visible node $v$: truth $\text{val}[v] \in \{\mathbf{T}, \mathbf{U}, \mathbf{F}\}$, cycle-free WFS rank $\rho[v] \in \mathbb{N} \cup \{\infty\}$ ($\rho[v] = \infty \iff \text{val}[v] = \mathbf{U}$; $\rho[v] = 0$ if $v$ is an unattacked root fact **or** collapsed by a nontrivial positive $D_V$-SCC), explicit flag $\text{pos\_cyc}[v] \in \{\text{False}, \text{True}\}$, and strict external falsifier count:
$$\sigma[w] = \big|\{u \in R^{\text{auth}}(w) : \text{val}[u]=\mathbf{T} \land \rho[u] < \rho[w]\}\big| + \big|\{d \in D_V(w) : \text{val}[d]=\mathbf{F} \land \rho[d] < \rho[w]\}\big| \quad (\text{for } \text{val}[w]=\mathbf{F};\; 0 \text{ otherwise}).$$

```python
def apply_stratum1_batch(store, add_edges, del_edges):
    # 1. Apply deletions FIRST, decrementing shield_cnt[head] before evaluating add skips
    del_actual, add_actual, seeds = [], [], set()
    for (s, t, et) in del_edges:  # REF: s refutes t (head=t); DEP: s depends on t (head=s)
        if remove_active_edge(store, s, t, et):  # decrements shield_cnt[head] if was strict shield
            del_actual.append((s, t, et, t if et == "REF" else s))
    for (s, t, et) in add_edges:
        if insert_active_edge(store, s, t, et):  # increments shield_cnt[head] if currently strict shield
            add_actual.append((s, t, et, t if et == "REF" else s))

    any_dep_del = any(et == "DEP" for (_, _, et, _) in del_actual)
    for (s, t, et, h) in del_actual:
        if store.val[h] == "T": continue
        if et == "REF" and store.val[h] == "F" and ((store.pos_cyc[h] and not any_dep_del) or store.shield_cnt[h] > 0): continue
        if et == "DEP" and store.val[h] == "F" and (not store.pos_cyc[h]) and store.shield_cnt[h] > 0: continue
        if et == "REF" and store.val[h] == "U" and store.val[s] == "F": continue
        if et == "DEP" and store.val[h] == "U" and store.val[t] == "T": continue
        seeds.add(h)
    for (s, t, et, h) in add_actual:
        if store.val[h] == "F" and ((store.pos_cyc[h] and not any_dep_del) or store.shield_cnt[h] > 0): continue
        if et == "REF" and store.val[s] == "F" and store.rank[s] < store.rank[h]: continue
        if et == "DEP" and store.val[t] == "T" and store.rank[t] < store.rank[h]: continue
        seeds.add(h)
    if not seeds:
        return set()

    # 2. Forward cone D with Pitfall A shield-loss tracking (pulls w in iff shield_cnt[w] - loss[w] <= 0!)
    D, q, loss = set(seeds), deque(seeds), defaultdict(int)
    while q:
        u = q.popleft()
        for (w, was_sh) in iter_outgoing_influences_with_shield_flag(store, u):
            if was_sh: loss[w] += 1
            if w not in D:
                if store.val[w] == "F" and (not store.pos_cyc[w]) and (store.shield_cnt[w] - loss[w] > 0):
                    continue
                D.add(w); q.append(w)

    out_was = snapshot_boundary_shields(store, D)  # {(u, w, et): was_shield} for u in D, w not in D

    # 3. Tarjan Positive-SCC on dep[D] -> collapse nontrivial SCCs / self-loops to (val='F', rank=0, pos_cyc=True)
    cyc_nodes = tarjan_nontrivial_positive_sccs(store.dep, D)
    new_val = propagate_kripke_kleene_queue_with_frozen_boundary(store, D, cyc_nodes)
    new_rank = dijkstra_boundary_aware_lyapunov_ranks(store, D, new_val, cyc_nodes)  # cyc_nodes seeded at rank 0

    # 4. Commit (val, rank, pos_cyc) on D, recompute shield_cnt[v] for v in D, and O(1) Pitfall B delta for w not in D
    commit_and_update_boundary_shields(store, D, new_val, new_rank, cyc_nodes, out_was)
    return D
```

- **Exact Complexity Statement (Astra + Neon + Carbon Unified):** Truth propagation runs in strictly linear $O(|D| + |E_{\text{incident}}(D)|)$; boundary-aware rank propagation runs in $O(|D|\log|D| + |E_{\text{incident}}(D)|)$ with a binary heap (or $O(|D| + \Delta\rho_{\max} + |E_{\text{incident}}(D)|)$ with radix/Dial buckets), and $O(1)$ (`|D| = 0` or `|D| = 1`) on shielded `10k` in/out hubs.

#### 1B. Locked Stratum 2 Temporal Currency Contract (Resolving All 3 Sub-Cases of Probe B)
1. **Two-Mode Positive Dependencies:** $D_C = \{\texttt{COPY\_OF}, \texttt{DERIVES\_CURRENT}\} \subseteq D_V$; $D_V \setminus D_C = \{\texttt{AMENDS}, \texttt{DERIVES\_SNAPSHOT}\}$; audit/refutation quotes use $\texttt{CITES\_AUDIT} \notin D_V$. Legacy `(u, v, DERIVES_FROM)` normalizes to `AMENDS` iff `u` also supersedes `v`, else `DERIVES_CURRENT`.
2. **Ordinary vs. Declared Version Lineage:**
   - **Ordinary `SUPERSEDE` (`lineage_id = None`, default):** Pure **Direct-$B_i$** ($B_i^{\text{ord}} = \bigvee_{j \in S^{\text{auth}}_{\text{ord}}(i)} V_j$ in Kleene $\{\mathbf{F} < \mathbf{U} < \mathbf{T}\}$). A refuted node ($V_j = \mathbf{F}$) exerts **zero** replacement force (`C REF+SUP B, B SUP A` leaves $A$ and $C$ both `CURRENT`, triggering `CONTESTED(MULTI_CURRENT)` on single-valued slots unless explicit `C SUP A` is emitted).
   - **Declared Version Lineage (`lineage_id != None`, matching `(tenant, entity, slot, lineage_id)` and authorized at $\Sigma_0$):** Propagate `definite_blocker` ($\exists$ strict lineage ancestor $h$ with $V_h = \mathbf{T}$) and `possible_blocker` ($\exists$ strict lineage ancestor $h$ with $V_h = \mathbf{U}$) in $O(|V| + |E|)$ on the acyclic condensation DAG of active lineage edges. Pure `X REFUTE B` (without `SUPERSEDE_LINEAGE`) leaves no valid superseder head above `B` and **reinstates `A` (`C_A = T`)**, whereas authorized `C REF+SUP_LINEAGE B, B SUP_LINEAGE A` covers `A` (`C_A = F, C_C = T`).
3. **Mutual `SUPERSEDE` Cycles ($A \leftrightarrow_{\text{SUP}} B$ & Self-Loops):**
   - Native write path rejects self-loops and cycle-closing `SUPERSEDE` edges with `SupersedeCycleError`.
   - On imported/bitemporal logs, partition $E_{\text{SUP}}^{\text{act}} = \{(u, v) : V_u \neq \mathbf{F}\}$ into Tarjan SCCs. Call $u$ **cycle-quarantined** iff $|\text{SCC}_{\text{SUP}}(u)| \ge 2$ or $(u, u) \in E_{\text{SUP}}^{\text{act}}$.
   - **Intra-cycle rule:** Every cycle-quarantined node $i$ receives $B_i^{\text{cyc}} = \mathbf{U}$ (`C_i = U` unless directly overridden by an unquarantined external $X$ with $V_X = \mathbf{T}$).
   - **External non-forwarding & Bob-branch safety rule:** Quarantined cycle edges **never forward transitive dominance** (so $X \xrightarrow{\text{SUP}} A$ on $A \leftrightarrow_{\text{SUP}} B$ gives $C_X = \mathbf{T}, C_A = \mathbf{F}, C_B = \mathbf{U}$, requiring explicit $X \xrightarrow{\text{SUP}} B$ to resolve $B$), and a quarantined node $B$ supplies **at most $\mathbf{U}$ (`possible`) blocker force** on outgoing edges $B \to Y$ (`C_Y = U` absent an unquarantined valid superseder of $Y$).
4. **Slot Reader Rule:** On a declared single-valued `(tenant, entity, slot)`, if $\ge 2$ `ADMIT` records assert distinct values **or** any unresolved `CONTESTED` (`V=U` or `C=U`) record exists for that slot with a distinct value not superseded by an `ADMIT` record, abstain with `CONTESTED` + witness IDs.

---

### Target 2 — Capability-Scoped Effective Authority Ceiling $\tilde{\alpha}_q(v, c)$ (`K=4` Stratum $\Sigma_{\text{auth}}$)

1. **Domain/Capability Scope + Foundation-Trust Ceiling:** Each event $v$ declares capability domain $\text{dom}(v)$ (e.g., `"clinical"`, `"user.preference"`, `"sensor.telemetry"`, default `"general"`), and grants/principals visible & unrevoked at $q = (vt_{\text{cut}}, tx_{\text{cut}})$ assign $\alpha_{\text{raw}, q}(v, c) \in \{0, \dots, L-1\}$. Over visible $G_q = (V_q, E_q)$, define:
   $$\tilde{\alpha}_q(v, c) = \begin{cases} 0 & \text{if } v \text{ lies in a nontrivial } D_V^q\text{-SCC or } \exists d \in D_V(v) \text{ invisible at } q, \\ \min\!\Big(\alpha_{\text{raw}, q}(v, c),\; \min_{d \in D_V^q(v)} \tilde{\alpha}_q(d, c)\Big) & \text{otherwise.} \end{cases}$$
2. **Audit/Refutation vs. Premise Reliance:**
   - Forensics/auditors (`auth = 4`) quoting or refuting a low-trust rumor $r_0$ (`auth = 0`) emit `CITES_AUDIT(aud, r_0)` ($\notin D_V$) + `REFUTE(aud, r_0)` (preserving $\tilde{\alpha}_q(\text{aud}, c) = 4$ and avoiding the $V_{r_0} \leftarrow \neg V_{r_0}$ deadlock).
   - Standalone re-attested replacements emit `SUPERSEDE` + `CITES_AUDIT` ($D_V = \emptyset$, keeping full $\alpha_{\text{raw}}$).
   - Unattested derivations (`COPY_OF`, `DERIVES_CURRENT`, `DERIVES_SNAPSHOT`, unattested `AMENDS` $\in D_V$) inherit the premise ceiling $\tilde{\alpha}_q$, blocking rumor-laundering.
3. **Symmetric Sword-and-Shield Qualification & Incremental Maintenance:**
   - A visible edge $e = (u \to w) \in \{\texttt{REFUTE}, \texttt{SUPERSEDE}\}$ is active in $E_q^{\text{auth}}$ **iff** $\tilde{\alpha}_q(u, \text{dom}(w)) \ge \tilde{\alpha}_q(w, \text{dom}(w))$ (so a rumor-tainted summary `raw=3, eff=0` has neither tier-3 attack power nor tier-3 defense armor!).
   - Under deletions or bitemporal rewinds, Tarjan SCC + topological min-propagation recomputes $\tilde{\alpha}_q$ on the dirty $D_V$-cone in $O(|D_{\text{auth}}| + |E_{\text{incident}}(D_{\text{auth}})|)$ time and emits exact edge-activation flips $\Delta E_{\Sigma_1}^\pm, \Delta E_{\Sigma_2}^\pm$ downstream with **zero feedback** into $\Sigma_{\text{auth}}$.

---

### Target 3 — Flagship #1 Forensics, Exact `A2g` Equations, & Up-Projection Scale Law

1. **Locked `f1_4096_ood` Forensic Corrections:**
   - Zero Legal-trained bridges exist (`fit_domain ∈ {CoSQA_Code, SciFact}` only; Legal is evaluation-only).
   - Full 18-cell zero-shot OOD `Centered Affine Ridge` range is **`0.0000–0.1704`** (`0.0000–0.1362` across the 12 `384↔4096` cells; `A2c` wins `16/18`, `A3_full` wins `17/18`, `SciFact→Legal_QuAD D384→D4096` is the single cell where Ridge `0.0295` edges `A3` `0.0270`).
   - At `SciFact` Hop 2 (`n=2000, 2560→1024`), $n-1 = 1999 > 1024$ yields **full thin rank `1024`** — whitepaper text MUST distinguish (a) the 561-D unobserved source orthogonal complement & ambient-vs-OOS projected norm bias, (b) noisy sample-tail singular modes `#400..#1024` given unit gain by Procrustes vs damped by Ridge, and (c) exact-zero singular completion ($n-1 < \min(d_{\text{in}}, d_{\text{out}})$).
   - `scratch/run_pristine_f1_4096_ood.py` is absent from checkout `18973789`; `indomain_dim_jumps` evaluated a post-DEV split (`SciFact 0.6866`, `CoSQA 0.2827`) while `zero_shot_ood` evaluated full query sets (`SciFact 0.6484`, `CoSQA 0.2831`), and `0.6484 ≠ 0.6451` ([s1_per_query_paired.jsonl](file:///D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/artifacts/flagship1-night1/20261121-flagship1-r1/s1/s1_per_query_paired.jsonl)) confirms `D384` encoder/loader provenance must be explicitly recorded in the committed `scripts/run_flagship1_4096_ood.py` rerun.

2. **Locked Exact `GeodesicPolarBridge` (`A2g`) Specification:**
   - **Poles & Dual Stability Guard (on `FIT`):** $\mu_x = \frac{1}{n_{\text{fit}}}\sum x_i$, $p_x = \mu_x / \|\mu_x\|_2$ (analogous $p_y$). Require **both** $\min(\|\mu_x\|_2, \|\mu_y\|_2) \ge \tau_{\text{pole}} = \mathbf{0.20}$ **and** angular standard error $\max_{z \in \{x,y\}} \sqrt{\frac{1 - \|\mu_z\|_2^2}{n_{\text{fit}}\|\mu_z\|_2^2}} \le \mathbf{0.15\text{ rad}}$; otherwise reject `A2g` fit and route cleanly to `A2c-Subspace+OOS` with receipt reason `POLE_UNSTABLE_FALLBACK`.
   - **2-Step Gram–Schmidt + `atan2` Log Map & $O(d)$ Householder Frames:**
     $$c = x^\top p_x,\quad h_1 = x - c p_x,\quad h = h_1 - (h_1^\top p_x) p_x,\quad \theta = \operatorname{atan2}(\|h\|_2, c),\quad \operatorname{Log}_{p_x}(x) = \operatorname{sinc\_inv}_{\text{collar}}(\theta)\, h$$
     Project into orthonormal Householder tangent bases $B_x \in \mathbb{R}^{d_x \times (d_x-1)}, B_y \in \mathbb{R}^{d_y \times (d_y-1)}$ ($B^\top p = 0, B^\top B = I_{d-1}$); fit truncated SVD $U_x^\top U_y = \tilde{U}_r \Sigma_r \tilde{V}_r^\top$, $Q_{\text{tan}} = \tilde{U}_r \tilde{V}_r^\top$.
   - **Disjoint `CAL` Scale & $C^1$ Exact Soft-Collar Exp Map ($r_0 = \pi - 0.10,\; \delta = 0.05$):**
     $$s_\theta = \frac{\|\operatorname{Log}_{p_y}(Y_{\text{cal}}) B_y\|_F}{\max(\|\operatorname{Log}_{p_x}(X_{\text{cal}}) B_x Q_{\text{tan}}\|_F, 10^{-8})}, \qquad r_{\text{eff}}(r) = \begin{cases} r & r \le r_0 \\ r_0 + \delta\tanh\!\big(\frac{r - r_0}{\delta}\big) & r > r_0 \end{cases}$$
     $$w = \big(s_\theta \operatorname{Log}_{p_x}(q) B_x Q_{\text{tan}}\big) B_y^\top, \qquad \operatorname{Exp}_{p_y}^{\text{collar}}(w) = \cos(r_{\text{eff}}(\|w\|_2))\,p_y + \frac{\sin(r_{\text{eff}}(\|w\|_2))}{\max(\|w\|_2, 10^{-30})}\,w.$$
   - **Geometry & Multi-Hop Contract:** Never call `A2g` globally conformal (radial stretch $s_\theta$ vs transverse $\sin(s_\theta\theta)/\sin\theta$). Collapse tangent maps analytically ($Q_{\text{tan},1}\cdots Q_{\text{tan},m}$) **only** when consecutive poles & frames match (`p_{y,k} == p_{x,k+1}`) and all `CAL`/query norms satisfy $\|w_k\|_2 \le r_0$; otherwise execute sequential geodesic relay.
   - **Pre-Registered Promotion Rule:** Promote `A2g` as optional narrow-cone bridge **iff** paired held-out `REPORT` `LCB95(A2g - A2c_OOS) > 0` on pre-registered narrow-to-wide swaps (`E2→E0`, `Legal_QuAD`) with zero OOD degradation beyond `-0.005`; else log in the Scientific Falsification Ledger.

3. **Locked Up-Projection Scale Rule (`A2c-OOS-Radius` vs `A2` vs `s_subspace`):**
   - Retire Neon's `s_subspace = ||Y_cal,c P_Q||_F / ||X_cal,c Q||_F` (falsified in `Probe 2`: inflates $1/s$ hub bias and degrades both retrieval and cone alignment).
   - Standardize `A2c-OOS-Radius` on disjoint `CAL`: **`s_oos = ||Y_cal,c||_F / ||X_cal,c Q_r||_F`** (required for multi-hop cone fidelity `0.9999` and `B4` backfill compatibility), and select between `s_oos` and unscaled `A2` (`s = 1.0`, which prevents $1/s > 1$ hub-bias amplification on terminal 1-hop query lifts $d_{\text{in}} < d_{\text{out}}$) strictly on `DEV/SELECT` before `REPORT` lock.

---

### Target 4 — Final Locked `PM-1 .. PM-5` Pre-Release Gate Table (Zero Residual Drift)

| Gate | Final Unanimous Council Specification & Pass Thresholds |
|---|---|
| **`PM-1a` (Structured `EK5 v3` $K=4$)** | **Differential Fuzzer:** `0` WFS or shield mismatches vs independent alternating-reduct oracle across exhaustive 2-node (`2,304` transitions), exhaustive 3-node single edits (`4,718,592` cases), `28,561` bitemporal cuts, and `100,000` random 3–8-node cyclic single+batch mutations. **Live `Zero-Shortcut v2`** (`N = 240`, 8 counterbalanced families × 30, persistent `EK5 v3` $K=4$, identical observable fields & `top_k=5` token budget across all arms, zero cross-clock bugs, zero unlinked `val_sup_cnt` boost): **`C4_v3 >= 98.0%` acc**, **`0.0%` zombie errors**, **`0.0%` unauthorized DoS overrides**, and paired **`LCB95 > +15.0 pp`** over each weak comparator (`C2_latest`, `C2b_bm25_jaccard`, `C3_kk`, and Astra's 4-line non-propagation rule). |
| **`PM-1b` (Blind Conversational)** | **Corpora:** Full `LongMemEval-S` (`N = 500`, `_abs` & `single-session-user` reported both within-500 and as separate cohorts with exact $N_{\text{ssu}}$) + 10-conversation balanced `LoCoMo` (`N = 300`, `30/conv`, compound evidence IDs split on `[;,\s]+`, both any-hit & all-required-hit recall, structured abstention parser, conversation-cluster bootstrap CIs, previously inspected 120 items tagged as dev-audited). **Deterministic Inference:** Fixed-batch (`batch_size=1`) BF16/FP16 or 3-pass controlled-order replication + prompt-SHA cache. **Regime A (Clean) Gate:** Against frozen `M1_hybrid_flat`, require **`point >= -1.0 pp` AND Tango/score matched-pair `LCB95 >= -3.0 pp`** on full unmodified `LongMemEval-S` ($N=500$, `84.1%` power at $q=0.05$), plus **`point >= -1.0 pp`** on both `single-session-user` and unmodified `LoCoMo` ($N=300$). *(Explicitly phrased as a 1-pp observed-loss cap with a 3-pp 95% uncertainty bound; 1-pp statistical non-inferiority is NOT claimed without $N \ge 3,925$.)* **Regime B (Question- & Gold-Blind Neutral Echoes, zero `"from earlier chat"` cues):** Preserve original **`>= +25.0 pp` (`LCB95 > +15.0 pp`)** over `M1_hybrid_flat` on `LongMemEval-S` KU (`LCB95 > +10.0 pp` on `LoCoMo`), and require **`LCB95 > 0.0 pp` vs `M1c_hybrid_jaccard_dedup`** to claim open-text benefit beyond lexical deduplication. |
| **`PM-2` (Coupled Dual-Drift Engine)** | Single persistent process crossing `384 → 1024 → 2560 → 4096` (testing down-projection, up-projection `384→4096`, and interior-peak `384→1024→384` `A2c-Polar`) + `TRA-1` + `EK5 v3` ($K=4$) + `M4c` on identical histories: **`0/12` unauthorized hostile `TRA-1` executions**; bridged-minus-native accuracy **`point >= -3.0 pp` AND `LCB95 >= -5.0 pp`** *(native ratio reported as diagnostic only)*; **`>= +20.0 pp` (`LCB95 > +12.0 pp`)** over bridged flat RAG under echo stress; and **`LCB95 > 0.0 pp`** over Composed Centered Affine Ridge on multi-hop retrieval (publishing both $2^r-1$ and linear nDCG@10). |
| **`PM-3` (Pinned Upstream Baselines)** | Pinned commits of unmodified **`mem0ai/mem0`** and **`getzep/graphiti`** under matched embedder, generator, write stream, and `top_k=5` token budget, plus runtime availability check for official **`ScaleMAS`** (if absent, emit dated `UNAVAILABLE_UPSTREAM_ARCHIVE` receipt, run `C1_bounded` as mechanism comparator, and record direct ScaleMAS parity as unavailable) + 1-factor-at-a-time `C4_v3` ablation. **Original mandatory gate:** direct parity comparison demonstrating positive stale-error reduction (`LCB95 > 0.0 pp`) at matched-or-lower tokens. **Newly proposed R1 stretch sub-gate (labeled separately):** `LCB95 > +10.0 pp` stale-error reduction or `>= 25%` token savings on multi-session conflict slices. |
| **`PM-4` (Inference & Blind Adjudication)** | $\ge 2$ generator families × $\ge 2$ independent blinded judge families (preregistered primary judge pair **`κ >= 0.85`** with 95% CI & confusion matrix) + **`100`-item stratified human audit** (25 agreements, 25 disagreements, 25 abstentions/`_abs`, 25 relative-date/negations). Two-phase offline deduplicated `(qid, norm_pred)` grading (`0` key conflicts); `paired_bootstrap_delta` fails closed (`ValueError`) on any ID mismatch and reports **both** 2.5th-percentile bootstrap and **Tango (1998) matched-pair score CI** (eliminating `0.00 pp` zero-discordance collapse). Consistent positive primary effect sign across all cells (`LCB95 > 0` required per cell where superiority is claimed). |
| **`PM-5` (Service SLO vs. 1M Kernel & Receipts)** | **(a) Persistent `100k` Service SLO:** Externally timed `accept → durable WAL/EK5 v3/index visibility → TRA-1 → retrieval → receipt` **`p95 <= 25.0 ms`** (reporting p50/p95/p99, QPS, RSS, crash/restart parity). **(b) Standalone `1M`-Node `EK5 v3` Kernel:** Sparse trees, power-law cyclic graphs ($\gamma \approx 2.3$, 10% `REFUTE`, 2% cycles), `10k` in/out hubs, rank-only cascades, and dense cascades (`max_dirty_edges` fallback ON vs OFF): **`0` spurious full-graph scans on certified sparse/no-op/shielded-hub workloads**, independent WFS & rank parity, and published empirical crossover sweep (reporting tested range if curves never cross; genuine global cascades count as legitimate full scans). **(c) Receipts:** `100%` SHA-256 coverage of all emitted CSVs, JSONLs, summaries, and manifests in `report_receipt.json` + mandatory `git_sha` in every manifest. |

---

### Target 5 — Unified Chapter Architecture, Single-Source Builder Schema, & HTML v2 Matrix

1. **Unified 8-Chapter + Appendix A Monograph Architecture (Reconciling Astra's Clean Tier Separation with Neon/Carbon Structure):**
   - **Frontmatter & Scoped Claim-Boundary Table:** Pre-release gate status banner + 3 strictly separated claims (Migration Geometry, Tier-1 Structured `EK5 v3` $K=4$, Tier-2 Open-Text `M4c` vs `M1c_hybrid_jaccard_dedup` with historical clean `42/50` vs `46/50` side-by-side).
   - **Ch. 1 — Dual-Drift Problem Statement & Release Status.**
   - **Ch. 2 — Metric Lineage (`mlr-v1`), `TRA-1` ABI, & Statistical Governance** (correcting historical linear graded gain $r_i$ vs $2^{r_i}-1$ binary equivalence, two-sided 95% CI 2.5th percentile + Tango score CI, `TRA1_OR_M4_NEEDS_REVIEW` + `8/8` blocks).
   - **Ch. 3 — Flagship #1 Geometry & Estimators** (3-Regime Stiefel Bottleneck Theorem, Frobenius-Cosine Prop. 1b, Disjoint OOS Projected Radius, `A2c-Polar`, Guarded `A2g` + Non-Conformality & Aligned-Pole Theorem, $O(n^2 d)$ `A3-Subspace`, Ridge Centroid-Collapse Proof).
   - **Ch. 4 — Flagship #1 Empirical Migration Results** (6-BEIR single/4-hop, `384↔4096` both directions labeled by `query → index` space, `SciFact` Hop 2 analysis, 18-cell `{Code, SciFact} → {Legal, Code, SciFact}` OOD matrix, 11-arm cached ablation, separate `B4` budget).
   - **Ch. 5 — Flagship #2 Evidence Semantics & Incremental Kernel (`EK5 v3`)** (Direct-$B_i$ vs Declared Lineage truth table, `COPY_OF` vs `AMENDS` vs `CITES_AUDIT`, $K=4/8$ Stratified Stabilization Theorem + Scoped $\tilde{\alpha}_q(v,c)$, Stale-Rank Counterexamples, Positive-SCC + WFS-Rank Theorem, `Zero-Shortcut v2` & `1M` kernel / `100k` service results).
   - **Ch. 6 — Flagship #2 Conversational Echo-Collapse Evidence** (Separated cleanly from Ch. 5! Full `LongMemEval-S` $N=500$, `single-session-user` cohort, 10-conv `LoCoMo` $N=300$, `42/50` vs `46/50` historical disclosure, `M1c_hybrid_jaccard_dedup` attribution, `_abs` / abstention accuracy).
   - **Ch. 7 — Failure Forensics, Archive Provenance & Related Work** (`pristine-b37e6a55` BOM/LF damage vs `pristine-b37e6a55-exact`, `EK5 v1/v2` failures, `LoCoMo` R0 sampling/grader-race/cue forensics, `iq01` & pruned hypotheses, positioned vs Mem0/Graphiti/ScaleMAS/ASPIC+).
   - **Ch. 8 — Integrated Proof Engine v1 & `PM-1..PM-5` Gate Matrix.**
   - **Appendix A — Three-Part Cryptographic Evidence Ledger:** Part I (`12` historical bundles, `s1 = 1,226.40s`, full 64-char `f2_iq04` `148d80bf3faadd77f9ad0f9eeb1f1a089d8e87b89025c9f645d4a685ad7e619b`, `f2_longmemeval` marked `SUPERSEDED_BY: f2_longmemeval_v2`, `iq01` falsifier, `2/2` `s1` ancillary JSON SHAs, `19` historical CSVs marked `PUBLICATION_PINNED_NOT_RECEIPT_BOUND`), Part II (`3` R3 bundles + `6/6` R3 artifact SHAs + `f1_4096_ood` `ce81b930...` metadata-gap disclosure), Part III (Post-R3 Proof-Model runs).

2. **Single-Source Registry Schema (`scripts/build_whitepaper_assets.py`):**
   - Pipeline: `discover_attempts -> verify_bytes -> normalize_observations -> validate_claims -> render` (`--check` exits non-zero on any byte drift, uncatalogued `report_receipt.json`, missing declared artifact, duplicate observation key, or rounded-delta mismatch).
   - Canonical observation record in `docs/research/verified_registry.json`:
     `{attempt_id, bundle_path, manifest_sha256, git_sha, verification_tier: "RECEIPT_BOUND" | "ANCILLARY_RECEIPT_BOUND" | "PUBLICATION_PINNED_UNHASHED_HISTORICAL" | "MANIFEST_AGGREGATE_ONLY", role: "ACTIVE" | "SUPERSEDED" | "FALSIFIER" | "AUDIT_BASELINE", method, dataset, fit_domain, eval_domain, query_dim, index_dim, hop, split, seed, metric, gain_convention, denominator, point_float64, ci_method, lcb95, ucb95, delta_is_from_rounded_manifest}`.

3. **HTML v2 Contrast & 6-Figure Static-SVG + Semantic-Table Specification:**
   - **Locked WCAG AAA `--warm` Hex:** **`#7d2f0a`** (`8.36:1` on `#fdf2ec`, `8.80:1` on `#fafaf8`, `9.20:1` on `#ffffff`; zero JS required to view any table, KPI, or SVG; native `<math>` MathML; `.release-status-banner` inside `<header class="hero">` visible in `@media print`; keyboard ARIA `tablist`; `375px` `.table-scroll` wrappers).
   - **Locked 6 Static SVGs (Each Paired With an Accessible `<table>` Naming Units, Denominator, Comparator & CI Source):**
     1. *Fig. 1:* `2 × 3` **6-Domain 4-Hop Small Multiples** (`SciFact`, `NFCorpus`, `FiQA2018`, `ArguAna`, `SCIDOCS`, `Quora`).
     2. *Fig. 2:* **Extreme Dimension (`384↔4096`, labeled by `query → index` direction) & 3-Hop Chain Panel** (`interval unavailable` badge on historical `ce81b930...` until per-query CSV rerun).
     3. *Fig. 3:* **18-Cell Zero-Shot OOD Matrix** (`fit_domain × eval_domain × swap`, showing Ridge centroid collapse alongside native C2O ceilings).
     4. *Fig. 4:* **4-Panel Aligned `0–100%` Accuracy + 95% CI Whiskers** (Structured `Zero-Shortcut v2` | Clean Regime A incl. `42/50 vs 46/50` | Blind Echo Stress Regime B incl. `M1c_hybrid_jaccard` | `_abs` & `CONTESTED` Abstention).
     5. *Fig. 5:* **Systems Latency Split** (`100k` End-to-End Service SLO `p50/p95/p99` vs `1M`-Node `EK5 v3` Kernel Topology & Crossover Sweep).
     6. *Fig. 6:* **Dual Explanatory Architecture SVGs** ((a) Extrinsic Gnomonic `A2c` vs Geodesic Polar `A2g` $C^1$-Collar on $\mathbb{S}^{d-1}$; (b) Stratified $K=4/8$ DAG $\Sigma_{\text{vis}} \prec \Sigma_{\text{auth}} \prec \Sigma_{\text{ep}} \prec \Sigma_{\text{cur}} \prec \Sigma_{\text{rank}}$).

---

## 3. Remaining Micro-Items Flagged for R3 Final Sign-Off

Zero fundamental mathematical or statistical disagreements remain after our R2 cross-check. Only **three tiny parameter conventions** should be explicitly rubber-stamped in R3 so all three models emit byte-identical specs:
1. **`A2g` Pole Guard Convention:** Confirm dual guard **`min(||mu_x||, ||mu_y||) >= 0.20` AND `angular_SE <= 0.15 rad`** + **$C^1$ exact soft-collar ($r_0 = \pi - 0.10, \delta = 0.05$)** (reconciling Neon's `0.15` + global `tanh` [`3.47°` bias] with Carbon's `0.25` + hard clamp and Astra's `< 1e-12°` exact interior requirement).
2. **Quarantined `SUPERSEDE`-Cycle Outgoing Force:** Confirm Astra's safer rule (`B_Y <- U` for outgoing `B -> Y` when `B` is in a nontrivial active `SUPERSEDE` cycle, rather than Neon's `B_Y <- T`) alongside non-forwarding cycle quarantine (`X -> A` on `A <-> B` leaves `C_B = U`).
3. **`--warm` Token Choice (`#7d2f0a` `8.36:1` vs `#8a3208` `7.51:1`):** Both pass WCAG AAA (`>= 7.0:1`) on `#fdf2ec`; Carbon recommends locking **`#7d2f0a`** for the extra `+0.85` contrast headroom in print/low-gamma displays.
