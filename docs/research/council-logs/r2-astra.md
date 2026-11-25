<!--
{
  "taskId": "node:delegated-task:command%3Amcp%3A3b84a70c-9824-441c-8750-b9902399a932%3Adelegate-task%3Acouncil-r2-astra-xhigh-20261121",
  "link": "[Council Refinement R2 Astra 6.2 XHigh : Cross-Check Peer R1 Proofs &...](t3-thread://v2/c5bb8594-9c79-4948-9251-bd83e8527cf6/thread:delegated-task:command%3Amcp%3A3b84a70c-9824-441c-8750-b9902399a932%3Adelegate-task%3Acouncil-r2-astra-xhigh-20261121)",
  "model": "gpt-6.2-astra",
  "executedModel": "gpt-6.2-astra",
  "options": [
    {
      "id": "reasoningEffort",
      "value": "xhigh"
    }
  ],
  "status": "completed"
}
-->

# Astra 6.2 XHigh — Refinement Round 2 review

**Advance the corrected specifications to R3. Keep public release blocked.** The recovered producer overturns one premise shared by all three R1 reports. The SciFact chain used **1,000 anchors per hop**. The **2,000-anchor** setting belongs to the direct jumps. At chain Hop 2, centering therefore leaves rank at most **999** in a 2,560→1,024 cross-product. At least **25 thin singular directions are unidentifiable**. Their contribution to the observed retrieval loss remains unmeasured. [Chain producer, line 367](C:/Users/mattm/.t3/userdata/providers/antigravity/ac0a3dfd6dddb20962cecff6ee5fe65e19d3923be20e52c5ab52ff877f7e4c32/antigravity-acp/brain/d68f68df-2385-4788-8983-e239ca3c27e0/scratch/run_pristine_f1_4096_ood.py:367) · [Direct-jump producer, line 292](C:/Users/mattm/.t3/userdata/providers/antigravity/ac0a3dfd6dddb20962cecff6ee5fe65e19d3923be20e52c5ab52ff877f7e4c32/antigravity-acp/brain/d68f68df-2385-4788-8983-e239ca3c27e0/scratch/run_pristine_f1_4096_ood.py:292)

I verified checkout `18973789615ecab1268a77ebd3c69a283607241c` and reviewed all three R1 reports. The algorithm comparisons below use in-memory reconstructions of the published rules. The peer store classes were unavailable as runnable source. The differential results apply to these reconstructions and the proposed repair. No repository files or Git state were changed.

## 1. Peer R1 cross-check audit

| Peer | R1 claim or algorithm | Verdict | Executable evidence or edge case |
|---|---|---|---|
| Carbon | Pitfall A subtracts dirty shields and Pitfall B updates external counts | **Passes the three-node stale-rank witness if every dirty predecessor is charged. Fails as a general earliest-rank cut.** | On `B→A, C→A`, adding `A→B` raises `B` from 0 to 2. Carbon correctly leaves `A` outside with the surviving `C` shield and updates its count to 1. Removing `C→A` then produces `A=B=U`. In a separate five-node probe, a newly grounded rank-0 attacker shortened an external node’s rank from 3 to 1. Carbon’s surviving older shield incorrectly froze rank 3. |
| Carbon | Every positive SCC intersecting a pruned cone is contained in it | **False as stated.** | Start with fact `x`, `w ← a, not x`, and fact `a`. Change `a` to `a ← w`. The new positive SCC is `{a,w}`. The unchanged root `x` permits pruning `w`, leaving `D={a}`. WFS is still correct because frozen `w=F` grounds `a=F`. The proof needs a grounded-boundary argument. |
| Neon | Boundary-aware ranks repair the five-node backward-skip failure | **Confirmed on the published sequence. The skip table preserves less than exact earliest rank.** | On `x→w→k→d`, adding `d→h` gives `h=T(4)`. Adding `h→w` then deleting `x→w` makes `w,k,d,h` all unknown. The stale implementation gives the wrong answer. Separately, adding an independent root attacker to `d=F(3)` makes its earliest rank 1, and deleting the only incoming attack on `h=T(4)` makes its rank 0. Both peers’ blanket destination-label skips missed these changes. |
| Neon and Carbon | Imported mutual `SUPERSEDE` and external dominance | **Neon’s per-target odd loop passes. Carbon’s whole-cycle override fails for ordinary edges.** | An independent alternating-reduct solver gave `C_A=F, C_B=U, C_X=T` for `A↔B, X→A`. With `C COPY_OF A`, the bare cycle gives `C_C=U`. Adding `X→A` gives `C_C=F`. Explicitly covering both `A` and `B` makes both obsolete. |
| Carbon and Neon | A foundation ceiling prevents authority laundering, while authority qualification precedes WFS | **Accept with typed endorsement and scoped grants.** | A tier-4 auditor making an independent finding about a tier-0 rumor retained tier 4. The same auditor endorsing the rumor had ceiling 0. A blanket minimum over every historical reference reduced both to 0. A fresh-versus-incremental minimum calculation matched on 10,000 edge and tier edits, including increases after deletion. |
| Carbon and Neon | A2c equator barrier, Ridge centroid collapse, and A3 shrinkage explain the receipt | **Conditional geometry confirmed. Two empirical explanations need correction.** | An ambient A2c example with source pole cosine `.89` mapped to **91.66°**. Ridge mapped two distinct nullspace queries to the identical centroid in a controlled example. The producer explicitly dispatches down-projections to full A3, and its up-projection changes both the retained target subspace and the shrinkage scale. |
| Astra self-audit | Hop 2 permits full thin rank, and clean PM-1 needs a separate denominator | **Correct the first claim for this run. Confirm the second.** | The chain source fixes 1,000 anchors. Separately, I counted the pinned LongMemEval-S file itself. Its 500 unique items include **70** `single-session-user`, **78** `knowledge-update`, and **30** `_abs` items across categories. The historical 50-item clean slice is a third denominator. |

[Carbon R1](D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-carbon.md:58) · [Neon R1](D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-neon.md:58) · [Astra R1](D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-astra.md:52) · [Pinned LongMemEval-S source](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/blob/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_s_cleaned.json)

## 2. Resolved specifications for Targets 1–5

### Target 1 — EK5 v3

**Store WFS truth and the earliest raw Fitting certificate separately.** For one conjunctive rule per atom, an empty body proves true at rank 0. A true body proves its head at one plus the maximum prerequisite rank. A false body proves its head at one plus the minimum falsifier rank. Raw unresolved has rank infinity. A positive-cycle WFS decision supplies a semantic value and an unfounded-set reason. It supplies no raw rank-0 witness.

The mandate’s `1→0` example has a precise answer. In `x→v, u→w`, with all arrows `REFUTE`, add `v→u`. Then `v=F(1)`, `u` goes from `T(0)` to `T(2)`, and `w` goes from `F(1)` to `F(3)`. At **old** `rank[w]=1`, the old strict count drops from 1 to 0. That loss invalidates the boundary and requires `w` in the rebuild. At **new** `rank[w]=3`, `u` is again an earlier witness and the exact count is 1. Carbon catches it if Pitfall A charges `u` on entry to `D`. Neon catches it with a forward-closed cone. A post-solve decrement that leaves `w` outside is insufficient.

**Freeze this conservative algorithm.** It keeps the constant-time cuts for certified redundant hub edits and provides an exact fallback.

```text
apply(old_snapshot, atomic_mutation, valid_cut, tx_cut):
    project visible records, attestations, grants and provenance at both cuts
    recompute affected scoped authority and effective relation multiplicities
    collect inserted/removed atoms and all heads whose effective body changed

    if exactly one effective REF literal changed at a resolved raw head:
        evaluate its edited body against the old raw certificate streams
        if the head has exactly the same raw value and earliest rank:
            update adjacency, multiplicity and witness aggregates for the edit
            retain epistemic values and continue to affected downstream strata

    otherwise:
        seed D with all changed heads and inserted/removed atoms
        traverse the union of old/new prerequisite-to-dependent edges
        allow one internal pruning rule:
            a non-seed old raw F(1) may be frozen if a post-edit
            unconditional T(0) root still directly refutes it
        if that root certificate cannot be established, include the head in D

        freeze incoming boundary WFS values and original raw certificate ranks
        on mandatory-positive edges in D, find SCCs including self-loops
        seed cyclic SCCs as WFS false and propagate three-valued consequences
        independently rebuild raw certificates in increasing arrival-rank order
        leave unresolved raw atoms at infinity

        update counters on edited edges and every edge incident to changed
        values or ranks, using both old and new endpoint ranks
        rebuild head-relative witness membership whenever the head rank changes
        publish semantic, rank, membership, insertion and removal deltas atomically

    recompute affected currency, slot decisions and index entries
    commit one snapshot version and receipt
```

The single-head cut preserves the entire Fitting stream of a resolved atom. Its value at every iteration follows from its sign and first finite rank. The remaining rules therefore see the same stream, and the grounded head can be fixed when evaluating residual WFS. The internal `F(1)` barrier has a separate proof: a surviving rank-0 root already achieves the earliest possible raw falsification. A generic surviving external witness has no such guarantee. Indexed root and min/max witness counts support common constant-time checks. Charge boundary ordering, all incident-edge updates, interval lookup and emitted index changes. A heap-based rank pass must include its ordering cost.

**Fresh differential result.** The proposed repair had zero discrepancies on all **65,536** two-node ordered snapshot transitions, **30,000** five-node edits plus **858** extra jumps, and **12,500** additional mixed transitions checked against a separate synchronous earliest-rank oracle. WFS labels were checked against alternating least reducts. These probes cover rule semantics and metadata. Persistence, concurrent readers and service latency still require implementation evidence. For the paper’s reduct oracle, if \(K=\operatorname{lfp}(\Gamma^2)\), the false set is \(At\setminus\Gamma(K)\). Carbon’s printed \(At\setminus\Gamma(At\setminus K)\) gives the wrong false atom on `a ← not b, b ← not c, c ← true`. [Current store](D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/iq04_temporal_evidence.py:844) · [Carbon’s theorem](D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/council-logs/r1-carbon.md:98)

**Logical Stratum 2 freezes validity \(V\) before computing currency.**

1. Ordinary authorized `SUPERSEDE` has direct force. Durable ancestry requires explicit matching `(tenant, entity, single-valued slot, version_lineage)` plus replacement authorization at each edge. Valid heads may cover strict ancestors through a factually refuted intermediate version. Pure `REFUTE` creates no replacement path.
2. Reject self-edges and cycle-closing native supersession writes. For imports, compute SCCs on visible authorized `SUPERSEDE` edges whose source has \(V\ne F\). For an internal edge \(j\to i\), use \(O_i\leftarrow V_j,\operatorname{not}O_i\). An active cyclic edge gives a possible blocker. For external ordinary edges, and certified strict ancestry, use \(E_i\leftarrow F_j\), where \(F_j=V_j\) for a normal source and is capped at \(U\) for a quarantined cyclic source. Preserve witness IDs. Exclude internal quarantined edges from ancestry traversal, including structurally cyclic imported lineage links. External coverage applies to each certified target separately.
3. Evaluate \(C_i\leftarrow V_i,\operatorname{not}E_i,\operatorname{not}O_i,\bigwedge_{p\in D_C(i)} C_p\). `COPY_OF` and `DERIVES_CURRENT` inherit validity and currency. A content-dependent `AMENDS` or `DERIVES_SNAPSHOT` inherits validity. Legacy `DERIVES_FROM` defaults to current inheritance until an explicit migration establishes snapshot semantics.
4. `C REF+SUP B, B SUP A` yields `A,C` current on ordinary edges and only `C` current on a certified lineage. A single-valued slot with incompatible current values or an unresolved competing value abstains with witnesses. An unrelated contested slot leaves the requested slot available.

The outgoing quarantine cap is a deliberate policy choice. On `A↔B, A SUP D`, Neon’s literal external rule makes `D` definitely obsolete. The capped rule gives `D` unresolved currency. The odd-loop encoding itself does not decide that policy.

### Target 2 — Effective authority and bitemporal updates

Freeze the production order as **visibility → authority-qualified relations → epistemic WFS → currency**. Retain the last two as the \(K=2\) logical core. For \(K=8\), integrity precedes use of the evidence and a continuous score orders admitted results only. Specificity remains a theoretical extension.

For claim scope \(k\), let \(b_k(v)\) be the visible credentialed evidentiary tier, \(P_k(v)\) its declared truth-endorsing foundations, and \(\bot\) below tier 0. Define

\[
a_k(v)=\min\!\left(b_k(v),\min_{p\in P_k(v)}a_k(p)\right).
\]

Select the **greatest** solution under the tier order, equivalently the minimum raw tier over the reachable endorsement ancestry, including the node itself. A known mandatory foundation missing at the cut contributes \(\bot\) and unresolved validity. This convention separates the authority calculation from WFS’s treatment of an ungrounded positive cycle. An authorized `REFUTE` or `SUPERSEDE` additionally requires a visible grant for the actor, action and target scope, and a tier meeting the target’s protection floor and applicable tier comparison. A preference grant confers no power over a sensor receipt.

| Assertion by a tier-4 auditor | Role of tier-0 rumor | Effective result |
|---|---|---|
| Copies the rumor, summarizes it as true, or derives an amendment whose value relies on it | Mandatory endorsement | Ceiling 0. It cannot defeat protected tier-2 evidence. |
| Independently finds a discrepancy and cites the rumor as the object examined | Evidentiary reference | The finding retains tier 4 within its grant. Verify the cited record’s identity and visibility. |
| Writes a snapshot assertion that relies on the historical value being true | Mandatory validity foundation and endorsement | Ceiling 0, even though later replacement of the base does not invalidate the historical derivation. |
| Independently establishes a replacement and cites the earlier version to identify what changed | Version reference plus independent evidence | The replacement receives the tier justified by its own foundations. |

The edge name `AMENDS` alone cannot distinguish the last two cases. Require an authenticated basis declaration. Relabeling a copied value as an audit must not erase mandatory provenance. At a historical cut, an independently visible finding can retain its tier while an edge to an invisible target remains inactive.

On a mutation or rewind, take the old/new reverse endorsement cone, reset affected nodes to their visible raw tiers, and recompute minima on the SCC condensation DAG with a frozen external boundary. This permits increases after a genuinely removed foundation or low-tier grant. A disappearing mandatory parent keeps its missing-parent sentinel until the claim version itself changes. Diff authorization on **incoming and outgoing** defeat and supersession edges, then seed the affected WFS heads. The snapshot, authority and WFS changes publish together. Stratification proves deterministic composition once each stratum has a specified canonical result. It does not grant constant-time authority cascades or rewinds.

### Target 3 — Receipt forensics and bridge equations

**The `f1_4096_ood` receipt verifies the following.**

| Question | Finding |
|---|---|
| Fit domains and range | The only fit domains are `CoSQA_Code` and `SciFact`. Centered affine Ridge spans **.0000–.1704** across all 18 rounded OOD rows and **.0000–.1362** across the 12 joining D384 and D4096. A2c beats Ridge in 16/18 and A3-full in 17/18. |
| SciFact native discrepancy | Direct jumps score `REPORT`, giving **.6866/.7266** for D384/D4096. OOD scores all **300** SciFact query IDs, giving **.6484/.7645**. Applying the producer’s splitter to the 300 IDs recovered from S1 gives **124 DEVELOPMENT, 90 SELECT, 86 REPORT**. |
| Hop 2 | The recovered chain producer samples **1,000** distinct document anchors per hop with hop-specific seeds. Hence \(\mathrm{rank}(X_c^\top Y_c)\le999\) for 2,560→1,024. The score **.4768 vs .5023** is measured on the composed route to D384. Arbitrary completion, noisy identified modes, radius estimation and composition are distinct possible contributors. |
| Down-projection A3 equality | The producer literally calls full A3 when `d_out <= d_in`. Equality is built into the experiment. As a control, independently changing source whitening from `tr/d` to active-subspace `tr/r` on a down-projection changed the fitted matrix by **7.79%** in relative Frobenius norm. |
| Up-projection A3 | On 384→4,096 the producer retains **384 target principal directions**. The shrinkage target changes from \(\mathrm{tr}(S_y)/4096\) to \(\sum_{i=1}^{384}\lambda_i/384\). Both the trace numerator and the modeled subspace change. The subspace arm wins both direct lifts and 5/6 OOD lifts. Isolating the divisor requires crossing truncation and shrinkage in separate arms. |
| Centroid collapse | Write the pre-normalized Ridge output as \(b+\delta W\), with \(b=\mu_{y,\mathrm{fit}}+(\mu_{x,\mathrm{OOD}}-\mu_{x,\mathrm{fit}})W\). Concentration near \(\hat b\) follows when \(\|\delta W\|\ll\|b\|\). The synthetic nullspace example collapsed exactly. The manifest contains retrieval scores and lacks the query dispersion, centroid alignment and spectra needed to establish this mechanism on its 18 cells. |

The recovered producer SHA-256 is `bf73f4dedb778214527daa7ef1e87b5fc96aafb955ad6cd489b55c69abf8a88c`. The receipt binds manifest `ce81b93031501542ffcd2dd1291b1e2014eea732e28a44add58b34e47e06f1f0`. It does not bind the producer or a per-query file. Also, its `centered_rad` geometry field is the **mean centered norm**. The bridge fit uses a **Frobenius/RMS** radius ratio. [Manifest](D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/artifacts/flagship1-night1/20261121-rigor-r3/f1_4096_ood/f1_4096_ood_manifest.json:178) · [A3 dispatcher](C:/Users/mattm/.t3/userdata/providers/antigravity/ac0a3dfd6dddb20962cecff6ee5fe65e19d3923be20e52c5ab52ff877f7e4c32/antigravity-acp/brain/d68f68df-2385-4788-8983-e239ca3c27e0/scratch/run_pristine_f1_4096_ood.py:140) · [REPORT evaluation](C:/Users/mattm/.t3/userdata/providers/antigravity/ac0a3dfd6dddb20962cecff6ee5fe65e19d3923be20e52c5ab52ff877f7e4c32/antigravity-acp/brain/d68f68df-2385-4788-8983-e239ca3c27e0/scratch/run_pristine_f1_4096_ood.py:350) · [OOD evaluation](C:/Users/mattm/.t3/userdata/providers/antigravity/ac0a3dfd6dddb20962cecff6ee5fe65e19d3923be20e52c5ab52ff877f7e4c32/antigravity-acp/brain/d68f68df-2385-4788-8983-e239ca3c27e0/scratch/run_pristine_f1_4096_ood.py:439)

**A2g geometry.** Carbon’s \(\pi/2\) barrier holds for \(\operatorname{normalize}(a p+w)\) with \(a>0,\ w\perp p\). The checked-in A2c has axial coefficient \(a+s(x-\mu_x)Qp\), which can be negative. Neon’s aligned-pole example gives **91.93°** at a 45° source angle. The paper should state the barrier for the tangent-displacement model. Radial geodesic stretch under an isometry is \(s\), transverse stretch is \(\sin(s\theta)/\sin\theta\), and projection may discard directions. At \(s=1.8,\theta=.9\), the two stretches are **1.8** and **1.2751**. [A2c transform](D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/spherical_migration_bridge.py:190)

**Freeze `GeodesicPolarBridge` for the prospective ablation as follows.**

- Use unit inputs and normalized FIT extrinsic means \(p_x,p_y\). Require both resultant lengths **≥.15** and both split-half pole angles **≤15°**. Use alternating halves of FIT pairs ordered by a frozen anchor-ID hash. Reject an undefined half-pole. A constructed sample had resultant `.433` and 60° disagreement. Another had resultant `.20` and zero disagreement. Keep `.25` as a preregistered sensitivity arm.
- Form deterministic orthonormal Householder tangent frames \(H_x,H_y\), with tangent coordinates of dimensions \(d_x-1,d_y-1\). Set \(c=\operatorname{clip}(x^\top p,-1,1)\), \(h=x-cp\), \(\theta=\operatorname{atan2}(\|h\|,c)\), and \(L_p(x)=\theta h/\|h\|\), with zero value at the pole. Reject input angles \(\theta\ge\pi-.05\) before dividing. A denominator clamp cannot recover the antipodal direction.
- With \(Z_x=L_{p_x}(X)H_x,\ Z_y=L_{p_y}(Y)H_y\), compute \(Z_x^\top Z_y=U\Sigma V^\top\). Retain \(\sigma_i>10^{-6}\sigma_1\) and set \(Q=U_rV_r^\top\). Reject zero rank or calibration RMS energy below \(10^{-8}\). Fit poles and \(Q\) on FIT and estimate \(s_\theta=\|Z_{y,\mathrm{cal}}\|_F/\|Z_{x,\mathrm{cal}}Q\|_F\) on disjoint paired CAL.
- For a query, \(w=s_\theta L_{p_x}(x)H_x QH_y^\top\). Use a **hard output radius** \(t=\min(\|w\|,\pi-.05)\), \(w'=tw/\|w\|\) with zero limit, and output \(\cos(t)p_y+\operatorname{sinc}(t)w'\), where \(\operatorname{sinc}(t)=\sin(t)/t\). Log clipping and route rejected inputs through the frozen fallback. Neon’s tanh arm changes an in-range `1.8` radians by **.1906**. Keep it as a distinct estimator. Exact composition requires matching intermediate poles, compatible frame changes, and inactive guards throughout.
- Freeze one primary retrieval endpoint before REPORT: equal-domain mean nDCG@10 on E2→E0 across the six named BEIR domains, using the same held-out IDs, gain definition, anchors, seeds and CSLS settings. Promote A2g as an optional route only if the multiplicity-adjusted paired **LCB exceeds +.005** against both `A2c-Subspace+OOS` and sequentially normalized A2c, the prespecified OOD control LCB is **≥−.002**, and the complete routed endpoint includes every refusal and fallback. Record coverage, clipping, p50/p95 and zero index writes. Otherwise retain the result in the ablation ledger. A `.05 ms` relay has no measured status here.

**A2c-Subspace+OOS.** Fit \(Q=U_rV_r^\top\) on the centered FIT cross-product with the same rank rule. Let \(P_Q=Q^\top Q\). On paired CAL, centered at **FIT** means, use

\[
s_{\rm sub}=\frac{\|Y_{\rm cal,c}P_Q\|_F}{\|X_{\rm cal,c}Q\|_F},
\qquad
f(x)=\operatorname{normalize}\bigl(\mu_y+s_{\rm sub}(x-\mu_x)Q\bigr).
\]

Reject degenerate energies. Register full-target numerator, `s=1`, and `max(1,s_sub)` as separate controls. In a unit-vector 3→5 lift with true target cone radius `.4`, projected OOS recovered `.4` and **6/6** top-one matches. The floor forced scale 1, introduced **23.92°** mean error and retrieved **0/6**. The result held with `CSLS k=10` using 5,000 independent reference directions. Use identical frozen reference pools in all CSLS arms. For a claim that this repairs SciFact Hop 2, require a positive adjusted paired REPORT lower bound over **both** historical-form A2c and composed centered Ridge on that endpoint. Radius recovery alone cannot pass it.

### Target 4 — Exact PM gates for R3 ratification

All `pp` values below are absolute percentage points. `LCB95` is the lower endpoint of a two-sided 95% interval. Prospective refinements are distinct from the original gate text. [Original PM-1 through PM-5](D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/docs/research/doctoral-whitepaper-chelated-memory-and-spherical-migration-2026-11-21.md:288)

| Gate | Frozen requirement |
|---|---|
| **PM-1a — structured** | **240** cases, eight counterbalanced families × 30, matched observable fields and `top_k=5` token budgets. Require **≥98%** accuracy, **zero** zombie reinstatements and authority bypasses, and **100%** independent WFS parity. Require **LCB95 >+15 pp** against each of `C2_latest`, `C2b_jaccard`, `C3_kk`, and the four-line observable baseline. Compare the independent exact TMS by parity. |
| **PM-1b — conversational** | Primary clean endpoint is the full unmodified **500-item LongMemEval-S** set versus `M1_hybrid_flat`. Require **point ≥−1 pp and LCB95 ≥−3 pp**. Separately require **point ≥−1 pp** on all **70** `single-session-user` items and report its interval and discordances. At 70 items this allows zero net lost answers. Report LoCoMo **300**, 30 per conversation across ten conversations, with its own conversation-cluster CI. The 800-item pooled estimate is descriptive. On question- and gold-blind KU stress require **point ≥+25 pp and LCB95 >+15 pp** versus M1. Include hybrid+Jaccard and BM25+Jaccard. A beyond-dedup claim requires **LCB95 >0** versus hybrid+Jaccard. Report `_abs`, previously inspected items, unseen items, and any-hit/all-required-hit separately. |
| **PM-2 — coupled drift** | One persistent service, identical histories, native/bridged × flat/evidence-aware, both projection directions and an interior-peak route. Require bridged-minus-native **point ≥−3 pp and LCB95 ≥−5 pp**. Under echo stress require **point ≥+20 pp and LCB95 >+12 pp** versus bridged flat. Require **zero unauthorized executions across at least 12** hostile routes. Require **LCB95 >0** on the preregistered multi-hop retrieval endpoint versus composed centered Ridge. Report direct Ridge and both linear and exponential gain. |
| **PM-3 — upstream comparisons** | Original gate is a direct side-by-side run against pinned, unmodified **ScaleMAS, Mem0 and Graphiti**, with adapter revisions, operation coverage, matched models, writes, tuning and retrieval budgets. If official ScaleMAS is unavailable, record the dated check, run `C1_bounded`, and mark the original ScaleMAS comparison **unfulfilled**. Register **LCB95 >+10 pp absolute stale-error reduction** at matched-or-lower retrieved tokens as a separate stretch result. A positive or ten-point effect was never a condition of the original comparison gate. |
| **PM-4 — inference and grading** | At least **2 generator families × 2 independent blinded judge families**. Freeze a named judge pair and require **Cohen’s κ ≥.85**, with interval, confusion table and human agreement. Human-grade **100 distinct items**, targeting 25 each for abstention, date/negation, other judge disagreement and other agreement, in that assignment priority. If a stratum contains fewer than 25, take all and fill deterministically from remaining items, disclosing achieved counts. Grade each frozen unique input once per judge, require **zero inconsistent keys**, and report all generator–judge cells. Require consistent primary win direction across cells and a positive cell-specific LCB wherever claiming superiority in that cell. Freeze prompt, rubric, model, parameters and replicate IDs. Use fixed scheduling or at least three controlled order replicates. |
| **PM-5 — service and kernel** | At **100k** records, externally time acceptance through durable mutation/index visibility, authorized bridge/retrieval and completed serialized receipt. Require **p95 ≤25.0 ms** under a frozen load. Separately benchmark the **1M-node kernel** on sparse, heavy-tailed cyclic, 10k in/out hubs and dense cascades. Sweep **1%–80% dirty**, fallback on/off against full recomputation. Require semantic and restart parity, **zero spurious full scans** on certified sparse/no-op/shielded fixtures, and **100%** new payload hash coverage. Report p50/p95/p99, QPS, RSS, edges visited, labels and ranks changed, fallback reason and recovery time. Publish the measured crossover or the tested range if the curves never cross. |

**Why the 500-item rule.** At 5% discordance, independent items and true equality, the normal planning approximation gives **84.1%** probability of meeting the −1 point/−3 bound pair and **70.5%** for the −1/−2.5 pair. Carbon’s 800-item −1/−2 pair gives **80.7%** at 4% discordance even before conversation clustering. These are planning illustrations. The selected rule makes a one-point observed-loss promise with a three-point uncertainty bound. The pinned dataset bytes have SHA-256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`. [Dataset revision](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/tree/98d7416c24c778c2fee6e6f3006e7a073259d48f)

**Pairing contract.** Before constructing maps, reject duplicate keys, missing or extra IDs, nonfinite scores, empty endpoints, and inconsistent cluster, split or replicate assignments. Require `IDs(A) == IDs(B) == expected_IDs`. Use the same resampled units on both arms, seed 42, 10,000 replicates and `.025/.975` quantiles. Resample conversations and related source-question families together where applicable. Replicates add a repeated measurement on the same item. The current helper silently intersects keys. [Current bootstrap](D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/scripts/night1_common.py:407)

For independent paired binary endpoints, preregister an exact conservative interval valid even at zero discordance. Compute separate **97.5% two-sided Clopper–Pearson** intervals for win and loss probabilities from their counts among all matched pairs. Return \([L_w-U_l,U_w-L_l]\), intersected with \([-1,1]\). Bonferroni gives at least 95% joint coverage. With 500 ties its lower endpoint is **−0.873 pp**. Use the same declared method across outcomes. For dependent clusters, report the cluster interval and flag a degenerate bootstrap as insufficient inferential evidence. An item-wise exact interval cannot resolve missing independent clusters.

### Target 5 — Whitepaper, HTML and registry

**Freeze seven chapters plus Appendix A.** Front matter carries the release banner and three scoped claims, each with comparator, denominator, best evidence and counterexample.

| Order | Contents |
|---|---|
| 1 | Dual-drift research question, estimands and claim boundaries |
| 2 | Evidence and methods, metric lineage, complete qrels, gain conventions, splits, pairing, bitemporal cuts and `TRA-1` |
| 3 | Migration geometry and results, including theorem restrictions, six-domain results, dimension jumps, the 1,000-anchor chain, OOD exceptions and the separate B4 budget |
| 4 | Evidence semantics and systems, including scoped authority, lineage, EK5 v3, rank counterexamples, kernel performance and service measurements |
| 5 | Structured and conversational evaluation, with clean, stress and abstention shown separately. Put historical **42/50 vs 46/50** beside its stress result |
| 6 | Failure forensics, falsification ledger, archive provenance, related work and claim limitations. Retain `iq01` and `TRA1_OR_M4_NEEDS_REVIEW` alongside the separate 8/8 hostile blocks |
| 7 | Integrated proof engine and PM-1 through PM-5, each with threshold, observed denominator, evidence status and outstanding dependency |
| Appendix A | Complete attempt ledger. Part I covers the 12 historical bundles. Part II covers the three current post-audit bundles and each subsequent post-R3 attempt, with its epoch and supersession recorded |

**Single source.** Use `docs/research/bundle_registry_spec.json` as the declared inventory and policy. `scripts/build_whitepaper_assets.py` performs `discover → verify_bytes → normalize → validate_bindings → render`. Its canonical output is `docs/research/verified_registry.json`. Generate Markdown blocks, embedded HTML data, semantic tables, SVG coordinates and print content from the same observations.

| Record | Required fields |
|---|---|
| `attempt` | `attempt_id`, bundle-relative path, epoch, role, receipt path/digest, manifest path/digest, nullable observed producer digest and Git SHA with provenance, status, `superseded_by`, warnings |
| `artifact` | Attempt ID, relative path, kind, receipt field or null, declared digest or null, observed full SHA-256, byte size, coverage class `receipt_bound`, `publication_only`, or `missing_declared` |
| `observation` | Stable key, attempt ID, artifact and JSON pointer or row selector, method/comparator, fit/eval domain, query/index dimensions, hop, split, seed, metric/gain/units, denominator, estimate, source precision, interval method/bounds, null reason |
| `claim_binding` | Claim/table/figure/cell ID, selected observation keys, comparator, calculation and rounding rule, evidence status, allowed role and required provenance |
| `gate` | Gate version, endpoint, threshold/operator, required cohort and pairing unit, observed value/CI or null, evidence IDs, and `pending/pass/fail/unavailable` |

My byte pass matched **15/15 primary manifests**, **6/6 listed R3 side-artifact hashes**, and **both S1 ancillary JSON digests**. The **19 historical CSVs** lack receipt-bound hashes. Inventory other side files individually, including S1’s historical per-query JSONL. A present-day digest identifies publication input bytes. Preserve its distinction from historical receipt coverage. Keep LongMemEval v1 superseded, v2 the historical primary, `iq01` a falsifier, and the pre-fix unified LoCoMo run a forensic baseline. For post-R3 runs require Git SHA, producer/config and split-ID digests, plus hashes for every emitted payload except the receipt itself. [S1 receipt](D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/artifacts/flagship1-night1/20261121-flagship1-r1/s1/report_receipt.json) · [Million-node receipt](D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/artifacts/flagship1-night1/20261121-rigor-r3/f2_ek5_v2_1m/report_receipt.json) · [Unified benchmark receipt](D:/GITHUB/CHELATEDAI-STAGE0-NIGHT1/artifacts/flagship1-night1/20261121-rigor-r3/f2_unified_zs_locomo/report_receipt.json)

`--check` rebuilds in memory and fails on inventory mismatch, unsafe paths, ambiguous manifest matches, digest mismatch, missing declared artifacts, duplicate observation keys, ambiguous headline bindings, or byte drift in any generated output. Compute deltas before display rounding where raw rows exist. Mark aggregate-derived deltas and unavailable intervals explicitly.

**HTML v2 verification matrix**

| Surface | Frozen acceptance check |
|---|---|
| Figure 1 | Static 2×3 six-domain migration small multiples, all visible, with one underlying data table |
| Figure 2 | Dimension jumps and chain hops labeled **query dimension → resident index dimension**, plus an 18-cell OOD fit/evaluation matrix. Show native endpoints, denominators and interval availability |
| Figure 3 | Four aligned 0–100% panels for structured, clean, stress and abstention endpoints. Show whiskers only for observed intervals. Pair each panel with a semantic table |
| Figure 4 | Separate 100k service latency and 1M kernel/crossover panels, with units, load, topology and fallback setting. Pending measurements remain visibly pending |
| Figure 5 | Two explanatory SVGs for ambient versus tangent geometry and the four production strata. Supply SVG titles, descriptions and adjacent textual equivalents |
| Accessibility and print | Native MathML, semantic table headers, visible keyboard focus, color-independent markers, 375 px scrollable table regions, and complete no-JS content. Print every panel, legend, digest and release banner. Place the banner inside `<header>`. Use **`--warm: #7d2f0a`**. Its recalculated contrast on `#fdf2ec` is **8.36:1**, correcting Carbon’s `7.84:1`. Neon’s alternative measures `7.51:1`. Check the remaining color pairs and interactions separately before making a page-wide accessibility claim |

## 3. Remaining R3 decisions and evidence limits

- **Ratify three policy choices explicitly:** the outgoing quarantine cap, the `.15` resultant plus 15° split-half A2g guard, and the 500-item clean endpoint with a −3 pp lower bound. Each has a concrete counterexample or sampling rationale. None is uniquely dictated by mathematics.
- **Correct the historical record before carrying it into v2.** The recovered producer establishes the 1,000-anchor chain setting, REPORT versus full-query evaluation, and the A3 down-projection dispatch. The manifest lacks a producer binding, anchor IDs, retained spectra and per-query scores. A new receipted ablation is needed to attribute the Hop-2 loss or promote a bridge.
- **Keep release acceptance pending.** The in-memory checks establish the specified small-graph behavior and identify incorrect generalizations in R1. They do not measure persistent EK5 v3, prospective retrieval superiority, upstream parity, or the 100k service SLO.
