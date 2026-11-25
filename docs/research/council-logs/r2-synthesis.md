# Tri-Model Council — Refinement Round 2 Synthesis & Cross-Check Mandate (2026-11-21)

**Council Roster:** `gemini-4-carbon` (`thought_level=high`), `gemini-4-neon` (`thought_level=xhigh`), `gpt-6.2-astra` (`reasoningEffort=xhigh`)  
**Stage:** **Refinement Round 2 of 3** (Cross-Checking Peer R1 Proofs, Stress-Testing Edge Cases, Locking Exact Algorithms, Theorems, Gates & Whitepaper/HTML Specs before R3 Final Sign-Off)  
**Read-Only Rule:** Do **NOT** edit repository files during R2–R3. Execute only read-only in-memory Python/math probes.

---

## 1. Refinement Round 1 Reports Archived (3/3 Complete)
- **Carbon R1:** `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1\docs\research\council-logs\r1-carbon.md`
- **Neon XHigh R1:** `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1\docs\research\council-logs\r1-neon.md`
- **Astra 6.2 XHigh R1:** `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1\docs\research\council-logs\r1-astra.md`

---

## 2. Major R1 Convergence Achieved (Ready to Lock Pending R2 Cross-Check)
1. **Direct-$B_i$ vs. Declared Lineage:** Neon formally conceded blind transitive `SUPERSEDE+` to Astra via the **Forged-Superseder Transitivity Attack** (`r1-neon.md §1.1`). All three models now converge on: **Direct-$B_i$ for ordinary claims** + **Transitive supersession ONLY on explicitly declared `(tenant, entity, single-valued slot, version_lineage)` edges** (Astra R1 §1, Carbon R1 §1.1) + explicit **two-mode dependency split** (`COPY_OF`/`DERIVES_CURRENT` $\in D_C$ vs `AMENDS`/`DERIVES_SNAPSHOT` $\in D_V \setminus D_C$).
2. **Stale-Rank Impossibility & Exact Linear Positive-SCC WFS:** Both Astra (`3-node` counterexample, `56,568` rank-only edits / `48` erroneous second-step skips) and Neon (`5-node` counterexample against mixing unranked skips with rank shields) independently proved that rank shields MUST preserve exact proof-rank certificates even when WFS truth labels stay unchanged. Astra verified the rank-aware positive-SCC WFS solver across **`4,718,592` 3-node single-edge edits + `65,536` 2-node jumps + `50,000` 3-node jumps + `28,561` bitemporal cuts (`0` mismatches)**; Carbon verified **`100,000` random cyclic ops (`0` mismatches)** and `dirty=0` on 10k in/out hubs; Neon verified **`51,280` ops (`0` mismatches)** and `0.002–0.006 ms` on 10k hubs.
3. **Stratified $K$-Axis Ordering:** All three models independently converged on **`K = 2` minimal mathematical core + `K = 4` executable production store** (**Bitemporal Visibility $\to$ Authority-Qualified Relations $\to$ Epistemic WFS $\to$ Temporal Currency**) + **`K = 8` architectural unification** (`TRA-1` Integrity at $\Sigma_0$, Bounded Continuous Tie-Breaker strictly inside `ADMIT` at $\Sigma_4$, Axis 6 Specificity scoped strictly as theoretical extension).
4. **Riemannian Geodesic Polar (`A2g`) & `f1_4096_ood` Diagnosis:** All three models proved complementary regimes of `A2g` vs `A2c` (Astra: `<1e-12°` vs `5.27°` on angular dilation `s=1.8`, undefined pole on `{±e_i}`; Carbon: $\pi/2$ gnomonic equator barrier vs isotropic pole jitter; Neon: `+0.0068` on `E2→E0` synthetic cone + sagitta flip proof), explained `Centered Affine Ridge`'s OOD centroid collapse (`0.0000–0.1704` full 18-cell range; `0.0000–0.1362` on 384↔4096), and explained `A3_subspace > A3_full` on `D4096→D384` up-projections (`tr/4096` vs `tr/r` shrinkage dilution).
5. **Single-Source Registry Builder & WCAG AAA HTML v2:** All three converged on `scripts/build_whitepaper_assets.py` with `--check` CI drift gate, static zero-JS tables/SVGs, native MathML, print-visible banner inside `<header>`, 6-domain small multiples, aligned CI-whisker panels, and WCAG AAA `--warm` (`#7d2f0a` = `7.84:1` / `#8a3208` = `7.51:1`).

---

## 3. Exact R2 Cross-Check & Adjudication Targets (Stress-Test Peer R1 Proposals!)

In Refinement Round 2, every model MUST explicitly cross-check the **other two models' R1 reports** and resolve ALL FIVE concrete technical tensions below so that R3 is pure final convergence:

### R2 Target 1 — Stress-Test All Three R1 Rank-Shielding & Positive-SCC Algorithms Against Each Other!
Compare **Astra R1 §1** (raw Fitting certificates separate from WFS truth; fast cut requires same raw truth AND earliest certificate rank), **Neon R1 §1.4** (boundary-aware Lyapunov Dijkstra ranks + strict `O(1)` skip rules), and **Carbon R1 §1.4** (`Theorem 1.1` with `dirty_shield_loss[w]` cone pruning + Pitfall B $O(1)$ external shield delta):
- **Adversarial Probe A (Check Carbon's `Theorem 1.1` against Astra's 3-node stale-rank counterexample!):** Suppose `u ∈ D` keeps its WFS truth label (`T -> T`) **but its rank increases** (`rank: 0 -> 2`), causing `u` to cease shielding an out-of-cone node `w ∉ D` (`rank(w) = 1`) whose `strict_shield_cnt[w]` drops from `1 -> 0`! Does Carbon's Pitfall A/B cone pruning miss putting `w` into `D`? Does Neon's `UnifiedWFSStore` catch rank-only increases propagating to `w`? Does Astra's raw-rank separation catch it? **Execute an in-memory multi-step counterexample test against all three variants and specify the ONE unified, bug-free pseudocode for `EK5 v3`!**
- **Adversarial Probe B (Check Neon's pure-WFS mutual-`SUPERSEDE` rule $B_i^{\text{cyc}} \leftarrow V_j \wedge \operatorname{not} B_i^{\text{cyc}}$ vs. Astra's write-time rejection + imported quarantine):** Does Neon's $B_i^{\text{cyc}}$ rule handle: (i) $A \leftrightarrow_{\text{SUP}} B$ with $C \xrightarrow{\text{COPY\_OF}} A$? (ii) $A \leftrightarrow_{\text{SUP}} B$ where external $X \xrightarrow{\text{SUP}} A$ (dominating $A$ only, NOT $B$ directly — what happens to $B$?!)? (iii) declared version-lineage `C REF+SUP B, B SUP A` (Astra R1 §1 vs Carbon R1 §1.1 vs Neon R1 §1.1)? Lock ONE crystal-clear Stratum 2 specification!

### R2 Target 2 — Stress-Test Carbon's Effective Authority Ceiling $\tilde{\alpha}(v)$ (Anti-Authority-Laundering) vs. Astra's Scoped Capability Authority!
- Carbon R1 (`Counterexample 2.3`) proposed propagating effective authority $\tilde{\alpha}(v) = \min(\alpha_{\text{raw}}(v), \min_{d \in D_V(v)} \tilde{\alpha}(d))$ along $D_V$ so a high-authority agent summarizing a tier-0 rumor cannot launder it to refute tier-2 truth.
- **Adversarial Probe:**
  1. What if an authorized auditor (`auth = 4`) writes a **refutation or snapshot-audit** referencing a rumor (`AMENDS` / `DERIVES_SNAPSHOT`) vs. endorsing a rumor (`COPY_OF` / `DERIVES_CURRENT`)? Should $\tilde{\alpha}(v)$ propagate over $D_V$ unconditionally, OR only over epistemic endorsement (`COPY_OF`/`DERIVES_FROM`), and how does $\tilde{\alpha}(v)$ update incrementally under deletions & bitemporal rewinds (when `r_0` is not yet visible at `(vt_cut, tx_cut)`)?
  2. How do we combine Carbon's monotone ceiling $\tilde{\alpha}(v)$ with Astra's **slot/capability-scoped authority** (so a user preference authority doesn't accidentally override a signed sensor receipt!) without violating the $K=4$ Stratified Stabilization Theorem?

### R2 Target 3 — Stress-Test Flagship #1 Exact Math, `A2g` Details, & `f1_4096_ood` Forensic Precision
1. **Cross-check Astra R1 §3's precision corrections on `f1_4096_ood`:** Confirm (i) zero Legal-trained bridges exist (only Code & SciFact fit domains), (ii) full OOD Ridge range `0.0000–0.1704` (`0.0000–0.1362` on 384↔4096), (iii) at `SciFact` Hop 2 (`n=2000, 2560→1024`), $n-1 = 1999 > 1024$ gives full thin rank `1024` (separating unobserved source complement from zero-singular SVD completion and in-sample radius bias!), and (iv) check why `SciFact` `native_old`/`full_c2o` differed between `indomain_dim_jumps` (`0.6866`/`0.7266`, `REPORT` split) and `zero_shot_ood` (`0.6484`/`0.7645`, full 300 queries!) — verify directly in `scratch/run_pristine_f1_4096_ood.py` or `f1_4096_ood_manifest.json`!
2. **Cross-check `A2g` (`Riemannian Geodesic Polar`) Exact Numerical Details:**
   - Reconcile **Astra R1 §3** (orthonormal tangent basis $(d_x-1)\times(d_y-1)$, `atan2`, stable `sinc`, noting exact composition when intermediate poles coincide inside injectivity radius $\|w\| < \pi$ AND noting `A2g` is NOT globally conformal: radial stretch $s$ vs transverse stretch $\sin(s\theta)/\sin\theta$!), **Neon R1 §3.1** (`pole_floor = 0.15`, smooth `0.95π tanh`), and **Carbon R1 §3.1** (`pole_floor = 0.25`, Householder tangent frames, gnomonic $\pi/2$ barrier).
   - Lock the EXACT `GeodesicPolarBridge` formula, `pole_floor` threshold, tangent projection method, smooth vs hard cut-locus guard, and exact promotion/falsification rule for our cached-embedding ablation run!
3. **Cross-check Carbon R1's `s >= 1.0` cap on `A2c-OOS-Radius` up-projection (`d_in < d_out`):** Does capping $s \ge 1.0$ help or hurt when up-projecting an isotropic encoder into a narrower cone (`E0 384 → E1 768` or `D384 → D4096`)? Compare against `s_subspace = ||Y_val,c P_Q||_F / ||X_val,c Q||_F` (Neon R1 §3.2.3) + `CSLS`!

### R2 Target 4 — Lock the FINAL Exact Numbers of `PM-1..PM-5` (Zero Residual Drift Across All 3 Models)
Resolve the final 3 minor numeric differences between Astra R1, Neon R1, and Carbon R1:
1. **`PM-1b` Clean Margin:** Reconcile Carbon's `N=800` pooled (`point >= -1.0 pp`, `LCB95 >= -2.0 pp`) vs. Neon's `N=500` (`point >= -1.0 pp`, `LCB95 >= -2.5 pp`) vs. Astra's `N=500` (`point >= -1.0 pp`, `LCB95 >= -3.0 pp`, explicitly warning that `single-session-user` is only a slice of `N=500` and reporting both `single-session-user` and full unmodified endpoints separately with cluster-aware CIs for shared LoCoMo conversations!).
2. **`PM-2` & `PM-3` Wording:** Lock Astra/Neon's absolute `-3.0 pp` (`LCB95 >= -5.0 pp`) for `PM-2` and exact labeling of original `PM-3` gate vs. any newly proposed R1 stretch sub-gate (`+10 pp` over upstream).
3. **`PM-4` & `PM-5`:** Confirm exact human-audit slice (`100` stratified items), exact `paired_bootstrap_delta` fail-closed contract + exact matched-pair interval option for sparse/zero-discordance ties (Astra R1 §4 catch!), and exact `PM-5` separation between `100k` end-to-end service SLO (`p95 <= 25 ms`) and `1M` kernel benchmark + crossover sweep.

### R2 Target 5 — Lock Single-Source Registry Schema, Exact Chapter Order, & HTML v2 Verification Matrix
Lock ONE shared chapter order (reconciling Astra's 9 chapters vs Neon/Carbon's 7 chapters + Appendix A), ONE exact `scripts/build_whitepaper_assets.py` schema (handling all **15** current verified bundles + `19` historical CSVs without receipt-bound hashes + `2` S1 ancillary JSON digests + post-R3 runs), ONE exact `--warm` AAA hex (`#7d2f0a` `7.84:1` vs `#8a3208` `7.51:1`), and ONE exact static-SVG + accessible-table figure list so the Implementation Pass has zero ambiguity!

---

## 4. Required Output Format for Refinement Round 2
1. **Peer R1 Cross-Check Audit Table** (`Peer`, `R1 Claim / Algorithm`, `Verdict`, `Executable Evidence / Edge Case Found`).
2. **Resolved Final Specifications for Targets 1–5** (concise, crystal-clear, ready to freeze in R3 — including exact unified `EK5 v3` pseudocode, exact `A2g` + `A2c-Subspace+OOS` equations, exact `PM-1..PM-5` table, and exact Whitepaper/HTML/Registry blueprint).
3. **Any Remaining Micro-Dissent or Open Risk for R3 Final Adjudication** (explicitly flag anything still non-unanimous so R3 resolves 100% of open items before we touch code!).
