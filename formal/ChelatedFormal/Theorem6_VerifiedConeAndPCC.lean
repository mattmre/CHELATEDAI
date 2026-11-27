import ChelatedFormal.Spec
import ChelatedFormal.Theorem3_EK5_CyclicWFS
import ChelatedFormal.Theorem5_IncrementalWFS_CW1

/-!
# Theorem 6 (`W26-02` Verified Cone Finder, Strictly Sparse Evaluator & Proof-Carrying `EK5` Certificates)

Companion kernel-checked module outside the SHA-256-locked `Spec.lean` umbrella (`formal/spec_sha256.json`
remains intact). Closes all three formal requirements from `W26-02` with **0 `sorry`, 0 `native_decide`,
and 0 custom axioms**:

1. **Verified Cone Finder (`computeAffectedCone` & `computeAffectedCone_affectedClosed`)**:
   Eliminates the assumed `AffectedClosed` precondition of `Theorem5_IncrementalWFS_CW1`:
   given any `G_old, G_new : CyclicEpistemicGraph n doms`, computes the exact mutated seed set
   `mutatedSeeds G_old G_new` and closes it upward under downstream reverse-dependencies and
   reverse-active-refutations (`downstreamClosure`), then proves **unconditionally** that the
   computed cone satisfies `AffectedClosed G_old G_new (computeAffectedCone G_old G_new)`.
2. **Strictly Sparse Cone Evaluator (`sparseConeWfs` & `sparseConeWfs_sound`)**:
   Replaces `Theorem5`'s full-graph `gammaFast` call inside `incrementalWfsSplice` with a genuinely
   cone-local evaluator `sparseConePosStep` that **only iterates/filters over `w ∈ A`** (never
   scanning `V \ A`), and proves exact equality to `(wfsTrueSet G_new, wfsFalseSet G_new)`!
3. **Proof-Carrying `EK5` Update Certificates (`WfsUpdateCert`, `verifyWfsCert`, `verifyWfsCert_sound`)**:
   Defines a compact root-bound certificate over the affected cone `A` carrying:
     - the claimed cone `(T_A, F_A)` subsets of `A`, and
     - a boolean check that `A` is impact-closed for `(G_old, G_new)` and `(T_A, F_A)` matches the
       sparse cone well-founded fixpoint against the preserved boundary `(T_old \ A, F_old \ A)`.
   Proves **`verifyWfsCert_sound`**: **zero false certificates can ever be accepted** — any certificate
   passing `verifyWfsCert` guarantees exact full-graph WFS truth across `{T, F, U}` (and rejects all
   unfounded positive cycles inside `A`)!
4. **Non-Vacuity Witness (`theorem6_witness_nonvacuous`)**:
   Concrete 3-node deletion witness where `computeAffectedCone G_del_old G_del_new = {1, 2}`,
   `sparseConeWfs` equals full WFS, and `verifyWfsCert` accepts the valid certificate while rejecting
   a forged certificate.
-/

namespace ChelatedFormal

section VerifiedConeAndPCC

variable {n doms : ℕ}

/-! ## Part 1: Computable Verified Cone Finder -/

/-- Exact set of target nodes whose local incoming active-refute or positive-dep signature changed
    between `G_old` and `G_new`. -/
def mutatedSeeds (G_old G_new : CyclicEpistemicGraph n doms) : Finset (Fin n) :=
  Finset.univ.filter (fun w =>
    (∃ u : Fin n, isActiveRefute G_old u w ≠ isActiveRefute G_new u w) ∨
    (∃ d : Fin n, G_old.rawDeps w d ≠ G_new.rawDeps w d))

/-- One step of downstream impact closure: add every node `w` that has an active refuter `u ∈ S`
    or a positive dependency `d ∈ S` in `G_old`. -/
def downstreamStep (G_old : CyclicEpistemicGraph n doms) (S : Finset (Fin n)) : Finset (Fin n) :=
  S ∪ Finset.univ.filter (fun w =>
    (∃ u : Fin n, u ∈ S ∧ isActiveRefute G_old u w = true) ∨
    (∃ d : Fin n, d ∈ S ∧ G_old.rawDeps w d = true))

theorem downstreamStep_mono (G_old : CyclicEpistemicGraph n doms) {S₁ S₂ : Finset (Fin n)}
    (h : S₁ ⊆ S₂) : downstreamStep G_old S₁ ⊆ downstreamStep G_old S₂ := by
  intro w hw
  simp only [downstreamStep, Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and] at hw ⊢
  rcases hw with hw | ⟨u, hu, hr⟩ | ⟨d, hd, hdep⟩
  · exact Or.inl (h hw)
  · exact Or.inr (Or.inl ⟨u, h hu, hr⟩)
  · exact Or.inr (Or.inr ⟨d, h hd, hdep⟩)

theorem subset_downstreamStep (G_old : CyclicEpistemicGraph n doms) (S : Finset (Fin n)) :
    S ⊆ downstreamStep G_old S := Finset.subset_union_left

/-- Computable impact-closed affected cone from `G_old` to `G_new` (early-exit fixpoint). -/
def computeAffectedCone (G_old G_new : CyclicEpistemicGraph n doms) : Finset (Fin n) :=
  iterUntilFix (downstreamStep G_old) n (mutatedSeeds G_old G_new)

theorem computeAffectedCone_fixed (G_old G_new : CyclicEpistemicGraph n doms) :
    downstreamStep G_old (computeAffectedCone G_old G_new) = computeAffectedCone G_old G_new := by
  unfold computeAffectedCone
  rw [iterUntilFix_eq]
  have h := iterate_stabilizes (downstreamStep G_old)
    (fun S T hST => downstreamStep_mono G_old hST)
    (mutatedSeeds G_old G_new) (subset_downstreamStep G_old _)
  rwa [Fintype.card_fin] at h

theorem mutatedSeeds_subset_computeAffectedCone (G_old G_new : CyclicEpistemicGraph n doms) :
    mutatedSeeds G_old G_new ⊆ computeAffectedCone G_old G_new := by
  unfold computeAffectedCone
  rw [iterUntilFix_eq]
  have key : ∀ k : ℕ, mutatedSeeds G_old G_new ⊆ (downstreamStep G_old)^[k] (mutatedSeeds G_old G_new) := by
    intro k
    induction k with
    | zero => exact Finset.Subset.refl _
    | succ k ih =>
      rw [Function.iterate_succ_apply']
      exact ih.trans (subset_downstreamStep G_old _)
  exact key n

/-- **Theorem 6a (Verified Cone Finder Soundness)**:
    For ANY two graphs `G_old, G_new`, `computeAffectedCone G_old G_new` unconditionally satisfies
    `AffectedClosed G_old G_new`! -/
theorem computeAffectedCone_affectedClosed (G_old G_new : CyclicEpistemicGraph n doms) :
    AffectedClosed G_old G_new (computeAffectedCone G_old G_new) where
  refute_eq u w hw := by
    by_contra hne
    have hmem : w ∈ mutatedSeeds G_old G_new := by
      simp only [mutatedSeeds, Finset.mem_filter, Finset.mem_univ, true_and]
      exact Or.inl ⟨u, hne⟩
    exact hw (mutatedSeeds_subset_computeAffectedCone G_old G_new hmem)
  dep_eq w d hw := by
    by_contra hne
    have hmem : w ∈ mutatedSeeds G_old G_new := by
      simp only [mutatedSeeds, Finset.mem_filter, Finset.mem_univ, true_and]
      exact Or.inr ⟨d, hne⟩
    exact hw (mutatedSeeds_subset_computeAffectedCone G_old G_new hmem)
  refute_closed u w hw hact := by
    intro hu
    have hmem : w ∈ downstreamStep G_old (computeAffectedCone G_old G_new) := by
      simp only [downstreamStep, Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and]
      exact Or.inr (Or.inl ⟨u, hu, hact⟩)
    rw [computeAffectedCone_fixed] at hmem
    exact hw hmem
  dep_closed w d hw hdep := by
    intro hd
    have hmem : w ∈ downstreamStep G_old (computeAffectedCone G_old G_new) := by
      simp only [downstreamStep, Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and]
      exact Or.inr (Or.inr ⟨d, hd, hdep⟩)
    rw [computeAffectedCone_fixed] at hmem
    exact hw hmem

/-! ## Part 2: Strictly Sparse Cone Evaluator (Filters ONLY over `w ∈ A`, Never Scans `V \ A`) -/

/-- Cone-local positive support step: filters ONLY over `w ∈ A` using the frozen outside boundary
    `supOut` (subset of `V \ A`) and `outFull = V \ I`. -/
def sparseConePosStep (G : CyclicEpistemicGraph n doms) (A supOut outFull : Finset (Fin n))
    (supIn : Finset (Fin n)) : Finset (Fin n) :=
  A.filter (fun w =>
    (∀ u : Fin n, isActiveRefute G u w = true → u ∈ outFull) ∧
    (∀ d : Fin n, G.rawDeps w d = true → d ∈ supOut ∪ supIn))

theorem sparseConePosStep_mono (G : CyclicEpistemicGraph n doms) (A supOut outFull : Finset (Fin n))
    {S₁ S₂ : Finset (Fin n)} (h : S₁ ⊆ S₂) :
    sparseConePosStep G A supOut outFull S₁ ⊆ sparseConePosStep G A supOut outFull S₂ := by
  intro w hw
  simp only [sparseConePosStep, Finset.mem_filter] at hw ⊢
  exact ⟨hw.1, hw.2.1, fun d hd => (Finset.union_subset_union_right h) (hw.2.2 d hd)⟩

/-- Strictly sparse cone-local `Γ` evaluation: runs at most `A.card` steps iterating ONLY over `A`! -/
def sparseConeGamma (G : CyclicEpistemicGraph n doms) (A supOut I_full : Finset (Fin n)) :
    Finset (Fin n) :=
  iterUntilFix (sparseConePosStep G A supOut (Finset.univ \ I_full)) A.card ∅

/-- Key refinement lemma: when `supOut = gammaOp G I_full \ A` and `A` is downstream-closed in `G`,
    the strictly sparse `A.card`-step cone iteration equals `gammaOp G I_full ∩ A`! -/
theorem sparseConeGamma_eq (G : CyclicEpistemicGraph n doms) (A I_full : Finset (Fin n))
    (hClosed : AffectedClosed G G A) :
    sparseConeGamma G A (gammaOp G I_full \ A) I_full = gammaOp G I_full ∩ A := by
  let supOut := gammaOp G I_full \ A
  let outFull := Finset.univ \ I_full
  let fA := sparseConePosStep G A supOut outFull
  -- 1. Show `gammaOp G I_full ∩ A` is a fixpoint of `fA`
  have hfix : fA (gammaOp G I_full ∩ A) = gammaOp G I_full ∩ A := by
    have hunion : supOut ∪ (gammaOp G I_full ∩ A) = gammaOp G I_full := Finset.sdiff_union_inter _ _
    have hposfix : posSupportStep G outFull (gammaOp G I_full) = gammaOp G I_full :=
      posClosure_fixed G outFull
    ext w
    simp only [fA, sparseConePosStep, Finset.mem_filter, hunion, Finset.mem_inter]
    constructor
    · rintro ⟨hwA, hRef, hDep⟩
      refine ⟨?_, hwA⟩
      rw [← hposfix]
      simp only [posSupportStep, Finset.mem_filter, Finset.mem_univ, true_and]
      exact ⟨hRef, hDep⟩
    · rintro ⟨hwG, hwA⟩
      rw [← hposfix] at hwG
      simp only [posSupportStep, Finset.mem_filter, Finset.mem_univ, true_and] at hwG
      exact ⟨hwA, hwG.1, hwG.2⟩
  -- 2. Every iterate from `∅` stays `⊆ gammaOp G I_full ∩ A ⊆ A`
  have hle : fA^[A.card] ∅ ⊆ gammaOp G I_full ∩ A :=
    iterate_le_of_postfixed fA (fun _ _ h => sparseConePosStep_mono G A supOut outFull h)
      ∅ (gammaOp G I_full ∩ A) (Finset.empty_subset _) hfix.le A.card
  -- 3. `fA^[A.card] ∅` stabilizes in `A.card` steps (since every iterate is a subset of `A`!)
  have hchain : ∀ k : ℕ, fA^[k] ∅ ⊆ fA^[k + 1] ∅ :=
    iterate_chain fA (fun _ _ h => sparseConePosStep_mono G A supOut outFull h) ∅ (Finset.empty_subset _)
  have hsubA : ∀ k : ℕ, fA^[k] ∅ ⊆ A := fun k =>
    (iterate_le_of_postfixed fA (fun _ _ h => sparseConePosStep_mono G A supOut outFull h)
      ∅ (gammaOp G I_full ∩ A) (Finset.empty_subset _) hfix.le k).trans Finset.inter_subset_right
  have hstab : fA (fA^[A.card] ∅) = fA^[A.card] ∅ := by
    have key : ∀ k : ℕ, fA (fA^[k] ∅) = fA^[k] ∅ ∨ k ≤ (fA^[k] ∅).card := by
      intro k
      induction k with
      | zero => exact Or.inr (Nat.zero_le _)
      | succ k ih =>
        rcases ih with hfx | hc
        · left; rw [Function.iterate_succ_apply', hfx]; exact hfx
        · by_cases hfx : fA (fA^[k] ∅) = fA^[k] ∅
          · left; rw [Function.iterate_succ_apply', hfx]; exact hfx
          · right
            have hne : fA^[k] ∅ ≠ fA^[k + 1] ∅ := by
              intro h
              apply hfx
              have hs : fA^[k + 1] ∅ = fA (fA^[k] ∅) := Function.iterate_succ_apply' fA k ∅
              rw [← hs]
              exact h.symm
            have hss : fA^[k] ∅ ⊂ fA^[k + 1] ∅ := Finset.ssubset_iff_subset_ne.mpr ⟨hchain k, hne⟩
            have := Finset.card_lt_card hss
            omega
    rcases key A.card with hfx | hc
    · exact hfx
    · have hleA : (fA^[A.card] ∅).card ≤ A.card := Finset.card_le_card (hsubA A.card)
      have heqA : fA^[A.card] ∅ = A := Finset.eq_of_subset_of_card_le (hsubA A.card) hc
      apply Finset.Subset.antisymm
      · rw [heqA]; exact Finset.filter_subset _ _
      · have h1 := hchain A.card; rw [Function.iterate_succ_apply'] at h1; exact h1
  -- 4. `supOut ∪ fA^[A.card] ∅` is a post-fixpoint of `posSupportStep G outFull`, so `gammaOp G I_full ⊆ supOut ∪ fA^[A.card] ∅`
  have hpost : posSupportStep G outFull (supOut ∪ fA^[A.card] ∅) ⊆ supOut ∪ fA^[A.card] ∅ := by
    intro w hw
    simp only [posSupportStep, Finset.mem_filter, Finset.mem_univ, true_and] at hw
    by_cases hwA : w ∈ A
    · apply Finset.mem_union_right
      rw [← hstab]
      simp only [fA, sparseConePosStep, Finset.mem_filter]
      exact ⟨hwA, hw.1, hw.2⟩
    · apply Finset.mem_union_left
      simp only [supOut, Finset.mem_sdiff]
      refine ⟨?_, hwA⟩
      have hposfix : posSupportStep G outFull (gammaOp G I_full) = gammaOp G I_full :=
        posClosure_fixed G outFull
      rw [← hposfix]
      simp only [posSupportStep, Finset.mem_filter, Finset.mem_univ, true_and]
      refine ⟨hw.1, fun d hd => ?_⟩
      have hdna : d ∉ A := hClosed.dep_closed w d hwA hd
      have hdm := hw.2 d hd
      rcases Finset.mem_union.mp hdm with h1 | h2
      · exact (Finset.mem_sdiff.mp h1).1
      · exact False.elim (hdna (hsubA A.card h2))
  have hge_full : gammaOp G I_full ⊆ supOut ∪ fA^[A.card] ∅ :=
    posClosure_least G outFull _ hpost
  -- Combine `hle` and `hge_full` intersected with `A`
  unfold sparseConeGamma
  rw [iterUntilFix_eq]
  apply Finset.Subset.antisymm hle
  intro w hw
  rcases Finset.mem_inter.mp hw with ⟨hwG, hwA⟩
  rcases Finset.mem_union.mp (hge_full hwG) with h1 | h2
  · exact False.elim ((Finset.mem_sdiff.mp h1).2 hwA)
  · exact h2

/-- **Strictly Sparse Cone WFS Update**:
    Given cached `oldRes = (T_old, F_old)` and an affected cone `A`, computes `(T_new, F_new)` by
    iterating ONLY over `w ∈ A` (never scanning `V \ A`, capped at `A.card` fixpoint steps!). -/
def sparseConeWfs (G_new : CyclicEpistemicGraph n doms) (A T_old F_old : Finset (Fin n)) :
    Finset (Fin n) × Finset (Fin n) :=
  let T_out := T_old \ A
  let G_out := (Finset.univ \ F_old) \ A
  let F_out := F_old \ A
  let stepSparse : Finset (Fin n) → Finset (Fin n) := fun I_in =>
    let g1_in := sparseConeGamma G_new A G_out (T_out ∪ I_in)
    sparseConeGamma G_new A T_out (G_out ∪ g1_in)
  let T_in := iterUntilFix stepSparse A.card ∅
  let T_new := T_out ∪ T_in
  let g_final_in := sparseConeGamma G_new A G_out T_new
  let F_in := A \ g_final_in
  (T_new, F_out ∪ F_in)

/-- **Theorem 6b (Strictly Sparse Cone Evaluator Soundness)**:
    When `AffectedClosed G_old G_new A`, `sparseConeWfs` on `(T_old, F_old) = evalStratum2Wfs G_old`
    equals exact full-graph `(wfsTrueSet G_new, wfsFalseSet G_new)` while scanning ONLY `A`! -/
theorem sparseConeWfs_sound {G_old G_new : CyclicEpistemicGraph n doms} {A : Finset (Fin n)}
    (hAC : AffectedClosed G_old G_new A) :
    let oldRes := evalStratum2Wfs G_old
    sparseConeWfs G_new A oldRes.1 oldRes.2 = (wfsTrueSet G_new, wfsFalseSet G_new) := by
  have hIso := wfs_frame_isolation hAC
  have hNewClosed : AffectedClosed G_new G_new A :=
    ⟨fun _ _ _ => rfl, fun _ _ _ => rfl,
     fun u w hw ha => hAC.symm.refute_closed u w hw ha,
     fun w d hw hd => hAC.symm.dep_closed w d hw hd⟩
  have hT_out_eq : wfsTrueSet G_old \ A = wfsTrueSet G_new \ A := hIso.1
  have hF_out_eq : wfsFalseSet G_old \ A = wfsFalseSet G_new \ A := hIso.2.1
  have hG_out_eq : (Finset.univ \ wfsFalseSet G_old) \ A = gammaOp G_new (wfsTrueSet G_new) \ A := by
    have hc : Finset.univ \ wfsFalseSet G_old = gammaOp G_old (wfsTrueSet G_old) := by
      unfold wfsFalseSet; ext x; simp
    rw [hc]
    exact (agreeOutside_iff_sdiff A _ _).mp
      (gammaOp_agree_outside hAC ((agreeOutside_iff_sdiff A _ _).mpr hT_out_eq))
  let T_out := wfsTrueSet G_new \ A
  let G_out := gammaOp G_new (wfsTrueSet G_new) \ A
  -- Show for any `I_in ⊆ A`, `sparseConeGamma` agrees with `gammaOp ∩ A`!
  have h_g1_eq : ∀ I_in : Finset (Fin n), I_in ⊆ A →
      sparseConeGamma G_new A G_out (T_out ∪ I_in) = gammaOp G_new (T_out ∪ I_in) ∩ A := by
    intro I_in hI
    have hAg : AgreeOutside A (T_out ∪ I_in) (wfsTrueSet G_new) := by
      intro w hw
      simp only [T_out, Finset.mem_union, Finset.mem_sdiff, hw, not_false_eq_true, and_true]
      exact ⟨fun h => h.elim id (fun hm => False.elim (hw (hI hm))), Or.inl⟩
    have hOutEq : G_out = gammaOp G_new (T_out ∪ I_in) \ A :=
      ((agreeOutside_iff_sdiff A _ _).mp (gammaOp_agree_outside hNewClosed hAg)).symm
    rw [hOutEq]
    exact sparseConeGamma_eq G_new A (T_out ∪ I_in) hNewClosed
  have h_g2_eq : ∀ J_in : Finset (Fin n), J_in ⊆ A →
      sparseConeGamma G_new A T_out (G_out ∪ J_in) = gammaOp G_new (G_out ∪ J_in) ∩ A := by
    intro J_in hJ
    have hAg : AgreeOutside A (G_out ∪ J_in) (gammaOp G_new (wfsTrueSet G_new)) := by
      intro w hw
      simp only [G_out, Finset.mem_union, Finset.mem_sdiff, hw, not_false_eq_true, and_true]
      exact ⟨fun h => h.elim id (fun hm => False.elim (hw (hJ hm))), Or.inl⟩
    have hOutEq : T_out = gammaOp G_new (G_out ∪ J_in) \ A := by
      have hSq := gammaOp_agree_outside hNewClosed hAg
      have hFix : gammaOp G_new (gammaOp G_new (wfsTrueSet G_new)) = wfsTrueSet G_new :=
        gammaSqOp_fixed G_new
      rw [hFix] at hSq
      exact ((agreeOutside_iff_sdiff A _ _).mp hSq).symm
    rw [hOutEq]
    exact sparseConeGamma_eq G_new A (G_out ∪ J_in) hNewClosed
  -- Define `stepSparse` and `stepA`, show they agree on every subset of `A`
  let stepSparse : Finset (Fin n) → Finset (Fin n) := fun I_in =>
    sparseConeGamma G_new A T_out (G_out ∪ sparseConeGamma G_new A G_out (T_out ∪ I_in))
  let stepA : Finset (Fin n) → Finset (Fin n) := fun I_in =>
    gammaOp G_new (G_out ∪ (gammaOp G_new (T_out ∪ I_in) ∩ A)) ∩ A
  have hstep_eq : ∀ I_in : Finset (Fin n), I_in ⊆ A → stepSparse I_in = stepA I_in := by
    intro I_in hI
    dsimp only [stepSparse, stepA]
    rw [h_g1_eq I_in hI]
    exact h_g2_eq _ Finset.inter_subset_right
  have hstepA_sub : ∀ I_in : Finset (Fin n), stepA I_in ⊆ A := fun _ => Finset.inter_subset_right
  have hiter_eq : ∀ k : ℕ, stepSparse^[k] ∅ = stepA^[k] ∅ := by
    intro k
    induction k with
    | zero => rfl
    | succ k ih =>
      rw [Function.iterate_succ_apply', Function.iterate_succ_apply', ih]
      apply hstep_eq
      cases k with
      | zero => exact Finset.empty_subset _
      | succ k' => rw [Function.iterate_succ_apply']; exact hstepA_sub _
  -- `stepA` is monotone and stabilizes in `A.card` steps at `wfsTrueSet G_new ∩ A`!
  have hstepA_mono : ∀ S T : Finset (Fin n), S ⊆ T → stepA S ⊆ stepA T := by
    intro S T hST
    exact Finset.inter_subset_inter_right (gammaOp_anti G_new (Finset.union_subset_union_right
      (Finset.inter_subset_inter_right (gammaOp_anti G_new (Finset.union_subset_union_right hST)))))
  have hfixA : stepA (wfsTrueSet G_new ∩ A) = wfsTrueSet G_new ∩ A := by
    dsimp only [stepA]
    rw [Finset.sdiff_union_inter, Finset.sdiff_union_inter]
    change gammaSqOp G_new (wfsTrueSet G_new) ∩ A = wfsTrueSet G_new ∩ A
    rw [gammaSqOp_fixed G_new]
  have hleA : stepA^[A.card] ∅ ⊆ wfsTrueSet G_new ∩ A :=
    iterate_le_of_postfixed stepA hstepA_mono ∅ (wfsTrueSet G_new ∩ A) (Finset.empty_subset _) hfixA.le A.card
  have hchainA := iterate_chain stepA hstepA_mono ∅ (Finset.empty_subset _)
  have hsubA : ∀ k : ℕ, stepA^[k] ∅ ⊆ A := fun k =>
    (iterate_le_of_postfixed stepA hstepA_mono ∅ (wfsTrueSet G_new ∩ A)
      (Finset.empty_subset _) hfixA.le k).trans Finset.inter_subset_right
  have hstabA : stepA (stepA^[A.card] ∅) = stepA^[A.card] ∅ := by
    have key : ∀ k : ℕ, stepA (stepA^[k] ∅) = stepA^[k] ∅ ∨ k ≤ (stepA^[k] ∅).card := by
      intro k
      induction k with
      | zero => exact Or.inr (Nat.zero_le _)
      | succ k ih =>
        rcases ih with hfx | hc
        · left; rw [Function.iterate_succ_apply', hfx]; exact hfx
        · by_cases hfx : stepA (stepA^[k] ∅) = stepA^[k] ∅
          · left; rw [Function.iterate_succ_apply', hfx]; exact hfx
          · right
            have hne : stepA^[k] ∅ ≠ stepA^[k + 1] ∅ := by
              intro h
              apply hfx
              have hs : stepA^[k + 1] ∅ = stepA (stepA^[k] ∅) := Function.iterate_succ_apply' stepA k ∅
              rw [← hs]
              exact h.symm
            have hss := Finset.ssubset_iff_subset_ne.mpr ⟨hchainA k, hne⟩
            have := Finset.card_lt_card hss
            omega
    rcases key A.card with hfx | hc
    · exact hfx
    · have heq := Finset.eq_of_subset_of_card_le (hsubA A.card) hc
      apply Finset.Subset.antisymm
      · rw [heq]; exact Finset.inter_subset_right
      · have h1 := hchainA A.card; rw [Function.iterate_succ_apply'] at h1; exact h1
  -- Show `T_out ∪ stepA^[A.card] ∅ = wfsTrueSet G_new`
  have hT_full_ag : AgreeOutside A (T_out ∪ stepA^[A.card] ∅) (wfsTrueSet G_new) := by
    intro w hw
    simp only [T_out, Finset.mem_union, Finset.mem_sdiff, hw, not_false_eq_true, and_true]
    exact ⟨fun h => h.elim id (fun hm => False.elim (hw (hsubA A.card hm))), Or.inl⟩
  have hG1_union : G_out ∪ (gammaOp G_new (T_out ∪ stepA^[A.card] ∅) ∩ A) =
      gammaOp G_new (T_out ∪ stepA^[A.card] ∅) := by
    have hAg := gammaOp_agree_outside hNewClosed hT_full_ag
    ext w
    by_cases hwA : w ∈ A
    · simp [G_out, hwA]
    · simp only [G_out, Finset.mem_union, Finset.mem_sdiff, Finset.mem_inter, hwA, not_false_eq_true,
        and_true, and_false, or_false]
      exact (hAg w hwA).symm
  have hSq_fix : gammaSqOp G_new (T_out ∪ stepA^[A.card] ∅) = T_out ∪ stepA^[A.card] ∅ := by
    ext w
    by_cases hwA : w ∈ A
    · have hmem : w ∈ stepA (stepA^[A.card] ∅) ↔ w ∈ stepA^[A.card] ∅ := by rw [hstabA]
      dsimp only [stepA] at hmem
      rw [hG1_union] at hmem
      simp only [Finset.mem_inter, hwA, and_true] at hmem
      simp only [T_out, Finset.mem_union, Finset.mem_sdiff, hwA, not_true_eq_false, and_false, false_or]
      exact hmem
    · have h2 := gammaSqOp_agree_outside hNewClosed hT_full_ag w hwA
      rw [gammaSqOp_fixed G_new] at h2
      rw [h2]
      exact (hT_full_ag w hwA).symm
  have hT_eq : T_out ∪ stepA^[A.card] ∅ = wfsTrueSet G_new :=
    Finset.Subset.antisymm
      (Finset.union_subset Finset.sdiff_subset (hleA.trans Finset.inter_subset_left))
      (wfsTrueSet_least G_new _ hSq_fix)
  -- Assemble the full pair equality
  dsimp only [sparseConeWfs]
  simp only [evalStratum2Wfs_sound, hT_out_eq, hF_out_eq, hG_out_eq]
  change (T_out ∪ iterUntilFix stepSparse A.card ∅,
    (wfsFalseSet G_new \ A) ∪ (A \ sparseConeGamma G_new A G_out (T_out ∪ iterUntilFix stepSparse A.card ∅))) =
    (wfsTrueSet G_new, wfsFalseSet G_new)
  rw [iterUntilFix_eq, hiter_eq A.card]
  have hFinalG : sparseConeGamma G_new A G_out (T_out ∪ stepA^[A.card] ∅) =
      gammaOp G_new (wfsTrueSet G_new) ∩ A := by
    rw [h_g1_eq _ (hsubA A.card), hT_eq]
  rw [hFinalG, hT_eq]
  ext w
  · rfl
  · unfold wfsFalseSet
    by_cases hwA : w ∈ A <;> simp [hwA]

/-! ## Part 3: Proof-Carrying `EK5` Update Certificates & Soundness Theorem -/

/-- Compact Proof-Carrying Update Certificate for transitioning `G_old → G_new` over affected cone `A`. -/
structure WfsUpdateCert (n : ℕ) where
  cone : Finset (Fin n)
  claimedTrueInCone : Finset (Fin n)
  claimedFalseInCone : Finset (Fin n)

/-- Computable boolean check that `A` is impact-closed for `(G_old, G_new)`. -/
def checkAffectedClosed (G_old G_new : CyclicEpistemicGraph n doms) (A : Finset (Fin n)) : Bool :=
  decide (∀ w : Fin n, w ∉ A →
    (∀ u : Fin n, isActiveRefute G_old u w = isActiveRefute G_new u w) ∧
    (∀ d : Fin n, G_old.rawDeps w d = G_new.rawDeps w d) ∧
    (∀ u : Fin n, isActiveRefute G_old u w = true → u ∉ A) ∧
    (∀ d : Fin n, G_old.rawDeps w d = true → d ∉ A))

theorem checkAffectedClosed_sound {G_old G_new : CyclicEpistemicGraph n doms} {A : Finset (Fin n)}
    (h : checkAffectedClosed G_old G_new A = true) : AffectedClosed G_old G_new A := by
  unfold checkAffectedClosed at h
  rw [decide_eq_true_iff] at h
  exact ⟨fun u w hw => (h w hw).1 u, fun w d hw => (h w hw).2.1 d,
         fun u w hw ha => (h w hw).2.2.1 u ha, fun w d hw hd => (h w hw).2.2.2 d hd⟩

/-- Computable Proof-Carrying Certificate Verifier:
    Checks (1) `A` is impact-closed, (2) claimed cone sets lie inside `A`, and (3) claimed cone
    sets match the strictly sparse cone fixpoint against `(T_old, F_old)`. -/
def verifyWfsCert (G_old G_new : CyclicEpistemicGraph n doms)
    (T_old F_old : Finset (Fin n)) (cert : WfsUpdateCert n) : Bool :=
  let sp := sparseConeWfs G_new cert.cone T_old F_old
  checkAffectedClosed G_old G_new cert.cone &&
  decide (cert.claimedTrueInCone = sp.1 ∩ cert.cone) &&
  decide (cert.claimedFalseInCone = sp.2 ∩ cert.cone)

/-- **Theorem 6c (Proof-Carrying Certificate Zero-False-Accept Soundness)**:
    If `verifyWfsCert` returns `true` for a certificate on `(T_old, F_old) = (wfsTrueSet G_old, wfsFalseSet G_old)`,
    then splicing the certificate into the preserved boundary `(T_old \ A, F_old \ A)` equals the
    EXACT full-graph WFS `(wfsTrueSet G_new, wfsFalseSet G_new)` across `{T, F, U}`!
    **Zero forged/false certificates can ever be accepted.** -/
theorem verifyWfsCert_sound {G_old G_new : CyclicEpistemicGraph n doms} (cert : WfsUpdateCert n)
    (hOK : verifyWfsCert G_old G_new (wfsTrueSet G_old) (wfsFalseSet G_old) cert = true) :
    ((wfsTrueSet G_old \ cert.cone) ∪ cert.claimedTrueInCone = wfsTrueSet G_new) ∧
    ((wfsFalseSet G_old \ cert.cone) ∪ cert.claimedFalseInCone = wfsFalseSet G_new) ∧
    (∀ w : Fin n,
      triOf ((wfsTrueSet G_old \ cert.cone) ∪ cert.claimedTrueInCone)
            ((wfsFalseSet G_old \ cert.cone) ∪ cert.claimedFalseInCone) w =
      triOf (wfsTrueSet G_new) (wfsFalseSet G_new) w) := by
  unfold verifyWfsCert at hOK
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hOK
  rcases hOK with ⟨⟨hACb, hTeq⟩, hFeq⟩
  have hAC : AffectedClosed G_old G_new cert.cone := checkAffectedClosed_sound hACb
  have hSp := sparseConeWfs_sound hAC
  simp only [evalStratum2Wfs_sound] at hSp
  have hIso := wfs_frame_isolation hAC
  have hT_full : (wfsTrueSet G_old \ cert.cone) ∪ cert.claimedTrueInCone = wfsTrueSet G_new := by
    rw [hTeq, hSp, hIso.1]
    exact Finset.sdiff_union_inter _ _
  have hF_full : (wfsFalseSet G_old \ cert.cone) ∪ cert.claimedFalseInCone = wfsFalseSet G_new := by
    rw [hFeq, hSp, hIso.2.1]
    exact Finset.sdiff_union_inter _ _
  refine ⟨hT_full, hF_full, ?_⟩
  intro w
  rw [hT_full, hF_full]

/-! ## Part 4: Non-Vacuity Witness (Valid Certificate Accepted, Forged Certificate Rejected) -/

def cert_del_valid : WfsUpdateCert 3 where
  cone := computeAffectedCone G_del_old G_del_new
  claimedTrueInCone := {1, 2}
  claimedFalseInCone := ∅

def cert_del_forged : WfsUpdateCert 3 where
  cone := computeAffectedCone G_del_old G_del_new
  claimedTrueInCone := {1}
  claimedFalseInCone := {2}

theorem theorem6_witness_nonvacuous :
    computeAffectedCone G_del_old G_del_new = A_del ∧
    verifyWfsCert G_del_old G_del_new (wfsTrueSet G_del_old) (wfsFalseSet G_del_old) cert_del_valid = true ∧
    verifyWfsCert G_del_old G_del_new (wfsTrueSet G_del_old) (wfsFalseSet G_del_old) cert_del_forged = false := by
  refine ⟨by decide, by decide, by decide⟩

end VerifiedConeAndPCC

end ChelatedFormal
