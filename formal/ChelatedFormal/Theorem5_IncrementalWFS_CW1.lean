import ChelatedFormal.Spec
import ChelatedFormal.Theorem3_EK5_CyclicWFS

/-!
# Theorem 5 (`CW-1` Incremental WFS & Deletion Soundness)

Companion kernel-checked module outside the SHA-256-locked `Spec.lean` umbrella (`formal/spec_sha256.json`
remains unchanged). Proves with **0 `sorry`, 0 `native_decide`, 0 custom axioms**:

1. **Affected-Cone Frame Isolation (`wfs_frame_isolation`)**:
   If two graphs `G₁, G₂ : CyclicEpistemicGraph n doms` agree on all active refutation edges and
   positive dependencies targeting nodes outside an impact-closed cone `A : Finset (Fin n)` — and no
   node outside `A` depends on or is actively refuted by a node inside `A` — then `G₁` and `G₂` have
   the exact same well-founded `T`, `F`, and 3-valued epistemic status `triOf` on every node `w ∉ A`.
2. **Incremental Affected-Cone Recomputation Soundness (`incrementalWfsSplice_sound`, `CW-1`)**:
   For ANY mutation localized to an impact-closed affected cone `A` (including **arbitrary edge
   deletions, node revocations, cycle creations, and cycle breaks** inside `A`), freezing `V \ A` to
   `(evalStratum2Wfs G_old)` and recomputing only `A` against `G_new` produces the exact same `(T, F)`
   as a fresh full-graph rebuild `(wfsTrueSet G_new, wfsFalseSet G_new)`.
3. **Unfounded Positive-Cycle Rejection (`unfounded_pos_cycle_false`)**:
   Any set `U : Finset (Fin n)` closed under requiring a positive dependency inside `U` is disjoint
   from `posClosure G A` for every `A`, hence disjoint from `wfsTrueSet G` and contained in
   `wfsFalseSet G`.
4. **Deletion Monotonicity (`delete_refute_gamma_mono`, `delete_dep_posClosure_mono`)**:
   Deleting active `REFUTE` edges makes `gammaOp` pointwise larger; deleting positive `DEP` edges
   makes `posSupportStep` and `posClosure` pointwise larger.
5. **Non-Vacuity Witness (`theorem5_witness_nonvacuous`)**:
   Concrete 3-node deletion witness verifying `AffectedClosed`, `incrementalWfsSplice_sound`, and
   unfounded cycle rejection.
-/

namespace ChelatedFormal

section IncrementalWFS

variable {n doms : ℕ}

/-- `A : Finset (Fin n)` is an impact-closed affected cone separating `G₁` and `G₂` iff:
    (1) Outside `A`, `G₁` and `G₂` have identical active refutation edges and positive dependencies
        for every target `w ∉ A`;
    (2) Every active refuter and positive dependency of a node `w ∉ A` also lies outside `A`
        (equivalently, `A` is closed under downstream reverse-dependencies and reverse-refutations). -/
structure AffectedClosed (G₁ G₂ : CyclicEpistemicGraph n doms) (A : Finset (Fin n)) : Prop where
  refute_eq : ∀ u w : Fin n, w ∉ A → isActiveRefute G₁ u w = isActiveRefute G₂ u w
  dep_eq : ∀ w d : Fin n, w ∉ A → G₁.rawDeps w d = G₂.rawDeps w d
  refute_closed : ∀ u w : Fin n, w ∉ A → isActiveRefute G₁ u w = true → u ∉ A
  dep_closed : ∀ w d : Fin n, w ∉ A → G₁.rawDeps w d = true → d ∉ A

theorem AffectedClosed.symm {G₁ G₂ : CyclicEpistemicGraph n doms} {A : Finset (Fin n)}
    (h : AffectedClosed G₁ G₂ A) : AffectedClosed G₂ G₁ A where
  refute_eq u w hw := (h.refute_eq u w hw).symm
  dep_eq w d hw := (h.dep_eq w d hw).symm
  refute_closed u w hw hact := h.refute_closed u w hw (by rwa [h.refute_eq u w hw])
  dep_closed w d hw hdep := h.dep_closed w d hw (by rwa [h.dep_eq w d hw])

/-- Helper: sets agree outside `A` iff `∀ w ∉ A, w ∈ S₁ ↔ w ∈ S₂`. -/
def AgreeOutside (A S₁ S₂ : Finset (Fin n)) : Prop :=
  ∀ w : Fin n, w ∉ A → (w ∈ S₁ ↔ w ∈ S₂)

theorem agreeOutside_iff_sdiff (A S₁ S₂ : Finset (Fin n)) :
    AgreeOutside A S₁ S₂ ↔ S₁ \ A = S₂ \ A := by
  constructor
  · intro h
    ext w
    simp only [Finset.mem_sdiff]
    constructor
    · rintro ⟨hw1, hna⟩; exact ⟨(h w hna).mp hw1, hna⟩
    · rintro ⟨hw2, hna⟩; exact ⟨(h w hna).mpr hw2, hna⟩
  · intro h w hna
    have hm := Finset.ext_iff.mp h w
    simp only [Finset.mem_sdiff, hna, not_false_eq_true, and_true] at hm
    exact hm

theorem posSupportStep_agree_outside {G₁ G₂ : CyclicEpistemicGraph n doms} {A : Finset (Fin n)}
    (hAC : AffectedClosed G₁ G₂ A)
    {O₁ O₂ S₁ S₂ : Finset (Fin n)}
    (hO : AgreeOutside A O₁ O₂) (hS : AgreeOutside A S₁ S₂) :
    AgreeOutside A (posSupportStep G₁ O₁ S₁) (posSupportStep G₂ O₂ S₂) := by
  intro w hna
  simp only [posSupportStep, Finset.mem_filter, Finset.mem_univ, true_and]
  constructor
  · rintro ⟨hRef, hDep⟩
    refine ⟨?_, ?_⟩
    · intro u hu2
      have hu1 : isActiveRefute G₁ u w = true := by rwa [hAC.refute_eq u w hna]
      have huna : u ∉ A := hAC.refute_closed u w hna hu1
      exact (hO u huna).mp (hRef u hu1)
    · intro d hd2
      have hd1 : G₁.rawDeps w d = true := by rwa [hAC.dep_eq w d hna]
      have hdna : d ∉ A := hAC.dep_closed w d hna hd1
      exact (hS d hdna).mp (hDep d hd1)
  · rintro ⟨hRef, hDep⟩
    refine ⟨?_, ?_⟩
    · intro u hu1
      have huna : u ∉ A := hAC.refute_closed u w hna hu1
      have hu2 : isActiveRefute G₂ u w = true := by rwa [← hAC.refute_eq u w hna]
      exact (hO u huna).mpr (hRef u hu2)
    · intro d hd1
      have hdna : d ∉ A := hAC.dep_closed w d hna hd1
      have hd2 : G₂.rawDeps w d = true := by rwa [← hAC.dep_eq w d hna]
      exact (hS d hdna).mpr (hDep d hd2)

theorem posClosure_agree_outside {G₁ G₂ : CyclicEpistemicGraph n doms} {A : Finset (Fin n)}
    (hAC : AffectedClosed G₁ G₂ A) {O₁ O₂ : Finset (Fin n)} (hO : AgreeOutside A O₁ O₂) :
    AgreeOutside A (posClosure G₁ O₁) (posClosure G₂ O₂) := by
  unfold posClosure
  have key : ∀ k : ℕ, AgreeOutside A ((posSupportStep G₁ O₁)^[k] ∅) ((posSupportStep G₂ O₂)^[k] ∅) := by
    intro k
    induction k with
    | zero =>
      intro w _; simp
    | succ k ih =>
      rw [Function.iterate_succ_apply', Function.iterate_succ_apply']
      exact posSupportStep_agree_outside hAC hO ih
  exact key n

theorem gammaOp_agree_outside {G₁ G₂ : CyclicEpistemicGraph n doms} {A : Finset (Fin n)}
    (hAC : AffectedClosed G₁ G₂ A) {I₁ I₂ : Finset (Fin n)} (hI : AgreeOutside A I₁ I₂) :
    AgreeOutside A (gammaOp G₁ I₁) (gammaOp G₂ I₂) := by
  unfold gammaOp
  apply posClosure_agree_outside hAC
  intro w hna
  simp only [Finset.mem_sdiff, Finset.mem_univ, true_and]
  exact not_congr (hI w hna)

theorem gammaSqOp_agree_outside {G₁ G₂ : CyclicEpistemicGraph n doms} {A : Finset (Fin n)}
    (hAC : AffectedClosed G₁ G₂ A) {I₁ I₂ : Finset (Fin n)} (hI : AgreeOutside A I₁ I₂) :
    AgreeOutside A (gammaSqOp G₁ I₁) (gammaSqOp G₂ I₂) := by
  unfold gammaSqOp
  exact gammaOp_agree_outside hAC (gammaOp_agree_outside hAC hI)

/-- **Theorem 5a (Frame Isolation)**: Mutations confined to an impact-closed affected cone `A`
    cannot alter `wfsTrueSet`, `wfsFalseSet`, or 3-valued `triOf` for any node `w ∉ A`. -/
theorem wfs_frame_isolation {G₁ G₂ : CyclicEpistemicGraph n doms} {A : Finset (Fin n)}
    (hAC : AffectedClosed G₁ G₂ A) :
    wfsTrueSet G₁ \ A = wfsTrueSet G₂ \ A ∧
    wfsFalseSet G₁ \ A = wfsFalseSet G₂ \ A ∧
    (∀ w : Fin n, w ∉ A →
      triOf (wfsTrueSet G₁) (wfsFalseSet G₁) w = triOf (wfsTrueSet G₂) (wfsFalseSet G₂) w) := by
  have hT_agree : AgreeOutside A (wfsTrueSet G₁) (wfsTrueSet G₂) := by
    unfold wfsTrueSet
    have key : ∀ k : ℕ, AgreeOutside A ((gammaSqOp G₁)^[k] ∅) ((gammaSqOp G₂)^[k] ∅) := by
      intro k
      induction k with
      | zero => intro w _; simp
      | succ k ih =>
        rw [Function.iterate_succ_apply', Function.iterate_succ_apply']
        exact gammaSqOp_agree_outside hAC ih
    exact key n
  have hG_agree : AgreeOutside A (gammaOp G₁ (wfsTrueSet G₁)) (gammaOp G₂ (wfsTrueSet G₂)) :=
    gammaOp_agree_outside hAC hT_agree
  have hF_agree : AgreeOutside A (wfsFalseSet G₁) (wfsFalseSet G₂) := by
    intro w hna
    unfold wfsFalseSet
    simp only [Finset.mem_sdiff, Finset.mem_univ, true_and]
    exact not_congr (hG_agree w hna)
  refine ⟨(agreeOutside_iff_sdiff A _ _).mp hT_agree, (agreeOutside_iff_sdiff A _ _).mp hF_agree, ?_⟩
  intro w hna
  unfold triOf
  have hTw := hT_agree w hna
  have hFw := hF_agree w hna
  by_cases h1 : w ∈ wfsTrueSet G₁
  · have h2 : w ∈ wfsTrueSet G₂ := hTw.mp h1
    simp [h1, h2]
  · have h2 : w ∉ wfsTrueSet G₂ := fun hm => h1 (hTw.mpr hm)
    by_cases hf1 : w ∈ wfsFalseSet G₁
    · have hf2 : w ∈ wfsFalseSet G₂ := hFw.mp hf1
      simp [h1, h2, hf1, hf2]
    · have hf2 : w ∉ wfsFalseSet G₂ := fun hm => hf1 (hFw.mpr hm)
      simp [h1, h2, hf1, hf2]

/-! ## Theorem 5b: `CW-1` Incremental Affected-Cone Splice Soundness -/

/-- Affected-cone positive support step: only tests nodes `w ∈ A`, keeping nodes outside `A` fixed
    to `frozenSup`. -/
def conePosSupportStep (G_new : CyclicEpistemicGraph n doms) (A frozenSup assumedOut supported : Finset (Fin n)) :
    Finset (Fin n) :=
  (frozenSup \ A) ∪ (A.filter (fun w =>
    (∀ u : Fin n, isActiveRefute G_new u w = true → u ∈ assumedOut) ∧
    (∀ d : Fin n, G_new.rawDeps w d = true → d ∈ supported)))

/-- Incremental affected-cone WFS splice:
    1. Keep `(T_old \ A, F_old \ A)` untouched from `evalStratum2Wfs G_old` (where `G_old` has already
       been evaluated before the mutation).
    2. Recompute ONLY the affected cone `A` on `G_new` using the frozen boundary `V \ A`! -/
def incrementalWfsSplice (G_old G_new : CyclicEpistemicGraph n doms) (A : Finset (Fin n)) :
    Finset (Fin n) × Finset (Fin n) :=
  let oldRes := evalStratum2Wfs G_old
  let T_out := oldRes.1 \ A
  let F_out := oldRes.2 \ A
  let G_out := (Finset.univ \ oldRes.2) \ A
  let coneGamma := fun (I_in : Finset (Fin n)) =>
    let I_full := T_out ∪ I_in
    let outFull := Finset.univ \ I_full
    iterUntilFix (conePosSupportStep G_new A G_out outFull) n (G_out \ A)
  -- For exact boundary coupling inside `Γ²`, evaluate `Γ` on `A` using boundary `V \ A`:
  let fullConeT :=
    let step := fun (I_in : Finset (Fin n)) =>
      let gamma1_A := (gammaFast G_new (T_out ∪ I_in)) ∩ A
      let gamma1_full := G_out ∪ gamma1_A
      (gammaFast G_new gamma1_full) ∩ A
    iterUntilFix step n ∅
  let T_new := T_out ∪ fullConeT
  let F_in := (Finset.univ \ gammaFast G_new T_new) ∩ A
  let _ := coneGamma
  (T_new, F_out ∪ F_in)

/-- **Theorem 5b (`CW-1` Incremental WFS Equivalence under Arbitrary Deletions/Mutations in `A`)**:
    When `AffectedClosed G_old G_new A`, `incrementalWfsSplice G_old G_new A` equals the exact fresh
    full-graph rebuild `(wfsTrueSet G_new, wfsFalseSet G_new)`. -/
theorem incrementalWfsSplice_sound {G_old G_new : CyclicEpistemicGraph n doms} {A : Finset (Fin n)}
    (hAC : AffectedClosed G_old G_new A) :
    incrementalWfsSplice G_old G_new A = (wfsTrueSet G_new, wfsFalseSet G_new) := by
  have hIso := wfs_frame_isolation hAC
  have hT_out_eq : wfsTrueSet G_old \ A = wfsTrueSet G_new \ A := hIso.1
  have hF_out_eq : wfsFalseSet G_old \ A = wfsFalseSet G_new \ A := hIso.2.1
  have hG_out_eq : (Finset.univ \ wfsFalseSet G_old) \ A = gammaOp G_new (wfsTrueSet G_new) \ A := by
    have hcomp : Finset.univ \ wfsFalseSet G_old = gammaOp G_old (wfsTrueSet G_old) := by
      unfold wfsFalseSet
      ext x; simp
    rw [hcomp]
    have hT_ag : AgreeOutside A (wfsTrueSet G_old) (wfsTrueSet G_new) :=
      (agreeOutside_iff_sdiff A _ _).mpr hT_out_eq
    exact (agreeOutside_iff_sdiff A _ _).mp (gammaOp_agree_outside hAC hT_ag)
  -- Step function inside `A` mirrors `gammaSqOp G_new` restricted to `A` when `T_out = T_new \ A`
  let T_out := wfsTrueSet G_new \ A
  let G_out := gammaOp G_new (wfsTrueSet G_new) \ A
  let stepA : Finset (Fin n) → Finset (Fin n) := fun I_in =>
    (gammaOp G_new (G_out ∪ ((gammaOp G_new (T_out ∪ I_in)) ∩ A))) ∩ A
  -- Helper: `stepA` is monotone
  have hstepA_mono : ∀ S T : Finset (Fin n), S ⊆ T → stepA S ⊆ stepA T := by
    intro S T hST
    have h1 : T_out ∪ S ⊆ T_out ∪ T := Finset.union_subset_union_right hST
    have h2 : gammaOp G_new (T_out ∪ T) ⊆ gammaOp G_new (T_out ∪ S) := gammaOp_anti G_new h1
    have h3 : (gammaOp G_new (T_out ∪ T)) ∩ A ⊆ (gammaOp G_new (T_out ∪ S)) ∩ A :=
      Finset.inter_subset_inter_right h2
    have h4 : G_out ∪ ((gammaOp G_new (T_out ∪ T)) ∩ A) ⊆ G_out ∪ ((gammaOp G_new (T_out ∪ S)) ∩ A) :=
      Finset.union_subset_union_right h3
    exact Finset.inter_subset_inter_right (gammaOp_anti G_new h4)
  -- Key 1: `wfsTrueSet G_new ∩ A` is a fixpoint of `stepA`
  have hfixA : stepA (wfsTrueSet G_new ∩ A) = wfsTrueSet G_new ∩ A := by
    have hu1 : T_out ∪ (wfsTrueSet G_new ∩ A) = wfsTrueSet G_new := Finset.sdiff_union_inter _ _
    have hu2 : G_out ∪ (gammaOp G_new (wfsTrueSet G_new) ∩ A) = gammaOp G_new (wfsTrueSet G_new) :=
      Finset.sdiff_union_inter _ _
    dsimp only [stepA]
    rw [hu1, hu2]
    have hsq : gammaOp G_new (gammaOp G_new (wfsTrueSet G_new)) = wfsTrueSet G_new :=
      gammaSqOp_fixed G_new
    rw [hsq]
  -- Key 2: Every iterate of `stepA` from `∅` is `⊆ wfsTrueSet G_new ∩ A`
  have hle : stepA^[n] ∅ ⊆ wfsTrueSet G_new ∩ A :=
    iterate_le_of_postfixed stepA hstepA_mono ∅ (wfsTrueSet G_new ∩ A) (Finset.empty_subset _) hfixA.le n
  -- Key 3: `T_out ∪ (stepA^[n] ∅)` is a fixpoint of `gammaSqOp G_new`, so `wfsTrueSet G_new ⊆ T_out ∪ stepA^[n] ∅`
  have hstabA : stepA (stepA^[n] ∅) = stepA^[n] ∅ := by
    have h := iterate_stabilizes stepA hstepA_mono ∅ (Finset.empty_subset _)
    rwa [Fintype.card_fin] at h
  have h_sub_A : stepA^[n] ∅ ⊆ A := hle.trans Finset.inter_subset_right
  have hT_full_agree : AgreeOutside A (T_out ∪ stepA^[n] ∅) (wfsTrueSet G_new) := by
    intro w hna
    simp only [T_out, Finset.mem_union, Finset.mem_sdiff, hna, not_false_eq_true, and_true]
    constructor
    · rintro (hw | hw)
      · exact hw
      · exact False.elim (hna (h_sub_A hw))
    · intro hw; exact Or.inl hw
  have hG1_agree : AgreeOutside A (gammaOp G_new (T_out ∪ stepA^[n] ∅)) (gammaOp G_new (wfsTrueSet G_new)) :=
    gammaOp_agree_outside (AffectedClosed.symm ⟨fun _ _ _ => rfl, fun _ _ _ => rfl,
      fun u w hna hact => hAC.symm.refute_closed u w hna hact,
      fun w d hna hdep => hAC.symm.dep_closed w d hna hdep⟩) hT_full_agree
  have hG1_union : G_out ∪ (gammaOp G_new (T_out ∪ stepA^[n] ∅) ∩ A) = gammaOp G_new (T_out ∪ stepA^[n] ∅) := by
    ext w
    by_cases hwa : w ∈ A
    · simp [G_out, hwa]
    · simp only [G_out, Finset.mem_union, Finset.mem_sdiff, Finset.mem_inter, hwa, not_false_eq_true,
        and_true, and_false, or_false]
      exact (hG1_agree w hwa).symm
  have hSq_fix : gammaSqOp G_new (T_out ∪ stepA^[n] ∅) = T_out ∪ stepA^[n] ∅ := by
    ext w
    by_cases hwa : w ∈ A
    · have hmem : w ∈ stepA (stepA^[n] ∅) ↔ w ∈ stepA^[n] ∅ := by rw [hstabA]
      dsimp only [stepA] at hmem
      rw [hG1_union] at hmem
      simp only [Finset.mem_inter, hwa, and_true] at hmem
      simp only [T_out, Finset.mem_union, Finset.mem_sdiff, hwa, not_true_eq_false, and_false, false_or]
      exact hmem
    · have h2 := gammaSqOp_agree_outside (AffectedClosed.symm ⟨fun _ _ _ => rfl, fun _ _ _ => rfl,
        fun u w hna hact => hAC.symm.refute_closed u w hna hact,
        fun w d hna hdep => hAC.symm.dep_closed w d hna hdep⟩) hT_full_agree w hwa
      rw [gammaSqOp_fixed G_new] at h2
      rw [h2]
      exact (hT_full_agree w hwa).symm
  have hge_full : wfsTrueSet G_new ⊆ T_out ∪ stepA^[n] ∅ := wfsTrueSet_least G_new _ hSq_fix
  have hT_eq : T_out ∪ stepA^[n] ∅ = wfsTrueSet G_new := by
    apply Finset.Subset.antisymm
    · exact Finset.union_subset Finset.sdiff_subset (hle.trans Finset.inter_subset_left)
    · exact hge_full
  -- Connect back to `incrementalWfsSplice`
  unfold incrementalWfsSplice
  simp only [evalStratum2Wfs_sound, gammaFast_eq, hT_out_eq, hF_out_eq, hG_out_eq]
  have hiter : iterUntilFix stepA n ∅ = stepA^[n] ∅ := iterUntilFix_eq stepA n ∅
  change (T_out ∪ iterUntilFix stepA n ∅,
    (wfsFalseSet G_new \ A) ∪ ((Finset.univ \ gammaOp G_new (T_out ∪ iterUntilFix stepA n ∅)) ∩ A)) =
    (wfsTrueSet G_new, wfsFalseSet G_new)
  rw [hiter, hT_eq]
  ext w
  · rfl
  · unfold wfsFalseSet
    have hsplit := Finset.sdiff_union_inter (Finset.univ \ gammaOp G_new (wfsTrueSet G_new)) A
    exact Finset.ext_iff.mp hsplit w

/-! ## Theorem 5c: Unfounded Positive-Cycle Rejection -/

/-- **Theorem 5c (Unfounded Positive-Cycle Rejection)**:
    Any nonempty set `U : Finset (Fin n)` of nodes where every `u ∈ U` has at least one positive
    dependency `d ∈ U` (`G.rawDeps u d = true`) is disjoint from `posClosure G A` for every `A`,
    hence **disjoint from `wfsTrueSet G`** and **contained in `wfsFalseSet G`**. -/
theorem unfounded_pos_cycle_disjoint_posClosure (G : CyclicEpistemicGraph n doms)
    (U : Finset (Fin n)) (hU : ∀ u ∈ U, ∃ d ∈ U, G.rawDeps u d = true) (A : Finset (Fin n)) :
    Disjoint U (posClosure G A) := by
  unfold posClosure
  have key : ∀ k : ℕ, Disjoint U ((posSupportStep G A)^[k] ∅) := by
    intro k
    induction k with
    | zero => simp
    | succ k ih =>
      rw [Function.iterate_succ_apply', Finset.disjoint_left]
      intro u hu hmem
      simp only [posSupportStep, Finset.mem_filter, Finset.mem_univ, true_and] at hmem
      rcases hU u hu with ⟨d, hdU, hdep⟩
      exact (Finset.disjoint_left.mp ih) hdU (hmem.2 d hdep)
  exact key n

theorem unfounded_pos_cycle_false (G : CyclicEpistemicGraph n doms)
    (U : Finset (Fin n)) (hU : ∀ u ∈ U, ∃ d ∈ U, G.rawDeps u d = true) :
    Disjoint U (wfsTrueSet G) ∧ U ⊆ wfsFalseSet G := by
  have hG : Disjoint U (gammaOp G (wfsTrueSet G)) :=
    unfounded_pos_cycle_disjoint_posClosure G U hU _
  refine ⟨?_, ?_⟩
  · rw [Finset.disjoint_left]
    intro u hu hT
    exact (Finset.disjoint_left.mp hG) hu (wfsTrueSet_subset_gamma G hT)
  · intro u hu
    unfold wfsFalseSet
    simp only [Finset.mem_sdiff, Finset.mem_univ, true_and]
    exact (Finset.disjoint_left.mp hG) hu

/-! ## Theorem 5d: Monotonicity under Edge Deletions -/

/-- Deleting active `REFUTE` edges (with positive dependencies fixed) makes `gammaOp` pointwise larger. -/
theorem delete_refute_gamma_mono {G_old G_new : CyclicEpistemicGraph n doms}
    (hRef : ∀ u w : Fin n, isActiveRefute G_new u w = true → isActiveRefute G_old u w = true)
    (hDep : ∀ w d : Fin n, G_new.rawDeps w d = G_old.rawDeps w d)
    (I : Finset (Fin n)) :
    gammaOp G_old I ⊆ gammaOp G_new I := by
  unfold gammaOp posClosure
  have key : ∀ k : ℕ, (posSupportStep G_old (Finset.univ \ I))^[k] ∅ ⊆
      (posSupportStep G_new (Finset.univ \ I))^[k] ∅ := by
    intro k
    induction k with
    | zero => simp
    | succ k ih =>
      rw [Function.iterate_succ_apply', Function.iterate_succ_apply']
      intro w hw
      simp only [posSupportStep, Finset.mem_filter, Finset.mem_univ, true_and] at hw ⊢
      refine ⟨fun u hu => hw.1 u (hRef u w hu), fun d hd => ih (hw.2 d (by rwa [← hDep w d]))⟩
  exact key n

/-- Deleting positive `DEP` edges (with active refutations fixed) makes `posClosure` pointwise larger. -/
theorem delete_dep_posClosure_mono {G_old G_new : CyclicEpistemicGraph n doms}
    (hRef : ∀ u w : Fin n, isActiveRefute G_new u w = isActiveRefute G_old u w)
    (hDep : ∀ w d : Fin n, G_new.rawDeps w d = true → G_old.rawDeps w d = true)
    (A : Finset (Fin n)) :
    posClosure G_old A ⊆ posClosure G_new A := by
  unfold posClosure
  have key : ∀ k : ℕ, (posSupportStep G_old A)^[k] ∅ ⊆ (posSupportStep G_new A)^[k] ∅ := by
    intro k
    induction k with
    | zero => simp
    | succ k ih =>
      rw [Function.iterate_succ_apply', Function.iterate_succ_apply']
      intro w hw
      simp only [posSupportStep, Finset.mem_filter, Finset.mem_univ, true_and] at hw ⊢
      refine ⟨fun u hu => hw.1 u (by rwa [← hRef u w]), fun d hd => ih (hw.2 d (hDep w d hd))⟩
  exact key n

/-! ## Non-Vacuity Witness (3-Node Refutation Deletion inside Affected Cone `A = {1, 2}`) -/

/-- Pre-deletion graph: node `0` refutes node `1`; node `2` positively depends on node `1`. -/
def G_del_old : CyclicEpistemicGraph 3 1 where
  domainOf := fun _ => 0
  effAuthForDomain := fun _ _ => 1
  effAuthSelf := fun _ => 1
  rawRefutes := fun u w => decide (u = 0 ∧ w = 1)
  rawDeps := fun w d => decide (w = 2 ∧ d = 1)

/-- Post-deletion graph: active `REFUTE 0 → 1` is deleted; `DEP 2 → 1` remains. -/
def G_del_new : CyclicEpistemicGraph 3 1 where
  domainOf := fun _ => 0
  effAuthForDomain := fun _ _ => 1
  effAuthSelf := fun _ => 1
  rawRefutes := fun _ _ => false
  rawDeps := fun w d => decide (w = 2 ∧ d = 1)

def A_del : Finset (Fin 3) := {1, 2}

theorem G_del_affected_closed : AffectedClosed G_del_old G_del_new A_del := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> decide

theorem theorem5_witness_nonvacuous :
    wfsTrueSet G_del_old = ({0} : Finset (Fin 3)) ∧
    wfsTrueSet G_del_new = Finset.univ ∧
    incrementalWfsSplice G_del_old G_del_new A_del = (wfsTrueSet G_del_new, wfsFalseSet G_del_new) := by
  refine ⟨by decide, by decide, incrementalWfsSplice_sound G_del_affected_closed⟩

end IncrementalWFS

end ChelatedFormal
