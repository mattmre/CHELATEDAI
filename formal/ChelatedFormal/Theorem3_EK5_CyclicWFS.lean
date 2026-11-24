import ChelatedFormal.Spec

/-!
# Theorem 3 (`EK5` Stratum 1–2): cyclic alternating-fixpoint Well-Founded Semantics

Proves `ChelatedFormal.Theorem3_EK5_CyclicWFS_Stmt` (locked in `Spec.lean`, SHA-256 in
`spec_sha256.json`) with **0 `sorry`, 0 `native_decide`, 0 custom axioms**:

* `posSupportStep` is monotone in `supported` and in `assumedOut`; hence `posClosure` is monotone in
  `assumedOut`, and `posClosure` is an `n`-step (cardinality-bounded) least fixpoint of `posSupportStep`;
* `gammaOp` is antimonotone, `gammaSqOp` is monotone;
* `wfsTrueSet = (gammaSqOp)^[n] ∅` is the least fixpoint of `gammaSqOp`, and
  `wfsTrueSet ⊆ gammaOp (wfsTrueSet)`, hence `Disjoint (wfsTrueSet) (wfsFalseSet)`;
* the equal-authority mutual `REFUTE` ring `G_cyc` has `T = F = ∅` while `{0}` is a non-empty two-valued
  fixpoint of `gammaOp` (so `lfp` is not the unique fixpoint);
* `evalStratum2Wfs` is a computable evaluator (early-exit fixpoint iteration, used by `ek5_ref_cli`)
  with a kernel-checked equality `evalStratum2Wfs_sound` to `(wfsTrueSet, wfsFalseSet)`;
* Stratum 1 → Stratum 2 non-interference: the authority ceiling `effAuthD` is a function of
  `(tier, D_V, missing)` alone, so changing `rawRefutes` (Stratum 2 input) cannot move it, and the
  target-domain predicate `activeRefute` is exactly `eff_auth_for_domain(u, domain(w)) ≥ eff_auth(w)`
  (`ek5_v3_stratified.py:322-327`, `:1180`).
-/

namespace ChelatedFormal

/-! ## Generic finite-lattice iteration lemmas -/

section Generic

variable {α : Type} [Fintype α] [DecidableEq α]

/-- Early-exit iteration: stop as soon as `f x = x`. -/
def iterUntilFix (f : Finset α → Finset α) : ℕ → Finset α → Finset α
  | 0, x => x
  | m + 1, x => if f x = x then x else iterUntilFix f m (f x)

/-- Early-exit iteration agrees with plain `m`-fold iteration (unconditionally). -/
theorem iterUntilFix_eq (f : Finset α → Finset α) :
    ∀ (m : ℕ) (x : Finset α), iterUntilFix f m x = f^[m] x := by
  intro m
  induction m with
  | zero => intro x; rfl
  | succ m ih =>
    intro x
    by_cases hfx : f x = x
    · have hfix : Function.IsFixedPt f x := hfx
      have h2 : f^[m + 1] x = x := hfix.iterate (m + 1)
      simp only [iterUntilFix, hfx, ite_true]
      exact h2.symm
    · simp only [iterUntilFix, hfx, ite_false]
      rw [ih (f x), Function.iterate_succ_apply]

theorem iterate_chain (f : Finset α → Finset α)
    (hmono : ∀ S T : Finset α, S ⊆ T → f S ⊆ f T) (x : Finset α) (hx : x ⊆ f x) :
    ∀ k : ℕ, f^[k] x ⊆ f^[k + 1] x := by
  intro k
  induction k with
  | zero => simpa using hx
  | succ k ih =>
    calc f^[k + 1] x = f (f^[k] x) := Function.iterate_succ_apply' f k x
      _ ⊆ f (f^[k + 1] x) := hmono _ _ ih
      _ = f^[k + 1 + 1] x := (Function.iterate_succ_apply' f (k + 1) x).symm

/-- A monotone ascending chain over `Finset α` is a fixpoint after `Fintype.card α` steps. -/
theorem iterate_stabilizes (f : Finset α → Finset α)
    (hmono : ∀ S T : Finset α, S ⊆ T → f S ⊆ f T) (x : Finset α) (hx : x ⊆ f x) :
    f (f^[Fintype.card α] x) = f^[Fintype.card α] x := by
  have hchain := iterate_chain f hmono x hx
  have key : ∀ k : ℕ, f (f^[k] x) = f^[k] x ∨ k ≤ (f^[k] x).card := by
    intro k
    induction k with
    | zero => exact Or.inr (Nat.zero_le _)
    | succ k ih =>
      rcases ih with hfix | hcard
      · left
        have hsucc : f^[k + 1] x = f^[k] x := by
          rw [Function.iterate_succ_apply', hfix]
        rw [hsucc]
        exact hfix
      · by_cases hfix : f (f^[k] x) = f^[k] x
        · left
          have hsucc : f^[k + 1] x = f^[k] x := by
            rw [Function.iterate_succ_apply', hfix]
          rw [hsucc]
          exact hfix
        · right
          have hsub : f^[k] x ⊆ f^[k + 1] x := hchain k
          have hne : f^[k] x ≠ f^[k + 1] x := by
            intro h
            apply hfix
            have : f^[k + 1] x = f (f^[k] x) := Function.iterate_succ_apply' f k x
            rw [← this]
            exact h.symm
          have hss : f^[k] x ⊂ f^[k + 1] x := Finset.ssubset_iff_subset_ne.mpr ⟨hsub, hne⟩
          have := Finset.card_lt_card hss
          omega
  rcases key (Fintype.card α) with hfix | hcard
  · exact hfix
  · have hle : (f^[Fintype.card α] x).card ≤ Fintype.card α := Finset.card_le_univ _
    have heq : (f^[Fintype.card α] x).card = Fintype.card α := le_antisymm hle hcard
    have huniv : f^[Fintype.card α] x = Finset.univ := Finset.eq_univ_of_card _ heq
    apply Finset.Subset.antisymm
    · rw [huniv]; exact Finset.subset_univ _
    · have hsub : f^[Fintype.card α] x ⊆ f^[Fintype.card α + 1] x := hchain _
      rw [Function.iterate_succ_apply'] at hsub
      exact hsub

/-- Least-fixpoint property: any post-fixpoint `S` containing the start dominates every iterate. -/
theorem iterate_le_of_postfixed (f : Finset α → Finset α)
    (hmono : ∀ S T : Finset α, S ⊆ T → f S ⊆ f T) (x S : Finset α) (hxS : x ⊆ S)
    (hS : f S ⊆ S) : ∀ k : ℕ, f^[k] x ⊆ S := by
  intro k
  induction k with
  | zero => simpa using hxS
  | succ k ih =>
    rw [Function.iterate_succ_apply']
    exact (hmono _ _ ih).trans hS

end Generic

/-! ## Monotonicity of the positive-support step and the closure -/

section Graph

variable {n doms : ℕ} (G : CyclicEpistemicGraph n doms)

theorem posSupportStep_mono_supported (A : Finset (Fin n)) {S₁ S₂ : Finset (Fin n)}
    (h : S₁ ⊆ S₂) : posSupportStep G A S₁ ⊆ posSupportStep G A S₂ := by
  intro w hw
  simp only [posSupportStep, Finset.mem_filter, Finset.mem_univ, true_and] at hw ⊢
  exact ⟨hw.1, fun d hd => h (hw.2 d hd)⟩

theorem posSupportStep_mono_out {A₁ A₂ : Finset (Fin n)} (h : A₁ ⊆ A₂) (S : Finset (Fin n)) :
    posSupportStep G A₁ S ⊆ posSupportStep G A₂ S := by
  intro w hw
  simp only [posSupportStep, Finset.mem_filter, Finset.mem_univ, true_and] at hw ⊢
  exact ⟨fun u hu => h (hw.1 u hu), hw.2⟩

theorem posClosure_mono {A₁ A₂ : Finset (Fin n)} (h : A₁ ⊆ A₂) :
    posClosure G A₁ ⊆ posClosure G A₂ := by
  unfold posClosure
  have key : ∀ k : ℕ, (posSupportStep G A₁)^[k] ∅ ⊆ (posSupportStep G A₂)^[k] ∅ := by
    intro k
    induction k with
    | zero => simp
    | succ k ih =>
      rw [Function.iterate_succ_apply', Function.iterate_succ_apply']
      exact (posSupportStep_mono_supported G A₁ ih).trans (posSupportStep_mono_out G h _)
  exact key n

/-- `posClosure` is a genuine fixpoint of `posSupportStep` after `n` steps (`card (Fin n) = n`). -/
theorem posClosure_fixed (A : Finset (Fin n)) :
    posSupportStep G A (posClosure G A) = posClosure G A := by
  have h := iterate_stabilizes (posSupportStep G A)
    (fun S T hST => posSupportStep_mono_supported G A hST) ∅ (Finset.empty_subset _)
  rw [Fintype.card_fin] at h
  exact h

/-- `posClosure` is the **least** fixpoint of `posSupportStep G A`. -/
theorem posClosure_least (A S : Finset (Fin n)) (hS : posSupportStep G A S ⊆ S) :
    posClosure G A ⊆ S := by
  unfold posClosure
  exact iterate_le_of_postfixed (posSupportStep G A)
    (fun S T hST => posSupportStep_mono_supported G A hST) ∅ S (Finset.empty_subset _) hS n

/-! ## `Γ` antimonotone, `Γ²` monotone, `T = lfp Γ²`, `T ∩ F = ∅` -/

theorem gammaOp_anti {I₁ I₂ : Finset (Fin n)} (h : I₁ ⊆ I₂) : gammaOp G I₂ ⊆ gammaOp G I₁ := by
  unfold gammaOp
  apply posClosure_mono
  intro x hx
  simp only [Finset.mem_sdiff, Finset.mem_univ, true_and] at hx ⊢
  exact fun hx1 => hx (h hx1)

theorem gammaSqOp_mono {I₁ I₂ : Finset (Fin n)} (h : I₁ ⊆ I₂) :
    gammaSqOp G I₁ ⊆ gammaSqOp G I₂ := by
  unfold gammaSqOp
  exact gammaOp_anti G (gammaOp_anti G h)

theorem gammaSqOp_fixed : gammaSqOp G (wfsTrueSet G) = wfsTrueSet G := by
  have h := iterate_stabilizes (gammaSqOp G) (fun S T hST => gammaSqOp_mono G hST) ∅
    (Finset.empty_subset _)
  rw [Fintype.card_fin] at h
  exact h

theorem wfsTrueSet_least (S : Finset (Fin n)) (hS : gammaSqOp G S = S) : wfsTrueSet G ⊆ S := by
  unfold wfsTrueSet
  exact iterate_le_of_postfixed (gammaSqOp G) (fun S T hST => gammaSqOp_mono G hST) ∅ S
    (Finset.empty_subset _) hS.le n

/-- Every iterate `U_k = (Γ²)^k ∅` is contained in `Γ U_k`. -/
theorem iterate_subset_gamma : ∀ k : ℕ, (gammaSqOp G)^[k] ∅ ⊆ gammaOp G ((gammaSqOp G)^[k] ∅) := by
  intro k
  induction k with
  | zero => simp
  | succ k ih =>
    have hstep : (gammaSqOp G)^[k + 1] ∅ = gammaOp G (gammaOp G ((gammaSqOp G)^[k] ∅)) := by
      rw [Function.iterate_succ_apply']
      rfl
    -- `U_{k+1} = Γ(Γ U_k) ⊆ Γ U_k` by antimonotonicity applied to `U_k ⊆ Γ U_k`
    have h1 : (gammaSqOp G)^[k + 1] ∅ ⊆ gammaOp G ((gammaSqOp G)^[k] ∅) := by
      rw [hstep]
      exact gammaOp_anti G ih
    -- antimonotonicity again turns `U_{k+1} ⊆ Γ U_k` into `Γ(Γ U_k) ⊆ Γ U_{k+1}`
    have h2 : gammaOp G (gammaOp G ((gammaSqOp G)^[k] ∅)) ⊆ gammaOp G ((gammaSqOp G)^[k + 1] ∅) :=
      gammaOp_anti G h1
    rw [← hstep] at h2
    exact h2

theorem wfsTrueSet_subset_gamma : wfsTrueSet G ⊆ gammaOp G (wfsTrueSet G) :=
  iterate_subset_gamma G n

theorem wfs_disjoint : Disjoint (wfsTrueSet G) (wfsFalseSet G) := by
  rw [Finset.disjoint_left]
  intro a ha hF
  unfold wfsFalseSet at hF
  simp only [Finset.mem_sdiff, Finset.mem_univ, true_and] at hF
  exact hF (wfsTrueSet_subset_gamma G ha)

/-! ## Cyclic `REFUTE` witness (equal-authority 2-ring) -/

/-- Two nodes in one domain with equal authority `1`, mutually refuting; no positive dependencies. -/
def G_cyc : CyclicEpistemicGraph 2 1 where
  domainOf := fun _ => 0
  effAuthForDomain := fun _ _ => 1
  effAuthSelf := fun _ => 1
  rawRefutes := fun u w => decide (u ≠ w)
  rawDeps := fun _ _ => false

theorem G_cyc_true : wfsTrueSet G_cyc = ∅ := by decide

theorem G_cyc_false : wfsFalseSet G_cyc = ∅ := by decide

theorem G_cyc_gamma_alt : gammaOp G_cyc ({0} : Finset (Fin 2)) = {0} := by decide

theorem G_cyc_witness :
    ∃ (G_cyc' : CyclicEpistemicGraph 2 1) (S_alt : Finset (Fin 2)),
      wfsTrueSet G_cyc' = ∅ ∧ wfsFalseSet G_cyc' = ∅ ∧ S_alt.Nonempty ∧ gammaOp G_cyc' S_alt = S_alt :=
  ⟨G_cyc, {0}, G_cyc_true, G_cyc_false, ⟨0, Finset.mem_singleton_self 0⟩, G_cyc_gamma_alt⟩

end Graph

/-! ## The main theorem -/

theorem theorem3_ek5_cyclic_wfs : Theorem3_EK5_CyclicWFS_Stmt := by
  refine ⟨?_, ?_, ?_, G_cyc_witness⟩
  · intro n doms G I₁ I₂ h
    exact gammaOp_anti G h
  · intro n doms G I₁ I₂ h
    exact gammaSqOp_mono G h
  · intro n doms G
    exact ⟨gammaSqOp_fixed G, fun S hS => wfsTrueSet_least G S hS, wfs_disjoint G⟩

/-! ## Computable evaluator bound to the spec (`ek5_ref_cli` Stratum-2 core) -/

section Eval

variable {n doms : ℕ} (G : CyclicEpistemicGraph n doms)

/-- Early-exit evaluation of `Γ_G(I)`. -/
def gammaFast (I : Finset (Fin n)) : Finset (Fin n) :=
  iterUntilFix (posSupportStep G (Finset.univ \ I)) n ∅

theorem gammaFast_eq : gammaFast G = gammaOp G := by
  funext I
  unfold gammaFast gammaOp posClosure
  exact iterUntilFix_eq _ _ _

/-- Computable Stratum-2 evaluator: `(T, F)` with `T = lfp(Γ²)` and `F = V \ Γ(T)`. -/
def evalStratum2Wfs : Finset (Fin n) × Finset (Fin n) :=
  let T := iterUntilFix (fun I => gammaFast G (gammaFast G I)) n ∅
  (T, Finset.univ \ gammaFast G T)

/-- **Soundness binding**: the executable evaluator equals the kernel-level `lfp(Γ²)` specification. -/
theorem evalStratum2Wfs_sound : evalStratum2Wfs G = (wfsTrueSet G, wfsFalseSet G) := by
  have hg := gammaFast_eq G
  have hT : iterUntilFix (fun I => gammaOp G (gammaOp G I)) n ∅ = wfsTrueSet G := by
    rw [iterUntilFix_eq]
    rfl
  unfold evalStratum2Wfs
  simp only [hg, hT]
  rfl

theorem evalStratum2Wfs_fst (G : CyclicEpistemicGraph n doms) :
    (evalStratum2Wfs G).1 = wfsTrueSet G := by
  rw [evalStratum2Wfs_sound]

theorem evalStratum2Wfs_snd (G : CyclicEpistemicGraph n doms) :
    (evalStratum2Wfs G).2 = wfsFalseSet G := by
  rw [evalStratum2Wfs_sound]

/-- Three-valued epistemic validity (`T`/`F`/`U`) of one node from `(T, F)`. -/
inductive Tri where
  | t | f | u
  deriving DecidableEq, Repr

def triOf (T F : Finset (Fin n)) (x : Fin n) : Tri :=
  if x ∈ T then Tri.t else if x ∈ F then Tri.f else Tri.u

theorem triOf_eval_sound (x : Fin n) :
    triOf (evalStratum2Wfs G).1 (evalStratum2Wfs G).2 x = triOf (wfsTrueSet G) (wfsFalseSet G) x := by
  rw [evalStratum2Wfs_sound]

theorem triOf_exclusive (x : Fin n) :
    (triOf (wfsTrueSet G) (wfsFalseSet G) x = Tri.t → x ∈ wfsTrueSet G ∧ x ∉ wfsFalseSet G) := by
  intro h
  have hT : x ∈ wfsTrueSet G := by
    unfold triOf at h
    by_contra hx
    simp only [hx, ite_false] at h
    split_ifs at h
  refine ⟨hT, ?_⟩
  intro hF
  exact (Finset.disjoint_left.mp (wfs_disjoint G)) hT hF

end Eval

/-! ## Stratum 1 → Stratum 2 non-interference and target-domain qualification -/

section Stratum1

variable {n doms : ℕ}

/-- Positive `D_V` cycle through `u` (`ek5_v3_stratified.py:_nontrivial_scc_nodes`). -/
def OnPosCycle (deps : Fin n → Fin n → Bool) (u : Fin n) : Prop :=
  Relation.TransGen (fun a b => deps a b = true) u u

open Classical in
/-- Base ceiling of one node: `0` if missing a mandatory parent or on a positive `D_V` cycle,
    else its raw tier. -/
noncomputable def baseAuth (tier : Fin n → ℕ) (deps : Fin n → Fin n → Bool)
    (missing : Fin n → Bool) (v : Fin n) : ℕ :=
  if missing v = true ∨ OnPosCycle deps v then 0 else tier v

/-- Anti-laundering ceiling `alpha_tilde(u) = min` of `baseAuth` over `u` and everything reachable from
    `u` along `D_V` (`ek5_v3_stratified.py:_recompute_authority_cone`).  A function of
    `(tier, deps, missing)` **only** — no `rawRefutes` / Stratum-2 value is an argument. -/
noncomputable def effAuthD (tier : Fin n → ℕ) (deps : Fin n → Fin n → Bool)
    (missing : Fin n → Bool) (u : Fin n) : ℕ :=
  sInf {k | ∃ v, (v = u ∨ Relation.TransGen (fun a b => deps a b = true) u v) ∧
    k = baseAuth tier deps missing v}

private theorem effAuthD_mem (tier : Fin n → ℕ) (deps : Fin n → Fin n → Bool)
    (missing : Fin n → Bool) (u : Fin n) :
    effAuthD tier deps missing u ∈ {k | ∃ v, (v = u ∨ Relation.TransGen (fun a b => deps a b = true) u v) ∧
      k = baseAuth tier deps missing v} := by
  apply Nat.sInf_mem
  exact ⟨baseAuth tier deps missing u, u, Or.inl rfl, rfl⟩

theorem effAuthD_le_tier (tier : Fin n → ℕ) (deps : Fin n → Fin n → Bool)
    (missing : Fin n → Bool) (u : Fin n) : effAuthD tier deps missing u ≤ tier u := by
  have h1 : effAuthD tier deps missing u ≤ baseAuth tier deps missing u :=
    Nat.sInf_le ⟨u, Or.inl rfl, rfl⟩
  have h2 : baseAuth tier deps missing u ≤ tier u := by
    unfold baseAuth
    split_ifs
    · exact Nat.zero_le _
    · exact le_refl _
  exact h1.trans h2

theorem effAuthD_missing (tier : Fin n → ℕ) (deps : Fin n → Fin n → Bool)
    (missing : Fin n → Bool) (u : Fin n) (hm : missing u = true) :
    effAuthD tier deps missing u = 0 := by
  have h1 : effAuthD tier deps missing u ≤ baseAuth tier deps missing u :=
    Nat.sInf_le ⟨u, Or.inl rfl, rfl⟩
  have h2 : baseAuth tier deps missing u = 0 := by
    unfold baseAuth
    rw [if_pos (Or.inl hm)]
  omega

theorem effAuthD_cycle (tier : Fin n → ℕ) (deps : Fin n → Fin n → Bool)
    (missing : Fin n → Bool) (u : Fin n) (hc : OnPosCycle deps u) :
    effAuthD tier deps missing u = 0 := by
  have h1 : effAuthD tier deps missing u ≤ baseAuth tier deps missing u :=
    Nat.sInf_le ⟨u, Or.inl rfl, rfl⟩
  have h2 : baseAuth tier deps missing u = 0 := by
    unfold baseAuth
    rw [if_pos (Or.inr hc)]
  omega

/-- Anti-laundering: a dependent never exceeds the ceiling of any of its `D_V` parents. -/
theorem effAuthD_le_parent (tier : Fin n → ℕ) (deps : Fin n → Fin n → Bool)
    (missing : Fin n → Bool) (w d : Fin n) (hd : deps w d = true) :
    effAuthD tier deps missing w ≤ effAuthD tier deps missing d := by
  obtain ⟨v, hv, hk⟩ := effAuthD_mem tier deps missing d
  apply Nat.sInf_le
  refine ⟨v, ?_, hk⟩
  rcases hv with rfl | hv
  · exact Or.inr (Relation.TransGen.single hd)
  · exact Or.inr (Relation.TransGen.head hd hv)

/-- Target-domain scoped capability (`_effective_authority_for_domain`): a node whose domain differs
    from the target domain, with neither being `general`, has authority `0` in that domain. -/
def scopedAuth (general : Fin doms) (domainOf : Fin n → Fin doms) (eff : Fin n → ℕ)
    (u : Fin n) (target : Fin doms) : ℕ :=
  if domainOf u ≠ general ∧ target ≠ general ∧ domainOf u ≠ target then 0 else eff u

/-- Graph assembled from Stratum-0/1 inputs plus the (Stratum-2) raw refutation relation. -/
noncomputable def mkGraph (general : Fin doms) (domainOf : Fin n → Fin doms) (tier : Fin n → ℕ)
    (deps : Fin n → Fin n → Bool) (missing : Fin n → Bool) (refutes : Fin n → Fin n → Bool) :
    CyclicEpistemicGraph n doms where
  domainOf := domainOf
  effAuthForDomain := scopedAuth general domainOf (effAuthD tier deps missing)
  effAuthSelf := effAuthD tier deps missing
  rawRefutes := refutes
  rawDeps := deps

/-- **Stratum 1 → Stratum 2 non-interference**: any change of `rawRefutes` (Stratum-2 input, hence of
    every Stratum-2 truth value) leaves the Stratum-1 authority assignment — both `eff_auth(w)` and the
    target-domain `eff_auth_for_domain(u, ·)` — and the `D_V` relation untouched. -/
theorem stratum1_noninterference (general : Fin doms) (domainOf : Fin n → Fin doms)
    (tier : Fin n → ℕ) (deps : Fin n → Fin n → Bool) (missing : Fin n → Bool)
    (r₁ r₂ : Fin n → Fin n → Bool) :
    (mkGraph general domainOf tier deps missing r₁).effAuthSelf =
        (mkGraph general domainOf tier deps missing r₂).effAuthSelf ∧
      (mkGraph general domainOf tier deps missing r₁).effAuthForDomain =
        (mkGraph general domainOf tier deps missing r₂).effAuthForDomain ∧
      (mkGraph general domainOf tier deps missing r₁).rawDeps =
        (mkGraph general domainOf tier deps missing r₂).rawDeps :=
  ⟨rfl, rfl, rfl⟩

/-- `isActiveRefute` (computable) coincides with the spec predicate `activeRefute`. -/
theorem isActiveRefute_iff (G : CyclicEpistemicGraph n doms) (u w : Fin n) :
    isActiveRefute G u w = true ↔ activeRefute G u w := by
  unfold isActiveRefute activeRefute
  simp [Bool.and_eq_true]

/-- Target-domain qualification (`ek5:1180`): for the assembled graph an edge `u → w` is active iff it is
    raw and `eff_auth_for_domain(u, domain(w)) ≥ eff_auth(w)`. -/
theorem mkGraph_activeRefute_iff (general : Fin doms) (domainOf : Fin n → Fin doms)
    (tier : Fin n → ℕ) (deps : Fin n → Fin n → Bool) (missing : Fin n → Bool)
    (refutes : Fin n → Fin n → Bool) (u w : Fin n) :
    activeRefute (mkGraph general domainOf tier deps missing refutes) u w ↔
      refutes u w = true ∧
        effAuthD tier deps missing w ≤
          scopedAuth general domainOf (effAuthD tier deps missing) u (domainOf w) :=
  Iff.rfl

/-- Cross-domain refutation of a node with positive authority is blocked. -/
theorem mkGraph_cross_domain_blocked (general : Fin doms) (domainOf : Fin n → Fin doms)
    (tier : Fin n → ℕ) (deps : Fin n → Fin n → Bool) (missing : Fin n → Bool)
    (refutes : Fin n → Fin n → Bool) (u w : Fin n)
    (hu : domainOf u ≠ general) (hw : domainOf w ≠ general) (hne : domainOf u ≠ domainOf w)
    (hpos : 0 < effAuthD tier deps missing w) :
    ¬ activeRefute (mkGraph general domainOf tier deps missing refutes) u w := by
  intro h
  have h2 := h.2
  change effAuthD tier deps missing w ≤ scopedAuth general domainOf (effAuthD tier deps missing) u (domainOf w) at h2
  unfold scopedAuth at h2
  rw [if_pos ⟨hu, hw, hne⟩] at h2
  omega

end Stratum1

/-! ## Non-vacuity witnesses -/

/-- 4-node witness with all three truth values: `0` unrefuted (`T`), `1` refuted by `0` (`F`),
    `2 ↔ 3` equal-authority mutual refutation (`U`). -/
def G_three_valued : CyclicEpistemicGraph 4 1 where
  domainOf := fun _ => 0
  effAuthForDomain := fun _ _ => 1
  effAuthSelf := fun _ => 1
  rawRefutes := fun u w => decide ((u = 0 ∧ w = 1) ∨ (u = 2 ∧ w = 3) ∨ (u = 3 ∧ w = 2))
  rawDeps := fun _ _ => false

theorem G_three_valued_true : wfsTrueSet G_three_valued = {0} := by decide

theorem G_three_valued_false : wfsFalseSet G_three_valued = {1} := by decide

theorem theorem3_witness_nonvacuous :
    (∃ G : CyclicEpistemicGraph 4 1,
        wfsTrueSet G = {0} ∧ wfsFalseSet G = {1} ∧
        (2 : Fin 4) ∉ wfsTrueSet G ∧ (2 : Fin 4) ∉ wfsFalseSet G ∧
        (3 : Fin 4) ∉ wfsTrueSet G ∧ (3 : Fin 4) ∉ wfsFalseSet G) ∧
    (∃ (G : CyclicEpistemicGraph 2 1) (S_alt : Finset (Fin 2)),
        wfsTrueSet G = ∅ ∧ wfsFalseSet G = ∅ ∧ S_alt.Nonempty ∧ gammaOp G S_alt = S_alt ∧
        S_alt ≠ wfsTrueSet G) := by
  refine ⟨⟨G_three_valued, G_three_valued_true, G_three_valued_false, ?_, ?_, ?_, ?_⟩,
    ⟨G_cyc, {0}, G_cyc_true, G_cyc_false, ⟨0, Finset.mem_singleton_self 0⟩, G_cyc_gamma_alt, ?_⟩⟩
  · rw [G_three_valued_true]; decide
  · rw [G_three_valued_false]; decide
  · rw [G_three_valued_true]; decide
  · rw [G_three_valued_false]; decide
  · rw [G_cyc_true]; decide

end ChelatedFormal
