import ChelatedFormal.Spec

namespace ChelatedFormal

open scoped InnerProductSpace

theorem radialClip_norm_sub_le {d : ℕ} (μ₀ x : Euc d) {ρ : ℝ} (hρ : 0 ≤ ρ) :
    ‖radialClip μ₀ x ρ - μ₀‖ ≤ ρ := by
  unfold radialClip
  split_ifs with h
  · exact h
  · have hPos : 0 < ‖x - μ₀‖ := lt_of_le_of_lt hρ (lt_of_not_ge h)
    rw [add_sub_cancel_left, norm_smul, Real.norm_eq_abs,
      abs_of_nonneg (div_nonneg hρ hPos.le), div_mul_cancel₀ _ hPos.ne']

theorem theorem4_anchored_clipped_ema : Theorem4_AnchoredClippedEMA_Stmt := by
  intro d μ₀ ρ α lam hρ k q
  induction k with
  | zero => simpa
  | succ k _ => exact radialClip_norm_sub_le _ _ hρ

/-- The direction of a centered query of norm at least `γ` changes by at most
`2 * ρ / γ` in chordal distance. Its cosine with the anchored direction is
at least `1 - 2 * (ρ / γ)^2`. -/
theorem theorem4_chordal_cosine_drift {d : ℕ} {μ₀ μ q : Euc d} {ρ γ : ℝ}
    (hρ : 0 ≤ ρ) (hγ : ρ < γ) (hq : γ ≤ ‖q - μ₀‖)
    (hμ : ‖μ - μ₀‖ ≤ ρ) :
    let u := q - μ₀
    let v := q - μ
    let a := ‖u‖⁻¹ • u
    let b := ‖v‖⁻¹ • v
    0 < ‖v‖ ∧ ‖a - b‖ ≤ 2 * ρ / γ ∧
      1 - 2 * (ρ / γ) ^ 2 ≤ ⟪a, b⟫_ℝ := by
  dsimp only
  let u := q - μ₀
  let v := q - μ
  let a := ‖u‖⁻¹ • u
  let b := ‖v‖⁻¹ • v
  change 0 < ‖v‖ ∧ ‖a - b‖ ≤ 2 * ρ / γ ∧ 1 - 2 * (ρ / γ) ^ 2 ≤ ⟪a, b⟫_ℝ
  have hγpos := lt_of_le_of_lt hρ hγ
  have hu : 0 < ‖u‖ := hγpos.trans_le hq
  have huv : ‖u - v‖ ≤ ρ := by
    have heq : u - v = μ - μ₀ := by dsimp [u, v]; abel
    rwa [heq]
  have hnorm : |‖u‖ - ‖v‖| ≤ ρ := (abs_norm_sub_norm_le u v).trans huv
  have hv : 0 < ‖v‖ := by
    have := (le_abs_self (‖u‖ - ‖v‖)).trans hnorm
    linarith
  have haUnit : ‖a‖ = 1 := by
    simp [a, norm_smul, hu.ne']
  have hbUnit : ‖b‖ = 1 := by
    simp [b, norm_smul, hv.ne']
  have hScaled : ‖v‖ • b = v := by
    rw [show b = ‖v‖⁻¹ • v from rfl, smul_smul, mul_inv_cancel₀ hv.ne', one_smul]
  have hDecomp : a - b = ‖u‖⁻¹ • ((u - v) - (‖u‖ - ‖v‖) • b) := by
    rw [sub_smul, hScaled]
    have hInner : u - v - (‖u‖ • b - v) = u - ‖u‖ • b := by abel
    rw [hInner, smul_sub, smul_smul, inv_mul_cancel₀ hu.ne', one_smul]
  have hChord : ‖a - b‖ ≤ 2 * ρ / γ := calc
    ‖a - b‖ = ‖(u - v) - (‖u‖ - ‖v‖) • b‖ / ‖u‖ := by
      rw [hDecomp, norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hu), div_eq_inv_mul]
    _ ≤ (‖u - v‖ + |‖u‖ - ‖v‖|) / ‖u‖ := by
      apply div_le_div_of_nonneg_right _ hu.le
      simpa [norm_smul, Real.norm_eq_abs, hbUnit] using
        norm_sub_le (u - v) ((‖u‖ - ‖v‖) • b)
    _ ≤ (2 * ρ) / ‖u‖ := div_le_div_of_nonneg_right (by linarith) hu.le
    _ ≤ (2 * ρ) / γ := div_le_div_of_nonneg_left (by positivity) hγpos hq
  refine ⟨hv, hChord, ?_⟩
  have hSquared := (sq_le_sq₀ (norm_nonneg (a - b)) (by positivity)).2 hChord
  rw [norm_sub_sq_real, haUnit, hbUnit] at hSquared
  rw [mul_div_assoc] at hSquared
  nlinarith

theorem theorem4_witness_nonvacuous :
    ∃ (μ₀ : Euc 1) (q : ℕ → Euc 1) (ρ α lam : ℝ),
      0 < ρ ∧ 0 < α ∧ 0 ≤ lam ∧ ρ < ‖q 0 - μ₀‖ ∧
      ‖radialClip μ₀ ((1 - α - lam) • μ₀ + α • q 0 + lam • μ₀) ρ - μ₀‖ = ρ := by
  let x : Euc 1 := EuclideanSpace.single 0 (2 : ℝ)
  have hx : ‖x‖ = 2 := by simp [x]
  refine ⟨0, fun _ => x, 1, 1, 0, by norm_num, by norm_num, le_rfl, ?_, ?_⟩
  · simp [hx]
  · norm_num [radialClip, hx, norm_smul, Real.norm_eq_abs]

end ChelatedFormal
