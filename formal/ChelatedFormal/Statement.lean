import ChelatedFormal.Spec
import ChelatedFormal.Theorem1a_MonotoneIsometry
import ChelatedFormal.Theorem1b_ProcrustesOptimality
import ChelatedFormal.Theorem2_AdjointZeroMuPerturbation
import ChelatedFormal.Theorem3_EK5_CyclicWFS
import ChelatedFormal.Theorem4_AnchoredClippedEMA
import ChelatedFormal.Theorem5_IncrementalWFS_CW1

/-!
# Master Wave-14A Formal Verification Bundle (`ChelatedFormal.Statement`)

Aggregates all five kernel-checked proof modules against the SHA-256-locked specification
`ChelatedFormal.Spec` (`formal/spec_sha256.json`) with **0 `sorry`, 0 `native_decide`, and 0 custom axioms**:

1. `Theorem1a_MonotoneIsometry`: Stiefel dimension monotonicity (`d₀ ≤ d₁`), conformal chain isometry,
   and `2 → 3 → 2` non-monotone rank-collapse counterexample.
2. `Theorem1b_ProcrustesOptimality`: Thin-SVD trace alignment optimality, unconditional upward-lifting
   Frobenius optimality, isotropic downward Frobenius corollary, and exact rational `3-4-5` (`137/25 > 5`)
   downward counterexample (in-sample plain SVD branch).
3. `Theorem2_AdjointZeroMuPerturbation`: Adjoint query-read duality, projector norm identity, unit-query
   `√(1 - η)` specialization, orthogonal-query fixture, and `ZeroMu + CSLS` forward perturbation bound.
4. `Theorem3_EK5_CyclicWFS`: Stratum 1–2 cyclic alternating-fixpoint Well-Founded Semantics (`Γ_P`
   antimonotone, `Γ_P²` monotone, `T = lfp(Γ_P²)`, `T ∩ F = ∅`, mutual `REFUTE` undecided ring witness,
   Stratum 1 → Stratum 2 non-interference, and `evalStratum2Wfs_sound` binding `ek5_ref_cli`).
   Strata 0, 1 and 3 are conformance-tested only (see "Empirical evidence" below), not kernel-bound.
5. `Theorem4_AnchoredClippedEMA`: Inductive `ρ`-ball drift invariance for `Anchored-Clipped EMA` (`M6`)
   plus `2ρ/γ` chordal and `1 - 2(ρ/γ)²` cosine stability bounds.

## What is inside the SHA-locked `Spec.lean` umbrella, and what is not

`all_wave14a_formal_theorems_verified` below is the conjunction of the ten `*_Stmt` propositions declared in
`ChelatedFormal.Spec` (`formal/spec_sha256.json`). Those ten statements are the SHA-256-locked claim.

The following are **companion proved lemmas OUTSIDE the locked umbrella**. They are kernel-checked and exported to
`nanoda_bin`, but they are not conjuncts of `all_wave14a_formal_theorems_verified` and are not named by `Spec.lean`:

* `ConformalChain.norm_mulVec` (`Theorem1a_MonotoneIsometry`): the per-step norm identity used by the chain isometry.
* `theorem4_chordal_cosine_drift` (`Theorem4_AnchoredClippedEMA`): the `2ρ/γ` chordal / cosine drift bounds.
* `evalStratum2Wfs_sound` (`Theorem3_EK5_CyclicWFS`): binds `evalStratum2Wfs` (the Stratum-2 core of `ek5_ref_cli`)
  to `(wfsTrueSet, wfsFalseSet)`.
* `wfs_frame_isolation`, `incrementalWfsSplice_sound`, `unfounded_pos_cycle_false`, `delete_refute_gamma_mono`,
  `delete_dep_posClosure_mono`, `theorem5_witness_nonvacuous` (`Theorem5_IncrementalWFS_CW1`): proves `CW-1`
  affected-cone frame isolation, incremental splice soundness under arbitrary deletions/mutations inside `A`,
  unfounded positive-cycle rejection, and edge-deletion monotonicity on `CyclicEpistemicGraph`.

## CSLS scale convention (2x)

`Spec.lean` bounds the CSLS-corrected score `cos - 0.5 * (r_q + r_d)`. The runtime scores
`2 * cos - r_q - r_d`, which is exactly `2 *` the Spec quantity, so the runtime value is twice the Spec value
(equivalently, the Spec value is `0.5 *` the runtime value). Any bound transferred from Spec to runtime scores must
be scaled by 2.

## Empirical evidence (conformance, NOT kernel-checked theorems)

The `1,000` multi-mutation sequences and `7,722` checkpoints exercised by `formal/run_conformance.py` are empirical
conformance evidence about the Python store `ek5_v3_stratified.StratifiedEvidenceStore`, `ek5_ref_cli`, and an
alternating-fixpoint oracle. They are not theorems and nothing in this file proves them.

* Stratum 2 (WFS, positive cycles and mutual refutation): kernel-bound by `evalStratum2Wfs_sound`.
* Strata 0, 1 and 3 (materialization, authority ceiling, temporal currency, eligibility): conformance-tested, unproven.
* incremental-with-deletions: conformance-tested across 1,000 deletion/cycle sequences, unproven.
-/

namespace ChelatedFormal

theorem all_wave14a_formal_theorems_verified :
    Theorem1a_MonotoneChainIsometry_Stmt ∧
    Theorem1a_NonMonotoneCollapse_Stmt ∧
    Theorem1b_TraceOptimality_Stmt ∧
    Theorem1b_UpwardLiftingFrobenius_Stmt ∧
    Theorem1b_IsotropicDownwardFrobenius_Stmt ∧
    Theorem1b_DownwardFrobeniusCounterexample_Stmt ∧
    Theorem2_AdjointAndProjector_Stmt ∧
    Theorem2_ZeroMuAndPerturbation_Stmt ∧
    Theorem3_EK5_CyclicWFS_Stmt ∧
    Theorem4_AnchoredClippedEMA_Stmt :=
  ⟨theorem1a_monotone_chain_isometry,
   theorem1a_non_monotone_collapse,
   theorem1b_trace_optimality,
   theorem1b_upward_lifting_frobenius,
   theorem1b_isotropic_downward_frobenius,
   theorem1b_downward_frobenius_counterexample,
   theorem2_adjoint_and_projector,
   theorem2_zeromu_and_perturbation,
   theorem3_ek5_cyclic_wfs,
   theorem4_anchored_clipped_ema⟩

end ChelatedFormal
