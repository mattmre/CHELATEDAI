# Training-time architecture hypothesis — slot-warp codebook + anisotropic σ-field + slot-subspace invariance

**Status: DRAFT for operator review — a formalization of the 2026-06-29 sketches into ONE checkable
hypothesis + a cheap toy test. Not a build commitment. Local-only (git-excluded).**

This turns three operator sketches — (1) σ-signaling, (2) "annealing is 1-D; want directional/
overlapping stacked gates during training," (3) "dense replaceable subspace neighbors / fold-warp
next tokens in a perturbation manifold" — into one concrete, novelty-checkable, prototypable design.
Every component is named so we can (a) novelty-check it precisely and (b) ablate it at toy scale
BEFORE any expensive custom-model training.

## The construct (one concrete reading)

Base encoder `f` stays **frozen** (keeps the guardrail; the new structure is a trained *head*).

1. **Slot codebook** `C = {S_1,…,S_K}` — K learned subspace WARPS, each low-rank:
   `S_k(z) = z + U_k V_kᵀ z`, `U_k,V_k ∈ R^{d×r}`, r≪d. (The "dense replaceable subspace neighbors";
   slots are swappable codebook entries.)
2. **Overlapping directional gate** `g(z) ∈ Δ^{K}` (soft, NOT top-1 — a token routes to several
   slots by direction): warped embedding `ẑ = Σ_k g_k(z)·S_k(z)`. ("directional overlapping stacked
   gates.")
3. **Anisotropic σ-field annealing** (the "not 1-D annealing"): each slot carries a per-direction
   noise vector `σ_k ∈ R^r`; during training the warp is perturbed `U_k → U_k + diag(σ_k)·ε`,
   `ε~N(0,I)`, and `σ_k` is annealed high→low over training (explore slot directions early, commit
   late). Generalizes scalar σ to a per-slot, per-direction σ-field.
4. **Slot-subspace invariance objective** (the "fold/warp next tokens in a perturbation manifold /
   localized invariance near the posts"):
   `L_inv = E_{x, δ∈span(slots)} [ d( out(ẑ(x)), out(ẑ(x)+δ) ) ]`
   — train the OUTPUT (retrieval similarity / logits) to be locally invariant to perturbations
   *along the slot subspaces*, i.e. the directions drift is expected to move.
5. **Replaceable value-position stacking** = drift adaptation by SWAP/ADD a slot (no base retrain):
   when drift is detected, fit/insert a slot `S_{K+1}` for the affected region; the gate routes to it.

**Total training loss:** `L_task + λ_inv·L_inv (+ codebook/usage regularizers)`, with σ-field annealed.

## The hypothesis (what it claims, and why it might beat the post-hoc shim)

> A representation trained with (slot-warp codebook + anisotropic-σ annealing + slot-subspace
> invariance) absorbs drift by *swapping a slot* into a subspace the model was trained to be robust
> in — so post-drift recovery via slot-swap **beats** post-hoc shimming on a frozen-but-not-
> invariance-trained base, AND the annealed σ-field gives a genuine explore→commit dynamic the
> deterministic post-bank lacked (which is WHY C5==C5s in the smokes).

Falsifiable: if slot-swap recovery ≤ post-hoc shim, OR the σ-field/invariance ablations don't move
the result, the hypothesis fails.

## Novelty positioning (each part is known; the COMBINATION is the only possible claim)

- vs **MoE**: slots are subspace-WARP codebook entries for representation *correction*, not FFN
  experts for compute; soft directional routing.
- vs **VQ-VAE codebooks / slot attention**: entries are *transforms* (warps), annealed by an
  *anisotropic σ-field*, with a *slot-subspace invariance* objective + *swap-for-drift*.
- vs **anisotropic diffusion noise schedules**: σ-field anneals a slot-transform codebook for
  retrieval drift, not latent denoising.
- vs **perturbation-robust training (Vaccine, 2402.01109; manifold reg)**: invariance is to the
  *modeled drift directions* (slot subspaces), tied to the swap-for-drift mechanism.
- vs **Drift-Adapter (2509.23471)**: that's post-hoc, frozen base, new→old direction; this bakes
  drift-robustness into training and adapts by slot-swap. Different axis.
- **Honest:** must be novelty-checked AGAIN once this exact construct exists; right now it's a
  plausible-but-unverified novel *combination* of well-known parts.

## Cheap toy test (GPU-light; do BEFORE any custom-base training)

Train only the HEAD (slots+gate+σ+invariance) on a frozen small encoder; synthetic or SciFact-small.
Conditions to compare on post-drift retrieval recovery:
- **B0** frozen base (floor) · **B1** post-hoc shim (our C3a) · **T0** slot head, swap-for-drift,
  NO σ-field / NO invariance (deterministic ablation) · **T1** + anisotropic σ-field · **T2** +
  slot-subspace invariance (full).
Claim holds iff **T2 > B1 > B0** AND **T2 > T1 > T0** (each added piece contributes). This isolates
whether σ-field and invariance actually matter, at a few-GPU-minutes cost — no custom base model,
no overnight run. Only if T2 wins decisively does the full custom-architecture path earn its cost.

## Recommended order (unchanged)

1. Full-scale **H5 post-hoc campaign** first (does the cheap path even have a ceiling worth leaving?).
2. This **toy ablation** (T0/T1/T2) to test the training-time hypothesis cheaply.
3. Only then, if T2 wins, scope the full custom-architecture build (the expensive path).
