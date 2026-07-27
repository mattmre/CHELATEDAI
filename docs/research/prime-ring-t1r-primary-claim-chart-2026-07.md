# PRW-T1R primary-source claim chart

**Review date:** 2026-07-25; reconditioned 2026-07-26

**Review status:** `BOUNDED_PRIMARY_SOURCE_CHART_RECONDITIONED`

**Novelty status:** `UNRESOLVED_SPECIALIST_REVIEW_REQUIRED`

**Evidence status:** literature-positioning artifact, not experimental evidence

## 1. Claim being positioned

For primes \(p\equiv3\pmod4\), the frozen two-type, eight-layer
Legendre-by-\(RM(1,3)\) bank contains \(32p\) distinct binary words of length
\(8p\). For fixed \(q\in(0,1/2)\), its wrong-type tie-or-better competitor-event
union has the reconditioned leading term

\[
B_p(q)=56\beta_q(d_p)+8\beta_q(d_p+4),
\qquad
d_p=\frac{7p-1}{2},
\]

with the remaining five distance shells and all pairwise intersections
asymptotically negligible relative to \(B_p(q)\). This is the narrow
`PRW-T1R-TE-EVENT-UNION` claim. Decoder-specific strict/inclusive tie semantics
are a separate corollary and must use the current executable proof boundary.

The claim is not a general coding theorem, a new error exponent, a new
Reed--Muller or quadratic-residue code, or evidence of retrieval utility.

## 2. Bounded search method

The primary-source search covered four collision neighborhoods:

1. BSC maximum-likelihood error probability from distance distributions;
2. asymptotic accuracy and failure modes of union bounds;
3. decoder-tie effects on BSC error exponents; and
4. Legendre-sequence pattern/weight distributions and quadratic-residue codes.

Searches used title/abstract and available full-text surfaces for the exact
terms above plus combinations of `Legendre`, `Reed-Muller`, `distance
spectrum`, `Bonferroni`, `BSC`, and `asymptotically tight union bound`.
This was a bounded claim chart, not a systematic review of every coding-theory
index, citation graph, book, thesis, or non-English source.

## 3. Claim chart

| Primary source | Established overlap | Difference from the frozen PRW-T1R claim | Collision judgment |
|---|---|---|---|
| [Barg, *On the asymptotic accuracy of the union bound* (2004/2005)](https://arxiv.org/abs/cs/0412111) | Directly studies when a union bound captures BSC maximum-likelihood error asymptotics. | Its result concerns rate regions and general/random-code reliability behavior, not the seven exact shells, five overlap orbits, or relative-\(1+o(1)\) event union of this frozen vanishing-rate family. | Broad method collision; not an exact theorem collision found. |
| [Barg and McGregor, *Distance distribution of binary codes and the error probability of decoding* (2004/2005)](https://arxiv.org/abs/cs/0407011) | Uses known binary-code distance distributions to lower-bound BSC maximum-likelihood error and reliability. | Does not supply this bank's distance/overlap enumerator or the stated 64-event leading coefficient. | Strong neighborhood collision; no exact construction collision found. |
| [Chang, Chen, Alajaji, and Han, *Decoder Ties Do Not Affect the Error Exponent of the Memoryless Binary Symmetric Channel* (2022)](https://mast.queensu.ca/~fady/Journal/it22-bsc-exp-dec-ties.pdf) | Proves for arbitrary sequences of distinct binary codes that tie handling does not change the BSC error exponent beyond a subexponential factor. | PRW-T1R concerns a relative leading coefficient and explicit strict-versus-inclusive tie constants, which exponent equality does not determine. | Decisive collision for any broad “ties change the exponent” claim; not for the exact relative asymptotic. |
| [Pan and Mow, *Asymptotically Tight MLD Bounds and Minimum-Variance Importance Sampling Estimator for Linear Block Codes Over BSCs* (2022)](https://hdl.handle.net/1783.1/118506) | Develops asymptotically tight BSC maximum-likelihood bounds and shows conventional union-bound machinery is not universally asymptotically tight. | Uses a different enumerating function and general linear-code framework; it does not derive this nonlinear frozen bank's shell and pair-orbit certificate. | Strong caution/control boundary; no exact theorem collision found. |
| [Ding, *Pattern Distributions of Legendre Sequences* (1998)](https://doi.org/10.1109/18.681353) | Establishes Legendre-sequence pattern properties and connects shifted Legendre sequences with quadratic-residue-code weight distributions. | Does not analyze the two phase signatures, eight-layer stacking, \(RM(1,3)\) polarity masks, seven wrong-type shells, or BSC event-union asymptotic used here. | Decisive ingredient collision; no exact stacked-bank collision found. |
| [Nguyen Q. A., Györfi, and Massey, *Constructions of Binary Constant-Weight Cyclic Codes and Cyclically Permutable Codes* (1992)](https://doi.org/10.1109/18.135636) | Represents each `GF(p)` symbol by a cyclic shift of a binary `p`-tuple, proves the binary distance is the outer-code Hamming distance times the inner representation distance when the representation is equidistant, and gives Legendre sequences with inner distance \((p+1)/2\) for the relevant primes. | It directly contains the unmasked PRW distance identity but does not state the frozen `RM(1,3)` masked seven-shell/five-overlap BSC event-union asymptotic. | Direct construction and distance-formula collision; exact event-union theorem equivalence remains unresolved. |

## 4. Positioning verdict

The claim chart rules out novelty language for the following ingredients:

- Legendre-sequence correlation or pattern structure;
- Reed--Muller masks and quadratic-residue-code relationships;
- distance/weight-spectrum analysis;
- pairwise union bounds, Bonferroni corrections, and BSC tail asymptotics; and
- the broad statement that decoder ties do not alter the BSC error exponent.

It also rules out presenting the Legendre-inner/outer-symbol construction or
the unmasked distance identity
`(outer Hamming distance) * (p + 1) / 2` as new. The 1992 construction is a
direct prior-art collision, not merely an adjacent ingredient.

The bounded search did not locate a primary source stating the exact
seven-shell/five-overlap enumeration and 64-event relative asymptotic for this
specific Legendre-by-\(RM(1,3)\) bank. That absence is a residual search gap,
not novelty evidence. A defensible description is:

> a narrow, construction-specific distance and event-overlap certificate
> assembled from established coding-theory ingredients.

Any publication claim still requires an information-theory specialist to
audit the proof, follow backward and forward citations, search additional
indexes, and decide whether the exact construction-specific statement is
already implicit in a broader theorem.

## 5. Queue consequence

The method-development claim chart is reconditioned and complete only for its
declared bounded search set. It does not promote `PRW-T1R`, establish novelty,
or authorize application claims. Publication-level positioning still requires
an independent proof/construction-equivalence audit, backward and forward
citation traversal from the 1992 construction, broader coding-theory index
coverage, and specialist review.

The frozen exact-Hamming all-state analytic tie corollary is complete
separately. The current float-FFT production scorer failed exact boundary-tie
parity in the 2026-07-26 audit, so no production-decoder claim may inherit the
analytic result.
