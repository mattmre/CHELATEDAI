# ResonanceDB paper: overlap with RHPC and required ordinary control

**Review date:** 2026-10-08. **Disposition:** direct representation/retrieval prior art; no copying, priority, semantic-utility, or RHPC acceptance claim.

## Sources and provenance

Aleksandr Listopad, [Wave-Based Semantic Memory with Resonance-Based Retrieval](https://arxiv.org/abs/2509.09691), arXiv v1, submitted **2025-08-21**. The full nine-page [v1 PDF](https://arxiv.org/pdf/2509.09691v1), including equations (2)/(3)/(5), experiments, limitations, and appendix, was reviewed. Retrieved PDF SHA-256: `5730ec0c2772c44675578d9616b17a2e0a49727803bc4075b522fbf4dc0cfc27`.

The [upstream repository](https://github.com/LexProfi/ResonanceDB/tree/3ee313c2d8311922a6428fcd398e8f82c23643c8) was inspected at commit `3ee313c2d8311922a6428fcd398e8f82c23643c8` (committer date 2026-09-04). Targeted source reading covered `JavaKernel.java`, `UnfoldedMath.java`, and their property/contract tests. No upstream code was copied, compiled, or executed; current source is distinct evidence from the 2025 paper's experiments.

## Closest project lane

The user's pictured field note, **Resonant Holographic Path Compression**, maps to **P4-REPRESENTATION / RHPC**, with [IQ-03 learned holographic composition](innovation-test-queue-2026-09-12/03-learned-holographic-composition.md) as its future learned experiment. Shared phase vocabulary is insufficient to identify the full mechanisms:

The [portable Stage-A protocol reference](references/rhpc-stage-a-method-dev-protocol-2026-08-25.md) preserves the inspected source bytes (SHA-256 `fa556285107fd74a202bfbb7140b24c4d29c0bc171a02615068e4ef8834b88d7`). It supplies the construction and acceptance contract for review; it is not a runnable RHPC checkout or authorization to execute official runs.

| Mechanism | Paper | Project proposal / contract |
| --- | --- | --- |
| Amplitude/phase representation | Stores individual complex semantic patterns | Complex codes represent expert choices and their route composition |
| Retrieval/alignment | Pairwise normalized interference score | Reconstruction alignment accompanies unbind/cleanup/rebind route factorization |
| Ordered route | Not the mechanism demonstrated | Station permutations bind expert position; unpermuted products lose order |
| Expert storage | Not a learned expert-bank compression study | Shared U/V bases, Q rotations, D spectra, and sparse R residuals; learned compression is a separate H1 claim |
| Speculative virtual experts | Not evaluated | Issue #106 proposes context-conditioned coefficient mixtures and separately measured speculative execution |
| Provenance / safety | Not our executable evidence contract | Lossless manifest, lineage, lock/cycle decisions, and independent admission gates remain necessary |

The paper's supplied phases and compact operator examples do not establish a learned semantic phase encoder or superiority under matched observations. Its 2025 exact-scan experiment reports approximately 135 ms average / 152 ms p95 for top-1 on 500K length-1024 patterns. These are reported measurements, not reproduced here. Our [IQ-05](innovation-test-queue-2026-09-12/05-wave-diffusion-multiplexing.md) concerns branch dynamics and is a more distant connection.

## Algebraic reduction derived in this review

For nonzero energies, set `u(z) = [Re(z), Im(z)]`, `E1 = ||u(z)||^2`, `E2 = ||u(w)||^2`, and `d = u(z) dot u(w) = Re(<z,w>)`. Expanding the paper's interference sum gives:

```text
S(z,w) = (E1 + E2 + 2*d) * sqrt(E1*E2) / (E1 + E2)^2
R = 2*sqrt(E1*E2)/(E1+E2)
S = R/2 * (1 + R*cosine(u(z),u(w)))
```

Thus its score is exactly a real dot product plus energy calibration on a 2L-dimensional encoding. When `E1=E2>0`, `S=(1+cosine(u(z),u(w)))/2`, for **arbitrary complex phases**, and retrieval rankings coincide with that cosine. This extends the paper's explicitly stated real/equal-norm case by the ordinary real-coordinate representation of a complex vector. At unequal energies, vanilla cosine alone can rank differently; the norm-calibrated control remains exact. If either energy is zero, S is zero.

Sign-phase initialization reconstructs the original signed real vector; it alone adds no semantic information. Phase enrichment can still be useful when an encoder supplies useful information, but its value must be tested against an ordinary real control retaining identical information. The inspected upstream `UnfoldedMath` and `UnfoldedMathPropertyTest` already express and test the unfolding identity; this reduction is not an allegation that upstream hides it.

## Testing integration and boundaries

[Seven constructed reduction checks](../../tests/test_resonance_score_reduction.py) cover 512 unequal-energy/random-phase pairs, eight 48-candidate normalized rankings, sign-phase reconstruction, a counterexample to uncalibrated cosine equivalence, zero/self/anti-phase cases, common phase rotation, and loss of information when a baseline erases phase. Command:

```powershell
python -m unittest tests.test_resonance_score_reduction -v
```

**Executed 2026-10-08:** seven tests, zero skips, zero failures, 0.023 seconds in the desktop preparation checkout. The independently derived algebraic fixtures use only the Python standard library.

These are mathematical METHOD_DEV fixtures. They do not reproduce the paper's semantic/scaling results, run RHPC official IDs, or implement the 24 separately specified MA cases. Future IQ-03 H0/H2 and issue #106 evaluation must include identical phase/role annotations, encoder training/access, precision, serialized bytes, and total work for candidate and control. Use the real norm-calibrated score as the retrieval control; ordinary same-code search/factorization remains necessary for RHPC's stronger claim. A metric change alone cannot demonstrate compression, route inference, or speculative speedup.

## Novelty and chronology

This paper is prior art for a broad phase-aware semantic-memory claim in our dated 2026 packets. It also cites earlier holographic/complex approaches; independently verified primary predecessors include [HolE (2015/2016)](https://arxiv.org/abs/1510.04935) and [ComplEx (2016)](https://proceedings.mlr.press/v48/trouillon16.html). Distinct ordered-route, compression, or virtual-expert claims still require their own full prior-art review and experiments.

The inspected Git roots begin on 2026-01-07; those dates bound only the available repository evidence. Earlier personal notes or conversations could change the chronology and must be authenticated before asserting priority. Similarity alone establishes neither copying nor independent origin. Keep the RHPC Stage-A protocol and its constructed/unconfirmed status unchanged.
