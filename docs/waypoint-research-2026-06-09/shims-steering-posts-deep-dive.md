# Shims / Steering Posts / Annealed Routes — Deep Dive

**Date:** 2026-06-09 · **Method:** direct web search (not the adversarial pipeline) plus
model knowledge, following up the five-concept assessment with the specific question:
*is the shim idea — blank shims, steering posts in multivariate subgroupings, annealed
routes more flexible than runtime matrix math on quantized subspaces — already out there
and working?*

**Short answer:** the building blocks are out there and working, and two papers from
Feb–May 2026 got very close to the steering-post part specifically. The *living* version —
routes through posts that anneal over time and get torn down under drift, applied to
retrieval pools — is not published anywhere found.

---

## Concept translation (operator language → field terminology)

| Operator's words | Field terminology |
|---|---|
| "Blank shims" | Identity-initialized insertable modules — start as a no-op, learn deviations later |
| "Steering posts" | A bank of anchor/steering vectors — fixed reference points nudged toward/away at runtime |
| "Multivariate subgroupings" | Query-conditioned subspaces — different dimension clusters matter per input type |
| "Annealed routes" | Learned paths/blends through the bank that start exploratory and gradually stabilize |
| "More flexible than matrix mult/division on quant calcs" | Nonlinear or flow-based transport instead of one fixed linear transform at runtime |

---

## What already exists and works

### Blank shims — solved, deployed at massive scale

- **LoRA** initializes one of its two matrices to zero → inserted module starts as exact
  identity. Every commercial fine-tuning service runs on this.
- **ControlNet "zero convolutions"** — same trick in diffusion models.
- **LLaMA-Adapter zero-gated attention, ReZero** — same family.
- Quant-riding version: **QLoRA, LoftQ, QA-LoRA** — small corrective adapters trained on
  top of *frozen quantized* weights, explicitly compensating quantization error. The
  repo's near-identity init (std 0.001) and the quant low-rank chelation scaffold (#255)
  are squarely in this established family.

### Steering posts — just published (Feb 2026)

- **Steering Vector Fields (SVF)** — [arXiv 2602.01654](https://arxiv.org/abs/2602.01654).
  Maintains a *bank* of steering anchors; at inference retrieves the K nearest anchors to
  the current activation and steers toward their centroid (normalized direction added to
  the residual stream with scalar strength). This is literally posts-in-subspace-groupings
  used as runtime steering targets.
- **CAST-style conditional steering** — only fires when the input projects onto a
  condition vector — i.e., a shim that activates only when its pattern shows up. See the
  [2026 activation-steering field guide](https://subhadipmitra.com/blog/2026/activation-steering-field-guide/).
- **SADI** — per-input binary masks selecting which neurons/heads to steer — dynamic,
  element-wise adaptation.

### Warping beyond matrix math — just published (May 2026)

- **FLAS (Flow-based Activation Steering)** — [arXiv 2605.05892](https://arxiv.org/abs/2605.05892).
  Learns a concept-conditioned *velocity field* transporting unsteered activations to
  steered ones — genuine nonlinear warping, not vector addition or a fixed matrix.
- **Compositional affine steering / conceptor logic** — naive addition of multiple
  steering vectors causes interference; structured composition schemes (distinct-layer
  injection, conceptor Boolean logic) are being invented. This independently confirms the
  congruent/parallel-subspace intuition: composition needs structure.

### Routing over a bank of small corrections — active, crowded

- **LD-MoLE** — [arXiv 2509.25684](https://arxiv.org/abs/2509.25684). Learnable dynamic
  routing for mixture-of-LoRA-experts; token-dependent, layer-wise expert allocation.
- **Queryable LoRA (May 2026)** — [arXiv 2605.08423](https://arxiv.org/html/2605.08423).
  Effective adapter built as a weighted sum over shared low-rank *atoms*, routed per
  instruction — "routes through a basis of small transforms," essentially.
- **SMoRA** — partitions a single LoRA into rank-level experts with sparse gating.
- **Poly-PRAG** — per-passage routing over latent LoRA experts in RAG settings.
- Overview: [Sparse Mixture of LoRA Experts](https://www.emergentmind.com/topics/sparse-mixture-of-lora-experts).

---

## What remains genuinely uncovered

1. **The lifecycle.** Every system above has a *static* bank — vectors/experts trained
   once, then frozen. Nobody manages the bank as a living structure: routes annealing
   from exploratory to stable, posts pruned ("disintegrated") when drift detectors fire,
   then re-annealed. The steering literature has no equivalent of a temperature controller.
2. **The domain.** Nearly all of this work steers *LLM hidden activations* (transformer
   residual stream). Applying post-and-route machinery to *retrieval embedding pools* —
   steering document/query geometry in a vector store — is a different application that
   steering papers don't touch and retrieval papers (Search-Adaptor, DIME) cover only
   with single linear/MLP corrections.
3. **Routes as trajectories, not one-shot gates.** MoLE-style routing makes one blending
   decision per token; SVF picks nearest neighbors once. A multi-hop *path* through
   several posts — where the path itself is the learned, annealed object — has no
   published equivalent found. FLAS's flow field is the nearest cousin (a flow is a
   continuous route) but is learned offline for fixed concepts, not annealed under drift.

---

## Honest caveats

- **The window is closing fast.** Six months ago "steering post banks" would have been
  novel outright; SVF and FLAS now exist. The defensible claim is the *living* bank with
  an anneal/prune lifecycle driven by drift signals, applied to retrieval pools.
- **"More flexible than matrix multiplication" needs precision.** Chaining linear
  transforms is still linear. Flexibility genuinely increases only via (a) nonlinearity
  between hops (FLAS-style velocity fields) or (b) conditional branching — which post you
  visit depends on where you are. The route concept has (b) built in; make that explicit
  in any writeup because it is the actual mechanism that beats a single matrix.

## The proving experiment

Inject drift, then show an annealed-route bank recovers retrieval quality where:

- a frozen SVF-style static bank degrades, and
- a one-shot MoLE-style router degrades.

SVF and LD-MoLE are the handed-to-us baselines. This slots directly into Phase II step 14
(drift-injection recovery) alongside the index-maintenance baselines (Ada-IVF/Quake) from
the five-concept assessment.

## Sources

- [Steering Vector Fields — arXiv 2602.01654](https://arxiv.org/abs/2602.01654)
- [FLAS: Flow-based Activation Steering — arXiv 2605.05892](https://arxiv.org/abs/2605.05892)
- [LD-MoLE — arXiv 2509.25684](https://arxiv.org/abs/2509.25684)
- [Queryable LoRA — arXiv 2605.08423](https://arxiv.org/html/2605.08423)
- [Activation Steering in 2026: A Practitioner's Field Guide](https://subhadipmitra.com/blog/2026/activation-steering-field-guide/)
- [Sparse Mixture of LoRA Experts (overview)](https://www.emergentmind.com/topics/sparse-mixture-of-lora-experts)
- [GrAInS: gradient-based attribution for inference-time steering — arXiv 2507.18043](https://arxiv.org/pdf/2507.18043)
